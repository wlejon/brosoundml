#include "rave_internal.h"

namespace brosoundml {

bt::Tensor Rave::Impl::run_ops(const std::vector<FlatOp>& ops, bt::Tensor h,
                               int Cin, int Lin, int& Cout, int& Lout) const {
    std::vector<bt::Tensor> stack;
    int C = Cin, L = Lin;
    for (const FlatOp& op : ops) {
        switch (op.kind) {
            case F_CONV: {
                int Lo;
                h = frun_conv(h, C, L, op.conv, op.stride, op.dil, op.left_pad, Lo);
                C = op.conv.cout; L = Lo;
                break;
            }
            case F_CONVT: {
                int Lo;
                h = frun_convt(h, C, L, op.conv, op.stride, op.pad, op.cpad, Lo);
                C = op.conv.cout; L = Lo;
                break;
            }
            case F_LEAKY:
                h = leaky(h, cfg.leaky_slope);
                break;
            case F_SNAKE: {
                bt::Tensor y;
                bt::snake_forward(h, op.alpha, /*beta=*/nullptr, /*N=*/1, C, L, y);
                h = std::move(y);
                break;
            }
            case F_PUSH:
                stack.push_back(h);
                break;
            case F_ADD:
                bt::add_inplace(h, stack.back());   // x + pushed residual input
                stack.pop_back();
                break;
        }
    }
    Cout = C; Lout = L;
    return h;
}

// Family A latent completion (deterministic): generate the discarded latent dims
// with the autoregressive prior, then map back to the decoder's input space.
bt::Tensor Rave::Impl::build_prior_latent(const float* latent, int n_latent, int T) const {
    const int full = cfg.full_latent_size;
    const int crop = n_latent;
    const int win  = prior_window;

    // z6: the assembled full latent in crop-space (channel-major, full x T). hist
    // holds the prior's input frames; hist[t] is the previous full latent frame
    // (z5[t-1], zeros at t=0). Output frame t depends only on the last `win` input
    // frames (causal, stride-1), so the prior runs over that trailing window.
    std::vector<float> z6(static_cast<std::size_t>(full) * T, 0.0f);
    std::vector<std::vector<float>> hist;
    hist.reserve(static_cast<std::size_t>(T) + 1);
    hist.emplace_back(full, 0.0f);
    std::vector<float> in;
    for (int t = 0; t < T; ++t) {
        const int len = static_cast<int>(hist.size());
        const int w0  = std::max(0, len - win);
        const int Lw  = len - w0;
        in.assign(static_cast<std::size_t>(full) * Lw, 0.0f);
        for (int j = 0; j < Lw; ++j) {
            const std::vector<float>& fr = hist[static_cast<std::size_t>(w0) + j];
            for (int c = 0; c < full; ++c)
                in[static_cast<std::size_t>(c) * Lw + j] = fr[c];
        }
        bt::Tensor inT = bt::Tensor::from_host_on(dev, in.data(), 1, full * Lw);
        int Co, Lo;
        bt::Tensor pr = run_ops(prior_ops, inT, full, Lw, Co, Lo);  // (1, 2*full*Lo)
        const std::vector<float> prh = pr.to_host_vector();         // prh[c*Lo + s]
        // Kept dims from the encoded crop; the rest from the prior MEAN (first
        // `full` of the 2*full output) at the last time step, gathered at
        // kld_idxs[crop:]. Deterministic — the prior mean, no reparameterisation.
        std::vector<float> frame(full);
        for (int c = 0; c < crop; ++c)
            frame[c] = latent[static_cast<std::size_t>(c) * T + t];
        for (int c = crop; c < full; ++c)
            frame[c] = prh[static_cast<std::size_t>(kld_idxs[c]) * Lo + (Lo - 1)];
        for (int c = 0; c < full; ++c)
            z6[static_cast<std::size_t>(c) * T + t] = frame[c];
        hist.push_back(std::move(frame));
    }
    bt::Tensor z6T = bt::Tensor::from_host_on(dev, z6.data(), 1, full * T);

    // Back to encoder space: inverse-PCA (use_pca) or scatter argsort(kld_idxs).
    bt::Tensor z7;
    if (use_pca) {
        int Lz;
        z7 = cconv(z6T, full, T, dec_pca, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Lz);
    } else {
        z7 = bt::Tensor::zeros_on(dev, 1, full * T, bt::Dtype::FP32);
        for (int j = 0; j < full; ++j)
            bt::copy_d2d(z6T, static_cast<std::size_t>(kld_perm[j]) * T,
                         z7, static_cast<std::size_t>(j) * T, T);
    }

    // Invert the gimbal affine: z10 = (z7 - b) / a, via batch_norm (running_mean=b,
    // running_var=1, eps=0, weight=1/a, beta=0).
    bt::Tensor z10;
    bt::batch_norm_inference(z7, gim_inv_a, gim_zero, gim_b, gim_one,
                             /*N=*/1, full, /*H=*/1, /*W=*/T, /*eps=*/0.0f, z10);
    return z10;
}

RaveLatent Rave::Impl::encode(const float* audio, int n) const {
    if (!is_loaded) fail("model not loaded");
    bt::DeviceScope scope(dev);

    // Right-pad to a whole frame so every stride divides evenly and T is exact.
    const int ratio = cfg.total_ratio;
    const int L = ((n + ratio - 1) / ratio) * ratio;
    std::vector<float> buf(static_cast<std::size_t>(L), 0.0f);
    for (int i = 0; i < n; ++i) buf[i] = audio[i];
    bt::Tensor x = bt::Tensor::from_host_on(dev, buf.data(), 1, L);

    // PQMF analysis.
    int Lb;
    bt::Tensor h = cconv(x, /*Cin=*/1, L, pqmf_fwd, /*stride=*/cfg.n_band, 1, 1, Lb);
    reverse_half_inplace(h, cfg.n_band, Lb);

    int C = cfg.n_band, Lc = Lb;
    if (flat) {
        h = run_ops(enc_ops, h, C, Lc, C, Lc);   // residual encoder / Snake topology
    } else {
        for (const EncLayer& el : enc) {
            int Lo;
            h = cconv(h, C, Lc, el.conv, el.stride, /*dilation=*/1, el.groups, Lo);
            C = el.conv.cout; Lc = Lo;
            if (el.has_bn) {
                bt::Tensor y;
                bt::batch_norm_inference(h, el.bn.g, el.bn.beta, el.bn.mean, el.bn.var,
                                         /*N=*/1, C, /*H=*/1, /*W=*/Lc, cfg.bn_eps, y);
                h = std::move(y);
            }
            if (el.act) h = leaky(h, cfg.leaky_slope);
        }
    }

    // Split (mean | scale): take the first `full` channels (the posterior mean).
    const int full = cfg.full_latent_size;
    const int T    = Lc;
    bt::Tensor mean = bt::Tensor::zeros_on(dev, 1, full * T, bt::Dtype::FP32);
    bt::copy_d2d(h, 0, mean, 0, full * T);

    // Gimbal latent affine (flat models that ship it): mean = mean*exp(log_a)+b,
    // folded into a batch_norm so it's one device op. Applied before the PCA.
    if (flat && has_gimbal) {
        bt::Tensor y;
        bt::batch_norm_inference(mean, gim_a, gim_b, gim_zero, gim_one,
                                 /*N=*/1, full, /*H=*/1, /*W=*/T, /*eps=*/0.0f, y);
        mean = std::move(y);
    }

    // Crop to the kept latent dims. Two schemes (see latent_mode): kld-gather
    // (prior_kld — keep encoder-space channels kld_idxs[:nl], no PCA) or PCA
    // projection then slice (prior_pca / noise_pca / legacy).
    const int nl = cfg.cropped_latent_size;
    bt::Tensor zc = bt::Tensor::zeros_on(dev, 1, nl * T, bt::Dtype::FP32);
    if (flat && latent_mode == "prior_kld") {
        for (int c = 0; c < nl; ++c)
            bt::copy_d2d(mean, static_cast<std::size_t>(kld_idxs[c]) * T,
                         zc, static_cast<std::size_t>(c) * T, T);
    } else {
        // z = pca @ (mean - latent_mean)  (folded into the 1x1 conv bias).
        int Lz;
        bt::Tensor z = cconv(mean, full, T, enc_pca, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Lz);
        bt::copy_d2d(z, 0, zc, 0, nl * T);
    }

    RaveLatent out;
    out.n_latent = nl;
    out.frames   = T;
    out.data     = zc.to_host_vector();   // (1, nl*T) -> channel-major data[c*T + t]
    return out;
}

std::vector<float> Rave::Impl::decode_core(const float* latent, int n_latent, int frames,
                                           const RaveDecodeOptions& opts, int channel) const {
    if (!is_loaded) fail("model not loaded");
    if (n_latent != cfg.cropped_latent_size)
        fail("decode: latent has " + std::to_string(n_latent) + " dims, expected " +
             std::to_string(cfg.cropped_latent_size));
    bt::DeviceScope scope(dev);

    const int full = cfg.full_latent_size;
    const int T    = frames;

    // ── FLAT decode: build the decoder-network input from the cropped latent.
    //    Two latent-completion schemes (latent_mode):
    //      prior_*   — an autoregressive prior fills the discarded dims frame by
    //                  frame, then the full latent is scattered (kld) / inverse-PCA'd
    //                  (pca) back to encoder space and the gimbal affine is inverted.
    //      noise_pca — fill the discarded dims with N(0,1) (or an injected pad),
    //                  then inverse-PCA (the .ts pre_process_latent). No prior, no
    //                  gimbal. The N(0,1) fill is the model's behaviour, NOT a
    //                  simplification — zero-padding here is what made decode silent.
    //    The result feeds the decoder op-list, then the flat synth + PQMF. ──
    if (flat) {
        const int nb = cfg.n_band;
        bt::Tensor z10;
        if (latent_mode.rfind("prior", 0) == 0) {
            z10 = build_prior_latent(latent, n_latent, T);
        } else {
            const int padDims = full - n_latent;
            bt::Tensor zf = bt::Tensor::zeros_on(dev, 1, full * T, bt::Dtype::FP32);
            bt::Tensor zsrc = bt::Tensor::from_host_on(dev, latent, 1, n_latent * T);
            bt::copy_d2d(zsrc, 0, zf, 0, n_latent * T);
            if (padDims > 0) {
                bt::Tensor padT;
                if (opts.latent_pad) {     // injected pad (parity / explicit per-channel)
                    const std::size_t blk = static_cast<std::size_t>(padDims) * T;
                    if (opts.latent_pad_len < static_cast<int>((static_cast<std::size_t>(channel) + 1) * blk))
                        fail("decode: injected latent_pad too small for channel " + std::to_string(channel));
                    padT = bt::Tensor::from_host_on(dev, opts.latent_pad + static_cast<std::size_t>(channel) * blk,
                                                    padDims, T);
                } else {                   // N(0,1) draw (per-channel counter for stereo)
                    padT = bt::Tensor::zeros_on(dev, padDims, T, bt::Dtype::FP32);
                    bt::randn(opts.seed ^ 0x52415645ull, static_cast<std::uint64_t>(channel), padT);
                    const float pad_std = opts.latent_pad_std > 0.0f ? opts.latent_pad_std : 1.0f;
                    bt::scale_inplace(padT, pad_std);
                }
                bt::copy_d2d(padT, 0, zf, static_cast<std::size_t>(n_latent) * T,
                             static_cast<std::size_t>(padDims) * T);
            }
            int Lz;
            z10 = cconv(zf, full, T, dec_pca, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Lz);
        }
        int C, Lc;
        bt::Tensor hh = run_ops(dec_ops, z10, full, T, C, Lc);

        bt::Tensor tw;   // multiband waveform (1, nb*Lc)
        if (synth_type == "amp_mod") {
            // decoder ends at 2*n_band: split (wave | amp); out = tanh(wave*sigmoid(amp)).
            const int half = C / 2;
            if (half != nb) fail("amp_mod synth: decoder out (" + std::to_string(C) +
                                 ") != 2*n_band");
            bt::Tensor wave = bt::Tensor::zeros_on(dev, 1, half * Lc, bt::Dtype::FP32);
            bt::Tensor amp  = bt::Tensor::zeros_on(dev, 1, half * Lc, bt::Dtype::FP32);
            bt::copy_d2d(hh, 0, wave, 0, half * Lc);
            bt::copy_d2d(hh, static_cast<std::size_t>(half) * Lc, amp, 0, half * Lc);
            bt::Tensor sa;
            bt::sigmoid_forward(amp, sa);
            bt::mul_inplace(wave, sa);
            bt::tanh_forward(wave, tw);
        } else {
            // "rave": tanh(wave) * mod_sigmoid(loud), loud broadcast across bands.
            if (loud_stride > 1) fail("flat decode: loud_stride>1 not supported");
            int Lw;
            bt::Tensor wave = cconv(hh, C, Lc, synth_wave, 1, 1, 1, Lw);
            int Ll;
            bt::Tensor loud = cconv(hh, C, Lc, synth_loud, 1, 1, 1, Ll);
            bt::tanh_forward(wave, tw);              // (1, nb*Lc), Lw == Lc
            bt::Tensor m = mod_sigmoid(loud);        // (1, 1*Lc)
            bt::Tensor mt = bt::Tensor::zeros_on(dev, 1, nb * Lc, bt::Dtype::FP32);
            for (int c = 0; c < nb; ++c) bt::copy_d2d(m, 0, mt, c * Lc, Lc);
            bt::mul_inplace(tw, mt);
        }

        // PQMF synthesis (same as the legacy tail): reverse_half, conv, *n_band,
        // polyphase interleave.
        reverse_half_inplace(tw, nb, Lc);
        int Ly;
        bt::Tensor y = cconv(tw, nb, Lc, pqmf_inv, 1, 1, 1, Ly);
        std::vector<float> Y = y.to_host_vector();   // (nb, Lc) channel-major
        std::vector<float> wav(static_cast<std::size_t>(nb) * Lc);
        for (int t = 0; t < Lc; ++t)
            for (int c = 0; c < nb; ++c)
                wav[static_cast<std::size_t>(t) * nb + c] =
                    static_cast<float>(nb) * Y[static_cast<std::size_t>(nb - 1 - c) * Lc + t];
        return wav;
    }

    // Legacy unprojection: pad nl -> full latents, then x = pca^T @ zf + latent_mean.
    // The discarded dims [n_latent, full) carry the per-channel pad — zero by default
    // (deterministic mono), or an independent N(0,1) draw for stereo width (the only
    // thing that differs between stereo channels).
    bt::Tensor zf = bt::Tensor::zeros_on(dev, 1, full * T, bt::Dtype::FP32);
    {
        bt::Tensor zsrc = bt::Tensor::from_host_on(dev, latent, 1, n_latent * T);
        bt::copy_d2d(zsrc, 0, zf, 0, n_latent * T);
        const int padDims = full - n_latent;
        if (padDims > 0 && (opts.latent_pad || opts.latent_pad_std > 0.0f)) {
            bt::Tensor padT;
            if (opts.latent_pad) {
                const std::size_t blk = static_cast<std::size_t>(padDims) * T;
                if (opts.latent_pad_len < static_cast<int>((static_cast<std::size_t>(channel) + 1) * blk))
                    fail("decode: injected latent_pad too small for channel " + std::to_string(channel));
                padT = bt::Tensor::from_host_on(dev, opts.latent_pad + static_cast<std::size_t>(channel) * blk,
                                                padDims, T);
            } else {
                padT = bt::Tensor::zeros_on(dev, padDims, T, bt::Dtype::FP32);
                bt::randn(opts.seed ^ 0x52415645ull, static_cast<std::uint64_t>(channel), padT);
                bt::scale_inplace(padT, opts.latent_pad_std);
            }
            bt::copy_d2d(padT, 0, zf, static_cast<std::size_t>(n_latent) * T,
                         static_cast<std::size_t>(padDims) * T);
        }
    }

    // Legacy unprojection: x = pca^T @ zf + latent_mean (inverse PCA before decode).
    int Lz;
    bt::Tensor h = cconv(zf, full, T, dec_pca, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Lz);

    // Decoder input conv.
    int C = full, Lc = T, Lo;
    h = cconv(h, C, Lc, dec_in, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Lo);
    C = dec_in.cout; Lc = Lo;

    // 4x [LeakyReLU, convT upsample, 3 residual blocks].
    for (const UpStack& st : dec_up) {
        bt::Tensor a = leaky(h, cfg.leaky_slope);
        int Lu;
        h = cconvt(a, C, Lc, st.convt, st.stride, Lu);
        C = st.convt.cout; Lc = Lu;
        for (const ResBlock& rb : st.res) {
            bt::Tensor r = leaky(h, cfg.leaky_slope);
            int L1;
            r = cconv(r, C, Lc, rb.c1, /*stride=*/1, rb.dil, /*groups=*/1, L1);
            r = leaky(r, cfg.leaky_slope);
            int L2;
            r = cconv(r, rb.c1.cout, Lc, rb.c2, /*stride=*/1, /*dilation=*/1, /*groups=*/1, L2);
            bt::add_inplace(h, r);   // x + branch; c2.cout == C
        }
    }

    // Synthesis: tanh(wave) * mod_sigmoid(loud), loud broadcast across bands.
    int Lw;
    bt::Tensor wave = cconv(h, C, Lc, synth_wave, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Lw);
    int Ll;
    bt::Tensor loud = cconv(h, C, Lc, synth_loud, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Ll);

    bt::Tensor tw;
    bt::tanh_forward(wave, tw);                 // (1, n_band*Lc)
    bt::Tensor m = mod_sigmoid(loud);           // (1, 1*Lc)
    bt::Tensor mt = bt::Tensor::zeros_on(dev, 1, cfg.n_band * Lc, bt::Dtype::FP32);
    for (int c = 0; c < cfg.n_band; ++c) bt::copy_d2d(m, 0, mt, c * Lc, Lc);
    bt::mul_inplace(tw, mt);                     // multiband waveform

    const int nb = cfg.n_band;

    // Optional stochastic noise-synth branch, added to the multiband waveform
    // before PQMF synthesis. Mirrors rave_reference.py::_noise_branch.
    if (opts.add_noise && has_noise) {
        // 3 stride-4 causal convs (LeakyReLU between, none after the last).
        bt::Tensor hn = h;
        int Cn = C, Ln = Lc;
        for (int i = 0; i < 3; ++i) {
            int Lo;
            hn = cconv(hn, Cn, Ln, synth_noise[i], /*stride=*/4, /*dilation=*/1,
                       /*groups=*/1, Lo);
            Cn = synth_noise[i].cout; Ln = Lo;
            if (i < 2) hn = leaky(hn, cfg.leaky_slope);
        }
        const int T_n = Ln;                  // noise frames; T_n * 64 == Lc
        const int R   = T_n * nb;            // one FFT row per (frame, band)
        if (T_n * 64 != Lc) fail("noise branch: frame/length mismatch");

        // amp = mod_sigmoid(hn - 5), transposed+reshaped to (R, noise_bands) in
        // (frame, band) row order. hn is (C_amp, T_n), C_amp = nb*noise_bands.
        const std::vector<float> hnh = hn.to_host_vector();   // [c*T_n + t]
        std::vector<float> amp(static_cast<std::size_t>(R) * noise_bands);
        for (int t = 0; t < T_n; ++t)
            for (int band = 0; band < nb; ++band)
                for (int j = 0; j < noise_bands; ++j)
                    amp[(static_cast<std::size_t>(t) * nb + band) * noise_bands + j] =
                        mod_sigmoid_scalar(
                            hnh[(static_cast<std::size_t>(band) * noise_bands + j) * T_n + t]
                            - 5.0f);
        bt::Tensor amp_d = bt::Tensor::from_host_on(dev, amp.data(), R, noise_bands);

        // ir = M @ amp  (amp_to_impulse_response folded to a matmul), -> (R, 64).
        bt::Tensor ir;
        bt::linear_forward_batched(noise_ir_mat, noise_ir_bias, amp_d, ir);

        // White noise in U(-1, 1): injected (reproducible / parity) or sampled.
        bt::Tensor nz = bt::Tensor::zeros_on(dev, R, 64, bt::Dtype::FP32);
        if (opts.noise) {
            if (opts.noise_len < R * 64) fail("decode: injected noise buffer too small");
            nz = bt::Tensor::from_host_on(dev, opts.noise, R, 64);
        } else {
            // Per-channel white noise (counter = channel) so stereo channels get
            // independent noise textures; channel 0 keeps counter 0.
            bt::rand_uniform(opts.seed, static_cast<std::uint64_t>(channel), nz);
            bt::scale_inplace(nz, 2.0f);
            bt::add_scalar_inplace(nz, -1.0f);
        }

        // fft_convolve(noise, ir): right-pad noise and left-pad ir to 128, then
        // irfft(rfft(sig) * rfft(ker)) and take the second half (64 samples).
        bt::Tensor sig = bt::Tensor::zeros_on(dev, R, 128, bt::Dtype::FP32);
        bt::Tensor ker = bt::Tensor::zeros_on(dev, R, 128, bt::Dtype::FP32);
        for (int r = 0; r < R; ++r) {
            bt::copy_d2d(nz, r * 64, sig, r * 128, 64);          // [0:64]
            bt::copy_d2d(ir, r * 64, ker, r * 128 + 64, 64);     // [64:128]
        }
        bt::Tensor Sf, Kf, Pf, conv;
        bt::rfft(sig, Sf);
        bt::rfft(ker, Kf);
        bt::complex_mul(Sf, Kf, Pf);
        bt::irfft(Pf, 128, conv);                                // (R, 128)
        // Take the second half (samples [64:128]) of each row.
        bt::Tensor tail = bt::Tensor::zeros_on(dev, R, 64, bt::Dtype::FP32);
        for (int r = 0; r < R; ++r) bt::copy_d2d(conv, r * 128 + 64, tail, r * 64, 64);

        // Permute (frame, band) -> (band, frame) and reshape to (nb, T_n*64=Lc),
        // then add to the multiband waveform.
        const std::vector<float> th = tail.to_host_vector();     // [r*64 + s]
        std::vector<float> nbuf(static_cast<std::size_t>(nb) * Lc);
        for (int band = 0; band < nb; ++band)
            for (int t = 0; t < T_n; ++t)
                for (int s = 0; s < 64; ++s)
                    nbuf[(static_cast<std::size_t>(band) * T_n + t) * 64 + s] =
                        th[(static_cast<std::size_t>(t) * nb + band) * 64 + s];
        bt::Tensor nb_d = bt::Tensor::from_host_on(dev, nbuf.data(), 1, nb * Lc);
        bt::add_inplace(tw, nb_d);
    }

    // PQMF synthesis: reverse_half, conv, *n_band, then polyphase interleave.
    reverse_half_inplace(tw, nb, Lc);
    int Ly;
    bt::Tensor y = cconv(tw, nb, Lc, pqmf_inv, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Ly);
    std::vector<float> Y = y.to_host_vector();   // (nb, Lc) channel-major, Ly == Lc

    std::vector<float> wav(static_cast<std::size_t>(nb) * Lc);
    for (int t = 0; t < Lc; ++t)
        for (int c = 0; c < nb; ++c)
            wav[static_cast<std::size_t>(t) * nb + c] =
                static_cast<float>(nb) * Y[static_cast<std::size_t>(nb - 1 - c) * Lc + t];

    return wav;
}

AudioBuffer Rave::Impl::decode(const float* latent, int n_latent, int frames,
                               const RaveDecodeOptions& opts) const {
    // Mono: a single channel with whatever pad opts specify (zero by default).
    return AudioBuffer(decode_core(latent, n_latent, frames, opts, /*channel=*/0),
                       cfg.sampling_rate);
}

RaveMultiBuffer Rave::Impl::decode_multi(const float* latent, int n_latent, int frames,
                                         const RaveDecodeOptions& opts) const {
    const int ch = opts.channels < 1 ? 1 : opts.channels;
    RaveMultiBuffer out;
    out.sample_rate = cfg.sampling_rate;
    out.channels    = ch;
    if (ch == 1)
        return { decode_core(latent, n_latent, frames, opts, 0), cfg.sampling_rate, 1 };

    // Decode each channel independently, then interleave: samples[t*ch + c].
    std::vector<std::vector<float>> chans;
    chans.reserve(ch);
    std::size_t per = 0;
    for (int c = 0; c < ch; ++c) {
        chans.push_back(decode_core(latent, n_latent, frames, opts, c));
        per = chans[c].size();
    }
    out.samples.resize(per * static_cast<std::size_t>(ch));
    for (int c = 0; c < ch; ++c)
        for (std::size_t t = 0; t < per; ++t)
            out.samples[t * ch + c] = chans[c][t];
    return out;
}

// ── public wrapper ───────────────────────────────────────────────────────────

Rave::Rave() : impl_(std::make_unique<Impl>()) {}
Rave::~Rave() = default;
Rave::Rave(Rave&&) noexcept = default;
Rave& Rave::operator=(Rave&&) noexcept = default;

void Rave::load(const std::string& model_dir, bt::Device device) {
    impl_->load(model_dir, device);
}

RaveLatent Rave::encode(const std::vector<float>& audio) const {
    return impl_->encode(audio.data(), static_cast<int>(audio.size()));
}
RaveLatent Rave::encode(const float* audio, int n) const {
    return impl_->encode(audio, n);
}

AudioBuffer Rave::decode(const RaveLatent& latent, const RaveDecodeOptions& opts) const {
    return impl_->decode(latent.data.data(), latent.n_latent, latent.frames, opts);
}
AudioBuffer Rave::decode(const float* latent, int n_latent, int frames,
                         const RaveDecodeOptions& opts) const {
    return impl_->decode(latent, n_latent, frames, opts);
}

RaveMultiBuffer Rave::decode_multi(const RaveLatent& latent, const RaveDecodeOptions& opts) const {
    return impl_->decode_multi(latent.data.data(), latent.n_latent, latent.frames, opts);
}
RaveMultiBuffer Rave::decode_multi(const float* latent, int n_latent, int frames,
                                   const RaveDecodeOptions& opts) const {
    return impl_->decode_multi(latent, n_latent, frames, opts);
}

const RaveConfig& Rave::config() const { return impl_->cfg; }
bool Rave::loaded() const { return impl_->is_loaded; }

}  // namespace brosoundml
