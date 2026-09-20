#include "kokoro_internal.h"

namespace brosoundml {

// ─── DecoderBackbone ───────────────────────────────────────────────────────

void DecoderBackbone::load_from(const stf::File& f) {
    const std::string where = "DecoderBackbone::load_from";
    const std::string p = "decoder.module.";

    // F0_conv / N_conv: Conv1d(1, 1, k=3, s=2, p=1) — downsample 2x.
    auto load_strided_conv = [&](const std::string& name, Conv1d& c) {
        c.in_channels  = 1;
        c.out_channels = 1;
        c.kernel_size  = 3;
        c.stride       = 2;
        c.padding      = 1;
        c.dilation     = 1;
        c.groups       = 1;
        upload(f, p + name + ".weight", 1, 1 * 3, c.W, where);
        upload(f, p + name + ".bias",   1, 1,     c.b, where);
    };
    load_strided_conv("F0_conv", F0_conv);
    load_strided_conv("N_conv",  N_conv);

    // asr_res: Sequential(Conv1d(512, 64, k=1)) — wrapped in a Sequential so
    // the state-dict key has a `.0.` infix.
    asr_res.in_channels  = 512;
    asr_res.out_channels = 64;
    asr_res.kernel_size  = 1;
    asr_res.padding      = 0;
    asr_res.stride       = 1;
    asr_res.dilation     = 1;
    asr_res.groups       = 1;
    upload(f, p + "asr_res.0.weight", 64, 512, asr_res.W, where);
    upload(f, p + "asr_res.0.bias",   64, 1,   asr_res.b, where);

    // encode: AdainResBlk1d(514, 1024, style=128, no upsample, learned_sc=True).
    load_adain_resblk(f, p + "encode", /*dim_in=*/514, /*dim_out=*/1024,
                      /*style_dim=*/128, /*upsample=*/false, encode, where);

    // decode[0..3]: 1090 -> 1024 (no upsample) x3, then 1090 -> 512 (upsample).
    decode.clear();
    decode.resize(4);
    load_adain_resblk(f, p + "decode.0", 1090, 1024, 128, false, decode[0], where);
    load_adain_resblk(f, p + "decode.1", 1090, 1024, 128, false, decode[1], where);
    load_adain_resblk(f, p + "decode.2", 1090, 1024, 128, false, decode[2], where);
    load_adain_resblk(f, p + "decode.3", 1090, 512,  128, true,  decode[3], where);
}

namespace {

// Concat NCL tensors along the channel axis. Each part is (1, C_i * L) with
// the same L; out is (1, (sum C_i) * L) with channel blocks laid end-to-end.
// Implemented as one copy_d2d per part (each part is contiguous in its NCL
// flat layout, so it copies as one slab into the right channel offset of out).
void cat_channels_ncl(const std::vector<const bt::Tensor*>& parts,
                      int L, bt::Tensor& out) {
    int C_total = 0;
    for (const auto* p : parts) C_total += p->cols / L;
    const bt::Device dev = parts.empty() ? bt::Device::CPU : parts[0]->device;
    out = bt::Tensor::zeros_on(dev, 1, C_total * L, bt::Dtype::FP32);
    int c_off = 0;
    for (const auto* p : parts) {
        const int C = p->cols / L;
        bt::copy_d2d(*p, 0, out, c_off * L, C * L);
        c_off += C;
    }
}

}  // namespace

void DecoderBackbone::forward(const bt::Tensor& asr,
                              const bt::Tensor& F0_pred,
                              const bt::Tensor& N_pred,
                              const bt::Tensor& ref_s,
                              int T,
                              bt::Tensor& gen_in) const {
    const int style_dim = 128;

    const bt::Device dev = asr.device;
    bt::Tensor style = bt::Tensor::zeros_on(ref_s.device, 1, style_dim, bt::Dtype::FP32);
    bt::copy_d2d(ref_s, 0, style, 0, style_dim);

    // F0_pred / N_pred: (1, 2*T) -> unsqueeze to (1, 1*(2T)) NCL -> stride-2
    // conv -> (1, 1*T). Pre-allocate every op-out on dev (Tensor::resize
    // preserves device; default-constructed Tensors are CPU and would crash
    // brotensor's CUDA dispatch).
    bt::Tensor F0_dn = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor N_dn  = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    F0_conv.forward(F0_pred, /*N=*/1, /*L=*/2 * T, F0_dn);
    N_conv .forward(N_pred,  /*N=*/1, /*L=*/2 * T, N_dn);

    // Concat [asr, F0_dn, N_dn] along channel axis: (1, 514*T).
    bt::Tensor dec_pre = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    cat_channels_ncl({&asr, &F0_dn, &N_dn}, T, dec_pre);

    // encode: AdainResBlk1d(514 -> 1024).
    bt::Tensor enc_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    int L_after = 0;
    adain_resblk_1d_forward(encode, dec_pre, T, style, L_after, enc_out);

    // asr_res = Conv1d(512 -> 64, k=1) over asr.
    bt::Tensor asr_res_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    asr_res.forward(asr, /*N=*/1, /*L=*/T, asr_res_out);

    // decode loop.
    bt::Tensor x = std::move(enc_out);
    int L_now = L_after;
    bool res = true;
    for (size_t i = 0; i < decode.size(); ++i) {
        if (res) {
            // Concat x (1024 channels) with asr_res (64) + F0_dn (1) + N_dn (1)
            // = 1090 channels at the same L_now=T.
            bt::Tensor catted = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
            cat_channels_ncl({&x, &asr_res_out, &F0_dn, &N_dn}, L_now, catted);
            x = std::move(catted);
        }
        bt::Tensor y = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        int L_next = 0;
        adain_resblk_1d_forward(decode[i], x, L_now, style, L_next, y);
        x = std::move(y);
        L_now = L_next;
        if (decode[i].upsample) res = false;
    }
    gen_in = std::move(x);
}

// ─── AdaINResBlock1 (Generator residual block) ─────────────────────────────

namespace {

void load_adain_resblock1(const stf::File& f, const std::string& prefix,
                          int channels, int kernel,
                          const std::vector<int>& dilations,
                          AdaINResBlock1Weights& w, const std::string& where) {
    w.channels    = channels;
    w.kernel_size = kernel;
    w.dilations   = dilations;
    for (int i = 0; i < 3; ++i) {
        // convs1[i]: dilation = dilations[i], padding = get_padding(k, dil) = ((k*dil - dil) / 2)
        const int dil1 = dilations[i];
        const int pad1 = (kernel * dil1 - dil1) / 2;
        w.convs1[i].in_channels  = channels;
        w.convs1[i].out_channels = channels;
        w.convs1[i].kernel_size  = kernel;
        w.convs1[i].padding      = pad1;
        w.convs1[i].dilation     = dil1;
        w.convs1[i].stride       = 1;
        w.convs1[i].groups       = 1;
        upload(f, prefix + ".convs1." + std::to_string(i) + ".weight",
               channels, channels * kernel, w.convs1[i].W, where);
        upload(f, prefix + ".convs1." + std::to_string(i) + ".bias",
               channels, 1, w.convs1[i].b, where);

        // convs2[i]: dilation = 1.
        const int pad2 = (kernel - 1) / 2;
        w.convs2[i].in_channels  = channels;
        w.convs2[i].out_channels = channels;
        w.convs2[i].kernel_size  = kernel;
        w.convs2[i].padding      = pad2;
        w.convs2[i].dilation     = 1;
        w.convs2[i].stride       = 1;
        w.convs2[i].groups       = 1;
        upload(f, prefix + ".convs2." + std::to_string(i) + ".weight",
               channels, channels * kernel, w.convs2[i].W, where);
        upload(f, prefix + ".convs2." + std::to_string(i) + ".bias",
               channels, 1, w.convs2[i].b, where);

        load_ada_in_1d(f, prefix + ".adain1." + std::to_string(i),
                       channels, /*style_dim=*/128, w.adain1[i], where);
        load_ada_in_1d(f, prefix + ".adain2." + std::to_string(i),
                       channels, /*style_dim=*/128, w.adain2[i], where);

        // alpha shape on disk is (1, channels, 1); we want (channels, 1).
        upload(f, prefix + ".alpha1." + std::to_string(i),
               channels, 1, w.alpha1[i], where);
        upload(f, prefix + ".alpha2." + std::to_string(i),
               channels, 1, w.alpha2[i], where);
    }
}

// Streamlined AdaIN1D + Snake Activation: evaluates AdaIN1D directly into out_ncl
// followed by in-place Snake1D activation, avoiding intermediate tensor reallocations.
inline void ada_in_1d_snake(const AdaIN1dWeights& w, const bt::Tensor& alpha,
                            int N, int C, int L,
                            const bt::Tensor& x_ncl, const bt::Tensor& style,
                            bt::Tensor& out_ncl) {
    ada_in_1d_styled(w, N, C, L, x_ncl, style, out_ncl);
    bt::snake_forward(out_ncl, alpha, /*beta=*/nullptr, N, C, L, out_ncl);
}

void adain_resblock1_forward(const AdaINResBlock1Weights& w,
                             bt::Tensor& x, int C, int L,
                             const bt::Tensor& style) {
    const bt::Device dev = x.device;
    bt::Tensor xt = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor c1_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor c2_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);

    for (int i = 0; i < 3; ++i) {
        // Streamlined n1(x, s) + Snake1D in-place
        ada_in_1d_snake(w.adain1[i], w.alpha1[i], 1, C, L, x, style, xt);

        // c1 (dilated conv)
        w.convs1[i].forward(xt, /*N=*/1, /*L=*/L, c1_out);

        // Streamlined n2(c1_out, s) + Snake1D in-place
        ada_in_1d_snake(w.adain2[i], w.alpha2[i], 1, C, L, c1_out, style, xt);

        // c2
        w.convs2[i].forward(xt, /*N=*/1, /*L=*/L, c2_out);

        // x = xt + x  (residual)
        bt::add_inplace(c2_out, x);
        x = std::move(c2_out);
    }
}

}  // namespace

// ─── Generator ─────────────────────────────────────────────────────────────

void Generator::load_from(const stf::File& f, const KokoroConfig& cfg) {
    const std::string where = "Generator::load_from";
    const std::string p = "decoder.module.generator.";

    n_fft         = cfg.decoder.gen_istft_n_fft;
    hop_size      = cfg.decoder.gen_istft_hop_size;
    win_size      = n_fft;
    num_upsamples = static_cast<int>(cfg.decoder.upsample_rates.size());
    num_kernels   = static_cast<int>(cfg.decoder.resblock_kernel_sizes.size());

    const int init_C = cfg.decoder.upsample_initial_channel;

    // ─── ups (ConvTranspose1d) ────────────────────────────────────────────
    ups_W.resize(num_upsamples);
    ups_b.resize(num_upsamples);
    ups_C_in.resize(num_upsamples);
    ups_C_out.resize(num_upsamples);
    ups_k.resize(num_upsamples);
    ups_stride.resize(num_upsamples);
    ups_pad.resize(num_upsamples);
    for (int i = 0; i < num_upsamples; ++i) {
        const int C_in  = init_C / (1 << i);
        const int C_out = init_C / (1 << (i + 1));
        const int kL    = cfg.decoder.upsample_kernel_sizes[i];
        const int s     = cfg.decoder.upsample_rates[i];
        ups_C_in[i]  = C_in;
        ups_C_out[i] = C_out;
        ups_k[i]     = kL;
        ups_stride[i]= s;
        ups_pad[i]   = (kL - s) / 2;
        // Weight on disk: (C_in, C_out, kL). brotensor wants (C_in, C_out*kL).
        upload(f, p + "ups." + std::to_string(i) + ".weight",
               C_in, C_out * kL, ups_W[i], where);
        upload(f, p + "ups." + std::to_string(i) + ".bias",
               C_out, 1, ups_b[i], where);
    }

    // ─── noise_convs ──────────────────────────────────────────────────────
    noise_convs.assign(num_upsamples, Conv1d{});
    {
        const int upsample_prod = [&] {
            int u = 1; for (int r : cfg.decoder.upsample_rates) u *= r; return u;
        }();
        (void)upsample_prod;
        for (int i = 0; i < num_upsamples; ++i) {
            const int C_cur = init_C / (1 << (i + 1));
            Conv1d& c = noise_convs[i];
            c.in_channels  = n_fft + 2;
            c.out_channels = C_cur;
            if (i + 1 < num_upsamples) {
                int stride_f0 = 1;
                for (int j = i + 1; j < num_upsamples; ++j) {
                    stride_f0 *= cfg.decoder.upsample_rates[j];
                }
                c.kernel_size = stride_f0 * 2;
                c.stride      = stride_f0;
                c.padding     = (stride_f0 + 1) / 2;
            } else {
                c.kernel_size = 1;
                c.stride      = 1;
                c.padding     = 0;
            }
            c.dilation = 1;
            c.groups   = 1;
            upload(f, p + "noise_convs." + std::to_string(i) + ".weight",
                   C_cur, (n_fft + 2) * c.kernel_size, c.W, where);
            upload(f, p + "noise_convs." + std::to_string(i) + ".bias",
                   C_cur, 1, c.b, where);
        }
    }

    // ─── noise_res ────────────────────────────────────────────────────────
    noise_res.resize(num_upsamples);
    for (int i = 0; i < num_upsamples; ++i) {
        const int C_cur = init_C / (1 << (i + 1));
        // From the upstream: kernels are 7 for the non-last, 11 for the last.
        const int kr = (i + 1 < num_upsamples) ? 7 : 11;
        load_adain_resblock1(f, p + "noise_res." + std::to_string(i),
                             C_cur, kr, /*dilations=*/{1, 3, 5}, noise_res[i], where);
    }

    // ─── resblocks ────────────────────────────────────────────────────────
    resblocks.resize(static_cast<std::size_t>(num_upsamples) * num_kernels);
    for (int i = 0; i < num_upsamples; ++i) {
        const int C_cur = init_C / (1 << (i + 1));
        for (int j = 0; j < num_kernels; ++j) {
            const int idx = i * num_kernels + j;
            load_adain_resblock1(f, p + "resblocks." + std::to_string(idx),
                                 C_cur,
                                 cfg.decoder.resblock_kernel_sizes[j],
                                 cfg.decoder.resblock_dilation_sizes[j],
                                 resblocks[idx], where);
        }
    }

    // ─── conv_post ────────────────────────────────────────────────────────
    {
        const int last_C = init_C / (1 << num_upsamples);
        conv_post.in_channels  = last_C;
        conv_post.out_channels = n_fft + 2;
        conv_post.kernel_size  = 7;
        conv_post.padding      = 3;
        conv_post.dilation     = 1;
        conv_post.stride       = 1;
        conv_post.groups       = 1;
        upload(f, p + "conv_post.weight",
               n_fft + 2, last_C * 7, conv_post.W, where);
        upload(f, p + "conv_post.bias",
               n_fft + 2, 1, conv_post.b, where);
    }
}

namespace {

// Build a periodic Hann window of length N — matches scipy.signal.get_window
// with fftbins=True (which torch.hann_window also produces with periodic=True).
bt::Tensor hann_window_periodic(int N) {
    bt::Tensor w = bt::Tensor::zeros_on(bt::Device::CPU, 1, N, bt::Dtype::FP32);
    float* d = w.host_f32_mut();
    for (int n = 0; n < N; ++n) {
        constexpr float kTwoPi = 6.28318530717958647692f;
        d[n] = 0.5f - 0.5f * std::cos(kTwoPi * static_cast<float>(n) /
                                              static_cast<float>(N));
    }
    return w;
}

}  // namespace

static void gdbg(const char* tag, const bt::Tensor& t) {
    static const bool on = []() {
        const char* v = std::getenv("BROSOUNDML_DEBUG_STAGES");
        return v && v[0] && v[0] != '0';
    }();
    if (!on || t.dtype != bt::Dtype::FP32) return;
    // Debug-only path: round-trip a device tensor through to_host_vector so
    // the stats stay computable regardless of where the tensor lives.
    std::vector<float> host_buf;
    const float* d = nullptr;
    if (t.device == bt::Device::CPU) {
        d = t.host_f32();
    } else {
        host_buf = t.to_host_vector();
        d = host_buf.data();
    }
    const std::size_t n = t.size();
    if (n == 0) { std::fprintf(stderr, "[gen]   %-32s empty\n", tag); return; }
    float mn = d[0], mx = d[0];
    int n_nan = 0, n_inf = 0;
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        float v = d[i];
        if (std::isnan(v)) { ++n_nan; continue; }
        if (std::isinf(v)) { ++n_inf; continue; }
        if (v < mn) mn = v; if (v > mx) mx = v;
        sum += v;
    }
    const double mean = sum / static_cast<double>(n);
    std::fprintf(stderr,
        "[gen]   %-32s n=%zu  min=%+.3e  max=%+.3e  mean=%+.3e  nan=%d  inf=%d\n",
        tag, n, mn, mx, mean, n_nan, n_inf);
}

void Generator::forward(const bt::Tensor& gen_in, int L_in,
                        const bt::Tensor& har, int frames,
                        const bt::Tensor& style,
                        bt::Tensor& audio,
                        const CancelCheck& cancel) const {
    bt::Tensor x = gen_in;   // (1, init_C * L_in) NCL
    int L = L_in;
    int C = ups_C_in[0];     // init_C
    // Device for every op-out below. gen_in's device drives the whole stack;
    // default-constructed Tensors land on CPU and brotensor's CUDA dispatch
    // refuses mixed-device calls (Tensor::resize preserves device).
    const bt::Device dev = gen_in.device;

    gdbg("gen_in", x);
    for (int i = 0; i < num_upsamples; ++i) {
        // Cooperative cancellation: a barge-in aborts the in-flight synthesis
        // here (this upsample loop is the generator's dominant cost). Leave
        // `audio` empty so the caller emits a cancelled (empty) buffer.
        if (cancel && cancel()) { audio = bt::Tensor{}; return; }
        // x <- leaky_relu(x, 0.1)
        {
            bt::Tensor tmp = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
            bt::leaky_relu_forward(x, 0.1f, tmp);
            x = std::move(tmp);
        }
        { char b[32]; std::snprintf(b, sizeof(b), "ups%d_after_lrelu", i); gdbg(b, x); }

        // x_source = noise_convs[i](har) (1, C_cur*L_after_noise_conv)
        bt::Tensor x_source = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        noise_convs[i].forward(har, /*N=*/1, /*L=*/frames, x_source);
        const int L_src = x_source.cols / ups_C_out[i];
        { char b[32]; std::snprintf(b, sizeof(b), "ups%d_noise_conv", i); gdbg(b, x_source); }

        // x_source = noise_res[i](x_source, s)
        adain_resblock1_forward(noise_res[i], x_source, ups_C_out[i], L_src, style);
        { char b[32]; std::snprintf(b, sizeof(b), "ups%d_noise_res", i); gdbg(b, x_source); }
        { char b[32]; std::snprintf(b, sizeof(b), "gen:noise[%d]", i);
          kokoro_profile_mark(dev, b); }

        // x = ups[i](x). Compute output length: L_up = (L-1)*stride - 2*pad + (kL-1) + 1
        const int kL = ups_k[i];
        const int s  = ups_stride[i];
        const int p  = ups_pad[i];
        const int L_up = (L - 1) * s - 2 * p + (kL - 1) + 1;
        bt::Tensor up_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        bt::conv_transpose1d_forward(x, ups_W[i], &ups_b[i],
                                     /*N=*/1, /*C_in=*/ups_C_in[i], /*L=*/L,
                                     /*C_out=*/ups_C_out[i], /*kL=*/kL,
                                     /*stride=*/s, /*padding=*/p,
                                     /*output_padding=*/0, /*dilation=*/1,
                                     up_out);
        x = std::move(up_out);
        int L_x = L_up;
        // Final upsample stage applies a left-pad-by-1 reflection.
        if (i == num_upsamples - 1) {
            bt::Tensor padded = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
            bt::pad1d_forward(x, 1, ups_C_out[i], L_x,
                              /*pad_left=*/1, /*pad_right=*/0, /*mode=*/1,
                              padded);
            x = std::move(padded);
            L_x += 1;
        }

        // x = x + x_source  (broadcast at the channel level — same C, same L expected)
        // Length should match L_src == L_x by Kokoro's design.
        if (L_src != L_x) {
            fail("Generator::forward",
                 "noise source length " + std::to_string(L_src) +
                 " != upsampled length " + std::to_string(L_x) +
                 " at stage " + std::to_string(i));
        }
        bt::add_inplace(x, x_source);
        { char b[32]; std::snprintf(b, sizeof(b), "ups%d_after_add", i); gdbg(b, x); }
        { char b[32]; std::snprintf(b, sizeof(b), "gen:ups[%d]", i);
          kokoro_profile_mark(dev, b); }

        // resblocks i*num_kernels..(i+1)*num_kernels -> averaged residual sum.
        bt::Tensor xs = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        for (int j = 0; j < num_kernels; ++j) {
            bt::Tensor xj = x;
            adain_resblock1_forward(resblocks[i * num_kernels + j], xj,
                                    ups_C_out[i], L_x, style);
            { char b[32]; std::snprintf(b, sizeof(b), "ups%d_resblock%d", i, j); gdbg(b, xj); }
            if (j == 0) {
                xs = std::move(xj);
            } else {
                bt::add_inplace(xs, xj);
            }
        }
        bt::scale_inplace(xs, 1.0f / static_cast<float>(num_kernels));
        x = std::move(xs);
        L = L_x;
        C = ups_C_out[i];
        { char b[32]; std::snprintf(b, sizeof(b), "ups%d_out", i); gdbg(b, x); }
        { char b[32]; std::snprintf(b, sizeof(b), "gen:resblocks[%d]", i);
          kokoro_profile_mark(dev, b); }
    }

    // x = leaky_relu(x); x = conv_post(x); split into spec / phase; iSTFT.
    {
        bt::Tensor tmp = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        bt::leaky_relu_forward(x, 0.01f, tmp);  // PyTorch leaky_relu default slope.
        x = std::move(tmp);
    }
    bt::Tensor post = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    conv_post.forward(x, /*N=*/1, /*L=*/L, post);
    gdbg("conv_post", post);
    kokoro_profile_mark(dev, "gen:conv_post");

    // post is (1, (n_fft+2)*L). The first (n_fft/2+1) channels are log-magnitude;
    // the next (n_fft/2+1) channels are pre-sin phase.
    //
    // Device-side assembly: the two channel blocks are contiguous in NCL, so
    // view each half in place, apply exp / sin elementwise, then transpose
    // NCL -> frame-major (L, n_freq) via nchw_to_sequence — the layout
    // complex_from_polar consumes. No host round-trip.
    const int n_freq = n_fft / 2 + 1;
    // `dev` already declared at the top of Generator::forward — reuse it.
    bt::Tensor mag_frames = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor pha_frames = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    {
        float* base = static_cast<float*>(post.data);
        bt::Tensor logmag = bt::Tensor::view(dev, base,
                                             1, n_freq * L, bt::Dtype::FP32);
        bt::Tensor phin   = bt::Tensor::view(
            dev, base + static_cast<std::size_t>(n_freq) * L,
            1, n_freq * L, bt::Dtype::FP32);
        bt::Tensor mag_ncl = bt::Tensor::empty_on(dev, 1, n_freq * L, bt::Dtype::FP32);
        bt::Tensor sin_ncl = bt::Tensor::empty_on(dev, 1, n_freq * L, bt::Dtype::FP32);
        bt::exp_forward(logmag, mag_ncl);
        bt::sin_forward(phin, sin_ncl);
        bt::nchw_to_sequence(mag_ncl, /*N=*/1, n_freq, /*H=*/1, /*W=*/L, mag_frames);
        bt::nchw_to_sequence(sin_ncl, /*N=*/1, n_freq, /*H=*/1, /*W=*/L, pha_frames);
    }
    kokoro_profile_mark(dev, "gen:magphase");

    // Build complex spectrogram on `dev`; istft consumes the same layout.
    bt::Tensor spec_complex = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);  // (frames, 2*n_freq) interleaved
    bt::complex_from_polar(mag_frames, pha_frames, spec_complex);

    // iSTFT: signal_len = (frames - 1) * hop for center-true mode. The window
    // is a small length-win_size lookup table — built on host (lookup) then
    // uploaded to `dev` so istft can consume it without a device mismatch.
    const int signal_len = (L - 1) * hop_size;
    bt::Tensor window_host = hann_window_periodic(win_size);
    bt::Tensor window = window_host.to(dev);
    if (audio.device != dev) {
        audio = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    }
    bt::istft(spec_complex, window,
              /*N=*/1, signal_len, n_fft, hop_size, win_size,
              /*center=*/true, /*normalized=*/false, audio);
}

// ─── HarmonicSource ────────────────────────────────────────────────────────

void HarmonicSource::load_from(const stf::File& f, const KokoroConfig& cfg) {
    const std::string where = "HarmonicSource::load_from";
    const std::string p = "decoder.module.generator.m_source.";
    sample_rate    = cfg.sample_rate;
    harmonic_num   = 8;
    n_fft          = cfg.decoder.gen_istft_n_fft;
    hop_size       = cfg.decoder.gen_istft_hop_size;
    win_size       = n_fft;
    int upr = 1;
    for (int r : cfg.decoder.upsample_rates) upr *= r;
    upsample_scale = upr * hop_size;
    sine_amp       = 0.1f;
    l_linear.in_features  = harmonic_num + 1;
    l_linear.out_features = 1;
    upload(f, p + "l_linear.weight", 1, harmonic_num + 1, l_linear.W, where);
    upload(f, p + "l_linear.bias",   1, 1,                l_linear.b, where);
}

void HarmonicSource::forward(const bt::Tensor& F0_pred, int frame_count,
                             int& signal_len, int& stft_frames,
                             bt::Tensor& har) const {
    signal_len = frame_count * upsample_scale;
    const float sr_f = static_cast<float>(sample_rate);
    const bt::Device dev = F0_pred.device;
    const std::vector<float> f0_host = F0_pred.to_host_vector();
    const float* f0_d = f0_host.data();
    const int dim = harmonic_num + 1;

    // Deterministic PRNG for bit-exact phase and noise synthesis across runs.
    uint32_t rng = 123456789u;
    auto next_u01 = [&rng]() -> float {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return static_cast<float>(rng & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
    };

    // Initial phases for harmonics: fundamental at 0, higher harmonics distributed in [0, 2pi)
    std::vector<float> phases(dim, 0.0f);
    for (int h = 1; h < dim; ++h) {
        phases[h] = next_u01() * 6.28318530717958647692f;
    }

    std::vector<float> sine_waves(static_cast<std::size_t>(signal_len) * dim, 0.0f);
    const float noise_std = 0.003f * sine_amp;
    const float uv_amp = sine_amp / 3.0f;

    for (int t = 0; t < signal_len; ++t) {
        const int frame = t / upsample_scale;
        const float f0 = f0_d[std::min(frame, frame_count - 1)];
        const bool voiced = (f0 > 0.0f);

        for (int h = 0; h < dim; ++h) {
            float sample_val = 0.0f;
            if (voiced) {
                const float omega = 6.28318530717958647692f *
                                    static_cast<float>(h + 1) * f0 / sr_f;
                phases[h] += omega;
                if (phases[h] > 6.28318530717958647692f) {
                    phases[h] -= 6.28318530717958647692f;
                }
                const float n_val = (next_u01() * 2.0f - 1.0f) * noise_std;
                sample_val = std::sin(phases[h]) * sine_amp + n_val;
            } else {
                sample_val = (next_u01() * 2.0f - 1.0f) * uv_amp;
            }
            sine_waves[static_cast<std::size_t>(t) * dim + h] = sample_val;
        }
    }

    // 2. l_linear: (signal_len, dim) -> (signal_len, 1).
    bt::Tensor sine_t = bt::Tensor::from_host_on(dev, sine_waves.data(),
                                                 signal_len, dim);
    bt::Tensor merged = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::linear_forward_batched(l_linear.W, l_linear.b, sine_t, merged);

    // 3. tanh.
    bt::Tensor tanh_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::tanh_forward(merged, tanh_out);

    // 4. STFT on har_source (1, signal_len).
    bt::Tensor har_source = bt::Tensor::zeros_on(dev, 1, signal_len, bt::Dtype::FP32);
    bt::copy_d2d(tanh_out, 0, har_source, 0, signal_len);

    bt::Tensor window_host = hann_window_periodic(win_size);
    bt::Tensor window = window_host.to(dev);
    bt::Tensor spec = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::stft(har_source, window, /*N=*/1, n_fft, hop_size, win_size,
             /*center=*/true, /*normalized=*/false, spec);

    const int n_freq = n_fft / 2 + 1;
    stft_frames = spec.rows;

    bt::Tensor mag_frames = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor pha_frames = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::complex_abs(spec, mag_frames);
    bt::complex_angle(spec, pha_frames);

    bt::Tensor mag_ncl = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor pha_ncl = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::sequence_to_nchw(mag_frames, /*N=*/1, /*C=*/n_freq,
                         /*H=*/1, /*W=*/stft_frames, mag_ncl);
    bt::sequence_to_nchw(pha_frames, /*N=*/1, /*C=*/n_freq,
                         /*H=*/1, /*W=*/stft_frames, pha_ncl);
    har = bt::Tensor::zeros_on(dev, 1, (2 * n_freq) * stft_frames, bt::Dtype::FP32);
    bt::copy_d2d(mag_ncl, 0, har, 0,                    n_freq * stft_frames);
    bt::copy_d2d(pha_ncl, 0, har, n_freq * stft_frames, n_freq * stft_frames);
}


}  // namespace brosoundml
