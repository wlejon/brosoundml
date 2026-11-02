#include "rave_internal.h"

namespace brosoundml {

void Rave::Impl::load(const std::string& model_dir, bt::Device device) {
    dev = device;

    const fs::path dir = model_dir;
    const std::string config_path = (dir / "config.json").string();
    const std::string weights_path = (dir / "model.safetensors").string();
    if (!fs::exists(config_path))  fail("no config.json under '" + model_dir + "'");
    if (!fs::exists(weights_path)) fail("no model.safetensors under '" + model_dir + "'");

    // ── config.json ──
    const j::Value root = j::parse(slurp(config_path));
    if (!root.is_object()) fail("config.json is not a JSON object");
    cfg.sampling_rate       = root.get_int("sampling_rate", 48000);
    cfg.full_latent_size    = root.get_int("full_latent_size", 128);
    cfg.cropped_latent_size = root.get_int("cropped_latent_size", 0);
    cfg.n_band              = root.get_int("n_band", 16);
    cfg.leaky_slope         = root.get_float("leaky_slope", 0.2f);
    cfg.bn_eps              = root.get_float("bn_eps", 1.0e-5f);
    if (cfg.cropped_latent_size <= 0) fail("config.json: cropped_latent_size must be > 0");

    const sf::File f = sf::File::open(weights_path);
    const int full = cfg.full_latent_size;

    // ── PQMF analysis / synthesis convs (shared; their left-pad == dil*(k-1)) ──
    pqmf_fwd = load_conv(f, "pqmf.forward_conv", /*groups=*/1, dev);
    pqmf_inv = load_conv(f, "pqmf.inverse_conv", /*groups=*/1, dev);

    // ── FLAT exports: topology-driven path (residual encoders, Snake, gimbal,
    //    amp-mod synth). Reuses PQMF + PCA; everything else comes from the op-list.
    if (root.get_string("format", "") == "flat") {
        flat = true;
        build_pca(f, full);
        load_flat(f, root);
        is_loaded = true;
        return;
    }

    // ── encoder: [conv, BN, LeakyReLU] x4 downsample, then [conv, LeakyReLU,
    //    grouped-conv(groups=2)] -> (mean | scale). Fixed RAVE v2 topology
    //    (mirrors rave_reference.py ENC / ENC_BN). ──
    struct Spec { int idx, stride, groups, bn; bool act; };
    const Spec specs[] = {
        {0,  1, 1,  1, true},
        {3,  4, 1,  4, true},
        {6,  4, 1,  7, true},
        {9,  4, 1, 10, true},
        {12, 2, 1, -1, true},
        {14, 1, 2, -1, false},
    };
    enc.clear();
    for (const Spec& s : specs) {
        EncLayer el;
        const std::string base = "encoder.net." + std::to_string(s.idx);
        el.conv   = load_conv(f, base, s.groups, dev);
        el.stride = s.stride;
        el.groups = s.groups;
        el.act    = s.act;
        if (s.bn >= 0) {
            el.has_bn = true;
            el.bn = load_bn(f, "encoder.net." + std::to_string(s.bn), el.conv.cout, dev);
        }
        enc.push_back(std::move(el));
    }

    // Encoder total downsample ratio = n_band * product(encoder strides).
    cfg.total_ratio = cfg.n_band;
    for (const EncLayer& el : enc) cfg.total_ratio *= el.stride;

    // ── PCA projection (fold the latent_mean shift into the 1x1 conv bias) ──
    build_pca(f, full);

    // ── decoder: input conv, then 4x [convT upsample + 3 residual blocks] ──
    dec_in = load_conv(f, "decoder.net.0", /*groups=*/1, dev);

    const int up_idx[]    = {1, 3, 5, 7};
    const int up_stride[] = {4, 4, 4, 2};
    const int res_dil[]   = {1, 3, 9};
    dec_up.clear();
    for (int u = 0; u < 4; ++u) {
        UpStack st;
        st.stride = up_stride[u];
        st.convt  = load_convt(f, "decoder.net." + std::to_string(up_idx[u]) + ".net.1", dev);
        const std::string rbase = "decoder.net." + std::to_string(up_idx[u] + 1) + ".net.";
        for (int b = 0; b < 3; ++b) {
            ResBlock rb;
            const std::string p = rbase + std::to_string(b) + ".aligned.branches.0.";
            rb.c1  = load_conv(f, p + "1", /*groups=*/1, dev);
            rb.dil = res_dil[b];
            rb.c2  = load_conv(f, p + "3", /*groups=*/1, dev);
            st.res.push_back(std::move(rb));
        }
        dec_up.push_back(std::move(st));
    }

    // ── synthesis branches ──
    synth_wave = load_conv(f, "decoder.synth.branches.0", /*groups=*/1, dev);
    synth_loud = load_conv(f, "decoder.synth.branches.1", /*groups=*/1, dev);

    // ── noise-synth branch (optional) ──
    // 3 stride-4 causal convs (net.0, net.2, net.4, LeakyReLU between) predict a
    // per-band amplitude response; amp_to_impulse_response turns each into a
    // 64-tap FIR, which fft_convolve applies to white noise. amp_to_impulse_
    // response (irfft -> roll -> Hann window -> zero-pad -> roll) is LINEAR in
    // the amplitude vector, so we fold it to a fixed (64 x noise_bands) matrix
    // here and apply it as one matmul at decode time.
    if (f.find("decoder.synth.branches.2.net.0.weight")) {
        synth_noise.clear();
        for (int i : {0, 2, 4})
            synth_noise.push_back(
                load_conv(f, "decoder.synth.branches.2.net." + std::to_string(i),
                          /*groups=*/1, dev));
        const int c_amp = synth_noise.back().cout;
        if (c_amp % cfg.n_band != 0)
            fail("noise branch: amp channels (" + std::to_string(c_amp) +
                 ") not divisible by n_band");
        noise_bands = c_amp / cfg.n_band;

        const int target = 64;                 // amp_to_impulse_response IR length
        const int FS = 2 * (noise_bands - 1);  // irfft length for noise_bands bins
        if (FS <= 0 || FS > target) fail("noise branch: unexpected noise_bands");

        // Build M (target x noise_bands) on CPU: column k is the IR of the unit
        // amplitude basis vector e_k, using brotensor's irfft + host roll/window.
        std::vector<float> Mh(static_cast<std::size_t>(target) * noise_bands, 0.0f);
        {
            bt::DeviceScope cpu(bt::Device::CPU);
            // Identity basis as interleaved-complex (noise_bands rows, im = 0).
            std::vector<float> basis(static_cast<std::size_t>(noise_bands) * 2 * noise_bands, 0.0f);
            for (int k = 0; k < noise_bands; ++k)
                basis[static_cast<std::size_t>(k) * 2 * noise_bands + 2 * k] = 1.0f;
            bt::Tensor cplx = bt::Tensor::from_host_on(bt::Device::CPU, basis.data(),
                                                       noise_bands, 2 * noise_bands);
            bt::Tensor a;
            bt::irfft(cplx, FS, a);                  // (noise_bands, FS)
            const std::vector<float> ah = a.to_host_vector();

            std::vector<float> hann(FS);
            const double two_pi = 6.283185307179586476925286766559;
            for (int n = 0; n < FS; ++n)
                hann[n] = static_cast<float>(0.5 * (1.0 - std::cos(two_pi * n / FS)));

            for (int k = 0; k < noise_bands; ++k) {
                const float* ak = &ah[static_cast<std::size_t>(k) * FS];
                std::vector<float> d(target, 0.0f);
                for (int i = 0; i < FS; ++i) {       // roll(+FS/2) then Hann, then zero-pad
                    const int src = ((i - FS / 2) % FS + FS) % FS;
                    d[i] = ak[src] * hann[i];
                }
                for (int i = 0; i < target; ++i) {   // roll(-FS/2) on the padded length
                    const int src = ((i + FS / 2) % target + target) % target;
                    Mh[static_cast<std::size_t>(i) * noise_bands + k] = d[src];
                }
            }
        }
        noise_ir_mat  = bt::Tensor::from_host_on(dev, Mh.data(), target, noise_bands);
        noise_ir_bias = bt::Tensor::zeros_on(dev, target, 1, bt::Dtype::FP32);
        has_noise = true;
    }

    is_loaded = true;
}

// PCA projection folded into two 1x1 convs (shared by the legacy and flat paths).
// encode:  z = pca @ (mean - latent_mean)  -> enc_pca.w = pca, bias = -(pca@mean)
// decode:  x = pca^T @ z + latent_mean     -> dec_pca.w = pca^T, bias = latent_mean
void Rave::Impl::build_pca(const sf::File& f, int full) {
    bt::Tensor pca_cpu, mean_cpu;
    {
        bt::DeviceScope cpu(bt::Device::CPU);
        sf::upload(need(f, "latent_pca"),  full, full, pca_cpu);
        sf::upload(need(f, "latent_mean"), full, 1,    mean_cpu);
    }
    const float* P = pca_cpu.host_f32();    // (full, full), row = out dim
    const float* M = mean_cpu.host_f32();   // (full)
    std::vector<float> enc_b(full), dec_w(static_cast<std::size_t>(full) * full), dec_b(full);
    for (int i = 0; i < full; ++i) {
        float s = 0.0f;
        for (int k = 0; k < full; ++k) s += P[static_cast<std::size_t>(i) * full + k] * M[k];
        enc_b[i] = -s;                       // encode bias: -(pca @ mean)
        dec_b[i] = M[i];                     // decode bias: latent_mean
    }
    for (int i = 0; i < full; ++i)           // dec weight = pca^T
        for (int k = 0; k < full; ++k)
            dec_w[static_cast<std::size_t>(i) * full + k] = P[static_cast<std::size_t>(k) * full + i];

    enc_pca.w = (dev == bt::Device::CPU) ? pca_cpu : pca_cpu.to(dev);
    enc_pca.b = bt::Tensor::from_host_on(dev, enc_b.data(), full, 1);
    enc_pca.has_b = true;
    enc_pca.cin = full; enc_pca.cout = full; enc_pca.k = 1; enc_pca.cin_pg = full;

    dec_pca.w = bt::Tensor::from_host_on(dev, dec_w.data(), full, full);
    dec_pca.b = bt::Tensor::from_host_on(dev, dec_b.data(), full, 1);
    dec_pca.has_b = true;
    dec_pca.cin = full; dec_pca.cout = full; dec_pca.k = 1; dec_pca.cin_pg = full;
}

// Parse the flat topology op-list + gimbal + synth metadata from config.json,
// loading every referenced weight by name. Mirrors convert-rave.py's emit order.
void Rave::Impl::load_flat(const sf::File& f, const j::Value& root) {
    auto opt_key = [](const j::Value& o, const char* k) -> std::string {
        const j::Value* v = o.find(k);
        return (v && v->is_string()) ? v->as_string() : std::string();
    };
    auto parse_ops = [&](const char* list_key, std::vector<FlatOp>& out) {
        const j::Value* arr = root.find(list_key);
        if (!arr || !arr->is_array()) fail(std::string("config: missing ") + list_key);
        for (const j::Value& o : arr->as_array()) {
            const std::string kind = o.get_string("kind", "");
            FlatOp op;
            if (kind == "conv") {
                op.kind = F_CONV;
                op.stride   = o.get_int("stride", 1);
                op.dil      = o.get_int("dil", 1);
                op.left_pad = o.get_int("left_pad", 0);
                op.conv = load_named_conv(f, o.at("w").as_string(), opt_key(o, "b"), dev);
            } else if (kind == "convT") {
                op.kind = F_CONVT;
                op.stride = o.get_int("stride", 1);
                op.pad    = o.get_int("pad", 0);
                op.cpad   = o.get_int("cpad", 0);
                op.conv = load_named_convt(f, o.at("w").as_string(), opt_key(o, "b"), dev);
            } else if (kind == "leaky") {
                op.kind = F_LEAKY;
            } else if (kind == "snake") {
                op.kind = F_SNAKE;
                const std::string ak = o.at("alpha").as_string();
                const int c = static_cast<int>(need(f, ak).shape[0]);
                op.alpha = up_vec(f, ak, c, dev);
            } else if (kind == "push") {
                op.kind = F_PUSH;
            } else if (kind == "add") {
                op.kind = F_ADD;
            } else {
                fail("config: unknown op kind '" + kind + "'");
            }
            out.push_back(std::move(op));
        }
    };
    parse_ops("encoder_ops", enc_ops);
    parse_ops("decoder_ops", dec_ops);

    // Latent-completion scheme + autoregressive prior (Family A).
    latent_mode = root.get_string("latent_mode", "noise_pca");
    use_pca     = root.get_bool("use_pca", true);
    kld_idxs    = root.get_int_array("kld_idxs", {});
    if (latent_mode.rfind("prior", 0) == 0) {
        if (kld_idxs.empty()) fail("config: prior latent_mode requires kld_idxs");
        parse_ops("prior_ops", prior_ops);
        // Decode scatter = argsort(kld_idxs): kld_perm[j] is the source channel in
        // the prior-ordered latent for output channel j.
        const int full = cfg.full_latent_size;
        kld_perm.assign(full, 0);
        for (int i = 0; i < full; ++i) kld_perm[i] = i;
        std::sort(kld_perm.begin(), kld_perm.end(),
                  [&](int a, int b) { return kld_idxs[a] < kld_idxs[b]; });
        // Receptive field of the (causal, stride-1) prior = 1 + sum of conv left-pads.
        prior_window = 1;
        for (const FlatOp& op : prior_ops)
            if (op.kind == F_CONV) prior_window += op.left_pad;
    }

    // total_ratio: config carries it; else recompute from the encoder strides.
    cfg.total_ratio = root.get_int("total_ratio", 0);
    if (cfg.total_ratio <= 0) {
        cfg.total_ratio = cfg.n_band;
        for (const FlatOp& op : enc_ops)
            if (op.kind == F_CONV) cfg.total_ratio *= op.stride;
    }

    // gimbal latent affine: mean = mean*exp(log_a) + b. Folded into a batch_norm
    // (running_mean=0, running_var=1, eps=0) so it runs as one device op.
    has_gimbal = root.get_bool("has_gimbal", false) && f.find("gimbal.log_a");
    if (has_gimbal) {
        const int full = cfg.full_latent_size;
        bt::Tensor log_a = up_vec(f, "gimbal.log_a", full, dev);
        bt::exp_forward(log_a, gim_a);
        bt::scale_inplace(log_a, -1.0f);     // reuse log_a (done with it) for the inverse
        bt::exp_forward(log_a, gim_inv_a);   // exp(-log_a) = 1/a
        gim_b    = up_vec(f, "gimbal.b", full, dev);
        gim_zero = bt::Tensor::zeros_on(dev, full, 1, bt::Dtype::FP32);
        std::vector<float> ones(static_cast<std::size_t>(full), 1.0f);
        gim_one  = bt::Tensor::from_host_on(dev, ones.data(), full, 1);
    }

    // synthesis: "rave" (wave/loud branches, loud broadcast over bands) or
    // "amp_mod" (decoder ends at 2*n_band -> tanh(wave*sigmoid(amp))).
    synth_type = root.get_string("synth_type", "rave");
    if (synth_type == "rave") {
        synth_wave = load_conv(f, "decoder.synth.branches.0", /*groups=*/1, dev);
        synth_loud = load_conv(f, "decoder.synth.branches.1", /*groups=*/1, dev);
        const j::Value* sv = root.find("synth");
        loud_stride = (sv && sv->is_object()) ? sv->get_int("loud_stride", 1) : 1;
    }
}


}  // namespace brosoundml
