#include "brosoundml/rave.h"

#include "brosoundml/detail/json.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;
namespace j  = detail::json;
namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: rave: " + msg);
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Upload a 2D-flattened weight to FP32 on `dev` (RAVE weights are F32 on disk).
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols, bt::Device dev) {
    bt::Tensor t;
    { bt::DeviceScope cpu(bt::Device::CPU); sf::upload(need(f, name), rows, cols, t); }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
bt::Tensor up_vec(const sf::File& f, const std::string& name, int c, bt::Device dev) {
    return up(f, name, c, 1, dev);
}

// ── parameter blocks ─────────────────────────────────────────────────────────

struct ConvW {
    bt::Tensor w;          // (Cout, (Cin/groups)*K) for conv; (Cin,(Cout)*K) for convT
    bt::Tensor b;
    bool       has_b = false;
    int        cin = 0, cout = 0, k = 0;
    int        cin_pg = 0;  // weight.shape[1] = Cin/groups; groups derived at runtime
};

struct BN {
    bt::Tensor g, beta, mean, var;
};

struct EncLayer {
    ConvW conv;
    int   stride  = 1;
    int   groups  = 1;
    bool  has_bn  = false;
    BN    bn;
    bool  act     = true;   // trailing LeakyReLU
};

struct ResBlock {
    ConvW c1;               // dilated conv
    int   dil = 1;
    ConvW c2;               // dilation-1 conv
};

struct UpStack {
    ConvW                 convt;
    int                   stride = 1;
    std::vector<ResBlock> res;
};

// Load a conv1d weight (PyTorch layout (Cout, Cin/groups, K)) from its view's
// 3D shape, flattening to brotensor's (Cout, (Cin/groups)*K).
ConvW load_conv(const sf::File& f, const std::string& name, int groups, bt::Device dev) {
    const sf::TensorView& wv = need(f, name + ".weight");
    if (wv.shape.size() != 3) fail("conv '" + name + "' is not rank-3");
    const int d0 = static_cast<int>(wv.shape[0]);
    const int d1 = static_cast<int>(wv.shape[1]);
    const int k  = static_cast<int>(wv.shape[2]);
    ConvW c;
    c.cout   = d0;
    c.cin    = d1 * groups;
    c.cin_pg = d1;
    c.k      = k;
    c.w      = up(f, name + ".weight", d0, d1 * k, dev);
    if (const sf::TensorView* bv = f.find(name + ".bias")) {
        c.b = up_vec(f, name + ".bias", static_cast<int>(bv->shape[0]), dev);
        c.has_b = true;
    }
    return c;
}

// Load a conv_transpose1d weight (PyTorch layout (Cin, Cout/groups, K)) — direct
// flatten to brotensor's (Cin, (Cout/groups)*K). groups == 1 for RAVE upsamplers.
ConvW load_convt(const sf::File& f, const std::string& name, bt::Device dev) {
    const sf::TensorView& wv = need(f, name + ".weight");
    if (wv.shape.size() != 3) fail("convT '" + name + "' is not rank-3");
    const int d0 = static_cast<int>(wv.shape[0]);   // Cin
    const int d1 = static_cast<int>(wv.shape[1]);   // Cout
    const int k  = static_cast<int>(wv.shape[2]);
    ConvW c;
    c.cin  = d0;
    c.cout = d1;
    c.k    = k;
    c.w    = up(f, name + ".weight", d0, d1 * k, dev);
    if (const sf::TensorView* bv = f.find(name + ".bias")) {
        c.b = up_vec(f, name + ".bias", static_cast<int>(bv->shape[0]), dev);
        c.has_b = true;
    }
    return c;
}

// ── flat-layout loaders (FLAT exports name weights by their full graph path; the
//    op-list gives exact weight/bias tensor keys, so load by name) ─────────────

// Conv1d weight (Cout, Cin/groups, K) by its full tensor name. groups is derived
// at runtime from the input channel count (groups = Cin / weight.shape[1]).
ConvW load_named_conv(const sf::File& f, const std::string& wname,
                      const std::string& bname, bt::Device dev) {
    const sf::TensorView& wv = need(f, wname);
    if (wv.shape.size() != 3) fail("flat conv '" + wname + "' is not rank-3");
    const int d0 = static_cast<int>(wv.shape[0]);   // Cout
    const int d1 = static_cast<int>(wv.shape[1]);   // Cin/groups
    const int k  = static_cast<int>(wv.shape[2]);
    ConvW c;
    c.cout = d0; c.cin_pg = d1; c.cin = d1; c.k = k;
    c.w = up(f, wname, d0, d1 * k, dev);
    if (!bname.empty()) {
        if (const sf::TensorView* bv = f.find(bname)) {
            c.b = up_vec(f, bname, static_cast<int>(bv->shape[0]), dev);
            c.has_b = true;
        }
    }
    return c;
}

// ConvTranspose1d weight (Cin, Cout/groups, K) by name; RAVE upsamplers use
// groups == 1, so Cout == weight.shape[1].
ConvW load_named_convt(const sf::File& f, const std::string& wname,
                       const std::string& bname, bt::Device dev) {
    const sf::TensorView& wv = need(f, wname);
    if (wv.shape.size() != 3) fail("flat convT '" + wname + "' is not rank-3");
    const int d0 = static_cast<int>(wv.shape[0]);   // Cin
    const int d1 = static_cast<int>(wv.shape[1]);   // Cout
    const int k  = static_cast<int>(wv.shape[2]);
    ConvW c;
    c.cin = d0; c.cout = d1; c.cin_pg = d1; c.k = k;
    c.w = up(f, wname, d0, d1 * k, dev);
    if (!bname.empty()) {
        if (const sf::TensorView* bv = f.find(bname)) {
            c.b = up_vec(f, bname, static_cast<int>(bv->shape[0]), dev);
            c.has_b = true;
        }
    }
    return c;
}

BN load_bn(const sf::File& f, const std::string& name, int c, bt::Device dev) {
    BN bn;
    bn.g    = up_vec(f, name + ".weight",       c, dev);
    bn.beta = up_vec(f, name + ".bias",         c, dev);
    bn.mean = up_vec(f, name + ".running_mean", c, dev);
    bn.var  = up_vec(f, name + ".running_var",  c, dev);
    return bn;
}

// ── offline-causal compute primitives (cached_conv with zeroed caches) ────────

// Causal conv: left-pad dilation*(k-1), then strided/dilated/grouped valid conv.
// Lout = (L-1)/stride + 1. Matches rave_reference.py::causal_conv.
bt::Tensor cconv(const bt::Tensor& x, int Cin, int L, const ConvW& c,
                 int stride, int dilation, int groups, int& Lout) {
    const int pad_left = dilation * (c.k - 1);
    bt::Tensor padded, y;
    bt::pad1d_forward(x, /*N=*/1, Cin, L, pad_left, /*pad_right=*/0, /*mode=*/0, padded);
    bt::conv1d(padded, c.w, c.has_b ? &c.b : nullptr, /*N=*/1, Cin, L + pad_left,
               c.cout, c.k, stride, /*padding=*/0, dilation, groups, y);
    Lout = (L - 1) / stride + 1;
    return y;
}

// Causal transposed conv: full conv_transpose1d, drop the last (k - stride).
// Lout = L*stride. Matches rave_reference.py::causal_convt.
bt::Tensor cconvt(const bt::Tensor& x, int Cin, int L, const ConvW& c, int stride, int& Lout) {
    bt::Tensor full;
    bt::conv_transpose1d_forward(x, c.w, c.has_b ? &c.b : nullptr, /*N=*/1, Cin, L,
                                 c.cout, c.k, stride, /*padding=*/0, /*output_padding=*/0,
                                 /*dilation=*/1, full);
    const int L_full = (L - 1) * stride + c.k;
    const int trim   = c.k - stride;
    Lout = L_full - trim;
    bt::Tensor y = bt::Tensor::zeros_on(x.device, 1, c.cout * Lout, bt::Dtype::FP32);
    for (int ch = 0; ch < c.cout; ++ch)
        bt::copy_d2d(full, ch * L_full, y, ch * Lout, Lout);
    return y;
}

// Flat causal conv with an EXPLICIT left-pad (read from the topology op-list).
// The pad is not always dilation*(k-1): newer cached convs store dil*(k-1)-(s-1),
// so a stride-2 k5 downsampler left-pads 3, not 4. groups = Cin / weight.shape[1].
bt::Tensor frun_conv(const bt::Tensor& x, int Cin, int L, const ConvW& c,
                     int stride, int dilation, int left_pad, int& Lout) {
    const int groups = Cin / c.cin_pg;
    bt::Tensor padded, y;
    bt::pad1d_forward(x, /*N=*/1, Cin, L, left_pad, /*pad_right=*/0, /*mode=*/0, padded);
    bt::conv1d(padded, c.w, c.has_b ? &c.b : nullptr, /*N=*/1, Cin, L + left_pad,
               c.cout, c.k, stride, /*padding=*/0, dilation, groups, y);
    Lout = (L + left_pad - dilation * (c.k - 1) - 1) / stride + 1;
    return y;
}

// Flat causal transposed conv (UpsampleLayer): prepend cpad, conv_transpose1d
// with the graph's padding, then trim the output to exactly L0*stride. Handles
// both the CachedPadding+convT(pad>0) and bare convT(pad=0) export conventions.
bt::Tensor frun_convt(const bt::Tensor& x, int Cin, int L0, const ConvW& c,
                      int stride, int pad, int cpad, int& Lout) {
    bt::Tensor xp;
    int Lin = L0;
    if (cpad > 0) { bt::pad1d_forward(x, 1, Cin, L0, cpad, 0, 0, xp); Lin = L0 + cpad; }
    const bt::Tensor& in = (cpad > 0) ? xp : x;
    // conv_transpose1d with padding=p == full transpose (padding 0) with p samples
    // dropped from each end. Run the padding-0 form (the only variant the legacy
    // path verified bit-exact) and slice [pad, pad+L0*stride) ourselves.
    bt::Tensor full;
    bt::conv_transpose1d_forward(in, c.w, c.has_b ? &c.b : nullptr, /*N=*/1, Cin, Lin,
                                 c.cout, c.k, stride, /*padding=*/0, /*output_padding=*/0,
                                 /*dilation=*/1, full);
    const int L_full = (Lin - 1) * stride + c.k;
    const int target = L0 * stride;
    Lout = target;
    bt::Tensor y = bt::Tensor::zeros_on(x.device, 1, c.cout * target, bt::Dtype::FP32);
    for (int ch = 0; ch < c.cout; ++ch)
        bt::copy_d2d(full, ch * L_full + pad, y, ch * target, target);
    return y;
}

bt::Tensor leaky(const bt::Tensor& x, float slope) {
    bt::Tensor y;
    bt::leaky_relu_forward(x, slope, y);
    return y;
}

// PQMF reverse_half (in place): multiply ODD channels by (-1)^(t+1) along time
// (t even -> -1, t odd -> +1). NOT a flat sign flip. x is (1, C*L) NCL.
void reverse_half_inplace(bt::Tensor& x, int C, int L) {
    std::vector<float> mask(static_cast<std::size_t>(C) * L, 1.0f);
    for (int c = 1; c < C; c += 2)
        for (int t = 0; t < L; ++t)
            mask[static_cast<std::size_t>(c) * L + t] = (t % 2 == 0) ? -1.0f : 1.0f;
    bt::Tensor m = bt::Tensor::from_host_on(x.device, mask.data(), 1, C * L);
    bt::mul_inplace(x, m);
}

// Host-side mod_sigmoid for the noise-branch amplitudes (matches the device op).
inline float mod_sigmoid_scalar(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return 2.0f * std::pow(s, 2.3f) + 1.0e-7f;
}

// mod_sigmoid(x) = 2*sigmoid(x)^2.3 + 1e-7, via s^2.3 = exp(2.3*log(s)).
bt::Tensor mod_sigmoid(const bt::Tensor& x) {
    bt::Tensor s;
    bt::sigmoid_forward(x, s);
    bt::Tensor lg;
    bt::log_forward(s, lg);
    bt::scale_inplace(lg, 2.3f);
    bt::Tensor p;
    bt::exp_forward(lg, p);             // s^2.3
    bt::scale_inplace(p, 2.0f);
    bt::add_scalar_inplace(p, 1.0e-7f);
    return p;
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ── flat topology op-list (newer RAVE exports) ───────────────────────────────
// One entry per emitted op. push…add brackets a residual block (add folds the
// pushed input back in). Replays scripts/rave_reference.py::TopoRave.run.
enum FKind { F_CONV, F_CONVT, F_LEAKY, F_SNAKE, F_PUSH, F_ADD };
struct FlatOp {
    FKind      kind = F_LEAKY;
    ConvW      conv;                 // F_CONV / F_CONVT
    int        stride = 1, dil = 1, left_pad = 0;   // F_CONV
    int        pad = 0, cpad = 0;                    // F_CONVT
    bt::Tensor alpha;                // F_SNAKE: per-channel (C,1)
};

}  // namespace

// ── Impl ─────────────────────────────────────────────────────────────────────

struct Rave::Impl {
    RaveConfig cfg;
    bt::Device dev = bt::Device::CPU;
    bool       is_loaded = false;

    // FLAT path (newer topology-driven exports). When false, the legacy
    // fixed-topology RAVE-v2 members below are used instead. The two paths share
    // only pqmf_fwd/pqmf_inv, the PCA convs (enc_pca/dec_pca), and the synth wave/
    // loud convs — every conv whose causal left-pad equals dilation*(k-1).
    bool flat = false;
    std::vector<FlatOp> enc_ops;
    std::vector<FlatOp> dec_ops;
    std::string         synth_type;       // "rave" | "amp_mod"
    int                 loud_stride = 1;
    bool                has_gimbal  = false;
    bt::Tensor          gim_a, gim_b;     // per-channel (full,1): mean = mean*a + b
    bt::Tensor          gim_zero, gim_one;// batch_norm running_mean/var to fold the affine

    ConvW pqmf_fwd;        // 1 -> n_band, stride n_band
    ConvW pqmf_inv;        // n_band -> n_band, stride 1

    std::vector<EncLayer> enc;

    // PCA projection as 1x1 convs. encode: z = pca @ mean + enc_pca_b, with
    // enc_pca_b = -(pca @ latent_mean). decode unproject: x = pca^T @ z + mean.
    ConvW enc_pca;         // weight (full,full), k=1; bias = enc_pca_b
    ConvW dec_pca;         // weight (full,full) = pca^T, k=1; bias = latent_mean

    ConvW                dec_in;      // decoder.net.0
    std::vector<UpStack> dec_up;      // 4 upsample stacks
    ConvW                synth_wave;  // decoder.synth.branches.0 (-> n_band)
    ConvW                synth_loud;  // decoder.synth.branches.1 (-> 1)

    // Noise-synth branch (decoder.synth.branches.2): 3 stride-4 convs -> per-band
    // amplitudes -> impulse response -> fft_convolve(white noise). Optional —
    // present only when the model ships the branch.
    bool                 has_noise = false;
    std::vector<ConvW>   synth_noise;   // 3 convs (net.0, net.2, net.4)
    int                  noise_bands = 0;
    bt::Tensor           noise_ir_mat;  // (ir_target=64, noise_bands): amp_to_impulse_response
                                        // folded to a matrix (the op is linear in amp)
    bt::Tensor           noise_ir_bias; // zeros (64, 1) for linear_forward_batched

    void load(const std::string& model_dir, bt::Device device);
    void build_pca(const sf::File& f, int full);
    void load_flat(const sf::File& f, const j::Value& root);
    // Replay a flat op-list over h (1, Cin*Lin); returns h (1, Cout*Lout).
    bt::Tensor run_ops(const std::vector<FlatOp>& ops, bt::Tensor h,
                       int Cin, int Lin, int& Cout, int& Lout) const;
    RaveLatent encode(const float* audio, int n) const;
    // Decode one channel to its mono waveform (frames * total_ratio samples).
    // `channel` selects this channel's latent pad (injected slice, or a seeded
    // N(0,1) draw) and noise draw — the sole source of L/R decorrelation.
    std::vector<float> decode_core(const float* latent, int n_latent, int frames,
                                   const RaveDecodeOptions& opts, int channel) const;
    AudioBuffer decode(const float* latent, int n_latent, int frames,
                       const RaveDecodeOptions& opts) const;
    RaveMultiBuffer decode_multi(const float* latent, int n_latent, int frames,
                                 const RaveDecodeOptions& opts) const;
};

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

    // z = pca @ (mean - latent_mean)  (folded into the 1x1 conv bias).
    int Lz;
    bt::Tensor z = cconv(mean, full, T, enc_pca, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Lz);

    // Crop to the kept latent dims.
    const int nl = cfg.cropped_latent_size;
    bt::Tensor zc = bt::Tensor::zeros_on(dev, 1, nl * T, bt::Dtype::FP32);
    bt::copy_d2d(z, 0, zc, 0, nl * T);

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

    // Unproject: pad nl -> full latents, then x = pca^T @ z + latent_mean. The
    // discarded dims [n_latent, full) carry the per-channel pad — zero by default
    // (deterministic mono), or an independent N(0,1) draw for stereo width. This
    // pad is the only thing that differs between stereo channels.
    bt::Tensor zf = bt::Tensor::zeros_on(dev, 1, full * T, bt::Dtype::FP32);
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
            // Independent per-channel draw: key derived from seed (so it never
            // collides with the noise stream), counter = channel.
            padT = bt::Tensor::zeros_on(dev, padDims, T, bt::Dtype::FP32);
            bt::randn(opts.seed ^ 0x52415645ull, static_cast<std::uint64_t>(channel), padT);
            bt::scale_inplace(padT, opts.latent_pad_std);
        }
        bt::copy_d2d(padT, 0, zf, static_cast<std::size_t>(n_latent) * T,
                     static_cast<std::size_t>(padDims) * T);
    }
    // ── FLAT decode: decoder.net consumes the full (zero-padded) latent DIRECTLY.
    //    Flat exports fold NO inverse-PCA into decode (the latent_pca rotation is
    //    applied only at encode; the decoder is trained in that rotated space) —
    //    unlike the legacy path's dec_pca unprojection below. Replay the decoder
    //    op-list, then the flat synth + PQMF. Deterministic (no noise branch). ──
    if (flat) {
        const int nb = cfg.n_band;
        int C, Lc;
        bt::Tensor hh = run_ops(dec_ops, zf, full, T, C, Lc);

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
