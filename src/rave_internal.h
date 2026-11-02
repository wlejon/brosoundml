#pragma once

#include "brosoundml/rave.h"

#include "brotensor/json.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <algorithm>
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
namespace j  = brotensor::json;
namespace fs = std::filesystem;

namespace {
namespace detail_rave {

[[noreturn]] inline void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: rave: " + msg);
}

[[maybe_unused]] inline const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Upload a 2D-flattened weight to FP32 on `dev` (RAVE weights are F32 on disk).
[[maybe_unused]] inline bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols, bt::Device dev) {
    bt::Tensor t;
    { bt::DeviceScope cpu(bt::Device::CPU); sf::upload(need(f, name), rows, cols, t); }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
[[maybe_unused]] inline bt::Tensor up_vec(const sf::File& f, const std::string& name, int c, bt::Device dev) {
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
[[maybe_unused]] inline ConvW load_conv(const sf::File& f, const std::string& name, int groups, bt::Device dev) {
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
[[maybe_unused]] inline ConvW load_convt(const sf::File& f, const std::string& name, bt::Device dev) {
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
[[maybe_unused]] inline ConvW load_named_conv(const sf::File& f, const std::string& wname,
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
[[maybe_unused]] inline ConvW load_named_convt(const sf::File& f, const std::string& wname,
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

[[maybe_unused]] inline BN load_bn(const sf::File& f, const std::string& name, int c, bt::Device dev) {
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
[[maybe_unused]] inline bt::Tensor cconv(const bt::Tensor& x, int Cin, int L, const ConvW& c,
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
[[maybe_unused]] inline bt::Tensor cconvt(const bt::Tensor& x, int Cin, int L, const ConvW& c, int stride, int& Lout) {
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
[[maybe_unused]] inline bt::Tensor frun_conv(const bt::Tensor& x, int Cin, int L, const ConvW& c,
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
[[maybe_unused]] inline bt::Tensor frun_convt(const bt::Tensor& x, int Cin, int L0, const ConvW& c,
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

[[maybe_unused]] inline bt::Tensor leaky(const bt::Tensor& x, float slope) {
    bt::Tensor y;
    bt::leaky_relu_forward(x, slope, y);
    return y;
}

// PQMF reverse_half (in place): multiply ODD channels by (-1)^(t+1) along time
// (t even -> -1, t odd -> +1). NOT a flat sign flip. x is (1, C*L) NCL.
[[maybe_unused]] inline void reverse_half_inplace(bt::Tensor& x, int C, int L) {
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
[[maybe_unused]] inline bt::Tensor mod_sigmoid(const bt::Tensor& x) {
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

[[maybe_unused]] inline std::string slurp(const std::string& path) {
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

}  // namespace detail_rave

namespace {
using namespace detail_rave;
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
    bt::Tensor          gim_inv_a;        // exp(-log_a) = 1/a, for the gimbal inverse
    bt::Tensor          gim_zero, gim_one;// batch_norm running_mean/var to fold the affine

    // Latent-completion scheme (how the discarded latent dims are filled at decode):
    //   "noise_pca" — Family B: fill with N(0,1), inverse-PCA, decode. No gimbal,
    //                 no prior. (pre_process_latent in the .ts.)
    //   "prior_pca" / "prior_kld" — Family A: an autoregressive prior generates the
    //                 discarded dims frame by frame; the result is inverse-PCA'd
    //                 (use_pca) or scattered back via argsort(kld_idxs), then the
    //                 gimbal affine is inverted before the decoder.
    std::string         latent_mode = "noise_pca";
    bool                use_pca     = true;
    std::vector<int>    kld_idxs;          // (full) channel permutation (Family A)
    std::vector<int>    kld_perm;          // argsort(kld_idxs) — the decode scatter
    std::vector<FlatOp> prior_ops;         // autoregressive prior op-list (Family A)
    int                 prior_window = 1;  // receptive field (frames) for the windowed AR loop

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
    // Family A latent completion: run the autoregressive prior frame-by-frame to
    // fill the discarded dims, then scatter (kld) / inverse-PCA (pca) back to
    // encoder space and invert the gimbal. Returns the (1, full*T) decoder input.
    bt::Tensor build_prior_latent(const float* latent, int n_latent, int T) const;
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


}  // namespace brosoundml
