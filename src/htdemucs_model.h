// HTDemucs (Demucs v4, hybrid transformer) — one network of the htdemucs_ft
// bag. Internal to brosoundml: include/brosoundml/htdemucs.h is the public
// surface (the bag, segmenting, normalisation, resampling); this is the
// network a segment runs through.
//
// The reference is demucs 4.0.1 (htdemucs.py, hdemucs.py, demucs.py,
// transformer.py). For htdemucs_ft every model has depth 4, all four layers
// on the frequency axis (2048 -> 512 -> 128 -> 32 -> 8 bins) beside a time
// branch of four strided convs (L -> L/4 ... -> L/256), no group norm in the
// encoder/decoder (norm_starts = depth), DConv residual branches in encoder
// and decoder (dconv_mode 3), a 512-wide five-layer cross transformer behind
// 384 <-> 512 channel up/downsamplers, and complex-as-channels output.
//
// Tensors are brotensor's (rows, cols) buffers; the NCHW / NCL dims travel
// beside them. A segment batch is B rows.
#pragma once

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace brosoundml::htd {

namespace bt = brotensor;

[[noreturn]] void fail(const std::string& msg);

// The kwargs of one HTDemucs network (config.json "kwargs").
struct NetConfig {
    int sources        = 4;
    int audio_channels = 2;
    int channels       = 48;
    int growth         = 2;
    int nfft           = 4096;
    int depth          = 4;
    int kernel_size    = 8;
    int stride         = 4;
    int context        = 1;
    int dconv_depth    = 2;
    int dconv_comp     = 8;
    int bottom_channels = 512;
    int t_layers       = 5;
    int t_heads        = 8;
    float t_hidden_scale = 4.0f;
    float freq_emb     = 0.2f;
    float emb_scale    = 10.0f;
    float t_max_period = 10000.0f;
    int samplerate     = 44100;
    int segment_samples = 0;   // int(segment * samplerate): the training length
};

// ── layers ───────────────────────────────────────────────────────────────────

// Conv1d / Conv2d / ConvTranspose{1,2}d, torch weights flattened into
// brotensor's layouts (conv: (C_out, C_in*kH*kW); transposed: (C_in,
// C_out*kH*kW)). 1D convs have kh = 1.
struct Conv {
    bt::Tensor w, b;
    int cin = 0, cout = 0, kh = 1, kw = 1, sh = 1, sw = 1, ph = 0, pw = 0, dw = 1;
    bool transpose = false;

    // X (N, cin*H*W) -> Y (N, cout*Ho*Wo).
    void forward(const bt::Tensor& X, int N, int H, int W, bt::Tensor& Y) const;
    int out_h(int H) const;
    int out_w(int W) const;
};

// GroupNorm(1, C): statistics over the whole (C, L) of a row, per-channel affine.
struct GroupNorm1 {
    bt::Tensor gamma, beta;   // (C,1)
    int C = 0;
    void forward(const bt::Tensor& X, int N, int L, bt::Tensor& Y) const;
};

// DConv residual branch (demucs.py): per layer d, x += LayerScale(GLU(GN(
// conv1x1(GELU(GN(conv k3 dil 2^d (x))))))). LayerScale is folded into the
// second norm's affine for the GLU's value half.
struct DConvLayer {
    Conv c1, c2;
    GroupNorm1 n1, n2;
    int dil = 1;
};
struct DConv {
    std::vector<DConvLayer> layers;
    int C = 0;
    // X (N, C*L), updated in place.
    void forward(bt::Tensor& X, int N, int L) const;
};

// HEncLayer (no norm, rewrite 1x1 + GLU, DConv).
struct EncLayer {
    bool freq = true;
    Conv conv, rewrite;
    DConv dconv;
};

// HDecLayer (no norm, rewrite (3x3 freq / 3 time) + GLU, DConv, conv_tr).
struct DecLayer {
    bool freq = true, last = false;
    Conv rewrite, conv_tr;
    DConv dconv;
};

struct Linear {
    bt::Tensor w, b;   // (out, in), (out, 1)
    int in = 0, out = 0;
};

struct LayerNorm {
    bt::Tensor gamma, beta;   // (D,1)
};

// One layer of the cross transformer (norm_first, layer scale, norm_out):
//   self:  x += g1 * SA(norm1(x));      x += g2 * FF(norm2(x)); x = norm_out(x)
//   cross: x = q + g1 * CA(norm1(q), norm2(k)); x += g2 * FF(norm3(x)); norm_out
// g1 / g2 are folded into the output projection / linear2.
struct TLayer {
    bool cross = false;
    LayerNorm norm1, norm2, norm3;
    Linear q, k, v, o, l1, l2;
    GroupNorm1 norm_out;
};

// Per-batch-size device constants (position tables, gather indices).
struct BatchConsts {
    int B = 0;
    bt::Tensor spec_idx;     // STFT rows -> CaC (B*C*2*F, 1)
    bt::Tensor ispec_idx;    // decoder rows -> (signal, bin, re/im) (B*S*C*F*2, 1)
    bt::Tensor freq_emb;     // freq_emb replicated B times (B*C0*F1, 1)
    bt::Tensor pos_x, pos_t; // (B*T1*Fr, D), (B*T2, D)
    bt::Tensor cu_x, cu_t;   // INT32 prefix sums of the two token streams (B+1, 1)
    // (B,C,F,T) -> (B,F,C,T) and back: row gathers, cached by (C, F).
    const std::pair<bt::Tensor, bt::Tensor>& permute(bt::Device dev, int C, int F) const;
    mutable std::mutex perm_mu;
    mutable std::map<int, std::pair<bt::Tensor, bt::Tensor>> perm_cache;
};

// ── one HTDemucs network ─────────────────────────────────────────────────────

class Net {
public:
    Net();
    ~Net();
    Net(const Net&) = delete;
    Net& operator=(const Net&) = delete;

    void load(const bt::safetensors::File& f, const NetConfig& cfg, bt::Device dev);

    // mix (B, C*L) — B segments of L = segment_samples planar samples, already
    // normalised by the caller (apply_model's input) -> out (B, S*C*L), the
    // model's raw output (sources-major, then channel).
    void forward(const bt::Tensor& mix, int B, bt::Tensor& out) const;

    const NetConfig& config() const { return cfg_; }

private:
    void encode_freq_input(const bt::Tensor& mix, int B, bt::Tensor& spec) const;
    void decode_freq_output(const bt::Tensor& x, int B, bt::Tensor& wave) const;
    void transformer(bt::Tensor& x, bt::Tensor& xt, int B, int Fr, int T1, int T2) const;
    void tlayer(const TLayer& L, bt::Tensor& x, const bt::Tensor* kv, int B, int Lq, int Lk,
                const BatchConsts& bc, bool time_stream) const;
    const BatchConsts& consts(int B) const;

    NetConfig cfg_;
    bt::Device dev_ = bt::Device::CPU;
    std::vector<EncLayer> enc_, tenc_;
    std::vector<DecLayer> dec_, tdec_;   // decoding order (innermost first)
    bt::Tensor freq_emb_;                // (C0*F1, 1): freq_emb * scale * W^T
    Linear up_, down_, up_t_, down_t_;
    LayerNorm norm_in_, norm_in_t_;
    std::vector<TLayer> layers_, layers_t_;
    bt::Tensor window_;                  // (1, nfft) periodic Hann
    bt::Tensor unit_gamma_, unit_beta_;  // (1,1) 1 and 0: an affine-free GroupNorm

    mutable std::mutex consts_mu_;
    mutable std::map<int, std::unique_ptr<BatchConsts>> consts_;
};

// ── shared helpers (htdemucs_ops.cpp) ───────────────────────────────────────

// GLU over the channel halves of each row: X (N, 2*D) -> Y (N, D) = a * sigmoid(b).
void glu(const bt::Tensor& X, int N, int D, bt::Tensor& Y);
// Per-row mean and unbiased std of X (R, n): mean, std (R,1).
void row_mean_std(const bt::Tensor& X, bt::Tensor& mean, bt::Tensor& std);
// Y = X * scale[r] + shift[r] per row of X (R, n); scale/shift (R,1). Y may be X.
void row_affine(const bt::Tensor& X, const bt::Tensor& scale, const bt::Tensor& shift,
                bt::Tensor& Y);
// Y = (X - mean[r]) / (eps + std[r]) per row.
void row_normalize(const bt::Tensor& X, const bt::Tensor& mean, const bt::Tensor& std,
                   float eps, bt::Tensor& Y);
// Reinterpret an owning tensor's buffer as (r, c) in place (same element count).
void reshape(bt::Tensor& t, int r, int c);
// (R, n) slice [c0, c0+w) of every row -> (R, w).
void slice_cols(const bt::Tensor& X, int c0, int w, bt::Tensor& Y);
// INT32 (n,1) device tensor.
bt::Tensor int_tensor(bt::Device dev, const std::vector<int>& v);
// FP32 constant tensor (r, c) filled with `value`.
bt::Tensor fill(bt::Device dev, int r, int c, float value);

}  // namespace brosoundml::htd
