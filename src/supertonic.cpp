#include "brosoundml/supertonic.h"

#include "supertonic_internal.h"

#include "brosoundml/detail/json.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#ifdef BROSOUNDML_HAS_CUDA
#include <brotensor/cuda_graph.h>
#endif

#include <algorithm>
#include <cmath>
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

// The weight structs (ConvW, ConvNeXtBlock, AttnLayer, StyleAttn, Gemm,
// VeFilm, VeRope, VeStyle, VeBlock) now live in supertonic_internal.h so the
// backward TU can compose against the same types. Pulled into this scope so
// every existing reference below resolves unchanged.
using namespace st_detail;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: supertonic: " + msg);
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Read an F32 view straight to a host float vector (small denorm / slope params).
std::vector<float> read_host(const sf::File& f, const std::string& name) {
    const sf::TensorView& v = need(f, name);
    if (v.dtype != sf::Dtype::F32) fail("tensor '" + name + "' is not F32");
    const float* p = reinterpret_cast<const float*>(v.data);
    return std::vector<float>(p, p + v.numel());
}

// Upload a 2D-flattened weight to FP32 on `dev` (Supertonic weights are F32).
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols,
              bt::Device dev) {
    bt::Tensor t;
    { bt::DeviceScope cpu(bt::Device::CPU); sf::upload(need(f, name), rows, cols, t); }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
bt::Tensor up_vec(const sf::File& f, const std::string& name, int c, bt::Device dev) {
    return up(f, name, c, 1, dev);
}

std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) fail("cannot open " + p.string());
    std::ostringstream ss; ss << in.rdbuf();
    return ss.str();
}

// ── parameter blocks ─────────────────────────────────────────────────────────

// Load an ONNX conv weight (Cout, Cin/groups, K) by its exact tensor name.
ConvW load_conv(const sf::File& f, const std::string& wname,
                const std::string& bname, bt::Device dev) {
    const sf::TensorView& wv = need(f, wname);
    if (wv.shape.size() != 3) fail("conv '" + wname + "' is not rank-3");
    ConvW c;
    c.cout   = static_cast<int>(wv.shape[0]);
    c.cin_pg = static_cast<int>(wv.shape[1]);
    c.k      = static_cast<int>(wv.shape[2]);
    c.w = up(f, wname, c.cout, c.cin_pg * c.k, dev);
    if (!bname.empty()) {
        if (const sf::TensorView* bv = f.find(bname)) {
            c.b = up_vec(f, bname, static_cast<int>(bv->shape[0]), dev);
            c.has_b = true;
        }
    }
    return c;
}

// pwconv2 with the ConvNeXt LayerScale gamma folded in: gamma * (W·h + b) =
// (gamma*W)·h + gamma*b, so we scale each output row's weights and bias by
// gamma_c host-side. Eliminates a per-channel broadcast multiply in the forward.
ConvW load_pwconv2_scaled(const sf::File& f, const std::string& prefix,
                          bt::Device dev) {
    const sf::TensorView& wv = need(f, prefix + ".pwconv2.weight");  // (512,2048,1)
    const int cout   = static_cast<int>(wv.shape[0]);
    const int cin_pg = static_cast<int>(wv.shape[1]);
    std::vector<float> w = read_host(f, prefix + ".pwconv2.weight");
    std::vector<float> b = read_host(f, prefix + ".pwconv2.bias");
    std::vector<float> g = read_host(f, prefix + ".gamma");  // (1,cout,1) -> cout
    if (static_cast<int>(g.size()) != cout || static_cast<int>(b.size()) != cout)
        fail("pwconv2 gamma/bias length mismatch in " + prefix);
    for (int c = 0; c < cout; ++c) {
        const float gc = g[c];
        for (int i = 0; i < cin_pg; ++i) w[static_cast<std::size_t>(c) * cin_pg + i] *= gc;
        b[c] *= gc;
    }
    ConvW out;
    out.cout = cout; out.cin_pg = cin_pg; out.k = 1; out.has_b = true;
    out.w = bt::Tensor::from_host_on(dev, w.data(), cout, cin_pg);
    out.b = bt::Tensor::from_host_on(dev, b.data(), cout, 1);
    return out;
}

// Load a folded ONNX MatMul weight [in, out] (a Linear, x @ W) + bias as a 1x1
// conv: conv weight is [out, in], so transpose host-side. Bias is [out].
ConvW load_linear_as_conv(const sf::File& f, const std::string& wname,
                          const std::string& bname, bt::Device dev) {
    const sf::TensorView& wv = need(f, wname);
    if (wv.shape.size() != 2) fail("linear '" + wname + "' is not rank-2");
    const int in  = static_cast<int>(wv.shape[0]);
    const int out = static_cast<int>(wv.shape[1]);
    const std::vector<float> w = read_host(f, wname);   // [in][out] row-major
    std::vector<float> wt(static_cast<std::size_t>(out) * in);
    for (int i = 0; i < in; ++i)
        for (int o = 0; o < out; ++o)
            wt[static_cast<std::size_t>(o) * in + i] = w[static_cast<std::size_t>(i) * out + o];
    ConvW c;
    c.cout = out; c.cin_pg = in; c.k = 1; c.has_b = true;
    c.w = bt::Tensor::from_host_on(dev, wt.data(), out, in);
    const std::vector<float> b = read_host(f, bname);
    c.b = bt::Tensor::from_host_on(dev, b.data(), out, 1);
    return c;
}

// Load an ONNX Gemm (transB=1) weight [out, in] + bias [out] as a Linear:
// y = x @ W^T + b. The weight is transposed host-side to [in, out] so a
// row-vector matmul (x [1,in] @ wt [in,out]) needs no further transpose.
Gemm load_gemm(const sf::File& f, const std::string& wname,
               const std::string& bname, bt::Device dev) {
    const sf::TensorView& wv = need(f, wname);
    if (wv.shape.size() != 2) fail("gemm '" + wname + "' is not rank-2");
    const int out = static_cast<int>(wv.shape[0]);
    const int in  = static_cast<int>(wv.shape[1]);
    const std::vector<float> w = read_host(f, wname);   // [out][in] row-major
    std::vector<float> wt(static_cast<std::size_t>(in) * out);
    for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i)
            wt[static_cast<std::size_t>(i) * out + o] = w[static_cast<std::size_t>(o) * in + i];
    Gemm g;
    g.in = in; g.out = out;
    g.wt = bt::Tensor::from_host_on(dev, wt.data(), in, out);
    const std::vector<float> b = read_host(f, bname);
    g.b = bt::Tensor::from_host_on(dev, b.data(), 1, out);
    return g;
}

// 1-D conv with an explicit (pad_left, pad_right) pre-pad in `mode` (0 zero,
// 2 replicate/edge). Output length == L + pad_left + pad_right - dilation*(k-1).
// groups == Cin for depthwise; k==1 convs pass pad 0. scratch is reused.
bt::Tensor pconv(const bt::Tensor& x, const ConvW& c, int Cin, int L,
                 int dilation, int groups, int pad_left, int pad_right,
                 int mode, bt::Tensor& scratch) {
    // A pointwise (1x1, full, unpadded) conv is exactly Y[Cout,L] = W[Cout,Cin] @
    // X[Cin,L] — a GEMM. brotensor's conv1d routes through the naive direct-conv
    // conv2d kernel (no FP32 fast path), which is ~20x slower than the tiled
    // matmul for these channel-heavy shapes, so express it as a matmul + a
    // channel-broadcast bias. (Depthwise / kernel>1 / padded convs stay convs.)
    if (c.k == 1 && groups == 1 && pad_left == 0 && pad_right == 0) {
        bt::Tensor y;
        const bt::Tensor xv = bt::Tensor::view(x.device, x.data, Cin, L, x.dtype);
        bt::matmul(c.w, xv, y);                 // [Cout, L], channel-major data
        if (c.has_b) bt::add_channel_bias_inplace(y, c.b, c.cout, L);
        // conv1d tags its output (1, Cout*L); the matmul tags it (Cout, L) with
        // the same channel-major bytes. Re-tag so the downstream shape checks
        // (pad1d / nchw_to_sequence expect rows==N) accept it — no data move.
        y.rows = 1;
        y.cols = c.cout * L;
        return y;
    }
    bt::Tensor y;
    const bt::Tensor* in = &x;
    int Lp = L;
    if (pad_left > 0 || pad_right > 0) {
        bt::pad1d_forward(x, /*N=*/1, Cin, L, pad_left, pad_right, mode, scratch);
        in = &scratch;
        Lp = L + pad_left + pad_right;
    }
    bt::conv1d(*in, c.w, c.has_b ? &c.b : nullptr, /*N=*/1, Cin, Lp,
               c.cout, c.k, /*stride=*/1, /*padding=*/0, dilation, groups, y);
    return y;
}

// Causal conv with LEFT-REPLICATE padding (the vocoder cached-conv's offline
// behaviour: clamp the first sample leftward by dilation*(k-1), output len L).
bt::Tensor rconv(const bt::Tensor& x, const ConvW& c, int Cin, int L,
                 int dilation, int groups, bt::Tensor& scratch) {
    return pconv(x, c, Cin, L, dilation, groups,
                 /*pad_left=*/dilation * (c.k - 1), /*pad_right=*/0,
                 /*mode=*/2 /*replicate*/, scratch);
}

// ─── text-encoder helpers (Glow-TTS relative-position attention) ─────────────

// Non-owning [rows x cols] view at a float offset into a channel-major tensor.
bt::Tensor sub_view(const bt::Tensor& t, int off_elems, int rows, int cols) {
    return bt::Tensor::view(t.device, static_cast<float*>(t.data) + off_elems,
                            rows, cols, t.dtype);
}

// Flat (1 x size) view — the NCHW/pad/slice ops require X shaped (N, C*H*W),
// so a buffer carrying matrix (rows, cols) metadata is flattened before they
// reinterpret it via explicit dim args.
bt::Tensor flat(const bt::Tensor& t) {
    return bt::Tensor::view(t.device, t.data, 1, t.rows * t.cols, t.dtype);
}

// Owning copy reshaped to (r, c) — brotensor has no free reshape, and matmul /
// add_inplace read (rows, cols), so a (1, r*c) op result is re-tagged here.
bt::Tensor reshape_owned(const bt::Tensor& t, int r, int c) {
    return bt::Tensor::view(t.device, t.data, r, c, t.dtype).clone();
}

// Transpose a [R x C] matrix to [C x R] (device op, no host roundtrip).
bt::Tensor transpose2d(const bt::Tensor& x, int R, int C) {
    bt::Tensor y;
    bt::nchw_to_sequence(flat(x), /*N=*/1, /*C=*/R, /*H=*/1, /*W=*/C, y);  // -> [C,R]
    return y;
}

// In-place row softmax over a [R x Cn] matrix (each row independently). One
// batched kernel launch over all R rows — not R per-row softmax_forward calls
// (the attention score matrices dominate the launch count otherwise).
void softmax_rows(bt::Tensor& m, int R, int Cn) {
    bt::softmax_rows_forward(m, m, R, Cn);
}

// Glow-TTS window-sliced relative embeddings: emb [2w+1, kc] -> [2T-1, kc].
bt::Tensor get_rel_emb(const bt::Tensor& emb, int T, int window) {
    const int kc = emb.cols;
    const int M  = 2 * window + 1;
    const int pad_len     = std::max(T - (window + 1), 0);
    const int slice_start = std::max((window + 1) - T, 0);
    bt::Tensor padded;
    int rows;
    if (pad_len > 0) {
        bt::pad2d_forward(flat(emb), 1, 1, M, kc, pad_len, pad_len, 0, 0, /*zero=*/0, padded);
        rows = M + 2 * pad_len;
    } else {
        padded = emb;
        rows = M;
    }
    bt::Tensor out;
    bt::slice2d_forward(flat(padded), 1, 1, rows, kc, slice_start, 0, 2 * T - 1, kc, out);
    return reshape_owned(out, 2 * T - 1, kc);
}

// _relative_position_to_absolute_position: [T, 2T-1] -> [T, T] (pad/reshape/slice).
bt::Tensor rel_to_abs(const bt::Tensor& x, int T) {
    bt::Tensor t1, t2, out;
    bt::pad2d_forward(flat(x), 1, 1, T, 2 * T - 1, 0, 0, 0, 1, /*zero=*/0, t1);  // [T, 2T]
    bt::pad2d_forward(flat(t1), 1, 1, 1, T * 2 * T, 0, 0, 0, T - 1, 0, t2);      // [1, 2T^2+T-1]
    bt::slice2d_forward(flat(t2), 1, 1, T + 1, 2 * T - 1, 0, T - 1, T, T, out);  // [T, T]
    return reshape_owned(out, T, T);
}

// _absolute_position_to_relative_position: [T, T] -> [T, 2T-1].
bt::Tensor abs_to_rel(const bt::Tensor& x, int T) {
    bt::Tensor t1, t2, out;
    bt::pad2d_forward(flat(x), 1, 1, T, T, 0, 0, 0, T - 1, /*zero=*/0, t1);      // [T, 2T-1]
    bt::pad2d_forward(flat(t1), 1, 1, 1, T * (2 * T - 1), 0, 0, T, 0, 0, t2);    // [1, 2T^2]
    bt::slice2d_forward(flat(t2), 1, 1, T, 2 * T, 0, 1, T, 2 * T - 1, out);      // [T, 2T-1]
    return reshape_owned(out, T, 2 * T - 1);
}

// Channel-wise LayerNorm of a [C x T] channel-major tensor (transpose, LN over
// C, transpose back). eps matches the upstream LayerNormalization (1e-6).
bt::Tensor layernorm_chan(const bt::Tensor& x, const bt::Tensor& g,
                          const bt::Tensor& b, int C, int T) {
    bt::Tensor seq, seqn, out;
    bt::nchw_to_sequence(flat(x), 1, C, 1, T, seq);                 // [T, C]
    bt::layernorm_forward_inference_batched(seq, g, b, seqn, 1.0e-6f);
    bt::sequence_to_nchw(flat(seqn), 1, C, 1, T, out);             // [C, T]
    return out;
}

// One ConvNeXt-1D block over [C,L] channel-major: depthwise conv (symmetric
// edge pad = 2*dilation each side) -> channel-wise LayerNorm -> 1x1 pwconv1 ->
// GELU -> 1x1 pwconv2 (LayerScale gamma folded into its weight/bias) -> residual.
// Shared by the text encoder, the duration predictor, and the flow field.
bt::Tensor convnext_block(const bt::Tensor& h, const ConvNeXtBlock& blk,
                          int C, int L, int dil, bt::Tensor& scratch) {
    const int pad = 2 * dil;
    bt::Tensor dwy = pconv(h, blk.dw, C, L, dil, C, pad, pad, /*edge=*/2, scratch);
    bt::Tensor seq, seqn, normed;
    bt::nchw_to_sequence(dwy, 1, C, 1, L, seq);
    bt::layernorm_forward_inference_batched(seq, blk.ln_g, blk.ln_b, seqn, 1.0e-6f);
    bt::sequence_to_nchw(seqn, 1, C, 1, L, normed);
    bt::Tensor a = pconv(normed, blk.pw1, C, L, 1, 1, 0, 0, 0, scratch);
    bt::Tensor ga; bt::gelu_exact_forward(a, ga);
    bt::Tensor y = pconv(ga, blk.pw2, blk.pw1.cout, L, 1, 1, 0, 0, 0, scratch);
    bt::add_inplace(y, h);
    return y;
}

// Text-encoder architecture constants (fixed for supertonic-3, from tts.json).
constexpr int kTeC      = 256;  // hidden channels
constexpr int kTeHeads  = 4;    // attn heads
constexpr int kTeKc     = 64;   // per-head channels (kTeC / kTeHeads)
constexpr int kTeWindow = 4;    // relative-attention window (emb_rel rows = 9)
constexpr int kNStyle   = 50;   // style tokens
constexpr int kSpteHeads = 2;   // style cross-attn heads
constexpr int kSpteHd    = 128; // per-head channels (kTeC / kSpteHeads)
constexpr float kSpteScale = 16.0f;  // QK divisor (upstream Div constant)

// Duration-predictor sentence-encoder constants (from duration_predictor.onnx).
constexpr int kDpC      = 64;   // hidden channels
constexpr int kDpHeads  = 2;    // attn heads
constexpr int kDpKc     = 32;   // per-head channels (kDpC / kDpHeads)
constexpr int kDpWindow = 4;    // relative-attention window (emb_rel rows = 9)
constexpr int kDpStyle  = 128;  // flattened style_dp (8*16)
constexpr int kDpHidden = 128;  // predictor hidden width

// One relative-position attention sub-layer: x [C,T] -> conv_o(attn) [C,T].
// Dims are passed in so the same body serves the text encoder (C=256, 4 heads,
// kc=64) and the duration predictor's sentence encoder (C=64, 2 heads, kc=32).
bt::Tensor rel_attention(const bt::Tensor& x, const AttnLayer& L, int T,
                         int C, int nh, int kc, int window) {
    bt::Tensor scr;
    bt::Tensor q = pconv(x, L.conv_q, C, T, 1, 1, 0, 0, 0, scr);  // [C,T]
    bt::Tensor k = pconv(x, L.conv_k, C, T, 1, 1, 0, 0, 0, scr);
    bt::Tensor v = pconv(x, L.conv_v, C, T, 1, 1, 0, 0, 0, scr);
    const float inv = 1.0f / std::sqrt(static_cast<float>(kc));

    const bt::Tensor key_rel  = get_rel_emb(L.emb_rel_k, T, window);  // [2T-1,kc]
    const bt::Tensor val_rel  = get_rel_emb(L.emb_rel_v, T, window);
    const bt::Tensor key_relT = transpose2d(key_rel, 2 * T - 1, kc);     // [kc,2T-1]

    std::vector<bt::Tensor> heads(nh);
    std::vector<const bt::Tensor*> hp(nh);
    for (int h = 0; h < nh; ++h) {
        const bt::Tensor qh = sub_view(q, h * kc * T, kc, T);  // [kc,T]
        const bt::Tensor kh = sub_view(k, h * kc * T, kc, T);
        const bt::Tensor vh = sub_view(v, h * kc * T, kc, T);
        bt::Tensor qhT = transpose2d(qh, kc, T);               // [T,kc]
        bt::scale_inplace(qhT, inv);

        bt::Tensor scores;     bt::matmul(qhT, kh, scores);       // [T,T]
        bt::Tensor rel_logits; bt::matmul(qhT, key_relT, rel_logits);  // [T,2T-1]
        bt::Tensor local = rel_to_abs(rel_logits, T);            // [T,T]
        bt::add_inplace(scores, local);
        softmax_rows(scores, T, T);

        bt::Tensor vhT = transpose2d(vh, kc, T);                 // [T,kc]
        bt::Tensor outh; bt::matmul(scores, vhT, outh);          // [T,kc]
        bt::Tensor relw  = abs_to_rel(scores, T);                // [T,2T-1]
        bt::Tensor outr; bt::matmul(relw, val_rel, outr);        // [T,kc]
        bt::add_inplace(outh, outr);

        heads[h] = transpose2d(outh, T, kc);                     // [kc,T]
        hp[h] = &heads[h];
    }
    bt::Tensor y; bt::concat_rows(hp, y);                        // [C,T] channel-major
    bt::Tensor scr2;
    return pconv(flat(y), L.conv_o, C, T, 1, 1, 0, 0, 0, scr2);  // [C,T]
}

// One tanh-gated style cross-attention: query [Cq,Lq], key source [Ck,S], value
// source [Ck,S] -> out_fc(MHA) [Cout,Lq] (tanh on keys, scale 1/16, softmax over
// S style tokens). All channel dims are read off the loaded 1x1-conv
// projections, so this serves both the text encoder (256->256->256) and the
// flow field's style blocks (512->256->512). kSpteHeads heads.
bt::Tensor style_attention(const bt::Tensor& query, const bt::Tensor& key_src,
                           const bt::Tensor& val_src, const StyleAttn& A,
                           int Lq, int S) {
    const int nh    = kSpteHeads;     // 2
    const int Cq    = A.wq.cin_pg;    // query input channels
    const int inner = A.wq.cout;      // q/k/v projection width
    const int hd    = inner / nh;     // per-head channels
    const int Ck    = A.wk.cin_pg;    // key/value input channels
    bt::Tensor scr;
    bt::Tensor q = pconv(query,   A.wq, Cq, Lq, 1, 1, 0, 0, 0, scr);  // [inner,Lq]
    bt::Tensor k = pconv(key_src, A.wk, Ck, S,  1, 1, 0, 0, 0, scr);  // [inner,S]
    bt::Tensor v = pconv(val_src, A.wv, Ck, S,  1, 1, 0, 0, 0, scr);  // [inner,S]
    const float inv = 1.0f / kSpteScale;

    std::vector<bt::Tensor> heads(nh);
    std::vector<const bt::Tensor*> hp(nh);
    for (int h = 0; h < nh; ++h) {
        const bt::Tensor qh = sub_view(q, h * hd * Lq, hd, Lq);  // [hd,Lq]
        const bt::Tensor kh = sub_view(k, h * hd * S,  hd, S);   // [hd,S]
        const bt::Tensor vh = sub_view(v, h * hd * S,  hd, S);   // [hd,S]
        bt::Tensor qhT = transpose2d(qh, hd, Lq);                // [Lq,hd]
        bt::Tensor ktanh; bt::tanh_forward(kh, ktanh);           // [hd,S]
        bt::Tensor scores; bt::matmul(qhT, ktanh, scores);       // [Lq,S]
        bt::scale_inplace(scores, inv);
        softmax_rows(scores, Lq, S);

        bt::Tensor vhT = transpose2d(vh, hd, S);                 // [S,hd]
        bt::Tensor outh; bt::matmul(scores, vhT, outh);          // [Lq,hd]
        heads[h] = transpose2d(outh, Lq, hd);                    // [hd,Lq]
        hp[h] = &heads[h];
    }
    bt::Tensor cat; bt::concat_rows(hp, cat);                    // [inner,Lq]
    bt::Tensor scr2;
    return pconv(flat(cat), A.wo, inner, Lq, 1, 1, 0, 0, 0, scr2);  // [Cout,Lq]
}

// ─── flow-field RoPE text cross-attention ────────────────────────────────────

constexpr int kVeC      = 512;   // field hidden channels
constexpr int kVeHeads  = 8;     // text-attn heads
constexpr int kVeHd     = 64;    // per-head channels
constexpr int kVeHalf   = 32;    // RoPE rotates the 32+32 split halves
constexpr float kVeAttnScale = 16.0f;  // QK divisor (both field attn types)

// Per-position RoPE cos/sin tables [half x len] from theta[half]; the angle for
// position p is (p/len)*theta[f] — positions are normalised by sequence length,
// so query (len L) and key (len T) align on a shared [0,1) scale.
void rope_tables(const std::vector<float>& theta, int len, bt::Device dev,
                 bt::Tensor& cosm, bt::Tensor& sinm) {
    const int n = static_cast<int>(theta.size());
    std::vector<float> c(static_cast<std::size_t>(n) * len), s(c.size());
    for (int f = 0; f < n; ++f)
        for (int p = 0; p < len; ++p) {
            const float a = (len > 0 ? static_cast<float>(p) / len : 0.0f) * theta[f];
            c[static_cast<std::size_t>(f) * len + p] = std::cos(a);
            s[static_cast<std::size_t>(f) * len + p] = std::sin(a);
        }
    cosm = bt::Tensor::from_host_on(dev, c.data(), n, len);
    sinm = bt::Tensor::from_host_on(dev, s.data(), n, len);
}

// RoPE on one head [2*half x len] channel-major: rotate the split halves —
// out[0:half] = x1*cos - x2*sin, out[half:] = x1*sin + x2*cos.
bt::Tensor rope_apply(const bt::Tensor& xh, int half, int len,
                      const bt::Tensor& cosm, const bt::Tensor& sinm) {
    const bt::Tensor x1 = reshape_owned(sub_view(xh, 0, half, len), half, len);
    const bt::Tensor x2 = reshape_owned(sub_view(xh, half * len, half, len), half, len);
    bt::Tensor a = x1.clone(); bt::mul_inplace(a, cosm);   // x1*cos
    bt::Tensor b = x2.clone(); bt::mul_inplace(b, sinm);   // x2*sin
    bt::axpby_inplace(a, b, 1.0f, -1.0f);                  // x1*cos - x2*sin
    bt::Tensor c = x1.clone(); bt::mul_inplace(c, sinm);   // x1*sin
    bt::Tensor d = x2.clone(); bt::mul_inplace(d, cosm);   // x2*cos
    bt::add_inplace(c, d);                                 // x1*sin + x2*cos
    std::vector<const bt::Tensor*> parts = {&a, &c};
    bt::Tensor out; bt::concat_rows(parts, out);           // flat (2*half*len, 1)
    // concat_rows yields a flat buffer; its layout is the channel-major
    // [2*half, len] we want, so re-tag the shape for the downstream matmul.
    return reshape_owned(out, 2 * half, len);
}

// Prebuilt RoPE cos/sin tables for one (L, T) pair, shared by every text-attn
// block (they all use the same theta) and by every flow step (L, T fixed for an
// utterance). Hoisted out of rope_attention so the per-step field body carries
// no host->device upload — a hard requirement for CUDA-graph capture, and a win
// regardless (the tables were rebuilt 16x per utterance before).
struct RopeTables { bt::Tensor cosL, sinL, cosT, sinT; };

RopeTables build_rope_tables(const std::vector<float>& theta, int L, int T,
                             bt::Device dev) {
    RopeTables rt;
    rope_tables(theta, L, dev, rt.cosL, rt.sinL);
    rope_tables(theta, T, dev, rt.cosT, rt.sinT);
    return rt;
}

// Text cross-attention with RoPE: query=hidden [512,L], key/value=text [256,T].
// 8 heads x 64; query positions normalised by L, key positions by T; scale 1/16.
bt::Tensor rope_attention(const bt::Tensor& h, const bt::Tensor& text_src,
                          const VeRope& A, int L, int T, const RopeTables& rt) {
    const int nh = kVeHeads, hd = kVeHd, half = kVeHalf, Ck = A.conv_k.cin_pg;
    bt::Tensor scr;
    bt::Tensor q = pconv(h,        A.conv_q, kVeC, L, 1, 1, 0, 0, 0, scr);  // [512,L]
    bt::Tensor k = pconv(text_src, A.conv_k, Ck,   T, 1, 1, 0, 0, 0, scr);  // [512,T]
    bt::Tensor v = pconv(text_src, A.conv_v, Ck,   T, 1, 1, 0, 0, 0, scr);  // [512,T]
    const float inv = 1.0f / kVeAttnScale;
    std::vector<bt::Tensor> heads(nh);
    std::vector<const bt::Tensor*> hp(nh);
    for (int hh = 0; hh < nh; ++hh) {
        const bt::Tensor qh = sub_view(q, hh * hd * L, hd, L);  // [64,L]
        const bt::Tensor kh = sub_view(k, hh * hd * T, hd, T);  // [64,T]
        const bt::Tensor vh = sub_view(v, hh * hd * T, hd, T);  // [64,T]
        bt::Tensor qr = rope_apply(qh, half, L, rt.cosL, rt.sinL);  // [64,L]
        bt::Tensor kr = rope_apply(kh, half, T, rt.cosT, rt.sinT);  // [64,T]
        bt::Tensor qrT = transpose2d(qr, hd, L);                // [L,64]
        bt::Tensor scores; bt::matmul(qrT, kr, scores);         // [L,T]
        bt::scale_inplace(scores, inv);
        softmax_rows(scores, L, T);
        bt::Tensor vhT = transpose2d(vh, hd, T);                // [T,64]
        bt::Tensor outh; bt::matmul(scores, vhT, outh);         // [L,64]
        heads[hh] = transpose2d(outh, L, hd);                   // [64,L]
        hp[hh] = &heads[hh];
    }
    bt::Tensor cat; bt::concat_rows(hp, cat);                   // [512,L]
    bt::Tensor scr2;
    return pconv(flat(cat), A.conv_o, kVeC, L, 1, 1, 0, 0, 0, scr2);  // [512,L]
}

// ─── UnicodeProcessor frontend (mirrors py/helper.py UnicodeProcessor) ─────────

// Decode UTF-8 bytes to a codepoint sequence; malformed leads -> U+FFFD.
std::u32string utf8_to_u32(const std::string& s) {
    std::u32string out;
    const std::size_t n = s.size();
    for (std::size_t i = 0; i < n;) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        char32_t cp; int extra;
        if (c < 0x80)            { cp = c;         extra = 0; }
        else if ((c >> 5) == 0x6){ cp = c & 0x1F; extra = 1; }
        else if ((c >> 4) == 0xE){ cp = c & 0x0F; extra = 2; }
        else if ((c >> 3) == 0x1E){ cp = c & 0x07; extra = 3; }
        else                     { cp = 0xFFFD;   extra = 0; }
        ++i;
        for (int k = 0; k < extra && i < n; ++k, ++i)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i]) & 0x3F);
        out.push_back(cp);
    }
    return out;
}

bool is_emoji_cp(char32_t c) {
    return (c >= 0x1F600 && c <= 0x1F64F) || (c >= 0x1F300 && c <= 0x1F5FF) ||
           (c >= 0x1F680 && c <= 0x1F6FF) || (c >= 0x1F700 && c <= 0x1F77F) ||
           (c >= 0x1F780 && c <= 0x1F7FF) || (c >= 0x1F800 && c <= 0x1F8FF) ||
           (c >= 0x1F900 && c <= 0x1F9FF) || (c >= 0x1FA00 && c <= 0x1FA6F) ||
           (c >= 0x1FA70 && c <= 0x1FAFF) || (c >= 0x2600  && c <= 0x26FF)  ||
           (c >= 0x2700  && c <= 0x27BF)  || (c >= 0x1F1E6 && c <= 0x1F1FF);
}

// The final-character set that suppresses the auto-appended period (the upstream
// regex [.!?;:,'"')\]}…。」』】〉》›»]).
bool ends_with_punct(char32_t c) {
    switch (c) {
        case U'.': case U'!': case U'?': case U';': case U':': case U',':
        case U'\'': case U'"': case U')': case U']': case U'}':
        case 0x2026: case 0x3002: case 0x300D: case 0x300F: case 0x3011:
        case 0x3009: case 0x300B: case 0x203A: case 0x00BB:
            return true;
        default: return false;
    }
}

void replace_all(std::u32string& s, const std::u32string& from,
                 const std::u32string& to) {
    if (from.empty()) return;
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::u32string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// NFKD compatibility decomposition over the BMP. The decomposition table is
// generated offline from the Unicode database (supertonic-convert/gen_nfkd.py);
// Hangul syllables decompose algorithmically; a canonical-combining-class pass
// reorders marks. ASCII passes through unchanged.
#include "supertonic_nfkd.inc"  // kNfkdN/kCccN + kNfkd*/kCcc* tables (BMP)

// Canonical combining class of `c` (0 for starters / unlisted).
int nfkd_ccc(char32_t c) {
    if (c > 0xFFFF) return 0;
    int lo = 0, hi = kCccN;
    while (lo < hi) { const int m = (lo + hi) / 2;
                      if (kCccKey[m] < c) lo = m + 1; else hi = m; }
    return (lo < kCccN && kCccKey[lo] == c) ? kCccVal[lo] : 0;
}

// Append the (already-recursed) NFKD decomposition of one codepoint.
void nfkd_decompose_one(char32_t c, std::u32string& out) {
    constexpr char32_t kSBase = 0xAC00;
    constexpr int kSCount = 11172, kLBase = 0x1100, kVBase = 0x1161,
                  kTBase = 0x11A7, kTCount = 28, kNCount = 21 * 28;
    if (c >= kSBase && c < kSBase + static_cast<char32_t>(kSCount)) {  // Hangul
        const int si = static_cast<int>(c - kSBase);
        out.push_back(static_cast<char32_t>(kLBase + si / kNCount));
        out.push_back(static_cast<char32_t>(kVBase + (si % kNCount) / kTCount));
        if (const int ti = si % kTCount) out.push_back(static_cast<char32_t>(kTBase + ti));
        return;
    }
    if (c <= 0xFFFF) {
        int lo = 0, hi = kNfkdN;
        while (lo < hi) { const int m = (lo + hi) / 2;
                          if (kNfkdKey[m] < c) lo = m + 1; else hi = m; }
        if (lo < kNfkdN && kNfkdKey[lo] == c) {
            for (int i = kNfkdOff[lo]; i < kNfkdOff[lo + 1]; ++i)
                out.push_back(static_cast<char32_t>(kNfkdPool[i]));
            return;
        }
    }
    out.push_back(c);
}

std::u32string nfkd(const std::u32string& in) {
    std::u32string d; d.reserve(in.size());
    for (char32_t c : in) nfkd_decompose_one(c, d);

    // Canonical ordering: stable-sort each maximal run of combining marks by ccc.
    const std::size_t n = d.size();
    for (std::size_t i = 0; i < n;) {
        if (nfkd_ccc(d[i]) == 0) { ++i; continue; }
        std::size_t j = i;
        while (j < n && nfkd_ccc(d[j]) != 0) ++j;
        for (std::size_t a = i + 1; a < j; ++a) {        // stable insertion sort
            const char32_t v = d[a]; const int cv = nfkd_ccc(v);
            std::size_t b = a;
            while (b > i && nfkd_ccc(d[b - 1]) > cv) { d[b] = d[b - 1]; --b; }
            d[b] = v;
        }
        i = j;
    }
    return d;
}

// The full _preprocess_text pipeline, codepoint-exact with the upstream.
std::u32string preprocess_text(const std::string& utf8, const std::string& lang) {
    static const char* kLangs[] = {
        "en","ko","ja","ar","bg","cs","da","de","el","es","et","fi","fr","hi",
        "hr","hu","id","it","lt","lv","nl","pl","pt","ro","ru","sk","sl","sv",
        "tr","uk","vi","na"};
    bool ok = false;
    for (const char* l : kLangs) if (lang == l) { ok = true; break; }
    if (!ok) fail("invalid language: " + lang);

    std::u32string t = nfkd(utf8_to_u32(utf8));

    // remove emoji
    { std::u32string o; o.reserve(t.size());
      for (char32_t c : t) if (!is_emoji_cp(c)) o.push_back(c);
      t.swap(o); }

    // single-codepoint dash/quote/symbol replacements, then special-symbol strip
    { std::u32string o; o.reserve(t.size());
      for (char32_t c : t) {
        switch (c) {
            case 0x2013: case 0x2011: case 0x2014: o.push_back(U'-'); break;
            case U'_':                              o.push_back(U' '); break;
            case 0x201C: case 0x201D:               o.push_back(U'"'); break;
            case 0x2018: case 0x2019: case 0x00B4: case U'`':
                                                    o.push_back(U'\''); break;
            case U'[': case U']': case U'|': case U'/': case U'#':
            case 0x2192: case 0x2190:               o.push_back(U' '); break;
            case 0x2665: case 0x2606: case 0x2661: case 0x00A9: case U'\\':
                break;  // remove ♥ ☆ ♡ © backslash
            default:                                o.push_back(c);
        }
      }
      t.swap(o); }

    // known-expression replacements
    replace_all(t, U"@",      U" at ");
    replace_all(t, U"e.g.,",  U"for example, ");
    replace_all(t, U"i.e.,",  U"that is, ");

    // tighten spacing before punctuation
    replace_all(t, U" ,", U",");
    replace_all(t, U" .", U".");
    replace_all(t, U" !", U"!");
    replace_all(t, U" ?", U"?");
    replace_all(t, U" ;", U";");
    replace_all(t, U" :", U":");
    replace_all(t, U" '", U"'");

    // collapse duplicate quotes
    while (t.find(U"\"\"") != std::u32string::npos) replace_all(t, U"\"\"", U"\"");
    while (t.find(U"''")   != std::u32string::npos) replace_all(t, U"''",   U"'");

    // collapse whitespace runs to a single space, then strip ends
    { std::u32string o; o.reserve(t.size()); bool sp = false;
      for (char32_t c : t) {
        const bool ws = (c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' ||
                         c == U'\f' || c == 0x0B);
        if (ws) { sp = true; continue; }
        if (sp && !o.empty()) o.push_back(U' ');
        sp = false; o.push_back(c);
      }
      t.swap(o); }

    // ensure a sentence-final punctuation
    if (t.empty() || !ends_with_punct(t.back())) t.push_back(U'.');

    // language tag wrap
    std::u32string lt(lang.begin(), lang.end());
    return U"<" + lt + U">" + t + U"</" + lt + U">";
}

// ─── sentence splitting for long-form synthesis ──────────────────────────────

// Encode a codepoint sequence back to UTF-8 (inverse of utf8_to_u32) — each
// split sentence is handed back to synthesize() as a UTF-8 string.
std::string u32_to_utf8(const std::u32string& s) {
    std::string out;
    for (char32_t c : s) {
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
        } else if (c < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (c >> 6)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else if (c < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (c >> 12)));
            out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (c >> 18)));
            out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
    }
    return out;
}

bool is_space_cp(char32_t c) {
    return c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == U'\f' ||
           c == 0x0B || c == 0xA0 /*nbsp*/ || c == 0x3000 /*ideographic space*/;
}
bool is_digit_cp(char32_t c) { return c >= U'0' && c <= U'9'; }
bool is_alpha_cp(char32_t c) {
    return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z');
}

// The lowercased ASCII-alpha run ending just before `i` (the punctuation), used
// to suppress a sentence split after a known abbreviation or a single-letter
// initial. Empty if the char before `i` is not alpha.
std::u32string preceding_word(const std::u32string& s, std::size_t i) {
    std::size_t b = i;
    while (b > 0 && is_alpha_cp(s[b - 1])) --b;
    std::u32string w = s.substr(b, i - b);
    for (char32_t& ch : w) if (ch >= U'A' && ch <= U'Z') ch = ch - U'A' + U'a';
    return w;
}

// A '.' right after one of these (case-insensitive) is an abbreviation, not a
// sentence end — splitting there would strand a tiny "Dr." / "etc." chunk. The
// list errs toward under-splitting (a slightly longer chunk is harmless; the
// flow field handles it and max_chars still bounds runaways).
bool is_abbrev(const std::u32string& w) {
    if (w.size() == 1) return true;  // single-letter initial: "J. R. R. Tolkien"
    static const std::u32string kAbbrev[] = {
        U"mr", U"mrs", U"ms", U"dr", U"prof", U"st", U"sr", U"jr", U"vs", U"etc",
        U"inc", U"ltd", U"co", U"corp", U"no", U"fig", U"al", U"dept", U"gov",
        U"sen", U"rep", U"gen", U"col", U"capt", U"lt", U"sgt", U"rev", U"hon",
        U"approx", U"e.g", U"i.e", U"a.m", U"p.m", U"u.s"};
    for (const std::u32string& a : kAbbrev) if (w == a) return true;
    return false;
}
bool is_sentence_end_cp(char32_t c) {
    return c == U'.' || c == U'!' || c == U'?' || c == 0x2026 /*…*/ ||
           c == 0x3002 /*。*/ || c == 0xFF01 /*！*/ || c == 0xFF1F /*？*/ ||
           c == 0xFF0E /*．*/;
}
// Closers that ride along after the terminal punctuation (quotes / brackets).
bool is_closer_cp(char32_t c) {
    return is_sentence_end_cp(c) || c == U'"' || c == U'\'' || c == U')' ||
           c == U']' || c == U'}' || c == 0x201D /*”*/ || c == 0x2019 /*’*/ ||
           c == 0x300D /*」*/ || c == 0x300F /*』*/ || c == 0x3011 /*】*/;
}
std::u32string trim_u32(const std::u32string& s, std::size_t a, std::size_t b) {
    while (a < b && is_space_cp(s[a])) ++a;
    while (b > a && is_space_cp(s[b - 1])) --b;
    return s.substr(a, b - a);
}

// Split text into sentence chunks (UTF-8). A boundary is sentence-final
// punctuation — skipping a '.' between digits (decimal) — followed by whitespace
// or end of text; trailing quote/bracket closers stay with the sentence. Any
// chunk still longer than max_chars codepoints is broken again at the last
// whitespace before the limit (hard cut if a single token is longer), so no
// chunk drives the flow field to an unwieldy latent length.
std::vector<std::string> split_sentences_impl(const std::string& text, int max_chars) {
    const std::u32string s = utf8_to_u32(text);
    const std::size_t n = s.size();
    std::vector<std::u32string> raw;
    std::size_t start = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!is_sentence_end_cp(s[i])) continue;
        if (s[i] == U'.' && i > 0 && i + 1 < n &&
            is_digit_cp(s[i - 1]) && is_digit_cp(s[i + 1]))
            continue;  // decimal point — not a sentence end
        if (s[i] == U'.' && i > 0 && is_alpha_cp(s[i - 1]) &&
            is_abbrev(preceding_word(s, i)))
            continue;  // abbreviation / initial — not a sentence end
        std::size_t j = i + 1;
        while (j < n && is_closer_cp(s[j])) ++j;  // absorb e.g. ?!").  runs
        if (j >= n || is_space_cp(s[j])) {
            raw.push_back(s.substr(start, j - start));
            while (j < n && is_space_cp(s[j])) ++j;  // skip the gap
            start = j;
            i = (j > 0) ? j - 1 : 0;
        }
    }
    if (start < n) raw.push_back(s.substr(start, n - start));

    std::vector<std::string> out;
    for (const std::u32string& c0 : raw) {
        std::u32string c = trim_u32(c0, 0, c0.size());
        while (static_cast<int>(c.size()) > max_chars) {
            std::size_t cut = static_cast<std::size_t>(max_chars);
            while (cut > 0 && !is_space_cp(c[cut])) --cut;  // back up to a space
            if (cut == 0) cut = static_cast<std::size_t>(max_chars);  // no space: hard cut
            std::u32string head = trim_u32(c, 0, cut);
            if (!head.empty()) out.push_back(u32_to_utf8(head));
            std::size_t rest = cut;
            while (rest < c.size() && is_space_cp(c[rest])) ++rest;
            c = trim_u32(c, rest, c.size());
        }
        if (!c.empty()) out.push_back(u32_to_utf8(c));
    }
    return out;
}

}  // namespace

// ─── Impl ────────────────────────────────────────────────────────────────────

struct Supertonic::Impl {
    SupertonicConfig cfg;
    bt::Device       dev = bt::Device::CPU;
    bool             ready = false;

    // Vocoder (autoencoder decoder) weights.
    float                      inv_scale = 1.0f;   // 1 / ttl.normalizer.scale
    std::vector<float>         latent_mean, latent_std;  // (latent_dim)
    ConvW                      conv_in;            // 24 -> hidden, kernel 7
    std::vector<ConvNeXtBlock> blocks;             // 10 dilated ConvNeXt blocks
    bt::Tensor                 bn_g, bn_b, bn_mean, bn_var;  // final BatchNorm
    ConvW                      bn_scale_conv;      // depthwise 1x1 = gamma/sqrt(var+eps);
                                                   // the frozen BN-inference input-grad adjoint
    ConvW                      head1;              // hidden -> 2*hidden, kernel 3
    float                      prelu_slope = 0.0f;
    ConvW                      head2;              // 2*hidden -> base_chunk, kernel 1

    // Text encoder (text_ids + TTL style -> text_emb). Loaded only if
    // text_encoder.safetensors is present alongside the vocoder.
    bool                       te_ready = false;
    std::vector<float>         te_char_emb;        // [8322 x 256] row-major (host gather)
    int                        te_vocab = 0;
    std::vector<ConvNeXtBlock> te_blocks;          // 6 ConvNeXt (ksz 5, dilated)
    std::vector<AttnLayer>     te_attn;            // 4 relative-position attn
    StyleAttn                  spte1, spte2;       // style cross-attn x2
    bt::Tensor                 spte_norm_g, spte_norm_b;  // final LayerNorm
    bt::Tensor                 te_style_key;       // [256 x 50] channel-major

    // Duration predictor (text_ids + style_dp -> scalar total duration). Loaded
    // only if duration_predictor.safetensors is present alongside the vocoder.
    bool                       dp_ready = false;
    std::vector<float>         dp_char_emb;        // [vocab x 64] row-major (host)
    int                        dp_vocab = 0;
    std::vector<float>         dp_sentence_token;  // [64] prepended sequence token
    std::vector<ConvNeXtBlock> dp_blocks;          // 6 ConvNeXt (ksz 5, dil 1)
    std::vector<AttnLayer>     dp_attn;            // 2 relative-position attn
    ConvW                      dp_proj;            // proj_out 1x1 (no bias)
    Gemm                       dp_pred0, dp_pred1; // predictor MLP (Gemm/PReLU/Gemm)
    float                      dp_prelu = 0.0f;    // predictor PReLU slope

    // Flow-matching vector estimator (the DiT field). Optional until present.
    bool                       ve_ready = false;
    ConvW                      ve_proj_in, ve_proj_out;  // 144<->512 convs (no bias)
    std::vector<float>         ve_t0w, ve_t0b, ve_t1w, ve_t1b;  // time mlp (host)
    std::vector<float>         ve_sin_freqs;       // [32] sinusoidal frequencies
    std::vector<VeBlock>       ve_blocks;          // 24 main blocks
    std::vector<ConvNeXtBlock> ve_last;            // 4 last_convnext blocks
    bt::Tensor                 ve_style_key_proto; // [256,50] learned style key
    std::vector<float>         ve_text_uncond;     // [256] text_special_token
    bt::Tensor                 ve_style_key_unc;   // [256,50] uncond style key
    bt::Tensor                 ve_style_val_unc;   // [256,50] uncond style value
    float                      ve_guid_cond = 4.0f, ve_guid_uncond = 3.0f;  // CFG

    // UnicodeProcessor frontend: codepoint -> token id. Loaded from
    // unicode_indexer.json (a flat 65536-entry uint16->id table, -1 = OOV).
    bool                       frontend_ready = false;
    std::vector<int>           unicode_index;      // 65536 entries (-1 = OOV)

    void load(const std::string& dir, bt::Device device);
    void load_text_encoder(const fs::path& path);
    void load_duration_predictor(const fs::path& path);
    void load_vector_estimator(const fs::path& path);
    AudioBuffer decode(const float* latent, int channels, int frames) const;
    // The vocoder conv stack over a de-chunked, de-normalised real latent
    // [latent_dim, LF] (conv_in -> 10 ConvNeXt -> BN -> head -> waveform).
    // decode() builds `dn` (de-chunk + de-normalise) then calls this.
    AudioBuffer decode_dn(const std::vector<float>& dn, int LF) const;
    std::vector<float> encode_text(const std::vector<int>& ids,
                                   const std::vector<float>& style) const;
    float predict_duration(const std::vector<int>& ids,
                           const std::vector<float>& style_dp) const;
    bt::Tensor run_field(const bt::Tensor& h0, const bt::Tensor& time_emb,
                         const bt::Tensor& text_src, const bt::Tensor& style_key,
                         const bt::Tensor& style_val, int L, int T,
                         const RopeTables& rt, const bt::Tensor& onesL) const;
    // One classifier-free-guided Euler step done entirely on-device, writing the
    // updated latent back into `noisyT` in place. No host<->device traffic and no
    // host-built tensors inside, so synthesize() can CUDA-graph-capture it and
    // replay it per step. All conditioning (text/style/rope/ones/dt) is prebuilt
    // once per utterance and handed in.
    void field_step(bt::Tensor& noisyT, const bt::Tensor& time_emb,
                    const bt::Tensor& text_cond, const bt::Tensor& text_unc,
                    const bt::Tensor& style_val_cond, const bt::Tensor& onesL,
                    const RopeTables& rt, int L, int T, float dt,
                    float guid_cond, float guid_uncond) const;
    std::vector<float> denoise(const std::vector<float>& noisy, int channels,
                               int frames, const std::vector<float>& text_emb,
                               int text_len, const std::vector<float>& style_ttl,
                               int current_step, int total_step) const;
    std::vector<int> text_to_ids(const std::string& text,
                                 const std::string& lang) const;
    AudioBuffer synthesize(const std::string& text, const std::string& lang,
                           const VoiceStyle& voice, int total_step, float speed,
                           std::uint64_t seed, float guidance) const;
    AudioBuffer synthesize_long(const std::string& text, const std::string& lang,
                                const VoiceStyle& voice, int total_step, float speed,
                                std::uint64_t seed, float gap_seconds,
                                int max_chars, float guidance) const;
};

void Supertonic::Impl::load(const std::string& dir, bt::Device device) {
    dev = device;
    const fs::path root(dir);

    // tts.json carries the autoencoder geometry; fall back to known defaults.
    if (fs::exists(root / "tts.json")) {
        const j::Value tts = j::parse(slurp(root / "tts.json"));
        if (const j::Value* ae = tts.find("ae")) {
            cfg.sample_rate = ae->get_int("sample_rate", cfg.sample_rate);
            cfg.base_chunk  = ae->get_int("base_chunk_size", cfg.base_chunk);
            cfg.latent_dim  = ae->get_int("ldim", cfg.latent_dim);
        }
        if (const j::Value* ttl = tts.find("ttl"))
            cfg.chunk = ttl->get_int("chunk_compress_factor", cfg.chunk);
    }

    const fs::path voc = root / "vocoder.safetensors";
    if (!fs::exists(voc)) fail("missing vocoder.safetensors under " + dir);
    const sf::File f = sf::File::open(voc.string());

    // De-normalisation: latent / scale, then * latent_std + latent_mean.
    const std::vector<float> scale = read_host(f, "tts.ttl.normalizer.scale");
    inv_scale   = (scale.empty() || scale[0] == 0.0f) ? 1.0f : 1.0f / scale[0];
    latent_mean = read_host(f, "tts.ae.latent_mean");   // (1,latent_dim,1)
    latent_std  = read_host(f, "tts.ae.latent_std");
    if (static_cast<int>(latent_mean.size()) != cfg.latent_dim ||
        static_cast<int>(latent_std.size())  != cfg.latent_dim)
        fail("latent_mean/std length != latent_dim");

    // Initial conv: latent_dim -> hidden (ONNX names this initializer onnx::Conv_*).
    conv_in = load_conv(f, "onnx::Conv_1441", "onnx::Conv_1442", dev);

    // 10 dilated ConvNeXt blocks. Dilations repeat 1,2,4 over the first six, then
    // 1 — taken from the depthwise weight's dilation in the upstream graph.
    static const int kDil[10] = {1, 2, 4, 1, 2, 4, 1, 1, 1, 1};
    blocks.clear();
    for (int i = 0; i < 10; ++i) {
        const std::string p = "tts.ae.decoder.convnext." + std::to_string(i);
        ConvNeXtBlock blk;
        blk.dil  = kDil[i];
        blk.dw   = load_conv(f, p + ".dwconv.net.weight", p + ".dwconv.net.bias", dev);
        blk.ln_g = up_vec(f, p + ".norm.norm.weight", blk.dw.cout, dev);
        blk.ln_b = up_vec(f, p + ".norm.norm.bias",   blk.dw.cout, dev);
        blk.pw1  = load_conv(f, p + ".pwconv1.weight", p + ".pwconv1.bias", dev);
        blk.pw2  = load_pwconv2_scaled(f, p, dev);
        blocks.push_back(std::move(blk));
    }

    // Final BatchNorm (inference: uses running stats).
    bn_g    = up_vec(f, "tts.ae.decoder.final_norm.norm.weight",       conv_in.cout, dev);
    bn_b    = up_vec(f, "tts.ae.decoder.final_norm.norm.bias",         conv_in.cout, dev);
    bn_mean = up_vec(f, "tts.ae.decoder.final_norm.norm.running_mean", conv_in.cout, dev);
    bn_var  = up_vec(f, "tts.ae.decoder.final_norm.norm.running_var",  conv_in.cout, dev);

    // Head: conv (kernel 3) -> PReLU (scalar slope) -> conv (kernel 1, no bias).
    head1 = load_conv(f, "tts.ae.decoder.head.layer1.net.weight",
                      "tts.ae.decoder.head.layer1.net.bias", dev);
    const std::vector<float> slope = read_host(f, "onnx::PRelu_1506");
    prelu_slope = slope.empty() ? 0.0f : slope[0];
    head2 = load_conv(f, "tts.ae.decoder.head.layer2.weight", "", dev);

    // Precompute the frozen-BN-inference input-grad adjoint as a depthwise 1x1
    // conv (per-channel scale gamma/sqrt(var+eps)) — lets the on-device encoder
    // reconstruction backward apply it without a per-step host round-trip.
    {
        const int C = conv_in.cout;
        const std::vector<float> gh = bn_g.to_host_vector(), vh = bn_var.to_host_vector();
        std::vector<float> sc(static_cast<std::size_t>(C));
        for (int c = 0; c < C; ++c) sc[c] = gh[c] / std::sqrt(vh[c] + 1.0e-5f);
        bn_scale_conv.cout = C; bn_scale_conv.cin_pg = 1; bn_scale_conv.k = 1;
        bn_scale_conv.has_b = false;
        bn_scale_conv.w = bt::Tensor::from_host_on(dev, sc.data(), C, 1);
    }

    ready = true;

    // Text encoder is optional today — load it when its weights are present.
    const fs::path te = root / "text_encoder.safetensors";
    if (fs::exists(te)) load_text_encoder(te);

    // Duration predictor is likewise optional until synthesize() lands.
    const fs::path dp = root / "duration_predictor.safetensors";
    if (fs::exists(dp)) load_duration_predictor(dp);

    const fs::path ve = root / "vector_estimator.safetensors";
    if (fs::exists(ve)) load_vector_estimator(ve);

    // UnicodeProcessor table: a flat list of 65536 ids (codepoint -> id, -1 OOV).
    const fs::path uix = root / "unicode_indexer.json";
    if (fs::exists(uix)) {
        const j::Value arr = j::parse(slurp(uix));
        const j::Value::Array& a = arr.as_array();
        unicode_index.assign(65536, -1);
        const std::size_t n = std::min<std::size_t>(a.size(), 65536);
        for (std::size_t i = 0; i < n; ++i)
            unicode_index[i] = static_cast<int>(a[i].as_number());
        frontend_ready = true;
    }
}

void Supertonic::Impl::load_text_encoder(const fs::path& path) {
    const sf::File f = sf::File::open(path.string());
    const std::string TE = "tts.ttl.text_encoder.";

    // Char embedding table [vocab x 256], kept host-resident for the gather.
    const sf::TensorView& emb = need(f, TE + "text_embedder.char_embedder.weight");
    if (emb.shape.size() != 2 || emb.shape[1] != kTeC)
        fail("char_embedder weight is not [vocab x 256]");
    te_vocab    = static_cast<int>(emb.shape[0]);
    te_char_emb = read_host(f, TE + "text_embedder.char_embedder.weight");

    // 6 ConvNeXt blocks (ksz 5). gamma folded into pwconv2 (load_pwconv2_scaled).
    te_blocks.clear();
    for (int i = 0; i < 6; ++i) {
        const std::string p = TE + "convnext.convnext." + std::to_string(i);
        ConvNeXtBlock blk;
        blk.dw   = load_conv(f, p + ".dwconv.weight", p + ".dwconv.bias", dev);
        blk.ln_g = up_vec(f, p + ".norm.norm.weight", blk.dw.cout, dev);
        blk.ln_b = up_vec(f, p + ".norm.norm.bias",   blk.dw.cout, dev);
        blk.pw1  = load_conv(f, p + ".pwconv1.weight", p + ".pwconv1.bias", dev);
        blk.pw2  = load_pwconv2_scaled(f, p, dev);
        te_blocks.push_back(std::move(blk));
    }

    // 4 relative-position attention layers.
    te_attn.clear();
    for (int i = 0; i < 4; ++i) {
        const std::string a = TE + "attn_encoder.attn_layers." + std::to_string(i) + ".";
        const std::string n1 = TE + "attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.";
        const std::string n2 = TE + "attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.";
        const std::string ff = TE + "attn_encoder.ffn_layers." + std::to_string(i) + ".";
        AttnLayer L;
        L.conv_q = load_conv(f, a + "conv_q.weight", a + "conv_q.bias", dev);
        L.conv_k = load_conv(f, a + "conv_k.weight", a + "conv_k.bias", dev);
        L.conv_v = load_conv(f, a + "conv_v.weight", a + "conv_v.bias", dev);
        L.conv_o = load_conv(f, a + "conv_o.weight", a + "conv_o.bias", dev);
        L.emb_rel_k = up(f, a + "emb_rel_k", 2 * kTeWindow + 1, kTeKc, dev);
        L.emb_rel_v = up(f, a + "emb_rel_v", 2 * kTeWindow + 1, kTeKc, dev);
        L.n1_g = up_vec(f, n1 + "weight", kTeC, dev);
        L.n1_b = up_vec(f, n1 + "bias",   kTeC, dev);
        L.n2_g = up_vec(f, n2 + "weight", kTeC, dev);
        L.n2_b = up_vec(f, n2 + "bias",   kTeC, dev);
        L.ffn1 = load_conv(f, ff + "conv_1.weight", ff + "conv_1.bias", dev);
        L.ffn2 = load_conv(f, ff + "conv_2.weight", ff + "conv_2.bias", dev);
        te_attn.push_back(std::move(L));
    }

    // Speech-prompted style cross-attention (two layers) + final LayerNorm.
    // Linears load by canonical name (the converter renames the ONNX-export
    // `onnx::MatMul_*` weights to `<scope>.linear.weight`).
    const std::string SP = "tts.ttl.speech_prompted_text_encoder.";
    auto load_spte = [&](const std::string& at) {
        const std::string b = SP + at + ".";
        StyleAttn A;
        A.wq = load_linear_as_conv(f, b + "W_query.linear.weight", b + "W_query.linear.bias", dev);
        A.wk = load_linear_as_conv(f, b + "W_key.linear.weight",   b + "W_key.linear.bias",   dev);
        A.wv = load_linear_as_conv(f, b + "W_value.linear.weight", b + "W_value.linear.bias", dev);
        A.wo = load_linear_as_conv(f, b + "out_fc.linear.weight",  b + "out_fc.linear.bias",  dev);
        return A;
    };
    spte1 = load_spte("attention1");
    spte2 = load_spte("attention2");
    spte_norm_g = up_vec(f, SP + "norm.norm.weight", kTeC, dev);
    spte_norm_b = up_vec(f, SP + "norm.norm.bias",   kTeC, dev);

    // Learned style-key prototype [1,50,256] -> channel-major [256,50].
    const bt::Tensor sk = up(f, "tts.ttl.style_encoder.style_token_layer.style_key",
                             kNStyle, kTeC, dev);             // [50,256]
    te_style_key = transpose2d(sk, kNStyle, kTeC);            // [256,50]

    te_ready = true;
}

std::vector<float> Supertonic::Impl::encode_text(
        const std::vector<int>& ids, const std::vector<float>& style) const {
    if (!te_ready) fail("encode_text() before text-encoder load");
    const int T = static_cast<int>(ids.size());
    if (T <= 0) fail("text_ids is empty");
    if (static_cast<int>(style.size()) != kNStyle * kTeC)
        fail("style_ttl must be 50*256 (token-major)");
    const int C = kTeC;

    // ── embedding: gather rows -> [C, T] channel-major (host) ──
    std::vector<float> emb(static_cast<std::size_t>(C) * T);
    for (int t = 0; t < T; ++t) {
        const int id = ids[t];
        if (id < 0 || id >= te_vocab) fail("text id out of range");
        const float* row = &te_char_emb[static_cast<std::size_t>(id) * C];
        for (int c = 0; c < C; ++c) emb[static_cast<std::size_t>(c) * T + t] = row[c];
    }
    bt::Tensor h = bt::Tensor::from_host_on(dev, emb.data(), 1, C * T);
    bt::Tensor scratch;

    // ── 6 ConvNeXt blocks (ksz 5, symmetric edge pad = 2*dilation each side) ──
    static const int kDil[6] = {1, 1, 2, 2, 4, 4};
    for (int i = 0; i < 6; ++i)
        h = convnext_block(h, te_blocks[i], C, T, kDil[i], scratch);
    const bt::Tensor x0 = h.clone();  // global residual skip (proj_out)

    // ── 4 relative-position attention layers ──
    for (const AttnLayer& L : te_attn) {
        bt::Tensor a = rel_attention(h, L, T, kTeC, kTeHeads, kTeKc, kTeWindow);
        bt::add_inplace(a, h);                        // h + attn
        bt::Tensor x1 = layernorm_chan(a, L.n1_g, L.n1_b, C, T);
        bt::Tensor f1 = pconv(x1, L.ffn1, C, T, 1, 1, 0, 0, 0, scratch);
        bt::Tensor fr; bt::relu_forward(f1, fr);
        bt::Tensor f2 = pconv(fr, L.ffn2, L.ffn1.cout, T, 1, 1, 0, 0, 0, scratch);
        bt::add_inplace(f2, x1);
        h = layernorm_chan(f2, L.n2_g, L.n2_b, C, T);
    }
    bt::add_inplace(h, x0);  // proj_out = attn_encoder_out + convnext_out

    // ── speech-prompted style cross-attention (key=style prototype, value=TTL) ──
    std::vector<float> sv(static_cast<std::size_t>(C) * kNStyle);  // [C,50] channel-major
    for (int s = 0; s < kNStyle; ++s)
        for (int c = 0; c < C; ++c)
            sv[static_cast<std::size_t>(c) * kNStyle + s] = style[static_cast<std::size_t>(s) * C + c];
    const bt::Tensor styleV = bt::Tensor::from_host_on(dev, sv.data(), 1, C * kNStyle);

    const bt::Tensor t0 = h;  // proj_out text [C,T]
    bt::Tensor x1 = style_attention(t0, te_style_key, styleV, spte1, T, kNStyle);
    bt::add_inplace(x1, t0);                            // residual to text
    bt::Tensor x2 = style_attention(x1, te_style_key, styleV, spte2, T, kNStyle);
    bt::add_inplace(x2, t0);                            // residual to text
    bt::Tensor out = layernorm_chan(x2, spte_norm_g, spte_norm_b, C, T);  // [C,T]
    return out.to_host_vector();
}

void Supertonic::Impl::load_duration_predictor(const fs::path& path) {
    const sf::File f = sf::File::open(path.string());
    const std::string SE = "tts.dp.sentence_encoder.";

    // Char embedding [vocab x 64], kept host-resident for the gather.
    const sf::TensorView& emb = need(f, SE + "text_embedder.char_embedder.weight");
    if (emb.shape.size() != 2 || emb.shape[1] != kDpC)
        fail("dp char_embedder weight is not [vocab x 64]");
    dp_vocab    = static_cast<int>(emb.shape[0]);
    dp_char_emb = read_host(f, SE + "text_embedder.char_embedder.weight");

    // Learned sentence token [1,64,1] -> [64], prepended at sequence position 0.
    dp_sentence_token = read_host(f, SE + "sentence_token");
    if (static_cast<int>(dp_sentence_token.size()) != kDpC)
        fail("dp sentence_token is not length 64");

    // 6 ConvNeXt blocks (ksz 5, dilation 1). gamma folded into pwconv2.
    dp_blocks.clear();
    for (int i = 0; i < 6; ++i) {
        const std::string p = SE + "convnext.convnext." + std::to_string(i);
        ConvNeXtBlock blk;
        blk.dw   = load_conv(f, p + ".dwconv.weight", p + ".dwconv.bias", dev);
        blk.ln_g = up_vec(f, p + ".norm.norm.weight", blk.dw.cout, dev);
        blk.ln_b = up_vec(f, p + ".norm.norm.bias",   blk.dw.cout, dev);
        blk.pw1  = load_conv(f, p + ".pwconv1.weight", p + ".pwconv1.bias", dev);
        blk.pw2  = load_pwconv2_scaled(f, p, dev);
        dp_blocks.push_back(std::move(blk));
    }

    // 2 relative-position attention layers (2 heads, kc 32, window 4).
    dp_attn.clear();
    for (int i = 0; i < 2; ++i) {
        const std::string a  = SE + "attn_encoder.attn_layers." + std::to_string(i) + ".";
        const std::string n1 = SE + "attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.";
        const std::string n2 = SE + "attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.";
        const std::string ff = SE + "attn_encoder.ffn_layers." + std::to_string(i) + ".";
        AttnLayer L;
        L.conv_q = load_conv(f, a + "conv_q.weight", a + "conv_q.bias", dev);
        L.conv_k = load_conv(f, a + "conv_k.weight", a + "conv_k.bias", dev);
        L.conv_v = load_conv(f, a + "conv_v.weight", a + "conv_v.bias", dev);
        L.conv_o = load_conv(f, a + "conv_o.weight", a + "conv_o.bias", dev);
        L.emb_rel_k = up(f, a + "emb_rel_k", 2 * kDpWindow + 1, kDpKc, dev);
        L.emb_rel_v = up(f, a + "emb_rel_v", 2 * kDpWindow + 1, kDpKc, dev);
        L.n1_g = up_vec(f, n1 + "weight", kDpC, dev);
        L.n1_b = up_vec(f, n1 + "bias",   kDpC, dev);
        L.n2_g = up_vec(f, n2 + "weight", kDpC, dev);
        L.n2_b = up_vec(f, n2 + "bias",   kDpC, dev);
        L.ffn1 = load_conv(f, ff + "conv_1.weight", ff + "conv_1.bias", dev);
        L.ffn2 = load_conv(f, ff + "conv_2.weight", ff + "conv_2.bias", dev);
        dp_attn.push_back(std::move(L));
    }

    // proj_out 1x1 conv [64,64,1] applied to the pooled sentence token (no bias).
    dp_proj = load_conv(f, SE + "proj_out.net.weight", "", dev);

    // Predictor MLP: Gemm(192->128) -> PReLU -> Gemm(128->1) -> Exp.
    dp_pred0 = load_gemm(f, "tts.dp.predictor.layers.0.weight",
                         "tts.dp.predictor.layers.0.bias", dev);
    dp_pred1 = load_gemm(f, "tts.dp.predictor.layers.1.weight",
                         "tts.dp.predictor.layers.1.bias", dev);
    const std::vector<float> slope = read_host(f, "tts.dp.predictor.activation.weight");
    dp_prelu = slope.empty() ? 0.0f : slope[0];

    dp_ready = true;
}

float Supertonic::Impl::predict_duration(
        const std::vector<int>& ids, const std::vector<float>& style_dp) const {
    if (!dp_ready) fail("predict_duration() before duration-predictor load");
    const int Tt = static_cast<int>(ids.size());
    if (Tt <= 0) fail("text_ids is empty");
    if (static_cast<int>(style_dp.size()) != kDpStyle)
        fail("style_dp must be 8*16 = 128 (token-major)");
    const int C = kDpC;
    const int T = Tt + 1;  // the learned sentence token is prepended at position 0

    // ── embedding: column 0 = sentence_token, columns 1..Tt = char rows ──
    std::vector<float> emb(static_cast<std::size_t>(C) * T);
    for (int c = 0; c < C; ++c)
        emb[static_cast<std::size_t>(c) * T + 0] = dp_sentence_token[c];
    for (int t = 0; t < Tt; ++t) {
        const int id = ids[t];
        if (id < 0 || id >= dp_vocab) fail("text id out of range");
        const float* row = &dp_char_emb[static_cast<std::size_t>(id) * C];
        for (int c = 0; c < C; ++c)
            emb[static_cast<std::size_t>(c) * T + (t + 1)] = row[c];
    }
    bt::Tensor h = bt::Tensor::from_host_on(dev, emb.data(), 1, C * T);
    bt::Tensor scratch;

    // ── 6 ConvNeXt blocks (ksz 5, dilation 1, symmetric edge pad = 2) ──
    for (int i = 0; i < 6; ++i)
        h = convnext_block(h, dp_blocks[i], C, T, 1, scratch);
    const bt::Tensor x0 = h.clone();  // global residual skip (attn_encoder out)

    // ── 2 relative-position attention layers ──
    for (const AttnLayer& L : dp_attn) {
        bt::Tensor a = rel_attention(h, L, T, kDpC, kDpHeads, kDpKc, kDpWindow);
        bt::add_inplace(a, h);
        bt::Tensor x1 = layernorm_chan(a, L.n1_g, L.n1_b, C, T);
        bt::Tensor f1 = pconv(x1, L.ffn1, C, T, 1, 1, 0, 0, 0, scratch);
        bt::Tensor fr; bt::relu_forward(f1, fr);
        bt::Tensor f2 = pconv(fr, L.ffn2, L.ffn1.cout, T, 1, 1, 0, 0, 0, scratch);
        bt::add_inplace(f2, x1);
        h = layernorm_chan(f2, L.n2_g, L.n2_b, C, T);
    }
    bt::add_inplace(h, x0);  // proj input = attn_encoder_out + convnext_out

    // ── pool the sentence token (column 0) -> proj_out conv -> [1 x 64] ──
    bt::Tensor tok;  // [C,1] channel-major (h viewed [C,T], slice col 0)
    bt::slice2d_forward(flat(h), 1, 1, C, T, 0, 0, C, 1, tok);
    bt::Tensor tokm = reshape_owned(tok, C, 1);
    bt::Tensor pooled = pconv(flat(tokm), dp_proj, C, 1, 1, 1, 0, 0, 0, scratch);  // [1,C]

    // ── predictor head: concat(pooled[64], style_dp[128]) -> MLP -> exp ──
    const bt::Tensor p641  = reshape_owned(pooled, C, 1);                  // [64,1]
    const bt::Tensor s1281 = bt::Tensor::from_host_on(dev, style_dp.data(),
                                                      kDpStyle, 1);        // [128,1]
    std::vector<const bt::Tensor*> parts = {&p641, &s1281};
    bt::Tensor cat; bt::concat_rows(parts, cat);                           // [192,1]

    bt::Tensor y0; bt::matmul(flat(cat), dp_pred0.wt, y0);                 // [1,128]
    bt::add_inplace(y0, dp_pred0.b);
    bt::Tensor act; bt::leaky_relu_forward(y0, dp_prelu, act);             // PReLU(scalar)
    bt::Tensor y1; bt::matmul(act, dp_pred1.wt, y1);                       // [1,1]
    bt::add_inplace(y1, dp_pred1.b);
    bt::Tensor dur; bt::exp_forward(y1, dur);
    const std::vector<float> hv = dur.to_host_vector();
    return hv.empty() ? 0.0f : hv[0];
}

void Supertonic::Impl::load_vector_estimator(const fs::path& path) {
    const sf::File f = sf::File::open(path.string());
    const std::string V = "vector_estimator.tts.ttl.vector_field.";

    ve_proj_in  = load_conv(f, V + "proj_in.net.weight",  "", dev);   // 144->512
    ve_proj_out = load_conv(f, V + "proj_out.net.weight", "", dev);   // 512->144

    // Time encoder MLP (kept host-resident; a tiny per-step scalar forward):
    // sinusoidal(t) [64] -> Gemm 64->256 -> Mish -> Gemm 256->64. Gemm weights
    // are [out,in].
    ve_t0w = read_host(f, V + "time_encoder.mlp.0.linear.weight");  // [256,64]
    ve_t0b = read_host(f, V + "time_encoder.mlp.0.linear.bias");    // [256]
    ve_t1w = read_host(f, V + "time_encoder.mlp.2.linear.weight");  // [64,256]
    ve_t1b = read_host(f, V + "time_encoder.mlp.2.linear.bias");    // [64]
    ve_sin_freqs = read_host(
        f, "/vector_estimator/vector_field/time_encoder/sinusoidal/Constant_3_output_0");

    auto load_convnext = [&](const std::string& p) {
        ConvNeXtBlock c;
        c.dw   = load_conv(f, p + ".dwconv.weight", p + ".dwconv.bias", dev);
        c.ln_g = up_vec(f, p + ".norm.norm.weight", c.dw.cout, dev);
        c.ln_b = up_vec(f, p + ".norm.norm.bias",   c.dw.cout, dev);
        c.pw1  = load_conv(f, p + ".pwconv1.weight", p + ".pwconv1.bias", dev);
        c.pw2  = load_pwconv2_scaled(f, p, dev);
        return c;
    };

    // RoPE theta/increments are a single shared parameter (exported once, on the
    // first text-attn block); every text-attn block reuses it. increments are the
    // implicit 0,1,2,... positions, so only theta is needed.
    const std::vector<float> rope_theta = read_host(f, V + "main_blocks.3.attn.theta");

    // 24 main blocks: a period-6 pattern (convnext, film, convnext, text-attn,
    // convnext, style-attn) repeated four times.
    ve_blocks.assign(24, VeBlock{});
    for (int i = 0; i < 24; ++i) {
        const std::string mb = V + "main_blocks." + std::to_string(i) + ".";
        VeBlock blk;
        const int r = i % 6;
        if (r == 0 || r == 2 || r == 4) {
            blk.type = 0;
            // ConvNeXt main-blocks hold a varying number of sub-blocks (4/1/1
            // per period) — load until the next sub-block's weight is absent.
            for (int s = 0;; ++s) {
                const std::string p = mb + "convnext." + std::to_string(s);
                if (!f.find(p + ".dwconv.weight")) break;
                blk.conv.push_back(load_convnext(p));
            }
        } else if (r == 1) {
            blk.type = 1;
            blk.film.w = up(f, mb + "linear.linear.weight", kVeHd, kVeC, dev);  // [64,512]
            blk.film.b = up(f, mb + "linear.linear.bias", 1, kVeC, dev);        // [1,512]
        } else if (r == 3) {
            blk.type = 2;
            const std::string a = mb + "attn.";
            blk.rope.conv_q = load_linear_as_conv(f, a + "W_query.linear.weight", a + "W_query.linear.bias", dev);
            blk.rope.conv_k = load_linear_as_conv(f, a + "W_key.linear.weight",   a + "W_key.linear.bias",   dev);
            blk.rope.conv_v = load_linear_as_conv(f, a + "W_value.linear.weight", a + "W_value.linear.bias", dev);
            blk.rope.conv_o = load_linear_as_conv(f, a + "out_fc.linear.weight",  a + "out_fc.linear.bias",  dev);
            blk.rope.theta  = rope_theta;                                       // shared [32]
            blk.rope.norm_g = up_vec(f, mb + "norm.norm.weight", kVeC, dev);
            blk.rope.norm_b = up_vec(f, mb + "norm.norm.bias",   kVeC, dev);
        } else {  // r == 5
            blk.type = 3;
            const std::string a = mb + "attention.";
            blk.style.attn.wq = load_linear_as_conv(f, a + "W_query.linear.weight", a + "W_query.linear.bias", dev);
            blk.style.attn.wk = load_linear_as_conv(f, a + "W_key.linear.weight",   a + "W_key.linear.bias",   dev);
            blk.style.attn.wv = load_linear_as_conv(f, a + "W_value.linear.weight", a + "W_value.linear.bias", dev);
            blk.style.attn.wo = load_linear_as_conv(f, a + "out_fc.linear.weight",  a + "out_fc.linear.bias",  dev);
            blk.style.norm_g  = up_vec(f, mb + "norm.norm.weight", kVeC, dev);
            blk.style.norm_b  = up_vec(f, mb + "norm.norm.bias",   kVeC, dev);
        }
        ve_blocks[i] = std::move(blk);
    }

    ve_last.clear();
    for (int i = 0; i < 4; ++i)
        ve_last.push_back(load_convnext(V + "last_convnext.convnext." + std::to_string(i)));

    // Learned style-key prototype + the unconditional special tokens, all stored
    // token-major [50,256] -> transposed to channel-major [256,50].
    const bt::Tensor proto = up(f, "/vector_estimator/Expand_output_0", kNStyle, kTeC, dev);
    ve_style_key_proto = transpose2d(proto, kNStyle, kTeC);
    const std::string UM = "vector_estimator.tts.ttl.uncond_masker.";
    ve_text_uncond = read_host(f, UM + "text_special_token");  // [256]
    const bt::Tensor sk = up(f, UM + "style_key_special_token",   kNStyle, kTeC, dev);
    ve_style_key_unc = transpose2d(sk, kNStyle, kTeC);
    const bt::Tensor sv = up(f, UM + "style_value_special_token", kNStyle, kTeC, dev);
    ve_style_val_unc = transpose2d(sv, kNStyle, kTeC);

    ve_ready = true;
}

// One pass of the flow field: hidden h0 [512,L] + time/text/style conditioning
// -> velocity field [144,L]. text_src is [256,T]; style_key/style_val [256,50].
bt::Tensor Supertonic::Impl::run_field(
        const bt::Tensor& h0, const bt::Tensor& time_emb,
        const bt::Tensor& text_src, const bt::Tensor& style_key,
        const bt::Tensor& style_val, int L, int T,
        const RopeTables& rt, const bt::Tensor& onesL) const {
    bt::Tensor h = h0.clone();
    bt::Tensor scratch;
    for (int i = 0; i < 24; ++i) {
        const VeBlock& blk = ve_blocks[i];
        if (blk.type == 0) {
            // Within a ConvNeXt main-block the sub-blocks are dilated 1,2,4,8...
            for (std::size_t s = 0; s < blk.conv.size(); ++s)
                h = convnext_block(h, blk.conv[s], kVeC, L, 1 << s, scratch);
        } else if (blk.type == 1) {
            // FiLM: h += (time_emb @ W + b), broadcast across all L positions.
            bt::Tensor proj; bt::matmul(time_emb, blk.film.w, proj);  // [1,512]
            bt::add_inplace(proj, blk.film.b);
            bt::Tensor pcol = transpose2d(proj, 1, kVeC);             // [512,1]
            bt::Tensor tiled; bt::matmul(pcol, onesL, tiled);         // [512,L]
            bt::add_inplace(h, tiled);
        } else if (blk.type == 2) {
            bt::Tensor a = rope_attention(h, text_src, blk.rope, L, T, rt);
            bt::add_inplace(a, h);                                    // residual
            h = layernorm_chan(a, blk.rope.norm_g, blk.rope.norm_b, kVeC, L);
        } else {  // type 3: style cross-attention
            bt::Tensor a = style_attention(h, style_key, style_val, blk.style.attn, L, kNStyle);
            bt::add_inplace(a, h);                                    // residual
            h = layernorm_chan(a, blk.style.norm_g, blk.style.norm_b, kVeC, L);
        }
    }
    for (const ConvNeXtBlock& c : ve_last)
        h = convnext_block(h, c, kVeC, L, 1, scratch);
    return pconv(flat(h), ve_proj_out, kVeC, L, 1, 1, 0, 0, 0, scratch);  // [144,L]
}

// One CFG Euler step, on-device and host-traffic-free, updating `noisyT` in place
// (noisyT is the running latent, carried [1, 144*L] channel-major). Same math as
// denoise() but with every conditioning input prebuilt: this is what synthesize()
// CUDA-graph-captures and replays per step.
void Supertonic::Impl::field_step(
        bt::Tensor& noisyT, const bt::Tensor& time_emb,
        const bt::Tensor& text_cond, const bt::Tensor& text_unc,
        const bt::Tensor& style_val_cond, const bt::Tensor& onesL,
        const RopeTables& rt, int L, int T, float dt,
        float guid_cond, float guid_uncond) const {
    bt::Tensor scratch;
    const int channels = cfg.latent_dim * cfg.chunk;  // 144
    // proj_in is shared by the conditional + unconditional passes (same noisy).
    bt::Tensor h0 = pconv(noisyT, ve_proj_in, channels, L, 1, 1, 0, 0, 0, scratch);  // [512,L]
    bt::Tensor fc = run_field(h0, time_emb, text_cond, ve_style_key_proto, style_val_cond, L, T, rt, onesL);
    bt::Tensor fu = run_field(h0, time_emb, text_unc,  ve_style_key_unc,   ve_style_val_unc,  L, T, rt, onesL);
    // CFG guidance + one Euler step: noisyT += (1/total_step)*((1+w)*cond - w*uncond).
    bt::axpby_inplace(fc, fu, guid_cond, -guid_uncond);  // fc = (1+w)*cond - w*uncond
    bt::scale_inplace(fc, dt);
    bt::add_inplace(noisyT, fc);  // in-place feedback; noisyT is both step input and output
}

std::vector<float> Supertonic::Impl::denoise(
        const std::vector<float>& noisy, int channels, int frames,
        const std::vector<float>& text_emb, int text_len,
        const std::vector<float>& style_ttl, int current_step, int total_step) const {
    if (!ve_ready) fail("denoise() before vector-estimator load");
    if (channels != 144) fail("noisy_latent must have 144 channels");
    const int L = frames, T = text_len;
    if (static_cast<int>(text_emb.size()) != kTeC * T) fail("text_emb size != 256*T");
    if (static_cast<int>(style_ttl.size()) != kNStyle * kTeC) fail("style_ttl size != 50*256");

    // ── time embedding (host): sinusoidal -> Gemm -> Mish -> Gemm -> [1,64] ──
    const float t = (total_step != 0) ? static_cast<float>(current_step) / total_step : 0.0f;
    std::vector<float> sino(2 * kVeHalf);
    for (int fr = 0; fr < kVeHalf; ++fr) {
        const float a = t * 1000.0f * ve_sin_freqs[fr];
        sino[fr]           = std::sin(a);
        sino[kVeHalf + fr] = std::cos(a);
    }
    std::vector<float> m0(256);
    for (int o = 0; o < 256; ++o) {
        float s = ve_t0b[o];
        for (int i = 0; i < 64; ++i) s += ve_t0w[static_cast<std::size_t>(o) * 64 + i] * sino[i];
        const float sp = std::log1p(std::exp(s));     // softplus
        m0[o] = s * std::tanh(sp);                    // Mish: x*tanh(softplus(x))
    }
    std::vector<float> tev(64);
    for (int o = 0; o < 64; ++o) {
        float s = ve_t1b[o];
        for (int i = 0; i < 256; ++i) s += ve_t1w[static_cast<std::size_t>(o) * 256 + i] * m0[i];
        tev[o] = s;
    }
    bt::Tensor time_emb = bt::Tensor::from_host_on(dev, tev.data(), 1, 64);

    // proj_in is shared by the conditional and unconditional passes (same noisy).
    bt::Tensor scratch;
    bt::Tensor noisyT = bt::Tensor::from_host_on(dev, noisy.data(), 1, channels * L);
    bt::Tensor h0 = pconv(noisyT, ve_proj_in, channels, L, 1, 1, 0, 0, 0, scratch);  // [512,L]

    // conditional sources: text_emb [256,T]; style key=prototype, value=style_ttl.
    bt::Tensor text_cond = bt::Tensor::from_host_on(dev, text_emb.data(), 1, kTeC * T);
    std::vector<float> sv(static_cast<std::size_t>(kTeC) * kNStyle);  // [256,50] chan-major
    for (int s = 0; s < kNStyle; ++s)
        for (int c = 0; c < kTeC; ++c)
            sv[static_cast<std::size_t>(c) * kNStyle + s] = style_ttl[static_cast<std::size_t>(s) * kTeC + c];
    bt::Tensor style_val_cond = bt::Tensor::from_host_on(dev, sv.data(), 1, kTeC * kNStyle);

    // unconditional text: the special token broadcast over T positions.
    std::vector<float> tu(static_cast<std::size_t>(kTeC) * T);
    for (int c = 0; c < kTeC; ++c)
        for (int p = 0; p < T; ++p) tu[static_cast<std::size_t>(c) * T + p] = ve_text_uncond[c];
    bt::Tensor text_unc = bt::Tensor::from_host_on(dev, tu.data(), 1, kTeC * T);

    const RopeTables rt = build_rope_tables(ve_blocks[3].rope.theta, L, T, dev);
    std::vector<float> onesh(static_cast<std::size_t>(L), 1.0f);
    const bt::Tensor onesL = bt::Tensor::from_host_on(dev, onesh.data(), 1, L);
    bt::Tensor field_cond = run_field(h0, time_emb, text_cond, ve_style_key_proto, style_val_cond, L, T, rt, onesL);
    bt::Tensor field_unc  = run_field(h0, time_emb, text_unc,  ve_style_key_unc,    ve_style_val_unc,  L, T, rt, onesL);

    // CFG guidance + one Euler step: denoised = noisy + (1/total_step)*(4*cond - 3*uncond).
    bt::axpby_inplace(field_cond, field_unc, ve_guid_cond, -ve_guid_uncond);
    const float dt = (total_step != 0) ? 1.0f / total_step : 0.0f;
    bt::scale_inplace(field_cond, dt);
    bt::add_inplace(field_cond, noisyT);
    return field_cond.to_host_vector();
}

std::vector<int> Supertonic::Impl::text_to_ids(const std::string& text,
                                               const std::string& lang) const {
    if (!frontend_ready) fail("text_to_ids() before unicode_indexer.json load");
    if (!te_ready) fail("text_to_ids() needs the text encoder (vocab) loaded");
    const std::u32string s = preprocess_text(text, lang);
    std::vector<int> ids;
    ids.reserve(s.size());
    for (char32_t c : s) {
        // numpy uint16 cast wraps codepoints >= 0x10000; -1 (OOV) -> last row.
        int id = unicode_index[static_cast<std::uint16_t>(c)];
        if (id < 0) id += te_vocab;
        ids.push_back(id);
    }
    return ids;
}

AudioBuffer Supertonic::Impl::synthesize(const std::string& text,
        const std::string& lang, const VoiceStyle& voice, int total_step,
        float speed, std::uint64_t seed, float guidance) const {
    if (!ready)    fail("synthesize() before load()");
    if (!te_ready) fail("synthesize() needs the text encoder");
    if (!dp_ready) fail("synthesize() needs the duration predictor");
    if (!ve_ready) fail("synthesize() needs the vector estimator");
    if (total_step < 1) fail("total_step must be >= 1");
    if (speed <= 0.0f)  fail("speed must be > 0");
    if (static_cast<int>(voice.ttl.size()) != kNStyle * kTeC)
        fail("voice.ttl must be 50*256");
    if (voice.dp.empty()) fail("voice.dp is empty");

    const std::vector<int> ids = text_to_ids(text, lang);
    const int T = static_cast<int>(ids.size());

    // total length (seconds), shortened by 1/speed, then conditioning embedding.
    const float dur = predict_duration(ids, voice.dp) / speed;
    const std::vector<float> text_emb = encode_text(ids, voice.ttl);

    // duration -> latent frame count: ceil(dur*sr / (base_chunk*chunk)).
    const int chunk_size = cfg.base_chunk * cfg.chunk;          // 512*6 = 3072
    const float wav_len  = dur * static_cast<float>(cfg.sample_rate);
    int latent_len = static_cast<int>((wav_len + chunk_size - 1) / chunk_size);
    if (latent_len < 1) latent_len = 1;
    const int channels = cfg.latent_dim * cfg.chunk;            // 24*6 = 144

    const int L = latent_len;

    // running latent: seed from N(0,1) on-device, carried as [1, 144*L]
    // channel-major. For a single unpadded sentence the upstream latent_mask is
    // all-ones, so the seed is the raw noise (filled flat — same layout as the
    // old [144,L] seed). field_step updates it in place each step.
    bt::Tensor noisyT = bt::Tensor::empty_on(dev, 1, channels * L);
    bt::randn(seed, 0, noisyT);

    // ── per-utterance conditioning, built once (L, T fixed across all steps) ──
    // conditional text [256,T]; style value = style_ttl transposed to chan-major.
    bt::Tensor text_cond = bt::Tensor::from_host_on(dev, text_emb.data(), 1, kTeC * T);
    std::vector<float> sv(static_cast<std::size_t>(kTeC) * kNStyle);
    for (int s = 0; s < kNStyle; ++s)
        for (int c = 0; c < kTeC; ++c)
            sv[static_cast<std::size_t>(c) * kNStyle + s] =
                voice.ttl[static_cast<std::size_t>(s) * kTeC + c];
    bt::Tensor style_val_cond = bt::Tensor::from_host_on(dev, sv.data(), 1, kTeC * kNStyle);

    // unconditional text: the special token broadcast over T positions.
    std::vector<float> tu(static_cast<std::size_t>(kTeC) * T);
    for (int c = 0; c < kTeC; ++c)
        for (int p = 0; p < T; ++p) tu[static_cast<std::size_t>(c) * T + p] = ve_text_uncond[c];
    bt::Tensor text_unc = bt::Tensor::from_host_on(dev, tu.data(), 1, kTeC * T);

    // RoPE tables + the all-ones broadcast row — shared across every step.
    const RopeTables rt = build_rope_tables(ve_blocks[3].rope.theta, L, T, dev);
    std::vector<float> onesh(static_cast<std::size_t>(L), 1.0f);
    const bt::Tensor onesL = bt::Tensor::from_host_on(dev, onesh.data(), 1, L);
    const float dt = (total_step != 0) ? 1.0f / total_step : 0.0f;

    // Classifier-free-guidance scale w: field = (1+w)*cond - w*uncond. w=3 (the
    // upstream default) reproduces the original 4/3 weights; higher w pushes the
    // flow harder toward the text/style conditioning (crisper, more articulated),
    // lower w relaxes it (flatter, breathier). Clamp to >= 0 (w<0 inverts CFG).
    const float guid = (guidance < 0.0f) ? 0.0f : guidance;
    const float gc = 1.0f + guid, gu = guid;

    // Every step's time embedding (sinusoidal -> Gemm -> Mish -> Gemm, [1,64])
    // depends only on step/total_step, so precompute them all host-side, upload
    // once, and stage one row per step on-device (a device copy, capture-safe).
    std::vector<float> all_tev(static_cast<std::size_t>(total_step) * 64);
    for (int step = 0; step < total_step; ++step) {
        const float t = (total_step != 0) ? static_cast<float>(step) / total_step : 0.0f;
        std::vector<float> sino(2 * kVeHalf);
        for (int fr = 0; fr < kVeHalf; ++fr) {
            const float a = t * 1000.0f * ve_sin_freqs[fr];
            sino[fr] = std::sin(a); sino[kVeHalf + fr] = std::cos(a);
        }
        std::vector<float> m0(256);
        for (int o = 0; o < 256; ++o) {
            float s = ve_t0b[o];
            for (int i = 0; i < 64; ++i) s += ve_t0w[static_cast<std::size_t>(o) * 64 + i] * sino[i];
            const float sp = std::log1p(std::exp(s));
            m0[o] = s * std::tanh(sp);  // Mish
        }
        for (int o = 0; o < 64; ++o) {
            float s = ve_t1b[o];
            for (int i = 0; i < 256; ++i) s += ve_t1w[static_cast<std::size_t>(o) * 256 + i] * m0[i];
            all_tev[static_cast<std::size_t>(step) * 64 + o] = s;
        }
    }
    bt::Tensor time_all = bt::Tensor::from_host_on(dev, all_tev.data(), total_step, 64);
    bt::Tensor time_emb = bt::Tensor::empty_on(dev, 1, 64);
    auto stage_time = [&](int step) { bt::copy_d2d(time_all, step * 64, time_emb, 0, 64); };

    // Step 0 runs eagerly — it also warms every scratch buffer so a subsequent
    // capture allocates nothing new. On CUDA the identical body is then captured
    // once and replayed for steps 1..N-1: the per-step kernel-launch overhead,
    // not GPU compute, dominates this 99M-param field, and the graph removes it.
    stage_time(0);
    field_step(noisyT, time_emb, text_cond, text_unc, style_val_cond, onesL, rt, L, T, dt, gc, gu);

#ifdef BROSOUNDML_HAS_CUDA
    if (dev == bt::Device::CUDA && total_step > 1) {
        bt::sync_all();
        stage_time(1);
        bt::CudaGraph graph;
        {
            bt::CudaGraphCapture cap;
            field_step(noisyT, time_emb, text_cond, text_unc, style_val_cond, onesL, rt, L, T, dt, gc, gu);
            graph = cap.finish();
        }
        for (int step = 1; step < total_step; ++step) {
            stage_time(step);   // stream-ordered ahead of the replay
            graph.launch();
        }
        bt::sync_all();
    } else
#endif
    {
        for (int step = 1; step < total_step; ++step) {
            stage_time(step);
            field_step(noisyT, time_emb, text_cond, text_unc, style_val_cond, onesL, rt, L, T, dt, gc, gu);
        }
    }

    std::vector<float> xt = noisyT.to_host_vector();
    return decode(xt.data(), channels, latent_len);
}

AudioBuffer Supertonic::Impl::synthesize_long(
        const std::string& text, const std::string& lang, const VoiceStyle& voice,
        int total_step, float speed, std::uint64_t seed,
        float gap_seconds, int max_chars, float guidance) const {
    if (max_chars < 16) max_chars = 16;
    std::vector<std::string> chunks = split_sentences_impl(text, max_chars);
    if (chunks.empty()) chunks.push_back(text);  // all-whitespace / unsplittable

    AudioBuffer out;
    out.sample_rate = cfg.sample_rate;
    const int gap = gap_seconds > 0.0f
        ? static_cast<int>(gap_seconds * static_cast<float>(cfg.sample_rate)) : 0;
    for (std::size_t k = 0; k < chunks.size(); ++k) {
        const AudioBuffer w = synthesize(chunks[k], lang, voice, total_step, speed,
                                         seed + static_cast<std::uint64_t>(k), guidance);
        if (k > 0 && gap > 0)
            out.samples.insert(out.samples.end(), static_cast<std::size_t>(gap), 0.0f);
        out.samples.insert(out.samples.end(), w.samples.begin(), w.samples.end());
    }
    return out;
}

AudioBuffer Supertonic::Impl::decode(const float* latent, int channels,
                                     int frames) const {
    if (!ready) fail("decode() before load()");
    const int D  = cfg.latent_dim;          // 24
    const int CC = cfg.chunk;                // 6
    if (channels != D * CC)
        fail("latent channels (" + std::to_string(channels) + ") != latent_dim*chunk");
    if (frames <= 0) fail("frames must be positive");
    const int LF = CC * frames;             // de-chunked frame count

    // ── de-chunk [D*CC, frames] -> [D, CC*frames] + de-normalise (host) ──
    // ONNX: reshape [144,F]->[D,CC,F], transpose to [D,F,CC], reshape [D,CC*F];
    // channel (d*CC+j) at frame t lands at de-chunked position t*CC+j.
    std::vector<float> dn(static_cast<std::size_t>(D) * LF);
    for (int d = 0; d < D; ++d) {
        const float sd = latent_std[d], mn = latent_mean[d];
        for (int t = 0; t < frames; ++t) {
            for (int jx = 0; jx < CC; ++jx) {
                const float v = latent[static_cast<std::size_t>(d * CC + jx) * frames + t];
                dn[static_cast<std::size_t>(d) * LF + (t * CC + jx)] = v * inv_scale * sd + mn;
            }
        }
    }

    return decode_dn(dn, LF);
}

AudioBuffer Supertonic::Impl::decode_dn(const std::vector<float>& dn, int LF) const {
    const int D = cfg.latent_dim;           // 24
    bt::Tensor scratch;
    bt::Tensor h = rconv(bt::Tensor::from_host_on(dev, dn.data(), 1, D * LF),
                         conv_in, D, LF, /*dilation=*/1, /*groups=*/1, scratch);
    const int C = conv_in.cout;             // hidden channels (512)

    for (const ConvNeXtBlock& blk : blocks) {
        bt::Tensor dwy = rconv(h, blk.dw, C, LF, blk.dil, /*groups=*/C, scratch);
        // channel-wise LayerNorm: [C,L] -> (L,C) -> LN over C -> [C,L].
        bt::Tensor seq, seqn, normed;
        bt::nchw_to_sequence(dwy, /*N=*/1, C, /*H=*/1, /*W=*/LF, seq);
        bt::layernorm_forward_inference_batched(seq, blk.ln_g, blk.ln_b, seqn, 1.0e-6f);
        bt::sequence_to_nchw(seqn, /*N=*/1, C, /*H=*/1, /*W=*/LF, normed);

        bt::Tensor a = rconv(normed, blk.pw1, C, LF, 1, 1, scratch);  // 1x1
        bt::Tensor ga, y;
        bt::gelu_exact_forward(a, ga);
        y = rconv(ga, blk.pw2, blk.pw1.cout, LF, 1, 1, scratch);      // 1x1, gamma folded
        bt::add_inplace(y, h);              // residual
        h = std::move(y);
    }

    bt::Tensor hb;
    bt::batch_norm_inference(h, bn_g, bn_b, bn_mean, bn_var, /*N=*/1, C, /*H=*/1,
                             /*W=*/LF, /*eps=*/1.0e-5f, hb);

    bt::Tensor hp, out;
    bt::Tensor hh = rconv(hb, head1, C, LF, 1, 1, scratch);   // kernel 3
    bt::leaky_relu_forward(hh, prelu_slope, hp);
    out = rconv(hp, head2, head1.cout, LF, 1, 1, scratch);    // kernel 1

    // ── output reshape [base_chunk, LF] -> waveform (interleave to time) ──
    // ONNX: transpose [B,base_chunk,LF]->[B,LF,base_chunk], flatten; sample at
    // position s*base_chunk + c.
    const int BC = head2.cout;              // base_chunk (512)
    const std::vector<float> od = out.to_host_vector();   // (BC*LF) channel-major
    AudioBuffer wav;
    wav.sample_rate = cfg.sample_rate;
    wav.samples.resize(static_cast<std::size_t>(BC) * LF);
    for (int c = 0; c < BC; ++c)
        for (int s = 0; s < LF; ++s)
            wav.samples[static_cast<std::size_t>(s) * BC + c] =
                od[static_cast<std::size_t>(c) * LF + s];
    return wav;
}

// ─── public shell ────────────────────────────────────────────────────────────

namespace {
// Flatten a (possibly nested) JSON array of numbers into `out`, in order.
void flatten_numbers(const j::Value& v, std::vector<float>& out) {
    if (v.is_array())
        for (const j::Value& e : v.as_array()) flatten_numbers(e, out);
    else if (v.is_number())
        out.push_back(static_cast<float>(v.as_number()));
}
}  // namespace

Supertonic::Supertonic() : impl_(std::make_unique<Impl>()) {}
Supertonic::~Supertonic() = default;
Supertonic::Supertonic(Supertonic&&) noexcept = default;
Supertonic& Supertonic::operator=(Supertonic&&) noexcept = default;

void Supertonic::load(const std::string& model_dir, bt::Device device) {
    impl_->load(model_dir, device);
}

AudioBuffer Supertonic::decode(const float* latent, int channels, int frames) const {
    return impl_->decode(latent, channels, frames);
}

AudioBuffer Supertonic::decode(const std::vector<float>& latent, int channels,
                               int frames) const {
    return impl_->decode(latent.data(), channels, frames);
}

std::vector<float> Supertonic::encode_text(const std::vector<int>& text_ids,
                                           const std::vector<float>& style_ttl) const {
    return impl_->encode_text(text_ids, style_ttl);
}

float Supertonic::predict_duration(const std::vector<int>& text_ids,
                                   const std::vector<float>& style_dp) const {
    return impl_->predict_duration(text_ids, style_dp);
}

std::vector<float> Supertonic::denoise(const std::vector<float>& noisy_latent,
                                       int channels, int frames,
                                       const std::vector<float>& text_emb,
                                       int text_len,
                                       const std::vector<float>& style_ttl,
                                       int current_step, int total_step) const {
    return impl_->denoise(noisy_latent, channels, frames, text_emb, text_len,
                          style_ttl, current_step, total_step);
}

VoiceStyle Supertonic::load_voice_style(const std::string& path) const {
    const j::Value root = j::parse(slurp(fs::path(path)));
    const j::Value* ttl = root.find("style_ttl");
    const j::Value* dp  = root.find("style_dp");
    if (!ttl || !dp) fail("voice style '" + path + "' missing style_ttl/style_dp");
    const j::Value* ttl_d = ttl->find("data");
    const j::Value* dp_d  = dp->find("data");
    if (!ttl_d || !dp_d) fail("voice style '" + path + "' missing data arrays");

    VoiceStyle vs;
    flatten_numbers(*ttl_d, vs.ttl);
    flatten_numbers(*dp_d,  vs.dp);
    if (static_cast<int>(vs.ttl.size()) != kNStyle * kTeC)
        fail("voice style ttl is " + std::to_string(vs.ttl.size()) + ", expected 50*256");
    if (vs.dp.empty()) fail("voice style dp is empty");
    return vs;
}

std::vector<int> Supertonic::text_to_ids(const std::string& text,
                                         const std::string& lang) const {
    return impl_->text_to_ids(text, lang);
}

AudioBuffer Supertonic::synthesize(const std::string& text, const std::string& lang,
                                   const VoiceStyle& voice, int total_step,
                                   float speed, std::uint64_t seed, float guidance) const {
    return impl_->synthesize(text, lang, voice, total_step, speed, seed, guidance);
}

AudioBuffer Supertonic::synthesize_long(const std::string& text, const std::string& lang,
                                        const VoiceStyle& voice, int total_step,
                                        float speed, std::uint64_t seed,
                                        float gap_seconds, int max_chars,
                                        float guidance) const {
    return impl_->synthesize_long(text, lang, voice, total_step, speed, seed,
                                  gap_seconds, max_chars, guidance);
}

std::vector<std::string> Supertonic::split_sentences(const std::string& text,
                                                     int max_chars) {
    if (max_chars < 16) max_chars = 16;
    return split_sentences_impl(text, max_chars);
}

const SupertonicConfig& Supertonic::config() const { return impl_->cfg; }
bool Supertonic::loaded() const { return impl_->ready; }
const std::vector<float>& Supertonic::latent_mean() const { return impl_->latent_mean; }
const std::vector<float>& Supertonic::latent_std() const { return impl_->latent_std; }

}  // namespace brosoundml
