#include "brosoundml/supertonic.h"

#include "brosoundml/detail/json.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

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

// A conv1d weight flattened to brotensor's (Cout, (Cin/groups)*K) OIL layout.
struct ConvW {
    bt::Tensor w, b;
    bool       has_b = false;
    int        cin_pg = 0, cout = 0, k = 0;  // cin_pg = weight.shape[1] = Cin/groups
};

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

struct ConvNeXtBlock {
    ConvW      dw;          // depthwise conv (groups = channels), kernel 7, dilated
    int        dil = 1;
    bt::Tensor ln_g, ln_b;  // channel-wise LayerNorm affine
    ConvW      pw1;         // 1x1, channels -> intermediate
    ConvW      pw2;         // 1x1, intermediate -> channels (gamma folded)
};

// One Glow-TTS relative-position attention block (text encoder attn_encoder):
// 1x1 q/k/v/o projections, windowed relative key/value embeddings, two
// channel-wise LayerNorms, and a 1x1-conv FFN (conv_1 -> relu -> conv_2).
struct AttnLayer {
    ConvW      conv_q, conv_k, conv_v, conv_o;  // 1x1
    bt::Tensor emb_rel_k, emb_rel_v;            // [2*window+1, kc]
    bt::Tensor n1_g, n1_b, n2_g, n2_b;          // post-attn / post-ffn LN
    ConvW      ffn1, ffn2;                      // 1x1 (C->filter, filter->C)
};

// One speech-prompted style cross-attention (tanh-gated, query=text,
// key=learned style prototype, value=style_ttl). Folded linears carried as
// 1x1 convs over channel-major activations.
struct StyleAttn {
    ConvW wq, wk, wv, wo;  // 1x1, from the transposed onnx::MatMul_* weights
};

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

// 1-D conv with an explicit (pad_left, pad_right) pre-pad in `mode` (0 zero,
// 2 replicate/edge). Output length == L + pad_left + pad_right - dilation*(k-1).
// groups == Cin for depthwise; k==1 convs pass pad 0. scratch is reused.
bt::Tensor pconv(const bt::Tensor& x, const ConvW& c, int Cin, int L,
                 int dilation, int groups, int pad_left, int pad_right,
                 int mode, bt::Tensor& scratch) {
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

// In-place row softmax over a [R x Cn] matrix (each row independently).
void softmax_rows(bt::Tensor& m, int R, int Cn) {
    for (int r = 0; r < R; ++r) {
        bt::Tensor row = sub_view(m, r * Cn, 1, Cn);
        bt::softmax_forward(row, row);
    }
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

// Text-encoder architecture constants (fixed for supertonic-3, from tts.json).
constexpr int kTeC      = 256;  // hidden channels
constexpr int kTeHeads  = 4;    // attn heads
constexpr int kTeKc     = 64;   // per-head channels (kTeC / kTeHeads)
constexpr int kTeWindow = 4;    // relative-attention window (emb_rel rows = 9)
constexpr int kNStyle   = 50;   // style tokens
constexpr int kSpteHeads = 2;   // style cross-attn heads
constexpr int kSpteHd    = 128; // per-head channels (kTeC / kSpteHeads)
constexpr float kSpteScale = 16.0f;  // QK divisor (upstream Div constant)

// One relative-position attention sub-layer: x [C,T] -> conv_o(attn) [C,T].
bt::Tensor rel_attention(const bt::Tensor& x, const AttnLayer& L, int T) {
    const int C = kTeC, nh = kTeHeads, kc = kTeKc;
    bt::Tensor scr;
    bt::Tensor q = pconv(x, L.conv_q, C, T, 1, 1, 0, 0, 0, scr);  // [C,T]
    bt::Tensor k = pconv(x, L.conv_k, C, T, 1, 1, 0, 0, 0, scr);
    bt::Tensor v = pconv(x, L.conv_v, C, T, 1, 1, 0, 0, 0, scr);
    const float inv = 1.0f / std::sqrt(static_cast<float>(kc));

    const bt::Tensor key_rel  = get_rel_emb(L.emb_rel_k, T, kTeWindow);  // [2T-1,kc]
    const bt::Tensor val_rel  = get_rel_emb(L.emb_rel_v, T, kTeWindow);
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

// One speech-prompted style cross-attention: query [C,T], key source [C,S],
// value source [C,S] -> out_fc(MHA) [C,T] (tanh on keys, scale 1/16, softmax
// over S style tokens). S == kNStyle.
bt::Tensor style_attention(const bt::Tensor& query, const bt::Tensor& key_src,
                           const bt::Tensor& val_src, const StyleAttn& A, int T) {
    const int C = kTeC, nh = kSpteHeads, hd = kSpteHd, S = kNStyle;
    bt::Tensor scr;
    bt::Tensor q = pconv(query,   A.wq, C, T, 1, 1, 0, 0, 0, scr);  // [C,T]
    bt::Tensor k = pconv(key_src, A.wk, C, S, 1, 1, 0, 0, 0, scr);  // [C,S]
    bt::Tensor v = pconv(val_src, A.wv, C, S, 1, 1, 0, 0, 0, scr);  // [C,S]
    const float inv = 1.0f / kSpteScale;

    std::vector<bt::Tensor> heads(nh);
    std::vector<const bt::Tensor*> hp(nh);
    for (int h = 0; h < nh; ++h) {
        const bt::Tensor qh = sub_view(q, h * hd * T, hd, T);  // [hd,T]
        const bt::Tensor kh = sub_view(k, h * hd * S, hd, S);  // [hd,S]
        const bt::Tensor vh = sub_view(v, h * hd * S, hd, S);  // [hd,S]
        bt::Tensor qhT = transpose2d(qh, hd, T);               // [T,hd]
        bt::Tensor ktanh; bt::tanh_forward(kh, ktanh);         // [hd,S]
        bt::Tensor scores; bt::matmul(qhT, ktanh, scores);     // [T,S]
        bt::scale_inplace(scores, inv);
        softmax_rows(scores, T, S);

        bt::Tensor vhT = transpose2d(vh, hd, S);               // [S,hd]
        bt::Tensor outh; bt::matmul(scores, vhT, outh);        // [T,hd]
        heads[h] = transpose2d(outh, T, hd);                   // [hd,T]
        hp[h] = &heads[h];
    }
    bt::Tensor cat; bt::concat_rows(hp, cat);                  // [C,T] channel-major
    bt::Tensor scr2;
    return pconv(flat(cat), A.wo, C, T, 1, 1, 0, 0, 0, scr2);  // [C,T]
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

    void load(const std::string& dir, bt::Device device);
    void load_text_encoder(const fs::path& path);
    AudioBuffer decode(const float* latent, int channels, int frames) const;
    std::vector<float> encode_text(const std::vector<int>& ids,
                                   const std::vector<float>& style) const;
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

    ready = true;

    // Text encoder is optional today — load it when its weights are present.
    const fs::path te = root / "text_encoder.safetensors";
    if (fs::exists(te)) load_text_encoder(te);
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
    const std::string SP = "tts.ttl.speech_prompted_text_encoder.";
    auto load_spte = [&](const std::string& at, const std::string& wq,
                         const std::string& wk, const std::string& wv,
                         const std::string& wo) {
        StyleAttn A;
        A.wq = load_linear_as_conv(f, wq, SP + at + ".W_query.linear.bias", dev);
        A.wk = load_linear_as_conv(f, wk, SP + at + ".W_key.linear.bias",   dev);
        A.wv = load_linear_as_conv(f, wv, SP + at + ".W_value.linear.bias", dev);
        A.wo = load_linear_as_conv(f, wo, SP + at + ".out_fc.linear.bias",  dev);
        return A;
    };
    spte1 = load_spte("attention1", "onnx::MatMul_3680", "onnx::MatMul_3681",
                      "onnx::MatMul_3682", "onnx::MatMul_3683");
    spte2 = load_spte("attention2", "onnx::MatMul_3684", "onnx::MatMul_3685",
                      "onnx::MatMul_3686", "onnx::MatMul_3687");
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
    for (int i = 0; i < 6; ++i) {
        const ConvNeXtBlock& blk = te_blocks[i];
        const int pad = 2 * kDil[i];
        bt::Tensor dwy = pconv(h, blk.dw, C, T, kDil[i], C, pad, pad, /*edge=*/2, scratch);
        bt::Tensor seq, seqn, normed;
        bt::nchw_to_sequence(dwy, 1, C, 1, T, seq);
        bt::layernorm_forward_inference_batched(seq, blk.ln_g, blk.ln_b, seqn, 1.0e-6f);
        bt::sequence_to_nchw(seqn, 1, C, 1, T, normed);
        bt::Tensor a = pconv(normed, blk.pw1, C, T, 1, 1, 0, 0, 0, scratch);
        bt::Tensor ga; bt::gelu_exact_forward(a, ga);
        bt::Tensor y = pconv(ga, blk.pw2, blk.pw1.cout, T, 1, 1, 0, 0, 0, scratch);
        bt::add_inplace(y, h);
        h = std::move(y);
    }
    const bt::Tensor x0 = h.clone();  // global residual skip (proj_out)

    // ── 4 relative-position attention layers ──
    for (const AttnLayer& L : te_attn) {
        bt::Tensor a = rel_attention(h, L, T);  // conv_o(attn) [C,T]
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
    bt::Tensor x1 = style_attention(t0, te_style_key, styleV, spte1, T);
    bt::add_inplace(x1, t0);                            // residual to text
    bt::Tensor x2 = style_attention(x1, te_style_key, styleV, spte2, T);
    bt::add_inplace(x2, t0);                            // residual to text
    bt::Tensor out = layernorm_chan(x2, spte_norm_g, spte_norm_b, C, T);  // [C,T]
    return out.to_host_vector();
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

const SupertonicConfig& Supertonic::config() const { return impl_->cfg; }
bool Supertonic::loaded() const { return impl_->ready; }

}  // namespace brosoundml
