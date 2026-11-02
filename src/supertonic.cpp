#include "brosoundml/supertonic.h"

#include "supertonic_internal.h"

#include <brotensor/cuda_graph.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;

using namespace st_detail;

namespace {

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

}  // namespace

namespace st_detail {

// Transpose a [R x C] matrix to [C x R] (device op, no host roundtrip).
bt::Tensor transpose2d(const bt::Tensor& x, int R, int C) {
    bt::Tensor y;
    bt::nchw_to_sequence(flat(x), /*N=*/1, /*C=*/R, /*H=*/1, /*W=*/C, y);  // -> [C,R]
    return y;
}

}  // namespace st_detail

namespace {

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

}  // namespace

// ─── Forward & Synthesis ─────────────────────────────────────────────────────

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

    if (bt::graph_capture_available(dev) && total_step > 1) {
        bt::sync_all();
        stage_time(1);
        bt::CudaGraph graph;
        {
            bt::CudaGraphCapture cap(dev);
            field_step(noisyT, time_emb, text_cond, text_unc, style_val_cond, onesL, rt, L, T, dt, gc, gu);
            graph = cap.finish();
        }
        for (int step = 1; step < total_step; ++step) {
            stage_time(step);   // stream-ordered ahead of the replay
            graph.launch();
        }
        bt::sync_all();
    } else
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

const SupertonicConfig& Supertonic::config() const { return impl_->cfg; }
bool Supertonic::loaded() const { return impl_->ready; }
const std::vector<float>& Supertonic::latent_mean() const { return impl_->latent_mean; }
const std::vector<float>& Supertonic::latent_std() const { return impl_->latent_std; }

}  // namespace brosoundml
