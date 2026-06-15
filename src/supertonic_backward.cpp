#include "supertonic_backward.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace brosoundml {
namespace st_detail {

namespace bt = brotensor;

namespace {

// Flat (1, rows*cols) view — the NCHW/pad ops want X shaped (N, C*H*W).
bt::Tensor flat(const bt::Tensor& t) {
    return bt::Tensor::view(t.device, t.data, 1, t.rows * t.cols, t.dtype);
}

// Architecture constants, mirroring supertonic.cpp's (anonymous) definitions.
constexpr int   kSpteHeads   = 2;       // style cross-attn heads
constexpr float kSpteScale   = 16.0f;   // style QK divisor
constexpr int   kVeC         = 512;     // field hidden channels
constexpr int   kVeHeads     = 8;       // text-attn heads
constexpr int   kVeHd        = 64;      // per-head channels
constexpr int   kVeHalf      = 32;      // RoPE rotates the 32+32 split halves
constexpr float kVeAttnScale = 16.0f;   // field QK divisor

// Non-owning [rows x cols] view at a float offset into a channel-major tensor.
bt::Tensor sub_view(const bt::Tensor& t, int off_elems, int rows, int cols) {
    return bt::Tensor::view(t.device, static_cast<float*>(t.data) + off_elems,
                            rows, cols, t.dtype);
}

// Owning [r x c] copy of a buffer's first r*c floats (re-tag the matmul shape).
bt::Tensor reshape_owned(const bt::Tensor& t, int r, int c) {
    return bt::Tensor::view(t.device, t.data, r, c, t.dtype).clone();
}

// [R x C] -> [C x R] (matches supertonic.cpp transpose2d / transpose2d_forward).
bt::Tensor transpose2d(const bt::Tensor& x, int R, int C) {
    bt::Tensor y;
    bt::nchw_to_sequence(flat(x), /*N=*/1, /*C=*/R, /*H=*/1, /*W=*/C, y);  // -> [C,R]
    return y;
}

// Per-position RoPE cos/sin tables [half x len]: position p angle (p/len)*theta[f].
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

// RoPE forward on one head [2*half x len] channel-major (== supertonic rope_apply):
//   out[0:half] = x1*cos - x2*sin,  out[half:] = x1*sin + x2*cos.
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
    bt::Tensor out; bt::concat_rows(parts, out);
    return reshape_owned(out, 2 * half, len);
}

// Adjoint of rope_apply (orthogonal rotation -> transpose = negated-angle rotate):
//   dx1 = d1*cos + d2*sin,  dx2 = -d1*sin + d2*cos.
bt::Tensor rope_apply_backward(const bt::Tensor& dout, int half, int len,
                               const bt::Tensor& cosm, const bt::Tensor& sinm) {
    const bt::Tensor d1 = reshape_owned(sub_view(dout, 0, half, len), half, len);
    const bt::Tensor d2 = reshape_owned(sub_view(dout, half * len, half, len), half, len);
    bt::Tensor a = d1.clone(); bt::mul_inplace(a, cosm);   // d1*cos
    bt::Tensor b = d2.clone(); bt::mul_inplace(b, sinm);   // d2*sin
    bt::add_inplace(a, b);                                 // dx1 = d1*cos + d2*sin
    bt::Tensor c = d1.clone(); bt::mul_inplace(c, sinm);   // d1*sin
    bt::Tensor e = d2.clone(); bt::mul_inplace(e, cosm);   // d2*cos
    bt::axpby_inplace(e, c, 1.0f, -1.0f);                  // dx2 = d2*cos - d1*sin
    std::vector<const bt::Tensor*> parts = {&a, &e};
    bt::Tensor out; bt::concat_rows(parts, out);
    return reshape_owned(out, 2 * half, len);
}

// Row-wise softmax backward over an [R x Cn] matrix. brotensor's softmax_backward
// is a single full-Jacobian over its whole flat buffer (no row-batched variant),
// so we apply it independently per row. dScaled is overwritten [R, Cn].
void softmax_rows_backward(const bt::Tensor& P, const bt::Tensor& dP,
                           int R, int Cn, bt::Tensor& dScaled) {
    dScaled = bt::Tensor::zeros_on(P.device, R, Cn);
    for (int r = 0; r < R; ++r) {
        const bt::Tensor pr  = sub_view(P,  r * Cn, 1, Cn);
        const bt::Tensor dpr = sub_view(dP, r * Cn, 1, Cn);
        bt::Tensor dlr;
        bt::softmax_backward(pr, dpr, dlr);                // dlr (1, Cn)
        for (int c = 0; c < Cn; ++c) dScaled[r * Cn + c] = dlr[c];
    }
}

}  // namespace

// ─── pconv / rconv ──────────────────────────────────────────────────────────

void pconv_forward_train(const bt::Tensor& x, const ConvW& c,
                         int Cin, int L, int dilation, int groups,
                         int pad_left, int pad_right, int mode,
                         bt::Tensor& y, PConvCache& cache) {
    cache.Cin = Cin; cache.L = L; cache.Cout = c.cout; cache.k = c.k;
    cache.dilation = dilation; cache.groups = groups;
    cache.pad_left = pad_left; cache.pad_right = pad_right; cache.mode = mode;

    // 1×1, full, unpadded conv == Y[Cout,L] = W[Cout,Cin] @ X[Cin,L] (a GEMM).
    if (c.k == 1 && groups == 1 && pad_left == 0 && pad_right == 0) {
        cache.matmul_path = true;
        cache.Lp = L;
        cache.L_out = L;
        const bt::Tensor xv = bt::Tensor::view(x.device, x.data, Cin, L, x.dtype);
        cache.x_in = xv.clone();              // B operand for matmul_backward's dA
        bt::matmul(c.w, xv, y);               // [Cout, L], channel-major data
        if (c.has_b) bt::add_channel_bias_inplace(y, c.b, c.cout, L);
        y.rows = 1;
        y.cols = c.cout * L;
        return;
    }

    cache.matmul_path = false;
    bt::Tensor scratch;
    const bt::Tensor* in = &x;
    int Lp = L;
    if (pad_left > 0 || pad_right > 0) {
        bt::pad1d_forward(x, /*N=*/1, Cin, L, pad_left, pad_right, mode, scratch);
        in = &scratch;
        Lp = L + pad_left + pad_right;
    }
    cache.Lp = Lp;
    cache.L_out = Lp - dilation * (c.k - 1);
    bt::conv1d(*in, c.w, c.has_b ? &c.b : nullptr, /*N=*/1, Cin, Lp,
               c.cout, c.k, /*stride=*/1, /*padding=*/0, dilation, groups, y);
}

void rconv_forward_train(const bt::Tensor& x, const ConvW& c,
                         int Cin, int L, int dilation, int groups,
                         bt::Tensor& y, PConvCache& cache) {
    pconv_forward_train(x, c, Cin, L, dilation, groups,
                        /*pad_left=*/dilation * (c.k - 1), /*pad_right=*/0,
                        /*mode=*/2 /*replicate*/, y, cache);
}

void pconv_backward(const ConvW& c, const PConvCache& cache,
                    const bt::Tensor& dY, bt::Tensor& dX) {
    const bt::Device dev = dY.device;

    if (cache.matmul_path) {
        // forward: y = matmul(c.w, xv)  (A = c.w [Cout,Cin], B = xv [Cin,L]).
        // matmul_backward accumulates dB = c.wᵀ @ dC = the input grad we want;
        // dA (= dC @ Bᵀ) is computed against the cached input and discarded.
        bt::Tensor dC = bt::Tensor::view(dev, dY.data, cache.Cout, cache.L_out,
                                         dY.dtype);
        bt::Tensor dW_scratch = bt::Tensor::zeros_on(dev, cache.Cout, cache.Cin);
        dX = bt::Tensor::zeros_on(dev, cache.Cin, cache.L);
        bt::matmul_backward(c.w, cache.x_in, dC, dW_scratch, dX);
        dX.rows = 1;
        dX.cols = cache.Cin * cache.L;
        return;
    }

    // forward: padded = pad1d(x); y = conv1d(padded). Backward is the adjoint
    // chain conv1d_backward_input then pad1d_backward.
    bt::Tensor dXp;  // grad over the padded input (1, Cin*Lp)
    bt::conv1d_backward_input(c.w, dY, /*N=*/1, cache.Cin, cache.Lp, cache.Cout,
                              cache.k, /*stride=*/1, /*padding=*/0,
                              cache.dilation, cache.groups, dXp);
    if (cache.pad_left == 0 && cache.pad_right == 0) {
        dX = dXp;  // no pad: padded input IS the input
        return;
    }
    bt::pad1d_backward(dXp, /*N=*/1, cache.Cin, cache.L, cache.pad_left,
                       cache.pad_right, cache.mode, dX);
}

// ─── seq transpose (host adjoint) ───────────────────────────────────────────

void transpose2d_forward(const bt::Tensor& x, int R, int C, bt::Tensor& y) {
    bt::nchw_to_sequence(flat(x), /*N=*/1, /*C=*/R, /*H=*/1, /*W=*/C, y);  // -> [C,R]
}

void transpose2d_backward(const bt::Tensor& dY, int R, int C, bt::Tensor& dX) {
    // Adjoint of a permutation is its inverse: sequence_to_nchw with the same
    // dims undoes nchw_to_sequence(.,1,R,1,C).
    bt::sequence_to_nchw(flat(dY), /*N=*/1, /*C=*/R, /*H=*/1, /*W=*/C, dX);  // -> [R,C]
    dX.rows = R;
    dX.cols = C;
}

// ─── ConvNeXt-1D block ──────────────────────────────────────────────────────

void convnext_block_forward_train(const bt::Tensor& h, const ConvNeXtBlock& blk,
                                  int C, int L, int dil,
                                  bt::Tensor& y, ConvNeXtCache& cache) {
    cache.C = C; cache.L = L; cache.dil = dil;
    const int pad = 2 * dil;

    // depthwise conv (groups = C), symmetric edge pad.
    bt::Tensor dwy;
    pconv_forward_train(h, blk.dw, C, L, dil, /*groups=*/C, pad, pad,
                        /*edge=*/2, dwy, cache.dw);

    // channel-wise LayerNorm: transpose to [L,C], LN over C, transpose back.
    bt::Tensor seq;
    bt::nchw_to_sequence(dwy, 1, C, 1, L, seq);                 // [L, C]
    bt::Tensor seqn, mean_r;
    bt::layernorm_forward_batched_with_caches(seq, blk.ln_g, blk.ln_b, seqn,
                                              cache.ln_xhat, mean_r,
                                              cache.ln_rstd, 1.0e-6f);
    bt::Tensor normed;
    bt::sequence_to_nchw(seqn, 1, C, 1, L, normed);            // [C, L]

    // 1×1 expand -> GELU -> 1×1 project.
    pconv_forward_train(normed, blk.pw1, C, L, 1, 1, 0, 0, 0, cache.a, cache.pw1);
    bt::Tensor ga; bt::gelu_exact_forward(cache.a, ga);
    pconv_forward_train(ga, blk.pw2, blk.pw1.cout, L, 1, 1, 0, 0, 0, y, cache.pw2);

    // residual.
    bt::add_inplace(y, h);
}

void convnext_block_backward(const ConvNeXtBlock& blk, const ConvNeXtCache& cache,
                             const bt::Tensor& dY, bt::Tensor& dX) {
    const int C = cache.C, L = cache.L;

    // y = yc + h: the residual sends dY straight through; the conv chain also
    // starts from dY (dyc = dY).
    bt::Tensor dGa;
    pconv_backward(blk.pw2, cache.pw2, dY, dGa);               // grad on gelu output

    bt::Tensor dA;
    bt::gelu_exact_backward(cache.a, dGa, dA);                 // grad on pw1 output

    bt::Tensor dNormed;
    pconv_backward(blk.pw1, cache.pw1, dA, dNormed);           // grad on LN output [1,C*L]

    // sequence_to_nchw adjoint: nchw_to_sequence routes [1,C*L] -> [L,C].
    bt::Tensor dSeqn;
    bt::nchw_to_sequence(dNormed, 1, C, 1, L, dSeqn);          // [L, C]

    // LayerNorm backward over C (weights frozen — dGamma/dBeta discarded).
    //   (dY_RD, Xhat_RD, gamma, Rstd_R) -> dX_RD, plus accumulated dGamma/dBeta.
    bt::Tensor dSeq;
    bt::Tensor dGamma = bt::Tensor::zeros_on(dY.device, C, 1);
    bt::Tensor dBeta  = bt::Tensor::zeros_on(dY.device, C, 1);
    bt::layernorm_backward_batched_with_caches(dSeqn, cache.ln_xhat,
                                               blk.ln_g, cache.ln_rstd,
                                               dSeq, dGamma, dBeta);

    // nchw_to_sequence adjoint: sequence_to_nchw routes [L,C] -> [1,C*L].
    bt::Tensor dDwy;
    bt::sequence_to_nchw(dSeq, 1, C, 1, L, dDwy);             // [C, L] as (1, C*L)
    dDwy.rows = 1;
    dDwy.cols = C * L;

    // depthwise conv backward -> grad into the block input via the conv chain.
    bt::Tensor dh_conv;
    pconv_backward(blk.dw, cache.dw, dDwy, dh_conv);

    // dX = (conv-chain grad) + (residual grad = dY).
    bt::add_inplace(dh_conv, dY);
    dX = dh_conv;
    dX.rows = 1;
    dX.cols = C * L;
}

// ─── tanh-gated style cross-attention ────────────────────────────────────────

void style_attention_forward_train(const bt::Tensor& query, const bt::Tensor& key_src,
                                   const bt::Tensor& val_src, const StyleAttn& A,
                                   int Lq, int S, bt::Tensor& y, StyleAttnCache& cache) {
    const int nh    = kSpteHeads;
    const int Cq    = A.wq.cin_pg;
    const int inner = A.wq.cout;
    const int hd    = inner / nh;
    const int Ck    = A.wk.cin_pg;
    cache.nh = nh; cache.hd = hd; cache.Lq = Lq; cache.S = S;
    cache.inner = inner; cache.Cq = Cq; cache.Ck = Ck;
    cache.inv = 1.0f / kSpteScale;

    PConvCache kc;  // key projection geometry — forward only (key path frozen)
    bt::Tensor q, k, v;
    pconv_forward_train(query,   A.wq, Cq, Lq, 1, 1, 0, 0, 0, q, cache.qc);  // [inner,Lq]
    pconv_forward_train(key_src, A.wk, Ck, S,  1, 1, 0, 0, 0, k, kc);        // [inner,S]
    pconv_forward_train(val_src, A.wv, Ck, S,  1, 1, 0, 0, 0, v, cache.vc);  // [inner,S]

    cache.q     = reshape_owned(q, inner, Lq);
    bt::Tensor ktanh; bt::tanh_forward(reshape_owned(k, inner, S), ktanh);   // [inner,S]
    cache.ktanh = ktanh;
    cache.v     = reshape_owned(v, inner, S);
    cache.scores = bt::Tensor::zeros_on(query.device, nh * Lq, S);

    std::vector<bt::Tensor> heads(nh);
    std::vector<const bt::Tensor*> hp(nh);
    for (int h = 0; h < nh; ++h) {
        const bt::Tensor qh = sub_view(cache.q,     h * hd * Lq, hd, Lq);    // [hd,Lq]
        const bt::Tensor kh = sub_view(cache.ktanh, h * hd * S,  hd, S);     // [hd,S]
        const bt::Tensor vh = sub_view(cache.v,     h * hd * S,  hd, S);     // [hd,S]
        bt::Tensor qhT = transpose2d(qh, hd, Lq);                            // [Lq,hd]
        bt::Tensor scores; bt::matmul(qhT, kh, scores);                      // [Lq,S]
        bt::scale_inplace(scores, cache.inv);
        bt::softmax_rows_forward(scores, scores, Lq, S);
        for (int i = 0; i < Lq * S; ++i) cache.scores[h * Lq * S + i] = scores[i];

        bt::Tensor vhT = transpose2d(vh, hd, S);                             // [S,hd]
        bt::Tensor outh; bt::matmul(scores, vhT, outh);                      // [Lq,hd]
        heads[h] = transpose2d(outh, Lq, hd);                                // [hd,Lq]
        hp[h] = &heads[h];
    }
    bt::Tensor cat; bt::concat_rows(hp, cat);                                // [inner,Lq]
    pconv_forward_train(flat(cat), A.wo, inner, Lq, 1, 1, 0, 0, 0, y, cache.oc);
}

void style_attention_backward(const StyleAttn& A, const StyleAttnCache& cache,
                              const bt::Tensor& dY,
                              bt::Tensor& dQuery, bt::Tensor& dValue) {
    const bt::Device dev = dY.device;
    const int nh = cache.nh, hd = cache.hd, Lq = cache.Lq, S = cache.S,
              inner = cache.inner;

    bt::Tensor dCat;  // grad on concat_rows(heads) = [inner,Lq]
    pconv_backward(A.wo, cache.oc, dY, dCat);

    bt::Tensor dQ = bt::Tensor::zeros_on(dev, inner, Lq);
    bt::Tensor dV = bt::Tensor::zeros_on(dev, inner, S);

    for (int h = 0; h < nh; ++h) {
        // head split of dCat (heads were stacked by concat_rows along channels).
        const bt::Tensor dHeadOut = sub_view(dCat, h * hd * Lq, hd, Lq);     // [hd,Lq]
        // head_out = transpose2d(outh, Lq, hd): adjoint -> dOuth [Lq,hd].
        bt::Tensor dOuth; transpose2d_backward(dHeadOut, /*R=*/Lq, /*C=*/hd, dOuth);

        // recover the forward operands for this head.
        const bt::Tensor Ph = sub_view(cache.scores, h * Lq * S, Lq, S);     // [Lq,S]
        const bt::Tensor vh = sub_view(cache.v, h * hd * S, hd, S);          // [hd,S]
        bt::Tensor vhT = transpose2d(vh, hd, S);                             // [S,hd]

        // outh = matmul(Ph, vhT): dP, dVhT (dVhT kept -> value grad).
        bt::Tensor dP   = bt::Tensor::zeros_on(dev, Lq, S);
        bt::Tensor dVhT = bt::Tensor::zeros_on(dev, S, hd);
        bt::matmul_backward(Ph, vhT, dOuth, dP, dVhT);

        // softmax (per row over S) then the 1/16 scale -> grad on raw scores.
        bt::Tensor dScaled; softmax_rows_backward(Ph, dP, Lq, S, dScaled);
        bt::scale_inplace(dScaled, cache.inv);

        // scores_raw = matmul(qhT, ktanh_h): dQhT kept; dKtanh DISCARDED (frozen key).
        const bt::Tensor qh = sub_view(cache.q, h * hd * Lq, hd, Lq);        // [hd,Lq]
        bt::Tensor qhT = transpose2d(qh, hd, Lq);                            // [Lq,hd]
        const bt::Tensor kh = sub_view(cache.ktanh, h * hd * S, hd, S);      // [hd,S]
        bt::Tensor dQhT    = bt::Tensor::zeros_on(dev, Lq, hd);
        bt::Tensor dKtanh  = bt::Tensor::zeros_on(dev, hd, S);
        bt::matmul_backward(qhT, kh, dScaled, dQhT, dKtanh);

        // qhT = transpose2d(qh, hd, Lq): adjoint -> dQh [hd,Lq].
        bt::Tensor dQh; transpose2d_backward(dQhT, /*R=*/hd, /*C=*/Lq, dQh);
        // vhT = transpose2d(vh, hd, S): adjoint -> dVh [hd,S].
        bt::Tensor dVh; transpose2d_backward(dVhT, /*R=*/hd, /*C=*/S, dVh);

        for (int i = 0; i < hd * Lq; ++i) dQ[h * hd * Lq + i] = dQh[i];
        for (int i = 0; i < hd * S;  ++i) dV[h * hd * S  + i] = dVh[i];
    }

    pconv_backward(A.wq, cache.qc, flat(dQ), dQuery);   // -> (1, Cq*Lq)
    pconv_backward(A.wv, cache.vc, flat(dV), dValue);   // -> (1, Ck*S)
}

// ─── RoPE text cross-attention block (attention + residual + post-attn LN) ────

void rope_attention_forward_train(const bt::Tensor& h, const bt::Tensor& text_src,
                                  const VeRope& A, int L, int T,
                                  bt::Tensor& y, RopeAttnCache& cache) {
    const int nh = kVeHeads, hd = kVeHd, half = kVeHalf, C = kVeC;
    const int Ck = A.conv_k.cin_pg;
    cache.nh = nh; cache.hd = hd; cache.half = half; cache.L = L; cache.T = T;
    cache.C = C; cache.inner = C; cache.inv = 1.0f / kVeAttnScale;

    PConvCache kc, vc;  // k/v projection geometry — forward only (text frozen)
    bt::Tensor q, k, v;
    pconv_forward_train(h,        A.conv_q, kVeC, L, 1, 1, 0, 0, 0, q, cache.qc);  // [512,L]
    pconv_forward_train(text_src, A.conv_k, Ck,   T, 1, 1, 0, 0, 0, k, kc);        // [512,T]
    pconv_forward_train(text_src, A.conv_v, Ck,   T, 1, 1, 0, 0, 0, v, vc);        // [512,T]

    // RoPE tables: query positions normalised by L, key positions by T.
    bt::Tensor cosT, sinT;
    rope_tables(A.theta, L, h.device, cache.cosL, cache.sinL);
    rope_tables(A.theta, T, h.device, cosT, sinT);

    const bt::Tensor qm = reshape_owned(q, C, L);
    const bt::Tensor km = reshape_owned(k, C, T);
    cache.v = reshape_owned(v, C, T);
    cache.qr = bt::Tensor::zeros_on(h.device, C, L);
    cache.kr = bt::Tensor::zeros_on(h.device, C, T);
    cache.scores = bt::Tensor::zeros_on(h.device, nh * L, T);

    std::vector<bt::Tensor> heads(nh);
    std::vector<const bt::Tensor*> hp(nh);
    for (int hh = 0; hh < nh; ++hh) {
        const bt::Tensor qh = sub_view(qm, hh * hd * L, hd, L);   // [64,L]
        const bt::Tensor kh = sub_view(km, hh * hd * T, hd, T);   // [64,T]
        const bt::Tensor vh = sub_view(cache.v, hh * hd * T, hd, T);  // [64,T]
        bt::Tensor qr = rope_apply(qh, half, L, cache.cosL, cache.sinL);  // [64,L]
        bt::Tensor kr = rope_apply(kh, half, T, cosT, sinT);              // [64,T]
        for (int i = 0; i < hd * L; ++i) cache.qr[hh * hd * L + i] = qr[i];
        for (int i = 0; i < hd * T; ++i) cache.kr[hh * hd * T + i] = kr[i];

        bt::Tensor qrT = transpose2d(qr, hd, L);                 // [L,64]
        bt::Tensor scores; bt::matmul(qrT, kr, scores);          // [L,T]
        bt::scale_inplace(scores, cache.inv);
        bt::softmax_rows_forward(scores, scores, L, T);
        for (int i = 0; i < L * T; ++i) cache.scores[hh * L * T + i] = scores[i];

        bt::Tensor vhT = transpose2d(vh, hd, T);                 // [T,64]
        bt::Tensor outh; bt::matmul(scores, vhT, outh);          // [L,64]
        heads[hh] = transpose2d(outh, L, hd);                    // [64,L]
        hp[hh] = &heads[hh];
    }
    bt::Tensor cat; bt::concat_rows(hp, cat);                    // [512,L]
    bt::Tensor attn;
    pconv_forward_train(flat(cat), A.conv_o, kVeC, L, 1, 1, 0, 0, 0, attn, cache.oc);

    // residual + post-attention channel-wise LayerNorm (eps 1e-6).
    bt::add_inplace(attn, h);                                    // r = attn + h, (1,C*L)
    bt::Tensor seq, seqn, mean_r;
    bt::nchw_to_sequence(attn, 1, C, 1, L, seq);                 // [L,C]
    bt::layernorm_forward_batched_with_caches(seq, A.norm_g, A.norm_b, seqn,
                                              cache.ln_xhat, mean_r,
                                              cache.ln_rstd, 1.0e-6f);
    bt::sequence_to_nchw(seqn, 1, C, 1, L, y);                   // [C,L]
    y.rows = 1; y.cols = C * L;
}

void rope_attention_backward(const VeRope& A, const RopeAttnCache& cache,
                             const bt::Tensor& dY, bt::Tensor& dH) {
    const bt::Device dev = dY.device;
    const int nh = cache.nh, hd = cache.hd, half = cache.half,
              L = cache.L, T = cache.T, C = cache.C;

    // post-attn LayerNorm backward (frozen affine — dGamma/dBeta discarded).
    bt::Tensor dSeqn;
    bt::nchw_to_sequence(dY, 1, C, 1, L, dSeqn);                 // [L,C]
    bt::Tensor dSeq;
    bt::Tensor dGamma = bt::Tensor::zeros_on(dev, C, 1);
    bt::Tensor dBeta  = bt::Tensor::zeros_on(dev, C, 1);
    bt::layernorm_backward_batched_with_caches(dSeqn, cache.ln_xhat, A.norm_g,
                                               cache.ln_rstd, dSeq, dGamma, dBeta);
    bt::Tensor dR;
    bt::sequence_to_nchw(dSeq, 1, C, 1, L, dR);                  // [C,L] as (1,C*L)
    dR.rows = 1; dR.cols = C * L;

    // r = attn + h: residual sends dR straight to dH; the attention path uses dR too.
    bt::Tensor dCat;
    pconv_backward(A.conv_o, cache.oc, dR, dCat);                // [512,L]

    bt::Tensor dQ = bt::Tensor::zeros_on(dev, C, L);
    for (int hh = 0; hh < nh; ++hh) {
        const bt::Tensor dHeadOut = sub_view(dCat, hh * hd * L, hd, L);      // [64,L]
        bt::Tensor dOuth; transpose2d_backward(dHeadOut, /*R=*/L, /*C=*/hd, dOuth);  // [L,64]

        const bt::Tensor Ph = sub_view(cache.scores, hh * L * T, L, T);      // [L,T]
        const bt::Tensor vh = sub_view(cache.v, hh * hd * T, hd, T);         // [64,T]
        bt::Tensor vhT = transpose2d(vh, hd, T);                             // [T,64]
        // outh = matmul(Ph, vhT): dVhT DISCARDED (frozen value).
        bt::Tensor dP   = bt::Tensor::zeros_on(dev, L, T);
        bt::Tensor dVhT = bt::Tensor::zeros_on(dev, T, hd);
        bt::matmul_backward(Ph, vhT, dOuth, dP, dVhT);

        bt::Tensor dScaled; softmax_rows_backward(Ph, dP, L, T, dScaled);
        bt::scale_inplace(dScaled, cache.inv);

        // scores_raw = matmul(qrT, kr): dKr DISCARDED (frozen key).
        const bt::Tensor qrh = sub_view(cache.qr, hh * hd * L, hd, L);       // [64,L]
        bt::Tensor qrT = transpose2d(qrh, hd, L);                            // [L,64]
        const bt::Tensor krh = sub_view(cache.kr, hh * hd * T, hd, T);       // [64,T]
        bt::Tensor dQrT = bt::Tensor::zeros_on(dev, L, hd);
        bt::Tensor dKr  = bt::Tensor::zeros_on(dev, hd, T);
        bt::matmul_backward(qrT, krh, dScaled, dQrT, dKr);

        bt::Tensor dQr; transpose2d_backward(dQrT, /*R=*/hd, /*C=*/L, dQr);  // [64,L]
        // RoPE adjoint on the query gradient.
        bt::Tensor dQh = rope_apply_backward(dQr, half, L, cache.cosL, cache.sinL);
        for (int i = 0; i < hd * L; ++i) dQ[hh * hd * L + i] = dQh[i];
    }

    bt::Tensor dH_attn;
    pconv_backward(A.conv_q, cache.qc, flat(dQ), dH_attn);       // (1, C*L)

    // dH = attention-path grad + residual grad (dR).
    bt::add_inplace(dH_attn, dR);
    dH = dH_attn;
    dH.rows = 1; dH.cols = C * L;
}

// ─── flow-field (vector-estimator) single-step backward ──────────────────────

void field_forward_train(const bt::Tensor& h0, const bt::Tensor& time_emb,
                         const bt::Tensor& text_src, const bt::Tensor& style_key,
                         const bt::Tensor& style_val, int L, int T, int S,
                         const std::vector<VeBlock>& blocks,
                         const std::vector<ConvNeXtBlock>& ve_last,
                         const ConvW& ve_proj_out,
                         const bt::Tensor& onesL,
                         bt::Tensor& field_out, FieldCache& cache) {
    const int C = kVeC;
    cache.C = C; cache.L = L; cache.S = S;
    cache.blocks.assign(blocks.size(), FieldBlockCache{});

    bt::Tensor h = h0.clone();
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        const VeBlock&    blk = blocks[i];
        FieldBlockCache&  bc  = cache.blocks[i];
        bc.type = blk.type;

        if (blk.type == 0) {
            // ConvNeXt sub-blocks in sequence, dilation 1<<s (own residual each).
            bc.conv.assign(blk.conv.size(), ConvNeXtCache{});
            for (std::size_t s = 0; s < blk.conv.size(); ++s) {
                bt::Tensor y;
                convnext_block_forward_train(h, blk.conv[s], C, L,
                                             1 << static_cast<int>(s), y, bc.conv[s]);
                h = y;
            }
        } else if (blk.type == 1) {
            // FiLM (additive-only): h += (time_emb @ W + b) broadcast across L.
            bt::Tensor proj; bt::matmul(time_emb, blk.film.w, proj);  // [1,512]
            bt::add_inplace(proj, blk.film.b);
            bt::Tensor pcol = transpose2d(proj, 1, C);                // [512,1]
            bt::Tensor tiled; bt::matmul(pcol, onesL, tiled);         // [512,L]
            bt::add_inplace(h, flat(tiled));
            // no cache: backward is the identity on dH.
        } else if (blk.type == 2) {
            bt::Tensor y;
            rope_attention_forward_train(h, text_src, blk.rope, L, T, y, bc.rope);
            h = y;
        } else {  // type 3: style cross-attention + residual + post-attn LN.
            bt::Tensor a;
            style_attention_forward_train(h, style_key, style_val, blk.style.attn,
                                          L, S, a, bc.style);          // [512,L]
            bt::add_inplace(a, h);                                     // r = a + h
            bt::Tensor seq, seqn, mean_r;
            bt::nchw_to_sequence(a, 1, C, 1, L, seq);                  // [L,C]
            bt::layernorm_forward_batched_with_caches(seq, blk.style.norm_g,
                                                      blk.style.norm_b, seqn,
                                                      bc.ln_xhat, mean_r,
                                                      bc.ln_rstd, 1.0e-6f);
            bt::Tensor hn;
            bt::sequence_to_nchw(seqn, 1, C, 1, L, hn);                // [C,L]
            hn.rows = 1; hn.cols = C * L;
            h = hn;
        }
    }

    // ve_last ConvNeXt blocks (dilation 1).
    cache.last.assign(ve_last.size(), ConvNeXtCache{});
    for (std::size_t i = 0; i < ve_last.size(); ++i) {
        bt::Tensor y;
        convnext_block_forward_train(h, ve_last[i], C, L, 1, y, cache.last[i]);
        h = y;
    }

    // ve_proj_out: 512 -> 144 (1×1).
    pconv_forward_train(flat(h), ve_proj_out, C, L, 1, 1, 0, 0, 0,
                        field_out, cache.proj_out);
}

void field_backward(const std::vector<VeBlock>& blocks,
                    const std::vector<ConvNeXtBlock>& ve_last,
                    const ConvW& ve_proj_out,
                    const FieldCache& cache,
                    const bt::Tensor& dY, bt::Tensor& dStyleVal) {
    const bt::Device dev = dY.device;
    const int C = cache.C, L = cache.L, S = cache.S;

    // The style gradient accumulator (channel-major [256, S]); set lazily.
    bool have_style = false;
    dStyleVal = bt::Tensor{};

    // ve_proj_out backward -> grad on h after ve_last.
    bt::Tensor dH;
    pconv_backward(ve_proj_out, cache.proj_out, dY, dH);     // (1, C*L)

    // ve_last ConvNeXt blocks in reverse.
    for (std::size_t i = ve_last.size(); i-- > 0;) {
        bt::Tensor dX;
        convnext_block_backward(ve_last[i], cache.last[i], dH, dX);
        dH = dX;
    }

    // VeBlocks in reverse order.
    for (std::size_t i = blocks.size(); i-- > 0;) {
        const VeBlock&         blk = blocks[i];
        const FieldBlockCache& bc  = cache.blocks[i];

        if (blk.type == 0) {
            // ConvNeXt sub-blocks in reverse (undo s = size-1 .. 0).
            for (std::size_t s = blk.conv.size(); s-- > 0;) {
                bt::Tensor dX;
                convnext_block_backward(blk.conv[s], bc.conv[s], dH, dX);
                dH = dX;
            }
        } else if (blk.type == 1) {
            // FiLM additive-only: identity on dH — nothing to do.
        } else if (blk.type == 2) {
            bt::Tensor dHn;
            rope_attention_backward(blk.rope, bc.rope, dH, dHn);
            dH = dHn;
        } else {  // type 3: style block.
            // post-attn LayerNorm backward (frozen affine -> dGamma/dBeta dropped).
            bt::Tensor dSeqn;
            bt::nchw_to_sequence(dH, 1, C, 1, L, dSeqn);              // [L,C]
            bt::Tensor dSeq;
            bt::Tensor dG = bt::Tensor::zeros_on(dev, C, 1);
            bt::Tensor dB = bt::Tensor::zeros_on(dev, C, 1);
            bt::layernorm_backward_batched_with_caches(dSeqn, bc.ln_xhat,
                                                       blk.style.norm_g, bc.ln_rstd,
                                                       dSeq, dG, dB);
            bt::Tensor dR;
            bt::sequence_to_nchw(dSeq, 1, C, 1, L, dR);              // [C,L] as (1,C*L)
            dR.rows = 1; dR.cols = C * L;

            // residual r = a + h: dA = dR (into style attn), dH_resid = dR.
            bt::Tensor dQuery, dValue;
            style_attention_backward(blk.style.attn, bc.style, dR, dQuery, dValue);

            // accumulate d(value) across all style blocks (sum).
            if (!have_style) { dStyleVal = dValue.clone(); have_style = true; }
            else             { bt::add_inplace(dStyleVal, dValue); }

            // dH = d(query) + residual grad.
            bt::add_inplace(dQuery, dR);
            dH = dQuery;
            dH.rows = 1; dH.cols = C * L;
        }
    }

    // grad into h0 (proj_in input — fixed noise/time/text) is discarded.
    // No style blocks -> a zero gradient sized to the value channels.
    if (!have_style) {
        int Ck = 0;
        for (const VeBlock& blk : blocks)
            if (blk.type == 3) { Ck = blk.style.attn.wk.cin_pg; break; }
        dStyleVal = bt::Tensor::zeros_on(dev, 1, Ck * S);
    }
}

// ─── vocoder (autoencoder decoder) ───────────────────────────────────────────

void vocoder_convnext_forward_train(const bt::Tensor& h, const ConvNeXtBlock& blk,
                                    int C, int L, int dil, bt::Tensor& y,
                                    ConvNeXtCache& cache) {
    cache.C = C; cache.L = L; cache.dil = dil;

    // depthwise conv — CAUSAL left-replicate pad (rconv), the vocoder variant.
    bt::Tensor dwy;
    rconv_forward_train(h, blk.dw, C, L, dil, /*groups=*/C, dwy, cache.dw);

    // channel-wise LayerNorm (inference math == training math; cache for backward).
    bt::Tensor seq; bt::nchw_to_sequence(dwy, 1, C, 1, L, seq);   // [L,C]
    bt::Tensor seqn, mean_r;
    bt::layernorm_forward_batched_with_caches(seq, blk.ln_g, blk.ln_b, seqn,
                                              cache.ln_xhat, mean_r,
                                              cache.ln_rstd, 1.0e-6f);
    bt::Tensor normed; bt::sequence_to_nchw(seqn, 1, C, 1, L, normed);  // [C,L]

    // 1×1 expand -> GELU -> 1×1 project (1×1 rconv == pconv matmul path).
    pconv_forward_train(normed, blk.pw1, C, L, 1, 1, 0, 0, 0, cache.a, cache.pw1);
    bt::Tensor ga; bt::gelu_exact_forward(cache.a, ga);
    pconv_forward_train(ga, blk.pw2, blk.pw1.cout, L, 1, 1, 0, 0, 0, y, cache.pw2);

    bt::add_inplace(y, h);                                       // residual
}

void vocoder_decode_forward_train(const bt::Tensor& latent_in,
                                  int D, int CC, int frames, float inv_scale,
                                  const std::vector<float>& latent_mean,
                                  const std::vector<float>& latent_std,
                                  const ConvW& conv_in,
                                  const std::vector<ConvNeXtBlock>& blocks,
                                  const bt::Tensor& bn_g, const bt::Tensor& bn_b,
                                  const bt::Tensor& bn_mean, const bt::Tensor& bn_var,
                                  float bn_eps, const ConvW& head1, float prelu_slope,
                                  const ConvW& head2, bt::Tensor& wave,
                                  VocoderCache& cache) {
    const bt::Device dev = latent_in.device;
    const int LF = CC * frames;
    cache.D = D; cache.CC = CC; cache.frames = frames; cache.LF = LF;

    // de-chunk + de-normalise (host): (d*CC+jx, t) -> dn[d, t*CC+jx].
    const std::vector<float> lin = latent_in.to_host_vector();
    std::vector<float> dn(static_cast<std::size_t>(D) * LF);
    for (int d = 0; d < D; ++d) {
        const float sd = latent_std[d], mn = latent_mean[d];
        for (int t = 0; t < frames; ++t)
            for (int jx = 0; jx < CC; ++jx) {
                const float v = lin[static_cast<std::size_t>(d * CC + jx) * frames + t];
                dn[static_cast<std::size_t>(d) * LF + (t * CC + jx)] = v * inv_scale * sd + mn;
            }
    }
    bt::Tensor dnt = bt::Tensor::from_host_on(dev, dn.data(), 1, D * LF);

    bt::Tensor h;
    rconv_forward_train(dnt, conv_in, D, LF, 1, 1, h, cache.conv_in);
    const int C = conv_in.cout; cache.C = C;

    cache.blocks.resize(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        bt::Tensor y;
        vocoder_convnext_forward_train(h, blocks[i], C, LF, blocks[i].dil, y,
                                       cache.blocks[i]);
        h = std::move(y);
    }

    // final BatchNorm (inference): per-channel affine. Cache the diagonal scale.
    bt::Tensor hb;
    bt::batch_norm_inference(h, bn_g, bn_b, bn_mean, bn_var, 1, C, 1, LF, bn_eps, hb);
    const std::vector<float> gh = bn_g.to_host_vector(), vh = bn_var.to_host_vector();
    std::vector<float> sc(C);
    for (int c = 0; c < C; ++c) sc[c] = gh[c] / std::sqrt(vh[c] + bn_eps);
    cache.bn_scale = bt::Tensor::from_host_on(dev, sc.data(), C, 1);

    // head: rconv(k3) -> leaky_relu -> rconv(k1).
    bt::Tensor hh;
    rconv_forward_train(hb, head1, C, LF, 1, 1, hh, cache.head1);
    cache.head1_out = hh.clone();
    bt::Tensor hp; bt::leaky_relu_forward(hh, prelu_slope, hp);
    bt::Tensor out;
    rconv_forward_train(hp, head2, head1.cout, LF, 1, 1, out, cache.head2);
    const int BC = head2.cout; cache.BC = BC;

    // output interleave (host): waveform[s*BC+c] = out[c*LF+s].
    const std::vector<float> od = out.to_host_vector();
    std::vector<float> wv(static_cast<std::size_t>(BC) * LF);
    for (int c = 0; c < BC; ++c)
        for (int s = 0; s < LF; ++s)
            wv[static_cast<std::size_t>(s) * BC + c] = od[static_cast<std::size_t>(c) * LF + s];
    wave = bt::Tensor::from_host_on(dev, wv.data(), 1, BC * LF);
}

void vocoder_decode_backward(const ConvW& conv_in,
                             const std::vector<ConvNeXtBlock>& blocks,
                             const ConvW& head1, float prelu_slope,
                             const ConvW& head2, float inv_scale,
                             const std::vector<float>& latent_std,
                             const VocoderCache& cache, const bt::Tensor& dWave,
                             bt::Tensor& dLatent) {
    const bt::Device dev = dWave.device;
    const int LF = cache.LF, C = cache.C, BC = cache.BC,
              D = cache.D, CC = cache.CC, frames = cache.frames;

    // output interleave adjoint: dOut[c*LF+s] = dWave[s*BC+c].
    const std::vector<float> dwv = dWave.to_host_vector();
    std::vector<float> dout(static_cast<std::size_t>(BC) * LF);
    for (int c = 0; c < BC; ++c)
        for (int s = 0; s < LF; ++s)
            dout[static_cast<std::size_t>(c) * LF + s] = dwv[static_cast<std::size_t>(s) * BC + c];
    bt::Tensor dOut = bt::Tensor::from_host_on(dev, dout.data(), 1, BC * LF);

    bt::Tensor dHp; pconv_backward(head2, cache.head2, dOut, dHp);
    bt::Tensor dHh; bt::leaky_relu_backward(cache.head1_out, dHp, prelu_slope, dHh);
    bt::Tensor dHb; pconv_backward(head1, cache.head1, dHh, dHb);

    // BatchNorm inference adjoint: diagonal scale gamma/sqrt(var+eps).
    const std::vector<float> sc = cache.bn_scale.to_host_vector();
    std::vector<float> dhv = dHb.to_host_vector();
    for (int c = 0; c < C; ++c)
        for (int l = 0; l < LF; ++l) dhv[static_cast<std::size_t>(c) * LF + l] *= sc[c];
    bt::Tensor dH = bt::Tensor::from_host_on(dev, dhv.data(), 1, C * LF);

    for (std::size_t i = blocks.size(); i-- > 0;) {
        bt::Tensor dX;
        convnext_block_backward(blocks[i], cache.blocks[i], dH, dX);
        dH = std::move(dX);
    }

    bt::Tensor dDn; pconv_backward(conv_in, cache.conv_in, dH, dDn);  // (1, D*LF)

    // de-chunk + de-norm adjoint (mn is a constant -> no grad; scale by inv_scale*std).
    const std::vector<float> ddn = dDn.to_host_vector();
    std::vector<float> dl(static_cast<std::size_t>(D) * CC * frames, 0.0f);
    for (int d = 0; d < D; ++d) {
        const float sd = inv_scale * latent_std[d];
        for (int t = 0; t < frames; ++t)
            for (int jx = 0; jx < CC; ++jx)
                dl[static_cast<std::size_t>(d * CC + jx) * frames + t] =
                    ddn[static_cast<std::size_t>(d) * LF + (t * CC + jx)] * sd;
    }
    dLatent = bt::Tensor::from_host_on(dev, dl.data(), 1, D * CC * frames);
}

// ─── ECAPA-TDNN speaker-encoder backward ─────────────────────────────────────

namespace {

// relu adjoint mask helper: dpre[i] = dpost[i] * (post[i] > 0).
void relu_mask_(const std::vector<float>& post, std::vector<float>& d) {
    for (std::size_t i = 0; i < d.size(); ++i) if (post[i] <= 0.0f) d[i] = 0.0f;
}

}  // namespace

// ── conv_same ─────────────────────────────────────────────────────────────────

void conv_same_forward_train(const QwenTtsSpkConv& c, const std::vector<float>& x,
                             int L, std::vector<float>& y, ConvSameCache& cache) {
    const bt::Device dev = c.w.device;   // weights pin the compute device
    bt::DeviceScope scope(dev);
    const int p = c.dilation * (c.k - 1) / 2;
    cache.cin = c.cin; cache.cout = c.cout; cache.k = c.k; cache.dilation = c.dilation;
    cache.L = L; cache.p = p; cache.Lp = L + 2 * p; cache.L_out = L;

    bt::Tensor X = bt::Tensor::from_host_on(dev, x.data(), 1, c.cin * L);
    bt::Tensor Xp;
    if (p > 0) bt::pad1d_forward(X, 1, c.cin, L, p, p, /*mode=*/1, Xp);  // reflect
    const bt::Tensor& in = (p > 0) ? Xp : X;
    bt::Tensor Y;
    bt::conv1d(in, c.w, &c.b, /*N=*/1, c.cin, cache.Lp, c.cout, c.k, /*stride=*/1,
               /*padding=*/0, c.dilation, Y);
    bt::Tensor Yh = (Y.device == bt::Device::CPU) ? std::move(Y) : Y.to(bt::Device::CPU);
    const float* yp = Yh.host_f32();
    y.assign(yp, yp + static_cast<std::size_t>(c.cout) * L);
}

void conv_same_backward(const QwenTtsSpkConv& c, const ConvSameCache& cache,
                        const std::vector<float>& dY, std::vector<float>& dX) {
    const bt::Device dev = c.w.device;
    bt::DeviceScope scope(dev);
    bt::Tensor dYt = bt::Tensor::from_host_on(dev, dY.data(), 1, cache.cout * cache.L_out);
    bt::Tensor dXp;  // grad over the (reflect-)padded input (1, cin*Lp)
    bt::conv1d_backward_input(c.w, dYt, /*N=*/1, cache.cin, cache.Lp, cache.cout,
                              cache.k, /*stride=*/1, /*padding=*/0, cache.dilation, dXp);
    bt::Tensor dXt;
    if (cache.p > 0)
        bt::pad1d_backward(dXp, /*N=*/1, cache.cin, cache.L, cache.p, cache.p,
                           /*mode=*/1, dXt);
    else
        dXt = std::move(dXp);
    bt::Tensor dXh = (dXt.device == bt::Device::CPU) ? std::move(dXt)
                                                     : dXt.to(bt::Device::CPU);
    const float* xp = dXh.host_f32();
    dX.assign(xp, xp + static_cast<std::size_t>(cache.cin) * cache.L);
}

// ── mel frontend ──────────────────────────────────────────────────────────────

void mel_forward_train(const float* wav, int n, const bt::Tensor& hann,
                       const bt::Tensor& mel_basis, int n_fft, int hop, int win,
                       int mel_dim, std::vector<float>& out, MelCache& cache) {
    bt::DeviceScope cpu(bt::Device::CPU);
    const int p = (n_fft - hop) / 2;
    const int Lp = n + 2 * p;
    const int nb = n_fft / 2 + 1;
    const int frames = 1 + (Lp - n_fft) / hop;
    const int M = mel_dim;
    cache.n = n; cache.Lp = Lp; cache.frames = frames; cache.nb = nb; cache.M = M;
    cache.n_fft = n_fft; cache.hop = hop; cache.win = win; cache.p_mel = p;

    bt::Tensor sig = bt::Tensor::from_host_on(bt::Device::CPU, wav, 1, n);
    bt::Tensor sigp;
    bt::pad1d_forward(sig, 1, 1, n, p, p, /*mode=*/1, sigp);   // reflect
    bt::Tensor spec;
    bt::stft(sigp, hann, /*N=*/1, n_fft, hop, win, /*center=*/false,
             /*normalized=*/false, spec);
    const float* s = spec.host_f32();          // (frames, 2*nb) interleaved
    const float* B = mel_basis.host_f32();      // (M, nb)

    cache.mag.assign(static_cast<std::size_t>(frames) * nb, 0.0f);
    cache.re.assign(cache.mag.size(), 0.0f);
    cache.im.assign(cache.mag.size(), 0.0f);
    for (int t = 0; t < frames; ++t) {
        const float* row = s + static_cast<std::size_t>(t) * 2 * nb;
        for (int b = 0; b < nb; ++b) {
            const float re = row[2 * b], im = row[2 * b + 1];
            const std::size_t idx = static_cast<std::size_t>(t) * nb + b;
            cache.re[idx] = re; cache.im[idx] = im;
            cache.mag[idx] = std::sqrt(re * re + im * im + 1e-9f);
        }
    }
    out.assign(static_cast<std::size_t>(M) * frames, 0.0f);
    cache.acc.assign(out.size(), 0.0f);
    for (int m = 0; m < M; ++m) {
        const float* bw = B + static_cast<std::size_t>(m) * nb;
        for (int t = 0; t < frames; ++t) {
            const float* mg = cache.mag.data() + static_cast<std::size_t>(t) * nb;
            double acc = 0.0;
            for (int b = 0; b < nb; ++b) acc += static_cast<double>(bw[b]) * mg[b];
            cache.acc[static_cast<std::size_t>(m) * frames + t] = static_cast<float>(acc);
            const float v = static_cast<float>(acc < 1e-5 ? 1e-5 : acc);
            out[static_cast<std::size_t>(m) * frames + t] = std::log(v);
        }
    }
}

void mel_backward(const bt::Tensor& hann, const bt::Tensor& mel_basis,
                  const MelCache& cache, const std::vector<float>& dOut,
                  std::vector<float>& dWav) {
    bt::DeviceScope cpu(bt::Device::CPU);
    const int frames = cache.frames, nb = cache.nb, M = cache.M;
    const float* B = mel_basis.host_f32();      // (M, nb)

    // log adjoint: v = max(acc,1e-5); d_acc = dOut/acc when acc>=1e-5 else 0.
    std::vector<float> d_acc(static_cast<std::size_t>(M) * frames, 0.0f);
    for (std::size_t i = 0; i < d_acc.size(); ++i) {
        const float a = cache.acc[i];
        d_acc[i] = (a >= 1e-5f) ? dOut[i] / a : 0.0f;
    }
    // mel = basis @ mag  ->  d_mag[t,b] = Σ_m basis[m,b] * d_acc[m,t].
    std::vector<float> d_mag(static_cast<std::size_t>(frames) * nb, 0.0f);
    for (int m = 0; m < M; ++m) {
        const float* bw = B + static_cast<std::size_t>(m) * nb;
        for (int t = 0; t < frames; ++t) {
            const float da = d_acc[static_cast<std::size_t>(m) * frames + t];
            float* dm = d_mag.data() + static_cast<std::size_t>(t) * nb;
            for (int b = 0; b < nb; ++b) dm[b] += bw[b] * da;
        }
    }
    // mag = sqrt(re²+im²+1e-9): d_re = d_mag·re/mag, d_im = d_mag·im/mag.
    std::vector<float> dspec(static_cast<std::size_t>(frames) * 2 * nb, 0.0f);
    for (int t = 0; t < frames; ++t)
        for (int b = 0; b < nb; ++b) {
            const std::size_t idx = static_cast<std::size_t>(t) * nb + b;
            const float mg = cache.mag[idx];
            const float dm = d_mag[idx];
            dspec[static_cast<std::size_t>(t) * 2 * nb + 2 * b]     = dm * cache.re[idx] / mg;
            dspec[static_cast<std::size_t>(t) * 2 * nb + 2 * b + 1] = dm * cache.im[idx] / mg;
        }
    bt::Tensor dSpec = bt::Tensor::from_host_on(bt::Device::CPU, dspec.data(),
                                                frames, 2 * nb);
    bt::Tensor dSigp;
    bt::stft_backward(dSpec, hann, /*N=*/1, cache.Lp, cache.n_fft, cache.hop,
                      cache.win, /*center=*/false, /*normalized=*/false, dSigp);
    bt::Tensor dWavT;
    bt::pad1d_backward(dSigp, /*N=*/1, /*C=*/1, cache.n, cache.p_mel, cache.p_mel,
                       /*mode=*/1, dWavT);
    dWav = dWavT.to_host_vector();
}

// ── Res2Net ───────────────────────────────────────────────────────────────────

void res2net_forward_train(const std::vector<QwenTtsSpkConv>& res2net,
                           const std::vector<float>& x, int C, int L, int scale,
                           std::vector<float>& y, Res2NetCache& cache) {
    const int hc = C / scale;
    cache.C = C; cache.L = L; cache.scale = scale; cache.hc = hc;
    cache.conv.assign(scale - 1, ConvSameCache{});
    cache.part.assign(scale - 1, std::vector<float>{});
    y.assign(static_cast<std::size_t>(C) * L, 0.0f);
    std::vector<float> prev;
    for (int i = 0; i < scale; ++i) {
        const float* chunk = x.data() + static_cast<std::size_t>(i) * hc * L;
        std::vector<float> part;
        if (i == 0) {
            part.assign(chunk, chunk + static_cast<std::size_t>(hc) * L);
        } else {
            std::vector<float> in(static_cast<std::size_t>(hc) * L);
            if (i == 1) std::copy(chunk, chunk + in.size(), in.begin());
            else for (std::size_t j = 0; j < in.size(); ++j) in[j] = chunk[j] + prev[j];
            conv_same_forward_train(res2net[i - 1], in, L, part, cache.conv[i - 1]);
            for (float& v : part) if (v < 0.0f) v = 0.0f;   // relu
            cache.part[i - 1] = part;
        }
        std::copy(part.begin(), part.end(),
                  y.begin() + static_cast<std::size_t>(i) * hc * L);
        prev = std::move(part);
    }
}

void res2net_backward(const std::vector<QwenTtsSpkConv>& res2net,
                      const Res2NetCache& cache, const std::vector<float>& dY,
                      std::vector<float>& dX) {
    const int C = cache.C, L = cache.L, scale = cache.scale, hc = cache.hc;
    dX.assign(static_cast<std::size_t>(C) * L, 0.0f);
    // chunk 0 passes straight through.
    std::copy(dY.begin(), dY.begin() + static_cast<std::size_t>(hc) * L, dX.begin());
    // per-chunk post-relu output gradients, seeded from dY chunks.
    std::vector<std::vector<float>> dpart(scale);
    for (int i = 1; i < scale; ++i) {
        const float* d = dY.data() + static_cast<std::size_t>(i) * hc * L;
        dpart[i].assign(d, d + static_cast<std::size_t>(hc) * L);
    }
    for (int i = scale - 1; i >= 1; --i) {
        std::vector<float> dpre = dpart[i];                  // relu' (mask = part>0)
        relu_mask_(cache.part[i - 1], dpre);
        std::vector<float> d_in;
        conv_same_backward(res2net[i - 1], cache.conv[i - 1], dpre, d_in);  // (hc*L)
        float* dxc = dX.data() + static_cast<std::size_t>(i) * hc * L;
        for (std::size_t j = 0; j < d_in.size(); ++j) dxc[j] += d_in[j];    // chunk_i
        if (i >= 2)                                                          // + prev=part_{i-1}
            for (std::size_t j = 0; j < d_in.size(); ++j) dpart[i - 1][j] += d_in[j];
    }
}

// ── Squeeze-excitation ──────────────────────────────────────────────────────

void se_forward_train(const QwenTtsSpkConv& se1, const QwenTtsSpkConv& se2,
                      const std::vector<float>& x, int C, int L,
                      std::vector<float>& y, SeCache& cache) {
    cache.C = C; cache.L = L; cache.x = x;
    std::vector<float> m(C, 0.0f);
    for (int c = 0; c < C; ++c) {
        const float* row = x.data() + static_cast<std::size_t>(c) * L;
        double s = 0.0;
        for (int t = 0; t < L; ++t) s += row[t];
        m[c] = static_cast<float>(s / L);
    }
    conv_same_forward_train(se1, m, /*L=*/1, cache.g1, cache.se1c);  // (se)
    for (float& v : cache.g1) if (v < 0.0f) v = 0.0f;                // relu
    conv_same_forward_train(se2, cache.g1, /*L=*/1, cache.g, cache.se2c);  // (C)
    for (float& v : cache.g) v = 1.0f / (1.0f + std::exp(-v));       // sigmoid
    y.assign(x.size(), 0.0f);
    for (int c = 0; c < C; ++c)
        for (int t = 0; t < L; ++t)
            y[static_cast<std::size_t>(c) * L + t] =
                x[static_cast<std::size_t>(c) * L + t] * cache.g[c];
}

void se_backward(const QwenTtsSpkConv& se1, const QwenTtsSpkConv& se2,
                 const SeCache& cache, const std::vector<float>& dY,
                 std::vector<float>& dX) {
    const int C = cache.C, L = cache.L;
    dX.assign(static_cast<std::size_t>(C) * L, 0.0f);
    std::vector<float> dg(C, 0.0f);
    for (int c = 0; c < C; ++c) {
        const float gc = cache.g[c];
        double acc = 0.0;
        for (int t = 0; t < L; ++t) {
            const std::size_t i = static_cast<std::size_t>(c) * L + t;
            dX[i] = dY[i] * gc;                       // through the multiply
            acc += static_cast<double>(dY[i]) * cache.x[i];
        }
        dg[c] = static_cast<float>(acc);
    }
    // sigmoid: g=σ(pre2) -> d_pre2 = dg·g·(1-g).
    std::vector<float> d_pre2(C, 0.0f);
    for (int c = 0; c < C; ++c) d_pre2[c] = dg[c] * cache.g[c] * (1.0f - cache.g[c]);
    std::vector<float> d_g1;
    conv_same_backward(se2, cache.se2c, d_pre2, d_g1);   // (se)
    relu_mask_(cache.g1, d_g1);                          // relu'
    std::vector<float> d_m;
    conv_same_backward(se1, cache.se1c, d_g1, d_m);      // (C)
    // time-mean adjoint: dx[c,t] += d_m[c]/L.
    for (int c = 0; c < C; ++c) {
        const float dmc = d_m[c] / static_cast<float>(L);
        for (int t = 0; t < L; ++t) dX[static_cast<std::size_t>(c) * L + t] += dmc;
    }
}

// ── Attentive statistics pooling ────────────────────────────────────────────

void asp_forward_train(const std::vector<float>& a, int agg, int T,
                       const QwenTtsSpkConv& asp_tdnn, const QwenTtsSpkConv& asp_conv,
                       std::vector<float>& pooled, AspCache& cache) {
    cache.agg = agg; cache.attn_ch = asp_tdnn.cout; cache.T = T; cache.a = a;
    // global mean / std per channel (uniform weights).
    cache.gmean.assign(agg, 0.0f); cache.gstd.assign(agg, 0.0f);
    cache.gvar.assign(agg, 0.0f);
    for (int c = 0; c < agg; ++c) {
        const float* row = a.data() + static_cast<std::size_t>(c) * T;
        double s = 0.0;
        for (int t = 0; t < T; ++t) s += row[t];
        const double mu = s / T;
        double var = 0.0;
        for (int t = 0; t < T; ++t) { const double d = row[t] - mu; var += d * d; }
        var /= T;
        cache.gmean[c] = static_cast<float>(mu);
        cache.gvar[c]  = static_cast<float>(var);
        cache.gstd[c]  = static_cast<float>(std::sqrt(std::max(var, 1e-12)));
    }
    // context = cat([a, gmean.expand, gstd.expand]) -> (3*agg, T).
    std::vector<float> ctx(static_cast<std::size_t>(3) * agg * T, 0.0f);
    std::copy(a.begin(), a.end(), ctx.begin());
    for (int c = 0; c < agg; ++c)
        for (int t = 0; t < T; ++t) {
            ctx[(static_cast<std::size_t>(agg) + c) * T + t] = cache.gmean[c];
            ctx[(static_cast<std::size_t>(2 * agg) + c) * T + t] = cache.gstd[c];
        }
    // attn = asp_conv(tanh(relu(asp_tdnn(ctx)))).
    conv_same_forward_train(asp_tdnn, ctx, T, cache.r, cache.tdnn_c);  // (attn_ch, T)
    for (float& v : cache.r) if (v < 0.0f) v = 0.0f;                   // relu
    cache.tanh_out.assign(cache.r.size(), 0.0f);
    for (std::size_t i = 0; i < cache.r.size(); ++i) cache.tanh_out[i] = std::tanh(cache.r[i]);
    std::vector<float> att;
    conv_same_forward_train(asp_conv, cache.tanh_out, T, att, cache.conv_c);  // (agg, T)
    // softmax over time per channel.
    cache.w.assign(static_cast<std::size_t>(agg) * T, 0.0f);
    for (int c = 0; c < agg; ++c) {
        float* row = att.data() + static_cast<std::size_t>(c) * T;
        float mx = row[0];
        for (int t = 1; t < T; ++t) mx = std::max(mx, row[t]);
        double sum = 0.0;
        for (int t = 0; t < T; ++t) { row[t] = std::exp(row[t] - mx); sum += row[t]; }
        const float inv = static_cast<float>(1.0 / sum);
        for (int t = 0; t < T; ++t) cache.w[static_cast<std::size_t>(c) * T + t] = row[t] * inv;
    }
    // attentive mean / std -> pooled (2*agg).
    cache.mu.assign(agg, 0.0f); cache.var.assign(agg, 0.0f);
    pooled.assign(static_cast<std::size_t>(2) * agg, 0.0f);
    for (int c = 0; c < agg; ++c) {
        const float* ar = a.data() + static_cast<std::size_t>(c) * T;
        const float* wr = cache.w.data() + static_cast<std::size_t>(c) * T;
        double mu = 0.0;
        for (int t = 0; t < T; ++t) mu += static_cast<double>(wr[t]) * ar[t];
        double var = 0.0;
        for (int t = 0; t < T; ++t) { const double d = ar[t] - mu; var += static_cast<double>(wr[t]) * d * d; }
        cache.mu[c]  = static_cast<float>(mu);
        cache.var[c] = static_cast<float>(var);
        pooled[c]       = static_cast<float>(mu);
        pooled[agg + c] = static_cast<float>(std::sqrt(std::max(var, 1e-12)));
    }
}

void asp_backward(const QwenTtsSpkConv& asp_tdnn, const QwenTtsSpkConv& asp_conv,
                  const AspCache& cache, const std::vector<float>& dPooled,
                  std::vector<float>& dA) {
    const int agg = cache.agg, T = cache.T;
    dA.assign(static_cast<std::size_t>(agg) * T, 0.0f);
    std::vector<float> d_w(static_cast<std::size_t>(agg) * T, 0.0f);
    // pooled -> attentive mean (mu) and std contributions to w and a.
    for (int c = 0; c < agg; ++c) {
        const float d_mu_out  = dPooled[c];
        const float std_out   = std::sqrt(std::max(cache.var[c], 1e-12f));
        const float d_var = (cache.var[c] > 1e-12f) ? dPooled[agg + c] * 0.5f / std_out : 0.0f;
        const float mu = cache.mu[c];
        for (int t = 0; t < T; ++t) {
            const std::size_t i = static_cast<std::size_t>(c) * T + t;
            const float a_ct = cache.a[i], w_ct = cache.w[i];
            const float diff = a_ct - mu;
            d_w[i] += d_mu_out * a_ct + d_var * diff * diff;         // mu + var explicit (w)
            dA[i]  += d_mu_out * w_ct + d_var * 2.0f * w_ct * diff;  // mu + var explicit (a)
        }
    }
    // softmax-over-time adjoint per channel: dscore = w·(d_w - Σ_t w·d_w).
    std::vector<float> d_score(static_cast<std::size_t>(agg) * T, 0.0f);
    for (int c = 0; c < agg; ++c) {
        double s = 0.0;
        for (int t = 0; t < T; ++t) {
            const std::size_t i = static_cast<std::size_t>(c) * T + t;
            s += static_cast<double>(cache.w[i]) * d_w[i];
        }
        for (int t = 0; t < T; ++t) {
            const std::size_t i = static_cast<std::size_t>(c) * T + t;
            d_score[i] = cache.w[i] * (d_w[i] - static_cast<float>(s));
        }
    }
    // asp_conv back -> tanh' -> relu' -> asp_tdnn back -> d_ctx (3*agg, T).
    std::vector<float> d_tanh;
    conv_same_backward(asp_conv, cache.conv_c, d_score, d_tanh);   // (attn_ch*T)
    std::vector<float> d_r(d_tanh.size(), 0.0f);
    for (std::size_t i = 0; i < d_tanh.size(); ++i)
        d_r[i] = d_tanh[i] * (1.0f - cache.tanh_out[i] * cache.tanh_out[i]);  // tanh'
    relu_mask_(cache.r, d_r);                                      // relu'
    std::vector<float> d_ctx;
    conv_same_backward(asp_tdnn, cache.tdnn_c, d_r, d_ctx);        // (3*agg*T)
    // context split: rows [0,agg) -> a directly; [agg,2agg) -> gmean; [2agg,3agg) -> gstd.
    for (int c = 0; c < agg; ++c)
        for (int t = 0; t < T; ++t)
            dA[static_cast<std::size_t>(c) * T + t] += d_ctx[static_cast<std::size_t>(c) * T + t];
    for (int c = 0; c < agg; ++c) {
        double dgmean = 0.0, dgstd = 0.0;
        for (int t = 0; t < T; ++t) {
            dgmean += d_ctx[(static_cast<std::size_t>(agg) + c) * T + t];
            dgstd  += d_ctx[(static_cast<std::size_t>(2 * agg) + c) * T + t];
        }
        const float d_gvar = (cache.gvar[c] > 1e-12f)
                                 ? static_cast<float>(dgstd) * 0.5f / cache.gstd[c] : 0.0f;
        const float gmean = cache.gmean[c];
        for (int t = 0; t < T; ++t) {
            const std::size_t i = static_cast<std::size_t>(c) * T + t;
            dA[i] += static_cast<float>(dgmean) / T;                       // gmean adjoint
            dA[i] += d_gvar * (2.0f / T) * (cache.a[i] - gmean);           // gstd adjoint
        }
    }
}

// ── full ECAPA encoder ──────────────────────────────────────────────────────

void ecapa_forward_train(const float* wav, int n, const bt::Tensor& hann,
                         const bt::Tensor& mel_basis, int n_fft, int hop, int win,
                         int mel_dim, const QwenTtsSpkConv& block0,
                         const std::vector<QwenTtsSpkSERes2>& blocks, int scale,
                         const QwenTtsSpkConv& mfa, const QwenTtsSpkConv& asp_tdnn,
                         const QwenTtsSpkConv& asp_conv, const QwenTtsSpkConv& fc,
                         std::vector<float>& embedding, EcapaCache& cache) {
    // mel -> (mel_dim, T)
    std::vector<float> x;
    mel_forward_train(wav, n, hann, mel_basis, n_fft, hop, win, mel_dim, x, cache.mel);
    const int T = cache.mel.frames;
    const int C = block0.cout;
    const int agg = mfa.cout;            // == 3*C
    cache.M = mel_dim; cache.T = T; cache.C = C; cache.agg = agg; cache.scale = scale;

    // initial TDNN + ReLU
    std::vector<float> hs;
    conv_same_forward_train(block0, x, T, hs, cache.block0);
    for (float& v : hs) if (v < 0.0f) v = 0.0f;
    cache.x0_relu = hs;

    // three SE-Res2Net blocks; collect outputs for aggregation.
    cache.blocks.assign(blocks.size(), EcapaBlockCache{});
    std::vector<std::vector<float>> outs;
    for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
        const QwenTtsSpkSERes2& blk = blocks[bi];
        EcapaBlockCache& bc = cache.blocks[bi];
        std::vector<float> y;
        conv_same_forward_train(blk.tdnn1, hs, T, y, bc.tdnn1);
        for (float& v : y) if (v < 0.0f) v = 0.0f;
        bc.a1_relu = y;
        std::vector<float> r;
        res2net_forward_train(blk.res2net, y, C, T, scale, r, bc.res2);
        std::vector<float> z;
        conv_same_forward_train(blk.tdnn2, r, T, z, bc.tdnn2);
        for (float& v : z) if (v < 0.0f) v = 0.0f;
        bc.a3_relu = z;
        std::vector<float> se_out;
        se_forward_train(blk.se1, blk.se2, z, C, T, se_out, bc.se);
        for (std::size_t j = 0; j < se_out.size(); ++j) se_out[j] += hs[j];  // residual
        hs = se_out;
        outs.push_back(hs);
    }

    // aggregate (cat 3 block outputs along channels) -> MFA + ReLU.
    std::vector<float> cat(static_cast<std::size_t>(agg) * T, 0.0f);
    for (std::size_t b = 0; b < outs.size(); ++b)
        std::copy(outs[b].begin(), outs[b].end(),
                  cat.begin() + static_cast<std::size_t>(b) * C * T);
    std::vector<float> a;
    conv_same_forward_train(mfa, cat, T, a, cache.mfa);
    for (float& v : a) if (v < 0.0f) v = 0.0f;

    // attentive statistics pooling -> pooled (2*agg).
    std::vector<float> pooled;
    asp_forward_train(a, agg, T, asp_tdnn, asp_conv, pooled, cache.asp);

    // final projection (2*agg -> enc_dim).
    conv_same_forward_train(fc, pooled, /*L=*/1, embedding, cache.fc);
}

void ecapa_backward(const QwenTtsSpkConv& block0,
                    const std::vector<QwenTtsSpkSERes2>& blocks, int scale,
                    const QwenTtsSpkConv& mfa, const QwenTtsSpkConv& asp_tdnn,
                    const QwenTtsSpkConv& asp_conv, const QwenTtsSpkConv& fc,
                    const bt::Tensor& hann, const bt::Tensor& mel_basis,
                    const EcapaCache& cache, const std::vector<float>& dEmbedding,
                    std::vector<float>& dWav) {
    (void)scale;   // res2net_backward reads scale from the cache; kept for API symmetry
    const int T = cache.T, C = cache.C;

    // fc back -> d_pooled (2*agg).
    std::vector<float> d_pooled;
    conv_same_backward(fc, cache.fc, dEmbedding, d_pooled);
    // attentive pooling back -> d_a (agg*T), gradient on relu(mfa) output.
    std::vector<float> d_a;
    asp_backward(asp_tdnn, asp_conv, cache.asp, d_pooled, d_a);
    // relu(mfa) back (mask = a>0, cached as asp.a) -> mfa back -> d_cat (agg*T).
    relu_mask_(cache.asp.a, d_a);
    std::vector<float> d_cat;
    conv_same_backward(mfa, cache.mfa, d_a, d_cat);

    // split d_cat into the three block-output gradients (C*T each).
    const int nblk = static_cast<int>(blocks.size());
    std::vector<std::vector<float>> g(nblk);
    for (int j = 0; j < nblk; ++j) {
        const float* p = d_cat.data() + static_cast<std::size_t>(j) * C * T;
        g[j].assign(p, p + static_cast<std::size_t>(C) * T);
    }

    // reverse the residual/sequential block stack.
    std::vector<float> d_x0;   // gradient on block0 relu output (hs_0)
    for (int j = nblk - 1; j >= 0; --j) {
        const QwenTtsSpkSERes2& blk = blocks[j];
        const EcapaBlockCache& bc = cache.blocks[j];
        const std::vector<float>& d_yfinal = g[j];   // grad on outs[j]
        // residual: outs[j] = se_out + hs_in -> d_se_out = d_yfinal, straight = d_yfinal.
        std::vector<float> d_y3;
        se_backward(blk.se1, blk.se2, bc.se, d_yfinal, d_y3);  // grad on relu(tdnn2) output
        relu_mask_(bc.a3_relu, d_y3);
        std::vector<float> d_y2;
        conv_same_backward(blk.tdnn2, bc.tdnn2, d_y3, d_y2);
        std::vector<float> d_y1;
        res2net_backward(blk.res2net, bc.res2, d_y2, d_y1);
        relu_mask_(bc.a1_relu, d_y1);
        std::vector<float> d_hsin_path;
        conv_same_backward(blk.tdnn1, bc.tdnn1, d_y1, d_hsin_path);
        // grad on hs_in = straight-through (d_yfinal) + conv path.
        std::vector<float> hsin(static_cast<std::size_t>(C) * T, 0.0f);
        for (std::size_t i = 0; i < hsin.size(); ++i) hsin[i] = d_yfinal[i] + d_hsin_path[i];
        if (j > 0)
            for (std::size_t i = 0; i < hsin.size(); ++i) g[j - 1][i] += hsin[i];
        else
            d_x0 = std::move(hsin);
    }

    // block0 relu back -> block0 conv back -> d_mel (M*T).
    relu_mask_(cache.x0_relu, d_x0);
    std::vector<float> d_mel;
    conv_same_backward(block0, cache.block0, d_x0, d_mel);

    // mel frontend back -> d(waveform).
    mel_backward(hann, mel_basis, cache.mel, d_mel, dWav);
}

}  // namespace st_detail
}  // namespace brosoundml
