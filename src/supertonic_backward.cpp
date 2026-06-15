#include "supertonic_backward.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

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

}  // namespace st_detail
}  // namespace brosoundml
