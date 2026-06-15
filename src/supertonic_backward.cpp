#include "supertonic_backward.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

namespace brosoundml {
namespace st_detail {

namespace bt = brotensor;

namespace {

// Flat (1, rows*cols) view — the NCHW/pad ops want X shaped (N, C*H*W).
bt::Tensor flat(const bt::Tensor& t) {
    return bt::Tensor::view(t.device, t.data, 1, t.rows * t.cols, t.dtype);
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

}  // namespace st_detail
}  // namespace brosoundml
