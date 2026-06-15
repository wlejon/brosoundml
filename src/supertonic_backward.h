#pragma once

// brosoundml/src/supertonic_backward.h — manual reverse-mode backward through
// the Supertonic-3 TTS forward atoms, for optimising an INPUT style matrix
// through a FROZEN model (there is no autograd in brotensor). Mirrors
// kokoro_decoder_backward.h: each forward atom has a `*_forward_train` that
// emits a Cache of the intermediates its matching `*_backward` consumes, and the
// backward threads gradients by composing the existing brotensor `*_backward`
// ops in reverse.
//
// This is the FOUNDATIONAL layer — the conv/transpose/convnext atoms the later
// agents (style_attention, the flow field, the vocoder, ECAPA) compose on top
// of. Every weight in these atoms is FROZEN: only the input gradient is
// threaded, no weight gradients are produced. Each atom is gradient-checked in
// isolation against finite differences (tests/test_supertonic_backward.cpp).
//
// Layout convention (matching supertonic.cpp's forward): a 1D activation is
// channel-major (1, C*L), element (c,l) at flat index c*L + l. The "sequence"
// layout used by LayerNorm is (L, C), row l column c.
//
// Private to the brosoundml build (lives under src/).

#include "supertonic_internal.h"

#include <brotensor/tensor.h>

namespace brosoundml {
namespace st_detail {

// ─── pconv / rconv input-backward ───────────────────────────────────────────
//
// supertonic's `pconv` is a 1-D conv with an explicit (pad_left, pad_right)
// pre-pad. It has two forward expressions, and the backward mirrors each:
//   • 1×1 / groups==1 / no-pad: forward is matmul(c.w, x) — a GEMM — so the
//     input grad is dX = c.wᵀ @ dY (via brotensor::matmul_backward; the cached
//     input is the discarded dA operand).
//   • otherwise: forward is pad1d_forward + conv1d, so the backward is
//     conv1d_backward_input + pad1d_backward.
// `rconv` is the causal left-replicate-pad variant (pad_left = dilation*(k-1),
// pad_right = 0, mode = 2); its backward IS pconv_backward — the same cache
// records the replicate-pad geometry, and pad1d_backward(mode=2) accumulates
// the padded-region gradients back onto the clamped edge sample.
//
// Weights are FROZEN: only dX is produced.

struct PConvCache {
    brotensor::Tensor x_in;          // matmul (1×1) path: cached input as [Cin, L]
    int  Cin = 0, L = 0, Cout = 0;   // forward shapes (L = unpadded input length)
    int  k = 1, dilation = 1, groups = 1;
    int  pad_left = 0, pad_right = 0, mode = 0;
    int  Lp = 0;                      // padded input length (conv path)
    int  L_out = 0;                   // output length
    bool matmul_path = false;
};

// Forward, identical math to supertonic.cpp's pconv, emitting the cache.
//   x: (1, Cin*L) channel-major.   y: (1, Cout*L_out), overwritten.
void pconv_forward_train(const brotensor::Tensor& x, const ConvW& c,
                         int Cin, int L, int dilation, int groups,
                         int pad_left, int pad_right, int mode,
                         brotensor::Tensor& y, PConvCache& cache);

// Causal left-replicate-pad pconv (the vocoder cached-conv offline behaviour).
//   x: (1, Cin*L).   y: (1, Cout*L), overwritten.
void rconv_forward_train(const brotensor::Tensor& x, const ConvW& c,
                         int Cin, int L, int dilation, int groups,
                         brotensor::Tensor& y, PConvCache& cache);

// Input-backward for pconv AND rconv (weights frozen).
//   dY: (1, Cout*L_out).   dX: (1, Cin*L), overwritten.
void pconv_backward(const ConvW& c, const PConvCache& cache,
                    const brotensor::Tensor& dY, brotensor::Tensor& dX);

// ─── seq transpose (host adjoint) ───────────────────────────────────────────
//
// supertonic's `transpose2d(x, R, C)` is nchw_to_sequence(flat(x),1,R,1,C): a
// pure [R×C] -> [C×R] permutation. Its adjoint is the inverse permutation,
// sequence_to_nchw with the same dims. Both the nchw_to_sequence /
// sequence_to_nchw adjoints used inside convnext are this same routing.

// y = xᵀ.   x: (R, C) (or (1, R*C) channel-major).   y: (C, R), overwritten.
void transpose2d_forward(const brotensor::Tensor& x, int R, int C,
                         brotensor::Tensor& y);

// dX = dYᵀ.   dY: (C, R).   dX: (R, C), overwritten.
void transpose2d_backward(const brotensor::Tensor& dY, int R, int C,
                          brotensor::Tensor& dX);

// ─── ConvNeXt-1D block backward ─────────────────────────────────────────────
//
// Forward (supertonic.cpp convnext_block), all over channel-major [C,L]:
//   dwy  = pconv(h, dw, groups=C, edge-pad 2*dil)        # depthwise
//   seq  = nchw_to_sequence(dwy)                          # [L,C]
//   seqn = layernorm(seq) over C  (eps 1e-6)
//   nrm  = sequence_to_nchw(seqn)                         # [C,L]
//   a    = pconv(nrm, pw1)                                # 1×1
//   ga   = gelu_exact(a)
//   yc   = pconv(ga, pw2)                                 # 1×1 (LayerScale folded)
//   y    = yc + h                                         # residual
// Backward threads in reverse: residual split, pw2 back, gelu back, pw1 back,
// transpose back, layernorm_backward, transpose back, depthwise pconv back.
// All conv / LN weights FROZEN — only dX is produced.

struct ConvNeXtCache {
    PConvCache        dw;       // depthwise conv geometry
    brotensor::Tensor ln_xhat;  // (L, C) layernorm normalised-input cache
    brotensor::Tensor ln_rstd;  // (L, 1) layernorm 1/sqrt(var+eps) cache
    PConvCache        pw1;      // 1×1 expand
    brotensor::Tensor a;        // pw1 output = gelu input, (1, pw1.cout*L)
    PConvCache        pw2;      // 1×1 project (gamma folded)
    int C = 0, L = 0, dil = 1;
};

//   h: (1, C*L).   y: (1, C*L), overwritten.
void convnext_block_forward_train(const brotensor::Tensor& h,
                                  const ConvNeXtBlock& blk,
                                  int C, int L, int dil,
                                  brotensor::Tensor& y, ConvNeXtCache& cache);

//   dY: (1, C*L).   dX: (1, C*L), overwritten.
void convnext_block_backward(const ConvNeXtBlock& blk, const ConvNeXtCache& cache,
                             const brotensor::Tensor& dY, brotensor::Tensor& dX);

}  // namespace st_detail
}  // namespace brosoundml
