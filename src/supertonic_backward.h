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

// ─── tanh-gated style cross-attention backward ───────────────────────────────
//
// Forward (supertonic.cpp style_attention), all channel-major:
//   q = pconv(query,   wq)  [inner,Lq]      k = pconv(key_src, wk)  [inner,S]
//   v = pconv(val_src, wv)  [inner,S]
//   per head (kSpteHeads=2, hd=inner/nh):
//     ktanh = tanh(kh)                                 # tanh GATE on keys
//     scores = (qhᵀ @ ktanh) * (1/16)  -> softmax over S keys   # scale 1/16
//     outh   = scores @ vhᵀ
//   y = pconv(concat_heads, wo)  [Cout,Lq]
// The KEY path is FROZEN (a fixed learned prototype): the backward threads
// through tanh+softmax w.r.t. scores to reach d(query), but does NOT emit
// d(key) — dKtanh is discarded. Two input grads are produced: d(query) (a field
// activation needing upstream grad) and d(value) (= style_ttl, the optimisation
// target). All weights FROZEN.

struct StyleAttnCache {
    PConvCache        qc, vc, oc;  // q / v / o projection conv geometry (key frozen)
    brotensor::Tensor q;           // (inner, Lq)  q projection output
    brotensor::Tensor ktanh;       // (inner, S)   tanh(k projection)
    brotensor::Tensor v;           // (inner, S)   v projection output
    brotensor::Tensor scores;      // (nh*Lq, S)   softmaxed weights, heads stacked
    int   nh = 0, hd = 0, Lq = 0, S = 0, inner = 0, Cq = 0, Ck = 0;
    float inv = 1.0f;              // attention scale (1/16)
};

//   query: (1, Cq*Lq).  key_src/val_src: (1, Ck*S).  y: (1, Cout*Lq), overwritten.
void style_attention_forward_train(const brotensor::Tensor& query,
                                   const brotensor::Tensor& key_src,
                                   const brotensor::Tensor& val_src,
                                   const StyleAttn& A, int Lq, int S,
                                   brotensor::Tensor& y, StyleAttnCache& cache);

//   dY: (1, Cout*Lq).  dQuery: (1, Cq*Lq), dValue: (1, Ck*S), both overwritten.
void style_attention_backward(const StyleAttn& A, const StyleAttnCache& cache,
                              const brotensor::Tensor& dY,
                              brotensor::Tensor& dQuery, brotensor::Tensor& dValue);

// ─── RoPE text cross-attention block backward ────────────────────────────────
//
// This atom is the WHOLE flow-field text-attention block (main_blocks %6==3),
// not just the bare attention: forward_train reproduces supertonic.cpp's
// rope_attention AND the residual + post-attention LayerNorm applied at its call
// site, so agent 3 can chain it as one unit. Forward (kVeC=512, kVeHeads=8,
// kVeHd=64, kVeHalf=32, scale 1/16), channel-major:
//   q = pconv(h, conv_q) [512,L]   k = pconv(text, conv_k) [512,T]   v = pconv(text, conv_v) [512,T]
//   per head: qr = RoPE(qh, posL), kr = RoPE(kh, posT)   # position p angle (p/len)*theta[f]
//             scores = (qrᵀ @ kr) * (1/16) -> softmax over T;  outh = scores @ vhᵀ
//   attn = pconv(concat_heads, conv_o) [512,L]
//   y    = LayerNorm_chan(attn + h, norm_g, norm_b)       # residual + post-attn LN
// key/value come from the FROZEN text encoding: only d(query)=d(h) is produced
// (dKr/dVhT discarded). RoPE is an orthogonal rotation, so its adjoint is the
// rotation by the negated angle (transpose of the rotation matrix), applied to
// the incoming q-gradient. LayerNorm affine FROZEN — dGamma/dBeta discarded.

struct RopeAttnCache {
    PConvCache        qc, oc;     // q / o projection conv geometry (k/v frozen)
    brotensor::Tensor qr;         // (inner, L)  RoPE'd q
    brotensor::Tensor kr;         // (inner, T)  RoPE'd k
    brotensor::Tensor v;          // (inner, T)  v projection output
    brotensor::Tensor scores;     // (nh*L, T)   softmaxed weights, heads stacked
    brotensor::Tensor cosL, sinL; // (half, L)   query-position RoPE tables (adjoint)
    brotensor::Tensor ln_xhat;    // (L, C)      post-attn LN normalised input
    brotensor::Tensor ln_rstd;    // (L, 1)      post-attn LN 1/sqrt(var+eps)
    int   nh = 0, hd = 0, half = 0, L = 0, T = 0, C = 0, inner = 0;
    float inv = 1.0f;             // attention scale (1/16)
};

//   h: (1, 512*L).  text_src: (1, Ck*T).  y: (1, 512*L), overwritten.
void rope_attention_forward_train(const brotensor::Tensor& h,
                                  const brotensor::Tensor& text_src,
                                  const VeRope& A, int L, int T,
                                  brotensor::Tensor& y, RopeAttnCache& cache);

//   dY: (1, 512*L).  dH: (1, 512*L), overwritten (= d(query)).
void rope_attention_backward(const VeRope& A, const RopeAttnCache& cache,
                             const brotensor::Tensor& dY, brotensor::Tensor& dH);

}  // namespace st_detail
}  // namespace brosoundml
