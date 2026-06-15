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

// ─── flow-field (vector-estimator) single-step backward ──────────────────────
//
// Assembles the per-op atoms above into the WHOLE conditional vector-estimator
// pass — the body of supertonic.cpp's run_field — and threads the running
// hidden-gradient in reverse through all 24 VeBlocks, harvesting the style
// (value) gradient at every VeStyle block. Every weight is FROZEN; the ONLY
// gradient that leaves field_backward is d(style_val), accumulated (summed) over
// all style blocks, in the channel-major [256, S] layout the forward consumed
// (the caller adjoint-transposes it back to token-major [50,256] style_ttl).
//
// Block handling, exactly mirroring run_field (see supertonic.cpp):
//   • type 0: blk.conv.size() ConvNeXt sub-blocks in sequence, dilation 1<<s.
//   • type 1 FiLM: ADDITIVE-ONLY (h += time_emb@W + b, broadcast over L) — the
//     bias is independent of h, so its backward is the IDENTITY on dH and it
//     contributes nothing to any gradient. No cache, no work in the reverse pass.
//   • type 2 rope: the rope_attention atom (attention + residual + post-attn LN
//     already bundled) — rope_attention_backward gives d(h) only (text frozen).
//   • type 3 style: style_attention + residual + post-attn channel LayerNorm at
//     the call site. Backward = LN back -> dR; residual splits dR into the
//     style-attention output grad and the straight-through term; style_attention
//     _backward yields d(query) [folded back into dH via the residual: dH =
//     dQuery + dR] and d(value) [accumulated into the style gradient].
//
// proj_in (the 144->512 conv ahead of run_field) is NOT part of this unit: it is
// driven by the fixed noisy latent in field_step, so the gradient into the field
// input h0 is discarded — field_backward stops at h0.

// Per-VeBlock sub-caches, tagged by block type (matches VeBlock::type).
struct FieldBlockCache {
    int                        type = 0;
    std::vector<ConvNeXtCache> conv;     // type 0: one cache per ConvNeXt sub-block
    RopeAttnCache              rope;      // type 2
    StyleAttnCache             style;     // type 3: the style cross-attention
    brotensor::Tensor          ln_xhat;   // type 3: post-attn LN normalised input (L,C)
    brotensor::Tensor          ln_rstd;   // type 3: post-attn LN 1/sqrt(var+eps)  (L,1)
};

// Everything field_backward needs to replay the conditional run_field in reverse.
struct FieldCache {
    std::vector<FieldBlockCache> blocks;    // one per VeBlock, in forward order
    std::vector<ConvNeXtCache>   last;       // ve_last ConvNeXt blocks (dil = 1)
    PConvCache                   proj_out;   // ve_proj_out (512 -> 144, 1×1)
    int C = 0, L = 0, S = 0;                 // field channels / frames / style tokens
};

// Conditional run_field forward (identical math), emitting the FieldCache.
//   h0:        (1, 512*L)  field hidden after proj_in.
//   time_emb:  (1, 64)     frozen time embedding (FiLM input).
//   text_src:  (1, 256*T)  frozen text encoding (rope k/v source).
//   style_key: (1, 256*S)  frozen learned style prototype (attn key source).
//   style_val: (1, 256*S)  the style matrix being optimised (attn value source).
//   onesL:     (1, L)      row of ones (FiLM broadcast across L positions).
//   field_out: (1, 144*L)  velocity field, overwritten.
void field_forward_train(const brotensor::Tensor& h0,
                         const brotensor::Tensor& time_emb,
                         const brotensor::Tensor& text_src,
                         const brotensor::Tensor& style_key,
                         const brotensor::Tensor& style_val,
                         int L, int T, int S,
                         const std::vector<VeBlock>& blocks,
                         const std::vector<ConvNeXtBlock>& ve_last,
                         const ConvW& ve_proj_out,
                         const brotensor::Tensor& onesL,
                         brotensor::Tensor& field_out, FieldCache& cache);

// Reverse pass: thread dY back to the accumulated style-value gradient.
//   dY:        (1, 144*L)  gradient on the field output.
//   dStyleVal: (1, 256*S)  d(style_val), summed over all VeStyle blocks,
//                          channel-major [256, S]; overwritten.
// The gradient into h0 (noise/time/text path) is discarded.
void field_backward(const std::vector<VeBlock>& blocks,
                    const std::vector<ConvNeXtBlock>& ve_last,
                    const ConvW& ve_proj_out,
                    const FieldCache& cache,
                    const brotensor::Tensor& dY,
                    brotensor::Tensor& dStyleVal);

// ─── vocoder (autoencoder decoder) backward ──────────────────────────────────
//
// Forward (supertonic.cpp decode): latent [D*CC, frames] -> waveform.
//   1. de-chunk + de-normalise (host): channel (d*CC+j) at frame t lands at
//      de-chunked position t*CC+j; value = v*inv_scale*std[d] + mean[d]. Gives
//      dn [D, LF] (LF = CC*frames).
//   2. conv_in: causal rconv (D -> C), kernel 7, dilation 1.
//   3. `blocks.size()` ConvNeXt blocks, each CAUSAL (the depthwise + pointwise
//      use rconv, not the symmetric-pad pconv of the field/encoder blocks) — so
//      vocoder_convnext_forward_train fills the SAME ConvNeXtCache the shared
//      convnext_block_backward consumes (pconv_backward replays either pad type
//      from the cache, so the backward needs no vocoder-specific variant).
//   4. final BatchNorm in INFERENCE mode (frozen running stats): a per-channel
//      affine, so its adjoint is the diagonal scale gamma/sqrt(var+eps) — NOT
//      batch_norm_backward (that is the training-mode gradient with the
//      cross-sample mean/var terms, wrong for frozen stats).
//   5. head1 (rconv k3) -> leaky_relu(slope) -> head2 (rconv k1) -> [BC, LF].
//   6. output interleave (host): waveform[s*BC+c] = out[c*LF+s].
// All weights FROZEN; only d(latent) is produced.

struct VocoderCache {
    int D = 0, CC = 0, frames = 0, LF = 0, C = 0, BC = 0;
    PConvCache                 conv_in;
    std::vector<ConvNeXtCache> blocks;   // causal ConvNeXt blocks (rconv depthwise)
    brotensor::Tensor          bn_scale; // (C,1) gamma/sqrt(var+eps), BN-inference adjoint
    PConvCache                 head1;    // rconv k3
    brotensor::Tensor          head1_out;// leaky_relu input (rconv head1 output) (1, h1c*LF)
    PConvCache                 head2;    // rconv k1
};

// One causal (rconv-depthwise) ConvNeXt block, filling a ConvNeXtCache that the
// shared convnext_block_backward reverses unchanged.
//   h: (1, C*L).  y: (1, C*L), overwritten.
void vocoder_convnext_forward_train(const brotensor::Tensor& h,
                                    const ConvNeXtBlock& blk, int C, int L, int dil,
                                    brotensor::Tensor& y, ConvNeXtCache& cache);

// Decoder forward (identical math to decode), emitting the VocoderCache.
//   latent_in: (1, D*CC*frames) channel-major (the field/denoise output).
//   wave:      (1, BC*LF), overwritten.
void vocoder_decode_forward_train(const brotensor::Tensor& latent_in,
                                  int D, int CC, int frames, float inv_scale,
                                  const std::vector<float>& latent_mean,
                                  const std::vector<float>& latent_std,
                                  const ConvW& conv_in,
                                  const std::vector<ConvNeXtBlock>& blocks,
                                  const brotensor::Tensor& bn_g,
                                  const brotensor::Tensor& bn_b,
                                  const brotensor::Tensor& bn_mean,
                                  const brotensor::Tensor& bn_var, float bn_eps,
                                  const ConvW& head1, float prelu_slope,
                                  const ConvW& head2,
                                  brotensor::Tensor& wave, VocoderCache& cache);

// Reverse pass.
//   dWave:   (1, BC*LF)        gradient on the output waveform.
//   dLatent: (1, D*CC*frames)  gradient on the decoder input; overwritten.
void vocoder_decode_backward(const ConvW& conv_in,
                             const std::vector<ConvNeXtBlock>& blocks,
                             const ConvW& head1, float prelu_slope,
                             const ConvW& head2, float inv_scale,
                             const std::vector<float>& latent_std,
                             const VocoderCache& cache,
                             const brotensor::Tensor& dWave,
                             brotensor::Tensor& dLatent);

}  // namespace st_detail
}  // namespace brosoundml
