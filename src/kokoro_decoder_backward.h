#pragma once

// brosoundml/src/kokoro_decoder_backward.h — manual reverse-mode backward
// through the Kokoro iSTFTNet decoder back half, for training a LoRA (or any
// fine-tune) over the decoder. There is no autograd tape in brotensor, so
// (like the LSTM) each forward atom emits the intermediates its matching
// backward needs, and the backward threads gradients by calling the existing
// *_backward ops in reverse.
//
// Private to the brosoundml build (lives under src/). The atoms are assembled
// bottom-up — AdaIN affine, then the residual blocks, the spectral head, and
// finally the backbone + generator — and each is gradient-checked in isolation
// against finite differences before the next is wired on top.
//
// Layout convention: the decoder back half runs with N = 1, channel-major NCL
// (1, C*L), element (c,l) at flat index c*L + l. The "sequence" layout used by
// modulate is (L, C), row l, column c.

#include <brotensor/tensor.h>

#include "brosoundml/kokoro_modules.h"   // AdainResBlk1dWeights, AdaINResBlock1Weights, DecoderBackbone

#include <array>
#include <vector>

namespace brosoundml {
namespace detail {

// ─── AdaIN affine (instance-norm + style-conditioned scale/shift) ───────────
//
// Forward: y = (1 + gamma) * InstanceNorm(x) + beta, channel-wise, with the
// affine (gamma, beta) supplied directly (the fc that produces them from the
// style vector — and the LoRA that rides on it — sit one level up). Instance
// norm is GroupNorm with num_groups == C. Caches `x_seq` (the normalised input
// in (L, C) sequence layout) for the backward.
//   x_ncl:(1, C*L)   gamma,beta:(C,1)   ->   y_ncl:(1, C*L)
void adain_affine_forward(const brotensor::Tensor& x_ncl,
                          const brotensor::Tensor& gamma,
                          const brotensor::Tensor& beta,
                          int C, int L, float eps,
                          brotensor::Tensor& y_ncl,
                          brotensor::Tensor& x_seq_cache);

// Backward of adain_affine_forward.
//   dY_ncl:(1, C*L)  ->  dX_ncl:(1, C*L) overwritten,
//                        dGamma,dBeta:(C,1) overwritten.
// `x_ncl` (the forward input) and `x_seq_cache` (from the forward) are read;
// GroupNorm's mean/rstd are recomputed from x_ncl (no extra cache).
void adain_affine_backward(const brotensor::Tensor& x_ncl,
                           const brotensor::Tensor& x_seq_cache,
                           const brotensor::Tensor& gamma,
                           const brotensor::Tensor& dY_ncl,
                           int C, int L, float eps,
                           brotensor::Tensor& dX_ncl,
                           brotensor::Tensor& dGamma,
                           brotensor::Tensor& dBeta);

// ─── AdainResBlk1d (StyleTTS2 / iSTFTNet decoder residual block) ─────────────
//
// Forward:  residual = conv2(lrelu(adain2(conv1(pool(lrelu(adain1(x,s)))), s)))
//           shortcut = (upsample ? nearest_2x(x) : x) -> conv1x1 if learned_sc
//           y = (residual + shortcut) / sqrt(2)
// `pool` is identity (non-upsampling) or a depthwise ConvTranspose1d (upsample).
//
// The two AdaIN affines are the LoRA injection points, so their (gamma, beta)
// are supplied directly (the fc + LoRA sit one level up) and the block emits
// dGamma1/dBeta1 (C_in) and dGamma2/dBeta2 (C_out). The conv / pool / shortcut
// weights are FROZEN — only conv input-backward is threaded, no weight grads.

// Intermediates the backward needs (mirrors the LSTM forward-cache pattern).
struct AdainResBlkCache {
    brotensor::Tensor x;          // block input (1, C_in*L_in)
    brotensor::Tensor x_seq1;     // adain1 normalised-sequence cache
    brotensor::Tensor r_pre;      // adain1 output, pre-leaky (1, C_in*L_in)
    brotensor::Tensor conv1_out;  // conv1 output = adain2 input (1, C_out*L_out)
    brotensor::Tensor x_seq2;     // adain2 normalised-sequence cache
    brotensor::Tensor n2_pre;     // adain2 output, pre-leaky (1, C_out*L_out)
    int L_in = 0;
    int L_out = 0;
};

void adain_resblk_1d_forward_train(const AdainResBlk1dWeights& w,
                                   const brotensor::Tensor& x, int L_in,
                                   const brotensor::Tensor& gamma1,
                                   const brotensor::Tensor& beta1,
                                   const brotensor::Tensor& gamma2,
                                   const brotensor::Tensor& beta2,
                                   float eps,
                                   int& L_out, brotensor::Tensor& y,
                                   AdainResBlkCache& cache);

void adain_resblk_1d_backward(const AdainResBlk1dWeights& w,
                              const AdainResBlkCache& cache,
                              const brotensor::Tensor& gamma1,
                              const brotensor::Tensor& gamma2,
                              const brotensor::Tensor& dY, float eps,
                              brotensor::Tensor& dX,
                              brotensor::Tensor& dGamma1,
                              brotensor::Tensor& dBeta1,
                              brotensor::Tensor& dGamma2,
                              brotensor::Tensor& dBeta2);

// ─── AdaINResBlock1 (Generator residual block, Snake1D) ─────────────────────
//
// Three chained sub-blocks; for i in 0..2:
//   xt = snake(adain1[i](x,s), alpha1[i]); xt = convs1[i](xt)   # dilated
//   xt = snake(adain2[i](xt,s), alpha2[i]); xt = convs2[i](xt)  # dilation 1
//   x  = xt + x
// C and L are preserved throughout. The six AdaIN affines are the LoRA
// injection points (gamma/beta supplied directly); the alphas and conv weights
// are frozen (snake's dAlpha and the conv weight grads are discarded). The
// block emits a (gamma,beta) grad pair per affine — arrays of 3.

struct AdainResBlock1Cache {
    std::array<brotensor::Tensor, 3> x_in;    // adain1 input per sub-block
    std::array<brotensor::Tensor, 3> x_seq1;  // adain1 norm cache
    std::array<brotensor::Tensor, 3> s1_pre;  // adain1 output = snake1 input
    std::array<brotensor::Tensor, 3> c1_out;  // conv1 output = adain2 input
    std::array<brotensor::Tensor, 3> x_seq2;  // adain2 norm cache
    std::array<brotensor::Tensor, 3> s2_pre;  // adain2 output = snake2 input
    int C = 0;
    int L = 0;
};

void adain_resblock1_forward_train(const AdaINResBlock1Weights& w,
                                   const brotensor::Tensor& x_in, int C, int L,
                                   const std::array<brotensor::Tensor, 3>& g1,
                                   const std::array<brotensor::Tensor, 3>& b1,
                                   const std::array<brotensor::Tensor, 3>& g2,
                                   const std::array<brotensor::Tensor, 3>& b2,
                                   brotensor::Tensor& x_out,
                                   AdainResBlock1Cache& cache);

void adain_resblock1_backward(const AdaINResBlock1Weights& w,
                              const AdainResBlock1Cache& cache,
                              const std::array<brotensor::Tensor, 3>& g1,
                              const std::array<brotensor::Tensor, 3>& g2,
                              const brotensor::Tensor& dX_out,
                              brotensor::Tensor& dX_in,
                              std::array<brotensor::Tensor, 3>& dG1,
                              std::array<brotensor::Tensor, 3>& dB1,
                              std::array<brotensor::Tensor, 3>& dG2,
                              std::array<brotensor::Tensor, 3>& dB2);

// ─── Spectral head (conv_post -> polar exp/sin -> iSTFT) ────────────────────
//
// Forward:  x = leaky(x, 0.01); post = conv_post(x);
//           mag = exp(post[:n_freq]); ang = sin(post[n_freq:]);
//           audio = iSTFT(complex_from_polar(mag, ang), window)
// (post is (1, (n_fft+2)*L) channel-major; the first/second n_freq channels are
// the log-magnitude / pre-sin phase.) conv_post is FROZEN, so the head has no
// param grad — it only propagates dAudio back to dX (the generator-stack
// output). complex_from_polar has no device backward and there is no sin op, so
// that adjoint runs on host, mirroring the forward's own host round-trip.

struct SpectralHeadCache {
    brotensor::Tensor x_pre;   // input to the final leaky_relu (1, C*L)
    brotensor::Tensor post;    // conv_post output (1, (n_fft+2)*L)
    int C = 0;
    int L = 0;
};

void spectral_head_forward_train(const Conv1d& conv_post,
                                 const brotensor::Tensor& x, int C, int L,
                                 const brotensor::Tensor& window,
                                 int n_fft, int hop, int win,
                                 brotensor::Tensor& audio,
                                 SpectralHeadCache& cache);

void spectral_head_backward(const Conv1d& conv_post,
                            const SpectralHeadCache& cache,
                            const brotensor::Tensor& window,
                            int n_fft, int hop, int win,
                            const brotensor::Tensor& dAudio,
                            brotensor::Tensor& dX);

// ─── DecoderBackbone assembly (everything up to the Generator) ──────────────
//
// Reverses DecoderBackbone::forward (kokoro_modules.cpp): the frozen front
// convs (F0_conv, N_conv, asr_res) and the [asr, F0_dn, N_dn] inputs are
// constant w.r.t. the trainable params, so the only graph edges that matter
// are the AdaIN affines inside the `encode` block and each `decode` block.
// There are two affines (gamma/beta) per block — affine1 at channels_in,
// affine2 at channels_out — and the LoRA that produces them rides one level up
// (the DecoderLora class supplies the (gamma, beta) and consumes their grads).
//
// The decode loop concatenates [x, asr_res, F0_dn, N_dn] along channels before
// each block while `res` is live (i.e. before the first upsample). On the
// backward, that block's dX therefore carries grads for all four cat parts;
// only the leading x channels flow to the previous block — the asr_res / F0 / N
// channels are grads w.r.t. frozen inputs and are dropped.

// One block's (gamma, beta) affine pair for both sub-AdaINs. Reused as the
// grad container on the backward (g1 := dGamma1, etc.).
struct BlockAffines {
    brotensor::Tensor g1, b1;   // affine1 — channels_in
    brotensor::Tensor g2, b2;   // affine2 — channels_out
};

// Intermediates the backbone backward needs: the per-block residual-block
// caches plus the cat bookkeeping (whether each decode block was fed the
// concatenated input, and the running x channel count to slice back out).
struct BackboneCache {
    AdainResBlkCache              encode;
    std::vector<AdainResBlkCache> decode;
    std::vector<char>            res_applied;  // per decode block: was [x,asr,F0,N] cat'd in
    std::vector<int>             x_channels;   // running x channels before that block's cat
    int T = 0;
};

// affines[0] is the encode block; affines[1 + i] is decode[i]. `eps` is the
// instance-norm epsilon shared by both AdaINs (Kokoro uses 1e-5 throughout).
void decoder_backbone_forward_train(const DecoderBackbone& bb,
                                    const brotensor::Tensor& asr,
                                    const brotensor::Tensor& F0_pred,
                                    const brotensor::Tensor& N_pred,
                                    int T,
                                    const std::vector<BlockAffines>& affines,
                                    float eps,
                                    brotensor::Tensor& gen_in,
                                    BackboneCache& cache);

// dAffines is sized n_blocks (= 1 + decode.size()) and filled with the
// (dGamma1, dBeta1, dGamma2, dBeta2) for each block, same indexing as `affines`.
// Grads w.r.t. asr / F0 / N / style are not produced (frozen).
void decoder_backbone_backward(const DecoderBackbone& bb,
                               const BackboneCache& cache,
                               const std::vector<BlockAffines>& affines,
                               const brotensor::Tensor& dGenIn,
                               float eps,
                               std::vector<BlockAffines>& dAffines);

// ─── Generator assembly (upsample loop → spectral head → audio) ─────────────
//
// Reverses Generator::forward (kokoro_modules.cpp). Per upsample stage i:
//   a   = leaky(x_in, 0.1)
//   u   = convT(ups[i], a)            [+ left-reflect-pad 1 on the final stage]
//   src = noise_res[i](noise_convs[i](har), s)       # AdaINResBlock1, LoRA
//   m   = u + src
//   x   = mean_j resblocks[i*K + j](m, s)            # K AdaINResBlock1, LoRA
// then the trailing spectral head (leaky 0.01 → conv_post → polar exp/sin →
// iSTFT) — the already-verified spectral_head atom.
//
// The trainable params are the AdaIN affines inside every AdaINResBlock1
// (noise_res[i] and resblocks[i*K+j]); ups / noise_convs / conv_post and `har`
// are frozen. The noise branch's dX terminates at the frozen noise_convs (har
// is constant w.r.t. the trainable params), so only noise_res's affine grads
// survive; dX threads back to the previous stage through the convT + leaky.

// All six AdaIN affines of one AdaINResBlock1 (adain1[0..2], adain2[0..2]).
// Reused as the grad container on the backward.
struct ResBlock1Affines {
    std::array<brotensor::Tensor, 3> g1, b1;   // adain1[0..2]
    std::array<brotensor::Tensor, 3> g2, b2;   // adain2[0..2]
};

struct GenStageCache {
    brotensor::Tensor               x_in;    // stage input, pre-leaky (1, C_in*L_in)
    AdainResBlock1Cache             noise;   // noise_res[i]
    std::vector<AdainResBlock1Cache> res;    // resblocks[i*K .. i*K+K-1]
    int C_in = 0, L_in = 0;                  // before convT
    int C_out = 0, L_out = 0;                // after convT (+ final pad)
};

struct GeneratorCache {
    std::vector<GenStageCache> stages;       // num_upsamples
    SpectralHeadCache          head;
    brotensor::Tensor          window;       // Hann window reused by the head backward
};

// noise_aff[i] feeds noise_res[i]; res_aff[i*K + j] feeds resblocks[i*K + j]
// (K = num_kernels). har: (1, C_har*frames). gen_in: (1, init_C*L_in).
void generator_forward_train(const Generator& gen,
                             const brotensor::Tensor& gen_in, int L_in,
                             const brotensor::Tensor& har, int frames,
                             const std::vector<ResBlock1Affines>& noise_aff,
                             const std::vector<ResBlock1Affines>& res_aff,
                             brotensor::Tensor& audio,
                             GeneratorCache& cache);

// dNoise_aff / dRes_aff are sized to match noise_aff / res_aff and filled with
// the six (dGamma, dBeta) pairs per block. `dGenIn` receives the grad into the
// generator input — the backbone output — which threads the loss back to the
// backbone's affines (the two assemblies run in series through gen_in). Grads
// w.r.t. har / style are not produced (frozen).
void generator_backward(const Generator& gen,
                        const GeneratorCache& cache,
                        const std::vector<ResBlock1Affines>& noise_aff,
                        const std::vector<ResBlock1Affines>& res_aff,
                        const brotensor::Tensor& dAudio,
                        brotensor::Tensor& dGenIn,
                        std::vector<ResBlock1Affines>& dNoise_aff,
                        std::vector<ResBlock1Affines>& dRes_aff);

}  // namespace detail
}  // namespace brosoundml
