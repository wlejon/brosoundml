#pragma once

// FastConformer / NEST encoder (internal). The acoustic backbone shared by the
// NeMo model family (Parakeet-TDT, streaming Sortformer). Composes the forward
// pass from brotensor ops; loaded from `model.safetensors` via load(). FP32 on
// whichever device the weights live on; every forward dispatches through
// brotensor device ops (CPU / CUDA / Metal).
//
// State-dict key scheme (HF-transformers Parakeet layout — the Sortformer
// converter remaps the native NeMo names onto this same scheme so one loader
// serves both):
//   encoder.subsampling.layers.{0,2,3,5,6}.{weight,bias}   (Conv2d stack)
//   encoder.subsampling.linear.{weight,bias}
//   encoder.layers.{i}.norm_feed_forward1/norm_self_att/norm_conv/
//                       norm_feed_forward2/norm_out.{weight,bias}
//   encoder.layers.{i}.feed_forward1/feed_forward2.linear1/linear2.{weight,bias}
//   encoder.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
//   encoder.layers.{i}.self_attn.relative_k_proj.weight
//   encoder.layers.{i}.self_attn.bias_u / bias_v
//   encoder.layers.{i}.conv.pointwise_conv1/depthwise_conv/pointwise_conv2.
//                       {weight,bias}
//   encoder.layers.{i}.conv.norm.{weight,bias,running_mean,running_var}
//
// The {bias} entries on the FFN / projection / conv weights are OPTIONAL: the
// HF Parakeet export omits them; the NeMo NEST checkpoint carries them
// (untie_biases). load() uses optional uploads and forward() applies a bias only
// when it is present.

#include "brosoundml/audio.h"
#include "brosoundml/fastconformer.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <string>
#include <vector>

namespace brosoundml {

// ─── One Conformer block ───────────────────────────────────────────────────
//
// Pre-norm residual topology (NeMo ConformerLayer):
//   x = x + 0.5 * ff1(LN_ff1(x))
//   x = x + attn(LN_att(x))          // Transformer-XL relative-position MHA
//   x = x + conv(LN_conv(x))         // GLU + depthwise + BatchNorm + SiLU
//   x = x + 0.5 * ff2(LN_ff2(x))
//   x = LN_out(x)
struct FastConformerBlock {
    // LayerNorms (gamma/beta each (C,1)).
    brotensor::Tensor n_ff1_w, n_ff1_b;
    brotensor::Tensor n_att_w, n_att_b;
    brotensor::Tensor n_conv_w, n_conv_b;
    brotensor::Tensor n_ff2_w, n_ff2_b;
    brotensor::Tensor n_out_w, n_out_b;

    // Macaron feed-forwards: linear2(silu(linear1(x))). Biases optional (empty
    // when the checkpoint omits them — HF Parakeet).
    brotensor::Tensor ff1_l1_w, ff1_l1_b, ff1_l2_w, ff1_l2_b;
    brotensor::Tensor ff2_l1_w, ff2_l1_b, ff2_l2_w, ff2_l2_b;

    // Relative-position self-attention. q/k/v/o projection biases optional. The
    // Transformer-XL learned per-head content/position biases pos_bias_u/v are
    // (num_heads, head_dim) flattened to (C,1); the q-projection bias (when
    // present) is folded into bias_u/bias_v at load so the rel-pos paths stay
    // single-add.
    brotensor::Tensor q_w, k_w, v_w, o_w;
    brotensor::Tensor k_b, v_b, o_b;    // (C,1) each, empty when absent
    brotensor::Tensor rel_k_w;          // relative_k_proj (C, C), no bias
    brotensor::Tensor bias_u, bias_v;   // (C, 1) each (q-bias folded in)

    // Convolution module. Conv biases optional.
    brotensor::Tensor pw1_w, pw1_b;     // pointwise_conv1 (2C, C, 1)
    brotensor::Tensor dw_w,  dw_b;      // depthwise_conv  (C, 1, k), groups=C
    brotensor::Tensor pw2_w, pw2_b;     // pointwise_conv2 (C, C, 1)
    brotensor::Tensor bn_w, bn_b;       // BatchNorm1d gamma/beta (C,1)
    brotensor::Tensor bn_mean, bn_var;  // running stats (C,1)

    // x: (T, C) in-place updated. pos_emb: (2T-1, C) relative positional
    // encoding for this T (shared across layers, built once per forward).
    // d_mask: optional length-T host key mask (1 valid / 0 pad) for self-
    // attention. conv_pad_mask: optional (C, T) device tensor (1 valid / 0 pad)
    // multiplied into the conv-module GLU output before the depthwise conv,
    // matching NeMo's masked_fill. Both null = no masking.
    void forward(brotensor::Tensor& x,
                 const FastConformerConfig& cfg,
                 const brotensor::Tensor& pos_emb,
                 const float* d_mask,
                 const brotensor::Tensor* conv_pad_mask) const;
};

// ─── FastConformer encoder ─────────────────────────────────────────────────
struct FastConformerEncoder {
    FastConformerConfig   cfg;
    brotensor::Device     device = brotensor::Device::CPU;

    // Mel front-end tables (host-resident; the STFT/filterbank run on the host
    // and the feature is uploaded to `device`).
    brotensor::Tensor mel_filter;       // (n_mels, n_fft/2+1) FP32 CPU
    brotensor::Tensor window;           // (1, win_length) FP32 CPU

    // Subsampling: 5 Conv2d weights (first full conv, then 2x dw+pw) + a final
    // Linear. Conv biases default-present (Conv2d bias=True); kept as tensors
    // (may be empty if a checkpoint omits them).
    brotensor::Tensor sub_conv_w[5], sub_conv_b[5];
    brotensor::Tensor sub_linear_w, sub_linear_b;

    std::vector<FastConformerBlock> layers;

    void load(const brotensor::safetensors::File& f,
              const FastConformerConfig& cfg,
              brotensor::Device device);

    // Host-side log-mel: (frames, n_mels) row-major FP32 (NeMo recipe). Sets
    // `frames`. Per-feature normalization is applied iff cfg.normalize_features.
    std::vector<float> log_mel(const AudioBuffer& audio, int& frames) const;

    // audio -> encoder hidden states (T_out, hidden_size) on `device`. Runs the
    // full graph: log-mel + subsampling pre-encode + Conformer stack. When
    // cfg.mask_padding is set, the trailing center-STFT pad frame is zeroed and
    // masked out of the valid frames.
    void forward(const AudioBuffer& audio, brotensor::Tensor& out) const;

    // Number of leading valid encoder frames for an input of `num_samples`
    // (= conv-subsampled (num_samples / hop_length)) when cfg.mask_padding, else
    // -1 (all frames valid). Used by the head to build its attention mask.
    int valid_output_frames(int num_samples) const;

    // ── Streaming split ──────────────────────────────────────────────────────
    // The streaming path (Sortformer AOSC) needs the pre-encode embeddings on
    // their own (they are what the speaker cache stores) and then runs the
    // Conformer stack over a concatenation [spkcache | fifo | chunk]. These two
    // entry points expose that split; forward() above is pre_encode+encode_layers
    // fused for the offline path.

    // mel (frames, n_mels) row-major host FP32 -> pre-encode embeddings
    // (T_out, hidden_size) on `device`, T_out = frames after 8x subsampling.
    void pre_encode(const std::vector<float>& mel, int frames,
                    brotensor::Tensor& out) const;

    // embs (T, hidden_size) on `device` -> Conformer-stack output (T,
    // hidden_size). Applies xscaling + the relative-position stack. valid_frames
    // (when >= 0 and < T) masks the trailing pad frames out of self-attention /
    // the conv module; -1 = all frames valid.
    void encode_layers(const brotensor::Tensor& embs, brotensor::Tensor& out,
                       int valid_frames = -1) const;
};

}  // namespace brosoundml
