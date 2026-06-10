#pragma once

// Parakeet-specific submodules (internal). One step up from the generic
// nn-modules: these structs know the HF Parakeet state-dict key naming and the
// FastConformer / TDT topology. Loaded from `model.safetensors` via load().
// FP32 on whichever device the weights live on; every forward dispatches
// through brotensor device ops (CPU / CUDA / Metal).
//
// HF key scheme (transformers ParakeetForTDT):
//   encoder.subsampling.layers.{0,2,3,5,6}.{weight,bias}   (Conv2d stack)
//   encoder.subsampling.linear.{weight,bias}
//   encoder.layers.{i}.norm_feed_forward1/norm_self_att/norm_conv/
//                       norm_feed_forward2/norm_out.{weight,bias}
//   encoder.layers.{i}.feed_forward1/feed_forward2.linear1/linear2.weight
//   encoder.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//   encoder.layers.{i}.self_attn.relative_k_proj.weight
//   encoder.layers.{i}.self_attn.bias_u / bias_v
//   encoder.layers.{i}.conv.pointwise_conv1/depthwise_conv/pointwise_conv2.weight
//   encoder.layers.{i}.conv.norm.{weight,bias,running_mean,running_var}
//   encoder_projector.{weight,bias}
//   decoder.embedding.weight
//   decoder.lstm.{weight_ih,weight_hh,bias_ih,bias_hh}_l{0,1}
//   decoder.decoder_projector.{weight,bias}
//   joint.head.{weight,bias}

#include "brosoundml/audio.h"
#include "brosoundml/parakeet.h"

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
struct ParakeetConformerBlock {
    // LayerNorms (gamma/beta each (C,1)).
    brotensor::Tensor n_ff1_w, n_ff1_b;
    brotensor::Tensor n_att_w, n_att_b;
    brotensor::Tensor n_conv_w, n_conv_b;
    brotensor::Tensor n_ff2_w, n_ff2_b;
    brotensor::Tensor n_out_w, n_out_b;

    // Macaron feed-forwards: linear2(silu(linear1(x))). No bias.
    brotensor::Tensor ff1_l1_w, ff1_l2_w;
    brotensor::Tensor ff2_l1_w, ff2_l2_w;

    // Relative-position self-attention. No projection bias; bias_u / bias_v are
    // the Transformer-XL learned per-head content/position biases (num_heads,
    // head_dim) flattened to (D,1).
    brotensor::Tensor q_w, k_w, v_w, o_w;
    brotensor::Tensor rel_k_w;          // relative_k_proj (D, D), no bias
    brotensor::Tensor bias_u, bias_v;   // (D, 1) each

    // Convolution module. No conv bias.
    brotensor::Tensor pw1_w;            // pointwise_conv1 (2C, C, 1)
    brotensor::Tensor dw_w;             // depthwise_conv  (C, 1, k), groups=C
    brotensor::Tensor pw2_w;            // pointwise_conv2 (C, C, 1)
    brotensor::Tensor bn_w, bn_b;       // BatchNorm1d gamma/beta (C,1)
    brotensor::Tensor bn_mean, bn_var;  // running stats (C,1)

    // x: (T, C) in-place updated. pos_emb: (2T-1, C) relative positional
    // encoding for this T (shared across layers, built once per forward).
    void forward(brotensor::Tensor& x,
                 const ParakeetEncoderConfig& cfg,
                 const brotensor::Tensor& pos_emb) const;
};

// ─── FastConformer encoder ─────────────────────────────────────────────────
struct ParakeetEncoder {
    ParakeetEncoderConfig cfg;
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

    std::vector<ParakeetConformerBlock> layers;

    void load(const brotensor::safetensors::File& f,
              const ParakeetEncoderConfig& cfg,
              brotensor::Device device);

    // Host-side log-mel: (frames, n_mels) row-major FP32 (NeMo recipe). Sets
    // `frames`.
    std::vector<float> log_mel(const AudioBuffer& audio, int& frames) const;

    // audio -> encoder hidden states (T_out, hidden_size) on `device`.
    void forward(const AudioBuffer& audio, brotensor::Tensor& out) const;
};

// ─── TDT prediction network (RNN-T decoder) ────────────────────────────────
//
// Token embedding -> num_decoder_layers LSTM -> decoder_projector. Stateful:
// the greedy decoder steps it one emitted token at a time, carrying (h,c) per
// layer. PyTorch nn.LSTM gate order i,f,g,o.
struct ParakeetPrediction {
    int hidden    = 0;
    int n_layers  = 0;

    brotensor::Tensor embedding;                 // (vocab, hidden)
    std::vector<brotensor::Tensor> w_ih, w_hh;   // (4*hidden, in/hidden) per layer
    std::vector<brotensor::Tensor> b_ih, b_hh;   // (4*hidden, 1) per layer
    brotensor::Tensor proj_w, proj_b;            // decoder_projector (hidden,hidden)

    brotensor::Device device = brotensor::Device::CPU;

    void load(const brotensor::safetensors::File& f,
              const ParakeetConfig& cfg, brotensor::Device device);

    // Per-layer hidden / cell state, each (1, hidden) on `device`.
    struct State {
        std::vector<brotensor::Tensor> h, c;
    };
    State init_state() const;

    // Advance one step on `token_id`: embed, run the LSTM stack, project.
    // Writes dec_proj (1, hidden) to `out`; mutates `st` in place.
    void step(int32_t token_id, State& st, brotensor::Tensor& out) const;
};

// ─── TDT joint network ─────────────────────────────────────────────────────
//
// logits = head( relu(enc_proj[t] + dec_proj) ). Split: first vocab_size are
// token logits, the remaining len(durations) are duration logits.
struct ParakeetJoint {
    brotensor::Tensor head_w, head_b;   // (vocab + n_durations, hidden)
    brotensor::Device device = brotensor::Device::CPU;

    void load(const brotensor::safetensors::File& f,
              const ParakeetConfig& cfg, brotensor::Device device);

    // enc_proj_row, dec_proj: each (1, hidden). Writes logits (1, V+nd) to out.
    void forward(const brotensor::Tensor& enc_proj_row,
                 const brotensor::Tensor& dec_proj,
                 brotensor::Tensor& out) const;
};

}  // namespace brosoundml
