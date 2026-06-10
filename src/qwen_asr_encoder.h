#pragma once

// Qwen3-ASR "AuT" audio encoder — log-mel front-end, per-chunk Conv2d stem,
// 18-layer pre-LN Transformer with block-windowed bidirectional attention,
// and the 2-layer projector into the decoder width.
//
// Internal to the qwen_asr target (not in the public include/ surface).
//
// Device-neutral (CPU + CUDA): the forward pass composes brotensor device ops
// (conv2d, layernorm, varlen flash attention, gelu_exact, matmul). Weights
// upload FP32 on every backend and compute stays FP32, so CUDA matches CPU to
// float round-off. The mel front-end runs on the host (the audio arrives as a
// host buffer and the tables are closed-form); features upload at the conv
// boundary.

#include "brosoundml/audio.h"
#include "brosoundml/qwen_asr.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <vector>

namespace brosoundml {

// One pre-LayerNorm encoder block (HF Qwen3ASRAudioEncoderLayer):
//   x += self_attn(self_attn_layer_norm(x))     // block-windowed, no RoPE
//   x += fc2(gelu(fc1(final_layer_norm(x))))
// Every projection carries a bias; LayerNorm eps is nn.LayerNorm's 1e-5.
struct QwenAsrEncoderLayer {
    brotensor::Tensor ln1_w, ln1_b;     // self_attn_layer_norm
    brotensor::Tensor qw, qb, kw, kb;   // self_attn q/k projections
    brotensor::Tensor vw, vb, ow, ob;   // self_attn v/out projections
    brotensor::Tensor ln2_w, ln2_b;     // final_layer_norm (FFN-side)
    brotensor::Tensor fc1_w, fc1_b;     // (ffn, d_model)
    brotensor::Tensor fc2_w, fc2_b;     // (d_model, ffn)
};

// The full encoder. load() widens the thinker.audio_tower.* weights to FP32
// on `device`; forward() maps 16 kHz mono PCM to (n_audio_tokens, output_dim)
// hidden states ready to splice into the decoder's embedding stream.
struct QwenAsrEncoder {
    // ── dims (from QwenAsrConfig) ──
    int num_mel_bins = 0, d_model = 0, num_layers = 0, num_heads = 0;
    int ffn_dim = 0, output_dim = 0, conv_hidden = 0;
    int n_window = 0, n_window_infer = 0;

    // ── mel front-end tables (CPU; built at load, no learned weights) ──
    brotensor::Tensor mel_filters;      // (num_mel_bins, 201) FP32
    brotensor::Tensor hann_window;      // (1, 400) FP32

    // ── weights (FP32 on the model device) ──
    brotensor::Tensor conv1_w, conv1_b; // (480, 1*3*3),   (480, 1)
    brotensor::Tensor conv2_w, conv2_b; // (480, 480*3*3), (480, 1)
    brotensor::Tensor conv3_w, conv3_b; // (480, 480*3*3), (480, 1)
    brotensor::Tensor conv_out_w;       // (896, 7680), no bias — applied as a
                                        // (d_model, conv_hidden, 16, 1) conv so
                                        // the (t, c*f) flatten never leaves the
                                        // device (row-major layouts coincide)
    std::vector<QwenAsrEncoderLayer> layers;
    brotensor::Tensor ln_post_w, ln_post_b;
    brotensor::Tensor proj1_w, proj1_b; // (d_model, d_model)
    brotensor::Tensor proj2_w, proj2_b; // (output_dim, d_model)

    // Per-chunk sinusoidal position table, (13, d_model) FP32 on the model
    // device — every conv chunk restarts at position 0 (upstream adds the
    // table to the padded (B, 13, d_model) conv output before unpadding).
    brotensor::Tensor pos_table;

    void load(const brotensor::safetensors::File& f, const QwenAsrConfig& cfg,
              brotensor::Device device = brotensor::Device::CPU);

    // 16 kHz mono PCM -> (frames, num_mel_bins) host log-mel features,
    // frames = sample_count / 160 (no 30 s padding). Throws on a sample-rate
    // mismatch or audio shorter than one hop.
    std::vector<float> log_mel(const AudioBuffer& audio, int& frames_out) const;

    // Full forward: audio -> (n_audio_tokens, output_dim) FP32 on the model
    // device. n_audio_tokens = 13 per full 100-frame chunk plus the conv
    // length of the partial tail chunk (upstream zero-pads the tail to the
    // batch max before the conv and masks the dead tokens after).
    void forward(const AudioBuffer& audio, brotensor::Tensor& out) const;
};

}  // namespace brosoundml
