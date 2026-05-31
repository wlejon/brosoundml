#pragma once

// Qwen3-TTS Code Predictor — the small "depth" transformer that turns the
// Talker's per-frame hidden state plus acoustic codebook 0 into the remaining
// 15 acoustic codebooks (the RQ-Transformer / Moshi pattern). Per frame it runs
// a short autoregressive sequence over the codebook axis: a 2-token prefill
// (Talker hidden, Talker-embedded codebook 0) then one step per generated code.
//
// It reuses the Qwen3 decoder layer (RMSNorm, GQA + per-head QK-norm, SwiGLU)
// but with *plain* single-axis RoPE — the depth axis has no M-RoPE — and full
// causal attention (no sliding window). Inputs/outputs live in the Talker
// hidden width (1024); the upstream small_to_mtp_projection is an identity here
// (the Code Predictor hidden size equals the Talker's) so it is omitted.
//
// Internal to the qwen_tts target. CPU FP32 only, mirroring the Talker: weights
// are BF16 on disk and widened to host FP32 on load under a DeviceScope(CPU).

#include "brosoundml/qwen_tts.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brosoundml {

// One depth-transformer layer — identical structure to a Talker layer.
struct QwenTtsCodePredictorLayer {
    brotensor::Tensor in_ln, post_ln;
    brotensor::Tensor qw, kw, vw, ow;
    brotensor::Tensor q_norm, k_norm;
    brotensor::Tensor gate, up, down;
};

struct QwenTtsCodePredictor {
    // ── dims ──
    int num_layers = 0;
    int hidden = 0, intermediate = 0;
    int n_q_heads = 0, n_kv_heads = 0, head_dim = 0;
    int vocab = 0;            // per-codebook code vocab (2048)
    int num_code_groups = 0;  // codebooks per frame incl. codebook 0 (16)
    float rms_eps = 1e-6f, rope_theta = 1e6f;

    // ── weights ──
    // codec_embedding[j] embeds codebook (j+1); lm_head[j] predicts codebook
    // (j+1). There are (num_code_groups - 1) of each.
    std::vector<brotensor::Tensor> codec_embedding;
    std::vector<brotensor::Tensor> lm_head;
    brotensor::Tensor final_norm;
    std::vector<QwenTtsCodePredictorLayer> layers;

    // Build from the talker.code_predictor.* tensors (BF16 -> FP32 on `device`;
    // q/k projections + q/k_norm permuted into brotensor's adjacent-pair RoPE
    // layout, mirroring the Talker).
    void load(const brotensor::safetensors::File& f,
              const QwenTtsCodePredictorConfig& cfg,
              brotensor::Device device = brotensor::Device::CPU);

    // Greedy expansion of one frame. `past_hidden` is the Talker hidden state
    // that produced codebook 0 (hidden floats); `c0_embed` is the Talker's
    // codec_embedding of that codebook-0 code (hidden floats). Writes the
    // (num_code_groups - 1) remaining codes (codebooks 1..15) to `out_codes`.
    void predict(const float* past_hidden, const float* c0_embed,
                 std::vector<int>& out_codes) const;

    // Raw pointer to codec_embedding table `table` row `id` (hidden floats).
    const float* codec_embedding_row(int table, int id) const;
};

}  // namespace brosoundml
