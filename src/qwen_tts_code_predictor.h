#pragma once

// Qwen3-TTS Code Predictor — the small "depth" transformer that turns the
// Talker's per-frame hidden state plus acoustic codebook 0 into the remaining
// 15 acoustic codebooks (the RQ-Transformer / Moshi pattern). Per frame it runs
// a short autoregressive sequence over the codebook axis: a 2-token prefill
// (Talker hidden, Talker-embedded codebook 0) then one step per generated code.
//
// It reuses the Qwen3 decoder layer (RMSNorm, GQA + per-head QK-norm, SwiGLU)
// but with *plain* single-axis RoPE — the depth axis has no M-RoPE — and full
// causal attention (no sliding window). Its *inputs* (the Talker hidden + the
// code embeddings) live in the Talker hidden width; the depth transformer itself
// runs at the Code Predictor hidden width. When the two differ (the 1.7B model:
// Talker 2048, Code Predictor 1024) the upstream small_to_mtp_projection
// (Linear talker_hidden -> cp_hidden + bias) maps every input row down before
// the transformer. When they are equal (the 0.6B model) the projection is an
// identity and is absent from the checkpoint, so it is skipped.
//
// Internal to the qwen_tts target. Device-neutral (CPU + CUDA), mirroring the
// Talker: FP32 weights on every backend, brotensor device ops throughout, FP32
// attention (flash on CPU, varlen on CUDA), q/k permuted at load for rope_apply.

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
    int hidden = 0, intermediate = 0;   // depth-transformer hidden width
    int talker_hidden = 0;              // input/embedding width (Talker hidden)
    int n_q_heads = 0, n_kv_heads = 0, head_dim = 0;
    int vocab = 0;            // per-codebook code vocab (2048)
    int num_code_groups = 0;  // codebooks per frame incl. codebook 0 (16)
    float rms_eps = 1e-6f, rope_theta = 1e6f;

    // ── weights ──
    // codec_embedding[j] embeds codebook (j+1) in the *Talker* hidden width;
    // lm_head[j] predicts codebook (j+1) from the depth-transformer hidden.
    // There are (num_code_groups - 1) of each.
    std::vector<brotensor::Tensor> codec_embedding;
    std::vector<brotensor::Tensor> lm_head;
    brotensor::Tensor final_norm;
    std::vector<QwenTtsCodePredictorLayer> layers;

    // small_to_mtp_projection: Linear(talker_hidden -> hidden) + bias, applied to
    // every input row (the Talker hidden and the code embeddings) before the
    // depth transformer. Empty when talker_hidden == hidden (identity; 0.6B).
    brotensor::Tensor mtp_proj_w, mtp_proj_b;
    bool has_mtp_proj = false;

    // Precomputed plain-RoPE cos/sin tables for the fixed depth positions
    // 0..num_code_groups-1, (num_code_groups, head_dim/2) FP32 on the model
    // device. The depth axis positions never change, so a frame's depth steps
    // slice rows from these instead of rebuilding + re-uploading per step.
    brotensor::Tensor rope_cos, rope_sin;

    // Build from the talker.code_predictor.* tensors (BF16 -> FP32 on `device`;
    // q/k projections + q/k_norm permuted into brotensor's adjacent-pair RoPE
    // layout, mirroring the Talker). `talker_hidden` is the Talker's hidden width
    // — the width of the code embeddings and the conditioning hidden, which the
    // small_to_mtp_projection maps to the depth-transformer width when they
    // differ.
    void load(const brotensor::safetensors::File& f,
              const QwenTtsCodePredictorConfig& cfg, int talker_hidden,
              brotensor::Device device = brotensor::Device::CPU);

    // Greedy expansion of one frame. `past_hidden` is the Talker hidden state
    // that produced codebook 0 (hidden floats); `c0_embed` is the Talker's
    // codec_embedding of that codebook-0 code (hidden floats). Writes the
    // (num_code_groups - 1) remaining codes (codebooks 1..15) to `out_codes`.
    // Host-pointer convenience wrapper over predict_dev (uploads the two rows).
    void predict(const float* past_hidden, const float* c0_embed,
                 std::vector<int>& out_codes) const;

    // Device-resident form used by the AR loop: `past_hidden` and `c0_embed` are
    // (1, hidden) tensors already on the model's device, so a frame expands with
    // no host round-trip (argmax runs on-device, only the chosen code id — 4
    // bytes per head — comes back). Same greedy result as predict().
    //
    // Sampling: with temperature == 0 (or counter == nullptr) every codebook is
    // the greedy argmax — the default, bit-exact with predict(). With
    // temperature > 0 each codebook 1..15 is drawn through
    // brotensor::sample_logits seeded by (key, *counter); *counter is advanced
    // one step per code so the draws compose with the Talker's codebook-0 stream.
    void predict_dev(const brotensor::Tensor& past_hidden,
                     const brotensor::Tensor& c0_embed,
                     std::vector<int>& out_codes,
                     float temperature = 0.0f, int top_k = 0, float top_p = 1.0f,
                     std::uint64_t key = 0,
                     std::uint64_t* counter = nullptr) const;

    // Raw pointer to codec_embedding table `table` row `id` (hidden floats).
    const float* codec_embedding_row(int table, int id) const;
};

}  // namespace brosoundml
