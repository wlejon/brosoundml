#pragma once

// Qwen3-ASR text decoder — the stock Qwen3 LLM backbone (28 layers for 0.6B)
// that autoregressively emits the transcript: GQA + per-head QK-norm + RoPE
// transformer over a mixed text/audio embedding stream, with a separate
// lm_head over the text vocab.
//
// Internal to the qwen_asr target (not in the public include/ surface).
//
// Structurally the Qwen3-TTS Talker minus M-RoPE (ASR position ids are
// identical across the three M-RoPE axes, which reduces to standard 1D RoPE)
// and minus the dual codec stream. Same device bridges (see
// qwen_tts_device.h): weights upload FP32 everywhere, q/k projections +
// q/k_norm are permuted at load into brotensor's adjacent-pair RoPE layout so
// rope_apply reproduces HF rotate-half.

#include "brosoundml/qwen_asr.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brosoundml {

// One Qwen3 decoder layer: RMSNorm, GQA self-attention with per-head QK-norm
// and RoPE, then a SwiGLU MLP. All projections are bias-free.
struct QwenAsrDecoderLayer {
    brotensor::Tensor in_ln, post_ln;   // (hidden,1) RMSNorm weights
    brotensor::Tensor qw, kw, vw, ow;   // projections, no bias
    brotensor::Tensor q_norm, k_norm;   // (head_dim,1) per-head QK RMSNorm
    brotensor::Tensor gate, up, down;   // SwiGLU MLP, no bias
};

// Growing KV cache for autoregressive stepping: per-layer post-RoPE K and V in
// the n_kv_heads (GQA) layout, FP32, on the model device. `len` counts cached
// tokens; `cap` the allocated rows (grown geometrically).
struct QwenAsrDecoderCache {
    int len = 0;
    int cap = 0;
    std::vector<brotensor::Tensor> k;   // [num_layers] (cap, n_kv_heads*head_dim)
    std::vector<brotensor::Tensor> v;
    void reset(int num_layers) {
        len = 0;
        cap = 0;
        k.assign(static_cast<std::size_t>(num_layers), {});
        v.assign(static_cast<std::size_t>(num_layers), {});
    }
};

struct QwenAsrDecoder {
    // ── dims ──
    int num_layers = 0;
    int hidden = 0, intermediate = 0;
    int n_q_heads = 0, n_kv_heads = 0, head_dim = 0;
    int vocab = 0;
    float rms_eps = 1e-6f, rope_theta = 1e6f;

    // ── weights ──
    brotensor::Tensor embed_tokens;     // (vocab, hidden)
    brotensor::Tensor lm_head;          // (vocab, hidden) — separate on disk
    brotensor::Tensor final_norm;       // (hidden,1) RMSNorm
    std::vector<QwenAsrDecoderLayer> layers;

    // Build from the thinker.model.* / thinker.lm_head tensors of
    // model.safetensors (BF16 -> FP32 on `device`).
    void load(const brotensor::safetensors::File& f, const QwenAsrConfig& cfg,
              brotensor::Device device = brotensor::Device::CPU);

    // Cached decoder pass over `n` new tokens whose RoPE positions are
    // cache->len .. cache->len+n-1 (0-based from the sequence start). The new
    // tokens are appended to the cache and each new query attends all cached
    // keys plus the new ones — a prefill (n=P into an empty cache) followed by
    // single-token steps reproduces a full causal forward. `embeds` is an
    // (n, hidden) FP32 device tensor. Writes hidden_out (n, hidden), post
    // final RMSNorm.
    void run_dev(const brotensor::Tensor& embeds, int n,
                 QwenAsrDecoderCache* cache,
                 brotensor::Tensor& hidden_out) const;

    // lm_head over the LAST row of an (n, hidden) device tensor -> (1, vocab)
    // logits (the greedy loop only ever needs the newest position).
    void logits_last(const brotensor::Tensor& hidden_rows,
                     brotensor::Tensor& logits_out) const;
};

}  // namespace brosoundml
