#pragma once

// Qwen3-TTS Talker — the autoregressive Qwen3 decoder backbone (28 layers) that
// drives synthesis: a dual text/codec embedding stream through a GQA + QK-norm
// + interleaved-M-RoPE transformer, with a codec_head over acoustic codebook 0.
//
// Internal to the qwen_tts target (not in the public include/ surface). Stage 3
// builds the forward pass (prefill) and the embedding helpers; the AR
// generation loop + Code Predictor + input assembly follow in stages 4-5.
//
// CPU FP32 only for now — like the codec decoder, the hand-rolled glue
// (interleaved M-RoPE, QK-norm, GQA causal attention) reads host buffers, so
// weights (BF16 on disk) are widened to FP32 on the host and the path is pinned
// to the CPU via a brotensor DeviceScope. A KV-cached / GPU path follows once
// the AR loop lands and the steps move onto the brotensor op surface.

#include "brosoundml/qwen_tts.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brosoundml {

// One Qwen3 decoder layer: RMSNorm, GQA self-attention with per-head QK-norm and
// interleaved M-RoPE, then a SwiGLU MLP. All projections are bias-free; the two
// residual adds are plain (no layer scale).
struct QwenTtsTalkerLayer {
    brotensor::Tensor in_ln, post_ln;          // (hidden,1) RMSNorm weights
    brotensor::Tensor qw, kw, vw, ow;          // projections, no bias
    brotensor::Tensor q_norm, k_norm;          // (head_dim,1) per-head QK RMSNorm
    brotensor::Tensor gate, up, down;          // SwiGLU MLP, no bias
};

// The Talker. load() widens the talker.* weights to host FP32; forward() runs
// one prefill pass over a precomputed embedding stream + 3-axis M-RoPE
// positions, returning the post-final-norm hidden states and codec_head logits.
struct QwenTtsTalker {
    // ── dims ──
    int num_layers = 0;
    int hidden = 0, intermediate = 0;
    int n_q_heads = 0, n_kv_heads = 0, head_dim = 0;
    int vocab = 0, text_vocab = 0, text_hidden = 0;
    float rms_eps = 1e-6f, rope_theta = 1e6f;
    std::vector<int> mrope_section;     // [24,20,20]
    bool mrope_interleaved = true;

    // ── weights ──
    brotensor::Tensor codec_embedding;  // (vocab, hidden)
    brotensor::Tensor text_embedding;   // (text_vocab, text_hidden)
    brotensor::Tensor tp_fc1_w, tp_fc1_b;  // text_projection fc1 (text_hidden->text_hidden)
    brotensor::Tensor tp_fc2_w, tp_fc2_b;  // text_projection fc2 (text_hidden->hidden)
    brotensor::Tensor codec_head;       // (vocab, hidden), no bias
    brotensor::Tensor final_norm;       // (hidden,1) RMSNorm
    std::vector<QwenTtsTalkerLayer> layers;

    // Build from the talker.* tensors of model.safetensors (BF16 -> host FP32).
    void load(const brotensor::safetensors::File& f, const QwenTtsTalkerConfig& cfg);

    // Prefill forward. `inputs_embeds` is T*hidden row-major (row t = frame t).
    // `pos3T` is the 3-axis M-RoPE position grid, axis-major: pos3T[a*T + t]
    // for a in {0,1,2}. Writes hidden_out (T,hidden) (post final RMSNorm) and
    // logits_out (T,vocab) = codec_head(hidden).
    void forward(const float* inputs_embeds, int T, const int32_t* pos3T,
                 brotensor::Tensor& hidden_out,
                 brotensor::Tensor& logits_out) const;

    // Embedding helpers (the two input streams). Each writes `hidden` floats.
    void codec_embed(int id, float* out) const;        // codec_embedding[id]
    void text_embed_proj(int id, float* out) const;     // text_projection(text_embedding[id])
};

}  // namespace brosoundml
