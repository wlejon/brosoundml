#pragma once

// Kokoro-specific submodules.
//
// One step up from the generic nn-modules in `modules.h`: these classes know
// the Kokoro state-dict naming and the per-stage forward-pass topology. Each
// submodule owns its weights and exposes a single `forward(...)`.
//
// Loaded directly from `weights/kokoro/model.safetensors` via
// `load_from(safetensors::File)`. The expected key prefixes match the
// upstream PyTorch state dict (with PyTorch's `weight_norm` parameterizations
// already fused into plain `.weight` by scripts/convert-kokoro.py).
//
// CPU FP32-only. The Kokoro forward pass moves to GPU once the AdaIN/affine
// op gap in brotensor (see CLAUDE.md) is filled — until then these submodules
// throw a clear runtime_error if the source tensors are not CPU FP32.

#include "brosoundml/kokoro.h"
#include "brosoundml/modules.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <string>
#include <vector>

namespace brosoundml {

// ─── PLBert ────────────────────────────────────────────────────────────────
//
// Kokoro's plBERT is a HuggingFace `AlbertModel`: factorised input embeddings
// (vocab → 128 → 768) + a single ALBERT layer applied `num_hidden_layers`
// times with shared weights, no pooler at inference. Public surface:
//
//   bert.load_from(weights, config.plbert);
//   bert.forward(input_ids, attention_mask, bert_dur);
//
// `input_ids` is the L-long phoneme id sequence with the upstream BOS/EOS
// convention (id 0 wrapping the real ids). `attention_mask` is L-long with
// 1=valid, 0=padding; pass an empty vector for unmasked input. `bert_dur`
// is sized to (L, hidden_size=768) on return.
struct PLBert {
    PLBertConfig         cfg;
    // Embeddings (all live at embedding_size=128 before the hidden projection).
    brotensor::Tensor    word_embeddings;        // (vocab_size, 128)
    brotensor::Tensor    position_embeddings;    // (max_position_embeddings, 128)
    brotensor::Tensor    token_type_embeddings;  // (2, 128)
    LayerNorm            emb_layernorm;          // (128, 1)
    // 128 -> hidden_size projection (ALBERT's factorised embedding).
    Linear               emb_to_hidden;
    // The one shared ALBERT layer (applied `num_hidden_layers` times).
    Linear               attn_q, attn_k, attn_v, attn_dense;
    LayerNorm            attn_layernorm;
    Linear               ffn, ffn_output;
    LayerNorm            full_layernorm;

    void load_from(const brotensor::safetensors::File& f, const PLBertConfig& c);
    void forward(const std::vector<int32_t>& input_ids,
                 const std::vector<int>& attention_mask,
                 brotensor::Tensor& bert_dur) const;
};

// ─── BertEncoder ───────────────────────────────────────────────────────────
//
// A single Linear (plBERT hidden_size → KokoroConfig::hidden_dim) — Kokoro's
// projection from plBERT space (768) to the predictor / decoder feature space
// (512). Loaded from `bert_encoder.module.*` keys; the forward call returns
// (hidden_dim, L) NCL to match the rest of the predictor stack.
struct BertEncoder {
    Linear projection;

    void load_from(const brotensor::safetensors::File& f,
                   int bert_hidden, int out_hidden);
    // bert_dur: (L, bert_hidden).  d_en: (out_hidden, L) NCL.
    void forward(const brotensor::Tensor& bert_dur,
                 brotensor::Tensor& d_en) const;
};

// ─── TextEncoder (StyleTTS2-style) ─────────────────────────────────────────
//
// Phoneme embedding -> depth × (Conv1d + per-channel LayerNorm + LeakyReLU) ->
// bidirectional LSTM. Matches kokoro/modules.py::TextEncoder. Output is the
// per-phoneme content feature in NCL order so the decoder can multiply it
// with the duration-aligned matrix.
struct TextEncoder {
    int channels    = 0;      // hidden_dim (= 512 in Kokoro-82M)
    int n_symbols   = 0;
    int kernel_size = 0;
    int depth       = 0;
    brotensor::Tensor    embedding;   // (n_symbols, channels)
    std::vector<Conv1d>  cnn;
    // The StyleTTS2 LayerNorm has per-channel (gamma, beta) — applied
    // per (n, l) position over the C axis. We carry the raw (C, 1) tensors.
    std::vector<brotensor::Tensor> ln_gamma, ln_beta;
    BiLSTM lstm;

    void load_from(const brotensor::safetensors::File& f, const KokoroConfig& cfg);
    // input_ids: L phoneme ids (with BOS/EOS).
    // text_mask: L-long 1=valid / 0=pad; empty for no padding.
    // t_en: (channels, L) NCL.
    void forward(const std::vector<int32_t>& input_ids,
                 const std::vector<int>& text_mask,
                 brotensor::Tensor& t_en) const;
};

// ─── Per-(N,L) channel-wise LayerNorm on NCL inputs ────────────────────────
//
// The StyleTTS2 / Kokoro CNN-stack LayerNorm: at every (n, l) position,
// normalise the length-C vector with per-channel `gamma`/`beta`. Exposed as
// a free function so the TextEncoder forward stays readable.
//   X, Y: (N, C*L) NCL.  gamma, beta: (C, 1).  CPU FP32-only.
void layernorm_1d_ncl(const brotensor::Tensor& X,
                      const brotensor::Tensor& gamma,
                      const brotensor::Tensor& beta,
                      int N, int C, int L, float eps,
                      brotensor::Tensor& Y);

}  // namespace brosoundml
