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

// ─── Style-conditioned norms (StyleTTS2 / Kokoro predictor) ────────────────
//
// AdaLayerNorm:
//   y = (1 + gamma(s)) * LayerNorm(x) + beta(s)
//   x: (L, C). gamma, beta: derived from style by `fc: C*2 <- style_dim`,
//   chunked into two (1, C) vectors broadcast across L.
//
// AdaIN1d:
//   y = (1 + gamma(s)) * InstanceNorm1d(x) + beta(s)
//   x: (1, C*L) NCL with N=1. Same style-conditioned (gamma, beta) shape,
//   broadcast across L within each channel.
struct AdaLayerNormWeights {
    int    channels  = 0;
    int    style_dim = 0;
    float  eps       = 1e-5f;
    Linear fc;   // (channels * 2, style_dim)
};

struct AdaIN1dWeights {
    int    channels  = 0;
    int    style_dim = 0;
    float  eps       = 1e-5f;
    Linear fc;   // (channels * 2, style_dim)
};

// ─── AdainResBlk1d (iSTFTNet decoder / predictor F0+N blocks) ──────────────
//
// The StyleTTS2 AdaIN residual block from kokoro/istftnet.py:
//   residual = conv2(actv(norm2(conv1(pool(actv(norm1(x, s)))), s)))
//   shortcut = (upsample ? nearest_2x : x)
//              -> conv1x1 if dim_in != dim_out
//   out = (residual + shortcut) / sqrt(2)
// pool is identity for the non-upsampling blocks, or a depthwise (groups=dim_in)
// ConvTranspose1d(k=3, stride=2, pad=1, out_pad=1) for the upsampling blocks
// in the predictor's F0 / N stacks.
struct AdainResBlk1dWeights {
    int                channels_in  = 0;
    int                channels_out = 0;
    bool               upsample     = false;
    bool               learned_sc   = false;  // dim_in != dim_out
    AdaIN1dWeights     norm1;        // channels_in
    AdaIN1dWeights     norm2;        // channels_out
    Conv1d             conv1;        // (channels_out, channels_in, 3)
    Conv1d             conv2;        // (channels_out, channels_out, 3)
    Conv1d             conv1x1;      // (channels_out, channels_in, 1) when learned_sc
    // Depthwise ConvTranspose1d weight for the upsampling pool: stored as
    // (channels_in, 1, 3) since groups == channels_in (== channels_out / dim_in).
    brotensor::Tensor  pool_W;       // (channels_in, 3)  flattened
    brotensor::Tensor  pool_b;       // (channels_in, 1)
};

// ─── DurationEncoder ───────────────────────────────────────────────────────
//
// Alternating BiLSTM + AdaLayerNorm blocks; between layers the style vector
// is concatenated back onto the feature axis so each LSTM sees
// (channels + style_dim) input. Loaded from
// `predictor.module.text_encoder.lstms.*`.
struct DurationEncoder {
    int channels  = 0;   // d_model = KokoroConfig::hidden_dim (512)
    int style_dim = 0;   // 128
    int nlayers   = 0;   // KokoroConfig::n_layer (3) — pairs of (BiLSTM, AdaLayerNorm)
    struct Block {
        BiLSTM             bilstm;
        AdaLayerNormWeights aln;
    };
    std::vector<Block> blocks;

    // d_en: (1, channels*L) NCL.  style: (1, style_dim).
    // d: (L, channels + style_dim) — the format ProsodyPredictor.lstm consumes.
    void forward(const brotensor::Tensor& d_en,
                 const brotensor::Tensor& style,
                 int L,
                 brotensor::Tensor& d) const;
};

// ─── Predictor (the full ProsodyPredictor) ─────────────────────────────────
//
// Owns duration encoder + duration LSTM + duration projection + shared LSTM +
// F0 / N AdaIN res blocks + final 1x1 conv projections. The forward pass
// returns every intermediate the test layer compares against — keep them in
// a struct so the test doesn't need a separate hand-rolled mirror.
struct Predictor {
    KokoroConfig                       cfg;
    DurationEncoder                    text_encoder;
    BiLSTM                             lstm;                // 640 -> 512
    Linear                             duration_proj;       // 512 -> max_dur (50)
    BiLSTM                             shared;              // 640 -> 512
    std::vector<AdainResBlk1dWeights>  F0_blocks;           // 3 blocks
    std::vector<AdainResBlk1dWeights>  N_blocks;            // 3 blocks
    Conv1d                             F0_proj;             // (1, channels/2, 1)
    Conv1d                             N_proj;              // (1, channels/2, 1)

    struct Output {
        brotensor::Tensor    d;            // (L, channels + style_dim)
        brotensor::Tensor    lstm_x;       // (L, channels)
        brotensor::Tensor    duration;     // (L, max_dur)
        std::vector<int32_t> pred_dur;     // (L,)
        brotensor::Tensor    en;           // (1, (channels+style_dim) * total_frames)
        brotensor::Tensor    F0_pred;      // (1, 2 * total_frames)
        brotensor::Tensor    N_pred;       // (1, 2 * total_frames)
    };

    void load_from(const brotensor::safetensors::File& f, const KokoroConfig& cfg);
    // d_en: BertEncoder output, (1, hidden_dim*L) NCL.
    // ref_s: full voice row (1, 2*style_dim). The predictor reads ref_s[:, style_dim:].
    // speed: predicted durations are divided by `speed` before rounding.
    void forward(const brotensor::Tensor& d_en,
                 const brotensor::Tensor& ref_s,
                 int L,
                 float speed,
                 Output& out) const;
};

// ─── Decoder backbone (everything up to the Generator) ────────────────────
//
// Mirrors kokoro/istftnet.py::Decoder.forward, except the trailing
// `self.generator(...)` call. Owns F0_conv, N_conv, asr_res, encode, and the
// 4-block decode list. Output is the (1, 512*L_after_upsample) tensor the
// Generator consumes — captured as `gen_in` in the reference dump.
struct DecoderBackbone {
    Conv1d F0_conv;     // (1, 1, 3) stride 2 padding 1
    Conv1d N_conv;      // (1, 1, 3) stride 2 padding 1
    Conv1d asr_res;     // (64, 512, 1) — 1x1 projection
    AdainResBlk1dWeights              encode;
    std::vector<AdainResBlk1dWeights> decode;    // 4 blocks; the last upsamples 2x

    void load_from(const brotensor::safetensors::File& f);
    // asr: (1, 512*T).  F0_pred / N_pred: (1, 2*T).  ref_s: (1, 256).
    // gen_in is written as (1, 512 * (2*T)) NCL — input to the Generator.
    void forward(const brotensor::Tensor& asr,
                 const brotensor::Tensor& F0_pred,
                 const brotensor::Tensor& N_pred,
                 const brotensor::Tensor& ref_s,
                 int T,
                 brotensor::Tensor& gen_in) const;
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
