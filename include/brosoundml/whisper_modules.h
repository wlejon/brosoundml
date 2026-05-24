#pragma once

// Whisper-specific submodules.
//
// One step up from the generic nn-modules in `modules.h`: these classes know
// the HF Whisper state-dict key naming and the per-stage forward-pass topology
// (log-mel front-end, encoder backbone). Decoder modules land in a later stage.
//
// Loaded directly from `model.safetensors` via `load_from(safetensors::File)`.
// Keys match the HuggingFace `transformers/models/whisper/modeling_whisper.py`
// state dict.
//
// CPU FP32-only. The Whisper forward pass moves to GPU once we have a
// fused-attention path on those backends — until then these submodules throw
// a clear runtime_error if the source tensors are not CPU FP32.

#include "brosoundml/audio.h"
#include "brosoundml/modules.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <string>
#include <vector>

namespace brosoundml {

// ─── LogMel ────────────────────────────────────────────────────────────────
//
// Whisper's log-mel front-end: 16 kHz mono FP32 PCM -> (num_mel_bins, 3000)
// FP32 log-mel feature tensor on the model device.
//
// Recipe (mirrors openai/whisper:whisper/audio.py):
//   1. Pad or truncate the audio to exactly 30 s (480 000 samples) at 16 kHz.
//   2. STFT with n_fft=400, hop=160, win=400, Hann window, center=True
//      (reflect-padded). 480 000 samples + center true give 3001 frames; the
//      last frame is dropped (`stft[..., :-1]`) to get exactly 3000.
//   3. Power spectrum = re*re + im*im over the (frames, 201) half-spectrum.
//   4. Mel-project: (n_mels, 201) @ (201, 3000) = (n_mels, 3000) using the
//      Slaney-normalised mel filterbank built at load time.
//   5. log10, clamp min 1e-10.
//   6. Dynamic-range clamp: log_spec = max(log_spec, log_spec.max() - 8.0).
//   7. Normalize: (log_spec + 4.0) / 4.0.
//
// No learnable weights; the mel filterbank + Hann window are built from
// closed-form formulas the first time `build()` is called. Whisper requires
// the input audio to be 16 kHz mono — `forward()` throws on a mismatch
// (resampling is the caller's problem).
struct LogMel {
    int num_mel_bins   = 0;
    int n_fft          = 400;
    int hop_length     = 160;
    int win_length     = 400;
    int n_frames       = 3000;          // fixed: 30 s at 100 Hz frame rate
    int sample_rate    = 16000;

    brotensor::Tensor mel_filters;      // (num_mel_bins, n_fft/2 + 1) FP32
    brotensor::Tensor hann_window;      // (1, win_length) FP32

    // Build mel_filters + hann_window from scratch; no safetensors needed.
    // Called once after WhisperConfig is parsed.
    void build(int num_mel_bins, brotensor::Device device);

    // audio.samples is mono FP32 [-1, 1] at sample_rate (must be 16 kHz).
    // Out: (num_mel_bins, n_frames) FP32 on the same device as mel_filters.
    void forward(const AudioBuffer& audio, brotensor::Tensor& out) const;
};

// ─── WhisperEncoderLayer ──────────────────────────────────────────────────
//
// One pre-LayerNorm Transformer block (HF naming):
//   residual = x
//   x = self_attn_layer_norm(x)
//   x = self_attn(x)              // standard MHA, no causal mask
//   x = residual + x
//   residual = x
//   x = final_layer_norm(x)       // HF names the FFN-side LN "final_layer_norm"
//   x = fc1(x); x = gelu(x); x = fc2(x)
//   x = residual + x
//
// Whisper quirk: the K-projection has NO bias on disk. To keep the shared
// `MHA` API stable, `load_from()` allocates `bk` as (D, 1) zeros — the math
// is unchanged.
struct WhisperEncoderLayer {
    int d_model = 0;
    int ffn_dim = 0;
    int num_heads = 0;

    LayerNorm self_attn_layer_norm;
    MHA       self_attn;
    LayerNorm final_layer_norm;
    Linear    fc1;
    Linear    fc2;

    void load_from(const brotensor::safetensors::File& f,
                   const std::string& prefix,
                   int d_model, int ffn_dim, int num_heads);

    // X: (L, d_model). Y: (L, d_model), resized. CPU FP32 only.
    void forward(const brotensor::Tensor& X, brotensor::Tensor& Y) const;
};

// ─── WhisperEncoder ────────────────────────────────────────────────────────
//
// The full Whisper encoder: 2x strided conv1d + GELU stem,
// sinusoidal/learned positional embedding (loaded from safetensors), pre-LN
// Transformer stack, final encoder LayerNorm. Input is the log-mel feature
// tensor in NCL ((num_mel_bins, 3000)); output is (max_source_positions,
// d_model) = (1500, d_model) in NLC order.
struct WhisperEncoder {
    int num_mel_bins = 0;
    int d_model = 0;
    int max_source_positions = 0;
    int encoder_layers = 0;
    int encoder_ffn_dim = 0;
    int encoder_attention_heads = 0;

    Conv1d                            conv1;            // (d_model, num_mel_bins, 3)
    Conv1d                            conv2;            // (d_model, d_model, 3)
    brotensor::Tensor                 embed_positions;  // (max_source_positions, d_model)
    std::vector<WhisperEncoderLayer>  layers;
    LayerNorm                         layer_norm;       // top-level encoder LN

    // Load every weight from the HF state dict (key prefix model.encoder.*).
    void load_from(const brotensor::safetensors::File& f,
                   int num_mel_bins,
                   int d_model,
                   int max_source_positions,
                   int encoder_layers,
                   int encoder_ffn_dim,
                   int encoder_attention_heads);

    // mel: (num_mel_bins, 3000) FP32 — the LogMel output reshaped to NCL with
    // N=1 (i.e. (1, num_mel_bins*3000)). hidden: (max_source_positions, d_model)
    // resized on output. CPU FP32 only.
    void forward(const brotensor::Tensor& mel, brotensor::Tensor& hidden) const;
};

// ─── KV cache ──────────────────────────────────────────────────────────────
//
// Per-layer state the Whisper decoder needs to run autoregressively in
// O(prompt_len + new_tokens) instead of O(N^2). One `WhisperLayerCache` per
// decoder layer:
//
//   * self-attn K/V grow by `T` rows on each forward pass (T = prompt length
//     in the prefill case, T = 1 in the single-step case). `self_k` and
//     `self_v` are pre-allocated to (max_target_positions, d_model); only the
//     first `self_len` rows are valid.
//
//   * cross-attn K/V are computed once from the encoder hidden states (via
//     `prime_cross` on the owning `WhisperDecoderLayer`) and reused for every
//     subsequent decode step. Shape (max_source_positions, d_model).
//
// All four tensors are CPU FP32. Reset between transcriptions with
// `WhisperKVCache::reset()`; the storage is kept so a fresh transcription is
// allocation-free.
struct WhisperLayerCache {
    brotensor::Tensor self_k;          // (max_target_positions, d_model)
    brotensor::Tensor self_v;          // (max_target_positions, d_model)
    int               self_len = 0;    // valid prefix length in self_k/self_v

    brotensor::Tensor cross_k;         // (max_source_positions, d_model)
    brotensor::Tensor cross_v;         // (max_source_positions, d_model)
    bool              cross_primed = false;
};

// Whole-decoder KV cache: one `WhisperLayerCache` per decoder layer plus a
// uniform reset / size accessor. `allocate()` pre-sizes every per-layer slab
// so the decode loop can append rows without a single allocation.
struct WhisperKVCache {
    std::vector<WhisperLayerCache> layers;

    // Allocate per-layer self-K/V at (max_target_positions, d_model) and
    // cross-K/V at (max_source_positions, d_model). Idempotent: calling twice
    // with the same shape leaves the cache unchanged.
    void allocate(int decoder_layers, int d_model,
                  int max_target_positions, int max_source_positions);

    // Zero `self_len` and clear `cross_primed` on every layer. Keeps the
    // storage so the next transcription is allocation-free.
    void reset();

    // Current generated length — every layer holds the same `self_len`, so we
    // surface layer 0's value (or 0 if the cache hasn't been allocated yet).
    int size() const { return layers.empty() ? 0 : layers[0].self_len; }
};

// ─── Low-level decoder attention kernels ───────────────────────────────────
//
// brosoundml's shared `MHA::forward` runs full self-attention with no causal
// mask and no cache; that is not enough for the Whisper decoder's two
// attention sublayers. Two new free functions cover the gap — both FP32 CPU
// only, both share the open-coded scalar attention style of
// `mha_attention_fp32` in modules.cpp.
//
// `mha_causal_cached_fp32`: causal self-attention with append-to-cache. The
// caller supplies the current step's input X ((T, D)), the four Q/K/V/O
// linear weights, and the layer's `WhisperLayerCache`. We project Q/K/V from
// X, append the freshly projected K/V rows into the cache (positions
// [self_len, self_len + T)), then run causal-masked softmax attention over
// the cache's first `self_len + T` rows for each of the T query positions —
// query row q attends to keys [0, self_len + q]. After the call the cache
// has grown by T rows.
//
// `cross_attn_cached_fp32`: cross-attention against pre-projected cached
// encoder K/V. Only Q is projected from the current step's input X; K/V
// already live in `cross_k` / `cross_v`. Full attention over all
// max_source_positions encoder positions (no mask, no causal).
//
// Both functions write the post-`Wo`-projection output to `out` (resized to
// (T, D)).
void mha_causal_cached_fp32(const brotensor::Tensor& X,
                            const brotensor::Tensor& Wq, const brotensor::Tensor& bq,
                            const brotensor::Tensor& Wk, const brotensor::Tensor& bk,
                            const brotensor::Tensor& Wv, const brotensor::Tensor& bv,
                            const brotensor::Tensor& Wo, const brotensor::Tensor& bo,
                            int num_heads,
                            WhisperLayerCache& cache,
                            brotensor::Tensor& out);

void cross_attn_cached_fp32(const brotensor::Tensor& X,
                            const brotensor::Tensor& Wq, const brotensor::Tensor& bq,
                            const brotensor::Tensor& Wo, const brotensor::Tensor& bo,
                            int num_heads,
                            const WhisperLayerCache& cache,
                            brotensor::Tensor& out);

// Prime a single layer's cross-attention cache from the encoder hidden state.
// Projects K and V through the supplied (Wk, V's Wv) — Whisper's K projection
// has no bias on disk, so `bk` is the zero-filled (D, 1) tensor the
// load-from path produced. After the call `cache.cross_primed = true` and
// `cache.cross_k`/`cache.cross_v` hold (max_source_positions, d_model).
void prime_cross_cache_fp32(const brotensor::Tensor& encoder_hidden,
                            const brotensor::Tensor& Wk, const brotensor::Tensor& bk,
                            const brotensor::Tensor& Wv, const brotensor::Tensor& bv,
                            WhisperLayerCache& cache);

// ─── WhisperDecoderLayer ───────────────────────────────────────────────────
//
// One pre-LayerNorm decoder block. The HF Whisper layout is:
//
//   residual = x
//   x = self_attn_layer_norm(x)
//   x = causal_self_attn(x, kv_cache=cache.self_*)   // grows cache
//   x = residual + x
//   residual = x
//   x = encoder_attn_layer_norm(x)
//   x = cross_attn(x, encoder_kv=cache.cross_*)       // reads cache
//   x = residual + x
//   residual = x
//   x = final_layer_norm(x)
//   x = fc1(x); x = gelu(x); x = fc2(x)
//   x = residual + x
//
// Whisper quirk: both self_attn.k_proj and encoder_attn.k_proj have NO bias
// on disk. `load_from` zero-fills the matching `bk` tensors so the kernel
// math is unchanged.
struct WhisperDecoderLayer {
    int d_model = 0;
    int ffn_dim = 0;
    int num_heads = 0;

    // Self-attention sublayer.
    LayerNorm         self_attn_layer_norm;
    brotensor::Tensor self_Wq, self_bq;    // (D, D), (D, 1)
    brotensor::Tensor self_Wk, self_bk;    // (D, D), (D, 1) — bk zero-filled
    brotensor::Tensor self_Wv, self_bv;    // (D, D), (D, 1)
    brotensor::Tensor self_Wo, self_bo;    // (D, D), (D, 1)

    // Cross-attention sublayer.
    LayerNorm         encoder_attn_layer_norm;
    brotensor::Tensor cross_Wq, cross_bq;  // (D, D), (D, 1)
    brotensor::Tensor cross_Wk, cross_bk;  // (D, D), (D, 1) — bk zero-filled
    brotensor::Tensor cross_Wv, cross_bv;  // (D, D), (D, 1)
    brotensor::Tensor cross_Wo, cross_bo;  // (D, D), (D, 1)

    // FFN sublayer.
    LayerNorm         final_layer_norm;
    Linear            fc1;                  // (ffn_dim, d_model)
    Linear            fc2;                  // (d_model, ffn_dim)

    void load_from(const brotensor::safetensors::File& f,
                   const std::string& prefix,
                   int d_model, int ffn_dim, int num_heads);

    // Project + cache the cross-attention K/V for this layer from the
    // encoder hidden state. Idempotent — re-priming overwrites the cache.
    void prime_cross(const brotensor::Tensor& encoder_hidden,
                     WhisperLayerCache& cache) const;

    // X: (T, d_model) — T new tokens to process. T == 1 for single-step,
    // T == prompt_len for prefill. `cache.cross_primed` must be true; the
    // call grows `cache.self_len` by T. Y: (T, d_model), resized.
    void forward(const brotensor::Tensor& X,
                 WhisperLayerCache& cache,
                 brotensor::Tensor& Y) const;
};

// ─── WhisperDecoder ────────────────────────────────────────────────────────
//
// Full Whisper decoder: learned token + (learned) positional embeddings,
// pre-LN Transformer stack with causal self-attention + cross-attention to
// the encoder, top-level LayerNorm, and a tied LM head.
//
// Tied LM head: upstream Whisper ties `proj_out.weight` to `embed_tokens`,
// so the HF safetensors typically omit `proj_out.weight`. We default to the
// tied path (logits = hidden @ embed_tokens.T); if the checkpoint *does*
// carry `proj_out.weight` we honour it instead. The boolean
// `proj_out_explicit` records which path was taken.
struct WhisperDecoder {
    int d_model               = 0;
    int decoder_layers        = 0;
    int decoder_ffn_dim       = 0;
    int decoder_attention_heads = 0;
    int vocab_size            = 0;
    int max_target_positions  = 0;
    int max_source_positions  = 0;

    brotensor::Tensor                embed_tokens;       // (vocab_size, d_model)
    brotensor::Tensor                embed_positions;    // (max_target_positions, d_model) LEARNED
    std::vector<WhisperDecoderLayer> layers;
    LayerNorm                        layer_norm;         // top-level decoder LN

    brotensor::Tensor                proj_out_weight;    // (vocab_size, d_model) — empty when tied
    bool                             proj_out_explicit = false;

    // Load every decoder weight from the HF state dict (key prefix
    // model.decoder.*). Throws on a missing required key; `proj_out.weight`
    // is optional (tied LM head is the default).
    void load_from(const brotensor::safetensors::File& f,
                   int d_model,
                   int decoder_layers,
                   int decoder_ffn_dim,
                   int decoder_attention_heads,
                   int vocab_size,
                   int max_target_positions,
                   int max_source_positions);

    // Prime every layer's cross-attention cache from encoder_hidden.
    // encoder_hidden: (max_source_positions, d_model). Must be called once
    // per audio clip before any forward() call.
    void prime_cross(const brotensor::Tensor& encoder_hidden,
                     WhisperKVCache& cache) const;

    // Run the decoder over `token_ids` (length T). `pos_offset` is the
    // cache's current `self_len` before the call — equals 0 on prefill and
    // the current generated length on a single-step. Writes logits of shape
    // (T, vocab_size) to `logits`. Throws if cross-attn cache not primed,
    // or if T + pos_offset exceeds max_target_positions.
    void forward(const std::int32_t* token_ids, int T,
                 int pos_offset,
                 WhisperKVCache& cache,
                 brotensor::Tensor& logits) const;
};

}  // namespace brosoundml
