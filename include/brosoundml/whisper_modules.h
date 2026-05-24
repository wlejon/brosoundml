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

}  // namespace brosoundml
