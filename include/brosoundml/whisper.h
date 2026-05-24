#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── Whisper ────────────────────────────────────────────────────────────────
//
// Whisper is OpenAI's encoder-decoder speech-to-text model. brosoundml targets
// the HF transformers checkpoints (whisper-tiny / -base / -small / -medium /
// -large-v3) — `config.json` + `model.safetensors` in a model directory, plus
// the tokenizer files consumed by brolm::whisper::Tokenizer (held by the
// caller; brosoundml itself takes already-tokenized prompts and emits token
// ids).
//
// Pipeline — the order brosoundml drives the components:
//
//   1. Log-mel front-end  16 kHz mono PCM -> log-mel spectrogram
//                         (num_mel_bins x 3000 frames, 30 s padded/truncated).
//                         Built from brotensor stft + mel-filterbank matmul +
//                         log. The 16 kHz / 25 ms / 10 ms / 80 (or 128) mels
//                         parameters are Whisper-fixed; only num_mel_bins
//                         varies (large-v3 uses 128).
//   2. Encoder            two strided conv1d stems + sinusoidal positional
//                         embeddings + a Transformer stack (pre-LN MHA + FFN).
//                         Outputs (1, max_source_positions, d_model).
//   3. Decoder            cross-attention Transformer with a KV cache.
//                         Autoregressively emits token ids conditioned on the
//                         encoder hidden states and the prompt (SOT, lang,
//                         task, no_timestamps?). Greedy decode for stage 5.
//   4. Tokenizer          brolm::whisper::Tokenizer (separate dep) maps
//                         id -> text. brosoundml returns the raw id stream
//                         plus a decoded transcript when the tokenizer is
//                         attached at the call site.
//
// brotensor op coverage: stft + complex_abs + matmul (mel front-end), conv1d,
// gelu, layer_norm, mha (composed via brosoundml's FP32 MHA / CrossAttention
// modules — see modules.h), embedding_lookup, sample_logits / argmax. No new
// brotensor op is required for an FP32 CPU first cut.
//
// STATUS: stage 1 — config + safetensors loading. transcribe() throws a
// staged std::runtime_error naming the stage until the encoder/decoder land.

// Model hyperparameters. Read from `config.json` by Whisper::load. Zero
// defaults are placeholders overwritten on a real load; the fixed
// `sample_rate` is Whisper's invariant 16 kHz input rate.
struct WhisperConfig {
    int sample_rate              = 16000;  // Whisper input rate (fixed)

    int vocab_size               = 0;
    int num_mel_bins             = 0;      // 80 for v1/v2 family, 128 for large-v3
    int d_model                  = 0;
    int max_source_positions     = 0;      // typically 1500 (30 s @ 50 Hz frame)
    int max_target_positions     = 0;      // typically 448

    int encoder_layers           = 0;
    int encoder_attention_heads  = 0;
    int encoder_ffn_dim          = 0;

    int decoder_layers           = 0;
    int decoder_attention_heads  = 0;
    int decoder_ffn_dim          = 0;

    // Special-token ids from the HF config — these are tokenizer-level facts
    // duplicated here so the decoder can run without holding a brolm handle.
    int pad_token_id             = 0;
    int eos_token_id             = 0;
    int decoder_start_token_id   = 0;      // <|startoftranscript|>
};

// The Whisper STT pipeline. Construct, load() a model directory, then
// transcribe(). Heavy state (weights, config, module graph) lives behind a
// pImpl so the public header stays free of brotensor module internals.
class Whisper {
public:
    Whisper();
    ~Whisper();
    Whisper(Whisper&&) noexcept;
    Whisper& operator=(Whisper&&) noexcept;
    Whisper(const Whisper&) = delete;
    Whisper& operator=(const Whisper&) = delete;

    // Load config.json + the safetensors weights from `model_dir`, placing
    // the weights on `device`. Throws std::runtime_error on a missing /
    // malformed model.
    void load(const std::string& model_dir,
              brotensor::Device device = brotensor::Device::CPU);

    // Result of a transcribe() call. `token_ids` is the raw decoded id
    // sequence (including SOT/lang/task prompt + content + EOS); callers
    // detokenize via brolm::whisper::Tokenizer::decode.
    struct Transcription {
        std::vector<int32_t> token_ids;
    };

    // Run the full pipeline: 16 kHz mono PCM -> token ids. `prompt_ids` is
    // the decoder prefix (typically the output of
    // brolm::whisper::Tokenizer::build_prompt(lang, task, with_timestamps)).
    // `max_new_tokens` caps the autoregressive loop and defaults to the
    // model's max_target_positions.
    Transcription transcribe(const AudioBuffer& audio,
                             const std::vector<int32_t>& prompt_ids,
                             int max_new_tokens = 0) const;

    const WhisperConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
