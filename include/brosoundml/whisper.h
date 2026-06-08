#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <functional>
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
// STATUS: complete. load() reads config.json + the safetensors weights onto
// the requested device; transcribe() runs the full log-mel ▶ encoder ▶
// autoregressive-decoder forward pass and returns token ids. The
// TranscribeOptions overload adds per-token streaming (on_token) and sequential
// long-form decode (audio > 30 s windowed into 30 s segments with timestamp
// seek) on top of the one-shot path.

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
    // detokenize via brolm::whisper::Tokenizer::decode. In long-form mode the
    // prompt prefix is emitted once, followed by the concatenated content of
    // every 30 s window (see TranscribeOptions::timestamp_begin_id).
    struct Transcription {
        std::vector<int32_t> token_ids;
    };

    // Streaming sink: invoked once per generated token, in decode order, with
    // the freshly produced id — the same ids that land in
    // Transcription::token_ids (content + any timestamp tokens; EOS is not
    // delivered, the greedy loop breaks on it). Lets a caller detokenize and
    // emit partial text mid-decode instead of waiting for the whole utterance.
    // Runs synchronously on the decode thread between steps — keep it cheap.
    // Empty (the default) = no streaming.
    using TokenCallback = std::function<void(int32_t token_id)>;

    // Options for the richer transcribe() overload — streaming emission and
    // long-form (>30 s) windowing on top of the legacy one-shot decode.
    struct TranscribeOptions {
        // Cap on the autoregressive loop; 0 => max_target_positions -
        // prompt_len. In long-form mode this caps EACH 30 s window independently.
        int max_new_tokens = 0;

        // Polled once per decoded token (the dominant cost): true => stop and
        // return what we have. Empty (the default) = no cancel.
        CancelCheck cancel = {};

        // Invoked once per generated token, as it is produced (see
        // TokenCallback). Empty (the default) = no streaming.
        TokenCallback on_token = {};

        // Long-form seek anchor: the id of the `<|0.00|>` timestamp token
        // (brolm::whisper::Tokenizer::first_timestamp_id()). When set (>= 0)
        // AND the audio is longer than 30 s, transcribe() windows the audio
        // into 30 s segments and advances the window by the last timestamp the
        // decoder emits in each segment — Whisper's standard sequential
        // long-form decode. When < 0 (the default) audio beyond 30 s is
        // truncated to the first window, matching the legacy behavior. For seek
        // to engage the prompt must request timestamps (built with
        // with_timestamps=true); a no-timestamps prompt degrades to fixed 30 s
        // hops. brosoundml stays tokenizer-free — only this single int crosses
        // the boundary; the caller still owns build_prompt / decode.
        int timestamp_begin_id = -1;
    };

    // Run the full pipeline: 16 kHz mono PCM -> token ids. `prompt_ids` is
    // the decoder prefix (typically the output of
    // brolm::whisper::Tokenizer::build_prompt(lang, task, with_timestamps)).
    // `max_new_tokens` caps the autoregressive loop and defaults to the
    // model's max_target_positions.
    //
    // `cancel` is checked once per decoded token (the dominant cost): when it
    // returns true the greedy loop stops and the call returns the prompt plus
    // whatever tokens were produced so far. The one-shot encode + prompt
    // prefill that precede the loop are not interruptible, but they are a small
    // fraction of a multi-second transcription. Empty (the default) = no cancel.
    //
    // This legacy overload always decodes a single 30 s window (audio beyond
    // 30 s is truncated). For per-token streaming or long-form windowing use
    // the TranscribeOptions overload below.
    Transcription transcribe(const AudioBuffer& audio,
                             const std::vector<int32_t>& prompt_ids,
                             int max_new_tokens = 0,
                             const CancelCheck& cancel = {}) const;

    // Streaming / long-form transcribe. Same pipeline as above, with per-token
    // emission (opts.on_token) and, when opts.timestamp_begin_id is set, an
    // arbitrary-length input windowed into 30 s segments instead of truncated
    // (see TranscribeOptions). Returns the full token stream just like the
    // legacy overload; on_token has already delivered each content token by the
    // time it returns.
    Transcription transcribe(const AudioBuffer& audio,
                             const std::vector<int32_t>& prompt_ids,
                             const TranscribeOptions& opts) const;

    const WhisperConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
