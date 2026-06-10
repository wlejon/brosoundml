#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── Qwen3-ASR ──────────────────────────────────────────────────────────────
//
// Qwen3-ASR (Alibaba, Jan 2026) is the open speech-to-text series built from
// Qwen3-Omni: 52-language ASR + language ID. brosoundml targets the HF
// checkpoints (Qwen/Qwen3-ASR-0.6B / -1.7B) — `config.json` +
// `model.safetensors` in a model directory, plus the Qwen BPE tokenizer files
// consumed by brolm::qwen::Tokenizer (held by the caller; brosoundml itself
// takes already-tokenized context text and emits token ids).
//
// Pipeline — the order brosoundml drives the components:
//
//   1. Log-mel front-end  16 kHz mono PCM -> (frames, 128) log-mel features.
//                         Whisper's exact recipe (25 ms / 10 ms / Slaney /
//                         log10 / max-8 clamp / (x+4)/4) but NOT padded to
//                         30 s — the encoder consumes the true frame count.
//   2. AuT audio encoder  per-chunk Conv2d stem (3x 3x3 stride-2, applied to
//                         independent 100-frame chunks; 8x time downsample,
//                         12.5 Hz token rate) + per-chunk sinusoidal
//                         positions + 18 pre-LN Transformer layers with
//                         block-windowed bidirectional attention (~104-token
//                         windows) + a 2-layer projector into the decoder
//                         width. Outputs (n_audio_tokens, 1024).
//   3. Qwen3 text decoder The stock Qwen3 LLM (28 layers, GQA 16/8, per-head
//                         QK RMSNorm, RoPE θ=1e6, SwiGLU). The chat-template
//                         prompt embeds the audio tokens between
//                         <|audio_start|> / <|audio_end|>; the transcript is
//                         decoded greedily with a KV cache.
//   4. Tokenizer          brolm::qwen::Tokenizer (separate dep) maps
//                         id -> text. brosoundml returns the raw id stream.
//
// The generated stream is the model's native output format:
//   "language <Language><asr_text>transcript…"
// where <asr_text> is the special id 151704; callers split on it (or just
// decode everything and split on the literal once detokenized).
//
// STATUS: complete. load() reads config.json + the safetensors weights onto
// the requested device; transcribe() runs the full mel ▶ encoder ▶
// autoregressive-decoder pipeline and returns the generated token ids.

// Model hyperparameters, read from `config.json` (thinker_config) by
// QwenAsr::load. Zero defaults are placeholders overwritten on a real load;
// `sample_rate` is the model's invariant 16 kHz input rate.
struct QwenAsrConfig {
    int sample_rate = 16000;            // fixed model input rate

    // ── audio encoder (thinker_config.audio_config) ──
    int num_mel_bins            = 0;    // 128
    int d_model                 = 0;    // 896
    int encoder_layers          = 0;    // 18
    int encoder_attention_heads = 0;    // 14
    int encoder_ffn_dim         = 0;    // 3584
    int output_dim              = 0;    // 1024 (decoder hidden width)
    int downsample_hidden_size  = 0;    // 480 (conv stem channels)
    int n_window                = 0;    // 50 — conv chunk = n_window*2 mel frames
    int n_window_infer          = 0;    // 800 — attention window in mel frames
    int max_source_positions    = 0;    // 1500 (sinusoid table length upstream)

    // ── text decoder (thinker_config.text_config) ──
    int   hidden_size         = 0;      // 1024
    int   num_hidden_layers   = 0;      // 28
    int   num_attention_heads = 0;      // 16
    int   num_key_value_heads = 0;      // 8 (GQA)
    int   head_dim            = 0;      // 128
    int   intermediate_size   = 0;      // 3072
    int   vocab_size          = 0;      // 151936
    float rms_norm_eps        = 1e-6f;
    float rope_theta          = 1e6f;

    // ── special token ids (thinker_config / generation_config.json) ──
    int audio_start_token_id = 0;       // 151669  <|audio_start|>
    int audio_end_token_id   = 0;       // 151670  <|audio_end|>
    int audio_token_id       = 0;       // 151676  <|audio_pad|>
    std::vector<int32_t> eos_token_ids; // {151643, 151645}
};

// The Qwen3-ASR pipeline. Construct, load() a model directory, then
// transcribe(). Heavy state (weights, config, module graph) lives behind a
// pImpl so the public header stays free of module internals.
class QwenAsr {
public:
    QwenAsr();
    ~QwenAsr();
    QwenAsr(QwenAsr&&) noexcept;
    QwenAsr& operator=(QwenAsr&&) noexcept;
    QwenAsr(const QwenAsr&) = delete;
    QwenAsr& operator=(const QwenAsr&) = delete;

    // Load config.json + the safetensors weights from `model_dir`, placing
    // the weights on `device`. Throws std::runtime_error on a missing /
    // malformed model.
    void load(const std::string& model_dir,
              brotensor::Device device = brotensor::Device::CPU);

    // Result of a transcribe() call. `token_ids` is the GENERATED id stream
    // only (the chat-template prompt and the audio placeholder tokens are
    // not echoed; the trailing EOS is stripped). The stream is the model's
    // "language <Language><asr_text>transcript" format — callers detokenize
    // via brolm::qwen::Tokenizer::decode and split on <asr_text> (id 151704).
    struct Transcription {
        std::vector<int32_t> token_ids;
    };

    // Streaming sink: invoked once per generated token, in decode order, with
    // the freshly produced id — the same ids that land in
    // Transcription::token_ids (EOS is not delivered, the greedy loop breaks
    // on it). Runs synchronously on the decode thread between steps — keep it
    // cheap. Empty (the default) = no streaming.
    using TokenCallback = std::function<void(int32_t token_id)>;

    struct TranscribeOptions {
        // Cap on the autoregressive loop; 0 => 1024 (a generous transcript
        // budget — the model emits ~3 tokens/second of speech).
        int max_new_tokens = 0;

        // Optional context biasing: already-tokenized text (Qwen BPE ids from
        // brolm::qwen::Tokenizer::encode) placed in the system block of the
        // chat template. The model biases recognition toward names / domain
        // terms appearing in the context. Empty (the default) = no context.
        std::vector<int32_t> context_ids;

        // Polled once per decoded token (the dominant cost): true => stop and
        // return what we have. Empty (the default) = no cancel.
        CancelCheck cancel = {};

        // Invoked once per generated token (see TokenCallback). Empty (the
        // default) = no streaming.
        TokenCallback on_token = {};
    };

    // Run the full pipeline: 16 kHz mono PCM -> generated token ids. Throws
    // on a sample-rate mismatch (resampling is the caller's problem) or on
    // audio shorter than one mel hop (10 ms).
    Transcription transcribe(const AudioBuffer& audio,
                             const TranscribeOptions& opts = {}) const;

    const QwenAsrConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
