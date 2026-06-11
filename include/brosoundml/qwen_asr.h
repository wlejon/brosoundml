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

    // <asr_text> — the separator the generated stream places between the
    // language ID and the transcript. A tokenizer-level invariant of the model
    // family (like sample_rate), not present in config.json — callers split
    // the id stream on it rather than on detokenized text (BPE decoders that
    // don't carry the added-tokens table render it as an empty string).
    int asr_text_token_id = 151704;

    // ── latent-tap geometry (the encode() surface) ──
    // The post-projector latents encode() emits are (T, latent_dim) at
    // latent_hz frames/second — exactly the rows the decoder consumes over the
    // <|audio_pad|> block. latent_dim == output_dim (the decoder hidden width);
    // latent_hz is the encoder's token rate: the mel front-end runs at
    // sample_rate / 160 frames/s and the Conv2d stem downsamples time 8x
    // (three stride-2 convs), so latent_hz = sample_rate / 160 / 8 = 12.5 Hz at
    // 16 kHz. Both are derived by load(); they are 0 before a load.
    int   latent_dim = 0;               // = output_dim (1024)
    float latent_hz  = 0.0f;            // 12.5 (latents per second)
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
    // "language <Language><asr_text>transcript" format — callers split the
    // id stream on config().asr_text_token_id, then detokenize each side via
    // brolm::qwen::Tokenizer::decode.
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
                             const TranscribeOptions& opts) const;

    // Default-options overload. (A separate overload rather than a defaulted
    // `opts = {}` argument: GCC 12 rejects a brace-init default argument for a
    // nested struct that has default member initializers — CWG2335.)
    Transcription transcribe(const AudioBuffer& audio) const;

    // Latent tap: run the AuT encoder + projector ONLY (no decoder) and return
    // the post-projector latents as a (T, config().latent_dim) FP32 tensor on
    // the model device, at config().latent_hz frames/second. These are exactly
    // the rows transcribe() splices into the decoder over the <|audio_pad|>
    // block — transcribe() calls encode() internally, so there is one encoder
    // path, not two. Throws before load() or on the same audio errors as
    // transcribe(). Same sample-rate / minimum-length contract.
    brotensor::Tensor encode(const AudioBuffer& audio) const;

    // Host-copy latent tap: same latents as encode(), copied into `out` as a
    // row-major (T, latent_dim) FP32 buffer (resized to T*latent_dim) on the
    // host regardless of the model device, and T returned. A bridge harness
    // that drives a decoder on a different device/library (e.g. brolm) needs
    // the latents on the host before re-uploading them to its own device.
    int encode_to_host(const AudioBuffer& audio, std::vector<float>& out) const;

    const QwenAsrConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ─── QwenAsrStream ──────────────────────────────────────────────────────────
//
// Incremental, mic-feed encoding over the AuT encoder + projector — the
// streaming counterpart of QwenAsr::encode(). feed() buffers samples to the
// next block boundary, runs the encoder over each completed block, and appends
// that block's projected latents. A block's latents are FINAL the moment the
// block completes: the encoder's attention is windowed within a block and never
// crosses block boundaries, so no later audio can revise an emitted latent and
// nothing is ever re-encoded.
//
// Block size is a config knob of `block_chunks` conv-chunks. One conv-chunk is
// n_window*2 mel frames (100 = ~1 s at 16 kHz) and yields ~13 latents, so a
// block is 1–8 s of audio (clamped into the [1, n_window_infer/chunk] range the
// model's dynamic attention windows support); the default 1 chunk emits ~13
// latents per ~1 s with the lowest latency.
//
// Each block is encoded exactly as one-shot encode() over that block of audio —
// block-local log-mel normalization and a single block-wide attention window —
// so a clip that fits in one block streams bit-identically to encode(). Over
// several blocks the stream diverges from a global encode() only by per-block
// mel normalization and the block-bounded attention window; that is the
// inherent, intended cost of bounded-latency streaming, and the same dynamic
// windowing the model was trained for. The encoder runs on the chosen device;
// latents are accumulated on the host (where a cross-device/library decoder
// bridge consumes them).
//
// Poll-free, single-producer (like WakeWord::feed): feed()/finish() return the
// number of newly finalized latent rows; no two threads may call feed()
// concurrently.
class QwenAsrStream {
public:
    QwenAsrStream();
    ~QwenAsrStream();
    QwenAsrStream(QwenAsrStream&&) noexcept;
    QwenAsrStream& operator=(QwenAsrStream&&) noexcept;
    QwenAsrStream(const QwenAsrStream&) = delete;
    QwenAsrStream& operator=(const QwenAsrStream&) = delete;

    // Load the encoder ONLY (config.json + the thinker.audio_tower.* weights —
    // no decoder) from `model_dir`, placing weights on `device`. `block_chunks`
    // is the block size in conv-chunks, clamped to [1, n_window_infer/chunk].
    // Throws std::runtime_error on a missing / malformed model.
    void load(const std::string& model_dir, int block_chunks = 1,
              brotensor::Device device = brotensor::Device::CPU);

    // Optional per-block sink: invoked once per finalized block with a host
    // pointer to that block's (n_rows, latent_dim) FP32 latents (row-major).
    // The pointer is valid only for the duration of the call.
    using BlockCallback = std::function<void(const float* rows, int n_rows)>;

    // Feed `n` mono 16 kHz samples. Encodes every block that completed on this
    // call and appends its latents, returning the number of newly finalized
    // latent rows (0 if no block boundary was crossed). Throws before load().
    int feed(const float* samples, int n, const BlockCallback& on_block = {});

    // Flush the trailing partial block (the buffered audio shorter than one full
    // block) as a final, shorter block. Returns the number of newly finalized
    // latent rows (0 if nothing is buffered). Idempotent once drained.
    int finish(const BlockCallback& on_block = {});

    // All finalized latents so far, row-major (frames(), config().latent_dim) on
    // the host. Grows as feed()/finish() finalize blocks.
    const std::vector<float>& latents() const;
    int frames() const;          // total finalized latent rows
    int block_chunks() const;    // block size in conv-chunks
    int block_frames() const;    // mel frames per full block (block_chunks*chunk)

    const QwenAsrConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
