#pragma once

#include "brosoundml/audio.h"
#include "brosoundml/fastconformer.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── Parakeet (NVIDIA FastConformer-TDT ASR) ───────────────────────────────
//
// Parakeet-TDT-0.6B-v3 is NVIDIA's multilingual streaming-capable speech-to-
// text model: a FastConformer encoder feeding a Token-and-Duration Transducer
// (TDT) decoder. brosoundml targets the HuggingFace `transformers` Parakeet
// checkpoints — `config.json` + `model.safetensors` in a model directory —
// not the native NeMo `.nemo` archive. The tokenizer (a unified SentencePiece
// unigram, 8192 BPE pieces + a blank) is the caller's responsibility, exactly
// as with Whisper: brosoundml emits token ids (and their encoder-frame
// positions for word timestamps); the caller detokenizes.
//
// Pipeline — the order brosoundml drives the components:
//
//   1. Log-mel front-end   16 kHz mono PCM -> (num_mel_bins=128, T) log-mel,
//                          NeMo recipe: pre-emphasis 0.97, STFT (n_fft=512,
//                          win=400, hop=160, Hann, center), power spectrum,
//                          Slaney mel filterbank, log(x + 2^-24), per-feature
//                          (per-mel-bin) mean/var normalization over time.
//   2. FastConformer enc   8x depthwise-separable conv2d subsampling, then 24
//                          Conformer blocks (½-FFN macaron, Transformer-XL
//                          relative-position MHA, conv module
//                          [pointwise→GLU→depthwise k=9→BatchNorm→SiLU→
//                          pointwise], ½-FFN, final LayerNorm). Outputs
//                          (T/8, d_model=1024); a projector maps that to the
//                          decoder width (640).
//   3. TDT decoder         a prediction network (token embedding + 2-layer
//                          LSTM + projection) and a joint network
//                          (relu(enc_proj + dec_proj) -> token+duration
//                          logits). Greedy TDT decode emits a token plus a
//                          frame-duration per step, skipping `duration`
//                          encoder frames at once — the source of TDT's speed.
//
// brotensor op coverage: stft + complex magnitude + matmul (mel front-end),
// conv2d (subsampling + conv module, depthwise via groups), silu, layernorm,
// batch_norm_inference, self_attention_bias_forward (the rel-pos attention,
// with the Transformer-XL position term supplied as the additive bias),
// embedding_lookup, matmul. No new brotensor op is required.
//
// Device-neutral: load(device) places every weight on the chosen backend and
// the forward pass dispatches through brotensor device ops (CPU / CUDA /
// Metal), the same bar Qwen3-TTS / Qwen3-ASR meet. FP32 throughout.

// Encoder hyperparameters — the nested `encoder_config` of the HF config.json.
// Parakeet rides the shared FastConformer encoder (see fastconformer.h); the
// HF Parakeet export carries no projection/conv biases and does not xscale, so
// the defaults already match.
using ParakeetEncoderConfig = FastConformerConfig;

// Top-level Parakeet-TDT hyperparameters — the HF config.json.
struct ParakeetConfig {
    int sample_rate          = 16000;  // model input rate (fixed)

    int vocab_size           = 8193;   // 8192 SentencePiece pieces + blank
    int blank_token_id       = 8192;   // TDT blank (distinct from pad)
    int pad_token_id         = 2;

    int decoder_hidden_size  = 640;    // LSTM + joint width
    int num_decoder_layers   = 2;      // prediction-network LSTM layers
    int max_symbols_per_step = 10;     // greedy emissions cap per frame
    // hidden_act of the joint network is ReLU (fixed).

    std::vector<int> durations = {0, 1, 2, 3, 4};  // TDT frame-skip choices

    ParakeetEncoderConfig encoder;

    // Seconds of audio one encoder frame spans: subsampling_factor * hop /
    // sample_rate = 8 * 160 / 16000 = 0.08 s. Used to map a token's encoder-
    // frame index to a wall-clock timestamp.
    double frame_seconds() const {
        return static_cast<double>(encoder.subsampling_factor) * 160.0 /
               static_cast<double>(sample_rate);
    }
};

// Opaque per-session decode state (the TDT prediction-network LSTM state),
// defined in parakeet.cpp so the public header stays free of the decoder's
// brotensor module internals.
struct ParakeetSessionState;
class ParakeetSession;

// The Parakeet STT pipeline. Construct, load() a model directory, then
// transcribe(). Heavy state (weights, config, module graph) lives behind a
// pImpl so the public header stays free of brotensor module internals.
class Parakeet {
public:
    Parakeet();
    ~Parakeet();
    Parakeet(Parakeet&&) noexcept;
    Parakeet& operator=(Parakeet&&) noexcept;
    Parakeet(const Parakeet&) = delete;
    Parakeet& operator=(const Parakeet&) = delete;

    // Load config.json + model.safetensors from `model_dir`, placing every
    // weight on `device`. Throws std::runtime_error on a missing / malformed
    // model.
    void load(const std::string& model_dir,
              brotensor::Device device = brotensor::Device::CPU);

    // Result of a transcribe() call. `token_ids` is the raw greedy TDT id
    // stream (SentencePiece piece ids, no blank/pad — the caller detokenizes).
    // `token_frames[i]` is the encoder-frame index at which token_ids[i] was
    // emitted; multiply by config().frame_seconds() for a start timestamp.
    struct Transcription {
        std::vector<int32_t> token_ids;
        std::vector<int32_t> token_frames;
    };

    // Streaming sink: invoked once per emitted token, in decode order. Empty
    // (the default) = no streaming. Runs synchronously on the decode thread.
    using TokenCallback = std::function<void(int32_t token_id)>;

    struct TranscribeOptions {
        // Cap on total emitted tokens; 0 => no cap (decode the whole clip).
        int max_new_tokens = 0;
        // Polled once per encoder frame advance: true => stop and return what
        // we have. Empty (the default) = no cancel.
        CancelCheck   cancel   = {};
        // Invoked once per emitted token. Empty (the default) = no streaming.
        TokenCallback on_token = {};
    };

    // Run the full pipeline: 16 kHz mono PCM -> token ids + frame positions.
    Transcription transcribe(const AudioBuffer& audio,
                             const TranscribeOptions& opts) const;

    // Default-options overload. (A separate overload rather than a defaulted
    // `opts = {}` argument: GCC 12 rejects a brace-init default argument for a
    // nested struct that has default member initializers — CWG2335.)
    Transcription transcribe(const AudioBuffer& audio) const;

    // ── Multi-stream session API (load-once weights, N decode sessions) ──────
    //
    // Allocate a fresh per-stream decode session — its own TDT prediction-net
    // LSTM state, on the model's device. N streams = one shared net + N
    // sessions; hold the model through a std::shared_ptr<const Parakeet> and
    // call these `const` methods on it. Mirrors Whisper::make_session(): the
    // session is bare per-decode scratch, the weights stay read-only in the
    // shared model. Throws if no model is loaded.
    //
    // Concurrency tier: CONCURRENT (the strongest — same as the conv streamers).
    // Parakeet's forward touches NO shared mutable model state: the prediction
    // LSTM (h, c) lives in the session, and the encoder / joint are pure `const`
    // forwards over caller-supplied buffers. So sessions are fully independent —
    // transcribe(session, ...) over two sessions never cross-talks, on CPU or
    // CUDA (the GPU's single stream serializes execution, never correctness).
    // The legacy transcribe(audio, ...) overloads already run a self-contained
    // decode with a local state, so this API is about parity, explicit
    // weight-sharing, and giving the per-decode state a home for future
    // cross-chunk streaming.
    ParakeetSession make_session() const;

    Transcription transcribe(ParakeetSession& session,
                             const AudioBuffer& audio,
                             const TranscribeOptions& opts) const;

    Transcription transcribe(ParakeetSession& session,
                             const AudioBuffer& audio) const;

    // Reset a session's prediction state to the start-of-utterance zero state.
    // Each transcribe() already re-inits per call, so this is only needed to
    // drop state a caller is holding between calls. Const.
    void reset(ParakeetSession& session) const;

    const ParakeetConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// A Parakeet decode session: the per-stream TDT prediction-network state, owned
// by the caller, that turns one load-once Parakeet into N independent
// transcription streams (see Parakeet::make_session and
// docs/multi-stream-sessions.md). Bare scratch — it does not own the model; hold
// the weights through a std::shared_ptr<const Parakeet> and pass both to
// transcribe(). Move-only (the state is device tensors). Build with
// Parakeet::make_session(); the LSTM-state internals stay behind the opaque
// ParakeetSessionState pImpl.
class ParakeetSession {
public:
    ParakeetSession();
    ~ParakeetSession();
    ParakeetSession(ParakeetSession&&) noexcept;
    ParakeetSession& operator=(ParakeetSession&&) noexcept;
    ParakeetSession(const ParakeetSession&) = delete;
    ParakeetSession& operator=(const ParakeetSession&) = delete;

private:
    friend class Parakeet;
    std::unique_ptr<ParakeetSessionState> state_;
};

}  // namespace brosoundml
