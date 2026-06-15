#pragma once

#include "brosoundml/audio.h"
#include "brosoundml/fastconformer.h"

#include <brotensor/tensor.h>

#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── Streaming Sortformer (NVIDIA speaker diarization) ──────────────────────
//
// diar_streaming_sortformer_4spk is NVIDIA's end-to-end streaming speaker
// diarizer: a NEST (Fast-Conformer) acoustic encoder feeding a Transformer
// encoder that emits, per 80 ms frame, an independent sigmoid activity
// probability for up to four speakers, with speaker labels assigned in
// arrival-time order. brosoundml targets the converted checkpoint —
// `config.json` + `model.safetensors` (see scripts/convert-sortformer.py) —
// not the native NeMo `.nemo` archive.
//
// Pipeline — the order brosoundml drives the components:
//
//   1. Log-mel front-end   16 kHz mono PCM -> (128, T) log-mel, NeMo recipe:
//                          pre-emphasis 0.97, STFT (n_fft=512, win=400, hop=160,
//                          symmetric Hann, center), power spectrum, Slaney mel
//                          filterbank, log(x + 2^-24). NO per-feature
//                          normalization (preprocessor normalize = NA).
//   2. FastConformer enc   8x conv subsampling (pre-encode) + 17 Conformer
//                          blocks (xscaling, projection/conv biases). Outputs
//                          (T/8, fc_d_model=512) at an 80 ms frame step. The
//                          shared FastConformerEncoder backs both this and
//                          Parakeet.
//   3. Sortformer head     encoder_proj 512 -> tf_d_model=192, an 18-layer
//                          post-LN Transformer encoder (inner 768, 8 heads,
//                          ReLU FFN), then a two-layer sigmoid head
//                          (192 -> 192 -> num_spks) -> (T, num_spks) speaker
//                          activity probabilities in [0, 1].
//
// Offline diarize() runs the whole clip in one pass (the NeMo non-streaming
// forward_infer path). Streaming inference with the Arrival-Order Speaker Cache
// rides a per-stream session (see make_session()).
//
// Device-neutral: load(device) places every weight on the chosen backend and
// the forward dispatches through brotensor device ops (CPU / CUDA / Metal),
// FP32 throughout. A CUDA build reproduces the CPU result (~1e-5 on the sigmoid
// tail).

// Sortformer transformer-head hyperparameters — `transformer_config`.
struct SortformerTransformerConfig {
    int  num_layers            = 18;
    int  hidden_size           = 192;   // tf_d_model
    int  inner_size            = 768;   // FFN inner width
    int  num_attention_heads   = 8;
    bool pre_ln                = false; // NeMo pre_ln=false => post-LN blocks
};

// Arrival-Order Speaker Cache / streaming hyperparameters — `streaming_config`.
// Lengths are in 80 ms diarization frames.
struct SortformerStreamingConfig {
    int   spkcache_len                = 188;
    int   fifo_len                    = 0;
    int   chunk_len                   = 188;
    int   spkcache_update_period      = 188;
    int   chunk_left_context          = 1;
    int   chunk_right_context         = 1;
    int   spkcache_sil_frames_per_spk = 3;
    float pred_score_threshold        = 0.25f;
    float scores_boost_latest         = 0.05f;
    float sil_threshold               = 0.2f;
    float strong_boost_rate           = 0.75f;
    float weak_boost_rate             = 1.5f;
    float min_pos_scores_rate         = 0.5f;
};

// Top-level Sortformer hyperparameters — the converted config.json.
struct SortformerConfig {
    int sample_rate = 16000;   // model input rate (fixed)
    int num_spks    = 4;       // maximum simultaneous speakers
    int fc_d_model  = 512;     // FastConformer encoder width
    int tf_d_model  = 192;     // Transformer-head width

    FastConformerConfig            encoder;
    SortformerTransformerConfig    transformer;
    SortformerStreamingConfig      streaming;

    // Seconds of audio one diarization frame spans: subsampling_factor * hop /
    // sample_rate = 8 * 160 / 16000 = 0.08 s.
    double frame_seconds() const {
        return static_cast<double>(encoder.subsampling_factor) * 160.0 /
               static_cast<double>(sample_rate);
    }
};

// Opaque per-stream streaming state (the Arrival-Order Speaker Cache + FIFO),
// defined in sortformer.cpp so the public header stays free of brotensor module
// internals.
struct SortformerSessionState;
class SortformerSession;

// The Sortformer diarization pipeline. Construct, load() a model directory, then
// diarize(). Heavy state (weights, config, module graph) lives behind a pImpl.
class Sortformer {
public:
    Sortformer();
    ~Sortformer();
    Sortformer(Sortformer&&) noexcept;
    Sortformer& operator=(Sortformer&&) noexcept;
    Sortformer(const Sortformer&) = delete;
    Sortformer& operator=(const Sortformer&) = delete;

    // Load config.json + model.safetensors from `model_dir`, placing every
    // weight on `device`. Throws std::runtime_error on a missing / malformed
    // model.
    void load(const std::string& model_dir,
              brotensor::Device device = brotensor::Device::CPU);

    // Per-frame speaker-activity probabilities. `probs` is row-major
    // (num_frames, num_speakers); probs[t*num_speakers + s] is the probability
    // speaker s is active in frame t (in [0, 1]). Speakers are ordered by
    // arrival time. Frame t starts at t * frame_seconds seconds.
    struct Diarization {
        int                num_frames    = 0;
        int                num_speakers  = 0;
        std::vector<float> probs;          // (num_frames * num_speakers)
        double             frame_seconds  = 0.08;
    };

    // Offline diarization: run the whole clip through the encoder + transformer
    // head in one pass (the NeMo non-streaming forward_infer path). 16 kHz mono
    // PCM in; a (T, num_spks) probability matrix out. Throws on misuse.
    Diarization diarize(const AudioBuffer& audio) const;

    // ── Streaming session API (load-once weights, N diarization sessions) ────
    //
    // Allocate a fresh per-stream session — its own Arrival-Order Speaker Cache
    // and FIFO queue, on the model's device. Feed audio chunk-by-chunk; each
    // feed() returns the speaker activity for the frames that chunk finalized.
    // Hold the model through a std::shared_ptr<const Sortformer> and call these
    // `const` methods on it.
    //
    // Concurrency tier: SERIALIZED — the streaming forward re-runs the encoder
    // over [spkcache | fifo | chunk] each step and shares the model's single GPU
    // stream; correctness is per-session (the cache lives in the session), but
    // throughput serializes across sessions on one device.
    SortformerSession make_session() const;

    // Feed the next contiguous block of 16 kHz mono PCM for `session`. Returns
    // the speaker activity for the diarization frames this call finalized
    // (appended in time order across calls). Pass `is_last = true` on the final
    // block to flush the FIFO tail. Throws if no model is loaded.
    Diarization feed(SortformerSession& session, const AudioBuffer& audio,
                     bool is_last = false) const;

    // Reset a session to the start-of-stream empty-cache state.
    void reset(SortformerSession& session) const;

    const SortformerConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// A Sortformer streaming session: the per-stream Arrival-Order Speaker Cache and
// FIFO queue, owned by the caller, that turns one load-once Sortformer into N
// independent diarization streams. Move-only (the state is device tensors).
// Build with Sortformer::make_session(); internals stay behind the opaque
// SortformerSessionState pImpl.
class SortformerSession {
public:
    SortformerSession();
    ~SortformerSession();
    SortformerSession(SortformerSession&&) noexcept;
    SortformerSession& operator=(SortformerSession&&) noexcept;
    SortformerSession(const SortformerSession&) = delete;
    SortformerSession& operator=(const SortformerSession&) = delete;

private:
    friend class Sortformer;
    std::unique_ptr<SortformerSessionState> state_;
};

}  // namespace brosoundml
