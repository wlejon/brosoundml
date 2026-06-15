#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── Supertonic-3 — flow-matching multilingual TTS (Supertone) ───────────────
//
// Supertonic 3 is a ~99M-param, faster-than-realtime, on-device TTS shipped
// upstream as four ONNX graphs. brosoundml runs INFERENCE by recomposing each
// graph from brotensor ops (no onnxruntime, no model-specific kernels), so a
// CUDA / Metal build runs on the GPU unchanged. The pipeline:
//
//   text  --UnicodeProcessor-->  text_ids (codepoint-level; no G2P)
//   text_ids, style_ttl  --text_encoder-->        text_emb
//   text_ids, style_dp   --duration_predictor-->  durations
//   noise, text_emb, style_ttl --vector_estimator(xN)--> latent  (flow matching)
//   latent  --vocoder-->  44.1 kHz waveform
//
// The model directory is the converted layout produced by the out-of-repo
// `supertonic-convert` tool (per-model safetensors + tts.json + the codepoint
// frontend tables + voice_styles/ presets), as hosted under
// brosoundml-data/supertonic.
//
// THIS BUILD synthesizes end to end: synthesize() runs the UnicodeProcessor
// frontend (codepoint -> id, no G2P), then chains the four graphs — duration
// predictor (length), text encoder (conditioning), the flow-matching vector
// estimator (total_step guided Euler steps), and the vocoder (latent ->
// waveform). The four graphs are also exposed as standalone stages
// (encode_text / predict_duration / denoise / decode) for parity testing.

// A voice preset: the two style matrices a preset JSON carries, as consumed by
// the duration predictor and the rest of the pipeline. Load one with
// Supertonic::load_voice_style().
struct VoiceStyle {
    std::vector<float> ttl;  // 50*256 token-major (style_ttl[s*256 + c])
    std::vector<float> dp;   // 8*16 row-major   (style_dp[i*16 + j])
};

// Model hyperparameters, read from the converted tts.json by Supertonic::load.
struct SupertonicConfig {
    int sample_rate = 44100;  // output rate (Hz)
    int latent_dim  = 24;     // autoencoder latent channels (ldim)
    int chunk       = 6;      // chunk_compress_factor: vector-field works in
                              // latent_dim*chunk = 144 channels; the vocoder
                              // de-chunks 144 -> (latent_dim, chunk*frames)
    int base_chunk  = 512;    // waveform samples synthesised per de-chunked frame
};

// The Supertonic TTS pipeline. Construct, load() a converted model directory,
// then (for now) decode() a flow-field latent to a waveform. Heavy state lives
// behind a pImpl so this header stays free of brotensor module internals.
//
// Multi-stream (docs/multi-stream-sessions.md): decode() is `const` and
// stateless, so one loaded Supertonic behind a std::shared_ptr<const Supertonic>
// serves N streams with zero cross-talk. (The autoregressive flow-matching loop
// will push its per-stream scratch into a SupertonicSession when it lands.)
class Supertonic {
public:
    Supertonic();
    ~Supertonic();
    Supertonic(Supertonic&&) noexcept;
    Supertonic& operator=(Supertonic&&) noexcept;
    Supertonic(const Supertonic&) = delete;
    Supertonic& operator=(const Supertonic&) = delete;

    // Load the converted model directory (config.json + tts.json + the per-model
    // *.safetensors), placing weights on `device`. Throws std::runtime_error on
    // a missing / malformed model. Only the vocoder weights are required today.
    void load(const std::string& model_dir,
              brotensor::Device device = brotensor::Device::CPU);

    // Vocoder: a flow-field latent -> waveform (mono, sample_rate Hz).
    //
    // `latent` is the vector_estimator's output BEFORE de-normalisation: a
    // (channels x frames) row-major, channel-major grid (latent[c*frames + t]),
    // with channels == latent_dim*chunk (144). The vocoder de-chunks it to
    // (latent_dim, chunk*frames), de-normalises with the autoencoder's stored
    // mean/std, and synthesises chunk*base_chunk (3072) samples per input frame.
    AudioBuffer decode(const float* latent, int channels, int frames) const;
    AudioBuffer decode(const std::vector<float>& latent, int channels,
                       int frames) const;

    // Decode a REAL (de-chunked, de-normalised) latent [latent_dim, frames_dechunked]
    // straight through the vocoder conv stack — the inverse pairing for the AE
    // encoder, whose output is exactly this. Unlike decode(), it does no de-chunk
    // or de-normalise; it synthesises base_chunk (512) samples per de-chunked
    // frame. Used by the encoder reconstruction loop. Throws if not loaded.
    AudioBuffer decode_real(const float* latent_real, int latent_dim,
                            int frames_dechunked) const;

    // Text encoder: token ids + a TTL style matrix -> conditioning embedding.
    //
    // `text_ids` are codepoint-level vocabulary ids (the UnicodeProcessor
    // output); `style_ttl` is the voice preset's TTL style, 50*256 row-major
    // (token-major: style_ttl[s*256 + c], 50 style tokens x 256). Returns the
    // text embedding as a (256 x T) channel-major grid (emb[c*T + t]), with
    // T == text_ids.size(). Processes a single unpadded sequence (the upstream
    // length mask is implicitly all-ones). Throws if the text-encoder weights
    // are absent or the style matrix is mis-sized.
    std::vector<float> encode_text(const std::vector<int>& text_ids,
                                   const std::vector<float>& style_ttl) const;

    // Duration predictor: token ids + a DP style matrix -> a single scalar total
    // duration for the utterance. The flow-matching estimator does its own
    // (RoPE cross-attention) text↔audio alignment, so this predicts only the
    // overall length, not a per-token duration.
    //
    // `text_ids` are the same codepoint-level vocabulary ids as encode_text;
    // `style_dp` is the voice preset's DP style, 8*16 = 128 floats row-major
    // (style_dp[i*16 + j]). The encoder prepends a learned sentence token,
    // pools it after a 6-block ConvNeXt + 2-layer relative-attention stack, and
    // an MLP head (Gemm/PReLU/Gemm/Exp) maps the pooled vector + style to the
    // positive scalar. Throws if the duration-predictor weights are absent or
    // the style matrix is mis-sized.
    float predict_duration(const std::vector<int>& text_ids,
                           const std::vector<float>& style_dp) const;

    // Flow-matching vector estimator: one classifier-free-guided denoising step.
    //
    // Given the current `noisy_latent` (channels==144 working channels x frames,
    // channel-major: noisy[c*frames + t]), the text conditioning `text_emb`
    // (256 x text_len channel-major, from encode_text) and the voice preset's
    // TTL style (50*256 token-major, as for encode_text), runs the DiT vector
    // field for both the conditional and unconditional prompt and returns the
    // updated latent after one Euler step with guidance baked in:
    //   denoised = noisy + (1/total_step) * (4*field_cond - 3*field_uncond).
    // `current_step`/`total_step` set the flow time t = current_step/total_step.
    // synthesize() calls this total_step times, feeding the result back as the
    // next noisy_latent. Returns the (144 x frames) channel-major latent. Throws
    // if the vector-estimator weights are absent or an argument is mis-sized.
    std::vector<float> denoise(const std::vector<float>& noisy_latent,
                               int channels, int frames,
                               const std::vector<float>& text_emb, int text_len,
                               const std::vector<float>& style_ttl,
                               int current_step, int total_step) const;

    // Load a voice preset JSON (the converted layout's voice_styles/<name>.json:
    // an object with style_ttl / style_dp, each {"dims":[1,N,M],"data":[...]}).
    // `path` is the only filesystem touch and is handed in by the caller. Throws
    // if the file is missing / malformed or a style matrix is mis-sized.
    VoiceStyle load_voice_style(const std::string& path) const;

    // UnicodeProcessor frontend: a UTF-8 string + ISO language tag -> token ids.
    // Mirrors the upstream codepoint pipeline — NFKD normalize, emoji/symbol
    // strip, punctuation tidy-up, a `<lang>…</lang>` wrap — then maps each
    // codepoint through unicode_indexer.json (out-of-vocab wraps to the embedding
    // table's last row, as ONNX Gather does). `lang` must be one of the 31
    // supported codes (or "na"). Throws if the frontend tables are absent.
    std::vector<int> text_to_ids(const std::string& text,
                                 const std::string& lang) const;

    // End-to-end synthesis: text + language + voice preset -> waveform.
    //
    // Runs the full pipeline — text_to_ids, predict_duration (scaled by 1/speed),
    // encode_text, a `total_step`-iteration classifier-free-guided Euler loop
    // through the vector estimator seeded from N(0,1) noise (Philox `seed`), and
    // the vocoder. `speed` > 1 shortens the utterance (upstream default 1.05);
    // `total_step` is the number of flow-matching steps (upstream default 8).
    // `guidance` is the classifier-free-guidance scale w: each Euler step applies
    // field = (1+w)*cond - w*uncond, so w=3 (default) reproduces upstream; raise
    // it to push harder toward the text/style conditioning (crisper, more
    // articulated), lower it to relax toward the unconditional field (flatter,
    // breathier). Clamped to >= 0. Returns mono FP32 PCM at config().sample_rate.
    // Throws if any stage's weights or the frontend tables are absent.
    AudioBuffer synthesize(const std::string& text, const std::string& lang,
                           const VoiceStyle& voice, int total_step = 8,
                           float speed = 1.05f, std::uint64_t seed = 0,
                           float guidance = 3.0f) const;

    // Long-form synthesis: split `text` into sentences, synthesize each through
    // synthesize() (so each chunk gets its own predicted duration and flow-field
    // pass at a tractable length), and concatenate with `gap_seconds` of silence
    // between sentences. The flow loop is graph-accelerated per chunk, and the
    // fixed per-chunk overheads amortise better over longer text.
    //
    // Sentence boundaries are found by a Unicode-aware scanner (not std::regex —
    // it can't express the lookbehind needed): a split lands at sentence-final
    // punctuation (. ! ? … and the CJK 。！？) followed by whitespace or end of
    // text, skipping decimals (a '.' between digits). Any sentence still longer
    // than `max_chars` codepoints is further split at a whitespace near the limit
    // so no single chunk drives the field to an unwieldy latent length. `seed` is
    // advanced per chunk (seed + chunk_index) so chunks differ yet stay
    // deterministic. With a single sentence this is just synthesize() (no gap).
    // `guidance` is the classifier-free-guidance scale (see synthesize()).
    AudioBuffer synthesize_long(const std::string& text, const std::string& lang,
                                const VoiceStyle& voice, int total_step = 8,
                                float speed = 1.05f, std::uint64_t seed = 0,
                                float gap_seconds = 0.3f, int max_chars = 300,
                                float guidance = 3.0f) const;

    // The sentence splitter used by synthesize_long, exposed for callers that
    // want to chunk text themselves (e.g. to stream or batch). Stateless — needs
    // no loaded model. Boundaries land at sentence-final punctuation (. ! ? … and
    // the CJK 。！？) followed by whitespace or end of text, skipping decimals and
    // common abbreviations / single-letter initials; trailing quotes/brackets
    // stay with the sentence; any chunk longer than `max_chars` codepoints is
    // broken at a nearby whitespace. Returns trimmed, non-empty UTF-8 chunks.
    static std::vector<std::string> split_sentences(const std::string& text,
                                                    int max_chars = 300);

    const SupertonicConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
