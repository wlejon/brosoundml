#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

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
// THIS BUILD implements the vocoder stage (latent -> waveform), the text
// encoder (text_ids + TTL style -> text_emb), and the duration predictor
// (text_ids + DP style -> scalar total duration). The flow-matching estimator
// lands next; until then synthesize() is unavailable and the public entry
// points are encode_text(), predict_duration(), and decode().

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

    const SupertonicConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
