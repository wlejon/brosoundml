#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── Kokoro-82M ────────────────────────────────────────────────────────────
//
// Kokoro is an 82M-parameter text-to-speech model derived from StyleTTS 2.
// Unlike StyleTTS 2 it does not sample a style with a diffusion model — the
// "voice" is a precomputed embedding (a voice pack), so synthesis is a single
// deterministic forward pass: phoneme token ids + a voice -> 24 kHz waveform.
//
// Pipeline — the order brosoundml drives the components:
//
//   1. G2P          text -> phonemes. EXTERNAL: Kokoro uses the misaki G2P
//                   frontend. brosoundml takes phoneme token ids as its input
//                   and does not bundle a G2P engine.
//   2. plBERT       a phoneme-level BERT (ALBERT-style, weight-shared layers)
//                   that encodes the phoneme sequence into context features.
//   3. Text encoder phoneme embedding -> 3x (conv1d + LayerNorm + LeakyReLU)
//                   -> bidirectional LSTM. Produces per-phoneme features.
//   4. Predictor    conditioned on the voice embedding: a duration predictor
//                   (LSTM + projection) emits a per-phoneme frame count; a
//                   length regulator expands the phoneme features to frame
//                   rate; F0 (pitch) and energy are predicted at frame rate.
//   5. Decoder      an iSTFTNet generator: AdaIN residual blocks + transposed-
//                   conv upsampling conditioned on (features, F0, energy,
//                   voice), a harmonic+noise source excitation, a final layer
//                   emitting an STFT magnitude/phase pair, and an iSTFT head
//                   that returns the waveform.
//
// brotensor op coverage: conv1d / conv_transpose1d / pad1d, leaky_relu, snake,
// stft / istft, group_norm (instance norm via num_groups == C) + modulate (the
// AdaIN affine), embedding_lookup, and self-attention all map directly onto the
// existing op surface. The one missing primitive is a recurrent (LSTM) op;
// brosoundml composes the LSTM cell from matmul + sigmoid + tanh per timestep
// for now — a fused brotensor lstm op is a later performance optimisation.
//
// STATUS: the repo is stood up; the Kokoro forward pass is in build-out. See
// the build-out plan in README.md. Until it lands, load() / synthesize() throw
// a std::runtime_error that names the stage.

// Model hyperparameters. Concrete values are read from the model's config.json
// by Kokoro::load — the zero defaults here are placeholders, not real config.
struct KokoroConfig {
    int sample_rate = 24000;   // Kokoro output rate (fixed)
    int n_fft       = 0;       // iSTFT head transform size
    int hop_length  = 0;       // iSTFT hop
    int win_length  = 0;       // iSTFT window length
    int hidden_dim  = 0;       // text encoder / predictor hidden width
    int style_dim   = 0;       // half a voice embedding (a voice is 2*style_dim)
    int n_tokens    = 0;       // phoneme vocabulary size
    int max_context = 0;       // phoneme-count limit for one utterance
};

// A Kokoro voice pack: the precomputed style embedding that conditions the
// predictor and the decoder. A pack ships as a tensor indexed by phoneme count
// — Kokoro selects the row matching the utterance length — so the same voice
// adapts its prosody to short vs. long inputs.
struct Voice {
    std::string       name;
    brotensor::Tensor packs;   // (max_context, voice_dim) — one style row per length

    // Select the style row for an utterance of `n_phonemes` phonemes.
    brotensor::Tensor pick_for(int n_phonemes) const;
};

// The Kokoro TTS pipeline. Construct, load() a model directory, then
// synthesize(). Heavy state (weights, config, module graph) lives behind a
// pImpl so the public header stays free of brotensor module internals.
class Kokoro {
public:
    Kokoro();
    ~Kokoro();
    Kokoro(Kokoro&&) noexcept;
    Kokoro& operator=(Kokoro&&) noexcept;
    Kokoro(const Kokoro&) = delete;
    Kokoro& operator=(const Kokoro&) = delete;

    // Load config.json + the safetensors weights from `model_dir`, placing the
    // weights on `device`. Throws std::runtime_error on a missing / malformed
    // model.
    void load(const std::string& model_dir,
              brotensor::Device device = brotensor::Device::CPU);

    // Load a single voice pack from a weights file.
    Voice load_voice(const std::string& voice_path) const;

    // Run the full pipeline: phoneme token ids (see the misaki G2P note above)
    // + a voice -> a mono 24 kHz waveform. `speed` scales the predicted
    // durations: > 1 speaks faster, < 1 slower.
    AudioBuffer synthesize(const std::vector<int32_t>& phoneme_ids,
                           const Voice& voice,
                           float speed = 1.0f) const;

    const KokoroConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
