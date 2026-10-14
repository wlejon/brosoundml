#pragma once

#include "brosoundml/audio.h"
#include "brosoundml/yue2_vae.h"   // StereoAudio

#include <brotensor/tensor.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── HTDemucs source separation (Demucs v4 htdemucs_ft) ──────────────────────
//
// Music in, four stems out: drums, bass, other, vocals. htdemucs_ft (Meta,
// MIT code and weights) is a bag of four hybrid-transformer Demucs networks,
// each fine-tuned for one stem, so stem k is network k's output k and asking
// for only `vocals` runs one network, not four.
//
// A network sees 7.8 s segments at 44.1 kHz through two branches: a spectral
// one (STFT 4096 / hop 1024, complex as channels, convs over frequency) and a
// temporal one (strided convs over the waveform), joined by a five-layer
// cross transformer, decoded back to a spectrogram (iSTFT) and a waveform and
// summed. separate() reproduces demucs.apply.apply_model as sep.py drives it:
//
//   resample to 44.1 kHz (torchaudio's default sinc / Hann kernel, width 6,
//   rolloff 0.99), normalise by the mono mix's mean / std, cut 7.8 s segments
//   every 5.85 s (overlap 0.25), each centred in its context (zeros past the
//   ends), run them, overlap-add with triangular weights, de-normalise,
//   resample back to the input rate.
//
// Weights: a folder from tools/htdemucs_convert.py — config.json plus
// <stem>.safetensors per network, FP32. FP32 on every backend; GPU by default.

struct HtDemucsConfig {
    std::vector<std::string> sources;   // {"drums", "bass", "other", "vocals"}
    std::vector<std::string> files;     // per source: its network's safetensors
    int sample_rate     = 44100;        // the networks' rate
    int channels        = 2;
    int segment_samples = 343980;       // 7.8 s at 44.1 kHz
    double overlap      = 0.25;
};

struct SeparateOptions {
    // Stems to produce, by name; empty = every source. Each runs its own network.
    std::vector<std::string> stems;
    // Segments per network forward (batching); 0 = 2.
    int max_batch = 0;
    // Called every 2 ms while the device runs a batch (a host loop feeding
    // audio keeps going); changes nothing in the result.
    std::function<void()> wait;
    // (done, total) network forwards, after each batch.
    std::function<void(int, int)> on_progress;
    // Polled between batches; a cancelled separation returns no stems.
    CancelCheck cancel;
};

struct Separation {
    std::vector<std::string> names;     // in the order asked (or the sources' order)
    std::vector<StereoAudio> stems;     // at the input's rate and length, planar stereo
    bool cancelled = false;

    const StereoAudio* find(const std::string& name) const;
};

class HtDemucs {
public:
    HtDemucs();
    ~HtDemucs();
    HtDemucs(HtDemucs&&) noexcept;
    HtDemucs& operator=(HtDemucs&&) noexcept;
    HtDemucs(const HtDemucs&) = delete;
    HtDemucs& operator=(const HtDemucs&) = delete;

    // `dir` holds config.json + the per-stem safetensors. `stems` limits the
    // networks loaded (empty = all four); separate() can then only ask for
    // those. Throws std::runtime_error on a missing / malformed model.
    void load(const std::string& dir,
              brotensor::Device device = brotensor::Device::CUDA,
              const std::vector<std::string>& stems = {});

    // Planar audio at audio.sample_rate (any rate; mono is duplicated to both
    // channels) -> the requested stems at that rate and length. Their sum is
    // close to, not exactly, the mix.
    Separation separate(const StereoAudio& audio, const SeparateOptions& opts = {}) const;

    const HtDemucsConfig& config() const;
    // The stems this instance can produce (the networks loaded).
    std::vector<std::string> loaded_stems() const;
    bool loaded() const;
    brotensor::Device device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
