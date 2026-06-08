#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── RAVE — Realtime Audio Variational autoEncoder (ACIDS / IRCAM) ───────────
//
// RAVE is a small (<20M-param), faster-than-realtime neural audio autoencoder.
// A multiband (PQMF) variational convolutional encoder compresses a waveform to
// a low-rate latent z of shape (n_latent x frames); a residual upsampling
// decoder synthesises a waveform back from z. The latent axes are PCA-sorted by
// variance — dim 0 tracks loudness, dim 1 pitch/centroid, the rest timbre — so
// editing the per-dimension time series (the lab use case) morphs the audio in
// musically meaningful ways.
//
// brosoundml runs INFERENCE of an exported RAVE v2 model. The exported `.ts`
// (the streaming `cached_conv` build used by the nn~/VST) is converted to
// safetensors + config.json offline by scripts/convert-rave.py; this loader
// reads that layout. The forward pass is a single fresh-cache (offline causal)
// pass — every op composes brotensor's existing surface, so a CUDA / Metal
// build runs on the GPU with no model-specific kernels.
//
// encode() is deterministic: it uses the posterior mean (no reparameterisation
// sampling). decode() runs the deterministic waveform + loudness synthesis
// branches by default, so a clip round-trips reproducibly — the right default
// for editing. Passing RaveDecodeOptions{add_noise=true} additionally runs the
// model's stochastic FFT filtered-noise synthesizer (breathy / unvoiced /
// textural energy); pin it with a `seed` or an injected `noise` buffer when you
// need reproducibility.

// Model hyperparameters, read from the converted config.json by Rave::load.
struct RaveConfig {
    int   sampling_rate       = 48000;   // model I/O rate (Hz)
    int   full_latent_size    = 128;     // encoder distribution width (pre-PCA crop)
    int   cropped_latent_size = 8;       // kept latent dims after PCA (= n_latent)
    int   n_band              = 16;      // PQMF band count
    float leaky_slope         = 0.2f;    // LeakyReLU negative slope
    float bn_eps              = 1.0e-5f; // encoder BatchNorm epsilon

    // Total time compression of the encoder: a waveform of L samples encodes to
    // L / total_ratio frames. = n_band * product(encoder strides) (2048 for the
    // b2048 ratio). Computed at load from the conv topology.
    int   total_ratio         = 0;
};

// An encoded latent: a (n_latent x frames) row-major, channel-major float grid
// (data[c * frames + t] is dim c at frame t). This is the surface the lab plots
// and edits as n_latent time-series curves before decode().
struct RaveLatent {
    std::vector<float> data;       // n_latent * frames, channel-major
    int                n_latent = 0;
    int                frames   = 0;

    float at(int dim, int frame) const {
        return data[static_cast<std::size_t>(dim) * frames + frame];
    }
    float& at(int dim, int frame) {
        return data[static_cast<std::size_t>(dim) * frames + frame];
    }
};

// decode() options. The default (no noise) is deterministic and reproducible —
// the right choice for latent editing. add_noise enables RAVE's stochastic FFT
// filtered-noise synthesizer (the third synthesis branch). Its white noise is
// resampled each call, so the output then varies unless pinned: set `seed` for a
// reproducible internal draw, or supply `noise` to inject the white noise
// verbatim (used by the parity test; also handy for fully deterministic noise).
//
// Stereo (decode_multi): RAVE has no stereo decoder — the VST's "stereo" runs
// the same mono decoder once per channel and concatenates the outputs. The two
// channels decorrelate solely because pre_process_latent pads the discarded
// latent dims [cropped_latent_size, full_latent_size) with INDEPENDENT N(0,1)
// noise per channel. We reproduce that exactly: latent_pad_std is the std of
// that per-channel pad (RAVE uses 1.0; this is the "stereo width" knob — 0 keeps
// the deterministic zero-pad, identical to mono decode()). Each channel's pad is
// drawn from `seed` with the channel index as the RNG counter, so a stereo
// decode is reproducible. latent_pad injects the pad verbatim (parity test).
struct RaveDecodeOptions {
    bool         add_noise = false;   // run the stochastic noise-synth branch
    std::uint64_t seed     = 0;       // RNG seed for white noise + latent pad (when not injected)
    const float* noise     = nullptr; // optional injected white noise in U(-1, 1), row-major:
                                      // (frames/64 * n_band) rows of 64. Used verbatim when set.
    int          noise_len = 0;       // element count of `noise`; must equal frames * total_ratio

    int          channels       = 1;     // decode_multi: output channel count (>=1)
    float        latent_pad_std = 0.0f;  // std of the N(0,1) pad on dims [n_latent, full); 0 = zero-pad.
                                         // RAVE-native = 1.0; acts as the stereo-width control.
    const float* latent_pad     = nullptr; // optional injected pad, channel-concatenated:
                                           // channels * (full - n_latent) * frames floats, each block
                                           // (full - n_latent, frames) channel-major. Used verbatim.
    int          latent_pad_len = 0;       // element count of `latent_pad`
};

// Multi-channel decode result (decode_multi). `samples` is INTERLEAVED:
// samples[t*channels + c] is channel c at frame t, so it drops straight into an
// interleaved sink (broaudio createClip). channels == 1 is a plain mono buffer.
struct RaveMultiBuffer {
    std::vector<float> samples;            // interleaved, FP32, nominally [-1, 1]
    int                sample_rate = 48000;
    int                channels    = 1;

    size_t frame_count() const { return channels ? samples.size() / channels : 0; }
    bool   empty()       const { return samples.empty(); }
};

// The RAVE autoencoder. Construct, load() a converted model directory, then
// encode() / decode(). Heavy state (weights, config, module graph) lives behind
// a pImpl so this header stays free of brotensor module internals.
class Rave {
public:
    Rave();
    ~Rave();
    Rave(Rave&&) noexcept;
    Rave& operator=(Rave&&) noexcept;
    Rave(const Rave&) = delete;
    Rave& operator=(const Rave&) = delete;

    // Load config.json + model.safetensors from `model_dir` (the output of
    // scripts/convert-rave.py), placing the weights on `device`. Throws
    // std::runtime_error on a missing / malformed model.
    void load(const std::string& model_dir,
              brotensor::Device device = brotensor::Device::CPU);

    // Waveform (mono, sampling_rate Hz) -> latent (cropped_latent_size x frames).
    // The input is right-padded to a whole frame (a multiple of total_ratio), so
    // frames == ceil(n / total_ratio). Deterministic (posterior mean).
    RaveLatent encode(const std::vector<float>& audio) const;
    RaveLatent encode(const float* audio, int n) const;

    // Latent -> waveform (mono, sampling_rate Hz). Produces frames * total_ratio
    // samples. Runs the deterministic waveform + loudness branches; pass
    // RaveDecodeOptions{add_noise=true} to add the stochastic noise-synth branch.
    AudioBuffer decode(const RaveLatent& latent,
                       const RaveDecodeOptions& opts = {}) const;
    AudioBuffer decode(const float* latent, int n_latent, int frames,
                       const RaveDecodeOptions& opts = {}) const;

    // Latent -> multi-channel waveform (interleaved). With opts.channels == 1 and
    // a zero latent pad this equals decode(); with channels >= 2 it runs the mono
    // decoder once per channel, each with an independent N(0,1) latent pad
    // (opts.latent_pad_std / latent_pad), reproducing RAVE's stereo-mode decode.
    // Each channel is frames * total_ratio samples long.
    RaveMultiBuffer decode_multi(const RaveLatent& latent,
                                 const RaveDecodeOptions& opts = {}) const;
    RaveMultiBuffer decode_multi(const float* latent, int n_latent, int frames,
                                 const RaveDecodeOptions& opts = {}) const;

    const RaveConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
