#pragma once

#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brosoundml {

// ─── MelFrontend ───────────────────────────────────────────────────────────
//
// Streaming log-mel feature extractor. Built once, then either:
//   • drip-fed via consume() one mic chunk at a time, emitting new (n_mels, T)
//     frames once enough samples have accumulated, or
//   • one-shot via compute_offline() over an entire AudioBuffer.
//
// Both paths produce bit-equivalent output for the same input — the streaming
// equivalence is locked by tests/test_mel.cpp. The class is single-producer:
// no two threads may call consume() / compute_offline() / reset() concurrently.
//
// Compute path (all brotensor ops):
//   1. stft           — frames the internal ring into interleaved-complex spec.
//   2. complex_abs    — magnitude spectrum.
//   3. matmul(mel_fb) — projects (n_bins,) → (n_mels,).
//   4. log_forward    — natural log with an eps clamp baked in upstream.
//
// Output layout: (n_mels, T) row-major FP32 — matches NCL with N=1, so it
// slots directly into Conv1d-based stacks. Values are log(max(mel, eps)) with
// eps = 1e-10f. There is NO center-padding: the first frame is emitted only
// after win_length samples have arrived. Framing matches the offline formula
//      n_frames(N) = 1 + (N - win_length) / hop_length   (floor; N >= win_length)
// exactly, on both paths.

enum class MelWindow : std::uint8_t {
    Hann = 0,  // periodic Hann (matches torchaudio default)
};

enum class MelFormula : std::uint8_t {
    // HTK mel: m = 2595 * log10(1 + f/700). Triangular filters over linearly-
    // spaced mel centres between fmin and fmax. No Slaney area-normalisation —
    // matches torchaudio MelSpectrogram(mel_scale="htk", norm=None), which is
    // what most KWS recipes train against.
    HTK = 0,
};

// Output compression applied to the linear mel energy.
//   Log  — log(max(mel, eps)). The classic KWS front-end. Absolute: a static
//          per-channel gain (microphone frequency response, spectral tilt)
//          passes straight through as an additive offset the model must learn.
//   PCEN — per-channel energy normalization (Wang et al. 2017, "Trainable
//          Frontend For Robust and Far-Field Keyword Spotting"). Divides each
//          channel by a causal IIR-smoothed energy estimate, then root-
//          compresses. The division cancels any slowly-varying per-channel
//          gain — i.e. it normalises out microphone/channel spectral tilt and
//          loudness before the model sees it. Streaming-friendly: the smoother
//          carries one float of state per mel bin.
//
//      PCEN(t,f) = ( E(t,f) / (eps + M(t,f))^alpha + delta )^r - delta^r
//      M(t,f)    = (1 - s)*M(t-1,f) + s*E(t,f),   M(0,f) = E(0,f)
enum class MelCompression : std::uint8_t {
    Log  = 0,
    PCEN = 1,
};

struct MelConfig {
    int        sample_rate = 16000;
    int        n_fft       = 512;
    int        win_length  = 400;     // 25 ms @ 16 kHz
    int        hop_length  = 160;     // 10 ms @ 16 kHz
    int        n_mels      = 40;
    float      fmin        = 20.0f;
    float      fmax        = 8000.0f; // Nyquist @ 16 kHz
    MelWindow  window      = MelWindow::Hann;
    MelFormula formula     = MelFormula::HTK;

    // Output compression. Defaults to Log for backward compatibility; the
    // wake recipe selects PCEN for channel/tilt robustness.
    MelCompression compression = MelCompression::Log;

    // PCEN parameters (used only when compression == PCEN). Defaults match
    // librosa.pcen at a 10 ms hop: smoother time-constant ~0.4 s (s≈0.025),
    // alpha 0.98, delta 2.0, r 0.5.
    float pcen_s     = 0.025f;   // IIR smoother coefficient
    float pcen_alpha = 0.98f;    // gain-normalisation exponent
    float pcen_delta = 2.0f;     // bias added before root compression
    float pcen_r     = 0.5f;     // root-compression exponent
    float pcen_eps   = 1e-6f;    // denominator floor
};

class MelFrontend {
public:
    explicit MelFrontend(const MelConfig& cfg,
                         brotensor::Device device = brotensor::Device::CPU);

    // Drop the streaming ring buffer. The mel filter matrix and the analysis
    // window are unchanged — only sample-side state is cleared.
    void reset();

    // Push `n` samples into the ring buffer, emit zero or more new mel frames.
    // Frames are appended along the time axis of `out_frames_appended`
    // (resize as needed if the call adds frames). Returns the number of new
    // frames emitted on this call. The function is no-op-cheap when fewer than
    // hop_length new samples have arrived since the last frame boundary.
    int consume(const float* samples, int n,
                brotensor::Tensor& out_frames_appended);

    // One-shot equivalent of reset() + consume(samples, n, out). `out` is
    // overwritten with shape (n_mels, T) where T = 1 + (n - win_length) / hop
    // (floor; or zero if n < win_length).
    void compute_offline(const float* samples, int n,
                         brotensor::Tensor& out);

    // Inspectors.
    const MelConfig&         config()          const { return cfg_; }
    const brotensor::Tensor& filter()          const { return mel_filter_; }
    const brotensor::Tensor& window()          const { return window_;    }
    brotensor::Device        device()          const { return device_;    }
    int                      frames_buffered() const;

private:
    MelConfig         cfg_;
    brotensor::Device device_;

    // (n_mels, n_fft/2 + 1) mel filterbank, FP32, resident on `device_`.
    brotensor::Tensor mel_filter_;
    // (n_fft/2 + 1, n_mels) transpose of mel_filter_, resident on `device_`.
    // The projection matmul wants this orientation; cached at construction so
    // the per-call path never round-trips the filterbank through the host.
    brotensor::Tensor mel_filter_T_;
    // (1, win_length) analysis window, FP32, resident on `device_`.
    brotensor::Tensor window_;

    // Streaming ring buffer (host side). Holds at most win_length - 1 leftover
    // samples plus the current batch under consideration; samples older than
    // the next frame's start are dropped after every emission.
    std::vector<float> ring_;
    // Number of samples discarded from the front of the conceptual stream
    // since the last reset(). Used only for diagnostics / frames_buffered().
    std::int64_t       samples_dropped_ = 0;

    // PCEN per-channel IIR smoother state (size n_mels when active). Carried
    // across consume() calls so streaming PCEN matches the offline pass
    // frame-for-frame. `pcen_init_` is false until the first frame after a
    // reset() seeds M with that frame's energy (avoids a startup transient).
    // Unused when cfg_.compression == Log.
    std::vector<float> pcen_m_;
    bool               pcen_init_ = false;

    // Apply cfg_.compression to a linear-mel host buffer laid out (T, n_mels)
    // row-major and write the result transposed to (n_mels, T) into `out` on
    // device_. For PCEN this advances pcen_m_ frame-by-frame (carrying state),
    // so callers must invoke it in stream order.
    void compress_and_emit(const std::vector<float>& linear_mel_TM, int T,
                           brotensor::Tensor& out);
};

}  // namespace brosoundml
