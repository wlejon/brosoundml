#pragma once

// Slaney-style mel filterbank + periodic Hann window builders, shared by the
// Whisper log-mel front-end (whisper_modules.cpp) and the Qwen3-ASR encoder
// (qwen_asr_encoder.cpp) — both models use WhisperFeatureExtractor's exact
// recipe (librosa mel(htk=False, norm="slaney"), torch.hann_window).
//
// Internal to brosoundml (not in the public include/ surface). Host-side,
// closed-form table builders — no brotensor dependency.

#include <cmath>
#include <cstddef>
#include <vector>

namespace brosoundml {
namespace melslaney {

// Slaney mel: linear below 1000 Hz (slope 3/200), logarithmic above. Used by
// librosa's `mel(..., htk=False)` and by openai/whisper's reference filterbank.
inline double hz_to_mel(double hz) {
    constexpr double f_min       = 0.0;
    constexpr double f_sp        = 200.0 / 3.0;                 // ≈ 66.667
    constexpr double min_log_hz  = 1000.0;
    constexpr double min_log_mel = (min_log_hz - f_min) / f_sp;  // == 15
    constexpr double logstep     = 0.06875177742094911;          // log(6.4) / 27

    if (hz >= min_log_hz) {
        return min_log_mel + std::log(hz / min_log_hz) / logstep;
    }
    return (hz - f_min) / f_sp;
}

inline double mel_to_hz(double mel) {
    constexpr double f_min       = 0.0;
    constexpr double f_sp        = 200.0 / 3.0;
    constexpr double min_log_hz  = 1000.0;
    constexpr double min_log_mel = (min_log_hz - f_min) / f_sp;
    constexpr double logstep     = 0.06875177742094911;

    if (mel >= min_log_mel) {
        return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    }
    return f_min + f_sp * mel;
}

// Build the (n_mels, n_fft/2 + 1) Slaney-normalised mel filterbank used by
// librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=0,
// fmax=sample_rate/2, htk=False, norm="slaney"). Returns the buffer flat in
// row-major order.
inline std::vector<float> build_filterbank(int n_mels, int n_fft,
                                           int sample_rate) {
    const int    n_bins = n_fft / 2 + 1;
    const double f_max  = sample_rate / 2.0;
    const double f_min  = 0.0;

    // FFT-bin centre frequencies in Hz: 0, sr/n_fft, 2*sr/n_fft, ...
    std::vector<double> fft_freqs(static_cast<std::size_t>(n_bins));
    for (int k = 0; k < n_bins; ++k) {
        fft_freqs[static_cast<std::size_t>(k)] =
            static_cast<double>(k) * sample_rate / n_fft;
    }

    // n_mels + 2 mel-spaced points, converted back to Hz.
    const double mel_min = hz_to_mel(f_min);
    const double mel_max = hz_to_mel(f_max);
    std::vector<double> mel_hz(static_cast<std::size_t>(n_mels) + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        const double m = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        mel_hz[static_cast<std::size_t>(i)] = mel_to_hz(m);
    }

    std::vector<float> fb(static_cast<std::size_t>(n_mels) * n_bins, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        const double lo = mel_hz[static_cast<std::size_t>(m)];
        const double ce = mel_hz[static_cast<std::size_t>(m) + 1];
        const double hi = mel_hz[static_cast<std::size_t>(m) + 2];
        // Slaney area normalisation: enorm = 2 / (hi - lo).
        const double enorm = 2.0 / (hi - lo);
        for (int k = 0; k < n_bins; ++k) {
            const double f = fft_freqs[static_cast<std::size_t>(k)];
            double w;
            if (f <= lo || f >= hi) {
                w = 0.0;
            } else if (f <= ce) {
                w = (f - lo) / (ce - lo);
            } else {
                w = (hi - f) / (hi - ce);
            }
            fb[static_cast<std::size_t>(m) * n_bins + k] =
                static_cast<float>(w * enorm);
        }
    }
    return fb;
}

// Periodic Hann window of length n: w[i] = 0.5*(1 - cos(2*pi*i/n)) — matches
// torch.hann_window(n) (periodic=True), the analysis window whisper/audio.py
// and WhisperFeatureExtractor pass to the STFT.
inline std::vector<float> build_hann_window(int n) {
    std::vector<float> w(static_cast<std::size_t>(n));
    constexpr double k_two_pi = 6.283185307179586;
    for (int i = 0; i < n; ++i) {
        const double phase = k_two_pi * static_cast<double>(i) /
                             static_cast<double>(n);
        w[static_cast<std::size_t>(i)] =
            static_cast<float>(0.5 * (1.0 - std::cos(phase)));
    }
    return w;
}

}  // namespace melslaney
}  // namespace brosoundml
