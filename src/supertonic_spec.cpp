#include "supertonic_spec.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: SupertonicSpec: " + msg);
}

// HTK mel: m = 2595 * log10(1 + f/700). Same convention as mel.cpp.
double hz_to_mel_htk(double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); }
double mel_to_hz_htk(double m)  { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); }

// Triangular HTK mel filterbank, (n_mels, n_bins) row-major.
std::vector<float> build_mel_filterbank_htk(int n_mels, int n_fft, int sample_rate,
                                            double fmin, double fmax) {
    const int n_bins = n_fft / 2 + 1;
    std::vector<double> bin_hz(static_cast<std::size_t>(n_bins));
    for (int k = 0; k < n_bins; ++k)
        bin_hz[static_cast<std::size_t>(k)] =
            static_cast<double>(k) * sample_rate / static_cast<double>(n_fft);
    const double mel_lo = hz_to_mel_htk(fmin);
    const double mel_hi = hz_to_mel_htk(fmax);
    std::vector<double> edges_hz(static_cast<std::size_t>(n_mels) + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        const double m = mel_lo + (mel_hi - mel_lo) * i / (n_mels + 1);
        edges_hz[static_cast<std::size_t>(i)] = mel_to_hz_htk(m);
    }
    std::vector<float> fb(static_cast<std::size_t>(n_mels) * n_bins, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        const double lo = edges_hz[static_cast<std::size_t>(m)];
        const double ce = edges_hz[static_cast<std::size_t>(m) + 1];
        const double hi = edges_hz[static_cast<std::size_t>(m) + 2];
        for (int k = 0; k < n_bins; ++k) {
            const double f = bin_hz[static_cast<std::size_t>(k)];
            double w = 0.0;
            if (f >= lo && f <= ce && ce > lo)       w = (f - lo) / (ce - lo);
            else if (f > ce && f <= hi && hi > ce)   w = (hi - f) / (hi - ce);
            if (w < 0.0) w = 0.0;
            fb[static_cast<std::size_t>(m) * n_bins + k] = static_cast<float>(w);
        }
    }
    return fb;
}

// Periodic Hann: w[n] = 0.5*(1 - cos(2*pi*n/N)) — matches torch.hann_window.
std::vector<float> build_hann_window(int win_length) {
    std::vector<float> w(static_cast<std::size_t>(win_length));
    for (int n = 0; n < win_length; ++n) {
        const double phase = 2.0 * 3.14159265358979323846 *
                             static_cast<double>(n) / static_cast<double>(win_length);
        w[static_cast<std::size_t>(n)] = static_cast<float>(0.5 * (1.0 - std::cos(phase)));
    }
    return w;
}

}  // namespace

SupertonicSpec::SupertonicSpec(bt::Device device, int sample_rate, int n_fft,
                               int win_length, int hop_length, int n_mels,
                               float fmin, float fmax, float eps)
    : dev_(device), sr_(sample_rate), n_fft_(n_fft), win_(win_length),
      hop_(hop_length), n_mels_(n_mels), n_bins_(n_fft / 2 + 1), eps_(eps) {
    if (sr_ <= 0 || n_fft_ <= 0 || win_ <= 0 || hop_ <= 0 || n_mels_ <= 0)
        fail("all of sample_rate/n_fft/win/hop/n_mels must be positive");
    if (win_ > n_fft_) fail("win_length > n_fft");
    if (!(fmin >= 0.0f && fmax > fmin)) fail("require 0 <= fmin < fmax");
    if (fmax > static_cast<float>(sr_) / 2.0f + 1e-3f) fail("fmax exceeds Nyquist");
    bt::init();

    const std::vector<float> fb =
        build_mel_filterbank_htk(n_mels_, n_fft_, sr_, fmin, fmax);
    std::vector<float> fb_T(static_cast<std::size_t>(n_bins_) * n_mels_);
    for (int m = 0; m < n_mels_; ++m)
        for (int k = 0; k < n_bins_; ++k)
            fb_T[static_cast<std::size_t>(k) * n_mels_ + m] =
                fb[static_cast<std::size_t>(m) * n_bins_ + k];
    mel_filter_T_ = bt::Tensor::from_host_on(dev_, fb_T.data(), n_bins_, n_mels_);

    const std::vector<float> w = build_hann_window(win_);
    window_ = bt::Tensor::from_host_on(dev_, w.data(), 1, win_);
}

bt::Tensor SupertonicSpec::compute(const float* signal, int L) const {
    if (L < 1 || signal == nullptr) fail("signal must be non-empty");
    const int T = 1 + L / hop_;  // torch.stft(center=True) frame count

    // STFT (center=True): reflect-pad n_fft/2 each side, then frame. Magnitude
    // and mel both live on-device; concat + log compression happen on the host
    // (one round-trip — this transform is cached per clip, not in a hot loop).
    bt::Tensor sig = bt::Tensor::from_host_on(dev_, signal, 1, L);
    bt::Tensor spec;
    bt::stft(sig, window_, /*N=*/1, n_fft_, hop_, win_,
             /*center=*/true, /*normalized=*/false, spec);
    if (spec.rows < T)
        fail("stft returned " + std::to_string(spec.rows) +
             " frames; expected >= " + std::to_string(T));

    bt::Tensor mag;  // (frames, n_bins)
    bt::complex_abs(spec, mag);
    bt::Tensor mel;  // (frames, n_mels) = mag @ filterbank^T
    bt::matmul(mag, mel_filter_T_, mel);

    const std::vector<float> mag_h = mag.to_host_vector();  // (frames, n_bins)
    const std::vector<float> mel_h = mel.to_host_vector();  // (frames, n_mels)
    const int frames = mag.rows;
    const int idim = n_bins_ + n_mels_;

    // Assemble [idim, T] channel-major: rows 0..n_bins-1 = log linear magnitude,
    // rows n_bins..idim-1 = log mel. Only the first T frames are kept (the
    // center=True formula yields exactly T; any extra trailing frame is dropped).
    std::vector<float> out(static_cast<std::size_t>(idim) * T);
    const auto logc = [this](float v) { return std::log(std::max(v, eps_)); };
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < n_bins_; ++k)
            out[static_cast<std::size_t>(k) * T + t] =
                logc(mag_h[static_cast<std::size_t>(t) * n_bins_ + k]);
        for (int m = 0; m < n_mels_; ++m)
            out[static_cast<std::size_t>(n_bins_ + m) * T + t] =
                logc(mel_h[static_cast<std::size_t>(t) * n_mels_ + m]);
    }
    (void)frames;
    return bt::Tensor::from_host_on(dev_, out.data(), idim, T);
}

}  // namespace brosoundml
