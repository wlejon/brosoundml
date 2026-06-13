#include "brosoundml/sensor_hub.h"

#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace brosoundml {

namespace {

constexpr float kDbFloor = -120.0f;

}  // namespace

SensorHub::SensorHub(const SensorHubConfig& cfg)
    : cfg_(cfg), mel_(cfg_.mel, brotensor::Device::CPU) {
    prev_frame_.assign(static_cast<std::size_t>(cfg_.mel.n_mels), 0.0f);
    publish();
}

int SensorHub::feed(const float* samples, int n) {
    if (!samples || n <= 0) return 0;
    const int win = cfg_.mel.win_length;
    const int hop = cfg_.mel.hop_length;
    const int M   = cfg_.mel.n_mels;

    // Mirror the front-end's stream: ring_ always starts exactly at the next
    // unemitted frame's window start, so frame f of this call covers
    // ring_[f*hop .. f*hop + win).
    ring_.insert(ring_.end(), samples, samples + n);

    brotensor::Tensor frames;   // (n_mels, F)
    const int F = mel_.consume(samples, n, frames);
    if (F <= 0) return 0;

    const std::vector<float> host = frames.to_host_vector();
    std::vector<float> mel_col(static_cast<std::size_t>(M));
    std::size_t off = 0;
    int done = 0;
    for (int f = 0; f < F; ++f) {
        if (off + static_cast<std::size_t>(win) > ring_.size()) break;
        for (int m = 0; m < M; ++m) {
            mel_col[static_cast<std::size_t>(m)] =
                host[static_cast<std::size_t>(m) * static_cast<std::size_t>(F) +
                     static_cast<std::size_t>(f)];
        }
        process_frame(ring_.data() + off, mel_col.data());
        publish();
        off += static_cast<std::size_t>(hop);
        ++done;
    }
    ring_.erase(ring_.begin(),
                ring_.begin() + static_cast<std::ptrdiff_t>(off));
    return done;
}

void SensorHub::feed_frame(const float* window, const float* mel_frame) {
    if (!window || !mel_frame) return;
    process_frame(window, mel_frame);
    publish();
}

void SensorHub::process_frame(const float* window, const float* mel_frame) {
    const int win = cfg_.mel.win_length;
    const int hop = cfg_.mel.hop_length;
    const int M   = cfg_.mel.n_mels;

    cur_.frames += 1;
    cur_.t = static_cast<double>((cur_.frames - 1) * hop + win) /
             static_cast<double>(cfg_.mel.sample_rate);

    // ── level ──
    double energy = 0.0;
    float  peak   = 0.0f;
    for (int i = 0; i < win; ++i) {
        const float v = window[i];
        energy += static_cast<double>(v) * static_cast<double>(v);
        const float a = std::abs(v);
        if (a > peak) peak = a;
    }
    const float rms = static_cast<float>(std::sqrt(energy / win));
    const float db  = rms > 1e-6f
        ? std::max(kDbFloor, 20.0f * std::log10(rms))
        : kDbFloor;
    cur_.rms  = rms;
    cur_.peak = peak;
    cur_.db   = db;

    // ── voice (adaptive-floor energy VAD) ──
    if (!floor_init_) {
        cur_.noise_floor_db = db;
        floor_init_ = true;
    } else if (db < cur_.noise_floor_db) {
        cur_.noise_floor_db = db;   // instant attack down
    } else {
        const float rise = cfg_.vad_floor_rise_dbps *
            static_cast<float>(hop) / static_cast<float>(cfg_.mel.sample_rate);
        cur_.noise_floor_db = std::min(cur_.noise_floor_db + rise, 0.0f);
    }
    cur_.snr_db = db - cur_.noise_floor_db;
    const bool hot = db > cfg_.vad_abs_floor_db &&
                     cur_.snr_db > cfg_.vad_snr_db;
    if (hot) {
        vad_hang_left_ = cfg_.vad_hang_frames;
    } else if (vad_hang_left_ > 0) {
        --vad_hang_left_;
    }
    const bool voice = hot || vad_hang_left_ > 0;
    if (voice && !cur_.voice) {
        cur_.voice_events += 1;
        cur_.last_voice_frame = cur_.frames - 1;
    }
    cur_.voice_frames = voice ? cur_.voice_frames + 1 : 0;
    cur_.voice = voice;

    // ── onset (positive PCEN spectral flux vs a slow EMA baseline) ──
    float flux = 0.0f;
    if (have_prev_) {
        for (int m = 0; m < M; ++m) {
            flux += std::max(0.0f, mel_frame[m] - prev_frame_[static_cast<std::size_t>(m)]);
        }
        flux /= static_cast<float>(M);
    }
    std::copy(mel_frame, mel_frame + M, prev_frame_.begin());
    have_prev_ = true;
    cur_.flux  = flux;
    bool onset = false;
    if (onset_cooldown_ > 0) {
        --onset_cooldown_;
    } else if (flux > cfg_.onset_abs && flux > flux_ema_ * cfg_.onset_ratio) {
        onset = true;
        onset_cooldown_ = cfg_.onset_refractory_frames;
        cur_.onsets += 1;
        cur_.last_onset_frame = cur_.frames - 1;
    }
    cur_.onset = onset;
    flux_ema_ = (1.0f - cfg_.onset_ema) * flux_ema_ + cfg_.onset_ema * flux;

    // ── tonality (normalized autocorrelation on the raw window) ──
    // Periodicity, not spectral peakiness: PCEN flattens static spectral
    // shape by design, and a raw-window autocorrelation peak catches harmonic
    // sounds (hums) that a single-bin peak measure would miss anyway.
    const int rate    = cfg_.mel.sample_rate;
    const int lag_min = std::max(2,
        static_cast<int>(static_cast<float>(rate) / cfg_.tonal_fmax_hz));
    const int lag_max = std::min(win - 64,
        static_cast<int>(static_cast<float>(rate) / cfg_.tonal_fmin_hz));
    float periodicity = 0.0f;
    float dominant_hz = 0.0f;
    if (lag_max > lag_min && energy > 1e-9) {
        nac_.assign(static_cast<std::size_t>(lag_max + 1), 0.0f);
        for (int lag = lag_min; lag <= lag_max; ++lag) {
            const int n = win - lag;
            double num = 0.0, e0 = 0.0, e1 = 0.0;
            for (int i = 0; i < n; ++i) {
                const double a = window[i];
                const double b = window[i + lag];
                num += a * b;
                e0  += a * a;
                e1  += b * b;
            }
            const double den = std::sqrt(e0 * e1);
            nac_[static_cast<std::size_t>(lag)] = den > 1e-12
                ? static_cast<float>(num / den)
                : 0.0f;
        }
        // The normalized autocorrelation always starts high near lag 0 — adjacent
        // samples of any band-limited signal are correlated — and decays. That
        // main-lobe shoulder is NOT periodicity: scanning from lag_min would snap
        // any low-pitched/smooth sound (a hum, a vowel) to lag_min and report the
        // ceiling frequency (fmax). A real period instead appears as a local
        // maximum after the shoulder, so search peaks only — take the strongest
        // local-max correlation as the periodicity, and report the shortest-period
        // peak within 5% of it (prefer the fundamental over its octaves).
        const auto is_peak = [&](int lag) {
            const float v = nac_[static_cast<std::size_t>(lag)];
            return v > nac_[static_cast<std::size_t>(lag - 1)] &&
                   v >= nac_[static_cast<std::size_t>(lag + 1)];
        };
        float best = 0.0f;
        for (int lag = lag_min + 1; lag < lag_max; ++lag)
            if (is_peak(lag) && nac_[static_cast<std::size_t>(lag)] > best)
                best = nac_[static_cast<std::size_t>(lag)];
        int best_lag = 0;
        for (int lag = lag_min + 1; lag < lag_max; ++lag) {
            if (is_peak(lag) && nac_[static_cast<std::size_t>(lag)] >= 0.95f * best) {
                best_lag = lag;
                break;
            }
        }
        periodicity = best;
        if (best_lag > 0) {
            // Integer-lag pitch is coarse: at a 16 kHz rate adjacent lags are
            // ~18% apart near 3 kHz, so a perfectly steady high whistle would
            // jump between two quantized frequencies frame to frame. Refine the
            // peak with a parabola through (lag-1, lag, lag+1) of the
            // normalized autocorrelation — the vertex is the true period to
            // sub-sample precision. This sharpens the HUD pitch readout and,
            // crucially, lets a pitch-stability measure tell a held whistle
            // from a wandering cough instead of from quantization noise.
            const double y0 = nac_[static_cast<std::size_t>(best_lag - 1)];
            const double y1 = nac_[static_cast<std::size_t>(best_lag)];
            const double y2 = nac_[static_cast<std::size_t>(best_lag + 1)];
            const double denom = y0 - 2.0 * y1 + y2;
            double refined = best_lag;
            if (std::abs(denom) > 1e-12) {
                const double delta = 0.5 * (y0 - y2) / denom;   // in (-1, 1)
                if (delta > -1.0 && delta < 1.0) refined = best_lag + delta;
            }
            dominant_hz = static_cast<float>(rate) / static_cast<float>(refined);
        }
    }
    cur_.periodicity = periodicity;
    cur_.dominant_hz = dominant_hz;
    const bool tonal = periodicity > cfg_.tonal_min_periodicity &&
                       db > cfg_.vad_abs_floor_db;
    if (tonal && !cur_.tonal) {
        cur_.tonal_events += 1;
        cur_.last_tonal_frame = cur_.frames - 1;
    }
    cur_.tonal_frames = tonal ? cur_.tonal_frames + 1 : 0;
    cur_.tonal = tonal;
}

void SensorHub::publish() {
    const std::uint32_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1, std::memory_order_release);   // odd: write in flight
    pub_ = cur_;
    seq_.store(s + 2, std::memory_order_release);   // even: stable
}

SensorSnapshot SensorHub::snapshot() const {
    for (;;) {
        const std::uint32_t s1 = seq_.load(std::memory_order_acquire);
        if (s1 & 1u) continue;
        SensorSnapshot out = pub_;
        const std::uint32_t s2 = seq_.load(std::memory_order_acquire);
        if (s1 == s2) return out;
    }
}

void SensorHub::reset() {
    mel_.reset();
    ring_.clear();
    std::fill(prev_frame_.begin(), prev_frame_.end(), 0.0f);
    have_prev_      = false;
    floor_init_     = false;
    flux_ema_       = 0.0f;
    vad_hang_left_  = 0;
    onset_cooldown_ = 0;
    cur_ = SensorSnapshot{};
    publish();
}

}  // namespace brosoundml
