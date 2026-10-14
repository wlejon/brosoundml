// Beat phase sensors (beat.h): yue2-research native/endless.js's
// spectralOnsets / audioOnsets / fitBeat, in the same order and precision.

#include "brosoundml/beat.h"

#include <algorithm>
#include <cmath>

namespace brosoundml {

namespace {

constexpr double kPi = 3.141592653589793;   // Math.PI
constexpr int kN = 2048, kBands = 48, kHop = 240;

struct Tables {
    double cos_[kN / 2], sin_[kN / 2], hann[kN];
    int band_of[kN / 2 + 1];
    Tables() {
        for (int i = 0; i < kN / 2; ++i) {
            cos_[i] = std::cos(2 * kPi * i / kN);
            sin_[i] = -std::sin(2 * kPi * i / kN);
        }
        for (int i = 0; i < kN; ++i) hann[i] = 0.5 - 0.5 * std::cos(2 * kPi * i / (kN - 1));
        for (int b = 0; b <= kN / 2; ++b) {
            band_of[b] = -1;
            const double f = static_cast<double>(b) * 24000 / kN;
            if (f < 60 || f >= 11000) continue;
            band_of[b] = std::min(kBands - 1, static_cast<int>(std::floor(kBands * std::log(f / 60) / std::log(11000.0 / 60))));
        }
    }
};

const Tables& tables() {
    static const Tables t;
    return t;
}

// In-place radix-2 FFT, endless.js's loop order.
void fft(double* re, double* im) {
    const Tables& T = tables();
    const int n = kN;
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const int step = n / len, half = len >> 1;
        for (int i = 0; i < n; i += len)
            for (int k = 0; k < half; ++k) {
                const double c = T.cos_[k * step], s = T.sin_[k * step];
                const int a = i + k, b = a + half;
                // Kept as separate products and sums (no fused multiply-add),
                // as JavaScript evaluates them.
                const volatile double rc = re[b] * c, is = im[b] * s, rs = re[b] * s, ic = im[b] * c;
                const double xr = rc - is, xi = rs + ic;
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
    }
}

std::vector<float> spectral(const float* mono, std::size_t count) {
    const Tables& T = tables();
    std::vector<double> m(count >> 1);
    for (std::size_t i = 0; i < m.size(); ++i) m[i] = 0.5 * (static_cast<double>(mono[2 * i]) + mono[2 * i + 1]);
    const long long avail = static_cast<long long>(m.size()) - kN;
    const int n = avail > 0 ? static_cast<int>(avail / kHop) : 0, lead = kN / 2 / kHop;
    std::vector<float> out(static_cast<std::size_t>(n + lead), 0.0f);
    std::vector<double> re(kN), im(kN), band(kBands), prev(kBands);
    bool have_prev = false;
    for (int f = 0; f < n; ++f) {
        for (int i = 0; i < kN; ++i) {
            re[i] = m[static_cast<std::size_t>(f) * kHop + i] * T.hann[i];
            im[i] = 0;
        }
        fft(re.data(), im.data());
        std::fill(band.begin(), band.end(), 0.0);
        for (int b = 0; b <= kN / 2; ++b)
            if (T.band_of[b] >= 0) {
                const volatile double p = re[b] * re[b], q = im[b] * im[b];
                band[T.band_of[b]] += p + q;
            }
        for (int j = 0; j < kBands; ++j) band[j] = std::log(1e-10 + band[j]);
        double s = 0;
        if (have_prev)
            for (int j = 0; j < kBands; ++j) s += std::max(0.0, band[j] - prev[j]);
        out[static_cast<std::size_t>(lead + f)] = static_cast<float>(s / kBands);
        prev = band;
        have_prev = true;
    }
    return out;
}

std::vector<float> energy(const float* mono, std::size_t count) {
    const int hop = 480;
    const std::size_t n = count / hop;
    std::vector<float> out(n, 0.0f);
    double p1 = 0, p2 = 0, prev = 0;
    for (std::size_t f = 0; f < n; ++f) {
        double s1 = 0, s2 = 0;
        for (std::size_t i = f * hop; i < (f + 1) * hop; ++i) {
            const double v = mono[i];
            s1 += v * v;
            const double d = v - prev;
            s2 += d * d;
            prev = v;
        }
        const double e1 = std::log(1e-6 + s1 / hop), e2 = std::log(1e-6 + s2 / hop);
        out[f] = f ? static_cast<float>(std::max(0.0, e1 - p1) + std::max(0.0, e2 - p2)) : 0.0f;
        p1 = e1;
        p2 = e2;
    }
    return out;
}

}  // namespace

std::vector<float> beat_onsets(const float* mono48k, std::size_t samples, BeatSensor sensor) {
    if (!mono48k || samples == 0) return {};
    return sensor == BeatSensor::Spectral ? spectral(mono48k, samples) : energy(mono48k, samples);
}

std::vector<float> beat_mono(const float* planar, std::size_t frames, std::size_t offset, std::size_t count) {
    if (offset >= frames) return {};
    const std::size_t n = std::min(count, frames - offset);
    std::vector<float> m(n);
    for (std::size_t i = 0; i < n; ++i)
        m[i] = static_cast<float>(0.5 * (static_cast<double>(planar[offset + i]) + planar[frames + offset + i]));
    return m;
}

BeatFit fit_beat(const std::vector<float>& env, double origin, double P0, bool search) {
    BeatFit best;
    double best_v = -1;
    if (!(P0 > 0) || env.size() < 2) return best;
    const int steps = search ? 15 : 0;
    const double last = static_cast<double>(env.size()) - 1;
    for (int j = -steps; j <= steps; ++j) {
        const double P = P0 * (1 + j * 0.002);
        for (int q = 0; q < 64; ++q) {
            double x = std::fmod((q / 64.0) * P - origin, P);
            if (x < 0) x += P;
            double s = 0;
            int c = 0;
            for (; x < last; x += P) {
                const std::size_t i = static_cast<std::size_t>(std::floor(x));
                const double f = x - static_cast<double>(i);
                s += env[i] * (1 - f) + env[i + 1] * f;
                ++c;
            }
            if (c && s / c > best_v) {
                best_v = s / c;
                best.period = P;
                best.phase = q / 64.0;
                best.strength = best_v;
                best.beats = c;
            }
        }
    }
    double mean = 0;
    for (float v : env) mean += v;
    mean /= static_cast<double>(env.size());
    best.contrast = mean > 0 ? best.strength / mean : 0;
    return best;
}

BeatFit beat_phase(const float* mono48k, std::size_t samples, double period_seconds, BeatSensor sensor, bool search) {
    return fit_beat(beat_onsets(mono48k, samples, sensor), 0, period_seconds * 100.0, search);
}

}  // namespace brosoundml
