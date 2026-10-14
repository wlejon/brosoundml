// The beat sensors (brosoundml/beat.h) on a click track whose beat is known:
// 90 bpm (2/3 s) with the first click at 0.25 s (phase 0.375 of a beat),
// clicks of 12 ms decaying noise over a quiet two-tone pad, 15 s at 48 kHz.
//   - spectral and energy flux: the comb finds the phase (within 0.03 beats)
//     with a clear contrast, and the beat count is the clicks';
//   - a period given 2% off, searched: the period comes back within 0.5%;
//   - a slice of the track fitted with its origin on the full timeline gives
//     the full track's phase (envelopes cut from one timeline agree);
//   - beat_mono averages planar stereo and clips its window;
//   - silence and an empty signal: no beat (contrast <= 1), no crash.
// CPU only, no weights.

#include "brosoundml/beat.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond, ...)                                \
    do {                                                \
        if (!(cond)) {                                  \
            std::fprintf(stderr, "FAIL: " __VA_ARGS__); \
            std::fprintf(stderr, "\n");                 \
            ++failures;                                 \
        }                                               \
    } while (0)

constexpr int kSr = 48000;
constexpr double kPi = 3.14159265358979323846;
constexpr double kPeriod = 2.0 / 3.0;   // seconds (90 bpm)
constexpr double kFirst = 0.25;         // the first click (seconds)

std::vector<float> click_track(double seconds) {
    const std::size_t n = static_cast<std::size_t>(seconds * kSr);
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / kSr;
        x[i] = static_cast<float>(0.03 * std::sin(2 * kPi * 220 * t) + 0.02 * std::sin(2 * kPi * 330 * t));
    }
    std::uint32_t s = 12345;
    for (double t = kFirst; t < seconds; t += kPeriod) {
        const std::size_t i0 = static_cast<std::size_t>(t * kSr);
        for (std::size_t j = 0; j < static_cast<std::size_t>(0.012 * kSr) && i0 + j < n; ++j) {
            s = s * 1664525u + 1013904223u;
            const double noise = (static_cast<double>(s >> 8) / 16777216.0) * 2 - 1;
            x[i0 + j] += static_cast<float>(0.8 * noise * std::exp(-static_cast<double>(j) / (0.003 * kSr)));
        }
    }
    return x;
}

double phase_distance(double a, double b) {
    double d = std::fmod(std::fabs(a - b), 1.0);
    return std::min(d, 1.0 - d);
}

}  // namespace

int main() {
    using namespace brosoundml;
    const double seconds = 15;
    const std::vector<float> x = click_track(seconds);
    const double want = kFirst / kPeriod;   // 0.375
    const int clicks = static_cast<int>(std::floor((seconds - kFirst) / kPeriod)) + 1;

    for (BeatSensor sensor : {BeatSensor::Spectral, BeatSensor::Energy}) {
        const bool spectral = sensor == BeatSensor::Spectral;
        const char* name = spectral ? "spectral" : "energy";
        const std::vector<float> env = beat_onsets(x.data(), x.size(), sensor);
        // The spectral envelope ends a window early (its last frame is the
        // last whole 2048-sample window at 24 kHz).
        const int frames = static_cast<int>(seconds * 100);
        CHECK(static_cast<int>(env.size()) <= frames && static_cast<int>(env.size()) >= frames - (spectral ? 10 : 1),
              "%s: %zu envelope frames for %.0f s at 100 fps", name, env.size(), seconds);
        const BeatFit f = beat_phase(x.data(), x.size(), kPeriod, sensor);
        // beat.h: the spectral flux reads onsets ~40 ms early.
        const double expect = spectral ? want - 0.04 / kPeriod : want;
        std::printf("%s: phase %.4f (want %.4f), period %.3f frames, strength %.4f, contrast %.2f, %d beats\n", name,
                    f.phase, expect, f.period, f.strength, f.contrast, f.beats);
        CHECK(phase_distance(f.phase, expect) < 0.03, "%s: phase %.4f, want %.4f", name, f.phase, expect);
        CHECK(std::fabs(f.period - kPeriod * 100) < 1e-9, "%s: an unsearched period is the one given (%.4f)", name, f.period);
        CHECK(f.contrast > 2, "%s: a click track has a clear beat (contrast %.2f)", name, f.contrast);
        CHECK(std::abs(f.beats - clicks) <= 1, "%s: %d beats, %d clicks", name, f.beats, clicks);

        // A period 2% off, searched.
        const BeatFit s = beat_phase(x.data(), x.size(), kPeriod * 1.02, sensor, true);
        std::printf("%s searched from %.3f: period %.3f, phase %.4f\n", name, kPeriod * 102, s.period, s.phase);
        CHECK(std::fabs(s.period / (kPeriod * 100) - 1) < 0.005, "%s: searched period %.3f, want %.3f", name, s.period,
              kPeriod * 100);

        // A slice from 3.1 s, its origin on the full timeline.
        const std::size_t off = static_cast<std::size_t>(3.1 * kSr);
        const std::vector<float> envs = beat_onsets(x.data() + off, x.size() - off, sensor);
        const BeatFit sl = fit_beat(envs, 310, kPeriod * 100);
        CHECK(phase_distance(sl.phase, f.phase) < 0.03, "%s: the slice's phase %.4f, the track's %.4f", name, sl.phase,
              f.phase);
    }

    // beat_mono: 0.5 (L + R), clipped to the signal.
    {
        const std::vector<float> planar = {1, 2, 3, 4, 10, 20, 30, 40};   // L = 1..4, R = 10..40
        const std::vector<float> m = beat_mono(planar.data(), 4, 1, 10);
        CHECK(m.size() == 3 && m[0] == 11 && m[1] == 16.5f && m[2] == 22, "beat_mono: [%g %g %g] (%zu)",
              m.size() > 0 ? m[0] : -1.f, m.size() > 1 ? m[1] : -1.f, m.size() > 2 ? m[2] : -1.f, m.size());
    }

    // Silence and nothing.
    {
        const std::vector<float> z(static_cast<std::size_t>(5 * kSr), 0.0f);
        for (BeatSensor sensor : {BeatSensor::Spectral, BeatSensor::Energy}) {
            const BeatFit f = beat_phase(z.data(), z.size(), kPeriod, sensor);
            CHECK(f.contrast <= 1.0 + 1e-9, "silence has no beat (contrast %.3f)", f.contrast);
            const BeatFit e = beat_phase(nullptr, 0, kPeriod, sensor);
            CHECK(e.beats == 0 && e.contrast == 0, "an empty signal: %d beats, contrast %.3f", e.beats, e.contrast);
        }
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
