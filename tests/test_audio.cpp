// AudioBuffer arithmetic + 16-bit PCM WAV round trip.
#include "brosoundml/audio.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static constexpr double kPi = 3.14159265358979323846;

static int failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

int main() {
    using brosoundml::AudioBuffer;

    // A 0.25 s, 440 Hz sine at 24 kHz, amplitude 0.5.
    const int   sr  = 24000;
    const int   n   = sr / 4;
    const float amp = 0.5f;
    AudioBuffer buf;
    buf.sample_rate = sr;
    buf.samples.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        buf.samples[i] =
            static_cast<float>(amp * std::sin(2.0 * kPi * 440.0 * i / sr));
    }

    CHECK(buf.frame_count() == static_cast<size_t>(n), "frame_count");
    CHECK(!buf.empty(), "not empty");
    CHECK(std::fabs(buf.duration_seconds() - 0.25) < 1e-6, "duration 0.25s");
    CHECK(std::fabs(buf.peak() - amp) < 1e-3f, "peak ~= amplitude");
    // RMS of a full-cycle sine is amplitude / sqrt(2).
    CHECK(std::fabs(buf.rms() - amp / std::sqrt(2.0f)) < 1e-2f, "rms ~= amp/sqrt2");

    // normalize() lifts the peak to the requested target.
    AudioBuffer norm = buf;
    norm.normalize(0.9f);
    CHECK(std::fabs(norm.peak() - 0.9f) < 1e-3f, "normalize to 0.9");

    // WAV round trip — write, read back, compare. 16-bit quantisation gives a
    // worst-case error of one LSB (~1/32767), so allow a small tolerance.
    const char* path = "brosoundml_test_audio_tmp.wav";
    buf.write_wav(path);
    AudioBuffer rt = brosoundml::read_wav(path);
    std::remove(path);

    CHECK(rt.sample_rate == sr, "round-trip sample_rate");
    CHECK(rt.frame_count() == buf.frame_count(), "round-trip frame_count");
    float max_err = 0.0f;
    for (size_t i = 0; i < rt.samples.size(); ++i) {
        max_err = std::max(max_err, std::fabs(rt.samples[i] - buf.samples[i]));
    }
    CHECK(max_err < 1e-3f, "round-trip sample fidelity");

    if (failures == 0) {
        std::printf("test_audio: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_audio: %d check(s) failed\n", failures);
    return 1;
}
