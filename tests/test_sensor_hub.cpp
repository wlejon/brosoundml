// brosoundml::SensorHub tests — tier-0 sensors on synthetic audio.
//
// Each scenario is a deterministic 16 kHz signal with a known ground truth:
// silence must read as nothing, a tone must read as voiced + tonal at the
// right frequency, white noise must read as voiced but NOT tonal, a click
// train must produce exactly one onset per click, and chunked feeding must
// match one-shot feeding (the bus is a streaming front-end — chunk size is
// an accident of the mic driver, never a signal).
#include "brosoundml/sensor_hub.h"

#include <brotensor/runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using brosoundml::SensorHub;
using brosoundml::SensorHubConfig;
using brosoundml::SensorSnapshot;

static int failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

namespace {

constexpr int    kRate = 16000;
constexpr double kPi   = 3.14159265358979323846;

std::vector<float> silence(double seconds) {
    return std::vector<float>(static_cast<std::size_t>(seconds * kRate), 0.0f);
}

void append_tone(std::vector<float>& s, double seconds, double hz, float amp) {
    const std::size_t n0 = s.size();
    const std::size_t n  = static_cast<std::size_t>(seconds * kRate);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(n0 + i) / kRate;
        s.push_back(amp * static_cast<float>(std::sin(2.0 * kPi * hz * t)));
    }
}

// Deterministic LCG white noise in [-amp, amp].
void append_noise(std::vector<float>& s, double seconds, float amp,
                  std::uint32_t seed = 0x12345u) {
    std::uint32_t x = seed;
    const std::size_t n = static_cast<std::size_t>(seconds * kRate);
    for (std::size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        const float u = static_cast<float>(x >> 8) /
                        static_cast<float>(1u << 24);   // [0,1)
        s.push_back(amp * (2.0f * u - 1.0f));
    }
}

// 16-sample noise burst — a physical "click" wider than one sample.
void place_click(std::vector<float>& s, double at_seconds, float amp) {
    std::size_t i0 = static_cast<std::size_t>(at_seconds * kRate);
    std::uint32_t x = static_cast<std::uint32_t>(i0) | 1u;
    for (std::size_t i = 0; i < 16 && i0 + i < s.size(); ++i) {
        x = x * 1664525u + 1013904223u;
        const float u = static_cast<float>(x >> 8) / static_cast<float>(1u << 24);
        s[i0 + i] = amp * (2.0f * u - 1.0f);
    }
}

int expected_frames(std::size_t n_samples, const SensorHubConfig& cfg) {
    if (n_samples < static_cast<std::size_t>(cfg.mel.win_length)) return 0;
    return 1 + static_cast<int>((n_samples - cfg.mel.win_length) /
                                cfg.mel.hop_length);
}

// ─── 1. silence reads as nothing ──────────────────────────────────────────
void test_silence() {
    SensorHubConfig cfg;
    SensorHub hub(cfg);
    const std::vector<float> s = silence(2.0);
    hub.feed(s.data(), static_cast<int>(s.size()));
    const SensorSnapshot sn = hub.snapshot();
    CHECK(sn.frames == expected_frames(s.size(), cfg),
          "silence: frame count matches the framing formula");
    CHECK(!sn.voice && sn.voice_events == 0, "silence: no voice");
    CHECK(sn.onsets == 0, "silence: no onsets");
    CHECK(!sn.tonal && sn.tonal_events == 0, "silence: not tonal");
    CHECK(sn.db <= -119.0f, "silence: level at the dB floor");
}

// ─── 2. tone: voiced, tonal, correct dominant frequency ───────────────────
void test_tone() {
    SensorHubConfig cfg;
    SensorHub hub(cfg);
    std::vector<float> s = silence(0.5);
    append_tone(s, 1.5, 1000.0, 0.1f);
    hub.feed(s.data(), static_cast<int>(s.size()));
    const SensorSnapshot sn = hub.snapshot();
    std::fprintf(stderr,
        "  [tone] periodicity=%.3f dominant=%.0f Hz tonal_frames=%lld "
        "voice_events=%lld last_voice_frame=%lld db=%.1f\n",
        sn.periodicity, sn.dominant_hz,
        static_cast<long long>(sn.tonal_frames),
        static_cast<long long>(sn.voice_events),
        static_cast<long long>(sn.last_voice_frame), sn.db);
    CHECK(sn.voice, "tone: voice active during the tone");
    CHECK(sn.voice_events == 1, "tone: exactly one silence->voice transition");
    CHECK(sn.last_voice_frame >= 40 && sn.last_voice_frame <= 60,
          "tone: voice transition lands near the 0.5 s tone onset");
    CHECK(sn.tonal, "tone: tonal at the end of a sustained tone");
    CHECK(sn.tonal_frames > 50, "tone: a long consecutive tonal run");
    CHECK(sn.periodicity > 0.9f, "tone: near-perfect periodicity");
    CHECK(sn.dominant_hz > 940.0f && sn.dominant_hz < 1070.0f,
          "tone: dominant_hz within one lag step of 1 kHz");
    CHECK(sn.db > -25.0f && sn.db < -15.0f,
          "tone: level near -20 dBFS (amp 0.1 sine ~ -23 dB rms)");
}

// ─── 2b. low-pitch tone: reports the fundamental, not the fmax ceiling ─────
// Regression: a low/smooth pitch (a hum, a vowel like "ummm" ~130 Hz) has a
// high autocorrelation at the shortest lag (the main-lobe shoulder). The old
// "smallest lag within 5% of the peak" rule snapped to lag_min and reported the
// 4 kHz ceiling for any such sound. The fundamental must win.
void test_low_tone() {
    SensorHubConfig cfg;
    SensorHub hub(cfg);
    std::vector<float> s = silence(0.5);
    append_tone(s, 1.5, 130.0, 0.1f);
    hub.feed(s.data(), static_cast<int>(s.size()));
    const SensorSnapshot sn = hub.snapshot();
    std::fprintf(stderr, "  [low-tone] periodicity=%.3f dominant=%.0f Hz\n",
        sn.periodicity, sn.dominant_hz);
    CHECK(sn.tonal, "low-tone: a sustained 130 Hz tone is tonal");
    CHECK(sn.periodicity > 0.9f, "low-tone: near-perfect periodicity");
    CHECK(sn.dominant_hz > 120.0f && sn.dominant_hz < 140.0f,
          "low-tone: dominant_hz tracks the 130 Hz fundamental, not the fmax ceiling");
}

// ─── 3. white noise: voiced but NOT tonal ──────────────────────────────────
void test_noise() {
    SensorHubConfig cfg;
    SensorHub hub(cfg);
    std::vector<float> s = silence(0.5);
    append_noise(s, 1.5, 0.1f);
    hub.feed(s.data(), static_cast<int>(s.size()));
    const SensorSnapshot sn = hub.snapshot();
    std::fprintf(stderr, "  [noise] periodicity=%.3f tonal_events=%lld\n",
        sn.periodicity, static_cast<long long>(sn.tonal_events));
    CHECK(sn.voice, "noise: voice active during the noise");
    CHECK(!sn.tonal, "noise: white noise is not tonal");
    CHECK(sn.periodicity < 0.4f,
          "noise: periodicity well below the tonal threshold");
}

// ─── 4. click train: one onset per click ──────────────────────────────────
void test_clicks() {
    SensorHubConfig cfg;
    SensorHub hub(cfg);
    std::vector<float> s = silence(2.5);
    const double clicks[] = {0.4, 0.65, 0.9, 1.15, 1.4, 1.65, 1.9};
    for (const double at : clicks) place_click(s, at, 0.8f);
    hub.feed(s.data(), static_cast<int>(s.size()));
    const SensorSnapshot sn = hub.snapshot();
    std::fprintf(stderr, "  [clicks] onsets=%lld last_onset_frame=%lld flux_now=%.3f\n",
        static_cast<long long>(sn.onsets),
        static_cast<long long>(sn.last_onset_frame), sn.flux);
    CHECK(sn.onsets == 7, "clicks: exactly one onset per click");
    // Last click at 1.9 s: first frame whose window reaches it is
    // ~(1.9*16000 - 400)/160 + 1 = ~189.
    CHECK(sn.last_onset_frame >= 185 && sn.last_onset_frame <= 195,
          "clicks: last onset lands on the last click's frame");
}

// ─── 5. chunked feeding == one-shot feeding ────────────────────────────────
void test_chunk_invariance() {
    SensorHubConfig cfg;
    std::vector<float> s = silence(0.4);
    append_tone(s, 0.6, 2000.0, 0.15f);
    place_click(s, 0.2, 0.8f);
    append_noise(s, 0.5, 0.05f);

    SensorHub one(cfg);
    one.feed(s.data(), static_cast<int>(s.size()));
    const SensorSnapshot a = one.snapshot();

    SensorHub chunked(cfg);
    for (std::size_t i = 0; i < s.size(); i += 7) {
        const int n = static_cast<int>(std::min<std::size_t>(7, s.size() - i));
        chunked.feed(s.data() + i, n);
    }
    const SensorSnapshot b = chunked.snapshot();

    CHECK(a.frames == b.frames,        "chunking: same frame count");
    CHECK(a.onsets == b.onsets,        "chunking: same onset count");
    CHECK(a.voice_events == b.voice_events, "chunking: same voice events");
    CHECK(a.tonal_events == b.tonal_events, "chunking: same tonal events");
    CHECK(std::abs(a.rms - b.rms) < 1e-6f,  "chunking: same final rms");
    CHECK(std::abs(a.flux - b.flux) < 1e-4f, "chunking: same final flux");
    CHECK(std::abs(a.periodicity - b.periodicity) < 1e-3f,
          "chunking: same final periodicity");
}

// ─── 6. reset drops state, hub keeps working ───────────────────────────────
void test_reset() {
    SensorHubConfig cfg;
    SensorHub hub(cfg);
    std::vector<float> s = silence(0.3);
    append_tone(s, 0.5, 1000.0, 0.1f);
    hub.feed(s.data(), static_cast<int>(s.size()));
    CHECK(hub.snapshot().frames > 0, "reset: frames advanced before reset");
    hub.reset();
    const SensorSnapshot z = hub.snapshot();
    CHECK(z.frames == 0 && z.onsets == 0 && !z.voice && !z.tonal,
          "reset: snapshot cleared");
    hub.feed(s.data(), static_cast<int>(s.size()));
    const SensorSnapshot again = hub.snapshot();
    CHECK(again.frames == expected_frames(s.size(), cfg),
          "reset: feeding works again after reset");
    CHECK(again.tonal, "reset: sensors live again after reset");
}

}  // namespace

int main() {
    brotensor::init();
    test_silence();
    test_tone();
    test_low_tone();
    test_noise();
    test_clicks();
    test_chunk_invariance();
    test_reset();
    if (failures == 0) std::printf("test_sensor_hub: all tests passed\n");
    return failures == 0 ? 0 : 1;
}
