// brosoundml::GestureSpotter tests — non-speech gesture matching on synthetic
// tier-0 sensor streams. A click train enrolled as a rhythm must self-fire and
// must NOT fire on a different rhythm; a whistle enrolled as a tone must
// self-fire and must NOT fire at a different pitch; a lone click is too sparse
// to enroll.
#include "brosoundml/gesture_spotter.h"
#include "brosoundml/sensor_hub.h"

#include <brotensor/runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using brosoundml::GestureConfig;
using brosoundml::GestureEvent;
using brosoundml::GestureKind;
using brosoundml::GestureSpotter;
using brosoundml::GestureView;
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
// A frequency sweep with continuous phase: every frame is locally sinusoidal
// (reads as tonal) but the pitch wanders f0 -> f1 across the clip — the shape of
// a cough / throat-clear that an only-checks-the-mean tone matcher fires on.
void append_chirp(std::vector<float>& s, double seconds, double f0, double f1,
                  float amp) {
    const std::size_t n = static_cast<std::size_t>(seconds * kRate);
    double phase = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double frac = n > 1 ? static_cast<double>(i) /
                                        static_cast<double>(n - 1) : 0.0;
        const double f = f0 + (f1 - f0) * frac;
        phase += 2.0 * kPi * f / kRate;
        s.push_back(amp * static_cast<float>(std::sin(phase)));
    }
}
void place_click(std::vector<float>& s, double at_seconds, float amp) {
    std::size_t i0 = static_cast<std::size_t>(at_seconds * kRate);
    std::uint32_t x = static_cast<std::uint32_t>(i0) | 1u;
    for (std::size_t i = 0; i < 16 && i0 + i < s.size(); ++i) {
        x = x * 1664525u + 1013904223u;
        const float u = static_cast<float>(x >> 8) / static_cast<float>(1u << 24);
        s[i0 + i] = amp * (2.0f * u - 1.0f);
    }
}

// Drive a clip through the matcher exactly as the live host would: one private
// SensorHub, frame by frame (prime one window, then one hop per frame),
// feeding each snapshot to the spotter. Returns all gestures fired.
std::vector<GestureEvent> run_clip(GestureSpotter& g, const SensorHubConfig& cfg,
                                   const std::vector<float>& clip) {
    SensorHub hub(cfg);
    std::vector<GestureEvent> fired;
    const int win = cfg.mel.win_length, hop = cfg.mel.hop_length;
    const int n = static_cast<int>(clip.size());
    int pos = 0;
    auto pump = [&](const SensorSnapshot& s) {
        auto ev = g.feed(s);
        for (auto& e : ev) fired.push_back(e);
    };
    if (n >= win) {
        hub.feed(clip.data(), win);
        pump(hub.snapshot());
        pos = win;
        while (pos + hop <= n) {
            hub.feed(clip.data() + pos, hop);
            pump(hub.snapshot());
            pos += hop;
        }
    }
    return fired;
}

// A 3-tap rhythm at the given inter-tap spacing (seconds), with lead/tail.
std::vector<float> rhythm_clip(double gap) {
    std::vector<float> s = silence(0.3 + 3 * gap + 0.3);
    place_click(s, 0.3, 0.8f);
    place_click(s, 0.3 + gap, 0.8f);
    place_click(s, 0.3 + 2 * gap, 0.8f);
    return s;
}
std::vector<float> tone_clip(double hz) {
    std::vector<float> s = silence(0.3);
    append_tone(s, 0.6, hz, 0.12f);
    auto tail = silence(0.3);
    s.insert(s.end(), tail.begin(), tail.end());
    return s;
}

// ─── 1. a rhythm gesture self-fires and rejects a different tempo ───────────
void test_rhythm() {
    GestureConfig cfg;
    GestureSpotter g(cfg);
    auto clip = rhythm_clip(0.25);
    const int b = g.enroll_from_audio("triple", clip.data(), (int)clip.size());
    CHECK(b == 3, "rhythm: enrolled 3 onsets");
    GestureView v;
    CHECK(g.inspect("triple", v) && v.kind == GestureKind::Rhythm,
          "rhythm: inspect reports a rhythm");
    CHECK(v.intervals.size() == 2, "rhythm: two inter-onset intervals");
    std::fprintf(stderr, "  [rhythm] intervals = %d, %d frames (%.0f ms each)\n",
                 v.intervals.empty() ? -1 : v.intervals[0],
                 v.intervals.size() < 2 ? -1 : v.intervals[1],
                 v.intervals.empty() ? 0.0 : v.intervals[0] * v.frame_ms);

    g.reset();
    auto self = run_clip(g, cfg.sensor, rhythm_clip(0.25));
    CHECK(self.size() >= 1 && self[0].name == "triple",
          "rhythm: self-fires on its own tempo");
    if (!self.empty())
        std::fprintf(stderr, "  [rhythm] self-fire conf=%.2f\n", self[0].confidence);

    g.reset();
    auto other = run_clip(g, cfg.sensor, rhythm_clip(0.5));   // 2x slower
    CHECK(other.empty(), "rhythm: does NOT fire on a different tempo");
}

// ─── 2. a tone gesture self-fires and rejects a different pitch ─────────────
void test_tone() {
    GestureConfig cfg;
    GestureSpotter g(cfg);
    auto clip = tone_clip(1200.0);
    const int b = g.enroll_from_audio("whistle", clip.data(), (int)clip.size());
    CHECK(b == 1, "tone: enrolled one tone beat");
    GestureView v;
    CHECK(g.inspect("whistle", v) && v.kind == GestureKind::Tone,
          "tone: inspect reports a tone");
    std::fprintf(stderr, "  [tone] enrolled %.0f Hz over %d frames\n",
                 v.tone_hz, v.tone_frames);
    CHECK(v.tone_hz > 1100.0f && v.tone_hz < 1300.0f,
          "tone: captured pitch near 1200 Hz");

    g.reset();
    auto self = run_clip(g, cfg.sensor, tone_clip(1200.0));
    CHECK(self.size() >= 1 && self[0].name == "whistle",
          "tone: self-fires at its own pitch");
    if (!self.empty())
        std::fprintf(stderr, "  [tone] self-fire conf=%.2f\n", self[0].confidence);

    g.reset();
    auto other = run_clip(g, cfg.sensor, tone_clip(700.0));   // far-off pitch
    CHECK(other.empty(), "tone: does NOT fire at a different pitch");
}

// ─── 2b. a tone gesture rejects a wandering pitch (cough / throat-clear) ─────
// The reported failure: a held whistle fires on a cough. A cough is a sustained
// tonal-ish run whose MEAN pitch can land in the enrolled band, but whose pitch
// sweeps the whole way through instead of holding. The stability gate must
// reject it while a clean re-whistle (test_tone above) still fires.
void test_tone_rejects_wander() {
    GestureConfig cfg;
    GestureSpotter g(cfg);
    auto clip = tone_clip(1200.0);
    g.enroll_from_audio("whistle", clip.data(), (int)clip.size());
    GestureView v;
    CHECK(g.inspect("whistle", v), "wander: whistle enrolled");
    std::fprintf(stderr, "  [wander] enrolled tone spread = %.3f\n", v.tone_spread);
    CHECK(v.tone_spread < 0.05f,
          "wander: a clean whistle enrolls as a steady (low-spread) pitch");

    g.reset();
    std::vector<float> sweep = silence(0.3);
    append_chirp(sweep, 0.5, 1000.0, 1500.0, 0.12f);  // mean ~1250: inside the
    auto tail = silence(0.3);                          // 12% band, but wide spread
    sweep.insert(sweep.end(), tail.begin(), tail.end());
    auto fired = run_clip(g, cfg.sensor, sweep);
    if (!fired.empty())
        std::fprintf(stderr, "  [wander] UNEXPECTED fire conf=%.2f\n",
                     fired[0].confidence);
    CHECK(fired.empty(),
          "wander: does NOT fire on a sweeping pitch (the cough failure mode)");
}

// ─── 3. a lone click is too sparse to enroll ────────────────────────────────
void test_too_sparse() {
    GestureConfig cfg;
    GestureSpotter g(cfg);
    std::vector<float> s = silence(0.8);
    place_click(s, 0.4, 0.8f);
    bool threw = false;
    try { g.enroll_from_audio("lonely", s.data(), (int)s.size()); }
    catch (const std::exception&) { threw = true; }
    CHECK(threw, "sparse: a single click is rejected as a gesture");
    CHECK(g.templates().empty(), "sparse: nothing enrolled");
}

// ─── 4. refractory: one fire per gesture, not one per matching frame ────────
void test_refractory() {
    GestureConfig cfg;
    GestureSpotter g(cfg);
    auto clip = tone_clip(1000.0);
    g.enroll_from_audio("hum", clip.data(), (int)clip.size());
    g.reset();
    // A LONG hold of the enrolled pitch: without refractory + once-per-run
    // gating this would fire on every qualifying frame.
    std::vector<float> longhold = silence(0.3);
    append_tone(longhold, 2.0, 1000.0, 0.12f);
    auto fired = run_clip(g, cfg.sensor, longhold);
    CHECK(fired.size() == 1, "refractory: a sustained tone fires exactly once");
}

}  // namespace

int main() {
    brotensor::init();
    test_rhythm();
    test_tone();
    test_tone_rejects_wander();
    test_too_sparse();
    test_refractory();
    if (failures == 0) std::printf("test_gesture_spotter: all tests passed\n");
    return failures == 0 ? 0 : 1;
}
