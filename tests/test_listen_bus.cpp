// brosoundml::ListenBus tests — the shared front-end must be a transparent
// substitute for each consumer's own front-end.
//
// The contract under test: a consumer driven THROUGH the bus sees exactly the
// stream it would have computed standalone. For the SensorHub that is checked
// bit-for-bit (same CPU mel, same windows → identical snapshots, chunked or
// not). For the PhonemeSpotter a tiny random PhonemeNet is built in-process
// (PhonemeNet::make + save, no weights on disk needed) and the bus-driven
// posterior stream is compared frame-for-frame against standalone feed().
// Plus the attach-time framing validation and the seam error paths.
#include "brosoundml/listen_bus.h"

#include "brosoundml/bc_resnet2d.h"
#include "brosoundml/phoneme_model.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace bt  = brotensor;
namespace bsm = brosoundml;

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

// A deterministic 3 s scenario with content for every sensor: lead silence,
// a tone (faded out — a hard cutoff is a real transient), noise, clicks.
std::vector<float> scenario() {
    std::vector<float> s(static_cast<std::size_t>(0.4 * kRate), 0.0f);
    const std::size_t tone_n = static_cast<std::size_t>(0.8 * kRate);
    const std::size_t fade   = static_cast<std::size_t>(0.01 * kRate);
    const std::size_t t0     = s.size();
    for (std::size_t i = 0; i < tone_n; ++i) {
        const float g = i >= tone_n - fade
            ? static_cast<float>(tone_n - i) / static_cast<float>(fade)
            : 1.0f;
        s.push_back(g * 0.1f * static_cast<float>(
            std::sin(2.0 * kPi * 1000.0 * static_cast<double>(t0 + i) / kRate)));
    }
    s.resize(s.size() + static_cast<std::size_t>(0.3 * kRate), 0.0f);
    std::uint32_t x = 0x2468ace1u;
    for (std::size_t i = 0; i < static_cast<std::size_t>(0.8 * kRate); ++i) {
        x = x * 1664525u + 1013904223u;
        const float u = static_cast<float>(x >> 8) / static_cast<float>(1u << 24);
        s.push_back(0.05f * (2.0f * u - 1.0f));
    }
    s.resize(s.size() + static_cast<std::size_t>(0.7 * kRate), 0.0f);
    // Two clicks in the final quiet stretch.
    for (double at : {2.45, 2.75}) {
        std::size_t i0 = static_cast<std::size_t>(at * kRate);
        std::uint32_t c = static_cast<std::uint32_t>(i0) | 1u;
        for (std::size_t i = 0; i < 16 && i0 + i < s.size(); ++i) {
            c = c * 1664525u + 1013904223u;
            const float u = static_cast<float>(c >> 8) / static_cast<float>(1u << 24);
            s[i0 + i] = 0.8f * (2.0f * u - 1.0f);
        }
    }
    return s;
}

bool snapshots_equal(const bsm::SensorSnapshot& a, const bsm::SensorSnapshot& b,
                     const char* tag) {
    bool ok = true;
    auto chk = [&](bool c, const char* field) {
        if (!c) {
            std::fprintf(stderr, "  mismatch [%s]: %s\n", tag, field);
            ok = false;
        }
    };
    chk(a.frames == b.frames, "frames");
    chk(std::fabs(a.rms - b.rms)   < 1e-6f, "rms");
    chk(std::fabs(a.peak - b.peak) < 1e-6f, "peak");
    chk(std::fabs(a.db - b.db)     < 1e-4f, "db");
    chk(a.voice == b.voice, "voice");
    chk(a.voice_events == b.voice_events, "voice_events");
    chk(a.voice_frames == b.voice_frames, "voice_frames");
    chk(a.last_voice_frame == b.last_voice_frame, "last_voice_frame");
    chk(std::fabs(a.flux - b.flux) < 1e-4f, "flux");
    chk(a.onsets == b.onsets, "onsets");
    chk(a.last_onset_frame == b.last_onset_frame, "last_onset_frame");
    chk(std::fabs(a.periodicity - b.periodicity) < 1e-4f, "periodicity");
    chk(std::fabs(a.dominant_hz - b.dominant_hz) < 1e-2f, "dominant_hz");
    chk(a.tonal == b.tonal, "tonal");
    chk(a.tonal_events == b.tonal_events, "tonal_events");
    return ok;
}

// ─── 1. bus-driven SensorHub ≡ standalone SensorHub ────────────────────────
void test_hub_equivalence() {
    const std::vector<float> s = scenario();

    bsm::SensorHub standalone;          // owns its own front-end
    bsm::SensorHub driven;              // fed through the bus
    bsm::ListenBus bus;
    bus.check_compatible(driven);       // default configs must agree

    // Awkward chunk size (prime, smaller than a hop) so frames straddle
    // chunk boundaries on both paths.
    const int chunk = 977;
    std::size_t i = 0;
    std::int64_t bus_frames = 0;
    while (i < s.size()) {
        const int n = static_cast<int>(
            std::min<std::size_t>(chunk, s.size() - i));
        standalone.feed(s.data() + i, n);
        const bsm::ListenFeedResult r = bus.feed(s.data() + i, n, &driven, nullptr);
        bus_frames += r.frames;
        i += static_cast<std::size_t>(n);
    }

    const bsm::SensorSnapshot a = standalone.snapshot();
    const bsm::SensorSnapshot b = driven.snapshot();
    CHECK(snapshots_equal(a, b, "hub"), "bus-driven hub matches standalone hub");
    CHECK(bus_frames == a.frames, "bus frame count matches the hub's");
    // Tone/noise segment starts are genuine onsets too; the exact count is
    // pinned by the cross-path equality above, this just proves the scenario
    // exercised the sensors at all.
    CHECK(a.onsets >= 2 && a.voice_events >= 1,
          "scenario exercised the sensors (clicks counted, voice seen)");
    std::fprintf(stderr,
        "  hub: frames=%lld onsets=%lld voice_events=%lld periodicity=%.3f\n",
        static_cast<long long>(b.frames), static_cast<long long>(b.onsets),
        static_cast<long long>(b.voice_events), b.periodicity);
}

// ─── 2. bus-driven PhonemeSpotter ≡ standalone feed() ──────────────────────
void test_spotter_equivalence() {
    // Tiny random PhonemeNet, saved + loaded through the spotter twice so the
    // two instances are weight-identical.
    bsm::PhonemeNetConfig cfg;
    cfg.c_stem = 8;
    cfg.c[0] = 8; cfg.c[1] = 12; cfg.c[2] = 16; cfg.c[3] = 24;
    bsm::PhonemeClassMap cm;
    cm.num_classes  = 6;
    cm.class_names  = {"sil", "AA", "IY", "K", "S", "T"};
    cm.class_to_ids = {{0, 1}, {10, 11}, {12, 13}, {20}, {21, 22}, {23}};
    cm.rebuild_inverse();

    auto m = bsm::PhonemeNet::make(cfg, cm, bt::Device::CPU);
    m.xavier_init_weights(4242);
    m.fuse_bn();   // save(fused=true) requires it — runtime-checkpoint style
    const std::string path = "test_listen_bus_tiny.bpm";
    m.save(path);

    bsm::PhonemeSpotter standalone;
    bsm::PhonemeSpotter driven;
    standalone.load(path, bt::Device::CPU);
    driven.load(path, bt::Device::CPU);
    std::filesystem::remove(path);

    bsm::ListenBus bus;
    bus.check_compatible(driven);

    // Enroll the same template in both (random weights — the point is the
    // posterior streams and event lists match, not that anything fires).
    standalone.enroll_from_classes("probe", {1, 3, 2});
    driven.enroll_from_classes("probe", {1, 3, 2});

    const std::vector<float> s = scenario();
    const int chunk = 1601;
    std::size_t i = 0;
    std::size_t spots_a = 0, spots_b = 0;
    float max_post_diff = 0.0f;
    while (i < s.size()) {
        const int n = static_cast<int>(
            std::min<std::size_t>(chunk, s.size() - i));
        spots_a += standalone.feed(s.data() + i, n).size();
        spots_b += bus.feed(s.data() + i, n, nullptr, &driven).spots.size();

        const std::vector<float> pa = standalone.last_posterior();
        const std::vector<float> pb = driven.last_posterior();
        if (pa.size() != pb.size()) {
            max_post_diff = 1e9f;
        } else {
            for (std::size_t k = 0; k < pa.size(); ++k) {
                max_post_diff = std::max(max_post_diff, std::fabs(pa[k] - pb[k]));
            }
        }
        i += static_cast<std::size_t>(n);
    }

    CHECK(max_post_diff < 1e-5f,
          "bus-driven posterior stream matches standalone feed()");
    CHECK(spots_a == spots_b, "bus-driven event count matches standalone");
    CHECK(std::fabs(standalone.prefix_progress() - driven.prefix_progress()) < 1e-6f,
          "prefix progress matches");
    std::fprintf(stderr, "  spotter: max posterior diff=%.2e, events=%zu/%zu\n",
                 max_post_diff, spots_a, spots_b);
}

// ─── 3. bus-driven WakeWord ≡ standalone feed() ─────────────────────────────
void test_wake_equivalence() {
    // Tiny random 2D BC-ResNet, saved + loaded twice so the two detectors are
    // weight-identical (random weights — the contract under test is that the
    // bus-driven score stream matches standalone feed(), not that it fires).
    bsm::BcResnet2dConfig cfg;   // default recipe; n_mels matches the bus
    auto m = bsm::BcResnet2d::make(cfg, bt::Device::CPU);
    m.xavier_init_weights(2026);
    m.fuse_bn();
    const std::string path = "test_listen_bus_tiny.bw";
    m.save(path);

    bsm::WakeWord standalone;
    bsm::WakeWord driven;
    standalone.load(path, bt::Device::CPU);
    driven.load(path, bt::Device::CPU);
    std::filesystem::remove(path);

    bsm::ListenBus bus;
    bus.check_compatible(driven);

    const std::vector<float> s = scenario();
    const int chunk = 1601;
    std::size_t i = 0;
    int fires_a = 0, fires_b = 0;
    float max_score_diff = 0.0f;
    while (i < s.size()) {
        const int n = static_cast<int>(
            std::min<std::size_t>(chunk, s.size() - i));
        if (standalone.feed(s.data() + i, n)) ++fires_a;
        const bsm::ListenFeedResult r =
            bus.feed(s.data() + i, n, nullptr, nullptr, &driven);
        if (r.wake_fired) ++fires_b;
        max_score_diff = std::max(max_score_diff,
            std::fabs(standalone.last_score() - driven.last_score()));
        i += static_cast<std::size_t>(n);
    }

    CHECK(max_score_diff < 1e-5f,
          "bus-driven wake score stream matches standalone feed()");
    CHECK(fires_a == fires_b, "bus-driven fire count matches standalone");
    std::fprintf(stderr, "  wake: max score diff=%.2e, fires=%d/%d\n",
                 max_score_diff, fires_a, fires_b);
}

// ─── 4. attach validation + seam error paths ───────────────────────────────
void test_validation() {
    bsm::ListenBus bus;

    bsm::SensorHubConfig bad;
    bad.mel.n_mels = 80;
    bsm::SensorHub hub80(bad);
    bool threw = false;
    try {
        bus.check_compatible(hub80);
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("n_mels") != std::string::npos;
    }
    CHECK(threw, "mismatched hub framing is rejected, naming the field");

    bsm::PhonemeSpotter unloaded;
    threw = false;
    try {
        bus.check_compatible(unloaded);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "unloaded spotter is rejected at attach");

    threw = false;
    try {
        float col[40] = {};
        unloaded.feed_mel(col, 1);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "feed_mel without a model throws");

    bsm::WakeWord unloaded_wake;
    threw = false;
    try {
        bus.check_compatible(unloaded_wake);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "unloaded wake detector is rejected at attach");

    threw = false;
    try {
        float col[40] = {};
        unloaded_wake.feed_mel(col, 1);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "wake feed_mel without a model throws");

    // Feeding with no consumers still advances the front-end.
    std::vector<float> quiet(kRate, 0.0f);
    const bsm::ListenFeedResult r =
        bus.feed(quiet.data(), static_cast<int>(quiet.size()), nullptr, nullptr);
    CHECK(r.frames > 90 && r.spots.empty(), "consumer-less feed still frames");
}

}  // namespace

int main() {
    bt::init();
    try {
        test_hub_equivalence();
        test_spotter_equivalence();
        test_wake_equivalence();
        test_validation();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: unhandled exception: %s\n", e.what());
        ++failures;
    }
    if (failures == 0) {
        std::fprintf(stderr, "test_listen_bus: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_listen_bus: %d FAILURE(S)\n", failures);
    return 1;
}
