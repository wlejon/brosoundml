// Wake-word detector contract — chunk 1.
//
// Locks the public surface of brosoundml::WakeWord (default config, move
// semantics, setter validation, last_score readout) and the staged-stub
// behaviour of the entry points that depend on stages 2/5/7. Checks here
// flip from "throws with stage tag" to "real behaviour" as those chunks land.
#include "brosoundml/wake.h"

#include <brotensor/runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

template <typename Fn>
static bool throws_runtime_error(Fn&& fn) {
    try { fn(); }
    catch (const std::runtime_error&) { return true; }
    catch (...) { return false; }
    return false;
}

// Capture the message of a thrown runtime_error so we can assert the stub
// names its missing stage. Returns empty if no runtime_error escaped.
template <typename Fn>
static std::string runtime_error_message(Fn&& fn) {
    try { fn(); }
    catch (const std::runtime_error& e) { return std::string(e.what()); }
    catch (...) {}
    return {};
}

static int run();
int main() {
    brotensor::init();
    try { return run(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "test_wake: uncaught exception: %s\n", e.what());
        return 2;
    }
    catch (...) {
        std::fprintf(stderr, "test_wake: uncaught non-std exception\n");
        return 2;
    }
}

static int run() {
    using brosoundml::WakeConfig;
    using brosoundml::WakeWord;

    // ─── Default config matches the "computer" recipe ──────────────────────
    {
        WakeConfig c;
        CHECK(c.sample_rate            == 16000, "default sample_rate 16 kHz");
        CHECK(c.hop_length             == 160,   "default hop 10 ms @ 16 kHz");
        CHECK(c.win_length             == 400,   "default win 25 ms @ 16 kHz");
        CHECK(c.n_fft                  == 512,   "default n_fft 512");
        CHECK(c.n_mels                 == 40,    "default n_mels 40");
        CHECK(c.receptive_field_frames == 100,   "default 1.0 s receptive field");
        CHECK(c.smoothing_hits         == 2,     "default smoothing hits 2");
        CHECK(c.smoothing_window       == 3,     "default smoothing window 3");
        CHECK(c.refractory_ms          == 500,   "default refractory 500 ms");
    }

    // ─── Default-constructed WakeWord ──────────────────────────────────────
    {
        WakeWord w;
        CHECK(!w.loaded(),                  "a fresh WakeWord is not loaded");
        CHECK(w.last_score() == 0.0f,       "last_score starts at 0.0");
        CHECK(w.config().sample_rate == 16000,
                                            "default WakeWord config carries default WakeConfig");
    }

    // ─── load() is staged out — throws a named-stage runtime_error ─────────
    {
        WakeWord w;
        const std::string msg = runtime_error_message([&] {
            w.load("nonexistent-wake-weights.bw");
        });
        CHECK(!msg.empty(),
              "load() throws std::runtime_error while stage 5 is unbuilt");
        CHECK(msg.find("WakeWord::load") != std::string::npos,
              "load() error names the entry point");
        CHECK(msg.find("stage 5") != std::string::npos,
              "load() error names the missing stage (stage 5)");
        CHECK(!w.loaded(),
              "loaded() stays false after a failed load");
    }

    // ─── feed() before load() refuses with the no-model error ──────────────
    {
        WakeWord w;
        std::vector<float> silence(160, 0.0f);  // one frame worth of input
        const std::string msg = runtime_error_message([&] {
            w.feed(silence.data(), static_cast<int>(silence.size()));
        });
        CHECK(msg.find("WakeWord::feed") != std::string::npos,
              "feed()-without-load names the entry point");
        CHECK(msg.find("no model loaded") != std::string::npos,
              "feed()-without-load explains the cause");
    }

    // ─── Detector-policy setters mutate config and validate input ──────────
    {
        WakeWord w;
        w.set_threshold(0.7f);
        CHECK(w.config().threshold == 0.7f, "set_threshold updates config");

        w.set_smoothing(3, 5);
        CHECK(w.config().smoothing_hits   == 3, "set_smoothing.hits updates");
        CHECK(w.config().smoothing_window == 5, "set_smoothing.window updates");

        // hits > window must be rejected — defenses against caller mistakes
        // that would otherwise silently make the detector unfireable.
        CHECK(throws_runtime_error([&] { w.set_smoothing(4, 3); }),
              "set_smoothing rejects hits > window");
        CHECK(throws_runtime_error([&] { w.set_smoothing(0, 3); }),
              "set_smoothing rejects hits == 0");
        CHECK(throws_runtime_error([&] { w.set_smoothing(1, 0); }),
              "set_smoothing rejects window == 0");

        w.set_refractory_ms(750);
        CHECK(w.config().refractory_ms == 750,
              "set_refractory_ms updates config");
        CHECK(throws_runtime_error([&] { w.set_refractory_ms(-1); }),
              "set_refractory_ms rejects negative");
    }

    // ─── reset() is a no-op pre-stream and clears last_score ───────────────
    {
        WakeWord w;
        w.reset();
        CHECK(w.last_score() == 0.0f, "reset leaves last_score at 0.0");
        CHECK(!w.loaded(),            "reset does not toggle loaded()");
    }

    // ─── Move construction / assignment preserves config and state ─────────
    {
        WakeWord w;
        w.set_threshold(0.42f);
        WakeWord moved = std::move(w);
        CHECK(moved.config().threshold == 0.42f,
              "move-constructed WakeWord keeps mutated config");
        CHECK(!moved.loaded(), "moved WakeWord is not loaded");

        WakeWord assigned;
        assigned = std::move(moved);
        CHECK(assigned.config().threshold == 0.42f,
              "move-assigned WakeWord keeps mutated config");
    }

    if (failures == 0) {
        std::printf("test_wake: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_wake: %d check(s) failed\n", failures);
    return 1;
}
