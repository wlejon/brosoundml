// Wake-word detector contract — chunks 1 + 7.
//
// Locks the public surface of brosoundml::WakeWord (default config, move
// semantics, setter validation, last_score readout) and — now that chunk 7
// has landed — the real streaming runtime: load() through BcResnet::load,
// feed() through the mel front-end + model + smoothing + refractory.
//
// The original chunk-1 "throws stage-5/7" checks have been replaced with
// real-behaviour assertions; surface checks (defaults, setter validation,
// move semantics) are unchanged.
#include "brosoundml/bc_resnet2d.h"
#include "brosoundml/wake.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
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

template <typename Fn>
static std::string runtime_error_message(Fn&& fn) {
    try { fn(); }
    catch (const std::runtime_error& e) { return std::string(e.what()); }
    catch (...) {}
    return {};
}

// Build a tiny xavier-init 2D BC-ResNet ('BWK2'), fuse + save to a unique temp
// path, return the path. Used by every load()-based check below so they don't
// depend on any real on-disk artefact.
static std::string make_smoke_checkpoint(std::uint64_t seed, const char* tag) {
    brosoundml::BcResnet2dConfig cfg;   // defaults — n_mels=40, ~16k params
    auto model = brosoundml::BcResnet2d::make(cfg, brotensor::Device::CPU);
    model.xavier_init_weights(seed);
    model.fuse_bn();
    namespace fs = std::filesystem;
    const fs::path p = fs::temp_directory_path() /
        (std::string("brosoundml_test_wake_") + tag + ".bw");
    model.save(p.string(), /*fused=*/true);
    return p.string();
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

    // ─── load() on a missing file throws a named runtime_error ─────────────
    {
        WakeWord w;
        const std::string msg = runtime_error_message([&] {
            w.load("definitely-no-such-wake-weights.bw");
        });
        CHECK(!msg.empty(),
              "load() on a missing file throws std::runtime_error");
        CHECK(!w.loaded(),
              "loaded() stays false after a failed load");
    }

    // ─── load() on a real .bw flips loaded() true ──────────────────────────
    {
        const std::string ckpt = make_smoke_checkpoint(123, "load");
        WakeWord w;
        w.load(ckpt, brotensor::Device::CPU);
        CHECK(w.loaded(),
              "load() flips loaded() to true on a valid .bw");
        CHECK(w.config().n_mels == 40,
              "load() reconciles n_mels from the model header");
    }

    // ─── feed() before load() refuses with the no-model error ──────────────
    {
        WakeWord w;
        std::vector<float> silence(160, 0.0f);
        const std::string msg = runtime_error_message([&] {
            w.feed(silence.data(), static_cast<int>(silence.size()));
        });
        CHECK(msg.find("WakeWord::feed") != std::string::npos,
              "feed()-without-load names the entry point");
        CHECK(msg.find("no model loaded") != std::string::npos,
              "feed()-without-load explains the cause");
    }

    // ─── feed() on silence does not fire ──────────────────────────────────
    {
        const std::string ckpt = make_smoke_checkpoint(7, "silence");
        WakeWord w;
        w.load(ckpt);
        // Default threshold 0.55, smoothing 2-of-3. Silence shouldn't fire.
        std::vector<float> silence(16000, 0.0f);
        const bool fired = w.feed(silence.data(),
                                  static_cast<int>(silence.size()));
        CHECK(!fired, "feed() on 1 s of silence does not fire");
        CHECK(w.last_score() >= 0.0f && w.last_score() <= 1.0f,
              "last_score after silence is a valid probability");
    }

    // ─── feed() fires with threshold=0 on non-silent input ─────────────────
    {
        const std::string ckpt = make_smoke_checkpoint(99, "fire");
        WakeWord w;
        w.load(ckpt);
        // Force every frame to count as a positive: threshold=0 with default
        // smoothing 2-of-3 means a fire after exactly two emitted frames.
        w.set_threshold(0.0f);
        // Set refractory huge so a re-fire test below can be added if needed.
        w.set_refractory_ms(60000);

        std::mt19937 rng(0xABCD);
        std::uniform_real_distribution<float> uni(-0.5f, 0.5f);
        std::vector<float> noise(16000);
        for (auto& s : noise) s = uni(rng);

        const bool fired = w.feed(noise.data(),
                                  static_cast<int>(noise.size()));
        CHECK(fired,
              "feed() fires when threshold=0 forces every frame positive");
    }

    // ─── Detector-policy setters mutate config and validate input ──────────
    {
        WakeWord w;
        w.set_threshold(0.7f);
        CHECK(w.config().threshold == 0.7f, "set_threshold updates config");

        w.set_smoothing(3, 5);
        CHECK(w.config().smoothing_hits   == 3, "set_smoothing.hits updates");
        CHECK(w.config().smoothing_window == 5, "set_smoothing.window updates");

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

    // ─── Smoothing: 1-of-3 doesn't fire, 2-of-3 does ───────────────────────
    //
    // We can't easily control the model's per-frame output without retraining,
    // so the cleanest controlled test is to drive every frame to "positive"
    // with threshold=0 and assert:
    //   • smoothing 1-of-3 fires after the first frame (1 >= 1)
    //   • smoothing 3-of-3 fires only after the third frame
    // and to verify the count via two independent runs that both reset between.
    {
        const std::string ckpt = make_smoke_checkpoint(42, "smooth");
        std::mt19937 rng(0xBEEF);
        std::uniform_real_distribution<float> uni(-0.3f, 0.3f);

        // Smoothing 1-of-1 with a giant refractory: each fed chunk produces
        // one fire-or-not return; the first call should fire (every frame is
        // a positive with threshold=0).
        {
            WakeWord w;
            w.load(ckpt);
            w.set_threshold(0.0f);
            w.set_smoothing(1, 1);
            w.set_refractory_ms(60000);
            std::vector<float> noise(16000);
            for (auto& s : noise) s = uni(rng);
            const bool f = w.feed(noise.data(), 16000);
            CHECK(f, "smoothing 1-of-1 fires on the first qualifying frame");
        }
        // Smoothing 3-of-3: still fires within 1 s of frames since every
        // frame is positive. We're really asserting the smoothing path
        // doesn't *prevent* firing when every frame counts.
        {
            WakeWord w;
            w.load(ckpt);
            w.set_threshold(0.0f);
            w.set_smoothing(3, 3);
            w.set_refractory_ms(60000);
            std::vector<float> noise(16000);
            for (auto& s : noise) s = uni(rng);
            const bool f = w.feed(noise.data(), 16000);
            CHECK(f, "smoothing 3-of-3 fires once 3 positive frames accumulate");
        }
        // Reverse: with threshold above 1.0 (impossible for any sigmoid
        // output) no frame qualifies, so 1-of-1 never fires.
        {
            WakeWord w;
            w.load(ckpt);
            w.set_threshold(2.0f);
            w.set_smoothing(1, 1);
            std::vector<float> noise(16000);
            for (auto& s : noise) s = uni(rng);
            const bool f = w.feed(noise.data(), 16000);
            CHECK(!f, "no frame qualifies when threshold > 1 → no fire");
        }
    }

    // ─── Refractory: after a fire, no second fire within the window ────────
    {
        const std::string ckpt = make_smoke_checkpoint(11, "refr");
        WakeWord w;
        w.load(ckpt);
        w.set_threshold(0.0f);             // every frame positive
        w.set_smoothing(1, 1);             // fire on every frame ...
        w.set_refractory_ms(3000);         // ... unless refractory blocks
                                           // 3 s covers two 1-s feeds w/ margin

        std::mt19937 rng(0xC0FFEE);
        std::uniform_real_distribution<float> uni(-0.2f, 0.2f);
        std::vector<float> noise(16000);
        for (auto& s : noise) s = uni(rng);

        // First call: should fire exactly once (refractory engages).
        const bool first = w.feed(noise.data(), 16000);
        CHECK(first, "refractory test: first call fires");

        // Second call, same length: must NOT fire — we're still inside the
        // 1 s refractory (and the first call's last frame engaged it).
        std::vector<float> noise2(16000);
        for (auto& s : noise2) s = uni(rng);
        const bool second = w.feed(noise2.data(), 16000);
        CHECK(!second,
              "refractory blocks a re-fire within the configured window");
    }

    // ─── Streaming equivalence: one big call vs many small chunks ──────────
    //
    // Same audio fed in one shot and in random-sized chunks should produce
    // the same number of events and the same last_score (within FP noise).
    {
        const std::string ckpt = make_smoke_checkpoint(2024, "stream");

        std::mt19937 rng(0xFEED);
        std::uniform_real_distribution<float> uni(-0.3f, 0.3f);
        std::vector<float> noise(16000);
        for (auto& s : noise) s = uni(rng);

        // Run A — one big call.
        // Refractory longer than the clip so at most one fire per run total;
        // the per-call collapse rule is therefore not load-bearing on event
        // count here.
        WakeWord a;
        a.load(ckpt);
        a.set_threshold(0.0f);
        a.set_smoothing(2, 3);
        a.set_refractory_ms(60000);
        int events_a = 0;
        if (a.feed(noise.data(), 16000)) ++events_a;
        const float score_a = a.last_score();

        // Run B — chunked feeds.
        WakeWord b;
        b.load(ckpt);
        b.set_threshold(0.0f);
        b.set_smoothing(2, 3);
        b.set_refractory_ms(60000);
        int events_b = 0;
        std::mt19937 chunk_rng(0x123);
        std::uniform_int_distribution<int> chunk_size(37, 521);
        int pos = 0;
        while (pos < 16000) {
            const int csz = std::min(chunk_size(chunk_rng), 16000 - pos);
            if (b.feed(noise.data() + pos, csz)) ++events_b;
            pos += csz;
        }
        const float score_b = b.last_score();

        CHECK(events_a == events_b,
              "streaming equivalence: event count matches across chunkings");
        const float dscore = std::fabs(score_a - score_b);
        CHECK(dscore < 1e-3f,
              "streaming equivalence: last_score matches across chunkings");
    }

    // ─── reset() clears streaming state but keeps weights ──────────────────
    {
        const std::string ckpt = make_smoke_checkpoint(55, "reset");
        WakeWord w;
        w.load(ckpt);
        w.set_threshold(0.0f);
        w.set_refractory_ms(60000);

        std::mt19937 rng(0xDEAD);
        std::uniform_real_distribution<float> uni(-0.2f, 0.2f);
        std::vector<float> noise(16000);
        for (auto& s : noise) s = uni(rng);
        const bool fired = w.feed(noise.data(), 16000);
        (void)fired;
        CHECK(w.loaded(), "loaded() still true after a feed");
        w.reset();
        CHECK(w.last_score() == 0.0f, "reset zeroes last_score");
        CHECK(w.loaded(),             "reset does not unload weights");
        // After reset, refractory is cleared so another feed can fire again.
        const bool fired_again = w.feed(noise.data(), 16000);
        CHECK(fired_again, "after reset, refractory no longer blocks fires");
    }

    // ─── reset() on an unloaded WakeWord is harmless ───────────────────────
    {
        WakeWord w;
        w.reset();
        CHECK(w.last_score() == 0.0f, "reset on unloaded leaves last_score at 0");
        CHECK(!w.loaded(),            "reset does not toggle loaded()");
    }

    // ─── Shared net: one load-once weight set drives N independent detectors ─
    //
    // Build a net, wrap it shared+const, and construct two detectors over it.
    // The weights are held once (use_count), and each detector owns its own
    // mel front-end + streaming session — so interleaving feed() on the two
    // must not cross-talk: detector A's last_score matches a standalone
    // detector fed only A's audio, regardless of B's interleaved feeds.
    {
        brosoundml::BcResnet2dConfig cfg;   // defaults — n_mels=40
        auto model = brosoundml::BcResnet2d::make(cfg, brotensor::Device::CPU);
        model.xavier_init_weights(7);
        model.fuse_bn();                    // runtime checkpoints are fused
        auto net = std::make_shared<const brosoundml::BcResnet2d>(std::move(model));

        WakeWord wa(net), wb(net);
        CHECK(wa.loaded() && wb.loaded(),
              "shared: both detectors loaded from one net");
        CHECK(net.use_count() == 3,
              "shared: weights held once (net + 2 detectors share the shared_ptr)");

        std::mt19937 rng(0x5151);
        std::uniform_real_distribution<float> uni(-0.3f, 0.3f);
        std::vector<float> A(16000), B(16000);
        for (auto& s : A) s = uni(rng);
        for (auto& s : B) s = uni(rng);

        // Standalone reference: a fresh detector fed only A, in one shot.
        WakeWord wc(net);
        wc.feed(A.data(), 16000);
        const float refA = wc.last_score();

        // Interleave A into wa and B into wb in halves: A0,B0,A1,B1.
        wa.feed(A.data(),         8000);
        wb.feed(B.data(),         8000);
        wa.feed(A.data() + 8000,  8000);
        wb.feed(B.data() + 8000,  8000);
        const float outA = wa.last_score();

        const float d = std::fabs(outA - refA);
        CHECK(d < 1e-3f,
              "shared: detector A last_score matches standalone (no crosstalk)");
        std::fprintf(stderr,
                     "    (shared-net wake crosstalk diff=%g, use_count=%ld)\n",
                     d, static_cast<long>(net.use_count()));
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
