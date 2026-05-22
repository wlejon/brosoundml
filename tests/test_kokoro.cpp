// Kokoro API surface: construction, default state, and the staged
// not-yet-implemented contract. Locks the public shape while the forward pass
// is in build-out (see README.md) — these checks flip to real behaviour as
// each stage lands.
#include "brosoundml/kokoro.h"

#include <cstdio>
#include <stdexcept>
#include <string>

static int failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

// Returns true if calling `fn` throws a std::runtime_error.
template <typename Fn>
static bool throws_runtime_error(Fn&& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

int main() {
    using brosoundml::Kokoro;
    using brosoundml::Voice;

    Kokoro k;
    CHECK(!k.loaded(), "a fresh Kokoro is not loaded");
    CHECK(k.config().sample_rate == 24000, "default config sample rate is 24 kHz");

    // The pipeline is in build-out: every weight-dependent entry point throws a
    // staged std::runtime_error rather than returning a wrong result.
    CHECK(throws_runtime_error([&] { k.load("nonexistent-model-dir"); }),
          "load() throws while model loading is in build-out");
    CHECK(throws_runtime_error([&] { k.load_voice("nonexistent-voice"); }),
          "load_voice() throws while voice loading is in build-out");
    CHECK(throws_runtime_error([&] {
              Voice v;
              k.synthesize({1, 2, 3}, v);
          }),
          "synthesize() throws while the forward pass is in build-out");
    CHECK(throws_runtime_error([&] {
              Voice v;
              (void)v.pick_for(8);
          }),
          "Voice::pick_for() throws while voice indexing is in build-out");

    // Move construction keeps the object usable (pImpl move).
    Kokoro moved = std::move(k);
    CHECK(!moved.loaded(), "moved-to Kokoro is not loaded");

    if (failures == 0) {
        std::printf("test_kokoro: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_kokoro: %d check(s) failed\n", failures);
    return 1;
}
