#include "brosoundml/wake.h"

#include <brotensor/runtime.h>

#include <atomic>
#include <stdexcept>
#include <string>

namespace brosoundml {

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

}  // namespace

// ─── WakeWord::Impl ────────────────────────────────────────────────────────
//
// Chunk 1 holds only the policy state — the detector configuration and the
// last-score readout. The feature ring, the BC-ResNet conv-state cache, the
// smoothing/refractory bookkeeping and the brotensor weight tensors land in
// later chunks (see the build-out plan in wake.h). Until then the entry
// points that depend on those land in throw a runtime_error naming the
// missing stage — the staged-stub convention used across this repo.
struct WakeWord::Impl {
    WakeConfig         config;
    bool               loaded = false;
    std::atomic<float> last_score{0.0f};
};

WakeWord::WakeWord() : impl_(std::make_unique<Impl>()) {}
WakeWord::~WakeWord() = default;
WakeWord::WakeWord(WakeWord&&) noexcept = default;
WakeWord& WakeWord::operator=(WakeWord&&) noexcept = default;

void WakeWord::load(const std::string& weights_path,
                    brotensor::Device /*device*/) {
    // Probe brotensor backends so a future Device::CUDA / Device::Metal load
    // reaches the dispatcher cleanly once stage 5 lands. Kept here rather
    // than at first feed() so the caller's load-time error path covers it.
    brotensor::init();
    (void)weights_path;
    fail("WakeWord::load",
         "stage 5 (BC-ResNet model + weight loader) not built yet — "
         "wake-word weights cannot be read until that chunk lands");
}

bool WakeWord::feed(const float* /*samples*/, int /*n*/) {
    if (!impl_->loaded) {
        fail("WakeWord::feed", "no model loaded; call WakeWord::load() first");
    }
    // Once stage 2 (front-end) and stage 7 (runtime + smoothing) land this
    // becomes the real streaming detector. Until then the loaded() guard
    // above keeps callers from reaching this line in tests.
    fail("WakeWord::feed",
         "stage 7 (streaming runtime + smoothing) not built yet");
}

void WakeWord::set_threshold(float t) {
    impl_->config.threshold = t;
}

void WakeWord::set_smoothing(int hits, int window) {
    if (window <= 0 || hits <= 0 || hits > window) {
        fail("WakeWord::set_smoothing",
             "invalid (hits, window): hits=" + std::to_string(hits) +
             " window=" + std::to_string(window) +
             " (require 0 < hits <= window)");
    }
    impl_->config.smoothing_hits   = hits;
    impl_->config.smoothing_window = window;
}

void WakeWord::set_refractory_ms(int ms) {
    if (ms < 0) {
        fail("WakeWord::set_refractory_ms",
             "negative refractory: " + std::to_string(ms));
    }
    impl_->config.refractory_ms = ms;
}

float WakeWord::last_score() const {
    return impl_->last_score.load(std::memory_order_relaxed);
}

void WakeWord::reset() {
    // Streaming-state reset lands with stage 7; the policy fields stay put.
    // last_score is conceptually part of streaming state, so clear it here so
    // the chunk-1 contract is already complete for this method.
    impl_->last_score.store(0.0f, std::memory_order_relaxed);
}

const WakeConfig& WakeWord::config() const { return impl_->config; }
bool              WakeWord::loaded() const { return impl_->loaded; }

}  // namespace brosoundml
