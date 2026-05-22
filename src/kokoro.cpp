#include "brosoundml/kokoro.h"

#include <stdexcept>

namespace brosoundml {

// ─── Kokoro::Impl ──────────────────────────────────────────────────────────
//
// Holds everything heavy: the parsed config, the device the weights live on,
// and (once the forward pass is built out) the module graph. Kept behind a
// pImpl so kokoro.h carries no brotensor module internals.
struct Kokoro::Impl {
    KokoroConfig      config;
    brotensor::Device device = brotensor::Device::CPU;
    bool              loaded = false;
    // TODO(build-out): plBERT, text encoder, predictor, and decoder module
    // graphs + their safetensors-loaded weight tensors land here.
};

// pImpl: the special members are defined here, where Impl is a complete type.
Kokoro::Kokoro() : impl_(std::make_unique<Impl>()) {}
Kokoro::~Kokoro() = default;
Kokoro::Kokoro(Kokoro&&) noexcept = default;
Kokoro& Kokoro::operator=(Kokoro&&) noexcept = default;

void Kokoro::load(const std::string& model_dir, brotensor::Device device) {
    impl_->device = device;
    // Build-out stage 1: parse config.json into KokoroConfig, then load
    // model.safetensors into the module graph. See README.md "Build-out plan".
    throw std::runtime_error(
        "brosoundml: Kokoro::load: model loading is not yet implemented "
        "(stage 1 of the README build-out plan) — requested '" + model_dir + "'");
}

Voice Kokoro::load_voice(const std::string& voice_path) const {
    throw std::runtime_error(
        "brosoundml: Kokoro::load_voice: voice-pack loading is not yet "
        "implemented (stage 1 of the README build-out plan) — requested '" +
        voice_path + "'");
}

AudioBuffer Kokoro::synthesize(const std::vector<int32_t>& /*phoneme_ids*/,
                               const Voice& /*voice*/,
                               float /*speed*/) const {
    throw std::runtime_error(
        "brosoundml: Kokoro::synthesize: the Kokoro forward pass is not yet "
        "implemented — see the build-out plan in README.md");
}

const KokoroConfig& Kokoro::config() const { return impl_->config; }
bool Kokoro::loaded() const { return impl_->loaded; }

// ─── Voice ─────────────────────────────────────────────────────────────────

brotensor::Tensor Voice::pick_for(int /*n_phonemes*/) const {
    throw std::runtime_error(
        "brosoundml: Voice::pick_for: voice-pack indexing is not yet "
        "implemented (stage 1 of the README build-out plan)");
}

}
