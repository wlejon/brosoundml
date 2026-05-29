#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "brosoundml/wake.h"

#include "brosoundml/bc_resnet2d.h"
#include "brosoundml/mel.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

constexpr std::uint32_t kMagicBWK2 = 0x324B5742u;  // 'B''W''K''2' — 2D BC-ResNet
constexpr std::uint32_t kMagicBWAK = 0x4B415742u;  // 'B''W''A''K' — legacy 1D

// Peek the 4-byte magic at the head of a checkpoint without parsing it.
std::uint32_t peek_magic(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) fail("WakeWord::load", "could not open '" + path + "'");
    std::uint32_t magic = 0;
    const bool ok = std::fread(&magic, 4, 1, fp) == 1;
    std::fclose(fp);
    if (!ok) fail("WakeWord::load", "could not read magic from '" + path + "'");
    return magic;
}

inline float sigmoidf(float x) {
    // Numerically stable sigmoid.
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

}  // namespace

// ─── WakeWord::Impl ────────────────────────────────────────────────────────
//
// Chunk-7 streaming runtime. Owns the BC-ResNet (loaded weights, conv-state
// cache, GAP ring) and the mel front-end (sample ring, mel filter, analysis
// window), plus the detector-policy bookkeeping: a smoothing-decision ring
// holding the last N frame-positive flags and a refractory counter measured
// in mel frames (NOT samples — once a fire happens we just count emitted mel
// frames down to zero).
//
// Allocation policy: every per-frame brotensor tensor (`frame_tensor`,
// `logit_tensor`, the per-call mel-batch tensor) is a member resident on the
// model device and reused across calls. Conv-state caches inside the model
// are likewise pre-sized by load(). feed() does no per-frame heap traffic
// beyond a small `std::vector<float>` append/erase on the sample ring.
struct WakeWord::Impl {
    // Public surface state.
    WakeConfig         config;
    bool               loaded = false;
    std::atomic<float> last_score{0.0f};

    // Streaming state.
    std::unique_ptr<MelFrontend> mel;       // built on load()
    std::unique_ptr<BcResnet2d>  model;     // loaded on load()
    bt::Device                   device = bt::Device::CPU;

    // Sample-side buffering: feed() appends incoming samples here, then drains
    // them in hop-sized chunks into the mel front-end. The mel front-end has
    // its own ring (covering the win_length - hop_length carry-over), so we
    // only need to hold what hasn't been handed to it yet — effectively zero
    // in steady state since we drain to the nearest hop boundary every call.
    std::vector<float> sample_ring;

    // Smoothing decision ring — stores the most recent `smoothing_window`
    // boolean frame-positives. Implemented as a fixed-size vector with a head
    // index so eviction is O(1).
    std::vector<std::uint8_t> decisions;
    int                       decision_head = 0;
    int                       decision_len  = 0;

    // Refractory: number of mel frames remaining during which fires are
    // suppressed. Decremented once per emitted frame.
    int refractory_frames = 0;

    // Warmup: the 2D model's streaming per-frame logit pools the head feature
    // over a GAP ring that only fills after `warmup_frames` frames. Before that
    // the logit is computed over a partial average (a few onset/silence frames)
    // that the model never saw in training — it can read as a confident fire on
    // any clip, which is a 100% false-accept at every stream start. We suppress
    // fires until `frames_since_reset >= warmup_frames` so the first trusted
    // decision pools a full window, matching the whole-clip GAP used in
    // training. Set from BcResnet2d::gap_window_frames() at load().
    int warmup_frames      = 0;
    int frames_since_reset = 0;

    // Thread-safe view of the detector policy. Loaded relaxed at the top of
    // every frame, matching the contract in wake.h.
    std::atomic<float> a_threshold{0.55f};
    std::atomic<int>   a_smoothing_hits{2};
    std::atomic<int>   a_smoothing_window{3};
    std::atomic<int>   a_refractory_ms{500};
};

WakeWord::WakeWord() : impl_(std::make_unique<Impl>()) {}
WakeWord::~WakeWord() = default;
WakeWord::WakeWord(WakeWord&&) noexcept = default;
WakeWord& WakeWord::operator=(WakeWord&&) noexcept = default;

void WakeWord::load(const std::string& weights_path,
                    brotensor::Device device) {
    brotensor::init();

    // Detect the checkpoint format. The microphone-robust recipe ships the 2D
    // BC-ResNet ('BWK2'); the legacy 1D model ('BWAK') keyed on the TTS
    // acquisition envelope and was dead on real speech, so it is not loadable
    // here — fail loudly pointing at the retrain rather than mis-running it.
    const std::uint32_t magic = peek_magic(weights_path);
    if (magic == kMagicBWAK) {
        fail("WakeWord::load",
             "'" + weights_path + "' is a legacy 1D ('BWAK') checkpoint; the "
             "wake stack now requires a 2D PCEN model ('BWK2') — retrain with "
             "brosoundml_wake_train");
    }
    if (magic != kMagicBWK2) {
        fail("WakeWord::load", "'" + weights_path + "' has unrecognised magic "
             "(expected 'BWK2')");
    }

    // Load the 2D BC-ResNet. Throws on bad-version / shape mismatch.
    auto model = std::make_unique<BcResnet2d>(
        BcResnet2d::load(weights_path, device));

    const int model_n_mels = model->config().n_mels;
    if (model_n_mels <= 0) {
        fail("WakeWord::load",
             "model header reports non-positive n_mels=" +
             std::to_string(model_n_mels));
    }
    // Trust the model. The .bw header is authoritative; if the caller's
    // WakeConfig.n_mels disagrees, overwrite ours and run with the model's.
    if (impl_->config.n_mels != model_n_mels) {
        impl_->config.n_mels = model_n_mels;
    }

    // Build the mel front-end with config matching the model. The 2D recipe is
    // trained on PCEN features (per-channel energy normalization that cancels
    // mic/channel spectral tilt), so the runtime front-end must match.
    MelConfig mcfg;
    mcfg.sample_rate = impl_->config.sample_rate;
    mcfg.n_fft       = impl_->config.n_fft;
    mcfg.win_length  = impl_->config.win_length;
    mcfg.hop_length  = impl_->config.hop_length;
    mcfg.n_mels      = impl_->config.n_mels;
    mcfg.fmin        = impl_->config.mel_fmin;
    mcfg.fmax        = impl_->config.mel_fmax;
    mcfg.compression = MelCompression::PCEN;
    auto mel = std::make_unique<MelFrontend>(mcfg, device);

    // Warmup span = the model's GAP-ring capacity (read before the move).
    const int warmup = model->gap_window_frames();

    // Commit on success.
    impl_->mel    = std::move(mel);
    impl_->model  = std::move(model);
    impl_->device = device;
    impl_->loaded = true;
    impl_->warmup_frames      = warmup > 0 ? warmup : 0;
    impl_->frames_since_reset = 0;

    // Mirror policy into atomics so feed() picks them up.
    impl_->a_threshold       .store(impl_->config.threshold,        std::memory_order_relaxed);
    impl_->a_smoothing_hits  .store(impl_->config.smoothing_hits,   std::memory_order_relaxed);
    impl_->a_smoothing_window.store(impl_->config.smoothing_window, std::memory_order_relaxed);
    impl_->a_refractory_ms   .store(impl_->config.refractory_ms,    std::memory_order_relaxed);

    // Size the decision ring for the current window. set_smoothing() can
    // grow/shrink this later — the ring is rebuilt on resize.
    impl_->decisions.assign(
        static_cast<std::size_t>(impl_->config.smoothing_window), 0u);
    impl_->decision_head = 0;
    impl_->decision_len  = 0;
    impl_->refractory_frames = 0;
    impl_->sample_ring.clear();
    impl_->last_score.store(0.0f, std::memory_order_relaxed);
}

bool WakeWord::feed(const float* samples, int n) {
    if (!impl_->loaded) {
        fail("WakeWord::feed", "no model loaded; call WakeWord::load() first");
    }
    if (n < 0) {
        fail("WakeWord::feed",
             "negative sample count " + std::to_string(n));
    }
    if (n > 0 && samples == nullptr) {
        fail("WakeWord::feed", "null samples with n > 0");
    }

    // Snapshot policy for the duration of this call.
    const float threshold        = impl_->a_threshold       .load(std::memory_order_relaxed);
    const int   smoothing_hits   = impl_->a_smoothing_hits  .load(std::memory_order_relaxed);
    const int   smoothing_window = impl_->a_smoothing_window.load(std::memory_order_relaxed);
    const int   refractory_ms    = impl_->a_refractory_ms   .load(std::memory_order_relaxed);

    const int sample_rate = impl_->config.sample_rate;
    const int hop_length  = impl_->config.hop_length;

    // If the smoothing window changed since the last call, rebuild the
    // decision ring. Cheap — at most a few bytes.
    if (static_cast<int>(impl_->decisions.size()) != smoothing_window) {
        impl_->decisions.assign(
            static_cast<std::size_t>(smoothing_window), 0u);
        impl_->decision_head = 0;
        impl_->decision_len  = 0;
    }

    // Append new samples to the ring.
    if (n > 0) {
        impl_->sample_ring.insert(impl_->sample_ring.end(),
                                  samples, samples + n);
    }

    bool fired_this_call = false;

    // Drain the ring into the mel front-end one hop at a time. The front-end
    // itself buffers the first (win_length - hop_length) samples internally,
    // so we hand it whatever we have and it emits zero-or-more new frames.
    //
    // We feed the front-end in one shot (cheapest call pattern) and then run
    // each emitted mel frame through the model individually.
    bt::Tensor new_frames;  // (n_mels, T_new) on impl_->device
    const int n_to_consume = static_cast<int>(impl_->sample_ring.size());
    int T_new = 0;
    if (n_to_consume > 0) {
        T_new = impl_->mel->consume(impl_->sample_ring.data(), n_to_consume,
                                    new_frames);
        // MelFrontend::consume has consumed every sample it can — drop them
        // from our side of the ring (its own ring keeps the carry-over).
        impl_->sample_ring.clear();
    }
    (void)hop_length;
    (void)sample_rate;

    if (T_new <= 0) {
        return false;
    }

    // Pull mel frames to host once so we can slice column-wise without a
    // per-frame device round-trip (the model wants a (n_mels, 1) tensor on
    // the same device each frame).
    const int n_mels = impl_->config.n_mels;
    std::vector<float> mel_host = new_frames.to_host_vector();   // (n_mels, T_new)

    // Reused per-frame tensors.
    bt::Tensor frame_tensor = bt::Tensor::empty_on(
        impl_->device, n_mels, 1, bt::Dtype::FP32);
    std::vector<float> frame_buf(static_cast<std::size_t>(n_mels), 0.0f);
    bt::Tensor logit_tensor;

    for (int t = 0; t < T_new; ++t) {
        // Copy column t of mel_host into a (n_mels, 1) host buffer, then
        // upload to the model device. (For CPU device this is a single
        // memcpy + from_host_on, which is cheaper than a per-cell loop.)
        for (int m = 0; m < n_mels; ++m) {
            frame_buf[static_cast<std::size_t>(m)] =
                mel_host[static_cast<std::size_t>(m) * T_new + t];
        }
        frame_tensor = bt::Tensor::from_host_on(impl_->device,
                                                frame_buf.data(),
                                                n_mels, 1);

        impl_->model->forward_streaming(frame_tensor, logit_tensor);

        // logit_tensor: (1, 1) on impl_->device.
        float logit = 0.0f;
        if (logit_tensor.device == bt::Device::CPU) {
            logit = logit_tensor.host_f32()[0];
        } else {
            bt::Tensor h = logit_tensor.to(bt::Device::CPU);
            logit = h.host_f32()[0];
        }
        const float score = sigmoidf(logit);
        impl_->last_score.store(score, std::memory_order_relaxed);

        // Update the decision ring (drop oldest, append new flag).
        const bool frame_positive = (score > threshold);
        if (smoothing_window > 0) {
            impl_->decisions[static_cast<std::size_t>(impl_->decision_head)] =
                frame_positive ? 1u : 0u;
            impl_->decision_head =
                (impl_->decision_head + 1) % smoothing_window;
            if (impl_->decision_len < smoothing_window) {
                ++impl_->decision_len;
            }
        }

        // Decrement refractory after every emitted frame.
        if (impl_->refractory_frames > 0) {
            --impl_->refractory_frames;
        }

        // Count frames since reset for the warmup guard (saturate to avoid
        // overflow on a long-running stream).
        if (impl_->frames_since_reset < impl_->warmup_frames) {
            ++impl_->frames_since_reset;
        }
        const bool warmed = impl_->frames_since_reset >= impl_->warmup_frames;

        // Fire test: warmed up AND refractory clear AND >= hits true in window.
        if (warmed && impl_->refractory_frames == 0 && !fired_this_call) {
            int hits = 0;
            for (int i = 0; i < impl_->decision_len; ++i) {
                if (impl_->decisions[static_cast<std::size_t>(i)]) {
                    ++hits;
                }
            }
            if (hits >= smoothing_hits) {
                fired_this_call = true;
                // Refractory in frames: refractory_ms * sample_rate / 1000
                // samples → divide by hop_length to convert to frames.
                const int refr_samples =
                    refractory_ms * impl_->config.sample_rate / 1000;
                impl_->refractory_frames =
                    refr_samples / impl_->config.hop_length;
                if (impl_->refractory_frames < 0) {
                    impl_->refractory_frames = 0;
                }
            }
        }
    }

    return fired_this_call;
}

void WakeWord::set_threshold(float t) {
    impl_->config.threshold = t;
    impl_->a_threshold.store(t, std::memory_order_relaxed);
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
    impl_->a_smoothing_hits  .store(hits,   std::memory_order_relaxed);
    impl_->a_smoothing_window.store(window, std::memory_order_relaxed);
}

void WakeWord::set_refractory_ms(int ms) {
    if (ms < 0) {
        fail("WakeWord::set_refractory_ms",
             "negative refractory: " + std::to_string(ms));
    }
    impl_->config.refractory_ms = ms;
    impl_->a_refractory_ms.store(ms, std::memory_order_relaxed);
}

float WakeWord::last_score() const {
    return impl_->last_score.load(std::memory_order_relaxed);
}

void WakeWord::reset() {
    impl_->sample_ring.clear();
    if (impl_->mel)   impl_->mel->reset();
    if (impl_->model) impl_->model->reset_streaming_state();
    std::fill(impl_->decisions.begin(), impl_->decisions.end(),
              static_cast<std::uint8_t>(0));
    impl_->decision_head = 0;
    impl_->decision_len  = 0;
    impl_->refractory_frames = 0;
    impl_->frames_since_reset = 0;
    impl_->last_score.store(0.0f, std::memory_order_relaxed);
}

const WakeConfig& WakeWord::config() const { return impl_->config; }
bool              WakeWord::loaded() const { return impl_->loaded; }

}  // namespace brosoundml
