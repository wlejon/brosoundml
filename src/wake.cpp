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

    // Streaming state. The net is SHARED and read-only — weights live once
    // behind the shared_ptr and may back N detectors; this detector owns only
    // `stream`, its private streaming session over that net.
    std::unique_ptr<MelFrontend>      mel;     // built on load()
    std::shared_ptr<const BcResnet2d> model;   // shared, loaded on load()
    BcResnet2dStreamState             stream;  // this detector's own session
    bt::Device                        device = bt::Device::CPU;
    MelConfig                    mcfg;      // the front-end framing, kept for
                                            // shared-front-end compatibility
                                            // checks (mel_config())

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

    // Reused per-frame tensors/buffers (resident on the model device, sized
    // on load()) — feed paths do no per-frame heap traffic through these.
    bt::Tensor         frame_tensor;
    bt::Tensor         logit_tensor;
    std::vector<float> frame_buf;

    // One PCEN mel column through the streaming model + the detector policy.
    // `block` is host FP32 (n_mels, T) row-major (MelFrontend's emit layout);
    // column `t` is sliced into frame_buf. The shared tail of feed() and
    // feed_mel(). `allow_fire` carries the caller's at-most-one-fire-per-call
    // rule: when false the decision ring / refractory / warmup still advance,
    // but the fire test is skipped (exactly the old in-loop behavior).
    bool step_mel_column(const float* block, int T, int t,
                         float threshold, int smoothing_hits,
                         int smoothing_window, int refractory_ms,
                         bool allow_fire);

    // Install a shared, loaded net: adopt its framing, build the matching PCEN
    // front-end, allocate this detector's private streaming session, and reset
    // the per-frame bookkeeping. Shared by load() and the shared-net ctor.
    void install_net(std::shared_ptr<const BcResnet2d> net) {
        const int model_n_mels = net->config().n_mels;
        if (model_n_mels <= 0)
            fail("WakeWord::install",
                 "model header reports non-positive n_mels=" +
                 std::to_string(model_n_mels));
        // Trust the model header; overwrite ours if it disagrees.
        if (config.n_mels != model_n_mels) config.n_mels = model_n_mels;

        // PCEN front-end matching the model's framing (the 2D recipe is trained
        // on PCEN — the runtime front-end must match).
        MelConfig m;
        m.sample_rate = config.sample_rate;
        m.n_fft       = config.n_fft;
        m.win_length  = config.win_length;
        m.hop_length  = config.hop_length;
        m.n_mels      = config.n_mels;
        m.fmin        = config.mel_fmin;
        m.fmax        = config.mel_fmax;
        m.compression = MelCompression::PCEN;
        const bt::Device dev = net->device();
        auto melfe = std::make_unique<MelFrontend>(m, dev);

        const int warmup = net->gap_window_frames();   // read before the move

        // Commit.
        mel    = std::move(melfe);
        model  = std::move(net);
        stream = model->make_stream_state();
        device = dev;
        mcfg   = m;
        loaded = true;
        warmup_frames      = warmup > 0 ? warmup : 0;
        frames_since_reset = 0;
        frame_buf.assign(static_cast<std::size_t>(config.n_mels), 0.0f);

        a_threshold       .store(config.threshold,        std::memory_order_relaxed);
        a_smoothing_hits  .store(config.smoothing_hits,   std::memory_order_relaxed);
        a_smoothing_window.store(config.smoothing_window, std::memory_order_relaxed);
        a_refractory_ms   .store(config.refractory_ms,    std::memory_order_relaxed);

        decisions.assign(static_cast<std::size_t>(config.smoothing_window), 0u);
        decision_head = 0;
        decision_len  = 0;
        refractory_frames = 0;
        sample_ring.clear();
        last_score.store(0.0f, std::memory_order_relaxed);
    }
};

bool WakeWord::Impl::step_mel_column(const float* block, int T, int t,
                                     float threshold, int smoothing_hits,
                                     int smoothing_window, int refractory_ms,
                                     bool allow_fire) {
    const int n_mels = config.n_mels;
    for (int m = 0; m < n_mels; ++m) {
        frame_buf[static_cast<std::size_t>(m)] =
            block[static_cast<std::size_t>(m) * T + t];
    }
    frame_tensor = bt::Tensor::from_host_on(device, frame_buf.data(),
                                            n_mels, 1);

    model->forward_streaming(stream, frame_tensor, logit_tensor);

    // logit_tensor: (1, 1) on `device`.
    float logit = 0.0f;
    if (logit_tensor.device == bt::Device::CPU) {
        logit = logit_tensor.host_f32()[0];
    } else {
        bt::Tensor h = logit_tensor.to(bt::Device::CPU);
        logit = h.host_f32()[0];
    }
    const float score = sigmoidf(logit);
    last_score.store(score, std::memory_order_relaxed);

    // Update the decision ring (drop oldest, append new flag).
    const bool frame_positive = (score > threshold);
    if (smoothing_window > 0) {
        decisions[static_cast<std::size_t>(decision_head)] =
            frame_positive ? 1u : 0u;
        decision_head = (decision_head + 1) % smoothing_window;
        if (decision_len < smoothing_window) {
            ++decision_len;
        }
    }

    // Decrement refractory after every emitted frame.
    if (refractory_frames > 0) {
        --refractory_frames;
    }

    // Count frames since reset for the warmup guard (saturate to avoid
    // overflow on a long-running stream).
    if (frames_since_reset < warmup_frames) {
        ++frames_since_reset;
    }
    const bool warmed = frames_since_reset >= warmup_frames;

    // Fire test: warmed up AND refractory clear AND >= hits true in window.
    if (warmed && refractory_frames == 0 && allow_fire) {
        int hits = 0;
        for (int i = 0; i < decision_len; ++i) {
            if (decisions[static_cast<std::size_t>(i)]) {
                ++hits;
            }
        }
        if (hits >= smoothing_hits) {
            // Refractory in frames: refractory_ms * sample_rate / 1000
            // samples → divide by hop_length to convert to frames.
            const int refr_samples =
                refractory_ms * config.sample_rate / 1000;
            refractory_frames = refr_samples / config.hop_length;
            if (refractory_frames < 0) {
                refractory_frames = 0;
            }
            return true;
        }
    }
    return false;
}

WakeWord::WakeWord() : impl_(std::make_unique<Impl>()) {}

WakeWord::WakeWord(std::shared_ptr<const BcResnet2d> net)
    : impl_(std::make_unique<Impl>()) {
    if (!net) fail("WakeWord", "null net");
    brotensor::init();
    impl_->install_net(std::move(net));
}

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

    // Load the 2D BC-ResNet (throws on bad-version / shape mismatch) and adopt
    // it as a shared, read-only net. install_net builds the matching front-end,
    // allocates this detector's session, and mirrors policy into the atomics.
    auto net = std::make_shared<const BcResnet2d>(
        BcResnet2d::load(weights_path, device));
    impl_->install_net(std::move(net));
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
    std::vector<float> mel_host = new_frames.to_host_vector();   // (n_mels, T_new)

    for (int t = 0; t < T_new; ++t) {
        if (impl_->step_mel_column(mel_host.data(), T_new, t, threshold,
                                   smoothing_hits, smoothing_window,
                                   refractory_ms, !fired_this_call)) {
            fired_this_call = true;
        }
    }

    return fired_this_call;
}

bool WakeWord::feed_mel(const float* mel, int n_frames) {
    if (!impl_->loaded) {
        fail("WakeWord::feed_mel",
             "no model loaded; call WakeWord::load() first");
    }
    if (n_frames < 0) {
        fail("WakeWord::feed_mel",
             "negative frame count " + std::to_string(n_frames));
    }
    if (n_frames > 0 && mel == nullptr) {
        fail("WakeWord::feed_mel", "null frames with n_frames > 0");
    }
    if (n_frames == 0) return false;

    // Snapshot policy for the duration of this call (same contract as feed()).
    const float threshold        = impl_->a_threshold       .load(std::memory_order_relaxed);
    const int   smoothing_hits   = impl_->a_smoothing_hits  .load(std::memory_order_relaxed);
    const int   smoothing_window = impl_->a_smoothing_window.load(std::memory_order_relaxed);
    const int   refractory_ms    = impl_->a_refractory_ms   .load(std::memory_order_relaxed);

    if (static_cast<int>(impl_->decisions.size()) != smoothing_window) {
        impl_->decisions.assign(
            static_cast<std::size_t>(smoothing_window), 0u);
        impl_->decision_head = 0;
        impl_->decision_len  = 0;
    }

    bool fired_this_call = false;
    for (int t = 0; t < n_frames; ++t) {
        if (impl_->step_mel_column(mel, n_frames, t, threshold,
                                   smoothing_hits, smoothing_window,
                                   refractory_ms, !fired_this_call)) {
            fired_this_call = true;
        }
    }
    return fired_this_call;
}

const MelConfig& WakeWord::mel_config() const {
    if (!impl_->loaded) {
        fail("WakeWord::mel_config",
             "no model loaded; call WakeWord::load() first");
    }
    return impl_->mcfg;
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
    if (impl_->model) impl_->model->reset(impl_->stream);
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
