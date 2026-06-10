#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "brosoundml/phoneme_spotter.h"

#include "brosoundml/mel.h"
#include "brosoundml/phoneme_model.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
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

constexpr double kNegInf = -std::numeric_limits<double>::infinity();
constexpr float  kLogEps = 1e-8f;

// ─── TemplateMatcher ─────────────────────────────────────────────────────────
//
// One streaming monotonic-Viterbi token-passing aligner for a single enrolled
// template. State `j` (0..L) = "matched the first j template phonemes, alignment
// ending at the current frame". S[j] is the accumulated log-posterior of the
// best path reaching state j at this frame; N[j] the number of matched frames on
// that path. State 0 is the launch pad: S[0] = 0 when entry is permitted (the
// word-boundary gate is open), else -INF, so a NEW match may only BEGIN right
// after silence while a match already in progress keeps advancing regardless.
//
// All state here is owned by the single producer thread (feed*). No atomics.
struct TemplateMatcher {
    std::string          name;
    std::vector<int>     classes;   // c[0..L-1], distinct adjacent phoneme classes
    SpotterConfig        policy;
    bool                 has_override = false;

    // DP arrays, length L+1 (index 0 = launch pad).
    std::vector<double>  S;
    std::vector<int>     N;

    // M-of-N smoothing ring of per-frame "confidence > threshold" flags.
    std::vector<std::uint8_t> hits_ring;
    int                  ring_head = 0;
    int                  ring_len  = 0;

    int                  refractory_frames = 0;   // counts down once per frame
    int                  furthest = 0;            // largest j with finite S[j]

    int L() const { return static_cast<int>(classes.size()); }

    void size_ring() {
        const int w = policy.smoothing_window > 0 ? policy.smoothing_window : 1;
        hits_ring.assign(static_cast<std::size_t>(w), 0u);
        ring_head = 0;
        ring_len  = 0;
    }

    void reset_dp() {
        const int L = this->L();
        S.assign(static_cast<std::size_t>(L) + 1, kNegInf);
        N.assign(static_cast<std::size_t>(L) + 1, 0);
        refractory_frames = 0;
        furthest = 0;
        size_ring();
    }

    // Advance one frame. `lp` is the length-K per-class log-posterior buffer,
    // `gate_open` whether a fresh entry may begin this frame. On a completed +
    // qualified + un-refractory alignment, writes the event and returns true
    // (and resets this template's DP so it must re-enter from silence).
    bool step(const std::vector<float>& lp, bool gate_open, SpotEvent& out_ev) {
        const int L = this->L();

        // Launch pad for this frame.
        S[0] = gate_open ? 0.0 : kNegInf;
        N[0] = 0;

        // Transitions: iterate j = L .. 1 so each candidate reads the PREVIOUS
        // frame's S[j-1]/N[j-1] (lower indices not yet touched this frame), and
        // the current-frame launch pad at j == 1.
        for (int j = L; j >= 1; --j) {
            const double s_stay = S[j];        // previous frame's S[j]
            const double s_adv  = S[j - 1];    // previous frame (or launch pad)
            const double best   = s_stay > s_adv ? s_stay : s_adv;
            if (best == kNegInf) {
                S[j] = kNegInf;
                N[j] = 0;
                continue;
            }
            const int prevN = (s_adv > s_stay) ? N[j - 1] : N[j];
            S[j] = best + static_cast<double>(lp[static_cast<std::size_t>(classes[
                       static_cast<std::size_t>(j - 1)])]);
            N[j] = prevN + 1;
        }

        // Prefix progress: furthest finite state.
        furthest = 0;
        for (int j = L; j >= 1; --j) {
            if (S[j] != kNegInf) { furthest = j; break; }
        }

        // Refractory ticks down every frame (matches WakeWord's per-frame decay).
        if (refractory_frames > 0) --refractory_frames;

        // Completion confidence: geometric-mean posterior over the matched span.
        const bool complete = (S[L] != kNegInf) && (N[L] > 0);
        float conf = 0.0f;
        if (complete) {
            conf = static_cast<float>(std::exp(S[L] / static_cast<double>(N[L])));
            if (conf > 1.0f) conf = 1.0f;   // guard FP overshoot
        }
        const bool frame_pos = complete && (conf > policy.threshold);

        // Update the smoothing ring.
        const int w = policy.smoothing_window > 0 ? policy.smoothing_window : 1;
        if (static_cast<int>(hits_ring.size()) != w) size_ring();
        hits_ring[static_cast<std::size_t>(ring_head)] = frame_pos ? 1u : 0u;
        ring_head = (ring_head + 1) % w;
        if (ring_len < w) ++ring_len;

        // Fire test.
        if (L >= policy.min_phonemes && complete && refractory_frames == 0) {
            int hits = 0;
            for (int i = 0; i < ring_len; ++i)
                if (hits_ring[static_cast<std::size_t>(i)]) ++hits;
            if (hits >= policy.smoothing_hits) {
                out_ev.name             = name;
                out_ev.confidence       = conf;
                out_ev.matched_phonemes = L;   // full match
                out_ev.template_len     = L;
                // Re-enter from silence for the next utterance: clear the DP,
                // the smoothing ring, and arm the refractory window.
                S.assign(static_cast<std::size_t>(L) + 1, kNegInf);
                N.assign(static_cast<std::size_t>(L) + 1, 0);
                furthest = 0;
                size_ring();            // drop stale positive flags
                refractory_frames = 0;  // set below from ms→frames by the owner
                return true;
            }
        }
        return false;
    }
};

}  // namespace

// ─── PhonemeSpotter::Impl ────────────────────────────────────────────────────
struct PhonemeSpotter::Impl {
    SpotterConfig   config;        // global defaults
    PhonemeClassMap class_map;
    bool            has_class_map = false;

    // Model + front-end (REAL path only).
    std::unique_ptr<PhonemeNet>  model;
    std::unique_ptr<MelFrontend> mel;
    bt::Device                   device = bt::Device::CPU;
    bool                         loaded = false;

    // Framing params used to convert refractory_ms -> frames. Defaults match the
    // PhonemeNet front-end (16 kHz, 10 ms hop); load() overwrites from the model.
    int sample_rate = 16000;
    int hop_length  = 160;

    // Enrolled templates (single-producer ownership).
    std::vector<TemplateMatcher> matchers;

    // Sample-side carry for the REAL feed() path.
    std::vector<float> sample_ring;

    // Recent consecutive-silence run length (for the entry gate). Initialised
    // "wide open" so the first word after a reset can match.
    int silence_run = 1 << 20;

    // ── Cross-thread telemetry (lock-free) ──
    std::atomic<float> a_prefix_progress{0.0f};

    // last_posterior seqlock: K is fixed once the class map is set, so the buffer
    // never reallocates and a seqlock-guarded copy is race-free without a lock.
    std::vector<float>          last_post;     // size K (0 until first frame)
    std::atomic<std::uint32_t>  post_seq{0};   // even = stable, odd = writing
    bool                        post_valid = false;

    int K() const { return class_map.num_classes; }

    int refractory_frames_for(const SpotterConfig& p) const {
        const int hop = hop_length > 0 ? hop_length : 1;
        long long s = static_cast<long long>(p.refractory_ms) *
                      static_cast<long long>(sample_rate) / 1000;
        long long f = s / hop;
        if (f < 0) f = 0;
        return static_cast<int>(f);
    }

    void publish_posterior(const float* p, int K) {
        const std::uint32_t s0 = post_seq.load(std::memory_order_relaxed);
        post_seq.store(s0 + 1, std::memory_order_release);   // odd: writing
        for (int k = 0; k < K; ++k)
            last_post[static_cast<std::size_t>(k)] = p[k];
        post_valid = true;
        post_seq.store(s0 + 2, std::memory_order_release);   // even: stable
    }

    void reset_stream_state() {
        if (mel)   mel->reset();
        if (model) model->reset_streaming_state();
        sample_ring.clear();
        silence_run = 1 << 20;   // post-reset counts as silence seen
        for (auto& m : matchers) m.reset_dp();
        a_prefix_progress.store(0.0f, std::memory_order_relaxed);
        // Invalidate the posterior snapshot.
        const std::uint32_t s0 = post_seq.load(std::memory_order_relaxed);
        post_seq.store(s0 + 1, std::memory_order_release);
        post_valid = false;
        post_seq.store(s0 + 2, std::memory_order_release);
    }
};

PhonemeSpotter::PhonemeSpotter() : impl_(std::make_unique<Impl>()) {}
PhonemeSpotter::~PhonemeSpotter() = default;
PhonemeSpotter::PhonemeSpotter(PhonemeSpotter&&) noexcept = default;
PhonemeSpotter& PhonemeSpotter::operator=(PhonemeSpotter&&) noexcept = default;

// ─── Class map / model install ──────────────────────────────────────────────

void PhonemeSpotter::set_class_map(const PhonemeClassMap& cm) {
    if (cm.num_classes <= 0) {
        fail("PhonemeSpotter::set_class_map",
             "class map has non-positive num_classes");
    }
    cm.rebuild_inverse();
    impl_->class_map     = cm;
    impl_->has_class_map = true;
    impl_->model.reset();
    impl_->mel.reset();
    impl_->loaded = false;
    impl_->matchers.clear();
    impl_->last_post.assign(static_cast<std::size_t>(cm.num_classes), 0.0f);
    impl_->post_valid = false;
    impl_->reset_stream_state();
}

void PhonemeSpotter::load(const std::string& weights_path, brotensor::Device device) {
    bt::init();

    auto model = std::make_unique<PhonemeNet>(PhonemeNet::load(weights_path, device));
    const PhonemeClassMap cm  = model->class_map();
    const PhonemeNetConfig& c = model->config();
    if (cm.num_classes <= 0) {
        fail("PhonemeSpotter::load", "model class map has non-positive num_classes");
    }
    cm.rebuild_inverse();

    // Build the PCEN mel front-end matching the model's framing (the recipe is
    // trained on PCEN — mirror WakeWord::load).
    MelConfig mcfg;
    mcfg.sample_rate = c.sample_rate;
    mcfg.n_fft       = c.n_fft;
    mcfg.win_length  = c.win_length;
    mcfg.hop_length  = c.hop_length;
    mcfg.n_mels      = c.n_mels;
    mcfg.compression = MelCompression::PCEN;
    auto mel = std::make_unique<MelFrontend>(mcfg, device);

    // Commit.
    impl_->model         = std::move(model);
    impl_->mel           = std::move(mel);
    impl_->device        = device;
    impl_->loaded        = true;
    impl_->class_map     = cm;
    impl_->has_class_map = true;
    impl_->sample_rate   = c.sample_rate;
    impl_->hop_length    = c.hop_length;
    impl_->matchers.clear();
    impl_->last_post.assign(static_cast<std::size_t>(cm.num_classes), 0.0f);
    impl_->post_valid = false;
    impl_->reset_stream_state();
}

const PhonemeClassMap& PhonemeSpotter::class_map() const { return impl_->class_map; }
bool PhonemeSpotter::loaded()        const { return impl_->loaded; }
bool PhonemeSpotter::has_class_map() const { return impl_->has_class_map; }

// ─── Enrollment helpers ──────────────────────────────────────────────────────

namespace {

// Collapse a class-id sequence into a template: drop the silence class (0),
// collapse consecutive duplicate classes.
std::vector<int> collapse_classes(const std::vector<int>& class_ids) {
    std::vector<int> out;
    out.reserve(class_ids.size());
    for (int c : class_ids) {
        if (c == 0) continue;                 // silence class — drop
        if (!out.empty() && out.back() == c) continue;  // collapse run
        out.push_back(c);
    }
    return out;
}

}  // namespace

int PhonemeSpotter::enroll_from_classes(const std::string& name,
                                        const std::vector<int>& class_ids,
                                        const SpotterConfig* policy_override) {
    if (!impl_->has_class_map) {
        fail("PhonemeSpotter::enroll_from_classes",
             "no class map; call load() or set_class_map() first");
    }
    const int K = impl_->K();
    for (int c : class_ids) {
        if (c < 0 || c >= K) {
            fail("PhonemeSpotter::enroll_from_classes",
                 "class id " + std::to_string(c) + " out of range [0," +
                 std::to_string(K) + ")");
        }
    }
    std::vector<int> tmpl = collapse_classes(class_ids);

    // Replace any existing template with this name.
    remove(name);

    TemplateMatcher m;
    m.name         = name;
    m.classes      = tmpl;
    m.policy       = policy_override ? *policy_override : impl_->config;
    m.has_override = (policy_override != nullptr);
    m.reset_dp();
    impl_->matchers.push_back(std::move(m));
    return static_cast<int>(tmpl.size());
}

int PhonemeSpotter::enroll(const std::string& name,
                           const std::vector<int>& phoneme_ids,
                           const SpotterConfig* policy_override) {
    if (!impl_->has_class_map) {
        fail("PhonemeSpotter::enroll",
             "no class map; call load() or set_class_map() first");
    }
    const PhonemeClassMap& cm = impl_->class_map;
    std::vector<int> class_ids;
    class_ids.reserve(phoneme_ids.size());
    for (int id : phoneme_ids) {
        if (cm.is_transparent(id)) continue;     // suprasegmental modifier — drop
        const int cls = cm.class_for_id(id);     // throws if unmapped (a bug)
        class_ids.push_back(cls);                // silence drop happens in collapse
    }
    return enroll_from_classes(name, class_ids, policy_override);
}

int PhonemeSpotter::enroll_from_audio(const std::string& name,
                                      const float* samples, int n,
                                      const SpotterConfig* policy_override) {
    if (!impl_->loaded) {
        fail("PhonemeSpotter::enroll_from_audio",
             "no model loaded; call load() first");
    }
    if (n < 0 || (n > 0 && samples == nullptr)) {
        fail("PhonemeSpotter::enroll_from_audio", "invalid samples");
    }

    // Run the reference audio through the front-end + model offline (a fresh
    // streaming pass, isolated from the live stream's caches).
    impl_->mel->reset();
    impl_->model->reset_streaming_state();

    bt::Tensor frames;   // (n_mels, T)
    const int T = impl_->mel->consume(samples, n, frames);
    std::vector<int> class_ids;
    if (T > 0) {
        bt::Tensor logits;   // (T, K)
        impl_->model->forward_streaming(frames, logits);
        std::vector<float> lh = logits.to_host_vector();   // (T, K) row-major
        const int K = impl_->K();
        class_ids.reserve(static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t) {
            int arg = 0; float best = lh[static_cast<std::size_t>(t) * K];
            for (int k = 1; k < K; ++k) {
                const float v = lh[static_cast<std::size_t>(t) * K + k];
                if (v > best) { best = v; arg = k; }
            }
            class_ids.push_back(arg);
        }
    }
    // Restore the live stream's clean state (the probe disturbed the caches).
    impl_->mel->reset();
    impl_->model->reset_streaming_state();

    return enroll_from_classes(name, class_ids, policy_override);
}

bool PhonemeSpotter::remove(const std::string& name) {
    auto& v = impl_->matchers;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i].name == name) {
            v.erase(v.begin() + static_cast<std::ptrdiff_t>(i));
            return true;
        }
    }
    return false;
}

void PhonemeSpotter::clear() { impl_->matchers.clear(); }

std::vector<std::string> PhonemeSpotter::templates() const {
    std::vector<std::string> out;
    out.reserve(impl_->matchers.size());
    for (auto& m : impl_->matchers) out.push_back(m.name);
    return out;
}

// ─── Streaming core ──────────────────────────────────────────────────────────

std::vector<SpotEvent> PhonemeSpotter::feed_posteriors(const float* posteriors,
                                                       int n_frames) {
    if (!impl_->has_class_map) {
        fail("PhonemeSpotter::feed_posteriors",
             "no class map; call load() or set_class_map() first");
    }
    if (n_frames < 0 || (n_frames > 0 && posteriors == nullptr)) {
        fail("PhonemeSpotter::feed_posteriors", "invalid posteriors");
    }
    const int K = impl_->K();

    std::vector<SpotEvent> events;
    std::vector<float> lp(static_cast<std::size_t>(K), 0.0f);

    for (int t = 0; t < n_frames; ++t) {
        const float* p = posteriors + static_cast<std::size_t>(t) * K;

        // Per-frame log-posteriors and the silence/argmax for the entry gate.
        int arg = 0; float pmax = p[0];
        for (int k = 0; k < K; ++k) {
            const float pk = p[k];
            lp[static_cast<std::size_t>(k)] =
                std::log(pk > kLogEps ? pk : kLogEps);
            if (pk > pmax) { pmax = pk; arg = k; }
        }
        const bool frame_silence = (arg == 0);

        // Publish telemetry (seqlock) before mutating matcher state.
        impl_->publish_posterior(p, K);

        // Drive every template's matcher.
        int best_prefix_num = 0, best_prefix_den = 1;
        for (auto& m : impl_->matchers) {
            const int L = m.L();
            const bool gate = impl_->silence_run >= m.policy.entry_silence_frames;
            SpotEvent ev;
            const bool fired = (L > 0) && m.step(lp, gate, ev);
            if (fired) {
                m.refractory_frames = impl_->refractory_frames_for(m.policy);
                events.push_back(ev);
            }
            // Prefix progress contribution (post-step furthest state).
            if (L > 0) {
                // furthest/L; pick the max fraction across templates.
                if (static_cast<long long>(m.furthest) * best_prefix_den >
                    static_cast<long long>(best_prefix_num) * L) {
                    best_prefix_num = m.furthest;
                    best_prefix_den = L;
                }
            }
        }
        impl_->a_prefix_progress.store(
            best_prefix_den > 0
                ? static_cast<float>(best_prefix_num) / static_cast<float>(best_prefix_den)
                : 0.0f,
            std::memory_order_relaxed);

        // Update the silence run AFTER this frame is processed (so the gate sees
        // only frames strictly before the current one).
        if (frame_silence) {
            if (impl_->silence_run < (1 << 20)) ++impl_->silence_run;
        } else {
            impl_->silence_run = 0;
        }
    }
    return events;
}

std::vector<SpotEvent> PhonemeSpotter::feed(const float* samples, int n) {
    if (!impl_->loaded) {
        fail("PhonemeSpotter::feed", "no model loaded; call load() first");
    }
    if (n < 0 || (n > 0 && samples == nullptr)) {
        fail("PhonemeSpotter::feed", "invalid samples");
    }
    if (n > 0) {
        impl_->sample_ring.insert(impl_->sample_ring.end(), samples, samples + n);
    }

    bt::Tensor new_frames;   // (n_mels, T_new)
    int T_new = 0;
    if (!impl_->sample_ring.empty()) {
        T_new = impl_->mel->consume(impl_->sample_ring.data(),
                                    static_cast<int>(impl_->sample_ring.size()),
                                    new_frames);
        impl_->sample_ring.clear();
    }
    if (T_new <= 0) return {};

    // forward_streaming over the whole new mel block -> (T_new, K) logits.
    bt::Tensor logits;
    impl_->model->forward_streaming(new_frames, logits);
    std::vector<float> lh = logits.to_host_vector();   // (T_new, K) row-major
    const int K = impl_->K();

    // Per-frame softmax to posteriors (host, matching WakeWord's host-side
    // sigmoid — a length-K reduction per frame, no device round-trip).
    std::vector<float> post(static_cast<std::size_t>(T_new) * K, 0.0f);
    for (int t = 0; t < T_new; ++t) {
        const float* row = lh.data() + static_cast<std::size_t>(t) * K;
        float mx = row[0];
        for (int k = 1; k < K; ++k) if (row[k] > mx) mx = row[k];
        double sum = 0.0;
        float* pr = post.data() + static_cast<std::size_t>(t) * K;
        for (int k = 0; k < K; ++k) {
            const float e = std::exp(row[k] - mx);
            pr[k] = e; sum += e;
        }
        const float inv = static_cast<float>(1.0 / (sum > 0.0 ? sum : 1.0));
        for (int k = 0; k < K; ++k) pr[k] *= inv;
    }

    return feed_posteriors(post.data(), T_new);
}

// ─── Telemetry readers ───────────────────────────────────────────────────────

std::vector<float> PhonemeSpotter::last_posterior() const {
    const int K = impl_->K();
    if (K <= 0) return {};
    for (;;) {
        const std::uint32_t s1 = impl_->post_seq.load(std::memory_order_acquire);
        if (s1 & 1u) continue;                       // a write is in flight
        if (!impl_->post_valid) {
            const std::uint32_t s2 =
                impl_->post_seq.load(std::memory_order_acquire);
            if (s1 == s2) return {};                 // no frame yet
            continue;
        }
        std::vector<float> out(impl_->last_post.begin(), impl_->last_post.end());
        const std::uint32_t s2 = impl_->post_seq.load(std::memory_order_acquire);
        if (s1 == s2) return out;                    // stable snapshot
    }
}

float PhonemeSpotter::prefix_progress() const {
    return impl_->a_prefix_progress.load(std::memory_order_relaxed);
}

void PhonemeSpotter::reset() { impl_->reset_stream_state(); }

const SpotterConfig& PhonemeSpotter::config() const { return impl_->config; }

void PhonemeSpotter::set_config(const SpotterConfig& cfg) {
    impl_->config = cfg;
    // Re-apply to templates enrolled WITHOUT an override; rebuild their smoothing
    // ring to the new window but keep their DP/refractory progress.
    for (auto& m : impl_->matchers) {
        if (!m.has_override) {
            m.policy = cfg;
            if (static_cast<int>(m.hits_ring.size()) !=
                (cfg.smoothing_window > 0 ? cfg.smoothing_window : 1)) {
                m.size_ring();
            }
        }
    }
}

}  // namespace brosoundml
