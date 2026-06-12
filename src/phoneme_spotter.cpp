#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "brosoundml/phoneme_spotter.h"

#include "brosoundml/mel.h"
#include "brosoundml/phoneme_model.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <array>
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
    // Per-state alternate classes (soft states): alts[j] holds up to kMaxAlts
    // runner-up class ids from the state's defining enrollment frame, -1 padded.
    // Empty vector = hard template (enroll_from_classes / enroll()), emission
    // reads the single class posterior exactly as before.
    static constexpr int kMaxAlts = 2;
    std::vector<std::array<int, kMaxAlts>> alts;
    SpotterConfig        policy;
    bool                 has_override = false;

    // DP arrays, length L+1 (index 0 = launch pad).
    std::vector<double>  S;
    std::vector<int>     N;
    // Per-phoneme coverage on the best path reaching each state. Cov[j] = count
    // of FULLY-PASSED phonemes (1..j-1) whose aligned run held >= 1 above-floor
    // frame; Cur[j] = whether the CURRENT phoneme j's run has yet seen an
    // above-floor frame. covered(j) = Cov[j] + Cur[j]. A template phoneme that is
    // only ever matched at the emission floor (the model never actually emitted
    // it) is NOT covered, so a fragment with whole phonemes absent can't fire
    // even though the floor keeps its geometric-mean confidence finite.
    std::vector<int>          Cov;
    std::vector<std::uint8_t> Cur;

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
        Cov.assign(static_cast<std::size_t>(L) + 1, 0);
        Cur.assign(static_cast<std::size_t>(L) + 1, 0u);
        refractory_frames = 0;
        furthest = 0;
        size_ring();
    }

    // Advance one frame. `lp` is the length-K per-class log-posterior buffer,
    // `p` the same frame's LINEAR posteriors (for soft-state mass sums), `pmax`
    // the frame's max posterior (linear, for competition normalization),
    // `gate_open` whether a fresh entry may begin this frame. On a completed +
    // qualified + un-refractory alignment, writes the event and returns true
    // (and resets this template's DP so it must re-enter from silence).
    bool step(const std::vector<float>& lp, const float* p, float pmax,
              bool gate_open, SpotEvent& out_ev) {
        const int L = this->L();

        // Per-frame emission floor: a template phoneme's contribution is clamped
        // to >= log(emission_floor) so an unreliable brief-phoneme posterior
        // (~0) can't multiplicatively underflow the geometric-mean confidence.
        const double log_floor = (policy.emission_floor > 0.0f)
            ? std::log(static_cast<double>(policy.emission_floor)) : kNegInf;

        // Competition normalization: subtract score_norm * log of the frame's
        // winning posterior (denominator capped below at score_norm_ref so
        // low-evidence frames aren't inflated — see SpotterConfig).
        double log_comp = 0.0;
        if (policy.score_norm > 0.0f) {
            const float ref   = policy.score_norm_ref > 0.0f
                                    ? policy.score_norm_ref : 0.5f;
            const float denom = pmax > ref ? pmax : ref;
            log_comp = static_cast<double>(policy.score_norm) *
                       std::log(static_cast<double>(denom > kLogEps ? denom
                                                                    : kLogEps));
        }

        // Launch pad for this frame.
        S[0]   = gate_open ? 0.0 : kNegInf;
        N[0]   = 0;
        Cov[0] = 0;
        Cur[0] = 0u;

        // Transitions: iterate j = L .. 1 so each candidate reads the PREVIOUS
        // frame's S[j-1]/N[j-1] (lower indices not yet touched this frame), and
        // the current-frame launch pad at j == 1.
        for (int j = L; j >= 1; --j) {
            const double s_stay = S[j];        // previous frame's S[j]
            const double s_adv  = S[j - 1];    // previous frame (or launch pad)
            const double best   = s_stay > s_adv ? s_stay : s_adv;
            if (best == kNegInf) {
                S[j]   = kNegInf;
                N[j]   = 0;
                Cov[j] = 0;
                Cur[j] = 0u;
                continue;
            }
            const bool   advance = (s_adv > s_stay);
            const int    prevN    = advance ? N[j - 1] : N[j];
            const int    cls      = classes[static_cast<std::size_t>(j - 1)];
            // Soft states: the emission is the frame's summed mass over the
            // state's class set (primary + alternates), absorbing identity
            // flips between confusable units. Hard templates (no alts) read
            // the single class log-posterior exactly as before.
            double raw_lp;
            if (alts.empty()) {
                raw_lp = static_cast<double>(lp[static_cast<std::size_t>(cls)]);
            } else {
                double mass = static_cast<double>(p[cls]);
                for (int a : alts[static_cast<std::size_t>(j - 1)])
                    if (a >= 0) mass += static_cast<double>(p[a]);
                raw_lp = std::log(mass > kLogEps ? mass : kLogEps);
            }
            // Coverage asks "was this phoneme actually emitted" — an ABSOLUTE
            // evidence question, so it tests the RAW posterior. Normalization
            // applies only to the DP score; otherwise a losing-but-nonzero
            // class (ratio above the floor) would count as covered and let a
            // template complete on frames where its phonemes never occurred.
            const bool   above    = raw_lp > log_floor;
            const double norm_emit = raw_lp - log_comp;
            const double emit     = (norm_emit < log_floor) ? log_floor : norm_emit;
            S[j] = best + emit;
            N[j] = prevN + 1;
            if (advance) {
                // Entered phoneme j fresh: bank phoneme j-1's coverage, start j.
                Cov[j] = Cov[j - 1] + (Cur[j - 1] ? 1 : 0);
                Cur[j] = above ? 1u : 0u;
            } else {
                // Stayed on phoneme j: keep its completed-phoneme count, OR in this
                // frame's coverage (a run is covered if ANY of its frames is above).
                Cur[j] = (Cur[j] || above) ? 1u : 0u;
            }
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
        const int  covered  = Cov[L] + (Cur[L] ? 1 : 0);  // phonemes actually emitted
        // Coverage requirement: absolute floor, scaled up proportionally for
        // long templates (see SpotterConfig::min_coverage_frac).
        int need_cov = policy.min_phonemes;
        if (policy.min_coverage_frac > 0.0f) {
            const int frac_cov = static_cast<int>(std::ceil(
                static_cast<double>(policy.min_coverage_frac) * L));
            if (frac_cov > need_cov) need_cov = frac_cov;
        }
        float conf = 0.0f;
        if (complete) {
            conf = static_cast<float>(std::exp(S[L] / static_cast<double>(N[L])));
            if (conf > 1.0f) conf = 1.0f;   // guard FP overshoot
        }
        const bool frame_pos = complete && (covered >= need_cov) &&
                               (conf > policy.threshold);

        // Update the smoothing ring.
        const int w = policy.smoothing_window > 0 ? policy.smoothing_window : 1;
        if (static_cast<int>(hits_ring.size()) != w) size_ring();
        hits_ring[static_cast<std::size_t>(ring_head)] = frame_pos ? 1u : 0u;
        ring_head = (ring_head + 1) % w;
        if (ring_len < w) ++ring_len;

        // Fire test. The template must be long enough (min_phonemes), fully
        // aligned (complete), and — crucially for citation templates under the
        // emission floor — at least need_cov of its phonemes must have been
        // ACTUALLY emitted (covered), so a fragment whose missing phonemes are
        // merely floored cannot complete a spurious match.
        if (L >= policy.min_phonemes && complete && covered >= need_cov &&
            refractory_frames == 0) {
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
                Cov.assign(static_cast<std::size_t>(L) + 1, 0);
                Cur.assign(static_cast<std::size_t>(L) + 1, 0u);
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
int  PhonemeSpotter::sample_rate()   const { return impl_->sample_rate; }

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

int PhonemeSpotter::enroll_from_posteriors(const std::string& name,
                                           const float* posteriors, int n_frames,
                                           const SpotterConfig* policy_override) {
    if (!impl_->has_class_map) {
        fail("PhonemeSpotter::enroll_from_posteriors",
             "no class map; call load() or set_class_map() first");
    }
    if (n_frames < 0 || (n_frames > 0 && posteriors == nullptr)) {
        fail("PhonemeSpotter::enroll_from_posteriors", "invalid posteriors");
    }
    const int K = impl_->K();
    const SpotterConfig& pol = policy_override ? *policy_override
                                               : impl_->config;
    const int n_alts = std::min(std::max(pol.enroll_alts, 0),
                                TemplateMatcher::kMaxAlts);

    // Argmax each frame; frames below the confidence gate become silence.
    // Collapse runs of equal primary class, dropping silence runs; each kept
    // run's DEFINING frame (its max-posterior frame) contributes the state's
    // alternate classes: runner-ups holding >= enroll_alt_mass, excluding
    // silence.
    std::vector<int> classes;
    std::vector<std::array<int, TemplateMatcher::kMaxAlts>> alts;
    int run_cls = -1, def_frame = -1;
    float def_conf = 0.0f;
    auto flush_run = [&] {
        if (run_cls <= 0) return;                  // silence / no run
        // Match collapse_classes(): silence is dropped BEFORE duplicate
        // collapse, so a class repeated across a silence gap is one state.
        if (!classes.empty() && classes.back() == run_cls) return;
        std::array<int, TemplateMatcher::kMaxAlts> a;
        a.fill(-1);
        if (n_alts > 0 && def_frame >= 0) {
            const float* row =
                posteriors + static_cast<std::size_t>(def_frame) * K;
            // Top n_alts classes besides the primary, by mass, gated.
            for (int slot = 0; slot < n_alts; ++slot) {
                int   best   = -1;
                float best_p = pol.enroll_alt_mass;
                for (int k = 1; k < K; ++k) {       // never silence (class 0)
                    if (k == run_cls) continue;
                    bool taken = false;
                    for (int s = 0; s < slot; ++s) taken |= (a[s] == k);
                    if (taken) continue;
                    if (row[k] >= best_p) { best = k; best_p = row[k]; }
                }
                if (best < 0) break;
                a[static_cast<std::size_t>(slot)] = best;
            }
        }
        classes.push_back(run_cls);
        alts.push_back(a);
    };
    for (int t = 0; t < n_frames; ++t) {
        const float* row = posteriors + static_cast<std::size_t>(t) * K;
        int arg = 0; float pa = row[0];
        for (int k = 1; k < K; ++k)
            if (row[k] > pa) { pa = row[k]; arg = k; }
        if (pol.enroll_conf_gate > 0.0f && pa < pol.enroll_conf_gate) arg = 0;
        if (arg != run_cls) {
            flush_run();
            run_cls = arg; def_frame = t; def_conf = pa;
        } else if (pa > def_conf) {
            def_frame = t; def_conf = pa;
        }
    }
    flush_run();

    remove(name);
    TemplateMatcher m;
    m.name         = name;
    m.classes      = classes;
    if (n_alts > 0) m.alts = std::move(alts);
    m.policy       = pol;
    m.has_override = (policy_override != nullptr);
    m.reset_dp();
    impl_->matchers.push_back(std::move(m));
    return static_cast<int>(classes.size());
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
    //
    // The reference is padded with lead/tail silence first. PCEN's smoother
    // seeds on the first frame's energy: a reference TRIMMED to the sound
    // (the common case for a file-based enroll) seeds at full energy and
    // yields a different feature stream — and therefore a different unit/class
    // path — than the same sound arriving out of ambient quiet, which is the
    // only way the live stream ever sees it. Measured on real recordings
    // (trimmed click/whistle enrolls): unpadded templates were churn that
    // neither matched their own sound in context nor stayed quiet on speech.
    // The lead also lets the matcher's entry-silence gate see a boundary when
    // the template is replayed for verification. Tail pad flushes the final
    // frames through the model's receptive field.
    const int pad = impl_->mel->config().sample_rate * 2 / 5;   // 0.4 s
    std::vector<float> padded(static_cast<std::size_t>(pad) * 2 +
                                  static_cast<std::size_t>(std::max(0, n)),
                              0.0f);
    if (n > 0) std::copy(samples, samples + n, padded.begin() + pad);

    impl_->mel->reset();
    impl_->model->reset_streaming_state();

    bt::Tensor frames;   // (n_mels, T)
    const int T = impl_->mel->consume(padded.data(),
                                      static_cast<int>(padded.size()), frames);
    std::vector<float> post;   // (T, K) row-major softmax
    int T_post = 0;
    if (T > 0) {
        bt::Tensor logits;   // (T, K)
        impl_->model->forward_streaming(frames, logits);
        std::vector<float> lh = logits.to_host_vector();   // (T, K) row-major
        const int K = impl_->K();
        T_post = logits.rows;
        post.resize(static_cast<std::size_t>(T_post) * K);
        for (int t = 0; t < T_post; ++t) {
            const float* row = &lh[static_cast<std::size_t>(t) * K];
            float mx = row[0];
            for (int k = 1; k < K; ++k) mx = std::max(mx, row[k]);
            double sum = 0.0;
            float* o = &post[static_cast<std::size_t>(t) * K];
            for (int k = 0; k < K; ++k) {
                o[k] = std::exp(row[k] - mx);
                sum += o[k];
            }
            const float inv = static_cast<float>(1.0 / (sum > 0 ? sum : 1.0));
            for (int k = 0; k < K; ++k) o[k] *= inv;
        }
    }
    // Restore the live stream's clean state (the probe disturbed the caches).
    impl_->mel->reset();
    impl_->model->reset_streaming_state();

    return enroll_from_posteriors(name, post.data(), T_post, policy_override);
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
            const bool fired = (L > 0) && m.step(lp, p, pmax, gate, ev);
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
    return run_mel_frames(new_frames, T_new);
}

std::vector<SpotEvent> PhonemeSpotter::feed_mel(const float* mel, int n_frames) {
    if (!impl_->loaded) {
        fail("PhonemeSpotter::feed_mel", "no model loaded; call load() first");
    }
    if (n_frames < 0 || (n_frames > 0 && mel == nullptr)) {
        fail("PhonemeSpotter::feed_mel", "invalid mel frames");
    }
    if (n_frames == 0) return {};
    const int M = impl_->mel->config().n_mels;
    bt::Tensor frames = bt::Tensor::from_host_on(impl_->device, mel, M, n_frames);
    return run_mel_frames(frames, n_frames);
}

std::vector<SpotEvent> PhonemeSpotter::run_mel_frames(const bt::Tensor& new_frames,
                                                      int T_new) {
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

const MelConfig& PhonemeSpotter::mel_config() const {
    if (!impl_->mel) {
        fail("PhonemeSpotter::mel_config", "no model loaded; call load() first");
    }
    return impl_->mel->config();
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
