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
#include <cstring>
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

}  // namespace

} // namespace brosoundml
#include "phoneme_spotter_matcher.h"
namespace brosoundml {

namespace {
using detail::kNegInf;
using detail::kLogEps;
using detail::TemplateMatcher;
}  // namespace

// ─── PhonemeSpotter::Impl ────────────────────────────────────────────────────
struct PhonemeSpotter::Impl {
    SpotterConfig   config;        // global defaults
    PhonemeClassMap class_map;
    bool            has_class_map = false;

    // Model + front-end (REAL path only). The net is SHARED and read-only —
    // weights live once behind the shared_ptr and may back N spotters; this
    // spotter owns only `stream`, its private streaming session over that net.
    std::shared_ptr<const PhonemeNet> model;
    PhonemeSession                stream;
    std::unique_ptr<MelFrontend>      mel;
    bt::Device                        device = bt::Device::CPU;
    bool                              loaded = false;

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

    // Per-template progress seqlock (fixed-size POD, mirrors SensorHub's
    // publish). Written once per posterior frame and on template-set changes.
    std::int64_t                frames_total = 0;     // monotonic, survives reset()
    std::uint32_t               tmpl_generation = 0;  // bumps on set changes
    ProgressSnapshot            prog_pub;
    std::atomic<std::uint32_t>  prog_seq{0};   // even = stable, odd = writing

    int K() const { return class_map.num_classes; }

    int refractory_frames_for(const SpotterConfig& p) const {
        const int hop = hop_length > 0 ? hop_length : 1;
        long long s = static_cast<long long>(p.refractory_ms) *
                      static_cast<long long>(sample_rate) / 1000;
        long long f = s / hop;
        if (f < 0) f = 0;
        return static_cast<int>(f);
    }

    // Rebuild + publish the progress snapshot from the matchers' current
    // alignment state. Producer thread only (frame loop + template-set
    // mutations), so reading the matchers here is race-free.
    void publish_progress(const std::vector<TemplateMatcher>& ms) {
        const std::uint32_t s0 = prog_seq.load(std::memory_order_relaxed);
        prog_seq.store(s0 + 1, std::memory_order_release);   // odd: writing
        prog_pub.frames     = frames_total;
        prog_pub.generation = tmpl_generation;
        int n = static_cast<int>(ms.size());
        if (n > ProgressSnapshot::kMaxTemplates)
            n = ProgressSnapshot::kMaxTemplates;
        prog_pub.count = n;
        for (int i = 0; i < n; ++i) {
            const TemplateMatcher& m = ms[static_cast<std::size_t>(i)];
            TemplateProgress& e = prog_pub.templates[i];
            const std::size_t len =
                std::min(m.name.size(), sizeof(e.name) - 1);
            std::memcpy(e.name, m.name.data(), len);
            e.name[len] = '\0';
            const int L = m.L();
            const int f = m.furthest;
            e.matched  = f;
            e.length   = L;
            e.progress = L > 0 ? static_cast<float>(f) / static_cast<float>(L)
                               : 0.0f;
            // Partial confidence: geometric-mean posterior over the matched
            // prefix — exp(S[f]/N[f]), the same statistic the completion
            // threshold tests at f == L.
            float conf = 0.0f;
            if (f > 0 && f < static_cast<int>(m.S.size()) &&
                m.S[static_cast<std::size_t>(f)] != kNegInf &&
                m.N[static_cast<std::size_t>(f)] > 0) {
                conf = static_cast<float>(
                    std::exp(m.S[static_cast<std::size_t>(f)] /
                             static_cast<double>(m.N[static_cast<std::size_t>(f)])));
                if (conf > 1.0f) conf = 1.0f;
            }
            e.confidence         = conf;
            e.completions        = m.completions;
            e.last_advance_frame = m.last_advance_frame;
            e.last_fire_frame    = m.last_fire_frame;
        }
        prog_seq.store(s0 + 2, std::memory_order_release);   // even: stable
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
        if (model) model->reset(stream);
        sample_ring.clear();
        silence_run = 1 << 20;   // post-reset counts as silence seen
        for (auto& m : matchers) m.reset_dp();
        a_prefix_progress.store(0.0f, std::memory_order_relaxed);
        // Invalidate the posterior snapshot.
        const std::uint32_t s0 = post_seq.load(std::memory_order_relaxed);
        post_seq.store(s0 + 1, std::memory_order_release);
        post_valid = false;
        post_seq.store(s0 + 2, std::memory_order_release);
        // Re-publish progress so readers see the cleared alignments. The
        // monotonic fields (frames, completions, last_*) deliberately survive.
        publish_progress(matchers);
    }

    // Install a shared, loaded net: build the matching PCEN front-end, allocate
    // this spotter's private streaming session, and adopt the net's class map +
    // framing. Shared by load() and the shared-net constructor.
    void install_net(std::shared_ptr<const PhonemeNet> net) {
        const PhonemeClassMap   cm = net->class_map();
        const PhonemeNetConfig& c  = net->config();
        if (cm.num_classes <= 0)
            fail("PhonemeSpotter::install",
                 "model class map has non-positive num_classes");
        cm.rebuild_inverse();
        const bt::Device dev = net->device();

        // PCEN mel front-end matching the model's framing (mirror WakeWord).
        MelConfig mcfg;
        mcfg.sample_rate = c.sample_rate;
        mcfg.n_fft       = c.n_fft;
        mcfg.win_length  = c.win_length;
        mcfg.hop_length  = c.hop_length;
        mcfg.n_mels      = c.n_mels;
        mcfg.compression = MelCompression::PCEN;
        auto melfe = std::make_unique<MelFrontend>(mcfg, dev);

        // Commit.
        model         = std::move(net);
        stream        = model->make_session();
        mel           = std::move(melfe);
        device        = dev;
        loaded        = true;
        class_map     = cm;
        has_class_map = true;
        sample_rate   = c.sample_rate;
        hop_length    = c.hop_length;
        matchers.clear();
        ++tmpl_generation;
        last_post.assign(static_cast<std::size_t>(cm.num_classes), 0.0f);
        post_valid = false;
        reset_stream_state();   // publishes the emptied progress snapshot
    }
};

PhonemeSpotter::PhonemeSpotter() : impl_(std::make_unique<Impl>()) {}

PhonemeSpotter::PhonemeSpotter(std::shared_ptr<const PhonemeNet> net)
    : impl_(std::make_unique<Impl>()) {
    if (!net) fail("PhonemeSpotter", "null net");
    bt::init();
    impl_->install_net(std::move(net));
}

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
    ++impl_->tmpl_generation;
    impl_->last_post.assign(static_cast<std::size_t>(cm.num_classes), 0.0f);
    impl_->post_valid = false;
    impl_->reset_stream_state();   // publishes the emptied progress snapshot
}

void PhonemeSpotter::load(const std::string& weights_path, brotensor::Device device) {
    bt::init();
    auto net = std::make_shared<const PhonemeNet>(
        PhonemeNet::load(weights_path, device));
    impl_->install_net(std::move(net));
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
    ++impl_->tmpl_generation;
    impl_->publish_progress(impl_->matchers);
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
    // silence. With enroll_gaps, internal silence runs >= gap_min_frames
    // become TIMED gap states instead of dropping (rhythm templates — see
    // SpotterConfig::enroll_gaps); note the conf gate folds inter-sound unit
    // churn into the measured gap, which is exactly where it belongs.
    std::vector<int> classes;
    std::vector<int> tmpl_gap_lo, tmpl_gap_hi;   // parallel to classes
    std::vector<std::array<int, TemplateMatcher::kMaxAlts>> alts;
    const bool  gaps_on = pol.enroll_gaps;
    const float gtol    = std::max(0.0f, pol.gap_tolerance);
    int run_cls = -1, run_len = 0, def_frame = -1;
    float def_conf = 0.0f;
    int pending_gap = 0;   // internal-silence frames awaiting the next sound run
    auto flush_run = [&] {
        if (run_cls < 0) return;                   // no run yet
        if (run_cls == 0) {
            // Silence run. With gaps on, bank INTERNAL silence for the next
            // sound run (leading silence has no preceding sound; trailing
            // silence never sees a following sound, so it dies unbanked).
            if (gaps_on && !classes.empty()) pending_gap += run_len;
            return;
        }
        // Enough banked silence before this sound run -> a timed gap state.
        const bool real_gap =
            gaps_on && pending_gap >= std::max(1, pol.gap_min_frames);
        if (real_gap) {
            const int lo = std::max(1, static_cast<int>(std::floor(
                static_cast<double>(pending_gap) * (1.0 - gtol))));
            const int hi = std::max(lo, static_cast<int>(std::ceil(
                static_cast<double>(pending_gap) * (1.0 + gtol))));
            classes.push_back(0);
            tmpl_gap_lo.push_back(lo);
            tmpl_gap_hi.push_back(hi);
            std::array<int, TemplateMatcher::kMaxAlts> ga;
            ga.fill(-1);
            alts.push_back(ga);
        }
        pending_gap = 0;
        // Match collapse_classes(): silence is dropped BEFORE duplicate
        // collapse, so a class repeated across a (sub-threshold) silence gap
        // is one state. A materialized gap breaks the adjacency, so the
        // repeat is a real second state.
        if (!real_gap && !classes.empty() && classes.back() == run_cls) return;
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
        tmpl_gap_lo.push_back(0);
        tmpl_gap_hi.push_back(0);
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
            run_cls = arg; run_len = 1; def_frame = t; def_conf = pa;
        } else {
            ++run_len;
            if (pa > def_conf) { def_frame = t; def_conf = pa; }
        }
    }
    flush_run();

    remove(name);
    TemplateMatcher m;
    m.name         = name;
    m.classes      = classes;
    m.gap_lo       = std::move(tmpl_gap_lo);
    m.gap_hi       = std::move(tmpl_gap_hi);
    if (n_alts > 0) m.alts = std::move(alts);
    m.policy       = pol;
    m.has_override = (policy_override != nullptr);
    m.reset_dp();
    impl_->matchers.push_back(std::move(m));
    ++impl_->tmpl_generation;
    impl_->publish_progress(impl_->matchers);
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

    // Enrollment runs the model over a SEPARATE throwaway session so it never
    // disturbs the live stream's caches; only this spotter's own mel front-end
    // needs the reset/restore dance below.
    impl_->mel->reset();
    PhonemeSession enroll_stream = impl_->model->make_session();

    bt::Tensor frames;   // (n_mels, T)
    const int T = impl_->mel->consume(padded.data(),
                                      static_cast<int>(padded.size()), frames);
    std::vector<float> post;   // (T, K) row-major softmax
    int T_post = 0;
    if (T > 0) {
        bt::Tensor logits;   // (T, K)
        impl_->model->forward_streaming(enroll_stream, frames, logits);
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
    // Restore the live front-end's clean state (the probe disturbed the mel
    // ring). The model's live session was never touched — enrollment ran on a
    // separate throwaway session.
    impl_->mel->reset();

    return enroll_from_posteriors(name, post.data(), T_post, policy_override);
}

bool PhonemeSpotter::remove(const std::string& name) {
    auto& v = impl_->matchers;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i].name == name) {
            v.erase(v.begin() + static_cast<std::ptrdiff_t>(i));
            ++impl_->tmpl_generation;
            impl_->publish_progress(v);
            return true;
        }
    }
    return false;
}

void PhonemeSpotter::clear() {
    impl_->matchers.clear();
    ++impl_->tmpl_generation;
    impl_->publish_progress(impl_->matchers);
}

std::vector<std::string> PhonemeSpotter::templates() const {
    std::vector<std::string> out;
    out.reserve(impl_->matchers.size());
    for (auto& m : impl_->matchers) out.push_back(m.name);
    return out;
}

bool PhonemeSpotter::inspect(const std::string& name, TemplateView& out) const {
    for (const auto& m : impl_->matchers) {
        if (m.name != name) continue;
        out.name      = m.name;
        out.threshold = m.policy.threshold;
        const int hop = impl_->hop_length > 0 ? impl_->hop_length : 160;
        const int sr  = impl_->sample_rate > 0 ? impl_->sample_rate : 16000;
        out.frame_ms  = 1000.0f * static_cast<float>(hop) / static_cast<float>(sr);
        out.has_gaps  = false;
        out.states.clear();
        out.states.reserve(m.classes.size());
        for (std::size_t j = 0; j < m.classes.size(); ++j) {
            TemplateState st;
            st.cls = m.classes[j];
            st.gap = (st.cls == 0);
            if (st.gap) {
                out.has_gaps = true;
                if (j < m.gap_lo.size()) st.gap_lo = m.gap_lo[j];
                if (j < m.gap_hi.size()) st.gap_hi = m.gap_hi[j];
            }
            out.states.push_back(st);
        }
        return true;
    }
    return false;
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
        ++impl_->frames_total;

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
            const int  prev_furthest = m.furthest;
            const bool fired = (L > 0) && m.step(lp, p, pmax, gate, ev);
            if (fired) {
                m.refractory_frames = impl_->refractory_frames_for(m.policy);
                ++m.completions;
                // A fire IS the prefix reaching L (step() then re-arms the DP,
                // so furthest reads 0 again) — record both telemetry marks.
                m.last_fire_frame    = impl_->frames_total;
                m.last_advance_frame = impl_->frames_total;
                // Absolute matched span on the frames axis: the fire is this
                // frame; the span ran matched_frames frames up to and including
                // it.
                ev.end_frame   = impl_->frames_total;
                ev.start_frame = impl_->frames_total -
                                 (ev.matched_frames > 0 ? ev.matched_frames - 1 : 0);
                events.push_back(ev);
            } else if (m.furthest > prev_furthest) {
                m.last_advance_frame = impl_->frames_total;
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
        impl_->publish_progress(impl_->matchers);

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
    // forward_streaming over the whole new mel block -> (T_new, K) logits,
    // advancing this spotter's private session over the shared net.
    bt::Tensor logits;
    impl_->model->forward_streaming(impl_->stream, new_frames, logits);
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

ProgressSnapshot PhonemeSpotter::progress_snapshot() const {
    ProgressSnapshot out;
    for (;;) {
        const std::uint32_t s1 = impl_->prog_seq.load(std::memory_order_acquire);
        if (s1 & 1u) continue;                       // writer mid-publish
        out = impl_->prog_pub;
        const std::uint32_t s2 = impl_->prog_seq.load(std::memory_order_acquire);
        if (s1 == s2) return out;                    // stable snapshot
    }
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
