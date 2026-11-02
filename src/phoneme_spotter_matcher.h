#pragma once

#include "brosoundml/phoneme_spotter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace brosoundml {
namespace detail {

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
    std::vector<int>     classes;   // c[0..L-1], distinct adjacent phoneme classes;
                                    // class 0 = a TIMED GAP state (rhythm templates)
    // Per-state gap duration window in frames, parallel to `classes`. (0,0) for
    // sound states. For a gap state, the legal dwell is [gap_lo, gap_hi]:
    // holding it past gap_hi kills the path, leaving before gap_lo is illegal.
    std::vector<int>     gap_lo, gap_hi;
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
    // R[j] = frames spent in state j on the best path reaching it (current
    // run length) — what the gap duration window is enforced against.
    std::vector<int>          R;
    // F[j] = consecutive FLOORED frames the best path has been riding in
    // state j (rhythm templates only — see kMaxFloorRun).
    std::vector<int>          F;

    // Rhythm templates: longest run of floored frames a state may ride. Long
    // enough to absorb intra-sound unit churn and a brief blip inside a gap
    // (30 ms at the 10 ms hop), far too short to stretch the rhythm's timing.
    static constexpr int kMaxFloorRun = 3;

    // M-of-N smoothing ring of per-frame "confidence > threshold" flags.
    std::vector<std::uint8_t> hits_ring;
    int                  ring_head = 0;
    int                  ring_len  = 0;

    int                  refractory_frames = 0;   // counts down once per frame
    int                  furthest = 0;            // largest j with finite S[j]
    bool                 has_gap = false;         // any timed gap state present

    // Progress telemetry (see TemplateProgress). History, not matcher state:
    // reset_dp() leaves these alone so pollers' counter diffs stay monotonic.
    std::int64_t         completions        = 0;
    std::int64_t         last_advance_frame = -1;
    std::int64_t         last_fire_frame    = -1;

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
        R.assign(static_cast<std::size_t>(L) + 1, 0);
        F.assign(static_cast<std::size_t>(L) + 1, 0);
        // Templates enrolled without gap data are all-sound: size the bound
        // vectors so step() can index them unconditionally.
        if (static_cast<int>(gap_lo.size()) != L) gap_lo.assign(static_cast<std::size_t>(L), 0);
        if (static_cast<int>(gap_hi.size()) != L) gap_hi.assign(static_cast<std::size_t>(L), 0);
        has_gap = false;
        for (int c : classes) has_gap |= (c == 0);
        refractory_frames = 0;
        furthest = 0;
        size_ring();
    }

    // State j (1..L) is a timed gap state. Gap states always carry bounds.
    bool is_gap(int j) const { return classes[static_cast<std::size_t>(j - 1)] == 0; }

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
        R[0]   = 0;
        F[0]   = 0;

        // Transitions: iterate j = L .. 1 so each candidate reads the PREVIOUS
        // frame's S[j-1]/N[j-1]/R[j-1] (lower indices not yet touched this
        // frame), and the current-frame launch pad at j == 1.
        for (int j = L; j >= 1; --j) {
            const int cls = classes[static_cast<std::size_t>(j - 1)];
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
            const bool above = raw_lp > log_floor;

            double s_stay = S[j];              // previous frame's S[j]
            // A timed GAP state may not be held past its window: rather than
            // merely scoring badly, an overlong gap is an ILLEGAL path.
            if (s_stay != kNegInf && is_gap(j) &&
                gap_hi[static_cast<std::size_t>(j - 1)] > 0 &&
                R[j] + 1 > gap_hi[static_cast<std::size_t>(j - 1)]) {
                s_stay = kNegInf;
            }
            double s_adv = S[j - 1];           // previous frame (or launch pad)
            // Leaving a timed GAP state requires its minimum dwell — a
            // too-short gap can never hand off to the next sound.
            if (s_adv != kNegInf && j >= 2 && is_gap(j - 1) &&
                R[j - 1] < gap_lo[static_cast<std::size_t>(j - 2)]) {
                s_adv = kNegInf;
            }
            // Rhythm templates: a state must be HEARD before the path may
            // move past it, and floor-idling is bounded. The emission floor
            // exists so an unreliable transient phoneme can't veto a speech
            // template, but in a TIMED template unlimited floor-riding breaks
            // the timing two ways (measured in the unit tests): a
            // floor-anchored fresh entry exploits the raw log-score length
            // bias to displace the genuine path mid-gap, and a sound state
            // idling on the floor after the gap stretches the window until a
            // late sound "completes" the rhythm. So for has_gap templates
            // only: (1) advancing PAST state j-1 requires that its evidence
            // was actually heard on this path (Cur — a mid-gap phantom entry
            // whose sound never occurred can't hop into the gap state and
            // ride its high silence scores, and a pure-floor ladder can't
            // pass ANY state); (2) STAYING floored is legal only for
            // kMaxFloorRun frames in a row — enough to absorb intra-sound
            // unit churn or a one-frame blip inside a gap, far too short to
            // bend the rhythm. Entering a state on floored frames IS legal
            // (bounded by (2)): the live front-end adapts to what it heard
            // seconds ago while enrollment ran on a fresh pass (measured:
            // a re-performance ~2 s after a fire missed entry entirely under
            // a strict advance-on-evidence rule), so each transition gets
            // kMaxFloorRun frames of onset-drift slack before its sound must
            // appear. Sound-only templates keep the legacy behavior (their
            // tuned recall/FAR numbers are untouched).
            if (has_gap) {
                if (s_adv != kNegInf && j >= 2 && !Cur[j - 1]) s_adv = kNegInf;
                if (s_stay != kNegInf && !above && F[j] >= kMaxFloorRun) s_stay = kNegInf;
            }
            const double best = s_stay > s_adv ? s_stay : s_adv;
            if (best == kNegInf) {
                S[j]   = kNegInf;
                N[j]   = 0;
                Cov[j] = 0;
                Cur[j] = 0u;
                R[j]   = 0;
                F[j]   = 0;
                continue;
            }
            const bool   advance = (s_adv > s_stay);
            const int    prevN    = advance ? N[j - 1] : N[j];
            const double norm_emit = raw_lp - log_comp;
            const double emit     = (norm_emit < log_floor) ? log_floor : norm_emit;
            S[j] = best + emit;
            N[j] = prevN + 1;
            R[j] = advance ? 1 : R[j] + 1;
            F[j] = above ? 0 : (advance ? 1 : F[j] + 1);
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

        // Prefix progress: furthest finite state. Rhythm templates may carry
        // short-lived floored probe paths (entry precedes evidence by up to
        // kMaxFloorRun frames), so count a state as reached only once its
        // evidence was actually heard — telemetry reports what matched, not
        // what is being speculatively held at the floor.
        furthest = 0;
        for (int j = L; j >= 1; --j) {
            if (S[j] != kNegInf && (!has_gap || Cur[j])) { furthest = j; break; }
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
                out_ev.matched_frames   = N[L];  // span length (captured pre-reset)
                // Re-enter from silence for the next utterance: clear the DP,
                // the smoothing ring, and arm the refractory window.
                S.assign(static_cast<std::size_t>(L) + 1, kNegInf);
                N.assign(static_cast<std::size_t>(L) + 1, 0);
                Cov.assign(static_cast<std::size_t>(L) + 1, 0);
                Cur.assign(static_cast<std::size_t>(L) + 1, 0u);
                R.assign(static_cast<std::size_t>(L) + 1, 0);
                F.assign(static_cast<std::size_t>(L) + 1, 0);
                furthest = 0;
                size_ring();            // drop stale positive flags
                refractory_frames = 0;  // set below from ms→frames by the owner
                return true;
            }
        }
        return false;
    }
};

}  // namespace detail
}  // namespace brosoundml
