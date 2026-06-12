// PhonemeSpotter tests — the streaming DP matcher driven by SYNTHETIC per-frame
// phoneme-class posteriors. Model-free and weights-free: a hand-built K=6 class
// map (sil,A,B,C,D,E) is installed via set_class_map(), templates are enrolled
// from class ids, and posterior frames are synthesized directly and pushed
// through feed_posteriors(). No Kokoro, no PhonemeNet, no mel — so this runs on
// CPU instantly with nothing on disk.
//
// Coverage: positive match, entry-silence gate, min-phoneme floor, wrong
// sequence, threshold + per-template override, refractory debounce, prefix
// progress, per-template progress snapshot, enroll collapse/silence-drop,
// remove/clear/templates, reset.

#include "brosoundml/phoneme_spotter.h"
#include "brosoundml/phoneme_data.h"

#include <brotensor/runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace bsm = brosoundml;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fail; } \
    else { std::fprintf(stderr, "  ok: %s\n", (msg)); } } while (0)

// ── A tiny hand-built class map: 0=sil, 1=A, 2=B, 3=C, 4=D, 5=E. Each id maps
//    to itself so class_for_id is total over [0,6). ───────────────────────────
static bsm::PhonemeClassMap toy_classmap() {
    bsm::PhonemeClassMap cm;
    cm.num_classes  = 6;
    cm.class_names  = {"sil", "A", "B", "C", "D", "E"};
    cm.class_to_ids = {{0}, {1}, {2}, {3}, {4}, {5}};
    cm.transparent_ids = {};
    cm.rebuild_inverse();
    return cm;
}

static constexpr int K = 6;

// Synthetic posterior stream builder. Each frame puts `mass` on the dominant
// class and spreads the remaining (1-mass) uniformly over the other K-1.
struct PostStream {
    std::vector<float> data;   // row-major (frames, K)
    int frames = 0;

    void add_frame(int cls, float mass) {
        const float other = (1.0f - mass) / static_cast<float>(K - 1);
        for (int k = 0; k < K; ++k) data.push_back(k == cls ? mass : other);
        ++frames;
    }
    void add_run(int cls, int n, float mass) { for (int i = 0; i < n; ++i) add_frame(cls, mass); }
    void add_silence(int n, float mass = 0.9f) { add_run(0, n, mass); }
    void add_word(const std::vector<int>& classes, int fpp, float mass) {
        for (int c : classes) add_run(c, fpp, mass);
    }
};

static std::vector<bsm::SpotEvent> feed(bsm::PhonemeSpotter& s, const PostStream& ps) {
    return s.feed_posteriors(ps.data.data(), ps.frames);
}

static bsm::PhonemeSpotter make_spotter() {
    bsm::PhonemeSpotter s;
    s.set_class_map(toy_classmap());
    return s;
}

// Clean rendering helpers.
static const std::vector<int> ABCD = {1, 2, 3, 4};

// ── 1. Positive match. ───────────────────────────────────────────────────────
static void test_positive() {
    auto s = make_spotter();
    CHECK(s.enroll_from_classes("abcd", ABCD) == 4, "enroll abcd -> len 4");

    PostStream ps;
    ps.add_silence(6);
    ps.add_word(ABCD, 4, 0.9f);
    ps.add_silence(4);
    auto ev = feed(s, ps);

    CHECK(ev.size() == 1, "positive: exactly one event");
    if (ev.size() == 1) {
        CHECK(ev[0].name == "abcd", "positive: event name == abcd");
        CHECK(ev[0].matched_phonemes == 4, "positive: matched_phonemes == 4");
        CHECK(ev[0].template_len == 4, "positive: template_len == 4");
        CHECK(ev[0].confidence > 0.6f, "positive: confidence > 0.6");
        std::fprintf(stderr, "    (confidence=%.3f)\n", ev[0].confidence);
    }
}

// ── 2. Entry-silence gate. ───────────────────────────────────────────────────
static void test_entry_gate() {
    auto s = make_spotter();
    s.enroll_from_classes("abcd", ABCD);

    // Garbage (closes the gate) immediately followed by the word, NO silence
    // between: the only legal entry point is start-of-stream (a garbage frame),
    // so the match is forced to drag the mismatched garbage frames into its span
    // and its geometric-mean confidence is crushed below threshold -> no fire.
    PostStream a;
    a.add_run(5, 12, 0.8f);          // 12 frames of class E (garbage, non-silence)
    a.add_word(ABCD, 4, 0.8f);       // word with no silence boundary before it
    auto ev_a = feed(s, a);
    CHECK(ev_a.empty(), "entry gate: garbage+word (no silence) does NOT fire");

    // Now a real silence boundary re-opens the gate and a fresh clean token
    // rides the next utterance to a high-confidence completion -> fires.
    PostStream b;
    b.add_silence(8);
    b.add_word(ABCD, 4, 0.8f);
    b.add_silence(4);
    auto ev_b = feed(s, b);
    CHECK(ev_b.size() == 1, "entry gate: silence gap + word DOES fire");
}

// ── 3. Min-phoneme floor. ────────────────────────────────────────────────────
static void test_min_phonemes() {
    auto s = make_spotter();
    CHECK(s.enroll_from_classes("ab", {1, 2}) == 2, "enroll ab -> len 2");

    PostStream ps;
    ps.add_silence(6);
    ps.add_word({1, 2}, 4, 0.85f);   // clean, but L=2 < default min_phonemes=3
    ps.add_silence(4);
    auto ev = feed(s, ps);
    CHECK(ev.empty(), "min floor: 2-phoneme template never fires (min=3)");
}

// ── 4. Wrong sequence / negative. ────────────────────────────────────────────
static void test_negative() {
    auto s = make_spotter();
    s.enroll_from_classes("abcd", ABCD);

    PostStream wrong;
    wrong.add_silence(6);
    wrong.add_word({5, 4, 3, 2}, 4, 0.8f);   // E,D,C,B — a different sequence
    wrong.add_silence(4);
    CHECK(feed(s, wrong).empty(), "negative: wrong sequence does not fire");

    s.reset();
    PostStream sil;
    sil.add_silence(40);
    CHECK(feed(s, sil).empty(), "negative: pure silence does not fire");
}

// ── 5. Threshold + per-template override. ────────────────────────────────────
static void test_threshold() {
    // Default threshold 0.40: a smeared word (mass 0.30 on the right class) has
    // a geometric-mean posterior ~0.30 < 0.40 -> no fire.
    {
        auto s = make_spotter();
        s.enroll_from_classes("abcd", ABCD);
        PostStream ps;
        ps.add_silence(6);
        ps.add_word(ABCD, 4, 0.30f);
        ps.add_silence(4);
        CHECK(feed(s, ps).empty(), "threshold: smeared word below default 0.40 -> no fire");
    }
    // Same stream, but a per-template override lowers the threshold to 0.20 -> fires.
    {
        auto s = make_spotter();
        bsm::SpotterConfig ov = s.config();
        ov.threshold = 0.20f;
        s.enroll_from_classes("abcd", ABCD, &ov);
        PostStream ps;
        ps.add_silence(6);
        ps.add_word(ABCD, 4, 0.30f);
        ps.add_silence(4);
        auto ev = feed(s, ps);
        CHECK(ev.size() == 1, "threshold: override 0.20 fires on the same smeared word");
    }
}

// ── 5b. Competition (posterior-ratio) score normalization. ──────────────────
static void test_score_norm() {
    // A weak-but-WINNING word: mass 0.35 on the right class each frame (others
    // get ~0.13). Raw geometric-mean confidence ~0.35 < default threshold 0.40
    // -> no fire. With score_norm=1 each frame scores p/max(p_argmax, 0.5) =
    // 0.35/0.5 = 0.70 -> fires at the SAME default threshold. This is the
    // per-template scale fix: a confidently-WON frame qualifies regardless of
    // the class's absolute posterior calibration.
    {
        auto s = make_spotter();
        s.enroll_from_classes("abcd", ABCD);
        PostStream ps;
        ps.add_silence(6);
        ps.add_word(ABCD, 4, 0.35f);
        ps.add_silence(4);
        CHECK(feed(s, ps).empty(), "score_norm off: weak-winning word below 0.40 -> no fire");
    }
    {
        auto s = make_spotter();
        bsm::SpotterConfig ov = s.config();
        ov.score_norm   = 1.0f;      // ref stays at the 0.5 default
        ov.min_phonemes = 4;         // toy K=6 spreads 0.13 on absent classes —
                                     // close enough to the 0.15 floor that a
                                     // 3-of-4 coverage completion can fire one
                                     // phoneme early; demand all four so the
                                     // fire lands on the true D-frame finish.
        s.enroll_from_classes("abcd", ABCD, &ov);
        PostStream ps;
        ps.add_silence(6);
        ps.add_word(ABCD, 4, 0.35f);
        ps.add_silence(4);
        auto ev = feed(s, ps);
        CHECK(ev.size() == 1, "score_norm on: same weak-winning word fires");
        if (ev.size() == 1) {
            CHECK(ev[0].confidence > 0.6f, "score_norm: confidence lifted to ~0.70");
            std::fprintf(stderr, "    (confidence=%.3f)\n", ev[0].confidence);
        }
    }
    // Normalization must NOT inflate a LOSING template: a confident E,D,C,B
    // utterance gives the abcd classes only ~0.07 each while the winner holds
    // 0.8, so the per-frame ratio (~0.08) is floored and never covered ->
    // coverage gate rejects the match.
    {
        auto s = make_spotter();
        bsm::SpotterConfig ov = s.config();
        ov.score_norm = 1.0f;
        s.enroll_from_classes("abcd", ABCD, &ov);
        PostStream ps;
        ps.add_silence(6);
        ps.add_word({5, 4, 3, 2}, 4, 0.8f);
        ps.add_silence(4);
        CHECK(feed(s, ps).empty(), "score_norm on: losing template still does not fire");
    }
}

// ── 5c. Proportional coverage gate. ──────────────────────────────────────────
static void test_coverage_frac() {
    // A 5-phoneme template where only the first THREE phonemes are actually
    // spoken: the trailing D,E are "matched" on silence frames at the emission
    // floor. covered=3 satisfies the absolute min_phonemes=3, so without the
    // proportional gate this fragment FIRES (conf ~0.63 — the floor keeps the
    // geometric mean afloat). That is the long-template hole: 3-of-5 is only
    // 60% real evidence, and it shrinks as templates grow.
    const std::vector<int> ABCDE = {1, 2, 3, 4, 5};
    {
        auto s = make_spotter();
        s.enroll_from_classes("abcde", ABCDE);
        PostStream ps;
        ps.add_silence(6);
        ps.add_word({1, 2, 3}, 4, 0.8f);   // A,B,C spoken; D,E never occur
        ps.add_silence(6);
        CHECK(feed(s, ps).size() == 1,
              "coverage_frac off: 3-of-5 fragment fires (the hole)");
    }
    // min_coverage_frac=0.8 demands ceil(0.8*5)=4 covered -> the fragment is
    // rejected ...
    {
        auto s = make_spotter();
        bsm::SpotterConfig ov = s.config();
        ov.min_coverage_frac = 0.8f;
        s.enroll_from_classes("abcde", ABCDE, &ov);
        PostStream ps;
        ps.add_silence(6);
        ps.add_word({1, 2, 3}, 4, 0.8f);
        ps.add_silence(6);
        CHECK(feed(s, ps).empty(),
              "coverage_frac 0.8: 3-of-5 fragment does NOT fire");
    }
    // ... while the fully spoken word still fires under the same gate.
    {
        auto s = make_spotter();
        bsm::SpotterConfig ov = s.config();
        ov.min_coverage_frac = 0.8f;
        s.enroll_from_classes("abcde", ABCDE, &ov);
        PostStream ps;
        ps.add_silence(6);
        ps.add_word(ABCDE, 4, 0.8f);
        ps.add_silence(6);
        CHECK(feed(s, ps).size() == 1,
              "coverage_frac 0.8: fully spoken word still fires");
    }
}

// ── 6. Refractory debounce. ──────────────────────────────────────────────────
static void test_refractory() {
    // refractory_ms 600 @ 16 kHz / 160 hop = 60 frames. Two utterances inside
    // that window collapse to one fire; spaced beyond it -> two fires.
    auto run = [](int gap_frames) {
        auto s = make_spotter();
        s.enroll_from_classes("abcd", ABCD);
        PostStream ps;
        ps.add_silence(5);
        ps.add_word(ABCD, 4, 0.8f);
        ps.add_silence(gap_frames);
        ps.add_word(ABCD, 4, 0.8f);
        ps.add_silence(4);
        return feed(s, ps).size();
    };
    CHECK(run(10)  == 1, "refractory: short gap (10f) -> one fire");
    CHECK(run(120) == 2, "refractory: long gap (120f) -> two fires");
}

// ── 7. Prefix progress. ──────────────────────────────────────────────────────
static void test_prefix_progress() {
    auto s = make_spotter();
    s.enroll_from_classes("abcd", ABCD);

    // Start-of-stream counts as a silence boundary, so a fresh token enters at
    // frame 0. With one frame per phoneme, the alignment advances exactly one
    // state per frame: after A,B the furthest matched state is 2 of 4 -> ~0.5.
    PostStream ps;
    ps.add_word({1, 2}, 1, 0.8f);   // A (1 frame), B (1 frame)
    auto ev = feed(s, ps);
    CHECK(ev.empty(), "progress: half a word does not fire");
    const float p = s.prefix_progress();
    std::fprintf(stderr, "    (prefix_progress=%.3f)\n", p);
    CHECK(std::fabs(p - 0.5f) < 0.05f, "progress: ~0.5 after 2 of 4 phonemes");
}

// ── 7b. Per-template progress snapshot (fused-surface telemetry). ────────────
static void test_progress_snapshot() {
    auto s = make_spotter();

    auto s0 = s.progress_snapshot();
    CHECK(s0.count == 0 && s0.frames == 0, "snapshot: empty before enroll");
    const std::uint32_t gen0 = s0.generation;

    s.enroll_from_classes("abcd", ABCD);
    s.enroll_from_classes("cec", {3, 5, 3});
    auto s1 = s.progress_snapshot();
    CHECK(s1.count == 2, "snapshot: two entries after enroll");
    CHECK(s1.generation != gen0, "snapshot: generation bumps on enroll");
    CHECK(std::string(s1.templates[0].name) == "abcd" &&
          std::string(s1.templates[1].name) == "cec",
          "snapshot: names in enroll order");
    CHECK(s1.templates[0].length == 4 && s1.templates[1].length == 3,
          "snapshot: per-entry template lengths");

    // Half a word: abcd's prefix reads 2/4 with a REAL partial confidence (the
    // geometric mean of two 0.8-mass frames), while cec only ever rides the
    // emission floor.
    PostStream half;
    half.add_word({1, 2}, 1, 0.8f);   // A, B — one frame per phoneme
    feed(s, half);
    auto s2 = s.progress_snapshot();
    CHECK(s2.frames == 2, "snapshot: frames counts posterior frames");
    {
        const auto& e = s2.templates[0];
        CHECK(e.matched == 2 && std::fabs(e.progress - 0.5f) < 0.05f,
              "snapshot: abcd matched 2/4 after half a word");
        CHECK(e.confidence > 0.6f && e.confidence <= 1.0f,
              "snapshot: partial confidence tracks the real prefix score");
        CHECK(e.last_advance_frame == 2,
              "snapshot: last_advance_frame == frame of the latest extension");
        CHECK(e.completions == 0 && e.last_fire_frame == -1,
              "snapshot: no completion recorded yet");
        CHECK(s2.templates[1].confidence < 0.3f,
              "snapshot: absent template's confidence stays at the floor");
        std::fprintf(stderr, "    (abcd conf=%.3f, cec conf=%.3f)\n",
                     e.confidence, s2.templates[1].confidence);
    }

    // A clean full utterance fires: completions ticks, last_fire_frame lands
    // inside the fed range, and the re-armed DP drops the prefix.
    PostStream full;
    full.add_silence(6);
    full.add_word(ABCD, 4, 0.9f);
    full.add_silence(4);
    auto ev = feed(s, full);
    CHECK(ev.size() == 1, "snapshot: full word fires once");
    auto s3 = s.progress_snapshot();
    CHECK(s3.frames == 2 + 26, "snapshot: frames accumulates across feeds");
    {
        const auto& e = s3.templates[0];
        CHECK(e.completions == 1, "snapshot: completions == 1 after the fire");
        CHECK(e.last_fire_frame > 2 && e.last_fire_frame <= s3.frames,
              "snapshot: last_fire_frame inside the fed range");
        CHECK(e.matched < e.length, "snapshot: DP re-armed after the fire");
    }

    // reset() clears alignments but the monotonic history survives, so a
    // poller diffing counters across a reset never sees them go backwards.
    s.reset();
    auto s4 = s.progress_snapshot();
    CHECK(s4.frames == s3.frames, "snapshot: frames survives reset()");
    CHECK(s4.templates[0].completions == 1,
          "snapshot: completions survives reset()");
    CHECK(s4.templates[0].matched == 0, "snapshot: matched cleared by reset()");

    // remove() shrinks the entry set and bumps the generation.
    const std::uint32_t gen_before = s4.generation;
    CHECK(s.remove("cec"), "snapshot: remove cec");
    auto s5 = s.progress_snapshot();
    CHECK(s5.count == 1 && std::string(s5.templates[0].name) == "abcd",
          "snapshot: one entry after remove");
    CHECK(s5.generation != gen_before, "snapshot: generation bumps on remove");

    // Oversized names are truncated, NUL-terminated, into the POD entry.
    s.enroll_from_classes(std::string(60, 'x'), ABCD);
    auto s6 = s.progress_snapshot();
    CHECK(std::string(s6.templates[1].name) == std::string(47, 'x'),
          "snapshot: long name truncated to the POD field");
}

// ── 8. enroll collapse + silence-drop. ───────────────────────────────────────
static void test_enroll_collapse() {
    auto s = make_spotter();
    // [A,A,sil,B,B,B,C] -> drop sil, collapse runs -> [A,B,C] (len 3).
    const int len = s.enroll_from_classes("abc", {1, 1, 0, 2, 2, 2, 3});
    CHECK(len == 3, "collapse: [A,A,sil,B,B,B,C] -> len 3");

    // And it actually behaves as [A,B,C]: a clean A,B,C utterance fires.
    PostStream ps;
    ps.add_silence(6);
    ps.add_word({1, 2, 3}, 4, 0.8f);
    ps.add_silence(4);
    auto ev = feed(s, ps);
    CHECK(ev.size() == 1 && ev[0].matched_phonemes == 3,
          "collapse: collapsed template matches A,B,C");
}

// ── 9. remove / clear / templates. ───────────────────────────────────────────
static void test_template_admin() {
    auto s = make_spotter();
    s.enroll_from_classes("abcd", ABCD);
    s.enroll_from_classes("abc", {1, 2, 3});
    CHECK(s.templates().size() == 2, "admin: two templates enrolled");

    CHECK(s.remove("abcd") == true, "admin: remove existing returns true");
    CHECK(s.remove("nope") == false, "admin: remove missing returns false");
    auto t = s.templates();
    CHECK(t.size() == 1 && t[0] == "abc", "admin: one template left (abc)");

    s.clear();
    CHECK(s.templates().empty(), "admin: clear drops all templates");
}

// ── 10. reset clears streaming state. ────────────────────────────────────────
static void test_reset() {
    auto s = make_spotter();
    s.enroll_from_classes("abcd", ABCD);

    PostStream half;
    half.add_silence(4);
    half.add_word({1, 2}, 3, 0.8f);   // A,B — partial
    feed(s, half);
    CHECK(s.prefix_progress() > 0.0f, "reset: progress advanced on partial word");

    s.reset();
    CHECK(s.prefix_progress() == 0.0f, "reset: progress cleared to 0");

    // The second half alone (no preceding A,B in the DP) must not complete.
    PostStream rest;
    rest.add_word({3, 4}, 3, 0.8f);   // C,D
    rest.add_silence(4);
    auto ev = feed(s, rest);
    CHECK(ev.empty(), "reset: dangling second half does not fire");
}

// ── 13. Soft states: enrollment alternates absorb unit-identity flips. ──────
// Self-supervised units are k-means cells; a re-performance flips frames
// between neighbouring cells. Enroll a stream whose frames carry a strong
// runner-up class, then detect a rendition where primary and runner-up have
// SWAPPED: soft templates (enroll_alts=2) must fire, hard ones must not.
static void test_soft_states() {
    // Frame with two designated masses; remainder spread over the rest.
    auto add_frame2 = [](PostStream& ps, int cls, float mass,
                         int alt, float alt_mass) {
        const float other = (1.0f - mass - alt_mass) / static_cast<float>(K - 2);
        for (int k = 0; k < K; ++k)
            ps.data.push_back(k == cls ? mass : (k == alt ? alt_mass : other));
        ++ps.frames;
    };

    // Enrollment: word [A,B,C], each state's frames hold 0.55 on the primary
    // and 0.30 on a non-template runner-up (A~D, B~E, C~D).
    PostStream en;
    en.add_silence(4);
    for (int i = 0; i < 3; ++i) add_frame2(en, 1, 0.55f, 4, 0.30f);
    for (int i = 0; i < 3; ++i) add_frame2(en, 2, 0.55f, 5, 0.30f);
    for (int i = 0; i < 3; ++i) add_frame2(en, 3, 0.55f, 4, 0.30f);
    en.add_silence(4);

    // Detection: the same word re-performed with the cells FLIPPED — frames
    // now win on the runner-up (0.70) with the enrolled primary at 0.12.
    PostStream det;
    det.add_silence(4);
    for (int i = 0; i < 3; ++i) add_frame2(det, 4, 0.70f, 1, 0.12f);
    for (int i = 0; i < 3; ++i) add_frame2(det, 5, 0.70f, 2, 0.12f);
    for (int i = 0; i < 3; ++i) add_frame2(det, 4, 0.70f, 3, 0.12f);
    det.add_silence(4);

    {   // Hard template (control): the flip sinks every emission -> no fire.
        auto s = make_spotter();
        bsm::SpotterConfig pol = s.config();
        pol.enroll_alts = 0;
        CHECK(s.enroll_from_posteriors("word", en.data.data(), en.frames,
                                       &pol) == 3,
              "soft: hard enroll from posteriors -> len 3");
        CHECK(feed(s, det).empty(), "soft: hard template misses the flipped take");
    }
    {   // Soft template: state mass = primary + alternate -> fires.
        auto s = make_spotter();
        bsm::SpotterConfig pol = s.config();
        pol.enroll_alts = 2;
        CHECK(s.enroll_from_posteriors("word", en.data.data(), en.frames,
                                       &pol) == 3,
              "soft: soft enroll from posteriors -> len 3");
        auto ev = feed(s, det);
        CHECK(ev.size() == 1, "soft: soft template fires on the flipped take");
        if (ev.size() == 1)
            CHECK(ev[0].confidence > 0.6f, "soft: flipped-take confidence > 0.6");
    }
    {   // Confidence gate: low-confidence churn frames are dropped at enroll.
        PostStream churn;
        churn.add_silence(4);
        churn.add_run(4, 2, 0.30f);   // churn below the gate
        churn.add_word({1, 2, 3}, 3, 0.85f);
        churn.add_run(5, 2, 0.30f);   // churn below the gate
        churn.add_silence(4);
        auto s = make_spotter();
        bsm::SpotterConfig pol = s.config();
        pol.enroll_conf_gate = 0.5f;
        CHECK(s.enroll_from_posteriors("word", churn.data.data(), churn.frames,
                                       &pol) == 3,
              "soft: conf gate strips churn -> len 3 (not 5)");
    }
}

int main() {
    brotensor::init();
    test_positive();
    test_entry_gate();
    test_min_phonemes();
    test_negative();
    test_threshold();
    test_score_norm();
    test_coverage_frac();
    test_refractory();
    test_prefix_progress();
    test_progress_snapshot();
    test_enroll_collapse();
    test_template_admin();
    test_reset();
    test_soft_states();

    if (g_fail == 0) std::fprintf(stderr, "test_phoneme_spotter: all checks passed\n");
    else             std::fprintf(stderr, "test_phoneme_spotter: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
