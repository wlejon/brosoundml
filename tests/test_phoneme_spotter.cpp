// PhonemeSpotter tests — the streaming DP matcher driven by SYNTHETIC per-frame
// phoneme-class posteriors. Model-free and weights-free: a hand-built K=6 class
// map (sil,A,B,C,D,E) is installed via set_class_map(), templates are enrolled
// from class ids, and posterior frames are synthesized directly and pushed
// through feed_posteriors(). No Kokoro, no PhonemeNet, no mel — so this runs on
// CPU instantly with nothing on disk.
//
// Coverage: positive match, entry-silence gate, min-phoneme floor, wrong
// sequence, threshold + per-template override, refractory debounce, prefix
// progress, enroll collapse/silence-drop, remove/clear/templates, reset.

#include "brosoundml/phoneme_spotter.h"
#include "brosoundml/phoneme_data.h"

#include <brotensor/runtime.h>

#include <cmath>
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

int main() {
    brotensor::init();
    test_positive();
    test_entry_gate();
    test_min_phonemes();
    test_negative();
    test_threshold();
    test_refractory();
    test_prefix_progress();
    test_enroll_collapse();
    test_template_admin();
    test_reset();

    if (g_fail == 0) std::fprintf(stderr, "test_phoneme_spotter: all checks passed\n");
    else             std::fprintf(stderr, "test_phoneme_spotter: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
