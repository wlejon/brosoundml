#define _CRT_SECURE_NO_WARNINGS
// Yue2::splice / Yue2::inpaint on a real song (minutes; YUE2_E2E=1). The
// given run's tokens (rock_gen_W0, capped at $YUE2_EDIT_FRAMES, default 1500)
// and their whole-song render (Yue2::synthesize) are the song being edited.
//   - the scorer: B after its own past is likelier than B after a cut that
//     skips 10 s of the song (mean NLL of B's first 5 s);
//   - regenerate 10 s: A and B are kept token for token, each candidate M is
//     exactly the span, the kept M is the best-scored and B follows it better
//     than it follows the cut;
//   - inpaint that span: the frames outside it are untouched, it solves at the
//     whole render's loudness (mean latent norm within 0.8), and its edges
//     jump no more than the song's own frame-to-frame steps do (4x median);
//   - a 2 s span of the original tokens re-solved close to the original
//     (cosine) and not loud: the short-window case;
//   - a cut closed: tokens A + B, the seam re-solved, its edges smooth;
//   - continue: A = the song's first 20 s, sampled on (capped) after it.
// $YUE2_EDIT_OUT: a folder for WAVs of each edit, to listen to.

#include "test_yue2_stream_common.h"

using namespace yue2_stream_test;

namespace {

constexpr int D = 64;

double cos_frames(const std::vector<float>& a, const std::vector<float>& b, int f0, int f1) {
    double s = 0;
    int n = 0;
    for (int f = f0; f < f1; ++f) {
        double ab = 0, aa = 0, bb = 0;
        for (int d = 0; d < D; ++d) {
            const double x = a[static_cast<std::size_t>(f) * D + d], y = b[static_cast<std::size_t>(f) * D + d];
            ab += x * y, aa += x * x, bb += y * y;
        }
        s += ab / (std::sqrt(aa * bb) + 1e-12);
        ++n;
    }
    return n ? s / n : 0;
}

double norm_frames(const std::vector<float>& a, int f0, int f1) {
    double s = 0;
    for (int f = f0; f < f1; ++f) {
        double q = 0;
        for (int d = 0; d < D; ++d) q += double(a[static_cast<std::size_t>(f) * D + d]) * a[static_cast<std::size_t>(f) * D + d];
        s += std::sqrt(q);
    }
    return f1 > f0 ? s / (f1 - f0) : 0;
}

// |lat[f] - lat[f - 1]|
double step_at(const std::vector<float>& a, int f) {
    double q = 0;
    for (int d = 0; d < D; ++d) {
        const double v = a[static_cast<std::size_t>(f) * D + d] - a[static_cast<std::size_t>(f - 1) * D + d];
        q += v * v;
    }
    return std::sqrt(q);
}

double median_step(const std::vector<float>& a, int frames) {
    std::vector<double> s;
    for (int f = 1; f < frames; ++f) s.push_back(step_at(a, f));
    std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
    return s[s.size() / 2];
}

std::vector<int32_t> slice(const std::vector<int32_t>& v, int a, int b) {
    return std::vector<int32_t>(v.begin() + a, v.begin() + b);
}

}  // namespace

int main() {
    auto Env_ = std::make_unique<Env>();
    Env& E = *Env_;
    if (!load_env(E)) return 0;
    const fs::path out = env("YUE2_EDIT_OUT");
    if (!out.empty()) fs::create_directories(out);
    auto wav = [&](const char* name, const std::vector<float>& lat) {
        if (out.empty()) return;
        E.vae.decode(lat.data(), static_cast<int>(lat.size() / D)).write_wav((out / name).string());
    };

    brosoundml::Yue2Params p = E.given;
    const brosoundml::Yue2Plan plan = E.y.plan(p);
    const int T = std::min(static_cast<int>(E.given_sem.size()), std::stoi(env("YUE2_EDIT_FRAMES", "1500")));
    const std::vector<int32_t> sem = slice(E.given_sem, 0, T);
    double t = now_s();
    const std::vector<float> whole = E.y.synthesize(p, plan, sem);
    std::printf("whole render: %d frames in %.1f s, norm %.2f\n", T, now_s() - t, norm_frames(whole, 0, T));
    wav("original.wav", whole);
    const double nw = norm_frames(whole, 0, T), med = median_step(whole, T);

    // The span edited: 10 s from 30 s.
    const int a = 750, b = 1000;
    CHECK(T >= b + 250, "the song is too short for the edits (%d frames)", T);

    // ── the scorer ──
    brosoundml::Yue2Splice nat;
    nat.before = slice(sem, 0, b);
    nat.after = slice(sem, b, b + 125);
    t = now_s();
    const auto rn = E.y.splice(p, plan, nat);
    brosoundml::Yue2Splice cut = nat;
    cut.before = slice(sem, 0, a);
    const auto rc = E.y.splice(p, plan, cut);
    std::printf("scorer: B after its own past %.3f nats, after a 10 s cut %.3f (%.1f s)\n", rn.scores[0], rc.scores[0],
                now_s() - t);
    CHECK(rn.scores[0] < rc.scores[0], "B is no likelier after its own past (%.3f) than after a cut (%.3f)", rn.scores[0],
          rc.scores[0]);

    // ── regenerate the span ──
    brosoundml::Yue2Splice re;
    re.before = slice(sem, 0, a);
    re.after = slice(sem, b, T);
    re.middle_frames = b - a;
    re.candidates = 3;
    p.seed = E.given.seed + 7;
    t = now_s();
    const auto rr = E.y.splice(p, plan, re);
    std::printf("regenerate: %.1f s; scores", now_s() - t);
    for (float s : rr.scores) std::printf(" %.3f", s);
    std::printf(" -> %d\n", rr.chosen);
    CHECK(static_cast<int>(rr.semantic.size()) == T, "regenerated song has %zu frames, not %d", rr.semantic.size(), T);
    CHECK(rr.middle0 == a && rr.after0 == b, "middle at [%d, %d), not [%d, %d)", rr.middle0, rr.after0, a, b);
    CHECK(rr.middles.size() == 3, "%zu candidates drawn, not 3", rr.middles.size());
    for (const auto& m : rr.middles) CHECK(static_cast<int>(m.size()) == b - a, "a middle of %zu frames", m.size());
    CHECK(slice(rr.semantic, 0, a) == re.before && slice(rr.semantic, b, T) == re.after, "A or B was not kept");
    for (std::size_t k = 0; k < rr.scores.size(); ++k)
        CHECK(rr.scores[rr.chosen] <= rr.scores[k], "the kept middle is not the best scored");
    CHECK(rr.scores[rr.chosen] < rc.scores[0], "B follows the new middle (%.3f) no better than the cut (%.3f)",
          rr.scores[rr.chosen], rc.scores[0]);
    CHECK(slice(rr.semantic, a, b) != slice(sem, a, b), "the middle came back unchanged");

    // ── inpaint the regenerated span ──
    std::vector<float> lat(whole);
    std::fill(lat.begin() + static_cast<std::size_t>(a) * D, lat.begin() + static_cast<std::size_t>(b) * D, 0.0f);
    t = now_s();
    const auto ip = E.y.inpaint(p, plan, rr.semantic, lat, a, b);
    const double ni = norm_frames(ip, a, b);
    std::printf("inpaint 10 s: %.1f s, norm %.2f (whole %.2f), edge steps %.2f / %.2f (median %.2f)\n", now_s() - t, ni,
                nw, step_at(ip, a), step_at(ip, b), med);
    wav("regenerated.wav", ip);
    CHECK(std::equal(ip.begin(), ip.begin() + static_cast<std::size_t>(a) * D, whole.begin()) &&
              std::equal(ip.begin() + static_cast<std::size_t>(b) * D, ip.end(), whole.begin() + static_cast<std::size_t>(b) * D),
          "frames outside the span changed");
    CHECK(ni < nw + 0.8 && ni > nw - 1.5, "the span solved at norm %.2f, the whole render %.2f", ni, nw);
    CHECK(step_at(ip, a) < 4 * med && step_at(ip, b) < 4 * med, "an edge of the span jumps (%.2f, %.2f; median %.2f)",
          step_at(ip, a), step_at(ip, b), med);

    // ── a short span of the original tokens ──
    {
        const int s0 = 500, s1 = 550;
        const auto sp = E.y.inpaint(E.given, plan, sem, whole, s0, s1);
        const double c = cos_frames(sp, whole, s0, s1), n2 = norm_frames(sp, s0, s1), n0 = norm_frames(whole, s0, s1);
        std::printf("inpaint 2 s of the original: cosine %.3f, norm %.2f (there %.2f)\n", c, n2, n0);
        CHECK(c > 0.6, "a re-solved 2 s span is far from the original (cosine %.3f)", c);
        CHECK(n2 < n0 + 0.8, "a 2 s span solved loud: %.2f against %.2f", n2, n0);
    }

    // ── close a cut ──
    {
        std::vector<int32_t> toks = slice(sem, 0, a);
        toks.insert(toks.end(), sem.begin() + b, sem.end());
        std::vector<float> cl(whole.begin(), whole.begin() + static_cast<std::size_t>(a) * D);
        cl.insert(cl.end(), whole.begin() + static_cast<std::size_t>(b) * D, whole.end());
        const int n = static_cast<int>(toks.size());
        const double raw = step_at(cl, a);
        const auto sc = E.y.inpaint(p, plan, toks, cl, a - 12, a + 12);
        std::printf("cut closed: seam step %.2f raw, after %.2f / %.2f / %.2f (median %.2f), norm %.2f\n", raw,
                    step_at(sc, a - 12), step_at(sc, a), step_at(sc, a + 12), med, norm_frames(sc, a - 12, a + 12));
        wav("cut.wav", sc);
        CHECK(static_cast<int>(sc.size()) == n * D, "the cut song has the wrong length");
        CHECK(step_at(sc, a - 12) < 4 * med && step_at(sc, a + 12) < 4 * med && step_at(sc, a) < 4 * med,
              "the closed seam jumps");
    }

    // ── continue ──
    {
        brosoundml::Yue2Splice co;
        co.before = slice(sem, 0, 500);
        co.continue_after = true;
        brosoundml::Yue2Params pc = p;
        pc.semantic_sampling.max_tokens = 1250;
        t = now_s();
        const auto r = E.y.splice(pc, plan, co);
        std::printf("continue: %zu frames (%s) in %.1f s\n", r.semantic.size(), r.truncated ? "capped" : "ended",
                    now_s() - t);
        CHECK(slice(r.semantic, 0, 500) == co.before, "continue changed A");
        CHECK(r.semantic.size() > 600, "continue sampled only %zu frames", r.semantic.size());
        if (!out.empty()) {
            std::vector<float> cl(r.semantic.size() * D, 0.0f);
            std::copy(whole.begin(), whole.begin() + 500 * D, cl.begin());
            const auto cv = E.y.inpaint(pc, plan, r.semantic, cl, 500, static_cast<int>(r.semantic.size()));
            std::printf("continue rendered: norm %.2f after A (A %.2f)\n",
                        norm_frames(cv, 500, static_cast<int>(r.semantic.size())), norm_frames(cv, 0, 500));
            wav("continued.wav", cv);
        }
    }

    std::printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
