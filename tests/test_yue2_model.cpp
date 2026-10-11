#define _CRT_SECURE_NO_WARNINGS
// YuE2-3B backbone parity on the GPU, against tests/fixtures/yue2_model.bin
// (tests/ref/gen_yue2_fixture.py, the reference package on CUDA, BF16):
//
//   1. AR logits of the last row for three real prompt prefixes (plan, a
//      37-token cut of it, the semantic prefix): argmax, top-k overlap, cosine.
//   2. Greedy (temperature 0) token streams through the session decode (CUDA
//      graph): ABC planning, semantic, and the cot=off CFG pair (legacy BF16).
//   3. NAR: one velocity on the chunk's noise at t = 0.75, then the full
//      32-step midpoint solve from that noise.
//   4. The hidden-state seam: a capture reports the residual; a whole-sequence
//      steer of zero changes nothing; a steer moves the captured residual of
//      the next layer and the logits.
// Skips without the fixture, the model folder, or a GPU.

#include "test_yue2_common.h"
#include "test_device.h"

#include "yue2_internal.h"

#include <algorithm>
#include <numeric>

using namespace yue2_test;
namespace y2 = brosoundml::yue2;

static int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

static std::vector<int> topk(const std::vector<float>& v, int k) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

static std::vector<float> prefill_logits(const y2::Model& m, const std::vector<int32_t>& ids, int lo, int hi,
                                         const y2::LayerHooks* hooks = nullptr) {
    auto s = m.ar_session(1, static_cast<int>(ids.size()) + 1, hooks);
    m.ar_prefill(*s, 0, ids, lo, hi);
    std::vector<float> out;
    m.ar_logits(*s, out);
    return out;
}

static int run(const fs::path& mdir, brotensor::Device dev) {
    Reader r(fixture("yue2_model.bin"));
    if (!r) {
        std::printf("SKIP: tests/fixtures/yue2_model.bin absent (run tests/ref/gen_yue2_fixture.py)\n");
        return 0;
    }
    CHECK(r.i32() == 0x314d5559, "model fixture magic");
    y2::Model m;
    auto t0 = y2::Clock::now();
    m.load(mdir.string(), dev);
    std::printf("load: %.2f s\n", y2::seconds_since(t0));
    const int V = m.config().vocab;

    // 1. AR logits
    const int nl = r.i32();
    std::vector<int32_t> sem_prefix;
    for (int i = 0; i < nl; ++i) {
        const auto ids = r.i32list();
        const auto ref = r.f32arr(V);
        if (i == nl - 1) sem_prefix = ids;
        t0 = y2::Clock::now();
        const auto got = prefill_logits(m, ids, 0, V);
        const double secs = y2::seconds_since(t0);
        const Diff d = diff(got, ref);
        const auto tg = topk(got, 10), tr = topk(ref, 10);
        int overlap = 0;
        for (int a : tg) overlap += std::count(tr.begin(), tr.end(), a) ? 1 : 0;
        std::printf("logits[%zu tokens]: argmax %d/%d, top10 overlap %d, cosine %.6f, max |diff| %.3g "
                    "(ref rms %.3g), prefill %.1f ms\n",
                    ids.size(), argmax(got), argmax(ref), overlap, d.cosine, d.max_abs, d.ref_rms, secs * 1e3);
        CHECK(argmax(got) == argmax(ref), "logits %d: argmax", i);
        CHECK(overlap >= 8, "logits %d: top-10 overlap %d", i, overlap);
        CHECK(d.cosine > 0.999, "logits %d: cosine %.6f", i, d.cosine);
        // The restricted-range rows are the same rows.
        const auto [lo, hi] = y2::logit_range(y2::Phase::Semantic);
        const auto part = prefill_logits(m, ids, lo, hi);
        const std::vector<float> full_part(got.begin() + lo, got.begin() + hi);
        CHECK(diff(part, full_part).max_abs < 0.07, "logits %d: range rows differ (%.3g)", i,
              diff(part, full_part).max_abs);
    }

    // 2. greedy streams
    for (int k = 0; k < 3; ++k) {
        const y2::Phase ph = r.i32() == 0 ? y2::Phase::Abc : y2::Phase::Semantic;
        const auto prefix = r.i32list(), neg = r.i32list();
        const float cfg = r.f32();
        const bool legacy = r.i32() != 0;
        brosoundml::Yue2Sampling s;
        s.temperature = r.f32();
        s.top_p = r.f32();
        s.top_k = r.i32();
        s.repetition_penalty = r.f32();
        s.penalty_window = r.i32();
        s.min_tokens = r.i32();
        s.max_tokens = r.i32();
        const auto ref = r.i32list();
        const bool trunc = r.i32() != 0;
        const auto run = y2::run_ar(m, ph, prefix, cfg != 1.0f ? &neg : nullptr, cfg, legacy, s, y2::make_uniform(0),
                                    nullptr, {}, {}, {}, nullptr);
        std::size_t same = 0;
        while (same < run.tokens.size() && same < ref.size() && run.tokens[same] == ref[same]) ++same;
        const double rate = run.count > 1 ? (run.count - 1) / std::max(1e-9, run.seconds - run.ttft) : 0;
        std::printf("greedy %s%s: %zu/%zu tokens identical (ours %zu, truncated %d/%d), TTFT %.1f ms, "
                    "%.1f tok/s\n",
                    ph == y2::Phase::Abc ? "abc" : "semantic", cfg != 1.0f ? " cfg" : "", same, ref.size(),
                    run.tokens.size(), run.truncated, trunc, run.ttft * 1e3, rate);
        if (same < ref.size()) {
            // Teacher-force the reference's tokens up to the split and look at
            // the greedy scores there: a split at a BF16 near-tie (the margin
            // within a few BF16 ulps of the scores) is GEMM accumulation order,
            // not a wrong forward.
            std::vector<int32_t> ids = prefix, nids = neg;
            ids.insert(ids.end(), ref.begin(), ref.begin() + same);
            nids.insert(nids.end(), ref.begin(), ref.begin() + same);
            const auto [lo, hi] = y2::logit_range(ph);
            const int n = hi - lo;
            auto row = prefill_logits(m, ids, lo, hi);
            if (cfg != 1.0f) {
                const auto un = prefill_logits(m, nids, lo, hi);
                std::vector<float> comb(n);
                y2::cfg_combine_bf16(row.data(), un.data(), cfg, comb.data(), n);
                row = comb;
            }
            const std::vector<int32_t> hist(ref.begin(), ref.begin() + same);
            const auto sc = y2::distribution(row.data(), n, ph, s, hist, static_cast<int>(same), legacy);
            const float a = sc[ref[same] - lo], b = sc[run.tokens[same] - lo];
            const float ulp = std::ldexp(1.0f, static_cast<int>(std::floor(std::log2(std::fabs(a) + 1e-30f))) - 7);
            std::printf("  split at %zu: ref token %d scores %.4f, ours %d scores %.4f (gap %.4f = %.1f BF16 ulp)\n",
                        same, ref[same], a, run.tokens[same], b, b - a, (b - a) / ulp);
            CHECK(std::fabs(b - a) <= 2 * ulp, "greedy stream %d splits at %zu by more than 2 BF16 ulp", k, same);
        } else {
            CHECK(run.tokens.size() == ref.size(), "greedy stream %d length", k);
        }
    }

    // 3. NAR
    {
        const auto ar = r.i32list();
        const int frames = r.i32();
        const auto noise = r.f32arr(static_cast<std::size_t>(frames) * 64);
        const double raw = r.f64();
        const auto vref = r.f32arr(static_cast<std::size_t>(frames) * 64);
        const int steps = r.i32();
        const auto lref = r.f32arr(static_cast<std::size_t>(frames) * 64);
        auto c = m.nar_chunk(ar, frames, {});
        const auto v = m.nar_velocity(*c, noise, raw);
        const Diff dv = diff(v, vref);
        std::printf("nar velocity (%zu AR tokens, %d frames): cosine %.6f, max |diff| %.3g, rms %.3g "
                    "(ref rms %.3g)\n",
                    ar.size(), frames, dv.cosine, dv.max_abs, dv.rms, dv.ref_rms);
        CHECK(dv.cosine > 0.999, "nar velocity cosine %.6f", dv.cosine);
        t0 = y2::Clock::now();
        const auto lat = m.nar_solve(*c, noise.data(), steps, {}, {});
        const double secs = y2::seconds_since(t0);
        const Diff dl = diff(lat, lref);
        std::printf("nar solve (%d steps): cosine %.6f, max |diff| %.3g, rms %.3g (ref rms %.3g), %.1f ms "
                    "= %.3f s per audio s\n",
                    steps, dl.cosine, dl.max_abs, dl.rms, dl.ref_rms, secs * 1e3, secs / (frames / 25.0));
        CHECK(dl.cosine > 0.995, "nar solve cosine %.6f", dl.cosine);
        // A chunk prefilled from a session's cache equals a fresh one.
        std::vector<int32_t> pre(ar.begin(), ar.end() - 1);
        auto sess = m.ar_session(1, static_cast<int>(ar.size()));
        m.ar_prefill(*sess, 0, pre, 0, 1);
        y2::Model::NarOptions o;
        o.reuse = sess.get();
        o.reuse_rows = static_cast<int>(pre.size());
        auto c2 = m.nar_chunk(ar, frames, o);
        // (The reused rows come from the session's prefill and the last row
        // from the single-row decode kernel, so they agree to BF16 rounding.)
        const Diff dr = diff(m.nar_velocity(*c2, noise, raw), v);
        std::printf("nar chunk from a session cache vs fresh: cosine %.7f, max |diff| %.3g\n", dr.cosine, dr.max_abs);
        CHECK(dr.cosine > 0.9999, "nar chunk from a reused cache differs (cosine %.7f)", dr.cosine);
    }

    // 4. hidden-state seam
    {
        const int H = m.config().hidden, L = 7;
        const auto [lo, hi] = y2::logit_range(y2::Phase::Semantic);
        const auto base = prefill_logits(m, sem_prefix, lo, hi);
        std::vector<std::vector<float>> caps(2);
        y2::LayerHooks hk;
        hk.capture.assign(m.config().layers, 0);
        hk.capture[L] = hk.capture[L + 1] = 1;
        int rows_seen = 0;
        hk.on_capture = [&](int layer, int pos0, int rows, int eval, const float* d) {
            CHECK(eval == -1 && pos0 == 0, "capture on the AR path");
            rows_seen = rows;
            caps[layer - L].assign(d + static_cast<std::size_t>(rows - 1) * H, d + static_cast<std::size_t>(rows) * H);
        };
        const auto cap_logits = prefill_logits(m, sem_prefix, lo, hi, &hk);
        CHECK(rows_seen == static_cast<int>(sem_prefix.size()), "capture rows %d", rows_seen);
        CHECK(diff(cap_logits, base).max_abs == 0.0, "capture changed the logits");
        const auto cap0 = caps;
        hk.steer.assign(m.config().layers, {});
        hk.steer[L].push_back(m.make_steer(std::vector<float>(H, 0.0f), 1.0f, 0, false));
        CHECK(diff(prefill_logits(m, sem_prefix, lo, hi, &hk), base).max_abs == 0.0, "a zero steer changed logits");
        std::vector<float> dir(H);
        for (int i = 0; i < H; ++i) dir[i] = (i % 7 == 0) ? 1.0f : -0.25f;
        hk.steer[L][0] = m.make_steer(dir, 4.0f, 0, false);
        const auto steered = prefill_logits(m, sem_prefix, lo, hi, &hk);
        double moved = 0;
        // BF16 residual: the add rounds at the residual's magnitude.
        for (int i = 0; i < H; ++i)
            moved = std::max(moved, std::fabs(caps[0][i] - cap0[0][i] - 4.0 * dir[i]) /
                                        (1.0 + std::fabs(static_cast<double>(cap0[0][i]))));
        std::printf("seam: steered residual at layer %d off by %.3g from +delta; next layer moved %.3g; "
                    "logits moved %.3g\n",
                    L, moved, diff(caps[1], cap0[1]).max_abs, diff(steered, base).max_abs);
        CHECK(moved < 0.02, "steered residual = captured + delta (%.3g)", moved);
        CHECK(diff(steered, base).max_abs > 0.1, "steer moved the logits");
    }
    // 5. decode cost against context length (timings only): a semantic
    // session prefilled with the prefix plus filler codec tokens, then 32
    // steps, each read back like the sampler does.
    for (int ctx : {600, 3000, 7000}) {
        std::vector<int32_t> ids = sem_prefix;
        for (int i = 0; static_cast<int>(ids.size()) < ctx; ++i) ids.push_back(y2::CODEC_OFFSET + (i * 7919) % 32768);
        const auto [lo, hi] = y2::logit_range(y2::Phase::Semantic);
        auto s = m.ar_session(1, ctx + 64);
        t0 = y2::Clock::now();
        m.ar_prefill(*s, 0, ids, lo, hi);
        std::vector<float> lg;
        m.ar_logits(*s, lg);
        const double pre = y2::seconds_since(t0);
        m.ar_step(*s, y2::CODEC_OFFSET + 1);   // warm-up + graph capture
        m.ar_logits(*s, lg);
        t0 = y2::Clock::now();
        for (int i = 0; i < 32; ++i) {
            m.ar_step(*s, y2::CODEC_OFFSET + 2 + i);
            m.ar_logits(*s, lg);
        }
        const double per = y2::seconds_since(t0) / 32;
        std::printf("decode at context %d: prefill %.1f ms, %.2f ms per token (%.1f tok/s)\n", ctx, pre * 1e3,
                    per * 1e3, 1.0 / per);
    }
    return 0;
}

int main() {
    brotensor::init();
    const fs::path mdir = model_dir();
    if (mdir.empty()) {
        std::printf("SKIP: no YuE2-3B folder ($YUE2_MODEL_DIR or ../YuE2-3B)\n");
        return 0;
    }
    const brotensor::Device dev = brosoundml_test::preferred_gpu();
    if (!dev.is_gpu()) {
        std::printf("SKIP: no GPU backend\n");
        return 0;
    }
    try {
        run(mdir, dev);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
