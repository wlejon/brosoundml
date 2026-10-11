#define _CRT_SECURE_NO_WARNINGS
// Speculative ABC planning (Yue2Params::abc_draft_tokens).
//
//   1. Host: spec_accept's emitted token is distributed exactly as the
//      scores' softmax, for drafts of high, low and zero probability
//      (200k trials each; total variation against the exact distribution);
//      greedy accepts only the argmax. lookup_draft on small cases.
//   2. GPU: one R-token decode step (Model::ar_step_rows) against R one-token
//      steps over the same tokens: every row's logits.
//   3. GPU: the greedy plan from the fixture's ABC prefix with drafts equals
//      the one-token greedy plan; accepted tokens per forward.
//   4. GPU: whole cot=full plans of tonight-awake at seeds 1..$YUE2_SPEC_SEEDS
//      (default 3) for each draft length in $YUE2_SPEC_DRAFTS (default
//      "0,4,7"): tokens per forward, tok/s, and the score's shape statistics
//      beside the one-token sampler's. $YUE2_AR_WEIGHTS picks the decode
//      weights (bf16 default).
// GPU parts skip without the model, the fixture or a GPU.

#include "test_yue2_common.h"
#include "test_device.h"

#include "brosoundml/detail/json.h"
#include "yue2_internal.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>

using namespace yue2_test;
namespace y2 = brosoundml::yue2;

namespace {

std::string env(const char* k, const std::string& d = {}) {
    const char* v = std::getenv(k);
    return v && *v ? std::string(v) : d;
}

void host_checks() {
    // A scores row over 1000 ids: 40 finite values, the rest removed.
    std::mt19937_64 g(7);
    std::normal_distribution<float> nd(0.0f, 1.5f);
    const float ninf = -std::numeric_limits<float>::infinity();
    std::vector<float> sc(1000, ninf);
    std::vector<int> live;
    for (int i = 0; i < 40; ++i) {
        const int id = static_cast<int>(g() % 1000);
        if (std::isfinite(sc[id])) continue;
        sc[id] = nd(g);
        live.push_back(id);
    }
    double mx = -1e30, z = 0;
    for (float v : sc) mx = std::max(mx, double(v));
    std::vector<double> p(sc.size(), 0.0);
    for (std::size_t i = 0; i < sc.size(); ++i)
        if (std::isfinite(sc[i])) z += (p[i] = std::exp(sc[i] - mx));
    for (double& v : p) v /= z;
    const int best = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
    int low = live[0];
    for (int id : live)
        if (p[id] < p[low]) low = id;
    int dead = 0;
    while (std::isfinite(sc[dead])) ++dead;
    const int lo = 5000;   // absolute ids are lo + index
    const y2::UniformFn u = y2::make_uniform(99);
    for (int d : {best, live[3], low, dead}) {
        const int trials = 200000;
        std::vector<int> hist(sc.size(), 0);
        int acc = 0;
        for (int t = 0; t < trials; ++t) {
            auto row = sc;
            int32_t next = -1;
            const bool ok = y2::spec_accept(row, lo + d, lo, 0.7f, u, &next);
            ++hist[ok ? d : next - lo];
            acc += ok;
        }
        double tv = 0;
        for (std::size_t i = 0; i < sc.size(); ++i) tv += std::fabs(double(hist[i]) / trials - p[i]);
        tv *= 0.5;
        std::printf("1. draft p=%.4f: accepted %.4f, emitted-token TV vs softmax %.4f\n", p[d], double(acc) / trials,
                    tv);
        CHECK(std::fabs(double(acc) / trials - p[d]) < 0.005, "acceptance rate for p=%.4f", p[d]);
        CHECK(tv < 0.01, "emitted distribution TV %.4f", tv);
    }
    {   // greedy
        auto row = sc;
        int32_t next = -1;
        CHECK(y2::spec_accept(row, lo + best, lo, 0.0f, u, &next), "greedy accepts the argmax");
        row = sc;
        CHECK(!y2::spec_accept(row, lo + low, lo, 0.0f, u, &next) && next == lo + best, "greedy rejects to the argmax");
    }
    {   // lookup_draft
        const std::vector<int32_t> c{1, 2, 3, 4, 5, 9, 2, 3};
        CHECK(y2::lookup_draft(c, 3) == std::vector<int32_t>({4, 5, 9}), "lookup 2-gram");
        CHECK(y2::lookup_draft(c, 1) == std::vector<int32_t>({4}), "lookup k=1");
        CHECK(y2::lookup_draft({1, 2, 3}, 4).empty(), "lookup no match");
        const std::vector<int32_t> r{7, 7, 7};
        CHECK(y2::lookup_draft(r, 2) == std::vector<int32_t>({7}), "lookup overlapping run");
    }
}

int gpu_checks(const fs::path& mdir, brotensor::Device dev) {
    const int V = y2::Config::from_json((mdir / "config.json").string()).vocab;
    Fixture fx;
    if (!read_fixture(fx, V)) {
        std::printf("SKIP: tests/fixtures/yue2_model.bin absent or unreadable\n");
        return 0;
    }
    const std::string aw = env("YUE2_AR_WEIGHTS", "bf16");
    const auto arw = aw == "int8" ? brosoundml::Yue2ArWeights::Int8
                     : aw == "fp8" ? brosoundml::Yue2ArWeights::Fp8
                                   : brosoundml::Yue2ArWeights::Bf16;
    const auto& prefix = fx.stream_prefix[0];
    const auto [lo, hi] = y2::logit_range(y2::Phase::Abc);
    const int n = hi - lo;
    {
        y2::Model m;
        m.load(mdir.string(), dev, arw);
        // 2. one R-row step vs R one-token steps.
        const std::vector<int32_t> feed{prefix[prefix.size() - 6], prefix[prefix.size() - 5], 31, 82, 91, 13};
        for (int R : {2, 6}) {
            auto a = m.ar_session(1, static_cast<int>(prefix.size()) + 16);
            auto b = m.ar_session(1, static_cast<int>(prefix.size()) + 16);
            m.ar_prefill(*a, 0, prefix, lo, hi);
            m.ar_prefill(*b, 0, prefix, lo, hi);
            m.ar_step(*a, 100);   // a graph for R = 1 exists before the R-row one
            m.ar_step(*b, 100);
            m.ar_step_rows(*a, feed.data(), R);
            std::vector<float> rows, one;
            m.ar_logits(*a, rows);
            CHECK(static_cast<int>(rows.size()) == R * n, "R-row logits size");
            double worst = 1;
            float maxd = 0;
            for (int r = 0; r < R; ++r) {
                m.ar_step(*b, feed[r]);
                m.ar_logits(*b, one);
                const std::vector<float> got(rows.begin() + r * n, rows.begin() + (r + 1) * n);
                const Diff d = diff(got, one);
                worst = std::min(worst, d.cosine);
                maxd = std::max(maxd, static_cast<float>(d.max_abs));
            }
            std::printf("2. %d-token step vs %d one-token steps (%s): worst row cosine %.7f, max |diff| %.3g\n", R, R,
                        aw.c_str(), worst, maxd);
            CHECK(worst > 0.99999, "R-row step rows differ from one-token steps");
            // The session is consistent afterwards: truncate and step again.
            m.ar_truncate(*a, 0, m.ar_len(*b, 0) - R + 1);
            m.ar_step(*a, feed[1]);
            m.ar_logits(*a, rows);
            auto c = m.ar_session(1, static_cast<int>(prefix.size()) + 16);
            m.ar_prefill(*c, 0, prefix, lo, hi);
            m.ar_step(*c, 100);
            m.ar_step(*c, feed[0]);
            m.ar_step(*c, feed[1]);
            m.ar_logits(*c, one);
            CHECK(diff(rows, one).cosine > 0.99999, "step after truncate (cosine %.7f)", diff(rows, one).cosine);
        }
        // 3. greedy with and without drafts.
        brosoundml::Yue2Params pp;
        brosoundml::Yue2Sampling s = pp.abc_sampling;
        s.temperature = 0.0f;
        s.max_tokens = std::atoi(env("YUE2_SPEC_TOKENS", "600").c_str());
        const auto r0 = y2::run_ar(m, y2::Phase::Abc, prefix, nullptr, 1.0f, false, s, y2::make_uniform(0), nullptr,
                                   {}, {}, {}, nullptr, 0);
        for (int k : {3, 7}) {
            const auto rk = y2::run_ar(m, y2::Phase::Abc, prefix, nullptr, 1.0f, false, s, y2::make_uniform(0),
                                       nullptr, {}, {}, {}, nullptr, k);
            std::size_t same = 0;
            while (same < r0.tokens.size() && same < rk.tokens.size() && r0.tokens[same] == rk.tokens[same]) ++same;
            std::printf("3. greedy plan, drafts %d: %zu/%zu tokens identical; %d tokens in %d forwards (%.2f per "
                        "forward, one-token: %d); %.1f vs %.1f tok/s\n",
                        k, same, r0.tokens.size(), rk.count, rk.forwards, double(rk.count) / std::max(1, rk.forwards),
                        r0.forwards, rk.count / rk.seconds, r0.count / r0.seconds);
            if (same < r0.tokens.size()) {
                // The multi-row step attends with other split points than
                // the one-token step (rounding only), so a greedy split is
                // allowed at a BF16 near-tie: score the split position fresh.
                std::vector<int32_t> ids = prefix;
                ids.insert(ids.end(), r0.tokens.begin(), r0.tokens.begin() + same);
                auto sess = m.ar_session(1, static_cast<int>(ids.size()) + 1);
                m.ar_prefill(*sess, 0, ids, lo, hi);
                std::vector<float> row;
                m.ar_logits(*sess, row);
                const std::vector<int32_t> hist(r0.tokens.begin(), r0.tokens.begin() + same);
                const auto sc = y2::distribution(row.data(), n, y2::Phase::Abc, s, hist, static_cast<int>(same), false);
                const float a = sc[r0.tokens[same] - lo], b = sc[rk.tokens[same] - lo];
                const float ulp =
                    std::ldexp(1.0f, static_cast<int>(std::floor(std::log2(std::fabs(a) + 1e-30f))) - 7);
                std::printf("   split at %zu: one-token pick scores %.4f, drafted pick %.4f (%.1f BF16 ulp)\n", same,
                            a, b, (b - a) / ulp);
                CHECK(std::fabs(b - a) <= 2 * ulp, "greedy plan with drafts %d splits at %zu off a near-tie", k,
                      same);
            }
        }
    }
    // 4. whole plans.
    std::ifstream rf((mdir / "examples" / "tonight-awake.json").string(), std::ios::binary);
    std::stringstream ss;
    ss << rf.rdbuf();
    const auto req = brosoundml::detail::json::parse(ss.str());
    brosoundml::Yue2 y;
    y.load(mdir.string(), dev, {arw});
    brosoundml::Yue2Params p;
    p.style = req.get_string("style", "");
    p.lyrics = req.get_string("lyrics", "");
    p.cot = brosoundml::Yue2Cot::Full;
    p.run_semantic = false;
    const int S = std::atoi(env("YUE2_SPEC_SEEDS", "3").c_str());
    std::stringstream ks(env("YUE2_SPEC_DRAFTS", "0,4,7"));
    std::string item;
    while (std::getline(ks, item, ',')) {
        p.abc_draft_tokens = std::atoi(item.c_str());
        PlanStats sum;
        int fw = 0;
        double secs = 0;
        for (int seed = 1; seed <= S; ++seed) {
            p.seed = static_cast<std::uint64_t>(seed);
            const auto r = y.generate(p);
            const auto st = plan_stats(r.plan.abc, static_cast<int>(r.plan.abc_ids.size()), r.plan.truncated,
                                       r.timing.plan_tps);
            std::printf("4. drafts %d seed %d: %d tokens%s in %d forwards (%.2f per forward), %d header fields, %d "
                        "bars, %d V: lines, %.1f tok/s\n",
                        p.abc_draft_tokens, seed, st.tokens, st.truncated ? " (truncated)" : "",
                        r.timing.plan_forwards, double(r.timing.plan_tokens) / std::max(1, r.timing.plan_forwards),
                        st.headers, st.bars, st.voices, st.tps);
            sum.tokens += st.tokens;
            sum.truncated += st.truncated;
            sum.headers += st.headers;
            sum.bars += st.bars;
            sum.voices += st.voices;
            fw += r.timing.plan_forwards;
            secs += r.timing.plan_seconds;
        }
        std::printf("4. drafts %d (%s) over %d seeds: mean %.0f tokens, %d truncated, %.1f header fields, %.0f bars, "
                    "%.1f V: lines; %.2f tokens per forward, %.1f tok/s\n",
                    p.abc_draft_tokens, aw.c_str(), S, double(sum.tokens) / S, sum.truncated,
                    double(sum.headers) / S, double(sum.bars) / S, double(sum.voices) / S,
                    double(sum.tokens + S) / std::max(1, fw), (sum.tokens + S) / secs);
    }
    return 0;
}

}  // namespace

int main() {
    host_checks();
    brotensor::init();
    const fs::path mdir = model_dir();
    const brotensor::Device dev = brosoundml_test::preferred_gpu();
    if (mdir.empty() || !dev.is_gpu()) {
        std::printf("SKIP GPU parts: no YuE2-3B folder or GPU\n");
    } else {
        try {
            gpu_checks(mdir, dev);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "exception: %s\n", e.what());
            return 1;
        }
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
