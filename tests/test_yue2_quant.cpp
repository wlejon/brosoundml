#define _CRT_SECURE_NO_WARNINGS
// YuE2-3B quantised AR decode (Yue2ArWeights Int8 / Fp8) against BF16, on the
// GPU. Each format is loaded in turn (BF16 first, as the reference), and:
//
//   A. Decode logits: each fixture prefix (tests/fixtures/yue2_model.bin)
//      prefilled up to its last token, which the decode step then feeds;
//      that row against the reference package's logits and against BF16's
//      decode: cosine, top-10 overlap, argmax.
//   B. Teacher-forced streams: BF16's greedy ABC and semantic streams
//      (fixture prefixes, $YUE2_QUANT_TOKENS tokens, default 400) fed through
//      each format; per position, logits cosine and top-10 overlap, and the
//      total variation between the stage's sampling distributions (its preset:
//      temperature, top-k, top-p, penalty) — 1 - TV is the chance one draw
//      picks the same token under both.
//   C. Greedy divergence: each format's own greedy rollout from the same
//      prefixes; the first token where it leaves BF16's.
//   D. Whole plans: Yue2::plan for the tonight-awake request at seeds
//      1..$YUE2_QUANT_SEEDS (default 3): token count, truncation, ABC header
//      fields, bars, voices, and the decode rate.
// The gates are loose (they catch a broken format, not a small drift); the
// numbers are the report. Skips without the model, the fixture or a GPU.
// $YUE2_QUANT_FORMATS (e.g. "int8") narrows the formats.

#include "test_yue2_common.h"
#include "test_device.h"

#include "brosoundml/detail/json.h"
#include "yue2_internal.h"

#include <algorithm>
#include <numeric>
#include <sstream>

using namespace yue2_test;
namespace y2 = brosoundml::yue2;
using brosoundml::Yue2ArWeights;

namespace {

const char* fmt_name(Yue2ArWeights f) {
    return f == Yue2ArWeights::Bf16 ? "bf16" : f == Yue2ArWeights::Int8 ? "int8" : "fp8";
}

std::vector<int> topk(const float* v, int n, int k) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}
int overlap(const float* a, const float* b, int n, int k) {
    const auto ta = topk(a, n, k), tb = topk(b, n, k);
    int o = 0;
    for (int x : ta) o += std::count(tb.begin(), tb.end(), x) ? 1 : 0;
    return o;
}
double cosine(const float* a, const float* b, int n) {
    double d = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) d += double(a[i]) * b[i], na += double(a[i]) * a[i], nb += double(b[i]) * b[i];
    return d / std::sqrt(na * nb + 1e-30);
}
// softmax over the non -inf scores
std::vector<double> probs(const std::vector<float>& sc) {
    double mx = -1e30;
    for (float v : sc) mx = std::max(mx, double(v));
    std::vector<double> p(sc.size(), 0.0);
    double z = 0;
    for (std::size_t i = 0; i < sc.size(); ++i)
        if (std::isfinite(sc[i])) z += (p[i] = std::exp(sc[i] - mx));
    for (double& v : p) v /= z;
    return p;
}

// Per format, per phase: the logits rows along BF16's greedy tokens and the
// format's own greedy tokens.
struct PhaseRun {
    std::vector<std::vector<float>> logits;   // teacher-forced rows
    std::vector<int32_t> greedy;
};

brosoundml::Yue2Sampling preset(y2::Phase ph) {
    brosoundml::Yue2Params p;
    return ph == y2::Phase::Abc ? p.abc_sampling : p.semantic_sampling;
}

// Greedy rollout of n tokens (no end token: masked), recording each row.
std::vector<int32_t> rollout(const y2::Model& m, y2::Phase ph, const std::vector<int32_t>& prefix,
                             const std::vector<int32_t>* forced, int n, std::vector<std::vector<float>>* rows) {
    const auto [lo, hi] = y2::logit_range(ph);
    auto s = m.ar_session(1, static_cast<int>(prefix.size()) + n + 1);
    m.ar_prefill(*s, 0, prefix, lo, hi);
    brosoundml::Yue2Sampling g = preset(ph);
    g.temperature = 0.0f;
    g.min_tokens = n + 1;
    std::vector<int32_t> toks;
    std::vector<float> row;
    for (int i = 0; i < n; ++i) {
        m.ar_logits(*s, row);
        if (rows) rows->push_back(row);
        int32_t t;
        if (forced) {
            t = (*forced)[i];
        } else {
            const auto sc = y2::distribution(row.data(), hi - lo, ph, g, toks, i, false);
            t = static_cast<int32_t>(std::max_element(sc.begin(), sc.end()) - sc.begin()) + lo;
        }
        toks.push_back(t);
        if (i + 1 < n) m.ar_step(*s, t);
    }
    return toks;
}

}  // namespace

int main() {
    brotensor::init();
    const fs::path mdir = model_dir();
    if (mdir.empty()) {
        std::printf("SKIP: no YuE2-3B folder\n");
        return 0;
    }
    const brotensor::Device dev = brosoundml_test::preferred_gpu();
    if (!dev.is_gpu()) {
        std::printf("SKIP: no GPU backend\n");
        return 0;
    }
    const int V = y2::Config::from_json((mdir / "config.json").string()).vocab;
    Fixture fx;
    if (!read_fixture(fx, V)) {
        std::printf("SKIP: tests/fixtures/yue2_model.bin absent or unreadable\n");
        return 0;
    }
    const char* ev = std::getenv("YUE2_QUANT_TOKENS");
    const int N = ev && *ev ? std::atoi(ev) : 400;
    const char* es = std::getenv("YUE2_QUANT_SEEDS");
    const int S = es && *es ? std::atoi(es) : 3;
    const char* ef = std::getenv("YUE2_QUANT_FORMATS");
    const std::string want = ef && *ef ? ef : "int8,fp8";

    std::ifstream rf((mdir / "examples" / "tonight-awake.json").string(), std::ios::binary);
    std::stringstream ss;
    ss << rf.rdbuf();
    const auto req = brosoundml::detail::json::parse(ss.str());

    std::vector<std::vector<float>> bf_a;   // A: BF16 decode rows
    PhaseRun bf[2];
    std::vector<PlanStats> bf_plans;
    try {
        for (Yue2ArWeights f : {Yue2ArWeights::Bf16, Yue2ArWeights::Int8, Yue2ArWeights::Fp8}) {
            if (f != Yue2ArWeights::Bf16 && want.find(fmt_name(f)) == std::string::npos) continue;
            const bool ref = f == Yue2ArWeights::Bf16;
            std::printf("── %s ──\n", fmt_name(f));
            {
                y2::Model m;
                m.load(mdir.string(), dev, f);
                // A.
                for (std::size_t i = 0; i < fx.prefixes.size(); ++i) {
                    const auto& ids = fx.prefixes[i];
                    std::vector<int32_t> head(ids.begin(), ids.end() - 1);
                    auto s = m.ar_session(1, static_cast<int>(ids.size()) + 1);
                    m.ar_prefill(*s, 0, head, 0, V);
                    m.ar_step(*s, ids.back());
                    std::vector<float> row;
                    m.ar_logits(*s, row);
                    const auto& rl = fx.ref_logits[i];
                    const int am = static_cast<int>(std::max_element(row.begin(), row.end()) - row.begin());
                    const int ar = static_cast<int>(std::max_element(rl.begin(), rl.end()) - rl.begin());
                    std::printf("A. decode logits [%zu tokens]: vs reference cosine %.6f, top10 %d, argmax %s",
                                ids.size(), cosine(row.data(), rl.data(), V), overlap(row.data(), rl.data(), V, 10),
                                am == ar ? "same" : "differs");
                    if (ref) {
                        bf_a.push_back(row);
                        std::printf("\n");
                    } else {
                        const auto& b = bf_a[i];
                        std::printf("; vs bf16 cosine %.6f, top10 %d\n", cosine(row.data(), b.data(), V),
                                    overlap(row.data(), b.data(), V, 10));
                        CHECK(overlap(row.data(), rl.data(), V, 10) >= 7, "%s A%zu top-10", fmt_name(f), i);
                        CHECK(cosine(row.data(), rl.data(), V) > 0.995, "%s A%zu cosine", fmt_name(f), i);
                    }
                }
                // B + C.
                for (int k = 0; k < 2; ++k) {
                    const y2::Phase ph = k == 0 ? y2::Phase::Abc : y2::Phase::Semantic;
                    const auto [lo, hi] = y2::logit_range(ph);
                    const int n = hi - lo;
                    const auto& prefix = fx.stream_prefix[k];
                    const auto t0 = y2::Clock::now();
                    if (ref) {
                        bf[k].greedy = rollout(m, ph, prefix, nullptr, N, &bf[k].logits);
                        std::printf("%s: BF16 greedy %d tokens in %.2f s\n", k ? "semantic" : "abc", N,
                                    y2::seconds_since(t0));
                        continue;
                    }
                    PhaseRun q;
                    rollout(m, ph, prefix, &bf[k].greedy, N, &q.logits);
                    q.greedy = rollout(m, ph, prefix, nullptr, N, nullptr);
                    const auto sp = preset(ph);
                    double cs = 0, tv = 0, tv_max = 0, cmin = 1;
                    int ov = 0;
                    for (int i = 0; i < N; ++i) {
                        const float* a = bf[k].logits[i].data();
                        const float* b = q.logits[i].data();
                        const double c = cosine(a, b, n);
                        cs += c;
                        cmin = std::min(cmin, c);
                        ov += overlap(a, b, n, 10);
                        const std::vector<int32_t> hist(bf[k].greedy.begin(), bf[k].greedy.begin() + i);
                        const auto pa = probs(y2::distribution(a, n, ph, sp, hist, i, false));
                        const auto pb = probs(y2::distribution(b, n, ph, sp, hist, i, false));
                        double d = 0;
                        for (int j = 0; j < n; ++j) d += std::fabs(pa[j] - pb[j]);
                        tv += 0.5 * d;
                        tv_max = std::max(tv_max, 0.5 * d);
                    }
                    int split = 0;
                    while (split < N && q.greedy[split] == bf[k].greedy[split]) ++split;
                    std::printf("B. %s teacher-forced %d tokens: logits cosine mean %.6f (min %.6f), top10 %.2f/10; "
                                "sampling TV mean %.4f (max %.3f)\n",
                                k ? "semantic" : "abc", N, cs / N, cmin, double(ov) / N, tv / N, tv_max);
                    std::printf("C. %s greedy: first divergence from bf16 at token %d of %d\n",
                                k ? "semantic" : "abc", split, N);
                    CHECK(cs / N > 0.99, "%s %s mean cosine", fmt_name(f), k ? "semantic" : "abc");
                    CHECK(tv / N < 0.25, "%s %s mean TV", fmt_name(f), k ? "semantic" : "abc");
                }
            }
            // D.
            {
                brosoundml::Yue2 y;
                y.load(mdir.string(), dev, {f});
                brosoundml::Yue2Params p;
                p.style = req.get_string("style", "");
                p.lyrics = req.get_string("lyrics", "");
                p.cot = brosoundml::Yue2Cot::Full;
                PlanStats sum;
                for (int seed = 1; seed <= S; ++seed) {
                    p.seed = static_cast<std::uint64_t>(seed);
                    const auto t0 = y2::Clock::now();
                    const auto plan = y.plan(p);
                    const double secs = y2::seconds_since(t0);
                    const PlanStats st = plan_stats(plan.abc, static_cast<int>(plan.abc_ids.size()), plan.truncated,
                                                    plan.abc_ids.size() / secs);
                    std::printf("D. plan seed %d: %d tokens%s, %d header fields, %d bars, %d V: lines, %.1f tok/s\n",
                                seed, st.tokens, st.truncated ? " (truncated)" : "", st.headers, st.bars, st.voices,
                                st.tps);
                    sum.tokens += st.tokens;
                    sum.truncated += st.truncated;
                    sum.headers += st.headers;
                    sum.bars += st.bars;
                    sum.voices += st.voices;
                    sum.tps += st.tps;
                    if (ref) bf_plans.push_back(st);
                }
                std::printf("D. %s over %d seeds: mean %.0f tokens, %d truncated, %.1f header fields, %.0f bars, "
                            "%.1f V: lines, %.1f tok/s\n",
                            fmt_name(f), S, double(sum.tokens) / S, sum.truncated, double(sum.headers) / S,
                            double(sum.bars) / S, double(sum.voices) / S, sum.tps / S);
                if (!ref) CHECK(sum.truncated == 0 || sum.truncated <= S / 2, "%s: plans truncated", fmt_name(f));
            }
        }
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
