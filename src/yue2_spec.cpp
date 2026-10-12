// Speculative AR decode with prompt-lookup drafts — see yue2_internal.h.
//
// ABC text is very regular (bar lines, repeated voices, lyrics copied from
// the prompt), so what followed the last earlier match of the context's tail
// is often what comes next. Each forward feeds the token just sampled plus up
// to `draft` such tokens (Model::ar_step_rows: every weight read once for all
// rows), and the drafts are verified in order by speculative sampling with a
// point-mass draft distribution: draft d is accepted with probability p(d)
// under the stage's own distribution at its position, and the first rejected
// position draws from p with d removed (renormalised) — max(0, p - q) for q
// the point mass at d. Every emitted token is therefore distributed exactly
// as the one-token sampler's; the uniforms are consumed differently, so a
// seed draws a different (equally likely) plan. Greedy accepts d iff it is
// the argmax, which reproduces the greedy plan.

#include "yue2_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace brosoundml::yue2 {

namespace {

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("brosoundml: Yue2: " + m); }

int argmax(const std::vector<float>& sc) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(sc.size()); ++i)
        if (sc[i] > sc[best]) best = i;
    return best;
}

// softmax(scores)[i], over the finite scores (draw's arithmetic).
double prob(const std::vector<float>& sc, int i) {
    if (!std::isfinite(sc[i])) return 0.0;
    const double mx = sc[argmax(sc)];
    double z = 0;
    for (float v : sc)
        if (std::isfinite(v)) z += std::exp(static_cast<double>(v) - mx);
    return std::exp(static_cast<double>(sc[i]) - mx) / z;
}

}  // namespace

bool spec_accept(std::vector<float>& sc, int32_t d, int lo, float temperature, const UniformFn& uniform,
                 int32_t* next) {
    const int di = d - lo;
    const bool in_range = di >= 0 && di < static_cast<int>(sc.size());
    if (temperature == 0.0f) {
        if (in_range && argmax(sc) == di) return true;
        *next = lo + argmax(sc);
        return false;
    }
    if (in_range && uniform() < prob(sc, di)) return true;
    if (in_range) {
        sc[di] = -std::numeric_limits<float>::infinity();
        // p(d) rounded short of 1 with nothing else left: d it is.
        if (std::none_of(sc.begin(), sc.end(), [](float v) { return std::isfinite(v); })) return true;
    }
    *next = draw(sc, lo, temperature, uniform);
    return false;
}

std::vector<int32_t> lookup_draft(const std::vector<int32_t>& ctx, int k, int n_max, int n_min) {
    const int L = static_cast<int>(ctx.size());
    if (k <= 0) return {};
    for (int n = std::min(n_max, L - 1); n >= std::max(1, n_min); --n) {
        const int32_t* tail = ctx.data() + (L - n);
        for (int j = L - n - 1; j >= 0; --j) {
            if (!std::equal(tail, tail + n, ctx.data() + j)) continue;
            const int from = j + n, to = std::min(L, from + k);
            if (to > from) return std::vector<int32_t>(ctx.begin() + from, ctx.begin() + to);
        }
    }
    return {};
}

ArRun run_ar_spec(const Model& m, Phase ph, const std::vector<int32_t>& prefix, bool legacy_off,
                  const Yue2Sampling& s, const UniformFn& uniform, const LayerHooks* hooks,
                  const CancelCheck& cancel, const TokenFn& on_token,
                  const std::function<void(int, int)>& on_progress, std::unique_ptr<ArSession>* keep, int draft) {
    if (draft < 1 || draft > 7) fail("draft tokens must be 1..7");
    ArRun r;
    const auto t0 = Clock::now();
    if (cancel && cancel()) {
        r.cancelled = true;
        return r;
    }
    const auto [lo, hi] = logit_range(ph);
    const int n = hi - lo;
    const int P = static_cast<int>(prefix.size());
    const int cap = std::min(P + s.max_tokens, CONTEXT);
    auto sess = m.ar_session(1, std::min(cap, P + 1024), hooks, CONTEXT);   // grows on demand
    m.ar_prefill(*sess, 0, prefix, lo, hi);
    const int32_t end = end_token(ph);
    std::vector<int32_t> ctx(prefix);
    std::vector<float> logits;
    bool eos = false;
    std::vector<float> biased(n);
    auto dist = [&](const float* row) {
        const int step = static_cast<int>(r.tokens.size());
        check_finite(row, n, ph == Phase::Abc ? "plan" : "semantic");
        if (s.end_bias_after > 0 && step > s.end_bias_after) {
            std::copy(row, row + n, biased.begin());
            apply_end_bias(biased.data(), end - lo, s, step);
            row = biased.data();
        }
        return distribution(row, n, ph, s, r.tokens, step, legacy_off);
    };
    // Emit one sampled token; false when generation ends here.
    auto emit = [&](int32_t tok) {
        if (r.count == 0) r.ttft = seconds_since(t0);
        ++r.count;
        if (on_progress) on_progress(r.count, s.max_tokens);
        if (tok == end) {
            eos = true;
            return false;
        }
        r.tokens.push_back(tok);
        ctx.push_back(tok);
        if (on_token) on_token(tok, *sess);
        return r.count < s.max_tokens;
    };

    m.ar_logits(*sess, logits);
    int32_t x = draw(dist(logits.data()), lo, s.temperature, uniform);
    while (true) {
        if (!emit(x)) break;
        if (cancel && cancel()) {
            r.cancelled = true;
            break;
        }
        // Drafts: each accepted one is emitted, so at most the remaining
        // budget (which also keeps the forward inside the session).
        const auto drafts = lookup_draft(ctx, std::min(draft, s.max_tokens - r.count));
        std::vector<int32_t> feed(1, x);
        feed.insert(feed.end(), drafts.begin(), drafts.end());
        const int base = m.ar_len(*sess, 0);
        m.ar_step_rows(*sess, feed.data(), static_cast<int>(feed.size()));
        ++r.forwards;
        m.ar_logits(*sess, logits);
        int accepted = 0;
        bool stop = false, have_next = false;
        int32_t next = 0;
        for (std::size_t i = 0; i < drafts.size(); ++i) {
            auto sc = dist(logits.data() + i * static_cast<std::size_t>(n));
            const int32_t d = drafts[i];
            if (spec_accept(sc, d, lo, s.temperature, uniform, &next)) {
                ++accepted;
                if (!emit(d)) {
                    stop = true;
                    break;
                }
                continue;
            }
            have_next = true;
            break;
        }
        m.ar_truncate(*sess, 0, base + 1 + accepted);
        if (stop) break;
        if (!have_next) next = draw(dist(logits.data() + drafts.size() * static_cast<std::size_t>(n)), lo,
                                    s.temperature, uniform);
        x = next;
    }
    r.truncated = !eos && !r.cancelled;
    r.seconds = seconds_since(t0);
    if (keep) *keep = std::move(sess);
    return r;
}

}  // namespace brosoundml::yue2
