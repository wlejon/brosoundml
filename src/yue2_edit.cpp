// Yue2 editing — splice() (new semantic tokens between kept ones) and
// inpaint() (new latents for the frames that changed), see yue2.h.

#include "yue2_internal.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace brosoundml {

using namespace yue2;

namespace {

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("brosoundml: Yue2: " + m); }

void check_codes(const std::vector<int32_t>& v, const char* what) {
    for (int32_t c : v)
        if (c < 0 || c >= CODEC_SIZE) fail(std::string(what) + ": codec id outside 0..32767");
}

// -log softmax(row)[i] over the n logits of a row.
double nll_of(const float* row, int n, int i) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < n; ++k) mx = std::max(mx, row[k]);
    double sum = 0;
    for (int k = 0; k < n; ++k) sum += std::exp(static_cast<double>(row[k] - mx));
    return mx + std::log(sum) - row[i];
}

// One semantic AR session over A, sampling and scoring the way run_ar does.
struct Sampler {
    const Model& m;
    Yue2Sampling s;
    float cfg = 1.0f;
    bool use_cfg = false, legacy_off = false;
    int lo = 0, n = 0;
    int32_t end = 0;
    int B = 1;          // branches
    int rows_max = 1;   // tokens one forward may feed (ar_step_rows)
    std::unique_ptr<ArSession> sess;
    std::vector<float> logits, combined;
    int logit_rows = 1;   // rows per branch in `logits`

    Sampler(const Model& model) : m(model) {}

    // Row r (of logit_rows) as the sampler sees it: CFG-combined when on.
    float* row(int r) {
        if (!use_cfg) return logits.data() + static_cast<std::size_t>(r) * n;
        combined.resize(n);
        cfg_combine_bf16(logits.data() + static_cast<std::size_t>(r) * n,
                         logits.data() + static_cast<std::size_t>(logit_rows + r) * n, cfg, combined.data(), n);
        return combined.data();
    }
    void step(int32_t tok) {
        m.ar_step(*sess, tok);
        m.ar_logits(*sess, logits);
        logit_rows = 1;
    }
    // Feed `count` tokens, rows_max per forward; `on_rows(first, R)` sees
    // each forward's rows (row r predicts the token after toks[first + r]).
    template <class F>
    void feed(const int32_t* toks, int count, F&& on_rows) {
        for (int j = 0; j < count;) {
            const int R = std::min(rows_max, count - j);
            m.ar_step_rows(*sess, toks + j, R);
            m.ar_logits(*sess, logits);
            logit_rows = R;
            on_rows(j, R);
            j += R;
        }
    }
    int32_t sample(float* r, const std::vector<int32_t>& hist, int pos, bool forbid_end, const UniformFn& u) {
        check_finite(r, n, "semantic");
        apply_end_bias(r, end - lo, s, pos);
        if (forbid_end) r[end - lo] = -std::numeric_limits<float>::infinity();
        const auto scores = distribution(r, n, Phase::Semantic, s, hist, pos, legacy_off);
        return draw(scores, lo, s.temperature, u);
    }
};

}  // namespace

Yue2SpliceResult Yue2::splice(const Yue2Params& p, const Yue2Plan& plan, const Yue2Splice& sp,
                              const Yue2Callbacks& cb, const CancelCheck& cancel) const {
    if (!loaded()) fail("not loaded");
    const Impl& I = *impl_;
    check_plan(I, p, plan);
    check_codes(sp.before, "splice before");
    check_codes(sp.after, "splice after");
    if (sp.middle_frames < 0) fail("splice: middle_frames must be >= 0");
    if (sp.candidates < 1 || sp.candidates > 32) fail("splice: candidates must be 1..32");
    if (sp.score_frames < 1) fail("splice: score_frames must be >= 1");
    const Model& m = I.model;
    brotensor::DeviceScope scope(m.device());
    const Encode enc = [&](const std::string& s) { return I.encode(s); };

    Sampler S(m);
    S.s = guarded_semantic(p, plan.abc);
    validate(S.s, "semantic");
    S.cfg = guidance(p);
    S.use_cfg = S.cfg != 1.0f;
    S.legacy_off = p.cot == Yue2Cot::Off;
    const auto [lo, hi] = logit_range(Phase::Semantic);
    S.lo = lo;
    S.n = hi - lo;
    S.end = end_token(Phase::Semantic);
    S.B = S.use_cfg ? 2 : 1;
    S.rows_max = (m.device() == brotensor::Device::CUDA || m.device() == brotensor::Device::CPU) ? 8 : 1;

    const int A = static_cast<int>(sp.before.size()), M = sp.middle_frames;
    const int Bn = static_cast<int>(sp.after.size());
    const int kept = A + M + Bn;
    if (kept > S.s.max_tokens) fail("splice: the song is longer than the semantic token budget");
    std::vector<int32_t> neg;
    if (S.use_cfg) neg = negative_prefix(p, enc, plan.abc_ids);
    const int longest = static_cast<int>(std::max(plan.prefix.size(), neg.size()));
    if (longest + kept > CONTEXT) fail("splice: prefix + song exceeds the 24576-token context");

    const StageHooks hooks = make_hooks(m, p, Yue2Stage::Semantic, cb);
    S.sess = m.ar_session(S.B, std::min(CONTEXT, longest + kept + 1024), hooks.get(), CONTEXT);
    std::vector<int32_t> hist;   // the song so far, absolute ids (the repetition penalty's history)
    for (int32_t c : sp.before) hist.push_back(c + CODEC_OFFSET);
    // A branch's prefill is queued: poll `cancel` every 2 ms until the device
    // is done rather than block (a caller feeding audio keeps going through a
    // long prefix). Each branch waits on its own: the next branch's queueing
    // blocks on the one before it (its uploads wait their turn on the stream).
    auto settled = [&] {
        if (!cancel) return true;
        auto next = std::chrono::steady_clock::now();
        while (!brotensor::idle(m.device())) {
            const auto now = std::chrono::steady_clock::now();
            if (now < next) {
                std::this_thread::yield();
                continue;
            }
            if (cancel()) return false;
            next = now + std::chrono::milliseconds(2);
        }
        return true;
    };
    for (int b = 0; b < S.B; ++b) {
        std::vector<int32_t> ids(b == 0 ? plan.prefix : neg);
        ids.insert(ids.end(), hist.begin(), hist.end());
        m.ar_prefill(*S.sess, b, ids, lo, hi);
        if (!settled()) {
            Yue2SpliceResult r;
            r.cancelled = true;
            return r;
        }
    }
    m.ar_logits(*S.sess, S.logits);
    S.logit_rows = 1;
    const std::vector<float> base_logits = S.logits;
    std::vector<int> base_len(S.B);
    for (int b = 0; b < S.B; ++b) base_len[b] = m.ar_len(*S.sess, b);

    Yue2SpliceResult res;
    res.middle0 = A;
    res.after0 = A + M;
    const int K = (Bn > 0 && M > 0) ? sp.candidates : 1;
    const int scored = std::min(sp.score_frames, Bn);
    const int total = K * (M + scored);
    int done = 0;
    auto progress = [&] {
        ++done;
        if (cb.on_progress) cb.on_progress(Yue2Stage::Semantic, done, std::max(total, done));
    };
    auto cancelled = [&] { return cancel && cancel(); };
    std::vector<int32_t> after_abs;
    for (int32_t c : sp.after) after_abs.push_back(c + CODEC_OFFSET);
    int held = -1;   // the candidate whose M + B[0, scored - 1) the session holds
    double best = std::numeric_limits<double>::infinity();
    auto stage_u = std::make_shared<std::size_t>(0);
    for (int k = 0; k < K; ++k) {
        if (k > 0) {
            for (int b = 0; b < S.B; ++b) m.ar_truncate(*S.sess, b, base_len[b]);
            S.logits = base_logits;
            S.logit_rows = 1;
        }
        const UniformFn u = k == 0 ? stage_uniform(p, stage_u) : make_uniform(p.seed + static_cast<std::uint64_t>(k));
        std::vector<int32_t> h = hist;
        std::vector<int32_t> mid;
        for (int i = 0; i < M; ++i) {
            if (cancelled()) { res.cancelled = true; return res; }
            const int32_t tok = S.sample(S.row(0), h, A + i, /*forbid_end=*/true, u);
            mid.push_back(tok - CODEC_OFFSET);
            h.push_back(tok);
            if (cb.on_semantic) cb.on_semantic(tok - CODEC_OFFSET);
            S.step(tok);
            progress();
        }
        // B's opening, teacher-forced: the current row predicts B[0], each
        // fed B[j] the next.
        double nll = 0;
        if (scored > 0) {
            if (cancelled()) { res.cancelled = true; return res; }
            nll += nll_of(S.row(0), S.n, after_abs[0] - lo);
            progress();
            S.feed(after_abs.data(), scored - 1, [&](int first, int R) {
                for (int r = 0; r < R; ++r) {
                    nll += nll_of(S.row(r), S.n, after_abs[first + r + 1] - lo);
                    progress();
                }
            });
            nll /= scored;
        }
        res.scores.push_back(static_cast<float>(nll));
        res.middles.push_back(mid);
        if (k == 0 || nll < best) {
            best = nll;
            res.chosen = k;
        }
        held = k;
    }
    res.semantic = sp.before;
    res.semantic.insert(res.semantic.end(), res.middles[res.chosen].begin(), res.middles[res.chosen].end());
    res.semantic.insert(res.semantic.end(), sp.after.begin(), sp.after.end());
    if (!sp.continue_after) return res;

    // C: the session at A + M* + B, then sampled on until the song ends.
    for (int32_t c : res.middles[res.chosen]) hist.push_back(c + CODEC_OFFSET);
    hist.insert(hist.end(), after_abs.begin(), after_abs.end());
    if (held == res.chosen) {
        // The session holds A + M* + B[0, scored - 1): feed the rest of B.
        const int from = std::max(0, scored - 1);
        if (Bn > from) S.feed(after_abs.data() + from, Bn - from, [](int, int) {});
    } else {
        for (int b = 0; b < S.B; ++b) m.ar_truncate(*S.sess, b, base_len[b]);
        S.logits = base_logits;
        S.logit_rows = 1;
        std::vector<int32_t> rest;
        for (int32_t c : res.middles[res.chosen]) rest.push_back(c + CODEC_OFFSET);
        rest.insert(rest.end(), after_abs.begin(), after_abs.end());
        if (!rest.empty()) S.feed(rest.data(), static_cast<int>(rest.size()), [](int, int) {});
    }
    const UniformFn u = make_uniform(p.seed ^ 0x9E3779B97F4A7C15ull);
    bool eos = false;
    float* r = S.row(S.logit_rows - 1);
    for (int pos = static_cast<int>(hist.size()); pos < S.s.max_tokens; ++pos) {
        if (cancelled()) { res.cancelled = true; return res; }
        if (longest + pos + 1 >= CONTEXT) break;
        const int32_t tok = S.sample(r, hist, pos, /*forbid_end=*/false, u);
        progress();
        if (tok == S.end) {
            eos = true;
            break;
        }
        hist.push_back(tok);
        res.semantic.push_back(tok - CODEC_OFFSET);
        if (cb.on_semantic) cb.on_semantic(tok - CODEC_OFFSET);
        S.step(tok);
        r = S.row(0);
    }
    res.truncated = !eos;
    return res;
}

std::vector<float> Yue2::inpaint(const Yue2Params& p, const Yue2Plan& plan, const std::vector<int32_t>& semantic,
                                 std::vector<float> latents, int a, int b, const Yue2Callbacks& cb,
                                 const CancelCheck& cancel) const {
    if (!loaded()) fail("not loaded");
    const Impl& I = *impl_;
    check_plan(I, p, plan);
    check_codes(semantic, "inpaint");
    const int T = static_cast<int>(semantic.size()), D = kLatentDim;
    if (latents.size() != static_cast<std::size_t>(T) * D) fail("inpaint: latents must hold semantic.size() * 64 floats");
    if (a < 0 || b > T || a >= b) fail("inpaint: span [a, b) must lie inside the song and be non-empty");
    const Model& m = I.model;
    brotensor::DeviceScope scope(m.device());
    const auto noise = song_noise(p, T);
    const StageHooks nh = make_hooks(m, p, Yue2Stage::Synthesize, cb);
    const StageHooks ah = make_hooks(m, p, Yue2Stage::Semantic, Yue2Callbacks{});
    // A window is 1000 frames where the song allows: kept frames fill what the
    // span does not (a short window solves loud: a 10 s span at norm 8.37 in
    // 500 frames, 8.09 in 1000, the whole render 8.01; 1500 bought little
    // more), 25 of lookahead past a tile that is not the last (solved, then
    // re-solved by the next).
    constexpr int kWindow = 1000, kLook = 25, kTileMax = kWindow - kLook;
    const int N = b - a;
    const int tiles = (N + kTileMax - 1) / kTileMax;
    const int tile = (N + tiles - 1) / tiles;
    const int total = tiles * p.ode_steps;
    int done = 0;
    for (int t0 = a; t0 < b;) {
        const int t1 = std::min(b, t0 + tile);
        const bool last = t1 == b;
        const int spare = std::max(0, kWindow - (t1 - t0) - (last ? 0 : kLook));
        const int rc = last ? std::min(T - b, std::max(kLook, spare / 2)) : 0;
        const int lc = std::min(t0, std::max(kLook, spare - rc));
        const int lo = t0 - lc, hi = last ? b + rc : std::min(T, t1 + kLook);
        std::vector<int32_t> ar(plan.prefix);
        for (int f = lo; f < hi; ++f) ar.push_back(semantic[f] + CODEC_OFFSET);
        ar.push_back(MUSIC_END);
        Model::NarOptions o;
        o.ar_hooks = ah.get();
        o.hooks = nh.get();
        o.frame0 = lo;
        o.capture_evals = p.capture.nar_evals;
        std::vector<float> out;
        {
            auto chunk = m.nar_chunk(ar, hi - lo, o);
            out = m.nar_solve(*chunk, noise.data() + static_cast<std::size_t>(lo) * D, p.ode_steps, cancel,
                              [&](int) {
                                  ++done;
                                  if (cb.on_progress) cb.on_progress(Yue2Stage::Synthesize, done, total);
                              },
                              lc > 0 ? latents.data() + static_cast<std::size_t>(lo) * D : nullptr, lc,
                              rc > 0 ? latents.data() + static_cast<std::size_t>(b) * D : nullptr, rc);
        }
        if (out.empty()) return {};   // cancelled
        std::copy(out.begin() + static_cast<std::size_t>(t0 - lo) * D, out.begin() + static_cast<std::size_t>(t1 - lo) * D,
                  latents.begin() + static_cast<std::size_t>(t0) * D);
        t0 = t1;
    }
    return latents;
}

}  // namespace brosoundml
