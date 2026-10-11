// Yue2 stage machinery: hooks, the AR sampling loop, NAR chunks — see
// yue2_internal.h.

#include "yue2_internal.h"

#include <algorithm>
#include <stdexcept>

namespace brosoundml::yue2 {

namespace {

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("brosoundml: Yue2: " + m); }

}  // namespace

UniformFn stage_uniform(const Yue2Params& p, const std::shared_ptr<std::size_t>& cursor) {
    if (p.test_uniforms.empty()) return make_uniform(p.seed);
    auto vals = std::make_shared<std::vector<double>>(p.test_uniforms);
    return [vals, cursor]() {
        if (*cursor >= vals->size()) fail("test_uniforms exhausted");
        return (*vals)[(*cursor)++];
    };
}

StageHooks make_hooks(const Model& m, const Yue2Params& p, Yue2Stage stage, const Yue2Callbacks& cb) {
    StageHooks sh;
    const int L = m.config().layers, H = m.config().hidden;
    const bool nar = stage == Yue2Stage::Synthesize;
    sh.hooks.steer.resize(L);
    sh.hooks.capture.assign(L, 0);
    for (const Yue2Steer& s : p.steer) {
        if (s.layer < 0 || s.layer >= L) fail("steer layer out of range");
        if (s.nar != nar) continue;
        if (!nar && ((stage == Yue2Stage::Plan && !s.plan) || (stage == Yue2Stage::Semantic && !s.semantic)))
            continue;
        if (s.per_position && s.pos0 < 0) fail("steer pos0 must be non-negative");
        sh.hooks.steer[s.layer].push_back(m.make_steer(s.delta, s.scale, s.pos0, s.per_position));
        sh.active = true;
    }
    const Yue2Capture& c = p.capture;
    const bool want = cb.on_hidden && !c.layers.empty() &&
                      (nar ? !c.nar_evals.empty()
                           : (stage == Yue2Stage::Plan ? c.plan : c.semantic));
    if (want) {
        for (int l : c.layers) {
            if (l < 0 || l >= L) fail("capture layer out of range");
            sh.hooks.capture[l] = 1;
        }
        auto fn = cb.on_hidden;
        sh.hooks.on_capture = [fn, stage, H](int layer, int pos0, int rows, int eval, const float* data) {
            Yue2Hidden h;
            h.stage = stage;
            h.layer = layer;
            h.pos0 = pos0;
            h.rows = rows;
            h.hidden = H;
            h.nar_eval = eval;
            h.data = data;
            fn(h);
        };
        sh.active = true;
    }
    return sh;
}

ArRun run_ar(const Model& m, Phase ph, const std::vector<int32_t>& prefix,
             const std::vector<int32_t>* negative, float cfg, bool legacy_off, const Yue2Sampling& s,
             const UniformFn& uniform, const LayerHooks* hooks, const CancelCheck& cancel,
             const TokenFn& on_token, const std::function<void(int, int)>& on_progress,
             std::unique_ptr<ArSession>* keep, int draft) {
    validate(s, ph == Phase::Abc ? "abc" : "semantic");
    if (static_cast<int>(prefix.size()) + s.max_tokens > CONTEXT)
        fail("prefix + requested generation budget exceeds 24576; no implicit truncation");
    const bool use_cfg = cfg != 1.0f;
    if (draft < 0 || draft > 7) fail("draft tokens must be 0..7");
    if (draft > 0 && !use_cfg)
        return run_ar_spec(m, ph, prefix, legacy_off, s, uniform, hooks, cancel, on_token, on_progress, keep, draft);
    if (use_cfg && !negative) fail("CFG requires a negative prefix");
    if (use_cfg && static_cast<int>(negative->size()) + s.max_tokens > CONTEXT)
        fail("negative prefix + generation budget exceeds context");
    ArRun r;
    const auto t0 = Clock::now();
    if (cancel && cancel()) {
        r.cancelled = true;
        return r;
    }
    const auto [lo, hi] = logit_range(ph);
    const int n = hi - lo;
    const int cap = static_cast<int>(std::max(prefix.size(), use_cfg ? negative->size() : 0)) + s.max_tokens;
    auto sess = m.ar_session(use_cfg ? 2 : 1, std::min(cap, CONTEXT), hooks);
    m.ar_prefill(*sess, 0, prefix, lo, hi);
    if (use_cfg) m.ar_prefill(*sess, 1, *negative, lo, hi);
    const int32_t end = end_token(ph);
    std::vector<float> logits, combined(use_cfg ? n : 0);
    bool eos = false;
    for (int step = 0; step < s.max_tokens; ++step) {
        if (cancel && cancel()) {
            r.cancelled = true;
            break;
        }
        m.ar_logits(*sess, logits);
        const float* row = logits.data();
        if (use_cfg) {
            cfg_combine_bf16(logits.data(), logits.data() + n, cfg, combined.data(), n);
            row = combined.data();
        }
        const auto scores = distribution(row, n, ph, s, r.tokens, step, legacy_off);
        const int32_t tok = draw(scores, lo, s.temperature, uniform);
        if (step == 0) r.ttft = seconds_since(t0);
        ++r.count;
        if (on_progress) on_progress(step + 1, s.max_tokens);
        if (tok == end) {
            eos = true;
            break;
        }
        r.tokens.push_back(tok);
        if (on_token) on_token(tok, *sess);
        if (step + 1 < s.max_tokens) {
            m.ar_step(*sess, tok);
            ++r.forwards;
        }
    }
    r.truncated = !eos && !r.cancelled;
    r.seconds = seconds_since(t0);
    if (keep) *keep = std::move(sess);
    return r;
}

std::vector<float> song_noise(const Yue2Params& p, int frames) {
    const std::size_t n = static_cast<std::size_t>(frames) * Yue2::kLatentDim;
    if (!p.test_noise.empty()) {
        if (p.test_noise.size() < n) fail("test_noise is shorter than the song");
        return std::vector<float>(p.test_noise.begin(), p.test_noise.begin() + n);
    }
    return torch_cpu_randn(p.seed, n);
}

std::vector<float> synth_chunk(const Model& m, const Yue2Params& p, const std::vector<int32_t>& prefix,
                               const std::vector<int32_t>& codec, const std::vector<float>& noise,
                               const ChunkSpec& spec, const StageHooks& nar_hooks, const StageHooks& ar_hooks,
                               const CancelCheck& cancel, const std::function<void(int)>& on_step) {
    std::vector<int32_t> ar(prefix);
    for (int f = spec.a; f < spec.b; ++f) ar.push_back(codec[f] + CODEC_OFFSET);
    ar.push_back(MUSIC_END);
    Model::NarOptions opt;
    opt.reuse = spec.reuse;
    opt.reuse_rows = spec.reuse ? spec.reuse_rows : 0;
    opt.ar_hooks = ar_hooks.get();
    opt.hooks = nar_hooks.get();
    opt.frame0 = spec.a;
    opt.capture_evals = p.capture.nar_evals;
    auto chunk = m.nar_chunk(ar, spec.b - spec.a, opt);
    return m.nar_solve(*chunk, noise.data() + static_cast<std::size_t>(spec.a) * Yue2::kLatentDim, p.ode_steps,
                       cancel, on_step);
}

std::string utf8_replace(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    const auto* b = reinterpret_cast<const unsigned char*>(s.data());
    const std::size_t n = s.size();
    static const char kRep[] = "\xEF\xBF\xBD";
    std::size_t i = 0;
    while (i < n) {
        const unsigned char c = b[i];
        int len = 0;
        unsigned char lo = 0x80, hi = 0xBF;
        if (c < 0x80) len = 1;
        else if (c >= 0xC2 && c <= 0xDF) len = 2;
        else if (c >= 0xE0 && c <= 0xEF) {
            len = 3;
            if (c == 0xE0) lo = 0xA0;
            if (c == 0xED) hi = 0x9F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            len = 4;
            if (c == 0xF0) lo = 0x90;
            if (c == 0xF4) hi = 0x8F;
        }
        if (len == 0) {
            out += kRep;
            ++i;
            continue;
        }
        std::size_t j = 1;
        for (; j < static_cast<std::size_t>(len) && i + j < n; ++j) {
            const unsigned char d = b[i + j];
            const unsigned char l = j == 1 ? lo : 0x80, h = j == 1 ? hi : 0xBF;
            if (d < l || d > h) break;
        }
        if (j == static_cast<std::size_t>(len)) {
            out.append(s, i, len);
            i += len;
        } else {
            out += kRep;   // the maximal invalid prefix becomes one U+FFFD
            i += j;
        }
    }
    return out;
}

}  // namespace brosoundml::yue2
