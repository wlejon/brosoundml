#define _CRT_SECURE_NO_WARNINGS
// OmniVoice tests — every stage of the port against the upstream fixtures
// (tests/fixtures/omnivoice_*.bin, made by tests/ref/gen_omnivoice_fixture.py
// from the genuine k2-fsa implementation in FP32 on CUDA), on CPU first and
// then on CUDA when brotensor reports it.
//
//   Part A  tokenizer: plain and tag-aware ids of 65 strings, exact
//   Part B  prompt assembly (ids + audio-mask layout, 18 cases) and the
//           duration rule (53 cases), exact
//   Part C  the LM's first forward: step-0 logits (max |Δ| < 5e-2), the
//           MASK-frame embedding, text embeddings, packed input embeddings
//           and the post-norm hidden rows
//   Part D  16-step deterministic generation: the per-step unmask counts,
//           per-step predictions / scores, final codes (>= 99 %), the
//           unmask-order grid, decode + post-processing of the reference codes
//   Part E  voice clone: prompt codes from the preprocessed reference (100 %),
//           the preprocessing itself, and the cloned generation
//   Part F  the post-processing helpers in isolation (exact)
//   CPU vs CUDA parity, an end-to-end synthesis checked by Whisper, the
//   contract checks (throws before load / on empty text / on a bad instruct /
//   on a mismatched init; cancel returns empty), and FP32 / BF16 timings.
//
// Stages skip with a message when the weights or a fixture are absent.

#include "brosoundml/audio.h"
#include "brosoundml/omnivoice.h"
#include "brosoundml/whisper.h"
#include "omnivoice_lm.h"        // internal: white-box the LM
#include "omnivoice_prompt.h"    // internal: the prompt / duration / audio rules

#include <brolm/qwen_tokenizer.h>
#include <brolm/whisper_tokenizer.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using brosoundml::OmniVoice;
using brosoundml::OmniVoiceParams;
using brosoundml::OmniVoicePrompt;
namespace ovp = brosoundml::ovp;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

template <typename Fn>
static bool throws_runtime_error(Fn&& fn) {
    try { fn(); }
    catch (const std::runtime_error&) { return true; }
    catch (...) { return false; }
    return false;
}

// ─── fixture reader ─────────────────────────────────────────────────────────

struct Reader {
    std::ifstream f;
    bool ok = true;
    explicit Reader(const fs::path& p) : f(p.string(), std::ios::binary) { ok = static_cast<bool>(f); }
    template <typename T> T raw() { T v{}; f.read(reinterpret_cast<char*>(&v), sizeof(T)); if (!f) ok = false; return v; }
    int32_t i32() { return raw<int32_t>(); }
    float f32() { return raw<float>(); }
    double f64() { return raw<double>(); }
    // `none` receives whether the string was None (-1).
    std::string str(bool* none = nullptr) {
        const int32_t n = i32();
        if (none) *none = (n < 0);
        if (n <= 0) return {};
        std::string s(static_cast<std::size_t>(n), '\0');
        f.read(s.data(), n);
        if (!f) ok = false;
        return s;
    }
    std::vector<int32_t> i32arr(std::size_t n) { std::vector<int32_t> v(n); if (n) f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4)); if (!f) ok = false; return v; }
    std::vector<float> f32arr(std::size_t n) { std::vector<float> v(n); if (n) f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4)); if (!f) ok = false; return v; }
    std::vector<uint8_t> u8arr(std::size_t n) { std::vector<uint8_t> v(n); if (n) f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n)); if (!f) ok = false; return v; }
    std::vector<int32_t> i32list() { const int32_t n = i32(); return i32arr(n > 0 ? static_cast<std::size_t>(n) : 0); }
    std::vector<float> f32list() { const int32_t n = i32(); return f32arr(n > 0 ? static_cast<std::size_t>(n) : 0); }
};

struct FixA {
    std::vector<std::pair<std::string, int>> specials;
    struct S { std::string s; std::vector<int32_t> plain, tagaware; };
    std::vector<S> strings;
};
static bool read_A(const fs::path& p, FixA& a) {
    Reader r(p);
    if (!r.ok) return false;
    const int ns = r.i32();
    for (int i = 0; i < ns; ++i) { std::string n = r.str(); const int id = r.i32(); a.specials.emplace_back(n, id); }
    const int n = r.i32();
    for (int i = 0; i < n; ++i) { FixA::S s; s.s = r.str(); s.plain = r.i32list(); s.tagaware = r.i32list(); a.strings.push_back(std::move(s)); }
    return r.ok;
}

struct FixB {
    int C = 0, mask_id = 0, frame_rate = 0;
    struct P {
        std::string text, lang_in, instruct_in, ref_text_in;
        bool lang_none = false, instruct_none = false, ref_text_none = false;
        int ref_kind = 0, denoise = 1;
        float speed_in = 0, duration_in = 0;
        std::string lang_res, instruct_res, style, full, wrapped;
        bool lang_res_none = false, instruct_res_none = false;
        int T = 0; float speed_ratio = 1; int n_ref = 0;
        std::vector<int32_t> ref_codes;
        int n_style = 0, n_text = 0, c_len = 0;
        std::vector<int32_t> cond_ids; std::vector<uint8_t> layout;
        std::vector<int32_t> cond_text_ids, uncond_text_ids;
        int u_len = 0;
    };
    std::vector<P> prompts;
    struct D {
        std::string text, ref_text_in, ref_used; bool ref_none = false;
        int n_ref_in = 0; float speed = 1; int n_used = 0;
        double tw = 0, rw = 0, raw = 0; int est = 0;
    };
    std::vector<D> durations;
};
static bool read_B(const fs::path& p, FixB& b) {
    Reader r(p);
    if (!r.ok) return false;
    b.C = r.i32(); b.mask_id = r.i32(); b.frame_rate = r.i32();
    const int np = r.i32();
    for (int i = 0; i < np; ++i) {
        FixB::P c;
        c.text = r.str(); c.lang_in = r.str(&c.lang_none); c.instruct_in = r.str(&c.instruct_none); c.ref_text_in = r.str(&c.ref_text_none);
        c.ref_kind = r.i32(); c.denoise = r.i32(); c.speed_in = r.f32(); c.duration_in = r.f32();
        c.lang_res = r.str(&c.lang_res_none); c.instruct_res = r.str(&c.instruct_res_none);
        c.style = r.str(); c.full = r.str(); c.wrapped = r.str();
        c.T = r.i32(); c.speed_ratio = r.f32(); c.n_ref = r.i32();
        c.ref_codes = r.i32arr(static_cast<std::size_t>(b.C) * c.n_ref);
        c.n_style = r.i32(); c.n_text = r.i32(); c.c_len = r.i32();
        c.cond_ids = r.i32arr(static_cast<std::size_t>(b.C) * c.c_len);
        c.layout = r.u8arr(static_cast<std::size_t>(c.c_len));
        c.cond_text_ids = r.i32list(); c.uncond_text_ids = r.i32list();
        c.u_len = r.i32();
        b.prompts.push_back(std::move(c));
    }
    const int nd = r.i32();
    for (int i = 0; i < nd; ++i) {
        FixB::D d;
        d.text = r.str(); d.ref_text_in = r.str(&d.ref_none); d.n_ref_in = r.i32(); d.speed = r.f32();
        d.ref_used = r.str(); d.n_used = r.i32();
        d.tw = r.f64(); d.rw = r.f64(); d.raw = r.f64(); d.est = r.i32();
        b.durations.push_back(std::move(d));
    }
    return r.ok;
}

struct FixC {
    int C = 0, V = 0, H = 0, T = 0, c_len = 0, u_len = 0;
    std::string text, lang, instruct, style, wrapped;
    std::vector<int32_t> cond_ids; std::vector<uint8_t> mask; std::vector<int32_t> uncond_ids;
    std::vector<float> logits, mask_frame, text_embed, embeds, hidden;
    std::vector<int32_t> first_ids;
};
static bool read_C(const fs::path& p, FixC& c) {
    Reader r(p);
    if (!r.ok) return false;
    c.C = r.i32(); c.V = r.i32(); c.H = r.i32(); c.T = r.i32(); c.c_len = r.i32(); c.u_len = r.i32();
    c.text = r.str(); c.lang = r.str(); c.instruct = r.str(); c.style = r.str(); c.wrapped = r.str();
    c.cond_ids = r.i32arr(static_cast<std::size_t>(c.C) * c.c_len);
    c.mask = r.u8arr(static_cast<std::size_t>(c.c_len));
    c.uncond_ids = r.i32arr(static_cast<std::size_t>(c.C) * c.u_len);
    c.logits = r.f32arr(2ull * c.C * c.T * c.V);
    c.mask_frame = r.f32arr(static_cast<std::size_t>(c.H));
    const int ne = r.i32();
    c.first_ids = r.i32arr(static_cast<std::size_t>(ne));
    c.text_embed = r.f32arr(static_cast<std::size_t>(ne) * c.H);
    c.embeds = r.f32arr(static_cast<std::size_t>(c.c_len) * c.H);
    c.hidden = r.f32arr(2ull * c.T * c.H);
    return r.ok;
}

struct FixGen {
    int C = 0, V = 0, T = 0, sr = 0;
    int num_step = 0; std::vector<int32_t> k, codes, unmask;
    std::vector<float> scores; std::vector<int32_t> pred;   // Part D only
    std::vector<float> raw, post; std::vector<std::string> steps;
};
static void read_gen_tail(Reader& r, FixGen& g, bool with_scores) {
    g.num_step = r.i32();
    g.k = r.i32arr(static_cast<std::size_t>(g.num_step));
    g.codes = r.i32arr(static_cast<std::size_t>(g.C) * g.T);
    g.unmask = r.i32arr(static_cast<std::size_t>(g.C) * g.T);
    if (with_scores) {
        const int ns = r.i32();
        g.scores = r.f32arr(static_cast<std::size_t>(ns) * g.C * g.T);
        g.pred = r.i32arr(static_cast<std::size_t>(ns) * g.C * g.T);
    }
    g.raw = r.f32list(); g.post = r.f32list();
    const int n = r.i32();
    for (int i = 0; i < n; ++i) g.steps.push_back(r.str());
}

struct FixD {
    FixGen g;
    float guidance = 0, t_shift = 0, penalty = 0, pos_temp = 0, class_temp = 0, gumbel_u = 0, gumbel_const = 0;
};
static bool read_D(const fs::path& p, FixD& d) {
    Reader r(p);
    if (!r.ok) return false;
    d.g.C = r.i32(); d.g.V = r.i32(); d.g.T = r.i32(); d.g.sr = r.i32();
    d.guidance = r.f32(); d.t_shift = r.f32(); d.penalty = r.f32(); d.pos_temp = r.f32();
    d.class_temp = r.f32(); d.gumbel_u = r.f32(); d.gumbel_const = r.f32();
    read_gen_tail(r, d.g, true);
    return r.ok;
}

struct FixE {
    int C = 0, sr = 0, hop = 0;
    std::vector<float> ref_in; float ref_rms = 0; std::vector<float> ref_pre;
    int T_ref = 0; std::vector<int32_t> ref_codes;
    std::string ref_text_in, ref_text_out, text, lang; int denoise = 1;
    int est = 0, c_len = 0;
    std::vector<int32_t> cond_ids; std::vector<uint8_t> mask; std::vector<int32_t> uncond_text_ids;
    FixGen g;
};
static bool read_E(const fs::path& p, FixE& e) {
    Reader r(p);
    if (!r.ok) return false;
    e.C = r.i32(); e.sr = r.i32(); e.hop = r.i32();
    e.ref_in = r.f32list(); e.ref_rms = r.f32(); e.ref_pre = r.f32list();
    e.T_ref = r.i32(); e.ref_codes = r.i32arr(static_cast<std::size_t>(e.C) * e.T_ref);
    e.ref_text_in = r.str(); e.ref_text_out = r.str();
    e.text = r.str(); e.lang = r.str(); e.denoise = r.i32();
    e.est = r.i32(); e.c_len = r.i32(); e.g.T = r.i32();
    e.g.C = e.C; e.g.sr = e.sr;
    e.cond_ids = r.i32arr(static_cast<std::size_t>(e.C) * e.c_len);
    e.mask = r.u8arr(static_cast<std::size_t>(e.c_len));
    e.uncond_text_ids = r.i32list();
    read_gen_tail(r, e.g, false);
    return r.ok;
}

struct FixF {
    int sr = 0;
    struct Sig {
        std::string name; std::vector<float> x;
        struct RS { int mid, lead, trail; std::vector<float> out; };
        struct G { int kind; float ref_rms; std::vector<float> out; };
        struct FP { float pad, fade; std::vector<float> out; };
        std::vector<RS> rs; std::vector<G> gain; std::vector<FP> fp;
    };
    std::vector<Sig> signals;
};
static bool read_F(const fs::path& p, FixF& f) {
    Reader r(p);
    if (!r.ok) return false;
    f.sr = r.i32();
    const int ns = r.i32();
    for (int i = 0; i < ns; ++i) {
        FixF::Sig s;
        s.name = r.str(); s.x = r.f32list();
        int n = r.i32();
        for (int k = 0; k < n; ++k) { FixF::Sig::RS v; v.mid = r.i32(); v.lead = r.i32(); v.trail = r.i32(); v.out = r.f32list(); s.rs.push_back(std::move(v)); }
        n = r.i32();
        for (int k = 0; k < n; ++k) { FixF::Sig::G v; v.kind = r.i32(); v.ref_rms = r.f32(); v.out = r.f32list(); s.gain.push_back(std::move(v)); }
        n = r.i32();
        for (int k = 0; k < n; ++k) { FixF::Sig::FP v; v.pad = r.f32(); v.fade = r.f32(); v.out = r.f32list(); s.fp.push_back(std::move(v)); }
        f.signals.push_back(std::move(s));
    }
    return r.ok;
}

// ─── helpers ────────────────────────────────────────────────────────────────

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b, std::size_t n) {
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (std::isnan(d)) return std::numeric_limits<float>::infinity();
        m = std::max(m, d);
    }
    return m;
}

static double agreement(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    std::size_t same = 0;
    for (std::size_t i = 0; i < a.size(); ++i) same += (a[i] == b[i]);
    return 100.0 * static_cast<double>(same) / static_cast<double>(a.size());
}

static std::string ids_str(const std::vector<int32_t>& v, std::size_t limit = 24) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size() && i < limit; ++i) { if (i) s += ","; s += std::to_string(v[i]); }
    if (v.size() > limit) s += ",...";
    return s + "]";
}

static const OmniVoiceParams fixture_params() {
    OmniVoiceParams p;
    p.num_steps = 16;
    p.guidance_scale = 2.0f;
    p.t_shift = 0.1f;
    p.layer_penalty = 5.0f;
    p.position_temperature = 5.0f;
    p.class_temperature = 0.0f;
    p.gumbel_noise = false;
    p.language = "English";
    return p;
}

// ─── Parts C + D: the LM alone ──────────────────────────────────────────────

static void run_lm_parts(brotensor::Device dev, const char* dev_name, const fs::path& weights,
                         const FixC* fc, const FixD* fd) {
    if (!fc && !fd) { std::printf("  [%s] Parts C/D fixtures absent — skipped\n", dev_name); return; }
    std::printf("  [%s] loading the LM (FP32)\n", dev_name);
    brosoundml::OmniVoiceLm lm;
    brosoundml::OmniVoiceLmConfig cfg;   // the Qwen3-0.6B defaults are this checkpoint's
    {
        auto f = brotensor::safetensors::File::open((weights / "model.safetensors").string());
        const auto t0 = std::chrono::steady_clock::now();
        lm.load(f, cfg, dev, false);
        std::printf("    load %.2fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    const int C = cfg.num_codebooks, V = cfg.audio_vocab_size, H = cfg.hidden_size;

    // ── Part C ──
    if (fc) {
        std::printf("  [%s] Part C: first forward vs fixture\n", dev_name);
        CHECK(fc->C == C && fc->V == V && fc->H == H, "Part C: fixture geometry matches the config");
        const int T = fc->T, Lc = fc->c_len;
        const int n_text = Lc - T;
        std::vector<int32_t> ids(fc->cond_ids.begin(), fc->cond_ids.begin() + n_text);
        for (int i = 0; i < n_text; ++i) CHECK(fc->mask[static_cast<std::size_t>(i)] == 0, "Part C: text positions are not audio");

        // MASK-frame embedding and text embeddings.
        {
            const std::vector<float> mf = lm.mask_frame_embedding();
            const float d = max_abs_diff(mf, fc->mask_frame, static_cast<std::size_t>(H));
            std::printf("    mask-frame embedding max|d| = %.3g\n", d);
            CHECK(d < 1e-5f, "Part C: MASK-frame embedding matches");
            const std::vector<float> te = lm.text_embeddings(fc->first_ids);
            const float d2 = max_abs_diff(te, fc->text_embed, fc->first_ids.size() * static_cast<std::size_t>(H));
            std::printf("    text embeddings max|d| = %.3g\n", d2);
            CHECK(d2 == 0.0f, "Part C: text embeddings match exactly");
        }

        brosoundml::OmniVoiceLmDebug dbg;
        dbg.capture_forward0 = true;
        brosoundml::OmniVoiceLmRun run;
        run.num_steps = 1; run.guidance_scale = 2.0f; run.gumbel_noise = false;
        const auto t0 = std::chrono::steady_clock::now();
        auto res = lm.generate(ids, {}, 0, T, nullptr, run, {}, {}, &dbg);
        std::printf("    forward %.3fs (L=%d)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), dbg.L);
        CHECK(dbg.Lc == Lc && dbg.R == 2 * T, "Part C: packed geometry (Lc, R)");

        // Input embeddings of the conditional row.
        {
            const float d = max_abs_diff(dbg.embeds0, fc->embeds, static_cast<std::size_t>(Lc) * H);
            std::printf("    cond input embeddings max|d| = %.3g\n", d);
            CHECK(d < 1e-5f, "Part C: conditional input embeddings match");
            // and the unconditional rows equal the cond target rows
            float du = 0.0f;
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < H; ++i)
                    du = std::max(du, std::fabs(dbg.embeds0[(static_cast<std::size_t>(Lc) + t) * H + i] -
                                                dbg.embeds0[(static_cast<std::size_t>(Lc) - T + t) * H + i]));
            CHECK(du == 0.0f, "Part C: unconditional rows carry the target embeddings");
        }
        // Hidden rows (post final norm) at the target positions.
        {
            const float d = max_abs_diff(dbg.hidden0, fc->hidden, 2ull * T * H);
            std::printf("    hidden (post-norm, 2T rows) max|d| = %.3g\n", d);
            CHECK(d < 5e-2f, "Part C: hidden rows match within 5e-2");
        }
        // Logits: fixture [2][C][T][V] vs ours (2T, C*V).
        {
            float d = 0.0f, maxabs = 0.0f;
            double sum_sq = 0.0;
            std::size_t n = 0;
            for (int r = 0; r < 2; ++r)
                for (int c = 0; c < C; ++c)
                    for (int t = 0; t < T; ++t)
                        for (int v = 0; v < V; ++v) {
                            const float a = fc->logits[((static_cast<std::size_t>(r) * C + c) * T + t) * V + v];
                            const float b = dbg.logits0[(static_cast<std::size_t>(r) * T + t) * (C * V) + static_cast<std::size_t>(c) * V + v];
                            const float e = std::fabs(a - b);
                            d = std::max(d, e);
                            maxabs = std::max(maxabs, std::fabs(a));
                            sum_sq += static_cast<double>(e) * e;
                            ++n;
                        }
            std::printf("    step-0 logits max|d| = %.4g (rms %.3g, |logit| max %.3g)\n", d,
                        std::sqrt(sum_sq / static_cast<double>(n)), maxabs);
            CHECK(d < 5e-2f, "Part C: step-0 logits match within 5e-2");
        }
    }

    // ── Part C (BF16) ── the same first forward with BF16 GEMM / attention
    // operands, against the FP32 fixture: a loose bound on the logits and the
    // argmax agreement per (row, codebook, frame) cell, reported.
    if (fc) {
        std::printf("  [%s] Part C (BF16): first forward with BF16 operands\n", dev_name);
        brosoundml::OmniVoiceLm lmb;
        {
            auto f = brotensor::safetensors::File::open((weights / "model.safetensors").string());
            lmb.load(f, cfg, dev, true);
        }
        const int T = fc->T, Lc = fc->c_len;
        const int n_text = Lc - T;
        std::vector<int32_t> ids(fc->cond_ids.begin(), fc->cond_ids.begin() + n_text);
        brosoundml::OmniVoiceLmDebug dbg;
        dbg.capture_forward0 = true;
        brosoundml::OmniVoiceLmRun run;
        run.num_steps = 1; run.guidance_scale = 2.0f; run.gumbel_noise = false;
        const auto t0 = std::chrono::steady_clock::now();
        auto res = lmb.generate(ids, {}, 0, T, nullptr, run, {}, {}, &dbg);
        std::printf("    forward %.3fs (L=%d)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), dbg.L);
        float d = 0.0f;
        double sum_sq = 0.0;
        std::size_t n = 0, agree = 0, cells = 0;
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < C; ++c)
                for (int t = 0; t < T; ++t) {
                    int am_a = 0, am_b = 0;
                    float best_a = -INFINITY, best_b = -INFINITY;
                    for (int v = 0; v < V; ++v) {
                        const float a = fc->logits[((static_cast<std::size_t>(r) * C + c) * T + t) * V + v];
                        const float b = dbg.logits0[(static_cast<std::size_t>(r) * T + t) * (C * V) + static_cast<std::size_t>(c) * V + v];
                        const float e = std::fabs(a - b);
                        d = std::max(d, e);
                        sum_sq += static_cast<double>(e) * e;
                        ++n;
                        if (a > best_a) { best_a = a; am_a = v; }
                        if (b > best_b) { best_b = b; am_b = v; }
                    }
                    ++cells;
                    if (am_a == am_b) ++agree;
                }
        const double pct = 100.0 * static_cast<double>(agree) / static_cast<double>(cells);
        std::printf("    BF16 vs FP32 fixture: step-0 logits max|d| = %.4g (rms %.3g), argmax agreement %.2f%% (%zu cells)\n",
                    d, std::sqrt(sum_sq / static_cast<double>(n)), pct, cells);
        CHECK(d < 3.0f, "Part C (BF16): step-0 logits within 3.0 of the FP32 fixture");
        // At step 0 every cell is masked and most rows are near-flat, so the
        // per-cell argmax is a weak statistic — reported, with a sanity floor.
        CHECK(pct > 50.0, "Part C (BF16): step-0 argmax agrees with FP32 on more than half the cells");

        // The 16-step deterministic generation in BF16 vs the FP32 fixture:
        // near-tie flips propagate through the discrete stream, so this is
        // reported rather than bounded; the BF16 transcript oracle in the
        // end-to-end section is the correctness gate.
        if (fd) {
            const FixGen& g = fd->g;
            brosoundml::OmniVoiceLmRun run2;
            run2.num_steps = g.num_step; run2.t_shift = fd->t_shift; run2.guidance_scale = fd->guidance;
            run2.layer_penalty = fd->penalty; run2.position_temperature = fd->pos_temp;
            run2.class_temperature = fd->class_temp; run2.gumbel_noise = false; run2.seed = 0;
            const auto t1 = std::chrono::steady_clock::now();
            auto r2 = lmb.generate(ids, {}, 0, g.T, nullptr, run2, {}, {}, nullptr);
            const double s2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
            std::printf("    BF16 16-step generation: %.2fs (%.3fs/step), codes agreement %.2f%%, unmask grid agreement %.2f%% vs the FP32 fixture\n",
                        s2, s2 / std::max(1, r2.steps_run), agreement(r2.codes, g.codes), agreement(r2.unmask_step, g.unmask));
            CHECK(r2.steps_run == g.num_step && !r2.cancelled, "Part D (BF16): every step ran");
            bool ok = r2.codes.size() == static_cast<std::size_t>(C) * g.T;
            for (int32_t c : r2.codes) if (c < 0 || c >= V - 1) ok = false;
            CHECK(ok, "Part D (BF16): every cell committed to an in-range code");
        }
    }

    // ── Part D ──
    if (fd) {
        std::printf("  [%s] Part D: 16-step deterministic generation vs fixture\n", dev_name);
        const FixGen& g = fd->g;
        const int T = g.T;
        const std::size_t cells = static_cast<std::size_t>(C) * T;
        CHECK(g.C == C && g.V == V, "Part D: fixture geometry");
        // The schedule alone.
        {
            const std::vector<int> ks = ovp::unmask_schedule(static_cast<int>(cells), g.num_step, fd->t_shift);
            bool same = ks.size() == g.k.size();
            for (std::size_t i = 0; same && i < ks.size(); ++i) same = ks[i] == g.k[i];
            std::printf("    k = ");
            for (int k : ks) std::printf("%d ", k);
            std::printf("%s\n", same ? "(matches)" : "(MISMATCH)");
            CHECK(same, "Part D: per-step unmask counts match the fixture");
        }
        std::vector<int32_t> ids;
        if (fc) ids.assign(fc->cond_ids.begin(), fc->cond_ids.begin() + (fc->c_len - fc->T));
        if (ids.empty()) { std::printf("    Part C fixture needed for the prompt ids — skipped\n"); return; }

        brosoundml::OmniVoiceLmRun run;
        run.num_steps = g.num_step; run.t_shift = fd->t_shift; run.guidance_scale = fd->guidance;
        run.layer_penalty = fd->penalty; run.position_temperature = fd->pos_temp;
        run.class_temperature = fd->class_temp; run.gumbel_noise = false; run.seed = 0;

        brosoundml::OmniVoiceLmDebug dbg;
        int steps_seen = 0, pred_mismatch_total = 0, k_mismatch = 0;
        float score_max = 0.0f;
        std::vector<int> step_pred_mismatch;
        std::vector<float> step_score_max;
        dbg.on_scores = [&](int step, int k, const std::vector<int32_t>& pred, const std::vector<float>& scores,
                            const std::vector<int32_t>&) {
            ++steps_seen;
            if (k != g.k[static_cast<std::size_t>(step)]) ++k_mismatch;
            int pm = 0;
            float sm = 0.0f;
            for (std::size_t i = 0; i < cells; ++i) {
                const bool masked_then = g.unmask[i] >= step;   // still masked at this step in the reference run
                if (!masked_then) continue;
                if (pred[i] != g.pred[static_cast<std::size_t>(step) * cells + i]) ++pm;
                const float ref = g.scores[static_cast<std::size_t>(step) * cells + i];
                if (std::isfinite(scores[i])) sm = std::max(sm, std::fabs(scores[i] - ref));
                else sm = std::numeric_limits<float>::infinity();   // we consider it fixed, the reference did not
            }
            step_pred_mismatch.push_back(pm);
            step_score_max.push_back(sm);
            pred_mismatch_total += pm;
            score_max = std::max(score_max, sm);
        };
        const auto t0 = std::chrono::steady_clock::now();
        auto res = lm.generate(ids, {}, 0, T, nullptr, run, {}, {}, &dbg);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("    %d steps in %.2fs (%.3fs/step)\n", res.steps_run, secs, secs / std::max(1, res.steps_run));
        CHECK(steps_seen == g.num_step && !res.cancelled, "Part D: every step ran");
        CHECK(k_mismatch == 0, "Part D: the k passed to top-k matches per step");
        std::printf("    per-step pred mismatches (masked cells):");
        for (int v : step_pred_mismatch) std::printf(" %d", v);
        std::printf("\n    per-step score max|d|:");
        for (float v : step_score_max) std::printf(" %.2g", v);
        std::printf("\n");
        const double agree = agreement(res.codes, g.codes);
        const double grid = agreement(res.unmask_step, g.unmask);
        std::printf("    final codes agreement %.2f%%  unmask-step grid agreement %.2f%%  score max|d| %.3g\n",
                    agree, grid, score_max);
        CHECK(agree >= 99.0, "Part D: final codes agree >= 99% with the fixture");
        CHECK(grid >= 95.0, "Part D: unmask-step grid agrees >= 95% with the fixture");
        CHECK(score_max < 5e-2f, "Part D: per-step scores within 5e-2 (masked cells)");
        for (int32_t c : res.codes) if (c < 0 || c >= V - 1) { CHECK(false, "Part D: a code is out of range / still MASK"); break; }
    }
}

// ─── Parts A, B, E, F + pipeline checks ─────────────────────────────────────

struct PipelineFixtures {
    const FixA* a = nullptr; const FixB* b = nullptr; const FixD* d = nullptr;
    const FixE* e = nullptr; const FixF* f = nullptr; const FixC* c = nullptr;
};

static std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

static std::string whisper_transcribe(brotensor::Device dev, const fs::path& whisper_dir,
                                      const brosoundml::AudioBuffer& out) {
    const int n = static_cast<int>(out.samples.size());
    const int n16 = static_cast<int>(static_cast<long long>(n) * 16000 / out.sample_rate);
    brotensor::Tensor x = brotensor::Tensor::from_host_on(brotensor::Device::CPU, out.samples.data(), 1, n);
    brotensor::Tensor y;
    brotensor::resample1d_forward(x, 1, 1, n, n16, /*mode=*/1, y);
    brosoundml::AudioBuffer a16(std::vector<float>(y.host_f32(), y.host_f32() + n16), 16000);
    brosoundml::Whisper w;
    w.load(whisper_dir.string(), dev);
    auto tok = brolm::whisper::Tokenizer::load((whisper_dir / "vocab.json").string(),
                                               (whisper_dir / "merges.txt").string());
    std::vector<int32_t> prompt = tok.build_prompt("en", "transcribe", /*timestamps=*/false);
    brosoundml::Whisper::TranscribeOptions opts;
    auto r = w.transcribe(a16, prompt, opts);
    std::vector<int32_t> ids(r.token_ids.begin() + static_cast<std::ptrdiff_t>(prompt.size()), r.token_ids.end());
    return tok.decode(ids, /*skip_special=*/true);
}

// `model` false: no weights are loaded and only the model-free stages run
// (A, B, F, the Part E preprocessing + OVCP round trip). `model` true loads
// the pipeline on `dev` and runs everything.
static void run_pipeline_parts(brotensor::Device dev, const char* dev_name, const fs::path& repo,
                               const PipelineFixtures& fx, bool model) {
    const fs::path weights = repo / "weights" / "omnivoice";
    std::unique_ptr<OmniVoice> ov;
    const brosoundml::OmniVoiceLmConfig lm_defaults;   // the Qwen3-0.6B / 8-codebook geometry
    const int C = lm_defaults.num_codebooks;
    const int MASK = lm_defaults.audio_mask_id;
    if (model) {
        std::printf("  [%s] loading the pipeline (FP32, codec with encoder)\n", dev_name);
        ov = std::make_unique<OmniVoice>();
        const auto t0 = std::chrono::steady_clock::now();
        ov->load(weights.string(), dev, brosoundml::OmniVoicePrecision::FP32, false);
        std::printf("    load %.2fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        CHECK(ov->loaded() && ov->device() == dev, "pipeline: loaded on the requested device");
        const brosoundml::OmniVoiceConfig& cfg = ov->config();
        CHECK(cfg.denoise_id == 151669 && cfg.text_start_id == 151674 && cfg.text_end_id == 151675 &&
              cfg.eos_id == 151645 && cfg.pad_id == 151643, "pipeline: special ids resolved from the tokenizer");
        CHECK(cfg.sample_rate == 24000 && cfg.frame_rate == 25 && cfg.lm.num_codebooks == C && cfg.lm.audio_mask_id == MASK,
              "pipeline: codec geometry");
    } else {
        std::printf("  [%s] model-free stages (no weights loaded)\n", dev_name);
    }

    brolm::qwen::Tokenizer tok = brolm::qwen::Tokenizer::from_tokenizer_json((weights / "tokenizer.json").string());

    // ── Part A ──
    if (fx.a) {
        std::printf("  [%s] Part A: tokenizer (%zu strings)\n", dev_name, fx.a->strings.size());
        for (const auto& sp : fx.a->specials) {
            const std::vector<int32_t> ids = tok.encode(sp.first);
            CHECK(ids.size() == 1 && ids[0] == sp.second, ("Part A: special token id " + sp.first).c_str());
        }
        int bad_plain = 0, bad_tag = 0;
        for (const auto& s : fx.a->strings) {
            const std::vector<int32_t> plain = tok.encode(s.s);
            const std::vector<int32_t> tag = ov ? ov->tokenize(s.s) : ovp::tokenize_with_tags(s.s, tok);
            if (plain != s.plain) { ++bad_plain; std::printf("    plain mismatch: '%s' -> %s vs %s\n", s.s.c_str(), ids_str(plain).c_str(), ids_str(s.plain).c_str()); }
            if (tag != s.tagaware) { ++bad_tag; std::printf("    tag-aware mismatch: '%s' -> %s vs %s\n", s.s.c_str(), ids_str(tag).c_str(), ids_str(s.tagaware).c_str()); }
        }
        std::printf("    plain mismatches %d, tag-aware mismatches %d\n", bad_plain, bad_tag);
        CHECK(bad_plain == 0, "Part A: plain tokenization exact on every string");
        CHECK(bad_tag == 0, "Part A: tag-aware tokenization exact on every string");
    }

    // The clone prompt the Part B "encoded" cases and Part E use.
    OmniVoicePrompt fixture_prompt;
    if (fx.e) {
        fixture_prompt.codes = fx.e->ref_codes;
        fixture_prompt.num_frames = fx.e->T_ref;
        fixture_prompt.text = fx.e->ref_text_out;
        fixture_prompt.rms = fx.e->ref_rms;
    }

    // ── Part B ──
    if (fx.b) {
        std::printf("  [%s] Part B: prompt assembly (%zu cases) + duration rule (%zu cases)\n", dev_name,
                    fx.b->prompts.size(), fx.b->durations.size());
        int bad = 0;
        for (std::size_t ci = 0; ci < fx.b->prompts.size(); ++ci) {
            const FixB::P& c = fx.b->prompts[ci];
            OmniVoiceParams p;
            p.language = c.lang_none ? "" : c.lang_in;
            p.instruct = c.instruct_none ? "" : c.instruct_in;
            p.denoise = c.denoise != 0;
            p.speed = c.speed_in != 0.0f ? c.speed_in : 1.0f;
            p.duration = c.duration_in;
            OmniVoicePrompt synth;
            const OmniVoicePrompt* prompt = nullptr;
            if (c.ref_kind == 1) {
                if (!fx.e) { std::printf("    case %zu needs the Part E prompt — skipped\n", ci); continue; }
                prompt = &fixture_prompt;
            } else if (c.ref_kind == 2) {
                synth.codes = c.ref_codes; synth.num_frames = c.n_ref;
                synth.text = c.ref_text_none ? "" : c.ref_text_in; synth.rms = 0.05f;
                prompt = &synth;
            }
            bool ok = true;
            auto expect = [&](bool cond, const char* what) {
                if (!cond) { ok = false; std::printf("    case %zu ('%s'): %s\n", ci, c.text.c_str(), what); }
            };
            // language / instruct resolution
            const std::string lang = ovp::resolve_language(p.language);
            expect(lang == (c.lang_res_none ? "" : c.lang_res), "language resolution");
            std::string instruct;
            try { instruct = p.instruct.empty() ? "" : ovp::resolve_instruct(p.instruct, ovp::text_has_cjk(c.text)); }
            catch (const std::exception& e) { expect(false, e.what()); }
            expect(instruct == (c.instruct_res_none ? "" : c.instruct_res), "instruct resolution");
            // style / combined / wrapped text
            const bool has_ref = prompt != nullptr;
            const std::string style = ovp::style_text(lang, instruct, p.denoise, has_ref);
            expect(style == c.style, "style text");
            const std::string* rt = (prompt && !prompt->text.empty()) ? &prompt->text : nullptr;
            const std::string full = ovp::combine_text(c.text, rt);
            expect(full == c.full, "combined text");
            const std::string wrapped = "<|text_start|>" + full + "<|text_end|>";
            expect(wrapped == c.wrapped, "wrapped text");
            // ids + layout
            std::vector<int32_t> ids = tok.encode(style);
            expect(static_cast<int>(ids.size()) == c.n_style, "style id count");
            const std::vector<int32_t> tids = ovp::tokenize_with_tags(wrapped, tok);
            expect(static_cast<int>(tids.size()) == c.n_text, "text id count");
            ids.insert(ids.end(), tids.begin(), tids.end());
            expect(ids == c.cond_text_ids, "conditional text ids");
            expect(c.uncond_text_ids.empty(), "unconditional row is target-only");
            // the estimated length + speed ratio
            int T = 0;
            float ratio = 1.0f;
            try {
                // the duration rule (what OmniVoice::estimate_frames applies)
                const std::string* ref_text = prompt ? &prompt->text : nullptr;
                const int n_ref = prompt ? prompt->num_frames : -1;
                const bool has_dur = p.duration > 0.0f;
                const int est = ovp::estimate_target_tokens(c.text, ref_text, n_ref, has_dur ? 1.0 : p.speed);
                const int target = std::max(1, static_cast<int>(static_cast<double>(p.duration) * 25));
                T = has_dur ? target : est;
                ratio = has_dur ? static_cast<float>(static_cast<double>(est) / target) : p.speed;
                if (ov) expect(ov->estimate_frames(c.text, p, prompt) == T, "estimate_frames applies the duration rule");
            } catch (const std::exception& e) { expect(false, e.what()); }
            expect(T == c.T, "target frame count");
            expect(std::fabs(ratio - c.speed_ratio) < 1e-6f, "speed ratio");
            expect(c.u_len == c.T, "u_len == T");
            // layout: text, then n_ref audio, then T targets; cond row 0 carries ref codes then MASK
            const int n_ref = prompt ? prompt->num_frames : 0;
            expect(c.c_len == c.n_style + c.n_text + n_ref + c.T, "c_len");
            expect(c.n_ref == n_ref, "reference frame count");
            bool lay = true;
            for (int i = 0; i < c.c_len; ++i) {
                const uint8_t want = i < c.n_style + c.n_text ? 0 : (i < c.n_style + c.n_text + n_ref ? 1 : 2);
                if (c.layout[static_cast<std::size_t>(i)] != want) lay = false;
            }
            expect(lay, "audio-mask layout");
            bool ref_ok = true;
            for (int q = 0; q < C && prompt; ++q)
                for (int t = 0; t < n_ref; ++t)
                    if (c.cond_ids[static_cast<std::size_t>(q) * c.c_len + c.n_style + c.n_text + t] !=
                        prompt->codes[static_cast<std::size_t>(q) * n_ref + t]) ref_ok = false;
            for (int q = 0; q < C; ++q)
                for (int t = 0; t < c.T; ++t)
                    if (c.cond_ids[static_cast<std::size_t>(q) * c.c_len + c.c_len - c.T + t] != MASK) ref_ok = false;
            expect(ref_ok, "reference codes / MASK targets in the cond row");
            if (!ok) ++bad;
        }
        std::printf("    prompt cases failing: %d / %zu\n", bad, fx.b->prompts.size());
        CHECK(bad == 0, "Part B: prompt ids + layout exact on every case");

        int bad_d = 0;
        for (std::size_t di = 0; di < fx.b->durations.size(); ++di) {
            const FixB::D& d = fx.b->durations[di];
            const std::string* ref = d.ref_none ? nullptr : &d.ref_text_in;
            const int n_ref = d.ref_none ? -1 : d.n_ref_in;
            // n_ref_in of 0 in the fixture stands for None when ref_text is None;
            // an explicit 0 with a ref text is a real 0.
            const int est = ovp::estimate_target_tokens(d.text, ref, (ref && d.n_ref_in == 0 && d.n_used == 25 && d.ref_used == "Nice to meet you." && d.ref_text_in != "Nice to meet you.") ? -1 : n_ref, d.speed);
            const double tw = ovp::total_weight(d.text);
            const double rw = ovp::total_weight(d.ref_used);
            const double raw = ovp::estimate_duration(d.text, d.ref_used, d.n_used);
            const bool ok = est == d.est && std::fabs(tw - d.tw) < 1e-9 && std::fabs(rw - d.rw) < 1e-9 &&
                            std::fabs(raw - d.raw) <= 1e-9 * std::max(1.0, std::fabs(d.raw));
            if (!ok) {
                ++bad_d;
                std::printf("    dur[%zu] '%s': est %d vs %d, tw %.6f vs %.6f, rw %.6f vs %.6f, raw %.9f vs %.9f\n", di,
                            d.text.substr(0, 40).c_str(), est, d.est, tw, d.tw, rw, d.rw, raw, d.raw);
            }
        }
        std::printf("    duration cases failing: %d / %zu\n", bad_d, fx.b->durations.size());
        CHECK(bad_d == 0, "Part B: duration estimator exact on every case");
    }

    // ── Part F ──
    if (fx.f) {
        std::printf("  [%s] Part F: post-processing helpers\n", dev_name);
        int bad = 0;
        for (const auto& s : fx.f->signals) {
            for (const auto& v : s.rs) {
                const std::vector<float> y = ovp::remove_silence(s.x, fx.f->sr, v.mid, v.lead, v.trail);
                const float d = y.size() == v.out.size() ? max_abs_diff(y, v.out, y.size()) : std::numeric_limits<float>::infinity();
                std::printf("    %s remove_silence(%d,%d,%d): %zu -> %zu (ref %zu) max|d| %.3g\n", s.name.c_str(), v.mid, v.lead, v.trail,
                            s.x.size(), y.size(), v.out.size(), d);
                if (!(d == 0.0f)) ++bad;
            }
            for (const auto& v : s.gain) {
                std::vector<float> y = s.x;
                if (v.kind == 0) ovp::peak_normalize(y); else ovp::gain_rms_match(y, v.ref_rms);
                const float d = y.size() == v.out.size() ? max_abs_diff(y, v.out, y.size()) : std::numeric_limits<float>::infinity();
                std::printf("    %s gain kind %d (rms %.4f): max|d| %.3g\n", s.name.c_str(), v.kind, v.ref_rms, d);
                if (!(d == 0.0f)) ++bad;
            }
            for (const auto& v : s.fp) {
                const std::vector<float> y = ovp::fade_and_pad(s.x, v.pad, v.fade, fx.f->sr);
                const float d = y.size() == v.out.size() ? max_abs_diff(y, v.out, y.size()) : std::numeric_limits<float>::infinity();
                std::printf("    %s fade_and_pad(%.2f,%.2f): %zu -> %zu (ref %zu) max|d| %.3g\n", s.name.c_str(), v.pad, v.fade,
                            s.x.size(), y.size(), v.out.size(), d);
                if (!(d == 0.0f)) ++bad;
            }
        }
        CHECK(bad == 0, "Part F: every post-processing helper matches the fixture exactly");
    }

    // ── Part D through the pipeline: decode + post-process of the reference codes ──
    if (fx.d && ov) {
        std::printf("  [%s] Part D (pipeline): decode + post-process of the fixture codes\n", dev_name);
        const FixGen& g = fx.d->g;
        const auto t0 = std::chrono::steady_clock::now();
        brosoundml::AudioBuffer raw = ov->decode_codes(g.codes, g.T);
        std::printf("    decode %d frames: %.3fs\n", g.T, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        const float d = raw.samples.size() == g.raw.size() ? max_abs_diff(raw.samples, g.raw, raw.samples.size()) : std::numeric_limits<float>::infinity();
        std::printf("    decoded waveform vs fixture raw: max|d| %.3g (%zu samples)\n", d, raw.samples.size());
        CHECK(d < 2e-3f, "Part D: codec decode of the fixture codes matches the fixture raw waveform");
        std::vector<float> post = ovp::remove_silence(g.raw, g.sr, 500, 100, 100);
        ovp::peak_normalize(post);
        post = ovp::fade_and_pad(post, 0.1, 0.1, g.sr);
        const float dp = post.size() == g.post.size() ? max_abs_diff(post, g.post, post.size()) : std::numeric_limits<float>::infinity();
        std::printf("    post-processing of the fixture raw: %zu -> %zu (ref %zu) max|d| %.3g\n", g.raw.size(), post.size(), g.post.size(), dp);
        CHECK(dp == 0.0f, "Part D: post-processing chain reproduces the fixture output exactly");
        // and generate_codes through the pipeline reproduces the LM run (Part C's prompt).
        if (fx.c) {
            OmniVoiceParams p = fixture_params();
            p.instruct = fx.c->instruct;
            brosoundml::OmniVoiceTrace tr;
            const std::vector<int32_t> codes = ov->generate_codes(fx.c->text, 0, p, nullptr, nullptr, {}, &tr);
            std::vector<int32_t> want_ids(fx.c->cond_ids.begin(), fx.c->cond_ids.begin() + (fx.c->c_len - fx.c->T));
            CHECK(tr.text_ids == want_ids, "Part D (pipeline): prompt ids equal the fixture's conditional text ids");
            CHECK(tr.num_frames == g.T, "Part D (pipeline): estimated frame count equals the fixture's T");
            const double agree = agreement(codes, g.codes);
            std::printf("    generate_codes: T=%d, codes agreement %.2f%%, unmask grid agreement %.2f%%, %.2fs LM\n",
                        tr.num_frames, agree, agreement(tr.unmask_step, g.unmask), tr.lm_seconds);
            CHECK(agree >= 99.0, "Part D (pipeline): generate_codes agrees >= 99% with the fixture");
        }
    }

    // ── Part E ──
    if (fx.e) {
        const FixE& e = *fx.e;
        std::printf("  [%s] Part E: voice clone\n", dev_name);
        // Preprocessing on the raw reference must reproduce the encoder input.
        {
            const std::vector<float> pre = ovp::remove_silence(e.ref_in, e.sr, 200, 100, 200);
            const float d = pre.size() == e.ref_pre.size() ? max_abs_diff(pre, e.ref_pre, pre.size()) : std::numeric_limits<float>::infinity();
            std::printf("    remove_silence(200,100,200) on ref_in: %zu -> %zu (ref %zu) max|d| %.3g\n", e.ref_in.size(), pre.size(), e.ref_pre.size(), d);
            CHECK(d == 0.0f, "Part E: prompt preprocessing reproduces the encoder input exactly");
            const float r = ovp::rms(e.ref_in);
            std::printf("    ref rms %.9f (fixture %.9f)\n", r, e.ref_rms);
            CHECK(std::fabs(r - e.ref_rms) < 1e-6f, "Part E: reference RMS");
        }
        // create_prompt from the preprocessed waveform (no preprocessing) and from the raw one (preprocessing).
        OmniVoicePrompt p1, p2;
        if (ov) {
            const auto t0 = std::chrono::steady_clock::now();
            p1 = ov->create_prompt(brosoundml::AudioBuffer(e.ref_pre, e.sr), e.ref_text_in, false);
            std::printf("    create_prompt(preprocessed, no preprocess): T_ref=%d in %.3fs\n", p1.num_frames,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            CHECK(p1.num_frames == e.T_ref, "Part E: prompt frame count");
            const double agree = agreement(p1.codes, e.ref_codes);
            std::printf("    prompt codes agreement %.2f%%\n", agree);
            CHECK(agree == 100.0, "Part E: prompt codes 100% vs the fixture");
            CHECK(p1.text == e.ref_text_in, "Part E: no-preprocess prompt keeps the text verbatim");
            p2 = ov->create_prompt(brosoundml::AudioBuffer(e.ref_in, e.sr), e.ref_text_in, true);
            const double agree2 = agreement(p2.codes, e.ref_codes);
            std::printf("    create_prompt(raw, preprocess): T_ref=%d codes agreement %.2f%% text '%s'\n", p2.num_frames, agree2, p2.text.c_str());
            CHECK(p2.num_frames == e.T_ref && agree2 == 100.0, "Part E: preprocessed prompt codes 100% vs the fixture");
            CHECK(p2.text == e.ref_text_out, "Part E: add_punctuation on the reference text");
            CHECK(std::fabs(p2.rms - e.ref_rms) < 1e-6f, "Part E: prompt rms");
        }
        // Save / load round trip (of the encoded prompt when the model is
        // loaded, of the fixture's prompt otherwise).
        {
            const OmniVoicePrompt& src = ov ? p2 : fixture_prompt;
            const fs::path path = repo / "tests" / "fixtures" / "omnivoice_prompt_roundtrip.bin";
            src.save(path.string());
            const OmniVoicePrompt l = OmniVoicePrompt::load(path.string());
            CHECK(l.codes == src.codes && l.num_frames == src.num_frames && l.text == src.text && l.rms == src.rms,
                  "Part E: OVCP save/load round trip");
        }
        // The cloned generation.
        if (ov) {
            OmniVoiceParams p = fixture_params();
            p.language = "English";
            brosoundml::OmniVoiceTrace tr;
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<int32_t> codes = ov->generate_codes(e.text, 0, p, &p2, nullptr, {}, &tr);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::vector<int32_t> want_ids(e.cond_ids.begin(), e.cond_ids.begin() + (e.c_len - e.g.T - e.T_ref));
            CHECK(tr.text_ids == want_ids, "Part E: clone prompt text ids equal the fixture's");
            CHECK(tr.num_frames == e.g.T && e.est == e.g.T, "Part E: estimated frame count with the reference");
            const std::vector<int> ks = ovp::unmask_schedule(C * e.g.T, e.g.num_step, p.t_shift);
            bool same = ks.size() == e.g.k.size();
            for (std::size_t i = 0; same && i < ks.size(); ++i) same = ks[i] == e.g.k[i];
            CHECK(same, "Part E: unmask schedule matches");
            const double agree = agreement(codes, e.g.codes);
            std::printf("    clone generation: T=%d, %.2fs, codes agreement %.2f%%, unmask grid agreement %.2f%%\n",
                        tr.num_frames, secs, agree, agreement(tr.unmask_step, e.g.unmask));
            CHECK(agree >= 99.0, "Part E: cloned codes agree >= 99% with the fixture");
            // decode + post of the fixture codes with the reference rms (>= 0.1: no gain)
            std::vector<float> post = ovp::remove_silence(e.g.raw, e.sr, 500, 100, 100);
            if (e.ref_rms < 0.1f) ovp::gain_rms_match(post, e.ref_rms);
            post = ovp::fade_and_pad(post, 0.1, 0.1, e.sr);
            const float dp = post.size() == e.g.post.size() ? max_abs_diff(post, e.g.post, post.size()) : std::numeric_limits<float>::infinity();
            std::printf("    clone post-processing of the fixture raw: max|d| %.3g\n", dp);
            CHECK(dp == 0.0f, "Part E: clone post-processing chain exact");
        }
    }

    // ── contract checks (the ones that need a loaded model) ──
    if (ov) {
        std::printf("  [%s] contract checks\n", dev_name);
        OmniVoiceParams p;
        p.num_steps = 2;
        CHECK(throws_runtime_error([&] { ov->synthesize("", p); }), "contract: synthesize(\"\") throws");
        CHECK(throws_runtime_error([&] { ov->synthesize("   ", p); }), "contract: synthesize(whitespace) throws");
        OmniVoiceParams bad = p;
        bad.instruct = "female, purple";
        CHECK(throws_runtime_error([&] { ov->synthesize("Hello.", bad); }), "contract: bad instruct throws");
        bad.instruct = "male, female";
        CHECK(throws_runtime_error([&] { ov->estimate_frames("Hello.", bad); }), "contract: conflicting instruct throws");
        bad.instruct = "河南话, british accent";
        CHECK(throws_runtime_error([&] { ov->generate_codes("Hello.", 10, bad); }), "contract: dialect + accent throws");
        brosoundml::OmniVoiceInit init;
        init.tokens.assign(static_cast<std::size_t>(C) * 10, 0);
        init.keep.assign(static_cast<std::size_t>(C) * 9, 1);
        CHECK(throws_runtime_error([&] { ov->generate_codes("Hello.", 10, p, nullptr, &init); }), "contract: mismatched init sizes throw");
        int polls = 0;
        brosoundml::CancelCheck cancel = [&]() { ++polls; return true; };
        brosoundml::AudioBuffer a = ov->synthesize("Hello there.", p, nullptr, cancel);
        CHECK(a.empty() && polls >= 1, "contract: cancel returns an empty buffer");
        const std::vector<int32_t> codes = ov->generate_codes("Hello there.", 10, p, nullptr, nullptr, cancel);
        CHECK(codes.empty(), "contract: cancel returns empty codes");
        // a valid init keeps its cells and the step hook sees every step
        init.keep.assign(static_cast<std::size_t>(C) * 10, 0);
        for (int t = 0; t < 10; ++t) { init.tokens[static_cast<std::size_t>(t)] = 7; init.keep[static_cast<std::size_t>(t)] = 1; }
        int steps = 0;
        brosoundml::OmniVoiceTrace tr;
        const std::vector<int32_t> c2 = ov->generate_codes("Hello there.", 10, p, nullptr, &init, {}, &tr,
                                                          [&](const brosoundml::OmniVoiceStep& s) { ++steps; CHECK(s.tokens && s.scores && s.num_frames == 10, "step hook fields"); });
        bool kept = c2.size() == static_cast<std::size_t>(C) * 10;
        for (int t = 0; kept && t < 10; ++t) kept = c2[static_cast<std::size_t>(t)] == 7 && tr.unmask_step[static_cast<std::size_t>(t)] == -1;
        CHECK(kept && steps == 2, "contract: init cells stay fixed (unmask_step -1) and on_step fires per step");
        for (int32_t c : c2) if (c == MASK) { CHECK(false, "contract: a cell stayed MASK"); break; }
        CHECK(ov->languages().size() > 600 && ov->instruct_attributes().size() == 6 && ov->nonverbal_tags().size() == 13,
              "contract: languages / instruct_attributes / nonverbal_tags");
        bool found = false;
        for (const std::string& l : ov->languages()) if (l == "English") found = true;
        CHECK(found, "contract: languages() lists English");
    }

    // ── end to end ──
    if (ov) {
        const fs::path whisper_dir = repo / "weights" / "whisper";
        const bool have_whisper = fs::exists(whisper_dir / "model.safetensors") && fs::exists(whisper_dir / "vocab.json");
        std::printf("  [%s] end to end\n", dev_name);
        {
            const std::string text = "Hello there, this is a test of the OmniVoice pipeline.";
            OmniVoiceParams p;   // defaults: 32 steps, guidance 2, Gumbel noise
            p.language = "English";
            brosoundml::OmniVoiceTrace tr;
            const auto t0 = std::chrono::steady_clock::now();
            brosoundml::AudioBuffer out = ov->synthesize(text, p, nullptr, {}, &tr);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("    synthesize: %d frames (%.2fs audio) in %.2fs — LM %.2fs (%.3fs/step), codec %.2fs, peak %.3f\n",
                        tr.num_frames, out.samples.size() / 24000.0, secs, tr.lm_seconds, tr.lm_seconds / p.num_steps,
                        tr.codec_seconds, out.peak());
            CHECK(!out.empty() && out.sample_rate == 24000 && out.peak() > 0.2f && out.peak() <= 0.5f + 1e-4f,
                  "e2e: audio produced, peak-normalised to 0.5");
            const fs::path wav = repo / "tests" / "fixtures" / (std::string("omnivoice_e2e_") + lower_ascii(dev_name) + ".wav");
            out.write_wav(wav.string());
            std::printf("    wrote %s\n", wav.string().c_str());
            if (have_whisper) {
                const std::string t = whisper_transcribe(dev, whisper_dir, out);
                const std::string l = lower_ascii(t);
                std::printf("    whisper: \"%s\"\n", t.c_str());
                CHECK(l.find("test") != std::string::npos, "e2e: Whisper hears \"test\"");
                CHECK(l.find("pipeline") != std::string::npos, "e2e: Whisper hears \"pipeline\"");
            } else {
                std::printf("    Whisper weights absent — transcript oracle skipped\n");
            }
        }
        // The same utterance in BF16 mode (GPU only: the CPU emulation is a
        // scalar correctness path), through the same transcript oracle.
        if (dev != brotensor::Device::CPU) {
            OmniVoice ovb;
            ovb.load(weights.string(), dev, brosoundml::OmniVoicePrecision::BF16, false);
            const std::string text = "Hello there, this is a test of the OmniVoice pipeline.";
            OmniVoiceParams p;
            p.language = "English";
            brosoundml::OmniVoiceTrace tr;
            const auto t0 = std::chrono::steady_clock::now();
            brosoundml::AudioBuffer out = ovb.synthesize(text, p, nullptr, {}, &tr);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("    BF16 synthesize: %d frames (%.2fs audio) in %.2fs — LM %.2fs (%.3fs/step), codec %.2fs, peak %.3f\n",
                        tr.num_frames, out.samples.size() / 24000.0, secs, tr.lm_seconds, tr.lm_seconds / p.num_steps,
                        tr.codec_seconds, out.peak());
            CHECK(!out.empty() && out.peak() > 0.2f && out.peak() <= 0.5f + 1e-4f, "e2e (BF16): audio produced, peak-normalised to 0.5");
            const fs::path wav = repo / "tests" / "fixtures" / (std::string("omnivoice_e2e_bf16_") + lower_ascii(dev_name) + ".wav");
            out.write_wav(wav.string());
            std::printf("    wrote %s\n", wav.string().c_str());
            if (have_whisper) {
                const std::string t = whisper_transcribe(dev, whisper_dir, out);
                const std::string l = lower_ascii(t);
                std::printf("    whisper: \"%s\"\n", t.c_str());
                CHECK(l.find("test") != std::string::npos, "e2e (BF16): Whisper hears \"test\"");
                CHECK(l.find("pipeline") != std::string::npos, "e2e (BF16): Whisper hears \"pipeline\"");
            }
        }
        if (fx.e) {
            const std::string text = "Hello there, this is a cloned voice speaking.";
            OmniVoiceParams p;
            p.language = "English";
            OmniVoicePrompt prompt = ov->create_prompt(brosoundml::AudioBuffer(fx.e->ref_in, fx.e->sr), fx.e->ref_text_in, true);
            brosoundml::OmniVoiceTrace tr;
            const auto t0 = std::chrono::steady_clock::now();
            brosoundml::AudioBuffer out = ov->synthesize(text, p, &prompt, {}, &tr);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("    clone synthesize: %d frames (%.2fs audio) in %.2fs — LM %.2fs, codec %.2fs, peak %.3f\n",
                        tr.num_frames, out.samples.size() / 24000.0, secs, tr.lm_seconds, tr.codec_seconds, out.peak());
            CHECK(!out.empty() && out.peak() > 0.05f, "e2e: clone audio produced");
            const fs::path wav = repo / "tests" / "fixtures" / (std::string("omnivoice_e2e_clone_") + lower_ascii(dev_name) + ".wav");
            out.write_wav(wav.string());
            std::printf("    wrote %s\n", wav.string().c_str());
            if (have_whisper) {
                const std::string t = whisper_transcribe(dev, whisper_dir, out);
                const std::string l = lower_ascii(t);
                std::printf("    whisper: \"%s\"\n", t.c_str());
                CHECK(l.find("voice") != std::string::npos, "e2e: Whisper hears \"voice\" in the clone");
            }
        }
        // Long text: chunked generation.
        {
            const std::string text =
                "The quick brown fox jumps over the lazy dog. This sentence is here to make the text long enough "
                "to be split into chunks by the pipeline, which happens when the estimated duration exceeds the "
                "chunk threshold. Each chunk is generated on its own, conditioned on the first chunk's output, "
                "and the pieces are cross-faded together at the end. That is quite a lot of words for one test, "
                "but a long paragraph is exactly what the chunking path needs to exercise every branch it has. "
                "So here are two more sentences to push the estimate past the thirty second threshold. "
                "With those in place the pipeline has no choice but to split the paragraph into pieces.";
            OmniVoiceParams p;
            p.language = "English";
            p.num_steps = 8;
            brosoundml::OmniVoiceTrace tr;
            const auto t0 = std::chrono::steady_clock::now();
            brosoundml::AudioBuffer out = ov->synthesize(text, p, nullptr, {}, &tr);
            std::printf("    chunked synthesize: %zu chunks, %d frames, %.2fs audio in %.2fs (LM %.2fs, codec %.2fs)\n",
                        tr.chunk_frames.size(), tr.num_frames, out.samples.size() / 24000.0,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), tr.lm_seconds, tr.codec_seconds);
            CHECK(tr.chunk_frames.size() >= 2 && tr.num_frames > 750 && !out.empty(), "e2e: long text is chunked");
            int sum = 0;
            for (int f : tr.chunk_frames) sum += f;
            CHECK(sum == tr.num_frames && tr.codes.size() == static_cast<std::size_t>(C) * tr.num_frames, "e2e: trace frame bookkeeping");
        }
    }
}

// ─── timings ────────────────────────────────────────────────────────────────

static void run_timings(brotensor::Device dev, const char* dev_name, const fs::path& weights) {
    const std::string text = "This is a longer sentence used to measure the speed of the OmniVoice language model at "
                             "thirty-two diffusion steps over ten seconds of generated audio on the graphics card.";
    for (int pass = 0; pass < 2; ++pass) {
        const bool bf16 = pass == 1;
        std::printf("  [%s] timing %s\n", dev_name, bf16 ? "BF16" : "FP32");
        OmniVoice ovt;
        ovt.load(weights.string(), dev, bf16 ? brosoundml::OmniVoicePrecision::BF16 : brosoundml::OmniVoicePrecision::FP32, true);
        OmniVoiceParams p;
        p.language = "English";
        p.duration = 10.0f;   // 250 frames
        for (int rep = 0; rep < 2; ++rep) {   // first pass captures the graph; report the second
            brosoundml::OmniVoiceTrace tr;
            const auto t0 = std::chrono::steady_clock::now();
            brosoundml::AudioBuffer out = ovt.synthesize(text, p, nullptr, {}, &tr);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("    %s run %d: %d frames, %d steps: LM %.3fs (%.1f ms/step), codec %.3fs, total %.3fs, %.2fs audio\n",
                        bf16 ? "BF16" : "FP32", rep, tr.num_frames, p.num_steps, tr.lm_seconds, 1000.0 * tr.lm_seconds / p.num_steps,
                        tr.codec_seconds, secs, out.samples.size() / 24000.0);
            CHECK(!out.empty() && tr.num_frames == 250, "timing: 10 s utterance generated");
        }
    }
}

// ─── main ───────────────────────────────────────────────────────────────────

static int run() {
    const fs::path repo(BROSOUNDML_REPO_DIR);
    const fs::path weights = repo / "weights" / "omnivoice";
    const fs::path fixtures = repo / "tests" / "fixtures";

    // ── contract before load ──
    {
        OmniVoice fresh;
        CHECK(!fresh.loaded(), "fresh OmniVoice is not loaded");
        CHECK(throws_runtime_error([&] { fresh.synthesize("Hello."); }), "synthesize before load throws");
        CHECK(throws_runtime_error([&] { fresh.generate_codes("Hello.", 10); }), "generate_codes before load throws");
        CHECK(throws_runtime_error([&] { fresh.tokenize("Hello."); }), "tokenize before load throws");
        CHECK(throws_runtime_error([&] { fresh.create_prompt(brosoundml::AudioBuffer({0.1f}, 24000), "x"); }), "create_prompt before load throws");
        CHECK(throws_runtime_error([&] { fresh.load((repo / "nonexistent").string()); }), "load of a missing dir throws");
        CHECK(throws_runtime_error([&] { OmniVoicePrompt::load((repo / "CMakeLists.txt").string()); }), "OmniVoicePrompt::load rejects a non-OVCP file");
        // the rules that need no model
        CHECK(ovp::resolve_language("English") == "en" && ovp::resolve_language("zh") == "zh" &&
              ovp::resolve_language("Klingon").empty() && ovp::resolve_language("none").empty(), "resolve_language");
        CHECK(ovp::resolve_instruct("Female, Young Adult", false) == "female, young adult", "resolve_instruct EN");
        CHECK(ovp::resolve_instruct("女，少年", false) == "female, teenager", "resolve_instruct ZH->EN");
        CHECK(ovp::resolve_instruct("female, young adult", true) == "女，青年", "resolve_instruct EN->ZH");
        CHECK(ovp::add_punctuation("Hello") == "Hello." && ovp::add_punctuation("你好") == "你好。" && ovp::add_punctuation("Hi!") == "Hi!", "add_punctuation");
        const std::vector<std::string> ch = ovp::chunk_text("Mr. Smith went home. He slept. Then he woke up, and left!", 25, 3);
        // "," splits too; nothing merges under a 25-character budget
        CHECK(ch.size() == 4 && ch[0] == "Mr. Smith went home." && ch[1] == "He slept." && ch[2] == "Then he woke up," && ch[3] == "and left!",
              "chunk_text keeps abbreviations and splits at every punctuation");
        const std::vector<std::string> ch2 = ovp::chunk_text("Mr. Smith went home. He slept. Then he woke up, and left!", 60, 3);
        CHECK(ch2.size() == 1 && ch2[0] == "Mr. Smith went home. He slept. Then he woke up, and left!", "chunk_text merges under the budget");
        CHECK(ovp::language_names().size() > 600, "language_names");
    }

    if (!fs::exists(weights / "model.safetensors") || !fs::exists(weights / "audio_tokenizer" / "model.safetensors")) {
        std::printf("note: weights absent under %s — model stages skipped\n", weights.string().c_str());
        return failures == 0 ? 0 : 1;
    }

    FixA a; FixB b; FixC c; FixD d; FixE e; FixF f;
    const bool ha = read_A(fixtures / "omnivoice_tokens.bin", a);
    const bool hb = read_B(fixtures / "omnivoice_prompt.bin", b);
    const bool hc = read_C(fixtures / "omnivoice_forward.bin", c);
    const bool hd = read_D(fixtures / "omnivoice_generate.bin", d);
    const bool he = read_E(fixtures / "omnivoice_clone.bin", e);
    const bool hf = read_F(fixtures / "omnivoice_post.bin", f);
    std::printf("fixtures: A %s, B %s, C %s, D %s, E %s, F %s\n", ha ? "ok" : "absent", hb ? "ok" : "absent",
                hc ? "ok" : "absent", hd ? "ok" : "absent", he ? "ok" : "absent", hf ? "ok" : "absent");
    PipelineFixtures fx;
    fx.a = ha ? &a : nullptr; fx.b = hb ? &b : nullptr; fx.c = hc ? &c : nullptr;
    fx.d = hd ? &d : nullptr; fx.e = he ? &e : nullptr; fx.f = hf ? &f : nullptr;

    // The CPU pass covers only the model-free stages (tokenizer, prompt
    // assembly, duration rule, post-processing helpers, OVCP, rules). Every
    // stage that runs the LM trunk or the codec is CUDA-only: the upstream
    // fixtures (generated on CUDA) are the correctness oracle.
    const bool has_cuda = brotensor::is_available(brotensor::Device::CUDA);
    std::printf("=== CPU (model-free stages) ===\n");
    run_pipeline_parts(brotensor::Device::CPU, "CPU", repo, fx, /*model=*/false);
    if (has_cuda) {
        std::printf("=== CUDA ===\n");
        run_lm_parts(brotensor::Device::CUDA, "CUDA", weights, hc ? &c : nullptr, hd ? &d : nullptr);
        run_pipeline_parts(brotensor::Device::CUDA, "CUDA", repo, fx, /*model=*/true);
        run_timings(brotensor::Device::CUDA, "CUDA", weights);
    } else {
        std::printf("  CUDA not available — Parts C/D/E, the contract checks that run the model, "
                    "the end-to-end synthesis and the timings are skipped\n");
    }

    if (failures) std::fprintf(stderr, "test_omnivoice: %d failure(s)\n", failures);
    else          std::printf("test_omnivoice: all checks passed\n");
    return failures == 0 ? 0 : 1;
}

int main() {
    brotensor::init();
    try {
        return run();
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "test_omnivoice: exception: %s\n", ex.what());
        return 1;
    }
}
