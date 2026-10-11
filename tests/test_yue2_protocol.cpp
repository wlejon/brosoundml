#define _CRT_SECURE_NO_WARNINGS
// YuE2 protocol + sampling parity (no weights, no GPU), against
// tests/fixtures/yue2_protocol.bin from tests/ref/gen_yue2_fixture.py:
//
//   1. qwen.tiktoken through brolm's tiktoken loader: ids == the reference
//      YuE2TextTokenizer.encode (encode_ordinary + NFC) on every string, and
//      decode round-trips.
//   2. Prompt prefixes: token_prefixes (plan / semantic), negative_prefix,
//      request.text() for cot full / melody / off.
//   3. chunk_ranges.
//   4. torch's CPU randn reproduced from mt19937 + ATen's Box-Muller.
//   5. sampling.distribution on fixed logits: removed set identical, kept
//      scores within FP32 rounding, plus the BF16 CFG combine and the legacy
//      cot=off BF16 path.
// Sections skip (printing why) when the fixture or the model folder is absent.

#include "test_yue2_common.h"

#include "yue2_protocol.h"

#include <brolm/qwen_tokenizer.h>

#include <algorithm>
#include <chrono>
#include <limits>

using namespace yue2_test;
namespace y2 = brosoundml::yue2;

static std::vector<std::string> specials() {
    std::vector<std::string> v = {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"};
    for (int i = 0; i < 200; ++i) v.push_back("<extra_" + std::to_string(i) + ">");
    v[204] = "<abc>";
    v[205] = "</abc>";
    return v;
}

static brosoundml::Yue2Sampling read_sampling(Reader& r) {
    brosoundml::Yue2Sampling s;
    s.temperature = r.f32();
    s.top_p = r.f32();
    s.top_k = r.i32();
    s.repetition_penalty = r.f32();
    s.penalty_window = r.i32();
    s.min_tokens = r.i32();
    s.max_tokens = r.i32();
    return s;
}

int main() {
    const fs::path mdir = model_dir();
    Reader r(fixture("yue2_protocol.bin"));
    if (!r) {
        std::printf("SKIP: tests/fixtures/yue2_protocol.bin absent (run tests/ref/gen_yue2_fixture.py)\n");
        return 0;
    }
    CHECK(r.i32() == 0x31505559, "protocol fixture magic");
    const bool have_tok = !mdir.empty() && fs::exists(mdir / "qwen.tiktoken");
    if (!have_tok) std::printf("SKIP: tokenizer sections (no YuE2-3B folder)\n");
    std::unique_ptr<brolm::qwen::Tokenizer> tok;
    if (have_tok)
        tok = std::make_unique<brolm::qwen::Tokenizer>(
            brolm::qwen::Tokenizer::from_tiktoken((mdir / "qwen.tiktoken").string(), specials()));

    // 1. tokenizer
    const int ns = r.i32();
    int tok_ok = 0;
    for (int i = 0; i < ns; ++i) {
        const std::string s = r.str();
        const auto ids = r.i32list();
        if (!tok) continue;
        const auto got = tok->encode_ordinary(s);
        if (got == ids) ++tok_ok;
        else CHECK(false, "tokenizer ids differ on string %d (%zu vs %zu ids)", i, got.size(), ids.size());
    }
    if (tok) {
        CHECK(tok->vocab_count() == 151643, "ordinary vocabulary size");
        const std::string s = "\xE4\xBB\x8A\xE6\x99\x9A\xE4\xB8\x8D\xE7\x9C\xA0 X:1\nK:C\n";
        CHECK(tok->decode(tok->encode_ordinary(s)) == s, "decode round trip");
        std::printf("tokenizer: %d/%d strings identical\n", tok_ok, ns);
    }

    // 2. prompts
    const int np = r.i32();
    for (int i = 0; i < np; ++i) {
        brosoundml::Yue2Params p;
        p.style = r.str();
        p.lyrics = r.str();
        const int cot = r.i32();
        p.cot = cot == 0 ? brosoundml::Yue2Cot::Off : cot == 1 ? brosoundml::Yue2Cot::Melody : brosoundml::Yue2Cot::Full;
        const std::string abc = r.str();
        const auto plan_prefix = r.i32list(), abc_ids = r.i32list(), sem_prefix = r.i32list(), neg = r.i32list();
        const std::string text = r.str();
        CHECK(y2::request_text(p.cot, p.style, p.lyrics) == text, "request text %d", i);
        if (!tok) continue;
        const y2::Encode enc = [&](const std::string& s) { return tok->encode_ordinary(s); };
        CHECK(y2::token_prefix(p, enc, nullptr) == plan_prefix, "plan prefix %d", i);
        const auto ids = cot == 0 ? std::vector<int32_t>{} : tok->encode_ordinary(abc);
        CHECK(ids == abc_ids, "abc ids %d", i);
        CHECK(y2::token_prefix(p, enc, cot == 0 ? nullptr : &ids) == sem_prefix, "semantic prefix %d", i);
        CHECK(y2::negative_prefix(p, enc, ids) == neg, "negative prefix %d", i);
    }
    std::printf("prompts: %d cases\n", np);

    // 3. chunk_ranges
    const int nr = r.i32();
    for (int i = 0; i < nr; ++i) {
        const int frames = r.i32(), pt = r.i32();
        const auto flat = r.i32list();
        std::vector<int32_t> got;
        for (auto [a, b] : y2::chunk_ranges(frames, pt)) {
            got.push_back(a);
            got.push_back(b);
        }
        CHECK(got == flat, "chunk_ranges(%d, %d)", frames, pt);
    }

    // 4. torch CPU randn
    {
        const uint64_t seed = r.u64();
        const int n = r.i32();
        const auto ref = r.f32arr(n);
        const auto got = y2::torch_cpu_randn(seed, n);
        const Diff d = diff(got, ref);
        std::printf("torch_cpu_randn(seed %llu, %d): max |diff| %.3g\n", static_cast<unsigned long long>(seed), n,
                    d.max_abs);
        CHECK(d.max_abs < 1e-5, "torch CPU randn reproduced (max %.3g)", d.max_abs);
    }

    // 5. distribution
    const int nc = r.i32();
    for (int c = 0; c < nc; ++c) {
        const y2::Phase ph = r.i32() == 0 ? y2::Phase::Abc : y2::Phase::Semantic;
        const auto s = read_sampling(r);
        const int step = r.i32();
        const bool legacy = r.i32() != 0;
        const float cfg = r.f32();
        const auto hist = r.i32list();
        const int n = r.i32();
        const auto cond = r.f32arr(n);
        std::vector<float> uncond;
        if (cfg != 1.0f) uncond = r.f32arr(n);
        const auto combined = r.f32arr(n);
        const auto ref = r.f32arr(n);
        std::vector<float> logits = cond;
        if (cfg != 1.0f) {
            y2::cfg_combine_bf16(cond.data(), uncond.data(), cfg, logits.data(), n);
            const Diff dc = diff(logits, combined);
            CHECK(dc.max_abs == 0.0, "case %d: BF16 CFG combine (max %.3g)", c, dc.max_abs);
        }
        const auto got = y2::distribution(logits.data(), n, ph, s, hist, step, legacy);
        int mask_diff = 0, kept = 0;
        std::vector<float> diff_vals;
        double max_err = 0;
        for (int i = 0; i < n; ++i) {
            const bool gi = std::isinf(got[i]), ri = std::isinf(ref[i]);
            if (gi != ri) {
                ++mask_diff;
                diff_vals.push_back(gi ? ref[i] : got[i]);
            } else if (!gi) {
                ++kept;
                max_err = std::max(max_err, static_cast<double>(std::fabs(got[i] - ref[i])) /
                                                std::max(1.0f, std::fabs(ref[i])));
            }
        }
        std::printf("distribution case %d (%s, T=%.2g, step %d%s): kept %d, mask diffs %d, max rel err %.3g\n", c,
                    ph == y2::Phase::Abc ? "abc" : "semantic", s.temperature, step, legacy ? ", legacy" : "", kept,
                    mask_diff, max_err);
        // In BF16 many scores tie; which of a tied run the top-p cut keeps is
        // the sort's tie order (CPU introsort in the fixture, CUDA's stable
        // radix sort in the pipeline), so a legacy case may swap kept ids
        // within one tied value — the same multiset of kept scores.
        bool tie_swap = legacy && mask_diff % 2 == 0;
        for (float v : diff_vals) tie_swap = tie_swap && v == diff_vals[0];
        if (mask_diff) std::printf("  mask diffs all at score %g: %s\n", diff_vals[0], tie_swap ? "yes" : "no");
        CHECK(mask_diff == 0 || tie_swap, "case %d: removed set differs at %d ids", c, mask_diff);
        CHECK(max_err < (legacy ? 1e-2 : 1e-6), "case %d: kept scores (rel %.3g)", c, max_err);
        // A draw at a fixed uniform lands inside the kept set.
        if (s.temperature > 0) {
            const auto [lo, hi] = y2::logit_range(ph);
            (void)hi;
            const int32_t t = y2::draw(got, lo, s.temperature, [] { return 0.5; });
            CHECK(!std::isinf(ref[t - lo]), "case %d: draw outside the kept set", c);
        }
    }

    // Host cost of one sampled token (distribution + draw), the per-token CPU
    // work between two decode steps.
    for (const y2::Phase ph : {y2::Phase::Abc, y2::Phase::Semantic}) {
        const auto [lo, hi] = y2::logit_range(ph);
        const int n = hi - lo;
        std::vector<float> lg(n);
        uint32_t st = 12345;
        for (auto& v : lg) {
            st = st * 1664525u + 1013904223u;
            v = static_cast<float>(st >> 8) * (8.0f / 16777216.0f) - 4.0f;
        }
        brosoundml::Yue2Params p;
        const auto& s = ph == y2::Phase::Abc ? p.abc_sampling : p.semantic_sampling;
        std::vector<int32_t> hist;
        for (int i = 0; i < 400; ++i) hist.push_back(lo + (i * 37) % n);
        const auto u = y2::make_uniform(1);
        const auto t0 = std::chrono::steady_clock::now();
        int32_t sink = 0;
        for (int i = 0; i < 200; ++i) {
            const auto sc = y2::distribution(lg.data(), n, ph, s, hist, 300, false);
            sink += y2::draw(sc, lo, s.temperature, u);
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200;
        std::printf("host sampling cost (%s, %d logits): %.0f us per token (%d)\n",
                    ph == y2::Phase::Abc ? "abc" : "semantic", n, us, sink & 1);
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
