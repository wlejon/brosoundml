// Yue2 — the public pipeline (see include/brosoundml/yue2.h): prompts, the
// plan and semantic AR stages, NAR synthesis (protocol chunks or streaming
// windows), callbacks and timing. The backbone is yue2_model.h.

#include "yue2_internal.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <stdexcept>

namespace brosoundml {

namespace {

using namespace yue2;

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("brosoundml: Yue2: " + m); }

const std::vector<std::string>& specials() {
    static const std::vector<std::string> s = [] {
        std::vector<std::string> v = {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>",
                                      "<S>",           "<X>",          "<mask>",     "<sep>"};
        for (int i = 0; i < 200; ++i) v.push_back("<extra_" + std::to_string(i) + ">");
        v[204] = "<abc>";
        v[205] = "</abc>";
        return v;
    }();
    return s;
}

void check_plan(const Core& I, const Yue2Params& p, const Yue2Plan& plan) {
    const Encode enc = [&](const std::string& s) { return I.encode(s); };
    const auto expect = p.cot == Yue2Cot::Off ? token_prefix(p, enc, nullptr) : token_prefix(p, enc, &plan.abc_ids);
    if (expect != plan.prefix) fail("plan prefix disagrees with the request / exact ABC ids");
}

}  // namespace

namespace yue2 {

PlanOut plan_impl(const Core& I, const Yue2Params& p, const Yue2Callbacks& cb,
                  const CancelCheck& cancel, const UniformFn& uniform) {
    const Encode enc = [&](const std::string& s) { return I.encode(s); };
    PlanOut o;
    if (p.cot == Yue2Cot::Off) {
        if (!p.abc.empty()) fail("an external ABC requires cot = Melody / Full");
        o.plan.prefix = token_prefix(p, enc, nullptr);
        return o;
    }
    if (!p.abc.empty()) {
        o.plan.abc = p.abc;
        o.plan.abc_ids = I.encode(p.abc);
        o.plan.prefix = token_prefix(p, enc, &o.plan.abc_ids);
        return o;
    }
    const StageHooks hooks = make_hooks(I.model, p, Yue2Stage::Plan, cb);
    const auto prefix = token_prefix(p, enc, nullptr);
    TokenFn on_tok;
    if (cb.on_abc)
        on_tok = [&](int32_t t, ArSession&) { cb.on_abc(I.tok->decode({t}), t); };
    std::function<void(int, int)> prog;
    if (cb.on_progress) prog = [&](int d, int t) { cb.on_progress(Yue2Stage::Plan, d, t); };
    o.run = run_ar(I.model, Phase::Abc, prefix, nullptr, 1.0f, false, p.abc_sampling, uniform, hooks.get(),
                   cancel, on_tok, prog, nullptr, p.abc_draft_tokens);
    o.plan.abc_ids = o.run.tokens;
    o.plan.abc = utf8_replace(I.tok->decode(o.plan.abc_ids));
    o.plan.prefix = token_prefix(p, enc, &o.plan.abc_ids);
    o.plan.truncated = o.run.truncated;
    return o;
}

}  // namespace yue2

Yue2::Yue2() : impl_(std::make_unique<Impl>()) {}
Yue2::~Yue2() = default;
Yue2::Yue2(Yue2&&) noexcept = default;
Yue2& Yue2::operator=(Yue2&&) noexcept = default;

bool Yue2::loaded() const { return impl_ && impl_->model.loaded() && impl_->tok.has_value(); }
brotensor::Device Yue2::device() const { return impl_->model.device(); }

void Yue2::load(const std::string& model_dir, brotensor::Device device, const Yue2LoadOptions& options) {
    brotensor::init();
    auto impl = std::make_unique<Impl>();
    impl->dir = model_dir;
    impl->tok = brolm::qwen::Tokenizer::from_tiktoken(model_dir + "/qwen.tiktoken", specials());
    if (impl->tok->vocab_count() != 151643) fail("expected the checkpoint-native qwen.tiktoken (151643 tokens)");
    impl->model.load(model_dir, device, options.ar_weights, options.keep_bf16_ar);
    impl_ = std::move(impl);
}

std::vector<int32_t> Yue2::tokenize(const std::string& text) const {
    if (!impl_->tok) fail("not loaded");
    return impl_->encode(text);
}

std::string Yue2::detokenize(const std::vector<int32_t>& ids) const {
    if (!impl_->tok) fail("not loaded");
    std::vector<int32_t> kept;
    const int32_t n_vocab = 151643 + static_cast<int32_t>(specials().size());
    for (int32_t i : ids)
        if (i >= 0 && i < n_vocab) kept.push_back(i);
    return utf8_replace(impl_->tok->decode(kept));
}

Yue2Plan Yue2::plan(const Yue2Params& p, const Yue2Callbacks& cb, const CancelCheck& cancel) const {
    if (!loaded()) fail("not loaded");
    return plan_impl(*impl_, p, cb, cancel, stage_uniform(p, std::make_shared<std::size_t>(0))).plan;
}

std::vector<int32_t> Yue2::semantic(const Yue2Params& p, const Yue2Plan& plan, const Yue2Callbacks& cb,
                                    const CancelCheck& cancel, bool* truncated) const {
    if (!loaded()) fail("not loaded");
    const Impl& I = *impl_;
    check_plan(I, p, plan);
    const Encode enc = [&](const std::string& s) { return I.encode(s); };
    const float g = guidance(p);
    std::vector<int32_t> neg;
    if (g != 1.0f) neg = negative_prefix(p, enc, plan.abc_ids);
    const StageHooks hooks = make_hooks(I.model, p, Yue2Stage::Semantic, cb);
    TokenFn on_tok;
    if (cb.on_semantic) on_tok = [&](int32_t t, ArSession&) { cb.on_semantic(t - CODEC_OFFSET); };
    std::function<void(int, int)> prog;
    if (cb.on_progress) prog = [&](int d, int t) { cb.on_progress(Yue2Stage::Semantic, d, t); };
    const ArRun r = run_ar(I.model, Phase::Semantic, plan.prefix, g != 1.0f ? &neg : nullptr, g,
                           p.cot == Yue2Cot::Off, guarded_semantic(p, plan.abc),
                           stage_uniform(p, std::make_shared<std::size_t>(0)), hooks.get(), cancel, on_tok, prog,
                           nullptr);
    if (truncated) *truncated = r.truncated;
    std::vector<int32_t> codes;
    for (int32_t t : r.tokens) codes.push_back(t - CODEC_OFFSET);
    return codes;
}

std::vector<float> Yue2::synthesize(const Yue2Params& p, const Yue2Plan& plan, const std::vector<int32_t>& semantic,
                                    const Yue2Callbacks& cb, const CancelCheck& cancel) const {
    if (!loaded()) fail("not loaded");
    const Impl& I = *impl_;
    check_plan(I, p, plan);
    for (int32_t c : semantic)
        if (c < 0 || c >= CODEC_SIZE) fail("codec id outside 0..32767");
    std::vector<float> out;
    bool cancelled = false;
    try {
        const int frames = static_cast<int>(semantic.size());
        if (frames > 0) {
            const auto noise = song_noise(p, frames);
            const StageHooks nh = make_hooks(I.model, p, Yue2Stage::Synthesize, cb);
            const StageHooks ah = make_hooks(I.model, p, Yue2Stage::Semantic, Yue2Callbacks{});
            std::vector<std::pair<int, int>> ranges;
            if (p.nar_chunk_frames > 0)
                for (int a = 0; a < frames; a += p.nar_chunk_frames)
                    ranges.emplace_back(a, std::min(frames, a + p.nar_chunk_frames));
            else
                ranges = chunk_ranges(frames, static_cast<int>(plan.prefix.size()));
            const int total = p.ode_steps * static_cast<int>(ranges.size());
            int done = 0;
            for (const auto& [a, b] : ranges) {
                ChunkSpec spec;
                spec.a = a;
                spec.b = b;
                auto lat = synth_chunk(I.model, p, plan.prefix, semantic, noise, spec, nh, ah, cancel, [&](int) {
                    ++done;
                    if (cb.on_progress) cb.on_progress(Yue2Stage::Synthesize, done, total);
                });
                if (lat.empty()) {
                    cancelled = true;
                    break;
                }
                if (cb.on_latents) cb.on_latents(a, b - a, lat.data());
                out.insert(out.end(), lat.begin(), lat.end());
            }
        }
    } catch (const std::exception& e) {
        if (cb.on_done) cb.on_done(false, e.what());
        throw;
    }
    if (cb.on_done) cb.on_done(cancelled, "");
    return out;
}

}  // namespace brosoundml
