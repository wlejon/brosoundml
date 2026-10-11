// Yue2::generate — plan -> semantic -> latents in one call, with the semantic
// session's KV cache reused for synthesis and, for nar_chunk_frames > 0, each
// window synthesized as soon as its codec tokens exist (streaming).
//
// A streaming window [a, b) is conditioned the way a protocol chunk is:
// prompt prefix + codec[a, b) + MUSIC_END on the AR path, then the NAR path
// over b - a frames with latent positions restarting at 0 and the song's
// noise frames [a, b). The prompt rows of its AR K/V are the semantic
// session's own (same tokens at the same positions); window 0 reuses its
// codec rows too. Windows run on the decode thread between tokens, so the
// semantic stream pauses for each window's ODE.

#include "yue2_internal.h"

#include <algorithm>
#include <stdexcept>

namespace brosoundml {

using namespace yue2;

Yue2Result Yue2::generate(const Yue2Params& p, const Yue2Callbacks& cb, const CancelCheck& cancel) const {
    if (!loaded()) throw std::runtime_error("brosoundml: Yue2: not loaded");
    const Impl& I = *impl_;
    const auto t0 = Clock::now();
    Yue2Result res;
    bool done_fired = false;
    auto fire_done = [&](bool c, const std::string& e) {
        if (done_fired) return;
        done_fired = true;
        if (cb.on_done) cb.on_done(c, e);
    };
    try {
        // One test-uniform cursor across both stages (plan first, then semantic).
        const auto cursor = std::make_shared<std::size_t>(0);
        // ── plan ──
        const PlanOut po = plan_impl(I, p, cb, cancel, stage_uniform(p, cursor));
        res.plan = po.plan;
        res.timing.plan_seconds = po.run.seconds;
        res.timing.plan_ttft = po.run.ttft;
        res.timing.plan_tokens = po.run.count;
        res.timing.plan_forwards = po.run.forwards;
        res.timing.plan_tps = po.run.seconds > 0 ? po.run.count / po.run.seconds : 0;
        if (po.run.cancelled || (cancel && cancel())) {
            res.cancelled = true;
            fire_done(true, "");
            return res;
        }
        if (!p.run_semantic) {
            fire_done(false, "");
            return res;
        }

        // ── semantic (+ streaming synthesis) ──
        const Encode enc = [&](const std::string& s) { return I.encode(s); };
        const float g = guidance(p);
        std::vector<int32_t> neg;
        if (g != 1.0f) neg = negative_prefix(p, enc, res.plan.abc_ids);
        const auto& prefix = res.plan.prefix;
        const int P = static_cast<int>(prefix.size());
        const StageHooks sh = make_hooks(I.model, p, Yue2Stage::Semantic, cb);
        const StageHooks ah = make_hooks(I.model, p, Yue2Stage::Semantic, Yue2Callbacks{});
        const StageHooks nh = make_hooks(I.model, p, Yue2Stage::Synthesize, cb);
        const int W = p.nar_chunk_frames;
        const bool synth = p.run_synthesize;
        std::vector<float> noise;
        if (synth) {
            const int budget = p.semantic_sampling.max_tokens;
            noise = p.test_noise.empty() ? song_noise(p, budget)
                                         : std::vector<float>(p.test_noise.begin(), p.test_noise.end());
        }
        double synth_s = 0;
        int ode_done = 0;
        bool synth_cancelled = false;
        auto emit = [&](int a, int b, ArSession* reuse, int cached_codec) {
            if (synth_cancelled) return;
            if (static_cast<std::size_t>(b) * Yue2::kLatentDim > noise.size())
                throw std::runtime_error("brosoundml: Yue2: test_noise is shorter than the song");
            const auto ts = Clock::now();
            ChunkSpec spec;
            spec.a = a;
            spec.b = b;
            if (reuse && p.reuse_prefix_kv) {
                spec.reuse = reuse;
                spec.reuse_rows = P + (a == 0 ? std::min(b, cached_codec) : 0);
            }
            const int windows_known = W > 0 ? (b + W - 1) / W : 1;
            auto lat = synth_chunk(I.model, p, prefix, res.semantic, noise, spec, nh, ah, cancel, [&](int) {
                ++ode_done;
                if (cb.on_progress)
                    cb.on_progress(Yue2Stage::Synthesize, ode_done, std::max(ode_done, p.ode_steps * windows_known));
            });
            synth_s += seconds_since(ts);
            if (lat.empty()) {
                synth_cancelled = true;
                return;
            }
            if (res.frames == 0) res.timing.first_latent_seconds = seconds_since(t0);
            if (cb.on_latents) cb.on_latents(a, b - a, lat.data());
            res.latents.insert(res.latents.end(), lat.begin(), lat.end());
            res.frames = b;
        };

        TokenFn on_tok = [&](int32_t t, ArSession& s) {
            res.semantic.push_back(t - CODEC_OFFSET);
            if (cb.on_semantic) cb.on_semantic(t - CODEC_OFFSET);
            const int n = static_cast<int>(res.semantic.size());
            if (synth && W > 0 && n % W == 0) emit(n - W, n, &s, I.model.ar_len(s, 0) - P);
        };
        std::function<void(int, int)> prog;
        if (cb.on_progress) prog = [&](int d, int t) { cb.on_progress(Yue2Stage::Semantic, d, t); };
        std::unique_ptr<ArSession> sess;
        const auto tsem = Clock::now();
        const ArRun r = run_ar(I.model, Phase::Semantic, prefix, g != 1.0f ? &neg : nullptr, g, p.cot == Yue2Cot::Off,
                               p.semantic_sampling, stage_uniform(p, cursor), sh.get(), cancel, on_tok, prog,
                               &sess);
        res.semantic_truncated = r.truncated;
        res.timing.semantic_seconds = seconds_since(tsem) - synth_s;
        res.timing.semantic_ttft = r.ttft;
        res.timing.semantic_tokens = r.count;
        res.timing.semantic_tps = res.timing.semantic_seconds > 0 ? r.count / res.timing.semantic_seconds : 0;
        if (r.cancelled || synth_cancelled) {
            res.cancelled = true;
            res.timing.synth_seconds = synth_s;
            fire_done(true, "");
            return res;
        }

        // ── remaining synthesis ──
        const int frames = static_cast<int>(res.semantic.size());
        if (synth && frames > res.frames) {
            const int cached = sess ? I.model.ar_len(*sess, 0) - P : 0;
            if (W > 0) {
                emit(res.frames, frames, sess.get(), cached);
            } else {
                for (const auto& [a, b] : chunk_ranges(frames, P)) {
                    emit(a, b, sess.get(), cached);
                    if (synth_cancelled) break;
                }
            }
        }
        sess.reset();
        res.timing.synth_seconds = synth_s;
        if (res.frames > 0)
            res.timing.nar_seconds_per_audio_second = synth_s / (static_cast<double>(res.frames) / kFrameRate);
        if (synth_cancelled) res.cancelled = true;
    } catch (const std::exception& e) {
        fire_done(false, e.what());
        throw;
    }
    fire_done(res.cancelled, "");
    return res;
}

}  // namespace brosoundml
