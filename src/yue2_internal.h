#pragma once

// Yue2 pipeline internals shared by yue2.cpp (public class, stage seams) and
// yue2_synth.cpp (NAR chunking + streaming windows).

#include "brosoundml/yue2.h"

#include "yue2_model_impl.h"
#include "yue2_protocol.h"

#include <brolm/qwen_tokenizer.h>

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace brosoundml {

namespace yue2 {
// The loaded pipeline state (Yue2::Impl is this; free functions take Core).
struct Core {
    Model model;
    std::optional<brolm::qwen::Tokenizer> tok;
    std::string dir;

    std::vector<int32_t> encode(const std::string& s) const { return tok->encode_ordinary(s); }
};
}  // namespace yue2

struct Yue2::Impl : yue2::Core {};

namespace yue2 {

using Clock = std::chrono::steady_clock;
inline double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Device-side hooks of one stage, built from Yue2Params::steer / capture.
struct StageHooks {
    LayerHooks hooks;
    bool active = false;
    const LayerHooks* get() const { return active ? &hooks : nullptr; }
};
StageHooks make_hooks(const Model& m, const Yue2Params& p, Yue2Stage stage, const Yue2Callbacks& cb);

// Uniform source of one stage: Yue2Params::test_uniforms from `*cursor` on,
// else a fresh generator seeded with the request seed (each stage reseeds,
// as the reference does).
UniformFn stage_uniform(const Yue2Params& p, const std::shared_ptr<std::size_t>& cursor);

// One AR stage's outcome (sampling.generate_tokens).
struct ArRun {
    std::vector<int32_t> tokens;   // sampled ids before the end token
    bool truncated = false;
    bool cancelled = false;
    double seconds = 0, ttft = 0;
    int count = 0;                 // tokens output, the end token included
};

// Called after each sampled non-end token, before it is fed back; the session
// then caches the prefix plus every earlier token.
using TokenFn = std::function<void(int32_t token, ArSession& session)>;

ArRun run_ar(const Model& m, Phase ph, const std::vector<int32_t>& prefix,
             const std::vector<int32_t>* negative, float cfg, bool legacy_off, const Yue2Sampling& s,
             const UniformFn& uniform, const LayerHooks* hooks, const CancelCheck& cancel,
             const TokenFn& on_token, const std::function<void(int, int)>& on_progress,
             std::unique_ptr<ArSession>* keep);

// The NAR noise for frames [0, frames): the test seam's, else the seed's
// torch CPU draw (prefix-stable: frame f's values do not depend on `frames`).
std::vector<float> song_noise(const Yue2Params& p, int frames);

// Synthesize one chunk/window and return its FP32 latents (empty = cancelled).
struct ChunkSpec {
    int a = 0, b = 0;                         // frames [a, b)
    const ArSession* reuse = nullptr;         // semantic session (or null)
    int reuse_rows = 0;
};
std::vector<float> synth_chunk(const Model& m, const Yue2Params& p, const std::vector<int32_t>& prefix,
                               const std::vector<int32_t>& codec, const std::vector<float>& noise,
                               const ChunkSpec& spec, const StageHooks& nar_hooks, const StageHooks& ar_hooks,
                               const CancelCheck& cancel, const std::function<void(int)>& on_step);

// The plan stage (yue2.cpp): the ABC sampled (or supplied) and the semantic
// prefix it makes.
struct PlanOut {
    Yue2Plan plan;
    ArRun run;
};
PlanOut plan_impl(const Core& I, const Yue2Params& p, const Yue2Callbacks& cb, const CancelCheck& cancel,
                  const UniformFn& uniform);

// Valid UTF-8 with each invalid sequence replaced by U+FFFD (Python's
// errors="replace").
std::string utf8_replace(const std::string& s);

}  // namespace yue2
}  // namespace brosoundml
