#pragma once

// OmniVoice language model — the Qwen3-0.6B trunk with audio embeddings and
// the eight-codebook audio heads, plus the masked-diffusion step loop that
// drives it. Internal to src/ (the pipeline in omnivoice.cpp owns one; the
// test includes this header to white-box the forward against the fixture).
//
// One generate() call runs `num_steps` full forwards over a single packed
// sequence:
//
//   rows [0, n_text)                 text ids (style + wrapped text) via embed_tokens
//   rows [n_text, n_text + n_ref)    reference codes: sum over codebooks of
//                                    audio_embeddings[c * V + code]
//   rows [.., Lc)                    the T target frames (same sum over the
//                                    current token grid, MASK while masked)
//   rows [Lc, Lc + T)                the unconditional document: the same T
//                                    target frames alone (guidance_scale != 0)
//
// The two documents attend only within themselves (non-causal GQA flash
// attention over row views) and each restarts its positions at 0, which is
// exactly upstream's two-row batch (the unconditional row is target-only).
// The 2T target rows are gathered, final-normed and projected by audio_heads
// into logits (2T, C*V) — the layout brotensor::masked_diffusion_scores takes.
//
// Per step: rebuild the target-frame embeddings from the token grid (device
// gathers + scatter into the persistent embedding buffer), forward, scores,
// top-k over the (1, C*T) score row, commit. On CUDA the rebuild + forward +
// heads are captured into one CUDA graph per (Lc, T) shape on the first step
// and replayed on every later step; scores / top-k / commit run eagerly
// (the scores op's per-step seed is a kernel argument, so it cannot live
// inside a replayed graph).

#include "brosoundml/audio.h"
#include "brosoundml/omnivoice.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace brosoundml {

// The schedule + selection parameters of one generate() call (the LM-side
// subset of OmniVoiceParams).
struct OmniVoiceLmRun {
    int   num_steps            = 32;
    float t_shift              = 0.1f;
    float guidance_scale       = 2.0f;
    float layer_penalty        = 5.0f;
    float position_temperature = 5.0f;
    float class_temperature    = 0.0f;
    bool  gumbel_noise         = true;
    std::uint64_t seed         = 0;
    std::uint64_t chunk        = 0;   // salts the per-step noise seed per text chunk
};

struct OmniVoiceLmResult {
    std::vector<int32_t> codes;        // C * T, [c * T + t]
    std::vector<int32_t> unmask_step;  // C * T, step index (-1 = fixed by init)
    bool   cancelled = false;
    int    steps_run = 0;
    double seconds   = 0;              // wall time of the step loop
};

// White-box observation for the fixture tests.
struct OmniVoiceLmDebug {
    // Capture the first forward: the packed input embeddings (L, H), the
    // post-final-norm hidden rows at the target positions (R, H) and the
    // logits (R, C*V), R = 2T (or T without guidance).
    bool capture_forward0 = false;
    int L = 0, Lc = 0, R = 0;
    std::vector<float> embeds0, hidden0, logits0;
    // Per step, after the scores op and before the commit: the predicted id
    // and the score of every cell (scores are -inf where already fixed), then
    // the grid after the commit.
    std::function<void(int step, int k, const std::vector<int32_t>& pred,
                       const std::vector<float>& scores,
                       const std::vector<int32_t>& tokens_after)> on_scores;
};

class OmniVoiceLm {
public:
    OmniVoiceLm();
    ~OmniVoiceLm();
    OmniVoiceLm(OmniVoiceLm&&) noexcept;
    OmniVoiceLm& operator=(OmniVoiceLm&&) noexcept;
    OmniVoiceLm(const OmniVoiceLm&) = delete;
    OmniVoiceLm& operator=(const OmniVoiceLm&) = delete;

    // Upload every weight of model.safetensors (keys llm.*, audio_embeddings,
    // audio_heads) to `dev`. `bf16` narrows the projection / MLP / audio_heads
    // weights (embeddings, norms and all activations stay FP32). embed_tokens
    // stays host-resident (620 MB; only a few dozen rows are read per call).
    void load(const brotensor::safetensors::File& f, const OmniVoiceLmConfig& cfg,
              brotensor::Device dev, bool bf16);

    // Run the masked-diffusion schedule. `text_ids` are the conditional
    // prompt's text ids, `ref_codes` C * n_ref reference codes ([c * n_ref +
    // t], n_ref may be 0), T the target frame count. `init` (nullable) fixes
    // cells before the schedule. `cancel` is polled once per step.
    OmniVoiceLmResult generate(const std::vector<int32_t>& text_ids,
                               const std::vector<int32_t>& ref_codes, int n_ref, int T,
                               const OmniVoiceInit* init, const OmniVoiceLmRun& run,
                               const CancelCheck& cancel, const OmniVoiceStepFn& on_step,
                               OmniVoiceLmDebug* dbg) const;

    // sum_c audio_embeddings[c * V + MASK] — the embedding of a masked frame.
    std::vector<float> mask_frame_embedding() const;
    // embed_tokens rows for `ids`, n * H.
    std::vector<float> text_embeddings(const std::vector<int32_t>& ids) const;

    const OmniVoiceLmConfig& config() const;
    bool loaded() const;
    bool bf16() const;
    brotensor::Device device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
