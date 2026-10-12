#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── YuE2-3B (song generation: lyrics + style -> score -> semantic -> latents) ─
//
// YuE2 (m-a-p, 2026) is one AR–NAR Mixture-of-Transformers backbone: every
// layer carries two attention projections and two MLPs, an AR path (causal,
// KV cached, next-token over a 184704-way vocabulary) and a NAR path (flow
// matching over 64-dim acoustic latents). Shape: Qwen3-like, 2048 hidden,
// 28 layers, GQA 16/8, head_dim 128, q/k RMSNorm, RoPE theta 1e6, SwiGLU 6144,
// 24576 context. Weights are BF16 (CC BY-NC 4.0, data shipped with
// attribution). Pipeline — the order brosoundml drives it, matching the
// reference yue2 package (protocol "yue2-native-v1"):
//
//   1. Plan        (cot = Full / Melody) AR sampling of an ABC score after
//                  [EOD] instruction+[Tags]style+[Lyrics]lyrics [ABC_START].
//                  Skipped when an ABC is supplied or cot = Off.
//   2. Semantic    AR sampling of codec tokens (25 Hz, 32768 codes) after the
//                  plan prefix [... ABC_END MUSIC_START], with optional CFG
//                  against a negative prefix (instruction only + same ABC).
//   3. Synthesize  NAR flow matching: the AR prefix (prompt + codec tokens)
//                  is prefilled once per chunk on the AR path, then a
//                  midpoint ODE (ode_steps) integrates noise -> latents on the
//                  NAR path, attending to that cached prefix.
//   4. Decode      NOT in this module. The YuE2 VAE (yue2_vae.h) turns the
//                  latents into 48 kHz stereo; a binding chains on_latents
//                  into its streaming decoder.
//
// ── Contract with the VAE / binding (do not change without telling them) ──
//   * Latents are frame-major float[frames * 64], i.e. [frame][channel], at
//     25 Hz, already in VAE-input space: exactly what the VAE decoder eats
//     (the reference passes NAR output to vae.decode with no scale/shift).
//   * on_latents emits only FINAL frames: frame0 starts at 0, is monotonic
//     and contiguous across calls, and no frame is ever re-emitted.
//   * on_done fires exactly once per generate() that reached synthesis or
//     was cancelled/failed after starting — after the last on_latents — so a
//     streaming decoder knows to flush its held-back halo. `cancelled` /
//     `error` say why it ended early.
//   * generate() never touches the VAE.

enum class Yue2Cot { Off, Melody, Full };

// Weights the AR decode (one token per forward: plan and semantic sampling)
// reads. Decode is weight-bandwidth bound, so its speed is the bytes per
// token. Bf16 = the checkpoint's. Int8 / Fp8 = weight-only quantised copies,
// one FP32 scale per output row (INT8 symmetric +-127, or FP8 E4M3), of every
// layer's AR q/k/v/o/gate/up/down and of lm_head; activations, accumulation
// and everything else stay as in Bf16. The NAR path and the VAE always use
// the BF16 weights. With Int8 / Fp8 the AR path's BF16 originals are dropped
// once quantised (keep_bf16_ar = false, the default): prefill dequantizes one
// matrix at a time into a shared scratch, so it runs on the same weights the
// decode reads, and the model holds ~3.5 GB less than with both copies.
// Int8 by default (ABC plan ~1.6x faster); Bf16 is the reference-parity
// choice, the one to pass when comparing with the PyTorch model.
enum class Yue2ArWeights { Bf16, Int8, Fp8 };

struct Yue2LoadOptions {
    Yue2ArWeights ar_weights = Yue2ArWeights::Int8;
    bool keep_bf16_ar = false;   // keep the BF16 AR weights beside the quantised copies (prefill reads them)
};

// One AR stage's sampling (protocol.Sampling). Defaults are the semantic
// stage's; Yue2Params::abc_sampling defaults to the ABC preset.
struct Yue2Sampling {
    float temperature        = 1.0f;   // 0 = greedy
    float top_p              = 0.95f;
    int   top_k              = 100;
    float repetition_penalty = 1.2f;   // windowed, sign-aware (logit<0 ? *a : /a)
    int   penalty_window     = 50;     // 1..100 most recent tokens
    int   min_tokens         = 200;    // end token masked before this
    int   max_tokens         = 9000;
    // Not protocol: after `end_bias_after` tokens (0 = never) the end
    // token's logit gains `end_bias_per_token` per token past it, so a song
    // that has outrun its plan ends (Yue2Params::semantic_length_guard).
    int   end_bias_after     = 0;
    float end_bias_per_token = 0.0f;
};

enum class Yue2Stage { Plan, Semantic, Synthesize };

// ── Residual-stream steering + capture (interpretability seam) ──────────────
// Both act on the residual stream AFTER a decoder layer (its MLP add — the
// stream layer + 1 reads), on the AR path (plan / semantic stages) and the
// NAR path (synthesis). Empty = off; when off the forward is unchanged and
// pays nothing. Like QwenTtsSampling::voice_steer, a delta is plain host
// floats of hidden_size (2048) width.
struct Yue2Steer {
    bool nar = false;          // false: AR path, true: NAR path
    int  layer = 0;            // 0 .. 27
    // hidden floats: added at every position. rows * hidden floats with
    // per_position: row r is added at position pos0 + r (AR: absolute token
    // position in the stage's sequence; NAR: absolute latent frame), rows
    // outside a forward's positions are ignored.
    std::vector<float> delta;
    float scale = 1.0f;
    bool  per_position = false;
    int   pos0 = 0;
    bool  plan = true, semantic = true;   // AR stages it applies to
};

struct Yue2Capture {
    std::vector<int> layers;              // AR + NAR layers to report (empty = off)
    bool plan = true, semantic = true;    // AR stages reported (branch 0 = conditional)
    // NAR velocity evaluations to report, counted per chunk/window from 0
    // (2 per ODE step: step s = evals 2s, 2s+1). Empty = no NAR capture.
    std::vector<int> nar_evals;
};

// One captured residual block (pointer valid during the callback only).
struct Yue2Hidden {
    Yue2Stage stage = Yue2Stage::Plan;
    int layer = 0;
    int pos0 = 0;          // AR: first token position; NAR: first latent frame
    int rows = 0;          // positions in `data`
    int hidden = 0;        // floats per row
    int nar_eval = -1;     // NAR: velocity evaluation within its chunk
    const float* data = nullptr;   // rows * hidden, FP32
};

struct Yue2Params {
    std::string style;                 // "[Tags]" text
    std::string lyrics;                // "[Lyrics]" text (sections like [verse])
    Yue2Cot     cot  = Yue2Cot::Full;
    std::string abc;                   // non-empty: use this score, skip planning
    std::uint64_t seed = 831001;       // each AR stage reseeds with it; NAR noise too
    float cfg_scale = -1.0f;           // < 0: protocol default (1.0 Full/Melody, 1.01 Off)

    Yue2Sampling abc_sampling{0.7f, 0.9f, 30, 1.005f, 100, 32, 4096};
    Yue2Sampling semantic_sampling{};

    // Speculative ABC planning: each decode forward also verifies up to this
    // many draft tokens copied from earlier in the prompt + score (prompt
    // lookup: what followed the latest earlier match of the last few tokens),
    // accepted by standard speculative sampling, so the plan's distribution
    // is exactly the one-token sampler's (a seed draws a different plan from
    // the one-token path's, though; greedy is the same plan). 0 = off (the
    // one-token path), at most 7; 5 by default. CUDA or the CPU.
    int abc_draft_tokens = 5;

    int ode_steps = 32;                // midpoint steps per chunk
    // NAR chunking. 0 = the protocol's chunks (chunk_ranges over the 24576
    // context: in practice one chunk per song, so latents arrive only after
    // all semantic tokens). > 0 = streaming mode: synthesize each window of
    // this many frames as soon as its semantic tokens exist (a quality /
    // latency trade-off; see docs).
    int nar_chunk_frames = 0;

    bool run_semantic   = true;        // false: stop after the plan
    bool run_synthesize = true;        // false: stop after semantic tokens

    std::vector<Yue2Steer> steer;      // residual-stream steering (empty = off)
    Yue2Capture capture;               // residual capture -> Yue2Callbacks::on_hidden

    // Reuse the semantic stage's KV cache for synthesis instead of prefilling
    // prompt + codec again (generate() only; protocol chunks and the prompt
    // part of streaming windows). Same weights and positions, so the K/V
    // differ from a fresh prefill only by kernel rounding.
    bool reuse_prefix_kv = true;

    // Song length guard (not protocol). The plan's length is estimated from
    // its ABC (the vocal voice's bars, M: and Q:; normal songs end within
    // ~1.03x of it). Past 1.15x the end token's logit gains 0.04 per token
    // (+1 a second), and the song is cut at 1.5x (at least + 20 s): a song a
    // strong semantic steer keeps from ending (|scale| >= 3 ran to 9000
    // tokens) ends instead. Songs that end on their own are unchanged. No
    // effect without an ABC (cot = Off) or with semantic_sampling's own
    // end_bias_after set.
    bool semantic_length_guard = true;

    // Test seam: uniforms for the AR token draws (one per sampled token, in
    // order; plan first, then semantic) replacing the seeded generator, and
    // the NAR noise (frames * 64) replacing the seed's torch CPU draw.
    std::vector<double> test_uniforms;
    std::vector<float>  test_noise;
};

// Stage results, also the seams a lab drives stage by stage.
struct Yue2Plan {
    std::string          abc;          // empty for cot = Off
    std::vector<int32_t> abc_ids;
    std::vector<int32_t> prefix;       // token prefix for the semantic stage
    bool truncated = false;            // hit max_tokens without ABC_END
};

struct Yue2Timing {
    double plan_seconds = 0, semantic_seconds = 0, synth_seconds = 0;
    double plan_ttft = 0, semantic_ttft = 0;     // first token after prefill
    int    plan_tokens = 0, semantic_tokens = 0;
    double plan_tps = 0, semantic_tps = 0;
    int    plan_forwards = 0;            // decode forwards of the plan (< tokens with drafts)
    // generate() only: wall time from the call to the first on_latents, and
    // the NAR seconds spent per second of audio synthesized.
    double first_latent_seconds = 0;
    double nar_seconds_per_audio_second = 0;
};

struct Yue2Result {
    Yue2Plan             plan;
    std::vector<int32_t> semantic;     // codec ids 0..32767 (CODEC_OFFSET removed)
    bool                 semantic_truncated = false;
    std::vector<float>   latents;      // frames * 64, frame-major
    int                  frames = 0;
    Yue2Timing           timing;
    bool                 cancelled = false;
};

struct Yue2Callbacks {
    // Residual blocks requested by Yue2Params::capture. Yue2Hidden::data is
    // valid only during the call; copy what you keep.
    std::function<void(const Yue2Hidden&)> on_hidden;
    // ABC text as it is sampled (UTF-8 pieces; may split a code point only at
    // token boundaries the tokenizer produces — the binding buffers).
    std::function<void(const std::string& piece, int32_t token)> on_abc;
    // Each sampled codec id (0..32767), in order.
    std::function<void(int32_t code)> on_semantic;
    // Final latent frames [frame0, frame0 + frames), frame-major * 64.
    std::function<void(int frame0, int frames, const float* data)> on_latents;
    // Progress within a stage: tokens so far / budget, or ODE steps done /
    // total over all chunks.
    std::function<void(Yue2Stage stage, int done, int total)> on_progress;
    // Exactly once, last. `error` empty unless generate() is about to throw.
    std::function<void(bool cancelled, const std::string& error)> on_done;
};

class Yue2 {
public:
    Yue2();
    ~Yue2();
    Yue2(Yue2&&) noexcept;
    Yue2& operator=(Yue2&&) noexcept;
    Yue2(const Yue2&) = delete;
    Yue2& operator=(const Yue2&) = delete;

    // config.json, model.safetensors and qwen.tiktoken from `model_dir`
    // (the YuE2-3B folder), weights on `device`. Throws std::runtime_error.
    void load(const std::string& model_dir,
              brotensor::Device device = brotensor::Device::CUDA,
              const Yue2LoadOptions& options = {});

    // The whole pipeline up to latents (see Yue2Params::run_*). `cancel` is
    // polled per token and per ODE step; a cancel returns what exists with
    // result.cancelled set (and fires on_done(true, "")).
    Yue2Result generate(const Yue2Params& params,
                        const Yue2Callbacks& callbacks = {},
                        const CancelCheck& cancel = {}) const;

    // Stage seams, for a lab that edits between stages (e.g. edit the ABC,
    // re-roll semantic with a new seed, re-synthesize with other ode_steps).
    Yue2Plan plan(const Yue2Params& params, const Yue2Callbacks& callbacks = {},
                  const CancelCheck& cancel = {}) const;
    std::vector<int32_t> semantic(const Yue2Params& params, const Yue2Plan& plan,
                                  const Yue2Callbacks& callbacks = {},
                                  const CancelCheck& cancel = {},
                                  bool* truncated = nullptr) const;
    // plan() and semantic() never fire on_done; synthesize() fires on_latents
    // per chunk/window and then on_done exactly once (also on cancel, and with
    // the error before it throws), as generate() does.
    std::vector<float> synthesize(const Yue2Params& params, const Yue2Plan& plan,
                                  const std::vector<int32_t>& semantic,
                                  const Yue2Callbacks& callbacks = {},
                                  const CancelCheck& cancel = {}) const;

    std::vector<int32_t> tokenize(const std::string& text) const;
    std::string detokenize(const std::vector<int32_t>& ids) const;

    bool loaded() const;
    brotensor::Device device() const;

    static constexpr int kLatentDim = 64;
    static constexpr int kFrameRate = 25;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
