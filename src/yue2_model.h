#pragma once

// YuE2-3B backbone — the AR–NAR Mixture-of-Transformers forward on brotensor.
// Internal to src/ (yue2.cpp drives it; the tests include it to white-box the
// AR logits and the NAR velocity against the reference fixtures).
//
// Every layer holds two attention projections (q/k/v/o + q/k RMSNorm) and two
// SwiGLU MLPs with their own pre-norms: the AR path (causal, KV cached,
// lm_head over the 184704-way vocabulary) and the NAR path (flow matching over
// 64-dim latents). The NAR path never runs on AR rows: a chunk prefills its AR
// tokens once on the AR path, keeps every layer's K/V, and the NAR rows attend
// bidirectionally to [AR K/V | their own K/V] (modeling_yue2.nar_velocity's
// hybrid mask, as nar.CachedNAR computes it).
//
// Numerics follow the BF16 reference: BF16 weights and activations, FP32
// accumulation in every GEMM / norm / attention, RoPE tables rounded to BF16
// (the reference casts cos/sin to the activation dtype). HF rotate_half RoPE
// is carried as brotensor's interleaved-pair rotation by permuting each head's
// q/k projection rows (and q_norm/k_norm) at load.
//
// Decode runs B branches (1, or 2 for a CFG pair) as B-row GEMMs — one weight
// read for both rows — captured as one CUDA graph per session and replayed per
// token; each branch keeps its own cache and position.

#include "brosoundml/audio.h"
#include "brosoundml/yue2.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace brosoundml::yue2 {

struct Config {
    int hidden = 2048, layers = 28, n_q = 16, n_kv = 8, head_dim = 128, inter = 6144;
    int vocab = 184704, latent = 64, max_latent_frames = 24576, max_pos = 24576;
    float eps = 1e-6f, rope_theta = 1000000.0f, timestep_shift = 1.0f;
    static Config from_json(const std::string& path);
};

// Residual-stream hooks of one path (AR or NAR), applied after a layer's MLP
// add (the stream the next layer reads). Empty hooks cost nothing.
struct SteerDev {
    brotensor::Tensor delta;    // (1, H) or (rows, H), scale already folded in
    int  pos0 = 0;              // per-position: first position row 0 covers
    bool per_position = false;  // false: delta row 0 is added at every position
};
struct LayerHooks {
    std::vector<std::vector<SteerDev>> steer;   // [layer], may be empty
    std::vector<char> capture;                  // [layer] 1 = report the residual
    // (layer, pos0, rows, eval, data[rows * H]) — host FP32 copy of the
    // residual after `layer` at positions [pos0, pos0 + rows); eval = the NAR
    // velocity evaluation index, -1 on the AR path.
    std::function<void(int, int, int, int, const float*)> on_capture;

    bool steers(int l) const { return l < static_cast<int>(steer.size()) && !steer[l].empty(); }
    bool captures(int l) const {
        return on_capture && l < static_cast<int>(capture.size()) && capture[l];
    }
    bool any_capture() const;
    bool any_per_position() const;
};

using ArWeights = Yue2ArWeights;

struct ArSession;   // B-branch KV caches + decode graph + logits rows
struct NarChunk;    // one chunk's AR K/V + NAR scratch + ODE state
struct ModelWeights;

class Model {
public:
    Model();
    ~Model();
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;

    // `ar` picks the weights the decode step reads (yue2.h, Yue2ArWeights).
    // `keep_bf16_ar` = false (with Int8 / Fp8) drops the AR path's BF16
    // q/k/v/o/gate/up/down and lm_head once quantised: prefill then
    // dequantizes a matrix at a time into one scratch (Yue2LoadOptions).
    void load(const std::string& model_dir, brotensor::Device dev, ArWeights ar = ArWeights::Bf16,
              bool keep_bf16_ar = true);
    // Device bytes of the loaded weights (model + quantised copies + scratch).
    std::size_t weight_bytes() const;
    ArWeights ar_weights() const;
    bool loaded() const;
    const Config& config() const;
    brotensor::Device device() const;

    // ── AR path ──
    // A session of `branches` caches, each holding up to `capacity` tokens.
    // `hooks` (nullable, must outlive the session) steer / capture every
    // forward of the session; steering applies to every branch, capture
    // reports branch 0. A capture or a per-position steer runs decode eagerly
    // (no graph); a whole-sequence steer vector rides inside the graph.
    // `max_capacity` > capacity lets the caches grow on demand (a prefill or
    // step past `capacity` reallocates them, one layer at a time, in steps of
    // at least a quarter); 0 = fixed at `capacity`.
    std::unique_ptr<ArSession> ar_session(int branches, int capacity,
                                          const LayerHooks* hooks = nullptr, int max_capacity = 0) const;
    // Make every branch's cache hold at least `rows` (up to the session's
    // max capacity, else throws). Invalidates the session's decode graphs.
    void ar_reserve(ArSession& s, int rows) const;
    int ar_capacity(const ArSession& s) const;
    // Replace the session's hooks (a steer turned on, changed or off; the
    // same object with new deltas counts); the decode graphs are recaptured
    // at the next step. `hooks` must outlive the session.
    void ar_set_hooks(ArSession& s, const LayerHooks* hooks) const;
    // Upload a host steering delta (n * H floats, n = 1 or rows) scaled by
    // `scale` at the model's dtype.
    SteerDev make_steer(const std::vector<float>& delta, float scale, int pos0, bool per_position) const;
    // Prefill `ids` into branch b (empty cache) and compute that branch's
    // logits row for its last token over lm_head rows [lo, hi).
    void ar_prefill(ArSession& s, int b, const std::vector<int32_t>& ids, int lo, int hi) const;
    // Feed `token` to every branch and compute each branch's next logits row
    // over the range the session was last prefilled with.
    void ar_step(ArSession& s, int32_t token) const;
    // Feed R tokens (1..8, the same to every branch) at consecutive positions
    // in one forward — each weight read once for all B * R rows — and compute
    // every row's logits: ar_logits then holds B * R rows, branch-major (row
    // b * R + r is the next-token row after tokens[r]). Draft verification.
    // R > 1 needs CUDA or the CPU.
    void ar_step_rows(ArSession& s, const int32_t* tokens, int R) const;
    // Cut branch b's cache back to `len` tokens (rejected drafts); the rows
    // past it are dead and the next step overwrites them.
    void ar_truncate(ArSession& s, int b, int len) const;
    // Host copy of the logits rows: branches * (hi - lo) floats, branch-major.
    void ar_logits(ArSession& s, std::vector<float>& out) const;
    // Watch query head `head` of layer `layer` on branch 0 (before the
    // steps it should see; recaptures the decode graphs). Each step then
    // copies the layer's query rows (one device copy inside the step), and
    // ar_watched() reads them: for every row of the last step, the head's
    // softmax over the row's keys (causal, the full row), summed over each
    // of `spans` (key columns [a, b)). out = R * spans.size(), row-major.
    // The cost is one head's dot products on the host per row plus the new
    // key rows' download.
    void ar_watch(ArSession& s, int layer, int head) const;
    void ar_watched(ArSession& s, const std::vector<std::pair<int, int>>& spans, std::vector<float>& out) const;
    int ar_len(const ArSession& s, int b) const;

    // ── NAR path ──
    // A chunk over AR tokens `ar` (prompt prefix + codec + MUSIC_END) and
    // `frames` latent frames. With `reuse` the first `reuse_rows` AR rows are
    // copied from that session's branch 0 cache (whose tokens must be
    // ar[0, reuse_rows)), and only the rest is prefilled.
    struct NarOptions {
        const ArSession* reuse = nullptr;   // copy rows [0, reuse_rows) of its branch 0
        int reuse_rows = 0;
        const LayerHooks* ar_hooks = nullptr;  // steer the chunk's own AR prefill (no capture)
        // NAR-row hooks: positions are absolute latent frames (frame0 + row);
        // LATENT_START/END rows take only whole-sequence steers and are never
        // reported; capture_evals names the velocity evaluations (2 per ODE
        // step, counted from 0 per chunk) whose residuals are reported.
        const LayerHooks* hooks = nullptr;
        int frame0 = 0;
        std::vector<int> capture_evals;
        // In place: the chunk's AR K/V and NAR rows live in this session's
        // branch 0 cache instead of a copy (`reuse` and `reuse_rows` name the
        // same session and its length; the rest of `ar` and the N NAR rows are
        // written past it, reserving room). The session's length is kept, so
        // those rows are scratch to it; it must not step, grow or be freed
        // while the chunk lives. Saves the copy of the whole song's K/V.
        ArSession* in_place = nullptr;
    };
    std::unique_ptr<NarChunk> nar_chunk(const std::vector<int32_t>& ar, int frames,
                                        const NarOptions& opt) const;
    // One velocity v(x, t) for a host state x (frames * 64, frame-major) at
    // the raw (pre-sigmoid) timestep — the per-op parity seam.
    std::vector<float> nar_velocity(NarChunk& c, const std::vector<float>& x, double raw_t) const;
    // The midpoint ODE from `noise` (frames * 64, FP32, rounded to BF16 like
    // the reference's state) over `steps` steps; returns FP32 latents.
    // `on_step(done)` after each step; `cancel` polled twice per step and,
    // while each evaluation runs on the device, every few milliseconds until
    // it finishes (returns an empty vector when cancelled): a caller that
    // services real-time work from `cancel` keeps doing so through a long
    // evaluation instead of waiting it out in the next sync. On CUDA an evaluation replays
    // as a graph (one per chunk and state buffer, captured at its first use)
    // unless it is in capture_evals; BROSOUNDML_YUE2_NO_GRAPH=1 runs all eager.
    // `known` (known_frames * 64, may be null): inpainting of the first
    // known_frames frames: before every velocity evaluation at time t they are
    // set to the flow interpolant t * noise + (1 - t) * known (each op rounded
    // like the reference's BF16 state), and they come back as `known`.
    // `known_right` (known_right_frames * 64) does the same for the last
    // known_right_frames frames: a span re-solved between two kept sides.
    std::vector<float> nar_solve(NarChunk& c, const float* noise, int steps,
                                 const CancelCheck& cancel,
                                 const std::function<void(int)>& on_step,
                                 const float* known = nullptr, int known_frames = 0,
                                 const float* known_right = nullptr, int known_right_frames = 0) const;

private:
    std::unique_ptr<ModelWeights> w_;
};

}  // namespace brosoundml::yue2
