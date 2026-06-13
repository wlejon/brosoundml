#pragma once

#include <brotensor/tensor.h>

#include "brosoundml/phoneme_data.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace brosoundml {

// ─── 2D BC-ResNet phoneme-posterior model ──────────────────────────────────
//
// The open-vocabulary keyword-spotting analogue of bc_resnet2d (the single-logit
// wake model). It shares bc_resnet2d's 2D Broadcasted-Residual trunk VERBATIM
// (stem conv2d → per-stage transition + normal BC blocks; each BC block = f2
// freq-depthwise + BN + ReLU, f1 freq-avg-pool → causal temporal depthwise + BN
// → broadcast-over-freq, pointwise mix, residual add, ReLU) but swaps the head
// and the loss:
//
//   • HEAD. bc_resnet2d pools the head feature over frequency AND over the whole
//     time window (global-average-pool) into ONE logit per clip. PhonemeNet
//     pools over frequency only, then applies a per-frame Linear(C_last → K) at
//     EVERY 10 ms frame, emitting a (T, K) tensor — one K-way phoneme-class
//     posterior per frame. There is NO temporal pool, NO GAP ring and NO
//     streaming warm-up: every frame is an independent classification over the
//     causal receptive field ending at that frame, so streaming is exactly the
//     trunk's per-conv causal time-ring caches with a pointwise-in-time head.
//
//   • LOSS. Framewise softmax cross-entropy (not BCE): the mean over frames of
//     softmax-CE(logits_t, label_t). Composed from brotensor's
//     softmax_xent_fused_batched (no new kernel). Two knobs keep silence from
//     dominating: a per-class weight vector (multiplies each frame's loss/grad)
//     and — folded into that same per-frame weight — silence handling (a class
//     weight of 0 on the silence class drops those frames entirely; a fractional
//     weight down-weights them). See train_step().
//
// SubSpectralNorm. bc_resnet2d's f2 normaliser is a plain BatchNorm2d in the
// shipped implementation (the SSN sub-band count is carried in the header but not
// applied in compute); PhonemeNet mirrors that exactly. Every layer therefore
// uses brotensor's conv2d / batch_norm / relu / linear forward+backward — there
// is no hand-rolled backward; the only bespoke piece is the framewise softmax-CE
// head, composed from brotensor ops and gradient-checked in the test.
//
// Checkpoint. A 'BPM1' file carrying the front-end framing params, the trunk
// hyperparameters, the embedded PhonemeClassMap (write_classmap/read_classmap),
// and the weights. K (the output dimension) comes from the class map.

struct PhonemeNetConfig {
    int n_mels = 40;                 // input frequency bins (image height H)

    // Front-end framing the stored/streamed PCM is meant to be framed with.
    // Carried in the checkpoint so the runtime spotter is self-describing;
    // defaults match PhonemeDatasetHeader.
    int sample_rate = 16000;
    int n_fft       = 512;
    int win_length  = 400;
    int hop_length  = 160;

    // Stem conv2d: 1 → c_stem, kernel (stem_kf × stem_kt), frequency stride
    // stem_sf (time stride 1), causal in time. ~2x the wake model's stem.
    int c_stem  = 32;
    int stem_kf = 5;
    int stem_kt = 5;
    int stem_sf = 2;                 // 40 → 20 frequency rows

    // BC-ResNet body. A 4-stage recipe (one more stage than the wake model) at
    // ~2x the channel width, lifting the causal receptive field to ~1 s and the
    // capacity to phoneme-discrimination scale. Encoded as fixed arrays so the
    // header serialises them verbatim.
    static constexpr int n_stages = 4;
    int c[n_stages]        = {32, 48, 64, 96};   // output channels per stage
    int fstride[n_stages]  = {2, 2, 1, 1};       // freq stride of each transition
    int tdil[n_stages]     = {1, 2, 4, 4};       // temporal dilation of the f1 conv
    int n_blocks[n_stages] = {2, 2, 2, 2};       // blocks per stage (incl. transition)

    // Depthwise kernels: f2 is (dw_kf × 1) over frequency; f1 is (1 × dw_kt) over
    // time (causal, with the stage's temporal dilation).
    int dw_kf = 3;
    int dw_kt = 5;

    int   ssn_subbands = 4;          // carried like BcResnet2dConfig (unused in compute)

    int   num_classes = 0;           // K — set from the class map at make()/load().
    float bn_eps = 1e-5f;
};

// Per-stream streaming scratch — the per-conv causal time-ring caches for ONE
// independent stream. The weights stay in the (shared, read-only) net; a
// Session holds caches only, so one load-once net can drive N asynchronous
// streams without copying weights. Allocate with make_session(), advance
// with forward_streaming(state, ...), zero with reset(state). Cheap to build;
// move-only (the caches are device tensors). Sessions advance independently —
// two interleaved streams over one net never cross-talk because each writes
// only its own Session.
struct PhonemeSession {
    std::vector<brotensor::Tensor> conv_caches;   // one per cache-bearing conv
};

class PhonemeNet {
public:
    PhonemeNet();
    ~PhonemeNet();
    PhonemeNet(PhonemeNet&&) noexcept;
    PhonemeNet& operator=(PhonemeNet&&) noexcept;
    PhonemeNet(const PhonemeNet&) = delete;
    PhonemeNet& operator=(const PhonemeNet&) = delete;

    // Build a zero-filled model. The class map fixes K (= class_map.num_classes)
    // and is carried for save(). cfg.num_classes is overwritten with K.
    static PhonemeNet make(const PhonemeNetConfig& cfg,
                           const PhonemeClassMap& class_map,
                           brotensor::Device device = brotensor::Device::CPU);

    // Read a 'BPM1' checkpoint; materialise every tensor on `device`. Runs
    // fuse_bn() effects if the file is flagged fused. Throws on magic/shape
    // mismatch.
    static PhonemeNet load(const std::string& path,
                           brotensor::Device device = brotensor::Device::CPU);

    // Serialise to 'BPM1'. fused=true is the runtime checkpoint; fused=false
    // keeps unfused BN params for continued training.
    void save(const std::string& path, bool fused = true) const;

    // Fold BN into the preceding conv. Idempotent.
    void fuse_bn();

    // One-shot. feats: (n_mels, T) FP32 on the model's device. out_logits:
    // resized to (T, K) — out_logits[t, k] is the logit for class k at frame t,
    // pooled over the causal receptive field ending at frame t.
    void forward(const brotensor::Tensor& feats,
                 brotensor::Tensor& out_logits) const;

    // ── Multi-stream session API (load-once weights, N independent streams) ──
    //
    // Allocate a fresh per-stream cache set on the model's device. The returned
    // state is independent of every other; N streams = one shared net + N of
    // these. Const (reads only layer shapes), so callable through a
    // shared_ptr<const PhonemeNet>.
    PhonemeSession make_session() const;

    // Streaming forward into a caller-owned state. new_feats: (n_mels, N).
    // out: (N, K). Reads weights, writes ONLY into `state` — so concurrent
    // sessions on one const net are isolated. A sequence of calls on one state
    // matches forward() over the concatenation (within FP rounding); the head
    // is pointwise in time, so no ring/warm-up beyond the trunk caches.
    void forward_streaming(PhonemeSession& state,
                           const brotensor::Tensor& new_feats,
                           brotensor::Tensor& out_logits) const;

    // Zero a stream's caches (clean restart on a silence boundary). Const.
    void reset(PhonemeSession& state) const;

    // Legacy single-stream streaming over an internally-owned state — equivalent
    // to make_session() once + forward_streaming(that, ...). Non-const (it
    // mutates the owned state); prefer the session API for multi-stream use.
    void forward_streaming(const brotensor::Tensor& new_feats,
                           brotensor::Tensor& out_logits);

    void reset_streaming_state();

    const PhonemeNetConfig& config() const;
    const PhonemeClassMap&  class_map() const;
    brotensor::Device       device() const;
    int receptive_field_frames() const;   // causal time lookback of a frame logit
    int param_count() const;
    bool fused() const;

    // ── Training surface ──
    // Xavier-init all conv/linear weights, zero biases, BN affine identity.
    void xavier_init_weights(std::uint64_t seed);

    // One Adam step on a minibatch of fixed-length clips.
    //   feats_batch: (B, n_mels*T) NCL FP32 — row b is clip b's (n_mels, T)
    //                features flattened (freq-major, then time).
    //   labels:      (B, T) FP32 integer class ids in [0, K) — labels[b, t] is
    //                the class of clip b's frame t. (Stored FP32, rounded to int;
    //                the data layer's int16 frame-label tracks cast directly.)
    //   class_weights: length-K FP32 weight per class (empty → uniform 1). Each
    //                frame's loss and gradient is multiplied by class_weights of
    //                its TRUE class. Silence handling folds in here: a weight of
    //                0 on the silence class drops those frames; a fractional
    //                weight down-weights them.
    // Returns the weighted-mean framewise softmax-CE loss = (Σ_f w_f·CE_f)/(Σ_f
    // w_f), where w_f is the frame's class weight (so uniform weights give the
    // plain mean framewise CE). lr==0 populates grads without moving params (the
    // finite-difference probe).
    float train_step(const brotensor::Tensor& feats_batch,
                     const brotensor::Tensor& labels,
                     int B, int T, float lr,
                     const std::vector<float>& class_weights);

    struct EvalMetrics {
        float loss = 0.0f;                      // weighted-mean framewise CE
        float frame_accuracy = 0.0f;            // argmax==label over all frames
        float nonsilence_frame_accuracy = 0.0f; // excludes silence-labeled frames
        int   n_frames = 0;
    };
    EvalMetrics eval_step(const brotensor::Tensor& feats_batch,
                          const brotensor::Tensor& labels,
                          int B, int T,
                          const std::vector<float>& class_weights);

    // ── Finite-difference gradient-check seam (mirrors bc_resnet2d's) ──
    std::vector<std::pair<std::string, int>> debug_params() const;
    float debug_get_param(const std::string& name, int idx) const;
    void  debug_set_param(const std::string& name, int idx, float value);
    float debug_grad(const std::string& name, int idx) const;

public:
    struct Impl;   // exposed for the .cpp binary-format helpers; not stable.
private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
