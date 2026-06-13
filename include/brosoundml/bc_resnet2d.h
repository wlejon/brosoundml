#pragma once

#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace brosoundml {

// ─── 2D BC-ResNet wake-word model ──────────────────────────────────────────
//
// A genuine 2D Broadcasted-Residual network (Kim et al., Interspeech 2021),
// the structurally-correct shape for microphone-robust keyword spotting and
// the replacement for the 1D-mel-as-channels BcResnet.
//
// Why 2D. The 1D model used the 40 mel bins as conv *channels* and convolved
// over time only, so its first layer learned a fixed weighting of absolute
// frequency bins — it baked the expected spectral envelope into the weights
// and collapsed when a real microphone's spectral tilt differed from the TTS
// training envelope. Here the input is a single-channel (freq × time) image
// and convolutions slide over BOTH axes with weights SHARED across frequency.
// The model therefore learns local time-frequency textures (onsets, formant
// motion, the /k/-/p/-/t/ bursts of "computer") that are invariant to spectral
// tilt and band shift — exactly the synthetic→real invariance we need.
//
// Broadcasted residual block (BC block). Splits the work into:
//   f2(x): frequency-wise depthwise 2D conv (kf×1) + SubSpectralNorm + ReLU.
//          Captures local spectral structure; cost stays low (depthwise).
//   f1(a): a = frequency-average-pool of f2(x) → (C, 1, T); temporal depthwise
//          conv (1×kt, CAUSAL in time) + BN. Captures temporal dynamics on the
//          frequency-collapsed signal, then is BROADCAST back across all
//          frequency rows and added to f2(x). A 1×1 pointwise conv mixes
//          channels. y = ReLU( residual(x) + pointwise(f2(x) + broadcast(f1)) ).
//
// SubSpectralNorm (SSN). BatchNorm applied independently within each of
// `ssn_subbands` contiguous frequency sub-bands — gives the network a mild
// per-band affine that improves frequency localisation without tying to
// absolute bins. The one module with a hand-rolled backward; everything else
// reuses brotensor's CUDA conv2d / batch_norm / relu / linear backward ops.
//
// Latency. Time convolutions are strictly causal (left-pad only, no
// lookahead), so a streaming forward emits a logit the moment a frame arrives
// — the design constraint is a ≤30 ms end-of-word→fire budget. Frequency
// convolutions are non-causal over the full 40 bins (free; no latency cost).
//
// Head: frequency-average-pool → temporal global-average-pool over the
// receptive-field window → Linear(C_last → 1) → logit.
//
// Topology is a compact BC-ResNet recipe sized for the always-on budget:
//
//   stem    conv2d(1 → c_stem, kf×kt, freq-stride sf)      + BN + ReLU
//   stage s (s = 0..n_stages-1):
//             transition BC block  (c_prev → c[s], freq-stride fstride[s],
//                                    temporal dilation tdil[s])
//             (n_blocks[s]-1) ×  normal BC block (c[s] → c[s], stride 1)
//   head    freq-avg-pool → temporal GAP → Linear(c_last → 1)

struct BcResnet2dConfig {
    int n_mels = 40;          // input frequency bins (image height H)

    // Stem conv2d: 1 → c_stem, kernel (stem_kf × stem_kt), frequency stride
    // stem_sf (time stride is always 1), causal in time.
    int c_stem  = 16;
    int stem_kf = 5;
    int stem_kt = 5;
    int stem_sf = 2;          // 40 → 20 frequency rows

    // BC-ResNet body. Fixed 3-stage recipe; widths/strides/dilations/depths
    // below. Encoded as small fixed arrays so the binary header can serialise
    // them verbatim. n_stages is locked at 3 for this recipe.
    static constexpr int n_stages = 3;
    int c[n_stages]        = {24, 32, 48};   // output channels per stage
    int fstride[n_stages]  = {2, 2, 1};      // freq stride of each transition block
    int tdil[n_stages]     = {1, 2, 4};      // temporal dilation of the f1 conv
    int n_blocks[n_stages] = {2, 2, 2};      // blocks per stage (incl. transition)

    // Depthwise kernel sizes: f2 is (dw_kf × 1) over frequency; f1 is
    // (1 × dw_kt) over time (causal, with the stage's temporal dilation).
    int dw_kf = 3;
    int dw_kt = 5;

    int   ssn_subbands = 4;   // SubSpectralNorm sub-bands (must divide the
                              // frequency height reaching each block)
    float bn_eps = 1e-5f;
};

// Per-stream streaming scratch for ONE independent wake stream. Unlike
// PhonemeNet the wake head pools over time, so a stream carries BOTH the
// per-conv causal time-ring caches AND the head's temporal global-average-pool
// ring (gap_window + its head/len). Weights stay in the (shared, read-only)
// net; a Session holds only this scratch, so one load-once net can drive N
// asynchronous streams without copying weights. Allocate with
// make_session(), advance with forward_streaming(state, ...), zero with
// reset(state). Move-only.
struct BcResnet2dSession {
    std::vector<brotensor::Tensor> conv_caches;   // one per cache-bearing conv
    std::vector<float>             gap_window;     // (gap_cap * c_last) ring
    int                            gap_len  = 0;
    int                            gap_head = 0;
};

class BcResnet2d {
public:
    BcResnet2d();
    ~BcResnet2d();
    BcResnet2d(BcResnet2d&&) noexcept;
    BcResnet2d& operator=(BcResnet2d&&) noexcept;
    BcResnet2d(const BcResnet2d&) = delete;
    BcResnet2d& operator=(const BcResnet2d&) = delete;

    // Build a zero-filled model on `device` (tests + trainer). Runtime callers
    // use load().
    static BcResnet2d make(const BcResnet2dConfig& cfg,
                           brotensor::Device device = brotensor::Device::CPU);

    // Read a 'BWK2' checkpoint; materialise every tensor on `device`. Runs
    // fuse_bn() if the file is flagged unfused. Throws on magic/shape mismatch.
    static BcResnet2d load(const std::string& path,
                           brotensor::Device device = brotensor::Device::CPU);

    // Serialise to 'BWK2'. fused=true is the runtime checkpoint; fused=false
    // keeps unfused BN/SSN params for continued training.
    void save(const std::string& path, bool fused = true) const;

    // Fold BN (and SSN's affine, which is fold-compatible) into the preceding
    // conv. Idempotent.
    void fuse_bn();

    // One-shot. feats: (n_mels, T) FP32 PCEN features on the model's device.
    // out_logit_per_frame: resized to (T, 1) — one logit per input frame, each
    // pooled over the causal receptive-field window ending at that frame.
    void forward(const brotensor::Tensor& feats,
                 brotensor::Tensor& out_logit_per_frame) const;

    // ── Multi-stream session API (load-once weights, N independent streams) ──
    //
    // Allocate a fresh per-stream state (conv caches + GAP ring) on the model's
    // device. N streams = one shared net + N of these. Const, so callable
    // through a shared_ptr<const BcResnet2d>.
    BcResnet2dSession make_session() const;

    // Streaming forward into a caller-owned state. new_feats: (n_mels, N).
    // out: (N, 1). Reads weights, writes ONLY into `state` (conv caches + GAP
    // ring) — so interleaved sessions on one const net never cross-talk. A
    // sequence of calls on one state matches forward() over the concatenation
    // (within FP rounding).
    void forward_streaming(BcResnet2dSession& state,
                           const brotensor::Tensor& new_feats,
                           brotensor::Tensor& out_logit_per_frame) const;

    // Zero a stream's caches + GAP ring (clean restart). Const.
    void reset(BcResnet2dSession& state) const;

    // Legacy single-stream streaming over an internally-owned state. Non-const;
    // prefer the session API for multi-stream use.
    void forward_streaming(const brotensor::Tensor& new_feats,
                           brotensor::Tensor& out_logit_per_frame);

    void reset_streaming_state();

    const BcResnet2dConfig& config() const;
    brotensor::Device       device() const;
    int receptive_field_frames() const;   // causal time lookback of the logit
    // Streaming GAP-ring capacity in frames. The per-frame streaming logit
    // pools the head feature over the last min(frames_seen, this) frames; only
    // once `frames_seen` reaches this value does the streaming pool match the
    // whole-clip GAP the model is trained on. Detectors should warm up for this
    // many frames before trusting a fire.
    int gap_window_frames() const;
    int param_count() const;
    bool fused() const;

    // ── Training surface (GPU) ──
    // Xavier-init all conv/linear weights, zero biases, SSN/BN affine identity.
    void xavier_init_weights(std::uint64_t seed);

    // One Adam step on a minibatch. feats_batch: (B, n_mels*T) NCL FP32;
    // labels: (B,1) FP32 0/1. Returns mean BCE-with-logits loss. All compute
    // on the model's device (CUDA in production).
    float train_step(const brotensor::Tensor& feats_batch,
                     const brotensor::Tensor& labels,
                     int B, int T, float lr, float pos_weight);

    struct EvalMetrics {
        float loss = 0.0f, accuracy = 0.0f, frr = 0.0f, fpr = 0.0f;
        int   n = 0;
    };
    EvalMetrics eval_step(const brotensor::Tensor& feats_batch,
                          const brotensor::Tensor& labels,
                          int B, int T, float pos_weight);

    // ── Test/debug seam for the hand-rolled backward ──
    // Enumerate the trainable parameters (name, element count) and read/write
    // individual elements, plus read the gradient produced by the most recent
    // train_step. Used by the finite-difference gradient check; lazily builds
    // the training state. Not a stable runtime API.
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
