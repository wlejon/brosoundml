#pragma once

#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── BC-ResNet wake-word model ─────────────────────────────────────────────
//
// Small causal temporal CNN that classifies a streaming log-mel input as
// "wake-word present" or not. Takes (n_mels, T) FP32 log-mel frames, emits a
// single binary logit per inference call.
//
// Topology (all defaults — caller may sweep the channel widths in chunk 6):
//
//   stem      Conv1d(n_mels → c0,           k=3, causal-pad)        + BN + ReLU
//   block1    DW   Conv1d(c0 → c0, k, groups=c0, dil=1, causal-pad) + BN + ReLU
//             PW   Conv1d(c0 → c1, k=1)                             + BN
//             residual: identity (c0 == c1)                         → add → ReLU
//   block2    DW   Conv1d(c1, k, groups=c1, dil=2, causal-pad)      + BN + ReLU
//             PW   Conv1d(c1 → c2, k=1)                             + BN
//             residual: Conv1d(c1 → c2, k=1) on the input            → add → ReLU
//   block3    DW   Conv1d(c2, k, groups=c2, dil=4, causal-pad)      + BN + ReLU
//             PW   Conv1d(c2 → c3, k=1)                             + BN
//             residual: Conv1d(c2 → c3, k=1) on the input            → add → ReLU
//   block4    DW   Conv1d(c3, k, groups=c3, dil=8, causal-pad)      + BN + ReLU
//             PW   Conv1d(c3 → c4, k=1)                             + BN
//             residual: Conv1d(c3 → c4, k=1) on the input            → add → ReLU
//   head      global-avg-pool over T → Linear(c4 → 1) → logit
//
// All Conv1d on the time axis are causal: left-padded by dilation*(k-1) so
// the output at frame t depends only on inputs at frames ≤ t. This is what
// lets forward_streaming() emit one new logit per new mel frame with zero
// lookahead, matching forward() bit-for-bit (modulo accumulation order).
//
// Receptive field with the default dilations is
//   stem(k=3) + sum_block (k-1)*dil_b = 2 + (k-1)*(1+2+4+8)
// frames — with k=9 that's 2 + 8*15 = 122 frames ≈ 1.22 s at 10 ms hop,
// comfortably covering the 1.0 s wake-word context window.
//
// Batch normalisation: not yet a brosoundml module. At training time (chunk 6)
// BN is composed inline as
//     y = gamma * (x - mean) / sqrt(var + eps) + beta
// with running mean/var tracked separately. At inference (this chunk) BN is
// folded into the preceding conv:
//     W'  = W * gamma / sqrt(var + eps)
//     b'  = (b - mean) * gamma / sqrt(var + eps) + beta
// after which BN is gone and every conv simply uses its fused weight/bias.
// load() runs fuse_bn() automatically if the on-disk format flags the weights
// as unfused.

struct BcResnetConfig {
    // Front-end
    int  n_mels = 40;

    // Channel widths. c0 is the stem output, c1..c4 are the post-pointwise
    // widths of blocks 1..4. The default (32, 32, 48, 56, 64) gives
    // ~36K parameters with k=9 — comfortably under the 50K cap.
    int  c0     = 32;
    int  c1     = 32;
    int  c2     = 48;
    int  c3     = 56;
    int  c4     = 64;

    // Per-block depthwise kernel size. The stem always uses k=3.
    int  kernel_size = 9;

    // Per-block dilation. The stem is always dilation 1.
    int  dil1   = 1;
    int  dil2   = 2;
    int  dil3   = 4;
    int  dil4   = 8;

    // BN eps used during the fold step. Matches PyTorch default.
    float bn_eps = 1e-5f;
};

class BcResnet {
public:
    BcResnet();
    ~BcResnet();
    BcResnet(BcResnet&&) noexcept;
    BcResnet& operator=(BcResnet&&) noexcept;
    BcResnet(const BcResnet&) = delete;
    BcResnet& operator=(const BcResnet&) = delete;

    // Build a model with the given config and zero-filled tensors on `device`.
    // Used by tests + chunk-6 training; runtime callers go through load().
    static BcResnet make(const BcResnetConfig& cfg,
                         brotensor::Device device = brotensor::Device::CPU);

    // Read a brosoundml-binary weight file. The header carries the config;
    // every tensor is materialised on `device`. If the file is flagged
    // unfused, fuse_bn() runs before returning. Throws std::runtime_error on
    // any I/O / shape / magic mismatch.
    static BcResnet load(const std::string& path,
                         brotensor::Device device = brotensor::Device::CPU);

    // Serialise to the brosoundml-binary format. `fused=true` is the runtime
    // checkpoint; `fused=false` is what chunk-6 training emits before the
    // final fold (so trainers can keep the unfused parameters around for
    // continued training).
    void save(const std::string& path, bool fused = true) const;

    // Fold every BN into its preceding Conv1d on the host. Idempotent — a
    // second call is a no-op. After this, the model carries no BN state and
    // forward() uses the rewritten W'/b' directly.
    void fuse_bn();

    // One-shot. log_mel: (n_mels, T) FP32 on the model's device.
    // out_logit: resized to (1, 1) on the same device.
    void forward(const brotensor::Tensor& log_mel,
                 brotensor::Tensor& out_logit) const;

    // Frame-by-frame. new_frames: (n_mels, N) FP32 on the model's device.
    // out_logit_per_frame: resized to (N, 1). Maintains an internal
    // per-conv ring cache so a sequence of forward_streaming() calls is
    // equivalent (within FP rounding) to one forward() over the concatenated
    // input — see test_bc_resnet for the locked equivalence.
    void forward_streaming(const brotensor::Tensor& new_frames,
                           brotensor::Tensor& out_logit_per_frame);

    // Clear every conv's ring cache. Returns the model to a clean-context
    // state — the next forward_streaming() call sees zero history.
    void reset_streaming_state();

    const BcResnetConfig& config() const;
    brotensor::Device     device() const;

    // Total input receptive field of the pooled logit, in frames. Each
    // post-block-4 feature looks back conv_rf input frames; the global
    // average pool itself is sized to conv_rf so the final logit's input
    // dependence is (conv_rf + conv_rf - 1) frames.
    int receptive_field_frames() const;

    // Trainable parameter count. Counts every weight + bias element in the
    // convolutions, the head linear, and (when not yet fused) the BN gammas /
    // betas. Does not count running mean/var.
    int param_count() const;

    // True after fuse_bn() has been called (or after load() consumed a fused
    // checkpoint).
    bool fused() const;

public:
    // Exposed so the .cpp's free-function binary-format helpers can walk the
    // model's tensors by name. Not part of the stable surface — treat as
    // implementation detail.
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
