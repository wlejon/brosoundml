#pragma once

// brosoundml/decoder_lora.h — a trainable LoRA over the Kokoro decoder's AdaIN
// style→(γ,β) projections. Hosts the conditioning gate, the per-fc adapters,
// and the DecoderLora model (Adam, checkpoint I/O, cached forward/backward over
// the decoder back half). The conditioning vector is left generic — any small
// control signal (e.g. a style/affect coordinate) can drive the gate; the
// decoder back half is made trainable by src/kokoro_decoder_backward.{h,cpp}.

#include <brotensor/tensor.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace brosoundml {

struct Linear;
struct DecoderBackbone;
struct Generator;

// Condition -> per-rank gate MLP for the decoder LoRA:  g(v) = W2 * tanh(W1 * v).
//
// There are NO biases, so g(0) = 0 EXACTLY — a zero condition drives every
// adapter's gated bottleneck to zero, the LoRA delta vanishes, and the decoder
// reproduces its own (base) voice. W2 is zero-initialised so training starts
// from the base model and departs smoothly (the standard LoRA B=0 convention).
// The single gate vector (length = LoRA rank) is shared across every adapter in
// the decoder; per-layer A/B still let each layer respond differently to it.
//
//   v  : (in_dim, 1)        control/condition vector (in_dim configurable)
//   W1 : (hidden, in_dim)
//   W2 : (gate_dim, hidden) gate_dim = LoRA rank
//   g  : (gate_dim, 1)
struct ConditioningGate {
    brotensor::Tensor W1, W2;      // weights (no biases)
    brotensor::Tensor dW1, dW2;    // accumulated grads — zero_grad() between steps
    int in_dim = 3;
    int hidden = 0;
    int gate_dim = 0;

    // cached forward activations for backward()
    brotensor::Tensor pre1_;       // W1 v
    brotensor::Tensor act_;        // tanh(W1 v)

    // Build on `dev`: W1 xavier-initialised from `seed`, W2 zeroed.
    static ConditioningGate make(int gate_dim, int hidden, brotensor::Device dev,
                        std::uint64_t seed);

    // Zero (allocating if needed) the gradient accumulators.
    void zero_grad();

    // g <- W2 * tanh(W1 * v).  Caches pre1_ / act_ for backward().
    void forward(const brotensor::Tensor& v, brotensor::Tensor& g);

    // Given dG:(gate_dim,1), accumulate dW1, dW2. The condition input is the
    // control signal, not a trainable, so dV is computed-and-discarded. Reads
    // the cached act_ and the supplied v (same v as the matching forward()).
    void backward(const brotensor::Tensor& v, const brotensor::Tensor& dG);
};

// ─── One LoRA adapter on a single AdaIN fc ──────────────────────────────────
//
// The AdaIN style→(γ,β) projection `fc` (out = 2C) is the injection point. The
// fc input is the style vector (constant over the clip), so the adapter delta
// is a per-clip constant (γ,β) offset:  (γ,β) = fc(s) + scale·B·(g(v)⊙(A·s)).
// A/B are trainable (Adam state alongside); the base fc is frozen. `fc` points
// into the decoder/generator weights and must outlive the DecoderLora.
struct DecoderLoraAdapter {
    const Linear* fc = nullptr;        // frozen base projection (out = 2C)
    int C = 0;                         // affine half-dim (γ and β each C)
    brotensor::Tensor A, B;            // (rank, style_dim), (2C, rank)
    brotensor::Tensor mA, vA, mB, vB;  // Adam moments
    brotensor::Tensor dA, dB;          // grad accumulators (zeroed per step)
    brotensor::Tensor h, hg;           // bottleneck cache from the forward
};

// Pre-decoder context for one clip — the frozen intermediates the LoRA decode
// runs over (produced by the Kokoro pipeline up to decode_back_half, or
// synthesised in tests). Pointers are borrowed for the duration of the call.
struct DecoderLoraContext {
    const brotensor::Tensor* asr     = nullptr;  // (1, hidden*T)
    const brotensor::Tensor* F0_pred = nullptr;  // (1, 2T)
    const brotensor::Tensor* N_pred  = nullptr;  // (1, 2T)
    const brotensor::Tensor* ref_s   = nullptr;  // (1, 2*style_dim) — style = first style_dim
    const brotensor::Tensor* har     = nullptr;  // (1, har_ch*frames)
    int total  = 0;                              // T (asr frame count)
    int frames = 0;                              // har frame count
};

// ─── DecoderLora — the trainable, conditioned decoder adapter ───────────────
//
// Owns one LoRA adapter per AdaIN fc across the decoder back half (backbone
// encode/decode + every Generator AdaINResBlock1) and the shared conditioning
// gate, plus the Adam state. The forward runs the cached `_train` assemblies
// with the LoRA (γ,β); the backward threads the assembly affine grads back
// through each adapter's lora_backward and the gate. The backbone + generator
// are passed in (their fc weights frozen, addressed by the adapters' `fc`).
class DecoderLora {
public:
    // Build adapters over a fully-loaded backbone + generator. rank = LoRA
    // rank (= gate width); `scale` the LoRA scaling; `seed` drives A/B and gate
    // W1 init (B small, gate W2 zero → exact base at init and at cond=0 always).
    static DecoderLora make(const DecoderBackbone& bb, const Generator& gen,
                            int rank, float scale, brotensor::Device dev,
                            std::uint64_t seed);

    DecoderLora() = default;
    ~DecoderLora();
    DecoderLora(DecoderLora&&) noexcept;
    DecoderLora& operator=(DecoderLora&&) noexcept;
    DecoderLora(const DecoderLora&) = delete;
    DecoderLora& operator=(const DecoderLora&) = delete;

    // Cached forward: decode `ctx` under condition `cond` (in_dim,1) → `audio`.
    // cond = 0 reproduces the base decoder exactly (gate identity).
    void forward(const DecoderBackbone& bb, const Generator& gen,
                 const DecoderLoraContext& ctx, const brotensor::Tensor& cond,
                 brotensor::Tensor& audio);

    // Backward of the last forward(): accumulate all LoRA + gate grads from the
    // upstream audio grad. `cond` must match the forward.
    void backward(const DecoderBackbone& bb, const Generator& gen,
                  const brotensor::Tensor& cond, const brotensor::Tensor& dAudio);

    void zero_grad();
    void adam_step(float lr, float beta1 = 0.9f, float beta2 = 0.999f,
                   float eps = 1e-8f);

    void save(const std::string& path) const;
    void load(const std::string& path);

    int rank()  const { return rank_; }
    float scale() const { return scale_; }
    ConditioningGate&       gate()       { return gate_; }
    const ConditioningGate& gate() const { return gate_; }
    // Flat view over every adapter (backbone then generator), for tests.
    std::vector<DecoderLoraAdapter*> adapters();

private:
    int rank_ = 0;
    float scale_ = 1.0f;
    int style_dim_ = 0;
    brotensor::Device dev_ = brotensor::Device::CPU;
    int step_ = 0;
    ConditioningGate gate_;

    // Adapters mirror the assembly affine structures: 2 per backbone block,
    // 6 per Generator AdaINResBlock1 (adain1[0..2], adain2[0..2]).
    std::vector<std::array<DecoderLoraAdapter, 2>> bb_;     // encode + decode blocks
    std::vector<std::array<DecoderLoraAdapter, 6>> noise_;  // noise_res per upsample
    std::vector<std::array<DecoderLoraAdapter, 6>> res_;    // resblocks

    // Gate Adam moments (the gate's own optimiser state lives here, not on the
    // ConditioningGate, which only holds weights + grads).
    brotensor::Tensor gW1_m_, gW1_v_, gW2_m_, gW2_v_;

    // Per-forward state reused by the backward.
    brotensor::Tensor style_col_;   // (style_dim, 1)
    brotensor::Tensor g_;           // gate output (rank, 1)
    void* bb_cache_ = nullptr;      // detail::BackboneCache  (opaque to the header)
    void* gen_cache_ = nullptr;     // detail::GeneratorCache
};

}  // namespace brosoundml
