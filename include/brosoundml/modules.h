#pragma once

// nn-module layer for brosoundml.
//
// Tiny inference-only modules built on top of brotensor ops — the rung between
// "raw op calls" and "named model components". Each module owns its weights
// (so they can live wherever the underlying brotensor::Tensor lives — CPU,
// CUDA, Metal) and exposes a single `forward(...)` that delegates to brotensor.
//
// No backward, no caches, no autograd state. Kokoro is the first consumer; the
// same primitives will compose other neural audio models (HiFi-GAN, Vocos,
// EnCodec) without each model having to re-derive the same Conv1d/LSTM glue.
//
// Conventions (matching PyTorch where it doesn't cost us):
//   - Linear weight layout is (out, in). bias is (out, 1).
//   - Conv1d weight is (C_out, (C_in/groups)*kL) OIL — brotensor's layout.
//   - LSTM gates pack as i, f, g, o along the row axis, identical to
//     torch.nn.LSTM. Separate W_ih + b_ih and W_hh + b_hh so a checkpoint can
//     be loaded as-is from torch.save state dicts.
//   - LayerNorm operates per-row over (N, D) inputs — every row independently
//     normalised against gamma/beta of shape (D, 1).

#include <brotensor/tensor.h>

#include <vector>

namespace brosoundml {

// ─── Linear: y = W x + b ───────────────────────────────────────────────────
//
// Single-vector form takes x:(in,1) and writes y:(out,1). The batched form
// takes X:(B,in) and writes Y:(B,out) in a single brotensor::linear_forward_batched
// call. A bias-less Linear leaves `b` empty and skips the bias add.
struct Linear {
    int               in_features  = 0;
    int               out_features = 0;
    brotensor::Tensor W;   // (out, in)
    brotensor::Tensor b;   // (out, 1) — empty when bias is disabled

    // y:(out,1) resized. x:(in,1). Throws on a shape mismatch.
    void forward(const brotensor::Tensor& x, brotensor::Tensor& y) const;

    // Y:(B,out) resized. X:(B,in). Throws on a shape mismatch.
    void forward_batched(const brotensor::Tensor& X, brotensor::Tensor& Y) const;
};

// ─── LayerNorm over (N, D) ─────────────────────────────────────────────────
//
// Applies brotensor::layernorm_forward independently per row. gamma/beta are
// (D, 1) tensors; eps defaults to 1e-5f to match PyTorch.
struct LayerNorm {
    int               features = 0;
    float             eps      = 1e-5f;
    brotensor::Tensor gamma;   // (D, 1)
    brotensor::Tensor beta;    // (D, 1)

    // Y:(N,D) resized. X:(N,D). Throws on a shape mismatch.
    void forward(const brotensor::Tensor& X, brotensor::Tensor& Y) const;
};

// ─── Conv1d (NCL) ──────────────────────────────────────────────────────────
//
// Thin wrapper over brotensor::conv1d. The module owns the convolution
// hyperparameters so call sites only pass the activation, the batch count,
// and the input length.
struct Conv1d {
    int               in_channels  = 0;
    int               out_channels = 0;
    int               kernel_size  = 0;
    int               stride       = 1;
    int               padding      = 0;
    int               dilation     = 1;
    int               groups       = 1;
    brotensor::Tensor W;   // (out, (in/groups)*kL)
    brotensor::Tensor b;   // (out, 1) — empty when bias is disabled

    // Y:(N, out_channels*L_out) resized. X:(N, in_channels*L).
    void forward(const brotensor::Tensor& X, int N, int L,
                 brotensor::Tensor& Y) const;
};

// ─── LSTM (composed from matmul + sigmoid + tanh) ──────────────────────────
//
// brotensor has no native recurrent primitive (see CLAUDE.md), so brosoundml
// composes the LSTM cell from per-timestep matmul + sigmoid + tanh. Two cells
// strung end-to-end give a bidirectional layer.
//
// PyTorch nn.LSTM gate order is i, f, g, o — packed along the row axis of
// W_ih and W_hh; same order in the matching biases. Both b_ih and b_hh are
// kept separate so a checkpoint loads without merging. The cell formula:
//   gates = W_ih @ x + b_ih + W_hh @ h_prev + b_hh
//   i = sigmoid(gates[:hidden])         f = sigmoid(gates[hidden:2*hidden])
//   g = tanh   (gates[2*hidden:3*hidden]) o = sigmoid(gates[3*hidden:4*hidden])
//   c = f * c_prev + i * g
//   h = o * tanh(c)
struct LSTMCellWeights {
    brotensor::Tensor W_ih;   // (4*hidden, input)
    brotensor::Tensor W_hh;   // (4*hidden, hidden)
    brotensor::Tensor b_ih;   // (4*hidden, 1)
    brotensor::Tensor b_hh;   // (4*hidden, 1)
};

// Single-direction LSTM: read X:(L, input_size) left-to-right, write
// Y:(L, hidden_size). h0 / c0 default to zero. Inference-only — no per-step
// cache; per-step intermediates are stack-allocated brotensor tensors.
struct LSTM {
    int             input_size  = 0;
    int             hidden_size = 0;
    LSTMCellWeights cell;

    void forward(const brotensor::Tensor& X, brotensor::Tensor& Y) const;
};

// Bidirectional LSTM: a forward cell over X plus a reverse cell over the
// reversed X; per timestep the two hidden vectors are concatenated, so
// Y:(L, 2*hidden_size). Both cells share the same input_size / hidden_size.
struct BiLSTM {
    int             input_size  = 0;
    int             hidden_size = 0;
    LSTMCellWeights forward_cell;
    LSTMCellWeights reverse_cell;

    void forward(const brotensor::Tensor& X, brotensor::Tensor& Y) const;
};

// ─── Multi-head self-attention (with Q/K/V/O biases) ──────────────────────
//
// PyTorch / Hugging Face MultiHeadAttention with separate weights + biases on
// every projection. The plBERT branch of Kokoro needs this layout; so does
// every transformer-encoder block in Whisper.
//
// brotensor's FP16 path (flash_attention_qkvo_forward) accepts separate
// Q/K/V/O weights and biases, but no FP32 attention op does — so the FP32
// CPU implementation projects Q/K/V via linear_forward_batched (which has
// bias) and then runs an open-coded scalar softmax + context kernel. This
// is the same loop kokoro_modules.cpp previously open-coded inline; lifting
// it into a module so Whisper can reuse it.
//
// Dispatch (today): FP32 CPU only. The FP16 / GPU path will arrive when
// Whisper-on-GPU performance becomes the bottleneck; until then a runtime
// check rejects non-FP32 / non-CPU inputs rather than silently falling back.
struct MHA {
    int               num_heads = 0;
    int               embed_dim = 0;   // D — must be divisible by num_heads
    brotensor::Tensor Wq;              // (D, D)
    brotensor::Tensor bq;              // (D, 1) — empty when bias disabled
    brotensor::Tensor Wk;              // (D, D)
    brotensor::Tensor bk;              // (D, 1)
    brotensor::Tensor Wv;              // (D, D)
    brotensor::Tensor bv;              // (D, 1)
    brotensor::Tensor Wo;              // (D, D)
    brotensor::Tensor bo;              // (D, 1)

    // Self-attention. X: (L, embed_dim). `d_mask` is an optional length-L
    // host pointer (1 valid / 0 invalid); pass nullptr when every key is
    // valid. out: (L, embed_dim), resized.
    void forward(const brotensor::Tensor& X,
                 const float* d_mask,
                 brotensor::Tensor& out) const;
};

// ─── Multi-head cross-attention (with Q/K/V/O biases) ─────────────────────
//
// Like MHA, but K and V are projected from a separate context tensor — so
// Wk and Wv are rectangular (D, ctx_dim). This is the layout the Whisper
// decoder's cross-attention layer needs, and the same shape SD1.5's
// cross-attention uses (different ctx_dim for the encoder hidden state).
//
// Same dispatch rules as MHA — FP32 CPU only for now.
struct CrossAttention {
    int               num_heads = 0;
    int               embed_dim = 0;   // D (query/output dim)
    int               ctx_dim   = 0;   // context (K/V) input dim
    brotensor::Tensor Wq;              // (D, D)
    brotensor::Tensor bq;              // (D, 1)
    brotensor::Tensor Wk;              // (D, ctx_dim)
    brotensor::Tensor bk;              // (D, 1)
    brotensor::Tensor Wv;              // (D, ctx_dim)
    brotensor::Tensor bv;              // (D, 1)
    brotensor::Tensor Wo;              // (D, D)
    brotensor::Tensor bo;              // (D, 1)

    // X:   (Lq, embed_dim) — query source.
    // Ctx: (Lk, ctx_dim)   — key/value source.
    // d_mask: optional length-Lk host pointer (1 valid / 0 invalid).
    // out: (Lq, embed_dim), resized.
    void forward(const brotensor::Tensor& X,
                 const brotensor::Tensor& Ctx,
                 const float* d_mask,
                 brotensor::Tensor& out) const;
};

// Low-level multi-head attention kernel: takes pre-projected Q (Lq, D),
// K (Lk, D), V (Lk, D); runs per-head scaled-dot-product softmax and a final
// Wo + bo projection. Both MHA::forward and CrossAttention::forward delegate
// to this — and call sites that already hold the projected Q/K/V can skip
// the module wrappers and call this directly (PLBert's ALBERT layer does).
// CPU FP32 only for now.
void mha_attention_fp32(const brotensor::Tensor& Q,
                        const brotensor::Tensor& K,
                        const brotensor::Tensor& V,
                        int num_heads,
                        const brotensor::Tensor& Wo,
                        const brotensor::Tensor& bo,
                        const float* d_mask,
                        brotensor::Tensor& out);

// ─── AdaIN1D (instance norm + per-channel affine) ──────────────────────────
//
// The "AdaIN" affine step common to StyleTTS2 / iSTFTNet decoders: normalise
// X per-channel (instance norm = GroupNorm with num_groups == C), then apply
// a learned per-channel scale + shift conditioned on the style embedding.
//
// X, Y: (N, C*L) NCL.  scale, shift: (C, 1) or (1, C) — the per-channel affine
// the caller has already projected from the style vector. Y is resized.
void ada_in_1d(const brotensor::Tensor& X,
               const brotensor::Tensor& scale,
               const brotensor::Tensor& shift,
               int N, int C, int L,
               brotensor::Tensor& Y);

}  // namespace brosoundml
