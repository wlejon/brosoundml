#include "brosoundml/modules.h"

#include <brotensor/ops.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

// Build a non-owning view into a contiguous slice of `parent`'s storage,
// starting at element `start` and spanning `count` elements, with the given
// (rows, cols) shape (where rows*cols == count). The view shares the parent's
// device and dtype; the parent must outlive every use of the returned view.
bt::Tensor element_view(const bt::Tensor& parent, std::size_t start,
                        int rows, int cols) {
    const std::size_t elem_bytes =
        static_cast<std::size_t>(bt::dtype_size_bytes(parent.dtype));
    void* base = static_cast<std::uint8_t*>(parent.data) + start * elem_bytes;
    return bt::Tensor::view(parent.device, base, rows, cols, parent.dtype);
}

}  // namespace

// ─── Linear ────────────────────────────────────────────────────────────────

void Linear::forward(const bt::Tensor& x, bt::Tensor& y) const {
    if (W.rows != out_features || W.cols != in_features) {
        fail("Linear::forward",
             "weight shape (" + std::to_string(W.rows) + "," +
             std::to_string(W.cols) + ") does not match (" +
             std::to_string(out_features) + "," +
             std::to_string(in_features) + ")");
    }
    if (b.rows != out_features || b.cols != 1) {
        fail("Linear::forward",
             "bias shape (" + std::to_string(b.rows) + "," +
             std::to_string(b.cols) + ") does not match (" +
             std::to_string(out_features) + ",1) — fill with zeros if a "
             "bias-less Linear is intended");
    }
    if (x.rows != in_features || x.cols != 1) {
        fail("Linear::forward",
             "input shape (" + std::to_string(x.rows) + "," +
             std::to_string(x.cols) + ") does not match (" +
             std::to_string(in_features) + ",1)");
    }
    bt::linear_forward(W, b, x, y);
}

void Linear::forward_batched(const bt::Tensor& X, bt::Tensor& Y) const {
    if (W.rows != out_features || W.cols != in_features) {
        fail("Linear::forward_batched",
             "weight shape mismatch with (out=" + std::to_string(out_features) +
             ", in=" + std::to_string(in_features) + ")");
    }
    if (b.rows != out_features || b.cols != 1) {
        fail("Linear::forward_batched", "bias shape mismatch — fill with zeros if no bias");
    }
    if (X.cols != in_features) {
        fail("Linear::forward_batched",
             "input cols=" + std::to_string(X.cols) + " != in_features=" +
             std::to_string(in_features));
    }
    bt::linear_forward_batched(W, b, X, Y);
}

// ─── LayerNorm ─────────────────────────────────────────────────────────────

void LayerNorm::forward(const bt::Tensor& X, bt::Tensor& Y) const {
    if (X.cols != features) {
        fail("LayerNorm::forward",
             "input cols=" + std::to_string(X.cols) + " != features=" +
             std::to_string(features));
    }
    if (gamma.rows != features || gamma.cols != 1 ||
        beta.rows  != features || beta.cols  != 1) {
        fail("LayerNorm::forward",
             "gamma/beta must be (" + std::to_string(features) + ",1)");
    }

    const int N = X.rows;
    const int D = X.cols;
    Y.resize(N, D, X.dtype);

    // Per-row LayerNorm: brotensor::layernorm_forward operates on a single
    // (D, 1) vector at a time. We loop, view each row as a (D, 1) buffer, and
    // discard the xhat / mean / rstd caches the op writes — those are only
    // useful for backward, and brosoundml is inference-only.
    bt::Tensor xhat;  // resized to (D, 1) on each call
    float mean_out = 0.0f, rstd_out = 0.0f;
    for (int r = 0; r < N; ++r) {
        bt::Tensor x_row = element_view(X, static_cast<std::size_t>(r) * D, D, 1);
        bt::Tensor y_row = element_view(Y, static_cast<std::size_t>(r) * D, D, 1);
        bt::layernorm_forward(x_row, gamma, beta, y_row, xhat,
                              mean_out, rstd_out, eps);
    }
}

// ─── Conv1d ────────────────────────────────────────────────────────────────

void Conv1d::forward(const bt::Tensor& X, int N, int L, bt::Tensor& Y) const {
    const bt::Tensor* bias_ptr = (b.rows == out_channels && b.cols == 1) ? &b
                                                                        : nullptr;
    bt::conv1d(X, W, bias_ptr, N, in_channels, L, out_channels, kernel_size,
               stride, padding, dilation, groups, Y);
}

// ─── LSTM cell (composed) ──────────────────────────────────────────────────

namespace {

// Allocate the per-step scratch tensors a single LSTM cell needs. Reused
// across timesteps so the time loop is allocation-free.
struct LSTMScratch {
    bt::Tensor gates_ih;   // (4H, 1)
    bt::Tensor gates_hh;   // (4H, 1)
    bt::Tensor i_t;        // (H, 1)
    bt::Tensor f_t;        // (H, 1)
    bt::Tensor g_t;        // (H, 1)
    bt::Tensor o_t;        // (H, 1)
    bt::Tensor tanh_c;     // (H, 1)
    bt::Tensor h_prev;     // (H, 1) — running hidden state
    bt::Tensor c_prev;     // (H, 1) — running cell state

    void init(int hidden, bt::Device device, bt::Dtype dtype) {
        const int H = hidden;
        gates_ih = bt::Tensor::zeros_on(device, 4 * H, 1, dtype);
        gates_hh = bt::Tensor::zeros_on(device, 4 * H, 1, dtype);
        i_t      = bt::Tensor::zeros_on(device, H, 1, dtype);
        f_t      = bt::Tensor::zeros_on(device, H, 1, dtype);
        g_t      = bt::Tensor::zeros_on(device, H, 1, dtype);
        o_t      = bt::Tensor::zeros_on(device, H, 1, dtype);
        tanh_c   = bt::Tensor::zeros_on(device, H, 1, dtype);
        h_prev   = bt::Tensor::zeros_on(device, H, 1, dtype);
        c_prev   = bt::Tensor::zeros_on(device, H, 1, dtype);
    }
};

// Run one direction of an LSTM over X:(L, input). On entry s.h_prev / s.c_prev
// hold h0 / c0 (typically zero). On exit they hold h_L / c_L. Y must already
// be sized to (L, hidden), and every row r of Y is overwritten with h_r.
void lstm_run(const LSTMCellWeights& w,
              const bt::Tensor& X, int L, int input_size, int hidden,
              bt::Tensor& Y, LSTMScratch& s,
              bool reverse) {
    const int H = hidden;
    for (int step = 0; step < L; ++step) {
        const int t = reverse ? (L - 1 - step) : step;
        bt::Tensor x_t = element_view(X, static_cast<std::size_t>(t) * input_size,
                                      input_size, 1);

        bt::linear_forward(w.W_ih, w.b_ih, x_t,        s.gates_ih);
        bt::linear_forward(w.W_hh, w.b_hh, s.h_prev,   s.gates_hh);
        bt::add_inplace(s.gates_ih, s.gates_hh);

        // gates_ih now holds the pre-activation gate stack; slice it into
        // four (H, 1) windows.
        bt::Tensor gi = element_view(s.gates_ih, 0 * H, H, 1);
        bt::Tensor gf = element_view(s.gates_ih, 1 * H, H, 1);
        bt::Tensor gg = element_view(s.gates_ih, 2 * H, H, 1);
        bt::Tensor go = element_view(s.gates_ih, 3 * H, H, 1);

        bt::sigmoid_forward(gi, s.i_t);
        bt::sigmoid_forward(gf, s.f_t);
        bt::tanh_forward   (gg, s.g_t);
        bt::sigmoid_forward(go, s.o_t);

        // c_new = f * c_prev + i * g — accumulate into f_t (reused as c_new).
        bt::mul_inplace(s.i_t, s.g_t);     // i_t <- i * g
        bt::mul_inplace(s.f_t, s.c_prev);  // f_t <- f * c_prev
        bt::add_inplace(s.f_t, s.i_t);     // f_t <- c_new
        // h = o * tanh(c_new).
        bt::tanh_forward(s.f_t, s.tanh_c);
        bt::mul_inplace(s.o_t, s.tanh_c);  // o_t <- h

        // Persist c_prev <- c_new, h_prev <- h, and copy h into Y[t,:].
        bt::copy_d2d(s.f_t, 0, s.c_prev, 0, H);
        bt::copy_d2d(s.o_t, 0, s.h_prev, 0, H);
        bt::copy_d2d(s.o_t, 0, Y,        t * H, H);
    }
}

void check_lstm_weights(const LSTMCellWeights& w, int input_size, int hidden,
                        const std::string& where) {
    const int four_h = 4 * hidden;
    auto expect = [&](const bt::Tensor& t, int r, int c, const char* name) {
        if (t.rows != r || t.cols != c) {
            fail(where, std::string(name) + " shape (" +
                 std::to_string(t.rows) + "," + std::to_string(t.cols) +
                 ") expected (" + std::to_string(r) + "," + std::to_string(c) + ")");
        }
    };
    expect(w.W_ih, four_h, input_size, "W_ih");
    expect(w.W_hh, four_h, hidden,     "W_hh");
    expect(w.b_ih, four_h, 1,          "b_ih");
    expect(w.b_hh, four_h, 1,          "b_hh");
}

}  // namespace

void LSTM::forward(const bt::Tensor& X, bt::Tensor& Y) const {
    if (X.cols != input_size) {
        fail("LSTM::forward",
             "input cols=" + std::to_string(X.cols) + " != input_size=" +
             std::to_string(input_size));
    }
    check_lstm_weights(cell, input_size, hidden_size, "LSTM::forward");
    const int L = X.rows;
    Y.resize(L, hidden_size, X.dtype);

    LSTMScratch s;
    s.init(hidden_size, X.device, X.dtype);
    lstm_run(cell, X, L, input_size, hidden_size, Y, s, /*reverse=*/false);
}

void BiLSTM::forward(const bt::Tensor& X, bt::Tensor& Y) const {
    if (X.cols != input_size) {
        fail("BiLSTM::forward",
             "input cols=" + std::to_string(X.cols) + " != input_size=" +
             std::to_string(input_size));
    }
    check_lstm_weights(forward_cell, input_size, hidden_size, "BiLSTM::forward(fwd)");
    check_lstm_weights(reverse_cell, input_size, hidden_size, "BiLSTM::forward(rev)");
    const int L = X.rows;
    const int H = hidden_size;
    Y.resize(L, 2 * H, X.dtype);

    bt::Tensor H_fwd = bt::Tensor::empty_on(X.device, L, H, X.dtype);
    bt::Tensor H_rev = bt::Tensor::empty_on(X.device, L, H, X.dtype);

    LSTMScratch s_fwd, s_rev;
    s_fwd.init(H, X.device, X.dtype);
    s_rev.init(H, X.device, X.dtype);

    lstm_run(forward_cell, X, L, input_size, H, H_fwd, s_fwd, /*reverse=*/false);
    lstm_run(reverse_cell, X, L, input_size, H, H_rev, s_rev, /*reverse=*/true);

    // Concat per-row: Y[t, :H] = H_fwd[t,:], Y[t, H:] = H_rev[t,:].
    for (int t = 0; t < L; ++t) {
        bt::copy_d2d(H_fwd, t * H, Y, t * 2 * H,     H);
        bt::copy_d2d(H_rev, t * H, Y, t * 2 * H + H, H);
    }
}

// ─── Multi-head attention (CPU FP32, with Q/K/V/O biases) ─────────────────
//
// The core attention kernel: pre-projected Q (Lq, D), K (Lk, D), V (Lk, D),
// scaled dot-product softmax per head, context concat, then a biased linear
// projection through Wo. Used by both MHA::forward (self-attention) and
// CrossAttention::forward (cross-attention) — they differ only in how Q vs.
// K/V are projected at the boundary.

void mha_attention_fp32(const bt::Tensor& Q,
                        const bt::Tensor& K,
                        const bt::Tensor& V,
                        int num_heads,
                        const bt::Tensor& Wo,
                        const bt::Tensor& bo,
                        const float* d_mask,
                        bt::Tensor& out) {
    const int D = Q.cols;
    if (V.rows != K.rows || V.cols != D || K.cols != D) {
        fail("mha_attn", "Q/K/V shape mismatch");
    }
    if (D % num_heads != 0) {
        fail("mha_attn", "embed_dim " + std::to_string(D) +
             " not divisible by num_heads " + std::to_string(num_heads));
    }

    // brotensor::flash_attention_forward does the per-head scaled-dot-product
    // softmax + V-mix in one device-dispatched op (CPU/CUDA/Metal). d_mask is
    // an optional length-Lk mask (1 valid / 0 invalid) on Q's device.
    bt::Tensor ctx;
    bt::flash_attention_forward(Q, K, V, d_mask, num_heads,
                                /*causal=*/false, ctx);
    bt::linear_forward_batched(Wo, bo, ctx, out);
}

namespace {

void check_attn_inputs(const std::string& where,
                       const bt::Tensor& X, int expected_dim) {
    if (X.dtype != bt::Dtype::FP32) {
        fail(where, "currently FP32 only");
    }
    if (X.cols != expected_dim) {
        fail(where, "input cols=" + std::to_string(X.cols) +
             " != expected dim=" + std::to_string(expected_dim));
    }
}

}  // namespace

void MHA::forward(const bt::Tensor& X,
                  const float* d_mask,
                  bt::Tensor& out) const {
    const std::string where = "MHA::forward";
    check_attn_inputs(where, X, embed_dim);

    bt::Tensor Q, K, V;
    bt::linear_forward_batched(Wq, bq, X, Q);
    bt::linear_forward_batched(Wk, bk, X, K);
    bt::linear_forward_batched(Wv, bv, X, V);

    mha_attention_fp32(Q, K, V, num_heads, Wo, bo, d_mask, out);
}

void CrossAttention::forward(const bt::Tensor& X,
                             const bt::Tensor& Ctx,
                             const float* d_mask,
                             bt::Tensor& out) const {
    const std::string where = "CrossAttention::forward";
    check_attn_inputs(where, X, embed_dim);
    check_attn_inputs(where, Ctx, ctx_dim);

    bt::Tensor Q, K, V;
    bt::linear_forward_batched(Wq, bq, X,   Q);
    bt::linear_forward_batched(Wk, bk, Ctx, K);
    bt::linear_forward_batched(Wv, bv, Ctx, V);

    mha_attention_fp32(Q, K, V, num_heads, Wo, bo, d_mask, out);
}

// ─── AdaIN1D ───────────────────────────────────────────────────────────────

void ada_in_1d(const bt::Tensor& X,
               const bt::Tensor& scale,
               const bt::Tensor& shift,
               int N, int C, int L,
               bt::Tensor& Y) {
    if (X.rows != N || X.cols != C * L) {
        fail("ada_in_1d",
             "X shape (" + std::to_string(X.rows) + "," +
             std::to_string(X.cols) + ") != (" + std::to_string(N) + "," +
             std::to_string(C * L) + ")");
    }
    if (X.dtype != bt::Dtype::FP32) {
        fail("ada_in_1d", "currently FP32 only");
    }
    if ((scale.rows * scale.cols) != C || (shift.rows * shift.cols) != C) {
        fail("ada_in_1d", "scale/shift must hold C=" + std::to_string(C) +
             " elements (any (C,1) or (1,C) layout)");
    }

    // Instance norm = GroupNorm with num_groups == C. Pass unit gamma / zero
    // beta so the affine step is left for the per-channel modulate below.
    std::vector<float> unit_host(static_cast<std::size_t>(C), 1.0f);
    bt::Tensor unit_gamma = bt::Tensor::from_host_on(X.device, unit_host.data(),
                                                    C, 1);
    bt::Tensor zero_beta  = bt::Tensor::zeros_on(X.device, C, 1, bt::Dtype::FP32);

    bt::Tensor X_norm;
    X_norm.resize(N, C * L, bt::Dtype::FP32);
    bt::group_norm_forward(X, unit_gamma, zero_beta,
                           N, C, /*H=*/1, /*W=*/L,
                           /*num_groups=*/C, /*eps=*/1e-5f, X_norm);

    // Per-channel affine Y[n,c,l] = X_norm[n,c,l] * scale[c] + shift[c],
    // composed via NCL -> (N*L, C) -> modulate -> NCL. nchw_to_sequence with
    // H=1 and W=L gives exactly the (N*L, C) layout modulate consumes.
    //
    // brotensor::modulate is AdaLN-style — it computes Y = X*(1+scale)+shift,
    // baking a "+1" into the scale to centre learned deltas around identity.
    // ada_in_1d's contract is the plain Y = X*scale + shift, so pre-subtract 1
    // from a clone of `scale` (kept on the input device) before the call.
    bt::Tensor scale_shifted = scale.clone();
    bt::add_scalar_inplace(scale_shifted, -1.0f);

    bt::Tensor X_seq;
    bt::nchw_to_sequence(X_norm, N, C, /*H=*/1, /*W=*/L, X_seq);

    bt::Tensor Y_seq;
    bt::modulate(X_seq, scale_shifted, shift, Y_seq);

    bt::sequence_to_nchw(Y_seq, N, C, /*H=*/1, /*W=*/L, Y);
}

}  // namespace brosoundml
