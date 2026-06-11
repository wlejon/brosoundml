#include "brosoundml/modules.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#ifdef BROSOUNDML_HAS_CUDA
#include <brotensor/cuda_graph.h>
#endif

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

    // brotensor::layernorm_forward_inference_batched normalises every row
    // independently in one device-dispatched op — no per-row dispatch, no
    // backward caches (xhat / mean / rstd) to allocate on the right device.
    bt::layernorm_forward_inference_batched(X, gamma, beta, Y, eps);
}

// ─── Conv1d ────────────────────────────────────────────────────────────────

void Conv1d::forward(const bt::Tensor& X, int N, int L, bt::Tensor& Y) const {
    const bt::Tensor* bias_ptr = (b.rows == out_channels && b.cols == 1) ? &b
                                                                        : nullptr;
    bt::conv1d(X, W, bias_ptr, N, in_channels, L, out_channels, kernel_size,
               stride, padding, dilation, groups, Y);
}

// ─── LSTM cell (composed) ──────────────────────────────────────────────────
//
// The input-side projection for every timestep is hoisted into one batched
// GEMM (gates_all = X @ W_ih^T + b_ih), leaving only the recurrent body —
// W_hh GEMV + gate activations + state update — inside the time loop. On
// CUDA that body is captured as a CUDA graph once per cell and replayed per
// step (the Qwen Talker launch-overhead fix); a step is then a staging copy
// in, one graph launch, and a row copy out instead of ~15 kernel launches.

namespace {

// Per-direction step state. The tensors are the fixed buffers a captured
// graph replays over, so they must stay alive (and unmoved) for the life of
// the plan; gates holds the staged pre-activation row (input projection) on
// entry to a step and the summed gate stack after it.
struct LstmDirState {
    bt::Tensor gates;      // (4H, 1) — staged input-gate row, then the gate stack
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
        gates    = bt::Tensor::zeros_on(device, 4 * H, 1, dtype);
        gates_hh = bt::Tensor::zeros_on(device, 4 * H, 1, dtype);
        i_t      = bt::Tensor::zeros_on(device, H, 1, dtype);
        f_t      = bt::Tensor::zeros_on(device, H, 1, dtype);
        g_t      = bt::Tensor::zeros_on(device, H, 1, dtype);
        o_t      = bt::Tensor::zeros_on(device, H, 1, dtype);
        tanh_c   = bt::Tensor::zeros_on(device, H, 1, dtype);
        h_prev   = bt::Tensor::zeros_on(device, H, 1, dtype);
        c_prev   = bt::Tensor::zeros_on(device, H, 1, dtype);
    }

    void zero_state() {
        bt::scale_inplace(h_prev, 0.0f);
        bt::scale_inplace(c_prev, 0.0f);
    }
};

// One recurrent step over the staged buffers: s.gates holds the timestep's
// input projection (W_ih @ x_t + b_ih) on entry; on exit s.h_prev / s.c_prev
// hold h_t / c_t. Allocation-free after the first run, so it is capturable.
void lstm_step_body(const LSTMCellWeights& w, int H, LstmDirState& s) {
    bt::linear_forward(w.W_hh, w.b_hh, s.h_prev, s.gates_hh);
    bt::add_inplace(s.gates, s.gates_hh);

    // s.gates now holds the pre-activation gate stack; slice it into four
    // (H, 1) windows.
    bt::Tensor gi = element_view(s.gates, 0 * H, H, 1);
    bt::Tensor gf = element_view(s.gates, 1 * H, H, 1);
    bt::Tensor gg = element_view(s.gates, 2 * H, H, 1);
    bt::Tensor go = element_view(s.gates, 3 * H, H, 1);

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

    // Persist c_prev <- c_new, h_prev <- h.
    bt::copy_d2d(s.f_t, 0, s.c_prev, 0, H);
    bt::copy_d2d(s.o_t, 0, s.h_prev, 0, H);
}

// Input-side projection for the whole sequence in one batched GEMM:
// gates_all[t, :] = W_ih @ x_t + b_ih, shape (L, 4H).
void lstm_input_proj(const LSTMCellWeights& w, const bt::Tensor& X,
                     bt::Tensor& gates_all) {
    gates_all = bt::Tensor::empty_on(X.device, X.rows, w.W_ih.rows, X.dtype);
    bt::linear_forward_batched(w.W_ih, w.b_ih, X, gates_all);
}

// Run one direction over the pre-projected gates_all:(L, 4H), writing row t
// of Y at column offset y_col (0 for a plain/forward direction, H for the
// reverse half of a BiLSTM's (L, 2H) output). y_stride is Y's row width.
void lstm_run(const LSTMCellWeights& w,
              const bt::Tensor& gates_all, int L, int hidden,
              bt::Tensor& Y, int y_col, int y_stride, LstmDirState& s,
              bool reverse) {
    const int H = hidden;
    for (int step = 0; step < L; ++step) {
        const int t = reverse ? (L - 1 - step) : step;
        bt::copy_d2d(gates_all, t * 4 * H, s.gates, 0, 4 * H);
        lstm_step_body(w, H, s);
        bt::copy_d2d(s.h_prev, 0, Y, t * y_stride + y_col, H);
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

// The opaque per-cell plan declared in modules.h: persistent step state for
// up to two directions plus the captured CUDA graph that replays their step
// bodies. L-independent — the time loop stages each step's inputs into the
// fixed buffers, so one capture serves every utterance length.
struct LstmGraphPlan {
    LstmDirState fwd;
    LstmDirState rev;        // initialised only for a BiLSTM plan
    bool         captured = false;
#ifdef BROSOUNDML_HAS_CUDA
    bt::CudaGraph graph;
#endif
};

#ifdef BROSOUNDML_HAS_CUDA
namespace {

// Capture-or-replay driver for one direction (fwd) or two (fwd + rev in a
// single graph). gates_all_r may be null for unidirectional. Per step: stage
// the per-timestep gate rows, replay the captured step, copy the hidden rows
// out. The first call warms the step body up (sizing every scratch buffer so
// the capture run never allocates), then captures.
void lstm_graph_forward(const LSTMCellWeights& w_f, const LSTMCellWeights* w_r,
                        const bt::Tensor& gates_all_f,
                        const bt::Tensor* gates_all_r,
                        int L, int H,
                        bt::Tensor& Y, int y_stride,
                        LstmGraphPlan& p) {
    if (!p.captured) {
        p.fwd.init(H, gates_all_f.device, gates_all_f.dtype);
        if (w_r) p.rev.init(H, gates_all_f.device, gates_all_f.dtype);
        // Warm-up: run the body once so every output buffer is allocated;
        // the capture re-run reuses those exact tensors.
        lstm_step_body(w_f, H, p.fwd);
        if (w_r) lstm_step_body(*w_r, H, p.rev);
        bt::sync_all();
        {
            bt::CudaGraphCapture cap;
            lstm_step_body(w_f, H, p.fwd);
            if (w_r) lstm_step_body(*w_r, H, p.rev);
            p.graph = cap.finish();
        }
        p.captured = true;
    }

    p.fwd.zero_state();
    if (w_r) p.rev.zero_state();

    for (int step = 0; step < L; ++step) {
        bt::copy_d2d(gates_all_f, step * 4 * H, p.fwd.gates, 0, 4 * H);
        if (w_r) {
            const int t_r = L - 1 - step;
            bt::copy_d2d(*gates_all_r, t_r * 4 * H, p.rev.gates, 0, 4 * H);
        }
        p.graph.launch();
        bt::copy_d2d(p.fwd.h_prev, 0, Y, step * y_stride, H);
        if (w_r) {
            const int t_r = L - 1 - step;
            bt::copy_d2d(p.rev.h_prev, 0, Y, t_r * y_stride + H, H);
        }
    }
}

}  // namespace
#endif  // BROSOUNDML_HAS_CUDA

void LSTM::forward(const bt::Tensor& X, bt::Tensor& Y) const {
    if (X.cols != input_size) {
        fail("LSTM::forward",
             "input cols=" + std::to_string(X.cols) + " != input_size=" +
             std::to_string(input_size));
    }
    check_lstm_weights(cell, input_size, hidden_size, "LSTM::forward");
    const int L = X.rows;
    // Allocate Y on X's device — a default-constructed out-param tensor lives
    // on CPU, and Tensor::resize preserves device, so a CPU Y would land in
    // a mixed-device copy_d2d below and crash on CUDA.
    Y = bt::Tensor::empty_on(X.device, L, hidden_size, X.dtype);

    bt::Tensor gates_all;
    lstm_input_proj(cell, X, gates_all);

#ifdef BROSOUNDML_HAS_CUDA
    if (X.device == bt::Device::CUDA) {
        if (!plan) plan = std::make_shared<LstmGraphPlan>();
        lstm_graph_forward(cell, nullptr, gates_all, nullptr,
                           L, hidden_size, Y, hidden_size, *plan);
        return;
    }
#endif
    LstmDirState s;
    s.init(hidden_size, X.device, X.dtype);
    lstm_run(cell, gates_all, L, hidden_size, Y,
             /*y_col=*/0, /*y_stride=*/hidden_size, s, /*reverse=*/false);
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
    // Allocate Y on X's device (same reasoning as LSTM::forward above).
    Y = bt::Tensor::empty_on(X.device, L, 2 * H, X.dtype);

    bt::Tensor gates_all_f, gates_all_r;
    lstm_input_proj(forward_cell, X, gates_all_f);
    lstm_input_proj(reverse_cell, X, gates_all_r);

#ifdef BROSOUNDML_HAS_CUDA
    if (X.device == bt::Device::CUDA) {
        if (!plan) plan = std::make_shared<LstmGraphPlan>();
        lstm_graph_forward(forward_cell, &reverse_cell,
                           gates_all_f, &gates_all_r,
                           L, H, Y, /*y_stride=*/2 * H, *plan);
        return;
    }
#endif
    LstmDirState s_fwd, s_rev;
    s_fwd.init(H, X.device, X.dtype);
    s_rev.init(H, X.device, X.dtype);
    // Each direction writes its half of every (L, 2H) row directly — no
    // separate H_fwd/H_rev buffers, no concat pass.
    lstm_run(forward_cell, gates_all_f, L, H, Y,
             /*y_col=*/0, /*y_stride=*/2 * H, s_fwd, /*reverse=*/false);
    lstm_run(reverse_cell, gates_all_r, L, H, Y,
             /*y_col=*/H, /*y_stride=*/2 * H, s_rev, /*reverse=*/true);
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
    // softmax + V-mix in one device-dispatched op. The CPU backend is FP32;
    // the CUDA backend requires FP16 (or BF16) Q/K/V/O. Mirror that split:
    // pass FP32 straight through on CPU, cast to FP16 for the flash core on
    // any non-CPU device, then cast the (Lq, D) output back to FP32 before
    // the Wo + bo projection. d_mask is FP32 either way.
    bt::Tensor ctx = bt::Tensor::empty_on(Q.device, Q.rows, D, Q.dtype);
    if (Q.device == bt::Device::CPU) {
        bt::flash_attention_forward(Q, K, V, d_mask, num_heads,
                                    /*causal=*/false, ctx);
    } else {
        bt::Tensor Qh = bt::Tensor::empty_on(Q.device, Q.rows, D, bt::Dtype::FP16);
        bt::Tensor Kh = bt::Tensor::empty_on(K.device, K.rows, D, bt::Dtype::FP16);
        bt::Tensor Vh = bt::Tensor::empty_on(V.device, V.rows, D, bt::Dtype::FP16);
        bt::cast(Q, Qh, bt::Dtype::FP16);
        bt::cast(K, Kh, bt::Dtype::FP16);
        bt::cast(V, Vh, bt::Dtype::FP16);
        bt::Tensor ctx_h = bt::Tensor::empty_on(Q.device, Q.rows, D, bt::Dtype::FP16);
        bt::flash_attention_forward(Qh, Kh, Vh, d_mask, num_heads,
                                    /*causal=*/false, ctx_h);
        bt::cast(ctx_h, ctx, bt::Dtype::FP32);
    }
    // Pre-allocate `out` on Q.device — default-constructed Tensor lives on
    // CPU and the CUDA linear_forward_batched does not migrate it.
    if (out.device != Q.device || out.rows != Q.rows || out.cols != D ||
        out.dtype != Q.dtype) {
        out = bt::Tensor::empty_on(Q.device, Q.rows, D, Q.dtype);
    }
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

    // Allocate projection outputs on X's device — see ada_in_1d / LSTM for
    // the same default-CPU pitfall.
    bt::Tensor Q = bt::Tensor::empty_on(X.device, X.rows, embed_dim, X.dtype);
    bt::Tensor K = bt::Tensor::empty_on(X.device, X.rows, embed_dim, X.dtype);
    bt::Tensor V = bt::Tensor::empty_on(X.device, X.rows, embed_dim, X.dtype);
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

    bt::Tensor Q = bt::Tensor::empty_on(X.device,   X.rows,   embed_dim, X.dtype);
    bt::Tensor K = bt::Tensor::empty_on(Ctx.device, Ctx.rows, embed_dim, Ctx.dtype);
    bt::Tensor V = bt::Tensor::empty_on(Ctx.device, Ctx.rows, embed_dim, Ctx.dtype);
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

    // Instance norm = GroupNorm with num_groups == C, and group_norm_forward's
    // gamma/beta are already the per-channel affine — exactly ada_in_1d's
    // Y = X_norm*scale + shift contract. One fused pass; no transpose chain.
    // scale/shift may arrive as (1,C) row vectors; group_norm expects (C,1) —
    // same C contiguous elements, so view them at the right shape.
    bt::Tensor scale_c = bt::Tensor::view(scale.device, scale.data, C, 1, scale.dtype);
    bt::Tensor shift_c = bt::Tensor::view(shift.device, shift.data, C, 1, shift.dtype);
    if (Y.device != X.device) {
        Y = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    }
    bt::group_norm_forward(X, scale_c, shift_c,
                           N, C, /*H=*/1, /*W=*/L,
                           /*num_groups=*/C, /*eps=*/1e-5f, Y);
}

}  // namespace brosoundml
