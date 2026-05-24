// brosoundml module-layer tests. Each module is exercised with hand-rolled
// synthetic weights so the expected output is computable by hand — no
// reference activations from an external framework involved.
#include "brosoundml/modules.h"

#include <brotensor/tensor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace bt = brotensor;

static int failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

static bool approx(float a, float b, float tol = 1e-5f) {
    return std::abs(a - b) <= tol;
}

template <typename Fn>
static bool throws_runtime_error(Fn&& fn) {
    try { fn(); }
    catch (const std::runtime_error&) { return true; }
    catch (...) { return false; }
    return false;
}

static bt::Tensor make_tensor(const std::vector<float>& data, int rows, int cols) {
    return bt::Tensor::from_host_on(bt::Device::CPU, data.data(), rows, cols);
}

// ─── Linear ────────────────────────────────────────────────────────────────

static void test_linear() {
    brosoundml::Linear lin;
    lin.in_features  = 3;
    lin.out_features = 2;
    // W = [[1, 2, 3], [4, 5, 6]]; b = [10, 20].
    lin.W = make_tensor({1, 2, 3, 4, 5, 6}, 2, 3);
    lin.b = make_tensor({10, 20}, 2, 1);

    // x = [1, 1, 1]^T → y = [1+2+3+10, 4+5+6+20] = [16, 35].
    bt::Tensor x = make_tensor({1, 1, 1}, 3, 1);
    bt::Tensor y;
    lin.forward(x, y);
    CHECK(y.rows == 2 && y.cols == 1, "Linear::forward output shape");
    CHECK(approx(y.host_f32()[0], 16.0f), "Linear::forward y[0]");
    CHECK(approx(y.host_f32()[1], 35.0f), "Linear::forward y[1]");

    // Batched: X = [[1,1,1], [2,2,2]] → Y = [[16,35], [22,50]].
    bt::Tensor X = make_tensor({1, 1, 1, 2, 2, 2}, 2, 3);
    bt::Tensor Y;
    lin.forward_batched(X, Y);
    CHECK(Y.rows == 2 && Y.cols == 2, "Linear::forward_batched output shape");
    CHECK(approx(Y.host_f32()[0], 16.0f) && approx(Y.host_f32()[1], 35.0f),
          "Linear::forward_batched row 0");
    CHECK(approx(Y.host_f32()[2], 22.0f) && approx(Y.host_f32()[3], 50.0f),
          "Linear::forward_batched row 1");

    CHECK(throws_runtime_error([&] {
              bt::Tensor bad_x = make_tensor({1, 1}, 2, 1);
              bt::Tensor out;
              lin.forward(bad_x, out);
          }),
          "Linear::forward rejects wrong input shape");
}

// ─── LayerNorm ─────────────────────────────────────────────────────────────

static void test_layernorm() {
    brosoundml::LayerNorm ln;
    ln.features = 4;
    ln.gamma = make_tensor({1, 1, 1, 1}, 4, 1);
    ln.beta  = make_tensor({0, 0, 0, 0}, 4, 1);

    // Two rows, each with a deterministic mean / variance. After LN with
    // gamma=1 / beta=0, each row's mean should be ~0 and its variance ~1.
    bt::Tensor X = make_tensor({1, 2, 3, 4,
                                10, 12, 14, 16}, 2, 4);
    bt::Tensor Y;
    ln.forward(X, Y);
    CHECK(Y.rows == 2 && Y.cols == 4, "LayerNorm output shape");

    for (int r = 0; r < 2; ++r) {
        float sum = 0, sumsq = 0;
        for (int c = 0; c < 4; ++c) {
            const float v = Y.host_f32()[r * 4 + c];
            sum += v;
            sumsq += v * v;
        }
        const float mean = sum / 4.0f;
        const float var  = sumsq / 4.0f - mean * mean;
        CHECK(approx(mean, 0.0f, 1e-4f),
              r == 0 ? "LayerNorm row 0 mean ~ 0" : "LayerNorm row 1 mean ~ 0");
        CHECK(approx(var, 1.0f, 1e-3f),
              r == 0 ? "LayerNorm row 0 variance ~ 1" : "LayerNorm row 1 variance ~ 1");
    }

    // Non-trivial gamma / beta: output should be gamma * normed + beta.
    ln.gamma = make_tensor({2, 2, 2, 2}, 4, 1);
    ln.beta  = make_tensor({5, 5, 5, 5}, 4, 1);
    bt::Tensor Y2;
    ln.forward(X, Y2);
    // Each row mean → 5, each row "variance" (after scaling) → 4.
    float sum = 0;
    for (int c = 0; c < 4; ++c) sum += Y2.host_f32()[c];
    CHECK(approx(sum / 4.0f, 5.0f, 1e-4f),
          "LayerNorm row 0 mean ~ beta=5 with non-unit gamma");
}

// ─── Conv1d ────────────────────────────────────────────────────────────────

static void test_conv1d() {
    // 1 input channel, 1 output channel, kernel [1, 1, 1]: a moving sum of 3.
    brosoundml::Conv1d conv;
    conv.in_channels  = 1;
    conv.out_channels = 1;
    conv.kernel_size  = 3;
    conv.stride       = 1;
    conv.padding      = 0;
    conv.dilation     = 1;
    conv.groups       = 1;
    conv.W = make_tensor({1, 1, 1}, 1, 3);
    conv.b = make_tensor({0}, 1, 1);

    bt::Tensor X = make_tensor({1, 2, 3, 4, 5}, 1, 5);  // (N=1, C*L=1*5)
    bt::Tensor Y;
    conv.forward(X, /*N=*/1, /*L=*/5, Y);
    // L_out = 5 - 3 + 1 = 3. Output: [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12].
    CHECK(Y.rows == 1 && Y.cols == 3, "Conv1d output shape");
    CHECK(approx(Y.host_f32()[0], 6.0f), "Conv1d y[0] = 6");
    CHECK(approx(Y.host_f32()[1], 9.0f), "Conv1d y[1] = 9");
    CHECK(approx(Y.host_f32()[2], 12.0f), "Conv1d y[2] = 12");

    // Bias non-zero shifts every output.
    conv.b = make_tensor({100}, 1, 1);
    bt::Tensor Y2;
    conv.forward(X, 1, 5, Y2);
    CHECK(approx(Y2.host_f32()[0], 106.0f), "Conv1d adds bias to y[0]");
}

// ─── LSTM (single direction) ───────────────────────────────────────────────

// Reference cell computation for a single timestep, so the test owns its
// ground truth without leaning on brotensor for it.
static void lstm_step_ref(int input_size, int hidden,
                          const std::vector<float>& W_ih,
                          const std::vector<float>& W_hh,
                          const std::vector<float>& b_ih,
                          const std::vector<float>& b_hh,
                          const std::vector<float>& x,
                          const std::vector<float>& h_prev,
                          const std::vector<float>& c_prev,
                          std::vector<float>& h_out,
                          std::vector<float>& c_out) {
    const int four_h = 4 * hidden;
    std::vector<float> gates(four_h, 0.0f);
    for (int r = 0; r < four_h; ++r) {
        float s = b_ih[r] + b_hh[r];
        for (int k = 0; k < input_size; ++k) s += W_ih[r * input_size + k] * x[k];
        for (int k = 0; k < hidden;     ++k) s += W_hh[r * hidden     + k] * h_prev[k];
        gates[r] = s;
    }
    auto sig  = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    auto tanh = [](float v) { return std::tanh(v); };
    h_out.assign(hidden, 0.0f);
    c_out.assign(hidden, 0.0f);
    for (int j = 0; j < hidden; ++j) {
        const float i = sig (gates[0 * hidden + j]);
        const float f = sig (gates[1 * hidden + j]);
        const float g = tanh(gates[2 * hidden + j]);
        const float o = sig (gates[3 * hidden + j]);
        c_out[j] = f * c_prev[j] + i * g;
        h_out[j] = o * std::tanh(c_out[j]);
    }
}

static void test_lstm() {
    const int input_size = 2;
    const int hidden     = 3;
    const int L          = 4;

    // Deterministic-but-non-uniform weights: small integers, FP32.
    std::vector<float> W_ih, W_hh, b_ih, b_hh;
    for (int r = 0; r < 4 * hidden; ++r) {
        for (int k = 0; k < input_size; ++k)
            W_ih.push_back(0.1f * static_cast<float>((r + k) % 7) - 0.3f);
    }
    for (int r = 0; r < 4 * hidden; ++r) {
        for (int k = 0; k < hidden; ++k)
            W_hh.push_back(0.05f * static_cast<float>((r * 3 + k) % 11) - 0.25f);
    }
    for (int r = 0; r < 4 * hidden; ++r) b_ih.push_back(0.01f * r);
    for (int r = 0; r < 4 * hidden; ++r) b_hh.push_back(-0.02f * r);

    brosoundml::LSTM lstm;
    lstm.input_size  = input_size;
    lstm.hidden_size = hidden;
    lstm.cell.W_ih = make_tensor(W_ih, 4 * hidden, input_size);
    lstm.cell.W_hh = make_tensor(W_hh, 4 * hidden, hidden);
    lstm.cell.b_ih = make_tensor(b_ih, 4 * hidden, 1);
    lstm.cell.b_hh = make_tensor(b_hh, 4 * hidden, 1);

    std::vector<float> X_data;
    for (int t = 0; t < L; ++t) {
        X_data.push_back(0.1f * static_cast<float>(t + 1));
        X_data.push_back(-0.2f * static_cast<float>(t + 1));
    }
    bt::Tensor X = make_tensor(X_data, L, input_size);
    bt::Tensor Y;
    lstm.forward(X, Y);
    CHECK(Y.rows == L && Y.cols == hidden, "LSTM output shape");

    std::vector<float> h_prev(hidden, 0.0f), c_prev(hidden, 0.0f);
    std::vector<float> h_step, c_step;
    for (int t = 0; t < L; ++t) {
        std::vector<float> x_t = {X_data[t * input_size + 0],
                                  X_data[t * input_size + 1]};
        lstm_step_ref(input_size, hidden, W_ih, W_hh, b_ih, b_hh, x_t,
                      h_prev, c_prev, h_step, c_step);
        for (int j = 0; j < hidden; ++j) {
            const float got = Y.host_f32()[t * hidden + j];
            if (!approx(got, h_step[j], 1e-5f)) {
                std::fprintf(stderr,
                    "FAIL: LSTM t=%d j=%d got=%g want=%g\n",
                    t, j, got, h_step[j]);
                ++failures;
            }
        }
        h_prev = h_step;
        c_prev = c_step;
    }
}

static void test_bilstm() {
    const int input_size = 2;
    const int hidden     = 2;
    const int L          = 3;

    // Both directions share weights — sanity check that the structure works,
    // not that brosoundml matches some external bidirectional reference.
    std::vector<float> W_ih(4 * hidden * input_size, 0.1f);
    std::vector<float> W_hh(4 * hidden * hidden,     0.05f);
    std::vector<float> b_ih(4 * hidden, 0.0f);
    std::vector<float> b_hh(4 * hidden, 0.0f);

    brosoundml::BiLSTM bi;
    bi.input_size  = input_size;
    bi.hidden_size = hidden;
    bi.forward_cell.W_ih = make_tensor(W_ih, 4 * hidden, input_size);
    bi.forward_cell.W_hh = make_tensor(W_hh, 4 * hidden, hidden);
    bi.forward_cell.b_ih = make_tensor(b_ih, 4 * hidden, 1);
    bi.forward_cell.b_hh = make_tensor(b_hh, 4 * hidden, 1);
    bi.reverse_cell = bi.forward_cell;  // same weights → bidirectional with identical cells

    std::vector<float> X_data;
    for (int t = 0; t < L; ++t) {
        X_data.push_back(0.3f * static_cast<float>(t + 1));
        X_data.push_back(0.2f * static_cast<float>(t + 1));
    }
    bt::Tensor X = make_tensor(X_data, L, input_size);
    bt::Tensor Y;
    bi.forward(X, Y);
    CHECK(Y.rows == L && Y.cols == 2 * hidden, "BiLSTM output shape (L, 2*H)");

    // Cross-check against running two separate LSTMs and concatenating —
    // brosoundml::LSTM gives us a smaller-surface oracle.
    brosoundml::LSTM lstm_fwd, lstm_rev;
    lstm_fwd.input_size = lstm_rev.input_size  = input_size;
    lstm_fwd.hidden_size = lstm_rev.hidden_size = hidden;
    lstm_fwd.cell = bi.forward_cell;
    lstm_rev.cell = bi.reverse_cell;

    bt::Tensor Y_fwd;
    lstm_fwd.forward(X, Y_fwd);

    // Reversed input → run forward LSTM → reverse output back.
    std::vector<float> X_rev_data(X_data.size());
    for (int t = 0; t < L; ++t) {
        X_rev_data[t * input_size + 0] = X_data[(L - 1 - t) * input_size + 0];
        X_rev_data[t * input_size + 1] = X_data[(L - 1 - t) * input_size + 1];
    }
    bt::Tensor X_rev = make_tensor(X_rev_data, L, input_size);
    bt::Tensor Y_rev_raw;
    lstm_rev.forward(X_rev, Y_rev_raw);

    for (int t = 0; t < L; ++t) {
        for (int j = 0; j < hidden; ++j) {
            const float got_fwd = Y.host_f32()[t * 2 * hidden + j];
            const float want_fwd = Y_fwd.host_f32()[t * hidden + j];
            const float got_rev = Y.host_f32()[t * 2 * hidden + hidden + j];
            const float want_rev = Y_rev_raw.host_f32()[(L - 1 - t) * hidden + j];
            CHECK(approx(got_fwd, want_fwd, 1e-5f),
                  "BiLSTM forward half matches a stand-alone LSTM");
            CHECK(approx(got_rev, want_rev, 1e-5f),
                  "BiLSTM reverse half matches a stand-alone LSTM on reversed input");
        }
    }
}

// ─── AdaIN1D ───────────────────────────────────────────────────────────────

static void test_ada_in_1d() {
    const int N = 1, C = 2, L = 4;
    // (N, C*L) — channel 0 has mean 2.5, variance 1.25;
    //            channel 1 has mean 10,  variance 5.
    bt::Tensor X = make_tensor({1, 2, 3, 4,    // ch0
                                7, 9, 11, 13}, // ch1
                               N, C * L);
    bt::Tensor scale = make_tensor({2.0f,  3.0f},  C, 1);
    bt::Tensor shift = make_tensor({0.5f, -1.0f},  C, 1);
    bt::Tensor Y;
    brosoundml::ada_in_1d(X, scale, shift, N, C, L, Y);
    CHECK(Y.rows == N && Y.cols == C * L, "ada_in_1d output shape");

    // For each channel: normalise to mean 0 / unit-var, then apply scale/shift.
    // Mean of each channel of the *output* should equal its `shift` value,
    // and the variance should equal `scale^2`.
    for (int c = 0; c < C; ++c) {
        float sum = 0, sumsq = 0;
        for (int l = 0; l < L; ++l) {
            const float v = Y.host_f32()[c * L + l];
            sum += v;
            sumsq += v * v;
        }
        const float mean = sum / L;
        const float var  = sumsq / L - mean * mean;
        const float expect_mean = (c == 0) ? 0.5f : -1.0f;
        const float expect_var  = (c == 0) ? 4.0f  :  9.0f;
        CHECK(approx(mean, expect_mean, 1e-3f),
              c == 0 ? "ada_in_1d channel 0 mean equals shift"
                     : "ada_in_1d channel 1 mean equals shift");
        CHECK(approx(var, expect_var, 1e-2f),
              c == 0 ? "ada_in_1d channel 0 variance equals scale^2"
                     : "ada_in_1d channel 1 variance equals scale^2");
    }
}

int main() {
    test_linear();
    test_layernorm();
    test_conv1d();
    test_lstm();
    test_bilstm();
    test_ada_in_1d();

    if (failures == 0) {
        std::printf("test_modules: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_modules: %d check(s) failed\n", failures);
    return 1;
}
