// brosoundml module-layer tests. Each module is exercised with hand-rolled
// synthetic weights so the expected output is computable by hand — no
// reference activations from an external framework involved.
//
// Each device-dependent test is parameterized over brotensor::Device so the
// CPU baseline runs unconditionally and the CUDA path runs additionally when
// `brotensor::is_available(Device::CUDA)` reports true. Module outputs run on
// CUDA are downloaded via `to_host_vector()` before comparing against the
// from-scratch CPU oracle.
#include "brosoundml/modules.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
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

static bt::Tensor make_tensor_on(bt::Device dev, const std::vector<float>& data,
                                 int rows, int cols) {
    return bt::Tensor::from_host_on(dev, data.data(), rows, cols);
}

// Download a (possibly device-resident) FP32 tensor to a host vector — works
// for tensors on CPU or CUDA. The returned vector mirrors row-major layout.
static std::vector<float> download(const bt::Tensor& t) {
    return t.to_host_vector();
}

// Convenience: tag CHECK messages with the device name without bloating each
// call site. Each test_* function passes `dev_name` through.
static std::string tag(const char* msg, const char* dev_name) {
    std::string s = "[";
    s += dev_name;
    s += "] ";
    s += msg;
    return s;
}

// ─── Linear ────────────────────────────────────────────────────────────────

static void test_linear(bt::Device dev, const char* dev_name) {
    brosoundml::Linear lin;
    lin.in_features  = 3;
    lin.out_features = 2;
    // W = [[1, 2, 3], [4, 5, 6]]; b = [10, 20].
    lin.W = make_tensor_on(dev, {1, 2, 3, 4, 5, 6}, 2, 3);
    lin.b = make_tensor_on(dev, {10, 20}, 2, 1);

    // x = [1, 1, 1]^T → y = [1+2+3+10, 4+5+6+20] = [16, 35].
    bt::Tensor x = make_tensor_on(dev, {1, 1, 1}, 3, 1);
    bt::Tensor y;
    lin.forward(x, y);
    auto yh = download(y);
    CHECK(y.rows == 2 && y.cols == 1, tag("Linear::forward output shape", dev_name).c_str());
    CHECK(approx(yh[0], 16.0f), tag("Linear::forward y[0]", dev_name).c_str());
    CHECK(approx(yh[1], 35.0f), tag("Linear::forward y[1]", dev_name).c_str());

    // Batched: X = [[1,1,1], [2,2,2]] → Y = [[16,35], [22,50]].
    bt::Tensor X = make_tensor_on(dev, {1, 1, 1, 2, 2, 2}, 2, 3);
    bt::Tensor Y;
    lin.forward_batched(X, Y);
    auto Yh = download(Y);
    CHECK(Y.rows == 2 && Y.cols == 2, tag("Linear::forward_batched output shape", dev_name).c_str());
    CHECK(approx(Yh[0], 16.0f) && approx(Yh[1], 35.0f),
          tag("Linear::forward_batched row 0", dev_name).c_str());
    CHECK(approx(Yh[2], 22.0f) && approx(Yh[3], 50.0f),
          tag("Linear::forward_batched row 1", dev_name).c_str());

    CHECK(throws_runtime_error([&] {
              bt::Tensor bad_x = make_tensor_on(dev, {1, 1}, 2, 1);
              bt::Tensor out;
              lin.forward(bad_x, out);
          }),
          tag("Linear::forward rejects wrong input shape", dev_name).c_str());
}

// ─── LayerNorm ─────────────────────────────────────────────────────────────

static void test_layernorm(bt::Device dev, const char* dev_name) {
    brosoundml::LayerNorm ln;
    ln.features = 4;
    ln.gamma = make_tensor_on(dev, {1, 1, 1, 1}, 4, 1);
    ln.beta  = make_tensor_on(dev, {0, 0, 0, 0}, 4, 1);

    bt::Tensor X = make_tensor_on(dev, {1, 2, 3, 4,
                                        10, 12, 14, 16}, 2, 4);
    bt::Tensor Y;
    ln.forward(X, Y);
    auto Yh = download(Y);
    CHECK(Y.rows == 2 && Y.cols == 4, tag("LayerNorm output shape", dev_name).c_str());

    for (int r = 0; r < 2; ++r) {
        float sum = 0, sumsq = 0;
        for (int c = 0; c < 4; ++c) {
            const float v = Yh[r * 4 + c];
            sum += v;
            sumsq += v * v;
        }
        const float mean = sum / 4.0f;
        const float var  = sumsq / 4.0f - mean * mean;
        CHECK(approx(mean, 0.0f, 1e-4f),
              r == 0 ? tag("LayerNorm row 0 mean ~ 0", dev_name).c_str()
                     : tag("LayerNorm row 1 mean ~ 0", dev_name).c_str());
        CHECK(approx(var, 1.0f, 1e-3f),
              r == 0 ? tag("LayerNorm row 0 variance ~ 1", dev_name).c_str()
                     : tag("LayerNorm row 1 variance ~ 1", dev_name).c_str());
    }

    ln.gamma = make_tensor_on(dev, {2, 2, 2, 2}, 4, 1);
    ln.beta  = make_tensor_on(dev, {5, 5, 5, 5}, 4, 1);
    bt::Tensor Y2;
    ln.forward(X, Y2);
    auto Y2h = download(Y2);
    float sum = 0;
    for (int c = 0; c < 4; ++c) sum += Y2h[c];
    CHECK(approx(sum / 4.0f, 5.0f, 1e-4f),
          tag("LayerNorm row 0 mean ~ beta=5 with non-unit gamma", dev_name).c_str());
}

// ─── Conv1d ────────────────────────────────────────────────────────────────

static void test_conv1d(bt::Device dev, const char* dev_name) {
    brosoundml::Conv1d conv;
    conv.in_channels  = 1;
    conv.out_channels = 1;
    conv.kernel_size  = 3;
    conv.stride       = 1;
    conv.padding      = 0;
    conv.dilation     = 1;
    conv.groups       = 1;
    conv.W = make_tensor_on(dev, {1, 1, 1}, 1, 3);
    conv.b = make_tensor_on(dev, {0}, 1, 1);

    bt::Tensor X = make_tensor_on(dev, {1, 2, 3, 4, 5}, 1, 5);
    bt::Tensor Y;
    conv.forward(X, /*N=*/1, /*L=*/5, Y);
    auto Yh = download(Y);
    CHECK(Y.rows == 1 && Y.cols == 3, tag("Conv1d output shape", dev_name).c_str());
    CHECK(approx(Yh[0], 6.0f, 1e-4f),  tag("Conv1d y[0] = 6", dev_name).c_str());
    CHECK(approx(Yh[1], 9.0f, 1e-4f),  tag("Conv1d y[1] = 9", dev_name).c_str());
    CHECK(approx(Yh[2], 12.0f, 1e-4f), tag("Conv1d y[2] = 12", dev_name).c_str());

    conv.b = make_tensor_on(dev, {100}, 1, 1);
    bt::Tensor Y2;
    conv.forward(X, 1, 5, Y2);
    auto Y2h = download(Y2);
    CHECK(approx(Y2h[0], 106.0f, 1e-4f), tag("Conv1d adds bias to y[0]", dev_name).c_str());
}

// ─── LSTM (single direction) ───────────────────────────────────────────────

// Reference cell computation. Stays CPU FP32 — it's the oracle we compare
// the device-resident module output against.
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

static void test_lstm(bt::Device dev, const char* dev_name) {
    const int input_size = 2;
    const int hidden     = 3;
    const int L          = 4;

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
    lstm.cell.W_ih = make_tensor_on(dev, W_ih, 4 * hidden, input_size);
    lstm.cell.W_hh = make_tensor_on(dev, W_hh, 4 * hidden, hidden);
    lstm.cell.b_ih = make_tensor_on(dev, b_ih, 4 * hidden, 1);
    lstm.cell.b_hh = make_tensor_on(dev, b_hh, 4 * hidden, 1);

    std::vector<float> X_data;
    for (int t = 0; t < L; ++t) {
        X_data.push_back(0.1f * static_cast<float>(t + 1));
        X_data.push_back(-0.2f * static_cast<float>(t + 1));
    }
    bt::Tensor X = make_tensor_on(dev, X_data, L, input_size);
    bt::Tensor Y;
    lstm.forward(X, Y);
    CHECK(Y.rows == L && Y.cols == hidden, tag("LSTM output shape", dev_name).c_str());
    auto Yh = download(Y);

    // Tolerance: LSTM accumulates per-step FP noise — looser than 1e-5 on
    // CUDA, still tight on CPU.
    const float tol = (dev == bt::Device::CPU) ? 1e-5f : 1e-3f;

    std::vector<float> h_prev(hidden, 0.0f), c_prev(hidden, 0.0f);
    std::vector<float> h_step, c_step;
    for (int t = 0; t < L; ++t) {
        std::vector<float> x_t = {X_data[t * input_size + 0],
                                  X_data[t * input_size + 1]};
        lstm_step_ref(input_size, hidden, W_ih, W_hh, b_ih, b_hh, x_t,
                      h_prev, c_prev, h_step, c_step);
        for (int j = 0; j < hidden; ++j) {
            const float got = Yh[t * hidden + j];
            if (!approx(got, h_step[j], tol)) {
                std::fprintf(stderr,
                    "FAIL: [%s] LSTM t=%d j=%d got=%g want=%g\n",
                    dev_name, t, j, got, h_step[j]);
                ++failures;
            }
        }
        h_prev = h_step;
        c_prev = c_step;
    }
}

static void test_bilstm(bt::Device dev, const char* dev_name) {
    const int input_size = 2;
    const int hidden     = 2;
    const int L          = 3;

    std::vector<float> W_ih(4 * hidden * input_size, 0.1f);
    std::vector<float> W_hh(4 * hidden * hidden,     0.05f);
    std::vector<float> b_ih(4 * hidden, 0.0f);
    std::vector<float> b_hh(4 * hidden, 0.0f);

    brosoundml::BiLSTM bi;
    bi.input_size  = input_size;
    bi.hidden_size = hidden;
    bi.forward_cell.W_ih = make_tensor_on(dev, W_ih, 4 * hidden, input_size);
    bi.forward_cell.W_hh = make_tensor_on(dev, W_hh, 4 * hidden, hidden);
    bi.forward_cell.b_ih = make_tensor_on(dev, b_ih, 4 * hidden, 1);
    bi.forward_cell.b_hh = make_tensor_on(dev, b_hh, 4 * hidden, 1);
    // Same-weights reverse cell: re-uploaded on the same device.
    bi.reverse_cell.W_ih = make_tensor_on(dev, W_ih, 4 * hidden, input_size);
    bi.reverse_cell.W_hh = make_tensor_on(dev, W_hh, 4 * hidden, hidden);
    bi.reverse_cell.b_ih = make_tensor_on(dev, b_ih, 4 * hidden, 1);
    bi.reverse_cell.b_hh = make_tensor_on(dev, b_hh, 4 * hidden, 1);

    std::vector<float> X_data;
    for (int t = 0; t < L; ++t) {
        X_data.push_back(0.3f * static_cast<float>(t + 1));
        X_data.push_back(0.2f * static_cast<float>(t + 1));
    }
    bt::Tensor X = make_tensor_on(dev, X_data, L, input_size);
    bt::Tensor Y;
    bi.forward(X, Y);
    CHECK(Y.rows == L && Y.cols == 2 * hidden,
          tag("BiLSTM output shape (L, 2*H)", dev_name).c_str());
    auto Yh = download(Y);

    // Stand-alone LSTM oracles run on the same device for an apples-to-apples
    // comparison (no cross-device numerical drift in the oracle).
    brosoundml::LSTM lstm_fwd, lstm_rev;
    lstm_fwd.input_size = lstm_rev.input_size  = input_size;
    lstm_fwd.hidden_size = lstm_rev.hidden_size = hidden;
    lstm_fwd.cell.W_ih = make_tensor_on(dev, W_ih, 4 * hidden, input_size);
    lstm_fwd.cell.W_hh = make_tensor_on(dev, W_hh, 4 * hidden, hidden);
    lstm_fwd.cell.b_ih = make_tensor_on(dev, b_ih, 4 * hidden, 1);
    lstm_fwd.cell.b_hh = make_tensor_on(dev, b_hh, 4 * hidden, 1);
    lstm_rev.cell.W_ih = make_tensor_on(dev, W_ih, 4 * hidden, input_size);
    lstm_rev.cell.W_hh = make_tensor_on(dev, W_hh, 4 * hidden, hidden);
    lstm_rev.cell.b_ih = make_tensor_on(dev, b_ih, 4 * hidden, 1);
    lstm_rev.cell.b_hh = make_tensor_on(dev, b_hh, 4 * hidden, 1);

    bt::Tensor Y_fwd;
    lstm_fwd.forward(X, Y_fwd);
    auto Y_fwd_h = download(Y_fwd);

    std::vector<float> X_rev_data(X_data.size());
    for (int t = 0; t < L; ++t) {
        X_rev_data[t * input_size + 0] = X_data[(L - 1 - t) * input_size + 0];
        X_rev_data[t * input_size + 1] = X_data[(L - 1 - t) * input_size + 1];
    }
    bt::Tensor X_rev = make_tensor_on(dev, X_rev_data, L, input_size);
    bt::Tensor Y_rev_raw;
    lstm_rev.forward(X_rev, Y_rev_raw);
    auto Y_rev_h = download(Y_rev_raw);

    const float tol = (dev == bt::Device::CPU) ? 1e-5f : 1e-3f;
    for (int t = 0; t < L; ++t) {
        for (int j = 0; j < hidden; ++j) {
            const float got_fwd = Yh[t * 2 * hidden + j];
            const float want_fwd = Y_fwd_h[t * hidden + j];
            const float got_rev = Yh[t * 2 * hidden + hidden + j];
            const float want_rev = Y_rev_h[(L - 1 - t) * hidden + j];
            CHECK(approx(got_fwd, want_fwd, tol),
                  tag("BiLSTM forward half matches a stand-alone LSTM", dev_name).c_str());
            CHECK(approx(got_rev, want_rev, tol),
                  tag("BiLSTM reverse half matches a stand-alone LSTM on reversed input", dev_name).c_str());
        }
    }
}

// ─── AdaIN1D ───────────────────────────────────────────────────────────────

static void test_ada_in_1d(bt::Device dev, const char* dev_name) {
    const int N = 1, C = 2, L = 4;
    bt::Tensor X = make_tensor_on(dev, {1, 2, 3, 4,
                                        7, 9, 11, 13}, N, C * L);
    bt::Tensor scale = make_tensor_on(dev, {2.0f,  3.0f}, C, 1);
    bt::Tensor shift = make_tensor_on(dev, {0.5f, -1.0f}, C, 1);
    bt::Tensor Y;
    brosoundml::ada_in_1d(X, scale, shift, N, C, L, Y);
    CHECK(Y.rows == N && Y.cols == C * L,
          tag("ada_in_1d output shape", dev_name).c_str());
    auto Yh = download(Y);

    for (int c = 0; c < C; ++c) {
        float sum = 0, sumsq = 0;
        for (int l = 0; l < L; ++l) {
            const float v = Yh[c * L + l];
            sum += v;
            sumsq += v * v;
        }
        const float mean = sum / L;
        const float var  = sumsq / L - mean * mean;
        const float expect_mean = (c == 0) ? 0.5f : -1.0f;
        const float expect_var  = (c == 0) ? 4.0f  :  9.0f;
        CHECK(approx(mean, expect_mean, 1e-3f),
              c == 0 ? tag("ada_in_1d channel 0 mean equals shift", dev_name).c_str()
                     : tag("ada_in_1d channel 1 mean equals shift", dev_name).c_str());
        CHECK(approx(var, expect_var, 1e-2f),
              c == 0 ? tag("ada_in_1d channel 0 variance equals scale^2", dev_name).c_str()
                     : tag("ada_in_1d channel 1 variance equals scale^2", dev_name).c_str());
    }
}

// ─── MHA / CrossAttention smoke ─────────────────────────────────────────────

static void test_mha_self(bt::Device dev, const char* dev_name) {
    const int D = 4;
    const int L = 3;
    const int H = 1;
    const float ID[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const float ZERO[4] = {0, 0, 0, 0};

    brosoundml::MHA mha;
    mha.num_heads = H;
    mha.embed_dim = D;
    mha.Wq = bt::Tensor::from_host_on(dev, ID, D, D);
    mha.Wk = bt::Tensor::from_host_on(dev, ID, D, D);
    mha.Wv = bt::Tensor::from_host_on(dev, ID, D, D);
    mha.Wo = bt::Tensor::from_host_on(dev, ID, D, D);
    mha.bq = bt::Tensor::from_host_on(dev, ZERO, D, 1);
    mha.bk = bt::Tensor::from_host_on(dev, ZERO, D, 1);
    mha.bv = bt::Tensor::from_host_on(dev, ZERO, D, 1);
    mha.bo = bt::Tensor::from_host_on(dev, ZERO, D, 1);

    const float X[12] = {
        10, 0, 0, 0,
         0, 10, 0, 0,
         0, 0, 10, 0,
    };
    bt::Tensor X_t  = bt::Tensor::from_host_on(dev, X, L, D);
    bt::Tensor out;
    mha.forward(X_t, /*d_mask=*/nullptr, out);
    CHECK(out.rows == L && out.cols == D,
          tag("MHA output shape (L, D)", dev_name).c_str());
    auto oh = download(out);
    // Softmax accumulates ~3e-3 noise on CUDA; CPU stays at 1e-2 because the
    // mass is sharp and the test bound is already loose.
    const float tol = 1e-2f;
    for (int r = 0; r < L; ++r) {
        for (int c = 0; c < D; ++c) {
            CHECK(approx(oh[r * D + c], X[r * D + c], tol),
                  tag("MHA self-attention concentrates on matching key", dev_name).c_str());
        }
    }
}

static void test_cross_attention(bt::Device dev, const char* dev_name) {
    const int D     = 4;
    const int D_ctx = 2;
    const int Lq    = 1;
    const int Lk    = 2;

    const float Wq_d[16] = {
        1, 0, 0, 0,  0, 1, 0, 0,
        0, 0, 1, 0,  0, 0, 0, 1,
    };
    const float Wkv_d[8] = {
        1, 0,  0, 1,
        0, 0,  0, 0,
    };
    const float Wo_d[16] = {
        1, 0, 0, 0,  0, 1, 0, 0,
        0, 0, 1, 0,  0, 0, 0, 1,
    };
    const float ZERO_D[4] = {0, 0, 0, 0};

    brosoundml::CrossAttention xa;
    xa.num_heads = 1;
    xa.embed_dim = D;
    xa.ctx_dim   = D_ctx;
    xa.Wq = bt::Tensor::from_host_on(dev, Wq_d,  D, D);
    xa.Wk = bt::Tensor::from_host_on(dev, Wkv_d, D, D_ctx);
    xa.Wv = bt::Tensor::from_host_on(dev, Wkv_d, D, D_ctx);
    xa.Wo = bt::Tensor::from_host_on(dev, Wo_d,  D, D);
    xa.bq = bt::Tensor::from_host_on(dev, ZERO_D, D, 1);
    xa.bk = bt::Tensor::from_host_on(dev, ZERO_D, D, 1);
    xa.bv = bt::Tensor::from_host_on(dev, ZERO_D, D, 1);
    xa.bo = bt::Tensor::from_host_on(dev, ZERO_D, D, 1);

    const float X_d[4]   = {10, 0, 0, 0};
    const float Ctx_d[4] = {1, 0,  0, 1};
    bt::Tensor X_t   = bt::Tensor::from_host_on(dev, X_d,   Lq, D);
    bt::Tensor Ctx_t = bt::Tensor::from_host_on(dev, Ctx_d, Lk, D_ctx);

    bt::Tensor out;
    xa.forward(X_t, Ctx_t, /*d_mask=*/nullptr, out);
    CHECK(out.rows == Lq && out.cols == D,
          tag("CrossAttention output shape (Lq, D)", dev_name).c_str());
    auto oh = download(out);
    CHECK(approx(oh[0], 1.0f, 1e-2f),
          tag("CrossAttention attends to matching ctx row", dev_name).c_str());
    CHECK(approx(oh[1], 0.0f, 1e-2f),
          tag("CrossAttention non-attended ctx row near zero", dev_name).c_str());
}

static void run_all(bt::Device dev, const char* dev_name) {
    test_linear(dev, dev_name);
    test_layernorm(dev, dev_name);
    test_conv1d(dev, dev_name);
    test_lstm(dev, dev_name);
    test_bilstm(dev, dev_name);
    test_ada_in_1d(dev, dev_name);
    test_mha_self(dev, dev_name);
    test_cross_attention(dev, dev_name);
}

int main() {
    bt::init();
    run_all(bt::Device::CPU, "CPU");
    if (bt::is_available(bt::Device::CUDA)) {
        run_all(bt::Device::CUDA, "CUDA");
    } else {
        std::printf("test_modules: CUDA not available — CUDA path skipped\n");
    }

    if (failures == 0) {
        std::printf("test_modules: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_modules: %d check(s) failed\n", failures);
    return 1;
}
