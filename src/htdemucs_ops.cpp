// HTDemucs building blocks — see htdemucs_model.h. Every op here is a
// composition of brotensor ops; nothing touches device memory directly.
#include "htdemucs_model.h"

#include <brotensor/ops.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml::htd {

void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: htdemucs: " + msg);
}

namespace {

// The same buffer seen as (r, c); r*c must match.
bt::Tensor reshaped(const bt::Tensor& t, int r, int c) {
    if (static_cast<long long>(r) * c != static_cast<long long>(t.rows) * t.cols)
        fail("reshape: element count mismatch");
    return bt::Tensor::view(t.device, t.data, r, c, t.dtype);
}

void ensure(bt::Tensor& t, bt::Device dev, int r, int c) {
    if (t.data == nullptr || t.device != dev || t.rows != r || t.cols != c ||
        t.dtype != bt::Dtype::FP32)
        t = bt::Tensor::empty_on(dev, r, c);
}

}  // namespace

bt::Tensor int_tensor(bt::Device dev, const std::vector<int>& v) {
    static_assert(sizeof(int) == 4, "int32 indices");
    return bt::Tensor::from_raw_bytes_on(dev, v.data(), static_cast<int>(v.size()), 1,
                                         bt::Dtype::INT32, v.size() * sizeof(int));
}

bt::Tensor fill(bt::Device dev, int r, int c, float value) {
    std::vector<float> h(static_cast<std::size_t>(r) * c, value);
    return bt::Tensor::from_host_on(dev, h.data(), r, c);
}

// ── Conv ─────────────────────────────────────────────────────────────────────

int Conv::out_h(int H) const {
    if (transpose) return (H - 1) * sh - 2 * ph + (kh - 1) + 1;
    return (H + 2 * ph - (kh - 1) - 1) / sh + 1;
}

int Conv::out_w(int W) const {
    if (transpose) return (W - 1) * sw - 2 * pw + dw * (kw - 1) + 1;
    return (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
}

void Conv::forward(const bt::Tensor& X, int N, int H, int W, bt::Tensor& Y) const {
    if (transpose)
        bt::conv_transpose2d_forward(X, w, &b, N, cin, H, W, cout, kh, kw, sh, sw, ph, pw,
                                     /*out_pad*/ 0, 0, /*dil*/ 1, dw, /*groups*/ 1, Y);
    else
        bt::conv2d_forward(X, w, &b, N, cin, H, W, cout, kh, kw, sh, sw, ph, pw,
                           /*dil_h*/ 1, dw, /*groups*/ 1, Y);
}

// ── GroupNorm(1, C) ──────────────────────────────────────────────────────────

void GroupNorm1::forward(const bt::Tensor& X, int N, int L, bt::Tensor& Y) const {
    bt::group_norm_forward(X, gamma, beta, N, C, 1, L, /*groups*/ 1, 1e-5f, Y);
}

// ── GLU ──────────────────────────────────────────────────────────────────────

void glu(const bt::Tensor& X, int N, int D, bt::Tensor& Y) {
    if (X.rows * static_cast<long long>(X.cols) != 2LL * N * D) fail("glu: shape mismatch");
    bt::Tensor g;
    ensure(Y, X.device, N, D);
    ensure(g, X.device, N, D);
    bt::copy_d2d_strided(X, 0, 2 * D, Y, 0, D, D, N);
    bt::copy_d2d_strided(X, D, 2 * D, g, 0, D, D, N);
    bt::sigmoid_forward(g, g);
    bt::mul_inplace(Y, g);
}

// ── DConv ────────────────────────────────────────────────────────────────────

void DConv::forward(bt::Tensor& X, int N, int L) const {
    bt::Tensor h, t;
    for (const DConvLayer& l : layers) {
        l.c1.forward(X, N, 1, L, h);                       // (N, hid*L)
        l.n1.forward(h, N, L, t);
        bt::gelu_exact_forward(t, t);
        l.c2.forward(t, N, 1, L, h);                       // (N, 2C*L)
        l.n2.forward(h, N, L, t);
        glu(t, N, C * L, h);                               // (N, C*L), LayerScale folded
        bt::add_inplace(X, h);
    }
}

// ── per-row statistics ───────────────────────────────────────────────────────

void row_mean_std(const bt::Tensor& X, bt::Tensor& mean, bt::Tensor& std) {
    const int R = X.rows, n = X.cols;
    if (n < 2) fail("row_mean_std: need at least 2 columns");
    bt::sum_rows(X, mean);
    bt::scale_inplace(mean, 1.0f / static_cast<float>(n));
    // Two passes (sum of squared deviations), never E[x^2] - E[x]^2.
    bt::Tensor ones = fill(X.device, R, 1, 1.0f), zero = fill(X.device, R, 1, 0.0f);
    bt::Tensor d;
    bt::batch_norm_inference(reshaped(X, 1, R * n), ones, zero, mean, ones, 1, R, 1, n, 0.0f, d);
    bt::mul_inplace(d, d);
    bt::Tensor ss;
    bt::sum_rows(reshaped(d, R, n), ss);
    bt::scale_inplace(ss, 1.0f / static_cast<float>(n - 1));
    // std = var * rsqrt(var); a silent row (var 0) gives 0, as torch does.
    bt::clamp(ss, 1e-30f, 3.0e38f);
    bt::Tensor r;
    bt::rsqrt_forward(ss, r);
    bt::mul_inplace(ss, r);
    std = std::move(ss);
}

void row_affine(const bt::Tensor& X, const bt::Tensor& scale, const bt::Tensor& shift,
                bt::Tensor& Y) {
    const int R = X.rows, n = X.cols;
    bt::Tensor zero = fill(X.device, R, 1, 0.0f), ones = fill(X.device, R, 1, 1.0f);
    if (&Y == &X) {
        bt::Tensor v = reshaped(X, 1, R * n);
        bt::batch_norm_inference(v, scale, shift, zero, ones, 1, R, 1, n, 0.0f, v);
        return;
    }
    ensure(Y, X.device, R, n);
    bt::Tensor yv = reshaped(Y, 1, R * n);
    bt::batch_norm_inference(reshaped(X, 1, R * n), scale, shift, zero, ones, 1, R, 1, n, 0.0f, yv);
}

void row_normalize(const bt::Tensor& X, const bt::Tensor& mean, const bt::Tensor& std,
                   float eps, bt::Tensor& Y) {
    const int R = X.rows, n = X.cols;
    // (x - m) / sqrt(var) with var = (eps + std)^2.
    bt::Tensor var = std.clone();
    bt::add_scalar_inplace(var, eps);
    bt::mul_inplace(var, var);
    bt::Tensor ones = fill(X.device, R, 1, 1.0f), zero = fill(X.device, R, 1, 0.0f);
    ensure(Y, X.device, R, n);
    bt::Tensor yv = reshaped(Y, 1, R * n);
    bt::batch_norm_inference(reshaped(X, 1, R * n), ones, zero, mean, var, 1, R, 1, n, 0.0f, yv);
}

void reshape(bt::Tensor& t, int r, int c) {
    if (static_cast<long long>(r) * c != static_cast<long long>(t.rows) * t.cols)
        fail("reshape: element count mismatch");
    t.rows = r;
    t.cols = c;
}

void slice_cols(const bt::Tensor& X, int c0, int w, bt::Tensor& Y) {
    ensure(Y, X.device, X.rows, w);
    bt::copy_d2d_strided(X, c0, X.cols, Y, 0, w, w, X.rows);
}

}  // namespace brosoundml::htd
