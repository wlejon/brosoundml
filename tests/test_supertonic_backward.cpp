// Finite-difference coverage for the Supertonic-3 backward FOUNDATION atoms
// (src/supertonic_backward.{h,cpp}): the conv / transpose / convnext-block
// input-backward kernels the later agents (style attention, flow field,
// vocoder, ECAPA) compose on top of. Built up one atom per chunk and each
// gradient-checked in isolation before the next relies on it.
//
// Verification methodology (mirrors test_kokoro_decoder_backward.cpp): the
// linear / smooth atoms (the convs and the transpose) are checked per element
// against a central finite difference; the LayerNorm-heavy ConvNeXt block is
// checked with a directional dot-product test instead (per-element FD of a
// stacked-norm Jacobian is unreliable in fp32). Each test owns a fixed RNG so
// test order never perturbs another's inputs. CPU-resident FP32.

#include "supertonic_backward.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;
namespace st = brosoundml::st_detail;

static int g_failures = 0;

static void expect_near(double a, double b, double abs_eps, double rel_eps,
                        const std::string& ctx) {
    const double d = std::fabs(a - b);
    const double m = std::fmax(std::fabs(a), std::fabs(b));
    if (d <= abs_eps || d <= rel_eps * m) return;
    std::printf("  FAIL %s: %.8g vs %.8g (|d|=%.3g)\n", ctx.c_str(), a, b, d);
    ++g_failures;
}

static Tensor make_random(int r, int c, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> u(-scale, scale);
    std::vector<float> v(static_cast<std::size_t>(r) * c);
    for (auto& x : v) x = u(rng);
    return Tensor::from_host_on(Device::CPU, v.data(), r, c);
}

// A conv weight in brotensor (Cout, (Cin/groups)*k) OIL layout.
static st::ConvW make_convw(int cout, int cin_pg, int k, bool bias,
                            std::mt19937& rng, float scale = 0.3f) {
    st::ConvW c;
    c.cout = cout; c.cin_pg = cin_pg; c.k = k; c.has_b = bias;
    c.w = make_random(cout, cin_pg * k, rng, scale);
    if (bias) c.b = make_random(cout, 1, rng, scale);
    return c;
}

// Per-element central finite-difference check of `analytic` w.r.t. tensor P.
static void fd_per_element(Tensor& P, const Tensor& analytic,
                           const std::function<double()>& loss_now,
                           double abs_tol, double rel_tol,
                           const std::string& name) {
    const float h = 1e-3f;
    for (int i = 0; i < P.size(); ++i) {
        float saved = P[i];
        P[i] = saved + h; double lp = loss_now();
        P[i] = saved - h; double lm = loss_now();
        P[i] = saved;
        double fd = (lp - lm) / (2.0 * h);
        expect_near(analytic[i], fd, abs_tol, rel_tol,
                    name + "[" + std::to_string(i) + "]");
    }
}

// Directional check: <analytic, v> against the central difference of the loss
// along a random direction v over the whole tensor P.
static void fd_directional(Tensor& P, const Tensor& analytic,
                           const std::function<double()>& loss_now,
                           std::mt19937& rng, const std::string& name,
                           double abs_tol = 4e-3, double rel_tol = 8e-3) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(static_cast<std::size_t>(P.size()));
    std::vector<float> saved(static_cast<std::size_t>(P.size()));
    double g_an = 0.0;
    for (int i = 0; i < P.size(); ++i) { v[i] = u(rng); saved[i] = P[i]; g_an += static_cast<double>(analytic[i]) * v[i]; }
    const float h = 1e-3f;
    for (int i = 0; i < P.size(); ++i) P[i] = saved[i] + h * v[i];
    double lp = loss_now();
    for (int i = 0; i < P.size(); ++i) P[i] = saved[i] - h * v[i];
    double lm = loss_now();
    for (int i = 0; i < P.size(); ++i) P[i] = saved[i];
    double g_fd = (lp - lm) / (2.0 * h);
    expect_near(g_an, g_fd, abs_tol, rel_tol, name);
}

static std::function<double()> mse_loss(const Tensor& out, const Tensor& target) {
    return [&out, &target]() {
        double s = 0.0;
        for (int i = 0; i < out.size(); ++i) { double d = out[i] - target[i]; s += 0.5 * d * d; }
        return s;
    };
}

// ─── pconv: 1×1 matmul path ─────────────────────────────────────────────────
static void test_pconv_1x1() {
    std::mt19937 rng(0x5701u);
    const int Cin = 6, Cout = 8, L = 10;
    st::ConvW c = make_convw(Cout, Cin, 1, /*bias=*/true, rng);

    Tensor x      = make_random(1, Cin * L, rng, 0.7f);
    Tensor target = make_random(1, Cout * L, rng, 0.5f);

    st::PConvCache cache; Tensor y;
    auto fwd = [&]() { st::pconv_forward_train(x, c, Cin, L, 1, 1, 0, 0, 0, y, cache); };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX; st::pconv_backward(c, cache, dY, dX);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_per_element(x, dX, loss, 2e-3, 5e-3, "pconv_1x1.dX");
}

// ─── pconv: padded k>1 conv path (groups=1) ─────────────────────────────────
static void test_pconv_k3() {
    std::mt19937 rng(0x5702u);
    const int Cin = 4, Cout = 5, L = 12, k = 3;
    st::ConvW c = make_convw(Cout, Cin, k, /*bias=*/true, rng);

    Tensor x      = make_random(1, Cin * L, rng, 0.7f);
    // k=3, dil=1, pad (1,1) -> L_out = L.
    Tensor target = make_random(1, Cout * L, rng, 0.5f);

    st::PConvCache cache; Tensor y;
    auto fwd = [&]() { st::pconv_forward_train(x, c, Cin, L, 1, 1, 1, 1, 0, y, cache); };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX; st::pconv_backward(c, cache, dY, dX);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_per_element(x, dX, loss, 2e-3, 5e-3, "pconv_k3.dX");
}

// ─── pconv: depthwise dilated edge-pad (groups=C), as used by ConvNeXt ───────
static void test_pconv_depthwise() {
    std::mt19937 rng(0x5703u);
    const int C = 6, L = 14, k = 5, dil = 2;
    const int pad = 2 * dil;
    st::ConvW c = make_convw(/*cout=*/C, /*cin_pg=*/1, k, /*bias=*/true, rng);

    Tensor x      = make_random(1, C * L, rng, 0.7f);
    Tensor target = make_random(1, C * L, rng, 0.5f);  // L_out = L

    st::PConvCache cache; Tensor y;
    auto fwd = [&]() { st::pconv_forward_train(x, c, C, L, dil, C, pad, pad, /*edge=*/2, y, cache); };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX; st::pconv_backward(c, cache, dY, dX);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_per_element(x, dX, loss, 2e-3, 5e-3, "pconv_dw.dX");
}

// ─── rconv: causal left-replicate-pad conv ──────────────────────────────────
static void test_rconv() {
    std::mt19937 rng(0x5704u);
    const int Cin = 3, Cout = 5, L = 12, k = 3, dil = 1;
    st::ConvW c = make_convw(Cout, Cin, k, /*bias=*/true, rng);

    Tensor x      = make_random(1, Cin * L, rng, 0.7f);
    Tensor target = make_random(1, Cout * L, rng, 0.5f);  // L_out = L

    st::PConvCache cache; Tensor y;
    auto fwd = [&]() { st::rconv_forward_train(x, c, Cin, L, dil, 1, y, cache); };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX; st::pconv_backward(c, cache, dY, dX);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_per_element(x, dX, loss, 2e-3, 5e-3, "rconv.dX");
}

// ─── seq transpose (host adjoint) ───────────────────────────────────────────
static void test_transpose() {
    std::mt19937 rng(0x5705u);
    const int R = 5, C = 7;
    Tensor x      = make_random(1, R * C, rng, 0.8f);  // [R,C] channel-major
    Tensor target = make_random(1, C * R, rng, 0.5f);  // [C,R]

    Tensor y;
    auto fwd = [&]() { st::transpose2d_forward(x, R, C, y); };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX; st::transpose2d_backward(dY, R, C, dX);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_per_element(x, dX, loss, 1e-3, 3e-3, "transpose.dX");
}

// ─── ConvNeXt-1D block ──────────────────────────────────────────────────────
static st::ConvNeXtBlock make_convnext(int C, int inter, int dil,
                                       std::mt19937& rng) {
    st::ConvNeXtBlock blk;
    blk.dil = dil;
    blk.dw  = make_convw(/*cout=*/C, /*cin_pg=*/1, /*k=*/5, /*bias=*/true, rng);  // depthwise
    blk.ln_g = make_random(C, 1, rng, 0.4f);
    blk.ln_b = make_random(C, 1, rng, 0.3f);
    blk.pw1 = make_convw(/*cout=*/inter, /*cin_pg=*/C, /*k=*/1, /*bias=*/true, rng);
    blk.pw2 = make_convw(/*cout=*/C, /*cin_pg=*/inter, /*k=*/1, /*bias=*/true, rng);
    return blk;
}

static void test_convnext(int dil, std::uint32_t seed, const char* label) {
    std::mt19937 rng(seed);
    const int C = 8, L = 16, inter = 16;
    st::ConvNeXtBlock blk = make_convnext(C, inter, dil, rng);

    Tensor x      = make_random(1, C * L, rng, 0.6f);
    Tensor target = make_random(1, C * L, rng, 0.5f);

    st::ConvNeXtCache cache; Tensor y;
    auto fwd = [&]() { st::convnext_block_forward_train(x, blk, C, L, dil, y, cache); };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX; st::convnext_block_backward(blk, cache, dY, dX);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_directional(x, dX, loss, rng, std::string(label) + ".dX");
}

int main() {
    brotensor::init();
    test_pconv_1x1();
    test_pconv_k3();
    test_pconv_depthwise();
    test_rconv();
    test_transpose();
    test_convnext(/*dil=*/1, 0xC0DE01u, "convnext_dil1");
    test_convnext(/*dil=*/2, 0xC0DE02u, "convnext_dil2");

    if (g_failures) { std::printf("test_supertonic_backward: %d failure(s)\n", g_failures); return 1; }
    std::printf("test_supertonic_backward: OK\n");
    return 0;
}
