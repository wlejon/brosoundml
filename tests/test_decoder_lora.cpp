// Standalone CPU coverage for the decoder-LoRA conditioning gate (ConditioningGate).
//
// Verifies:
//   * Neutral identity: g(0,0,0) = 0 EXACTLY — for the freshly-made gate AND
//     for arbitrary (randomised) weights, since there are no biases. This is
//     what guarantees the slider center reproduces the base model.
//   * Zero-init: a freshly-made gate has W2 == 0 (LoRA B=0 convention), so the
//     gate is identically zero before any training.
//   * Backward is exact: dW1, dW2 match central finite differences of a scalar
//     MSE loss (checked with randomised, non-zero weights).
//
// CPU-resident, FP32. Plain executable; exits non-zero on any failure.

#include "brosoundml/decoder_lora.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;
using brosoundml::ConditioningGate;

static int g_failures = 0;

static void expect_near(double a, double b, double abs_eps, double rel_eps,
                        const std::string& ctx) {
    const double d = std::fabs(a - b);
    const double m = std::fmax(std::fabs(a), std::fabs(b));
    if (d <= abs_eps || d <= rel_eps * m) return;
    std::printf("  FAIL %s: %.8g vs %.8g (|d|=%.3g)\n", ctx.c_str(), a, b, d);
    ++g_failures;
}

static void fill_random(Tensor& t, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> u(-scale, scale);
    for (int i = 0; i < t.size(); ++i) t[i] = u(rng);
}

int main() {
    brotensor::init();
    std::mt19937 rng(20260606u);

    const int gate_dim = 4, hidden = 16;
    ConditioningGate g = ConditioningGate::make(gate_dim, hidden, Device::CPU, 12345u);

    // ── zero-init: W2 is exactly zero ───────────────────────────────────────
    for (int i = 0; i < g.W2.size(); ++i)
        expect_near(g.W2[i], 0.0, 0.0, 0.0, "W2_zero_init[" + std::to_string(i) + "]");

    // ── neutral identity on the fresh (W2==0) gate: g(0) == 0 ───────────────
    Tensor v0 = Tensor::zeros_on(Device::CPU, 3, 1);
    Tensor gv;
    g.forward(v0, gv);
    for (int i = 0; i < gv.size(); ++i)
        expect_near(gv[i], 0.0, 0.0, 0.0, "g0_fresh[" + std::to_string(i) + "]");

    // ── neutral identity is STRUCTURAL: randomise both weights, g(0) still 0 ─
    fill_random(g.W1, rng, 0.7f);
    fill_random(g.W2, rng, 0.7f);
    g.forward(v0, gv);
    for (int i = 0; i < gv.size(); ++i)
        expect_near(gv[i], 0.0, 0.0, 0.0, "g0_random[" + std::to_string(i) + "]");

    // ── backward finite-difference (non-zero weights, non-zero v) ───────────
    std::vector<float> vh = {0.2f, -0.5f, 0.7f};
    Tensor v = Tensor::from_host_on(Device::CPU, vh.data(), 3, 1);
    Tensor target = Tensor::mat(gate_dim, 1);
    fill_random(target, rng, 0.5f);

    auto loss_of = [&](const Tensor& gout) {
        double s = 0.0;
        for (int i = 0; i < gout.size(); ++i) { double d = gout[i] - target[i]; s += 0.5 * d * d; }
        return s;
    };

    g.zero_grad();
    Tensor gout;
    g.forward(v, gout);
    Tensor dG = Tensor::mat(gate_dim, 1);
    for (int i = 0; i < dG.size(); ++i) dG[i] = gout[i] - target[i];
    g.backward(v, dG);

    const float eps = 1e-3f;
    auto fd_check = [&](Tensor& P, const Tensor& analytic, const char* name) {
        for (int i = 0; i < P.size(); ++i) {
            float saved = P[i];
            P[i] = saved + eps; Tensor gp; g.forward(v, gp); double lp = loss_of(gp);
            P[i] = saved - eps; Tensor gm; g.forward(v, gm); double lm = loss_of(gm);
            P[i] = saved;
            double fd = (lp - lm) / (2.0 * eps);
            expect_near(analytic[i], fd, 2e-3, 3e-3,
                        std::string(name) + "[" + std::to_string(i) + "]");
        }
    };
    fd_check(g.W1, g.dW1, "dW1");
    fd_check(g.W2, g.dW2, "dW2");

    if (g_failures) { std::printf("test_decoder_lora: %d failure(s)\n", g_failures); return 1; }
    std::printf("test_decoder_lora: OK\n");
    return 0;
}
