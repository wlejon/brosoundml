// M3: finite-difference check of the AE encoder TRAINING backward (weight
// gradients). The encoder is LayerNorm-heavy, so per-element FD of the stacked
// Jacobian is unreliable in fp32 — we use a directional dot-product test (the
// same methodology as test_supertonic_backward / test_bc_resnet2d_train):
//
//   L(w) = <R, encoder_forward(spec; w)>  (R a fixed random cotangent)
//   analytic directional derivative = <grad, dir>   (grad from backward(R))
//   central FD along dir = (L(w+eps*dir) - L(w-eps*dir)) / (2*eps)
//
// over a RANDOM direction across all weights at once. A small encoder keeps the
// two forward passes cheap. CPU-resident FP32.

#include "supertonic_encoder.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;

// All trainable weight tensors of the encoder, in a fixed order; the matching
// gradient tensors must be gathered in the SAME order.
static std::vector<Tensor*> gather_weights(brosoundml::SupertonicEncoder& e) {
    std::vector<Tensor*> v{ &e.conv_in.w, &e.conv_in.b };
    for (auto& b : e.blocks) {
        v.push_back(&b.dw.w);  v.push_back(&b.dw.b);
        v.push_back(&b.ln_g);  v.push_back(&b.ln_b);
        v.push_back(&b.pw1.w); v.push_back(&b.pw1.b);
        v.push_back(&b.pw2.w); v.push_back(&b.pw2.b);
    }
    v.push_back(&e.proj_out.w); v.push_back(&e.proj_out.b);
    return v;
}
static std::vector<Tensor*> gather_grads(brosoundml::SupertonicEncoderGrads& g) {
    std::vector<Tensor*> v{ &g.conv_in.w, &g.conv_in.b };
    for (auto& b : g.blocks) {
        v.push_back(&b.dw.w);  v.push_back(&b.dw.b);
        v.push_back(&b.ln_g);  v.push_back(&b.ln_b);
        v.push_back(&b.pw1.w); v.push_back(&b.pw1.b);
        v.push_back(&b.pw2.w); v.push_back(&b.pw2.b);
    }
    v.push_back(&g.proj_out.w); v.push_back(&g.proj_out.b);
    return v;
}

int main() {
    brotensor::init();
    const Device dev = Device::CPU;

    // Tiny encoder so the FD forwards are cheap.
    brosoundml::SupertonicEncoder enc;
    enc.idim = 9; enc.hidden = 6; enc.latent_dim = 3; enc.num_layers = 2;
    enc.ksz_init = 7; enc.ksz = 5; enc.intermediate = 12;
    enc.init(dev, /*seed=*/7);

    const int T = 12;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    std::vector<float> spec_h(static_cast<std::size_t>(enc.idim) * T);
    for (auto& x : spec_h) x = u(rng);
    Tensor spec = Tensor::from_host_on(dev, spec_h.data(), enc.idim, T);

    // Fixed random cotangent R (dL/d latent).
    std::vector<float> R(static_cast<std::size_t>(enc.latent_dim) * T);
    for (auto& x : R) x = u(rng);
    Tensor dLatent = Tensor::from_host_on(dev, R.data(), enc.latent_dim, T);

    // Analytic gradients.
    brosoundml::SupertonicEncoderCache cache;
    brosoundml::SupertonicEncoderGrads grads;
    grads.zero(enc);
    Tensor latent;
    enc.forward_train(spec, T, latent, cache);
    enc.backward(cache, dLatent, grads);

    // L(w) = <R, forward(spec)>.
    auto loss = [&]() {
        Tensor lat = enc.forward(spec, T);
        const std::vector<float> h = lat.to_host_vector();
        double s = 0.0;
        for (std::size_t i = 0; i < h.size(); ++i) s += static_cast<double>(h[i]) * R[i];
        return s;
    };

    std::vector<Tensor*> W = gather_weights(enc);
    std::vector<Tensor*> G = gather_grads(grads);
    if (W.size() != G.size()) { std::printf("FAILED gather size mismatch\n"); return 1; }

    // A random direction per weight tensor; analytic directional derivative.
    std::vector<std::vector<float>> dir(W.size());
    std::vector<std::vector<float>> orig(W.size());
    double analytic = 0.0;
    for (std::size_t t = 0; t < W.size(); ++t) {
        orig[t] = W[t]->to_host_vector();
        const std::vector<float> gh = G[t]->to_host_vector();
        dir[t].resize(orig[t].size());
        for (std::size_t i = 0; i < dir[t].size(); ++i) {
            dir[t][i] = u(rng);
            analytic += static_cast<double>(gh[i]) * dir[t][i];
        }
    }

    auto perturb = [&](float eps) {
        for (std::size_t t = 0; t < W.size(); ++t) {
            std::vector<float> w = orig[t];
            for (std::size_t i = 0; i < w.size(); ++i) w[i] += eps * dir[t][i];
            *W[t] = Tensor::from_host_on(dev, w.data(), W[t]->rows, W[t]->cols);
        }
    };
    auto restore = [&]() {
        for (std::size_t t = 0; t < W.size(); ++t)
            *W[t] = Tensor::from_host_on(dev, orig[t].data(), W[t]->rows, W[t]->cols);
    };

    int failures = 0;
    for (float eps : {1.0e-3f, 3.0e-4f}) {
        perturb(eps);  const double Lp = loss();
        perturb(-eps); const double Lm = loss();
        restore();
        const double fd = (Lp - Lm) / (2.0 * eps);
        const double diff = std::fabs(fd - analytic);
        const double mag  = std::fmax(std::fabs(fd), std::fabs(analytic));
        const bool ok = diff <= 5.0e-3 || diff <= 2.0e-2 * mag;
        std::printf("  eps=%.0e: analytic=%.6g  fd=%.6g  |d|=%.3g  %s\n",
                    eps, analytic, fd, diff, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    }

    if (failures == 0) { std::printf("PASS test_supertonic_encoder_backward\n"); return 0; }
    std::printf("FAILED test_supertonic_encoder_backward (%d)\n", failures);
    return 1;
}
