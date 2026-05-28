// Tests for the BcResnet2d GPU training surface:
//   • Finite-difference gradient check on a sample of parameter elements across
//     every layer type (the primary correctness gate for the hand-rolled backward).
//   • Loss strictly decreases over consecutive train_steps on a fixed batch.
//   • Overfits a tiny batch to near-zero loss.
//   • Determinism: same seed → same loss trajectory.
//   • Runs on CPU and (when available) CUDA.

#include "brosoundml/bc_resnet2d.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace bt  = brotensor;
namespace bsm = brosoundml;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fail; } } while (0)

// A small config keeps the finite-difference probes cheap while exercising every
// layer type (stem, transition + normal BC blocks, head).
static bsm::BcResnet2dConfig tiny_cfg() {
    bsm::BcResnet2dConfig c;     // 40 mels, 3 stages of {2,2,2} blocks
    return c;
}

static std::pair<bt::Tensor, bt::Tensor>
make_batch(bt::Device dev, int B, int n_mels, int T, int n_pos, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-2.0f, 1.5f);
    std::vector<float> mel(static_cast<std::size_t>(B) * n_mels * T);
    for (auto& x : mel) x = u(rng);
    std::vector<float> lab(static_cast<std::size_t>(B), 0.0f);
    for (int b = 0; b < n_pos; ++b) lab[b] = 1.0f;
    return {bt::Tensor::from_host_on(dev, mel.data(), B, n_mels * T),
            bt::Tensor::from_host_on(dev, lab.data(), B, 1)};
}

// Finite-difference gradient check. train_step(lr=0) returns the train-mode mean
// loss and populates the analytic grads without moving the parameters, so it is a
// clean FD loss probe. We compare a sample of elements spanning every layer type.
static void test_gradient_check(bt::Device dev, const char* dn) {
    auto cfg = tiny_cfg();
    auto m = bsm::BcResnet2d::make(cfg, dev);
    m.xavier_init_weights(789);

    const int B = 4, T = 24, n_mels = cfg.n_mels;
    auto [mel, lab] = make_batch(dev, B, n_mels, T, /*n_pos=*/2, /*seed=*/19);

    // Populate analytic grads at the current point.
    m.train_step(mel, lab, B, T, /*lr=*/0.0f, /*pos_weight=*/1.0f);

    // Pick representative params: one of each named layer (first element), plus the
    // head. Cover stem, both block kinds, all conv/bn roles.
    auto params = m.debug_params();
    std::vector<std::pair<std::string, int>> probes;
    for (auto& [name, sz] : params) {
        // Probe element 0 and a middle element of each tensor — cheap but broad.
        probes.push_back({name, 0});
        if (sz > 3) probes.push_back({name, sz / 2});
    }

    // Two-eps consistency: a correct analytic gradient agrees with central FD as
    // eps→0; a ReLU kink crossing makes the smaller-eps secant *worse* but with a
    // characteristic O(1) jump. We probe at two eps and accept an element if either
    // matches — a genuine backward bug would fail at both.
    auto fd_at = [&](const std::string& name, int idx, float eps) {
        const float orig = m.debug_get_param(name, idx);
        m.debug_set_param(name, idx, orig + eps);
        const float Lp = m.train_step(mel, lab, B, T, 0.0f, 1.0f);
        m.debug_set_param(name, idx, orig - eps);
        const float Lm = m.train_step(mel, lab, B, T, 0.0f, 1.0f);
        m.debug_set_param(name, idx, orig);
        return (Lp - Lm) / (2.0f * eps);
    };

    int checked = 0, passed = 0;
    float worst_rel = 0.0f; std::string worst;
    for (auto& [name, idx] : probes) {
        const float g_analytic = m.debug_grad(name, idx);
        const float g_fd1 = fd_at(name, idx, 1e-3f);
        const float g_fd2 = fd_at(name, idx, 3e-4f);
        ++checked;
        auto rel_of = [&](float fd) {
            const float denom = std::max(1e-2f, std::max(std::fabs(g_analytic), std::fabs(fd)));
            return std::fabs(g_analytic - fd) / denom;
        };
        const float rel = std::min(rel_of(g_fd1), rel_of(g_fd2));
        // Pass if the better of the two eps agrees within 3% relative, or the
        // absolute error is below FD round-off floor (tiny gradients).
        const bool ok = rel < 3e-2f ||
                        std::min(std::fabs(g_analytic - g_fd1),
                                 std::fabs(g_analytic - g_fd2)) < 5e-4f;
        if (ok) ++passed;
        else {
            std::fprintf(stderr, "  [%s] grad mismatch %s[%d]: analytic=%g fd(1e-3)=%g fd(3e-4)=%g rel=%g\n",
                         dn, name.c_str(), idx, g_analytic, g_fd1, g_fd2, rel);
        }
        if (rel > worst_rel) { worst_rel = rel; worst = name; }
    }
    std::fprintf(stderr, "[%s] grad_check: %d/%d passed, worst rel=%g (%s)\n",
                 dn, passed, checked, worst_rel, worst.c_str());
    // ReLU kinks make a few single-element secants unreliable even with correct
    // gradients; require the large majority to agree tightly.
    CHECK(passed >= static_cast<int>(0.95 * checked + 0.5),
          (std::string("[") + dn + "] >=95% of FD gradient checks pass").c_str());
}

static void test_loss_decreases(bt::Device dev, const char* dn) {
    auto cfg = tiny_cfg();
    auto m = bsm::BcResnet2d::make(cfg, dev);
    m.xavier_init_weights(123);
    const int B = 8, T = 60;
    auto [mel, lab] = make_batch(dev, B, cfg.n_mels, T, /*n_pos=*/4, /*seed=*/7);

    const float l0 = m.train_step(mel, lab, B, T, 1e-2f, 1.0f);
    float l = l0;
    for (int s = 0; s < 39; ++s) l = m.train_step(mel, lab, B, T, 1e-2f, 1.0f);
    std::fprintf(stderr, "[%s] loss_decreases: %g -> %g\n", dn, l0, l);
    CHECK(l < l0 * 0.6f, (std::string("[") + dn + "] loss drops below 0.6x in 40 steps").c_str());
}

static void test_overfit(bt::Device dev, const char* dn) {
    auto cfg = tiny_cfg();
    auto m = bsm::BcResnet2d::make(cfg, dev);
    m.xavier_init_weights(456);
    const int B = 4, T = 50;
    auto [mel, lab] = make_batch(dev, B, cfg.n_mels, T, /*n_pos=*/2, /*seed=*/11);
    float l = 0.0f;
    for (int s = 0; s < 250; ++s) l = m.train_step(mel, lab, B, T, 1e-2f, 1.0f);
    std::fprintf(stderr, "[%s] overfit final loss = %g\n", dn, l);
    CHECK(l < 5e-2f, (std::string("[") + dn + "] overfit 4 samples to <5e-2").c_str());
}

static void test_determinism(bt::Device dev, const char* dn) {
    auto cfg = tiny_cfg();
    const int B = 4, T = 50;
    auto [mel, lab] = make_batch(dev, B, cfg.n_mels, T, 2, 31);
    auto m1 = bsm::BcResnet2d::make(cfg, dev); m1.xavier_init_weights(555);
    auto m2 = bsm::BcResnet2d::make(cfg, dev); m2.xavier_init_weights(555);
    float l1 = 0, l2 = 0;
    for (int s = 0; s < 5; ++s) {
        l1 = m1.train_step(mel, lab, B, T, 1e-3f, 1.0f);
        l2 = m2.train_step(mel, lab, B, T, 1e-3f, 1.0f);
    }
    std::fprintf(stderr, "[%s] determinism |Δ|=%g\n", dn, std::fabs(l1 - l2));
    CHECK(std::fabs(l1 - l2) < 1e-5f, (std::string("[") + dn + "] same seed → same loss").c_str());
}

static void run_device(bt::Device dev, const char* dn) {
    test_gradient_check(dev, dn);
    test_loss_decreases(dev, dn);
    test_overfit(dev, dn);
    test_determinism(dev, dn);
}

int main() {
    bt::init();
    run_device(bt::Device::CPU, "CPU");
    if (bt::is_available(bt::Device::CUDA)) run_device(bt::Device::CUDA, "CUDA");
    if (g_fail == 0) std::fprintf(stderr, "test_bc_resnet2d_train: all checks passed\n");
    else             std::fprintf(stderr, "test_bc_resnet2d_train: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
