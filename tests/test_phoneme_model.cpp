// PhonemeNet tests, mirroring the two bc_resnet2d tests:
//   A. forward() ≡ forward_streaming() (single call + chunked) — proves the
//      causal trunk caches + the pointwise-in-time K-class head stream exactly.
//   B. finite-difference gradient check on the framewise softmax-CE head + the
//      shared 2D BC-ResNet trunk backward (the correctness gate).
//   + save→load round-trip (config + class map + forward output preserved).
//
// Device choice mirrors test_bc_resnet2d_train.cpp: bt::init() then run on CPU,
// and additionally on CUDA when available. The gradient check is NOT CUDA-only —
// bc_resnet2d's train test runs it on CPU first — so this test's primary path is
// CPU (build/Debug).

#include "brosoundml/phoneme_model.h"
#include "brosoundml/phoneme_data.h"

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

static std::string tag(const char* m, const char* dev) {
    return std::string("[") + dev + "] " + m;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.0f;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    if (a.size() != b.size()) d = 1e9f;   // size mismatch is a hard failure
    return d;
}

// A small hand-built K=6 class map: class 0 = silence, 1..5 own a couple ids
// each. Only num_classes drives compute; the map is carried for save/load.
static bsm::PhonemeClassMap small_classmap() {
    bsm::PhonemeClassMap cm;
    cm.num_classes = 6;
    cm.class_names = {"sil", "AA", "IY", "K", "S", "T"};
    cm.class_to_ids = {
        {0, 1},      // sil (e.g. pad + punctuation)
        {10, 11},    // AA
        {12, 13},    // IY
        {20},        // K
        {21, 22},    // S
        {23},        // T
    };
    cm.transparent_ids = {};
    cm.rebuild_inverse();
    return cm;
}

// A compact config for streaming/IO: small K head but the full trunk recipe.
static bsm::PhonemeNetConfig stream_cfg() {
    bsm::PhonemeNetConfig c;     // defaults: 40 mels, 4 stages
    return c;
}

// A cheaper trunk for the finite-difference probes (smaller channels keeps the
// per-element double-forward cost low) — still exercises every layer type.
static bsm::PhonemeNetConfig tiny_cfg() {
    bsm::PhonemeNetConfig c;
    c.c_stem = 8;
    c.c[0] = 8; c.c[1] = 12; c.c[2] = 16; c.c[3] = 24;
    return c;
}

static std::vector<float> random_feats(int n_mels, int T, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    std::vector<float> v(static_cast<std::size_t>(n_mels) * T);
    for (auto& x : v) x = u(rng);
    return v;
}

static std::pair<bt::Tensor, bt::Tensor>
make_batch(bt::Device dev, int B, int n_mels, int T, int K, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-2.0f, 1.5f);
    std::vector<float> mel(static_cast<std::size_t>(B) * n_mels * T);
    for (auto& x : mel) x = u(rng);
    std::uniform_int_distribution<int> ci(0, K - 1);
    std::vector<float> lab(static_cast<std::size_t>(B) * T);
    for (auto& x : lab) x = static_cast<float>(ci(rng));
    return {bt::Tensor::from_host_on(dev, mel.data(), B, n_mels * T),
            bt::Tensor::from_host_on(dev, lab.data(), B, T)};
}

// ── A. forward ≡ streaming, + save/load round-trip ──────────────────────────
static void test_forward_stream_io(bt::Device dev, const char* dn) {
    auto cfg = stream_cfg();
    auto cm  = small_classmap();
    auto m = bsm::PhonemeNet::make(cfg, cm, dev);
    m.xavier_init_weights(1234);
    const int K = m.config().num_classes;

    CHECK(K == cm.num_classes, tag("K == class map K", dn).c_str());
    CHECK(m.param_count() > 0, tag("param_count > 0", dn).c_str());
    CHECK(m.receptive_field_frames() > 0, tag("rf > 0", dn).c_str());

    const int T = 100;
    auto feats_host = random_feats(cfg.n_mels, T, 7);
    bt::Tensor feats = bt::Tensor::from_host_on(dev, feats_host.data(), cfg.n_mels, T);

    bt::Tensor out;
    m.forward(feats, out);
    CHECK(out.rows == T && out.cols == K, tag("forward shape (T,K)", dn).c_str());
    std::vector<float> out_host = out.to_host_vector();
    bool finite = true;
    for (float v : out_host) if (!std::isfinite(v)) finite = false;
    CHECK(finite, tag("forward outputs finite", dn).c_str());

    // save (unfused) / load round-trip.
    const std::string path = std::string("test_phoneme_model_") + dn + ".bpm";
    m.save(path, /*fused=*/false);
    bsm::PhonemeNet m2 = bsm::PhonemeNet::load(path, dev);
    CHECK(m2.config().num_classes == K, tag("load K matches", dn).c_str());
    CHECK(m2.config().sample_rate == cfg.sample_rate &&
          m2.config().hop_length == cfg.hop_length &&
          m2.config().c[3] == cfg.c[3],
          tag("load config matches", dn).c_str());
    CHECK(m2.class_map() == cm, tag("load class map matches", dn).c_str());
    bt::Tensor out2; m2.forward(feats, out2);
    const float d_io = max_abs_diff(out_host, out2.to_host_vector());
    CHECK(d_io <= 1e-3f, tag("save/load forward matches", dn).c_str());

    // Streaming == one-shot, single call over all T.
    bsm::PhonemeNet ms = bsm::PhonemeNet::load(path, dev);
    ms.reset_streaming_state();
    bt::Tensor s_out; ms.forward_streaming(feats, s_out);
    const float d_stream = max_abs_diff(out_host, s_out.to_host_vector());
    CHECK(d_stream <= 1e-3f,
          (tag("streaming(single) == one-shot (diff=", dn) +
           std::to_string(d_stream) + ")").c_str());

    // Streaming == one-shot, chunked (cache continuity).
    ms.reset_streaming_state();
    std::vector<float> chunked;
    const int chunk = 17;
    for (int start = 0; start < T; start += chunk) {
        const int w = std::min(chunk, T - start);
        std::vector<float> sub(static_cast<std::size_t>(cfg.n_mels) * w);
        for (int r = 0; r < cfg.n_mels; ++r)
            for (int t = 0; t < w; ++t)
                sub[static_cast<std::size_t>(r) * w + t] =
                    feats_host[static_cast<std::size_t>(r) * T + (start + t)];
        bt::Tensor cf = bt::Tensor::from_host_on(dev, sub.data(), cfg.n_mels, w);
        bt::Tensor co; ms.forward_streaming(cf, co);
        std::vector<float> coh = co.to_host_vector();
        chunked.insert(chunked.end(), coh.begin(), coh.end());
    }
    const float d_chunk = max_abs_diff(out_host, chunked);
    CHECK(d_chunk <= 1e-3f,
          (tag("streaming(chunked) == one-shot (diff=", dn) +
           std::to_string(d_chunk) + ")").c_str());

    std::fprintf(stderr, "[%s] params=%d rf=%d K=%d io=%g stream=%g chunk=%g\n",
                 dn, m.param_count(), m.receptive_field_frames(), K,
                 d_io, d_stream, d_chunk);
    std::remove(path.c_str());
}

// ── B. finite-difference gradient check ─────────────────────────────────────
static void test_gradient_check(bt::Device dev, const char* dn) {
    auto cfg = tiny_cfg();
    auto cm  = small_classmap();
    auto m = bsm::PhonemeNet::make(cfg, cm, dev);
    m.xavier_init_weights(789);
    const int K = m.config().num_classes;

    // A roomy batch keeps the finite-difference secants smooth: framewise-CE +
    // BN batch statistics couple parameter perturbations through downstream ReLU
    // kinks, and averaging over more frames damps that curvature (bc_resnet2d's
    // clip-pooled BCE is intrinsically smoother and gets away with B=4).
    const int B = 8, T = 28, n_mels = cfg.n_mels;
    auto [mel, lab] = make_batch(dev, B, n_mels, T, K, /*seed=*/19);
    // Non-uniform class weights (down-weight silence) — exercises the weighting.
    std::vector<float> cw(K, 1.0f); cw[0] = 0.3f;

    // Populate analytic grads at the current point (lr=0 keeps params fixed).
    m.train_step(mel, lab, B, T, /*lr=*/0.0f, cw);

    auto params = m.debug_params();
    std::vector<std::pair<std::string, int>> probes;
    for (auto& [name, sz] : params) {
        probes.push_back({name, 0});
        if (sz > 3) probes.push_back({name, sz / 2});
    }

    auto fd_at = [&](const std::string& name, int idx, float eps) {
        const float orig = m.debug_get_param(name, idx);
        m.debug_set_param(name, idx, orig + eps);
        const float Lp = m.train_step(mel, lab, B, T, 0.0f, cw);
        m.debug_set_param(name, idx, orig - eps);
        const float Lm = m.train_step(mel, lab, B, T, 0.0f, cw);
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
        const bool ok = rel < 3e-2f ||
                        std::min(std::fabs(g_analytic - g_fd1),
                                 std::fabs(g_analytic - g_fd2)) < 5e-4f;
        if (ok) ++passed;
        else
            std::fprintf(stderr, "  [%s] grad mismatch %s[%d]: analytic=%g fd(1e-3)=%g fd(3e-4)=%g rel=%g\n",
                         dn, name.c_str(), idx, g_analytic, g_fd1, g_fd2, rel);
        if (rel > worst_rel) { worst_rel = rel; worst = name; }
    }
    std::fprintf(stderr, "[%s] grad_check: %d/%d passed, worst rel=%g (%s)\n",
                 dn, passed, checked, worst_rel, worst.c_str());
    CHECK(passed >= static_cast<int>(0.95 * checked + 0.5),
          (std::string("[") + dn + "] >=95% of FD gradient checks pass").c_str());
}

// Loss should drop on a fixed batch (sanity that the trunk + head train).
static void test_loss_decreases(bt::Device dev, const char* dn) {
    auto cfg = tiny_cfg();
    auto cm  = small_classmap();
    auto m = bsm::PhonemeNet::make(cfg, cm, dev);
    m.xavier_init_weights(123);
    const int K = m.config().num_classes;
    const int B = 6, T = 40;
    auto [mel, lab] = make_batch(dev, B, cfg.n_mels, T, K, /*seed=*/7);
    std::vector<float> cw;   // uniform

    const float l0 = m.train_step(mel, lab, B, T, 1e-2f, cw);
    float l = l0;
    for (int s = 0; s < 59; ++s) l = m.train_step(mel, lab, B, T, 1e-2f, cw);
    auto ev = m.eval_step(mel, lab, B, T, cw);
    std::fprintf(stderr, "[%s] loss_decreases: %g -> %g  eval{loss=%g acc=%g ns_acc=%g n=%d}\n",
                 dn, l0, l, ev.loss, ev.frame_accuracy, ev.nonsilence_frame_accuracy, ev.n_frames);
    CHECK(l < l0, (std::string("[") + dn + "] framewise CE loss decreases").c_str());
}

// ── C. two sessions, one shared net, interleaved — zero cache crosstalk ──────
// Proves the Session extraction: a single load-once net drives two
// independent streams. Each stream's interleaved result must match its own
// standalone single-stream result bit-for-bit (within FP rounding), which can
// only hold if neither session touches the other's caches.
static void test_multistream(bt::Device dev, const char* dn) {
    auto cfg = stream_cfg();
    auto cm  = small_classmap();
    auto m = bsm::PhonemeNet::make(cfg, cm, dev);
    m.xavier_init_weights(4242);
    const bsm::PhonemeNet& net = m;   // drive the whole test through the const API
    const int nm = cfg.n_mels, T = 96;

    const auto fa = random_feats(nm, T, 101);
    const auto fb = random_feats(nm, T, 202);

    // Standalone reference per stream: its own fresh session, single forward.
    auto ref = [&](const std::vector<float>& f) {
        auto st = net.make_session();
        bt::Tensor feats = bt::Tensor::from_host_on(dev, f.data(), nm, T);
        bt::Tensor out; net.forward_streaming(st, feats, out);
        return out.to_host_vector();
    };
    const std::vector<float> refA = ref(fa);
    const std::vector<float> refB = ref(fb);

    // Interleaved: two sessions on ONE net, fed alternating chunks A,B,A,B,...
    auto stA = net.make_session();
    auto stB = net.make_session();
    std::vector<float> outA, outB;
    auto feed_chunk = [&](bsm::PhonemeSession& st, const std::vector<float>& f,
                          int start, int w, std::vector<float>& acc) {
        std::vector<float> sub(static_cast<std::size_t>(nm) * w);
        for (int r = 0; r < nm; ++r)
            for (int t = 0; t < w; ++t)
                sub[static_cast<std::size_t>(r) * w + t] =
                    f[static_cast<std::size_t>(r) * T + (start + t)];
        bt::Tensor cf = bt::Tensor::from_host_on(dev, sub.data(), nm, w);
        bt::Tensor co; net.forward_streaming(st, cf, co);
        const auto coh = co.to_host_vector();
        acc.insert(acc.end(), coh.begin(), coh.end());
    };
    const int chunk = 11;
    for (int start = 0; start < T; start += chunk) {
        const int w = std::min(chunk, T - start);
        feed_chunk(stA, fa, start, w, outA);
        feed_chunk(stB, fb, start, w, outB);
    }

    const float dA = max_abs_diff(refA, outA);
    const float dB = max_abs_diff(refB, outB);
    CHECK(dA <= 1e-3f, (tag("interleaved stream A == standalone (diff=", dn) +
                        std::to_string(dA) + ")").c_str());
    CHECK(dB <= 1e-3f, (tag("interleaved stream B == standalone (diff=", dn) +
                        std::to_string(dB) + ")").c_str());
    // Guard against a vacuous pass: the two streams must genuinely differ.
    CHECK(max_abs_diff(refA, refB) > 1e-3f,
          tag("streams A and B are distinct", dn).c_str());
    std::fprintf(stderr, "[%s] multistream: dA=%g dB=%g\n", dn, dA, dB);
}

static void run_device(bt::Device dev, const char* dn) {
    test_forward_stream_io(dev, dn);
    test_multistream(dev, dn);
    test_gradient_check(dev, dn);
    test_loss_decreases(dev, dn);
}

int main() {
    bt::init();
    run_device(bt::Device::CPU, "CPU");
    if (bt::is_available(bt::Device::CUDA)) run_device(bt::Device::CUDA, "CUDA");
    if (g_fail == 0) std::fprintf(stderr, "test_phoneme_model: all checks passed\n");
    else             std::fprintf(stderr, "test_phoneme_model: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
