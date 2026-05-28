// BcResnet2d tests: forward shape/finiteness, save/load round-trip, BN-fold
// equivalence, and streaming == one-shot (single call + chunked).

#include "brosoundml/bc_resnet2d.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
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
    return d;
}

static std::vector<float> random_feats(int n_mels, int T, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    std::vector<float> v(static_cast<std::size_t>(n_mels) * T);
    for (auto& x : v) x = u(rng);
    return v;
}

static void run_device(bt::Device dev, const char* dn) {
    bsm::BcResnet2dConfig cfg;   // defaults: 40 mels, 3 stages
    bsm::BcResnet2d m = bsm::BcResnet2d::make(cfg, dev);
    m.xavier_init_weights(1234);

    CHECK(m.param_count() > 0, tag("param_count > 0", dn).c_str());
    CHECK(m.receptive_field_frames() > 0, tag("rf > 0", dn).c_str());

    const int T = 100;
    auto feats_host = random_feats(cfg.n_mels, T, 7);
    bt::Tensor feats = bt::Tensor::from_host_on(dev, feats_host.data(), cfg.n_mels, T);

    // 1. Forward shape + finiteness.
    bt::Tensor out;
    m.forward(feats, out);
    CHECK(out.rows == T && out.cols == 1, tag("forward shape (T,1)", dn).c_str());
    std::vector<float> out_host = out.to_host_vector();
    bool finite = true;
    for (float v : out_host) if (!std::isfinite(v)) finite = false;
    CHECK(finite, tag("forward outputs finite", dn).c_str());

    // 2. Save (unfused) / load round-trip → identical forward.
    const std::string path = std::string("test_bcr2d_") + dn + ".bw";
    m.save(path, /*fused=*/false);
    bsm::BcResnet2d m2 = bsm::BcResnet2d::load(path, dev);
    bt::Tensor out2; m2.forward(feats, out2);
    const float d_io = max_abs_diff(out_host, out2.to_host_vector());
    CHECK(d_io <= 1e-3f, tag("save/load forward matches", dn).c_str());

    // 3. BN-fold equivalence (identity BN at init → fold changes nothing).
    m.fuse_bn();
    bt::Tensor out_fused; m.forward(feats, out_fused);
    const float d_fuse = max_abs_diff(out_host, out_fused.to_host_vector());
    CHECK(d_fuse <= 1e-3f,
          (tag("fused forward matches unfused (diff=", dn) +
           std::to_string(d_fuse) + ")").c_str());

    // 4. Streaming == one-shot, single call over all T.
    bsm::BcResnet2d ms = bsm::BcResnet2d::load(path, dev);
    ms.reset_streaming_state();
    bt::Tensor s_out; ms.forward_streaming(feats, s_out);
    const float d_stream = max_abs_diff(out_host, s_out.to_host_vector());
    CHECK(d_stream <= 1e-3f,
          (tag("streaming(single) == one-shot (diff=", dn) +
           std::to_string(d_stream) + ")").c_str());

    // 5. Streaming == one-shot, chunked (tests cache continuity).
    ms.reset_streaming_state();
    std::vector<float> chunked;
    const int chunk = 17;   // deliberately not a divisor of T
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

    std::fprintf(stderr, "[%s] params=%d rf=%d io=%g fuse=%g stream=%g chunk=%g\n",
                 dn, m.param_count(), m.receptive_field_frames(),
                 d_io, d_fuse, d_stream, d_chunk);
    std::remove(path.c_str());
}

int main() {
    bt::init();
    run_device(bt::Device::CPU, "CPU");
    if (bt::is_available(bt::Device::CUDA)) run_device(bt::Device::CUDA, "CUDA");
    if (g_fail == 0) std::fprintf(stderr, "test_bc_resnet2d: all checks passed\n");
    else             std::fprintf(stderr, "test_bc_resnet2d: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
