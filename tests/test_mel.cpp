// brosoundml::MelFrontend tests. Exercises the streaming log-mel front-end
// against itself — the load-bearing check is that consume()-in-chunks of an
// arbitrary signal produces frames bit-equivalent to compute_offline() on the
// same buffer (within FP32 STFT noise). The CPU baseline runs unconditionally;
// CUDA / Metal additionally run when brotensor reports them available, mirroring
// test_kokoro.cpp's run_real_smoke pattern.
#include "brosoundml/mel.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace bt = brotensor;
using brosoundml::MelConfig;
using brosoundml::MelFrontend;

static int failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

static std::string tag(const char* msg, const char* dev_name) {
    return std::string("[") + dev_name + "] " + msg;
}

static float max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b) {
    float m = 0.0f;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const float d = std::abs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// Deterministic 2-second 16 kHz signal: sum of three sines + a small DC offset
// noise carrier. Reproducible across runs and platforms.
static std::vector<float> make_test_signal(int n_samples, int sample_rate) {
    std::vector<float> s(static_cast<std::size_t>(n_samples), 0.0f);
    const double sr = static_cast<double>(sample_rate);
    for (int i = 0; i < n_samples; ++i) {
        const double t = static_cast<double>(i) / sr;
        const double v = 0.5 * std::sin(2.0 * 3.14159265358979323846 * 440.0 * t)
                       + 0.3 * std::sin(2.0 * 3.14159265358979323846 * 880.0 * t)
                       + 0.2 * std::sin(2.0 * 3.14159265358979323846 * 1760.0 * t);
        s[static_cast<std::size_t>(i)] = static_cast<float>(v);
    }
    return s;
}

// ─── 1. Mel filter shape + structure ──────────────────────────────────────
static void test_filter_shape(bt::Device dev, const char* dev_name) {
    MelConfig cfg;
    MelFrontend fe(cfg, dev);
    const bt::Tensor& fb = fe.filter();
    const int expected_bins = cfg.n_fft / 2 + 1;
    CHECK(fb.rows == cfg.n_mels,
          tag("filter rows == n_mels", dev_name).c_str());
    CHECK(fb.cols == expected_bins,
          tag("filter cols == n_fft/2+1", dev_name).c_str());

    std::vector<float> host = fb.to_host_vector();
    // Row sums all positive (every mel bin has at least one bin overlap).
    for (int m = 0; m < cfg.n_mels; ++m) {
        float s = 0.0f;
        for (int k = 0; k < expected_bins; ++k) {
            const float v = host[static_cast<std::size_t>(m) * expected_bins + k];
            CHECK(v >= 0.0f, tag("filter weights non-negative", dev_name).c_str());
            s += v;
        }
        CHECK(s > 0.0f, tag("each mel row sums positive", dev_name).c_str());
    }
    // Adjacent rows overlap: their pointwise product has at least one positive
    // bin (triangular filters share a centre/edge).
    int overlap_rows = 0;
    for (int m = 0; m + 1 < cfg.n_mels; ++m) {
        bool any = false;
        for (int k = 0; k < expected_bins; ++k) {
            const float a = host[static_cast<std::size_t>(m) * expected_bins + k];
            const float b = host[static_cast<std::size_t>(m + 1) * expected_bins + k];
            if (a > 0.0f && b > 0.0f) { any = true; break; }
        }
        if (any) ++overlap_rows;
    }
    CHECK(overlap_rows >= cfg.n_mels - 2,
          tag("adjacent mel rows overlap (banded triangular)", dev_name).c_str());
}

// ─── 2. Offline frame count matches the no-pad rule ───────────────────────
static void test_offline_shape(bt::Device dev, const char* dev_name) {
    MelConfig cfg;
    MelFrontend fe(cfg, dev);

    // Pick a clean N: N = win_length + 50 * hop_length ⇒ T = 51.
    const int N = cfg.win_length + 50 * cfg.hop_length;
    std::vector<float> s = make_test_signal(N, cfg.sample_rate);
    bt::Tensor out;
    fe.compute_offline(s.data(), N, out);
    const int expected_T = 1 + (N - cfg.win_length) / cfg.hop_length;
    CHECK(out.rows == cfg.n_mels,
          tag("offline rows == n_mels", dev_name).c_str());
    CHECK(out.cols == expected_T,
          tag("offline cols == 1 + (N - win_length)/hop_length", dev_name).c_str());

    // Sub-window input ⇒ zero frames.
    bt::Tensor empty_out;
    std::vector<float> tiny(static_cast<std::size_t>(cfg.win_length - 1), 0.0f);
    fe.compute_offline(tiny.data(), static_cast<int>(tiny.size()), empty_out);
    CHECK(empty_out.cols == 0,
          tag("sub-window input emits zero frames", dev_name).c_str());
}

// ─── 3. Streaming == offline (THE load-bearing test) ──────────────────────
static float test_streaming_equivalence(bt::Device dev, const char* dev_name) {
    MelConfig cfg;
    MelFrontend fe(cfg, dev);

    const int N = cfg.sample_rate * 2;  // 2 seconds = 32000 samples
    std::vector<float> s = make_test_signal(N, cfg.sample_rate);

    // Offline reference.
    bt::Tensor offline_out;
    fe.compute_offline(s.data(), N, offline_out);
    std::vector<float> offline_host = offline_out.to_host_vector();

    // Streaming: feed random-sized chunks.
    fe.reset();
    bt::Tensor stream_out;
    std::mt19937 rng(7);  // fixed seed per task spec
    std::uniform_int_distribution<int> chunk_dist(13, 1031);
    int cursor = 0;
    while (cursor < N) {
        int chunk = chunk_dist(rng);
        if (cursor + chunk > N) chunk = N - cursor;
        fe.consume(s.data() + cursor, chunk, stream_out);
        cursor += chunk;
    }
    std::vector<float> stream_host = stream_out.to_host_vector();

    CHECK(stream_out.rows == offline_out.rows &&
          stream_out.cols == offline_out.cols,
          tag("streaming output shape == offline shape", dev_name).c_str());

    const float diff = max_abs_diff(stream_host, offline_host);
    const bool ok = (diff <= 1e-4f);
    CHECK(ok, (tag("streaming bit-eq offline within 1e-4 (diff=", dev_name)
               + std::to_string(diff) + ")").c_str());
    return diff;
}

// ─── 4. Reset clears streaming state ──────────────────────────────────────
static void test_reset_clears_state(bt::Device dev, const char* dev_name) {
    MelConfig cfg;
    MelFrontend fe(cfg, dev);

    const int N = cfg.sample_rate;  // 1 s
    std::vector<float> s = make_test_signal(N, cfg.sample_rate);
    const int half = N / 2;

    // Feed first half (state now dirty), reset, then stream the second half.
    bt::Tensor scratch;
    fe.consume(s.data(), half, scratch);
    fe.reset();

    bt::Tensor streamed_second;
    fe.consume(s.data() + half, N - half, streamed_second);
    std::vector<float> streamed_host = streamed_second.to_host_vector();

    bt::Tensor offline_second;
    fe.compute_offline(s.data() + half, N - half, offline_second);
    std::vector<float> offline_host = offline_second.to_host_vector();

    CHECK(streamed_second.rows == offline_second.rows &&
          streamed_second.cols == offline_second.cols,
          tag("post-reset streamed shape == offline", dev_name).c_str());
    const float diff = max_abs_diff(streamed_host, offline_host);
    CHECK(diff <= 1e-4f,
          (tag("reset wipes ring (diff=", dev_name)
           + std::to_string(diff) + ")").c_str());
}

// ─── 5. Eps clamp: silence yields log(eps) finite values ──────────────────
static void test_eps_clamp(bt::Device dev, const char* dev_name) {
    MelConfig cfg;
    MelFrontend fe(cfg, dev);
    const int N = cfg.win_length + 10 * cfg.hop_length;
    std::vector<float> zeros(static_cast<std::size_t>(N), 0.0f);
    bt::Tensor out;
    fe.compute_offline(zeros.data(), N, out);
    std::vector<float> host = out.to_host_vector();
    const float expected = std::log(1e-10f);
    for (float v : host) {
        CHECK(std::isfinite(v),
              tag("silence: every output is finite", dev_name).c_str());
        CHECK(std::abs(v - expected) <= 1e-4f,
              tag("silence: output equals log(eps)", dev_name).c_str());
    }
}

// ─── 6. No NaN / Inf on a noise burst ─────────────────────────────────────
static void test_no_nan_on_noise(bt::Device dev, const char* dev_name) {
    MelConfig cfg;
    MelFrontend fe(cfg, dev);
    const int N = cfg.win_length + 100 * cfg.hop_length;
    std::vector<float> noise(static_cast<std::size_t>(N));
    std::mt19937 rng(159);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : noise) v = dist(rng);
    bt::Tensor out;
    fe.compute_offline(noise.data(), N, out);
    std::vector<float> host = out.to_host_vector();
    for (float v : host) {
        CHECK(std::isfinite(v),
              tag("noise: every output finite (no NaN/Inf)", dev_name).c_str());
    }
}

// ─── Device runner ────────────────────────────────────────────────────────
static void run_all(bt::Device dev, const char* dev_name) {
    test_filter_shape(dev, dev_name);
    test_offline_shape(dev, dev_name);
    const float stream_diff = test_streaming_equivalence(dev, dev_name);
    std::fprintf(stderr, "[%s] streaming max-abs-diff vs offline: %g\n",
                 dev_name, stream_diff);
    test_reset_clears_state(dev, dev_name);
    test_eps_clamp(dev, dev_name);
    test_no_nan_on_noise(dev, dev_name);
}

int main() {
    bt::init();
    try {
        run_all(bt::Device::CPU, "CPU");
        if (bt::is_available(bt::Device::CUDA)) {
            run_all(bt::Device::CUDA, "CUDA");
        }
        if (bt::is_available(bt::Device::Metal)) {
            run_all(bt::Device::Metal, "Metal");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
        ++failures;
    }
    if (failures == 0) {
        std::fprintf(stderr, "test_mel: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_mel: %d failure(s)\n", failures);
    return 1;
}
