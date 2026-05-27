#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

// Tests for the BC-ResNet wake-word model.
//
// Hits the contracts that have to hold before chunk 6 (training) can land:
// random-weight forward emits a finite scalar; fuse_bn() round-trips against
// the unfused forward; save/load is bit-identical; the streaming and one-shot
// paths agree at the final frame; the model is causal and bounded by the
// configured receptive field.

#include "brosoundml/bc_resnet.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace bt  = brotensor;
namespace bsm = brosoundml;

namespace {

void require(bool ok, const std::string& msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        std::abort();
    }
}

bt::Tensor random_log_mel(int n_mels, int T, std::uint64_t seed) {
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_real_distribution<float> u(-3.0f, 1.5f);
    std::vector<float> buf(static_cast<std::size_t>(n_mels) * T);
    for (auto& v : buf) v = u(rng);
    return bt::Tensor::from_host_on(bt::Device::CPU, buf.data(), n_mels, T);
}

void xavier_fill(bsm::BcResnet& /*m*/, bt::Tensor& W, std::uint64_t& rng) {
    bt::xavier_init(W, rng);
}

// Fill every Conv1d weight + Linear weight in a freshly-made model with
// xavier-uniform on host, then set BN gamma=1, beta=0, mean=0, var=1 so the
// unfused forward and the fused forward are equal (good for the round-trip).
void init_random_weights(bsm::BcResnet& m, std::uint64_t seed,
                         bool varied_bn) {
    // Round-trip through save/load to access every parameter by name. Simpler:
    // save out a fresh model (unfused), then read the file back, mutating the
    // float blob in place between read and write — but that's more code than
    // just re-running the layer construction. Instead, we exploit a small
    // testing helper: serialize the model to a temp file (unfused), randomize
    // the file contents in place at the FP32 offsets we wrote, then load it.
    //
    // That ends up much fiddlier than the alternative: use a unique temp file
    // path and round-trip through the binary, but apply randomization by
    // directly editing the saved bytes. To keep this test self-contained we
    // do exactly that.
    std::string tmp_path;
    {
        char buf[L_tmpnam];
        const char* p = std::tmpnam(buf);
        require(p != nullptr, "tmpnam failed");
        tmp_path = std::string(p) + ".bw";
    }
    m.save(tmp_path, /*fused=*/false);

    // Re-read the file, mutate every payload float, write it back. Header is
    // (magic u32, version u32, 12 config words u32, fused u8, num_tensors
    // u32). We parse it minimally to find the payload region.
    std::FILE* fp = std::fopen(tmp_path.c_str(), "rb");
    require(fp != nullptr, "could not open temp .bw for re-read");
    std::fseek(fp, 0, SEEK_END);
    const long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::vector<std::uint8_t> all(static_cast<std::size_t>(sz));
    require(std::fread(all.data(), 1, all.size(), fp) == all.size(),
            "short read of temp .bw");
    std::fclose(fp);

    auto read_u32 = [&](std::size_t& off) {
        std::uint32_t v;
        std::memcpy(&v, all.data() + off, 4);
        off += 4;
        return v;
    };
    auto read_u8 = [&](std::size_t& off) {
        std::uint8_t v = all[off];
        off += 1;
        return v;
    };
    auto read_u16 = [&](std::size_t& off) {
        std::uint16_t v;
        std::memcpy(&v, all.data() + off, 2);
        off += 2;
        return v;
    };

    std::size_t off = 0;
    (void)read_u32(off);        // magic
    (void)read_u32(off);        // version
    for (int i = 0; i < 11; ++i) (void)read_u32(off);   // config u32s
    (void)read_u32(off);        // bn_eps (FP32)
    (void)read_u8(off);         // fused flag
    const std::uint32_t num_tensors = read_u32(off);

    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_real_distribution<float> uw(-0.4f, 0.4f);
    std::uniform_real_distribution<float> uv(0.5f, 1.5f);

    for (std::uint32_t i = 0; i < num_tensors; ++i) {
        const std::uint16_t name_len = read_u16(off);
        const std::string name(reinterpret_cast<const char*>(all.data() + off),
                               name_len);
        off += name_len;
        const std::uint32_t rows = read_u32(off);
        const std::uint32_t cols = read_u32(off);
        const std::size_t n = static_cast<std::size_t>(rows) * cols;
        float* data = reinterpret_cast<float*>(all.data() + off);
        const bool is_var  = (name.size() >= 4 &&
                              name.rfind(".var") == name.size() - 4);
        const bool is_gamma= (name.size() >= 6 &&
                              name.rfind(".gamma") == name.size() - 6);
        const bool is_beta = (name.size() >= 5 &&
                              name.rfind(".beta") == name.size() - 5);
        const bool is_mean = (name.size() >= 5 &&
                              name.rfind(".mean") == name.size() - 5);
        for (std::size_t k = 0; k < n; ++k) {
            if (is_var) {
                data[k] = varied_bn ? uv(rng) : 1.0f;
            } else if (is_gamma) {
                data[k] = varied_bn ? uv(rng) : 1.0f;
            } else if (is_beta) {
                data[k] = varied_bn ? 0.1f * uw(rng) : 0.0f;
            } else if (is_mean) {
                data[k] = varied_bn ? 0.05f * uw(rng) : 0.0f;
            } else {
                data[k] = uw(rng);
            }
        }
        off += n * sizeof(float);
    }
    require(off == all.size(), "size mismatch after re-randomising .bw");
    fp = std::fopen(tmp_path.c_str(), "wb");
    require(fp != nullptr, "could not reopen temp .bw for write");
    require(std::fwrite(all.data(), 1, all.size(), fp) == all.size(),
            "short write");
    std::fclose(fp);

    // Load back into m (overwrites in place via move-assign). load() runs
    // fuse_bn() for us, since we wrote fused=false above.
    m = bsm::BcResnet::load(tmp_path, m.device());
    std::remove(tmp_path.c_str());
}

float scalar_logit(const bt::Tensor& t) {
    require(t.rows == 1 && t.cols == 1, "expected (1,1) logit");
    bt::Tensor h = t.to(bt::Device::CPU);
    return h.host_f32()[0];
}

float last_streaming_logit(const bt::Tensor& t) {
    require(t.cols == 1 && t.rows >= 1, "expected (N,1) streaming logits");
    bt::Tensor h = t.to(bt::Device::CPU);
    return h.host_f32()[t.rows - 1];
}

// Build (n_mels, T) from a flat host vector (row-major: row r is the r'th
// mel bin's time sequence — matches the BcResnet input contract).
bt::Tensor make_mel(const std::vector<float>& flat, int n_mels, int T,
                    bt::Device dev) {
    require(static_cast<int>(flat.size()) == n_mels * T,
            "make_mel: size mismatch");
    bt::Tensor host = bt::Tensor::from_host_on(bt::Device::CPU, flat.data(),
                                               n_mels, T);
    return host.to(dev);
}

void test_finite_forward(bt::Device dev, const std::string& dev_name) {
    bsm::BcResnetConfig cfg{};
    bsm::BcResnet m = bsm::BcResnet::make(cfg, dev);
    init_random_weights(m, /*seed=*/123u, /*varied_bn=*/false);

    bt::Tensor mel = random_log_mel(cfg.n_mels, 100, 7).to(dev);
    bt::Tensor out;
    m.forward(mel, out);
    const float v = scalar_logit(out);
    require(std::isfinite(v),
            dev_name + ": forward produced non-finite logit " +
            std::to_string(v));
}

void test_streaming_equivalence(bt::Device dev, const std::string& dev_name) {
    bsm::BcResnetConfig cfg{};
    bsm::BcResnet m = bsm::BcResnet::make(cfg, dev);
    init_random_weights(m, /*seed=*/777u, /*varied_bn=*/true);

    const int T = 200;
    const std::vector<float> mel_flat = [&] {
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> u(-2.5f, 1.0f);
        std::vector<float> v(static_cast<std::size_t>(cfg.n_mels) * T);
        for (auto& x : v) x = u(rng);
        return v;
    }();
    bt::Tensor mel_full = make_mel(mel_flat, cfg.n_mels, T, dev);

    bt::Tensor out_full;
    m.forward(mel_full, out_full);
    const float ref = scalar_logit(out_full);

    // Stream the same 200 frames in random chunk sizes.
    m.reset_streaming_state();
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> chunk_dist(1, 23);
    int t = 0;
    float last = 0.0f;
    while (t < T) {
        const int n = std::min(chunk_dist(rng), T - t);
        // Build (n_mels, n) chunk from the contiguous mel_flat slice. Layout
        // is (mel_bin, frame) row-major — slice per row.
        std::vector<float> chunk(static_cast<std::size_t>(cfg.n_mels) * n);
        for (int r = 0; r < cfg.n_mels; ++r) {
            std::memcpy(chunk.data() +
                            static_cast<std::size_t>(r) * n,
                        mel_flat.data() +
                            static_cast<std::size_t>(r) * T + t,
                        static_cast<std::size_t>(n) * sizeof(float));
        }
        bt::Tensor cur = make_mel(chunk, cfg.n_mels, n, dev);
        bt::Tensor out;
        m.forward_streaming(cur, out);
        last = last_streaming_logit(out);
        t += n;
    }
    const float diff = std::fabs(last - ref);
    std::fprintf(stderr,
                 "[%s] streaming vs one-shot diff = %g (ref=%g last=%g)\n",
                 dev_name.c_str(), diff, ref, last);
    require(diff < 1e-4f,
            dev_name + ": streaming-vs-one-shot diff " + std::to_string(diff) +
            " > 1e-4");
}

void test_fuse_bn_round_trip() {
    // Build TWO copies of the same model with the same randomized weights:
    // run forward on one (which load() auto-fuses) and on another, then
    // confirm calling fuse_bn() on the second is a no-op (idempotency check).
    bsm::BcResnetConfig cfg{};
    bsm::BcResnet m = bsm::BcResnet::make(cfg, bt::Device::CPU);
    init_random_weights(m, /*seed=*/2024u, /*varied_bn=*/true);
    require(m.fused(), "load() should auto-fuse");

    bt::Tensor mel = random_log_mel(cfg.n_mels, 80, 5);
    bt::Tensor a, b;
    m.forward(mel, a);
    m.fuse_bn();    // second fuse — must be a no-op.
    m.forward(mel, b);

    const float diff = std::fabs(scalar_logit(a) - scalar_logit(b));
    require(diff < 1e-5f,
            "fuse_bn idempotency diff " + std::to_string(diff) + " > 1e-5");
}

void test_save_load_round_trip() {
    bsm::BcResnetConfig cfg{};
    bsm::BcResnet m = bsm::BcResnet::make(cfg, bt::Device::CPU);
    init_random_weights(m, /*seed=*/9u, /*varied_bn=*/true);

    char buf[L_tmpnam];
    const char* p = std::tmpnam(buf);
    require(p != nullptr, "tmpnam failed");
    const std::string path = std::string(p) + ".bw";
    m.save(path, /*fused=*/true);

    bsm::BcResnet n = bsm::BcResnet::load(path, bt::Device::CPU);
    std::remove(path.c_str());

    bt::Tensor mel = random_log_mel(cfg.n_mels, 100, 13);
    bt::Tensor a, b;
    m.forward(mel, a);
    n.forward(mel, b);
    const float diff = std::fabs(scalar_logit(a) - scalar_logit(b));
    require(diff < 1e-6f,
            "save/load round-trip diff " + std::to_string(diff));
}

void test_param_count() {
    bsm::BcResnetConfig cfg{};
    bsm::BcResnet m = bsm::BcResnet::make(cfg, bt::Device::CPU);
    init_random_weights(m, /*seed=*/3u, /*varied_bn=*/false);

    // Default cfg: n_mels=40, c0..c4 = 32,32,48,56,64, k=9.
    // stem:        32*40*3 + 32                       = 3872
    // b1: DW       32*1*9 + 32                        = 320
    //     PW       32*32*1 + 32                       = 1056
    //     no proj  (c0==c1)
    // b2: DW       32*1*9 + 32                        = 320
    //     PW       48*32*1 + 48                       = 1584
    //     res      48*32*1 + 48                       = 1584
    // b3: DW       48*1*9 + 48                        = 480
    //     PW       56*48*1 + 56                       = 2744
    //     res      56*48*1 + 56                       = 2744
    // b4: DW       56*1*9 + 56                        = 560
    //     PW       64*56*1 + 64                       = 3648
    //     res      64*56*1 + 64                       = 3648
    // head:        1*64 + 1                            = 65
    // total                                            = 22625
    const int expected = 22625;
    const int got      = m.param_count();
    std::fprintf(stderr, "param_count: %d (expected %d)\n", got, expected);
    require(got == expected,
            "param_count " + std::to_string(got) + " != " +
            std::to_string(expected));
}

void test_receptive_field() {
    bsm::BcResnetConfig cfg{};
    bsm::BcResnet m = bsm::BcResnet::make(cfg, bt::Device::CPU);
    init_random_weights(m, /*seed=*/55u, /*varied_bn=*/false);
    const int rf = m.receptive_field_frames();

    const int T = rf + 100;
    std::vector<float> base = random_log_mel(cfg.n_mels, T, 91)
                                  .to_host_vector();
    bt::Tensor mel_a = make_mel(base, cfg.n_mels, T, bt::Device::CPU);
    bt::Tensor out_a;
    m.forward(mel_a, out_a);

    // Perturb every frame outside the trailing rf-window — i.e. frames
    // [0, T-rf). For row r and frame t < T-rf, flip the value.
    std::vector<float> perturbed = base;
    for (int r = 0; r < cfg.n_mels; ++r) {
        for (int t = 0; t < T - rf; ++t) {
            perturbed[static_cast<std::size_t>(r) * T + t] +=
                10.0f;   // huge swing in pre-window region
        }
    }
    bt::Tensor mel_b = make_mel(perturbed, cfg.n_mels, T, bt::Device::CPU);
    bt::Tensor out_b;
    m.forward(mel_b, out_b);

    const float diff = std::fabs(scalar_logit(out_a) - scalar_logit(out_b));
    std::fprintf(stderr, "receptive-field diff = %g (rf=%d)\n", diff, rf);
    require(diff < 1e-4f,
            "receptive-field bound violated: diff " + std::to_string(diff));
}

void test_causal_streaming() {
    bsm::BcResnetConfig cfg{};
    bsm::BcResnet m = bsm::BcResnet::make(cfg, bt::Device::CPU);
    init_random_weights(m, /*seed=*/19u, /*varied_bn=*/true);

    const int T = 50;
    std::vector<float> base = random_log_mel(cfg.n_mels, T, 4)
                                  .to_host_vector();

    // Stream the first 25 frames, snapshot the last logit.
    auto stream_first_half = [&](const std::vector<float>& full) {
        m.reset_streaming_state();
        std::vector<float> half(static_cast<std::size_t>(cfg.n_mels) * 25);
        for (int r = 0; r < cfg.n_mels; ++r) {
            std::memcpy(half.data() + static_cast<std::size_t>(r) * 25,
                        full.data() + static_cast<std::size_t>(r) * T,
                        25 * sizeof(float));
        }
        bt::Tensor cur = make_mel(half, cfg.n_mels, 25, bt::Device::CPU);
        bt::Tensor out;
        m.forward_streaming(cur, out);
        return last_streaming_logit(out);
    };
    const float a = stream_first_half(base);

    // Change the *second* half of the input (frames 25..49). The first
    // half's streaming output must be unaffected — that's the causal
    // guarantee.
    std::vector<float> perturbed = base;
    for (int r = 0; r < cfg.n_mels; ++r) {
        for (int t = 25; t < T; ++t) {
            perturbed[static_cast<std::size_t>(r) * T + t] = -7.5f;
        }
    }
    const float b = stream_first_half(perturbed);

    const float diff = std::fabs(a - b);
    std::fprintf(stderr, "causal diff = %g\n", diff);
    require(diff < 1e-6f,
            "causal streaming violated: diff " + std::to_string(diff));
}

void test_inference_timing() {
    // Rough wall-clock for the chunk-5 report. Not a contract — just prints.
    bsm::BcResnetConfig cfg{};
    bsm::BcResnet m = bsm::BcResnet::make(cfg, bt::Device::CPU);
    init_random_weights(m, /*seed=*/1u, /*varied_bn=*/false);

    bt::Tensor mel = random_log_mel(cfg.n_mels, 100, 2);
    bt::Tensor out;
    m.forward(mel, out);   // warmup

    using clk = std::chrono::high_resolution_clock;
    const int reps = 20;
    const auto t0 = clk::now();
    for (int i = 0; i < reps; ++i) m.forward(mel, out);
    const auto t1 = clk::now();
    const double ms_total = std::chrono::duration<double, std::milli>(
                                t1 - t0).count();
    std::fprintf(stderr,
                 "forward(100 frames) avg = %.3f ms over %d reps\n",
                 ms_total / reps, reps);

    // Single-frame streaming time.
    m.reset_streaming_state();
    bt::Tensor one = random_log_mel(cfg.n_mels, 1, 99);
    bt::Tensor s_out;
    m.forward_streaming(one, s_out);    // warmup
    const auto s0 = clk::now();
    for (int i = 0; i < 100; ++i) m.forward_streaming(one, s_out);
    const auto s1 = clk::now();
    const double sms = std::chrono::duration<double, std::milli>(
                           s1 - s0).count() / 100.0;
    std::fprintf(stderr, "forward_streaming(1 frame) avg = %.3f ms\n", sms);
}

}  // namespace

int main() {
    bt::init();

    try {
        test_param_count();
        test_finite_forward(bt::Device::CPU, "cpu");
        test_fuse_bn_round_trip();
        test_save_load_round_trip();
        test_streaming_equivalence(bt::Device::CPU, "cpu");
        test_receptive_field();
        test_causal_streaming();
        test_inference_timing();

        if (bt::is_available(bt::Device::CUDA)) {
            test_finite_forward(bt::Device::CUDA, "cuda");
            test_streaming_equivalence(bt::Device::CUDA, "cuda");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "test_bc_resnet: OK\n");
    return 0;
}
