#define _CRT_SECURE_NO_WARNINGS
// YuE2 VAE tests.
//
//   1. Contract: a fresh Yue2Vae throws before load(); config after load.
//   2. Shape algebra vs the reference: required_halo() for several core sizes
//      (modeling_vae's _dependency_interval) and output_length() = 1920*T - 64.
//   3. CPU: decode + encode of the "short" case vs the fixture
//      (tests/fixtures/yue2_vae.bin, from tests/ref/gen_yue2_vae_fixture.py:
//      the yue2 package's YuE2VAE in FP32 on CUDA).
//   4. GPU (whichever backend brotensor reports): the "long" case's decode,
//      decode_tiled (the reference's tiled output) and encode; then
//      Yue2VaeStream fed in ragged pushes, whose chunks must concatenate to
//      the full decode, contiguous from sample 0.
//   5. decoder_only: encode throws, decode still matches.
//
// Real-weights sections skip (printing why) when the VAE (BROSOUNDML_YUE2_VAE,
// weights/yue2-vae, or the Hugging Face cache) or the fixture is absent.
#include "brosoundml/yue2_vae.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_device.h"

namespace fs = std::filesystem;
using namespace brosoundml;

static int failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

template <typename Fn>
static bool throws_runtime_error(Fn&& fn) {
    try { fn(); }
    catch (const std::runtime_error&) { return true; }
    catch (...) { return false; }
    return false;
}

// ─── fixture ────────────────────────────────────────────────────────────────

struct Case {
    int S = 0, T = 0, N = 0, core = 0, halo = 0;
    std::vector<float> input, latents_cm, full, tiled;
    std::vector<float> latents;   // frame-major, converted
};

struct Fixture {
    std::vector<std::pair<int, int>> halos;
    std::vector<Case> cases;
};

static bool read_fixture(const fs::path& path, Fixture& fx) {
    std::ifstream f(path.string(), std::ios::binary);
    if (!f) return false;
    auto i32 = [&]() { int32_t v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; };
    auto rd = [&](std::vector<float>& v, std::size_t n) {
        v.resize(n);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
    };
    if (i32() != 0x31455659) return false;
    const int nh = i32();
    for (int i = 0; i < nh; ++i) { const int c = i32(); const int h = i32(); fx.halos.push_back({c, h}); }
    fx.cases.resize(static_cast<std::size_t>(i32()));
    for (Case& c : fx.cases) {
        c.S = i32(); c.T = i32(); c.N = i32(); c.core = i32(); c.halo = i32();
        rd(c.input, std::size_t(2) * c.S);
        rd(c.latents_cm, std::size_t(64) * c.T);
        rd(c.full, std::size_t(2) * c.N);
        rd(c.tiled, std::size_t(2) * c.N);
        c.latents.resize(c.latents_cm.size());
        for (int t = 0; t < c.T; ++t)
            for (int ch = 0; ch < 64; ++ch)
                c.latents[std::size_t(t) * 64 + ch] = c.latents_cm[std::size_t(ch) * c.T + t];
    }
    return static_cast<bool>(f);
}

struct Diff { double max_abs = 0, mean_abs = 0, ref_max = 0; };
static Diff diff(const float* a, const float* b, std::size_t n) {
    Diff d;
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double e = std::fabs(double(a[i]) - b[i]);
        d.max_abs = std::max(d.max_abs, e);
        sum += e;
        d.ref_max = std::max(d.ref_max, std::fabs(double(b[i])));
    }
    d.mean_abs = n ? sum / double(n) : 0.0;
    return d;
}

static void report(const char* what, const Diff& d) {
    std::printf("  %-34s max|d| %.3e  mean|d| %.3e  (ref max %.3f)\n", what, d.max_abs, d.mean_abs, d.ref_max);
}

static fs::path find_vae() {
    if (const char* e = std::getenv("BROSOUNDML_YUE2_VAE"); e && *e) return e;
    const fs::path local = fs::path(BROSOUNDML_REPO_DIR) / "weights" / "yue2-vae";
    if (fs::exists(local / "model.safetensors")) return local;
    const char* home = std::getenv("USERPROFILE");
    if (!home) home = std::getenv("HOME");
    if (home) {
        const fs::path snaps = fs::path(home) / ".cache" / "huggingface" / "hub" /
                               "models--m-a-p--YuE2-Vae" / "snapshots";
        std::error_code ec;
        if (fs::is_directory(snaps, ec))
            for (const auto& d : fs::directory_iterator(snaps))
                if (fs::exists(d.path() / "model.safetensors")) return d.path();
    }
    return {};
}

// Audio tolerances: the reference is cuDNN FP32 through ~40 conv layers; the
// waveform peaks near 0.8.
static constexpr double kWaveMax = 2e-3, kWaveMean = 1e-4;
static constexpr double kLatMax = 2e-3, kLatMean = 2e-4;

static void check_decode(const char* what, const StereoAudio& a, const Case& c, const std::vector<float>& ref) {
    CHECK(a.frames == c.N && a.channels == 2 && a.sample_rate == 48000, "decode: shape");
    if (a.frames != c.N) return;
    const Diff d = diff(a.samples.data(), ref.data(), ref.size());
    report(what, d);
    CHECK(d.max_abs < kWaveMax && d.mean_abs < kWaveMean, what);
}

static void check_encode(const char* what, const Yue2Vae& vae, const Case& c) {
    StereoAudio in;
    in.frames = c.S;
    in.samples = c.input;
    int T = 0;
    const std::vector<float> z = vae.encode(in, &T);
    CHECK(T == c.T, "encode: frame count");
    if (T != c.T) return;
    const Diff d = diff(z.data(), c.latents.data(), z.size());
    report(what, d);
    CHECK(d.max_abs < kLatMax && d.mean_abs < kLatMean, what);
}

int main() {
    brotensor::init();

    // ── 1. contract ──
    {
        Yue2Vae vae;
        CHECK(!vae.loaded(), "fresh VAE is not loaded");
        float z[64] = {};
        CHECK(throws_runtime_error([&] { (void)vae.decode(z, 1); }), "decode before load throws");
        CHECK(throws_runtime_error([&] { (void)vae.required_halo(); }), "required_halo before load throws");
        CHECK(throws_runtime_error([&] { vae.load("does/not/exist", brotensor::Device::CPU); }),
              "load of a missing dir throws");
    }

    const fs::path dir = find_vae();
    if (dir.empty()) {
        std::printf("SKIP: no YuE2 VAE (BROSOUNDML_YUE2_VAE, weights/yue2-vae, HF cache)\n");
        return failures ? 1 : 0;
    }
    Fixture fx;
    const bool have_fx = read_fixture(fs::path(BROSOUNDML_REPO_DIR) / "tests" / "fixtures" / "yue2_vae.bin", fx);
    if (!have_fx) std::printf("SKIP numeric: no tests/fixtures/yue2_vae.bin (run tests/ref/gen_yue2_vae_fixture.py)\n");

    // ── 2 + 3. CPU ──
    {
        Yue2Vae vae;
        vae.load(dir.string(), brotensor::Device::CPU);
        const Yue2VaeConfig& cfg = vae.config();
        CHECK(cfg.sample_rate == 48000 && cfg.latent_dim == 64 && cfg.downsampling_ratio == 1920,
              "config fields");
        CHECK(vae.has_encoder(), "full load has the encoder");
        CHECK(vae.output_length(10) == 1920 * 10 - 64, "output_length");
        CHECK(throws_runtime_error([&] { (void)vae.decode(nullptr, 0); }), "decode of nothing throws");
        if (have_fx) {
            for (auto [core, halo] : fx.halos) {
                const int got = vae.required_halo(core);
                if (got != halo) std::printf("  required_halo(%d) = %d, reference %d\n", core, got, halo);
                CHECK(got == halo, "required_halo matches the reference");
            }
            const Case& c = fx.cases[0];
            CHECK(vae.output_length(c.T) == c.N, "output_length matches the reference");
            std::printf("CPU, short case (T=%d):\n", c.T);
            check_decode("decode", vae.decode(c.latents.data(), c.T), c, c.full);
            check_encode("encode", vae, c);
        }
    }

    // ── 4. GPU ──
    const brotensor::Device gpu = brosoundml_test::preferred_gpu();
    if (gpu == brotensor::Device::CPU) {
        std::printf("SKIP GPU: no GPU backend\n");
    } else if (have_fx) {
        Yue2Vae vae;
        vae.load(dir.string(), gpu);
        for (const Case& c : fx.cases) {
            std::printf("%s, case T=%d (core %d, halo %d):\n", brosoundml_test::device_name(gpu), c.T, c.core, c.halo);
            check_decode("decode", vae.decode(c.latents.data(), c.T), c, c.full);
            int progress = 0, total = 0;
            check_decode("decode_tiled vs reference tiled",
                         vae.decode_tiled(c.latents.data(), c.T, c.core, c.halo,
                                          [&](int d, int t) { progress = d; total = t; }),
                         c, c.tiled);
            CHECK(progress == total && total == (c.T + c.core - 1) / c.core, "decode_tiled progress");
            check_encode("encode", vae, c);

            // Ragged pushes through the stream, small core so it spans many tiles.
            std::vector<float> out(std::size_t(2) * c.N, 0.0f);
            std::int64_t next = 0;
            bool contiguous = true;
            int chunks = 0;
            Yue2VaeStream stream(vae, [&](const StereoAudio& a, std::int64_t s0) {
                contiguous = contiguous && s0 == next && s0 + a.frames <= c.N;
                if (s0 + a.frames <= c.N)
                    for (int ch = 0; ch < 2; ++ch)
                        std::copy_n(a.channel(ch), a.frames, out.data() + std::size_t(ch) * c.N + s0);
                next = s0 + a.frames;
                ++chunks;
            }, 16, 0, /*first_core_frames=*/16);
            static const int kSizes[] = {7, 13, 1, 30, 4, 25};
            int f = 0, k = 0;
            while (f < c.T) {
                const int n = std::min(c.T - f, kSizes[k++ % 6]);
                stream.push(f, n, c.latents.data() + std::size_t(f) * 64);
                f += n;
            }
            CHECK(stream.frames_decoded() < c.T, "stream holds back the tail until finish()");
            stream.finish();
            stream.finish();   // idempotent
            CHECK(contiguous, "stream chunks are contiguous from sample 0");
            CHECK(next == c.N && stream.samples_emitted() == c.N, "stream emits exactly the full length");
            CHECK(chunks == (c.T + 15) / 16, "stream: one chunk per core");
            const Diff d = diff(out.data(), c.full.data(), out.size());
            report("stream vs full", d);
            CHECK(d.max_abs < kWaveMax && d.mean_abs < kWaveMean, "stream matches the full decode");
            CHECK(throws_runtime_error([&] { stream.push(c.T, 1, c.latents.data()); }), "push after finish throws");
            stream.reset();
            CHECK(throws_runtime_error([&] { stream.push(5, 1, c.latents.data()); }), "push with a gap throws");

            // Growing tiles (default first 8, core 50): frame-by-frame pushes
            // give 8, 16, 32, 50, ...; whole NAR windows (50) give a 38-frame
            // tile on the first push. Both concatenate to the full decode.
            for (const int step : {1, 50}) {
                std::fill(out.begin(), out.end(), 0.0f);
                std::vector<int> tiles;
                std::int64_t at = 0;
                double first_ms = -1;
                auto t_push = std::chrono::steady_clock::now();
                Yue2VaeStream grow(vae, [&](const StereoAudio& a, std::int64_t s0) {
                    if (first_ms < 0)
                        first_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_push).count();
                    if (s0 == at && s0 + a.frames <= c.N)
                        for (int ch = 0; ch < 2; ++ch)
                            std::copy_n(a.channel(ch), a.frames, out.data() + std::size_t(ch) * c.N + s0);
                    at = s0 + a.frames;
                    tiles.push_back((a.frames + 1919) / 1920);
                });
                int pushes_before_audio = 0;
                for (int fr = 0; fr < c.T; fr += step) {
                    if (tiles.empty()) { ++pushes_before_audio; t_push = std::chrono::steady_clock::now(); }
                    grow.push(fr, std::min(step, c.T - fr), c.latents.data() + std::size_t(fr) * 64);
                }
                grow.finish();
                std::string sizes;
                for (int t : tiles) sizes += std::to_string(t) + " ";
                std::printf("  growing tiles, pushes of %d: %s(first tile %.1f ms, after %d push(es))\n",
                            step, sizes.c_str(), first_ms, pushes_before_audio);
                CHECK(at == c.N, "growing stream emits exactly the full length");
                const Diff g = diff(out.data(), c.full.data(), out.size());
                report("growing stream vs full", g);
                CHECK(g.max_abs < kWaveMax && g.mean_abs < kWaveMean, "growing stream matches the full decode");
                if (step == 1) {
                    CHECK(tiles.size() >= 2 && tiles[0] == 8 && tiles[1] == 16, "frame pushes: tiles start 8, 16");
                    for (std::size_t i = 1; i + 1 < tiles.size(); ++i)
                        CHECK(tiles[i] == std::min(50, 2 * tiles[i - 1]), "frame pushes: each tile doubles up to the core");
                } else if (c.T > 50) {
                    CHECK(pushes_before_audio == 1 && tiles[0] == 50 - 12, "window pushes: audio after the first window");
                }
            }
        }

        // ── 5. decoder_only ──
        Yue2Vae dec;
        dec.load(dir.string(), gpu, /*decoder_only=*/true);
        CHECK(!dec.has_encoder(), "decoder_only has no encoder");
        const Case& c = fx.cases[0];
        StereoAudio in;
        in.frames = c.S;
        in.samples = c.input;
        CHECK(throws_runtime_error([&] { int T; (void)dec.encode(in, &T); }), "decoder_only encode throws");
        check_decode("decoder_only decode", dec.decode(c.latents.data(), c.T), c, c.full);
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
