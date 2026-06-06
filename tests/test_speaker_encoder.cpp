#define _CRT_SECURE_NO_WARNINGS  // std::getenv gate below (MSVC C4996)
// Standalone SpeakerEncoder: the always-on misuse contract, plus an opt-in
// parity smoke proving the ~18 MB artifact's x-vector is bit-identical to the
// full-checkpoint QwenTts::embed_speaker on the same clip. That bit-exactness is
// the safety argument for the artifact: anything fit against the full-load
// embedding (Kokoro's voice_bridge) keeps working unchanged.
#include "brosoundml/speaker_encoder.h"
#include "brosoundml/qwen_tts.h"
#include "brosoundml/audio.h"

#include <brotensor/runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int failures = 0;

#define CHECK(cond, msg)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::fprintf(stderr, "FAIL: %s\n", (msg));    \
            ++failures;                                   \
        }                                                 \
    } while (0)

template <typename Fn>
static bool throws_runtime_error(Fn&& fn) {
    try { fn(); }
    catch (const std::runtime_error&) { return true; }
    catch (...) { return false; }
    return false;
}

// A deterministic ~1 s 24 kHz "voice-ish" clip: a few harmonics with light
// amplitude modulation. Content is irrelevant — both encoders see the same
// samples, so the embeddings must agree exactly.
static brosoundml::AudioBuffer make_clip() {
    constexpr double kTwoPi = 6.283185307179586;
    const int sr = 24000, n = sr;  // 1 s
    brosoundml::AudioBuffer b;
    b.sample_rate = sr;
    b.samples.resize(n);
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / sr;
        double s = 0.6 * std::sin(kTwoPi * 140.0 * t)
                 + 0.3 * std::sin(kTwoPi * 280.0 * t)
                 + 0.15 * std::sin(kTwoPi * 525.0 * t);
        s *= 0.8 + 0.2 * std::sin(kTwoPi * 3.0 * t);  // 3 Hz tremolo
        b.samples[i] = static_cast<float>(0.5 * s);
    }
    return b;
}

// data-repo convention: BROSOUNDML_DATA_DIR env > ../brosoundml-data.
static fs::path data_dir() {
    if (const char* env = std::getenv("BROSOUNDML_DATA_DIR"); env && *env)
        return fs::path(env);
    return fs::path(BROSOUNDML_REPO_DIR).parent_path() / "brosoundml-data";
}

static int run() {
    using brosoundml::SpeakerEncoder;

    // ─── Always-on misuse contract ─────────────────────────────────────────
    {
        SpeakerEncoder se;
        CHECK(!se.loaded(), "a fresh SpeakerEncoder is not loaded");
        CHECK(throws_runtime_error([&] { se.load("nonexistent-speaker-encoder-dir"); }),
              "load() on a missing directory throws");
        brosoundml::AudioBuffer clip = make_clip();
        CHECK(throws_runtime_error([&] { (void)se.embed(clip); }),
              "embed() before load() throws");
    }

    // ─── Loaded contract + full-checkpoint parity (opt-in, skipped if absent) ─
    const fs::path art  = data_dir() / "qwen-tts" / "speaker-encoder";
    const fs::path base = fs::path(BROSOUNDML_REPO_DIR) / "weights" /
                          "qwen-tts" / "0.6B-Base";

    if (fs::exists(art / "model.safetensors")) {
        SpeakerEncoder se;
        se.load(art.string());
        CHECK(se.loaded(), "load() sets loaded()");
        CHECK(se.enc_dim() == 1024, "enc_dim is 1024");
        CHECK(se.sample_rate() == 24000, "sample_rate is 24 kHz");

        const brosoundml::AudioBuffer clip = make_clip();
        std::vector<float> x = se.embed(clip);
        CHECK(x.size() == static_cast<size_t>(se.enc_dim()),
              "embed() returns enc_dim floats");
        double norm = 0.0;
        for (float v : x) norm += static_cast<double>(v) * v;
        CHECK(norm > 0.0 && std::isfinite(norm), "embedding is finite and non-zero");

        // Parity vs the full Base checkpoint — bit-identical (same weights, same
        // host-side graph). Skipped when the 2.5 GB Base isn't checked out.
        if (fs::exists(base / "model.safetensors")) {
            brosoundml::QwenTts q;
            q.load(base.string());  // CPU
            std::vector<float> ref = q.embed_speaker(clip);
            CHECK(ref.size() == x.size(), "Base embed_speaker has the same width");
            float maxabs = 0.0f;
            for (size_t i = 0; i < x.size() && i < ref.size(); ++i)
                maxabs = std::max(maxabs, std::fabs(x[i] - ref[i]));
            CHECK(maxabs == 0.0f,
                  "standalone embedding is bit-identical to QwenTts::embed_speaker");
            std::fprintf(stderr, "[parity] max|Δ| vs full checkpoint = %.3g\n", maxabs);
        } else {
            std::fprintf(stderr, "[skip] Base checkpoint absent — parity check skipped\n");
        }
    } else {
        std::fprintf(stderr,
            "[skip] speaker-encoder artifact absent (%s) — run brosoundml_build_speaker_encoder\n",
            art.string().c_str());
    }

    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::fprintf(stderr, "ok\n");
    return 0;
}

int main() {
    brotensor::init();
    try { return run(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "test_speaker_encoder: uncaught exception: %s\n", e.what());
        return 2;
    }
}
