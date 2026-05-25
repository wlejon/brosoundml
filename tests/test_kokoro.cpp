// Kokoro stage-1 loader contract: config.json + model.safetensors parsing,
// raw-FP32 voice pack loading, and Voice::pick_for row indexing. The forward
// pass is still in build-out, so synthesize() must still throw a staged
// std::runtime_error naming the stage.
#include "brosoundml/kokoro.h"

#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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

// A minimal but well-formed Kokoro config.json — every key the loader treats
// as required must be present, with shapes that match real Kokoro defaults so
// downstream stages can rely on the structure.
static const char* kConfigJson = R"json({
    "n_token": 178,
    "hidden_dim": 512,
    "style_dim": 128,
    "n_layer": 3,
    "n_mels": 80,
    "dim_in": 64,
    "max_dur": 50,
    "max_conv_dim": 512,
    "text_encoder_kernel_size": 5,
    "decoder": {
        "type": "istftnet",
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [20, 12],
        "upsample_rates": [10, 6],
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "gen_istft_n_fft": 20,
        "gen_istft_hop_size": 5
    },
    "plbert": {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
        "max_position_embeddings": 512,
        "num_hidden_layers": 12,
        "vocab_size": 178
    },
    "vocab": {
        "a": 1,
        "b": 2,
        "c": 3
    }
})json";

static void write_file(const fs::path& p, const std::string& body) {
    std::ofstream f(p, std::ios::binary);
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
}

static void write_voice(const fs::path& p, int rows, int voice_dim) {
    std::vector<float> data(static_cast<std::size_t>(rows) * voice_dim);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < voice_dim; ++c) {
            data[static_cast<std::size_t>(r) * voice_dim + c] =
                static_cast<float>(r) + 0.01f * static_cast<float>(c);
        }
    }
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
}

// Write a one-tensor safetensors file. The Kokoro loader doesn't (yet) verify
// any specific tensor exists; it just opens the file via brotensor::safetensors
// so the weights stay alive for later module uploads.
static void write_stub_weights(const fs::path& p) {
    std::vector<float> w(8, 0.0f);
    brotensor::safetensors::WriteEntry e;
    e.name      = "stub";
    e.dtype     = brotensor::safetensors::Dtype::F32;
    e.shape     = {2, 4};
    e.host_data = w.data();
    e.bytes     = w.size() * sizeof(float);
    brotensor::safetensors::write_file(p.string(), {e});
}

static int run();
int main() {
    brotensor::init();
    try { return run(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "test_kokoro: uncaught exception: %s\n", e.what());
        return 2;
    }
    catch (...) { std::fprintf(stderr, "test_kokoro: uncaught non-std exception\n"); return 2; }
}
static int run() {
    using brosoundml::AudioBuffer;
    using brosoundml::Kokoro;
    using brosoundml::Voice;

    // ─── Default-constructed Kokoro ────────────────────────────────────────
    {
        Kokoro k;
        CHECK(!k.loaded(), "a fresh Kokoro is not loaded");
        CHECK(k.config().sample_rate == 24000,
              "default config sample rate is 24 kHz");

        // Load on a nonexistent directory must throw a "no config.json" error.
        CHECK(throws_runtime_error([&] { k.load("nonexistent-kokoro-dir"); }),
              "load() on a missing directory throws");

        // load_voice before a successful load must refuse.
        CHECK(throws_runtime_error([&] { k.load_voice("anything.bin"); }),
              "load_voice() before load() throws");
    }

    // ─── Real loader path ──────────────────────────────────────────────────
    const fs::path root = fs::temp_directory_path() / "brosoundml_kokoro_test";
    fs::remove_all(root);
    fs::create_directories(root);

    write_file(root / "config.json", kConfigJson);
    write_stub_weights(root / "model.safetensors");

    // Stage 1 used to load a stub safetensors successfully; now that load()
    // also populates every submodule, the stub no longer satisfies the loader.
    // The synthetic path now tests that a config-only checkpoint fails fast
    // with a missing-key error from the submodule loader (proves config
    // parsing was reached). Full Kokoro-load semantics are exercised by the
    // real-weights smoke below.
    {
        Kokoro k;
        bool threw = false;
        std::string msg;
        try { k.load(root.string()); }
        catch (const std::runtime_error& e) { threw = true; msg = e.what(); }
        CHECK(threw, "load() throws when submodule weights are missing");
        CHECK(msg.find("missing") != std::string::npos,
              "load() reports a missing tensor / key");
        CHECK(!k.loaded(),
              "Kokoro::loaded() stays false after a failed load");
    }

    fs::remove_all(root);

    // ─── Real-weights smoke (opt-in) ───────────────────────────────────────
    //
    // Runs on CPU and (if available) CUDA. Synthesized audio is not required
    // to be sample-identical between devices — only that each device produces
    // bounded, finite, non-silent audio on the same input.
    auto run_real_smoke = [&](brotensor::Device dev, const char* dev_name) {
    const fs::path real_root  = fs::path(BROSOUNDML_REPO_DIR) / "weights" / "kokoro";
    const fs::path real_model = real_root / "model.safetensors";
    if (fs::exists(real_model)) {
        Kokoro real;
        real.load(real_root.string(), dev);
        const auto& rcfg = real.config();
        auto tag = [&](const char* s) {
            static std::string buf;
            buf = "["; buf += dev_name; buf += "] "; buf += s;
            return buf.c_str();
        };
        CHECK(rcfg.n_tokens    == 178, tag("real Kokoro: n_tokens == 178"));
        CHECK(rcfg.hidden_dim  == 512, tag("real Kokoro: hidden_dim == 512"));
        CHECK(rcfg.style_dim   == 128, tag("real Kokoro: style_dim == 128"));
        CHECK(rcfg.plbert.hidden_size         == 768, tag("real Kokoro: plbert.hidden_size"));
        CHECK(rcfg.plbert.num_attention_heads == 12,  tag("real Kokoro: plbert.heads"));
        CHECK(rcfg.plbert.num_hidden_layers   == 12,  tag("real Kokoro: plbert.layers"));
        CHECK(rcfg.vocab.size() > 100,
              tag("real Kokoro: vocab map populated"));

        const fs::path real_voice = real_root / "voices" / "af_heart.bin";
        if (fs::exists(real_voice)) {
            Voice v = real.load_voice(real_voice.string());
            CHECK(v.packs.rows == 510 && v.packs.cols == 256,
                  tag("real Kokoro voice af_heart: shape (510, 256)"));

            const std::vector<int32_t> phonemes = {50, 47, 54, 54, 57};
            AudioBuffer audio = real.synthesize(phonemes, v, 1.0f);
            CHECK(audio.sample_rate == 24000, tag("synthesize: 24 kHz output"));
            CHECK(audio.samples.size() >= 1000,
                  tag("synthesize: nontrivial sample count"));

            bool finite = true;
            float max_abs = 0.0f;
            for (float s : audio.samples) {
                if (!std::isfinite(s)) { finite = false; break; }
                if (std::abs(s) > max_abs) max_abs = std::abs(s);
            }
            CHECK(finite, tag("synthesize: all output samples are finite"));
            CHECK(max_abs > 1e-4f, tag("synthesize: output is not silent"));
            CHECK(max_abs < 10.0f, tag("synthesize: output is bounded"));
            std::printf("    [%s] synthesize: %zu samples, peak |x| = %.4f\n",
                        dev_name, audio.samples.size(), max_abs);

            // Only save the CPU output (deterministic baseline) to avoid
            // overwriting it with the device-variant version.
            if (dev == brotensor::Device::CPU) {
                const fs::path out_wav = real_root / "synth_hello.wav";
                audio.write_wav(out_wav.string());
                std::printf("    synthesize: wrote %s\n",
                            out_wav.string().c_str());
            }
        }
    }
    };  // run_real_smoke

    run_real_smoke(brotensor::Device::CPU, "CPU");
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        run_real_smoke(brotensor::Device::CUDA, "CUDA");
    }

    if (failures == 0) {
        std::printf("test_kokoro: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_kokoro: %d check(s) failed\n", failures);
    return 1;
}
