// Kokoro stage-1 loader contract: config.json + model.safetensors parsing,
// raw-FP32 voice pack loading, and Voice::pick_for row indexing. The forward
// pass is still in build-out, so synthesize() must still throw a staged
// std::runtime_error naming the stage.
#include "brosoundml/kokoro.h"

#include <brotensor/safetensors.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
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

int main() {
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

    // Scope the loaded Kokoro so the mmap'd safetensors file is released
    // before we remove_all() the temp directory below — on Windows you
    // cannot delete a file that is still mapped.
    {
    Kokoro k;
    k.load(root.string());
    CHECK(k.loaded(), "Kokoro is loaded after a successful load()");

    const auto& cfg = k.config();
    CHECK(cfg.n_tokens == 178,                "n_token parsed");
    CHECK(cfg.hidden_dim == 512,              "hidden_dim parsed");
    CHECK(cfg.style_dim == 128,               "style_dim parsed");
    CHECK(cfg.n_layer == 3,                   "n_layer parsed");
    CHECK(cfg.n_mels == 80,                   "n_mels parsed");
    CHECK(cfg.dim_in == 64,                   "dim_in parsed");
    CHECK(cfg.max_dur == 50,                  "max_dur parsed");
    CHECK(cfg.max_conv_dim == 512,            "max_conv_dim parsed");
    CHECK(cfg.text_encoder_kernel_size == 5,  "text_encoder_kernel_size parsed");
    CHECK(cfg.sample_rate == 24000,           "sample_rate is fixed at 24 kHz");

    CHECK(cfg.decoder.upsample_initial_channel == 512, "decoder.upsample_initial_channel parsed");
    CHECK(cfg.decoder.upsample_kernel_sizes.size() == 2 &&
          cfg.decoder.upsample_kernel_sizes[0] == 20 &&
          cfg.decoder.upsample_kernel_sizes[1] == 12,
          "decoder.upsample_kernel_sizes parsed");
    CHECK(cfg.decoder.upsample_rates.size() == 2 &&
          cfg.decoder.upsample_rates[0] == 10 &&
          cfg.decoder.upsample_rates[1] == 6,
          "decoder.upsample_rates parsed");
    CHECK(cfg.decoder.resblock_kernel_sizes.size() == 3, "decoder.resblock_kernel_sizes parsed");
    CHECK(cfg.decoder.resblock_dilation_sizes.size() == 3 &&
          cfg.decoder.resblock_dilation_sizes[0].size() == 3 &&
          cfg.decoder.resblock_dilation_sizes[0][2] == 5,
          "decoder.resblock_dilation_sizes parsed (nested array)");
    CHECK(cfg.decoder.gen_istft_n_fft == 20,    "decoder.gen_istft_n_fft parsed");
    CHECK(cfg.decoder.gen_istft_hop_size == 5,  "decoder.gen_istft_hop_size parsed");

    CHECK(cfg.plbert.hidden_size == 768,              "plbert.hidden_size parsed");
    CHECK(cfg.plbert.num_attention_heads == 12,       "plbert.num_attention_heads parsed");
    CHECK(cfg.plbert.intermediate_size == 2048,       "plbert.intermediate_size parsed");
    CHECK(cfg.plbert.max_position_embeddings == 512,  "plbert.max_position_embeddings parsed");
    CHECK(cfg.plbert.num_hidden_layers == 12,         "plbert.num_hidden_layers parsed");
    CHECK(cfg.plbert.vocab_size == 178,               "plbert.vocab_size parsed");

    CHECK(cfg.vocab.size() == 3 && cfg.vocab.at("b") == 2,
          "vocab map parsed");

    // ─── Voice loading & indexing ──────────────────────────────────────────
    const int voice_dim = 2 * cfg.style_dim;  // 256
    const int rows      = 7;                  // small for the test
    const fs::path voice_path = root / "test_voice.bin";
    write_voice(voice_path, rows, voice_dim);

    Voice voice = k.load_voice(voice_path.string());
    CHECK(voice.name == "test_voice",         "voice name comes from the file stem");
    CHECK(voice.packs.rows == rows,           "voice packs row count matches the file");
    CHECK(voice.packs.cols == voice_dim,      "voice packs col count is 2*style_dim");
    CHECK(voice.packs.device == brotensor::Device::CPU,
          "voice packs land on CPU");

    // pick_for picks row n-1 (Kokoro convention).
    brotensor::Tensor row3 = voice.pick_for(3);
    CHECK(row3.rows == 1 && row3.cols == voice_dim,
          "pick_for returns a (1, voice_dim) row");
    // Row index 2 (since pick_for(3) -> row 2): each element is 2.0 + 0.01*c.
    const float* row3_data = row3.host_f32();
    CHECK(std::abs(row3_data[0] - 2.0f) < 1e-6f,
          "pick_for(3) returns row index 2 (Kokoro indexing)");
    CHECK(std::abs(row3_data[10] - (2.0f + 0.1f)) < 1e-5f,
          "pick_for column values are intact");

    CHECK(throws_runtime_error([&] { (void)voice.pick_for(0); }),
          "pick_for(0) throws (below valid range)");
    CHECK(throws_runtime_error([&] { (void)voice.pick_for(rows + 1); }),
          "pick_for above row count throws");
    CHECK(throws_runtime_error([&] {
              Voice empty;
              (void)empty.pick_for(1);
          }),
          "pick_for on an empty Voice throws");

    // A wrong file size for the configured voice_dim must reject.
    const fs::path bad_voice = root / "bad_voice.bin";
    {
        std::ofstream f(bad_voice, std::ios::binary);
        const float garbage[5] = {0, 0, 0, 0, 0};  // 20 bytes — not a multiple of 256*4
        f.write(reinterpret_cast<const char*>(garbage), sizeof(garbage));
    }
    CHECK(throws_runtime_error([&] { (void)k.load_voice(bad_voice.string()); }),
          "load_voice rejects a file whose size doesn't divide by voice_dim*4");

    // ─── synthesize() still stubbed (stages 2–5) ───────────────────────────
    CHECK(throws_runtime_error([&] { k.synthesize({1, 2, 3}, voice); }),
          "synthesize() throws while the forward pass is in build-out");

    // ─── Move semantics over the pImpl + the open safetensors file ─────────
    Kokoro moved = std::move(k);
    CHECK(moved.loaded(), "moved-to Kokoro retains loaded state");
    CHECK(moved.config().n_tokens == 178,
          "moved-to Kokoro retains config");
    }  // end of loaded-Kokoro scope — releases the mmap before remove_all().

    fs::remove_all(root);

    // ─── Real-weights smoke (opt-in) ───────────────────────────────────────
    //
    // Runs only if the converted upstream Kokoro-82M artifacts are present
    // under <repo>/weights/kokoro/. Skipped silently otherwise so CI / fresh
    // clones still pass this test. Produce the artifacts with:
    //   scripts/download-kokoro.sh && python scripts/convert-kokoro.py
    const fs::path real_root  = fs::path(BROSOUNDML_REPO_DIR) / "weights" / "kokoro";
    const fs::path real_model = real_root / "model.safetensors";
    if (fs::exists(real_model)) {
        Kokoro real;
        real.load(real_root.string());
        const auto& rcfg = real.config();
        // Kokoro-82M's published config: 178-token phoneme vocab, hidden=512,
        // style=128, plBERT 768/12/12. If the loader silently dropped fields,
        // these checks fire.
        CHECK(rcfg.n_tokens    == 178, "real Kokoro: n_tokens == 178");
        CHECK(rcfg.hidden_dim  == 512, "real Kokoro: hidden_dim == 512");
        CHECK(rcfg.style_dim   == 128, "real Kokoro: style_dim == 128");
        CHECK(rcfg.plbert.hidden_size         == 768, "real Kokoro: plbert.hidden_size");
        CHECK(rcfg.plbert.num_attention_heads == 12,  "real Kokoro: plbert.heads");
        CHECK(rcfg.plbert.num_hidden_layers   == 12,  "real Kokoro: plbert.layers");
        CHECK(rcfg.vocab.size() > 100,
              "real Kokoro: vocab map populated");

        const fs::path real_voice = real_root / "voices" / "af_heart.bin";
        if (fs::exists(real_voice)) {
            Voice v = real.load_voice(real_voice.string());
            CHECK(v.packs.rows == 510 && v.packs.cols == 256,
                  "real Kokoro voice af_heart: shape (510, 256)");
        }
    }

    if (failures == 0) {
        std::printf("test_kokoro: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_kokoro: %d check(s) failed\n", failures);
    return 1;
}
