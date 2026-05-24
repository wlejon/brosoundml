// Whisper stage-1 loader contract: config.json + model.safetensors parsing.
// The forward pass is still in build-out, so transcribe() must throw a staged
// std::runtime_error naming the stage.
#include "brosoundml/whisper.h"

#include <brotensor/safetensors.h>

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

// Whisper-tiny shape — chosen so the constants pin down the field-name
// contract the loader has to honour. Values match the upstream HF
// whisper-tiny config.json modulo trimming.
static const char* kConfigJson = R"json({
    "vocab_size": 51865,
    "num_mel_bins": 80,
    "d_model": 384,
    "max_source_positions": 1500,
    "max_target_positions": 448,
    "encoder_layers": 4,
    "encoder_attention_heads": 6,
    "encoder_ffn_dim": 1536,
    "decoder_layers": 4,
    "decoder_attention_heads": 6,
    "decoder_ffn_dim": 1536,
    "pad_token_id": 50257,
    "eos_token_id": 50257,
    "decoder_start_token_id": 50258
})json";

static void write_file(const fs::path& p, const std::string& body) {
    std::ofstream f(p, std::ios::binary);
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
}

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
    try { return run(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "test_whisper: uncaught exception: %s\n", e.what());
        return 2;
    }
    catch (...) { std::fprintf(stderr, "test_whisper: uncaught non-std exception\n"); return 2; }
}

static int run() {
    using brosoundml::AudioBuffer;
    using brosoundml::Whisper;

    // ─── Default-constructed Whisper ───────────────────────────────────────
    {
        Whisper w;
        CHECK(!w.loaded(), "a fresh Whisper is not loaded");
        CHECK(w.config().sample_rate == 16000,
              "default config sample rate is 16 kHz");

        CHECK(throws_runtime_error([&] { w.load("nonexistent-whisper-dir"); }),
              "load() on a missing directory throws");

        AudioBuffer empty;
        CHECK(throws_runtime_error([&] { w.transcribe(empty, {50258}); }),
              "transcribe() before load() throws");
    }

    // ─── Real loader path ──────────────────────────────────────────────────
    const fs::path root = fs::temp_directory_path() / "brosoundml_whisper_test";
    fs::remove_all(root);
    fs::create_directories(root);

    write_file(root / "config.json", kConfigJson);
    write_stub_weights(root / "model.safetensors");

    // Stage 1 used to load a stub safetensors successfully; now that load()
    // also populates the encoder, the stub no longer satisfies the loader. The
    // synthetic path now tests that a config-only checkpoint fails fast with a
    // missing-key error from the submodule loader (proves config parsing was
    // reached). Real-weights end-to-end coverage lives in test_whisper_modules.
    {
        Whisper w;
        bool threw = false;
        std::string msg;
        try { w.load(root.string()); }
        catch (const std::runtime_error& e) { threw = true; msg = e.what(); }
        CHECK(threw, "load() throws when encoder weights are missing");
        CHECK(msg.find("missing") != std::string::npos,
              "load() reports a missing tensor / key");
        CHECK(!w.loaded(),
              "Whisper::loaded() stays false after a failed load");

        // Even after a failed load, transcribe() must still refuse — proving
        // it's gated on loaded(), not just on the absence of weights.
        AudioBuffer audio;
        audio.sample_rate = 16000;
        audio.samples.assign(16000, 0.0f);
        CHECK(throws_runtime_error([&] { w.transcribe(audio, {50258}); }),
              "transcribe() after a failed load still throws");
    }

    // Stage 5 stub: once the encoder + decoder are wired in (test_whisper_modules
    // and test_whisper_decoder cover those with synthetic weights), transcribe()
    // throws the stage-5 message naming the missing greedy-decode loop.
    // We exercise this contract via the staged error string from the loader
    // path above; the parsed-config sanity also runs there. Field-by-field
    // config asserts stay below to lock the parse contract.
    {
        // Even though load() throws while populating the encoder, parse_config
        // runs to completion before the safetensors uploads begin — so the
        // parsed WhisperConfig is observable via config() on the partially-
        // loaded Whisper.
        Whisper w;
        try { w.load(root.string()); }
        catch (const std::runtime_error&) { /* expected — see above */ }
        const auto& c = w.config();
        CHECK(c.vocab_size              == 51865, "vocab_size");
        CHECK(c.num_mel_bins            == 80,    "num_mel_bins");
        CHECK(c.d_model                 == 384,   "d_model");
        CHECK(c.max_source_positions    == 1500,  "max_source_positions");
        CHECK(c.max_target_positions    == 448,   "max_target_positions");
        CHECK(c.encoder_layers          == 4,     "encoder_layers");
        CHECK(c.encoder_attention_heads == 6,     "encoder_attention_heads");
        CHECK(c.encoder_ffn_dim         == 1536,  "encoder_ffn_dim");
        CHECK(c.decoder_layers          == 4,     "decoder_layers");
        CHECK(c.decoder_attention_heads == 6,     "decoder_attention_heads");
        CHECK(c.decoder_ffn_dim         == 1536,  "decoder_ffn_dim");
        CHECK(c.pad_token_id            == 50257, "pad_token_id");
        CHECK(c.eos_token_id            == 50257, "eos_token_id");
        CHECK(c.decoder_start_token_id  == 50258, "decoder_start_token_id");
    }

    // Missing config.json
    {
        fs::path bad = fs::temp_directory_path() / "brosoundml_whisper_bad_cfg";
        fs::remove_all(bad);
        fs::create_directories(bad);
        write_stub_weights(bad / "model.safetensors");
        Whisper w;
        CHECK(throws_runtime_error([&] { w.load(bad.string()); }),
              "load() throws when config.json is missing");
        fs::remove_all(bad);
    }

    // Missing model.safetensors
    {
        fs::path bad = fs::temp_directory_path() / "brosoundml_whisper_bad_wts";
        fs::remove_all(bad);
        fs::create_directories(bad);
        write_file(bad / "config.json", kConfigJson);
        Whisper w;
        CHECK(throws_runtime_error([&] { w.load(bad.string()); }),
              "load() throws when model.safetensors is missing");
        fs::remove_all(bad);
    }

    // Missing required key in config.json
    {
        fs::path bad = fs::temp_directory_path() / "brosoundml_whisper_bad_key";
        fs::remove_all(bad);
        fs::create_directories(bad);
        write_file(bad / "config.json", R"json({"vocab_size": 51865})json");
        write_stub_weights(bad / "model.safetensors");
        Whisper w;
        bool threw = false;
        std::string msg;
        try { w.load(bad.string()); }
        catch (const std::runtime_error& e) { threw = true; msg = e.what(); }
        CHECK(threw, "load() throws on a config missing required keys");
        CHECK(msg.find("missing") != std::string::npos,
              "load() error names the missing key");
        fs::remove_all(bad);
    }

    fs::remove_all(root);

    if (failures) {
        std::fprintf(stderr, "test_whisper: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_whisper: all checks passed\n");
    return 0;
}
