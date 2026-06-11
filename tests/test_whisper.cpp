// Whisper stage-1 loader contract: config.json + model.safetensors parsing.
// The forward pass is still in build-out, so transcribe() must throw a staged
// std::runtime_error naming the stage.
#include "brosoundml/audio.h"
#include "brosoundml/whisper.h"
#include "brosoundml/whisper_modules.h"   // white-box CPU↔CUDA logits parity

#include <brolm/whisper_tokenizer.h>

#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cctype>
#include <chrono>
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
    brotensor::init();
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

    // Stages 5-6 are live: transcribe() runs the full pipeline. The greedy
    // loop is exercised end-to-end by test_whisper_e2e (synthetic weights);
    // here we just keep the parse-contract checks for config.json.
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

    // ─── Real-weights smoke (opt-in) ───────────────────────────────────────
    //
    // Runs only if a converted Whisper checkpoint is present under
    // <repo>/weights/whisper/. Run on CPU and (if available) CUDA. Token
    // argmax can tip differently between devices on FP noise, so the
    // filename-target substring check is CPU-only.
    //
    // `out` (optional) captures the run for the cross-device parity block
    // below: the generated token ids, the decoded transcript, and the wall
    // time of the main transcribe() call.
    struct RealRun {
        bool                 ran = false;
        std::vector<int32_t> generated;
        std::string          transcript;
        double               seconds = 0.0;
    };
    auto run_real_smoke = [&](brotensor::Device dev, const char* dev_name,
                              bool enforce_filename_target,
                              RealRun* out = nullptr) {
        const fs::path real_root  = fs::path(BROSOUNDML_REPO_DIR) / "weights" / "whisper";
        const fs::path real_model = real_root / "model.safetensors";
        const fs::path vocab_path = real_root / "vocab.json";
        const fs::path merges_path = real_root / "merges.txt";
        const fs::path test_wav   = real_root / "test_audio_en.wav";
        if (!(fs::exists(real_model) && fs::exists(vocab_path) &&
              fs::exists(merges_path))) {
            return;
        }
        Whisper real;
        real.load(real_root.string(), dev);
        if (!real.loaded()) {
            std::fprintf(stderr, "FAIL: [%s] real Whisper: loaded()\n", dev_name);
            ++failures;
            return;
        }

        auto tok = brolm::whisper::Tokenizer::load(vocab_path.string(),
                                                    merges_path.string());
        if (!fs::exists(test_wav)) return;

        AudioBuffer audio = brosoundml::read_wav(test_wav.string());
        if (audio.sample_rate != 16000) {
            std::fprintf(stderr,
                         "FAIL: [%s] real Whisper: test_audio_en.wav is 16 kHz\n",
                         dev_name);
            ++failures;
            return;
        }

        std::vector<int32_t> prompt =
            tok.build_prompt("en", "transcribe", /*with_timestamps=*/false);

        // Cancellation: an always-true cancel breaks the greedy loop on its
        // first iteration, so the result is exactly the prompt (no content
        // tokens). This is the real-weights proof that .cancel() stops the
        // decode rather than letting it run to completion.
        auto cancelled = real.transcribe(audio, prompt, 0, [] { return true; });
        if (cancelled.token_ids.size() != prompt.size()) {
            std::fprintf(stderr,
                         "FAIL: [%s] real Whisper: cancelled transcribe should "
                         "return prompt only (got %zu vs %zu)\n",
                         dev_name, cancelled.token_ids.size(), prompt.size());
            ++failures;
        }

        const auto t0 = std::chrono::steady_clock::now();
        auto result = real.transcribe(audio, prompt);
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        if (!(result.token_ids.size() > prompt.size())) {
            std::fprintf(stderr,
                         "FAIL: [%s] real Whisper: transcribe generated at "
                         "least one token\n", dev_name);
            ++failures;
            return;
        }

        std::vector<int32_t> generated(
            result.token_ids.begin() + prompt.size(),
            result.token_ids.end());
        // All ids in vocab range and the count is bounded.
        bool ids_ok = !generated.empty();
        for (auto id : generated) {
            if (id < 0 || id >= real.config().vocab_size) { ids_ok = false; break; }
        }
        if (!ids_ok) {
            std::fprintf(stderr,
                         "FAIL: [%s] real Whisper: generated ids out of vocab\n",
                         dev_name);
            ++failures;
        }
        std::string transcript = tok.decode(generated, /*skip_special=*/true);
        if (transcript.empty()) {
            std::fprintf(stderr,
                         "FAIL: [%s] real Whisper: decoded transcript is non-empty\n",
                         dev_name);
            ++failures;
        }
        std::printf("    [%s] real Whisper transcript: %s\n",
                    dev_name, transcript.c_str());
        std::printf("    [%s] transcribe: %.2f s (%zu tokens, %.1f tok/s)\n",
                    dev_name, secs, generated.size(),
                    secs > 0.0 ? static_cast<double>(generated.size()) / secs
                               : 0.0);
        if (out) {
            out->ran        = true;
            out->generated  = generated;
            out->transcript = transcript;
            out->seconds    = secs;
        }

        if (enforce_filename_target) {
            const std::string stem = test_wav.stem().string();
            const std::string marker = "test_audio_en_";
            if (stem.size() > marker.size() &&
                stem.compare(0, marker.size(), marker) == 0) {
                std::string target = stem.substr(marker.size());
                auto lower = [](std::string s) {
                    for (auto& c : s) {
                        c = static_cast<char>(std::tolower(
                            static_cast<unsigned char>(c)));
                    }
                    return s;
                };
                if (lower(transcript).find(lower(target)) == std::string::npos) {
                    std::fprintf(stderr,
                                 "FAIL: [%s] real Whisper transcript does not"
                                 " contain expected substring '%s'\n",
                                 dev_name, target.c_str());
                    ++failures;
                } else {
                    std::printf("    [%s] matched filename target '%s'\n",
                                dev_name, target.c_str());
                }
            }
        }
    };
    // CPU enforces the filename-target substring (the deterministic baseline);
    // CUDA only checks that the pipeline runs and produces a well-formed
    // transcript — token argmax may tip on FP noise.
    RealRun cpu_run, cuda_run;
    run_real_smoke(brotensor::Device::CPU, "CPU",
                   /*enforce_filename_target=*/true, &cpu_run);
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        run_real_smoke(brotensor::Device::CUDA, "CUDA",
                       /*enforce_filename_target=*/false, &cuda_run);
    }
    if (brotensor::is_available(brotensor::Device::Metal)) {
        run_real_smoke(brotensor::Device::Metal, "Metal",
                       /*enforce_filename_target=*/false);
    }

    // ─── CPU↔CUDA parity (opt-in: real weights + a CUDA device) ────────────
    //
    // Two levels, mirroring how test_qwen_asr pins its cross-device contract:
    //
    //  1. End-to-end: the greedy transcripts must agree. The decoder runs
    //     FP32 on both devices, so the only divergence source is the
    //     encoder's FP16 flash-attention core — far below greedy-argmax
    //     margins on real speech.
    //  2. White-box: encode + prompt prefill on each device and compare the
    //     last-position logits within an FP16-attention tolerance (the same
    //     5e-2 bound test_qwen_asr uses for its upstream-logits check).
    if (cpu_run.ran && cuda_run.ran) {
        std::printf("  [parity] CPU %.2f s vs CUDA %.2f s (%.1fx)\n",
                    cpu_run.seconds, cuda_run.seconds,
                    cuda_run.seconds > 0.0 ? cpu_run.seconds / cuda_run.seconds
                                           : 0.0);
        CHECK(cpu_run.transcript == cuda_run.transcript,
              "CPU and CUDA transcripts match");
        std::printf("  [parity] token streams %s (%zu vs %zu tokens)\n",
                    cpu_run.generated == cuda_run.generated ? "identical"
                                                            : "differ",
                    cpu_run.generated.size(), cuda_run.generated.size());

        // White-box prefill-logits parity over the real checkpoint.
        const fs::path real_root =
            fs::path(BROSOUNDML_REPO_DIR) / "weights" / "whisper";
        const fs::path test_wav = real_root / "test_audio_en.wav";
        Whisper cfg_probe;
        cfg_probe.load(real_root.string(), brotensor::Device::CPU);
        const brosoundml::WhisperConfig& c = cfg_probe.config();

        auto tok = brolm::whisper::Tokenizer::load(
            (real_root / "vocab.json").string(),
            (real_root / "merges.txt").string());
        const std::vector<int32_t> prompt =
            tok.build_prompt("en", "transcribe", /*with_timestamps=*/false);
        AudioBuffer audio = brosoundml::read_wav(test_wav.string());

        auto prefill_last_logits = [&](brotensor::Device dev) {
            namespace bt = brotensor;
            auto f = bt::safetensors::File::open(
                (real_root / "model.safetensors").string());
            brosoundml::LogMel mel;
            mel.build(c.num_mel_bins, dev);
            brosoundml::WhisperEncoder enc;
            enc.load_from(f, c.num_mel_bins, c.d_model, c.max_source_positions,
                          c.encoder_layers, c.encoder_ffn_dim,
                          c.encoder_attention_heads, dev);
            brosoundml::WhisperDecoder dec;
            dec.load_from(f, c.d_model, c.decoder_layers, c.decoder_ffn_dim,
                          c.decoder_attention_heads, c.vocab_size,
                          c.max_target_positions, c.max_source_positions, dev);
            brosoundml::WhisperKVCache cache;
            cache.allocate(c.decoder_layers, c.d_model,
                           c.max_target_positions, c.max_source_positions, dev);

            bt::Tensor m = bt::Tensor::empty_on(bt::Device::CPU, 0, 0,
                                                bt::Dtype::FP32);
            mel.forward(audio, m);
            bt::Tensor hidden = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
            enc.forward(m, hidden);
            dec.prime_cross(hidden, cache);
            bt::Tensor logits = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
            dec.forward(prompt.data(), static_cast<int>(prompt.size()),
                        /*pos_offset=*/0, cache, logits);

            bt::Tensor host = (logits.device == bt::Device::CPU)
                                  ? std::move(logits)
                                  : logits.to(bt::Device::CPU);
            const int V = host.cols;
            const float* last =
                host.host_f32() +
                static_cast<std::size_t>(host.rows - 1) * V;
            return std::vector<float>(last, last + V);
        };

        const std::vector<float> lc = prefill_last_logits(brotensor::Device::CPU);
        const std::vector<float> lg = prefill_last_logits(brotensor::Device::CUDA);
        CHECK(lc.size() == lg.size(), "parity: logits width matches");
        float max_d = 0.0f;
        double sum_d = 0.0;
        for (std::size_t i = 0; i < lc.size(); ++i) {
            const float d = std::fabs(lc[i] - lg[i]);
            if (d > max_d) max_d = d;
            sum_d += d;
        }
        std::printf("  [parity] prefill logits max|Δ|=%.2e  mean|Δ|=%.2e\n",
                    max_d, lc.empty() ? 0.0 : sum_d / static_cast<double>(lc.size()));
        CHECK(max_d < 5e-2f,
              "CPU vs CUDA prefill logits within FP16-attention tolerance");
    }

    if (failures) {
        std::fprintf(stderr, "test_whisper: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_whisper: all checks passed\n");
    return 0;
}
