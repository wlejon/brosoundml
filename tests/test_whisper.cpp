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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
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

// ─── Synthetic decoder fixture (captured-vs-eager step test) ───────────────
//
// A tiny but complete model.decoder.* checkpoint, mirroring the generator in
// test_whisper_decoder.cpp: random projections + biases, identity LayerNorms,
// and NO k_proj biases on disk (the Whisper quirk load_from must zero-fill).
namespace stepstub {

namespace stf = brotensor::safetensors;

struct StubBuffers {
    std::vector<std::vector<float>> bufs;
    std::vector<stf::WriteEntry>    entries;

    void add(const std::string& name, const std::vector<std::int64_t>& shape,
             std::vector<float>&& payload) {
        bufs.push_back(std::move(payload));
        stf::WriteEntry e;
        e.name      = name;
        e.dtype     = stf::Dtype::F32;
        e.shape     = shape;
        e.host_data = bufs.back().data();
        e.bytes     = bufs.back().size() * sizeof(float);
        entries.push_back(std::move(e));
    }
};

static std::vector<float> rand_vec(std::mt19937& rng, std::size_t n,
                                   float scale = 0.05f) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

static void write_stub_decoder(const fs::path& path,
                               int d_model, int max_tgt,
                               int n_layers, int ffn, int vocab) {
    std::mt19937 rng(20260611);
    StubBuffers sb;
    const std::string p = "model.decoder.";
    auto ones  = [](std::size_t n) { return std::vector<float>(n, 1.0f); };
    auto zeros = [](std::size_t n) { return std::vector<float>(n, 0.0f); };
    const std::size_t dd = static_cast<std::size_t>(d_model) * d_model;

    sb.add(p + "embed_tokens.weight", {vocab, d_model},
           rand_vec(rng, static_cast<std::size_t>(vocab) * d_model));
    sb.add(p + "embed_positions.weight", {max_tgt, d_model},
           rand_vec(rng, static_cast<std::size_t>(max_tgt) * d_model, 0.02f));

    for (int i = 0; i < n_layers; ++i) {
        const std::string lp = p + "layers." + std::to_string(i) + ".";
        sb.add(lp + "self_attn_layer_norm.weight", {d_model}, ones(d_model));
        sb.add(lp + "self_attn_layer_norm.bias",   {d_model}, zeros(d_model));
        sb.add(lp + "self_attn.q_proj.weight", {d_model, d_model}, rand_vec(rng, dd));
        sb.add(lp + "self_attn.q_proj.bias",   {d_model}, rand_vec(rng, d_model));
        sb.add(lp + "self_attn.k_proj.weight", {d_model, d_model}, rand_vec(rng, dd));
        // NB: NO self_attn.k_proj.bias on disk — load_from zero-fills it.
        sb.add(lp + "self_attn.v_proj.weight", {d_model, d_model}, rand_vec(rng, dd));
        sb.add(lp + "self_attn.v_proj.bias",   {d_model}, rand_vec(rng, d_model));
        sb.add(lp + "self_attn.out_proj.weight", {d_model, d_model}, rand_vec(rng, dd));
        sb.add(lp + "self_attn.out_proj.bias",   {d_model}, rand_vec(rng, d_model));

        sb.add(lp + "encoder_attn_layer_norm.weight", {d_model}, ones(d_model));
        sb.add(lp + "encoder_attn_layer_norm.bias",   {d_model}, zeros(d_model));
        sb.add(lp + "encoder_attn.q_proj.weight", {d_model, d_model}, rand_vec(rng, dd));
        sb.add(lp + "encoder_attn.q_proj.bias",   {d_model}, rand_vec(rng, d_model));
        sb.add(lp + "encoder_attn.k_proj.weight", {d_model, d_model}, rand_vec(rng, dd));
        // NB: NO encoder_attn.k_proj.bias on disk.
        sb.add(lp + "encoder_attn.v_proj.weight", {d_model, d_model}, rand_vec(rng, dd));
        sb.add(lp + "encoder_attn.v_proj.bias",   {d_model}, rand_vec(rng, d_model));
        sb.add(lp + "encoder_attn.out_proj.weight", {d_model, d_model}, rand_vec(rng, dd));
        sb.add(lp + "encoder_attn.out_proj.bias",   {d_model}, rand_vec(rng, d_model));

        sb.add(lp + "final_layer_norm.weight", {d_model}, ones(d_model));
        sb.add(lp + "final_layer_norm.bias",   {d_model}, zeros(d_model));
        sb.add(lp + "fc1.weight", {ffn, d_model},
               rand_vec(rng, static_cast<std::size_t>(ffn) * d_model));
        sb.add(lp + "fc1.bias",   {ffn}, rand_vec(rng, ffn));
        sb.add(lp + "fc2.weight", {d_model, ffn},
               rand_vec(rng, static_cast<std::size_t>(d_model) * ffn));
        sb.add(lp + "fc2.bias",   {d_model}, rand_vec(rng, d_model));
    }

    sb.add(p + "layer_norm.weight", {d_model}, ones(d_model));
    sb.add(p + "layer_norm.bias",   {d_model}, zeros(d_model));
    stf::write_file(path.string(), sb.entries);
}

}  // namespace stepstub

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

    // ─── Captured vs eager decode step (structural, synthetic weights) ─────
    //
    // Drives the DECODER directly (no encoder needed — a synthetic
    // (max_src, d_model) "encoder hidden" feeds prime_cross): prefill 4
    // prompt tokens eagerly, then generate 8 fixed steps twice over identical
    // inputs — run A through the eager forward(), run B through the captured
    // session (step_begin / step_mask_prefill / step_decode) — and compare
    // the FULL (1, vocab) logits of every step bit-exactly. On CPU (or with
    // BROSOUNDML_DISABLE_STEP_GRAPH) step_begin returns false and run B falls
    // back to the eager path, so the case degrades to eager == eager.
    {
        namespace bt = brotensor;
        const int D = 64, NL = 2, NH = 4, V = 128, FFN = 128;
        const int MAX_TGT = 32, MAX_SRC = 16;
        const fs::path stub_path =
            fs::temp_directory_path() / "brosoundml_whisper_stepgraph.safetensors";
        stepstub::write_stub_decoder(stub_path, D, MAX_TGT, NL, FFN, V);

        auto fetch = [](const bt::Tensor& t) {
            std::vector<float> out(static_cast<std::size_t>(t.rows) * t.cols);
            if (t.device == bt::Device::CPU) {
                std::memcpy(out.data(), t.host_f32(), out.size() * sizeof(float));
            } else {
                bt::Tensor h = t.to(bt::Device::CPU);
                std::memcpy(out.data(), h.host_f32(), out.size() * sizeof(float));
            }
            return out;
        };

        auto run_case = [&](bt::Device dev, const char* dev_name) {
            auto f = bt::safetensors::File::open(stub_path.string());
            brosoundml::WhisperDecoder dec;
            dec.load_from(f, D, NL, FFN, NH, V, MAX_TGT, MAX_SRC, dev);
            brosoundml::WhisperKVCache cache;
            cache.allocate(NL, D, MAX_TGT, MAX_SRC, dev);

            std::mt19937 rng(424242);
            std::vector<float> henc = stepstub::rand_vec(
                rng, static_cast<std::size_t>(MAX_SRC) * D, 0.5f);
            bt::Tensor enc_hidden =
                bt::Tensor::from_host_on(dev, henc.data(), MAX_SRC, D);

            const std::vector<int32_t> prompt = {1, 2, 3, 4};
            std::vector<int32_t> steps(8);
            for (int i = 0; i < 8; ++i) steps[i] = (7 * i + 3) % V;

            // ── run A: eager forward() per token ──
            std::vector<std::vector<float>> eager;
            {
                cache.reset();
                dec.prime_cross(enc_hidden, cache);
                bt::Tensor logits = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
                dec.forward(prompt.data(), static_cast<int>(prompt.size()),
                            /*pos_offset=*/0, cache, logits);
                for (int i = 0; i < 8; ++i) {
                    dec.forward(&steps[static_cast<std::size_t>(i)], 1,
                                cache.size(), cache, logits);   // (1, V)
                    eager.push_back(fetch(logits));
                }
            }

            // ── run B: captured session over the same inputs ──
            std::vector<std::vector<float>> captured;
            bool used_graph = false;
            {
                cache.reset();
                dec.prime_cross(enc_hidden, cache);
                bt::Tensor logits = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
                dec.forward(prompt.data(), static_cast<int>(prompt.size()),
                            /*pos_offset=*/0, cache, logits);
                used_graph = dec.step_begin(cache);
                if (used_graph) {
                    dec.step_mask_prefill(static_cast<int>(prompt.size()));
                }
                for (int i = 0; i < 8; ++i) {
                    if (used_graph) {
                        dec.step_decode(steps[static_cast<std::size_t>(i)],
                                        cache.size(), cache, logits);
                    } else {
                        dec.forward(&steps[static_cast<std::size_t>(i)], 1,
                                    cache.size(), cache, logits);
                    }
                    captured.push_back(fetch(logits));
                }
            }

            if (dev == bt::Device::CUDA) {
                CHECK(used_graph,
                      "step_begin returns true on a CUDA-resident decoder");
            } else {
                CHECK(!used_graph,
                      "step_begin returns false off CUDA (eager fallback)");
            }

            int mismatched = -1;
            for (int i = 0; i < 8; ++i) {
                const auto& a = eager[static_cast<std::size_t>(i)];
                const auto& b = captured[static_cast<std::size_t>(i)];
                if (a.size() != static_cast<std::size_t>(V) ||
                    b.size() != static_cast<std::size_t>(V) ||
                    std::memcmp(a.data(), b.data(),
                                a.size() * sizeof(float)) != 0) {
                    mismatched = i;
                    break;
                }
            }
            if (mismatched >= 0) {
                std::fprintf(stderr,
                             "FAIL: [%s] captured decode step %d logits are "
                             "not bit-identical to eager\n",
                             dev_name, mismatched);
                ++failures;
            } else {
                std::printf("    [%s] captured-vs-eager step logits "
                            "bit-identical over 8 steps (graph %s)\n",
                            dev_name, used_graph ? "captured" : "fallback");
            }
        };

        run_case(brotensor::Device::CPU, "CPU");
        if (brotensor::is_available(brotensor::Device::CUDA)) {
            run_case(brotensor::Device::CUDA, "CUDA");
        }
        fs::remove(stub_path);
    }

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
        // Warm repeat (GPU only): the first transcribe pays one-time costs —
        // on CUDA the captured decode session's warm-up + graph instantiate —
        // so a second identical run shows the steady-state rate the long-form
        // (multi-window) path actually sees.
        if (dev != brotensor::Device::CPU) {
            const auto w0 = std::chrono::steady_clock::now();
            auto warm = real.transcribe(audio, prompt);
            const auto w1 = std::chrono::steady_clock::now();
            const double wsecs = std::chrono::duration<double>(w1 - w0).count();
            const std::size_t wtoks =
                warm.token_ids.size() > prompt.size()
                    ? warm.token_ids.size() - prompt.size() : 0;
            std::printf("    [%s] transcribe (warm): %.2f s (%zu tokens, %.1f tok/s)\n",
                        dev_name, wsecs, wtoks,
                        wsecs > 0.0 ? static_cast<double>(wtoks) / wsecs : 0.0);

            // Decode-only rate: a 1-token run pays the same fixed costs
            // (mel + encoder + prefill) as the full run, so the delta divided
            // by the extra tokens isolates the per-step decode-loop rate that
            // the whole-pipeline tok/s above dilutes on short clips.
            Whisper::TranscribeOptions one_opts;
            one_opts.max_new_tokens = 1;
            const auto d0 = std::chrono::steady_clock::now();
            auto one = real.transcribe(audio, prompt, one_opts);
            const auto d1 = std::chrono::steady_clock::now();
            const double one_secs = std::chrono::duration<double>(d1 - d0).count();
            const std::size_t one_toks =
                one.token_ids.size() > prompt.size()
                    ? one.token_ids.size() - prompt.size() : 0;
            if (wtoks > one_toks && wsecs > one_secs) {
                const double dt = wsecs - one_secs;
                const double dn = static_cast<double>(wtoks - one_toks);
                std::printf("    [%s] decode loop: %.1f tok/s (%.2f ms/step)\n",
                            dev_name, dn / dt, 1e3 * dt / dn);
            }
        }
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
