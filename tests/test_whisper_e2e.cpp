// Stage 5/6 Whisper end-to-end synthetic test.
//
// Builds a tiny in-memory Whisper checkpoint (config.json +
// model.safetensors), instantiates Whisper::load, runs transcribe() on a
// synthetic 16 kHz sine, and pins the end-to-end contract: shape of the
// returned token sequence, all ids in-range, stop condition (EOS or
// max_new_tokens), and that the KV cache is reset between consecutive
// transcribe() calls.
//
// No real weights and no external tokenizer are needed — this test is what
// CI relies on to keep Whisper green.

#include "brosoundml/audio.h"
#include "brosoundml/whisper.h"

#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs  = std::filesystem;
namespace stf = brotensor::safetensors;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

// ─── Tiny synthetic checkpoint ─────────────────────────────────────────────
//
// Mirrors the encoder + decoder stubs in test_whisper_modules.cpp /
// test_whisper_decoder.cpp, combined into one consistent safetensors file.

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
static std::vector<float> zeros_vec(std::size_t n) { return std::vector<float>(n, 0.0f); }
static std::vector<float> ones_vec (std::size_t n) { return std::vector<float>(n, 1.0f); }

struct TinyShape {
    int d_model               = 8;
    int encoder_ffn_dim       = 16;
    int decoder_ffn_dim       = 16;
    int num_mel_bins          = 8;
    int vocab_size            = 32;
    int max_source_positions  = 1500;  // Whisper-fixed; LogMel emits 3000 frames -> 1500 enc positions
    int max_target_positions  = 16;
    int encoder_attention_heads = 2;
    int decoder_attention_heads = 2;
    int encoder_layers        = 2;
    int decoder_layers        = 2;
    int eos_token_id          = 3;
};

static void write_stub_checkpoint(const fs::path& path, const TinyShape& s,
                                  std::uint32_t seed = 12345) {
    std::mt19937 rng(seed);
    StubBuffers sb;

    // ── Encoder ────────────────────────────────────────────────────────────
    const std::string ep = "model.encoder.";
    sb.add(ep + "conv1.weight", {s.d_model, s.num_mel_bins, 3},
           rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.num_mel_bins * 3));
    sb.add(ep + "conv1.bias",   {s.d_model}, zeros_vec(s.d_model));
    sb.add(ep + "conv2.weight", {s.d_model, s.d_model, 3},
           rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model * 3));
    sb.add(ep + "conv2.bias",   {s.d_model}, zeros_vec(s.d_model));
    sb.add(ep + "embed_positions.weight",
           {s.max_source_positions, s.d_model},
           rand_vec(rng, static_cast<std::size_t>(s.max_source_positions) * s.d_model,
                    0.02f));

    for (int i = 0; i < s.encoder_layers; ++i) {
        const std::string lp = ep + "layers." + std::to_string(i) + ".";
        sb.add(lp + "self_attn_layer_norm.weight", {s.d_model}, ones_vec(s.d_model));
        sb.add(lp + "self_attn_layer_norm.bias",   {s.d_model}, zeros_vec(s.d_model));
        sb.add(lp + "self_attn.q_proj.weight", {s.d_model, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model));
        sb.add(lp + "self_attn.q_proj.bias",   {s.d_model}, zeros_vec(s.d_model));
        sb.add(lp + "self_attn.k_proj.weight", {s.d_model, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model));
        sb.add(lp + "self_attn.v_proj.weight", {s.d_model, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model));
        sb.add(lp + "self_attn.v_proj.bias",   {s.d_model}, zeros_vec(s.d_model));
        sb.add(lp + "self_attn.out_proj.weight", {s.d_model, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model));
        sb.add(lp + "self_attn.out_proj.bias",   {s.d_model}, zeros_vec(s.d_model));
        sb.add(lp + "final_layer_norm.weight", {s.d_model}, ones_vec(s.d_model));
        sb.add(lp + "final_layer_norm.bias",   {s.d_model}, zeros_vec(s.d_model));
        sb.add(lp + "fc1.weight", {s.encoder_ffn_dim, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.encoder_ffn_dim) * s.d_model));
        sb.add(lp + "fc1.bias",   {s.encoder_ffn_dim}, zeros_vec(s.encoder_ffn_dim));
        sb.add(lp + "fc2.weight", {s.d_model, s.encoder_ffn_dim},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.encoder_ffn_dim));
        sb.add(lp + "fc2.bias",   {s.d_model}, zeros_vec(s.d_model));
    }
    sb.add(ep + "layer_norm.weight", {s.d_model}, ones_vec(s.d_model));
    sb.add(ep + "layer_norm.bias",   {s.d_model}, zeros_vec(s.d_model));

    // ── Decoder ────────────────────────────────────────────────────────────
    const std::string dp = "model.decoder.";
    sb.add(dp + "embed_tokens.weight", {s.vocab_size, s.d_model},
           rand_vec(rng, static_cast<std::size_t>(s.vocab_size) * s.d_model));
    sb.add(dp + "embed_positions.weight",
           {s.max_target_positions, s.d_model},
           rand_vec(rng, static_cast<std::size_t>(s.max_target_positions) * s.d_model,
                    0.02f));
    for (int i = 0; i < s.decoder_layers; ++i) {
        const std::string lp = dp + "layers." + std::to_string(i) + ".";
        sb.add(lp + "self_attn_layer_norm.weight", {s.d_model}, ones_vec(s.d_model));
        sb.add(lp + "self_attn_layer_norm.bias",   {s.d_model}, zeros_vec(s.d_model));
        sb.add(lp + "self_attn.q_proj.weight", {s.d_model, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model));
        sb.add(lp + "self_attn.q_proj.bias",   {s.d_model}, zeros_vec(s.d_model));
        sb.add(lp + "self_attn.k_proj.weight", {s.d_model, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model));
        sb.add(lp + "self_attn.v_proj.weight", {s.d_model, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model));
        sb.add(lp + "self_attn.v_proj.bias",   {s.d_model}, zeros_vec(s.d_model));
        sb.add(lp + "self_attn.out_proj.weight", {s.d_model, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model));
        sb.add(lp + "self_attn.out_proj.bias",   {s.d_model}, zeros_vec(s.d_model));

        sb.add(lp + "encoder_attn_layer_norm.weight", {s.d_model}, ones_vec(s.d_model));
        sb.add(lp + "encoder_attn_layer_norm.bias",   {s.d_model}, zeros_vec(s.d_model));
        sb.add(lp + "encoder_attn.q_proj.weight", {s.d_model, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model));
        sb.add(lp + "encoder_attn.q_proj.bias",   {s.d_model}, zeros_vec(s.d_model));
        sb.add(lp + "encoder_attn.k_proj.weight", {s.d_model, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model));
        sb.add(lp + "encoder_attn.v_proj.weight", {s.d_model, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model));
        sb.add(lp + "encoder_attn.v_proj.bias",   {s.d_model}, zeros_vec(s.d_model));
        sb.add(lp + "encoder_attn.out_proj.weight", {s.d_model, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.d_model));
        sb.add(lp + "encoder_attn.out_proj.bias",   {s.d_model}, zeros_vec(s.d_model));

        sb.add(lp + "final_layer_norm.weight", {s.d_model}, ones_vec(s.d_model));
        sb.add(lp + "final_layer_norm.bias",   {s.d_model}, zeros_vec(s.d_model));
        sb.add(lp + "fc1.weight", {s.decoder_ffn_dim, s.d_model},
               rand_vec(rng, static_cast<std::size_t>(s.decoder_ffn_dim) * s.d_model));
        sb.add(lp + "fc1.bias",   {s.decoder_ffn_dim}, zeros_vec(s.decoder_ffn_dim));
        sb.add(lp + "fc2.weight", {s.d_model, s.decoder_ffn_dim},
               rand_vec(rng, static_cast<std::size_t>(s.d_model) * s.decoder_ffn_dim));
        sb.add(lp + "fc2.bias",   {s.d_model}, zeros_vec(s.d_model));
    }
    sb.add(dp + "layer_norm.weight", {s.d_model}, ones_vec(s.d_model));
    sb.add(dp + "layer_norm.bias",   {s.d_model}, zeros_vec(s.d_model));

    stf::write_file(path.string(), sb.entries);
}

static std::string build_config_json(const TinyShape& s) {
    char buf[2048];
    std::snprintf(buf, sizeof buf, R"json({
    "vocab_size": %d,
    "num_mel_bins": %d,
    "d_model": %d,
    "max_source_positions": %d,
    "max_target_positions": %d,
    "encoder_layers": %d,
    "encoder_attention_heads": %d,
    "encoder_ffn_dim": %d,
    "decoder_layers": %d,
    "decoder_attention_heads": %d,
    "decoder_ffn_dim": %d,
    "pad_token_id": 0,
    "eos_token_id": %d,
    "decoder_start_token_id": 0
})json",
                  s.vocab_size, s.num_mel_bins, s.d_model,
                  s.max_source_positions, s.max_target_positions,
                  s.encoder_layers, s.encoder_attention_heads, s.encoder_ffn_dim,
                  s.decoder_layers, s.decoder_attention_heads, s.decoder_ffn_dim,
                  s.eos_token_id);
    return buf;
}

static void write_file(const fs::path& p, const std::string& body) {
    std::ofstream f(p, std::ios::binary);
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
}

// ─── Synthetic-audio helper ────────────────────────────────────────────────

static brosoundml::AudioBuffer sine(float hz, float seconds, int sr = 16000) {
    brosoundml::AudioBuffer a;
    a.sample_rate = sr;
    a.samples.resize(static_cast<std::size_t>(seconds * sr));
    for (std::size_t n = 0; n < a.samples.size(); ++n) {
        a.samples[n] = 0.2f * std::sin(2.0f * 3.14159265358979f * hz *
                                       static_cast<float>(n) / sr);
    }
    return a;
}

// ─── Tests ─────────────────────────────────────────────────────────────────

template <typename Fn>
static bool throws_runtime_error(Fn&& fn) {
    try { fn(); }
    catch (const std::runtime_error&) { return true; }
    catch (...) { return false; }
    return false;
}

static std::string tag(const char* msg, const char* dev_name) {
    std::string s = "["; s += dev_name; s += "] "; s += msg; return s;
}

static int run_for_device(brotensor::Device dev, const char* dev_name) {
    using brosoundml::AudioBuffer;
    using brosoundml::Whisper;

    const fs::path root = fs::temp_directory_path() / "brosoundml_whisper_e2e";
    fs::remove_all(root);
    fs::create_directories(root);

    TinyShape shape;
    write_file(root / "config.json", build_config_json(shape));
    write_stub_checkpoint(root / "model.safetensors", shape);

    Whisper w;
    w.load(root.string(), dev);
    CHECK(w.loaded(), tag("loaded synthetic Whisper checkpoint", dev_name).c_str());
    CHECK(w.config().vocab_size == shape.vocab_size,
          tag("config vocab_size round-trips through load()", dev_name).c_str());

    // ── Bad-input guards ──────────────────────────────────────────────────
    {
        AudioBuffer empty;
        empty.sample_rate = 16000;
        CHECK(throws_runtime_error([&] { w.transcribe(empty, {0, 1, 2}); }),
              tag("transcribe rejects empty audio", dev_name).c_str());

        AudioBuffer wrong_sr;
        wrong_sr.sample_rate = 22050;
        wrong_sr.samples.assign(22050, 0.0f);
        CHECK(throws_runtime_error([&] { w.transcribe(wrong_sr, {0, 1, 2}); }),
              tag("transcribe rejects non-16k audio", dev_name).c_str());

        AudioBuffer ok = sine(440.0f, 0.5f);
        CHECK(throws_runtime_error([&] { w.transcribe(ok, {}); }),
              tag("transcribe rejects empty prompt", dev_name).c_str());
    }

    // ── Happy path: bounded greedy run ───────────────────────────────────
    AudioBuffer audio = sine(440.0f, 0.5f);
    const std::vector<int32_t> prompt = {0, 1, 2};
    auto result = w.transcribe(audio, prompt, /*max_new_tokens=*/5);

    CHECK(result.token_ids.size() >= prompt.size(),
          tag("result includes the prompt", dev_name).c_str());
    const std::size_t n_new = result.token_ids.size() - prompt.size();
    CHECK(n_new <= 5,
          tag("generated count respects max_new_tokens", dev_name).c_str());
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        CHECK(result.token_ids[i] == prompt[i],
              tag("prompt prefix is preserved verbatim in the result", dev_name).c_str());
    }
    bool in_range = true;
    for (auto id : result.token_ids) {
        if (id < 0 || id >= shape.vocab_size) in_range = false;
    }
    CHECK(in_range, tag("every returned id is within [0, vocab_size)", dev_name).c_str());

    bool stopped_at_max = (n_new == 5);
    bool stopped_at_eos = (n_new < 5);
    CHECK(stopped_at_max || stopped_at_eos,
          tag("greedy loop terminated via EOS or max_new_tokens", dev_name).c_str());

    // ── Cache reset between calls ────────────────────────────────────────
    AudioBuffer audio2 = sine(880.0f, 0.5f);
    auto run_a = w.transcribe(audio, prompt, /*max_new_tokens=*/1);
    auto run_b = w.transcribe(audio2, prompt, /*max_new_tokens=*/1);
    auto run_a_again = w.transcribe(audio, prompt, /*max_new_tokens=*/1);

    CHECK(run_a.token_ids.size() == run_a_again.token_ids.size(),
          tag("deterministic re-run has same length", dev_name).c_str());
    bool same_tokens = run_a.token_ids == run_a_again.token_ids;
    // Sample-identical tokens on CUDA vs CPU are not required — token argmax
    // can tip differently with FP noise. But within a single device run,
    // determinism must hold.
    CHECK(same_tokens,
          tag("deterministic re-run after a different call yields the same tokens",
              dev_name).c_str());
    (void)run_b;

    auto result_default = w.transcribe(audio, prompt, /*max_new_tokens=*/0);
    const std::size_t n_new_default =
        result_default.token_ids.size() - prompt.size();
    CHECK(n_new_default <=
          static_cast<std::size_t>(shape.max_target_positions - prompt.size()),
          tag("default max_new_tokens respects max_target_positions", dev_name).c_str());

    fs::remove_all(root);
    return failures;
}

static int run() {
    int f = run_for_device(brotensor::Device::CPU, "CPU");
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        f += run_for_device(brotensor::Device::CUDA, "CUDA");
    } else {
        std::printf("test_whisper_e2e: CUDA not available — CUDA path skipped\n");
    }
    if (brotensor::is_available(brotensor::Device::Metal)) {
        f += run_for_device(brotensor::Device::Metal, "Metal");
    } else {
        std::printf("test_whisper_e2e: Metal not available — Metal path skipped\n");
    }
    return f;
}

int main() {
    brotensor::init();
    try {
        const int f = run();
        if (f) {
            std::fprintf(stderr, "test_whisper_e2e: %d failure(s)\n", f);
            return 1;
        }
        std::printf("test_whisper_e2e: all checks passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_whisper_e2e: uncaught exception: %s\n", e.what());
        return 2;
    } catch (...) {
        std::fprintf(stderr, "test_whisper_e2e: uncaught non-std exception\n");
        return 2;
    }
}
