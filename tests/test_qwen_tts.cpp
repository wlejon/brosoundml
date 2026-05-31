// Qwen3-TTS stage-1 loader contract: config.json + speech_tokenizer/config.json
// parsing and safetensors validation. The forward pass is still in build-out,
// so synthesize() must throw a staged std::runtime_error naming the stage.
#include "brosoundml/qwen_tts.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
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

static int run();
int main() {
    brotensor::init();
    try { return run(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "test_qwen_tts: uncaught exception: %s\n", e.what());
        return 2;
    }
    catch (...) { std::fprintf(stderr, "test_qwen_tts: uncaught non-std exception\n"); return 2; }
}

static int run() {
    using brosoundml::QwenTts;
    using brosoundml::QwenTtsVariant;

    // ─── Default-constructed QwenTts ───────────────────────────────────────
    {
        QwenTts q;
        CHECK(!q.loaded(), "a fresh QwenTts is not loaded");
        CHECK(q.config().sample_rate == 24000, "default config sample rate is 24 kHz");
        CHECK(q.speakers().empty(), "a fresh QwenTts has no speakers");

        CHECK(throws_runtime_error([&] { q.load("nonexistent-qwen-tts-dir"); }),
              "load() on a missing directory throws");
        CHECK(throws_runtime_error([&] { q.synthesize("hi", "serena"); }),
              "synthesize() before load() throws");
    }

    // ─── Real-weights smoke (opt-in) ───────────────────────────────────────
    //
    // CPU and (if available) CUDA. Stage 1 only validates the loader contract;
    // synthesize() must still throw the staged stub.
    auto run_real_smoke = [&](brotensor::Device dev, const char* dev_name) {
        const fs::path root = fs::path(BROSOUNDML_REPO_DIR) / "weights" /
                              "qwen-tts" / "0.6B-customvoice";
        if (!fs::exists(root / "model.safetensors")) return;

        auto tag = [&](const char* s) {
            static std::string buf;
            buf = "["; buf += dev_name; buf += "] "; buf += s;
            return buf.c_str();
        };

        QwenTts q;
        q.load(root.string(), dev);
        CHECK(q.loaded(), tag("real load() sets loaded()"));

        const auto& c  = q.config();
        const auto& tk = c.talker;
        CHECK(c.variant == QwenTtsVariant::CustomVoice, tag("variant == CustomVoice"));
        CHECK(c.sample_rate == 24000, tag("sample_rate == 24000"));

        CHECK(tk.transformer.hidden_size         == 1024, tag("talker hidden_size == 1024"));
        CHECK(tk.transformer.num_hidden_layers   == 28,   tag("talker layers == 28"));
        CHECK(tk.transformer.num_attention_heads == 16,   tag("talker q heads == 16"));
        CHECK(tk.transformer.num_key_value_heads == 8,    tag("talker kv heads == 8"));
        CHECK(tk.transformer.head_dim            == 128,  tag("talker head_dim == 128"));
        CHECK(tk.num_code_groups  == 16,     tag("talker num_code_groups == 16"));
        CHECK(tk.vocab_size       == 3072,   tag("talker codec vocab == 3072"));
        CHECK(tk.text_vocab_size  == 151936, tag("talker text vocab == 151936"));
        CHECK(tk.text_hidden_size == 2048,   tag("talker text_hidden == 2048"));
        CHECK(tk.mrope_section == std::vector<int>({24, 20, 20}),
              tag("talker mrope_section == [24,20,20]"));
        CHECK(tk.position_id_per_seconds == 13, tag("talker position_id_per_seconds == 13"));

        CHECK(tk.code_predictor.transformer.num_hidden_layers == 5,
              tag("code predictor layers == 5"));
        CHECK(tk.code_predictor.vocab_size == 2048, tag("code predictor vocab == 2048"));

        const auto& cd = c.codec;
        CHECK(cd.num_quantizers          == 16,    tag("codec num_quantizers == 16"));
        CHECK(cd.semantic_codebook_size  == 4096,  tag("codec semantic codebook == 4096"));
        CHECK(cd.latent_dim              == 1024,  tag("codec latent_dim == 1024"));
        CHECK(cd.decode_upsample_rate    == 1920,  tag("codec decode_upsample_rate == 1920"));
        CHECK(cd.output_sample_rate      == 24000, tag("codec output_sample_rate == 24000"));
        CHECK(cd.upsample_rates    == std::vector<int>({8, 5, 4, 3}),
              tag("codec upsample_rates == [8,5,4,3]"));
        CHECK(cd.upsampling_ratios == std::vector<int>({2, 2}),
              tag("codec upsampling_ratios == [2,2]"));
        // Sanity: the two upsampling stages multiply to the total rate.
        int prod = 1;
        for (int r : cd.upsample_rates)    prod *= r;
        for (int r : cd.upsampling_ratios) prod *= r;
        CHECK(prod == cd.decode_upsample_rate,
              tag("upsample stages multiply to decode_upsample_rate"));

        // Preset speakers: 9 in CustomVoice, including "serena".
        auto spk = q.speakers();
        CHECK(spk.size() == 9, tag("CustomVoice has 9 speakers"));
        CHECK(std::find(spk.begin(), spk.end(), "serena") != spk.end(),
              tag("speakers include 'serena'"));
        CHECK(tk.spk_dialect.at("dylan") == "beijing_dialect",
              tag("dylan tagged beijing_dialect"));

        // Forward pass is staged out: synthesize() throws, but only after the
        // obvious caller errors (unknown speaker / language) are reported.
        CHECK(throws_runtime_error([&] { q.synthesize("hello", "no_such_speaker"); }),
              tag("synthesize() rejects an unknown speaker"));
        CHECK(throws_runtime_error([&] { q.synthesize("hello", "serena", "klingon"); }),
              tag("synthesize() rejects an unsupported language"));
        bool staged = false;
        std::string msg;
        try { q.synthesize("hello", "serena", "english"); }
        catch (const std::runtime_error& e) { staged = true; msg = e.what(); }
        CHECK(staged, tag("synthesize() throws the staged stub for valid input"));
        CHECK(msg.find("not yet built") != std::string::npos,
              tag("staged stub names the build-out stage"));

        std::printf("    [%s] qwen-tts loader: %zu speakers, codec x%d -> %d Hz\n",
                    dev_name, spk.size(), cd.decode_upsample_rate,
                    cd.output_sample_rate);
    };

    run_real_smoke(brotensor::Device::CPU, "CPU");
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        run_real_smoke(brotensor::Device::CUDA, "CUDA");
    }
    if (brotensor::is_available(brotensor::Device::Metal)) {
        run_real_smoke(brotensor::Device::Metal, "Metal");
    }

    if (failures == 0) {
        std::printf("test_qwen_tts: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_qwen_tts: %d check(s) failed\n", failures);
    return 1;
}
