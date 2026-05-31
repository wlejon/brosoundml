#define _CRT_SECURE_NO_WARNINGS  // std::getenv gate below (MSVC C4996)
// Qwen3-TTS stage-1 loader contract: config.json + speech_tokenizer/config.json
// parsing and safetensors validation. The forward pass is still in build-out,
// so synthesize() must throw a staged std::runtime_error naming the stage.
#include "brosoundml/qwen_tts.h"

#include "qwen_tts_talker.h"          // internal: white-box Talker forward validation
#include "qwen_tts_code_predictor.h"  // internal: white-box Code Predictor validation
#include "qwen_tts_generate.h"        // internal: dual-track AR loop

#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <algorithm>
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

        // ─── Codec decode vs ground-truth fixture (opt-in, CPU) ────────────
        //
        // decode_codes() runs the codec tail (codes -> 24 kHz) on the CPU in
        // FP32 regardless of the load device, so this numeric check runs once,
        // under the CPU invocation. Fixtures are produced by the genuine
        // upstream decoder — see tests/ref/gen_qwen_tts_codec_fixture.py — and
        // are gitignored (*.bin); the check skips when they are absent.
        //
        // The full x1920 decode is heavy on the CPU FP32 path (minutes), so it
        // is additionally gated behind BROSOUNDML_RUN_CODEC_FIXTURE to keep the
        // default suite fast — set it to run the numeric validation:
        //   BROSOUNDML_RUN_CODEC_FIXTURE=1 ctest -R qwen_tts -C Release
        if (dev == brotensor::Device::CPU && std::getenv("BROSOUNDML_RUN_CODEC_FIXTURE")) {
            const fs::path fix_dir = fs::path(BROSOUNDML_REPO_DIR) / "tests" / "fixtures";
            auto check_fixture = [&](const char* fname) {
                const fs::path path = fix_dir / fname;
                std::ifstream f(path.string(), std::ios::binary);
                if (!f) return;  // not generated locally — skip
                int32_t hdr[3];
                f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
                const int K = hdr[0], T = hdr[1], n = hdr[2];
                std::vector<int32_t> codes(static_cast<std::size_t>(K) * T);
                std::vector<float>   ref(static_cast<std::size_t>(n));
                f.read(reinterpret_cast<char*>(codes.data()),
                       static_cast<std::streamsize>(codes.size() * sizeof(int32_t)));
                f.read(reinterpret_cast<char*>(ref.data()),
                       static_cast<std::streamsize>(ref.size() * sizeof(float)));
                CHECK(static_cast<bool>(f), "fixture read complete");

                brosoundml::AudioBuffer out = q.decode_codes(codes, K, T);
                CHECK(out.sample_rate == 24000, "decode_codes returns 24 kHz");
                CHECK(static_cast<int>(out.samples.size()) == n,
                      "decode_codes returns T*1920 samples");
                if (static_cast<int>(out.samples.size()) != n) return;

                double max_abs = 0.0, sum_abs = 0.0;
                for (int i = 0; i < n; ++i) {
                    const double d = std::fabs(static_cast<double>(out.samples[i]) - ref[i]);
                    max_abs = std::max(max_abs, d);
                    sum_abs += d;
                }
                const double mean_abs = sum_abs / n;
                std::printf("    [codec fixture %s] T=%d  max|Δ|=%.2e  mean|Δ|=%.2e\n",
                            fname, T, max_abs, mean_abs);
                // FP32 vs FP32 reference; only float reassociation across the
                // deep (x1920) stack differs.
                CHECK(max_abs  < 2e-3, "codec decode matches upstream (max abs)");
                CHECK(mean_abs < 1e-4, "codec decode matches upstream (mean abs)");
            };
            check_fixture("qwen_tts_codec_small.bin");  // T < sliding_window
            check_fixture("qwen_tts_codec.bin");        // T > sliding_window (72)
        }

        // ─── Talker forward vs ground-truth fixture (opt-in, CPU) ──────────
        //
        // White-box check of the 28-layer Talker (dual embedding + QK-norm +
        // interleaved M-RoPE + codec_head). Builds the internal QwenTtsTalker
        // directly and validates a prefill forward + the embedding helpers
        // against the genuine upstream model. Fixture is gitignored and made by
        // tests/ref/gen_talker_fixture (see the codec ref script's header for
        // the transformers==4.57.3 venv recipe); skipped when absent.
        if (dev == brotensor::Device::CPU) {
            const fs::path tpath =
                fs::path(BROSOUNDML_REPO_DIR) / "tests" / "fixtures" / "qwen_tts_talker.bin";
            std::ifstream f(tpath.string(), std::ios::binary);
            if (f) {
                int32_t hdr[5];
                f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
                const int T = hdr[0], H = hdr[1], V = hdr[2], nT = hdr[3], nC = hdr[4];
                std::vector<int32_t> pos(static_cast<std::size_t>(3) * T);
                std::vector<float> ie(static_cast<std::size_t>(T) * H);
                std::vector<float> ref_h(static_cast<std::size_t>(T) * H);
                std::vector<float> ref_l(static_cast<std::size_t>(T) * V);
                std::vector<int32_t> text_ids(nT);
                std::vector<float> ref_tp(static_cast<std::size_t>(nT) * H);
                std::vector<int32_t> codec_ids(nC);
                std::vector<float> ref_ce(static_cast<std::size_t>(nC) * H);
                auto rd = [&](void* p, std::size_t bytes) {
                    f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(bytes));
                };
                rd(pos.data(), pos.size() * 4);
                rd(ie.data(), ie.size() * 4);
                rd(ref_h.data(), ref_h.size() * 4);
                rd(ref_l.data(), ref_l.size() * 4);
                rd(text_ids.data(), text_ids.size() * 4);
                rd(ref_tp.data(), ref_tp.size() * 4);
                rd(codec_ids.data(), codec_ids.size() * 4);
                rd(ref_ce.data(), ref_ce.size() * 4);
                CHECK(static_cast<bool>(f), "talker fixture read complete");

                brosoundml::QwenTtsTalker talker;
                {
                    auto w = brotensor::safetensors::File::open((root / "model.safetensors").string());
                    talker.load(w, c.talker);
                }
                CHECK(talker.hidden == H, "talker hidden matches fixture");
                CHECK(talker.vocab == V, "talker vocab matches fixture");

                brotensor::Tensor hid, log;
                talker.forward(ie.data(), T, pos.data(), hid, log);

                auto cmp = [&](const char* what, const brotensor::Tensor& got,
                               const std::vector<float>& ref, double tol) {
                    double max_abs = 0.0, sum_abs = 0.0;
                    const float* g = got.host_f32();
                    for (std::size_t i = 0; i < ref.size(); ++i) {
                        const double d = std::fabs(static_cast<double>(g[i]) - ref[i]);
                        max_abs = std::max(max_abs, d);
                        sum_abs += d;
                    }
                    std::printf("    [talker %s] max|Δ|=%.2e  mean|Δ|=%.2e\n",
                                what, max_abs, sum_abs / ref.size());
                    CHECK(max_abs < tol, what);
                };
                cmp("hidden states", hid, ref_h, 5e-2);
                cmp("codec_head logits", log, ref_l, 5e-2);

                // Embedding helpers.
                double emax = 0.0;
                std::vector<float> buf(H);
                for (int i = 0; i < nT; ++i) {
                    talker.text_embed_proj(text_ids[i], buf.data());
                    for (int d = 0; d < H; ++d)
                        emax = std::max(emax, std::fabs((double)buf[d] - ref_tp[i * H + d]));
                }
                std::printf("    [talker text_projection] max|Δ|=%.2e\n", emax);
                CHECK(emax < 5e-3, "talker text_embed_proj matches upstream");
                double cmax = 0.0;
                for (int i = 0; i < nC; ++i) {
                    talker.codec_embed(codec_ids[i], buf.data());
                    for (int d = 0; d < H; ++d)
                        cmax = std::max(cmax, std::fabs((double)buf[d] - ref_ce[i * H + d]));
                }
                CHECK(cmax < 1e-5, "talker codec_embed matches upstream");

                // ── KV-cache self-consistency ──────────────────────────────
                // A cached prefill of T-1 tokens followed by a single-token
                // step must reproduce the uncached full forward at the last
                // position — proving run()'s cache path against the verified
                // (fixture-checked) prefill path. run() reads positions as
                // pos3n[a*n + t], so repack the fixture grid (a*T + t) for the
                // n=T-1 prefix and the lone stepped token.
                if (T >= 2) {
                    const int np = T - 1;
                    std::vector<int32_t> pre_pos(static_cast<std::size_t>(3) * np);
                    int32_t step_pos[3];
                    for (int a = 0; a < 3; ++a) {
                        for (int t = 0; t < np; ++t)
                            pre_pos[a * np + t] = pos[a * T + t];
                        step_pos[a] = pos[a * T + (T - 1)];
                    }
                    brosoundml::QwenTtsTalkerCache cache;
                    cache.reset(talker.num_layers);
                    brotensor::Tensor h_pre, h_step;
                    talker.run(ie.data(), np, pre_pos.data(), &cache, h_pre);
                    talker.run(ie.data() + static_cast<std::size_t>(np) * H, 1,
                               step_pos, &cache, h_step);
                    double cache_max = 0.0;
                    const float* hs = h_step.host_f32();
                    const float* hf = hid.host_f32() + static_cast<std::size_t>(T - 1) * H;
                    for (int d = 0; d < H; ++d)
                        cache_max = std::max(cache_max, std::fabs((double)hs[d] - hf[d]));
                    std::printf("    [talker kv-cache step] max|Δ|=%.2e\n", cache_max);
                    CHECK(cache_max < 1e-4, "talker cached step matches uncached forward");
                }
            }
        }

        // ─── Code Predictor + AR loop vs ground-truth fixture (CPU) ────────
        //
        // White-box check of the stage-4 path: the 5-layer Code Predictor
        // expansion (codebooks 1..15) and the dual-track AR loop (Talker +
        // Code Predictor → frame stream). Builds QwenTtsTalker + QwenTtsCodePredictor
        // directly and matches the genuine upstream Qwen3TTSTalkerForConditional-
        // Generation (do_sample=False). The discrete codes must match EXACTLY.
        // Fixture by tests/ref/gen_qwen_tts_ar_fixture.py; gitignored, skipped
        // when absent.
        if (dev == brotensor::Device::CPU) {
            const fs::path apath =
                fs::path(BROSOUNDML_REPO_DIR) / "tests" / "fixtures" / "qwen_tts_ar.bin";
            std::ifstream f(apath.string(), std::ios::binary);
            if (f) {
                auto rd = [&](void* p, std::size_t bytes) {
                    f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(bytes));
                };
                // Part A header
                int32_t a_hdr[3];
                rd(a_hdr, sizeof(a_hdr));
                const int H = a_hdr[0], n_codes = a_hdr[1], c0 = a_hdr[2];
                std::vector<float>   past_hidden(static_cast<std::size_t>(H));
                std::vector<int32_t> cp_codes(n_codes);
                rd(past_hidden.data(), past_hidden.size() * 4);
                rd(cp_codes.data(), cp_codes.size() * 4);
                // Part B header
                int32_t b_hdr[6];
                rd(b_hdr, sizeof(b_hdr));
                const int T = b_hdr[0], L = b_hdr[1], F = b_hdr[2], G = b_hdr[3],
                          eos_id = b_hdr[4], rope_delta = b_hdr[5];
                std::vector<float>   prefill(static_cast<std::size_t>(T) * H);
                std::vector<int32_t> pos3(static_cast<std::size_t>(3) * T);
                std::vector<float>   trailing(static_cast<std::size_t>(L) * H);
                std::vector<float>   tts_pad(static_cast<std::size_t>(H));
                std::vector<int32_t> frames(static_cast<std::size_t>(F) * G);
                rd(prefill.data(), prefill.size() * 4);
                rd(pos3.data(), pos3.size() * 4);
                rd(trailing.data(), trailing.size() * 4);
                rd(tts_pad.data(), tts_pad.size() * 4);
                rd(frames.data(), frames.size() * 4);
                CHECK(static_cast<bool>(f), tag("AR fixture read complete"));

                brosoundml::QwenTtsTalker talker;
                brosoundml::QwenTtsCodePredictor cp;
                {
                    auto w = brotensor::safetensors::File::open((root / "model.safetensors").string());
                    talker.load(w, c.talker);
                    cp.load(w, c.talker.code_predictor);
                }
                CHECK(talker.hidden == H, tag("AR fixture hidden matches"));
                CHECK(cp.num_code_groups == G, tag("code predictor num_code_groups matches"));

                // Part A — standalone Code Predictor (codebooks 1..15).
                std::vector<int> got_cp;
                cp.predict(past_hidden.data(), talker.codec_embedding_row(c0), got_cp);
                int cp_mismatch = 0;
                for (int i = 0; i < n_codes; ++i)
                    if (got_cp[i] != cp_codes[i]) ++cp_mismatch;
                std::printf("    [code predictor] %d/%d codes match\n",
                            n_codes - cp_mismatch, n_codes);
                CHECK(cp_mismatch == 0, tag("code predictor codes match upstream exactly"));

                // Part B — dual-track AR loop.
                brosoundml::QwenTtsGenParams params;
                params.eos_id = eos_id;
                params.max_frames = F;
                params.rope_delta = rope_delta;
                std::vector<int32_t> got_frames;
                const int got_F = generate_codes(
                    talker, cp, prefill.data(), T, pos3.data(),
                    trailing.data(), L, tts_pad.data(), params, got_frames);
                CHECK(got_F == F, tag("AR loop emits the expected frame count"));
                int fr_mismatch = 0;
                if (got_F == F) {
                    for (std::size_t i = 0; i < frames.size(); ++i)
                        if (got_frames[i] != frames[i]) ++fr_mismatch;
                }
                std::printf("    [AR loop] F=%d  code mismatches=%d/%d\n",
                            F, fr_mismatch, F * G);
                CHECK(fr_mismatch == 0, tag("AR loop frames match upstream exactly"));
            }
        }
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
