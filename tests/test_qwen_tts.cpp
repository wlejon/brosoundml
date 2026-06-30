#define _CRT_SECURE_NO_WARNINGS  // std::getenv gate below (MSVC C4996)
// Qwen3-TTS stage-1 loader contract: config.json + speech_tokenizer/config.json
// parsing and safetensors validation. The forward pass is still in build-out,
// so synthesize() must throw a staged std::runtime_error naming the stage.
#include "brosoundml/qwen_tts.h"

#include "qwen_tts_talker.h"          // internal: white-box Talker forward validation
#include "qwen_tts_code_predictor.h"  // internal: white-box Code Predictor validation
#include "qwen_tts_generate.h"        // internal: dual-track AR loop + prefill assembly
#include "qwen_tts_speaker_encoder.h" // internal: Base ECAPA-TDNN x-vector encoder

#include <brolm/qwen_tokenizer.h>     // text -> Qwen BPE ids (stage-5 tokenizer check)

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
    //
    // Carries the CPU sampled-code stream out of the per-device lambda so the
    // CUDA run can assert bit-identical sampling parity (device-neutrality for
    // the stochastic path, not just greedy).
    std::vector<int32_t> sampled_ref;
    int sampled_ref_F = -1;
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

        // synthesize() now runs the full pipeline; it still rejects the obvious
        // caller errors (unknown speaker / language) before any heavy work. The
        // full end-to-end run is exercised by the stage-5 fixture block below
        // (and gated behind the heavy-decode env), not here.
        CHECK(throws_runtime_error([&] { q.synthesize("hello", "no_such_speaker"); }),
              tag("synthesize() rejects an unknown speaker"));
        CHECK(throws_runtime_error([&] { q.synthesize("hello", "serena", "klingon"); }),
              tag("synthesize() rejects an unsupported language"));

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
        if (std::getenv("BROSOUNDML_RUN_CODEC_FIXTURE")) {
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
                std::printf("    [%s codec fixture %s] T=%d  max|Δ|=%.2e  mean|Δ|=%.2e\n",
                            dev_name, fname, T, max_abs, mean_abs);
                // FP32 vs FP32 reference on both backends; only float
                // reassociation across the deep (x1920) stack differs.
                CHECK(max_abs  < 2e-3, tag("codec decode matches upstream (max abs)"));
                CHECK(mean_abs < 1e-4, tag("codec decode matches upstream (mean abs)"));
            };
            check_fixture("qwen_tts_codec_small.bin");  // T < sliding_window
            check_fixture("qwen_tts_codec.bin");        // T > sliding_window (72)

            // ─── Codec ENCODE vs ground-truth fixture ──────────────────────
            // encode_audio() is the analysis path (waveform -> RVQ codes), the
            // inverse of decode_codes(). Fixtures (waveform + codes) come from
            // the genuine upstream encoder — see .scratch/gen_encode_fixture.py.
            //
            // Codes are discrete, so unlike the continuous decode check there is
            // an inherent floor: an argmin over the codebook flips on the rare
            // frame that is near-equidistant between two codewords once FP32
            // reassociation differs from PyTorch's. That floor is small (~1-2
            // frames per codebook) and uniform across codebooks — the semantic
            // codebook (a direct argmin on the front-end features, no residual
            // cascade) stays exact, which is the real "front-end is faithful"
            // signal; CPU and CUDA agree bit-for-bit. >=97% also cleanly catches
            // structural regressions (e.g. a wrong attention window read ~93%).
            auto check_encode = [&](const char* fname) {
                const fs::path path = fix_dir / fname;
                std::ifstream f(path.string(), std::ios::binary);
                if (!f) return;  // not generated locally — skip
                int32_t hdr[3];
                f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
                const int K = hdr[0], T = hdr[1], n = hdr[2];
                std::vector<int32_t> ref(static_cast<std::size_t>(K) * T);
                std::vector<float>   wav(static_cast<std::size_t>(n));
                f.read(reinterpret_cast<char*>(ref.data()),
                       static_cast<std::streamsize>(ref.size() * sizeof(int32_t)));
                f.read(reinterpret_cast<char*>(wav.data()),
                       static_cast<std::streamsize>(wav.size() * sizeof(float)));
                CHECK(static_cast<bool>(f), "encode fixture read complete");

                int Tout = 0;
                brosoundml::AudioBuffer in(std::move(wav), 24000);
                std::vector<int32_t> got = q.encode_audio(in, &Tout);
                CHECK(Tout == T, tag("encode_audio frame count matches upstream"));
                CHECK(static_cast<int>(got.size()) == K * T,
                      tag("encode_audio returns K*T codes"));
                if (Tout != T || static_cast<int>(got.size()) != K * T) return;

                int mism = 0;
                std::vector<int> per_cb(K, 0);
                for (int k = 0; k < K; ++k)
                    for (int t = 0; t < T; ++t) {
                        const std::size_t i = static_cast<std::size_t>(k) * T + t;
                        if (got[i] != ref[i]) { ++mism; ++per_cb[k]; }
                    }
                const double rate = 1.0 - static_cast<double>(mism) / (K * T);
                std::printf("    [%s encode fixture %s] T=%d  codes match=%.4f  per-codebook mism:",
                            dev_name, fname, T, rate);
                for (int k = 0; k < K; ++k) std::printf(" %d", per_cb[k]);
                std::printf("\n");
                CHECK(per_cb[0] == 0, tag("encode semantic codebook exact"));
                CHECK(rate >= 0.97, tag("encode matches upstream codes (>=97%)"));
            };
            check_encode("qwen_tts_encode_small.bin");  // T < sliding_window (250)
            check_encode("qwen_tts_encode.bin");        // T > sliding_window
        }

        // ─── Talker forward vs ground-truth fixture (opt-in, CPU) ──────────
        //
        // White-box check of the 28-layer Talker (dual embedding + QK-norm +
        // interleaved M-RoPE + codec_head). Builds the internal QwenTtsTalker
        // directly and validates a prefill forward + the embedding helpers
        // against the genuine upstream model. Fixture is gitignored and made by
        // tests/ref/gen_talker_fixture (see the codec ref script's header for
        // the transformers==4.57.3 venv recipe); skipped when absent.
        {
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
                    talker.load(w, c.talker, dev);
                }
                CHECK(talker.hidden == H, "talker hidden matches fixture");
                CHECK(talker.vocab == V, "talker vocab matches fixture");

                brotensor::Tensor hid, log;
                talker.forward(ie.data(), T, pos.data(), hid, log);

                auto cmp = [&](const char* what, const brotensor::Tensor& got,
                               const std::vector<float>& ref, double tol) {
                    double max_abs = 0.0, sum_abs = 0.0;
                    brotensor::Tensor gc = got.to(brotensor::Device::CPU);
                    const float* g = gc.host_f32();
                    for (std::size_t i = 0; i < ref.size(); ++i) {
                        const double d = std::fabs(static_cast<double>(g[i]) - ref[i]);
                        max_abs = std::max(max_abs, d);
                        sum_abs += d;
                    }
                    std::printf("    [%s talker %s] max|Δ|=%.2e  mean|Δ|=%.2e\n",
                                dev_name, what, max_abs, sum_abs / ref.size());
                    CHECK(max_abs < tol, tag(what));
                };
                // Attention is FP32 on both backends (CPU flash; CUDA via
                // flash_attention_varlen_forward's FP32 kernel), so CUDA tracks
                // CPU — and upstream — to float round-off. Same bound for both.
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
                std::printf("    [%s talker text_projection] max|Δ|=%.2e\n", dev_name, emax);
                CHECK(emax < 5e-3, tag("talker text_embed_proj matches upstream"));
                double cmax = 0.0;
                for (int i = 0; i < nC; ++i) {
                    talker.codec_embed(codec_ids[i], buf.data());
                    for (int d = 0; d < H; ++d)
                        cmax = std::max(cmax, std::fabs((double)buf[d] - ref_ce[i * H + d]));
                }
                CHECK(cmax < 1e-5, tag("talker codec_embed matches upstream"));

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
                    brotensor::Tensor h_step_c = h_step.to(brotensor::Device::CPU);
                    brotensor::Tensor hid_c = hid.to(brotensor::Device::CPU);
                    const float* hs = h_step_c.host_f32();
                    const float* hf = hid_c.host_f32() + static_cast<std::size_t>(T - 1) * H;
                    for (int d = 0; d < H; ++d)
                        cache_max = std::max(cache_max, std::fabs((double)hs[d] - hf[d]));
                    std::printf("    [%s talker kv-cache step] max|Δ|=%.2e\n", dev_name, cache_max);
                    CHECK(cache_max < 1e-4,
                          tag("talker cached step matches uncached forward"));
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
        {
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
                    talker.load(w, c.talker, dev);
                    cp.load(w, c.talker.code_predictor, c.talker.transformer.hidden_size, dev);
                }
                CHECK(talker.hidden == H, tag("AR fixture hidden matches"));
                CHECK(cp.num_code_groups == G, tag("code predictor num_code_groups matches"));

                // Part A — standalone Code Predictor (codebooks 1..15). Source
                // the codebook-0 embedding device-safely (the table may be on GPU).
                std::vector<float> c0_embed(static_cast<std::size_t>(H));
                talker.codec_embed(c0, c0_embed.data());
                std::vector<int> got_cp;
                cp.predict(past_hidden.data(), c0_embed.data(), got_cp);
                int cp_mismatch = 0;
                for (int i = 0; i < n_codes; ++i)
                    if (got_cp[i] != cp_codes[i]) ++cp_mismatch;
                std::printf("    [code predictor] %d/%d codes match\n",
                            n_codes - cp_mismatch, n_codes);
                // FP32 attention on both backends ⇒ the greedy argmax stream is
                // bit-identical on CPU and CUDA, so exact-code match holds on both.
                CHECK(cp_mismatch == 0, tag("code predictor codes match upstream exactly"));

                // Part B — dual-track AR loop.
                brosoundml::QwenTtsGenParams params;
                params.eos_id = eos_id;
                params.max_frames = F;
                params.rope_delta = rope_delta;
                std::vector<int32_t> got_frames;
                brosoundml::QwenTtsGenState gen;   // per-stream AR scratch
                const int got_F = generate_codes(
                    talker, cp, gen, prefill.data(), T, pos3.data(),
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

        // ─── End-to-end synthesis vs ground-truth fixture (CPU) ────────────
        //
        // Stage-5 white-box check: the tokenizer, the CustomVoice prefill
        // assembly, and the AR loop with the upstream codebook-0 logits policy
        // (suppress top-1024 codec tokens, min_new_tokens=2, repetition_penalty
        // 1.05, greedy). Matches the genuine upstream assembly + talker
        // generation; the code stream must match exactly. Codec decode (codes
        // -> waveform) is stage 2, already verified, so this compares codes.
        // Fixture by tests/ref/gen_qwen_tts_synth_fixture.py; skipped if absent.
        {
            const fs::path spath =
                fs::path(BROSOUNDML_REPO_DIR) / "tests" / "fixtures" / "qwen_tts_synth.bin";
            std::ifstream f(spath.string(), std::ios::binary);
            if (f) {
                auto rd = [&](void* p, std::size_t bytes) {
                    f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(bytes));
                };
                int32_t text_len = 0;
                rd(&text_len, 4);
                std::string text(static_cast<std::size_t>(text_len), '\0');
                rd(text.data(), static_cast<std::size_t>(text_len));
                int32_t h2[2];
                rd(h2, sizeof(h2));
                const int H = h2[0], n_ids = h2[1];
                std::vector<int32_t> ref_ids(n_ids);
                rd(ref_ids.data(), ref_ids.size() * 4);
                int32_t sl[2];
                rd(sl, sizeof(sl));
                const int spk_id = sl[0], language_id = sl[1];
                int32_t T = 0; rd(&T, 4);
                std::vector<float> ref_prefill(static_cast<std::size_t>(T) * H);
                rd(ref_prefill.data(), ref_prefill.size() * 4);
                int32_t L = 0; rd(&L, 4);
                std::vector<float> ref_trailing(static_cast<std::size_t>(L) * H);
                rd(ref_trailing.data(), ref_trailing.size() * 4);
                std::vector<float> ref_pad(static_cast<std::size_t>(H));
                rd(ref_pad.data(), ref_pad.size() * 4);
                int32_t fg[2]; rd(fg, sizeof(fg));
                const int F = fg[0], G = fg[1];
                std::vector<int32_t> ref_frames(static_cast<std::size_t>(F) * G);
                rd(ref_frames.data(), ref_frames.size() * 4);
                CHECK(static_cast<bool>(f), tag("synth fixture read complete"));

                brosoundml::QwenTtsTalker talker;
                brosoundml::QwenTtsCodePredictor cp;
                {
                    auto w = brotensor::safetensors::File::open((root / "model.safetensors").string());
                    talker.load(w, c.talker, dev);
                    cp.load(w, c.talker.code_predictor, c.talker.transformer.hidden_size, dev);
                }
                auto tok = brolm::qwen::Tokenizer::load(
                    (root / "vocab.json").string(), (root / "merges.txt").string());
                tok.register_special_token("<|im_start|>", c.im_start_id);
                tok.register_special_token("<|im_end|>",   c.im_end_id);

                // (1) tokenizer: the chat prompt must reproduce the upstream ids.
                const std::string prompt =
                    "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
                std::vector<int32_t> ids = tok.encode(prompt);
                bool ids_match = (ids.size() == ref_ids.size());
                if (ids_match)
                    for (std::size_t i = 0; i < ids.size(); ++i)
                        if (ids[i] != ref_ids[i]) { ids_match = false; break; }
                std::printf("    [synth tokenizer] %zu ids, match=%d\n", ids.size(), (int)ids_match);
                CHECK(ids_match, tag("tokenizer reproduces upstream chat-prompt ids"));
                if (!ids_match) ids = ref_ids;  // fall back so the rest still exercises

                // (2) prefill assembly: numeric match vs upstream.
                brosoundml::QwenTtsConfig cfg = c;
                std::vector<float> prefill, trailing, tts_pad;
                int gotT = 0, gotL = 0;
                brosoundml::assemble_talker_prefill(
                    talker, cfg, ids, /*instruct_ids=*/{}, spk_id,
                    /*spk_embed=*/nullptr, /*spk_steer=*/nullptr, language_id,
                    prefill, gotT, trailing, gotL, tts_pad);
                CHECK(gotT == T, tag("assembled prefill length matches"));
                CHECK(gotL == L, tag("assembled trailing length matches"));
                auto cmpv = [&](const char* what, const std::vector<float>& got,
                                const std::vector<float>& ref, double tol) {
                    if (got.size() != ref.size()) { CHECK(false, what); return; }
                    double mx = 0.0;
                    for (std::size_t i = 0; i < ref.size(); ++i)
                        mx = std::max(mx, std::fabs((double)got[i] - ref[i]));
                    std::printf("    [synth %s] max|Δ|=%.2e\n", what, mx);
                    CHECK(mx < tol, what);
                };
                if (gotT == T) cmpv("prefill", prefill, ref_prefill, 5e-3);
                if (gotL == L) cmpv("trailing", trailing, ref_trailing, 5e-3);
                cmpv("tts_pad", tts_pad, ref_pad, 5e-3);

                // (3) AR loop with the upstream codebook-0 logits policy.
                std::vector<int32_t> pos3(static_cast<std::size_t>(3) * gotT);
                for (int a = 0; a < 3; ++a)
                    for (int t = 0; t < gotT; ++t) pos3[a * gotT + t] = t;
                brosoundml::QwenTtsGenParams gp;
                gp.eos_id      = c.talker.codec_eos_id;
                gp.max_frames  = F;
                gp.rope_delta  = 0;
                gp.suppress_lo = c.talker.vocab_size - 1024;
                gp.suppress_hi = c.talker.vocab_size;
                gp.min_frames  = 2;
                gp.repetition_penalty = 1.05f;
                std::vector<int32_t> got_frames;
                brosoundml::QwenTtsGenState gen;   // per-stream AR scratch
                const int gotF = brosoundml::generate_codes(
                    talker, cp, gen, prefill.data(), gotT, pos3.data(),
                    trailing.data(), gotL, tts_pad.data(), gp, got_frames);
                CHECK(gotF == F, tag("synth AR loop emits the expected frame count"));
                int sm = 0;
                if (gotF == F)
                    for (std::size_t i = 0; i < ref_frames.size(); ++i)
                        if (got_frames[i] != ref_frames[i]) ++sm;
                std::printf("    [synth AR loop] F=%d  code mismatches=%d/%d\n", F, sm, F * G);
                CHECK(gotF == F, tag("synth AR loop frame count matches"));
                CHECK(sm == 0, tag("synth code stream matches upstream exactly"));

                // Multi-stream (SERIALIZED tier): two independent gen-states over
                // the one shared Talker + Code Predictor, driven interleaved, must
                // each reproduce the single-stream code stream bit-for-bit — the
                // captured graphs and KV caches now live per-stream, so nothing
                // cross-talks. (Serialized = no *concurrent* overlap; serial reuse
                // across streams is exact, which is what this checks.)
                {
                    brosoundml::QwenTtsGenState genA, genB;
                    std::vector<int32_t> fA1, fB, fA2;
                    const int fa1 = brosoundml::generate_codes(
                        talker, cp, genA, prefill.data(), gotT, pos3.data(),
                        trailing.data(), gotL, tts_pad.data(), gp, fA1);
                    const int fb = brosoundml::generate_codes(   // B between A's runs
                        talker, cp, genB, prefill.data(), gotT, pos3.data(),
                        trailing.data(), gotL, tts_pad.data(), gp, fB);
                    const int fa2 = brosoundml::generate_codes(  // A reused
                        talker, cp, genA, prefill.data(), gotT, pos3.data(),
                        trailing.data(), gotL, tts_pad.data(), gp, fA2);
                    CHECK(fa1 == gotF && fA1 == got_frames,
                          tag("session A code stream matches the single-stream run"));
                    CHECK(fb == gotF && fB == got_frames,
                          tag("session B (interleaved) matches — no cross-talk"));
                    CHECK(fa2 == gotF && fA2 == got_frames,
                          tag("session A reused matches (graph replay is stable)"));
                    std::printf("    [synth sessions] 2 gen-states over 1 model, "
                                "interleaved, bit-exact (F=%d)\n", gotF);
                }

                // Sampling path: temperature > 0 routes codebook 0 AND the Code
                // Predictor through the seeded device-counter sampler. Verify it
                // is (a) reproducible for a fixed seed, (b) seed-sensitive, and
                // (c) bit-identical CPU vs CUDA (device-neutral stochastic path).
                {
                    brosoundml::QwenTtsGenParams sp = gp;
                    sp.temperature = 0.8f;
                    sp.top_k       = 50;
                    sp.top_p       = 0.95f;
                    sp.seed        = 1234;

                    brosoundml::QwenTtsGenState gs1, gs2, gs3;
                    std::vector<int32_t> s1, s2, s3;
                    const int f1 = brosoundml::generate_codes(
                        talker, cp, gs1, prefill.data(), gotT, pos3.data(),
                        trailing.data(), gotL, tts_pad.data(), sp, s1);
                    // same seed again -> identical stream (reproducible).
                    const int f2 = brosoundml::generate_codes(
                        talker, cp, gs2, prefill.data(), gotT, pos3.data(),
                        trailing.data(), gotL, tts_pad.data(), sp, s2);
                    CHECK(f1 == f2 && s1 == s2,
                          tag("sampling is reproducible for a fixed seed"));
                    // reusing one gen-state with the same seed -> still identical
                    // (the per-utterance counter reset + stable captured graph).
                    const int f1b = brosoundml::generate_codes(
                        talker, cp, gs1, prefill.data(), gotT, pos3.data(),
                        trailing.data(), gotL, tts_pad.data(), sp, s3);
                    CHECK(f1b == f1 && s3 == s1,
                          tag("sampling reused gen-state reproduces (counter reset)"));
                    // a different seed -> a different stream (sampling is live).
                    brosoundml::QwenTtsGenParams sp2 = sp;
                    sp2.seed = 9876;
                    std::vector<int32_t> sd;
                    brosoundml::generate_codes(
                        talker, cp, gs3, prefill.data(), gotT, pos3.data(),
                        trailing.data(), gotL, tts_pad.data(), sp2, sd);
                    CHECK(sd != s1, tag("sampling responds to the seed"));

                    std::printf("    [synth sampling] F=%d  seed-stable=%d  "
                                "seed-sensitive=%d\n",
                                f1, (int)(s1 == s2), (int)(sd != s1));

                    // CPU vs CUDA must agree exactly on the sampled stream.
                    if (dev == brotensor::Device::CPU) {
                        sampled_ref = s1;
                        sampled_ref_F = f1;
                    } else if (sampled_ref_F >= 0) {
                        CHECK(f1 == sampled_ref_F && s1 == sampled_ref,
                              tag("sampling stream is bit-identical CPU vs CUDA"));
                        std::printf("    [synth sampling] CPU/CUDA parity: %s\n",
                                    (f1 == sampled_ref_F && s1 == sampled_ref)
                                        ? "exact" : "MISMATCH");
                    }
                }

                // (4) the public synthesize() path runs end to end (codes ->
                // 24 kHz). Gated behind the heavy-decode env like the codec
                // fixture, since it runs the full x1920 decode.
                if (std::getenv("BROSOUNDML_RUN_CODEC_FIXTURE")) {
                    QwenTts q2;
                    q2.load(root.string(), dev);   // full pipeline on the active device
                    brosoundml::AudioBuffer wav = q2.synthesize(text, "serena", "english");
                    CHECK(wav.sample_rate == 24000, tag("synthesize() returns 24 kHz"));
                    CHECK(!wav.samples.empty(), tag("synthesize() returns audio"));
                    std::printf("    [%s synthesize] %zu samples (%.2fs)\n",
                                dev_name, wav.samples.size(), wav.samples.size() / 24000.0);

                    // Public multi-voice session API: two sessions over the one
                    // loaded model, interleaved, each reproduce the single-call
                    // waveform bit-for-bit (codec is deterministic over the codes).
                    // GPU-only — the x1920 codec on CPU is far too slow to run
                    // four times here (CPU is not a Qwen synthesis target).
                    if (dev != brotensor::Device::CPU) {
                    brosoundml::QwenTtsSession sA = q2.make_session();
                    brosoundml::QwenTtsSession sB = q2.make_session();
                    auto wA1 = q2.synthesize(sA, text, "serena", "english");
                    auto wB  = q2.synthesize(sB, text, "serena", "english");
                    auto wA2 = q2.synthesize(sA, text, "serena", "english");
                    CHECK(wA1.samples == wav.samples,
                          tag("session synthesize matches the single-call waveform"));
                    CHECK(wB.samples == wav.samples,
                          tag("interleaved session waveform matches — no cross-talk"));
                    CHECK(wA2.samples == wA1.samples,
                          tag("reused session waveform is stable"));
                    q2.reset(sA);
                    auto wA3 = q2.synthesize(sA, text, "serena", "english");
                    CHECK(wA3.samples == wav.samples, tag("session after reset matches"));
                    std::printf("    [%s sessions] 2 QwenTtsSession over 1 model, "
                                "bit-exact, no cross-talk\n", dev_name);
                    }
                }
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

    // ─── Base speaker encoder (ECAPA-TDNN x-vector) vs ground-truth fixture ──
    //
    // White-box check of the zero-shot voice clone's enrollment front end: the
    // mel frontend + the ECAPA-TDNN speaker encoder, built directly from the
    // Base checkpoint and compared to the genuine upstream extract_speaker_embedding
    // (.scratch/gen_speaker_fixture.py). The mel must match tightly (STFT +
    // librosa slaney basis are deterministic); the embedding, after a deep
    // conv/pooling net, is checked by cosine similarity (the x-vector is used as
    // a direction in the Talker prefill). Opt-in + skipped when absent.
    if (std::getenv("BROSOUNDML_RUN_CODEC_FIXTURE")) {
        const fs::path base = fs::path(BROSOUNDML_REPO_DIR) / "weights" /
                              "qwen-tts" / "0.6B-Base";
        const fs::path fix_dir = fs::path(BROSOUNDML_REPO_DIR) / "tests" / "fixtures";
        if (fs::exists(base / "model.safetensors")) {
            brosoundml::QwenTtsSpeakerEncoderConfig sc;
            sc.present = true;
            sc.enc_channels     = {512, 512, 512, 512, 1536};
            sc.enc_kernel_sizes = {5, 3, 3, 3, 1};
            sc.enc_dilations    = {1, 2, 3, 4, 1};
            auto w = brotensor::safetensors::File::open((base / "model.safetensors").string());
            brosoundml::QwenTtsSpeakerEncoder enc;
            enc.load(w, sc);

            auto check_speaker = [&](const char* fname) {
                std::ifstream f((fix_dir / fname).string(), std::ios::binary);
                if (!f) return;  // not generated locally — skip
                int32_t hdr[4];
                f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
                const int n = hdr[0], M = hdr[1], F = hdr[2], D = hdr[3];
                std::vector<float> wav(n), mel(static_cast<std::size_t>(M) * F),
                                   emb(D);
                f.read(reinterpret_cast<char*>(wav.data()), n * 4);
                f.read(reinterpret_cast<char*>(mel.data()),
                       static_cast<std::streamsize>(mel.size()) * 4);
                f.read(reinterpret_cast<char*>(emb.data()), D * 4);
                CHECK(static_cast<bool>(f), "speaker fixture read complete");

                // mel front-end
                std::vector<float> got_mel; int gf = 0;
                enc.mel(wav.data(), n, got_mel, gf);
                CHECK(gf == F, "speaker mel frame count matches upstream");
                double mel_max = 0.0;
                if (gf == F)
                    for (std::size_t i = 0; i < mel.size(); ++i)
                        mel_max = std::max(mel_max, std::fabs(double(got_mel[i]) - mel[i]));

                // embedding (cosine similarity + max abs diff)
                std::vector<float> got = enc.embed(wav.data(), n);
                CHECK(static_cast<int>(got.size()) == D, "speaker embedding width matches");
                double dot = 0, na = 0, nb = 0, emax = 0;
                if (static_cast<int>(got.size()) == D)
                    for (int i = 0; i < D; ++i) {
                        dot += double(got[i]) * emb[i];
                        na  += double(got[i]) * got[i];
                        nb  += double(emb[i]) * emb[i];
                        emax = std::max(emax, std::fabs(double(got[i]) - emb[i]));
                    }
                const double cos = dot / (std::sqrt(na * nb) + 1e-12);
                std::printf("    [speaker fixture %s] mel=(%d,%d) mel_max|d|=%.2e  "
                            "emb cos=%.6f max|d|=%.3e\n", fname, M, F, mel_max, cos, emax);
                CHECK(mel_max < 1e-3, "speaker mel matches upstream front end");
                CHECK(cos > 0.999, "speaker embedding matches upstream (cosine)");
            };
            check_speaker("qwen_tts_speaker_small.bin");
            check_speaker("qwen_tts_speaker.bin");

            // ─── Base zero-shot voice clone (x-vector) vs ground-truth ──────
            //
            // End-to-end check of synthesize_clone()'s deterministic core: the
            // x-vector splice into the Talker prefill + the greedy AR loop, vs
            // the genuine upstream Base clone (tests/ref/gen_qwen_tts_clone_fixture.py).
            // The reference clip is encoded by the C++ ECAPA encoder (above), so
            // this exercises the whole enroll -> prefill -> generate path. Builds
            // the Talker / Code Predictor white-box on CPU (the greedy stream is
            // CPU==CUDA identical, as the synth fixture establishes).
            std::ifstream cf((fix_dir / "qwen_tts_clone.bin").string(), std::ios::binary);
            if (cf) {
                auto rd = [&](void* p, std::size_t b) {
                    cf.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(b));
                };
                int32_t tlen = 0; rd(&tlen, 4);
                std::string text(tlen, '\0'); rd(text.data(), tlen);
                int32_t hh[2]; rd(hh, sizeof(hh));
                const int H = hh[0], n_ids = hh[1];
                std::vector<int32_t> ref_ids(n_ids); rd(ref_ids.data(), n_ids * 4);
                int32_t language_id = 0; rd(&language_id, 4);
                int32_t enc_dim = 0; rd(&enc_dim, 4);
                std::vector<float> ref_xvec(enc_dim); rd(ref_xvec.data(), enc_dim * 4);
                int32_t n_ref = 0; rd(&n_ref, 4);
                std::vector<float> ref_wav(n_ref); rd(ref_wav.data(), n_ref * 4);
                int32_t T = 0; rd(&T, 4);
                std::vector<float> ref_prefill(static_cast<std::size_t>(T) * H);
                rd(ref_prefill.data(), ref_prefill.size() * 4);
                int32_t L = 0; rd(&L, 4);
                std::vector<float> ref_trailing(static_cast<std::size_t>(L) * H);
                rd(ref_trailing.data(), ref_trailing.size() * 4);
                std::vector<float> ref_pad(H); rd(ref_pad.data(), H * 4);
                int32_t fg[2]; rd(fg, sizeof(fg));
                const int F = fg[0], G = fg[1];
                std::vector<int32_t> ref_frames(static_cast<std::size_t>(F) * G);
                rd(ref_frames.data(), ref_frames.size() * 4);
                CHECK(static_cast<bool>(cf), "clone fixture read complete");

                // Config via a real Base load (also exercises the Base load path).
                QwenTts qb;
                qb.load(base.string(), brotensor::Device::CPU);
                const auto& c = qb.config();
                CHECK(c.variant == QwenTtsVariant::Base, "clone: variant == Base");
                CHECK(c.speaker_encoder.present, "clone: Base ships a speaker encoder");

                brosoundml::QwenTtsTalker talker;
                brosoundml::QwenTtsCodePredictor cp;
                talker.load(w, c.talker, brotensor::Device::CPU);
                cp.load(w, c.talker.code_predictor, c.talker.transformer.hidden_size,
                        brotensor::Device::CPU);
                auto tok = brolm::qwen::Tokenizer::load(
                    (base / "vocab.json").string(), (base / "merges.txt").string());
                tok.register_special_token("<|im_start|>", c.im_start_id);
                tok.register_special_token("<|im_end|>",   c.im_end_id);
                std::vector<int32_t> ids = tok.encode(
                    "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n");
                bool ids_match = (ids.size() == ref_ids.size());
                for (std::size_t i = 0; ids_match && i < ids.size(); ++i)
                    ids_match = (ids[i] == ref_ids[i]);
                CHECK(ids_match, "clone: tokenizer reproduces upstream ids");
                if (!ids_match) ids = ref_ids;

                // my x-vector from the reference clip (cosine vs upstream)
                std::vector<float> xv = enc.embed(ref_wav.data(), n_ref);
                double dot = 0, na = 0, nb = 0;
                for (int i = 0; i < enc_dim; ++i) {
                    dot += double(xv[i]) * ref_xvec[i];
                    na += double(xv[i]) * xv[i]; nb += double(ref_xvec[i]) * ref_xvec[i];
                }
                CHECK(dot / (std::sqrt(na * nb) + 1e-12) > 0.999,
                      "clone: x-vector matches upstream (cosine)");

                // prefill assembly with the x-vector spliced into the speaker slot
                brosoundml::QwenTtsConfig cfg = c;
                std::vector<float> prefill, trailing, tts_pad;
                int gotT = 0, gotL = 0;
                brosoundml::assemble_talker_prefill(
                    talker, cfg, ids, /*instruct_ids=*/{}, /*spk_id=*/-1,
                    xv.data(), /*spk_steer=*/nullptr, language_id,
                    prefill, gotT, trailing, gotL, tts_pad);
                CHECK(gotT == T, "clone: prefill length matches");
                CHECK(gotL == L, "clone: trailing length matches");
                auto cmpc = [&](const char* what, const std::vector<float>& got,
                                const std::vector<float>& ref, double tol) {
                    if (got.size() != ref.size()) { CHECK(false, what); return; }
                    double mx = 0.0;
                    for (std::size_t i = 0; i < ref.size(); ++i)
                        mx = std::max(mx, std::fabs(double(got[i]) - ref[i]));
                    std::printf("    [clone %s] max|d|=%.2e\n", what, mx);
                    CHECK(mx < tol, what);
                };
                if (gotT == T) cmpc("prefill", prefill, ref_prefill, 5e-3);
                if (gotL == L) cmpc("trailing", trailing, ref_trailing, 5e-3);

                // greedy AR loop -> frames must match upstream exactly
                std::vector<int32_t> pos3(static_cast<std::size_t>(3) * gotT);
                for (int a = 0; a < 3; ++a)
                    for (int t = 0; t < gotT; ++t) pos3[a * gotT + t] = t;
                brosoundml::QwenTtsGenParams gp;
                gp.eos_id = c.talker.codec_eos_id;
                gp.max_frames = F;
                gp.suppress_lo = c.talker.vocab_size - 1024;
                gp.suppress_hi = c.talker.vocab_size;
                gp.min_frames = 2;
                gp.repetition_penalty = 1.05f;
                std::vector<int32_t> got_frames;
                brosoundml::QwenTtsGenState gen;   // per-stream AR scratch
                const int gotF = brosoundml::generate_codes(
                    talker, cp, gen, prefill.data(), gotT, pos3.data(),
                    trailing.data(), gotL, tts_pad.data(), gp, got_frames);
                int sm = 0;
                if (gotF == F)
                    for (std::size_t i = 0; i < ref_frames.size(); ++i)
                        if (got_frames[i] != ref_frames[i]) ++sm;
                std::printf("    [clone AR loop] F=%d code mismatches=%d/%d\n", F, sm, F * G);
                CHECK(gotF == F, "clone: AR frame count matches upstream");
                CHECK(sm == 0, "clone: code stream matches upstream exactly");

                // public synthesize_clone() runs end to end (codes -> 24 kHz)
                brosoundml::AudioBuffer rwav(std::vector<float>(
                    ref_wav.begin(), ref_wav.end()), 24000);
                brosoundml::AudioBuffer out = qb.synthesize_clone(text, rwav, "english");
                CHECK(out.sample_rate == 24000, "clone: synthesize_clone returns 24 kHz");
                CHECK(!out.samples.empty(), "clone: synthesize_clone returns audio");
                std::printf("    [synthesize_clone] %zu samples (%.2fs)\n",
                            out.samples.size(), out.samples.size() / 24000.0);
            }
        }
    }

    if (failures == 0) {
        std::printf("test_qwen_tts: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_qwen_tts: %d check(s) failed\n", failures);
    return 1;
}
