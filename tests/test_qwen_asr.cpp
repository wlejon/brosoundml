// Qwen3-ASR pipeline validation.
//
// Contract checks run always. The numeric blocks need the real weights
// (scripts/download-qwen-asr.sh) plus the ground-truth fixture generated from
// the genuine upstream model (tests/ref/gen_qwen_asr_fixture.py); they skip
// silently when either is absent. Every boundary the fixture captures is
// compared — log-mel features, encoder output, prefill logits, and the greedy
// token stream — on CPU and (when available) CUDA.

#include "brosoundml/qwen_asr.h"

#include "qwen_asr_decoder.h"   // internal: white-box prefill-logits check
#include "qwen_asr_encoder.h"   // internal: white-box mel / encoder checks
#include "qwen_tts_device.h"    // internal: qtd:: gather/linear helpers

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
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
namespace bt = brotensor;

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

// ─── Fixture (see gen_qwen_asr_fixture.py for the layout) ──────────────────

struct Fixture {
    std::vector<float>        audio;
    int                       frames = 0, n_mels = 0;
    std::vector<float>        mel;          // frame-major (frames, n_mels)
    int                       n_audio = 0, enc_dim = 0;
    std::vector<float>        enc;
    std::vector<std::int32_t> prompt_ids;
    std::vector<float>        logits;       // prefill, last position
    std::vector<std::int32_t> gen_ids;
};

static bool read_fixture(const fs::path& path, Fixture& fx) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::int32_t hdr[8];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    const std::int32_t n_samples = hdr[0];
    fx.frames  = hdr[1];
    fx.n_mels  = hdr[2];
    fx.n_audio = hdr[3];
    fx.enc_dim = hdr[4];
    const std::int32_t prompt_len = hdr[5];
    const std::int32_t vocab      = hdr[6];
    const std::int32_t n_gen      = hdr[7];
    auto rd_f = [&f](std::vector<float>& v, std::size_t n) {
        v.resize(n);
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(n * sizeof(float)));
    };
    auto rd_i = [&f](std::vector<std::int32_t>& v, std::size_t n) {
        v.resize(n);
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(n * sizeof(std::int32_t)));
    };
    rd_f(fx.audio, static_cast<std::size_t>(n_samples));
    rd_f(fx.mel, static_cast<std::size_t>(fx.frames) * fx.n_mels);
    rd_f(fx.enc, static_cast<std::size_t>(fx.n_audio) * fx.enc_dim);
    rd_i(fx.prompt_ids, static_cast<std::size_t>(prompt_len));
    rd_f(fx.logits, static_cast<std::size_t>(vocab));
    rd_i(fx.gen_ids, static_cast<std::size_t>(n_gen));
    return static_cast<bool>(f);
}

// max / mean absolute difference between a device tensor and host expeconst.
static void diff_stats(const bt::Tensor& got, const float* want,
                       std::size_t n, float& max_d, float& mean_d) {
    bt::Tensor host = (got.device == bt::Device::CPU)
                          ? got : got.to(bt::Device::CPU);
    const float* g = host.host_f32();
    double sum = 0.0;
    max_d = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float d = std::fabs(g[i] - want[i]);
        if (d > max_d) max_d = d;
        sum += d;
    }
    mean_d = static_cast<float>(sum / static_cast<double>(n ? n : 1));
}

static int run();
int main() {
    brotensor::init();
    try { return run(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "test_qwen_asr: uncaught exception: %s\n", e.what());
        return 2;
    }
    catch (...) {
        std::fprintf(stderr, "test_qwen_asr: uncaught non-std exception\n");
        return 2;
    }
}

static int run() {
    using brosoundml::AudioBuffer;
    using brosoundml::QwenAsr;

    // ─── Contract checks (no weights needed) ───────────────────────────────
    {
        QwenAsr a;
        CHECK(!a.loaded(), "a fresh QwenAsr is not loaded");
        CHECK(a.config().sample_rate == 16000, "default config sample rate is 16 kHz");
        CHECK(throws_runtime_error([&] { a.load("nonexistent-qwen-asr-dir"); }),
              "load() on a missing directory throws");
        CHECK(throws_runtime_error([&] {
                  AudioBuffer buf(std::vector<float>(16000, 0.0f), 16000);
                  a.transcribe(buf);
              }),
              "transcribe() before load() throws");
    }

    // ─── Fixture-driven parity vs the genuine upstream model ──────────────
    const fs::path root(BROSOUNDML_REPO_DIR);
    const fs::path model_dir = root / "weights" / "qwen-asr" / "0.6B";
    const fs::path fix_path  = root / "tests" / "fixtures" / "qwen_asr.bin";
    if (!fs::exists(model_dir / "model.safetensors") || !fs::exists(fix_path)) {
        std::printf("[skip] qwen-asr weights or fixture absent "
                    "(scripts/download-qwen-asr.sh + tests/ref/gen_qwen_asr_fixture.py)\n");
        return failures == 0 ? 0 : 1;
    }

    Fixture fx;
    CHECK(read_fixture(fix_path, fx), "fixture read complete");
    if (failures) return 1;

    auto run_device = [&](bt::Device dev, const char* dev_name) {
        QwenAsr asr;
        asr.load(model_dir.string(), dev);
        CHECK(asr.loaded(), "load() succeeds");
        CHECK(asr.config().vocab_size == static_cast<int>(fx.logits.size()),
              "config vocab matches fixture");

        const AudioBuffer audio(fx.audio, 16000);

        // White-box stages (internal modules re-loaded on the same device —
        // cheap next to the transcribe below, keeps QwenAsr's internals
        // private).
        auto f = bt::safetensors::File::open((model_dir / "model.safetensors").string());
        brosoundml::QwenAsrEncoder enc;
        enc.load(f, asr.config(), dev);

        // 1. log-mel features.
        {
            int frames = 0;
            const std::vector<float> mel = enc.log_mel(audio, frames);
            CHECK(frames == fx.frames, "mel frame count matches fixture");
            if (frames == fx.frames) {
                float max_d = 0.0f, mean_d = 0.0f;
                double sum = 0.0;
                for (std::size_t i = 0; i < mel.size(); ++i) {
                    const float d = std::fabs(mel[i] - fx.mel[i]);
                    if (d > max_d) max_d = d;
                    sum += d;
                }
                mean_d = static_cast<float>(sum / mel.size());
                std::printf("    [%s mel]    max|Δ|=%.2e  mean|Δ|=%.2e\n",
                            dev_name, max_d, mean_d);
                CHECK(max_d < 1e-4f, "log-mel matches the official front-end");
            }
        }

        // 2. encoder output.
        {
            bt::Tensor out;
            enc.forward(audio, out);
            CHECK(out.rows == fx.n_audio, "encoder token count matches fixture");
            CHECK(out.cols == fx.enc_dim, "encoder width matches fixture");
            if (out.rows == fx.n_audio && out.cols == fx.enc_dim) {
                float max_d = 0.0f, mean_d = 0.0f;
                diff_stats(out, fx.enc.data(), fx.enc.size(), max_d, mean_d);
                std::printf("    [%s enc]    max|Δ|=%.2e  mean|Δ|=%.2e\n",
                            dev_name, max_d, mean_d);
                CHECK(max_d < 2e-2f, "encoder output matches upstream");
            }

            // The public latent tap must be the SAME path as the white-box
            // encoder (encode() just wraps it) — bit-exact, not merely close —
            // and carry the documented geometry.
            CHECK(asr.config().latent_dim == fx.enc_dim,
                  "config.latent_dim == encoder width");
            CHECK(std::fabs(asr.config().latent_hz - 12.5f) < 1e-4f,
                  "config.latent_hz == 12.5");
            bt::Tensor lat = asr.encode(audio);
            CHECK(lat.rows == fx.n_audio && lat.cols == asr.config().latent_dim,
                  "encode() shape == (n_audio, latent_dim)");
            if (lat.rows == out.rows && lat.cols == out.cols) {
                float max_d = 0.0f, mean_d = 0.0f;
                bt::Tensor out_host = (out.device == bt::Device::CPU)
                                          ? out : out.to(bt::Device::CPU);
                diff_stats(lat, out_host.host_f32(), out_host.size(),
                           max_d, mean_d);
                CHECK(max_d == 0.0f, "encode() bit-matches the encoder forward");
            }

            // Host-copy accessor returns the same rows on the host.
            std::vector<float> host;
            const int T = asr.encode_to_host(audio, host);
            CHECK(T == fx.n_audio, "encode_to_host row count");
            CHECK(host.size() ==
                      static_cast<std::size_t>(T) * asr.config().latent_dim,
                  "encode_to_host buffer size");
            if (T == out.rows) {
                float max_d = 0.0f, mean_d = 0.0f;
                diff_stats(lat, host.data(), host.size(), max_d, mean_d);
                CHECK(max_d == 0.0f, "encode_to_host matches encode()");
            }
        }

        // 3. prefill logits at the last prompt position.
        {
            brosoundml::QwenAsrDecoder dec;
            dec.load(f, asr.config(), dev);

            bt::Tensor audio_emb;
            enc.forward(audio, audio_emb);

            // Assemble the fixture's prompt embedding stream: text embeddings
            // everywhere, encoder rows over the <|audio_pad|> block.
            const int P = static_cast<int>(fx.prompt_ids.size());
            const int H = asr.config().hidden_size;
            const int audio_id = asr.config().audio_token_id;
            bt::DeviceScope scope(dev);
            bt::Tensor embeds = brosoundml::qtd::gather_rows(dec.embed_tokens,
                                                             fx.prompt_ids);
            int audio_at = -1;
            for (int i = 0; i < P; ++i) {
                if (fx.prompt_ids[static_cast<std::size_t>(i)] == audio_id) {
                    audio_at = i;
                    break;
                }
            }
            CHECK(audio_at >= 0, "fixture prompt contains the audio block");
            bt::copy_d2d(audio_emb, 0, embeds, audio_at * H, fx.n_audio * H);

            brosoundml::QwenAsrDecoderCache cache;
            cache.reset(dec.num_layers);
            bt::Tensor hidden, logits;
            dec.run_dev(embeds, P, &cache, hidden);
            dec.logits_last(hidden, logits);

            float max_d = 0.0f, mean_d = 0.0f;
            diff_stats(logits, fx.logits.data(), fx.logits.size(), max_d, mean_d);
            std::printf("    [%s logits] max|Δ|=%.2e  mean|Δ|=%.2e\n",
                        dev_name, max_d, mean_d);
            CHECK(max_d < 5e-2f, "prefill logits match upstream");
        }

        // 4. end-to-end greedy transcription (token-exact).
        {
            QwenAsr::TranscribeOptions opts;
            opts.max_new_tokens = static_cast<int>(fx.gen_ids.size()) + 8;
            const auto res = asr.transcribe(audio, opts);
            std::printf("    [%s tokens] got %zu, want %zu\n", dev_name,
                        res.token_ids.size(), fx.gen_ids.size());
            CHECK(res.token_ids == fx.gen_ids,
                  "greedy token stream matches upstream");
            if (res.token_ids != fx.gen_ids) {
                const std::size_t n =
                    std::min(res.token_ids.size(), fx.gen_ids.size());
                for (std::size_t i = 0; i < n; ++i) {
                    if (res.token_ids[i] != fx.gen_ids[i]) {
                        std::fprintf(stderr,
                                     "    first divergence at %zu: got %d want %d\n",
                                     i, res.token_ids[i], fx.gen_ids[i]);
                        break;
                    }
                }
            }
        }
    };

    std::printf("  [cpu]\n");
    run_device(bt::Device::CPU, "cpu");
    if (bt::is_available(bt::Device::CUDA)) {
        std::printf("  [cuda]\n");
        run_device(bt::Device::CUDA, "cuda");
    }

    if (failures) {
        std::fprintf(stderr, "test_qwen_asr: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_qwen_asr: all checks passed\n");
    return 0;
}
