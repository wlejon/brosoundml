#define _CRT_SECURE_NO_WARNINGS
// HiggsAudio v2 codec (OmniVoice audio_tokenizer) tests.
//
//   1. Loader contract — a fresh HiggsCodec, config fields after load(),
//      decode() / encode() throwing before load and on bad shapes.
//   2. Decoder vs the reference fixture (tests/fixtures/higgs_codec.bin, made
//      by tests/ref/gen_higgs_codec_fixture.py from transformers'
//      HiggsAudioV2TokenizerModel in FP32 on CUDA): max |Δ| on the waveform.
//   3. Encoder vs the fixture: per-codebook exact-match rate of the RVQ codes,
//      plus every intermediate stage (the 16 kHz semantic input, HuBERT's
//      mean hidden state, the semantic-encoder output, the acoustic latent,
//      the fc output) so a mismatch localises. The stages are checked with the
//      reference's own 16 kHz input substituted for brosoundml's resample,
//      which isolates the model from the resampler; the code agreement is
//      reported both ways, and the resampler itself against torchaudio's.
//   4. CPU vs CUDA parity (when brotensor reports CUDA): waveform + codes.
//   5. decoder_only = true: loads without the encoder, decode still matches.
//   6. Round trip on the test clip: encode -> decode -> Whisper (weights/whisper,
//      when present) transcript contains "test".
//
// Real-weights sections skip (printing why) when weights/omnivoice/
// audio_tokenizer or the fixture is absent.
#include "brosoundml/audio.h"
#include "brosoundml/higgs_codec.h"
#include "brosoundml/whisper.h"

#include "higgs_codec_model.h"   // internal: the staged encoder trace

#include <brolm/whisper_tokenizer.h>

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cctype>
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

// ─── fixture ────────────────────────────────────────────────────────────────

struct FixtureCase {
    int n24 = 0, n16 = 0, Th = 0, T = 0, K = 0, n_out = 0;
    std::vector<float> input, sem16, hubert, sem_enc, acoustic, fc_out, decoded;
    std::vector<int32_t> codes;
};

static bool read_fixture(const fs::path& path, std::vector<FixtureCase>& cases) {
    std::ifstream f(path.string(), std::ios::binary);
    if (!f) return false;
    int32_t hdr[2];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!f || hdr[0] != 0x31434748) return false;
    cases.resize(static_cast<std::size_t>(hdr[1]));
    auto rd_f = [&](std::vector<float>& v, std::size_t n) {
        v.resize(n);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
    };
    for (FixtureCase& c : cases) {
        int32_t h[6];
        f.read(reinterpret_cast<char*>(h), sizeof(h));
        c.n24 = h[0]; c.n16 = h[1]; c.Th = h[2]; c.T = h[3]; c.K = h[4]; c.n_out = h[5];
        rd_f(c.input, static_cast<std::size_t>(c.n24));
        rd_f(c.sem16, static_cast<std::size_t>(c.n16));
        rd_f(c.hubert, static_cast<std::size_t>(c.Th) * 768);
        rd_f(c.sem_enc, static_cast<std::size_t>(768) * c.T);
        rd_f(c.acoustic, static_cast<std::size_t>(256) * c.T);
        rd_f(c.fc_out, static_cast<std::size_t>(1024) * c.T);
        c.codes.resize(static_cast<std::size_t>(c.K) * c.T);
        f.read(reinterpret_cast<char*>(c.codes.data()),
               static_cast<std::streamsize>(c.codes.size() * sizeof(int32_t)));
        rd_f(c.decoded, static_cast<std::size_t>(c.n_out));
    }
    return static_cast<bool>(f);
}

// max |a-b|, mean |a-b|, and max|b| (the reference scale) over n entries.
struct Diff { double max_abs = 0, mean_abs = 0, ref_max = 0; };
static Diff diff(const float* a, const float* b, std::size_t n) {
    Diff d;
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double e = std::fabs(static_cast<double>(a[i]) - b[i]);
        d.max_abs = std::max(d.max_abs, e);
        sum += e;
        d.ref_max = std::max(d.ref_max, std::fabs(static_cast<double>(b[i])));
    }
    d.mean_abs = n ? sum / static_cast<double>(n) : 0.0;
    return d;
}

// Per-codebook exact-match rates of two K x T code streams; returns the total.
static double code_agreement(const std::vector<int32_t>& a, const std::vector<int32_t>& b,
                             int K, int T, std::vector<double>& per_cb) {
    per_cb.assign(static_cast<std::size_t>(K), 0.0);
    long total = 0;
    for (int k = 0; k < K; ++k) {
        long m = 0;
        for (int t = 0; t < T; ++t) {
            const std::size_t i = static_cast<std::size_t>(k) * T + t;
            if (a[i] == b[i]) ++m;
        }
        per_cb[k] = static_cast<double>(m) / T;
        total += m;
    }
    return static_cast<double>(total) / (static_cast<double>(K) * T);
}

static void print_rates(const char* label, const std::vector<double>& per_cb, double total) {
    std::printf("      %-28s total=%.4f  per-codebook:", label, total);
    for (double r : per_cb) std::printf(" %.3f", r);
    std::printf("\n");
}

// Results kept from the CPU pass for the CUDA parity check.
struct DeviceResults {
    std::vector<std::vector<float>>   wav;    // decode(fixture codes) per case
    std::vector<std::vector<int32_t>> codes;  // encode(fixture input) per case
};

static void run_device(brotensor::Device dev, const char* dev_name, const std::string& dir,
                       const std::vector<FixtureCase>& cases, DeviceResults& out,
                       const DeviceResults* cpu) {
    using brosoundml::HiggsCodec;
    auto tag = [&](const char* m) {
        static std::string s;
        s = std::string("[") + dev_name + "] " + m;
        return s.c_str();
    };
    std::printf("  [%s] loading %s\n", dev_name, dir.c_str());

    HiggsCodec codec;
    codec.load(dir, dev);
    CHECK(codec.loaded(), tag("load() succeeds on the real weights"));
    CHECK(codec.has_encoder(), tag("full load has the encoder"));
    CHECK(codec.device() == dev, tag("device() reports the load device"));
    const brosoundml::HiggsCodecConfig& cfg = codec.config();
    CHECK(cfg.sample_rate == 24000, tag("config: sample_rate 24000"));
    CHECK(cfg.hop_length == 960, tag("config: hop_length 960"));
    CHECK(cfg.frame_rate == 25, tag("config: frame_rate 25"));
    CHECK(cfg.num_quantizers == 8, tag("config: 8 quantizers"));
    CHECK(cfg.codebook_size == 1024, tag("config: codebook_size 1024"));
    CHECK(cfg.codebook_dim == 64, tag("config: codebook_dim 64"));
    CHECK(cfg.hidden_size == 1024, tag("config: hidden_size 1024"));
    CHECK(cfg.acoustic_dim == 256, tag("config: acoustic_dim 256"));
    CHECK(cfg.semantic_dim == 768, tag("config: semantic_dim 768"));
    CHECK(cfg.semantic_sample_rate == 16000, tag("config: semantic_sample_rate 16000"));
    CHECK(cfg.semantic_downsample_factor == 2, tag("config: semantic_downsample_factor 2"));
    CHECK(cfg.encoder_dim == 64 && cfg.decoder_dim == 1024, tag("config: encoder/decoder dims"));
    CHECK((cfg.decoder_rates == std::vector<int>{8, 5, 4, 2, 3}), tag("config: decoder_rates {8,5,4,2,3}"));
    CHECK((cfg.encoder_rates == std::vector<int>{8, 5, 4, 2, 3}), tag("config: encoder_rates {8,5,4,2,3}"));
    CHECK(cfg.has_encoder, tag("config: has_encoder"));

    // ── bad shapes ──
    {
        std::vector<int32_t> one(8, 0);
        CHECK(throws_runtime_error([&] { codec.decode(one.data(), 9, 1); }),
              tag("decode() with too many quantizers throws"));
        CHECK(throws_runtime_error([&] { codec.decode(one.data(), 1, 0); }),
              tag("decode() with zero frames throws"));
        CHECK(throws_runtime_error([&] { codec.decode(one, 4, 1); }),
              tag("decode(vector) with a size mismatch throws"));
        std::vector<int32_t> oob(8, 1024);
        CHECK(throws_runtime_error([&] { codec.decode(oob, 8, 1); }),
              tag("decode() with an out-of-range code throws"));
        CHECK(throws_runtime_error([&] { codec.encode(brosoundml::AudioBuffer{}); }),
              tag("encode() of empty audio throws"));
        // A single frame of zeros decodes to exactly hop_length samples.
        std::vector<int32_t> z(8, 0);
        brosoundml::AudioBuffer a = codec.decode(z, 8, 1);
        CHECK(a.sample_rate == 24000, tag("decode() returns 24 kHz"));
        CHECK(a.samples.size() == 960, tag("decode() of one frame returns hop_length samples"));
        // Fewer levels is allowed (coarser audio).
        brosoundml::AudioBuffer a2 = codec.decode(z.data(), 3, 1);
        CHECK(a2.samples.size() == 960, tag("decode() with 3 of 8 levels works"));
    }

    if (cases.empty()) {
        std::printf("  [%s] fixture absent — numeric checks skipped "
                    "(python3 tests/ref/gen_higgs_codec_fixture.py)\n", dev_name);
        return;
    }

    // White-box model for the staged encoder trace.
    brosoundml::HiggsCodecModel wm;
    wm.load(dir, dev, /*decoder_only=*/false);

    out.wav.assign(cases.size(), {});
    out.codes.assign(cases.size(), {});
    for (std::size_t ci = 0; ci < cases.size(); ++ci) {
        const FixtureCase& c = cases[ci];
        std::printf("    case %zu: n24=%d T=%d\n", ci, c.n24, c.T);

        // ── decoder vs fixture ──
        {
            brosoundml::AudioBuffer wav = codec.decode(c.codes, c.K, c.T);
            CHECK(static_cast<int>(wav.samples.size()) == c.n_out, tag("decode() returns T*hop samples"));
            if (static_cast<int>(wav.samples.size()) == c.n_out) {
                const Diff d = diff(wav.samples.data(), c.decoded.data(), c.decoded.size());
                std::printf("      decode  max|Δ|=%.3e mean|Δ|=%.3e (ref peak %.3f)\n",
                            d.max_abs, d.mean_abs, d.ref_max);
                CHECK(d.max_abs < 1e-3, tag("decode matches the reference waveform (max abs)"));
                CHECK(d.mean_abs < 1e-4, tag("decode matches the reference waveform (mean abs)"));
            }
            out.wav[ci] = std::move(wav.samples);
        }

        // ── encoder vs fixture (public path: brosoundml's own resampler) ──
        {
            int T = 0;
            brosoundml::AudioBuffer in(c.input, 24000);
            std::vector<int32_t> codes = codec.encode(in, &T);
            CHECK(T == c.T, tag("encode() frame count matches the reference"));
            CHECK(static_cast<int>(codes.size()) == c.K * c.T, tag("encode() returns K*T codes"));
            if (T == c.T && static_cast<int>(codes.size()) == c.K * c.T) {
                std::vector<double> per;
                const double total = code_agreement(codes, c.codes, c.K, c.T, per);
                print_rates("codes (own resample)", per, total);
                CHECK(per[0] >= 0.99, tag("codebook 0 agrees with the reference (>=99%, own resample)"));
                CHECK(total >= 0.97, tag("codes agree with the reference (>=97%, own resample)"));
            }
            out.codes[ci] = std::move(codes);
        }

        // ── staged trace with the reference's 16 kHz semantic input ──
        {
            brosoundml::HiggsEncodeTrace tr;
            int T = 0;
            std::vector<int32_t> codes = wm.encode(c.input.data(), c.n24, &T, &tr,
                                                   c.sem16.data(), c.n16);
            CHECK(T == c.T && tr.Th == c.Th, tag("staged encode: frame counts match"));
            if (T == c.T && tr.Th == c.Th) {
                const Diff dh = diff(tr.hubert.data(), c.hubert.data(), c.hubert.size());
                const Diff ds = diff(tr.sem_enc.data(), c.sem_enc.data(), c.sem_enc.size());
                const Diff da = diff(tr.acoustic.data(), c.acoustic.data(), c.acoustic.size());
                const Diff de = diff(tr.fc_out.data(), c.fc_out.data(), c.fc_out.size());
                std::printf("      hubert   max|Δ|=%.3e mean|Δ|=%.3e (ref max %.3f)\n", dh.max_abs, dh.mean_abs, dh.ref_max);
                std::printf("      sem_enc  max|Δ|=%.3e mean|Δ|=%.3e (ref max %.3f)\n", ds.max_abs, ds.mean_abs, ds.ref_max);
                std::printf("      acoustic max|Δ|=%.3e mean|Δ|=%.3e (ref max %.3f)\n", da.max_abs, da.mean_abs, da.ref_max);
                std::printf("      fc_out   max|Δ|=%.3e mean|Δ|=%.3e (ref max %.3f)\n", de.max_abs, de.mean_abs, de.ref_max);
                CHECK(dh.max_abs < 1e-3 * std::max(1.0, dh.ref_max), tag("HuBERT mean hidden matches (1e-3 rel)"));
                CHECK(ds.max_abs < 1e-3 * std::max(1.0, ds.ref_max), tag("semantic encoder matches (1e-3 rel)"));
                CHECK(da.max_abs < 1e-3 * std::max(1.0, da.ref_max), tag("acoustic latent matches (1e-3 rel)"));
                CHECK(de.max_abs < 1e-3 * std::max(1.0, de.ref_max), tag("fc output matches (1e-3 rel)"));
                std::vector<double> per;
                const double total = code_agreement(codes, c.codes, c.K, c.T, per);
                print_rates("codes (reference resample)", per, total);
                CHECK(per[0] >= 0.99, tag("codebook 0 agrees with the reference (>=99%, reference resample)"));
                CHECK(total >= 0.90, tag("codes agree with the reference (>=90%, reference resample)"));
            }
            // The resampler alone: brosoundml's windowed-sinc 24 -> 16 kHz
            // (pad1d + strided conv1d) vs torchaudio's.
            brosoundml::HiggsEncodeTrace tr2;
            wm.encode(c.input.data(), c.n24, &T, &tr2);
            CHECK(tr2.n16 == c.n16, tag("own resample yields the reference sample count"));
            if (tr2.n16 == c.n16) {
                const Diff dr = diff(tr2.sem16.data(), c.sem16.data(), c.sem16.size());
                std::printf("      sem16 (own vs reference resample) max|Δ|=%.3e mean|Δ|=%.3e (ref peak %.3f)\n",
                            dr.max_abs, dr.mean_abs, dr.ref_max);
                CHECK(dr.max_abs < 1e-5, tag("24 -> 16 kHz resample matches torchaudio (1e-5)"));
            }
        }

        // ── CPU vs CUDA ──
        if (cpu && ci < cpu->wav.size() && cpu->wav[ci].size() == out.wav[ci].size() &&
            !cpu->wav[ci].empty()) {
            const Diff d = diff(out.wav[ci].data(), cpu->wav[ci].data(), cpu->wav[ci].size());
            std::printf("      CPU/CUDA decode max|Δ|=%.3e mean|Δ|=%.3e\n", d.max_abs, d.mean_abs);
            CHECK(d.max_abs < 1e-4, tag("CUDA decode tracks CPU (max abs < 1e-4)"));
            if (cpu->codes[ci].size() == out.codes[ci].size() && !out.codes[ci].empty()) {
                std::vector<double> per;
                const double total = code_agreement(out.codes[ci], cpu->codes[ci], c.K, c.T, per);
                print_rates("CPU/CUDA codes", per, total);
                CHECK(per[0] >= 0.99, tag("CUDA codebook 0 tracks CPU (>=99%)"));
                CHECK(total >= 0.95, tag("CUDA codes track CPU (>=95%)"));
            }
        }
    }

    // ── decoder_only ──
    {
        HiggsCodec d;
        d.load(dir, dev, /*decoder_only=*/true);
        CHECK(d.loaded(), tag("decoder_only load succeeds"));
        CHECK(!d.has_encoder(), tag("decoder_only has no encoder"));
        CHECK(!d.config().has_encoder, tag("decoder_only config.has_encoder is false"));
        brosoundml::AudioBuffer in(cases[0].input, 24000);
        CHECK(throws_runtime_error([&] { d.encode(in); }), tag("decoder_only encode() throws"));
        const FixtureCase& c = cases.back();
        brosoundml::AudioBuffer wav = d.decode(c.codes, c.K, c.T);
        const Diff dd = diff(wav.samples.data(), out.wav.back().data(), out.wav.back().size());
        std::printf("    decoder_only decode vs full-load decode max|Δ|=%.3e\n", dd.max_abs);
        CHECK(dd.max_abs < 1e-6, tag("decoder_only decode matches the full load"));
    }
}

// Encode -> decode the test clip and ask Whisper what it hears.
static void round_trip(brotensor::Device dev, const char* dev_name, const std::string& dir,
                       const fs::path& clip, const fs::path& whisper_dir) {
    std::printf("  [%s] round trip on %s\n", dev_name, clip.filename().string().c_str());
    brosoundml::HiggsCodec codec;
    codec.load(dir, dev);
    brosoundml::AudioBuffer in = brosoundml::read_wav(clip.string());
    int T = 0;
    std::vector<int32_t> codes = codec.encode(in, &T);   // 16 kHz in: resampled inside
    const int expect_T = static_cast<int>((static_cast<long long>(in.samples.size()) * 24000 / in.sample_rate + 959) / 960);
    CHECK(T == expect_T, "round trip: frame count follows the 24 kHz length");
    brosoundml::AudioBuffer out = codec.decode(codes, codec.config().num_quantizers, T);
    CHECK(out.samples.size() == static_cast<std::size_t>(T) * 960, "round trip: decoded length");
    CHECK(out.peak() > 0.05f && out.peak() < 1.5f, "round trip: decoded audio has a sane peak");
    std::printf("    T=%d  in peak %.3f  out peak %.3f  codes[0][0:8] =", T, in.peak(), out.peak());
    for (int t = 0; t < std::min(T, 8); ++t) std::printf(" %d", codes[t]);
    std::printf("\n");

    if (!fs::exists(whisper_dir / "model.safetensors") || !fs::exists(whisper_dir / "vocab.json")) {
        std::printf("    Whisper weights absent — transcript oracle skipped\n");
        return;
    }
    // 24 kHz -> 16 kHz for Whisper (linear, host).
    const int n = static_cast<int>(out.samples.size());
    const int n16 = static_cast<int>(static_cast<long long>(n) * 16000 / 24000);
    brotensor::Tensor x = brotensor::Tensor::from_host_on(brotensor::Device::CPU, out.samples.data(), 1, n);
    brotensor::Tensor y;
    brotensor::resample1d_forward(x, 1, 1, n, n16, /*mode=*/1, y);
    brosoundml::AudioBuffer a16(std::vector<float>(y.host_f32(), y.host_f32() + n16), 16000);

    brosoundml::Whisper w;
    w.load(whisper_dir.string(), dev);
    auto tok = brolm::whisper::Tokenizer::load((whisper_dir / "vocab.json").string(),
                                               (whisper_dir / "merges.txt").string());
    std::vector<int32_t> prompt = tok.build_prompt("en", "transcribe", /*timestamps=*/false);
    brosoundml::Whisper::TranscribeOptions opts;
    auto r = w.transcribe(a16, prompt, opts);
    std::vector<int32_t> ids(r.token_ids.begin() + static_cast<std::ptrdiff_t>(prompt.size()), r.token_ids.end());
    std::string text = tok.decode(ids, /*skip_special=*/true);
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::printf("    whisper: \"%s\"\n", text.c_str());
    CHECK(lower.find("test") != std::string::npos, "round trip: Whisper hears \"test\" in the decoded clip");
}

static int run() {
    using brosoundml::HiggsCodec;

    // ─── contract before load ───────────────────────────────────────────────
    {
        HiggsCodec c;
        CHECK(!c.loaded(), "a fresh HiggsCodec is not loaded");
        CHECK(!c.has_encoder(), "a fresh HiggsCodec has no encoder");
        CHECK(c.config().sample_rate == 24000, "default config sample_rate 24000");
        CHECK(c.config().hop_length == 960, "default config hop_length 960");
        CHECK(c.config().num_quantizers == 8, "default config num_quantizers 8");
        std::vector<int32_t> codes(8, 0);
        CHECK(throws_runtime_error([&] { c.decode(codes, 8, 1); }), "decode() before load() throws");
        CHECK(throws_runtime_error([&] { c.decode(codes.data(), 8, 1); }), "decode(ptr) before load() throws");
        CHECK(throws_runtime_error([&] { c.encode(brosoundml::AudioBuffer(std::vector<float>(960, 0.f), 24000)); }),
              "encode() before load() throws");
        CHECK(throws_runtime_error([&] { c.load("nonexistent-higgs-codec-dir"); }),
              "load() on a missing directory throws");
        CHECK(!c.loaded(), "a failed load() leaves the codec unloaded");
        HiggsCodec moved = std::move(c);
        CHECK(!moved.loaded(), "move keeps the unloaded state");
    }

    const fs::path repo(BROSOUNDML_REPO_DIR);
    const fs::path dir = repo / "weights" / "omnivoice" / "audio_tokenizer";
    if (!fs::exists(dir / "model.safetensors") || !fs::exists(dir / "config.json")) {
        std::printf("SKIP: %s absent (HiggsAudio v2 tokenizer weights) — real-weights checks skipped\n",
                    dir.string().c_str());
        return failures == 0 ? 0 : 1;
    }

    std::vector<FixtureCase> cases;
    const fs::path fixture = repo / "tests" / "fixtures" / "higgs_codec.bin";
    if (!read_fixture(fixture, cases)) {
        cases.clear();
        std::printf("note: fixture %s absent — reference comparisons skipped\n", fixture.string().c_str());
    }

    DeviceResults cpu_res;
    run_device(brotensor::Device::CPU, "CPU", dir.string(), cases, cpu_res, nullptr);
    const bool has_cuda = brotensor::is_available(brotensor::Device::CUDA);
    if (has_cuda) {
        DeviceResults cuda_res;
        run_device(brotensor::Device::CUDA, "CUDA", dir.string(), cases, cuda_res, &cpu_res);
    } else {
        std::printf("  CUDA not available — device parity skipped\n");
    }

    const fs::path clip = repo / "weights" / "qwen-tts-hello-there-this-is-a-test-of-th.wav";
    if (fs::exists(clip)) {
        round_trip(has_cuda ? brotensor::Device::CUDA : brotensor::Device::CPU,
                   has_cuda ? "CUDA" : "CPU", dir.string(), clip, repo / "weights" / "whisper");
    } else {
        std::printf("  test clip absent — round trip skipped\n");
    }

    if (failures) std::fprintf(stderr, "test_higgs_codec: %d failure(s)\n", failures);
    else          std::printf("test_higgs_codec: all checks passed\n");
    return failures == 0 ? 0 : 1;
}

int main() {
    brotensor::init();
    try { return run(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "test_higgs_codec: uncaught exception: %s\n", e.what());
        return 2;
    }
    catch (...) { std::fprintf(stderr, "test_higgs_codec: uncaught non-std exception\n"); return 2; }
}
