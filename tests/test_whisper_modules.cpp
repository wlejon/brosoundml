// Stage 2-3 Whisper module unit tests: log-mel front-end + encoder.
//
// Self-contained — generates a tiny synthetic encoder checkpoint on disk
// (2 layers, d_model=8, ffn=16, n_mels=8, heads=2), drives it through
// `WhisperEncoder::load_from` + `forward`, and checks shapes / numerics. No
// real Whisper weights required; the bigger end-to-end smoke test lands with
// the decoder in stage 4.
#include "brosoundml/audio.h"
#include "brosoundml/whisper_modules.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace bt = brotensor;
namespace stf = brotensor::safetensors;

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

// ─── Synthetic weight generator ────────────────────────────────────────────
//
// Builds the in-memory float vectors for a tiny whisper-shaped encoder, then
// dumps them to a single safetensors file via brotensor::safetensors::write.
// The numeric values come from a deterministic PRNG with a fixed seed so the
// test is reproducible; we don't care about a specific output value, only
// that shapes and finiteness hold.

struct StubBuffers {
    // We keep every payload alive in this struct so the WriteEntry pointers
    // in `entries` stay valid until write_file consumes them.
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

static std::vector<float> zeros_vec(std::size_t n) {
    return std::vector<float>(n, 0.0f);
}

static std::vector<float> ones_vec(std::size_t n) {
    return std::vector<float>(n, 1.0f);
}

static void write_stub_encoder(const fs::path& path,
                               int n_mels, int d_model, int max_src,
                               int n_layers, int ffn, int n_heads) {
    std::mt19937 rng(12345);
    StubBuffers sb;
    const std::string p = "model.encoder.";

    // conv1 (d_model, n_mels, 3)
    sb.add(p + "conv1.weight", {d_model, n_mels, 3},
           rand_vec(rng, static_cast<std::size_t>(d_model) * n_mels * 3));
    sb.add(p + "conv1.bias",   {d_model}, zeros_vec(d_model));
    // conv2 (d_model, d_model, 3)
    sb.add(p + "conv2.weight", {d_model, d_model, 3},
           rand_vec(rng, static_cast<std::size_t>(d_model) * d_model * 3));
    sb.add(p + "conv2.bias",   {d_model}, zeros_vec(d_model));
    // embed_positions (max_src, d_model)
    sb.add(p + "embed_positions.weight", {max_src, d_model},
           rand_vec(rng, static_cast<std::size_t>(max_src) * d_model, 0.02f));

    for (int i = 0; i < n_layers; ++i) {
        const std::string lp = p + "layers." + std::to_string(i) + ".";
        sb.add(lp + "self_attn_layer_norm.weight", {d_model}, ones_vec(d_model));
        sb.add(lp + "self_attn_layer_norm.bias",   {d_model}, zeros_vec(d_model));

        sb.add(lp + "self_attn.q_proj.weight", {d_model, d_model},
               rand_vec(rng, static_cast<std::size_t>(d_model) * d_model));
        sb.add(lp + "self_attn.q_proj.bias",   {d_model}, zeros_vec(d_model));
        sb.add(lp + "self_attn.k_proj.weight", {d_model, d_model},
               rand_vec(rng, static_cast<std::size_t>(d_model) * d_model));
        // NB: NO k_proj.bias on disk — load_from must allocate it zero.
        sb.add(lp + "self_attn.v_proj.weight", {d_model, d_model},
               rand_vec(rng, static_cast<std::size_t>(d_model) * d_model));
        sb.add(lp + "self_attn.v_proj.bias",   {d_model}, zeros_vec(d_model));
        sb.add(lp + "self_attn.out_proj.weight", {d_model, d_model},
               rand_vec(rng, static_cast<std::size_t>(d_model) * d_model));
        sb.add(lp + "self_attn.out_proj.bias",   {d_model}, zeros_vec(d_model));

        sb.add(lp + "final_layer_norm.weight", {d_model}, ones_vec(d_model));
        sb.add(lp + "final_layer_norm.bias",   {d_model}, zeros_vec(d_model));

        sb.add(lp + "fc1.weight", {ffn, d_model},
               rand_vec(rng, static_cast<std::size_t>(ffn) * d_model));
        sb.add(lp + "fc1.bias",   {ffn}, zeros_vec(ffn));
        sb.add(lp + "fc2.weight", {d_model, ffn},
               rand_vec(rng, static_cast<std::size_t>(d_model) * ffn));
        sb.add(lp + "fc2.bias",   {d_model}, zeros_vec(d_model));
    }

    sb.add(p + "layer_norm.weight", {d_model}, ones_vec(d_model));
    sb.add(p + "layer_norm.bias",   {d_model}, zeros_vec(d_model));

    (void)n_heads;
    stf::write_file(path.string(), sb.entries);
}

// ─── LogMel tests ──────────────────────────────────────────────────────────

static void test_logmel_filterbank_shape() {
    brosoundml::LogMel m;
    m.build(/*num_mel_bins=*/80, bt::Device::CPU);
    CHECK(m.mel_filters.rows == 80, "mel_filters has 80 rows");
    CHECK(m.mel_filters.cols == 201, "mel_filters has n_fft/2+1 == 201 cols");
    CHECK(m.hann_window.rows == 1 && m.hann_window.cols == 400,
          "hann_window is (1, 400)");

    const float* fb = m.mel_filters.host_f32();
    bool all_nonneg = true;
    bool any_positive_row = false;
    int  prev_peak = -1;
    bool peaks_monotone = true;
    for (int row = 0; row < 80; ++row) {
        float row_sum = 0.0f;
        int peak_idx = 0;
        float peak_v = -1.0f;
        for (int k = 0; k < 201; ++k) {
            const float v = fb[row * 201 + k];
            if (v < 0.0f) all_nonneg = false;
            row_sum += v;
            if (v > peak_v) { peak_v = v; peak_idx = k; }
        }
        if (row_sum > 0.0f) any_positive_row = true;
        // Mel-bin peaks are non-decreasing in fft-bin index (ties allowed at
        // the very low end where multiple mel triangles fit inside one fft bin).
        if (peak_idx < prev_peak) peaks_monotone = false;
        prev_peak = peak_idx;
    }
    CHECK(all_nonneg, "mel filterbank is non-negative");
    CHECK(any_positive_row, "mel filterbank has at least one positive row");
    CHECK(peaks_monotone, "mel filter peak frequencies are non-decreasing");
    // And across the full filterbank, the last peak should be far above the
    // first — Slaney spans 0 -> Nyquist.
    CHECK(prev_peak > 100,
          "mel filter peaks span into the upper FFT-bin range");

    // hann_window is symmetric-ish and bounded in [0,1].
    const float* hw = m.hann_window.host_f32();
    bool hw_bounded = true;
    for (int n = 0; n < 400; ++n) {
        if (hw[n] < -1e-6f || hw[n] > 1.0f + 1e-6f) hw_bounded = false;
    }
    CHECK(hw_bounded, "hann_window values are in [0, 1]");
}

static void test_logmel_forward_sine() {
    brosoundml::LogMel m;
    m.build(/*num_mel_bins=*/80, bt::Device::CPU);

    // 440 Hz sine, 1 s at 16 kHz; the front-end pads to 30 s so a 1-s clip is
    // fine. The peak mel bin should land near 440 Hz, i.e. roughly mel ≈ 6.6
    // -> bin index near 6 (Slaney mel: linear region, slope 3/200).
    brosoundml::AudioBuffer audio;
    audio.sample_rate = 16000;
    audio.samples.resize(16000);
    constexpr double k_two_pi = 6.283185307179586;
    for (int n = 0; n < 16000; ++n) {
        audio.samples[n] = 0.5f * static_cast<float>(
            std::sin(k_two_pi * 440.0 * n / 16000.0));
    }

    bt::Tensor out;
    m.forward(audio, out);
    CHECK(out.rows == 80,   "log-mel output has num_mel_bins rows");
    CHECK(out.cols == 3000, "log-mel output has 3000 frames");

    // After normalisation, every value is in roughly [-1, 1] (specifically
    // [(floor + 4)/4, (max + 4)/4], with floor = max - 8).
    const float* d = out.host_f32();
    const std::size_t total = static_cast<std::size_t>(80) * 3000;
    bool finite = true;
    float min_v = 1e30f, max_v = -1e30f;
    for (std::size_t i = 0; i < total; ++i) {
        if (!std::isfinite(d[i])) finite = false;
        if (d[i] < min_v) min_v = d[i];
        if (d[i] > max_v) max_v = d[i];
    }
    CHECK(finite, "log-mel output is all-finite");
    CHECK(min_v >= -1.5f && max_v <= 1.5f,
          "log-mel output is normalised to roughly [-1, 1]");

    // Energy concentration: the row with the largest peak (max over frames)
    // should be a low-index mel bin, since 440 Hz sits in the bottom of the
    // mel scale.
    int best_row = 0;
    float best_peak = -1e30f;
    for (int r = 0; r < 80; ++r) {
        float row_peak = -1e30f;
        // Look at frames in the first second (the actual sine region).
        for (int f = 0; f < 100; ++f) {
            const float v = d[static_cast<std::size_t>(r) * 3000 + f];
            if (v > row_peak) row_peak = v;
        }
        if (row_peak > best_peak) { best_peak = row_peak; best_row = r; }
    }
    CHECK(best_row < 20,
          "440 Hz sine concentrates energy in the low-mel-bin range");

    // Reject non-16 kHz audio loudly.
    brosoundml::AudioBuffer bad;
    bad.sample_rate = 22050;
    bad.samples.assign(1000, 0.0f);
    bt::Tensor tmp;
    CHECK(throws_runtime_error([&] { m.forward(bad, tmp); }),
          "log-mel refuses non-16-kHz audio");
}

// ─── Encoder tests ─────────────────────────────────────────────────────────

static void test_encoder_layer_forward_shape() {
    const int d_model = 8, ffn = 16, n_heads = 2;
    const fs::path tmp = fs::temp_directory_path() /
                         "brosoundml_whisper_modules_layer.safetensors";
    fs::remove(tmp);
    // The layer-only test still needs a complete encoder file because layers
    // share the same on-disk layout. Build a 1-layer file just for this check.
    write_stub_encoder(tmp, /*n_mels=*/8, d_model,
                       /*max_src=*/4, /*n_layers=*/1, ffn, n_heads);
    brosoundml::WhisperEncoderLayer layer;
    {
        stf::File f = stf::File::open(tmp.string());
        layer.load_from(f, "model.encoder.layers.0.", d_model, ffn, n_heads);
    }

    // Check the K-bias was zero-filled to (d_model, 1).
    CHECK(layer.self_attn.bk.rows == d_model && layer.self_attn.bk.cols == 1,
          "k_proj bias is (d_model, 1) zero-filled");
    const float* bk = layer.self_attn.bk.host_f32();
    bool all_zero = true;
    for (int i = 0; i < d_model; ++i) if (bk[i] != 0.0f) all_zero = false;
    CHECK(all_zero, "k_proj bias is all zeros");

    // Forward pass on a (L, d_model) input must preserve shape.
    const int L = 5;
    bt::Tensor X = bt::Tensor::zeros_on(bt::Device::CPU, L, d_model, bt::Dtype::FP32);
    float* xd = X.host_f32_mut();
    for (int i = 0; i < L * d_model; ++i) xd[i] = 0.01f * static_cast<float>(i);
    bt::Tensor Y;
    layer.forward(X, Y);
    CHECK(Y.rows == L && Y.cols == d_model,
          "encoder layer preserves (L, d_model) shape");
    bool finite = true;
    const float* yd = Y.host_f32();
    for (int i = 0; i < L * d_model; ++i) if (!std::isfinite(yd[i])) finite = false;
    CHECK(finite, "encoder layer output is all-finite");

    fs::remove(tmp);
}

static void test_encoder_full_forward() {
    // Tiny but realistic-shape encoder: max_source_positions = 1500 still works
    // (it's just a 1500-row positional embedding), but the test runs 30 s of
    // input through a 2-layer / 8-channel stack so this stays fast.
    //
    // Note: the encoder downsamples by 2 (conv2 stride 2), so n_frames input
    // must equal 2 * max_source_positions. We use n_frames=3000 / max_src=1500
    // because that's the production shape. d_model=8, ffn=16 keep the heavy
    // matmul costs low.
    const int n_mels = 8, d_model = 8, max_src = 1500;
    const int n_layers = 2, ffn = 16, n_heads = 2;
    const fs::path tmp = fs::temp_directory_path() /
                         "brosoundml_whisper_modules_full.safetensors";
    fs::remove(tmp);
    write_stub_encoder(tmp, n_mels, d_model, max_src, n_layers, ffn, n_heads);
    brosoundml::WhisperEncoder enc;
    {
        stf::File f = stf::File::open(tmp.string());
        enc.load_from(f, n_mels, d_model, max_src, n_layers, ffn, n_heads);
    }

    CHECK(static_cast<int>(enc.layers.size()) == n_layers,
          "encoder loaded the requested layer count");

    // Build a synthetic mel input: (n_mels, 3000).
    bt::Tensor mel = bt::Tensor::zeros_on(bt::Device::CPU, n_mels, 3000,
                                          bt::Dtype::FP32);
    float* md = mel.host_f32_mut();
    for (int r = 0; r < n_mels; ++r) {
        for (int c = 0; c < 3000; ++c) {
            md[r * 3000 + c] = 0.01f * static_cast<float>((r + c) % 17 - 8);
        }
    }
    bt::Tensor hidden;
    enc.forward(mel, hidden);
    CHECK(hidden.rows == max_src && hidden.cols == d_model,
          "encoder output is (max_source_positions, d_model)");
    bool finite = true;
    const float* hd = hidden.host_f32();
    const std::size_t total = static_cast<std::size_t>(max_src) * d_model;
    for (std::size_t i = 0; i < total; ++i) {
        if (!std::isfinite(hd[i])) { finite = false; break; }
    }
    CHECK(finite, "encoder output is all-finite");

    fs::remove(tmp);
}

int main() {
    try {
        test_logmel_filterbank_shape();
        test_logmel_forward_sine();
        test_encoder_layer_forward_shape();
        test_encoder_full_forward();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_whisper_modules: uncaught exception: %s\n",
                     e.what());
        return 2;
    } catch (...) {
        std::fprintf(stderr, "test_whisper_modules: uncaught non-std exception\n");
        return 2;
    }
    if (failures) {
        std::fprintf(stderr, "test_whisper_modules: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_whisper_modules: all checks passed\n");
    return 0;
}
