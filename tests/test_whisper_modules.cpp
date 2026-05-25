// Stage 2-3 Whisper module unit tests: log-mel front-end + encoder.
//
// Self-contained — generates a tiny synthetic encoder checkpoint on disk
// (2 layers, d_model=8, ffn=16, n_mels=8, heads=2), drives it through
// `WhisperEncoder::load_from` + `forward`, and checks shapes / numerics. No
// real Whisper weights required; the bigger end-to-end smoke test lands with
// the decoder in stage 4.
#include "brosoundml/audio.h"
#include "brosoundml/whisper_modules.h"

#include <brotensor/runtime.h>
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

static std::string tagmsg(const char* msg, const char* dev_name) {
    std::string s = "["; s += dev_name; s += "] "; s += msg; return s;
}

static void test_logmel_filterbank_shape(bt::Device dev, const char* dev_name) {
    (void)dev_name;  // filterbank/Hann tables are CPU-resident regardless of dev
    brosoundml::LogMel m;
    m.build(/*num_mel_bins=*/80, dev);
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

static void test_logmel_forward_sine(bt::Device dev, const char* dev_name) {
    brosoundml::LogMel m;
    m.build(/*num_mel_bins=*/80, dev);

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
    CHECK(out.rows == 80,   tagmsg("log-mel output has num_mel_bins rows", dev_name).c_str());
    CHECK(out.cols == 3000, tagmsg("log-mel output has 3000 frames", dev_name).c_str());

    // After normalisation, every value is in roughly [-1, 1] (specifically
    // [(floor + 4)/4, (max + 4)/4], with floor = max - 8).
    std::vector<float> out_host = out.to_host_vector();
    const float* d = out_host.data();
    const std::size_t total = static_cast<std::size_t>(80) * 3000;
    bool finite = true;
    float min_v = 1e30f, max_v = -1e30f;
    for (std::size_t i = 0; i < total; ++i) {
        if (!std::isfinite(d[i])) finite = false;
        if (d[i] < min_v) min_v = d[i];
        if (d[i] > max_v) max_v = d[i];
    }
    CHECK(finite, tagmsg("log-mel output is all-finite", dev_name).c_str());
    CHECK(min_v >= -1.5f && max_v <= 1.5f,
          tagmsg("log-mel output is normalised to roughly [-1, 1]", dev_name).c_str());

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
          tagmsg("440 Hz sine concentrates energy in the low-mel-bin range", dev_name).c_str());

    brosoundml::AudioBuffer bad;
    bad.sample_rate = 22050;
    bad.samples.assign(1000, 0.0f);
    bt::Tensor tmp;
    CHECK(throws_runtime_error([&] { m.forward(bad, tmp); }),
          tagmsg("log-mel refuses non-16-kHz audio", dev_name).c_str());
}

// ─── Encoder tests ─────────────────────────────────────────────────────────

static void test_encoder_layer_forward_shape(bt::Device dev, const char* dev_name) {
    const int d_model = 8, ffn = 16, n_heads = 2;
    const fs::path tmp = fs::temp_directory_path() /
                         "brosoundml_whisper_modules_layer.safetensors";
    fs::remove(tmp);
    write_stub_encoder(tmp, /*n_mels=*/8, d_model,
                       /*max_src=*/4, /*n_layers=*/1, ffn, n_heads);
    brosoundml::WhisperEncoderLayer layer;
    {
        stf::File f = stf::File::open(tmp.string());
        layer.load_from(f, "model.encoder.layers.0.", d_model, ffn, n_heads, dev);
    }

    // Check the K-bias was zero-filled to (d_model, 1). bk allocates on `dev`.
    CHECK(layer.self_attn.bk.rows == d_model && layer.self_attn.bk.cols == 1,
          tagmsg("k_proj bias is (d_model, 1) zero-filled", dev_name).c_str());
    std::vector<float> bk_host = layer.self_attn.bk.to_host_vector();
    bool all_zero = true;
    for (int i = 0; i < d_model; ++i) if (bk_host[i] != 0.0f) all_zero = false;
    CHECK(all_zero, tagmsg("k_proj bias is all zeros", dev_name).c_str());

    // Forward pass on a (L, d_model) input must preserve shape. The encoder
    // layer expects X on the same device as its weights — which for upload()
    // means CPU today; the layer's `load_from` device parameter only governs
    // the zero-filled bk allocation (the bias-add will run on `dev`).
    const int L = 5;
    std::vector<float> xh(L * d_model);
    for (int i = 0; i < L * d_model; ++i) xh[i] = 0.01f * static_cast<float>(i);
    bt::Tensor X = bt::Tensor::from_host_on(dev, xh.data(), L, d_model);
    bt::Tensor Y;
    layer.forward(X, Y);
    CHECK(Y.rows == L && Y.cols == d_model,
          tagmsg("encoder layer preserves (L, d_model) shape", dev_name).c_str());
    std::vector<float> y_host = Y.to_host_vector();
    bool finite = true;
    for (int i = 0; i < L * d_model; ++i) if (!std::isfinite(y_host[i])) finite = false;
    CHECK(finite, tagmsg("encoder layer output is all-finite", dev_name).c_str());

    fs::remove(tmp);
}

static void test_encoder_full_forward(bt::Device dev, const char* dev_name) {
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
        enc.load_from(f, n_mels, d_model, max_src, n_layers, ffn, n_heads, dev);
    }

    CHECK(static_cast<int>(enc.layers.size()) == n_layers,
          tagmsg("encoder loaded the requested layer count", dev_name).c_str());

    // Build a synthetic mel input: (n_mels, 3000) on `dev`.
    std::vector<float> mh(static_cast<std::size_t>(n_mels) * 3000);
    for (int r = 0; r < n_mels; ++r) {
        for (int c = 0; c < 3000; ++c) {
            mh[r * 3000 + c] = 0.01f * static_cast<float>((r + c) % 17 - 8);
        }
    }
    bt::Tensor mel = bt::Tensor::from_host_on(dev, mh.data(), n_mels, 3000);
    bt::Tensor hidden;
    enc.forward(mel, hidden);
    CHECK(hidden.rows == max_src && hidden.cols == d_model,
          tagmsg("encoder output is (max_source_positions, d_model)", dev_name).c_str());
    std::vector<float> h_host = hidden.to_host_vector();
    bool finite = true;
    const std::size_t total = static_cast<std::size_t>(max_src) * d_model;
    for (std::size_t i = 0; i < total; ++i) {
        if (!std::isfinite(h_host[i])) { finite = false; break; }
    }
    CHECK(finite, tagmsg("encoder output is all-finite", dev_name).c_str());

    fs::remove(tmp);
}

static void run_all(bt::Device dev, const char* dev_name) {
    test_logmel_filterbank_shape(dev, dev_name);
    test_logmel_forward_sine(dev, dev_name);
    test_encoder_layer_forward_shape(dev, dev_name);
    test_encoder_full_forward(dev, dev_name);
}

int main() {
    bt::init();
    try {
        run_all(bt::Device::CPU, "CPU");
        if (bt::is_available(bt::Device::CUDA)) {
            run_all(bt::Device::CUDA, "CUDA");
        } else {
            std::printf("test_whisper_modules: CUDA not available — CUDA path skipped\n");
        }
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
