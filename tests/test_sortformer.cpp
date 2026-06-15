// Sortformer (streaming speaker diarization) end-to-end synthetic test.
//
// Builds a tiny in-memory Sortformer checkpoint (config.json + model.safetensors
// with every converted key the loader expects, including the NEST projection /
// FFN / conv biases), loads it, runs diarize() and the streaming feed() path on
// a synthetic 16 kHz sine, and pins the contract: the output is a (T x num_spks)
// matrix of probabilities in [0,1], a re-run is bit-identical, bad inputs throw,
// and the streaming session reduces to the offline forward for a sub-chunk clip.
// No real weights needed — this is what CI relies on. Runs on CPU and (when
// present) CUDA. A CUDA-only real-weights smoke runs if the converted checkpoint
// is on disk.

#include "brosoundml/audio.h"
#include "brosoundml/sortformer.h"

#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace fs  = std::filesystem;
namespace stf = brotensor::safetensors;

static int failures = 0;
#define CHECK(cond, msg)                                   \
    do {                                                   \
        if (!(cond)) {                                     \
            std::fprintf(stderr, "FAIL: %s\n", (msg));     \
            ++failures;                                    \
        }                                                  \
    } while (0)

struct StubBuffers {
    std::vector<std::vector<float>> bufs;
    std::vector<stf::WriteEntry>    entries;
    void add(const std::string& name, const std::vector<std::int64_t>& shape,
             std::vector<float>&& payload) {
        bufs.push_back(std::move(payload));
        stf::WriteEntry e;
        e.name = name; e.dtype = stf::Dtype::F32; e.shape = shape;
        e.host_data = bufs.back().data();
        e.bytes = bufs.back().size() * sizeof(float);
        entries.push_back(std::move(e));
    }
};

static std::mt19937 g_rng(11);
static std::vector<float> randv(std::size_t n, float s = 0.05f) {
    std::uniform_real_distribution<float> d(-s, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g_rng);
    return v;
}
static std::vector<float> zeros(std::size_t n) { return std::vector<float>(n, 0.0f); }
static std::vector<float> ones(std::size_t n)  { return std::vector<float>(n, 1.0f); }

// Tiny shape: 8x subsampling (16 mels -> 2 freq), 2 conformer layers, 2
// transformer layers, 4 speakers. fc_d_model == encoder hidden_size.
struct Tiny {
    int n_mels  = 16;
    int C       = 32;   // fc_d_model / encoder hidden_size
    int heads   = 2;    // head_dim 16
    int inter   = 64;
    int conv_ch = 8;
    int conv_k  = 9;
    int enc_l   = 2;
    int tf_d    = 24;   // tf_d_model (head_dim 12 with 2 heads)
    int tf_in   = 48;
    int tf_h    = 2;
    int tf_l    = 2;
    int spk     = 4;
};

static void write_ckpt(const fs::path& path, const Tiny& t) {
    StubBuffers sb;
    const int C = t.C, ch = t.conv_ch;

    // ── subsampling (with conv biases) ──
    const std::string sp = "encoder.subsampling.";
    sb.add(sp + "layers.0.weight", {ch, 1, 3, 3}, randv((std::size_t)ch * 9));
    sb.add(sp + "layers.0.bias",   {ch},          zeros(ch));
    sb.add(sp + "layers.2.weight", {ch, 1, 3, 3}, randv((std::size_t)ch * 9));
    sb.add(sp + "layers.2.bias",   {ch},          zeros(ch));
    sb.add(sp + "layers.3.weight", {ch, ch, 1, 1}, randv((std::size_t)ch * ch));
    sb.add(sp + "layers.3.bias",   {ch},          zeros(ch));
    sb.add(sp + "layers.5.weight", {ch, 1, 3, 3}, randv((std::size_t)ch * 9));
    sb.add(sp + "layers.5.bias",   {ch},          zeros(ch));
    sb.add(sp + "layers.6.weight", {ch, ch, 1, 1}, randv((std::size_t)ch * ch));
    sb.add(sp + "layers.6.bias",   {ch},          zeros(ch));
    const int freq_out = 2;   // 16 -> 8 -> 4 -> 2
    sb.add(sp + "linear.weight", {C, ch * freq_out},
           randv((std::size_t)C * ch * freq_out));
    sb.add(sp + "linear.bias",   {C}, zeros(C));

    // ── conformer layers (NEST: projection / FFN / conv biases present) ──
    for (int i = 0; i < t.enc_l; ++i) {
        const std::string L = "encoder.layers." + std::to_string(i) + ".";
        for (const char* nm : {"norm_feed_forward1", "norm_self_att", "norm_conv",
                               "norm_feed_forward2", "norm_out"}) {
            sb.add(L + nm + ".weight", {C}, ones(C));
            sb.add(L + nm + ".bias",   {C}, zeros(C));
        }
        sb.add(L + "feed_forward1.linear1.weight", {t.inter, C}, randv((std::size_t)t.inter * C));
        sb.add(L + "feed_forward1.linear1.bias",   {t.inter},    zeros(t.inter));
        sb.add(L + "feed_forward1.linear2.weight", {C, t.inter}, randv((std::size_t)C * t.inter));
        sb.add(L + "feed_forward1.linear2.bias",   {C},          zeros(C));
        sb.add(L + "feed_forward2.linear1.weight", {t.inter, C}, randv((std::size_t)t.inter * C));
        sb.add(L + "feed_forward2.linear1.bias",   {t.inter},    zeros(t.inter));
        sb.add(L + "feed_forward2.linear2.weight", {C, t.inter}, randv((std::size_t)C * t.inter));
        sb.add(L + "feed_forward2.linear2.bias",   {C},          zeros(C));
        for (const char* pj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
            sb.add(L + "self_attn." + pj + ".weight", {C, C}, randv((std::size_t)C * C));
            sb.add(L + "self_attn." + pj + ".bias",   {C},    zeros(C));
        }
        sb.add(L + "self_attn.relative_k_proj.weight", {C, C}, randv((std::size_t)C * C));
        sb.add(L + "self_attn.bias_u", {t.heads, C / t.heads}, zeros(C));
        sb.add(L + "self_attn.bias_v", {t.heads, C / t.heads}, zeros(C));
        sb.add(L + "conv.pointwise_conv1.weight", {2 * C, C, 1}, randv((std::size_t)2 * C * C));
        sb.add(L + "conv.pointwise_conv1.bias",   {2 * C},       zeros(2 * C));
        sb.add(L + "conv.depthwise_conv.weight",  {C, 1, t.conv_k}, randv((std::size_t)C * t.conv_k));
        sb.add(L + "conv.depthwise_conv.bias",    {C},           zeros(C));
        sb.add(L + "conv.pointwise_conv2.weight", {C, C, 1},     randv((std::size_t)C * C));
        sb.add(L + "conv.pointwise_conv2.bias",   {C},           zeros(C));
        sb.add(L + "conv.norm.weight",       {C}, ones(C));
        sb.add(L + "conv.norm.bias",         {C}, zeros(C));
        sb.add(L + "conv.norm.running_mean", {C}, zeros(C));
        sb.add(L + "conv.norm.running_var",  {C}, ones(C));
    }

    // ── sortformer head: encoder_proj + transformer + sigmoid head ──
    sb.add("sortformer.encoder_proj.weight", {t.tf_d, C}, randv((std::size_t)t.tf_d * C));
    sb.add("sortformer.encoder_proj.bias",   {t.tf_d},    zeros(t.tf_d));
    for (int i = 0; i < t.tf_l; ++i) {
        const std::string L = "transformer.layers." + std::to_string(i) + ".";
        sb.add(L + "norm1.weight", {t.tf_d}, ones(t.tf_d));
        sb.add(L + "norm1.bias",   {t.tf_d}, zeros(t.tf_d));
        sb.add(L + "norm2.weight", {t.tf_d}, ones(t.tf_d));
        sb.add(L + "norm2.bias",   {t.tf_d}, zeros(t.tf_d));
        for (const char* pj : {"q", "k", "v", "o"}) {
            sb.add(L + "attn." + pj + ".weight", {t.tf_d, t.tf_d}, randv((std::size_t)t.tf_d * t.tf_d));
            sb.add(L + "attn." + pj + ".bias",   {t.tf_d},         zeros(t.tf_d));
        }
        sb.add(L + "ff.in.weight",  {t.tf_in, t.tf_d}, randv((std::size_t)t.tf_in * t.tf_d));
        sb.add(L + "ff.in.bias",    {t.tf_in},         zeros(t.tf_in));
        sb.add(L + "ff.out.weight", {t.tf_d, t.tf_in}, randv((std::size_t)t.tf_d * t.tf_in));
        sb.add(L + "ff.out.bias",   {t.tf_d},          zeros(t.tf_d));
    }
    sb.add("sortformer.first_hidden_to_hidden.weight", {t.tf_d, t.tf_d}, randv((std::size_t)t.tf_d * t.tf_d));
    sb.add("sortformer.first_hidden_to_hidden.bias",   {t.tf_d},         zeros(t.tf_d));
    sb.add("sortformer.single_hidden_to_spks.weight",  {t.spk, t.tf_d},  randv((std::size_t)t.spk * t.tf_d));
    sb.add("sortformer.single_hidden_to_spks.bias",    {t.spk},          zeros(t.spk));

    stf::write_file(path.string(), sb.entries);
}

static std::string config_json(const Tiny& t) {
    char buf[2048];
    std::snprintf(buf, sizeof buf, R"json({
  "model_type": "sortformer_diar",
  "sample_rate": 16000,
  "num_spks": %d,
  "fc_d_model": %d,
  "tf_d_model": %d,
  "encoder_config": {
    "num_mel_bins": %d, "hidden_size": %d, "num_hidden_layers": %d,
    "num_attention_heads": %d, "intermediate_size": %d, "conv_kernel_size": %d,
    "subsampling_factor": 8, "subsampling_conv_channels": %d,
    "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2,
    "scale_input": true, "attention_bias": true, "convolution_bias": true,
    "normalize_features": false
  },
  "transformer_config": {
    "num_layers": %d, "hidden_size": %d, "inner_size": %d,
    "num_attention_heads": %d, "pre_ln": false
  },
  "streaming_config": {
    "spkcache_len": 16, "fifo_len": 0, "chunk_len": 16,
    "spkcache_update_period": 16, "chunk_left_context": 1, "chunk_right_context": 1,
    "spkcache_sil_frames_per_spk": 1
  }
})json",
                  t.spk, t.C, t.tf_d, t.n_mels, t.C, t.enc_l, t.heads, t.inter,
                  t.conv_k, t.conv_ch, t.tf_l, t.tf_d, t.tf_in, t.tf_h);
    return buf;
}

static void write_text(const fs::path& p, const std::string& body) {
    std::ofstream f(p, std::ios::binary);
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
}

static brosoundml::AudioBuffer sine(float hz, float seconds, int sr = 16000) {
    brosoundml::AudioBuffer a;
    a.sample_rate = sr;
    a.samples.resize(static_cast<std::size_t>(seconds * sr));
    for (std::size_t n = 0; n < a.samples.size(); ++n)
        a.samples[n] = 0.2f * std::sin(2.0f * 3.14159265f * hz *
                                       static_cast<float>(n) / sr);
    return a;
}

template <typename Fn>
static bool throws(Fn&& fn) {
    try { fn(); } catch (const std::runtime_error&) { return true; }
    catch (...) { return false; }
    return false;
}

static std::string tag(const char* m, const char* d) {
    std::string s = "["; s += d; s += "] "; s += m; return s;
}

static int run_for_device(brotensor::Device dev, const char* dn) {
    using brosoundml::AudioBuffer;
    using brosoundml::Sortformer;
    using brosoundml::SortformerSession;

    const fs::path root = fs::temp_directory_path() / "brosoundml_sortformer_e2e";
    fs::remove_all(root);
    fs::create_directories(root);

    Tiny t;
    write_text(root / "config.json", config_json(t));
    write_ckpt(root / "model.safetensors", t);

    Sortformer m;
    m.load(root.string(), dev);
    CHECK(m.loaded(), tag("loaded synthetic Sortformer checkpoint", dn).c_str());
    CHECK(m.config().num_spks == t.spk, tag("config num_spks round-trips", dn).c_str());
    CHECK(m.config().encoder.num_hidden_layers == t.enc_l,
          tag("nested encoder_config parsed", dn).c_str());
    CHECK(m.config().transformer.num_layers == t.tf_l,
          tag("transformer_config parsed", dn).c_str());

    // Bad-input guards.
    {
        AudioBuffer empty; empty.sample_rate = 16000;
        CHECK(throws([&] { m.diarize(empty); }),
              tag("diarize rejects empty audio", dn).c_str());
        AudioBuffer wrong; wrong.sample_rate = 22050; wrong.samples.assign(22050, 0.0f);
        CHECK(throws([&] { m.diarize(wrong); }),
              tag("diarize rejects non-16k audio", dn).c_str());
    }

    // Happy path: a (T x num_spks) probability matrix in [0,1].
    AudioBuffer audio = sine(330.0f, 0.6f);
    auto d = m.diarize(audio);
    CHECK(d.num_speakers == t.spk, tag("num_speakers == num_spks", dn).c_str());
    CHECK(d.num_frames > 0, tag("produced at least one diar frame", dn).c_str());
    CHECK(d.probs.size() == static_cast<std::size_t>(d.num_frames) * d.num_speakers,
          tag("probs sized num_frames * num_speakers", dn).c_str());
    bool in01 = true;
    for (float p : d.probs) if (p < 0.0f || p > 1.0f) in01 = false;
    CHECK(in01, tag("every probability is in [0, 1]", dn).c_str());
    CHECK(std::abs(d.frame_seconds - 0.08) < 1e-9,
          tag("frame_seconds == 0.08", dn).c_str());

    // Deterministic re-run (continuous sigmoid output — exact match expected).
    auto d2 = m.diarize(audio);
    bool same = d2.probs == d.probs;
    CHECK(same, tag("diarize is deterministic across runs", dn).c_str());

    // Streaming session: for a sub-chunk clip the AOSC loop is a single chunk and
    // reduces to the offline forward — feed(is_last) must match diarize().
    {
        SortformerSession s = m.make_session();
        auto ds = m.feed(s, audio, /*is_last=*/true);
        CHECK(ds.num_speakers == t.spk,
              tag("streaming feed num_speakers", dn).c_str());
        CHECK(ds.num_frames > 0, tag("streaming feed produced frames", dn).c_str());
        bool sin01 = true;
        for (float p : ds.probs) if (p < 0.0f || p > 1.0f) sin01 = false;
        CHECK(sin01, tag("streaming probs in [0,1]", dn).c_str());
        // Single-chunk streaming == offline over the valid frames.
        const int n = std::min(ds.num_frames, d.num_frames);
        bool match = true;
        for (int i = 0; i < n * t.spk; ++i)
            if (std::abs(ds.probs[static_cast<std::size_t>(i)] -
                         d.probs[static_cast<std::size_t>(i)]) > 1e-4f)
                match = false;
        CHECK(match, tag("single-chunk streaming matches offline diarize", dn).c_str());
        m.reset(s);   // must not throw
        AudioBuffer empty; empty.sample_rate = 16000;
        auto pending = m.feed(s, audio, /*is_last=*/false);
        CHECK(pending.num_frames == 0,
              tag("feed without is_last buffers (returns no frames)", dn).c_str());
        auto flushed = m.feed(s, empty, /*is_last=*/true);
        CHECK(flushed.num_frames > 0,
              tag("feed flushes buffered audio on is_last", dn).c_str());
    }

    fs::remove_all(root);
    return failures;
}

// CUDA-only real-weights smoke: load the converted checkpoint (if present) and
// confirm a real diarization runs and is well-formed. CPU is skipped — the real
// 17-layer encoder is slow on CPU and the synthetic test already covers the
// contract; numerical parity is pinned by scripts/sortformer_parity*.py.
static int run_real_smoke(brotensor::Device dev, const char* dn) {
    using brosoundml::AudioBuffer;
    using brosoundml::Sortformer;

    const fs::path root = fs::path(BROSOUNDML_REPO_DIR) / "weights" / "sortformer" / "4spk-v2.1";
    if (!(fs::exists(root / "config.json") && fs::exists(root / "model.safetensors")))
        return 0;

    auto sp = std::make_shared<Sortformer>();
    sp->load(root.string(), dev);
    if (!sp->loaded()) { std::fprintf(stderr, "FAIL: [%s] real Sortformer load\n", dn); return 1; }

    AudioBuffer audio = sine(220.0f, 3.0f);   // 3 s, single chunk
    auto d = sp->diarize(audio);
    int fails = 0;
    auto expect = [&](bool ok, const char* msg) {
        if (!ok) { std::fprintf(stderr, "FAIL: [%s] %s\n", dn, msg); ++fails; }
    };
    expect(d.num_speakers == 4, "real Sortformer: 4 speakers");
    expect(d.num_frames > 30, "real Sortformer: ~37 frames for 3s");
    bool in01 = true;
    for (float p : d.probs) if (p < 0.0f || p > 1.0f) in01 = false;
    expect(in01, "real Sortformer: probabilities in [0,1]");
    if (fails == 0)
        std::printf("    [%s] real Sortformer: %d frames x 4 speakers, in-range\n",
                    dn, d.num_frames);
    return fails;
}

int main() {
    brotensor::init();
    try {
        int f = run_for_device(brotensor::Device::CPU, "CPU");
        if (brotensor::is_available(brotensor::Device::CUDA)) {
            f += run_for_device(brotensor::Device::CUDA, "CUDA");
            f += run_real_smoke(brotensor::Device::CUDA, "CUDA");
        } else {
            std::printf("test_sortformer: CUDA not available — CUDA path skipped\n");
        }
        if (f) {
            std::fprintf(stderr, "test_sortformer: %d failure(s)\n", f);
            return 1;
        }
        std::printf("test_sortformer: all checks passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_sortformer: exception: %s\n", e.what());
        return 1;
    }
}
