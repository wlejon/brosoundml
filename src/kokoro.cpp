#include "brosoundml/kokoro.h"

#include "brosoundml/detail/json.h"
#include "brosoundml/kokoro_modules.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

// Forward decl — implemented in kokoro_modules.cpp. The Kokoro `load_from`
// chain doesn't take a `device` parameter (the public submodule headers are
// `load_from(File, ...)` only), so this is how kokoro.cpp tells the upload
// helper which device to migrate every weight onto. Called once before the
// load block and reset to CPU after it.
void set_kokoro_load_device(brotensor::Device d);

namespace fs = std::filesystem;
namespace j  = detail::json;

namespace {

// Print min/mean/max/std of a tensor's host FP32 data to stderr, prefixed by
// `tag`. Triggered only when `BROSOUNDML_DEBUG_STAGES=1`. Lets us spot the
// first stage whose stats diverge from the Python reference.
void debug_stats(const char* tag, const brotensor::Tensor& t) {
    static const bool on = []() {
        const char* v = std::getenv("BROSOUNDML_DEBUG_STAGES");
        return v && v[0] && v[0] != '0';
    }();
    if (!on) return;
    if (t.dtype != brotensor::Dtype::FP32) {
        std::fprintf(stderr, "[stage] %-24s dtype != FP32\n", tag);
        return;
    }
    // Stats are a debug-only host operation; if the tensor lives on a GPU,
    // round-trip it through to_host_vector() before reading.
    std::vector<float> host_buf;
    const float* d = nullptr;
    if (t.device == brotensor::Device::CPU) {
        d = t.host_f32();
    } else {
        host_buf = t.to_host_vector();
        d = host_buf.data();
    }
    const std::size_t n = t.size();
    if (n == 0) { std::fprintf(stderr, "[stage] %-24s empty\n", tag); return; }
    float mn = d[0], mx = d[0];
    double sum = 0, sumsq = 0;
    for (std::size_t i = 0; i < n; ++i) {
        float v = d[i];
        if (v < mn) mn = v; if (v > mx) mx = v;
        sum += v; sumsq += static_cast<double>(v) * v;
    }
    const double mean = sum / static_cast<double>(n);
    const double var  = sumsq / static_cast<double>(n) - mean * mean;
    const double std  = var > 0 ? std::sqrt(var) : 0.0;
    std::fprintf(stderr,
        "[stage] %-24s rows=%d cols=%d  min=%+.4f  max=%+.4f  mean=%+.4f  std=%.4f\n",
        tag, t.rows, t.cols, mn, mx, mean, std);
}

void debug_vec(const char* tag, const std::vector<int>& v) {
    static const bool on = []() {
        const char* x = std::getenv("BROSOUNDML_DEBUG_STAGES");
        return x && x[0] && x[0] != '0';
    }();
    if (!on) return;
    int sum = 0, mn = (v.empty() ? 0 : v[0]), mx = mn;
    for (int x : v) { sum += x; if (x < mn) mn = x; if (x > mx) mx = x; }
    std::fprintf(stderr,
        "[stage] %-24s n=%zu  sum=%d  min=%d  max=%d\n",
        tag, v.size(), sum, mn, mx);
}


[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

std::string slurp(const std::string& path, const std::string& where) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail(where, "cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int int_at(const j::Value& obj, const std::string& key, const std::string& where) {
    const j::Value* v = obj.find(key);
    if (!v) fail(where, "config.json missing required key '" + key + "'");
    return static_cast<int>(v->as_number());
}

std::vector<int> int_array(const j::Value& v) {
    std::vector<int> out;
    out.reserve(v.as_array().size());
    for (const auto& e : v.as_array()) out.push_back(static_cast<int>(e.as_number()));
    return out;
}

void parse_decoder(const j::Value& obj, IStftNetConfig& dec, const std::string& where) {
    dec.upsample_initial_channel = int_at(obj, "upsample_initial_channel", where);
    dec.gen_istft_n_fft          = int_at(obj, "gen_istft_n_fft",          where);
    dec.gen_istft_hop_size       = int_at(obj, "gen_istft_hop_size",       where);
    dec.upsample_kernel_sizes    = int_array(obj.at("upsample_kernel_sizes"));
    dec.upsample_rates           = int_array(obj.at("upsample_rates"));
    dec.resblock_kernel_sizes    = int_array(obj.at("resblock_kernel_sizes"));
    for (const auto& row : obj.at("resblock_dilation_sizes").as_array()) {
        dec.resblock_dilation_sizes.push_back(int_array(row));
    }
}

void parse_plbert(const j::Value& obj, PLBertConfig& pl, const std::string& where) {
    pl.hidden_size             = int_at(obj, "hidden_size",             where);
    pl.num_attention_heads     = int_at(obj, "num_attention_heads",     where);
    pl.intermediate_size       = int_at(obj, "intermediate_size",       where);
    pl.max_position_embeddings = int_at(obj, "max_position_embeddings", where);
    pl.num_hidden_layers       = int_at(obj, "num_hidden_layers",       where);
    // vocab_size: prefer the plbert-local key, otherwise fall back to the
    // outer n_token (Kokoro's plBERT shares its vocab with the phoneme set).
    const j::Value* vs = obj.find("vocab_size");
    pl.vocab_size = vs ? static_cast<int>(vs->as_number()) : 0;
}

KokoroConfig parse_config(const std::string& path) {
    const std::string text = slurp(path, "Kokoro::load");
    const j::Value    root = j::parse(text);
    if (!root.is_object()) fail("Kokoro::load", "config.json is not a JSON object");

    KokoroConfig c;
    const std::string where = "Kokoro::load";

    c.n_tokens                 = int_at(root, "n_token",                  where);
    c.hidden_dim               = int_at(root, "hidden_dim",               where);
    c.style_dim                = int_at(root, "style_dim",                where);
    c.n_layer                  = int_at(root, "n_layer",                  where);
    c.max_dur                  = int_at(root, "max_dur",                  where);
    c.text_encoder_kernel_size = int_at(root, "text_encoder_kernel_size", where);
    // Optional fields — not every Kokoro-style config exposes them.
    c.n_mels       = root.get_int("n_mels",       0);
    c.dim_in       = root.get_int("dim_in",       0);
    c.max_conv_dim = root.get_int("max_conv_dim", 0);

    // Upstream Kokoro names the iSTFTNet sub-config `istftnet`; some forks
    // and our synthetic test fixture use `decoder`. Accept either.
    const j::Value* decoder_obj = root.find("istftnet");
    if (!decoder_obj) decoder_obj = root.find("decoder");
    if (!decoder_obj) fail(where, "config.json missing 'istftnet' (or 'decoder')");
    parse_decoder(*decoder_obj, c.decoder, where);
    parse_plbert (root.at("plbert"),  c.plbert,  where);
    if (c.plbert.vocab_size == 0) c.plbert.vocab_size = c.n_tokens;

    if (const j::Value* vocab = root.find("vocab"); vocab && vocab->is_object()) {
        for (const auto& m : vocab->as_object()) {
            c.vocab.emplace(m.first, static_cast<int>(m.second.as_number()));
        }
    }
    return c;
}

}  // namespace

// ─── Kokoro::Impl ──────────────────────────────────────────────────────────
//
// Holds the parsed config, the device the weights live on, and the open
// safetensors file (mmap'd). Tensor uploads happen lazily as the module
// graph is built out in later stages — the file stays alive for the life
// of the Kokoro instance so TensorView pointers remain valid until each
// upload finishes.
struct Kokoro::Impl {
    KokoroConfig                config;
    brotensor::Device           device = brotensor::Device::CPU;
    bool                        loaded = false;
    // Submodules are populated by Kokoro::load. The safetensors file is only
    // alive for the duration of load(); the per-submodule upload calls copy
    // the weights out, so we don't keep the mmap around afterwards.
    PLBert                      bert;
    BertEncoder                 bert_encoder;
    TextEncoder                 text_encoder;
    Predictor                   predictor;
    DecoderBackbone             decoder;
    HarmonicSource              hsource;
    Generator                   generator;

    // The decoder back half (decoder ▶ harmonic source ▶ generator ▶ PCM),
    // shared by synthesize() and decode_from(). Defined out-of-line below.
    AudioBuffer decode_back_half(const brotensor::Tensor& ref_s,
                                 const brotensor::Tensor& asr,
                                 const brotensor::Tensor& F0_pred,
                                 const brotensor::Tensor& N_pred,
                                 int total,
                                 const CancelCheck& cancel,
                                 KokoroTrace* trace_out);
};

Kokoro::Kokoro() : impl_(std::make_unique<Impl>()) {}
Kokoro::~Kokoro() = default;
Kokoro::Kokoro(Kokoro&&) noexcept = default;
Kokoro& Kokoro::operator=(Kokoro&&) noexcept = default;

void Kokoro::load(const std::string& model_dir, brotensor::Device device) {
    // Ensure brotensor has probed all available backends; required before any
    // non-CPU device is reachable. Idempotent — safe to call from every entry
    // point.
    brotensor::init();

    const fs::path dir         = model_dir;
    const fs::path config_path = dir / "config.json";
    const fs::path weight_path = dir / "model.safetensors";

    if (!fs::exists(config_path)) {
        fail("Kokoro::load", "no config.json under '" + model_dir + "'");
    }
    if (!fs::exists(weight_path)) {
        fail("Kokoro::load", "no model.safetensors under '" + model_dir + "'");
    }

    impl_->config = parse_config(config_path.string());
    impl_->device = device;

    // Open the safetensors file just long enough to upload every submodule's
    // weights — release the mmap as soon as we leave this scope so callers can
    // delete the source file without first destructing the Kokoro instance.
    {
        auto weights = brotensor::safetensors::File::open(weight_path.string());
        // Tell the file-local upload helper in kokoro_modules.cpp to migrate
        // every weight to `device`. Reset on exit so other Kokoro instances
        // (or a later CPU-only load) start from a clean CPU default.
        set_kokoro_load_device(device);
        struct DevReset {
            ~DevReset() { set_kokoro_load_device(brotensor::Device::CPU); }
        } _reset;
        impl_->bert.load_from         (weights, impl_->config.plbert);
        impl_->bert_encoder.load_from (weights, impl_->config.plbert.hidden_size,
                                                impl_->config.hidden_dim);
        impl_->text_encoder.load_from (weights, impl_->config);
        impl_->predictor.load_from    (weights, impl_->config);
        impl_->decoder.load_from      (weights);
        impl_->hsource.load_from      (weights, impl_->config);
        impl_->generator.load_from    (weights, impl_->config);
    }

    impl_->loaded = true;
}

Voice Kokoro::load_voice(const std::string& voice_path) const {
    if (!impl_->loaded) {
        fail("Kokoro::load_voice", "no model loaded; call Kokoro::load() first");
    }

    const int voice_dim = 2 * impl_->config.style_dim;
    if (voice_dim <= 0) {
        fail("Kokoro::load_voice",
             "style_dim is 0 in the loaded config — refusing to load a voice");
    }

    std::ifstream f(voice_path, std::ios::binary | std::ios::ate);
    if (!f) fail("Kokoro::load_voice", "cannot open '" + voice_path + "'");
    const std::streamsize bytes = f.tellg();
    f.seekg(0);

    const std::size_t elem_bytes = sizeof(float);
    if (bytes <= 0 ||
        static_cast<std::size_t>(bytes) % (voice_dim * elem_bytes) != 0) {
        fail("Kokoro::load_voice",
             "voice file size " + std::to_string(bytes) +
             " bytes is not a multiple of voice_dim * 4 (" +
             std::to_string(voice_dim * elem_bytes) +
             ") for '" + voice_path + "'");
    }
    const int rows = static_cast<int>(
        static_cast<std::size_t>(bytes) / (voice_dim * elem_bytes));

    std::vector<float> buf(static_cast<std::size_t>(rows) * voice_dim);
    if (!f.read(reinterpret_cast<char*>(buf.data()), bytes)) {
        fail("Kokoro::load_voice", "short read on '" + voice_path + "'");
    }

    Voice voice;
    voice.name  = fs::path(voice_path).stem().string();
    voice.packs = brotensor::Tensor::from_host_on(brotensor::Device::CPU,
                                                  buf.data(), rows, voice_dim);
    return voice;
}

Voice Kokoro::make_voice(const std::vector<float>& style,
                         const std::string& name) const {
    if (!impl_) fail("Kokoro::make_voice", "no model loaded; call Kokoro::load() first");
    const int voice_dim = 2 * impl_->config.style_dim;
    if (voice_dim <= 0)
        fail("Kokoro::make_voice", "style_dim is 0 in the loaded config");
    if (style.empty() ||
        style.size() % static_cast<std::size_t>(voice_dim) != 0) {
        fail("Kokoro::make_voice",
             "style length " + std::to_string(style.size()) +
             " is not a positive multiple of voice_dim (" +
             std::to_string(voice_dim) + ")");
    }

    // A single style point is broadcast across the upstream pack height (510
    // rows, one style per utterance length) so it renders for any phoneme
    // count; a full table is taken verbatim.
    constexpr int kBroadcastRows = 510;
    std::vector<float> buf;
    int rows;
    if (style.size() == static_cast<std::size_t>(voice_dim)) {
        rows = kBroadcastRows;
        buf.resize(static_cast<std::size_t>(rows) * voice_dim);
        for (int r = 0; r < rows; ++r) {
            std::memcpy(buf.data() + static_cast<std::size_t>(r) * voice_dim,
                        style.data(), voice_dim * sizeof(float));
        }
    } else {
        rows = static_cast<int>(style.size() / voice_dim);
        buf  = style;
    }

    Voice voice;
    voice.name  = name;
    voice.packs = brotensor::Tensor::from_host_on(brotensor::Device::CPU,
                                                  buf.data(), rows, voice_dim);
    return voice;
}

// The decoder back half: decoder backbone ▶ harmonic source ▶ generator ▶ PCM.
// Shared by synthesize() (fresh predictor output) and decode_from() (edited
// intermediates). `asr` (1 × hidden*total), `F0_pred`/`N_pred` (1 × 2*total),
// and `ref_s` (1 × 2*style_dim) all live on impl.device; `total` is the frame
// count (= asr width). Mirrors the upstream order; the only approximation is
// the harmonic-source branch (see the synthesize() comment / README caveat).
AudioBuffer Kokoro::Impl::decode_back_half(const brotensor::Tensor& ref_s,
                                           const brotensor::Tensor& asr,
                                           const brotensor::Tensor& F0_pred,
                                           const brotensor::Tensor& N_pred,
                                           int total,
                                           const CancelCheck& cancel,
                                           KokoroTrace* trace_out) {
    const brotensor::Device dev = device;

    // Decoder backbone -> generator input.
    brotensor::Tensor gen_in = brotensor::Tensor::empty_on(dev, 0, 0, brotensor::Dtype::FP32);
    decoder.forward(asr, F0_pred, N_pred, ref_s, total, gen_in);
    const int L_gen = 2 * total;
    debug_stats("09_gen_in", gen_in);
    if (trace_out) trace_out->add("gen_in", config.hidden_dim, L_gen, gen_in);

    // Generator. har_frames matches the time axis at the Generator's iSTFT input:
    // L_gen -> ×∏(upsample_rates) -> reflection_pad (+1).
    int upsample_prod = 1;
    for (int r : config.decoder.upsample_rates) upsample_prod *= r;
    const int har_frames = L_gen * upsample_prod + 1;
    const int har_channels = config.decoder.gen_istft_n_fft + 2;
    brotensor::Tensor har_stub = brotensor::Tensor::empty_on(dev, 0, 0, brotensor::Dtype::FP32);
    int sig_len = 0, hframes = 0;
    hsource.forward(F0_pred, /*frame_count=*/2 * total, sig_len, hframes, har_stub);
    if (hframes != har_frames) {
        fail("Kokoro::decode",
             "har_frames mismatch: hsource=" + std::to_string(hframes) +
             " vs expected=" + std::to_string(har_frames));
    }
    (void)har_channels;

    // The decoder style is ref_s[:, :style_dim]; slice it via copy_d2d.
    brotensor::Tensor style_dec = brotensor::Tensor::zeros_on(
        dev, 1, config.style_dim, brotensor::Dtype::FP32);
    brotensor::copy_d2d(ref_s, 0, style_dec, 0, config.style_dim);

    debug_stats("10_har_stub", har_stub);
    if (trace_out) trace_out->add("har", har_channels, hframes, har_stub);

    if (cancel && cancel()) return {};

    brotensor::Tensor audio_t = brotensor::Tensor::empty_on(dev, 0, 0, brotensor::Dtype::FP32);
    generator.forward(gen_in, L_gen, har_stub, har_frames, style_dec, audio_t, cancel);
    debug_stats("11_audio", audio_t);
    if (trace_out) trace_out->add("audio", 1, static_cast<int>(audio_t.size()), audio_t);

    AudioBuffer out;
    out.sample_rate = config.sample_rate;
    out.samples = audio_t.to_host_vector();
    return out;
}

AudioBuffer Kokoro::synthesize(const std::vector<int32_t>& phoneme_ids,
                               const Voice& voice,
                               float speed,
                               std::vector<int32_t>* pred_dur_out,
                               const CancelCheck& cancel,
                               KokoroTrace* trace_out) const {
    if (!impl_->loaded) {
        fail("Kokoro::synthesize", "no model loaded; call Kokoro::load() first");
    }
    if (phoneme_ids.empty()) {
        fail("Kokoro::synthesize", "phoneme_ids is empty");
    }
    if (voice.packs.rows <= 0 || voice.packs.cols != 2 * impl_->config.style_dim) {
        fail("Kokoro::synthesize", "voice pack shape does not match 2*style_dim");
    }

    // Wrap phoneme ids with the upstream BOS/EOS convention: [0, ...ids, 0].
    std::vector<int32_t> ids;
    ids.reserve(phoneme_ids.size() + 2);
    ids.push_back(0);
    ids.insert(ids.end(), phoneme_ids.begin(), phoneme_ids.end());
    ids.push_back(0);
    const int L = static_cast<int>(ids.size());
    if (trace_out) trace_out->add_ints("phonemes", ids);

    // Style row: upstream picks ref_s = voice[len(input_ids) - 1]. Voice packs
    // are loaded on CPU (raw host data on disk); upload the selected row to
    // the model's device so every downstream op sees device-matched operands.
    brotensor::Tensor ref_s_host = voice.pick_for(L);
    brotensor::Tensor ref_s = ref_s_host.to(impl_->device);

    // 1. plBERT. Pre-allocate every out-tensor on the model's device — a
    // default-constructed brotensor::Tensor lives on CPU and brotensor's CUDA
    // dispatch refuses mixed-device calls (Tensor::resize preserves device).
    const brotensor::Device dev = impl_->device;
    brotensor::Tensor bert_dur = brotensor::Tensor::empty_on(dev, 0, 0, brotensor::Dtype::FP32);
    impl_->bert.forward(ids, /*attention_mask=*/{}, bert_dur);
    debug_stats("01_bert_dur", bert_dur);
    if (trace_out) trace_out->add("bert_dur", L, impl_->config.plbert.hidden_size, bert_dur);

    // 2. bert_encoder.
    brotensor::Tensor d_en = brotensor::Tensor::empty_on(dev, 0, 0, brotensor::Dtype::FP32);
    impl_->bert_encoder.forward(bert_dur, d_en);
    debug_stats("02_d_en", d_en);
    if (trace_out) trace_out->add("d_en", impl_->config.hidden_dim, L, d_en);

    // 3. text_encoder (StyleTTS2 phoneme CNN+BiLSTM).
    brotensor::Tensor t_en = brotensor::Tensor::empty_on(dev, 0, 0, brotensor::Dtype::FP32);
    impl_->text_encoder.forward(ids, /*text_mask=*/{}, t_en);
    debug_stats("06_t_en", t_en);
    if (trace_out) trace_out->add("t_en", impl_->config.hidden_dim, L, t_en);

    // 4. Predictor: duration + F0 + N.
    Predictor::Output po;
    impl_->predictor.forward(d_en, ref_s, L, speed, po);
    debug_stats("05_F0_pred", po.F0_pred);
    debug_stats("05_N_pred",  po.N_pred);
    debug_vec  ("03_pred_dur", po.pred_dur);
    if (trace_out) {
        trace_out->add_ints("pred_dur", po.pred_dur);
        trace_out->add("F0_pred", 1, static_cast<int>(po.F0_pred.size()), po.F0_pred);
        trace_out->add("N_pred",  1, static_cast<int>(po.N_pred.size()),  po.N_pred);
    }

    // Surface the per-phoneme frame counts when the caller asked for them.
    // po.pred_dur is indexed over the BOS/EOS-wrapped `ids` (length L), so the
    // output has one entry per wrapped token.
    if (pred_dur_out) *pred_dur_out = po.pred_dur;

    // Cooperative cancellation: the front stages (plBERT / encoders / predictor)
    // are comparatively cheap; bail before the expensive length-regulate +
    // decoder + generator back half if the turn was barged in.
    if (cancel && cancel()) return {};

    // 5. Length-regulate t_en into asr. The expansion is an irregular gather
    //    along L (each phoneme is repeated pred_dur[l] times); there's no
    //    brotensor op for this NCL-axis gather, so build the result on host
    //    via to_host_vector / from_host_on. The work is O(C*total) memory —
    //    negligible compared to the device-side decoder forward that follows.
    const int total = std::accumulate(po.pred_dur.begin(), po.pred_dur.end(), 0);
    const int C_hidden = impl_->config.hidden_dim;
    std::vector<float> asr_host(static_cast<std::size_t>(C_hidden) * total);
    {
        const std::vector<float> te_host = t_en.to_host_vector();
        int t = 0;
        for (int l = 0; l < L; ++l) {
            const int reps = po.pred_dur[l];
            for (int r = 0; r < reps; ++r) {
                for (int c = 0; c < C_hidden; ++c) {
                    asr_host[static_cast<std::size_t>(c) * total + t] =
                        te_host[static_cast<std::size_t>(c) * L + l];
                }
                ++t;
            }
        }
    }
    brotensor::Tensor asr = brotensor::Tensor::from_host_on(
        impl_->device, asr_host.data(), 1, C_hidden * total);

    debug_stats("07_asr", asr);
    if (trace_out) trace_out->add("asr", C_hidden, total, asr);

    if (cancel && cancel()) return {};

    // 6. Decoder ▶ harmonic source ▶ generator. Factored so decode_from() can
    //    re-run exactly this back half from edited intermediates.
    return impl_->decode_back_half(ref_s, asr, po.F0_pred, po.N_pred, total,
                                   cancel, trace_out);
}

AudioBuffer Kokoro::synthesize_stream(
    const std::vector<std::vector<int32_t>>& phoneme_chunks,
    const Voice& voice,
    const KokoroStreamChunkFn& on_chunk,
    float speed,
    const CancelCheck& cancel) const {
    if (!impl_->loaded) {
        fail("Kokoro::synthesize_stream",
             "no model loaded; call Kokoro::load() first");
    }

    AudioBuffer out;
    out.sample_rate = impl_->config.sample_rate;

    for (const std::vector<int32_t>& chunk : phoneme_chunks) {
        if (cancel && cancel()) break;   // barge-in: drop remaining chunks
        if (chunk.empty()) continue;     // nothing to synthesize for this chunk

        // Each chunk is a self-contained utterance — synthesize() applies the
        // BOS/EOS wrap, duration prediction, and length-aware voice row. It
        // returns an empty buffer if cancelled mid-chunk. Capture the chunk's
        // per-phoneme durations so on_chunk can align words to the streamed
        // audio without a second synthesize().
        std::vector<int32_t> pred_dur;
        AudioBuffer seg = synthesize(chunk, voice, speed,
                                     &pred_dur, cancel,
                                     /*trace_out=*/nullptr);
        if (seg.samples.empty()) continue;

        if (on_chunk) on_chunk(seg.samples.data(),
                               static_cast<int>(seg.samples.size()),
                               pred_dur.data(),
                               static_cast<int>(pred_dur.size()));
        out.samples.insert(out.samples.end(),
                           seg.samples.begin(), seg.samples.end());
    }
    return out;
}

AudioBuffer Kokoro::decode_from(const Voice& voice,
                                int n_phonemes_wrapped,
                                const std::vector<float>& asr_host,
                                int total,
                                const std::vector<float>& F0_host,
                                const std::vector<float>& N_host,
                                const CancelCheck& cancel,
                                KokoroTrace* trace_out) const {
    if (!impl_->loaded) {
        fail("Kokoro::decode_from", "no model loaded; call Kokoro::load() first");
    }
    if (voice.packs.rows <= 0 || voice.packs.cols != 2 * impl_->config.style_dim) {
        fail("Kokoro::decode_from", "voice pack shape does not match 2*style_dim");
    }
    const int C_hidden = impl_->config.hidden_dim;
    if (total <= 0) fail("Kokoro::decode_from", "total must be > 0");
    if (static_cast<long long>(asr_host.size()) !=
        static_cast<long long>(C_hidden) * total) {
        fail("Kokoro::decode_from",
             "asr size " + std::to_string(asr_host.size()) + " != hidden_dim*total (" +
             std::to_string(C_hidden) + "*" + std::to_string(total) + ")");
    }
    if (static_cast<int>(F0_host.size()) != 2 * total ||
        static_cast<int>(N_host.size())  != 2 * total) {
        fail("Kokoro::decode_from",
             "F0_pred/N_pred size must be 2*total (" + std::to_string(2 * total) + ")");
    }

    // Pick the same style row the originating synthesize() used (the phoneme
    // count is invariant under prosody edits) and upload the edited grids.
    brotensor::Tensor ref_s = voice.pick_for(n_phonemes_wrapped).to(impl_->device);
    brotensor::Tensor asr = brotensor::Tensor::from_host_on(
        impl_->device, asr_host.data(), 1, C_hidden * total);
    brotensor::Tensor F0 = brotensor::Tensor::from_host_on(
        impl_->device, F0_host.data(), 1, 2 * total);
    brotensor::Tensor N = brotensor::Tensor::from_host_on(
        impl_->device, N_host.data(), 1, 2 * total);

    return impl_->decode_back_half(ref_s, asr, F0, N, total, cancel, trace_out);
}

DecoderLoraContext KokoroDecodePrep::context() const {
    DecoderLoraContext ctx;
    ctx.asr     = &asr;
    ctx.F0_pred = &F0_pred;
    ctx.N_pred  = &N_pred;
    ctx.ref_s   = &ref_s;
    ctx.har     = &har;
    ctx.total   = total;
    ctx.frames  = frames;
    return ctx;
}

// Front half + harmonic source, packaged for the decoder-LoRA trainer. Mirrors
// the synthesize() front half (plBERT ▶ encoders ▶ predictor ▶ length-regulate)
// and the decode_back_half har/frames block — but stops before the decoder
// backbone and generator, which the DecoderLora runs itself (cached, with the
// LoRA affines). Deliberately a parallel copy of the inference front half so
// synthesize()/decode_from() stay untouched.
KokoroDecodePrep Kokoro::prepare_decode_context(
    const std::vector<int32_t>& phoneme_ids,
    const Voice& voice,
    float speed,
    const std::vector<float>* F0_override,
    const std::vector<float>* N_override) const {
    if (!impl_->loaded) {
        fail("Kokoro::prepare_decode_context",
             "no model loaded; call Kokoro::load() first");
    }
    if (phoneme_ids.empty()) {
        fail("Kokoro::prepare_decode_context", "phoneme_ids is empty");
    }
    if (voice.packs.rows <= 0 ||
        voice.packs.cols != 2 * impl_->config.style_dim) {
        fail("Kokoro::prepare_decode_context",
             "voice pack shape does not match 2*style_dim");
    }

    const brotensor::Device dev = impl_->device;

    // BOS/EOS wrap, matching synthesize().
    std::vector<int32_t> ids;
    ids.reserve(phoneme_ids.size() + 2);
    ids.push_back(0);
    ids.insert(ids.end(), phoneme_ids.begin(), phoneme_ids.end());
    ids.push_back(0);
    const int L = static_cast<int>(ids.size());

    brotensor::Tensor ref_s = voice.pick_for(L).to(dev);

    // 1. plBERT ▶ 2. bert_encoder.
    brotensor::Tensor bert_dur = brotensor::Tensor::empty_on(dev, 0, 0, brotensor::Dtype::FP32);
    impl_->bert.forward(ids, /*attention_mask=*/{}, bert_dur);
    brotensor::Tensor d_en = brotensor::Tensor::empty_on(dev, 0, 0, brotensor::Dtype::FP32);
    impl_->bert_encoder.forward(bert_dur, d_en);

    // 3. text_encoder.
    brotensor::Tensor t_en = brotensor::Tensor::empty_on(dev, 0, 0, brotensor::Dtype::FP32);
    impl_->text_encoder.forward(ids, /*text_mask=*/{}, t_en);

    // 4. predictor: durations + F0 + N.
    Predictor::Output po;
    impl_->predictor.forward(d_en, ref_s, L, speed, po);

    // 5. length-regulate t_en → asr (irregular per-phoneme gather, host-side —
    //    identical to synthesize()).
    const int total = std::accumulate(po.pred_dur.begin(), po.pred_dur.end(), 0);
    if (total <= 0) {
        fail("Kokoro::prepare_decode_context",
             "predicted total frame count is 0");
    }
    const int C_hidden = impl_->config.hidden_dim;
    std::vector<float> asr_host(static_cast<std::size_t>(C_hidden) * total);
    {
        const std::vector<float> te_host = t_en.to_host_vector();
        int t = 0;
        for (int l = 0; l < L; ++l) {
            const int reps = po.pred_dur[l];
            for (int r = 0; r < reps; ++r) {
                for (int c = 0; c < C_hidden; ++c) {
                    asr_host[static_cast<std::size_t>(c) * total + t] =
                        te_host[static_cast<std::size_t>(c) * L + l];
                }
                ++t;
            }
        }
    }

    KokoroDecodePrep prep;
    prep.total     = total;
    prep.ref_s     = std::move(ref_s);
    prep.asr       = brotensor::Tensor::from_host_on(dev, asr_host.data(), 1,
                                                     C_hidden * total);
    prep.backbone  = &impl_->decoder;
    prep.generator = &impl_->generator;

    // F0 / N: predicted, or caller-supplied prosody (each 2*total floats).
    auto take_grid = [&](const std::vector<float>* ov, brotensor::Tensor pred,
                         const char* which) -> brotensor::Tensor {
        if (!ov) return pred;
        if (static_cast<int>(ov->size()) != 2 * total) {
            fail("Kokoro::prepare_decode_context",
                 std::string(which) + " override size must be 2*total (" +
                 std::to_string(2 * total) + ")");
        }
        return brotensor::Tensor::from_host_on(dev, ov->data(), 1, 2 * total);
    };
    prep.F0_pred = take_grid(F0_override, std::move(po.F0_pred), "F0");
    prep.N_pred  = take_grid(N_override,  std::move(po.N_pred),  "N");

    // Harmonic source — mirrors decode_back_half (kokoro.cpp:362-375). frames is
    // the iSTFT-input time axis: L_gen ▸ ×∏(upsample_rates) ▸ reflect-pad (+1).
    int upsample_prod = 1;
    for (int r : impl_->config.decoder.upsample_rates) upsample_prod *= r;
    const int L_gen = 2 * total;
    const int har_frames = L_gen * upsample_prod + 1;
    brotensor::Tensor har_stub = brotensor::Tensor::empty_on(dev, 0, 0, brotensor::Dtype::FP32);
    int sig_len = 0, hframes = 0;
    impl_->hsource.forward(prep.F0_pred, /*frame_count=*/2 * total, sig_len, hframes,
                           har_stub);
    if (hframes != har_frames) {
        fail("Kokoro::prepare_decode_context",
             "har_frames mismatch: hsource=" + std::to_string(hframes) +
             " vs expected=" + std::to_string(har_frames));
    }
    prep.frames = hframes;
    prep.har    = std::move(har_stub);

    return prep;
}

const KokoroConfig& Kokoro::config() const { return impl_->config; }
bool Kokoro::loaded() const { return impl_->loaded; }

// ─── KokoroTrace ─────────────────────────────────────────────────────────────

void KokoroTrace::add(std::string name, int h, int w, const brotensor::Tensor& t) {
    std::vector<float> host = t.to_host_vector();
    const int n = static_cast<int>(host.size());
    if (h <= 0 || w <= 0 || static_cast<long long>(h) * w != n) { h = 1; w = n; }
    stages.push_back(Stage{std::move(name), h, w, std::move(host)});
}

void KokoroTrace::add_ints(std::string name, const std::vector<int32_t>& v) {
    std::vector<float> host(v.begin(), v.end());
    const int w = static_cast<int>(host.size());
    stages.push_back(Stage{std::move(name), 1, w, std::move(host)});
}

// ─── Voice ─────────────────────────────────────────────────────────────────

brotensor::Tensor Voice::pick_for(int n_phonemes) const {
    if (packs.rows <= 0 || packs.cols <= 0) {
        fail("Voice::pick_for", "voice pack is empty");
    }
    if (n_phonemes < 1 || n_phonemes > packs.rows) {
        fail("Voice::pick_for",
             "n_phonemes=" + std::to_string(n_phonemes) +
             " out of range [1, " + std::to_string(packs.rows) + "]");
    }
    // Upstream Kokoro indexes voice[n_phonemes - 1]: row 0 carries the style
    // for a single-phoneme utterance, row k-1 for a k-phoneme utterance. The
    // packs tensor stays on whichever device load_voice put it on (CPU today
    // — file data — but the slice is correct for any device); the row copy
    // uses copy_d2d so a future device-resident voice pack works transparently.
    brotensor::Tensor row = brotensor::Tensor::empty_on(
        packs.device, 1, packs.cols, packs.dtype);
    brotensor::copy_d2d(packs, (n_phonemes - 1) * packs.cols,
                        row, 0, packs.cols);
    return row;
}

}  // namespace brosoundml
