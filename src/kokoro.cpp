#include "brosoundml/kokoro.h"

#include "brosoundml/detail/json.h"

#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace fs = std::filesystem;
namespace j  = detail::json;

namespace {

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
    brotensor::safetensors::File weights;
};

Kokoro::Kokoro() : impl_(std::make_unique<Impl>()) {}
Kokoro::~Kokoro() = default;
Kokoro::Kokoro(Kokoro&&) noexcept = default;
Kokoro& Kokoro::operator=(Kokoro&&) noexcept = default;

void Kokoro::load(const std::string& model_dir, brotensor::Device device) {
    const fs::path dir         = model_dir;
    const fs::path config_path = dir / "config.json";
    const fs::path weight_path = dir / "model.safetensors";

    if (!fs::exists(config_path)) {
        fail("Kokoro::load", "no config.json under '" + model_dir + "'");
    }
    if (!fs::exists(weight_path)) {
        fail("Kokoro::load", "no model.safetensors under '" + model_dir + "'");
    }

    impl_->config  = parse_config(config_path.string());
    impl_->device  = device;
    impl_->weights = brotensor::safetensors::File::open(weight_path.string());
    impl_->loaded  = true;
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

AudioBuffer Kokoro::synthesize(const std::vector<int32_t>& /*phoneme_ids*/,
                               const Voice& /*voice*/,
                               float /*speed*/) const {
    if (!impl_->loaded) {
        fail("Kokoro::synthesize", "no model loaded; call Kokoro::load() first");
    }
    fail("Kokoro::synthesize",
         "the Kokoro forward pass is not yet implemented — "
         "stages 2–5 of the README build-out plan");
}

const KokoroConfig& Kokoro::config() const { return impl_->config; }
bool Kokoro::loaded() const { return impl_->loaded; }

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
    if (packs.device != brotensor::Device::CPU) {
        // brosoundml currently loads voice packs on the host (see load_voice).
        // A GPU-resident row slice belongs in a brotensor slice op, not here.
        fail("Voice::pick_for",
             "GPU-resident voice packs are not supported yet; load voice on CPU");
    }

    // Upstream Kokoro indexes voice[n_phonemes - 1]: row 0 carries the style
    // for a single-phoneme utterance, row k-1 for a k-phoneme utterance.
    brotensor::Tensor row = brotensor::Tensor::empty_on(
        brotensor::Device::CPU, 1, packs.cols, packs.dtype);
    const auto row_bytes = static_cast<std::size_t>(packs.cols)
                         * static_cast<std::size_t>(brotensor::dtype_size_bytes(packs.dtype));
    const auto offset = static_cast<std::size_t>(n_phonemes - 1) * row_bytes;
    std::memcpy(row.host_raw_mut(),
                static_cast<const uint8_t*>(packs.data) + offset,
                row_bytes);
    return row;
}

}  // namespace brosoundml
