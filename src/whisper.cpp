#include "brosoundml/whisper.h"

#include "brosoundml/detail/json.h"

#include <brotensor/safetensors.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

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

WhisperConfig parse_config(const std::string& path) {
    const std::string text = slurp(path, "Whisper::load");
    const j::Value    root = j::parse(text);
    if (!root.is_object()) fail("Whisper::load", "config.json is not a JSON object");

    WhisperConfig c;
    const std::string where = "Whisper::load";

    c.vocab_size              = int_at(root, "vocab_size",              where);
    c.num_mel_bins            = int_at(root, "num_mel_bins",            where);
    c.d_model                 = int_at(root, "d_model",                 where);
    c.max_source_positions    = int_at(root, "max_source_positions",    where);
    c.max_target_positions    = int_at(root, "max_target_positions",    where);
    c.encoder_layers          = int_at(root, "encoder_layers",          where);
    c.encoder_attention_heads = int_at(root, "encoder_attention_heads", where);
    c.encoder_ffn_dim         = int_at(root, "encoder_ffn_dim",         where);
    c.decoder_layers          = int_at(root, "decoder_layers",          where);
    c.decoder_attention_heads = int_at(root, "decoder_attention_heads", where);
    c.decoder_ffn_dim         = int_at(root, "decoder_ffn_dim",         where);
    c.pad_token_id            = root.get_int("pad_token_id",           0);
    c.eos_token_id            = root.get_int("eos_token_id",           0);
    c.decoder_start_token_id  = root.get_int("decoder_start_token_id", 0);
    return c;
}

}  // namespace

// ─── Whisper::Impl ─────────────────────────────────────────────────────────
//
// Stage 1: holds the parsed config and the device. Submodules (encoder,
// decoder, mel front-end) get added as later stages land — at which point
// load() will populate them from the safetensors file the same way Kokoro
// does.
struct Whisper::Impl {
    WhisperConfig     config;
    brotensor::Device device = brotensor::Device::CPU;
    bool              loaded = false;
};

Whisper::Whisper() : impl_(std::make_unique<Impl>()) {}
Whisper::~Whisper() = default;
Whisper::Whisper(Whisper&&) noexcept = default;
Whisper& Whisper::operator=(Whisper&&) noexcept = default;

void Whisper::load(const std::string& model_dir, brotensor::Device device) {
    const fs::path dir         = model_dir;
    const fs::path config_path = dir / "config.json";
    const fs::path weight_path = dir / "model.safetensors";

    if (!fs::exists(config_path)) {
        fail("Whisper::load", "no config.json under '" + model_dir + "'");
    }
    if (!fs::exists(weight_path)) {
        fail("Whisper::load", "no model.safetensors under '" + model_dir + "'");
    }

    impl_->config = parse_config(config_path.string());
    impl_->device = device;

    // Stage 1 only opens-and-closes the safetensors file to confirm it's a
    // valid container. Per-submodule uploads land in later stages.
    {
        auto weights = brotensor::safetensors::File::open(weight_path.string());
        (void)weights;
    }

    impl_->loaded = true;
}

Whisper::Transcription Whisper::transcribe(const AudioBuffer& audio,
                                           const std::vector<int32_t>& prompt_ids,
                                           int max_new_tokens) const {
    if (!impl_->loaded) {
        fail("Whisper::transcribe", "no model loaded; call Whisper::load() first");
    }
    (void)audio;
    (void)prompt_ids;
    (void)max_new_tokens;
    fail("Whisper::transcribe",
         "not yet implemented — stage 2 (log-mel front-end + encoder) pending");
}

const WhisperConfig& Whisper::config() const { return impl_->config; }
bool Whisper::loaded() const { return impl_->loaded; }

}
