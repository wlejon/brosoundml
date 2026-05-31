#include "brosoundml/qwen_tts.h"

#include "qwen_tts_codec.h"

#include "brosoundml/detail/json.h"

#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

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

const j::Value& obj_at(const j::Value& obj, const std::string& key,
                       const std::string& where) {
    const j::Value* v = obj.find(key);
    if (!v || !v->is_object()) {
        fail(where, "config.json missing required object '" + key + "'");
    }
    return *v;
}

// Shared Qwen3 block fields. `head_dim` may be absent in some sub-configs; it
// then falls back to hidden_size / num_attention_heads.
void parse_transformer(const j::Value& obj, QwenTtsTransformerConfig& t,
                       const std::string& where) {
    t.hidden_size         = int_at(obj, "hidden_size",         where);
    t.intermediate_size   = int_at(obj, "intermediate_size",   where);
    t.num_hidden_layers   = int_at(obj, "num_hidden_layers",   where);
    t.num_attention_heads = int_at(obj, "num_attention_heads", where);
    t.num_key_value_heads = obj.get_int("num_key_value_heads", t.num_attention_heads);
    t.head_dim            = obj.get_int("head_dim",
                                t.num_attention_heads > 0
                                    ? t.hidden_size / t.num_attention_heads : 0);
    t.rms_norm_eps        = obj.get_float("rms_norm_eps", 1e-6f);
    t.rope_theta          = obj.get_float("rope_theta", 1000000.0f);
}

void parse_talker(const j::Value& tk, QwenTtsTalkerConfig& c,
                  const std::string& where) {
    parse_transformer(tk, c.transformer, where);

    c.num_code_groups  = int_at(tk, "num_code_groups",  where);
    c.vocab_size       = int_at(tk, "vocab_size",       where);
    c.text_vocab_size  = int_at(tk, "text_vocab_size",  where);
    c.text_hidden_size = int_at(tk, "text_hidden_size", where);
    c.position_id_per_seconds = tk.get_int("position_id_per_seconds", 0);

    if (const j::Value* rs = tk.find("rope_scaling"); rs && rs->is_object()) {
        c.mrope_interleaved = rs->get_bool("interleaved", true);
        if (const j::Value* ms = rs->find("mrope_section");
            ms && ms->is_array()) {
            for (const auto& e : ms->as_array()) {
                c.mrope_section.push_back(static_cast<int>(e.as_number()));
            }
        }
    }

    c.codec_bos_id       = tk.get_int("codec_bos_id",        0);
    c.codec_eos_id       = tk.get_int("codec_eos_token_id",  0);
    c.codec_pad_id       = tk.get_int("codec_pad_id",        0);
    c.codec_think_id     = tk.get_int("codec_think_id",      0);
    c.codec_nothink_id   = tk.get_int("codec_nothink_id",    0);
    c.codec_think_bos_id = tk.get_int("codec_think_bos_id",  0);
    c.codec_think_eos_id = tk.get_int("codec_think_eos_id",  0);

    if (const j::Value* lid = tk.find("codec_language_id");
        lid && lid->is_object()) {
        for (const auto& m : lid->as_object()) {
            c.codec_language_id.emplace(m.first,
                                        static_cast<int>(m.second.as_number()));
        }
    }
    if (const j::Value* spk = tk.find("spk_id"); spk && spk->is_object()) {
        for (const auto& m : spk->as_object()) {
            c.spk_id.emplace(m.first, static_cast<int>(m.second.as_number()));
        }
    }
    // spk_is_dialect values are either `false` or a dialect string
    // (e.g. "sichuan_dialect"). Keep only the string form, "" otherwise.
    if (const j::Value* sd = tk.find("spk_is_dialect"); sd && sd->is_object()) {
        for (const auto& m : sd->as_object()) {
            c.spk_dialect.emplace(m.first,
                                  m.second.is_string() ? m.second.as_string()
                                                       : std::string());
        }
    }

    const j::Value& cp = obj_at(tk, "code_predictor_config", where);
    parse_transformer(cp, c.code_predictor.transformer, where);
    c.code_predictor.vocab_size      = int_at(cp, "vocab_size", where);
    c.code_predictor.num_code_groups = cp.get_int("num_code_groups",
                                                   c.num_code_groups);
}

QwenTtsVariant parse_variant(const std::string& s) {
    if (s == "custom_voice" || s == "customvoice") return QwenTtsVariant::CustomVoice;
    if (s == "voice_design" || s == "voicedesign") return QwenTtsVariant::VoiceDesign;
    return QwenTtsVariant::Base;  // "base" and anything unrecognised
}

QwenTtsConfig parse_model_config(const std::string& path) {
    const std::string text = slurp(path, "QwenTts::load");
    const j::Value    root = j::parse(text);
    if (!root.is_object()) fail("QwenTts::load", "config.json is not a JSON object");

    const std::string where = "QwenTts::load";
    QwenTtsConfig c;

    c.variant      = parse_variant(root.get_string("tts_model_type", "base"));
    c.model_size   = root.get_string("tts_model_size", "");
    c.tts_bos_id   = root.get_int("tts_bos_token_id", 0);
    c.tts_eos_id   = root.get_int("tts_eos_token_id", 0);
    c.tts_pad_id   = root.get_int("tts_pad_token_id", 0);
    c.im_start_id  = root.get_int("im_start_token_id", 0);
    c.im_end_id    = root.get_int("im_end_token_id", 0);
    c.assistant_id = root.get_int("assistant_token_id", 0);

    parse_talker(obj_at(root, "talker_config", where), c.talker, where);
    return c;
}

void parse_codec_config(const std::string& path, QwenTtsCodecConfig& c) {
    const std::string text = slurp(path, "QwenTts::load");
    const j::Value    root = j::parse(text);
    if (!root.is_object()) {
        fail("QwenTts::load", "speech_tokenizer/config.json is not a JSON object");
    }
    const std::string where = "QwenTts::load (codec)";

    c.output_sample_rate   = root.get_int("output_sample_rate", 24000);
    c.decode_upsample_rate = root.get_int("decode_upsample_rate", 0);

    const j::Value& dc = obj_at(root, "decoder_config", where);
    c.latent_dim              = int_at(dc, "latent_dim",      where);
    c.codebook_dim            = int_at(dc, "codebook_dim",    where);
    c.codebook_size           = int_at(dc, "codebook_size",   where);
    c.decoder_dim             = int_at(dc, "decoder_dim",     where);
    c.num_quantizers          = int_at(dc, "num_quantizers",  where);
    c.num_semantic_quantizers = dc.get_int("num_semantic_quantizers", 1);
    c.semantic_codebook_size  = dc.get_int("semantic_codebook_size", 0);
    c.sliding_window          = dc.get_int("sliding_window", 0);

    parse_transformer(dc, c.pre_transformer, where);
    // The codec uses a smaller rms_norm_eps default (1e-5) than the Talker.
    c.pre_transformer.rms_norm_eps = dc.get_float("rms_norm_eps", 1e-5f);
    c.pre_transformer.rope_theta   = dc.get_float("rope_theta", 10000.0f);

    c.upsample_rates    = dc.get_int_array("upsample_rates",    {});
    c.upsampling_ratios = dc.get_int_array("upsampling_ratios", {});
}

// Confirm a representative set of tensors is present so a wrong / truncated
// checkpoint fails fast at load() rather than mid-forward in a later stage.
void require_tensors(const brotensor::safetensors::File& f,
                     const std::vector<std::string>& names,
                     const std::string& which) {
    for (const auto& n : names) {
        if (!f.find(n)) {
            fail("QwenTts::load",
                 which + " missing required tensor '" + n + "'");
        }
    }
}

}  // namespace

// ─── QwenTts::Impl ──────────────────────────────────────────────────────────
//
// Holds the parsed config and the device the weights live on. As later stages
// land, the module graph (Talker, Code Predictor, codec decoder) is built here
// and weights are uploaded during load(); the safetensors files are only
// mmap'd for the duration of load().
struct QwenTts::Impl {
    QwenTtsConfig         config;
    brotensor::Device     device = brotensor::Device::CPU;
    bool                  loaded = false;
    QwenTtsCodecDecoder   codec;   // built in load(); runs decode_codes()
};

QwenTts::QwenTts() : impl_(std::make_unique<Impl>()) {}
QwenTts::~QwenTts() = default;
QwenTts::QwenTts(QwenTts&&) noexcept = default;
QwenTts& QwenTts::operator=(QwenTts&&) noexcept = default;

void QwenTts::load(const std::string& model_dir, brotensor::Device device) {
    brotensor::init();

    const fs::path dir          = model_dir;
    const fs::path cfg_path     = dir / "config.json";
    const fs::path weight_path  = dir / "model.safetensors";
    const fs::path codec_dir    = dir / "speech_tokenizer";
    const fs::path codec_cfg    = codec_dir / "config.json";
    const fs::path codec_weight = codec_dir / "model.safetensors";

    if (!fs::exists(cfg_path))
        fail("QwenTts::load", "no config.json under '" + model_dir + "'");
    if (!fs::exists(weight_path))
        fail("QwenTts::load", "no model.safetensors under '" + model_dir + "'");
    if (!fs::exists(codec_cfg))
        fail("QwenTts::load",
             "no speech_tokenizer/config.json under '" + model_dir + "'");
    if (!fs::exists(codec_weight))
        fail("QwenTts::load",
             "no speech_tokenizer/model.safetensors under '" + model_dir + "'");

    impl_->config = parse_model_config(cfg_path.string());
    parse_codec_config(codec_cfg.string(), impl_->config.codec);
    impl_->device = device;

    // Validate both checkpoints contain the modules later stages will load.
    const QwenTtsTalkerConfig& tk = impl_->config.talker;
    const int last_talker = tk.transformer.num_hidden_layers - 1;
    const int last_cp     = tk.code_predictor.transformer.num_hidden_layers - 1;
    {
        auto w = brotensor::safetensors::File::open(weight_path.string());
        require_tensors(w, {
            "talker.model.text_embedding.weight",
            "talker.model.codec_embedding.weight",
            "talker.text_projection.linear_fc2.weight",
            "talker.codec_head.weight",
            "talker.model.norm.weight",
            "talker.model.layers." + std::to_string(last_talker) +
                ".self_attn.q_proj.weight",
            "talker.code_predictor.lm_head.0.weight",
            "talker.code_predictor.model.layers." + std::to_string(last_cp) +
                ".input_layernorm.weight",
        }, "model.safetensors");
    }
    {
        auto w = brotensor::safetensors::File::open(codec_weight.string());
        require_tensors(w, {
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
            "decoder.pre_conv.conv.weight",
            "decoder.pre_transformer.norm.weight",
            "decoder.upsample.0.0.conv.weight",
            "decoder.decoder.0.conv.weight",
        }, "speech_tokenizer/model.safetensors");
        // Stage 2: build the codec decoder (codes -> waveform) from these
        // weights. The Talker / Code Predictor follow in later stages.
        impl_->codec.load(w, impl_->config.codec);
    }

    impl_->loaded = true;
}

AudioBuffer QwenTts::synthesize(const std::string& /*text*/,
                                const std::string& speaker,
                                const std::string& language) const {
    if (!impl_->loaded) {
        fail("QwenTts::synthesize", "no model loaded; call QwenTts::load() first");
    }
    // Surface the obvious caller errors before the staged stub so they're
    // reported as soon as the forward pass lands.
    const QwenTtsTalkerConfig& tk = impl_->config.talker;
    if (!tk.spk_id.empty() && tk.spk_id.find(speaker) == tk.spk_id.end()) {
        fail("QwenTts::synthesize", "unknown speaker '" + speaker + "'");
    }
    if (!tk.codec_language_id.empty() &&
        tk.codec_language_id.find(language) == tk.codec_language_id.end()) {
        fail("QwenTts::synthesize", "unsupported language '" + language + "'");
    }
    fail("QwenTts::synthesize",
         "forward pass not yet built (stage 2+: Talker / Code Predictor / "
         "codec decoder)");
}

AudioBuffer QwenTts::decode_codes(const std::vector<int32_t>& codes,
                                  int num_quantizers, int num_frames) const {
    if (!impl_->loaded) {
        fail("QwenTts::decode_codes", "no model loaded; call QwenTts::load() first");
    }
    if (num_quantizers <= 0 || num_frames < 0) {
        fail("QwenTts::decode_codes", "num_quantizers and num_frames must be positive");
    }
    if (codes.size() != static_cast<std::size_t>(num_quantizers) *
                            static_cast<std::size_t>(num_frames)) {
        fail("QwenTts::decode_codes",
             "codes size (" + std::to_string(codes.size()) + ") != num_quantizers*"
             "num_frames (" + std::to_string(num_quantizers) + "*" +
             std::to_string(num_frames) + ")");
    }
    std::vector<float> wav;
    impl_->codec.decode(codes.data(), num_quantizers, num_frames, wav);
    return AudioBuffer(std::move(wav), impl_->config.codec.output_sample_rate);
}

std::vector<std::string> QwenTts::speakers() const {
    std::vector<std::string> out;
    out.reserve(impl_->config.talker.spk_id.size());
    for (const auto& kv : impl_->config.talker.spk_id) out.push_back(kv.first);
    return out;
}

const QwenTtsConfig& QwenTts::config() const { return impl_->config; }
bool QwenTts::loaded() const { return impl_->loaded; }

}  // namespace brosoundml
