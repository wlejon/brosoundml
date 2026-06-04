#include "brosoundml/qwen_tts.h"

#include "qwen_tts_codec.h"
#include "qwen_tts_codec_encoder.h"
#include "qwen_tts_speaker_encoder.h"
#include "qwen_tts_code_predictor.h"
#include "qwen_tts_generate.h"
#include "qwen_tts_talker.h"

#include "brosoundml/detail/json.h"

#include <brolm/qwen_tokenizer.h>

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
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

    // Base ships an ECAPA-TDNN speaker encoder (config.json carries only enc_dim
    // + sample_rate; the channel/kernel/dilation lists and mel params are the
    // upstream defaults). Absent on CustomVoice / VoiceDesign.
    if (const j::Value* se = root.find("speaker_encoder_config");
        se && se->is_object()) {
        QwenTtsSpeakerEncoderConfig& s = c.speaker_encoder;
        s.present     = true;
        s.mel_dim     = se->get_int("mel_dim", 128);
        s.enc_dim     = se->get_int("enc_dim", 1024);
        s.sample_rate = se->get_int("sample_rate", 24000);
        s.enc_channels     = se->get_int_array("enc_channels", {512, 512, 512, 512, 1536});
        s.enc_kernel_sizes = se->get_int_array("enc_kernel_sizes", {5, 3, 3, 3, 1});
        s.enc_dilations    = se->get_int_array("enc_dilations", {1, 2, 3, 4, 1});
        s.res2net_scale       = se->get_int("enc_res2net_scale", 8);
        s.se_channels         = se->get_int("enc_se_channels", 128);
        s.attention_channels  = se->get_int("enc_attention_channels", 128);
    }
    return c;
}

void parse_codec_config(const std::string& path, QwenTtsCodecConfig& c) {
    const std::string text = slurp(path, "QwenTts::load");
    const j::Value    root = j::parse(text);
    if (!root.is_object()) {
        fail("QwenTts::load", "speech_tokenizer/config.json is not a JSON object");
    }
    const std::string where = "QwenTts::load (codec)";

    c.output_sample_rate     = root.get_int("output_sample_rate", 24000);
    c.decode_upsample_rate   = root.get_int("decode_upsample_rate", 0);
    c.encode_downsample_rate = root.get_int("encode_downsample_rate", 0);

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

    // ── encoder (the HF-Mimi analysis stack: waveform -> codes) ──
    const j::Value& ec = obj_at(root, "encoder_config", where);
    QwenTtsCodecEncoderConfig& e = c.encoder;
    e.num_filters          = int_at(ec, "num_filters",          where);
    e.kernel_size          = int_at(ec, "kernel_size",          where);
    e.last_kernel_size     = int_at(ec, "last_kernel_size",     where);
    e.residual_kernel_size = int_at(ec, "residual_kernel_size", where);
    e.compress             = int_at(ec, "compress",             where);
    e.codebook_dim         = int_at(ec, "codebook_dim",         where);
    e.ratios               = ec.get_int_array("upsampling_ratios", {});
    e.sliding_window       = ec.get_int("sliding_window", 0);
    e.valid_num_quantizers = root.get_int("encoder_valid_num_quantizers",
                                          c.num_quantizers);
    parse_transformer(ec, e.transformer, where);
    // encoder_transformer norms are LayerNorm (eps "norm_eps"); reuse the field.
    e.transformer.rms_norm_eps = ec.get_float("norm_eps", 1e-5f);
    e.transformer.rope_theta   = ec.get_float("rope_theta", 10000.0f);
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
    QwenTtsCodecDecoder   codec;       // built in load(); runs decode_codes()
    QwenTtsCodecEncoder   codec_enc;   // built in load(); runs encode_audio()
    QwenTtsSpeakerEncoder spk_enc;     // Base only: ref clip -> x-vector
    QwenTtsTalker         talker;      // 28-layer Qwen3 decoder backbone
    QwenTtsCodePredictor  code_pred;   // 5-layer depth transformer
    std::unique_ptr<brolm::qwen::Tokenizer> tokenizer;  // text -> Qwen BPE ids
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
        // Stages 3-4: build the Talker + Code Predictor (the AR generator).
        impl_->talker.load(w, impl_->config.talker, device);
        impl_->code_pred.load(w, impl_->config.talker.code_predictor,
                              impl_->config.talker.transformer.hidden_size, device);

        // Base ships an ECAPA-TDNN speaker encoder (in this same checkpoint) for
        // the zero-shot voice clone. Loaded host-side (enrollment is one-shot).
        if (impl_->config.speaker_encoder.present) {
            require_tensors(w, {
                "speaker_encoder.blocks.0.conv.weight",
                "speaker_encoder.asp.tdnn.conv.weight",
                "speaker_encoder.fc.weight",
            }, "model.safetensors");
            impl_->spk_enc.load(w, impl_->config.speaker_encoder);
        }
    }

    // Text tokenizer (Qwen byte-level BPE). The chat prompt uses the
    // <|im_start|> / <|im_end|> specials, which the tokenizer auto-registers
    // from vocab.json.
    {
        const fs::path vocab  = dir / "vocab.json";
        const fs::path merges = dir / "merges.txt";
        if (!fs::exists(vocab) || !fs::exists(merges))
            fail("QwenTts::load",
                 "no vocab.json / merges.txt under '" + model_dir + "'");
        impl_->tokenizer = std::make_unique<brolm::qwen::Tokenizer>(
            brolm::qwen::Tokenizer::load(vocab.string(), merges.string()));
        // The ChatML control tokens live in tokenizer_config.json's added
        // tokens (ids above vocab.json's range), so register them explicitly
        // from the config — the prompt is built from <|im_start|> / <|im_end|>.
        impl_->tokenizer->register_special_token("<|im_start|>", impl_->config.im_start_id);
        impl_->tokenizer->register_special_token("<|im_end|>",   impl_->config.im_end_id);
    }
    {
        auto w = brotensor::safetensors::File::open(codec_weight.string());
        require_tensors(w, {
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
            "decoder.pre_conv.conv.weight",
            "decoder.pre_transformer.norm.weight",
            "decoder.upsample.0.0.conv.weight",
            "decoder.decoder.0.conv.weight",
            "encoder.encoder.layers.0.conv.weight",
            "encoder.encoder_transformer.layers.0.self_attn.q_proj.weight",
            "encoder.downsample.conv.weight",
            "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight",
            "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight",
        }, "speech_tokenizer/model.safetensors");
        // Build the codec decoder (codes -> waveform) and encoder (waveform ->
        // codes) from these weights. The Talker / Code Predictor follow below.
        impl_->codec.load(w, impl_->config.codec, device);
        impl_->codec_enc.load(w, impl_->config.codec, device);
    }

    impl_->loaded = true;
}

AudioBuffer QwenTts::synthesize(const std::string& text,
                                const std::string& speaker,
                                const std::string& language,
                                const std::string& instruct,
                                const CancelCheck& cancel) const {
    if (!impl_->loaded) {
        fail("QwenTts::synthesize", "no model loaded; call QwenTts::load() first");
    }
    const QwenTtsConfig&       cfg = impl_->config;
    const QwenTtsTalkerConfig& tk  = cfg.talker;

    // Resolve speaker / language to codec tokens (and apply the dialect
    // override the upstream pipeline does for dialect speakers). A model with no
    // speaker presets (VoiceDesign) has an empty spk_id table — leave spk_id -1.
    // The speaker name is then only consulted for the dialect override below.
    int spk_id = -1;
    if (!tk.spk_id.empty()) {
        auto it = tk.spk_id.find(speaker);
        if (it == tk.spk_id.end()) fail("QwenTts::synthesize", "unknown speaker '" + speaker + "'");
        spk_id = it->second;
    }
    int language_id = -1;
    const bool is_auto = (language == "auto");
    if (!is_auto) {
        auto it = tk.codec_language_id.find(language);
        if (it == tk.codec_language_id.end())
            fail("QwenTts::synthesize", "unsupported language '" + language + "'");
        language_id = it->second;
    }
    if (is_auto || language == "chinese") {
        auto dit = tk.spk_dialect.find(speaker);
        if (dit != tk.spk_dialect.end() && !dit->second.empty()) {
            auto lit = tk.codec_language_id.find(dit->second);
            if (lit != tk.codec_language_id.end()) language_id = lit->second;
        }
    }

    // Tokenize the body chat prompt (the text to speak).
    const std::string prompt =
        "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    const std::vector<int32_t> input_ids = impl_->tokenizer->encode(prompt);
    if (input_ids.size() < 9) {  // role(3) + >=1 body + trailing(5)
        fail("QwenTts::synthesize", "prompt tokenized to too few tokens");
    }

    // VoiceDesign (and 1.7B CustomVoice) take a natural-language voice
    // instruction, tokenized as its own user turn and prepended to the prefill.
    // Empty = no instruction (the only mode the 0.6B CustomVoice checkpoint
    // supports). The 0.6B CustomVoice ignores any instruct, matching upstream.
    std::vector<int32_t> instruct_ids;
    const bool ignore_instruct =
        (cfg.variant == QwenTtsVariant::CustomVoice && cfg.model_size == "0b6");
    if (!instruct.empty() && !ignore_instruct) {
        const std::string ins_prompt = "<|im_start|>user\n" + instruct + "<|im_end|>\n";
        instruct_ids = impl_->tokenizer->encode(ins_prompt);
    }

    return synth_core(input_ids, instruct_ids, spk_id, /*spk_embed=*/nullptr,
                      language_id, cancel);
}

AudioBuffer QwenTts::synthesize_clone(const std::string& text,
                                      const AudioBuffer& ref,
                                      const std::string& language,
                                      const CancelCheck& cancel) const {
    if (!impl_->loaded) {
        fail("QwenTts::synthesize_clone", "no model loaded; call QwenTts::load() first");
    }
    if (!impl_->config.speaker_encoder.present) {
        fail("QwenTts::synthesize_clone",
             "loaded checkpoint has no speaker encoder; the zero-shot clone "
             "needs a Base-variant model");
    }
    if (ref.samples.empty()) {
        fail("QwenTts::synthesize_clone", "reference audio is empty");
    }
    const QwenTtsConfig&       cfg = impl_->config;
    const QwenTtsTalkerConfig& tk  = cfg.talker;

    // Resolve language (Base has no preset speakers / dialects).
    int language_id = -1;
    if (language != "auto") {
        auto it = tk.codec_language_id.find(language);
        if (it == tk.codec_language_id.end())
            fail("QwenTts::synthesize_clone", "unsupported language '" + language + "'");
        language_id = it->second;
    }

    // Enroll: reference clip -> 24 kHz mono -> ECAPA-TDNN x-vector (host floats).
    const std::vector<float> spk_embed = embed_speaker(ref);

    // Tokenize the body chat prompt (the text to speak).
    const std::string prompt =
        "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    const std::vector<int32_t> input_ids = impl_->tokenizer->encode(prompt);
    if (input_ids.size() < 9) {
        fail("QwenTts::synthesize_clone", "prompt tokenized to too few tokens");
    }

    return synth_core(input_ids, /*instruct_ids=*/{}, /*spk_id=*/-1,
                      spk_embed.data(), language_id, cancel);
}

std::vector<float> QwenTts::embed_speaker(const AudioBuffer& ref) const {
    if (!impl_->loaded) {
        fail("QwenTts::embed_speaker", "no model loaded; call QwenTts::load() first");
    }
    if (!impl_->config.speaker_encoder.present) {
        fail("QwenTts::embed_speaker",
             "loaded checkpoint has no speaker encoder; needs a Base-variant model");
    }
    if (ref.samples.empty()) {
        fail("QwenTts::embed_speaker", "reference audio is empty");
    }

    // Reference clip -> the encoder's rate (24 kHz) mono -> ECAPA-TDNN x-vector.
    const int sr = impl_->config.speaker_encoder.sample_rate;
    const float* wav = ref.samples.data();
    int n = static_cast<int>(ref.samples.size());
    std::vector<float> resampled;
    if (ref.sample_rate != sr && ref.sample_rate > 0) {
        const int n_out = static_cast<int>(
            std::llround(static_cast<double>(n) * sr / ref.sample_rate));
        brotensor::Tensor x =
            brotensor::Tensor::from_host_on(brotensor::Device::CPU, wav, 1, n);
        brotensor::Tensor y;
        brotensor::resample1d_forward(x, /*N=*/1, /*C=*/1, n, n_out, /*mode=*/1, y);
        resampled.assign(y.host_f32(), y.host_f32() + n_out);
        wav = resampled.data();
        n = n_out;
    }
    return impl_->spk_enc.embed(wav, n);
}

AudioBuffer QwenTts::synth_core(const std::vector<int32_t>& input_ids,
                                const std::vector<int32_t>& instruct_ids,
                                int spk_id, const float* spk_embed,
                                int language_id, const CancelCheck& cancel) const {
    const QwenTtsConfig&       cfg = impl_->config;
    const QwenTtsTalkerConfig& tk  = cfg.talker;

    // Assemble the prefill embedding stream + trailing-text embeddings.
    std::vector<float> prefill, trailing, tts_pad;
    int T = 0, L = 0;
    assemble_talker_prefill(impl_->talker, cfg, input_ids, instruct_ids, spk_id,
                            spk_embed, language_id, prefill, T, trailing, L, tts_pad);

    // Prefill M-RoPE positions: plain 0..T-1 on all three axes (the Talker's
    // get_rope_index is cumsum of an all-ones mask), rope_delta 0.
    std::vector<int32_t> pos3(static_cast<std::size_t>(3) * T);
    for (int a = 0; a < 3; ++a)
        for (int t = 0; t < T; ++t) pos3[a * T + t] = t;

    // Run the dual-track AR loop with the upstream codebook-0 logits policy.
    QwenTtsGenParams gp;
    gp.eos_id      = tk.codec_eos_id;
    gp.max_frames  = 4096;
    gp.rope_delta  = 0;
    gp.suppress_lo = tk.vocab_size - 1024;
    gp.suppress_hi = tk.vocab_size;
    gp.min_frames  = 2;
    gp.repetition_penalty = 1.05f;
    std::vector<int32_t> frames;
    const int F = generate_codes(impl_->talker, impl_->code_pred, prefill.data(),
                                 T, pos3.data(), trailing.data(), L, tts_pad.data(),
                                 gp, frames, cancel);
    // Cancelled mid-loop: discard the partial code stream and return silence.
    if (cancel && cancel()) return AudioBuffer{};
    const int G = tk.num_code_groups;

    // Transpose frame-major [F][G] to the codebook-major layout decode_codes
    // expects: codes[k * F + t].
    std::vector<int32_t> codes(static_cast<std::size_t>(G) * F);
    for (int t = 0; t < F; ++t)
        for (int k = 0; k < G; ++k)
            codes[static_cast<std::size_t>(k) * F + t] = frames[static_cast<std::size_t>(t) * G + k];

    return decode_codes(codes, G, F);
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

std::vector<int32_t> QwenTts::encode_audio(const AudioBuffer& ref,
                                           int* num_frames_out) const {
    if (!impl_->loaded) {
        fail("QwenTts::encode_audio", "no model loaded; call QwenTts::load() first");
    }
    if (ref.samples.empty()) {
        fail("QwenTts::encode_audio", "reference audio is empty");
    }
    const int target = impl_->config.codec.output_sample_rate;

    // Resample to the codec rate if needed (linear; the reference clip is
    // preprocessing, outside the bit-exact code path). Mono already — AudioBuffer
    // is single-channel.
    const float* wav = ref.samples.data();
    int n = static_cast<int>(ref.samples.size());
    std::vector<float> resampled;
    if (ref.sample_rate != target && ref.sample_rate > 0) {
        const int n_out = static_cast<int>(
            std::llround(static_cast<double>(n) * target / ref.sample_rate));
        brotensor::Tensor x =
            brotensor::Tensor::from_host_on(brotensor::Device::CPU, wav, 1, n);
        brotensor::Tensor y;
        brotensor::resample1d_forward(x, /*N=*/1, /*C=*/1, n, n_out, /*mode=*/1, y);
        resampled.assign(y.host_f32(), y.host_f32() + n_out);
        wav = resampled.data();
        n = n_out;
    }

    std::vector<int32_t> codes;
    const int T = impl_->codec_enc.encode(wav, n, codes);
    if (num_frames_out) *num_frames_out = T;
    return codes;
}

std::vector<std::string> QwenTts::speakers() const {
    std::vector<std::string> out;
    out.reserve(impl_->config.talker.spk_id.size());
    for (const auto& kv : impl_->config.talker.spk_id) out.push_back(kv.first);
    return out;
}

std::vector<std::string> QwenTts::languages() const {
    std::vector<std::string> out;
    for (const auto& kv : impl_->config.talker.codec_language_id)
        if (kv.first.find("dialect") == std::string::npos) out.push_back(kv.first);
    return out;
}

std::string QwenTts::speaker_dialect(const std::string& speaker) const {
    auto it = impl_->config.talker.spk_dialect.find(speaker);
    return it != impl_->config.talker.spk_dialect.end() ? it->second : std::string();
}

const QwenTtsConfig& QwenTts::config() const { return impl_->config; }
bool QwenTts::loaded() const { return impl_->loaded; }

}  // namespace brosoundml
