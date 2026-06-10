#include "brosoundml/qwen_asr.h"

#include "qwen_asr_decoder.h"
#include "qwen_asr_encoder.h"
#include "qwen_tts_device.h"   // qtd:: device-neutral helpers (model-agnostic)

#include "brosoundml/detail/json.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <cstdint>
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
namespace bt = brotensor;

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

QwenAsrConfig parse_config(const std::string& config_path,
                           const std::string& gen_config_path) {
    const std::string where = "QwenAsr::load";
    const j::Value root = j::parse(slurp(config_path, where));
    if (!root.is_object()) fail(where, "config.json is not a JSON object");

    const j::Value& tk = obj_at(root, "thinker_config", where);
    const j::Value& ac = obj_at(tk, "audio_config", where);
    const j::Value& tx = obj_at(tk, "text_config", where);

    QwenAsrConfig c;
    c.num_mel_bins            = int_at(ac, "num_mel_bins", where);
    c.d_model                 = int_at(ac, "d_model", where);
    c.encoder_layers          = int_at(ac, "encoder_layers", where);
    c.encoder_attention_heads = int_at(ac, "encoder_attention_heads", where);
    c.encoder_ffn_dim         = int_at(ac, "encoder_ffn_dim", where);
    c.output_dim              = int_at(ac, "output_dim", where);
    c.downsample_hidden_size  = int_at(ac, "downsample_hidden_size", where);
    c.n_window                = int_at(ac, "n_window", where);
    c.n_window_infer          = int_at(ac, "n_window_infer", where);
    c.max_source_positions    = ac.get_int("max_source_positions", 1500);

    c.hidden_size         = int_at(tx, "hidden_size", where);
    c.num_hidden_layers   = int_at(tx, "num_hidden_layers", where);
    c.num_attention_heads = int_at(tx, "num_attention_heads", where);
    c.num_key_value_heads = tx.get_int("num_key_value_heads", c.num_attention_heads);
    c.head_dim            = tx.get_int("head_dim",
                                c.num_attention_heads > 0
                                    ? c.hidden_size / c.num_attention_heads : 0);
    c.intermediate_size   = int_at(tx, "intermediate_size", where);
    c.vocab_size          = int_at(tx, "vocab_size", where);
    c.rms_norm_eps        = tx.get_float("rms_norm_eps", 1e-6f);
    c.rope_theta          = tx.get_float("rope_theta", 1000000.0f);

    c.audio_start_token_id = int_at(tk, "audio_start_token_id", where);
    c.audio_end_token_id   = int_at(tk, "audio_end_token_id", where);
    c.audio_token_id       = int_at(tk, "audio_token_id", where);

    // Latent-tap geometry (the encode() surface): width is the decoder hidden
    // (= output_dim); rate is the mel frame rate (sample_rate / hop, hop = 160)
    // after the conv stem's 8x time downsample (three stride-2 convs).
    c.latent_dim = c.output_dim;
    c.latent_hz  = static_cast<float>(c.sample_rate) / 160.0f / 8.0f;

    // EOS set from generation_config.json when present (eos_token_id is an
    // id array there); the documented Qwen3-ASR pair otherwise.
    c.eos_token_ids = {151643, 151645};   // <|endoftext|>, <|im_end|>
    if (fs::exists(gen_config_path)) {
        const j::Value g = j::parse(slurp(gen_config_path, where));
        if (g.is_object()) {
            const std::vector<int> eos = g.get_int_array("eos_token_id", {});
            if (!eos.empty()) {
                c.eos_token_ids.assign(eos.begin(), eos.end());
            }
        }
    }
    return c;
}

// Fixed Qwen BPE ids for the pieces of the ASR chat template (tokenizer-level
// facts duplicated here so the decoder can run without a brolm handle, the
// same stance Whisper takes on its special ids). The template is
//   <|im_start|>system\n{context}<|im_end|>\n
//   <|im_start|>user\n<|audio_start|>{audio}<|audio_end|><|im_end|>\n
//   <|im_start|>assistant\n
// with {audio} = one <|audio_pad|> per encoder output token, replaced by the
// audio embeddings at prefill.
constexpr std::int32_t kImStart   = 151644;   // <|im_start|>
constexpr std::int32_t kImEnd     = 151645;   // <|im_end|>
constexpr std::int32_t kSystem    = 8948;     // "system"
constexpr std::int32_t kUser      = 872;      // "user"
constexpr std::int32_t kAssistant = 77091;    // "assistant"
constexpr std::int32_t kNewline   = 198;      // "\n"

}  // namespace

// ─── Impl ───────────────────────────────────────────────────────────────────

struct QwenAsr::Impl {
    QwenAsrConfig  config;
    QwenAsrEncoder encoder;
    QwenAsrDecoder decoder;
    bt::Device     device = bt::Device::CPU;
    bool           loaded = false;
};

QwenAsr::QwenAsr() : impl_(std::make_unique<Impl>()) {}
QwenAsr::~QwenAsr() = default;
QwenAsr::QwenAsr(QwenAsr&&) noexcept = default;
QwenAsr& QwenAsr::operator=(QwenAsr&&) noexcept = default;

const QwenAsrConfig& QwenAsr::config() const { return impl_->config; }
bool QwenAsr::loaded() const { return impl_->loaded; }

bt::Tensor QwenAsr::encode(const AudioBuffer& audio) const {
    const std::string where = "QwenAsr::encode";
    if (!impl_->loaded) fail(where, "load() not called");
    bt::DeviceScope scope(impl_->device);
    bt::Tensor latents;
    impl_->encoder.forward(audio, latents);   // (T, latent_dim) on the model device
    return latents;
}

int QwenAsr::encode_to_host(const AudioBuffer& audio,
                            std::vector<float>& out) const {
    const bt::Tensor latents = encode(audio);
    const int T = latents.rows;
    out.resize(static_cast<std::size_t>(T) * latents.cols);
    qtd::to_host(latents, out.data());
    return T;
}

void QwenAsr::load(const std::string& model_dir, bt::Device device) {
    const std::string where = "QwenAsr::load";
    const fs::path dir(model_dir);
    const fs::path config_path  = dir / "config.json";
    const fs::path weights_path = dir / "model.safetensors";
    if (!fs::exists(config_path))
        fail(where, "no config.json under '" + model_dir + "'");
    if (!fs::exists(weights_path))
        fail(where, "no model.safetensors under '" + model_dir + "'");

    impl_->config = parse_config(config_path.string(),
                                 (dir / "generation_config.json").string());
    impl_->device = device;

    auto f = bt::safetensors::File::open(weights_path.string());

    impl_->encoder.load(f, impl_->config, device);
    impl_->decoder.load(f, impl_->config, device);
    impl_->loaded = true;
}

QwenAsr::Transcription QwenAsr::transcribe(const AudioBuffer& audio,
                                           const TranscribeOptions& opts) const {
    const std::string where = "QwenAsr::transcribe";
    if (!impl_->loaded) fail(where, "load() not called");
    const Impl& im = *impl_;
    const QwenAsrConfig& cfg = im.config;
    const bt::Device dev = im.device;
    bt::DeviceScope scope(dev);

    // ── 1. audio -> encoder hidden states ──
    // Exactly the latent tap encode() exposes — one encoder path, not two.
    bt::Tensor audio_embeds = encode(audio);   // (n_audio, latent_dim)
    const int n_audio = audio_embeds.rows;

    // ── 2. chat-template prompt around the audio block ──
    std::vector<std::int32_t> prefix = {kImStart, kSystem, kNewline};
    prefix.insert(prefix.end(), opts.context_ids.begin(), opts.context_ids.end());
    prefix.insert(prefix.end(),
                  {kImEnd, kNewline, kImStart, kUser, kNewline,
                   static_cast<std::int32_t>(cfg.audio_start_token_id)});
    const std::vector<std::int32_t> suffix = {
        static_cast<std::int32_t>(cfg.audio_end_token_id), kImEnd, kNewline,
        kImStart, kAssistant, kNewline};

    const int n_pre = static_cast<int>(prefix.size());
    const int n_suf = static_cast<int>(suffix.size());
    const int P = n_pre + n_audio + n_suf;

    // Text embeddings for the template tokens; the audio rows come straight
    // from the encoder (upstream scatters them over the <|audio_pad|>
    // positions — here those positions are a contiguous block, so the prompt
    // embedding stream is a 3-part concat).
    bt::Tensor embeds = bt::Tensor::zeros_on(dev, P, cfg.hidden_size,
                                             bt::Dtype::FP32);
    {
        bt::Tensor pre = qtd::gather_rows(im.decoder.embed_tokens, prefix);
        bt::Tensor suf = qtd::gather_rows(im.decoder.embed_tokens, suffix);
        const int H = cfg.hidden_size;
        bt::copy_d2d(pre, 0, embeds, 0, n_pre * H);
        bt::copy_d2d(audio_embeds, 0, embeds, n_pre * H, n_audio * H);
        bt::copy_d2d(suf, 0, embeds, (n_pre + n_audio) * H, n_suf * H);
    }

    // ── 3. prefill + greedy decode ──
    QwenAsrDecoderCache cache;
    cache.reset(im.decoder.num_layers);

    auto is_eos = [&cfg](std::int32_t id) {
        for (std::int32_t e : cfg.eos_token_ids)
            if (id == e) return true;
        return false;
    };

    // One greedy step: hidden -> last-row logits -> on-device argmax. The
    // (1,1) INT32 index stays on the device for the next embedding gather;
    // a 4-byte readback hands the id to the EOS check / output stream.
    auto greedy_step = [&im, dev](const bt::Tensor& hidden,
                                  bt::Tensor& idx_dev) -> std::int32_t {
        bt::Tensor logits;
        im.decoder.logits_last(hidden, logits);          // (1, vocab)
        idx_dev = bt::Tensor::empty_on(dev, 1, 1, bt::Dtype::INT32);
        bt::argmax_rows(logits, idx_dev);
        const bt::Tensor host = idx_dev.to(bt::Device::CPU);
        return *static_cast<const std::int32_t*>(host.data);
    };

    Transcription out;
    const int budget = opts.max_new_tokens > 0 ? opts.max_new_tokens : 1024;

    bt::Tensor hidden;
    im.decoder.run_dev(embeds, P, &cache, hidden);
    bt::Tensor idx_dev;
    std::int32_t token = greedy_step(hidden, idx_dev);

    while (static_cast<int>(out.token_ids.size()) < budget) {
        if (is_eos(token)) break;
        out.token_ids.push_back(token);
        if (opts.on_token) opts.on_token(token);
        if (opts.cancel && opts.cancel()) break;
        bt::Tensor emb = qtd::gather_rows(im.decoder.embed_tokens, idx_dev);
        im.decoder.run_dev(emb, 1, &cache, hidden);
        token = greedy_step(hidden, idx_dev);
    }
    return out;
}

}  // namespace brosoundml
