#include "brosoundml/whisper.h"

#include "brosoundml/detail/json.h"
#include "brosoundml/whisper_modules.h"

#include <brotensor/safetensors.h>

#include <algorithm>
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

    // Stage 2: log-mel front-end. Built once at load() time.
    LogMel            log_mel;
    // Stage 3: encoder. Holds all encoder-side weights + per-layer modules.
    WhisperEncoder    encoder;
    // Stage 4: decoder + KV cache. The cache is owned by Impl so consecutive
    // transcriptions reuse the same per-layer slabs (allocation-free decode
    // loop, once the loop lands in stage 5).
    WhisperDecoder    decoder;
    mutable WhisperKVCache cache;

    // Encode an audio buffer to (max_source_positions, d_model) hidden states.
    // Called from the future transcribe() path; exposed here so the test layer
    // can lock the contract before the decoder lands. The result is the cross-
    // attention key/value source the decoder consumes.
    void encode_audio(const AudioBuffer& audio,
                      brotensor::Tensor& hidden) const {
        brotensor::Tensor mel;
        log_mel.forward(audio, mel);
        encoder.forward(mel, hidden);
    }

    // Decoder prefill: T = prompt_ids.size(). Primes the cross-attn slot from
    // encoder_hidden, runs the decoder once over the whole prompt, and writes
    // the (T, vocab_size) logits to `logits`. After the call
    // `cache.size() == T`. The caller's greedy loop reads logits.row(T-1) to
    // pick the first generated token. NOTE: caller must `cache.reset()` first.
    void decode_prefill(const std::int32_t* prompt_ids, int T,
                        const brotensor::Tensor& encoder_hidden,
                        brotensor::Tensor& logits) const {
        decoder.prime_cross(encoder_hidden, cache);
        decoder.forward(prompt_ids, T, /*pos_offset=*/0, cache, logits);
    }

    // Single-step decode: run the decoder over one token at the current
    // cache length, writing (1, vocab_size) logits. The cache grows by one
    // row. The caller is responsible for `decode_prefill` having been called
    // first (cross-attn cache must already be primed).
    void decode_step(std::int32_t token_id,
                     brotensor::Tensor& logits) const {
        const int pos_offset = cache.size();
        decoder.forward(&token_id, /*T=*/1, pos_offset, cache, logits);
    }
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

    auto weights = brotensor::safetensors::File::open(weight_path.string());

    // Stage 2: build the log-mel front-end (no learnable weights). LogMel
    // keeps its host-built mel filterbank + Hann window CPU-resident and
    // uploads the final feature tensor to `device` at the end of forward().
    impl_->log_mel.build(impl_->config.num_mel_bins, device);

    // Stage 3: load the encoder backbone from the safetensors file.
    impl_->encoder.load_from(weights,
                             impl_->config.num_mel_bins,
                             impl_->config.d_model,
                             impl_->config.max_source_positions,
                             impl_->config.encoder_layers,
                             impl_->config.encoder_ffn_dim,
                             impl_->config.encoder_attention_heads,
                             device);

    // Stage 4: load the decoder + pre-allocate the per-layer KV cache. Cache
    // storage outlives every transcribe() call so the eventual greedy loop
    // never reallocates.
    impl_->decoder.load_from(weights,
                             impl_->config.d_model,
                             impl_->config.decoder_layers,
                             impl_->config.decoder_ffn_dim,
                             impl_->config.decoder_attention_heads,
                             impl_->config.vocab_size,
                             impl_->config.max_target_positions,
                             impl_->config.max_source_positions,
                             device);
    impl_->cache.allocate(impl_->config.decoder_layers,
                          impl_->config.d_model,
                          impl_->config.max_target_positions,
                          impl_->config.max_source_positions,
                          device);

    impl_->loaded = true;
}

namespace {

// Greedy argmax over the last row of a (T, V) logits tensor. The decode loop
// runs once per generated token, so we materialise the single (1, V) last
// row on the host and pick its argmax with a scalar loop — cheaper than
// allocating a (T, 1) argmax_rows output and downloading one int32.
int32_t argmax_last_row(const brotensor::Tensor& logits) {
    const int T = logits.rows;
    const int V = logits.cols;
    std::vector<float> row(static_cast<std::size_t>(V));
    if (logits.device == brotensor::Device::CPU) {
        const float* p = logits.host_f32() + static_cast<std::size_t>(T - 1) * V;
        std::copy(p, p + V, row.begin());
    } else {
        // Build a (1, V) view on the device over the final row, then move to
        // CPU. Tensor::to copies the buffer; the .data pointer is the device
        // buffer, and view + to is the standard scalar-readout path.
        float* last_row_dev = static_cast<float*>(logits.data)
                            + static_cast<std::size_t>(T - 1) * V;
        brotensor::Tensor last_view = brotensor::Tensor::view(
            logits.device, last_row_dev, 1, V, brotensor::Dtype::FP32);
        brotensor::Tensor last_host = last_view.to(brotensor::Device::CPU);
        const float* p = last_host.host_f32();
        std::copy(p, p + V, row.begin());
    }
    int   best_i = 0;
    float best_v = row[0];
    for (int v = 1; v < V; ++v) {
        if (row[static_cast<std::size_t>(v)] > best_v) {
            best_v = row[static_cast<std::size_t>(v)];
            best_i = v;
        }
    }
    return static_cast<int32_t>(best_i);
}

}  // namespace

Whisper::Transcription Whisper::transcribe(const AudioBuffer& audio,
                                           const std::vector<int32_t>& prompt_ids,
                                           int max_new_tokens) const {
    if (!impl_->loaded) {
        fail("Whisper::transcribe", "no model loaded; call Whisper::load() first");
    }
    if (audio.samples.empty()) {
        fail("Whisper::transcribe", "audio buffer is empty");
    }
    if (audio.sample_rate != impl_->config.sample_rate) {
        fail("Whisper::transcribe",
             "audio.sample_rate must be 16000 Hz (Whisper-fixed); "
             "resampling is the caller's responsibility");
    }
    if (prompt_ids.empty()) {
        fail("Whisper::transcribe",
             "prompt_ids must be non-empty "
             "(typically tokenizer.build_prompt(lang, task, with_timestamps))");
    }

    const int prompt_len = static_cast<int>(prompt_ids.size());
    if (prompt_len >= impl_->config.max_target_positions) {
        fail("Whisper::transcribe",
             "prompt_ids length >= max_target_positions; nothing to generate");
    }
    int budget = max_new_tokens;
    if (budget <= 0) {
        budget = impl_->config.max_target_positions - prompt_len;
    } else {
        const int hard_cap = impl_->config.max_target_positions - prompt_len;
        if (budget > hard_cap) budget = hard_cap;
    }

    // 1. Encode the audio once.
    brotensor::Tensor hidden;
    impl_->encode_audio(audio, hidden);

    // 2. Reset cache for this transcription, then prefill the prompt.
    impl_->cache.reset();
    brotensor::Tensor logits;
    impl_->decode_prefill(prompt_ids.data(), prompt_len, hidden, logits);

    // 3. Greedy loop. `logits` is mutated in-place by decode_step (no fresh
    // tensor per step, per project memory: Tensor copy ctor is deep).
    std::vector<int32_t> generated;
    generated.reserve(static_cast<std::size_t>(budget));
    for (int step = 0; step < budget; ++step) {
        const int32_t next_id = argmax_last_row(logits);
        if (next_id == impl_->config.eos_token_id) break;
        generated.push_back(next_id);
        if (static_cast<int>(prompt_len + generated.size())
            >= impl_->config.max_target_positions) {
            break;
        }
        impl_->decode_step(next_id, logits);
    }

    Transcription out;
    out.token_ids = prompt_ids;
    out.token_ids.insert(out.token_ids.end(),
                         generated.begin(), generated.end());
    return out;
}

const WhisperConfig& Whisper::config() const { return impl_->config; }
bool Whisper::loaded() const { return impl_->loaded; }

}
