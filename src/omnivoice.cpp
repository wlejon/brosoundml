// OmniVoice pipeline — see include/brosoundml/omnivoice.h and docs/omnivoice.md.
//
// Ports OmniVoice.generate() / create_voice_clone_prompt() / _preprocess_all /
// _generate_chunked / _decode_and_post_process of upstream
// omnivoice/models/omnivoice.py: the text and duration rules live in
// omnivoice_prompt.cpp, the model in omnivoice_lm.cpp, the codec in
// higgs_codec.cpp.

#include "brosoundml/omnivoice.h"

#include "brosoundml/detail/json.h"
#include "higgs_codec_common.h"
#include "omnivoice_lm.h"
#include "omnivoice_prompt.h"

#include <brolm/qwen_tokenizer.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;
namespace j = brosoundml::detail::json;
namespace fs = std::filesystem;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: OmniVoice: " + msg);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot read '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// One text chunk's LM run.
struct ChunkRun {
    std::vector<int32_t> text_ids;
    OmniVoiceLmResult result;
    int T = 0;
};

}  // namespace

// ─── OmniVoicePrompt serialisation ("OVCP") ─────────────────────────────────
//
//   char[4] "OVCP"; u32 version = 1; i32 num_codebooks; i32 num_frames;
//   f32 rms; i32 text_bytes; u8 text[text_bytes] (UTF-8);
//   i32 codes[num_codebooks * num_frames]   ([q * num_frames + t])
// Little-endian throughout.

void OmniVoicePrompt::save(const std::string& path) const {
    if (num_frames < 0 || (num_frames > 0 && codes.size() % static_cast<std::size_t>(num_frames) != 0))
        fail("prompt codes do not divide into num_frames");
    const int32_t C = num_frames > 0 ? static_cast<int32_t>(codes.size() / static_cast<std::size_t>(num_frames)) : 0;
    std::ofstream f(path, std::ios::binary);
    if (!f) fail("cannot write '" + path + "'");
    auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    f.write("OVCP", 4);
    w32(1u);
    w32(static_cast<uint32_t>(C));
    w32(static_cast<uint32_t>(num_frames));
    f.write(reinterpret_cast<const char*>(&rms), 4);
    w32(static_cast<uint32_t>(text.size()));
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!codes.empty())
        f.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes.size() * 4));
    if (!f) fail("write failed for '" + path + "'");
}

OmniVoicePrompt OmniVoicePrompt::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot read '" + path + "'");
    char magic[4];
    f.read(magic, 4);
    if (!f || std::memcmp(magic, "OVCP", 4) != 0) fail("'" + path + "' is not an OmniVoice prompt (OVCP)");
    auto r32 = [&]() { uint32_t v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; };
    const uint32_t version = r32();
    if (version != 1) fail("unsupported OVCP version " + std::to_string(version));
    const int32_t C = static_cast<int32_t>(r32());
    const int32_t T = static_cast<int32_t>(r32());
    OmniVoicePrompt p;
    f.read(reinterpret_cast<char*>(&p.rms), 4);
    const int32_t nb = static_cast<int32_t>(r32());
    if (!f || C < 0 || T < 0 || nb < 0 || C > 64) fail("malformed OVCP header in '" + path + "'");
    p.text.resize(static_cast<std::size_t>(nb));
    if (nb) f.read(p.text.data(), nb);
    p.num_frames = T;
    p.codes.resize(static_cast<std::size_t>(C) * static_cast<std::size_t>(T));
    if (!p.codes.empty()) f.read(reinterpret_cast<char*>(p.codes.data()), static_cast<std::streamsize>(p.codes.size() * 4));
    if (!f) fail("truncated OVCP file '" + path + "'");
    return p;
}

// ─── Impl ───────────────────────────────────────────────────────────────────

struct OmniVoice::Impl {
    OmniVoiceConfig cfg;
    bt::Device dev = bt::Device::CPU;
    OmniVoicePrecision precision = OmniVoicePrecision::FP32;
    bool loaded = false;
    std::unique_ptr<brolm::qwen::Tokenizer> tok;
    OmniVoiceLm lm;
    HiggsCodec codec;

    int C() const { return cfg.lm.num_codebooks; }

    void require_loaded() const {
        if (!loaded) fail("no model loaded");
    }

    // _resolve_language + _resolve_instruct (use_zh from the target text).
    void resolve(const std::string& text, const OmniVoiceParams& p, std::string& lang,
                 std::string& instruct) const {
        lang = ovp::resolve_language(p.language);
        instruct = p.instruct.empty() ? std::string()
                                      : ovp::resolve_instruct(p.instruct, ovp::text_has_cjk(text));
    }

    // _preprocess_all's length rule: the frame count and the speed ratio that
    // scales chunk estimates.
    int estimate(const std::string& text, const OmniVoiceParams& p, const OmniVoicePrompt* prompt,
                 float* speed_ratio) const {
        const std::string* ref_text = prompt ? &prompt->text : nullptr;
        const int n_ref = prompt ? prompt->num_frames : -1;
        const bool has_dur = p.duration > 0.0f;
        const double item_speed = has_dur ? 1.0 : static_cast<double>(p.speed);
        const int est = ovp::estimate_target_tokens(text, ref_text, n_ref, item_speed);
        if (has_dur) {
            const int target = std::max(1, static_cast<int>(static_cast<double>(p.duration) * cfg.frame_rate));
            if (speed_ratio) *speed_ratio = static_cast<float>(static_cast<double>(est) / target);
            return target;
        }
        if (speed_ratio) *speed_ratio = p.speed;
        return est;
    }

    // _prepare_inference_inputs' text rows: style ids + wrapped text ids.
    std::vector<int32_t> prompt_ids(const std::string& text, const std::string& lang,
                                    const std::string& instruct, bool denoise,
                                    const std::string* ref_text, bool has_ref) const {
        const std::string style = ovp::style_text(lang, instruct, denoise, has_ref);
        std::vector<int32_t> ids = tok->encode(style);
        const std::string* rt = (ref_text && !ref_text->empty()) ? ref_text : nullptr;
        const std::string wrapped = "<|text_start|>" + ovp::combine_text(text, rt) + "<|text_end|>";
        const std::vector<int32_t> tids = ovp::tokenize_with_tags(wrapped, *tok);
        ids.insert(ids.end(), tids.begin(), tids.end());
        return ids;
    }

    OmniVoiceLmRun lm_run(const OmniVoiceParams& p, std::uint64_t chunk) const {
        OmniVoiceLmRun r;
        r.num_steps = p.num_steps;
        r.t_shift = p.t_shift;
        r.guidance_scale = p.guidance_scale;
        r.layer_penalty = p.layer_penalty;
        r.position_temperature = p.position_temperature;
        r.class_temperature = p.class_temperature;
        r.gumbel_noise = p.gumbel_noise;
        r.seed = p.seed;
        r.chunk = chunk;
        return r;
    }

    // One chunk through the LM. `ref_codes` / `ref_text` describe the
    // reference (n_ref == 0: none).
    ChunkRun run_chunk(const std::string& text, int T, const std::string& lang,
                       const std::string& instruct, const OmniVoiceParams& p,
                       const std::vector<int32_t>& ref_codes, int n_ref, const std::string& ref_text,
                       const OmniVoiceInit* init, std::uint64_t chunk, const CancelCheck& cancel,
                       const OmniVoiceStepFn& on_step) const {
        ChunkRun cr;
        cr.T = T;
        cr.text_ids = prompt_ids(text, lang, instruct, p.denoise, n_ref > 0 ? &ref_text : nullptr, n_ref > 0);
        cr.result = lm.generate(cr.text_ids, ref_codes, n_ref, T, init, lm_run(p, chunk), cancel, on_step, nullptr);
        return cr;
    }

    // Resample to the codec rate on the host (torchaudio's sinc kernel, the
    // codec's own resampler).
    std::vector<float> to_codec_rate(const AudioBuffer& a) const {
        if (a.sample_rate == cfg.sample_rate || a.sample_rate <= 0) return a.samples;
        bt::DeviceScope scope(bt::Device::CPU);
        const int n = static_cast<int>(a.samples.size());
        bt::Tensor x = bt::Tensor::from_host_on(bt::Device::CPU, a.samples.data(), 1, n);
        const hcodec::SincResampler rs = hcodec::make_sinc_resampler(a.sample_rate, cfg.sample_rate, bt::Device::CPU);
        int n_out = 0;
        bt::Tensor y = hcodec::sinc_resample(rs, x, n, &n_out);
        return std::vector<float>(y.host_f32(), y.host_f32() + n_out);
    }
};

// ─── class ──────────────────────────────────────────────────────────────────

OmniVoice::OmniVoice() : impl_(new Impl()) {}
OmniVoice::~OmniVoice() = default;
OmniVoice::OmniVoice(OmniVoice&&) noexcept = default;
OmniVoice& OmniVoice::operator=(OmniVoice&&) noexcept = default;

void OmniVoice::load(const std::string& model_dir, bt::Device device, OmniVoicePrecision precision,
                     bool codec_decoder_only) {
    Impl& m = *impl_;
    m.loaded = false;
    const fs::path d(model_dir);
    const std::string config_path = (d / "config.json").string();
    const std::string tok_path = (d / "tokenizer.json").string();
    const std::string weights_path = (d / "model.safetensors").string();
    const std::string codec_dir = (d / "audio_tokenizer").string();
    if (!fs::exists(config_path)) fail("no config.json under '" + model_dir + "'");
    if (!fs::exists(tok_path)) fail("no tokenizer.json under '" + model_dir + "'");
    if (!fs::exists(weights_path)) fail("no model.safetensors under '" + model_dir + "'");
    if (!fs::exists(codec_dir)) fail("no audio_tokenizer/ under '" + model_dir + "'");
    bt::init();
    if (!bt::is_available(device)) fail("requested device is not available");
    if (precision == OmniVoicePrecision::BF16 && device == bt::Device::CPU)
        fail("BF16 weights need a GPU device");

    // ── config.json ──
    OmniVoiceConfig cfg;
    {
        const j::Value root = j::parse(slurp(config_path));
        if (!root.is_object()) fail("config.json is not a JSON object");
        const j::Value* llm = root.find("llm_config");
        if (!llm || !llm->is_object()) fail("config.json has no llm_config");
        OmniVoiceLmConfig& l = cfg.lm;
        l.hidden_size = llm->get_int("hidden_size", l.hidden_size);
        l.intermediate_size = llm->get_int("intermediate_size", l.intermediate_size);
        l.num_hidden_layers = llm->get_int("num_hidden_layers", l.num_hidden_layers);
        l.num_attention_heads = llm->get_int("num_attention_heads", l.num_attention_heads);
        l.num_key_value_heads = llm->get_int("num_key_value_heads", l.num_key_value_heads);
        l.head_dim = llm->get_int("head_dim", l.hidden_size / std::max(1, l.num_attention_heads));
        l.rms_norm_eps = llm->get_float("rms_norm_eps", l.rms_norm_eps);
        l.vocab_size = llm->get_int("vocab_size", l.vocab_size);
        if (const j::Value* rp = llm->find("rope_parameters"); rp && rp->is_object())
            l.rope_theta = rp->get_float("rope_theta", l.rope_theta);
        else
            l.rope_theta = llm->get_float("rope_theta", l.rope_theta);
        if (llm->get_string("model_type", "qwen3") != "qwen3") fail("config.json: llm_config.model_type is not qwen3");
        l.num_codebooks = root.get_int("num_audio_codebook", l.num_codebooks);
        l.audio_vocab_size = root.get_int("audio_vocab_size", l.audio_vocab_size);
        l.audio_mask_id = root.get_int("audio_mask_id", l.audio_mask_id);
        l.codebook_weights = root.get_int_array("audio_codebook_weights", {});
        if (l.num_codebooks < 1 || l.audio_vocab_size < 2 || l.audio_mask_id < 0 || l.audio_mask_id >= l.audio_vocab_size)
            fail("config.json: bad codebook geometry");
    }

    // ── tokenizer ──
    static const char* kSpecials[] = {"<|denoise|>", "<|lang_start|>", "<|lang_end|>", "<|instruct_start|>",
                                      "<|instruct_end|>", "<|text_start|>", "<|text_end|>"};
    std::unique_ptr<brolm::qwen::Tokenizer> tok(new brolm::qwen::Tokenizer(
        brolm::qwen::Tokenizer::from_tokenizer_json(tok_path, std::vector<std::string>(std::begin(kSpecials), std::end(kSpecials)))));
    auto special_id = [&](const char* s) -> int {
        const std::vector<int32_t> ids = tok->encode(s);
        return ids.size() == 1 ? ids[0] : -1;
    };
    cfg.denoise_id = special_id("<|denoise|>");
    cfg.lang_start_id = special_id("<|lang_start|>");
    cfg.lang_end_id = special_id("<|lang_end|>");
    cfg.instruct_start_id = special_id("<|instruct_start|>");
    cfg.instruct_end_id = special_id("<|instruct_end|>");
    cfg.text_start_id = special_id("<|text_start|>");
    cfg.text_end_id = special_id("<|text_end|>");
    for (const char* s : kSpecials)
        if (special_id(s) < 0) fail(std::string("tokenizer.json lacks the special token ") + s);
    {
        std::string eos = "<|im_end|>", pad = "<|endoftext|>";
        const std::string tc = (d / "tokenizer_config.json").string();
        if (fs::exists(tc)) {
            const j::Value root = j::parse(slurp(tc));
            if (root.is_object()) {
                eos = root.get_string("eos_token", eos);
                pad = root.get_string("pad_token", pad);
            }
        }
        cfg.eos_id = special_id(eos.c_str());
        cfg.pad_id = special_id(pad.c_str());
    }

    // ── weights ──
    {
        sf::File f = sf::File::open(weights_path);
        m.lm.load(f, cfg.lm, device, precision == OmniVoicePrecision::BF16);
    }
    m.codec.load(codec_dir, device, codec_decoder_only);
    cfg.codec = m.codec.config();
    cfg.sample_rate = cfg.codec.sample_rate;
    cfg.frame_rate = cfg.codec.frame_rate;

    m.cfg = cfg;
    m.dev = device;
    m.precision = precision;
    m.tok = std::move(tok);
    m.loaded = true;
}

AudioBuffer OmniVoice::synthesize(const std::string& text, const OmniVoiceParams& params,
                                  const OmniVoicePrompt* prompt, const CancelCheck& cancel,
                                  OmniVoiceTrace* trace, const OmniVoiceStepFn& on_step) const {
    const Impl& m = *impl_;
    m.require_loaded();
    if (ovp::strip(text).empty()) fail("empty text");
    if (params.num_steps < 1) fail("num_steps must be >= 1");
    if (prompt && (prompt->num_frames < 0 ||
                   prompt->codes.size() != static_cast<std::size_t>(m.C()) * static_cast<std::size_t>(prompt->num_frames)))
        fail("prompt codes must hold num_codebooks * num_frames entries");
    const int C = m.C();
    const int sr = m.cfg.sample_rate;
    std::string lang, instruct;
    m.resolve(text, params, lang, instruct);

    float ratio = 1.0f;
    const int T = m.estimate(text, params, prompt, &ratio);
    const int threshold = static_cast<int>(static_cast<double>(params.audio_chunk_threshold) * m.cfg.frame_rate);

    std::vector<int32_t> no_codes;
    const std::vector<int32_t>& ref_codes = prompt ? prompt->codes : no_codes;
    const int n_ref = prompt ? prompt->num_frames : 0;
    const std::string ref_text = prompt ? prompt->text : std::string();

    std::vector<ChunkRun> runs;
    double lm_seconds = 0;
    auto finish_run = [&](ChunkRun&& cr) {
        lm_seconds += cr.result.seconds;
        const bool cancelled = cr.result.cancelled;
        runs.push_back(std::move(cr));
        return !cancelled;
    };

    if (T <= threshold || params.audio_chunk_duration <= 0.0f) {
        if (!finish_run(m.run_chunk(text, T, lang, instruct, params, ref_codes, n_ref, ref_text,
                                    nullptr, 0, cancel, on_step)))
            return AudioBuffer({}, sr);
    } else {
        // _generate_chunked: split at punctuation into ~audio_chunk_duration
        // pieces, each estimated with the speed ratio.
        const std::size_t n_chars = ovp::to_codepoints(text).size();
        const double avg = static_cast<double>(T) / static_cast<double>(std::max<std::size_t>(1, n_chars));
        const int chunk_len = static_cast<int>(static_cast<double>(params.audio_chunk_duration) * m.cfg.frame_rate / avg);
        const std::vector<std::string> chunks = ovp::chunk_text(text, chunk_len, 3);
        if (chunks.empty()) fail("text chunking produced nothing");
        if (n_ref > 0) {
            for (std::size_t ci = 0; ci < chunks.size(); ++ci) {
                const int Tc = ovp::estimate_target_tokens(chunks[ci], &ref_text, n_ref, ratio);
                if (!finish_run(m.run_chunk(chunks[ci], Tc, lang, instruct, params, ref_codes, n_ref, ref_text,
                                            nullptr, static_cast<std::uint64_t>(ci), cancel, on_step)))
                    return AudioBuffer({}, sr);
            }
        } else {
            // Chunk 0 alone, then it becomes the reference for the rest.
            const int T0 = ovp::estimate_target_tokens(chunks[0], nullptr, -1, ratio);
            if (!finish_run(m.run_chunk(chunks[0], T0, lang, instruct, params, no_codes, 0, std::string(),
                                        nullptr, 0, cancel, on_step)))
                return AudioBuffer({}, sr);
            const std::vector<int32_t> codes0 = runs[0].result.codes;
            const std::string& text0 = chunks[0];
            for (std::size_t ci = 1; ci < chunks.size(); ++ci) {
                const int Tc = ovp::estimate_target_tokens(chunks[ci], &text0, T0, ratio);
                if (!finish_run(m.run_chunk(chunks[ci], Tc, lang, instruct, params, codes0, T0, text0,
                                            nullptr, static_cast<std::uint64_t>(ci), cancel, on_step)))
                    return AudioBuffer({}, sr);
            }
        }
    }

    // Decode every chunk, cross-fade, post-process.
    const auto c0 = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> waves;
    for (const ChunkRun& cr : runs)
        waves.push_back(m.codec.decode(cr.result.codes, C, cr.T).samples);
    const double codec_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
    std::vector<float> audio = ovp::cross_fade_chunks(waves, sr);

    if (params.postprocess_output) audio = ovp::remove_silence(audio, sr, 500, 100, 100);
    if (prompt) {
        if (prompt->rms < 0.1f) ovp::gain_rms_match(audio, prompt->rms);
    } else {
        ovp::peak_normalize(audio);
    }
    audio = ovp::fade_and_pad(audio, params.pad_duration, params.fade_duration, sr);

    if (trace) {
        int total = 0;
        trace->chunk_frames.clear();
        for (const ChunkRun& cr : runs) { total += cr.T; trace->chunk_frames.push_back(cr.T); }
        trace->text_ids = runs[0].text_ids;
        trace->num_frames = total;
        trace->codes.assign(static_cast<std::size_t>(C) * total, 0);
        trace->unmask_step.assign(static_cast<std::size_t>(C) * total, -1);
        int off = 0;
        for (const ChunkRun& cr : runs) {
            for (int q = 0; q < C; ++q)
                for (int t = 0; t < cr.T; ++t) {
                    trace->codes[static_cast<std::size_t>(q) * total + off + t] = cr.result.codes[static_cast<std::size_t>(q) * cr.T + t];
                    trace->unmask_step[static_cast<std::size_t>(q) * total + off + t] = cr.result.unmask_step[static_cast<std::size_t>(q) * cr.T + t];
                }
            off += cr.T;
        }
        trace->lm_seconds = lm_seconds;
        trace->codec_seconds = codec_seconds;
    }
    return AudioBuffer(std::move(audio), sr);
}

std::vector<int32_t> OmniVoice::generate_codes(const std::string& text, int num_frames,
                                               const OmniVoiceParams& params, const OmniVoicePrompt* prompt,
                                               const OmniVoiceInit* init, const CancelCheck& cancel,
                                               OmniVoiceTrace* trace, const OmniVoiceStepFn& on_step) const {
    const Impl& m = *impl_;
    m.require_loaded();
    if (ovp::strip(text).empty()) fail("empty text");
    if (params.num_steps < 1) fail("num_steps must be >= 1");
    if (prompt && (prompt->num_frames < 0 ||
                   prompt->codes.size() != static_cast<std::size_t>(m.C()) * static_cast<std::size_t>(prompt->num_frames)))
        fail("prompt codes must hold num_codebooks * num_frames entries");
    std::string lang, instruct;
    m.resolve(text, params, lang, instruct);
    const int T = num_frames > 0 ? num_frames : m.estimate(text, params, prompt, nullptr);
    if (init && (init->tokens.size() != static_cast<std::size_t>(m.C()) * static_cast<std::size_t>(T) ||
                 init->keep.size() != init->tokens.size()))
        fail("init grid must hold num_codebooks * num_frames tokens and keep flags");

    std::vector<int32_t> no_codes;
    ChunkRun cr = m.run_chunk(text, T, lang, instruct, params, prompt ? prompt->codes : no_codes,
                              prompt ? prompt->num_frames : 0, prompt ? prompt->text : std::string(),
                              init, 0, cancel, on_step);
    if (trace) {
        trace->text_ids = cr.text_ids;
        trace->num_frames = T;
        trace->codes = cr.result.codes;
        trace->unmask_step = cr.result.unmask_step;
        trace->chunk_frames = {T};
        trace->lm_seconds = cr.result.seconds;
        trace->codec_seconds = 0;
    }
    if (cr.result.cancelled) return {};
    return cr.result.codes;
}

int OmniVoice::estimate_frames(const std::string& text, const OmniVoiceParams& params,
                               const OmniVoicePrompt* prompt) const {
    const Impl& m = *impl_;
    m.require_loaded();
    std::string lang, instruct;
    m.resolve(text, params, lang, instruct);
    return m.estimate(text, params, prompt, nullptr);
}

OmniVoicePrompt OmniVoice::create_prompt(const AudioBuffer& ref, const std::string& ref_text,
                                         bool preprocess) const {
    const Impl& m = *impl_;
    m.require_loaded();
    if (!m.codec.has_encoder()) fail("the codec encoder was not loaded (codec_decoder_only)");
    if (ref.samples.empty()) fail("empty reference audio");
    const int sr = m.cfg.sample_rate;
    std::vector<float> wav = m.to_codec_rate(ref);

    // create_voice_clone_prompt: RMS (kept for output matching), boost of a
    // quiet clip, optional trimming + silence removal, frame alignment.
    const float ref_rms = ovp::rms(wav);
    if (ref_rms > 0.0f && ref_rms < 0.1f)
        for (float& v : wav) { const float a = v * 0.1f; v = a / ref_rms; }
    if (preprocess) {
        if (ref_text.empty()) wav = ovp::trim_long_audio(wav, sr, 15.0, 3.0, 20.0);
        wav = ovp::remove_silence(wav, sr, 200, 100, 200);
        if (wav.empty()) fail("reference audio is empty after silence removal; try preprocess = false");
    }
    const int hop = m.cfg.codec.hop_length;
    const std::size_t clip = wav.size() % static_cast<std::size_t>(hop);
    if (clip) wav.resize(wav.size() - clip);
    if (wav.empty()) fail("reference audio is shorter than one codec frame");

    OmniVoicePrompt p;
    p.codes = m.codec.encode(AudioBuffer(wav, sr), &p.num_frames);
    p.text = preprocess ? ovp::add_punctuation(ref_text) : ref_text;
    p.rms = ref_rms;
    return p;
}

AudioBuffer OmniVoice::decode_codes(const std::vector<int32_t>& codes, int num_frames) const {
    const Impl& m = *impl_;
    m.require_loaded();
    return m.codec.decode(codes, m.C(), num_frames);
}

std::vector<int32_t> OmniVoice::encode_audio(const AudioBuffer& audio, int* num_frames_out) const {
    const Impl& m = *impl_;
    m.require_loaded();
    return m.codec.encode(audio, num_frames_out);
}

std::vector<int32_t> OmniVoice::tokenize(const std::string& text) const {
    const Impl& m = *impl_;
    m.require_loaded();
    return ovp::tokenize_with_tags(text, *m.tok);
}

std::vector<std::string> OmniVoice::languages() const { return ovp::language_names(); }

std::vector<OmniVoice::InstructCategory> OmniVoice::instruct_attributes() const {
    return ovp::instruct_categories();
}

std::vector<std::string> OmniVoice::nonverbal_tags() const { return ovp::nonverbal_tags(); }

const OmniVoiceConfig& OmniVoice::config() const { return impl_->cfg; }
bool OmniVoice::loaded() const { return impl_->loaded; }
bt::Device OmniVoice::device() const { return impl_->dev; }

}  // namespace brosoundml
