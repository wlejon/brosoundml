#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace brosoundml {

// ─── Qwen3-TTS (12 Hz multi-codebook) ───────────────────────────────────────
//
// Qwen3-TTS is Alibaba's open-weight TTS series (Jan 2026). brosoundml targets
// the 12 Hz multi-codebook track — an end-to-end discrete-token model, no
// diffusion, no external vocoder. Pipeline — the order brosoundml drives it:
//
//   1. Text         text -> Qwen BPE token ids (brolm::qwen tokenizer; the
//                   vocab.json + merges.txt ship in the model dir).
//   2. Talker       a Qwen3 decoder backbone over a dual stream: text-token
//                   embeddings (text_vocab @ text_hidden, projected to the
//                   Talker hidden by text_projection) interleaved with codec
//                   token embeddings. M-RoPE positions. Per frame it emits a
//                   hidden state and, via codec_head, acoustic codebook 0.
//   3. Code         a small 5-layer depth transformer that, conditioned on the
//      Predictor    Talker hidden, autoregressively emits acoustic codebooks
//                   1..15 (one codec_embedding + one lm_head per codebook).
//   4. Codec        the bundled 12 Hz codec (speech_tokenizer/): 16 RVQ codes
//      decoder      per frame -> dequantize -> pre_conv -> an 8-layer windowed
//                   transformer -> ConvNeXt upsample (x2 x2) -> a SEANet causal
//                   conv decoder with Snake activations (x8 x5 x4 x3) -> 24 kHz.
//                   Total upsample 1920; 12.5 Hz * 1920 = 24 kHz.
//
// brotensor op coverage: rms_norm, rope, silu/SwiGLU, GQA self-attention,
// conv1d / conv_transpose1d, snake, embedding_lookup, softmax, sample_logits /
// argmax — all already on the op surface. No new kernels (Qwen-TTS adds no op
// brotensor lacks). See docs/qwen-tts-weights.md for the full tensor map.
//
// STATUS: complete (CPU FP32). Stage 1 (weight loading) — load() parses
// config.json + speech_tokenizer/config.json and validates both safetensors.
// Stage 2 (codec decoder) — decode_codes() runs the bundled 12 Hz codec tail
// (RVQ codes -> 24 kHz waveform), matching upstream to FP32 round-off. Stage 3
// (Talker forward) — the 28-layer Qwen3 decoder backbone (dual text/codec
// embedding, GQA + QK-norm + interleaved M-RoPE, codec_head; src/qwen_tts_talker.*).
// Stage 4 (Code Predictor + AR loop) — the 5-layer depth transformer
// (src/qwen_tts_code_predictor.*) and the dual-track generation loop
// (src/qwen_tts_generate.*, Talker KV cache + Code Predictor -> per-frame RVQ
// codes). Stage 5 (synthesize) — the Qwen BPE tokenizer + CustomVoice chat
// prefill assembly + the AR loop (with the upstream codebook-0 logits policy:
// suppress the top-1024 codec tokens, min_new_tokens, repetition_penalty,
// greedy) -> decode_codes. Tokenizer ids, prefill, and the full code stream are
// verified against the genuine upstream model.
//
// Runs device-neutrally on CPU and CUDA: load(device) places the weights on the
// chosen backend and the whole pipeline (Talker, Code Predictor, AR loop, codec)
// dispatches through brotensor device ops. Compute is FP32 on both backends —
// attention uses flash_attention_varlen_forward's FP32 CUDA kernel — so CUDA
// reproduces the CPU/upstream discrete-code stream bit-for-bit (the codec tail
// then matches to ~1e-5). Sampling and the Base-variant zero-shot voice clone
// (speaker / codec encoder) are later targets.

// Qwen3-style transformer block hyperparameters. Shared by the Talker, the
// Code Predictor, and (with layer-scale / windowed-attention variations) the
// codec's pre_transformer.
struct QwenTtsTransformerConfig {
    int   hidden_size         = 0;
    int   intermediate_size   = 0;
    int   num_hidden_layers   = 0;
    int   num_attention_heads = 0;   // query heads
    int   num_key_value_heads = 0;   // KV heads (GQA)
    int   head_dim            = 0;   // independent of hidden_size
    float rms_norm_eps        = 1e-6f;
    float rope_theta          = 1000000.0f;
};

// The Code Predictor depth transformer — emits acoustic codebooks 1..15 given
// the Talker's per-frame hidden state (codebook 0 comes from the Talker's
// codec_head). Holds (num_code_groups - 1) codec_embedding tables and lm_heads.
struct QwenTtsCodePredictorConfig {
    QwenTtsTransformerConfig transformer;
    int vocab_size      = 0;   // per-codebook code vocab (2048)
    int num_code_groups = 0;   // total codebooks per frame incl. codebook 0 (16)
};

// The Talker — a Qwen3 decoder backbone over a dual text/codec token stream.
struct QwenTtsTalkerConfig {
    QwenTtsTransformerConfig transformer;

    int num_code_groups  = 0;   // total RVQ codebooks per frame (16)
    int vocab_size       = 0;   // Talker codec vocab incl. specials (3072)
    int text_vocab_size  = 0;   // text-token vocab (151936, Qwen BPE)
    int text_hidden_size = 0;   // text-embedding width before text_projection (2048)

    // M-RoPE: positions advance position_id_per_seconds per audio second; the
    // head_dim/2 rotary dims split into the sections in mrope_section.
    int              position_id_per_seconds = 0;   // (13)
    std::vector<int> mrope_section;                 // ([24, 20, 20])
    bool             mrope_interleaved = true;

    // Codec special-token ids, in the Talker's codec vocab space.
    int codec_bos_id       = 0, codec_eos_id     = 0, codec_pad_id      = 0;
    int codec_think_id     = 0, codec_nothink_id = 0;
    int codec_think_bos_id = 0, codec_think_eos_id = 0;

    // language name -> codec language id (prepended to the codec stream).
    std::unordered_map<std::string, int> codec_language_id;

    // CustomVoice: preset speaker name -> codec speaker token id, plus an
    // optional dialect tag per speaker ("" when the speaker is not a dialect).
    std::unordered_map<std::string, int>         spk_id;
    std::unordered_map<std::string, std::string> spk_dialect;

    QwenTtsCodePredictorConfig code_predictor;
};

// The codec ENCODER (speech_tokenizer/ `encoder.*`) — 24 kHz waveform -> RVQ
// codes, the analysis path and inverse of the decoder. An HF-Mimi stack: a
// SEANet down-sampler (ELU + causal convs), a causal sliding-window transformer
// (LayerNorm + GELU), a /N downsample conv, and a split-RVQ that emits
// `valid_num_quantizers` euclidean codes per 12.5 Hz frame.
struct QwenTtsCodecEncoderConfig {
    int num_filters          = 0;   // SEANet base width (64)
    int kernel_size          = 0;   // SEANet conv kernel (7)
    int last_kernel_size     = 0;   // final conv kernel (3)
    int residual_kernel_size = 0;   // residual-unit dilated conv kernel (3)
    int compress             = 0;   // residual-unit channel divisor (2)
    int codebook_dim         = 0;   // VQ space dim the input_proj maps into (256)
    std::vector<int> ratios;        // SEANet strides in decode order ([8, 6, 5, 4]);
                                    // applied reversed, so the encoder downsamples 4·5·6·8

    QwenTtsTransformerConfig transformer;  // encoder_transformer (LayerNorm + GELU MLP)
    int sliding_window = 0;         // (250)
    int valid_num_quantizers = 0;   // codes kept per frame (16: 1 semantic + 15 acoustic)
};

// The Base-variant speaker encoder (`speaker_encoder.*`) — an ECAPA-TDNN
// x-vector extractor. A reference clip's log-mel (128-band, 24 kHz) -> a single
// `enc_dim` speaker embedding, which the zero-shot voice clone splices into the
// Talker codec prefill where a CustomVoice preset speaker token would sit. Only
// the Base checkpoint ships one (`present == false` otherwise). The channel /
// kernel / dilation lists and the mel-frontend params are upstream defaults
// (config.json carries only enc_dim + sample_rate).
struct QwenTtsSpeakerEncoderConfig {
    bool present     = false;   // true only for the Base variant
    int  mel_dim     = 128;     // log-mel bands (encoder input channels)
    int  enc_dim     = 1024;    // output x-vector width
    int  sample_rate = 24000;   // required input rate

    std::vector<int> enc_channels;       // [512, 512, 512, 512, 1536]
    std::vector<int> enc_kernel_sizes;   // [5, 3, 3, 3, 1]
    std::vector<int> enc_dilations;      // [1, 2, 3, 4, 1]
    int res2net_scale      = 8;
    int se_channels        = 128;
    int attention_channels = 128;

    // Mel frontend (librosa slaney basis + log compression): STFT n_fft / hop /
    // win, and the mel band edges.
    int n_fft = 1024, hop_size = 256, win_size = 1024, fmin = 0, fmax = 12000;
};

// The bundled 12 Hz codec decoder (speech_tokenizer/) — 16 RVQ codes -> 24 kHz.
struct QwenTtsCodecConfig {
    // RVQ quantizers (codebooks stored as EMA embedding_sum / cluster_usage).
    int latent_dim              = 0;
    int codebook_dim            = 0;
    int codebook_size           = 0;
    int decoder_dim             = 0;
    int num_quantizers          = 0;   // (16)
    int num_semantic_quantizers = 0;   // (1)
    int semantic_codebook_size  = 0;   // (4096)

    // pre_transformer (windowed attention + layer scale; no QK-norm).
    QwenTtsTransformerConfig pre_transformer;
    int sliding_window = 0;            // (72)

    // Upsampling. The two stages multiply to decode_upsample_rate:
    //   upsampling_ratios — ConvNeXt upsample blocks ([2, 2])
    //   upsample_rates    — SEANet transposed-conv decoder rates ([8, 5, 4, 3])
    std::vector<int> upsampling_ratios;
    std::vector<int> upsample_rates;
    int decode_upsample_rate = 0;      // prod(both) = 1920
    int encode_downsample_rate = 0;    // SEANet ratios · downsample stride = 1920
    int output_sample_rate   = 24000;

    QwenTtsCodecEncoderConfig encoder;  // the `encoder.*` analysis stack
};

enum class QwenTtsVariant { Base, CustomVoice, VoiceDesign };

// Optional sampling controls for synthesize(). The default (temperature == 0)
// is the greedy, deterministic policy that reproduces the upstream code stream
// bit-for-bit. temperature > 0 turns the autoregressive loop stochastic: every
// code — codebook 0 and the Code Predictor's codebooks 1..15 — is drawn through
// brotensor::sample_logits (temperature -> softmax -> top_k -> top_p ->
// inverse-CDF), seeded by `seed` so a fixed seed gives a repeatable utterance
// and different seeds give natural take-to-take variation.
struct QwenTtsSampling {
    float         temperature = 0.0f;  // 0 = greedy (deterministic, default)
    int           top_k       = 0;     // 0 = no top-k cap
    float         top_p       = 1.0f;  // 1 = no nucleus cap
    std::uint64_t seed        = 0;     // RNG seed for reproducible draws
};

// Top-level Qwen3-TTS config, read from config.json by QwenTts::load.
struct QwenTtsConfig {
    QwenTtsVariant variant     = QwenTtsVariant::Base;
    std::string    model_size;          // "0b6" / "1b7"
    int            sample_rate = 24000;  // fixed 24 kHz output

    // Text-side framing tokens (Qwen chat + tts), in the text vocab.
    int tts_bos_id  = 0, tts_eos_id = 0, tts_pad_id   = 0;
    int im_start_id = 0, im_end_id  = 0, assistant_id = 0;

    QwenTtsTalkerConfig       talker;
    QwenTtsCodecConfig        codec;
    QwenTtsSpeakerEncoderConfig speaker_encoder;  // Base only (present == true)
};

// The Qwen3-TTS pipeline. Construct, load() a model directory, then
// synthesize(). Heavy state (weights, config, module graph) lives behind a
// pImpl so the public header stays free of brotensor module internals.
class QwenTts {
public:
    QwenTts();
    ~QwenTts();
    QwenTts(QwenTts&&) noexcept;
    QwenTts& operator=(QwenTts&&) noexcept;
    QwenTts(const QwenTts&) = delete;
    QwenTts& operator=(const QwenTts&) = delete;

    // Load config.json + model.safetensors and the bundled
    // speech_tokenizer/{config.json,model.safetensors} from `model_dir`,
    // placing weights on `device`. Throws std::runtime_error on a missing /
    // malformed model.
    void load(const std::string& model_dir,
              brotensor::Device device = brotensor::Device::CPU);

    // Synthesize speech for `text` in `language`. The voice is chosen by variant:
    //   CustomVoice — pass a preset `speaker` name (see speakers()); `instruct`
    //                 is ignored on the 0.6B checkpoint, honoured on 1.7B.
    //   VoiceDesign — pass a natural-language `instruct` describing the voice
    //                 (e.g. "a warm, low-pitched elderly storyteller"); `speaker`
    //                 is unused (the model has no presets).
    // Returns mono 24 kHz PCM.
    //
    // `cancel` is polled once per generated frame in the autoregressive loop
    // (the dominant cost — Talker step + Code Predictor per 12.5 Hz frame); when
    // it returns true the loop aborts and synthesize() returns an empty buffer
    // (the partial code stream is discarded). Empty (the default) = no cancel.
    AudioBuffer synthesize(const std::string& text,
                           const std::string& speaker,
                           const std::string& language = "english",
                           const std::string& instruct = "",
                           const CancelCheck& cancel = {},
                           const QwenTtsSampling& sampling = {}) const;

    // Zero-shot voice clone (Base variant only): synthesize `text` in the voice
    // of a reference clip `ref`. The clip is encoded to a speaker x-vector by the
    // ECAPA-TDNN speaker encoder and spliced into the Talker prefill where a
    // CustomVoice preset speaker token would sit — "x-vector-only" enrollment, no
    // reference transcript. `ref` is downmixed/resampled to 24 kHz internally.
    // `cancel` is polled once per generated frame (see synthesize()). Throws if no
    // model is loaded, the loaded checkpoint is not a Base variant (no speaker
    // encoder), or `ref` is empty. Returns mono 24 kHz PCM.
    AudioBuffer synthesize_clone(const std::string& text,
                                 const AudioBuffer& ref,
                                 const std::string& language = "english",
                                 const CancelCheck& cancel = {},
                                 const QwenTtsSampling& sampling = {}) const;

    // Encode a reference clip into the ECAPA-TDNN speaker x-vector — exactly the
    // enrollment step synthesize_clone runs, exposed on its own. `ref` is
    // resampled to the encoder's rate (24 kHz) as needed and treated as mono.
    // Returns `enc_dim` (1024) host floats — a speaker-identity embedding. The
    // honest audio->identity front-end for training an adapter that maps a voice
    // into another model's style space (e.g. Kokoro's). Base variant only;
    // throws if no model is loaded, the checkpoint has no speaker encoder, or
    // `ref` is empty.
    std::vector<float> embed_speaker(const AudioBuffer& ref) const;

    // Decode a precomputed code stream straight through the bundled 12 Hz codec
    // decoder to a 24 kHz waveform — the deterministic tail of synthesis, usable
    // on its own once the Talker / Code Predictor have produced (or a caller
    // holds) the per-frame RVQ codes. `codes` holds `num_quantizers * num_frames`
    // entries laid out codes[k * num_frames + t] (codebook-major). Throws if no
    // model is loaded or the count disagrees. Runs on the load device (FP32 on
    // CPU and CUDA alike).
    AudioBuffer decode_codes(const std::vector<int32_t>& codes,
                             int num_quantizers, int num_frames) const;

    // Encode a reference waveform into the bundled 12 Hz codec's RVQ codes — the
    // analysis path (the inverse of decode_codes), and the acoustic conditioning
    // a zero-shot voice clone consumes. `ref` is downmixed/resampled to 24 kHz
    // mono and right-padded to a whole frame as needed. Returns
    // `num_quantizers * num_frames` codes laid out codes[k*num_frames + t]
    // (codebook-major — the same layout decode_codes accepts, so encode ▸ decode
    // round-trips), with num_quantizers == config().codec.encoder.valid_num_quantizers
    // (16: one semantic + fifteen acoustic). *num_frames_out, when non-null,
    // receives num_frames. Throws if no model is loaded or `ref` is empty. Runs
    // on the load device (FP32 on CPU and CUDA alike).
    std::vector<int32_t> encode_audio(const AudioBuffer& ref,
                                      int* num_frames_out = nullptr) const;

    // Preset speaker names available in this checkpoint (CustomVoice only;
    // empty for Base / VoiceDesign).
    std::vector<std::string> speakers() const;

    // Selectable language names for synthesize() (the codec_language_id keys,
    // excluding the dialect tags, which are reached via their dialect speaker).
    // "auto" is always valid (no language tag) but is not included here.
    std::vector<std::string> languages() const;

    // The dialect tag for a preset speaker ("sichuan_dialect" / "beijing_dialect"),
    // or "" if the speaker is not a dialect voice / is unknown. Lets a UI badge
    // the dialect speakers without hard-coding the speaker list.
    std::string speaker_dialect(const std::string& speaker) const;

    const QwenTtsConfig& config() const;
    bool loaded() const;

private:
    // Shared synthesis tail: tokenize-resolved inputs -> prefill -> AR loop ->
    // codec decode. `spk_embed` (hidden floats, or null) is the Base clone's
    // x-vector spliced into the prefill speaker slot; `spk_id` the CustomVoice
    // preset token (or -1). Both callers (synthesize / synthesize_clone) resolve
    // language + speaker then delegate here.
    AudioBuffer synth_core(const std::vector<int32_t>& input_ids,
                           const std::vector<int32_t>& instruct_ids,
                           int spk_id, const float* spk_embed, int language_id,
                           const CancelCheck& cancel,
                           const QwenTtsSampling& sampling) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
