#pragma once

// CLAP — Contrastive Language-Audio Pretraining (LAION, `laion/larger_clap_general`).
//
// Embeds a sound clip and a text prompt into one 512-d joint space, so a clip
// can be scored against prompts without anyone listening to it ("a dog
// barking" vs "xylophone" vs "synthetic electronic beeps"). transformers'
// ClapModel, hand-written on brotensor:
//
//   front-end   any-rate mono -> 48 kHz (brosoundml::resample) -> a 10 s
//               window (shorter clips repeat- or silence-padded, longer ones
//               cropped or windowed, see ClapAudioOptions) -> log-mel: 1024-pt periodic
//               Hann STFT, hop 480, centred, power, 64 Slaney mel bins over
//               50..14000 Hz, 10*log10(max(x, 1e-10)) -> (1001, 64).
//   audio tower HTSAT (unfused): BatchNorm over mel bins, bicubic time
//               stretch to 1024 frames, fold into a 256x256 image, 4x4 patch
//               embed (128-d), four Swin stages (depths 2/2/12/2, heads
//               4/8/16/32, window 8, shifted windows, patch merging), final
//               LayerNorm, mean pool -> 1024-d.
//   text tower  RoBERTa-base (12 x 768, post-LN), byte-level BPE via the
//               checkpoint's tokenizer.json, pooler (tanh dense over <s>).
//   projections Linear -> ReLU -> Linear into 512-d per tower, L2-normalised.
//
// Similarity is the cosine of the two unit embeddings; the model's logit
// scale (exp(logit_scale_a) ~ 38.7) turns cosines into logits, and a softmax
// over the given prompts is what `ClapScore::probability` holds.
//
// Device-neutral: every tower op dispatches through brotensor on the load
// device (CPU / CUDA); the log-mel front-end runs on the host, as Whisper's
// does. FP32 throughout. docs/clap.md has the architecture and parity notes.

#include <brosoundml/audio.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

struct ClapConfig {
    // front-end (preprocessor_config.json)
    int    sample_rate  = 48000;
    int    max_samples  = 480000;   // 10 s window
    int    n_fft        = 1024;
    int    hop_length   = 480;
    int    num_mel_bins = 64;
    double f_min        = 50.0;
    double f_max        = 14000.0;
    // audio tower (config.json audio_config)
    std::vector<int> depths    = {2, 2, 12, 2};
    std::vector<int> heads     = {4, 8, 16, 32};
    int   patch_embed_dim      = 128;
    int   patch_size           = 4;
    int   window_size          = 8;
    int   spec_size            = 256;
    int   audio_hidden         = 1024;
    float audio_ln_eps         = 1e-5f;
    // text tower (config.json text_config)
    int   vocab_size           = 50265;
    int   text_hidden          = 768;
    int   text_layers          = 12;
    int   text_heads           = 12;
    int   text_intermediate    = 3072;
    int   max_position         = 514;
    int   pad_token_id         = 1;
    float text_ln_eps          = 1e-12f;
    // joint space
    int   projection_dim       = 512;

    int frames() const { return max_samples / hop_length + 1; }   // 1001
};

// How a clip shorter than the 10 s window is filled out to it.
enum class ClapPad {
    Auto,      // Silence below kClapShortClipSeconds, Repeat from there on.
    Repeat,    // transformers' "repeatpad", the reference and the checkpoint's
               // preprocessor_config: whole tiles of the clip, then zeros. A
               // 0.3 s one-shot becomes 33 hits in a row, which CLAP hears as
               // a loop or a rhythm rather than one event.
    Silence,   // the clip once at the start, then zeros (transformers' "pad").
};

// Auto's switch-over: a clip shorter than this is a one-shot (a hit, a
// gunshot, a blip) that tiling would turn into a pattern; a longer one
// already carries its own texture and keeps the reference's tiling.
constexpr double kClapShortClipSeconds = 2.0;

// How a clip longer than the 10 s window is reduced to model input. A clip of
// 10 s or less is padded per ClapAudioOptions::pad.
enum class ClapLongMode {
    Mean,   // embed every 10 s window (ceil(n / 10 s) of them, spread evenly
            // from the start to the end, overlapping as needed) and average
            // the unit embeddings, renormalised. Covers the whole clip.
    Crop,   // one 10 s crop, as transformers' "rand_trunc" — but at a chosen
            // offset rather than a random one.
};

struct ClapAudioOptions {
    ClapLongMode long_mode   = ClapLongMode::Mean;
    // Crop mode: offset in 48 kHz samples, clamped to the clip. -1 = centre.
    int          crop_offset = -1;
    // A clip shorter than 10 s: how it is filled out (see ClapPad).
    ClapPad      pad         = ClapPad::Auto;
};

struct ClapScore {
    std::vector<float> similarity;       // per prompt: cosine in [-1, 1]
    std::vector<float> logits;           // similarity * logit_scale
    std::vector<float> probability;      // softmax(logits) over the prompts
    std::vector<float> audio_embedding;  // 512-d, unit length
};

class Clap {
public:
    Clap();
    ~Clap();
    Clap(Clap&&) noexcept;
    Clap& operator=(Clap&&) noexcept;
    Clap(const Clap&) = delete;
    Clap& operator=(const Clap&) = delete;

    // Load config.json + model.safetensors + tokenizer.json from `dir` (the
    // layout scripts/download-clap.sh + convert-clap.py produce), weights on
    // `device`. Throws std::runtime_error on a missing file / tensor.
    void load(const std::string& dir, brotensor::Device device);

    bool loaded() const;
    const ClapConfig& config() const;
    brotensor::Device device() const;
    float logit_scale() const;       // exp(logit_scale_a): audio-side logit scale
    float text_logit_scale() const;  // exp(logit_scale_t)

    // ── front-end (host) ───────────────────────────────────────────────────
    // Mono samples at the model rate (48 kHz), resampled from any input rate.
    std::vector<float> to_model_rate(const AudioBuffer& audio) const;
    // 48 kHz samples -> the (frames x num_mel_bins) log-mel of one 10 s
    // window, frame-major. A longer input is cropped at `crop_offset`
    // (clamped; -1 = centre); a shorter one is padded per `pad` (the
    // default, Repeat, is the reference's front-end).
    std::vector<float> log_mel(const std::vector<float>& wave48, int crop_offset = -1,
                               ClapPad pad = ClapPad::Repeat) const;
    // The 10 s window of 48 kHz samples log_mel analyses: the crop of a
    // longer input, or a shorter one padded per `pad` (Auto resolved by
    // resolve_pad).
    std::vector<float> fill_window(const std::vector<float>& wave48, int crop_offset = -1,
                                   ClapPad pad = ClapPad::Repeat) const;
    // Auto -> Silence for `n` samples under kClapShortClipSeconds at the
    // model rate, else Repeat; Repeat / Silence pass through.
    ClapPad resolve_pad(ClapPad pad, int n) const;
    // Start offsets of the 10 s windows Mean mode embeds for `n` samples.
    std::vector<int> window_starts(int n) const;

    // ── towers ─────────────────────────────────────────────────────────────
    // (frames x num_mel_bins) log-mel -> 512-d unit audio embedding.
    // `pooled` (optional) receives the 1024-d pre-projection HTSAT output.
    std::vector<float> embed_mel(const std::vector<float>& mel,
                                 std::vector<float>* pooled = nullptr) const;
    // Any-rate clip -> 512-d unit audio embedding (the long-clip policy per
    // `opts`). Throws on an empty clip.
    std::vector<float> embed_audio(const AudioBuffer& audio,
                                   const ClapAudioOptions& opts = {}) const;

    // RoBERTa token ids with <s> ... </s>, truncated to the position table.
    std::vector<std::int32_t> tokenize(const std::string& text) const;
    // Token ids (as tokenize() returns) -> 512-d unit text embedding.
    std::vector<float> embed_ids(const std::vector<std::int32_t>& ids) const;
    std::vector<float> embed_text(const std::string& text) const;

    // ── scoring ────────────────────────────────────────────────────────────
    ClapScore score(const AudioBuffer& audio, const std::vector<std::string>& prompts,
                    const ClapAudioOptions& opts = {}) const;
    // Score a cached audio embedding against cached text embeddings
    // (`text` holds n_prompts * 512 floats, prompt-major).
    static ClapScore score_embeddings(const std::vector<float>& audio_embedding,
                                      const std::vector<float>& text, int n_prompts,
                                      float logit_scale);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
