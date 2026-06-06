#pragma once

#include "brosoundml/audio.h"
#include "brosoundml/decoder_lora.h"   // DecoderLoraContext

#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace brosoundml {

// Decoder back-half modules — defined in kokoro_modules.h, referenced here only
// as borrowed handles in KokoroDecodePrep (the DecoderLora trains against them).
struct DecoderBackbone;
struct Generator;

// ─── Kokoro-82M ────────────────────────────────────────────────────────────
//
// Kokoro is an 82M-parameter text-to-speech model derived from StyleTTS 2.
// Unlike StyleTTS 2 it does not sample a style with a diffusion model — the
// "voice" is a precomputed embedding (a voice pack), so synthesis is a single
// deterministic forward pass: phoneme token ids + a voice -> 24 kHz waveform.
//
// Pipeline — the order brosoundml drives the components:
//
//   1. G2P          text -> phonemes. EXTERNAL: Kokoro uses the misaki G2P
//                   frontend. brosoundml takes phoneme token ids as its input
//                   and does not bundle a G2P engine.
//   2. plBERT       a phoneme-level BERT (ALBERT-style, weight-shared layers)
//                   that encodes the phoneme sequence into context features.
//   3. Text encoder phoneme embedding -> 3x (conv1d + LayerNorm + LeakyReLU)
//                   -> bidirectional LSTM. Produces per-phoneme features.
//   4. Predictor    conditioned on the voice embedding: a duration predictor
//                   (LSTM + projection) emits a per-phoneme frame count; a
//                   length regulator expands the phoneme features to frame
//                   rate; F0 (pitch) and energy are predicted at frame rate.
//   5. Decoder      an iSTFTNet generator: AdaIN residual blocks + transposed-
//                   conv upsampling conditioned on (features, F0, energy,
//                   voice), a harmonic+noise source excitation, a final layer
//                   emitting an STFT magnitude/phase pair, and an iSTFT head
//                   that returns the waveform.
//
// brotensor op coverage: conv1d / conv_transpose1d / pad1d, leaky_relu, snake,
// stft / istft, group_norm (instance norm via num_groups == C) + modulate (the
// AdaIN affine), embedding_lookup, and self-attention all map directly onto the
// existing op surface. The one missing primitive is a recurrent (LSTM) op;
// brosoundml composes the LSTM cell from matmul + sigmoid + tanh per timestep
// for now — a fused brotensor lstm op is a later performance optimisation.
//
// STATUS: complete. load() reads config.json + the safetensors weights;
// synthesize() runs the full plBERT ▶ text-encoder ▶ duration/F0/energy
// predictors ▶ iSTFTNet decoder forward pass and returns 24 kHz mono PCM. The
// one approximation is the harmonic-source branch (see the Caveat in README.md).

// iSTFTNet decoder hyperparameters — the Kokoro vocoder branch. Drives the
// upsampling stack, the AdaIN residual blocks, and the iSTFT head.
struct IStftNetConfig {
    int                            upsample_initial_channel = 0;
    std::vector<int>               upsample_kernel_sizes;       // per upsample stage
    std::vector<int>               upsample_rates;              // per upsample stage
    std::vector<int>               resblock_kernel_sizes;       // shared across stages
    std::vector<std::vector<int>>  resblock_dilation_sizes;     // per resblock kernel
    int                            gen_istft_n_fft  = 0;        // iSTFT head transform size
    int                            gen_istft_hop_size = 0;      // iSTFT head hop
};

// plBERT — the phoneme-level BERT (ALBERT-style, weight-shared layers) that
// runs over the phoneme sequence before the text encoder consumes it.
struct PLBertConfig {
    int hidden_size              = 0;
    int num_attention_heads      = 0;
    int intermediate_size        = 0;
    int max_position_embeddings  = 0;
    int num_hidden_layers        = 0;
    int vocab_size               = 0;   // phoneme vocab (mirrors KokoroConfig::n_tokens)
};

// Model hyperparameters. Read from the model's `config.json` by Kokoro::load —
// the zero defaults below are placeholders that get overwritten on a real
// load. The fixed-rate `sample_rate` is the one true default: Kokoro outputs
// 24 kHz unconditionally.
struct KokoroConfig {
    int sample_rate              = 24000;   // Kokoro output rate (fixed)
    int n_tokens                 = 0;       // phoneme vocabulary size
    int hidden_dim               = 0;       // text encoder / predictor hidden width
    int style_dim                = 0;       // half a voice embedding (voice_dim = 2*style_dim)
    int n_layer                  = 0;       // text encoder layer count
    int n_mels                   = 0;       // mel spectrogram bands (training-side; not used at inference)
    int dim_in                   = 0;       // text encoder input channel count
    int max_dur                  = 0;       // duration predictor max frame count per phoneme
    int max_conv_dim             = 0;       // decoder feature-stack channel cap
    int text_encoder_kernel_size = 0;       // conv kernel in the text encoder CNN

    IStftNetConfig decoder;
    PLBertConfig   plbert;

    // Phoneme string → token id. Optional convenience for callers without a
    // separate G2P pipeline that returns ids directly — Kokoro itself only
    // needs the inverse mapping at inference.
    std::unordered_map<std::string, int> vocab;
};

// A Kokoro voice pack: the precomputed style embedding that conditions the
// predictor and the decoder. A pack ships as a tensor indexed by phoneme count
// — Kokoro selects the row matching the utterance length — so the same voice
// adapts its prosody to short vs. long inputs.
struct Voice {
    std::string       name;
    brotensor::Tensor packs;   // (max_context, voice_dim) — one style row per length

    // Select the style row for an utterance of `n_phonemes` phonemes.
    brotensor::Tensor pick_for(int n_phonemes) const;
};

// ─── KokoroDecodePrep ──────────────────────────────────────────────────────
//
// The device-resident pre-decoder intermediates the decoder LoRA trains over,
// produced by Kokoro::prepare_decode_context — exactly the inputs the decoder
// back half (decoder backbone ▶ harmonic source ▶ generator) consumes, plus
// borrowed references to the frozen backbone + generator the DecoderLora is
// built against. Owning the tensors keeps them alive while the LoRA forward
// borrows into them via context(); the prep must outlive any DecoderLora call
// that reads its context. The backbone/generator pointers alias the loaded
// Kokoro's modules, so the source Kokoro must outlive this prep.
struct KokoroDecodePrep {
    brotensor::Tensor ref_s;     // (1, 2*style_dim) — style = first style_dim cols
    brotensor::Tensor asr;       // (1, hidden_dim*total)
    brotensor::Tensor F0_pred;   // (1, 2*total)
    brotensor::Tensor N_pred;    // (1, 2*total)
    brotensor::Tensor har;       // (1, (n_fft+2)*frames) — harmonic source stack
    int total  = 0;              // asr frame count T
    int frames = 0;              // harmonic-source frame count
    const DecoderBackbone* backbone  = nullptr;   // frozen, aliases Kokoro::Impl
    const Generator*       generator = nullptr;   // frozen, aliases Kokoro::Impl

    // A borrowing DecoderLoraContext view over the owned tensors. Valid only
    // while this prep is alive (and unmoved).
    DecoderLoraContext context() const;
};

// ─── KokoroTrace ─────────────────────────────────────────────────────────
//
// Optional per-stage capture of the synthesis pipeline's intermediate tensors,
// for introspection / visualization. Pass a KokoroTrace* to synthesize() and
// each named stage is copied to host as a row-major (h x w) FP32 grid, in
// pipeline order:
//
//   phonemes   (1 x L)          BOS/EOS-wrapped input ids (as floats)
//   bert_dur   (L x 768)        plBERT contextual phoneme features
//   d_en       (hidden x L)     predictor conditioning stream (NCL)
//   t_en       (hidden x L)     text-encoder content features (NCL)
//   pred_dur   (1 x L)          per-phoneme frame counts (the alignment)
//   F0_pred    (1 x 2T)         pitch contour at frame rate
//   N_pred     (1 x 2T)         energy contour at frame rate
//   asr        (hidden x T)     duration-aligned content (NCL)
//   gen_in     (hidden x 2T)    decoder-backbone output (NCL)
//   har        ((n_fft+2) x F)  harmonic-source excitation stack
//   audio      (1 x N)          final 24 kHz waveform
//
// h/w are logical dims chosen for visualization; `data` is row-major h*w. The
// capture is a host copy per stage and only happens when a trace is requested,
// so a normal synthesize() pays nothing.
struct KokoroTrace {
    struct Stage {
        std::string        name;
        int                h = 0;
        int                w = 0;
        std::vector<float> data;   // h*w, row-major (row r = channel/feature r)
    };
    std::vector<Stage> stages;

    // Copy a tensor to host and record it as an (h x w) stage. If h*w doesn't
    // match the element count, falls back to a (1 x size) row.
    void add(std::string name, int h, int w, const brotensor::Tensor& t);
    // Record an integer vector (e.g. pred_dur) as a (1 x n) float stage.
    void add_ints(std::string name, const std::vector<int32_t>& v);
};

// The Kokoro TTS pipeline. Construct, load() a model directory, then
// synthesize(). Heavy state (weights, config, module graph) lives behind a
// pImpl so the public header stays free of brotensor module internals.
class Kokoro {
public:
    Kokoro();
    ~Kokoro();
    Kokoro(Kokoro&&) noexcept;
    Kokoro& operator=(Kokoro&&) noexcept;
    Kokoro(const Kokoro&) = delete;
    Kokoro& operator=(const Kokoro&) = delete;

    // Load config.json + the safetensors weights from `model_dir`, placing the
    // weights on `device`. Throws std::runtime_error on a missing / malformed
    // model.
    void load(const std::string& model_dir,
              brotensor::Device device = brotensor::Device::CPU);

    // Load a single voice pack from a weights file. The file is a raw
    // little-endian FP32 buffer of `rows * voice_dim` elements, row-major,
    // where `voice_dim = 2 * style_dim` (from KokoroConfig) and `rows` is the
    // maximum phoneme count supported by the pack (510 in the upstream Kokoro
    // distribution). The row count is inferred from the file size:
    // `file_size_bytes / (voice_dim * 4)` must divide evenly. PyTorch .pt
    // voice packs from the upstream distribution should be converted to this
    // raw format once, by the caller — brosoundml does not pull in a pickle
    // reader. The returned Voice's `packs` tensor lives on Device::CPU; the
    // row selected at synthesize() time is uploaded to the model's device.
    Voice load_voice(const std::string& voice_path) const;

    // Build a Voice from raw style data instead of a file — for authoring,
    // blending, or otherwise synthesizing a voice the app constructs in memory.
    // `style` is either:
    //   - exactly `voice_dim` (= 2*style_dim) floats: a single style point,
    //     broadcast across all length rows so it works for any utterance, or
    //   - a whole multiple of `voice_dim`: a full rows*voice_dim table, as-is.
    // The returned Voice's `packs` tensor lives on Device::CPU, matching
    // load_voice(); the row selected at synthesize() time is uploaded to the
    // model's device. Throws if no model is loaded or the length is not a
    // multiple of voice_dim.
    Voice make_voice(const std::vector<float>& style,
                     const std::string& name = "custom") const;

    // Run the full pipeline: phoneme token ids (see the misaki G2P note above)
    // + a voice -> a mono 24 kHz waveform. `speed` scales the predicted
    // durations: > 1 speaks faster, < 1 slower.
    //
    // When `pred_dur_out` is non-null, it receives the predictor's per-phoneme
    // frame counts — one entry per token in the BOS/EOS-wrapped sequence, so
    // its length is `phoneme_ids.size() + 2` ([0, ...ids, 0]). The output
    // sample count is a fixed multiple of the summed frame count, so callers
    // can recover per-phoneme timing as
    // `frame_offset * (samples.size() / sum(pred_dur))`.
    //
    // `cancel` is checked between the pipeline stages and inside the iSTFTNet
    // generator's per-upsample loop (the dominant cost): when it returns true
    // the call aborts and returns an empty AudioBuffer. There is no internal
    // autoregressive loop, so cancellation is at stage/upsample-level
    // granularity rather than per-sample. Empty (the default) = no cancel.
    //
    // When `trace_out` is non-null it is filled with a per-stage host copy of
    // the pipeline's intermediate tensors (see KokoroTrace) for visualization.
    AudioBuffer synthesize(const std::vector<int32_t>& phoneme_ids,
                           const Voice& voice,
                           float speed = 1.0f,
                           std::vector<int32_t>* pred_dur_out = nullptr,
                           const CancelCheck& cancel = {},
                           KokoroTrace* trace_out = nullptr) const;

    // Re-run only the decoder back half (decoder ▶ harmonic source ▶ generator)
    // from edited intermediates — for prosody editing. The four inputs all come
    // from a prior synthesize() trace (the 'asr', 'F0_pred', 'N_pred' stages and
    // the 'phonemes' stage length); the caller edits any of them and re-decodes,
    // skipping plBERT / the encoders / the predictor entirely.
    //
    //   asr      duration-aligned content, `hidden_dim * total` floats, row-major
    //            channel-major (the 'asr' stage: data[c*total + t]).
    //   total    frame count = asr width = sum of the per-phoneme durations.
    //   F0_pred  pitch contour, `2 * total` floats ('F0_pred' stage).
    //   N_pred   energy contour, `2 * total` floats ('N_pred' stage).
    //   n_phonemes_wrapped  the BOS/EOS-wrapped phoneme count (= the 'phonemes'
    //            stage length), used to pick the voice's style row. Editing F0/N
    //            or durations never changes the phoneme count, so it's unchanged
    //            from the originating synthesize().
    //
    // Editing F0/N only: pass the original asr unchanged. Editing durations:
    // rebuild asr (re-expand the text-encoder features) and resample F0/N to the
    // new frame count before calling. Throws on shape mismatch. `cancel` /
    // `trace_out` behave as in synthesize().
    AudioBuffer decode_from(const Voice& voice,
                            int n_phonemes_wrapped,
                            const std::vector<float>& asr,
                            int total,
                            const std::vector<float>& F0_pred,
                            const std::vector<float>& N_pred,
                            const CancelCheck& cancel = {},
                            KokoroTrace* trace_out = nullptr) const;

    // Training hook for the decoder LoRA: run the front half
    // (plBERT ▶ encoders ▶ predictor ▶ length-regulate) plus the harmonic
    // source, and return the device-resident decoder-back-half inputs —
    // asr / F0_pred / N_pred / ref_s / har with their frame counts — alongside
    // borrowed handles to the frozen decoder backbone + generator. This is the
    // pre-decoder context an DecoderLora::forward consumes; the decoder + the
    // generator are skipped here (the LoRA runs its own cached versions).
    //
    // `phoneme_ids` and `voice` match synthesize(); `speed` scales durations.
    // When `F0_override` / `N_override` are non-null they replace the predicted
    // pitch / energy contours (each must be exactly 2*total floats, matching the
    // frame count the predictor's durations produced) — the prosody-editing seam
    // mirrored from decode_from. The result owns its tensors; see KokoroDecodePrep.
    KokoroDecodePrep prepare_decode_context(
        const std::vector<int32_t>& phoneme_ids,
        const Voice& voice,
        float speed = 1.0f,
        const std::vector<float>* F0_override = nullptr,
        const std::vector<float>* N_override  = nullptr) const;

    const KokoroConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
