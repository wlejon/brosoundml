#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace brosoundml {

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
    AudioBuffer synthesize(const std::vector<int32_t>& phoneme_ids,
                           const Voice& voice,
                           float speed = 1.0f,
                           std::vector<int32_t>* pred_dur_out = nullptr,
                           const CancelCheck& cancel = {}) const;

    const KokoroConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
