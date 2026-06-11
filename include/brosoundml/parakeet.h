#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── Parakeet (NVIDIA FastConformer-TDT ASR) ───────────────────────────────
//
// Parakeet-TDT-0.6B-v3 is NVIDIA's multilingual streaming-capable speech-to-
// text model: a FastConformer encoder feeding a Token-and-Duration Transducer
// (TDT) decoder. brosoundml targets the HuggingFace `transformers` Parakeet
// checkpoints — `config.json` + `model.safetensors` in a model directory —
// not the native NeMo `.nemo` archive. The tokenizer (a unified SentencePiece
// unigram, 8192 BPE pieces + a blank) is the caller's responsibility, exactly
// as with Whisper: brosoundml emits token ids (and their encoder-frame
// positions for word timestamps); the caller detokenizes.
//
// Pipeline — the order brosoundml drives the components:
//
//   1. Log-mel front-end   16 kHz mono PCM -> (num_mel_bins=128, T) log-mel,
//                          NeMo recipe: pre-emphasis 0.97, STFT (n_fft=512,
//                          win=400, hop=160, Hann, center), power spectrum,
//                          Slaney mel filterbank, log(x + 2^-24), per-feature
//                          (per-mel-bin) mean/var normalization over time.
//   2. FastConformer enc   8x depthwise-separable conv2d subsampling, then 24
//                          Conformer blocks (½-FFN macaron, Transformer-XL
//                          relative-position MHA, conv module
//                          [pointwise→GLU→depthwise k=9→BatchNorm→SiLU→
//                          pointwise], ½-FFN, final LayerNorm). Outputs
//                          (T/8, d_model=1024); a projector maps that to the
//                          decoder width (640).
//   3. TDT decoder         a prediction network (token embedding + 2-layer
//                          LSTM + projection) and a joint network
//                          (relu(enc_proj + dec_proj) -> token+duration
//                          logits). Greedy TDT decode emits a token plus a
//                          frame-duration per step, skipping `duration`
//                          encoder frames at once — the source of TDT's speed.
//
// brotensor op coverage: stft + complex magnitude + matmul (mel front-end),
// conv2d (subsampling + conv module, depthwise via groups), silu, layernorm,
// batch_norm_inference, self_attention_bias_forward (the rel-pos attention,
// with the Transformer-XL position term supplied as the additive bias),
// embedding_lookup, matmul. No new brotensor op is required.
//
// Device-neutral: load(device) places every weight on the chosen backend and
// the forward pass dispatches through brotensor device ops (CPU / CUDA /
// Metal), the same bar Qwen3-TTS / Qwen3-ASR meet. FP32 throughout.

// Encoder hyperparameters — the nested `encoder_config` of the HF config.json.
struct ParakeetEncoderConfig {
    int  num_mel_bins                 = 128;
    int  hidden_size                  = 1024;   // d_model
    int  num_hidden_layers            = 24;
    int  num_attention_heads          = 8;
    int  intermediate_size            = 4096;   // FFN inner width
    int  conv_kernel_size             = 9;      // depthwise conv module kernel
    int  subsampling_factor           = 8;
    int  subsampling_conv_channels    = 256;
    int  subsampling_conv_kernel_size = 3;
    int  subsampling_conv_stride      = 2;
    int  max_position_embeddings      = 5000;
    bool scale_input                  = false;  // sqrt(d_model) input scaling
    bool attention_bias               = false;  // q/k/v/o + FFN linear biases
    bool convolution_bias             = false;  // conv-module conv biases
    // hidden_act is SiLU/Swish (fixed for the FastConformer encoder).
};

// Top-level Parakeet-TDT hyperparameters — the HF config.json.
struct ParakeetConfig {
    int sample_rate          = 16000;  // model input rate (fixed)

    int vocab_size           = 8193;   // 8192 SentencePiece pieces + blank
    int blank_token_id       = 8192;   // TDT blank (distinct from pad)
    int pad_token_id         = 2;

    int decoder_hidden_size  = 640;    // LSTM + joint width
    int num_decoder_layers   = 2;      // prediction-network LSTM layers
    int max_symbols_per_step = 10;     // greedy emissions cap per frame
    // hidden_act of the joint network is ReLU (fixed).

    std::vector<int> durations = {0, 1, 2, 3, 4};  // TDT frame-skip choices

    ParakeetEncoderConfig encoder;

    // Seconds of audio one encoder frame spans: subsampling_factor * hop /
    // sample_rate = 8 * 160 / 16000 = 0.08 s. Used to map a token's encoder-
    // frame index to a wall-clock timestamp.
    double frame_seconds() const {
        return static_cast<double>(encoder.subsampling_factor) * 160.0 /
               static_cast<double>(sample_rate);
    }
};

// The Parakeet STT pipeline. Construct, load() a model directory, then
// transcribe(). Heavy state (weights, config, module graph) lives behind a
// pImpl so the public header stays free of brotensor module internals.
class Parakeet {
public:
    Parakeet();
    ~Parakeet();
    Parakeet(Parakeet&&) noexcept;
    Parakeet& operator=(Parakeet&&) noexcept;
    Parakeet(const Parakeet&) = delete;
    Parakeet& operator=(const Parakeet&) = delete;

    // Load config.json + model.safetensors from `model_dir`, placing every
    // weight on `device`. Throws std::runtime_error on a missing / malformed
    // model.
    void load(const std::string& model_dir,
              brotensor::Device device = brotensor::Device::CPU);

    // Result of a transcribe() call. `token_ids` is the raw greedy TDT id
    // stream (SentencePiece piece ids, no blank/pad — the caller detokenizes).
    // `token_frames[i]` is the encoder-frame index at which token_ids[i] was
    // emitted; multiply by config().frame_seconds() for a start timestamp.
    struct Transcription {
        std::vector<int32_t> token_ids;
        std::vector<int32_t> token_frames;
    };

    // Streaming sink: invoked once per emitted token, in decode order. Empty
    // (the default) = no streaming. Runs synchronously on the decode thread.
    using TokenCallback = std::function<void(int32_t token_id)>;

    struct TranscribeOptions {
        // Cap on total emitted tokens; 0 => no cap (decode the whole clip).
        int max_new_tokens = 0;
        // Polled once per encoder frame advance: true => stop and return what
        // we have. Empty (the default) = no cancel.
        CancelCheck   cancel   = {};
        // Invoked once per emitted token. Empty (the default) = no streaming.
        TokenCallback on_token = {};
    };

    // Run the full pipeline: 16 kHz mono PCM -> token ids + frame positions.
    Transcription transcribe(const AudioBuffer& audio,
                             const TranscribeOptions& opts) const;

    // Default-options overload. (A separate overload rather than a defaulted
    // `opts = {}` argument: GCC 12 rejects a brace-init default argument for a
    // nested struct that has default member initializers — CWG2335.)
    Transcription transcribe(const AudioBuffer& audio) const;

    const ParakeetConfig& config() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
