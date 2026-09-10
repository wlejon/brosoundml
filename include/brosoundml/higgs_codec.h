#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── HiggsAudio v2 tokenizer (25 Hz, 8-codebook RVQ) ────────────────────────
//
// The neural audio codec OmniVoice speaks: Boson AI's HiggsAudio v2 tokenizer
// (transformers `HiggsAudioV2TokenizerModel`). 24 kHz mono in, 25 codes per
// second out — eight residual-VQ codebooks of 1024 entries — and back. A
// frame is hop_length = 960 samples.
//
//   decode   codes (8, T) -> per-level codebook lookup [1024, 64] -> project_out
//            (64 -> 1024) -> sum over the 8 levels -> fc2 (1024 -> 256) -> a
//            DAC synthesis stack: conv (256 -> 1024, k7), five decoder blocks
//            (Snake, ConvTranspose1d x{8,5,4,2,3} with kernel 2*stride, padding
//            ceil(stride/2), output_padding stride%2, then three residual
//            units Snake -> conv k7 dilation {1,3,9} -> Snake -> conv k1 with a
//            centred crop), Snake, conv (32 -> 1, k7). No output tanh. Weight
//            norm is already folded on disk.
//   encode   the acoustic branch (a DAC analysis stack: conv 1 -> 64 k7, five
//            encoder blocks of three residual units + Snake + a strided conv
//            with kernel 2*stride, 64 -> 128 -> 256 -> 512 -> 1024 -> 2048,
//            Snake, conv 2048 -> 256 k3) in parallel with the semantic branch
//            (resample 24 -> 16 kHz, pad 160 each side, HuBERT-base, the mean
//            of all 13 hidden states, every 2nd frame, a small SemanticEncoder
//            conv stack -> 768), concatenated [256 | 768] -> fc (1024 -> 1024)
//            -> residual VQ (per level project_in 1024 -> 64, nearest code by
//            Euclidean distance, subtract the decoded residual).
//
// brotensor op coverage: conv1d / conv_transpose1d / pad1d, snake, elu, gelu,
// layernorm / group_norm, flash attention, embedding_lookup, vq_encode,
// linear. The 24 -> 16 kHz resample is torchaudio's windowed sinc composed
// from pad1d + a strided conv1d (one output channel per phase), so the
// semantic branch sees the reference's input. No new kernels. Weights are F32
// on disk and stay FP32 on every backend, so CUDA tracks CPU to float
// round-off.
//
// The decoder alone serves synthesis (OmniVoice -> codes -> waveform); the
// encoder (with HuBERT) serves voice cloning (reference clip -> codes) and
// analysis. Both halves load from the one audio_tokenizer/ directory.

// Snapshot of audio_tokenizer/config.json — the fields the module graph needs.
struct HiggsCodecConfig {
    int sample_rate    = 24000;
    int hop_length     = 960;    // samples per frame
    int frame_rate     = 25;     // ceil(sample_rate / hop_length)
    int num_quantizers = 8;      // RVQ levels actually used (from target_bandwidths)
    int codebook_size  = 1024;   // entries per level
    int codebook_dim   = 64;     // per-level code width (project_in/out)
    int hidden_size    = 1024;   // RVQ input/output width (fc / project_out target)
    int acoustic_dim   = 256;    // DAC latent width (encoder out / fc2 out)
    int semantic_dim   = 768;    // HuBERT hidden width
    int semantic_sample_rate       = 16000;
    int semantic_downsample_factor = 2;
    int encoder_dim = 64;        // DAC encoder first conv width
    int decoder_dim = 1024;      // DAC decoder first conv width
    std::vector<int> encoder_rates;  // DAC encoder strides, in order
    std::vector<int> decoder_rates;  // DAC decoder strides, in order ({8,5,4,2,3})
    bool has_encoder = true;         // encoder + HuBERT weights present
};

// The codec. load() reads audio_tokenizer/{config.json, model.safetensors} and
// places every weight on `device`; decode() and encode() then dispatch through
// brotensor device ops (FP32 on every backend).
class HiggsCodec {
public:
    HiggsCodec();
    ~HiggsCodec();
    HiggsCodec(HiggsCodec&&) noexcept;
    HiggsCodec& operator=(HiggsCodec&&) noexcept;
    HiggsCodec(const HiggsCodec&) = delete;
    HiggsCodec& operator=(const HiggsCodec&) = delete;

    // `dir` is the audio_tokenizer directory (config.json + model.safetensors).
    // `decoder_only` skips the DAC encoder + HuBERT (halves the load; encode()
    // then throws). Throws std::runtime_error on a missing / malformed model.
    void load(const std::string& dir,
              brotensor::Device device = brotensor::Device::CPU,
              bool decoder_only = false);

    // codes -> 24 kHz PCM. `codes` holds num_quantizers * num_frames entries
    // laid out codes[q * num_frames + t] (codebook-major); num_quantizers may
    // be <= config().num_quantizers (fewer levels = coarser audio). Output has
    // num_frames * hop_length samples. Throws if no model is loaded or a code
    // is out of range.
    AudioBuffer decode(const int32_t* codes, int num_quantizers, int num_frames) const;
    AudioBuffer decode(const std::vector<int32_t>& codes, int num_quantizers,
                       int num_frames) const;

    // 24 kHz PCM -> codes. `audio` is resampled to 24 kHz mono as needed and
    // right-padded to a whole frame. Returns num_quantizers * num_frames codes
    // laid out codes[q * num_frames + t] — the same layout decode() takes, so
    // encode ▸ decode round-trips. *num_frames_out receives num_frames.
    // Throws if no model is loaded, the encoder was not loaded, or `audio` is
    // empty.
    std::vector<int32_t> encode(const AudioBuffer& audio, int* num_frames_out = nullptr) const;

    const HiggsCodecConfig& config() const;
    bool loaded() const;
    bool has_encoder() const;
    brotensor::Device device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
