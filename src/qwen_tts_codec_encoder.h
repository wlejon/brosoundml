#pragma once

// Qwen3-TTS 12 Hz codec ENCODER (speech_tokenizer/ `encoder.*`) — 24 kHz mono
// waveform -> RVQ codes. The analysis path of the codec and the inverse of
// QwenTtsCodecDecoder; internal to the qwen_tts target. QwenTts::Impl owns one
// and QwenTts::encode_audio() runs it.
//
// Architecture (HF Mimi — asymmetric with the DAC-style decoder): a SEANet
// down-sampling encoder — input causal conv, then per-ratio [residual unit,
// ELU, strided causal conv], a final ELU + conv — at strides 4·5·6·8 = 960; a
// causal sliding-window transformer (LayerNorm + GELU MLP + LayerScale, RoPE)
// at 25 Hz; a causal /2 downsample conv to 12.5 Hz; and a
// SplitResidualVectorQuantizer that projects to the 256-d VQ space and emits one
// semantic + fifteen acoustic euclidean codes per frame (brotensor
// vq_encode_forward). Composed from brotensor ops plus the shared glue in
// qwen_tts_codec_common.h. Device-neutral (CPU + CUDA), FP32 on every backend.

#include "brosoundml/qwen_tts.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <vector>

namespace brosoundml {

// One SEANet residual unit: ELU, dilated causal conv (dim -> dim/compress, k),
// ELU, pointwise causal conv (dim/compress -> dim, k1), residual add.
struct QwenTtsEncResUnit {
    brotensor::Tensor c1w, c1b;   // (hidden, dim, k)   conv1: dim -> hidden
    brotensor::Tensor c2w, c2b;   // (dim, hidden, 1)   conv2: hidden -> dim
    int dim = 0, hidden = 0, kernel = 0;
};

// One SEANet downsample stage: a residual unit at `in_dim`, an ELU, then a
// strided causal conv (in_dim -> out_dim, k = 2*stride) that downsamples by
// `stride`.
struct QwenTtsEncBlock {
    QwenTtsEncResUnit unit;            // num_residual_layers == 1
    brotensor::Tensor down_w, down_b;  // (out_dim, in_dim, 2*stride)
    int in_dim = 0, out_dim = 0, stride = 0;
};

// One encoder_transformer block: LayerNorm (w+b), causal windowed self-attn
// (RoPE, no QK-norm, bias-free), LayerScale, LayerNorm (w+b), GELU MLP
// (fc1/fc2, bias-free), LayerScale.
struct QwenTtsEncTfLayer {
    brotensor::Tensor in_ln_w, in_ln_b, post_ln_w, post_ln_b;
    brotensor::Tensor qw, kw, vw, ow;          // (inner, model) projections, no bias
    brotensor::Tensor attn_scale, mlp_scale;   // (model,1) LayerScale diagonals
    brotensor::Tensor fc1_w, fc2_w;            // GELU MLP, no bias
};

// One split-RVQ group: an input_proj (model_dim -> codebook_dim, k1, bias-free)
// and its euclidean codebooks (EMA-combined, one per residual layer).
struct QwenTtsEncQuantGroup {
    brotensor::Tensor input_proj;             // (codebook_dim, model_dim)
    std::vector<brotensor::Tensor> codebook;  // each (bins, codebook_dim)
};

struct QwenTtsCodecEncoder {
    // ── dims (snapshot of the parsed encoder config) ──
    int num_filters = 0, model_dim = 0, codebook_dim = 0, codebook_bins = 0;
    int kernel = 0, last_kernel = 0, last_dim = 0;
    int downsample_stride = 0, downsample_rate = 0;
    int num_heads = 0, head_dim = 0, sliding_window = 0;
    int n_semantic = 0, n_acoustic = 0;
    float ln_eps = 1e-5f, rope_theta = 1e4f;

    // ── SEANet ──
    brotensor::Tensor conv_in_w, conv_in_b;    // (num_filters, 1, kernel)
    std::vector<QwenTtsEncBlock> blocks;       // one per ratio
    brotensor::Tensor conv_out_w, conv_out_b;  // (model_dim, last_dim, last_kernel)

    // ── encoder_transformer ──
    std::vector<QwenTtsEncTfLayer> layers;

    // ── downsample conv (model_dim -> model_dim, k = 2*stride, stride), no bias ──
    brotensor::Tensor down_w;

    // ── quantizer ──
    QwenTtsEncQuantGroup semantic, acoustic;

    // Build from the `encoder.*` tensors of speech_tokenizer/model.safetensors,
    // placing weights on `device` (FP32; q/k permuted into brotensor's
    // adjacent-pair RoPE layout).
    void load(const brotensor::safetensors::File& f, const QwenTtsCodecConfig& cfg,
              brotensor::Device device = brotensor::Device::CPU);

    // Encode `n_samples` of 24 kHz mono PCM (`wav`) into
    // (n_semantic + n_acoustic) * T codes, written to `codes` laid out
    // codes[k*T + t] (codebook-major). `wav` is right-padded to a whole frame
    // internally. Returns T (frame count).
    int encode(const float* wav, int n_samples, std::vector<int32_t>& codes) const;
};

}  // namespace brosoundml
