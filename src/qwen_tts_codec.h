#pragma once

// Qwen3-TTS 12 Hz codec decoder (speech_tokenizer/) — codes -> 24 kHz waveform.
//
// Internal to the qwen_tts target (not part of the public include/ surface).
// QwenTts::Impl owns one of these; it is built during QwenTts::load() from the
// `decoder.*` tensors of speech_tokenizer/model.safetensors and run by
// QwenTts::decode_codes().
//
// This is the deterministic tail of Qwen3-TTS synthesis: a SplitResidualVQ
// dequantizer, a causal pre-conv, an 8-layer windowed-attention transformer,
// two ConvNeXt upsample stages (x2 x2), and a DAC-style SnakeBeta SEANet
// decoder (x8 x5 x4 x3) — total upsample 1920, so a 12.5 Hz code stream maps to
// 24 kHz audio. Composed entirely from brotensor ops (conv1d / conv_transpose1d
// / snake / rms_norm / matmul / linear) plus a little hand-rolled glue
// (rotate-half RoPE, sliding-window attention, exact GELU, channel LayerNorm).
//
// CPU FP32 only for now: the glue reads host float buffers, so weights are
// uploaded to the host regardless of the requested device. The GPU path follows
// once the hand-rolled steps move onto the brotensor op surface — matching the
// staged-build convention in CLAUDE.md.

#include "brosoundml/qwen_tts.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <vector>

namespace brosoundml {

// One Qwen3-style transformer block of the codec's pre_transformer: RMSNorm,
// windowed-causal self-attention (no QK-norm), LayerScale on each residual, and
// a SwiGLU MLP. All projections are bias-free.
struct QwenTtsCodecTfLayer {
    brotensor::Tensor in_ln, post_ln;          // (D,1) RMSNorm weights
    brotensor::Tensor qw, kw, vw, ow;          // (out,in) projections, no bias
    brotensor::Tensor attn_scale, mlp_scale;   // (D,1) LayerScale diagonals
    brotensor::Tensor gate, up, down;          // (.,.) SwiGLU MLP, no bias
};

// One DAC residual unit: SnakeBeta, dilated causal conv (k7), SnakeBeta,
// pointwise causal conv (k1), with a residual add. alpha/beta are stored
// pre-exponentiated (the upstream SnakeBeta exponentiates its raw parameters).
struct QwenTtsCodecResUnit {
    brotensor::Tensor a1_alpha, a1_beta, a2_alpha, a2_beta;  // (C,1) snake params
    brotensor::Tensor c1w, c1b;   // dilated conv (C->C, k7)
    brotensor::Tensor c2w, c2b;   // pointwise conv (C->C, k1)
    int dim = 0;
    int dilation = 1;
};

// One SEANet decoder block: SnakeBeta, a transposed conv that upsamples by
// `rate` (in_dim -> out_dim), then three dilated residual units (1, 3, 9).
struct QwenTtsCodecDecBlock {
    brotensor::Tensor s_alpha, s_beta;   // (in_dim,1) pre-snake params
    brotensor::Tensor tconv_w, tconv_b;  // transposed conv (in_dim,out_dim,2*rate)
    int in_dim = 0, out_dim = 0, rate = 0;
    std::vector<QwenTtsCodecResUnit> units;
};

// One ConvNeXt upsample stage: a transposed conv (x factor) followed by a
// ConvNeXt block (depthwise k7, LayerNorm, pointwise 1->4->1 with GELU, gamma
// scale, residual).
struct QwenTtsCodecUpStage {
    brotensor::Tensor tconv_w, tconv_b;   // (latent,latent,factor) stride factor
    brotensor::Tensor dw_w, dw_b;         // depthwise causal conv (latent, k7)
    brotensor::Tensor ln_w, ln_b;         // (latent,1) LayerNorm
    brotensor::Tensor pw1_w, pw1_b;       // Linear latent -> 4*latent
    brotensor::Tensor pw2_w, pw2_b;       // Linear 4*latent -> latent
    brotensor::Tensor gamma;              // (latent,1) ConvNeXt layer scale
    int factor = 0;
};

// The codec decoder. load() uploads every weight to the host; decode() runs the
// full codes -> waveform forward pass and appends the 24 kHz samples to `wav`.
struct QwenTtsCodecDecoder {
    // ── dims (snapshot of the parsed codec config) ──
    int num_quantizers = 16;
    int num_semantic   = 1;
    int vq_dim         = 0;   // codebook embedding dim (codebook_dim/2)
    int codebook_dim   = 0;   // RVQ output width = pre_conv input
    int codebook_bins  = 0;
    int latent_dim     = 0;   // pre_conv output / transformer latent width
    int hidden         = 0;   // transformer hidden_size
    int decoder_dim    = 0;   // SEANet first conv width
    int num_heads = 0, head_dim = 0, sliding_window = 0;
    float rms_eps = 1e-5f, rope_theta = 1e4f;
    std::vector<int> upsample_rates;     // SEANet: {8,5,4,3}
    std::vector<int> upsampling_ratios;  // ConvNeXt: {2,2}

    // ── quantizer ──
    std::vector<brotensor::Tensor> codebook;  // num_quantizers x (bins, vq_dim), EMA-combined
    brotensor::Tensor out_proj_first;         // (latent_dim_proj=codebook_dim, vq_dim) for the semantic group
    brotensor::Tensor out_proj_rest;          // same, acoustic group

    // ── pre_conv (codebook_dim -> latent_dim, k3) ──
    brotensor::Tensor pre_conv_w, pre_conv_b;

    // ── pre_transformer ──
    brotensor::Tensor in_proj_w, in_proj_b;    // latent_dim -> hidden
    brotensor::Tensor out_proj_w, out_proj_b;  // hidden -> latent_dim
    brotensor::Tensor final_norm;              // (hidden,1) RMSNorm
    std::vector<QwenTtsCodecTfLayer> layers;

    // ── upsample stages + SEANet decoder ──
    std::vector<QwenTtsCodecUpStage> upsample;
    brotensor::Tensor dec0_w, dec0_b;          // latent_dim -> decoder_dim, k7
    std::vector<QwenTtsCodecDecBlock> blocks;
    brotensor::Tensor dec_final_alpha, dec_final_beta;  // SnakeBeta on output_dim
    brotensor::Tensor dec_out_w, dec_out_b;    // output_dim -> 1, k7

    // Build from the `decoder.*` tensors of speech_tokenizer/model.safetensors,
    // placing weights on `device` (FP32 on every backend; q/k permuted into
    // brotensor's adjacent-pair RoPE layout).
    void load(const brotensor::safetensors::File& f, const QwenTtsCodecConfig& cfg,
              brotensor::Device device = brotensor::Device::CPU);

    // Decode `num_frames` frames of `num_quantizers` codes (laid out
    // codes[k*num_frames + t]) to 24 kHz PCM, appended to `wav`.
    void decode(const int32_t* codes, int num_quantizers, int num_frames,
                std::vector<float>& wav) const;
};

}  // namespace brosoundml
