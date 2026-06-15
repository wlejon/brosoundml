#pragma once

#include <brotensor/tensor.h>

namespace brosoundml {

// ─── FastConformer encoder config ──────────────────────────────────────────
//
// Shared hyperparameters for NVIDIA's FastConformer / NEST encoder — the
// acoustic backbone reused across the NeMo model family (Parakeet-TDT ASR and
// streaming Sortformer diarization both ride it). The encoder is one graph:
//
//   8x depthwise-separable conv2d subsampling (pre-encode) -> N Conformer
//   blocks (½-FFN macaron, Transformer-XL relative-position MHA, conv module
//   [pointwise → GLU → depthwise → BatchNorm → SiLU → pointwise], ½-FFN, final
//   LayerNorm).
//
// The struct is a superset of what either model needs; the differences between
// the HF Parakeet checkpoint and the native NeMo NEST checkpoint are captured
// as flags here:
//   - `scale_input` (NeMo `xscaling`): scale the pre-encode output by
//     sqrt(d_model) before the Conformer stack. Parakeet: false. NEST: true.
//   - projection / FFN / conv biases: present in NEST (`untie_biases`), absent
//     in the HF Parakeet export. The loader picks them up when present and the
//     forward applies them by presence, so a single code path serves both.
//   - `normalize_features` (NeMo preprocessor `normalize`): per-mel-bin mean/var
//     normalization over time. Parakeet: per_feature (true). NEST/Sortformer:
//     NA (false).
struct FastConformerConfig {
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
    bool normalize_features           = true;   // per-mel-bin mean/var over time
    // hidden_act is SiLU/Swish (fixed for the FastConformer encoder).

    // NeMo center-STFT yields one trailing pad frame (valid mel frames =
    // num_samples / hop_length). When set, the encoder zeroes that pad mel
    // frame, key-masks the corresponding pad encoder frame in self-attention,
    // and zeroes it before the conv module — matching NeMo's length handling.
    // Off by default (the HF Parakeet path does not mask); Sortformer turns it
    // on. The pad output frame is still produced (callers mask/ignore it), only
    // its influence on the valid frames is removed.
    bool mask_padding                 = false;
};

}  // namespace brosoundml
