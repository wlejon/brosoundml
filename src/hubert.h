#pragma once

// HuBERT-base (16 kHz) — the semantic branch of the HiggsAudio v2 encoder.
// Internal to brosoundml (not part of the public include/ surface);
// HiggsCodecModel owns one, built from the `semantic_model.*` tensors of
// audio_tokenizer/model.safetensors.
//
// transformers `HubertModel` with the OmniVoice tokenizer's config
// (do_stable_layer_norm = false, feat_extract_norm = group,
// feat_proj_layer_norm = true, conv_bias = false):
//
//   feature extractor   7 valid (unpadded) strided convs 1 -> 512 -> ... -> 512,
//                       kernels {10,3,3,3,3,2,2}, strides {5,2,2,2,2,2,2}
//                       (total stride 320, receptive field 400), no bias;
//                       layer 0 is followed by a per-channel GroupNorm
//                       (num_groups == channels); every layer ends in GELU.
//   feature projection  LayerNorm(512) -> Linear(512 -> 768).
//   positional conv     grouped Conv1d(768, k=128, groups=16, padding=64),
//                       the SamePad trim of the last sample, GELU; added to
//                       the projected features, then encoder.layer_norm.
//   12 post-LN layers   h = LN(h + attn(h)); h = LN2(h + fc2(gelu(fc1(h)))).
//                       Attention: 12 heads of 64, biases everywhere, no
//                       mask (the whole clip attends to itself).
//
// forward_mean_hidden() returns the mean of the 13 hidden states
// (`output_hidden_states`: the post-positional, post-layer_norm embedding and
// each layer's output) — the feature the Higgs tokenizer consumes. The
// positional conv's weight norm (parametrizations original0 = g, original1 =
// v, dim = 2) is folded at load: w = g * v / ||v|| with the norm over (in, out)
// per kernel position. Composed entirely from brotensor ops (conv1d,
// group_norm, layernorm, gelu, flash attention, linear); FP32 on every
// backend.

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <string>
#include <vector>

namespace brosoundml {

struct HubertConfig {
    int hidden_size         = 768;
    int num_hidden_layers   = 12;
    int num_attention_heads = 12;
    int intermediate_size   = 3072;
    std::vector<int> conv_dim    = {512, 512, 512, 512, 512, 512, 512};
    std::vector<int> conv_kernel = {10, 3, 3, 3, 3, 2, 2};
    std::vector<int> conv_stride = {5, 2, 2, 2, 2, 2, 2};
    bool conv_bias = false;
    int  num_conv_pos_embeddings       = 128;
    int  num_conv_pos_embedding_groups = 16;
    float layer_norm_eps = 1e-5f;
    bool feat_extract_norm_group = true;   // "group": GroupNorm after conv 0
    bool feat_proj_layer_norm    = true;
    bool do_stable_layer_norm    = false;  // post-LN encoder (the only mode here)
};

// One feature-extractor conv (valid, strided, GELU). Layer 0 also carries the
// per-channel GroupNorm affine.
struct HubertConvLayer {
    brotensor::Tensor w;          // (Cout, Cin*k)
    brotensor::Tensor b;          // (Cout,1) — empty when conv_bias is false
    brotensor::Tensor gn_w, gn_b; // (Cout,1) — layer 0 only
    int cin = 0, cout = 0, k = 0, stride = 0;
    bool has_bias = false, has_gn = false;
};

// One post-LN transformer layer.
struct HubertLayer {
    brotensor::Tensor qw, qb, kw, kb, vw, vb, ow, ob;   // (D,D) + (D,1)
    brotensor::Tensor ln_w, ln_b;                       // after the attention residual
    brotensor::Tensor fc1_w, fc1_b;                     // (I,D) + (I,1)
    brotensor::Tensor fc2_w, fc2_b;                     // (D,I) + (D,1)
    brotensor::Tensor fln_w, fln_b;                     // final_layer_norm
};

struct HubertModel {
    HubertConfig cfg;
    std::vector<HubertConvLayer> conv;
    brotensor::Tensor fp_ln_w, fp_ln_b;   // feature_projection.layer_norm (512)
    brotensor::Tensor fp_w, fp_b;         // feature_projection.projection (768,512)
    brotensor::Tensor pos_w, pos_b;       // pos_conv_embed, weight-norm folded (768, 48*128)
    brotensor::Tensor enc_ln_w, enc_ln_b; // encoder.layer_norm
    std::vector<HubertLayer> layers;

    // Build from `<prefix>.*` (e.g. "semantic_model") of `f`, placing weights
    // on `device`. Throws std::runtime_error on a missing / mis-shaped tensor.
    void load(const brotensor::safetensors::File& f, const std::string& prefix,
              const HubertConfig& config, brotensor::Device device);

    bool loaded() const { return !layers.empty(); }

    // Frames the feature extractor yields for `n_samples` of 16 kHz input.
    int output_frames(int n_samples) const;

    // wav16: (1, n_samples) NCL FP32 on the model's device (already padded
    // however the caller wants). Returns the mean of the 13 hidden states as a
    // (Th, hidden_size) sequence on the same device; *Th_out receives Th.
    brotensor::Tensor forward_mean_hidden(const brotensor::Tensor& wav16, int n_samples,
                                          int* Th_out) const;
};

}  // namespace brosoundml
