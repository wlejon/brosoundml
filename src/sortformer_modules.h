#pragma once

// Sortformer-specific submodules (internal): the Transformer-encoder head that
// sits on top of the shared FastConformer encoder, plus the two-layer sigmoid
// output head. Composes the forward from the shared nn-modules (modules.h);
// loaded from `model.safetensors` via load(). FP32 on whichever device the
// weights live on.
//
// Converted state-dict key scheme (scripts/convert-sortformer.py):
//   sortformer.encoder_proj.{weight,bias}              (fc_d_model -> tf_d_model)
//   transformer.layers.{i}.norm1.{weight,bias}         (post-attn LayerNorm)
//   transformer.layers.{i}.attn.{q,k,v,o}.{weight,bias}
//   transformer.layers.{i}.norm2.{weight,bias}         (post-FFN LayerNorm)
//   transformer.layers.{i}.ff.in.{weight,bias}         (tf -> inner)
//   transformer.layers.{i}.ff.out.{weight,bias}        (inner -> tf)
//   sortformer.first_hidden_to_hidden.{weight,bias}    (tf -> tf)
//   sortformer.single_hidden_to_spks.{weight,bias}     (tf -> num_spks)

#include "brosoundml/modules.h"
#include "brosoundml/sortformer.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <vector>

namespace brosoundml {

// ─── One post-LN Transformer-encoder block (NeMo TransformerEncoderBlock) ──
//
//   a = MHA(x)                         // first_sub_layer self-attention
//   x = norm1(x + a)                   // layer_norm_1
//   f = ff_out(relu(ff_in(x)))         // second_sub_layer position-wise FFN
//   x = norm2(x + f)                   // layer_norm_2
//
// (pre_ln = false in the shipped checkpoint; the pre-LN variant is not loaded.)
struct SortformerTransformerBlock {
    LayerNorm norm1;
    MHA       attn;
    LayerNorm norm2;
    Linear    ff_in;    // (inner, tf)
    Linear    ff_out;   // (tf, inner)

    // x: (T, tf_d_model) updated into out. d_mask: optional length-T key mask.
    void forward(const brotensor::Tensor& x, const float* d_mask,
                 brotensor::Tensor& out) const;
};

// ─── Sortformer head: encoder_proj -> Transformer stack -> sigmoid head ────
struct SortformerHead {
    SortformerConfig  cfg;
    brotensor::Device device = brotensor::Device::CPU;

    Linear encoder_proj;                            // (tf, fc)
    std::vector<SortformerTransformerBlock> layers;
    Linear first_hidden_to_hidden;                  // (tf, tf)
    Linear single_hidden_to_spks;                   // (num_spks, tf)

    void load(const brotensor::safetensors::File& f,
              const SortformerConfig& cfg, brotensor::Device device);

    // enc_out: (T, fc_d_model) FastConformer states. Writes (T, num_spks)
    // speaker activity probabilities (post-sigmoid) to `preds`. d_mask optional.
    void forward(const brotensor::Tensor& enc_out, const float* d_mask,
                 brotensor::Tensor& preds) const;
};

}  // namespace brosoundml
