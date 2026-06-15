#pragma once

// brosoundml/src/supertonic_encoder.h — the Supertonic AE audio→latent encoder.
//
// Supertone released only the AE DECODER (the vocoder). The encoder that
// produced the latents the flow model was trained against was never exported,
// so we TRAIN our own: it maps a [idim, T] spec (SupertonicSpec) to the real
// (de-normalised, de-chunked) latent [latent_dim, T] that Supertonic::decode_real
// inverts. Trained by reconstruction through the FROZEN decoder.
//
// Architecture (a near-mirror of the decoder; we own the exact shape):
//   conv_in  : idim -> hidden, kernel 7, symmetric edge-pad (wide initial context)
//   blocks   : num_layers ConvNeXt-1D, kernel 5, dilation 1 (the FD-verified
//              codebase-standard block — k5 keeps the 2*dil pad length-preserving)
//   proj_out : hidden -> latent_dim, 1x1
// Non-causal and length-preserving: T spec frames -> T latent frames.
//
// Weights are TRAINABLE. init() Xavier-initialises them on a device; the trainer
// owns the forward_train caches + backward (supertonic_encoder_backward).

#include "supertonic_internal.h"
#include "supertonic_backward.h"   // st_detail forward_train atoms + caches

#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brosoundml {

struct SupertonicEncoder;

// Per-call activations the training backward consumes. The 1×1 convs and the
// LayerNorm are fully recoverable from the st_detail caches; only the two
// conv-path convs (conv_in, depthwise dw) need their unpadded input cached so
// the weight-gradient pass can re-pad and call conv1d_backward_weight.
struct SupertonicEncoderCache {
    int T = 0;
    brotensor::Tensor spec_in;             // (idim, T) conv_in unpadded input
    st_detail::PConvCache conv_in;
    struct Block {
        brotensor::Tensor   h_in;          // (hidden, T) dw unpadded input
        st_detail::ConvNeXtCache cn;       // dw geom, LN caches, pw1/pw2 (x_in cached)
    };
    std::vector<Block> blocks;
    st_detail::PConvCache proj_out;        // matmul path: x_in = final hidden (hidden, T)
};

// Weight gradients, same shape as the encoder weights (the .w/.b of each ConvW
// hold dW/dB; ln_g/ln_b hold the LayerNorm affine grads). Accumulate across a
// batch; zero() once per optimiser step.
struct SupertonicEncoderGrads {
    st_detail::ConvW                      conv_in;
    std::vector<st_detail::ConvNeXtBlock> blocks;
    st_detail::ConvW                      proj_out;
    void zero(const SupertonicEncoder& enc);   // allocate zeroed grads matching shapes
};

struct SupertonicEncoder {
    int idim         = 1253;  // SupertonicSpec idim (1025 mag + 228 mel)
    int hidden       = 512;
    int latent_dim   = 24;
    int num_layers   = 10;
    int ksz_init     = 7;     // conv_in kernel
    int ksz          = 5;     // ConvNeXt depthwise kernel (length-preserving block)
    int intermediate = 2048;  // ConvNeXt pointwise expansion
    brotensor::Device dev = brotensor::Device::CPU;

    st_detail::ConvW                      conv_in;   // idim -> hidden, k = ksz_init
    std::vector<st_detail::ConvNeXtBlock> blocks;    // num_layers, k = ksz, dil 1
    st_detail::ConvW                      proj_out;  // hidden -> latent_dim, 1x1

    // Xavier-initialise all weights on `device`, deterministically from `seed`.
    void init(brotensor::Device device, std::uint64_t seed = 1234);

    // spec: (idim, T) channel-major. Returns latent (latent_dim, T) channel-major.
    brotensor::Tensor forward(const brotensor::Tensor& spec, int T) const;

    // Training forward: same math as forward(), but emits the cache the backward
    // consumes. `latent` is (latent_dim, T) channel-major.
    void forward_train(const brotensor::Tensor& spec, int T,
                       brotensor::Tensor& latent, SupertonicEncoderCache& cache) const;

    // Reverse pass: dLatent (latent_dim, T) channel-major -> accumulated weight
    // gradients in `grads` (caller zero()s once per optimiser step). The input
    // gradient (to the spec) is not produced — the spec is a fixed transform.
    void backward(const SupertonicEncoderCache& cache,
                  const brotensor::Tensor& dLatent,
                  SupertonicEncoderGrads& grads) const;
};

}  // namespace brosoundml
