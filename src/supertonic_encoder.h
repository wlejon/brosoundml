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

#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brosoundml {

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
};

}  // namespace brosoundml
