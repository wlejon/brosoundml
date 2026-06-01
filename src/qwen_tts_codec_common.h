#pragma once

// Shared internal glue for the Qwen3-TTS 12 Hz codec — used by both the decoder
// (codes -> waveform, qwen_tts_codec.*) and the encoder (waveform -> codes,
// qwen_tts_codec_encoder.*). The two halves are architecturally asymmetric (the
// encoder is an HF-Mimi analysis stack with ELU + LayerNorm + a downsampling
// SEANet; the decoder is a DAC-style synthesis stack with SnakeBeta + RMSNorm +
// ConvNeXt upsampling), but they share weight upload, NCL<->SEQ layout swaps,
// causal convolution, the EMA codebook combine, LayerScale, and the windowed
// codec attention. Those live here so neither file re-rolls them.

#include "qwen_tts_device.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {
namespace qcodec {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;

[[noreturn]] inline void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: qwen-tts codec: " + msg);
}

inline const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Upload a 2D-flattened weight to FP32 on `dev` (codec weights are F32 on disk).
inline bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols,
                     bt::Device dev) {
    bt::Tensor t;
    { bt::DeviceScope cpu(bt::Device::CPU); sf::upload(need(f, name), rows, cols, t); }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
inline bt::Tensor up_vec(const sf::File& f, const std::string& name, int c, bt::Device dev) {
    return up(f, name, c, 1, dev);
}

// SnakeBeta parameter [C], exponentiated host-side (the upstream SnakeBeta
// applies exp() to its raw alpha/beta), then placed on `dev`.
inline bt::Tensor up_snake(const sf::File& f, const std::string& name, int c, bt::Device dev) {
    bt::Tensor t;
    { bt::DeviceScope cpu(bt::Device::CPU); sf::upload(need(f, name), c, 1, t); }
    float* p = t.host_f32_mut();
    for (int i = 0; i < c; ++i) p[i] = std::exp(p[i]);
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}

// EMA codebook -> effective embedding: embed[k] = embedding_sum[k] /
// max(cluster_usage[k], 1e-5). Combined host-side, then placed on `dev`. The
// sum tensor is named "embedding_sum" in the decoder's custom quantizer and
// "embed_sum" in the encoder's HF-Mimi quantizer, hence `sum_name`.
inline bt::Tensor load_codebook(const sf::File& f, const std::string& prefix,
                                int bins, int dim, bt::Device dev,
                                const char* sum_name = "embedding_sum") {
    bt::Tensor embed, usage;
    {
        bt::DeviceScope cpu(bt::Device::CPU);
        sf::upload(need(f, prefix + "." + sum_name), bins, dim, embed);
        sf::upload(need(f, prefix + ".cluster_usage"), bins, 1, usage);
    }
    float* e = embed.host_f32_mut();
    const float* u = usage.host_f32();
    for (int k = 0; k < bins; ++k) {
        const float d = 1.0f / (u[k] > 1e-5f ? u[k] : 1e-5f);
        float* row = e + static_cast<std::size_t>(k) * dim;
        for (int j = 0; j < dim; ++j) row[j] *= d;
    }
    return (dev == bt::Device::CPU) ? embed : embed.to(dev);
}

// ── layout converters (on-device gather/scatter) ─────────────────────────────
// NCL (1, C*L) viewed channel-major as (C, L) <-> SEQ (L, C). An NCL buffer is
// exactly NCHW with N=1, H=1, W=L, so brotensor's nchw_to_sequence /
// sequence_to_nchw (FP32, CPU+CUDA+Metal) are the transpose — no host round-trip.
inline bt::Tensor ncl_to_seq(const bt::Tensor& x, int C, int L) {
    bt::Tensor y;
    bt::nchw_to_sequence(x, /*N=*/1, C, /*H=*/1, /*W=*/L, y);   // (L, C)
    return y;
}
inline bt::Tensor seq_to_ncl(const bt::Tensor& x, int L, int C) {
    bt::Tensor y;
    bt::sequence_to_nchw(x, /*N=*/1, C, /*H=*/1, /*W=*/L, y);   // (1, C*L)
    return y;
}

// Stride-1 causal conv: left-pad dilation*(k-1), valid conv, Lout == L.
inline bt::Tensor causal_conv(const bt::Tensor& x_ncl, int Cin, int L, const bt::Tensor& w,
                              const bt::Tensor* b, int Cout, int k, int dilation, int groups) {
    bt::Tensor scratch, y;
    bt::causal_conv1d(x_ncl, w, b, /*N=*/1, Cin, L, Cout, k, /*stride=*/1,
                      dilation, groups, scratch, y);
    return y;
}

// Strided causal conv (EnCodec/Mimi downsample): left-pad (k - stride), valid
// stride-`stride` conv. With L a multiple of `stride` the extra right padding
// EnCodec adds to cover the tail is zero, so Lout == L / stride. Callers that
// rely on that (the SEANet encoder) pre-pad the waveform to a multiple of the
// total downsample so every stage divides evenly.
inline bt::Tensor strided_causal_conv(const bt::Tensor& x_ncl, int Cin, int L,
                                      const bt::Tensor& w, const bt::Tensor* b,
                                      int Cout, int k, int stride) {
    bt::Tensor padded, y;
    const int pad_left = k - stride;
    bt::pad1d_forward(x_ncl, /*N=*/1, Cin, L, pad_left, /*pad_right=*/0, /*mode=*/0, padded);
    bt::conv1d(padded, w, b, /*N=*/1, Cin, L + pad_left, Cout, k, stride,
               /*padding=*/0, /*dilation=*/1, /*groups=*/1, y);
    return y;
}

// Causal transposed conv: conv_transpose1d then trim (kernel-stride) from the
// right per channel (left_pad is 0 for these causal upsamplers). Lout = L*stride.
inline bt::Tensor trans_conv(const bt::Tensor& x_ncl, int Cin, int L, const bt::Tensor& w,
                             const bt::Tensor* b, int Cout, int k, int stride) {
    bt::Tensor full;
    bt::conv_transpose1d_forward(x_ncl, w, b, /*N=*/1, Cin, L, Cout, k, stride,
                                 /*padding=*/0, /*output_padding=*/0,
                                 /*dilation=*/1, full);
    const int L_full = (L - 1) * stride + k;
    const int trim   = k - stride;
    const int L_out  = L_full - trim;
    bt::Tensor y = bt::Tensor::zeros_on(x_ncl.device, 1, Cout * L_out, bt::Dtype::FP32);
    for (int c = 0; c < Cout; ++c)
        bt::copy_d2d(full, c * L_full, y, c * L_out, L_out);   // drop the right tail
    return y;
}

// SnakeBeta in place (alpha/beta pre-exp'd).
inline void snake(bt::Tensor& x_ncl, int C, int L, const bt::Tensor& alpha,
                  const bt::Tensor& beta) {
    bt::snake_forward(x_ncl, alpha, &beta, /*N=*/1, C, L, x_ncl);
}

// hs(T,C) += scale[c] * delta(T,C)  (LayerScale residual).
inline void add_layerscale(bt::Tensor& hs, const bt::Tensor& delta, const bt::Tensor& scale) {
    bt::Tensor scaled;
    bt::broadcast_mul(delta, scale, scaled);
    bt::add_inplace(hs, scaled);
}

// Windowed causal self-attention over pre-RoPE Q/K/V (T, num_heads*head_dim).
// brotensor's flash_attention_windowed_forward (FP32, CPU+CUDA) does the whole
// thing on-device: query i attends keys [max(0, i-window+1), i]. window >= T
// degenerates to plain causal, so this one call covers both the within-window
// regime and the long-audio path.
inline bt::Tensor codec_attn(const bt::Tensor& q, const bt::Tensor& k, const bt::Tensor& v,
                             int num_heads, int window) {
    bt::Tensor O;
    bt::flash_attention_windowed_forward(q, k, v, /*d_mask=*/nullptr, num_heads,
                                         window, O);
    return O;
}

}  // namespace qcodec
}  // namespace brosoundml
