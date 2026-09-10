#pragma once

// Shared internal glue for the HiggsAudio v2 codec (src/higgs_codec.cpp) and
// its HuBERT semantic branch (src/hubert.cpp): the error prefix, safetensors
// upload helpers, the NCL <-> SEQ layout swaps, and the "same"-padded
// non-causal convolutions the DAC / HuBERT / SemanticEncoder stacks are built
// from. Everything dispatches through brotensor device ops, FP32 on every
// backend. The Qwen3-TTS codec has its own copy of the same idea
// (qwen_tts_codec_common.h) — that one is causal and SnakeBeta; this one is
// centred padding and plain Snake, so the two stay separate.

#include "qwen_tts_device.h"   // qtd::linear / gather_rows / to_host / upload_idx

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {
namespace hcodec {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;

[[noreturn]] inline void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: higgs codec: " + msg);
}

inline const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Upload a 2D-flattened F32 weight to `dev`. `rows*cols` must equal the
// on-disk element count (sf::upload checks the byte count).
inline bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols,
                     bt::Device dev) {
    const sf::TensorView& v = need(f, name);
    if (v.dtype != sf::Dtype::F32) fail("tensor '" + name + "' is not F32");
    bt::Tensor t;
    { bt::DeviceScope cpu(bt::Device::CPU); sf::upload(v, rows, cols, t); }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
inline bt::Tensor up_vec(const sf::File& f, const std::string& name, int c, bt::Device dev) {
    return up(f, name, c, 1, dev);
}

// ── layout converters (on-device gather/scatter) ─────────────────────────────
// NCL (1, C*L) viewed channel-major as (C, L) <-> SEQ (L, C). An NCL buffer is
// NCHW with N=1, H=1, W=L, so brotensor's nchw_to_sequence / sequence_to_nchw
// are the transpose — no host round-trip.
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

// Stride-1 "same" conv: symmetric zero padding dilation*(k-1)/2 (k odd), so
// Lout == L. torch Conv1d(padding=((k-1)//2)*dilation).
inline bt::Tensor conv_same(const bt::Tensor& x_ncl, int Cin, int L, const bt::Tensor& w,
                            const bt::Tensor* b, int Cout, int k, int dilation = 1,
                            int groups = 1) {
    bt::Tensor y;
    bt::conv1d(x_ncl, w, b, /*N=*/1, Cin, L, Cout, k, /*stride=*/1,
               /*padding=*/((k - 1) / 2) * dilation, dilation, groups, y);
    return y;
}

// Strided conv with explicit symmetric padding (DAC's downsampling conv:
// kernel 2*stride, padding ceil(stride/2)). Lout = (L + 2*pad - k)/stride + 1.
inline bt::Tensor conv_strided(const bt::Tensor& x_ncl, int Cin, int L, const bt::Tensor& w,
                               const bt::Tensor* b, int Cout, int k, int stride, int pad) {
    bt::Tensor y;
    bt::conv1d(x_ncl, w, b, /*N=*/1, Cin, L, Cout, k, stride, pad,
               /*dilation=*/1, /*groups=*/1, y);
    return y;
}

// Valid (unpadded) strided conv — HuBERT's feature extractor.
inline bt::Tensor conv_valid(const bt::Tensor& x_ncl, int Cin, int L, const bt::Tensor& w,
                             const bt::Tensor* b, int Cout, int k, int stride) {
    return conv_strided(x_ncl, Cin, L, w, b, Cout, k, stride, /*pad=*/0);
}

// DAC decoder upsampler: ConvTranspose1d(kernel 2*stride, padding
// ceil(stride/2), output_padding stride % 2) -> Lout == L*stride exactly.
// brotensor's transposed-conv weight layout (C_in, C_out*k) is torch's
// ConvTranspose1d [C_in, C_out, k] flattened, so the checkpoint uploads as-is.
inline bt::Tensor trans_conv_up(const bt::Tensor& x_ncl, int Cin, int L, const bt::Tensor& w,
                                const bt::Tensor* b, int Cout, int stride) {
    bt::Tensor y;
    bt::conv_transpose1d_forward(x_ncl, w, b, /*N=*/1, Cin, L, Cout, /*kL=*/2 * stride,
                                 stride, /*padding=*/(stride + 1) / 2,
                                 /*output_padding=*/stride % 2, /*dilation=*/1, y);
    return y;
}

// Plain Snake in place: y = x + (1/alpha_c) * sin^2(alpha_c * x). The DAC
// Snake1d stores alpha raw (no exp), (1, C, 1) on disk -> (C, 1) here.
inline void snake(bt::Tensor& x_ncl, int C, int L, const bt::Tensor& alpha) {
    bt::snake_forward(x_ncl, alpha, /*beta=*/nullptr, /*N=*/1, C, L, x_ncl);
}

inline bt::Tensor elu(const bt::Tensor& x) {
    bt::Tensor y;
    bt::elu_forward(x, y);
    return y;
}

// Exact (erf) GELU in place — HuBERT's `gelu`.
inline void gelu_inplace(bt::Tensor& x) {
    bt::gelu_exact_forward(x, x);
}

// LayerNorm (weight + bias) over a (T, D) sequence.
inline bt::Tensor lnorm(const bt::Tensor& x, const bt::Tensor& w, const bt::Tensor& b, float eps) {
    bt::Tensor y;
    bt::layernorm_forward_inference_batched(x, w, b, y, eps);
    return y;
}

// ── windowed-sinc resampler (torchaudio.functional.resample) ─────────────────
// The reference resamples 24 -> 16 kHz with torchaudio's sinc_interp_hann
// kernel (lowpass_filter_width 6, rolloff 0.99), and HuBERT is sensitive to
// the difference between that and a linear interpolation (the RVQ codes move
// by ~1-4% per codebook). torchaudio's resample *is* a strided conv1d — one
// output channel per output phase, stride = the reduced input rate — so it is
// composed here from pad1d + conv1d with the kernel built host-side once.
struct SincResampler {
    int orig = 1, ratio_new = 1;   // rates reduced by their gcd
    int width = 0, K = 0;          // half-width, kernel length 2*width + orig
    brotensor::Tensor kernel;      // (ratio_new, 1*K) OIL, on the device
    bool identity() const { return orig == ratio_new; }
};

inline SincResampler make_sinc_resampler(int orig_hz, int new_hz, bt::Device dev) {
    SincResampler r;
    if (orig_hz <= 0 || new_hz <= 0) fail("resample: rates must be positive");
    int g = orig_hz, b = new_hz;
    while (b) { const int t = g % b; g = b; b = t; }
    r.orig = orig_hz / g;
    r.ratio_new = new_hz / g;
    if (r.identity()) return r;
    const int    lowpass_width = 6;
    const double rolloff       = 0.99;
    const double base = static_cast<double>(std::min(r.orig, r.ratio_new)) * rolloff;
    r.width = static_cast<int>(std::ceil(lowpass_width * r.orig / base));
    r.K = 2 * r.width + r.orig;
    const double scale = base / r.orig;
    std::vector<float> h(static_cast<std::size_t>(r.ratio_new) * r.K);
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < r.ratio_new; ++i)
        for (int k = 0; k < r.K; ++k) {
            const double idx = static_cast<double>(k - r.width) / r.orig;
            double t = (-static_cast<double>(i) / r.ratio_new + idx) * base;
            t = std::max(-static_cast<double>(lowpass_width), std::min(static_cast<double>(lowpass_width), t));
            const double win = std::cos(t * pi / lowpass_width / 2.0);
            t *= pi;
            const double sinc = (t == 0.0) ? 1.0 : std::sin(t) / t;
            h[static_cast<std::size_t>(i) * r.K + k] = static_cast<float>(sinc * win * win * scale);
        }
    r.kernel = bt::Tensor::from_host_on(dev, h.data(), r.ratio_new, r.K);
    return r;
}

// x: (1, n) on the resampler's device -> (1, ceil(n * new / orig)); *n_out
// receives the length. Identity rates return x unchanged.
inline bt::Tensor sinc_resample(const SincResampler& r, const bt::Tensor& x, int n, int* n_out) {
    if (r.identity()) { if (n_out) *n_out = n; return x; }
    const int target = static_cast<int>(
        (static_cast<long long>(r.ratio_new) * n + r.orig - 1) / r.orig);   // ceil
    bt::Tensor padded;
    bt::pad1d_forward(x, /*N=*/1, /*C=*/1, n, r.width, r.width + r.orig, /*mode=*/0, padded);
    const int Lp = n + 2 * r.width + r.orig;
    bt::Tensor phases;   // (1, ratio_new * L'), L' = (Lp - K)/orig + 1
    bt::conv1d(padded, r.kernel, /*bias=*/nullptr, /*N=*/1, /*C_in=*/1, Lp, r.ratio_new, r.K,
               /*stride=*/r.orig, /*padding=*/0, /*dilation=*/1, /*groups=*/1, phases);
    const int Lq = (Lp - r.K) / r.orig + 1;
    bt::Tensor inter = ncl_to_seq(phases, r.ratio_new, Lq);   // (Lq, ratio_new): phase-interleaved
    if (Lq * r.ratio_new < target) fail("resample: internal length mismatch");
    bt::Tensor y = bt::Tensor::zeros_on(x.device, 1, target, bt::Dtype::FP32);
    bt::copy_d2d(inter, 0, y, 0, target);
    if (n_out) *n_out = target;
    return y;
}

// Download an (n,1) INT32 device tensor into host ints.
inline void download_idx(const bt::Tensor& idx, int n, std::int32_t* dst) {
    if (idx.device == bt::Device::CPU) {
        const auto* p = static_cast<const std::int32_t*>(idx.data);
        for (int t = 0; t < n; ++t) dst[t] = p[t];
    } else {
        bt::Tensor host = idx.to(bt::Device::CPU);
        const auto* p = static_cast<const std::int32_t*>(host.data);
        for (int t = 0; t < n; ++t) dst[t] = p[t];
    }
}

}  // namespace hcodec
}  // namespace brosoundml
