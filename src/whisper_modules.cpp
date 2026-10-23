#include "brosoundml/whisper_modules.h"

#include "mel_slaney.h"

#include <brotensor/detail/cpu/thread_pool.h>
#include <brotensor/detail/dispatch.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt  = brotensor;
namespace stf = brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

// Thread-local target device for the upload helper. Each public `load_from`
// in this file sets this from its own `device` parameter and the upload helper
// migrates every uploaded tensor to it.
thread_local bt::Device g_load_device = bt::Device::CPU;

void upload(const stf::File& f, const std::string& key,
            int rows, int cols, bt::Tensor& dst,
            const std::string& where) {
    const stf::TensorView* view = f.find(key);
    if (!view) fail(where, "missing safetensors key '" + key + "'");
    const std::int64_t n = view->numel();
    if (n != static_cast<std::int64_t>(rows) * cols) {
        fail(where, "tensor '" + key + "' has " + std::to_string(n) +
                    " elements; expected " +
                    std::to_string(static_cast<std::int64_t>(rows) * cols));
    }

    if (view->dtype == stf::Dtype::F32) {
        stf::upload(*view, rows, cols, dst);
    } else if (view->dtype == stf::Dtype::F16 || view->dtype == stf::Dtype::BF16) {
        const bool isF16 = (view->dtype == stf::Dtype::F16);
        const auto* src = reinterpret_cast<const std::uint16_t*>(view->data);
        std::vector<float> tmp(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
            tmp[static_cast<std::size_t>(i)] =
                isF16 ? bt::fp16_bits_to_fp32(src[i]) : bt::bf16_bits_to_fp32(src[i]);
        dst = bt::Tensor::empty_on(bt::Device::CPU, rows, cols, bt::Dtype::FP32);
        std::memcpy(dst.data, tmp.data(), tmp.size() * sizeof(float));
    } else {
        fail(where, "tensor '" + key + "' has unsupported dtype " +
                    std::to_string(static_cast<int>(view->dtype)) +
                    " (need F32, F16, or BF16)");
    }
    if (dst.device != g_load_device) {
        dst = dst.to(g_load_device);
    }
}

void load_linear(const stf::File& f, const std::string& prefix,
                 int out_features, int in_features,
                 Linear& lin, const std::string& where) {
    lin.in_features  = in_features;
    lin.out_features = out_features;
    upload(f, prefix + ".weight", out_features, in_features, lin.W, where);
    upload(f, prefix + ".bias",   out_features, 1,           lin.b, where);
}

void load_layernorm(const stf::File& f, const std::string& prefix,
                    int features, float eps,
                    LayerNorm& ln, const std::string& where) {
    ln.features = features;
    ln.eps      = eps;
    upload(f, prefix + ".weight", features, 1, ln.gamma, where);
    upload(f, prefix + ".bias",   features, 1, ln.beta,  where);
}

}  // namespace

// ─── LogMel ────────────────────────────────────────────────────────────────

void LogMel::build(int n_mels, bt::Device dev) {
    const std::string where = "LogMel::build";
    if (n_mels <= 0) fail(where, "num_mel_bins must be positive");
    device       = dev;
    num_mel_bins = n_mels;
    n_fft        = 400;
    hop_length   = 160;
    win_length   = 400;
    n_frames     = 3000;
    sample_rate  = 16000;

    const int n_bins = n_fft / 2 + 1;   // 201

    std::vector<float> fb = melslaney::build_filterbank(num_mel_bins, n_fft, sample_rate);
    mel_filters = bt::Tensor::zeros_on(bt::Device::CPU, num_mel_bins, n_bins,
                                       bt::Dtype::FP32);
    std::memcpy(mel_filters.host_f32_mut(), fb.data(),
                fb.size() * sizeof(float));

    // Pre-transpose mel_filters to (n_bins, num_mel_bins) once at build time so
    // forward() does not re-transpose the filterbank on every audio clip.
    mel_filters_T = bt::Tensor::zeros_on(bt::Device::CPU, n_bins, num_mel_bins,
                                         bt::Dtype::FP32);
    {
        float* dst = mel_filters_T.host_f32_mut();
        for (int m = 0; m < num_mel_bins; ++m) {
            for (int k = 0; k < n_bins; ++k) {
                dst[k * num_mel_bins + m] = fb[static_cast<std::size_t>(m) * n_bins + k];
            }
        }
    }

    std::vector<float> hw = melslaney::build_hann_window(win_length);
    hann_window = bt::Tensor::zeros_on(bt::Device::CPU, 1, win_length, bt::Dtype::FP32);
    std::memcpy(hann_window.host_f32_mut(), hw.data(), hw.size() * sizeof(float));
}

void LogMel::forward(const AudioBuffer& audio, bt::Tensor& out) const {
    const std::string where = "LogMel::forward";
    if (mel_filters.rows == 0) fail(where, "LogMel::build() not called");
    if (audio.sample_rate != sample_rate) {
        fail(where, "audio sample_rate " + std::to_string(audio.sample_rate) +
                    " != Whisper's required " + std::to_string(sample_rate));
    }

    const int signal_len = sample_rate * 30;  // 480000
    const int n_bins     = n_fft / 2 + 1;     // 201

    // ─── 1. Pad / truncate to 30 s ─────────────────────────────────────────
    bt::Tensor signal = bt::Tensor::zeros_on(bt::Device::CPU, 1, signal_len,
                                             bt::Dtype::FP32);
    const std::size_t n_in = std::min<std::size_t>(audio.samples.size(),
                                                   static_cast<std::size_t>(signal_len));
    if (n_in > 0) {
        std::memcpy(signal.host_f32_mut(), audio.samples.data(),
                    n_in * sizeof(float));
    }

    // ─── 2. STFT (center=True, reflect-pad) ────────────────────────────────
    bt::Tensor spec;
    bt::stft(signal, hann_window, /*N=*/1, n_fft, hop_length, win_length,
             /*center=*/true, /*normalized=*/false, spec);
    const int frames = spec.rows;
    if (frames < n_frames + 1) {
        fail(where, "stft returned " + std::to_string(frames) +
                    " frames; expected at least " + std::to_string(n_frames + 1));
    }

    // ─── 3+4. Power spectrum (drop the last frame), parallelized across frames
    bt::Tensor power = bt::Tensor::zeros_on(bt::Device::CPU, n_frames, n_bins,
                                            bt::Dtype::FP32);
    {
        const float* sd = spec.host_f32();
        float* pd       = power.host_f32_mut();
        constexpr int kChunk = 128;
        const std::size_t n_chunks = static_cast<std::size_t>((n_frames + kChunk - 1) / kChunk);
        bt::detail::cpu::parallel_for(n_chunks, [&](std::size_t ci) {
            const int f0 = static_cast<int>(ci) * kChunk;
            const int f1 = std::min(n_frames, f0 + kChunk);
            for (int f = f0; f < f1; ++f) {
                const float* src = sd + static_cast<std::size_t>(f) * 2 * n_bins;
                float*       dst = pd + static_cast<std::size_t>(f) * n_bins;
                for (int k = 0; k < n_bins; ++k) {
                    const float re = src[2 * k + 0];
                    const float im = src[2 * k + 1];
                    dst[k] = re * re + im * im;
                }
            }
        });
    }

    // ─── 5. Mel projection ────────────────────────────────────────────────
    bt::Tensor mel_T;
    if (mel_filters_T.rows == n_bins && mel_filters_T.cols == num_mel_bins) {
        bt::matmul(power, mel_filters_T, mel_T);  // (n_frames, n_mels)
    } else {
        bt::Tensor fb_T = bt::Tensor::zeros_on(bt::Device::CPU, n_bins, num_mel_bins,
                                               bt::Dtype::FP32);
        const float* src = mel_filters.host_f32();
        float* dst       = fb_T.host_f32_mut();
        for (int m = 0; m < num_mel_bins; ++m) {
            for (int k = 0; k < n_bins; ++k) {
                dst[k * num_mel_bins + m] = src[m * n_bins + k];
            }
        }
        bt::matmul(power, fb_T, mel_T);
    }

    // ─── 6. log10 with clamp, then dynamic-range clamp ────────────────────
    out = bt::Tensor::zeros_on(bt::Device::CPU, num_mel_bins, n_frames,
                               bt::Dtype::FP32);
    float* od = out.host_f32_mut();
    const float* md = mel_T.host_f32();
    const int M = num_mel_bins;
    const int F = n_frames;
    constexpr int kFrameChunk = 128;
    const std::size_t n_frame_chunks = static_cast<std::size_t>((F + kFrameChunk - 1) / kFrameChunk);
    bt::detail::cpu::parallel_for(n_frame_chunks, [&](std::size_t ci) {
        const int f0 = static_cast<int>(ci) * kFrameChunk;
        const int f1 = std::min(F, f0 + kFrameChunk);
        for (int f = f0; f < f1; ++f) {
            for (int m = 0; m < M; ++m) {
                float v = md[static_cast<std::size_t>(f) * M + m];
                if (v < 1e-10f) v = 1e-10f;
                od[static_cast<std::size_t>(m) * F + f] = std::log10(v);
            }
        }
    });
    const std::size_t total = static_cast<std::size_t>(M) * F;
    float global_max = -1e30f;
    for (std::size_t i = 0; i < total; ++i) {
        if (od[i] > global_max) global_max = od[i];
    }
    const float floor_v = global_max - 8.0f;
    constexpr std::size_t kElemChunk = 4096;
    const std::size_t n_elem_chunks = (total + kElemChunk - 1) / kElemChunk;
    bt::detail::cpu::parallel_for(n_elem_chunks, [&](std::size_t ci) {
        const std::size_t i0 = ci * kElemChunk;
        const std::size_t i1 = std::min(total, i0 + kElemChunk);
        for (std::size_t i = i0; i < i1; ++i) {
            float v = od[i];
            if (v < floor_v) v = floor_v;
            od[i] = (v + 4.0f) * 0.25f;
        }
    });

    if (device != bt::Device::CPU) {
        out = out.to(device);
    }
}

// ─── WhisperEncoderLayer ───────────────────────────────────────────────────

void WhisperEncoderLayer::load_from(const stf::File& f,
                                    const std::string& prefix,
                                    int dm, int ffn, int nh,
                                    bt::Device device) {
    const std::string where = "WhisperEncoderLayer::load_from";
    const bt::Device prev_dev = g_load_device;
    g_load_device = device;
    struct DevGuard { bt::Device& slot; bt::Device prev;
        ~DevGuard() { slot = prev; } } _guard{g_load_device, prev_dev};
    d_model   = dm;
    ffn_dim   = ffn;
    num_heads = nh;

    load_layernorm(f, prefix + "self_attn_layer_norm", dm, 1e-5f,
                   self_attn_layer_norm, where);

    self_attn.num_heads = nh;
    self_attn.embed_dim = dm;
    upload(f, prefix + "self_attn.q_proj.weight",   dm, dm, self_attn.Wq, where);
    upload(f, prefix + "self_attn.q_proj.bias",     dm, 1,  self_attn.bq, where);
    upload(f, prefix + "self_attn.k_proj.weight",   dm, dm, self_attn.Wk, where);
    self_attn.bk = bt::Tensor::zeros_on(device, dm, 1, bt::Dtype::FP32);
    upload(f, prefix + "self_attn.v_proj.weight",   dm, dm, self_attn.Wv, where);
    upload(f, prefix + "self_attn.v_proj.bias",     dm, 1,  self_attn.bv, where);
    upload(f, prefix + "self_attn.out_proj.weight", dm, dm, self_attn.Wo, where);
    upload(f, prefix + "self_attn.out_proj.bias",   dm, 1,  self_attn.bo, where);

    load_layernorm(f, prefix + "final_layer_norm", dm, 1e-5f,
                   final_layer_norm, where);

    load_linear(f, prefix + "fc1", ffn, dm, fc1, where);
    load_linear(f, prefix + "fc2", dm, ffn, fc2, where);
}

void WhisperEncoderLayer::forward(const bt::Tensor& X, bt::Tensor& Y) const {
    const std::string where = "WhisperEncoderLayer::forward";
    if (X.dtype != bt::Dtype::FP32) {
        fail(where, "currently FP32 only");
    }
    if (X.cols != d_model) {
        fail(where, "X cols=" + std::to_string(X.cols) +
                    " != d_model=" + std::to_string(d_model));
    }
    bt::Tensor x_ln = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    self_attn_layer_norm.forward(X, x_ln);                 // (L, D)
    bt::Tensor attn_out = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    self_attn.forward(x_ln, /*d_mask=*/nullptr, attn_out); // (L, D)
    bt::add_inplace(attn_out, X);                           // residual

    bt::Tensor ffn_ln = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    final_layer_norm.forward(attn_out, ffn_ln);            // (L, D)
    bt::Tensor h1 = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    fc1.forward_batched(ffn_ln, h1);                       // (L, ffn_dim)
    bt::Tensor h1_act = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    bt::gelu_forward(h1, h1_act);
    bt::Tensor h2 = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    fc2.forward_batched(h1_act, h2);                       // (L, D)
    bt::add_inplace(h2, attn_out);                          // residual

    Y = std::move(h2);
}

// ─── WhisperEncoder ────────────────────────────────────────────────────────

void WhisperEncoder::load_from(const stf::File& f,
                               int n_mel, int dm, int max_src,
                               int n_layer, int ffn, int n_head,
                               bt::Device device) {
    const std::string where = "WhisperEncoder::load_from";
    const bt::Device prev_dev = g_load_device;
    g_load_device = device;
    struct DevGuard { bt::Device& slot; bt::Device prev;
        ~DevGuard() { slot = prev; } } _guard{g_load_device, prev_dev};
    num_mel_bins             = n_mel;
    d_model                  = dm;
    max_source_positions     = max_src;
    encoder_layers           = n_layer;
    encoder_ffn_dim          = ffn;
    encoder_attention_heads  = n_head;

    const std::string p = "model.encoder.";

    conv1.in_channels  = n_mel;
    conv1.out_channels = dm;
    conv1.kernel_size  = 3;
    conv1.stride       = 1;
    conv1.padding      = 1;
    conv1.dilation     = 1;
    conv1.groups       = 1;
    upload(f, p + "conv1.weight", dm, n_mel * 3, conv1.W, where);
    upload(f, p + "conv1.bias",   dm, 1,         conv1.b, where);

    conv2.in_channels  = dm;
    conv2.out_channels = dm;
    conv2.kernel_size  = 3;
    conv2.stride       = 2;
    conv2.padding      = 1;
    conv2.dilation     = 1;
    conv2.groups       = 1;
    upload(f, p + "conv2.weight", dm, dm * 3, conv2.W, where);
    upload(f, p + "conv2.bias",   dm, 1,      conv2.b, where);

    upload(f, p + "embed_positions.weight", max_src, dm, embed_positions, where);

    layers.clear();
    layers.resize(static_cast<std::size_t>(n_layer));
    for (int i = 0; i < n_layer; ++i) {
        const std::string lp = p + "layers." + std::to_string(i) + ".";
        layers[static_cast<std::size_t>(i)].load_from(f, lp, dm, ffn, n_head, device);
    }

    load_layernorm(f, p + "layer_norm", dm, 1e-5f, layer_norm, where);
}

void WhisperEncoder::forward(const bt::Tensor& mel, bt::Tensor& hidden) const {
    const std::string where = "WhisperEncoder::forward";
    if (mel.dtype != bt::Dtype::FP32) {
        fail(where, "currently FP32 only");
    }
    int n_mel, n_frames;
    bt::Tensor mel_ncl;
    if (mel.rows == num_mel_bins && mel.cols > 0) {
        n_mel    = num_mel_bins;
        n_frames = mel.cols;
        mel_ncl = bt::Tensor::view(mel.device, mel.data,
                                   1, n_mel * n_frames, bt::Dtype::FP32);
    } else if (mel.rows == 1 && mel.cols == num_mel_bins * /*n_frames=*/3000) {
        n_mel    = num_mel_bins;
        n_frames = 3000;
        mel_ncl = bt::Tensor::view(mel.device, mel.data, 1,
                                   n_mel * n_frames, bt::Dtype::FP32);
    } else {
        fail(where, "mel input must be (num_mel_bins, n_frames) or (1, n_mel*n_frames)");
    }

    bt::Tensor x1 = bt::Tensor::empty_on(mel.device, 0, 0, bt::Dtype::FP32);
    conv1.forward(mel_ncl, /*N=*/1, /*L=*/n_frames, x1);    // (1, d_model*n_frames)
    bt::Tensor x1g = bt::Tensor::empty_on(mel.device, 0, 0, bt::Dtype::FP32);
    bt::gelu_forward(x1, x1g);

    bt::Tensor x2 = bt::Tensor::empty_on(mel.device, 0, 0, bt::Dtype::FP32);
    conv2.forward(x1g, /*N=*/1, /*L=*/n_frames, x2);
    bt::Tensor x2g = bt::Tensor::empty_on(mel.device, 0, 0, bt::Dtype::FP32);
    bt::gelu_forward(x2, x2g);
    const int L = n_frames / 2;
    if (L != max_source_positions) {
        fail(where, "post-conv length " + std::to_string(L) +
                    " != max_source_positions=" + std::to_string(max_source_positions));
    }

    bt::Tensor x = bt::Tensor::empty_on(mel.device, 0, 0, bt::Dtype::FP32);
    bt::nchw_to_sequence(x2g, /*N=*/1, /*C=*/d_model, /*H=*/1, /*W=*/L, x);

    bt::add_inplace(x, embed_positions);

    bt::Tensor y = bt::Tensor::empty_on(mel.device, 0, 0, bt::Dtype::FP32);
    for (const auto& layer : layers) {
        layer.forward(x, y);
        x = std::move(y);
    }

    layer_norm.forward(x, hidden);
}

}  // namespace brosoundml
