#include "brosoundml/whisper_modules.h"

#include "mel_slaney.h"

#include <brotensor/ops.h>
#include <brotensor/safetensors.h>
#include <brotensor/detail/dispatch.h>

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

// Upload a host int32 index buffer to a (n, 1) INT32 tensor on `dev`.
// brotensor lacks `Tensor::from_host_int32_on`, so we allocate the device
// tensor and reach through `detail::alloc_for(dev).memcpy_h2d` to populate it
// — the public memcpy_h2d hook on the backend vtable. CUDA's
// `embedding_lookup_forward` reads `d_idx` as a DEVICE pointer, so calls that
// previously passed `host_vector.data()` straight through crashed CUDA with
// an illegal memory access.
bt::Tensor upload_int32_idx(bt::Device dev, const std::int32_t* host_idx, int n) {
    bt::Tensor t = bt::Tensor::empty_on(dev, n, 1, bt::Dtype::INT32);
    if (n == 0) return t;
    if (dev == bt::Device::CPU) {
        std::memcpy(t.data, host_idx, static_cast<std::size_t>(n) * sizeof(std::int32_t));
    } else {
        bt::detail::alloc_for(dev).memcpy_h2d(
            t.data, host_idx,
            static_cast<std::size_t>(n) * sizeof(std::int32_t));
    }
    return t;
}

// Thread-local target device for the upload helper. Each public `load_from`
// in this file sets this from its own `device` parameter and the upload helper
// migrates every uploaded tensor to it. safetensors::upload lands the tensor
// on CPU; without this migration step, weights stay on CPU even when the rest
// of the module runs on CUDA, and brotensor's dispatcher refuses the mixed
// call. (This is the upload-time half of the systemic CUDA-port bug.)
thread_local bt::Device g_load_device = bt::Device::CPU;

// Upload a tensor named `key` from `f` into `dst` reshaped to (rows, cols).
// Mirrors the helper in kokoro_modules.cpp; on top of the safetensors upload
// (CPU-resident), this migrates the destination to `g_load_device` so the
// rest of the module sees device-matched weights.
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

    // brosoundml's Whisper forward is F32 throughout (CPU and CUDA). Upstream
    // openai/whisper checkpoints ship their weights as F16 (and some other
    // sources as BF16), so widen those host-side to F32 before uploading rather
    // than rejecting them — this lets an unmodified upstream model.safetensors
    // load directly, with no offline FP32 re-cast step.
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
    // safetensors::upload uses Tensor::from_host which lands on
    // brotensor::default_device() — that's CUDA once init() registers it, not
    // CPU. Migrate unconditionally to g_load_device so weights end up exactly
    // where this layer's forward expects them.
    if (dst.device != g_load_device) {
        dst = dst.to(g_load_device);
    }
}

// Slaney mel filterbank — shared builder in mel_slaney.h (also used by the
// Qwen3-ASR encoder front-end).

}  // namespace

// ─── LogMel ────────────────────────────────────────────────────────────────

void LogMel::build(int n_mels, bt::Device dev) {
    const std::string where = "LogMel::build";
    if (n_mels <= 0) fail(where, "num_mel_bins must be positive");
    // Store the target device for forward()'s output. mel_filters and
    // hann_window themselves stay CPU-resident — the inner STFT/projection
    // pipeline runs on the host, see LogMel::forward.
    device       = dev;
    num_mel_bins = n_mels;
    n_fft        = 400;
    hop_length   = 160;
    win_length   = 400;
    n_frames     = 3000;
    sample_rate  = 16000;

    const int n_bins = n_fft / 2 + 1;   // 201

    // Mel filterbank — built on CPU; the forward path's matmul consumes it
    // host-side, so we don't pay an upload for tables we only read from the
    // host. Same logic for hann_window below.
    std::vector<float> fb = melslaney::build_filterbank(num_mel_bins, n_fft, sample_rate);
    mel_filters = bt::Tensor::zeros_on(bt::Device::CPU, num_mel_bins, n_bins,
                                       bt::Dtype::FP32);
    std::memcpy(mel_filters.host_f32_mut(), fb.data(),
                fb.size() * sizeof(float));

    // Periodic Hann analysis window — whisper/audio.py passes
    // torch.hann_window(N_FFT), which defaults to periodic=True. Shared
    // builder in mel_slaney.h.
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
    // mel_filters/hann_window are intentionally CPU-resident — the host-side
    // pipeline below consumes them directly; the final result is uploaded to
    // `device` at the end of this function.

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
    // Remaining samples already zero.

    // ─── 2. STFT (center=True, reflect-pad) ────────────────────────────────
    // Output shape: (N*frames, 2*(n_fft/2+1)) = (frames, 402) with N=1.
    // With center=True and signal_len=480000, frames = 1 + 480000/160 = 3001.
    bt::Tensor spec;
    bt::stft(signal, hann_window, /*N=*/1, n_fft, hop_length, win_length,
             /*center=*/true, /*normalized=*/false, spec);
    const int frames = spec.rows;
    if (frames < n_frames + 1) {
        fail(where, "stft returned " + std::to_string(frames) +
                    " frames; expected at least " + std::to_string(n_frames + 1));
    }

    // ─── 3+4. Power spectrum (drop the last frame) ────────────────────────
    // Compute pow[f, k] = re*re + im*im over (n_frames, n_bins) — cheaper than
    // complex_abs followed by a square, and avoids materialising the magnitude.
    bt::Tensor power = bt::Tensor::zeros_on(bt::Device::CPU, n_frames, n_bins,
                                            bt::Dtype::FP32);
    {
        const float* sd = spec.host_f32();
        float* pd       = power.host_f32_mut();
        for (int f = 0; f < n_frames; ++f) {
            const float* src = sd + static_cast<std::size_t>(f) * 2 * n_bins;
            float*       dst = pd + static_cast<std::size_t>(f) * n_bins;
            for (int k = 0; k < n_bins; ++k) {
                const float re = src[2 * k + 0];
                const float im = src[2 * k + 1];
                dst[k] = re * re + im * im;
            }
        }
    }

    // ─── 5. Mel projection ────────────────────────────────────────────────
    // mel_filters: (n_mels, n_bins). power: (n_frames, n_bins). We want
    // (n_mels, n_frames) = mel_filters @ power.T. brotensor::matmul has no
    // transpose flag, so multiply (n_frames, n_bins) @ (n_bins, n_mels) by
    // transposing the filterbank inline; the result is (n_frames, n_mels)
    // which we transpose to (n_mels, n_frames) when filling `out`.
    bt::Tensor mel_T;
    {
        // Build filterbank transpose: (n_bins, n_mels).
        bt::Tensor fb_T = bt::Tensor::zeros_on(bt::Device::CPU, n_bins, num_mel_bins,
                                               bt::Dtype::FP32);
        const float* src = mel_filters.host_f32();
        float* dst       = fb_T.host_f32_mut();
        for (int m = 0; m < num_mel_bins; ++m) {
            for (int k = 0; k < n_bins; ++k) {
                dst[k * num_mel_bins + m] = src[m * n_bins + k];
            }
        }
        bt::matmul(power, fb_T, mel_T);  // (n_frames, n_mels)
    }

    // ─── 6. log10 with clamp, then dynamic-range clamp ────────────────────
    // log_spec = log10(max(mel, 1e-10)); then clamp to >= max(log_spec) - 8.0;
    // then normalize: (log_spec + 4.0) / 4.0.
    // Allocate `out` fresh on CPU regardless of the caller's previous binding
    // (Tensor::resize preserves device, so a CUDA-resident `out` from a prior
    // call would refuse host_f32_mut). The final upload to `device` at the
    // bottom of this function handles the move when device != CPU.
    out = bt::Tensor::zeros_on(bt::Device::CPU, num_mel_bins, n_frames,
                               bt::Dtype::FP32);
    float* od = out.host_f32_mut();
    const float* md = mel_T.host_f32();
    float global_max = -1e30f;
    for (int f = 0; f < n_frames; ++f) {
        for (int m = 0; m < num_mel_bins; ++m) {
            float v = md[static_cast<std::size_t>(f) * num_mel_bins + m];
            if (v < 1e-10f) v = 1e-10f;
            const float lv = std::log10(v);
            od[static_cast<std::size_t>(m) * n_frames + f] = lv;
            if (lv > global_max) global_max = lv;
        }
    }
    const float floor_v = global_max - 8.0f;
    const std::size_t total = static_cast<std::size_t>(num_mel_bins) * n_frames;
    for (std::size_t i = 0; i < total; ++i) {
        float v = od[i];
        if (v < floor_v) v = floor_v;
        od[i] = (v + 4.0f) / 4.0f;
    }

    // Upload the host-built mel feature to the model device. This is the one
    // boundary where LogMel stays CPU-resident by design: STFT + filterbank
    // + log10 are one-shot per audio clip and the input arrives as a host
    // vector, so we run the pipeline on the host and ship the (small) result
    // to the encoder's device.
    if (device != bt::Device::CPU) {
        out = out.to(device);
    }
}

// ─── WhisperEncoderLayer ───────────────────────────────────────────────────

namespace {

// Load a Linear (out, in) + (out, 1) bias pair under `<prefix>.{weight,bias}`.
void load_linear(const stf::File& f, const std::string& prefix,
                 int out_features, int in_features,
                 Linear& lin, const std::string& where) {
    lin.in_features  = in_features;
    lin.out_features = out_features;
    upload(f, prefix + ".weight", out_features, in_features, lin.W, where);
    upload(f, prefix + ".bias",   out_features, 1,           lin.b, where);
}

// Load a LayerNorm gamma / beta pair under `<prefix>.{weight,bias}`.
void load_layernorm(const stf::File& f, const std::string& prefix,
                    int features, float eps,
                    LayerNorm& ln, const std::string& where) {
    ln.features = features;
    ln.eps      = eps;
    upload(f, prefix + ".weight", features, 1, ln.gamma, where);
    upload(f, prefix + ".bias",   features, 1, ln.beta,  where);
}

}  // namespace

void WhisperEncoderLayer::load_from(const stf::File& f,
                                    const std::string& prefix,
                                    int dm, int ffn, int nh,
                                    bt::Device device) {
    const std::string where = "WhisperEncoderLayer::load_from";
    // Route every `upload(...)` below through `device` (host-side load lands
    // on CPU; we want every weight on the model's device).
    const bt::Device prev_dev = g_load_device;
    g_load_device = device;
    struct DevGuard { bt::Device& slot; bt::Device prev;
        ~DevGuard() { slot = prev; } } _guard{g_load_device, prev_dev};
    d_model   = dm;
    ffn_dim   = ffn;
    num_heads = nh;

    load_layernorm(f, prefix + "self_attn_layer_norm", dm, 1e-5f,
                   self_attn_layer_norm, where);

    // MHA: Whisper's K-projection has no bias on disk. To keep the shared MHA
    // API stable, we allocate bk as (D, 1) zeros and load Wq/Wk/Wv/Wo + their
    // biases (sans bk) directly.
    self_attn.num_heads = nh;
    self_attn.embed_dim = dm;
    upload(f, prefix + "self_attn.q_proj.weight",   dm, dm, self_attn.Wq, where);
    upload(f, prefix + "self_attn.q_proj.bias",     dm, 1,  self_attn.bq, where);
    upload(f, prefix + "self_attn.k_proj.weight",   dm, dm, self_attn.Wk, where);
    // Whisper's K-projection has no bias on disk; allocate a zero (D,1) on
    // the same device the rest of the layer's weights load to so the MHA
    // kernel's bias-add stays on that device.
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
    // ─── Self-attention residual ───────────────────────────────────────────
    // Pre-allocate every op-out tensor on X.device; default-constructed
    // bt::Tensor lives on CPU and brotensor's CUDA dispatch refuses mixed-
    // device calls. Tensor::resize preserves the device field, so a (0, 0)
    // pre-alloc with the right device is enough for ops that re-size their
    // out param.
    bt::Tensor x_ln = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    self_attn_layer_norm.forward(X, x_ln);                 // (L, D)
    bt::Tensor attn_out = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    self_attn.forward(x_ln, /*d_mask=*/nullptr, attn_out); // (L, D)
    bt::add_inplace(attn_out, X);                           // residual

    // ─── FFN residual ──────────────────────────────────────────────────────
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

    // conv1: (d_model, num_mel_bins, 3)  stride=1, padding=1.
    conv1.in_channels  = n_mel;
    conv1.out_channels = dm;
    conv1.kernel_size  = 3;
    conv1.stride       = 1;
    conv1.padding      = 1;
    conv1.dilation     = 1;
    conv1.groups       = 1;
    upload(f, p + "conv1.weight", dm, n_mel * 3, conv1.W, where);
    upload(f, p + "conv1.bias",   dm, 1,         conv1.b, where);

    // conv2: (d_model, d_model, 3)  stride=2, padding=1.
    conv2.in_channels  = dm;
    conv2.out_channels = dm;
    conv2.kernel_size  = 3;
    conv2.stride       = 2;
    conv2.padding      = 1;
    conv2.dilation     = 1;
    conv2.groups       = 1;
    upload(f, p + "conv2.weight", dm, dm * 3, conv2.W, where);
    upload(f, p + "conv2.bias",   dm, 1,      conv2.b, where);

    // Positional embedding: (max_source_positions, d_model). Whisper bakes
    // the sinusoidal embedding into the checkpoint as a regular weight.
    upload(f, p + "embed_positions.weight", max_src, dm, embed_positions, where);

    // Stack of layers; populated once at load time so the per-layer forward
    // loop never copy-constructs a module.
    layers.clear();
    layers.resize(static_cast<std::size_t>(n_layer));
    for (int i = 0; i < n_layer; ++i) {
        const std::string lp = p + "layers." + std::to_string(i) + ".";
        layers[static_cast<std::size_t>(i)].load_from(f, lp, dm, ffn, n_head, device);
    }

    // Final encoder LayerNorm.
    load_layernorm(f, p + "layer_norm", dm, 1e-5f, layer_norm, where);
}

void WhisperEncoder::forward(const bt::Tensor& mel, bt::Tensor& hidden) const {
    const std::string where = "WhisperEncoder::forward";
    if (mel.dtype != bt::Dtype::FP32) {
        fail(where, "currently FP32 only");
    }
    // Accept either (num_mel_bins, n_frames) "flat 2D" or NCL (1, n_mel*n_frames).
    int n_mel, n_frames;
    bt::Tensor mel_ncl;
    if (mel.rows == num_mel_bins && mel.cols > 0) {
        n_mel    = num_mel_bins;
        n_frames = mel.cols;
        // Reshape view: (num_mel_bins, n_frames) is already row-major NCL with
        // N=1 — the memory layout matches (1, num_mel_bins * n_frames).
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

    // ─── conv1 + GELU ──────────────────────────────────────────────────────
    // Pre-allocate every op-out tensor on mel.device; default-constructed
    // bt::Tensor lives on CPU, brotensor's CUDA dispatch refuses mixed-device
    // calls, and Tensor::resize preserves the device field.
    bt::Tensor x1 = bt::Tensor::empty_on(mel.device, 0, 0, bt::Dtype::FP32);
    conv1.forward(mel_ncl, /*N=*/1, /*L=*/n_frames, x1);    // (1, d_model*n_frames)
    bt::Tensor x1g = bt::Tensor::empty_on(mel.device, 0, 0, bt::Dtype::FP32);
    bt::gelu_forward(x1, x1g);

    // ─── conv2 (stride=2) + GELU ───────────────────────────────────────────
    bt::Tensor x2 = bt::Tensor::empty_on(mel.device, 0, 0, bt::Dtype::FP32);
    conv2.forward(x1g, /*N=*/1, /*L=*/n_frames, x2);
    bt::Tensor x2g = bt::Tensor::empty_on(mel.device, 0, 0, bt::Dtype::FP32);
    bt::gelu_forward(x2, x2g);
    // Output length after stride-2 conv with padding=1, kernel=3:
    //   L_out = (L + 2*1 - 1*(3-1) - 1)/2 + 1 = L/2.
    const int L = n_frames / 2;
    if (L != max_source_positions) {
        fail(where, "post-conv length " + std::to_string(L) +
                    " != max_source_positions=" + std::to_string(max_source_positions));
    }

    // ─── Transpose NCL -> NLC: (1, D*L) -> (L, D) ──────────────────────────
    // brotensor::nchw_to_sequence with N=1, C=d_model, H=1, W=L maps NCL ->
    // (N*H*W, C) = (L, d_model) — exactly the transpose we want, device-
    // dispatched on x2g's device.
    bt::Tensor x = bt::Tensor::empty_on(mel.device, 0, 0, bt::Dtype::FP32);
    bt::nchw_to_sequence(x2g, /*N=*/1, /*C=*/d_model, /*H=*/1, /*W=*/L, x);

    // ─── Add positional embedding ──────────────────────────────────────────
    // x and embed_positions are both (L, d_model) FP32 on the same device.
    bt::add_inplace(x, embed_positions);

    // ─── Pre-LN Transformer stack ──────────────────────────────────────────
    bt::Tensor y = bt::Tensor::empty_on(mel.device, 0, 0, bt::Dtype::FP32);
    for (const auto& layer : layers) {
        layer.forward(x, y);
        x = std::move(y);
    }

    // ─── Final encoder LayerNorm ───────────────────────────────────────────
    layer_norm.forward(x, hidden);
}

// ─── WhisperKVCache ────────────────────────────────────────────────────────

void WhisperKVCache::allocate(int decoder_layers, int d_model,
                              int max_target_positions, int max_source_positions,
                              bt::Device device) {
    const std::string where = "WhisperKVCache::allocate";
    if (decoder_layers <= 0) fail(where, "decoder_layers must be positive");
    if (d_model <= 0)        fail(where, "d_model must be positive");
    if (max_target_positions <= 0) fail(where, "max_target_positions must be positive");
    if (max_source_positions <= 0) fail(where, "max_source_positions must be positive");

    layers.clear();
    layers.resize(static_cast<std::size_t>(decoder_layers));
    for (auto& l : layers) {
        l.self_k = bt::Tensor::zeros_on(device, max_target_positions,
                                        d_model, bt::Dtype::FP32);
        l.self_v = bt::Tensor::zeros_on(device, max_target_positions,
                                        d_model, bt::Dtype::FP32);
        l.cross_k = bt::Tensor::zeros_on(device, max_source_positions,
                                         d_model, bt::Dtype::FP32);
        l.cross_v = bt::Tensor::zeros_on(device, max_source_positions,
                                         d_model, bt::Dtype::FP32);
        l.self_len = 0;
        l.cross_primed = false;
    }
}

void WhisperKVCache::reset() {
    for (auto& l : layers) {
        l.self_len = 0;
        l.cross_primed = false;
    }
}

// ─── mha_causal_cached_fp32 ────────────────────────────────────────────────

void mha_causal_cached_fp32(const bt::Tensor& X,
                            const bt::Tensor& Wq, const bt::Tensor& bq,
                            const bt::Tensor& Wk, const bt::Tensor& bk,
                            const bt::Tensor& Wv, const bt::Tensor& bv,
                            const bt::Tensor& Wo, const bt::Tensor& bo,
                            int num_heads,
                            WhisperLayerCache& cache,
                            bt::Tensor& out) {
    const std::string where = "mha_causal_cached_fp32";
    if (X.dtype != bt::Dtype::FP32) {
        fail(where, "currently FP32 only");
    }
    const int T = X.rows;
    const int D = X.cols;
    if (D <= 0 || T <= 0) fail(where, "X must be non-empty");
    if (D % num_heads != 0) {
        fail(where, "embed_dim " + std::to_string(D) +
             " not divisible by num_heads " + std::to_string(num_heads));
    }
    if (cache.self_k.rows == 0 || cache.self_k.cols != D ||
        cache.self_v.rows == 0 || cache.self_v.cols != D) {
        fail(where, "cache.self_k/self_v not pre-allocated to (>=T, D)");
    }
    if (cache.self_len + T > cache.self_k.rows) {
        fail(where, "cache overflow: self_len=" + std::to_string(cache.self_len) +
             " + T=" + std::to_string(T) + " > " +
             std::to_string(cache.self_k.rows));
    }
    // The decode driver only ever calls this in two shapes: prefill
    // (self_len == 0, T == prompt_len) where Lq == Lk and causal masking
    // applies, and single-step (T == 1) where the one query attends to every
    // cached key. The mixed case (self_len > 0 && T > 1) would need a custom
    // per-query mask and never occurs — reject it explicitly.
    if (cache.self_len > 0 && T > 1) {
        fail(where, "mixed prefill + step not supported "
             "(self_len=" + std::to_string(cache.self_len) +
             ", T=" + std::to_string(T) + ")");
    }

    // Project Q, K, V for the current T input rows. Pre-allocate on X.device
    // so brotensor's dispatch sees matched devices (the dispatcher refuses
    // mixed-device calls; Tensor::resize preserves the device field).
    bt::Tensor Q    = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    bt::Tensor Knew = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    bt::Tensor Vnew = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    bt::linear_forward_batched(Wq, bq, X, Q);     // (T, D)
    bt::linear_forward_batched(Wk, bk, X, Knew);  // (T, D)
    bt::linear_forward_batched(Wv, bv, X, Vnew);  // (T, D)

    // Append Knew, Vnew into the cache at rows [self_len, self_len + T).
    // copy_d2d handles host or device buffers uniformly.
    bt::copy_d2d(Knew, 0, cache.self_k, cache.self_len * D, T * D);
    bt::copy_d2d(Vnew, 0, cache.self_v, cache.self_len * D, T * D);

    const int Lk_total = cache.self_len + T;

    // Build (Lk_total, D) K/V views over the live prefix of the cache. The
    // cache's underlying buffer is row-major (rows, D), so the first
    // Lk_total rows live contiguously at offset 0.
    bt::Tensor K_live = bt::Tensor::view(cache.self_k.device,
                                         cache.self_k.data,
                                         Lk_total, D, bt::Dtype::FP32);
    bt::Tensor V_live = bt::Tensor::view(cache.self_v.device,
                                         cache.self_v.data,
                                         Lk_total, D, bt::Dtype::FP32);

    // Two regimes, both expressible via flash_attention_forward:
    //  - prefill (self_len==0, T==prompt_len): Lq == Lk_total, use causal=true.
    //  - single-step (T==1): one query attends to every key in the cache; no
    //    mask needed, causal=false.
    const bool causal = (cache.self_len == 0);
    bt::Tensor ctx = bt::Tensor::empty_on(X.device, T, D, bt::Dtype::FP32);
    if (X.device == bt::Device::CPU) {
        bt::flash_attention_forward(Q, K_live, V_live,
                                    /*d_mask=*/nullptr, num_heads, causal, ctx);
    } else {
        // GPU: flash_attention_windowed_forward runs FP32 directly (the
        // qwen_tts_talker pattern), so the cache never round-trips through
        // FP16 — the previous path re-cast the whole live K/V prefix every
        // decode step. window <= 0 is unbounded causal with the Lq queries
        // occupying the LAST Lq positions of the Lk sequence:
        //  - prefill: Lq == Lk_total, query r attends [0, r] (== causal=true);
        //  - single-step: the one query sits at position Lk_total-1 and
        //    attends the whole cache (== the causal=false full-cache case).
        bt::flash_attention_windowed_forward(Q, K_live, V_live,
                                             /*d_mask=*/nullptr, num_heads,
                                             /*window=*/0, ctx);
    }

    // Pre-allocate `out` on X.device too — same reasoning.
    if (out.device != X.device || out.rows != T || out.cols != D ||
        out.dtype != bt::Dtype::FP32) {
        out = bt::Tensor::empty_on(X.device, T, D, bt::Dtype::FP32);
    }
    bt::linear_forward_batched(Wo, bo, ctx, out);

    cache.self_len += T;
}

// ─── cross_attn_cached_fp32 ────────────────────────────────────────────────

void cross_attn_cached_fp32(const bt::Tensor& X,
                            const bt::Tensor& Wq, const bt::Tensor& bq,
                            const bt::Tensor& Wo, const bt::Tensor& bo,
                            int num_heads,
                            const WhisperLayerCache& cache,
                            bt::Tensor& out) {
    const std::string where = "cross_attn_cached_fp32";
    if (X.dtype != bt::Dtype::FP32) {
        fail(where, "currently FP32 only");
    }
    if (!cache.cross_primed) {
        fail(where, "cross-attn cache not primed — call prime_cross first");
    }
    const int T  = X.rows;
    const int D  = X.cols;
    const int Lk = cache.cross_k.rows;
    if (D <= 0 || T <= 0) fail(where, "X must be non-empty");
    if (D % num_heads != 0) {
        fail(where, "embed_dim " + std::to_string(D) +
             " not divisible by num_heads " + std::to_string(num_heads));
    }
    if (cache.cross_k.cols != D || cache.cross_v.cols != D ||
        cache.cross_v.rows != Lk) {
        fail(where, "cross_k/cross_v shape mismatch");
    }

    bt::Tensor Q = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    bt::linear_forward_batched(Wq, bq, X, Q);  // (T, D)

    // Cross-attention: full attention over all Lk encoder positions, no mask,
    // not causal. K/V already live in the cache (pre-projected by
    // prime_cross_cache_fp32). flash_attention_forward handles per-head
    // scaled-dot-product softmax + V-mix in one device-dispatched op.
    bt::Tensor ctx = bt::Tensor::empty_on(X.device, T, D, bt::Dtype::FP32);
    if (X.device == bt::Device::CPU) {
        bt::flash_attention_forward(Q, cache.cross_k, cache.cross_v,
                                    /*d_mask=*/nullptr, num_heads,
                                    /*causal=*/false, ctx);
    } else {
        // GPU: flash_attention_windowed_forward runs FP32 directly, so the
        // (Lk, D) encoder K/V never re-casts to FP16 — the previous path paid
        // that full-cache cast on EVERY decode step. The windowed op is
        // implicitly causal with the Lq queries occupying the last Lq
        // positions, so a single query (Lq == 1) attends all Lk keys — exactly
        // non-causal full attention. T == 1 on every decode step; the
        // prompt-length prefill (once per 30 s window) issues one Lq == 1 call
        // per query row, since a multi-row windowed call would causally clip
        // the earlier rows' view of the encoder.
        if (T == 1) {
            bt::flash_attention_windowed_forward(Q, cache.cross_k, cache.cross_v,
                                                 /*d_mask=*/nullptr, num_heads,
                                                 /*window=*/0, ctx);
        } else {
            bt::Tensor ctx_row = bt::Tensor::empty_on(X.device, 1, D, bt::Dtype::FP32);
            for (int t = 0; t < T; ++t) {
                bt::Tensor q_row = bt::Tensor::view(
                    X.device,
                    static_cast<float*>(Q.data) + static_cast<std::size_t>(t) * D,
                    1, D, bt::Dtype::FP32);
                bt::flash_attention_windowed_forward(q_row, cache.cross_k,
                                                     cache.cross_v,
                                                     /*d_mask=*/nullptr, num_heads,
                                                     /*window=*/0, ctx_row);
                bt::copy_d2d(ctx_row, 0, ctx, t * D, D);
            }
        }
    }

    if (out.device != X.device || out.rows != T || out.cols != D ||
        out.dtype != bt::Dtype::FP32) {
        out = bt::Tensor::empty_on(X.device, T, D, bt::Dtype::FP32);
    }
    bt::linear_forward_batched(Wo, bo, ctx, out);
}

// ─── prime_cross_cache_fp32 ────────────────────────────────────────────────

void prime_cross_cache_fp32(const bt::Tensor& encoder_hidden,
                            const bt::Tensor& Wk, const bt::Tensor& bk,
                            const bt::Tensor& Wv, const bt::Tensor& bv,
                            WhisperLayerCache& cache) {
    const std::string where = "prime_cross_cache_fp32";
    if (encoder_hidden.dtype != bt::Dtype::FP32) {
        fail(where, "currently FP32 only");
    }
    const int L = encoder_hidden.rows;
    const int D = encoder_hidden.cols;
    if (cache.cross_k.rows != L || cache.cross_k.cols != D ||
        cache.cross_v.rows != L || cache.cross_v.cols != D) {
        fail(where, "cross_k/cross_v cache shape (" +
             std::to_string(cache.cross_k.rows) + "," +
             std::to_string(cache.cross_k.cols) + ") != encoder hidden (" +
             std::to_string(L) + "," + std::to_string(D) +
             ") — call WhisperKVCache::allocate with the matching dims first");
    }

    bt::Tensor Kproj = bt::Tensor::empty_on(encoder_hidden.device, 0, 0, bt::Dtype::FP32);
    bt::Tensor Vproj = bt::Tensor::empty_on(encoder_hidden.device, 0, 0, bt::Dtype::FP32);
    bt::linear_forward_batched(Wk, bk, encoder_hidden, Kproj);
    bt::linear_forward_batched(Wv, bv, encoder_hidden, Vproj);

    bt::copy_d2d(Kproj, 0, cache.cross_k, 0, L * D);
    bt::copy_d2d(Vproj, 0, cache.cross_v, 0, L * D);
    cache.cross_primed = true;
}

// ─── WhisperDecoderLayer ───────────────────────────────────────────────────

void WhisperDecoderLayer::load_from(const stf::File& f,
                                    const std::string& prefix,
                                    int dm, int ffn, int nh,
                                    bt::Device device) {
    const std::string where = "WhisperDecoderLayer::load_from";
    const bt::Device prev_dev = g_load_device;
    g_load_device = device;
    struct DevGuard { bt::Device& slot; bt::Device prev;
        ~DevGuard() { slot = prev; } } _guard{g_load_device, prev_dev};
    d_model   = dm;
    ffn_dim   = ffn;
    num_heads = nh;

    // Self-attention LN.
    load_layernorm(f, prefix + "self_attn_layer_norm", dm, 1e-5f,
                   self_attn_layer_norm, where);

    // Self-attention projections. K has NO bias on disk; we zero-fill bk.
    upload(f, prefix + "self_attn.q_proj.weight",   dm, dm, self_Wq, where);
    upload(f, prefix + "self_attn.q_proj.bias",     dm, 1,  self_bq, where);
    upload(f, prefix + "self_attn.k_proj.weight",   dm, dm, self_Wk, where);
    self_bk = bt::Tensor::zeros_on(device, dm, 1, bt::Dtype::FP32);
    upload(f, prefix + "self_attn.v_proj.weight",   dm, dm, self_Wv, where);
    upload(f, prefix + "self_attn.v_proj.bias",     dm, 1,  self_bv, where);
    upload(f, prefix + "self_attn.out_proj.weight", dm, dm, self_Wo, where);
    upload(f, prefix + "self_attn.out_proj.bias",   dm, 1,  self_bo, where);

    // Cross-attention LN.
    load_layernorm(f, prefix + "encoder_attn_layer_norm", dm, 1e-5f,
                   encoder_attn_layer_norm, where);

    // Cross-attention projections. K has NO bias on disk; we zero-fill bk.
    upload(f, prefix + "encoder_attn.q_proj.weight",   dm, dm, cross_Wq, where);
    upload(f, prefix + "encoder_attn.q_proj.bias",     dm, 1,  cross_bq, where);
    upload(f, prefix + "encoder_attn.k_proj.weight",   dm, dm, cross_Wk, where);
    cross_bk = bt::Tensor::zeros_on(device, dm, 1, bt::Dtype::FP32);
    upload(f, prefix + "encoder_attn.v_proj.weight",   dm, dm, cross_Wv, where);
    upload(f, prefix + "encoder_attn.v_proj.bias",     dm, 1,  cross_bv, where);
    upload(f, prefix + "encoder_attn.out_proj.weight", dm, dm, cross_Wo, where);
    upload(f, prefix + "encoder_attn.out_proj.bias",   dm, 1,  cross_bo, where);

    // FFN.
    load_layernorm(f, prefix + "final_layer_norm", dm, 1e-5f,
                   final_layer_norm, where);
    load_linear(f, prefix + "fc1", ffn, dm, fc1, where);
    load_linear(f, prefix + "fc2", dm, ffn, fc2, where);
}

void WhisperDecoderLayer::prime_cross(const bt::Tensor& encoder_hidden,
                                      WhisperLayerCache& cache) const {
    prime_cross_cache_fp32(encoder_hidden,
                           cross_Wk, cross_bk, cross_Wv, cross_bv,
                           cache);
}

void WhisperDecoderLayer::forward(const bt::Tensor& X,
                                  WhisperLayerCache& cache,
                                  bt::Tensor& Y) const {
    const std::string where = "WhisperDecoderLayer::forward";
    if (X.dtype != bt::Dtype::FP32) {
        fail(where, "currently FP32 only");
    }
    if (X.cols != d_model) {
        fail(where, "X cols=" + std::to_string(X.cols) +
                    " != d_model=" + std::to_string(d_model));
    }

    // ─── Causal self-attention residual ────────────────────────────────────
    // Pre-allocate every op-out tensor on X.device (Tensor::resize preserves
    // device; default-constructed Tensors are CPU and would crash CUDA dispatch).
    bt::Tensor x_ln = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    self_attn_layer_norm.forward(X, x_ln);                  // (T, D)
    bt::Tensor self_out = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    mha_causal_cached_fp32(x_ln,
                           self_Wq, self_bq,
                           self_Wk, self_bk,
                           self_Wv, self_bv,
                           self_Wo, self_bo,
                           num_heads, cache, self_out);     // (T, D)
    bt::add_inplace(self_out, X);                            // residual

    // ─── Cross-attention residual ──────────────────────────────────────────
    bt::Tensor x_ca_ln = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    encoder_attn_layer_norm.forward(self_out, x_ca_ln);
    bt::Tensor cross_out = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    cross_attn_cached_fp32(x_ca_ln,
                           cross_Wq, cross_bq,
                           cross_Wo, cross_bo,
                           num_heads, cache, cross_out);    // (T, D)
    bt::add_inplace(cross_out, self_out);                    // residual

    // ─── FFN residual ──────────────────────────────────────────────────────
    bt::Tensor ffn_ln = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    final_layer_norm.forward(cross_out, ffn_ln);
    bt::Tensor h1 = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    fc1.forward_batched(ffn_ln, h1);
    bt::Tensor h1_act = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    bt::gelu_forward(h1, h1_act);
    bt::Tensor h2 = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    fc2.forward_batched(h1_act, h2);
    bt::add_inplace(h2, cross_out);                          // residual

    Y = std::move(h2);
}

// ─── WhisperDecoder ────────────────────────────────────────────────────────

void WhisperDecoder::load_from(const stf::File& f,
                               int dm, int n_layer, int ffn, int n_head,
                               int vocab, int max_tgt, int max_src,
                               bt::Device device) {
    const std::string where = "WhisperDecoder::load_from";
    const bt::Device prev_dev = g_load_device;
    g_load_device = device;
    struct DevGuard { bt::Device& slot; bt::Device prev;
        ~DevGuard() { slot = prev; } } _guard{g_load_device, prev_dev};
    d_model                  = dm;
    decoder_layers           = n_layer;
    decoder_ffn_dim          = ffn;
    decoder_attention_heads  = n_head;
    vocab_size               = vocab;
    max_target_positions     = max_tgt;
    max_source_positions     = max_src;

    const std::string p = "model.decoder.";

    upload(f, p + "embed_tokens.weight",    vocab,   dm, embed_tokens,    where);
    upload(f, p + "embed_positions.weight", max_tgt, dm, embed_positions, where);

    layers.clear();
    layers.resize(static_cast<std::size_t>(n_layer));
    for (int i = 0; i < n_layer; ++i) {
        const std::string lp = p + "layers." + std::to_string(i) + ".";
        layers[static_cast<std::size_t>(i)].load_from(f, lp, dm, ffn, n_head, device);
    }

    load_layernorm(f, p + "layer_norm", dm, 1e-5f, layer_norm, where);

    // Tied LM head: `proj_out.weight` is usually absent (tied to
    // embed_tokens). If present, prefer it.
    proj_out_explicit = false;
    if (const stf::TensorView* v = f.find("proj_out.weight")) {
        (void)v;
        upload(f, "proj_out.weight", vocab, dm, proj_out_weight, where);
        proj_out_explicit = true;
    } else {
        proj_out_weight = bt::Tensor{};  // empty signals tied mode
    }

    // Build the matmul-ready LM head once: (vocab, d_model) -> (d_model,
    // vocab). nchw_to_sequence with N=1, C=vocab, H=1, W=d_model is the
    // device-dispatched transpose. forward() consumes this directly, so the
    // per-token decode loop never re-transposes the table.
    const bt::Tensor& Wlm = proj_out_explicit ? proj_out_weight : embed_tokens;
    lm_head_T = bt::Tensor::empty_on(Wlm.device, 0, 0, bt::Dtype::FP32);
    bt::nchw_to_sequence(Wlm, /*N=*/1, /*C=*/vocab, /*H=*/1, /*W=*/dm, lm_head_T);
}

void WhisperDecoder::prime_cross(const bt::Tensor& encoder_hidden,
                                 WhisperKVCache& cache) const {
    const std::string where = "WhisperDecoder::prime_cross";
    if (static_cast<int>(cache.layers.size()) != decoder_layers) {
        fail(where, "cache layer count " +
             std::to_string(cache.layers.size()) + " != decoder_layers=" +
             std::to_string(decoder_layers));
    }
    if (encoder_hidden.rows != max_source_positions ||
        encoder_hidden.cols != d_model) {
        fail(where, "encoder_hidden shape (" +
             std::to_string(encoder_hidden.rows) + "," +
             std::to_string(encoder_hidden.cols) + ") != (" +
             std::to_string(max_source_positions) + "," +
             std::to_string(d_model) + ")");
    }
    for (int i = 0; i < decoder_layers; ++i) {
        layers[static_cast<std::size_t>(i)].prime_cross(
            encoder_hidden, cache.layers[static_cast<std::size_t>(i)]);
    }
}

void WhisperDecoder::forward(const std::int32_t* token_ids, int T,
                             int pos_offset,
                             WhisperKVCache& cache,
                             bt::Tensor& logits) const {
    const std::string where = "WhisperDecoder::forward";
    if (T <= 0)               fail(where, "T must be positive");
    if (pos_offset < 0)       fail(where, "pos_offset must be non-negative");
    if (pos_offset + T > max_target_positions) {
        fail(where, "pos_offset+T=" + std::to_string(pos_offset + T) +
             " exceeds max_target_positions=" + std::to_string(max_target_positions));
    }
    if (static_cast<int>(cache.layers.size()) != decoder_layers) {
        fail(where, "cache layer count " +
             std::to_string(cache.layers.size()) +
             " != decoder_layers=" + std::to_string(decoder_layers));
    }
    for (int i = 0; i < decoder_layers; ++i) {
        const auto& l = cache.layers[static_cast<std::size_t>(i)];
        if (!l.cross_primed) {
            fail(where, "cache layer " + std::to_string(i) +
                 " cross-attn not primed; call prime_cross first");
        }
        if (l.self_len != pos_offset) {
            fail(where, "cache layer " + std::to_string(i) +
                 " self_len=" + std::to_string(l.self_len) +
                 " != pos_offset=" + std::to_string(pos_offset));
        }
    }

    // ─── Embed tokens ──────────────────────────────────────────────────────
    // embedding_lookup_forward resizes+dtype-sets `x` to match the table; we
    // still pre-set the device so the resize lands on embed_tokens.device.
    // CUDA `embedding_lookup_forward` reads `d_idx` as a DEVICE pointer (the
    // backend kernel indexes through it), so when the table lives on CUDA we
    // must upload the host-side token_ids to a device buffer first.
    bt::Tensor x = bt::Tensor::empty_on(embed_tokens.device, 0, 0, bt::Dtype::FP32);
    if (embed_tokens.device == bt::Device::CPU) {
        bt::embedding_lookup_forward(embed_tokens, token_ids, T, x);  // (T, d_model)
    } else {
        bt::Tensor idx_dev = upload_int32_idx(embed_tokens.device, token_ids, T);
        bt::embedding_lookup_forward(embed_tokens,
                                     static_cast<const std::int32_t*>(idx_dev.data),
                                     T, x);
    }

    // ─── Add learned positional embedding (rows [pos_offset, pos_offset+T)) ─
    // embed_positions is (max_target_positions, d_model), row-major on the
    // model's device. The T rows we want sit contiguously at byte offset
    // pos_offset*d_model*sizeof(float); view them as a (T, d_model) tensor
    // and add_inplace dispatches on x's device.
    {
        float* pe_slab = static_cast<float*>(embed_positions.data)
                       + static_cast<std::size_t>(pos_offset) * d_model;
        bt::Tensor pe_view = bt::Tensor::view(embed_positions.device, pe_slab,
                                              T, d_model, bt::Dtype::FP32);
        bt::add_inplace(x, pe_view);
    }

    // ─── Pre-LN Transformer stack ──────────────────────────────────────────
    bt::Tensor y = bt::Tensor::empty_on(x.device, 0, 0, bt::Dtype::FP32);
    for (int i = 0; i < decoder_layers; ++i) {
        layers[static_cast<std::size_t>(i)].forward(
            x, cache.layers[static_cast<std::size_t>(i)], y);
        x = std::move(y);
        // After move, `y` is moved-from; re-seed for the next iter so the
        // dispatcher continues to see an x.device-resident out tensor.
        y = bt::Tensor::empty_on(x.device, 0, 0, bt::Dtype::FP32);
    }

    // ─── Final decoder LayerNorm ───────────────────────────────────────────
    bt::Tensor hidden = bt::Tensor::empty_on(x.device, 0, 0, bt::Dtype::FP32);
    layer_norm.forward(x, hidden);          // (T, d_model)

    // ─── LM head: tied (default) or explicit proj_out ──────────────────────
    // For both paths the math is the same: logits = hidden (T, D) @ W^T,
    // where W is (vocab, D). brotensor::matmul has no transpose flag, so
    // load_from pre-transposed the table into lm_head_T ((d_model, vocab)) —
    // the decode loop only pays the (T, D) @ (D, vocab) matmul per step.
    if (lm_head_T.rows != d_model || lm_head_T.cols != vocab_size) {
        fail(where, "lm_head_T not built — load_from() must run first");
    }
    // Pre-allocate logits on the same device too; the caller's logits tensor
    // is default-constructed (CPU) and matmul resizes but preserves device.
    if (logits.device != hidden.device || logits.rows != T ||
        logits.cols != vocab_size || logits.dtype != bt::Dtype::FP32) {
        logits = bt::Tensor::empty_on(hidden.device, T, vocab_size, bt::Dtype::FP32);
    }
    bt::matmul(hidden, lm_head_T, logits);  // (T, vocab_size)
}

}  // namespace brosoundml
