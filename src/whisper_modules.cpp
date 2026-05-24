#include "brosoundml/whisper_modules.h"

#include <brotensor/ops.h>
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

// Upload a tensor named `key` from `f` into `dst` reshaped to (rows, cols).
// Mirrors the helper in kokoro_modules.cpp.
void upload(const stf::File& f, const std::string& key,
            int rows, int cols, bt::Tensor& dst,
            const std::string& where) {
    const stf::TensorView* view = f.find(key);
    if (!view) fail(where, "missing safetensors key '" + key + "'");
    if (view->dtype != stf::Dtype::F32) {
        fail(where, "tensor '" + key + "' is not F32 (got dtype " +
                    std::to_string(static_cast<int>(view->dtype)) + ")");
    }
    const std::int64_t n = view->numel();
    if (n != static_cast<std::int64_t>(rows) * cols) {
        fail(where, "tensor '" + key + "' has " + std::to_string(n) +
                    " elements; expected " +
                    std::to_string(static_cast<std::int64_t>(rows) * cols));
    }
    stf::upload(*view, rows, cols, dst);
}

// ─── Mel-scale conversions (Slaney) ────────────────────────────────────────
//
// Slaney mel: linear below 1000 Hz (slope 1/(200/3) = 3/200), logarithmic
// above. Used by librosa's `mel(..., htk=False)` and by openai/whisper's
// reference filterbank.
double hz_to_mel_slaney(double hz) {
    constexpr double f_min      = 0.0;
    constexpr double f_sp       = 200.0 / 3.0;          // ≈ 66.667
    constexpr double min_log_hz = 1000.0;
    constexpr double min_log_mel = (min_log_hz - f_min) / f_sp;  // == 15
    constexpr double logstep    = 0.06875177742094911;  // log(6.4) / 27

    if (hz >= min_log_hz) {
        return min_log_mel + std::log(hz / min_log_hz) / logstep;
    }
    return (hz - f_min) / f_sp;
}

double mel_to_hz_slaney(double mel) {
    constexpr double f_min      = 0.0;
    constexpr double f_sp       = 200.0 / 3.0;
    constexpr double min_log_hz = 1000.0;
    constexpr double min_log_mel = (min_log_hz - f_min) / f_sp;
    constexpr double logstep    = 0.06875177742094911;

    if (mel >= min_log_mel) {
        return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    }
    return f_min + f_sp * mel;
}

// Build the (n_mels, n_fft/2 + 1) Slaney-normalised mel filterbank used by
// librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
// fmin=0, fmax=sample_rate/2, htk=False, norm="slaney"). Returns the buffer
// flat in row-major order.
std::vector<float> build_mel_filterbank(int n_mels, int n_fft, int sample_rate) {
    const int    n_bins = n_fft / 2 + 1;
    const double f_max  = sample_rate / 2.0;
    const double f_min  = 0.0;

    // FFT-bin centre frequencies in Hz: 0, sr/n_fft, 2*sr/n_fft, ...
    std::vector<double> fft_freqs(n_bins);
    for (int k = 0; k < n_bins; ++k) {
        fft_freqs[k] = static_cast<double>(k) * sample_rate / n_fft;
    }

    // n_mels + 2 mel-spaced points, converted back to Hz.
    const double mel_min = hz_to_mel_slaney(f_min);
    const double mel_max = hz_to_mel_slaney(f_max);
    std::vector<double> mel_hz(static_cast<std::size_t>(n_mels) + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        const double m = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        mel_hz[static_cast<std::size_t>(i)] = mel_to_hz_slaney(m);
    }

    std::vector<float> fb(static_cast<std::size_t>(n_mels) * n_bins, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        const double lo = mel_hz[static_cast<std::size_t>(m)];
        const double ce = mel_hz[static_cast<std::size_t>(m) + 1];
        const double hi = mel_hz[static_cast<std::size_t>(m) + 2];
        // Slaney area normalisation: enorm = 2 / (hi - lo).
        const double enorm = 2.0 / (hi - lo);
        for (int k = 0; k < n_bins; ++k) {
            const double f = fft_freqs[static_cast<std::size_t>(k)];
            double w;
            if (f <= lo || f >= hi) {
                w = 0.0;
            } else if (f <= ce) {
                w = (f - lo) / (ce - lo);
            } else {
                w = (hi - f) / (hi - ce);
            }
            fb[static_cast<std::size_t>(m) * n_bins + k] =
                static_cast<float>(w * enorm);
        }
    }
    return fb;
}

}  // namespace

// ─── LogMel ────────────────────────────────────────────────────────────────

void LogMel::build(int n_mels, bt::Device device) {
    const std::string where = "LogMel::build";
    if (n_mels <= 0) fail(where, "num_mel_bins must be positive");
    if (device != bt::Device::CPU) {
        fail(where, "stage 2 not yet ported off CPU");
    }
    num_mel_bins = n_mels;
    n_fft        = 400;
    hop_length   = 160;
    win_length   = 400;
    n_frames     = 3000;
    sample_rate  = 16000;

    const int n_bins = n_fft / 2 + 1;   // 201

    // Mel filterbank.
    std::vector<float> fb = build_mel_filterbank(num_mel_bins, n_fft, sample_rate);
    mel_filters = bt::Tensor::zeros_on(device, num_mel_bins, n_bins, bt::Dtype::FP32);
    std::memcpy(mel_filters.host_f32_mut(), fb.data(),
                fb.size() * sizeof(float));

    // Hann window of length win_length (periodic = false matches torch's
    // default analysis window). Whisper passes `torch.hann_window(N_FFT)`
    // which is the periodic variant by default — librosa uses sym=True.
    // OpenAI's whisper/audio.py uses torch.hann_window which defaults to
    // periodic=True: w[n] = 0.5 * (1 - cos(2*pi*n/N)) with N == win_length.
    hann_window = bt::Tensor::zeros_on(device, 1, win_length, bt::Dtype::FP32);
    float* w = hann_window.host_f32_mut();
    constexpr double k_two_pi = 6.283185307179586;
    for (int n = 0; n < win_length; ++n) {
        const double phase = k_two_pi * static_cast<double>(n) /
                             static_cast<double>(win_length);
        w[n] = static_cast<float>(0.5 * (1.0 - std::cos(phase)));
    }
}

void LogMel::forward(const AudioBuffer& audio, bt::Tensor& out) const {
    const std::string where = "LogMel::forward";
    if (mel_filters.rows == 0) fail(where, "LogMel::build() not called");
    if (audio.sample_rate != sample_rate) {
        fail(where, "audio sample_rate " + std::to_string(audio.sample_rate) +
                    " != Whisper's required " + std::to_string(sample_rate));
    }
    if (mel_filters.device != bt::Device::CPU) {
        fail(where, "stage 2 not yet ported off CPU");
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
    out.resize(num_mel_bins, n_frames, bt::Dtype::FP32);
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
                                    int dm, int ffn, int nh) {
    const std::string where = "WhisperEncoderLayer::load_from";
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
    self_attn.bk = bt::Tensor::zeros_on(bt::Device::CPU, dm, 1, bt::Dtype::FP32);
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
    if (X.device != bt::Device::CPU || X.dtype != bt::Dtype::FP32) {
        fail(where, "stage 3 not yet ported off CPU");
    }
    if (X.cols != d_model) {
        fail(where, "X cols=" + std::to_string(X.cols) +
                    " != d_model=" + std::to_string(d_model));
    }
    // ─── Self-attention residual ───────────────────────────────────────────
    bt::Tensor x_ln;
    self_attn_layer_norm.forward(X, x_ln);                 // (L, D)
    bt::Tensor attn_out;
    self_attn.forward(x_ln, /*d_mask=*/nullptr, attn_out); // (L, D)
    bt::add_inplace(attn_out, X);                           // residual

    // ─── FFN residual ──────────────────────────────────────────────────────
    bt::Tensor ffn_ln;
    final_layer_norm.forward(attn_out, ffn_ln);            // (L, D)
    bt::Tensor h1;
    fc1.forward_batched(ffn_ln, h1);                       // (L, ffn_dim)
    bt::Tensor h1_act;
    bt::gelu_forward(h1, h1_act);
    bt::Tensor h2;
    fc2.forward_batched(h1_act, h2);                       // (L, D)
    bt::add_inplace(h2, attn_out);                          // residual

    Y = std::move(h2);
}

// ─── WhisperEncoder ────────────────────────────────────────────────────────

void WhisperEncoder::load_from(const stf::File& f,
                               int n_mel, int dm, int max_src,
                               int n_layer, int ffn, int n_head) {
    const std::string where = "WhisperEncoder::load_from";
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
        layers[static_cast<std::size_t>(i)].load_from(f, lp, dm, ffn, n_head);
    }

    // Final encoder LayerNorm.
    load_layernorm(f, p + "layer_norm", dm, 1e-5f, layer_norm, where);
}

void WhisperEncoder::forward(const bt::Tensor& mel, bt::Tensor& hidden) const {
    const std::string where = "WhisperEncoder::forward";
    if (mel.device != bt::Device::CPU || mel.dtype != bt::Dtype::FP32) {
        fail(where, "stage 3 not yet ported off CPU");
    }
    // Accept either (num_mel_bins, n_frames) "flat 2D" or NCL (1, n_mel*n_frames).
    int n_mel, n_frames;
    bt::Tensor mel_ncl;
    if (mel.rows == num_mel_bins && mel.cols > 0) {
        n_mel    = num_mel_bins;
        n_frames = mel.cols;
        // Reshape view: (num_mel_bins, n_frames) is already row-major NCL with
        // N=1 — the memory layout matches (1, num_mel_bins * n_frames).
        mel_ncl = bt::Tensor::view(bt::Device::CPU, mel.data,
                                   1, n_mel * n_frames, bt::Dtype::FP32);
    } else if (mel.rows == 1 && mel.cols == num_mel_bins * /*n_frames=*/3000) {
        n_mel    = num_mel_bins;
        n_frames = 3000;
        mel_ncl = bt::Tensor::view(bt::Device::CPU, mel.data, 1,
                                   n_mel * n_frames, bt::Dtype::FP32);
    } else {
        fail(where, "mel input must be (num_mel_bins, n_frames) or (1, n_mel*n_frames)");
    }

    // ─── conv1 + GELU ──────────────────────────────────────────────────────
    bt::Tensor x1;
    conv1.forward(mel_ncl, /*N=*/1, /*L=*/n_frames, x1);    // (1, d_model*n_frames)
    bt::Tensor x1g;
    bt::gelu_forward(x1, x1g);

    // ─── conv2 (stride=2) + GELU ───────────────────────────────────────────
    bt::Tensor x2;
    conv2.forward(x1g, /*N=*/1, /*L=*/n_frames, x2);
    bt::Tensor x2g;
    bt::gelu_forward(x2, x2g);
    // Output length after stride-2 conv with padding=1, kernel=3:
    //   L_out = (L + 2*1 - 1*(3-1) - 1)/2 + 1 = L/2.
    const int L = n_frames / 2;
    if (L != max_source_positions) {
        fail(where, "post-conv length " + std::to_string(L) +
                    " != max_source_positions=" + std::to_string(max_source_positions));
    }

    // ─── Transpose NCL -> NLC: (1, D*L) -> (L, D) ──────────────────────────
    bt::Tensor x = bt::Tensor::zeros_on(bt::Device::CPU, L, d_model, bt::Dtype::FP32);
    {
        const float* src = x2g.host_f32();
        float* dst = x.host_f32_mut();
        for (int c = 0; c < d_model; ++c) {
            for (int l = 0; l < L; ++l) {
                dst[l * d_model + c] = src[c * L + l];
            }
        }
    }

    // ─── Add positional embedding ──────────────────────────────────────────
    {
        const float* pe = embed_positions.host_f32();
        float* xd = x.host_f32_mut();
        const std::size_t n = static_cast<std::size_t>(L) * d_model;
        for (std::size_t i = 0; i < n; ++i) xd[i] += pe[i];
    }

    // ─── Pre-LN Transformer stack ──────────────────────────────────────────
    bt::Tensor y;
    for (const auto& layer : layers) {
        layer.forward(x, y);
        x = std::move(y);
    }

    // ─── Final encoder LayerNorm ───────────────────────────────────────────
    layer_norm.forward(x, hidden);
}

// ─── WhisperKVCache ────────────────────────────────────────────────────────

void WhisperKVCache::allocate(int decoder_layers, int d_model,
                              int max_target_positions, int max_source_positions) {
    const std::string where = "WhisperKVCache::allocate";
    if (decoder_layers <= 0) fail(where, "decoder_layers must be positive");
    if (d_model <= 0)        fail(where, "d_model must be positive");
    if (max_target_positions <= 0) fail(where, "max_target_positions must be positive");
    if (max_source_positions <= 0) fail(where, "max_source_positions must be positive");

    layers.clear();
    layers.resize(static_cast<std::size_t>(decoder_layers));
    for (auto& l : layers) {
        l.self_k = bt::Tensor::zeros_on(bt::Device::CPU, max_target_positions,
                                        d_model, bt::Dtype::FP32);
        l.self_v = bt::Tensor::zeros_on(bt::Device::CPU, max_target_positions,
                                        d_model, bt::Dtype::FP32);
        l.cross_k = bt::Tensor::zeros_on(bt::Device::CPU, max_source_positions,
                                         d_model, bt::Dtype::FP32);
        l.cross_v = bt::Tensor::zeros_on(bt::Device::CPU, max_source_positions,
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
    if (X.device != bt::Device::CPU || X.dtype != bt::Dtype::FP32) {
        fail(where, "currently CPU FP32 only");
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

    // Project Q, K, V for the current T input rows.
    bt::Tensor Q, Knew, Vnew;
    bt::linear_forward_batched(Wq, bq, X, Q);     // (T, D)
    bt::linear_forward_batched(Wk, bk, X, Knew);  // (T, D)
    bt::linear_forward_batched(Wv, bv, X, Vnew);  // (T, D)

    // Append Knew, Vnew into the cache at rows [self_len, self_len + T).
    {
        const std::size_t row_bytes = static_cast<std::size_t>(D) * sizeof(float);
        float*       kdst = cache.self_k.host_f32_mut()
                          + static_cast<std::size_t>(cache.self_len) * D;
        float*       vdst = cache.self_v.host_f32_mut()
                          + static_cast<std::size_t>(cache.self_len) * D;
        std::memcpy(kdst, Knew.host_f32(), static_cast<std::size_t>(T) * row_bytes);
        std::memcpy(vdst, Vnew.host_f32(), static_cast<std::size_t>(T) * row_bytes);
    }

    const int head_dim = D / num_heads;
    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int Lk_total = cache.self_len + T;

    const float* Qd = Q.host_f32();
    const float* Kd = cache.self_k.host_f32();
    const float* Vd = cache.self_v.host_f32();

    bt::Tensor ctx = bt::Tensor::zeros_on(bt::Device::CPU, T, D, bt::Dtype::FP32);
    float* ctx_d = ctx.host_f32_mut();

    std::vector<float> scores(static_cast<std::size_t>(Lk_total), 0.0f);

    // Query row q (0..T-1) corresponds to absolute position self_len + q;
    // it may attend to keys [0, self_len + q].
    for (int h = 0; h < num_heads; ++h) {
        for (int q = 0; q < T; ++q) {
            const int abs_q  = cache.self_len + q;
            const int kv_end = abs_q + 1;                 // exclusive
            const float* q_vec = Qd + static_cast<std::size_t>(q) * D + h * head_dim;

            float max_score = -INFINITY;
            for (int k = 0; k < kv_end; ++k) {
                const float* k_vec = Kd + static_cast<std::size_t>(k) * D + h * head_dim;
                float s = 0.0f;
                for (int j = 0; j < head_dim; ++j) s += q_vec[j] * k_vec[j];
                s *= scale;
                scores[static_cast<std::size_t>(k)] = s;
                if (s > max_score) max_score = s;
            }
            float sum = 0.0f;
            for (int k = 0; k < kv_end; ++k) {
                scores[static_cast<std::size_t>(k)] =
                    std::exp(scores[static_cast<std::size_t>(k)] - max_score);
                sum += scores[static_cast<std::size_t>(k)];
            }
            const float inv = sum > 0 ? 1.0f / sum : 0.0f;
            for (int k = 0; k < kv_end; ++k) {
                scores[static_cast<std::size_t>(k)] *= inv;
            }

            float* ctx_row = ctx_d + static_cast<std::size_t>(q) * D + h * head_dim;
            for (int k = 0; k < kv_end; ++k) {
                const float w = scores[static_cast<std::size_t>(k)];
                const float* v_vec = Vd + static_cast<std::size_t>(k) * D + h * head_dim;
                for (int j = 0; j < head_dim; ++j) ctx_row[j] += w * v_vec[j];
            }
        }
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
    if (X.device != bt::Device::CPU || X.dtype != bt::Dtype::FP32) {
        fail(where, "currently CPU FP32 only");
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

    bt::Tensor Q;
    bt::linear_forward_batched(Wq, bq, X, Q);  // (T, D)

    const int head_dim = D / num_heads;
    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const float* Qd = Q.host_f32();
    const float* Kd = cache.cross_k.host_f32();
    const float* Vd = cache.cross_v.host_f32();

    bt::Tensor ctx = bt::Tensor::zeros_on(bt::Device::CPU, T, D, bt::Dtype::FP32);
    float* ctx_d = ctx.host_f32_mut();

    std::vector<float> scores(static_cast<std::size_t>(Lk), 0.0f);

    for (int h = 0; h < num_heads; ++h) {
        for (int q = 0; q < T; ++q) {
            const float* q_vec = Qd + static_cast<std::size_t>(q) * D + h * head_dim;

            float max_score = -INFINITY;
            for (int k = 0; k < Lk; ++k) {
                const float* k_vec = Kd + static_cast<std::size_t>(k) * D + h * head_dim;
                float s = 0.0f;
                for (int j = 0; j < head_dim; ++j) s += q_vec[j] * k_vec[j];
                s *= scale;
                scores[static_cast<std::size_t>(k)] = s;
                if (s > max_score) max_score = s;
            }
            float sum = 0.0f;
            for (int k = 0; k < Lk; ++k) {
                scores[static_cast<std::size_t>(k)] =
                    std::exp(scores[static_cast<std::size_t>(k)] - max_score);
                sum += scores[static_cast<std::size_t>(k)];
            }
            const float inv = sum > 0 ? 1.0f / sum : 0.0f;
            for (int k = 0; k < Lk; ++k) {
                scores[static_cast<std::size_t>(k)] *= inv;
            }

            float* ctx_row = ctx_d + static_cast<std::size_t>(q) * D + h * head_dim;
            for (int k = 0; k < Lk; ++k) {
                const float w = scores[static_cast<std::size_t>(k)];
                const float* v_vec = Vd + static_cast<std::size_t>(k) * D + h * head_dim;
                for (int j = 0; j < head_dim; ++j) ctx_row[j] += w * v_vec[j];
            }
        }
    }

    bt::linear_forward_batched(Wo, bo, ctx, out);
}

// ─── prime_cross_cache_fp32 ────────────────────────────────────────────────

void prime_cross_cache_fp32(const bt::Tensor& encoder_hidden,
                            const bt::Tensor& Wk, const bt::Tensor& bk,
                            const bt::Tensor& Wv, const bt::Tensor& bv,
                            WhisperLayerCache& cache) {
    const std::string where = "prime_cross_cache_fp32";
    if (encoder_hidden.device != bt::Device::CPU ||
        encoder_hidden.dtype  != bt::Dtype::FP32) {
        fail(where, "currently CPU FP32 only");
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

    bt::Tensor Kproj, Vproj;
    bt::linear_forward_batched(Wk, bk, encoder_hidden, Kproj);
    bt::linear_forward_batched(Wv, bv, encoder_hidden, Vproj);

    const std::size_t bytes = static_cast<std::size_t>(L) * D * sizeof(float);
    std::memcpy(cache.cross_k.host_f32_mut(), Kproj.host_f32(), bytes);
    std::memcpy(cache.cross_v.host_f32_mut(), Vproj.host_f32(), bytes);
    cache.cross_primed = true;
}

// ─── WhisperDecoderLayer ───────────────────────────────────────────────────

void WhisperDecoderLayer::load_from(const stf::File& f,
                                    const std::string& prefix,
                                    int dm, int ffn, int nh) {
    const std::string where = "WhisperDecoderLayer::load_from";
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
    self_bk = bt::Tensor::zeros_on(bt::Device::CPU, dm, 1, bt::Dtype::FP32);
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
    cross_bk = bt::Tensor::zeros_on(bt::Device::CPU, dm, 1, bt::Dtype::FP32);
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
    if (X.device != bt::Device::CPU || X.dtype != bt::Dtype::FP32) {
        fail(where, "stage 4 not yet ported off CPU");
    }
    if (X.cols != d_model) {
        fail(where, "X cols=" + std::to_string(X.cols) +
                    " != d_model=" + std::to_string(d_model));
    }

    // ─── Causal self-attention residual ────────────────────────────────────
    bt::Tensor x_ln;
    self_attn_layer_norm.forward(X, x_ln);                  // (T, D)
    bt::Tensor self_out;
    mha_causal_cached_fp32(x_ln,
                           self_Wq, self_bq,
                           self_Wk, self_bk,
                           self_Wv, self_bv,
                           self_Wo, self_bo,
                           num_heads, cache, self_out);     // (T, D)
    bt::add_inplace(self_out, X);                            // residual

    // ─── Cross-attention residual ──────────────────────────────────────────
    bt::Tensor x_ca_ln;
    encoder_attn_layer_norm.forward(self_out, x_ca_ln);
    bt::Tensor cross_out;
    cross_attn_cached_fp32(x_ca_ln,
                           cross_Wq, cross_bq,
                           cross_Wo, cross_bo,
                           num_heads, cache, cross_out);    // (T, D)
    bt::add_inplace(cross_out, self_out);                    // residual

    // ─── FFN residual ──────────────────────────────────────────────────────
    bt::Tensor ffn_ln;
    final_layer_norm.forward(cross_out, ffn_ln);
    bt::Tensor h1;
    fc1.forward_batched(ffn_ln, h1);
    bt::Tensor h1_act;
    bt::gelu_forward(h1, h1_act);
    bt::Tensor h2;
    fc2.forward_batched(h1_act, h2);
    bt::add_inplace(h2, cross_out);                          // residual

    Y = std::move(h2);
}

// ─── WhisperDecoder ────────────────────────────────────────────────────────

void WhisperDecoder::load_from(const stf::File& f,
                               int dm, int n_layer, int ffn, int n_head,
                               int vocab, int max_tgt, int max_src) {
    const std::string where = "WhisperDecoder::load_from";
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
        layers[static_cast<std::size_t>(i)].load_from(f, lp, dm, ffn, n_head);
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
    bt::Tensor x;
    bt::embedding_lookup_forward(embed_tokens, token_ids, T, x);  // (T, d_model)

    // ─── Add learned positional embedding (rows [pos_offset, pos_offset+T)) ─
    {
        const float* pe_all = embed_positions.host_f32();
        float* xd = x.host_f32_mut();
        for (int t = 0; t < T; ++t) {
            const float* pe_row = pe_all +
                static_cast<std::size_t>(pos_offset + t) * d_model;
            float* x_row = xd + static_cast<std::size_t>(t) * d_model;
            for (int c = 0; c < d_model; ++c) x_row[c] += pe_row[c];
        }
    }

    // ─── Pre-LN Transformer stack ──────────────────────────────────────────
    bt::Tensor y;
    for (int i = 0; i < decoder_layers; ++i) {
        layers[static_cast<std::size_t>(i)].forward(
            x, cache.layers[static_cast<std::size_t>(i)], y);
        x = std::move(y);
    }

    // ─── Final decoder LayerNorm ───────────────────────────────────────────
    bt::Tensor hidden;
    layer_norm.forward(x, hidden);          // (T, d_model)

    // ─── LM head: tied (default) or explicit proj_out ──────────────────────
    // For both paths the math is the same: logits = hidden (T, D) @ W^T,
    // where W is (vocab, D). brotensor::matmul has no transpose flag, so we
    // build a (D, vocab) view by transposing the embedding table at call
    // time. The transpose is one allocation and one tight loop per
    // forward() — acceptable for stage 4; can be cached / fused later.
    const bt::Tensor& W = proj_out_explicit ? proj_out_weight : embed_tokens;
    bt::Tensor W_T = bt::Tensor::zeros_on(bt::Device::CPU, d_model, vocab_size,
                                          bt::Dtype::FP32);
    {
        const float* src = W.host_f32();
        float* dst = W_T.host_f32_mut();
        for (int v = 0; v < vocab_size; ++v) {
            for (int c = 0; c < d_model; ++c) {
                dst[c * vocab_size + v] = src[v * d_model + c];
            }
        }
    }
    bt::matmul(hidden, W_T, logits);        // (T, vocab_size)
}

}  // namespace brosoundml
