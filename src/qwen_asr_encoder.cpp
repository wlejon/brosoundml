#include "qwen_asr_encoder.h"

#include "mel_slaney.h"
#include "qwen_tts_device.h"   // qtd:: device-neutral helpers (model-agnostic)

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: QwenAsrEncoder: " + msg);
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Upload a weight to FP32 on `dev`. The checkpoint is BF16 on disk;
// upload_compute_checked under a CPU scope widens it to host FP32, then it is
// migrated to the target device (same pattern as the Qwen3-TTS Talker).
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols,
              bt::Device dev) {
    bt::Tensor t;
    {
        bt::DeviceScope cpu(bt::Device::CPU);
        sf::upload_compute_checked(need(f, name), rows, cols, t, name);
    }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
bt::Tensor up_vec(const sf::File& f, const std::string& name, int n,
                  bt::Device dev) {
    return up(f, name, n, 1, dev);
}

// Output length of one 3x3 / stride-2 / pad-1 conv axis: (n - 1)/2 + 1.
int conv_len(int n) { return (n - 1) / 2 + 1; }

// Tokens after the 3-conv stem for an n-frame chunk (time axis).
int tokens_after_stem(int n) { return conv_len(conv_len(conv_len(n))); }

constexpr int kNFft      = 400;
constexpr int kHopLength = 160;
constexpr int kNBins     = kNFft / 2 + 1;   // 201

}  // namespace

void QwenAsrEncoder::load(const sf::File& f, const QwenAsrConfig& cfg,
                          bt::Device dev) {
    num_mel_bins   = cfg.num_mel_bins;
    d_model        = cfg.d_model;
    num_layers     = cfg.encoder_layers;
    num_heads      = cfg.encoder_attention_heads;
    ffn_dim        = cfg.encoder_ffn_dim;
    output_dim     = cfg.output_dim;
    conv_hidden    = cfg.downsample_hidden_size;
    n_window       = cfg.n_window;
    n_window_infer = cfg.n_window_infer;

    if (num_mel_bins <= 0 || d_model <= 0 || num_layers <= 0 || n_window <= 0)
        fail("config not parsed (zero dims)");
    if (d_model % num_heads != 0)
        fail("d_model not divisible by encoder_attention_heads");
    if (n_window_infer % (n_window * 2) != 0)
        fail("n_window_infer not a multiple of the conv chunk size");

    // ── mel tables (host-resident, closed-form) ──
    {
        std::vector<float> fb =
            melslaney::build_filterbank(num_mel_bins, kNFft, cfg.sample_rate);
        mel_filters = bt::Tensor::zeros_on(bt::Device::CPU, num_mel_bins, kNBins,
                                           bt::Dtype::FP32);
        std::memcpy(mel_filters.host_f32_mut(), fb.data(),
                    fb.size() * sizeof(float));
        std::vector<float> hw = melslaney::build_hann_window(kNFft);
        hann_window = bt::Tensor::zeros_on(bt::Device::CPU, 1, kNFft,
                                           bt::Dtype::FP32);
        std::memcpy(hann_window.host_f32_mut(), hw.data(),
                    hw.size() * sizeof(float));
    }

    // ── conv stem ──
    const std::string P = "thinker.audio_tower.";
    conv1_w = up(f, P + "conv2d1.weight", conv_hidden, 1 * 3 * 3, dev);
    conv1_b = up_vec(f, P + "conv2d1.bias", conv_hidden, dev);
    conv2_w = up(f, P + "conv2d2.weight", conv_hidden, conv_hidden * 3 * 3, dev);
    conv2_b = up_vec(f, P + "conv2d2.bias", conv_hidden, dev);
    conv3_w = up(f, P + "conv2d3.weight", conv_hidden, conv_hidden * 3 * 3, dev);
    conv3_b = up_vec(f, P + "conv2d3.bias", conv_hidden, dev);

    // conv_out is a bias-free Linear(conv_hidden*freq_out -> d_model) over the
    // (t, c*f) flatten of the stem output. Row-major (d_model, c*f) coincides
    // with OIHW (d_model, conv_hidden, freq_out, 1), so the same buffer drives
    // a full-height conv2d and the flatten never leaves the device.
    const int freq_out = tokens_after_stem(num_mel_bins);   // 128 -> 16
    conv_out_w = up(f, P + "conv_out.weight", d_model, conv_hidden * freq_out, dev);

    // ── transformer stack ──
    layers.clear();
    layers.resize(static_cast<std::size_t>(num_layers));
    for (int i = 0; i < num_layers; ++i) {
        const std::string L = P + "layers." + std::to_string(i) + ".";
        QwenAsrEncoderLayer& el = layers[static_cast<std::size_t>(i)];
        el.ln1_w = up_vec(f, L + "self_attn_layer_norm.weight", d_model, dev);
        el.ln1_b = up_vec(f, L + "self_attn_layer_norm.bias", d_model, dev);
        el.qw = up(f, L + "self_attn.q_proj.weight", d_model, d_model, dev);
        el.qb = up_vec(f, L + "self_attn.q_proj.bias", d_model, dev);
        el.kw = up(f, L + "self_attn.k_proj.weight", d_model, d_model, dev);
        el.kb = up_vec(f, L + "self_attn.k_proj.bias", d_model, dev);
        el.vw = up(f, L + "self_attn.v_proj.weight", d_model, d_model, dev);
        el.vb = up_vec(f, L + "self_attn.v_proj.bias", d_model, dev);
        el.ow = up(f, L + "self_attn.out_proj.weight", d_model, d_model, dev);
        el.ob = up_vec(f, L + "self_attn.out_proj.bias", d_model, dev);
        el.ln2_w = up_vec(f, L + "final_layer_norm.weight", d_model, dev);
        el.ln2_b = up_vec(f, L + "final_layer_norm.bias", d_model, dev);
        el.fc1_w = up(f, L + "fc1.weight", ffn_dim, d_model, dev);
        el.fc1_b = up_vec(f, L + "fc1.bias", ffn_dim, dev);
        el.fc2_w = up(f, L + "fc2.weight", d_model, ffn_dim, dev);
        el.fc2_b = up_vec(f, L + "fc2.bias", d_model, dev);
    }
    ln_post_w = up_vec(f, P + "ln_post.weight", d_model, dev);
    ln_post_b = up_vec(f, P + "ln_post.bias", d_model, dev);
    proj1_w = up(f, P + "proj1.weight", d_model, d_model, dev);
    proj1_b = up_vec(f, P + "proj1.bias", d_model, dev);
    proj2_w = up(f, P + "proj2.weight", output_dim, d_model, dev);
    proj2_b = up_vec(f, P + "proj2.bias", output_dim, dev);

    // ── per-chunk sinusoidal positions (sin | cos halves, Whisper-style) ──
    {
        const int max_tokens = tokens_after_stem(n_window * 2);   // 100 -> 13
        const int half = d_model / 2;
        const double lti = std::log(10000.0) / (half - 1);
        std::vector<float> pe(static_cast<std::size_t>(max_tokens) * d_model);
        for (int t = 0; t < max_tokens; ++t) {
            for (int i = 0; i < half; ++i) {
                const double ang = t * std::exp(-lti * i);
                pe[static_cast<std::size_t>(t) * d_model + i] =
                    static_cast<float>(std::sin(ang));
                pe[static_cast<std::size_t>(t) * d_model + half + i] =
                    static_cast<float>(std::cos(ang));
            }
        }
        pos_table = bt::Tensor::from_host_on(dev, pe.data(), max_tokens, d_model);
    }
}

std::vector<float> QwenAsrEncoder::log_mel(const AudioBuffer& audio,
                                           int& frames_out) const {
    if (mel_filters.rows == 0) fail("load() not called");
    if (audio.sample_rate != 16000) {
        fail("audio sample_rate " + std::to_string(audio.sample_rate) +
             " != the model's required 16000");
    }
    const int L = static_cast<int>(audio.samples.size());
    if (L < kHopLength) fail("audio shorter than one mel hop (10 ms)");

    bt::DeviceScope cpu(bt::Device::CPU);

    // STFT (center=True, reflect-pad): 1 + L/hop frames; the last frame is
    // dropped (WhisperFeatureExtractor's `stft[..., :-1]`) — no 30 s padding,
    // the encoder consumes the true frame count.
    bt::Tensor signal = bt::Tensor::from_host_on(bt::Device::CPU,
                                                 audio.samples.data(), 1, L);
    bt::Tensor spec;
    bt::stft(signal, hann_window, /*N=*/1, kNFft, kHopLength, kNFft,
             /*center=*/true, /*normalized=*/false, spec);
    const int frames = spec.rows - 1;
    if (frames < 1) fail("stft produced no frames");

    // Power spectrum over the kept frames.
    bt::Tensor power = bt::Tensor::zeros_on(bt::Device::CPU, frames, kNBins,
                                            bt::Dtype::FP32);
    {
        const float* sd = spec.host_f32();
        float*       pd = power.host_f32_mut();
        for (int fr = 0; fr < frames; ++fr) {
            const float* src = sd + static_cast<std::size_t>(fr) * 2 * kNBins;
            float*       dst = pd + static_cast<std::size_t>(fr) * kNBins;
            for (int k = 0; k < kNBins; ++k) {
                const float re = src[2 * k + 0];
                const float im = src[2 * k + 1];
                dst[k] = re * re + im * im;
            }
        }
    }

    // Mel projection: (frames, 201) @ (201, n_mels) -> frame-major (frames,
    // n_mels), the layout the chunk assembly below wants.
    bt::Tensor mel_ft;
    {
        bt::Tensor fb_T = bt::Tensor::zeros_on(bt::Device::CPU, kNBins,
                                               num_mel_bins, bt::Dtype::FP32);
        const float* src = mel_filters.host_f32();
        float*       dst = fb_T.host_f32_mut();
        for (int m = 0; m < num_mel_bins; ++m)
            for (int k = 0; k < kNBins; ++k)
                dst[static_cast<std::size_t>(k) * num_mel_bins + m] =
                    src[static_cast<std::size_t>(m) * kNBins + k];
        bt::matmul(power, fb_T, mel_ft);
    }

    // log10 with clamp, whole-utterance dynamic-range clamp, (x+4)/4.
    const std::size_t total =
        static_cast<std::size_t>(frames) * num_mel_bins;
    std::vector<float> out(total);
    const float* md = mel_ft.host_f32();
    float global_max = -1e30f;
    for (std::size_t i = 0; i < total; ++i) {
        const float lv = std::log10(std::max(md[i], 1e-10f));
        out[i] = lv;
        if (lv > global_max) global_max = lv;
    }
    const float floor_v = global_max - 8.0f;
    for (std::size_t i = 0; i < total; ++i)
        out[i] = (std::max(out[i], floor_v) + 4.0f) / 4.0f;

    frames_out = frames;
    return out;
}

void QwenAsrEncoder::forward(const AudioBuffer& audio, bt::Tensor& out) const {
    const bt::Device dev = proj2_w.device;
    bt::DeviceScope scope(dev);

    int frames = 0;
    const std::vector<float> mel = log_mel(audio, frames);   // (frames, n_mels)

    // ── per-chunk conv stem, all chunks batched as N ──
    // Chunks are n_window*2 = 100 mel frames; a partial tail is zero-padded to
    // the batch max (upstream pad_sequence) and its dead tokens dropped after
    // the conv. With a single short chunk the "max" is its own length.
    const int chunk_frames = n_window * 2;
    const int n_chunks = (frames + chunk_frames - 1) / chunk_frames;
    const int tail_len = frames - (n_chunks - 1) * chunk_frames;
    const int chunk_w  = std::min(frames, chunk_frames);
    const int t_out    = tokens_after_stem(chunk_w);    // tokens per padded chunk
    const int n_valid  = (n_chunks - 1) * t_out + tokens_after_stem(tail_len);

    // Host-assemble (n_chunks, 1*n_mels*chunk_w) NCHW slabs: H=mel, W=time.
    bt::Tensor x;
    {
        std::vector<float> slab(static_cast<std::size_t>(n_chunks) *
                                    num_mel_bins * chunk_w,
                                0.0f);
        for (int c = 0; c < n_chunks; ++c) {
            const int start = c * chunk_frames;
            const int len   = std::min(chunk_frames, frames - start);
            float* row = slab.data() +
                         static_cast<std::size_t>(c) * num_mel_bins * chunk_w;
            for (int t = 0; t < len; ++t)
                for (int m = 0; m < num_mel_bins; ++m)
                    row[static_cast<std::size_t>(m) * chunk_w + t] =
                        mel[static_cast<std::size_t>(start + t) * num_mel_bins + m];
        }
        x = bt::Tensor::from_host_on(dev, slab.data(), n_chunks,
                                     num_mel_bins * chunk_w);
    }

    // 3x (3x3, stride 2, pad 1) + exact GELU. H: n_mels -> n_mels/8 (rounded
    // up per layer); W: chunk_w -> t_out.
    int H = num_mel_bins, W = chunk_w;
    bt::Tensor y;
    bt::conv2d_forward(x, conv1_w, &conv1_b, n_chunks, 1, H, W,
                       conv_hidden, 3, 3, 2, 2, 1, 1, 1, 1, y);
    H = conv_len(H); W = conv_len(W);
    bt::gelu_exact_forward(y, y);
    bt::conv2d_forward(y, conv2_w, &conv2_b, n_chunks, conv_hidden, H, W,
                       conv_hidden, 3, 3, 2, 2, 1, 1, 1, 1, x);
    H = conv_len(H); W = conv_len(W);
    bt::gelu_exact_forward(x, x);
    bt::conv2d_forward(x, conv3_w, &conv3_b, n_chunks, conv_hidden, H, W,
                       conv_hidden, 3, 3, 2, 2, 1, 1, 1, 1, y);
    H = conv_len(H); W = conv_len(W);
    bt::gelu_exact_forward(y, y);

    // conv_out as a full-height (H x 1) conv: (n_chunks, d_model*1*W), then
    // NCHW -> sequence gives rows ordered (chunk, t) — the concatenated token
    // stream — with no host round-trip.
    bt::conv2d_forward(y, conv_out_w, nullptr, n_chunks, conv_hidden, H, W,
                       d_model, H, 1, 1, 1, 0, 0, 1, 1, x);
    bt::Tensor hs;
    bt::nchw_to_sequence(x, n_chunks, d_model, 1, W, hs);   // (n_chunks*W, d_model)

    // ── per-chunk sinusoidal positions (each chunk restarts at 0) ──
    {
        std::vector<std::int32_t> idx(static_cast<std::size_t>(n_chunks) * t_out);
        for (int c = 0; c < n_chunks; ++c)
            for (int t = 0; t < t_out; ++t)
                idx[static_cast<std::size_t>(c) * t_out + t] = t;
        bt::Tensor pos = qtd::gather_rows(pos_table, idx);
        bt::add_inplace_batched(hs, pos);
    }

    // Drop the tail chunk's dead (zero-pad-born) tokens: the valid rows are a
    // contiguous prefix, so a view re-lengths the token stream in place.
    bt::Tensor valid = bt::Tensor::view(dev, hs.data, n_valid, d_model,
                                        bt::Dtype::FP32);

    // ── block-diagonal attention windows ──
    // n_window_infer mel frames per window => t_out * (n_window_infer /
    // chunk_frames) tokens; the last window keeps the remainder.
    const int window_tokens = t_out * (n_window_infer / chunk_frames);
    std::vector<std::int32_t> cu;
    cu.push_back(0);
    for (int p = window_tokens; p < n_valid; p += window_tokens)
        cu.push_back(p);
    cu.push_back(n_valid);
    const int n_windows = static_cast<int>(cu.size()) - 1;
    const int max_seqlen = std::min(window_tokens, n_valid);
    bt::Tensor cu_dev = qtd::upload_idx(dev, cu.data(),
                                        static_cast<int>(cu.size()));
    const auto* cu_ptr = static_cast<const std::int32_t*>(cu_dev.data);

    // ── transformer stack (pre-LN, bidirectional windowed MHA, GELU FFN) ──
    const int head_dim = d_model / num_heads;
    constexpr float kLnEps = 1e-5f;   // nn.LayerNorm default
    for (const QwenAsrEncoderLayer& el : layers) {
        bt::Tensor normed;
        bt::layernorm_forward_inference_batched(valid, el.ln1_w, el.ln1_b,
                                                normed, kLnEps);
        bt::Tensor q, k, v;
        qtd::linear(el.qw, &el.qb, normed, q);
        qtd::linear(el.kw, &el.kb, normed, k);
        qtd::linear(el.vw, &el.vb, normed, v);
        bt::Tensor ctx;
        bt::flash_attention_varlen_forward(q, k, v, cu_ptr, cu_ptr, n_windows,
                                           max_seqlen, max_seqlen, num_heads,
                                           head_dim, /*causal=*/false, ctx);
        bt::Tensor attn;
        qtd::linear(el.ow, &el.ob, ctx, attn);
        bt::add_inplace_batched(valid, attn);

        bt::Tensor n2;
        bt::layernorm_forward_inference_batched(valid, el.ln2_w, el.ln2_b,
                                                n2, kLnEps);
        bt::Tensor h1;
        qtd::linear(el.fc1_w, &el.fc1_b, n2, h1);
        bt::gelu_exact_forward(h1, h1);
        bt::Tensor h2;
        qtd::linear(el.fc2_w, &el.fc2_b, h1, h2);
        bt::add_inplace_batched(valid, h2);
    }

    // ── final norm + projector into the decoder width ──
    bt::Tensor post;
    bt::layernorm_forward_inference_batched(valid, ln_post_w, ln_post_b,
                                            post, kLnEps);
    bt::Tensor p1;
    qtd::linear(proj1_w, &proj1_b, post, p1);
    bt::gelu_exact_forward(p1, p1);
    qtd::linear(proj2_w, &proj2_b, p1, out);   // (n_valid, output_dim)
}

}  // namespace brosoundml
