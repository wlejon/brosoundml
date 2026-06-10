#include "parakeet_modules.h"

#include "mel_slaney.h"
#include "qwen_tts_device.h"   // qtd:: device-neutral helpers

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
    throw std::runtime_error("brosoundml: ParakeetEncoder: " + msg);
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Upload a weight to FP32 on `dev` (widening F16/BF16 host-side).
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
// Optional upload: returns an empty tensor when the key is absent.
bt::Tensor up_opt(const sf::File& f, const std::string& name, int rows, int cols,
                  bt::Device dev) {
    if (!f.find(name)) return bt::Tensor{};
    return up(f, name, rows, cols, dev);
}

// Mel front-end constants (NeMo FastConformer AudioToMelSpectrogram).
constexpr int   kNFft       = 512;
constexpr int   kWinLength  = 400;    // 25 ms @ 16 kHz
constexpr int   kHopLength  = 160;    // 10 ms @ 16 kHz
constexpr int   kNBins      = kNFft / 2 + 1;   // 257
constexpr float kPreemph    = 0.97f;
constexpr float kLogGuard   = 5.9604645e-08f;  // 2^-24, NeMo log_zero_guard add
constexpr float kNormEps    = 1e-5f;           // NeMo normalize_batch CONSTANT
constexpr float kLnEps      = 1e-5f;           // nn.LayerNorm default
constexpr float kBnEps      = 1e-5f;           // nn.BatchNorm1d default

// Output length of one conv axis (kernel 3, stride 2, pad 1): (n - 1)/2 + 1.
int conv_len(int n) { return (n - 1) / 2 + 1; }

// Build the Transformer-XL relative-position additive bias for one attention
// layer, fully on the host. matrix_bd[q,j] = Qv_h[q] . p_h[j] (Qv already
// scaled by 1/sqrt(head_dim)); the NeMo rel_shift then maps it to
//   bias[h, q, k] = matrix_bd[q, (T-1-q) + k].
// Returns a row-major (num_heads*T, T) FP32 buffer (row h*T+q is head h,
// query q). Qv_host: (T, C); p_host: (2T-1, C), C = num_heads*head_dim.
std::vector<float> build_rel_pos_bias(const std::vector<float>& Qv_host,
                                      const std::vector<float>& p_host,
                                      int T, int num_heads, int head_dim) {
    const int C = num_heads * head_dim;
    const int P = 2 * T - 1;
    std::vector<float> bias(static_cast<std::size_t>(num_heads) * T * T, 0.0f);
    for (int h = 0; h < num_heads; ++h) {
        const int co = h * head_dim;
        for (int q = 0; q < T; ++q) {
            const float* qv = Qv_host.data() +
                              static_cast<std::size_t>(q) * C + co;
            const int base = (T - 1 - q);   // column offset of k=0 in matrix_bd
            float* out = bias.data() +
                         (static_cast<std::size_t>(h) * T + q) * T;
            for (int k = 0; k < T; ++k) {
                const int j = base + k;     // in [0, P)
                const float* pj = p_host.data() +
                                  static_cast<std::size_t>(j) * C + co;
                float s = 0.0f;
                for (int d = 0; d < head_dim; ++d) s += qv[d] * pj[d];
                out[k] = s;
            }
        }
    }
    (void)P;
    return bias;
}

}  // namespace

// ─── ParakeetConformerBlock::forward ───────────────────────────────────────

void ParakeetConformerBlock::forward(bt::Tensor& x,
                                     const ParakeetEncoderConfig& cfg,
                                     const bt::Tensor& pos_emb) const {
    const bt::Device dev = x.device;
    const int T   = x.rows;
    const int C   = cfg.hidden_size;
    const int nh  = cfg.num_attention_heads;
    const int dk  = C / nh;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(dk));

    // ── ½-FFN macaron #1 ────────────────────────────────────────────────────
    {
        bt::Tensor xn;
        bt::layernorm_forward_inference_batched(x, n_ff1_w, n_ff1_b, xn, kLnEps);
        bt::Tensor h; qtd::linear(ff1_l1_w, nullptr, xn, h);   // (T, inter)
        bt::silu_forward(h, h);
        bt::Tensor o; qtd::linear(ff1_l2_w, nullptr, h, o);    // (T, C)
        bt::scale_inplace(o, 0.5f);
        bt::add_inplace_batched(x, o);
    }

    // ── Relative-position self-attention ────────────────────────────────────
    {
        bt::Tensor xn;
        bt::layernorm_forward_inference_batched(x, n_att_w, n_att_b, xn, kLnEps);

        // Position term query: Qv = (xn @ q_w^T + bias_v) * 1/sqrt(dk).
        bt::Tensor Qv; qtd::linear(q_w, &bias_v, xn, Qv);      // (T, C)
        bt::scale_inplace(Qv, attn_scale);
        // Relative key projection of the positional encoding.
        bt::Tensor p; qtd::linear(rel_k_w, nullptr, pos_emb, p);  // (2T-1, C)

        // rel_shift bias on host, then upload.
        std::vector<float> Qv_h(static_cast<std::size_t>(T) * C);
        std::vector<float> p_h(static_cast<std::size_t>(2 * T - 1) * C);
        qtd::to_host(Qv, Qv_h.data());
        qtd::to_host(p,  p_h.data());
        std::vector<float> bias_h = build_rel_pos_bias(Qv_h, p_h, T, nh, dk);
        bt::Tensor attn_bias = bt::Tensor::from_host_on(dev, bias_h.data(),
                                                        nh * T, T);

        // Content term folds bias_u into the q projection bias; scale applies
        // 1/sqrt(dk) to the QK dot. attn_bias carries the position term.
        bt::Tensor O = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        bt::self_attention_bias_forward(xn, q_w, k_w, v_w, o_w,
                                        &bias_u, nullptr, nullptr, nullptr,
                                        /*d_mask=*/nullptr, &attn_bias,
                                        nh, attn_scale, O);
        bt::add_inplace_batched(x, O);
    }

    // ── Convolution module ──────────────────────────────────────────────────
    {
        bt::Tensor xn;
        bt::layernorm_forward_inference_batched(x, n_conv_w, n_conv_b, xn, kLnEps);
        // (T, C) -> NCL (1, C*T).
        bt::Tensor xt; bt::sequence_to_nchw(xn, 1, C, 1, T, xt);

        // pointwise_conv1: C -> 2C, then GLU (a * sigmoid(b)) over the channel
        // split. In NCL (1, 2C*T) channel c occupies [c*T, (c+1)*T), so the
        // first C channels (a) and next C channels (b) are contiguous halves.
        bt::Tensor u;
        bt::conv1d(xt, pw1_w, nullptr, 1, C, T, 2 * C, 1, 1, 0, 1, 1, u);
        bt::Tensor a = bt::Tensor::view(dev, u.data, C, T, bt::Dtype::FP32);
        bt::Tensor b = bt::Tensor::view(
            dev, static_cast<float*>(u.data) + static_cast<std::size_t>(C) * T,
            C, T, bt::Dtype::FP32);
        bt::Tensor bs = bt::Tensor::empty_on(dev, C, T, bt::Dtype::FP32);
        bt::sigmoid_forward(b, bs);
        bt::Tensor glu = a.clone();
        bt::mul_inplace(glu, bs);                              // (C, T)

        // depthwise_conv (k, groups=C, 'same' padding) -> BatchNorm -> SiLU.
        const int k   = cfg.conv_kernel_size;
        const int pad = (k - 1) / 2;
        bt::Tensor glu_ncl =
            bt::Tensor::view(dev, glu.data, 1, C * T, bt::Dtype::FP32);
        bt::Tensor dw;
        bt::conv1d(glu_ncl, dw_w, nullptr, 1, C, T, C, k, 1, pad, 1, C, dw);
        bt::Tensor bn;
        bt::batch_norm_inference(dw, bn_w, bn_b, bn_mean, bn_var,
                                 1, C, 1, T, kBnEps, bn);
        bt::silu_forward(bn, bn);

        // pointwise_conv2: C -> C, then back to (T, C).
        bt::Tensor pw2;
        bt::conv1d(bn, pw2_w, nullptr, 1, C, T, C, 1, 1, 0, 1, 1, pw2);
        bt::Tensor co; bt::nchw_to_sequence(pw2, 1, C, 1, T, co);   // (T, C)
        bt::add_inplace_batched(x, co);
    }

    // ── ½-FFN macaron #2 ────────────────────────────────────────────────────
    {
        bt::Tensor xn;
        bt::layernorm_forward_inference_batched(x, n_ff2_w, n_ff2_b, xn, kLnEps);
        bt::Tensor h; qtd::linear(ff2_l1_w, nullptr, xn, h);
        bt::silu_forward(h, h);
        bt::Tensor o; qtd::linear(ff2_l2_w, nullptr, h, o);
        bt::scale_inplace(o, 0.5f);
        bt::add_inplace_batched(x, o);
    }

    // ── Final LayerNorm ─────────────────────────────────────────────────────
    {
        bt::Tensor xn;
        bt::layernorm_forward_inference_batched(x, n_out_w, n_out_b, xn, kLnEps);
        x = std::move(xn);
    }
}

// ─── ParakeetEncoder::load ─────────────────────────────────────────────────

void ParakeetEncoder::load(const sf::File& f, const ParakeetEncoderConfig& c,
                           bt::Device dev) {
    cfg    = c;
    device = dev;

    if (cfg.hidden_size % cfg.num_attention_heads != 0)
        fail("hidden_size not divisible by num_attention_heads");
    if (cfg.subsampling_factor != 8)
        fail("only subsampling_factor 8 is supported");

    // ── mel tables (host-resident, closed-form) ──
    {
        std::vector<float> fb =
            melslaney::build_filterbank(cfg.num_mel_bins, kNFft, 16000);
        mel_filter = bt::Tensor::zeros_on(bt::Device::CPU, cfg.num_mel_bins,
                                          kNBins, bt::Dtype::FP32);
        std::memcpy(mel_filter.host_f32_mut(), fb.data(),
                    fb.size() * sizeof(float));
        std::vector<float> hw = melslaney::build_hann_window(kWinLength);
        window = bt::Tensor::zeros_on(bt::Device::CPU, 1, kWinLength,
                                      bt::Dtype::FP32);
        std::memcpy(window.host_f32_mut(), hw.data(), hw.size() * sizeof(float));
    }

    // ── subsampling conv stack ──
    // ModuleList layers indices: 0 = Conv2d(1->ch), 1 = ReLU, 2 = dw, 3 = pw,
    // 4 = ReLU, 5 = dw, 6 = pw, 7 = ReLU. Conv weights at 0,2,3,5,6.
    const int ch = cfg.subsampling_conv_channels;
    const int kk = cfg.subsampling_conv_kernel_size;   // 3
    const std::string sp = "encoder.subsampling.";
    const int conv_idx[5] = {0, 2, 3, 5, 6};
    // (rows, cols) per conv weight in OIHW-flat form.
    const int conv_cols[5] = {1 * kk * kk, 1 * kk * kk, ch * 1 * 1,
                              1 * kk * kk, ch * 1 * 1};
    for (int i = 0; i < 5; ++i) {
        const std::string base =
            sp + "layers." + std::to_string(conv_idx[i]) + ".";
        sub_conv_w[i] = up(f, base + "weight", ch, conv_cols[i], dev);
        sub_conv_b[i] = up_opt(f, base + "bias", ch, 1, dev);
    }
    // Final linear: (channels * freq_out) -> hidden_size. freq_out = mel after
    // three stride-2 conv axes.
    const int freq_out = conv_len(conv_len(conv_len(cfg.num_mel_bins)));
    sub_linear_w = up(f, sp + "linear.weight", cfg.hidden_size, ch * freq_out, dev);
    sub_linear_b = up_vec(f, sp + "linear.bias", cfg.hidden_size, dev);

    // ── conformer blocks ──
    const int C = cfg.hidden_size;
    const int inter = cfg.intermediate_size;
    layers.clear();
    layers.resize(static_cast<std::size_t>(cfg.num_hidden_layers));
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string L = "encoder.layers." + std::to_string(i) + ".";
        ParakeetConformerBlock& bl = layers[static_cast<std::size_t>(i)];

        bl.n_ff1_w  = up_vec(f, L + "norm_feed_forward1.weight", C, dev);
        bl.n_ff1_b  = up_vec(f, L + "norm_feed_forward1.bias",   C, dev);
        bl.n_att_w  = up_vec(f, L + "norm_self_att.weight",      C, dev);
        bl.n_att_b  = up_vec(f, L + "norm_self_att.bias",        C, dev);
        bl.n_conv_w = up_vec(f, L + "norm_conv.weight",          C, dev);
        bl.n_conv_b = up_vec(f, L + "norm_conv.bias",            C, dev);
        bl.n_ff2_w  = up_vec(f, L + "norm_feed_forward2.weight", C, dev);
        bl.n_ff2_b  = up_vec(f, L + "norm_feed_forward2.bias",   C, dev);
        bl.n_out_w  = up_vec(f, L + "norm_out.weight",           C, dev);
        bl.n_out_b  = up_vec(f, L + "norm_out.bias",             C, dev);

        bl.ff1_l1_w = up(f, L + "feed_forward1.linear1.weight", inter, C, dev);
        bl.ff1_l2_w = up(f, L + "feed_forward1.linear2.weight", C, inter, dev);
        bl.ff2_l1_w = up(f, L + "feed_forward2.linear1.weight", inter, C, dev);
        bl.ff2_l2_w = up(f, L + "feed_forward2.linear2.weight", C, inter, dev);

        bl.q_w     = up(f, L + "self_attn.q_proj.weight", C, C, dev);
        bl.k_w     = up(f, L + "self_attn.k_proj.weight", C, C, dev);
        bl.v_w     = up(f, L + "self_attn.v_proj.weight", C, C, dev);
        bl.o_w     = up(f, L + "self_attn.o_proj.weight", C, C, dev);
        bl.rel_k_w = up(f, L + "self_attn.relative_k_proj.weight", C, C, dev);
        bl.bias_u  = up_vec(f, L + "self_attn.bias_u", C, dev);   // (nh,dk)->(C,1)
        bl.bias_v  = up_vec(f, L + "self_attn.bias_v", C, dev);

        bl.pw1_w = up(f, L + "conv.pointwise_conv1.weight", 2 * C, C, dev);
        bl.dw_w  = up(f, L + "conv.depthwise_conv.weight", C, cfg.conv_kernel_size, dev);
        bl.pw2_w = up(f, L + "conv.pointwise_conv2.weight", C, C, dev);
        bl.bn_w    = up_vec(f, L + "conv.norm.weight",       C, dev);
        bl.bn_b    = up_vec(f, L + "conv.norm.bias",         C, dev);
        bl.bn_mean = up_vec(f, L + "conv.norm.running_mean", C, dev);
        bl.bn_var  = up_vec(f, L + "conv.norm.running_var",  C, dev);
    }
}

// ─── ParakeetEncoder::log_mel ──────────────────────────────────────────────

std::vector<float> ParakeetEncoder::log_mel(const AudioBuffer& audio,
                                            int& frames_out) const {
    if (mel_filter.rows == 0) fail("load() not called");
    if (audio.sample_rate != 16000)
        fail("audio sample_rate " + std::to_string(audio.sample_rate) +
             " != the model's required 16000");
    const int n_mels = cfg.num_mel_bins;
    const int L = static_cast<int>(audio.samples.size());
    if (L < kWinLength) fail("audio shorter than one analysis window (25 ms)");

    bt::DeviceScope cpu(bt::Device::CPU);

    // Pre-emphasis: y[0]=x[0]; y[n]=x[n]-0.97*x[n-1].
    std::vector<float> pre(static_cast<std::size_t>(L));
    pre[0] = audio.samples[0];
    for (int n = 1; n < L; ++n)
        pre[static_cast<std::size_t>(n)] =
            audio.samples[static_cast<std::size_t>(n)] -
            kPreemph * audio.samples[static_cast<std::size_t>(n - 1)];

    bt::Tensor signal = bt::Tensor::from_host_on(bt::Device::CPU, pre.data(), 1, L);
    bt::Tensor spec;
    bt::stft(signal, window, /*N=*/1, kNFft, kHopLength, kWinLength,
             /*center=*/true, /*normalized=*/false, spec);
    const int frames = spec.rows;
    if (frames < 1) fail("stft produced no frames");

    // Power spectrum.
    bt::Tensor power = bt::Tensor::zeros_on(bt::Device::CPU, frames, kNBins,
                                            bt::Dtype::FP32);
    {
        const float* sd = spec.host_f32();
        float*       pd = power.host_f32_mut();
        for (int fr = 0; fr < frames; ++fr) {
            const float* src = sd + static_cast<std::size_t>(fr) * 2 * kNBins;
            float*       dst = pd + static_cast<std::size_t>(fr) * kNBins;
            for (int kb = 0; kb < kNBins; ++kb) {
                const float re = src[2 * kb + 0];
                const float im = src[2 * kb + 1];
                dst[kb] = re * re + im * im;
            }
        }
    }

    // Mel projection -> (frames, n_mels).
    bt::Tensor mel_ft;
    {
        bt::Tensor fb_T = bt::Tensor::zeros_on(bt::Device::CPU, kNBins, n_mels,
                                               bt::Dtype::FP32);
        const float* src = mel_filter.host_f32();
        float*       dst = fb_T.host_f32_mut();
        for (int m = 0; m < n_mels; ++m)
            for (int kb = 0; kb < kNBins; ++kb)
                dst[static_cast<std::size_t>(kb) * n_mels + m] =
                    src[static_cast<std::size_t>(m) * kNBins + kb];
        bt::matmul(power, fb_T, mel_ft);
    }

    // log(x + 2^-24).
    const std::size_t total = static_cast<std::size_t>(frames) * n_mels;
    std::vector<float> out(total);
    const float* md = mel_ft.host_f32();
    for (std::size_t i = 0; i < total; ++i)
        out[i] = std::log(md[i] + kLogGuard);

    // Per-feature (per-mel-bin) mean/var normalization over time. NeMo uses an
    // unbiased std plus a 1e-5 floor.
    for (int m = 0; m < n_mels; ++m) {
        double mean = 0.0;
        for (int fr = 0; fr < frames; ++fr)
            mean += out[static_cast<std::size_t>(fr) * n_mels + m];
        mean /= frames;
        double var = 0.0;
        for (int fr = 0; fr < frames; ++fr) {
            const double d = out[static_cast<std::size_t>(fr) * n_mels + m] - mean;
            var += d * d;
        }
        var /= std::max(1, frames - 1);
        const float inv = 1.0f / (static_cast<float>(std::sqrt(var)) + kNormEps);
        const float mf  = static_cast<float>(mean);
        for (int fr = 0; fr < frames; ++fr) {
            float& v = out[static_cast<std::size_t>(fr) * n_mels + m];
            v = (v - mf) * inv;
        }
    }

    frames_out = frames;
    return out;
}

// ─── ParakeetEncoder::forward ──────────────────────────────────────────────

void ParakeetEncoder::forward(const AudioBuffer& audio, bt::Tensor& out) const {
    const bt::Device dev = sub_linear_w.device;
    bt::DeviceScope scope(dev);

    const int C      = cfg.hidden_size;
    const int n_mels = cfg.num_mel_bins;
    const int ch     = cfg.subsampling_conv_channels;
    const int kk     = cfg.subsampling_conv_kernel_size;

    int frames = 0;
    const std::vector<float> mel = log_mel(audio, frames);   // (frames, n_mels)

    // Assemble NCHW (1, 1*n_mels*frames): H = mel (freq), W = time.
    bt::Tensor x;
    {
        std::vector<float> slab(static_cast<std::size_t>(n_mels) * frames);
        for (int t = 0; t < frames; ++t)
            for (int m = 0; m < n_mels; ++m)
                slab[static_cast<std::size_t>(m) * frames + t] =
                    mel[static_cast<std::size_t>(t) * n_mels + m];
        x = bt::Tensor::from_host_on(dev, slab.data(), 1, n_mels * frames);
    }

    // 8x depthwise-separable conv subsampling. layers: full conv, then 2x
    // (depthwise, pointwise), each followed by ReLU.
    auto bias_ptr = [](const bt::Tensor& b) -> const bt::Tensor* {
        return b.rows > 0 ? &b : nullptr;
    };
    int H = n_mels, W = frames;
    bt::Tensor y;
    // layers[0]: Conv2d(1->ch, k, s2, p1) + ReLU.
    bt::conv2d_forward(x, sub_conv_w[0], bias_ptr(sub_conv_b[0]),
                       1, 1, H, W, ch, kk, kk, 2, 2, 1, 1, 1, 1, y);
    H = conv_len(H); W = conv_len(W);
    bt::relu_forward(y, y);
    // layers[2,3]: depthwise + pointwise + ReLU.
    bt::conv2d_forward(y, sub_conv_w[1], bias_ptr(sub_conv_b[1]),
                       1, ch, H, W, ch, kk, kk, 2, 2, 1, 1, 1, ch, x);
    H = conv_len(H); W = conv_len(W);
    bt::conv2d_forward(x, sub_conv_w[2], bias_ptr(sub_conv_b[2]),
                       1, ch, H, W, ch, 1, 1, 1, 1, 0, 0, 1, 1, y);
    bt::relu_forward(y, y);
    // layers[5,6]: depthwise + pointwise + ReLU.
    bt::conv2d_forward(y, sub_conv_w[3], bias_ptr(sub_conv_b[3]),
                       1, ch, H, W, ch, kk, kk, 2, 2, 1, 1, 1, ch, x);
    H = conv_len(H); W = conv_len(W);
    bt::conv2d_forward(x, sub_conv_w[4], bias_ptr(sub_conv_b[4]),
                       1, ch, H, W, ch, 1, 1, 1, 1, 0, 0, 1, 1, y);
    bt::relu_forward(y, y);

    // Flatten (C=ch, F=H, T=W) -> (T, ch*H) and project to d_model. The conv
    // output buffer (1, ch*H*W) reinterpreted as (1, ch*H, 1, W) transposes via
    // nchw_to_sequence to (W, ch*H) — exactly NeMo's transpose(1,2).reshape.
    const int T = W;
    bt::Tensor seq;
    bt::nchw_to_sequence(y, 1, ch * H, 1, T, seq);          // (T, ch*H)
    bt::Tensor h0;
    qtd::linear(sub_linear_w, &sub_linear_b, seq, h0);      // (T, C)

    // Relative positional encoding (shared across layers). position_ids run
    // (T-1) .. -(T-1); pe[idx, 2i]=sin(pos*inv_freq_i), pe[idx,2i+1]=cos.
    bt::Tensor pos_emb;
    {
        const int P    = 2 * T - 1;
        const int half = C / 2;
        std::vector<float> pe(static_cast<std::size_t>(P) * C);
        for (int idx = 0; idx < P; ++idx) {
            const double pos = static_cast<double>(T - 1 - idx);
            float* row = pe.data() + static_cast<std::size_t>(idx) * C;
            for (int i = 0; i < half; ++i) {
                const double inv =
                    std::exp(-(2.0 * i / C) * std::log(10000.0));
                const double ang = pos * inv;
                row[2 * i]     = static_cast<float>(std::sin(ang));
                row[2 * i + 1] = static_cast<float>(std::cos(ang));
            }
        }
        pos_emb = bt::Tensor::from_host_on(dev, pe.data(), P, C);
    }

    // Conformer stack.
    bt::Tensor x_seq = std::move(h0);
    for (const ParakeetConformerBlock& bl : layers)
        bl.forward(x_seq, cfg, pos_emb);

    out = std::move(x_seq);   // (T, hidden_size)
}

}  // namespace brosoundml
