#include "qwen_tts_speaker_encoder.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {
namespace {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;

constexpr double kPi = 3.14159265358979323846;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: qwen-tts speaker encoder: " + msg);
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Load a Conv1d (`prefix`.weight / `prefix`.bias). Speaker-encoder weights are
// BF16 on disk (like the talker); upload_compute_checked widens to the CPU
// compute dtype (FP32). The FP32 weights are then moved to `dev` — keeping
// FP32 on the GPU (conv2d_forward has an FP32 path) so the convolutions match
// the CPU reference bit-for-bit rather than narrowing to FP16.
QwenTtsSpkConv load_conv(const sf::File& f, const std::string& prefix,
                         int cin, int cout, int k, int dilation, bt::Device dev) {
    QwenTtsSpkConv c;
    c.cin = cin; c.cout = cout; c.k = k; c.dilation = dilation;
    {
        bt::DeviceScope cpu(bt::Device::CPU);
        sf::upload_compute_checked(need(f, prefix + ".weight"), cout, cin * k, c.w,
                                   prefix + ".weight");
        sf::upload_compute_checked(need(f, prefix + ".bias"), cout, 1, c.b,
                                   prefix + ".bias");
    }
    if (dev != bt::Device::CPU) {
        c.w = c.w.to(dev);
        c.b = c.b.to(dev);
    }
    return c;
}

// 1D conv with torch padding="same", padding_mode="reflect": reflect-pad
// dilation*(k-1)/2 each side (no-op for k==1), then a valid conv. Input/output
// are channel-major host buffers (C*L). Returns the output (cout * L).
std::vector<float> conv_same(const QwenTtsSpkConv& c, const std::vector<float>& x,
                             int L) {
    const bt::Device dev = c.w.device;   // weights pin the compute device
    bt::DeviceScope scope(dev);
    const int p = c.dilation * (c.k - 1) / 2;
    bt::Tensor X = bt::Tensor::from_host_on(dev, x.data(), 1, c.cin * L);
    bt::Tensor Xp;
    if (p > 0) bt::pad1d_forward(X, 1, c.cin, L, p, p, /*mode=*/1, Xp);  // reflect
    const bt::Tensor& in = (p > 0) ? Xp : X;
    const int Lp = L + 2 * p;
    bt::Tensor Y;
    bt::conv1d(in, c.w, &c.b, /*N=*/1, c.cin, Lp, c.cout, c.k, /*stride=*/1,
               /*padding=*/0, c.dilation, Y);
    const int Lout = Lp - c.dilation * (c.k - 1);   // == L
    bt::Tensor Yh = (Y.device == bt::Device::CPU) ? std::move(Y)
                                                  : Y.to(bt::Device::CPU);
    const float* yp = Yh.host_f32();
    return std::vector<float>(yp, yp + static_cast<std::size_t>(c.cout) * Lout);
}

void relu_(std::vector<float>& x) {
    for (float& v : x) if (v < 0.0f) v = 0.0f;
}

// Res2Net: split the C channels into `scale` equal chunks along the channel
// axis; chunk 0 passes through, chunk i (>=1) runs sub-conv[i-1] over (chunk_i +
// running_output), accumulating. Channel-major (C*L) in/out.
std::vector<float> res2net(const QwenTtsSpkSERes2& blk, const std::vector<float>& x,
                           int C, int L, int scale) {
    const int hc = C / scale;                 // hidden channels per chunk (64)
    std::vector<float> out(static_cast<std::size_t>(C) * L);
    std::vector<float> prev;                  // previous chunk's output (hc*L)
    for (int i = 0; i < scale; ++i) {
        const float* chunk = x.data() + static_cast<std::size_t>(i) * hc * L;
        std::vector<float> part;
        if (i == 0) {
            part.assign(chunk, chunk + static_cast<std::size_t>(hc) * L);
        } else {
            std::vector<float> in(static_cast<std::size_t>(hc) * L);
            if (i == 1) {
                std::copy(chunk, chunk + in.size(), in.begin());
            } else {
                for (std::size_t j = 0; j < in.size(); ++j) in[j] = chunk[j] + prev[j];
            }
            part = conv_same(blk.res2net[i - 1], in, L);
            relu_(part);
        }
        std::copy(part.begin(), part.end(),
                  out.begin() + static_cast<std::size_t>(i) * hc * L);
        prev = std::move(part);
    }
    return out;
}

// Squeeze-excitation: per-channel time-mean -> 1x1 conv(C->se)+ReLU -> 1x1
// conv(se->C)+sigmoid -> per-channel gate broadcast over time.
std::vector<float> se_block(const QwenTtsSpkSERes2& blk, const std::vector<float>& x,
                            int C, int L) {
    std::vector<float> m(C, 0.0f);
    for (int c = 0; c < C; ++c) {
        const float* row = x.data() + static_cast<std::size_t>(c) * L;
        double s = 0.0;
        for (int t = 0; t < L; ++t) s += row[t];
        m[c] = static_cast<float>(s / L);
    }
    std::vector<float> g = conv_same(blk.se1, m, /*L=*/1);   // (se,)
    relu_(g);
    g = conv_same(blk.se2, g, /*L=*/1);                      // (C,)
    for (float& v : g) v = 1.0f / (1.0f + std::exp(-v));     // sigmoid
    std::vector<float> out(x.size());
    for (int c = 0; c < C; ++c)
        for (int t = 0; t < L; ++t)
            out[static_cast<std::size_t>(c) * L + t] =
                x[static_cast<std::size_t>(c) * L + t] * g[c];
    return out;
}

}  // namespace

void QwenTtsSpeakerEncoder::load(const sf::File& f,
                                 const QwenTtsSpeakerEncoderConfig& c) {
    cfg = c;
    // Run the convolution stack on the default device (GPU when available); the
    // mel frontend tensors below stay on CPU. brotensor::init() has already run
    // (SpeakerEncoder::load calls it), so default_device() is resolved.
    device = bt::default_device();
    const bt::Device dev = device;
    const std::string P = "speaker_encoder.";
    const std::vector<int>& ch  = cfg.enc_channels;       // [512,512,512,512,1536]
    const std::vector<int>& ks  = cfg.enc_kernel_sizes;   // [5,3,3,3,1]
    const std::vector<int>& dil = cfg.enc_dilations;      // [1,2,3,4,1]
    const int nblk = static_cast<int>(ch.size());         // 5
    const int agg  = ch.back();                           // 1536
    const int scale = cfg.res2net_scale;

    // Initial TDNN: mel_dim -> ch[0], k = ks[0].
    block0 = load_conv(f, P + "blocks.0.conv", cfg.mel_dim, ch[0], ks[0], dil[0], dev);

    // Three SE-Res2Net blocks (indices 1 .. nblk-2 == 1,2,3).
    blocks.clear();
    for (int i = 1; i < nblk - 1; ++i) {
        const std::string b = P + "blocks." + std::to_string(i) + ".";
        QwenTtsSpkSERes2 s;
        s.tdnn1 = load_conv(f, b + "tdnn1.conv", ch[i - 1], ch[i], 1, 1, dev);
        const int hc = ch[i] / scale;
        for (int r = 0; r < scale - 1; ++r)
            s.res2net.push_back(load_conv(
                f, b + "res2net_block.blocks." + std::to_string(r) + ".conv",
                hc, hc, ks[i], dil[i], dev));
        s.tdnn2 = load_conv(f, b + "tdnn2.conv", ch[i], ch[i], 1, 1, dev);
        s.se1 = load_conv(f, b + "se_block.conv1", ch[i], cfg.se_channels, 1, 1, dev);
        s.se2 = load_conv(f, b + "se_block.conv2", cfg.se_channels, ch[i], 1, 1, dev);
        blocks.push_back(std::move(s));
    }

    // Aggregation, attentive stats pooling, final projection.
    mfa      = load_conv(f, P + "mfa.conv", agg, agg, ks.back(), dil.back(), dev);
    asp_tdnn = load_conv(f, P + "asp.tdnn.conv", agg * 3, cfg.attention_channels, 1, 1, dev);
    asp_conv = load_conv(f, P + "asp.conv", cfg.attention_channels, agg, 1, 1, dev);
    fc       = load_conv(f, P + "fc", agg * 2, cfg.enc_dim, 1, 1, dev);

    // ── mel frontend constants ──
    bt::DeviceScope cpu(bt::Device::CPU);
    // Periodic Hann window (torch.hann_window default).
    std::vector<float> w(cfg.win_size);
    for (int n = 0; n < cfg.win_size; ++n)
        w[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * n / cfg.win_size));
    hann = bt::Tensor::from_host_on(bt::Device::CPU, w.data(), 1, cfg.win_size);

    // librosa slaney mel filterbank (sr, n_fft, n_mels, fmin, fmax, htk=False).
    const int nb = cfg.n_fft / 2 + 1;
    const double f_sp = 200.0 / 3.0, min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp, logstep = std::log(6.4) / 27.0;
    auto hz2mel = [&](double hz) {
        double m = hz / f_sp;
        if (hz >= min_log_hz) m = min_log_mel + std::log(hz / min_log_hz) / logstep;
        return m;
    };
    auto mel2hz = [&](double mel) {
        double hz = f_sp * mel;
        if (mel >= min_log_mel) hz = min_log_hz * std::exp(logstep * (mel - min_log_mel));
        return hz;
    };
    const int M = cfg.mel_dim;
    std::vector<double> mel_f(M + 2);
    const double mmin = hz2mel(cfg.fmin), mmax = hz2mel(cfg.fmax);
    for (int i = 0; i < M + 2; ++i)
        mel_f[i] = mel2hz(mmin + (mmax - mmin) * i / (M + 1));
    std::vector<float> basis(static_cast<std::size_t>(M) * nb, 0.0f);
    for (int i = 0; i < M; ++i) {
        const double fdl = mel_f[i + 1] - mel_f[i];
        const double fdu = mel_f[i + 2] - mel_f[i + 1];
        const double enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (int b = 0; b < nb; ++b) {
            const double freq = static_cast<double>(b) * cfg.sample_rate / cfg.n_fft;
            const double lower = (freq - mel_f[i]) / fdl;
            const double upper = (mel_f[i + 2] - freq) / fdu;
            const double tri = std::max(0.0, std::min(lower, upper));
            basis[static_cast<std::size_t>(i) * nb + b] = static_cast<float>(tri * enorm);
        }
    }
    mel_basis = bt::Tensor::from_host_on(bt::Device::CPU, basis.data(), M, nb);
}

void QwenTtsSpeakerEncoder::mel(const float* wav, int n,
                                std::vector<float>& out, int& n_frames) const {
    bt::DeviceScope cpu(bt::Device::CPU);
    const int p = (cfg.n_fft - cfg.hop_size) / 2;   // 384
    bt::Tensor sig = bt::Tensor::from_host_on(bt::Device::CPU, wav, 1, n);
    bt::Tensor sigp;
    bt::pad1d_forward(sig, 1, 1, n, p, p, /*mode=*/1, sigp);   // reflect
    const int Lp = n + 2 * p;
    bt::Tensor spec;
    bt::stft(sigp, hann, /*N=*/1, cfg.n_fft, cfg.hop_size, cfg.win_size,
             /*center=*/false, /*normalized=*/false, spec);
    const int nb = cfg.n_fft / 2 + 1;
    const int frames = 1 + (Lp - cfg.n_fft) / cfg.hop_size;
    n_frames = frames;
    const float* s = spec.host_f32();           // (frames, 2*nb) interleaved
    const float* B = mel_basis.host_f32();       // (mel_dim, nb)
    const int M = cfg.mel_dim;

    // magnitude (frames, nb)
    std::vector<float> mag(static_cast<std::size_t>(frames) * nb);
    for (int t = 0; t < frames; ++t) {
        const float* row = s + static_cast<std::size_t>(t) * 2 * nb;
        for (int b = 0; b < nb; ++b) {
            const float re = row[2 * b], im = row[2 * b + 1];
            mag[static_cast<std::size_t>(t) * nb + b] = std::sqrt(re * re + im * im + 1e-9f);
        }
    }
    // mel = basis @ mag^T, then log-compress; channel-major out[m*frames + t].
    out.assign(static_cast<std::size_t>(M) * frames, 0.0f);
    for (int m = 0; m < M; ++m) {
        const float* bw = B + static_cast<std::size_t>(m) * nb;
        for (int t = 0; t < frames; ++t) {
            const float* mg = mag.data() + static_cast<std::size_t>(t) * nb;
            double acc = 0.0;
            for (int b = 0; b < nb; ++b) acc += static_cast<double>(bw[b]) * mg[b];
            const float v = static_cast<float>(acc < 1e-5 ? 1e-5 : acc);
            out[static_cast<std::size_t>(m) * frames + t] = std::log(v);
        }
    }
}

std::vector<float> QwenTtsSpeakerEncoder::embed(const float* wav, int n) const {
    // mel -> (mel_dim, T)
    std::vector<float> x;
    int T = 0;
    mel(wav, n, x, T);

    const int scale = cfg.res2net_scale;
    const int C = cfg.enc_channels[0];      // block working width (512)
    const int agg = cfg.enc_channels.back();// 1536

    // initial TDNN + ReLU
    x = conv_same(block0, x, T);
    relu_(x);

    // three SE-Res2Net blocks; collect their outputs for aggregation
    std::vector<std::vector<float>> outs;
    std::vector<float> hs = x;
    for (const QwenTtsSpkSERes2& blk : blocks) {
        std::vector<float> y = conv_same(blk.tdnn1, hs, T); relu_(y);
        y = res2net(blk, y, C, T, scale);
        y = conv_same(blk.tdnn2, y, T); relu_(y);
        y = se_block(blk, y, C, T);
        for (std::size_t j = 0; j < y.size(); ++j) y[j] += hs[j];   // residual
        hs = y;
        outs.push_back(hs);
    }

    // aggregate (cat the 3 block outputs along channels) -> MFA + ReLU
    std::vector<float> cat(static_cast<std::size_t>(agg) * T);
    for (std::size_t b = 0; b < outs.size(); ++b)
        std::copy(outs[b].begin(), outs[b].end(),
                  cat.begin() + b * C * T);
    std::vector<float> a = conv_same(mfa, cat, T); relu_(a);   // (agg, T)

    // ── attentive statistics pooling ──
    // global mean / std per channel (uniform weights)
    std::vector<float> gmean(agg), gstd(agg);
    for (int c = 0; c < agg; ++c) {
        const float* row = a.data() + static_cast<std::size_t>(c) * T;
        double s = 0.0;
        for (int t = 0; t < T; ++t) s += row[t];
        const double mu = s / T;
        double var = 0.0;
        for (int t = 0; t < T; ++t) { const double d = row[t] - mu; var += d * d; }
        var /= T;
        gmean[c] = static_cast<float>(mu);
        gstd[c]  = static_cast<float>(std::sqrt(std::max(var, 1e-12)));
    }
    // context = cat([a, gmean.expand, gstd.expand]) -> (3*agg, T)
    std::vector<float> ctx(static_cast<std::size_t>(3) * agg * T);
    std::copy(a.begin(), a.end(), ctx.begin());
    for (int c = 0; c < agg; ++c)
        for (int t = 0; t < T; ++t) {
            ctx[(static_cast<std::size_t>(agg) + c) * T + t] = gmean[c];
            ctx[(static_cast<std::size_t>(2 * agg) + c) * T + t] = gstd[c];
        }
    // attn = asp_conv(tanh(asp_tdnn(ctx)))
    std::vector<float> att = conv_same(asp_tdnn, ctx, T); relu_(att);
    for (float& v : att) v = std::tanh(v);
    att = conv_same(asp_conv, att, T);              // (agg, T), no activation
    // softmax over time per channel
    for (int c = 0; c < agg; ++c) {
        float* row = att.data() + static_cast<std::size_t>(c) * T;
        float mx = row[0];
        for (int t = 1; t < T; ++t) mx = std::max(mx, row[t]);
        double sum = 0.0;
        for (int t = 0; t < T; ++t) { row[t] = std::exp(row[t] - mx); sum += row[t]; }
        const float inv = static_cast<float>(1.0 / sum);
        for (int t = 0; t < T; ++t) row[t] *= inv;
    }
    // attentive mean / std -> pooled (2*agg)
    std::vector<float> pooled(static_cast<std::size_t>(2) * agg);
    for (int c = 0; c < agg; ++c) {
        const float* ar = a.data() + static_cast<std::size_t>(c) * T;
        const float* wr = att.data() + static_cast<std::size_t>(c) * T;
        double mu = 0.0;
        for (int t = 0; t < T; ++t) mu += static_cast<double>(wr[t]) * ar[t];
        double var = 0.0;
        for (int t = 0; t < T; ++t) {
            const double d = ar[t] - mu;
            var += static_cast<double>(wr[t]) * d * d;
        }
        pooled[c] = static_cast<float>(mu);
        pooled[agg + c] = static_cast<float>(std::sqrt(std::max(var, 1e-12)));
    }
    // final projection (2*agg -> enc_dim)
    return conv_same(fc, pooled, /*L=*/1);
}

}  // namespace brosoundml
