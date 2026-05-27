#include "brosoundml/wake_data.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

}  // namespace

// ─── Noise generators ──────────────────────────────────────────────────────

std::vector<float> gen_white_noise(int n_samples, float amplitude,
                                   std::mt19937& rng) {
    if (n_samples < 0) fail("gen_white_noise", "n_samples < 0");
    std::vector<float> out(static_cast<std::size_t>(n_samples));
    // Box-Muller-style uniform → gaussian-ish via std::normal_distribution. We
    // want approximately unit variance before scaling so that `amplitude`
    // doubles as a peak/rms knob across callers.
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : out) v = amplitude * nd(rng);
    return out;
}

std::vector<float> gen_pink_noise(int n_samples, float amplitude,
                                  std::mt19937& rng) {
    if (n_samples < 0) fail("gen_pink_noise", "n_samples < 0");
    // Voss-McCartney with 7 octaves — approximately -3 dB/oct.
    constexpr int kOct = 7;
    float rows[kOct] = {0};
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int j = 0; j < kOct; ++j) rows[j] = nd(rng);
    std::vector<float> out(static_cast<std::size_t>(n_samples));
    const float scale = amplitude / std::sqrt(static_cast<float>(kOct));
    // Pre-roll so the filter state is warm before the first emitted sample.
    for (int i = 0; i < 32 + n_samples; ++i) {
        for (int k = 0; k < kOct; ++k) {
            const unsigned mask = (1u << k) - 1u;
            if ((static_cast<unsigned>(i) & mask) == 0) rows[k] = nd(rng);
        }
        if (i >= 32) {
            float s = 0.0f;
            for (int k = 0; k < kOct; ++k) s += rows[k];
            out[static_cast<std::size_t>(i - 32)] = scale * s;
        }
    }
    return out;
}

std::vector<float> gen_brown_noise(int n_samples, float amplitude,
                                   std::mt19937& rng) {
    if (n_samples < 0) fail("gen_brown_noise", "n_samples < 0");
    // Integrated white noise → -6 dB/oct (true brown / Brownian). Leak the
    // accumulator with a one-pole high-pass to keep DC bounded over long runs.
    std::vector<float> out(static_cast<std::size_t>(n_samples));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    constexpr float kLeak = 0.999f;
    float acc = 0.0f;
    // The integrator dramatically amplifies low frequencies; this constant
    // brings the resulting RMS roughly back to `amplitude` for n_samples in
    // the 1-second-at-16-kHz range.
    constexpr float kNormalise = 0.05f;
    for (int i = 0; i < n_samples; ++i) {
        acc = kLeak * acc + nd(rng);
        out[static_cast<std::size_t>(i)] = amplitude * kNormalise * acc;
    }
    return out;
}

const char* noise_kind_name(NoiseKind k) {
    switch (k) {
        case NoiseKind::White: return "white";
        case NoiseKind::Pink:  return "pink";
        case NoiseKind::Brown: return "brown";
    }
    return "white";
}

std::vector<float> gen_noise(NoiseKind k, int n_samples, float amplitude,
                             std::mt19937& rng) {
    switch (k) {
        case NoiseKind::White: return gen_white_noise(n_samples, amplitude, rng);
        case NoiseKind::Pink:  return gen_pink_noise (n_samples, amplitude, rng);
        case NoiseKind::Brown: return gen_brown_noise(n_samples, amplitude, rng);
    }
    return {};
}

// ─── Level helpers ─────────────────────────────────────────────────────────

float rms(const std::vector<float>& x) {
    if (x.empty()) return 0.0f;
    double acc = 0.0;
    for (auto v : x) acc += static_cast<double>(v) * v;
    return static_cast<float>(std::sqrt(acc / static_cast<double>(x.size())));
}

float peak(const std::vector<float>& x) {
    float p = 0.0f;
    for (auto v : x) {
        const float a = std::fabs(v);
        if (a > p) p = a;
    }
    return p;
}

void apply_gain_db(std::vector<float>& x, float gain_db) {
    const float g = std::pow(10.0f, gain_db / 20.0f);
    for (auto& v : x) v *= g;
}

void peak_normalize(std::vector<float>& x, float target) {
    const float p = peak(x);
    if (p <= 0.0f) return;
    const float g = target / p;
    for (auto& v : x) v *= g;
}

// ─── SNR mixer ─────────────────────────────────────────────────────────────

std::vector<float> mix_at_snr(const std::vector<float>& signal,
                              const std::vector<float>& noise,
                              float snr_db) {
    if (signal.size() != noise.size())
        fail("mix_at_snr", "signal and noise lengths differ");
    const float sig_rms = rms(signal);
    const float nse_rms = rms(noise);
    // Silent-speech / silent-noise edge cases: nothing meaningful to mix.
    if (sig_rms <= 0.0f || nse_rms <= 0.0f) return signal;
    // target noise rms = sig_rms / 10^(snr_db/20).
    const float target_nse_rms = sig_rms / std::pow(10.0f, snr_db / 20.0f);
    const float scale = target_nse_rms / nse_rms;
    std::vector<float> out(signal.size());
    for (std::size_t i = 0; i < signal.size(); ++i) {
        out[i] = signal[i] + scale * noise[i];
    }
    return out;
}

// ─── RIR ───────────────────────────────────────────────────────────────────

std::vector<float> gen_rir(int sample_rate, int rir_len, float t60_seconds,
                           int n_early_taps, std::mt19937& rng) {
    if (sample_rate <= 0) fail("gen_rir", "sample_rate <= 0");
    if (rir_len     <= 0) fail("gen_rir", "rir_len <= 0");
    if (t60_seconds <= 0) fail("gen_rir", "t60_seconds <= 0");
    std::vector<float> h(static_cast<std::size_t>(rir_len), 0.0f);
    // Direct path at sample 0 — unit gain.
    h[0] = 1.0f;
    // Exponential decay envelope: -60 dB at t60_seconds.
    //   env[n] = 10^(-3 * n / (t60 * sr))
    const float decay_per_sample =
        -3.0f / (t60_seconds * static_cast<float>(sample_rate));
    // Early reflections: a handful of randomly-placed taps in the first 50 ms.
    const int early_window = std::min(rir_len - 1,
                                      sample_rate / 20);  // 50 ms
    if (early_window > 0) {
        std::uniform_int_distribution<int>   pos_d(1, early_window);
        std::uniform_real_distribution<float> mag_d(0.2f, 0.7f);
        std::uniform_real_distribution<float> sign_d(-1.0f, 1.0f);
        for (int t = 0; t < n_early_taps; ++t) {
            const int   n = pos_d(rng);
            const float env =
                std::pow(10.0f, decay_per_sample * static_cast<float>(n));
            const float s = sign_d(rng) < 0 ? -1.0f : 1.0f;
            h[static_cast<std::size_t>(n)] += s * mag_d(rng) * env;
        }
    }
    // Late-field tail: white-noise modulated by the envelope.
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int n = std::max(1, early_window); n < rir_len; ++n) {
        const float env =
            std::pow(10.0f, decay_per_sample * static_cast<float>(n));
        h[static_cast<std::size_t>(n)] += 0.1f * env * nd(rng);
    }
    return h;
}

std::vector<float> convolve_rir(const std::vector<float>& signal,
                                const std::vector<float>& rir) {
    if (signal.empty()) fail("convolve_rir", "empty signal");
    if (rir   .empty()) fail("convolve_rir", "empty rir");
    const int N = static_cast<int>(signal.size());
    const int K = static_cast<int>(rir.size());
    std::vector<float> out(static_cast<std::size_t>(N), 0.0f);
    // Causal convolution, output truncated to input length:
    //   y[n] = sum_{k=0..min(K-1, n)} h[k] * x[n - k]
    // K is small (~3200 at 16 kHz / 200 ms) and N is ~16000, so the O(N*K)
    // direct form is fine. No FFT machinery needed.
    for (int n = 0; n < N; ++n) {
        const int k_hi = std::min(K - 1, n);
        float y = 0.0f;
        for (int k = 0; k <= k_hi; ++k) {
            y += rir[static_cast<std::size_t>(k)] *
                 signal[static_cast<std::size_t>(n - k)];
        }
        out[static_cast<std::size_t>(n)] = y;
    }
    return out;
}

// ─── Resampler ─────────────────────────────────────────────────────────────

std::vector<float> resample_to(const std::vector<float>& in,
                               int in_rate, int out_rate) {
    if (in.empty())   fail("resample_to", "empty input");
    if (in_rate  <= 0) fail("resample_to", "in_rate <= 0");
    if (out_rate <= 0) fail("resample_to", "out_rate <= 0");
    const int L_in = static_cast<int>(in.size());
    if (in_rate == out_rate) return in;
    // L_out chosen so duration is preserved (rounded to nearest sample).
    const std::int64_t L_out64 =
        (static_cast<std::int64_t>(L_in) * out_rate + in_rate / 2) / in_rate;
    const int L_out = static_cast<int>(std::max<std::int64_t>(1, L_out64));
    // brotensor::resample1d_forward takes X as (N, C*L_in) — for N=1, C=1
    // that's just a (1, L_in) FP32 tensor.
    bt::Tensor X = bt::Tensor::from_host_on(bt::Device::CPU, in.data(), 1, L_in);
    bt::Tensor Y;
    bt::resample1d_forward(X, /*N=*/1, /*C=*/1, L_in, L_out,
                           /*mode=*/1 /*linear*/, Y);
    return Y.to_host_vector();
}

// ─── Clip shaping ──────────────────────────────────────────────────────────

std::vector<float> crop_or_pad_centered(const std::vector<float>& samples,
                                        int target_len) {
    if (target_len <= 0)
        fail("crop_or_pad_centered", "target_len <= 0");
    const int N = static_cast<int>(samples.size());
    std::vector<float> out(static_cast<std::size_t>(target_len), 0.0f);
    if (N == 0) return out;
    if (N >= target_len) {
        // Centre crop.
        const int start = (N - target_len) / 2;
        std::memcpy(out.data(), samples.data() + start,
                    static_cast<std::size_t>(target_len) * sizeof(float));
    } else {
        // Centre pad.
        const int start = (target_len - N) / 2;
        std::memcpy(out.data() + start, samples.data(),
                    static_cast<std::size_t>(N) * sizeof(float));
    }
    return out;
}

// ─── Manifest ──────────────────────────────────────────────────────────────

std::string csv_escape_field(const std::string& s) {
    bool needs_quote = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') { needs_quote = true; break; }
    }
    if (!needs_quote) return s;
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

Manifest::Manifest(const std::string& path) : path_(path) {
    // Open binary so \n stays \n on Windows — load-bearing for the
    // deterministic byte-comparison test.
    FILE* f = nullptr;
#if defined(_MSC_VER)
    if (fopen_s(&f, path.c_str(), "wb") != 0) f = nullptr;
#else
    f = std::fopen(path.c_str(), "wb");
#endif
    if (!f) fail("Manifest::Manifest", "cannot open '" + path + "' for write");
    fp_ = f;
}

Manifest::~Manifest() {
    if (fp_) {
        std::fclose(static_cast<FILE*>(fp_));
        fp_ = nullptr;
    }
}

void Manifest::append(const ManifestRow& row) {
    if (!fp_) fail("Manifest::append", "manifest is closed");
    FILE* f = static_cast<FILE*>(fp_);
    if (!wrote_header_) {
        std::fputs("path,label,class,voice,speed,snr_db,noise_kind,seed\n", f);
        wrote_header_ = true;
    }
    char speed_buf[32];
    char snr_buf  [32];
    // Fixed 6-decimal formatting so byte-identical seeded runs match.
    std::snprintf(speed_buf, sizeof(speed_buf), "%.6f",
                  static_cast<double>(row.speed));
    std::snprintf(snr_buf,   sizeof(snr_buf),   "%.6f",
                  static_cast<double>(row.snr_db));
    std::string line;
    line.reserve(128);
    line += csv_escape_field(row.path);  line += ',';
    line += std::to_string(row.label);   line += ',';
    line += csv_escape_field(row.clazz); line += ',';
    line += csv_escape_field(row.voice); line += ',';
    line += speed_buf;                   line += ',';
    line += snr_buf;                     line += ',';
    line += csv_escape_field(row.noise_kind); line += ',';
    line += std::to_string(row.seed);    line += '\n';
    std::fwrite(line.data(), 1, line.size(), f);
    ++rows_;
}

}  // namespace brosoundml
