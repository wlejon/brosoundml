#include "brosoundml/wake_data.h"

#include "brosoundml/audio.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
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

// ─── Biquad IIR filtering ────────────────────────────────────────────────

namespace {

constexpr double kPi = 3.14159265358979323846;

// Clamp a corner frequency to a safe (0, Nyquist) interval so the bilinear
// designers never see f0 == 0 or f0 >= sr/2 (which blow up tan / produce NaNs).
float clamp_freq(float sr, float f0) {
    const float nyq = 0.5f * sr;
    const float lo = 1.0f;
    const float hi = 0.999f * nyq;
    if (f0 < lo) return lo;
    if (f0 > hi) return hi;
    return f0;
}

Biquad normalize(double b0, double b1, double b2,
                 double a0, double a1, double a2) {
    Biquad q;
    q.b0 = static_cast<float>(b0 / a0);
    q.b1 = static_cast<float>(b1 / a0);
    q.b2 = static_cast<float>(b2 / a0);
    q.a1 = static_cast<float>(a1 / a0);
    q.a2 = static_cast<float>(a2 / a0);
    return q;
}

}  // namespace

Biquad design_lowpass(float sr, float f0, float q) {
    f0 = clamp_freq(sr, f0);
    if (q <= 0.0f) q = 0.707f;
    const double w0 = 2.0 * kPi * f0 / sr;
    const double cw = std::cos(w0), sw = std::sin(w0);
    const double alpha = sw / (2.0 * q);
    const double b0 = (1.0 - cw) / 2.0;
    const double b1 = 1.0 - cw;
    const double b2 = (1.0 - cw) / 2.0;
    const double a0 = 1.0 + alpha;
    const double a1 = -2.0 * cw;
    const double a2 = 1.0 - alpha;
    return normalize(b0, b1, b2, a0, a1, a2);
}

Biquad design_highpass(float sr, float f0, float q) {
    f0 = clamp_freq(sr, f0);
    if (q <= 0.0f) q = 0.707f;
    const double w0 = 2.0 * kPi * f0 / sr;
    const double cw = std::cos(w0), sw = std::sin(w0);
    const double alpha = sw / (2.0 * q);
    const double b0 =  (1.0 + cw) / 2.0;
    const double b1 = -(1.0 + cw);
    const double b2 =  (1.0 + cw) / 2.0;
    const double a0 = 1.0 + alpha;
    const double a1 = -2.0 * cw;
    const double a2 = 1.0 - alpha;
    return normalize(b0, b1, b2, a0, a1, a2);
}

Biquad design_peaking(float sr, float f0, float q, float gain_db) {
    f0 = clamp_freq(sr, f0);
    if (q <= 0.0f) q = 0.707f;
    const double A = std::pow(10.0, gain_db / 40.0);
    const double w0 = 2.0 * kPi * f0 / sr;
    const double cw = std::cos(w0), sw = std::sin(w0);
    const double alpha = sw / (2.0 * q);
    const double b0 = 1.0 + alpha * A;
    const double b1 = -2.0 * cw;
    const double b2 = 1.0 - alpha * A;
    const double a0 = 1.0 + alpha / A;
    const double a1 = -2.0 * cw;
    const double a2 = 1.0 - alpha / A;
    return normalize(b0, b1, b2, a0, a1, a2);
}

Biquad design_low_shelf(float sr, float f0, float q, float gain_db) {
    f0 = clamp_freq(sr, f0);
    if (q <= 0.0f) q = 0.707f;
    const double A = std::pow(10.0, gain_db / 40.0);
    const double w0 = 2.0 * kPi * f0 / sr;
    const double cw = std::cos(w0), sw = std::sin(w0);
    const double alpha = sw / (2.0 * q);
    const double tsa = 2.0 * std::sqrt(A) * alpha;
    const double b0 =      A * ((A + 1.0) - (A - 1.0) * cw + tsa);
    const double b1 =  2.0 * A * ((A - 1.0) - (A + 1.0) * cw);
    const double b2 =      A * ((A + 1.0) - (A - 1.0) * cw - tsa);
    const double a0 =          (A + 1.0) + (A - 1.0) * cw + tsa;
    const double a1 = -2.0 *   ((A - 1.0) + (A + 1.0) * cw);
    const double a2 =          (A + 1.0) + (A - 1.0) * cw - tsa;
    return normalize(b0, b1, b2, a0, a1, a2);
}

Biquad design_high_shelf(float sr, float f0, float q, float gain_db) {
    f0 = clamp_freq(sr, f0);
    if (q <= 0.0f) q = 0.707f;
    const double A = std::pow(10.0, gain_db / 40.0);
    const double w0 = 2.0 * kPi * f0 / sr;
    const double cw = std::cos(w0), sw = std::sin(w0);
    const double alpha = sw / (2.0 * q);
    const double tsa = 2.0 * std::sqrt(A) * alpha;
    const double b0 =       A * ((A + 1.0) + (A - 1.0) * cw + tsa);
    const double b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cw);
    const double b2 =       A * ((A + 1.0) + (A - 1.0) * cw - tsa);
    const double a0 =           (A + 1.0) - (A - 1.0) * cw + tsa;
    const double a1 =  2.0 *    ((A - 1.0) - (A + 1.0) * cw);
    const double a2 =           (A + 1.0) - (A - 1.0) * cw - tsa;
    return normalize(b0, b1, b2, a0, a1, a2);
}

void apply_biquad(std::vector<float>& x, const Biquad& bq) {
    // Direct-Form-II transposed, double-precision state for numerical headroom.
    double z1 = 0.0, z2 = 0.0;
    for (auto& v : x) {
        const double in = static_cast<double>(v);
        const double out = bq.b0 * in + z1;
        z1 = bq.b1 * in - bq.a1 * out + z2;
        z2 = bq.b2 * in - bq.a2 * out;
        v = static_cast<float>(out);
    }
}

// ─── Acquisition-channel augmentation ──────────────────────────────────────

void apply_random_eq(std::vector<float>& x, int sr, std::mt19937& rng) {
    if (x.empty()) return;
    const float sf = static_cast<float>(sr);
    std::uniform_real_distribution<float> tilt_db(-6.0f, 6.0f);
    // Broad spectral tilt: low shelf one way, high shelf the other.
    const float tilt = tilt_db(rng);
    apply_biquad(x, design_low_shelf (sf, 200.0f, 0.707f,  tilt));
    apply_biquad(x, design_high_shelf(sf, 4000.0f, 0.707f, -tilt));
    // A couple of random peaking bands for resonant microphone colouration.
    std::uniform_int_distribution<int>    n_bands(1, 3);
    std::uniform_real_distribution<float> band_hz(300.0f, 6000.0f);
    std::uniform_real_distribution<float> band_q(0.7f, 3.0f);
    std::uniform_real_distribution<float> band_db(-8.0f, 8.0f);
    const int nb = n_bands(rng);
    for (int i = 0; i < nb; ++i) {
        apply_biquad(x, design_peaking(sf, band_hz(rng), band_q(rng),
                                       band_db(rng)));
    }
}

void apply_proximity_boost(std::vector<float>& x, int sr, std::mt19937& rng) {
    if (x.empty()) return;
    std::uniform_real_distribution<float> f0(80.0f, 250.0f);
    std::uniform_real_distribution<float> gain(2.0f, 10.0f);
    apply_biquad(x, design_low_shelf(static_cast<float>(sr), f0(rng), 0.707f,
                                     gain(rng)));
}

void apply_bandlimit(std::vector<float>& x, int sr, std::mt19937& rng) {
    if (x.empty()) return;
    const float sf = static_cast<float>(sr);
    std::uniform_real_distribution<float> hp_hz(60.0f, 200.0f);
    std::uniform_real_distribution<float> lp_hz(3400.0f, 7500.0f);
    apply_biquad(x, design_highpass(sf, hp_hz(rng), 0.707f));
    apply_biquad(x, design_lowpass (sf, lp_hz(rng), 0.707f));
}

void apply_drc(std::vector<float>& x, int sr, const CompressorConfig& cfg) {
    if (x.empty()) return;
    const float ratio = cfg.ratio > 1.0f ? cfg.ratio : 1.0f;
    // One-pole attack/release coefficients on the envelope.
    const float att = std::exp(-1.0f /
        (std::max(1e-3f, cfg.attack_ms)  * 1e-3f * static_cast<float>(sr)));
    const float rel = std::exp(-1.0f /
        (std::max(1e-3f, cfg.release_ms) * 1e-3f * static_cast<float>(sr)));
    const float makeup = std::pow(10.0f, cfg.makeup_db / 20.0f);
    float env = 0.0f;   // linear envelope of |x|
    for (auto& v : x) {
        const float a = std::fabs(v);
        // Attack when rising, release when falling.
        env = (a > env) ? att * env + (1.0f - att) * a
                        : rel * env + (1.0f - rel) * a;
        float gain = 1.0f;
        if (env > 1e-9f) {
            const float lvl_db = 20.0f * std::log10(env);
            if (lvl_db > cfg.threshold_db) {
                // Gain reduction (dB) = (over) * (1 - 1/ratio).
                const float over = lvl_db - cfg.threshold_db;
                const float gr_db = -over * (1.0f - 1.0f / ratio);
                gain = std::pow(10.0f, gr_db / 20.0f);
            }
        }
        v *= gain * makeup;
    }
}

std::vector<float> pitch_jitter(const std::vector<float>& x, int sr,
                                float semitones) {
    if (x.empty() || sr <= 0) return x;
    if (std::fabs(semitones) < 1e-3f) return x;
    const double ratio = std::pow(2.0, semitones / 12.0);   // >1 ⇒ pitch up
    // Resample to sr/ratio samples (same count it would have at a lower rate),
    // then keep them as sr samples: content is compressed/stretched in time and
    // pitch shifts by `ratio`.
    const int out_rate = std::max(1,
        static_cast<int>(std::lround(static_cast<double>(sr) / ratio)));
    if (out_rate == sr) return x;
    return resample_to(x, sr, out_rate);
}

std::vector<float> apply_acquisition_channel(const std::vector<float>& x,
                                             int sr, int target_len,
                                             std::mt19937& rng,
                                             const AcquisitionChannel& cfg) {
    if (x.empty() || target_len <= 0) return x;
    std::vector<float> y = x;

    // Each enabled stage fires with its own probability so clips get varied
    // *subsets* of the chain, not all-or-nothing.
    auto chance = [&](float p) {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        return u(rng) < p;
    };

    // 1) Pitch jitter (speaker diversity) — couples duration; re-crop next.
    if (cfg.do_pitch && chance(0.7f)) {
        std::uniform_real_distribution<float> st(-2.0f, 2.0f);
        y = pitch_jitter(y, sr, st(rng));
    }
    // Land back on the fixed grid before the rest of the chain.
    y = crop_or_pad_centered(y, target_len);

    // 2) Room impulse response (reverb) — reuse the existing synthetic RIR.
    if (cfg.do_rir && chance(0.5f)) {
        std::uniform_real_distribution<float> t60(0.1f, 0.4f);
        std::uniform_int_distribution<int>    taps(4, 12);
        const float t = t60(rng);
        const int rir_len = std::min(target_len,
            std::max(1, static_cast<int>(t * static_cast<float>(sr))));
        const auto rir = gen_rir(sr, rir_len, t, taps(rng), rng);
        y = convolve_rir(y, rir);
    }
    // 3) Microphone colouration EQ.
    if (cfg.do_eq && chance(0.85f)) apply_random_eq(y, sr, rng);
    // 4) Proximity LF boost.
    if (cfg.do_proximity && chance(0.4f)) apply_proximity_boost(y, sr, rng);
    // 5) Band-limiting.
    if (cfg.do_bandlimit && chance(0.7f)) apply_bandlimit(y, sr, rng);
    // 6) Dynamic-range compression.
    if (cfg.do_drc && chance(0.6f)) {
        std::uniform_real_distribution<float> thr(-30.0f, -15.0f);
        std::uniform_real_distribution<float> rat(2.0f, 6.0f);
        std::uniform_real_distribution<float> mk(0.0f, 6.0f);
        CompressorConfig cc;
        cc.threshold_db = thr(rng);
        cc.ratio        = rat(rng);
        cc.makeup_db    = mk(rng);
        apply_drc(y, sr, cc);
    }
    return y;
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

// ─── Manifest reader ───────────────────────────────────────────────────────

namespace {

// RFC-4180-ish CSV row tokeniser. Walks `body` starting at `pos`, fills
// `out_fields`, advances `pos` past the row terminator, returns true if a row
// was read and false on end-of-input. Throws on malformed quoting.
bool read_csv_row(const std::string& body, std::size_t& pos,
                  std::vector<std::string>& out_fields,
                  std::size_t line_no_for_msg) {
    out_fields.clear();
    if (pos >= body.size()) return false;
    std::string field;
    bool in_quotes = false;
    while (pos < body.size()) {
        const char c = body[pos];
        if (in_quotes) {
            if (c == '"') {
                if (pos + 1 < body.size() && body[pos + 1] == '"') {
                    field.push_back('"');
                    pos += 2;
                } else {
                    in_quotes = false;
                    ++pos;
                }
            } else {
                field.push_back(c);
                ++pos;
            }
        } else {
            if (c == '"') {
                if (!field.empty())
                    fail("read_manifest",
                         "stray quote mid-field at line " +
                         std::to_string(line_no_for_msg));
                in_quotes = true;
                ++pos;
            } else if (c == ',') {
                out_fields.push_back(std::move(field));
                field.clear();
                ++pos;
            } else if (c == '\n') {
                ++pos;
                out_fields.push_back(std::move(field));
                return true;
            } else if (c == '\r') {
                // Tolerate CRLF on the off-chance someone hand-edited the file.
                ++pos;
                if (pos < body.size() && body[pos] == '\n') ++pos;
                out_fields.push_back(std::move(field));
                return true;
            } else {
                field.push_back(c);
                ++pos;
            }
        }
    }
    if (in_quotes)
        fail("read_manifest",
             "unterminated quoted field at line " +
             std::to_string(line_no_for_msg));
    out_fields.push_back(std::move(field));
    return true;
}

// Whitelist of class labels accepted by the chunk-3 producer.
// The probe-style synthetic stimulus families (silence/tone/formant/am/sweep)
// are added by the chunk-7 dataset expansion to anchor the band-energy
// shortcut failure modes directly into the negative class.
bool is_known_class(const std::string& c) {
    return c == "positive" || c == "confusable" ||
           c == "sentence" || c == "noise" ||
           c == "silence"  || c == "tone"   ||
           c == "formant"  || c == "am"     || c == "sweep";
}

}  // namespace

std::vector<ManifestRow> read_manifest(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("read_manifest", "cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string body = ss.str();

    std::size_t pos = 0;
    std::vector<std::string> fields;
    std::size_t line = 1;

    if (!read_csv_row(body, pos, fields, line))
        fail("read_manifest", "empty file '" + path + "'");
    static const char* kHeader[] = {
        "path", "label", "class", "voice", "speed",
        "snr_db", "noise_kind", "seed"
    };
    if (fields.size() != 8)
        fail("read_manifest", "header column count != 8 in '" + path + "'");
    for (std::size_t i = 0; i < 8; ++i) {
        if (fields[i] != kHeader[i])
            fail("read_manifest",
                 "header column " + std::to_string(i) +
                 " is '" + fields[i] + "', expected '" + kHeader[i] + "'");
    }

    std::vector<ManifestRow> rows;
    while (true) {
        ++line;
        if (!read_csv_row(body, pos, fields, line)) break;
        // A single empty unterminated trailing field = blank tail. Skip.
        if (fields.size() == 1 && fields[0].empty()) break;
        if (fields.size() != 8)
            fail("read_manifest",
                 "row at line " + std::to_string(line) + " has " +
                 std::to_string(fields.size()) + " columns, expected 8");
        ManifestRow r;
        r.path  = fields[0];
        try { r.label = std::stoi(fields[1]); }
        catch (...) {
            fail("read_manifest",
                 "row at line " + std::to_string(line) +
                 ": bad label '" + fields[1] + "'");
        }
        r.clazz = fields[2];
        if (!is_known_class(r.clazz))
            fail("read_manifest",
                 "row at line " + std::to_string(line) +
                 ": unknown class '" + r.clazz + "'");
        r.voice = fields[3];
        try { r.speed  = std::stof(fields[4]); }
        catch (...) {
            fail("read_manifest",
                 "row at line " + std::to_string(line) +
                 ": bad speed '" + fields[4] + "'");
        }
        try { r.snr_db = std::stof(fields[5]); }
        catch (...) {
            fail("read_manifest",
                 "row at line " + std::to_string(line) +
                 ": bad snr_db '" + fields[5] + "'");
        }
        r.noise_kind = fields[6];
        try { r.seed = std::stoull(fields[7]); }
        catch (...) {
            fail("read_manifest",
                 "row at line " + std::to_string(line) +
                 ": bad seed '" + fields[7] + "'");
        }
        rows.push_back(std::move(r));
    }
    return rows;
}

// ─── Audio stats ───────────────────────────────────────────────────────────

AudioStats compute_audio_stats(const std::vector<float>& x) {
    AudioStats s;
    if (x.empty()) return s;
    double sq = 0.0;
    for (float v : x) {
        if (std::isnan(v) || std::isinf(v)) { s.has_nan = true; continue; }
        const float a = std::fabs(v);
        if (a > s.peak) s.peak = a;
        if (a < 1e-4f)  ++s.silent_samples;
        if (a >= 0.999f) ++s.clipped_samples;
        sq += static_cast<double>(v) * v;
    }
    s.rms = static_cast<float>(std::sqrt(sq / static_cast<double>(x.size())));
    return s;
}

// ─── Dataset validation ────────────────────────────────────────────────────

namespace {

void bump(std::vector<std::pair<std::string, int>>& kv, const std::string& k) {
    for (auto& p : kv) if (p.first == k) { ++p.second; return; }
    kv.emplace_back(k, 1);
}

std::string speed_bucket_key(float speed) {
    // Round speed to the nearest 0.05.
    const float q = std::round(speed / 0.05f) * 0.05f;
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.2f", static_cast<double>(q));
    return buf;
}

std::string snr_bucket_key(const std::string& noise_kind, float snr_db) {
    if (noise_kind.empty()) return "clean";
    // 5 dB buckets, floor-toward-zero indexing.
    const int b = static_cast<int>(std::floor(snr_db / 5.0f)) * 5;
    return std::to_string(b);
}

bool starts_with(const std::string& s, const std::string& pfx) {
    return s.size() >= pfx.size() &&
           std::memcmp(s.data(), pfx.data(), pfx.size()) == 0;
}

}  // namespace

DatasetReport validate_dataset(const std::vector<ManifestRow>& rows,
                               const std::string& dataset_root,
                               const ValidationConfig& cfg) {
    DatasetReport r;
    r.total_rows = static_cast<int>(rows.size());

    std::vector<float> peaks;
    std::vector<float> rmses;
    peaks.reserve(rows.size());
    rmses.reserve(rows.size());

    for (const auto& row : rows) {
        bool rejected = false;

        bump(r.class_counts, row.clazz);
        bump(r.label_counts, std::to_string(row.label));
        if (!row.voice.empty()) bump(r.voice_counts, row.voice);
        bump(r.noise_counts,
             row.noise_kind.empty() ? std::string("clean") : row.noise_kind);
        bump(r.speed_counts, speed_bucket_key(row.speed));
        bump(r.snr_counts,   snr_bucket_key(row.noise_kind, row.snr_db));

        if (!is_known_class(row.clazz)) {
            ++r.unknown_class;
            rejected = true;
        }

        // Label vs. path-prefix.
        if (row.label == 1 && !starts_with(row.path, "positives/")) {
            ++r.label_path_mismatch;
            rejected = true;
        } else if (row.label == 0 && !starts_with(row.path, "negatives/")) {
            ++r.label_path_mismatch;
            rejected = true;
        }

        const std::string abs = dataset_root.empty()
            ? row.path
            : (dataset_root + "/" + row.path);

        std::error_code ec;
        if (!std::filesystem::exists(abs, ec)) {
            ++r.missing_files;
            ++r.rejected_clips;
            continue;
        }

        AudioBuffer buf;
        try { buf = read_wav(abs); }
        catch (const std::exception&) {
            ++r.malformed_wavs;
            ++r.rejected_clips;
            continue;
        }

        if (buf.sample_rate != cfg.expected_sample_rate) {
            ++r.sample_rate_mismatch;
            rejected = true;
        }
        const double dur = buf.duration_seconds();
        if (dur < cfg.min_duration_s || dur > cfg.max_duration_s) {
            ++r.duration_out_of_range;
            rejected = true;
        }

        const AudioStats s = compute_audio_stats(buf.samples);
        peaks.push_back(s.peak);
        rmses.push_back(s.rms);

        const float frac_silent =
            buf.samples.empty() ? 0.0f
            : static_cast<float>(s.silent_samples) /
              static_cast<float>(buf.samples.size());
        const float frac_clipped =
            buf.samples.empty() ? 0.0f
            : static_cast<float>(s.clipped_samples) /
              static_cast<float>(buf.samples.size());

        if (frac_silent >= cfg.max_silent_fraction) {
            ++r.silent_clips;
            rejected = true;
        }
        if (frac_clipped >= cfg.max_clipped_fraction) {
            ++r.clipped_clips;
            rejected = true;
        }
        if (s.has_nan) {
            ++r.nan_clips;
            rejected = true;
        }

        ++r.decoded_clips;
        if (rejected) ++r.rejected_clips;
    }

    if (!peaks.empty()) {
        float pmin = peaks[0], pmax = peaks[0]; double psum = 0.0;
        float rmin = rmses[0], rmax = rmses[0]; double rsum = 0.0;
        for (std::size_t i = 0; i < peaks.size(); ++i) {
            const float p = peaks[i], q = rmses[i];
            if (p < pmin) pmin = p; if (p > pmax) pmax = p;
            if (q < rmin) rmin = q; if (q > rmax) rmax = q;
            psum += p; rsum += q;
        }
        const double n = static_cast<double>(peaks.size());
        const double pmean = psum / n, rmean = rsum / n;
        double pvar = 0.0, rvar = 0.0;
        for (std::size_t i = 0; i < peaks.size(); ++i) {
            const double dp = peaks[i] - pmean, dq = rmses[i] - rmean;
            pvar += dp * dp; rvar += dq * dq;
        }
        pvar /= n; rvar /= n;
        r.peak_min = pmin; r.peak_max = pmax;
        r.peak_mean = static_cast<float>(pmean);
        r.peak_std  = static_cast<float>(std::sqrt(pvar));
        r.rms_min = rmin; r.rms_max = rmax;
        r.rms_mean = static_cast<float>(rmean);
        r.rms_std  = static_cast<float>(std::sqrt(rvar));
    }

    return r;
}

bool report_passes(const DatasetReport& r, const ValidationConfig& cfg,
                   std::string* reason) {
    auto set = [&](const std::string& s) { if (reason) *reason = s; };

    if (r.total_rows == 0) { set("dataset is empty"); return false; }
    if (r.missing_files > 0) {
        set("missing files: " + std::to_string(r.missing_files));
        return false;
    }
    if (r.malformed_wavs > 0) {
        set("malformed wavs: " + std::to_string(r.malformed_wavs));
        return false;
    }
    if (r.sample_rate_mismatch > 0) {
        set("sample-rate mismatch: " +
            std::to_string(r.sample_rate_mismatch) + " clip(s)");
        return false;
    }
    if (r.duration_out_of_range > 0) {
        set("duration out of range: " +
            std::to_string(r.duration_out_of_range) + " clip(s)");
        return false;
    }
    if (r.label_path_mismatch > 0) {
        set("label/path-prefix mismatch: " +
            std::to_string(r.label_path_mismatch) + " row(s)");
        return false;
    }
    if (r.unknown_class > 0) {
        set("unknown class label: " +
            std::to_string(r.unknown_class) + " row(s)");
        return false;
    }
    if (r.nan_clips > 0) {
        set("NaN/Inf samples: " + std::to_string(r.nan_clips) + " clip(s)");
        return false;
    }

    if (r.total_rows > 0) {
        const float n = static_cast<float>(r.total_rows);
        const float silent_pct  = 100.0f * static_cast<float>(r.silent_clips) / n;
        const float clipped_pct = 100.0f * static_cast<float>(r.clipped_clips) / n;
        if (silent_pct > cfg.max_silent_pct) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                "silent clips %.2f%% exceeds max %.2f%%",
                static_cast<double>(silent_pct),
                static_cast<double>(cfg.max_silent_pct));
            set(buf);
            return false;
        }
        if (clipped_pct > cfg.max_clipped_pct) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                "clipped clips %.2f%% exceeds max %.2f%%",
                static_cast<double>(clipped_pct),
                static_cast<double>(cfg.max_clipped_pct));
            set(buf);
            return false;
        }
    }

    if (cfg.require_positive_class || cfg.require_negative_class) {
        int pos = 0, neg = 0;
        for (const auto& kv : r.label_counts) {
            if      (kv.first == "1") pos = kv.second;
            else if (kv.first == "0") neg = kv.second;
        }
        if (cfg.require_positive_class && pos == 0) {
            set("no positive-class clips present");
            return false;
        }
        if (cfg.require_negative_class && neg == 0) {
            set("no negative-class clips present");
            return false;
        }
    }

    if (reason) reason->clear();
    return true;
}

void print_report(const DatasetReport& r, std::FILE* out) {
    auto dump = [&](const char* title,
                    const std::vector<std::pair<std::string, int>>& kv) {
        std::fprintf(out, "  %s:\n", title);
        auto sorted = kv;
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& p : sorted)
            std::fprintf(out, "    %-20s %d\n", p.first.c_str(), p.second);
    };

    std::fprintf(out, "dataset report\n");
    std::fprintf(out, "  total rows:           %d\n", r.total_rows);
    std::fprintf(out, "  decoded clips:        %d\n", r.decoded_clips);
    std::fprintf(out, "  missing files:        %d\n", r.missing_files);
    std::fprintf(out, "  malformed wavs:       %d\n", r.malformed_wavs);
    std::fprintf(out, "  sample-rate mismatch: %d\n", r.sample_rate_mismatch);
    std::fprintf(out, "  duration out of rng:  %d\n", r.duration_out_of_range);
    std::fprintf(out, "  label/path mismatch:  %d\n", r.label_path_mismatch);
    std::fprintf(out, "  unknown class:        %d\n", r.unknown_class);
    std::fprintf(out, "  silent clips:         %d\n", r.silent_clips);
    std::fprintf(out, "  clipped clips:        %d\n", r.clipped_clips);
    std::fprintf(out, "  NaN clips:            %d\n", r.nan_clips);
    std::fprintf(out, "  rejected clips:       %d\n", r.rejected_clips);
    std::fprintf(out, "  peak: min=%.4f mean=%.4f max=%.4f std=%.4f\n",
                 static_cast<double>(r.peak_min),
                 static_cast<double>(r.peak_mean),
                 static_cast<double>(r.peak_max),
                 static_cast<double>(r.peak_std));
    std::fprintf(out, "  rms : min=%.4f mean=%.4f max=%.4f std=%.4f\n",
                 static_cast<double>(r.rms_min),
                 static_cast<double>(r.rms_mean),
                 static_cast<double>(r.rms_max),
                 static_cast<double>(r.rms_std));
    dump("by label",      r.label_counts);
    dump("by class",      r.class_counts);
    dump("by voice",      r.voice_counts);
    dump("by noise kind", r.noise_counts);
    dump("by speed",      r.speed_counts);
    dump("by snr bucket", r.snr_counts);
}

}  // namespace brosoundml
