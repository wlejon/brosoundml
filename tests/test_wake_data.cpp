// Unit tests for the pure-DSP / IO helpers used by brosoundml_wake_synth.
// No Kokoro / no G2P here — this test must remain runnable when no model
// weights are present on disk.

#include "brosoundml/wake_data.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
namespace bt = brotensor;
using brosoundml::Manifest;
using brosoundml::ManifestRow;
using brosoundml::NoiseKind;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

// Sum of squares in the lower vs. upper half of an STFT magnitude spectrum,
// averaged across frames. Helper for the noise-spectrum sanity checks.
static void spectral_halves(const std::vector<float>& x, int n_fft, int hop,
                            double& lo_pow, double& hi_pow) {
    const int win = n_fft;
    std::vector<float> w(static_cast<std::size_t>(win));
    for (int n = 0; n < win; ++n) {
        const double phase = 2.0 * 3.14159265358979323846 *
                             static_cast<double>(n) / static_cast<double>(win);
        w[static_cast<std::size_t>(n)] =
            static_cast<float>(0.5 * (1.0 - std::cos(phase)));
    }
    bt::Tensor sig = bt::Tensor::from_host_on(bt::Device::CPU, x.data(), 1,
                                              static_cast<int>(x.size()));
    bt::Tensor win_t = bt::Tensor::from_host_on(bt::Device::CPU, w.data(), 1, win);
    bt::Tensor spec, mag;
    bt::stft(sig, win_t, /*N=*/1, n_fft, hop, win,
             /*center=*/false, /*normalized=*/false, spec);
    bt::complex_abs(spec, mag);
    const int n_bins = n_fft / 2 + 1;
    const int frames = mag.rows;
    std::vector<float> m = mag.to_host_vector();
    lo_pow = 0.0;
    hi_pow = 0.0;
    const int half = n_bins / 2;
    for (int f = 0; f < frames; ++f) {
        // Skip DC (k=0) — it's not informative about roll-off.
        for (int k = 1; k < half; ++k) {
            const float v = m[static_cast<std::size_t>(f * n_bins + k)];
            lo_pow += static_cast<double>(v) * v;
        }
        for (int k = half; k < n_bins; ++k) {
            const float v = m[static_cast<std::size_t>(f * n_bins + k)];
            hi_pow += static_cast<double>(v) * v;
        }
    }
}

int main() {
    bt::init();

    // ─── White noise ───────────────────────────────────────────────────
    {
        std::mt19937 r1(123), r2(123), r3(456);
        auto a = brosoundml::gen_white_noise(8192, 1.0f, r1);
        auto b = brosoundml::gen_white_noise(8192, 1.0f, r2);
        auto c = brosoundml::gen_white_noise(8192, 1.0f, r3);
        CHECK(a.size() == 8192, "white: shape");
        CHECK(a == b, "white: deterministic for same seed");
        CHECK(a != c, "white: differs for different seeds");
        // Mean ≈ 0.
        double mean = 0;
        for (auto v : a) mean += v;
        mean /= a.size();
        CHECK(std::fabs(mean) < 0.05, "white: zero mean");
        // Variance scales with amplitude^2.
        std::mt19937 r4(123), r5(123);
        auto d_small = brosoundml::gen_white_noise(8192, 0.5f, r4);
        auto d_large = brosoundml::gen_white_noise(8192, 2.0f, r5);
        const float rms_s = brosoundml::rms(d_small);
        const float rms_l = brosoundml::rms(d_large);
        // Same seed ⇒ proportional samples ⇒ rms ratio == amplitude ratio (4×).
        CHECK(std::fabs(rms_l / rms_s - 4.0f) < 0.01f,
              "white: rms scales linearly with amplitude");
    }

    // ─── Pink / brown determinism + spectral roll-off ──────────────────
    {
        std::mt19937 ra(7), rb(7);
        auto pa = brosoundml::gen_pink_noise(16384, 0.5f, ra);
        auto pb = brosoundml::gen_pink_noise(16384, 0.5f, rb);
        CHECK(pa == pb, "pink: deterministic for same seed");
        double lo = 0, hi = 0;
        spectral_halves(pa, 1024, 512, lo, hi);
        CHECK(lo > hi, "pink: low-half power > high-half power");

        std::mt19937 rc(11), rd(11);
        auto ba = brosoundml::gen_brown_noise(16384, 0.5f, rc);
        auto bb = brosoundml::gen_brown_noise(16384, 0.5f, rd);
        CHECK(ba == bb, "brown: deterministic for same seed");
        spectral_halves(ba, 1024, 512, lo, hi);
        // Brown rolls off harder: lo should dominate by >> 10x in power.
        CHECK(lo > 10.0 * hi, "brown: low-half power >> high-half power");
    }

    // ─── SNR mixer ─────────────────────────────────────────────────────
    {
        // Sinusoidal speech-stand-in + white noise of known scale.
        const int N = 4096;
        std::vector<float> sig(static_cast<std::size_t>(N));
        std::vector<float> nse;
        for (int n = 0; n < N; ++n) {
            sig[static_cast<std::size_t>(n)] =
                std::sin(2.0f * 3.14159265f * 440.0f * n / 16000.0f);
        }
        std::mt19937 rng(99);
        nse = brosoundml::gen_white_noise(N, 1.0f, rng);
        const float sig_rms = brosoundml::rms(sig);

        for (float snr_db : {-6.0f, 0.0f, 6.0f, 20.0f}) {
            auto mixed = brosoundml::mix_at_snr(sig, nse, snr_db);
            // Recover added noise from mix.
            std::vector<float> added(static_cast<std::size_t>(N));
            for (int i = 0; i < N; ++i)
                added[static_cast<std::size_t>(i)] =
                    mixed[static_cast<std::size_t>(i)] -
                    sig  [static_cast<std::size_t>(i)];
            const float actual = 20.0f * std::log10(
                sig_rms / brosoundml::rms(added));
            CHECK(std::fabs(actual - snr_db) < 0.1f,
                  ("mix_at_snr hits target SNR within 0.1 dB at " +
                   std::to_string(snr_db)).c_str());
        }
        // Silent-speech edge case.
        std::vector<float> silent(static_cast<std::size_t>(N), 0.0f);
        auto out = brosoundml::mix_at_snr(silent, nse, 0.0f);
        CHECK(out == silent, "mix_at_snr: silent signal returned unchanged");
    }

    // ─── RIR + convolution ────────────────────────────────────────────
    {
        std::mt19937 rng(42);
        auto rir = brosoundml::gen_rir(16000, 3200, 0.25f, 8, rng);
        CHECK(rir.size() == 3200, "rir: length matches request");
        CHECK(std::fabs(rir[0] - 1.0f) < 1e-6f, "rir: direct path unit gain");
        // Causal test: convolution of an impulse at index k yields output
        // whose first non-zero sample is also at k.
        std::vector<float> imp(2048, 0.0f);
        imp[10] = 1.0f;
        auto y = brosoundml::convolve_rir(imp, rir);
        CHECK(y.size() == imp.size(), "convolve_rir: preserves length");
        for (int n = 0; n < 10; ++n)
            CHECK(y[static_cast<std::size_t>(n)] == 0.0f,
                  "convolve_rir: output is causal (zero before input)");
        CHECK(std::fabs(y[10] - 1.0f) < 1e-5f,
              "convolve_rir: direct path passes through at unit gain");
        // Rough RMS preservation: a unit-direct-path RIR with light tail
        // should leave a broadband signal within ~2x RMS.
        std::vector<float> sig(2048);
        std::normal_distribution<float> nd(0.0f, 0.3f);
        for (auto& v : sig) v = nd(rng);
        auto y2 = brosoundml::convolve_rir(sig, rir);
        const float r0 = brosoundml::rms(sig);
        const float r1 = brosoundml::rms(y2);
        CHECK(r1 > 0.5f * r0 && r1 < 2.0f * r0,
              "convolve_rir: RMS within factor of 2");
    }

    // ─── Manifest writer ──────────────────────────────────────────────
    {
        fs::path tmp = fs::temp_directory_path() / "brosoundml_wake_manifest.csv";
        {
            Manifest m(tmp.string());
            m.append({"positives/a.wav", 1, "positive", "af_alloy",
                      1.0f, 0.0f, "", 1});
            m.append({"negatives/b,c.wav", 0, "sentence", "af_bella",
                      0.95f, 5.0f, "pink", 2});
            m.append({"negatives/quote\"x.wav", 0, "noise", "", 1.0f,
                      0.0f, "white", 3});
        }
        // Read body inside its own scope so the ifstream is destructed before
        // we delete the temp file (avoids the open-handle remove failure on
        // some Windows configurations).
        std::string body;
        {
            std::ifstream f(tmp.string(), std::ios::binary);
            std::ostringstream ss; ss << f.rdbuf();
            body = ss.str();
        }
        CHECK(body.find("path,label,class,voice,speed,snr_db,noise_kind,seed\n") == 0,
              "manifest: header on first row");
        CHECK(body.find("\"negatives/b,c.wav\"") != std::string::npos,
              "manifest: quotes a field containing a comma");
        CHECK(body.find("\"negatives/quote\"\"x.wav\"") != std::string::npos,
              "manifest: escapes embedded double quotes");
        // No CRs (binary mode preserves \n).
        CHECK(body.find('\r') == std::string::npos,
              "manifest: line endings are \\n (no \\r)");
        std::error_code ec;
        fs::remove(tmp, ec);
    }

    // ─── Resampler wrapper ───────────────────────────────────────────
    {
        // 24 kHz sine at 1 kHz → resample to 16 kHz, peak STFT bin should
        // still be at ~1 kHz.
        const int in_rate = 24000;
        const int out_rate = 16000;
        const int N = in_rate;  // 1 second
        std::vector<float> sine(static_cast<std::size_t>(N));
        for (int n = 0; n < N; ++n)
            sine[static_cast<std::size_t>(n)] =
                std::sin(2.0f * 3.14159265f * 1000.0f * n / in_rate);
        auto rs = brosoundml::resample_to(sine, in_rate, out_rate);
        // Length should be roughly 1 s @ 16 kHz.
        CHECK(std::abs(static_cast<int>(rs.size()) - out_rate) <= 1,
              "resample_to: output length matches duration");

        // Spectrum check on the resampled signal.
        const int n_fft = 2048;
        std::vector<float> win(static_cast<std::size_t>(n_fft));
        for (int n = 0; n < n_fft; ++n) {
            const double phase = 2.0 * 3.14159265358979323846 *
                                 static_cast<double>(n) /
                                 static_cast<double>(n_fft);
            win[static_cast<std::size_t>(n)] =
                static_cast<float>(0.5 * (1.0 - std::cos(phase)));
        }
        bt::Tensor sig = bt::Tensor::from_host_on(bt::Device::CPU, rs.data(),
                                                  1, static_cast<int>(rs.size()));
        bt::Tensor wt = bt::Tensor::from_host_on(bt::Device::CPU, win.data(),
                                                 1, n_fft);
        bt::Tensor spec, mag;
        bt::stft(sig, wt, 1, n_fft, n_fft / 2, n_fft, false, false, spec);
        bt::complex_abs(spec, mag);
        std::vector<float> mh = mag.to_host_vector();
        const int n_bins = n_fft / 2 + 1;
        const int frames = mag.rows;
        std::vector<double> band(static_cast<std::size_t>(n_bins), 0.0);
        for (int f = 0; f < frames; ++f) {
            for (int k = 0; k < n_bins; ++k)
                band[static_cast<std::size_t>(k)] +=
                    mh[static_cast<std::size_t>(f * n_bins + k)];
        }
        int peak_bin = 0;
        for (int k = 1; k < n_bins; ++k)
            if (band[static_cast<std::size_t>(k)] >
                band[static_cast<std::size_t>(peak_bin)]) peak_bin = k;
        // Expected bin = 1000 * n_fft / out_rate = 128.
        const int expected = (1000 * n_fft) / out_rate;
        CHECK(std::abs(peak_bin - expected) <= 1,
              "resample_to: peak frequency preserved within one bin");
    }

    // ─── Biquad: low-pass kills the high half of the spectrum ───────────
    {
        std::mt19937 rng(2024);
        auto x = brosoundml::gen_white_noise(16384, 1.0f, rng);
        double lo0 = 0, hi0 = 0;
        spectral_halves(x, 1024, 512, lo0, hi0);
        CHECK(hi0 > 0.0, "biquad: baseline noise has high-half energy");
        brosoundml::apply_biquad(x, brosoundml::design_lowpass(16000.f, 1000.f,
                                                               0.707f));
        double lo1 = 0, hi1 = 0;
        spectral_halves(x, 1024, 512, lo1, hi1);
        // 1 kHz LPF leaves the lower spectral half dominant by a wide margin.
        CHECK(lo1 > 50.0 * hi1, "biquad lowpass: lo-half >> hi-half after filter");
        CHECK(hi1 < 0.1 * hi0, "biquad lowpass: high-half energy collapses");
    }

    // ─── Shelves move band power in the expected direction ──────────────
    {
        std::mt19937 r1(55), r2(55);
        auto x  = brosoundml::gen_white_noise(16384, 1.0f, r1);
        auto xb = brosoundml::gen_white_noise(16384, 1.0f, r2);  // same noise
        double lo0 = 0, hi0 = 0; spectral_halves(x, 1024, 512, lo0, hi0);
        brosoundml::apply_biquad(xb,
            brosoundml::design_low_shelf(16000.f, 300.f, 0.707f, +12.f));
        double lo1 = 0, hi1 = 0; spectral_halves(xb, 1024, 512, lo1, hi1);
        CHECK(lo1 > lo0 * 1.5, "low_shelf +12dB: low-half power rises");
        CHECK(std::fabs(hi1 - hi0) < hi0 * 0.5,
              "low_shelf +12dB: high-half power roughly unchanged");
    }

    // ─── DRC compresses the loud/quiet level ratio ──────────────────────
    {
        // Quiet sustained tone (≈ threshold) with a loud central burst. A
        // feed-forward compressor lets the burst's leading transient through
        // (finite attack), so crest-factor isn't a robust witness; the right
        // invariant is that the steady-state loud-region level is pulled toward
        // the quiet-region level — i.e. dynamic range is compressed.
        const int N = 16000;
        std::vector<float> x(static_cast<std::size_t>(N));
        for (int n = 0; n < N; ++n) {
            const float base = 0.05f * std::sin(2.f * 3.14159265f * 300.f * n / 16000.f);
            const float burst = (n > 7000 && n < 9000)
                ? 0.9f * std::sin(2.f * 3.14159265f * 300.f * n / 16000.f) : 0.f;
            x[static_cast<std::size_t>(n)] = base + burst;
        }
        auto slice_rms = [](const std::vector<float>& v, int a, int b) {
            double acc = 0.0;
            for (int n = a; n < b; ++n) acc += static_cast<double>(v[n]) * v[n];
            return std::sqrt(acc / std::max(1, b - a));
        };
        // Steady-state windows, clear of the attack transient at n≈7000.
        const double loud0  = slice_rms(x, 8000, 9000);
        const double quiet0 = slice_rms(x,  3000, 5000);
        brosoundml::CompressorConfig cc;
        cc.threshold_db = -26.f; cc.ratio = 6.f; cc.makeup_db = 0.f;
        brosoundml::apply_drc(x, 16000, cc);
        const double loud1  = slice_rms(x, 8000, 9000);
        const double quiet1 = slice_rms(x,  3000, 5000);
        CHECK(loud1 / std::max(1e-12, quiet1) <
              0.5 * (loud0 / std::max(1e-12, quiet0)),
              "drc: loud/quiet level ratio compressed");
        CHECK(brosoundml::compute_audio_stats(x).has_nan == false,
              "drc: output finite");
    }

    // ─── pitch_jitter: shifts sample count, no-op at zero ───────────────
    {
        const int N = 16000;
        std::vector<float> x(static_cast<std::size_t>(N));
        for (int n = 0; n < N; ++n)
            x[static_cast<std::size_t>(n)] =
                std::sin(2.f * 3.14159265f * 200.f * n / 16000.f);
        auto up   = brosoundml::pitch_jitter(x, 16000, +2.f);
        auto down = brosoundml::pitch_jitter(x, 16000, -2.f);
        auto zero = brosoundml::pitch_jitter(x, 16000, 0.f);
        CHECK(up.size()   < x.size(), "pitch_jitter +: fewer samples (pitch up)");
        CHECK(down.size() > x.size(), "pitch_jitter -: more samples (pitch down)");
        CHECK(zero.size() == x.size(), "pitch_jitter 0: no-op length");
    }

    // ─── apply_acquisition_channel: grid-locked, finite, deterministic ──
    {
        const int N = 16000;
        std::mt19937 src(1);
        std::vector<float> clean(static_cast<std::size_t>(N));
        for (int n = 0; n < N; ++n)
            clean[static_cast<std::size_t>(n)] =
                0.3f * std::sin(2.f * 3.14159265f * 500.f * n / 16000.f);
        std::mt19937 ra(7), rb(7), rc(8);
        auto a = brosoundml::apply_acquisition_channel(clean, 16000, N, ra);
        auto b = brosoundml::apply_acquisition_channel(clean, 16000, N, rb);
        auto c = brosoundml::apply_acquisition_channel(clean, 16000, N, rc);
        CHECK(static_cast<int>(a.size()) == N,
              "acq channel: output re-landed on target grid");
        CHECK(a == b, "acq channel: deterministic for same rng seed");
        CHECK(a != c, "acq channel: differs for different rng seed");
        CHECK(brosoundml::compute_audio_stats(a).has_nan == false,
              "acq channel: output finite");
    }

    if (failures) {
        std::fprintf(stderr, "test_wake_data: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_wake_data: OK\n");
    return 0;
}
