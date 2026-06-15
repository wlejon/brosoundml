// M1: the Supertonic AE encoder input transform (src/supertonic_spec.{h,cpp}).
// A pure tone must concentrate linear-magnitude energy in the expected rfft
// bin, the output must be [idim, T] with T = 1 + L/hop, and every value finite.
// CPU-resident FP32 — deterministic, no model weights needed.

#include "supertonic_spec.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;

static int g_failures = 0;

static void check(bool ok, const char* ctx) {
    if (ok) return;
    std::printf("  FAIL %s\n", ctx);
    ++g_failures;
}

int main() {
    brotensor::init();

    const int sr = 44100, n_fft = 2048, win = 2048, hop = 512, n_mels = 228;
    brosoundml::SupertonicSpec spec(Device::CPU, sr, n_fft, win, hop, n_mels);

    const int idim = n_fft / 2 + 1 + n_mels;  // 1025 + 228 = 1253
    check(spec.idim() == idim, "idim == 1253");

    // 1 s of a 440 Hz sine.
    const int L = sr;
    const double f0 = 440.0;
    std::vector<float> sig(static_cast<std::size_t>(L));
    for (int n = 0; n < L; ++n)
        sig[static_cast<std::size_t>(n)] =
            0.5f * static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * f0 *
                                               n / sr));

    Tensor feat = spec.compute(sig.data(), L);
    const int T = 1 + L / hop;  // 1 + 86 = 87
    check(feat.rows == idim, "feat.rows == idim");
    check(feat.cols == T, "feat.cols == T (1 + L/hop)");

    const std::vector<float> h = feat.to_host_vector();  // [idim, T] channel-major
    bool all_finite = true;
    for (float v : h) if (!std::isfinite(v)) { all_finite = false; break; }
    check(all_finite, "all values finite");

    // The 440 Hz tone's rfft peak bin: round(f0 * n_fft / sr) = round(20.4) = 20.
    // Average each linear-magnitude row over interior frames (skip the reflect-
    // padded edges) and confirm the peak row sits at the expected bin.
    const int peak_expected = static_cast<int>(std::lround(f0 * n_fft / sr));
    int peak_row = 0;
    float peak_val = -1e30f;
    const int t0 = 4, t1 = T - 4;
    for (int k = 0; k < n_fft / 2 + 1; ++k) {
        double s = 0.0;
        for (int t = t0; t < t1; ++t) s += h[static_cast<std::size_t>(k) * T + t];
        const float avg = static_cast<float>(s / (t1 - t0));
        if (avg > peak_val) { peak_val = avg; peak_row = k; }
    }
    check(std::abs(peak_row - peak_expected) <= 1, "linear-mag peak at ~bin 20");

    // Mel rows (idim region n_bins..idim-1) must also be finite and not all
    // equal to the log-eps floor (the tone should excite some mel bands).
    const int n_bins = n_fft / 2 + 1;
    float mel_max = -1e30f, mel_min = 1e30f;
    for (int m = 0; m < n_mels; ++m) {
        const float v = h[static_cast<std::size_t>(n_bins + m) * T + T / 2];
        mel_max = std::fmax(mel_max, v);
        mel_min = std::fmin(mel_min, v);
    }
    check(mel_max > mel_min + 1.0f, "mel band has dynamic range");

    if (g_failures == 0) {
        std::printf("PASS test_supertonic_spec (T=%d, peak_row=%d)\n", T, peak_row);
        return 0;
    }
    std::printf("FAILED test_supertonic_spec (%d failures)\n", g_failures);
    return 1;
}
