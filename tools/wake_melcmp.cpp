// brosoundml_wake_melcmp — diff the model's actual log-mel front-end output
// between two clips, and report each clip's peak wake score.
//
// Answers "what specifically is different acoustically" between a clip the
// wake model fires on and one it ignores, in the exact feature space the model
// consumes: 40-bin HTK log-mel (16 kHz, 25 ms win, 10 ms hop), via the model's
// own MelFrontend — not a reimplementation.
//
// For each clip it locates the loudest 1-second window (100 frames), averages
// the log-mel over that window into a 40-bin profile, and prints the two
// profiles side by side with their difference, plus low/mid/high band sums,
// spectral tilt, and the WakeWord peak sigmoid score.
//
//   brosoundml_wake_melcmp <fires.wav> <ignored.wav> [--weights PATH]

#include "brosoundml/audio.h"
#include "brosoundml/mel.h"
#include "brosoundml/wake.h"
#include "brosoundml/wake_data.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace bsm = brosoundml;

namespace {

constexpr int kSR  = 16000;
constexpr int kRF  = 100;   // receptive field in frames (1.0 s)

struct Clip {
    std::string name;
    std::vector<float> prof;   // 40-bin avg log-mel over loudest 1 s
    float peak_score = 0.0f;
    float in_peak = 0.0f;
    double dur_s = 0.0;
    int n_frames = 0;
};

std::vector<float> load16k(const std::string& path, float& in_peak, double& dur_s) {
    auto ab = bsm::read_wav(path);
    if (ab.sample_rate != kSR)
        ab.samples = bsm::resample_to(ab.samples, ab.sample_rate, kSR);
    in_peak = 0.0f;
    for (float v : ab.samples) { float a = std::fabs(v); if (a > in_peak) in_peak = a; }
    dur_s = static_cast<double>(ab.samples.size()) / kSR;
    // Equalise level to the AGC/training target (0.99 peak) so the log-mel
    // comparison reflects spectral SHAPE, not the raw amplitude difference
    // between a normalised TTS clip and a quieter live recording.
    if (in_peak > 0.0f) {
        const float g = 0.99f / in_peak;
        for (float& v : ab.samples) v *= g;
    }
    return ab.samples;
}

// Peak sigmoid score over the whole clip, using the real detector front-end+net.
float peak_score(bsm::WakeWord& w, const std::vector<float>& s) {
    w.reset();
    float best = 0.0f;
    const int N = static_cast<int>(s.size());
    for (int pos = 0; pos < N; pos += 320) {
        const int csz = std::min(320, N - pos);
        (void)w.feed(s.data() + pos, csz);
        const float v = w.last_score();
        if (v > best) best = v;
    }
    return best;
}

Clip analyze(const std::string& path, bsm::MelFrontend& mf, bsm::WakeWord& w) {
    Clip c;
    c.name = path;
    auto s = load16k(path, c.in_peak, c.dur_s);
    c.peak_score = peak_score(w, s);

    brotensor::Tensor out;
    mf.compute_offline(s.data(), static_cast<int>(s.size()), out);
    std::vector<float> M = out.to_host_vector();   // (40, T) row-major
    const int n_mels = mf.config().n_mels;
    const int T = (n_mels > 0) ? static_cast<int>(M.size()) / n_mels : 0;
    c.n_frames = T;

    c.prof.assign(static_cast<std::size_t>(n_mels), 0.0f);
    if (T <= 0) return c;

    // Per-frame total energy (sum over mel bins), then find the kRF-frame
    // window with the highest summed energy — the "loudest 1 s".
    std::vector<double> fe(static_cast<std::size_t>(T), 0.0);
    for (int t = 0; t < T; ++t) {
        double e = 0.0;
        for (int m = 0; m < n_mels; ++m) e += M[static_cast<std::size_t>(m) * T + t];
        fe[static_cast<std::size_t>(t)] = e;
    }
    const int win = std::min(kRF, T);
    double run = 0.0; for (int t = 0; t < win; ++t) run += fe[static_cast<std::size_t>(t)];
    double best = run; int best_start = 0;
    for (int t = win; t < T; ++t) {
        run += fe[static_cast<std::size_t>(t)] - fe[static_cast<std::size_t>(t - win)];
        if (run > best) { best = run; best_start = t - win + 1; }
    }
    // Average log-mel over the winning window.
    for (int m = 0; m < n_mels; ++m) {
        double acc = 0.0;
        for (int t = best_start; t < best_start + win; ++t)
            acc += M[static_cast<std::size_t>(m) * T + t];
        c.prof[static_cast<std::size_t>(m)] = static_cast<float>(acc / win);
    }
    return c;
}

// Mel-bin centre frequency (HTK), for labeling.
float mel_center_hz(int bin, int n_mels, float fmin, float fmax) {
    auto hz2mel = [](float f){ return 2595.0f * std::log10(1.0f + f / 700.0f); };
    auto mel2hz = [](float m){ return 700.0f * (std::pow(10.0f, m / 2595.0f) - 1.0f); };
    const float m0 = hz2mel(fmin), m1 = hz2mel(fmax);
    const float mm = m0 + (m1 - m0) * (bin + 0.5f) / n_mels;
    return mel2hz(mm);
}

}  // namespace

int main(int argc, char** argv) try {
    std::string a_path, b_path, weights = "weights/wake/computer.bw";
    bool use_pcen = false;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--weights" && i + 1 < argc) weights = argv[++i];
        else if (a == "--pcen") use_pcen = true;
        else pos.push_back(a);
    }
    if (pos.size() < 2) {
        std::fprintf(stderr,
            "usage: brosoundml_wake_melcmp <fires.wav> <ignored.wav> [--weights PATH]\n");
        return 2;
    }
    a_path = pos[0]; b_path = pos[1];

    brotensor::init();
    bsm::MelConfig mc;   // defaults match the model recipe
    if (use_pcen) mc.compression = bsm::MelCompression::PCEN;
    bsm::MelFrontend mf(mc, brotensor::Device::CPU);
    bsm::WakeWord w;
    w.load(weights, brotensor::Device::CPU);
    w.set_threshold(1.01f);   // never fire; read raw scores

    Clip A = analyze(a_path, mf, w);
    Clip B = analyze(b_path, mf, w);

    const int n = mc.n_mels;
    std::printf("front-end: %d-bin HTK log-mel, %d Hz, win %d, hop %d, fmax %.0f\n\n",
                n, mc.sample_rate, mc.win_length, mc.hop_length, mc.fmax);
    std::printf("A (fires?):  %s\n     in_peak=%.3f dur=%.2fs frames=%d  PEAK SCORE=%.4f\n",
                A.name.c_str(), A.in_peak, A.dur_s, A.n_frames, A.peak_score);
    std::printf("B (ignored): %s\n     in_peak=%.3f dur=%.2fs frames=%d  PEAK SCORE=%.4f\n\n",
                B.name.c_str(), B.in_peak, B.dur_s, B.n_frames, B.peak_score);

    std::printf("per-bin avg log-mel over each clip's loudest 1 s:\n");
    std::printf("  %3s %7s   %8s %8s %8s\n", "bin", "~Hz", "A", "B", "B-A");
    double loA[3] = {0,0,0}, loB[3] = {0,0,0};  // low/mid/high band sums
    for (int m = 0; m < n; ++m) {
        const float hz = mel_center_hz(m, n, mc.fmin, mc.fmax);
        const float av = A.prof[static_cast<std::size_t>(m)];
        const float bv = B.prof[static_cast<std::size_t>(m)];
        std::printf("  %3d %7.0f   %8.3f %8.3f %+8.3f\n", m, hz, av, bv, bv - av);
        const int band = hz < 1000.0f ? 0 : (hz < 4000.0f ? 1 : 2);
        loA[band] += av; loB[band] += bv;
    }
    std::printf("\nband-sum log-mel (low<1k, mid1-4k, high>4k):\n");
    std::printf("  %-6s  A=%8.2f  B=%8.2f  B-A=%+8.2f\n", "low",  loA[0], loB[0], loB[0]-loA[0]);
    std::printf("  %-6s  A=%8.2f  B=%8.2f  B-A=%+8.2f\n", "mid",  loA[1], loB[1], loB[1]-loA[1]);
    std::printf("  %-6s  A=%8.2f  B=%8.2f  B-A=%+8.2f\n", "high", loA[2], loB[2], loB[2]-loA[2]);

    double sumA = loA[0]+loA[1]+loA[2], sumB = loB[0]+loB[1]+loB[2];
    std::printf("\ntotal avg log-mel:  A=%.2f  B=%.2f  (B-A=%+.2f)\n",
                sumA, sumB, sumB - sumA);
    std::printf("spectral tilt (high-low):  A=%+.2f  B=%+.2f\n",
                loA[2]-loA[0], loB[2]-loB[0]);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "wake_melcmp: %s\n", e.what());
    return 1;
}
