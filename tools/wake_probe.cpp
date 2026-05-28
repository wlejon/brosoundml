#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

// brosoundml_wake_probe — figure out what the trained wake model actually
// responds to.
//
// Drives a battery of synthetic 1-s stimuli (silence, sine tones across
// frequency × amplitude, white/pink/brown noise across amplitude, linear chirp
// sweeps, AM-modulated tones, click trains, two-tone vowel-formant pairs)
// through WakeWord::feed and records the peak sigmoid score over the clip.
//
// Then — if --dataset is passed — it samples N positives and N negatives at
// random from the manifest and scores them the same way, so the synthetic
// scores can be compared against known-firing and known-quiet inputs.
//
// Output: a sorted-by-score ranking. The top of the list tells you what the
// model loves; the bottom tells you what it ignores; positions of the dataset
// rows tell you where the real positive manifold sits relative to the probes.

#include "brosoundml/audio.h"
#include "brosoundml/wake.h"
#include "brosoundml/wake_data.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs  = std::filesystem;
namespace bsm = brosoundml;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

constexpr int   kSR        = 16000;
constexpr int   kClipLen   = kSR;          // 1.0 s per stimulus
constexpr int   kChunk     = 320;          // 20 ms feed chunks
constexpr float kHugeThresh = 1.01f;       // never fires — we read raw scores

struct Probe {
    std::string category;   // "silence" | "tone" | "noise" | "sweep" | "am" | "click" | "formant" | "positive" | "negative"
    std::string label;      // human-readable
    std::vector<float> samples;  // length kClipLen at 16 kHz
};

// ── Generators ────────────────────────────────────────────────────────────

std::vector<float> silence(float amp) {
    std::vector<float> x(kClipLen, 0.0f);
    if (amp > 0.0f) {
        // Tiny DC isn't really silence — use very-low-amp white noise instead.
        std::mt19937 rng(0xDEAD);
        std::uniform_real_distribution<float> u(-amp, amp);
        for (auto& v : x) v = u(rng);
    }
    return x;
}

std::vector<float> sine(float hz, float amp) {
    std::vector<float> x(kClipLen);
    const double w = 2.0 * 3.14159265358979323846 * hz / kSR;
    for (int n = 0; n < kClipLen; ++n) {
        x[static_cast<std::size_t>(n)] =
            amp * static_cast<float>(std::sin(w * n));
    }
    return x;
}

std::vector<float> two_tone(float f1, float f2, float amp) {
    std::vector<float> x(kClipLen);
    const double w1 = 2.0 * 3.14159265358979323846 * f1 / kSR;
    const double w2 = 2.0 * 3.14159265358979323846 * f2 / kSR;
    for (int n = 0; n < kClipLen; ++n) {
        x[static_cast<std::size_t>(n)] = amp * 0.5f *
            (static_cast<float>(std::sin(w1 * n)) +
             static_cast<float>(std::sin(w2 * n)));
    }
    return x;
}

// Linear chirp f0 → f1 over the clip.
std::vector<float> sweep(float f0, float f1, float amp) {
    std::vector<float> x(kClipLen);
    const double T = static_cast<double>(kClipLen) / kSR;
    const double k = (f1 - f0) / T;
    for (int n = 0; n < kClipLen; ++n) {
        const double t = static_cast<double>(n) / kSR;
        const double phase = 2.0 * 3.14159265358979323846 *
                             (f0 * t + 0.5 * k * t * t);
        x[static_cast<std::size_t>(n)] =
            amp * static_cast<float>(std::sin(phase));
    }
    return x;
}

// Amplitude-modulated tone: carrier_hz × (0.5 + 0.5·sin(2π·mod_hz·t)).
std::vector<float> am_tone(float carrier_hz, float mod_hz, float amp) {
    std::vector<float> x(kClipLen);
    const double wc = 2.0 * 3.14159265358979323846 * carrier_hz / kSR;
    const double wm = 2.0 * 3.14159265358979323846 * mod_hz / kSR;
    for (int n = 0; n < kClipLen; ++n) {
        const double env = 0.5 + 0.5 * std::sin(wm * n);
        x[static_cast<std::size_t>(n)] = amp *
            static_cast<float>(env * std::sin(wc * n));
    }
    return x;
}

// Click train at given Hz — each click is a single-sample impulse.
std::vector<float> clicks(float rate_hz, float amp) {
    std::vector<float> x(kClipLen, 0.0f);
    const int period = std::max(1, static_cast<int>(kSR / rate_hz));
    for (int n = 0; n < kClipLen; n += period) {
        x[static_cast<std::size_t>(n)] = amp;
    }
    return x;
}

std::vector<float> noise(bsm::NoiseKind k, float amp, std::uint32_t seed) {
    std::mt19937 rng(seed);
    return bsm::gen_noise(k, kClipLen, amp, rng);
}

// ── Scoring ───────────────────────────────────────────────────────────────

// Push a buffer through WakeWord::feed kChunk at a time, return peak score.
float peak_score(bsm::WakeWord& w, const std::vector<float>& samples) {
    w.reset();
    float best = 0.0f;
    const int N = static_cast<int>(samples.size());
    int pos = 0;
    while (pos < N) {
        const int csz = std::min(kChunk, N - pos);
        (void)w.feed(samples.data() + pos, csz);
        const float s = w.last_score();
        if (s > best) best = s;
        pos += csz;
    }
    return best;
}

// ── CLI ───────────────────────────────────────────────────────────────────

struct Args {
    std::string weights = "weights/wake/computer.bw";
    std::string dataset = "../brosoundml-data/wake/computer";
    int         dataset_sample = 20;   // N positives + N negatives
    std::uint32_t seed   = 1337;
    bool        help     = false;
};

void print_help() {
    std::printf(
        "brosoundml_wake_probe — characterise what the wake model responds to\n"
        "  --weights PATH       .bw checkpoint (default weights/wake/computer.bw)\n"
        "  --dataset DIR        manifest root for real-clip baselines\n"
        "                       (default ../brosoundml-data/wake/computer)\n"
        "  --dataset-sample N   how many positives + negatives to sample (default 20)\n"
        "  --seed N             RNG seed (default 1337)\n");
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) fail("cli", "missing value for " + k);
            return std::string(argv[++i]);
        };
        if      (k == "--weights")        a.weights = next();
        else if (k == "--dataset")        a.dataset = next();
        else if (k == "--dataset-sample") a.dataset_sample = std::stoi(next());
        else if (k == "--seed")           a.seed = static_cast<std::uint32_t>(
                                                  std::stoul(next()));
        else if (k == "--help" || k == "-h") { a.help = true; return true; }
        else fail("cli", "unknown flag '" + k + "'");
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) try {
    Args a;
    if (!parse_args(argc, argv, a) || a.help) { print_help(); return 0; }

    brotensor::init();
    if (!fs::exists(a.weights)) {
        fail("wake_probe", "weights not found: " + a.weights);
    }
    bsm::WakeWord w;
    w.load(a.weights, brotensor::Device::CPU);
    w.set_threshold(kHugeThresh);   // never fire — we only read scores

    std::printf("wake_probe: weights='%s'\n", a.weights.c_str());
    std::printf("            sample_rate=%d  clip=%.1fs  chunk=%d samples\n",
                kSR, kClipLen / float(kSR), kChunk);
    std::printf("            n_mels=%d  hop=%d  win=%d\n",
                w.config().n_mels, w.config().hop_length, w.config().win_length);

    // ── Build the probe battery ──
    std::vector<Probe> probes;

    // Silence + near-silence
    for (float amp : {0.0f, 1e-4f, 1e-3f, 1e-2f}) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "silence amp=%.4f", amp);
        probes.push_back({"silence", nm, silence(amp)});
    }

    // Pure tones — frequency × amplitude grid.
    const std::vector<float> tone_hz = {
        100.f, 250.f, 500.f, 1000.f, 1500.f, 2000.f, 2500.f,
        3000.f, 4000.f, 5000.f, 6000.f, 7000.f
    };
    const std::vector<float> tone_amp = {0.02f, 0.05f, 0.1f, 0.3f, 0.7f};
    for (float hz : tone_hz) {
        for (float amp : tone_amp) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "tone %4.0fHz amp=%.2f", hz, amp);
            probes.push_back({"tone", nm, sine(hz, amp)});
        }
    }

    // Noise × amplitude.
    for (auto k : {bsm::NoiseKind::White, bsm::NoiseKind::Pink,
                   bsm::NoiseKind::Brown}) {
        for (float amp : {0.02f, 0.05f, 0.1f, 0.3f, 0.7f}) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "noise %s amp=%.2f",
                          bsm::noise_kind_name(k), amp);
            probes.push_back({"noise", nm,
                              noise(k, amp, a.seed + static_cast<std::uint32_t>(
                                                std::hash<std::string>{}(nm)))});
        }
    }

    // Chirps.
    for (float amp : {0.1f, 0.3f}) {
        {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "sweep 200->4000Hz amp=%.2f", amp);
            probes.push_back({"sweep", nm, sweep(200.f, 4000.f, amp)});
        }
        {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "sweep 4000->200Hz amp=%.2f", amp);
            probes.push_back({"sweep", nm, sweep(4000.f, 200.f, amp)});
        }
        {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "sweep 100->8000Hz amp=%.2f", amp);
            probes.push_back({"sweep", nm, sweep(100.f, 8000.f, amp)});
        }
    }

    // AM tones — speech-rate envelope on speech-band carriers.
    for (float carrier : {500.f, 1000.f, 2000.f, 3000.f}) {
        for (float mod : {4.f, 10.f, 25.f}) {
            for (float amp : {0.1f, 0.3f}) {
                char nm[80];
                std::snprintf(nm, sizeof(nm),
                              "am %4.0fHz x %2.0fHz amp=%.2f",
                              carrier, mod, amp);
                probes.push_back({"am", nm, am_tone(carrier, mod, amp)});
            }
        }
    }

    // Click trains — plosive cadence.
    for (float rate : {5.f, 20.f, 50.f, 100.f}) {
        for (float amp : {0.3f, 0.7f}) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "clicks %3.0fHz amp=%.2f", rate, amp);
            probes.push_back({"click", nm, clicks(rate, amp)});
        }
    }

    // Two-tone vowel-formant pairs (F1, F2 approximations).
    struct Formant { const char* vowel; float f1; float f2; };
    const std::vector<Formant> vowels = {
        {"ee", 300.f, 2800.f},
        {"ih", 400.f, 2000.f},
        {"eh", 600.f, 1900.f},
        {"ae", 700.f, 1700.f},
        {"ah", 800.f, 1200.f},
        {"aw", 600.f,  900.f},
        {"uh", 600.f, 1200.f},
        {"oo", 300.f,  900.f},
        // The vowel-shaped sequence in "computer" approximately spans:
        //   "uh" → "ow" → "oo" → "er"
        {"ow", 500.f,  800.f},
        {"er", 500.f, 1400.f},
    };
    for (const auto& v : vowels) {
        for (float amp : {0.1f, 0.3f}) {
            char nm[80];
            std::snprintf(nm, sizeof(nm), "formant /%s/ %4.0f+%4.0fHz amp=%.2f",
                          v.vowel, v.f1, v.f2, amp);
            probes.push_back({"formant", nm, two_tone(v.f1, v.f2, amp)});
        }
    }

    // ── Real-data baselines ──
    int loaded_pos = 0, loaded_neg = 0;
    if (!a.dataset.empty() && fs::exists(fs::path(a.dataset) / "manifest.csv")) {
        const std::string manifest =
            (fs::path(a.dataset) / "manifest.csv").string();
        auto rows = bsm::read_manifest(manifest);
        std::vector<int> pos_idx, neg_idx;
        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            (rows[static_cast<std::size_t>(i)].label == 1 ? pos_idx : neg_idx)
                .push_back(i);
        }
        std::mt19937 rng(a.seed);
        std::shuffle(pos_idx.begin(), pos_idx.end(), rng);
        std::shuffle(neg_idx.begin(), neg_idx.end(), rng);
        const int npos = std::min(a.dataset_sample,
                                  static_cast<int>(pos_idx.size()));
        const int nneg = std::min(a.dataset_sample,
                                  static_cast<int>(neg_idx.size()));
        auto load_clip = [&](int ri, const char* tag) {
            const auto& r = rows[static_cast<std::size_t>(ri)];
            const fs::path p = fs::path(a.dataset) / r.path;
            try {
                auto ab = bsm::read_wav(p.string());
                if (ab.sample_rate != kSR) {
                    ab.samples = bsm::resample_to(ab.samples,
                                                   ab.sample_rate, kSR);
                }
                if (static_cast<int>(ab.samples.size()) < kClipLen) {
                    ab.samples.resize(kClipLen, 0.0f);
                } else if (static_cast<int>(ab.samples.size()) > kClipLen) {
                    ab.samples.resize(kClipLen);
                }
                char nm[160];
                std::snprintf(nm, sizeof(nm), "%s %s (%s)",
                              tag, r.path.c_str(), r.clazz.c_str());
                probes.push_back({tag, nm, std::move(ab.samples)});
                return true;
            } catch (const std::exception&) {
                return false;
            }
        };
        for (int i = 0; i < npos; ++i) {
            if (load_clip(pos_idx[static_cast<std::size_t>(i)], "positive"))
                ++loaded_pos;
        }
        for (int i = 0; i < nneg; ++i) {
            if (load_clip(neg_idx[static_cast<std::size_t>(i)], "negative"))
                ++loaded_neg;
        }
    }

    std::printf("            probes=%zu  (pos=%d neg=%d real)\n\n",
                probes.size(), loaded_pos, loaded_neg);

    // ── Score every probe ──
    struct Scored { float score; std::string category; std::string label; };
    std::vector<Scored> out;
    out.reserve(probes.size());
    for (auto& p : probes) {
        const float s = peak_score(w, p.samples);
        out.push_back({s, p.category, p.label});
    }

    // Sort high-to-low.
    std::sort(out.begin(), out.end(),
              [](const Scored& a, const Scored& b) { return a.score > b.score; });

    // Per-category summary first.
    struct CatStats { int n = 0; float max = 0.f; double sum = 0.; };
    std::vector<std::pair<std::string, CatStats>> cats;
    auto get_cat = [&](const std::string& c) -> CatStats& {
        for (auto& kv : cats) if (kv.first == c) return kv.second;
        cats.push_back({c, CatStats{}});
        return cats.back().second;
    };
    for (const auto& r : out) {
        auto& c = get_cat(r.category);
        ++c.n;
        c.sum += r.score;
        if (r.score > c.max) c.max = r.score;
    }
    std::printf("=== per-category summary ===\n");
    std::printf("  %-10s  %4s   %7s   %7s\n", "category", "n", "mean", "max");
    // Stable sort cats by max desc for readability.
    std::sort(cats.begin(), cats.end(),
              [](const auto& a, const auto& b) {
                  return a.second.max > b.second.max;
              });
    for (const auto& kv : cats) {
        const float mean = kv.second.n > 0
            ? static_cast<float>(kv.second.sum / kv.second.n) : 0.0f;
        std::printf("  %-10s  %4d   %7.4f   %7.4f\n",
                    kv.first.c_str(), kv.second.n, mean, kv.second.max);
    }

    // Full ranking — top + bottom + dataset rows always shown.
    std::printf("\n=== full ranking (peak sigmoid score per stimulus) ===\n");
    for (const auto& r : out) {
        std::printf("  %7.4f  [%-9s]  %s\n",
                    r.score, r.category.c_str(), r.label.c_str());
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "wake_probe: %s\n", e.what());
    return 1;
}
