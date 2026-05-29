#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

// brosoundml_wake_test — wake-word evaluation CLI.
//
// Three modes (selected by which flag(s) appear):
//
//   --wav PATH       Mode A: feed a single WAV and print every detection.
//   --dataset DIR    Mode B: run every clip in a chunk-3 manifest, print
//                     FRR / FPR (and optional threshold-sweep + per-class).
//   --bench          Bench mode: time WakeWord::feed over 10 s of synthetic
//                     input and report frames/sec + ms/frame.

#include "brosoundml/audio.h"
#include "brosoundml/wake.h"
#include "brosoundml/wake_data.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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

struct Args {
    std::string weights;
    std::string wav;
    std::string dataset;
    float       threshold        = 0.55f;
    bool        threshold_sweep  = false;
    bool        per_class        = false;
    bool        bench            = false;
    int         chunk_samples    = 320;   // ~20 ms @ 16 kHz
    bool        trace            = false; // single-clip: print per-chunk score
    bool        help             = false;
};

void print_help() {
    std::printf(
        "brosoundml_wake_test — wake-word eval CLI\n"
        "  --weights PATH      .bw checkpoint (required)\n"
        "  --wav PATH          single-clip mode: report detections in this WAV\n"
        "  --dataset DIR       dataset mode: FRR/FPR over a chunk-3 manifest\n"
        "  --threshold F       detection threshold (default 0.55)\n"
        "  --threshold-sweep   in dataset mode, print an ROC table over 0.05..0.95\n"
        "  --per-class         in dataset mode, also break FPR by class\n"
        "  --bench             time feed() over 10 s of synthetic input\n"
        "  --chunk-samples N   feed chunk size (default 320 = ~20 ms)\n");
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) fail("cli", "missing value for " + k);
            return std::string(argv[++i]);
        };
        if      (k == "--weights")          a.weights = next();
        else if (k == "--wav")              a.wav     = next();
        else if (k == "--dataset")          a.dataset = next();
        else if (k == "--threshold")        a.threshold = std::stof(next());
        else if (k == "--threshold-sweep")  a.threshold_sweep = true;
        else if (k == "--per-class")        a.per_class = true;
        else if (k == "--bench")            a.bench = true;
        else if (k == "--chunk-samples")    a.chunk_samples = std::stoi(next());
        else if (k == "--trace")            a.trace = true;
        else if (k == "--help" || k == "-h"){ a.help = true; return true; }
        else fail("cli", "unknown flag '" + k + "'");
    }
    return true;
}

// Mode A — single WAV. Returns exit code (0 if at least one detection).
int run_single_wav(const Args& a) {
    if (a.weights.empty()) fail("wake_test", "--weights is required");
    if (!fs::exists(a.weights)) {
        fail("wake_test", "weights file not found: '" + a.weights + "'");
    }
    if (!fs::exists(a.wav)) {
        fail("wake_test", "wav file not found: '" + a.wav + "'");
    }

    auto audio = bsm::read_wav(a.wav);
    if (audio.sample_rate != 16000) {
        std::printf("note: wav rate %d != 16000, resampling\n",
                    audio.sample_rate);
        audio.samples = bsm::resample_to(audio.samples, audio.sample_rate,
                                          16000);
        audio.sample_rate = 16000;
    }

    bsm::WakeWord w;
    w.load(a.weights, brotensor::Device::CPU);
    w.set_threshold(a.threshold);

    std::printf("wake_test: weights='%s' wav='%s'\n",
                a.weights.c_str(), a.wav.c_str());
    std::printf("           threshold=%.3f  samples=%zu  duration=%.3fs\n",
                a.threshold, audio.samples.size(),
                audio.duration_seconds());

    int detections = 0;
    const int N = static_cast<int>(audio.samples.size());
    int pos = 0;
    float peak_score = 0.0f;
    double peak_t = 0.0;
    while (pos < N) {
        const int csz = std::min(a.chunk_samples, N - pos);
        const bool fired = w.feed(audio.samples.data() + pos, csz);
        const double t = (pos + csz) / 16000.0;
        const float sc = w.last_score();
        if (sc > peak_score) { peak_score = sc; peak_t = t; }
        if (a.trace) std::printf("  t=%.3f  score=%.4f\n", t, sc);
        if (fired) {
            ++detections;
            std::printf("  detection #%d  t=%.3fs  score=%.4f\n",
                        detections, t, sc);
        }
        pos += csz;
    }
    // Peak score over the whole clip — the model's strongest response,
    // independent of the fire policy (threshold/smoothing/warmup). A near-zero
    // peak means the model never lit up at all; a high peak that didn't fire
    // points at the detector policy rather than the model.
    std::printf("           detections=%d  peak_score=%.4f @ t=%.3fs"
                "  final_score=%.4f\n",
                detections, peak_score, peak_t, w.last_score());
    return detections > 0 ? 0 : 1;
}

// Mode B — dataset eval.
int run_dataset(const Args& a) {
    if (a.weights.empty()) fail("wake_test", "--weights is required");
    if (!fs::exists(a.weights)) {
        fail("wake_test", "weights file not found: '" + a.weights + "'");
    }
    const fs::path root = fs::path(a.dataset);
    const fs::path manifest = root / "manifest.csv";
    if (!fs::exists(manifest)) {
        fail("wake_test", "manifest not found at '" + manifest.string() + "'");
    }

    auto rows = bsm::read_manifest(manifest.string());
    if (rows.empty()) fail("wake_test", "manifest is empty");
    std::printf("wake_test: dataset='%s' rows=%zu\n",
                root.string().c_str(), rows.size());

    // Pre-load each WAV once (resampling if needed) to keep the loop tight.
    struct Clip {
        std::vector<float> samples;
        int                label;
        std::string        clazz;
    };
    std::vector<Clip> clips;
    clips.reserve(rows.size());
    int skipped = 0;
    for (const auto& r : rows) {
        const fs::path p = root / r.path;
        if (!fs::exists(p)) { ++skipped; continue; }
        try {
            auto a_buf = bsm::read_wav(p.string());
            if (a_buf.sample_rate != 16000) {
                a_buf.samples = bsm::resample_to(a_buf.samples,
                                                  a_buf.sample_rate, 16000);
            }
            clips.push_back(Clip{std::move(a_buf.samples), r.label, r.clazz});
        } catch (const std::exception&) {
            ++skipped;
        }
    }
    if (skipped > 0) std::printf("           skipped=%d (missing/malformed)\n",
                                  skipped);

    bsm::WakeWord w;
    w.load(a.weights, brotensor::Device::CPU);

    auto eval_at = [&](float thr) {
        struct Bucket { int n = 0; int fired = 0; };
        Bucket pos_b, neg_b;
        std::vector<std::pair<std::string, Bucket>> by_class;
        auto get_class = [&](const std::string& c) -> Bucket& {
            for (auto& kv : by_class) if (kv.first == c) return kv.second;
            by_class.push_back({c, Bucket{}});
            return by_class.back().second;
        };

        w.set_threshold(thr);
        for (const auto& c : clips) {
            w.reset();
            bool any = false;
            const int N = static_cast<int>(c.samples.size());
            int pos = 0;
            while (pos < N) {
                const int csz = std::min(a.chunk_samples, N - pos);
                if (w.feed(c.samples.data() + pos, csz)) any = true;
                pos += csz;
            }
            if (c.label == 1) {
                ++pos_b.n;  if (any) ++pos_b.fired;
            } else {
                ++neg_b.n;  if (any) ++neg_b.fired;
            }
            Bucket& cb = get_class(c.clazz);
            ++cb.n; if (any) ++cb.fired;
        }
        struct Result {
            float frr, fpr;
            int pos_n, pos_fired, neg_n, neg_fired;
            std::vector<std::pair<std::string, Bucket>> by_class;
        };
        Result r;
        r.pos_n = pos_b.n; r.pos_fired = pos_b.fired;
        r.neg_n = neg_b.n; r.neg_fired = neg_b.fired;
        r.frr = pos_b.n > 0
            ? 100.0f * static_cast<float>(pos_b.n - pos_b.fired) / pos_b.n
            : 0.0f;
        r.fpr = neg_b.n > 0
            ? 100.0f * static_cast<float>(neg_b.fired) / neg_b.n
            : 0.0f;
        r.by_class = std::move(by_class);
        return r;
    };

    const auto def = eval_at(a.threshold);
    std::printf("\n=== threshold = %.3f ===\n", a.threshold);
    std::printf("FRR = %.2f%%  (positives missed:  %d/%d)\n",
                def.frr, def.pos_n - def.pos_fired, def.pos_n);
    std::printf("FPR = %.2f%%  (negatives flagged: %d/%d)\n",
                def.fpr, def.neg_fired, def.neg_n);

    if (a.per_class) {
        std::printf("\nper-class breakdown:\n");
        for (const auto& kv : def.by_class) {
            const float rate = kv.second.n > 0
                ? 100.0f * static_cast<float>(kv.second.fired) / kv.second.n
                : 0.0f;
            std::printf("  %-12s n=%4d  fired=%4d  rate=%.2f%%\n",
                        kv.first.c_str(), kv.second.n, kv.second.fired, rate);
        }
    }

    if (a.threshold_sweep) {
        std::printf("\nthreshold sweep:\n");
        std::printf("  thr     FRR%%    FPR%%\n");
        for (int i = 0; i < 10; ++i) {
            const float thr = 0.05f + 0.10f * static_cast<float>(i);
            const auto rs = eval_at(thr);
            std::printf("  %.2f   %6.2f  %6.2f\n", thr, rs.frr, rs.fpr);
        }
    }

    return 0;
}

// Bench mode — 10 s of synthetic input through feed(), timed.
int run_bench(const Args& a) {
    if (a.weights.empty()) fail("wake_test", "--weights is required");
    if (!fs::exists(a.weights)) {
        fail("wake_test", "weights file not found: '" + a.weights + "'");
    }

    bsm::WakeWord w;
    w.load(a.weights, brotensor::Device::CPU);

    const int sr = 16000;
    const int total_samples = sr * 10;
    std::vector<float> input(static_cast<std::size_t>(total_samples));
    std::mt19937 rng(0x5EED);
    std::uniform_real_distribution<float> uni(-0.3f, 0.3f);
    for (auto& s : input) s = uni(rng);

    const int hop = w.config().hop_length;
    const int expected_frames = (total_samples - w.config().win_length) / hop + 1;

    // Warm-up — one second to settle conv-state caches.
    {
        const int warm = sr;
        int pos = 0;
        while (pos < warm) {
            const int csz = std::min(a.chunk_samples, warm - pos);
            (void)w.feed(input.data() + pos, csz);
            pos += csz;
        }
        w.reset();
    }

    // Timed run.
    const auto t0 = std::chrono::steady_clock::now();
    int detections = 0;
    int pos = 0;
    while (pos < total_samples) {
        const int csz = std::min(a.chunk_samples, total_samples - pos);
        if (w.feed(input.data() + pos, csz)) ++detections;
        pos += csz;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double ms_per_frame = expected_frames > 0
        ? (secs * 1000.0) / expected_frames : 0.0;
    const double frames_per_sec = expected_frames > 0
        ? expected_frames / secs : 0.0;
    const double rtf = secs / 10.0;   // real-time factor

    std::printf("bench: 10.000 s audio, %d frames @ %d ms hop\n",
                expected_frames, 1000 / (sr / hop));
    std::printf("       wall = %.3f s  (rtf = %.4f)\n", secs, rtf);
    std::printf("       %.2f frames/sec  %.4f ms/frame\n",
                frames_per_sec, ms_per_frame);
    std::printf("       detections=%d  budget=<2.0 ms/frame  %s\n",
                detections, ms_per_frame < 2.0 ? "PASS" : "OVER");
    return 0;
}

}  // namespace

int main(int argc, char** argv) try {
    Args a;
    if (!parse_args(argc, argv, a) || a.help) { print_help(); return 0; }
    brotensor::init();

    if (a.bench)             return run_bench(a);
    if (!a.wav.empty())      return run_single_wav(a);
    if (!a.dataset.empty())  return run_dataset(a);

    std::fprintf(stderr, "wake_test: nothing to do — pass --wav, --dataset or --bench\n");
    print_help();
    return 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "wake_test: %s\n", e.what());
    return 1;
}
