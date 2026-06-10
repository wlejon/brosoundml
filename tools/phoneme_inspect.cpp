// brosoundml_phoneme_inspect — validate a phoneme-posterior BPDS dataset.
//
// The phoneme analogue of wake_inspect: reads a frame-labelled BPDS file,
// runs validate_phoneme_dataset (per-clip length-invariant checks, label
// range, sample-rate, per-class frame coverage, dataset-wide silence
// fraction), prints the report (incl. the per-class frame histogram), and
// exits 0 if the report clears every configured gate, 1 otherwise — a drop-in
// pre-training CI gate.

#include "brosoundml/phoneme_data.h"

#include <brotensor/runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "phoneme_inspect: %s\n", msg.c_str());
    std::exit(2);
}

void usage(std::FILE* out) {
    std::fprintf(out,
        "Usage: brosoundml_phoneme_inspect [flags]\n"
        "  --dataset PATH            BPDS file (default ../brosoundml-data/phoneme/english.bpds)\n"
        "  --expected-rate N         Required sample rate (default 16000)\n"
        "  --min-frames-per-class N  Per-class minimum frame coverage (default 1)\n"
        "  --max-silence-fraction F  Max dataset-wide silence fraction (default 0.95)\n"
        "  --strict                  Fail on any structural anomaly\n"
        "  --quiet                   Suppress progress; print summary only\n"
        "  -h, --help                Show this help\n");
}

std::string env_or_empty(const char* name) {
    if (const char* v = std::getenv(name)) return std::string(v);
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    std::string default_data = env_or_empty("BROSOUNDML_DATA_DIR");
    if (default_data.empty()) default_data = "../brosoundml-data";

    std::string dataset_path = default_data + "/phoneme/english.bpds";
    int       expected_rate      = 16000;
    long long min_frames_per_cls = 1;
    float     max_silence_frac   = 0.95f;
    bool      strict = false;
    bool      quiet  = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string(what) + " needs a value");
            return argv[++i];
        };
        if      (a == "--dataset")              dataset_path = next("--dataset");
        else if (a == "--expected-rate")        expected_rate = std::atoi(next("--expected-rate").c_str());
        else if (a == "--min-frames-per-class") min_frames_per_cls = std::strtoll(next("--min-frames-per-class").c_str(), nullptr, 10);
        else if (a == "--max-silence-fraction") max_silence_frac = std::strtof(next("--max-silence-fraction").c_str(), nullptr);
        else if (a == "--strict")               strict = true;
        else if (a == "--quiet")                quiet  = true;
        else if (a == "-h" || a == "--help") { usage(stdout); return 0; }
        else die("unknown flag '" + a + "'");
    }

    try {
        brotensor::init();

        if (!quiet)
            std::fprintf(stderr, "reading dataset %s\n", dataset_path.c_str());
        auto ds = brosoundml::read_phoneme_dataset(dataset_path);
        if (!quiet)
            std::fprintf(stderr, "  %zu clips, K=%d classes\n",
                         ds.clips.size(), ds.class_map.num_classes);

        brosoundml::PhonemeValidationConfig cfg;
        cfg.expected_sample_rate  = expected_rate;
        cfg.min_frames_per_class  = min_frames_per_cls;
        cfg.max_silence_fraction  = max_silence_frac;

        // --strict tightens the dataset-wide silence gate to a vanishing
        // fraction so even a mostly-silent dataset is flagged.
        if (strict) cfg.max_silence_fraction = 0.0f;

        auto report = brosoundml::validate_phoneme_dataset(ds, cfg);
        brosoundml::print_report(report, ds.class_map, stdout);

        const double sil_frac = (report.total_frames > 0)
            ? static_cast<double>(report.silence_frames) /
              static_cast<double>(report.total_frames)
            : 0.0;
        std::fprintf(stdout,
            "summary: clips=%d frames=%lld silence_frames=%lld (%.3f) classes=%d\n",
            report.total_clips, report.total_frames, report.silence_frames,
            sil_frac, ds.class_map.num_classes);

        std::string reason;
        const bool ok = brosoundml::report_passes(report, ds.class_map, cfg, &reason);
        if (ok) {
            std::fprintf(stdout, "PASS\n");
            return 0;
        }
        std::fprintf(stdout, "FAIL: %s\n", reason.c_str());
        return 1;
    } catch (const std::exception& e) {
        die(std::string("error: ") + e.what());
    }
}
