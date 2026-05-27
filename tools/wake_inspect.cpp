// brosoundml_wake_inspect — validate a chunk-3 wake-word dataset.
//
// Walks the manifest, opens every referenced WAV, runs per-clip integrity
// checks (sample rate, duration, silent fraction, clipping, NaN), and rolls
// the results into a DatasetReport. Exits 0 if the report clears every
// configured threshold and 1 otherwise — drop into CI as a pre-training gate.

#include "brosoundml/audio.h"
#include "brosoundml/wake_data.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "wake_inspect: %s\n", msg.c_str());
    std::exit(2);
}

void usage(std::FILE* out) {
    std::fprintf(out,
        "Usage: brosoundml_wake_inspect [flags]\n"
        "  --dataset DIR             Dataset root (default ../brosoundml-data/wake/computer)\n"
        "  --manifest PATH           Manifest CSV (default <dataset>/manifest.csv)\n"
        "  --expected-rate N         Required WAV sample rate (default 16000)\n"
        "  --expected-duration-s F   Target duration in seconds (default 1.0)\n"
        "  --duration-tol-s F        +/- tolerance on duration (default 0.05)\n"
        "  --max-silent-pct F        Max %% of clips flagged silent (default 1.0)\n"
        "  --max-clipped-pct F       Max %% of clips flagged clipped (default 0.5)\n"
        "  --dump-listen N DIR       Copy N random valid clips into DIR for spot-checking\n"
        "  --seed N                  RNG seed for --dump-listen (default 42)\n"
        "  --strict                  Fail on any anomaly, not only threshold violations\n"
        "  --quiet                   Suppress per-row progress; print summary only\n"
        "  -h, --help                Show this help\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string dataset_dir   = "../brosoundml-data/wake/computer";
    std::string manifest_path;
    int   expected_rate       = 16000;
    float expected_duration_s = 1.0f;
    float duration_tol_s      = 0.05f;
    float max_silent_pct      = 1.0f;
    float max_clipped_pct     = 0.5f;
    int   dump_listen_n       = 0;
    std::string dump_listen_dir;
    std::uint64_t seed        = 42;
    bool strict = false;
    bool quiet  = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string(what) + " needs a value");
            return argv[++i];
        };
        if      (a == "--dataset")             dataset_dir = next("--dataset");
        else if (a == "--manifest")            manifest_path = next("--manifest");
        else if (a == "--expected-rate")       expected_rate = std::atoi(next("--expected-rate").c_str());
        else if (a == "--expected-duration-s") expected_duration_s = std::strtof(next("--expected-duration-s").c_str(), nullptr);
        else if (a == "--duration-tol-s")      duration_tol_s = std::strtof(next("--duration-tol-s").c_str(), nullptr);
        else if (a == "--max-silent-pct")      max_silent_pct = std::strtof(next("--max-silent-pct").c_str(), nullptr);
        else if (a == "--max-clipped-pct")     max_clipped_pct = std::strtof(next("--max-clipped-pct").c_str(), nullptr);
        else if (a == "--dump-listen") {
            dump_listen_n   = std::atoi(next("--dump-listen N").c_str());
            dump_listen_dir = next("--dump-listen DIR");
        }
        else if (a == "--seed")    seed = std::strtoull(next("--seed").c_str(), nullptr, 10);
        else if (a == "--strict")  strict = true;
        else if (a == "--quiet")   quiet  = true;
        else if (a == "-h" || a == "--help") { usage(stdout); return 0; }
        else die("unknown flag '" + a + "'");
    }

    if (manifest_path.empty())
        manifest_path = dataset_dir + "/manifest.csv";

    try {
        brotensor::init();

        if (!quiet)
            std::fprintf(stderr, "reading manifest %s\n", manifest_path.c_str());
        auto rows = brosoundml::read_manifest(manifest_path);
        if (!quiet)
            std::fprintf(stderr, "  %zu rows\n", rows.size());

        brosoundml::ValidationConfig cfg;
        cfg.expected_sample_rate = expected_rate;
        cfg.min_duration_s = expected_duration_s - duration_tol_s;
        cfg.max_duration_s = expected_duration_s + duration_tol_s;
        cfg.max_silent_pct  = max_silent_pct;
        cfg.max_clipped_pct = max_clipped_pct;

        // For --strict we treat any per-clip anomaly as fatal — drop the
        // tolerance to a vanishing fraction so even a single offending sample
        // counts.
        if (strict) {
            cfg.max_silent_pct  = 0.0f;
            cfg.max_clipped_pct = 0.0f;
        }

        // Pre-validation progress for sanity on large datasets. The library
        // helper itself is silent — keep the printer out of testable code.
        if (!quiet)
            std::fprintf(stderr, "validating %zu rows ...\n", rows.size());

        auto report = brosoundml::validate_dataset(rows, dataset_dir, cfg);

        if (!quiet && rows.size() >= 500)
            std::fprintf(stderr, "  done\n");

        brosoundml::print_report(report, stdout);

        // ─── Optional: copy a random valid sample into listen/ ─────────
        if (dump_listen_n > 0 && !dump_listen_dir.empty()) {
            std::vector<std::size_t> candidates;
            candidates.reserve(rows.size());
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const auto& row = rows[i];
                const std::string abs = dataset_dir + "/" + row.path;
                std::error_code ec;
                if (fs::exists(abs, ec)) candidates.push_back(i);
            }
            std::mt19937 rng(static_cast<std::uint32_t>(seed));
            std::shuffle(candidates.begin(), candidates.end(), rng);
            const std::size_t n = std::min<std::size_t>(
                static_cast<std::size_t>(dump_listen_n), candidates.size());
            fs::create_directories(dump_listen_dir);
            for (std::size_t k = 0; k < n; ++k) {
                const auto& row = rows[candidates[k]];
                const std::string src = dataset_dir + "/" + row.path;
                fs::path dst = fs::path(dump_listen_dir) /
                    fs::path(row.path).filename();
                std::error_code ec;
                fs::copy_file(src, dst,
                              fs::copy_options::overwrite_existing, ec);
                if (ec) std::fprintf(stderr,
                    "  warn: copy %s -> %s failed: %s\n",
                    src.c_str(), dst.string().c_str(), ec.message().c_str());
            }
            std::fprintf(stdout, "listen: copied %zu clip(s) to %s\n",
                         n, dump_listen_dir.c_str());
        }

        std::string reason;
        const bool ok = brosoundml::report_passes(report, cfg, &reason);
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
