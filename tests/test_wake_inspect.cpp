// Unit tests for the dataset-validator helpers behind brosoundml_wake_inspect.
// All synthetic — no Kokoro, no on-disk dataset fixture required.

#include "brosoundml/audio.h"
#include "brosoundml/wake_data.h"

#include <brotensor/runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

namespace { int tmp_counter = 0; }
static fs::path make_tmp_root() {
    auto root = fs::temp_directory_path() /
        ("brosoundml_wake_inspect_" + std::to_string(++tmp_counter));
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    return root;
}

// Write a tiny mono PCM WAV at a chosen sample rate by leaning on AudioBuffer.
static void write_wav(const fs::path& p, const std::vector<float>& samples,
                      int sample_rate) {
    fs::create_directories(p.parent_path());
    brosoundml::AudioBuffer buf(samples, sample_rate);
    buf.write_wav(p.string());
}

static std::vector<float> sine(int n, float freq_hz, int sr) {
    std::vector<float> x(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] = 0.5f * std::sin(
            2.0f * 3.14159265f * freq_hz *
            static_cast<float>(i) / static_cast<float>(sr));
    }
    return x;
}

int main() {
    brotensor::init();

    // ─── Manifest round-trip ───────────────────────────────────────────
    {
        auto root = make_tmp_root();
        const auto mpath = (root / "manifest.csv").string();
        {
            brosoundml::Manifest m(mpath);
            m.append({"positives/a.wav",       1, "positive",    "af_alloy",
                      1.0f,  0.0f, "",       7});
            m.append({"negatives/b,c.wav",     0, "sentence",    "af_bella",
                      0.95f, 5.0f, "pink",   11});
            m.append({"negatives/q\"x.wav",    0, "noise",       "",
                      1.0f,  0.0f, "white",  13});
            m.append({"negatives/conf_x.wav",  0, "confusable",  "am_eric",
                      1.05f, 10.0f,"brown",  19});
        }

        auto rows = brosoundml::read_manifest(mpath);
        CHECK(rows.size() == 4, "read_manifest: row count");
        CHECK(rows[0].path == "positives/a.wav",   "rt: row0 path");
        CHECK(rows[0].label == 1,                  "rt: row0 label");
        CHECK(rows[0].clazz == "positive",         "rt: row0 class");
        CHECK(rows[0].voice == "af_alloy",         "rt: row0 voice");
        CHECK(std::fabs(rows[0].speed - 1.0f) < 1e-5f,  "rt: row0 speed");
        CHECK(std::fabs(rows[0].snr_db - 0.0f) < 1e-5f, "rt: row0 snr");
        CHECK(rows[0].noise_kind.empty(),          "rt: row0 noise");
        CHECK(rows[0].seed == 7,                   "rt: row0 seed");
        CHECK(rows[1].path == "negatives/b,c.wav", "rt: row1 unescapes comma");
        CHECK(rows[2].path == "negatives/q\"x.wav","rt: row2 unescapes quote");
        CHECK(rows[3].clazz == "confusable",       "rt: row3 class");

        // Unknown-class rejection.
        const auto bad = (root / "bad.csv").string();
        {
            brosoundml::Manifest m(bad);
            brosoundml::ManifestRow r;
            r.path = "x.wav"; r.label = 0; r.clazz = "weird";
            r.voice = ""; r.speed = 1.0f; r.snr_db = 0.0f;
            r.noise_kind = ""; r.seed = 1;
            m.append(r);
        }
        bool threw = false;
        try { (void)brosoundml::read_manifest(bad); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw, "read_manifest: unknown class throws");

        // Missing header rejection.
        const auto hdr = (root / "noheader.csv").string();
        {
            std::FILE* f = nullptr;
#if defined(_MSC_VER)
            (void)fopen_s(&f, hdr.c_str(), "wb");
#else
            f = std::fopen(hdr.c_str(), "wb");
#endif
            std::fputs("nope,nada\n", f);
            std::fclose(f);
        }
        threw = false;
        try { (void)brosoundml::read_manifest(hdr); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw, "read_manifest: bad header throws");

        std::error_code ec; fs::remove_all(root, ec);
    }

    // ─── Audio stats ───────────────────────────────────────────────────
    {
        std::vector<float> silence(8000, 0.0f);
        auto s = brosoundml::compute_audio_stats(silence);
        CHECK(s.peak == 0.0f,           "stats: silence peak");
        CHECK(s.rms  == 0.0f,           "stats: silence rms");
        CHECK(s.silent_samples == 8000, "stats: silence count");
        CHECK(s.clipped_samples == 0,   "stats: silence clipped");
        CHECK(!s.has_nan,               "stats: silence no nan");

        auto sn = sine(16000, 440.0f, 16000);
        auto ss = brosoundml::compute_audio_stats(sn);
        CHECK(std::fabs(ss.peak - 0.5f) < 0.01f, "stats: sine peak ~= amp");
        const float expect_rms = 0.5f / std::sqrt(2.0f);
        CHECK(std::fabs(ss.rms - expect_rms) < 0.01f,
              "stats: sine rms ~= amp/sqrt(2)");

        std::vector<float> clipped(1000, 0.9995f);
        auto sc = brosoundml::compute_audio_stats(clipped);
        CHECK(sc.clipped_samples == 1000, "stats: clipped count");

        std::vector<float> withnan(100, 0.1f);
        withnan[42] = std::nanf("");
        auto sx = brosoundml::compute_audio_stats(withnan);
        CHECK(sx.has_nan, "stats: nan flagged");
    }

    // ─── Label / path consistency ─────────────────────────────────────
    {
        auto root = make_tmp_root();
        fs::create_directories(root / "positives");
        fs::create_directories(root / "negatives");
        const int sr = 16000;
        auto good = sine(sr, 440.0f, sr);
        write_wav(root / "positives" / "ok.wav",  good, sr);
        write_wav(root / "negatives" / "bad.wav", good, sr);

        std::vector<brosoundml::ManifestRow> rows = {
            // label=1 but path under negatives/  →  mismatch
            {"negatives/bad.wav", 1, "positive", "v", 1.0f, 0.0f, "", 1},
            // label=0 but path under positives/  →  mismatch
            {"positives/ok.wav",  0, "noise",    "",  1.0f, 0.0f, "white", 2},
        };
        brosoundml::ValidationConfig cfg;
        auto rep = brosoundml::validate_dataset(rows, root.string(), cfg);
        CHECK(rep.label_path_mismatch == 2,
              "validate: catches mismatched label/path prefixes");

        std::error_code ec; fs::remove_all(root, ec);
    }

    // ─── Threshold gating ─────────────────────────────────────────────
    {
        brosoundml::DatasetReport rep;
        rep.total_rows = 100;
        // Need at least one of each class to satisfy class-balance.
        rep.label_counts = {{"0", 50}, {"1", 50}};
        brosoundml::ValidationConfig cfg;
        std::string reason;
        CHECK(brosoundml::report_passes(rep, cfg, &reason),
              "report_passes: empty anomalies → true");
        CHECK(reason.empty(), "report_passes: clean reason");

        // 2% silent > 1% threshold.
        rep.silent_clips = 2;
        CHECK(!brosoundml::report_passes(rep, cfg, &reason),
              "report_passes: silent above threshold → false");
        CHECK(reason.find("silent") != std::string::npos,
              "report_passes: reason names silent");

        rep.silent_clips = 0;
        rep.clipped_clips = 1;  // 1% > 0.5%
        CHECK(!brosoundml::report_passes(rep, cfg, &reason),
              "report_passes: clipped above threshold → false");
        CHECK(reason.find("clipped") != std::string::npos,
              "report_passes: reason names clipped");

        // Missing class → fail with that reason.
        brosoundml::DatasetReport rep2;
        rep2.total_rows = 5;
        rep2.label_counts = {{"0", 5}};
        std::string r2;
        CHECK(!brosoundml::report_passes(rep2, cfg, &r2),
              "report_passes: missing positive class → false");
        CHECK(r2.find("positive") != std::string::npos,
              "report_passes: reason names missing positive");
    }

    // ─── Sample-rate mismatch ─────────────────────────────────────────
    {
        auto root = make_tmp_root();
        fs::create_directories(root / "positives");
        const int sr_bad = 8000;
        auto buf = sine(sr_bad, 440.0f, sr_bad);
        write_wav(root / "positives" / "lo.wav", buf, sr_bad);

        std::vector<brosoundml::ManifestRow> rows = {
            {"positives/lo.wav", 1, "positive", "v", 1.0f, 0.0f, "", 1},
        };
        brosoundml::ValidationConfig cfg;  // expects 16 kHz, 0.95-1.05 s
        auto rep = brosoundml::validate_dataset(rows, root.string(), cfg);
        CHECK(rep.sample_rate_mismatch == 1,
              "validate: 8 kHz wav vs 16 kHz expectation flagged");

        std::error_code ec; fs::remove_all(root, ec);
    }

    // ─── Missing file is caught ───────────────────────────────────────
    {
        auto root = make_tmp_root();
        std::vector<brosoundml::ManifestRow> rows = {
            {"positives/nope.wav", 1, "positive", "v", 1.0f, 0.0f, "", 1},
            {"negatives/n.wav",    0, "noise",    "",  1.0f, 0.0f, "white", 2},
        };
        brosoundml::ValidationConfig cfg;
        auto rep = brosoundml::validate_dataset(rows, root.string(), cfg);
        CHECK(rep.missing_files == 2, "validate: missing files counted");
        std::string reason;
        CHECK(!brosoundml::report_passes(rep, cfg, &reason),
              "report_passes: missing files → false");
        CHECK(reason.find("missing") != std::string::npos,
              "report_passes: reason names missing files");
        std::error_code ec; fs::remove_all(root, ec);
    }

    if (failures) {
        std::fprintf(stderr, "test_wake_inspect: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_wake_inspect: OK\n");
    return 0;
}
