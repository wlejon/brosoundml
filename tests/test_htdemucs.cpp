#define _CRT_SECURE_NO_WARNINGS
// HTDemucs (htdemucs_ft) source separation tests.
//
//   1. Contract: separate() before load() throws; load() of a missing folder
//      throws.
//   2. GPU, real weights: every fixture case (tests/fixtures/htdemucs.bin, from
//      tests/ref/gen_htdemucs_fixture.py: demucs 4.0.1 apply_model in strict
//      FP32 on CUDA, driven like sep.py) separated and compared per stem by
//      SDR, which must reach 50 dB (FP32 against FP32).
//   3. Options: a stems subset comes back in the order asked; progress counts
//      every network forward; a cancel before the first batch returns no
//      stems; mono input equals the same channel given as stereo; a network
//      that was not loaded cannot be asked for.
//
// Sections 2-3 skip (printing why) without a GPU backend, the weights
// (BROSOUNDML_HTDEMUCS or weights/htdemucs_ft from tools/htdemucs_convert.py)
// or the fixture.
#include "brosoundml/htdemucs.h"

#include <brotensor/runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_device.h"

namespace fs = std::filesystem;
using namespace brosoundml;

static int failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

template <typename Fn>
static bool throws_runtime_error(Fn&& fn) {
    try { fn(); }
    catch (const std::runtime_error&) { return true; }
    catch (...) { return false; }
    return false;
}

struct Case {
    int sample_rate = 0, frames = 0;
    std::vector<std::string> names;
    std::vector<float> input;
    std::vector<std::vector<float>> stems;
};

static bool read_fixture(const fs::path& path, std::vector<Case>& cases) {
    std::ifstream f(path.string(), std::ios::binary);
    if (!f) return false;
    auto i32 = [&]() { int32_t v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; };
    auto rd = [&](std::vector<float>& v, std::size_t n) {
        v.resize(n);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
    };
    if (i32() != 0x31445448) return false;
    const int n = i32();
    for (int c = 0; c < n; ++c) {
        Case k;
        k.sample_rate = i32();
        k.frames = i32();
        const int ns = i32();
        for (int s = 0; s < ns; ++s) {
            std::string name(static_cast<std::size_t>(i32()), '\0');
            f.read(name.data(), static_cast<std::streamsize>(name.size()));
            k.names.push_back(name);
        }
        rd(k.input, static_cast<std::size_t>(2) * k.frames);
        k.stems.resize(ns);
        for (auto& s : k.stems) rd(s, static_cast<std::size_t>(2) * k.frames);
        cases.push_back(std::move(k));
    }
    return static_cast<bool>(f);
}

static double sdr(const std::vector<float>& ref, const std::vector<float>& est) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double r = ref[i], e = r - est[i];
        num += r * r;
        den += e * e;
    }
    return 10.0 * std::log10(num / std::max(den, 1e-30));
}

static fs::path model_dir() {
    if (const char* e = std::getenv("BROSOUNDML_HTDEMUCS")) return e;
    return fs::path(BROSOUNDML_REPO_DIR) / "weights" / "htdemucs_ft";
}

static StereoAudio audio_of(const Case& k) {
    StereoAudio a;
    a.channels = 2;
    a.frames = k.frames;
    a.sample_rate = k.sample_rate;
    a.samples = k.input;
    return a;
}

int main() {
    brotensor::init();

    // ── 1. contract ────────────────────────────────────────────────────────
    {
        HtDemucs m;
        CHECK(!m.loaded(), "fresh instance reports loaded");
        StereoAudio a;
        a.frames = 100;
        a.samples.assign(200, 0.0f);
        CHECK(throws_runtime_error([&] { m.separate(a); }), "separate before load must throw");
        CHECK(throws_runtime_error([&] { m.load((fs::path(BROSOUNDML_REPO_DIR) / "no-such-model").string(),
                                               brotensor::Device::CPU); }),
              "load of a missing folder must throw");
    }

    const brotensor::Device gpu = brosoundml_test::preferred_gpu();
    const fs::path dir = model_dir();
    const fs::path fixture = fs::path(BROSOUNDML_REPO_DIR) / "tests" / "fixtures" / "htdemucs.bin";
    std::vector<Case> cases;
    if (!gpu.is_gpu()) {
        std::printf("skip: no GPU backend\n");
    } else if (!fs::exists(dir / "config.json")) {
        std::printf("skip: no weights at %s (tools/htdemucs_convert.py)\n", dir.string().c_str());
    } else if (!read_fixture(fixture, cases)) {
        std::printf("skip: no fixture at %s (tests/ref/gen_htdemucs_fixture.py)\n",
                    fixture.string().c_str());
    } else {
        std::printf("GPU: %s\n", brosoundml_test::device_name(gpu));
        HtDemucs m;
        m.load(dir.string(), gpu);
        CHECK(m.loaded(), "loaded");
        CHECK(m.config().sample_rate == 44100, "model rate 44100");
        CHECK(m.loaded_stems().size() == 4, "four networks loaded");

        // ── 2. parity ──────────────────────────────────────────────────────
        for (const Case& k : cases) {
            SeparateOptions o;
            o.stems = k.names;
            int last_done = 0, last_total = 0;
            o.on_progress = [&](int d, int t) { last_done = d; last_total = t; };
            const auto t0 = std::chrono::steady_clock::now();
            const Separation sep = m.separate(audio_of(k), o);
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            CHECK(!sep.cancelled, "not cancelled");
            CHECK(sep.names == k.names, "stems in the order asked");
            CHECK(last_done == last_total && last_total > 0, "progress reaches total");
            std::printf("%d Hz, %d frames, %.0f ms:", k.sample_rate, k.frames, ms);
            for (std::size_t s = 0; s < k.names.size() && s < sep.stems.size(); ++s) {
                const StereoAudio& st = sep.stems[s];
                const bool shape = st.frames == k.frames && st.channels == 2 &&
                                   st.sample_rate == k.sample_rate;
                CHECK(shape, "stem at the input's rate and length");
                if (!shape) continue;
                const double db = sdr(k.stems[s], st.samples);
                std::printf("  %s %.1f dB", k.names[s].c_str(), db);
                CHECK(db >= 50.0, ("SDR >= 50 dB: " + k.names[s]).c_str());
            }
            std::printf("\n");
        }

        // ── 3. options ─────────────────────────────────────────────────────
        const Case& k = cases.back();
        {
            SeparateOptions o;
            o.stems = {"vocals"};
            o.cancel = [] { return true; };
            const Separation sep = m.separate(audio_of(k), o);
            CHECK(sep.cancelled && sep.stems.empty(), "cancel returns no stems");
        }
        {
            StereoAudio mono = audio_of(k);
            mono.channels = 1;
            mono.samples.resize(static_cast<std::size_t>(k.frames));
            StereoAudio dup = audio_of(k);
            std::copy(mono.samples.begin(), mono.samples.end(), dup.samples.begin() + k.frames);
            SeparateOptions o;
            o.stems = {"bass", "vocals"};
            const Separation a = m.separate(mono, o), b = m.separate(dup, o);
            CHECK(a.names.size() == 2 && a.names[0] == "bass", "subset order");
            CHECK(a.stems.size() == 2 && b.stems.size() == 2 && a.stems[1].samples == b.stems[1].samples,
                  "mono input equals the channel duplicated");
            CHECK(a.find("vocals") != nullptr && a.find("drums") == nullptr, "find");
        }
        {
            HtDemucs v;
            v.load(dir.string(), gpu, {"vocals"});
            CHECK(v.loaded_stems() == std::vector<std::string>{"vocals"}, "vocals-only load");
            SeparateOptions o;
            o.stems = {"drums"};
            CHECK(throws_runtime_error([&] { v.separate(audio_of(k), o); }),
                  "a stem whose network was not loaded must throw");
            o.stems = {"karaoke"};
            CHECK(throws_runtime_error([&] { v.separate(audio_of(k), o); }), "unknown stem must throw");
        }
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("All htdemucs checks passed.\n");
    return 0;
}
