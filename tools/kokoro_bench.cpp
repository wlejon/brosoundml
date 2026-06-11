// brosoundml::kokoro_bench — Kokoro synthesis throughput benchmark.
//
// Phonemizes English text with the in-tree G2P (no external phonemizer),
// then runs Kokoro::synthesize repeatedly and reports per-iteration wall
// time, audio seconds, and the realtime factor (audio_s / synth_s).
//
// Set BROSOUNDML_KOKORO_PROFILE=1 to get the per-stage breakdown printed by
// the library on every synthesize call.
//
// Usage:
//   brosoundml_kokoro_bench [--model DIR] [--voice PATH] [--device cpu|cuda]
//                           [--text "..."] [--iters N] [--warmup N]
//                           [--speed F] [--out file.wav]
//
// Defaults resolve through the standard data-dir convention:
//   BROSOUNDML_DATA_DIR env > ../brosoundml-data

#include "brosoundml/audio.h"
#include "brosoundml/kokoro.h"
#include "brosoundml/g2p/lexicon.h"
#include "brosoundml/g2p/morphology.h"
#include "brosoundml/g2p/phoneme_adapter.h"
#include "brosoundml/g2p/phonemizer.h"
#include "brosoundml/g2p/pos_tagger.h"
#include "brosoundml/g2p/special_cases.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace g  = brosoundml::g2p;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "kokoro_bench: %s\n", msg.c_str());
    std::exit(2);
}

std::string default_data_dir() {
    if (const char* v = std::getenv("BROSOUNDML_DATA_DIR")) {
        if (v[0] && fs::exists(v)) return v;
    }
    return "../brosoundml-data";
}

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir, voice_path, device_str = "cuda";
    std::string lexicon_path, pos_path, out_path;
    std::string text =
        "The quick brown fox jumps over the lazy dog, and then runs far away "
        "into the forest where the evening light falls softly between the "
        "tall ancient trees.";
    int   iters  = 5;
    int   warmup = 1;
    float speed  = 1.0f;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string(what) + " needs a value");
            return argv[++i];
        };
        if      (a == "--model")      model_dir    = next("--model");
        else if (a == "--voice")      voice_path   = next("--voice");
        else if (a == "--device")     device_str   = next("--device");
        else if (a == "--text")       text         = next("--text");
        else if (a == "--iters")      iters        = std::atoi(next("--iters").c_str());
        else if (a == "--warmup")     warmup       = std::atoi(next("--warmup").c_str());
        else if (a == "--speed")      speed        = std::stof(next("--speed"));
        else if (a == "--lexicon")    lexicon_path = next("--lexicon");
        else if (a == "--pos-tagger") pos_path     = next("--pos-tagger");
        else if (a == "--out")        out_path     = next("--out");
        else if (a == "-h" || a == "--help") {
            std::printf("Usage: brosoundml_kokoro_bench [--model DIR] [--voice PATH] "
                        "[--device cpu|cuda] [--text STR] [--iters N] [--warmup N] "
                        "[--speed F] [--out file.wav]\n");
            return 0;
        }
        else die("unknown flag '" + a + "'");
    }

    const std::string data_dir = default_data_dir();
    if (model_dir.empty())    model_dir    = data_dir + "/kokoro";
    if (voice_path.empty())   voice_path   = model_dir + "/voices/af_heart.bin";
    if (lexicon_path.empty()) lexicon_path = data_dir + "/g2p/lexicon_en_us.bin";
    if (pos_path.empty())     pos_path     = data_dir + "/pos_tagger/model.bin";

    try {
        brotensor::init();
        brotensor::Device dev = brotensor::Device::CPU;
        if      (device_str == "cpu")  dev = brotensor::Device::CPU;
        else if (device_str == "cuda") dev = brotensor::Device::CUDA;
        else die("--device must be cpu|cuda");
        if (dev != brotensor::Device::CPU && !brotensor::is_available(dev))
            die("--device " + device_str + " not available in this build");

        // ─── Load model + voice ────────────────────────────────────────
        const auto t_load = clk::now();
        brosoundml::Kokoro k;
        k.load(model_dir, dev);
        const double load_ms = ms_since(t_load);
        brosoundml::Voice voice = k.load_voice(voice_path);
        std::fprintf(stderr, "model: %s  device: %s  load: %.0f ms\n",
                     model_dir.c_str(), device_str.c_str(), load_ms);
        std::fprintf(stderr, "voice: %s (%dx%d)\n",
                     voice.name.c_str(), voice.packs.rows, voice.packs.cols);

        // ─── In-tree G2P: text -> phoneme ids ──────────────────────────
        if (!fs::exists(lexicon_path)) die("missing lexicon: " + lexicon_path);
        if (!fs::exists(pos_path))     die("missing pos tagger: " + pos_path);
        g::Lexicon        lex   = g::Lexicon::load(lexicon_path);
        g::Morphology     morph(lex);
        g::SpecialCases   sc(lex);
        g::PosTagger      tagger = g::PosTagger::load(pos_path);
        g::PhonemeAdapter adapter(k.config().vocab);
        g::Phonemizer     phon(tagger, lex, morph, sc, adapter);

        const std::vector<int32_t> ids = phon.phonemize(text);
        if (ids.empty()) die("G2P produced no phonemes for the input text");
        std::fprintf(stderr, "text: %zu chars -> %zu phonemes\n",
                     text.size(), ids.size());

        // ─── Warmup ────────────────────────────────────────────────────
        brosoundml::AudioBuffer audio;
        for (int w = 0; w < warmup; ++w) {
            const auto t0 = clk::now();
            audio = k.synthesize(ids, voice, speed);
            std::fprintf(stderr, "warmup %d: %.0f ms\n", w, ms_since(t0));
        }

        // ─── Timed iterations ──────────────────────────────────────────
        double sum_ms = 0.0;
        for (int it = 0; it < iters; ++it) {
            const auto t0 = clk::now();
            audio = k.synthesize(ids, voice, speed);
            const double ms = ms_since(t0);
            sum_ms += ms;
            const double audio_s = static_cast<double>(audio.samples.size()) /
                                   audio.sample_rate;
            std::fprintf(stderr, "iter %d: %8.1f ms  audio %.2f s  %.2fx RT\n",
                         it, ms, audio_s, audio_s / (ms / 1000.0));
        }
        if (iters > 0) {
            const double mean_ms = sum_ms / iters;
            const double audio_s = static_cast<double>(audio.samples.size()) /
                                   audio.sample_rate;
            std::fprintf(stderr,
                         "mean: %.1f ms for %.2f s audio  ->  %.2fx realtime\n",
                         mean_ms, audio_s, audio_s / (mean_ms / 1000.0));
        }

        if (!out_path.empty()) {
            audio.write_wav(out_path);
            std::fprintf(stderr, "wrote %s (%zu samples)\n",
                         out_path.c_str(), audio.samples.size());
        }
        return 0;
    } catch (const std::exception& e) {
        die(std::string("error: ") + e.what());
    }
}
