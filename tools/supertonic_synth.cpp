// Supertonic-3 one-shot driver: load a converted model, synthesize one line,
// write a WAV. The quick listening check for the end-to-end pipeline (the
// frontend + flow-matching loop + vocoder), and small enough to profile.
//
//   supertonic_synth [model_dir] [text] [voice] [lang] [out.wav] [steps] [speed]
//
// `voice` is a preset name (M1..M5 / F1..F5, resolved to
// <model_dir>/voice_styles/<voice>.json) or a path to a voice-style JSON.
// model_dir defaults to $BROSOUNDML_SUPERTONIC_DIR, else ../brosoundml-data/supertonic.

#define _CRT_SECURE_NO_WARNINGS

#include "brosoundml/audio.h"
#include "brosoundml/supertonic.h"

#include <brotensor/runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

int main(int argc, char** argv) {
    brotensor::init();
    const brotensor::Device dev = brotensor::is_available(brotensor::Device::CUDA)
                                      ? brotensor::Device::CUDA
                                      : brotensor::Device::CPU;

    std::string root = argc > 1 ? argv[1] : "";
    if (root.empty()) {
        if (const char* e = std::getenv("BROSOUNDML_SUPERTONIC_DIR"); e && *e) root = e;
        else root = "../brosoundml-data/supertonic";
    }
    const std::string text  = argc > 2 ? argv[2]
        : "Hello, and welcome. This sentence was spoken by Supertonic, running on brosoundml.";
    const std::string voice = argc > 3 ? argv[3] : "M1";
    const std::string lang  = argc > 4 ? argv[4] : "en";
    const std::string out   = argc > 5 ? argv[5] : "supertonic_say.wav";
    const int   steps = argc > 6 ? std::atoi(argv[6]) : 8;
    const float speed = argc > 7 ? static_cast<float>(std::atof(argv[7])) : 1.05f;

    // A bare preset name resolves under voice_styles/; anything path-like is used as-is.
    std::string voice_path = voice;
    if (voice.find('/') == std::string::npos && voice.find('\\') == std::string::npos &&
        voice.find(".json") == std::string::npos)
        voice_path = (fs::path(root) / "voice_styles" / (voice + ".json")).string();

    std::printf("device: %s\n", dev == brotensor::Device::CUDA ? "CUDA" : "CPU");

    brosoundml::Supertonic model;
    auto t0 = clk::now();
    try {
        model.load(root, dev);
    } catch (const std::exception& e) {
        std::printf("load failed: %s\n", e.what());
        return 1;
    }
    std::printf("load: %.0f ms\n", ms_since(t0));

    brosoundml::VoiceStyle style;
    try {
        style = model.load_voice_style(voice_path);
    } catch (const std::exception& e) {
        std::printf("voice load failed: %s\n", e.what());
        return 1;
    }

    t0 = clk::now();
    brosoundml::AudioBuffer wav;
    try {
        wav = model.synthesize(text, lang, style, steps, speed);
    } catch (const std::exception& e) {
        std::printf("synthesize failed: %s\n", e.what());
        return 1;
    }
    brotensor::sync_all();
    const double synth_ms = ms_since(t0);
    const double audio_s  = static_cast<double>(wav.samples.size()) / wav.sample_rate;
    std::printf("synthesize: %.0f ms  audio=%.2fs  RT=%.2fx  (voice=%s lang=%s steps=%d speed=%.2f)\n",
                synth_ms, audio_s, audio_s / (synth_ms / 1000.0),
                voice.c_str(), lang.c_str(), steps, speed);

    wav.write_wav(out);
    std::printf("wrote %s\n", out.c_str());
    return 0;
}
