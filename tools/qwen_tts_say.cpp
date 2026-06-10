// Minimal Qwen3-TTS one-shot driver: load a checkpoint, synthesize one line,
// write a WAV. Small enough to sit under nsys/ncu for kernel-level profiling
// (the bench's codec sweeps would otherwise dominate the trace), and handy for
// quick listening checks.
//
//   qwen_tts_say [model_dir] [text] [speaker] [language] [out.wav]

#define _CRT_SECURE_NO_WARNINGS

#include "brosoundml/audio.h"
#include "brosoundml/qwen_tts.h"

#include <brotensor/runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

int main(int argc, char** argv) {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available\n");
        return 1;
    }
    const std::string root     = argc > 1 ? argv[1] : "weights/qwen-tts/0.6B-customvoice";
    const std::string text     = argc > 2 ? argv[2]
        : "The quick brown fox jumps over the lazy dog near the riverbank at dawn.";
    const std::string speaker  = argc > 3 ? argv[3] : "serena";
    const std::string language = argc > 4 ? argv[4] : "english";
    const std::string out      = argc > 5 ? argv[5] : "qwen_say.wav";

    const bool bf16 = std::getenv("BROSOUNDML_QWEN_BF16") != nullptr;
    const auto precision = bf16 ? brosoundml::QwenTtsWeightPrecision::BF16
                                : brosoundml::QwenTtsWeightPrecision::FP32;

    brosoundml::QwenTts q;
    auto t0 = clk::now();
    q.load(root, brotensor::Device::CUDA, precision);
    std::printf("load: %.0f ms%s\n", ms_since(t0), bf16 ? "  (BF16 weights)" : "");

    t0 = clk::now();
    brosoundml::AudioBuffer wav = q.synthesize(text, speaker, language);
    brotensor::sync_all();
    const double ms = ms_since(t0);
    const double audio_s = static_cast<double>(wav.samples.size()) / wav.sample_rate;
    std::printf("synthesize: %.0f ms  audio=%.2fs  RT=%.2fx\n",
                ms, audio_s, audio_s / (ms / 1000.0));

    wav.write_wav(out);
    std::printf("wrote %s\n", out.c_str());
    return 0;
}
