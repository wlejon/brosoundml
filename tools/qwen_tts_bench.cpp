// Wall-clock benchmark for the Qwen3-TTS pipeline on CUDA. Uses only the public
// QwenTts API (load / decode_codes / synthesize) so it builds unchanged against
// any revision — letting us A/B the on-device codec work against the old
// host-fallback path by reverting just the implementation files.
//
//   qwen_tts_bench [model_dir]
//
// Times the codec decode across a range of frame counts (the sliding-window
// attention engages past 72 frames) and the full synthesize() path.

#define _CRT_SECURE_NO_WARNINGS

#include "brosoundml/qwen_tts.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using brosoundml::QwenTts;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

int main(int argc, char** argv) {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping bench\n");
        return 0;
    }
    const std::string root = (argc > 1)
        ? argv[1] : std::string("weights/qwen-tts/0.6B-customvoice");

    const bool bf16 = std::getenv("BROSOUNDML_QWEN_BF16") != nullptr;

    QwenTts q;
    q.load(root, brotensor::Device::CUDA,
           bf16 ? brosoundml::QwenTtsWeightPrecision::BF16
                : brosoundml::QwenTtsWeightPrecision::FP32);
    const int K = q.config().codec.num_quantizers;
    const int win = q.config().codec.sliding_window;
    std::printf("loaded %s on CUDA (K=%d codebooks, window=%d frames, 12.5 Hz%s)\n\n",
                root.c_str(), K, win, bf16 ? ", BF16 weights" : "");

    std::printf("== codec decode_codes (CUDA) ==\n");
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, 1023);
    for (int T : {72, 144, 300, 600, 1200}) {
        std::vector<int32_t> codes(static_cast<std::size_t>(K) * T);
        for (auto& c : codes) c = dist(rng);
        q.decode_codes(codes, K, T);          // warm up
        brotensor::sync_all();
        double best = 1e30;
        for (int i = 0; i < 5; ++i) {
            auto t0 = clk::now();
            auto wav = q.decode_codes(codes, K, T);
            brotensor::sync_all();
            best = std::min(best, ms_since(t0));
        }
        const double audio_s = T / 12.5;
        std::printf("  T=%4d  %8.1f ms   audio=%5.2fs  RT=%6.1fx  %.3f ms/frame  (windowed=%s)\n",
                    T, best, audio_s, audio_s / (best / 1000.0), best / T,
                    T > win ? "yes" : "no");
    }

    std::printf("\n== full synthesize() (CUDA) ==\n");
    const char* texts[] = {
        "Hello there.",
        "The quick brown fox jumps over the lazy dog near the riverbank at dawn, "
        "while a curious cat watches quietly from the old wooden fence nearby.",
    };
    for (const char* text : texts) {
        q.synthesize(text, "serena", "english");   // warm up
        brotensor::sync_all();
        double best = 1e30;
        std::size_t n = 0;
        for (int i = 0; i < 3; ++i) {
            auto t0 = clk::now();
            auto wav = q.synthesize(text, "serena", "english");
            brotensor::sync_all();
            best = std::min(best, ms_since(t0));
            n = wav.samples.size();
        }
        const double audio_s = static_cast<double>(n) / 24000.0;
        std::printf("  %8.1f ms  audio=%5.2fs  RT=%5.2fx  | \"%.48s\"\n",
                    best, audio_s, audio_s / (best / 1000.0), text);
    }
    return 0;
}
