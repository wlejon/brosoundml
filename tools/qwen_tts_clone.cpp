// Qwen3-TTS Base zero-shot voice clone: speak a line in the voice of a
// reference clip. Enrolls the clip via the ECAPA-TDNN speaker encoder (x-vector)
// and runs synthesize_clone — no reference transcript needed.
//
//   qwen_tts_clone <ref.wav> "<text to speak>" [out.wav] [language] [model_dir]
//
// model_dir defaults to a Base checkpoint (the only variant with a speaker
// encoder). Prefers CUDA; the speaker encoder runs host-side either way
// (enrollment is one-shot), the AR loop + codec decode run on the load device.

#define _CRT_SECURE_NO_WARNINGS

#include "brosoundml/audio.h"
#include "brosoundml/qwen_tts.h"

#include <brotensor/runtime.h>

#include <chrono>
#include <cstdio>
#include <string>

using brosoundml::AudioBuffer;
using brosoundml::QwenTts;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: qwen_tts_clone <ref.wav> \"<text>\" [out.wav] [language] [model_dir]\n");
        return 2;
    }
    const std::string ref_path = argv[1];
    const std::string text     = argv[2];
    const std::string out_path = (argc > 3) ? argv[3] : "clone.wav";
    const std::string language = (argc > 4) ? argv[4] : "english";
    const std::string root     = (argc > 5)
        ? argv[5] : std::string("weights/qwen-tts/0.6B-Base");

    brotensor::init();
    const bool cuda = brotensor::is_available(brotensor::Device::CUDA);
    const brotensor::Device dev =
        cuda ? brotensor::Device::CUDA : brotensor::Device::CPU;

    QwenTts q;
    q.load(root, dev);
    std::printf("loaded %s on %s\n", root.c_str(), cuda ? "CUDA" : "CPU");
    if (!q.config().speaker_encoder.present) {
        std::fprintf(stderr,
            "error: %s has no speaker encoder; the zero-shot clone needs a Base "
            "variant\n", root.c_str());
        return 1;
    }

    AudioBuffer ref = brosoundml::read_wav(ref_path);
    std::printf("reference: %s  %.2fs @ %d Hz\n", ref_path.c_str(),
                ref.duration_seconds(), ref.sample_rate);

    auto t0 = clk::now();
    AudioBuffer out = q.synthesize_clone(text, ref, language);
    brotensor::sync_all();
    const double ms =
        std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    const double audio_s = out.duration_seconds();
    std::printf("cloned: %zu samples  %.2fs  (%.1f ms, %.2fx RT)\n  \"%.60s\"\n",
                out.samples.size(), audio_s, ms, audio_s / (ms / 1000.0),
                text.c_str());

    out.write_wav(out_path);
    std::printf("wrote %s\n", out_path.c_str());
    return 0;
}
