// brosoundml::qwen_asr_transcribe — command-line Qwen3-ASR driver.
//
// Reads a 16 kHz mono WAV, runs QwenAsr::transcribe end-to-end against a
// model directory + the bundled Qwen BPE tokenizer files, and prints the
// detected language and transcript to stdout. Errors go to stderr.
//
// Usage:
//   brosoundml_qwen_asr_transcribe <wav> <model_dir>
//                                  [--device cpu|cuda] [--context TEXT]
//                                  [--max-new-tokens N] [--stream] [--ids]
//
// Notes:
//   * The WAV must be 16 kHz mono PCM — the model's input rate is fixed and
//     resampling lives outside this CLI (use ffmpeg or sox to convert).
//   * `--context` biases recognition toward the given text (names, domain
//     terms); it lands in the chat template's system block.
//   * `--stream` prints the raw model stream incrementally as each token
//     decodes instead of waiting for the whole clip.
//   * The model emits "language <Language><asr_text>transcript"; this CLI
//     splits that into a "language:" line and the transcript.

#include "brosoundml/audio.h"
#include "brosoundml/qwen_asr.h"

#include <brolm/qwen_tokenizer.h>

#include <brotensor/runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "brosoundml_qwen_asr_transcribe: %s\n", msg.c_str());
    std::exit(2);
}

void print_usage() {
    std::printf(
        "Usage:\n"
        "  brosoundml_qwen_asr_transcribe <wav> <model_dir>\n"
        "                                 [--device cpu|cuda] [--context TEXT]\n"
        "                                 [--max-new-tokens N] [--stream] [--ids]\n");
}

constexpr std::int32_t kAsrTextId = 151704;   // <asr_text>

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 2;
    }
    const std::string wav_path  = argv[1];
    const std::string model_dir = argv[2];

    std::string device_name = "auto";
    std::string context;
    int  max_new_tokens = 0;
    bool stream = false;
    bool print_ids = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string(what) + " needs a value");
            return argv[++i];
        };
        if      (a == "--device")         device_name = next("--device");
        else if (a == "--context")        context = next("--context");
        else if (a == "--max-new-tokens") max_new_tokens = std::atoi(next("--max-new-tokens").c_str());
        else if (a == "--stream")         stream = true;
        else if (a == "--ids")            print_ids = true;
        else if (a == "-h" || a == "--help") { print_usage(); return 0; }
        else die("unknown argument '" + a + "' (try --help)");
    }

    try {
        brotensor::init();
        brotensor::Device dev = brotensor::Device::CPU;
        if (device_name == "cuda") {
            dev = brotensor::Device::CUDA;
        } else if (device_name == "auto") {
            if (brotensor::is_available(brotensor::Device::CUDA))
                dev = brotensor::Device::CUDA;
        } else if (device_name != "cpu") {
            die("unknown --device '" + device_name + "' (want cpu or cuda)");
        }

        const brosoundml::AudioBuffer audio = brosoundml::read_wav(wav_path);
        std::fprintf(stderr, "audio: %.2fs @ %d Hz\n",
                     audio.duration_seconds(), audio.sample_rate);

        auto tok = brolm::qwen::Tokenizer::load(
            (fs::path(model_dir) / "vocab.json").string(),
            (fs::path(model_dir) / "merges.txt").string());

        brosoundml::QwenAsr asr;
        const auto t0 = std::chrono::steady_clock::now();
        asr.load(model_dir, dev);
        const auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "loaded in %.1fs (%s)\n",
                     std::chrono::duration<double>(t1 - t0).count(),
                     dev == brotensor::Device::CUDA ? "cuda" : "cpu");

        brosoundml::QwenAsr::TranscribeOptions opts;
        opts.max_new_tokens = max_new_tokens;
        if (!context.empty()) opts.context_ids = tok.encode(context);
        if (stream) {
            opts.on_token = [&tok](std::int32_t id) {
                std::printf("%s", tok.decode({id}).c_str());
                std::fflush(stdout);
            };
        }

        const auto res = asr.transcribe(audio, opts);
        const auto t2 = std::chrono::steady_clock::now();
        if (stream) std::printf("\n");
        std::fprintf(stderr, "transcribed in %.2fs (%zu tokens)\n",
                     std::chrono::duration<double>(t2 - t1).count(),
                     res.token_ids.size());

        if (print_ids) {
            for (std::size_t i = 0; i < res.token_ids.size(); ++i)
                std::printf("%s%d", i ? " " : "", res.token_ids[i]);
            std::printf("\n");
        }

        // Split "language <Language><asr_text>transcript".
        std::vector<std::int32_t> head, tail;
        bool seen_asr_text = false;
        for (std::int32_t id : res.token_ids) {
            if (id == kAsrTextId) { seen_asr_text = true; continue; }
            (seen_asr_text ? tail : head).push_back(id);
        }
        if (seen_asr_text) {
            std::fprintf(stderr, "%s\n", tok.decode(head).c_str());
            if (!stream) std::printf("%s\n", tok.decode(tail).c_str());
        } else if (!stream) {
            std::printf("%s\n", tok.decode(res.token_ids).c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        die(e.what());
    }
}
