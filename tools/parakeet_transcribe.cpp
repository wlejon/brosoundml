// brosoundml::parakeet_transcribe — command-line Parakeet (FastConformer-TDT)
// driver.
//
// Reads a 16 kHz mono WAV, runs Parakeet::transcribe end-to-end against a model
// directory + the matching SentencePiece tokenizer (tokenizer.json), and prints
// the decoded transcript. Errors go to stderr.
//
// Usage:
//   brosoundml_parakeet_transcribe <wav> <model_dir>
//                                  [--device auto|cpu|cuda]
//                                  [--max-new-tokens N] [--stream]
//                                  [--timestamps]
//
// Notes:
//   * The WAV must be 16 kHz mono PCM — Parakeet's input rate is fixed;
//     resample externally (ffmpeg -ar 16000 -ac 1 ...).
//   * <model_dir> holds config.json, model.safetensors and tokenizer.json.
//   * --device defaults to CUDA when a GPU backend is present, else CPU.
//   * --timestamps prints one "start\tpiece" line per token (start = encoder
//     frame index * 0.08 s) instead of the single transcript line.

#include "brosoundml/audio.h"
#include "brosoundml/parakeet.h"

#include <brolm/tokenizer_t5.h>

#include <brotensor/runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "brosoundml_parakeet_transcribe: %s\n", msg.c_str());
    std::exit(2);
}

void print_usage() {
    std::printf(
        "Usage:\n"
        "  brosoundml_parakeet_transcribe <wav> <model_dir>\n"
        "                                 [--device auto|cpu|cuda]\n"
        "                                 [--max-new-tokens N] [--stream]\n"
        "                                 [--timestamps]\n"
        "\n"
        "  <wav>        16 kHz mono PCM WAV (resample externally if needed).\n"
        "  <model_dir>  Directory with config.json, model.safetensors,\n"
        "               tokenizer.json.\n"
        "\n"
        "Options:\n"
        "  --device D          auto (default), cpu, or cuda.\n"
        "  --max-new-tokens N  Cap emitted tokens (0 = whole clip).\n"
        "  --stream            Print the transcript incrementally as it decodes.\n"
        "  --timestamps        Print per-token start times instead of one line.\n"
        "  -h, --help          Show this help and exit.\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string device_arg = "auto";
    int         max_new    = 0;
    bool        stream     = false;
    bool        timestamps = false;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string(what) + " needs a value");
            return argv[++i];
        };
        if      (a == "-h" || a == "--help") { print_usage(); return 0; }
        else if (a == "--device")           device_arg = next("--device");
        else if (a == "--max-new-tokens")   max_new = std::atoi(next("--max-new-tokens").c_str());
        else if (a == "--stream")           stream = true;
        else if (a == "--timestamps")       timestamps = true;
        else if (!a.empty() && a[0] == '-')  die("unknown flag '" + a + "'");
        else                                positional.push_back(std::move(a));
    }
    if (positional.size() != 2) {
        print_usage();
        die("expected exactly two positional args: <wav> <model_dir>");
    }
    const std::string wav_path  = positional[0];
    const std::string model_dir = positional[1];

    try {
        brotensor::init();

        // Device selection: GPU-first, with explicit override.
        brotensor::Device device = brotensor::Device::CPU;
        if (device_arg == "cuda") {
            if (!brotensor::is_available(brotensor::Device::CUDA))
                die("--device cuda requested but no CUDA backend is available");
            device = brotensor::Device::CUDA;
        } else if (device_arg == "cpu") {
            device = brotensor::Device::CPU;
        } else if (device_arg == "auto") {
            device = brotensor::is_available(brotensor::Device::CUDA)
                         ? brotensor::Device::CUDA
                         : brotensor::Device::CPU;
        } else {
            die("--device must be auto, cpu, or cuda");
        }

        // 1. Load model.
        brosoundml::Parakeet model;
        model.load(model_dir, device);

        // 2. Load tokenizer (tokenizer.json under the model dir).
        namespace fs = std::filesystem;
        const std::string tok_path = (fs::path(model_dir) / "tokenizer.json").string();
        auto tok = brolm::t5::Tokenizer::load(tok_path);

        // 3. Read WAV.
        brosoundml::AudioBuffer audio = brosoundml::read_wav(wav_path);
        if (audio.sample_rate != 16000) {
            die("WAV is " + std::to_string(audio.sample_rate) +
                " Hz; Parakeet requires 16 kHz mono PCM. "
                "Resample externally (ffmpeg -ar 16000 -ac 1 ...) and retry.");
        }
        std::fprintf(stderr,
                     "brosoundml_parakeet_transcribe: %.2fs audio on %s\n",
                     audio.duration_seconds(),
                     device == brotensor::Device::CUDA ? "CUDA" : "CPU");

        brosoundml::Parakeet::TranscribeOptions opts;
        opts.max_new_tokens = max_new;

        // --stream: re-decode the running id list each token and print the
        // newly revealed suffix so the transcript grows live on stdout.
        std::vector<int32_t> generated;
        std::size_t printed = 0;
        if (stream && !timestamps) {
            opts.on_token = [&](int32_t id) {
                generated.push_back(id);
                std::string text = tok.decode(generated);
                if (text.size() > printed) {
                    std::fwrite(text.data() + printed, 1, text.size() - printed,
                                stdout);
                    std::fflush(stdout);
                    printed = text.size();
                }
            };
        }

        // 4. Run.
        auto result = model.transcribe(audio, opts);

        if (timestamps) {
            const double fs = model.config().frame_seconds();
            for (std::size_t i = 0; i < result.token_ids.size(); ++i) {
                std::string piece = tok.decode({result.token_ids[i]});
                std::printf("%7.2f\t%s\n",
                            result.token_frames[i] * fs, piece.c_str());
            }
        } else if (stream) {
            std::printf("\n");
        } else {
            std::printf("%s\n", tok.decode(result.token_ids).c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        die(std::string("error: ") + e.what());
    }
}
