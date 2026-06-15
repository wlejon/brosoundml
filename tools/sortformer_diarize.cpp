// brosoundml::sortformer_diarize — command-line streaming-Sortformer diarizer.
//
// Reads a 16 kHz mono WAV, runs Sortformer::diarize end-to-end against a model
// directory (config.json + model.safetensors from scripts/convert-sortformer.py),
// and prints RTTM speaker segments. Errors go to stderr.
//
// Usage:
//   brosoundml_sortformer_diarize <wav> <model_dir>
//                                 [--device auto|cpu|cuda]
//                                 [--threshold T] [--uri NAME]
//                                 [--probs-out FILE]
//
// Notes:
//   * The WAV must be 16 kHz mono PCM (resample externally: ffmpeg -ar 16000
//     -ac 1 ...).
//   * --threshold (default 0.5): a speaker is "active" in a frame when its
//     probability exceeds T; contiguous active frames merge into one segment.
//   * --probs-out FILE dumps the raw (T x S) probability matrix as a little-
//     endian binary: int32 T, int32 S, then T*S float32 row-major. Used by the
//     parity harness.

#include "brosoundml/audio.h"
#include "brosoundml/sortformer.h"

#include <brotensor/runtime.h>

#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "brosoundml_sortformer_diarize: %s\n", msg.c_str());
    std::exit(2);
}

void print_usage() {
    std::printf(
        "Usage:\n"
        "  brosoundml_sortformer_diarize <wav> <model_dir>\n"
        "                                [--device auto|cpu|cuda]\n"
        "                                [--threshold T] [--uri NAME]\n"
        "                                [--probs-out FILE]\n");
}

void dump_probs(const std::string& path,
                const brosoundml::Sortformer::Diarization& d) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) die("cannot open --probs-out file '" + path + "'");
    std::int32_t T = d.num_frames, S = d.num_speakers;
    std::fwrite(&T, sizeof(T), 1, f);
    std::fwrite(&S, sizeof(S), 1, f);
    std::fwrite(d.probs.data(), sizeof(float), d.probs.size(), f);
    std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
    std::string device_arg = "auto";
    std::string uri        = "audio";
    std::string probs_out;
    float       threshold  = 0.5f;
    bool        streaming  = false;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string(what) + " needs a value");
            return argv[++i];
        };
        if      (a == "-h" || a == "--help") { print_usage(); return 0; }
        else if (a == "--device")    device_arg = next("--device");
        else if (a == "--threshold") threshold = std::atof(next("--threshold").c_str());
        else if (a == "--uri")       uri = next("--uri");
        else if (a == "--streaming") streaming = true;
        else if (a == "--probs-out") probs_out = next("--probs-out");
        else if (!a.empty() && a[0] == '-') die("unknown flag '" + a + "'");
        else                         positional.push_back(std::move(a));
    }
    if (positional.size() != 2) {
        print_usage();
        die("expected exactly two positional args: <wav> <model_dir>");
    }
    const std::string wav_path  = positional[0];
    const std::string model_dir = positional[1];

    try {
        brotensor::init();

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

        brosoundml::Sortformer model;
        model.load(model_dir, device);

        brosoundml::AudioBuffer audio = brosoundml::read_wav(wav_path);
        if (audio.sample_rate != 16000)
            die("WAV is " + std::to_string(audio.sample_rate) +
                " Hz; Sortformer requires 16 kHz mono PCM.");
        std::fprintf(stderr, "brosoundml_sortformer_diarize: %.2fs audio on %s\n",
                     audio.duration_seconds(),
                     device == brotensor::Device::CUDA ? "CUDA" : "CPU");

        brosoundml::Sortformer::Diarization d;
        const auto t0 = std::chrono::steady_clock::now();
        if (streaming) {
            auto session = model.make_session();
            d = model.feed(session, audio, /*is_last=*/true);
        } else {
            d = model.diarize(audio);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr,
                     "brosoundml_sortformer_diarize: diarized in %.1f ms "
                     "(RTF %.4f, %d frames)\n",
                     ms, (ms / 1000.0) / audio.duration_seconds(), d.num_frames);

        if (!probs_out.empty()) dump_probs(probs_out, d);

        // RTTM: merge contiguous active frames per speaker into segments.
        const int T = d.num_frames, S = d.num_speakers;
        const double fs = d.frame_seconds;
        for (int s = 0; s < S; ++s) {
            int run_start = -1;
            for (int t = 0; t <= T; ++t) {
                const bool active =
                    t < T && d.probs[static_cast<std::size_t>(t) * S + s] > threshold;
                if (active && run_start < 0) {
                    run_start = t;
                } else if (!active && run_start >= 0) {
                    const double start = run_start * fs;
                    const double dur   = (t - run_start) * fs;
                    std::printf("SPEAKER %s 1 %.2f %.2f <NA> <NA> speaker_%d <NA> <NA>\n",
                                uri.c_str(), start, dur, s);
                    run_start = -1;
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        die(std::string("error: ") + e.what());
    }
}
