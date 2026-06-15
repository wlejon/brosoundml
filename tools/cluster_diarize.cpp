// brosoundml::cluster_diarize — clustering speaker diarizer CLI.
//
// Reads a 16 kHz mono WAV and diarizes it with ClusterDiarizer (Sortformer VAD +
// ECAPA x-vectors + centered-cosine clustering), printing RTTM segments. Built to
// A/B against brosoundml_sortformer_diarize on the same clip — the clustering
// path resolves similar-voiced speakers the 4-slot Sortformer head collapses.
//
//   brosoundml_cluster_diarize <wav> <sortformer_dir> <speaker_encoder_dir>
//                              [--device auto|cpu|cuda]
//                              [--cluster-threshold T] [--vad T] [--uri NAME]

#include "brosoundml/audio.h"
#include "brosoundml/cluster_diarizer.h"

#include <brotensor/runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {
[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "brosoundml_cluster_diarize: %s\n", m.c_str());
    std::exit(2);
}
}  // namespace

int main(int argc, char** argv) {
    std::string device_arg = "auto", uri = "audio";
    brosoundml::ClusterDiarizer::Config cfg;

    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* w) -> std::string {
            if (i + 1 >= argc) die(std::string(w) + " needs a value");
            return argv[++i];
        };
        if      (a == "-h" || a == "--help") {
            std::printf("usage: brosoundml_cluster_diarize <wav> <sortformer_dir> "
                        "<speaker_encoder_dir> [--device auto|cpu|cuda] "
                        "[--cluster-threshold T] [--vad T] [--uri NAME]\n");
            return 0;
        }
        else if (a == "--device")            device_arg = next("--device");
        else if (a == "--cluster-threshold") cfg.cluster_threshold = std::atof(next("--cluster-threshold").c_str());
        else if (a == "--vad")               cfg.vad_threshold = std::atof(next("--vad").c_str());
        else if (a == "--window")            cfg.window_seconds = std::atof(next("--window").c_str());
        else if (a == "--hop")               cfg.hop_seconds = std::atof(next("--hop").c_str());
        else if (a == "--uri")               uri = next("--uri");
        else if (!a.empty() && a[0] == '-')  die("unknown flag '" + a + "'");
        else                                 pos.push_back(std::move(a));
    }
    if (pos.size() != 3)
        die("expected: <wav> <sortformer_dir> <speaker_encoder_dir>");

    try {
        brotensor::init();
        brotensor::Device device = brotensor::Device::CPU;
        if (device_arg == "cuda" || device_arg == "auto")
            device = brotensor::is_available(brotensor::Device::CUDA)
                         ? brotensor::Device::CUDA : brotensor::Device::CPU;
        else if (device_arg != "cpu") die("--device must be auto, cpu, or cuda");

        brosoundml::ClusterDiarizer diar;
        diar.load(pos[1], pos[2], device);

        brosoundml::AudioBuffer audio = brosoundml::read_wav(pos[0]);
        if (audio.sample_rate != 16000)
            die("WAV is " + std::to_string(audio.sample_rate) +
                " Hz; need 16 kHz mono PCM.");

        const auto t0 = std::chrono::steady_clock::now();
        brosoundml::ClusterDiarizer::Diarization d = diar.diarize(audio, cfg);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr,
            "cluster_diarize: %.2fs audio, %d speaker(s), %.0f ms (cluster-thr %.2f)\n",
            audio.duration_seconds(), d.num_speakers, ms, cfg.cluster_threshold);

        const int T = d.num_frames, S = d.num_speakers;
        const double fps = d.frame_seconds;
        for (int s = 0; s < S; ++s) {
            int run = -1;
            for (int t = 0; t <= T; ++t) {
                const bool on = t < T && d.probs[static_cast<std::size_t>(t) * S + s] > 0.5f;
                if (on && run < 0) run = t;
                else if (!on && run >= 0) {
                    std::printf("SPEAKER %s 1 %.2f %.2f <NA> <NA> speaker_%d <NA> <NA>\n",
                                uri.c_str(), run * fps, (t - run) * fps, s);
                    run = -1;
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        die(std::string("error: ") + e.what());
    }
}
