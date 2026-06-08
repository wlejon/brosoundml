// brosoundml::synth — command-line Kokoro driver.
//
// Reads a comma-separated phoneme-id sequence (or several, one per line) from
// stdin or a file, runs Kokoro::synthesize for each one against a chosen
// voice, and writes the resulting 24 kHz WAV(s) next to the input or to a
// caller-specified path.
//
// Usage:
//   synth --model <model_dir> --voice <voice.bin> --out <out.wav> "1,2,3,..."
//   synth --model <model_dir> --voice <voice.bin> --out-dir <dir> \
//         --ids-file lines.txt --speed 1.0
//   synth --model <model_dir> --voice <voice.bin> --out <out.wav> \
//         --ids-file lines.txt --stream      # one WAV, each line a chunk
//
// With --stream, the --ids-file lines are treated as phoneme chunks
// (one sentence/clause per line) and fed to Kokoro::synthesize_stream: each
// chunk's audio is emitted as it finishes (first-chunk latency) and the
// concatenation is written to --out.
//
// Use scripts/phonemize.py to convert English text -> id sequence.

#include "brosoundml/kokoro.h"

#include <brotensor/tensor.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "synth: %s\n", msg.c_str());
    std::exit(2);
}

std::vector<int32_t> parse_ids(const std::string& line) {
    std::vector<int32_t> out;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // Trim whitespace.
        size_t a = tok.find_first_not_of(" \t\r\n");
        size_t b = tok.find_last_not_of (" \t\r\n");
        if (a == std::string::npos) continue;
        tok = tok.substr(a, b - a + 1);
        if (tok.empty()) continue;
        try {
            out.push_back(static_cast<int32_t>(std::stoi(tok)));
        } catch (...) {
            die("invalid phoneme id '" + tok + "' in line");
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir, voice_path, out_path, out_dir, ids_file, ids_inline, device_str = "cpu";
    float speed = 1.0f;
    bool trace = false;
    bool stream = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string(what) + " needs a value");
            return argv[++i];
        };
        if      (a == "--model")     model_dir  = next("--model");
        else if (a == "--voice")     voice_path = next("--voice");
        else if (a == "--out")       out_path   = next("--out");
        else if (a == "--out-dir")   out_dir    = next("--out-dir");
        else if (a == "--ids-file")  ids_file   = next("--ids-file");
        else if (a == "--speed")     speed      = std::stof(next("--speed"));
        else if (a == "--device")    device_str = next("--device");
        else if (a == "--trace")     trace      = true;
        else if (a == "--stream")    stream     = true;
        else if (a == "-h" || a == "--help") {
            std::printf("Usage: synth --model DIR --voice voice.bin "
                        "(--out file.wav PHONEMEIDS | --out-dir DIR --ids-file FILE)\n");
            return 0;
        }
        else if (!a.empty() && a[0] != '-') ids_inline = a;
        else die("unknown flag '" + a + "'");
    }

    if (model_dir.empty())  die("--model is required");
    if (voice_path.empty()) die("--voice is required");

    try {
        brotensor::init();
        brotensor::Device dev = brotensor::Device::CPU;
        if      (device_str == "cpu")   dev = brotensor::Device::CPU;
        else if (device_str == "cuda")  dev = brotensor::Device::CUDA;
        else if (device_str == "metal") dev = brotensor::Device::Metal;
        else die("--device must be one of cpu|cuda|metal");
        if (dev != brotensor::Device::CPU && !brotensor::is_available(dev))
            die("--device " + device_str + " not available in this build");

        brosoundml::Kokoro k;
        k.load(model_dir, dev);
        std::fprintf(stderr, "device: %s\n", device_str.c_str());
        brosoundml::Voice voice = k.load_voice(voice_path);
        std::fprintf(stderr, "loaded voice %s (%dx%d)\n",
                     voice.name.c_str(), voice.packs.rows, voice.packs.cols);

        // Streaming: --ids-file lines are phoneme chunks, emitted as each one
        // finishes and concatenated into a single --out WAV.
        if (stream) {
            if (ids_file.empty()) die("--stream requires --ids-file (one chunk per line)");
            if (out_path.empty()) die("--stream requires --out (the concatenated WAV)");
            std::ifstream f(ids_file);
            if (!f) die("cannot open --ids-file '" + ids_file + "'");
            std::vector<std::vector<int32_t>> chunks;
            std::string line;
            while (std::getline(f, line)) {
                if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
                chunks.push_back(parse_ids(line));
            }
            std::fprintf(stderr, "streaming %zu chunk(s) -> %s\n",
                         chunks.size(), out_path.c_str());
            int idx = 0;
            auto audio = k.synthesize_stream(
                chunks, voice,
                [&](const float*, int n) {
                    std::fprintf(stderr, "  chunk %d: %d samples (%.2fs)\n",
                                 idx++, n, n / 24000.0);
                },
                speed);
            audio.write_wav(out_path);
            std::fprintf(stderr, "  wrote %zu samples total\n",
                         audio.samples.size());
            return 0;
        }

        if (!ids_inline.empty()) {
            if (out_path.empty()) die("--out is required when phoneme ids are inline");
            const auto ids = parse_ids(ids_inline);
            std::fprintf(stderr, "synthesizing %zu phonemes -> %s\n",
                         ids.size(), out_path.c_str());
            brosoundml::KokoroTrace tr;
            auto audio = k.synthesize(ids, voice, speed, nullptr, {},
                                      trace ? &tr : nullptr);
            audio.write_wav(out_path);
            std::fprintf(stderr, "  wrote %zu samples\n", audio.samples.size());
            if (trace) {
                std::fprintf(stderr, "  trace: %zu stages\n", tr.stages.size());
                for (const auto& s : tr.stages) {
                    double mn = 0, mx = 0, sum = 0;
                    if (!s.data.empty()) {
                        mn = mx = s.data[0];
                        for (float v : s.data) { mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); sum += v; }
                    }
                    std::fprintf(stderr, "    %-10s  %4d x %-7d  (%zu)  min=%.3f max=%.3f mean=%.3f\n",
                                 s.name.c_str(), s.h, s.w, s.data.size(),
                                 mn, mx, s.data.empty() ? 0.0 : sum / s.data.size());
                }
            }
            return 0;
        }

        if (ids_file.empty()) die("provide either inline ids or --ids-file");
        if (out_dir.empty())  die("--out-dir is required when --ids-file is set");

        std::ifstream f(ids_file);
        if (!f) die("cannot open --ids-file '" + ids_file + "'");
        std::string line;
        int idx = 0;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const auto ids = parse_ids(line);
            const std::string out = out_dir + "/synth_" + std::to_string(idx) + ".wav";
            std::fprintf(stderr, "[%d] %zu phonemes -> %s\n",
                         idx, ids.size(), out.c_str());
            auto audio = k.synthesize(ids, voice, speed);
            audio.write_wav(out);
            std::fprintf(stderr, "    %zu samples\n", audio.samples.size());
            ++idx;
        }
        return 0;
    } catch (const std::exception& e) {
        die(std::string("error: ") + e.what());
    }
}
