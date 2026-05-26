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
//
// Use scripts/phonemize.py to convert English text -> id sequence.

#include "brosoundml/kokoro.h"

#include <brotensor/tensor.h>
#include <brotensor/runtime.h>

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

        if (!ids_inline.empty()) {
            if (out_path.empty()) die("--out is required when phoneme ids are inline");
            const auto ids = parse_ids(ids_inline);
            std::fprintf(stderr, "synthesizing %zu phonemes -> %s\n",
                         ids.size(), out_path.c_str());
            auto audio = k.synthesize(ids, voice, speed);
            audio.write_wav(out_path);
            std::fprintf(stderr, "  wrote %zu samples\n", audio.samples.size());
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
