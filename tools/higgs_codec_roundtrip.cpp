// brosoundml_higgs_codec_roundtrip — HiggsAudio v2 codec encode -> decode CLI.
//
// Usage:
//   brosoundml_higgs_codec_roundtrip <audio_tokenizer_dir> <in.wav> <out.wav>
//                                    [--device cpu|cuda] [--levels N]
//
// Reads a 16-bit PCM WAV (any rate; resampled to 24 kHz mono inside encode()),
// encodes it to 25 Hz RVQ codes, prints the frame count and a codebook-0
// snippet, decodes the codes back (all 8 levels, or the first --levels N for a
// coarser reconstruction) and writes the 24 kHz result.
#include "brosoundml/audio.h"
#include "brosoundml/higgs_codec.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "brosoundml_higgs_codec_roundtrip: %s\n", msg.c_str());
    std::exit(2);
}

void print_usage() {
    std::printf(
        "Usage:\n"
        "  brosoundml_higgs_codec_roundtrip <audio_tokenizer_dir> <in.wav> <out.wav>\n"
        "                                   [--device cpu|cuda] [--levels N]\n"
        "\n"
        "  <audio_tokenizer_dir>  config.json + model.safetensors (OmniVoice audio_tokenizer/)\n"
        "  <in.wav>               16-bit PCM WAV, any rate (resampled to 24 kHz mono)\n"
        "  <out.wav>              24 kHz mono reconstruction\n"
        "  --device D             cpu (default) or cuda\n"
        "  --levels N             decode with the first N of 8 RVQ levels (default 8)\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string device = "cpu";
    int levels = 0;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string(what) + " needs a value");
            return argv[++i];
        };
        if      (a == "-h" || a == "--help") { print_usage(); return 0; }
        else if (a == "--device") device = next("--device");
        else if (a == "--levels") levels = std::atoi(next("--levels").c_str());
        else if (!a.empty() && a[0] == '-') die("unknown flag '" + a + "'");
        else positional.push_back(a);
    }
    if (positional.size() != 3) {
        print_usage();
        die("expected <audio_tokenizer_dir> <in.wav> <out.wav>");
    }

    brotensor::Device dev = brotensor::Device::CPU;
    if (device == "cuda")     dev = brotensor::Device::CUDA;
    else if (device != "cpu") die("--device must be cpu or cuda");

    try {
        brotensor::init();
        if (!brotensor::is_available(dev)) die("device '" + device + "' is not available in this build");

        using clock = std::chrono::steady_clock;
        brosoundml::HiggsCodec codec;
        auto t0 = clock::now();
        codec.load(positional[0], dev);
        const auto& cfg = codec.config();
        std::fprintf(stderr, "loaded %s on %s in %.2fs (hop %d, %d Hz frames, %d x %d codes)\n",
                     positional[0].c_str(), device.c_str(),
                     std::chrono::duration<double>(clock::now() - t0).count(),
                     cfg.hop_length, cfg.frame_rate, cfg.num_quantizers, cfg.codebook_size);
        if (levels <= 0 || levels > cfg.num_quantizers) levels = cfg.num_quantizers;

        brosoundml::AudioBuffer in = brosoundml::read_wav(positional[1]);
        std::fprintf(stderr, "input: %zu samples @ %d Hz (%.2fs, peak %.3f)\n",
                     in.samples.size(), in.sample_rate, in.duration_seconds(), in.peak());

        t0 = clock::now();
        int T = 0;
        std::vector<int32_t> codes = codec.encode(in, &T);
        const double enc_s = std::chrono::duration<double>(clock::now() - t0).count();
        std::printf("T=%d frames (%.2fs of audio), encode %.2fs\n", T,
                    static_cast<double>(T) * cfg.hop_length / cfg.sample_rate, enc_s);
        std::printf("codebook 0:");
        for (int t = 0; t < std::min(T, 16); ++t) std::printf(" %d", codes[t]);
        if (T > 16) std::printf(" ...");
        std::printf("\n");

        t0 = clock::now();
        brosoundml::AudioBuffer out = codec.decode(codes.data(), levels, T);
        const double dec_s = std::chrono::duration<double>(clock::now() - t0).count();
        std::printf("decoded %zu samples @ %d Hz with %d level(s) in %.2fs (peak %.3f)\n",
                    out.samples.size(), out.sample_rate, levels, dec_s, out.peak());
        out.write_wav(positional[2]);
        std::printf("wrote %s\n", positional[2].c_str());
        return 0;
    } catch (const std::exception& e) {
        die(std::string("error: ") + e.what());
    }
}
