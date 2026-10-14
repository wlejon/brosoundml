// brosoundml_separate — HTDemucs (htdemucs_ft) source separation from the
// command line: planar float32 stereo in, one planar float32 file per stem out.
//
// Usage:
//   brosoundml_separate <model_dir> <in.f32> <out_dir>
//                       [--rate 48000] [--stems vocals,drums] [--batch N]
//                       [--offset S] [--seconds S] [--repeat N]
//                       [--device auto|cpu|gpu|cuda|vulkan|vk|metal]
//
//   <in.f32>   planar float32, channel 0 then channel 1 (yue2-research's
//              audio.f32), at --rate (default 48000).
//   --offset / --seconds cut a window of the input first.
//   --repeat N runs the separation N times and reports each run's wall time
//              (the first includes warm-up), in ms and ms per audio second.
// Writes <out_dir>/<stem>.f32 (planar, input rate and length).

#include "tool_device.h"

#include "brosoundml/htdemucs.h"

#include <brotensor/runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "brosoundml_separate: %s\n", msg.c_str());
    std::exit(2);
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) out.push_back(item);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: brosoundml_separate <model_dir> <in.f32> <out_dir> [--rate 48000] "
                    "[--stems a,b] [--batch N] [--offset S] [--seconds S] [--repeat N] [--device %s]\n",
                    brosoundml_tool::kDeviceChoices);
        return 2;
    }
    const std::string model_dir = argv[1], in_path = argv[2], out_dir = argv[3];
    int rate = 48000, batch = 0, repeat = 1;
    double offset = 0.0, seconds = 0.0;
    std::string device_arg = "auto";
    std::vector<std::string> stems;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + a);
            return argv[++i];
        };
        if (a == "--rate") rate = std::atoi(next().c_str());
        else if (a == "--stems") stems = split(next());
        else if (a == "--batch") batch = std::atoi(next().c_str());
        else if (a == "--offset") offset = std::atof(next().c_str());
        else if (a == "--seconds") seconds = std::atof(next().c_str());
        else if (a == "--repeat") repeat = std::atoi(next().c_str());
        else if (a == "--device") device_arg = next();
        else die("unknown option " + a);
    }

    try {
        brotensor::init();
        brotensor::Device dev;
        std::string err;
        if (!brosoundml_tool::resolve_device(device_arg, dev, err)) die(err);

        std::ifstream in(in_path, std::ios::binary | std::ios::ate);
        if (!in) die("cannot read " + in_path);
        const std::streamsize bytes = in.tellg();
        in.seekg(0);
        std::vector<float> raw(static_cast<std::size_t>(bytes) / sizeof(float));
        in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(float)));
        const std::size_t total = raw.size() / 2;
        const std::size_t s0 = std::min(total, static_cast<std::size_t>(offset * rate));
        std::size_t n = total - s0;
        if (seconds > 0) n = std::min(n, static_cast<std::size_t>(seconds * rate));
        if (n == 0) die("empty input window");
        brosoundml::StereoAudio audio;
        audio.channels = 2;
        audio.frames = static_cast<int>(n);
        audio.sample_rate = rate;
        audio.samples.resize(2 * n);
        for (int c = 0; c < 2; ++c)
            std::copy(raw.begin() + c * total + s0, raw.begin() + c * total + s0 + n,
                      audio.samples.begin() + c * n);

        brosoundml::HtDemucs model;
        auto t0 = std::chrono::steady_clock::now();
        model.load(model_dir, dev, stems);
        auto t1 = std::chrono::steady_clock::now();
        std::printf("loaded %s on %s in %.0f ms\n", model_dir.c_str(), brosoundml_tool::device_name(dev),
                    std::chrono::duration<double, std::milli>(t1 - t0).count());

        brosoundml::SeparateOptions opts;
        opts.stems = stems;
        opts.max_batch = batch;
        brosoundml::Separation sep;
        const double audio_s = double(n) / rate;
        for (int r = 0; r < repeat; ++r) {
            brotensor::sync_all();
            auto a = std::chrono::steady_clock::now();
            sep = model.separate(audio, opts);
            auto b = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(b - a).count();
            std::printf("run %d: %.1f ms for %.2f s of audio (%.1f ms per audio second)\n", r, ms, audio_s,
                        ms / audio_s);
        }
        std::filesystem::create_directories(out_dir);
        for (std::size_t i = 0; i < sep.names.size(); ++i) {
            const std::string p = (std::filesystem::path(out_dir) / (sep.names[i] + ".f32")).string();
            std::ofstream out(p, std::ios::binary);
            out.write(reinterpret_cast<const char*>(sep.stems[i].samples.data()),
                      static_cast<std::streamsize>(sep.stems[i].samples.size() * sizeof(float)));
            std::printf("wrote %s\n", p.c_str());
        }
    } catch (const std::exception& e) {
        die(e.what());
    }
    return 0;
}
