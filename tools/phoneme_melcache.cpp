#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

// brosoundml_phoneme_melcache — precompute the training front-end for a BPDS
// shard into a BPMC mel-feature cache.
//
// phoneme_train recomputes the PCEN mel of every clip at startup; for a
// corpus-sized shard that pass dominates wall-clock and repeats on every run.
// This tool runs it ONCE and serializes exactly what the trainer builds in
// memory (freq-major (n_mels, n_frames) PCEN mel + the label track), so
// `phoneme_train --dataset shard.bpmc` loads with a single sequential read.
//
//   --dataset PATH[,...]  input BPDS shard(s); class maps + framing must match
//   --out PATH            output BPMC (default: first input with .bpmc ext)

#include "brosoundml/mel.h"
#include "brosoundml/phoneme_data.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace bt  = brotensor;
namespace bsm = brosoundml;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml_phoneme_melcache: " + msg);
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out; std::string cur;
    for (char ch : s) { if (ch==',') { if(!cur.empty()) out.push_back(cur); cur.clear(); }
                        else cur.push_back(ch); }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

}  // namespace

int main(int argc, char** argv) try {
    std::string dataset, out;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) fail("missing value for " + k); return argv[++i]; };
        if      (k == "--dataset") dataset = need();
        else if (k == "--out")     out = need();
        else if (k == "-h" || k == "--help") {
            std::printf(
                "brosoundml_phoneme_melcache — precompute PCEN mels: BPDS -> BPMC\n\n"
                "  --dataset PATH[,...]  input BPDS shard(s), comma-separated\n"
                "  --out PATH            output BPMC (default: first input, .bpmc)\n");
            return 0;
        }
        else if (!k.empty() && k[0] != '-' && dataset.empty()) dataset = k;
        else fail("unknown arg: " + k);
    }
    if (dataset.empty()) fail("--dataset is required");
    const std::vector<std::string> paths = split_csv(dataset);
    if (out.empty()) {
        out = paths[0];
        const auto dot = out.find_last_of('.');
        out = (dot == std::string::npos ? out : out.substr(0, dot)) + ".bpmc";
    }

    bt::init();

    auto ds = bsm::read_phoneme_dataset(paths[0]);
    for (std::size_t i = 1; i < paths.size(); ++i) {
        auto extra = bsm::read_phoneme_dataset(paths[i]);
        if (!(extra.class_map == ds.class_map))
            fail("class map mismatch: " + paths[i]);
        const auto& h = ds.header; const auto& e = extra.header;
        if (e.sample_rate != h.sample_rate || e.n_fft != h.n_fft ||
            e.win_length != h.win_length || e.hop_length != h.hop_length ||
            e.n_mels != h.n_mels)
            fail("framing mismatch: " + paths[i]);
        for (auto& c : extra.clips) ds.clips.push_back(std::move(c));
    }
    const int n_mels = ds.header.n_mels;
    std::fprintf(stderr, "phoneme_melcache: %zu clips -> %s\n",
                 ds.clips.size(), out.c_str());

    bsm::MelConfig mcfg;
    mcfg.sample_rate = ds.header.sample_rate;
    mcfg.n_fft       = ds.header.n_fft;
    mcfg.win_length  = ds.header.win_length;
    mcfg.hop_length  = ds.header.hop_length;
    mcfg.n_mels      = ds.header.n_mels;
    mcfg.compression = bsm::MelCompression::PCEN;
    bsm::MelFrontend mel(mcfg, bt::Device::CPU);

    bsm::PhonemeMelCacheWriter writer(
        out, ds.header, static_cast<std::uint32_t>(mcfg.compression),
        ds.class_map);

    const auto t0 = std::chrono::steady_clock::now();
    std::size_t total_frames = 0;
    for (std::size_t ci = 0; ci < ds.clips.size(); ++ci) {
        const auto& clip = ds.clips[ci];
        const auto pcm = clip.pcm_float();
        bt::Tensor m;                              // (n_mels, T_mel)
        mel.reset();
        mel.compute_offline(pcm.data(), static_cast<int>(pcm.size()), m);
        const int Tmel = m.cols;
        std::vector<float> mh(static_cast<std::size_t>(n_mels) * Tmel);
        std::memcpy(mh.data(), m.host_f32(), mh.size() * sizeof(float));
        // BPDS labels are frame-aligned to this framing; clamp defensively to
        // the mel frame count exactly like the trainer does.
        std::vector<std::int16_t> labels(static_cast<std::size_t>(Tmel), 0);
        const int nlab = static_cast<int>(clip.labels.size());
        for (int t = 0; t < Tmel && t < nlab; ++t)
            labels[static_cast<std::size_t>(t)] = clip.labels[static_cast<std::size_t>(t)];
        writer.append(mh, labels);
        total_frames += static_cast<std::size_t>(Tmel);
        if ((ci + 1) % 5000 == 0)
            std::fprintf(stderr, "  %zu / %zu clips ...\n", ci + 1, ds.clips.size());
    }
    writer.finalize();

    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "phoneme_melcache done: %d clips, %zu frames in %lld ms -> %s\n",
                 writer.clips(), total_frames,
                 static_cast<long long>(
                     std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()),
                 out.c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "phoneme_melcache: %s\n", e.what());
    return 1;
}
