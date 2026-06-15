// Quick probe: does the ECAPA-TDNN speaker encoder place distinct voices in
// separable x-vector regions? Loads the standalone speaker encoder, embeds each
// WAV on the command line, L2-normalizes, and prints the pairwise cosine matrix.
//
//   brosoundml_speaker_embed_probe <enc_dir> <a.wav> <b.wav> [c.wav ...]
//
// Used to validate clustering-based diarization: if same-voice pairs score high
// and different-voice pairs score clearly lower, cosine clustering will separate
// speakers Sortformer's 4-slot head collapses.

#include "brosoundml/audio.h"
#include "brosoundml/speaker_encoder.h"

#include <brotensor/runtime.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string base_name(const std::string& p) {
    return fs::path(p).parent_path().filename().string() + "/" +
           fs::path(p).filename().string();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <enc_dir> [--mean-out FILE] <a.wav> <b.wav> [c.wav ...]\n",
            argv[0]);
        return 2;
    }
    brotensor::init();

    const std::string enc_dir = argv[1];
    std::string mean_out;
    std::vector<std::string> paths;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mean-out" && i + 1 < argc) { mean_out = argv[++i]; continue; }
        paths.emplace_back(std::move(a));
    }

    brosoundml::SpeakerEncoder enc;
    enc.load(enc_dir);
    std::fprintf(stderr, "loaded speaker encoder (dim %d, %d Hz)\n",
                 enc.enc_dim(), enc.sample_rate());

    const int D = enc.enc_dim();
    std::vector<std::vector<float>> xs;     // L2-normalized (for the cosine views)
    std::vector<double> raw_sum(static_cast<std::size_t>(D), 0.0);  // for the mean
    for (const std::string& p : paths) {
        brosoundml::AudioBuffer a = brosoundml::read_wav(p);
        std::vector<float> x = enc.embed(a);
        for (int t = 0; t < D; ++t) raw_sum[static_cast<std::size_t>(t)] += x[t];
        // L2 normalize
        double n = 0.0;
        for (float v : x) n += static_cast<double>(v) * v;
        n = std::sqrt(std::max(1e-12, n));
        for (float& v : x) v = static_cast<float>(v / n);
        xs.push_back(std::move(x));
        std::fprintf(stderr, "  embedded %s\n", base_name(p).c_str());
    }

    // Population mean of the RAW x-vectors (what ClusterDiarizer centers against).
    if (!mean_out.empty()) {
        std::vector<float> mean(static_cast<std::size_t>(D));
        for (int t = 0; t < D; ++t)
            mean[static_cast<std::size_t>(t)] =
                static_cast<float>(raw_sum[static_cast<std::size_t>(t)] / paths.size());
        std::FILE* f = std::fopen(mean_out.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", mean_out.c_str()); return 1; }
        std::fwrite(mean.data(), sizeof(float), mean.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "wrote population mean (%d floats, %zu clips) -> %s\n",
                     D, paths.size(), mean_out.c_str());
    }

    auto print_matrix = [&](const char* title,
                            const std::vector<std::vector<float>>& v) {
        auto cos = [&](int i, int k) {
            double d = 0.0;
            for (int t = 0; t < D; ++t) d += static_cast<double>(v[i][t]) * v[k][t];
            return d;
        };
        std::printf("\n%s:\n      ", title);
        for (size_t k = 0; k < paths.size(); ++k) std::printf(" %5zu", k);
        std::printf("\n");
        for (size_t i = 0; i < paths.size(); ++i) {
            std::printf("%2zu %-18s", i, base_name(paths[i]).substr(0, 18).c_str());
            for (size_t k = 0; k < paths.size(); ++k)
                std::printf(" %5.2f", cos((int)i, (int)k));
            std::printf("\n");
        }
    };

    print_matrix("raw cosine similarity", xs);

    // Mean-center then re-normalize: x-vectors live in a narrow cone, so cosine
    // is only speaker-discriminative after subtracting the global mean (whitening
    // the shared component). Center against the mean of this batch.
    std::vector<float> mean(D, 0.0f);
    for (const auto& x : xs) for (int t = 0; t < D; ++t) mean[t] += x[t];
    for (int t = 0; t < D; ++t) mean[t] /= static_cast<float>(xs.size());
    std::vector<std::vector<float>> cen = xs;
    for (auto& x : cen) {
        for (int t = 0; t < D; ++t) x[t] -= mean[t];
        double n = 0.0;
        for (float v : x) n += static_cast<double>(v) * v;
        n = std::sqrt(std::max(1e-12, n));
        for (float& v : x) v = static_cast<float>(v / n);
    }
    print_matrix("mean-centered cosine similarity", cen);
    return 0;
}
