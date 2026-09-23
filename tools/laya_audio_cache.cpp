// brosoundml_laya_audio_cache — speech-encoder latents of streaming windows.
//
// For each aligned utterance, samples window end times t_end uniformly in
// [0.4 s, duration + 0.8 s] (so windows hold speech onsets, mid-utterance
// speech and trailing silence) and encodes the last --window seconds of the
// stream at t_end (laya_audio::window_audio) with the Qwen3-ASR AuT encoder +
// projector: ~12.5 latents/s of 1024-d, the soft tokens Qwen3-ASR's own LLM
// reads. The latents go to a 'LAC1' cache as FP16; labels are derived later
// from the alignment, so one cache serves every question set.
//
// Usage:
//   brosoundml_laya_audio_cache --align A.tsv --out C.lac
//        [--model-dir weights/qwen-asr/0.6B] [--window 3.0]
//        [--per-utt 4] [--every 1] [--limit 0] [--seed 1]
//        [--fp32] [--check N]
// The encoder runs FP16 on the GPU unless --fp32; --check N compares the
// first N windows against an FP32 encoder.

#include "laya_audio_data.h"

#include "brosoundml/audio.h"
#include "brosoundml/qwen_asr.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "laya_audio_cache: %s\n", msg.c_str());
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    std::string align, out, model_dir = "weights/qwen-asr/0.6B";
    float window_s = 3.0f;
    int per_utt = 4, every = 1, limit = 0, check = 0;
    bool half = true;
    uint32_t seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--align") align = next();
        else if (a == "--out") out = next();
        else if (a == "--model-dir") model_dir = next();
        else if (a == "--window") window_s = std::stof(next());
        else if (a == "--per-utt") per_utt = std::atoi(next().c_str());
        else if (a == "--every") every = std::max(1, std::atoi(next().c_str()));
        else if (a == "--limit") limit = std::atoi(next().c_str());
        else if (a == "--seed") seed = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--fp32") half = false;
        else if (a == "--check") check = std::atoi(next().c_str());
        else die("unknown argument " + a);
    }
    if (align.empty() || out.empty()) die("need --align and --out");

    try {
        brotensor::init();
        const brotensor::Device dev =
            brotensor::is_available(brotensor::Device::CUDA) ? brotensor::Device::CUDA : brotensor::Device::CPU;
        const std::vector<laya_audio::AlignedUtterance> utts = laya_audio::read_alignments(align);
        brosoundml::QwenAsr asr;
        asr.load_encoder(model_dir, dev, half);
        const int dim = asr.config().latent_dim;
        // --check N: the first N windows also run through an FP32 encoder;
        // report how far the FP16 latents sit from it (per-row cosine, and
        // the error norm relative to the latent norm).
        brosoundml::QwenAsr ref;
        if (check > 0) ref.load_encoder(model_dir, dev, false);
        double worst_rel = 0, sum_rel = 0, worst_cos = 1;
        int n_checked = 0;
        std::vector<float> host_ref;
        laya_audio::CacheWriter writer(out, window_s, dim);

        std::mt19937 rng(seed);
        double enc_ms = 0;
        int n_enc = 0, kept = 0;
        std::vector<float> host;
        for (std::size_t u = 0; u < utts.size(); ++u) {
            if (u % static_cast<std::size_t>(every)) continue;
            if (limit > 0 && kept >= limit) break;
            ++kept;
            const std::vector<float> pcm = laya_audio::load_audio_16k(utts[u].wav);
            std::uniform_real_distribution<float> t_dist(0.4f, utts[u].duration_s + 0.8f);
            for (int k = 0; k < per_utt; ++k) {
                const float t_end = t_dist(rng);
                brosoundml::AudioBuffer win(laya_audio::window_audio(pcm, t_end, window_s, rng()), 16000);
                const auto t0 = std::chrono::steady_clock::now();
                const int frames = asr.encode_to_host(win, host);
                enc_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ++n_enc;
                if (n_checked < check) {
                    ref.encode_to_host(win, host_ref);
                    double en = 0, rn = 0;
                    for (int r = 0; r < frames; ++r) {
                        double dot = 0, a2 = 0, b2 = 0;
                        for (int c = 0; c < dim; ++c) {
                            const std::size_t i = static_cast<std::size_t>(r) * dim + c;
                            const double a = host[i], b = host_ref[i];
                            dot += a * b;
                            a2 += a * a;
                            b2 += b * b;
                            en += (a - b) * (a - b);
                        }
                        rn += b2;
                        worst_cos = std::min(worst_cos, dot / std::sqrt(a2 * b2 + 1e-30));
                    }
                    const double rel = std::sqrt(en / (rn + 1e-30));
                    worst_rel = std::max(worst_rel, rel);
                    sum_rel += rel;
                    if (++n_checked == check) {
                        std::fprintf(stderr, "check vs FP32 over %d windows: |d|/|x| mean %.4f worst %.4f, worst row cosine %.5f\n",
                                     n_checked, sum_rel / n_checked, worst_rel, worst_cos);
                    }
                }
                laya_audio::CachedWindow w;
                w.utt = static_cast<int>(u);
                w.t_end = t_end;
                w.frames = frames;
                w.lat.resize(host.size());
                for (std::size_t i = 0; i < host.size(); ++i) w.lat[i] = brotensor::fp32_to_fp16_bits(host[i]);
                writer.add(w);
            }
            if (kept % 500 == 0) {
                std::fprintf(stderr, "%d utterances, %d windows, encode %.2f ms/window\n", kept, writer.count(),
                             enc_ms / std::max(1, n_enc));
            }
        }
        std::fprintf(stderr, "done: %d windows from %d utterances, encode %.2f ms/window (%.1f s windows)\n",
                     writer.count(), kept, enc_ms / std::max(1, n_enc), window_s);
        return 0;
    } catch (const std::exception& e) {
        die(e.what());
    }
}
