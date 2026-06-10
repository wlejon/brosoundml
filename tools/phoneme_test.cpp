#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

// brosoundml_phoneme_test — frame-level phoneme-posterior eval CLI.
//
// Loads a 'BPM1' PhonemeNet checkpoint (class map restored from the file),
// runs every eval clip's PCEN-mel through forward() to (T,K) logits, takes the
// per-frame argmax and scores it against the BPDS frame labels. Reports overall
// frame accuracy, non-silence frame accuracy (true label != silence), an
// optional per-class recall/precision table, and a K×K confusion matrix
// (compact summary to stdout, full matrix to --confusion).
//
// This is the FRAME-LEVEL measurement tool only — the streaming latency / false-
// accept sweep is a later chunk. It exits 0 unless a load/IO error occurs.

#include "brosoundml/mel.h"
#include "brosoundml/phoneme_data.h"
#include "brosoundml/phoneme_model.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs  = std::filesystem;
namespace bt  = brotensor;
namespace bsm = brosoundml;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

struct Args {
    std::string weights;
    std::string dataset;
    std::string device       = "auto";
    std::string confusion;                  // optional full-matrix dump
    float       val_frac     = 0.1f;
    int         seed         = 42;
    int         train_frames = 100;
    bool        whole_clip   = false;
    bool        all          = false;       // eval the whole set, not just val
    bool        per_class    = false;
    bool        help         = false;
};

void print_help() {
    std::printf(
        "brosoundml_phoneme_test — frame-level phoneme eval\n"
        "  --weights PATH      .bpm checkpoint (required)\n"
        "  --dataset PATH      BPDS dataset (required)\n"
        "  --val-frac F        held-out split to reproduce (default 0.1)\n"
        "  --seed N            split seed (default 42)\n"
        "  --all               eval the WHOLE set, not just the held-out split\n"
        "  --train-frames T    centered eval window length (default 100)\n"
        "  --whole-clip        run each clip end-to-end instead of a window\n"
        "  --per-class         print a per-class recall/precision table\n"
        "  --confusion PATH    dump the full K x K confusion matrix to PATH\n"
        "  --device cpu|cuda   target device (default auto)\n");
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) fail("cli", "missing value for " + k);
            return std::string(argv[++i]);
        };
        if      (k == "--weights")      a.weights = next();
        else if (k == "--dataset")      a.dataset = next();
        else if (k == "--val-frac")     a.val_frac = std::stof(next());
        else if (k == "--seed")         a.seed = std::stoi(next());
        else if (k == "--train-frames") a.train_frames = std::stoi(next());
        else if (k == "--whole-clip")   a.whole_clip = true;
        else if (k == "--all")          a.all = true;
        else if (k == "--per-class")    a.per_class = true;
        else if (k == "--confusion")    a.confusion = next();
        else if (k == "--device")       a.device = next();
        else if (k == "--help" || k == "-h") { a.help = true; return true; }
        else fail("cli", "unknown flag '" + k + "'");
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) try {
    Args a;
    if (!parse_args(argc, argv, a) || a.help) { print_help(); return 0; }
    if (a.weights.empty()) fail("phoneme_test", "--weights is required");
    if (a.dataset.empty()) fail("phoneme_test", "--dataset is required");
    if (!fs::exists(a.weights))
        fail("phoneme_test", "weights not found at '" + a.weights + "'");
    if (!fs::exists(a.dataset))
        fail("phoneme_test", "dataset not found at '" + a.dataset + "'");

    bt::init();
    bt::Device device = bt::Device::CPU;
    if (a.device == "auto") {
        device = bt::is_available(bt::Device::CUDA) ? bt::Device::CUDA
                                                    : bt::Device::CPU;
    } else if (a.device == "cuda")  device = bt::Device::CUDA;
    else if  (a.device == "metal") device = bt::Device::Metal;
    else if  (a.device == "cpu")   device = bt::Device::CPU;
    else fail("phoneme_test", "unknown --device '" + a.device + "'");
    const char* dev_name = (device == bt::Device::CUDA)  ? "CUDA"  :
                           (device == bt::Device::Metal) ? "Metal" : "CPU";

    // ── Model (class map restored from the checkpoint) ──
    bsm::PhonemeNet model = bsm::PhonemeNet::load(a.weights, device);
    const auto& cm = model.class_map();
    const int K = cm.num_classes;
    const int sil = cm.silence_class();

    // ── Dataset ──
    auto ds = bsm::read_phoneme_dataset(a.dataset);
    const int n_mels = ds.header.n_mels;
    if (ds.class_map.num_classes != K) {
        fail("phoneme_test", "dataset K=" +
             std::to_string(ds.class_map.num_classes) +
             " != checkpoint K=" + std::to_string(K));
    }

    std::printf("phoneme_test: weights='%s' dataset='%s'\n",
                a.weights.c_str(), a.dataset.c_str());
    std::printf("              device=%s  K=%d  clips=%zu  mode=%s\n",
                dev_name, K, ds.clips.size(),
                a.whole_clip ? "whole-clip" : "centered-window");

    // ── Reproduce the trainer's held-out split (same seed/shuffle) ──
    std::vector<int> eval_idx;
    if (a.all) {
        eval_idx.resize(ds.clips.size());
        for (int i = 0; i < static_cast<int>(eval_idx.size()); ++i)
            eval_idx[i] = i;
    } else {
        std::vector<int> all(ds.clips.size());
        for (int i = 0; i < static_cast<int>(all.size()); ++i) all[i] = i;
        std::mt19937 rng(static_cast<std::uint32_t>(a.seed) ^ 0xC3C3C3C3u);
        std::shuffle(all.begin(), all.end(), rng);
        const int n_val = std::max(1, static_cast<int>(
            std::round(a.val_frac * static_cast<float>(all.size()))));
        const int n_train = std::max(1, static_cast<int>(all.size()) - n_val);
        eval_idx.assign(all.begin() + n_train, all.end());
    }
    std::printf("              eval_clips=%zu (%s)\n", eval_idx.size(),
                a.all ? "ALL" : "held-out");

    // ── PCEN mel front-end from the dataset header ──
    bsm::MelConfig mcfg;
    mcfg.sample_rate = ds.header.sample_rate;
    mcfg.n_fft       = ds.header.n_fft;
    mcfg.win_length  = ds.header.win_length;
    mcfg.hop_length  = ds.header.hop_length;
    mcfg.n_mels      = ds.header.n_mels;
    mcfg.compression = bsm::MelCompression::PCEN;
    bsm::MelFrontend mel(mcfg, bt::Device::CPU);

    const int T = a.train_frames;

    // ── Accumulators ──
    long long total = 0, correct = 0;
    long long nonsil_total = 0, nonsil_correct = 0;
    std::vector<long long> counts(static_cast<std::size_t>(K) * K, 0);  // [true*K+pred]

    for (int ci : eval_idx) {
        const auto& clip = ds.clips[static_cast<std::size_t>(ci)];
        const auto pcm = clip.pcm_float();
        bt::Tensor melt;   // (n_mels, n_frames) on CPU
        mel.compute_offline(pcm.data(), static_cast<int>(pcm.size()), melt);
        const int nf = melt.cols;
        if (nf <= 0) continue;

        // Choose the frame range: whole clip, or a centered T-frame window.
        int t0 = 0, span = nf;
        if (!a.whole_clip && nf > T) { t0 = (nf - T) / 2; span = T; }

        // Build the (n_mels, span) feature window and upload to the device.
        std::vector<float> host(static_cast<std::size_t>(n_mels) * span);
        const float* src = melt.host_f32();
        for (int m = 0; m < n_mels; ++m) {
            std::memcpy(host.data() + static_cast<std::size_t>(m) * span,
                        src + static_cast<std::size_t>(m) * nf + t0,
                        static_cast<std::size_t>(span) * sizeof(float));
        }
        bt::Tensor feats = bt::Tensor::from_host_on(device, host.data(),
                                                    n_mels, span);
        bt::Tensor logits;   // (span, K)
        model.forward(feats, logits);
        const int Tout = logits.rows;
        const std::vector<float> lg_host = logits.to_host_vector();
        const float* lg = lg_host.data();

        for (int t = 0; t < Tout; ++t) {
            const int gt = static_cast<int>(
                clip.labels[static_cast<std::size_t>(t0 + t)]);
            if (gt < 0 || gt >= K) continue;
            const float* row = lg + static_cast<std::size_t>(t) * K;
            int pred = 0; float best = row[0];
            for (int k = 1; k < K; ++k)
                if (row[k] > best) { best = row[k]; pred = k; }
            ++counts[static_cast<std::size_t>(gt) * K + pred];
            ++total;
            if (pred == gt) ++correct;
            if (gt != sil) {
                ++nonsil_total;
                if (pred == gt) ++nonsil_correct;
            }
        }
    }

    const double frame_acc = total > 0
        ? static_cast<double>(correct) / static_cast<double>(total) : 0.0;
    const double nonsil_acc = nonsil_total > 0
        ? static_cast<double>(nonsil_correct) / static_cast<double>(nonsil_total)
        : 0.0;

    // ── Per-class recall / precision ──
    if (a.per_class) {
        std::printf("\nper-class (name: support recall precision):\n");
        for (int k = 0; k < K; ++k) {
            long long true_k = 0, pred_k = 0;
            const long long diag = counts[static_cast<std::size_t>(k) * K + k];
            for (int j = 0; j < K; ++j) {
                true_k += counts[static_cast<std::size_t>(k) * K + j];
                pred_k += counts[static_cast<std::size_t>(j) * K + k];
            }
            if (true_k == 0 && pred_k == 0) continue;  // absent class
            const double rec  = true_k > 0
                ? static_cast<double>(diag) / static_cast<double>(true_k) : 0.0;
            const double prec = pred_k > 0
                ? static_cast<double>(diag) / static_cast<double>(pred_k) : 0.0;
            const std::string nm = (k < static_cast<int>(cm.class_names.size()))
                ? cm.class_names[static_cast<std::size_t>(k)]
                : std::to_string(k);
            std::printf("  %-10s sup=%-7lld rec=%.3f prec=%.3f\n",
                        nm.c_str(), true_k, rec, prec);
        }
    }

    // ── Confusion matrix dump ──
    if (!a.confusion.empty()) {
        std::ofstream out(a.confusion);
        if (!out) fail("phoneme_test", "cannot open --confusion '" +
                        a.confusion + "'");
        out << "# K=" << K << " rows=true cols=pred\n";
        out << "true\\pred";
        for (int k = 0; k < K; ++k) {
            const std::string nm = (k < static_cast<int>(cm.class_names.size()))
                ? cm.class_names[static_cast<std::size_t>(k)] : std::to_string(k);
            out << ',' << nm;
        }
        out << '\n';
        for (int i = 0; i < K; ++i) {
            const std::string nm = (i < static_cast<int>(cm.class_names.size()))
                ? cm.class_names[static_cast<std::size_t>(i)] : std::to_string(i);
            out << nm;
            for (int j = 0; j < K; ++j)
                out << ',' << counts[static_cast<std::size_t>(i) * K + j];
            out << '\n';
        }
        std::printf("\nconfusion matrix (%dx%d) written -> '%s'\n", K, K,
                    a.confusion.c_str());
    }

    // Compact confusion summary: the top off-diagonal confusions.
    {
        struct Conf { int t, p; long long n; };
        std::vector<Conf> off;
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j)
                if (i != j && counts[static_cast<std::size_t>(i) * K + j] > 0)
                    off.push_back({i, j, counts[static_cast<std::size_t>(i) * K + j]});
        std::sort(off.begin(), off.end(),
                  [](const Conf& x, const Conf& y) { return x.n > y.n; });
        std::printf("\ntop confusions (true -> pred : count):\n");
        const int show = std::min<int>(8, static_cast<int>(off.size()));
        for (int i = 0; i < show; ++i) {
            auto nm = [&](int k) {
                return (k < static_cast<int>(cm.class_names.size()))
                    ? cm.class_names[static_cast<std::size_t>(k)]
                    : std::to_string(k);
            };
            std::printf("  %-10s -> %-10s : %lld\n",
                        nm(off[static_cast<std::size_t>(i)].t).c_str(),
                        nm(off[static_cast<std::size_t>(i)].p).c_str(),
                        off[static_cast<std::size_t>(i)].n);
        }
    }

    std::printf("\nframe_acc=%.4f nonsil_acc=%.4f n_frames=%lld nonsil_frames=%lld\n",
                frame_acc, nonsil_acc, total, nonsil_total);
    std::printf("gate(nonsil_acc>=0.85): %s\n",
                nonsil_acc >= 0.85 ? "PASS" : "below-gate (smoke runs expected)");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "phoneme_test: %s\n", e.what());
    return 1;
}
