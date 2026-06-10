#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

// brosoundml_phoneme_train — per-frame phoneme-posterior trainer.
//
// The open-vocabulary analogue of wake_train. Reads a chunk-2 BPDS dataset
// (PCM + per-mel-frame phoneme-class labels + the embedded class map),
// computes the PCEN-mel for every clip ONCE into host memory at startup, then
// trains the 2D BC-ResNet phoneme head (PhonemeNet) with Adam + framewise
// softmax cross-entropy on fixed-length T-frame windows sampled from each clip.
// Evaluates on a deterministic held-out split each epoch and writes a fused-BN
// inference-ready 'BPM1' checkpoint (with the class map embedded) at the end.
//
// Compute runs on the model's device — CUDA in production (the 2D backward is
// device-resident through brotensor's conv2d/batch_norm/relu ops). The PCEN
// front-end is computed once on the CPU during the in-memory prep step.
//
// Differences from wake_train:
//   • per-FRAME int labels (B,T), not one label per clip.
//   • variable-length clips windowed to a fixed training T (≈ 1 s receptive
//     field). Short clips are zero-padded on the mel with silence-padded labels.
//   • framewise softmax-CE, not BCE-with-logits.
//   • a length-K per-class weight vector (built from the frame histogram), not
//     a single positive-class pos_weight.
//   • mel features are kept IN MEMORY (the dataset fits — one BPDS file, no
//     per-clip WAV re-reads), not in a disk cache.

#include "brosoundml/mel.h"
#include "brosoundml/phoneme_data.h"
#include "brosoundml/phoneme_model.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
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

std::string env_or_empty(const char* name) {
    if (const char* v = std::getenv(name)) return std::string(v);
    return {};
}

// caller-supplied > BROSOUNDML_DATA_DIR env > ../brosoundml-data
std::string default_data_dir() {
    const auto env = env_or_empty("BROSOUNDML_DATA_DIR");
    if (!env.empty() && fs::exists(env)) return env;
    return "../brosoundml-data";
}

struct Args {
    std::string dataset;                             // default <data>/phoneme/english.bpds
    std::string out_checkpoint = "weights/phoneme/english.bpm";
    std::string resume;                              // empty = none
    std::string device         = "auto";            // 'auto'|'cpu'|'cuda'|'metal'
    std::string class_weights  = "sqrt-inv";        // 'inv'|'sqrt-inv'|'uniform'
    float       val_frac       = 0.1f;
    int         epochs         = 50;
    int         batch_size     = 64;
    float       lr             = 1e-3f;
    float       lr_min         = 1e-5f;
    float       silence_weight = 0.3f;
    int         train_frames   = 100;                // ~1 s window = receptive field
    int         seed           = 42;
    int         save_every     = 5;
    bool        small          = false;
    bool        help           = false;

    // Model-capacity overrides (empty/0 = keep PhonemeNetConfig defaults).
    int         c_stem         = 0;                   // stem channels
    std::string channels;                            // comma list, len = n_stages
    std::string blocks;                              // comma list, len = n_stages
};

// Parse "a,b,c,d" into exactly `n` ints. Throws on a bad count.
std::vector<int> parse_int_list(const std::string& s, int n,
                                const std::string& flag) {
    std::vector<int> out;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        out.push_back(std::stoi(s.substr(i, j - i)));
        i = j + 1;
    }
    if (static_cast<int>(out.size()) != n)
        fail("cli", flag + " expects " + std::to_string(n) +
             " comma-separated values, got " + std::to_string(out.size()));
    return out;
}

void print_help() {
    std::cout <<
        "brosoundml_phoneme_train — per-frame phoneme-posterior trainer\n"
        "  --dataset PATH       BPDS dataset (default <data>/phoneme/english.bpds)\n"
        "  --out-checkpoint P   final fused .bpm (default weights/phoneme/english.bpm)\n"
        "  --val-frac F         held-out split (default 0.1)\n"
        "  --epochs N           training epochs (default 50)\n"
        "  --batch-size N       (default 64)\n"
        "  --lr F               peak LR (default 1e-3)\n"
        "  --lr-min F           cosine floor (default 1e-5)\n"
        "  --train-frames T     fixed training window length in frames (default 100)\n"
        "  --silence-weight W   class-0 (silence) weight multiplier (default 0.3)\n"
        "  --class-weights M    inv|sqrt-inv|uniform per-class weighting (default sqrt-inv)\n"
        "  --seed N             (default 42)\n"
        "  --save-every N       periodic checkpoint cadence (default 5)\n"
        "  --resume PATH        warm-start from a .bpm\n"
        "  --device cpu|cuda    target device (default auto — CUDA if available)\n"
        "  --c-stem N           stem channels (default 32)\n"
        "  --channels a,b,c,d   per-stage output channels (default 32,48,64,96)\n"
        "  --blocks a,b,c,d     blocks per stage incl. transition (default 2,2,2,2)\n"
        "  --small              4 epochs / batch 8 — smoke-test preset\n";
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) fail("cli", "missing value for " + k);
            return std::string(argv[++i]);
        };
        if      (k == "--dataset")        a.dataset = next();
        else if (k == "--out-checkpoint") a.out_checkpoint = next();
        else if (k == "--val-frac")       a.val_frac = std::stof(next());
        else if (k == "--epochs")         a.epochs = std::stoi(next());
        else if (k == "--batch-size")     a.batch_size = std::stoi(next());
        else if (k == "--lr")             a.lr = std::stof(next());
        else if (k == "--lr-min")         a.lr_min = std::stof(next());
        else if (k == "--train-frames")   a.train_frames = std::stoi(next());
        else if (k == "--silence-weight") a.silence_weight = std::stof(next());
        else if (k == "--class-weights")  a.class_weights = next();
        else if (k == "--seed")           a.seed = std::stoi(next());
        else if (k == "--save-every")     a.save_every = std::stoi(next());
        else if (k == "--resume")         a.resume = next();
        else if (k == "--c-stem")         a.c_stem = std::stoi(next());
        else if (k == "--channels")       a.channels = next();
        else if (k == "--blocks")         a.blocks = next();
        else if (k == "--small")          a.small = true;
        else if (k == "--device")         a.device = next();
        else if (k == "--help" || k == "-h") { a.help = true; return true; }
        else fail("cli", "unknown flag '" + k + "'");
    }
    if (a.small) { a.epochs = 4; a.batch_size = 8; }
    return true;
}

float cosine_lr(int epoch, int total_epochs, float lr_peak, float lr_min) {
    if (total_epochs <= 1) return lr_peak;
    const float t = static_cast<float>(epoch) /
                    static_cast<float>(total_epochs - 1);
    const float pi = 3.14159265358979323846f;
    return lr_min + 0.5f * (lr_peak - lr_min) * (1.0f + std::cos(pi * t));
}

// One clip's PCEN-mel (n_mels, n_frames) flattened freq-major + its int frame
// labels. Kept resident in host memory; windows are sampled per epoch.
struct ClipFeat {
    std::vector<float> mel;       // n_mels * n_frames, row m at [m*n_frames .. ]
    std::vector<int>   labels;    // n_frames class ids
    int                n_frames = 0;
};

// Extract a fixed T-frame window into row b of a (B, n_mels*T) feats batch
// (freq-major then time: dst[m*T + t]) + a (B,T) label row. A window starting
// at frame t0 copies [t0, t0+T); frames past the clip end (short clips) are
// zero-padded on the mel and silence-padded (class 0) on the labels.
void write_window(const ClipFeat& cf, int t0, int T, int n_mels,
                  float* mel_dst, float* lab_dst) {
    const int nf = cf.n_frames;
    for (int m = 0; m < n_mels; ++m) {
        float*       drow = mel_dst + static_cast<std::size_t>(m) * T;
        const float* srow = cf.mel.data() + static_cast<std::size_t>(m) * nf;
        for (int t = 0; t < T; ++t) {
            const int st = t0 + t;
            drow[t] = (st < nf) ? srow[st] : 0.0f;
        }
    }
    for (int t = 0; t < T; ++t) {
        const int st = t0 + t;
        lab_dst[t] = (st < nf)
            ? static_cast<float>(cf.labels[static_cast<std::size_t>(st)])
            : 0.0f;   // silence pad
    }
}

// Pack a minibatch of B windows. `offsets[b]` is the window start for clip
// idx[start+b]. Host-stage then a single h2d upload (the from_host_on pattern).
void pack_batch(const std::vector<ClipFeat>& feats,
                const std::vector<int>& idx, const std::vector<int>& offsets,
                int start, int B, int n_mels, int T,
                bt::Device device, bt::Tensor& mel_out, bt::Tensor& lab_out) {
    const std::size_t row = static_cast<std::size_t>(n_mels) * T;
    std::vector<float> mel_host(static_cast<std::size_t>(B) * row, 0.0f);
    std::vector<float> lab_host(static_cast<std::size_t>(B) * T, 0.0f);
    for (int b = 0; b < B; ++b) {
        const int ci = idx[static_cast<std::size_t>(start + b)];
        write_window(feats[static_cast<std::size_t>(ci)],
                     offsets[static_cast<std::size_t>(start + b)], T, n_mels,
                     mel_host.data() + static_cast<std::size_t>(b) * row,
                     lab_host.data() + static_cast<std::size_t>(b) * T);
    }
    mel_out = bt::Tensor::from_host_on(device, mel_host.data(), B,
                                       static_cast<int>(row));
    lab_out = bt::Tensor::from_host_on(device, lab_host.data(), B, T);
}

}  // namespace

int main(int argc, char** argv) try {
    Args a;
    if (!parse_args(argc, argv, a) || a.help) { print_help(); return 0; }

    if (a.dataset.empty())
        a.dataset = default_data_dir() + "/phoneme/english.bpds";
    if (!fs::exists(a.dataset))
        fail("phoneme_train", "dataset not found at '" + a.dataset + "'");
    if (fs::path(a.out_checkpoint).has_parent_path())
        fs::create_directories(fs::path(a.out_checkpoint).parent_path());

    // ── Device selection ──
    bt::init();
    bt::Device device = bt::Device::CPU;
    if (a.device == "auto") {
        device = bt::is_available(bt::Device::CUDA) ? bt::Device::CUDA
                                                    : bt::Device::CPU;
    } else if (a.device == "cuda")  device = bt::Device::CUDA;
    else if  (a.device == "metal") device = bt::Device::Metal;
    else if  (a.device == "cpu")   device = bt::Device::CPU;
    else fail("phoneme_train", "unknown --device '" + a.device + "'");
    const char* dev_name = (device == bt::Device::CUDA)  ? "CUDA"  :
                           (device == bt::Device::Metal) ? "Metal" : "CPU";

    // ── Dataset ──
    auto ds = bsm::read_phoneme_dataset(a.dataset);
    const int K = ds.class_map.num_classes;
    const int n_mels = ds.header.n_mels;
    const int T = a.train_frames;
    if (ds.clips.empty()) fail("phoneme_train", "dataset has no clips");
    if (K < 2)            fail("phoneme_train", "class map has K<2");

    std::cout << "phoneme_train: dataset='" << a.dataset << "'\n"
              << "               device=" << dev_name
              << "  clips=" << ds.clips.size()
              << "  K=" << K << "  n_mels=" << n_mels
              << "  train_frames=" << T << "\n";

    // ── PCEN mel front-end from the dataset header ──
    bsm::MelConfig mcfg;
    mcfg.sample_rate = ds.header.sample_rate;
    mcfg.n_fft       = ds.header.n_fft;
    mcfg.win_length  = ds.header.win_length;
    mcfg.hop_length  = ds.header.hop_length;
    mcfg.n_mels      = ds.header.n_mels;
    mcfg.compression = bsm::MelCompression::PCEN;
    bsm::MelFrontend mel(mcfg, bt::Device::CPU);

    // ── Extract PCEN mel + labels for every clip ONCE into host memory ──
    std::vector<ClipFeat> feats(ds.clips.size());
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::size_t total_frames = 0;
        for (std::size_t ci = 0; ci < ds.clips.size(); ++ci) {
            const auto& clip = ds.clips[ci];
            const auto pcm = clip.pcm_float();
            bt::Tensor out;   // (n_mels, T_mel)
            mel.compute_offline(pcm.data(), static_cast<int>(pcm.size()), out);
            const int Tmel = out.cols;
            ClipFeat cf;
            cf.n_frames = Tmel;
            cf.mel.assign(static_cast<std::size_t>(n_mels) * Tmel, 0.0f);
            std::memcpy(cf.mel.data(), out.host_f32(),
                        cf.mel.size() * sizeof(float));
            // The BPDS label track is already frame-aligned to this framing;
            // align defensively to the mel frame count (clamp/truncate).
            cf.labels.resize(static_cast<std::size_t>(Tmel), 0);
            const int nlab = static_cast<int>(clip.labels.size());
            for (int t = 0; t < Tmel; ++t) {
                cf.labels[static_cast<std::size_t>(t)] = (t < nlab)
                    ? static_cast<int>(clip.labels[static_cast<std::size_t>(t)])
                    : 0;
            }
            total_frames += static_cast<std::size_t>(Tmel);
            feats[ci] = std::move(cf);
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::cout << "               mel extracted: " << total_frames
                  << " frames in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                        t1 - t0).count() << " ms (in-memory)\n";
    }

    // ── Deterministic clip-level train/val split ──
    std::vector<int> all(ds.clips.size());
    for (int i = 0; i < static_cast<int>(all.size()); ++i) all[i] = i;
    {
        std::mt19937 rng(static_cast<std::uint32_t>(a.seed) ^ 0xC3C3C3C3u);
        std::shuffle(all.begin(), all.end(), rng);
    }
    const int n_val = std::max(1, static_cast<int>(
        std::round(a.val_frac * static_cast<float>(all.size()))));
    const int n_train = std::max(1, static_cast<int>(all.size()) - n_val);
    std::vector<int> train_idx(all.begin(), all.begin() + n_train);
    std::vector<int> val_idx(all.begin() + n_train, all.end());
    std::cout << "               train=" << train_idx.size()
              << "  val=" << val_idx.size() << "\n";

    // ── Per-class weight vector ──
    //
    // Build a length-K weight from the TRAIN-split per-class frame histogram:
    //   freq[k]   = frames labelled class k (over the train split)
    //   inv:       w[k] = 1 / freq[k]
    //   sqrt-inv:  w[k] = 1 / sqrt(freq[k])
    //   uniform:   w[k] = 1
    // Classes absent from the train split get w[k] = 0 (they never appear, so
    // their weight is irrelevant and is excluded from the normalisation).
    // Normalise so the MEAN weight over PRESENT classes ≈ 1
    //   (w[k] *= n_present / Σ_present w), then MULTIPLY the silence class
    // (k == 0) by --silence-weight. The resulting vector is passed verbatim to
    // train_step / eval_step, which scale each frame's loss+grad by the weight
    // of its TRUE class.
    std::vector<float> class_weights(static_cast<std::size_t>(K), 1.0f);
    {
        std::vector<long long> freq(static_cast<std::size_t>(K), 0);
        for (int ci : train_idx) {
            for (int lab : feats[static_cast<std::size_t>(ci)].labels) {
                if (lab >= 0 && lab < K) ++freq[static_cast<std::size_t>(lab)];
            }
        }
        for (int k = 0; k < K; ++k) {
            const double f = static_cast<double>(freq[static_cast<std::size_t>(k)]);
            double w;
            if (f <= 0.0)                       w = 0.0;
            else if (a.class_weights == "inv")  w = 1.0 / f;
            else if (a.class_weights == "sqrt-inv") w = 1.0 / std::sqrt(f);
            else if (a.class_weights == "uniform")  w = 1.0;
            else fail("phoneme_train", "unknown --class-weights '" +
                       a.class_weights + "'");
            class_weights[static_cast<std::size_t>(k)] = static_cast<float>(w);
        }
        // Normalise mean over present classes to ≈ 1.
        double sum = 0.0; int present = 0;
        for (int k = 0; k < K; ++k) {
            if (freq[static_cast<std::size_t>(k)] > 0) {
                sum += class_weights[static_cast<std::size_t>(k)]; ++present;
            }
        }
        if (sum > 0.0 && present > 0) {
            const float scale = static_cast<float>(present / sum);
            for (auto& w : class_weights) w *= scale;
        }
        // Down-weight silence.
        class_weights[0] *= a.silence_weight;
        std::cout << "               class-weights=" << a.class_weights
                  << "  silence_weight=" << a.silence_weight
                  << "  w[sil]=" << class_weights[0] << "\n";
    }

    // ── Model ──
    bsm::PhonemeNetConfig cfg;
    cfg.n_mels      = n_mels;
    cfg.sample_rate = ds.header.sample_rate;
    cfg.n_fft       = ds.header.n_fft;
    cfg.win_length  = ds.header.win_length;
    cfg.hop_length  = ds.header.hop_length;
    if (a.c_stem > 0) cfg.c_stem = a.c_stem;
    if (!a.channels.empty()) {
        const auto ch = parse_int_list(a.channels, bsm::PhonemeNetConfig::n_stages,
                                       "--channels");
        for (int s = 0; s < bsm::PhonemeNetConfig::n_stages; ++s) cfg.c[s] = ch[s];
    }
    if (!a.blocks.empty()) {
        const auto bl = parse_int_list(a.blocks, bsm::PhonemeNetConfig::n_stages,
                                       "--blocks");
        for (int s = 0; s < bsm::PhonemeNetConfig::n_stages; ++s)
            cfg.n_blocks[s] = bl[s];
    }
    if (a.c_stem > 0 || !a.channels.empty() || !a.blocks.empty()) {
        std::cout << "               model: c_stem=" << cfg.c_stem
                  << " channels=" << cfg.c[0] << "," << cfg.c[1] << ","
                  << cfg.c[2] << "," << cfg.c[3]
                  << " blocks=" << cfg.n_blocks[0] << "," << cfg.n_blocks[1] << ","
                  << cfg.n_blocks[2] << "," << cfg.n_blocks[3] << "\n";
    }
    bsm::PhonemeNet model = a.resume.empty()
        ? bsm::PhonemeNet::make(cfg, ds.class_map, device)
        : bsm::PhonemeNet::load(a.resume, device);
    if (a.resume.empty())
        model.xavier_init_weights(static_cast<std::uint64_t>(a.seed));
    std::cout << "               params=" << model.param_count()
              << "  receptive_field=" << model.receptive_field_frames()
              << " frames\n";

    // ── Training loop ──
    const int B = a.batch_size;
    const int n_train_batches = std::max(1, static_cast<int>(train_idx.size()) / B);
    std::cout << "               steps/epoch=" << n_train_batches << "\n\n";

    float best_val_nonsil = -1.0f;
    int   best_epoch      = -1;
    const auto t_start = std::chrono::steady_clock::now();

    // Per-clip deterministic window offset for val (centered) — same every
    // epoch so val numbers are comparable.
    auto centered_offset = [&](int ci) {
        const int nf = feats[static_cast<std::size_t>(ci)].n_frames;
        return (nf > T) ? (nf - T) / 2 : 0;
    };

    for (int epoch = 0; epoch < a.epochs; ++epoch) {
        const float lr = cosine_lr(epoch, a.epochs, a.lr, a.lr_min);
        std::mt19937 rng(static_cast<std::uint32_t>(a.seed) +
                         static_cast<std::uint32_t>(epoch) * 9176u + 1u);
        std::shuffle(train_idx.begin(), train_idx.end(), rng);

        // Random window offset per training clip this epoch (one window each).
        std::vector<int> train_off(train_idx.size(), 0);
        for (std::size_t i = 0; i < train_idx.size(); ++i) {
            const int nf = feats[static_cast<std::size_t>(train_idx[i])].n_frames;
            if (nf > T) {
                std::uniform_int_distribution<int> d(0, nf - T);
                train_off[i] = d(rng);
            }
        }

        double train_loss_sum = 0.0; int train_seen = 0;
        for (int step = 0; step < n_train_batches; ++step) {
            bt::Tensor mel_batch, labels;
            pack_batch(feats, train_idx, train_off, step * B, B, n_mels, T,
                       device, mel_batch, labels);
            const float loss = model.train_step(mel_batch, labels, B, T, lr,
                                                 class_weights);
            train_loss_sum += loss; ++train_seen;
        }

        // ── Val eval (centered window per clip) ──
        double v_loss = 0.0, v_acc = 0.0, v_nonsil = 0.0;
        long long v_frames = 0; int v_batches = 0;
        std::vector<int> val_off(val_idx.size(), 0);
        for (std::size_t i = 0; i < val_idx.size(); ++i)
            val_off[i] = centered_offset(val_idx[i]);
        for (int s = 0; s < static_cast<int>(val_idx.size()); s += B) {
            const int eB = std::min(B, static_cast<int>(val_idx.size()) - s);
            if (eB == 0) break;
            bt::Tensor mel_batch, labels;
            pack_batch(feats, val_idx, val_off, s, eB, n_mels, T, device,
                       mel_batch, labels);
            const auto m = model.eval_step(mel_batch, labels, eB, T,
                                           class_weights);
            v_loss   += m.loss;
            v_acc    += static_cast<double>(m.frame_accuracy) * m.n_frames;
            v_nonsil += static_cast<double>(m.nonsilence_frame_accuracy) *
                        m.n_frames;   // weighted by frames; nonsil acc is an
                                      // approximation but stable for tracking
            v_frames += m.n_frames;
            ++v_batches;
        }
        const float train_loss = train_seen > 0
            ? static_cast<float>(train_loss_sum / train_seen) : 0.0f;
        const float val_loss = v_batches > 0
            ? static_cast<float>(v_loss / v_batches) : 0.0f;
        const float val_acc = v_frames > 0
            ? static_cast<float>(v_acc / static_cast<double>(v_frames)) : 0.0f;
        const float val_nonsil = v_frames > 0
            ? static_cast<float>(v_nonsil / static_cast<double>(v_frames)) : 0.0f;

        std::cout << "epoch " << (epoch + 1) << "/" << a.epochs
                  << "  train_loss=" << train_loss
                  << "  val_loss=" << val_loss
                  << "  val_acc=" << val_acc
                  << "  val_nonsil_acc=" << val_nonsil
                  << "  lr=" << lr << "\n";

        if (val_nonsil > best_val_nonsil) {
            best_val_nonsil = val_nonsil;
            best_epoch = epoch + 1;
            model.save(a.out_checkpoint + ".best", /*fused=*/false);
        }
        if (a.save_every > 0 && ((epoch + 1) % a.save_every == 0)) {
            model.save(a.out_checkpoint + ".epoch" + std::to_string(epoch + 1),
                       /*fused=*/false);
        }
    }

    // ── Final fuse + save (class map is embedded by save) ──
    model.fuse_bn();
    model.save(a.out_checkpoint, /*fused=*/true);
    const auto t_end = std::chrono::steady_clock::now();
    const auto wall_s = std::chrono::duration_cast<std::chrono::seconds>(
                            t_end - t_start).count();

    std::cout << "\nsummary:\n"
              << "  best_epoch=" << best_epoch
              << "  best_val_nonsil_acc=" << best_val_nonsil << "\n"
              << "  params=" << model.param_count()
              << "  receptive_field=" << model.receptive_field_frames()
              << " frames  K=" << K << "\n"
              << "  fused checkpoint -> '" << a.out_checkpoint << "'\n"
              << "  wall_clock=" << wall_s << "s\n";
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "phoneme_train: %s\n", e.what());
    return 1;
}
