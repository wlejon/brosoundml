#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

// brosoundml_wake_train — wake-word BC-ResNet trainer.
//
// Reads a chunk-3 synth-dataset manifest, pre-caches log-mel tensors on disk
// (so each epoch is just tensor I/O), trains the chunk-5 BcResnet with Adam +
// fused BCE-with-logits, evaluates on a stratified held-out split, and writes
// a fused-BN inference-ready `.bw` checkpoint at the end of training.
//
// All compute is on the CPU. The model is 22k parameters and the chunk-5
// unfused BN backward is host-only, so CPU is the only valid device.

#include "brosoundml/audio.h"
#include "brosoundml/bc_resnet.h"
#include "brosoundml/mel.h"
#include "brosoundml/wake_data.h"

#include <brotensor/ops.h>
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
#include <fstream>
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

struct Args {
    std::string dataset        = "../brosoundml-data/wake/computer";
    std::string cache_dir;                          // <dataset>/mel-cache by default
    std::string out_checkpoint = "weights/wake/computer.bw";
    std::string resume;                             // empty = none
    std::string device         = "auto";            // 'auto'|'cpu'|'cuda'|'metal'
    float       val_frac       = 0.1f;
    int         epochs         = 50;
    int         batch_size     = 32;
    float       lr             = 1e-3f;
    float       lr_min         = 1e-5f;
    float       pos_weight     = 3.0f;
    int         seed           = 42;
    int         save_every     = 5;
    bool        small          = false;
    bool        help           = false;
};

void print_help() {
    std::cout <<
        "brosoundml_wake_train — wake-word trainer\n"
        "  --dataset DIR        chunk-3 dataset root (manifest.csv inside)\n"
        "  --cache-dir DIR      mel-cache dir (default <dataset>/mel-cache)\n"
        "  --out-checkpoint P   final fused .bw (default weights/wake/computer.bw)\n"
        "  --val-frac F         held-out split (default 0.1)\n"
        "  --epochs N           training epochs (default 50)\n"
        "  --batch-size N       (default 32)\n"
        "  --lr F               peak LR (default 1e-3)\n"
        "  --lr-min F           cosine floor (default 1e-5)\n"
        "  --pos-weight F       BCE positive-class weight (default 3.0)\n"
        "  --seed N             (default 42)\n"
        "  --save-every N       periodic checkpoint cadence (default 5)\n"
        "  --resume PATH        warm-start from a .bw\n"
        "  --device cpu|cuda    target device (default auto — CUDA if available)\n"
        "  --small              3 epochs / batch 4 — smoke-test preset\n";
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) fail("cli", "missing value for " + k);
            return std::string(argv[++i]);
        };
        if      (k == "--dataset")        a.dataset = next();
        else if (k == "--cache-dir")      a.cache_dir = next();
        else if (k == "--out-checkpoint") a.out_checkpoint = next();
        else if (k == "--val-frac")       a.val_frac = std::stof(next());
        else if (k == "--epochs")         a.epochs = std::stoi(next());
        else if (k == "--batch-size")     a.batch_size = std::stoi(next());
        else if (k == "--lr")             a.lr = std::stof(next());
        else if (k == "--lr-min")         a.lr_min = std::stof(next());
        else if (k == "--pos-weight")     a.pos_weight = std::stof(next());
        else if (k == "--seed")           a.seed = std::stoi(next());
        else if (k == "--save-every")     a.save_every = std::stoi(next());
        else if (k == "--resume")         a.resume = next();
        else if (k == "--small")          a.small = true;
        else if (k == "--device")         a.device = next();
        else if (k == "--help" || k == "-h") { a.help = true; return true; }
        else fail("cli", "unknown flag '" + k + "'");
    }
    if (a.small) { a.epochs = 3; a.batch_size = 4; }
    return true;
}

// ─── Mel cache ────────────────────────────────────────────────────────────
//
// Each WAV maps to a single .mel file:
//   header  : magic 'BMEL' u32, version u32, n_mels u32, T u32, mel_cfg_hash u32
//   payload : n_mels * T  FP32 (row-major (n_mels, T))
// `mel_cfg_hash` is a tiny FNV-1a over the MelConfig fields so changing the
// front-end auto-invalidates every cached file. Cache miss → recompute via
// MelFrontend::compute_offline and write atomically (write to tmp then rename).

constexpr std::uint32_t kMelCacheMagic   = 0x4C454D42u;  // 'BMEL'
constexpr std::uint32_t kMelCacheVersion = 1u;

std::uint32_t mel_cfg_hash(const bsm::MelConfig& c) {
    std::uint32_t h = 0x811C9DC5u;
    auto bump = [&](std::uint32_t x) {
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<std::uint8_t>((x >> (i * 8)) & 0xFFu);
            h *= 0x01000193u;
        }
    };
    bump(static_cast<std::uint32_t>(c.sample_rate));
    bump(static_cast<std::uint32_t>(c.n_fft));
    bump(static_cast<std::uint32_t>(c.win_length));
    bump(static_cast<std::uint32_t>(c.hop_length));
    bump(static_cast<std::uint32_t>(c.n_mels));
    std::uint32_t fmin_bits, fmax_bits;
    std::memcpy(&fmin_bits, &c.fmin, 4);
    std::memcpy(&fmax_bits, &c.fmax, 4);
    bump(fmin_bits);
    bump(fmax_bits);
    bump(static_cast<std::uint32_t>(c.window));
    bump(static_cast<std::uint32_t>(c.formula));
    // Compression + PCEN params — changing the front-end output must
    // invalidate every cached mel.
    bump(static_cast<std::uint32_t>(c.compression));
    if (c.compression == bsm::MelCompression::PCEN) {
        std::uint32_t b;
        std::memcpy(&b, &c.pcen_s,     4); bump(b);
        std::memcpy(&b, &c.pcen_alpha, 4); bump(b);
        std::memcpy(&b, &c.pcen_delta, 4); bump(b);
        std::memcpy(&b, &c.pcen_r,     4); bump(b);
        std::memcpy(&b, &c.pcen_eps,   4); bump(b);
    }
    return h;
}

// Read a cached mel; returns true on hit (and fills buf to length n_mels*T).
bool try_load_cached(const std::string& path, int n_mels, int T,
                     std::uint32_t cfg_hash, std::vector<float>& buf) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::uint32_t magic = 0, version = 0, cn = 0, cT = 0, ch = 0;
    if (std::fread(&magic, 4, 1, fp) != 1 ||
        std::fread(&version, 4, 1, fp) != 1 ||
        std::fread(&cn, 4, 1, fp) != 1 ||
        std::fread(&cT, 4, 1, fp) != 1 ||
        std::fread(&ch, 4, 1, fp) != 1) {
        std::fclose(fp); return false;
    }
    if (magic != kMelCacheMagic || version != kMelCacheVersion ||
        static_cast<int>(cn) != n_mels || static_cast<int>(cT) != T ||
        ch != cfg_hash) {
        std::fclose(fp); return false;
    }
    buf.assign(static_cast<std::size_t>(n_mels) * T, 0.0f);
    const std::size_t want = buf.size() * sizeof(float);
    const bool ok = std::fread(buf.data(), 1, want, fp) == want;
    std::fclose(fp);
    return ok;
}

void save_cached(const std::string& path, int n_mels, int T,
                 std::uint32_t cfg_hash, const std::vector<float>& buf) {
    const std::string tmp = path + ".tmp";
    {
        std::FILE* fp = std::fopen(tmp.c_str(), "wb");
        if (!fp) fail("wake_train::cache", "could not open '" + tmp +
                       "' for writing");
        const std::uint32_t magic = kMelCacheMagic;
        const std::uint32_t ver   = kMelCacheVersion;
        const std::uint32_t cn    = static_cast<std::uint32_t>(n_mels);
        const std::uint32_t cT    = static_cast<std::uint32_t>(T);
        std::fwrite(&magic,    4, 1, fp);
        std::fwrite(&ver,      4, 1, fp);
        std::fwrite(&cn,       4, 1, fp);
        std::fwrite(&cT,       4, 1, fp);
        std::fwrite(&cfg_hash, 4, 1, fp);
        std::fwrite(buf.data(), sizeof(float), buf.size(), fp);
        std::fclose(fp);
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        // Fallback: copy then remove (Windows rename across devices etc.)
        fs::copy_file(tmp, path, fs::copy_options::overwrite_existing, ec);
        fs::remove(tmp, ec);
    }
}

// Compute + cache the mel for one wav path. Returns the (n_mels*T) FP32 buf.
std::vector<float> mel_for(const std::string& wav_abs,
                           const std::string& cache_path,
                           bsm::MelFrontend& mel,
                           const bsm::MelConfig& mcfg,
                           int T_expected, std::uint32_t cfg_hash) {
    std::vector<float> buf;
    if (try_load_cached(cache_path, mcfg.n_mels, T_expected, cfg_hash, buf)) {
        return buf;
    }
    // Miss → recompute.
    auto audio = bsm::read_wav(wav_abs);
    if (audio.sample_rate != mcfg.sample_rate) {
        fail("wake_train::mel",
             "wav '" + wav_abs + "' rate " + std::to_string(audio.sample_rate) +
             " != mel sample_rate " + std::to_string(mcfg.sample_rate));
    }
    bt::Tensor out;
    mel.compute_offline(audio.samples.data(),
                        static_cast<int>(audio.samples.size()), out);
    // Truncate / pad along T to T_expected. Wake-synth clips are 1.0 s ±2 ms
    // so they naturally land at T_expected = 1 + (16000-400)/160 = 98. A
    // sample one frame short gets zero-padded; one frame long is right-clipped.
    const int T_got = out.cols;
    buf.assign(static_cast<std::size_t>(mcfg.n_mels) * T_expected, 0.0f);
    const int T_copy = std::min(T_got, T_expected);
    const float* src = out.host_f32();
    for (int m = 0; m < mcfg.n_mels; ++m) {
        std::memcpy(buf.data() + static_cast<std::size_t>(m) * T_expected,
                    src     + static_cast<std::size_t>(m) * T_got,
                    static_cast<std::size_t>(T_copy) * sizeof(float));
    }
    fs::create_directories(fs::path(cache_path).parent_path());
    save_cached(cache_path, mcfg.n_mels, T_expected, cfg_hash, buf);
    return buf;
}

// Build a stratified train/val split with a deterministic shuffle.
void split_stratified(const std::vector<bsm::ManifestRow>& rows, float val_frac,
                      std::uint32_t seed,
                      std::vector<int>& train_idx,
                      std::vector<int>& val_idx) {
    std::vector<int> pos, neg;
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        (rows[static_cast<std::size_t>(i)].label == 1 ? pos : neg).push_back(i);
    }
    auto shuffle_and_split = [&](std::vector<int>& v, std::uint32_t s) {
        std::mt19937 rng(s);
        std::shuffle(v.begin(), v.end(), rng);
        const int n_val = std::max(1, static_cast<int>(
            std::round(val_frac * static_cast<float>(v.size()))));
        const int n_train = static_cast<int>(v.size()) - n_val;
        for (int i = 0; i < n_train; ++i) train_idx.push_back(v[i]);
        for (int i = n_train; i < static_cast<int>(v.size()); ++i) val_idx.push_back(v[i]);
    };
    shuffle_and_split(pos, seed ^ 0xA1A1A1A1u);
    shuffle_and_split(neg, seed ^ 0xB2B2B2B2u);
    // Re-shuffle the combined splits so the order is not "all positives then
    // all negatives" — Adam doesn't care but later batch-stat-driven layers
    // (training BN) do.
    std::mt19937 rng(seed ^ 0xC3C3C3C3u);
    std::shuffle(train_idx.begin(), train_idx.end(), rng);
    std::shuffle(val_idx.begin(),   val_idx.end(),   rng);
}

float cosine_lr(int epoch, int total_epochs, float lr_peak, float lr_min) {
    if (total_epochs <= 1) return lr_peak;
    const float t = static_cast<float>(epoch) /
                    static_cast<float>(total_epochs - 1);
    const float pi = 3.14159265358979323846f;
    return lr_min + 0.5f * (lr_peak - lr_min) * (1.0f + std::cos(pi * t));
}

// Pack a minibatch: copy each mel row (n_mels*T floats) into (B, n_mels*T)
// NCL.  Labels (B,1) FP32 0/1.
//
// When `volume_rng` is non-null, every clip in the batch is volume-augmented
// by sampling α ∈ [kVolMin, kVolMax] uniformly and adding `2 * ln(α)` to
// every log-mel bin. That's mathematically identical to scaling the source
// waveform by α before mel extraction (since log(α²·mel) = log(mel)+2·ln(α)),
// so the cache stays valid. Each call to pack_batch with a non-null rng
// draws fresh α's, giving the model a different volume per clip per epoch.
//
// The training data was peak-normalized to 0.99 — a desktop mic delivers
// speech at 0.02–0.10 peak. Without this augmentation, BC-ResNet learns to
// rely on absolute log-mel magnitude and stops firing on quiet input. With
// kVolMin=0.05 / kVolMax=0.95, every training clip gets at least 5%
// reduction and up to 95% — covering the real mic delivery range and well
// past it. (Earlier kVolMin=0.20 left a gap below 0.20 peak that desktop
// mics often land in, which the chunk-7 probe flagged.)
constexpr float kVolMin = 0.05f;
constexpr float kVolMax = 0.95f;
constexpr float kLogEpsFloor = -23.025851f;  // ln(1e-10) — matches mel kEps

// Build host-side batch tensors first (cheap host loops with the volume
// augmentation), then upload to `device` in a single h2d copy. Allocating
// directly on a GPU device and looping with host_f32 would be a per-cell
// roundtrip — unworkable. The host-stage-then-upload pattern is what
// brotensor::Tensor::from_host_on is for.
void pack_batch(const std::vector<std::vector<float>>& cache_buf,
                const std::vector<int>& idx, int start, int B,
                int n_mels, int T,
                const std::vector<bsm::ManifestRow>& rows,
                bt::Device device,
                bt::Tensor& mel_out, bt::Tensor& lab_out,
                std::mt19937* volume_rng = nullptr) {
    const std::size_t row_size = static_cast<std::size_t>(n_mels) * T;
    std::vector<float> mel_host(static_cast<std::size_t>(B) * row_size, 0.0f);
    std::vector<float> lab_host(static_cast<std::size_t>(B), 0.0f);
    std::uniform_real_distribution<float> vol_dist(kVolMin, kVolMax);
    for (int b = 0; b < B; ++b) {
        const int ri = idx[static_cast<std::size_t>(start + b)];
        float* dst = mel_host.data() + static_cast<std::size_t>(b) * row_size;
        std::memcpy(dst,
                    cache_buf[static_cast<std::size_t>(ri)].data(),
                    row_size * sizeof(float));
        if (volume_rng) {
            const float alpha = vol_dist(*volume_rng);
            const float delta = 2.0f * std::log(alpha);   // negative
            for (std::size_t k = 0; k < row_size; ++k) {
                float v = dst[k] + delta;
                if (v < kLogEpsFloor) v = kLogEpsFloor;
                dst[k] = v;
            }
        }
        lab_host[static_cast<std::size_t>(b)] =
            static_cast<float>(rows[static_cast<std::size_t>(ri)].label);
    }
    mel_out = bt::Tensor::from_host_on(device, mel_host.data(),
                                        B, static_cast<int>(row_size));
    lab_out = bt::Tensor::from_host_on(device, lab_host.data(), B, 1);
}

}  // namespace

int main(int argc, char** argv) try {
    Args a;
    if (!parse_args(argc, argv, a) || a.help) { print_help(); return 0; }

    // ── Resolve paths ──
    const fs::path dataset_root = fs::path(a.dataset);
    const fs::path manifest_path = dataset_root / "manifest.csv";
    if (!fs::exists(manifest_path)) {
        fail("wake_train", "manifest not found at '" + manifest_path.string() +
                            "'");
    }
    if (a.cache_dir.empty()) {
        a.cache_dir = (dataset_root / "mel-cache").string();
    }
    fs::create_directories(a.cache_dir);
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
    else fail("wake_train", "unknown --device '" + a.device + "'");

    const char* dev_name = (device == bt::Device::CUDA)  ? "CUDA"  :
                           (device == bt::Device::Metal) ? "Metal" : "CPU";

    // ── Mel front-end + cache key ──
    // Mel features are cached to disk as device-agnostic FP32 bytes; we keep
    // the offline mel computation on CPU because it's a one-shot prep step
    // (and it's already plenty fast at our dataset sizes).
    bsm::MelConfig mcfg;   // defaults match wake-synth: 16 kHz, 40 mels, 25 ms / 10 ms
    bsm::MelFrontend mel(mcfg, bt::Device::CPU);
    const int T_expected = 1 + (mcfg.sample_rate - mcfg.win_length) /
                            mcfg.hop_length;   // 1.0 s clip → 98 frames
    const std::uint32_t cfg_hash = mel_cfg_hash(mcfg);

    std::cout << "wake_train: dataset='" << dataset_root.string() << "'\n"
              << "            cache='"   << a.cache_dir << "'\n"
              << "            device="   << dev_name << "\n"
              << "            n_mels="   << mcfg.n_mels
              << "  T="                  << T_expected
              << "  cfg_hash=0x" << std::hex << cfg_hash << std::dec << "\n";

    // ── Manifest + stratified split ──
    auto rows = bsm::read_manifest(manifest_path.string());
    std::cout << "            rows="     << rows.size() << "\n";
    if (rows.empty()) fail("wake_train", "manifest is empty");

    std::vector<int> train_idx, val_idx;
    split_stratified(rows, a.val_frac, static_cast<std::uint32_t>(a.seed),
                     train_idx, val_idx);
    std::cout << "            train="    << train_idx.size()
              << "  val="                 << val_idx.size() << "\n";

    // ── Build / refresh mel cache for every clip we'll use ──
    std::vector<std::vector<float>> cache_buf(rows.size());
    {
        auto t0 = std::chrono::steady_clock::now();
        int processed = 0;
        auto walk = [&](const std::vector<int>& idx) {
            for (int ri : idx) {
                const auto& row = rows[static_cast<std::size_t>(ri)];
                const fs::path wav_abs   = dataset_root / row.path;
                const fs::path cache_abs = fs::path(a.cache_dir) /
                                          (row.path + ".mel");
                cache_buf[static_cast<std::size_t>(ri)] =
                    mel_for(wav_abs.string(), cache_abs.string(), mel, mcfg,
                            T_expected, cfg_hash);
                ++processed;
                if (processed % 100 == 0) {
                    std::cout << "            cache progress: " << processed
                              << "/" << (train_idx.size() + val_idx.size())
                              << "\n";
                }
            }
        };
        walk(train_idx);
        walk(val_idx);
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "            cache built in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                        t1 - t0).count() << " ms\n";
    }

    // ── Model ──
    // Bumped from the BcResnetConfig defaults (32,32,48,56,64 ≈ 22k params).
    // The chunk-7 probe showed the tiny model collapsed onto a narrow
    // band-energy shortcut (any sustained tone in 500-2000 Hz fired it at 1.0)
    // because it had enough capacity to memorise that shortcut but not the
    // phonetic structure of "computer". Widening to ~55k params gives the
    // network room to learn temporal phonetic patterns without breaching the
    // 2 ms/frame inference budget on CPU.
    bsm::BcResnetConfig bcfg;
    bcfg.c0 =  64;
    bcfg.c1 =  80;
    bcfg.c2 =  96;
    bcfg.c3 = 112;
    bcfg.c4 = 128;
    if (bcfg.n_mels != mcfg.n_mels) {
        fail("wake_train", "BcResnet n_mels=" + std::to_string(bcfg.n_mels) +
              " != MelConfig n_mels=" + std::to_string(mcfg.n_mels));
    }
    bsm::BcResnet model = a.resume.empty()
        ? bsm::BcResnet::make(bcfg, device)
        : bsm::BcResnet::load(a.resume, device);
    if (a.resume.empty()) {
        model.xavier_init_weights(static_cast<std::uint64_t>(a.seed));
    }
    std::cout << "            params=" << model.param_count() << "\n";

    // ── Training loop ──
    const int B   = a.batch_size;
    const int n_train_batches = static_cast<int>(train_idx.size()) / B;
    std::cout << "            steps/epoch=" << n_train_batches << "\n\n";

    float best_val_loss = std::numeric_limits<float>::infinity();
    int   best_epoch    = -1;
    const auto t_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < a.epochs; ++epoch) {
        const float lr = cosine_lr(epoch, a.epochs, a.lr, a.lr_min);

        // Shuffle training order — seed + epoch keeps it reproducible.
        std::mt19937 rng(static_cast<std::uint32_t>(a.seed) +
                         static_cast<std::uint32_t>(epoch * 9176u));
        std::shuffle(train_idx.begin(), train_idx.end(), rng);

        // Separate rng for volume augmentation so it advances independently
        // of the shuffle draws (keeps either-or determinism debugging clean).
        std::mt19937 vol_rng(static_cast<std::uint32_t>(a.seed) ^
                             static_cast<std::uint32_t>(epoch * 0xD1B54A35u));

        double train_loss_sum = 0.0;
        int    train_seen     = 0;

        for (int step = 0; step < n_train_batches; ++step) {
            bt::Tensor mel_batch, labels;
            pack_batch(cache_buf, train_idx, step * B, B, mcfg.n_mels,
                        T_expected, rows, device, mel_batch, labels, &vol_rng);
            const float loss = model.train_step(mel_batch, labels, B,
                                                T_expected, lr, a.pos_weight);
            train_loss_sum += loss;
            ++train_seen;
        }

        // Eval over the entire val set (one big batch is fine — small).
        double val_loss_sum = 0.0;
        double val_acc_sum  = 0.0;
        double val_frr_sum  = 0.0;
        double val_fpr_sum  = 0.0;
        int    val_batches  = 0;
        for (int s = 0; s < static_cast<int>(val_idx.size()); s += B) {
            const int eB = std::min(B, static_cast<int>(val_idx.size()) - s);
            if (eB == 0) break;
            bt::Tensor mel_batch, labels;
            pack_batch(cache_buf, val_idx, s, eB, mcfg.n_mels,
                        T_expected, rows, device, mel_batch, labels);
            const auto m = model.eval_step(mel_batch, labels, eB, T_expected,
                                            a.pos_weight);
            val_loss_sum += m.loss;
            val_acc_sum  += m.accuracy;
            val_frr_sum  += m.frr;
            val_fpr_sum  += m.fpr;
            ++val_batches;
        }
        const float val_loss = val_batches > 0
            ? static_cast<float>(val_loss_sum / val_batches) : 0.0f;
        const float val_acc  = val_batches > 0
            ? static_cast<float>(val_acc_sum / val_batches) : 0.0f;
        const float val_frr  = val_batches > 0
            ? static_cast<float>(val_frr_sum / val_batches) : 0.0f;
        const float val_fpr  = val_batches > 0
            ? static_cast<float>(val_fpr_sum / val_batches) : 0.0f;
        const float train_loss = train_seen > 0
            ? static_cast<float>(train_loss_sum / train_seen) : 0.0f;

        std::cout << "epoch " << (epoch + 1) << "/" << a.epochs
                  << "  train_loss=" << train_loss
                  << "  val_loss="   << val_loss
                  << "  val_acc="    << val_acc
                  << "  frr="        << val_frr
                  << "  fpr="        << val_fpr
                  << "  lr="         << lr << "\n";

        // Best-on-val checkpoint (unfused — for resume).
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            best_epoch    = epoch + 1;
            const std::string best_path = a.out_checkpoint + ".best";
            model.save(best_path, /*fused=*/false);
        }
        // Periodic checkpoint.
        if (a.save_every > 0 && ((epoch + 1) % a.save_every == 0)) {
            const std::string p = a.out_checkpoint + ".epoch" +
                                  std::to_string(epoch + 1);
            model.save(p, /*fused=*/false);
        }
    }

    // ── Final fuse + save ──
    model.fuse_bn();
    model.save(a.out_checkpoint, /*fused=*/true);
    const auto t_end = std::chrono::steady_clock::now();
    const auto wall_s = std::chrono::duration_cast<std::chrono::seconds>(
                            t_end - t_start).count();

    std::cout << "\nsummary:\n"
              << "  best_epoch=" << best_epoch
              << "  best_val_loss=" << best_val_loss << "\n"
              << "  fused checkpoint -> '" << a.out_checkpoint << "'\n"
              << "  wall_clock=" << wall_s << "s\n";
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "wake_train: %s\n", e.what());
    return 1;
}
