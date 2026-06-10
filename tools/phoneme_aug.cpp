#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

// brosoundml_phoneme_aug — waveform-domain augmentation: BPDS in -> BPDS out.
//
// The calibration sweep's root cause was training-data realism: clean synth /
// studio speech gives posteriors that collapse on noisy, reverberant, or
// level-shifted audio. This tool multiplies an existing labelled shard with
// acoustically-degraded variants while keeping the frame labels UNCHANGED —
// every augmentation here is time-aligned (additive noise at a sampled SNR,
// direct-path-aligned room impulse responses, gain), never time-warping.
//
//   --dataset PATH[,...]  input BPDS shard(s)
//   --out PATH            output BPDS (augmented variants only; train on
//                         clean.bpds,aug.bpds to keep the originals)
//   --noise-dir DIR       additive-noise corpus, recursive *.wav @16k (MUSAN)
//   --rir-dir DIR         room impulse responses, recursive *.wav @16k
//   --variants K          augmented copies per clip (default 2)
//   --p-noise/--p-rir P   per-variant application probabilities
//   --snr-min/--snr-max   additive-noise SNR range in dB (default 5..25)
//   --gain-db G           per-variant uniform gain in [-G, +G] dB (default 6)
//
// A variant that ends up with neither noise nor RIR still gets gain, so no
// emitted clip is a bit-exact duplicate of its source.

#include "brosoundml/audio.h"
#include "brosoundml/phoneme_data.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs  = std::filesystem;
namespace bt  = brotensor;
namespace bsm = brosoundml;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml_phoneme_aug: " + msg);
}

struct Args {
    std::string dataset, out, noise_dir, rir_dir;
    int   variants = 2;
    float p_noise = 0.8f, p_rir = 0.5f;
    float snr_min = 5.0f, snr_max = 25.0f;
    float gain_db = 6.0f;
    int   noise_cache = 256;   // noise files preloaded into RAM
    int   rir_cache   = 512;   // RIRs preloaded into RAM
    int   seed = 1234;
    bool  help = false;
};

void print_help() {
    std::printf(
        "brosoundml_phoneme_aug — time-aligned waveform augmentation, BPDS -> BPDS\n\n"
        "  --dataset PATH[,...]   input BPDS shard(s)\n"
        "  --out PATH             output BPDS (augmented variants only)\n"
        "  --noise-dir DIR        noise corpus, recursive *.wav 16k (e.g. MUSAN)\n"
        "  --rir-dir DIR          room impulse responses, recursive *.wav 16k\n"
        "  --variants K           augmented copies per clip (default 2)\n"
        "  --p-noise P            P(additive noise) per variant (default 0.8)\n"
        "  --p-rir P              P(reverb) per variant (default 0.5)\n"
        "  --snr-min/--snr-max F  SNR range dB (default 5..25)\n"
        "  --gain-db F            uniform gain range +-F dB (default 6)\n"
        "  --noise-cache N        noise files preloaded (default 256)\n"
        "  --rir-cache N          RIRs preloaded (default 512)\n"
        "  --seed N\n  -h --help\n");
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out; std::string cur;
    for (char ch : s) { if (ch==',') { if(!cur.empty()) out.push_back(cur); cur.clear(); }
                        else cur.push_back(ch); }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) fail("missing value"); return argv[++i]; };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "-h" || k == "--help") a.help = true;
        else if (k == "--dataset")     a.dataset = need(i);
        else if (k == "--out")         a.out = need(i);
        else if (k == "--noise-dir")   a.noise_dir = need(i);
        else if (k == "--rir-dir")     a.rir_dir = need(i);
        else if (k == "--variants")    a.variants = std::stoi(need(i));
        else if (k == "--p-noise")     a.p_noise = std::stof(need(i));
        else if (k == "--p-rir")       a.p_rir = std::stof(need(i));
        else if (k == "--snr-min")     a.snr_min = std::stof(need(i));
        else if (k == "--snr-max")     a.snr_max = std::stof(need(i));
        else if (k == "--gain-db")     a.gain_db = std::stof(need(i));
        else if (k == "--noise-cache") a.noise_cache = std::stoi(need(i));
        else if (k == "--rir-cache")   a.rir_cache = std::stoi(need(i));
        else if (k == "--seed")        a.seed = std::stoi(need(i));
        else fail("unknown arg: " + k);
    }
    return a;
}

// Recursively collect *.wav under `dir`, shuffle, load up to `cap` of them as
// 16 kHz mono float (files at any other rate are skipped). Empty dir -> empty.
std::vector<std::vector<float>> load_pool(const std::string& dir, int cap,
                                          int target_sr, std::mt19937& rng,
                                          const char* what) {
    std::vector<std::vector<float>> pool;
    if (dir.empty()) return pool;
    std::vector<fs::path> files;
    for (const auto& e : fs::recursive_directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().extension() == ".wav")
            files.push_back(e.path());
    }
    std::shuffle(files.begin(), files.end(), rng);
    int skipped_sr = 0;
    for (const auto& p : files) {
        if (cap > 0 && (int)pool.size() >= cap) break;
        bsm::AudioBuffer ab;
        try { ab = bsm::read_wav(p.string()); } catch (...) { continue; }
        if (ab.sample_rate != target_sr) { ++skipped_sr; continue; }
        if (ab.samples.size() < 1600) continue;   // <0.1 s is useless
        pool.push_back(std::move(ab.samples));
    }
    std::fprintf(stderr, "  %s pool: %zu files loaded (of %zu found, %d wrong-rate)\n",
                 what, pool.size(), files.size(), skipped_sr);
    return pool;
}

double rms(const std::vector<float>& x) {
    if (x.empty()) return 0.0;
    double s = 0.0;
    for (float v : x) s += (double)v * v;
    return std::sqrt(s / (double)x.size());
}

// Prepare one RIR: align to the direct path (drop everything before the peak)
// so convolution does not time-shift the signal relative to its labels, trim
// to at most 0.5 s, and peak-normalize.
std::vector<float> prep_rir(std::vector<float> h, int sr) {
    std::size_t peak = 0; float pv = 0.0f;
    for (std::size_t i = 0; i < h.size(); ++i)
        if (std::fabs(h[i]) > pv) { pv = std::fabs(h[i]); peak = i; }
    if (pv <= 0.0f) return {};
    h.erase(h.begin(), h.begin() + (std::ptrdiff_t)peak);
    const std::size_t max_len = (std::size_t)(sr / 2);
    if (h.size() > max_len) h.resize(max_len);
    for (float& v : h) v /= pv;
    return h;
}

// y = (x conv h)[:n] via one rfft/complex_mul/irfft round trip (CPU tensors).
std::vector<float> fft_convolve_head(const std::vector<float>& x,
                                     const std::vector<float>& h) {
    const int n = (int)x.size();
    const int m = (int)h.size();
    if (n == 0 || m == 0) return x;
    int N = 1;
    while (N < n + m - 1) N <<= 1;

    bt::Tensor X = bt::Tensor::zeros_on(bt::Device::CPU, 1, N);
    bt::Tensor H = bt::Tensor::zeros_on(bt::Device::CPU, 1, N);
    std::memcpy(X.host_f32_mut(), x.data(), (std::size_t)n * sizeof(float));
    std::memcpy(H.host_f32_mut(), h.data(), (std::size_t)m * sizeof(float));

    bt::Tensor Xf, Hf, Yf, y;
    bt::rfft(X, Xf);
    bt::rfft(H, Hf);
    bt::complex_mul(Xf, Hf, Yf);
    bt::irfft(Yf, N, y);

    std::vector<float> out((std::size_t)n);
    std::memcpy(out.data(), y.host_f32(), (std::size_t)n * sizeof(float));
    return out;
}

}  // namespace

int main(int argc, char** argv) try {
    Args a = parse_args(argc, argv);
    if (a.help || a.dataset.empty() || a.out.empty()) { print_help(); return a.help ? 0 : 1; }
    if (a.noise_dir.empty() && a.rir_dir.empty())
        fail("need at least one of --noise-dir / --rir-dir");

    bt::init();
    std::mt19937 rng((std::uint32_t)a.seed);

    // ── Inputs ──
    const auto paths = split_csv(a.dataset);
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
    const int sr = ds.header.sample_rate;
    std::fprintf(stderr, "phoneme_aug: %zu clips x %d variants -> %s\n",
                 ds.clips.size(), a.variants, a.out.c_str());

    // ── Noise / RIR pools ──
    auto noise = load_pool(a.noise_dir, a.noise_cache, sr, rng, "noise");
    auto rirs_raw = load_pool(a.rir_dir, a.rir_cache, sr, rng, "rir");
    std::vector<std::vector<float>> rirs;
    for (auto& h : rirs_raw) {
        auto p = prep_rir(std::move(h), sr);
        if (!p.empty()) rirs.push_back(std::move(p));
    }
    if (!a.noise_dir.empty() && noise.empty()) fail("noise pool is empty");
    if (!a.rir_dir.empty()   && rirs.empty())  fail("rir pool is empty");

    bsm::PhonemeDatasetWriter writer(a.out, ds.header, ds.class_map);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    const auto t0 = std::chrono::steady_clock::now();
    long n_noise = 0, n_rir = 0;
    for (std::size_t ci = 0; ci < ds.clips.size(); ++ci) {
        const auto& clip = ds.clips[ci];
        const std::vector<float> clean = clip.pcm_float();
        if (clean.empty()) continue;

        for (int v = 0; v < a.variants; ++v) {
            std::vector<float> x = clean;

            if (!rirs.empty() && uni(rng) < a.p_rir) {
                const auto& h = rirs[rng() % rirs.size()];
                x = fft_convolve_head(x, h);
                ++n_rir;
            }

            if (!noise.empty() && uni(rng) < a.p_noise) {
                const auto& nz = noise[rng() % noise.size()];
                const double sx = rms(x);
                if (sx > 1e-6) {
                    // Random segment, looped if the noise file is shorter.
                    std::vector<float> seg(x.size());
                    std::size_t off = rng() % nz.size();
                    for (std::size_t i = 0; i < seg.size(); ++i)
                        seg[i] = nz[(off + i) % nz.size()];
                    const double sn = rms(seg);
                    if (sn > 1e-9) {
                        std::uniform_real_distribution<float> snr(a.snr_min, a.snr_max);
                        const double g = sx / (sn * std::pow(10.0, snr(rng) / 20.0));
                        for (std::size_t i = 0; i < x.size(); ++i)
                            x[i] += (float)(g * seg[i]);
                        ++n_noise;
                    }
                }
            }

            std::uniform_real_distribution<float> gdb(-a.gain_db, a.gain_db);
            float g = std::pow(10.0f, gdb(rng) / 20.0f);
            // Keep headroom: if the variant would clip, scale it back under 1.
            float peak = 0.0f;
            for (float s : x) peak = std::max(peak, std::fabs(s) * g);
            if (peak > 0.99f) g *= 0.99f / peak;
            for (float& s : x) s *= g;

            writer.append(x, clip.labels);
        }
        if ((ci + 1) % 2000 == 0)
            std::fprintf(stderr, "  %zu / %zu clips ...\n", ci + 1, ds.clips.size());
    }
    writer.finalize();

    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "phoneme_aug done: %d clips (noise applied %ld, rir applied %ld) in %lld s -> %s\n",
        writer.clips(), n_noise, n_rir,
        (long long)std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count(),
        a.out.c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "phoneme_aug: %s\n", e.what());
    return 1;
}
