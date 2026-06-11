#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

// brosoundml_sound_units — build a GENERAL acoustic-unit class space so the
// streaming spotter can enroll NON-WORD sounds (clicks, whistles, hums, ...).
//
// The PhonemeSpotter's matcher is class-space agnostic: it aligns a sequence of
// class ids against a per-frame posterior stream, and enroll_from_audio needs
// no text. Its only "word" dependency is the class INVENTORY — English phoneme
// classes, in which a whistle has no class to land on. This tool replaces that
// inventory with self-supervised units (HuBERT-style discovery, one k-means
// iteration): cluster PCEN-mel frames from a sound-diverse corpus into K units,
// paint per-frame unit labels, and emit a standard BPDS that phoneme_train
// consumes unchanged. The trained checkpoint then drives the SAME spotter —
// "these few sounds in this order" instead of "these phonemes in this order".
//
// Subcommands:
//   gen     --out DIR [--count N] [--seed S]
//           Synthesize labelled non-word target sounds (clicks, whistles,
//           hums) with controlled variation, as 16 kHz mono WAVs. Guarantees
//           the unit inventory has coverage of the sound families the spotter
//           must enroll, independent of what the corpora happen to contain.
//   kmeans  --inputs DIR[,...] --out centroids.bsuc [--units K]
//           [--max-frames N] [--stack S]
//           Fit K unit centroids on PCEN-mel frames (stacked +/-S frames of
//           context) sampled from every WAV under the input dirs. Silence-ish
//           frames (PCM RMS below the floor) are EXCLUDED — silence is class 0
//           by construction and must not waste cluster capacity.
//   label   --inputs DIR[,...] --centroids centroids.bsuc --out shard.bpds
//           [--max-clips N] [--min-dur S] [--max-dur S] [--smooth 0|1]
//           [--chunk-dur S]
//           Assign each frame its nearest unit (1..K; silence -> 0), mode-of-3
//           smooth the track (unit ids are nominal, so mode not median), and
//           write the BPDS with an identity class map (class i owns id i).
//           --chunk-dur slices a file LONGER than max-dur into chunk-dur
//           pieces instead of dropping it (MUSAN music files are whole songs;
//           without chunking the music source would vanish through the
//           duration filter).
//
// The PCEN front-end here is the model's own (MelConfig defaults + PCEN), so
// the units the model trains on are computed from exactly the features it will
// see when streaming.

#include "brosoundml/audio.h"
#include "brosoundml/mel.h"
#include "brosoundml/phoneme_data.h"
#include "brosoundml/wake_data.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
    throw std::runtime_error("brosoundml_sound_units: " + msg);
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur.push_back(ch);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Recursively collect .wav paths under each input dir (sorted for determinism).
std::vector<std::string> collect_wavs(const std::vector<std::string>& dirs) {
    std::vector<std::string> out;
    for (const auto& d : dirs) {
        if (!fs::is_directory(d)) fail("not a directory: " + d);
        for (const auto& e : fs::recursive_directory_iterator(d)) {
            if (e.is_regular_file() && e.path().extension() == ".wav")
                out.push_back(e.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Load a WAV as 16 kHz mono float (resampling if needed).
std::vector<float> load_16k(const std::string& path) {
    bsm::AudioBuffer ab = bsm::read_wav(path);
    if (ab.sample_rate == 16000) return ab.samples;
    return bsm::resample_to(ab.samples, ab.sample_rate, 16000);
}

// Per-frame PCM RMS on the mel framing grid (frame t covers
// [t*hop, t*hop + win)). Used for the silence rule in kmeans + label.
std::vector<float> frame_rms(const std::vector<float>& pcm, int win, int hop,
                             int n_frames) {
    std::vector<float> out(static_cast<std::size_t>(n_frames), 0.0f);
    for (int t = 0; t < n_frames; ++t) {
        const std::size_t a = static_cast<std::size_t>(t) * hop;
        double acc = 0.0;
        for (int i = 0; i < win; ++i) {
            const double v = pcm[a + static_cast<std::size_t>(i)];
            acc += v * v;
        }
        out[static_cast<std::size_t>(t)] =
            static_cast<float>(std::sqrt(acc / win));
    }
    return out;
}

constexpr float kSilenceRms = 3e-3f;   // ~-50 dBFS: below this a frame is silence

bsm::MelConfig unit_mel_config() {
    bsm::MelConfig m;                       // defaults = the model's framing
    m.compression = bsm::MelCompression::PCEN;
    return m;
}

// Compute the freq-major (n_mels, T) PCEN mel of a clip on CPU; returns T.
int clip_mel(bsm::MelFrontend& mel, const std::vector<float>& pcm,
             std::vector<float>& out) {
    bt::Tensor m;
    mel.reset();
    mel.compute_offline(pcm.data(), static_cast<int>(pcm.size()), m);
    out = m.to_host_vector();
    const int n_mels = unit_mel_config().n_mels;
    return static_cast<int>(out.size()) / n_mels;
}

// Stacked feature for frame t: mel columns t-S..t+S (edge-clamped),
// flattened to (2S+1)*n_mels floats. Mel buffer is freq-major (n_mels, T).
void stacked_frame(const std::vector<float>& mel, int n_mels, int T, int t,
                   int S, float* dst) {
    int k = 0;
    for (int o = -S; o <= S; ++o) {
        int tc = t + o;
        if (tc < 0) tc = 0;
        if (tc >= T) tc = T - 1;
        for (int f = 0; f < n_mels; ++f)
            dst[k++] = mel[static_cast<std::size_t>(f) * T + tc];
    }
}

// ─── Centroid file (BSUC) ────────────────────────────────────────────────────
// u32 magic 'B''S''U''C', u32 version=1, u32 K, u32 dim, u32 stack,
// then K*dim f32 centroids.
constexpr std::uint32_t kMagicBSUC = 0x43555342u;

void write_centroids(const std::string& path, int K, int dim, int stack,
                     const std::vector<float>& c) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) fail("cannot open for write: " + path);
    const std::uint32_t hdr[5] = {kMagicBSUC, 1u, static_cast<std::uint32_t>(K),
                                  static_cast<std::uint32_t>(dim),
                                  static_cast<std::uint32_t>(stack)};
    std::fwrite(hdr, sizeof(hdr), 1, f);
    std::fwrite(c.data(), sizeof(float), c.size(), f);
    std::fclose(f);
}

struct Centroids {
    int K = 0, dim = 0, stack = 0;
    std::vector<float> c;   // (K, dim)
};

Centroids read_centroids(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) fail("cannot open: " + path);
    std::uint32_t hdr[5];
    if (std::fread(hdr, sizeof(hdr), 1, f) != 1 || hdr[0] != kMagicBSUC ||
        hdr[1] != 1u) {
        std::fclose(f);
        fail("bad centroid file: " + path);
    }
    Centroids out;
    out.K = static_cast<int>(hdr[2]);
    out.dim = static_cast<int>(hdr[3]);
    out.stack = static_cast<int>(hdr[4]);
    out.c.resize(static_cast<std::size_t>(out.K) * out.dim);
    if (std::fread(out.c.data(), sizeof(float), out.c.size(), f) !=
        out.c.size()) {
        std::fclose(f);
        fail("truncated centroid file: " + path);
    }
    std::fclose(f);
    return out;
}

int nearest_centroid(const Centroids& C, const float* x) {
    int best = 0;
    double best_d = 1e300;
    for (int k = 0; k < C.K; ++k) {
        const float* c = C.c.data() + static_cast<std::size_t>(k) * C.dim;
        double d = 0.0;
        for (int i = 0; i < C.dim; ++i) {
            const double e = static_cast<double>(x[i]) - c[i];
            d += e * e;
        }
        if (d < best_d) { best_d = d; best = k; }
    }
    return best;
}

// ─── gen: synthetic non-word target sounds ───────────────────────────────────

void write_clip(const std::string& dir, const std::string& name, int idx,
                std::vector<float> pcm) {
    bsm::AudioBuffer ab;
    ab.sample_rate = 16000;
    ab.samples     = std::move(pcm);
    char fn[64];
    std::snprintf(fn, sizeof(fn), "%s_%03d.wav", name.c_str(), idx);
    ab.write_wav(dir + "/" + fn);
}

void fade_edges(std::vector<float>& x, int n_fade) {
    const int n = static_cast<int>(x.size());
    for (int i = 0; i < n_fade && i < n; ++i) {
        const float g = static_cast<float>(i) / n_fade;
        x[static_cast<std::size_t>(i)] *= g;
        x[static_cast<std::size_t>(n - 1 - i)] *= g;
    }
}

// Center `sound` in a silent clip with timing jitter, so the entry gate sees a
// genuine boundary on both sides.
std::vector<float> embed_in_silence(const std::vector<float>& sound,
                                    std::mt19937& rng) {
    std::uniform_int_distribution<int> lead(1600, 4800);   // 0.1-0.3 s
    const int pre  = lead(rng);
    const int post = lead(rng);
    std::vector<float> out(static_cast<std::size_t>(pre) + sound.size() + post,
                           0.0f);
    std::copy(sound.begin(), sound.end(), out.begin() + pre);
    return out;
}

void cmd_gen(const std::string& out_dir, int count, unsigned seed) {
    fs::create_directories(out_dir);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    for (int i = 0; i < count; ++i) {
        // click: a 3-12 ms decaying burst — white noise + a damped resonance,
        // the shape of mouth clicks / key taps / finger snaps.
        {
            const int   n     = 48 + static_cast<int>(uni(rng) * 144);
            const float fres  = 1500.0f + uni(rng) * 3500.0f;
            const float decay = 600.0f + uni(rng) * 1400.0f;
            const float amp   = 0.3f + uni(rng) * 0.5f;
            std::vector<float> s(static_cast<std::size_t>(n));
            std::normal_distribution<float> g(0.0f, 1.0f);
            for (int t = 0; t < n; ++t) {
                const float tt  = static_cast<float>(t) / 16000.0f;
                const float env = std::exp(-decay * tt);
                s[static_cast<std::size_t>(t)] = amp * env *
                    (0.5f * g(rng) +
                     0.5f * std::sin(2.0f * 3.14159265f * fres * tt));
            }
            write_clip(out_dir, "click", i, embed_in_silence(s, rng));
        }
        // whistle: a pure tone 0.3-1.0 s with a pitch contour (flat / up /
        // down / up-down) and light vibrato.
        {
            const int   n  = 4800 + static_cast<int>(uni(rng) * 11200);
            const float f0 = 800.0f + uni(rng) * 2000.0f;
            const float sweep = (uni(rng) * 2.0f - 1.0f) * 0.6f;  // +/-60% by end
            const bool  bend  = uni(rng) < 0.4f;                  // up-down arc
            const float vib_hz = 4.0f + uni(rng) * 3.0f;
            const float vib    = 0.01f + uni(rng) * 0.02f;
            const float amp    = 0.25f + uni(rng) * 0.35f;
            std::vector<float> s(static_cast<std::size_t>(n));
            double phase = 0.0;
            for (int t = 0; t < n; ++t) {
                const float u = static_cast<float>(t) / n;
                float f = f0 * (1.0f + sweep * (bend ? 4.0f * u * (1.0f - u)
                                                     : u));
                f *= 1.0f + vib * std::sin(2.0f * 3.14159265f * vib_hz * t /
                                           16000.0f);
                phase += 2.0 * 3.14159265358979 * f / 16000.0;
                s[static_cast<std::size_t>(t)] =
                    amp * static_cast<float>(std::sin(phase));
            }
            fade_edges(s, 160);
            write_clip(out_dir, "whistle", i, embed_in_silence(s, rng));
        }
        // hum: low f0 with a 1/k harmonic stack and slow drift — closed-mouth
        // voicing without phonetic structure.
        {
            const int   n  = 6400 + static_cast<int>(uni(rng) * 12800);
            const float f0 = 110.0f + uni(rng) * 150.0f;
            const float drift = (uni(rng) * 2.0f - 1.0f) * 0.12f;
            const int   harmonics = 4 + static_cast<int>(uni(rng) * 4.0f);
            const float amp = 0.2f + uni(rng) * 0.3f;
            std::vector<float> s(static_cast<std::size_t>(n), 0.0f);
            std::vector<double> phase(static_cast<std::size_t>(harmonics), 0.0);
            for (int t = 0; t < n; ++t) {
                const float u = static_cast<float>(t) / n;
                const float f = f0 * (1.0f + drift * u);
                float v = 0.0f;
                for (int h = 0; h < harmonics; ++h) {
                    phase[static_cast<std::size_t>(h)] +=
                        2.0 * 3.14159265358979 * f * (h + 1) / 16000.0;
                    v += std::sin(static_cast<float>(
                             phase[static_cast<std::size_t>(h)])) / (h + 1);
                }
                s[static_cast<std::size_t>(t)] = amp * v / 1.5f;
            }
            fade_edges(s, 320);
            write_clip(out_dir, "hum", i, embed_in_silence(s, rng));
        }
    }
    std::printf("sound_units gen: %d x {click,whistle,hum} -> %s\n", count,
                out_dir.c_str());
}

// ─── kmeans: fit the unit inventory ──────────────────────────────────────────

void cmd_kmeans(const std::vector<std::string>& dirs, const std::string& out,
                int K, long long max_frames, int stack, unsigned seed,
                bt::Device dev) {
    const auto wavs = collect_wavs(dirs);
    if (wavs.empty()) fail("no .wav files under inputs");
    const bsm::MelConfig mc = unit_mel_config();
    bsm::MelFrontend mel(mc, dev);
    const int dim = (2 * stack + 1) * mc.n_mels;

    // Sample non-silent stacked frames evenly across clips.
    std::vector<float> X;
    X.reserve(static_cast<std::size_t>(std::min<long long>(max_frames, 1 << 20)) *
              dim);
    long long total = 0;
    const long long per_clip =
        std::max<long long>(8, max_frames / static_cast<long long>(wavs.size()));
    std::mt19937 rng(seed);
    std::vector<float> mbuf;
    int used_clips = 0;
    for (const auto& w : wavs) {
        if (total >= max_frames) break;
        std::vector<float> pcm;
        try { pcm = load_16k(w); } catch (...) { continue; }
        if (static_cast<int>(pcm.size()) < mc.win_length) continue;
        const int T = clip_mel(mel, pcm, mbuf);
        if (T <= 0) continue;
        const auto rms = frame_rms(pcm, mc.win_length, mc.hop_length, T);
        std::vector<int> cand;
        for (int t = 0; t < T; ++t)
            if (rms[static_cast<std::size_t>(t)] >= kSilenceRms)
                cand.push_back(t);
        if (cand.empty()) continue;
        std::shuffle(cand.begin(), cand.end(), rng);
        const int take = static_cast<int>(
            std::min<long long>(per_clip, static_cast<long long>(cand.size())));
        const std::size_t base = X.size();
        X.resize(base + static_cast<std::size_t>(take) * dim);
        for (int i = 0; i < take; ++i)
            stacked_frame(mbuf, mc.n_mels, T, cand[static_cast<std::size_t>(i)],
                          stack, X.data() + base + static_cast<std::size_t>(i) * dim);
        total += take;
        ++used_clips;
        if (used_clips % 2000 == 0)
            std::fprintf(stderr, "  %d clips, %lld frames ...\n", used_clips,
                         total);
    }
    const long long N = total;
    if (N < K * 16) fail("too few frames for K=" + std::to_string(K));
    std::fprintf(stderr, "kmeans: %lld frames (dim %d) from %d clips, K=%d\n",
                 N, dim, used_clips, K);

    // k-means++ init.
    std::vector<float> C(static_cast<std::size_t>(K) * dim);
    std::vector<double> d2(static_cast<std::size_t>(N),
                           std::numeric_limits<double>::max());
    std::uniform_int_distribution<long long> pick(0, N - 1);
    long long first = pick(rng);
    std::copy_n(X.data() + first * dim, dim, C.data());
    for (int k = 1; k < K; ++k) {
        const float* prev = C.data() + static_cast<std::size_t>(k - 1) * dim;
        double sum = 0.0;
        for (long long i = 0; i < N; ++i) {
            const float* x = X.data() + i * dim;
            double d = 0.0;
            for (int j = 0; j < dim; ++j) {
                const double e = static_cast<double>(x[j]) - prev[j];
                d += e * e;
            }
            if (d < d2[static_cast<std::size_t>(i)])
                d2[static_cast<std::size_t>(i)] = d;
            sum += d2[static_cast<std::size_t>(i)];
        }
        std::uniform_real_distribution<double> r(0.0, sum);
        double target = r(rng), acc = 0.0;
        long long sel = N - 1;
        for (long long i = 0; i < N; ++i) {
            acc += d2[static_cast<std::size_t>(i)];
            if (acc >= target) { sel = i; break; }
        }
        std::copy_n(X.data() + sel * dim, dim,
                    C.data() + static_cast<std::size_t>(k) * dim);
    }

    // Lloyd iterations.
    std::vector<int> assign(static_cast<std::size_t>(N), 0);
    std::vector<double> acc(static_cast<std::size_t>(K) * dim);
    std::vector<long long> cnt(static_cast<std::size_t>(K));
    Centroids CC;
    CC.K = K; CC.dim = dim; CC.stack = stack;
    for (int it = 0; it < 30; ++it) {
        CC.c = C;
        double inertia = 0.0;
        std::fill(acc.begin(), acc.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0ll);
        for (long long i = 0; i < N; ++i) {
            const float* x = X.data() + i * dim;
            const int a = nearest_centroid(CC, x);
            assign[static_cast<std::size_t>(i)] = a;
            ++cnt[static_cast<std::size_t>(a)];
            double* ar = acc.data() + static_cast<std::size_t>(a) * dim;
            for (int j = 0; j < dim; ++j) ar[j] += x[j];
        }
        for (int k = 0; k < K; ++k) {
            if (cnt[static_cast<std::size_t>(k)] == 0) {
                // Re-seed an empty cluster from a random frame.
                const long long s = pick(rng);
                std::copy_n(X.data() + s * dim, dim,
                            C.data() + static_cast<std::size_t>(k) * dim);
                continue;
            }
            float* cr = C.data() + static_cast<std::size_t>(k) * dim;
            const double* ar = acc.data() + static_cast<std::size_t>(k) * dim;
            for (int j = 0; j < dim; ++j)
                cr[j] = static_cast<float>(ar[j] /
                                           cnt[static_cast<std::size_t>(k)]);
        }
        // Inertia against the centroids the assignment used.
        for (long long i = 0; i < N; ++i) {
            const float* x = X.data() + i * dim;
            const float* cr = CC.c.data() +
                static_cast<std::size_t>(assign[static_cast<std::size_t>(i)]) * dim;
            for (int j = 0; j < dim; ++j) {
                const double e = static_cast<double>(x[j]) - cr[j];
                inertia += e * e;
            }
        }
        std::fprintf(stderr, "  iter %2d  inertia %.4e\n", it, inertia / N);
    }
    write_centroids(out, K, dim, stack, C);
    std::printf("sound_units kmeans: K=%d dim=%d stack=%d -> %s\n", K, dim,
                stack, out.c_str());
}

// ─── label: paint unit labels, write BPDS ────────────────────────────────────

void cmd_label(const std::vector<std::string>& dirs,
               const std::string& centroid_path, const std::string& out,
               int max_clips, float min_dur, float max_dur, bool smooth,
               float chunk_dur, bt::Device dev) {
    const Centroids C = read_centroids(centroid_path);
    const bsm::MelConfig mc = unit_mel_config();
    bsm::MelFrontend mel(mc, dev);

    // Identity class map: class 0 = silence, classes 1..K = units (class i
    // owns id i), no transparent ids. K+1 total.
    bsm::PhonemeClassMap cm;
    cm.num_classes = C.K + 1;
    cm.class_names.push_back("sil");
    cm.class_to_ids.push_back({0});
    for (int k = 0; k < C.K; ++k) {
        char nm[16];
        std::snprintf(nm, sizeof(nm), "u%02d", k);
        cm.class_names.push_back(nm);
        cm.class_to_ids.push_back({k + 1});
    }
    cm.rebuild_inverse();

    bsm::PhonemeDatasetHeader hdr;
    hdr.sample_rate = mc.sample_rate;
    hdr.n_fft       = mc.n_fft;
    hdr.win_length  = mc.win_length;
    hdr.hop_length  = mc.hop_length;
    hdr.n_mels      = mc.n_mels;
    bsm::PhonemeDatasetWriter writer(out, hdr, cm);

    const auto wavs = collect_wavs(dirs);
    if (wavs.empty()) fail("no .wav files under inputs");
    std::vector<float> mbuf, feat(static_cast<std::size_t>(C.dim));
    int written = 0;

    auto label_one = [&](const std::vector<float>& pcm) {
        const int T = clip_mel(mel, pcm, mbuf);
        if (T <= 0) return;
        const auto rms = frame_rms(pcm, mc.win_length, mc.hop_length, T);

        std::vector<int16_t> labels(static_cast<std::size_t>(T), 0);
        for (int t = 0; t < T; ++t) {
            if (rms[static_cast<std::size_t>(t)] < kSilenceRms) continue;
            stacked_frame(mbuf, mc.n_mels, T, t, C.stack, feat.data());
            labels[static_cast<std::size_t>(t)] =
                static_cast<int16_t>(1 + nearest_centroid(C, feat.data()));
        }
        if (smooth && T >= 3) {
            // Mode-of-3: unit ids are nominal, so a median would invent ids.
            std::vector<int16_t> sm = labels;
            for (int t = 1; t + 1 < T; ++t) {
                const int16_t a = labels[static_cast<std::size_t>(t) - 1];
                const int16_t b = labels[static_cast<std::size_t>(t)];
                const int16_t c = labels[static_cast<std::size_t>(t) + 1];
                if (a == c && a != b) sm[static_cast<std::size_t>(t)] = a;
            }
            labels.swap(sm);
        }
        writer.append(pcm, labels);
        ++written;
        if (written % 2000 == 0)
            std::fprintf(stderr, "  %d clips ...\n", written);
    };

    for (const auto& w : wavs) {
        if (max_clips > 0 && written >= max_clips) break;
        std::vector<float> pcm;
        try { pcm = load_16k(w); } catch (...) { continue; }
        const float dur = static_cast<float>(pcm.size()) / mc.sample_rate;
        if (dur < min_dur) continue;
        if (dur <= max_dur) {
            label_one(pcm);
            continue;
        }
        if (chunk_dur <= 0.0f) continue;   // over-long and chunking off: drop
        const std::size_t step =
            static_cast<std::size_t>(chunk_dur * mc.sample_rate);
        for (std::size_t a = 0; a + step / 2 < pcm.size(); a += step) {
            if (max_clips > 0 && written >= max_clips) break;
            const std::size_t b = std::min(pcm.size(), a + step);
            if (static_cast<float>(b - a) / mc.sample_rate < min_dur) break;
            label_one(std::vector<float>(pcm.begin() + a, pcm.begin() + b));
        }
    }
    writer.finalize();
    std::printf("sound_units label: %d clips, K=%d (+sil) -> %s\n", written,
                C.K, out.c_str());
}

void print_help() {
    std::printf(
        "brosoundml_sound_units — self-supervised acoustic-unit class space\n\n"
        "  gen    --out DIR [--count N=200] [--seed S=1234]\n"
        "  kmeans --inputs DIR[,...] --out centroids.bsuc [--units K=64]\n"
        "         [--max-frames N=1500000] [--stack S=1] [--seed S=1234]\n"
        "  label  --inputs DIR[,...] --centroids FILE --out shard.bpds\n"
        "         [--max-clips N=0] [--min-dur 0.3] [--max-dur 20] [--smooth 1]\n"
        "         [--chunk-dur S=0  slice over-long files instead of dropping]\n"
        "  kmeans/label also take --device cuda|cpu (default: cuda if available)\n");
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) { print_help(); return 1; }
    const std::string cmd = argv[1];
    std::string inputs, out, centroids, device;
    int count = 200, units = 64, stack = 1, max_clips = 0, smooth = 1;
    long long max_frames = 1500000;
    float min_dur = 0.3f, max_dur = 20.0f, chunk_dur = 0.0f;
    unsigned seed = 1234;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) fail("missing value for " + std::string(argv[i]));
        return argv[++i];
    };
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "-h" || k == "--help") { print_help(); return 0; }
        else if (k == "--inputs") inputs = need(i);
        else if (k == "--out") out = need(i);
        else if (k == "--centroids") centroids = need(i);
        else if (k == "--count") count = std::stoi(need(i));
        else if (k == "--units") units = std::stoi(need(i));
        else if (k == "--stack") stack = std::stoi(need(i));
        else if (k == "--max-frames") max_frames = std::stoll(need(i));
        else if (k == "--max-clips") max_clips = std::stoi(need(i));
        else if (k == "--min-dur") min_dur = std::stof(need(i));
        else if (k == "--max-dur") max_dur = std::stof(need(i));
        else if (k == "--chunk-dur") chunk_dur = std::stof(need(i));
        else if (k == "--smooth") smooth = std::stoi(need(i));
        else if (k == "--device") device = need(i);
        else if (k == "--seed") seed = static_cast<unsigned>(std::stoul(need(i)));
        else fail("unknown arg: " + k);
    }
    if (out.empty()) fail("--out is required");

    bt::init();
    bt::Device dev = bt::Device::CPU;
    if (device.empty()) {
        if (bt::is_available(bt::Device::CUDA)) dev = bt::Device::CUDA;
    } else if (device == "cuda") {
        if (!bt::is_available(bt::Device::CUDA)) fail("--device cuda: not available");
        dev = bt::Device::CUDA;
    } else if (device != "cpu") {
        fail("--device must be cuda or cpu");
    }
    if (cmd == "kmeans" || cmd == "label")
        std::fprintf(stderr, "sound_units: mel device %s\n",
                     dev == bt::Device::CUDA ? "CUDA" : "CPU");

    if (cmd == "gen")         cmd_gen(out, count, seed);
    else if (cmd == "kmeans") cmd_kmeans(split_csv(inputs), out, units,
                                         max_frames, stack, seed, dev);
    else if (cmd == "label")  cmd_label(split_csv(inputs), centroids, out,
                                        max_clips, min_dur, max_dur,
                                        smooth != 0, chunk_dur, dev);
    else fail("unknown subcommand: " + cmd);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "sound_units: %s\n", e.what());
    return 1;
}
