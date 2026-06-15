#include "brosoundml/cluster_diarizer.h"

#include "brosoundml/sortformer.h"
#include "brosoundml/speaker_encoder.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

float dot(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0;
    const int n = static_cast<int>(a.size());
    for (int i = 0; i < n; ++i) d += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(d);
}

void l2norm(std::vector<float>& v) {
    double n = 0.0;
    for (float x : v) n += static_cast<double>(x) * x;
    n = std::sqrt(std::max(1e-12, n));
    for (float& x : v) x = static_cast<float>(x / n);
}

// One embedding window: its frame span [f0, f1), center frame, and the centered+
// normalized x-vector used for clustering.
struct Window {
    int                f0 = 0, f1 = 0;
    int                center = 0;
    std::vector<float> emb;     // centered, L2-normalized (set after centering)
    int                cluster = -1;
};

// Centroid-linkage agglomerative clustering on cosine of centered embeddings.
// Merges the closest pair while their cosine exceeds `thr`, then keeps merging
// (ignoring thr) until at most `max_k` clusters remain. Returns a per-window
// cluster id in [0, k).
std::vector<int> cluster_cosine(const std::vector<Window>& wins, float thr,
                                int max_k) {
    const int n = static_cast<int>(wins.size());
    const int D = n ? static_cast<int>(wins[0].emb.size()) : 0;
    std::vector<int> owner(n);
    for (int i = 0; i < n; ++i) owner[i] = i;

    // Active clusters: running sum of member embeddings + count + normalized
    // centroid. cosine(centroid_i, centroid_j) is centroid-linkage similarity.
    struct C { std::vector<float> sum; std::vector<float> cen; int count; bool live; };
    std::vector<C> cs(n);
    for (int i = 0; i < n; ++i) {
        cs[i].sum = wins[i].emb;
        cs[i].cen = wins[i].emb;
        cs[i].count = 1;
        cs[i].live = true;
    }
    int live = n;

    auto recompute_centroid = [&](C& c) {
        c.cen = c.sum;
        l2norm(c.cen);
    };

    while (live > 1) {
        float best = -2.0f;
        int bi = -1, bj = -1;
        for (int i = 0; i < n; ++i) {
            if (!cs[i].live) continue;
            for (int j = i + 1; j < n; ++j) {
                if (!cs[j].live) continue;
                const float s = dot(cs[i].cen, cs[j].cen);
                if (s > best) { best = s; bi = i; bj = j; }
            }
        }
        if (bi < 0) break;
        const bool over_cap = live > max_k;
        if (best <= thr && !over_cap) break;     // converged within the cap

        // Merge bj into bi.
        for (int d = 0; d < D; ++d) cs[bi].sum[d] += cs[bj].sum[d];
        cs[bi].count += cs[bj].count;
        recompute_centroid(cs[bi]);
        cs[bj].live = false;
        for (int i = 0; i < n; ++i) if (owner[i] == bj) owner[i] = bi;
        --live;
    }

    // Compact owner ids to a dense [0, k) range.
    std::vector<int> remap(n, -1);
    int k = 0;
    std::vector<int> out(n, 0);
    for (int i = 0; i < n; ++i) {
        int r = owner[i];
        if (remap[r] < 0) remap[r] = k++;
        out[i] = remap[r];
    }
    return out;
}

}  // namespace

// ─── Impl ────────────────────────────────────────────────────────────────────

struct ClusterDiarizer::Impl {
    Sortformer         vad;       // speech activity / VAD
    SpeakerEncoder     enc;       // ECAPA-TDNN x-vectors
    bt::Device         device = bt::Device::CPU;
    bool               loaded = false;
    std::vector<float> pop_mean;  // fixed centering vector (xvector_mean.f32), or empty
};

ClusterDiarizer::ClusterDiarizer() : impl_(std::make_unique<Impl>()) {}
ClusterDiarizer::~ClusterDiarizer() = default;
ClusterDiarizer::ClusterDiarizer(ClusterDiarizer&&) noexcept = default;
ClusterDiarizer& ClusterDiarizer::operator=(ClusterDiarizer&&) noexcept = default;

void ClusterDiarizer::load(const std::string& sortformer_dir,
                           const std::string& speaker_encoder_dir,
                           bt::Device device) {
    bt::init();
    impl_->device = device;
    impl_->vad.load(sortformer_dir, device);
    impl_->enc.load(speaker_encoder_dir);

    // Optional fixed population mean for centering (the x-vector "centering
    // vector"): xvector_mean.f32 beside the encoder, enc_dim() little-endian
    // float32. Centering against a fixed population mean — rather than each
    // recording's own mean — keeps both a minority and a majority speaker stable;
    // a per-recording mean collapses the majority speaker toward zero and
    // over-splits it. Absent file → fall back to the per-recording mean.
    const std::filesystem::path mean_path =
        std::filesystem::path(speaker_encoder_dir) / "xvector_mean.f32";
    impl_->pop_mean.clear();
    if (std::filesystem::exists(mean_path)) {
        const int D = impl_->enc.enc_dim();
        std::ifstream f(mean_path, std::ios::binary);
        std::vector<float> m(static_cast<std::size_t>(D));
        if (f.read(reinterpret_cast<char*>(m.data()),
                   static_cast<std::streamsize>(m.size() * sizeof(float))))
            impl_->pop_mean = std::move(m);
    }
    impl_->loaded = true;
}

bool ClusterDiarizer::loaded() const { return impl_->loaded; }

ClusterDiarizer::Diarization ClusterDiarizer::diarize(const AudioBuffer& audio,
                                                      const Config& cfg) const {
    if (!impl_->loaded)
        fail("ClusterDiarizer::diarize", "no model loaded; call load() first");
    if (audio.empty())
        fail("ClusterDiarizer::diarize", "audio buffer is empty");

    // ── 1. Sortformer activity → per-frame VAD (probability any speaker is on) ──
    const Sortformer::Diarization act = impl_->vad.diarize(audio);
    const int    T  = act.num_frames;
    const int    NS = act.num_speakers;
    const double fs = act.frame_seconds;

    Diarization out;
    out.num_frames    = T;
    out.frame_seconds = fs;
    out.num_speakers  = 0;
    if (T <= 0) return out;

    std::vector<unsigned char> speech(static_cast<std::size_t>(T), 0);
    for (int t = 0; t < T; ++t) {
        double none = 1.0;   // P(no speaker active) = prod (1 - p_s)
        for (int s = 0; s < NS; ++s)
            none *= 1.0 - static_cast<double>(act.probs[static_cast<std::size_t>(t) * NS + s]);
        if (1.0 - none > cfg.vad_threshold) speech[static_cast<std::size_t>(t)] = 1;
    }

    // ── 2. contiguous speech runs ──
    struct Run { int t0, t1; };
    std::vector<Run> runs;
    for (int t = 0; t < T;) {
        if (!speech[static_cast<std::size_t>(t)]) { ++t; continue; }
        int e = t;
        while (e < T && speech[static_cast<std::size_t>(e)]) ++e;
        runs.push_back({t, e});
        t = e;
    }
    if (runs.empty()) return out;   // no speech → zero speakers

    // ── 3. slide embedding windows over each run, embed each ──
    const int step_samples = static_cast<int>(std::lround(fs * audio.sample_rate));
    const int win_f = std::max(1, static_cast<int>(std::lround(cfg.window_seconds / fs)));
    const int hop_f = std::max(1, static_cast<int>(std::lround(cfg.hop_seconds / fs)));
    const int min_f = std::max(1, static_cast<int>(std::lround(cfg.min_window_seconds / fs)));
    const int total = static_cast<int>(audio.samples.size());

    std::vector<Window> wins;
    std::vector<std::vector<float>> raw;   // uncentered x-vectors
    for (const Run& r : runs) {
        const int rlen = r.t1 - r.t0;
        if (rlen < min_f && rlen * fs < cfg.min_window_seconds * 0.5) continue;  // blip
        for (int s = r.t0; s < r.t1; s += hop_f) {
            int e = std::min(s + win_f, r.t1);
            if (e - s < min_f) {                 // tail shorter than a window
                if (s == r.t0) e = r.t1;          // short run: one window = run
                else break;                       // already covered by prior window
            }
            const int a0 = std::min(total, s * step_samples);
            const int a1 = std::min(total, e * step_samples);
            if (a1 - a0 < step_samples) continue;
            AudioBuffer w;
            w.sample_rate = audio.sample_rate;
            w.samples.assign(audio.samples.begin() + a0, audio.samples.begin() + a1);
            Window win;
            win.f0 = s; win.f1 = e; win.center = (s + e) / 2;
            wins.push_back(win);
            raw.push_back(impl_->enc.embed(w));
            if (e >= r.t1) break;
        }
    }
    if (wins.empty()) return out;

    // ── 4. mean-center (cone removal) + normalize → discriminative cosine space ──
    const int D = static_cast<int>(raw[0].size());
    std::vector<float> mean;
    if (static_cast<int>(impl_->pop_mean.size()) == D) {
        mean = impl_->pop_mean;                 // fixed population mean (preferred)
    } else {
        mean.assign(static_cast<std::size_t>(D), 0.0f);   // per-recording fallback
        for (const auto& e : raw) for (int d = 0; d < D; ++d) mean[d] += e[d];
        for (int d = 0; d < D; ++d) mean[d] /= static_cast<float>(raw.size());
    }
    for (std::size_t i = 0; i < wins.size(); ++i) {
        std::vector<float> c = raw[i];
        for (int d = 0; d < D; ++d) c[d] -= mean[d];
        l2norm(c);
        wins[i].emb = std::move(c);
    }

    // ── (debug) per-window centers + centered-cosine matrix, for threshold tuning ──
    if (std::getenv("BROSOUNDML_DIAR_DEBUG")) {
        std::fprintf(stderr, "[diar] %zu windows (center s):", wins.size());
        for (const Window& w : wins)
            std::fprintf(stderr, " %.2f", w.center * fs);
        std::fprintf(stderr, "\n[diar] centered-cosine matrix:\n");
        for (std::size_t i = 0; i < wins.size(); ++i) {
            std::fprintf(stderr, "  %5.2f:", wins[i].center * fs);
            for (std::size_t j = 0; j < wins.size(); ++j)
                std::fprintf(stderr, " %5.2f", dot(wins[i].emb, wins[j].emb));
            std::fprintf(stderr, "\n");
        }
    }

    // ── 5. cluster ──
    std::vector<int> lab = cluster_cosine(wins, cfg.cluster_threshold,
                                          std::max(1, cfg.max_speakers));
    for (std::size_t i = 0; i < wins.size(); ++i) wins[i].cluster = lab[i];
    int k = 0;
    for (int c : lab) k = std::max(k, c + 1);

    // ── 6. assign each speech frame to the nearest window's cluster ──
    std::vector<int> frame_spk(static_cast<std::size_t>(T), -1);
    for (int t = 0; t < T; ++t) {
        if (!speech[static_cast<std::size_t>(t)]) continue;
        int best = -1, bestd = std::numeric_limits<int>::max();
        for (const Window& w : wins) {
            const int dd = std::abs(w.center - t);
            if (dd < bestd) { bestd = dd; best = w.cluster; }
        }
        frame_spk[static_cast<std::size_t>(t)] = best;
    }

    // ── 7. fold clusters with too little total speech into the nearest one ──
    if (k > 1) {
        // centroid per cluster (mean of member window embeddings)
        std::vector<std::vector<float>> cen(k, std::vector<float>(D, 0.0f));
        std::vector<int> cnt(k, 0);
        for (const Window& w : wins) {
            for (int d = 0; d < D; ++d) cen[w.cluster][d] += w.emb[d];
            ++cnt[w.cluster];
        }
        for (int c = 0; c < k; ++c) if (cnt[c]) l2norm(cen[c]);

        std::vector<int> dur(k, 0);
        for (int t = 0; t < T; ++t)
            if (frame_spk[static_cast<std::size_t>(t)] >= 0) ++dur[frame_spk[static_cast<std::size_t>(t)]];
        const int min_dur = static_cast<int>(std::lround(cfg.min_speaker_seconds / fs));

        std::vector<int> merge_to(k);
        for (int c = 0; c < k; ++c) merge_to[c] = c;
        for (int c = 0; c < k; ++c) {
            if (dur[c] >= min_dur || cnt[c] == 0) continue;
            int best = -1; float bestsim = -2.0f;
            for (int o = 0; o < k; ++o) {
                if (o == c || cnt[o] == 0 || dur[o] < min_dur) continue;
                const float s = dot(cen[c], cen[o]);
                if (s > bestsim) { bestsim = s; best = o; }
            }
            if (best < 0)   // no big-enough neighbor; attach to globally largest
                for (int o = 0; o < k; ++o)
                    if (o != c && cnt[o] && dur[o] > (best < 0 ? -1 : dur[best])) best = o;
            if (best >= 0) merge_to[c] = best;
        }
        for (int t = 0; t < T; ++t) {
            int& f = frame_spk[static_cast<std::size_t>(t)];
            if (f >= 0) f = merge_to[f];
        }
    }

    // ── 8. relabel speakers in arrival order ──
    std::vector<int> first(static_cast<std::size_t>(k), -1);
    for (int t = 0; t < T; ++t) {
        const int c = frame_spk[static_cast<std::size_t>(t)];
        if (c >= 0 && first[static_cast<std::size_t>(c)] < 0)
            first[static_cast<std::size_t>(c)] = t;
    }
    std::vector<int> order;
    for (int c = 0; c < k; ++c) if (first[static_cast<std::size_t>(c)] >= 0) order.push_back(c);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return first[static_cast<std::size_t>(a)] <
                                         first[static_cast<std::size_t>(b)]; });
    std::vector<int> relabel(static_cast<std::size_t>(k), -1);
    for (std::size_t i = 0; i < order.size(); ++i)
        relabel[static_cast<std::size_t>(order[i])] = static_cast<int>(i);
    const int S = static_cast<int>(order.size());

    // ── 9. emit one-hot per-frame probabilities ──
    out.num_speakers = S;
    out.probs.assign(static_cast<std::size_t>(T) * S, 0.0f);
    for (int t = 0; t < T; ++t) {
        const int c = frame_spk[static_cast<std::size_t>(t)];
        if (c < 0) continue;
        const int sp = relabel[static_cast<std::size_t>(c)];
        if (sp >= 0) out.probs[static_cast<std::size_t>(t) * S + sp] = 1.0f;
    }
    return out;
}

}  // namespace brosoundml
