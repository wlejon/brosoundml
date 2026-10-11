#define _CRT_SECURE_NO_WARNINGS
// YuE2-3B end to end on the GPU (minutes; runs only with YUE2_E2E=1), on the
// request ../YuE2-3B/examples/tonight-awake.json (or $YUE2_REQUEST).
//
//   A. Synthesis parity on a whole song: with $YUE2_REF_DIR (the reference
//      CLI's output directory: prefix.npy, abc_tokens.npy, semantic.npy,
//      latent.npy) the reference's plan + semantic tokens go through our
//      synthesize(); the noise is the same torch CPU draw, so the latents
//      compare element-wise.
//   B. generate() in protocol mode (one chunk per song): timings (TTFT, tokens
//      per second, time to first latent, NAR seconds per audio second) and
//      latent statistics beside the reference song's.
//   C. generate() streaming (nar_chunk_frames = $YUE2_STREAM_FRAMES, default
//      50 = 2 s): the same seed gives the same tokens, so the windowed latents
//      compare to B's; the jump across a window boundary is measured against
//      the frame-to-frame step inside windows.
// With $YUE2_VAE_DIR each song is also decoded (Yue2VaeStream fed from
// on_latents / on_done, as an app would) and written to $YUE2_OUT_DIR as WAV.
// Numbers are reported; nothing judges the audio.

#include "test_yue2_common.h"

#include "brotensor/json.h"
#include "brosoundml/yue2.h"
#include "brosoundml/yue2_vae.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <sstream>

using namespace yue2_test;
using brosoundml::Yue2;

namespace {

std::string env(const char* k, const std::string& d = {}) {
    const char* v = std::getenv(k);
    return v && *v ? std::string(v) : d;
}

double now_s() {
    using C = std::chrono::steady_clock;
    static const auto t0 = C::now();
    return std::chrono::duration<double>(C::now() - t0).count();
}

struct Stats {
    double mean = 0, std = 0;
    std::vector<double> dim_std;   // per latent channel
    double step = 0;               // mean |x[t] - x[t-1]| per element
};
Stats stats(const std::vector<float>& x, int frames) {
    Stats s;
    s.dim_std.assign(64, 0.0);
    std::vector<double> dm(64, 0.0);
    for (int t = 0; t < frames; ++t)
        for (int d = 0; d < 64; ++d) dm[d] += x[t * 64 + d];
    for (int d = 0; d < 64; ++d) {
        dm[d] /= frames;
        s.mean += dm[d] / 64;
    }
    double var = 0;
    for (int t = 0; t < frames; ++t)
        for (int d = 0; d < 64; ++d) {
            const double e = x[t * 64 + d] - dm[d];
            s.dim_std[d] += e * e;
            const double g = x[t * 64 + d] - s.mean;
            var += g * g;
            if (t) s.step += std::fabs(x[t * 64 + d] - x[(t - 1) * 64 + d]);
        }
    for (auto& v : s.dim_std) v = std::sqrt(v / frames);
    s.std = std::sqrt(var / (static_cast<double>(frames) * 64));
    s.step /= std::max(1.0, (frames - 1) * 64.0);
    return s;
}
double corr(const std::vector<double>& a, const std::vector<double>& b) {
    double ma = 0, mb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) ma += a[i] / a.size(), mb += b[i] / b.size();
    double sab = 0, saa = 0, sbb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
    }
    return sab / std::sqrt(saa * sbb + 1e-30);
}
void print_stats(const char* name, const Stats& s, int frames) {
    std::printf("  %s: %d frames (%.1f s), mean %.4f, std %.4f, mean |step| %.4f\n", name, frames, frames / 25.0,
                s.mean, s.std, s.step);
}

struct Decoder {
    const brosoundml::Yue2Vae* vae = nullptr;
    std::unique_ptr<brosoundml::Yue2VaeStream> stream;
    brosoundml::StereoAudio audio;
    double first_audio = -1, t0 = 0;
    void start(double t) {
        if (!vae) return;
        t0 = t;
        audio = {};
        first_audio = -1;
        stream = std::make_unique<brosoundml::Yue2VaeStream>(
            *vae, [this](const brosoundml::StereoAudio& c, std::int64_t) {
                if (first_audio < 0) first_audio = now_s() - t0;
                // planar chunks: append per channel
                brosoundml::StereoAudio n;
                n.frames = audio.frames + c.frames;
                n.samples.resize(static_cast<std::size_t>(n.frames) * 2);
                for (int ch = 0; ch < 2; ++ch) {
                    std::copy(audio.channel(ch), audio.channel(ch) + audio.frames, n.channel(ch));
                    std::copy(c.channel(ch), c.channel(ch) + c.frames, n.channel(ch) + audio.frames);
                }
                audio = std::move(n);
            });
    }
};

}  // namespace

int main() {
    if (env("YUE2_E2E") != "1") {
        std::printf("SKIP: set YUE2_E2E=1 (minutes of GPU)\n");
        return 0;
    }
    brotensor::init();
    const fs::path mdir = model_dir();
    if (mdir.empty()) {
        std::printf("SKIP: no YuE2-3B folder\n");
        return 0;
    }
    const fs::path req_path = env("YUE2_REQUEST", (mdir / "examples" / "tonight-awake.json").string());
    std::ifstream rf(req_path.string(), std::ios::binary);
    std::stringstream ss;
    ss << rf.rdbuf();
    const auto req = brotensor::json::parse(ss.str());
    brosoundml::Yue2Params p;
    p.style = req.get_string("style", "");
    p.lyrics = req.get_string("lyrics", "");
    const std::string cot = req.get_string("cot", "full");
    p.cot = cot == "off" ? brosoundml::Yue2Cot::Off : cot == "melody" ? brosoundml::Yue2Cot::Melody
                                                                          : brosoundml::Yue2Cot::Full;
    p.seed = static_cast<std::uint64_t>(req.get_int("seed", 831001));
    p.abc_draft_tokens = std::atoi(env("YUE2_ABC_DRAFTS", "0").c_str());   // speculative planning
    const fs::path out = env("YUE2_OUT_DIR", (repo() / "build" / "yue2_e2e").string());
    fs::create_directories(out);

    Yue2 y;
    double t = now_s();
    // $YUE2_AR_WEIGHTS = bf16 (default) / int8 / fp8: the decode's weights.
    brosoundml::Yue2LoadOptions lo;
    const std::string aw = env("YUE2_AR_WEIGHTS", "bf16");
    lo.ar_weights = aw == "int8" ? brosoundml::Yue2ArWeights::Int8
                    : aw == "fp8" ? brosoundml::Yue2ArWeights::Fp8
                                  : brosoundml::Yue2ArWeights::Bf16;
    y.load(mdir.string(), brotensor::Device::CUDA, lo);
    std::printf("load %.1f s (AR decode weights %s)\n", now_s() - t, aw.c_str());
    std::unique_ptr<brosoundml::Yue2Vae> vae;
    if (const std::string vd = env("YUE2_VAE_DIR"); !vd.empty()) {
        vae = std::make_unique<brosoundml::Yue2Vae>();
        vae->load(vd, brotensor::Device::CUDA, /*decoder_only=*/true);
    }
    Decoder dec;
    dec.vae = vae.get();

    // Reference song.
    const fs::path ref = env("YUE2_REF_DIR");
    std::vector<long long> lshape;
    std::vector<float> ref_lat;
    std::vector<int32_t> ref_sem;
    if (!ref.empty()) {
        ref_lat = read_npy<float>(ref / "latent.npy", &lshape);
        ref_sem = read_npy<int32_t>(ref / "semantic.npy");
    }
    const int ref_frames = ref_lat.empty() ? 0 : static_cast<int>(lshape[0]);
    Stats ref_st;
    if (ref_frames) {
        ref_st = stats(ref_lat, ref_frames);
        std::printf("reference song:\n");
        print_stats("latents", ref_st, ref_frames);
    }

    // YUE2_E2E_SEMANTIC=n: only time n semantic tokens after the reference
    // plan (a profiling seam), then stop.
    if (const int ns = std::atoi(env("YUE2_E2E_SEMANTIC", "0").c_str()); ns > 0 && !ref.empty()) {
        brosoundml::Yue2Plan plan;
        plan.prefix = read_npy<int32_t>(ref / "prefix.npy");
        plan.abc_ids = read_npy<int32_t>(ref / "abc_tokens.npy");
        brosoundml::Yue2Params q = p;
        q.semantic_sampling.min_tokens = ns;
        q.semantic_sampling.max_tokens = ns;
        t = now_s();
        const auto codes = y.semantic(q, plan);
        const double secs = now_s() - t;
        std::printf("semantic only: %zu tokens after a %zu-token prefix in %.2f s = %.1f tok/s\n", codes.size(),
                    plan.prefix.size(), secs, codes.size() / secs);
        return 0;
    }

    // YUE2_E2E_TTFA=1: time to first latent and first audio only, for
    // cot = full and cot = off, streaming at $YUE2_STREAM_FRAMES (default 50)
    // with Yue2VaeStream's defaults; each run is cancelled once audio exists.
    // One cot = off run first warms the GPU (graph capture, allocator).
    if (env("YUE2_E2E_TTFA") == "1") {
        if (!vae) {
            std::printf("SKIP: YUE2_E2E_TTFA needs YUE2_VAE_DIR\n");
            return 0;
        }
        const int W = std::atoi(env("YUE2_STREAM_FRAMES", "50").c_str());
        auto ttfa = [&](brosoundml::Yue2Cot c, bool print) {
            brosoundml::Yue2Params q = p;
            q.cot = c;
            q.nar_chunk_frames = W;
            brosoundml::Yue2Callbacks cb;
            cb.on_latents = [&](int f0, int n, const float* d) { dec.stream->push(f0, n, d); };
            const double t0 = now_s();
            dec.start(t0);
            const auto r = y.generate(q, cb, [&] { return dec.first_audio >= 0; });
            const auto& tm = r.timing;
            if (print)
                std::printf("TTFA cot=%s W=%d: first latent %.2f s, first audio %.2f s (plan %d tokens in %d forwards, "
                            "%.2f s at %.1f tok/s; semantic TTFT %.1f ms)\n",
                            c == brosoundml::Yue2Cot::Off ? "off" : "full", W, tm.first_latent_seconds,
                            dec.first_audio, tm.plan_tokens, tm.plan_forwards, tm.plan_seconds, tm.plan_tps,
                            tm.semantic_ttft * 1e3);
        };
        ttfa(brosoundml::Yue2Cot::Off, false);
        ttfa(brosoundml::Yue2Cot::Full, true);
        ttfa(brosoundml::Yue2Cot::Off, true);
        return failures ? 1 : 0;
    }

    // A. synthesis parity.
    if (ref_frames && !ref_sem.empty()) {
        brosoundml::Yue2Plan plan;
        plan.prefix = read_npy<int32_t>(ref / "prefix.npy");
        plan.abc_ids = read_npy<int32_t>(ref / "abc_tokens.npy");
        t = now_s();
        const auto lat = y.synthesize(p, plan, ref_sem);
        const double secs = now_s() - t;
        const Diff d = diff(lat, ref_lat);
        std::printf("A. synthesize(reference tokens): %zu frames, cosine %.6f, rms %.4f (ref rms %.4f), max %.3f, "
                    "%.2f s = %.3f s per audio s\n",
                    ref_sem.size(), d.cosine, d.rms, d.ref_rms, d.max_abs, secs, secs / (ref_sem.size() / 25.0));
        CHECK(lat.size() == ref_lat.size(), "A: latent count");
        CHECK(d.cosine > 0.99, "A: synthesis cosine %.6f", d.cosine);
    }

    // B / C.
    auto run = [&](const char* name, int W, std::vector<float>* keep) {
        brosoundml::Yue2Params q = p;
        q.nar_chunk_frames = W;
        brosoundml::Yue2Callbacks cb;
        std::vector<int> bounds;
        cb.on_latents = [&](int f0, int n, const float* d) {
            bounds.push_back(f0);
            if (dec.stream) dec.stream->push(f0, n, d);
        };
        cb.on_done = [&](bool, const std::string&) {
            if (dec.stream) dec.stream->finish();
        };
        const double t0 = now_s();
        dec.start(t0);
        const auto r = y.generate(q, cb);
        const double total = now_s() - t0;
        const auto& tm = r.timing;
        std::printf("%s (nar_chunk_frames %d): %.1f s total for %d frames (%.1f s of audio)\n", name, W, total,
                    r.frames, r.frames / 25.0);
        std::printf("  plan: %d tokens, TTFT %.1f ms, %.1f tok/s, %.1f s%s\n", tm.plan_tokens, tm.plan_ttft * 1e3,
                    tm.plan_tps, tm.plan_seconds, r.plan.truncated ? " (truncated)" : "");
        std::printf("  semantic: %d tokens, TTFT %.1f ms, %.1f tok/s (%.2fx the 25 Hz real-time need), %.1f s%s\n",
                    tm.semantic_tokens, tm.semantic_ttft * 1e3, tm.semantic_tps, tm.semantic_tps / 25.0,
                    tm.semantic_seconds, r.semantic_truncated ? " (truncated)" : "");
        std::printf("  synthesis: %.1f s, %.3f NAR s per audio s; first latent at %.2f s; %zu latent callbacks\n",
                    tm.synth_seconds, tm.nar_seconds_per_audio_second, tm.first_latent_seconds, bounds.size());
        if (dec.stream)
            std::printf("  first audio at %.2f s (Yue2VaeStream)\n", dec.first_audio);
        const Stats st = stats(r.latents, r.frames);
        print_stats("latents", st, r.frames);
        if (ref_frames)
            std::printf("  vs reference song: std ratio %.3f, per-channel std correlation %.4f\n", st.std / ref_st.std,
                        corr(st.dim_std, ref_st.dim_std));
        if (W > 0) {
            // Window seams: the step across a boundary vs the mean step.
            double seam = 0;
            int ns = 0;
            for (int b : bounds) {
                if (b == 0) continue;
                for (int dd = 0; dd < 64; ++dd)
                    seam += std::fabs(r.latents[b * 64 + dd] - r.latents[(b - 1) * 64 + dd]);
                ++ns;
            }
            if (ns) std::printf("  window seams: mean |step| %.4f across %d boundaries = %.2fx the in-song mean\n",
                                seam / (ns * 64.0), ns, seam / (ns * 64.0) / st.step);
        }
        if (dec.stream) {
            const fs::path wav = out / (std::string(name) + ".wav");
            dec.audio.write_wav(wav.string());
            std::printf("  wrote %s (%.1f s)\n", wav.string().c_str(), dec.audio.duration_seconds());
        }
        CHECK(r.frames > 0 && !r.latents.empty(), "%s: no latents", name);
        if (keep) *keep = r.latents;
        return r;
    };
    std::vector<float> lat_b;
    const auto rb = run("B_protocol", 0, &lat_b);
    const int W = std::atoi(env("YUE2_STREAM_FRAMES", "50").c_str());
    std::vector<float> lat_c;
    const auto rc = run("C_streaming", W, &lat_c);
    if (rb.semantic == rc.semantic) {
        const Diff d = diff(lat_c, lat_b);
        std::printf("C vs B (same tokens): latent cosine %.4f, rms diff %.4f (B rms %.4f)\n", d.cosine, d.rms,
                    d.ref_rms);
        // Per position within a window: is the error at the edges?
        std::vector<double> err(W, 0.0);
        std::vector<int> cnt(W, 0);
        for (int f = 0; f < rb.frames; ++f) {
            const int k = f % W;
            for (int dd = 0; dd < 64; ++dd) err[k] += std::fabs(lat_c[f * 64 + dd] - lat_b[f * 64 + dd]);
            cnt[k] += 64;
        }
        std::printf("  mean |C - B| by frame within the window: first %.4f, quarter %.4f, middle %.4f, last %.4f\n",
                    err[0] / cnt[0], err[W / 4] / cnt[W / 4], err[W / 2] / cnt[W / 2], err[W - 1] / cnt[W - 1]);
    } else {
        std::printf("C's tokens differ from B's (%zu vs %zu)\n", rc.semantic.size(), rb.semantic.size());
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
