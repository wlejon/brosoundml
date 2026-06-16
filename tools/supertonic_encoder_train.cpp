// supertonic_encoder_train — train the Supertonic AE audio→latent encoder by
// reconstruction through the FROZEN decoder. Supertone never released the
// encoder; this learns one so any clip can be mapped to the latent space the
// flow model generates in (the prerequisite for reference-clip cloning and a
// ControlNet-style flow-matching adapter).
//
// Loop (mirrors wake_train / phoneme_train): scan a 44.1 kHz wav folder, and per
// clip:  audio -> SupertonicSpec -> encoder -> real latent -> frozen decode ->
// multi-resolution STFT loss vs the input -> backprop to the encoder weights ->
// Adam. The decoder is frozen; only the encoder trains. Checkpoint to
// safetensors. CUDA by default (the recon loss runs the full vocoder per step).
//
//   supertonic_encoder_train --data <wavdir> --model <modeldir> --out enc.safetensors
//   supertonic_encoder_train --data <wavdir> --overfit 4   (smoke: loss must fall)

#include "supertonic_encoder.h"
#include "supertonic_spec.h"

#include "brosoundml/audio.h"
#include "brosoundml/supertonic.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace bt = brotensor;
namespace bsm = brosoundml;
namespace fs = std::filesystem;

namespace {

struct Args {
    std::string data;
    std::string model = "D:/projects/brosoundml-data/supertonic";
    std::string out   = "encoder.safetensors";
    std::string cache;           // spec-cache dir (empty = no caching)
    int    epochs    = 4;
    float  lr        = 5.0e-6f;  // peak lr (after warmup). Backprop through the
                                 // frozen vocoder is ill-conditioned; higher lr
                                 // overshoots/diverges.
    int    warmup    = 200;      // linear lr warmup steps (0 = off). Ramping in
                                 // from ~0 avoids the cold-start grad explosion.
    float  lr_floor_frac = 0.1f; // cosine-decay floor as a fraction of peak lr;
                                 // decaying near the optimum kills the bounce.
    int    max_clips = 0;        // 0 = all
    float  max_secs  = 4.0f;     // trim each clip to bound compute
    float  val_frac  = 0.02f;
    int    overfit   = 0;        // >0: overfit N clips (smoke), ignores epochs/val
    int    save_every = 1;
    int    accum     = 16;       // gradient-accumulation batch: per-clip grads are
                                 // norm-clipped then averaged over this many clips
                                 // before one Adam step. Batch-1 SGD diverges on the
                                 // ill-conditioned frozen-vocoder objective; the
                                 // per-clip gradient is far too noisy to step on.
    float  clip      = 1.0f;     // per-clip grad L2-norm clip (0 = off). Bounds each
                                 // clip's contribution to the batch so loud clips
                                 // (huge residual) can't dominate the step, and tames
                                 // the cold-start frozen-vocoder explosion.
    float  init_scale = 0.0f;    // proj_out weight scale at init. 0 => z_real ==
                                 // latent_mean EXACTLY for every clip at step 0
                                 // (ControlNet-style zero init): a guaranteed
                                 // in-distribution start the blocks deviate from only
                                 // as training earns it. Nonzero lets loud-clip specs
                                 // push z_real off-mean at init -> decode diverges.
    float  dlatent_clamp = 1.0e3f; // clamp |dLatent| (decode-backward output) to
                                   // tame frozen-vocoder Inf spikes (0 = off).
    std::uint64_t seed = 1234;
    bool   cpu = false;
    bool   profile = false;   // time each phase (sync-bracketed) over the run
};

[[noreturn]] void usage(const char* msg) {
    std::fprintf(stderr, "supertonic_encoder_train: %s\n", msg);
    std::exit(2);
}

Args parse(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) { if (i + 1 >= argc) usage("missing arg value"); return std::string(argv[++i]); };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--data")       a.data = need(i);
        else if (k == "--model")      a.model = need(i);
        else if (k == "--out")        a.out = need(i);
        else if (k == "--cache")      a.cache = need(i);
        else if (k == "--epochs")     a.epochs = std::stoi(need(i));
        else if (k == "--lr")         a.lr = std::stof(need(i));
        else if (k == "--warmup")     a.warmup = std::stoi(need(i));
        else if (k == "--lr-floor-frac") a.lr_floor_frac = std::stof(need(i));
        else if (k == "--max-clips")  a.max_clips = std::stoi(need(i));
        else if (k == "--max-secs")   a.max_secs = std::stof(need(i));
        else if (k == "--val-frac")   a.val_frac = std::stof(need(i));
        else if (k == "--overfit")    a.overfit = std::stoi(need(i));
        else if (k == "--save-every") a.save_every = std::stoi(need(i));
        else if (k == "--accum")      a.accum = std::max(1, std::stoi(need(i)));
        else if (k == "--clip")       a.clip = std::stof(need(i));
        else if (k == "--init-scale") a.init_scale = std::stof(need(i));
        else if (k == "--dlatent-clamp") a.dlatent_clamp = std::stof(need(i));
        else if (k == "--seed")       a.seed = std::stoull(need(i));
        else if (k == "--cpu")        a.cpu = true;
        else if (k == "--profile")    a.profile = true;
        else usage(("unknown arg: " + k).c_str());
    }
    if (a.data.empty()) usage("--data <wavdir> required");
    return a;
}

// Gather every trainable weight tensor of the encoder in a fixed order; the
// matching grad / Adam-moment tensors are gathered in the SAME order.
std::vector<bt::Tensor*> gather_w(bsm::SupertonicEncoder& e) {
    std::vector<bt::Tensor*> v{ &e.conv_in.w, &e.conv_in.b };
    for (auto& b : e.blocks) {
        v.push_back(&b.dw.w);  v.push_back(&b.dw.b);
        v.push_back(&b.ln_g);  v.push_back(&b.ln_b);
        v.push_back(&b.pw1.w); v.push_back(&b.pw1.b);
        v.push_back(&b.pw2.w); v.push_back(&b.pw2.b);
    }
    v.push_back(&e.proj_out.w); v.push_back(&e.proj_out.b);
    return v;
}
std::vector<bt::Tensor*> gather_g(bsm::SupertonicEncoderGrads& g) {
    std::vector<bt::Tensor*> v{ &g.conv_in.w, &g.conv_in.b };
    for (auto& b : g.blocks) {
        v.push_back(&b.dw.w);  v.push_back(&b.dw.b);
        v.push_back(&b.ln_g);  v.push_back(&b.ln_b);
        v.push_back(&b.pw1.w); v.push_back(&b.pw1.b);
        v.push_back(&b.pw2.w); v.push_back(&b.pw2.b);
    }
    v.push_back(&g.proj_out.w); v.push_back(&g.proj_out.b);
    return v;
}
// Stable names for the checkpoint (so the loader can re-key by structure).
std::vector<std::string> weight_names(const bsm::SupertonicEncoder& e) {
    std::vector<std::string> n{ "conv_in.w", "conv_in.b" };
    for (int i = 0; i < e.num_layers; ++i) {
        const std::string p = "block." + std::to_string(i) + ".";
        n.push_back(p + "dw.w");  n.push_back(p + "dw.b");
        n.push_back(p + "ln_g");  n.push_back(p + "ln_b");
        n.push_back(p + "pw1.w"); n.push_back(p + "pw1.b");
        n.push_back(p + "pw2.w"); n.push_back(p + "pw2.b");
    }
    n.push_back("proj_out.w"); n.push_back("proj_out.b");
    return n;
}

void save_encoder(const std::string& path, bsm::SupertonicEncoder& enc) {
    std::vector<bt::Tensor*> W = gather_w(enc);
    std::vector<std::string> names = weight_names(enc);
    // Host copies must outlive write_file (it reads host_data lazily).
    std::vector<std::vector<float>> host(W.size());
    std::vector<bt::safetensors::WriteEntry> entries;
    entries.reserve(W.size());
    for (std::size_t i = 0; i < W.size(); ++i) {
        host[i] = W[i]->to_host_vector();
        bt::safetensors::WriteEntry e;
        e.name = names[i];
        e.dtype = bt::safetensors::Dtype::F32;
        e.shape = { W[i]->rows, W[i]->cols };
        e.host_data = host[i].data();
        e.bytes = host[i].size() * sizeof(float);
        entries.push_back(e);
    }
    bt::safetensors::write_file(path, entries);
}

// Pick the largest frame count T that is a multiple of CC and fits `L` samples
// at hop `hop` under torch.stft(center=True): T = 1 + L/hop. Returns 0 if too
// short. Also returns the input length to feed the spec so it yields exactly T.
int aligned_T(int L, int hop, int CC, int& L_used) {
    const int T_full = 1 + L / hop;
    const int T = (T_full / CC) * CC;
    if (T < CC) { L_used = 0; return 0; }
    L_used = (T - 1) * hop;
    return T;
}

// ── clip cache ──────────────────────────────────────────────────────────────
// Per clip, two stages are deterministic functions of the clip + config and so
// constant across every epoch: the encoder input spec [idim, T], and the target
// STFT magnitudes for the recon loss (one tensor per resolution). Caching both
// turns the ~31 ms (spec) + ~50 ms (target STFT) recompute into a one-time cost
// (build-on-demand: first epoch writes, later epochs read) — and a full hit lets
// the loop skip reading the wav entirely. fp32 on disk. Key = clip basename under
// --cache dir; the header carries the config so a stale cache is recomputed.
constexpr std::uint32_t kClipMagic = 0x53504332;  // "SPC2"

struct ClipCache {
    bt::Tensor              spec;   // [idim, T]
    std::vector<bt::Tensor> mags;   // target STFT magnitudes, one per resolution
    int T = 0, L_used = 0, n = 0;   // n = recon comparison length the mags match
};

std::string clip_cache_path(const std::string& cache_dir, const std::string& clip) {
    return (fs::path(cache_dir) / (fs::path(clip).stem().string() + ".clip")).string();
}

// Returns true and fills `c` if a valid cache for this config exists.
bool load_clip_cache(const std::string& path, bt::Device dev, int idim, int hop,
                     float max_secs, ClipCache& c) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::uint32_t magic = 0; int c_idim = 0, c_T = 0, c_L = 0, c_hop = 0; float c_secs = 0.0f;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&c_idim), sizeof(c_idim));
    f.read(reinterpret_cast<char*>(&c_T), sizeof(c_T));
    f.read(reinterpret_cast<char*>(&c_L), sizeof(c_L));
    f.read(reinterpret_cast<char*>(&c_hop), sizeof(c_hop));
    f.read(reinterpret_cast<char*>(&c_secs), sizeof(c_secs));
    if (!f || magic != kClipMagic || c_idim != idim || c_hop != hop ||
        c_secs != max_secs || c_T <= 0) return false;
    std::vector<float> spec(static_cast<std::size_t>(c_idim) * c_T);
    f.read(reinterpret_cast<char*>(spec.data()), spec.size() * sizeof(float));
    int n = 0, num_mags = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    f.read(reinterpret_cast<char*>(&num_mags), sizeof(num_mags));
    if (!f || num_mags < 0 || num_mags > 8) return false;
    std::vector<bt::Tensor> mags;
    mags.reserve(num_mags);
    for (int i = 0; i < num_mags; ++i) {
        int r = 0, cc = 0;
        f.read(reinterpret_cast<char*>(&r), sizeof(r));
        f.read(reinterpret_cast<char*>(&cc), sizeof(cc));
        if (!f || r <= 0 || cc <= 0) return false;
        std::vector<float> m(static_cast<std::size_t>(r) * cc);
        f.read(reinterpret_cast<char*>(m.data()), m.size() * sizeof(float));
        if (!f) return false;
        mags.push_back(bt::Tensor::from_host_on(dev, m.data(), r, cc));
    }
    c.spec = bt::Tensor::from_host_on(dev, spec.data(), c_idim, c_T);
    c.mags = std::move(mags);
    c.T = c_T; c.L_used = c_L; c.n = n;
    return true;
}

void save_clip_cache(const std::string& path, const std::vector<float>& spec, int idim,
                     int T, int L_used, int hop, float max_secs, int n,
                     const std::vector<bt::Tensor>& mags) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    f.write(reinterpret_cast<const char*>(&kClipMagic), sizeof(kClipMagic));
    f.write(reinterpret_cast<const char*>(&idim), sizeof(idim));
    f.write(reinterpret_cast<const char*>(&T), sizeof(T));
    f.write(reinterpret_cast<const char*>(&L_used), sizeof(L_used));
    f.write(reinterpret_cast<const char*>(&hop), sizeof(hop));
    f.write(reinterpret_cast<const char*>(&max_secs), sizeof(max_secs));
    f.write(reinterpret_cast<const char*>(spec.data()), spec.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    const int num_mags = static_cast<int>(mags.size());
    f.write(reinterpret_cast<const char*>(&num_mags), sizeof(num_mags));
    for (const auto& m : mags) {
        const std::vector<float> h = m.to_host_vector();
        f.write(reinterpret_cast<const char*>(&m.rows), sizeof(m.rows));
        f.write(reinterpret_cast<const char*>(&m.cols), sizeof(m.cols));
        f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    bt::init();
    const bt::Device dev = (!a.cpu && bt::is_available(bt::Device::CUDA))
                               ? bt::Device::CUDA : bt::Device::CPU;
    std::printf("device: %s\n", dev == bt::Device::CUDA ? "CUDA" : "CPU");

    bsm::Supertonic model;
    model.load(a.model, dev);
    if (!model.loaded()) { std::fprintf(stderr, "model load failed: %s\n", a.model.c_str()); return 1; }
    const int CC = model.config().chunk, SR = model.config().sample_rate;
    const int hop = 512;
    std::printf("model: %d kHz, chunk %d\n", SR / 1000, CC);

    bsm::SupertonicSpec spec(dev, SR, 2048, 2048, hop, 228);
    bsm::SupertonicEncoder enc;
    enc.init(dev, a.seed);
    // Start the encoder in the decoder's latent distribution: bias proj_out to the
    // per-channel latent mean and shrink its weights so z_real ≈ latent_mean (the
    // normalised flow latent ≈ 0, the vocoder's expected centre) at step 0. Random
    // init feeds the frozen decoder out-of-distribution latents whose backward gain
    // explodes the encoder gradient to NaN; clipping can't fix the bad direction.
    {
        const std::vector<float>& lm = model.latent_mean();
        bt::scale_inplace(enc.proj_out.w, a.init_scale);
        enc.proj_out.b = bt::Tensor::from_host_on(dev, lm.data(), enc.latent_dim, 1);
    }
    std::printf("encoder: idim %d hidden %d latent %d layers %d (%zu weight tensors), "
                "proj_out init_scale %.3g\n",
                enc.idim, enc.hidden, enc.latent_dim, enc.num_layers,
                gather_w(enc).size(), a.init_scale);

    // Scan wavs.
    std::vector<std::string> clips;
    for (const auto& e : fs::directory_iterator(a.data))
        if (e.is_regular_file() && e.path().extension() == ".wav")
            clips.push_back(e.path().string());
    std::sort(clips.begin(), clips.end());
    std::mt19937 rng(static_cast<std::uint32_t>(a.seed));
    std::shuffle(clips.begin(), clips.end(), rng);
    if (clips.empty()) { std::fprintf(stderr, "no .wav under %s\n", a.data.c_str()); return 1; }
    // Hold out a few clips for the reconstruction gate (skipped in overfit mode).
    std::vector<std::string> val;
    if (a.overfit > 0) {
        if (static_cast<int>(clips.size()) > a.overfit) clips.resize(a.overfit);
    } else {
        const int val_n = std::min<int>(4, std::max<int>(1,
            static_cast<int>(a.val_frac * clips.size())));
        val.assign(clips.begin(), clips.begin() + val_n);
        clips.erase(clips.begin(), clips.begin() + val_n);
        if (a.max_clips > 0 && static_cast<int>(clips.size()) > a.max_clips)
            clips.resize(a.max_clips);
    }
    std::printf("clips: %zu train, %zu val\n", clips.size(), val.size());
    if (!a.cache.empty()) {
        fs::create_directories(a.cache);
        std::printf("clip cache: %s (spec + target mags, build-on-demand)\n", a.cache.c_str());
    }

    // Adam state (m, v) + per-clip grads + a batch accumulator, all matching the
    // encoder weight layout. `grads` holds one clip's backward; `gacc` sums the
    // norm-clipped per-clip grads across an accumulation batch; Adam steps on the
    // averaged accumulator.
    bsm::SupertonicEncoderGrads grads, gacc, mst, vst;
    grads.zero(enc); gacc.zero(enc); mst.zero(enc); vst.zero(enc);
    std::vector<bt::Tensor*> W  = gather_w(enc);
    std::vector<bt::Tensor*> G  = gather_g(grads);
    std::vector<bt::Tensor*> GA = gather_g(gacc);
    std::vector<bt::Tensor*> M  = gather_g(mst);
    std::vector<bt::Tensor*> V  = gather_g(vst);
    const int max_samp = static_cast<int>(a.max_secs * SR);

    using Clock = std::chrono::steady_clock;
    auto secs = [](Clock::time_point x, Clock::time_point y) {
        return std::chrono::duration<double>(y - x).count();
    };
    auto tp = [&]() { if (a.profile) bt::sync(dev); return Clock::now(); };
    double pt[5] = {0, 0, 0, 0, 0};  // spec / fwd / recon / bwd / adam
    long pn = 0;

    // Global gradient L2-norm clip — backprop through the frozen vocoder from a
    // randomly-initialised encoder feeds out-of-distribution latents into the
    // decoder, whose backward gain runs the encoder grad away (-> NaN). Clipping
    // the per-step grad norm keeps the encoder in-distribution and convergent.
    // Returns false if the grad norm is non-finite (Inf/NaN) — the caller then
    // skips the Adam step so one bad clip can't poison every weight. Otherwise
    // clips the global norm to a.clip (if enabled).
    auto clip_grads = [&]() -> bool {
        double ss = 0.0;
        for (bt::Tensor* g : G) {
            bt::Tensor sq = g->clone();
            bt::mul_inplace(sq, *g);
            bt::Tensor rs, cs;
            bt::sum_rows(sq, rs);
            bt::sum_cols(rs, cs);
            ss += cs.to_host_vector()[0];
        }
        if (!std::isfinite(ss)) return false;
        const double gn = std::sqrt(ss);
        if (a.clip > 0.0f && gn > a.clip && gn > 0.0) {
            const float s = static_cast<float>(a.clip / gn);
            for (bt::Tensor* g : G) bt::scale_inplace(*g, s);
        }
        return true;
    };
    long skipped = 0;

    int step = 0;
    const int epochs = (a.overfit > 0) ? std::max(a.epochs, 40) : a.epochs;

    // lr schedule: linear warmup from ~0 to peak over `warmup` steps, then cosine
    // decay to lr_floor_frac·peak across the rest. Warmup avoids the cold-start
    // grad explosion; the decay tail removes the optimum-region bounce. A "step" is
    // one Adam update — one per accumulation batch, not per clip — so total steps is
    // epochs × ceil(clips/accum). Warmup is capped so it can't swallow a short run.
    const long batches_per_epoch =
        (static_cast<long>(clips.size()) + a.accum - 1) / a.accum;
    const long total_steps = static_cast<long>(epochs) * batches_per_epoch;
    const int warmup = std::min<int>(a.warmup,
                                     std::max<int>(1, static_cast<int>(total_steps / 10)));
    const float lr_floor = a.lr * a.lr_floor_frac;
    auto lr_at = [&](int s) -> float {                 // s = 1-based applied step
        if (s <= warmup) return a.lr * static_cast<float>(s) / static_cast<float>(warmup);
        const long denom = std::max<long>(1, total_steps - warmup);
        float t = static_cast<float>(s - warmup) / static_cast<float>(denom);
        if (t > 1.0f) t = 1.0f;
        const float c = 0.5f * (1.0f + std::cos(3.14159265358979f * t));
        return lr_floor + (a.lr - lr_floor) * c;
    };
    std::printf("lr schedule: warmup %d steps -> peak %.3g -> cosine floor %.3g "
                "over %ld steps (accum %d clips/step)\n",
                warmup, a.lr, lr_floor, total_steps, a.accum);
    long accn = 0;  // clips accumulated into the current batch
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(clips.begin(), clips.end(), rng);
        double loss_sum = 0.0; int seen = 0;
        for (const std::string& path : clips) {
            const std::string cpath = a.cache.empty() ? std::string()
                                                      : clip_cache_path(a.cache, path);
            const auto t0 = tp();
            // Constant-across-epochs prep: spec + target STFT magnitudes. A full
            // cache hit needs no wav read; a miss builds and (if caching) writes.
            ClipCache cc;
            bool have = !cpath.empty() &&
                        load_clip_cache(cpath, dev, enc.idim, hop, a.max_secs, cc);
            if (!have) {
                bsm::AudioBuffer wav;
                try { wav = bsm::read_wav(path); } catch (...) { continue; }
                if (wav.sample_rate != SR || wav.empty()) continue;
                int L = static_cast<int>(wav.samples.size());
                if (max_samp > 0 && L > max_samp) L = max_samp;
                int L_used = 0;
                const int T = aligned_T(L, hop, CC, L_used);
                if (T == 0) continue;
                cc.spec = spec.compute(wav.samples.data(), L_used);       // [1253, T]
                cc.T = T; cc.L_used = L_used;
                const int nwave = model.config().base_chunk * T;          // LF = T
                cc.n = std::min(static_cast<int>(wav.samples.size()), nwave);
                bt::Tensor target = bt::Tensor::from_host_on(dev, wav.samples.data(), 1, cc.n);
                cc.mags = model.recon_target_mags(target, cc.n);
                if (!cpath.empty())
                    save_clip_cache(cpath, cc.spec.to_host_vector(), enc.idim, cc.T,
                                    cc.L_used, hop, a.max_secs, cc.n, cc.mags);
            }
            bt::Tensor latent;
            bsm::SupertonicEncoderCache cache;
            const auto t1 = tp();
            enc.forward_train(cc.spec, cc.T, latent, cache);             // [24, T] device

            // Fully on-device recon: z_real / dLatent never leave the GPU; the
            // target STFT is replayed from the (cached) magnitudes.
            const auto t2 = tp();
            bt::Tensor dLatent;
            const float loss = model.recon_loss_and_grad(latent, /*LF=*/cc.T, cc.mags,
                                                          cc.n, dLatent);
            // The frozen-vocoder backward gain produces occasional Inf spikes in
            // dLatent (LayerNorm 1/std on low-variance frames); clamp to a finite
            // range so they don't become non-finite encoder grads (skipped steps).
            if (a.dlatent_clamp > 0.0f)
                bt::clamp(dLatent, -a.dlatent_clamp, a.dlatent_clamp);

            const auto t3 = tp();
            grads.zero(enc);
            enc.backward(cache, dLatent, grads);
            const auto t4 = tp();

            // Per-clip: norm-clip (skip if non-finite) so each clip contributes a
            // bounded gradient, then add into the batch accumulator. Adam steps only
            // once the batch is full — averaging over `accum` clips gives a stable
            // descent direction that batch-1 SGD can't on this objective.
            const bool grad_ok = clip_grads();
            if (grad_ok) {
                for (std::size_t i = 0; i < GA.size(); ++i) bt::add_inplace(*GA[i], *G[i]);
                ++accn;
            } else {
                ++skipped;  // non-finite grad: drop this clip from the batch
            }
            if (accn >= a.accum) {
                for (bt::Tensor* g : GA) bt::scale_inplace(*g, 1.0f / static_cast<float>(accn));
                ++step;
                const float lr = lr_at(step);
                for (std::size_t i = 0; i < W.size(); ++i)
                    bt::adam_step(*W[i], *GA[i], *M[i], *V[i], lr, 0.9f, 0.999f, 1.0e-8f, step);
                gacc.zero(enc);
                accn = 0;
            }
            const auto t5 = tp();
            if (a.profile) {
                pt[0] += secs(t0, t1); pt[1] += secs(t1, t2); pt[2] += secs(t2, t3);
                pt[3] += secs(t3, t4); pt[4] += secs(t4, t5); ++pn;
            }

            loss_sum += loss; ++seen;
            if (seen % 200 == 0)
                std::printf("  epoch %d  %d/%zu  loss %.5f\n", epoch, seen, clips.size(),
                            loss_sum / seen);
        }
        // Flush a partial final batch so its clips aren't dropped.
        if (accn > 0) {
            for (bt::Tensor* g : GA) bt::scale_inplace(*g, 1.0f / static_cast<float>(accn));
            ++step;
            const float lr = lr_at(step);
            for (std::size_t i = 0; i < W.size(); ++i)
                bt::adam_step(*W[i], *GA[i], *M[i], *V[i], lr, 0.9f, 0.999f, 1.0e-8f, step);
            gacc.zero(enc);
            accn = 0;
        }
        std::printf("epoch %d: mean loss %.6f over %d clips (%ld non-finite grads skipped)\n",
                    epoch, seen ? loss_sum / seen : 0.0, seen, skipped);
        if (a.overfit == 0 && a.save_every > 0 && (epoch + 1) % a.save_every == 0) {
            save_encoder(a.out, enc);
            std::printf("  saved %s\n", a.out.c_str());
        }
    }

    save_encoder(a.out, enc);
    std::printf("done. saved %s\n", a.out.c_str());

    if (a.profile && pn > 0) {
        const double tot = pt[0] + pt[1] + pt[2] + pt[3] + pt[4];
        std::printf("profile over %ld clips (ms/clip): prep %.1f  fwd %.1f  recon %.1f"
                    "  bwd %.1f  adam %.1f  | total %.1f (%.1f clips/s)\n", pn,
                    1e3 * pt[0] / pn, 1e3 * pt[1] / pn, 1e3 * pt[2] / pn,
                    1e3 * pt[3] / pn, 1e3 * pt[4] / pn, 1e3 * tot / pn, pn / tot);
    }

    // Reconstruction gate: encode + frozen-decode each held-out clip, report the
    // recon loss, and write input/recon wavs beside the checkpoint for listening.
    const fs::path outdir = fs::path(a.out).parent_path();
    for (std::size_t vi = 0; vi < val.size(); ++vi) {
        bsm::AudioBuffer wav;
        try { wav = bsm::read_wav(val[vi]); } catch (...) { continue; }
        if (wav.sample_rate != SR || wav.empty()) continue;
        int L = static_cast<int>(wav.samples.size());
        if (max_samp > 0 && L > max_samp) L = max_samp;
        int L_used = 0;
        const int T = aligned_T(L, hop, CC, L_used);
        if (T == 0) continue;

        bt::Tensor feat = spec.compute(wav.samples.data(), L_used);
        bt::Tensor latent = enc.forward(feat, T);
        const std::vector<float> z_real = latent.to_host_vector();
        const int nwave = model.config().base_chunk * T;
        std::vector<float> dZ;
        const float loss = model.recon_loss_and_grad(
            z_real.data(), enc.latent_dim, T, wav.samples.data(),
            std::min(static_cast<int>(wav.samples.size()), nwave), dZ);

        bsm::AudioBuffer recon = model.decode_real(z_real.data(), enc.latent_dim, T);
        bsm::AudioBuffer in_clip; in_clip.sample_rate = SR;
        in_clip.samples.assign(wav.samples.begin(),
                               wav.samples.begin() + std::min<int>(L_used, nwave));
        const std::string tag = "val" + std::to_string(vi);
        in_clip.write_wav((outdir / (tag + "_input.wav")).string());
        recon.write_wav((outdir / (tag + "_recon.wav")).string());
        std::printf("  recon %s: loss %.5f  (%s_input.wav / %s_recon.wav)\n",
                    fs::path(val[vi]).filename().string().c_str(), loss,
                    tag.c_str(), tag.c_str());
    }
    return 0;
}
