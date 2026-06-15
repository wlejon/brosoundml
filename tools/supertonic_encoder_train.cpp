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
#include <cstdio>
#include <filesystem>
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
    int    epochs    = 4;
    float  lr        = 3.0e-4f;
    int    max_clips = 0;        // 0 = all
    float  max_secs  = 4.0f;     // trim each clip to bound compute
    float  val_frac  = 0.02f;
    int    overfit   = 0;        // >0: overfit N clips (smoke), ignores epochs/val
    int    save_every = 1;
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
        else if (k == "--epochs")     a.epochs = std::stoi(need(i));
        else if (k == "--lr")         a.lr = std::stof(need(i));
        else if (k == "--max-clips")  a.max_clips = std::stoi(need(i));
        else if (k == "--max-secs")   a.max_secs = std::stof(need(i));
        else if (k == "--val-frac")   a.val_frac = std::stof(need(i));
        else if (k == "--overfit")    a.overfit = std::stoi(need(i));
        else if (k == "--save-every") a.save_every = std::stoi(need(i));
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
    std::printf("encoder: idim %d hidden %d latent %d layers %d (%zu weight tensors)\n",
                enc.idim, enc.hidden, enc.latent_dim, enc.num_layers, gather_w(enc).size());

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

    // Adam state (m, v) + per-step grads, all matching the encoder weight layout.
    bsm::SupertonicEncoderGrads grads, mst, vst;
    grads.zero(enc); mst.zero(enc); vst.zero(enc);
    std::vector<bt::Tensor*> W = gather_w(enc);
    std::vector<bt::Tensor*> G = gather_g(grads);
    std::vector<bt::Tensor*> M = gather_g(mst);
    std::vector<bt::Tensor*> V = gather_g(vst);
    const int max_samp = static_cast<int>(a.max_secs * SR);

    using Clock = std::chrono::steady_clock;
    auto secs = [](Clock::time_point x, Clock::time_point y) {
        return std::chrono::duration<double>(y - x).count();
    };
    auto tp = [&]() { if (a.profile) bt::sync(dev); return Clock::now(); };
    double pt[5] = {0, 0, 0, 0, 0};  // spec / fwd / recon / bwd / adam
    long pn = 0;

    int step = 0;
    const int epochs = (a.overfit > 0) ? std::max(a.epochs, 40) : a.epochs;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(clips.begin(), clips.end(), rng);
        double loss_sum = 0.0; int seen = 0;
        for (const std::string& path : clips) {
            bsm::AudioBuffer wav;
            try { wav = bsm::read_wav(path); } catch (...) { continue; }
            if (wav.sample_rate != SR || wav.empty()) continue;
            int L = static_cast<int>(wav.samples.size());
            if (max_samp > 0 && L > max_samp) L = max_samp;
            int L_used = 0;
            const int T = aligned_T(L, hop, CC, L_used);
            if (T == 0) continue;

            const auto t0 = tp();
            bt::Tensor feat = spec.compute(wav.samples.data(), L_used);   // [1253, T]
            bt::Tensor latent;
            bsm::SupertonicEncoderCache cache;
            const auto t1 = tp();
            enc.forward_train(feat, T, latent, cache);                   // [24, T] device

            // Fully on-device recon: z_real / dLatent never leave the GPU.
            const int nwave = model.config().base_chunk * T;  // base_chunk * LF, LF = T
            const int target_len = std::min(static_cast<int>(wav.samples.size()), nwave);
            bt::Tensor target = bt::Tensor::from_host_on(dev, wav.samples.data(), 1, target_len);
            const auto t2 = tp();
            bt::Tensor dLatent;
            const float loss = model.recon_loss_and_grad(latent, /*LF=*/T, target, dLatent);

            const auto t3 = tp();
            grads.zero(enc);
            enc.backward(cache, dLatent, grads);
            const auto t4 = tp();

            ++step;
            for (std::size_t i = 0; i < W.size(); ++i)
                bt::adam_step(*W[i], *G[i], *M[i], *V[i], a.lr, 0.9f, 0.999f, 1.0e-8f, step);
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
        std::printf("epoch %d: mean loss %.6f over %d clips\n", epoch,
                    seen ? loss_sum / seen : 0.0, seen);
        if (a.overfit == 0 && a.save_every > 0 && (epoch + 1) % a.save_every == 0) {
            save_encoder(a.out, enc);
            std::printf("  saved %s\n", a.out.c_str());
        }
    }

    save_encoder(a.out, enc);
    std::printf("done. saved %s\n", a.out.c_str());

    if (a.profile && pn > 0) {
        const double tot = pt[0] + pt[1] + pt[2] + pt[3] + pt[4];
        std::printf("profile over %ld clips (ms/clip): spec %.1f  fwd %.1f  recon %.1f"
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
