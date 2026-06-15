// Supertonic-3 vocoder parity: run the brotensor-composed vocoder on a fixed
// latent and check it matches the upstream ONNX reference (dumped by the
// out-of-repo supertonic-convert tool's --parity mode).
//
// Model + reference live under brosoundml-data/supertonic (env
// BROSOUNDML_SUPERTONIC_DIR overrides). The test SKIPs (exit 0) when the
// directory or its weights are absent, so a clean checkout without the data
// repo still passes.

#include "brosoundml/supertonic.h"

#include <brotensor/runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<float> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<float> out(static_cast<std::size_t>(n) / sizeof(float));
    f.read(reinterpret_cast<char*>(out.data()), n);
    return out;
}

static float max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: survive a crash
    brotensor::init();  // probe + register GPU backends (CPU is always present)
    std::string dir;
    if (const char* e = std::getenv("BROSOUNDML_SUPERTONIC_DIR"); e && *e) {
        dir = e;
    } else {
#ifdef BROSOUNDML_REPO_DIR
        dir = std::string(BROSOUNDML_REPO_DIR) + "/../brosoundml-data/supertonic";
#endif
    }
    if (dir.empty() || !fs::exists(fs::path(dir) / "vocoder.safetensors")) {
        std::printf("SKIP: supertonic model not found (set BROSOUNDML_SUPERTONIC_DIR)\n");
        return 0;
    }

    const std::string pdir = dir + "/parity";
    const std::vector<float> latent = read_bin(pdir + "/vocoder.in.latent.bin");
    const std::vector<float> ref    = read_bin(pdir + "/vocoder.out.wav_tts.bin");
    if (latent.empty() || ref.empty()) {
        std::printf("SKIP: parity bins missing under %s\n", pdir.c_str());
        return 0;
    }

    // The reference was generated with input latent [1, 144, 32].
    const int channels = 144;
    const int frames   = static_cast<int>(latent.size()) / channels;  // 32
    if (frames * channels != static_cast<int>(latent.size())) {
        std::printf("FAIL: latent size %zu not divisible by %d channels\n",
                    latent.size(), channels);
        return 1;
    }

    // Run on CUDA when the backend is present (the flow loop is heavy on CPU);
    // parity tolerances below hold on either backend (device-neutral bit-faithful).
    const brotensor::Device dev = brotensor::is_available(brotensor::Device::CUDA)
                                      ? brotensor::Device::CUDA
                                      : brotensor::Device::CPU;
    std::printf("device: %s\n", dev == brotensor::Device::CUDA ? "CUDA" : "CPU");

    brosoundml::Supertonic model;
    try {
        model.load(dir, dev);
    } catch (const std::exception& e) {
        std::printf("FAIL: load threw: %s\n", e.what());
        return 1;
    }
    const brosoundml::AudioBuffer wav = model.decode(latent, channels, frames);

    std::printf("vocoder: in[%d x %d] -> %zu samples (ref %zu), sr=%d\n",
                channels, frames, wav.samples.size(), ref.size(), wav.sample_rate);

    if (wav.samples.size() != ref.size()) {
        std::printf("FAIL: output length %zu != reference %zu\n",
                    wav.samples.size(), ref.size());
        return 1;
    }

    const float err = max_abs_err(wav.samples, ref);
    std::printf("vocoder max-abs-err vs ONNX reference: %.3e\n", err);

    // Bit-faithful bar: brotensor vs onnxruntime FP32 differ only in accumulation
    // order (~1e-6 observed). 1e-4 leaves margin while still catching real drift.
    const float tol = 1.0e-4f;
    if (err >= tol) {
        std::printf("FAIL: vocoder parity error %.3e >= tol %.3e\n", err, tol);
        return 1;
    }
    std::printf("PASS: vocoder matches ONNX reference within %.1e\n", tol);

    // ── text encoder parity (text_ids + style_ttl -> text_emb [256 x T]) ──
    if (!fs::exists(fs::path(dir) / "text_encoder.safetensors")) {
        std::printf("SKIP: text_encoder weights absent\n");
        return 0;
    }
    const std::vector<float> tid_f = read_bin(pdir + "/text_encoder.in.text_ids.bin");
    const std::vector<float> styl  = read_bin(pdir + "/text_encoder.in.style_ttl.bin");
    const std::vector<float> te_ref = read_bin(pdir + "/text_encoder.out.text_emb.bin");
    if (tid_f.empty() || styl.empty() || te_ref.empty()) {
        std::printf("SKIP: text_encoder parity bins missing\n");
        return 0;
    }
    std::vector<int> tids(tid_f.size());
    for (std::size_t i = 0; i < tid_f.size(); ++i)
        tids[i] = static_cast<int>(tid_f[i] + 0.5f);

    std::vector<float> te_emb;
    try {
        te_emb = model.encode_text(tids, styl);
    } catch (const std::exception& e) {
        std::printf("FAIL: encode_text threw: %s\n", e.what());
        return 1;
    }
    std::printf("text_encoder: T=%zu -> %zu floats (ref %zu)\n",
                tids.size(), te_emb.size(), te_ref.size());
    if (te_emb.size() != te_ref.size()) {
        std::printf("FAIL: text_emb size %zu != reference %zu\n",
                    te_emb.size(), te_ref.size());
        return 1;
    }
    const float te_err = max_abs_err(te_emb, te_ref);
    std::printf("text_encoder max-abs-err vs ONNX reference: %.3e\n", te_err);
    // Deeper graph (4 attn layers + 2 cross-attn) accumulates more FP32 reorder
    // drift than the vocoder; 2e-3 is still tight enough to catch real bugs.
    const float te_tol = 2.0e-3f;
    if (te_err >= te_tol) {
        std::printf("FAIL: text_encoder parity error %.3e >= tol %.3e\n",
                    te_err, te_tol);
        return 1;
    }
    std::printf("PASS: text_encoder matches ONNX reference within %.1e\n", te_tol);

    // ── duration predictor (text_ids + style_dp -> scalar total duration) ──
    if (!fs::exists(fs::path(dir) / "duration_predictor.safetensors")) {
        std::printf("SKIP: duration_predictor weights absent\n");
        return 0;
    }
    const std::vector<float> dp_tid = read_bin(pdir + "/duration_predictor.in.text_ids.bin");
    const std::vector<float> dp_sty = read_bin(pdir + "/duration_predictor.in.style_dp.bin");
    const std::vector<float> dp_ref = read_bin(pdir + "/duration_predictor.out.duration.bin");
    if (dp_tid.empty() || dp_sty.empty() || dp_ref.empty()) {
        std::printf("SKIP: duration_predictor parity bins missing\n");
        return 0;
    }
    std::vector<int> dp_ids(dp_tid.size());
    for (std::size_t i = 0; i < dp_tid.size(); ++i)
        dp_ids[i] = static_cast<int>(dp_tid[i] + 0.5f);

    float dur = 0.0f;
    try {
        dur = model.predict_duration(dp_ids, dp_sty);
    } catch (const std::exception& e) {
        std::printf("FAIL: predict_duration threw: %s\n", e.what());
        return 1;
    }
    const float dp_err = std::fabs(dur - dp_ref[0]);
    std::printf("duration_predictor: T=%zu -> %.6f (ref %.6f), abs-err %.3e\n",
                dp_ids.size(), dur, dp_ref[0], dp_err);
    const float dp_tol = 1.0e-3f;
    if (dp_err >= dp_tol) {
        std::printf("FAIL: duration_predictor parity error %.3e >= tol %.3e\n",
                    dp_err, dp_tol);
        return 1;
    }
    std::printf("PASS: duration_predictor matches ONNX reference within %.1e\n", dp_tol);

    // ── vector estimator (one guided flow-matching denoising step) ──
    if (!fs::exists(fs::path(dir) / "vector_estimator.safetensors")) {
        std::printf("SKIP: vector_estimator weights absent\n");
        return 0;
    }
    const std::vector<float> ve_noisy = read_bin(pdir + "/vector_estimator.in.noisy_latent.bin");
    const std::vector<float> ve_text  = read_bin(pdir + "/vector_estimator.in.text_emb.bin");
    const std::vector<float> ve_style = read_bin(pdir + "/vector_estimator.in.style_ttl.bin");
    const std::vector<float> ve_cur   = read_bin(pdir + "/vector_estimator.in.current_step.bin");
    const std::vector<float> ve_tot   = read_bin(pdir + "/vector_estimator.in.total_step.bin");
    const std::vector<float> ve_ref   = read_bin(pdir + "/vector_estimator.out.denoised_latent.bin");
    if (ve_noisy.empty() || ve_text.empty() || ve_style.empty() || ve_ref.empty()) {
        std::printf("SKIP: vector_estimator parity bins missing\n");
        return 0;
    }
    const int ve_ch = 144;
    const int ve_L  = static_cast<int>(ve_noisy.size()) / ve_ch;           // 32
    const int ve_T  = static_cast<int>(ve_text.size()) / 256;              // 24
    const int ve_cs = ve_cur.empty() ? 0 : static_cast<int>(ve_cur[0] + 0.5f);
    const int ve_ts = ve_tot.empty() ? 8 : static_cast<int>(ve_tot[0] + 0.5f);

    std::vector<float> ve_out;
    try {
        ve_out = model.denoise(ve_noisy, ve_ch, ve_L, ve_text, ve_T, ve_style, ve_cs, ve_ts);
    } catch (const std::exception& e) {
        std::printf("FAIL: denoise threw: %s\n", e.what());
        return 1;
    }
    std::printf("vector_estimator: 144x%d, T=%d, step %d/%d -> %zu floats (ref %zu)\n",
                ve_L, ve_T, ve_cs, ve_ts, ve_out.size(), ve_ref.size());
    if (ve_out.size() != ve_ref.size()) {
        std::printf("FAIL: denoised size %zu != reference %zu\n", ve_out.size(), ve_ref.size());
        return 1;
    }
    const float ve_err = max_abs_err(ve_out, ve_ref);
    std::printf("vector_estimator max-abs-err vs ONNX reference: %.3e\n", ve_err);
    // 64M-param DiT with 8 attention blocks + CFG; deepest graph here, so it
    // accumulates the most FP32 reorder drift. 5e-3 stays tight on a unit-scale
    // latent while tolerating the reorder.
    const float ve_tol = 5.0e-3f;
    if (ve_err >= ve_tol) {
        std::printf("FAIL: vector_estimator parity error %.3e >= tol %.3e\n", ve_err, ve_tol);
        return 1;
    }
    std::printf("PASS: vector_estimator matches ONNX reference within %.1e\n", ve_tol);

    // ── end-to-end synthesis (UnicodeProcessor frontend + flow loop + vocoder) ──
    const fs::path m1 = fs::path(dir) / "voice_styles" / "M1.json";
    if (!fs::exists(m1)) {
        std::printf("SKIP: voice_styles/M1.json absent\n");
        return 0;
    }
    brosoundml::VoiceStyle voice;
    try {
        voice = model.load_voice_style(m1.string());
    } catch (const std::exception& e) {
        std::printf("FAIL: load_voice_style threw: %s\n", e.what());
        return 1;
    }

    // Frontend: a known string -> non-empty ids, all in [0, vocab); the lang tag
    // wrap means the id stream is strictly longer than the bare text.
    const std::string sample = "Hello, world.";
    std::vector<int> ids;
    try {
        ids = model.text_to_ids(sample, "en");
    } catch (const std::exception& e) {
        std::printf("FAIL: text_to_ids threw: %s\n", e.what());
        return 1;
    }
    std::printf("text_to_ids(\"%s\", en) -> %zu ids\n", sample.c_str(), ids.size());
    if (ids.size() < sample.size()) {  // "<en>" + text + "</en>" is longer
        std::printf("FAIL: id stream %zu shorter than text %zu\n",
                    ids.size(), sample.size());
        return 1;
    }
    for (int id : ids)
        if (id < 0) { std::printf("FAIL: negative id %d leaked\n", id); return 1; }

    const int total_step = 8;
    brosoundml::AudioBuffer syn, syn2;
    try {
        syn  = model.synthesize(sample, "en", voice, total_step, 1.05f, /*seed=*/7);
        syn2 = model.synthesize(sample, "en", voice, total_step, 1.05f, /*seed=*/7);
    } catch (const std::exception& e) {
        std::printf("FAIL: synthesize threw: %s\n", e.what());
        return 1;
    }
    std::printf("synthesize: %zu samples (%.2fs) @ %d Hz\n", syn.samples.size(),
                syn.samples.size() / static_cast<double>(syn.sample_rate),
                syn.sample_rate);

    if (syn.sample_rate != model.config().sample_rate) {
        std::printf("FAIL: sample_rate %d != config %d\n",
                    syn.sample_rate, model.config().sample_rate);
        return 1;
    }
    // Output length is a whole number of de-chunked frames (chunk*base_chunk).
    const std::size_t frame_samples =
        static_cast<std::size_t>(model.config().chunk) * model.config().base_chunk;
    if (syn.samples.empty() || syn.samples.size() % frame_samples != 0) {
        std::printf("FAIL: %zu samples not a multiple of %zu\n",
                    syn.samples.size(), frame_samples);
        return 1;
    }
    // Finite and not pure silence.
    float peak = 0.0f;
    for (float s : syn.samples) {
        if (!std::isfinite(s)) { std::printf("FAIL: non-finite sample\n"); return 1; }
        peak = std::max(peak, std::fabs(s));
    }
    if (peak < 1.0e-4f) { std::printf("FAIL: output is silent (peak %.2e)\n", peak); return 1; }
    // Same seed -> bit-identical (deterministic, no hidden global RNG state).
    if (syn2.samples.size() != syn.samples.size() ||
        max_abs_err(syn.samples, syn2.samples) != 0.0f) {
        std::printf("FAIL: synthesis not deterministic for a fixed seed\n");
        return 1;
    }
    std::printf("PASS: synthesize produced %.2fs of finite, non-silent, "
                "deterministic audio (peak %.3f)\n",
                syn.samples.size() / static_cast<double>(syn.sample_rate), peak);
    return 0;
}
