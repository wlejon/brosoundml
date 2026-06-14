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

    brosoundml::Supertonic model;
    model.load(dir, brotensor::Device::CPU);
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
    return 0;
}
