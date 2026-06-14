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
    return 0;
}
