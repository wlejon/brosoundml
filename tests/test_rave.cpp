// RAVE v2 encode/decode numerical parity against the torch reference
// (scripts/rave_reference.py). This is a real-weights test: it needs a converted
// RAVE model plus the dumped fixtures, located via the BROSOUNDML_RAVE_DIR
// environment variable. When that is unset or incomplete the test SKIPs (exit 0)
// so the default `ctest` run stays self-contained.
//
// Prepare the fixtures once (see scripts/convert-rave.py + rave_reference.py):
//   python scripts/convert-rave.py model.ts  $DIR
//   python scripts/rave_reference.py --dump  $DIR  $DIR
//   BROSOUNDML_RAVE_DIR=$DIR  ./brosoundml_test_rave

#include "brosoundml/rave.h"

#include <brotensor/runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::fprintf(stderr, "FAIL: %s\n", (msg));    \
            ++failures;                                   \
        }                                                 \
    } while (0)

static std::vector<float> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<float> out(static_cast<std::size_t>(n) / sizeof(float));
    f.read(reinterpret_cast<char*>(out.data()), n);
    return out;
}

// Max abs error and a coarse relative scale, for reporting.
static float max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: keep progress on a crash
    const char* dir_env = std::getenv("BROSOUNDML_RAVE_DIR");
    if (!dir_env || !*dir_env) {
        std::printf("SKIP: BROSOUNDML_RAVE_DIR not set (real-weights RAVE parity test)\n");
        return 0;
    }
    const std::string dir = dir_env;

    const std::vector<float> input     = read_bin(dir + "/input.bin");
    const std::vector<float> ref_latent = read_bin(dir + "/latent.bin");
    const std::vector<float> ref_decode = read_bin(dir + "/decode_det.bin");
    std::ifstream model_check(dir + "/model.safetensors", std::ios::binary);
    if (input.empty() || ref_latent.empty() || ref_decode.empty() || !model_check) {
        std::printf("SKIP: model/fixtures missing under '%s' "
                    "(need model.safetensors, input.bin, latent.bin, decode_det.bin)\n",
                    dir.c_str());
        return 0;
    }

    brotensor::init();
    const brotensor::Device dev =
        brotensor::is_available(brotensor::Device::CUDA) ? brotensor::Device::CUDA
                                                         : brotensor::Device::CPU;
    std::printf("RAVE parity on device: %s\n", brotensor::device_name(dev));

    brosoundml::Rave rave;
    rave.load(dir, dev);
    CHECK(rave.loaded(), "model loaded");

    const auto& cfg = rave.config();
    const int nl = cfg.cropped_latent_size;
    const int frames = static_cast<int>(ref_latent.size()) / nl;
    std::printf("config: sr=%d full=%d nl=%d n_band=%d ratio=%d | frames=%d\n",
                cfg.sampling_rate, cfg.full_latent_size, nl, cfg.n_band,
                cfg.total_ratio, frames);

    // ── encode parity ──
    brosoundml::RaveLatent z = rave.encode(input);
    CHECK(z.n_latent == nl, "encode n_latent");
    CHECK(z.frames == frames, "encode frames");
    CHECK(z.data.size() == ref_latent.size(), "encode latent size");
    const float enc_err = max_abs_err(z.data, ref_latent);
    std::printf("encode  max-abs-err = %.3e\n", enc_err);
    CHECK(enc_err < 2.0e-3f, "encode latent matches reference");

    // ── decode parity (decode the REFERENCE latent to isolate the decoder) ──
    brosoundml::AudioBuffer y = rave.decode(ref_latent.data(), nl, frames);
    CHECK(static_cast<int>(y.samples.size()) == frames * cfg.total_ratio, "decode length");
    CHECK(y.sample_rate == cfg.sampling_rate, "decode sample rate");
    const float dec_err = max_abs_err(y.samples, ref_decode);
    std::printf("decode  max-abs-err = %.3e  (out rms=%.4f)\n", dec_err, y.rms());
    CHECK(dec_err < 2.0e-3f, "decode waveform matches reference");

    // ── noise-branch parity (inject the reference's white noise, compare the
    //    stochastic synth bit-close) ── optional: only when the noise fixtures
    //    are present (older fixture dumps lack them).
    const std::vector<float> ref_noise        = read_bin(dir + "/noise.bin");
    const std::vector<float> ref_decode_noise = read_bin(dir + "/decode_noise.bin");
    if (!ref_noise.empty() && !ref_decode_noise.empty()) {
        brosoundml::RaveDecodeOptions opts;
        opts.add_noise = true;
        opts.noise     = ref_noise.data();
        opts.noise_len = static_cast<int>(ref_noise.size());
        brosoundml::AudioBuffer yn = rave.decode(ref_latent.data(), nl, frames, opts);
        CHECK(yn.samples.size() == ref_decode_noise.size(), "noise decode length");
        const float noise_err = max_abs_err(yn.samples, ref_decode_noise);
        std::printf("noise   max-abs-err = %.3e  (out rms=%.4f)\n", noise_err, yn.rms());
        CHECK(noise_err < 2.0e-3f, "noise-branch waveform matches reference");
        // Sanity: noise actually changed the output vs the deterministic decode.
        CHECK(max_abs_err(yn.samples, ref_decode) > 1.0e-4f, "noise alters the output");
    } else {
        std::printf("note: noise fixtures absent — skipping noise-branch parity\n");
    }

    // ── stereo parity (decode_multi) ── optional: only when the stereo fixtures
    //    are present. Inject the reference's two per-channel latent pads and
    //    confirm the interleaved 2-channel decode matches the stacked reference
    //    bit-close, and that the channels actually differ.
    const std::vector<float> ref_pad    = read_bin(dir + "/latent_pad.bin");
    const std::vector<float> ref_stereo = read_bin(dir + "/decode_stereo.bin");
    if (!ref_pad.empty() && !ref_stereo.empty()) {
        const int ch  = 2;
        const int per = frames * cfg.total_ratio;          // samples per channel
        brosoundml::RaveDecodeOptions opts;
        opts.channels       = ch;
        opts.latent_pad     = ref_pad.data();
        opts.latent_pad_len = static_cast<int>(ref_pad.size());
        brosoundml::RaveMultiBuffer ys = rave.decode_multi(ref_latent.data(), nl, frames, opts);
        CHECK(ys.channels == ch, "stereo channel count");
        CHECK(static_cast<int>(ys.samples.size()) == per * ch, "stereo interleaved length");
        CHECK(ref_stereo.size() == static_cast<std::size_t>(per) * ch, "stereo fixture length");
        // De-interleave our output and compare to the (channel-major) reference.
        float st_err = 0.0f, lr_l1 = 0.0f;
        for (int c = 0; c < ch; ++c)
            for (int t = 0; t < per; ++t) {
                const float ours = ys.samples[static_cast<std::size_t>(t) * ch + c];
                const float ref  = ref_stereo[static_cast<std::size_t>(c) * per + t];
                st_err = std::max(st_err, std::fabs(ours - ref));
            }
        for (int t = 0; t < per; ++t)
            lr_l1 += std::fabs(ys.samples[static_cast<std::size_t>(t) * ch + 0] -
                               ys.samples[static_cast<std::size_t>(t) * ch + 1]);
        lr_l1 /= per;
        std::printf("stereo  max-abs-err = %.3e  (L/R l1=%.4f)\n", st_err, lr_l1);
        CHECK(st_err < 2.0e-3f, "stereo decode matches reference");
        CHECK(lr_l1 > 1.0e-4f, "stereo channels are decorrelated (L != R)");
    } else {
        std::printf("note: stereo fixtures absent — skipping stereo parity\n");
    }

    // ── full round trip sanity (our latent -> decode) ──
    brosoundml::AudioBuffer rt = rave.decode(z);
    CHECK(!rt.empty(), "round-trip non-empty");

    if (failures == 0) std::printf("test_rave: PASS\n");
    return failures == 0 ? 0 : 1;
}
