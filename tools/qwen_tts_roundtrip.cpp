// Codec round-trip for Qwen3-TTS: encode a reference clip into the bundled
// 12 Hz RVQ code stream and decode it straight back to 24 kHz, writing the
// reconstruction. This exercises the codec *analysis* path (encode_audio) on
// arbitrary real audio — the inverse of decode_codes and the acoustic
// conditioning a zero-shot voice clone enrolls from — end to end, not just
// against the upstream fixture.
//
//   qwen_tts_roundtrip <input.wav> [output.wav] [model_dir]
//
// Prints the analysis stats (frames, codes/frame, code rate) and, when the
// reconstruction lines up sample-for-sample after the codec's 1920-sample
// framing, an error metric against the (resampled) input.

#define _CRT_SECURE_NO_WARNINGS

#include "brosoundml/audio.h"
#include "brosoundml/qwen_tts.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using brosoundml::AudioBuffer;
using brosoundml::QwenTts;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: qwen_tts_roundtrip <input.wav> [output.wav] [model_dir]\n");
        return 2;
    }
    const std::string in_path  = argv[1];
    const std::string out_path = (argc > 2) ? argv[2] : "roundtrip.wav";
    const std::string root     = (argc > 3)
        ? argv[3] : std::string("weights/qwen-tts/0.6B-customvoice");

    brotensor::init();
    const bool cuda = brotensor::is_available(brotensor::Device::CUDA);
    const brotensor::Device dev =
        cuda ? brotensor::Device::CUDA : brotensor::Device::CPU;

    QwenTts q;
    q.load(root, dev);
    std::printf("loaded %s on %s\n", root.c_str(), cuda ? "CUDA" : "CPU");

    AudioBuffer ref = brosoundml::read_wav(in_path);
    std::printf("input: %s  %.2fs  @ %d Hz  (%zu samples)\n",
                in_path.c_str(), ref.duration_seconds(), ref.sample_rate,
                ref.samples.size());

    // ── encode: waveform -> RVQ codes ──
    int num_frames = 0;
    auto t0 = clk::now();
    std::vector<int32_t> codes = q.encode_audio(ref, &num_frames);
    brotensor::sync_all();
    const double enc_ms = ms_since(t0);
    const int K = q.config().codec.encoder.valid_num_quantizers;
    const double audio_s = num_frames / 12.5;
    std::printf("encode: %d frames x %d codes = %zu codes  (%.1f ms, %.1fx RT)\n",
                num_frames, K, codes.size(), enc_ms,
                audio_s / (enc_ms / 1000.0));

    // ── decode: codes -> waveform (same codebook-major layout) ──
    t0 = clk::now();
    AudioBuffer recon = q.decode_codes(codes, K, num_frames);
    brotensor::sync_all();
    const double dec_ms = ms_since(t0);
    std::printf("decode: %zu samples  @ %d Hz  (%.1f ms, %.1fx RT)\n",
                recon.samples.size(), recon.sample_rate, dec_ms,
                recon.duration_seconds() / (dec_ms / 1000.0));

    // ── error vs. the (24 kHz-resampled) input, over the common length ──
    // encode right-pads the input to a whole 1920-sample frame, so the
    // reconstruction can be a fraction of a frame longer than the source.
    if (recon.sample_rate == ref.sample_rate) {
        const std::size_t n = std::min(ref.samples.size(), recon.samples.size());
        double se = 0.0, sref = 0.0;
        float peak = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(recon.samples[i]) - ref.samples[i];
            se += d * d;
            sref += static_cast<double>(ref.samples[i]) * ref.samples[i];
            peak = std::max(peak, std::fabs(recon.samples[i] - ref.samples[i]));
        }
        const double rmse = (n > 0) ? std::sqrt(se / static_cast<double>(n)) : 0.0;
        const double snr  = (se > 0.0) ? 10.0 * std::log10(sref / se) : 0.0;
        std::printf("recon vs input (%zu samples): rmse=%.4g  peak|d|=%.4g  snr=%.1f dB\n",
                    n, rmse, peak, snr);
    } else {
        std::printf("recon at %d Hz vs input at %d Hz - skipping error metric\n",
                    recon.sample_rate, ref.sample_rate);
    }

    recon.write_wav(out_path);
    std::printf("wrote %s\n", out_path.c_str());
    return 0;
}
