#pragma once

// laya-audio: training-time audio augmentation, applied to a whole utterance
// (16 kHz) before its windows are encoded, so every window of it shares the
// same room and background:
//
//   reverb   convolve with a room impulse response (FFT), aligned on the
//            RIR's direct-path peak so word times do not move, rescaled to
//            the dry signal's RMS;
//   noise    a random stretch of a noise recording at an SNR drawn from
//            [noise_snr_lo, noise_snr_hi] dB against the utterance RMS;
//   music    the same with a music recording and its own SNR range.
//
// Each is applied independently with its probability. Sources are lists of
// audio files (anything load_audio_16k reads), decoded on demand; a bounded
// pool of decoded noise/music tracks is kept so long tracks are not decoded
// per utterance.

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace laya_audio {

struct AugmentConfig {
    std::string rir_list, noise_list, music_list;  // one path per line
    float p_rir = 0.4f, p_noise = 0.4f, p_music = 0.15f;
    float noise_snr_lo = 0.0f, noise_snr_hi = 20.0f;
    float music_snr_lo = 5.0f, music_snr_hi = 20.0f;
    int pool = 48;  // decoded tracks kept per source
};

struct AugmentApplied {
    bool rir = false, noise = false, music = false;
};

class Augmenter {
public:
    void load(const AugmentConfig& cfg);
    bool enabled() const { return !rirs_.empty() || !noises_.files.empty() || !music_.files.empty(); }
    AugmentApplied apply(std::vector<float>& pcm, std::mt19937& rng);

private:
    struct Source {
        std::vector<std::string> files;
        std::vector<std::shared_ptr<const std::vector<float>>> pool;
        std::shared_ptr<const std::vector<float>> hold;  // the last decoded track
    };
    const std::vector<float>& track(Source& s, std::mt19937& rng);
    void mix(std::vector<float>& pcm, Source& s, float snr_db, std::mt19937& rng);
    AugmentConfig cfg_;
    std::vector<std::string> rirs_;
    Source noises_, music_;
};

// Linear convolution y = x * h (full length), by FFT.
std::vector<float> fft_convolve(const std::vector<float>& x, const std::vector<float>& h);

}  // namespace laya_audio
