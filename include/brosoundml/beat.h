#pragma once

// Beat phase sensors: an onset envelope of the audio at 100 frames a second,
// and a comb fitted to it at a known period, giving the phase of the beat.
//
// Measured in yue2-research (FINDINGS 10, native/endless.js, where these are
// ported from bit for bit): on 10 s windows of YuE2 streams under a score,
// the spectral-flux comb phase is within 0.1 beats of librosa's onset
// envelope in 90% of windows and half a beat off in 4% (the energy flux: 82%
// and 11%). The YuE2 endless stream reads a bridge's beat with it, and
// bro.music.beat exposes it for any audio.
//
// Determinism: every step is done in the order and precision endless.js does
// it (doubles, the envelope stored as float), so the same audio gives the
// same envelope and the same fit as the harness.

#include <cstddef>
#include <vector>

namespace brosoundml {

enum class BeatSensor {
    Spectral,   // spectral flux: 2048-point Hann FFT of the signal averaged to 24 kHz every 240 samples,
                // power in 48 log bands (60 Hz .. 11 kHz), log, positive frame difference, band mean;
                // frames indexed by their centres
    Energy,     // energy flux: the log energy of the signal and of its first difference (a crude
                // high-pass) per 480 samples, each differenced and rectified, summed
};

// The onset envelope of mono 48 kHz audio, 100 frames a second (frame i is
// the audio around i / 100 s).
std::vector<float> beat_onsets(const float* mono48k, std::size_t samples, BeatSensor sensor = BeatSensor::Spectral);

// Mono of planar stereo [L | R] (`frames` samples a channel), samples
// [offset, offset + count) clipped to the signal: float(0.5 * (L + R)).
std::vector<float> beat_mono(const float* planar, std::size_t frames, std::size_t offset, std::size_t count);

struct BeatFit {
    double period = 0;       // in envelope frames
    double phase = 0;        // in beats [0, 1): beats fall at (phase + n) * period - origin frames of the envelope
    double strength = 0;     // the comb's mean envelope value at the beats
    double contrast = 0;     // strength over the envelope's mean (1 = no beat; 0 when the envelope is empty)
    int beats = 0;           // beats the comb summed
};

// The comb fit: for each phase q/64 of a beat (and each period within +-3%
// of `period` in 0.2% steps when `search_period`), the mean of the envelope
// (linearly interpolated) at the beat positions; the best is kept. `origin`
// is the envelope frame 0's position on the caller's timeline, so phases of
// envelopes cut from one timeline agree: beats at frames (phase * P + n * P)
// of that timeline.
BeatFit fit_beat(const std::vector<float>& envelope, double origin, double period, bool search_period = false);

// The beat of mono 48 kHz audio at a known period (seconds): its envelope
// fitted with origin 0, the period and phase reported in envelope frames /
// beats as fit_beat does (so phase * period / 100 is the first beat's time
// in seconds).
BeatFit beat_phase(const float* mono48k, std::size_t samples, double period_seconds,
                   BeatSensor sensor = BeatSensor::Spectral, bool search_period = false);

}  // namespace brosoundml
