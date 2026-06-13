#pragma once

#include "brosoundml/mel.h"

#include <atomic>
#include <cstdint>
#include <vector>

namespace brosoundml {

// ─── SensorHub — tier-0 acoustic sensor bus ──────────────────────────────────
//
// The fast, cheap, always-on layer of the listening stack. One streaming PCEN
// mel front-end (the same recipe WakeWord and PhonemeSpotter run) drives a set
// of per-frame DSP sensors — no model, no enrollment — and publishes every
// signal into one lock-free snapshot a fuser can poll from any thread:
//
//   level     frame RMS / peak / dBFS over the analysis window.
//   voice     energy VAD: adaptive noise floor (instant attack down, slow
//             release up) + SNR gate + hangover. Catches "someone is making
//             sound" within one frame of it starting.
//   onset     spectral flux on the PCEN mel frames: percussive transients
//             (clicks, taps, snaps) that a sequence matcher structurally
//             cannot hold onto fire here in a single frame.
//   tonality  normalized-autocorrelation periodicity on the raw analysis
//             window + the period's frequency: sustained periodic sounds
//             (whistles, hums — pure OR harmonic) read as a high-periodicity
//             run with a stable dominant_hz. Deliberately NOT a PCEN-mel
//             measure: PCEN's job is to flatten static spectral shape, so a
//             sustained tone's mel frame goes flat (measured peakiness 2.1 on
//             a clean 1 kHz sine vs 1.7 on white noise — no separation).
//
// Latency: every sensor updates once per mel frame (hop_length samples —
// 10 ms at the default recipe), and the snapshot is published before feed()
// returns, so a reader sees a transient at most one frame + one poll after it
// hits the mic. PCEN makes the mel-derived sensors (onset, tonality)
// gain-robust by construction; the level/VAD pair intentionally stays on raw
// PCM so the stack keeps one absolute-loudness signal.
//
// Edges are poll-safe: momentary booleans (`onset` is true only on its
// triggering frame) are paired with monotonic counters (`onsets`,
// `voice_events`, `tonal_events`) and last-event frame indices, so a reader
// polling slower than the frame rate still observes every event as a counter
// delta instead of hoping to land on the hot frame.
//
// Thread-safety: single-producer. feed()/reset() run on ONE thread (the audio
// inference thread). snapshot() is a lock-free seqlock read, safe from any
// other thread at any time. No locks anywhere.
//
// This is deliberately the SKELETON of the sensor stack: model-driven
// consumers (PhonemeNet posteriors, the spotter's matchers) attach alongside
// these sensors in later steps so the whole stack shares one front-end.

struct SensorHubConfig {
    // Shared front-end. Defaults to the KWS recipe (16 kHz / 512 FFT / 25 ms
    // window / 10 ms hop / 40 PCEN mels) so the hub's frames are the same
    // features the wake/spotter models consume.
    MelConfig mel;

    // ─ voice (energy VAD) ───────────────────────────────────────────────
    float vad_abs_floor_db    = -55.0f; // below this the frame can never be voice
    float vad_snr_db          = 8.0f;   // frame must beat the noise floor by this
    float vad_floor_rise_dbps = 6.0f;   // noise-floor release rate (dB per second);
                                        // attack down to a quieter frame is instant
    int   vad_hang_frames     = 25;     // hold `voice` this many frames past the
                                        // last qualifying frame (250 ms @ 10 ms hop)

    // ─ onset (spectral flux) ────────────────────────────────────────────
    float onset_ratio  = 2.5f;   // flux must exceed its slow EMA by this factor ...
    float onset_abs    = 0.05f;  // ... and this absolute PCEN-flux floor (rejects
                                 // near-silence micro-flux where the EMA is ~0)
    float onset_ema    = 0.05f;  // EMA coefficient for the flux baseline (~0.5 s
                                 // time constant @ 10 ms hop)
    int   onset_refractory_frames = 5;  // one onset per 50 ms

    // ─ tonality (autocorrelation periodicity) ───────────────────────────
    float tonal_min_periodicity = 0.60f;  // normalized autocorrelation peak in
                                          // [0,1]: clean tones/hums sit > 0.9,
                                          // white noise near 0
    float tonal_fmin_hz = 80.0f;   // pitch search range — down to a low hum ...
    float tonal_fmax_hz = 4000.0f; // ... up to a sharp whistle

    SensorHubConfig() { mel.compression = MelCompression::PCEN; }
};

// One coherent reading of every sensor, all taken on the same mel frame.
// Plain POD — the seqlock publishes it as a single struct copy.
struct SensorSnapshot {
    std::int64_t frames = 0;   // mel frames processed since reset()
    double       t      = 0.0; // stream time (s) at the END of the latest frame

    // level (raw PCM over the latest analysis window)
    float rms  = 0.0f;
    float peak = 0.0f;
    float db   = -120.0f;      // 20*log10(rms), floored at -120

    // voice
    bool         voice          = false;
    float        noise_floor_db = -120.0f;
    float        snr_db         = 0.0f;   // db - noise_floor_db
    std::int64_t voice_frames   = 0;      // consecutive frames voice has been true
    std::int64_t voice_events   = 0;      // silence→voice transitions since reset
    std::int64_t last_voice_frame = -1;   // frame index of the latest transition

    // onset
    float        flux   = 0.0f;           // this frame's positive PCEN flux
    bool         onset  = false;          // true ONLY on the triggering frame
    std::int64_t onsets = 0;              // total onsets since reset
    std::int64_t last_onset_frame = -1;

    // tonality
    float        periodicity  = 0.0f;     // normalized autocorrelation peak [0,1]
    float        dominant_hz  = 0.0f;     // frequency of the winning period
                                          // (meaningful when periodicity is high)
    bool         tonal        = false;
    std::int64_t tonal_frames = 0;        // consecutive frames tonal has been true
    std::int64_t tonal_events = 0;        // nontonal→tonal transitions since reset
    std::int64_t last_tonal_frame = -1;

    // spectral shape (PCEN mel centroid — "brightness")
    float        centroid = 0.0f;         // energy-weighted mel-bin center,
                                          // normalized to [0,1] (0 = low/dark,
                                          // 1 = high/bright). A cheap, gain-robust
                                          // timbre axis: a tap reads low, a finger
                                          // snap high. The gesture matcher uses it
                                          // to tell beats of different sound shape
                                          // apart even when they land on the same
                                          // beat (along with periodicity/pitch).
};

class SensorHub {
public:
    explicit SensorHub(const SensorHubConfig& cfg = {});

    // Push `n` mono FP32 samples at config().mel.sample_rate. Advances the
    // front-end and every sensor one mel frame at a time as enough samples
    // accumulate; publishes a snapshot per frame. Returns frames processed
    // this call. Single producer.
    int feed(const float* samples, int n);

    // Driven mode — the seam ListenBus calls so the whole stack shares ONE
    // front-end. Advances every sensor one frame from externally computed
    // inputs: `window` is the raw analysis window the frame was framed from
    // (config().mel.win_length samples), `mel_frame` the matching compressed
    // mel column (config().mel.n_mels values). Publishes the snapshot before
    // returning. The hub's own front-end and sample ring are NOT touched, so
    // don't interleave feed() and feed_frame() within one stream (the two
    // paths' PCEN/ring state would diverge). Single producer, same thread
    // contract as feed().
    void feed_frame(const float* window, const float* mel_frame);

    // Drop all streaming state (mel ring, sample ring, PCEN smoother, sensor
    // state, counters). Configuration is kept.
    void reset();

    // Latest coherent sensor reading. Lock-free (seqlock); any thread.
    SensorSnapshot snapshot() const;

    const SensorHubConfig& config() const { return cfg_; }
    int sample_rate() const { return cfg_.mel.sample_rate; }

private:
    void process_frame(const float* window, const float* mel_frame);
    void publish();

    SensorHubConfig cfg_;
    MelFrontend     mel_;

    // Sample ring mirroring the front-end's framing (first frame after
    // win_length samples, hop_length thereafter) so level/VAD see exactly the
    // window each mel frame was computed from.
    std::vector<float> ring_;

    // Per-frame sensor state (producer thread only).
    std::vector<float> prev_frame_;     // previous PCEN frame (flux reference)
    std::vector<float> nac_;            // per-lag autocorrelation scratch
    bool   have_prev_     = false;
    bool   floor_init_    = false;
    float  flux_ema_      = 0.0f;
    int    vad_hang_left_ = 0;
    int    onset_cooldown_ = 0;

    // The published snapshot. `cur_` is the producer's working copy; `pub_`
    // is the seqlock-guarded copy readers snapshot. Fixed-size POD, so the
    // guarded write never allocates.
    SensorSnapshot cur_;
    SensorSnapshot pub_;
    mutable std::atomic<std::uint32_t> seq_{0};  // even = stable, odd = writing
};

}  // namespace brosoundml
