#pragma once

#include "brosoundml/sensor_hub.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brosoundml {

// ─── GestureSpotter — non-speech gesture matching over tier-0 sensors ────────
//
// PhonemeSpotter is speech-only: its model emits phoneme-class posteriors, so a
// whistle or a click train run through it decodes to unstable garbage tokens
// that won't even match their own re-performance. The right tier for non-speech
// is tier-0 — SensorHub already turns the stream into exactly the right
// primitives: ONSETS (percussive transients — clicks, taps, claps) and TONAL
// runs (sustained periodic sound — whistles, hums — with a dominant pitch).
//
// GestureSpotter is the open-vocabulary matcher over THAT stream, the tier-0
// analogue of PhonemeSpotter. It consumes SensorHub's per-frame SensorSnapshot
// (it owns no DSP — in the shared listen host it reads the same hub bro.sense
// runs, so a gesture costs nothing beyond the sensors already computed) and
// fires named events when an enrolled gesture re-occurs:
//
//   rhythm   >= 2 onsets — the inter-onset intervals are the template; a
//            re-performance whose taps land at the same spacing (within a
//            tempo tolerance) fires. A single lone transient is too ambiguous
//            to enroll (it would fire on every tap), so rhythms need >= 2 hits.
//   tone     a sustained tonal run — its dominant pitch (and minimum duration)
//            is the template; a held whistle/hum at the same pitch fires.
//
// A gesture is classified at enroll from what the clip actually contained
// (>= min_onsets onsets -> rhythm; else a long-enough tonal run -> tone; else
// too sparse, rejected). enroll runs the reference clip through a private
// SensorHub offline; the live path is feed(snapshot) per frame.
//
// Thread-safety: single-producer, same contract as PhonemeSpotter. feed() /
// enroll / remove / clear / reset run on ONE thread. No locks.

struct GestureConfig {
    // Match tolerances (per-template values can override at enroll).
    float tempo_tol = 0.40f;   // rhythm: each observed inter-onset interval must
                               // be within this fraction of the enrolled one;
                               // also the tone min-duration slack.
    float pitch_tol = 0.12f;   // tone: dominant pitch must be within this
                               // fraction of the enrolled pitch.
    int   refractory_frames = 40;  // suppress re-fires of the SAME gesture this
                                   // many frames (~400 ms at the 10 ms hop).
    int   min_onsets    = 2;   // a rhythm gesture needs at least this many onsets.
    int   min_tone_frames = 8; // a tone gesture's run must last at least this
                               // long to enroll AND to fire (~80 ms).

    // Front-end + sensors for the offline enroll pass and (informationally) the
    // framing the live snapshots are expected to come from. Defaults to the KWS
    // recipe, matching the shared listen host.
    SensorHubConfig sensor;
};

enum class GestureKind { Rhythm, Tone };

struct GestureEvent {
    std::string name;
    float       confidence = 0.0f;   // [0,1], 1 == exact reproduction
    GestureKind kind = GestureKind::Rhythm;
};

// Human-legible view of an enrolled gesture (cf. PhonemeSpotter::TemplateView):
// what the clip became, so a tool can show "captured a 3-tap rhythm at
// 250/250 ms" or "captured a 1200 Hz tone".
struct GestureView {
    std::string name;
    GestureKind kind = GestureKind::Rhythm;
    float       frame_ms = 10.0f;
    // Rhythm: inter-onset intervals in frames (n_onsets == intervals+1).
    std::vector<int> intervals;
    // Tone:
    float tone_hz = 0.0f;
    int   tone_frames = 0;
};

class GestureSpotter {
public:
    explicit GestureSpotter(const GestureConfig& cfg = {});

    const GestureConfig& config() const { return cfg_; }
    void set_config(const GestureConfig& cfg);   // updates non-override templates
    int  sample_rate() const { return cfg_.sensor.mel.sample_rate; }

    // ── Enrollment ──
    // Enroll by example: run the reference clip (mono FP32 at sample_rate())
    // through a private SensorHub, extract its gesture, store the template.
    // Returns the number of "beats" (rhythm: onsets; tone: 1). Throws if the
    // clip is too sparse to be a gesture (no rhythm and no sustained tone).
    int enroll_from_audio(const std::string& name, const float* samples, int n,
                          const GestureConfig* policy_override = nullptr);

    // Enroll from a pre-computed SensorSnapshot stream (one per frame, in
    // order) — the seam enroll_from_audio uses internally, exposed for tests
    // and callers that already have the sensor frames.
    int enroll_from_snapshots(const std::string& name,
                              const SensorSnapshot* snaps, int n,
                              const GestureConfig* policy_override = nullptr);

    bool remove(const std::string& name);
    void clear();
    std::vector<std::string> templates() const;
    bool inspect(const std::string& name, GestureView& out) const;

    // ── Streaming ──
    // Advance every template's matcher one frame from a SensorHub snapshot.
    // Returns gestures that completed on this frame (usually empty). The caller
    // feeds consecutive snapshots from ONE hub stream (the shared listen host,
    // or a private hub).
    std::vector<GestureEvent> feed(const SensorSnapshot& snap);

    // Drop all streaming match state (onset rings, tone-run trackers,
    // refractory). Keeps templates.
    void reset();

private:
    struct Template {
        std::string  name;
        GestureKind  kind = GestureKind::Rhythm;
        GestureConfig policy;
        bool         has_override = false;
        // Rhythm:
        std::vector<int> intervals;     // enrolled inter-onset intervals (frames)
        // Tone:
        float tone_hz = 0.0f;
        int   tone_frames = 0;

        // ── streaming match state ──
        std::vector<std::int64_t> onset_hist;   // recent onset frame indices
        int          refractory = 0;
        bool         in_run = false;            // tone: currently in a tonal run
        std::int64_t run_start = 0;
        double       run_hz_sum = 0.0;
        int          run_len = 0;
        bool         run_fired = false;         // fired once for the current run
    };

    GestureConfig          cfg_;
    std::vector<Template>  tmpls_;

    // Build a Template from a snapshot stream (shared by both enroll paths).
    static Template build_template(const std::string& name,
                                   const SensorSnapshot* snaps, int n,
                                   const GestureConfig& pol, bool has_override);
};

}  // namespace brosoundml
