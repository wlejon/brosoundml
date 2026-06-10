#pragma once

#include <brotensor/tensor.h>

#include "brosoundml/phoneme_data.h"

#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── PhonemeSpotter — open-vocabulary streaming keyword spotter ──────────────
//
// The runtime that turns PhonemeNet's per-frame K-way phoneme-class posteriors
// into keyword detections against enrolled phoneme-sequence templates. It is the
// open-vocabulary analogue of WakeWord: instead of one trained-in keyword logit,
// the spotter scores a *streaming Viterbi alignment* of each enrolled template
// (a sequence of distinct adjacent phoneme classes) against the posterior stream
// and fires when an alignment completes with a high enough geometric-mean
// posterior, gated by an entry-silence word boundary, an M-of-N smoother, and a
// per-template refractory debounce.
//
// Two seams, mirroring WakeWord but split for testability:
//   • REAL path: load() the PhonemeNet checkpoint (+ its embedded class map),
//     enroll templates from phonemizer ids / class ids / reference audio, then
//     push mic PCM through feed(). Internally feed() runs the PCEN mel
//     front-end → forward_streaming → per-frame softmax → feed_posteriors().
//   • OFFLINE / TEST path: set_class_map() installs an inventory with NO model
//     on disk, enabling enroll() + feed_posteriors() so the whole matcher can be
//     unit-tested on synthetic posterior streams with zero weights / DSP / GPU.
//
// Thread-safety: single-producer. feed()/feed_posteriors()/enroll()/remove()/
// clear()/reset() all run on ONE thread (the audio thread). The scalar
// cross-thread readers (prefix_progress(), last_posterior()) are lock-free —
// relaxed atomics / a seqlock — for a UI thread to poll. No locks anywhere.

struct SpotterConfig {
    // Detector policy DEFAULTS (per-template values can override at enroll time).
    float threshold      = 0.40f;   // geometric-mean per-frame posterior over the
                                    // matched template span must exceed this to fire.
    int   smoothing_hits   = 2;     // M-of-N: need this many qualifying frames ...
    int   smoothing_window = 3;     // ... within the last N frames.
    int   refractory_ms  = 600;     // suppress re-fires of the SAME template this long.
    int   min_phonemes   = 3;       // templates shorter than this never fire (noise floor).
    int   entry_silence_frames = 2; // a match may only BEGIN after >= this many recent
                                    // silence frames (word-boundary entry gate).
};

struct SpotEvent {
    std::string name;            // enrolled template name that fired
    float       confidence = 0.0f;   // geometric-mean posterior over the matched span, [0,1]
    int         matched_phonemes = 0;  // == template length on a full match
    int         template_len = 0;
};

class PhonemeSpotter {
public:
    PhonemeSpotter();
    ~PhonemeSpotter();
    PhonemeSpotter(PhonemeSpotter&&) noexcept;
    PhonemeSpotter& operator=(PhonemeSpotter&&) noexcept;
    PhonemeSpotter(const PhonemeSpotter&) = delete;
    PhonemeSpotter& operator=(const PhonemeSpotter&) = delete;

    // REAL use: load the phoneme model + its embedded class map onto device.
    void load(const std::string& weights_path,
              brotensor::Device device = brotensor::Device::CPU);

    // TEST / offline seam: install a class map WITHOUT a model, enabling enroll()
    // and feed_posteriors() with no weights on disk. (load() also sets the class
    // map.) Resets any streaming state and drops any model.
    void set_class_map(const PhonemeClassMap& cm);

    const PhonemeClassMap& class_map() const;
    bool loaded() const;          // model present (feed(samples) usable)
    bool has_class_map() const;   // enroll()/feed_posteriors() usable

    // ── Enrollment ──
    // Enroll a template from a Kokoro PHONEME-ID sequence (e.g. a phonemizer's
    // output ids). Maps each id -> class via class_map; DROPS silence-class ids
    // and transparent (suprasegmental) ids; collapses consecutive duplicate
    // classes into one. Returns the resulting template length (0 if empty).
    int enroll(const std::string& name, const std::vector<int>& phoneme_ids,
               const SpotterConfig* policy_override = nullptr);

    // Enroll directly from CLASS ids (already in [0,K), the matcher's alphabet).
    // Same silence-drop + duplicate-collapse. Useful for tests and for callers
    // that already speak class ids.
    int enroll_from_classes(const std::string& name,
                            const std::vector<int>& class_ids,
                            const SpotterConfig* policy_override = nullptr);

    // Enroll by running reference AUDIO through the model: feeds the samples,
    // takes per-frame argmax, collapses runs, drops silence -> the class
    // sequence becomes the template. Requires load(). Returns template len.
    int enroll_from_audio(const std::string& name, const float* samples, int n,
                          const SpotterConfig* policy_override = nullptr);

    bool remove(const std::string& name);
    void clear();                 // drop all templates
    std::vector<std::string> templates() const;

    // ── Streaming ──
    // REAL use: push mono FP32 PCM @ model sample_rate. Internally: MelFrontend
    // (PCEN) -> per-frame forward_streaming -> softmax -> feed_posteriors().
    // Returns events that fired DURING this call (possibly empty / multiple).
    std::vector<SpotEvent> feed(const float* samples, int n);

    // TEST / offline seam: push pre-softmaxed per-frame posteriors directly,
    // bypassing mel+model. `posteriors` is row-major (n_frames, K); each row is a
    // probability distribution over the K classes. Returns events fired this call.
    std::vector<SpotEvent> feed_posteriors(const float* posteriors, int n_frames);

    // ── Cross-thread readers (lock-free) ──
    // Most recent posterior frame (copy of the (K) vector); empty before first
    // frame. Lock-free seqlock snapshot.
    std::vector<float> last_posterior() const;
    // Best current prefix progress across all templates, in [0,1]
    // (matched_phonemes / template_len of the furthest-advanced template).
    float prefix_progress() const;

    void reset();   // drop ALL streaming state (mel ring, model conv cache, every
                    // template's DP state, smoothing, refractory). Keeps weights,
                    // class map, and enrolled templates.

    const SpotterConfig& config() const;        // global defaults
    void set_config(const SpotterConfig& cfg);  // updates defaults for templates
                                                // enrolled WITHOUT an override.

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
