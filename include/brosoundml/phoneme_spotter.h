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
    float emission_floor = 0.15f;   // per-frame log-posterior floor in the matcher: no
                                    // single frame may drive a template phoneme's
                                    // contribution below log(emission_floor). Frame-level
                                    // posteriors for brief/transient phonemes (stop
                                    // bursts, glides, flaps) are unreliable and routinely
                                    // collapse to ~0; without a floor one such phoneme
                                    // multiplicatively crushes the whole geometric-mean
                                    // confidence to zero, so a CITATION template (enroll
                                    // from g2p ids, where every canonical phoneme must
                                    // align) can never fire even when the word is clearly
                                    // present. The floor bounds that veto so the score
                                    // degrades gracefully instead of underflowing. Must be
                                    // < threshold (a garbage span where every template
                                    // class is absent scores ~emission_floor, so the
                                    // threshold still rejects it). 0 disables.
    float score_norm = 0.0f;        // competition-normalization strength in [0,1]. At 0
                                    // (off) a frame contributes its RAW log-posterior, so
                                    // the geometric-mean confidence rides each class's
                                    // absolute calibration — and that calibration varies
                                    // systematically by phoneme type (long vowels/nasals
                                    // post near 0.9 when present; stop bursts/fricatives
                                    // peak far lower). Templates therefore live on
                                    // DIFFERENT score scales (measured on CAMEO: at one
                                    // fixed threshold, per-keyword FAR spans 0.000 to
                                    // 0.29), so no single threshold transfers across
                                    // words. At s > 0 the contribution becomes
                                    //   log p[c] − s · log max(p_argmax, score_norm_ref)
                                    // i.e. the posterior RATIO against the frame's winner:
                                    // a weak-but-WINNING emission scores ~1 regardless of
                                    // class, while a class that is losing the frame is
                                    // punished — putting all templates on one scale.
    float min_coverage_frac = 0.0f; // proportional coverage gate: a completion must
                                    // have at least max(min_phonemes,
                                    // ceil(min_coverage_frac * L)) template phonemes
                                    // with real (raw above-floor) evidence. The
                                    // absolute min_phonemes alone protects SHORT
                                    // templates well (3 of 4 = 75% evidence) but
                                    // long ones barely (3 of 9 = 33%): on dense
                                    // speech the DP can stitch a few genuinely-
                                    // occurring common classes into a "completion"
                                    // of a long template whose other phonemes all
                                    // ride the emission floor — measured on
                                    // LibriSpeech test-clean, a 6-phoneme nonsense
                                    // word ("boolooroo") false-fired on 17.6% of
                                    // negative clips at its EER threshold. 0
                                    // disables (absolute gate only).
    int   enroll_alts = 0;          // per-state ALTERNATE classes kept when enrolling
                                    // from audio/posteriors. Self-supervised units are
                                    // k-means cells; a re-performance of the same sound
                                    // lands near the same cell BOUNDARIES and its frames
                                    // flip between neighbouring units take-to-take
                                    // (measured on AVP re-takes: cores share order but
                                    // ~half the unit ids differ). Each enrolled state
                                    // then also records up to this many runner-up
                                    // classes from its defining frame, and the matcher
                                    // scores the frame's SUMMED mass over the state's
                                    // class set. OFF (0, hard argmax states) by default:
                                    // measured on AVP re-takes + LibriSpeech/VocalSound
                                    // negatives, the extra mass lifted negatives as much
                                    // as re-takes (no ROC gain) and wrecked short
                                    // templates' FAR (a 3-state click: 7% -> 96% of
                                    // speech clips at thr 0.50). Available for unit
                                    // spaces with better-separated cells.
    float enroll_alt_mass = 0.10f;  // a runner-up must hold at least this posterior
                                    // mass on the state's defining frame to be kept.
    float enroll_conf_gate = 0.0f;  // frames whose argmax posterior is below this are
                                    // treated as SILENCE at enrollment: room tone above
                                    // the silence-RMS floor and PCEN onset transients
                                    // label as low-confidence unit churn (~0.3-0.6 vs
                                    // ~0.8+ for the sound itself) that re-takes can
                                    // never reproduce; gating keeps only the sound's
                                    // confident core in the template. 0 disables.
    float score_norm_ref = 0.5f;    // denominator floor for score_norm. Pure ratio
                                    // (dividing by p_argmax itself) would inflate MUSHY
                                    // frames — in babble/noise the winner may hold only
                                    // ~0.2, and any template class near it scores ~1, so
                                    // everything false-fires. Capping the denominator at
                                    // this reference ("what a confidently-emitted phoneme
                                    // posts") keeps low-evidence frames scored near their
                                    // absolute posterior while confident frames get the
                                    // full ratio treatment.
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
    int  sample_rate() const;     // feed()'s expected PCM rate (the loaded
                                  // model's front-end rate; 16000 before load)

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
    // Honors enroll_conf_gate / enroll_alts (see SpotterConfig).
    int enroll_from_audio(const std::string& name, const float* samples, int n,
                          const SpotterConfig* policy_override = nullptr);

    // Enroll from a per-frame POSTERIOR stream (n_frames rows of K, row-major,
    // each row summing to ~1) — the same representation feed_posteriors()
    // consumes. Applies enroll_conf_gate, drops silence, collapses runs, and
    // records up to enroll_alts runner-up classes per state (the soft-state
    // mechanism enroll_from_audio uses internally). Requires a class map.
    int enroll_from_posteriors(const std::string& name, const float* posteriors,
                               int n_frames,
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
