#pragma once

#include "brosoundml/mel.h"
#include "brosoundml/phoneme_spotter.h"
#include "brosoundml/sensor_hub.h"
#include "brosoundml/wake.h"

#include <vector>

namespace brosoundml {

// ─── ListenBus — shared front-end for the listening stack ───────────────────
//
// One streaming PCEN mel front-end + one raw-sample ring driving every
// attached consumer of the stack, so N listeners cost ONE feature pass over
// the live audio:
//
//   SensorHub        tier-0 DSP sensors — fed per frame (raw analysis window
//                    + the matching mel column) via SensorHub::feed_frame.
//   PhonemeSpotter   tier-1/2 — each call's new-frame mel block goes through
//                    ONE PhonemeNet forward (PhonemeSpotter::feed_mel); the
//                    spotter fans the resulting posteriors out to its enrolled
//                    template matchers, and its lock-free readers
//                    (last_posterior / prefix_progress) expose the stream to
//                    any further pollers.
//   WakeWord         tier-2 keyword detector — the same new-frame mel block
//                    drives the streaming BC-ResNet + detector policy
//                    (WakeWord::feed_mel). Joined once the AGC-free recipe
//                    landed: the model is trained level-invariant (random
//                    presentation level + PCEN), so it hears the same raw
//                    no-AGC stream as every other consumer.
//
// Consumers are passed PER CALL rather than stored. feed() is single-producer
// (one audio/inference thread); a caller that owns that thread changes the
// consumer set by changing what it passes — no cross-thread coordination
// lives inside the bus. (bro swaps the whole inference-task closure to change
// membership, so the bus is only ever touched from task closures, in order.)
// Pass nullptr for a consumer that isn't active.
//
// The front-end runs on CPU: at 100 frames/s the FFT + filterbank costs
// microseconds, and every consumer wants the frames host-side anyway (the
// spotter uploads the block to its model's device itself).
//
// Compatibility: a consumer must have been built/trained on the bus's exact
// framing (rate / FFT / window / hop / mels / PCEN parameters).
// check_compatible() validates that — call it when a consumer joins (it
// throws std::runtime_error naming the mismatched field); feed() trusts the
// caller and does not re-validate per call.
//
// Thread-safety: single producer. feed()/reset() on one thread; the bus has
// no cross-thread readers of its own (poll the consumers' lock-free surfaces).

struct ListenFeedResult {
    int frames = 0;                  // mel frames advanced this call
    std::vector<SpotEvent> spots;    // spotter completions fired this call
    bool wake_fired = false;         // wake detector fired this call
};

class ListenBus {
public:
    // Defaults to the KWS recipe (16 kHz / 512 FFT / 25 ms window / 10 ms hop
    // / 40 mels). PCEN compression is forced on — both model recipes train
    // on it, and the tier-0 mel sensors assume it.
    explicit ListenBus(const MelConfig& mel = MelConfig{});

    // Throw std::runtime_error (naming the field) if the consumer's framing
    // differs from the bus front-end. The spotter/wake overloads require
    // load() (their framing comes from the checkpoint header).
    void check_compatible(const SensorHub& hub) const;
    void check_compatible(const PhonemeSpotter& spotter) const;
    void check_compatible(const WakeWord& wake) const;

    // Push `n` mono FP32 samples at sample_rate(); advance the front-end and
    // every non-null consumer. Single producer.
    ListenFeedResult feed(const float* samples, int n,
                          SensorHub* hub, PhonemeSpotter* spotter,
                          WakeWord* wake = nullptr);

    // Drop the bus's stream state (mel ring, PCEN smoother, sample ring).
    // Consumers keep their own streaming state — reset them alongside.
    void reset();

    const MelConfig& config() const { return cfg_; }
    int sample_rate() const { return cfg_.sample_rate; }

private:
    MelConfig   cfg_;
    MelFrontend mel_;

    // Raw samples mirroring the front-end's framing (ring_ always starts at
    // the next unemitted frame's window start), so frame f of a call covers
    // ring_[f*hop .. f*hop + win) — the window SensorHub::feed_frame wants.
    std::vector<float> ring_;

    std::vector<float> col_;    // per-frame mel-column scratch (n_mels)
    std::vector<float> block_;  // repack scratch for a partial block
};

}  // namespace brosoundml
