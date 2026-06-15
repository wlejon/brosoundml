#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── Clustering speaker diarization (embedding + cosine AHC) ─────────────────
//
// An alternative to end-to-end Sortformer for the case Sortformer's 4-slot head
// cannot handle: telling apart acoustically *similar* voices (e.g. two women in
// the same pitch range). Sortformer learns speaker slots jointly with VAD and
// exposes no "how different must two voices be" control, so similar speakers
// collapse into one slot — a limit of the reference model itself, reproduced
// bit-for-bit by the C++ port.
//
// This pipeline separates the two jobs Sortformer fuses:
//
//   1. WHERE is speech   — Sortformer's per-frame activity (any speaker active)
//                          is a strong, parity-verified VAD. We reuse only that.
//   2. WHO is speaking    — each speech window is embedded with the ECAPA-TDNN
//                          x-vector encoder (SpeakerEncoder), the embeddings are
//                          mean-centered (raw x-vectors sit in a narrow cone
//                          where every cosine is ~0.95; centering against the
//                          recording mean is what makes them speaker-
//                          discriminative — within-speaker cosine goes positive,
//                          cross-speaker negative), then agglomeratively
//                          clustered by cosine. `cluster_threshold` is the
//                          tunable knob Sortformer lacks.
//
// Unlike Sortformer this assigns one speaker per window, so it does not split
// overlapped speech — the trade for resolving similar voices. Speaker count is
// discovered from the threshold (capped at max_speakers); labels are emitted in
// arrival-time order to match the Sortformer surface.
//
// Owns its own Sortformer (VAD) and SpeakerEncoder (x-vectors); load() places
// the Sortformer encoder on `device` (the speaker encoder runs host-side).

class ClusterDiarizer {
public:
    struct Config {
        float vad_threshold       = 0.40f;  // frame is speech if P(any spk) > this
        float window_seconds      = 2.50f;  // embedding window length — long enough
                                            // for a stable x-vector (shorter
                                            // windows lose the speaker margin to
                                            // noise and merge similar voices)
        float hop_seconds         = 1.00f;  // window shift
        float min_window_seconds  = 0.60f;  // skip speech runs shorter than this
        float cluster_threshold   = 0.40f;  // merge clusters while centered cos > this
                                            // (population-centered cosine: lone
                                            // speakers cluster high, distinct
                                            // speakers fall below ~0.4)
        float min_speaker_seconds = 1.00f;  // fold clusters smaller than this away
        int   max_speakers        = 8;      // hard cap on discovered speakers
    };

    // Per-frame speaker activity, same shape/semantics as Sortformer::Diarization:
    // probs is row-major (num_frames, num_speakers); for this clusterer each
    // speech frame is one-hot (1.0 for its assigned speaker), non-speech all-zero.
    struct Diarization {
        int                num_frames    = 0;
        int                num_speakers  = 0;
        std::vector<float> probs;            // (num_frames * num_speakers)
        double             frame_seconds  = 0.08;
    };

    ClusterDiarizer();
    ~ClusterDiarizer();
    ClusterDiarizer(ClusterDiarizer&&) noexcept;
    ClusterDiarizer& operator=(ClusterDiarizer&&) noexcept;
    ClusterDiarizer(const ClusterDiarizer&) = delete;
    ClusterDiarizer& operator=(const ClusterDiarizer&) = delete;

    // Load the Sortformer model dir (config.json + model.safetensors) used as the
    // VAD and the standalone speaker-encoder artifact dir (config.json +
    // model.safetensors). The Sortformer encoder is placed on `device`; the
    // speaker encoder runs host-side. Throws std::runtime_error on a bad model.
    void load(const std::string& sortformer_dir,
              const std::string& speaker_encoder_dir,
              brotensor::Device device = brotensor::Device::CPU);

    // Offline diarization: 16 kHz mono PCM in, a (T, num_speakers) one-hot
    // activity matrix out. num_speakers is discovered from cluster_threshold.
    Diarization diarize(const AudioBuffer& audio, const Config& cfg) const;
    Diarization diarize(const AudioBuffer& audio) const { return diarize(audio, Config{}); }

    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
