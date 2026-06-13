#pragma once

#include "brosoundml/audio.h"
#include "brosoundml/bc_resnet2d.h"
#include "brosoundml/mel.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>

namespace brosoundml {

// ─── Wake-word detector ────────────────────────────────────────────────────
//
// A small streaming convolutional classifier that fires when a target keyword
// ("computer" is the first target) is spoken. The shipped model is a 2D
// BC-ResNet (BcResnet2d): a PCEN mel front-end feeds a single-channel
// (freq x time) image into Broadcasted-Residual conv2d blocks (weights shared
// across frequency, strictly causal in time), then a frequency-average-pool →
// temporal global-average-pool → linear head → single logit. The earlier 1D
// model (mel bins as conv channels) keyed on the TTS training spectral envelope
// and collapsed on real microphones, so load() rejects the legacy 'BWAK'
// checkpoint and requires the 2D 'BWK2' model. Designed for an always-on mic
// loop with a per-frame inference cost in the low milliseconds; device-neutral
// (load() takes a brotensor::Device, CPU or CUDA).
//
// Latency budget (the design constraint everything below is sized to):
//
//     ┌─────────────────────────────────────────┬────────────┐
//     │ stage                                   │ budget     │
//     ├─────────────────────────────────────────┼────────────┤
//     │ feature extract (1 new frame)           │ < 0.5 ms   │
//     │ model forward    (1 new frame)          │ < 2.0 ms   │
//     │ smoothing window (2-of-3 @ 10 ms hop)   │   20.0 ms  │
//     └─────────────────────────────────────────┴────────────┘
//
// Audio in: mono FP32 PCM at WakeConfig::sample_rate (16 kHz default).
// Feed samples in any chunk size; the detector buffers and emits at the
// configured hop rate. Returns true exactly once per detected event,
// debounced by `refractory_ms`.
//
// STATUS: complete — the full streaming runtime and its training toolchain
// have landed. load() reads the 2D BC-ResNet weights; feed() runs the streaming
// PCEN mel front-end (stft → magnitude → mel → PCEN) ▶ model ▶ a GAP-ring warmup
// guard ▶ 2-of-3 smoothing ▶ refractory debounce, returning true once per
// detected event.
//
// Supporting tools (built under tools/):
//   brosoundml_wake_synth      Kokoro-driven dataset builder
//   brosoundml_wake_inspect    dataset validator
//   brosoundml_wake_train      backward + Adam + BCE-with-logits
//   brosoundml_wake_test       held-out evaluation
//   brosoundml_wake_probe      front-end diagnostics
//   brosoundml_wake_melcmp     mel front-end comparison

// Front-end + model + detector hyperparameters. The defaults match the
// "computer" recipe: 16 kHz, 10 ms hop, 25 ms window, 40-bin PCEN mel,
// 1-second receptive field, 2-of-3 smoothing with a 500 ms refractory.
// Loading a trained model overwrites the front-end / model fields from the
// weights file's header — only the detector-policy fields (threshold,
// smoothing, refractory) are caller-tunable at runtime.
struct WakeConfig {
    // ─ Front-end ────────────────────────────────────────────────────────
    int   sample_rate = 16000;   // input rate the model expects
    int   n_fft       = 512;     // STFT window size in samples
    int   win_length  = 400;     // analysis window length (25 ms @ 16 kHz)
    int   hop_length  = 160;     // hop between frames     (10 ms @ 16 kHz)
    int   n_mels      = 40;      // mel filterbank size
    float mel_fmin    = 20.0f;   // Hz — mel filterbank low edge
    float mel_fmax    = 8000.0f; // Hz — mel filterbank high edge (Nyquist @ 16k)

    // ─ Model ────────────────────────────────────────────────────────────
    int receptive_field_frames = 100;  // 100 frames × 10 ms hop = 1.00 s context

    // ─ Detector policy ──────────────────────────────────────────────────
    float threshold       = 0.55f;  // sigmoid(logit) > threshold ⇒ frame-positive
    int   smoothing_hits  = 2;      // need this many frame-positives ...
    int   smoothing_window= 3;      // ... within the last N frames to fire
    int   refractory_ms   = 500;    // ignore re-fires for this long after a hit
};

// Streaming wake-word detector. Construct, load() a weights file, then push
// mic samples through feed(). Heavy state — the mel front-end's ring buffer,
// the model's conv-state cache, the smoothing/refractory bookkeeping — lives
// behind a pImpl so this header stays free of brotensor module internals.
//
// Thread-safety: single-producer. feed() is intended to run on the audio
// callback thread; setters and lastScore() may be read from another thread
// (atomically), but no two threads may call feed() concurrently.
class WakeWord {
public:
    WakeWord();
    ~WakeWord();
    WakeWord(WakeWord&&) noexcept;
    WakeWord& operator=(WakeWord&&) noexcept;
    WakeWord(const WakeWord&) = delete;
    WakeWord& operator=(const WakeWord&) = delete;

    // Construct over a SHARED, already-loaded 2D BC-ResNet (the multi-stream
    // entry). The weights live once behind the shared_ptr; this detector owns
    // only its own streaming session (a BcResnet2dSession) plus its
    // front-end and detector bookkeeping. Build N detectors over one net to run
    // the same wake word on N asynchronous streams (e.g. mic + system-audio
    // loopback) without copying weights. Front-end framing comes from the net.
    // Throws if net is null. Detector-policy fields keep their current values.
    explicit WakeWord(std::shared_ptr<const BcResnet2d> net);

    // Load a trained wake-word checkpoint. The file carries the front-end and
    // model hyperparameters in its header; the loader overwrites the matching
    // WakeConfig fields, then materialises every module on `device`. Throws
    // std::runtime_error on a missing or malformed file. Detector-policy
    // fields (threshold, smoothing, refractory) are not touched — they keep
    // whatever the caller set, or their defaults.
    void load(const std::string& weights_path,
              brotensor::Device device = brotensor::Device::CPU);

    // Push `n` mono FP32 samples at WakeConfig::sample_rate. The detector
    // buffers internally and advances the streaming front-end / model state
    // one frame at a time once enough samples have accumulated. Returns true
    // exactly once per detected event — on the frame the smoothing rule
    // crosses its threshold — then suppresses re-fires for `refractory_ms`.
    // Throws std::runtime_error if no model is loaded.
    bool feed(const float* samples, int n);

    // Shared-front-end seam (what ListenBus drives): push PRECOMPUTED PCEN
    // mel frames — host FP32, (n_mels, n_frames) row-major, MelFrontend's
    // emit layout — through the streaming model + detector policy, bypassing
    // the detector's own front-end. The frames must come from a front-end
    // configured exactly like mel_config() (the bus validates this at
    // attach). Don't mix feed() and feed_mel() on one stream — the internal
    // front-end's sample state is not advanced by this path. Same fire
    // semantics as feed(): true at most once per call, then refractory.
    // Requires load().
    bool feed_mel(const float* mel, int n_frames);

    // The loaded model's front-end configuration, for shared-front-end
    // compatibility checks. Requires load().
    const MelConfig& mel_config() const;

    // Detector-policy setters. Cheap, thread-safe relative to feed(): the
    // values are read (relaxed) at the top of every frame. A change takes
    // effect on the next frame, not retroactively over the smoothing window.
    void set_threshold(float t);
    void set_smoothing(int hits, int window);
    void set_refractory_ms(int ms);

    // Most recent per-frame score — sigmoid(logit) in [0, 1]. 0 before the
    // first frame has been processed. Useful for offline calibration and
    // for surfacing a confidence value to the caller.
    float last_score() const;

    // Drop all streaming state — feature ring, conv-state cache, smoothing
    // history, refractory counter. The loaded weights are kept. Call this on
    // a long silence boundary if the caller wants a clean restart.
    void reset();

    const WakeConfig& config() const;
    bool              loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
