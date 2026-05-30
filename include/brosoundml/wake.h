#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <string>

namespace brosoundml {

// ─── Wake-word detector ────────────────────────────────────────────────────
//
// A small streaming convolutional classifier that fires when a target keyword
// ("computer" is the first target) is spoken. The model is BC-ResNet-style:
// log-mel front-end → 4 depthwise-separable Conv1d residual blocks → global
// average pool over time → linear head → single logit. Designed for an
// always-on mic loop with a per-frame inference cost in the low milliseconds.
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
// have landed. load() reads the BC-ResNet weights; feed() runs the streaming
// log-mel front-end (stft → magnitude → mel → log) ▶ model ▶ 2-of-3 smoothing
// ▶ refractory debounce, returning true once per detected event.
//
// Supporting tools (built under tools/):
//   brosoundml_wake_synth     Kokoro-driven dataset builder
//   brosoundml_wake_inspect   dataset validator
//   brosoundml_wake_train     backward + Adam + BCE-with-logits
//   brosoundml_wake_test      held-out evaluation

// Front-end + model + detector hyperparameters. The defaults match the
// "computer" recipe: 16 kHz, 10 ms hop, 25 ms window, 40-bin log-mel,
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
