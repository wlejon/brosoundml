#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace brosoundml {

// ─── CancelCheck ─────────────────────────────────────────────────────────────
//
// Cooperative cancellation for the long-running inference calls (Whisper
// transcribe, Kokoro synthesize). A call invokes it at safe checkpoints
// (between autoregressive steps / pipeline stages); returning true asks the
// call to abort early and return whatever it has so far — the caller typically
// discards a cancelled result. An empty function (the default) never cancels,
// so existing callers are unaffected. The callback runs on the calling thread
// only, so an implementation backed by a std::atomic needs no extra
// synchronisation.
using CancelCheck = std::function<bool()>;

// ─── AudioBuffer ───────────────────────────────────────────────────────────
//
// A block of mono PCM audio held as normalised FP32 samples nominally in
// [-1, 1]. AudioBuffer is brosoundml's waveform currency — the output of
// synthesis and the unit of file I/O. Kokoro emits mono 24 kHz; multi-channel
// audio is out of scope (downmixed to mono on read).
struct AudioBuffer {
    std::vector<float> samples;            // mono, FP32, nominally [-1, 1]
    int                sample_rate = 24000;

    AudioBuffer() = default;
    AudioBuffer(std::vector<float> s, int sr)
        : samples(std::move(s)), sample_rate(sr) {}

    size_t frame_count() const { return samples.size(); }
    bool   empty()       const { return samples.empty(); }

    // Playback length in seconds (0 for an empty buffer or non-positive rate).
    double duration_seconds() const;

    // Peak absolute amplitude; 0 for an empty buffer.
    float peak() const;

    // Root-mean-square amplitude; 0 for an empty buffer.
    float rms() const;

    // Scale every sample so peak() == target. No-op when the buffer is silent
    // or already at the target. Guards against clipping on file write.
    void normalize(float target = 0.99f);

    // Write a 16-bit PCM mono WAV file. Samples are clamped to [-1, 1] and
    // quantised. Throws std::runtime_error if the file cannot be opened.
    void write_wav(const std::string& path) const;
};

// Read a 16-bit PCM WAV file into a mono AudioBuffer (a stereo file is
// downmixed by averaging its channels). Throws std::runtime_error on a missing
// file or a header brosoundml does not support (non-PCM, non-16-bit).
AudioBuffer read_wav(const std::string& path);

}
