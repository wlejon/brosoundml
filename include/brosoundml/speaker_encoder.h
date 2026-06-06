#pragma once

// Standalone Qwen3-TTS speaker encoder — the Base variant's ECAPA-TDNN x-vector
// extractor, lifted out of the full ~2.5 GB checkpoint so voice cloning can
// enrol a reference clip without loading the Talker + codec + tokenizer it never
// touches. The encoder is ~18 MB (≈1 % of the Base model.safetensors); loading
// just it makes enrollment effectively instant.
//
// The artifact is a small two-file directory produced offline by the
// brosoundml_build_speaker_encoder tool (tools/build_speaker_encoder.cpp) and
// housed in brosoundml-data
// (qwen-tts/speaker-encoder/{config.json, model.safetensors}). The embedding it
// produces is bit-identical to QwenTts::embed_speaker on the same clip — same
// weights, same host-side ECAPA graph — so anything fit against the full-load
// embedding (e.g. Kokoro's voice_bridge) keeps working unchanged.
//
// Enrollment is one-shot and runs host-side (CPU) regardless of where other
// models live; there is no device parameter. Heavy state (the loaded weights +
// mel frontend) lives behind a pImpl so this header stays free of brotensor
// module internals, matching the other pipeline classes.

#include "brosoundml/audio.h"

#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

class SpeakerEncoder {
public:
    SpeakerEncoder();
    ~SpeakerEncoder();
    SpeakerEncoder(SpeakerEncoder&&) noexcept;
    SpeakerEncoder& operator=(SpeakerEncoder&&) noexcept;
    SpeakerEncoder(const SpeakerEncoder&) = delete;
    SpeakerEncoder& operator=(const SpeakerEncoder&) = delete;

    // Load a standalone speaker-encoder artifact directory holding config.json
    // (carrying the `speaker_encoder_config` block) and model.safetensors (the
    // `speaker_encoder.*` tensors). Throws std::runtime_error on a missing /
    // malformed artifact.
    void load(const std::string& dir);

    // Encode a reference clip into the ECAPA-TDNN speaker x-vector — exactly the
    // enrollment QwenTts::synthesize_clone runs, on its own. `ref` is treated as
    // mono and resampled to the encoder's rate (24 kHz) as needed. Returns
    // `enc_dim` (1024) host floats — a speaker-identity embedding. Throws if no
    // artifact is loaded or `ref` is empty.
    std::vector<float> embed(const AudioBuffer& ref) const;

    bool loaded() const;
    int  enc_dim() const;       // x-vector width (1024)
    int  sample_rate() const;   // required input rate (24000); resampled to it

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
