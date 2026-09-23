#pragma once

// bro.tts internals shared by native_soundml_tts*.cpp: the wrapper structs
// behind each host class, the class handles, and the marshalling every
// synthesis path shares ({ samples, sampleRate } results, trace stages,
// audio arguments).
//
// Every model is single-owner: `busy` (a ModelGate, shared with the model's
// sessions) rejects a second concurrent op, and the sync codec / prompt
// methods refuse to run while an async op holds the GPU. The gate is
// released BEFORE onDone fires so a callback can queue the next utterance.

#include "host_soundml_internal.h"

#include <brosoundml/kokoro.h>
#include <brosoundml/qwen_tts.h>
#include <brosoundml/omnivoice.h>
#include <brosoundml/supertonic.h>
#include <brosoundml/speaker_encoder.h>
#include <brosoundml/g2p/phoneme_adapter.h>

namespace brosoundml::api {

// ---- wrappers ---------------------------------------------------------------

struct HostKokoro {
    std::shared_ptr<brosoundml::Kokoro> model;
    // Lazily built on the first encodePhonemes(); borrows the model's vocab.
    std::unique_ptr<brosoundml::g2p::PhonemeAdapter> adapter;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;
};

struct HostVoice {
    brosoundml::Voice voice;
};

struct HostKokoroSession {
    std::shared_ptr<brosoundml::Kokoro> model;
    ModelGate busy;                                   // shared with the model
    brotensor::Device device = brotensor::Device::CPU;
    std::unique_ptr<brosoundml::KokoroSession> session;   // null until a voice is bound
};

struct HostQwenTts {
    std::shared_ptr<brosoundml::QwenTts> model;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;
};

struct HostQwenTtsSession {
    std::shared_ptr<brosoundml::QwenTts> model;
    ModelGate busy;
    brotensor::Device device = brotensor::Device::CPU;
    brosoundml::QwenTtsSession session;
};

struct HostOmniVoice {
    std::shared_ptr<brosoundml::OmniVoice> model;
    brotensor::Device device = brotensor::Device::CPU;
    brosoundml::OmniVoicePrecision precision = brosoundml::OmniVoicePrecision::BF16;
    bool decoderOnly = false;
    ModelGate busy;
};

struct HostSupertonic {
    std::shared_ptr<brosoundml::Supertonic> model;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;
};

struct HostSupertonicVoice {
    brosoundml::VoiceStyle style;
    std::string name;
};

struct HostSpeakerEncoder {
    std::shared_ptr<brosoundml::SpeakerEncoder> enc;
};

// ---- class handles (defined in native_soundml_tts.cpp) ---------------------

extern HostClass g_kokoroClass;
extern HostClass g_voiceClass;
extern HostClass g_kokoroSessionClass;
extern HostClass g_qwenTtsClass;
extern HostClass g_qwenTtsSessionClass;
extern HostClass g_omniVoiceClass;
extern HostClass g_supertonicClass;
extern HostClass g_supertonicVoiceClass;
extern HostClass g_speakerEncoderClass;

template <typename T>
T* unwrapTtsAs(const HostClass& cls, Value v) {
    if (!cls.isInstance(v)) return nullptr;
    return static_cast<T*>(cls.unwrap(v));
}

// ---- per-family installers --------------------------------------------------

void installTtsKokoroClasses();
void installTtsQwenClasses();
void installTtsOmniVoiceClasses();
void installTtsSupertonicClasses();
void installTtsHiggsClasses();

// bro.tts.loadHiggsCodec (native_soundml_tts_higgs.cpp) and its class handle.
Value loadHiggsCodec(Value, std::span<const Value> args);
extern HostClass g_higgsCodecClass;

// ---- dispatch entry points (bro.tts.synthesize / synthesizeStream / decodeFrom)
//
// `args` are the raw JS arguments INCLUDING the model at [0]; `modelVal` is
// the rooted model handle.

Value kokoroSynthesizeAsync(Value modelVal, HostKokoro* w, std::span<const Value> args);
Value kokoroSynthesizeStream(Value modelVal, HostKokoro* w, std::span<const Value> args);
Value kokoroDecodeFromAsync(Value modelVal, HostKokoro* w, std::span<const Value> args);
Value qwenTtsSynthesizeAsync(Value modelVal, HostQwenTts* w, std::span<const Value> args);
Value qwenTtsSynthesizeStream(Value modelVal, HostQwenTts* w, std::span<const Value> args);
Value omniVoiceLaunch(Value modelVal, Value textVal, Value optsVal, bool codesMode);
Value supertonicSynthesizeAsync(Value modelVal, HostSupertonic* w, std::span<const Value> args);

// ---- shared marshalling -----------------------------------------------------

// { samples: Float32Array, sampleRate }
inline Value makeAudioResult(const std::vector<float>& samples, int sampleRate) {
    ObjectBuilder res;
    {
        ev::Persistent s(makeFloat32Array(samples));
        res.set("samples", s.get());
    }
    res.set("sampleRate", ev::fromDouble(sampleRate));
    return res.get();
}

// [{ name, h, w, data: Float32Array }] from a Kokoro / Qwen trace's stages.
template <typename Stages>
Value makeStagesArray(const Stages& stages) {
    return hostArrayOf(stages.size(), [&](size_t i) -> Value {
        const auto& s = stages[i];
        ObjectBuilder st;
        st.set("name", s.name);
        st.set("h", ev::fromDouble(s.h));
        st.set("w", ev::fromDouble(s.w));
        ev::Persistent d(makeFloat32Array(s.data));
        st.set("data", d.get());
        return st.get();
    });
}

// (samples, sampleRate?) from (Float32Array, number), (Float32Array,
// { sampleRate }) or ({ samples, sampleRate }); the default rate is 24 kHz.
// False when no samples were read.
inline bool readAudioArgs(std::span<const Value> args, size_t at, brosoundml::AudioBuffer& out,
                          int defaultRate = 24000) {
    if (!hasArg(args, at)) return false;
    std::string err;
    if (!readAudioBuffer(args[at], out, err, defaultRate)) return false;
    if (hasArg(args, at + 1)) {
        if (ev::isNumber(args[at + 1])) {
            out.sample_rate = static_cast<int>(ev::toDouble(args[at + 1]));
        } else if (ev::isObject(args[at + 1]) && !ev::isTypedArray(args[at + 1])) {
            ev::Persistent o(args[at + 1]);
            int sr = out.sample_rate;
            getIntOpt(o.get(), "sampleRate", sr);
            out.sample_rate = sr;
        }
    }
    return !out.samples.empty() && out.sample_rate > 0;
}

// Read an optional uint64 seed: a number, or a BigInt for the full 64 bits.
inline void getSeedOpt(Value opts, std::uint64_t& dst) {
    if (!ev::isObject(opts)) return;
    Value v = ev::getProperty(opts, "seed");
    if (ev::isNumber(v)) dst = static_cast<std::uint64_t>(static_cast<int64_t>(ev::toDouble(v)));
    else if (ev::isBigInt(v)) dst = ev::toUint64(v);
}

} // namespace brosoundml::api
