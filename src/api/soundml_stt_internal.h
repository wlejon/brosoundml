#pragma once

// bro.stt internals shared by native_soundml_stt*.cpp: the wrapper structs
// behind each host class, the class handles, and the streaming-job shape
// every async transcribe uses.
//
// The model + its single-owner gate are held by shared_ptr so they outlive
// the JS model handle whenever a session is still alive, and so EVERY
// inference over one model — bro.stt.transcribe(model), model.transcribe()
// AND every session.transcribe() — serializes on the ONE busy flag.
// brosoundml's session tier for these models is SHARED WEIGHTS / SERIALIZED
// decode (one GPU stream, a shared captured decoder step-graph), so
// concurrent decode over sessions of one model is unsupported and is gated
// here.

#include "host_soundml_internal.h"

#include <brosoundml/whisper.h>
#include <brosoundml/parakeet.h>
#include <brosoundml/qwen_asr.h>
#include <brolm/whisper_tokenizer.h>
#include <brolm/tokenizer_t5.h>

namespace brosoundml::api {

// ---- wrappers ---------------------------------------------------------------

struct HostWhisperTokenizer {
    std::unique_ptr<brolm::whisper::Tokenizer> tok;
};

struct HostWhisperModel {
    std::shared_ptr<brosoundml::Whisper> model;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;
};

struct HostWhisperSession {
    std::shared_ptr<brosoundml::Whisper> model;
    ModelGate busy;                           // shared with the model
    brotensor::Device device = brotensor::Device::CPU;
    brosoundml::WhisperSession session;
};

struct HostParakeetTokenizer {
    std::unique_ptr<brolm::t5::Tokenizer> tok;
};

struct HostParakeetModel {
    std::shared_ptr<brosoundml::Parakeet> model;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;
};

struct HostParakeetSession {
    std::shared_ptr<brosoundml::Parakeet> model;
    ModelGate busy;
    brotensor::Device device = brotensor::Device::CPU;
    brosoundml::ParakeetSession session;
};

struct HostQwenAsrModel {
    std::shared_ptr<brosoundml::QwenAsr> model;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;
};

struct HostQwenAsrSession {
    std::shared_ptr<brosoundml::QwenAsr> model;
    ModelGate busy;
    brotensor::Device device = brotensor::Device::CPU;
    brosoundml::QwenAsrSession session;
};

// Encoder-only streaming tap. feed()/finish() run synchronously on the JS
// thread (one encoder block is ~1 s of audio — cheap on the GPU), no gate.
struct HostQwenAsrStream {
    std::unique_ptr<brosoundml::QwenAsrStream> stream;
    brotensor::Device device = brotensor::Device::CPU;
};

// ---- class handles (defined in native_soundml_stt.cpp) ---------------------

extern HostClass g_whisperTokenizerClass;
extern HostClass g_whisperModelClass;
extern HostClass g_whisperSessionClass;
extern HostClass g_parakeetTokenizerClass;
extern HostClass g_parakeetModelClass;
extern HostClass g_parakeetSessionClass;
extern HostClass g_qwenAsrModelClass;
extern HostClass g_qwenAsrSessionClass;
extern HostClass g_qwenAsrStreamClass;

template <typename T>
T* unwrapAs(const HostClass& cls, Value v) {
    if (!cls.isInstance(v)) return nullptr;
    return static_cast<T*>(cls.unwrap(v));
}

// ---- per-family installers --------------------------------------------------

void installSttWhisperClasses();
void installSttParakeetClasses();
void installSttQwenClasses();

// ---- transcribe entry points ------------------------------------------------
//
// Each takes the already-validated model / session and the raw JS arguments
// AFTER the model (audio first). Sync (no onDone) answers the ids; async
// answers an AsyncHandle. They throw TypeErrors for bad shapes.

Value whisperTranscribe(Value modelVal, HostWhisperModel* model, HostWhisperSession* session,
                        std::span<const Value> args, const char* fn);
Value parakeetTranscribe(Value modelVal, HostParakeetModel* model, HostParakeetSession* session,
                         std::span<const Value> args, const char* fn);
Value qwenAsrTranscribe(Value modelVal, HostQwenAsrModel* model, HostQwenAsrSession* session,
                        std::span<const Value> args, const char* fn);

// ---- streaming job base -----------------------------------------------------
//
// Shared between the work thread (sole writer of the result vectors and the
// token ring) and the JS thread (sole reader / caller of onDone). Held by
// shared_ptr, captured into work, poll and done.
struct SttJobBase {
    brosoundml::AudioBuffer audio;
    int maxNew = 0;
    std::vector<int32_t> tokenIds;            // filled by work()
    ev::Persistent onDone;
    ev::Persistent onToken;
    ev::Persistent modelRef;                  // keeps the JS model handle alive
    bool hasOnToken = false;
    SpscSlots<int32_t> tokens;                // onToken handoff

    void publishToken(int32_t id) { tokens.push(id); }

    // JS thread: fire onToken for everything published so far.
    void drainTokens() {
        if (!hasOnToken) return;
        tokens.drain([this](int32_t id) {
            callCallback1(onToken.get(), ev::fromDouble(id));
        });
    }
};

inline constexpr size_t kSttTokenSlots = 1u << 16;

// Build the model TranscribeOptions' cancel check from the job's flag.
inline brosoundml::CancelCheck cancelCheckOf(const std::atomic<bool>& cancel) {
    return [&cancel] { return cancel.load(std::memory_order_acquire); };
}

} // namespace brosoundml::api
