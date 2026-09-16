#pragma once

#include <functional>
#include <string>

namespace brosoundml::api {

/// Mounts the 9 audio/voice AI subsystems onto `bro` (once per realm):
///   - bro.stt (Whisper, Parakeet-TDT, Qwen3-ASR + streaming encoder)
///   - bro.tts (Kokoro, Qwen3-TTS, OmniVoice, Supertonic, SpeakerEncoder)
///   - bro.diar (streaming Sortformer, ClusterDiarizer)
///   - bro.rave (RAVE neural audio autoencoder)
///   - bro.wake / bro.kws (wake word, open-vocabulary keyword spotting)
///   - bro.sense / bro.gesture / bro.listen (sensor hub, gesture spotter,
///     N-stream listen bus)
void installSoundML();

/// Pump the async jobs (background loads / transcriptions / syntheses):
/// drains streamed tokens and steps to their JS callbacks and fires onDone /
/// onReady for finished jobs. The host calls this once per frame on the
/// thread that owns the realm; a headless advanceTime() counts as a frame.
void tickSoundML();

/// Cancel + join every in-flight job and release its rooted callbacks. Call
/// at realm teardown, before the runtime and brotensor go away.
void shutdownSoundML();

/// Resolve model / asset paths the way the host's fs module does (app-
/// relative base paths). Identity when unset.
void setPathResolver(std::function<std::string(const std::string&)> resolver);

/// Route the bindings' [INFO] lines ("Whisper loaded on CUDA") to the host's
/// logger. stderr when unset.
void setLogHook(std::function<void(const std::string&)> hook);

} // namespace brosoundml::api

using brosoundml::api::installSoundML;
