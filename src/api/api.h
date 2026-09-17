#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace broaudio {
class Engine;
}

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

/// The compute half of installSoundML for a realm on a thread that is NOT
/// the host's main thread (a Worker): bro.stt, bro.tts, bro.diar and
/// bro.rave — models that read files and run inference on the calling
/// thread's own async jobs — and nothing that taps the audio engine or the
/// inference scheduler (wake / kws / sense / gesture / listen stay main-
/// thread only). The class objects are process-global but what they hold is
/// per thread (host_class.h), so a Worker realm installs its own. Pair with
/// tickSoundMLAsync() from that thread's loop.
void installSoundMLCompute();

/// Pump the async jobs (background loads / transcriptions / syntheses):
/// drains streamed tokens and steps to their JS callbacks and fires onDone /
/// onReady for finished jobs. The host calls this once per frame on the
/// thread that owns the realm; a headless advanceTime() counts as a frame.
void tickSoundML();

/// tickSoundML minus the listen tenants: only the calling thread's async
/// jobs, which is all a Worker realm has. Per thread — a job's callbacks are
/// Persistents of the realm that launched it and are delivered there.
void tickSoundMLAsync();

/// Cancel + join every in-flight job and release its rooted callbacks. Call
/// at realm teardown, before the runtime and brotensor go away.
void shutdownSoundML();

/// Resolve model / asset paths the way the host's fs module does (app-
/// relative base paths). Identity when unset.
void setPathResolver(std::function<std::string(const std::string&)> resolver);

/// Route the bindings' [INFO] lines ("Whisper loaded on CUDA") to the host's
/// logger. stderr when unset.
void setLogHook(std::function<void(const std::string&)> hook);

/// The host's broaudio engine, for the listen host's live sources: a mic
/// stream installs a raw 16 kHz tap on it, and stats() reads that tap. Null
/// (the default) leaves scripted feed() working and makes listen()/start()
/// on a live source throw "audio engine not available". The pointer must
/// outlive every open stream — call shutdownSoundML() before the engine
/// dies.
void setAudioEngine(broaudio::Engine* engine);

/// Where the listen host's per-stream feed runs. A host with an audio-
/// inference worker supplies it here: `addPump` registers a closure the
/// worker calls on every drain cycle (bro's AudioInference: every ~5 ms
/// threaded, or once per headless advanceTime step inline) and returns an
/// id; `removePump` unregisters it and, when threaded, returns only after
/// no call of that closure can still be running (a barrier — the caller
/// mutates the stream's models right after); `threaded` says which mode the
/// worker is in, deciding whether a JS feed() writes the live ring (threaded:
/// events surface later through callbacks) or runs the bus inline on the
/// calling thread (headless: deterministic, the result comes back). Unset,
/// every feed runs inline.
struct InferenceScheduler {
    std::function<std::uint32_t(std::function<void()> pump)> addPump;
    std::function<void(std::uint32_t id)>                    removePump;
    std::function<bool()>                                    threaded;
};
void setInferenceScheduler(InferenceScheduler scheduler);

} // namespace brosoundml::api

using brosoundml::api::installSoundML;
