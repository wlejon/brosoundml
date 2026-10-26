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

// ---- Access ------------------------------------------------------------------
//
// Every native that opens a path the app named asks the host's access gate
// (embed/embed_access.h, bronze::embed::checkAccess) first, on the calling JS
// thread, before anything is opened, stat'ed or queued. The gate is handed the
// path as it will be opened (after the path resolver where the native
// resolves), absolute and lexically normal, in UTF-8.
//   AccessKind::Read:
//     every model-directory loader (a directory covers what is in it):
//     bro.stt.loadWhisper / loadParakeet / loadQwenAsr / loadQwenAsrStream,
//     bro.tts.loadKokoro / loadQwen / loadOmniVoice / loadSupertonic /
//     loadSpeakerEncoder / loadHiggsCodec, bro.diar.loadSortformer and
//     loadClusterDiarizer (both directories), bro.rave.loadRave,
//     bro.music.loadYuE2 / loadVae / loadSeparator, bro.ear.loadClap (a dir
//     given; the default weights location is the library's own);
//     the file loaders: bro.stt.loadTokenizer (vocabPath, mergesPath,
//     addedTokensPath), loadParakeetTokenizer, Kokoro.loadVoice,
//     Supertonic.loadVoiceStyle, OmniVoice.loadPrompt, bro.kws.load and
//     bro.wake.load / bro.wake.listen (opts.weights, when it will be read),
//     new BcResnet2d(path | { weights }), VoiceAgent.setVad(path) and
//     setKokoro(model, voicePath); a WAV path handed as audio
//     (QwenTts.synthesizeClone's ref, ClapModel.score / embedAudio's clip);
//     bro.tts.setAssetRoot / setAssets (each path, when set; nothing is kept
//     when one is refused).
//   AccessKind::Write: OmniVoice.savePrompt and YuE2Stream.saveSong (the file
//     written).
// soundml_listen_host.cpp's live sources ask the capture gate instead.
//
// A refusal is thrown from the call as an Error carrying the gate's sentence as
// its message, code "EACCES" and `path` (the path as the gate was handed it),
// named "PermissionRevoked" for a grant taken back while the app ran; a loader
// given opts.onReady throws it too, so nothing is queued. Nothing is read or
// written.

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
/// As setAudioEngine, but asked for only when a live source first needs the
/// engine, so a host that never listens never opens its audio device.
void setAudioEngineProvider(std::function<broaudio::Engine*()> provider);

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
