#pragma once

// ─── ListenHost — the shared listening front-ends behind bro.listen ─────────
//
// A ListenStream is one independent listening pipeline: a single audio SOURCE
// (the microphone, the system-audio render mix, or one application's audio) →
// one raw (no-AGC) 16 kHz ring → one brosoundml::ListenBus (PCEN mel front-end)
// → up to one each of {SensorHub, PhonemeSpotter, WakeWord, GestureSpotter}
// attached as members. One feature pass, one forward per attached model, N
// listeners — all hearing THAT source.
//
// Multiple streams run concurrently and independently: a mic stream driving
// wake/kws for voice commands can run alongside a system-audio stream driving
// streaming STT for the audio the machine is playing, with NO mixing — each is
// its own source, ring, bus, retention and tenant set. Streams never share
// audio; spinning up one per channel is how you listen to L/R separately.
//
// Threading / membership: each stream's bus is single-producer and is ONLY
// touched from its inference pump. A membership change replaces the whole pump
// (removePump + addPump through the host's InferenceScheduler, api.h): the
// scheduler applies both, in order, on its worker between drains, so the old
// closure never overlaps the new one and the main thread never touches the
// bus. The ring has exactly one producer — the mic-tap callback OR the
// LoopbackCapture callback — so SPSC holds. No locks anywhere. Streams are
// created/destroyed on the main thread.
//
// onSpots/onWake/onGestures run on the INFERENCE thread (or inline, from a
// headless feed); tenants publish into their own SPSC delivery from there.

#include <broaudio/loopback_capture.h>
#include <broaudio/mic_tap.h>
#include <brosoundml/listen_bus.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace brosoundml::api {

using ListenSpotsFn =
    std::function<void(const std::vector<brosoundml::SpotEvent>&)>;
// Invoked on the inference thread after EVERY bus feed while a wake member is
// attached (not only on fires) so the tenant can publish per-block score
// telemetry alongside the fire flag.
using ListenWakeFn = std::function<void(bool fired)>;
// Invoked on the inference thread after a bus feed when a gesture member is
// attached and at least one gesture fired this block.
using ListenGesturesFn =
    std::function<void(const std::vector<brosoundml::GestureEvent>&)>;

// What a stream listens to.
struct ListenSource {
    enum class Kind { Mic, SystemLoopback, ProcessLoopback };
    Kind          kind    = Kind::Mic;
    std::uint32_t pid     = 0;      // ProcessLoopback target
    bool          exclude = false;  // ProcessLoopback: capture all EXCEPT the tree
    int           channel = -1;     // loopback: -1 downmix all; >=0 pick one channel
};

using StreamId = std::uint32_t;
inline constexpr StreamId kInvalidStream = 0;

struct ListenRetentionInfo {
    bool         active = false;
    int          seconds = 0;
    int          rate = 0;
    int          hop = 0;
    std::int64_t streamFrame = 0;
    std::int64_t heldFrames = 0;
};

// ─── Host lifecycle ─────────────────────────────────────────────────────────

// Close every stream and detach all members (realm / engine teardown). The
// audio engine and scheduler set through api.h stay set.
void shutdownListenHost();

// Is the host wired to a broaudio engine (live sources possible)?
bool listenHostAudioAvailable();
// Is the host's scheduler a worker thread (feeds go through the live ring)?
// False without a scheduler or under a headless inline stepper.
bool listenHostThreaded();
// Is live mic capture running on the audio engine? A scripted feed() refuses
// to run then (two producers on one ring).
bool listenHostMicCapturing();

// ─── Streams ───────────────────────────────────────────────────────────────

// Open a stream on `src`. For loopback sources the capture starts immediately
// (audio flows / can be retained without any model attached). Returns
// kInvalidStream if the source is unavailable (e.g. loopback unsupported, or
// the target process is gone). Main thread only.
StreamId listenHostOpen(const ListenSource& src);

// Close a stream: detach its members, stop its source, free its infra. Safe on
// an unknown id. Main thread only.
void listenHostClose(StreamId id);

bool listenHostValid(StreamId id);

// The id of the shared default-microphone stream (the one the global tenant
// bindings target), creating it if needed. Stable for the session — the default
// mic stream is never erased, only its infra cycles with membership. Tenant
// bindings resolve `bro.kws`/`bro.wake`/… (no stream handle) to this id. Main
// thread only.
StreamId listenHostDefaultMicId();

// Is render-side (loopback / per-process) capture available on this build/OS?
bool listenHostLoopbackSupported();
// Applications currently holding a render audio session (for an app picker).
std::vector<broaudio::AudioProcess> listenHostEnumerateApps();

// ─── Per-stream member attach / detach (pass nullptr to detach) ──────────────
// Throws std::runtime_error on a front-end framing mismatch or a source that
// cannot start. Main thread only.

void listenStreamSetHub(StreamId id, std::shared_ptr<brosoundml::SensorHub> hub);
void listenStreamSetSpotter(StreamId id,
                            std::shared_ptr<brosoundml::PhonemeSpotter> spotter,
                            brotensor::Device device, ListenSpotsFn onSpots);
void listenStreamSetWake(StreamId id, std::shared_ptr<brosoundml::WakeWord> wake,
                         brotensor::Device device, ListenWakeFn onWake);
void listenStreamSetGesture(StreamId id,
                            std::shared_ptr<brosoundml::GestureSpotter> gesture,
                            ListenGesturesFn onGestures);

// The stream's mic-tap stats — for the tenants' stats() surfaces. False when
// the stream has no tap (a loopback stream, infra down, no audio engine).
bool listenStreamTapStats(StreamId id, broaudio::MicTapStats& out);

// Manual feed for tests / scripted scenarios. Threaded: write the shared ring.
// Headless: run the bus synchronously on this thread for ALL attached members
// of this stream and return the result.
void listenStreamWriteRing(StreamId id, const float* samples, int n);
brosoundml::ListenFeedResult listenStreamFeedInline(StreamId id,
                                                    const float* samples, int n);
// Feed the stream, letting the host pick the path: threaded → write the live
// ring (events surface via member callbacks, returns empty); headless/no worker
// → run the bus synchronously now and return the result. The handle's .feed().
brosoundml::ListenFeedResult listenStreamFeed(StreamId id,
                                              const float* samples, int n);

// ─── Stream retention (opt-in) ──────────────────────────────────────────────
//
// A ring of the RAW samples driving a stream, so recent audio can be scrubbed /
// replayed by frame range (e.g. to hear exactly what a match fired on). Source-
// agnostic: it captures whatever feeds the stream — a mic tap, a scripted
// feed(), or a system-audio / per-process loopback source. Off by default (no
// memory cost until enabled).
//
// Single-producer writer (the feed thread, at the bus chokepoint); the reader
// (main thread) copies a behind-the-head window, race-free without a lock
// because a slot is only overwritten after a full retention period.
//
// Frame axis: samples-consumed / hop — the same axis bro.sense reports. A fresh
// stream restarts the axis at 0.

void listenStreamSetRetention(StreamId id, int seconds);
std::int64_t listenStreamFrame(StreamId id);
int listenStreamReadAudio(StreamId id, std::int64_t startFrame,
                          std::int64_t endFrame, std::vector<float>& out);
ListenRetentionInfo listenStreamRetentionInfo(StreamId id);

}  // namespace brosoundml::api
