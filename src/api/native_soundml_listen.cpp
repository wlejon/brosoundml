// bro.listen — N concurrent unmixed listening streams (mic / system loopback /
// per-app) with the tenant views attached, plus the default mic stream's
// retention surface. The host owns all audio infra and member state
// (soundml_listen_host.h); a ListenStream handle holds nothing but the id.

#include "soundml_listen_internal.h"

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml::api {

namespace {

// ─── ListenStream handle — one independent listening pipeline ────────────────
//
// `bro.listen.open(source)` returns one of these. Closing (explicitly via
// .close() or when the handle is collected) frees the stream's source + infra.
// StreamIds are monotonic (never reused), so a stale handle can never address
// a different stream — its methods just no-op once invalid. The finalizer is
// Deferred: closing a stream detaches its pump (a scheduler barrier) and
// releases its source, host work that must not run inside a collection.
struct StreamHandle {
    StreamId           id   = kInvalidStream;
    ListenSource::Kind kind = ListenSource::Kind::Mic;
    ~StreamHandle() {
        if (id != kInvalidStream) listenHostClose(id);
    }
};

HostClass g_streamClass;

const char* kindName(ListenSource::Kind k) {
    switch (k) {
        case ListenSource::Kind::Mic:             return "mic";
        case ListenSource::Kind::SystemLoopback:  return "system";
        case ListenSource::Kind::ProcessLoopback: return "process";
    }
    return "?";
}

StreamHandle* streamSelf(Value self) {
    if (!g_streamClass.isInstance(self)) return nullptr;
    return static_cast<StreamHandle*>(g_streamClass.unwrap(self));
}

int retentionSecondsArg(std::span<const Value> a) {
    return (!a.empty() && ev::isNumber(a[0])) ? static_cast<int>(ev::toDouble(a[0])) : 0;
}

// { active, seconds, rate, hop, frameRate, streamFrame, heldFrames, heldSeconds }
// — retention status for a UI / scrubber.
Value makeRetentionInfo(const ListenRetentionInfo& r) {
    ObjectBuilder o;
    o.set("active", r.active);
    o.set("seconds", static_cast<double>(r.seconds));
    o.set("rate", static_cast<double>(r.rate));
    o.set("hop", static_cast<double>(r.hop));
    o.set("frameRate", r.hop > 0 ? static_cast<double>(r.rate) / r.hop : 0.0);
    o.set("streamFrame", static_cast<double>(r.streamFrame));
    o.set("heldFrames", static_cast<double>(r.heldFrames));
    o.set("heldSeconds", r.rate > 0 ? static_cast<double>(r.heldFrames * r.hop) / r.rate : 0.0);
    return o.get();
}

// audio(startFrame, endFrame) -> Float32Array | null. The retained raw PCM for
// the inclusive frame range (frames axis == samples / hop, same as
// bro.sense.snapshot().frames). null when retention is off or the range fell
// outside the held window (too old / future).
Value readAudioRange(StreamId id, std::span<const Value> a, const char* who) {
    if (a.size() < 2 || !ev::isNumber(a[0]) || !ev::isNumber(a[1]))
        return ev::throwTypeError(std::string(who) + "(startFrame, endFrame): two frame indices required");
    const auto startF = static_cast<std::int64_t>(ev::toDouble(a[0]));
    const auto endF   = static_cast<std::int64_t>(ev::toDouble(a[1]));
    std::vector<float> out;
    const int n = listenStreamReadAudio(id, startF, endF, out);
    if (n <= 0) return ev::null();
    return makeFloat32Array(out);
}

// ─── Default-mic retention surface (bro.listen.*) ────────────────────────────

// bro.listen.retain(seconds) — enable/resize raw-audio retention to `seconds`
// of the shared stream (0 disables and frees it). Source-agnostic: captures
// whatever drives the host (mic, scripted feed, loopback). Takes effect
// immediately when the stream is live, else on the next start.
Value jsRetain(Value, std::span<const Value> a) {
    listenStreamSetRetention(listenHostDefaultMicId(), retentionSecondsArg(a));
    return ev::undefined();
}

Value jsAudio(Value, std::span<const Value> a) {
    return readAudioRange(listenHostDefaultMicId(), a, "bro.listen.audio");
}

// bro.listen.frame() -> current stream frame (total samples consumed / hop).
Value jsFrame(Value, std::span<const Value>) {
    return ev::fromDouble(static_cast<double>(listenStreamFrame(listenHostDefaultMicId())));
}

Value jsInfo(Value, std::span<const Value>) {
    return makeRetentionInfo(listenStreamRetentionInfo(listenHostDefaultMicId()));
}

// ─── ListenStream methods ────────────────────────────────────────────────────

Value jsStreamRetain(Value self, std::span<const Value> a) {
    StreamHandle* h = streamSelf(self);
    if (!h) return ev::throwTypeError("retain: not a ListenStream");
    listenStreamSetRetention(h->id, retentionSecondsArg(a));
    return ev::undefined();
}

Value jsStreamAudio(Value self, std::span<const Value> a) {
    StreamHandle* h = streamSelf(self);
    if (!h) return ev::throwTypeError("audio: not a ListenStream");
    return readAudioRange(h->id, a, "stream.audio");
}

Value jsStreamFrame(Value self, std::span<const Value>) {
    StreamHandle* h = streamSelf(self);
    if (!h) return ev::throwTypeError("frame: not a ListenStream");
    return ev::fromDouble(static_cast<double>(listenStreamFrame(h->id)));
}

Value jsStreamInfo(Value self, std::span<const Value>) {
    StreamHandle* h = streamSelf(self);
    if (!h) return ev::throwTypeError("info: not a ListenStream");
    return makeRetentionInfo(listenStreamRetentionInfo(h->id));
}

// stream.feed(Float32Array) — scripted/headless feed at the stream's rate. The
// host picks the path (threaded ring write vs inline bus run). For driving
// retention + attached models in tests; live capture writes the ring itself.
Value jsStreamFeed(Value self, std::span<const Value> a) {
    StreamHandle* h = streamSelf(self);
    if (!h) return ev::throwTypeError("feed: not a ListenStream");
    if (a.empty()) return ev::throwTypeError("stream.feed(Float32Array): samples required");
    bool ok = false;
    std::vector<float> samples = readPcmArg(a, 0, ok);
    if (!ok || samples.empty())
        return ev::throwTypeError("stream.feed: samples must be a non-empty Float32Array");
    try {
        listenStreamFeed(h->id, samples.data(), static_cast<int>(samples.size()));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("stream.feed: ") + e.what());
    }
    return ev::undefined();
}

// stream.close() — detach members, stop the source, free infra. Idempotent.
Value jsStreamClose(Value self, std::span<const Value>) {
    StreamHandle* h = streamSelf(self);
    if (!h) return ev::throwTypeError("close: not a ListenStream");
    if (h->id != kInvalidStream) {
        listenHostClose(h->id);
        h->id = kInvalidStream;
    }
    return ev::undefined();
}

// Parse a bro.listen.open() source argument into a ListenSource.
//   undefined | "mic" | {mic:true}                         -> Mic
//   "system" | {system:true}                               -> SystemLoopback
//   {process: pid} | {pid: n, exclude?: bool}              -> ProcessLoopback
//   any loopback source may carry { channel: n } (-1 downmix; >=0 pick one).
bool parseSource(Value v, ListenSource& out, std::string& err) {
    out = ListenSource{};
    if (ev::isUndefined(v) || ev::isNull(v)) return true;  // default mic
    if (ev::isString(v)) {
        const std::string sv = ev::toUtf8(v);
        if (sv == "mic")    { out.kind = ListenSource::Kind::Mic; return true; }
        if (sv == "system") { out.kind = ListenSource::Kind::SystemLoopback; return true; }
        err = "source string must be 'mic' or 'system' (use {process:pid} for an app)";
        return false;
    }
    if (!ev::isObject(v)) { err = "source must be a string or an object"; return false; }
    ev::Persistent root(v);
    const bool hasProcess = hasProperty(root.get(), "process");
    const bool hasPid     = hasProperty(root.get(), "pid");
    if (hasProcess || hasPid) {
        out.kind    = ListenSource::Kind::ProcessLoopback;
        out.pid     = static_cast<std::uint32_t>(
            getPropertyDouble(root.get(), hasProcess ? "process" : "pid", 0.0));
        out.exclude = getPropertyBool(root.get(), "exclude", false);
    } else if (getPropertyBool(root.get(), "system", false)) {
        out.kind = ListenSource::Kind::SystemLoopback;
    } else {
        out.kind = ListenSource::Kind::Mic;  // {} or {mic:true}
    }
    if (hasProperty(root.get(), "channel"))
        out.channel = getPropertyInt(root.get(), "channel", -1);
    return true;
}

// bro.listen.open(source?) -> ListenStream | throws.
// Opens an independent listening pipeline on `source`. Loopback sources start
// capturing immediately. Throws if the source is unavailable (loopback
// unsupported on this build/OS, or the target process is gone).
Value jsOpen(Value, std::span<const Value> a) {
    ListenSource src;
    std::string err;
    if (!parseSource(argAt(a, 0), src, err))
        return ev::throwTypeError("bro.listen.open: " + err);
    if (src.kind != ListenSource::Kind::Mic && !listenHostLoopbackSupported())
        return ev::throwError(
            "bro.listen.open: loopback/per-app capture not supported on this build/OS");
    if (!listenHostAudioAvailable())
        return ev::throwError("bro.listen.open: audio engine not available");

    std::string openErr;
    const StreamId id = listenHostOpen(src, &openErr);
    if (id == kInvalidStream) {
        if (!openErr.empty()) {
            return ev::throwError("bro.listen.open: " + openErr);
        }
        return ev::throwError(
            "bro.listen.open: could not open the stream (source unavailable, "
            "or the audio subsystem is not ready)");
    }

    auto* h = new StreamHandle{id, src.kind};
    ObjectBuilder wrapper(g_streamClass.make(
        h, [](void* p) { delete static_cast<StreamHandle*>(p); }, ev::Finalize::Deferred));
    // Per-stream tenant views. Each carries this stream's id and exposes a
    // listen-host model binding scoped to it: stream.kws / .wake (weights load
    // once via the namespace op; each stream attaches its own spotter/session
    // over the shared net) and the model-free stream.sense / .gesture (each
    // stream gets its own SensorHub / GestureSpotter). The globals (bro.kws,
    // bro.wake, bro.sense, bro.gesture) target the shared default-mic stream.
    { ev::Persistent v(makeKwsView(id));     wrapper.set("kws", v.get()); }
    { ev::Persistent v(makeWakeView(id));    wrapper.set("wake", v.get()); }
    { ev::Persistent v(makeSenseView(id));   wrapper.set("sense", v.get()); }
    { ev::Persistent v(makeGestureView(id)); wrapper.set("gesture", v.get()); }
    return wrapper.get();
}

// bro.listen.supported() -> bool. Is render-side (system / per-app) capture
// available on this build/OS? (Mic streams are always available.)
Value jsSupported(Value, std::span<const Value>) {
    return ev::fromBool(listenHostLoopbackSupported());
}

// bro.listen.apps() -> [{ pid, name }, ...]. Applications currently holding a
// render audio session — the candidates for {process: pid}. Empty when loopback
// is unsupported.
Value jsApps(Value, std::span<const Value>) {
    const std::vector<broaudio::AudioProcess> apps = listenHostEnumerateApps();
    return hostArrayOf(apps.size(), [&](size_t i) {
        ObjectBuilder o;
        o.set("pid", static_cast<double>(apps[i].pid));
        o.set("name", apps[i].name);
        return o.get();
    });
}

}  // namespace

void installListen(ObjectBuilder& bro) {
    g_streamClass.install("ListenStream", 0, nullptr,
        [](ObjectBuilder& b) {
            b.accessor("id", [](Value self, std::span<const Value>) -> Value {
                StreamHandle* h = streamSelf(self);
                return ev::fromDouble(h ? static_cast<double>(h->id) : 0.0);
            });
            b.accessor("kind", [](Value self, std::span<const Value>) -> Value {
                StreamHandle* h = streamSelf(self);
                return ev::fromUtf8(h ? kindName(h->kind) : "?");
            });
            b.accessor("valid", [](Value self, std::span<const Value>) -> Value {
                StreamHandle* h = streamSelf(self);
                return ev::fromBool(h && listenHostValid(h->id));
            });
            b.def("retain", 1, jsStreamRetain);
            b.def("audio",  2, jsStreamAudio);
            b.def("frame",  0, jsStreamFrame);
            b.def("info",   0, jsStreamInfo);
            b.def("feed",   1, jsStreamFeed);
            b.def("close",  0, jsStreamClose);
        },
        /*global=*/false);

    ObjectBuilder listen;
    listen.def("open",      1, jsOpen);
    listen.def("supported", 0, jsSupported);
    listen.def("apps",      0, jsApps);
    listen.def("retain",    1, jsRetain);
    listen.def("audio",     2, jsAudio);
    listen.def("frame",     0, jsFrame);
    listen.def("info",      0, jsInfo);
    listen.set("ListenStream", g_streamClass.constructor());
    bro.set("listen", listen.get());
}

}  // namespace brosoundml::api
