// bro.gesture — non-speech acoustic gesture spotting (rhythm / tone) over the
// listen host. The matcher rides the stream's SensorHub snapshot, so it needs
// that stream's bro.sense active to fire. bro.gesture targets the default mic
// stream, stream.gesture any other ListenStream; each stream enrolls its own
// gestures. Fires cross inference → main through a per-tenant SPSC ring the
// frame pump drains into onGesture(name, confidence, kind, span).

#include "soundml_listen_internal.h"

#include <brosoundml/gesture_spotter.h>
#include <brotensor/runtime.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brosoundml::api {

namespace {

constexpr std::uint64_t kEventSlots = 64;

// ─── One stream's gesture tenant ─────────────────────────────────────────────
//
// Address-stable (held by unique_ptr in g_gesture.tenants), so the inference-
// thread onGestures closure can capture a raw GestureTenant* and publish into it.
struct GestureTenant {
    StreamId streamId = kInvalidStream;

    std::shared_ptr<brosoundml::GestureSpotter> spotter;

    std::vector<std::string> names;   // listen()-time snapshot; index i = idx i
    ev::Persistent           onGesture;

    int                        eventIdx[kEventSlots]   = {};
    float                      eventConf[kEventSlots]  = {};
    std::uint8_t               eventTone[kEventSlots]  = {};   // 1 = tone, 0 = rhythm
    std::int64_t               eventStart[kEventSlots] = {};   // matched-span frames
    std::int64_t               eventEnd[kEventSlots]   = {};   //   (SensorHub frames axis)
    std::atomic<std::uint64_t> produced{0};
    std::atomic<std::uint64_t> drained{0};

    bool listening = false;
};

struct GestureNamespace {
    std::unordered_map<StreamId, std::unique_ptr<GestureTenant>> tenants;
};

GestureNamespace g_gesture;

// A per-stream view object: `stream.gesture`. Carries only the stream id.
struct GestureView {
    StreamId streamId = kInvalidStream;
};

HostClass g_gestureViewClass;

// Overlay gesture-policy keys: tempoTol, pitchTol, pitchStabilityTol, shapeTol,
// refractoryFrames, minOnsets, minToneFrames, onsetSigFrames.
void readPolicy(Value obj, brosoundml::GestureConfig& cfg) {
    if (!ev::isObject(obj)) return;
    ev::Persistent root(obj);
    getFloatOpt(root.get(), "tempoTol", cfg.tempo_tol);
    getFloatOpt(root.get(), "pitchTol", cfg.pitch_tol);
    getFloatOpt(root.get(), "pitchStabilityTol", cfg.pitch_stability_tol);
    getFloatOpt(root.get(), "shapeTol", cfg.shape_tol);
    getIntOpt(root.get(), "refractoryFrames", cfg.refractory_frames);
    getIntOpt(root.get(), "minOnsets",        cfg.min_onsets);
    getIntOpt(root.get(), "minToneFrames",    cfg.min_tone_frames);
    getIntOpt(root.get(), "onsetSigFrames",   cfg.onset_sig_frames);
}

void publishEvent(GestureTenant* t, int nameIdx, float confidence, bool isTone,
                  std::int64_t startFrame, std::int64_t endFrame) {
    const std::uint64_t p = t->produced.load(std::memory_order_relaxed);
    if (p - t->drained.load(std::memory_order_acquire) >= kEventSlots) return;
    t->eventIdx[p % kEventSlots]   = nameIdx;
    t->eventConf[p % kEventSlots]  = confidence;
    t->eventTone[p % kEventSlots]  = isTone ? 1u : 0u;
    t->eventStart[p % kEventSlots] = startFrame;
    t->eventEnd[p % kEventSlots]   = endFrame;
    t->produced.store(p + 1, std::memory_order_release);
}

int nameIndexOf(const std::vector<std::string>& names, const std::string& n) {
    for (std::size_t i = 0; i < names.size(); ++i)
        if (names[i] == n) return static_cast<int>(i);
    return -1;
}

const char* kindName(brosoundml::GestureKind k) {
    return k == brosoundml::GestureKind::Tone ? "tone" : "rhythm";
}

// ─── Tenant registry ─────────────────────────────────────────────────────────

GestureTenant* findTenant(StreamId id) {
    auto it = g_gesture.tenants.find(id);
    return it == g_gesture.tenants.end() ? nullptr : it->second.get();
}

GestureTenant* ensureTenant(StreamId id) {
    if (id == kInvalidStream) return nullptr;
    if (GestureTenant* t = findTenant(id)) return t;
    auto t = std::make_unique<GestureTenant>();
    t->streamId = id;
    GestureTenant* p = t.get();
    g_gesture.tenants[id] = std::move(t);
    return p;
}

// Lazily create the tenant's spotter on first enroll.
brosoundml::GestureSpotter& ensureSpotter(GestureTenant* t) {
    if (!t->spotter) t->spotter = std::make_shared<brosoundml::GestureSpotter>();
    return *t->spotter;
}

void stopListening(GestureTenant* t) {
    if (!t->listening) return;
    listenStreamSetGesture(t->streamId, nullptr, nullptr);
    t->onGesture = ev::Persistent();
    t->produced.store(0, std::memory_order_relaxed);
    t->drained.store(0, std::memory_order_relaxed);
    t->names.clear();
    t->listening = false;
}

bool refuseWhileListening(const GestureTenant* t, const char* what, Value& thrown) {
    if (!t || !t->listening) return false;
    thrown = ev::throwError(std::string("bro.gesture.") + what +
        ": not allowed while this stream is listening (enroll/remove/clear/reset "
        "share the matcher's feed thread — stop() first)");
    return true;
}

// The stream `this` addresses: a GestureView → its stream; bro.gesture →
// default mic.
StreamId streamOf(Value self) {
    if (g_gestureViewClass.isInstance(self)) {
        auto* v = static_cast<GestureView*>(g_gestureViewClass.unwrap(self));
        if (v) return v->streamId;
    }
    return listenHostDefaultMicId();
}

// ─── JS-callable functions ───────────────────────────────────────────────────

// enrollFromAudio(name, samples, policy?) -> beats
//   samples: Float32Array mono PCM at sampleRate(). Runs the clip through a
//   private SensorHub and stores the extracted rhythm/tone template on THIS
//   stream's spotter.
Value jsEnrollFromAudio(Value self, std::span<const Value> a) {
    GestureTenant* t = ensureTenant(streamOf(self));
    if (!t) return ev::throwError("bro.gesture.enrollFromAudio: no stream");
    Value thrown;
    if (refuseWhileListening(t, "enrollFromAudio", thrown)) return thrown;
    if (a.size() < 2 || !isStringArg(a, 0))
        return ev::throwTypeError(
            "bro.gesture.enrollFromAudio(name, samples, policy?): name and samples required");
    const std::string name = strAt(a, 0);
    bool ok = false;
    std::vector<float> samples = readPcmArg(a, 1, ok);
    if (!ok || samples.empty())
        return ev::throwTypeError("bro.gesture.enrollFromAudio: samples must be a non-empty Float32Array");
    try {
        brosoundml::GestureSpotter& g = ensureSpotter(t);
        brosoundml::GestureConfig pol = g.config();
        const bool hasPol = isObjectArg(a, 2);
        if (hasPol) readPolicy(a[2], pol);
        const int beats = g.enroll_from_audio(name, samples.data(),
                                              static_cast<int>(samples.size()),
                                              hasPol ? &pol : nullptr);
        return ev::fromDouble(beats);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.gesture.enrollFromAudio: ") + e.what());
    }
}

Value jsRemove(Value self, std::span<const Value> a) {
    GestureTenant* t = findTenant(streamOf(self));
    if (!t || !t->spotter) return ev::throwError("bro.gesture.remove: nothing enrolled");
    Value thrown;
    if (refuseWhileListening(t, "remove", thrown)) return thrown;
    if (!isStringArg(a, 0)) return ev::throwTypeError("bro.gesture.remove(name): name required");
    return ev::fromBool(t->spotter->remove(strAt(a, 0)));
}

Value jsClear(Value self, std::span<const Value>) {
    GestureTenant* t = findTenant(streamOf(self));
    if (!t || !t->spotter) return ev::undefined();
    Value thrown;
    if (refuseWhileListening(t, "clear", thrown)) return thrown;
    t->spotter->clear();
    return ev::undefined();
}

Value jsTemplates(Value self, std::span<const Value>) {
    GestureTenant* t = findTenant(streamOf(self));
    if (!t) return makeStringArray({});
    return makeStringArray(t->listening ? t->names
                           : (t->spotter ? t->spotter->templates() : std::vector<std::string>{}));
}

// inspect(name) -> { name, kind, frameMs, intervalsMs:[...], onsets:[...],
//   toneHz, toneMs, toneSpread } or null. The legible view of an enrolled
//   gesture on THIS stream.
Value jsInspect(Value self, std::span<const Value> a) {
    GestureTenant* t = findTenant(streamOf(self));
    if (!t || !t->spotter) return ev::null();
    if (!isStringArg(a, 0)) return ev::throwTypeError("bro.gesture.inspect(name): name required");
    brosoundml::GestureView v;
    if (!t->spotter->inspect(strAt(a, 0), v)) return ev::null();

    ObjectBuilder o;
    o.set("name", v.name);
    o.set("kind", kindName(v.kind));
    o.set("frameMs", static_cast<double>(v.frame_ms));
    {
        ev::Persistent arr(hostArrayOf(v.intervals.size(), [&](size_t i) {
            return ev::fromDouble(static_cast<double>(v.intervals[i]) * v.frame_ms);
        }));
        o.set("intervalsMs", arr.get());
    }
    {
        ev::Persistent onsets(hostArrayOf(v.onsets.size(), [&](size_t i) {
            ObjectBuilder b;
            b.set("voiced", static_cast<double>(v.onsets[i].voiced));
            b.set("pitchHz", static_cast<double>(v.onsets[i].pitch));
            b.set("bright", static_cast<double>(v.onsets[i].bright));
            return b.get();
        }));
        o.set("onsets", onsets.get());
    }
    o.set("toneHz", static_cast<double>(v.tone_hz));
    o.set("toneMs", static_cast<double>(v.tone_frames) * v.frame_ms);
    o.set("toneSpread", static_cast<double>(v.tone_spread));
    return o.get();
}

Value jsReset(Value self, std::span<const Value>) {
    GestureTenant* t = findTenant(streamOf(self));
    if (!t || !t->spotter) return ev::undefined();
    Value thrown;
    if (refuseWhileListening(t, "reset", thrown)) return thrown;
    t->spotter->reset();
    return ev::undefined();
}

// listen({ onGesture }) — start matching on THIS stream. Requires at least one
// enrolled gesture; needs the stream's bro.sense active to actually fire (the
// matcher reads the SensorHub snapshot).
Value jsListen(Value self, std::span<const Value> a) {
    const StreamId sid = streamOf(self);
    GestureTenant* t = ensureTenant(sid);
    if (!isObjectArg(a, 0))
        return ev::throwTypeError("bro.gesture.listen(opts): opts object required");
    ev::Persistent cb = getFunctionOpt(a[0], "onGesture");
    if (!ev::isFunction(cb.get()))
        return ev::throwTypeError("bro.gesture.listen: opts.onGesture (function) required");
    if (!t || !t->spotter)
        return ev::throwError("bro.gesture.listen: enroll a gesture first");
    if (t->listening)
        return ev::throwError("bro.gesture.listen: this stream is already listening (stop() first)");
    if (!listenHostAudioAvailable())
        return ev::throwError("bro.gesture.listen: audio engine not available");
    const std::vector<std::string> names = t->spotter->templates();
    if (names.empty())
        return ev::throwError(
            "bro.gesture.listen: no gestures enrolled on this stream (enrollFromAudio first)");
    try {
        t->names     = names;
        t->onGesture = cb;
        t->produced.store(0, std::memory_order_relaxed);
        t->drained.store(0, std::memory_order_relaxed);

        listenStreamSetGesture(
            sid, t->spotter,
            [t, names](const std::vector<brosoundml::GestureEvent>& events) {
                for (const auto& ev : events) {
                    const int idx = nameIndexOf(names, ev.name);
                    if (idx >= 0)
                        publishEvent(t, idx, ev.confidence,
                                     ev.kind == brosoundml::GestureKind::Tone,
                                     ev.start_frame, ev.end_frame);
                }
            });
        t->listening = true;
        logInfo("[gesture] listening on stream " + std::to_string(sid) + " (" +
                std::to_string(names.size()) + (names.size() == 1 ? " gesture)" : " gestures)"));
        return ev::undefined();
    } catch (const std::exception& e) {
        t->listening = true;
        stopListening(t);
        return ev::throwError(std::string("bro.gesture.listen: ") + e.what());
    }
}

Value jsStop(Value self, std::span<const Value>) {
    if (GestureTenant* t = findTenant(streamOf(self))) stopListening(t);
    return ev::undefined();
}

Value jsIsActive(Value self, std::span<const Value>) {
    GestureTenant* t = findTenant(streamOf(self));
    return ev::fromBool(t && t->listening);
}

Value jsSampleRate(Value self, std::span<const Value>) {
    GestureTenant* t = findTenant(streamOf(self));
    return ev::fromDouble((t && t->spotter) ? t->spotter->sample_rate()
                                            : brosoundml::GestureConfig{}.sensor.mel.sample_rate);
}

// Drain one tenant's gesture event ring into its onGesture callback (main
// thread). Re-resolves the tenant after every call.
void drainTenant(StreamId id) {
    GestureTenant* t = findTenant(id);
    if (!t || !t->listening) return;
    const std::uint64_t produced = t->produced.load(std::memory_order_acquire);
    std::uint64_t drained = t->drained.load(std::memory_order_relaxed);
    if (drained >= produced || !ev::isFunction(t->onGesture.get())) return;
    while (drained < produced) {
        const int   idx  = t->eventIdx[drained % kEventSlots];
        const float conf = t->eventConf[drained % kEventSlots];
        const bool  tone = t->eventTone[drained % kEventSlots] != 0u;
        const std::int64_t startF = t->eventStart[drained % kEventSlots];
        const std::int64_t endF   = t->eventEnd[drained % kEventSlots];
        drained++;
        t->drained.store(drained, std::memory_order_release);
        if (idx < 0 || idx >= static_cast<int>(t->names.size())) continue;
        // 4th arg: the matched span on the SensorHub frames axis (align with
        // bro.sense.snapshot().frames). Backward-compatible with
        // onGesture(name, conf, kind) handlers.
        ev::Persistent cb(t->onGesture);
        ev::Persistent name(ev::fromUtf8(t->names[static_cast<std::size_t>(idx)]));
        ev::Persistent kind(ev::fromUtf8(tone ? "tone" : "rhythm"));
        ev::Persistent span(makeSpan(startF, endF));
        const Value args[4] = {name.get(), ev::fromDouble(conf), kind.get(), span.get()};
        callTenantCallback(cb, std::span<const Value>(args, 4), "gesture");
        t = findTenant(id);
        if (!t || !t->listening) break;
    }
}

void defineGestureOps(ObjectBuilder& b) {
    b.def("enrollFromAudio", 3, jsEnrollFromAudio);
    b.def("remove",          1, jsRemove);
    b.def("clear",           0, jsClear);
    b.def("templates",       0, jsTemplates);
    b.def("inspect",         1, jsInspect);
    b.def("reset",           0, jsReset);
    b.def("listen",          1, jsListen);
    b.def("stop",            0, jsStop);
    b.def("isActive",        0, jsIsActive);
    b.def("sampleRate",      0, jsSampleRate);
}

}  // namespace

Value makeGestureView(StreamId id) {
    auto* v = new GestureView{id};
    return g_gestureViewClass.make(v, [](void* p) { delete static_cast<GestureView*>(p); });
}

void tickGesture() {
    std::vector<StreamId> ids;
    ids.reserve(g_gesture.tenants.size());
    for (const auto& kv : g_gesture.tenants) ids.push_back(kv.first);
    for (StreamId id : ids) {
        auto it = g_gesture.tenants.find(id);
        if (it == g_gesture.tenants.end()) continue;
        // Prune a tenant whose stream has closed (handle .close()'d or GC'd).
        // The stream's teardown removed its pump (a barrier), so the
        // onGestures closure can no longer run — safe to drop. Default mic is
        // never invalid.
        if (!listenHostValid(id)) {
            stopListening(it->second.get());   // detach is a no-op — stream gone
            g_gesture.tenants.erase(it);
            continue;
        }
        drainTenant(id);
    }
}

void cleanupGesture() {
    for (auto& kv : g_gesture.tenants) stopListening(kv.second.get());
    g_gesture.tenants.clear();
}

void installGesture(ObjectBuilder& bro) {
    g_gestureViewClass.install("GestureStreamView", 0, nullptr,
        [](ObjectBuilder& b) {
            b.accessor("active", [](Value self, std::span<const Value>) -> Value {
                GestureTenant* t = findTenant(streamOf(self));
                return ev::fromBool(t && t->listening);
            });
            defineGestureOps(b);
        },
        /*global=*/false);

    ObjectBuilder ges;
    ges.def("init", 0, [](Value, std::span<const Value>) -> Value {
        try {
            brotensor::init();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("bro.gesture.init: ") + e.what());
        }
        return ev::undefined();
    });
    defineGestureOps(ges);
    ges.set("GestureStreamView", g_gestureViewClass.constructor());
    bro.set("gesture", ges.get());
}

}  // namespace brosoundml::api
