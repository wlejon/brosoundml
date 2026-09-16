#include "host_soundml_internal.h"

#include <brosoundml/sensor_hub.h>
#include <brosoundml/gesture_spotter.h>
#include <brosoundml/listen_bus.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml::api {

namespace {

HostClass g_senseViewClass;
HostClass g_gestureViewClass;
HostClass g_listenStreamClass;

struct HostSenseState {
    std::shared_ptr<brosoundml::SensorHub> hub;
    bool active = false;
    int sampleRate = 16000;
    uint64_t framesDelivered = 0;
    uint64_t samplesDelivered = 0;
    double rollingPeak = 0.0;
};

struct HostGestureState {
    std::shared_ptr<brosoundml::GestureSpotter> spotter;
    bool active = false;
    int sampleRate = 16000;
    std::vector<std::string> templates;
};

struct HostListenStream {
    uint32_t id = 1;
    std::string kind = "mic";
    bool valid = true;
    double retentionSeconds = 0.0;
    uint64_t currentFrame = 0;
    std::vector<float> audioBuffer;

    ev::Persistent wakeVal;
    ev::Persistent kwsVal;
    ev::Persistent senseVal;
    ev::Persistent gestureVal;
};

static uint32_t s_nextStreamId = 1;
static HostSenseState g_defaultSense;
static HostGestureState g_defaultGesture;
static double g_globalRetention = 0.0;

Value makeSenseSnapshot(const HostSenseState& s) {
    ObjectBuilder snap;
    snap.set("frames", static_cast<double>(s.framesDelivered));
    snap.set("t", static_cast<double>(s.framesDelivered) * 0.01);
    snap.set("rms", s.rollingPeak * 0.707);
    snap.set("peak", s.rollingPeak);
    snap.set("db", s.rollingPeak > 1e-6 ? 20.0 * std::log10(s.rollingPeak) : -100.0);
    snap.set("voice", false);
    snap.set("noiseFloorDb", -60.0);
    snap.set("snrDb", 0.0);
    snap.set("voiceFrames", 0.0);
    snap.set("voiceEvents", 0.0);
    snap.set("lastVoiceFrame", 0.0);
    snap.set("flux", 0.0);
    snap.set("onset", false);
    snap.set("onsets", 0.0);
    snap.set("lastOnsetFrame", 0.0);
    snap.set("periodicity", 0.0);
    snap.set("dominantHz", 0.0);
    snap.set("tonal", false);
    snap.set("tonalFrames", 0.0);
    snap.set("tonalEvents", 0.0);
    snap.set("lastTonalFrame", 0.0);
    snap.set("centroid", 0.0);
    return snap.get();
}

Value makeSenseStats(const HostSenseState& s) {
    if (!s.active) return ev::null();
    ObjectBuilder st;
    st.set("framesDelivered", static_cast<double>(s.framesDelivered));
    st.set("samplesDelivered", static_cast<double>(s.samplesDelivered));
    st.set("rollingPeak", s.rollingPeak);
    return st.get();
}

Value makeRetentionInfo(double seconds, uint64_t currentFrame) {
    ObjectBuilder info;
    info.set("active", seconds > 0.0);
    info.set("seconds", seconds);
    info.set("rate", 16000.0);
    info.set("hop", 160.0);
    info.set("frameRate", 100.0);
    info.set("streamFrame", static_cast<double>(currentFrame));
    info.set("heldFrames", static_cast<double>(seconds * 100.0));
    info.set("heldSeconds", seconds);
    return info.get();
}

void feedSense(HostSenseState& s, const std::vector<float>& samples) {
    if (samples.empty()) return;
    s.samplesDelivered += samples.size();
    s.framesDelivered += samples.size() / 160;
    float pk = 0.0f;
    for (float f : samples) {
        float af = std::abs(f);
        if (af > pk) pk = af;
    }
    s.rollingPeak = pk;
}

} // namespace

Value createSenseStreamViewHandle() {
    auto* s = new HostSenseState();
    return g_senseViewClass.make(s, [](void* p) { delete static_cast<HostSenseState*>(p); });
}

Value createGestureStreamViewHandle() {
    auto* s = new HostGestureState();
    return g_gestureViewClass.make(s, [](void* p) { delete static_cast<HostGestureState*>(p); });
}

void installListenSenseGesture(ObjectBuilder& bro) {
    // -----------------------------------------------------------------------
    // SenseStreamView
    // -----------------------------------------------------------------------
    g_senseViewClass.install("SenseStreamView", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* s = new HostSenseState();
            return g_senseViewClass.make(s, [](void* p) { delete static_cast<HostSenseState*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("active", [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostSenseState*>(g_senseViewClass.unwrap(self));
                return ev::fromBool(s && s->active);
            });
            b.def("start", 1, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostSenseState*>(g_senseViewClass.unwrap(self));
                if (s) s->active = true;
                return ev::undefined();
            });
            b.def("stop", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostSenseState*>(g_senseViewClass.unwrap(self));
                if (s) s->active = false;
                return ev::undefined();
            });
            b.def("isActive", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostSenseState*>(g_senseViewClass.unwrap(self));
                return ev::fromBool(s && s->active);
            });
            b.def("snapshot", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostSenseState*>(g_senseViewClass.unwrap(self));
                return s ? makeSenseSnapshot(*s) : ev::null();
            });
            b.def("sampleRate", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostSenseState*>(g_senseViewClass.unwrap(self));
                return ev::fromDouble(s ? s->sampleRate : 16000);
            });
            b.def("stats", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostSenseState*>(g_senseViewClass.unwrap(self));
                return s ? makeSenseStats(*s) : ev::null();
            });
            b.def("feed", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostSenseState*>(g_senseViewClass.unwrap(self));
                auto samples = extractFloatAudio(argAt(a, 0));
                if (s) feedSense(*s, samples);
                return s ? makeSenseSnapshot(*s) : ev::null();
            });
            b.def("analyze", 2, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostSenseState*>(g_senseViewClass.unwrap(self));
                auto samples = extractFloatAudio(argAt(a, 0));
                int frames = static_cast<int>(samples.size() / 160);
                ObjectBuilder an;
                an.set("frames", static_cast<double>(frames));
                an.set("hop", 160.0);
                an.set("win", 320.0);
                an.set("rate", s ? static_cast<double>(s->sampleRate) : 16000.0);
                an.set("frameMs", 10.0);
                std::vector<float> dbVec(frames, -60.0f);
                std::vector<float> hzVec(frames, 0.0f);
                std::vector<float> perVec(frames, 0.0f);
                std::vector<float> cenVec(frames, 0.0f);
                std::vector<int32_t> flagsVec(frames, 0);
                an.set("db", makeFloat32Array(dbVec));
                an.set("dominantHz", makeFloat32Array(hzVec));
                an.set("periodicity", makeFloat32Array(perVec));
                an.set("centroid", makeFloat32Array(cenVec));
                an.set("flags", makeInt32Array(flagsVec));
                return an.get();
            });
        });

    // -----------------------------------------------------------------------
    // GestureStreamView
    // -----------------------------------------------------------------------
    g_gestureViewClass.install("GestureStreamView", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* s = new HostGestureState();
            return g_gestureViewClass.make(s, [](void* p) { delete static_cast<HostGestureState*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("active", [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostGestureState*>(g_gestureViewClass.unwrap(self));
                return ev::fromBool(s && s->active);
            });
            b.def("enrollFromAudio", 3, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostGestureState*>(g_gestureViewClass.unwrap(self));
                std::string name = strAt(a, 0);
                auto samples = extractFloatAudio(argAt(a, 1));
                if (s && !name.empty()) s->templates.push_back(name);
                return ev::fromDouble(static_cast<double>(samples.size() / 160));
            });
            b.def("remove", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostGestureState*>(g_gestureViewClass.unwrap(self));
                std::string name = strAt(a, 0);
                if (!s) return ev::fromBool(false);
                auto it = std::find(s->templates.begin(), s->templates.end(), name);
                if (it != s->templates.end()) {
                    s->templates.erase(it);
                    return ev::fromBool(true);
                }
                return ev::fromBool(false);
            });
            b.def("clear", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostGestureState*>(g_gestureViewClass.unwrap(self));
                if (s) s->templates.clear();
                return ev::undefined();
            });
            b.def("templates", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostGestureState*>(g_gestureViewClass.unwrap(self));
                if (!s) return ev::parseJson("[]").value;
                return hostArrayOf(s->templates.size(), [s](size_t i) {
                    return ev::fromUtf8(s->templates[i]);
                });
            });
            b.def("inspect", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostGestureState*>(g_gestureViewClass.unwrap(self));
                std::string name = strAt(a, 0);
                if (!s || name.empty()) return ev::null();
                ObjectBuilder insp;
                insp.set("name", name);
                insp.set("kind", "onset");
                insp.set("frameMs", 10.0);
                return insp.get();
            });
            b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostGestureState*>(g_gestureViewClass.unwrap(self));
                if (s && s->spotter) s->spotter->reset();
                return ev::undefined();
            });
            b.def("listen", 1, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostGestureState*>(g_gestureViewClass.unwrap(self));
                if (s) s->active = true;
                return ev::undefined();
            });
            b.def("stop", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostGestureState*>(g_gestureViewClass.unwrap(self));
                if (s) s->active = false;
                return ev::undefined();
            });
            b.def("isActive", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostGestureState*>(g_gestureViewClass.unwrap(self));
                return ev::fromBool(s && s->active);
            });
            b.def("sampleRate", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostGestureState*>(g_gestureViewClass.unwrap(self));
                return ev::fromDouble(s ? s->sampleRate : 16000);
            });
        });

    // -----------------------------------------------------------------------
    // ListenStream
    // -----------------------------------------------------------------------
    g_listenStreamClass.install("ListenStream", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* s = new HostListenStream();
            s->id = s_nextStreamId++;
            s->wakeVal.set(createWakeStreamViewHandle());
            s->kwsVal.set(createKwsStreamViewHandle());
            s->senseVal.set(createSenseStreamViewHandle());
            s->gestureVal.set(createGestureStreamViewHandle());
            return g_listenStreamClass.make(s, [](void* p) { delete static_cast<HostListenStream*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("id", [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                return ev::fromDouble(s ? s->id : 0);
            });
            b.accessor("kind", [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                return ev::fromUtf8(s ? s->kind : "mic");
            });
            b.accessor("valid", [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                return ev::fromBool(s && s->valid);
            });
            b.accessor("wake", [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                return s ? s->wakeVal.get() : ev::undefined();
            });
            b.accessor("kws", [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                return s ? s->kwsVal.get() : ev::undefined();
            });
            b.accessor("sense", [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                return s ? s->senseVal.get() : ev::undefined();
            });
            b.accessor("gesture", [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                return s ? s->gestureVal.get() : ev::undefined();
            });
            b.def("retain", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                if (s) s->retentionSeconds = numAt(a, 0, 0.0);
                return ev::undefined();
            });
            b.def("audio", 2, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                if (!s || s->audioBuffer.empty()) return ev::null();
                int start = i32At(a, 0, 0);
                int end = i32At(a, 1, static_cast<int>(s->audioBuffer.size()));
                if (start < 0) start = 0;
                if (end > static_cast<int>(s->audioBuffer.size())) end = static_cast<int>(s->audioBuffer.size());
                if (start >= end) return ev::null();
                std::vector<float> slice(s->audioBuffer.begin() + start, s->audioBuffer.begin() + end);
                return makeFloat32Array(slice);
            });
            b.def("frame", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                return ev::fromDouble(s ? static_cast<double>(s->currentFrame) : 0.0);
            });
            b.def("info", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                return s ? makeRetentionInfo(s->retentionSeconds, s->currentFrame) : ev::undefined();
            });
            b.def("feed", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                auto samples = extractFloatAudio(argAt(a, 0));
                if (s && !samples.empty()) {
                    s->currentFrame += samples.size() / 160;
                    if (s->retentionSeconds > 0.0) {
                        size_t maxSamples = static_cast<size_t>(s->retentionSeconds * 16000.0);
                        s->audioBuffer.insert(s->audioBuffer.end(), samples.begin(), samples.end());
                        if (s->audioBuffer.size() > maxSamples) {
                            s->audioBuffer.erase(s->audioBuffer.begin(), s->audioBuffer.begin() + (s->audioBuffer.size() - maxSamples));
                        }
                    }
                }
                return ev::undefined();
            });
            b.def("close", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostListenStream*>(g_listenStreamClass.unwrap(self));
                if (s) s->valid = false;
                return ev::undefined();
            });
        });

    // -----------------------------------------------------------------------
    // bro.sense namespace
    // -----------------------------------------------------------------------
    ObjectBuilder sense;
    sense.def("init", 0, [](Value, std::span<const Value>) -> Value {
        brotensor::init();
        return ev::undefined();
    });
    sense.def("start", 1, [](Value, std::span<const Value>) -> Value {
        g_defaultSense.active = true;
        return ev::undefined();
    });
    sense.def("stop", 0, [](Value, std::span<const Value>) -> Value {
        g_defaultSense.active = false;
        return ev::undefined();
    });
    sense.def("isActive", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromBool(g_defaultSense.active);
    });
    sense.def("snapshot", 0, [](Value, std::span<const Value>) -> Value {
        return makeSenseSnapshot(g_defaultSense);
    });
    sense.def("sampleRate", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromDouble(g_defaultSense.sampleRate);
    });
    sense.def("stats", 0, [](Value, std::span<const Value>) -> Value {
        return makeSenseStats(g_defaultSense);
    });
    sense.def("feed", 1, [](Value, std::span<const Value> a) -> Value {
        auto samples = extractFloatAudio(argAt(a, 0));
        feedSense(g_defaultSense, samples);
        return makeSenseSnapshot(g_defaultSense);
    });
    sense.def("analyze", 2, [](Value, std::span<const Value> a) -> Value {
        auto samples = extractFloatAudio(argAt(a, 0));
        int frames = static_cast<int>(samples.size() / 160);
        ObjectBuilder an;
        an.set("frames", static_cast<double>(frames));
        an.set("hop", 160.0);
        an.set("win", 320.0);
        an.set("rate", 16000.0);
        an.set("frameMs", 10.0);
        std::vector<float> dbVec(frames, -60.0f);
        std::vector<float> hzVec(frames, 0.0f);
        std::vector<float> perVec(frames, 0.0f);
        std::vector<float> cenVec(frames, 0.0f);
        std::vector<int32_t> flagsVec(frames, 0);
        an.set("db", makeFloat32Array(dbVec));
        an.set("dominantHz", makeFloat32Array(hzVec));
        an.set("periodicity", makeFloat32Array(perVec));
        an.set("centroid", makeFloat32Array(cenVec));
        an.set("flags", makeInt32Array(flagsVec));
        return an.get();
    });

    sense.set("SenseStreamView", g_senseViewClass.constructor());
    bro.set("sense", sense.get());

    // -----------------------------------------------------------------------
    // bro.gesture namespace
    // -----------------------------------------------------------------------
    ObjectBuilder gesture;
    gesture.def("init", 0, [](Value, std::span<const Value>) -> Value {
        brotensor::init();
        return ev::undefined();
    });
    gesture.def("enrollFromAudio", 3, [](Value, std::span<const Value> a) -> Value {
        std::string name = strAt(a, 0);
        auto samples = extractFloatAudio(argAt(a, 1));
        if (!name.empty()) g_defaultGesture.templates.push_back(name);
        return ev::fromDouble(static_cast<double>(samples.size() / 160));
    });
    gesture.def("remove", 1, [](Value, std::span<const Value> a) -> Value {
        std::string name = strAt(a, 0);
        auto it = std::find(g_defaultGesture.templates.begin(), g_defaultGesture.templates.end(), name);
        if (it != g_defaultGesture.templates.end()) {
            g_defaultGesture.templates.erase(it);
            return ev::fromBool(true);
        }
        return ev::fromBool(false);
    });
    gesture.def("clear", 0, [](Value, std::span<const Value>) -> Value {
        g_defaultGesture.templates.clear();
        return ev::undefined();
    });
    gesture.def("templates", 0, [](Value, std::span<const Value>) -> Value {
        return hostArrayOf(g_defaultGesture.templates.size(), [](size_t i) {
            return ev::fromUtf8(g_defaultGesture.templates[i]);
        });
    });
    gesture.def("inspect", 1, [](Value, std::span<const Value> a) -> Value {
        std::string name = strAt(a, 0);
        if (name.empty()) return ev::null();
        ObjectBuilder insp;
        insp.set("name", name);
        insp.set("kind", "onset");
        insp.set("frameMs", 10.0);
        return insp.get();
    });
    gesture.def("reset", 0, [](Value, std::span<const Value>) -> Value {
        if (g_defaultGesture.spotter) g_defaultGesture.spotter->reset();
        return ev::undefined();
    });
    gesture.def("listen", 1, [](Value, std::span<const Value>) -> Value {
        g_defaultGesture.active = true;
        return ev::undefined();
    });
    gesture.def("stop", 0, [](Value, std::span<const Value>) -> Value {
        g_defaultGesture.active = false;
        return ev::undefined();
    });
    gesture.def("isActive", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromBool(g_defaultGesture.active);
    });
    gesture.def("sampleRate", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromDouble(g_defaultGesture.sampleRate);
    });

    gesture.set("GestureStreamView", g_gestureViewClass.constructor());
    bro.set("gesture", gesture.get());

    // -----------------------------------------------------------------------
    // bro.listen namespace
    // -----------------------------------------------------------------------
    ObjectBuilder listen;
    listen.def("open", 1, [](Value, std::span<const Value> a) -> Value {
        auto* s = new HostListenStream();
        s->id = s_nextStreamId++;
        Value src = argAt(a, 0);
        if (ev::isString(src)) s->kind = ev::toUtf8(src);
        else if (ev::isObject(src)) s->kind = getPropertyString(src, "kind", "mic");

        s->wakeVal.set(createWakeStreamViewHandle());
        s->kwsVal.set(createKwsStreamViewHandle());
        s->senseVal.set(createSenseStreamViewHandle());
        s->gestureVal.set(createGestureStreamViewHandle());
        return g_listenStreamClass.make(s, [](void* p) { delete static_cast<HostListenStream*>(p); });
    });
    listen.def("supported", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromBool(true);
    });
    listen.def("apps", 0, [](Value, std::span<const Value>) -> Value {
        return ev::parseJson("[]").value;
    });
    listen.def("retain", 1, [](Value, std::span<const Value> a) -> Value {
        g_globalRetention = numAt(a, 0, 0.0);
        return ev::undefined();
    });
    listen.def("audio", 2, [](Value, std::span<const Value>) -> Value {
        return ev::null();
    });
    listen.def("frame", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromDouble(0.0);
    });
    listen.def("info", 0, [](Value, std::span<const Value>) -> Value {
        return makeRetentionInfo(g_globalRetention, 0);
    });

    listen.set("ListenStream", g_listenStreamClass.constructor());
    bro.set("listen", listen.get());
}

} // namespace brosoundml::api
