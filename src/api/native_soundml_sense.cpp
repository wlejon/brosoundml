// bro.sense — the model-free acoustic sensor hub (VAD / onset / tonality)
// over the listen host. bro.sense targets the default mic stream,
// stream.sense any other ListenStream; each stream gets its own SensorHub.
// The hub's snapshot IS the delivery — poll-only, no per-frame callback.

#include "soundml_listen_internal.h"

#include <brosoundml/sensor_hub.h>
#include <brotensor/runtime.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brosoundml::api {

namespace {

// ─── One stream's sensor tenant ──────────────────────────────────────────────
struct SenseTenant {
    StreamId                               streamId = kInvalidStream;
    // The hub. Owned by the tenant while active; the stream's pump closure
    // holds a second strong ref so a late pump after stop() is harmless.
    std::shared_ptr<brosoundml::SensorHub> hub;
    bool                                   active = false;
};

struct SenseNamespace {
    std::unordered_map<StreamId, std::unique_ptr<SenseTenant>> tenants;
};

SenseNamespace g_sense;

// A per-stream view object: `stream.sense`. Carries only the stream id.
struct SenseView {
    StreamId streamId = kInvalidStream;
};

HostClass g_senseViewClass;

// Overlay sensor-policy keys present on `obj` onto `cfg` (flat keys — the
// config is small enough that nesting would just be ceremony).
void readConfig(Value obj, brosoundml::SensorHubConfig& cfg) {
    if (!ev::isObject(obj)) return;
    ev::Persistent root(obj);
    getFloatOpt(root.get(), "vadFloorDb",  cfg.vad_abs_floor_db);
    getFloatOpt(root.get(), "vadSnrDb",    cfg.vad_snr_db);
    getFloatOpt(root.get(), "vadRiseDbps", cfg.vad_floor_rise_dbps);
    getIntOpt(root.get(),   "vadHangFrames", cfg.vad_hang_frames);
    getFloatOpt(root.get(), "onsetRatio",  cfg.onset_ratio);
    getFloatOpt(root.get(), "onsetAbs",    cfg.onset_abs);
    getFloatOpt(root.get(), "onsetEma",    cfg.onset_ema);
    getIntOpt(root.get(),   "onsetRefractoryFrames", cfg.onset_refractory_frames);
    getFloatOpt(root.get(), "tonalMinPeriodicity", cfg.tonal_min_periodicity);
    getFloatOpt(root.get(), "tonalFminHz", cfg.tonal_fmin_hz);
    getFloatOpt(root.get(), "tonalFmaxHz", cfg.tonal_fmax_hz);
}

Value makeSnapshot(const brosoundml::SensorSnapshot& s) {
    ObjectBuilder o;
    o.set("frames", static_cast<double>(s.frames));
    o.set("t",      static_cast<double>(s.t));
    o.set("rms",  static_cast<double>(s.rms));
    o.set("peak", static_cast<double>(s.peak));
    o.set("db",   static_cast<double>(s.db));
    o.set("voice",        s.voice);
    o.set("noiseFloorDb", static_cast<double>(s.noise_floor_db));
    o.set("snrDb",        static_cast<double>(s.snr_db));
    o.set("voiceFrames",  static_cast<double>(s.voice_frames));
    o.set("voiceEvents",  static_cast<double>(s.voice_events));
    o.set("lastVoiceFrame", static_cast<double>(s.last_voice_frame));
    o.set("flux",   static_cast<double>(s.flux));
    o.set("onset",  s.onset);
    o.set("onsets", static_cast<double>(s.onsets));
    o.set("lastOnsetFrame", static_cast<double>(s.last_onset_frame));
    o.set("periodicity", static_cast<double>(s.periodicity));
    o.set("dominantHz",  static_cast<double>(s.dominant_hz));
    o.set("tonal",       s.tonal);
    o.set("tonalFrames", static_cast<double>(s.tonal_frames));
    o.set("tonalEvents", static_cast<double>(s.tonal_events));
    o.set("lastTonalFrame", static_cast<double>(s.last_tonal_frame));
    o.set("centroid",    static_cast<double>(s.centroid));
    return o.get();
}

// ─── Tenant registry ─────────────────────────────────────────────────────────

SenseTenant* findTenant(StreamId id) {
    auto it = g_sense.tenants.find(id);
    return it == g_sense.tenants.end() ? nullptr : it->second.get();
}

void dropTenant(StreamId id) {
    auto it = g_sense.tenants.find(id);
    if (it == g_sense.tenants.end()) return;
    if (it->second->active)
        listenStreamSetHub(id, nullptr);   // detach (no-op if the stream is gone)
    g_sense.tenants.erase(it);
}

// The stream `this` addresses: a SenseView → its stream; bro.sense → default
// mic. Returns kInvalidStream (and prunes the tenant) if a view's stream has
// closed.
StreamId streamOf(Value self) {
    if (g_senseViewClass.isInstance(self)) {
        auto* v = static_cast<SenseView*>(g_senseViewClass.unwrap(self));
        if (v) {
            if (!listenHostValid(v->streamId)) { dropTenant(v->streamId); return kInvalidStream; }
            return v->streamId;
        }
    }
    return listenHostDefaultMicId();
}

SenseTenant* ensureTenant(StreamId id) {
    if (id == kInvalidStream) return nullptr;
    if (SenseTenant* t = findTenant(id)) return t;
    auto t = std::make_unique<SenseTenant>();
    t->streamId = id;
    SenseTenant* p = t.get();
    g_sense.tenants[id] = std::move(t);
    return p;
}

void stopSensing(SenseTenant* t) {
    if (!t->active) return;
    // Detach from the listen host. The host replaces (or tears down) the
    // stream's pump; any other member (bro.kws) keeps rolling.
    listenStreamSetHub(t->streamId, nullptr);
    t->hub.reset();
    t->active = false;
}

// ─── JS-callable functions ───────────────────────────────────────────────────

// start(opts?) — build the hub and go live on THIS stream.
//   opts (all optional): vadFloorDb, vadSnrDb, vadRiseDbps, vadHangFrames,
//   onsetRatio, onsetAbs, onsetEma, onsetRefractoryFrames,
//   tonalMinPeriodicity, tonalFminHz, tonalFmaxHz.
Value jsStart(Value self, std::span<const Value> a) {
    const StreamId sid = streamOf(self);
    if (sid == kInvalidStream) return ev::throwError("bro.sense.start: stream is closed");
    if (!listenHostAudioAvailable())
        return ev::throwError("bro.sense.start: audio engine not available");
    SenseTenant* t = ensureTenant(sid);
    if (t->active)
        return ev::throwError("bro.sense.start: this stream is already sensing (stop() first)");

    // The hub is pure CPU DSP, but its mel front-end runs on brotensor ops —
    // init() is idempotent and cheap.
    try {
        brotensor::init();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.sense.start: ") + e.what());
    }

    try {
        brosoundml::SensorHubConfig cfg;
        if (!a.empty()) readConfig(a[0], cfg);
        auto hub = std::make_shared<brosoundml::SensorHub>(cfg);

        // Join the listen host on this stream: one source + ring + pump drive
        // the hub (alongside bro.kws's spotter, if live) off ONE mel pass. The
        // hub's snapshot IS the delivery — no per-frame callback.
        listenStreamSetHub(sid, hub);
        t->hub    = hub;
        t->active = true;

        logInfo("[sense] sensor bus active on stream " + std::to_string(sid) +
                " (hub=" + std::to_string(hub->sample_rate()) +
                " Hz, hop=" + std::to_string(cfg.mel.hop_length) + ")");
        return ev::undefined();
    } catch (const std::exception& e) {
        t->active = true;   // let stopSensing clear the partial state
        stopSensing(t);
        return ev::throwError(std::string("bro.sense.start: ") + e.what());
    }
}

Value jsStop(Value self, std::span<const Value>) {
    if (SenseTenant* t = findTenant(streamOf(self))) stopSensing(t);
    return ev::undefined();
}

Value jsIsActive(Value self, std::span<const Value>) {
    SenseTenant* t = findTenant(streamOf(self));
    return ev::fromBool(t && t->active);
}

// snapshot() -> {frames, t, rms, peak, db, voice, ..., tonal, ...}
// Lock-free seqlock read of THIS stream's latest coherent sensor frame; null
// when inactive. Counters (onsets, voiceEvents, tonalEvents) are monotonic, so a
// poller detects events as deltas even when the boolean has already cleared.
Value jsSnapshot(Value self, std::span<const Value>) {
    SenseTenant* t = findTenant(streamOf(self));
    if (!t || !t->hub) return ev::null();
    return makeSnapshot(t->hub->snapshot());
}

Value jsSampleRate(Value self, std::span<const Value>) {
    SenseTenant* t = findTenant(streamOf(self));
    return ev::fromDouble((t && t->hub) ? t->hub->sample_rate() : 0);
}

// Diagnostic surface over THIS stream's mic tap (cf. bro.kws.stats). Null for a
// non-mic loopback stream or when not sensing.
Value jsStats(Value self, std::span<const Value>) {
    SenseTenant* t = findTenant(streamOf(self));
    if (!t || !t->active) return ev::null();
    return makeTapStats(t->streamId);
}

// Manual feed for tests / scripted scenarios on THIS stream. Samples must
// already be at the hub's rate. Mode split mirrors bro.kws.feed:
//   - Headless (no inference worker): the stream's bus runs synchronously and
//     the post-feed snapshot comes back.
//   - Threaded: samples go into the stream's ring; poll snapshot() as usual.
// Refuses to run while live MIC capture is active (two-producer race).
Value jsFeed(Value self, std::span<const Value> a) {
    SenseTenant* t = findTenant(streamOf(self));
    if (!t || !t->active || !t->hub)
        return ev::throwError("bro.sense.feed: this stream is not sensing");
    if (listenHostMicCapturing())
        return ev::throwError(
            "bro.sense.feed: cannot feed while live mic capture is active "
            "(feed is for headless/offline use; the live tap already writes the ring)");
    bool ok = false;
    std::vector<float> samples = readPcmArg(a, 0, ok);
    if (!ok) return ev::throwTypeError("bro.sense.feed(Float32Array)");
    const int n = static_cast<int>(samples.size());
    if (listenHostThreaded()) {
        listenStreamWriteRing(t->streamId, samples.data(), n);
        return ev::undefined();
    }
    try {
        listenStreamFeedInline(t->streamId, samples.data(), n);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.sense.feed: ") + e.what());
    }
    return makeSnapshot(t->hub->snapshot());
}

// bro.sense.analyze(samples, opts?) -> per-frame sensor timeline for a clip.
// Runs a PRIVATE SensorHub over the clip offline — no live tap, no stream, no
// effect on any bus, callable any time — and returns columnar arrays. Namespace
// op (not stream-scoped). flags packs bit0=voice, bit1=tonal, bit2=onset.
Value jsAnalyze(Value, std::span<const Value> a) {
    bool ok = false;
    std::vector<float> pcm = readPcmArg(a, 0, ok);
    if (!ok || pcm.empty())
        return ev::throwTypeError("bro.sense.analyze(Float32Array, opts?)");
    try {
        brotensor::init();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.sense.analyze: ") + e.what());
    }

    brosoundml::SensorHubConfig cfg;
    if (a.size() >= 2) readConfig(a[1], cfg);

    std::vector<float>   db, hz, per, cen;
    std::vector<int32_t> flags;
    const int win = cfg.mel.win_length, hop = cfg.mel.hop_length;
    try {
        brosoundml::SensorHub hub(cfg);
        const float* p = pcm.data();
        const int n = static_cast<int>(pcm.size());
        auto take = [&](const brosoundml::SensorSnapshot& s) {
            db.push_back(s.db);
            hz.push_back(s.dominant_hz);
            per.push_back(s.periodicity);
            cen.push_back(s.centroid);
            int32_t f = 0;
            if (s.voice) f |= 1;
            if (s.tonal) f |= 2;
            if (s.onset) f |= 4;
            flags.push_back(f);
        };
        // Prime one window, then advance one hop per frame — identical framing to
        // the enroll path, so frame indices line up with what a gesture captures.
        if (n >= win) {
            hub.feed(p, win);
            take(hub.snapshot());
            int pos = win;
            while (pos + hop <= n) {
                hub.feed(p + pos, hop);
                take(hub.snapshot());
                pos += hop;
            }
        }
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.sense.analyze: ") + e.what());
    }

    ObjectBuilder o;
    o.set("frames", static_cast<double>(db.size()));
    o.set("hop",  static_cast<double>(hop));
    o.set("win",  static_cast<double>(win));
    o.set("rate", static_cast<double>(cfg.mel.sample_rate));
    o.set("frameMs", 1000.0 * static_cast<double>(hop) / static_cast<double>(cfg.mel.sample_rate));
    { ev::Persistent v(makeFloat32Array(db));  o.set("db", v.get()); }
    { ev::Persistent v(makeFloat32Array(hz));  o.set("dominantHz", v.get()); }
    { ev::Persistent v(makeFloat32Array(per)); o.set("periodicity", v.get()); }
    { ev::Persistent v(makeFloat32Array(cen)); o.set("centroid", v.get()); }
    { ev::Persistent v(makeInt32Array(flags)); o.set("flags", v.get()); }
    return o.get();
}

void defineSenseOps(ObjectBuilder& b) {
    b.def("start",      1, jsStart);
    b.def("stop",       0, jsStop);
    b.def("isActive",   0, jsIsActive);
    b.def("snapshot",   0, jsSnapshot);
    b.def("sampleRate", 0, jsSampleRate);
    b.def("stats",      0, jsStats);
    b.def("feed",       1, jsFeed);
    b.def("analyze",    2, jsAnalyze);   // offline; ignores the stream
}

}  // namespace

Value makeSenseView(StreamId id) {
    auto* v = new SenseView{id};
    return g_senseViewClass.make(v, [](void* p) { delete static_cast<SenseView*>(p); });
}

void cleanupSense() {
    for (auto& kv : g_sense.tenants) stopSensing(kv.second.get());
    g_sense.tenants.clear();
}

void installSense(ObjectBuilder& bro) {
    g_senseViewClass.install("SenseStreamView", 0, nullptr,
        [](ObjectBuilder& b) {
            b.accessor("active", [](Value self, std::span<const Value>) -> Value {
                SenseTenant* t = findTenant(streamOf(self));
                return ev::fromBool(t && t->active);
            });
            defineSenseOps(b);
        },
        /*global=*/false);

    ObjectBuilder sense;
    sense.def("init", 0, [](Value, std::span<const Value>) -> Value {
        try {
            brotensor::init();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("bro.sense.init: ") + e.what());
        }
        return ev::undefined();
    });
    defineSenseOps(sense);
    sense.set("SenseStreamView", g_senseViewClass.constructor());
    bro.set("sense", sense.get());
}

}  // namespace brosoundml::api
