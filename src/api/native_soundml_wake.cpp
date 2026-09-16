// bro.wake — streaming wake-word detection over the listen host.
//
// One BC-ResNet checkpoint loads ONCE (bro.wake.load / the first listen()
// with weights) and is shared by every stream's detector: bro.wake.listen
// runs it on the default mic stream, stream.wake.listen on any other
// ListenStream. Fires publish through an atomic the frame pump drains into
// onFire; the detector keeps feeding while suspended (its window must roll)
// but fires are not counted.

#include "soundml_listen_internal.h"

#include <brosoundml/bc_resnet2d.h>
#include <brosoundml/wake.h>
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

// ─── One stream's wake-word tenant ───────────────────────────────────────────
//
// Address-stable (held by unique_ptr in g_wake.tenants), so the inference-thread
// onWake closure can capture a raw WakeTenant* and publish into its atomics.
struct WakeTenant {
    StreamId streamId = kInvalidStream;

    // This stream's detector, built over the shared net. Held for score reads /
    // atomic setters; the inference pump closure holds the strong ref that runs
    // feed(). Dropped on stop() so the model is destroyed off the main thread.
    std::shared_ptr<brosoundml::WakeWord> wake;
    ev::Persistent                        onFire;

    // Published by the inference thread, drained by tickWake (main thread).
    // scoreMaxX10000: max of wake->last_score()*10000 across every feed since
    // listen(). firePending: detected-and-not-suspended fires awaiting delivery.
    std::atomic<int> scoreMaxX10000{0};
    std::atomic<int> firePending{0};

    // Action gate. Written by suspend()/resume() (main), read by the inference
    // thread inside the onWake hook.
    std::atomic<bool> suspended{false};

    bool active = false;
};

struct WakeNamespace {
    // The weights, loaded once and shared by every stream's detector.
    std::shared_ptr<const brosoundml::BcResnet2d> net;
    brotensor::Device                             device = brotensor::Device::CPU;
    std::unordered_map<StreamId, std::unique_ptr<WakeTenant>> tenants;
};

WakeNamespace g_wake;

// A per-stream view object: `stream.wake`. Carries only the stream id.
struct WakeView {
    StreamId streamId = kInvalidStream;
};

HostClass g_wakeViewClass;

// ─── Tenant registry ─────────────────────────────────────────────────────────

WakeTenant* findTenant(StreamId id) {
    auto it = g_wake.tenants.find(id);
    return it == g_wake.tenants.end() ? nullptr : it->second.get();
}

WakeTenant* ensureTenant(StreamId id) {
    if (id == kInvalidStream) return nullptr;
    if (WakeTenant* t = findTenant(id)) return t;
    auto t = std::make_unique<WakeTenant>();
    t->streamId = id;
    WakeTenant* p = t.get();
    g_wake.tenants[id] = std::move(t);
    return p;
}

// Stop a tenant's detection: detach its WakeWord from its stream (a host
// barrier), drop the score ref, release its onFire, clear its atomics.
void stopTenant(WakeTenant* t) {
    if (!t->active) return;
    // Detach from the listen host. The host replaces (or tears down) the
    // stream's pump; any other member (bro.kws / bro.sense) keeps rolling.
    listenStreamSetWake(t->streamId, nullptr, brotensor::Device::CPU, nullptr);
    t->wake.reset();
    t->onFire = ev::Persistent();
    t->firePending.store(0, std::memory_order_relaxed);
    t->scoreMaxX10000.store(0, std::memory_order_relaxed);
    t->suspended.store(false, std::memory_order_relaxed);
    t->active = false;
}

void unloadAll() {
    for (auto& kv : g_wake.tenants) stopTenant(kv.second.get());
    g_wake.tenants.clear();
    g_wake.net.reset();
}

// The onWake hook for one tenant. Runs on the inference thread after every bus
// feed (not only on fires). Captures the tenant pointer (address-stable) and
// the detector's score ref; writes only the tenant's atomics (which live until
// the tenant is dropped), so a stale hook that runs once more before a
// membership swap is harmless.
ListenWakeFn makeOnWake(WakeTenant* t, std::shared_ptr<brosoundml::WakeWord> wake) {
    return [t, wake](bool fired) {
        const int sx = static_cast<int>(wake->last_score() * 10000.0f);
        int prev = t->scoreMaxX10000.load(std::memory_order_relaxed);
        while (sx > prev &&
               !t->scoreMaxX10000.compare_exchange_weak(
                   prev, sx, std::memory_order_relaxed)) {
            // prev reloaded by compare_exchange_weak on failure
        }
        if (fired && !t->suspended.load(std::memory_order_relaxed)) {
            t->firePending.fetch_add(1, std::memory_order_release);
        }
    };
}

// Get-or-load the shared net. If already loaded, returns it (weights ignored —
// unload() first to swap). Otherwise loads from `weights` (required) on `dev`.
bool ensureNet(const std::string& weights, brotensor::Device dev, std::string& err) {
    if (g_wake.net) return true;
    if (weights.empty()) {
        err = "no model loaded — pass opts.weights (or call bro.wake.load first)";
        return false;
    }
    try {
        std::shared_ptr<const brosoundml::BcResnet2d> net;
        {
            brotensor::DeviceScope scope(dev);
            net = std::make_shared<const brosoundml::BcResnet2d>(
                brosoundml::BcResnet2d::load(resolvePath(weights), dev));
        }
        g_wake.net    = std::move(net);
        g_wake.device = dev;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

// The stream `this` addresses: a WakeView → its stream; bro.wake → default mic.
StreamId streamOf(Value self) {
    if (g_wakeViewClass.isInstance(self)) {
        auto* v = static_cast<WakeView*>(g_wakeViewClass.unwrap(self));
        if (v) return v->streamId;
    }
    return listenHostDefaultMicId();
}

// ─── JS-callable functions ───────────────────────────────────────────────────

// bro.wake.load({ weights, device? }) — load the BC-ResNet checkpoint ONCE into
// the shared net (optional; listen() lazy-loads from its own weights too).
// Drops any existing tenants (their detectors referenced the old net).
Value jsLoad(Value, std::span<const Value> a) {
    if (!isObjectArg(a, 0))
        return ev::throwTypeError("bro.wake.load(opts): opts object required (weights, ...)");
    ev::Persistent opts(a[0]);
    std::string weights = getPropertyString(opts.get(), "weights");
    if (weights.empty())
        return ev::throwTypeError("bro.wake.load: opts.weights (model path) required");
    // init() BEFORE the device probe: GPU backends register on the driver probe.
    brotensor::Device dev = autoDevice();
    {
        std::string err;
        if (!parseDeviceOpt(opts.get(), dev, err))
            return ev::throwTypeError("bro.wake.load: " + err);
    }
    unloadAll();
    std::string err;
    if (!ensureNet(weights, dev, err))
        return ev::throwError("bro.wake.load: " + err);
    logInfo(std::string("[wake] BcResnet2d loaded on ") + deviceName(dev) +
            ", shared across streams");
    return ev::undefined();
}

Value jsUnload(Value, std::span<const Value>) {
    unloadAll();
    return ev::undefined();
}

// bro.wake.listen({ weights?, onFire, threshold?, refractoryMs?, smoothing? })
// Start wake detection on THIS stream. Loads the shared net from opts.weights if
// none is loaded yet. A second listen() on the same stream implicitly stops the
// previous detector first. The detector-policy fields apply to THIS stream's
// detector only.
Value jsListen(Value self, std::span<const Value> a) {
    const StreamId sid = streamOf(self);
    if (!isObjectArg(a, 0))
        return ev::throwTypeError("bro.wake.listen(opts): opts object required (weights, onFire, ...)");
    ev::Persistent opts(a[0]);

    ev::Persistent onFire = getFunctionOpt(opts.get(), "onFire");
    if (!ev::isFunction(onFire.get()))
        return ev::throwTypeError("bro.wake.listen: opts.onFire (function) required");
    if (!listenHostAudioAvailable())
        return ev::throwError("bro.wake.listen: audio engine not available");

    std::string weights = getPropertyString(opts.get(), "weights");   // optional once loaded

    // init() BEFORE the device probe.
    brotensor::Device dev = g_wake.net ? g_wake.device : autoDevice();
    if (!g_wake.net) {
        std::string err;
        if (!parseDeviceOpt(opts.get(), dev, err))
            return ev::throwTypeError("bro.wake.listen: " + err);
    }
    {
        std::string err;
        if (!ensureNet(weights, dev, err))
            return ev::throwError("bro.wake.listen: " + err);
    }

    const double threshold = getPropertyDouble(opts.get(), "threshold", 0.85);
    const int refractoryMs = getPropertyInt(opts.get(), "refractoryMs", 500);
    int smoothingHits = -1, smoothingWindow = -1;
    {
        Value sm = ev::getProperty(opts.get(), "smoothing");
        if (ev::isObject(sm)) {
            ev::Persistent smRoot(sm);
            getIntOpt(smRoot.get(), "hits", smoothingHits);
            getIntOpt(smRoot.get(), "window", smoothingWindow);
        }
    }

    WakeTenant* t = ensureTenant(sid);
    if (!t) return ev::throwError("bro.wake.listen: no stream");
    stopTenant(t);   // implicit-stop on re-listen

    try {
        auto wake = std::make_shared<brosoundml::WakeWord>(g_wake.net);
        wake->set_threshold(static_cast<float>(threshold));
        if (smoothingHits > 0 && smoothingWindow > 0)
            wake->set_smoothing(smoothingHits, smoothingWindow);
        if (refractoryMs >= 0) wake->set_refractory_ms(refractoryMs);

        t->firePending.store(0, std::memory_order_relaxed);
        t->suspended.store(false, std::memory_order_relaxed);
        t->scoreMaxX10000.store(0, std::memory_order_relaxed);

        // Join the listen host on this stream. The host runs ONE PCEN mel pass
        // and hands the new-frame block to WakeWord::feed_mel alongside any
        // other attached tenant. Throws on a front-end mismatch or source
        // failure (the catch detaches, tearing down member-less infra).
        listenStreamSetWake(sid, wake, dev, makeOnWake(t, wake));

        t->wake   = std::move(wake);
        t->onFire = onFire;
        t->active = true;

        logInfo("[wake] listening on stream " + std::to_string(sid) +
                " (device=" + deviceName(dev) + ", threshold=" + std::to_string(threshold) +
                ", model=" + std::to_string(t->wake->config().sample_rate) + " Hz)");
        return ev::undefined();
    } catch (const std::exception& e) {
        listenStreamSetWake(sid, nullptr, brotensor::Device::CPU, nullptr);
        return ev::throwError(std::string("bro.wake.listen: ") + e.what());
    }
}

Value jsStop(Value self, std::span<const Value>) {
    if (WakeTenant* t = findTenant(streamOf(self))) stopTenant(t);
    return ev::undefined();
}

Value jsSuspend(Value self, std::span<const Value>) {
    if (WakeTenant* t = findTenant(streamOf(self)))
        t->suspended.store(true, std::memory_order_relaxed);
    return ev::undefined();
}

Value jsResume(Value self, std::span<const Value>) {
    if (WakeTenant* t = findTenant(streamOf(self)))
        t->suspended.store(false, std::memory_order_relaxed);
    return ev::undefined();
}

Value jsLastScore(Value self, std::span<const Value>) {
    WakeTenant* t = findTenant(streamOf(self));
    return ev::fromDouble((t && t->wake) ? t->wake->last_score() : 0.0);
}

Value jsIsActive(Value self, std::span<const Value>) {
    WakeTenant* t = findTenant(streamOf(self));
    return ev::fromBool(t && t->active);
}

Value jsIsSuspended(Value self, std::span<const Value>) {
    WakeTenant* t = findTenant(streamOf(self));
    return ev::fromBool(t && t->suspended.load(std::memory_order_relaxed));
}

Value jsIsLoaded(Value, std::span<const Value>) {
    return ev::fromBool(static_cast<bool>(g_wake.net));
}

Value jsSetThreshold(Value self, std::span<const Value> a) {
    if (a.empty() || !ev::isNumber(a[0]))
        return ev::throwTypeError("bro.wake.setThreshold(value): number required");
    const double th = ev::toDouble(a[0]);
    // WakeWord setters are atomic; safe to call while the inference thread feeds.
    WakeTenant* t = findTenant(streamOf(self));
    if (t && t->wake) t->wake->set_threshold(static_cast<float>(th));
    return ev::undefined();
}

// Diagnostic surface over THIS stream's mic tap (cf. bro.kws.stats). Returns
// { framesDelivered, samplesDelivered, rollingPeak, scoreMax } or null when no
// tap is installed (e.g. a non-mic loopback stream) or nothing is listening.
Value jsStats(Value self, std::span<const Value>) {
    WakeTenant* t = findTenant(streamOf(self));
    if (!t || !t->active) return ev::null();
    const double scoreMax = t->scoreMaxX10000.load(std::memory_order_relaxed) / 10000.0;
    return makeTapStats(t->streamId, [scoreMax](ObjectBuilder& o) {
        o.set("scoreMax", scoreMax);
    });
}

// Manual feed for tests / scripted scenarios on THIS stream. Samples must
// already be at the wake model's native rate (pass it as the optional second
// arg to assert). Mode split:
//   - Headless (no inference worker): the stream's bus runs synchronously on
//     this thread and the call returns whether the detector fired (onFire also
//     fires on the next tick unless suspended).
//   - Threaded: samples go into the stream's ring; the fire surfaces via onFire
//     on the next tickWake. Returns undefined.
// Refuses to run while live MIC capture is active (two-producer race).
Value jsFeed(Value self, std::span<const Value> a) {
    WakeTenant* t = findTenant(streamOf(self));
    if (!t || !t->active || !t->wake)
        return ev::throwError("bro.wake.feed: no active detector on this stream");
    if (listenHostMicCapturing())
        return ev::throwError(
            "bro.wake.feed: cannot feed while live mic capture is active "
            "(feed is for headless/offline use; the live tap already writes the ring)");
    bool ok = false;
    std::vector<float> samples = readPcmArg(a, 0, ok);
    if (!ok) return ev::throwTypeError("bro.wake.feed(Float32Array, sampleRate?)");
    const int wakeRate = t->wake->config().sample_rate;
    if (a.size() >= 2 && ev::isNumber(a[1])) {
        const int r = static_cast<int>(ev::toDouble(a[1]));
        if (r > 0 && r != wakeRate) {
            return ev::throwTypeError(
                "bro.wake.feed: sampleRate=" + std::to_string(r) +
                " must equal the wake rate=" + std::to_string(wakeRate) +
                " (use bro.mic to feed mic-rate audio through the resampler)");
        }
    }
    const int n = static_cast<int>(samples.size());
    if (listenHostThreaded()) {
        listenStreamWriteRing(t->streamId, samples.data(), n);
        return ev::undefined();
    }
    try {
        const brosoundml::ListenFeedResult r =
            listenStreamFeedInline(t->streamId, samples.data(), n);
        return ev::fromBool(r.wake_fired);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.wake.feed: ") + e.what());
    }
}

// Deliver one tenant's pending fires (main thread). Re-resolves the tenant
// after every call: an onFire may stop() or unload() it.
void drainTenant(StreamId id) {
    WakeTenant* t = findTenant(id);
    if (!t || !t->active) return;
    const int fires = t->firePending.exchange(0, std::memory_order_acquire);
    if (fires <= 0 || !ev::isFunction(t->onFire.get())) return;
    for (int i = 0; i < fires; ++i) {
        ev::Persistent cb(t->onFire);
        callTenantCallback(cb, std::span<const Value>(), "wake");
        t = findTenant(id);
        if (!t || !t->active) break;
    }
}

void defineWakeOps(ObjectBuilder& b) {
    b.def("listen",       1, jsListen);
    b.def("stop",         0, jsStop);
    b.def("suspend",      0, jsSuspend);
    b.def("resume",       0, jsResume);
    b.def("lastScore",    0, jsLastScore);
    b.def("isActive",     0, jsIsActive);
    b.def("isSuspended",  0, jsIsSuspended);
    b.def("isLoaded",     0, jsIsLoaded);
    b.def("setThreshold", 1, jsSetThreshold);
    b.def("stats",        0, jsStats);
    b.def("feed",         2, jsFeed);
}

}  // namespace

Value makeWakeView(StreamId id) {
    auto* v = new WakeView{id};
    return g_wakeViewClass.make(v, [](void* p) { delete static_cast<WakeView*>(p); });
}

void tickWake() {
    // Walk a snapshot of ids: an onFire may stop() / unload() and reshape the
    // registry under a live iterator.
    std::vector<StreamId> ids;
    ids.reserve(g_wake.tenants.size());
    for (const auto& kv : g_wake.tenants) ids.push_back(kv.first);
    for (StreamId id : ids) {
        auto it = g_wake.tenants.find(id);
        if (it == g_wake.tenants.end()) continue;
        WakeTenant* t = it->second.get();
        // Prune a tenant whose stream has closed. The stream's teardown removed
        // its pump (a barrier), so the onWake closure can no longer run — safe
        // to drop. Default mic is never invalid.
        if (!listenHostValid(id)) {
            t->active = false;   // detach is a no-op — stream gone
            t->onFire = ev::Persistent();
            g_wake.tenants.erase(it);
            continue;
        }
        drainTenant(id);
    }
}

void cleanupWake() {
    unloadAll();
}

void installWake(ObjectBuilder& bro) {
    // The per-stream ops shared by bro.wake and stream.wake: the SAME bodies
    // sit on the WakeStreamView prototype and on the namespace. Instances
    // come from bro.listen.open(); the class is exposed for instanceof only.
    g_wakeViewClass.install("WakeStreamView", 0, nullptr,
        [](ObjectBuilder& b) {
            b.accessor("active", [](Value self, std::span<const Value>) -> Value {
                WakeTenant* t = findTenant(streamOf(self));
                return ev::fromBool(t && t->active);
            });
            defineWakeOps(b);
        },
        /*global=*/false);

    ObjectBuilder wake;
    wake.def("init", 0, [](Value, std::span<const Value>) -> Value {
        try {
            brotensor::init();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("bro.wake.init: ") + e.what());
        }
        return ev::undefined();
    });
    // Namespace ops (shared net — not stream-scoped).
    wake.def("load",   1, jsLoad);
    wake.def("unload", 0, jsUnload);
    // Per-stream ops — on bro.wake they target the shared default-mic stream.
    defineWakeOps(wake);
    wake.set("WakeStreamView", g_wakeViewClass.constructor());
    bro.set("wake", wake.get());
}

}  // namespace brosoundml::api
