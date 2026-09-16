// bro.kws — open-vocabulary keyword spotting over the listen host.
//
// One PhonemeNet checkpoint (+ its embedded class map) loads ONCE via
// bro.kws.load and is shared by every stream's PhonemeSpotter: bro.kws
// targets the default mic stream, stream.kws any other ListenStream. Each
// stream enrolls its own templates and listens independently; fired spots
// cross inference → main through a per-tenant SPSC ring the frame pump
// drains into onSpot(name, confidence, span).

#include "soundml_listen_internal.h"

#include <brosoundml/phoneme_model.h>
#include <brosoundml/phoneme_spotter.h>
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

// Fired events cross inference -> main as (template index, confidence) pairs
// in a fixed SPSC slot ring. 64 slots is generous — spot events arrive at
// human speech cadence; on overflow the newest event is dropped.
constexpr std::uint64_t kEventSlots = 64;

// ─── One stream's keyword-spotting tenant ────────────────────────────────────
//
// Address-stable (held by unique_ptr in g_kws.tenants), so the inference-thread
// onSpots closure can capture a raw KwsTenant* and publish into its ring.
struct KwsTenant {
    StreamId streamId = kInvalidStream;

    // This stream's spotter, built over the shared net. Owned by the tenant
    // across listen()/stop(); the inference pump closure holds a second strong
    // ref while listening.
    std::shared_ptr<brosoundml::PhonemeSpotter> spotter;

    // Template-name snapshot taken at listen() — index i names eventIdx[i].
    // The enrolled set cannot change while listening, so the snapshot is stable
    // for the pump's lifetime. Main thread reads it in tickKws.
    std::vector<std::string> names;

    ev::Persistent onSpot;

    // SPSC event ring: the inference thread writes slot produced%kEventSlots
    // then bumps produced; tickKws (main) drains up to produced and advances
    // drained. The producer drops events when the ring is full.
    int                        eventIdx[kEventSlots]   = {};
    float                      eventConf[kEventSlots]  = {};
    std::int64_t               eventStart[kEventSlots] = {};   // matched-span frames
    std::int64_t               eventEnd[kEventSlots]   = {};   //   (absolute, frames axis)
    std::atomic<std::uint64_t> produced{0};
    std::atomic<std::uint64_t> drained{0};

    // Gates onSpot delivery, never audio processing (the posterior stream and
    // every template's DP state keep rolling so resume never faces a cold
    // matcher). Written by suspend()/resume() (main), read by the inference
    // thread.
    std::atomic<bool> suspended{false};

    bool listening = false;
};

struct KwsNamespace {
    // The weights, loaded once and shared by every stream's spotter.
    std::shared_ptr<const brosoundml::PhonemeNet> net;
    brotensor::Device                             device = brotensor::Device::CPU;
    // Detector-policy defaults set at load(); applied to each new tenant spotter
    // and as the base for per-template enroll overrides.
    brosoundml::SpotterConfig defaultPolicy;

    std::unordered_map<StreamId, std::unique_ptr<KwsTenant>> tenants;
};

KwsNamespace g_kws;

// A per-stream view object: `stream.kws`. Carries only the stream id.
struct KwsView {
    StreamId streamId = kInvalidStream;
};

HostClass g_kwsViewClass;

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Overlay detector-policy keys present on `obj` onto `cfg`:
//   threshold, refractoryMs, minPhonemes, entrySilenceFrames, emissionFloor,
//   minCoverage, scoreNorm, enrollGaps, gapMinFrames, gapTolerance,
//   smoothing: { hits, window }.
// Used for the global defaults (load) and per-template overrides (enroll).
void readPolicy(Value obj, brosoundml::SpotterConfig& cfg) {
    if (!ev::isObject(obj)) return;
    ev::Persistent root(obj);
    getFloatOpt(root.get(), "threshold",     cfg.threshold);
    getFloatOpt(root.get(), "emissionFloor", cfg.emission_floor);
    // Proportional coverage gate: a completion must have at least
    // ceil(minCoverage * L) of the template's phonemes ACTUALLY emitted (not
    // merely riding the emission floor). This is what stops a long phrase from
    // firing on a short suffix. 0 (default) = absolute min_phonemes gate only.
    getFloatOpt(root.get(), "minCoverage",   cfg.min_coverage_frac);
    // Competition-normalization strength [0,1]: puts templates of differing
    // phoneme make-up on one score scale so a single threshold transfers
    // across them. 0 (default) = raw posterior.
    getFloatOpt(root.get(), "scoreNorm",     cfg.score_norm);
    getFloatOpt(root.get(), "gapTolerance",  cfg.gap_tolerance);
    getIntOpt(root.get(), "refractoryMs",       cfg.refractory_ms);
    getIntOpt(root.get(), "minPhonemes",        cfg.min_phonemes);
    getIntOpt(root.get(), "entrySilenceFrames", cfg.entry_silence_frames);
    getIntOpt(root.get(), "gapMinFrames",       cfg.gap_min_frames);
    getBoolOpt(root.get(), "enrollGaps",        cfg.enroll_gaps);
    Value sm = ev::getProperty(root.get(), "smoothing");
    if (ev::isObject(sm)) {
        ev::Persistent smRoot(sm);
        getIntOpt(smRoot.get(), "hits",   cfg.smoothing_hits);
        getIntOpt(smRoot.get(), "window", cfg.smoothing_window);
    }
}

// The per-template override at args[i], based on the spotter's defaults.
bool readPolicyOverride(std::span<const Value> a, size_t i,
                        const brosoundml::PhonemeSpotter& spotter,
                        brosoundml::SpotterConfig& out) {
    if (!isObjectArg(a, i)) return false;
    out = spotter.config();
    readPolicy(a[i], out);
    return true;
}

int nameIndexOf(const std::vector<std::string>& names, const std::string& n) {
    for (std::size_t i = 0; i < names.size(); ++i)
        if (names[i] == n) return static_cast<int>(i);
    return -1;
}

// Publish one fired event into a tenant's SPSC ring (inference thread). Drops
// the newest event on overflow — never corrupts the ring.
void publishEvent(KwsTenant* t, int nameIdx, float confidence,
                  std::int64_t startFrame, std::int64_t endFrame) {
    const std::uint64_t p = t->produced.load(std::memory_order_relaxed);
    if (p - t->drained.load(std::memory_order_acquire) >= kEventSlots) return;
    t->eventIdx[p % kEventSlots]   = nameIdx;
    t->eventConf[p % kEventSlots]  = confidence;
    t->eventStart[p % kEventSlots] = startFrame;
    t->eventEnd[p % kEventSlots]   = endFrame;
    t->produced.store(p + 1, std::memory_order_release);
}

// ─── Tenant registry ─────────────────────────────────────────────────────────

KwsTenant* findTenant(StreamId id) {
    auto it = g_kws.tenants.find(id);
    return it == g_kws.tenants.end() ? nullptr : it->second.get();
}

// Get-or-create the tenant for `id`, building its spotter over the shared net.
// Returns nullptr if no net is loaded.
KwsTenant* ensureTenant(StreamId id) {
    if (id == kInvalidStream || !g_kws.net) return nullptr;
    if (KwsTenant* t = findTenant(id)) return t;
    auto t = std::make_unique<KwsTenant>();
    t->streamId = id;
    t->spotter  = std::make_shared<brosoundml::PhonemeSpotter>(g_kws.net);
    t->spotter->set_config(g_kws.defaultPolicy);
    KwsTenant* p = t.get();
    g_kws.tenants[id] = std::move(t);
    return p;
}

// Stop a tenant's spotting: detach its spotter from its stream (a host barrier),
// release its onSpot, and clear the live snapshot. Keeps the spotter + templates
// so listening can resume without re-enrolling.
void stopListening(KwsTenant* t) {
    if (!t->listening) return;
    // Detach from the listen host. The host replaces (or tears down) the
    // stream's pump; any other member (bro.sense) keeps rolling.
    listenStreamSetSpotter(t->streamId, nullptr, brotensor::Device::CPU, nullptr);
    t->onSpot = ev::Persistent();
    t->produced.store(0, std::memory_order_relaxed);
    t->drained.store(0, std::memory_order_relaxed);
    t->suspended.store(false, std::memory_order_relaxed);
    t->names.clear();
    t->listening = false;
}

// Drop every tenant (stop + free) and the shared net.
void unloadAll() {
    for (auto& kv : g_kws.tenants) stopListening(kv.second.get());
    g_kws.tenants.clear();
    g_kws.net.reset();
}

// Reject single-producer mutators while the inference thread owns this stream's
// feed(). Returns a thrown value, or undefined when allowed.
bool refuseWhileListening(const KwsTenant* t, const char* what, Value& thrown) {
    if (!t || !t->listening) return false;
    thrown = ev::throwError(std::string("bro.kws.") + what +
        ": not allowed while this stream is listening (enroll/remove/clear/reset "
        "share the spotter's feed thread — stop() first)");
    return true;
}

// The stream `this` addresses: a KwsView → its stream; bro.kws → default mic.
StreamId streamOf(Value self) {
    if (g_kwsViewClass.isInstance(self)) {
        auto* v = static_cast<KwsView*>(g_kwsViewClass.unwrap(self));
        if (v) return v->streamId;
    }
    return listenHostDefaultMicId();
}

// ─── JS-callable functions ───────────────────────────────────────────────────

// bro.kws.load({ weights, device?, threshold?, refractoryMs?, smoothing?,
//                minPhonemes?, entrySilenceFrames?, emissionFloor?,
//                minCoverage?, scoreNorm?, enrollGaps?, gapMinFrames?,
//                gapTolerance? })
// Load the PhonemeNet checkpoint (+ its embedded class map) ONCE into the shared
// net and set the global detector-policy defaults. Drops any existing tenants
// (their spotters referenced the old net). Enroll templates next, then listen().
Value jsLoad(Value, std::span<const Value> a) {
    if (!isObjectArg(a, 0))
        return ev::throwTypeError("bro.kws.load(opts): opts object required (weights, ...)");
    ev::Persistent opts(a[0]);
    std::string weights = getPropertyString(opts.get(), "weights");
    if (weights.empty())
        return ev::throwTypeError("bro.kws.load: opts.weights (PhonemeNet checkpoint path) required");
    // init() BEFORE the device probe — the GPU backends only register on the
    // driver probe, so a probe before init() silently lands on CPU.
    brotensor::Device dev;
    try {
        dev = autoDevice();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.kws.load: ") + e.what());
    }
    {
        std::string err;
        if (!parseDeviceOpt(opts.get(), dev, err))
            return ev::throwTypeError("bro.kws.load: " + err);
    }

    try {
        // Load the net once and share it (read-only) across every stream's
        // spotter. The weights live here exactly once.
        std::shared_ptr<const brosoundml::PhonemeNet> net;
        {
            brotensor::DeviceScope scope(dev);
            net = std::make_shared<const brosoundml::PhonemeNet>(
                brosoundml::PhonemeNet::load(resolvePath(weights), dev));
        }

        // Replacing the net invalidates existing tenant spotters — drop them.
        unloadAll();

        brosoundml::SpotterConfig cfg;            // struct defaults
        readPolicy(opts.get(), cfg);

        g_kws.net           = std::move(net);
        g_kws.device        = dev;
        g_kws.defaultPolicy = cfg;
        logInfo("[kws] PhonemeNet loaded on " + std::string(deviceName(dev)) +
                " (K=" + std::to_string(g_kws.net->class_map().num_classes) +
                ", " + std::to_string(g_kws.net->config().sample_rate) +
                " Hz, threshold=" + std::to_string(cfg.threshold) +
                "), shared across streams");
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.kws.load: ") + e.what());
    }
}

Value jsUnload(Value, std::span<const Value>) {
    unloadAll();
    return ev::undefined();
}

// bro.kws.enroll(name, phonemeIds, policy?) -> template length
//   phonemeIds: Int32Array/number[] of Kokoro phoneme ids — exactly what
//   bro.tts.phonemize(text) returns. Silence/suprasegmental ids are dropped
//   and duplicate adjacent classes collapsed by the library. Enrolls on the
//   tenant for THIS stream (bro.kws → default mic; stream.kws → that stream).
Value jsEnroll(Value self, std::span<const Value> a) {
    KwsTenant* t = ensureTenant(streamOf(self));
    if (!t) return ev::throwError("bro.kws.enroll: call bro.kws.load first");
    Value thrown;
    if (refuseWhileListening(t, "enroll", thrown)) return thrown;
    if (a.size() < 2 || !isStringArg(a, 0))
        return ev::throwTypeError("bro.kws.enroll(name, phonemeIds, policy?): name and ids required");
    const std::string name = strAt(a, 0);
    std::vector<int32_t> ids = readInt32Array(a[1]);
    if (ids.empty())
        return ev::throwTypeError(
            "bro.kws.enroll: phonemeIds must be a non-empty Int32Array or number[] "
            "(use bro.tts.phonemize(text))");
    try {
        brosoundml::SpotterConfig pol;
        const bool hasPol = readPolicyOverride(a, 2, *t->spotter, pol);
        const std::vector<int> idv(ids.begin(), ids.end());
        const int len = t->spotter->enroll(name, idv, hasPol ? &pol : nullptr);
        if (len <= 0)
            return ev::throwError("bro.kws.enroll: '" + name +
                "' produced an empty template (only silence/suprasegmental ids?)");
        return ev::fromDouble(len);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.kws.enroll: ") + e.what());
    }
}

// bro.kws.enrollFromAudio(name, samples, policy?) -> template length
//   samples: Float32Array of mono PCM at bro.kws.sampleRate() — runs the model
//   over the reference audio and uses the argmax class sequence as the template.
Value jsEnrollFromAudio(Value self, std::span<const Value> a) {
    KwsTenant* t = ensureTenant(streamOf(self));
    if (!t || !t->spotter->loaded())
        return ev::throwError("bro.kws.enrollFromAudio: call bro.kws.load first");
    Value thrown;
    if (refuseWhileListening(t, "enrollFromAudio", thrown)) return thrown;
    if (a.size() < 2 || !isStringArg(a, 0))
        return ev::throwTypeError(
            "bro.kws.enrollFromAudio(name, samples, policy?): name and samples required");
    const std::string name = strAt(a, 0);
    bool ok = false;
    std::vector<float> samples = readPcmArg(a, 1, ok);
    if (!ok || samples.empty())
        return ev::throwTypeError("bro.kws.enrollFromAudio: samples must be a non-empty Float32Array");
    try {
        brosoundml::SpotterConfig pol;
        const bool hasPol = readPolicyOverride(a, 2, *t->spotter, pol);
        brotensor::DeviceScope scope(g_kws.device);
        const int len = t->spotter->enroll_from_audio(
            name, samples.data(), static_cast<int>(samples.size()), hasPol ? &pol : nullptr);
        if (len <= 0)
            return ev::throwError("bro.kws.enrollFromAudio: '" + name +
                "' produced an empty template (silence-only audio?)");
        return ev::fromDouble(len);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.kws.enrollFromAudio: ") + e.what());
    }
}

// bro.kws.enrollFromClasses(name, classIds, policy?) -> template length
//   classIds: Int32Array/number[] of phoneme-CLASS ids (already in [0,K) — the
//   matcher's own alphabet, e.g. an edited sequence from bro.kws.inspect). The
//   library drops the silence class (0) and collapses adjacent duplicates. This
//   is the re-enroll path for an edited template.
Value jsEnrollFromClasses(Value self, std::span<const Value> a) {
    KwsTenant* t = ensureTenant(streamOf(self));
    if (!t) return ev::throwError("bro.kws.enrollFromClasses: call bro.kws.load first");
    Value thrown;
    if (refuseWhileListening(t, "enrollFromClasses", thrown)) return thrown;
    if (a.size() < 2 || !isStringArg(a, 0))
        return ev::throwTypeError(
            "bro.kws.enrollFromClasses(name, classIds, policy?): name and classIds required");
    const std::string name = strAt(a, 0);
    std::vector<int32_t> ids = readInt32Array(a[1]);
    if (ids.empty())
        return ev::throwTypeError(
            "bro.kws.enrollFromClasses: classIds must be a non-empty Int32Array or number[]");
    try {
        brosoundml::SpotterConfig pol;
        const bool hasPol = readPolicyOverride(a, 2, *t->spotter, pol);
        const std::vector<int> idv(ids.begin(), ids.end());
        const int len = t->spotter->enroll_from_classes(name, idv, hasPol ? &pol : nullptr);
        if (len <= 0)
            return ev::throwError("bro.kws.enrollFromClasses: '" + name +
                "' produced an empty template (only silence-class ids?)");
        return ev::fromDouble(len);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.kws.enrollFromClasses: ") + e.what());
    }
}

Value jsRemove(Value self, std::span<const Value> a) {
    KwsTenant* t = findTenant(streamOf(self));
    if (!t) return ev::throwError("bro.kws.remove: nothing enrolled");
    Value thrown;
    if (refuseWhileListening(t, "remove", thrown)) return thrown;
    if (!isStringArg(a, 0)) return ev::throwTypeError("bro.kws.remove(name): name required");
    return ev::fromBool(t->spotter->remove(strAt(a, 0)));
}

Value jsClear(Value self, std::span<const Value>) {
    KwsTenant* t = findTenant(streamOf(self));
    if (!t) return ev::undefined();
    Value thrown;
    if (refuseWhileListening(t, "clear", thrown)) return thrown;
    t->spotter->clear();
    return ev::undefined();
}

Value jsTemplates(Value self, std::span<const Value>) {
    KwsTenant* t = findTenant(streamOf(self));
    if (!t) return makeStringArray({});
    // While listening, return the snapshot (the live template list shares the
    // feed thread); otherwise read the spotter directly.
    return makeStringArray(t->listening ? t->names : t->spotter->templates());
}

// bro.kws.inspect(name) -> { name, threshold, frameMs, hasGaps, states:[...] }
//   or null if no such template. Safe to call while listening (reads immutable
//   per-template structure).
Value jsInspect(Value self, std::span<const Value> a) {
    KwsTenant* t = findTenant(streamOf(self));
    if (!t) return ev::null();
    if (!isStringArg(a, 0)) return ev::throwTypeError("bro.kws.inspect(name): name required");
    brosoundml::TemplateView view;
    if (!t->spotter->inspect(strAt(a, 0), view)) return ev::null();

    const auto& cm = t->spotter->class_map();
    const int   K  = cm.num_classes;

    ObjectBuilder obj;
    obj.set("name", view.name);
    obj.set("threshold", static_cast<double>(view.threshold));
    obj.set("frameMs", static_cast<double>(view.frame_ms));
    obj.set("hasGaps", view.has_gaps);
    {
        ev::Persistent states(hostArrayOf(view.states.size(), [&](size_t i) {
            const brosoundml::TemplateState& st = view.states[i];
            const char* label =
                st.gap ? "gap"
                       : (st.cls >= 0 && st.cls < K &&
                          st.cls < static_cast<int>(cm.class_names.size()))
                             ? cm.class_names[static_cast<std::size_t>(st.cls)].c_str()
                             : "?";
            ObjectBuilder tv;
            tv.set("cls", static_cast<double>(st.cls));
            tv.set("label", label);
            tv.set("gap", st.gap);
            tv.set("gapLo", static_cast<double>(st.gap_lo));
            tv.set("gapHi", static_cast<double>(st.gap_hi));
            return tv.get();
        }));
        obj.set("states", states.get());
    }
    return obj.get();
}

Value jsReset(Value self, std::span<const Value>) {
    KwsTenant* t = findTenant(streamOf(self));
    if (!t) return ev::undefined();
    Value thrown;
    if (refuseWhileListening(t, "reset", thrown)) return thrown;
    t->spotter->reset();
    return ev::undefined();
}

// bro.kws.listen({ onSpot }) — start live spotting on THIS stream. Requires a
// loaded net and at least one enrolled template on this stream.
Value jsListen(Value self, std::span<const Value> a) {
    const StreamId sid = streamOf(self);
    KwsTenant* t = ensureTenant(sid);
    if (!t || !t->spotter->loaded())
        return ev::throwError("bro.kws.listen: call bro.kws.load first");
    if (t->listening)
        return ev::throwError("bro.kws.listen: this stream is already listening (stop() first)");
    if (!isObjectArg(a, 0))
        return ev::throwTypeError("bro.kws.listen(opts): opts object required");
    ev::Persistent onSpot = getFunctionOpt(a[0], "onSpot");
    if (!ev::isFunction(onSpot.get()))
        return ev::throwTypeError("bro.kws.listen: opts.onSpot (function) required");
    if (!listenHostAudioAvailable())
        return ev::throwError("bro.kws.listen: audio engine not available");

    const std::vector<std::string> names = t->spotter->templates();
    if (names.empty())
        return ev::throwError(
            "bro.kws.listen: no templates enrolled on this stream (bro.kws.enroll first)");

    try {
        t->names  = names;
        t->onSpot = onSpot;
        t->produced.store(0, std::memory_order_relaxed);
        t->drained.store(0, std::memory_order_relaxed);
        t->suspended.store(false, std::memory_order_relaxed);

        // Join the listen host on this stream. The stream's single pump drives
        // the bus (mel -> one PhonemeNet forward) and calls this hook on the
        // inference thread with whatever fired. Captures the tenant pointer
        // (address-stable in g_kws.tenants) and the name snapshot (the enrolled
        // set cannot change while listening, so name -> index lookups stay
        // valid); the tenant's atomics live until it is dropped.
        listenStreamSetSpotter(
            sid, t->spotter, g_kws.device,
            [t, names](const std::vector<brosoundml::SpotEvent>& events) {
                if (t->suspended.load(std::memory_order_relaxed)) return;
                for (const auto& ev : events) {
                    const int idx = nameIndexOf(names, ev.name);
                    if (idx >= 0)
                        publishEvent(t, idx, ev.confidence, ev.start_frame, ev.end_frame);
                }
            });
        t->listening = true;

        logInfo("[kws] listening on stream " + std::to_string(sid) +
                " (device=" + deviceName(g_kws.device) + ", " +
                std::to_string(names.size()) + (names.size() == 1 ? " template" : " templates") +
                ", model=" + std::to_string(t->spotter->sample_rate()) + " Hz)");
        return ev::undefined();
    } catch (const std::exception& e) {
        t->listening = true;   // let stopListening clear the partial state
        stopListening(t);
        return ev::throwError(std::string("bro.kws.listen: ") + e.what());
    }
}

Value jsStop(Value self, std::span<const Value>) {
    if (KwsTenant* t = findTenant(streamOf(self))) stopListening(t);
    return ev::undefined();
}

Value jsSuspend(Value self, std::span<const Value>) {
    if (KwsTenant* t = findTenant(streamOf(self)))
        t->suspended.store(true, std::memory_order_relaxed);
    return ev::undefined();
}

Value jsResume(Value self, std::span<const Value>) {
    if (KwsTenant* t = findTenant(streamOf(self)))
        t->suspended.store(false, std::memory_order_relaxed);
    return ev::undefined();
}

Value jsIsActive(Value self, std::span<const Value>) {
    KwsTenant* t = findTenant(streamOf(self));
    return ev::fromBool(t && t->listening);
}

Value jsIsSuspended(Value self, std::span<const Value>) {
    KwsTenant* t = findTenant(streamOf(self));
    return ev::fromBool(t && t->suspended.load(std::memory_order_relaxed));
}

Value jsIsLoaded(Value, std::span<const Value>) {
    return ev::fromBool(static_cast<bool>(g_kws.net));
}

Value jsSampleRate(Value, std::span<const Value>) {
    return ev::fromDouble(g_kws.net ? g_kws.net->config().sample_rate : 0);
}

// Best current prefix progress across this stream's templates, [0,1] — a
// lock-free library read, safe while the inference thread feeds. For UI meters.
Value jsPrefixProgress(Value self, std::span<const Value>) {
    KwsTenant* t = findTenant(streamOf(self));
    return ev::fromDouble(t ? t->spotter->prefix_progress() : 0.0);
}

// Per-template alignment telemetry for THIS stream — the spotter's contribution
// to the fused listening surface. One coherent lock-free snapshot. Null until
// this stream has a spotter (i.e. something was enrolled / it listened).
Value jsProgress(Value self, std::span<const Value>) {
    KwsTenant* t = findTenant(streamOf(self));
    if (!t) return ev::null();
    const brosoundml::ProgressSnapshot s = t->spotter->progress_snapshot();
    ObjectBuilder obj;
    obj.set("frames", static_cast<double>(s.frames));
    obj.set("generation", static_cast<double>(s.generation));
    {
        ev::Persistent arr(hostArrayOf(static_cast<size_t>(s.count), [&](size_t i) {
            const brosoundml::TemplateProgress& e = s.templates[i];
            ObjectBuilder tv;
            tv.set("name", e.name);
            tv.set("matched", static_cast<double>(e.matched));
            tv.set("length", static_cast<double>(e.length));
            tv.set("progress", static_cast<double>(e.progress));
            tv.set("confidence", static_cast<double>(e.confidence));
            tv.set("completions", static_cast<double>(e.completions));
            tv.set("lastAdvanceFrame", static_cast<double>(e.last_advance_frame));
            tv.set("lastFireFrame", static_cast<double>(e.last_fire_frame));
            return tv.get();
        }));
        obj.set("templates", arr.get());
    }
    return obj.get();
}

// bro.kws.posterior(topK=3) -> { frame, top: [{ cls, label, p }, ...] } or null.
// The model's RAW per-frame readout for THIS stream — what PhonemeNet is hearing
// now, independent of any template. last_posterior() is a lock-free seqlock
// read, safe while the inference thread feeds.
Value jsPosterior(Value self, std::span<const Value> a) {
    KwsTenant* t = findTenant(streamOf(self));
    if (!t || !t->spotter->loaded()) return ev::null();
    const std::vector<float> post = t->spotter->last_posterior();
    if (post.empty()) return ev::null();

    int topK = 3;
    if (!a.empty() && ev::isNumber(a[0])) {
        const int k = static_cast<int>(ev::toDouble(a[0]));
        if (k > 0) topK = k;
    }
    const int K = static_cast<int>(post.size());
    if (topK > K) topK = K;

    const auto& cm = t->spotter->class_map();
    // Partial top-K by repeated argmax (K is small — a few dozen classes).
    std::vector<int> taken(static_cast<std::size_t>(K), 0);
    struct Top { int cls; float p; };
    std::vector<Top> top;
    for (int r = 0; r < topK; ++r) {
        int best = -1;
        float bestP = -1.0f;
        for (int c = 0; c < K; ++c) {
            if (taken[static_cast<std::size_t>(c)]) continue;
            if (post[static_cast<std::size_t>(c)] > bestP) { bestP = post[static_cast<std::size_t>(c)]; best = c; }
        }
        if (best < 0) break;
        taken[static_cast<std::size_t>(best)] = 1;
        top.push_back(Top{best, bestP});
    }
    ObjectBuilder obj;
    obj.set("frame", static_cast<double>(t->spotter->progress_snapshot().frames));
    {
        ev::Persistent arr(hostArrayOf(top.size(), [&](size_t i) {
            const char* label = (top[i].cls < static_cast<int>(cm.class_names.size()))
                                    ? cm.class_names[static_cast<std::size_t>(top[i].cls)].c_str() : "?";
            ObjectBuilder e;
            e.set("cls", static_cast<double>(top[i].cls));
            e.set("label", label);
            e.set("p", static_cast<double>(top[i].p));
            return e.get();
        }));
        obj.set("top", arr.get());
    }
    return obj.get();
}

// Diagnostic surface over THIS stream's mic tap (cf. bro.wake.stats). Null for a
// non-mic (loopback) stream or when not listening.
Value jsStats(Value self, std::span<const Value>) {
    KwsTenant* t = findTenant(streamOf(self));
    if (!t || !t->listening) return ev::null();
    return makeTapStats(t->streamId);
}

Value makeEventArray(const std::vector<brosoundml::SpotEvent>& events) {
    return hostArrayOf(events.size(), [&](size_t i) {
        ObjectBuilder o;
        o.set("name", events[i].name);
        o.set("confidence", static_cast<double>(events[i].confidence));
        return o.get();
    });
}

// Manual feed for tests / scripted scenarios on THIS stream. Samples must
// already be at the spotter's rate. Mirrors bro.wake.feed's mode split:
//   - Headless (no inference worker): the stream's bus runs synchronously on
//     this thread — it is ONE stream, so the feed advances every attached
//     tenant (bro.sense included) — and the fired events come back as
//     [{name, confidence}]. Suspended fires are still returned (the caller
//     asked) but not queued for onSpot.
//   - Threaded: samples go into the stream's live ring; events surface via
//     onSpot. Returns undefined.
// Refuses to run while live MIC capture is active (two-producer race on the
// ring). Loopback streams: prefer headless/inline for scripted feeds.
Value jsFeed(Value self, std::span<const Value> a) {
    KwsTenant* t = findTenant(streamOf(self));
    if (!t || !t->listening)
        return ev::throwError("bro.kws.feed: this stream is not listening");
    if (listenHostMicCapturing())
        return ev::throwError(
            "bro.kws.feed: cannot feed while live mic capture is active "
            "(feed is for headless/offline use; the live tap already writes the ring)");
    bool ok = false;
    std::vector<float> samples = readPcmArg(a, 0, ok);
    if (!ok) return ev::throwTypeError("bro.kws.feed(Float32Array)");
    const int n = static_cast<int>(samples.size());
    if (listenHostThreaded()) {
        listenStreamWriteRing(t->streamId, samples.data(), n);
        return ev::undefined();
    }
    brosoundml::ListenFeedResult r;
    try {
        // The host runs the device scope and delivers spots through the same
        // onSpots hook the live path uses (suspended-gated there).
        r = listenStreamFeedInline(t->streamId, samples.data(), n);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.kws.feed: ") + e.what());
    }
    return makeEventArray(r.spots);
}

// Drain one tenant's SPSC event ring into its onSpot callback (main thread).
// Re-resolves the tenant after every call: an onSpot may stop() / unload() it.
void drainTenant(StreamId id) {
    KwsTenant* t = findTenant(id);
    if (!t || !t->listening) return;
    const std::uint64_t produced = t->produced.load(std::memory_order_acquire);
    std::uint64_t drained = t->drained.load(std::memory_order_relaxed);
    if (drained >= produced || !ev::isFunction(t->onSpot.get())) return;
    while (drained < produced) {
        const int          idx    = t->eventIdx[drained % kEventSlots];
        const float        conf   = t->eventConf[drained % kEventSlots];
        const std::int64_t startF = t->eventStart[drained % kEventSlots];
        const std::int64_t endF   = t->eventEnd[drained % kEventSlots];
        drained++;
        // Publish the consumption BEFORE the JS call: the producer only needs
        // the slot back, and onSpot may run for a while.
        t->drained.store(drained, std::memory_order_release);
        if (idx < 0 || idx >= static_cast<int>(t->names.size())) continue;
        // 3rd arg: the matched span on the frames axis (align with
        // bro.kws.progress().frames / bro.sense frames). Backward-compatible —
        // existing onSpot(name, conf) handlers ignore it.
        ev::Persistent cb(t->onSpot);
        ev::Persistent name(ev::fromUtf8(t->names[static_cast<std::size_t>(idx)]));
        ev::Persistent span(makeSpan(startF, endF));
        const Value args[3] = {name.get(), ev::fromDouble(conf), span.get()};
        callTenantCallback(cb, std::span<const Value>(args, 3), "kws");
        t = findTenant(id);
        if (!t || !t->listening) break;
    }
}

void defineKwsOps(ObjectBuilder& b) {
    b.def("enroll",            3, jsEnroll);
    b.def("enrollFromAudio",   3, jsEnrollFromAudio);
    b.def("enrollFromClasses", 3, jsEnrollFromClasses);
    b.def("inspect",           1, jsInspect);
    b.def("remove",            1, jsRemove);
    b.def("clear",             0, jsClear);
    b.def("templates",         0, jsTemplates);
    b.def("reset",             0, jsReset);
    b.def("listen",            1, jsListen);
    b.def("stop",              0, jsStop);
    b.def("suspend",           0, jsSuspend);
    b.def("resume",            0, jsResume);
    b.def("isActive",          0, jsIsActive);
    b.def("isSuspended",       0, jsIsSuspended);
    b.def("isLoaded",          0, jsIsLoaded);
    b.def("sampleRate",        0, jsSampleRate);
    b.def("prefixProgress",    0, jsPrefixProgress);
    b.def("progress",          0, jsProgress);
    b.def("posterior",         1, jsPosterior);
    b.def("stats",             0, jsStats);
    b.def("feed",              1, jsFeed);
}

}  // namespace

Value makeKwsView(StreamId id) {
    auto* v = new KwsView{id};
    return g_kwsViewClass.make(v, [](void* p) { delete static_cast<KwsView*>(p); });
}

void tickKws() {
    std::vector<StreamId> ids;
    ids.reserve(g_kws.tenants.size());
    for (const auto& kv : g_kws.tenants) ids.push_back(kv.first);
    for (StreamId id : ids) {
        auto it = g_kws.tenants.find(id);
        if (it == g_kws.tenants.end()) continue;
        // Prune a tenant whose stream has closed (handle .close()'d or GC'd).
        // The stream's teardown removed its pump (a barrier), so the onSpots
        // closure can no longer run — safe to drop the tenant. Default mic is
        // never invalid, so its tenant is never pruned here.
        if (!listenHostValid(id)) {
            stopListening(it->second.get());   // detach is a no-op — stream gone
            g_kws.tenants.erase(it);
            continue;
        }
        drainTenant(id);
    }
}

void cleanupKws() {
    unloadAll();
}

void installKws(ObjectBuilder& bro) {
    g_kwsViewClass.install("KwsStreamView", 0, nullptr,
        [](ObjectBuilder& b) {
            b.accessor("active", [](Value self, std::span<const Value>) -> Value {
                KwsTenant* t = findTenant(streamOf(self));
                return ev::fromBool(t && t->listening);
            });
            defineKwsOps(b);
        },
        /*global=*/false);

    ObjectBuilder kws;
    kws.def("init", 0, [](Value, std::span<const Value>) -> Value {
        try {
            brotensor::init();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("bro.kws.init: ") + e.what());
        }
        return ev::undefined();
    });
    // Namespace ops (shared net — not stream-scoped).
    kws.def("load",   1, jsLoad);
    kws.def("unload", 0, jsUnload);
    // Per-stream ops — on bro.kws they target the shared default-mic stream.
    defineKwsOps(kws);
    kws.set("KwsStreamView", g_kwsViewClass.constructor());
    bro.set("kws", kws.get());
}

}  // namespace brosoundml::api
