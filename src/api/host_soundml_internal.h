#pragma once

// Shared plumbing for the brosoundml JS bindings (bro.stt / tts / diar / rave
// / wake / kws / sense / gesture / listen). Every namespace file includes
// this and nothing else of its siblings.
//
// Conventions the bindings follow:
//   - Argument shape errors are TypeErrors, thrown synchronously and naming
//     the argument ("loadWhisper(modelDir, opts?): path required").
//   - Runtime failures (a missing model dir, a model that threw) are plain
//     Errors carrying the entry point as a prefix ("loadWhisper: ...").
//   - Loaders run on the GPU by default (CUDA, then Metal, then CPU) and
//     honour opts.device = 'cpu' | 'cuda' | 'metal'; anything else is a
//     TypeError. No model has a CPU fallback the caller did not ask for.
//   - A heavy call with an onDone / onReady callback runs on a background
//   thread through soundml_async.h; the same call without one blocks.

#include "api.h"
#include "embed/embed.h"
#include "host_class.h"
#include "object_builder.h"
#include "soundml_async.h"

#include <brosoundml/audio.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace brosoundml::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

// ---------------------------------------------------------------------------
// Argument extraction
// ---------------------------------------------------------------------------

inline Value argAt(std::span<const Value> args, size_t i) {
    return i < args.size() ? args[i] : ev::undefined();
}

inline bool hasArg(std::span<const Value> args, size_t i) {
    return i < args.size() && !ev::isUndefined(args[i]);
}

inline double numAt(std::span<const Value> args, size_t i, double def = 0.0) {
    if (i >= args.size()) return def;
    Value v = args[i];
    if (ev::isObject(v) || ev::isUndefined(v) || ev::isNull(v)) return def;
    double d = ev::toDouble(v);
    return std::isnan(d) ? def : d;
}

inline int32_t i32At(std::span<const Value> args, size_t i, int32_t def = 0) {
    return static_cast<int32_t>(static_cast<int64_t>(numAt(args, i, def)));
}

inline bool boolAt(std::span<const Value> args, size_t i, bool def = false) {
    if (i >= args.size() || ev::isUndefined(args[i])) return def;
    return ev::toBool(args[i]);
}

// The string at `i`, or "" for anything that is not a string.
inline std::string strAt(std::span<const Value> args, size_t i) {
    if (i >= args.size() || !ev::isString(args[i])) return "";
    return ev::toUtf8(args[i]);
}

inline bool isStringArg(std::span<const Value> args, size_t i) {
    return i < args.size() && ev::isString(args[i]);
}

inline bool isObjectArg(std::span<const Value> args, size_t i) {
    return i < args.size() && ev::isObject(args[i]);
}

// ---------------------------------------------------------------------------
// Option-object readers. The *Opt forms leave `dst` untouched when the key is
// absent (undefined / null) so a struct of upstream defaults survives a
// partial options object.
// ---------------------------------------------------------------------------

inline bool hasProperty(Value obj, std::string_view name) {
    if (!ev::isObject(obj)) return false;
    Value v = ev::getProperty(obj, name);
    return !ev::isUndefined(v) && !ev::isNull(v);
}

inline double getPropertyDouble(Value obj, std::string_view name, double def = 0.0) {
    if (!ev::isObject(obj)) return def;
    Value v = ev::getProperty(obj, name);
    if (ev::isUndefined(v) || ev::isNull(v) || ev::isObject(v)) return def;
    double d = ev::toDouble(v);
    return std::isnan(d) ? def : d;
}

inline int32_t getPropertyInt(Value obj, std::string_view name, int32_t def = 0) {
    return static_cast<int32_t>(static_cast<int64_t>(getPropertyDouble(obj, name, def)));
}

inline bool getPropertyBool(Value obj, std::string_view name, bool def = false) {
    if (!ev::isObject(obj)) return def;
    Value v = ev::getProperty(obj, name);
    if (ev::isUndefined(v) || ev::isNull(v)) return def;
    return ev::toBool(v);
}

inline std::string getPropertyString(Value obj, std::string_view name, const std::string& def = "") {
    if (!ev::isObject(obj)) return def;
    Value v = ev::getProperty(obj, name);
    if (!ev::isString(v)) return def;
    return ev::toUtf8(v);
}

inline void getIntOpt(Value obj, std::string_view name, int& dst) {
    if (!ev::isObject(obj)) return;
    Value v = ev::getProperty(obj, name);
    if (ev::isNumber(v)) dst = static_cast<int>(ev::toDouble(v));
}

inline void getFloatOpt(Value obj, std::string_view name, float& dst) {
    if (!ev::isObject(obj)) return;
    Value v = ev::getProperty(obj, name);
    if (ev::isNumber(v)) dst = static_cast<float>(ev::toDouble(v));
}

inline void getDoubleOpt(Value obj, std::string_view name, double& dst) {
    if (!ev::isObject(obj)) return;
    Value v = ev::getProperty(obj, name);
    if (ev::isNumber(v)) dst = ev::toDouble(v);
}

inline void getBoolOpt(Value obj, std::string_view name, bool& dst) {
    if (!ev::isObject(obj)) return;
    Value v = ev::getProperty(obj, name);
    if (!ev::isUndefined(v) && !ev::isNull(v)) dst = ev::toBool(v);
}

inline void getStrOpt(Value obj, std::string_view name, std::string& dst) {
    if (!ev::isObject(obj)) return;
    Value v = ev::getProperty(obj, name);
    if (ev::isString(v)) dst = ev::toUtf8(v);
}

// A callback property, rooted. Persistent over `undefined` when absent or
// not a function; test with isFunction(p.get()).
inline ev::Persistent getFunctionOpt(Value obj, std::string_view name) {
    if (!ev::isObject(obj)) return ev::Persistent();
    Value v = ev::getProperty(obj, name);
    if (!ev::isFunction(v)) return ev::Persistent();
    return ev::Persistent(v);
}

// ---------------------------------------------------------------------------
// Typed arrays
// ---------------------------------------------------------------------------

inline Value makeFloat32Array(std::span<const float> data) {
    Value view = ev::createTypedArray(ev::elements::Float32, static_cast<uint32_t>(data.size()));
    if (!data.empty()) {
        std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(data.data()), data.size_bytes());
        ev::fillTypedArray(view, bytes);
    }
    return view;
}

inline Value makeFloat32Array(const std::vector<float>& v) {
    return makeFloat32Array(std::span<const float>(v.data(), v.size()));
}

inline Value makeInt32Array(std::span<const int32_t> data) {
    Value view = ev::createTypedArray(ev::elements::Int32, static_cast<uint32_t>(data.size()));
    if (!data.empty()) {
        std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(data.data()), data.size_bytes());
        ev::fillTypedArray(view, bytes);
    }
    return view;
}

inline Value makeInt32Array(const std::vector<int32_t>& v) {
    return makeInt32Array(std::span<const int32_t>(v.data(), v.size()));
}

inline Value makeStringArray(const std::vector<std::string>& v) {
    return hostArrayOf(v.size(), [&v](size_t i) { return ev::fromUtf8(v[i]); });
}

// A Float32Array view, or a number[] copied element by element. Empty for
// anything else. `ok` says whether the value had an array shape at all.
inline std::vector<float> readFloat32Array(Value v, bool* ok = nullptr) {
    if (ok) *ok = false;
    if (ev::isTypedArray(v)) {
        auto info = ev::typedArrayInfo(v);
        if (info.data && info.elementKind == ev::elements::Float32) {
            if (ok) *ok = true;
            const float* p = reinterpret_cast<const float*>(info.data);
            return std::vector<float>(p, p + info.elementCount);
        }
        if (info.data && info.elementKind == ev::elements::Float64) {
            if (ok) *ok = true;
            const double* p = reinterpret_cast<const double*>(info.data);
            std::vector<float> out(info.elementCount);
            for (size_t i = 0; i < info.elementCount; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        return {};
    }
    if (ev::isObject(v) && !ev::isFunction(v)) {
        Value lenVal = ev::getProperty(v, "length");
        if (!ev::isNumber(lenVal)) return {};
        if (ok) *ok = true;
        ev::Persistent root(v);
        uint32_t len = static_cast<uint32_t>(ev::toDouble(lenVal));
        std::vector<float> out;
        out.reserve(len);
        for (uint32_t i = 0; i < len; ++i) {
            out.push_back(static_cast<float>(ev::toDouble(ev::getElement(root.get(), i))));
        }
        return out;
    }
    return {};
}

// An Int32Array (or any integer typed array) view, or a number[] copied.
inline std::vector<int32_t> readInt32Array(Value v, bool* ok = nullptr) {
    if (ok) *ok = false;
    if (ev::isTypedArray(v)) {
        auto info = ev::typedArrayInfo(v);
        if (!info.data) return {};
        if (ok) *ok = true;
        std::vector<int32_t> out(info.elementCount);
        const auto kind = info.elementKind;
        const size_t n = info.elementCount;
        auto widen = [&](auto* p) { for (size_t i = 0; i < n; ++i) out[i] = static_cast<int32_t>(p[i]); };
        if (kind == ev::elements::Int32) {
            const int32_t* p = reinterpret_cast<const int32_t*>(info.data);
            out.assign(p, p + n);
        } else if (kind == ev::elements::Uint32) {
            widen(reinterpret_cast<const uint32_t*>(info.data));
        } else if (kind == ev::elements::Int16) {
            widen(reinterpret_cast<const int16_t*>(info.data));
        } else if (kind == ev::elements::Uint16) {
            widen(reinterpret_cast<const uint16_t*>(info.data));
        } else if (kind == ev::elements::Int8) {
            widen(reinterpret_cast<const int8_t*>(info.data));
        } else if (kind == ev::elements::Uint8 || kind == ev::elements::Uint8Clamped) {
            widen(reinterpret_cast<const uint8_t*>(info.data));
        } else if (kind == ev::elements::Float32) {
            widen(reinterpret_cast<const float*>(info.data));
        } else if (kind == ev::elements::Float64) {
            widen(reinterpret_cast<const double*>(info.data));
        } else {
            if (ok) *ok = false;
            return {};
        }
        return out;
    }
    if (ev::isObject(v) && !ev::isFunction(v)) {
        Value lenVal = ev::getProperty(v, "length");
        if (!ev::isNumber(lenVal)) return {};
        if (ok) *ok = true;
        ev::Persistent root(v);
        uint32_t len = static_cast<uint32_t>(ev::toDouble(lenVal));
        std::vector<int32_t> out;
        out.reserve(len);
        for (uint32_t i = 0; i < len; ++i) {
            out.push_back(static_cast<int32_t>(ev::toDouble(ev::getElement(root.get(), i))));
        }
        return out;
    }
    return {};
}

// A Uint8Array (any 1-byte typed array) or a number[] / boolean[] as bytes.
inline std::vector<uint8_t> readByteArray(Value v) {
    std::vector<uint8_t> out;
    if (ev::isTypedArray(v)) {
        auto info = ev::typedArrayInfo(v);
        if (info.data && info.bytesPerElement == 1) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(info.data);
            out.assign(p, p + info.elementCount);
            return out;
        }
        auto ints = readInt32Array(v);
        out.reserve(ints.size());
        for (int32_t x : ints) out.push_back(x != 0 ? 1 : 0);
        return out;
    }
    if (ev::isObject(v) && !ev::isFunction(v)) {
        Value lenVal = ev::getProperty(v, "length");
        if (!ev::isNumber(lenVal)) return out;
        ev::Persistent root(v);
        uint32_t len = static_cast<uint32_t>(ev::toDouble(lenVal));
        out.reserve(len);
        for (uint32_t i = 0; i < len; ++i) {
            out.push_back(ev::toBool(ev::getElement(root.get(), i)) ? 1 : 0);
        }
    }
    return out;
}

// Read a JS audio argument — a bare Float32Array (assumed `defaultRate`) or
// { samples: Float32Array|number[], sampleRate } — into an AudioBuffer.
// False + `err` on a shape error (the caller throws a TypeError).
inline bool readAudioBuffer(Value v, brosoundml::AudioBuffer& out, std::string& err,
                            int defaultRate = 16000) {
    out.sample_rate = defaultRate;
    out.samples.clear();
    if (ev::isTypedArray(v)) {
        bool ok = false;
        out.samples = readFloat32Array(v, &ok);
        if (!ok) {
            err = "audio must be a Float32Array or { samples, sampleRate } object";
            return false;
        }
        return true;
    }
    if (!ev::isObject(v) || ev::isFunction(v)) {
        err = "audio must be a Float32Array or { samples, sampleRate } object";
        return false;
    }
    ev::Persistent root(v);
    Value sr = ev::getProperty(root.get(), "sampleRate");
    if (ev::isNumber(sr)) out.sample_rate = static_cast<int>(ev::toDouble(sr));
    Value s = ev::getProperty(root.get(), "samples");
    bool ok = false;
    out.samples = readFloat32Array(s, &ok);
    if (!ok) {
        err = "audio.samples must be a Float32Array or number[]";
        return false;
    }
    return true;
}

inline Value audioBufferToJs(const brosoundml::AudioBuffer& buf) {
    ObjectBuilder res;
    {
        ev::Persistent s(makeFloat32Array(buf.samples));
        res.set("samples", s.get());
    }
    res.set("sampleRate", static_cast<double>(buf.sample_rate));
    return res.get();
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

// The default device — CUDA, then Metal, then CPU. Calls brotensor::init()
// first (idempotent) so the GPU probes have run.
inline brotensor::Device autoDevice() {
    brotensor::init();
    if (brotensor::is_available(brotensor::Device::CUDA))  return brotensor::Device::CUDA;
    if (brotensor::is_available(brotensor::Device::Metal)) return brotensor::Device::Metal;
    return brotensor::Device::CPU;
}

inline const char* deviceName(brotensor::Device d) {
    switch (d.type) {
        case brotensor::DeviceType::CUDA:  return "CUDA";
        case brotensor::DeviceType::Metal: return "Metal";
        case brotensor::DeviceType::CPU:   return "CPU";
    }
    return "?";
}

// Parse opts.device. Missing key: `out` untouched, true. A string naming a
// device: `out` set, true. Anything else: `err` set, false — the caller
// throws a TypeError. `explicitDevice` reports whether the key was given.
inline bool parseDeviceOpt(Value opts, brotensor::Device& out, std::string& err,
                           bool* explicitDevice = nullptr) {
    if (explicitDevice) *explicitDevice = false;
    if (!ev::isObject(opts)) return true;
    Value v = ev::getProperty(opts, "device");
    if (ev::isUndefined(v) || ev::isNull(v)) return true;
    if (!ev::isString(v)) {
        err = "opts.device must be a string ('cpu', 'cuda', or 'metal')";
        return false;
    }
    std::string sv = ev::toUtf8(v);
    if (explicitDevice) *explicitDevice = true;
    if (sv == "cpu" || sv == "CPU")     { out = brotensor::Device::CPU;   return true; }
    if (sv == "cuda" || sv == "CUDA")   { out = brotensor::Device::CUDA;  return true; }
    if (sv == "metal" || sv == "Metal") { out = brotensor::Device::Metal; return true; }
    err = "opts.device must be 'cpu', 'cuda', or 'metal' (got '" + sv + "')";
    return false;
}

// ---------------------------------------------------------------------------
// Single-owner gate: one op in flight per model. Copyable — a session shares
// its model's gate — so every inference over one set of weights serializes
// on the ONE flag (brosoundml's GPU session tier is shared weights /
// serialized decode).
// ---------------------------------------------------------------------------

class ModelGate {
public:
    bool tryClaim() {
        bool expected = false;
        return busy_->compare_exchange_strong(expected, true);
    }
    void release() { busy_->store(false, std::memory_order_release); }
    bool isBusy() const { return busy_->load(std::memory_order_acquire); }

private:
    std::shared_ptr<std::atomic<bool>> busy_ = std::make_shared<std::atomic<bool>>(false);
};

// ---------------------------------------------------------------------------
// Host hooks
// ---------------------------------------------------------------------------

// Resolve a model / asset path the way the host's fs module does
// (app-relative base paths). Identity until the host installs a resolver
// (brosoundml::api::setPathResolver).
std::string resolvePath(const std::string& path);

// Log line on the host's [INFO] channel (stderr when no host hook).
void logInfo(const std::string& line);

// The host's broaudio engine and inference scheduler (api.h setters); null /
// empty until the host provides them.
broaudio::Engine* audioEngine();
const InferenceScheduler& inferenceScheduler();

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

// Call `cb(args...)` and swallow anything it throws (a callback's exception
// must not unwind through native frames). The values in `args` must be raw
// values obtained since the last allocation — root them first.
inline void callCallback(Value cb, std::span<const Value> args) {
    if (!ev::isFunction(cb)) return;
    ev::call(cb, ev::undefined(), args);
}

inline void callCallback1(Value cb, Value a0) {
    const Value args[1] = {a0};
    callCallback(cb, std::span<const Value>(args, 1));
}

inline void callCallback2(Value cb, Value a0, Value a1) {
    const Value args[2] = {a0, a1};
    callCallback(cb, std::span<const Value>(args, 2));
}

// Object.keys(obj) as strings (own enumerable string keys), for the few
// option bags keyed by user-chosen names (a logit-bias map).
inline std::vector<std::string> objectKeys(Value obj) {
    std::vector<std::string> out;
    if (!ev::isObject(obj)) return out;
    ev::Persistent root(obj);
    auto objectCtor = ev::globalValue("Object");
    if (!objectCtor.found || !ev::isObject(objectCtor.value)) return out;
    ev::Persistent keysFn(ev::getProperty(objectCtor.value, "keys"));
    if (!ev::isFunction(keysFn.get())) return out;
    const Value args[1] = {root.get()};
    ev::CallResult r = ev::call(keysFn.get(), ev::undefined(), std::span<const Value>(args, 1));
    if (r.thrown || !ev::isObject(r.value)) return out;
    ev::Persistent arr(r.value);
    const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(arr.get(), "length")));
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) out.push_back(ev::toUtf8(ev::getElement(arr.get(), i)));
    return out;
}

// A model's CancelCheck from the async job's cancel flag. The check runs
// synchronously inside the model call on the work thread, so capturing the
// flag by reference is safe.
inline brosoundml::CancelCheck cancelCheckOf(const std::atomic<bool>& cancel) {
    return [&cancel] { return cancel.load(std::memory_order_acquire); };
}

// { cancelled, error? } — the second argument of every onDone.
inline Value makeDoneInfo(bool cancelled, const std::string& error) {
    ObjectBuilder info;
    info.set("cancelled", cancelled);
    if (!error.empty()) info.set("error", error);
    return info.get();
}

// ---------------------------------------------------------------------------
// Lock-free SPSC slot ring: the work thread (sole writer) publishes items
// through `produced`; the JS-thread poll drains up to it. Pre-sized so the
// writer never reallocates while the reader reads; an overrun drops the item
// (the full result still arrives through onDone).
// ---------------------------------------------------------------------------

template <typename T>
struct SpscSlots {
    std::vector<T>      slots;
    std::atomic<size_t> produced{0};
    size_t              drained = 0;

    void reserve(size_t n) { slots.resize(n); }

    // Work thread.
    bool push(const T& item) {
        const size_t idx = produced.load(std::memory_order_relaxed);
        if (idx >= slots.size()) return false;
        slots[idx] = item;
        produced.store(idx + 1, std::memory_order_release);
        return true;
    }
    template <typename F>
    bool emplace(F&& fill) {
        const size_t idx = produced.load(std::memory_order_relaxed);
        if (idx >= slots.size()) return false;
        fill(slots[idx]);
        produced.store(idx + 1, std::memory_order_release);
        return true;
    }

    // JS thread: visit every item published since the last drain.
    template <typename F>
    void drain(F&& visit) {
        const size_t n = produced.load(std::memory_order_acquire);
        while (drained < n) {
            visit(slots[drained]);
            ++drained;
        }
    }
};

// ---------------------------------------------------------------------------
// Module installers (one per namespace file)
// ---------------------------------------------------------------------------

void installStt(ObjectBuilder& bro);
void installTts(ObjectBuilder& bro);
void installDiar(ObjectBuilder& bro);
void installRave(ObjectBuilder& bro);
// The listen-host tenants (soundml_listen_internal.h has their seams).
void installWake(ObjectBuilder& bro);
void installKws(ObjectBuilder& bro);
void installSense(ObjectBuilder& bro);
void installGesture(ObjectBuilder& bro);
void installListen(ObjectBuilder& bro);

} // namespace brosoundml::api
