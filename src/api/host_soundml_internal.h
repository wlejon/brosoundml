#pragma once

#include "embed/embed.h"
#include "host_class.h"
#include "object_builder.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <brotensor/tensor.h>

namespace brosoundml::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

// ---------------------------------------------------------------------------
// Argument Extraction Helpers
// ---------------------------------------------------------------------------

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

inline uint32_t u32At(std::span<const Value> args, size_t i, uint32_t def = 0) {
    return static_cast<uint32_t>(static_cast<int64_t>(numAt(args, i, def)));
}

inline bool boolAt(std::span<const Value> args, size_t i, bool def = false) {
    if (i >= args.size() || ev::isUndefined(args[i])) return def;
    return ev::toBool(args[i]);
}

inline std::string strAt(std::span<const Value> args, size_t i, const std::string& def = "") {
    if (i >= args.size() || ev::isUndefined(args[i]) || ev::isNull(args[i]) || ev::isSymbol(args[i])) return def;
    return ev::toUtf8(args[i]);
}

inline Value argAt(std::span<const Value> args, size_t i) {
    return i < args.size() ? args[i] : ev::undefined();
}

inline bool hasArg(std::span<const Value> args, size_t i) {
    return i < args.size() && !ev::isUndefined(args[i]);
}

inline double getPropertyDouble(Value obj, std::string_view name, double def = 0.0) {
    if (!ev::isObject(obj)) return def;
    Value v = ev::getProperty(obj, name);
    if (ev::isUndefined(v) || ev::isNull(v)) return def;
    double d = ev::toDouble(v);
    return std::isnan(d) ? def : d;
}

inline int32_t getPropertyInt(Value obj, std::string_view name, int32_t def = 0) {
    return static_cast<int32_t>(getPropertyDouble(obj, name, def));
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
    if (ev::isUndefined(v) || ev::isNull(v) || !ev::isString(v)) return def;
    return ev::toUtf8(v);
}

// ---------------------------------------------------------------------------
// TypedArray & Buffer Helpers
// ---------------------------------------------------------------------------

inline Value makeFloat32Array(std::span<const float> data) {
    Value view = ev::createTypedArray(ev::elements::Float32, static_cast<uint32_t>(data.size()));
    if (!data.empty()) {
        std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(data.data()), data.size_bytes());
        ev::fillTypedArray(view, bytes);
    }
    return view;
}

inline Value makeInt32Array(std::span<const int32_t> data) {
    Value view = ev::createTypedArray(ev::elements::Int32, static_cast<uint32_t>(data.size()));
    if (!data.empty()) {
        std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(data.data()), data.size_bytes());
        ev::fillTypedArray(view, bytes);
    }
    return view;
}

inline std::vector<float> extractFloatAudio(Value v) {
    if (ev::isObject(v)) {
        Value samplesProp = ev::getProperty(v, "samples");
        if (!ev::isUndefined(samplesProp) && ev::isTypedArray(samplesProp)) {
            auto info = ev::typedArrayInfo(samplesProp);
            if (info.data && info.bytesPerElement == sizeof(float)) {
                const float* ptr = reinterpret_cast<const float*>(info.data);
                return std::vector<float>(ptr, ptr + info.elementCount);
            }
        }
    }
    if (ev::isTypedArray(v)) {
        auto info = ev::typedArrayInfo(v);
        if (info.data && info.bytesPerElement == sizeof(float)) {
            const float* ptr = reinterpret_cast<const float*>(info.data);
            return std::vector<float>(ptr, ptr + info.elementCount);
        }
    }
    if (ev::isObject(v)) {
        Value lenVal = ev::getProperty(v, "length");
        if (!ev::isUndefined(lenVal) && !ev::isObject(lenVal)) {
            uint32_t len = static_cast<uint32_t>(ev::toDouble(lenVal));
            std::vector<float> res;
            res.reserve(len);
            for (uint32_t i = 0; i < len; ++i) {
                res.push_back(static_cast<float>(ev::toDouble(ev::getElement(v, i))));
            }
            return res;
        }
    }
    return {};
}

inline std::vector<int32_t> extractInt32Array(Value v) {
    if (ev::isTypedArray(v)) {
        auto info = ev::typedArrayInfo(v);
        if (info.data && info.bytesPerElement == sizeof(int32_t)) {
            const int32_t* ptr = reinterpret_cast<const int32_t*>(info.data);
            return std::vector<int32_t>(ptr, ptr + info.elementCount);
        }
    }
    if (ev::isObject(v)) {
        Value lenVal = ev::getProperty(v, "length");
        if (!ev::isUndefined(lenVal) && !ev::isObject(lenVal)) {
            uint32_t len = static_cast<uint32_t>(ev::toDouble(lenVal));
            std::vector<int32_t> res;
            res.reserve(len);
            for (uint32_t i = 0; i < len; ++i) {
                res.push_back(static_cast<int32_t>(ev::toDouble(ev::getElement(v, i))));
            }
            return res;
        }
    }
    return {};
}

inline Value makeAsyncHandle() {
    ObjectBuilder b;
    b.def("cancel", 0, [](Value, std::span<const Value>) -> Value {
        return ev::undefined();
    });
    return b.get();
}

inline brotensor::Device parseDeviceOpt(Value opts) {
    if (!ev::isObject(opts)) return brotensor::Device::CPU;
    std::string dev = getPropertyString(opts, "device", "CPU");
    if (dev == "cuda" || dev == "CUDA") return brotensor::Device::CUDA;
    if (dev == "metal" || dev == "Metal") return brotensor::Device::Metal;
    return brotensor::Device::CPU;
}

inline std::string deviceName(const brotensor::Device& dev) {
    if (dev == brotensor::Device::CUDA) return "CUDA";
    if (dev == brotensor::Device::Metal) return "Metal";
    return "CPU";
}

inline void triggerCallback(Value cb, Value arg1, Value arg2 = ev::undefined()) {
    if (!ev::isFunction(cb)) return;
    if (!ev::isUndefined(arg2)) {
        const Value args[2] = {arg1, arg2};
        ev::call(cb, ev::undefined(), std::span<const Value>(args, 2));
    } else {
        const Value args[1] = {arg1};
        ev::call(cb, ev::undefined(), std::span<const Value>(args, 1));
    }
}

// ---------------------------------------------------------------------------
// Module Installer Declarations
// ---------------------------------------------------------------------------

void installStt(ObjectBuilder& bro);
void installTts(ObjectBuilder& bro);
void installDiar(ObjectBuilder& bro);
void installRave(ObjectBuilder& bro);
void installWakeAndKws(ObjectBuilder& bro);
void installListenSenseGesture(ObjectBuilder& bro);

// Views shared between wake/kws and listen
Value createWakeStreamViewHandle();
Value createKwsStreamViewHandle();
Value createSenseStreamViewHandle();
Value createGestureStreamViewHandle();

} // namespace brosoundml::api
