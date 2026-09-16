#pragma once

// TRANSITIONAL: the helper names the pre-port stubs still use, over the
// current host_soundml_internal.h. Each stub drops this include as it is
// replaced by its real binding; delete this header with the last one.

#include "host_soundml_internal.h"

namespace brosoundml::api {

inline std::vector<float> extractFloatAudio(Value v) {
    brosoundml::AudioBuffer buf;
    std::string err;
    if (!readAudioBuffer(v, buf, err)) return {};
    return std::move(buf.samples);
}

inline std::vector<int32_t> extractInt32Array(Value v) {
    return readInt32Array(v);
}

inline Value makeAsyncHandle() {
    ObjectBuilder b;
    b.def("cancel", 0, [](Value, std::span<const Value>) -> Value { return ev::undefined(); });
    return b.get();
}

inline brotensor::Device parseDeviceOpt(Value opts) {
    brotensor::Device dev = autoDevice();
    std::string err;
    parseDeviceOpt(opts, dev, err);
    return dev;
}

inline void triggerCallback(Value cb, Value arg1, Value arg2 = ev::undefined()) {
    if (!ev::isFunction(cb)) return;
    if (!ev::isUndefined(arg2)) callCallback2(cb, arg1, arg2);
    else callCallback1(cb, arg1);
}

} // namespace brosoundml::api
