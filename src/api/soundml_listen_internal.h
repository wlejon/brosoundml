#pragma once

// The seams between the listen-host tenants (bro.wake / kws / sense / gesture)
// and bro.listen. Each tenant file owns its namespace object, its per-stream
// view class and its tenant registry; bro.listen mints the views for a
// stream handle, and api.cpp pumps the deliveries and tears everything down.
//
// Every tenant follows one shape:
//   - A per-stream VIEW object (`stream.kws`, `stream.wake`, ...) carries only
//     the stream id. The namespace functions and the view methods are the
//     SAME bodies: called on a view they address that stream, called on the
//     namespace (bro.kws.enroll(...)) they address the shared default-mic
//     stream. streamOf(self) is that seam.
//   - A TENANT holds the stream's model / hub, its rooted JS callback and an
//     SPSC delivery the inference thread publishes into; tick*() drains it on
//     the JS thread. Tenants whose stream has closed are pruned on the tick.
//   - Single-producer discipline: while a stream is listening its model is
//     fed from the inference pump only, so the mutators (enroll / remove /
//     clear / reset) refuse until stop() detaches it (a scheduler barrier).

#include "host_soundml_internal.h"
#include "soundml_listen_host.h"

#include <cstdio>

namespace brosoundml::api {

// Per-stream view instances for a ListenStream handle.
Value makeWakeView(StreamId id);
Value makeKwsView(StreamId id);
Value makeSenseView(StreamId id);
Value makeGestureView(StreamId id);

// Deliver pending fires / spots / gestures to their JS callbacks (frame pump).
void tickWake();
void tickKws();
void tickGesture();

// Stop every tenant, release its rooted callbacks and drop its models. Runs
// before the listen host itself shuts down.
void cleanupWake();
void cleanupKws();
void cleanupSense();
void cleanupGesture();

// Shared helpers for the tenants' JS surfaces.

// A Float32Array / Float64Array / number[] argument as PCM. `ok` reports the
// shape; an empty vector with ok==true is a legitimately empty clip.
inline std::vector<float> readPcmArg(std::span<const Value> a, size_t i, bool& ok) {
    ok = false;
    if (i >= a.size()) return {};
    return readFloat32Array(a[i], &ok);
}

// { framesDelivered, samplesDelivered, rollingPeak } from the stream's mic
// tap, or null when the stream has no tap (a loopback stream, no audio
// engine, infra down). `extra` decorates the object before it is returned.
inline Value makeTapStats(StreamId id, const std::function<void(ObjectBuilder&)>& extra = nullptr) {
    broaudio::MicTapStats s;
    if (!listenStreamTapStats(id, s)) return ev::null();
    ObjectBuilder o;
    o.set("framesDelivered", static_cast<double>(s.framesDelivered));
    o.set("samplesDelivered", static_cast<double>(s.samplesDelivered));
    o.set("rollingPeak", static_cast<double>(s.rollingPeak));
    if (extra) extra(o);
    return o.get();
}

// { startFrame, endFrame, matchedFrames } — a matched span on the frames axis.
inline Value makeSpan(std::int64_t startF, std::int64_t endF) {
    ObjectBuilder span;
    span.set("startFrame", static_cast<double>(startF));
    span.set("endFrame", static_cast<double>(endF));
    span.set("matchedFrames", static_cast<double>(endF >= startF ? endF - startF + 1 : 0));
    return span.get();
}

// Call a rooted JS callback, reporting (not propagating) anything it throws.
inline void callTenantCallback(const ev::Persistent& cb, std::span<const Value> args, const char* who) {
    if (!ev::isFunction(cb.get())) return;
    ev::CallResult r = ev::call(cb.get(), ev::undefined(), args);
    if (!r.thrown) return;
    std::string what = "?";
    if (ev::isObject(r.value)) {
        Value msg = ev::getProperty(r.value, "message");
        if (ev::isString(msg)) what = ev::toUtf8(msg);
    } else if (ev::isString(r.value)) {
        what = ev::toUtf8(r.value);
    }
    std::fprintf(stderr, "[ERROR] [%s] callback threw: %s\n", who, what.c_str());
}

}  // namespace brosoundml::api
