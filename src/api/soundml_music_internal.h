#pragma once

// bro.music's shared pieces: the model handles and the option / audio
// shapes native_soundml_music.cpp defines, for the stream binding
// (native_soundml_music_stream.cpp).

#include "host_soundml_internal.h"

#include <brosoundml/yue2.h>
#include <brosoundml/yue2_vae.h>

#include <memory>
#include <span>
#include <string>

namespace brosoundml::api::music {

struct HostYue2 {
    std::shared_ptr<brosoundml::Yue2> model;
    brotensor::Device device = brotensor::Device::CPU;
    std::string arWeights = "bf16";   // the AR decode weights it was loaded with
    ModelGate busy;
};

struct HostYue2Vae {
    std::shared_ptr<brosoundml::Yue2Vae> vae;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;
};

HostYue2* yue2Of(Value v);
HostYue2Vae* vaeOf(Value v);

// Params common to every entry point; false + err on a shape error.
bool readParams(Value opts, brosoundml::Yue2Params& p, std::string& err);
// { samples (planar Float32Array), channels, frames, sampleRate }
Value makeStereo(const brosoundml::StereoAudio& a);
// { abc, abcIds (Int32Array), prefix (Int32Array), truncated }
Value makePlan(const brosoundml::Yue2Plan& plan);

// model.stream(opts) -> YuE2Stream (native_soundml_music_stream.cpp)
Value yue2Stream(Value self, std::span<const Value> a);
void installStreamClass();
// { energy, texture, percussion, arrangement, density, timbre: { min, max } }: the clamps
// Yue2Knobs::clamp applies, as a stream's knobRanges and bro.music.knobRanges.
Value knobRanges();

}  // namespace brosoundml::api::music
