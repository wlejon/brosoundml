#pragma once

// The endless stream's JS shapes (native_soundml_music_endless_js.cpp), for
// its binding (native_soundml_music_endless.cpp).

#include "soundml_music_internal.h"

#include <brosoundml/yue2_endless.h>

namespace brosoundml::api::music {

// { style, score, knobs, tag, at } merged into `t` (fields left out keep their
// value; score null = none). lyrics / roll / lyricWindow are an error (removed).
bool readTarget(Value v, brosoundml::Yue2EndlessTarget& t, std::string& err);
Value makeTarget(const brosoundml::Yue2EndlessTarget& t);
// An array of targets (each merged into the one before it, from `from`).
bool readPath(Value v, const brosoundml::Yue2EndlessTarget& from, std::vector<brosoundml::Yue2EndlessTarget>& out,
              std::string& err);
bool readEndlessOptions(Value v, brosoundml::Yue2EndlessOptions& o, std::string& err);
Value makeEndlessEvent(const brosoundml::Yue2EndlessEvent& e);
Value makeEndlessStats(const brosoundml::Yue2EndlessStats& s);
Value makeChunkRow(const brosoundml::Yue2EndlessChunk& r);

}  // namespace brosoundml::api::music
