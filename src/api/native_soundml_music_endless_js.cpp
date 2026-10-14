// The endless stream's JS shapes: targets, options, chunk rows, events and
// stats (soundml_music_endless.h), and bro.music.beat (the beat sensor,
// brosoundml/beat.h).

#include "soundml_music_endless.h"

#include <brosoundml/beat.h>

#include <algorithm>
#include <cmath>

namespace brosoundml::api::music {

namespace {

using brosoundml::Yue2EndlessEvent;

const char* kindName(Yue2EndlessEvent::Kind k) {
    using K = Yue2EndlessEvent::Kind;
    switch (k) {
        case K::Planned: return "planned";
        case K::Sampled: return "sampled";
        case K::Built: return "built";
        case K::StyleChange: return "styleChange";
        case K::PhaseFit: return "phaseFit";
        case K::Heard: return "heard";
        case K::BreakWarning: return "breakWarning";
        case K::ChunkStart: return "chunkStart";
        case K::Branch: return "branch";
        case K::Committed: return "committed";
        case K::Discarded: return "discarded";
        case K::Knobs: return "knobs";
        case K::Error: return "error";
        case K::Done: return "done";
    }
    return "done";
}

void readSeconds(Value o, std::string_view name, double& dst) { getDoubleOpt(o, name, dst); }

// A string property, or null / '' for none; false when it is something else.
bool readText(Value o, std::string_view name, std::string& dst, std::string& err) {
    ev::Persistent v(ev::getProperty(o, name));
    if (ev::isUndefined(v.get())) return true;
    if (ev::isNull(v.get())) {
        dst.clear();
        return true;
    }
    if (!ev::isString(v.get())) {
        err = std::string(name) + " must be a string or null";
        return false;
    }
    dst = ev::toUtf8(v.get());
    return true;
}

Value intArray(const std::vector<int>& v) {
    return hostArrayOf(v.size(), [&v](size_t i) { return ev::fromDouble(v[i]); });
}

}  // namespace

bool readTarget(Value v, brosoundml::Yue2EndlessTarget& t, std::string& err) {
    if (!ev::isObject(v)) {
        err = "a target must be an object ({ style, score, lyrics, roll, knobs, at })";
        return false;
    }
    ev::Persistent o(v);
    if (!readText(o.get(), "style", t.style, err) || !readText(o.get(), "score", t.score, err) ||
        !readText(o.get(), "lyrics", t.lyrics, err) || !readText(o.get(), "tag", t.tag, err))
        return false;
    getBoolOpt(o.get(), "roll", t.roll);
    getIntOpt(o.get(), "lyricWindow", t.lyric_window);
    if (t.lyric_window < 1 || t.lyric_window > 32) {
        err = "lyricWindow must be 1..32";
        return false;
    }
    {
        ev::Persistent k(ev::getProperty(o.get(), "knobs"));
        if (ev::isObject(k.get())) readKnobs(k.get(), t.knobs);
        t.knobs = brosoundml::Yue2Knobs::clamp(t.knobs);
    }
    t.at = -1;
    readSeconds(o.get(), "at", t.at);
    return true;
}

Value makeTarget(const brosoundml::Yue2EndlessTarget& t) {
    ObjectBuilder o;
    o.set("style", t.style);
    if (t.score.empty()) o.set("score", ev::null());
    else o.set("score", t.score);
    o.set("lyrics", t.lyrics);
    o.set("roll", t.roll);
    o.set("lyricWindow", static_cast<double>(t.lyric_window));
    {
        ev::Persistent k(makeKnobs(t.knobs));
        o.set("knobs", k.get());
    }
    o.set("tag", t.tag);
    if (t.at >= 0) o.set("at", t.at);
    return o.get();
}

bool readPath(Value v, const brosoundml::Yue2EndlessTarget& from, std::vector<brosoundml::Yue2EndlessTarget>& out,
              std::string& err) {
    ev::Persistent list(v);
    if (!ev::isObject(list.get())) {
        err = "a path must be an array of targets";
        return false;
    }
    const int n = getPropertyInt(list.get(), "length", 0);
    brosoundml::Yue2EndlessTarget prev = from;
    for (int i = 0; i < n; ++i) {
        ev::Persistent e(ev::getElement(list.get(), static_cast<uint32_t>(i)));
        brosoundml::Yue2EndlessTarget t = prev;
        if (!readTarget(e.get(), t, err)) {
            err = "path[" + std::to_string(i) + "]: " + err;
            return false;
        }
        if (t.at < 0) {
            err = "path[" + std::to_string(i) + "]: at (program seconds) is required";
            return false;
        }
        out.push_back(t);
        prev = t;
    }
    return true;
}

bool readEndlessOptions(Value v, brosoundml::Yue2EndlessOptions& o, std::string& err) {
    if (!ev::isObject(v)) return true;
    ev::Persistent r(v);
    getIntOpt(r.get(), "chunkFrames", o.chunk_frames);
    getIntOpt(r.get(), "memory", o.memory);
    getIntOpt(r.get(), "changeMemory", o.change_memory);
    getIntOpt(r.get(), "lookahead", o.lookahead);
    getIntOpt(r.get(), "narLeft", o.nar_left);
    getIntOpt(r.get(), "odeSteps", o.ode_steps);
    getFloatOpt(r.get(), "cfgScale", o.cfg_scale);
    {
        ev::Persistent s(ev::getProperty(r.get(), "seed"));
        if (ev::isNumber(s.get())) o.seed = static_cast<std::uint64_t>(static_cast<int64_t>(ev::toDouble(s.get())));
    }
    {
        std::string m;
        getStrOpt(r.get(), "changeMode", m);
        if (m == "auto") o.change_mode = brosoundml::Yue2ChangeMode::Auto;
        else if (m == "drop") o.change_mode = brosoundml::Yue2ChangeMode::Drop;
        else if (m == "bridge") o.change_mode = brosoundml::Yue2ChangeMode::Bridge;
        else if (m == "plain") o.change_mode = brosoundml::Yue2ChangeMode::Plain;
        else if (!m.empty()) {
            err = "changeMode must be 'auto', 'drop', 'bridge' or 'plain'";
            return false;
        }
    }
    getIntOpt(r.get(), "bridgeCandidates", o.bridge_candidates);
    getIntOpt(r.get(), "bridgeScoreFrames", o.bridge_score_frames);
    getBoolOpt(r.get(), "paced", o.paced);
    readSeconds(r.get(), "emitLeadSeconds", o.emit_lead_seconds);
    readSeconds(r.get(), "decodeLeadSeconds", o.decode_lead_seconds);
    readSeconds(r.get(), "aheadSeconds", o.ahead_seconds);
    readSeconds(r.get(), "bridgeLeadSeconds", o.bridge_lead_seconds);
    getBoolOpt(r.get(), "retarget", o.retarget);
    getIntOpt(r.get(), "maxChunks", o.max_chunks);
    getIntOpt(r.get(), "maxBranches", o.max_branches);
    readSeconds(r.get(), "branchFadeSeconds", o.branch_fade_seconds);
    getBoolOpt(r.get(), "chunkData", o.chunk_data);
    {
        ev::Persistent t(ev::getProperty(r.get(), "target"));
        if (!ev::isUndefined(t.get()) && !readTarget(t.get(), o.target, err)) {
            err = "target: " + err;
            return false;
        }
        o.target.at = -1;
    }
    {
        ev::Persistent p(ev::getProperty(r.get(), "path"));
        if (!ev::isUndefined(p.get()) && !ev::isNull(p.get()) && !readPath(p.get(), o.target, o.path, err)) return false;
    }
    if (!o.paced && o.max_chunks <= 0) {
        err = "an unpaced stream (paced: false) needs maxChunks";
        return false;
    }
    return true;
}

Value makeChunkRow(const brosoundml::Yue2EndlessChunk& c) {
    ObjectBuilder o;
    o.set("index", static_cast<double>(c.index));
    o.set("branch", static_cast<double>(c.branch));
    o.set("tag", c.tag);
    o.set("style", c.style);
    o.set("scored", c.scored);
    o.set("change", c.change);
    o.set("how", c.how);
    o.set("memory", static_cast<double>(c.memory));
    o.set("retargeted", c.retargeted);
    o.set("spliceSeconds", c.splice_seconds);
    o.set("inpaintSeconds", c.inpaint_seconds);
    o.set("decodeSeconds", c.decode_seconds);
    o.set("listenSeconds", c.listen_seconds);
    o.set("narWindow", static_cast<double>(c.nar_window));
    o.set("probeRender", c.probe_render);
    o.set("latentNorm", c.latent_norm);
    o.set("rendered", c.rendered);
    if (!c.bridge_scores.empty()) {
        ev::Persistent s(makeFloat32Array(c.bridge_scores));
        o.set("bridgeScores", s.get());
    }
    if (c.phase_fit) {
        ObjectBuilder f;
        f.set("P", c.period);
        f.set("streamPhase", c.stream_phase);
        f.set("freshPhase", c.fresh_phase);
        f.set("cut", static_cast<double>(c.cut));
        f.set("seconds", c.fit_seconds);
        ev::Persistent fp(f.get());
        o.set("phaseFit", fp.get());
        o.set("trim", static_cast<double>(c.cut));
    }
    o.set("lyrics", c.lyrics);
    if (c.lyric_from >= 0) {
        ev::Persistent l(intArray({c.lyric_from, c.lyric_to}));
        o.set("lyricLines", l.get());
    }
    if (!c.heard.empty() || !c.heard_lines.empty()) {
        o.set("heardText", c.heard);
        ev::Persistent h(intArray(c.heard_lines));
        o.set("heardLines", h.get());
    }
    {
        ev::Persistent k(makeKnobs(c.knobs));
        o.set("knobs", k.get());
    }
    return o.get();
}

Value makeEndlessEvent(const brosoundml::Yue2EndlessEvent& e) {
    ObjectBuilder o;
    o.set("kind", kindName(e.kind));
    o.set("chunk", static_cast<double>(e.chunk));
    o.set("branch", static_cast<double>(e.branch));
    o.set("tag", e.tag);
    o.set("sample", static_cast<double>(e.sample));
    o.set("seconds", e.seconds);
    o.set("value", e.value);
    o.set("text", e.text);
    if (e.row) {
        ev::Persistent r(makeChunkRow(*e.row));
        o.set("row", r.get());
    }
    if (e.tokens) {
        ev::Persistent t(makeInt32Array(*e.tokens));
        o.set("tokens", t.get());
    }
    if (e.latents) {
        ev::Persistent l(makeFloat32Array(*e.latents));
        o.set("latents", l.get());
        o.set("frame0", static_cast<double>(e.frame0));
    }
    return o.get();
}

Value makeEndlessStats(const brosoundml::Yue2EndlessStats& s) {
    ObjectBuilder o;
    o.set("playhead", static_cast<double>(s.playhead));
    o.set("emitted", static_cast<double>(s.emitted));
    o.set("decoded", static_cast<double>(s.decoded));
    o.set("playing", static_cast<double>(s.playing));
    o.set("sampled", static_cast<double>(s.sampled));
    o.set("rendered", static_cast<double>(s.rendered));
    o.set("committed", static_cast<double>(s.committed));
    o.set("openChunk", static_cast<double>(s.open_chunk));
    o.set("openSample", static_cast<double>(s.open_sample));
    o.set("aheadSeconds", s.ahead_seconds);
    {
        ev::Persistent t(makeTarget(s.target));
        o.set("target", t.get());
    }
    o.set("path", static_cast<double>(s.path));
    {
        ev::Persistent c(hostArrayOf(s.chunks.size(), [&](size_t i) { return makeChunkRow(s.chunks[i]); }));
        o.set("chunks", c.get());
    }
    {
        ev::Persistent b(hostArrayOf(s.branches.size(), [&](size_t i) -> Value {
            const auto& x = s.branches[i];
            ObjectBuilder bo;
            bo.set("id", static_cast<double>(x.id));
            bo.set("fork", static_cast<double>(x.fork));
            bo.set("sampled", static_cast<double>(x.sampled));
            bo.set("rendered", static_cast<double>(x.rendered));
            bo.set("tag", x.tag);
            bo.set("committing", x.committing);
            return bo.get();
        }));
        o.set("branches", b.get());
    }
    o.set("worstGapSeconds", s.worst_gap_seconds);
    {
        ev::Persistent g(hostArrayOf(s.gaps.size(), [&](size_t i) -> Value {
            const auto& x = s.gaps[i];
            ObjectBuilder go;
            go.set("seconds", x.seconds);
            go.set("what", x.what);
            go.set("chunk", static_cast<double>(x.song));
            go.set("emitted", static_cast<double>(x.emitted));
            go.set("lead", x.lead);
            return go.get();
        }));
        o.set("gaps", g.get());
    }
    {
        ObjectBuilder wo;
        for (const auto& w : s.work) wo.set(w.what.c_str(), w.seconds);
        ev::Persistent w(wo.get());
        o.set("work", w.get());
    }
    {
        ObjectBuilder co;
        co.set("splice", s.splice_seconds);
        co.set("render", s.render_seconds);
        co.set("decode", s.decode_seconds);
        co.set("listen", s.listen_seconds);
        ev::Persistent c(co.get());
        o.set("costSeconds", c.get());
    }
    return o.get();
}

// bro.music.beat(audio, { period | bpm, sensor?, searchPeriod?, envelope? })
//   audio: a mono Float32Array at 48 kHz, or { samples, channels?, frames?,
//   sampleRate } (planar stereo averaged to mono). -> { phase, period,
//   periodFrames, firstBeat, strength, contrast, beats, envelope? }
Value musicBeat(Value, std::span<const Value> a) {
    if (!hasArg(a, 0)) return ev::throwTypeError("beat(audio, opts): audio required");
    ev::Persistent audio(a[0]);
    ev::Persistent opts(isObjectArg(a, 1) ? a[1] : ev::undefined());
    std::vector<float> mono;
    int rate = 48000;
    if (ev::isTypedArray(audio.get())) {
        mono = readFloat32Array(audio.get());
    } else if (ev::isObject(audio.get())) {
        bool ok = false;
        std::vector<float> s = readFloat32Array(ev::getProperty(audio.get(), "samples"), &ok);
        if (!ok) return ev::throwTypeError("beat: audio.samples must be a Float32Array");
        const int ch = getPropertyInt(audio.get(), "channels", 1);
        rate = getPropertyInt(audio.get(), "sampleRate", 48000);
        if (ch == 2) {
            const std::size_t frames = s.size() / 2;
            mono = brosoundml::beat_mono(s.data(), frames, 0, frames);
        } else if (ch == 1) {
            mono = std::move(s);
        } else {
            return ev::throwTypeError("beat: audio.channels must be 1 or 2");
        }
    } else {
        return ev::throwTypeError("beat(audio, opts): audio must be a Float32Array or { samples, channels, sampleRate }");
    }
    if (rate != 48000) return ev::throwTypeError("beat: audio must be 48 kHz");
    double period = 0, bpm = 0;
    bool search = false, wantEnv = false;
    std::string sensor = "spectral";
    if (ev::isObject(opts.get())) {
        getDoubleOpt(opts.get(), "period", period);
        getDoubleOpt(opts.get(), "bpm", bpm);
        getBoolOpt(opts.get(), "searchPeriod", search);
        getBoolOpt(opts.get(), "envelope", wantEnv);
        getStrOpt(opts.get(), "sensor", sensor);
    }
    if (period <= 0 && bpm > 0) period = 60.0 / bpm;
    if (!(period > 0)) return ev::throwTypeError("beat: opts.period (seconds) or opts.bpm required");
    if (sensor != "spectral" && sensor != "energy") return ev::throwTypeError("beat: sensor must be 'spectral' or 'energy'");
    const auto env = brosoundml::beat_onsets(mono.data(), mono.size(),
                                             sensor == "energy" ? brosoundml::BeatSensor::Energy : brosoundml::BeatSensor::Spectral);
    const brosoundml::BeatFit f = brosoundml::fit_beat(env, 0, period * 100.0, search);
    ObjectBuilder o;
    o.set("phase", f.phase);
    o.set("period", f.period / 100.0);
    o.set("periodFrames", f.period);
    o.set("firstBeat", f.phase * f.period / 100.0);
    o.set("strength", f.strength);
    o.set("contrast", f.contrast);
    o.set("beats", static_cast<double>(f.beats));
    if (wantEnv) {
        ev::Persistent e(makeFloat32Array(env));
        o.set("envelope", e.get());
    }
    return o.get();
}

}  // namespace brosoundml::api::music
