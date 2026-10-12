// model.stream(opts) -> YuE2Stream: streaming playback with a continuous
// radio and live steering over brosoundml::Yue2Stream (yue2_stream.h).
//
// The stream's worker runs on a background thread for its whole life
// (launchAsyncJob). It claims the model's and the VAE's gates until it ends.
// Audio chunks, events and (optionally) latents cross to the JS thread
// through a mutex-guarded queue, which each tick's poll drains. The data is
// paced by the player (emitLeadSeconds ahead), so it stays small.
// Controls (enqueue, setPlayhead, setKnobs, interrupt, skip, setNextPlan,
// setPlan, drop, makeFull, setAutoFull, stop) are Yue2Stream's own thread-safe calls, so they never block on the
// model. While the stream runs, the job roots the stream object, so a
// stream nobody references keeps playing until stop().
#include "soundml_music_internal.h"
#include "soundml_async.h"

#include <brosoundml/yue2_stream.h>

#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <variant>

namespace brosoundml::api::music {

namespace {

HostClass g_streamClass;

struct AudioItem {
    brosoundml::StereoAudio audio;
    std::int64_t sample0 = 0;
};
struct LatItem {
    int song = 0, frame0 = 0, frames = 0;
    std::vector<float> data;
};
using Item = std::variant<AudioItem, brosoundml::Yue2StreamEvent, LatItem>;

struct StreamJob {
    std::shared_ptr<brosoundml::Yue2> model;
    std::shared_ptr<brosoundml::Yue2Vae> vae;
    ModelGate gate, vaeGate;
    std::unique_ptr<brosoundml::Yue2Stream> stream;
    brosoundml::Yue2Knobs knobs;                     // JS side: what setKnobs merges into
    std::atomic<const std::atomic<bool>*> cancel{nullptr};
    std::atomic<bool> running{true};

    std::mutex mu;   // work -> JS
    std::deque<Item> out;

    bool wantLat = false;
    ev::Persistent onAudio, onEvent, onLatents, onDone, modelRef, vaeRef, selfRef;
};

struct HostStream {
    std::shared_ptr<StreamJob> job;
    ~HostStream() {
        if (job && job->stream && job->running.load()) job->stream->stop();
    }
};

StreamJob* jobOf(Value v) {
    auto* h = g_streamClass.isInstance(v) ? static_cast<HostStream*>(g_streamClass.unwrap(v)) : nullptr;
    return h ? h->job.get() : nullptr;
}

const char* kindName(brosoundml::Yue2StreamEvent::Kind k) {
    using K = brosoundml::Yue2StreamEvent::Kind;
    switch (k) {
        case K::Plan: return "plan";
        case K::SongStart: return "songStart";
        case K::SemanticDone: return "semanticDone";
        case K::Window: return "window";
        case K::Interrupt: return "interrupt";
        case K::Knobs: return "knobs";
        case K::BreakWarning: return "breakWarning";
        case K::SongEnd: return "songEnd";
        case K::FirstAudio: return "firstAudio";
        case K::Error: return "error";
        case K::Idle: return "idle";
        case K::Want: return "want";
        case K::Built: return "built";
        case K::Dropped: return "dropped";
        case K::Full: return "full";
    }
    return "idle";
}

// Knobs are flat in JS: energy, texture, percussion and one field per
// semantic axis (arrangement, then the style axes).
void readKnobs(Value v, brosoundml::Yue2Knobs& k) {
    if (!ev::isObject(v)) return;
    ev::Persistent o(v);
    getFloatOpt(o.get(), "energy", k.energy);
    getFloatOpt(o.get(), "texture", k.texture);
    getFloatOpt(o.get(), "percussion", k.percussion);
    for (const auto& a : brosoundml::Yue2Directions::builtin_ref().semantic) {
        float f = k.get(a.name);
        getFloatOpt(o.get(), a.name, f);
        k.set(a.name, f);
    }
}

Value makeKnobs(const brosoundml::Yue2Knobs& k) {
    ObjectBuilder o;
    o.set("energy", static_cast<double>(k.energy));
    o.set("texture", static_cast<double>(k.texture));
    o.set("percussion", static_cast<double>(k.percussion));
    for (const auto& a : brosoundml::Yue2Directions::builtin_ref().semantic)
        o.set(a.name, static_cast<double>(k.get(a.name)));
    return o.get();
}

// A song: Yue2Params' fields (readParams) + semantic (Int32Array of codec
// ids: render these instead of sampling) + tag.
bool readSong(Value v, brosoundml::Yue2Song& s, std::string& err) {
    if (!ev::isObject(v)) {
        err = "a song must be an object ({ style, lyrics, ... })";
        return false;
    }
    ev::Persistent o(v);
    if (!readParams(o.get(), s.params, err)) return false;
    ev::Persistent sem(ev::getProperty(o.get(), "semantic"));
    if (!ev::isUndefined(sem.get()) && !ev::isNull(sem.get())) {
        s.semantic = readInt32Array(sem.get());
        if (s.semantic.empty()) {
            err = "song.semantic must be a non-empty Int32Array";
            return false;
        }
    }
    getStrOpt(o.get(), "tag", s.tag);
    return true;
}

void readSeconds(Value o, std::string_view name, double& dst) {
    float f = static_cast<float>(dst);
    getFloatOpt(o, name, f);
    dst = f;
}

bool readStreamOptions(Value v, brosoundml::Yue2StreamOptions& o, std::string& err) {
    if (!ev::isObject(v)) return true;
    ev::Persistent r(v);
    ev::Persistent w(ev::getProperty(r.get(), "windows"));
    if (!ev::isUndefined(w.get()) && !ev::isNull(w.get())) {
        const std::vector<int32_t> ws = readInt32Array(w.get());
        std::vector<int> list;
        if (ws.empty() && ev::isObject(w.get())) {   // a plain array of numbers
            const int n = getPropertyInt(w.get(), "length", 0);
            for (int i = 0; i < n; ++i)
                list.push_back(static_cast<int>(ev::toDouble(ev::getElement(w.get(), static_cast<uint32_t>(i)))));
        } else {
            list.assign(ws.begin(), ws.end());
        }
        if (list.empty()) {
            err = "windows must list NAR window sizes in frames";
            return false;
        }
        o.windows = list;
    }
    getIntOpt(r.get(), "left", o.left);
    getIntOpt(r.get(), "right", o.right);
    getBoolOpt(r.get(), "paced", o.paced);
    getBoolOpt(r.get(), "radio", o.radio);
    readSeconds(r.get(), "emitLeadSeconds", o.emit_lead_seconds);
    readSeconds(r.get(), "decodeLeadSeconds", o.decode_lead_seconds);
    readSeconds(r.get(), "crossfadeSeconds", o.crossfade_seconds);
    readSeconds(r.get(), "skipFadeSeconds", o.skip_fade_seconds);
    readSeconds(r.get(), "interruptLatencySeconds", o.interrupt_latency_seconds);
    readSeconds(r.get(), "knobLatencySeconds", o.knob_latency_seconds);
    readSeconds(r.get(), "wantGraceSeconds", o.want_grace_seconds);
    getIntOpt(r.get(), "aheadSongs", o.ahead_songs);
    if (o.ahead_songs < 1 || o.ahead_songs > 8) {
        err = "aheadSongs must be 1..8";
        return false;
    }
    readSeconds(r.get(), "previewSeconds", o.preview_seconds);
    if (o.preview_seconds != 0 && !(o.preview_seconds >= 10 && o.preview_seconds <= 600)) {
        err = "previewSeconds must be 0 or 10..600";
        return false;
    }
    getBoolOpt(r.get(), "autoFull", o.auto_full);
    readSeconds(r.get(), "airMarginSeconds", o.air_margin_seconds);
    ev::Persistent k(ev::getProperty(r.get(), "knobs"));
    readKnobs(k.get(), o.knobs);
    o.knobs = brosoundml::Yue2Knobs::clamp(o.knobs);
    return true;
}

Value makeEvent(const brosoundml::Yue2StreamEvent& e) {
    ObjectBuilder o;
    o.set("kind", kindName(e.kind));
    o.set("song", static_cast<double>(e.song));
    o.set("tag", e.tag);
    o.set("sample", static_cast<double>(e.sample));
    o.set("frame0", static_cast<double>(e.frame0));
    o.set("frames", static_cast<double>(e.frames));
    o.set("seconds", e.seconds);
    o.set("value", e.value);
    o.set("text", e.text);
    return o.get();
}

Value makeStats(const brosoundml::Yue2StreamStats& s) {
    ObjectBuilder o;
    o.set("playhead", static_cast<double>(s.playhead));
    o.set("emitted", static_cast<double>(s.emitted));
    {
        ev::Persistent k(makeKnobs(s.knobs));
        o.set("knobs", k.get());
    }
    o.set("firstAudioSeconds", s.first_audio_seconds);
    o.set("interrupt1Seconds", s.interrupt1_seconds);
    o.set("interrupt2Seconds", s.interrupt2_seconds);
    ev::Persistent songs(hostArrayOf(s.songs.size(), [&](size_t i) -> Value {
        const auto& g = s.songs[i];
        ObjectBuilder so;
        so.set("index", static_cast<double>(g.index));
        so.set("tag", g.tag);
        so.set("planned", g.planned);
        so.set("semanticDone", g.semantic_done);
        so.set("semanticFrames", static_cast<double>(g.semantic_frames));
        so.set("latentFrames", static_cast<double>(g.latent_frames));
        so.set("decodedFrames", static_cast<double>(g.decoded_frames));
        so.set("start", static_cast<double>(g.start));
        so.set("end", static_cast<double>(g.end));
        so.set("latentNorm", g.latent_norm);
        so.set("built", g.built);
        so.set("ready", g.ready);
        so.set("onAir", g.on_air);
        so.set("hostBytes", static_cast<double>(g.host_bytes));
        so.set("preview", g.preview);
        so.set("previewFrames", static_cast<double>(g.preview_frames));
        so.set("estimateFrames", static_cast<double>(g.estimate_frames));
        so.set("ended", g.ended);
        ev::Persistent k(makeKnobs(g.knobs));
        so.set("knobs", k.get());
        return so.get();
    }));
    o.set("songs", songs.get());
    o.set("ahead", static_cast<double>(s.ahead));
    o.set("readyAhead", static_cast<double>(s.ready_ahead));
    o.set("hostBytes", static_cast<double>(s.host_bytes));
    return o.get();
}

void pollStream(StreamJob& j) {
    std::deque<Item> items;
    {
        std::lock_guard<std::mutex> lk(j.mu);
        items.swap(j.out);
    }
    for (Item& it : items) {
        if (auto* a = std::get_if<AudioItem>(&it)) {
            if (!ev::isFunction(j.onAudio.get())) continue;
            ev::Persistent v(makeStereo(a->audio));
            ev::setProperty(v.get(), "sample0", ev::fromDouble(static_cast<double>(a->sample0)));
            callCallback1(j.onAudio.get(), v.get());
        } else if (auto* e = std::get_if<brosoundml::Yue2StreamEvent>(&it)) {
            if (!ev::isFunction(j.onEvent.get())) continue;
            ev::Persistent v(makeEvent(*e));
            callCallback1(j.onEvent.get(), v.get());
        } else if (auto* l = std::get_if<LatItem>(&it)) {
            ObjectBuilder o;
            o.set("song", static_cast<double>(l->song));
            o.set("frame0", static_cast<double>(l->frame0));
            o.set("frames", static_cast<double>(l->frames));
            {
                ev::Persistent d(makeFloat32Array(l->data));
                o.set("latents", d.get());
            }
            ev::Persistent v(o.get());
            callCallback1(j.onLatents.get(), v.get());
        }
    }
}

Value streamControl(Value self, const char* fn, const std::function<Value(StreamJob&)>& f) {
    StreamJob* j = jobOf(self);
    if (!j || !j->stream) return ev::throwTypeError(std::string(fn) + ": not a YuE2 stream");
    return f(*j);
}

void decorateStream(ObjectBuilder& b) {
    b.def("enqueue", 1, [](Value self, std::span<const Value> a) -> Value {
        ev::Persistent arg(argAt(a, 0));
        return streamControl(self, "enqueue", [&](StreamJob& j) -> Value {
            brosoundml::Yue2Song s;
            std::string err;
            if (!readSong(arg.get(), s, err)) return ev::throwTypeError("enqueue(song): " + err);
            j.stream->enqueue(std::move(s));
            return ev::undefined();
        });
    });
    b.def("setPlayhead", 1, [](Value self, std::span<const Value> a) -> Value {
        const double n = numAt(a, 0, 0);
        return streamControl(self, "setPlayhead", [&](StreamJob& j) -> Value {
            j.stream->set_playhead(static_cast<std::int64_t>(std::max(0.0, n)));
            return ev::undefined();
        });
    });
    b.def("setKnobs", 1, [](Value self, std::span<const Value> a) -> Value {
        ev::Persistent arg(argAt(a, 0));
        return streamControl(self, "setKnobs", [&](StreamJob& j) -> Value {
            if (!ev::isObject(arg.get())) return ev::throwTypeError("setKnobs(knobs): an object required");
            readKnobs(arg.get(), j.knobs);   // fields left out keep their value
            j.knobs = brosoundml::Yue2Knobs::clamp(j.knobs);
            j.stream->set_knobs(j.knobs);
            return makeKnobs(j.knobs);
        });
    });
    b.def("interrupt", 1, [](Value self, std::span<const Value> a) -> Value {
        const double amt = hasArg(a, 0) ? numAt(a, 0, 1.0) : 1.0;
        return streamControl(self, "interrupt", [&](StreamJob& j) -> Value {
            j.stream->interrupt(static_cast<float>(std::clamp(amt, -1.0, 1.0)));
            return ev::undefined();
        });
    });
    b.def("skip", 0, [](Value self, std::span<const Value>) -> Value {
        return streamControl(self, "skip", [](StreamJob& j) -> Value {
            j.stream->skip();
            return ev::undefined();
        });
    });
    b.def("setNextPlan", 1, [](Value self, std::span<const Value> a) -> Value {
        if (!isStringArg(a, 0)) return ev::throwTypeError("setNextPlan(abc): an ABC string required");
        const std::string abc = strAt(a, 0);
        return streamControl(self, "setNextPlan", [&](StreamJob& j) -> Value {
            j.stream->set_next_plan(abc);
            return ev::undefined();
        });
    });
    b.def("setPlan", 2, [](Value self, std::span<const Value> a) -> Value {
        if (!isStringArg(a, 1)) return ev::throwTypeError("setPlan(song, abc): a song index and an ABC string required");
        const int song = static_cast<int>(numAt(a, 0, -1));
        const std::string abc = strAt(a, 1);
        return streamControl(self, "setPlan", [&](StreamJob& j) -> Value {
            j.stream->set_plan(song, abc);
            return ev::undefined();
        });
    });
    b.def("drop", 1, [](Value self, std::span<const Value> a) -> Value {
        if (!hasArg(a, 0)) return ev::throwTypeError("drop(song): a song index required");
        const int song = static_cast<int>(numAt(a, 0, -1));
        return streamControl(self, "drop", [&](StreamJob& j) -> Value {
            j.stream->drop(song);
            return ev::undefined();
        });
    });
    b.def("makeFull", 0, [](Value self, std::span<const Value> a) -> Value {
        const int song = hasArg(a, 0) ? static_cast<int>(numAt(a, 0, -1)) : -1;
        return streamControl(self, "makeFull", [&](StreamJob& j) -> Value {
            j.stream->make_full(song);
            return ev::undefined();
        });
    });
    b.def("setAutoFull", 1, [](Value self, std::span<const Value> a) -> Value {
        const bool on = hasArg(a, 0) && ev::toBool(a[0]);
        return streamControl(self, "setAutoFull", [&](StreamJob& j) -> Value {
            j.stream->set_auto_full(on);
            return ev::undefined();
        });
    });
    b.def("stop", 0, [](Value self, std::span<const Value>) -> Value {
        return streamControl(self, "stop", [](StreamJob& j) -> Value {
            j.stream->stop();
            return ev::undefined();
        });
    });
    b.accessor("stats", [](Value self, std::span<const Value>) -> Value {
        return streamControl(self, "stats", [](StreamJob& j) -> Value { return makeStats(j.stream->stats()); });
    });
    b.accessor("knobs", [](Value self, std::span<const Value>) -> Value {
        return streamControl(self, "knobs", [](StreamJob& j) -> Value { return makeKnobs(j.knobs); });
    });
    b.accessor("running", [](Value self, std::span<const Value>) -> Value {
        StreamJob* j = jobOf(self);
        return ev::fromBool(j && j->running.load());
    });
    // The clamps (the research's break points stay outside them).
    b.accessor("knobRanges", [](Value, std::span<const Value>) -> Value { return knobRanges(); });
}

}  // namespace

Value knobRanges() {
    brosoundml::Yue2Knobs lo, hi;
    lo.energy = lo.texture = lo.percussion = -1e9f;
    hi.energy = hi.texture = hi.percussion = 1e9f;
    const auto& axes = brosoundml::Yue2Directions::builtin_ref().semantic;
    for (const auto& a : axes) {
        lo.set(a.name, -1e9f);
        hi.set(a.name, 1e9f);
    }
    lo = brosoundml::Yue2Knobs::clamp(lo);
    hi = brosoundml::Yue2Knobs::clamp(hi);
    ObjectBuilder o;
    auto range = [&](const std::string& name, float l, float h) {
        ObjectBuilder r;
        r.set("min", static_cast<double>(l));
        r.set("max", static_cast<double>(h));
        o.set(name, r.get());
    };
    range("energy", lo.energy, hi.energy);
    range("texture", lo.texture, hi.texture);
    range("percussion", lo.percussion, hi.percussion);
    for (const auto& a : axes) range(a.name, lo.get(a.name), hi.get(a.name));
    return o.get();
}

void installStreamClass() { g_streamClass.install("YuE2Stream", 0, nullptr, decorateStream); }

// model.stream({ vae, songs?, onAudio, onEvent?, onLatents?, onDone?, ...options })
Value yue2Stream(Value self, std::span<const Value> a) {
    ev::Persistent selfRoot(self);
    HostYue2* w = yue2Of(selfRoot.get());
    if (!w) return ev::throwTypeError("stream: not a YuE2 model");
    if (!w->model || !w->model->loaded()) return ev::throwError("stream: model is not loaded");
    if (!isObjectArg(a, 0)) return ev::throwTypeError("stream(opts): opts with a vae required");
    ev::Persistent opts(a[0]);
    ev::Persistent vaeVal(ev::getProperty(opts.get(), "vae"));
    HostYue2Vae* vw = vaeOf(vaeVal.get());
    if (!vw || !vw->vae || !vw->vae->loaded()) return ev::throwTypeError("stream(opts): opts.vae must be a loaded YuE2 VAE");

    auto job = std::make_shared<StreamJob>();
    brosoundml::Yue2StreamOptions o;
    std::string err;
    if (!readStreamOptions(opts.get(), o, err)) return ev::throwTypeError("stream(opts): " + err);
    std::vector<brosoundml::Yue2Song> songs;
    {
        ev::Persistent list(ev::getProperty(opts.get(), "songs"));
        if (ev::isObject(list.get())) {
            const int n = getPropertyInt(list.get(), "length", 0);
            for (int i = 0; i < n; ++i) {
                ev::Persistent e(ev::getElement(list.get(), static_cast<uint32_t>(i)));
                brosoundml::Yue2Song s;
                if (!readSong(e.get(), s, err))
                    return ev::throwTypeError("stream(opts): songs[" + std::to_string(i) + "]: " + err);
                songs.push_back(std::move(s));
            }
        }
    }
    job->onAudio = getFunctionOpt(opts.get(), "onAudio");
    job->onEvent = getFunctionOpt(opts.get(), "onEvent");
    job->onLatents = getFunctionOpt(opts.get(), "onLatents");
    job->onDone = getFunctionOpt(opts.get(), "onDone");
    job->wantLat = ev::isFunction(job->onLatents.get());
    job->knobs = o.knobs;

    StreamJob* jp = job.get();
    if (job->wantLat)
        o.on_latents = [jp](int song, int f0, int n, const float* d) {
            LatItem l{song, f0, n, std::vector<float>(d, d + static_cast<std::size_t>(n) * 64)};
            std::lock_guard<std::mutex> lk(jp->mu);
            jp->out.emplace_back(std::move(l));
        };
    o.cancel = [jp] {
        const std::atomic<bool>* c = jp->cancel.load();
        return c && c->load();
    };
    auto onAudio = [jp](const brosoundml::StereoAudio& c, std::int64_t s0) {
        std::lock_guard<std::mutex> lk(jp->mu);
        jp->out.emplace_back(AudioItem{c, s0});
    };
    auto onEvent = [jp](const brosoundml::Yue2StreamEvent& e) {
        std::lock_guard<std::mutex> lk(jp->mu);
        jp->out.emplace_back(e);
    };
    try {
        job->stream = std::make_unique<brosoundml::Yue2Stream>(*w->model, *vw->vae, std::move(o), onAudio, onEvent);
    } catch (const std::exception& e) {
        return ev::throwTypeError(std::string("stream(opts): ") + e.what());
    }
    for (auto& s : songs) job->stream->enqueue(std::move(s));

    if (!w->busy.tryClaim()) return ev::throwError("stream: an operation is already in flight on this model");
    if (!vw->busy.tryClaim()) {
        w->busy.release();
        return ev::throwError("stream: an operation is already in flight on the VAE");
    }
    job->model = w->model;
    job->vae = vw->vae;
    job->gate = w->busy;
    job->vaeGate = vw->busy;
    job->modelRef = ev::Persistent(selfRoot.get());
    job->vaeRef = ev::Persistent(vaeVal.get());
    const brotensor::Device dev = w->device;

    auto host = std::make_unique<HostStream>();
    host->job = job;
    ev::Persistent obj(g_streamClass.createInstance(std::move(host)));
    job->selfRef = ev::Persistent(obj.get());

    auto work = [job, dev](const std::atomic<bool>& cancel) {
        job->cancel.store(&cancel);
        brotensor::DeviceScope scope(dev);
        job->stream->run();
    };
    auto poll = [job] { pollStream(*job); };
    auto done = [job](bool cancelled, const std::string& error) {
        job->running.store(false);
        job->gate.release();
        job->vaeGate.release();
        pollStream(*job);
        ev::Persistent self(job->selfRef.get());
        job->selfRef = ev::Persistent();
        if (!ev::isFunction(job->onDone.get())) return;
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback1(job->onDone.get(), info.get());
    };
    (void)launchAsyncJob(std::move(work), std::move(poll), std::move(done));
    return obj.get();
}

}  // namespace brosoundml::api::music
