// model.endless(opts) -> YuE2Endless: the endless stream of 10 s chunks over
// brosoundml::Yue2Endless (yue2_endless.h), steered by targets, paths and
// branches.
//
// The stream's worker runs on a background thread for its whole life
// (launchAsyncJob) and claims the model's and the VAE's gates (and the
// Whisper model's, when one listens for rolled lyrics) until it ends. Audio
// chunks and events cross to the JS thread through a mutex-guarded queue
// that each tick's poll drains; controls are Yue2Endless's own thread-safe
// calls. While it runs the job roots the stream object.

#include "soundml_music_endless.h"
#include "soundml_async.h"
#include "soundml_stt_internal.h"

#include <atomic>
#include <deque>
#include <mutex>
#include <variant>

namespace brosoundml::api::music {

namespace {

HostClass g_endlessClass;

struct AudioItem {
    brosoundml::StereoAudio audio;
    std::int64_t sample0 = 0;
};
using Item = std::variant<AudioItem, brosoundml::Yue2EndlessEvent>;

struct EndlessJob {
    std::shared_ptr<brosoundml::Yue2> model;
    std::shared_ptr<brosoundml::Yue2Vae> vae;
    std::shared_ptr<brosoundml::Whisper> whisper;
    ModelGate gate, vaeGate, whisperGate;
    bool hasWhisper = false;
    std::unique_ptr<brosoundml::Yue2Endless> stream;
    brosoundml::Yue2EndlessTarget target;   // JS side: what setTarget merges into
    std::atomic<const std::atomic<bool>*> cancel{nullptr};
    std::atomic<bool> running{true};
    std::mutex mu;
    std::deque<Item> out;
    ev::Persistent onAudio, onEvent, onDone, modelRef, vaeRef, whisperRef, tokRef, selfRef;
};

struct HostEndless {
    std::shared_ptr<EndlessJob> job;
    ~HostEndless() {
        if (job && job->stream && job->running.load()) job->stream->stop();
    }
};

EndlessJob* jobOf(Value v) {
    auto* h = g_endlessClass.isInstance(v) ? static_cast<HostEndless*>(g_endlessClass.unwrap(v)) : nullptr;
    return h ? h->job.get() : nullptr;
}

void pollEndless(EndlessJob& j) {
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
        } else if (auto* e = std::get_if<brosoundml::Yue2EndlessEvent>(&it)) {
            if (!ev::isFunction(j.onEvent.get())) continue;
            ev::Persistent v(makeEndlessEvent(*e));
            callCallback1(j.onEvent.get(), v.get());
        }
    }
}

Value control(Value self, const char* fn, const std::function<Value(EndlessJob&)>& f) {
    EndlessJob* j = jobOf(self);
    if (!j || !j->stream) return ev::throwTypeError(std::string(fn) + ": not a YuE2 endless stream");
    return f(*j);
}

void decorateEndless(ObjectBuilder& b) {
    b.def("setPlayhead", 1, [](Value self, std::span<const Value> a) -> Value {
        const double n = numAt(a, 0, 0);
        return control(self, "setPlayhead", [&](EndlessJob& j) -> Value {
            j.stream->set_playhead(static_cast<std::int64_t>(std::max(0.0, n)));
            return ev::undefined();
        });
    });
    // setTarget(partial): merged into the current target; at (program
    // seconds) puts it on the path instead of now. Returns the merged target.
    b.def("setTarget", 1, [](Value self, std::span<const Value> a) -> Value {
        ev::Persistent arg(argAt(a, 0));
        return control(self, "setTarget", [&](EndlessJob& j) -> Value {
            brosoundml::Yue2EndlessTarget t = j.target;
            std::string err;
            if (!readTarget(arg.get(), t, err)) return ev::throwTypeError("setTarget(target): " + err);
            if (t.at < 0) j.target = t;
            j.stream->set_target(t);
            return makeTarget(t);
        });
    });
    b.def("setPath", 1, [](Value self, std::span<const Value> a) -> Value {
        ev::Persistent arg(argAt(a, 0));
        return control(self, "setPath", [&](EndlessJob& j) -> Value {
            std::vector<brosoundml::Yue2EndlessTarget> path;
            std::string err;
            if (!readPath(arg.get(), j.target, path, err)) return ev::throwTypeError("setPath(path): " + err);
            j.stream->set_path(std::move(path));
            return ev::undefined();
        });
    });
    b.def("setKnobs", 1, [](Value self, std::span<const Value> a) -> Value {
        ev::Persistent arg(argAt(a, 0));
        return control(self, "setKnobs", [&](EndlessJob& j) -> Value {
            if (!ev::isObject(arg.get())) return ev::throwTypeError("setKnobs(knobs): an object required");
            readKnobs(arg.get(), j.target.knobs);
            j.target.knobs = brosoundml::Yue2Knobs::clamp(j.target.knobs);
            j.stream->set_knobs(j.target.knobs);
            return makeKnobs(j.target.knobs);
        });
    });
    // setScore(abc | null): the held score of the target now.
    b.def("setScore", 1, [](Value self, std::span<const Value> a) -> Value {
        const bool none = !hasArg(a, 0) || ev::isNull(a[0]) || ev::isUndefined(a[0]);
        if (!none && !isStringArg(a, 0)) return ev::throwTypeError("setScore(abc | null): an ABC string or null");
        const std::string abc = none ? std::string() : strAt(a, 0);
        return control(self, "setScore", [&](EndlessJob& j) -> Value {
            j.target.score = abc;
            j.target.at = -1;
            j.stream->set_target(j.target);
            return ev::undefined();
        });
    });
    b.def("branch", 2, [](Value self, std::span<const Value> a) -> Value {
        ev::Persistent arg(argAt(a, 0));
        ev::Persistent pathArg(argAt(a, 1));
        return control(self, "branch", [&](EndlessJob& j) -> Value {
            brosoundml::Yue2EndlessTarget t = j.target;
            std::string err;
            if (!readTarget(arg.get(), t, err)) return ev::throwTypeError("branch(target, path?): " + err);
            t.at = -1;
            std::vector<brosoundml::Yue2EndlessTarget> path;
            if (!ev::isUndefined(pathArg.get()) && !ev::isNull(pathArg.get()) && !readPath(pathArg.get(), t, path, err))
                return ev::throwTypeError("branch(target, path?): " + err);
            return ev::fromDouble(j.stream->branch(t, std::move(path)));
        });
    });
    b.def("commitBranch", 1, [](Value self, std::span<const Value> a) -> Value {
        const int id = static_cast<int>(numAt(a, 0, -1));
        return control(self, "commitBranch", [&](EndlessJob& j) -> Value {
            j.stream->commit_branch(id);
            return ev::undefined();
        });
    });
    b.def("discardBranch", 1, [](Value self, std::span<const Value> a) -> Value {
        const int id = static_cast<int>(numAt(a, 0, -1));
        return control(self, "discardBranch", [&](EndlessJob& j) -> Value {
            j.stream->discard_branch(id);
            return ev::undefined();
        });
    });
    b.def("stop", 0, [](Value self, std::span<const Value>) -> Value {
        return control(self, "stop", [](EndlessJob& j) -> Value {
            j.stream->stop();
            return ev::undefined();
        });
    });
    b.accessor("stats", [](Value self, std::span<const Value>) -> Value {
        return control(self, "stats", [](EndlessJob& j) -> Value { return makeEndlessStats(j.stream->stats()); });
    });
    b.accessor("target", [](Value self, std::span<const Value>) -> Value {
        return control(self, "target", [](EndlessJob& j) -> Value { return makeTarget(j.target); });
    });
    b.accessor("running", [](Value self, std::span<const Value>) -> Value {
        EndlessJob* j = jobOf(self);
        return ev::fromBool(j && j->running.load());
    });
    b.accessor("knobRanges", [](Value, std::span<const Value>) -> Value { return knobRanges(); });
}

}  // namespace

void installEndlessClass() { g_endlessClass.install("YuE2Endless", 0, nullptr, decorateEndless); }

// model.endless({ vae, target, path?, whisper?, whisperTokenizer?, onAudio, onEvent?, onDone?, ...options })
Value yue2Endless(Value self, std::span<const Value> a) {
    ev::Persistent selfRoot(self);
    HostYue2* w = yue2Of(selfRoot.get());
    if (!w) return ev::throwTypeError("endless: not a YuE2 model");
    if (!w->model || !w->model->loaded()) return ev::throwError("endless: model is not loaded");
    if (!isObjectArg(a, 0)) return ev::throwTypeError("endless(opts): opts with a vae required");
    ev::Persistent opts(a[0]);
    ev::Persistent vaeVal(ev::getProperty(opts.get(), "vae"));
    HostYue2Vae* vw = vaeOf(vaeVal.get());
    if (!vw || !vw->vae || !vw->vae->loaded()) return ev::throwTypeError("endless(opts): opts.vae must be a loaded YuE2 VAE");

    auto job = std::make_shared<EndlessJob>();
    brosoundml::Yue2EndlessOptions o;
    std::string err;
    if (!readEndlessOptions(opts.get(), o, err)) return ev::throwTypeError("endless(opts): " + err);

    // The listener for rolled lyrics: a Whisper model and its tokenizer.
    ev::Persistent whVal(ev::getProperty(opts.get(), "whisper"));
    ev::Persistent tokVal(ev::getProperty(opts.get(), "whisperTokenizer"));
    HostWhisperModel* wm = nullptr;
    HostWhisperTokenizer* wt = nullptr;
    if (!ev::isUndefined(whVal.get()) && !ev::isNull(whVal.get())) {
        wm = unwrapAs<HostWhisperModel>(g_whisperModelClass, whVal.get());
        wt = unwrapAs<HostWhisperTokenizer>(g_whisperTokenizerClass, tokVal.get());
        if (!wm || !wm->model || !wm->model->loaded())
            return ev::throwTypeError("endless(opts): opts.whisper must be a loaded Whisper model (bro.stt.loadWhisper)");
        if (!wt || !wt->tok)
            return ev::throwTypeError("endless(opts): opts.whisper needs opts.whisperTokenizer (bro.stt.loadTokenizer)");
        const brolm::whisper::Tokenizer* tok = wt->tok.get();
        o.listen = brosoundml::Yue2Endless::whisper_listener(
            *wm->model, tok->build_prompt("en", "transcribe", false),
            [tok](const std::vector<int32_t>& ids) { return tok->decode(ids, true); });
    }
    job->onAudio = getFunctionOpt(opts.get(), "onAudio");
    job->onEvent = getFunctionOpt(opts.get(), "onEvent");
    job->onDone = getFunctionOpt(opts.get(), "onDone");
    job->target = o.target;

    EndlessJob* jp = job.get();
    o.cancel = [jp] {
        const std::atomic<bool>* c = jp->cancel.load();
        return c && c->load();
    };
    auto onAudio = [jp](const brosoundml::StereoAudio& c, std::int64_t s0) {
        std::lock_guard<std::mutex> lk(jp->mu);
        jp->out.emplace_back(AudioItem{c, s0});
    };
    auto onEvent = [jp](const brosoundml::Yue2EndlessEvent& e) {
        std::lock_guard<std::mutex> lk(jp->mu);
        jp->out.emplace_back(e);
    };
    try {
        job->stream = std::make_unique<brosoundml::Yue2Endless>(*w->model, *vw->vae, std::move(o), onAudio, onEvent);
    } catch (const std::exception& e) {
        return ev::throwTypeError(std::string("endless(opts): ") + e.what());
    }

    if (!w->busy.tryClaim()) return ev::throwError("endless: an operation is already in flight on this model");
    if (!vw->busy.tryClaim()) {
        w->busy.release();
        return ev::throwError("endless: an operation is already in flight on the VAE");
    }
    if (wm && !wm->busy.tryClaim()) {
        w->busy.release();
        vw->busy.release();
        return ev::throwError("endless: an operation is already in flight on the Whisper model");
    }
    job->model = w->model;
    job->vae = vw->vae;
    job->gate = w->busy;
    job->vaeGate = vw->busy;
    job->modelRef = ev::Persistent(selfRoot.get());
    job->vaeRef = ev::Persistent(vaeVal.get());
    if (wm) {
        job->whisper = wm->model;
        job->whisperGate = wm->busy;
        job->hasWhisper = true;
        job->whisperRef = ev::Persistent(whVal.get());
        job->tokRef = ev::Persistent(tokVal.get());
    }
    const brotensor::Device dev = w->device;

    auto host = std::make_unique<HostEndless>();
    host->job = job;
    ev::Persistent obj(g_endlessClass.createInstance(std::move(host)));
    job->selfRef = ev::Persistent(obj.get());

    auto work = [job, dev](const std::atomic<bool>& cancel) {
        job->cancel.store(&cancel);
        brotensor::DeviceScope scope(dev);
        job->stream->run();
    };
    auto poll = [job] { pollEndless(*job); };
    auto done = [job](bool cancelled, const std::string& error) {
        job->running.store(false);
        job->gate.release();
        job->vaeGate.release();
        if (job->hasWhisper) job->whisperGate.release();
        pollEndless(*job);
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
