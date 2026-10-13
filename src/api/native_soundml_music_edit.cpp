// bro.music editing: model.splice(plan, opts) and model.inpaint(plan,
// semantic, latents, a, b, opts), Yue2::splice / Yue2::inpaint (yue2.h) as
// async jobs. A take's tokens are cut and joined with splice, the frames that
// changed re-synthesized with inpaint, and vae.decode renders the result.

#include "soundml_music_internal.h"
#include "soundml_async.h"

#include <atomic>

namespace brosoundml::api::music {

namespace {

const char* stageName(brosoundml::Yue2Stage s) {
    return s == brosoundml::Yue2Stage::Synthesize ? "synthesize" : s == brosoundml::Yue2Stage::Plan ? "plan" : "semantic";
}

struct EditJob {
    std::shared_ptr<brosoundml::Yue2> model;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate gate;
    brosoundml::Yue2Params params;
    brosoundml::Yue2Plan plan;
    // splice
    brosoundml::Yue2Splice splice;
    brosoundml::Yue2SpliceResult spliced;
    // inpaint
    std::vector<int32_t> semantic;
    std::vector<float> latents;
    int a = 0, b = 0;
    bool cancelled = false;
    std::atomic<int> done{0}, total{0}, stage{0};
    int seen = -1;
    ev::Persistent self, onProgress, onDone;
};

// Common start: the model, the request params, the plan; claims the model.
// False = throw `err` (a TypeError when `typeErr`).
bool startEdit(Value self, Value planArg, Value opts, const char* fn, const std::shared_ptr<EditJob>& job,
               std::string& err, bool& typeErr) {
    typeErr = true;
    auto* w = yue2Of(self);
    if (!w) { err = std::string(fn) + ": not a YuE2 model"; return false; }
    typeErr = false;
    if (!w->model || !w->model->loaded()) { err = std::string(fn) + ": model is not loaded"; return false; }
    typeErr = true;
    if (!readPlan(planArg, job->plan, err) || !readParams(opts, job->params, err)) {
        err = std::string(fn) + ": " + err;
        return false;
    }
    job->onProgress = getFunctionOpt(opts, "onProgress");
    job->onDone = getFunctionOpt(opts, "onDone");
    typeErr = false;
    if (!w->busy.tryClaim()) { err = std::string(fn) + ": an operation is already in flight on this model"; return false; }
    job->model = w->model;
    job->device = w->device;
    job->gate = w->busy;
    job->self = ev::Persistent(self);
    return true;
}

Value throwStart(const std::string& err, bool typeErr) {
    return typeErr ? ev::throwTypeError(err) : ev::throwError(err);
}

brosoundml::Yue2Callbacks progressCallbacks(EditJob& j) {
    brosoundml::Yue2Callbacks cb;
    cb.on_progress = [&j](brosoundml::Yue2Stage s, int d, int t) {
        j.stage.store(static_cast<int>(s));
        j.total.store(t);
        j.done.store(d);
    };
    return cb;
}

void pollProgress(EditJob& j) {
    if (!ev::isFunction(j.onProgress.get())) return;
    const int d = j.done.load();
    if (d == j.seen) return;
    j.seen = d;
    ObjectBuilder o;
    o.set("stage", stageName(static_cast<brosoundml::Yue2Stage>(j.stage.load())));
    o.set("done", ev::fromDouble(d));
    o.set("total", ev::fromDouble(j.total.load()));
    ev::Persistent ob(o.get());
    callCallback1(j.onProgress.get(), ob.get());
}

Value makeSpliceResult(const brosoundml::Yue2SpliceResult& r) {
    ObjectBuilder o;
    {
        ev::Persistent s(makeInt32Array(r.semantic));
        o.set("semantic", s.get());
    }
    o.set("frames", ev::fromDouble(static_cast<double>(r.semantic.size())));
    o.set("middle0", ev::fromDouble(r.middle0));
    o.set("after0", ev::fromDouble(r.after0));
    o.set("chosen", ev::fromDouble(r.chosen));
    {
        ev::Persistent s(makeFloat32Array(r.scores));
        o.set("scores", s.get());
    }
    {
        ev::Persistent m(hostArrayOf(r.middles.size(), [&r](size_t i) { return makeInt32Array(r.middles[i]); }));
        o.set("middles", m.get());
    }
    o.set("truncated", ev::fromBool(r.truncated));
    return o.get();
}

}  // namespace

// model.splice(plan, opts) -> AsyncHandle. opts: the request params (style,
// lyrics, cot, seed, cfgScale, semanticSampling, steer: the plan's own), plus
// before / after (Int32Array codec ids), middleFrames, candidates,
// scoreFrames, continue, onProgress, onDone(result, info).
Value yue2Splice(Value self, std::span<const Value> a) {
    ev::Persistent selfRoot(self);
    ev::Persistent planArg(argAt(a, 0));
    ev::Persistent opts(isObjectArg(a, 1) ? a[1] : ev::undefined());
    if (!ev::isObject(opts.get())) return ev::throwTypeError("splice(plan, opts): opts required");
    auto job = std::make_shared<EditJob>();
    brosoundml::Yue2Splice& s = job->splice;
    s.before = readInt32Array(ev::getProperty(opts.get(), "before"));
    s.after = readInt32Array(ev::getProperty(opts.get(), "after"));
    getIntOpt(opts.get(), "middleFrames", s.middle_frames);
    getIntOpt(opts.get(), "candidates", s.candidates);
    getIntOpt(opts.get(), "scoreFrames", s.score_frames);
    getBoolOpt(opts.get(), "continue", s.continue_after);
    if (s.middle_frames < 0) return ev::throwTypeError("splice: middleFrames must be >= 0");
    if (s.candidates < 1 || s.candidates > 32) return ev::throwTypeError("splice: candidates must be 1..32");
    if (s.before.empty() && s.after.empty() && s.middle_frames == 0 && !s.continue_after)
        return ev::throwTypeError("splice: nothing to make (before, after, middleFrames and continue are all empty)");
    std::string err;
    bool typeErr = false;
    if (!startEdit(selfRoot.get(), planArg.get(), opts.get(), "splice", job, err, typeErr)) return throwStart(err, typeErr);
    auto work = [job](const std::atomic<bool>& cancel) {
        brotensor::DeviceScope scope(job->device);
        job->spliced = job->model->splice(job->params, job->plan, job->splice, progressCallbacks(*job), cancelCheckOf(cancel));
        job->cancelled = job->spliced.cancelled;
    };
    auto poll = [job] { pollProgress(*job); };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        if (!ev::isFunction(job->onDone.get())) return;
        const bool c = cancelled || job->cancelled;
        ev::Persistent result(error.empty() && !c ? makeSpliceResult(job->spliced) : ev::undefined());
        ev::Persistent info(makeDoneInfo(c, error));
        callCallback2(job->onDone.get(), result.get(), info.get());
    };
    return launchAsyncJob(std::move(work), std::move(poll), std::move(done));
}

// model.inpaint(plan, semantic, latents, a, b, opts) -> AsyncHandle;
// onDone({ latents, frames }, info): `latents` with frames [a, b) re-solved.
Value yue2Inpaint(Value self, std::span<const Value> a) {
    ev::Persistent selfRoot(self);
    ev::Persistent planArg(argAt(a, 0));
    ev::Persistent opts(isObjectArg(a, 5) ? a[5] : ev::undefined());
    auto job = std::make_shared<EditJob>();
    job->semantic = hasArg(a, 1) ? readInt32Array(a[1]) : std::vector<int32_t>{};
    job->latents = hasArg(a, 2) ? readFloat32Array(a[2]) : std::vector<float>{};
    const int T = static_cast<int>(job->semantic.size());
    if (T == 0) return ev::throwTypeError("inpaint(plan, semantic, latents, a, b, opts): semantic must be a non-empty Int32Array");
    if (job->latents.size() != static_cast<size_t>(T) * brosoundml::Yue2::kLatentDim)
        return ev::throwTypeError("inpaint: latents.length must be semantic.length * 64");
    if (!hasArg(a, 3) || !hasArg(a, 4) || !ev::isNumber(a[3]) || !ev::isNumber(a[4]))
        return ev::throwTypeError("inpaint: a and b (frames) required");
    job->a = static_cast<int>(ev::toDouble(a[3]));
    job->b = static_cast<int>(ev::toDouble(a[4]));
    if (job->a < 0 || job->b > T || job->a >= job->b)
        return ev::throwTypeError("inpaint: [a, b) must be a non-empty span inside the song");
    std::string err;
    bool typeErr = false;
    if (!startEdit(selfRoot.get(), planArg.get(), opts.get(), "inpaint", job, err, typeErr)) return throwStart(err, typeErr);
    auto work = [job](const std::atomic<bool>& cancel) {
        brotensor::DeviceScope scope(job->device);
        job->latents = job->model->inpaint(job->params, job->plan, job->semantic, std::move(job->latents), job->a,
                                           job->b, progressCallbacks(*job), cancelCheckOf(cancel));
        job->cancelled = job->latents.empty();
    };
    auto poll = [job] { pollProgress(*job); };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        if (!ev::isFunction(job->onDone.get())) return;
        const bool c = cancelled || job->cancelled;
        Value result = ev::undefined();
        ev::Persistent res;
        if (error.empty() && !c) {
            ObjectBuilder o;
            {
                ev::Persistent l(makeFloat32Array(job->latents));
                o.set("latents", l.get());
            }
            o.set("frames", ev::fromDouble(static_cast<double>(job->semantic.size())));
            res = ev::Persistent(o.get());
            result = res.get();
        }
        ev::Persistent rr(result);
        ev::Persistent info(makeDoneInfo(c, error));
        callCallback2(job->onDone.get(), rr.get(), info.get());
    };
    return launchAsyncJob(std::move(work), std::move(poll), std::move(done));
}

}  // namespace brosoundml::api::music
