// bro.music — YuE2-3B song generation (lyrics + style -> ABC plan -> semantic
// codec tokens -> NAR flow-matching latents) and the YuE2 VAE (latents <->
// 48 kHz stereo). A generate job takes an optional VAE and chains the
// backbone's final latent frames into a Yue2VaeStream, so audio reaches
// onAudio while later frames are still being synthesized.
//
// Every op on a model is single-owner (ModelGate); a generate with a VAE
// claims both gates. Loading is GPU by default (opts.device overrides).
// Streaming callbacks are drained on the JS thread each tick (poll); the
// work thread publishes through fixed-capacity SPSC slots.
#include "soundml_loader.h"
#include "soundml_music_internal.h"

#include <brosoundml/yue2.h>
#include <brosoundml/yue2_stream.h>
#include <brosoundml/yue2_vae.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace brosoundml::api {

namespace {
HostClass g_yue2Class;
HostClass g_yue2VaeClass;
}  // namespace

namespace music {

HostYue2* yue2Of(Value v) {
    return g_yue2Class.isInstance(v) ? static_cast<HostYue2*>(g_yue2Class.unwrap(v)) : nullptr;
}
HostYue2Vae* vaeOf(Value v) {
    return g_yue2VaeClass.isInstance(v) ? static_cast<HostYue2Vae*>(g_yue2VaeClass.unwrap(v)) : nullptr;
}

const char* stageName(brosoundml::Yue2Stage s) {
    switch (s) {
        case brosoundml::Yue2Stage::Plan: return "plan";
        case brosoundml::Yue2Stage::Semantic: return "semantic";
        case brosoundml::Yue2Stage::Synthesize: return "synthesize";
    }
    return "plan";
}

// ── option parsing ───────────────────────────────────────────────────────────

void readSeed(Value opts, std::uint64_t& dst) {
    Value v = ev::getProperty(opts, "seed");
    if (ev::isNumber(v)) dst = static_cast<std::uint64_t>(static_cast<int64_t>(ev::toDouble(v)));
    else if (ev::isBigInt(v)) dst = ev::toUint64(v);
}

void readSampling(Value obj, brosoundml::Yue2Sampling& s) {
    if (!ev::isObject(obj)) return;
    ev::Persistent o(obj);
    getFloatOpt(o.get(), "temperature", s.temperature);
    getFloatOpt(o.get(), "topP", s.top_p);
    getIntOpt(o.get(), "topK", s.top_k);
    getFloatOpt(o.get(), "repetitionPenalty", s.repetition_penalty);
    getIntOpt(o.get(), "penaltyWindow", s.penalty_window);
    getIntOpt(o.get(), "minTokens", s.min_tokens);
    getIntOpt(o.get(), "maxTokens", s.max_tokens);
    getIntOpt(o.get(), "endBiasAfter", s.end_bias_after);
    getFloatOpt(o.get(), "endBiasPerToken", s.end_bias_per_token);
}

std::vector<int> readIntList(Value v) {
    if (ev::isUndefined(v) || ev::isNull(v)) return {};
    if (ev::isNumber(v)) return {static_cast<int>(ev::toDouble(v))};
    const std::vector<int32_t> raw = readInt32Array(v);
    return std::vector<int>(raw.begin(), raw.end());
}

// Params common to every entry point; false + err on a shape error.
bool readParams(Value opts, brosoundml::Yue2Params& p, std::string& err) {
    if (!ev::isObject(opts)) return true;
    ev::Persistent root(opts);
    getStrOpt(root.get(), "style", p.style);
    getStrOpt(root.get(), "lyrics", p.lyrics);
    getStrOpt(root.get(), "abc", p.abc);
    {
        std::string cot;
        getStrOpt(root.get(), "cot", cot);
        if (cot == "full") p.cot = brosoundml::Yue2Cot::Full;
        else if (cot == "melody") p.cot = brosoundml::Yue2Cot::Melody;
        else if (cot == "off") p.cot = brosoundml::Yue2Cot::Off;
        else if (!cot.empty()) { err = "cot must be 'full', 'melody' or 'off'"; return false; }
    }
    readSeed(root.get(), p.seed);
    getFloatOpt(root.get(), "cfgScale", p.cfg_scale);
    {
        ev::Persistent s(ev::getProperty(root.get(), "abcSampling"));
        readSampling(s.get(), p.abc_sampling);
    }
    {
        ev::Persistent s(ev::getProperty(root.get(), "semanticSampling"));
        readSampling(s.get(), p.semantic_sampling);
    }
    getIntOpt(root.get(), "odeSteps", p.ode_steps);
    getIntOpt(root.get(), "narChunkFrames", p.nar_chunk_frames);
    getBoolOpt(root.get(), "runSemantic", p.run_semantic);
    getBoolOpt(root.get(), "runSynthesize", p.run_synthesize);
    getBoolOpt(root.get(), "reusePrefixKv", p.reuse_prefix_kv);
    getIntOpt(root.get(), "abcDraftTokens", p.abc_draft_tokens);
    if (p.ode_steps < 1) { err = "odeSteps must be >= 1"; return false; }
    if (p.abc_draft_tokens < 0 || p.abc_draft_tokens > 7) { err = "abcDraftTokens must be 0..7"; return false; }

    // steer: [{ layer, nar?, delta: Float32Array, scale?, perPosition?, pos0?, plan?, semantic? }]
    {
        ev::Persistent st(ev::getProperty(root.get(), "steer"));
        if (ev::isObject(st.get())) {
            const int n = getPropertyInt(st.get(), "length", 0);
            for (int i = 0; i < n; ++i) {
                ev::Persistent e(ev::getElement(st.get(), static_cast<uint32_t>(i)));
                if (!ev::isObject(e.get())) { err = "steer[" + std::to_string(i) + "] must be an object"; return false; }
                brosoundml::Yue2Steer s;
                getIntOpt(e.get(), "layer", s.layer);
                getBoolOpt(e.get(), "nar", s.nar);
                getFloatOpt(e.get(), "scale", s.scale);
                getBoolOpt(e.get(), "perPosition", s.per_position);
                getIntOpt(e.get(), "pos0", s.pos0);
                getBoolOpt(e.get(), "plan", s.plan);
                getBoolOpt(e.get(), "semantic", s.semantic);
                s.delta = readFloat32Array(ev::getProperty(e.get(), "delta"));
                if (s.delta.empty()) { err = "steer[" + std::to_string(i) + "].delta must be a non-empty Float32Array"; return false; }
                p.steer.push_back(std::move(s));
            }
        }
    }
    // capture: { layers: number[], plan?, semantic?, narEvals?: number[] }
    {
        ev::Persistent cap(ev::getProperty(root.get(), "capture"));
        if (ev::isObject(cap.get())) {
            p.capture.layers = readIntList(ev::getProperty(cap.get(), "layers"));
            getBoolOpt(cap.get(), "plan", p.capture.plan);
            getBoolOpt(cap.get(), "semantic", p.capture.semantic);
            p.capture.nar_evals = readIntList(ev::getProperty(cap.get(), "narEvals"));
        }
    }
    return true;
}

// { abc, abcIds, prefix, truncated } <-> Yue2Plan.
bool readPlan(Value v, brosoundml::Yue2Plan& plan, std::string& err) {
    if (!ev::isObject(v)) { err = "plan must be an object (from plan() or generate())"; return false; }
    ev::Persistent o(v);
    getStrOpt(o.get(), "abc", plan.abc);
    plan.abc_ids = readInt32Array(ev::getProperty(o.get(), "abcIds"));
    plan.prefix = readInt32Array(ev::getProperty(o.get(), "prefix"));
    getBoolOpt(o.get(), "truncated", plan.truncated);
    if (plan.prefix.empty()) { err = "plan.prefix must be a non-empty Int32Array"; return false; }
    return true;
}

Value makePlan(const brosoundml::Yue2Plan& plan) {
    ObjectBuilder o;
    o.set("abc", plan.abc);
    {
        ev::Persistent a(makeInt32Array(plan.abc_ids));
        o.set("abcIds", a.get());
    }
    {
        ev::Persistent a(makeInt32Array(plan.prefix));
        o.set("prefix", a.get());
    }
    o.set("truncated", ev::fromBool(plan.truncated));
    return o.get();
}

}  // namespace music

using namespace music;

namespace {

// Length of the longest prefix of `s` that ends on a UTF-8 code-point boundary.
size_t utf8CompletePrefix(const std::string& s) {
    size_t n = s.size(), i = n;
    int back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = static_cast<unsigned char>(s[i - 1]);
        if ((c & 0xC0) != 0x80) {   // a lead byte (or ASCII) at i-1
            const int need = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
            return (n - (i - 1) >= static_cast<size_t>(need)) ? n : i - 1;
        }
        --i;
        ++back;
    }
    return n;
}

// ── the generate / stage job ─────────────────────────────────────────────────

enum class Mode { Generate, Plan, Semantic, Synthesize };

struct AbcPiece { std::string text; int32_t token = 0; };
struct LatChunk { int frame0 = 0, frames = 0; std::vector<float> data; };
struct AudioChunk { std::int64_t sample0 = 0; brosoundml::StereoAudio audio; };

// One captured residual block, rows of consecutive positions merged.
struct HiddenBlock {
    brosoundml::Yue2Stage stage = brosoundml::Yue2Stage::Plan;
    int layer = 0, pos0 = 0, rows = 0, hidden = 0, narEval = -1;
    std::vector<float> data;
};

struct MusicJob {
    Mode mode = Mode::Generate;
    brosoundml::Yue2Params params;
    brosoundml::Yue2Plan planIn;
    std::vector<int32_t> semanticIn;

    std::shared_ptr<brosoundml::Yue2> model;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate gate;
    std::shared_ptr<brosoundml::Yue2Vae> vae;   // optional: stream latents to audio
    ModelGate vaeGate;
    int vaeCore = 0, vaeHalo = 0, vaeFirst = 0;

    // work -> JS
    SpscSlots<AbcPiece> abc;
    SpscSlots<int32_t> sem;
    SpscSlots<LatChunk> lat;
    SpscSlots<AudioChunk> audio;
    std::atomic<std::uint32_t> progSeq{0};
    std::atomic<int> progStage{0}, progDone{0}, progTotal{0};
    std::uint32_t progSeen = 0;
    std::string abcPending;   // JS side: bytes held back mid code point

    // results (work thread writes, done reads after join)
    brosoundml::Yue2Result res;
    std::vector<brosoundml::StereoAudio> audioParts;   // stream chunks, in order
    brosoundml::StereoAudio audioOut;
    std::vector<HiddenBlock> hidden;

    bool wantAbc = false, wantSem = false, wantLat = false, wantAudio = false, wantProg = false;
    ev::Persistent onAbc, onSemantic, onLatents, onAudio, onProgress, onDone, modelRef, vaeRef;
};

void captureHidden(MusicJob& job, const brosoundml::Yue2Hidden& h) {
    if (!h.data || h.rows <= 0 || h.hidden <= 0) return;
    for (auto it = job.hidden.rbegin(); it != job.hidden.rend(); ++it) {
        HiddenBlock& b = *it;
        if (b.stage == h.stage && b.layer == h.layer && b.narEval == h.nar_eval && b.hidden == h.hidden) {
            if (b.pos0 + b.rows == h.pos0) {
                b.data.insert(b.data.end(), h.data, h.data + static_cast<size_t>(h.rows) * h.hidden);
                b.rows += h.rows;
                return;
            }
            break;
        }
    }
    HiddenBlock b;
    b.stage = h.stage;
    b.layer = h.layer;
    b.pos0 = h.pos0;
    b.rows = h.rows;
    b.hidden = h.hidden;
    b.narEval = h.nar_eval;
    b.data.assign(h.data, h.data + static_cast<size_t>(h.rows) * h.hidden);
    job.hidden.push_back(std::move(b));
}

void runMusicJob(const std::shared_ptr<MusicJob>& job, const std::atomic<bool>& cancel) {
    MusicJob& j = *job;
    brotensor::DeviceScope scope(j.device);
    auto cancelFn = cancelCheckOf(cancel);

    std::optional<brosoundml::Yue2VaeStream> stream;
    if (j.vae) {
        stream.emplace(*j.vae, [&j](const brosoundml::StereoAudio& a, std::int64_t s0) {
            j.audioParts.push_back(a);
            if (j.wantAudio)
                j.audio.emplace([&](AudioChunk& slot) { slot.sample0 = s0; slot.audio = a; });
        }, j.vaeCore, j.vaeHalo, j.vaeFirst);
    }

    brosoundml::Yue2Callbacks cb;
    if (j.wantAbc)
        cb.on_abc = [&j](const std::string& piece, int32_t tok) {
            j.abc.emplace([&](AbcPiece& s) { s.text = piece; s.token = tok; });
        };
    if (j.wantSem) cb.on_semantic = [&j](int32_t code) { j.sem.push(code); };
    cb.on_latents = [&j, &stream](int frame0, int frames, const float* data) {
        if (j.wantLat)
            j.lat.emplace([&](LatChunk& s) {
                s.frame0 = frame0;
                s.frames = frames;
                s.data.assign(data, data + static_cast<size_t>(frames) * brosoundml::Yue2::kLatentDim);
            });
        if (stream) stream->push(frame0, frames, data);
    };
    cb.on_progress = [&j](brosoundml::Yue2Stage st, int done, int total) {
        j.progStage.store(static_cast<int>(st), std::memory_order_relaxed);
        j.progDone.store(done, std::memory_order_relaxed);
        j.progTotal.store(total, std::memory_order_relaxed);
        j.progSeq.fetch_add(1, std::memory_order_release);
    };
    if (!j.params.capture.layers.empty())
        cb.on_hidden = [&j](const brosoundml::Yue2Hidden& h) { captureHidden(j, h); };

    switch (j.mode) {
        case Mode::Generate:
            j.res = j.model->generate(j.params, cb, cancelFn);
            break;
        case Mode::Plan:
            j.res.plan = j.model->plan(j.params, cb, cancelFn);
            break;
        case Mode::Semantic:
            j.res.plan = j.planIn;
            j.res.semantic = j.model->semantic(j.params, j.planIn, cb, cancelFn, &j.res.semantic_truncated);
            break;
        case Mode::Synthesize:
            j.res.plan = j.planIn;
            j.res.semantic = j.semanticIn;
            j.res.latents = j.model->synthesize(j.params, j.planIn, j.semanticIn, cb, cancelFn);
            j.res.frames = static_cast<int>(j.res.latents.size() / brosoundml::Yue2::kLatentDim);
            break;
    }
    j.res.cancelled = j.res.cancelled || cancel.load();
    // The tail's right context is the end of the song; a cancelled song keeps
    // what already played and flushes nothing.
    if (stream && !j.res.cancelled) stream->finish();

    // The whole song, planar: the chunks laid end to end per channel.
    if (!j.audioParts.empty()) {
        brosoundml::StereoAudio& out = j.audioOut;
        out.channels = j.audioParts.front().channels;
        out.sample_rate = j.audioParts.front().sample_rate;
        out.frames = 0;
        for (const auto& p : j.audioParts) out.frames += p.frames;
        out.samples.resize(static_cast<size_t>(out.channels) * out.frames);
        int at = 0;
        for (const auto& p : j.audioParts) {
            for (int c = 0; c < out.channels; ++c) std::copy_n(p.channel(c), p.frames, out.channel(c) + at);
            at += p.frames;
        }
        std::vector<brosoundml::StereoAudio>().swap(j.audioParts);
    }
}

Value makeTiming(const brosoundml::Yue2Timing& t) {
    ObjectBuilder o;
    o.set("planSeconds", ev::fromDouble(t.plan_seconds));
    o.set("semanticSeconds", ev::fromDouble(t.semantic_seconds));
    o.set("synthSeconds", ev::fromDouble(t.synth_seconds));
    o.set("planTtft", ev::fromDouble(t.plan_ttft));
    o.set("semanticTtft", ev::fromDouble(t.semantic_ttft));
    o.set("planTokens", ev::fromDouble(t.plan_tokens));
    o.set("semanticTokens", ev::fromDouble(t.semantic_tokens));
    o.set("planTps", ev::fromDouble(t.plan_tps));
    o.set("planForwards", ev::fromDouble(t.plan_forwards));
    o.set("semanticTps", ev::fromDouble(t.semantic_tps));
    o.set("firstLatentSeconds", ev::fromDouble(t.first_latent_seconds));
    o.set("narSecondsPerAudioSecond", ev::fromDouble(t.nar_seconds_per_audio_second));
    return o.get();
}

Value makeMusicResult(MusicJob& j) {
    ObjectBuilder o;
    {
        ev::Persistent p(makePlan(j.res.plan));
        o.set("plan", p.get());
    }
    o.set("abc", j.res.plan.abc);
    {
        ev::Persistent s(makeInt32Array(j.res.semantic));
        o.set("semantic", s.get());
    }
    o.set("semanticTruncated", ev::fromBool(j.res.semantic_truncated));
    {
        ev::Persistent l(makeFloat32Array(j.res.latents));
        o.set("latents", l.get());
    }
    o.set("frames", ev::fromDouble(j.res.frames));
    {
        ev::Persistent t(makeTiming(j.res.timing));
        o.set("timing", t.get());
    }
    o.set("cancelled", ev::fromBool(j.res.cancelled));
    if (j.vae && j.audioOut.frames > 0) {
        ev::Persistent a(makeStereo(j.audioOut));
        o.set("audio", a.get());
    }
    if (!j.hidden.empty()) {
        ev::Persistent arr(hostArrayOf(j.hidden.size(), [&](size_t i) -> Value {
            const HiddenBlock& b = j.hidden[i];
            ObjectBuilder h;
            h.set("stage", std::string(stageName(b.stage)));
            h.set("layer", ev::fromDouble(b.layer));
            h.set("pos0", ev::fromDouble(b.pos0));
            h.set("rows", ev::fromDouble(b.rows));
            h.set("hidden", ev::fromDouble(b.hidden));
            h.set("narEval", ev::fromDouble(b.narEval));
            ev::Persistent d(makeFloat32Array(b.data));
            h.set("data", d.get());
            return h.get();
        }));
        o.set("hidden", arr.get());
    }
    return o.get();
}

void pollMusicJob(MusicJob& j) {
    if (j.wantAbc) {
        std::string text;
        std::vector<int32_t> toks;
        j.abc.drain([&](AbcPiece& p) {
            text += p.text;
            toks.push_back(p.token);
            std::string().swap(p.text);
        });
        if (!toks.empty()) {
            j.abcPending += text;
            const size_t n = utf8CompletePrefix(j.abcPending);
            ev::Persistent s(ev::fromUtf8(j.abcPending.substr(0, n)));
            j.abcPending.erase(0, n);
            ev::Persistent t(makeInt32Array(toks));
            callCallback2(j.onAbc.get(), s.get(), t.get());
        }
    }
    if (j.wantSem) {
        std::vector<int32_t> codes;
        j.sem.drain([&](int32_t c) { codes.push_back(c); });
        if (!codes.empty()) {
            ev::Persistent c(makeInt32Array(codes));
            callCallback1(j.onSemantic.get(), c.get());
        }
    }
    if (j.wantLat) {
        j.lat.drain([&](LatChunk& c) {
            ObjectBuilder o;
            o.set("frame0", ev::fromDouble(c.frame0));
            o.set("frames", ev::fromDouble(c.frames));
            {
                ev::Persistent d(makeFloat32Array(c.data));
                o.set("latents", d.get());
            }
            ev::Persistent ob(o.get());
            callCallback1(j.onLatents.get(), ob.get());
            std::vector<float>().swap(c.data);
        });
    }
    if (j.wantAudio) {
        j.audio.drain([&](AudioChunk& c) {
            ev::Persistent a(makeStereo(c.audio));
            ev::setProperty(a.get(), "sample0", ev::fromDouble(static_cast<double>(c.sample0)));
            callCallback1(j.onAudio.get(), a.get());
            std::vector<float>().swap(c.audio.samples);
        });
    }
    if (j.wantProg) {
        const std::uint32_t seq = j.progSeq.load(std::memory_order_acquire);
        if (seq != j.progSeen) {
            j.progSeen = seq;
            ObjectBuilder o;
            o.set("stage", std::string(stageName(static_cast<brosoundml::Yue2Stage>(j.progStage.load()))));
            o.set("done", ev::fromDouble(j.progDone.load()));
            o.set("total", ev::fromDouble(j.progTotal.load()));
            ev::Persistent ob(o.get());
            callCallback1(j.onProgress.get(), ob.get());
        }
    }
}

// Shared launcher: `optsAt` indexes the options object in `a`. The caller has
// filled job->mode / planIn / semanticIn.
Value launchMusic(Value self, std::span<const Value> a, size_t optsAt, std::shared_ptr<MusicJob> job,
                  const char* fn) {
    ev::Persistent selfRoot(self);
    auto* w = yue2Of(selfRoot.get());
    if (!w) return ev::throwTypeError(std::string(fn) + ": not a YuE2 model");
    if (!w->model || !w->model->loaded()) return ev::throwError(std::string(fn) + ": model is not loaded");
    ev::Persistent opts(isObjectArg(a, optsAt) ? a[optsAt] : ev::undefined());
    std::string err;
    if (!readParams(opts.get(), job->params, err)) return ev::throwTypeError(std::string(fn) + ": " + err);

    HostYue2Vae* vw = nullptr;
    if (ev::isObject(opts.get())) {
        job->onAbc = getFunctionOpt(opts.get(), "onAbc");
        job->onSemantic = getFunctionOpt(opts.get(), "onSemantic");
        job->onLatents = getFunctionOpt(opts.get(), "onLatents");
        job->onAudio = getFunctionOpt(opts.get(), "onAudio");
        job->onProgress = getFunctionOpt(opts.get(), "onProgress");
        job->onDone = getFunctionOpt(opts.get(), "onDone");
        ev::Persistent vaeVal(ev::getProperty(opts.get(), "vae"));
        if (!ev::isUndefined(vaeVal.get()) && !ev::isNull(vaeVal.get())) {
            vw = vaeOf(vaeVal.get());
            if (!vw || !vw->vae || !vw->vae->loaded())
                return ev::throwTypeError(std::string(fn) + ": opts.vae must be a loaded YuE2 VAE");
            job->vaeRef = ev::Persistent(vaeVal.get());
            getIntOpt(opts.get(), "decodeCoreFrames", job->vaeCore);
            getIntOpt(opts.get(), "decodeHaloFrames", job->vaeHalo);
            getIntOpt(opts.get(), "decodeFirstFrames", job->vaeFirst);
            if (job->vaeHalo > 0 && job->vaeHalo < vw->vae->required_halo(job->vaeCore > 0 ? job->vaeCore : 50))
                return ev::throwTypeError(std::string(fn) + ": decodeHaloFrames is below the VAE's required halo");
        }
    }
    job->wantAbc = ev::isFunction(job->onAbc.get());
    job->wantSem = ev::isFunction(job->onSemantic.get());
    job->wantLat = ev::isFunction(job->onLatents.get());
    job->wantAudio = vw && ev::isFunction(job->onAudio.get());
    job->wantProg = ev::isFunction(job->onProgress.get());
    // Slot capacities: every token / chunk a run can produce. Latent and audio
    // chunks are at most one per frame of the 24576-token context.
    if (job->wantAbc) job->abc.reserve(static_cast<size_t>(std::max(0, job->params.abc_sampling.max_tokens)) + 16);
    if (job->wantSem) job->sem.reserve(static_cast<size_t>(std::max(0, job->params.semantic_sampling.max_tokens)) + 16);
    if (job->wantLat) job->lat.reserve(24576);
    if (job->wantAudio) job->audio.reserve(24576);

    if (!w->busy.tryClaim()) return ev::throwError(std::string(fn) + ": an operation is already in flight on this model");
    if (vw && !vw->busy.tryClaim()) {
        w->busy.release();
        return ev::throwError(std::string(fn) + ": an operation is already in flight on the VAE");
    }
    job->model = w->model;
    job->device = w->device;
    job->gate = w->busy;
    job->modelRef = ev::Persistent(selfRoot.get());
    if (vw) {
        job->vae = vw->vae;
        job->vaeGate = vw->busy;
    }

    auto work = [job](const std::atomic<bool>& cancel) { runMusicJob(job, cancel); };
    auto poll = [job] { pollMusicJob(*job); };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        if (job->vae) job->vaeGate.release();
        if (!ev::isFunction(job->onDone.get())) return;
        job->res.cancelled = job->res.cancelled || cancelled;
        ev::Persistent result(makeMusicResult(*job));
        ev::Persistent info(makeDoneInfo(job->res.cancelled, error));
        callCallback2(job->onDone.get(), result.get(), info.get());
    };
    return launchAsyncJob(std::move(work), std::move(poll), std::move(done));
}

// model.generate(opts) -> AsyncHandle
Value yue2Generate(Value self, std::span<const Value> a) {
    auto job = std::make_shared<MusicJob>();
    job->mode = Mode::Generate;
    return launchMusic(self, a, 0, std::move(job), "generate");
}

// model.plan(opts) -> AsyncHandle (onDone result.plan)
Value yue2Plan(Value self, std::span<const Value> a) {
    auto job = std::make_shared<MusicJob>();
    job->mode = Mode::Plan;
    return launchMusic(self, a, 0, std::move(job), "plan");
}

// model.semantic(plan, opts) -> AsyncHandle (onDone result.semantic)
Value yue2Semantic(Value self, std::span<const Value> a) {
    ev::Persistent selfRoot(self);
    auto job = std::make_shared<MusicJob>();
    job->mode = Mode::Semantic;
    std::string err;
    if (!readPlan(argAt(a, 0), job->planIn, err)) return ev::throwTypeError("semantic(plan, opts): " + err);
    return launchMusic(selfRoot.get(), a, 1, std::move(job), "semantic");
}

// model.synthesize(plan, semantic, opts) -> AsyncHandle (onDone result.latents [+ audio])
Value yue2Synthesize(Value self, std::span<const Value> a) {
    ev::Persistent selfRoot(self);
    auto job = std::make_shared<MusicJob>();
    job->mode = Mode::Synthesize;
    std::string err;
    if (!readPlan(argAt(a, 0), job->planIn, err)) return ev::throwTypeError("synthesize(plan, semantic, opts): " + err);
    job->semanticIn = hasArg(a, 1) ? readInt32Array(a[1]) : std::vector<int32_t>{};
    if (job->semanticIn.empty())
        return ev::throwTypeError("synthesize(plan, semantic, opts): semantic must be a non-empty Int32Array");
    return launchMusic(selfRoot.get(), a, 2, std::move(job), "synthesize");
}

void decorateYue2(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = yue2Of(self);
        return ev::fromBool(w && w->model && w->model->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = yue2Of(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("busy", [](Value self, std::span<const Value>) -> Value {
        auto* w = yue2Of(self);
        return ev::fromBool(w && w->busy.isBusy());
    });
    b.accessor("arWeights", [](Value self, std::span<const Value>) -> Value {
        auto* w = yue2Of(self);
        return ev::fromUtf8(w ? w->arWeights : std::string("bf16"));
    });
    b.accessor("latentDim", [](Value, std::span<const Value>) -> Value {
        return ev::fromDouble(brosoundml::Yue2::kLatentDim);
    });
    b.accessor("frameRate", [](Value, std::span<const Value>) -> Value {
        return ev::fromDouble(brosoundml::Yue2::kFrameRate);
    });
    b.def("generate", 1, yue2Generate);
    b.def("plan", 1, yue2Plan);
    b.def("semantic", 2, yue2Semantic);
    b.def("synthesize", 3, yue2Synthesize);
    b.def("stream", 1, music::yue2Stream);
    b.def("endless", 1, music::yue2Endless);
    b.def("splice", 2, music::yue2Splice);
    b.def("inpaint", 6, music::yue2Inpaint);
    b.def("tokenize", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = yue2Of(self);
        if (!w || !w->model || !w->model->loaded()) return ev::throwError("tokenize: model is not loaded");
        if (!isStringArg(a, 0)) return ev::throwTypeError("tokenize(text): text string required");
        try {
            return makeInt32Array(w->model->tokenize(strAt(a, 0)));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("tokenize: ") + e.what());
        }
    });
    b.def("detokenize", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = yue2Of(self);
        if (!w || !w->model || !w->model->loaded()) return ev::throwError("detokenize: model is not loaded");
        if (!hasArg(a, 0)) return ev::throwTypeError("detokenize(ids): Int32Array required");
        try {
            return ev::fromUtf8(w->model->detokenize(readInt32Array(a[0])));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("detokenize: ") + e.what());
        }
    });
}

// ── VAE ──────────────────────────────────────────────────────────────────────

// vae.decode(latents, opts?) -> { samples, channels, frames, sampleRate } (sync)
//                              | AsyncHandle (opts.onDone; opts.onChunk streams)
//   latents: frame-major Float32Array (frames * 64). opts.coreFrames /
//   haloFrames tile the decode (exact); opts.onChunk(chunk) receives audio in
//   order (chunk.sample0) as tiles finish, tiles growing from opts.firstFrames
//   (default 8) to coreFrames.
Value vaeDecode(Value self, std::span<const Value> a) {
    ev::Persistent selfRoot(self);
    auto* w = vaeOf(selfRoot.get());
    if (!w || !w->vae || !w->vae->loaded()) return ev::throwError("decode: VAE is not loaded");
    if (!hasArg(a, 0)) return ev::throwTypeError("decode(latents, opts?): latents Float32Array required");
    std::vector<float> lat = readFloat32Array(a[0]);
    const int D = w->vae->config().latent_dim;
    if (lat.empty() || lat.size() % static_cast<size_t>(D) != 0)
        return ev::throwTypeError("decode: latents.length must be a positive multiple of " + std::to_string(D));
    const int frames = static_cast<int>(lat.size() / static_cast<size_t>(D));
    ev::Persistent opts(isObjectArg(a, 1) ? a[1] : ev::undefined());
    int core = 0, halo = 0, first = 0;
    if (ev::isObject(opts.get())) {
        getIntOpt(opts.get(), "coreFrames", core);
        getIntOpt(opts.get(), "haloFrames", halo);
        getIntOpt(opts.get(), "firstFrames", first);
    }
    if (halo > 0 && halo < w->vae->required_halo(core))
        return ev::throwTypeError("decode: haloFrames is below the required halo (" +
                                  std::to_string(w->vae->required_halo(core)) + ")");
    ev::Persistent onDone = getFunctionOpt(opts.get(), "onDone");
    ev::Persistent onChunk = getFunctionOpt(opts.get(), "onChunk");
    ev::Persistent onProgress = getFunctionOpt(opts.get(), "onProgress");
    if (w->busy.isBusy()) return ev::throwError("decode: an operation is already in flight on this VAE");

    if (!ev::isFunction(onDone.get()) && !ev::isFunction(onChunk.get())) {
        try {
            brotensor::DeviceScope scope(w->device);
            return makeStereo(core > 0 ? w->vae->decode_tiled(lat.data(), frames, core, halo)
                                       : w->vae->decode(lat.data(), frames));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("decode: ") + e.what());
        }
    }

    struct Job {
        std::shared_ptr<brosoundml::Yue2Vae> vae;
        brotensor::Device device;
        ModelGate gate;
        std::vector<float> lat;
        int frames = 0, core = 0, halo = 0, first = 0;
        brosoundml::StereoAudio out;
        SpscSlots<AudioChunk> chunks;
        std::atomic<int> tilesDone{0}, tilesTotal{0};
        int tilesSeen = 0;
        bool streaming = false;
        ev::Persistent self, onDone, onChunk, onProgress;
    };
    auto job = std::make_shared<Job>();
    job->vae = w->vae;
    job->device = w->device;
    job->gate = w->busy;
    job->lat = std::move(lat);
    job->frames = frames;
    job->core = core;
    job->halo = halo;
    job->first = first;
    job->streaming = ev::isFunction(onChunk.get());
    job->self = ev::Persistent(selfRoot.get());
    job->onDone = onDone;
    job->onChunk = onChunk;
    job->onProgress = onProgress;
    if (job->streaming) job->chunks.reserve(static_cast<size_t>(frames) + 2);
    if (!w->busy.tryClaim()) return ev::throwError("decode: an operation is already in flight on this VAE");

    auto work = [job](const std::atomic<bool>& cancel) {
        brotensor::DeviceScope scope(job->device);
        if (job->streaming) {
            brosoundml::Yue2VaeStream stream(*job->vae, [&](const brosoundml::StereoAudio& c, std::int64_t s0) {
                job->chunks.emplace([&](AudioChunk& slot) { slot.sample0 = s0; slot.audio = c; });
            }, job->core, job->halo, job->first);
            // Push a first tile's worth at a time: tiles grow as they would
            // under live latents, and cancel is checked between them.
            const int step = job->first > 0 ? job->first : 8;
            for (int f = 0; f < job->frames && !cancel.load(); f += step) {
                const int n = std::min(step, job->frames - f);
                stream.push(f, n, job->lat.data() + static_cast<size_t>(f) * brosoundml::Yue2::kLatentDim);
            }
            if (!cancel.load()) stream.finish();
        } else {
            job->out = job->vae->decode_tiled(job->lat.data(), job->frames, job->core, job->halo,
                                              [&](int d, int t) {
                                                  job->tilesTotal.store(t);
                                                  job->tilesDone.store(d);
                                              },
                                              cancelCheckOf(cancel));
        }
    };
    auto poll = [job] {
        if (job->streaming) {
            job->chunks.drain([&](AudioChunk& c) {
                ev::Persistent ch(makeStereo(c.audio));
                ev::setProperty(ch.get(), "sample0", ev::fromDouble(static_cast<double>(c.sample0)));
                callCallback1(job->onChunk.get(), ch.get());
                std::vector<float>().swap(c.audio.samples);
            });
        } else if (ev::isFunction(job->onProgress.get())) {
            const int d = job->tilesDone.load();
            if (d != job->tilesSeen) {
                job->tilesSeen = d;
                ObjectBuilder o;
                o.set("done", ev::fromDouble(d));
                o.set("total", ev::fromDouble(job->tilesTotal.load()));
                ev::Persistent ob(o.get());
                callCallback1(job->onProgress.get(), ob.get());
            }
        }
    };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        if (!ev::isFunction(job->onDone.get())) return;
        ev::Persistent result(job->streaming ? ev::undefined() : makeStereo(job->out));
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), result.get(), info.get());
    };
    return launchAsyncJob(std::move(work), std::move(poll), std::move(done));
}

// vae.encode(audio, opts?) -> { latents, frames } (sync) | AsyncHandle (opts.onDone)
Value vaeEncode(Value self, std::span<const Value> a) {
    ev::Persistent selfRoot(self);
    auto* w = vaeOf(selfRoot.get());
    if (!w || !w->vae || !w->vae->loaded()) return ev::throwError("encode: VAE is not loaded");
    if (!w->vae->has_encoder()) return ev::throwError("encode: the VAE was loaded decoderOnly");
    if (!hasArg(a, 0)) return ev::throwTypeError("encode(audio, opts?): audio required");
    brosoundml::StereoAudio in;
    std::string err;
    if (!readStereo(a[0], in, err)) return ev::throwTypeError("encode: " + err);
    ev::Persistent opts(isObjectArg(a, 1) ? a[1] : ev::undefined());
    ev::Persistent onDone = getFunctionOpt(opts.get(), "onDone");
    if (w->busy.isBusy()) return ev::throwError("encode: an operation is already in flight on this VAE");

    auto makeResult = [](const std::vector<float>& lat, int frames) -> Value {
        ObjectBuilder o;
        {
            ev::Persistent l(makeFloat32Array(lat));
            o.set("latents", l.get());
        }
        o.set("frames", ev::fromDouble(frames));
        return o.get();
    };
    if (!ev::isFunction(onDone.get())) {
        try {
            brotensor::DeviceScope scope(w->device);
            int frames = 0;
            std::vector<float> lat = w->vae->encode(in, &frames);
            return makeResult(lat, frames);
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encode: ") + e.what());
        }
    }
    struct Job {
        std::shared_ptr<brosoundml::Yue2Vae> vae;
        brotensor::Device device;
        ModelGate gate;
        brosoundml::StereoAudio in;
        std::vector<float> lat;
        int frames = 0;
        ev::Persistent self, onDone;
    };
    auto job = std::make_shared<Job>();
    job->vae = w->vae;
    job->device = w->device;
    job->gate = w->busy;
    job->in = std::move(in);
    job->self = ev::Persistent(selfRoot.get());
    job->onDone = onDone;
    if (!w->busy.tryClaim()) return ev::throwError("encode: an operation is already in flight on this VAE");
    auto work = [job](const std::atomic<bool>&) {
        brotensor::DeviceScope scope(job->device);
        job->lat = job->vae->encode(job->in, &job->frames);
    };
    auto done = [job, makeResult](bool cancelled, const std::string& error) {
        job->gate.release();
        ev::Persistent result(error.empty() && !cancelled ? makeResult(job->lat, job->frames) : ev::undefined());
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), result.get(), info.get());
    };
    return launchAsyncJob(std::move(work), nullptr, std::move(done));
}

void decorateYue2Vae(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = vaeOf(self);
        return ev::fromBool(w && w->vae && w->vae->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = vaeOf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("busy", [](Value self, std::span<const Value>) -> Value {
        auto* w = vaeOf(self);
        return ev::fromBool(w && w->busy.isBusy());
    });
    b.accessor("hasEncoder", [](Value self, std::span<const Value>) -> Value {
        auto* w = vaeOf(self);
        return ev::fromBool(w && w->vae && w->vae->has_encoder());
    });
    b.accessor("sampleRate", [](Value self, std::span<const Value>) -> Value {
        auto* w = vaeOf(self);
        return ev::fromDouble(w && w->vae ? w->vae->config().sample_rate : 0);
    });
    b.accessor("channels", [](Value self, std::span<const Value>) -> Value {
        auto* w = vaeOf(self);
        return ev::fromDouble(w && w->vae ? w->vae->config().channels : 0);
    });
    b.accessor("latentDim", [](Value self, std::span<const Value>) -> Value {
        auto* w = vaeOf(self);
        return ev::fromDouble(w && w->vae ? w->vae->config().latent_dim : 0);
    });
    b.accessor("downsamplingRatio", [](Value self, std::span<const Value>) -> Value {
        auto* w = vaeOf(self);
        return ev::fromDouble(w && w->vae ? w->vae->config().downsampling_ratio : 0);
    });
    b.accessor("frameRate", [](Value self, std::span<const Value>) -> Value {
        auto* w = vaeOf(self);
        if (!w || !w->vae || w->vae->config().downsampling_ratio <= 0) return ev::fromDouble(0);
        return ev::fromDouble(double(w->vae->config().sample_rate) / w->vae->config().downsampling_ratio);
    });
    b.def("requiredHalo", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = vaeOf(self);
        if (!w || !w->vae || !w->vae->loaded()) return ev::throwError("requiredHalo: VAE is not loaded");
        return ev::fromDouble(w->vae->required_halo(i32At(a, 0, 0)));
    });
    b.def("outputLength", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = vaeOf(self);
        if (!w || !w->vae || !w->vae->loaded()) return ev::throwError("outputLength: VAE is not loaded");
        const int f = i32At(a, 0, 0);
        if (f <= 0) return ev::throwTypeError("outputLength(frames): frames must be positive");
        return ev::fromDouble(w->vae->output_length(f));
    });
    b.def("decode", 2, vaeDecode);
    b.def("encode", 2, vaeEncode);
}

// ── loaders ──────────────────────────────────────────────────────────────────

Value musicInit(Value, std::span<const Value>) {
    try {
        brotensor::init();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.music.init: ") + e.what());
    }
    return ev::undefined();
}

// bro.music.loadYuE2(modelDir, opts?) -> YuE2 | AsyncHandle (opts.onReady)
//   opts.arWeights: 'int8' (default) | 'fp8' | 'bf16', the weights the
//   one-token AR decode (plan + semantic sampling) reads; 'bf16' is the
//   checkpoint's own (reference parity).
Value loadYuE2(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    ev::Persistent opts;
    if (!modelLoaderArgs("loadYuE2", args, dir, dev, opts)) return ev::undefined();
    const std::string ar = ev::isObject(opts.get()) ? getPropertyString(opts.get(), "arWeights", "int8") : "int8";
    brosoundml::Yue2LoadOptions lo;
    if (ar == "bf16") lo.ar_weights = brosoundml::Yue2ArWeights::Bf16;
    else if (ar == "int8") lo.ar_weights = brosoundml::Yue2ArWeights::Int8;
    else if (ar == "fp8") lo.ar_weights = brosoundml::Yue2ArWeights::Fp8;
    else return ev::throwTypeError("loadYuE2: arWeights must be 'bf16', 'int8' or 'fp8', got '" + ar + "'");
    return runModelLoader<HostYue2>("loadYuE2", opts.get(), g_yue2Class, [dir, dev, lo, ar] {
        auto w = std::make_unique<HostYue2>();
        w->device = dev;
        w->arWeights = ar;
        w->model = std::make_shared<brosoundml::Yue2>();
        {
            brotensor::DeviceScope scope(dev);
            w->model->load(dir, dev, lo);
        }
        logInfo(std::string("[music] YuE2 loaded on ") + deviceName(dev) + " (AR decode weights " + ar + ")");
        return w;
    });
}

// bro.music.loadVae(vaeDir, opts?) -> YuE2Vae | AsyncHandle (opts.onReady)
//   opts.decoderOnly skips the encoder.
Value loadVae(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    ev::Persistent opts;
    if (!modelLoaderArgs("loadVae", args, dir, dev, opts)) return ev::undefined();
    const bool decoderOnly = ev::isObject(opts.get()) && getPropertyBool(opts.get(), "decoderOnly");
    return runModelLoader<HostYue2Vae>("loadVae", opts.get(), g_yue2VaeClass, [dir, dev, decoderOnly] {
        auto w = std::make_unique<HostYue2Vae>();
        w->device = dev;
        w->vae = std::make_shared<brosoundml::Yue2Vae>();
        {
            brotensor::DeviceScope scope(dev);
            w->vae->load(dir, dev, decoderOnly);
        }
        logInfo(std::string("[music] YuE2 VAE loaded on ") + deviceName(dev));
        return w;
    });
}

}  // namespace

void installMusic(ObjectBuilder& bro) {
    g_yue2Class.install("YuE2", 0, nullptr, decorateYue2);
    g_yue2VaeClass.install("YuE2Vae", 0, nullptr, decorateYue2Vae);
    music::installStreamClass();
    music::installEndlessClass();
    music::installSeparatorClass();

    ObjectBuilder music;
    music.def("init", 0, musicInit);
    music.def("beat", 2, music::musicBeat);
    music.def("loadYuE2", 2, loadYuE2);
    music.def("loadVae", 2, loadVae);
    music.def("loadSeparator", 2, music::loadSeparator);
    music.set("YuE2", g_yue2Class.constructor());
    music.set("YuE2Vae", g_yue2VaeClass.constructor());
    // A stream's knob clamps with no model or stream: a fresh object each
    // read, the one a running stream's knobRanges gives.
    music.accessor("knobRanges", [](Value, std::span<const Value>) -> Value { return music::knobRanges(); });
    // The energy knob's latent direction (64 floats): a stream decodes
    // latents + energy[f] * direction, so a take's audio is that sum.
    music.accessor("energyDirection", [](Value, std::span<const Value>) -> Value {
        return makeFloat32Array(brosoundml::Yue2Directions::builtin_ref().latent);
    });
    bro.set("music", music.get());
}

} // namespace brosoundml::api
