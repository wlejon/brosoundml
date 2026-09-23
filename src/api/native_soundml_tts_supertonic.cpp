// bro.tts — Supertonic-3: flow-matching multilingual TTS, text-driven
// (codepoint frontend, no phoneme step); a voice is a VoiceStyle preset. The
// loader lives in native_soundml_tts.cpp.
#include "soundml_tts_internal.h"

#include <filesystem>

namespace brosoundml::api {

namespace {

HostSupertonic* stSelf(Value self) {
    return static_cast<HostSupertonic*>(g_supertonicClass.unwrap(self));
}
HostSupertonicVoice* voiceOf(Value v) {
    return unwrapTtsAs<HostSupertonicVoice>(g_supertonicVoiceClass, v);
}

struct SupertonicOpts {
    std::string language = "en";
    int steps = 8;
    float speed = 1.05f;
    std::uint64_t seed = 0;
    bool longForm = false;
    float gapSeconds = 0.3f;
    float guidance = 3.0f;
};

void readSupertonicOpts(Value opts, SupertonicOpts& o) {
    if (!ev::isObject(opts)) return;
    ev::Persistent root(opts);  // every read below allocates
    getStrOpt(root.get(), "language", o.language);
    getIntOpt(root.get(), "steps", o.steps);
    getFloatOpt(root.get(), "speed", o.speed);
    getFloatOpt(root.get(), "gapSeconds", o.gapSeconds);
    getFloatOpt(root.get(), "guidance", o.guidance);
    getSeedOpt(root.get(), o.seed);
    getBoolOpt(root.get(), "longForm", o.longForm);
    if (o.steps < 1) o.steps = 1;
}

brosoundml::AudioBuffer runSupertonic(const brosoundml::Supertonic& model, const std::string& text,
                                      const SupertonicOpts& o, const brosoundml::VoiceStyle& style) {
    return o.longForm
        ? model.synthesize_long(text, o.language, style, o.steps, o.speed, o.seed, o.gapSeconds, 300, o.guidance)
        : model.synthesize(text, o.language, style, o.steps, o.speed, o.seed, o.guidance);
}

Value makeStyleVoice(brosoundml::VoiceStyle style, std::string name) {
    auto vw = std::make_unique<HostSupertonicVoice>();
    vw->style = std::move(style);
    vw->name = std::move(name);
    return g_supertonicVoiceClass.createInstance(std::move(vw));
}

void decorateSupertonicVoice(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = voiceOf(self);
        return ev::fromBool(w && !w->style.ttl.empty());
    });
    b.accessor("name", [](Value self, std::span<const Value>) -> Value {
        auto* w = voiceOf(self);
        return ev::fromUtf8(w ? w->name : "");
    });
    // The two style matrices as row-major Float32Arrays: ttl = 50x256, dp = 8x16.
    b.accessor("ttl", [](Value self, std::span<const Value>) -> Value {
        auto* w = voiceOf(self);
        return makeFloat32Array(w ? w->style.ttl : std::vector<float>{});
    });
    b.accessor("dp", [](Value self, std::span<const Value>) -> Value {
        auto* w = voiceOf(self);
        return makeFloat32Array(w ? w->style.dp : std::vector<float>{});
    });
    b.accessor("ttlRows", [](Value, std::span<const Value>) -> Value { return ev::fromDouble(50); });
    b.accessor("ttlCols", [](Value, std::span<const Value>) -> Value { return ev::fromDouble(256); });
    b.accessor("dpRows", [](Value, std::span<const Value>) -> Value { return ev::fromDouble(8); });
    b.accessor("dpCols", [](Value, std::span<const Value>) -> Value { return ev::fromDouble(16); });
}

// supertonic.loadVoiceStyle(path) -> SupertonicVoice (voice_styles/<name>.json)
Value loadVoiceStyle(Value self, std::span<const Value> a) {
    auto* w = stSelf(self);
    if (!w) return ev::throwTypeError("loadVoiceStyle: not a Supertonic");
    if (!isStringArg(a, 0)) return ev::throwTypeError("loadVoiceStyle(path): path string required");
    if (!w->model || !w->model->loaded()) return ev::throwError("loadVoiceStyle: model is not loaded");
    const std::string path = resolvePath(strAt(a, 0));
    try {
        return makeStyleVoice(w->model->load_voice_style(path), std::filesystem::path(path).stem().string());
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadVoiceStyle: ") + e.what());
    }
}

// supertonic.createVoice(ttl, dp, name?) -> SupertonicVoice from raw matrices.
Value createVoice(Value self, std::span<const Value> a) {
    auto* w = stSelf(self);
    if (!w) return ev::throwTypeError("createVoice: not a Supertonic");
    if (a.size() < 2) return ev::throwTypeError("createVoice(ttl, dp, name?): ttl and dp required");
    brosoundml::VoiceStyle style;
    style.ttl = readFloat32Array(a[0]);
    style.dp = readFloat32Array(a[1]);
    if (style.ttl.size() != 50u * 256u) return ev::throwTypeError("createVoice: ttl must have 50*256 = 12800 floats");
    if (style.dp.size() != 8u * 16u) return ev::throwTypeError("createVoice: dp must have 8*16 = 128 floats");
    return makeStyleVoice(std::move(style), isStringArg(a, 2) ? strAt(a, 2) : "custom");
}

// supertonic.synthesize(text, opts) -> { samples, sampleRate }  (sync)
Value synthesizeSync(Value self, std::span<const Value> a) {
    auto* w = stSelf(self);
    if (!w) return ev::throwTypeError("synthesize: not a Supertonic");
    if (!isStringArg(a, 0)) return ev::throwTypeError("synthesize(text, opts): text string required");
    if (!w->model || !w->model->loaded()) return ev::throwError("synthesize: model is not loaded");
    HostSupertonicVoice* vw = nullptr;
    SupertonicOpts o;
    if (isObjectArg(a, 1)) {
        ev::Persistent opts(a[1]);
        vw = voiceOf(ev::getProperty(opts.get(), "voice"));
        readSupertonicOpts(opts.get(), o);
    }
    if (!vw) return ev::throwTypeError("synthesize: opts.voice must be a SupertonicVoice (from loadVoiceStyle)");
    if (w->busy.isBusy()) return ev::throwError("synthesize: an operation is already in flight on this model");
    try {
        brotensor::DeviceScope scope(w->device);
        auto buf = runSupertonic(*w->model, strAt(a, 0), o, vw->style);
        return makeAudioResult(buf.samples, buf.sample_rate);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("synthesize: ") + e.what());
    }
}

void decorateSupertonic(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = stSelf(self);
        return ev::fromBool(w && w->model && w->model->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = stSelf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("sampleRate", [](Value self, std::span<const Value>) -> Value {
        auto* w = stSelf(self);
        return ev::fromDouble(w && w->model ? w->model->config().sample_rate : 0);
    });
    b.def("loadVoiceStyle", 1, loadVoiceStyle);
    b.def("createVoice", 3, createVoice);
    b.def("synthesize", 2, synthesizeSync);
}

}  // namespace

// bro.tts.synthesize(supertonic, text, opts) -> AsyncHandle
//   opts.voice (required), language, steps, speed, seed, longForm,
//   gapSeconds, guidance, onDone(result, info). Supertonic has no per-step
//   cancel hook (the flow loop is short), so cancel is not polled.
Value supertonicSynthesizeAsync(Value modelVal, HostSupertonic* w, std::span<const Value> args) {
    ev::Persistent modelRoot(modelVal);  // rooted: the reads below allocate
    if (!isStringArg(args, 1)) return ev::throwTypeError("synthesize(supertonic, text, opts): text string required");
    if (!w->model || !w->model->loaded()) return ev::throwError("synthesize: model is not loaded");
    struct Job {
        std::string text;
        SupertonicOpts o;
        brosoundml::VoiceStyle style;
        std::vector<float> samples;
        int sampleRate = 44100;
        ModelGate gate;
        brotensor::Device device = brotensor::Device::CPU;
        std::shared_ptr<brosoundml::Supertonic> model;
        ev::Persistent onDone, modelRef, voiceRef;
    };
    auto job = std::make_shared<Job>();
    job->text = strAt(args, 1);
    HostSupertonicVoice* vw = nullptr;
    if (isObjectArg(args, 2)) {
        ev::Persistent opts(args[2]);
        job->voiceRef = ev::Persistent(ev::getProperty(opts.get(), "voice"));
        vw = voiceOf(job->voiceRef.get());
        readSupertonicOpts(opts.get(), job->o);
        job->onDone = getFunctionOpt(opts.get(), "onDone");
    }
    if (!vw) return ev::throwTypeError("synthesize: opts.voice must be a SupertonicVoice (from loadVoiceStyle)");
    if (!w->busy.tryClaim()) return ev::throwError("synthesize: an operation is already in flight on this model");
    job->style = vw->style;
    job->gate = w->busy;
    job->device = w->device;
    job->model = w->model;
    job->modelRef = ev::Persistent(modelRoot.get());
    auto work = [job](const std::atomic<bool>&) {
        brotensor::DeviceScope scope(job->device);
        auto buf = runSupertonic(*job->model, job->text, job->o, job->style);
        job->samples = std::move(buf.samples);
        job->sampleRate = buf.sample_rate;
    };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        if (!ev::isFunction(job->onDone.get())) return;
        ev::Persistent result(makeAudioResult(job->samples, job->sampleRate));
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), result.get(), info.get());
    };
    return launchAsyncJob(std::move(work), nullptr, std::move(done));
}

void installTtsSupertonicClasses() {
    g_supertonicVoiceClass.install("SupertonicVoice", 0, nullptr, decorateSupertonicVoice);
    g_supertonicClass.install("SupertonicModel", 0, nullptr, decorateSupertonic);
}

} // namespace brosoundml::api
