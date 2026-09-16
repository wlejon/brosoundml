// bro.tts — OmniVoice: masked-diffusion zero-shot TTS (k2-fsa). The JS
// surface maps 1:1 onto brosoundml/omnivoice.h: every option name is the
// camelCase of the OmniVoiceParams field; a prompt is the plain-object form
// of OmniVoicePrompt ({ codes, numFrames, text, rms }); an init grid is
// OmniVoiceInit ({ tokens, keep }); a trace is OmniVoiceTrace; a step is
// OmniVoiceStep with its three host arrays copied out (the C++ pointers are
// only valid during the callback, which runs on the worker thread).
// The loader lives in native_soundml_tts.cpp.
#include "soundml_tts_internal.h"

namespace brosoundml::api {

namespace {

HostOmniVoice* omniSelf(Value self) {
    return static_cast<HostOmniVoice*>(g_omniVoiceClass.unwrap(self));
}

const char* precisionName(brosoundml::OmniVoicePrecision p) {
    return p == brosoundml::OmniVoicePrecision::BF16 ? "bf16" : "fp32";
}

// opts -> OmniVoiceParams. Omitted keys keep the upstream defaults.
void readOmniParams(Value opts, brosoundml::OmniVoiceParams& p) {
    if (!ev::isObject(opts)) return;
    getIntOpt(opts, "numSteps", p.num_steps);
    getFloatOpt(opts, "tShift", p.t_shift);
    getFloatOpt(opts, "guidanceScale", p.guidance_scale);
    getFloatOpt(opts, "layerPenalty", p.layer_penalty);
    getFloatOpt(opts, "positionTemperature", p.position_temperature);
    getFloatOpt(opts, "classTemperature", p.class_temperature);
    getBoolOpt(opts, "gumbelNoise", p.gumbel_noise);
    getSeedOpt(opts, p.seed);
    getFloatOpt(opts, "speed", p.speed);
    getFloatOpt(opts, "duration", p.duration);
    getStrOpt(opts, "language", p.language);
    getStrOpt(opts, "instruct", p.instruct);
    getBoolOpt(opts, "denoise", p.denoise);
    getBoolOpt(opts, "preprocessPrompt", p.preprocess_prompt);
    getBoolOpt(opts, "postprocess", p.postprocess_output);
    getFloatOpt(opts, "chunkDuration", p.audio_chunk_duration);
    getFloatOpt(opts, "chunkThreshold", p.audio_chunk_threshold);
    getFloatOpt(opts, "padDuration", p.pad_duration);
    getFloatOpt(opts, "fadeDuration", p.fade_duration);
}

// { codes: Int32Array|number[], numFrames?, text?, rms? } -> OmniVoicePrompt.
// numFrames defaults to codes.length / numCodebooks. False + err on a
// malformed object (the caller turns that into a TypeError).
bool promptFromJs(Value pv, int numCodebooks, brosoundml::OmniVoicePrompt& out, std::string& err) {
    if (!ev::isObject(pv) || ev::isFunction(pv)) {
        err = "prompt must be an object { codes, numFrames, text, rms } (from createPrompt / loadPrompt)";
        return false;
    }
    ev::Persistent root(pv);
    out.codes = readInt32Array(ev::getProperty(root.get(), "codes"));
    if (out.codes.empty()) {
        err = "prompt.codes must be a non-empty Int32Array or number[]";
        return false;
    }
    out.num_frames = 0;
    getIntOpt(root.get(), "numFrames", out.num_frames);
    if (out.num_frames <= 0 && numCodebooks > 0)
        out.num_frames = static_cast<int>(out.codes.size() / static_cast<size_t>(numCodebooks));
    if (out.num_frames <= 0 ||
        static_cast<size_t>(numCodebooks) * static_cast<size_t>(out.num_frames) != out.codes.size()) {
        err = "prompt.codes length must equal numCodebooks * prompt.numFrames";
        return false;
    }
    out.text.clear();
    getStrOpt(root.get(), "text", out.text);
    out.rms = 0.0f;
    getFloatOpt(root.get(), "rms", out.rms);
    return true;
}

// opts.prompt -> OmniVoicePrompt. 0 when absent, 1 when read, -1 (+ err)
// when present but malformed.
int readOmniPrompt(Value opts, int numCodebooks, brosoundml::OmniVoicePrompt& out, std::string& err) {
    if (!ev::isObject(opts)) return 0;
    Value pv = ev::getProperty(opts, "prompt");
    if (ev::isUndefined(pv) || ev::isNull(pv)) return 0;
    return promptFromJs(pv, numCodebooks, out, err) ? 1 : -1;
}

Value promptToJs(const brosoundml::OmniVoicePrompt& p) {
    ObjectBuilder obj;
    {
        ev::Persistent c(makeInt32Array(p.codes));
        obj.set("codes", c.get());
    }
    obj.set("numFrames", ev::fromDouble(p.num_frames));
    obj.set("text", p.text);
    obj.set("rms", ev::fromDouble(p.rms));
    return obj.get();
}

// opts.init -> OmniVoiceInit ({ tokens: Int32Array, keep: Uint8Array }). Sets
// `has`; false + err when present but malformed. The frame-count match is
// left to generate_codes, which reports it through onError.
bool readOmniInit(Value opts, brosoundml::OmniVoiceInit& out, bool& has, std::string& err) {
    has = false;
    if (!ev::isObject(opts)) return true;
    Value iv = ev::getProperty(opts, "init");
    if (ev::isUndefined(iv) || ev::isNull(iv)) return true;
    if (!ev::isObject(iv)) {
        err = "init must be { tokens: Int32Array, keep: Uint8Array }";
        return false;
    }
    ev::Persistent root(iv);
    out.tokens = readInt32Array(ev::getProperty(root.get(), "tokens"));
    out.keep = readByteArray(ev::getProperty(root.get(), "keep"));
    if (out.tokens.empty() || out.keep.size() != out.tokens.size()) {
        err = "init.tokens and init.keep must be the same non-empty length (numCodebooks * numFrames)";
        return false;
    }
    has = true;
    return true;
}

Value traceToJs(const brosoundml::OmniVoiceTrace& tr) {
    ObjectBuilder obj;
    auto setI32 = [&](const char* k, const std::vector<int32_t>& v) {
        ev::Persistent a(makeInt32Array(v));
        obj.set(k, a.get());
    };
    setI32("textIds", tr.text_ids);
    obj.set("numFrames", ev::fromDouble(tr.num_frames));
    setI32("codes", tr.codes);
    setI32("unmaskStep", tr.unmask_step);
    {
        ev::Persistent a(makeFloat32Array(tr.confidence));
        obj.set("confidence", a.get());
    }
    setI32("chunkFrames", tr.chunk_frames);
    obj.set("lmSeconds", ev::fromDouble(tr.lm_seconds));
    obj.set("codecSeconds", ev::fromDouble(tr.codec_seconds));
    return obj.get();
}

Value configToJs(const HostOmniVoice* w) {
    const auto& c = w->model->config();
    ObjectBuilder obj;
    obj.set("sampleRate", ev::fromDouble(c.sample_rate));
    obj.set("frameRate", ev::fromDouble(c.frame_rate));
    obj.set("numCodebooks", ev::fromDouble(c.lm.num_codebooks));
    obj.set("audioVocabSize", ev::fromDouble(c.lm.audio_vocab_size));
    obj.set("maskId", ev::fromDouble(c.lm.audio_mask_id));
    obj.set("hiddenSize", ev::fromDouble(c.lm.hidden_size));
    obj.set("numLayers", ev::fromDouble(c.lm.num_hidden_layers));
    obj.set("precision", precisionName(w->precision));
    obj.set("device", deviceName(w->device));
    obj.set("decoderOnly", ev::fromBool(w->decoderOnly));
    return obj.get();
}

// The sync methods touch the GPU (codec) or read state an in-flight worker
// may be using; refuse while an async op holds the model. Returns false with
// the exception already raised.
bool checkIdle(const HostOmniVoice* w, const char* fn) {
    if (!w->model || !w->model->loaded()) {
        ev::throwError(std::string(fn) + ": model is not loaded");
        return false;
    }
    if (w->busy.isBusy()) {
        ev::throwError(std::string(fn) + ": an operation is already in flight on this model");
        return false;
    }
    return true;
}

// omni.decodeCodes(codes, numFrames?) -> { samples, sampleRate }   (sync)
Value omniDecodeCodes(Value self, std::span<const Value> a) {
    auto* w = omniSelf(self);
    if (!w) return ev::throwTypeError("decodeCodes: not an OmniVoice");
    if (!hasArg(a, 0)) return ev::throwTypeError("decodeCodes(codes, numFrames?): codes required");
    std::vector<int32_t> codes = readInt32Array(a[0]);
    if (codes.empty()) return ev::throwTypeError("decodeCodes: codes must be a non-empty Int32Array");
    if (!checkIdle(w, "decodeCodes")) return ev::undefined();
    const int nq = w->model->config().lm.num_codebooks;
    int nf = hasArg(a, 1) && ev::isNumber(a[1]) ? i32At(a, 1) : 0;
    if (nf <= 0) nf = static_cast<int>(codes.size() / static_cast<size_t>(nq));
    if (nf <= 0 || static_cast<size_t>(nq) * static_cast<size_t>(nf) != codes.size())
        return ev::throwTypeError("decodeCodes: codes.length must equal numCodebooks (" +
                                  std::to_string(nq) + ") * numFrames");
    try {
        brotensor::DeviceScope scope(w->device);
        return audioBufferToJs(w->model->decode_codes(codes, nf));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("decodeCodes: ") + e.what());
    }
}

// omni.encodeAudio(samples, sampleRate?) -> { codes, numFrames, numCodebooks }  (sync)
Value omniEncodeAudio(Value self, std::span<const Value> a) {
    auto* w = omniSelf(self);
    if (!w) return ev::throwTypeError("encodeAudio: not an OmniVoice");
    brosoundml::AudioBuffer audio;
    if (!readAudioArgs(a, 0, audio))
        return ev::throwTypeError("encodeAudio(samples, sampleRate?): samples must be a non-empty Float32Array");
    if (!checkIdle(w, "encodeAudio")) return ev::undefined();
    if (w->decoderOnly)
        return ev::throwError("encodeAudio: the model was loaded with decoderOnly (no codec encoder)");
    try {
        brotensor::DeviceScope scope(w->device);
        int nf = 0;
        std::vector<int32_t> codes = w->model->encode_audio(audio, &nf);
        ObjectBuilder obj;
        {
            ev::Persistent c(makeInt32Array(codes));
            obj.set("codes", c.get());
        }
        obj.set("numFrames", ev::fromDouble(nf));
        obj.set("numCodebooks", ev::fromDouble(w->model->config().lm.num_codebooks));
        return obj.get();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("encodeAudio: ") + e.what());
    }
}

struct OmniPromptJob {
    std::shared_ptr<const brosoundml::OmniVoice> model;
    brotensor::Device device = brotensor::Device::CPU;
    brosoundml::AudioBuffer ref;
    std::string refText;
    bool preprocess = true;
    brosoundml::OmniVoicePrompt out;
    ModelGate gate;
    ev::Persistent modelRef, onDone, onError;
};

// omni.createPrompt(samples, opts?) -> prompt (sync) | AsyncHandle (opts.onDone)
//   opts.sampleRate (default 24000), opts.refText, opts.preprocess (default
//   true). The async path claims the gate: the encoder shares the device.
Value omniCreatePrompt(Value self, std::span<const Value> a) {
    auto* w = omniSelf(self);
    if (!w) return ev::throwTypeError("createPrompt: not an OmniVoice");
    brosoundml::AudioBuffer ref;
    if (!readAudioArgs(a, 0, ref))
        return ev::throwTypeError("createPrompt(samples, opts?): samples must be a non-empty Float32Array");
    ev::Persistent opts(isObjectArg(a, 1) ? a[1] : ev::undefined());
    std::string refText;
    bool preprocess = true;
    getStrOpt(opts.get(), "refText", refText);
    getBoolOpt(opts.get(), "preprocess", preprocess);
    if (!checkIdle(w, "createPrompt")) return ev::undefined();
    if (w->decoderOnly)
        return ev::throwError("createPrompt: the model was loaded with decoderOnly (no codec encoder)");

    ev::Persistent onDone = getFunctionOpt(opts.get(), "onDone");
    if (!ev::isFunction(onDone.get())) {
        try {
            brotensor::DeviceScope scope(w->device);
            return promptToJs(w->model->create_prompt(ref, refText, preprocess));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("createPrompt: ") + e.what());
        }
    }

    if (!w->busy.tryClaim())
        return ev::throwError("createPrompt: an operation is already in flight on this model");
    auto job = std::make_shared<OmniPromptJob>();
    job->model = w->model;
    job->device = w->device;
    job->ref = std::move(ref);
    job->refText = std::move(refText);
    job->preprocess = preprocess;
    job->gate = w->busy;
    job->modelRef = ev::Persistent(self);
    job->onDone = onDone;
    job->onError = getFunctionOpt(opts.get(), "onError");

    auto work = [job](const std::atomic<bool>&) {
        brotensor::DeviceScope scope(job->device);
        job->out = job->model->create_prompt(job->ref, job->refText, job->preprocess);
    };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        if (!error.empty() || cancelled) {
            if (ev::isFunction(job->onError.get())) {
                ev::Persistent msg(ev::fromUtf8(error.empty() ? "createPrompt cancelled" : error));
                callCallback1(job->onError.get(), msg.get());
            }
            return;
        }
        ev::Persistent out(promptToJs(job->out));
        callCallback1(job->onDone.get(), out.get());
    };
    return launchAsyncJob(std::move(work), nullptr, std::move(done));
}

// omni.savePrompt(prompt, path) / omni.loadPrompt(path) -> prompt  (.ovcp)
Value omniSavePrompt(Value self, std::span<const Value> a) {
    auto* w = omniSelf(self);
    if (!w) return ev::throwTypeError("savePrompt: not an OmniVoice");
    if (!hasArg(a, 0) || !isStringArg(a, 1))
        return ev::throwTypeError("savePrompt(prompt, path): prompt and path required");
    brosoundml::OmniVoicePrompt p;
    std::string err;
    const int nq = w->model && w->model->loaded() ? w->model->config().lm.num_codebooks : 8;
    if (!promptFromJs(a[0], nq, p, err)) return ev::throwTypeError("savePrompt: " + err);
    try {
        p.save(resolvePath(strAt(a, 1)));
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("savePrompt: ") + e.what());
    }
}

Value omniLoadPrompt(Value self, std::span<const Value> a) {
    auto* w = omniSelf(self);
    if (!w) return ev::throwTypeError("loadPrompt: not an OmniVoice");
    if (!isStringArg(a, 0)) return ev::throwTypeError("loadPrompt(path): path string required");
    try {
        return promptToJs(brosoundml::OmniVoicePrompt::load(resolvePath(strAt(a, 0))));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadPrompt: ") + e.what());
    }
}

// omni.estimateFrames(text, opts?) -> number  (opts.prompt / speed / duration)
Value omniEstimateFrames(Value self, std::span<const Value> a) {
    auto* w = omniSelf(self);
    if (!w) return ev::throwTypeError("estimateFrames: not an OmniVoice");
    if (!isStringArg(a, 0)) return ev::throwTypeError("estimateFrames(text, opts?): text string required");
    if (!w->model || !w->model->loaded()) return ev::throwError("estimateFrames: model is not loaded");
    const std::string text = strAt(a, 0);
    brosoundml::OmniVoiceParams params;
    brosoundml::OmniVoicePrompt prompt;
    int hasPrompt = 0;
    if (isObjectArg(a, 1)) {
        ev::Persistent opts(a[1]);
        readOmniParams(opts.get(), params);
        std::string err;
        hasPrompt = readOmniPrompt(opts.get(), w->model->config().lm.num_codebooks, prompt, err);
        if (hasPrompt < 0) return ev::throwTypeError("estimateFrames: " + err);
    }
    try {
        return ev::fromDouble(w->model->estimate_frames(text, params, hasPrompt ? &prompt : nullptr));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("estimateFrames: ") + e.what());
    }
}

Value omniTokenize(Value self, std::span<const Value> a) {
    auto* w = omniSelf(self);
    if (!w) return ev::throwTypeError("tokenize: not an OmniVoice");
    if (!isStringArg(a, 0)) return ev::throwTypeError("tokenize(text): text string required");
    if (!w->model || !w->model->loaded()) return ev::throwError("tokenize: model is not loaded");
    try {
        return makeInt32Array(w->model->tokenize(strAt(a, 0)));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("tokenize: ") + e.what());
    }
}

Value omniLanguages(Value self, std::span<const Value>) {
    auto* w = omniSelf(self);
    if (!w) return ev::throwTypeError("languages: not an OmniVoice");
    if (!w->model || !w->model->loaded()) return ev::throwError("languages: model is not loaded");
    return makeStringArray(w->model->languages());
}

// omni.instructAttributes() -> [{ name, values: string[] }]
Value omniInstructAttributes(Value self, std::span<const Value>) {
    auto* w = omniSelf(self);
    if (!w) return ev::throwTypeError("instructAttributes: not an OmniVoice");
    if (!w->model || !w->model->loaded()) return ev::throwError("instructAttributes: model is not loaded");
    const auto cats = w->model->instruct_attributes();
    return hostArrayOf(cats.size(), [&](size_t i) -> Value {
        ObjectBuilder o;
        o.set("name", cats[i].name);
        ev::Persistent vals(makeStringArray(cats[i].values));
        o.set("values", vals.get());
        return o.get();
    });
}

Value omniNonverbalTags(Value self, std::span<const Value>) {
    auto* w = omniSelf(self);
    if (!w) return ev::throwTypeError("nonverbalTags: not an OmniVoice");
    if (!w->model || !w->model->loaded()) return ev::throwError("nonverbalTags: model is not loaded");
    return makeStringArray(w->model->nonverbal_tags());
}

// omni.unload() — drop the weights (GPU memory included). Refused while an
// op is in flight; afterwards `loaded` is false and every method throws.
Value omniUnload(Value self, std::span<const Value>) {
    auto* w = omniSelf(self);
    if (!w) return ev::throwTypeError("unload: not an OmniVoice");
    if (w->busy.isBusy()) return ev::throwError("unload: an operation is in flight on this model");
    try {
        brotensor::DeviceScope scope(w->device);
        w->model = std::make_shared<brosoundml::OmniVoice>();   // an unloaded pipeline
    } catch (const std::exception& e) {
        return ev::throwError(std::string("unload: ") + e.what());
    }
    return ev::undefined();
}

// ── Async synthesize / generateCodes ─────────────────────────────────────────
//
// One launcher serves both. opts.onStep(step) observes every diffusion step
// on the JS thread: the C++ callback fires on the worker thread with pointers
// valid only for the call, so it copies the grid + scores + confidence into
// the next pre-sized slot; the JS-thread poll drains them. A chunked
// synthesize restarts `step` at 0 per chunk, so the slot count is
// numSteps * kOmniMaxChunks + 1 (256 chunks is over an hour of long-form
// text) and the worker drops a step rather than overrun.
constexpr size_t kOmniMaxChunks = 256;

struct OmniStepSlot {
    int step = 0, numSteps = 0, numFrames = 0, numCodebooks = 0, unmasked = 0;
    int chunk = 0, numChunks = 1;
    std::vector<int32_t> tokens;
    std::vector<float> scores;
    std::vector<float> confidence;
};

struct OmniJob {
    std::shared_ptr<const brosoundml::OmniVoice> model;
    brotensor::Device device = brotensor::Device::CPU;
    std::string text;
    brosoundml::OmniVoiceParams params;
    bool hasPrompt = false;
    brosoundml::OmniVoicePrompt prompt;
    bool codesMode = false;
    int frames = 0;                       // <= 0 -> estimate
    bool hasInit = false;
    brosoundml::OmniVoiceInit init;
    bool wantTrace = false;
    brosoundml::OmniVoiceTrace trace;
    std::vector<float> samples;
    int sampleRate = 24000;
    std::vector<int32_t> codes;
    int numFrames = 0;
    int numCodebooks = 8;
    ModelGate gate;
    ev::Persistent onDone, onError, onStep, modelRef;
    bool hasOnStep = false;
    SpscSlots<OmniStepSlot> steps;
};

Value stepToJs(OmniStepSlot& s) {
    ObjectBuilder st;
    st.set("step", ev::fromDouble(s.step));
    st.set("numSteps", ev::fromDouble(s.numSteps));
    st.set("chunk", ev::fromDouble(s.chunk));
    st.set("numChunks", ev::fromDouble(s.numChunks));
    st.set("numFrames", ev::fromDouble(s.numFrames));
    st.set("numCodebooks", ev::fromDouble(s.numCodebooks));
    st.set("unmasked", ev::fromDouble(s.unmasked));
    {
        ev::Persistent t(makeInt32Array(s.tokens));
        st.set("tokens", t.get());
    }
    {
        ev::Persistent v(makeFloat32Array(s.scores));
        st.set("scores", v.get());
    }
    {
        ev::Persistent v(makeFloat32Array(s.confidence));
        st.set("confidence", v.get());
    }
    return st.get();
}

}  // namespace

Value omniVoiceLaunch(Value modelVal, Value textVal, Value optsVal, bool codesMode) {
    const std::string fn = codesMode ? "generateCodes" : "synthesize";
    auto* w = omniSelf(modelVal);
    if (!w) return ev::throwTypeError(fn + ": not an OmniVoice");
    if (!w->model || !w->model->loaded()) return ev::throwError(fn + ": model is not loaded");
    if (!ev::isString(textVal) || ev::toUtf8(textVal).empty())
        return ev::throwTypeError(fn + "(text, opts?): non-empty text string required");

    auto job = std::make_shared<OmniJob>();
    job->model = w->model;
    job->device = w->device;
    job->text = ev::toUtf8(textVal);
    job->codesMode = codesMode;
    job->numCodebooks = w->model->config().lm.num_codebooks;
    job->gate = w->busy;
    job->modelRef = ev::Persistent(modelVal);

    ev::Persistent opts(ev::isObject(optsVal) ? optsVal : ev::undefined());
    if (ev::isObject(opts.get())) {
        readOmniParams(opts.get(), job->params);
        std::string err;
        const int pr = readOmniPrompt(opts.get(), job->numCodebooks, job->prompt, err);
        if (pr < 0) return ev::throwTypeError(fn + ": " + err);
        job->hasPrompt = pr > 0;
        job->wantTrace = getPropertyBool(opts.get(), "trace");
        if (codesMode) {
            getIntOpt(opts.get(), "frames", job->frames);
            if (!readOmniInit(opts.get(), job->init, job->hasInit, err))
                return ev::throwTypeError(fn + ": " + err);
        }
        job->onDone = getFunctionOpt(opts.get(), "onDone");
        job->onError = getFunctionOpt(opts.get(), "onError");
        job->onStep = getFunctionOpt(opts.get(), "onStep");
    }
    if (job->params.num_steps < 1) return ev::throwTypeError(fn + ": numSteps must be >= 1");

    if (!w->busy.tryClaim())
        return ev::throwError(fn + ": an operation is already in flight on this model");

    job->hasOnStep = ev::isFunction(job->onStep.get());
    if (job->hasOnStep)
        job->steps.reserve(static_cast<size_t>(job->params.num_steps) * (codesMode ? 1 : kOmniMaxChunks) + 1);

    auto work = [job](const std::atomic<bool>& cancel) {
        brotensor::DeviceScope scope(job->device);
        brosoundml::CancelCheck cancelFn = cancelCheckOf(cancel);
        brosoundml::OmniVoiceStepFn stepFn;
        if (job->hasOnStep) {
            stepFn = [job](const brosoundml::OmniVoiceStep& s) {
                job->steps.emplace([&](OmniStepSlot& slot) {
                    slot.step = s.step; slot.numSteps = s.num_steps;
                    slot.chunk = s.chunk; slot.numChunks = s.num_chunks;
                    slot.numFrames = s.num_frames; slot.numCodebooks = s.num_codebooks;
                    slot.unmasked = s.unmasked;
                    const size_t n = static_cast<size_t>(s.num_frames) * static_cast<size_t>(s.num_codebooks);
                    if (s.tokens) slot.tokens.assign(s.tokens, s.tokens + n);
                    if (s.scores) slot.scores.assign(s.scores, s.scores + n);
                    if (s.confidence) slot.confidence.assign(s.confidence, s.confidence + n);
                });
            };
        }
        const brosoundml::OmniVoicePrompt* prompt = job->hasPrompt ? &job->prompt : nullptr;
        brosoundml::OmniVoiceTrace* trace = job->wantTrace ? &job->trace : nullptr;
        if (job->codesMode) {
            job->codes = job->model->generate_codes(job->text, job->frames, job->params, prompt,
                                                    job->hasInit ? &job->init : nullptr, cancelFn,
                                                    trace, stepFn);
            job->numFrames = job->numCodebooks > 0
                ? static_cast<int>(job->codes.size() / static_cast<size_t>(job->numCodebooks)) : 0;
        } else {
            auto buf = job->model->synthesize(job->text, job->params, prompt, cancelFn, trace, stepFn);
            job->samples = std::move(buf.samples);
            job->sampleRate = buf.sample_rate;
        }
    };

    auto poll = [job] {
        if (!job->hasOnStep) return;
        job->steps.drain([&](OmniStepSlot& slot) {
            ev::Persistent st(stepToJs(slot));
            callCallback1(job->onStep.get(), st.get());
            std::vector<int32_t>().swap(slot.tokens);   // release the slot's memory
            std::vector<float>().swap(slot.scores);
            std::vector<float>().swap(slot.confidence);
        });
    };

    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();   // before the callbacks, so one may start the next op
        if (!error.empty() && ev::isFunction(job->onError.get())) {
            ev::Persistent msg(ev::fromUtf8(error));
            callCallback1(job->onError.get(), msg.get());
        }
        if (!ev::isFunction(job->onDone.get())) return;
        ObjectBuilder result;
        if (job->codesMode) {
            ev::Persistent c(makeInt32Array(job->codes));
            result.set("codes", c.get());
            result.set("numFrames", ev::fromDouble(job->numFrames));
            result.set("numCodebooks", ev::fromDouble(job->numCodebooks));
        } else {
            ev::Persistent s(makeFloat32Array(job->samples));
            result.set("samples", s.get());
            result.set("sampleRate", ev::fromDouble(job->sampleRate));
        }
        if (job->wantTrace && !cancelled && error.empty()) {
            ev::Persistent t(traceToJs(job->trace));
            result.set("trace", t.get());
        }
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), result.get(), info.get());
    };

    return launchAsyncJob(std::move(work), std::move(poll), std::move(done));
}

namespace {

void decorateOmniVoice(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = omniSelf(self);
        return ev::fromBool(w && w->model && w->model->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = omniSelf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("precision", [](Value self, std::span<const Value>) -> Value {
        auto* w = omniSelf(self);
        return ev::fromUtf8(w ? precisionName(w->precision) : "fp32");
    });
    b.accessor("sampleRate", [](Value self, std::span<const Value>) -> Value {
        auto* w = omniSelf(self);
        return ev::fromDouble(w && w->model && w->model->loaded() ? w->model->config().sample_rate : 0);
    });
    b.accessor("config", [](Value self, std::span<const Value>) -> Value {
        auto* w = omniSelf(self);
        if (!w || !w->model || !w->model->loaded()) return ev::null();
        return configToJs(w);
    });
    b.def("synthesize", 2, [](Value self, std::span<const Value> a) -> Value {
        if (!hasArg(a, 0)) return ev::throwTypeError("synthesize(text, opts?): text string required");
        return omniVoiceLaunch(self, a[0], hasArg(a, 1) ? a[1] : ev::undefined(), false);
    });
    b.def("generateCodes", 2, [](Value self, std::span<const Value> a) -> Value {
        if (!hasArg(a, 0)) return ev::throwTypeError("generateCodes(text, opts?): text string required");
        return omniVoiceLaunch(self, a[0], hasArg(a, 1) ? a[1] : ev::undefined(), true);
    });
    b.def("decodeCodes", 2, omniDecodeCodes);
    b.def("encodeAudio", 2, omniEncodeAudio);
    b.def("createPrompt", 2, omniCreatePrompt);
    b.def("savePrompt", 2, omniSavePrompt);
    b.def("loadPrompt", 1, omniLoadPrompt);
    b.def("estimateFrames", 2, omniEstimateFrames);
    b.def("tokenize", 1, omniTokenize);
    b.def("languages", 0, omniLanguages);
    b.def("instructAttributes", 0, omniInstructAttributes);
    b.def("nonverbalTags", 0, omniNonverbalTags);
    b.def("unload", 0, omniUnload);
}

}  // namespace

void installTtsOmniVoiceClasses() {
    g_omniVoiceClass.install("OmniVoice", 0, nullptr, decorateOmniVoice);
}

} // namespace brosoundml::api
