// bro.tts — Qwen3-TTS (text-driven: CustomVoice presets, VoiceDesign
// instructs, Base voice clones from an x-vector) with its sessions and the
// streaming AR tail, plus the standalone ECAPA-TDNN SpeakerEncoder. Loaders
// live in native_soundml_tts.cpp.
#include "soundml_tts_internal.h"

#include <cstdlib>

namespace brosoundml::api {

namespace {

HostQwenTts* qwenSelf(Value self) {
    return static_cast<HostQwenTts*>(g_qwenTtsClass.unwrap(self));
}
HostQwenTtsSession* sessionSelf(Value self) {
    return static_cast<HostQwenTtsSession*>(g_qwenTtsSessionClass.unwrap(self));
}
HostSpeakerEncoder* encSelf(Value self) {
    return static_cast<HostSpeakerEncoder*>(g_speakerEncoderClass.unwrap(self));
}

const char* variantName(brosoundml::QwenTtsVariant v) {
    switch (v) {
        case brosoundml::QwenTtsVariant::Base: return "base";
        case brosoundml::QwenTtsVariant::CustomVoice: return "custom_voice";
        case brosoundml::QwenTtsVariant::VoiceDesign: return "voice_design";
    }
    return "base";
}

// opts.speaker / language / instruct (all optional strings).
struct QwenSynthOpts {
    std::string speaker;
    std::string language = "english";
    std::string instruct;
    std::vector<float> xvector;          // Base designer: replaces the preset
    brosoundml::QwenTtsSampling sampling;
    bool wantTrace = false;
};

void readQwenOpts(Value opts, QwenSynthOpts& o) {
    if (!ev::isObject(opts)) return;
    ev::Persistent root(opts);
    getStrOpt(root.get(), "speaker", o.speaker);
    getStrOpt(root.get(), "language", o.language);
    getStrOpt(root.get(), "instruct", o.instruct);
    o.wantTrace = getPropertyBool(root.get(), "trace");
    auto& s = o.sampling;
    getFloatOpt(root.get(), "temperature", s.temperature);
    getFloatOpt(root.get(), "topP", s.top_p);
    getFloatOpt(root.get(), "repetitionPenalty", s.repetition_penalty);
    getFloatOpt(root.get(), "adaptive", s.adaptive);
    getIntOpt(root.get(), "topK", s.top_k);
    getSeedOpt(root.get(), s.seed);
    // logitBias: { codeId: delta, ... } additive codebook-0 bias.
    {
        Value lb = ev::getProperty(root.get(), "logitBias");
        if (ev::isObject(lb) && !ev::isFunction(lb)) {
            ev::Persistent lbRoot(lb);
            for (const std::string& key : objectKeys(lbRoot.get())) {
                Value v = ev::getProperty(lbRoot.get(), key);
                if (ev::isNumber(v))
                    s.logit_bias.emplace_back(std::atoi(key.c_str()), static_cast<float>(ev::toDouble(v)));
            }
        }
    }
    auto readVec = [&](const char* key, std::vector<float>& dst) {
        Value v = ev::getProperty(root.get(), key);
        if (!ev::isUndefined(v) && !ev::isNull(v)) dst = readFloat32Array(v);
    };
    readVec("voiceSteer", s.voice_steer);
    readVec("speakerVector", s.speaker_vector);
    readVec("xvector", o.xvector);
}

Value makeQwenResult(const std::vector<float>& samples, int rate, const brosoundml::QwenTtsTrace* trace) {
    ev::Persistent out(makeAudioResult(samples, rate));
    if (trace) {
        ev::Persistent st(makeStagesArray(trace->stages));
        ev::setProperty(out.get(), "stages", st.get());
    }
    return out.get();
}

// qwen.synthesize(text, opts?) -> { samples, sampleRate, stages? }  (sync)
Value qwenSynthesizeSync(Value self, std::span<const Value> a) {
    auto* w = qwenSelf(self);
    if (!w) return ev::throwTypeError("synthesize: not a QwenTts");
    if (!isStringArg(a, 0)) return ev::throwTypeError("synthesize(text, opts?): text string required");
    if (!w->model || !w->model->loaded()) return ev::throwError("synthesize: model is not loaded");
    QwenSynthOpts o;
    if (isObjectArg(a, 1)) readQwenOpts(a[1], o);
    if (w->busy.isBusy()) return ev::throwError("synthesize: an operation is already in flight on this model");
    try {
        brotensor::DeviceScope scope(w->device);
        brosoundml::QwenTtsTrace trace;
        brosoundml::AudioBuffer buf;
        if (!o.xvector.empty())
            buf = w->model->synthesize_with_xvector(strAt(a, 0), o.xvector, o.language, {}, o.sampling,
                                                    o.wantTrace ? &trace : nullptr);
        else
            buf = w->model->synthesize(strAt(a, 0), o.speaker, o.language, o.instruct, {}, o.sampling,
                                       o.wantTrace ? &trace : nullptr);
        return makeQwenResult(buf.samples, buf.sample_rate, o.wantTrace ? &trace : nullptr);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("synthesize: ") + e.what());
    }
}

// qwen.synthesizeClone(text, ref, opts?) -> { samples, sampleRate }  (sync, Base)
//   ref: a WAV path, a Float32Array (opts.sampleRate) or { samples, sampleRate }.
Value qwenSynthesizeClone(Value self, std::span<const Value> a) {
    auto* w = qwenSelf(self);
    if (!w) return ev::throwTypeError("synthesizeClone: not a QwenTts");
    if (!isStringArg(a, 0)) return ev::throwTypeError("synthesizeClone(text, ref, opts?): text string required");
    if (!hasArg(a, 1)) return ev::throwTypeError("synthesizeClone(text, ref, opts?): ref (WAV path or samples) required");
    if (!w->model || !w->model->loaded()) return ev::throwError("synthesizeClone: model is not loaded");
    QwenSynthOpts o;
    if (isObjectArg(a, 2)) readQwenOpts(a[2], o);
    if (w->busy.isBusy()) return ev::throwError("synthesizeClone: an operation is already in flight on this model");
    try {
        brotensor::DeviceScope scope(w->device);
        brosoundml::AudioBuffer ref;
        if (ev::isString(a[1])) {
            ref = brosoundml::read_wav(resolvePath(ev::toUtf8(a[1])));
        } else {
            std::string err;
            if (!readAudioBuffer(a[1], ref, err, 24000)) return ev::throwTypeError("synthesizeClone: " + err);
            if (isObjectArg(a, 2)) getIntOpt(a[2], "sampleRate", ref.sample_rate);
        }
        auto buf = w->model->synthesize_clone(strAt(a, 0), ref, o.language, {}, o.sampling);
        return makeAudioResult(buf.samples, buf.sample_rate);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("synthesizeClone: ") + e.what());
    }
}

// qwen.synthesizeFromXvector(text, xvec, opts?) -> { samples, sampleRate, stages? }
Value qwenSynthesizeFromXvector(Value self, std::span<const Value> a) {
    auto* w = qwenSelf(self);
    if (!w) return ev::throwTypeError("synthesizeFromXvector: not a QwenTts");
    if (!isStringArg(a, 0)) return ev::throwTypeError("synthesizeFromXvector(text, xvec, opts?): text string required");
    std::vector<float> xvec = hasArg(a, 1) ? readFloat32Array(a[1]) : std::vector<float>{};
    if (xvec.empty()) return ev::throwTypeError("synthesizeFromXvector: xvec must be a non-empty Float32Array");
    if (!w->model || !w->model->loaded()) return ev::throwError("synthesizeFromXvector: model is not loaded");
    QwenSynthOpts o;
    if (isObjectArg(a, 2)) readQwenOpts(a[2], o);
    if (w->busy.isBusy()) return ev::throwError("synthesizeFromXvector: an operation is already in flight on this model");
    try {
        brotensor::DeviceScope scope(w->device);
        brosoundml::QwenTtsTrace trace;
        auto buf = w->model->synthesize_with_xvector(strAt(a, 0), xvec, o.language, {}, o.sampling,
                                                     o.wantTrace ? &trace : nullptr);
        return makeQwenResult(buf.samples, buf.sample_rate, o.wantTrace ? &trace : nullptr);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("synthesizeFromXvector: ") + e.what());
    }
}

// qwen.decodeCodes(codes, numQuantizers?, numFrames?) -> { samples, sampleRate }
//   codes are codebook-major (codes[k*numFrames + t]); numQuantizers defaults
//   to the codec's and numFrames to codes.length / numQuantizers.
Value qwenDecodeCodes(Value self, std::span<const Value> a) {
    auto* w = qwenSelf(self);
    if (!w) return ev::throwTypeError("decodeCodes: not a QwenTts");
    if (!hasArg(a, 0)) return ev::throwTypeError("decodeCodes(codes, numQuantizers?, numFrames?): codes required");
    std::vector<int32_t> codes = readInt32Array(a[0]);
    if (codes.empty()) return ev::throwTypeError("decodeCodes: codes must be a non-empty Int32Array");
    if (!w->model || !w->model->loaded()) return ev::throwError("decodeCodes: model is not loaded");
    int nq = hasArg(a, 1) && ev::isNumber(a[1]) ? i32At(a, 1) : w->model->config().codec.num_quantizers;
    int nf = hasArg(a, 2) && ev::isNumber(a[2]) ? i32At(a, 2) : 0;
    if (nq > 0 && nf <= 0) nf = static_cast<int>(codes.size() / static_cast<size_t>(nq));
    if (nq <= 0 || nf <= 0 || static_cast<size_t>(nq) * static_cast<size_t>(nf) != codes.size())
        return ev::throwTypeError("decodeCodes: numQuantizers*numFrames must equal codes.length");
    if (w->busy.isBusy()) return ev::throwError("decodeCodes: an operation is already in flight on this model");
    try {
        brotensor::DeviceScope scope(w->device);
        return audioBufferToJs(w->model->decode_codes(codes, nq, nf));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("decodeCodes: ") + e.what());
    }
}

// qwen.encodeAudio(audio, opts?) -> { codes, numQuantizers, numFrames }
Value qwenEncodeAudio(Value self, std::span<const Value> a) {
    auto* w = qwenSelf(self);
    if (!w) return ev::throwTypeError("encodeAudio: not a QwenTts");
    brosoundml::AudioBuffer ref;
    if (!readAudioArgs(a, 0, ref)) return ev::throwTypeError("encodeAudio(audio, opts?): audio must be a non-empty Float32Array");
    if (!w->model || !w->model->loaded()) return ev::throwError("encodeAudio: model is not loaded");
    if (w->busy.isBusy()) return ev::throwError("encodeAudio: an operation is already in flight on this model");
    try {
        brotensor::DeviceScope scope(w->device);
        int nf = 0;
        std::vector<int32_t> codes = w->model->encode_audio(ref, &nf);
        const int nq = nf > 0 ? static_cast<int>(codes.size() / static_cast<size_t>(nf)) : 0;
        ObjectBuilder obj;
        {
            ev::Persistent c(makeInt32Array(codes));
            obj.set("codes", c.get());
        }
        obj.set("numQuantizers", ev::fromDouble(nq));
        obj.set("numFrames", ev::fromDouble(nf));
        return obj.get();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("encodeAudio: ") + e.what());
    }
}

// qwen.embedSpeaker(audio, opts?) -> Float32Array(encDim)  (sync, Base only)
Value qwenEmbedSpeaker(Value self, std::span<const Value> a) {
    auto* w = qwenSelf(self);
    if (!w) return ev::throwTypeError("embedSpeaker: not a QwenTts");
    brosoundml::AudioBuffer ref;
    if (!readAudioArgs(a, 0, ref)) return ev::throwTypeError("embedSpeaker(audio, opts?): audio must be a non-empty Float32Array");
    if (!w->model || !w->model->loaded()) return ev::throwError("embedSpeaker: model is not loaded");
    if (w->busy.isBusy()) return ev::throwError("embedSpeaker: an operation is already in flight on this model");
    try {
        brotensor::DeviceScope scope(w->device);
        return makeFloat32Array(w->model->embed_speaker(ref));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("embedSpeaker: ") + e.what());
    }
}

// The async job behind bro.tts.synthesize(qwen, ...) and session.synthesize.
struct QwenJob {
    std::string text;
    QwenSynthOpts o;
    brosoundml::QwenTtsTrace trace;
    std::vector<float> samples;
    int sampleRate = 24000;
    ModelGate gate;
    brotensor::Device device = brotensor::Device::CPU;
    std::shared_ptr<brosoundml::QwenTts> model;
    brosoundml::QwenTtsSession* session = nullptr;   // owned by the rooted handle
    ev::Persistent onDone, modelRef;
};

Value launchQwenJob(std::shared_ptr<QwenJob> job) {
    auto work = [job](const std::atomic<bool>& cancel) {
        brotensor::DeviceScope scope(job->device);
        auto cancelFn = cancelCheckOf(cancel);
        auto* trace = job->o.wantTrace ? &job->trace : nullptr;
        brosoundml::AudioBuffer buf;
        if (job->session) {
            if (!job->o.xvector.empty())
                buf = job->model->synthesize_with_xvector(*job->session, job->text, job->o.xvector,
                                                          job->o.language, cancelFn, job->o.sampling, trace);
            else
                buf = job->model->synthesize(*job->session, job->text, job->o.speaker, job->o.language,
                                             job->o.instruct, cancelFn, job->o.sampling, trace);
        } else if (!job->o.xvector.empty()) {
            buf = job->model->synthesize_with_xvector(job->text, job->o.xvector, job->o.language, cancelFn,
                                                      job->o.sampling, trace);
        } else {
            buf = job->model->synthesize(job->text, job->o.speaker, job->o.language, job->o.instruct,
                                         cancelFn, job->o.sampling, trace);
        }
        job->samples = std::move(buf.samples);
        job->sampleRate = buf.sample_rate;
    };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        if (!ev::isFunction(job->onDone.get())) return;
        const bool trace = job->o.wantTrace && !cancelled && error.empty();
        ev::Persistent result(makeQwenResult(job->samples, job->sampleRate, trace ? &job->trace : nullptr));
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), result.get(), info.get());
    };
    return launchAsyncJob(std::move(work), nullptr, std::move(done));
}

std::shared_ptr<QwenJob> makeQwenJob(std::span<const Value> a, size_t textAt, const char* fn) {
    if (!isStringArg(a, textAt)) {
        ev::throwTypeError(std::string(fn) + "(text, opts?): text string required");
        return nullptr;
    }
    auto job = std::make_shared<QwenJob>();
    job->text = strAt(a, textAt);
    if (isObjectArg(a, textAt + 1)) {
        ev::Persistent opts(a[textAt + 1]);
        readQwenOpts(opts.get(), job->o);
        job->onDone = getFunctionOpt(opts.get(), "onDone");
    }
    return job;
}

// session.synthesize(text, opts?) -> AsyncHandle
Value qwenSessionSynthesize(Value self, std::span<const Value> a) {
    auto* sw = sessionSelf(self);
    if (!sw) return ev::throwTypeError("synthesize: not a QwenTtsSession");
    auto job = makeQwenJob(a, 0, "synthesize");
    if (!job) return ev::undefined();
    if (!sw->busy.tryClaim()) return ev::throwError("synthesize: an operation is already in flight on this model");
    job->gate = sw->busy;
    job->device = sw->device;
    job->model = sw->model;
    job->session = &sw->session;
    job->modelRef = ev::Persistent(self);
    return launchQwenJob(std::move(job));
}

}  // namespace

// bro.tts.synthesize(qwen, text, opts?) -> AsyncHandle
Value qwenTtsSynthesizeAsync(Value modelVal, HostQwenTts* w, std::span<const Value> args) {
    auto job = makeQwenJob(args, 1, "synthesize");
    if (!job) return ev::undefined();
    if (!w->busy.tryClaim()) return ev::throwError("synthesize: an operation is already in flight on this model");
    job->gate = w->busy;
    job->device = w->device;
    job->model = w->model;
    job->modelRef = ev::Persistent(modelVal);
    return launchQwenJob(std::move(job));
}

// bro.tts.synthesizeStream(qwen, text, opts?) -> AsyncHandle
//   opts.onChunk(Float32Array) per decoded codec chunk (opts.chunkFrames,
//   default 25 = ~2 s), then onDone with the whole buffer.
Value qwenTtsSynthesizeStream(Value modelVal, HostQwenTts* w, std::span<const Value> args) {
    if (!isStringArg(args, 1)) return ev::throwTypeError("synthesizeStream(qwen, text, opts?): text string required");
    struct StreamJob {
        std::string text;
        QwenSynthOpts o;
        int chunkFrames = 25;
        std::vector<float> samples;
        int sampleRate = 24000;
        ModelGate gate;
        brotensor::Device device = brotensor::Device::CPU;
        std::shared_ptr<brosoundml::QwenTts> model;
        ev::Persistent onChunk, onDone, modelRef;
        bool hasOnChunk = false;
        SpscSlots<std::vector<float>> chunks;
    };
    auto job = std::make_shared<StreamJob>();
    job->text = strAt(args, 1);
    if (isObjectArg(args, 2)) {
        ev::Persistent opts(args[2]);
        readQwenOpts(opts.get(), job->o);
        int cf = job->chunkFrames;
        getIntOpt(opts.get(), "chunkFrames", cf);
        if (cf > 0) job->chunkFrames = cf;
        job->onChunk = getFunctionOpt(opts.get(), "onChunk");
        job->onDone = getFunctionOpt(opts.get(), "onDone");
    }
    job->hasOnChunk = ev::isFunction(job->onChunk.get());
    // At most (max_frames / chunkFrames) + 1 chunks; the generator caps at
    // 4096 frames regardless of chunkFrames.
    job->chunks.reserve(4098);
    if (!w->busy.tryClaim()) return ev::throwError("synthesizeStream: an operation is already in flight on this model");
    job->gate = w->busy;
    job->device = w->device;
    job->model = w->model;
    job->modelRef = ev::Persistent(modelVal);

    auto work = [job](const std::atomic<bool>& cancel) {
        brotensor::DeviceScope scope(job->device);
        auto onChunkCb = [job](const float* s, int n) {
            job->chunks.emplace([&](std::vector<float>& slot) { slot.assign(s, s + n); });
        };
        auto cancelFn = cancelCheckOf(cancel);
        brosoundml::AudioBuffer buf;
        if (!job->o.xvector.empty())
            buf = job->model->synthesize_stream_with_xvector(job->text, job->o.xvector, job->chunkFrames,
                                                             onChunkCb, job->o.language, cancelFn,
                                                             job->o.sampling);
        else
            buf = job->model->synthesize_stream(job->text, job->o.speaker, job->chunkFrames, onChunkCb,
                                                job->o.language, job->o.instruct, cancelFn, job->o.sampling);
        job->samples = std::move(buf.samples);
        job->sampleRate = buf.sample_rate;
    };
    auto poll = [job] {
        if (!job->hasOnChunk) return;
        job->chunks.drain([&](std::vector<float>& chunk) {
            ev::Persistent arr(makeFloat32Array(chunk));
            callCallback1(job->onChunk.get(), arr.get());
            std::vector<float>().swap(chunk);
        });
    };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        if (!ev::isFunction(job->onDone.get())) return;
        ev::Persistent result(makeAudioResult(job->samples, job->sampleRate));
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), result.get(), info.get());
    };
    return launchAsyncJob(std::move(work), std::move(poll), std::move(done));
}

namespace {

void decorateQwenTts(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = qwenSelf(self);
        return ev::fromBool(w && w->model && w->model->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = qwenSelf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("sampleRate", [](Value self, std::span<const Value>) -> Value {
        auto* w = qwenSelf(self);
        return ev::fromDouble(w && w->model ? w->model->config().sample_rate : 0);
    });
    b.accessor("variant", [](Value self, std::span<const Value>) -> Value {
        auto* w = qwenSelf(self);
        return ev::fromUtf8(w && w->model && w->model->loaded() ? variantName(w->model->config().variant) : "");
    });
    b.accessor("modelSize", [](Value self, std::span<const Value>) -> Value {
        auto* w = qwenSelf(self);
        return ev::fromUtf8(w && w->model && w->model->loaded() ? w->model->config().model_size : "");
    });
    b.def("synthesize", 2, qwenSynthesizeSync);
    b.def("synthesizeClone", 3, qwenSynthesizeClone);
    b.def("synthesizeFromXvector", 3, qwenSynthesizeFromXvector);
    b.def("decodeCodes", 3, qwenDecodeCodes);
    b.def("encodeAudio", 2, qwenEncodeAudio);
    b.def("embedSpeaker", 2, qwenEmbedSpeaker);
    b.def("speakers", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = qwenSelf(self);
        if (!w) return ev::throwTypeError("speakers: not a QwenTts");
        if (!w->model || !w->model->loaded()) return ev::throwError("speakers: model is not loaded");
        return makeStringArray(w->model->speakers());
    });
    b.def("languages", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = qwenSelf(self);
        if (!w) return ev::throwTypeError("languages: not a QwenTts");
        if (!w->model || !w->model->loaded()) return ev::throwError("languages: model is not loaded");
        return makeStringArray(w->model->languages());
    });
    b.def("speakerDialect", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = qwenSelf(self);
        if (!w) return ev::throwTypeError("speakerDialect: not a QwenTts");
        if (!isStringArg(a, 0)) return ev::throwTypeError("speakerDialect(name): name string required");
        if (!w->model || !w->model->loaded()) return ev::throwError("speakerDialect: model is not loaded");
        return ev::fromUtf8(w->model->speaker_dialect(strAt(a, 0)));
    });
    b.def("createSession", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = qwenSelf(self);
        if (!w) return ev::throwTypeError("createSession: not a QwenTts");
        if (!w->model || !w->model->loaded()) return ev::throwError("createSession: model is not loaded");
        try {
            brotensor::DeviceScope scope(w->device);
            auto sw = std::make_unique<HostQwenTtsSession>();
            sw->model = w->model;
            sw->busy = w->busy;
            sw->device = w->device;
            sw->session = w->model->make_session();
            return g_qwenTtsSessionClass.createInstance(std::move(sw));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("createSession: ") + e.what());
        }
    });
}

void decorateQwenTtsSession(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* s = sessionSelf(self);
        return ev::fromBool(s && s->model && s->model->loaded());
    });
    b.accessor("variant", [](Value self, std::span<const Value>) -> Value {
        auto* s = sessionSelf(self);
        return ev::fromUtf8(s && s->model && s->model->loaded() ? variantName(s->model->config().variant) : "");
    });
    b.def("synthesize", 2, qwenSessionSynthesize);
    b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
        auto* s = sessionSelf(self);
        if (!s) return ev::throwTypeError("reset: not a QwenTtsSession");
        if (s->busy.isBusy()) return ev::throwError("reset: a synthesis is in flight on this model");
        try {
            brotensor::DeviceScope scope(s->device);
            s->model->reset(s->session);
        } catch (const std::exception& e) {
            return ev::throwError(std::string("reset: ") + e.what());
        }
        return ev::undefined();
    });
}

// enc.embedSpeaker(audio, opts?) -> Float32Array (sync) | AsyncHandle (opts.onDone)
Value speakerEmbed(Value self, std::span<const Value> a) {
    auto* w = encSelf(self);
    if (!w) return ev::throwTypeError("embedSpeaker: not a SpeakerEncoder");
    if (!w->enc || !w->enc->loaded()) return ev::throwError("embedSpeaker: encoder is not loaded");
    brosoundml::AudioBuffer ref;
    if (!readAudioArgs(a, 0, ref)) return ev::throwTypeError("embedSpeaker(audio, opts?): audio must be a non-empty Float32Array");
    ev::Persistent opts(isObjectArg(a, 1) ? a[1] : ev::undefined());
    ev::Persistent onDone = getFunctionOpt(opts.get(), "onDone");
    if (!ev::isFunction(onDone.get())) {
        try {
            return makeFloat32Array(w->enc->embed(ref));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("embedSpeaker: ") + e.what());
        }
    }
    struct State {
        std::shared_ptr<brosoundml::SpeakerEncoder> enc;
        brosoundml::AudioBuffer ref;
        std::vector<float> out;
        ev::Persistent self, onDone, onError;
    };
    auto st = std::make_shared<State>();
    st->enc = w->enc;
    st->ref = std::move(ref);
    st->self = ev::Persistent(self);
    st->onDone = onDone;
    st->onError = getFunctionOpt(opts.get(), "onError");
    auto work = [st](const std::atomic<bool>&) { st->out = st->enc->embed(st->ref); };
    auto done = [st](bool, const std::string& error) {
        if (!error.empty()) {
            if (ev::isFunction(st->onError.get())) {
                ev::Persistent msg(ev::fromUtf8(error));
                callCallback1(st->onError.get(), msg.get());
            }
            return;
        }
        ev::Persistent arr(makeFloat32Array(st->out));
        callCallback1(st->onDone.get(), arr.get());
    };
    return launchAsyncJob(std::move(work), nullptr, std::move(done));
}

void decorateSpeakerEncoder(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = encSelf(self);
        return ev::fromBool(w && w->enc && w->enc->loaded());
    });
    b.accessor("device", [](Value, std::span<const Value>) -> Value {
        return ev::fromUtf8(deviceName(brotensor::default_device()));
    });
    b.accessor("encDim", [](Value self, std::span<const Value>) -> Value {
        auto* w = encSelf(self);
        return ev::fromDouble(w && w->enc && w->enc->loaded() ? w->enc->enc_dim() : 0);
    });
    b.accessor("sampleRate", [](Value self, std::span<const Value>) -> Value {
        auto* w = encSelf(self);
        return ev::fromDouble(w && w->enc && w->enc->loaded() ? w->enc->sample_rate() : 0);
    });
    b.def("embedSpeaker", 2, speakerEmbed);
    b.def("embed", 2, speakerEmbed);
}

}  // namespace

void installTtsQwenClasses() {
    g_qwenTtsSessionClass.install("QwenTtsSession", 0, nullptr, decorateQwenTtsSession);
    g_qwenTtsClass.install("QwenTtsModel", 0, nullptr, decorateQwenTts);
    g_speakerEncoderClass.install("SpeakerEncoder", 0, nullptr, decorateSpeakerEncoder);
}

} // namespace brosoundml::api
