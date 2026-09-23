// bro.tts — Kokoro: the phoneme-driven TTS (Voice packs, sessions, the
// traced pipeline and the decodeFrom prosody-editing seam). The loader and
// the phonemizer live in native_soundml_tts.cpp.
#include "soundml_tts_internal.h"

namespace brosoundml::api {

namespace {

HostKokoro* kokoroSelf(Value self) {
    return static_cast<HostKokoro*>(g_kokoroClass.unwrap(self));
}
HostVoice* voiceOf(Value v) { return unwrapTtsAs<HostVoice>(g_voiceClass, v); }
HostKokoroSession* sessionSelf(Value self) {
    return static_cast<HostKokoroSession*>(g_kokoroSessionClass.unwrap(self));
}

Value makeVoice(brosoundml::Voice voice) {
    auto vw = std::make_unique<HostVoice>();
    vw->voice = std::move(voice);
    return g_voiceClass.createInstance(std::move(vw));
}

void decorateVoice(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = voiceOf(self);
        return ev::fromBool(w && w->voice.packs.rows > 0);
    });
    b.accessor("name", [](Value self, std::span<const Value>) -> Value {
        auto* w = voiceOf(self);
        return ev::fromUtf8(w ? w->voice.name : "");
    });
    b.accessor("rows", [](Value self, std::span<const Value>) -> Value {
        auto* w = voiceOf(self);
        return ev::fromDouble(w ? w->voice.packs.rows : 0);
    });
    b.accessor("cols", [](Value self, std::span<const Value>) -> Value {
        auto* w = voiceOf(self);
        return ev::fromDouble(w ? w->voice.packs.cols : 0);
    });
    // The full style table (rows*cols, row-major) for blending / perturbing
    // and feeding back through kokoro.createVoice().
    b.accessor("data", [](Value self, std::span<const Value>) -> Value {
        auto* w = voiceOf(self);
        if (!w || w->voice.packs.rows == 0) return makeFloat32Array(std::vector<float>{});
        return makeFloat32Array(w->voice.packs.to_host_vector());
    });
}

// { samples, sampleRate, durations, stages? }
Value makeKokoroResult(const std::vector<float>& samples, int rate, const std::vector<int32_t>& durations,
                       const brosoundml::KokoroTrace* trace) {
    ev::Persistent out(makeAudioResult(samples, rate));
    {
        ev::Persistent d(makeInt32Array(durations));
        ev::setProperty(out.get(), "durations", d.get());
    }
    if (trace) {
        ev::Persistent st(makeStagesArray(trace->stages));
        ev::setProperty(out.get(), "stages", st.get());
    }
    return out.get();
}

// loadVoice(path, opts?) -> Voice (sync) | AsyncHandle (opts.onReady)
Value kokoroLoadVoice(Value self, std::span<const Value> a) {
    ev::Persistent selfRoot(self);  // `this` is a plain copy; the reads below allocate
    auto* w = kokoroSelf(self);
    if (!w) return ev::throwTypeError("loadVoice: not a Kokoro");
    if (!isStringArg(a, 0)) return ev::throwTypeError("loadVoice(path, opts?): path string required");
    if (!w->model || !w->model->loaded()) return ev::throwError("loadVoice: model is not loaded");
    const std::string path = resolvePath(strAt(a, 0));
    ev::Persistent opts(isObjectArg(a, 1) ? a[1] : ev::undefined());
    ev::Persistent onReady = getFunctionOpt(opts.get(), "onReady");
    if (!ev::isFunction(onReady.get())) {
        try {
            brotensor::DeviceScope scope(w->device);
            return makeVoice(w->model->load_voice(path));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("loadVoice: ") + e.what());
        }
    }
    struct State {
        std::shared_ptr<brosoundml::Kokoro> model;
        brotensor::Device device;
        std::string path;
        brosoundml::Voice voice;
        ev::Persistent modelRef, onReady, onError;
    };
    auto st = std::make_shared<State>();
    st->model = w->model;
    st->device = w->device;
    st->path = path;
    st->modelRef = ev::Persistent(selfRoot.get());
    st->onReady = onReady;
    st->onError = getFunctionOpt(opts.get(), "onError");
    auto work = [st](const std::atomic<bool>&) {
        brotensor::DeviceScope scope(st->device);
        st->voice = st->model->load_voice(st->path);
    };
    auto done = [st](bool, const std::string& error) {
        if (!error.empty()) {
            if (ev::isFunction(st->onError.get())) {
                ev::Persistent msg(ev::fromUtf8(error));
                callCallback1(st->onError.get(), msg.get());
            }
            return;
        }
        ev::Persistent v(makeVoice(std::move(st->voice)));
        callCallback1(st->onReady.get(), v.get());
    };
    return launchAsyncJob(std::move(work), nullptr, std::move(done));
}

// createVoice(data, name?) -> Voice from raw style floats (voice_dim values
// broadcast across the rows, or a whole rows*voice_dim table).
Value kokoroCreateVoice(Value self, std::span<const Value> a) {
    auto* w = kokoroSelf(self);
    if (!w) return ev::throwTypeError("createVoice: not a Kokoro");
    if (!hasArg(a, 0)) return ev::throwTypeError("createVoice(data, name?): data required");
    std::vector<float> data = readFloat32Array(a[0]);
    if (data.empty()) return ev::throwTypeError("createVoice: data must be a non-empty Float32Array or number[]");
    const std::string name = isStringArg(a, 1) ? strAt(a, 1) : "custom";
    try {
        brotensor::DeviceScope scope(w->device);
        return makeVoice(w->model->make_voice(data, name));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("createVoice: ") + e.what());
    }
}

// synthesize(phonemeIds, voice, opts?) -> { samples, sampleRate, durations }
// synthesizeTraced(...)                -> + stages: [{ name, h, w, data }]
Value kokoroSynthesizeSync(Value self, std::span<const Value> a, bool traced) {
    const std::string fn = traced ? "synthesizeTraced" : "synthesize";
    auto* w = kokoroSelf(self);
    if (!w) return ev::throwTypeError(fn + ": not a Kokoro");
    if (a.size() < 2)
        return ev::throwTypeError(fn + "(phonemeIds, voice, opts?): phonemeIds and voice required");
    std::vector<int32_t> ids = readInt32Array(a[0]);
    if (ids.empty()) return ev::throwTypeError(fn + ": phonemeIds must be a non-empty Int32Array or number[]");
    auto* vw = voiceOf(a[1]);
    if (!vw) return ev::throwTypeError(fn + ": voice must be a Voice (returned by loadVoice)");
    float speed = 1.0f;
    if (isObjectArg(a, 2)) getFloatOpt(a[2], "speed", speed);
    if (w->busy.isBusy()) return ev::throwError(fn + ": an operation is already in flight on this model");
    try {
        brotensor::DeviceScope scope(w->device);
        std::vector<int32_t> durations;
        brosoundml::KokoroTrace trace;
        auto buf = w->model->synthesize(ids, vw->voice, speed, &durations, {}, traced ? &trace : nullptr);
        return makeKokoroResult(buf.samples, buf.sample_rate, durations, traced ? &trace : nullptr);
    } catch (const std::exception& e) {
        return ev::throwError(fn + ": " + e.what());
    }
}

// decodeFrom(voice, asr, F0, N, nPhonemes, opts?) -> { samples, sampleRate, stages? }
Value kokoroDecodeFromSync(Value self, std::span<const Value> a) {
    auto* w = kokoroSelf(self);
    if (!w) return ev::throwTypeError("decodeFrom: not a Kokoro");
    if (a.size() < 5)
        return ev::throwTypeError("decodeFrom(voice, asr, F0, N, nPhonemes, opts?): need voice, asr, F0, N, nPhonemes");
    auto* vw = voiceOf(a[0]);
    if (!vw) return ev::throwTypeError("decodeFrom: voice must be a Voice (from createVoice/loadVoice)");
    std::vector<float> asr = readFloat32Array(a[1]);
    std::vector<float> F0 = readFloat32Array(a[2]);
    std::vector<float> N = readFloat32Array(a[3]);
    if (asr.empty() || F0.empty() || N.empty())
        return ev::throwTypeError("decodeFrom: asr, F0 and N must be non-empty Float32Arrays");
    if (F0.size() % 2 != 0) return ev::throwTypeError("decodeFrom: F0 length must be even (2*total)");
    const int nph = i32At(a, 4);
    const int total = static_cast<int>(F0.size() / 2);
    const bool wantTrace = isObjectArg(a, 5) && getPropertyBool(a[5], "trace");
    if (w->busy.isBusy()) return ev::throwError("decodeFrom: an operation is already in flight on this model");
    try {
        brotensor::DeviceScope scope(w->device);
        brosoundml::KokoroTrace trace;
        auto buf = w->model->decode_from(vw->voice, nph, asr, total, F0, N, {}, wantTrace ? &trace : nullptr);
        ev::Persistent out(makeAudioResult(buf.samples, buf.sample_rate));
        if (wantTrace) {
            ev::Persistent st(makeStagesArray(trace.stages));
            ev::setProperty(out.get(), "stages", st.get());
        }
        return out.get();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("decodeFrom: ") + e.what());
    }
}

// vocab() -> { phoneme: id, ... }
Value kokoroVocab(Value self, std::span<const Value>) {
    auto* w = kokoroSelf(self);
    if (!w) return ev::throwTypeError("vocab: not a Kokoro");
    ObjectBuilder obj;
    for (const auto& kv : w->model->config().vocab) obj.set(kv.first, ev::fromDouble(kv.second));
    return obj.get();
}

// encodePhonemes(ipa) -> Int32Array (the codepoint -> id stage only)
Value kokoroEncodePhonemes(Value self, std::span<const Value> a) {
    auto* w = kokoroSelf(self);
    if (!w) return ev::throwTypeError("encodePhonemes: not a Kokoro");
    if (!isStringArg(a, 0)) return ev::throwTypeError("encodePhonemes(ipa): string required");
    try {
        if (!w->adapter)
            w->adapter = std::make_unique<brosoundml::g2p::PhonemeAdapter>(w->model->config().vocab);
        return makeInt32Array(w->adapter->encode(strAt(a, 0)));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("encodePhonemes: ") + e.what());
    }
}

// createSession(voice?) -> KokoroSession. The voice may also come later, on
// session.synthesize(ids, voice, opts) / setVoice(voice).
Value kokoroCreateSession(Value self, std::span<const Value> a) {
    auto* w = kokoroSelf(self);
    if (!w) return ev::throwTypeError("createSession: not a Kokoro");
    if (!w->model || !w->model->loaded()) return ev::throwError("createSession: model is not loaded");
    HostVoice* vw = hasArg(a, 0) && !ev::isUndefined(a[0]) ? voiceOf(a[0]) : nullptr;
    if (hasArg(a, 0) && !ev::isUndefined(a[0]) && !vw)
        return ev::throwTypeError("createSession(voice): voice must be a Voice (loadVoice/createVoice)");
    try {
        brotensor::DeviceScope scope(w->device);
        auto sw = std::make_unique<HostKokoroSession>();
        sw->busy = w->busy;
        sw->device = w->device;
        sw->model = w->model;
        if (vw) sw->session = std::make_unique<brosoundml::KokoroSession>(w->model, vw->voice);
        return g_kokoroSessionClass.createInstance(std::move(sw));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("createSession: ") + e.what());
    }
}

// The async Kokoro synthesis job (bro.tts.synthesize(kokoro, ...) and
// session.synthesize): the forward on a work thread, cancel polled between
// pipeline stages, onDone({ samples, sampleRate, durations, stages? }, info).
struct KokoroJob {
    std::vector<int32_t> ids;
    float speed = 1.0f;
    brosoundml::Voice voice;                        // a copy (device tensor)
    bool wantTrace = false;
    brosoundml::KokoroTrace trace;
    std::vector<float> samples;
    int sampleRate = 24000;
    std::vector<int32_t> durations;
    ModelGate gate;
    brotensor::Device device = brotensor::Device::CPU;
    ev::Persistent onDone, modelRef, voiceRef;
};

Value launchKokoroJob(std::shared_ptr<KokoroJob> job, std::function<void(KokoroJob&, const std::atomic<bool>&)> run) {
    auto work = [job, run](const std::atomic<bool>& cancel) { run(*job, cancel); };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        if (!ev::isFunction(job->onDone.get())) return;
        const bool trace = job->wantTrace && !cancelled && error.empty();
        ev::Persistent result(makeKokoroResult(job->samples, job->sampleRate, job->durations,
                                               trace ? &job->trace : nullptr));
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), result.get(), info.get());
    };
    return launchAsyncJob(std::move(work), nullptr, std::move(done));
}

// session.synthesize(phonemeIds, voice?, opts?) -> AsyncHandle
Value kokoroSessionSynthesize(Value self, std::span<const Value> a) {
    ev::Persistent selfRoot(self);  // `this` is a plain copy; the reads below allocate
    auto* sw = sessionSelf(self);
    if (!sw) return ev::throwTypeError("synthesize: not a KokoroSession");
    if (!hasArg(a, 0)) return ev::throwTypeError("synthesize(phonemeIds, voice?, opts?): phonemeIds required");
    auto job = std::make_shared<KokoroJob>();
    job->ids = readInt32Array(a[0]);
    if (job->ids.empty()) return ev::throwTypeError("synthesize: phonemeIds must be a non-empty Int32Array or number[]");
    size_t optsAt = 1;
    HostVoice* vw = hasArg(a, 1) ? voiceOf(a[1]) : nullptr;
    if (vw) optsAt = 2;
    if (!vw && !sw->session)
        return ev::throwTypeError("synthesize: this session has no voice; pass a Voice (loadVoice/createVoice)");
    ev::Persistent opts(isObjectArg(a, optsAt) ? a[optsAt] : ev::undefined());
    getFloatOpt(opts.get(), "speed", job->speed);
    job->wantTrace = getPropertyBool(opts.get(), "trace");
    job->onDone = getFunctionOpt(opts.get(), "onDone");
    if (!sw->busy.tryClaim()) return ev::throwError("synthesize: an operation is already in flight on this model");
    try {
        brotensor::DeviceScope scope(sw->device);
        if (vw) {
            if (!sw->session) sw->session = std::make_unique<brosoundml::KokoroSession>(sw->model, vw->voice);
            else sw->session->set_voice(vw->voice);
        }
    } catch (const std::exception& e) {
        sw->busy.release();
        return ev::throwError(std::string("synthesize: ") + e.what());
    }
    job->gate = sw->busy;
    job->device = sw->device;
    job->modelRef = ev::Persistent(selfRoot.get());
    brosoundml::KokoroSession* session = sw->session.get();
    return launchKokoroJob(job, [session](KokoroJob& j, const std::atomic<bool>& cancel) {
        brotensor::DeviceScope scope(j.device);
        auto buf = session->synthesize(j.ids, j.speed, &j.durations, cancelCheckOf(cancel),
                                       j.wantTrace ? &j.trace : nullptr);
        j.samples = std::move(buf.samples);
        j.sampleRate = buf.sample_rate;
    });
}

Value kokoroSessionSetVoice(Value self, std::span<const Value> a) {
    auto* sw = sessionSelf(self);
    if (!sw) return ev::throwTypeError("setVoice: not a KokoroSession");
    auto* vw = hasArg(a, 0) ? voiceOf(a[0]) : nullptr;
    if (!vw) return ev::throwTypeError("setVoice(voice): voice must be a Voice");
    if (sw->busy.isBusy()) return ev::throwError("setVoice: a synthesis is in flight on this model");
    brotensor::DeviceScope scope(sw->device);
    if (!sw->session) sw->session = std::make_unique<brosoundml::KokoroSession>(sw->model, vw->voice);
    else sw->session->set_voice(vw->voice);
    return ev::undefined();
}

}  // namespace

// bro.tts.synthesize(kokoro, phonemeIds, voice, opts?) -> AsyncHandle
Value kokoroSynthesizeAsync(Value modelVal, HostKokoro* w, std::span<const Value> args) {
    ev::Persistent modelRoot(modelVal);  // rooted: the reads below allocate
    if (args.size() < 3)
        return ev::throwTypeError("synthesize(kokoro, phonemeIds, voice, opts?): kokoro, phonemeIds and voice required");
    auto job = std::make_shared<KokoroJob>();
    job->ids = readInt32Array(args[1]);
    if (job->ids.empty()) return ev::throwTypeError("synthesize: phonemeIds must be a non-empty Int32Array or number[]");
    auto* vw = voiceOf(args[2]);
    if (!vw) return ev::throwTypeError("synthesize: voice must be a Voice (returned by loadVoice)");
    ev::Persistent opts(isObjectArg(args, 3) ? args[3] : ev::undefined());
    getFloatOpt(opts.get(), "speed", job->speed);
    job->wantTrace = getPropertyBool(opts.get(), "trace");
    job->onDone = getFunctionOpt(opts.get(), "onDone");
    if (!w->busy.tryClaim()) return ev::throwError("synthesize: an operation is already in flight on this model");
    job->gate = w->busy;
    job->device = w->device;
    job->voice = vw->voice;
    job->modelRef = ev::Persistent(modelRoot.get());
    job->voiceRef = ev::Persistent(args[2]);
    auto model = w->model;
    return launchKokoroJob(job, [model](KokoroJob& j, const std::atomic<bool>& cancel) {
        brotensor::DeviceScope scope(j.device);
        auto buf = model->synthesize(j.ids, j.voice, j.speed, &j.durations, cancelCheckOf(cancel),
                                     j.wantTrace ? &j.trace : nullptr);
        j.samples = std::move(buf.samples);
        j.sampleRate = buf.sample_rate;
    });
}

// bro.tts.decodeFrom(kokoro, voice, asr, F0, N, nPhonemes, opts?) -> AsyncHandle
Value kokoroDecodeFromAsync(Value modelVal, HostKokoro* w, std::span<const Value> args) {
    ev::Persistent modelRoot(modelVal);  // rooted: the reads below allocate
    if (args.size() < 6)
        return ev::throwTypeError("decodeFrom(kokoro, voice, asr, F0, N, nPhonemes, opts?): voice, asr, F0, N and nPhonemes required");
    auto* vw = voiceOf(args[1]);
    if (!vw) return ev::throwTypeError("decodeFrom: voice must be a Voice");
    struct DecodeJob {
        brosoundml::Voice voice;
        std::vector<float> asr, F0, N;
        int total = 0, nph = 0;
        bool wantTrace = false;
        brosoundml::KokoroTrace trace;
        std::vector<float> samples;
        int sampleRate = 24000;
        ModelGate gate;
        brotensor::Device device = brotensor::Device::CPU;
        std::shared_ptr<brosoundml::Kokoro> model;
        ev::Persistent onDone, modelRef, voiceRef;
    };
    auto job = std::make_shared<DecodeJob>();
    job->asr = readFloat32Array(args[2]);
    job->F0 = readFloat32Array(args[3]);
    job->N = readFloat32Array(args[4]);
    if (job->asr.empty() || job->F0.empty() || job->N.empty())
        return ev::throwTypeError("decodeFrom: asr, F0 and N must be non-empty Float32Arrays");
    if (job->F0.size() % 2 != 0) return ev::throwTypeError("decodeFrom: F0 length must be even (2*total)");
    job->nph = i32At(args, 5);
    job->total = static_cast<int>(job->F0.size() / 2);
    ev::Persistent opts(isObjectArg(args, 6) ? args[6] : ev::undefined());
    job->wantTrace = getPropertyBool(opts.get(), "trace");
    job->onDone = getFunctionOpt(opts.get(), "onDone");
    if (!w->busy.tryClaim()) return ev::throwError("decodeFrom: an operation is already in flight on this model");
    job->gate = w->busy;
    job->device = w->device;
    job->model = w->model;
    job->voice = vw->voice;
    job->modelRef = ev::Persistent(modelRoot.get());
    job->voiceRef = ev::Persistent(args[1]);
    auto work = [job](const std::atomic<bool>& cancel) {
        brotensor::DeviceScope scope(job->device);
        auto buf = job->model->decode_from(job->voice, job->nph, job->asr, job->total, job->F0, job->N,
                                           cancelCheckOf(cancel), job->wantTrace ? &job->trace : nullptr);
        job->samples = std::move(buf.samples);
        job->sampleRate = buf.sample_rate;
    };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        if (!ev::isFunction(job->onDone.get())) return;
        ev::Persistent result(makeAudioResult(job->samples, job->sampleRate));
        if (job->wantTrace && !cancelled && error.empty()) {
            ev::Persistent st(makeStagesArray(job->trace.stages));
            ev::setProperty(result.get(), "stages", st.get());
        }
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), result.get(), info.get());
    };
    return launchAsyncJob(std::move(work), nullptr, std::move(done));
}

// bro.tts.synthesizeStream(kokoro, phonemeChunks, voice, opts?) -> AsyncHandle
//   phonemeChunks: an array of id chunks (each Int32Array | number[]), or one
//   flat id array treated as a single chunk. opts.onChunk(samples, durations)
//   fires per chunk; onDone gets the concatenation.
Value kokoroSynthesizeStream(Value modelVal, HostKokoro* w, std::span<const Value> args) {
    ev::Persistent modelRoot(modelVal);  // rooted: the reads below allocate
    if (args.size() < 3)
        return ev::throwTypeError("synthesizeStream(kokoro, phonemeChunks, voice, opts?): kokoro, phonemeChunks and voice required");
    struct StreamJob {
        std::vector<std::vector<int32_t>> chunks;
        brosoundml::Voice voice;
        float speed = 1.0f;
        std::vector<float> samples;
        int sampleRate = 24000;
        ModelGate gate;
        brotensor::Device device = brotensor::Device::CPU;
        std::shared_ptr<brosoundml::Kokoro> model;
        ev::Persistent onChunk, onDone, modelRef, voiceRef;
        bool hasOnChunk = false;
        struct Slot { std::vector<float> samples; std::vector<int32_t> durations; };
        SpscSlots<Slot> slots;
    };
    auto job = std::make_shared<StreamJob>();
    {
        ev::Persistent chunksRoot(args[1]);
        bool isChunkArray = false;
        if (ev::isObject(chunksRoot.get()) && !ev::isTypedArray(chunksRoot.get())) {
            Value first = ev::getElement(chunksRoot.get(), 0);
            isChunkArray = ev::isTypedArray(first) || (ev::isObject(first) && !ev::isFunction(first));
        }
        if (isChunkArray) {
            const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(chunksRoot.get(), "length")));
            for (uint32_t i = 0; i < n; ++i) {
                std::vector<int32_t> c = readInt32Array(ev::getElement(chunksRoot.get(), i));
                if (!c.empty()) job->chunks.push_back(std::move(c));
            }
        } else {
            std::vector<int32_t> c = readInt32Array(chunksRoot.get());
            if (!c.empty()) job->chunks.push_back(std::move(c));
        }
    }
    if (job->chunks.empty())
        return ev::throwTypeError("synthesizeStream: phonemeChunks must be a non-empty array of Int32Array chunks");
    auto* vw = voiceOf(args[2]);
    if (!vw) return ev::throwTypeError("synthesizeStream: voice must be a Voice (returned by loadVoice)");
    ev::Persistent opts(isObjectArg(args, 3) ? args[3] : ev::undefined());
    getFloatOpt(opts.get(), "speed", job->speed);
    job->onChunk = getFunctionOpt(opts.get(), "onChunk");
    job->onDone = getFunctionOpt(opts.get(), "onDone");
    job->hasOnChunk = ev::isFunction(job->onChunk.get());
    job->slots.reserve(job->chunks.size());
    if (!w->busy.tryClaim()) return ev::throwError("synthesizeStream: an operation is already in flight on this model");
    job->gate = w->busy;
    job->device = w->device;
    job->model = w->model;
    job->voice = vw->voice;
    job->modelRef = ev::Persistent(modelRoot.get());
    job->voiceRef = ev::Persistent(args[2]);

    auto work = [job](const std::atomic<bool>& cancel) {
        brotensor::DeviceScope scope(job->device);
        auto onChunkCb = [job](const float* s, int n, const int32_t* d, int nd) {
            job->slots.emplace([&](StreamJob::Slot& slot) {
                slot.samples.assign(s, s + n);
                if (d && nd > 0) slot.durations.assign(d, d + nd);
            });
        };
        auto buf = job->model->synthesize_stream(job->chunks, job->voice, onChunkCb, job->speed,
                                                 cancelCheckOf(cancel));
        job->samples = std::move(buf.samples);
        job->sampleRate = buf.sample_rate;
    };
    auto poll = [job] {
        if (!job->hasOnChunk) return;
        job->slots.drain([&](StreamJob::Slot& slot) {
            ev::Persistent s(makeFloat32Array(slot.samples));
            ev::Persistent d(makeInt32Array(slot.durations));
            callCallback2(job->onChunk.get(), s.get(), d.get());
            std::vector<float>().swap(slot.samples);
            std::vector<int32_t>().swap(slot.durations);
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

void decorateKokoro(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = kokoroSelf(self);
        return ev::fromBool(w && w->model && w->model->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = kokoroSelf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    auto cfgInt = [&](const char* name, int brosoundml::KokoroConfig::*field) {
        b.accessor(name, [field](Value self, std::span<const Value>) -> Value {
            auto* w = kokoroSelf(self);
            return ev::fromDouble(w && w->model ? w->model->config().*field : 0);
        });
    };
    cfgInt("sampleRate", &brosoundml::KokoroConfig::sample_rate);
    cfgInt("nTokens", &brosoundml::KokoroConfig::n_tokens);
    cfgInt("hiddenDim", &brosoundml::KokoroConfig::hidden_dim);
    cfgInt("styleDim", &brosoundml::KokoroConfig::style_dim);
    cfgInt("nLayer", &brosoundml::KokoroConfig::n_layer);
    b.def("loadVoice", 2, kokoroLoadVoice);
    b.def("createVoice", 2, kokoroCreateVoice);
    b.def("synthesize", 3, [](Value self, std::span<const Value> a) -> Value {
        return kokoroSynthesizeSync(self, a, false);
    });
    b.def("synthesizeTraced", 3, [](Value self, std::span<const Value> a) -> Value {
        return kokoroSynthesizeSync(self, a, true);
    });
    b.def("decodeFrom", 6, kokoroDecodeFromSync);
    b.def("vocab", 0, kokoroVocab);
    b.def("encodePhonemes", 1, kokoroEncodePhonemes);
    b.def("createSession", 1, kokoroCreateSession);
}

void decorateKokoroSession(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* s = sessionSelf(self);
        return ev::fromBool(s && s->model && s->model->loaded());
    });
    b.def("synthesize", 3, kokoroSessionSynthesize);
    b.def("setVoice", 1, kokoroSessionSetVoice);
    b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
        auto* s = sessionSelf(self);
        if (!s) return ev::throwTypeError("reset: not a KokoroSession");
        return ev::undefined();   // Kokoro sessions carry no decode state
    });
}

}  // namespace

void installTtsKokoroClasses() {
    g_voiceClass.install("Voice", 0, nullptr, decorateVoice);
    g_kokoroSessionClass.install("KokoroSession", 0, nullptr, decorateKokoroSession);
    g_kokoroClass.install("KokoroModel", 0, nullptr, decorateKokoro);
}

} // namespace brosoundml::api
