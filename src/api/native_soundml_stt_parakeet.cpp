// bro.stt — Parakeet-TDT: the SentencePiece tokenizer (a HF tokenizer.json
// unigram, parsed by brolm::t5::Tokenizer), the model, and its decode
// sessions. Loaders live in native_soundml_stt.cpp.
#include "soundml_stt_internal.h"

namespace brosoundml::api {

namespace {

HostParakeetTokenizer* tokSelf(Value self) {
    return static_cast<HostParakeetTokenizer*>(g_parakeetTokenizerClass.unwrap(self));
}
HostParakeetModel* modelSelf(Value self) {
    return static_cast<HostParakeetModel*>(g_parakeetModelClass.unwrap(self));
}
HostParakeetSession* sessionSelf(Value self) {
    return static_cast<HostParakeetSession*>(g_parakeetSessionClass.unwrap(self));
}

// { tokenIds, tokenFrames, frameOffsets } — tokenFrames[i] is the encoder
// frame token i was emitted at (× frameSeconds for a start time). The doc
// spells the same array frameOffsets; both names answer.
Value makeParakeetResult(const std::vector<int32_t>& ids, const std::vector<int32_t>& frames) {
    ObjectBuilder res;
    {
        ev::Persistent v(makeInt32Array(ids));
        res.set("tokenIds", v.get());
    }
    {
        ev::Persistent v(makeInt32Array(frames));
        res.set("tokenFrames", v.get());
        res.set("frameOffsets", v.get());
    }
    return res.get();
}

void decorateParakeetTokenizer(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = tokSelf(self);
        return ev::fromBool(w && w->tok != nullptr);
    });
    b.accessor("vocabCount", [](Value self, std::span<const Value>) -> Value {
        auto* w = tokSelf(self);
        return ev::fromDouble(w && w->tok ? static_cast<double>(w->tok->vocab_count()) : 0);
    });
    b.accessor("padId", [](Value self, std::span<const Value>) -> Value {
        auto* w = tokSelf(self);
        return ev::fromDouble(w && w->tok ? w->tok->pad_id() : -1);
    });
    b.accessor("eosId", [](Value self, std::span<const Value>) -> Value {
        auto* w = tokSelf(self);
        return ev::fromDouble(w && w->tok ? w->tok->eos_id() : -1);
    });
    b.accessor("unkId", [](Value self, std::span<const Value>) -> Value {
        auto* w = tokSelf(self);
        return ev::fromDouble(w && w->tok ? w->tok->unk_id() : -1);
    });
    // tokenize(text) / encode(text) -> Int32Array: unigram pieces, no eos/pad.
    auto tokenize = [](Value self, std::span<const Value> a) -> Value {
        auto* w = tokSelf(self);
        if (!w || !w->tok) return ev::throwTypeError("encode: not a ParakeetTokenizer");
        if (!isStringArg(a, 0)) return ev::throwTypeError("encode(text): text required");
        try {
            return makeInt32Array(w->tok->tokenize(strAt(a, 0)));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encode: ") + e.what());
        }
    };
    b.def("tokenize", 1, tokenize);
    b.def("encode", 1, tokenize);
    // decode(ids) -> string: ids outside the vocab (blank / pad) are skipped,
    // so the raw Parakeet id stream decodes directly.
    b.def("decode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = tokSelf(self);
        if (!w || !w->tok) return ev::throwTypeError("decode: not a ParakeetTokenizer");
        bool ok = false;
        auto ids = readInt32Array(argAt(a, 0), &ok);
        if (!ok) return ev::throwTypeError("decode(ids): ids must be an Int32Array or number[]");
        try {
            return ev::fromUtf8(w->tok->decode(ids));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("decode: ") + e.what());
        }
    });
}

struct ParakeetJob : SttJobBase {
    std::shared_ptr<brosoundml::Parakeet> model;
    brosoundml::ParakeetSession* session = nullptr;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate gate;
    std::vector<int32_t> tokenFrames;
};

brosoundml::Parakeet::Transcription runParakeet(ParakeetJob& job,
                                                const brosoundml::Parakeet::TranscribeOptions& o) {
    brotensor::DeviceScope scope(job.device);
    if (job.session) return job.model->transcribe(*job.session, job.audio, o);
    return job.model->transcribe(job.audio, o);
}

}  // namespace

Value parakeetTranscribe(Value modelVal, HostParakeetModel* model, HostParakeetSession* session,
                         std::span<const Value> args, const char* fn) {
    ev::Persistent modelRoot(modelVal);  // the reads below allocate
    const std::string pre = std::string(fn) + ": ";
    auto job = std::make_shared<ParakeetJob>();
    if (session) {
        job->model = session->model;
        job->session = &session->session;
        job->device = session->device;
        job->gate = session->busy;
    } else {
        job->model = model->model;
        job->device = model->device;
        job->gate = model->busy;
    }
    if (!job->model || !job->model->loaded()) return ev::throwError(pre + "model is not loaded");
    if (!hasArg(args, 0))
        return ev::throwTypeError(std::string(fn) + "(audio, opts?): audio required");
    std::string err;
    if (!readAudioBuffer(argAt(args, 0), job->audio, err, job->model->config().sample_rate))
        return ev::throwTypeError(pre + err);

    ev::Persistent optsRoot(isObjectArg(args, 1) ? args[1] : ev::undefined());
    getIntOpt(optsRoot.get(), "maxNewTokens", job->maxNew);
    job->onDone = getFunctionOpt(optsRoot.get(), "onDone");
    job->onToken = getFunctionOpt(optsRoot.get(), "onToken");
    job->hasOnToken = ev::isFunction(job->onToken.get());
    const bool async = ev::isFunction(job->onDone.get());

    if (job->gate.isBusy())
        return ev::throwError(pre + "an operation is already in flight on this model");

    if (!async) {
        brosoundml::Parakeet::TranscribeOptions o;
        o.max_new_tokens = job->maxNew;
        if (job->hasOnToken)
            o.on_token = [job](int32_t id) { callCallback1(job->onToken.get(), ev::fromDouble(id)); };
        try {
            auto out = runParakeet(*job, o);
            return makeParakeetResult(out.token_ids, out.token_frames);
        } catch (const std::exception& e) {
            return ev::throwError(pre + e.what());
        }
    }

    if (!job->gate.tryClaim())
        return ev::throwError(pre + "an operation is already in flight on this model");
    job->modelRef = ev::Persistent(modelRoot.get());
    if (job->hasOnToken) job->tokens.reserve(kSttTokenSlots);

    auto work = [job](const std::atomic<bool>& cancel) {
        brosoundml::Parakeet::TranscribeOptions o;
        o.max_new_tokens = job->maxNew;
        o.cancel = cancelCheckOf(cancel);
        if (job->hasOnToken) o.on_token = [job](int32_t id) { job->publishToken(id); };
        auto out = runParakeet(*job, o);
        job->tokenIds = std::move(out.token_ids);
        job->tokenFrames = std::move(out.token_frames);
    };
    auto poll = [job] { job->drainTokens(); };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        ev::Persistent res(makeParakeetResult(job->tokenIds, job->tokenFrames));
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), res.get(), info.get());
    };
    return launchAsyncJob(std::move(work), std::move(poll), std::move(done));
}

namespace {

void decorateParakeetModel(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        return ev::fromBool(w && w->model && w->model->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("sampleRate", [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        return ev::fromDouble(w && w->model ? w->model->config().sample_rate : 0);
    });
    b.accessor("vocabSize", [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        return ev::fromDouble(w && w->model ? w->model->config().vocab_size : 0);
    });
    b.accessor("blankTokenId", [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        return ev::fromDouble(w && w->model ? w->model->config().blank_token_id : -1);
    });
    b.accessor("frameSeconds", [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        return ev::fromDouble(w && w->model ? w->model->config().frame_seconds() : 0.0);
    });
    b.def("transcribe", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* w = modelSelf(self);
        if (!w) return ev::throwTypeError("transcribe: not a ParakeetModel");
        return parakeetTranscribe(self, w, nullptr, a, "transcribe");
    });
    b.def("createSession", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        if (!w) return ev::throwTypeError("createSession: not a ParakeetModel");
        if (!w->model || !w->model->loaded()) return ev::throwError("createSession: model is not loaded");
        try {
            brotensor::DeviceScope scope(w->device);
            auto sw = std::make_unique<HostParakeetSession>();
            sw->model = w->model;
            sw->busy = w->busy;
            sw->device = w->device;
            sw->session = w->model->make_session();
            return g_parakeetSessionClass.createInstance(std::move(sw));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("createSession: ") + e.what());
        }
    });
}

void decorateParakeetSession(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* s = sessionSelf(self);
        return ev::fromBool(s && s->model && s->model->loaded());
    });
    b.def("transcribe", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* s = sessionSelf(self);
        if (!s) return ev::throwTypeError("transcribe: not a ParakeetSession");
        return parakeetTranscribe(self, nullptr, s, a, "transcribe");
    });
    b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
        auto* s = sessionSelf(self);
        if (!s) return ev::throwTypeError("reset: not a ParakeetSession");
        if (s->busy.isBusy()) return ev::throwError("reset: a transcribe is in flight on this model");
        try {
            brotensor::DeviceScope scope(s->device);
            s->model->reset(s->session);
        } catch (const std::exception& e) {
            return ev::throwError(std::string("reset: ") + e.what());
        }
        return ev::undefined();
    });
}

}  // namespace

void installSttParakeetClasses() {
    g_parakeetTokenizerClass.install("ParakeetTokenizer", 0, nullptr, decorateParakeetTokenizer);
    g_parakeetSessionClass.install("ParakeetSession", 0, nullptr, decorateParakeetSession);
    g_parakeetModelClass.install("ParakeetModel", 0, nullptr, decorateParakeetModel);
}

} // namespace brosoundml::api
