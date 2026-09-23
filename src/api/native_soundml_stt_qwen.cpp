// bro.stt — Qwen3-ASR: the model, its decode sessions, the latent tap and
// the encoder-only streaming class. Loaders live in native_soundml_stt.cpp.
#include "soundml_stt_internal.h"

namespace brosoundml::api {

namespace {

HostQwenAsrModel* modelSelf(Value self) {
    return static_cast<HostQwenAsrModel*>(g_qwenAsrModelClass.unwrap(self));
}
HostQwenAsrSession* sessionSelf(Value self) {
    return static_cast<HostQwenAsrSession*>(g_qwenAsrSessionClass.unwrap(self));
}
HostQwenAsrStream* streamSelf(Value self) {
    return static_cast<HostQwenAsrStream*>(g_qwenAsrStreamClass.unwrap(self));
}

struct QwenJob : SttJobBase {
    std::shared_ptr<brosoundml::QwenAsr> model;
    brosoundml::QwenAsrSession* session = nullptr;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate gate;
    std::vector<int32_t> contextIds;
};

brosoundml::QwenAsr::Transcription runQwen(QwenJob& job,
                                           const brosoundml::QwenAsr::TranscribeOptions& o) {
    brotensor::DeviceScope scope(job.device);
    if (job.session) return job.model->transcribe(*job.session, job.audio, o);
    return job.model->transcribe(job.audio, o);
}

}  // namespace

Value qwenAsrTranscribe(Value modelVal, HostQwenAsrModel* model, HostQwenAsrSession* session,
                        std::span<const Value> args, const char* fn) {
    ev::Persistent modelRoot(modelVal);  // the reads below allocate
    const std::string pre = std::string(fn) + ": ";
    auto job = std::make_shared<QwenJob>();
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
    if (hasProperty(optsRoot.get(), "contextIds")) {
        bool ok = false;
        job->contextIds = readInt32Array(ev::getProperty(optsRoot.get(), "contextIds"), &ok);
        if (!ok) return ev::throwTypeError(pre + "opts.contextIds must be an Int32Array or number[]");
    }
    job->onDone = getFunctionOpt(optsRoot.get(), "onDone");
    job->onToken = getFunctionOpt(optsRoot.get(), "onToken");
    job->hasOnToken = ev::isFunction(job->onToken.get());
    const bool async = ev::isFunction(job->onDone.get());

    if (job->gate.isBusy())
        return ev::throwError(pre + "an operation is already in flight on this model");

    if (!async) {
        brosoundml::QwenAsr::TranscribeOptions o;
        o.max_new_tokens = job->maxNew;
        o.context_ids = job->contextIds;
        if (job->hasOnToken)
            o.on_token = [job](int32_t id) { callCallback1(job->onToken.get(), ev::fromDouble(id)); };
        try {
            auto out = runQwen(*job, o);
            return makeInt32Array(out.token_ids);
        } catch (const std::exception& e) {
            return ev::throwError(pre + e.what());
        }
    }

    if (!job->gate.tryClaim())
        return ev::throwError(pre + "an operation is already in flight on this model");
    job->modelRef = ev::Persistent(modelRoot.get());
    if (job->hasOnToken) job->tokens.reserve(kSttTokenSlots);

    auto work = [job](const std::atomic<bool>& cancel) {
        brosoundml::QwenAsr::TranscribeOptions o;
        o.max_new_tokens = job->maxNew;
        o.context_ids = job->contextIds;
        o.cancel = cancelCheckOf(cancel);
        if (job->hasOnToken) o.on_token = [job](int32_t id) { job->publishToken(id); };
        auto out = runQwen(*job, o);
        job->tokenIds = std::move(out.token_ids);
    };
    auto poll = [job] { job->drainTokens(); };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        ev::Persistent ids(makeInt32Array(job->tokenIds));
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), ids.get(), info.get());
    };
    return launchAsyncJob(std::move(work), std::move(poll), std::move(done));
}

namespace {

// { latents: Float32Array (frames*latentDim, row-major), frames, latentDim, latentHz }
Value makeLatentResult(const std::vector<float>& rows, int frames, const brosoundml::QwenAsrConfig& cfg) {
    ObjectBuilder res;
    {
        ev::Persistent v(makeFloat32Array(rows));
        res.set("latents", v.get());
    }
    res.set("frames", ev::fromDouble(frames));
    res.set("latentDim", ev::fromDouble(cfg.latent_dim));
    res.set("latentHz", ev::fromDouble(cfg.latent_hz));
    return res.get();
}

void decorateQwenAsrModel(ObjectBuilder& b) {
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
    b.accessor("latentDim", [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        return ev::fromDouble(w && w->model ? w->model->config().latent_dim : 0);
    });
    b.accessor("latentHz", [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        return ev::fromDouble(w && w->model ? w->model->config().latent_hz : 0.0);
    });
    b.accessor("vocabSize", [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        return ev::fromDouble(w && w->model ? w->model->config().vocab_size : 0);
    });
    b.accessor("asrTextId", [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        return ev::fromDouble(w && w->model ? w->model->config().asr_text_token_id : -1);
    });
    b.def("transcribe", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* w = modelSelf(self);
        if (!w) return ev::throwTypeError("transcribe: not a QwenAsrModel");
        return qwenAsrTranscribe(self, w, nullptr, a, "transcribe");
    });
    // encode(audio) -> { latents, frames, latentDim, latentHz }: the encoder +
    // projector only, synchronous (one clip is a single encoder pass).
    b.def("encode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = modelSelf(self);
        if (!w) return ev::throwTypeError("encode: not a QwenAsrModel");
        if (!w->model || !w->model->loaded()) return ev::throwError("encode: model is not loaded");
        if (!hasArg(a, 0)) return ev::throwTypeError("encode(audio): audio required");
        brosoundml::AudioBuffer audio;
        std::string err;
        if (!readAudioBuffer(argAt(a, 0), audio, err, w->model->config().sample_rate))
            return ev::throwTypeError("encode: " + err);
        if (w->busy.isBusy()) return ev::throwError("encode: an operation is already in flight on this model");
        try {
            brotensor::DeviceScope scope(w->device);
            std::vector<float> rows;
            const int frames = w->model->encode_to_host(audio, rows);
            return makeLatentResult(rows, frames, w->model->config());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encode: ") + e.what());
        }
    });
    b.def("createSession", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        if (!w) return ev::throwTypeError("createSession: not a QwenAsrModel");
        if (!w->model || !w->model->loaded()) return ev::throwError("createSession: model is not loaded");
        try {
            brotensor::DeviceScope scope(w->device);
            auto sw = std::make_unique<HostQwenAsrSession>();
            sw->model = w->model;
            sw->busy = w->busy;
            sw->device = w->device;
            sw->session = w->model->make_session();
            return g_qwenAsrSessionClass.createInstance(std::move(sw));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("createSession: ") + e.what());
        }
    });
}

void decorateQwenAsrSession(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* s = sessionSelf(self);
        return ev::fromBool(s && s->model && s->model->loaded());
    });
    b.def("transcribe", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* s = sessionSelf(self);
        if (!s) return ev::throwTypeError("transcribe: not a QwenAsrSession");
        return qwenAsrTranscribe(self, nullptr, s, a, "transcribe");
    });
    b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
        auto* s = sessionSelf(self);
        if (!s) return ev::throwTypeError("reset: not a QwenAsrSession");
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

// feed(audio) -> number of latent rows finalized by this call. Accepts a bare
// Float32Array or { samples, sampleRate } (must be the model's 16 kHz).
Value streamFeed(Value self, std::span<const Value> a) {
    auto* s = streamSelf(self);
    if (!s || !s->stream) return ev::throwTypeError("feed: not a QwenAsrStream");
    if (!s->stream->loaded()) return ev::throwError("feed: stream is not loaded");
    if (!hasArg(a, 0)) return ev::throwTypeError("feed(samples): samples required");
    brosoundml::AudioBuffer audio;
    std::string err;
    const int rate = s->stream->config().sample_rate;
    if (!readAudioBuffer(argAt(a, 0), audio, err, rate)) return ev::throwTypeError("feed: " + err);
    if (audio.sample_rate != rate)
        return ev::throwRangeError("feed: samples must be " + std::to_string(rate) + " Hz");
    try {
        brotensor::DeviceScope scope(s->device);
        return ev::fromDouble(s->stream->feed(audio.samples.data(), static_cast<int>(audio.samples.size())));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("feed: ") + e.what());
    }
}

void decorateQwenAsrStream(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* s = streamSelf(self);
        return ev::fromBool(s && s->stream && s->stream->loaded());
    });
    b.accessor("sampleRate", [](Value self, std::span<const Value>) -> Value {
        auto* s = streamSelf(self);
        return ev::fromDouble(s && s->stream && s->stream->loaded() ? s->stream->config().sample_rate : 0);
    });
    b.accessor("latentDim", [](Value self, std::span<const Value>) -> Value {
        auto* s = streamSelf(self);
        return ev::fromDouble(s && s->stream && s->stream->loaded() ? s->stream->config().latent_dim : 0);
    });
    b.accessor("latentHz", [](Value self, std::span<const Value>) -> Value {
        auto* s = streamSelf(self);
        return ev::fromDouble(s && s->stream && s->stream->loaded() ? s->stream->config().latent_hz : 0.0);
    });
    b.accessor("frames", [](Value self, std::span<const Value>) -> Value {
        auto* s = streamSelf(self);
        return ev::fromDouble(s && s->stream && s->stream->loaded() ? s->stream->frames() : 0);
    });
    b.accessor("blockChunks", [](Value self, std::span<const Value>) -> Value {
        auto* s = streamSelf(self);
        return ev::fromDouble(s && s->stream && s->stream->loaded() ? s->stream->block_chunks() : 0);
    });
    b.accessor("blockFrames", [](Value self, std::span<const Value>) -> Value {
        auto* s = streamSelf(self);
        return ev::fromDouble(s && s->stream && s->stream->loaded() ? s->stream->block_frames() : 0);
    });
    b.def("feed", 1, streamFeed);
    // finish() -> rows finalized from the trailing partial block.
    b.def("finish", 0, [](Value self, std::span<const Value>) -> Value {
        auto* s = streamSelf(self);
        if (!s || !s->stream) return ev::throwTypeError("finish: not a QwenAsrStream");
        if (!s->stream->loaded()) return ev::throwError("finish: stream is not loaded");
        try {
            brotensor::DeviceScope scope(s->device);
            return ev::fromDouble(s->stream->finish());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("finish: ") + e.what());
        }
    });
    // latents(start = 0, count = frames - start) -> Float32Array copy of
    // rows [start, start + count), row-major (count, latentDim).
    b.def("latents", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* s = streamSelf(self);
        if (!s || !s->stream) return ev::throwTypeError("latents: not a QwenAsrStream");
        if (!s->stream->loaded()) return ev::throwError("latents: stream is not loaded");
        const int frames = s->stream->frames();
        const int dim = s->stream->config().latent_dim;
        const int start = hasArg(a, 0) && !ev::isUndefined(a[0]) ? i32At(a, 0) : 0;
        const int count = hasArg(a, 1) && !ev::isUndefined(a[1]) ? i32At(a, 1) : frames - start;
        if (start < 0 || start > frames)
            return ev::throwRangeError("latents: start out of range [0, " + std::to_string(frames) + "]");
        if (count < 0 || start + count > frames)
            return ev::throwRangeError("latents: count out of range [0, " + std::to_string(frames - start) + "]");
        const auto& all = s->stream->latents();
        const size_t off = static_cast<size_t>(start) * static_cast<size_t>(dim);
        const size_t n = static_cast<size_t>(count) * static_cast<size_t>(dim);
        return makeFloat32Array(std::span<const float>(all.data() + off, n));
    });
}

}  // namespace

void installSttQwenClasses() {
    g_qwenAsrSessionClass.install("QwenAsrSession", 0, nullptr, decorateQwenAsrSession);
    g_qwenAsrModelClass.install("QwenAsrModel", 0, nullptr, decorateQwenAsrModel);
    g_qwenAsrStreamClass.install("QwenAsrStream", 0, nullptr, decorateQwenAsrStream);
}

} // namespace brosoundml::api
