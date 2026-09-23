// bro.stt — Whisper: the HF-format tokenizer, the model, and its decode
// sessions. Loaders live in native_soundml_stt.cpp.
#include "soundml_stt_internal.h"

namespace brosoundml::api {

namespace {

HostWhisperTokenizer* tokSelf(Value self) {
    return static_cast<HostWhisperTokenizer*>(g_whisperTokenizerClass.unwrap(self));
}

// ── WhisperTokenizer ──────────────────────────────────────────────────────

void decorateWhisperTokenizer(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = tokSelf(self);
        return ev::fromBool(w && w->tok != nullptr);
    });
    auto idGetter = [&b](const char* name, int (brolm::whisper::Tokenizer::*fn)() const) {
        b.accessor(name, [fn](Value self, std::span<const Value>) -> Value {
            auto* w = tokSelf(self);
            if (!w || !w->tok) return ev::fromDouble(-1);
            return ev::fromDouble(((*w->tok).*fn)());
        });
    };
    idGetter("eosId",            &brolm::whisper::Tokenizer::eos_id);
    idGetter("sotId",            &brolm::whisper::Tokenizer::sot_id);
    idGetter("noSpeechId",       &brolm::whisper::Tokenizer::no_speech_id);
    idGetter("noTimestampsId",   &brolm::whisper::Tokenizer::no_timestamps_id);
    idGetter("transcribeId",     &brolm::whisper::Tokenizer::transcribe_id);
    idGetter("translateId",      &brolm::whisper::Tokenizer::translate_id);
    idGetter("firstTimestampId", &brolm::whisper::Tokenizer::first_timestamp_id);
    idGetter("lastTimestampId",  &brolm::whisper::Tokenizer::last_timestamp_id);
    b.accessor("vocabCount", [](Value self, std::span<const Value>) -> Value {
        auto* w = tokSelf(self);
        return ev::fromDouble(w && w->tok ? static_cast<double>(w->tok->vocab_count()) : 0);
    });
    b.accessor("mergeCount", [](Value self, std::span<const Value>) -> Value {
        auto* w = tokSelf(self);
        return ev::fromDouble(w && w->tok ? static_cast<double>(w->tok->merge_count()) : 0);
    });

    // encode(text, addSpecial=false) -> Int32Array
    b.def("encode", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* w = tokSelf(self);
        if (!w || !w->tok) return ev::throwTypeError("encode: not a WhisperTokenizer");
        if (!isStringArg(a, 0)) return ev::throwTypeError("encode(text, addSpecial?): text required");
        try {
            return makeInt32Array(w->tok->encode(strAt(a, 0), boolAt(a, 1)));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encode: ") + e.what());
        }
    });
    // decode(ids, skipSpecial=false) -> string
    b.def("decode", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* w = tokSelf(self);
        if (!w || !w->tok) return ev::throwTypeError("decode: not a WhisperTokenizer");
        bool ok = false;
        auto ids = readInt32Array(argAt(a, 0), &ok);
        if (!ok) return ev::throwTypeError("decode(ids): ids must be an Int32Array or number[]");
        try {
            return ev::fromUtf8(w->tok->decode(ids, boolAt(a, 1)));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("decode: ") + e.what());
        }
    });
    // buildPrompt(language='en', task='transcribe', withTimestamps=true) -> Int32Array
    b.def("buildPrompt", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* w = tokSelf(self);
        if (!w || !w->tok) return ev::throwTypeError("buildPrompt: not a WhisperTokenizer");
        std::string lang = isStringArg(a, 0) ? strAt(a, 0) : "en";
        std::string task = isStringArg(a, 1) ? strAt(a, 1) : "transcribe";
        const bool withTs = boolAt(a, 2, true);
        try {
            return makeInt32Array(w->tok->build_prompt(lang, task, withTs));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("buildPrompt: ") + e.what());
        }
    });
    b.def("isTimestamp", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = tokSelf(self);
        if (!w || !w->tok) return ev::throwTypeError("isTimestamp: not a WhisperTokenizer");
        if (!hasArg(a, 0)) return ev::throwTypeError("isTimestamp(id): id required");
        return ev::fromBool(w->tok->is_timestamp(i32At(a, 0)));
    });
    b.def("timestampSeconds", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = tokSelf(self);
        if (!w || !w->tok) return ev::throwTypeError("timestampSeconds: not a WhisperTokenizer");
        if (!hasArg(a, 0)) return ev::throwTypeError("timestampSeconds(id): id required");
        return ev::fromDouble(w->tok->timestamp_seconds(i32At(a, 0)));
    });
}

// ── transcribe (sync + async) ─────────────────────────────────────────────

struct WhisperJob : SttJobBase {
    std::shared_ptr<brosoundml::Whisper> model;
    brosoundml::WhisperSession* session = nullptr;  // null: the model's own path
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate gate;
    std::vector<int32_t> prompt;
    int timestampBeginId = -1;
    int noTimestampsId = -1;

    // Long-form window marks, published by the decode thread alongside the
    // tokens. Each carries the token index it precedes, so the poll fires
    // onWindow BEFORE the tokens of that window.
    struct WindowSlot { double start = 0.0; size_t at = 0; };
    SpscSlots<WindowSlot> windows;
    std::vector<brosoundml::Whisper::Transcription::WindowMark> windowMarks;  // the full list, for onDone
    bool hasOnWindow = false;
    ev::Persistent onWindow;

    void drainAll() {
        if (!hasOnWindow) { drainTokens(); return; }
        // Interleave: windows whose `at` <= drained token count fire first.
        const size_t nTok = tokens.produced.load(std::memory_order_acquire);
        const size_t nWin = windows.produced.load(std::memory_order_acquire);
        while (tokens.drained < nTok || windows.drained < nWin) {
            if (windows.drained < nWin && windows.slots[windows.drained].at <= tokens.drained) {
                callCallback1(onWindow.get(), ev::fromDouble(windows.slots[windows.drained].start));
                ++windows.drained;
                continue;
            }
            if (tokens.drained < nTok) {
                if (hasOnToken) callCallback1(onToken.get(), ev::fromDouble(tokens.slots[tokens.drained]));
                ++tokens.drained;
                continue;
            }
            break;
        }
    }
};

brosoundml::Whisper::TranscribeOptions optionsFor(WhisperJob& job) {
    brosoundml::Whisper::TranscribeOptions o;
    o.max_new_tokens = job.maxNew;
    o.timestamp_begin_id = job.timestampBeginId;
    o.no_timestamps_id = job.noTimestampsId;
    return o;
}

brosoundml::Whisper::Transcription runWhisper(WhisperJob& job,
                                              const brosoundml::Whisper::TranscribeOptions& o) {
    brotensor::DeviceScope scope(job.device);
    if (job.session) return job.model->transcribe(*job.session, job.audio, job.prompt, o);
    return job.model->transcribe(job.audio, job.prompt, o);
}

}  // namespace

Value whisperTranscribe(Value modelVal, HostWhisperModel* model, HostWhisperSession* session,
                        std::span<const Value> args, const char* fn) {
    ev::Persistent modelRoot(modelVal);  // the reads below allocate
    const std::string pre = std::string(fn) + ": ";
    auto job = std::make_shared<WhisperJob>();
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
        return ev::throwTypeError(std::string(fn) + "(audio, promptIds|opts, opts?): audio required");
    std::string err;
    if (!readAudioBuffer(argAt(args, 0), job->audio, err, job->model->config().sample_rate))
        return ev::throwTypeError(pre + err);

    // promptOrOpts: an id array, or the options object carrying opts.prompt.
    ev::Persistent optsRoot;
    {
        Value promptVal = argAt(args, 1);
        const bool promptIsArray = ev::isTypedArray(promptVal) ||
            (ev::isObject(promptVal) && !ev::isFunction(promptVal) &&
             ev::isNumber(ev::getProperty(promptVal, "length")));
        if (promptIsArray) {
            job->prompt = readInt32Array(argAt(args, 1));
            if (isObjectArg(args, 2)) optsRoot.set(args[2]);
        } else if (isObjectArg(args, 1)) {  // promptVal is stale after the length read
            optsRoot.set(args[1]);
            job->prompt = readInt32Array(ev::getProperty(optsRoot.get(), "prompt"));
        } else if (isObjectArg(args, 2)) {
            optsRoot.set(args[2]);
            job->prompt = readInt32Array(ev::getProperty(optsRoot.get(), "prompt"));
        }
    }
    if (job->prompt.empty())
        return ev::throwTypeError(pre + "promptIds must be a non-empty Int32Array or number[] "
                                        "(use tokenizer.buildPrompt(lang, task))");

    getIntOpt(optsRoot.get(), "maxNewTokens", job->maxNew);
    getIntOpt(optsRoot.get(), "timestampBeginId", job->timestampBeginId);
    getIntOpt(optsRoot.get(), "noTimestampsId", job->noTimestampsId);
    job->onDone = getFunctionOpt(optsRoot.get(), "onDone");
    job->onToken = getFunctionOpt(optsRoot.get(), "onToken");
    job->onWindow = getFunctionOpt(optsRoot.get(), "onWindow");
    job->hasOnToken = ev::isFunction(job->onToken.get());
    job->hasOnWindow = ev::isFunction(job->onWindow.get());
    const bool async = ev::isFunction(job->onDone.get());

    if (job->gate.isBusy())
        return ev::throwError(pre + "an operation is already in flight on this model");

    // ── Sync path: the callbacks fire straight back into JS from this thread.
    if (!async) {
        brosoundml::Whisper::TranscribeOptions o = optionsFor(*job);
        if (job->hasOnToken)
            o.on_token = [job](int32_t id) { callCallback1(job->onToken.get(), ev::fromDouble(id)); };
        if (job->hasOnWindow)
            o.on_window = [job](double start) { callCallback1(job->onWindow.get(), ev::fromDouble(start)); };
        try {
            auto out = runWhisper(*job, o);
            return makeInt32Array(out.token_ids);
        } catch (const std::exception& e) {
            return ev::throwError(pre + e.what());
        }
    }

    // ── Async path.
    if (!job->gate.tryClaim())
        return ev::throwError(pre + "an operation is already in flight on this model");
    job->modelRef = ev::Persistent(modelRoot.get());
    if (job->hasOnToken) job->tokens.reserve(kSttTokenSlots);
    if (job->hasOnWindow) job->windows.reserve(4096);

    auto work = [job](const std::atomic<bool>& cancel) {
        brosoundml::Whisper::TranscribeOptions o = optionsFor(*job);
        o.cancel = cancelCheckOf(cancel);
        if (job->hasOnToken) o.on_token = [job](int32_t id) { job->publishToken(id); };
        if (job->hasOnWindow) {
            o.on_window = [job](double start) {
                job->windows.emplace([&](WhisperJob::WindowSlot& s) {
                    s.start = start;
                    s.at = job->tokens.produced.load(std::memory_order_relaxed);
                });
            };
        }
        auto out = runWhisper(*job, o);
        job->tokenIds = std::move(out.token_ids);
        job->windowMarks = std::move(out.windows);
    };
    auto poll = [job] { job->drainAll(); };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();   // before the callbacks: onDone may start the next op
        ev::Persistent ids(makeInt32Array(job->tokenIds));
        ev::Persistent info(makeDoneInfo(cancelled, error));
        // info.windows: [{ start, at }] — where each long-form window began,
        // `at` indexing the id array. Without it a caller that did not stream
        // cannot read the (per-window, restarting) timestamps as absolute
        // times. Absent for a short-form decode.
        if (!job->windowMarks.empty()) {
            const auto& marks = job->windowMarks;
            ev::Persistent ws(hostArrayOf(marks.size(), [&marks](size_t i) {
                ObjectBuilder o;
                o.set("start", marks[i].start_seconds);
                o.set("at", static_cast<double>(marks[i].first_token));
                return o.get();
            }));
            info.set(ev::setProperty(info.get(), "windows", ws.get()));
        }
        callCallback2(job->onDone.get(), ids.get(), info.get());
    };
    return launchAsyncJob(std::move(work), std::move(poll), std::move(done));
}

namespace {

HostWhisperModel* modelSelf(Value self) {
    return static_cast<HostWhisperModel*>(g_whisperModelClass.unwrap(self));
}
HostWhisperSession* sessionSelf(Value self) {
    return static_cast<HostWhisperSession*>(g_whisperSessionClass.unwrap(self));
}

void decorateWhisperModel(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        return ev::fromBool(w && w->model && w->model->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    auto cfgGetter = [&b](const char* name, int WhisperConfig::*field) {
        b.accessor(name, [field](Value self, std::span<const Value>) -> Value {
            auto* w = modelSelf(self);
            if (!w || !w->model) return ev::fromDouble(0);
            return ev::fromDouble(w->model->config().*field);
        });
    };
    cfgGetter("sampleRate",          &WhisperConfig::sample_rate);
    cfgGetter("numMelBins",          &WhisperConfig::num_mel_bins);
    cfgGetter("dModel",              &WhisperConfig::d_model);
    cfgGetter("maxSourcePositions",  &WhisperConfig::max_source_positions);
    cfgGetter("maxTargetPositions",  &WhisperConfig::max_target_positions);
    cfgGetter("vocabSize",           &WhisperConfig::vocab_size);
    cfgGetter("eosTokenId",          &WhisperConfig::eos_token_id);
    cfgGetter("decoderStartTokenId", &WhisperConfig::decoder_start_token_id);

    // transcribe(audio, promptIds|opts, opts?) -> Int32Array | AsyncHandle (opts.onDone)
    b.def("transcribe", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* w = modelSelf(self);
        if (!w) return ev::throwTypeError("transcribe: not a WhisperModel");
        return whisperTranscribe(self, w, nullptr, a, "transcribe");
    });
    b.def("createSession", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = modelSelf(self);
        if (!w) return ev::throwTypeError("createSession: not a WhisperModel");
        if (!w->model || !w->model->loaded()) return ev::throwError("createSession: model is not loaded");
        try {
            brotensor::DeviceScope scope(w->device);
            auto sw = std::make_unique<HostWhisperSession>();
            sw->model = w->model;
            sw->busy = w->busy;
            sw->device = w->device;
            sw->session = w->model->make_session();
            return g_whisperSessionClass.createInstance(std::move(sw));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("createSession: ") + e.what());
        }
    });
}

void decorateWhisperSession(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* s = sessionSelf(self);
        return ev::fromBool(s && s->model && s->model->loaded());
    });
    b.def("transcribe", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* s = sessionSelf(self);
        if (!s) return ev::throwTypeError("transcribe: not a WhisperSession");
        return whisperTranscribe(self, nullptr, s, a, "transcribe");
    });
    b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
        auto* s = sessionSelf(self);
        if (!s) return ev::throwTypeError("reset: not a WhisperSession");
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

void installSttWhisperClasses() {
    g_whisperTokenizerClass.install("WhisperTokenizer", 0, nullptr, decorateWhisperTokenizer);
    g_whisperSessionClass.install("WhisperSession", 0, nullptr, decorateWhisperSession);
    g_whisperModelClass.install("WhisperModel", 0, nullptr, decorateWhisperModel);
}

} // namespace brosoundml::api
