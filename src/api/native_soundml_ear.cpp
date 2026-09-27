// bro.ear.loadClap — CLAP (laion/larger_clap_general): score a sound clip
// against text prompts, so a script can judge how a clip sounds without
// listening to it. See include/brosoundml/clap.h and docs/clap.md.
//
// bro.ear is shared: broaudio mounts measure / compare / spectrogram on it
// earlier in the same realm. installEar adds loadClap and ClapModel to the
// object that is already there and creates bro.ear only when it is absent —
// it never replaces the object.
//
//   bro.ear.loadClap(dir?, opts?)  -> ClapModel | AsyncHandle (opts.onReady)
//     dir defaults to <brosoundml>/weights/clap (the asset root that
//     bro.tts.setAssetRoot set, else ../brosoundml, ./brosoundml, .).
//   model.score(clip, prompts, opts?)
//     -> { scores, similarities, logits, embedding, bestIndex, best }
//   model.embedAudio(clip, opts?)       -> Float32Array(512)
//   model.embedText(prompts, opts?)     -> Float32Array(512) | Float32Array[]
//   model.scoreEmbedding(embedding, prompts) -> the score() result
//   model.dispose()
// score / embedAudio / embedText run on a work thread when opts.onDone is a
// function (onDone(result, { cancelled, error? }); the call answers an
// AsyncHandle), and block otherwise.
#include "soundml_loader.h"

#include <brosoundml/clap.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>

namespace brosoundml::api {

std::string ttsAssetRootOverride();   // native_soundml_tts.cpp (bro.tts.setAssetRoot)

namespace {

HostClass g_clapClass;

struct HostClap {
    std::shared_ptr<brosoundml::Clap> model;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;
};

HostClap* clapOf(Value v) {
    return g_clapClass.isInstance(v) ? static_cast<HostClap*>(g_clapClass.unwrap(v)) : nullptr;
}

std::string defaultClapDir(std::vector<std::string>& tried) {
    namespace fs = std::filesystem;
    std::vector<std::string> roots;
    if (const std::string r = ttsAssetRootOverride(); !r.empty()) roots.push_back(r);
    for (const char* r : {"../brosoundml", "./brosoundml", "."}) roots.emplace_back(r);
    std::error_code ec;
    for (const std::string& r : roots) {
        const std::string d = r + "/weights/clap";
        tried.push_back(d);
        if (fs::exists(d + "/model.safetensors", ec)) return d;
    }
    return "";
}

// ── argument readers ──────────────────────────────────────────────────────────

// A clip: a WAV path; a Float32Array (at `defaultRate`); { samples, sampleRate };
// or a Web Audio AudioBuffer (getChannelData / numberOfChannels / sampleRate),
// whose channels are averaged to mono.
// `root` is re-read after every allocating call (embed.h GC contract).
bool readClip(const ev::Persistent& root, int defaultRate, brosoundml::AudioBuffer& out, std::string& err) {
    if (ev::isString(root.get())) {
        try {
            out = brosoundml::read_wav(resolvePath(ev::toUtf8(root.get())));
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
        return true;
    }
    if (ev::isObject(root.get()) && !ev::isTypedArray(root.get()) && !ev::isFunction(root.get())) {
        ev::Persistent get(ev::getProperty(root.get(), "getChannelData"));
        if (ev::isFunction(get.get())) {
            const int channels = getPropertyInt(root.get(), "numberOfChannels", 1);
            const int rate = getPropertyInt(root.get(), "sampleRate", 0);
            if (channels < 1 || rate <= 0) {
                err = "AudioBuffer needs numberOfChannels >= 1 and a positive sampleRate";
                return false;
            }
            out.samples.clear();
            out.sample_rate = rate;
            for (int c = 0; c < channels; ++c) {
                const Value args[1] = {ev::fromDouble(c)};
                ev::CallResult r = ev::call(get.get(), root.get(), std::span<const Value>(args, 1));
                bool ok = false;
                const std::vector<float> ch = r.thrown ? std::vector<float>{} : readFloat32Array(r.value, &ok);
                if (!ok) {
                    err = "AudioBuffer.getChannelData(" + std::to_string(c) + ") did not return a Float32Array";
                    return false;
                }
                if (c == 0) out.samples.assign(ch.size(), 0.0f);
                if (ch.size() != out.samples.size()) {
                    err = "AudioBuffer channels differ in length";
                    return false;
                }
                for (std::size_t i = 0; i < ch.size(); ++i) out.samples[i] += ch[i];
            }
            if (channels > 1)
                for (float& s : out.samples) s /= static_cast<float>(channels);
            return true;
        }
    }
    return readAudioBuffer(root.get(), out, err, defaultRate);
}

// One prompt: text to encode, or a cached 512-d text embedding.
struct Prompt {
    std::string text;
    std::vector<float> embedding;
};

// A string, or an array of strings / Float32Array text embeddings.
bool readPrompts(Value v, int dim, std::vector<Prompt>& out, std::string& err) {
    out.clear();
    auto one = [&](Value e) -> bool {
        Prompt p;
        if (ev::isString(e)) {
            p.text = ev::toUtf8(e);
        } else if (ev::isTypedArray(e)) {
            p.embedding = readFloat32Array(e);
            if (static_cast<int>(p.embedding.size()) != dim) {
                err = "a cached prompt embedding must be a Float32Array of " + std::to_string(dim);
                return false;
            }
        } else {
            err = "prompts must be strings or Float32Array text embeddings";
            return false;
        }
        out.push_back(std::move(p));
        return true;
    };
    if (ev::isString(v)) return one(v);
    if (!ev::isObject(v) || ev::isTypedArray(v) || ev::isFunction(v)) {
        err = "prompts must be a string or an array";
        return false;
    }
    ev::Persistent root(v);
    Value len = ev::getProperty(root.get(), "length");
    if (!ev::isNumber(len)) {
        err = "prompts must be a string or an array";
        return false;
    }
    const uint32_t n = static_cast<uint32_t>(ev::toDouble(len));
    if (n == 0) {
        err = "prompts must not be empty";
        return false;
    }
    for (uint32_t i = 0; i < n; ++i)
        if (!one(ev::getElement(root.get(), i))) return false;
    return true;
}

struct ClipOpts {
    int sampleRate = 48000;
    brosoundml::ClapAudioOptions audio;
};

// opts.sampleRate (for a bare Float32Array), opts.long ('mean' | 'crop'),
// opts.cropAt (seconds; default: centred), opts.pad ('silence' | 'repeat' |
// 'auto'; default auto: silence under 2 s, repeat from 2 s to 10 s).
// False + err on a bad value.
bool readClipOpts(const ev::Persistent& root, ClipOpts& o, std::string& err) {
    if (!ev::isObject(root.get())) return true;
    // Each read may allocate (and move the object): take root.get() afresh.
    getIntOpt(root.get(), "sampleRate", o.sampleRate);
    if (o.sampleRate <= 0) {
        err = "opts.sampleRate must be positive";
        return false;
    }
    const std::string mode = getPropertyString(root.get(), "long", "mean");
    if (mode == "mean") {
        o.audio.long_mode = brosoundml::ClapLongMode::Mean;
    } else if (mode == "crop") {
        o.audio.long_mode = brosoundml::ClapLongMode::Crop;
    } else {
        err = "opts.long must be 'mean' or 'crop'";
        return false;
    }
    if (hasProperty(root.get(), "pad")) {
        const std::string pad = getPropertyString(root.get(), "pad", "");
        if (pad == "silence") {
            o.audio.pad = brosoundml::ClapPad::Silence;
        } else if (pad == "repeat") {
            o.audio.pad = brosoundml::ClapPad::Repeat;
        } else if (pad == "auto") {
            o.audio.pad = brosoundml::ClapPad::Auto;
        } else {
            err = "opts.pad must be 'silence', 'repeat' or 'auto'";
            return false;
        }
    }
    if (hasProperty(root.get(), "cropAt")) {
        const double s = getPropertyDouble(root.get(), "cropAt", 0.0);
        if (!(s >= 0.0)) {
            err = "opts.cropAt must be a non-negative number of seconds";
            return false;
        }
        o.audio.crop_offset = static_cast<int>(std::lround(s * 48000.0));
    }
    return true;
}

// ── compute (work-thread safe: no JS) ─────────────────────────────────────────

struct ScoreOut {
    brosoundml::ClapScore score;
    std::vector<std::string> texts;
};

ScoreOut runScore(const brosoundml::Clap& m, std::vector<float> audioEmbedding, const std::vector<Prompt>& prompts) {
    ScoreOut out;
    const int dim = m.config().projection_dim;
    std::vector<float> text;
    text.reserve(prompts.size() * static_cast<std::size_t>(dim));
    for (const Prompt& p : prompts) {
        const std::vector<float> e = p.embedding.empty() ? m.embed_text(p.text) : p.embedding;
        text.insert(text.end(), e.begin(), e.end());
        out.texts.push_back(p.text);
    }
    out.score = brosoundml::Clap::score_embeddings(audioEmbedding, text, static_cast<int>(prompts.size()),
                                                   m.logit_scale());
    return out;
}

Value makeScoreResult(const ScoreOut& s) {
    ObjectBuilder res;
    {
        ev::Persistent a(makeFloat32Array(s.score.probability));
        res.set("scores", a.get());
    }
    {
        ev::Persistent a(makeFloat32Array(s.score.similarity));
        res.set("similarities", a.get());
    }
    {
        ev::Persistent a(makeFloat32Array(s.score.logits));
        res.set("logits", a.get());
    }
    {
        ev::Persistent a(makeFloat32Array(s.score.audio_embedding));
        res.set("embedding", a.get());
    }
    int best = -1;
    for (std::size_t i = 0; i < s.score.probability.size(); ++i)
        if (best < 0 || s.score.probability[i] > s.score.probability[static_cast<std::size_t>(best)])
            best = static_cast<int>(i);
    res.set("bestIndex", static_cast<double>(best));
    if (best >= 0 && !s.texts[static_cast<std::size_t>(best)].empty())
        res.set("best", s.texts[static_cast<std::size_t>(best)]);
    else
        res.set("best", ev::null());
    return res.get();
}

Value makeTextResult(const std::vector<std::vector<float>>& embs, bool single) {
    if (single) return makeFloat32Array(embs.empty() ? std::vector<float>{} : embs[0]);
    return hostArrayOf(embs.size(), [&embs](size_t i) { return makeFloat32Array(embs[i]); });
}

// ── the sync-or-async runner shared by score / embedAudio / embedText ────────

template <typename R>
Value runOp(const char* fn, Value self, HostClap* w, Value opts, std::function<R(const brosoundml::Clap&)> compute,
            std::function<Value(const R&)> toJs) {
    const std::string pre = std::string(fn) + ": ";
    ev::Persistent selfRoot(self);
    ev::Persistent onDone = getFunctionOpt(opts, "onDone");
    if (w->busy.isBusy()) return ev::throwError(pre + "an operation is already in flight on this model");
    std::shared_ptr<brosoundml::Clap> model = w->model;
    const brotensor::Device dev = w->device;
    if (!ev::isFunction(onDone.get())) {
        try {
            R r;
            {
                brotensor::DeviceScope scope(dev);
                r = compute(*model);
            }
            return toJs(r);
        } catch (const std::exception& e) {
            return ev::throwError(pre + e.what());
        }
    }
    struct Job {
        std::shared_ptr<brosoundml::Clap> model;
        brotensor::Device dev;
        ModelGate gate;
        std::function<R(const brosoundml::Clap&)> compute;
        std::function<Value(const R&)> toJs;
        R result{};
        ev::Persistent onDone, modelRef;
    };
    auto job = std::make_shared<Job>();
    job->model = std::move(model);
    job->dev = dev;
    job->gate = w->busy;
    job->compute = std::move(compute);
    job->toJs = std::move(toJs);
    job->onDone = std::move(onDone);
    if (!job->gate.tryClaim()) return ev::throwError(pre + "an operation is already in flight on this model");
    job->modelRef = ev::Persistent(selfRoot.get());
    auto work = [job](const std::atomic<bool>&) {
        brotensor::DeviceScope scope(job->dev);
        job->result = job->compute(*job->model);
    };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        ev::Persistent res(error.empty() && !cancelled ? job->toJs(job->result) : ev::null());
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), res.get(), info.get());
    };
    return launchAsyncJob(std::move(work), nullptr, std::move(done));
}

HostClap* loadedSelf(Value self, const char* fn) {
    HostClap* w = clapOf(self);
    if (!w) {
        ev::throwTypeError(std::string(fn) + ": not a ClapModel");
        return nullptr;
    }
    if (!w->model || !w->model->loaded()) {
        ev::throwError(std::string(fn) + ": model is not loaded (disposed?)");
        return nullptr;
    }
    return w;
}

// model.score(clip, prompts, opts?)
Value clapScore(Value self, std::span<const Value> a) {
    HostClap* w = loadedSelf(self, "score");
    if (!w) return ev::undefined();
    if (a.size() < 2) return ev::throwTypeError("score(clip, prompts, opts?): clip and prompts required");
    ev::Persistent selfRoot(self), clipRoot(a[0]), promptsRoot(a[1]);
    ev::Persistent opts(isObjectArg(a, 2) ? a[2] : ev::undefined());
    std::string err;
    ClipOpts co;
    if (!readClipOpts(opts, co, err)) return ev::throwTypeError("score: " + err);
    auto clip = std::make_shared<brosoundml::AudioBuffer>();
    if (!readClip(clipRoot, co.sampleRate, *clip, err)) return ev::throwTypeError("score: " + err);
    if (clip->samples.empty()) return ev::throwTypeError("score: clip is empty");
    auto prompts = std::make_shared<std::vector<Prompt>>();
    if (!readPrompts(promptsRoot.get(), w->model->config().projection_dim, *prompts, err))
        return ev::throwTypeError("score: " + err);
    const brosoundml::ClapAudioOptions ao = co.audio;
    return runOp<ScoreOut>(
        "score", selfRoot.get(), w, opts.get(),
        [clip, prompts, ao](const brosoundml::Clap& m) { return runScore(m, m.embed_audio(*clip, ao), *prompts); },
        [](const ScoreOut& s) { return makeScoreResult(s); });
}

// model.scoreEmbedding(embedding, prompts, opts?) — score a cached audio embedding.
Value clapScoreEmbedding(Value self, std::span<const Value> a) {
    HostClap* w = loadedSelf(self, "scoreEmbedding");
    if (!w) return ev::undefined();
    if (a.size() < 2) return ev::throwTypeError("scoreEmbedding(embedding, prompts, opts?): embedding and prompts required");
    ev::Persistent selfRoot(self), promptsRoot(a[1]);
    ev::Persistent opts(isObjectArg(a, 2) ? a[2] : ev::undefined());
    const int dim = w->model->config().projection_dim;
    auto emb = std::make_shared<std::vector<float>>(readFloat32Array(a[0]));
    if (static_cast<int>(emb->size()) != dim)
        return ev::throwTypeError("scoreEmbedding: embedding must be a Float32Array of " + std::to_string(dim));
    auto prompts = std::make_shared<std::vector<Prompt>>();
    std::string err;
    if (!readPrompts(promptsRoot.get(), dim, *prompts, err)) return ev::throwTypeError("scoreEmbedding: " + err);
    return runOp<ScoreOut>(
        "scoreEmbedding", selfRoot.get(), w, opts.get(),
        [emb, prompts](const brosoundml::Clap& m) { return runScore(m, *emb, *prompts); },
        [](const ScoreOut& s) { return makeScoreResult(s); });
}

// model.embedAudio(clip, opts?) -> Float32Array(512)
Value clapEmbedAudio(Value self, std::span<const Value> a) {
    HostClap* w = loadedSelf(self, "embedAudio");
    if (!w) return ev::undefined();
    if (!hasArg(a, 0)) return ev::throwTypeError("embedAudio(clip, opts?): clip required");
    ev::Persistent selfRoot(self), clipRoot(a[0]);
    ev::Persistent opts(isObjectArg(a, 1) ? a[1] : ev::undefined());
    std::string err;
    ClipOpts co;
    if (!readClipOpts(opts, co, err)) return ev::throwTypeError("embedAudio: " + err);
    auto clip = std::make_shared<brosoundml::AudioBuffer>();
    if (!readClip(clipRoot, co.sampleRate, *clip, err)) return ev::throwTypeError("embedAudio: " + err);
    if (clip->samples.empty()) return ev::throwTypeError("embedAudio: clip is empty");
    const brosoundml::ClapAudioOptions ao = co.audio;
    return runOp<std::vector<float>>(
        "embedAudio", selfRoot.get(), w, opts.get(),
        [clip, ao](const brosoundml::Clap& m) { return m.embed_audio(*clip, ao); },
        [](const std::vector<float>& e) { return makeFloat32Array(e); });
}

// model.embedText(prompts, opts?) -> Float32Array (a string) | Float32Array[] (an array)
Value clapEmbedText(Value self, std::span<const Value> a) {
    HostClap* w = loadedSelf(self, "embedText");
    if (!w) return ev::undefined();
    if (!hasArg(a, 0)) return ev::throwTypeError("embedText(prompts, opts?): a string or an array of strings required");
    ev::Persistent selfRoot(self), promptsRoot(a[0]);
    ev::Persistent opts(isObjectArg(a, 1) ? a[1] : ev::undefined());
    const bool single = ev::isString(promptsRoot.get());
    auto prompts = std::make_shared<std::vector<Prompt>>();
    std::string err;
    if (!readPrompts(promptsRoot.get(), w->model->config().projection_dim, *prompts, err))
        return ev::throwTypeError("embedText: " + err);
    for (const Prompt& p : *prompts)
        if (!p.embedding.empty()) return ev::throwTypeError("embedText: prompts must be strings");
    return runOp<std::vector<std::vector<float>>>(
        "embedText", selfRoot.get(), w, opts.get(),
        [prompts](const brosoundml::Clap& m) {
            std::vector<std::vector<float>> out;
            for (const Prompt& p : *prompts) out.push_back(m.embed_text(p.text));
            return out;
        },
        [single](const std::vector<std::vector<float>>& e) { return makeTextResult(e, single); });
}

void decorateClap(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = clapOf(self);
        return ev::fromBool(w && w->model && w->model->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = clapOf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("sampleRate", [](Value self, std::span<const Value>) -> Value {
        auto* w = clapOf(self);
        return ev::fromDouble(w && w->model && w->model->loaded() ? w->model->config().sample_rate : 0);
    });
    b.accessor("windowSeconds", [](Value self, std::span<const Value>) -> Value {
        auto* w = clapOf(self);
        if (!w || !w->model || !w->model->loaded()) return ev::fromDouble(0);
        const auto& c = w->model->config();
        return ev::fromDouble(static_cast<double>(c.max_samples) / c.sample_rate);
    });
    b.accessor("embeddingSize", [](Value self, std::span<const Value>) -> Value {
        auto* w = clapOf(self);
        return ev::fromDouble(w && w->model && w->model->loaded() ? w->model->config().projection_dim : 0);
    });
    b.accessor("logitScale", [](Value self, std::span<const Value>) -> Value {
        auto* w = clapOf(self);
        return ev::fromDouble(w && w->model && w->model->loaded() ? w->model->logit_scale() : 0);
    });
    b.def("score", 3, clapScore);
    b.def("scoreEmbedding", 3, clapScoreEmbedding);
    b.def("embedAudio", 2, clapEmbedAudio);
    b.def("embedText", 2, clapEmbedText);
    b.def("dispose", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = clapOf(self);
        if (!w) return ev::throwTypeError("dispose: not a ClapModel");
        if (w->busy.isBusy()) return ev::throwError("dispose: an operation is in flight on this model");
        w->model.reset();
        return ev::undefined();
    });
}

// bro.ear.loadClap(dir?, opts?) -> ClapModel | AsyncHandle (opts.onReady)
Value loadClap(Value, std::span<const Value> args) {
    std::string dir;
    std::size_t optsIndex = 0;
    if (isStringArg(args, 0)) {
        dir = resolvePath(strAt(args, 0));
        optsIndex = 1;
    } else if (hasArg(args, 0) && !isObjectArg(args, 0) && !ev::isNull(args[0])) {
        return ev::throwTypeError("loadClap(dir?, opts?): dir must be a path string");
    } else if (hasArg(args, 0) && ev::isNull(args[0])) {
        optsIndex = 1;
    }
    brotensor::Device dev = brotensor::Device::CPU;
    ev::Persistent opts;
    if (!loaderDevice("loadClap", args, optsIndex, dev, opts)) return ev::undefined();
    if (dir.empty()) {
        std::vector<std::string> tried;
        dir = defaultClapDir(tried);
        if (dir.empty()) {
            std::string list;
            for (const std::string& t : tried) list += (list.empty() ? "" : ", ") + t;
            return ev::throwError("loadClap: no CLAP weights found (tried " + list +
                                  "); run brosoundml's scripts/download-clap.sh + convert-clap.py or pass a dir");
        }
    }
    return runModelLoader<HostClap>("loadClap", opts.get(), g_clapClass, [dir, dev] {
        auto w = std::make_unique<HostClap>();
        w->device = dev;
        w->model = std::make_shared<brosoundml::Clap>();
        w->model->load(dir, dev);
        logInfo(std::string("[ear] CLAP loaded on ") + deviceName(dev) + " from " + dir);
        return w;
    });
}

}  // namespace

void installEar(ObjectBuilder& bro) {
    g_clapClass.install("ClapModel", 0, nullptr, decorateClap);

    // Mount onto the bro.ear broaudio already installed, or create it.
    ev::Persistent existing(ev::getProperty(bro.get(), "ear"));
    const bool had = ev::isObject(existing.get()) && !ev::isFunction(existing.get());
    ObjectBuilder ear = had ? ObjectBuilder(existing.get()) : ObjectBuilder();
    ear.def("loadClap", 2, loadClap);
    ear.set("ClapModel", g_clapClass.constructor());
    if (!had) bro.set("ear", ear.get());
}

}  // namespace brosoundml::api
