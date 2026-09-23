// bro.diar — speaker diarization: streaming Sortformer (4-speaker
// FastConformer + Transformer head, arrival-order speaker cache) and the
// ClusterDiarizer (Sortformer VAD + ECAPA x-vectors + centered-cosine
// clustering, discovers the speaker count).
//
// Loading is GPU by default (CUDA > Metal > CPU by availability); opts.device
// picks explicitly and must be a string. Model methods run synchronously on
// the JS thread; bro.diar.diarize / clusterDiarize run on a work thread and
// report through opts.onDone(result|null, { cancelled, error? }). One op in
// flight per model (shared with its sessions): the gate is released before
// onDone runs so a callback may start the next op synchronously.
#include "soundml_loader.h"

#include <brosoundml/sortformer.h>
#include <brosoundml/cluster_diarizer.h>

namespace brosoundml::api {

namespace {

HostClass g_sortformerClass;
HostClass g_sortformerSessionClass;
HostClass g_clusterDiarizerClass;

struct HostSortformer {
    std::shared_ptr<brosoundml::Sortformer> model;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;  // shared with every session of this model
};

struct HostSortformerSession {
    std::shared_ptr<brosoundml::Sortformer> model;
    brosoundml::SortformerSession session;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;
};

struct HostClusterDiarizer {
    std::shared_ptr<brosoundml::ClusterDiarizer> model;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;
};

HostSortformer* sortformerOf(Value v) {
    return g_sortformerClass.isInstance(v) ? static_cast<HostSortformer*>(g_sortformerClass.unwrap(v)) : nullptr;
}
HostSortformerSession* sessionOf(Value v) {
    return g_sortformerSessionClass.isInstance(v)
        ? static_cast<HostSortformerSession*>(g_sortformerSessionClass.unwrap(v)) : nullptr;
}
HostClusterDiarizer* clusterOf(Value v) {
    return g_clusterDiarizerClass.isInstance(v)
        ? static_cast<HostClusterDiarizer*>(g_clusterDiarizerClass.unwrap(v)) : nullptr;
}

// { numFrames, numSpeakers, frameSeconds, probs: Float32Array(numFrames*numSpeakers) }
Value makeDiarization(int numFrames, int numSpeakers, double frameSeconds, const std::vector<float>& probs) {
    ObjectBuilder res;
    res.set("numFrames", ev::fromDouble(numFrames));
    res.set("numSpeakers", ev::fromDouble(numSpeakers));
    res.set("frameSeconds", ev::fromDouble(frameSeconds));
    {
        ev::Persistent arr(makeFloat32Array(probs));
        res.set("probs", arr.get());
    }
    return res.get();
}
Value makeDiarization(const brosoundml::Sortformer::Diarization& d) {
    return makeDiarization(d.num_frames, d.num_speakers, d.frame_seconds, d.probs);
}
Value makeDiarization(const brosoundml::ClusterDiarizer::Diarization& d) {
    return makeDiarization(d.num_frames, d.num_speakers, d.frame_seconds, d.probs);
}

// ClusterDiarizer::Config from a JS opts object; any missing key keeps the
// library default.
void parseClusterConfig(Value opts, brosoundml::ClusterDiarizer::Config& cfg) {
    if (!ev::isObject(opts)) return;
    ev::Persistent root(opts);  // every read below allocates
    getFloatOpt(root.get(), "vadThreshold", cfg.vad_threshold);
    getFloatOpt(root.get(), "windowSeconds", cfg.window_seconds);
    getFloatOpt(root.get(), "hopSeconds", cfg.hop_seconds);
    getFloatOpt(root.get(), "minWindowSeconds", cfg.min_window_seconds);
    getFloatOpt(root.get(), "clusterThreshold", cfg.cluster_threshold);
    getFloatOpt(root.get(), "minSpeakerSeconds", cfg.min_speaker_seconds);
    getIntOpt(root.get(), "maxSpeakers", cfg.max_speakers);
}

// ---------------------------------------------------------------------------
// Sortformer
// ---------------------------------------------------------------------------

// model.diarize(audio) -> Diarization (sync, whole clip)
Value sortformerDiarize(Value self, std::span<const Value> a) {
    auto* w = sortformerOf(self);
    if (!w) return ev::throwTypeError("diarize: not a Sortformer");
    if (!hasArg(a, 0)) return ev::throwTypeError("diarize(audio): audio required");
    if (w->busy.isBusy()) return ev::throwError("diarize: an operation is already in flight on this model");
    brosoundml::AudioBuffer audio;
    std::string err;
    if (!readAudioBuffer(a[0], audio, err)) return ev::throwTypeError("diarize: " + err);
    try {
        brotensor::DeviceScope scope(w->device);
        return makeDiarization(w->model->diarize(audio));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("diarize: ") + e.what());
    }
}

// model.createSession() -> SortformerSession (its own speaker cache; shares
// the model's weights and busy gate)
Value sortformerCreateSession(Value self, std::span<const Value>) {
    auto* w = sortformerOf(self);
    if (!w) return ev::throwTypeError("createSession: not a Sortformer");
    if (!w->model || !w->model->loaded()) return ev::throwError("createSession: model is not loaded");
    try {
        auto sw = std::make_unique<HostSortformerSession>();
        sw->model = w->model;
        sw->device = w->device;
        sw->busy = w->busy;
        {
            brotensor::DeviceScope scope(w->device);
            sw->session = w->model->make_session();
        }
        return g_sortformerSessionClass.createInstance(std::move(sw));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("createSession: ") + e.what());
    }
}

void decorateSortformer(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = sortformerOf(self);
        return ev::fromBool(w && w->model && w->model->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = sortformerOf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("busy", [](Value self, std::span<const Value>) -> Value {
        auto* w = sortformerOf(self);
        return ev::fromBool(w && w->busy.isBusy());
    });
    b.accessor("sampleRate", [](Value self, std::span<const Value>) -> Value {
        auto* w = sortformerOf(self);
        return ev::fromDouble(w && w->model ? w->model->config().sample_rate : 0);
    });
    b.accessor("numSpeakers", [](Value self, std::span<const Value>) -> Value {
        auto* w = sortformerOf(self);
        return ev::fromDouble(w && w->model ? w->model->config().num_spks : 0);
    });
    b.accessor("frameSeconds", [](Value self, std::span<const Value>) -> Value {
        auto* w = sortformerOf(self);
        return ev::fromDouble(w && w->model ? w->model->config().frame_seconds() : 0.0);
    });
    b.accessor("fcDModel", [](Value self, std::span<const Value>) -> Value {
        auto* w = sortformerOf(self);
        return ev::fromDouble(w && w->model ? w->model->config().fc_d_model : 0);
    });
    b.accessor("tfDModel", [](Value self, std::span<const Value>) -> Value {
        auto* w = sortformerOf(self);
        return ev::fromDouble(w && w->model ? w->model->config().tf_d_model : 0);
    });
    b.def("diarize", 1, sortformerDiarize);
    b.def("createSession", 0, sortformerCreateSession);
}

// ---------------------------------------------------------------------------
// SortformerSession — feed() runs synchronously on the JS thread: a live
// window is small and the forward is ~20x realtime on GPU. Sessions isolate
// state; every forward over one model shares its GPU stream, so feed()
// rejects while an async diarize holds the gate.
// ---------------------------------------------------------------------------

// session.feed(audio, isLast=false) -> Diarization. Buffers audio; on
// isLast=true runs the streaming loop over what was buffered and returns the
// frames it finalized (numFrames=0 while only buffering).
Value sessionFeed(Value self, std::span<const Value> a) {
    auto* sw = sessionOf(self);
    if (!sw) return ev::throwTypeError("feed: not a SortformerSession");
    if (!hasArg(a, 0)) return ev::throwTypeError("feed(audio, isLast?): audio required");
    if (sw->busy.isBusy()) return ev::throwError("feed: an operation is already in flight on this model");
    brosoundml::AudioBuffer audio;
    std::string err;
    if (!readAudioBuffer(a[0], audio, err)) return ev::throwTypeError("feed: " + err);
    const bool isLast = hasArg(a, 1) && ev::toBool(a[1]);
    try {
        brotensor::DeviceScope scope(sw->device);
        return makeDiarization(sw->model->feed(sw->session, audio, isLast));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("feed: ") + e.what());
    }
}

// session.reset() — clear the speaker cache for a fresh stream.
Value sessionReset(Value self, std::span<const Value>) {
    auto* sw = sessionOf(self);
    if (!sw) return ev::throwTypeError("reset: not a SortformerSession");
    if (sw->busy.isBusy()) return ev::throwError("reset: an operation is in flight on this model");
    try {
        brotensor::DeviceScope scope(sw->device);
        sw->model->reset(sw->session);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("reset: ") + e.what());
    }
    return ev::undefined();
}

void decorateSortformerSession(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = sessionOf(self);
        return ev::fromBool(w && w->model && w->model->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = sessionOf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("busy", [](Value self, std::span<const Value>) -> Value {
        auto* w = sessionOf(self);
        return ev::fromBool(w && w->busy.isBusy());
    });
    b.accessor("numSpeakers", [](Value self, std::span<const Value>) -> Value {
        auto* w = sessionOf(self);
        return ev::fromDouble(w && w->model ? w->model->config().num_spks : 0);
    });
    b.accessor("frameSeconds", [](Value self, std::span<const Value>) -> Value {
        auto* w = sessionOf(self);
        return ev::fromDouble(w && w->model ? w->model->config().frame_seconds() : 0.0);
    });
    b.def("feed", 2, sessionFeed);
    b.def("reset", 0, sessionReset);
}

// ---------------------------------------------------------------------------
// ClusterDiarizer
// ---------------------------------------------------------------------------

// model.diarize(audio, opts?) -> Diarization (sync; opts carries the Config
// knobs: clusterThreshold, vadThreshold, windowSeconds, hopSeconds,
// minWindowSeconds, minSpeakerSeconds, maxSpeakers)
Value clusterDiarizeSync(Value self, std::span<const Value> a) {
    auto* w = clusterOf(self);
    if (!w) return ev::throwTypeError("diarize: not a ClusterDiarizer");
    if (!hasArg(a, 0)) return ev::throwTypeError("diarize(audio, opts?): audio required");
    if (w->busy.isBusy()) return ev::throwError("diarize: an operation is already in flight on this model");
    brosoundml::AudioBuffer audio;
    std::string err;
    if (!readAudioBuffer(a[0], audio, err)) return ev::throwTypeError("diarize: " + err);
    brosoundml::ClusterDiarizer::Config cfg;
    if (hasArg(a, 1)) parseClusterConfig(a[1], cfg);
    try {
        brotensor::DeviceScope scope(w->device);
        return makeDiarization(w->model->diarize(audio, cfg));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("diarize: ") + e.what());
    }
}

void decorateClusterDiarizer(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = clusterOf(self);
        return ev::fromBool(w && w->model && w->model->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = clusterOf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("busy", [](Value self, std::span<const Value>) -> Value {
        auto* w = clusterOf(self);
        return ev::fromBool(w && w->busy.isBusy());
    });
    b.def("diarize", 2, clusterDiarizeSync);
}

// ---------------------------------------------------------------------------
// bro.diar namespace
// ---------------------------------------------------------------------------

Value diarInit(Value, std::span<const Value>) {
    try {
        brotensor::init();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.diar.init: ") + e.what());
    }
    return ev::undefined();
}

// bro.diar.loadSortformer(modelDir, opts?) -> Sortformer | AsyncHandle
Value loadSortformer(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    ev::Persistent opts;
    if (!modelLoaderArgs("loadSortformer", args, dir, dev, opts)) return ev::undefined();
    return runModelLoader<HostSortformer>("loadSortformer", opts.get(), g_sortformerClass, [dir, dev] {
        auto w = std::make_unique<HostSortformer>();
        w->device = dev;
        w->model = std::make_shared<brosoundml::Sortformer>();
        {
            brotensor::DeviceScope scope(dev);
            w->model->load(dir, dev);
        }
        logInfo(std::string("[diar] Sortformer loaded on ") + deviceName(dev));
        return w;
    });
}

// bro.diar.loadClusterDiarizer(sortformerDir, speakerEncoderDir, opts?)
//   -> ClusterDiarizer | AsyncHandle
Value loadClusterDiarizer(Value, std::span<const Value> args) {
    if (!isStringArg(args, 0) || !isStringArg(args, 1))
        return ev::throwTypeError("loadClusterDiarizer(sortformerDir, speakerEncoderDir, opts?): two paths required");
    const std::string sortformerDir = resolvePath(strAt(args, 0));
    const std::string encoderDir = resolvePath(strAt(args, 1));
    brotensor::Device dev = brotensor::Device::CPU;
    ev::Persistent opts;
    if (!loaderDevice("loadClusterDiarizer", args, 2, dev, opts)) return ev::undefined();
    return runModelLoader<HostClusterDiarizer>("loadClusterDiarizer", opts.get(), g_clusterDiarizerClass,
                                               [sortformerDir, encoderDir, dev] {
        auto w = std::make_unique<HostClusterDiarizer>();
        w->device = dev;
        w->model = std::make_shared<brosoundml::ClusterDiarizer>();
        {
            brotensor::DeviceScope scope(dev);
            w->model->load(sortformerDir, encoderDir, dev);
        }
        logInfo(std::string("[diar] ClusterDiarizer loaded on ") + deviceName(dev));
        return w;
    });
}

// Whole-clip diarization on a work thread; shared by bro.diar.diarize and
// bro.diar.clusterDiarize. `run` is the model call, `gate` the claimed gate.
template <typename Result>
Value launchDiarJob(Value modelVal, ModelGate gate, brotensor::Device device, Value opts,
                    std::function<Result()> run) {
    struct Job {
        std::function<Result()> run;
        Result result;
        ModelGate gate;
        brotensor::Device device = brotensor::Device::CPU;
        ev::Persistent onDone, modelRef;
    };
    auto job = std::make_shared<Job>();
    job->run = std::move(run);
    job->gate = gate;
    job->device = device;
    job->modelRef = ev::Persistent(modelVal);  // keep the model alive; rooted before any allocation
    job->onDone = getFunctionOpt(opts, "onDone");
    auto work = [job](const std::atomic<bool>&) {
        brotensor::DeviceScope scope(job->device);
        job->result = job->run();
    };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        if (!ev::isFunction(job->onDone.get())) return;
        ev::Persistent res((error.empty() && !cancelled) ? makeDiarization(job->result) : ev::null());
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), res.get(), info.get());
    };
    return launchAsyncJob(std::move(work), nullptr, std::move(done));
}

// bro.diar.diarize(model, audio, opts?) -> AsyncHandle
//   opts.onDone(result|null, { cancelled, error? })
Value diarDiarize(Value, std::span<const Value> a) {
    if (a.size() < 2) return ev::throwTypeError("diarize(model, audio, opts?): model and audio required");
    auto* w = sortformerOf(a[0]);
    if (!w) return ev::throwTypeError("diarize: first arg must be a Sortformer");
    auto audio = std::make_shared<brosoundml::AudioBuffer>();
    std::string err;
    if (!readAudioBuffer(a[1], *audio, err)) return ev::throwTypeError("diarize: " + err);
    Value opts = isObjectArg(a, 2) ? a[2] : ev::undefined();
    if (!w->busy.tryClaim()) return ev::throwError("diarize: an operation is already in flight on this model");
    std::shared_ptr<brosoundml::Sortformer> model = w->model;
    return launchDiarJob<brosoundml::Sortformer::Diarization>(a[0], w->busy, w->device, opts,
        [model, audio] { return model->diarize(*audio); });
}

// bro.diar.clusterDiarize(model, audio, opts?) -> AsyncHandle
//   opts.onDone(result|null, { cancelled, error? }) plus the Config knobs.
Value diarClusterDiarize(Value, std::span<const Value> a) {
    if (a.size() < 2) return ev::throwTypeError("clusterDiarize(model, audio, opts?): model and audio required");
    auto* w = clusterOf(a[0]);
    if (!w) return ev::throwTypeError("clusterDiarize: first arg must be a ClusterDiarizer");
    auto audio = std::make_shared<brosoundml::AudioBuffer>();
    std::string err;
    if (!readAudioBuffer(a[1], *audio, err)) return ev::throwTypeError("clusterDiarize: " + err);
    ev::Persistent opts(isObjectArg(a, 2) ? a[2] : ev::undefined());
    brosoundml::ClusterDiarizer::Config cfg;
    parseClusterConfig(opts.get(), cfg);
    if (!w->busy.tryClaim()) return ev::throwError("clusterDiarize: an operation is already in flight on this model");
    std::shared_ptr<brosoundml::ClusterDiarizer> model = w->model;
    return launchDiarJob<brosoundml::ClusterDiarizer::Diarization>(a[0], w->busy, w->device, opts.get(),
        [model, audio, cfg] { return model->diarize(*audio, cfg); });
}

}  // namespace

void installDiar(ObjectBuilder& bro) {
    g_sortformerClass.install("Sortformer", 0, nullptr, decorateSortformer);
    g_sortformerSessionClass.install("SortformerSession", 0, nullptr, decorateSortformerSession);
    g_clusterDiarizerClass.install("ClusterDiarizer", 0, nullptr, decorateClusterDiarizer);

    ObjectBuilder diar;
    diar.def("init", 0, diarInit);
    diar.def("loadSortformer", 2, loadSortformer);
    diar.def("loadClusterDiarizer", 3, loadClusterDiarizer);
    diar.def("diarize", 3, diarDiarize);
    diar.def("clusterDiarize", 3, diarClusterDiarize);
    diar.set("Sortformer", g_sortformerClass.constructor());
    diar.set("SortformerSession", g_sortformerSessionClass.constructor());
    diar.set("ClusterDiarizer", g_clusterDiarizerClass.constructor());
    bro.set("diar", diar.get());
}

} // namespace brosoundml::api
