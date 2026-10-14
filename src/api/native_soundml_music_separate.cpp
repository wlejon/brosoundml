// bro.music.loadSeparator — HTDemucs (Demucs v4 htdemucs_ft) source
// separation: music in, drums / bass / other / vocals out, each stem from its
// own fine-tuned network (brosoundml/htdemucs.h). Also bro.music's shared
// stereo audio shapes (makeStereo / readStereo).
//
//   const sep = bro.music.loadSeparator(dir, { stems?, device?, onReady? });
//   sep.separate(audio, { stems?, sampleRate?, onProgress?, onDone? })
//     -> { names, stems: { <name>: { samples, channels: 2, frames, sampleRate } } }
//
// separate() is synchronous without onDone, an AsyncHandle with it. A
// separator runs one job at a time (ModelGate).
#include "soundml_loader.h"
#include "soundml_music_internal.h"

#include <brosoundml/htdemucs.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml::api {

namespace music {

Value makeStereo(const brosoundml::StereoAudio& a) {
    ObjectBuilder o;
    {
        ev::Persistent s(makeFloat32Array(a.samples));
        o.set("samples", s.get());
    }
    o.set("channels", ev::fromDouble(a.channels));
    o.set("frames", ev::fromDouble(a.frames));
    o.set("sampleRate", ev::fromDouble(a.sample_rate));
    return o.get();
}

bool readStereo(Value v, brosoundml::StereoAudio& out, std::string& err, int default_rate) {
    if (ev::isObject(v) && !ev::isTypedArray(v) && !ev::isFunction(v)) {
        ev::Persistent o(v);
        const int ch = getPropertyInt(o.get(), "channels", 1);
        brosoundml::AudioBuffer flat;
        if (!readAudioBuffer(o.get(), flat, err, default_rate)) return false;
        if (ch < 1 || flat.samples.size() % static_cast<size_t>(ch) != 0) {
            err = "audio.samples.length must be a multiple of audio.channels";
            return false;
        }
        out.channels = ch;
        out.frames = static_cast<int>(flat.samples.size() / static_cast<size_t>(ch));
        out.sample_rate = flat.sample_rate;
        out.samples = std::move(flat.samples);
    } else {
        brosoundml::AudioBuffer flat;
        if (!readAudioBuffer(v, flat, err, default_rate)) return false;
        out.channels = 1;
        out.frames = static_cast<int>(flat.samples.size());
        out.sample_rate = flat.sample_rate;
        out.samples = std::move(flat.samples);
    }
    if (out.frames <= 0 || out.sample_rate <= 0) { err = "audio is empty"; return false; }
    return true;
}

namespace {

HostClass g_separatorClass;

struct HostSeparator {
    std::shared_ptr<brosoundml::HtDemucs> model;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;
};

HostSeparator* separatorOf(Value v) {
    return g_separatorClass.isInstance(v) ? static_cast<HostSeparator*>(g_separatorClass.unwrap(v))
                                          : nullptr;
}

// A string or an array of strings; false when it is neither.
bool readStems(Value v, std::vector<std::string>& out) {
    out.clear();
    if (ev::isUndefined(v) || ev::isNull(v)) return true;
    if (ev::isString(v)) {
        out.push_back(ev::toUtf8(v));
        return true;
    }
    if (!ev::isObject(v) || ev::isFunction(v)) return false;
    ev::Persistent root(v);
    Value len = ev::getProperty(root.get(), "length");
    if (!ev::isNumber(len)) return false;
    const uint32_t n = static_cast<uint32_t>(ev::toDouble(len));
    for (uint32_t i = 0; i < n; ++i) {
        Value e = ev::getElement(root.get(), i);
        if (!ev::isString(e)) return false;
        out.push_back(ev::toUtf8(e));
    }
    return true;
}

Value makeSeparation(const brosoundml::Separation& s) {
    ObjectBuilder stems;
    for (std::size_t i = 0; i < s.names.size() && i < s.stems.size(); ++i) {
        ev::Persistent a(makeStereo(s.stems[i]));
        stems.set(s.names[i], a.get());
    }
    ev::Persistent st(stems.get());
    ev::Persistent names(makeStringArray(s.names));
    ObjectBuilder o;
    o.set("names", names.get());
    o.set("stems", st.get());
    return o.get();
}

// sep.separate(audio, opts?) -> { names, stems } (sync) | AsyncHandle (opts.onDone)
Value separatorSeparate(Value self, std::span<const Value> a) {
    ev::Persistent selfRoot(self);
    auto* w = separatorOf(selfRoot.get());
    if (!w || !w->model || !w->model->loaded()) return ev::throwError("separate: separator is not loaded");
    if (!hasArg(a, 0)) return ev::throwTypeError("separate(audio, opts?): audio required");
    ev::Persistent opts(isObjectArg(a, 1) ? a[1] : ev::undefined());
    const int rate = getPropertyInt(opts.get(), "sampleRate", 48000);
    if (rate <= 0) return ev::throwTypeError("separate: sampleRate must be positive");
    brosoundml::StereoAudio in;
    std::string err;
    if (!readStereo(a[0], in, err, rate)) return ev::throwTypeError("separate: " + err);
    if (hasProperty(opts.get(), "sampleRate")) in.sample_rate = rate;
    if (in.channels != 1 && in.channels != 2) return ev::throwTypeError("separate: audio must be mono or stereo");
    brosoundml::SeparateOptions so;
    if (!readStems(ev::isObject(opts.get()) ? ev::getProperty(opts.get(), "stems") : ev::undefined(), so.stems))
        return ev::throwTypeError("separate: stems must be a stem name or an array of them");
    const std::vector<std::string> have = w->model->loaded_stems();
    for (const std::string& s : so.stems) {
        if (std::find(have.begin(), have.end(), s) == have.end())
            return ev::throwTypeError("separate: stem '" + s + "' is not one this separator loaded");
    }
    ev::Persistent onDone = getFunctionOpt(opts.get(), "onDone");
    ev::Persistent onProgress = getFunctionOpt(opts.get(), "onProgress");
    if (w->busy.isBusy()) return ev::throwError("separate: an operation is already in flight on this separator");

    if (!ev::isFunction(onDone.get())) {
        try {
            brotensor::DeviceScope scope(w->device);
            return makeSeparation(w->model->separate(in, so));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("separate: ") + e.what());
        }
    }
    struct Job {
        std::shared_ptr<brosoundml::HtDemucs> model;
        brotensor::Device device;
        ModelGate gate;
        brosoundml::StereoAudio in;
        brosoundml::SeparateOptions so;
        brosoundml::Separation out;
        std::atomic<int> done{0}, total{0};
        int seen = 0;
        ev::Persistent self, onDone, onProgress;
    };
    auto job = std::make_shared<Job>();
    job->model = w->model;
    job->device = w->device;
    job->gate = w->busy;
    job->in = std::move(in);
    job->so = std::move(so);
    job->self = ev::Persistent(selfRoot.get());
    job->onDone = onDone;
    job->onProgress = onProgress;
    if (!w->busy.tryClaim()) return ev::throwError("separate: an operation is already in flight on this separator");
    auto work = [job](const std::atomic<bool>& cancel) {
        brotensor::DeviceScope scope(job->device);
        brosoundml::SeparateOptions so = job->so;
        so.cancel = cancelCheckOf(cancel);
        so.on_progress = [&](int d, int t) {
            job->total.store(t);
            job->done.store(d);
        };
        job->out = job->model->separate(job->in, so);
    };
    auto poll = [job] {
        if (!ev::isFunction(job->onProgress.get())) return;
        const int d = job->done.load();
        if (d == job->seen) return;
        job->seen = d;
        ObjectBuilder o;
        o.set("done", ev::fromDouble(d));
        o.set("total", ev::fromDouble(job->total.load()));
        ev::Persistent ob(o.get());
        callCallback1(job->onProgress.get(), ob.get());
    };
    auto done = [job](bool cancelled, const std::string& error) {
        job->gate.release();
        cancelled = cancelled || job->out.cancelled;
        ev::Persistent result(error.empty() && !cancelled ? makeSeparation(job->out) : ev::undefined());
        ev::Persistent info(makeDoneInfo(cancelled, error));
        callCallback2(job->onDone.get(), result.get(), info.get());
    };
    return launchAsyncJob(std::move(work), std::move(poll), std::move(done));
}

void decorateSeparator(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = separatorOf(self);
        return ev::fromBool(w && w->model && w->model->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = separatorOf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("busy", [](Value self, std::span<const Value>) -> Value {
        auto* w = separatorOf(self);
        return ev::fromBool(w && w->busy.isBusy());
    });
    b.accessor("sources", [](Value self, std::span<const Value>) -> Value {
        auto* w = separatorOf(self);
        return makeStringArray(w && w->model && w->model->loaded() ? w->model->config().sources
                                                                   : std::vector<std::string>{});
    });
    b.accessor("stems", [](Value self, std::span<const Value>) -> Value {
        auto* w = separatorOf(self);
        return makeStringArray(w && w->model ? w->model->loaded_stems() : std::vector<std::string>{});
    });
    b.accessor("sampleRate", [](Value self, std::span<const Value>) -> Value {
        auto* w = separatorOf(self);
        return ev::fromDouble(w && w->model && w->model->loaded() ? w->model->config().sample_rate : 0);
    });
    b.def("separate", 2, separatorSeparate);
}

}  // namespace

// bro.music.loadSeparator(dir, opts?) -> Separator | AsyncHandle (opts.onReady)
//   opts.stems: the networks to load (default all four; ['vocals'] loads one).
Value loadSeparator(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    ev::Persistent opts;
    if (!modelLoaderArgs("loadSeparator", args, dir, dev, opts)) return ev::undefined();
    std::vector<std::string> stems;
    if (!readStems(ev::isObject(opts.get()) ? ev::getProperty(opts.get(), "stems") : ev::undefined(), stems))
        return ev::throwTypeError("loadSeparator: stems must be a stem name or an array of them");
    return runModelLoader<HostSeparator>("loadSeparator", opts.get(), g_separatorClass, [dir, dev, stems] {
        auto w = std::make_unique<HostSeparator>();
        w->device = dev;
        w->model = std::make_shared<brosoundml::HtDemucs>();
        {
            brotensor::DeviceScope scope(dev);
            w->model->load(dir, dev, stems);
        }
        logInfo(std::string("[music] HTDemucs separator loaded on ") + deviceName(dev));
        return w;
    });
}

void installSeparatorClass() {
    g_separatorClass.install("Separator", 0, nullptr, decorateSeparator);
}

}  // namespace music

}  // namespace brosoundml::api
