#include "host_soundml_internal.h"

#include <brosoundml/sortformer.h>
#include <brosoundml/cluster_diarizer.h>
#include <brotensor/runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace brosoundml::api {

namespace {

HostClass g_sortformerClass;
HostClass g_sortformerSessionClass;
HostClass g_clusterDiarizerClass;

struct HostSortformer {
    std::shared_ptr<brosoundml::Sortformer> model;
    std::string device = "CPU";
    bool busy = false;
};

struct HostSortformerSession {
    std::shared_ptr<brosoundml::Sortformer> model;
    std::unique_ptr<brosoundml::SortformerSession> session;
    std::string device = "CPU";
    bool busy = false;
};

struct HostClusterDiarizer {
    std::shared_ptr<brosoundml::ClusterDiarizer> model;
    std::string device = "CPU";
    bool busy = false;
};

Value makeDiarResult(int numFrames, int numSpeakers, double frameSeconds,
                     const std::vector<float>& probs) {
    ObjectBuilder res;
    res.set("numFrames", static_cast<double>(numFrames));
    res.set("numSpeakers", static_cast<double>(numSpeakers));
    res.set("frameSeconds", frameSeconds);
    res.set("probs", makeFloat32Array(probs));
    return res.get();
}

} // namespace

void installDiar(ObjectBuilder& bro) {
    // -----------------------------------------------------------------------
    // SortformerSession
    // -----------------------------------------------------------------------
    g_sortformerSessionClass.install("SortformerSession", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostSortformerSession();
            return g_sortformerSessionClass.make(h, [](void* p) { delete static_cast<HostSortformerSession*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("device", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostSortformerSession*>(g_sortformerSessionClass.unwrap(self));
                return ev::fromUtf8(h ? h->device : "CPU");
            });
            b.accessor("busy", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostSortformerSession*>(g_sortformerSessionClass.unwrap(self));
                return ev::fromBool(h && h->busy);
            });
            b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostSortformerSession*>(g_sortformerSessionClass.unwrap(self));
                if (h && h->model && h->session) {
                    try {
                        h->model->reset(*h->session);
                    } catch (...) {}
                }
                return ev::undefined();
            });
            b.def("feed", 2, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostSortformerSession*>(g_sortformerSessionClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));
                bool isLast = boolAt(a, 1, true);

                int numFrames = 0;
                int numSpeakers = 4;
                double frameSeconds = 0.08;
                std::vector<float> probs;

                if (h && h->model && h->session && !audio.empty()) {
                    try {
                        brosoundml::AudioBuffer ab;
                        ab.samples = std::move(audio);
                        ab.sample_rate = 16000;
                        auto d = h->model->feed(*h->session, ab, isLast);
                        numFrames = d.num_frames;
                        numSpeakers = d.num_speakers;
                        frameSeconds = d.frame_seconds;
                        probs = std::move(d.probs);
                    } catch (...) {}
                }
                return makeDiarResult(numFrames, numSpeakers, frameSeconds, probs);
            });
        });

    // -----------------------------------------------------------------------
    // Sortformer
    // -----------------------------------------------------------------------
    g_sortformerClass.install("Sortformer", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostSortformer();
            return g_sortformerClass.make(h, [](void* p) { delete static_cast<HostSortformer*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("device", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostSortformer*>(g_sortformerClass.unwrap(self));
                return ev::fromUtf8(h ? h->device : "CPU");
            });
            b.accessor("busy", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostSortformer*>(g_sortformerClass.unwrap(self));
                return ev::fromBool(h && h->busy);
            });
            b.def("createSession", 0, [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostSortformer*>(g_sortformerClass.unwrap(self));
                auto* sess = new HostSortformerSession();
                sess->model = h ? h->model : nullptr;
                sess->device = h ? h->device : "CPU";
                if (h && h->model) {
                    try {
                        sess->session = std::make_unique<brosoundml::SortformerSession>(h->model->make_session());
                    } catch (...) {}
                }
                return g_sortformerSessionClass.make(sess, [](void* p) { delete static_cast<HostSortformerSession*>(p); });
            });
            b.def("diarize", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostSortformer*>(g_sortformerClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));
                int numFrames = 0;
                int numSpeakers = 4;
                double frameSeconds = 0.08;
                std::vector<float> probs;
                if (h && h->model && !audio.empty()) {
                    try {
                        brosoundml::AudioBuffer ab;
                        ab.samples = std::move(audio);
                        ab.sample_rate = 16000;
                        auto d = h->model->diarize(ab);
                        numFrames = d.num_frames;
                        numSpeakers = d.num_speakers;
                        frameSeconds = d.frame_seconds;
                        probs = std::move(d.probs);
                    } catch (...) {}
                }
                return makeDiarResult(numFrames, numSpeakers, frameSeconds, probs);
            });
        });

    // -----------------------------------------------------------------------
    // ClusterDiarizer
    // -----------------------------------------------------------------------
    g_clusterDiarizerClass.install("ClusterDiarizer", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostClusterDiarizer();
            return g_clusterDiarizerClass.make(h, [](void* p) { delete static_cast<HostClusterDiarizer*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("device", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostClusterDiarizer*>(g_clusterDiarizerClass.unwrap(self));
                return ev::fromUtf8(h ? h->device : "CPU");
            });
            b.accessor("busy", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostClusterDiarizer*>(g_clusterDiarizerClass.unwrap(self));
                return ev::fromBool(h && h->busy);
            });
            b.def("diarize", 2, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostClusterDiarizer*>(g_clusterDiarizerClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));
                int numFrames = 0;
                int numSpeakers = 2;
                double frameSeconds = 0.08;
                std::vector<float> probs;
                if (h && h->model && !audio.empty()) {
                    try {
                        brosoundml::AudioBuffer ab;
                        ab.samples = std::move(audio);
                        ab.sample_rate = 16000;
                        auto d = h->model->diarize(ab);
                        numFrames = d.num_frames;
                        numSpeakers = d.num_speakers;
                        frameSeconds = d.frame_seconds;
                        probs = std::move(d.probs);
                    } catch (...) {}
                }
                return makeDiarResult(numFrames, numSpeakers, frameSeconds, probs);
            });
        });

    // -----------------------------------------------------------------------
    // bro.diar namespace
    // -----------------------------------------------------------------------
    ObjectBuilder diar;

    diar.def("init", 0, [](Value, std::span<const Value>) -> Value {
        brotensor::init();
        return ev::undefined();
    });

    diar.def("loadSortformer", 2, [](Value, std::span<const Value> a) -> Value {
        std::string modelDir = strAt(a, 0);
        Value opts = argAt(a, 1);
        brotensor::Device dev = parseDeviceOpt(opts);
        auto* h = new HostSortformer();
        h->device = deviceName(dev);
        if (!modelDir.empty()) {
            try {
                h->model = std::make_shared<brosoundml::Sortformer>();
                h->model->load(modelDir, dev);
            } catch (...) {}
        }
        Value val = g_sortformerClass.make(h, [](void* p) { delete static_cast<HostSortformer*>(p); });
        if (ev::isObject(opts)) {
            Value onReady = ev::getProperty(opts, "onReady");
            if (ev::isFunction(onReady)) triggerCallback(onReady, val);
        }
        return val;
    });

    diar.def("diarize", 3, [](Value, std::span<const Value> a) -> Value {
        Value modelVal = argAt(a, 0);
        auto audio = extractFloatAudio(argAt(a, 1));
        Value opts = argAt(a, 2);

        int numFrames = 0;
        int numSpeakers = 4;
        double frameSeconds = 0.08;
        std::vector<float> probs;

        auto* h = static_cast<HostSortformer*>(g_sortformerClass.unwrap(modelVal));
        if (h && h->model && !audio.empty()) {
            try {
                brosoundml::AudioBuffer ab;
                ab.samples = std::move(audio);
                ab.sample_rate = 16000;
                auto d = h->model->diarize(ab);
                numFrames = d.num_frames;
                numSpeakers = d.num_speakers;
                frameSeconds = d.frame_seconds;
                probs = std::move(d.probs);
            } catch (...) {}
        }

        Value res = makeDiarResult(numFrames, numSpeakers, frameSeconds, probs);
        if (ev::isObject(opts)) {
            Value onDone = ev::getProperty(opts, "onDone");
            if (ev::isFunction(onDone)) triggerCallback(onDone, res);
        }
        return res;
    });

    diar.def("loadClusterDiarizer", 3, [](Value, std::span<const Value> a) -> Value {
        std::string embDir = strAt(a, 0);
        std::string vadDir = strAt(a, 1);
        Value opts = argAt(a, 2);
        brotensor::Device dev = parseDeviceOpt(opts);
        auto* h = new HostClusterDiarizer();
        h->device = deviceName(dev);
        if (!embDir.empty() && !vadDir.empty()) {
            try {
                h->model = std::make_shared<brosoundml::ClusterDiarizer>();
                h->model->load(embDir, vadDir, dev);
            } catch (...) {}
        }
        Value val = g_clusterDiarizerClass.make(h, [](void* p) { delete static_cast<HostClusterDiarizer*>(p); });
        if (ev::isObject(opts)) {
            Value onReady = ev::getProperty(opts, "onReady");
            if (ev::isFunction(onReady)) triggerCallback(onReady, val);
        }
        return val;
    });

    diar.def("clusterDiarize", 3, [](Value, std::span<const Value> a) -> Value {
        Value modelVal = argAt(a, 0);
        auto audio = extractFloatAudio(argAt(a, 1));
        Value opts = argAt(a, 2);

        int numFrames = 0;
        int numSpeakers = 2;
        double frameSeconds = 0.08;
        std::vector<float> probs;

        auto* h = static_cast<HostClusterDiarizer*>(g_clusterDiarizerClass.unwrap(modelVal));
        if (h && h->model && !audio.empty()) {
            try {
                brosoundml::AudioBuffer ab;
                ab.samples = std::move(audio);
                ab.sample_rate = 16000;
                auto d = h->model->diarize(ab);
                numFrames = d.num_frames;
                numSpeakers = d.num_speakers;
                frameSeconds = d.frame_seconds;
                probs = std::move(d.probs);
            } catch (...) {}
        }

        Value res = makeDiarResult(numFrames, numSpeakers, frameSeconds, probs);
        if (ev::isObject(opts)) {
            Value onDone = ev::getProperty(opts, "onDone");
            if (ev::isFunction(onDone)) triggerCallback(onDone, res);
        }
        return res;
    });

    // Expose class constructors on bro.diar
    diar.set("Sortformer", g_sortformerClass.constructor());
    diar.set("SortformerSession", g_sortformerSessionClass.constructor());
    diar.set("ClusterDiarizer", g_clusterDiarizerClass.constructor());

    bro.set("diar", diar.get());
}

} // namespace brosoundml::api
