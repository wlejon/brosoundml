#include "soundml_legacy_compat.h"

#include <brosoundml/rave.h>
#include <brotensor/runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace brosoundml::api {

namespace {

HostClass g_raveClass;

struct HostRave {
    std::shared_ptr<brosoundml::Rave> model;
    std::string device = "CPU";
    bool loaded = true;
    int sampleRate = 48000;
    int nLatent = 8;
    int fullLatent = 16;
    int nBand = 16;
    int totalRatio = 2048;
};

} // namespace

void installRave(ObjectBuilder& bro) {
    // -----------------------------------------------------------------------
    // Rave
    // -----------------------------------------------------------------------
    g_raveClass.install("Rave", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostRave();
            return g_raveClass.make(h, [](void* p) { delete static_cast<HostRave*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostRave*>(g_raveClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("sampleRate", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostRave*>(g_raveClass.unwrap(self));
                return ev::fromDouble(h ? h->sampleRate : 48000);
            });
            b.accessor("nLatent", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostRave*>(g_raveClass.unwrap(self));
                return ev::fromDouble(h ? h->nLatent : 8);
            });
            b.accessor("fullLatent", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostRave*>(g_raveClass.unwrap(self));
                return ev::fromDouble(h ? h->fullLatent : 16);
            });
            b.accessor("nBand", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostRave*>(g_raveClass.unwrap(self));
                return ev::fromDouble(h ? h->nBand : 16);
            });
            b.accessor("totalRatio", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostRave*>(g_raveClass.unwrap(self));
                return ev::fromDouble(h ? h->totalRatio : 2048);
            });
            b.def("encode", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostRave*>(g_raveClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));

                std::vector<float> latent;
                int nLatent = h ? h->nLatent : 8;
                int frames = 0;

                if (h && h->model && !audio.empty()) {
                    try {
                        auto l = h->model->encode(audio);
                        latent = std::move(l.data);
                        nLatent = l.n_latent;
                        frames = l.frames;
                    } catch (...) {}
                }

                ObjectBuilder res;
                res.set("latent", makeFloat32Array(latent));
                res.set("nLatent", static_cast<double>(nLatent));
                res.set("frames", static_cast<double>(frames));
                return res.get();
            });
            b.def("decode", 3, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostRave*>(g_raveClass.unwrap(self));
                auto latent = extractFloatAudio(argAt(a, 0));
                int frames = i32At(a, 1, 0);
                Value opts = argAt(a, 2);

                int channels = getPropertyInt(opts, "channels", 1);
                int sr = h ? h->sampleRate : 48000;
                std::vector<float> samples;

                if (h && h->model && !latent.empty() && frames > 0) {
                    try {
                        brosoundml::RaveDecodeOptions decOpts;
                        decOpts.add_noise = getPropertyBool(opts, "addNoise", false);
                        decOpts.seed = static_cast<uint64_t>(getPropertyDouble(opts, "seed", 0.0));
                        decOpts.channels = channels;
                        decOpts.latent_pad_std = static_cast<float>(getPropertyDouble(opts, "stereoWidth", 1.0));

                        if (channels > 1) {
                            auto mb = h->model->decode_multi(latent.data(), h->nLatent, frames, decOpts);
                            samples = std::move(mb.samples);
                            sr = mb.sample_rate;
                        } else {
                            auto buf = h->model->decode(latent.data(), h->nLatent, frames, decOpts);
                            samples = std::move(buf.samples);
                            sr = buf.sample_rate;
                        }
                    } catch (...) {}
                }

                ObjectBuilder res;
                res.set("samples", makeFloat32Array(samples));
                res.set("sampleRate", static_cast<double>(sr));
                res.set("channels", static_cast<double>(channels));
                return res.get();
            });
        });

    // -----------------------------------------------------------------------
    // bro.rave namespace
    // -----------------------------------------------------------------------
    ObjectBuilder rave;

    rave.def("init", 0, [](Value, std::span<const Value>) -> Value {
        brotensor::init();
        return ev::undefined();
    });

    rave.def("loadRave", 2, [](Value, std::span<const Value> a) -> Value {
        std::string dir = strAt(a, 0);
        Value opts = argAt(a, 1);
        brotensor::Device dev = parseDeviceOpt(opts);
        auto* h = new HostRave();
        h->device = deviceName(dev);
        if (!dir.empty()) {
            try {
                h->model = std::make_shared<brosoundml::Rave>();
                h->model->load(dir, dev);
                h->sampleRate = h->model->config().sampling_rate;
                h->nLatent = h->model->config().cropped_latent_size;
                h->fullLatent = h->model->config().full_latent_size;
                h->nBand = h->model->config().n_band;
                h->totalRatio = h->model->config().total_ratio;
            } catch (...) {}
        }
        Value val = g_raveClass.make(h, [](void* p) { delete static_cast<HostRave*>(p); });
        if (ev::isObject(opts)) {
            Value onReady = ev::getProperty(opts, "onReady");
            if (ev::isFunction(onReady)) triggerCallback(onReady, val);
        }
        return val;
    });

    // Expose constructor on bro.rave
    rave.set("Rave", g_raveClass.constructor());

    bro.set("rave", rave.get());
}

} // namespace brosoundml::api
