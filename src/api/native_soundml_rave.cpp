// bro.rave — RAVE v2 neural audio autoencoder: encode a waveform to a
// low-rate (nLatent x frames) latent grid, edit it, decode it back (mono, or
// multi-channel with RAVE's stochastic latent pad for stereo decorrelation).
//
// Loading is GPU by default (CUDA > Metal > CPU by availability); opts.device
// picks explicitly and must be a string. encode / decode run synchronously.
#include "soundml_loader.h"

#include <brosoundml/rave.h>

namespace brosoundml::api {

namespace {

HostClass g_raveClass;

struct HostRave {
    std::shared_ptr<brosoundml::Rave> rave;
    brotensor::Device device = brotensor::Device::CPU;
};

HostRave* raveOf(Value v) {
    return g_raveClass.isInstance(v) ? static_cast<HostRave*>(g_raveClass.unwrap(v)) : nullptr;
}

Value makeMultiBuffer(const std::vector<float>& samples, int sampleRate, int channels) {
    ObjectBuilder res;
    {
        ev::Persistent arr(makeFloat32Array(samples));
        res.set("samples", arr.get());
    }
    res.set("sampleRate", ev::fromDouble(sampleRate));
    res.set("channels", ev::fromDouble(channels));
    return res.get();
}

// rave.encode(audio) -> { latent: Float32Array(nLatent*frames, channel-major), nLatent, frames }
Value raveEncode(Value self, std::span<const Value> a) {
    auto* w = raveOf(self);
    if (!w) return ev::throwTypeError("encode: not a Rave");
    if (!hasArg(a, 0)) return ev::throwTypeError("encode(audio): audio Float32Array required");
    std::vector<float> audio = readFloat32Array(a[0]);
    if (audio.empty()) return ev::throwTypeError("encode: audio must be a non-empty Float32Array");
    try {
        brotensor::DeviceScope scope(w->device);
        brosoundml::RaveLatent z = w->rave->encode(audio);
        ObjectBuilder res;
        {
            ev::Persistent arr(makeFloat32Array(z.data));
            res.set("latent", arr.get());
        }
        res.set("nLatent", ev::fromDouble(z.n_latent));
        res.set("frames", ev::fromDouble(z.frames));
        return res.get();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("encode: ") + e.what());
    }
}

// rave.decode(latent, frames, opts?) -> { samples, sampleRate, channels }
//   opts.addNoise, seed, channels (>1 runs decode_multi, interleaved output),
//   stereoWidth (latent-pad std; defaults to RAVE-native 1.0 when channels > 1).
Value raveDecode(Value self, std::span<const Value> a) {
    auto* w = raveOf(self);
    if (!w) return ev::throwTypeError("decode: not a Rave");
    if (a.size() < 2) return ev::throwTypeError("decode(latent, frames): latent and frames required");
    std::vector<float> latent = readFloat32Array(a[0]);
    if (latent.empty()) return ev::throwTypeError("decode: latent must be a non-empty Float32Array");
    const int frames = i32At(a, 1, 0);
    if (frames <= 0 || latent.size() % static_cast<size_t>(frames) != 0)
        return ev::throwTypeError("decode: latent.length must be a positive multiple of frames");
    const int nLatent = static_cast<int>(latent.size() / static_cast<size_t>(frames));

    brosoundml::RaveDecodeOptions opts;
    bool widthSet = false;
    if (isObjectArg(a, 2)) {
        ev::Persistent o(a[2]);
        getBoolOpt(o.get(), "addNoise", opts.add_noise);
        if (hasProperty(o.get(), "seed"))
            opts.seed = static_cast<std::uint64_t>(static_cast<int64_t>(getPropertyDouble(o.get(), "seed", 0.0)));
        if (hasProperty(o.get(), "channels")) {
            const int c = getPropertyInt(o.get(), "channels", 1);
            opts.channels = c < 1 ? 1 : c;
        }
        if (hasProperty(o.get(), "stereoWidth")) {
            opts.latent_pad_std = static_cast<float>(getPropertyDouble(o.get(), "stereoWidth", 1.0));
            widthSet = true;
        }
    }
    if (opts.channels > 1 && !widthSet) opts.latent_pad_std = 1.0f;

    try {
        brotensor::DeviceScope scope(w->device);
        if (opts.channels <= 1) {
            brosoundml::AudioBuffer buf = w->rave->decode(latent.data(), nLatent, frames, opts);
            return makeMultiBuffer(buf.samples, buf.sample_rate, 1);
        }
        brosoundml::RaveMultiBuffer buf = w->rave->decode_multi(latent.data(), nLatent, frames, opts);
        return makeMultiBuffer(buf.samples, buf.sample_rate, buf.channels);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("decode: ") + e.what());
    }
}

void decorateRave(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = raveOf(self);
        return ev::fromBool(w && w->rave && w->rave->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = raveOf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("sampleRate", [](Value self, std::span<const Value>) -> Value {
        auto* w = raveOf(self);
        return ev::fromDouble(w && w->rave ? w->rave->config().sampling_rate : 0);
    });
    b.accessor("nLatent", [](Value self, std::span<const Value>) -> Value {
        auto* w = raveOf(self);
        return ev::fromDouble(w && w->rave ? w->rave->config().cropped_latent_size : 0);
    });
    b.accessor("fullLatent", [](Value self, std::span<const Value>) -> Value {
        auto* w = raveOf(self);
        return ev::fromDouble(w && w->rave ? w->rave->config().full_latent_size : 0);
    });
    b.accessor("nBand", [](Value self, std::span<const Value>) -> Value {
        auto* w = raveOf(self);
        return ev::fromDouble(w && w->rave ? w->rave->config().n_band : 0);
    });
    b.accessor("totalRatio", [](Value self, std::span<const Value>) -> Value {
        auto* w = raveOf(self);
        return ev::fromDouble(w && w->rave ? w->rave->config().total_ratio : 0);
    });
    b.def("encode", 1, raveEncode);
    b.def("decode", 3, raveDecode);
}

Value raveInit(Value, std::span<const Value>) {
    try {
        brotensor::init();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.rave.init: ") + e.what());
    }
    return ev::undefined();
}

// bro.rave.loadRave(modelDir, opts?) -> Rave | AsyncHandle (opts.onReady)
Value loadRave(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    Value opts = ev::undefined();
    if (!modelLoaderArgs("loadRave", args, dir, dev, opts)) return ev::undefined();
    return runModelLoader<HostRave>("loadRave", opts, g_raveClass, [dir, dev] {
        auto w = std::make_unique<HostRave>();
        w->device = dev;
        w->rave = std::make_shared<brosoundml::Rave>();
        {
            brotensor::DeviceScope scope(dev);
            w->rave->load(dir, dev);
        }
        logInfo(std::string("[rave] RAVE loaded on ") + deviceName(dev));
        return w;
    });
}

}  // namespace

void installRave(ObjectBuilder& bro) {
    g_raveClass.install("Rave", 0, nullptr, decorateRave);

    ObjectBuilder rave;
    rave.def("init", 0, raveInit);
    rave.def("loadRave", 2, loadRave);
    rave.set("Rave", g_raveClass.constructor());
    bro.set("rave", rave.get());
}

} // namespace brosoundml::api
