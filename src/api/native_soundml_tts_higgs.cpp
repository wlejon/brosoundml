// bro.tts.loadHiggsCodec / HiggsCodec — the HiggsAudio v2 tokenizer on its
// own: 24 kHz PCM <-> 8-codebook RVQ codes at 25 frames/s, without the
// OmniVoice language model. OmniVoice exposes the same codec through
// encodeAudio / decodeCodes, but only after loading its 0.6B LM; this handle
// loads just the audio_tokenizer/ directory, for codec round-trips, code
// analysis and editing.
//
// encode / decode are synchronous GPU calls (the codec is a single conv
// stack; a clip takes milliseconds to a second), gated like every model:
// one op at a time per handle.
#include "soundml_tts_internal.h"
#include "soundml_loader.h"

#include <brosoundml/higgs_codec.h>

namespace brosoundml::api {

HostClass g_higgsCodecClass;

namespace {

struct HostHiggsCodec {
    std::shared_ptr<brosoundml::HiggsCodec> codec;
    brotensor::Device device = brotensor::Device::CPU;
    ModelGate busy;
};

HostHiggsCodec* codecSelf(Value self) {
    return static_cast<HostHiggsCodec*>(g_higgsCodecClass.unwrap(self));
}

// Raises and returns null unless `self` is a loaded, idle codec.
HostHiggsCodec* readyCodec(Value self, const char* fn) {
    auto* w = codecSelf(self);
    if (!w) {
        ev::throwTypeError(std::string(fn) + ": not a HiggsCodec");
        return nullptr;
    }
    if (!w->codec || !w->codec->loaded()) {
        ev::throwError(std::string(fn) + ": codec is not loaded");
        return nullptr;
    }
    if (w->busy.isBusy()) {
        ev::throwError(std::string(fn) + ": an operation is already in flight on this codec");
        return nullptr;
    }
    return w;
}

// codec.encode(samples, sampleRate?) -> { codes: Int32Array, numFrames, numQuantizers }
//   samples: Float32Array (sampleRate default 24000), or { samples, sampleRate }.
//   Resampled to 24 kHz mono and right-padded to a whole 960-sample frame.
//   codes[q * numFrames + t], codebook-major — the layout decode() takes.
Value codecEncode(Value self, std::span<const Value> a) {
    auto* w = readyCodec(self, "encode");
    if (!w) return ev::undefined();
    if (!w->codec->has_encoder())
        return ev::throwError("encode: the codec was loaded with decoderOnly (no encoder / HuBERT)");
    brosoundml::AudioBuffer audio;
    if (!readAudioArgs(a, 0, audio, w->codec->config().sample_rate))
        return ev::throwTypeError("encode(samples, sampleRate?): samples must be a non-empty Float32Array");
    if (audio.sample_rate <= 0) return ev::throwTypeError("encode: sampleRate must be positive");
    if (!w->busy.tryClaim()) return ev::throwError("encode: an operation is already in flight on this codec");
    std::vector<int32_t> codes;
    int nf = 0;
    try {
        brotensor::DeviceScope scope(w->device);
        codes = w->codec->encode(audio, &nf);
    } catch (const std::exception& e) {
        w->busy.release();
        return ev::throwError(std::string("encode: ") + e.what());
    }
    w->busy.release();
    ObjectBuilder obj;
    {
        ev::Persistent c(makeInt32Array(codes));
        obj.set("codes", c.get());
    }
    obj.set("numFrames", ev::fromDouble(nf));
    obj.set("numQuantizers", ev::fromDouble(w->codec->config().num_quantizers));
    return obj.get();
}

// codec.decode(codes, numFrames?) -> { samples: Float32Array, sampleRate }
// codec.decode(codes, { numFrames?, numQuantizers? })
//   codes: Int32Array | number[], codebook-major (codes[q * numFrames + t]).
//   numQuantizers may be below config numQuantizers (fewer levels = coarser
//   audio); numFrames defaults to codes.length / numQuantizers.
Value codecDecode(Value self, std::span<const Value> a) {
    auto* w = readyCodec(self, "decode");
    if (!w) return ev::undefined();
    if (!hasArg(a, 0)) return ev::throwTypeError("decode(codes, numFrames?): codes required");
    std::vector<int32_t> codes = readInt32Array(a[0]);
    if (codes.empty()) return ev::throwTypeError("decode: codes must be a non-empty Int32Array or number[]");
    int nq = w->codec->config().num_quantizers;
    int nf = 0;
    if (hasArg(a, 1)) {
        if (ev::isNumber(a[1])) {
            nf = i32At(a, 1);
        } else if (ev::isObject(a[1])) {
            ev::Persistent o(a[1]);
            getIntOpt(o.get(), "numFrames", nf);
            getIntOpt(o.get(), "numQuantizers", nq);
        } else {
            return ev::throwTypeError("decode: arg 1 must be numFrames or { numFrames?, numQuantizers? }");
        }
    }
    if (nq < 1 || nq > w->codec->config().num_quantizers)
        return ev::throwTypeError("decode: numQuantizers must be in 1.." +
                                  std::to_string(w->codec->config().num_quantizers));
    if (nf <= 0) nf = static_cast<int>(codes.size() / static_cast<size_t>(nq));
    if (nf <= 0 || static_cast<size_t>(nq) * static_cast<size_t>(nf) != codes.size())
        return ev::throwTypeError("decode: codes.length must equal numQuantizers (" + std::to_string(nq) +
                                  ") * numFrames");
    if (!w->busy.tryClaim()) return ev::throwError("decode: an operation is already in flight on this codec");
    brosoundml::AudioBuffer out;
    try {
        brotensor::DeviceScope scope(w->device);
        out = w->codec->decode(codes, nq, nf);
    } catch (const std::exception& e) {
        w->busy.release();
        return ev::throwError(std::string("decode: ") + e.what());
    }
    w->busy.release();
    return audioBufferToJs(out);
}

void decorateHiggsCodec(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = codecSelf(self);
        return ev::fromBool(w && w->codec && w->codec->loaded());
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* w = codecSelf(self);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    b.accessor("hasEncoder", [](Value self, std::span<const Value>) -> Value {
        auto* w = codecSelf(self);
        return ev::fromBool(w && w->codec && w->codec->loaded() && w->codec->has_encoder());
    });
    b.accessor("busy", [](Value self, std::span<const Value>) -> Value {
        auto* w = codecSelf(self);
        return ev::fromBool(w && w->busy.isBusy());
    });
    auto cfgGetter = [&b](const char* name, int HiggsCodecConfig::*field) {
        b.accessor(name, [field](Value self, std::span<const Value>) -> Value {
            auto* w = codecSelf(self);
            if (!w || !w->codec || !w->codec->loaded()) return ev::fromDouble(0);
            return ev::fromDouble(w->codec->config().*field);
        });
    };
    cfgGetter("sampleRate",    &HiggsCodecConfig::sample_rate);
    cfgGetter("hopLength",     &HiggsCodecConfig::hop_length);
    cfgGetter("frameRate",     &HiggsCodecConfig::frame_rate);
    cfgGetter("numQuantizers", &HiggsCodecConfig::num_quantizers);
    cfgGetter("codebookSize",  &HiggsCodecConfig::codebook_size);
    b.def("encode", 2, codecEncode);
    b.def("decode", 2, codecDecode);
    // Drop the weights now rather than at GC. Refused while an op runs.
    b.def("dispose", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = codecSelf(self);
        if (!w) return ev::throwTypeError("dispose: not a HiggsCodec");
        if (w->busy.isBusy()) return ev::throwError("dispose: an operation is already in flight on this codec");
        w->codec.reset();
        return ev::undefined();
    });
}

}  // namespace

// bro.tts.loadHiggsCodec(dir, opts?) -> HiggsCodec | AsyncHandle (opts.onReady)
//   dir: the audio_tokenizer directory (config.json + model.safetensors) —
//   OmniVoice's <model>/audio_tokenizer. opts.device (GPU by default),
//   opts.decoderOnly (skip the encoder + HuBERT; encode() then throws),
//   opts.onReady / opts.onError for a background load.
Value loadHiggsCodec(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    ev::Persistent opts;
    if (!modelLoaderArgs("loadHiggsCodec", args, dir, dev, opts)) return ev::undefined();
    const bool decoderOnly = getPropertyBool(opts.get(), "decoderOnly");
    return runModelLoader<HostHiggsCodec>("loadHiggsCodec", opts.get(), g_higgsCodecClass,
                                          [dir, dev, decoderOnly] {
        auto w = std::make_unique<HostHiggsCodec>();
        w->device = dev;
        w->codec = std::make_shared<brosoundml::HiggsCodec>();
        {
            brotensor::DeviceScope scope(dev);
            w->codec->load(dir, dev, decoderOnly);
        }
        logInfo(std::string("[tts] HiggsAudio v2 codec loaded on ") + deviceName(dev) +
                (decoderOnly ? " (decoder only)" : ""));
        return w;
    });
}

void installTtsHiggsClasses() {
    g_higgsCodecClass.install("HiggsCodec", 0, nullptr, decorateHiggsCodec);
}

} // namespace brosoundml::api
