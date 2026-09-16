#include "soundml_legacy_compat.h"

#include <brosoundml/kokoro.h>
#include <brosoundml/qwen_tts.h>
#include <brosoundml/omnivoice.h>
#include <brosoundml/supertonic.h>
#include <brosoundml/speaker_encoder.h>
#include <brosoundml/g2p/phonemizer.h>
#include <brotensor/runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace brosoundml::api {

namespace {

HostClass g_kokoroModelClass;
HostClass g_voiceClass;
HostClass g_kokoroSessionClass;
HostClass g_qwenTtsModelClass;
HostClass g_qwenTtsSessionClass;
HostClass g_omniVoiceClass;
HostClass g_supertonicModelClass;
HostClass g_supertonicVoiceClass;
HostClass g_speakerEncoderClass;

struct HostVoice {
    std::string name = "af";
    bool loaded = true;
    std::unique_ptr<brosoundml::Voice> voice;
};

struct HostKokoroModel {
    std::shared_ptr<brosoundml::Kokoro> model;
    std::string device = "CPU";
    bool loaded = true;
};

struct HostKokoroSession {
    std::shared_ptr<brosoundml::Kokoro> model;
    std::unique_ptr<brosoundml::KokoroSession> session;
    bool loaded = true;
};

struct HostQwenTtsModel {
    std::shared_ptr<brosoundml::QwenTts> model;
    std::string device = "CPU";
    std::string variant = "base";
    bool loaded = true;
};

struct HostQwenTtsSession {
    std::shared_ptr<brosoundml::QwenTts> model;
    std::unique_ptr<brosoundml::QwenTtsSession> session;
    std::string variant = "base";
    bool loaded = true;
};

struct HostOmniVoice {
    std::shared_ptr<brosoundml::OmniVoice> model;
    std::string device = "CPU";
    bool loaded = true;
};

struct HostSupertonicVoice {
    std::string name = "default";
    bool loaded = true;
};

struct HostSupertonicModel {
    std::shared_ptr<brosoundml::Supertonic> model;
    std::string device = "CPU";
    bool loaded = true;
};

struct HostSpeakerEncoder {
    std::shared_ptr<brosoundml::SpeakerEncoder> encoder;
    std::string device = "CPU";
    bool loaded = true;
};

static std::string g_assetRoot;
static std::string g_lexiconPath;
static std::string g_posPath;
static std::string g_kokoroConfigPath;

Value makeSynthResult(const std::vector<float>& samples, int sampleRate = 24000) {
    ObjectBuilder res;
    res.set("samples", makeFloat32Array(samples));
    res.set("sampleRate", static_cast<double>(sampleRate));
    return res.get();
}

} // namespace

void installTts(ObjectBuilder& bro) {
    // -----------------------------------------------------------------------
    // Voice
    // -----------------------------------------------------------------------
    g_voiceClass.install("Voice", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostVoice();
            return g_voiceClass.make(h, [](void* p) { delete static_cast<HostVoice*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostVoice*>(g_voiceClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("name", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostVoice*>(g_voiceClass.unwrap(self));
                return ev::fromUtf8(h ? h->name : "af");
            });
        });

    // -----------------------------------------------------------------------
    // KokoroSession
    // -----------------------------------------------------------------------
    g_kokoroSessionClass.install("KokoroSession", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostKokoroSession();
            return g_kokoroSessionClass.make(h, [](void* p) { delete static_cast<HostKokoroSession*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostKokoroSession*>(g_kokoroSessionClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.def("reset", 0, [](Value, std::span<const Value>) -> Value {
                return ev::undefined();
            });
            b.def("synthesize", 3, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostKokoroSession*>(g_kokoroSessionClass.unwrap(self));
                auto phonemes = extractInt32Array(argAt(a, 0));
                Value opts = argAt(a, 2);
                std::vector<float> samples;
                int sr = 24000;
                if (h && h->session && !phonemes.empty()) {
                    try {
                        auto buf = h->session->synthesize(phonemes);
                        samples = std::move(buf.samples);
                        sr = buf.sample_rate;
                    } catch (...) {}
                }
                Value res = makeSynthResult(samples, sr);
                if (ev::isObject(opts)) {
                    Value onDone = ev::getProperty(opts, "onDone");
                    if (ev::isFunction(onDone)) triggerCallback(onDone, res);
                }
                return res;
            });
        });

    // -----------------------------------------------------------------------
    // KokoroModel
    // -----------------------------------------------------------------------
    g_kokoroModelClass.install("KokoroModel", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostKokoroModel();
            return g_kokoroModelClass.make(h, [](void* p) { delete static_cast<HostKokoroModel*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostKokoroModel*>(g_kokoroModelClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("device", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostKokoroModel*>(g_kokoroModelClass.unwrap(self));
                return ev::fromUtf8(h ? h->device : "CPU");
            });
            b.def("encodePhonemes", 1, [](Value, std::span<const Value> a) -> Value {
                std::string ipa = strAt(a, 0);
                std::vector<int32_t> tokens;
                for (unsigned char c : ipa) tokens.push_back(static_cast<int32_t>(c));
                return makeInt32Array(tokens);
            });
            b.def("loadVoice", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostKokoroModel*>(g_kokoroModelClass.unwrap(self));
                std::string path = strAt(a, 0);
                auto* v = new HostVoice();
                v->name = path.empty() ? "af" : path;
                if (h && h->model && !path.empty()) {
                    try {
                        v->voice = std::make_unique<brosoundml::Voice>(h->model->load_voice(path));
                    } catch (...) {}
                }
                return g_voiceClass.make(v, [](void* p) { delete static_cast<HostVoice*>(p); });
            });
            b.def("createSession", 0, [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostKokoroModel*>(g_kokoroModelClass.unwrap(self));
                auto* sess = new HostKokoroSession();
                sess->model = h ? h->model : nullptr;
                return g_kokoroSessionClass.make(sess, [](void* p) { delete static_cast<HostKokoroSession*>(p); });
            });
        });

    // -----------------------------------------------------------------------
    // QwenTtsSession
    // -----------------------------------------------------------------------
    g_qwenTtsSessionClass.install("QwenTtsSession", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostQwenTtsSession();
            return g_qwenTtsSessionClass.make(h, [](void* p) { delete static_cast<HostQwenTtsSession*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenTtsSession*>(g_qwenTtsSessionClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("variant", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenTtsSession*>(g_qwenTtsSessionClass.unwrap(self));
                return ev::fromUtf8(h ? h->variant : "base");
            });
            b.def("reset", 0, [](Value, std::span<const Value>) -> Value {
                return ev::undefined();
            });
            b.def("synthesize", 2, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostQwenTtsSession*>(g_qwenTtsSessionClass.unwrap(self));
                std::string text = strAt(a, 0);
                Value opts = argAt(a, 1);
                std::string speaker = getPropertyString(opts, "speaker", "");
                std::vector<float> samples;
                int sr = 24000;
                if (h && h->model && h->session && !text.empty()) {
                    try {
                        auto buf = h->model->synthesize(*h->session, text, speaker);
                        samples = std::move(buf.samples);
                        sr = buf.sample_rate;
                    } catch (...) {}
                }
                Value res = makeSynthResult(samples, sr);
                if (ev::isObject(opts)) {
                    Value onDone = ev::getProperty(opts, "onDone");
                    if (ev::isFunction(onDone)) triggerCallback(onDone, res);
                }
                return res;
            });
        });

    // -----------------------------------------------------------------------
    // QwenTtsModel
    // -----------------------------------------------------------------------
    g_qwenTtsModelClass.install("QwenTtsModel", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostQwenTtsModel();
            return g_qwenTtsModelClass.make(h, [](void* p) { delete static_cast<HostQwenTtsModel*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenTtsModel*>(g_qwenTtsModelClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("device", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenTtsModel*>(g_qwenTtsModelClass.unwrap(self));
                return ev::fromUtf8(h ? h->device : "CPU");
            });
            b.accessor("variant", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenTtsModel*>(g_qwenTtsModelClass.unwrap(self));
                return ev::fromUtf8(h ? h->variant : "base");
            });
            b.def("createSession", 0, [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenTtsModel*>(g_qwenTtsModelClass.unwrap(self));
                auto* sess = new HostQwenTtsSession();
                sess->model = h ? h->model : nullptr;
                sess->variant = h ? h->variant : "base";
                if (h && h->model) {
                    try {
                        sess->session = std::make_unique<brosoundml::QwenTtsSession>(h->model->make_session());
                    } catch (...) {}
                }
                return g_qwenTtsSessionClass.make(sess, [](void* p) { delete static_cast<HostQwenTtsSession*>(p); });
            });
            b.def("encodeAudio", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostQwenTtsModel*>(g_qwenTtsModelClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));
                std::vector<int32_t> codes;
                int frames = 0;
                if (h && h->model && !audio.empty()) {
                    try {
                        brosoundml::AudioBuffer ab;
                        ab.samples = std::move(audio);
                        ab.sample_rate = 24000;
                        codes = h->model->encode_audio(ab, &frames);
                    } catch (...) {}
                }
                ObjectBuilder res;
                res.set("codes", makeInt32Array(codes));
                res.set("numFrames", static_cast<double>(frames));
                return res.get();
            });
            b.def("decodeCodes", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostQwenTtsModel*>(g_qwenTtsModelClass.unwrap(self));
                auto codes = extractInt32Array(argAt(a, 0));
                std::vector<float> samples;
                if (h && h->model && !codes.empty()) {
                    try {
                        int numQuant = 16;
                        int numFrames = static_cast<int>(codes.size()) / numQuant;
                        if (numFrames > 0) {
                            auto buf = h->model->decode_codes(codes, numQuant, numFrames);
                            samples = std::move(buf.samples);
                        }
                    } catch (...) {}
                }
                return makeFloat32Array(samples);
            });
        });

    // -----------------------------------------------------------------------
    // OmniVoice
    // -----------------------------------------------------------------------
    g_omniVoiceClass.install("OmniVoice", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostOmniVoice();
            return g_omniVoiceClass.make(h, [](void* p) { delete static_cast<HostOmniVoice*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostOmniVoice*>(g_omniVoiceClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("device", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostOmniVoice*>(g_omniVoiceClass.unwrap(self));
                return ev::fromUtf8(h ? h->device : "CPU");
            });
            b.def("synthesize", 2, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostOmniVoice*>(g_omniVoiceClass.unwrap(self));
                std::string text = strAt(a, 0);
                Value opts = argAt(a, 1);
                std::vector<float> samples;
                int sr = 24000;
                if (h && h->model && !text.empty()) {
                    try {
                        auto buf = h->model->synthesize(text);
                        samples = std::move(buf.samples);
                        sr = buf.sample_rate;
                    } catch (...) {}
                }
                Value res = makeSynthResult(samples, sr);
                if (ev::isObject(opts)) {
                    Value onDone = ev::getProperty(opts, "onDone");
                    if (ev::isFunction(onDone)) triggerCallback(onDone, res);
                }
                return res;
            });
            b.def("reset", 0, [](Value, std::span<const Value>) -> Value {
                return ev::undefined();
            });
        });

    // -----------------------------------------------------------------------
    // SupertonicVoice
    // -----------------------------------------------------------------------
    g_supertonicVoiceClass.install("SupertonicVoice", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostSupertonicVoice();
            return g_supertonicVoiceClass.make(h, [](void* p) { delete static_cast<HostSupertonicVoice*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostSupertonicVoice*>(g_supertonicVoiceClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("name", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostSupertonicVoice*>(g_supertonicVoiceClass.unwrap(self));
                return ev::fromUtf8(h ? h->name : "default");
            });
        });

    // -----------------------------------------------------------------------
    // SupertonicModel
    // -----------------------------------------------------------------------
    g_supertonicModelClass.install("SupertonicModel", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostSupertonicModel();
            return g_supertonicModelClass.make(h, [](void* p) { delete static_cast<HostSupertonicModel*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostSupertonicModel*>(g_supertonicModelClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("device", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostSupertonicModel*>(g_supertonicModelClass.unwrap(self));
                return ev::fromUtf8(h ? h->device : "CPU");
            });
            b.def("loadVoiceStyle", 1, [](Value, std::span<const Value> a) -> Value {
                std::string path = strAt(a, 0);
                auto* v = new HostSupertonicVoice();
                v->name = path.empty() ? "default" : path;
                return g_supertonicVoiceClass.make(v, [](void* p) { delete static_cast<HostSupertonicVoice*>(p); });
            });
        });

    // -----------------------------------------------------------------------
    // SpeakerEncoder
    // -----------------------------------------------------------------------
    g_speakerEncoderClass.install("SpeakerEncoder", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostSpeakerEncoder();
            return g_speakerEncoderClass.make(h, [](void* p) { delete static_cast<HostSpeakerEncoder*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostSpeakerEncoder*>(g_speakerEncoderClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("device", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostSpeakerEncoder*>(g_speakerEncoderClass.unwrap(self));
                return ev::fromUtf8(h ? h->device : "CPU");
            });
            b.def("embedSpeaker", 2, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostSpeakerEncoder*>(g_speakerEncoderClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));
                Value opts = argAt(a, 1);
                std::vector<float> emb;
                if (h && h->encoder && !audio.empty()) {
                    try {
                        brosoundml::AudioBuffer ab;
                        ab.samples = std::move(audio);
                        ab.sample_rate = 16000;
                        emb = h->encoder->embed(ab);
                    } catch (...) {}
                }
                Value res = makeFloat32Array(emb);
                if (ev::isObject(opts)) {
                    Value onDone = ev::getProperty(opts, "onDone");
                    if (ev::isFunction(onDone)) triggerCallback(onDone, res);
                }
                return res;
            });
        });

    // -----------------------------------------------------------------------
    // bro.tts namespace
    // -----------------------------------------------------------------------
    ObjectBuilder tts;

    tts.def("init", 0, [](Value, std::span<const Value>) -> Value {
        brotensor::init();
        return ev::undefined();
    });

    tts.def("loadKokoro", 2, [](Value, std::span<const Value> a) -> Value {
        std::string dir = strAt(a, 0);
        Value opts = argAt(a, 1);
        brotensor::Device dev = parseDeviceOpt(opts);
        auto* h = new HostKokoroModel();
        h->device = deviceName(dev);
        if (!dir.empty()) {
            try {
                h->model = std::make_shared<brosoundml::Kokoro>();
                h->model->load(dir, dev);
            } catch (...) {}
        }
        Value val = g_kokoroModelClass.make(h, [](void* p) { delete static_cast<HostKokoroModel*>(p); });
        if (ev::isObject(opts)) {
            Value onReady = ev::getProperty(opts, "onReady");
            if (ev::isFunction(onReady)) triggerCallback(onReady, val);
        }
        return val;
    });

    tts.def("loadQwen", 2, [](Value, std::span<const Value> a) -> Value {
        std::string dir = strAt(a, 0);
        Value opts = argAt(a, 1);
        brotensor::Device dev = parseDeviceOpt(opts);
        auto* h = new HostQwenTtsModel();
        h->device = deviceName(dev);
        if (!dir.empty()) {
            try {
                h->model = std::make_shared<brosoundml::QwenTts>();
                h->model->load(dir, dev);
            } catch (...) {}
        }
        Value val = g_qwenTtsModelClass.make(h, [](void* p) { delete static_cast<HostQwenTtsModel*>(p); });
        if (ev::isObject(opts)) {
            Value onReady = ev::getProperty(opts, "onReady");
            if (ev::isFunction(onReady)) triggerCallback(onReady, val);
        }
        return val;
    });

    tts.def("loadOmniVoice", 2, [](Value, std::span<const Value> a) -> Value {
        std::string dir = strAt(a, 0);
        Value opts = argAt(a, 1);
        brotensor::Device dev = parseDeviceOpt(opts);
        auto* h = new HostOmniVoice();
        h->device = deviceName(dev);
        if (!dir.empty()) {
            try {
                h->model = std::make_shared<brosoundml::OmniVoice>();
                h->model->load(dir, dev);
            } catch (...) {}
        }
        Value val = g_omniVoiceClass.make(h, [](void* p) { delete static_cast<HostOmniVoice*>(p); });
        if (ev::isObject(opts)) {
            Value onReady = ev::getProperty(opts, "onReady");
            if (ev::isFunction(onReady)) triggerCallback(onReady, val);
        }
        return val;
    });

    tts.def("loadSupertonic", 2, [](Value, std::span<const Value> a) -> Value {
        std::string dir = strAt(a, 0);
        Value opts = argAt(a, 1);
        brotensor::Device dev = parseDeviceOpt(opts);
        auto* h = new HostSupertonicModel();
        h->device = deviceName(dev);
        if (!dir.empty()) {
            try {
                h->model = std::make_shared<brosoundml::Supertonic>();
                h->model->load(dir, dev);
            } catch (...) {}
        }
        Value val = g_supertonicModelClass.make(h, [](void* p) { delete static_cast<HostSupertonicModel*>(p); });
        if (ev::isObject(opts)) {
            Value onReady = ev::getProperty(opts, "onReady");
            if (ev::isFunction(onReady)) triggerCallback(onReady, val);
        }
        return val;
    });

    tts.def("loadSpeakerEncoder", 2, [](Value, std::span<const Value> a) -> Value {
        std::string dir = strAt(a, 0);
        Value opts = argAt(a, 1);
        brotensor::Device dev = parseDeviceOpt(opts);
        auto* h = new HostSpeakerEncoder();
        h->device = deviceName(dev);
        if (!dir.empty()) {
            try {
                h->encoder = std::make_shared<brosoundml::SpeakerEncoder>();
                h->encoder->load(dir);
            } catch (...) {}
        }
        Value val = g_speakerEncoderClass.make(h, [](void* p) { delete static_cast<HostSpeakerEncoder*>(p); });
        if (ev::isObject(opts)) {
            Value onReady = ev::getProperty(opts, "onReady");
            if (ev::isFunction(onReady)) triggerCallback(onReady, val);
        }
        return val;
    });

    tts.def("phonemize", 2, [](Value, std::span<const Value> a) -> Value {
        std::string text = strAt(a, 0);
        std::vector<int32_t> tokens;
        for (unsigned char c : text) tokens.push_back(static_cast<int32_t>(c));
        return makeInt32Array(tokens);
    });

    tts.def("setAssetRoot", 1, [](Value, std::span<const Value> a) -> Value {
        g_assetRoot = strAt(a, 0);
        return ev::undefined();
    });

    tts.def("setAssets", 1, [](Value, std::span<const Value> a) -> Value {
        Value opts = argAt(a, 0);
        if (ev::isObject(opts)) {
            g_assetRoot = getPropertyString(opts, "root", g_assetRoot);
            g_lexiconPath = getPropertyString(opts, "lexicon", g_lexiconPath);
            g_posPath = getPropertyString(opts, "pos", g_posPath);
            g_kokoroConfigPath = getPropertyString(opts, "kokoroConfig", g_kokoroConfigPath);
        }
        return ev::undefined();
    });

    tts.def("synthesize", 4, [](Value, std::span<const Value> a) -> Value {
        Value modelVal = argAt(a, 0);
        Value textOrPhonemes = argAt(a, 1);
        Value voiceOrOpts = argAt(a, 2);
        Value opts = (a.size() >= 4 && ev::isObject(a[3])) ? a[3]
                   : (a.size() >= 3 && ev::isObject(a[2])) ? a[2] : ev::undefined();
        std::vector<float> samples;
        int sr = 24000;

        auto* km = static_cast<HostKokoroModel*>(g_kokoroModelClass.unwrap(modelVal));
        if (km && km->model) {
            auto phonemes = extractInt32Array(textOrPhonemes);
            auto* hv = static_cast<HostVoice*>(g_voiceClass.unwrap(voiceOrOpts));
            if (hv && hv->voice) {
                try {
                    auto buf = km->model->synthesize(phonemes, *hv->voice);
                    samples = std::move(buf.samples);
                    sr = buf.sample_rate;
                } catch (...) {}
            }
        }
        auto* qm = static_cast<HostQwenTtsModel*>(g_qwenTtsModelClass.unwrap(modelVal));
        if (qm && qm->model) {
            std::string text = ev::isString(textOrPhonemes) ? ev::toUtf8(textOrPhonemes) : "";
            std::string speaker = getPropertyString(opts, "speaker", "");
            try {
                auto buf = qm->model->synthesize(text, speaker);
                samples = std::move(buf.samples);
                sr = buf.sample_rate;
            } catch (...) {}
        }
        auto* om = static_cast<HostOmniVoice*>(g_omniVoiceClass.unwrap(modelVal));
        if (om && om->model) {
            std::string text = ev::isString(textOrPhonemes) ? ev::toUtf8(textOrPhonemes) : "";
            try {
                auto buf = om->model->synthesize(text);
                samples = std::move(buf.samples);
                sr = buf.sample_rate;
            } catch (...) {}
        }

        Value res = makeSynthResult(samples, sr);
        if (ev::isObject(opts)) {
            Value onDone = ev::getProperty(opts, "onDone");
            if (ev::isFunction(onDone)) triggerCallback(onDone, res);
        }
        return res;
    });

    tts.def("synthesizeStream", 4, [](Value, std::span<const Value> a) -> Value {
        Value opts = (a.size() >= 4 && ev::isObject(a[3])) ? a[3]
                   : (a.size() >= 3 && ev::isObject(a[2])) ? a[2] : ev::undefined();
        std::vector<float> samples;
        Value res = makeSynthResult(samples, 24000);
        if (ev::isObject(opts)) {
            Value onChunk = ev::getProperty(opts, "onChunk");
            if (ev::isFunction(onChunk)) triggerCallback(onChunk, res);
            Value onDone = ev::getProperty(opts, "onDone");
            if (ev::isFunction(onDone)) triggerCallback(onDone, res);
        }
        return makeAsyncHandle();
    });

    tts.def("decodeFrom", 7, [](Value, std::span<const Value> a) -> Value {
        Value opts = argAt(a, 6);
        std::vector<float> samples;
        Value res = makeSynthResult(samples, 24000);
        if (ev::isObject(opts)) {
            Value onDone = ev::getProperty(opts, "onDone");
            if (ev::isFunction(onDone)) triggerCallback(onDone, res);
        }
        return res;
    });

    // Expose class constructors on bro.tts
    tts.set("KokoroModel", g_kokoroModelClass.constructor());
    tts.set("Voice", g_voiceClass.constructor());
    tts.set("KokoroSession", g_kokoroSessionClass.constructor());
    tts.set("QwenTtsModel", g_qwenTtsModelClass.constructor());
    tts.set("QwenTtsSession", g_qwenTtsSessionClass.constructor());
    tts.set("OmniVoice", g_omniVoiceClass.constructor());
    tts.set("SupertonicModel", g_supertonicModelClass.constructor());
    tts.set("SupertonicVoice", g_supertonicVoiceClass.constructor());
    tts.set("SpeakerEncoder", g_speakerEncoderClass.constructor());

    bro.set("tts", tts.get());
}

} // namespace brosoundml::api
