#include "host_soundml_internal.h"

#include <brosoundml/whisper.h>
#include <brosoundml/parakeet.h>
#include <brosoundml/qwen_asr.h>
#include <brotensor/runtime.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml::api {

namespace {

HostClass g_whisperTokenizerClass;
HostClass g_whisperModelClass;
HostClass g_whisperSessionClass;
HostClass g_parakeetTokenizerClass;
HostClass g_parakeetModelClass;
HostClass g_parakeetSessionClass;
HostClass g_qwenAsrModelClass;
HostClass g_qwenAsrSessionClass;
HostClass g_qwenAsrStreamClass;

struct HostWhisperTokenizer {
    bool loaded = true;
};

struct HostWhisperModel {
    std::shared_ptr<brosoundml::Whisper> model;
    std::string device = "CPU";
    bool loaded = true;
};

struct HostWhisperSession {
    std::shared_ptr<brosoundml::Whisper> model;
    std::unique_ptr<brosoundml::WhisperSession> session;
    bool loaded = true;
};

struct HostParakeetTokenizer {
    bool loaded = true;
};

struct HostParakeetModel {
    std::shared_ptr<brosoundml::Parakeet> model;
    std::string device = "CPU";
    bool loaded = true;
};

struct HostParakeetSession {
    std::shared_ptr<brosoundml::Parakeet> model;
    std::unique_ptr<brosoundml::ParakeetSession> session;
    bool loaded = true;
};

struct HostQwenAsrModel {
    std::shared_ptr<brosoundml::QwenAsr> model;
    std::string device = "CPU";
    bool loaded = true;
};

struct HostQwenAsrSession {
    std::shared_ptr<brosoundml::QwenAsr> model;
    std::unique_ptr<brosoundml::QwenAsrSession> session;
    bool loaded = true;
};

struct HostQwenAsrStream {
    std::shared_ptr<brosoundml::QwenAsrStream> stream;
    bool loaded = true;
};

Value makeTranscribeResult(const std::vector<int32_t>& tokenIds,
                           const std::vector<int32_t>& frameOffsets = {}) {
    ObjectBuilder res;
    res.set("tokenIds", makeInt32Array(tokenIds));
    if (!frameOffsets.empty()) {
        res.set("frameOffsets", makeInt32Array(frameOffsets));
    }
    return res.get();
}

} // namespace

void installStt(ObjectBuilder& bro) {
    // -----------------------------------------------------------------------
    // WhisperTokenizer
    // -----------------------------------------------------------------------
    g_whisperTokenizerClass.install("WhisperTokenizer", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostWhisperTokenizer();
            return g_whisperTokenizerClass.make(h, [](void* p) { delete static_cast<HostWhisperTokenizer*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostWhisperTokenizer*>(g_whisperTokenizerClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.def("encode", 1, [](Value, std::span<const Value> a) -> Value {
                std::string text = strAt(a, 0);
                std::vector<int32_t> tokens;
                tokens.reserve(text.size());
                for (unsigned char c : text) tokens.push_back(static_cast<int32_t>(c));
                return makeInt32Array(tokens);
            });
            b.def("decode", 1, [](Value, std::span<const Value> a) -> Value {
                auto tokens = extractInt32Array(argAt(a, 0));
                std::string s;
                s.reserve(tokens.size());
                for (int32_t t : tokens) {
                    if (t >= 32 && t <= 126) s.push_back(static_cast<char>(t));
                    else s.push_back(' ');
                }
                return ev::fromUtf8(s);
            });
            b.def("buildPrompt", 1, [](Value, std::span<const Value>) -> Value {
                const int32_t defaultPrompt[] = {50258, 50259, 50359, 50363};
                return makeInt32Array(defaultPrompt);
            });
        });

    // -----------------------------------------------------------------------
    // WhisperSession
    // -----------------------------------------------------------------------
    g_whisperSessionClass.install("WhisperSession", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostWhisperSession();
            return g_whisperSessionClass.make(h, [](void* p) { delete static_cast<HostWhisperSession*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostWhisperSession*>(g_whisperSessionClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostWhisperSession*>(g_whisperSessionClass.unwrap(self));
                if (h && h->model && h->session) h->model->reset(*h->session);
                return ev::undefined();
            });
            b.def("transcribe", 3, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostWhisperSession*>(g_whisperSessionClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));
                Value opts = (a.size() >= 3 && ev::isObject(a[2])) ? a[2]
                           : (a.size() >= 2 && ev::isObject(a[1])) ? a[1] : ev::undefined();
                std::vector<int32_t> tokens;
                if (h && h->model && h->session && !audio.empty()) {
                    try {
                        brosoundml::AudioBuffer ab;
                        ab.samples = std::move(audio);
                        ab.sample_rate = 16000;
                        auto t = h->model->transcribe(*h->session, ab, {});
                        tokens = std::move(t.token_ids);
                    } catch (...) {}
                }
                Value res = makeTranscribeResult(tokens);
                if (ev::isObject(opts)) {
                    Value onDone = ev::getProperty(opts, "onDone");
                    if (ev::isFunction(onDone)) triggerCallback(onDone, res);
                }
                return res;
            });
        });

    // -----------------------------------------------------------------------
    // WhisperModel
    // -----------------------------------------------------------------------
    g_whisperModelClass.install("WhisperModel", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostWhisperModel();
            return g_whisperModelClass.make(h, [](void* p) { delete static_cast<HostWhisperModel*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostWhisperModel*>(g_whisperModelClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("device", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostWhisperModel*>(g_whisperModelClass.unwrap(self));
                return ev::fromUtf8(h ? h->device : "CPU");
            });
            b.def("createSession", 0, [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostWhisperModel*>(g_whisperModelClass.unwrap(self));
                auto* sess = new HostWhisperSession();
                sess->model = h ? h->model : nullptr;
                if (h && h->model) {
                    try {
                        sess->session = std::make_unique<brosoundml::WhisperSession>(h->model->make_session());
                    } catch (...) {}
                }
                return g_whisperSessionClass.make(sess, [](void* p) { delete static_cast<HostWhisperSession*>(p); });
            });
            b.def("transcribe", 3, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostWhisperModel*>(g_whisperModelClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));
                Value opts = (a.size() >= 3 && ev::isObject(a[2])) ? a[2]
                           : (a.size() >= 2 && ev::isObject(a[1])) ? a[1] : ev::undefined();
                std::vector<int32_t> tokens;
                if (h && h->model && !audio.empty()) {
                    try {
                        brosoundml::AudioBuffer ab;
                        ab.samples = std::move(audio);
                        ab.sample_rate = 16000;
                        auto t = h->model->transcribe(ab, {});
                        tokens = std::move(t.token_ids);
                    } catch (...) {}
                }
                Value res = makeTranscribeResult(tokens);
                if (ev::isObject(opts)) {
                    Value onDone = ev::getProperty(opts, "onDone");
                    if (ev::isFunction(onDone)) triggerCallback(onDone, res);
                }
                return res;
            });
        });

    // -----------------------------------------------------------------------
    // ParakeetTokenizer
    // -----------------------------------------------------------------------
    g_parakeetTokenizerClass.install("ParakeetTokenizer", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostParakeetTokenizer();
            return g_parakeetTokenizerClass.make(h, [](void* p) { delete static_cast<HostParakeetTokenizer*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostParakeetTokenizer*>(g_parakeetTokenizerClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.def("encode", 1, [](Value, std::span<const Value> a) -> Value {
                std::string text = strAt(a, 0);
                std::vector<int32_t> tokens;
                for (unsigned char c : text) tokens.push_back(static_cast<int32_t>(c));
                return makeInt32Array(tokens);
            });
            b.def("decode", 1, [](Value, std::span<const Value> a) -> Value {
                auto tokens = extractInt32Array(argAt(a, 0));
                std::string s;
                for (int32_t t : tokens) {
                    if (t >= 32 && t <= 126) s.push_back(static_cast<char>(t));
                }
                return ev::fromUtf8(s);
            });
        });

    // -----------------------------------------------------------------------
    // ParakeetSession
    // -----------------------------------------------------------------------
    g_parakeetSessionClass.install("ParakeetSession", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostParakeetSession();
            return g_parakeetSessionClass.make(h, [](void* p) { delete static_cast<HostParakeetSession*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostParakeetSession*>(g_parakeetSessionClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostParakeetSession*>(g_parakeetSessionClass.unwrap(self));
                if (h && h->model && h->session) h->model->reset(*h->session);
                return ev::undefined();
            });
            b.def("transcribe", 2, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostParakeetSession*>(g_parakeetSessionClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));
                Value opts = argAt(a, 1);
                std::vector<int32_t> tokens;
                std::vector<int32_t> offsets;
                if (h && h->model && h->session && !audio.empty()) {
                    try {
                        brosoundml::AudioBuffer ab;
                        ab.samples = std::move(audio);
                        ab.sample_rate = 16000;
                        auto t = h->model->transcribe(*h->session, ab);
                        tokens = std::move(t.token_ids);
                        offsets = std::move(t.token_frames);
                    } catch (...) {}
                }
                Value res = makeTranscribeResult(tokens, offsets);
                if (ev::isObject(opts)) {
                    Value onDone = ev::getProperty(opts, "onDone");
                    if (ev::isFunction(onDone)) triggerCallback(onDone, res);
                }
                return res;
            });
        });

    // -----------------------------------------------------------------------
    // ParakeetModel
    // -----------------------------------------------------------------------
    g_parakeetModelClass.install("ParakeetModel", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostParakeetModel();
            return g_parakeetModelClass.make(h, [](void* p) { delete static_cast<HostParakeetModel*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostParakeetModel*>(g_parakeetModelClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("device", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostParakeetModel*>(g_parakeetModelClass.unwrap(self));
                return ev::fromUtf8(h ? h->device : "CPU");
            });
            b.def("createSession", 0, [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostParakeetModel*>(g_parakeetModelClass.unwrap(self));
                auto* sess = new HostParakeetSession();
                sess->model = h ? h->model : nullptr;
                if (h && h->model) {
                    try {
                        sess->session = std::make_unique<brosoundml::ParakeetSession>(h->model->make_session());
                    } catch (...) {}
                }
                return g_parakeetSessionClass.make(sess, [](void* p) { delete static_cast<HostParakeetSession*>(p); });
            });
            b.def("transcribe", 2, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostParakeetModel*>(g_parakeetModelClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));
                Value opts = argAt(a, 1);
                std::vector<int32_t> tokens;
                std::vector<int32_t> offsets;
                if (h && h->model && !audio.empty()) {
                    try {
                        brosoundml::AudioBuffer ab;
                        ab.samples = std::move(audio);
                        ab.sample_rate = 16000;
                        auto t = h->model->transcribe(ab);
                        tokens = std::move(t.token_ids);
                        offsets = std::move(t.token_frames);
                    } catch (...) {}
                }
                Value res = makeTranscribeResult(tokens, offsets);
                if (ev::isObject(opts)) {
                    Value onDone = ev::getProperty(opts, "onDone");
                    if (ev::isFunction(onDone)) triggerCallback(onDone, res);
                }
                return res;
            });
        });

    // -----------------------------------------------------------------------
    // QwenAsrSession
    // -----------------------------------------------------------------------
    g_qwenAsrSessionClass.install("QwenAsrSession", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostQwenAsrSession();
            return g_qwenAsrSessionClass.make(h, [](void* p) { delete static_cast<HostQwenAsrSession*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenAsrSession*>(g_qwenAsrSessionClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenAsrSession*>(g_qwenAsrSessionClass.unwrap(self));
                if (h && h->model && h->session) h->model->reset(*h->session);
                return ev::undefined();
            });
            b.def("transcribe", 2, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostQwenAsrSession*>(g_qwenAsrSessionClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));
                Value opts = argAt(a, 1);
                std::vector<int32_t> tokens;
                if (h && h->model && h->session && !audio.empty()) {
                    try {
                        brosoundml::AudioBuffer ab;
                        ab.samples = std::move(audio);
                        ab.sample_rate = 16000;
                        auto t = h->model->transcribe(*h->session, ab);
                        tokens = std::move(t.token_ids);
                    } catch (...) {}
                }
                Value res = makeTranscribeResult(tokens);
                if (ev::isObject(opts)) {
                    Value onDone = ev::getProperty(opts, "onDone");
                    if (ev::isFunction(onDone)) triggerCallback(onDone, res);
                }
                return res;
            });
        });

    // -----------------------------------------------------------------------
    // QwenAsrModel
    // -----------------------------------------------------------------------
    g_qwenAsrModelClass.install("QwenAsrModel", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostQwenAsrModel();
            return g_qwenAsrModelClass.make(h, [](void* p) { delete static_cast<HostQwenAsrModel*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenAsrModel*>(g_qwenAsrModelClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("device", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenAsrModel*>(g_qwenAsrModelClass.unwrap(self));
                return ev::fromUtf8(h ? h->device : "CPU");
            });
            b.def("createSession", 0, [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenAsrModel*>(g_qwenAsrModelClass.unwrap(self));
                auto* sess = new HostQwenAsrSession();
                sess->model = h ? h->model : nullptr;
                if (h && h->model) {
                    try {
                        sess->session = std::make_unique<brosoundml::QwenAsrSession>(h->model->make_session());
                    } catch (...) {}
                }
                return g_qwenAsrSessionClass.make(sess, [](void* p) { delete static_cast<HostQwenAsrSession*>(p); });
            });
            b.def("transcribe", 2, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostQwenAsrModel*>(g_qwenAsrModelClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));
                Value opts = argAt(a, 1);
                std::vector<int32_t> tokens;
                if (h && h->model && !audio.empty()) {
                    try {
                        brosoundml::AudioBuffer ab;
                        ab.samples = std::move(audio);
                        ab.sample_rate = 16000;
                        auto t = h->model->transcribe(ab);
                        tokens = std::move(t.token_ids);
                    } catch (...) {}
                }
                Value res = makeTranscribeResult(tokens);
                if (ev::isObject(opts)) {
                    Value onDone = ev::getProperty(opts, "onDone");
                    if (ev::isFunction(onDone)) triggerCallback(onDone, res);
                }
                return res;
            });
        });

    // -----------------------------------------------------------------------
    // QwenAsrStream
    // -----------------------------------------------------------------------
    g_qwenAsrStreamClass.install("QwenAsrStream", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* h = new HostQwenAsrStream();
            return g_qwenAsrStreamClass.make(h, [](void* p) { delete static_cast<HostQwenAsrStream*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenAsrStream*>(g_qwenAsrStreamClass.unwrap(self));
                return ev::fromBool(h && h->loaded);
            });
            b.accessor("frames", [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenAsrStream*>(g_qwenAsrStreamClass.unwrap(self));
                return ev::fromDouble(h && h->stream ? h->stream->frames() : 0);
            });
            b.def("feed", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* h = static_cast<HostQwenAsrStream*>(g_qwenAsrStreamClass.unwrap(self));
                auto audio = extractFloatAudio(argAt(a, 0));
                int newFrames = 0;
                if (h && h->stream && !audio.empty()) {
                    try {
                        newFrames = h->stream->feed(audio.data(), static_cast<int>(audio.size()));
                    } catch (...) {}
                }
                return ev::fromDouble(newFrames);
            });
            b.def("finish", 0, [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenAsrStream*>(g_qwenAsrStreamClass.unwrap(self));
                if (h && h->stream) {
                    try {
                        h->stream->finish();
                    } catch (...) {}
                }
                ObjectBuilder res;
                res.set("frames", ev::fromDouble(h && h->stream ? h->stream->frames() : 0));
                if (h && h->stream) {
                    res.set("latents", makeFloat32Array(h->stream->latents()));
                } else {
                    res.set("latents", makeFloat32Array({}));
                }
                return res.get();
            });
            b.def("latents", 0, [](Value self, std::span<const Value>) -> Value {
                auto* h = static_cast<HostQwenAsrStream*>(g_qwenAsrStreamClass.unwrap(self));
                if (h && h->stream) {
                    return makeFloat32Array(h->stream->latents());
                }
                return makeFloat32Array({});
            });
        });

    // -----------------------------------------------------------------------
    // bro.stt namespace
    // -----------------------------------------------------------------------
    ObjectBuilder stt;

    stt.def("init", 0, [](Value, std::span<const Value>) -> Value {
        brotensor::init();
        return ev::undefined();
    });

    stt.def("loadWhisper", 2, [](Value, std::span<const Value> a) -> Value {
        std::string dir = strAt(a, 0);
        Value opts = argAt(a, 1);
        brotensor::Device dev = parseDeviceOpt(opts);
        auto* h = new HostWhisperModel();
        h->device = deviceName(dev);
        if (!dir.empty()) {
            try {
                h->model = std::make_shared<brosoundml::Whisper>();
                h->model->load(dir, dev);
            } catch (...) {}
        }
        Value modelVal = g_whisperModelClass.make(h, [](void* p) { delete static_cast<HostWhisperModel*>(p); });
        if (ev::isObject(opts)) {
            Value onReady = ev::getProperty(opts, "onReady");
            if (ev::isFunction(onReady)) triggerCallback(onReady, modelVal);
        }
        return modelVal;
    });

    stt.def("loadTokenizer", 1, [](Value, std::span<const Value>) -> Value {
        auto* h = new HostWhisperTokenizer();
        return g_whisperTokenizerClass.make(h, [](void* p) { delete static_cast<HostWhisperTokenizer*>(p); });
    });

    stt.def("loadParakeet", 2, [](Value, std::span<const Value> a) -> Value {
        std::string dir = strAt(a, 0);
        Value opts = argAt(a, 1);
        brotensor::Device dev = parseDeviceOpt(opts);
        auto* h = new HostParakeetModel();
        h->device = deviceName(dev);
        if (!dir.empty()) {
            try {
                h->model = std::make_shared<brosoundml::Parakeet>();
                h->model->load(dir, dev);
            } catch (...) {}
        }
        Value modelVal = g_parakeetModelClass.make(h, [](void* p) { delete static_cast<HostParakeetModel*>(p); });
        if (ev::isObject(opts)) {
            Value onReady = ev::getProperty(opts, "onReady");
            if (ev::isFunction(onReady)) triggerCallback(onReady, modelVal);
        }
        return modelVal;
    });

    stt.def("loadParakeetTokenizer", 2, [](Value, std::span<const Value>) -> Value {
        auto* h = new HostParakeetTokenizer();
        return g_parakeetTokenizerClass.make(h, [](void* p) { delete static_cast<HostParakeetTokenizer*>(p); });
    });

    stt.def("loadQwenAsr", 2, [](Value, std::span<const Value> a) -> Value {
        std::string dir = strAt(a, 0);
        Value opts = argAt(a, 1);
        brotensor::Device dev = parseDeviceOpt(opts);
        auto* h = new HostQwenAsrModel();
        h->device = deviceName(dev);
        if (!dir.empty()) {
            try {
                h->model = std::make_shared<brosoundml::QwenAsr>();
                h->model->load(dir, dev);
            } catch (...) {}
        }
        Value modelVal = g_qwenAsrModelClass.make(h, [](void* p) { delete static_cast<HostQwenAsrModel*>(p); });
        if (ev::isObject(opts)) {
            Value onReady = ev::getProperty(opts, "onReady");
            if (ev::isFunction(onReady)) triggerCallback(onReady, modelVal);
        }
        return modelVal;
    });

    stt.def("loadQwenAsrStream", 2, [](Value, std::span<const Value> a) -> Value {
        std::string dir = strAt(a, 0);
        Value opts = argAt(a, 1);
        brotensor::Device dev = parseDeviceOpt(opts);
        int blockChunks = getPropertyInt(opts, "blockChunks", 1);
        auto* h = new HostQwenAsrStream();
        if (!dir.empty()) {
            try {
                h->stream = std::make_shared<brosoundml::QwenAsrStream>();
                h->stream->load(dir, blockChunks > 0 ? blockChunks : 1, dev);
            } catch (...) {}
        }
        Value streamVal = g_qwenAsrStreamClass.make(h, [](void* p) { delete static_cast<HostQwenAsrStream*>(p); });
        if (ev::isObject(opts)) {
            Value onReady = ev::getProperty(opts, "onReady");
            if (ev::isFunction(onReady)) triggerCallback(onReady, streamVal);
        }
        return streamVal;
    });

    stt.def("transcribe", 4, [](Value, std::span<const Value> a) -> Value {
        Value modelVal = argAt(a, 0);
        auto audio = extractFloatAudio(argAt(a, 1));
        Value opts = (a.size() >= 4 && ev::isObject(a[3])) ? a[3]
                   : (a.size() >= 3 && ev::isObject(a[2])) ? a[2] : ev::undefined();

        std::vector<int32_t> tokens;
        std::vector<int32_t> offsets;

        auto* wm = static_cast<HostWhisperModel*>(g_whisperModelClass.unwrap(modelVal));
        if (wm && wm->model && !audio.empty()) {
            try {
                brosoundml::AudioBuffer ab;
                ab.samples = std::move(audio);
                ab.sample_rate = 16000;
                auto t = wm->model->transcribe(ab, {});
                tokens = std::move(t.token_ids);
            } catch (...) {}
        }
        auto* pm = static_cast<HostParakeetModel*>(g_parakeetModelClass.unwrap(modelVal));
        if (pm && pm->model && !audio.empty()) {
            try {
                brosoundml::AudioBuffer ab;
                ab.samples = std::move(audio);
                ab.sample_rate = 16000;
                auto t = pm->model->transcribe(ab);
                tokens = std::move(t.token_ids);
                offsets = std::move(t.token_frames);
            } catch (...) {}
        }
        auto* qm = static_cast<HostQwenAsrModel*>(g_qwenAsrModelClass.unwrap(modelVal));
        if (qm && qm->model && !audio.empty()) {
            try {
                brosoundml::AudioBuffer ab;
                ab.samples = std::move(audio);
                ab.sample_rate = 16000;
                auto t = qm->model->transcribe(ab);
                tokens = std::move(t.token_ids);
            } catch (...) {}
        }

        Value res = makeTranscribeResult(tokens, offsets);
        if (ev::isObject(opts)) {
            Value onDone = ev::getProperty(opts, "onDone");
            if (ev::isFunction(onDone)) triggerCallback(onDone, res);
        }
        return res;
    });

    // Expose class constructors on bro.stt
    stt.set("WhisperTokenizer", g_whisperTokenizerClass.constructor());
    stt.set("WhisperModel", g_whisperModelClass.constructor());
    stt.set("WhisperSession", g_whisperSessionClass.constructor());
    stt.set("ParakeetTokenizer", g_parakeetTokenizerClass.constructor());
    stt.set("ParakeetModel", g_parakeetModelClass.constructor());
    stt.set("ParakeetSession", g_parakeetSessionClass.constructor());
    stt.set("QwenAsrModel", g_qwenAsrModelClass.constructor());
    stt.set("QwenAsrSession", g_qwenAsrSessionClass.constructor());
    stt.set("QwenAsrStream", g_qwenAsrStreamClass.constructor());

    bro.set("stt", stt.get());
}

} // namespace brosoundml::api
