#include "host_soundml_internal.h"
#include "soundml_stt_internal.h"
#include "soundml_tts_internal.h"
#include "brosoundml/voice_agent.h"
#include "brosoundml/whisper.h"
#include "brosoundml/kokoro.h"
#include "brosoundml/bc_resnet2d.h"

#include <memory>
#include <string>
#include <vector>

namespace brosoundml::api {

HostClass g_voiceAgentClass;
HostClass g_bcResnet2dClass;

struct HostBcResnet2d {
    std::shared_ptr<const BcResnet2d> model;
};

struct HostVoiceAgent {
    std::unique_ptr<VoiceAgent> agent;
    ev::Persistent onStateChangedCb;
    ev::Persistent onSpeechStartCb;
    ev::Persistent onSpeechEndCb;
    ev::Persistent onTranscriptCb;
    ev::Persistent onResponseTextCb;
    ev::Persistent onAudioOutputCb;
    ev::Persistent onBargeInCb;
    ev::Persistent textHandlerCb;
    ev::Persistent sttHandlerCb;
    ev::Persistent ttsHandlerCb;
    ev::Persistent tokenizerCb;
    ev::Persistent phonemizerCb;
};

namespace {

HostVoiceAgent* agentSelf(Value self) {
    if (!g_voiceAgentClass.isInstance(self)) return nullptr;
    return static_cast<HostVoiceAgent*>(g_voiceAgentClass.unwrap(self));
}

inline const char* voiceAgentStateToJs(VoiceAgentState s) {
    switch (s) {
        case VoiceAgentState::Idle:      return "idle";
        case VoiceAgentState::Listening: return "listening";
        case VoiceAgentState::Thinking:  return "thinking";
        case VoiceAgentState::Speaking:  return "speaking";
    }
    return "unknown";
}

void wireAgentCallbacks(HostVoiceAgent* host) {
    host->agent->on_state_changed([host](VoiceAgentState old_s, VoiceAgentState new_s) {
        if (ev::isFunction(host->onStateChangedCb.get())) {
            Value a0 = ev::fromUtf8(voiceAgentStateToJs(old_s));
            Value a1 = ev::fromUtf8(voiceAgentStateToJs(new_s));
            callCallback2(host->onStateChangedCb.get(), a0, a1);
        }
    });

    host->agent->on_speech_start([host]() {
        if (ev::isFunction(host->onSpeechStartCb.get())) {
            callCallback(host->onSpeechStartCb.get(), {});
        }
    });

    host->agent->on_speech_end([host](const AudioBuffer& utterance) {
        if (ev::isFunction(host->onSpeechEndCb.get())) {
            Value a0 = makeFloat32Array(utterance.samples);
            callCallback1(host->onSpeechEndCb.get(), a0);
        }
    });

    host->agent->on_transcript([host](const std::string& transcript) {
        if (ev::isFunction(host->onTranscriptCb.get())) {
            Value a0 = ev::fromUtf8(transcript);
            callCallback1(host->onTranscriptCb.get(), a0);
        }
    });

    host->agent->on_response_text([host](const std::string& responseText) {
        if (ev::isFunction(host->onResponseTextCb.get())) {
            Value a0 = ev::fromUtf8(responseText);
            callCallback1(host->onResponseTextCb.get(), a0);
        }
    });

    host->agent->on_audio_output([host](const AudioBuffer& audio) {
        if (ev::isFunction(host->onAudioOutputCb.get())) {
            Value a0 = makeFloat32Array(audio.samples);
            callCallback1(host->onAudioOutputCb.get(), a0);
        }
    });

    host->agent->on_barge_in([host]() {
        if (ev::isFunction(host->onBargeInCb.get())) {
            callCallback(host->onBargeInCb.get(), {});
        }
    });
}

void attachSttHandler(HostVoiceAgent* h, Value fn) {
    if (ev::isUndefined(fn) || ev::isNull(fn)) {
        h->sttHandlerCb = ev::Persistent();
        h->agent->set_stt_handler(nullptr);
    } else if (ev::isFunction(fn)) {
        h->sttHandlerCb = ev::Persistent(fn);
        h->agent->set_stt_handler([h](const AudioBuffer& utterance) -> std::string {
            if (!ev::isFunction(h->sttHandlerCb.get())) return "";
            Value a0 = makeFloat32Array(utterance.samples);
            ev::CallResult r = ev::call(h->sttHandlerCb.get(), ev::undefined(), std::span<const Value>(&a0, 1));
            if (!r.thrown && ev::isString(r.value)) {
                return ev::toUtf8(r.value);
            }
            return "";
        });
    }
}

void attachTextHandler(HostVoiceAgent* h, Value fn) {
    if (ev::isUndefined(fn) || ev::isNull(fn)) {
        h->textHandlerCb = ev::Persistent();
        h->agent->set_text_handler(nullptr);
    } else if (ev::isFunction(fn)) {
        h->textHandlerCb = ev::Persistent(fn);
        h->agent->set_text_handler([h](const std::string& query) -> std::string {
            if (!ev::isFunction(h->textHandlerCb.get())) return "";
            Value a0 = ev::fromUtf8(query);
            ev::CallResult r = ev::call(h->textHandlerCb.get(), ev::undefined(), std::span<const Value>(&a0, 1));
            if (!r.thrown && ev::isString(r.value)) {
                return ev::toUtf8(r.value);
            }
            return "";
        });
    }
}

void attachTtsHandler(HostVoiceAgent* h, Value fn) {
    if (ev::isUndefined(fn) || ev::isNull(fn)) {
        h->ttsHandlerCb = ev::Persistent();
        h->agent->set_tts_handler(nullptr);
    } else if (ev::isFunction(fn)) {
        h->ttsHandlerCb = ev::Persistent(fn);
        h->agent->set_tts_handler([h](const std::string& text, VoiceAgent::TtsChunkCallback chunk_cb) {
            if (!ev::isFunction(h->ttsHandlerCb.get())) return;
            Value emitCb = ev::makeFunction([&chunk_cb, h](Value, std::span<const Value> cArgs) -> Value {
                if (cArgs.empty()) return ev::undefined();
                AudioBuffer buf;
                buf.sample_rate = h->agent->config().tts_sample_rate;
                bool ok = false;
                if (ev::isObject(cArgs[0]) && !ev::isTypedArray(cArgs[0]) && hasProperty(cArgs[0], "samples")) {
                    buf.samples = readFloat32Array(ev::getProperty(cArgs[0], "samples"), &ok);
                    buf.sample_rate = getPropertyInt(cArgs[0], "sampleRate", buf.sample_rate);
                } else {
                    buf.samples = readFloat32Array(cArgs[0], &ok);
                }
                if (ok && !buf.samples.empty()) {
                    chunk_cb(buf);
                }
                return ev::undefined();
            }, 1);

            Value a0 = ev::fromUtf8(text);
            const Value callArgs[2] = {a0, emitCb};
            ev::CallResult r = ev::call(h->ttsHandlerCb.get(), ev::undefined(), std::span<const Value>(callArgs, 2));
            if (!r.thrown && !ev::isUndefined(r.value) && !ev::isNull(r.value)) {
                bool ok = false;
                auto samples = readFloat32Array(r.value, &ok);
                if (ok && !samples.empty()) {
                    AudioBuffer buf;
                    buf.sample_rate = h->agent->config().tts_sample_rate;
                    buf.samples = std::move(samples);
                    chunk_cb(buf);
                }
            }
        });
    }
}

void attachTokenizer(HostVoiceAgent* h, Value fn) {
    if (ev::isUndefined(fn) || ev::isNull(fn)) {
        h->tokenizerCb = ev::Persistent();
        h->agent->set_stt_tokenizer(nullptr);
    } else if (g_whisperTokenizerClass.isInstance(fn)) {
        h->tokenizerCb = ev::Persistent(fn);
        auto* wt = static_cast<HostWhisperTokenizer*>(g_whisperTokenizerClass.unwrap(fn));
        if (wt && wt->tok) {
            auto* tok = wt->tok.get();
            h->agent->set_stt_tokenizer([tok](const std::vector<int32_t>& ids) {
                return tok->decode(ids, /*skip_special=*/true);
            });
        }
    } else if (g_parakeetTokenizerClass.isInstance(fn)) {
        h->tokenizerCb = ev::Persistent(fn);
        auto* pt = static_cast<HostParakeetTokenizer*>(g_parakeetTokenizerClass.unwrap(fn));
        if (pt && pt->tok) {
            auto* tok = pt->tok.get();
            h->agent->set_stt_tokenizer([tok](const std::vector<int32_t>& ids) {
                return tok->decode(ids);
            });
        }
    } else if (ev::isFunction(fn)) {
        h->tokenizerCb = ev::Persistent(fn);
        h->agent->set_stt_tokenizer([h](const std::vector<int32_t>& ids) -> std::string {
            if (!ev::isFunction(h->tokenizerCb.get())) return "";
            Value a0 = makeInt32Array(ids);
            ev::CallResult r = ev::call(h->tokenizerCb.get(), ev::undefined(), std::span<const Value>(&a0, 1));
            if (!r.thrown && ev::isString(r.value)) {
                return ev::toUtf8(r.value);
            }
            return "";
        });
    }
}

void attachPhonemizer(HostVoiceAgent* h, Value fn) {
    if (ev::isUndefined(fn) || ev::isNull(fn)) {
        h->phonemizerCb = ev::Persistent();
        h->agent->set_phonemizer(VoiceAgent::PhonemeHandler{});
        h->agent->set_phonemizer(std::shared_ptr<g2p::Phonemizer>{});
    } else if (ev::isFunction(fn)) {
        h->phonemizerCb = ev::Persistent(fn);
        h->agent->set_phonemizer([h](const std::string& text) -> std::vector<int32_t> {
            if (!ev::isFunction(h->phonemizerCb.get())) return {};
            Value a0 = ev::fromUtf8(text);
            ev::CallResult r = ev::call(h->phonemizerCb.get(), ev::undefined(), std::span<const Value>(&a0, 1));
            if (!r.thrown && ev::isTypedArray(r.value)) {
                bool ok = false;
                return readInt32Array(r.value, &ok);
            }
            return {};
        });
    }
}

Value createVoiceAgentInstance(std::span<const Value> a) {
    VoiceAgentConfig cfg;
    if (!a.empty() && ev::isObject(a[0])) {
        Value opts = a[0];
        getIntOpt(opts, "sampleRate", cfg.sample_rate);
        getIntOpt(opts, "sample_rate", cfg.sample_rate);
        getIntOpt(opts, "ttsSampleRate", cfg.tts_sample_rate);
        getIntOpt(opts, "tts_sample_rate", cfg.tts_sample_rate);
        getFloatOpt(opts, "vadThreshold", cfg.vad_threshold);
        getFloatOpt(opts, "vad_threshold", cfg.vad_threshold);
        getFloatOpt(opts, "vadEnergyThreshold", cfg.vad_energy_threshold);
        getFloatOpt(opts, "vad_energy_threshold", cfg.vad_energy_threshold);
        getIntOpt(opts, "minSpeechFrames", cfg.min_speech_frames);
        getIntOpt(opts, "min_speech_frames", cfg.min_speech_frames);
        getIntOpt(opts, "silenceHangoverFrames", cfg.silence_hangover_frames);
        getIntOpt(opts, "silence_hangover_frames", cfg.silence_hangover_frames);
        getIntOpt(opts, "maxUtteranceFrames", cfg.max_utterance_frames);
        getIntOpt(opts, "max_utterance_frames", cfg.max_utterance_frames);
        getBoolOpt(opts, "enableBargeIn", cfg.enable_barge_in);
        getBoolOpt(opts, "enable_barge_in", cfg.enable_barge_in);
        getIntOpt(opts, "hopLength", cfg.hop_length);
        getIntOpt(opts, "hop_length", cfg.hop_length);
        getIntOpt(opts, "winLength", cfg.win_length);
        getIntOpt(opts, "win_length", cfg.win_length);
        getIntOpt(opts, "nMels", cfg.n_mels);
        getIntOpt(opts, "n_mels", cfg.n_mels);
    }

    auto host = std::make_unique<HostVoiceAgent>();
    try {
        host->agent = std::make_unique<VoiceAgent>(cfg);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("VoiceAgent: ") + e.what());
    }

    HostVoiceAgent* rawHost = host.get();
    wireAgentCallbacks(rawHost);

    if (!a.empty() && ev::isObject(a[0])) {
        Value opts = a[0];
        if (hasProperty(opts, "onStateChanged")) rawHost->onStateChangedCb = getFunctionOpt(opts, "onStateChanged");
        if (hasProperty(opts, "onSpeechStart")) rawHost->onSpeechStartCb = getFunctionOpt(opts, "onSpeechStart");
        if (hasProperty(opts, "onSpeechEnd")) rawHost->onSpeechEndCb = getFunctionOpt(opts, "onSpeechEnd");
        if (hasProperty(opts, "onTranscript")) rawHost->onTranscriptCb = getFunctionOpt(opts, "onTranscript");
        if (hasProperty(opts, "onResponseText")) rawHost->onResponseTextCb = getFunctionOpt(opts, "onResponseText");
        if (hasProperty(opts, "onAudioOutput")) rawHost->onAudioOutputCb = getFunctionOpt(opts, "onAudioOutput");
        if (hasProperty(opts, "onBargeIn")) rawHost->onBargeInCb = getFunctionOpt(opts, "onBargeIn");
        if (hasProperty(opts, "sttHandler")) attachSttHandler(rawHost, ev::getProperty(opts, "sttHandler"));
        if (hasProperty(opts, "textHandler")) attachTextHandler(rawHost, ev::getProperty(opts, "textHandler"));
        if (hasProperty(opts, "ttsHandler")) attachTtsHandler(rawHost, ev::getProperty(opts, "ttsHandler"));
        if (hasProperty(opts, "tokenizer")) attachTokenizer(rawHost, ev::getProperty(opts, "tokenizer"));
        if (hasProperty(opts, "sttTokenizer")) attachTokenizer(rawHost, ev::getProperty(opts, "sttTokenizer"));
        if (hasProperty(opts, "phonemizer")) attachPhonemizer(rawHost, ev::getProperty(opts, "phonemizer"));
    }

    return g_voiceAgentClass.createInstance(std::move(host));
}

Value ctorVoiceAgent(Value, std::span<const Value> a) {
    return createVoiceAgentInstance(a);
}

void decorateVoiceAgent(ObjectBuilder& b) {
    // ── Accessors ──
    b.accessor("state", [](Value self, std::span<const Value>) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("state: not a VoiceAgent");
        return ev::fromUtf8(voiceAgentStateToJs(h->agent->state()));
    });

    b.accessor("lastVadScore", [](Value self, std::span<const Value>) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("lastVadScore: not a VoiceAgent");
        return ev::fromDouble(h->agent->last_vad_score());
    });

    b.accessor("lastEnergy", [](Value self, std::span<const Value>) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("lastEnergy: not a VoiceAgent");
        return ev::fromDouble(h->agent->last_energy());
    });

    b.accessor("utteranceFrameCount", [](Value self, std::span<const Value>) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("utteranceFrameCount: not a VoiceAgent");
        return ev::fromDouble(h->agent->utterance_frame_count());
    });

    b.accessor("isIdle", [](Value self, std::span<const Value>) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("isIdle: not a VoiceAgent");
        return ev::fromBool(h->agent->is_idle());
    });

    b.accessor("isListening", [](Value self, std::span<const Value>) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("isListening: not a VoiceAgent");
        return ev::fromBool(h->agent->is_listening());
    });

    b.accessor("isThinking", [](Value self, std::span<const Value>) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("isThinking: not a VoiceAgent");
        return ev::fromBool(h->agent->is_thinking());
    });

    b.accessor("isSpeaking", [](Value self, std::span<const Value>) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("isSpeaking: not a VoiceAgent");
        return ev::fromBool(h->agent->is_speaking());
    });

    b.accessor("config", [](Value self, std::span<const Value>) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("config: not a VoiceAgent");
        const auto& c = h->agent->config();
        ObjectBuilder obj;
        obj.set("sampleRate", static_cast<double>(c.sample_rate));
        obj.set("ttsSampleRate", static_cast<double>(c.tts_sample_rate));
        obj.set("vadThreshold", static_cast<double>(c.vad_threshold));
        obj.set("vadEnergyThreshold", static_cast<double>(c.vad_energy_threshold));
        obj.set("minSpeechFrames", static_cast<double>(c.min_speech_frames));
        obj.set("silenceHangoverFrames", static_cast<double>(c.silence_hangover_frames));
        obj.set("maxUtteranceFrames", static_cast<double>(c.max_utterance_frames));
        obj.set("enableBargeIn", c.enable_barge_in);
        obj.set("hopLength", static_cast<double>(c.hop_length));
        obj.set("winLength", static_cast<double>(c.win_length));
        obj.set("nMels", static_cast<double>(c.n_mels));
        return obj.get();
    });

    // ── Methods ──
    b.def("feed", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("feed: not a VoiceAgent");
        if (a.empty()) return ev::throwTypeError("feed(samples): samples required");
        if (ev::isTypedArray(a[0])) {
            auto info = ev::typedArrayInfo(a[0]);
            if (info.data && info.elementKind == ev::elements::Float32) {
                h->agent->feed(reinterpret_cast<const float*>(info.data), static_cast<int>(info.elementCount));
                return ev::undefined();
            }
        }
        if (ev::isObject(a[0]) && !ev::isTypedArray(a[0]) && hasProperty(a[0], "samples")) {
            Value samplesProp = ev::getProperty(a[0], "samples");
            bool ok = false;
            auto vec = readFloat32Array(samplesProp, &ok);
            if (ok) {
                h->agent->feed(vec.data(), static_cast<int>(vec.size()));
                return ev::undefined();
            }
        }
        bool ok = false;
        auto vec = readFloat32Array(a[0], &ok);
        if (!ok) return ev::throwTypeError("feed(samples): samples must be a Float32Array or number[]");
        h->agent->feed(vec.data(), static_cast<int>(vec.size()));
        return ev::undefined();
    });

    b.def("speak", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("speak: not a VoiceAgent");
        if (!isStringArg(a, 0)) return ev::throwTypeError("speak(text): text required");
        h->agent->speak(strAt(a, 0));
        return ev::undefined();
    });

    b.def("interrupt", 0, [](Value self, std::span<const Value>) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("interrupt: not a VoiceAgent");
        h->agent->interrupt();
        return ev::undefined();
    });

    b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("reset: not a VoiceAgent");
        h->agent->reset();
        return ev::undefined();
    });

    // ── Model Attachments ──
    b.def("setVad", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("setVad: not a VoiceAgent");
        if (a.empty() || ev::isUndefined(a[0]) || ev::isNull(a[0])) {
            h->agent->set_vad_model(nullptr);
            return self;
        }
        Value vadVal = a[0];
        if (g_bcResnet2dClass.isInstance(vadVal)) {
            auto* w = static_cast<HostBcResnet2d*>(g_bcResnet2dClass.unwrap(vadVal));
            if (w && w->model) {
                h->agent->set_vad_model(w->model);
                return self;
            }
        }
        void* ptr = ev::handleData(vadVal);
        if (ptr) {
            auto* w = static_cast<HostBcResnet2d*>(ptr);
            if (w && w->model) {
                h->agent->set_vad_model(w->model);
                return self;
            }
        }
        if (isStringArg(a, 0)) {
            try {
                auto net = std::make_shared<const BcResnet2d>(
                    BcResnet2d::load(resolvePath(strAt(a, 0)), brotensor::Device::CPU));
                h->agent->set_vad_model(net);
                return self;
            } catch (const std::exception& e) {
                return ev::throwError(std::string("setVad: failed to load model: ") + e.what());
            }
        }
        return ev::throwTypeError("setVad(vadModel): valid BcResnet2d model required");
    });

    b.def("setWhisper", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("setWhisper: not a VoiceAgent");
        if (a.empty() || ev::isUndefined(a[0]) || ev::isNull(a[0])) {
            h->agent->set_whisper(nullptr);
            return self;
        }
        Value wVal = a[0];
        std::shared_ptr<Whisper> whisper;
        if (g_whisperModelClass.isInstance(wVal)) {
            auto* wm = static_cast<HostWhisperModel*>(g_whisperModelClass.unwrap(wVal));
            if (wm && wm->model) whisper = wm->model;
        } else if (g_whisperSessionClass.isInstance(wVal)) {
            auto* ws = static_cast<HostWhisperSession*>(g_whisperSessionClass.unwrap(wVal));
            if (ws && ws->model) whisper = ws->model;
        } else {
            void* ptr = ev::handleData(wVal);
            if (ptr) {
                auto* wm = static_cast<HostWhisperModel*>(ptr);
                if (wm && wm->model) whisper = wm->model;
            }
        }
        if (!whisper) {
            return ev::throwTypeError("setWhisper(whisperModel): valid WhisperModel required");
        }
        h->agent->set_whisper(std::move(whisper));
        if (a.size() > 1 && !ev::isUndefined(a[1]) && !ev::isNull(a[1])) {
            attachTokenizer(h, a[1]);
        }
        return self;
    });

    b.def("setParakeet", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("setParakeet: not a VoiceAgent");
        if (a.empty() || ev::isUndefined(a[0]) || ev::isNull(a[0])) {
            h->agent->set_parakeet(nullptr);
            return self;
        }
        Value pVal = a[0];
        std::shared_ptr<Parakeet> parakeet;
        if (g_parakeetModelClass.isInstance(pVal)) {
            auto* pm = static_cast<HostParakeetModel*>(g_parakeetModelClass.unwrap(pVal));
            if (pm && pm->model) parakeet = pm->model;
        } else if (g_parakeetSessionClass.isInstance(pVal)) {
            auto* ps = static_cast<HostParakeetSession*>(g_parakeetSessionClass.unwrap(pVal));
            if (ps && ps->model) parakeet = ps->model;
        } else {
            void* ptr = ev::handleData(pVal);
            if (ptr) {
                auto* pm = static_cast<HostParakeetModel*>(ptr);
                if (pm && pm->model) parakeet = pm->model;
            }
        }
        if (!parakeet) {
            return ev::throwTypeError("setParakeet(parakeetModel): valid ParakeetModel required");
        }
        h->agent->set_parakeet(std::move(parakeet));
        if (a.size() > 1 && !ev::isUndefined(a[1]) && !ev::isNull(a[1])) {
            attachTokenizer(h, a[1]);
        }
        return self;
    });

    b.def("setKokoro", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("setKokoro: not a VoiceAgent");
        if (a.empty() || ev::isUndefined(a[0]) || ev::isNull(a[0])) {
            h->agent->set_kokoro(nullptr, {});
            return self;
        }
        Value kVal = a[0];
        std::shared_ptr<Kokoro> kokoro;
        if (g_kokoroClass.isInstance(kVal)) {
            auto* km = static_cast<HostKokoro*>(g_kokoroClass.unwrap(kVal));
            if (km) kokoro = km->model;
        } else {
            void* ptr = ev::handleData(kVal);
            if (ptr) {
                auto* km = static_cast<HostKokoro*>(ptr);
                if (km) kokoro = km->model;
            }
        }
        if (!kokoro) {
            return ev::throwTypeError("setKokoro(kokoroModel, voice): valid KokoroModel required");
        }

        Voice voice;
        if (a.size() > 1 && !ev::isUndefined(a[1]) && !ev::isNull(a[1])) {
            if (g_voiceClass.isInstance(a[1])) {
                auto* hv = static_cast<HostVoice*>(g_voiceClass.unwrap(a[1]));
                if (hv) voice = hv->voice;
            } else if (isStringArg(a, 1)) {
                std::string vpath = resolvePath(strAt(a, 1));
                try {
                    voice = kokoro->load_voice(vpath);
                } catch (const std::exception& e) {
                    return ev::throwError(std::string("setKokoro: failed to load voice: ") + e.what());
                }
            } else {
                return ev::throwTypeError("setKokoro: voice must be a Voice object or voice path string");
            }
        }
        h->agent->set_kokoro(std::move(kokoro), std::move(voice));
        if (a.size() > 2 && !ev::isUndefined(a[2]) && !ev::isNull(a[2])) {
            attachPhonemizer(h, a[2]);
        }
        return self;
    });

    b.def("setTokenizer", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("setTokenizer: not a VoiceAgent");
        if (a.empty() || ev::isUndefined(a[0]) || ev::isNull(a[0])) {
            attachTokenizer(h, ev::undefined());
        } else {
            attachTokenizer(h, a[0]);
        }
        return self;
    });

    b.def("setSttTokenizer", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("setSttTokenizer: not a VoiceAgent");
        if (a.empty() || ev::isUndefined(a[0]) || ev::isNull(a[0])) {
            attachTokenizer(h, ev::undefined());
        } else {
            attachTokenizer(h, a[0]);
        }
        return self;
    });

    b.def("setPhonemizer", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("setPhonemizer: not a VoiceAgent");
        if (a.empty() || ev::isUndefined(a[0]) || ev::isNull(a[0])) {
            attachPhonemizer(h, ev::undefined());
        } else {
            attachPhonemizer(h, a[0]);
        }
        return self;
    });

    // ── Custom Handlers ──
    b.def("setSttHandler", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("setSttHandler: not a VoiceAgent");
        if (a.empty() || ev::isUndefined(a[0]) || ev::isNull(a[0])) {
            attachSttHandler(h, ev::undefined());
        } else if (ev::isFunction(a[0])) {
            attachSttHandler(h, a[0]);
        } else {
            return ev::throwTypeError("setSttHandler(fn): argument must be a function");
        }
        return self;
    });

    b.def("setTextHandler", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("setTextHandler: not a VoiceAgent");
        if (a.empty() || ev::isUndefined(a[0]) || ev::isNull(a[0])) {
            attachTextHandler(h, ev::undefined());
        } else if (ev::isFunction(a[0])) {
            attachTextHandler(h, a[0]);
        } else {
            return ev::throwTypeError("setTextHandler(fn): argument must be a function");
        }
        return self;
    });

    b.def("setTtsHandler", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* h = agentSelf(self);
        if (!h || !h->agent) return ev::throwTypeError("setTtsHandler: not a VoiceAgent");
        if (a.empty() || ev::isUndefined(a[0]) || ev::isNull(a[0])) {
            attachTtsHandler(h, ev::undefined());
        } else if (ev::isFunction(a[0])) {
            attachTtsHandler(h, a[0]);
        } else {
            return ev::throwTypeError("setTtsHandler(fn): argument must be a function");
        }
        return self;
    });

    // ── Callback Setters ──
    auto makeSetter = [](const char* name, ev::Persistent HostVoiceAgent::*member) {
        return [name, member](Value self, std::span<const Value> a) -> Value {
            auto* h = agentSelf(self);
            if (!h || !h->agent) return ev::throwTypeError(std::string(name) + ": not a VoiceAgent");
            if (a.empty() || ev::isUndefined(a[0]) || ev::isNull(a[0])) {
                h->*member = ev::Persistent();
            } else if (ev::isFunction(a[0])) {
                h->*member = ev::Persistent(a[0]);
            } else {
                return ev::throwTypeError(std::string(name) + ": argument must be a function");
            }
            return self;
        };
    };

    b.def("onStateChanged", 1, makeSetter("onStateChanged", &HostVoiceAgent::onStateChangedCb));
    b.def("onSpeechStart",  1, makeSetter("onSpeechStart",  &HostVoiceAgent::onSpeechStartCb));
    b.def("onSpeechEnd",    1, makeSetter("onSpeechEnd",    &HostVoiceAgent::onSpeechEndCb));
    b.def("onTranscript",   1, makeSetter("onTranscript",   &HostVoiceAgent::onTranscriptCb));
    b.def("onResponseText", 1, makeSetter("onResponseText", &HostVoiceAgent::onResponseTextCb));
    b.def("onAudioOutput",  1, makeSetter("onAudioOutput",  &HostVoiceAgent::onAudioOutputCb));
    b.def("onBargeIn",      1, makeSetter("onBargeIn",      &HostVoiceAgent::onBargeInCb));
}

Value ctorBcResnet(Value, std::span<const Value> a) {
    if (a.empty()) return ev::throwTypeError("BcResnet2d: path or options required");
    try {
        std::shared_ptr<const BcResnet2d> net;
        if (isStringArg(a, 0)) {
            std::string path = resolvePath(strAt(a, 0));
            brotensor::Device dev = brotensor::Device::CPU;
            if (a.size() > 1 && ev::isObject(a[1])) {
                std::string err;
                parseDeviceOpt(a[1], dev, err);
            }
            net = std::make_shared<const BcResnet2d>(BcResnet2d::load(path, dev));
        } else if (isObjectArg(a, 0)) {
            Value opts = a[0];
            brotensor::Device dev = brotensor::Device::CPU;
            std::string err;
            parseDeviceOpt(opts, dev, err);
            if (hasProperty(opts, "weights")) {
                std::string path = resolvePath(getPropertyString(opts, "weights"));
                net = std::make_shared<const BcResnet2d>(BcResnet2d::load(path, dev));
            } else {
                BcResnet2dConfig cfg;
                getIntOpt(opts, "nMels", cfg.n_mels);
                getIntOpt(opts, "n_mels", cfg.n_mels);
                net = std::make_shared<const BcResnet2d>(BcResnet2d::make(cfg, dev));
            }
        } else {
            return ev::throwTypeError("BcResnet2d: expected path string or config object");
        }
        auto hostVad = std::make_unique<HostBcResnet2d>();
        hostVad->model = std::move(net);
        return g_bcResnet2dClass.createInstance(std::move(hostVad));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("BcResnet2d: ") + e.what());
    }
}

void decorateBcResnet(ObjectBuilder& b) {
    b.accessor("loaded", [](Value self, std::span<const Value>) -> Value {
        auto* w = static_cast<HostBcResnet2d*>(g_bcResnet2dClass.unwrap(self));
        return ev::fromBool(w && w->model != nullptr);
    });
}

}  // namespace

void installVoiceAgent(ObjectBuilder& bro) {
    if (!g_voiceAgentClass.installed()) {
        g_voiceAgentClass.install("VoiceAgent", 1, ctorVoiceAgent, decorateVoiceAgent, /*global=*/true);
    }
    if (!g_bcResnet2dClass.installed()) {
        g_bcResnet2dClass.install("BcResnet2d", 1, ctorBcResnet, decorateBcResnet, /*global=*/true);
    }

    Value soundmlVal = ev::getProperty(bro.get(), "soundml");
    if (!ev::isObject(soundmlVal)) {
        soundmlVal = ev::createObject();
        bro.set("soundml", soundmlVal);
    }
    ObjectBuilder soundml(soundmlVal);
    soundml.set("VoiceAgent", g_voiceAgentClass.constructor());
    soundml.set("BcResnet2d", g_bcResnet2dClass.constructor());

    auto createFn = [](Value, std::span<const Value> a) -> Value {
        return createVoiceAgentInstance(a);
    };
    bro.def("createVoiceAgent", 1, createFn);
    soundml.def("createVoiceAgent", 1, createFn);
}

}  // namespace brosoundml::api
