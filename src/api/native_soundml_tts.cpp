// bro.tts — namespace: the loaders (sync, or async through opts.onReady /
// opts.onError), the English phonemizer, the generic synthesize /
// synthesizeStream / decodeFrom dispatch, and the class handles shared by
// native_soundml_tts_{kokoro,qwen,omnivoice,supertonic}.cpp.
//
// Loading is GPU by default (CUDA > Metal > CPU by availability); opts.device
// picks explicitly and must be a string. OmniVoice refuses the CPU unless it
// is asked for by name: its LM is a full Qwen3-0.6B forward per diffusion
// step, and a silent CPU fallback would turn a ~1 s synthesis into minutes.
#include "soundml_tts_internal.h"

#include <brosoundml/g2p/lexicon.h>
#include <brosoundml/g2p/pos_tagger.h>
#include <brosoundml/g2p/morphology.h>
#include <brosoundml/g2p/special_cases.h>
#include <brosoundml/g2p/phonemizer.h>
#include <brosoundml/detail/json.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace brosoundml::api {

HostClass g_kokoroClass;
HostClass g_voiceClass;
HostClass g_kokoroSessionClass;
HostClass g_qwenTtsClass;
HostClass g_qwenTtsSessionClass;
HostClass g_omniVoiceClass;
HostClass g_supertonicClass;
HostClass g_supertonicVoiceClass;
HostClass g_speakerEncoderClass;

namespace {

// Run a build on the JS thread (sync) or on a work thread (opts.onReady is a
// function), wrapping the result in `cls`. The build touches no JS state.
template <typename W>
Value runLoader(const char* fn, Value opts, const HostClass& cls,
                std::function<std::unique_ptr<W>()> build) {
    struct State {
        std::string fn;
        const HostClass* cls = nullptr;
        std::function<std::unique_ptr<W>()> build;
        std::unique_ptr<W> w;
        ev::Persistent onReady, onError;
    };
    auto st = std::make_shared<State>();
    st->fn = fn;
    st->cls = &cls;
    st->build = std::move(build);
    st->onReady = getFunctionOpt(opts, "onReady");
    st->onError = getFunctionOpt(opts, "onError");
    if (!ev::isFunction(st->onReady.get())) {
        try {
            return cls.createInstance(st->build());
        } catch (const std::exception& e) {
            return ev::throwError(st->fn + ": " + e.what());
        }
    }
    auto work = [st](const std::atomic<bool>&) { st->w = st->build(); };
    auto done = [st](bool, const std::string& error) {
        if (!error.empty() || !st->w) {
            if (ev::isFunction(st->onError.get())) {
                ev::Persistent msg(ev::fromUtf8(error.empty() ? st->fn + " failed" : error));
                callCallback1(st->onError.get(), msg.get());
            }
            return;
        }
        ev::Persistent inst(st->cls->createInstance(std::move(st->w)));
        callCallback1(st->onReady.get(), inst.get());
    };
    return launchAsyncJob(std::move(work), nullptr, std::move(done));
}

// Shared prologue of the model loaders: path + device (+ whether opts.device
// was given explicitly).
bool loaderArgs(const char* fn, std::span<const Value> args, std::string& dir, brotensor::Device& dev,
                Value& opts, bool* explicitDevice = nullptr) {
    if (!isStringArg(args, 0)) {
        ev::throwTypeError(std::string(fn) + "(modelDir, opts?): path required");
        return false;
    }
    dir = resolvePath(strAt(args, 0));
    brotensor::init();
    dev = autoDevice();
    opts = isObjectArg(args, 1) ? args[1] : ev::undefined();
    std::string err;
    if (!parseDeviceOpt(opts, dev, err, explicitDevice)) {
        ev::throwTypeError(std::string(fn) + ": " + err);
        return false;
    }
    return true;
}

Value ttsInit(Value, std::span<const Value>) {
    try {
        brotensor::init();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.tts.init: ") + e.what());
    }
    return ev::undefined();
}

// bro.tts.loadKokoro(modelDir, opts?) -> KokoroModel | AsyncHandle
Value loadKokoro(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    Value opts = ev::undefined();
    if (!loaderArgs("loadKokoro", args, dir, dev, opts)) return ev::undefined();
    return runLoader<HostKokoro>("loadKokoro", opts, g_kokoroClass, [dir, dev] {
        auto w = std::make_unique<HostKokoro>();
        w->device = dev;
        w->model = std::make_shared<brosoundml::Kokoro>();
        {
            brotensor::DeviceScope scope(dev);
            w->model->load(dir, dev);
        }
        logInfo(std::string("[tts] Kokoro loaded on ") + deviceName(dev));
        return w;
    });
}

// bro.tts.loadQwen(modelDir, opts?) -> QwenTtsModel | AsyncHandle
Value loadQwen(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    Value opts = ev::undefined();
    if (!loaderArgs("loadQwen", args, dir, dev, opts)) return ev::undefined();
    return runLoader<HostQwenTts>("loadQwen", opts, g_qwenTtsClass, [dir, dev] {
        auto w = std::make_unique<HostQwenTts>();
        w->device = dev;
        w->model = std::make_shared<brosoundml::QwenTts>();
        {
            brotensor::DeviceScope scope(dev);
            w->model->load(dir, dev);
        }
        logInfo(std::string("[tts] Qwen3-TTS loaded on ") + deviceName(dev));
        return w;
    });
}

// bro.tts.loadOmniVoice(modelDir, opts?) -> OmniVoice | AsyncHandle
//   opts.device: GPU by default; the CPU only when named. opts.precision:
//   'bf16' (default on CUDA) | 'fp32' (default elsewhere). opts.decoderOnly:
//   skip the codec encoder + HuBERT (createPrompt / encodeAudio then throw).
Value loadOmniVoice(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    Value opts = ev::undefined();
    bool explicitDevice = false;
    if (!loaderArgs("loadOmniVoice", args, dir, dev, opts, &explicitDevice)) return ev::undefined();
    if (dev.type == brotensor::DeviceType::CPU && !explicitDevice)
        return ev::throwError("loadOmniVoice: no GPU backend is available (CUDA/Metal) and OmniVoice's "
                              "language model is not practical on the CPU; pass { device: 'cpu' } "
                              "to run it there anyway");
    auto precision = dev.type == brotensor::DeviceType::CUDA ? brosoundml::OmniVoicePrecision::BF16
                                                              : brosoundml::OmniVoicePrecision::FP32;
    bool decoderOnly = false;
    if (ev::isObject(opts)) {
        Value pv = ev::getProperty(opts, "precision");
        if (!ev::isUndefined(pv) && !ev::isNull(pv)) {
            const std::string prec = ev::isString(pv) ? ev::toUtf8(pv) : "";
            if (prec == "bf16") precision = brosoundml::OmniVoicePrecision::BF16;
            else if (prec == "fp32") precision = brosoundml::OmniVoicePrecision::FP32;
            else return ev::throwTypeError("loadOmniVoice: opts.precision must be 'fp32' or 'bf16'");
        }
        decoderOnly = getPropertyBool(opts, "decoderOnly");
    }
    return runLoader<HostOmniVoice>("loadOmniVoice", opts, g_omniVoiceClass, [dir, dev, precision, decoderOnly] {
        auto w = std::make_unique<HostOmniVoice>();
        w->device = dev;
        w->precision = precision;
        w->decoderOnly = decoderOnly;
        w->model = std::make_shared<brosoundml::OmniVoice>();
        {
            brotensor::DeviceScope scope(dev);
            w->model->load(dir, dev, precision, decoderOnly);
        }
        logInfo(std::string("[tts] OmniVoice loaded on ") + deviceName(dev) + " (" +
                (precision == brosoundml::OmniVoicePrecision::BF16 ? "bf16" : "fp32") +
                (decoderOnly ? ", codec decoder only)" : ")"));
        return w;
    });
}

// bro.tts.loadSupertonic(modelDir, opts?) -> SupertonicModel | AsyncHandle
Value loadSupertonic(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    Value opts = ev::undefined();
    if (!loaderArgs("loadSupertonic", args, dir, dev, opts)) return ev::undefined();
    return runLoader<HostSupertonic>("loadSupertonic", opts, g_supertonicClass, [dir, dev] {
        auto w = std::make_unique<HostSupertonic>();
        w->device = dev;
        w->model = std::make_shared<brosoundml::Supertonic>();
        {
            brotensor::DeviceScope scope(dev);
            w->model->load(dir, dev);
        }
        logInfo(std::string("[tts] Supertonic loaded on ") + deviceName(dev));
        return w;
    });
}

// bro.tts.loadSpeakerEncoder(dir, opts?) -> SpeakerEncoder | AsyncHandle
//   The encoder places its conv stack on brotensor's default device (the
//   GPU when available); opts.device sets that default first.
Value loadSpeakerEncoder(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    Value opts = ev::undefined();
    if (!loaderArgs("loadSpeakerEncoder", args, dir, dev, opts)) return ev::undefined();
    return runLoader<HostSpeakerEncoder>("loadSpeakerEncoder", opts, g_speakerEncoderClass, [dir, dev] {
        auto w = std::make_unique<HostSpeakerEncoder>();
        w->enc = std::make_shared<brosoundml::SpeakerEncoder>();
        {
            brotensor::DeviceScope scope(dev);
            w->enc->load(dir);
        }
        logInfo(std::string("[tts] speaker encoder loaded on ") + deviceName(dev));
        return w;
    });
}

// ── G2P (text -> Kokoro phoneme ids): a lazily built English phonemizer ──────
//
// Assets: the g2p lexicon and POS-tagger weights (brosoundml-data sibling)
// and a Kokoro config.json (for the phoneme vocab). Explicit setAssets()
// paths win; else the sibling layout rooted at setAssetRoot(); else the
// well-known sibling paths relative to the cwd.
struct PhonemizerState {
    std::unordered_map<std::string, int> vocab;
    std::unique_ptr<brosoundml::g2p::Lexicon> lexicon;
    std::unique_ptr<brosoundml::g2p::PosTagger> tagger;
    std::unique_ptr<brosoundml::g2p::Morphology> morphology;
    std::unique_ptr<brosoundml::g2p::SpecialCases> special;
    std::unique_ptr<brosoundml::g2p::PhonemeAdapter> adapter;
    std::unique_ptr<brosoundml::g2p::Phonemizer> phonemizer;
};

std::string g_assetRoot, g_lexiconPath, g_posPath, g_kokoroConfigPath;
std::unique_ptr<PhonemizerState> g_phonemizer;

bool fileExists(const std::string& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

std::string brosoundmlRoot() {
    if (!g_assetRoot.empty()) return g_assetRoot;
    for (const char* p : {"../brosoundml", "./brosoundml"})
        if (fileExists(std::string(p) + "/weights/kokoro/config.json")) return p;
    return "../brosoundml";
}

std::unordered_map<std::string, int> loadKokoroVocab(const std::string& configPath) {
    namespace j = brosoundml::detail::json;
    std::ifstream f(configPath, std::ios::binary);
    std::ostringstream os;
    os << f.rdbuf();
    const std::string text = os.str();
    if (text.empty()) throw std::runtime_error("config.json empty or unreadable: " + configPath);
    const j::Value root = j::parse(text);
    const j::Value* v = root.find("vocab");
    if (!v || !v->is_object()) throw std::runtime_error("config.json missing 'vocab' object: " + configPath);
    std::unordered_map<std::string, int> vocab;
    for (const auto& m : v->as_object()) vocab.emplace(m.first, static_cast<int>(m.second.as_number()));
    return vocab;
}

std::unique_ptr<PhonemizerState> buildPhonemizer() {
    const std::string repo = brosoundmlRoot();
    const std::string data = repo + "/../brosoundml-data";
    const std::string lexBin = !g_lexiconPath.empty() ? g_lexiconPath : data + "/g2p/lexicon_en_us.bin";
    const std::string posBin = !g_posPath.empty() ? g_posPath : data + "/pos_tagger/model.bin";
    const std::string kokCfg = !g_kokoroConfigPath.empty() ? g_kokoroConfigPath : repo + "/weights/kokoro/config.json";
    const char* hint = " (pass an explicit path via bro.tts.setAssets({...}) or a sibling root via bro.tts.setAssetRoot)";
    if (!fileExists(lexBin)) throw std::runtime_error("missing lexicon: " + lexBin + hint);
    if (!fileExists(posBin)) throw std::runtime_error("missing POS tagger weights: " + posBin + hint);
    if (!fileExists(kokCfg)) throw std::runtime_error("missing Kokoro config.json: " + kokCfg + hint);
    auto st = std::make_unique<PhonemizerState>();
    st->vocab = loadKokoroVocab(kokCfg);
    st->lexicon = std::make_unique<brosoundml::g2p::Lexicon>(brosoundml::g2p::Lexicon::load(lexBin));
    st->tagger = std::make_unique<brosoundml::g2p::PosTagger>(brosoundml::g2p::PosTagger::load(posBin));
    st->morphology = std::make_unique<brosoundml::g2p::Morphology>(*st->lexicon);
    st->special = std::make_unique<brosoundml::g2p::SpecialCases>(*st->lexicon);
    st->adapter = std::make_unique<brosoundml::g2p::PhonemeAdapter>(st->vocab);
    st->phonemizer = std::make_unique<brosoundml::g2p::Phonemizer>(*st->tagger, *st->lexicon, *st->morphology,
                                                                    *st->special, *st->adapter);
    return st;
}

// bro.tts.setAssetRoot(path): the brosoundml repo root; clears explicit paths.
Value setAssetRoot(Value, std::span<const Value> a) {
    if (!isStringArg(a, 0)) return ev::throwTypeError("setAssetRoot(path): path string required");
    g_assetRoot = resolvePath(strAt(a, 0));
    g_lexiconPath.clear();
    g_posPath.clear();
    g_kokoroConfigPath.clear();
    g_phonemizer.reset();
    return ev::undefined();
}

// bro.tts.setAssets({ root?, lexicon?, posTagger? | pos?, kokoroConfig? })
Value setAssets(Value, std::span<const Value> a) {
    if (!isObjectArg(a, 0)) return ev::throwTypeError("setAssets(opts): options object required");
    ev::Persistent opts(a[0]);
    auto take = [&](const char* key, std::string& dst) {
        std::string s;
        getStrOpt(opts.get(), key, s);
        if (!s.empty()) dst = resolvePath(s);
    };
    take("root", g_assetRoot);
    take("lexicon", g_lexiconPath);
    take("posTagger", g_posPath);
    take("pos", g_posPath);
    take("kokoroConfig", g_kokoroConfigPath);
    g_phonemizer.reset();
    return ev::undefined();
}

// bro.tts.phonemize(text) -> Int32Array (English; no language option yet)
Value phonemize(Value, std::span<const Value> a) {
    if (!isStringArg(a, 0)) return ev::throwTypeError("phonemize(text): text required");
    try {
        if (!g_phonemizer) {
            brotensor::init();
            g_phonemizer = buildPhonemizer();
        }
        return makeInt32Array(g_phonemizer->phonemizer->phonemize(strAt(a, 0)));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("phonemize: ") + e.what());
    }
}

// bro.tts.synthesize(model, ...) -> AsyncHandle, dispatched on the class:
//   Kokoro:     synthesize(kokoro, phonemeIds, voice, opts?)
//   QwenTts / Supertonic / OmniVoice: synthesize(model, text, opts?)
Value ttsSynthesize(Value, std::span<const Value> args) {
    if (args.size() < 1)
        return ev::throwTypeError("synthesize(model, ...): a Kokoro, QwenTts, Supertonic, or OmniVoice model is required");
    ev::Persistent modelRoot(args[0]);
    if (auto* q = unwrapTtsAs<HostQwenTts>(g_qwenTtsClass, modelRoot.get()))
        return qwenTtsSynthesizeAsync(modelRoot.get(), q, args);
    if (auto* s = unwrapTtsAs<HostSupertonic>(g_supertonicClass, modelRoot.get()))
        return supertonicSynthesizeAsync(modelRoot.get(), s, args);
    if (unwrapTtsAs<HostOmniVoice>(g_omniVoiceClass, modelRoot.get())) {
        if (args.size() < 2) return ev::throwTypeError("synthesize(omni, text, opts?): text string required");
        return omniVoiceLaunch(modelRoot.get(), args[1], hasArg(args, 2) ? args[2] : ev::undefined(), false);
    }
    if (auto* k = unwrapTtsAs<HostKokoro>(g_kokoroClass, modelRoot.get()))
        return kokoroSynthesizeAsync(modelRoot.get(), k, args);
    return ev::throwTypeError("synthesize: arg 0 must be a Kokoro, QwenTts, Supertonic, or OmniVoice");
}

// bro.tts.synthesizeStream(model, ...) -> AsyncHandle
//   QwenTts: synthesizeStream(qwen, text, opts?)              (chunks the AR tail)
//   Kokoro:  synthesizeStream(kokoro, phonemeChunks, voice, opts?) (chunks the input)
Value ttsSynthesizeStream(Value, std::span<const Value> args) {
    if (args.size() < 1)
        return ev::throwTypeError("synthesizeStream(model, ...): a Kokoro or QwenTts model is required");
    ev::Persistent modelRoot(args[0]);
    if (auto* q = unwrapTtsAs<HostQwenTts>(g_qwenTtsClass, modelRoot.get()))
        return qwenTtsSynthesizeStream(modelRoot.get(), q, args);
    if (auto* k = unwrapTtsAs<HostKokoro>(g_kokoroClass, modelRoot.get()))
        return kokoroSynthesizeStream(modelRoot.get(), k, args);
    return ev::throwTypeError("synthesizeStream: arg 0 must be a Kokoro or QwenTts");
}

// bro.tts.decodeFrom(kokoro, voice, asr, F0, N, nPhonemes, opts?) -> AsyncHandle
Value ttsDecodeFrom(Value, std::span<const Value> args) {
    ev::Persistent modelRoot(hasArg(args, 0) ? args[0] : ev::undefined());
    auto* k = unwrapTtsAs<HostKokoro>(g_kokoroClass, modelRoot.get());
    if (!k)
        return ev::throwTypeError("decodeFrom(kokoro, voice, asr, F0, N, nPhonemes, opts?): a Kokoro model is required");
    return kokoroDecodeFromAsync(modelRoot.get(), k, args);
}

}  // namespace

void installTts(ObjectBuilder& bro) {
    installTtsKokoroClasses();
    installTtsQwenClasses();
    installTtsOmniVoiceClasses();
    installTtsSupertonicClasses();

    ObjectBuilder tts;
    tts.def("init", 0, ttsInit);
    tts.def("loadKokoro", 2, loadKokoro);
    tts.def("loadQwen", 2, loadQwen);
    tts.def("loadOmniVoice", 2, loadOmniVoice);
    tts.def("loadSupertonic", 2, loadSupertonic);
    tts.def("loadSpeakerEncoder", 2, loadSpeakerEncoder);
    tts.def("phonemize", 2, phonemize);
    tts.def("setAssetRoot", 1, setAssetRoot);
    tts.def("setAssets", 1, setAssets);
    tts.def("synthesize", 4, ttsSynthesize);
    tts.def("synthesizeStream", 4, ttsSynthesizeStream);
    tts.def("decodeFrom", 7, ttsDecodeFrom);

    // The class objects (not constructible from JS), for instanceof checks
    // and weights-free introspection of their prototypes.
    tts.set("KokoroModel", g_kokoroClass.constructor());
    tts.set("Voice", g_voiceClass.constructor());
    tts.set("KokoroSession", g_kokoroSessionClass.constructor());
    tts.set("QwenTtsModel", g_qwenTtsClass.constructor());
    tts.set("QwenTtsSession", g_qwenTtsSessionClass.constructor());
    tts.set("OmniVoice", g_omniVoiceClass.constructor());
    tts.set("SupertonicModel", g_supertonicClass.constructor());
    tts.set("SupertonicVoice", g_supertonicVoiceClass.constructor());
    tts.set("SpeakerEncoder", g_speakerEncoderClass.constructor());

    bro.set("tts", tts.get());
}

} // namespace brosoundml::api
