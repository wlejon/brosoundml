// bro.stt — namespace: loaders (sync, or async through opts.onReady /
// opts.onError), the generic bro.stt.transcribe(model, ...) dispatch, and
// the class handles shared by native_soundml_stt_{whisper,parakeet,qwen}.cpp.
//
// Loading is GPU by default (CUDA > Metal > CPU by availability); opts.device
// picks explicitly and must be a string. Every loader resolves its path the
// way the host's fs module does (setPathResolver).
#include "soundml_stt_internal.h"

namespace brosoundml::api {

HostClass g_whisperTokenizerClass;
HostClass g_whisperModelClass;
HostClass g_whisperSessionClass;
HostClass g_parakeetTokenizerClass;
HostClass g_parakeetModelClass;
HostClass g_parakeetSessionClass;
HostClass g_qwenAsrModelClass;
HostClass g_qwenAsrSessionClass;
HostClass g_qwenAsrStreamClass;

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
        ev::Persistent onReady;
        ev::Persistent onError;
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

// Shared prologue of the model loaders: path + device.
bool loaderArgs(const char* fn, std::span<const Value> args, std::string& dir,
                brotensor::Device& dev, Value& opts) {
    if (!isStringArg(args, 0)) {
        ev::throwTypeError(std::string(fn) + "(modelDir, opts?): path required");
        return false;
    }
    dir = resolvePath(strAt(args, 0));
    brotensor::init();
    dev = autoDevice();
    opts = isObjectArg(args, 1) ? args[1] : ev::undefined();
    std::string err;
    if (!parseDeviceOpt(opts, dev, err)) {
        ev::throwTypeError(std::string(fn) + ": " + err);
        return false;
    }
    return true;
}

Value sttInit(Value, std::span<const Value>) {
    try {
        brotensor::init();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.stt.init: ") + e.what());
    }
    return ev::undefined();
}

// bro.stt.loadWhisper(modelDir, opts?) -> WhisperModel | AsyncHandle
Value loadWhisper(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    Value opts = ev::undefined();
    if (!loaderArgs("loadWhisper", args, dir, dev, opts)) return ev::undefined();
    return runLoader<HostWhisperModel>("loadWhisper", opts, g_whisperModelClass, [dir, dev] {
        auto w = std::make_unique<HostWhisperModel>();
        w->device = dev;
        w->model = std::make_shared<brosoundml::Whisper>();
        {
            brotensor::DeviceScope scope(dev);
            w->model->load(dir, dev);
        }
        logInfo(std::string("[stt] Whisper loaded on ") + deviceName(dev));
        return w;
    });
}

// bro.stt.loadTokenizer({ vocabPath, mergesPath, addedTokensPath?, onReady?, onError? })
Value loadTokenizer(Value, std::span<const Value> args) {
    if (!isObjectArg(args, 0)) return ev::throwTypeError("loadTokenizer(opts): opts object required");
    ev::Persistent opts(args[0]);
    std::string vocab = getPropertyString(opts.get(), "vocabPath");
    std::string merges = getPropertyString(opts.get(), "mergesPath");
    if (vocab.empty() || merges.empty())
        return ev::throwTypeError("loadTokenizer: opts.vocabPath and opts.mergesPath required");
    std::string added = getPropertyString(opts.get(), "addedTokensPath");
    vocab = resolvePath(vocab);
    merges = resolvePath(merges);
    if (!added.empty()) added = resolvePath(added);
    return runLoader<HostWhisperTokenizer>("loadTokenizer", opts.get(), g_whisperTokenizerClass,
        [vocab, merges, added] {
            auto t = std::make_unique<HostWhisperTokenizer>();
            t->tok = std::make_unique<brolm::whisper::Tokenizer>(
                brolm::whisper::Tokenizer::load(vocab, merges, added));
            return t;
        });
}

// bro.stt.loadParakeet(modelDir, opts?) -> ParakeetModel | AsyncHandle
Value loadParakeet(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    Value opts = ev::undefined();
    if (!loaderArgs("loadParakeet", args, dir, dev, opts)) return ev::undefined();
    return runLoader<HostParakeetModel>("loadParakeet", opts, g_parakeetModelClass, [dir, dev] {
        auto w = std::make_unique<HostParakeetModel>();
        w->device = dev;
        w->model = std::make_shared<brosoundml::Parakeet>();
        {
            brotensor::DeviceScope scope(dev);
            w->model->load(dir, dev);
        }
        logInfo(std::string("[stt] Parakeet loaded on ") + deviceName(dev));
        return w;
    });
}

// bro.stt.loadParakeetTokenizer(tokenizerJsonPath, opts?) -> ParakeetTokenizer | AsyncHandle
Value loadParakeetTokenizer(Value, std::span<const Value> args) {
    if (!isStringArg(args, 0))
        return ev::throwTypeError("loadParakeetTokenizer(tokenizerJsonPath, opts?): path required");
    const std::string path = resolvePath(strAt(args, 0));
    Value opts = isObjectArg(args, 1) ? args[1] : ev::undefined();
    return runLoader<HostParakeetTokenizer>("loadParakeetTokenizer", opts, g_parakeetTokenizerClass,
        [path] {
            auto t = std::make_unique<HostParakeetTokenizer>();
            t->tok = std::make_unique<brolm::t5::Tokenizer>(brolm::t5::Tokenizer::load(path));
            return t;
        });
}

// bro.stt.loadQwenAsr(modelDir, opts?) -> QwenAsrModel | AsyncHandle
Value loadQwenAsr(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    Value opts = ev::undefined();
    if (!loaderArgs("loadQwenAsr", args, dir, dev, opts)) return ev::undefined();
    return runLoader<HostQwenAsrModel>("loadQwenAsr", opts, g_qwenAsrModelClass, [dir, dev] {
        auto w = std::make_unique<HostQwenAsrModel>();
        w->device = dev;
        w->model = std::make_shared<brosoundml::QwenAsr>();
        {
            brotensor::DeviceScope scope(dev);
            w->model->load(dir, dev);
        }
        logInfo(std::string("[stt] Qwen3-ASR loaded on ") + deviceName(dev));
        return w;
    });
}

// bro.stt.loadQwenAsrStream(modelDir, opts?) -> QwenAsrStream | AsyncHandle
//   opts.blockChunks: block size in ~1 s conv-chunks (default 1).
Value loadQwenAsrStream(Value, std::span<const Value> args) {
    std::string dir;
    brotensor::Device dev = brotensor::Device::CPU;
    Value opts = ev::undefined();
    if (!loaderArgs("loadQwenAsrStream", args, dir, dev, opts)) return ev::undefined();
    int blockChunks = 1;
    getIntOpt(opts, "blockChunks", blockChunks);
    return runLoader<HostQwenAsrStream>("loadQwenAsrStream", opts, g_qwenAsrStreamClass,
        [dir, dev, blockChunks] {
            auto w = std::make_unique<HostQwenAsrStream>();
            w->device = dev;
            w->stream = std::make_unique<brosoundml::QwenAsrStream>();
            {
                brotensor::DeviceScope scope(dev);
                w->stream->load(dir, blockChunks, dev);
            }
            logInfo(std::string("[stt] Qwen3-ASR stream encoder loaded on ") + deviceName(dev));
            return w;
        });
}

// bro.stt.transcribe(model, audio, ...) — dispatch on the model's class:
//   transcribe(whisper, audio, promptIds, opts?)   promptIds required
//   transcribe(parakeet | qwenAsr, audio, opts?)
//   also accepts a session of any of the three.
Value sttTranscribe(Value, std::span<const Value> args) {
    if (args.size() < 2)
        return ev::throwTypeError("transcribe(model, audio, ...): model and audio required");
    ev::Persistent modelRoot(args[0]);
    std::span<const Value> rest = args.subspan(1);

    if (auto* p = unwrapAs<HostParakeetModel>(g_parakeetModelClass, modelRoot.get()))
        return parakeetTranscribe(modelRoot.get(), p, nullptr, rest, "transcribe");
    if (auto* ps = unwrapAs<HostParakeetSession>(g_parakeetSessionClass, modelRoot.get()))
        return parakeetTranscribe(modelRoot.get(), nullptr, ps, rest, "transcribe");
    if (auto* q = unwrapAs<HostQwenAsrModel>(g_qwenAsrModelClass, modelRoot.get()))
        return qwenAsrTranscribe(modelRoot.get(), q, nullptr, rest, "transcribe");
    if (auto* qs = unwrapAs<HostQwenAsrSession>(g_qwenAsrSessionClass, modelRoot.get()))
        return qwenAsrTranscribe(modelRoot.get(), nullptr, qs, rest, "transcribe");

    HostWhisperModel* w = unwrapAs<HostWhisperModel>(g_whisperModelClass, modelRoot.get());
    HostWhisperSession* ws = w ? nullptr
                               : unwrapAs<HostWhisperSession>(g_whisperSessionClass, modelRoot.get());
    if (!w && !ws)
        return ev::throwTypeError("transcribe: arg 0 must be a Whisper, Parakeet, or QwenAsr");
    if (args.size() < 3)
        return ev::throwTypeError(
            "transcribe(whisper, audio, promptIds, opts?): whisper, audio and promptIds required");
    return whisperTranscribe(modelRoot.get(), w, ws, rest, "transcribe");
}

}  // namespace

void installStt(ObjectBuilder& bro) {
    installSttWhisperClasses();
    installSttParakeetClasses();
    installSttQwenClasses();

    ObjectBuilder stt;
    stt.def("init", 0, sttInit);
    stt.def("loadWhisper", 2, loadWhisper);
    stt.def("loadTokenizer", 1, loadTokenizer);
    stt.def("loadParakeet", 2, loadParakeet);
    stt.def("loadParakeetTokenizer", 2, loadParakeetTokenizer);
    stt.def("loadQwenAsr", 2, loadQwenAsr);
    stt.def("loadQwenAsrStream", 2, loadQwenAsrStream);
    stt.def("transcribe", 4, sttTranscribe);

    // The class constructors, for instanceof checks.
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
