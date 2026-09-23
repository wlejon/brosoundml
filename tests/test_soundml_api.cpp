// Standalone Bronze JavaScript API test for brosoundml_api: the namespaces
// mount, the class objects exist and refuse `new`, init() is idempotent, and
// bad arguments raise the documented TypeErrors. No weights, no downloads —
// the same weights-free contract bro's tests/{stt,tts,diar,rave}/*_binding.js
// assert from JavaScript.
//
// GC contract (embed.h): every Value that outlives an allocating call
// (getProperty, call, fromUtf8, ...) rides in an ev::Persistent, so the test
// also passes under BRONZE_GC_STRESS=1 BRONZE_GC_POISON=1 (the _gcstress
// ctest variant).
#include "../src/api/api.h"
#include "embed/embed.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace ev = bronze::embed;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++g_failures;
    std::cerr << "FAIL: " << what << std::endl;
}

// Args built lazily, after the callee is rooted, and rooted themselves.
using ArgMaker = std::vector<ev::Value> (*)();

std::vector<ev::Value> noArgs() { return {}; }

ev::Persistent ns(const ev::Persistent& bro, const char* name) {
    ev::Persistent v(ev::getProperty(bro.get(), name));
    check(ev::isObject(v.get()), std::string("bro.") + name + " is an object");
    return v;
}

// Call obj.fn(args) and require a thrown value whose .name is `errorName` and
// whose message contains `needle` (empty = any message).
void expectThrows(const ev::Persistent& obj, const char* fn, ArgMaker makeArgs,
                  const char* errorName, const char* needle) {
    ev::Persistent f(ev::getProperty(obj.get(), fn));
    check(ev::isFunction(f.get()), std::string(fn) + " is a function");
    if (!ev::isFunction(f.get())) return;
    std::vector<ev::Persistent> roots;
    for (ev::Value v : makeArgs()) roots.emplace_back(v);  // one arg per maker: no allocation between
    std::vector<ev::Value> args;
    for (auto& r : roots) args.push_back(r.get());
    ev::CallResult r = ev::call(f.get(), obj.get(), std::span<const ev::Value>(args));
    check(r.thrown, std::string(fn) + " throws");
    if (!r.thrown) return;
    ev::Persistent err(r.value);
    const std::string name = ev::toUtf8(ev::getProperty(err.get(), "name"));
    const std::string msg = ev::toUtf8(ev::getProperty(err.get(), "message"));
    check(name == errorName, std::string(fn) + " throws a " + errorName + " (got " + name + ": " + msg + ")");
    if (needle && *needle)
        check(msg.find(needle) != std::string::npos,
              std::string(fn) + " message names '" + needle + "': " + msg);
    std::cout << "  " << fn << " -> " << name << ": " << msg << std::endl;
}

void expectInitIdempotent(const ev::Persistent& obj, const char* label) {
    ev::Persistent f(ev::getProperty(obj.get(), "init"));
    check(ev::isFunction(f.get()), std::string(label) + ".init is a function");
    if (!ev::isFunction(f.get())) return;
    for (int i = 0; i < 2; ++i) {
        ev::CallResult r = ev::call(f.get(), obj.get(), {});
        check(!r.thrown && ev::isUndefined(r.value), std::string(label) + ".init() returns undefined");
    }
}

// The class objects are exposed for instanceof, never constructible.
void expectClassNotConstructible(const ev::Persistent& obj, const char* cls) {
    ev::Persistent c(ev::getProperty(obj.get(), cls));
    check(ev::isFunction(c.get()), std::string(cls) + " class object exists");
    if (!ev::isFunction(c.get())) return;
    ev::CallResult r = ev::construct(c.get(), {});
    check(r.thrown, std::string("new ") + cls + "() throws");
}

// A method called on a receiver of the wrong class must throw, not cast.
void expectBrandChecked(const ev::Persistent& obj, const char* cls, const char* method) {
    ev::Persistent c(ev::getProperty(obj.get(), cls));
    if (!ev::isFunction(c.get())) return;
    ev::Persistent proto(ev::getProperty(c.get(), "prototype"));
    ev::Persistent m(ev::getProperty(proto.get(), method));
    check(ev::isFunction(m.get()), std::string(cls) + ".prototype." + method + " is a function");
    if (!ev::isFunction(m.get())) return;
    ev::Persistent plain(ev::createObject());
    ev::CallResult r = ev::call(m.get(), plain.get(), {});
    check(r.thrown, std::string(cls) + ".prototype." + method + " on a plain object throws");
}

}  // namespace

int main() {
    std::cout << "Installing brosoundml API into the Bronze realm..." << std::endl;
    brosoundml::api::installSoundML();
    std::cout << "Installed." << std::endl;

    ev::GlobalValue g = ev::globalValue("bro");
    check(g.found && ev::isObject(g.value), "global bro exists");
    if (!g.found) return 1;
    ev::Persistent bro(g.value);

    // Every namespace mounts, whatever weights are around.
    for (const char* name : {"stt", "tts", "diar", "rave", "wake", "kws", "sense", "gesture", "listen"})
        ns(bro, name);

    // ── bro.stt ────────────────────────────────────────────────────────────
    {
        ev::Persistent stt = ns(bro, "stt");
        expectInitIdempotent(stt, "bro.stt");
        expectThrows(stt, "loadWhisper", noArgs, "TypeError", "path");
        expectThrows(stt, "loadParakeet", noArgs, "TypeError", "path");
        expectThrows(stt, "loadQwenAsr", noArgs, "TypeError", "path");
        expectThrows(stt, "transcribe", noArgs, "TypeError", "model and audio");
        for (const char* cls : {"WhisperModel", "WhisperTokenizer", "WhisperSession", "ParakeetModel", "QwenAsrModel"})
            expectClassNotConstructible(stt, cls);
        expectBrandChecked(stt, "WhisperModel", "transcribe");
    }

    // ── bro.tts ────────────────────────────────────────────────────────────
    {
        ev::Persistent tts = ns(bro, "tts");
        expectInitIdempotent(tts, "bro.tts");
        expectThrows(tts, "loadKokoro", noArgs, "TypeError", "path");
        expectThrows(tts, "loadQwen", noArgs, "TypeError", "path");
        expectThrows(tts, "loadOmniVoice", noArgs, "TypeError", "path");
        expectThrows(tts, "loadSupertonic", noArgs, "TypeError", "path");
        expectThrows(tts, "loadHiggsCodec", noArgs, "TypeError", "path");
        expectThrows(tts, "synthesize", noArgs, "TypeError", "");
        for (const char* cls : {"KokoroModel", "Voice", "KokoroSession", "QwenTtsModel", "OmniVoice",
                                "SupertonicModel", "SpeakerEncoder", "HiggsCodec"})
            expectClassNotConstructible(tts, cls);
        expectBrandChecked(tts, "KokoroModel", "synthesize");
        expectBrandChecked(tts, "HiggsCodec", "decode");
    }

    // ── bro.diar ───────────────────────────────────────────────────────────
    {
        ev::Persistent diar = ns(bro, "diar");
        expectInitIdempotent(diar, "bro.diar");
        expectThrows(diar, "loadSortformer", noArgs, "TypeError", "path");
        expectThrows(diar, "loadClusterDiarizer",
                     [] { return std::vector<ev::Value>{ev::fromUtf8("x")}; },
                     "TypeError", "two paths");
        expectThrows(diar, "diarize", noArgs, "TypeError", "model and audio");
        expectThrows(diar, "clusterDiarize", noArgs, "TypeError", "model and audio");
        for (const char* cls : {"Sortformer", "SortformerSession", "ClusterDiarizer"})
            expectClassNotConstructible(diar, cls);
    }

    // ── bro.rave ───────────────────────────────────────────────────────────
    {
        ev::Persistent rave = ns(bro, "rave");
        expectInitIdempotent(rave, "bro.rave");
        expectThrows(rave, "loadRave", noArgs, "TypeError", "path");
        expectThrows(rave, "loadRave",
                     [] { return std::vector<ev::Value>{ev::fromDouble(42)}; },
                     "TypeError", "path");
        expectClassNotConstructible(rave, "Rave");
    }

    // ── bro.listen ─────────────────────────────────────────────────────────
    {
        ev::Persistent listen = ns(bro, "listen");
        ev::Persistent supFn(ev::getProperty(listen.get(), "supported"));
        check(ev::isFunction(supFn.get()), "bro.listen.supported is a function");
        if (ev::isFunction(supFn.get())) {
            ev::CallResult r = ev::call(supFn.get(), listen.get(), {});
            check(!r.thrown && ev::isBool(r.value), "bro.listen.supported() returns bool");
#if defined(__linux__)
            check(ev::toBool(r.value) == true, "bro.listen.supported() is true on Linux");
#endif
        }
        expectThrows(listen, "open",
                     [] { return std::vector<ev::Value>{ev::fromDouble(1234)}; },
                     "TypeError", "source must be a string or an object");
    }

    brosoundml::api::shutdownSoundML();

    if (g_failures) {
        std::cerr << g_failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "All brosoundml_api standalone tests passed successfully!" << std::endl;
    return 0;
}
