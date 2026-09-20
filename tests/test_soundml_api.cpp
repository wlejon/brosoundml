// Standalone Bronze JavaScript API test for brosoundml_api: the namespaces
// mount, the class objects exist and refuse `new`, init() is idempotent, and
// bad arguments raise the documented TypeErrors. No weights, no downloads —
// the same weights-free contract bro's tests/{stt,tts,diar,rave}/*_binding.js
// assert from JavaScript.
#include "../src/api/api.h"
#include "embed/embed.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace ev = bronze::embed;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++g_failures;
    std::cerr << "FAIL: " << what << std::endl;
}

ev::Value ns(ev::Value bro, const char* name) {
    ev::Value v = ev::getProperty(bro, name);
    check(ev::isObject(v), std::string("bro.") + name + " is an object");
    return v;
}

// Call obj.fn(args) and require a thrown value whose .name is `errorName` and
// whose message contains `needle` (empty = any message).
void expectThrows(ev::Value obj, const char* fn, std::span<const ev::Value> args,
                  const char* errorName, const char* needle) {
    ev::Value f = ev::getProperty(obj, fn);
    check(ev::isFunction(f), std::string(fn) + " is a function");
    if (!ev::isFunction(f)) return;
    ev::CallResult r = ev::call(f, obj, args);
    check(r.thrown, std::string(fn) + " throws");
    if (!r.thrown) return;
    const std::string name = ev::toUtf8(ev::getProperty(r.value, "name"));
    const std::string msg = ev::toUtf8(ev::getProperty(r.value, "message"));
    check(name == errorName, std::string(fn) + " throws a " + errorName + " (got " + name + ": " + msg + ")");
    if (needle && *needle)
        check(msg.find(needle) != std::string::npos,
              std::string(fn) + " message names '" + needle + "': " + msg);
    std::cout << "  " << fn << " -> " << name << ": " << msg << std::endl;
}

void expectInitIdempotent(ev::Value obj, const char* label) {
    ev::Value f = ev::getProperty(obj, "init");
    check(ev::isFunction(f), std::string(label) + ".init is a function");
    if (!ev::isFunction(f)) return;
    for (int i = 0; i < 2; ++i) {
        ev::CallResult r = ev::call(f, obj, {});
        check(!r.thrown && ev::isUndefined(r.value), std::string(label) + ".init() returns undefined");
    }
}

// The class objects are exposed for instanceof, never constructible.
void expectClassNotConstructible(ev::Value obj, const char* cls) {
    ev::Value c = ev::getProperty(obj, cls);
    check(ev::isFunction(c), std::string(cls) + " class object exists");
    if (!ev::isFunction(c)) return;
    ev::CallResult r = ev::construct(c, {});
    check(r.thrown, std::string("new ") + cls + "() throws");
}

}  // namespace

int main() {
    std::cout << "Installing brosoundml API into the Bronze realm..." << std::endl;
    brosoundml::api::installSoundML();

    ev::GlobalValue g = ev::globalValue("bro");
    check(g.found && ev::isObject(g.value), "global bro exists");
    if (!g.found) return 1;
    ev::Value bro = g.value;

    // Every namespace mounts, whatever weights are around.
    for (const char* name : {"stt", "tts", "diar", "rave", "wake", "kws", "sense", "gesture", "listen"})
        ns(bro, name);

    // ── bro.stt ────────────────────────────────────────────────────────────
    {
        ev::Value stt = ns(bro, "stt");
        expectInitIdempotent(stt, "bro.stt");
        expectThrows(stt, "loadWhisper", {}, "TypeError", "path");
        expectThrows(stt, "loadParakeet", {}, "TypeError", "path");
        expectThrows(stt, "loadQwenAsr", {}, "TypeError", "path");
        expectThrows(stt, "transcribe", {}, "TypeError", "model and audio");
        for (const char* cls : {"WhisperModel", "WhisperTokenizer", "WhisperSession", "ParakeetModel", "QwenAsrModel"})
            expectClassNotConstructible(stt, cls);
    }

    // ── bro.tts ────────────────────────────────────────────────────────────
    {
        ev::Value tts = ns(bro, "tts");
        expectInitIdempotent(tts, "bro.tts");
        expectThrows(tts, "loadKokoro", {}, "TypeError", "path");
        expectThrows(tts, "loadQwen", {}, "TypeError", "path");
        expectThrows(tts, "loadOmniVoice", {}, "TypeError", "path");
        expectThrows(tts, "loadSupertonic", {}, "TypeError", "path");
        expectThrows(tts, "synthesize", {}, "TypeError", "");
        for (const char* cls : {"KokoroModel", "Voice", "KokoroSession", "QwenTtsModel", "OmniVoice",
                                "SupertonicModel", "SpeakerEncoder"})
            expectClassNotConstructible(tts, cls);
    }

    // ── bro.diar ───────────────────────────────────────────────────────────
    {
        ev::Value diar = ns(bro, "diar");
        expectInitIdempotent(diar, "bro.diar");
        expectThrows(diar, "loadSortformer", {}, "TypeError", "path");
        ev::Value onePath[] = {ev::fromUtf8("x")};
        expectThrows(diar, "loadClusterDiarizer", onePath, "TypeError", "two paths");
        expectThrows(diar, "diarize", {}, "TypeError", "model and audio");
        expectThrows(diar, "clusterDiarize", {}, "TypeError", "model and audio");
        for (const char* cls : {"Sortformer", "SortformerSession", "ClusterDiarizer"})
            expectClassNotConstructible(diar, cls);
    }

    // ── bro.rave ───────────────────────────────────────────────────────────
    {
        ev::Value rave = ns(bro, "rave");
        expectInitIdempotent(rave, "bro.rave");
        expectThrows(rave, "loadRave", {}, "TypeError", "path");
        ev::Value notAPath[] = {ev::fromDouble(42)};
        expectThrows(rave, "loadRave", notAPath, "TypeError", "path");
        expectClassNotConstructible(rave, "Rave");
    }

    // ── bro.listen ─────────────────────────────────────────────────────────
    {
        ev::Value listen = ns(bro, "listen");
        ev::Value supFn = ev::getProperty(listen, "supported");
        check(ev::isFunction(supFn), "bro.listen.supported is a function");
        if (ev::isFunction(supFn)) {
            ev::CallResult r = ev::call(supFn, listen, {});
            check(!r.thrown && ev::isBool(r.value), "bro.listen.supported() returns bool");
#if defined(__linux__)
            check(ev::toBool(r.value) == true, "bro.listen.supported() is true on Linux");
#endif
        }
        ev::Value notSource[] = {ev::fromDouble(1234)};
        expectThrows(listen, "open", notSource, "TypeError", "source must be a string or an object");
    }

    brosoundml::api::shutdownSoundML();

    if (g_failures) {
        std::cerr << g_failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "All brosoundml_api standalone tests passed successfully!" << std::endl;
    return 0;
}
