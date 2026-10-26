// brosoundml's natives under the host's access gate (embed/embed_access.h;
// api.h, "Access"): each loader asks the gate about the paths it was given
// before anything is opened or queued, a refusal is an Error carrying the
// gate's sentence, code EACCES and the path, and with the gate allowing it the
// same call gets past the gate to its ordinary not-found error. No weights:
// every path under the refused directory is refused whether it exists or not.
// The writers (OmniVoice.savePrompt, YuE2Stream.saveSong) need a loaded model
// and are not reached here. Run from test_soundml_api.cpp's main.

#include "embed/embed.h"
#include "embed/embed_access.h"
#include "eval/eval.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ev = bronze::embed;

#define GCHECK(cond, msg)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "CHECK failed: " #cond " (" << __FILE__ << ":"        \
                      << __LINE__ << "): " << (msg) << std::endl;              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

std::string utf8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

// The path as the gate is handed it.
std::string gatePath(const fs::path& p) {
    return utf8(fs::absolute(p).lexically_normal());
}

std::string js(const fs::path& p) {
    std::string out = "'";
    for (char c : utf8(p)) {
        if (c == '\\' || c == '\'') out += '\\';
        out += c;
    }
    return out + "'";
}

std::string evalString(const std::string& src) {
    auto r = bronze::eval::evalScript(src);
    if (r.thrown) return "<thrown>";
    return ev::isString(r.value) ? ev::toUtf8(r.value) : "<not a string>";
}

// Runs `stmt`; "ok", the refusal as "denied|message|code|path", or
// "error|<code>|<message>" for any other throw.
std::string syncJs(const std::string& stmt) {
    return evalString(
        "(function() { try { " + stmt + "; return 'ok'; } catch (e) {\n"
        "  if (e && e.code === 'EACCES') return 'denied|' + e.message + '|' + e.code + '|' + e.path;\n"
        "  return 'error|' + (e && e.code) + '|' + (e && e.message); } })()");
}

std::string refusal(const fs::path& p) {
    const std::string path = gatePath(p);
    return "denied|no files access: " + path + "|EACCES|" + path;
}

bool isOrdinaryError(const std::string& r) {
    return r.rfind("error|", 0) == 0 && r.find("EACCES") == std::string::npos;
}

struct TestGate {
    std::mutex mu;
    std::vector<std::string> refused;
    std::atomic<int> calls{0};
    std::string ask(const std::string& path) {
        ++calls;
        std::lock_guard lock(mu);
        for (const auto& p : refused) {
            if (path.compare(0, p.size(), p) == 0) return "no files access: " + path;
        }
        return "";
    }
};

} // namespace

void runAccessGateTests() {
    std::cout << "Testing the access gate..." << std::endl;

    const fs::path root = fs::temp_directory_path() / "brosoundml_api_gate";
    const fs::path ok = root / "ok";
    const fs::path hidden = root / "hidden";
    fs::create_directories(ok);
    fs::create_directories(hidden);
    // A file that exists under the refused directory: refused all the same.
    std::ofstream(hidden / "tokenizer.json") << "not a tokenizer";

    auto gate = std::make_shared<TestGate>();
    gate->refused.push_back(gatePath(hidden));
    ev::setAccessGate([gate](const ev::AccessRequest& r) -> std::string {
        if (r.kind != ev::AccessKind::Read && r.kind != ev::AccessKind::Write) return "";
        return gate->ask(std::string(r.subject));
    });

    std::string r = syncJs("globalThis._gateAgent = bro.createVoiceAgent()");
    GCHECK(r == "ok", "createVoiceAgent -> " + r);

    struct Case { std::string call; fs::path refusedPath; bool succeedsWhenAllowed; };
    std::vector<Case> cases;
    for (const char* loader : {"stt.loadWhisper", "stt.loadParakeet", "stt.loadQwenAsr", "stt.loadQwenAsrStream",
                               "tts.loadKokoro", "tts.loadQwen", "tts.loadOmniVoice", "tts.loadSupertonic",
                               "tts.loadSpeakerEncoder", "tts.loadHiggsCodec", "diar.loadSortformer",
                               "rave.loadRave", "music.loadYuE2", "music.loadVae", "music.loadSeparator",
                               "ear.loadClap"}) {
        cases.push_back({std::string("bro.") + loader + "(" + js(hidden / "model") + ")", hidden / "model", false});
    }
    // A loader given opts.onReady is refused at the call: nothing is queued.
    cases.push_back({"bro.stt.loadWhisper(" + js(hidden / "model") + ", { onReady() {}, onError() {} })",
                     hidden / "model", true});
    cases.push_back({"bro.diar.loadClusterDiarizer(" + js(ok / "sf") + ", " + js(hidden / "enc") + ")",
                     hidden / "enc", false});
    cases.push_back({"bro.stt.loadTokenizer({ vocabPath: " + js(ok / "vocab.json") + ", mergesPath: " +
                         js(hidden / "merges.txt") + " })",
                     hidden / "merges.txt", false});
    cases.push_back({"bro.stt.loadParakeetTokenizer(" + js(hidden / "tokenizer.json") + ")",
                     hidden / "tokenizer.json", false});
    cases.push_back({"bro.kws.load({ weights: " + js(hidden / "phoneme.safetensors") + " })",
                     hidden / "phoneme.safetensors", false});
    cases.push_back({"bro.wake.load({ weights: " + js(hidden / "wake.safetensors") + " })",
                     hidden / "wake.safetensors", false});
    cases.push_back({"new BcResnet2d(" + js(hidden / "vad.safetensors") + ")", hidden / "vad.safetensors", false});
    cases.push_back({"new BcResnet2d({ weights: " + js(hidden / "vad.safetensors") + " })",
                     hidden / "vad.safetensors", false});
    cases.push_back({"globalThis._gateAgent.setVad(" + js(hidden / "vad.safetensors") + ")",
                     hidden / "vad.safetensors", false});
    cases.push_back({"bro.tts.setAssetRoot(" + js(hidden / "assets") + ")", hidden / "assets", true});
    cases.push_back({"bro.tts.setAssets({ lexicon: " + js(ok / "lex.bin") + ", pos: " + js(hidden / "pos.bin") + " })",
                     hidden / "pos.bin", true});

    // Refused: the gate's sentence, EACCES and the absolute path, before the
    // native looks at the disk.
    for (const auto& c : cases) {
        r = syncJs(c.call);
        GCHECK(r == refusal(c.refusedPath), c.call + " -> " + r);
    }
    GCHECK(fs::exists(hidden / "tokenizer.json"), "the refused file went away");

    // Allowed: the same calls get past the gate, to their own not-found
    // errors (setAssets / setAssetRoot only record the paths; an onReady load
    // is queued and fails later).
    {
        std::lock_guard lock(gate->mu);
        gate->refused.clear();
    }
    const int callsBefore = gate->calls.load();
    for (const auto& c : cases) {
        r = syncJs(c.call);
        GCHECK(c.succeedsWhenAllowed ? r == "ok" : isOrdinaryError(r), "allowed " + c.call + " -> " + r);
    }
    GCHECK(gate->calls.load() > callsBefore, "the allowing gate was not asked");

    ev::setAccessGate(nullptr);
    bronze::eval::evalScript("bro.tts.setAssetRoot(''); delete globalThis._gateAgent;");
    std::error_code ec;
    fs::remove_all(root, ec);
    std::cout << "  access gate [PASS]" << std::endl;
}
