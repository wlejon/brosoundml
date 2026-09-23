// Weights-gated Bronze JS API test for brosoundml_api: real model loads and
// the async paths (onDone / onReady through the AsyncHandle registry) driven
// from JavaScript, pumped by tickSoundML() the way the engine's frame loop
// pumps them. Each section skips when its weights are absent under
// <repo>/weights; with none present the test passes having checked nothing
// but the install.
//
// Run it under BRONZE_GC_STRESS=1 BRONZE_GC_POISON=1 as well (the _gcstress
// ctest variant): the async paths are where a Value held across an
// allocation hides.
#include "../src/api/api.h"
#include "embed/embed.h"
#include "eval/eval.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace ev = bronze::embed;
namespace fs = std::filesystem;

namespace {

int g_failures = 0;

std::string jsString(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\\' || c == '\'') out += '\\';
        out += c;
    }
    return out + "'";
}

// Run `code`; a throw is a failure named `desc`.
bool runEval(const std::string& code, const std::string& desc) {
    ev::CallResult r = bronze::eval::evalScript(code);
    if (!r.thrown) {
        std::cout << "  PASS [" << desc << "]" << std::endl;
        return true;
    }
    std::string msg = "unknown error";
    if (ev::isObject(r.value)) {
        ev::Persistent err(r.value);
        ev::Value m = ev::getProperty(err.get(), "message");
        msg = ev::isString(m) ? ev::toUtf8(m) : ev::toUtf8(err.get());
    } else if (ev::isString(r.value)) {
        msg = ev::toUtf8(r.value);
    }
    std::cerr << "FAIL [" << desc << "]: " << msg << std::endl;
    ++g_failures;
    return false;
}

// Pump the async registry until globalThis[flag] is truthy (or time out).
bool pumpUntil(const char* flag, int timeoutSec) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
    while (std::chrono::steady_clock::now() < deadline) {
        brosoundml::api::tickSoundML();
        ev::GlobalValue gt = ev::globalValue("globalThis");
        if (gt.found && ev::isObject(gt.value)) {
            ev::Value v = ev::getProperty(gt.value, flag);
            if (ev::isBool(v) && ev::toBool(v)) return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::cerr << "FAIL: timed out waiting for " << flag << std::endl;
    ++g_failures;
    return false;
}

}  // namespace

int main() {
    const fs::path weights = fs::path(BROSOUNDML_REPO_DIR) / "weights";
    std::cout << "=== brosoundml_api weights test (" << weights.string() << ") ===" << std::endl;
    brosoundml::api::installSoundML();

    // ── HiggsCodec: encode ▸ decode round trip ─────────────────────────────
    const fs::path codecDir = weights / "omnivoice" / "audio_tokenizer";
    if (fs::exists(codecDir / "config.json")) {
        runEval("globalThis.CODEC_DIR = " + jsString(codecDir.generic_string()) + ";", "codec path");
        runEval(R"JS(
            const codec = bro.tts.loadHiggsCodec(CODEC_DIR);
            if (!(codec instanceof bro.tts.HiggsCodec)) throw new Error('not a HiggsCodec');
            if (!codec.loaded || !codec.hasEncoder) throw new Error('codec not loaded with encoder');
            if (codec.sampleRate !== 24000 || codec.hopLength !== 960) throw new Error('config: ' + codec.sampleRate + '/' + codec.hopLength);
            const n = 24000;
            const pcm = new Float32Array(n);
            for (let i = 0; i < n; i++) pcm[i] = 0.3 * Math.sin(2 * Math.PI * 220 * i / 24000);
            const enc = codec.encode(pcm);
            if (enc.numFrames !== 25) throw new Error('numFrames ' + enc.numFrames);
            if (enc.codes.length !== enc.numQuantizers * enc.numFrames) throw new Error('codes length ' + enc.codes.length);
            for (let i = 0; i < enc.codes.length; i++)
                if (enc.codes[i] < 0 || enc.codes[i] >= codec.codebookSize) throw new Error('code out of range');
            const dec = codec.decode(enc.codes, enc.numFrames);
            if (dec.sampleRate !== 24000 || dec.samples.length !== enc.numFrames * 960)
                throw new Error('decode shape ' + dec.samples.length + ' @ ' + dec.sampleRate);
            let energy = 0;
            for (let i = 0; i < dec.samples.length; i++) energy += dec.samples[i] * dec.samples[i];
            if (!(energy > 1)) throw new Error('decoded audio is silent: ' + energy);
            // Fewer RVQ levels: the first 4 codebooks of the same frames.
            const coarse = codec.decode(enc.codes.subarray(0, 4 * enc.numFrames), { numQuantizers: 4 });
            if (coarse.samples.length !== enc.numFrames * 960) throw new Error('coarse decode length');
            let threw = false;
            try { codec.decode(new Int32Array(7)); } catch (e) { threw = e instanceof TypeError; }
            if (!threw) throw new Error('bad codes length did not throw a TypeError');
            threw = false;
            try { bro.tts.HiggsCodec.prototype.encode.call({}, pcm); } catch (e) { threw = true; }
            if (!threw) throw new Error('encode on a plain object did not throw');
            codec.dispose();
            if (codec.loaded) throw new Error('loaded after dispose');
        )JS", "HiggsCodec encode/decode round trip");
    } else {
        std::cout << "  SKIP HiggsCodec (no " << codecDir.string() << ")" << std::endl;
    }

    // ── Whisper: async transcribe through onDone, streamed tokens ──────────
    const fs::path whisperDir = weights / "whisper";
    if (fs::exists(whisperDir / "model.safetensors") && fs::exists(whisperDir / "vocab.json")) {
        runEval("globalThis.WHISPER_DIR = " + jsString(whisperDir.generic_string()) + ";", "whisper path");
        const bool launched = runEval(R"JS(
            globalThis.wDone = false;
            globalThis.wLoaded = false;
            bro.stt.loadWhisper(WHISPER_DIR, { onReady: (m) => { globalThis.whisper = m; globalThis.wLoaded = true; },
                                               onError: (e) => { globalThis.wError = String(e); globalThis.wLoaded = true; } });
        )JS", "Whisper background load");
        if (launched && pumpUntil("wLoaded", 300)) {
            runEval(R"JS(
                if (globalThis.wError) throw new Error('load failed: ' + wError);
                if (!(whisper instanceof bro.stt.WhisperModel)) throw new Error('onReady did not deliver a WhisperModel');
                const tok = bro.stt.loadTokenizer({ vocabPath: WHISPER_DIR + '/vocab.json',
                                                    mergesPath: WHISPER_DIR + '/merges.txt' });
                const prompt = tok.buildPrompt('en', 'transcribe', false);
                const pcm = new Float32Array(16000 * 2);
                for (let i = 0; i < pcm.length; i++) pcm[i] = 0.05 * Math.sin(2 * Math.PI * 300 * i / 16000);
                globalThis.wTokens = 0;
                globalThis.handle = bro.stt.transcribe(whisper, pcm, prompt, {
                    maxNewTokens: 16,
                    onToken: () => { globalThis.wTokens++; },
                    onDone: (ids, info) => {
                        globalThis.wIds = ids; globalThis.wInfo = info; globalThis.wDone = true;
                    },
                });
                let threw = false;
                try { bro.stt.transcribe(whisper, pcm, prompt, { onDone: () => {} }); } catch (e) { threw = true; }
                if (!threw) throw new Error('a second transcribe while busy did not throw');
            )JS", "Whisper async transcribe launch + busy guard");
            if (pumpUntil("wDone", 300)) {
                runEval(R"JS(
                    if (!(wIds instanceof Int32Array) || wIds.length === 0) throw new Error('onDone ids: ' + wIds);
                    if (wInfo.cancelled !== false) throw new Error('info.cancelled ' + wInfo.cancelled);
                    if (wInfo.error) throw new Error('info.error ' + wInfo.error);
                    if (!handle.finished) throw new Error('handle.finished false after onDone');
                )JS", "Whisper onDone result");
            }
        }
    } else {
        std::cout << "  SKIP Whisper (no " << whisperDir.string() << ")" << std::endl;
    }

    // ── Kokoro: voice + async synthesize ───────────────────────────────────
    const fs::path kokoroDir = weights / "kokoro";
    fs::path voice = kokoroDir / "voices" / "af_heart.bin";
    if (!fs::exists(voice) && fs::is_directory(kokoroDir / "voices")) {
        for (const auto& e : fs::directory_iterator(kokoroDir / "voices"))
            if (e.path().extension() == ".bin") { voice = e.path(); break; }
    }
    if (fs::exists(kokoroDir / "model.safetensors") && fs::exists(voice)) {
        runEval("globalThis.KOKORO_DIR = " + jsString(kokoroDir.generic_string()) +
                "; globalThis.KOKORO_VOICE = " + jsString(voice.generic_string()) + ";", "kokoro paths");
        const bool launched = runEval(R"JS(
            const k = bro.tts.loadKokoro(KOKORO_DIR);
            const v = k.loadVoice(KOKORO_VOICE);
            if (!(v instanceof bro.tts.Voice)) throw new Error('loadVoice did not return a Voice');
            globalThis.kDone = false;
            bro.tts.synthesize(k, new Int32Array([0, 50, 83, 54, 156, 57, 135, 0]), v, {
                onDone: (res, info) => { globalThis.kRes = res; globalThis.kInfo = info; globalThis.kDone = true; },
            });
            let threw = false;
            try { bro.tts.synthesize(k, new Int32Array([0, 50, 0]), v, { onDone: () => {} }); } catch (e) { threw = true; }
            if (!threw) throw new Error('a second synthesize while busy did not throw');
        )JS", "Kokoro async synthesize launch + busy guard");
        if (launched && pumpUntil("kDone", 300)) {
            runEval(R"JS(
                if (kInfo.error) throw new Error('info.error ' + kInfo.error);
                if (!kRes || !(kRes.samples instanceof Float32Array) || kRes.samples.length === 0)
                    throw new Error('no samples');
                if (kRes.sampleRate !== 24000) throw new Error('sampleRate ' + kRes.sampleRate);
            )JS", "Kokoro onDone result");
        }
    } else {
        std::cout << "  SKIP Kokoro (no " << kokoroDir.string() << " model + voice)" << std::endl;
    }

    brosoundml::api::shutdownSoundML();
    if (g_failures) {
        std::cerr << g_failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "All brosoundml_api weights tests passed." << std::endl;
    return 0;
}
