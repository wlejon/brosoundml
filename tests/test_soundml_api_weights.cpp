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

    // ── CLAP: bro.ear.loadClap from the default location, score / embed ────
    const fs::path clapDir = weights / "clap";
    const fs::path humWav = weights / "mhmm.wav";
    if (fs::exists(clapDir / "model.safetensors") && fs::exists(humWav)) {
        runEval("globalThis.REPO_DIR = " + jsString(fs::path(BROSOUNDML_REPO_DIR).generic_string()) +
                "; globalThis.HUM_WAV = " + jsString(humWav.generic_string()) + ";", "clap paths");
        // No dir: the default is <asset root>/weights/clap.
        const bool launched = runEval(R"JS(
            bro.tts.setAssetRoot(REPO_DIR);
            globalThis.cLoaded = false;
            bro.ear.loadClap({ onReady: (m) => { globalThis.clap = m; globalThis.cLoaded = true; },
                               onError: (e) => { globalThis.cError = String(e); globalThis.cLoaded = true; } });
        )JS", "CLAP background load (default dir)");
        if (launched && pumpUntil("cLoaded", 600)) {
            runEval(R"JS(
                if (globalThis.cError) throw new Error('load failed: ' + cError);
                if (!(clap instanceof bro.ear.ClapModel)) throw new Error('onReady did not deliver a ClapModel');
                if (!clap.loaded || clap.sampleRate !== 48000 || clap.embeddingSize !== 512 || clap.windowSeconds !== 10)
                    throw new Error('config ' + clap.sampleRate + '/' + clap.embeddingSize + '/' + clap.windowSeconds);
                if (Math.abs(clap.logitScale - 38.6647) > 1e-3) throw new Error('logitScale ' + clap.logitScale);
                const prompts = ['a dog barking', 'a person humming', 'a whistle', 'xylophone'];
                const norm = (v) => { let s = 0; for (const x of v) s += x * x; return Math.sqrt(s); };
                const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

                // A file path clip; the reference ranks the hum at 0.958 'a person humming'.
                const r = clap.score(HUM_WAV, prompts);
                if (!(r.scores instanceof Float32Array) || r.scores.length !== 4) throw new Error('scores');
                if (r.similarities.length !== 4 || r.logits.length !== 4) throw new Error('similarities/logits');
                if (!(r.embedding instanceof Float32Array) || r.embedding.length !== 512) throw new Error('embedding');
                if (Math.abs(norm(r.embedding) - 1) > 1e-4) throw new Error('embedding not unit: ' + norm(r.embedding));
                let sum = 0; for (const p of r.scores) sum += p;
                if (Math.abs(sum - 1) > 1e-5) throw new Error('scores sum ' + sum);
                if (r.bestIndex !== 1 || r.best !== 'a person humming' || !(r.scores[1] > 0.9))
                    throw new Error('best ' + r.best + ' ' + r.scores[1]);
                // scores = softmax(similarities * logitScale), logits = similarities * logitScale.
                const l = Array.from(r.similarities, (s) => s * clap.logitScale);
                const mx = Math.max(...l), e = l.map((x) => Math.exp(x - mx)), z = e.reduce((a, b) => a + b);
                for (let i = 0; i < 4; i++) {
                    if (Math.abs(r.logits[i] - l[i]) > 1e-3) throw new Error('logit ' + i);
                    if (Math.abs(r.scores[i] - e[i] / z) > 1e-5) throw new Error('softmax ' + i);
                }

                // The same clip as a bare Float32Array, { samples, sampleRate } and an AudioBuffer.
                const n = 44100, pcm = new Float32Array(n);
                for (let i = 0; i < n; i++) pcm[i] = 0.2 * Math.sin(2 * Math.PI * 440 * i / 44100);
                const a1 = clap.embedAudio(pcm, { sampleRate: 44100 });
                const a2 = clap.embedAudio({ samples: pcm, sampleRate: 44100 });
                const ab = { numberOfChannels: 2, sampleRate: 44100, length: n, getChannelData: (c) => pcm };
                const a3 = clap.embedAudio(ab);
                if (maxDiff(a1, a2) !== 0 || maxDiff(a1, a3) > 1e-6) throw new Error('clip forms differ');
                const a48 = clap.embedAudio(pcm);   // read as 48 kHz: a different pitch
                if (maxDiff(a1, a48) < 1e-3) throw new Error('sampleRate ignored');

                // Text embeddings, cached prompts and scoreEmbedding.
                const t = clap.embedText('a person humming');
                if (!(t instanceof Float32Array) || t.length !== 512) throw new Error('embedText(string)');
                const ts = clap.embedText(prompts);
                if (!Array.isArray(ts) || ts.length !== 4 || maxDiff(ts[1], t) !== 0) throw new Error('embedText(array)');
                const r2 = clap.scoreEmbedding(r.embedding, [ts[0], 'a person humming', ts[2], 'xylophone']);
                if (maxDiff(r2.scores, r.scores) > 1e-6) throw new Error('scoreEmbedding disagrees with score');
                if (r2.best !== 'a person humming') throw new Error('best of a text prompt: ' + r2.best);
                const r3 = clap.score(HUM_WAV, [ts[0], ts[1]]);
                if (r3.bestIndex !== 1 || r3.best !== null) throw new Error('cached-only best ' + r3.best);

                // Long clips: the window mean by default, one crop on request.
                const long = new Float32Array(48000 * 13);
                for (let i = 0; i < long.length; i++) long[i] = 0.2 * Math.sin(2 * Math.PI * (i < 48000 * 6 ? 440 : 880) * i / 48000);
                const mean = clap.embedAudio(long), head = clap.embedAudio(long, { long: 'crop', cropAt: 0 });
                const tail = clap.embedAudio(long, { long: 'crop', cropAt: 99 });
                if (maxDiff(mean, head) < 1e-3 || maxDiff(head, tail) < 1e-3) throw new Error('long modes coincide');

                const expectType = (f, what) => {
                    let ok = false;
                    try { f(); } catch (e) { ok = e instanceof TypeError; }
                    if (!ok) throw new Error(what + ' did not throw a TypeError');
                };
                expectType(() => clap.score(pcm), 'score without prompts');
                expectType(() => clap.score(pcm, []), 'score with no prompts');
                expectType(() => clap.score(pcm, [42]), 'a numeric prompt');
                expectType(() => clap.score(pcm, [new Float32Array(3)]), 'a short cached prompt');
                expectType(() => clap.score(new Float32Array(0), 'x'), 'an empty clip');
                expectType(() => clap.embedAudio(pcm, { long: 'middle' }), 'long: middle');
                expectType(() => clap.embedAudio(42), 'a numeric clip');
                expectType(() => clap.embedText([new Float32Array(512)]), 'embedText of an embedding');
                expectType(() => clap.scoreEmbedding(new Float32Array(4), 'x'), 'a short audio embedding');

                // Async score + busy guard.
                globalThis.sDone = false;
                clap.score(HUM_WAV, prompts, { onDone: (res, info) => { globalThis.sRes = res; globalThis.sInfo = info; globalThis.sDone = true; } });
                let threw = false;
                try { clap.embedText('x'); } catch (e) { threw = true; }
                if (!threw) throw new Error('a sync call while busy did not throw');
                globalThis.cHum = r;
            )JS", "CLAP sync surface + async score launch");
            if (pumpUntil("sDone", 120)) {
                runEval(R"JS(
                    if (sInfo.error || sInfo.cancelled) throw new Error('info ' + JSON.stringify(sInfo));
                    let d = 0;
                    for (let i = 0; i < 4; i++) d = Math.max(d, Math.abs(sRes.scores[i] - cHum.scores[i]));
                    if (d !== 0) throw new Error('async score differs from sync by ' + d);
                    globalThis.eDone = false;
                    clap.embedText(['a whistle', 'rain'], { onDone: (res) => {
                        globalThis.eText = res;
                        clap.embedAudio(HUM_WAV, { onDone: (emb, info) => {
                            globalThis.eAudio = emb; globalThis.eInfo = info; globalThis.eDone = true;
                        } });
                    } });
                )JS", "CLAP async score result");
                if (pumpUntil("eDone", 120)) {
                    runEval(R"JS(
                        if (!Array.isArray(eText) || eText.length !== 2 || eText[0].length !== 512) throw new Error('async embedText');
                        if (eInfo.error) throw new Error(eInfo.error);
                        for (let i = 0; i < 512; i++) if (eAudio[i] !== cHum.embedding[i]) throw new Error('async embedAudio differs');
                        clap.dispose();
                        if (clap.loaded) throw new Error('loaded after dispose');
                        let threw = false;
                        try { clap.embedText('x'); } catch (e) { threw = true; }
                        if (!threw) throw new Error('a call after dispose did not throw');
                    )JS", "CLAP chained async embedText + embedAudio, dispose");
                }
            }
        }
    } else {
        std::cout << "  SKIP CLAP (no " << clapDir.string() << " / " << humWav.string() << ")" << std::endl;
    }

    brosoundml::api::shutdownSoundML();
    if (g_failures) {
        std::cerr << g_failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "All brosoundml_api weights tests passed." << std::endl;
    return 0;
}
