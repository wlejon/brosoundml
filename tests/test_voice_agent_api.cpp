#include "../src/api/api.h"
#include "embed/embed.h"
#include "eval/eval.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace ev = bronze::embed;

static int g_failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::cerr << "FAIL: " << (msg) << std::endl;              \
            ++g_failures;                                             \
        } else {                                                      \
            std::cout << "  PASS: " << (msg) << std::endl;            \
        }                                                             \
    } while (0)

static void runEval(std::string_view code, const std::string& desc) {
    ev::CallResult r = bronze::eval::evalScript(code);
    if (r.thrown) {
        std::string errStr = "unknown error";
        if (ev::isObject(r.value)) {
            ev::Value msgVal = ev::getProperty(r.value, "message");
            if (ev::isString(msgVal)) errStr = ev::toUtf8(msgVal);
            else errStr = ev::toUtf8(r.value);
        } else if (ev::isString(r.value)) {
            errStr = ev::toUtf8(r.value);
        }
        std::cerr << "FAIL [" << desc << "]: " << errStr << std::endl;
        ++g_failures;
    } else {
        std::cout << "  PASS [" << desc << "]" << std::endl;
    }
}

int main() {
    std::cout << "=== Test VoiceAgent Bronze API ===" << std::endl;

    brosoundml::api::installSoundML();

    ev::GlobalValue g = ev::globalValue("bro");
    CHECK(g.found && ev::isObject(g.value), "global bro exists");
    if (!g.found) return 1;

    // ── Test 1: Full pipeline test (speech -> STT -> LLM -> TTS -> output -> idle) ──
    std::cout << "\n--- Test 1: Full Pipeline (Speech -> STT -> LLM -> TTS -> Audio Output) ---" << std::endl;
    runEval(R"JS(
        globalThis.testEvents = {
            speechStarted: false,
            speechEnded: false,
            speechEndSamples: 0,
            transcript: "",
            responseText: "",
            ttsChunks: 0,
            stateTransitions: [],
            bargeInFired: false
        };

        // Create agent with new bro.soundml.VoiceAgent
        const agent = new bro.soundml.VoiceAgent({
            sampleRate: 16000,
            vadEnergyThreshold: 0.01,
            minSpeechFrames: 2,
            silenceHangoverFrames: 4
        });
        globalThis.agent = agent;

        // Verify initial state
        if (agent.state !== 'idle') throw new Error("Expected initial state 'idle', got: " + agent.state);
        if (agent.isIdle !== true) throw new Error("Expected isIdle === true");
        if (agent.isListening !== false) throw new Error("Expected isListening === false");
        if (agent.isThinking !== false) throw new Error("Expected isThinking === false");
        if (agent.isSpeaking !== false) throw new Error("Expected isSpeaking === false");

        // Verify config
        const cfg = agent.config;
        if (cfg.sampleRate !== 16000) throw new Error("Config sampleRate mismatch: " + cfg.sampleRate);
        if (cfg.vadEnergyThreshold < 0.009 || cfg.vadEnergyThreshold > 0.011) {
            throw new Error("Config vadEnergyThreshold mismatch: " + cfg.vadEnergyThreshold);
        }

        // Register callbacks
        agent.onStateChanged((oldState, newState) => {
            testEvents.stateTransitions.push(oldState + '->' + newState);
        });

        agent.onSpeechStart(() => {
            testEvents.speechStarted = true;
        });

        agent.onSpeechEnd((utterance) => {
            testEvents.speechEnded = true;
            testEvents.speechEndSamples = utterance.length;
        });

        agent.onTranscript((t) => {
            testEvents.transcript = t;
        });

        agent.onResponseText((r) => {
            testEvents.responseText = r;
        });

        agent.onAudioOutput((chunk) => {
            testEvents.ttsChunks++;
            if (chunk.length !== 2400) {
                throw new Error("TTS chunk length expected 2400, got: " + chunk.length);
            }
        });

        // Set custom handlers
        agent.setSttHandler((audio) => {
            if (!audio || audio.length === 0) throw new Error("STT received empty audio");
            return "turn on the living room lights";
        });

        agent.setTextHandler((query) => {
            if (query !== "turn on the living room lights") {
                throw new Error("TextHandler unexpected query: " + query);
            }
            return "Living room lights turned on.";
        });

        agent.setTtsHandler((text, emitChunk) => {
            if (text !== "Living room lights turned on.") {
                throw new Error("TtsHandler unexpected text: " + text);
            }
            // Emit 2 chunks
            emitChunk(new Float32Array(2400).fill(0.1));
            emitChunk(new Float32Array(2400).fill(0.2));
        });

        function makeSine(n, freq, amp, rate) {
            const arr = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                arr[i] = amp * Math.sin(2.0 * Math.PI * freq * i / rate);
            }
            return arr;
        }

        // Feed speech chunks (amp 0.5 > 0.01 threshold)
        const speechChunk = makeSine(480, 440, 0.5, 16000);
        for (let i = 0; i < 5; i++) {
            agent.feed(speechChunk);
        }

        if (!testEvents.speechStarted) throw new Error("Speech start callback did not fire");
        if (!agent.isListening) throw new Error("Agent should be in listening state");
        if (agent.state !== 'listening') throw new Error("Agent.state should be 'listening'");

        // Feed silence chunks to end utterance
        const silenceChunk = makeSine(160, 0, 0.0, 16000);
        for (let i = 0; i < 10; i++) {
            agent.feed(silenceChunk);
        }

        // Verify end of speech, transcript, response, TTS, and return to idle
        if (!testEvents.speechEnded) throw new Error("Speech end callback did not fire");
        if (testEvents.speechEndSamples <= 0) throw new Error("Utterance length was 0");
        if (testEvents.transcript !== "turn on the living room lights") {
            throw new Error("Transcript mismatch: " + testEvents.transcript);
        }
        if (testEvents.responseText !== "Living room lights turned on.") {
            throw new Error("Response text mismatch: " + testEvents.responseText);
        }
        if (testEvents.ttsChunks !== 2) {
            throw new Error("Expected 2 TTS chunks, got: " + testEvents.ttsChunks);
        }
        if (!agent.isIdle) throw new Error("Agent should have returned to idle");
        if (agent.state !== 'idle') throw new Error("agent.state should be 'idle'");

        // Verify state transitions sequence: idle->listening, listening->thinking, thinking->speaking, speaking->idle
        const transitions = testEvents.stateTransitions.join(', ');
        if (!transitions.includes('idle->listening')) throw new Error("Missing idle->listening transition: " + transitions);
        if (!transitions.includes('listening->thinking')) throw new Error("Missing listening->thinking transition: " + transitions);
        if (!transitions.includes('thinking->speaking')) throw new Error("Missing thinking->speaking transition: " + transitions);
        if (!transitions.includes('speaking->idle')) throw new Error("Missing speaking->idle transition: " + transitions);
    )JS", "Pipeline speech -> listening -> thinking -> speaking -> idle");

    // ── Test 2: Convenience constructor and direct speak/interrupt ──
    std::cout << "\n--- Test 2: Convenience Factory and speak / interrupt / reset ---" << std::endl;
    runEval(R"JS(
        // Convenience factory bro.createVoiceAgent(config)
        const agent2 = bro.createVoiceAgent({
            sampleRate: 16000
        });

        if (!agent2 || !agent2.isIdle) throw new Error("agent2 creation failed");

        let outputReceived = false;
        agent2.onAudioOutput((chunk) => {
            outputReceived = true;
        });

        // speak triggers TTS and transitions speaking -> idle
        agent2.speak("Hello from speak");
        if (!outputReceived) throw new Error("speak did not trigger audio output");
        if (!agent2.isIdle) throw new Error("agent2 did not return to idle after speak");

        // Interrupt returns to idle
        agent2.interrupt();
        if (!agent2.isIdle) throw new Error("agent2 not idle after interrupt");

        // Reset resets buffers
        agent2.reset();
        if (!agent2.isIdle) throw new Error("agent2 not idle after reset");
        if (agent2.utteranceFrameCount !== 0) throw new Error("utteranceFrameCount not 0 after reset");
    )JS", "Convenience factory and speak/interrupt/reset");

    // ── Test 3: Barge-in interruption during speech ──
    std::cout << "\n--- Test 3: Barge-in Detection ---" << std::endl;
    runEval(R"JS(
        const agent3 = new bro.soundml.VoiceAgent({
            sampleRate: 16000,
            vadEnergyThreshold: 0.01,
            minSpeechFrames: 2,
            enableBargeIn: true
        });

        let bargeInFired = false;
        agent3.onBargeIn(() => {
            bargeInFired = true;
        });

        // TTS handler that feeds speech input while in speaking state
        agent3.setTtsHandler((text, emitChunk) => {
            const speechChunk = new Float32Array(480);
            for (let i = 0; i < 480; i++) {
                speechChunk[i] = 0.5 * Math.sin(2.0 * Math.PI * 440 * i / 16000);
            }
            agent3.feed(speechChunk);
            agent3.feed(speechChunk);
        });

        agent3.speak("Start speaking to trigger barge-in");
        if (!bargeInFired) throw new Error("Barge-in callback did not fire");
        if (!agent3.isListening) throw new Error("Agent should transition to listening on barge-in");
    )JS", "Barge-in detection during speaking");

    // ── Test 4: Model attachment methods and argument validations ──
    std::cout << "\n--- Test 4: Model Attachments and TypeError Validations ---" << std::endl;
    runEval(R"JS(
        const agent4 = new bro.soundml.VoiceAgent();

        // Null / undefined clears model
        agent4.setVad(null);
        agent4.setWhisper(null);
        agent4.setKokoro(null);

        // Invalid types throw TypeErrors
        let threw = false;
        try {
            agent4.setVad(12345);
        } catch (e) {
            threw = (e instanceof TypeError);
        }
        if (!threw) throw new Error("setVad with number should throw TypeError");

        threw = false;
        try {
            agent4.setWhisper({ notAWhisper: true });
        } catch (e) {
            threw = (e instanceof TypeError);
        }
        if (!threw) throw new Error("setWhisper with invalid object should throw TypeError");

        threw = false;
        try {
            agent4.setKokoro("notKokoro");
        } catch (e) {
            threw = (e instanceof TypeError);
        }
        if (!threw) throw new Error("setKokoro with string should throw TypeError");

        // Method feed with invalid input
        threw = false;
        try {
            agent4.feed("notAudio");
        } catch (e) {
            threw = (e instanceof TypeError);
        }
        if (!threw) throw new Error("feed with string should throw TypeError");

        // speak with non-string
        threw = false;
        try {
            agent4.speak(1234);
        } catch (e) {
            threw = (e instanceof TypeError);
        }
        if (!threw) throw new Error("speak with number should throw TypeError");
    )JS", "Model attachment and TypeError validations");

    brosoundml::api::shutdownSoundML();

    if (g_failures > 0) {
        std::cerr << "\n" << g_failures << " test(s) FAILED!" << std::endl;
        return 1;
    }
    std::cout << "\nAll VoiceAgent API tests passed successfully!" << std::endl;
    return 0;
}
