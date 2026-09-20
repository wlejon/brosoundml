#include "brosoundml/voice_agent.h"
#include "brosoundml/bc_resnet2d.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

static std::vector<float> make_sine_chunk(int n_samples, float freq, float amp, int sample_rate = 16000) {
    std::vector<float> samples(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        samples[i] = amp * std::sin(2.0f * 3.14159265f * freq * static_cast<float>(i) / static_cast<float>(sample_rate));
    }
    return samples;
}

int main() {
    std::printf("test_voice_agent: starting tests\n");

    // ── Test 1: Initial state and configuration ──
    {
        brosoundml::VoiceAgentConfig cfg;
        cfg.sample_rate = 16000;
        cfg.min_speech_frames = 2;
        cfg.silence_hangover_frames = 5;
        brosoundml::VoiceAgent agent(cfg);

        CHECK(agent.is_idle(), "Initial state is Idle");
        CHECK(agent.state() == brosoundml::VoiceAgentState::Idle, "State enum is Idle");
        CHECK(agent.config().sample_rate == 16000, "Sample rate matches config");
    }

    // ── Test 2: Full pipeline: Speech -> STT -> Text LLM -> TTS -> Audio Output ──
    {
        brosoundml::VoiceAgentConfig cfg;
        cfg.sample_rate = 16000;
        cfg.vad_energy_threshold = 0.01f;
        cfg.min_speech_frames = 2;
        cfg.silence_hangover_frames = 4;
        brosoundml::VoiceAgent agent(cfg);

        bool speech_started = false;
        bool speech_ended = false;
        std::string transcript_received;
        std::string response_received;
        int tts_chunks_received = 0;
        int state_transitions = 0;

        agent.on_state_changed([&](brosoundml::VoiceAgentState, brosoundml::VoiceAgentState) {
            state_transitions++;
        });

        agent.on_speech_start([&] {
            speech_started = true;
        });

        agent.on_speech_end([&](const brosoundml::AudioBuffer& utt) {
            speech_ended = true;
            CHECK(!utt.samples.empty(), "Utterance audio buffer is non-empty");
        });

        agent.set_stt_handler([](const brosoundml::AudioBuffer&) -> std::string {
            return "turn on the living room lights";
        });

        agent.set_text_handler([](const std::string& query) -> std::string {
            CHECK(query == "turn on the living room lights", "Text handler received correct query");
            return "Living room lights turned on.";
        });

        agent.set_tts_handler([](const std::string& text, brosoundml::VoiceAgent::TtsChunkCallback chunk_cb) {
            CHECK(text == "Living room lights turned on.", "TTS handler received correct response text");
            // Emit 2 audio chunks
            brosoundml::AudioBuffer c1;
            c1.sample_rate = 24000;
            c1.samples.assign(2400, 0.1f);
            chunk_cb(c1);

            brosoundml::AudioBuffer c2;
            c2.sample_rate = 24000;
            c2.samples.assign(2400, 0.2f);
            chunk_cb(c2);
        });

        agent.on_transcript([&](const std::string& t) {
            transcript_received = t;
        });

        agent.on_response_text([&](const std::string& r) {
            response_received = r;
        });

        agent.on_audio_output([&](const brosoundml::AudioBuffer& chunk) {
            tts_chunks_received++;
            CHECK(chunk.samples.size() == 2400, "TTS audio chunk has expected sample count");
        });

        // Feed speech chunks (sine wave with amplitude 0.5 > 0.01 threshold)
        // Hop is 160 samples (10ms)
        auto speech_chunk = make_sine_chunk(480, 440.0f, 0.5f);
        for (int i = 0; i < 5; ++i) {
            agent.feed(speech_chunk.data(), static_cast<int>(speech_chunk.size()));
        }

        CHECK(speech_started, "Speech start detected and callback fired");
        CHECK(agent.is_listening(), "Agent transitioned to Listening state");

        // Feed silence chunks (amplitude 0.0 < 0.01) to trigger end of speech
        auto silence_chunk = make_sine_chunk(160, 0.0f, 0.0f);
        for (int i = 0; i < 10; ++i) {
            agent.feed(silence_chunk.data(), static_cast<int>(silence_chunk.size()));
        }

        CHECK(speech_ended, "Speech end detected and utterance callback fired");
        CHECK(transcript_received == "turn on the living room lights", "Transcript callback received text");
        CHECK(response_received == "Living room lights turned on.", "Response callback received text");
        CHECK(tts_chunks_received == 2, "TTS output delivered 2 chunks");
        CHECK(agent.is_idle(), "Agent returned to Idle after completing full dialogue turn");
        CHECK(state_transitions >= 4, "State machine transitioned through Idle->Listening->Thinking->Speaking->Idle");
    }

    // ── Test 3: Barge-in interruption during speaking ──
    {
        brosoundml::VoiceAgentConfig cfg;
        cfg.sample_rate = 16000;
        cfg.vad_energy_threshold = 0.01f;
        cfg.min_speech_frames = 2;
        cfg.enable_barge_in = true;
        brosoundml::VoiceAgent agent(cfg);

        bool barge_in_fired = false;
        agent.on_barge_in([&] {
            barge_in_fired = true;
        });

        // Set TTS handler that keeps agent in speaking until interrupted
        agent.set_tts_handler([&](const std::string&, brosoundml::VoiceAgent::TtsChunkCallback) {
            // Simulate speaking state
            auto speech_chunk = make_sine_chunk(480, 440.0f, 0.5f);
            agent.feed(speech_chunk.data(), static_cast<int>(speech_chunk.size()));
            agent.feed(speech_chunk.data(), static_cast<int>(speech_chunk.size()));
        });

        // Trigger manual speech
        agent.speak("Long assistant reply...");
        CHECK(barge_in_fired, "Barge-in callback fired upon speech during speaking");
        CHECK(agent.is_listening(), "Agent switched to Listening after barge-in");
    }

    // ── Test 4: Manual speak and interrupt ──
    {
        brosoundml::VoiceAgent agent;
        bool output_received = false;
        bool samples_are_clean = false;

        agent.on_audio_output([&](const brosoundml::AudioBuffer& chunk) {
            output_received = true;
            // Clean empty output rather than flat DC offset buffer (0.05f)
            samples_are_clean = chunk.samples.empty();
        });

        agent.speak("Test direct speech");
        CHECK(output_received, "Default TTS emitted audio chunk");
        CHECK(samples_are_clean, "Default TTS with no Kokoro produces clean empty output without DC offset");
        CHECK(agent.is_idle(), "Returned to idle after speech");

        agent.speak("Another speech");
        agent.interrupt();
        CHECK(agent.is_idle(), "Interrupt returns agent to idle");
    }

    // ── Test 5: BcResnet2d VAD Model Integration ──
    {
        brosoundml::BcResnet2dConfig vad_cfg;
        vad_cfg.n_mels = 40;
        auto vad_model = std::make_shared<brosoundml::BcResnet2d>(
            brosoundml::BcResnet2d::make(vad_cfg, brotensor::Device::CPU)
        );

        brosoundml::VoiceAgentConfig cfg;
        cfg.sample_rate = 16000;
        cfg.vad_threshold = 0.0f; // always active for test
        cfg.vad_energy_threshold = 0.001f;
        cfg.min_speech_frames = 2;
        cfg.silence_hangover_frames = 4;

        brosoundml::VoiceAgent agent(cfg);
        agent.set_vad_model(vad_model);

        auto audio_chunk = make_sine_chunk(320, 300.0f, 0.2f);
        agent.feed(audio_chunk.data(), static_cast<int>(audio_chunk.size()));
        CHECK(agent.last_energy() > 0.01f, "Energy computed from audio");
    }

    // ── Test 6: Real token decoding and phonemizer integration ──
    {
        brosoundml::VoiceAgentConfig cfg;
        cfg.sample_rate = 16000;
        cfg.vad_energy_threshold = 0.01f;
        cfg.min_speech_frames = 2;
        cfg.silence_hangover_frames = 4;
        brosoundml::VoiceAgent agent(cfg);

        // 1. Without STT model or tokenizer, no fake transcript is emitted
        bool transcript_emitted = false;
        agent.on_transcript([&](const std::string&) {
            transcript_emitted = true;
        });

        auto speech_chunk = make_sine_chunk(480, 440.0f, 0.5f);
        for (int i = 0; i < 5; ++i) {
            agent.feed(speech_chunk.data(), static_cast<int>(speech_chunk.size()));
        }
        auto silence_chunk = make_sine_chunk(160, 0.0f, 0.0f);
        for (int i = 0; i < 10; ++i) {
            agent.feed(silence_chunk.data(), static_cast<int>(silence_chunk.size()));
        }

        CHECK(!transcript_emitted, "Without STT model or tokenizer, no fake transcript is emitted");
        CHECK(agent.is_idle(), "Agent returns to idle cleanly when transcript is empty");

        // 2. Real STT tokenizer decoding
        bool tokenizer_called = false;
        agent.set_stt_tokenizer([&](const std::vector<int32_t>& tokens) -> std::string {
            tokenizer_called = true;
            CHECK(!tokens.empty(), "Tokenizer received tokens");
            return "real recognized command";
        });

        // 3. Real phonemizer
        bool phonemizer_called = false;
        agent.set_phonemizer([&](const std::string& text) -> std::vector<int32_t> {
            phonemizer_called = true;
            CHECK(text == "Hello world test", "Phonemizer received real text");
            return {12, 34, 56};
        });

        agent.speak("Hello world test");
        CHECK(phonemizer_called, "Real phonemizer called with real text");
        CHECK(agent.is_idle(), "Agent returns to idle after speaking");
    }

    if (failures == 0) {
        std::printf("test_voice_agent: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_voice_agent: %d check(s) failed\n", failures);
    return 1;
}
