#pragma once

#include "brosoundml/audio.h"
#include "brosoundml/bc_resnet2d.h"
#include "brosoundml/kokoro.h"
#include "brosoundml/mel.h"
#include "brosoundml/parakeet.h"
#include "brosoundml/whisper.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace brosoundml {

namespace g2p {
class Phonemizer;
}

// ─── VoiceAgentState ───────────────────────────────────────────────────────
enum class VoiceAgentState {
    Idle,       // Waiting for user speech
    Listening,  // User speech detected, actively accumulating input audio
    Thinking,   // End of speech detected, running STT and text generation
    Speaking    // Synthesizing and streaming response audio via TTS
};

inline const char* voice_agent_state_name(VoiceAgentState s) {
    switch (s) {
        case VoiceAgentState::Idle:      return "Idle";
        case VoiceAgentState::Listening: return "Listening";
        case VoiceAgentState::Thinking:  return "Thinking";
        case VoiceAgentState::Speaking:  return "Speaking";
    }
    return "Unknown";
}

// ─── VoiceAgentConfig ──────────────────────────────────────────────────────
struct VoiceAgentConfig {
    int sample_rate = 16000;              // Input microphone sample rate (Hz)
    int tts_sample_rate = 24000;          // Output TTS sample rate (Hz)

    // Voice Activity Detection (VAD)
    float vad_threshold = 0.5f;           // Probability threshold for speech
    float vad_energy_threshold = 0.002f;  // Minimum RMS energy for active speech
    int min_speech_frames = 3;            // Frames required to enter Listening state
    int silence_hangover_frames = 25;     // Consecutive silence frames to end utterance (~250ms @ 10ms hop)
    int max_utterance_frames = 1500;      // Hard cap on single utterance length (~15s)

    // Full-duplex & Barge-in
    bool enable_barge_in = true;          // Allow user speech during Speaking to interrupt output

    // Mel front-end parameters (16 kHz, 10 ms hop, 25 ms window, 40 mels)
    int hop_length = 160;
    int win_length = 400;
    int n_mels = 40;
};

// ─── VoiceAgent ────────────────────────────────────────────────────────────
//
// Unified Streaming Duplex Voice Agent Pipeline Harness.
// Coordinates:
//   1. Audio input stream -> PCEN Mel front-end
//   2. Streaming VAD (BC-ResNet2D / energy detection)
//   3. Speech-to-Text transcription (Whisper / Parakeet / custom STT)
//   4. Streaming text callback (LLM / dialog engine)
//   5. Chunked TTS synthesis (Kokoro-82M / custom TTS) -> audio output chunks
//   6. Full-duplex interruption / barge-in handling
//
class VoiceAgent {
public:
    explicit VoiceAgent(VoiceAgentConfig config = {});
    ~VoiceAgent();

    VoiceAgent(VoiceAgent&&) noexcept;
    VoiceAgent& operator=(VoiceAgent&&) noexcept;
    VoiceAgent(const VoiceAgent&) = delete;
    VoiceAgent& operator=(const VoiceAgent&) = delete;

    // ── Pipeline Component Attachment ──
    void set_vad_model(std::shared_ptr<const BcResnet2d> vad_model);
    void set_whisper(std::shared_ptr<Whisper> whisper);
    void set_parakeet(std::shared_ptr<Parakeet> parakeet);
    void set_kokoro(std::shared_ptr<Kokoro> kokoro, Voice voice);

    // Real token decoding and phonemizer handler types
    using SttTokenizer = std::function<std::string(const std::vector<int32_t>& token_ids)>;
    using PhonemeHandler = std::function<std::vector<int32_t>(const std::string& text)>;

    void set_stt_tokenizer(SttTokenizer tokenizer);
    void set_whisper(std::shared_ptr<Whisper> whisper, SttTokenizer tokenizer);
    void set_parakeet(std::shared_ptr<Parakeet> parakeet, SttTokenizer tokenizer);
    void set_phonemizer(PhonemeHandler phonemizer);
    void set_phonemizer(std::shared_ptr<g2p::Phonemizer> phonemizer);
    void set_kokoro(std::shared_ptr<Kokoro> kokoro, Voice voice, PhonemeHandler phonemizer);
    void set_kokoro(std::shared_ptr<Kokoro> kokoro, Voice voice, std::shared_ptr<g2p::Phonemizer> phonemizer);

    // Custom injectable handlers (for mocking, custom LLM backends, or custom STT/TTS)
    using SttHandler = std::function<std::string(const AudioBuffer& utterance)>;
    using TextHandler = std::function<std::string(const std::string& user_query)>;
    using TtsChunkCallback = std::function<void(const AudioBuffer& audio_chunk)>;
    using TtsHandler = std::function<void(const std::string& text, TtsChunkCallback chunk_cb)>;

    void set_stt_handler(SttHandler handler);
    void set_text_handler(TextHandler handler);
    void set_tts_handler(TtsHandler handler);

    // ── Event Callbacks ──
    using StateCallback = std::function<void(VoiceAgentState old_state, VoiceAgentState new_state)>;
    using SpeechEventCallback = std::function<void()>;
    using UtteranceCallback = std::function<void(const AudioBuffer& utterance)>;
    using TranscriptCallback = std::function<void(const std::string& transcript)>;
    using ResponseCallback = std::function<void(const std::string& text_response)>;
    using AudioOutputCallback = std::function<void(const AudioBuffer& pcm_chunk)>;

    void on_state_changed(StateCallback cb);
    void on_speech_start(SpeechEventCallback cb);
    void on_speech_end(UtteranceCallback cb);
    void on_transcript(TranscriptCallback cb);
    void on_response_text(ResponseCallback cb);
    void on_audio_output(AudioOutputCallback cb);
    void on_barge_in(SpeechEventCallback cb);

    // ── Streaming Input & Control ──
    // Feed incoming microphone PCM audio samples (mono FP32 nominally in [-1, 1]).
    void feed(const float* samples, int num_samples);
    void feed(const AudioBuffer& audio);

    // Directly trigger agent speech response (transitions to Speaking and invokes TTS)
    void speak(const std::string& text);

    // Manually trigger interruption / cancellation (resets to Idle)
    void interrupt();

    // Reset all internal state and buffers
    void reset();

    // ── State Queries ──
    VoiceAgentState state() const;
    const VoiceAgentConfig& config() const;
    bool is_idle() const { return state() == VoiceAgentState::Idle; }
    bool is_listening() const { return state() == VoiceAgentState::Listening; }
    bool is_thinking() const { return state() == VoiceAgentState::Thinking; }
    bool is_speaking() const { return state() == VoiceAgentState::Speaking; }

    // Last detected metrics
    float last_vad_score() const;
    float last_energy() const;
    int utterance_frame_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace brosoundml
