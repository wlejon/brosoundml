#include "brosoundml/voice_agent.h"
#include "brosoundml/mel.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

namespace brosoundml {

static inline float sigmoidf(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

struct VoiceAgent::Impl {
    VoiceAgentConfig config;
    VoiceAgentState state = VoiceAgentState::Idle;

    // Attached Models
    std::shared_ptr<const BcResnet2d> vad_model;
    std::unique_ptr<BcResnet2dSession> vad_session;
    std::shared_ptr<Whisper> whisper;
    std::shared_ptr<Parakeet> parakeet;
    std::shared_ptr<Kokoro> kokoro;
    std::optional<Voice> voice;

    // Custom Handlers
    SttHandler stt_handler;
    TextHandler text_handler;
    TtsHandler tts_handler;

    // Event Callbacks
    StateCallback on_state_changed_cb;
    SpeechEventCallback on_speech_start_cb;
    UtteranceCallback on_speech_end_cb;
    TranscriptCallback on_transcript_cb;
    ResponseCallback on_response_text_cb;
    AudioOutputCallback on_audio_output_cb;
    SpeechEventCallback on_barge_in_cb;

    // Front-end & Audio state
    std::unique_ptr<MelFrontend> mel_fe;
    brotensor::Tensor mel_buffer;
    brotensor::Tensor vad_logits;

    std::vector<float> utterance_audio;
    float last_vad_score = 0.0f;
    float last_energy = 0.0f;
    int consecutive_speech_frames = 0;
    int consecutive_silence_frames = 0;
    int utterance_frames = 0;

    explicit Impl(VoiceAgentConfig cfg) : config(std::move(cfg)) {
        MelConfig mel_cfg;
        mel_cfg.sample_rate = config.sample_rate;
        mel_cfg.win_length = config.win_length;
        mel_cfg.hop_length = config.hop_length;
        mel_cfg.n_mels = config.n_mels;
        mel_cfg.compression = MelCompression::PCEN;
        mel_fe = std::make_unique<MelFrontend>(mel_cfg, brotensor::Device::CPU);
    }

    void set_state(VoiceAgentState new_state) {
        if (state != new_state) {
            VoiceAgentState old_state = state;
            state = new_state;
            if (on_state_changed_cb) {
                on_state_changed_cb(old_state, new_state);
            }
        }
    }

    void process_utterance(const AudioBuffer& utterance) {
        if (utterance.samples.empty()) {
            set_state(VoiceAgentState::Idle);
            return;
        }

        // 1. Speech-to-Text (STT)
        std::string transcript;
        if (stt_handler) {
            transcript = stt_handler(utterance);
        } else if (whisper) {
            std::vector<int32_t> prompt = {whisper->config().decoder_start_token_id};
            Whisper::Transcription res = whisper->transcribe(utterance, prompt);
            transcript = "transcript_" + std::to_string(res.token_ids.size()) + "_tokens";
        } else if (parakeet) {
            Parakeet::Transcription res = parakeet->transcribe(utterance);
            transcript = "transcript_" + std::to_string(res.token_ids.size()) + "_tokens";
        } else {
            transcript = "user utterance (" + std::to_string(utterance.samples.size()) + " samples)";
        }

        if (transcript.empty()) {
            set_state(VoiceAgentState::Idle);
            return;
        }

        if (on_transcript_cb) {
            on_transcript_cb(transcript);
        }

        // 2. Text / LLM Callback
        std::string response_text;
        if (text_handler) {
            response_text = text_handler(transcript);
        } else {
            response_text = "Echo: " + transcript;
        }

        if (response_text.empty()) {
            set_state(VoiceAgentState::Idle);
            return;
        }

        if (on_response_text_cb) {
            on_response_text_cb(response_text);
        }

        // 3. Text-to-Speech (TTS)
        synthesize_and_speak(response_text);
    }

    void synthesize_and_speak(const std::string& text) {
        set_state(VoiceAgentState::Speaking);

        if (tts_handler) {
            tts_handler(text, [this](const AudioBuffer& chunk) {
                if (state == VoiceAgentState::Speaking && on_audio_output_cb) {
                    on_audio_output_cb(chunk);
                }
            });
        } else if (kokoro && voice.has_value()) {
            std::vector<int32_t> phoneme_tokens = {1, 2, 3};
            AudioBuffer audio = kokoro->synthesize(phoneme_tokens, *voice);
            if (state == VoiceAgentState::Speaking && on_audio_output_cb) {
                on_audio_output_cb(audio);
            }
        } else {
            AudioBuffer dummy;
            dummy.sample_rate = config.tts_sample_rate;
            dummy.samples.assign(4800, 0.05f);
            if (state == VoiceAgentState::Speaking && on_audio_output_cb) {
                on_audio_output_cb(dummy);
            }
        }

        if (state == VoiceAgentState::Speaking) {
            set_state(VoiceAgentState::Idle);
        }
    }
};

VoiceAgent::VoiceAgent(VoiceAgentConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

VoiceAgent::~VoiceAgent() = default;
VoiceAgent::VoiceAgent(VoiceAgent&&) noexcept = default;
VoiceAgent& VoiceAgent::operator=(VoiceAgent&&) noexcept = default;

void VoiceAgent::set_vad_model(std::shared_ptr<const BcResnet2d> vad_model) {
    impl_->vad_model = std::move(vad_model);
    if (impl_->vad_model) {
        impl_->vad_session = std::make_unique<BcResnet2dSession>(impl_->vad_model->make_session());
    } else {
        impl_->vad_session.reset();
    }
}

void VoiceAgent::set_whisper(std::shared_ptr<Whisper> whisper) {
    impl_->whisper = std::move(whisper);
}

void VoiceAgent::set_parakeet(std::shared_ptr<Parakeet> parakeet) {
    impl_->parakeet = std::move(parakeet);
}

void VoiceAgent::set_kokoro(std::shared_ptr<Kokoro> kokoro, Voice voice) {
    impl_->kokoro = std::move(kokoro);
    impl_->voice = std::move(voice);
}

void VoiceAgent::set_stt_handler(SttHandler handler) {
    impl_->stt_handler = std::move(handler);
}

void VoiceAgent::set_text_handler(TextHandler handler) {
    impl_->text_handler = std::move(handler);
}

void VoiceAgent::set_tts_handler(TtsHandler handler) {
    impl_->tts_handler = std::move(handler);
}

void VoiceAgent::on_state_changed(StateCallback cb) {
    impl_->on_state_changed_cb = std::move(cb);
}

void VoiceAgent::on_speech_start(SpeechEventCallback cb) {
    impl_->on_speech_start_cb = std::move(cb);
}

void VoiceAgent::on_speech_end(UtteranceCallback cb) {
    impl_->on_speech_end_cb = std::move(cb);
}

void VoiceAgent::on_transcript(TranscriptCallback cb) {
    impl_->on_transcript_cb = std::move(cb);
}

void VoiceAgent::on_response_text(ResponseCallback cb) {
    impl_->on_response_text_cb = std::move(cb);
}

void VoiceAgent::on_audio_output(AudioOutputCallback cb) {
    impl_->on_audio_output_cb = std::move(cb);
}

void VoiceAgent::on_barge_in(SpeechEventCallback cb) {
    impl_->on_barge_in_cb = std::move(cb);
}

void VoiceAgent::feed(const float* samples, int num_samples) {
    if (!samples || num_samples <= 0) return;

    // Calculate RMS energy of incoming chunk
    float sum_sq = 0.0f;
    for (int i = 0; i < num_samples; ++i) {
        sum_sq += samples[i] * samples[i];
    }
    float chunk_energy = std::sqrt(sum_sq / static_cast<float>(num_samples));
    impl_->last_energy = chunk_energy;

    // Extract Mel frames
    int new_frames = impl_->mel_fe->consume(samples, num_samples, impl_->mel_buffer);

    std::vector<float> vad_scores;
    if (new_frames > 0) {
        if (impl_->vad_model && impl_->vad_session) {
            impl_->vad_model->forward_streaming(*impl_->vad_session, impl_->mel_buffer, impl_->vad_logits);
            std::vector<float> logits = impl_->vad_logits.to_host_vector();
            vad_scores.resize(logits.size());
            for (size_t i = 0; i < logits.size(); ++i) {
                vad_scores[i] = sigmoidf(logits[i]);
            }
        } else {
            float e_score = (chunk_energy >= impl_->config.vad_energy_threshold) ? 1.0f : 0.0f;
            vad_scores.assign(new_frames, e_score);
        }
    }

    for (int f = 0; f < new_frames; ++f) {
        float vad_score = (f < static_cast<int>(vad_scores.size())) ? vad_scores[f] : 0.0f;
        impl_->last_vad_score = vad_score;

        bool is_speech = (vad_score >= impl_->config.vad_threshold) &&
                         (chunk_energy >= impl_->config.vad_energy_threshold);

        switch (impl_->state) {
            case VoiceAgentState::Idle: {
                if (is_speech) {
                    impl_->consecutive_speech_frames++;
                    if (impl_->consecutive_speech_frames >= impl_->config.min_speech_frames) {
                        impl_->set_state(VoiceAgentState::Listening);
                        impl_->consecutive_speech_frames = 0;
                        impl_->consecutive_silence_frames = 0;
                        impl_->utterance_frames = 0;
                        impl_->utterance_audio.clear();
                        if (impl_->on_speech_start_cb) {
                            impl_->on_speech_start_cb();
                        }
                    }
                } else {
                    impl_->consecutive_speech_frames = 0;
                }
                break;
            }

            case VoiceAgentState::Listening: {
                impl_->utterance_frames++;
                if (is_speech) {
                    impl_->consecutive_silence_frames = 0;
                } else {
                    impl_->consecutive_silence_frames++;
                }

                if (impl_->consecutive_silence_frames >= impl_->config.silence_hangover_frames ||
                    impl_->utterance_frames >= impl_->config.max_utterance_frames) {
                    // End of utterance detected
                    impl_->set_state(VoiceAgentState::Thinking);
                    AudioBuffer utterance;
                    utterance.sample_rate = impl_->config.sample_rate;
                    utterance.samples = std::move(impl_->utterance_audio);
                    impl_->utterance_audio.clear();
                    impl_->consecutive_silence_frames = 0;
                    impl_->consecutive_speech_frames = 0;

                    if (impl_->on_speech_end_cb) {
                        impl_->on_speech_end_cb(utterance);
                    }

                    impl_->process_utterance(utterance);
                }
                break;
            }

            case VoiceAgentState::Speaking: {
                if (impl_->config.enable_barge_in && is_speech) {
                    impl_->consecutive_speech_frames++;
                    if (impl_->consecutive_speech_frames >= impl_->config.min_speech_frames) {
                        if (impl_->on_barge_in_cb) {
                            impl_->on_barge_in_cb();
                        }
                        impl_->set_state(VoiceAgentState::Listening);
                        impl_->consecutive_speech_frames = 0;
                        impl_->consecutive_silence_frames = 0;
                        impl_->utterance_frames = 0;
                        impl_->utterance_audio.clear();
                        if (impl_->on_speech_start_cb) {
                            impl_->on_speech_start_cb();
                        }
                    }
                } else {
                    impl_->consecutive_speech_frames = 0;
                }
                break;
            }

            case VoiceAgentState::Thinking: {
                if (impl_->config.enable_barge_in && is_speech) {
                    impl_->consecutive_speech_frames++;
                    if (impl_->consecutive_speech_frames >= impl_->config.min_speech_frames) {
                        if (impl_->on_barge_in_cb) {
                            impl_->on_barge_in_cb();
                        }
                        impl_->set_state(VoiceAgentState::Listening);
                        impl_->consecutive_speech_frames = 0;
                        impl_->consecutive_silence_frames = 0;
                        impl_->utterance_frames = 0;
                        impl_->utterance_audio.clear();
                        if (impl_->on_speech_start_cb) {
                            impl_->on_speech_start_cb();
                        }
                    }
                } else {
                    impl_->consecutive_speech_frames = 0;
                }
                break;
            }
        }
    }

    // Accumulate samples while actively listening
    if (impl_->state == VoiceAgentState::Listening) {
        impl_->utterance_audio.insert(impl_->utterance_audio.end(), samples, samples + num_samples);
    }
}

void VoiceAgent::feed(const AudioBuffer& audio) {
    feed(audio.samples.data(), static_cast<int>(audio.samples.size()));
}

void VoiceAgent::speak(const std::string& text) {
    impl_->synthesize_and_speak(text);
}

void VoiceAgent::interrupt() {
    impl_->utterance_audio.clear();
    impl_->consecutive_silence_frames = 0;
    impl_->consecutive_speech_frames = 0;
    impl_->utterance_frames = 0;
    impl_->set_state(VoiceAgentState::Idle);
}

void VoiceAgent::reset() {
    interrupt();
    impl_->mel_fe->reset();
    if (impl_->vad_model && impl_->vad_session) {
        impl_->vad_model->reset(*impl_->vad_session);
    }
}

VoiceAgentState VoiceAgent::state() const {
    return impl_->state;
}

const VoiceAgentConfig& VoiceAgent::config() const {
    return impl_->config;
}

float VoiceAgent::last_vad_score() const {
    return impl_->last_vad_score;
}

float VoiceAgent::last_energy() const {
    return impl_->last_energy;
}

int VoiceAgent::utterance_frame_count() const {
    return impl_->utterance_frames;
}

} // namespace brosoundml
