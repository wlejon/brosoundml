#pragma once

// LayaListener — the streaming runtime of the Laya audio adapter (prototype;
// the public API shape it would take is in docs/laya-audio.md).
//
// Every hop (default 30 ms) the last `window_s` seconds of the stream are
// encoded by the Qwen3-ASR AuT encoder (FP16), projected into Laya's
// embedding width, and every registered question is answered in one packed
// Laya forward over that shared soft state. Keywords are questions ("The word
// "X" is spoken in this audio."), so any word can be registered at runtime
// with no enrollment; so can any other noul question about the audio.

#include "laya_audio_task.h"

#include "brosoundml/qwen_asr.h"

#include <brolm/laya.h>

#include <string>
#include <vector>

namespace laya_audio {

struct ListenerConfig {
    std::string encoder_dir = "weights/qwen-asr/0.6B";
    std::string laya_dir = "D:/projects/laya";
    std::string projector;  // .mlp from brosoundml_laya_audio_train
    float window_s = 3.0f;
    int hop_ms = 30;
    float temperature = 1.9834f;  // the checkpoint's noul:2 temperature
};

struct HopResult {
    double t_end = 0;           // stream time at the end of the window (s)
    std::vector<float> p;       // one calibrated probability per question
    double encode_ms = 0, laya_ms = 0, total_ms = 0;
};

class LayaListener {
public:
    void load(const ListenerConfig& cfg);
    // Questions answered each hop; returns the question's index in HopResult::p.
    int add_question(const std::string& instructions);
    int add_keyword(const std::string& word) { return add_question(question_text(Question::Keyword, word)); }
    void clear_questions() { questions_.clear(); }
    void reset();  // new stream
    // Push 16 kHz mono PCM; returns one result per hop completed by this push.
    std::vector<HopResult> feed(const float* pcm, int n);
    // The frozen Laya (for text questions against the same checkpoint).
    brolm::laya::DecisionModel& model() { return model_; }
    const ListenerConfig& config() const { return cfg_; }

private:
    HopResult run_hop_();
    ListenerConfig cfg_;
    brosoundml::QwenAsr asr_;
    brolm::laya::DecisionModel model_;
    Mlp proj_;
    std::unique_ptr<ItemBuilder> builder_;
    std::vector<std::string> questions_;
    std::vector<float> ring_;  // the last window_s seconds
    long long samples_ = 0, next_hop_ = 0;
};

}  // namespace laya_audio
