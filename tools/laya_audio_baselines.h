#pragma once

// laya-audio baselines: the alternatives the Laya adapter is measured against.
//
//   AsrBaseline      Parakeet-TDT 0.6B on the window -> normalized words. Used
//                    as a string match ("keyword in the transcript") and as
//                    the state of a text Laya question (ASR text into Laya).
//   PhonemeBaseline  the open-vocabulary phoneme spotter: PhonemeNet
//                    posteriors over the window, g2p-enrolled keyword
//                    template, score = best completion confidence.

#include <memory>
#include <string>
#include <vector>

namespace laya_audio {

class AsrBaseline {
public:
    AsrBaseline();
    ~AsrBaseline();
    void load(const std::string& parakeet_dir);
    // 16 kHz window -> transcript (normalized words joined by spaces).
    std::string transcribe(const std::vector<float>& pcm16k);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class PhonemeBaseline {
public:
    PhonemeBaseline();
    ~PhonemeBaseline();
    // weights: .bpm checkpoint; data_dir holds g2p/lexicon_en_us.bin and
    // pos_tagger/model.bin; kokoro_dir supplies the phoneme vocabulary.
    void load(const std::string& weights, const std::string& data_dir, const std::string& kokoro_dir);
    void set_window(const std::vector<float>& pcm16k);  // runs the net once
    float score(const std::string& keyword);             // over the current window

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace laya_audio
