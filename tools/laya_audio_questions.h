#pragma once

// laya-audio: free-form questions about live speech (docs/laya-audio.md,
// "Free-form questions").
//
//   * The question bank (laya_audio_questions.tsv): noul questions grouped
//     in families, with a split per family (train / heldout / para) and per
//     phrasing (train / heldout), and optionally a rule that derives a REAL
//     label for a window from a label file or from the corpus subset.
//   * Label files (brosoundml_laya_audio_labels): tagged time spans per
//     utterance.
//   * The text side: what a window's transcript is (gold words or ASR
//     words), how it is framed as a Laya state, and text-Laya scoring of
//     (state, question) pairs, the ceiling / baseline / distillation teacher.
//   * The audio side: the adapter's answer to (window, question) pairs.

#include "laya_audio_task.h"

#include "brolm/laya.h"

#include <cmath>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace laya_audio {

struct BankQuestion {
    std::string family;          // "intent.alarm"
    std::string family_split;    // "train", "heldout", "para"
    std::string phrasing_split;  // "train", "heldout"
    std::string text;
    std::string rule;  // "" | "tag:a=b|c=d*" | "subset:x|y"
    bool trained() const { return family_split == "train" && phrasing_split == "train"; }
    // "seen family / seen phrasing", "seen family / new phrasing", "new family", "paralinguistic"
    std::string group() const;
};

std::vector<BankQuestion> load_bank(const std::string& path);

struct SpanLabel {
    float t0 = -1, t1 = -1;  // t0 < 0: the whole utterance
    std::string tag;         // "intent=alarm_set"
};

// Tagged spans per utterance id, and the tag namespaces ("intent", "da")
// the file uses.
class LabelSet {
public:
    void load(const std::string& path);
    const std::vector<SpanLabel>* of(const std::string& utt) const;
    bool has_namespace(const std::string& ns) const { return namespaces_.count(ns) != 0; }
    bool empty() const { return by_utt_.empty(); }

private:
    std::unordered_map<std::string, std::vector<SpanLabel>> by_utt_;
    std::map<std::string, int> namespaces_;
};

// The real label of a question for the window ending at t_end: 1, 0, or -1
// (no label: rule not applicable to this set, or a span only partly heard).
// A span is heard if at least half of it, or 2 s of it, lies in the window;
// a window with no overlap at all with any matching span is negative.
int real_label(const std::string& rule, const AlignedUtterance& u, const LabelSet* labels, float t_end,
               float window_s);

// The gold transcript of a window: the words whose midpoint lies in the
// heard part of the window, in time order.
std::string window_words(const AlignedUtterance& u, float t_end, float window_s);

// A transcript framed as a text-Laya state.
std::string transcript_state(const std::string& words);

// Text Laya's logit (yes - no, pre-temperature) for each (state, question)
// pair: pairs index into `states` and `questions`. Batched through
// forward_items, `batch_items` at a time.
std::vector<float> score_text(brolm::laya::DecisionModel& model, const std::vector<std::string>& states,
                              const std::vector<std::string>& questions,
                              const std::vector<std::pair<int, int>>& pairs, int batch_items = 256);

// The adapter's logit for each (cache window, question) pair: the window's
// latents through `proj` into the state span. Pairs should be grouped by
// window (consecutive pairs of one window share its soft rows).
std::vector<float> score_audio(brolm::laya::DecisionModel& model, ItemBuilder& builder, Mlp& proj,
                               const WindowCache& cache, const std::vector<std::string>& questions,
                               const std::vector<std::pair<int, int>>& pairs, int batch_windows = 16);

double pearson(const std::vector<float>& a, const std::vector<float>& b);

inline float sigmoid_t(float z, float temperature) { return 1.0f / (1.0f + std::exp(-z / temperature)); }

}  // namespace laya_audio
