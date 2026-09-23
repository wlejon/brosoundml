#pragma once

// laya-audio: evaluation metrics shared by the eval and listen tools.

#include "laya_audio_task.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace laya_audio {

// TPR at the largest threshold whose FPR stays <= max_fpr (NaN if a class
// is empty).
double tpr_at_fpr(const std::vector<float>& s, const std::vector<int>& l, double max_fpr);

// The lowest threshold whose FPR over `neg` stays <= max_fpr.
float threshold_at_fpr(std::vector<float> neg, double max_fpr);

// Scores of one method, NaN = not applicable to that probe.
struct Method {
    std::string name;
    std::vector<float> score;
};

// Rows "category method AUC TPR@1% TPR@5% pos neg" for the standard probe
// categories (keyword all / seen / unseen / vs hard neg, speaking, word_end),
// each prefixed by `set`.
void report_categories(const std::string& set, const std::vector<Probe>& probes, const std::vector<Method>& methods);

// Training examples per word: the number of training utterances holding the
// word, 0 for held-out words (never asked in training).
std::unordered_map<std::string, int> training_counts(const std::vector<AlignedUtterance>& utts);

// Examples-per-word curve: keyword positives bucketed by the word's training
// count (0, 1, 2-3, 4-7, ... 256+); per bucket the recall at the global
// thresholds that hold the FPR over ALL keyword negatives of the set to 1 %
// and 0.1 %, and the AUC of the bucket's positives against all negatives.
void report_curve(const std::string& set, const std::vector<Probe>& probes, const std::vector<float>& score,
                  const std::unordered_map<std::string, int>& counts);

}  // namespace laya_audio
