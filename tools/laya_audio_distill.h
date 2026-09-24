#pragma once

// laya-audio: distillation of free-form questions into the projector.
//
// The teacher is text Laya on a window's gold transcript (possibly a
// different, larger checkpoint than the student's frozen Laya). Per training
// window, `cand` questions are drawn from the TRAINED part of the bank
// (train families, train phrasings; a family first, then one of its
// phrasings), the teacher answers all of them in one batched text forward,
// and `keep` are trained on: the teacher's most-yes half and a random rest,
// so rare yes answers are seen often enough. The student is trained towards
// the teacher's probability (soft-target binary cross entropy).

#include "laya_audio_corpus.h"
#include "laya_audio_questions.h"

#include <random>
#include <string>
#include <vector>

namespace laya_audio {

struct DistillItem {
    int k = 0;           // index of the window in the batch
    int q = 0;           // bank index
    float target = 0.f;  // teacher probability of yes
};

class Distiller {
public:
    void init(const std::string& bank_path, brolm::laya::DecisionModel* teacher, float teacher_temperature, int cand,
              int keep);
    const std::vector<BankQuestion>& bank() const { return bank_; }
    const std::string& text(int q) const { return bank_[static_cast<std::size_t>(q)].text; }

    std::vector<DistillItem> sample(const Corpus& c, const std::vector<int>& wins, std::mt19937& rng);
    double last_teacher_ms() const { return teacher_ms_; }

    // Fixed dev probes for one domain: `n_windows` windows x every non-para
    // question, with the teacher's probabilities; report() prints the
    // student's agreement by group.
    struct Dev {
        std::string name;
        std::vector<std::pair<int, int>> pairs;  // (cache window, bank index)
        std::vector<float> teacher_p;
    };
    Dev make_dev(const Corpus& c, int domain, int n_windows, std::mt19937& rng);
    void report(brolm::laya::DecisionModel& student, ItemBuilder& builder, Mlp& proj, const WindowCache& cache,
                const Dev& dev, float student_temperature, const std::string& tag);

private:
    std::vector<BankQuestion> bank_;
    std::vector<std::vector<int>> train_families_;  // trained phrasings per train family
    std::vector<std::string> qtext_;
    brolm::laya::DecisionModel* teacher_ = nullptr;
    float t_teacher_ = 1.0f;
    int cand_ = 12, keep_ = 4;
    double teacher_ms_ = 0;
};

}  // namespace laya_audio
