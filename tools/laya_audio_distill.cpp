#include "laya_audio_distill.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace laya_audio {

namespace {
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

void Distiller::init(const std::string& bank_path, brolm::laya::DecisionModel* teacher, float teacher_temperature,
                     int cand, int keep) {
    bank_ = load_bank(bank_path);
    teacher_ = teacher;
    t_teacher_ = teacher_temperature;
    cand_ = cand;
    keep_ = keep;
    std::map<std::string, std::vector<int>> fam;
    for (std::size_t i = 0; i < bank_.size(); ++i) {
        qtext_.push_back(bank_[i].text);
        if (bank_[i].trained()) fam[bank_[i].family].push_back(static_cast<int>(i));
    }
    for (auto& [f, qs] : fam) train_families_.push_back(qs);
    if (train_families_.empty()) throw std::runtime_error("laya_audio: no trained questions in " + bank_path);
    std::fprintf(stderr, "distill: %zu questions, %zu trained families\n", bank_.size(), train_families_.size());
}

std::vector<DistillItem> Distiller::sample(const Corpus& c, const std::vector<int>& wins, std::mt19937& rng) {
    const double t0 = now_ms();
    std::vector<std::string> states;
    std::vector<std::pair<int, int>> pairs;
    std::uniform_int_distribution<std::size_t> fam(0, train_families_.size() - 1);
    for (std::size_t k = 0; k < wins.size(); ++k) {
        const CachedWindow& w = c.cache.windows[static_cast<std::size_t>(wins[k])];
        const AlignedUtterance& u = c.utts[static_cast<std::size_t>(w.utt)];
        states.push_back(window_words(u, w.t_end, c.cache.window_s));
        std::vector<std::size_t> fs;
        for (int tries = 0; static_cast<int>(fs.size()) < cand_ && tries < 20 * cand_; ++tries) {
            const std::size_t f = fam(rng);
            if (std::find(fs.begin(), fs.end(), f) == fs.end()) fs.push_back(f);
        }
        for (std::size_t f : fs) {
            const std::vector<int>& qs = train_families_[f];
            pairs.push_back({static_cast<int>(k), qs[std::uniform_int_distribution<std::size_t>(0, qs.size() - 1)(rng)]});
        }
    }
    const std::vector<float> z = score_transcripts(*teacher_, states, qtext_, pairs);
    std::vector<DistillItem> out;
    for (std::size_t k = 0, p = 0; k < wins.size(); ++k) {
        std::vector<DistillItem> cands;
        for (; p < pairs.size() && pairs[p].first == static_cast<int>(k); ++p)
            cands.push_back({static_cast<int>(k), pairs[p].second, sigmoid_t(z[p], t_teacher_)});
        std::shuffle(cands.begin(), cands.end(), rng);
        // Most-yes half first, then random ones.
        const int top = keep_ / 2;
        std::partial_sort(cands.begin(), cands.begin() + std::min<std::size_t>(top, cands.size()), cands.end(),
                          [](const DistillItem& a, const DistillItem& b) { return a.target > b.target; });
        for (int i = 0; i < std::min<int>(keep_, static_cast<int>(cands.size())); ++i) out.push_back(cands[i]);
    }
    teacher_ms_ = now_ms() - t0;
    return out;
}

Distiller::Dev Distiller::make_dev(const Corpus& c, int domain, int n_windows, std::mt19937& rng) {
    Dev d;
    d.name = c.domains[static_cast<std::size_t>(domain)].name;
    std::vector<int> wins = c.windows_of(domain);
    std::shuffle(wins.begin(), wins.end(), rng);
    wins.resize(std::min<std::size_t>(wins.size(), static_cast<std::size_t>(n_windows)));
    std::sort(wins.begin(), wins.end());
    std::vector<std::string> states;
    std::vector<std::pair<int, int>> tpairs;
    for (std::size_t k = 0; k < wins.size(); ++k) {
        const CachedWindow& w = c.cache.windows[static_cast<std::size_t>(wins[k])];
        states.push_back(window_words(c.utts[static_cast<std::size_t>(w.utt)], w.t_end, c.cache.window_s));
        for (std::size_t q = 0; q < bank_.size(); ++q) {
            if (bank_[q].family_split == "para") continue;
            d.pairs.push_back({wins[k], static_cast<int>(q)});
            tpairs.push_back({static_cast<int>(k), static_cast<int>(q)});
        }
    }
    const std::vector<float> z = score_transcripts(*teacher_, states, qtext_, tpairs);
    for (float v : z) d.teacher_p.push_back(sigmoid_t(v, t_teacher_));
    return d;
}

void Distiller::report(brolm::laya::DecisionModel& student, ItemBuilder& builder, Mlp& proj, const WindowCache& cache,
                       const Dev& dev, float student_temperature, const std::string& tag) {
    const std::vector<float> z = score_audio(student, builder, proj, cache, qtext_, dev.pairs);
    std::map<std::string, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < dev.pairs.size(); ++i)
        groups[bank_[static_cast<std::size_t>(dev.pairs[i].second)].group()].push_back(i);
    std::printf("[%s %s] free-form vs teacher:", tag.c_str(), dev.name.c_str());
    for (const auto& [g, idx] : groups) {
        std::vector<float> ps, pt;
        std::vector<int> yes;
        for (std::size_t i : idx) {
            ps.push_back(sigmoid_t(z[i], student_temperature));
            pt.push_back(dev.teacher_p[i]);
            yes.push_back(dev.teacher_p[i] >= 0.5f);
        }
        std::printf(" | %s AUC %.3f r %.3f", g.c_str(), auc(ps, yes), pearson(ps, pt));
    }
    std::printf("\n");
    std::fflush(stdout);
}

}  // namespace laya_audio
