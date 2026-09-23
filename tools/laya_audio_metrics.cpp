#include "laya_audio_metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace laya_audio {

namespace {
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
}

double tpr_at_fpr(const std::vector<float>& s, const std::vector<int>& l, double max_fpr) {
    std::vector<std::size_t> idx(s.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return s[a] > s[b]; });
    double P = 0, N = 0;
    for (int v : l) (v == 1 ? P : N) += 1;
    if (P == 0 || N == 0) return kNaN;
    double tp = 0, fp = 0, best = 0;
    for (std::size_t i = 0; i < idx.size();) {
        std::size_t j = i;
        while (j < idx.size() && s[idx[j]] == s[idx[i]]) {
            (l[idx[j]] == 1 ? tp : fp) += 1;
            ++j;
        }
        if (fp / N <= max_fpr) best = tp / P;
        i = j;
    }
    return best;
}

float threshold_at_fpr(std::vector<float> neg, double max_fpr) {
    if (neg.empty()) return kNaN;
    std::sort(neg.begin(), neg.end(), std::greater<float>());
    // Scores strictly above neg[k] are passed by at most k negatives.
    const std::size_t k = static_cast<std::size_t>(std::floor(max_fpr * static_cast<double>(neg.size())));
    return k < neg.size() ? std::nextafter(neg[k], std::numeric_limits<float>::infinity()) : neg.back();
}

void report_categories(const std::string& set, const std::vector<Probe>& probes, const std::vector<Method>& methods) {
    struct Cat {
        const char* name;
        std::function<bool(const Probe&)> in;
    };
    auto kw = [](const Probe& p) { return p.q == Question::Keyword; };
    const std::vector<Cat> cats = {
        {"keyword (all)", kw},
        {"keyword seen", [&](const Probe& p) { return kw(p) && !p.unseen; }},
        {"keyword unseen", [&](const Probe& p) { return kw(p) && p.unseen; }},
        {"keyword vs hard neg", [&](const Probe& p) { return kw(p) && (p.label == 1 || p.hard_neg); }},
        {"speaking", [](const Probe& p) { return p.q == Question::Speaking; }},
        {"word_end", [](const Probe& p) { return p.q == Question::WordEnd; }},
    };
    std::printf("\n%-16s %-22s %-12s %8s %8s %8s %7s %7s\n", "set", "category", "method", "AUC", "TPR@1%", "TPR@5%",
                "pos", "neg");
    for (const Cat& c : cats) {
        for (const Method& m : methods) {
            std::vector<float> s;
            std::vector<int> l;
            for (std::size_t i = 0; i < probes.size(); ++i) {
                if (!c.in(probes[i]) || std::isnan(m.score[i])) continue;
                s.push_back(m.score[i]);
                l.push_back(probes[i].label);
            }
            if (s.empty()) continue;
            const int pos = static_cast<int>(std::count(l.begin(), l.end(), 1));
            std::printf("%-16s %-22s %-12s %8.4f %8.4f %8.4f %7d %7d\n", set.c_str(), c.name, m.name.c_str(),
                        auc(s, l), tpr_at_fpr(s, l, 0.01), tpr_at_fpr(s, l, 0.05), pos,
                        static_cast<int>(l.size()) - pos);
        }
    }
    std::fflush(stdout);
}

std::unordered_map<std::string, int> training_counts(const std::vector<AlignedUtterance>& utts) {
    std::unordered_map<std::string, int> c;
    for (const AlignedUtterance& u : utts) {
        std::unordered_set<std::string> in_utt;
        for (const TimedWord& w : u.words) in_utt.insert(w.word);
        for (const std::string& w : in_utt)
            if (!held_out_keyword(w)) ++c[w];
    }
    return c;
}

void report_curve(const std::string& set, const std::vector<Probe>& probes, const std::vector<float>& score,
                  const std::unordered_map<std::string, int>& counts) {
    std::vector<float> neg;
    for (std::size_t i = 0; i < probes.size(); ++i)
        if (probes[i].q == Question::Keyword && probes[i].label == 0 && !std::isnan(score[i])) neg.push_back(score[i]);
    if (neg.empty()) return;
    const float thr1 = threshold_at_fpr(neg, 0.01), thr01 = threshold_at_fpr(neg, 0.001);
    constexpr int kB = 10;
    const char* names[kB] = {"0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256+"};
    auto bucket = [](int c) {
        if (c <= 0) return 0;
        int b = 1;
        while (b < kB - 1 && c >= (1 << b)) ++b;
        return b;
    };
    std::vector<std::vector<float>> pos(kB);
    std::vector<std::unordered_set<std::string>> words(kB);
    for (std::size_t i = 0; i < probes.size(); ++i) {
        const Probe& p = probes[i];
        if (p.q != Question::Keyword || p.label != 1 || std::isnan(score[i])) continue;
        const auto it = counts.find(p.keyword);
        const int b = bucket(it == counts.end() ? 0 : it->second);
        pos[b].push_back(score[i]);
        words[b].insert(p.keyword);
    }
    std::printf("\n%-16s examples-per-word curve (%zu keyword negatives; thresholds at 1%% / 0.1%% FPR)\n",
                set.c_str(), neg.size());
    std::printf("%-16s %-9s %6s %6s %10s %10s %8s\n", "set", "examples", "words", "pos", "recall@1%", "recall@.1%",
                "AUC");
    for (int b = 0; b < kB; ++b) {
        if (pos[b].empty()) continue;
        double r1 = 0, r01 = 0;
        for (float s : pos[b]) {
            r1 += s >= thr1;
            r01 += s >= thr01;
        }
        std::vector<float> s = pos[b];
        std::vector<int> l(pos[b].size(), 1);
        s.insert(s.end(), neg.begin(), neg.end());
        l.resize(s.size(), 0);
        std::printf("%-16s %-9s %6zu %6zu %10.3f %10.3f %8.4f\n", set.c_str(), names[b], words[b].size(),
                    pos[b].size(), r1 / pos[b].size(), r01 / pos[b].size(), auc(s, l));
    }
    std::fflush(stdout);
}

}  // namespace laya_audio
