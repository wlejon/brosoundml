#include "laya_audio_questions.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace laya_audio {

namespace bt = brotensor;

namespace {

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream is(s);
    while (std::getline(is, cur, sep)) out.push_back(cur);
    return out;
}

bool tag_matches(const std::string& pattern, const std::string& tag) {
    if (!pattern.empty() && pattern.back() == '*')
        return tag.compare(0, pattern.size() - 1, pattern, 0, pattern.size() - 1) == 0;
    return pattern == tag;
}

// The heard part of a window: the stream is silent outside [0, duration].
void heard_range(const AlignedUtterance& u, float t_end, float window_s, float& lo, float& hi) {
    lo = std::max(t_end - window_s, 0.0f);
    hi = u.duration_s > 0 ? std::min(t_end, u.duration_s + 0.1f) : t_end;
}

}  // namespace

std::string BankQuestion::group() const {
    if (family_split == "para") return "paralinguistic";
    if (family_split == "heldout") return "new family";
    return phrasing_split == "train" ? "seen family+phrasing" : "seen family, new phrasing";
}

std::vector<BankQuestion> load_bank(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("laya_audio: cannot open question bank " + path);
    std::vector<BankQuestion> out;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> c = split(line, '\t');
        if (c.size() < 4) throw std::runtime_error("laya_audio: bad question bank line: " + line);
        BankQuestion q{c[0], c[1], c[2], c[3], c.size() > 4 ? c[4] : std::string()};
        out.push_back(std::move(q));
    }
    return out;
}

void LabelSet::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("laya_audio: cannot open labels " + path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::vector<std::string> c = split(line, '\t');
        if (c.size() < 4) continue;
        SpanLabel s{std::stof(c[1]), std::stof(c[2]), c[3]};
        const std::size_t eq = s.tag.find('=');
        if (eq != std::string::npos) ++namespaces_[s.tag.substr(0, eq)];
        by_utt_[c[0]].push_back(std::move(s));
    }
}

const std::vector<SpanLabel>* LabelSet::of(const std::string& utt) const {
    const auto it = by_utt_.find(utt);
    return it == by_utt_.end() ? nullptr : &it->second;
}

int real_label(const std::string& rule, const AlignedUtterance& u, const LabelSet* labels, float t_end,
               float window_s) {
    if (rule.empty()) return -1;
    if (rule.rfind("subset:", 0) == 0) {
        for (const std::string& p : split(rule.substr(7), '|'))
            if (u.subset.rfind(p, 0) == 0) return 1;
        return 0;
    }
    if (rule.rfind("tag:", 0) != 0 || !labels) return -1;
    const std::vector<std::string> pats = split(rule.substr(4), '|');
    bool applies = false;
    for (const std::string& p : pats) {
        const std::size_t eq = p.find('=');
        if (eq != std::string::npos && labels->has_namespace(p.substr(0, eq))) applies = true;
    }
    if (!applies) return -1;
    float lo, hi;
    heard_range(u, t_end, window_s, lo, hi);
    // An utterance-level label covers the utterance's speech extent.
    float s0 = 1e9f, s1 = -1;
    for (const TimedWord& w : u.words)
        if (w.t1 > w.t0 && w.t0 >= 0) {
            s0 = std::min(s0, w.t0);
            s1 = std::max(s1, w.t1);
        }
    if (s1 < 0) {
        s0 = 0;
        s1 = u.duration_s;
    }
    const std::vector<SpanLabel>* spans = labels->of(u.id);
    int label = 0;
    if (spans) {
        for (const SpanLabel& s : *spans) {
            bool m = false;
            for (const std::string& p : pats) m |= tag_matches(p, s.tag);
            if (!m) continue;
            const float a = s.t0 < 0 ? s0 : s.t0, b = s.t0 < 0 ? s1 : s.t1;
            const float ov = std::min(b, hi) - std::max(a, lo);
            if (ov <= 0) continue;
            if (ov >= 0.5f * (b - a) || ov >= 2.0f) return 1;
            label = -1;  // partly heard
        }
    }
    return label;
}

std::string window_words(const AlignedUtterance& u, float t_end, float window_s) {
    float lo, hi;
    heard_range(u, t_end, window_s, lo, hi);
    std::vector<std::pair<float, const std::string*>> ws;
    for (const TimedWord& w : u.words) {
        if (w.t1 <= w.t0 || w.t0 < 0) continue;
        const float mid = 0.5f * (w.t0 + w.t1);
        if (mid >= lo && mid <= hi) ws.push_back({w.t0, &w.word});
    }
    std::stable_sort(ws.begin(), ws.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string out;
    for (const auto& [t, w] : ws) {
        if (!out.empty()) out += ' ';
        out += *w;
    }
    return out;
}

std::string transcript_state(const std::string& words) {
    if (words.empty()) return "Audio transcript: (no speech)";
    return "Audio transcript: \"" + words + "\"";
}

std::vector<float> score_text(brolm::laya::DecisionModel& model, const std::vector<std::string>& states,
                              const std::vector<std::string>& questions,
                              const std::vector<std::pair<int, int>>& pairs, int batch_items) {
    std::vector<float> out(pairs.size(), 0.0f);
    std::vector<brolm::laya::SequenceResult> seqs;
    std::vector<brolm::laya::LayaItem> items;
    std::vector<std::size_t> idx;
    auto flush = [&]() {
        if (items.empty()) return;
        const std::vector<brolm::laya::LayaItemLogits> r = model.forward_items(items);
        for (std::size_t k = 0; k < idx.size(); ++k) out[idx[k]] = r[k].logits[1] - r[k].logits[0];
        seqs.clear();
        items.clear();
        idx.clear();
    };
    seqs.reserve(static_cast<std::size_t>(batch_items));
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        brolm::laya::LayaQuestion q;
        q.id = "q";
        q.type = "noul";
        q.instructions = questions[static_cast<std::size_t>(pairs[i].second)];
        seqs.push_back(model.build_sequence(states[static_cast<std::size_t>(pairs[i].first)], q));
        idx.push_back(i);
        if (static_cast<int>(seqs.size()) == batch_items) {
            for (const auto& s : seqs) items.push_back(brolm::laya::LayaItem::of(s, q.qtype_index()));
            flush();
        }
    }
    for (const auto& s : seqs) items.push_back(brolm::laya::LayaItem::of(s, 2));
    flush();
    return out;
}

std::vector<float> score_audio(brolm::laya::DecisionModel& model, ItemBuilder& builder, Mlp& proj,
                               const WindowCache& cache, const std::vector<std::string>& questions,
                               const std::vector<std::pair<int, int>>& pairs, int batch_windows) {
    std::vector<float> out(pairs.size(), 0.0f);
    std::size_t p = 0;
    while (p < pairs.size()) {
        // Up to batch_windows windows and kMaxItems items per forward (the
        // packed encoder's launch shapes bound one forward's size).
        constexpr std::size_t kMaxItems = 256;
        std::vector<int> wins;
        std::size_t e = p;
        while (e < pairs.size() && e - p < kMaxItems) {
            if (wins.empty() || wins.back() != pairs[e].first) {
                if (static_cast<int>(wins.size()) == batch_windows) break;
                wins.push_back(pairs[e].first);
            }
            ++e;
        }
        const int F = cache.windows[static_cast<std::size_t>(wins.front())].frames;
        const bt::Tensor lat = gather_latents(cache, wins);
        bt::Tensor y, soft;
        proj.forward(lat, y);
        bt::cast(y, soft, bt::compute_dtype());
        std::vector<brolm::laya::LayaItem> items;
        for (std::size_t i = p, k = 0; i < e; ++i) {
            if (pairs[i].first != wins[k]) ++k;
            items.push_back(builder.item(questions[static_cast<std::size_t>(pairs[i].second)], F, static_cast<int>(k) * F));
        }
        const std::vector<brolm::laya::LayaItemLogits> r = model.forward_items(items, &soft);
        for (std::size_t i = p; i < e; ++i) out[i] = r[i - p].logits[1] - r[i - p].logits[0];
        p = e;
    }
    return out;
}

double pearson(const std::vector<float>& a, const std::vector<float>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    if (n < 2) return std::nan("");
    double ma = 0, mb = 0;
    for (std::size_t i = 0; i < n; ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;
    double sab = 0, saa = 0, sbb = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
    }
    return saa > 0 && sbb > 0 ? sab / std::sqrt(saa * sbb) : std::nan("");
}

}  // namespace laya_audio
