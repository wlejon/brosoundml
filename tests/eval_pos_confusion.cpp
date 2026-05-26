// Confusion-matrix analyser for the POS tagger.
// Runs PosTagger::tag (the real inference path) over a UD validation bin and
// reports overall accuracy, per-tag P/R/F1, top confusion pairs, and a callout
// for the tags G2P actually consumes (heteronym / allomorphy disambiguation).

#include "brosoundml/g2p/pos_tagger.h"
#include "brosoundml/g2p/tags.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace g = brosoundml::g2p;

namespace {

[[noreturn]] void die(const std::string& msg) {
    throw std::runtime_error("eval_pos_confusion: " + msg);
}

struct Sentence {
    std::string                bytes;
    std::vector<std::uint16_t> word_start;
    std::vector<std::uint16_t> word_len;
    std::vector<std::uint8_t>  xpos;
};

std::vector<Sentence> load_val(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("could not open '" + path + "'");
    auto u32 = [&] { std::uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); return v; };
    auto u16 = [&] { std::uint16_t v; f.read(reinterpret_cast<char*>(&v), 2); return v; };
    const std::uint32_t magic = u32();
    if (magic != 0x504F5302u) die("bad magic in '" + path + "'");
    const std::uint32_t num_tags = u32();
    if (static_cast<int>(num_tags) != g::NUM_TAGS)
        die("dataset NUM_TAGS=" + std::to_string(num_tags) +
            " != compiled " + std::to_string(g::NUM_TAGS));
    const std::uint32_t n = u32();
    std::vector<Sentence> out; out.reserve(n);
    for (std::uint32_t s = 0; s < n; ++s) {
        Sentence sent;
        const std::uint32_t nb = u32();
        sent.bytes.resize(nb);
        f.read(sent.bytes.data(), nb);
        const std::uint32_t nw = u32();
        sent.xpos.resize(nw);
        f.read(reinterpret_cast<char*>(sent.xpos.data()), nw);
        std::vector<std::uint8_t> upos(nw);    // skip the upos block
        f.read(reinterpret_cast<char*>(upos.data()), nw);
        sent.word_start.resize(nw);
        for (std::uint32_t i = 0; i < nw; ++i) sent.word_start[i] = u16();
        sent.word_len.resize(nw);
        for (std::uint32_t i = 0; i < nw; ++i) sent.word_len[i] = u16();
        out.push_back(std::move(sent));
    }
    return out;
}

// Tags G2P relies on for heteronym disambiguation and allomorphy.
const std::set<std::string> kG2PRelevantTags = {
    "NN", "NNS", "NNP", "NNPS",
    "VB", "VBD", "VBZ", "VBN", "VBP", "VBG",
    "JJ", "JJR", "JJS",
    "RB", "RBR", "RBS",
    "DT", "IN", "TO", "MD", "PRP", "PRP$", "CC"
};

}  // namespace

int main(int argc, char** argv) {
    std::string val_path, weights_path;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nxt = [&] { if (i + 1 >= argc) die("missing value for " + k); return std::string(argv[++i]); };
        if      (k == "--val")     val_path     = nxt();
        else if (k == "--weights") weights_path = nxt();
        else die("unknown arg: " + k);
    }
    if (val_path.empty() || weights_path.empty()) {
        std::cerr << "usage: eval_pos_confusion --val <pos_val.bin> --weights <model.bin>\n";
        return 2;
    }

    auto val = load_val(val_path);
    auto tagger = g::PosTagger::load(weights_path);

    const int N = g::NUM_TAGS;
    std::vector<std::vector<std::uint64_t>> conf(N, std::vector<std::uint64_t>(N, 0));
    std::uint64_t total = 0, correct = 0;
    std::uint64_t skipped = 0;

    for (const auto& s : val) {
        const auto preds = tagger.tag(s.bytes);
        if (preds.size() != s.xpos.size()) { ++skipped; continue; }
        for (std::size_t i = 0; i < preds.size(); ++i) {
            const int gold = s.xpos[i];
            const int pred = static_cast<int>(preds[i].tag);
            conf[gold][pred]++;
            total++;
            if (gold == pred) correct++;
        }
    }

    auto tag_name = [&](int i) { return std::string(g::kPosTagNames[i]); };

    // Per-tag stats.
    struct TagStat { int idx; std::uint64_t support, tp, fp, fn; double p, r, f1; };
    std::vector<TagStat> stats(N);
    for (int g_i = 0; g_i < N; ++g_i) {
        std::uint64_t support = 0, tp = conf[g_i][g_i], fn = 0;
        for (int p = 0; p < N; ++p) support += conf[g_i][p];
        fn = support - tp;
        std::uint64_t fp = 0;
        for (int g2 = 0; g2 < N; ++g2) if (g2 != g_i) fp += conf[g2][g_i];
        const double p_v = (tp + fp > 0) ? double(tp) / (tp + fp) : 0.0;
        const double r_v = (tp + fn > 0) ? double(tp) / (tp + fn) : 0.0;
        const double f1  = (p_v + r_v > 0) ? 2.0 * p_v * r_v / (p_v + r_v) : 0.0;
        stats[g_i] = TagStat{g_i, support, tp, fp, fn, p_v, r_v, f1};
    }

    // Confusion pairs: (gold, pred) where gold != pred, sorted by count desc.
    struct Conf { int gold, pred; std::uint64_t count; };
    std::vector<Conf> pairs;
    for (int g_i = 0; g_i < N; ++g_i)
        for (int p = 0; p < N; ++p)
            if (g_i != p && conf[g_i][p] > 0)
                pairs.push_back(Conf{g_i, p, conf[g_i][p]});
    std::sort(pairs.begin(), pairs.end(),
              [](const Conf& a, const Conf& b) { return a.count > b.count; });

    // ── Report ──
    std::printf("eval: %llu words tagged, %llu correct, accuracy %.4f\n",
                static_cast<unsigned long long>(total),
                static_cast<unsigned long long>(correct),
                total ? double(correct) / total : 0.0);
    if (skipped) std::printf("eval: %llu sentences skipped (token-count mismatch)\n",
                             static_cast<unsigned long long>(skipped));

    std::printf("\nper-tag (sorted by support):\n");
    std::printf("%-8s %8s %8s %8s %8s %8s\n", "tag", "support", "P", "R", "F1", "errors");
    auto by_support = stats;
    std::sort(by_support.begin(), by_support.end(),
              [](const TagStat& a, const TagStat& b) { return a.support > b.support; });
    for (const auto& t : by_support) {
        if (t.support == 0) continue;
        std::printf("%-8s %8llu %8.4f %8.4f %8.4f %8llu\n",
                    tag_name(t.idx).c_str(),
                    static_cast<unsigned long long>(t.support),
                    t.p, t.r, t.f1,
                    static_cast<unsigned long long>(t.fn));
    }

    std::printf("\ntop 25 confusion pairs (gold -> pred):\n");
    const int max_pairs = std::min<int>(25, static_cast<int>(pairs.size()));
    for (int i = 0; i < max_pairs; ++i) {
        const auto& c = pairs[i];
        const std::uint64_t supp = stats[c.gold].support;
        const double frac = supp ? double(c.count) / supp : 0.0;
        std::printf("  %-6s -> %-6s  %6llu  (%.1f%% of %s)\n",
                    tag_name(c.gold).c_str(), tag_name(c.pred).c_str(),
                    static_cast<unsigned long long>(c.count),
                    100.0 * frac, tag_name(c.gold).c_str());
    }

    std::printf("\nG2P-relevant tags only:\n");
    std::printf("%-8s %8s %8s %8s %8s %8s\n", "tag", "support", "P", "R", "F1", "errors");
    std::uint64_t g_total = 0, g_correct = 0;
    for (const auto& t : by_support) {
        const std::string nm = tag_name(t.idx);
        if (!kG2PRelevantTags.count(nm)) continue;
        if (t.support == 0) continue;
        std::printf("%-8s %8llu %8.4f %8.4f %8.4f %8llu\n",
                    nm.c_str(),
                    static_cast<unsigned long long>(t.support),
                    t.p, t.r, t.f1,
                    static_cast<unsigned long long>(t.fn));
        g_total   += t.support;
        g_correct += t.tp;
    }
    std::printf("G2P-relevant accuracy: %.4f (%llu / %llu)\n",
                g_total ? double(g_correct) / g_total : 0.0,
                static_cast<unsigned long long>(g_correct),
                static_cast<unsigned long long>(g_total));

    return 0;
}
