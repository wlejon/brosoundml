// brosoundml_laya_audio_ask — free-form questions about speech windows:
// the adapter against text Laya on the gold transcript (the ceiling) and
// text Laya on an ASR transcript (the baseline), per question family.
//
// Every question of the bank (laya_audio_questions.tsv) is asked of every
// sampled window of every set, by three methods:
//   audio   the adapter: projected AuT latents in Laya's state span
//   gold    text Laya on the window's gold words (the ceiling, and the
//           distillation teacher)
//   asr     (--asr) text Laya on Parakeet's words for the window, decoded
//           with full left context up to the window end (no lookahead)
//
// Reports, per set:
//   agreement with the ceiling, by question group (seen family+phrasing /
//   seen family, new phrasing / new family / paralinguistic) and family:
//   the ceiling's yes-rate (p >= 0.5), AUC of the method's score against
//   the ceiling's yes/no, Pearson r of the probabilities, mean |dp|.
// and, pooled over sets and per set, REAL-label results for every question
// whose rule yields a label: AUC and balanced accuracy at p >= 0.5.
//
// Usage:
//   brosoundml_laya_audio_ask --bank tools/laya_audio_questions.tsv --proj P.mlp
//        --set NAME,ALIGN,CACHE[,LABELS] [--set ...] [--laya DIR] [--temperature 1]
//        [--windows 400] [--asr] [--seed 1] [--dump PREFIX] [--families a,b] [--no-para]
//        [--text-laya DIR --text-temperature T]   (text methods on another checkpoint, e.g. the teacher)
//   brosoundml_laya_audio_ask --bank Q.tsv --text transcripts.txt   (the ceiling on typed transcripts)

#include "laya_audio_baselines.h"
#include "laya_audio_corpus.h"
#include "laya_audio_questions.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace bt = brotensor;
using namespace laya_audio;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "laya_audio_ask: %s\n", msg.c_str());
    std::exit(2);
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream is(s);
    while (std::getline(is, cur, sep)) out.push_back(cur);
    return out;
}

struct Rec {
    int set = 0, window = 0, q = 0, label = -1;
    float z[3] = {0, 0, 0};  // audio, gold, asr (NaN when not run)
};

const char* kMethod[3] = {"audio", "gold", "asr"};

struct Agree {
    double yes_rate = 0, auc[3] = {}, r[3] = {}, mad[3] = {};
    int n = 0;
};

// Agreement of audio / asr with the gold-text ceiling over a group of records.
Agree agreement(const std::vector<const Rec*>& rs, float T, bool has_asr) {
    Agree a;
    a.n = static_cast<int>(rs.size());
    if (rs.empty()) return a;
    std::vector<int> yes;
    std::vector<float> p[3];
    for (const Rec* r : rs) {
        const float pg = sigmoid_t(r->z[1], T);
        yes.push_back(pg >= 0.5f);
        a.yes_rate += pg >= 0.5f;
        for (int m = 0; m < 3; ++m) p[m].push_back(sigmoid_t(r->z[m], T));
    }
    a.yes_rate /= rs.size();
    for (int m : {0, 2}) {
        if (m == 2 && !has_asr) continue;
        a.auc[m] = auc(p[m], yes);
        a.r[m] = pearson(p[m], p[1]);
        double d = 0;
        for (std::size_t i = 0; i < p[m].size(); ++i) d += std::fabs(p[m][i] - p[1][i]);
        a.mad[m] = d / p[m].size();
    }
    return a;
}

void print_agree(const std::string& label, const Agree& a, bool has_asr) {
    std::printf("  %-34s n %6d  ceiling yes %.3f | audio AUC %.3f r %.3f |dp| %.3f", label.c_str(), a.n, a.yes_rate,
                a.auc[0], a.r[0], a.mad[0]);
    if (has_asr) std::printf(" | asr AUC %.3f r %.3f |dp| %.3f", a.auc[2], a.r[2], a.mad[2]);
    std::printf("\n");
}

struct RealRow {
    int pos = 0, neg = 0;
    double auc[3] = {}, bacc[3] = {};
};

RealRow real_metrics(const std::vector<const Rec*>& rs, float T, bool has_asr) {
    RealRow row;
    std::vector<int> l;
    std::vector<float> s[3];
    for (const Rec* r : rs) {
        if (r->label < 0) continue;
        l.push_back(r->label);
        (r->label ? row.pos : row.neg)++;
        for (int m = 0; m < 3; ++m) s[m].push_back(r->z[m]);
    }
    for (int m = 0; m < 3; ++m) {
        if (m == 2 && !has_asr) continue;
        row.auc[m] = auc(s[m], l);
        double tp = 0, tn = 0;
        for (std::size_t i = 0; i < l.size(); ++i) {
            const bool yes = sigmoid_t(s[m][i], T) >= 0.5f;
            tp += l[i] && yes;
            tn += !l[i] && !yes;
        }
        row.bacc[m] = 0.5 * (row.pos ? tp / row.pos : std::nan("")) + 0.5 * (row.neg ? tn / row.neg : std::nan(""));
    }
    return row;
}

void print_real(const std::string& label, const RealRow& r, bool has_asr) {
    std::printf("  %-44s pos %5d neg %6d |", label.c_str(), r.pos, r.neg);
    for (int m = 0; m < 3; ++m) {
        if (m == 2 && !has_asr) continue;
        std::printf(" %s AUC %.3f bacc %.3f |", kMethod[m], r.auc[m], r.bacc[m]);
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string bank_path, proj_path, laya = "D:/projects/laya/multilingual", dump, text_file;
    std::string parakeet_dir = "weights/parakeet/0.6b-v3";
    std::vector<std::string> sets, families;
    int n_windows = 400;
    uint32_t seed = 1;
    bool asr = false, para = true;
    float T = 1.0f, Tt = -1.0f;
    std::string text_laya;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(k + " needs a value");
            return argv[++i];
        };
        if (k == "--bank") bank_path = next();
        else if (k == "--proj") proj_path = next();
        else if (k == "--laya") laya = next();
        else if (k == "--set") sets.push_back(next());
        else if (k == "--windows") n_windows = std::atoi(next().c_str());
        else if (k == "--seed") seed = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (k == "--asr") asr = true;
        else if (k == "--temperature") T = std::stof(next());
        else if (k == "--dump") dump = next();
        else if (k == "--families") families = split(next(), ',');
        else if (k == "--no-para") para = false;
        else if (k == "--parakeet-dir") parakeet_dir = next();
        else if (k == "--text") text_file = next();
        else if (k == "--text-laya") text_laya = next();
        else if (k == "--text-temperature") Tt = std::stof(next());
        else die("unknown argument " + k);
    }
    if (bank_path.empty() || ((proj_path.empty() || sets.empty()) && text_file.empty()))
        die("need --bank --proj --set (or --bank --text)");

    try {
        bt::init();
        std::vector<BankQuestion> bank;
        for (BankQuestion& q : load_bank(bank_path)) {
            if (!para && q.family_split == "para") continue;
            bool keep = families.empty();
            for (const std::string& f : families) keep |= q.family.rfind(f, 0) == 0;
            if (keep) bank.push_back(std::move(q));
        }
        std::vector<std::string> qtext;
        for (const BankQuestion& q : bank) qtext.push_back(q.text);
        std::fprintf(stderr, "%zu questions\n", bank.size());

        Corpus ev;
        std::vector<LabelSet> labels(sets.size());
        std::vector<std::string> names;
        for (std::size_t s = 0; s < sets.size(); ++s) {
            const std::vector<std::string> f = split(sets[s], ',');
            if (f.size() < 3) die("bad --set " + sets[s]);
            ev.add(DomainSpec{f[0], 1.0f, f[1], {f[2]}});
            if (f.size() > 3 && !f[3].empty()) labels[s].load(f[3]);
            names.push_back(f[0]);
        }

        brolm::laya::DecisionModel model, text_own;
        model.load_model(laya);
        // The text methods (ceiling, ASR baseline) may use another checkpoint
        // (--text-laya), e.g. the distillation teacher.
        brolm::laya::DecisionModel* tm = &model;
        if (!text_laya.empty() && text_laya != laya) {
            text_own.load_model(text_laya);
            tm = &text_own;
        }
        if (Tt <= 0) Tt = T;
        if (!text_file.empty()) {
            // Text mode: every bank question against each line (a transcript)
            // of the file, as the ceiling sees it; a check of the framing.
            std::ifstream f(text_file);
            std::vector<std::string> states, lines;
            std::string line;
            while (std::getline(f, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                lines.push_back(line);
                states.push_back(transcript_state(line));
            }
            std::vector<std::pair<int, int>> pairs;
            for (std::size_t s = 0; s < states.size(); ++s)
                for (std::size_t q = 0; q < bank.size(); ++q) pairs.push_back({static_cast<int>(s), static_cast<int>(q)});
            const std::vector<float> z = score_text(*tm, states, qtext, pairs);
            for (std::size_t s = 0; s < states.size(); ++s) {
                std::printf("\n%s\n", lines[s].c_str());
                for (std::size_t q = 0; q < bank.size(); ++q) {
                    const float p = sigmoid_t(z[s * bank.size() + q], Tt);
                    if (p >= 0.2f) std::printf("  %.2f  %-16s %s\n", p, bank[q].family.c_str(), bank[q].text.c_str());
                }
            }
            return 0;
        }
        ItemBuilder builder(model);
        Mlp proj;
        proj.load(proj_path);
        AsrBaseline asr_m;
        if (asr) asr_m.load(parakeet_dir);

        std::FILE* dq = nullptr;
        std::FILE* dw = nullptr;
        if (!dump.empty()) {
            dq = std::fopen((dump + "_answers.tsv").c_str(), "wb");
            dw = std::fopen((dump + "_windows.tsv").c_str(), "wb");
            if (!dq || !dw) die("cannot write dump " + dump);
            std::fprintf(dq, "set\twindow\tq\tlabel\tz_audio\tz_gold\tz_asr\n");
            std::fprintf(dw, "set\twindow\tutt\tt_end\tgold\tasr\n");
            for (std::size_t q = 0; q < bank.size(); ++q)
                std::fprintf(dq, "#q\t%zu\t%s\t%s\t%s\t%s\n", q, bank[q].family.c_str(), bank[q].family_split.c_str(),
                             bank[q].phrasing_split.c_str(), bank[q].text.c_str());
        }

        std::vector<Rec> all;
        for (std::size_t d = 0; d < ev.domains.size(); ++d) {
            std::mt19937 rng(seed);
            std::vector<int> wins = ev.windows_of(static_cast<int>(d));
            std::shuffle(wins.begin(), wins.end(), rng);
            wins.resize(std::min<std::size_t>(wins.size(), static_cast<std::size_t>(n_windows)));
            std::sort(wins.begin(), wins.end());
            const LabelSet* lab = labels[d].empty() ? nullptr : &labels[d];

            std::vector<std::string> gold_words, asr_words;
            std::vector<std::pair<int, int>> apairs, tpairs;
            std::vector<Rec> recs;
            int cur_utt = -1;
            std::vector<float> pcm;
            const double t_asr0 = now_ms();
            for (std::size_t k = 0; k < wins.size(); ++k) {
                const CachedWindow& cw = ev.cache.windows[static_cast<std::size_t>(wins[k])];
                const AlignedUtterance& u = ev.utts[static_cast<std::size_t>(cw.utt)];
                const std::string gold = window_words(u, cw.t_end, ev.cache.window_s);
                gold_words.push_back(gold);
                std::string hyp;
                if (asr) {
                    if (cw.utt != cur_utt) {
                        pcm = load_audio_16k(u.wav);
                        cur_utt = cw.utt;
                    }
                    const std::vector<std::string> ws =
                        normalize_words(asr_m.transcribe_stream_window(pcm, cw.t_end, ev.cache.window_s));
                    for (const std::string& w : ws) hyp += (hyp.empty() ? "" : " ") + w;
                    asr_words.push_back(hyp);
                }
                if (dw)
                    std::fprintf(dw, "%s\t%d\t%s\t%.2f\t%s\t%s\n", names[d].c_str(), wins[k], u.id.c_str(), cw.t_end,
                                 gold.c_str(), hyp.c_str());
                for (std::size_t q = 0; q < bank.size(); ++q) {
                    Rec r;
                    r.set = static_cast<int>(d);
                    r.window = wins[k];
                    r.q = static_cast<int>(q);
                    r.label = real_label(bank[q].rule, u, lab, cw.t_end, ev.cache.window_s);
                    recs.push_back(r);
                    apairs.push_back({wins[k], static_cast<int>(q)});
                    tpairs.push_back({static_cast<int>(k), static_cast<int>(q)});
                }
            }
            const double asr_ms = asr ? (now_ms() - t_asr0) / wins.size() : 0;
            const double t0 = now_ms();
            const std::vector<float> za = score_audio(model, builder, proj, ev.cache, qtext, apairs);
            const double t1 = now_ms();
            const std::vector<float> zg = score_transcripts(*tm, gold_words, qtext, tpairs);
            const double t2 = now_ms();
            std::vector<float> zr(recs.size(), std::nanf(""));
            if (asr) zr = score_transcripts(*tm, asr_words, qtext, tpairs);
            // Records hold calibrated logits (each checkpoint's temperature
            // applied), so the metrics below use T = 1.
            for (std::size_t i = 0; i < recs.size(); ++i) {
                recs[i].z[0] = za[i] / T;
                recs[i].z[1] = zg[i] / Tt;
                recs[i].z[2] = zr[i] / Tt;
                if (dq)
                    std::fprintf(dq, "%s\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\n", names[d].c_str(), recs[i].window, recs[i].q,
                                 recs[i].label, recs[i].z[0], recs[i].z[1], recs[i].z[2]);
            }
            std::printf("\n== %s: %zu windows x %zu questions; audio %.1f s, gold text %.1f s%s\n", names[d].c_str(),
                        wins.size(), bank.size(), (t1 - t0) / 1000, (t2 - t1) / 1000,
                        asr ? (", ASR " + std::to_string(static_cast<int>(asr_ms)) + " ms/window").c_str() : "");
            // Agreement with the ceiling, by group and family.
            std::map<std::string, std::vector<const Rec*>> by_group, by_family;
            for (const Rec& r : recs) {
                by_group[bank[static_cast<std::size_t>(r.q)].group()].push_back(&r);
                by_family[bank[static_cast<std::size_t>(r.q)].family].push_back(&r);
            }
            std::printf(" agreement with text Laya on the gold transcript, by group:\n");
            for (const auto& [g, rs] : by_group) print_agree(g, agreement(rs, 1.0f, asr), asr);
            std::printf(" by family:\n");
            for (const auto& [f, rs] : by_family) print_agree(f, agreement(rs, 1.0f, asr), asr);
            std::fflush(stdout);
            all.insert(all.end(), recs.begin(), recs.end());
        }
        if (dq) std::fclose(dq);
        if (dw) std::fclose(dw);

        // Real labels: per family pooled over sets, then per (family, set)
        // and per question.
        std::printf("\n== real labels (AUC of the logit; balanced accuracy at p >= 0.5)\n");
        std::map<std::string, std::vector<const Rec*>> fam, fam_set, per_q;
        for (const Rec& r : all) {
            if (r.label < 0) continue;
            const BankQuestion& q = bank[static_cast<std::size_t>(r.q)];
            fam[q.family + " [" + q.group() + "]"].push_back(&r);
            fam_set[q.family + " @ " + names[static_cast<std::size_t>(r.set)]].push_back(&r);
            per_q[q.family + " #" + std::to_string(r.q)].push_back(&r);
        }
        std::printf(" by family, all sets:\n");
        for (const auto& [k, rs] : fam) print_real(k, real_metrics(rs, 1.0f, asr), asr);
        std::printf(" by family and set:\n");
        for (const auto& [k, rs] : fam_set) print_real(k, real_metrics(rs, 1.0f, asr), asr);
        std::printf(" by question:\n");
        for (const auto& [k, rs] : per_q) print_real(k, real_metrics(rs, 1.0f, asr), asr);
        std::printf(" questions: ");
        for (std::size_t q = 0; q < bank.size(); ++q)
            if (!bank[q].rule.empty()) std::printf("#%zu \"%s\"  ", q, bank[q].text.c_str());
        std::printf("\n");
        return 0;
    } catch (const std::exception& e) {
        die(e.what());
    }
}
