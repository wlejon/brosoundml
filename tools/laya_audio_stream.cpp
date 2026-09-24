// brosoundml_laya_audio_stream — free-form questions on a live stream.
//
//   meeting  Stream a stretch of an AMI meeting (single distant mic) hop by
//            hop with a panel of questions (one per line of --panel). Writes
//            a per-hop trace (TSV) of every answer, and every --text-every
//            hops the ceiling (text Laya on the window's gold words) for
//            the same questions. Summary: per-hop latency, and per question
//            the yes-rate, agreement with the ceiling along the stream, and
//            the longest yes-runs with the words spoken there.
//   utts     Stream labelled utterances (SLURP / Timers and Such test), each
//            followed by 0.6 s of silence, asking one question per family
//            whose real-label rule applies (laya_audio_questions.tsv). Per
//            (utterance, question): the answer at the end of speech, the max
//            over the stream, and the first hop that crosses --thr (time from
//            the end of speech). Baselines: text Laya on the gold words and on
//            Parakeet's transcript of the stream so far, the latter also on
//            prefixes ending 0.9 / 0.6 / 0.3 s before and 0 / 0.3 s after the
//            end of speech (when the ASR route could first decide), with its
//            compute cost.
//
// Usage:
//   brosoundml_laya_audio_stream meeting --proj P --laya DIR --temperature T --panel panel.txt
//        --meeting ES2004a --ami-align A[+A] [--from 0] [--to 600] [--text-every 10] --trace out.tsv
//   brosoundml_laya_audio_stream utts --proj P --laya DIR --temperature T --bank Q.tsv
//        --align A --labels L [--limit 300] [--thr 0.5] [--asr] [--seed 1]

#include "laya_audio_baselines.h"
#include "laya_audio_listener.h"
#include "laya_audio_questions.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace bt = brotensor;
using namespace laya_audio;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "laya_audio_stream: %s\n", msg.c_str());
    std::exit(2);
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double pct(std::vector<double> v, double q) {
    if (v.empty()) return std::nan("");
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * (v.size() - 1) + 0.5))];
}

std::vector<AlignedUtterance> read_many(const std::string& plus_list) {
    std::vector<AlignedUtterance> out;
    std::istringstream is(plus_list);
    std::string p;
    while (std::getline(is, p, '+'))
        if (!p.empty())
            for (AlignedUtterance& u : read_alignments(p)) out.push_back(std::move(u));
    return out;
}

std::vector<HopResult> stream(LayaListener& L, const std::vector<float>& pcm) {
    L.reset();
    std::vector<HopResult> hops;
    const int chunk = L.config().hop_ms * 16;
    for (std::size_t s = 0; s < pcm.size(); s += static_cast<std::size_t>(chunk)) {
        const int n = static_cast<int>(std::min<std::size_t>(chunk, pcm.size() - s));
        for (HopResult& h : L.feed(pcm.data() + s, n)) hops.push_back(std::move(h));
    }
    return hops;
}

struct Opts {
    ListenerConfig cfg;
    std::string panel, meeting, ami_align, trace, bank, align, labels;
    float from = 0, to = 600, thr = 0.5f;
    int text_every = 10, limit = 300;
    bool asr = false;
    uint32_t seed = 1;
};

// ---------------------------------------------------------------- meeting

int run_meeting(const Opts& o) {
    std::vector<std::string> panel;
    {
        std::ifstream f(o.panel);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty() && line[0] != '#') panel.push_back(line);
        }
    }
    if (panel.empty()) die("empty panel " + o.panel);
    // The meeting's words on the streamed stretch's clock (chunks overlap at
    // their edges, so words are de-duplicated).
    AlignedUtterance m;
    m.id = o.meeting;
    std::string audio;
    std::set<std::pair<int, std::string>> seen;
    for (const AlignedUtterance& c : read_many(o.ami_align)) {
        if (c.speaker != o.meeting) continue;
        const std::size_t at = c.wav.rfind('@');
        const float c0 = std::stof(c.wav.substr(at + 1));
        audio = c.wav.substr(0, at);
        for (const TimedWord& w : c.words) {
            const float t0 = w.t0 + c0 - o.from, t1 = w.t1 + c0 - o.from;
            if (t1 <= 0 || t0 >= o.to - o.from) continue;
            if (!seen.insert({static_cast<int>(std::lround(t0 * 100)), w.word}).second) continue;
            m.words.push_back({w.word, t0, t1, true});
        }
    }
    if (audio.empty()) die("meeting " + o.meeting + " not in " + o.ami_align);
    char spec[64];
    std::snprintf(spec, sizeof(spec), "@%.2f:%.2f", o.from, o.to);
    const std::vector<float> pcm = load_audio_16k(audio + spec);
    m.duration_s = static_cast<float>(pcm.size()) / 16000.0f;

    LayaListener L;
    L.load(o.cfg);
    for (const std::string& q : panel) L.add_question(q);
    const std::vector<HopResult> hops = stream(L, pcm);

    // Ceiling every text_every hops.
    std::vector<std::string> words;
    std::vector<std::pair<int, int>> pairs;
    std::vector<int> text_hop;
    for (std::size_t h = 0; h < hops.size(); h += static_cast<std::size_t>(o.text_every)) {
        const std::string w = window_words(m, static_cast<float>(hops[h].t_end), o.cfg.window_s);
        text_hop.push_back(static_cast<int>(h));
        words.push_back(w);
        for (std::size_t q = 0; q < panel.size(); ++q)
            pairs.push_back({static_cast<int>(words.size()) - 1, static_cast<int>(q)});
    }
    const std::vector<float> zg = score_transcripts(L.model(), words, panel, pairs);

    std::FILE* f = std::fopen(o.trace.c_str(), "wb");
    if (!f) die("cannot write " + o.trace);
    std::fprintf(f, "t_end\thop_ms");
    for (std::size_t q = 0; q < panel.size(); ++q) std::fprintf(f, "\taudio_q%zu", q);
    for (std::size_t q = 0; q < panel.size(); ++q) std::fprintf(f, "\tgold_q%zu", q);
    std::fprintf(f, "\tgold_words\n");
    for (std::size_t q = 0; q < panel.size(); ++q) std::fprintf(f, "#q%zu\t%s\n", q, panel[q].c_str());
    for (std::size_t h = 0, t = 0; h < hops.size(); ++h) {
        std::fprintf(f, "%.2f\t%.2f", hops[h].t_end, hops[h].total_ms);
        for (float p : hops[h].p) std::fprintf(f, "\t%.3f", p);
        const bool has_text = t < text_hop.size() && text_hop[t] == static_cast<int>(h);
        for (std::size_t q = 0; q < panel.size(); ++q) {
            if (has_text) std::fprintf(f, "\t%.3f", sigmoid_t(zg[t * panel.size() + q], o.cfg.temperature));
            else std::fprintf(f, "\t");
        }
        std::fprintf(f, "\t%s\n", has_text ? words[t].c_str() : "");
        if (has_text) ++t;
    }
    std::fclose(f);

    std::vector<double> ms, enc, lay;
    for (const HopResult& h : hops) {
        ms.push_back(h.total_ms);
        enc.push_back(h.encode_ms);
        lay.push_back(h.laya_ms);
    }
    std::printf("meeting %s %.0f-%.0f s: %zu hops of %d ms, %zu questions per hop\n", o.meeting.c_str(), o.from, o.to,
                hops.size(), o.cfg.hop_ms, panel.size());
    std::printf("per hop: total p50 %.1f p95 %.1f max %.1f ms (encode+project p50 %.1f, laya p50 %.1f)\n",
                pct(ms, 0.5), pct(ms, 0.95), pct(ms, 1.0), pct(enc, 0.5), pct(lay, 0.5));
    for (std::size_t q = 0; q < panel.size(); ++q) {
        std::vector<float> pa, pg;
        std::vector<int> gy;
        int yes = 0;
        for (const HopResult& h : hops) yes += h.p[q] >= 0.5f;
        for (std::size_t t = 0; t < text_hop.size(); ++t) {
            pa.push_back(hops[static_cast<std::size_t>(text_hop[t])].p[q]);
            const float g = sigmoid_t(zg[t * panel.size() + q], o.cfg.temperature);
            pg.push_back(g);
            gy.push_back(g >= 0.5f);
        }
        int gyes = 0;
        for (int v : gy) gyes += v;
        std::printf("\nq%zu \"%s\"\n  audio yes on %.1f %% of hops; ceiling yes on %.1f %% of sampled hops; along "
                    "the stream r %.3f, AUC vs ceiling %.3f\n",
                    q, panel[q].c_str(), 100.0 * yes / hops.size(), 100.0 * gyes / gy.size(), pearson(pa, pg),
                    auc(pa, gy));
        // The three longest yes-runs, with the words heard at their peak.
        struct Run {
            std::size_t a, b, peak;
        };
        std::vector<Run> runs;
        for (std::size_t h = 0; h < hops.size();) {
            if (hops[h].p[q] < 0.5f) {
                ++h;
                continue;
            }
            Run r{h, h, h};
            while (r.b < hops.size() && hops[r.b].p[q] >= 0.5f) {
                if (hops[r.b].p[q] > hops[r.peak].p[q]) r.peak = r.b;
                ++r.b;
            }
            runs.push_back(r);
            h = r.b;
        }
        std::sort(runs.begin(), runs.end(), [](const Run& x, const Run& y) { return x.b - x.a > y.b - y.a; });
        for (std::size_t k = 0; k < std::min<std::size_t>(3, runs.size()); ++k) {
            const Run& r = runs[k];
            const float te = static_cast<float>(hops[r.peak].t_end);
            std::printf("  %.1f-%.1f s (peak %.2f): \"%s\"\n", hops[r.a].t_end, hops[r.b - 1].t_end, hops[r.peak].p[q],
                        window_words(m, te, o.cfg.window_s).c_str());
        }
    }
    return 0;
}

// ---------------------------------------------------------------- utts

int run_utts(const Opts& o) {
    const std::vector<AlignedUtterance> utts = read_alignments(o.align);
    LabelSet labels;
    labels.load(o.labels);
    // One question (the first phrasing) per family with a tag rule that applies.
    std::vector<BankQuestion> qs;
    std::set<std::string> fams;
    for (const BankQuestion& q : load_bank(o.bank)) {
        if (q.rule.rfind("tag:", 0) != 0 || fams.count(q.family)) continue;
        bool applies = false;
        std::istringstream is(q.rule.substr(4));
        std::string p;
        while (std::getline(is, p, '|')) {
            const std::size_t eq = p.find('=');
            if (eq != std::string::npos && labels.has_namespace(p.substr(0, eq))) applies = true;
        }
        if (!applies) continue;
        fams.insert(q.family);
        qs.push_back(q);
    }
    std::vector<std::string> qtext;
    for (const BankQuestion& q : qs) qtext.push_back(q.text);
    std::fprintf(stderr, "%zu questions: ", qs.size());
    for (const BankQuestion& q : qs) std::fprintf(stderr, "%s ", q.family.c_str());
    std::fprintf(stderr, "\n");

    LayaListener L;
    L.load(o.cfg);
    for (const std::string& q : qtext) L.add_question(q);
    AsrBaseline asr_m;
    if (o.asr) asr_m.load("weights/parakeet/0.6b-v3");
    const float T = o.cfg.temperature;
    const std::vector<float> kPrefix = {-0.9f, -0.6f, -0.3f, 0.0f, 0.3f};

    std::vector<int> order(utts.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::mt19937 rng(o.seed);
    std::shuffle(order.begin(), order.end(), rng);
    // Per (utterance, question) records.
    struct Rec {
        int q, label;
        float end_audio, max_audio, gold, asr_end;
        float first_cross;                // s after speech end, NaN = never
        std::vector<float> asr_prefix;    // p at each kPrefix point
    };
    std::vector<Rec> recs;
    std::vector<double> hop_ms, asr_ms, text_ms;
    int done = 0;
    for (int ui : order) {
        if (done >= o.limit) break;
        const AlignedUtterance& u = utts[static_cast<std::size_t>(ui)];
        float s_end = -1;
        for (const TimedWord& w : u.words) s_end = std::max(s_end, w.t1);
        if (s_end <= 0) continue;
        s_end = std::min(s_end, u.duration_s);
        std::vector<float> pcm = load_audio_16k(u.wav);
        pcm.resize(pcm.size() + 9600, 0.0f);
        const std::vector<HopResult> hops = stream(L, pcm);
        for (const HopResult& h : hops) hop_ms.push_back(h.total_ms);
        const float t_label = s_end + 0.3f;
        // Text baselines: gold words, ASR at the end and on prefixes.
        std::vector<std::string> states = {window_words(u, t_label, o.cfg.window_s)};
        if (o.asr) {
            for (float dt : kPrefix) {
                const float te = std::max(0.3f, s_end + dt);
                std::vector<float> pre(pcm.begin(), pcm.begin() + std::min<std::size_t>(pcm.size(), static_cast<std::size_t>(te * 16000)));
                const double t0 = now_ms();
                std::string hyp;
                for (const std::string& w : normalize_words(asr_m.transcribe(pre))) hyp += (hyp.empty() ? "" : " ") + w;
                asr_ms.push_back(now_ms() - t0);
                states.push_back(hyp);
            }
        }
        std::vector<std::pair<int, int>> pairs;
        for (std::size_t s = 0; s < states.size(); ++s)
            for (std::size_t q = 0; q < qs.size(); ++q) pairs.push_back({static_cast<int>(s), static_cast<int>(q)});
        const double t1 = now_ms();
        const std::vector<float> z = score_transcripts(L.model(), states, qtext, pairs);
        text_ms.push_back((now_ms() - t1) / states.size());
        for (std::size_t q = 0; q < qs.size(); ++q) {
            Rec r;
            r.q = static_cast<int>(q);
            r.label = real_label(qs[q].rule, u, &labels, t_label, o.cfg.window_s);
            if (r.label < 0) continue;
            r.end_audio = 0;
            r.max_audio = 0;
            r.first_cross = std::nanf("");
            double best_dt = 1e9;
            for (const HopResult& h : hops) {
                if (h.t_end > s_end + 0.6) break;
                r.max_audio = std::max(r.max_audio, h.p[q]);
                if (std::isnan(r.first_cross) && h.p[q] >= o.thr) r.first_cross = static_cast<float>(h.t_end - s_end);
                if (std::fabs(h.t_end - t_label) < best_dt) {
                    best_dt = std::fabs(h.t_end - t_label);
                    r.end_audio = h.p[q];
                }
            }
            r.gold = sigmoid_t(z[q], T);
            r.asr_end = std::nanf("");
            if (o.asr) {
                for (std::size_t k = 0; k < kPrefix.size(); ++k)
                    r.asr_prefix.push_back(sigmoid_t(z[(k + 1) * qs.size() + q], T));
                r.asr_end = r.asr_prefix[4];  // +0.3 s, the same point as end_audio
            }
            recs.push_back(r);
        }
        if (++done % 50 == 0) std::fprintf(stderr, "%d utterances streamed\n", done);
    }

    std::printf("streamed %d utterances of %s, %zu questions per hop, thr %.2f\n", done, o.align.c_str(), qs.size(), o.thr);
    std::printf("per hop: p50 %.1f p95 %.1f ms", pct(hop_ms, 0.5), pct(hop_ms, 0.95));
    if (o.asr) std::printf(" | ASR (Parakeet, utterance so far) p50 %.0f ms per call; text Laya %.1f ms per transcript",
                           pct(asr_ms, 0.5), pct(text_ms, 0.5));
    std::printf("\n");
    auto report = [&](const std::string& name, const std::vector<const Rec*>& rs) {
        std::vector<int> l;
        std::vector<float> s[5];
        std::vector<double> fc;
        int pos = 0, neg_fire = 0, neg = 0, early[5] = {0, 0, 0, 0, 0};
        for (const Rec* r : rs) {
            l.push_back(r->label);
            s[0].push_back(r->end_audio);
            s[1].push_back(r->max_audio);
            s[2].push_back(r->gold);
            s[3].push_back(o.asr ? r->asr_end : 0.f);
            if (r->label) {
                ++pos;
                if (!std::isnan(r->first_cross)) fc.push_back(r->first_cross);
                for (std::size_t k = 0; k < r->asr_prefix.size(); ++k) early[k] += r->asr_prefix[k] >= o.thr;
            } else {
                ++neg;
                neg_fire += !std::isnan(r->first_cross);
            }
        }
        auto bacc = [&](const std::vector<float>& v) {
            double tp = 0, tn = 0;
            for (std::size_t i = 0; i < l.size(); ++i) {
                tp += l[i] && v[i] >= o.thr;
                tn += !l[i] && v[i] < o.thr;
            }
            return 0.5 * tp / std::max(1, pos) + 0.5 * tn / std::max(1, neg);
        };
        std::printf("  %-18s pos %4d neg %5d | audio end AUC %.3f bacc %.3f | audio stream-max AUC %.3f bacc %.3f | "
                    "gold AUC %.3f bacc %.3f",
                    name.c_str(), pos, neg, auc(s[0], l), bacc(s[0]), auc(s[1], l), bacc(s[1]), auc(s[2], l), bacc(s[2]));
        if (o.asr) std::printf(" | asr AUC %.3f bacc %.3f", auc(s[3], l), bacc(s[3]));
        std::printf("\n     audio first >= thr on %.3f of positives, at speech end %+.2f / %+.2f / %+.2f s (p10/p50/p90); "
                    "fires on %.3f of negatives",
                    double(fc.size()) / std::max(1, pos), pct(fc, 0.1), pct(fc, 0.5), pct(fc, 0.9),
                    double(neg_fire) / std::max(1, neg));
        if (o.asr) {
            std::printf("\n     asr->text yes on positives with the prefix ending at");
            for (std::size_t k = 0; k < kPrefix.size(); ++k)
                std::printf(" %+.1f s: %.3f", kPrefix[k], double(early[k]) / std::max(1, pos));
        }
        std::printf("\n");
    };
    std::map<std::string, std::vector<const Rec*>> by_family;
    std::vector<const Rec*> all;
    for (const Rec& r : recs) {
        by_family[qs[static_cast<std::size_t>(r.q)].family + " [" + qs[static_cast<std::size_t>(r.q)].group() + "]"].push_back(&r);
        all.push_back(&r);
    }
    for (const auto& [k, rs] : by_family) report(k, rs);
    report("ALL", all);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) die("usage: brosoundml_laya_audio_stream meeting|utts ...");
    const std::string mode = argv[1];
    Opts o;
    o.cfg.laya_dir = "D:/projects/laya/multilingual";
    o.cfg.temperature = 1.0f;
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(k + " needs a value");
            return argv[++i];
        };
        if (k == "--proj") o.cfg.projector = next();
        else if (k == "--laya") o.cfg.laya_dir = next();
        else if (k == "--temperature") o.cfg.temperature = std::stof(next());
        else if (k == "--window") o.cfg.window_s = std::stof(next());
        else if (k == "--panel") o.panel = next();
        else if (k == "--meeting") o.meeting = next();
        else if (k == "--ami-align") o.ami_align = next();
        else if (k == "--from") o.from = std::stof(next());
        else if (k == "--to") o.to = std::stof(next());
        else if (k == "--text-every") o.text_every = std::max(1, std::atoi(next().c_str()));
        else if (k == "--trace") o.trace = next();
        else if (k == "--bank") o.bank = next();
        else if (k == "--align") o.align = next();
        else if (k == "--labels") o.labels = next();
        else if (k == "--limit") o.limit = std::atoi(next().c_str());
        else if (k == "--thr") o.thr = std::stof(next());
        else if (k == "--asr") o.asr = true;
        else if (k == "--seed") o.seed = static_cast<uint32_t>(std::atoi(next().c_str()));
        else die("unknown argument " + k);
    }
    if (o.cfg.projector.empty()) die("need --proj");
    try {
        bt::init();
        if (mode == "meeting") {
            if (o.panel.empty() || o.meeting.empty() || o.ami_align.empty() || o.trace.empty())
                die("meeting needs --panel --meeting --ami-align --trace");
            return run_meeting(o);
        }
        if (mode == "utts") {
            if (o.bank.empty() || o.align.empty() || o.labels.empty()) die("utts needs --bank --align --labels");
            return run_utts(o);
        }
        die("unknown mode " + mode);
    } catch (const std::exception& e) {
        die(e.what());
    }
}
