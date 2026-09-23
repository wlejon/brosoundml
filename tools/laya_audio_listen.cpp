// brosoundml_laya_audio_listen — the adapter as a stream: per-hop latency,
// detection delay after a keyword ends, and false alarms per hour.
//
// Each dev utterance (+1 s of trailing silence) is pushed through a
// LayaListener in hop-sized chunks, like a microphone. Per utterance the
// listener asks: up to two keywords that occur in it once (>= 4 letters), a
// spelling neighbour of one of them and `--neg` random vocabulary words that
// do not occur, plus speaking / word_end. From the per-hop probabilities:
//
//   keyword delay   first hop at or after the word's onset with p >= thr,
//                   measured from the aligned word END (negative = fired
//                   before the word finished);
//   miss            no such hop within 1.5 s after the word end;
//   false alarms    rising edges of p >= thr on absent keywords, per hour of
//                   (keyword x audio);
//   speaking        onset delay after the first word starts, release delay
//                   after the last word ends.
//
// Then a timing pass: hops with 1 and with 10 keyword questions.
//
// Usage:
//   brosoundml_laya_audio_listen --align-train A --align-dev A --proj P
//        [--utts 60] [--neg 4] [--hop-ms 30] [--window 3.0] [--seed 3]

#include "laya_audio_listener.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace bt = brotensor;
using namespace laya_audio;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "laya_audio_listen: %s\n", msg.c_str());
    std::exit(2);
}

double pct(std::vector<double> v, double q) {
    if (v.empty()) return std::nan("");
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * (v.size() - 1) + 0.5))];
}

struct Trace {
    std::string word;
    bool positive = false;
    float t0 = 0, t1 = 0;       // the occurrence, positives only
    std::vector<float> p;       // per hop
};

}  // namespace

int main(int argc, char** argv) {
    std::string align_train, align_dev;
    ListenerConfig cfg;
    int n_utts = 60, n_neg = 4;
    uint32_t seed = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(k + " needs a value");
            return argv[++i];
        };
        if (k == "--align-train") align_train = next();
        else if (k == "--align-dev") align_dev = next();
        else if (k == "--proj") cfg.projector = next();
        else if (k == "--laya") cfg.laya_dir = next();
        else if (k == "--encoder-dir") cfg.encoder_dir = next();
        else if (k == "--utts") n_utts = std::atoi(next().c_str());
        else if (k == "--neg") n_neg = std::atoi(next().c_str());
        else if (k == "--hop-ms") cfg.hop_ms = std::atoi(next().c_str());
        else if (k == "--window") cfg.window_s = std::stof(next());
        else if (k == "--seed") seed = static_cast<uint32_t>(std::atoi(next().c_str()));
        else die("unknown argument " + k);
    }
    if (align_train.empty() || align_dev.empty() || cfg.projector.empty())
        die("need --align-train --align-dev --proj");

    try {
        bt::init();
        const std::vector<AlignedUtterance> tr_utts = read_alignments(align_train);
        const std::vector<AlignedUtterance> dv_utts = read_alignments(align_dev);
        Vocab vocab;
        {
            std::vector<AlignedUtterance> both = tr_utts;
            both.insert(both.end(), dv_utts.begin(), dv_utts.end());
            vocab.build(both, 2, false);
        }
        LayaListener L;
        L.load(cfg);
        std::mt19937 rng(seed);
        std::vector<int> order(dv_utts.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        const std::vector<float> thrs = {0.5f, 0.7f, 0.9f};
        std::vector<std::vector<double>> delays(thrs.size());
        std::vector<int> misses(thrs.size(), 0), fa(thrs.size(), 0), early(thrs.size(), 0);
        std::vector<double> sp_on, sp_off, hop_ms, enc_ms, laya_ms;
        int n_pos = 0;
        double neg_hours = 0;
        int done = 0;
        for (int ui : order) {
            if (done >= n_utts) break;
            const AlignedUtterance& u = dv_utts[static_cast<std::size_t>(ui)];
            if (u.words.empty() || u.duration_s < 2.0f) continue;
            std::vector<Trace> traces;
            for (const TimedWord& w : u.words) {
                if (w.word.size() < 4 || w.t0 < 0 || traces.size() >= 2) continue;
                int cnt = 0;
                for (const TimedWord& x : u.words) cnt += x.word == w.word;
                bool dup = false;
                for (const Trace& t : traces) dup |= t.word == w.word;
                if (cnt == 1 && !dup) traces.push_back({w.word, true, w.t0, w.t1, {}});
            }
            if (traces.empty()) continue;
            auto absent = [&](const std::string& k) {
                for (const TimedWord& x : u.words)
                    if (x.word == k) return false;
                return true;
            };
            for (const std::string& nb : vocab.neighbours(traces[0].word)) {
                if (absent(nb)) {
                    traces.push_back({nb, false, 0, 0, {}});
                    break;
                }
            }
            for (int k = 0, tries = 0; k < n_neg && tries < 100; ++tries) {
                const std::string& w = vocab.sample(rng);
                if (!absent(w)) continue;
                traces.push_back({w, false, 0, 0, {}});
                ++k;
            }
            L.clear_questions();
            for (const Trace& t : traces) L.add_keyword(t.word);
            const int q_sp = L.add_question(question_text(Question::Speaking));
            L.add_question(question_text(Question::WordEnd));

            std::vector<float> pcm = load_audio_16k(u.wav);
            pcm.resize(pcm.size() + 16000, 0.0f);
            L.reset();
            std::vector<HopResult> hops;
            const int chunk = cfg.hop_ms * 16;
            for (std::size_t s = 0; s < pcm.size(); s += static_cast<std::size_t>(chunk)) {
                const int n = static_cast<int>(std::min<std::size_t>(chunk, pcm.size() - s));
                for (HopResult& h : L.feed(pcm.data() + s, n)) hops.push_back(std::move(h));
            }
            for (const HopResult& h : hops) {
                hop_ms.push_back(h.total_ms);
                enc_ms.push_back(h.encode_ms);
                laya_ms.push_back(h.laya_ms);
            }
            const double audio_h = hops.size() * cfg.hop_ms / 3.6e6;
            for (std::size_t q = 0; q < traces.size(); ++q) {
                Trace& t = traces[q];
                for (const HopResult& h : hops) t.p.push_back(h.p[q]);
                if (!t.positive) neg_hours += audio_h;
                else ++n_pos;
                for (std::size_t k = 0; k < thrs.size(); ++k) {
                    if (!t.positive) {
                        bool on = false;
                        for (float v : t.p) {
                            if (v >= thrs[k] && !on) ++fa[k];
                            on = v >= thrs[k];
                        }
                        continue;
                    }
                    bool fired = false, before = false;
                    for (std::size_t j = 0; j < hops.size(); ++j) {
                        const double te = hops[j].t_end;
                        if (t.p[j] >= thrs[k] && te < t.t0) before = true;
                        if (te < t.t0 || te > t.t1 + 1.5) continue;
                        if (t.p[j] >= thrs[k]) {
                            delays[k].push_back(te - t.t1);
                            fired = true;
                            break;
                        }
                    }
                    if (!fired) ++misses[k];
                    if (before) ++early[k];
                }
            }
            // Speaking onset / release.
            {
                const float first = u.words.front().t0, last = u.words.back().t1;
                for (const HopResult& h : hops)
                    if (h.t_end >= first && h.p[static_cast<std::size_t>(q_sp)] >= 0.5f) {
                        sp_on.push_back(h.t_end - first);
                        break;
                    }
                for (const HopResult& h : hops)
                    if (h.t_end >= last && h.p[static_cast<std::size_t>(q_sp)] < 0.5f) {
                        sp_off.push_back(h.t_end - last);
                        break;
                    }
            }
            ++done;
            if (done % 10 == 0) std::fprintf(stderr, "%d utterances streamed\n", done);
        }

        std::printf("streamed %d utterances, %zu hops of %d ms, window %.1f s, %d positive keyword occurrences\n",
                    done, hop_ms.size(), cfg.hop_ms, cfg.window_s, n_pos);
        std::printf("per hop (%d-%d questions): total p50 %.1f p95 %.1f ms | encode+project p50 %.1f | laya p50 %.1f\n",
                    n_neg + 3, n_neg + 5, pct(hop_ms, 0.5), pct(hop_ms, 0.95), pct(enc_ms, 0.5), pct(laya_ms, 0.5));
        for (std::size_t k = 0; k < thrs.size(); ++k) {
            std::printf("thr %.1f: recall %.3f, delay after word end p10 %+.2f p50 %+.2f p90 %+.2f s, "
                        "fired before onset %d, false alarms %.1f / keyword-hour\n",
                        thrs[k], 1.0 - static_cast<double>(misses[k]) / std::max(1, n_pos), pct(delays[k], 0.1),
                        pct(delays[k], 0.5), pct(delays[k], 0.9), early[k], fa[k] / std::max(1e-9, neg_hours));
        }
        std::printf("speaking (p >= 0.5): onset delay p50 %.2f p90 %.2f s, release after last word p50 %.2f p90 %.2f s\n",
                    pct(sp_on, 0.5), pct(sp_on, 0.9), pct(sp_off, 0.5), pct(sp_off, 0.9));

        // Timing: 1 and 10 keyword questions on live audio.
        const AlignedUtterance& u = dv_utts[static_cast<std::size_t>(order.front())];
        std::vector<float> pcm = load_audio_16k(u.wav);
        for (int nq : {1, 10}) {
            L.clear_questions();
            for (int k = 0; k < nq; ++k) L.add_keyword(vocab.sample(rng));
            L.reset();
            std::vector<double> tot, enc, lay;
            for (int rep = 0; rep < 3; ++rep) {
                for (std::size_t s = 0; s + 480 <= pcm.size(); s += 480) {
                    for (const HopResult& h : L.feed(pcm.data() + s, 480)) {
                        tot.push_back(h.total_ms);
                        enc.push_back(h.encode_ms);
                        lay.push_back(h.laya_ms);
                    }
                }
            }
            std::printf("timing, %2d keyword question(s): hop p50 %.1f p95 %.1f ms (encode+project %.1f, laya %.1f) "
                        "over %zu hops\n",
                        nq, pct(tot, 0.5), pct(tot, 0.95), pct(enc, 0.5), pct(lay, 0.5), tot.size());
        }
        return 0;
    } catch (const std::exception& e) {
        die(e.what());
    }
}
