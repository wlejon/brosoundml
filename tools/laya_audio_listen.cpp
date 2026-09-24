// brosoundml_laya_audio_listen — the adapter as a stream: detection timing,
// early-fire ("boop") operating points, false alarms per hour, and per-hop
// latency.
//
// Each utterance (+1 s of trailing silence) is pushed through a LayaListener
// in hop-sized chunks, like a microphone. Per utterance the listener asks: up
// to two keywords that occur in it once (>= 4 letters), a spelling neighbour
// of one of them and `--neg` random vocabulary words that do not occur (from
// the utterance's language), plus speaking / word_end.
//
// Hold-time operating points (threshold x consecutive hops):
//   delay     first gated hop at or after the word's onset, from the aligned
//             word END (negative = fired before the word finished); a miss
//             is no such hop within 1.5 s after the end;
//   FA        rising edges of the gate on absent keywords per keyword-hour.
//
// Boop sweep (single-hop threshold): recall, median fire time after the word
// end, false boops per keyword-hour of speech, the fraction of false boops
// the 150 ms hold (5 hops) would retract ("unboop": the run above threshold
// ends before 5 hops), the fraction of TRUE boops it would retract, and the
// median time the held (confirmed) fire lands after the word end.
//
// --nonspeech: stream non-speech recordings instead, asking --neg random
// keywords; false alarms per keyword-hour of non-speech per operating point
// and the fraction of hops the speaking question says yes.
//
// Then a timing pass: hops with 1 and with 10 keyword questions.
//
// Usage:
//   brosoundml_laya_audio_listen --vocab-align A[+A] --stream ALIGN --proj P
//        [--laya D:/projects/laya] [--temperature 1] [--utts 60] [--neg 4]
//        [--hop-ms 30] [--window 3.0] [--seed 3] [--nonspeech] [--no-timing]

#include "laya_audio_corpus.h"
#include "laya_audio_listener.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <sstream>
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
    float t0 = 0, t1 = 0;  // the occurrence, positives only
    std::vector<float> p;  // per hop
};

// Hops where p >= thr has held for `consec` consecutive hops.
std::vector<char> gate(const std::vector<float>& p, float thr, int consec) {
    std::vector<char> g(p.size(), 0);
    for (std::size_t j = 0, run = 0; j < p.size(); ++j) {
        run = p[j] >= thr ? run + 1 : 0;
        g[j] = run >= static_cast<std::size_t>(consec);
    }
    return g;
}

// Rising edges of p >= thr, each with the length of its run in hops.
std::vector<int> runs_above(const std::vector<float>& p, float thr) {
    std::vector<int> out;
    int run = 0;
    for (float v : p) {
        if (v >= thr) ++run;
        else if (run) {
            out.push_back(run);
            run = 0;
        }
    }
    if (run) out.push_back(run);
    return out;
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

}  // namespace

int main(int argc, char** argv) {
    std::string vocab_align, stream;
    ListenerConfig cfg;
    cfg.temperature = 1.0f;
    int n_utts = 60, n_neg = 4;
    uint32_t seed = 3;
    bool nonspeech = false, timing = true;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(k + " needs a value");
            return argv[++i];
        };
        if (k == "--vocab-align") vocab_align = next();
        else if (k == "--stream") stream = next();
        else if (k == "--proj") cfg.projector = next();
        else if (k == "--laya") cfg.laya_dir = next();
        else if (k == "--encoder-dir") cfg.encoder_dir = next();
        else if (k == "--temperature") cfg.temperature = std::stof(next());
        else if (k == "--utts") n_utts = std::atoi(next().c_str());
        else if (k == "--neg") n_neg = std::atoi(next().c_str());
        else if (k == "--hop-ms") cfg.hop_ms = std::atoi(next().c_str());
        else if (k == "--window") cfg.window_s = std::stof(next());
        else if (k == "--seed") seed = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (k == "--nonspeech") nonspeech = true;
        else if (k == "--no-timing") timing = false;
        else die("unknown argument " + k);
    }
    if (vocab_align.empty() || stream.empty() || cfg.projector.empty()) die("need --vocab-align --stream --proj");

    try {
        bt::init();
        const std::vector<AlignedUtterance> utts = read_alignments(stream);
        VocabSet vocab;
        {
            std::vector<AlignedUtterance> v = read_many(vocab_align);
            v.insert(v.end(), utts.begin(), utts.end());
            vocab.build(v, 2, false);
        }
        LayaListener L;
        L.load(cfg);
        std::mt19937 rng(seed);
        std::vector<int> order(utts.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        // Hold-time operating points: threshold, and hops it must hold for.
        const std::vector<float> thrs = {0.5f, 0.7f, 0.9f, 0.7f, 0.9f, 0.9f};
        const std::vector<int> consec = {1, 1, 1, 5, 5, 10};
        std::vector<std::vector<double>> delays(thrs.size());
        std::vector<int> misses(thrs.size(), 0), fa(thrs.size(), 0);
        // Boop sweep.
        // Thresholds up to 0.995: a checkpoint calibrated sharper (the
        // multilingual one, T = 1) reaches the low false-boop rates only there.
        const std::vector<float> boop = {0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.95f, 0.98f, 0.99f, 0.995f};
        constexpr int kHold = 5;
        std::vector<int> b_hit(boop.size(), 0), b_fb(boop.size(), 0), b_fb_retracted(boop.size(), 0);
        std::vector<int> b_tb(boop.size(), 0), b_tb_retracted(boop.size(), 0);
        std::vector<std::vector<double>> b_fire(boop.size()), b_confirm(boop.size());
        std::vector<double> sp_on, sp_off, hop_ms, enc_ms, laya_ms;
        int n_pos = 0, sp_yes = 0, sp_hops = 0;
        double neg_hours = 0;
        int done = 0;
        for (int ui : order) {
            if (done >= n_utts) break;
            const AlignedUtterance& u = utts[static_cast<std::size_t>(ui)];
            std::vector<Trace> traces;
            Vocab& voc = vocab.for_utterance(u, rng);
            auto absent = [&](const std::string& k) {
                for (const TimedWord& x : u.words)
                    if (x.word == k) return false;
                return true;
            };
            if (!nonspeech) {
                if (u.words.empty() || u.duration_s < 2.0f) continue;
                for (const TimedWord& w : u.words) {
                    if (w.word.size() < 4 || w.t0 < 0 || w.t1 > u.duration_s || traces.size() >= 2) continue;
                    int cnt = 0;
                    for (const TimedWord& x : u.words) cnt += x.word == w.word;
                    bool dup = false;
                    for (const Trace& t : traces) dup |= t.word == w.word;
                    if (cnt == 1 && !dup) traces.push_back({w.word, true, w.t0, w.t1, {}});
                }
                if (traces.empty()) continue;
                for (const std::string& nb : voc.neighbours(traces[0].word)) {
                    if (absent(nb)) {
                        traces.push_back({nb, false, 0, 0, {}});
                        break;
                    }
                }
            }
            for (int k = 0, tries = 0; k < n_neg && tries < 100; ++tries) {
                const std::string& w = voc.sample(rng);
                if (!absent(w)) continue;
                traces.push_back({w, false, 0, 0, {}});
                ++k;
            }
            L.clear_questions();
            for (const Trace& t : traces) L.add_keyword(t.word);
            const int q_sp = L.add_question(question_text(Question::Speaking));
            L.add_question(question_text(Question::WordEnd));

            std::vector<float> pcm = load_audio_16k(u.wav);
            if (!nonspeech) pcm.resize(pcm.size() + 16000, 0.0f);
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
                // Non-speech: the first 3 s still hold the stream's leading
                // silence; count speaking answers once the window is full.
                if (nonspeech && h.t_end >= cfg.window_s) {
                    ++sp_hops;
                    sp_yes += h.p[static_cast<std::size_t>(q_sp)] >= 0.5f;
                }
            }
            const double audio_h = hops.size() * cfg.hop_ms / 3.6e6;
            for (std::size_t q = 0; q < traces.size(); ++q) {
                Trace& t = traces[q];
                for (const HopResult& h : hops) t.p.push_back(h.p[q]);
                if (!t.positive) neg_hours += audio_h;
                else ++n_pos;
                for (std::size_t k = 0; k < thrs.size(); ++k) {
                    const std::vector<char> g = gate(t.p, thrs[k], consec[k]);
                    if (!t.positive) {
                        bool on = false;
                        for (char v : g) {
                            if (v && !on) ++fa[k];
                            on = v != 0;
                        }
                        continue;
                    }
                    bool fired = false;
                    for (std::size_t j = 0; j < hops.size(); ++j) {
                        const double te = hops[j].t_end;
                        if (te < t.t0 || te > t.t1 + 1.5) continue;
                        if (g[j]) {
                            delays[k].push_back(te - t.t1);
                            fired = true;
                            break;
                        }
                    }
                    if (!fired) ++misses[k];
                }
                for (std::size_t k = 0; k < boop.size(); ++k) {
                    if (!t.positive) {
                        for (int r : runs_above(t.p, boop[k])) {
                            ++b_fb[k];
                            b_fb_retracted[k] += r < kHold;
                        }
                        continue;
                    }
                    // First hop in the occurrence's window above threshold; its
                    // run length decides whether the hold would retract it.
                    for (std::size_t j = 0; j < hops.size(); ++j) {
                        const double te = hops[j].t_end;
                        if (te < t.t0 || te > t.t1 + 1.5 || t.p[j] < boop[k]) continue;
                        ++b_hit[k];
                        b_fire[k].push_back(te - t.t1);
                        std::size_t e = j;
                        while (e < hops.size() && t.p[e] >= boop[k]) ++e;
                        ++b_tb[k];
                        if (e - j < static_cast<std::size_t>(kHold)) ++b_tb_retracted[k];
                        break;
                    }
                    const std::vector<char> g = gate(t.p, boop[k], kHold);
                    for (std::size_t j = 0; j < hops.size(); ++j) {
                        const double te = hops[j].t_end;
                        if (te < t.t0 || te > t.t1 + 1.5) continue;
                        if (g[j]) {
                            b_confirm[k].push_back(te - t.t1);
                            break;
                        }
                    }
                }
            }
            if (!nonspeech) {
                float first = 1e9f, last = 0;
                for (const TimedWord& w : u.words) {
                    if (w.t1 <= 0) continue;
                    first = std::min(first, std::max(0.0f, w.t0));
                    last = std::max(last, std::min(w.t1, u.duration_s));
                }
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
            if (done % 20 == 0) std::fprintf(stderr, "%d utterances streamed\n", done);
        }

        std::printf("streamed %d %s utterances (%s), %zu hops of %d ms, window %.1f s, %d positive keyword "
                    "occurrences, %.2f absent-keyword hours\n",
                    done, nonspeech ? "non-speech" : "speech", stream.c_str(), hop_ms.size(), cfg.hop_ms,
                    cfg.window_s, n_pos, neg_hours);
        std::printf("per hop (%d-%d questions): total p50 %.1f p95 %.1f ms | encode+project p50 %.1f | laya p50 %.1f\n",
                    n_neg + 2, n_neg + 5, pct(hop_ms, 0.5), pct(hop_ms, 0.95), pct(enc_ms, 0.5), pct(laya_ms, 0.5));
        for (std::size_t k = 0; k < thrs.size(); ++k) {
            if (nonspeech) {
                std::printf("thr %.1f x %2d hops: false alarms %.1f / keyword-hour of non-speech\n", thrs[k],
                            consec[k], fa[k] / std::max(1e-9, neg_hours));
                continue;
            }
            std::printf("thr %.1f x %2d hops: recall %.3f, delay after word end p10 %+.2f p50 %+.2f p90 %+.2f s, "
                        "false alarms %.1f / keyword-hour\n",
                        thrs[k], consec[k], 1.0 - static_cast<double>(misses[k]) / std::max(1, n_pos),
                        pct(delays[k], 0.1), pct(delays[k], 0.5), pct(delays[k], 0.9),
                        fa[k] / std::max(1e-9, neg_hours));
        }
        std::printf("\nboop sweep (single hop), hold = %d hops (%d ms):\n", kHold, kHold * cfg.hop_ms);
        std::printf("%6s %8s %12s %14s %14s %14s %14s\n", "thr", "recall", "fire p50 (s)", "false boops/kh",
                    "false retract", "true retract", "confirm p50 (s)");
        for (std::size_t k = 0; k < boop.size(); ++k) {
            std::printf("%6.3f %8.3f %+12.2f %14.1f %14.3f %14.3f %+14.2f\n", boop[k],
                        double(b_hit[k]) / std::max(1, n_pos), pct(b_fire[k], 0.5),
                        b_fb[k] / std::max(1e-9, neg_hours), double(b_fb_retracted[k]) / std::max(1, b_fb[k]),
                        double(b_tb_retracted[k]) / std::max(1, b_tb[k]), pct(b_confirm[k], 0.5));
        }
        if (nonspeech) {
            std::printf("speaking (p >= 0.5) on %.4f of full-window non-speech hops\n",
                        double(sp_yes) / std::max(1, sp_hops));
        } else {
            std::printf("speaking (p >= 0.5): onset delay p50 %.2f p90 %.2f s, release after last word p50 %.2f "
                        "p90 %.2f s\n",
                        pct(sp_on, 0.5), pct(sp_on, 0.9), pct(sp_off, 0.5), pct(sp_off, 0.9));
        }
        std::fflush(stdout);

        if (timing) {
            // Timing: 1 and 10 keyword questions on live audio.
            const AlignedUtterance& u = utts[static_cast<std::size_t>(order.front())];
            std::vector<float> pcm = load_audio_16k(u.wav);
            Vocab& voc = vocab.get("en");
            for (int nq : {1, 10}) {
                L.clear_questions();
                for (int k = 0; k < nq; ++k) L.add_keyword(voc.sample(rng));
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
                std::printf("timing, %2d keyword question(s): hop p50 %.1f p95 %.1f ms (encode+project %.1f, laya "
                            "%.1f) over %zu hops\n",
                            nq, pct(tot, 0.5), pct(tot, 0.95), pct(enc, 0.5), pct(lay, 0.5), tot.size());
            }
        }
        return 0;
    } catch (const std::exception& e) {
        die(e.what());
    }
}
