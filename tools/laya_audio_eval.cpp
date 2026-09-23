// brosoundml_laya_audio_eval — held-out evaluation of the Laya audio adapter
// against its alternatives, on the same dev windows and questions.
//
// Dev windows come from speakers never seen in training (LibriTTS-R
// dev-clean); the probe set is rebuilt exactly as the train tool builds its
// dev set (same seed). Methods, each scored per probe:
//
//   laya_audio   projected AuT latents in Laya's state span (the adapter)
//   asr_match    Parakeet transcript of the window contains the keyword (0/1)
//   asr_laya     text Laya, same question, state = Parakeet transcript
//   oracle_laya  text Laya, same question, state = the true words in the window
//   phoneme      open-vocabulary phoneme spotter, g2p-enrolled keyword
//   probe        dedicated MLP head on the last two latent frames (speaking /
//                word_end; from the train tool's --probe mode)
//   energy       RMS level of the last 0.3 s (speaking)
//
// Reported: AUC per question and keyword category (seen / unseen keywords,
// hard / random negatives), TPR at 1 % and 5 % FPR, and per-window cost.
//
// Usage:
//   brosoundml_laya_audio_eval --align-train A --align-dev A --cache-dev C
//        --proj proj.mlp [--probe-head head.mlp] [--laya D:/projects/laya]
//        [--eval-windows 1500] [--seed 1] [--no-asr] [--no-phoneme]
//        [--parakeet-dir weights/parakeet/0.6b-v3]
//        [--phoneme-weights weights/phoneme/english.bpm]
//        [--data-dir D:/projects/brosoundml-data] [--kokoro-dir weights/kokoro]

#include "laya_audio_baselines.h"
#include "laya_audio_task.h"

#include "brolm/laya.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace bt = brotensor;
using namespace laya_audio;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "laya_audio_eval: %s\n", msg.c_str());
    std::exit(2);
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// TPR at the largest threshold whose FPR stays <= max_fpr.
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

struct Method {
    std::string name;
    std::vector<float> score;  // NaN = not applicable
};

void report(const std::vector<Probe>& probes, const std::vector<Method>& methods) {
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
        {"keyword vs random neg", [&](const Probe& p) { return kw(p) && (p.label == 1 || !p.hard_neg); }},
        {"speaking", [](const Probe& p) { return p.q == Question::Speaking; }},
        {"word_end", [](const Probe& p) { return p.q == Question::WordEnd; }},
    };
    std::printf("\n%-24s %-12s %8s %8s %8s %7s %7s\n", "category", "method", "AUC", "TPR@1%", "TPR@5%", "pos", "neg");
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
            std::printf("%-24s %-12s %8.4f %8.4f %8.4f %7d %7d\n", c.name, m.name.c_str(), auc(s, l),
                        tpr_at_fpr(s, l, 0.01), tpr_at_fpr(s, l, 0.05), pos, static_cast<int>(l.size()) - pos);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string align_train, align_dev, cache_dev, proj_path, probe_path, laya = "D:/projects/laya";
    std::string parakeet_dir = "weights/parakeet/0.6b-v3", phoneme_w = "weights/phoneme/english.bpm";
    std::string data_dir = "D:/projects/brosoundml-data", kokoro_dir = "weights/kokoro";
    int eval_windows = 1500;
    uint32_t seed = 1;
    bool asr = true, phoneme = true;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(k + " needs a value");
            return argv[++i];
        };
        if (k == "--align-train") align_train = next();
        else if (k == "--align-dev") align_dev = next();
        else if (k == "--cache-dev") cache_dev = next();
        else if (k == "--proj") proj_path = next();
        else if (k == "--probe-head") probe_path = next();
        else if (k == "--laya") laya = next();
        else if (k == "--eval-windows") eval_windows = std::atoi(next().c_str());
        else if (k == "--seed") seed = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (k == "--no-asr") asr = false;
        else if (k == "--no-phoneme") phoneme = false;
        else if (k == "--parakeet-dir") parakeet_dir = next();
        else if (k == "--phoneme-weights") phoneme_w = next();
        else if (k == "--data-dir") data_dir = next();
        else if (k == "--kokoro-dir") kokoro_dir = next();
        else die("unknown argument " + k);
    }
    if (align_train.empty() || align_dev.empty() || cache_dev.empty() || proj_path.empty())
        die("need --align-train --align-dev --cache-dev --proj");

    try {
        bt::init();
        const std::vector<AlignedUtterance> tr_utts = read_alignments(align_train);
        const std::vector<AlignedUtterance> dv_utts = read_alignments(align_dev);
        const WindowCache dv = read_cache(cache_dev);

        // Probe set: the same construction (and RNG order) as the train tool.
        Vocab all_vocab, seen;
        {
            std::vector<AlignedUtterance> both = tr_utts;
            both.insert(both.end(), dv_utts.begin(), dv_utts.end());
            all_vocab.build(both, 2, false);
        }
        seen.build(tr_utts, 1, true);
        std::mt19937 rng(seed);
        std::vector<int> wins(dv.windows.size());
        std::iota(wins.begin(), wins.end(), 0);
        std::shuffle(wins.begin(), wins.end(), rng);
        wins.resize(std::min<std::size_t>(wins.size(), static_cast<std::size_t>(eval_windows)));
        std::sort(wins.begin(), wins.end());
        const std::vector<Probe> probes = make_probes(dv, dv_utts, wins, all_vocab, &seen, 2, rng);
        std::fprintf(stderr, "%zu dev windows, %zu probes\n", wins.size(), probes.size());

        brolm::laya::DecisionModel model;
        model.load_model(laya);
        ItemBuilder builder(model);
        Mlp proj;
        proj.load(proj_path);

        std::vector<Method> methods;
        double t0 = now_ms();
        methods.push_back({"laya_audio", score_laya(model, builder, proj, dv, probes)});
        std::fprintf(stderr, "laya_audio scored in %.1f s\n", (now_ms() - t0) / 1000);

        // Text-Laya scorer for (state text, probe) pairs, keyword probes only.
        auto text_laya = [&](const std::vector<std::string>& state_of_window) {
            std::vector<float> out(probes.size(), kNaN);
            std::vector<std::size_t> which;
            std::vector<brolm::laya::SequenceResult> seqs;
            std::vector<int> qt;
            auto flush = [&] {
                std::vector<brolm::laya::LayaItem> items;
                for (std::size_t k = 0; k < seqs.size(); ++k) items.push_back(brolm::laya::LayaItem::of(seqs[k], qt[k]));
                const auto r = model.forward_items(items);
                for (std::size_t k = 0; k < which.size(); ++k) out[which[k]] = r[k].logits[1] - r[k].logits[0];
                which.clear();
                seqs.clear();
                qt.clear();
            };
            for (std::size_t i = 0; i < probes.size(); ++i) {
                if (probes[i].q != Question::Keyword) continue;
                brolm::laya::LayaQuestion q;
                q.id = "q";
                q.type = "noul";
                q.instructions = question_text(probes[i].q, probes[i].keyword);
                const std::string& st = state_of_window[static_cast<std::size_t>(probes[i].window)];
                seqs.push_back(model.build_sequence(st.empty() ? std::string("(silence)") : st, q));
                qt.push_back(q.qtype_index());
                which.push_back(i);
                if (seqs.size() == 64) flush();
            }
            if (!seqs.empty()) flush();
            return out;
        };

        // Oracle: the true words fully inside the window.
        {
            std::vector<std::string> truth(dv.windows.size());
            for (int w : wins) {
                const CachedWindow& cw = dv.windows[static_cast<std::size_t>(w)];
                const WindowFacts f = window_facts(dv_utts[static_cast<std::size_t>(cw.utt)], cw.t_end, dv.window_s);
                std::string s;
                for (const std::string& x : f.inside) s += (s.empty() ? "" : " ") + x;
                truth[static_cast<std::size_t>(w)] = s;
            }
            methods.push_back({"oracle_laya", text_laya(truth)});
        }

        // Audio-domain baselines: re-cut each window from its wav.
        if (asr || phoneme) {
            AsrBaseline asr_m;
            PhonemeBaseline ph_m;
            if (asr) asr_m.load(parakeet_dir);
            if (phoneme) ph_m.load(phoneme_w, data_dir, kokoro_dir);
            std::vector<std::string> hyp(dv.windows.size());
            std::vector<float> ph(probes.size(), kNaN), energy(probes.size(), kNaN);
            double asr_ms = 0, ph_ms = 0;
            int cur_utt = -1;
            std::vector<float> pcm;
            std::size_t p = 0;
            for (int w : wins) {
                const CachedWindow& cw = dv.windows[static_cast<std::size_t>(w)];
                if (cw.utt != cur_utt) {
                    pcm = load_audio_16k(dv_utts[static_cast<std::size_t>(cw.utt)].wav);
                    cur_utt = cw.utt;
                }
                const std::vector<float> win = window_audio(pcm, cw.t_end, dv.window_s, 7u);
                if (asr) {
                    const double a0 = now_ms();
                    hyp[static_cast<std::size_t>(w)] = asr_m.transcribe(win);
                    asr_ms += now_ms() - a0;
                }
                if (phoneme) {
                    const double a0 = now_ms();
                    ph_m.set_window(win);
                    ph_ms += now_ms() - a0;
                }
                double e2 = 0;
                const std::size_t tail = static_cast<std::size_t>(kTailS * 16000);
                for (std::size_t i = win.size() - tail; i < win.size(); ++i) e2 += double(win[i]) * win[i];
                const float db = static_cast<float>(10.0 * std::log10(e2 / tail + 1e-12));
                for (; p < probes.size() && probes[p].window == w; ++p) {
                    if (probes[p].q == Question::Speaking) energy[p] = db;
                    if (phoneme && probes[p].q == Question::Keyword) ph[p] = ph_m.score(probes[p].keyword);
                }
            }
            std::fprintf(stderr, "per window: parakeet %.1f ms, phoneme net %.1f ms\n", asr_ms / wins.size(),
                         ph_ms / wins.size());
            if (asr) {
                std::vector<float> match(probes.size(), kNaN);
                for (std::size_t i = 0; i < probes.size(); ++i) {
                    if (probes[i].q != Question::Keyword) continue;
                    const std::vector<std::string> ws =
                        normalize_words(hyp[static_cast<std::size_t>(probes[i].window)]);
                    match[i] = std::find(ws.begin(), ws.end(), probes[i].keyword) != ws.end() ? 1.0f : 0.0f;
                }
                methods.push_back({"asr_match", match});
                methods.push_back({"asr_laya", text_laya(hyp)});
            }
            if (phoneme) methods.push_back({"phoneme", ph});
            methods.push_back({"energy", energy});
        }

        if (!probe_path.empty()) {
            Mlp head;
            head.load(probe_path);
            std::vector<float> s(probes.size(), kNaN);
            for (std::size_t base = 0; base < wins.size(); base += 1024) {
                const std::size_t n = std::min<std::size_t>(1024, wins.size() - base);
                std::vector<uint16_t> bits(n * 2 * dv.dim);
                for (std::size_t k = 0; k < n; ++k) {
                    const CachedWindow& cw = dv.windows[static_cast<std::size_t>(wins[base + k])];
                    std::copy(cw.lat.end() - 2 * dv.dim, cw.lat.end(),
                              bits.begin() + static_cast<std::ptrdiff_t>(k * 2 * dv.dim));
                }
                bt::Tensor Y;
                head.forward(upload_fp16_as_fp32(bits.data(), static_cast<int>(n), 2 * dv.dim), Y);
                const std::vector<float> z = Y.to_host_vector();
                for (std::size_t i = 0; i < probes.size(); ++i) {
                    const auto it = std::find(wins.begin() + static_cast<std::ptrdiff_t>(base),
                                              wins.begin() + static_cast<std::ptrdiff_t>(base + n), probes[i].window);
                    if (it == wins.begin() + static_cast<std::ptrdiff_t>(base + n)) continue;
                    const std::size_t k = static_cast<std::size_t>(it - wins.begin()) - base;
                    if (probes[i].q == Question::Speaking) s[i] = z[k * 2];
                    if (probes[i].q == Question::WordEnd) s[i] = z[k * 2 + 1];
                }
            }
            methods.push_back({"probe", s});
        }

        report(probes, methods);

        // How binary the adapter's answers are: calibrated noul probabilities
        // (the checkpoint's noul temperature) that sit in the undecided middle.
        {
            const float temp = 1.9834f;
            int mid = 0, n = 0;
            for (const float z : methods[0].score) {
                const float pr = 1.0f / (1.0f + std::exp(-z / temp));
                if (pr > 0.1f && pr < 0.9f) ++mid;
                ++n;
            }
            std::printf("\nlaya_audio: %.1f %% of answers with 0.1 < p < 0.9 (calibrated, T=%.4f)\n", 100.0 * mid / n,
                        temp);
        }
        return 0;
    } catch (const std::exception& e) {
        die(e.what());
    }
}
