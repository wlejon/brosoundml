// brosoundml_laya_audio_eval — held-out evaluation of a Laya audio adapter,
// per evaluation set (domain), with the same probes for every adapter.
//
// Probes (the questions asked of each window) are built from a FIXED
// negative vocabulary (--vocab-align) and seed, so two adapters evaluated
// with the same --vocab-align / --set / --seed answer identical questions.
// Which keywords count as "seen" (asked in training) is adapter-specific
// (--train-align: that adapter's training alignments); the curve counts come
// from --count-align (default: --train-align).
//
// Methods per probe:
//   laya_audio   the adapter (projected AuT latents in Laya's state span)
//   asr_stream   (--asr) Parakeet over the utterance up to the window end;
//                keyword among the words whose midpoint is in the window
//   probe        (--probe-head) dedicated MLP on the last two latent frames
//   energy       RMS level of the last 0.3 s (speaking), with --asr
//
// Per set: AUC / TPR at 1 % and 5 % FPR by category, and the
// examples-per-word curve. Non-speech sets (no words) have no keyword
// positives; they report the fraction of keyword and speaking answers above
// fixed probabilities instead.
//
// Usage:
//   brosoundml_laya_audio_eval --vocab-align A[+A...] --train-align A[+A...]
//        --set NAME,ALIGN,CACHE [--set ...] --proj proj.mlp
//        [--laya D:/projects/laya] [--count-align A[+A]] [--eval-windows 1500]
//        [--seed 1] [--asr] [--probe-head head.mlp] [--temperature T]
//        [--parakeet-dir weights/parakeet/0.6b-v3]

#include "laya_audio_baselines.h"
#include "laya_audio_corpus.h"
#include "laya_audio_metrics.h"
#include "laya_audio_task.h"

#include "brolm/laya.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <sstream>
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

std::vector<AlignedUtterance> read_many(const std::string& plus_list) {
    std::vector<AlignedUtterance> out;
    std::istringstream is(plus_list);
    std::string p;
    while (std::getline(is, p, '+')) {
        if (p.empty()) continue;
        for (AlignedUtterance& u : read_alignments(p)) out.push_back(std::move(u));
    }
    return out;
}

// Non-speech set: how often the adapter says yes with nothing to hear.
void report_nonspeech(const std::string& set, const std::vector<Probe>& probes, const std::vector<float>& z,
                      float temperature) {
    const float thr[3] = {0.5f, 0.7f, 0.9f};
    int n_kw = 0, n_sp = 0, kw_hit[3] = {0, 0, 0}, sp_hit[3] = {0, 0, 0};
    for (std::size_t i = 0; i < probes.size(); ++i) {
        const float p = 1.0f / (1.0f + std::exp(-z[i] / temperature));
        if (probes[i].q == Question::Keyword) {
            ++n_kw;
            for (int k = 0; k < 3; ++k) kw_hit[k] += p >= thr[k];
        } else if (probes[i].q == Question::Speaking) {
            ++n_sp;
            for (int k = 0; k < 3; ++k) sp_hit[k] += p >= thr[k];
        }
    }
    std::printf("\n%-16s non-speech: keyword answers p>=0.5 %.4f p>=0.7 %.4f p>=0.9 %.4f (%d) | "
                "speaking p>=0.5 %.4f p>=0.7 %.4f p>=0.9 %.4f (%d)\n",
                set.c_str(), double(kw_hit[0]) / n_kw, double(kw_hit[1]) / n_kw, double(kw_hit[2]) / n_kw, n_kw,
                double(sp_hit[0]) / n_sp, double(sp_hit[1]) / n_sp, double(sp_hit[2]) / n_sp, n_sp);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    std::string vocab_align, train_align, count_align, proj_path, probe_path, laya = "D:/projects/laya";
    std::string parakeet_dir = "weights/parakeet/0.6b-v3";
    std::vector<std::string> sets;
    int eval_windows = 1500;
    uint32_t seed = 1;
    bool asr = false;
    float temperature = 1.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(k + " needs a value");
            return argv[++i];
        };
        if (k == "--vocab-align") vocab_align = next();
        else if (k == "--train-align") train_align = next();
        else if (k == "--count-align") count_align = next();
        else if (k == "--set") sets.push_back(next());
        else if (k == "--proj") proj_path = next();
        else if (k == "--probe-head") probe_path = next();
        else if (k == "--laya") laya = next();
        else if (k == "--eval-windows") eval_windows = std::atoi(next().c_str());
        else if (k == "--seed") seed = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (k == "--asr") asr = true;
        else if (k == "--temperature") temperature = std::stof(next());
        else if (k == "--parakeet-dir") parakeet_dir = next();
        else die("unknown argument " + k);
    }
    if (vocab_align.empty() || train_align.empty() || sets.empty() || proj_path.empty())
        die("need --vocab-align --train-align --set --proj");
    if (count_align.empty()) count_align = train_align;

    try {
        bt::init();
        Corpus ev;
        for (const std::string& s : sets) ev.add(parse_domain(s, false));

        // Negatives: a fixed vocabulary (same for every adapter) + the sets'.
        VocabSet neg_vocab;
        {
            std::vector<AlignedUtterance> v = read_many(vocab_align);
            v.insert(v.end(), ev.utts.begin(), ev.utts.end());
            neg_vocab.build(v, 2, false);
        }
        const std::vector<AlignedUtterance> tr_utts = read_many(train_align);
        Vocab seen;
        seen.build(tr_utts, 1, true);
        const std::unordered_map<std::string, int> counts =
            count_align == train_align ? training_counts(tr_utts) : training_counts(read_many(count_align));
        std::fprintf(stderr, "adapter training vocabulary: %d words\n", seen.size());

        brolm::laya::DecisionModel model;
        model.load_model(laya);
        ItemBuilder builder(model);
        Mlp proj;
        proj.load(proj_path);
        AsrBaseline asr_m;
        if (asr) asr_m.load(parakeet_dir);
        Mlp head;
        if (!probe_path.empty()) head.load(probe_path);

        for (std::size_t d = 0; d < ev.domains.size(); ++d) {
            const std::string& name = ev.domains[d].name;
            std::mt19937 rng(seed);
            std::vector<int> wins = ev.windows_of(static_cast<int>(d));
            std::shuffle(wins.begin(), wins.end(), rng);
            wins.resize(std::min<std::size_t>(wins.size(), static_cast<std::size_t>(eval_windows)));
            std::sort(wins.begin(), wins.end());
            // Probe construction marks nothing: `unseen` is set below against
            // this adapter's own training vocabulary.
            Vocab none;
            std::vector<Probe> probes = make_probes(ev.cache, ev.utts, wins, neg_vocab, &none, 2, rng);
            for (Probe& p : probes)
                if (p.q == Question::Keyword) p.unseen = held_out_keyword(p.keyword) || !seen.contains(p.keyword);
            std::fprintf(stderr, "%s: %zu windows, %zu probes\n", name.c_str(), wins.size(), probes.size());

            std::vector<Method> methods;
            const double t0 = now_ms();
            methods.push_back({"laya_audio", score_laya(model, builder, proj, ev.cache, probes)});
            std::fprintf(stderr, "%s: laya_audio scored in %.1f s\n", name.c_str(), (now_ms() - t0) / 1000);

            bool has_words = false;
            for (int u = ev.domains[d].u0; u < ev.domains[d].u1 && !has_words; ++u)
                has_words = !ev.utts[static_cast<std::size_t>(u)].words.empty();

            if (asr && has_words) {
                std::vector<float> match(probes.size(), kNaN), energy(probes.size(), kNaN);
                int cur_utt = -1;
                std::vector<float> pcm;
                std::size_t p = 0;
                for (int w : wins) {
                    const CachedWindow& cw = ev.cache.windows[static_cast<std::size_t>(w)];
                    if (cw.utt != cur_utt) {
                        pcm = load_audio_16k(ev.utts[static_cast<std::size_t>(cw.utt)].wav);
                        cur_utt = cw.utt;
                    }
                    const std::vector<std::string> hyp =
                        normalize_words(asr_m.transcribe_stream_window(pcm, cw.t_end, ev.cache.window_s));
                    const std::vector<float> win = window_audio(pcm, cw.t_end, ev.cache.window_s, 7u);
                    double e2 = 0;
                    const std::size_t tail = static_cast<std::size_t>(kTailS * 16000);
                    for (std::size_t i = win.size() - tail; i < win.size(); ++i) e2 += double(win[i]) * win[i];
                    const float db = static_cast<float>(10.0 * std::log10(e2 / tail + 1e-12));
                    for (; p < probes.size() && probes[p].window == w; ++p) {
                        if (probes[p].q == Question::Speaking) energy[p] = db;
                        if (probes[p].q == Question::Keyword)
                            match[p] = std::find(hyp.begin(), hyp.end(), probes[p].keyword) != hyp.end() ? 1.0f : 0.0f;
                    }
                }
                methods.push_back({"asr_stream", match});
                methods.push_back({"energy", energy});
            }
            if (!probe_path.empty()) {
                std::vector<float> s(probes.size(), kNaN);
                const int dim = ev.cache.dim;
                std::vector<uint16_t> bits(wins.size() * 2 * dim);
                for (std::size_t k = 0; k < wins.size(); ++k) {
                    const CachedWindow& cw = ev.cache.windows[static_cast<std::size_t>(wins[k])];
                    std::copy(cw.lat.end() - 2 * dim, cw.lat.end(), bits.begin() + static_cast<std::ptrdiff_t>(k * 2 * dim));
                }
                bt::Tensor Y;
                head.forward(upload_fp16_as_fp32(bits.data(), static_cast<int>(wins.size()), 2 * dim), Y);
                const std::vector<float> z = Y.to_host_vector();
                for (std::size_t i = 0; i < probes.size(); ++i) {
                    const std::size_t k = static_cast<std::size_t>(
                        std::lower_bound(wins.begin(), wins.end(), probes[i].window) - wins.begin());
                    if (probes[i].q == Question::Speaking) s[i] = z[k * 2];
                    if (probes[i].q == Question::WordEnd) s[i] = z[k * 2 + 1];
                }
                methods.push_back({"probe", s});
            }

            if (has_words) {
                report_categories(name, probes, methods);
                report_curve(name, probes, methods[0].score, counts);
            } else {
                report_nonspeech(name, probes, methods[0].score, temperature);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        die(e.what());
    }
}
