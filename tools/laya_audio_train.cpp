// brosoundml_laya_audio_train — train the audio projector into frozen Laya.
//
// The AuT latents of a streaming window (a 'LAC1' cache) go through a
// 2-layer MLP into Laya's embedding width and fill the state span of noul
// questions (laya_audio_task.h: keyword X spoken / someone speaking / a word
// just ended). Laya is frozen; LayaGrad carries dLoss/dlogit back to the soft
// rows and the projector trains on that (LLaVA-style). Loss: binary cross
// entropy on logit[true] - logit[false].
//
// Training data is a weighted mix of domains (laya_audio_corpus.h): each
// batch window picks a domain by weight, then one of its windows. Keyword
// negatives come from the window's language vocabulary. Dev domains are
// reported separately.
//
// With --probe the tool instead trains the dedicated-head baseline: a small
// MLP on the last two latent frames predicting speaking / word_end directly,
// no Laya involved.
//
// Usage:
//   brosoundml_laya_audio_train --train NAME,WEIGHT,ALIGN,CACHE[+CACHE] ...
//        --dev NAME,ALIGN,CACHE ... --out proj.mlp
//        [--laya D:/projects/laya] [--steps 6000] [--batch 16] [--lr 3e-4]
//        [--eval-every 500] [--eval-windows 600] [--probe] [--seed 1] [--init proj.mlp]
//        [--bank questions.tsv [--teacher LAYA_DIR] [--teacher-temperature T] [--student-temperature T]
//         [--free-cand 12] [--free-keep 4] [--free-weight 1] [--free-dev-windows 80]]
//   --bank adds free-form question distillation (laya_audio_distill.h): the
//   teacher is text Laya on each window's gold transcript; per window
//   --free-keep of --free-cand sampled trained questions are learned as soft
//   targets next to the keyword / speaking / word-end tasks.
//   (--align-train A --cache-train C[,C] --align-dev A --cache-dev C is the
//    single-domain form of the first adapter.)

#include "laya_audio_corpus.h"
#include "laya_audio_distill.h"
#include "laya_audio_task.h"

#include "brolm/laya.h"
#include "brolm/laya_grad.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

namespace bt = brotensor;
using namespace laya_audio;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "laya_audio_train: %s\n", msg.c_str());
    std::exit(2);
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

struct Args {
    std::vector<std::string> train, dev;
    std::string align_train, cache_train, align_dev, cache_dev, out, laya = "D:/projects/laya";
    int steps = 6000, batch = 16, eval_every = 500, eval_windows = 600, hidden = 1024;
    float lr = 3e-4f;
    bool probe = false;
    uint32_t seed = 1;
    // Free-form distillation (laya_audio_distill.h).
    std::string bank, teacher, init;
    float teacher_temperature = 1.0f, student_temperature = 1.0f, free_weight = 1.0f;
    int free_cand = 12, free_keep = 4, free_dev_windows = 80;
};

// AUC report over dev probes, by question and keyword category.
void report(const std::vector<Probe>& probes, const std::vector<float>& score, const std::string& tag) {
    auto sub = [&](auto pred) {
        std::vector<float> s;
        std::vector<int> l;
        for (std::size_t i = 0; i < probes.size(); ++i)
            if (pred(probes[i])) {
                s.push_back(score[i]);
                l.push_back(probes[i].label);
            }
        return auc(s, l);
    };
    auto kw = [](const Probe& p) { return p.q == Question::Keyword; };
    const double k_all = sub(kw);
    const double k_seen = sub([&](const Probe& p) { return kw(p) && !p.unseen; });
    const double k_unseen = sub([&](const Probe& p) { return kw(p) && p.unseen; });
    const double k_hard = sub([&](const Probe& p) { return kw(p) && (p.label == 1 || p.hard_neg); });
    const double sp = sub([](const Probe& p) { return p.q == Question::Speaking; });
    const double we = sub([](const Probe& p) { return p.q == Question::WordEnd; });
    std::printf("[%s] keyword AUC all %.4f seen %.4f unseen %.4f vs hard neg %.4f | speaking %.4f word_end %.4f\n",
                tag.c_str(), k_all, k_seen, k_unseen, k_hard, sp, we);
    std::fflush(stdout);
}

int train_probe(const Args& a, const Corpus& tr, const Corpus& dv) {
    // Features: the last two latent frames of the window (the last ~160 ms).
    const int dim = tr.cache.dim;
    auto features = [&](const WindowCache& c, const std::vector<int>& wins) {
        std::vector<uint16_t> bits(static_cast<std::size_t>(wins.size()) * 2 * dim);
        for (std::size_t k = 0; k < wins.size(); ++k) {
            const CachedWindow& w = c.windows[static_cast<std::size_t>(wins[k])];
            std::copy(w.lat.end() - 2 * dim, w.lat.end(), bits.begin() + static_cast<std::ptrdiff_t>(k * 2 * dim));
        }
        return upload_fp16_as_fp32(bits.data(), static_cast<int>(wins.size()), 2 * dim);
    };
    auto labels = [&](const Corpus& c, int wi, float* y) {
        const CachedWindow& w = c.cache.windows[static_cast<std::size_t>(wi)];
        const WindowFacts f = window_facts(c.utts[static_cast<std::size_t>(w.utt)], w.t_end, c.cache.window_s);
        y[0] = f.speaking ? 1.0f : 0.0f;
        y[1] = f.word_end ? 1.0f : 0.0f;
    };
    Mlp head;
    head.init(2 * dim, 256, 2, a.seed);
    std::mt19937 rng(a.seed);
    const int B = 256;
    for (int step = 1; step <= a.steps; ++step) {
        std::vector<int> wins(B);
        for (int& w : wins) w = tr.sample_window(rng);
        bt::Tensor Y;
        head.forward(features(tr.cache, wins), Y);
        const std::vector<float> z = Y.to_host_vector();
        std::vector<float> dz(z.size());
        for (int b = 0; b < B; ++b) {
            float y[2];
            labels(tr, wins[static_cast<std::size_t>(b)], y);
            for (int j = 0; j < 2; ++j) dz[static_cast<std::size_t>(b) * 2 + j] = (sigmoid(z[b * 2 + j]) - y[j]) / B;
        }
        head.zero_grad();
        head.backward(bt::Tensor::from_host_on(bt::default_device(), dz.data(), B, 2));
        head.adam(a.lr);
        if (step % a.eval_every == 0 || step == a.steps) {
            for (std::size_t d = 0; d < dv.domains.size(); ++d) {
                std::vector<float> s0, s1;
                std::vector<int> l0, l1;
                const std::vector<int> all = dv.windows_of(static_cast<int>(d));
                for (std::size_t base = 0; base < all.size(); base += 1024) {
                    const std::vector<int> wins2(all.begin() + static_cast<std::ptrdiff_t>(base),
                                                 all.begin() + static_cast<std::ptrdiff_t>(std::min(all.size(), base + 1024)));
                    bt::Tensor Yd;
                    head.forward(features(dv.cache, wins2), Yd);
                    const std::vector<float> zd = Yd.to_host_vector();
                    for (std::size_t k = 0; k < wins2.size(); ++k) {
                        float y[2];
                        labels(dv, wins2[k], y);
                        s0.push_back(zd[k * 2]);
                        s1.push_back(zd[k * 2 + 1]);
                        l0.push_back(static_cast<int>(y[0]));
                        l1.push_back(static_cast<int>(y[1]));
                    }
                }
                std::printf("[probe step %d %s] speaking AUC %.4f word_end AUC %.4f (%zu windows)\n", step,
                            dv.domains[d].name.c_str(), auc(s0, l0), auc(s1, l1), s0.size());
            }
            std::fflush(stdout);
        }
    }
    if (!a.out.empty()) head.save(a.out);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(k + " needs a value");
            return argv[++i];
        };
        if (k == "--train") a.train.push_back(next());
        else if (k == "--dev") a.dev.push_back(next());
        else if (k == "--align-train") a.align_train = next();
        else if (k == "--cache-train") a.cache_train = next();
        else if (k == "--align-dev") a.align_dev = next();
        else if (k == "--cache-dev") a.cache_dev = next();
        else if (k == "--out") a.out = next();
        else if (k == "--laya") a.laya = next();
        else if (k == "--steps") a.steps = std::atoi(next().c_str());
        else if (k == "--batch") a.batch = std::atoi(next().c_str());
        else if (k == "--lr") a.lr = std::stof(next());
        else if (k == "--eval-every") a.eval_every = std::atoi(next().c_str());
        else if (k == "--eval-windows") a.eval_windows = std::atoi(next().c_str());
        else if (k == "--hidden") a.hidden = std::atoi(next().c_str());
        else if (k == "--probe") a.probe = true;
        else if (k == "--seed") a.seed = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (k == "--bank") a.bank = next();
        else if (k == "--teacher") a.teacher = next();
        else if (k == "--teacher-temperature") a.teacher_temperature = std::stof(next());
        else if (k == "--student-temperature") a.student_temperature = std::stof(next());
        else if (k == "--free-weight") a.free_weight = std::stof(next());
        else if (k == "--free-cand") a.free_cand = std::atoi(next().c_str());
        else if (k == "--free-keep") a.free_keep = std::atoi(next().c_str());
        else if (k == "--free-dev-windows") a.free_dev_windows = std::atoi(next().c_str());
        else if (k == "--init") a.init = next();
        else die("unknown argument " + k);
    }
    if (!a.align_train.empty()) {
        std::string caches = a.cache_train;
        std::replace(caches.begin(), caches.end(), ',', '+');
        a.train.push_back("libritts,1," + a.align_train + "," + caches);
    }
    if (!a.align_dev.empty()) a.dev.push_back("libritts-dev," + a.align_dev + "," + a.cache_dev);
    if (a.train.empty() || a.dev.empty()) die("need --train and --dev domains");

    try {
        bt::init();
        Corpus tr, dv;
        for (const std::string& s : a.train) tr.add(parse_domain(s, true));
        for (const std::string& s : a.dev) dv.add(parse_domain(s, false));
        for (const Corpus::Domain& d : tr.domains)
            std::fprintf(stderr, "train %-14s weight %.3f: %d utterances, %d windows\n", d.name.c_str(), d.weight,
                         d.u1 - d.u0, d.w1 - d.w0);
        for (const Corpus::Domain& d : dv.domains)
            std::fprintf(stderr, "dev   %-14s %d utterances, %d windows\n", d.name.c_str(), d.u1 - d.u0, d.w1 - d.w0);
        if (a.probe) return train_probe(a, tr, dv);

        brolm::laya::DecisionModel model;
        model.load_model(a.laya);
        const int D = model.encoder().config().hidden_size;
        brolm::laya::LayaGrad grad(model);
        ItemBuilder builder(model);

        VocabSet train_vocab, all_vocab;
        train_vocab.build(tr.utts, 2, /*exclude_held_out=*/true);
        {
            std::vector<AlignedUtterance> both = tr.utts;
            both.insert(both.end(), dv.utts.begin(), dv.utts.end());
            all_vocab.build(both, 2, false);
        }
        Vocab seen;
        seen.build(tr.utts, 1, true);
        for (const std::string& l : train_vocab.languages())
            std::fprintf(stderr, "keyword vocab %s: train %d, all %d\n", l.c_str(), train_vocab.get(l).size(),
                         all_vocab.get(l).size());

        // Fixed dev probe sets, one per dev domain.
        std::mt19937 rng(a.seed);
        std::vector<std::vector<Probe>> dev_probes(dv.domains.size());
        for (std::size_t d = 0; d < dv.domains.size(); ++d) {
            std::vector<int> wins = dv.windows_of(static_cast<int>(d));
            std::shuffle(wins.begin(), wins.end(), rng);
            wins.resize(std::min<std::size_t>(wins.size(), static_cast<std::size_t>(a.eval_windows)));
            std::sort(wins.begin(), wins.end());
            dev_probes[d] = make_probes(dv.cache, dv.utts, wins, all_vocab, &seen, 2, rng);
        }

        // Free-form distillation: teacher = text Laya (--teacher, default
        // the student's own checkpoint) on the window's gold transcript.
        Distiller distill;
        brolm::laya::DecisionModel teacher_own;
        std::vector<Distiller::Dev> free_dev;
        if (!a.bank.empty()) {
            brolm::laya::DecisionModel* teacher = &model;
            if (!a.teacher.empty() && a.teacher != a.laya) {
                teacher_own.load_model(a.teacher);
                teacher = &teacher_own;
            }
            distill.init(a.bank, teacher, a.teacher_temperature, a.free_cand, a.free_keep);
            for (std::size_t d = 0; d < dv.domains.size(); ++d)
                free_dev.push_back(distill.make_dev(dv, static_cast<int>(d), a.free_dev_windows, rng));
        }

        Mlp proj;
        if (!a.init.empty()) {
            proj.load(a.init);
            if (proj.d_in() != tr.cache.dim || proj.d_out() != D) die("--init projector does not fit this Laya");
        } else {
            proj.init(tr.cache.dim, a.hidden, D, a.seed);
        }

        float scale = 1024.0f;
        int good_run = 0, skipped = 0;
        double loss_acc = 0, free_acc = 0, teacher_acc = 0, ms_acc = 0, fwd_acc = 0, bwd_acc = 0;
        int n_acc = 0;
        for (int step = 1; step <= a.steps; ++step) {
            const double t0 = now_ms();
            std::vector<int> wins(static_cast<std::size_t>(a.batch));
            for (int& w : wins) w = tr.sample_window(rng);
            std::sort(wins.begin(), wins.end());
            wins.erase(std::unique(wins.begin(), wins.end()), wins.end());
            const std::vector<Probe> probes = make_probes(tr.cache, tr.utts, wins, train_vocab, nullptr, 1, rng);
            const int F = tr.cache.windows[static_cast<std::size_t>(wins.front())].frames;
            // Teacher before the student's forward (it may share the model).
            const std::vector<DistillItem> free =
                a.bank.empty() ? std::vector<DistillItem>{} : distill.sample(tr, wins, rng);

            const bt::Tensor lat = gather_latents(tr.cache, wins);
            bt::Tensor y, soft;
            proj.forward(lat, y);
            bt::cast(y, soft, bt::compute_dtype());
            std::vector<brolm::laya::LayaItem> items;
            for (std::size_t i = 0, k = 0; i < probes.size(); ++i) {
                if (probes[i].window != wins[k]) ++k;
                items.push_back(builder.item(question_text(probes[i].q, probes[i].keyword), F, static_cast<int>(k) * F));
            }
            for (const DistillItem& d : free) items.push_back(builder.item(distill.text(d.q), F, d.k * F));
            const std::vector<float> logits = grad.forward(items, &soft);
            std::vector<float> dlog(logits.size(), 0.0f);
            double loss = 0, free_loss = 0;
            const float inv_n = 1.0f / static_cast<float>(items.size());
            for (std::size_t i = 0; i < probes.size(); ++i) {
                const float z = logits[2 * i + 1] - logits[2 * i];
                const float yv = static_cast<float>(probes[i].label);
                const float p = sigmoid(z);
                loss += -(yv * std::log(std::max(p, 1e-7f)) + (1 - yv) * std::log(std::max(1 - p, 1e-7f)));
                dlog[2 * i + 1] = (p - yv) * inv_n;
                dlog[2 * i] = -(p - yv) * inv_n;
            }
            // Free-form: soft-target BCE on the student's calibrated probability.
            const float Ts = a.student_temperature;
            for (std::size_t j = 0; j < free.size(); ++j) {
                const std::size_t i = probes.size() + j;
                const float z = (logits[2 * i + 1] - logits[2 * i]) / Ts;
                const float yv = free[j].target, p = sigmoid(z);
                free_loss += -(yv * std::log(std::max(p, 1e-7f)) + (1 - yv) * std::log(std::max(1 - p, 1e-7f)));
                const float g = a.free_weight * (p - yv) / Ts * inv_n;
                dlog[2 * i + 1] = g;
                dlog[2 * i] = -g;
            }
            bt::Tensor d_soft;
            if (!grad.backward(dlog, d_soft, scale)) {
                scale *= 0.5f;
                good_run = 0;
                ++skipped;
                continue;
            }
            if (++good_run == 1000 && scale < 65536.0f) {
                scale *= 2.0f;
                good_run = 0;
            }
            proj.zero_grad();
            proj.backward(d_soft);
            const int warm = 200;
            const float lr = a.lr * std::min(1.0f, static_cast<float>(step) / warm) *
                             (0.5f * (1.0f + std::cos(3.14159265f * static_cast<float>(step) / a.steps)));
            proj.adam(lr);
            bt::sync(bt::default_device());
            loss_acc += loss / probes.size();
            free_acc += free.empty() ? 0.0 : free_loss / free.size();
            teacher_acc += free.empty() ? 0.0 : distill.last_teacher_ms();
            ms_acc += now_ms() - t0;
            fwd_acc += grad.last_forward_ms();
            bwd_acc += grad.last_backward_ms();
            ++n_acc;
            if (step % 50 == 0) {
                std::printf("step %d loss %.4f free %.4f  %.0f ms/step (teacher %.0f laya fwd %.0f bwd %.0f, %zu items)  "
                            "scale %.0f skipped %d\n",
                            step, loss_acc / n_acc, free_acc / n_acc, ms_acc / n_acc, teacher_acc / n_acc,
                            fwd_acc / n_acc, bwd_acc / n_acc, items.size(), scale, skipped);
                std::fflush(stdout);
                loss_acc = free_acc = teacher_acc = ms_acc = fwd_acc = bwd_acc = 0;
                n_acc = 0;
            }
            if (step % a.eval_every == 0 || step == a.steps) {
                for (std::size_t d = 0; d < dv.domains.size(); ++d) {
                    const std::vector<float> s = score_laya(model, builder, proj, dv.cache, dev_probes[d]);
                    report(dev_probes[d], s, "step " + std::to_string(step) + " " + dv.domains[d].name);
                }
                for (const Distiller::Dev& fd : free_dev)
                    distill.report(model, builder, proj, dv.cache, fd, Ts, "step " + std::to_string(step));
                if (!a.out.empty()) proj.save(a.out);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        die(e.what());
    }
}
