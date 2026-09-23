// brosoundml_laya_audio_train — train the audio projector into frozen Laya.
//
// The AuT latents of a streaming window (a 'LAC1' cache) go through a
// 2-layer MLP into Laya's embedding width and fill the state span of noul
// questions (laya_audio_task.h: keyword X spoken / someone speaking / a word
// just ended). Laya is frozen; LayaGrad carries dLoss/dlogit back to the soft
// rows and the projector trains on that (LLaVA-style). Loss: binary cross
// entropy on logit[true] - logit[false].
//
// With --probe the tool instead trains the dedicated-head baseline: a small
// MLP on the last two latent frames predicting speaking / word_end directly,
// no Laya involved.
//
// Usage:
//   brosoundml_laya_audio_train --align-train A --cache-train C
//        --align-dev A --cache-dev C --out proj.mlp
//        [--laya D:/projects/laya] [--steps 6000] [--batch 16] [--lr 3e-4]
//        [--eval-every 500] [--eval-windows 1500] [--probe] [--seed 1]

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
    std::string align_train, cache_train, align_dev, cache_dev, out, laya = "D:/projects/laya";
    int steps = 6000, batch = 16, eval_every = 500, eval_windows = 1500, hidden = 1024;
    float lr = 3e-4f;
    bool probe = false;
    uint32_t seed = 1;
};

// AUC report over dev probes, by question and keyword category.
void report(const std::vector<Probe>& probes, const std::vector<float>& score, const char* tag) {
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
    const double k_rand = sub([&](const Probe& p) { return kw(p) && (p.label == 1 || !p.hard_neg); });
    const double sp = sub([](const Probe& p) { return p.q == Question::Speaking; });
    const double we = sub([](const Probe& p) { return p.q == Question::WordEnd; });
    std::printf("[%s] keyword AUC all %.4f seen %.4f unseen %.4f | vs hard neg %.4f vs random neg %.4f | "
                "speaking %.4f word_end %.4f\n",
                tag, k_all, k_seen, k_unseen, k_hard, k_rand, sp, we);
    std::fflush(stdout);
}

int train_probe(const Args& a, const WindowCache& tr, const std::vector<AlignedUtterance>& tr_utts,
                const WindowCache& dv, const std::vector<AlignedUtterance>& dv_utts) {
    // Features: the last two latent frames of the window (the last ~160 ms).
    const int dim = tr.dim;
    auto features = [&](const WindowCache& c, const std::vector<int>& wins) {
        std::vector<uint16_t> bits(static_cast<std::size_t>(wins.size()) * 2 * dim);
        for (std::size_t k = 0; k < wins.size(); ++k) {
            const CachedWindow& w = c.windows[static_cast<std::size_t>(wins[k])];
            std::copy(w.lat.end() - 2 * dim, w.lat.end(), bits.begin() + static_cast<std::ptrdiff_t>(k * 2 * dim));
        }
        return upload_fp16_as_fp32(bits.data(), static_cast<int>(wins.size()), 2 * dim);
    };
    auto labels = [&](const WindowCache& c, const std::vector<AlignedUtterance>& utts, int wi, float* y) {
        const CachedWindow& w = c.windows[static_cast<std::size_t>(wi)];
        const WindowFacts f = window_facts(utts[static_cast<std::size_t>(w.utt)], w.t_end, c.window_s);
        y[0] = f.speaking ? 1.0f : 0.0f;
        y[1] = f.word_end ? 1.0f : 0.0f;
    };
    Mlp head;
    head.init(2 * dim, 256, 2, a.seed);
    std::mt19937 rng(a.seed);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(tr.windows.size()) - 1);
    const int B = 256;
    for (int step = 1; step <= a.steps; ++step) {
        std::vector<int> wins(B);
        for (int& w : wins) w = pick(rng);
        const bt::Tensor X = features(tr, wins);
        bt::Tensor Y;
        head.forward(X, Y);
        const std::vector<float> z = Y.to_host_vector();
        std::vector<float> dz(z.size());
        for (int b = 0; b < B; ++b) {
            float y[2];
            labels(tr, tr_utts, wins[static_cast<std::size_t>(b)], y);
            for (int j = 0; j < 2; ++j) dz[static_cast<std::size_t>(b) * 2 + j] = (sigmoid(z[b * 2 + j]) - y[j]) / B;
        }
        head.zero_grad();
        head.backward(bt::Tensor::from_host_on(bt::default_device(), dz.data(), B, 2));
        head.adam(a.lr);
        if (step % a.eval_every == 0 || step == a.steps) {
            std::vector<float> s0, s1;
            std::vector<int> l0, l1;
            for (std::size_t base = 0; base < dv.windows.size(); base += 1024) {
                std::vector<int> wins2;
                for (std::size_t i = base; i < std::min(dv.windows.size(), base + 1024); ++i)
                    wins2.push_back(static_cast<int>(i));
                bt::Tensor Yd;
                head.forward(features(dv, wins2), Yd);
                const std::vector<float> zd = Yd.to_host_vector();
                for (std::size_t k = 0; k < wins2.size(); ++k) {
                    float y[2];
                    labels(dv, dv_utts, wins2[k], y);
                    s0.push_back(zd[k * 2]);
                    s1.push_back(zd[k * 2 + 1]);
                    l0.push_back(static_cast<int>(y[0]));
                    l1.push_back(static_cast<int>(y[1]));
                }
            }
            std::printf("[probe step %d] dev speaking AUC %.4f word_end AUC %.4f (%zu windows)\n", step, auc(s0, l0),
                        auc(s1, l1), s0.size());
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
        if (k == "--align-train") a.align_train = next();
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
        else die("unknown argument " + k);
    }
    if (a.align_train.empty() || a.cache_train.empty() || a.align_dev.empty() || a.cache_dev.empty())
        die("need --align-train --cache-train --align-dev --cache-dev");

    try {
        bt::init();
        const std::vector<AlignedUtterance> tr_utts = read_alignments(a.align_train);
        const std::vector<AlignedUtterance> dv_utts = read_alignments(a.align_dev);
        const WindowCache tr = read_caches(a.cache_train);
        const WindowCache dv = read_cache(a.cache_dev);
        std::fprintf(stderr, "train %zu windows, dev %zu windows, dim %d\n", tr.windows.size(), dv.windows.size(),
                     tr.dim);
        if (a.probe) return train_probe(a, tr, tr_utts, dv, dv_utts);

        brolm::laya::DecisionModel model;
        model.load_model(a.laya);
        const int D = model.encoder().config().hidden_size;
        brolm::laya::LayaGrad grad(model);
        ItemBuilder builder(model);

        Vocab train_vocab, all_vocab;
        train_vocab.build(tr_utts, 2, /*exclude_held_out=*/true);
        {
            std::vector<AlignedUtterance> both = tr_utts;
            both.insert(both.end(), dv_utts.begin(), dv_utts.end());
            all_vocab.build(both, 2, false);
        }
        Vocab seen;
        seen.build(tr_utts, 1, true);
        std::fprintf(stderr, "keyword vocab: train %d, all %d\n", train_vocab.size(), all_vocab.size());

        // Fixed dev probe set.
        std::mt19937 rng(a.seed);
        std::vector<int> dev_wins(dv.windows.size());
        std::iota(dev_wins.begin(), dev_wins.end(), 0);
        std::shuffle(dev_wins.begin(), dev_wins.end(), rng);
        dev_wins.resize(std::min<std::size_t>(dev_wins.size(), static_cast<std::size_t>(a.eval_windows)));
        std::sort(dev_wins.begin(), dev_wins.end());
        const std::vector<Probe> dev_probes = make_probes(dv, dv_utts, dev_wins, all_vocab, &seen, 2, rng);

        Mlp proj;
        proj.init(tr.dim, a.hidden, D, a.seed);

        std::uniform_int_distribution<int> pick(0, static_cast<int>(tr.windows.size()) - 1);
        float scale = 1024.0f;
        int good_run = 0, skipped = 0;
        double loss_acc = 0, ms_acc = 0, fwd_acc = 0, bwd_acc = 0;
        int n_acc = 0;
        for (int step = 1; step <= a.steps; ++step) {
            const double t0 = now_ms();
            std::vector<int> wins(static_cast<std::size_t>(a.batch));
            for (int& w : wins) w = pick(rng);
            std::sort(wins.begin(), wins.end());
            wins.erase(std::unique(wins.begin(), wins.end()), wins.end());
            const std::vector<Probe> probes = make_probes(tr, tr_utts, wins, train_vocab, nullptr, 1, rng);
            const int F = tr.windows[static_cast<std::size_t>(wins.front())].frames;

            const bt::Tensor lat = gather_latents(tr, wins);
            bt::Tensor y, soft;
            proj.forward(lat, y);
            bt::cast(y, soft, bt::compute_dtype());
            std::vector<brolm::laya::LayaItem> items;
            for (std::size_t i = 0, k = 0; i < probes.size(); ++i) {
                if (probes[i].window != wins[k]) ++k;
                items.push_back(builder.item(question_text(probes[i].q, probes[i].keyword), F, static_cast<int>(k) * F));
            }
            const std::vector<float> logits = grad.forward(items, &soft);
            std::vector<float> dlog(logits.size(), 0.0f);
            double loss = 0;
            const float inv_n = 1.0f / static_cast<float>(probes.size());
            for (std::size_t i = 0; i < probes.size(); ++i) {
                const float z = logits[2 * i + 1] - logits[2 * i];
                const float yv = static_cast<float>(probes[i].label);
                const float p = sigmoid(z);
                loss += -(yv * std::log(std::max(p, 1e-7f)) + (1 - yv) * std::log(std::max(1 - p, 1e-7f)));
                dlog[2 * i + 1] = (p - yv) * inv_n;
                dlog[2 * i] = -(p - yv) * inv_n;
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
            ms_acc += now_ms() - t0;
            fwd_acc += grad.last_forward_ms();
            bwd_acc += grad.last_backward_ms();
            ++n_acc;
            if (step % 50 == 0) {
                std::printf("step %d loss %.4f  %.0f ms/step (laya fwd %.0f bwd %.0f, %zu items)  scale %.0f skipped %d\n",
                            step, loss_acc / n_acc, ms_acc / n_acc, fwd_acc / n_acc, bwd_acc / n_acc, probes.size(),
                            scale, skipped);
                std::fflush(stdout);
                loss_acc = ms_acc = fwd_acc = bwd_acc = 0;
                n_acc = 0;
            }
            if (step % a.eval_every == 0 || step == a.steps) {
                const std::vector<float> s = score_laya(model, builder, proj, dv, dev_probes);
                report(dev_probes, s, ("dev step " + std::to_string(step)).c_str());
                if (!a.out.empty()) proj.save(a.out);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        die(e.what());
    }
}
