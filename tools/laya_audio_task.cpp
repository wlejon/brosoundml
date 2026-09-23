#include "laya_audio_task.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace laya_audio {

namespace bt = brotensor;

const char* question_name(Question q) {
    switch (q) {
        case Question::Keyword: return "keyword";
        case Question::Speaking: return "speaking";
        case Question::WordEnd: return "word_end";
    }
    return "?";
}

std::string question_text(Question q, const std::string& keyword) {
    switch (q) {
        case Question::Keyword: return "The word \"" + keyword + "\" is spoken in this audio.";
        case Question::Speaking: return "Someone is speaking at the end of this audio.";
        case Question::WordEnd: return "A word has just been spoken, ending at the end of this audio.";
    }
    return {};
}

bool held_out_keyword(const std::string& w) {
    uint32_t h = 2166136261u;  // FNV-1a
    for (unsigned char c : w) h = (h ^ c) * 16777619u;
    return h % 10u == 0u;
}

WindowFacts window_facts(const AlignedUtterance& u, float t_end, float window_s) {
    WindowFacts f;
    const float ts = t_end - window_s;
    for (const TimedWord& w : u.words) {
        if (w.t0 < 0 || w.t1 <= w.t0) continue;
        if (w.t1 > t_end - kTailS && w.t0 < t_end) f.speaking = true;
        if (w.t1 > t_end - kTailS && w.t1 <= t_end) f.word_end = true;
        if (w.t0 >= ts && w.t1 <= t_end) f.inside.push_back(w.word);
        else if (w.t1 > ts && w.t0 < t_end) f.partial.insert(w.word);
    }
    return f;
}

int keyword_label(const WindowFacts& f, const std::string& w) {
    if (std::find(f.inside.begin(), f.inside.end(), w) != f.inside.end()) return 1;
    return f.partial.count(w) ? -1 : 0;
}

int edit_distance(const std::string& a, const std::string& b, int cap) {
    const int n = static_cast<int>(a.size()), m = static_cast<int>(b.size());
    if (std::abs(n - m) > cap) return cap + 1;
    std::vector<int> prev(m + 1), cur(m + 1);
    std::iota(prev.begin(), prev.end(), 0);
    for (int i = 1; i <= n; ++i) {
        cur[0] = i;
        int row_min = cur[0];
        for (int j = 1; j <= m; ++j) {
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
            row_min = std::min(row_min, cur[j]);
        }
        if (row_min > cap) return cap + 1;
        std::swap(prev, cur);
    }
    return prev[m];
}

void Vocab::build(const std::vector<AlignedUtterance>& utts, int min_count, bool exclude_held_out) {
    std::unordered_map<std::string, int> count;
    for (const AlignedUtterance& u : utts)
        for (const TimedWord& w : u.words) ++count[w.word];
    words_.clear();
    for (const auto& [w, c] : count) {
        if (c < min_count || w.size() < 3) continue;
        if (exclude_held_out && held_out_keyword(w)) continue;
        words_.push_back(w);
    }
    std::sort(words_.begin(), words_.end());
    cum_.clear();
    index_.clear();
    double acc = 0;
    for (std::size_t i = 0; i < words_.size(); ++i) {
        acc += std::sqrt(static_cast<double>(count[words_[i]]));  // flattened frequency
        cum_.push_back(acc);
        index_[words_[i]] = static_cast<int>(i);
    }
    nb_cache_.clear();
}

const std::string& Vocab::sample(std::mt19937& rng) const {
    std::uniform_real_distribution<double> d(0.0, cum_.back());
    const auto it = std::upper_bound(cum_.begin(), cum_.end(), d(rng));
    return words_[std::min<std::size_t>(static_cast<std::size_t>(it - cum_.begin()), words_.size() - 1)];
}

const std::vector<std::string>& Vocab::neighbours(const std::string& w) {
    auto it = nb_cache_.find(w);
    if (it != nb_cache_.end()) return it->second;
    std::vector<std::string> nb;
    for (const std::string& v : words_) {
        if (v == w) continue;
        if (edit_distance(v, w, 2) <= 2) nb.push_back(v);
    }
    return nb_cache_.emplace(w, std::move(nb)).first->second;
}

brolm::laya::LayaItem ItemBuilder::item(const std::string& instructions, int n_soft, int soft_row) {
    const std::string key = instructions + '\x1f' + std::to_string(n_soft);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        brolm::laya::LayaQuestion q;
        q.id = "q";
        q.type = "noul";
        q.instructions = instructions;
        const std::string probe = "audio";
        const brolm::laya::SequenceResult seq = m_.build_sequence(probe, q);
        const int n_state = static_cast<int>(m_.tokenizer().encode_state(probe).size());
        const int state_pos = static_cast<int>(seq.input_ids.size()) - 1 - n_state;
        Seq s;
        s.ids.assign(seq.input_ids.begin(), seq.input_ids.begin() + state_pos);
        s.soft_pos = state_pos;
        s.ids.insert(s.ids.end(), static_cast<std::size_t>(n_soft), m_.tokenizer().mask_token_id());
        s.ids.push_back(seq.input_ids.back());
        s.markers = seq.marker_pos;
        for (int32_t p : s.markers)
            if (p >= state_pos) throw std::runtime_error("laya_audio: marker inside the state span");
        s.qtype = q.qtype_index();
        it = cache_.emplace(key, std::move(s)).first;
    }
    const Seq& s = it->second;
    brolm::laya::LayaItem li{s.ids.data(), static_cast<int>(s.ids.size()), s.markers.data(),
                             static_cast<int>(s.markers.size()), s.qtype};
    li.soft_pos = s.soft_pos;
    li.soft_count = n_soft;
    li.soft_row = soft_row;
    return li;
}

std::vector<Probe> make_probes(const WindowCache& cache, const std::vector<AlignedUtterance>& utts,
                               const std::vector<int>& windows, Vocab& neg_vocab, const Vocab* seen, int kw_pos,
                               std::mt19937& rng) {
    const bool training = seen == nullptr;
    std::vector<Probe> out;
    for (int wi : windows) {
        const CachedWindow& w = cache.windows[static_cast<std::size_t>(wi)];
        const AlignedUtterance& u = utts[static_cast<std::size_t>(w.utt)];
        const WindowFacts f = window_facts(u, w.t_end, cache.window_s);
        auto unseen = [&](const std::string& k) { return training ? false : (held_out_keyword(k) || !seen->contains(k)); };

        std::vector<std::string> pos;
        for (const std::string& k : f.inside) {
            if (k.size() < 3) continue;
            if (training && held_out_keyword(k)) continue;
            if (std::find(pos.begin(), pos.end(), k) == pos.end()) pos.push_back(k);
        }
        std::shuffle(pos.begin(), pos.end(), rng);
        const int n_pos = std::min<int>(training ? 1 : kw_pos, static_cast<int>(pos.size()));
        for (int i = 0; i < n_pos; ++i) out.push_back({wi, Question::Keyword, pos[i], 1, unseen(pos[i]), false});

        const int n_neg = training ? 1 : kw_pos;
        auto absent = [&](const std::string& k) { return keyword_label(f, k) == 0; };
        // Hard negatives: spelling neighbours of words in the window.
        std::vector<std::string> hard;
        for (const std::string& k : f.inside) {
            if (k.size() < 3) continue;
            for (const std::string& nb : neg_vocab.neighbours(k))
                if (absent(nb) && std::find(hard.begin(), hard.end(), nb) == hard.end()) hard.push_back(nb);
        }
        std::shuffle(hard.begin(), hard.end(), rng);
        const int n_hard = std::min<int>(n_neg, static_cast<int>(hard.size()));
        for (int i = 0; i < n_hard; ++i) out.push_back({wi, Question::Keyword, hard[i], 0, unseen(hard[i]), true});
        for (int i = 0, tries = 0; i < n_neg && tries < 50; ++tries) {
            const std::string& k = neg_vocab.sample(rng);
            if (!absent(k)) continue;
            out.push_back({wi, Question::Keyword, k, 0, unseen(k), false});
            ++i;
        }
        out.push_back({wi, Question::Speaking, {}, f.speaking ? 1 : 0, false, false});
        out.push_back({wi, Question::WordEnd, {}, f.word_end ? 1 : 0, false, false});
    }
    return out;
}

double auc(const std::vector<float>& score, const std::vector<int>& label) {
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < score.size(); ++i)
        if (label[i] == 0 || label[i] == 1) idx.push_back(i);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return score[a] < score[b]; });
    double rank_sum = 0, n_pos = 0, n_neg = 0;
    for (std::size_t i = 0; i < idx.size();) {
        std::size_t j = i;
        while (j < idx.size() && score[idx[j]] == score[idx[i]]) ++j;
        const double avg_rank = 0.5 * static_cast<double>(i + 1 + j);  // 1-based mean rank of the tie run
        for (std::size_t k = i; k < j; ++k) {
            if (label[idx[k]] == 1) {
                rank_sum += avg_rank;
                ++n_pos;
            } else {
                ++n_neg;
            }
        }
        i = j;
    }
    if (n_pos == 0 || n_neg == 0) return std::numeric_limits<double>::quiet_NaN();
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg);
}

WindowCache read_caches(const std::string& list) {
    WindowCache all;
    std::size_t s = 0;
    while (s <= list.size()) {
        std::size_t e = list.find(',', s);
        if (e == std::string::npos) e = list.size();
        if (e > s) {
            WindowCache c = read_cache(list.substr(s, e - s));
            if (all.windows.empty()) {
                all.window_s = c.window_s;
                all.dim = c.dim;
            } else if (c.window_s != all.window_s || c.dim != all.dim) {
                throw std::runtime_error("laya_audio: caches differ in window length or dim");
            }
            for (CachedWindow& w : c.windows) all.windows.push_back(std::move(w));
        }
        s = e + 1;
    }
    return all;
}

bt::Tensor upload_fp16_as_fp32(const uint16_t* bits, int rows, int cols) {
    const bt::Device dev = bt::default_device();
    bt::Tensor h = bt::Tensor::from_raw_bytes_on(dev, bits, rows, cols, bt::Dtype::FP16,
                                                 static_cast<std::size_t>(rows) * cols * sizeof(uint16_t));
    bt::Tensor out;
    bt::cast(h, out, bt::Dtype::FP32);
    return out;
}

bt::Tensor gather_latents(const WindowCache& cache, const std::vector<int>& windows) {
    const int F = cache.windows[static_cast<std::size_t>(windows.front())].frames;
    std::vector<uint16_t> bits(static_cast<std::size_t>(windows.size()) * F * cache.dim);
    for (std::size_t k = 0; k < windows.size(); ++k) {
        const CachedWindow& w = cache.windows[static_cast<std::size_t>(windows[k])];
        if (w.frames != F) throw std::runtime_error("laya_audio: windows with different frame counts in one batch");
        std::copy(w.lat.begin(), w.lat.end(), bits.begin() + static_cast<std::ptrdiff_t>(k * F * cache.dim));
    }
    return upload_fp16_as_fp32(bits.data(), static_cast<int>(windows.size()) * F, cache.dim);
}

std::vector<float> score_laya(brolm::laya::DecisionModel& model, ItemBuilder& builder, Mlp& proj,
                              const WindowCache& cache, const std::vector<Probe>& probes, int batch_windows) {
    std::vector<float> out(probes.size(), 0.0f);
    std::size_t p = 0;
    while (p < probes.size()) {
        // Next run of probes covering up to batch_windows distinct windows.
        std::vector<int> wins;
        std::size_t e = p;
        while (e < probes.size()) {
            if (wins.empty() || wins.back() != probes[e].window) {
                if (static_cast<int>(wins.size()) == batch_windows) break;
                wins.push_back(probes[e].window);
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
            if (probes[i].window != wins[k]) ++k;
            items.push_back(builder.item(question_text(probes[i].q, probes[i].keyword), F, static_cast<int>(k) * F));
        }
        const std::vector<brolm::laya::LayaItemLogits> r = model.forward_items(items, &soft);
        for (std::size_t i = p; i < e; ++i) out[i] = r[i - p].logits[1] - r[i - p].logits[0];
        p = e;
    }
    return out;
}

namespace {

bt::Tensor randn(int rows, int cols, float scale, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v(static_cast<std::size_t>(rows) * cols);
    for (float& x : v) x = nd(rng);
    return bt::Tensor::from_host_on(bt::default_device(), v.data(), rows, cols);
}

bt::Tensor zeros(int rows, int cols) { return bt::Tensor::zeros_on(bt::default_device(), rows, cols, bt::Dtype::FP32); }

}  // namespace

void Mlp::init(int d_in, int d_hidden, int d_out, uint64_t seed, float out_scale) {
    d_in_ = d_in;
    d_hidden_ = d_hidden;
    d_out_ = d_out;
    step_ = 0;
    std::mt19937 rng(static_cast<uint32_t>(seed));
    W1_ = randn(d_hidden, d_in, 1.0f / std::sqrt(static_cast<float>(d_in)), rng);
    b1_ = zeros(d_hidden, 1);
    W2_ = randn(d_out, d_hidden, out_scale / std::sqrt(static_cast<float>(d_hidden)), rng);
    b2_ = zeros(d_out, 1);
    gW1_ = zeros(d_hidden, d_in);
    gb1_ = zeros(d_hidden, 1);
    gW2_ = zeros(d_out, d_hidden);
    gb2_ = zeros(d_out, 1);
    const bt::Tensor* p[4] = {&W1_, &b1_, &W2_, &b2_};
    for (int i = 0; i < 4; ++i) {
        m_[i] = zeros(p[i]->rows, p[i]->cols);
        v_[i] = zeros(p[i]->rows, p[i]->cols);
    }
}

void Mlp::forward(const bt::Tensor& X, bt::Tensor& Y) {
    X_ = X;
    bt::linear_forward_batched(W1_, b1_, X_, pre_);
    bt::gelu_exact_forward(pre_, act_);
    bt::linear_forward_batched(W2_, b2_, act_, Y);
}

void Mlp::backward(const bt::Tensor& dY) {
    bt::Tensor d_act, d_pre, d_x;
    bt::linear_backward_batched(W2_, act_, dY, d_act, gW2_, gb2_);
    bt::gelu_exact_backward(pre_, d_act, d_pre);
    bt::linear_backward_batched(W1_, X_, d_pre, d_x, gW1_, gb1_);
}

void Mlp::zero_grad() {
    for (bt::Tensor* g : {&gW1_, &gb1_, &gW2_, &gb2_}) bt::scale_inplace(*g, 0.0f);
}

void Mlp::adam(float lr, float beta1, float beta2, float eps) {
    ++step_;
    bt::Tensor* p[4] = {&W1_, &b1_, &W2_, &b2_};
    bt::Tensor* g[4] = {&gW1_, &gb1_, &gW2_, &gb2_};
    for (int i = 0; i < 4; ++i) bt::adam_step(*p[i], *g[i], m_[i], v_[i], lr, beta1, beta2, eps, step_);
}

void Mlp::save(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("laya_audio: cannot write " + path);
    const int hdr[4] = {0x4d4c5031 /* 'MLP1' */, d_in_, d_hidden_, d_out_};
    std::fwrite(hdr, sizeof(int), 4, f);
    for (const bt::Tensor* t : {&W1_, &b1_, &W2_, &b2_}) {
        const std::vector<float> h = t->to_host_vector();
        std::fwrite(h.data(), sizeof(float), h.size(), f);
    }
    std::fclose(f);
}

void Mlp::load(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("laya_audio: cannot open " + path);
    int hdr[4] = {};
    if (std::fread(hdr, sizeof(int), 4, f) != 4 || hdr[0] != 0x4d4c5031) {
        std::fclose(f);
        throw std::runtime_error("laya_audio: not an MLP file: " + path);
    }
    init(hdr[1], hdr[2], hdr[3], 0);
    for (bt::Tensor* t : {&W1_, &b1_, &W2_, &b2_}) {
        std::vector<float> h(static_cast<std::size_t>(t->rows) * t->cols);
        if (std::fread(h.data(), sizeof(float), h.size(), f) != h.size()) {
            std::fclose(f);
            throw std::runtime_error("laya_audio: truncated MLP file: " + path);
        }
        *t = bt::Tensor::from_host_on(bt::default_device(), h.data(), t->rows, t->cols);
    }
    std::fclose(f);
}

}  // namespace laya_audio
