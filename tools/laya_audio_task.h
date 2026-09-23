#pragma once

// laya-audio: the task definition shared by the train / eval / listen tools.
//
//   * Labels. A streaming window is the last `window_s` seconds of audio at
//     t_end. From the word alignment of its utterance:
//       keyword "X"   1 if an occurrence of X lies fully inside the window,
//                     ignored if one straddles the window start or ends
//                     after t_end (partly heard), 0 otherwise;
//       speaking      1 if any word overlaps the last kTailS seconds;
//       word_end      1 if a word ended within the last kTailS seconds.
//   * Questions. Each label is a Laya noul question whose state is the
//     window's projected audio (soft rows); the instructions are fixed text.
//   * Keyword split. Words hashing to 0 mod 10 are held out: never asked in
//     training, so held-out AUC measures the open-vocabulary claim.
//   * Projector. A 2-layer GELU MLP with FP32 master weights and Adam (also
//     the dedicated-probe baseline, with a 1-wide output).

#include "laya_audio_data.h"

#include "brolm/laya.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace laya_audio {

constexpr float kTailS = 0.3f;  // "at the end of this audio" = the last 0.3 s

enum class Question { Keyword = 0, Speaking = 1, WordEnd = 2 };
const char* question_name(Question q);
std::string question_text(Question q, const std::string& keyword = {});

bool held_out_keyword(const std::string& w);

// What a window holds, from the alignment.
struct WindowFacts {
    std::vector<std::string> inside;          // words fully inside the window
    std::unordered_set<std::string> partial;  // words cut by a window edge
    bool speaking = false;
    bool word_end = false;
};
WindowFacts window_facts(const AlignedUtterance& u, float t_end, float window_s);
// 1 / 0 / -1 (ignore) for a keyword question.
int keyword_label(const WindowFacts& f, const std::string& w);

// Frequency-weighted keyword vocabulary (words of >= 3 characters seen at
// least `min_count` times), with spelling neighbours for hard negatives.
class Vocab {
public:
    void build(const std::vector<AlignedUtterance>& utts, int min_count, bool exclude_held_out);
    bool empty() const { return words_.empty(); }
    int size() const { return static_cast<int>(words_.size()); }
    bool contains(const std::string& w) const { return index_.count(w) != 0; }
    const std::string& sample(std::mt19937& rng) const;
    // Vocabulary words within edit distance 2 of w (not w itself), cached.
    const std::vector<std::string>& neighbours(const std::string& w);

private:
    std::vector<std::string> words_;
    std::vector<double> cum_;
    std::unordered_map<std::string, int> index_;
    std::unordered_map<std::string, std::vector<std::string>> nb_cache_;
};

int edit_distance(const std::string& a, const std::string& b, int cap);

// Laya items whose state is `n_soft` soft rows. The token sequence for an
// instruction string is built once ([CLS] question [SEP] markers [SEP]
// <n_soft placeholders> [SEP]) and cached; items point into the cache.
class ItemBuilder {
public:
    explicit ItemBuilder(brolm::laya::DecisionModel& m) : m_(m) {}
    brolm::laya::LayaItem item(const std::string& instructions, int n_soft, int soft_row);

private:
    struct Seq {
        std::vector<int32_t> ids, markers;
        int soft_pos = 0, qtype = 0;
    };
    brolm::laya::DecisionModel& m_;
    std::unordered_map<std::string, Seq> cache_;  // key: instructions + '\x1f' + n_soft (node-stable)
};

// One question about one cached window.
struct Probe {
    int window = 0;
    Question q = Question::Keyword;
    std::string keyword;
    int label = 0;
    bool unseen = false;    // keyword never asked in training (held out, or not in the train vocab)
    bool hard_neg = false;  // negative within edit distance 2 of a word in the window
};

// Probes for every window of a cache. Training (`seen` = null): one positive
// keyword when the window holds one, one hard and one random negative
// (training vocab, held-out words excluded), speaking and word_end. Eval:
// up to `kw_pos` positives and the same number each of hard and random
// negatives drawn from `neg_vocab` (all words, held-out included), with
// `unseen` marked against the training vocabulary `seen`. Negatives come
// from the vocabulary of the window's language (laya_audio_corpus.h).
class VocabSet;
std::vector<Probe> make_probes(const WindowCache& cache, const std::vector<AlignedUtterance>& utts,
                               const std::vector<int>& windows, VocabSet& neg_vocab, const Vocab* seen, int kw_pos,
                               std::mt19937& rng);

// Rank AUC of scores against 0/1 labels (ties count half). NaN if a class is empty.
double auc(const std::vector<float>& score, const std::vector<int>& label);

// y = W2 gelu(W1 x + b1) + b2, FP32 on the default device.
class Mlp {
public:
    void init(int d_in, int d_hidden, int d_out, uint64_t seed, float out_scale = 1.0f);
    int d_in() const { return d_in_; }
    int d_out() const { return d_out_; }
    // Keeps what backward needs.
    void forward(const brotensor::Tensor& X, brotensor::Tensor& Y);
    // Accumulates weight gradients of the last forward from dY.
    void backward(const brotensor::Tensor& dY);
    void zero_grad();
    void adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    int d_in_ = 0, d_hidden_ = 0, d_out_ = 0, step_ = 0;
    brotensor::Tensor W1_, b1_, W2_, b2_;
    brotensor::Tensor gW1_, gb1_, gW2_, gb2_;
    brotensor::Tensor m_[4], v_[4];
    brotensor::Tensor X_, pre_, act_;  // saved forward
};

// Gather windows' latents (FP16 bits, frames x dim each, all the same frame
// count) into one FP32 device tensor, window k at rows [k*frames, ...).
brotensor::Tensor gather_latents(const WindowCache& cache, const std::vector<int>& windows);

// Laya's answer to each probe (logit[true] - logit[false], pre-temperature)
// with the window projected through `proj`. Windows are processed
// `batch_windows` at a time through DecisionModel::forward_items.
std::vector<float> score_laya(brolm::laya::DecisionModel& model, ItemBuilder& builder, Mlp& proj,
                              const WindowCache& cache, const std::vector<Probe>& probes, int batch_windows = 32);

// read_cache over a comma-separated list of caches of one alignment file
// (same window length and dim), windows concatenated.
WindowCache read_caches(const std::string& comma_list);

// Upload FP16 bits as an FP32 (rows, cols) tensor on the default device.
brotensor::Tensor upload_fp16_as_fp32(const uint16_t* bits, int rows, int cols);

}  // namespace laya_audio
