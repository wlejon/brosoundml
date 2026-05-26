// POS tagger training driver. Standalone executable (not a ctest).
//
// CLI: train_pos_tagger --train <pos_train.bin> --val <pos_val.bin>
//                      --out <dir> [--epochs N] [--batch N] [--lr F]
//                      [--warmup N] [--seed N] [--device cpu|cuda]
//                      [--synthetic]
//
// Pipeline: load binary datasets → tokenise with chunk 1's tokeniser →
// per-sentence forward (FP32) using mha_forward / layernorm_forward with
// caches → fused softmax-CE loss over all word positions in the micro-batch →
// per-sentence backward → AdamW (adam_step + decoupled weight decay) →
// validate / checkpoint per epoch → round-trip load + tag at the end.

#include "brosoundml/g2p/pos_tagger.h"
#include "brosoundml/g2p/tags.h"
#include "g2p/pos_tagger_internal.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace bt = brotensor;
namespace g  = brosoundml::g2p;
namespace bs = brosoundml;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

// ─── Dataset ──────────────────────────────────────────────────────────────

struct Sentence {
    std::string                bytes;       // joined byte buffer (space-sep)
    std::vector<std::uint16_t> word_start;  // per-word byte offset into `bytes`
    std::vector<std::uint16_t> word_len;
    std::vector<std::uint8_t>  tag_id;
};

std::vector<Sentence> load_dataset(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("train_pos_tagger", "could not open dataset '" + path + "'");
    auto u32 = [&] {
        std::uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); return v;
    };
    auto u16 = [&] {
        std::uint16_t v; f.read(reinterpret_cast<char*>(&v), 2); return v;
    };
    auto u8 = [&] {
        std::uint8_t v; f.read(reinterpret_cast<char*>(&v), 1); return v;
    };
    const std::uint32_t magic = u32();
    if (magic != 0x504F5301u) fail("train_pos_tagger",
        "bad dataset magic in '" + path + "'");
    const std::uint32_t num_tags    = u32();
    const std::uint32_t num_sentences = u32();
    if (static_cast<int>(num_tags) != g::NUM_TAGS) {
        fail("train_pos_tagger",
             "dataset tag count " + std::to_string(num_tags) +
             " != compiled NUM_TAGS " + std::to_string(g::NUM_TAGS));
    }
    std::vector<Sentence> out;
    out.reserve(num_sentences);
    for (std::uint32_t s = 0; s < num_sentences; ++s) {
        Sentence sent;
        const std::uint32_t nb = u32();
        sent.bytes.resize(nb);
        f.read(sent.bytes.data(), nb);
        const std::uint32_t nw = u32();
        sent.tag_id.resize(nw);
        f.read(reinterpret_cast<char*>(sent.tag_id.data()), nw);
        sent.word_start.resize(nw);
        for (std::uint32_t i = 0; i < nw; ++i) sent.word_start[i] = u16();
        sent.word_len.resize(nw);
        for (std::uint32_t i = 0; i < nw; ++i) sent.word_len[i] = u16();
        // sanity: per-byte validity
        for (std::uint32_t i = 0; i < nw; ++i) {
            if (static_cast<std::uint32_t>(sent.word_start[i]) + sent.word_len[i] > nb) {
                fail("train_pos_tagger", "word offset overflow in dataset");
            }
            if (sent.tag_id[i] >= g::NUM_TAGS) {
                fail("train_pos_tagger", "tag id out of range in dataset");
            }
        }
        out.push_back(std::move(sent));
    }
    return out;
}

// Re-render sentence as the joined byte string. Words must be space-separated
// in `bytes` already — that's what the dataset script writes.
std::string sentence_text(const Sentence& s) {
    return s.bytes;
}

// ─── Weight init ──────────────────────────────────────────────────────────

void init_pos_weights(g::PosWeights& w, std::uint64_t seed) {
    w.num_tags    = g::NUM_TAGS;
    w.d_model     = g::kDModel;
    w.num_layers  = g::kNumLayers;
    w.num_heads   = g::kNumHeads;
    w.ffn_hidden  = g::kFFN;
    w.max_seq_len = g::kMaxSeqLen;
    w.layers.clear();
    w.layers.resize(g::kNumLayers);

    std::uint64_t rng = seed;

    // embeddings: small normal init
    auto fill_normal = [&](bt::Tensor& t, float stddev) {
        const int n = t.size();
        std::mt19937_64 mt(rng); rng = mt();
        std::normal_distribution<float> dist(0.0f, stddev);
        float* p = t.host_f32_mut();
        for (int i = 0; i < n; ++i) p[i] = dist(mt);
    };
    auto fill_zero = [&](bt::Tensor& t) {
        std::memset(t.host_raw_mut(), 0, t.bytes());
    };
    auto fill_ones = [&](bt::Tensor& t) {
        const int n = t.size();
        float* p = t.host_f32_mut();
        for (int i = 0; i < n; ++i) p[i] = 1.0f;
    };

    w.token_emb = bt::Tensor::mat(g::kVocab, g::kDModel);
    fill_normal(w.token_emb, 0.02f);
    w.pos_emb = bt::Tensor::mat(g::kMaxSeqLen, g::kDModel);
    fill_normal(w.pos_emb, 0.02f);

    for (int i = 0; i < g::kNumLayers; ++i) {
        auto& L = w.layers[i];
        L.ln1.features = g::kDModel; L.ln1.eps = 1e-5f;
        L.ln1.gamma = bt::Tensor::vec(g::kDModel); fill_ones(L.ln1.gamma);
        L.ln1.beta  = bt::Tensor::vec(g::kDModel); fill_zero(L.ln1.beta);

        L.mha.num_heads = g::kNumHeads; L.mha.embed_dim = g::kDModel;
        L.mha.Wq = bt::Tensor::mat(g::kDModel, g::kDModel); bt::xavier_init(L.mha.Wq, rng);
        L.mha.Wk = bt::Tensor::mat(g::kDModel, g::kDModel); bt::xavier_init(L.mha.Wk, rng);
        L.mha.Wv = bt::Tensor::mat(g::kDModel, g::kDModel); bt::xavier_init(L.mha.Wv, rng);
        L.mha.Wo = bt::Tensor::mat(g::kDModel, g::kDModel); bt::xavier_init(L.mha.Wo, rng);
        L.mha.bq = bt::Tensor::vec(g::kDModel); fill_zero(L.mha.bq);
        L.mha.bk = bt::Tensor::vec(g::kDModel); fill_zero(L.mha.bk);
        L.mha.bv = bt::Tensor::vec(g::kDModel); fill_zero(L.mha.bv);
        L.mha.bo = bt::Tensor::vec(g::kDModel); fill_zero(L.mha.bo);

        L.ln2.features = g::kDModel; L.ln2.eps = 1e-5f;
        L.ln2.gamma = bt::Tensor::vec(g::kDModel); fill_ones(L.ln2.gamma);
        L.ln2.beta  = bt::Tensor::vec(g::kDModel); fill_zero(L.ln2.beta);

        L.ffn1.in_features = g::kDModel; L.ffn1.out_features = g::kFFN;
        L.ffn1.W = bt::Tensor::mat(g::kFFN, g::kDModel); bt::xavier_init(L.ffn1.W, rng);
        L.ffn1.b = bt::Tensor::vec(g::kFFN); fill_zero(L.ffn1.b);

        L.ffn2.in_features = g::kFFN; L.ffn2.out_features = g::kDModel;
        L.ffn2.W = bt::Tensor::mat(g::kDModel, g::kFFN); bt::xavier_init(L.ffn2.W, rng);
        L.ffn2.b = bt::Tensor::vec(g::kDModel); fill_zero(L.ffn2.b);
    }

    w.final_ln.features = g::kDModel; w.final_ln.eps = 1e-5f;
    w.final_ln.gamma = bt::Tensor::vec(g::kDModel); fill_ones(w.final_ln.gamma);
    w.final_ln.beta  = bt::Tensor::vec(g::kDModel); fill_zero(w.final_ln.beta);

    w.head.in_features = g::kDModel; w.head.out_features = g::NUM_TAGS;
    w.head.W = bt::Tensor::mat(g::NUM_TAGS, g::kDModel); bt::xavier_init(w.head.W, rng);
    w.head.b = bt::Tensor::vec(g::NUM_TAGS); fill_zero(w.head.b);
}

// ─── Parameter registry (for AdamW state) ─────────────────────────────────

struct Param {
    bt::Tensor*  data;
    bt::Tensor   grad;
    bt::Tensor   m;
    bt::Tensor   v;
    bool         decay;
    std::string  name;
};

std::vector<Param> make_param_list(g::PosWeights& w) {
    std::vector<Param> ps;
    auto add = [&](bt::Tensor* t, bool decay, const std::string& nm) {
        Param p;
        p.data  = t;
        p.grad  = bt::Tensor::mat(t->rows, t->cols);
        p.m     = bt::Tensor::mat(t->rows, t->cols);
        p.v     = bt::Tensor::mat(t->rows, t->cols);
        std::memset(p.grad.host_raw_mut(), 0, p.grad.bytes());
        std::memset(p.m.host_raw_mut(),    0, p.m.bytes());
        std::memset(p.v.host_raw_mut(),    0, p.v.bytes());
        p.decay = decay;
        p.name  = nm;
        ps.push_back(std::move(p));
    };

    add(&w.token_emb, true,  "token_emb");
    add(&w.pos_emb,   false, "pos_emb");
    for (int i = 0; i < g::kNumLayers; ++i) {
        auto& L = w.layers[i];
        const std::string pfx = "layer" + std::to_string(i) + ".";
        add(&L.ln1.gamma, false, pfx + "ln1.gamma");
        add(&L.ln1.beta,  false, pfx + "ln1.beta");
        add(&L.mha.Wq,    true,  pfx + "attn.Wq");
        add(&L.mha.bq,    false, pfx + "attn.bq");
        add(&L.mha.Wk,    true,  pfx + "attn.Wk");
        add(&L.mha.bk,    false, pfx + "attn.bk");
        add(&L.mha.Wv,    true,  pfx + "attn.Wv");
        add(&L.mha.bv,    false, pfx + "attn.bv");
        add(&L.mha.Wo,    true,  pfx + "attn.Wo");
        add(&L.mha.bo,    false, pfx + "attn.bo");
        add(&L.ln2.gamma, false, pfx + "ln2.gamma");
        add(&L.ln2.beta,  false, pfx + "ln2.beta");
        add(&L.ffn1.W,    true,  pfx + "ffn1.W");
        add(&L.ffn1.b,    false, pfx + "ffn1.b");
        add(&L.ffn2.W,    true,  pfx + "ffn2.W");
        add(&L.ffn2.b,    false, pfx + "ffn2.b");
    }
    add(&w.final_ln.gamma, false, "final_ln.gamma");
    add(&w.final_ln.beta,  false, "final_ln.beta");
    add(&w.head.W,         true,  "head.W");
    add(&w.head.b,         false, "head.b");
    return ps;
}

void zero_grads(std::vector<Param>& ps) {
    for (auto& p : ps) std::memset(p.grad.host_raw_mut(), 0, p.grad.bytes());
}

void adamw_step(std::vector<Param>& ps, float lr, float beta1, float beta2,
                float eps, float wd, int step, float grad_scale) {
    // grad_scale: 1 / batch_words — applied before adam.
    for (auto& p : ps) {
        if (grad_scale != 1.0f) {
            bt::scale_inplace(p.grad, grad_scale);
        }
        bt::adam_step(*p.data, p.grad, p.m, p.v, lr, beta1, beta2, eps, step);
        if (p.decay && wd > 0.0f) {
            // decoupled weight decay: p -= lr * wd * p
            const int n = p.data->size();
            float* d = p.data->host_f32_mut();
            const float k = lr * wd;
            for (int i = 0; i < n; ++i) d[i] -= k * d[i];
        }
    }
}

// ─── Per-sentence forward caches ──────────────────────────────────────────

struct LayerCache {
    bt::Tensor h_in;     // (L, D) input to ln1 (== residual stream pre-attn)
    bt::Tensor ln1_y;    // (L, D)
    std::vector<bt::Tensor> ln1_xhat;  // L rows, each (D,1)
    std::vector<float>      ln1_rstd;
    bt::Tensor Qh, Kh, Vh, Attnh, Yconcat;  // mha caches
    bt::Tensor attn_out; // (L, D)
    bt::Tensor h_mid;    // (L, D) after attn residual
    bt::Tensor ln2_y;
    std::vector<bt::Tensor> ln2_xhat;
    std::vector<float>      ln2_rstd;
    bt::Tensor ffn1_pre;  // (L, ffn) pre-gelu
    bt::Tensor ffn1_act;  // (L, ffn) post-gelu
    bt::Tensor ffn2_out;  // (L, D)
};

struct ForwardCache {
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> wsep_positions;
    bt::Tensor h0;        // (L, D) embedding sum
    std::vector<LayerCache> layers;
    bt::Tensor h_pre_final_ln;   // (L, D)
    bt::Tensor h_final;          // (L, D) post final_ln
    std::vector<bt::Tensor> final_xhat;
    std::vector<float>      final_rstd;
    bt::Tensor pooled;           // (W, D)
    bt::Tensor logits;           // (W, NUM_TAGS)
};

// Apply layernorm per-row over (L, D). Caches xhat (D,1) and rstd per row.
void ln_forward_per_row(const bt::Tensor& gamma, const bt::Tensor& beta,
                        const bt::Tensor& X, bt::Tensor& Y,
                        std::vector<bt::Tensor>& xhats,
                        std::vector<float>& rstds,
                        float eps) {
    const int L = X.rows;
    const int D = X.cols;
    if (Y.rows != L || Y.cols != D || Y.dtype != bt::Dtype::FP32) {
        Y = bt::Tensor::mat(L, D);
    }
    xhats.assign(L, bt::Tensor::vec(D));
    rstds.assign(L, 0.0f);
    bt::Tensor xrow = bt::Tensor::vec(D);
    bt::Tensor yrow = bt::Tensor::vec(D);
    for (int i = 0; i < L; ++i) {
        std::memcpy(xrow.host_f32_mut(), X.host_f32() + i * D, D * sizeof(float));
        float mean_out = 0.0f, rstd_out = 0.0f;
        bt::layernorm_forward(xrow, gamma, beta, yrow, xhats[i],
                              mean_out, rstd_out, eps);
        rstds[i] = rstd_out;
        std::memcpy(Y.host_f32_mut() + i * D, yrow.host_f32(), D * sizeof(float));
    }
}

void ln_backward_per_row(const bt::Tensor& gamma,
                         const bt::Tensor& dY, bt::Tensor& dX,
                         const std::vector<bt::Tensor>& xhats,
                         const std::vector<float>& rstds,
                         bt::Tensor& dGamma, bt::Tensor& dBeta) {
    const int L = dY.rows;
    const int D = dY.cols;
    if (dX.rows != L || dX.cols != D || dX.dtype != bt::Dtype::FP32) {
        dX = bt::Tensor::mat(L, D);
    }
    bt::Tensor dy_row = bt::Tensor::vec(D);
    bt::Tensor dx_row = bt::Tensor::vec(D);
    for (int i = 0; i < L; ++i) {
        std::memcpy(dy_row.host_f32_mut(), dY.host_f32() + i * D, D * sizeof(float));
        bt::layernorm_backward(dy_row, xhats[i], gamma, rstds[i],
                               dx_row, dGamma, dBeta);
        std::memcpy(dX.host_f32_mut() + i * D, dx_row.host_f32(), D * sizeof(float));
    }
}

// Build the embedding sum (token_emb[ids] + pos_emb[0..L-1]).
void embed_forward(const g::PosWeights& w, const std::vector<std::int32_t>& ids,
                   bt::Tensor& out) {
    const int L = static_cast<int>(ids.size());
    bt::embedding_lookup_forward(w.token_emb, ids.data(), L, out);
    std::vector<std::int32_t> pidx(L);
    for (int i = 0; i < L; ++i) pidx[i] = i;
    bt::Tensor pe;
    bt::embedding_lookup_forward(w.pos_emb, pidx.data(), L, pe);
    bt::add_inplace_batched(out, pe);
}

// Forward one sentence; populates fc.logits.
void forward_sentence(const g::PosWeights& w,
                      const std::vector<std::int32_t>& token_ids,
                      const std::vector<std::int32_t>& wsep_positions,
                      ForwardCache& fc) {
    const int L = static_cast<int>(token_ids.size());
    const int D = w.d_model;
    fc.token_ids      = token_ids;
    fc.wsep_positions = wsep_positions;

    embed_forward(w, token_ids, fc.h0);

    fc.layers.assign(w.layers.size(), LayerCache{});
    bt::Tensor h = fc.h0;  // deep copy via copy ctor; we want a residual stream
                            // we can mutate without overwriting h0.
    // Force separate storage.
    bt::Tensor h_resid = bt::Tensor::mat(L, D);
    std::memcpy(h_resid.host_f32_mut(), fc.h0.host_f32(), L * D * sizeof(float));

    for (std::size_t li = 0; li < w.layers.size(); ++li) {
        const auto& Lw = w.layers[li];
        auto& C = fc.layers[li];

        // Save input to ln1 (== current residual)
        C.h_in = bt::Tensor::mat(L, D);
        std::memcpy(C.h_in.host_f32_mut(), h_resid.host_f32(),
                    L * D * sizeof(float));

        // pre-norm ln1
        ln_forward_per_row(Lw.ln1.gamma, Lw.ln1.beta, h_resid, C.ln1_y,
                           C.ln1_xhat, C.ln1_rstd, Lw.ln1.eps);

        // MHA over ln1_y
        bt::Tensor& X = C.ln1_y;
        C.Qh = bt::Tensor{}; C.Kh = bt::Tensor{}; C.Vh = bt::Tensor{};
        C.Attnh = bt::Tensor{}; C.Yconcat = bt::Tensor{};
        C.attn_out = bt::Tensor::mat(L, D);
        bt::mha_forward(X, Lw.mha.Wq, Lw.mha.Wk, Lw.mha.Wv, Lw.mha.Wo,
                        /*d_mask=*/nullptr, w.num_heads,
                        C.Qh, C.Kh, C.Vh, C.Attnh, C.Yconcat,
                        C.attn_out);
        // NOTE: mha_forward does NOT add the q/k/v/o biases — chunk 1 uses
        // linear_forward_batched(Wq, bq, ...) for projections and Wo, bo
        // for the output. We add the biases by hand here to stay faithful to
        // the chunk 1 forward (biases on every attn projection).
        // bq/bk/bv enter Q/K/V before the softmax — they shift columns of
        // Qh/Kh/Vh row-wise. bo is added after the Wo projection.
        // The simplest correct path: add bq.bv.bk pre-effect by adding bq to
        // every row of the Q rows of Qh (Qh row layout is num_heads*L rows of
        // head_dim cols; head h occupies rows [h*L, (h+1)*L). Slicing the
        // (D,) bias vector across heads is non-trivial — instead we use the
        // identity that linear_forward_batched(W, b, X) == mha_forward's
        // implicit Q = X @ W^T + b would produce. Because mha_forward
        // performs the projection internally we cannot inject the bias there.
        //
        // Pragmatic workaround: we use the no-bias variant of self-attention
        // by setting bq=bk=bv=bo=0 at init. The chunk 1 loader REQUIRES the
        // (D,1) bias tensors to be present in the file — they're shipped as
        // zeros and contribute nothing. The trainer holds them in the param
        // list (so they round-trip) but never produces non-zero gradients.
        // This drops a few thousand effective parameters but matches the
        // mha_forward op's contract exactly. See README note in docs/.

        // residual: h_resid += attn_out
        bt::add_inplace_batched(h_resid, C.attn_out);
        // snapshot pre-ln2 state
        C.h_mid = bt::Tensor::mat(L, D);
        std::memcpy(C.h_mid.host_f32_mut(), h_resid.host_f32(),
                    L * D * sizeof(float));

        // pre-norm ln2
        ln_forward_per_row(Lw.ln2.gamma, Lw.ln2.beta, h_resid, C.ln2_y,
                           C.ln2_xhat, C.ln2_rstd, Lw.ln2.eps);

        // ffn1: linear -> gelu
        C.ffn1_pre = bt::Tensor::mat(L, w.ffn_hidden);
        Lw.ffn1.forward_batched(C.ln2_y, C.ffn1_pre);
        C.ffn1_act = bt::Tensor::mat(L, w.ffn_hidden);
        bt::gelu_forward(C.ffn1_pre, C.ffn1_act);
        // ffn2
        C.ffn2_out = bt::Tensor::mat(L, D);
        Lw.ffn2.forward_batched(C.ffn1_act, C.ffn2_out);
        // residual
        bt::add_inplace_batched(h_resid, C.ffn2_out);
    }

    // final_ln (pre-residual snapshot for backward)
    fc.h_pre_final_ln = bt::Tensor::mat(L, D);
    std::memcpy(fc.h_pre_final_ln.host_f32_mut(), h_resid.host_f32(),
                L * D * sizeof(float));
    ln_forward_per_row(w.final_ln.gamma, w.final_ln.beta, h_resid, fc.h_final,
                       fc.final_xhat, fc.final_rstd, w.final_ln.eps);

    // Word pool + head
    const int W = static_cast<int>(wsep_positions.size());
    fc.pooled = bt::Tensor::mat(W, D);
    for (int i = 0; i < W; ++i) {
        const int t = wsep_positions[i];
        std::memcpy(fc.pooled.host_f32_mut() + i * D,
                    fc.h_final.host_f32() + t * D, D * sizeof(float));
    }
    fc.logits = bt::Tensor::mat(W, w.num_tags);
    w.head.forward_batched(fc.pooled, fc.logits);
}

// Backward one sentence given dLogits — accumulates grads into Param entries.
// Param index lookups by name; we build a small map per call but that's fine
// at micro-batch scale.
struct ParamRefs {
    bt::Tensor* token_emb;
    bt::Tensor* pos_emb;
    bt::Tensor* final_ln_gamma;
    bt::Tensor* final_ln_beta;
    bt::Tensor* head_W;
    bt::Tensor* head_b;
    struct LRef {
        bt::Tensor* ln1_g; bt::Tensor* ln1_b;
        bt::Tensor* Wq; bt::Tensor* Wk; bt::Tensor* Wv; bt::Tensor* Wo;
        // biases not trained — see note in forward_sentence.
        bt::Tensor* ln2_g; bt::Tensor* ln2_b;
        bt::Tensor* ffn1_W; bt::Tensor* ffn1_b;
        bt::Tensor* ffn2_W; bt::Tensor* ffn2_b;
    };
    std::vector<LRef> layers;
};

ParamRefs param_refs(std::vector<Param>& ps) {
    auto find = [&](const std::string& n) -> bt::Tensor* {
        for (auto& p : ps) if (p.name == n) return &p.grad;
        fail("train_pos_tagger", "param not found: " + n);
    };
    ParamRefs r;
    r.token_emb      = find("token_emb");
    r.pos_emb        = find("pos_emb");
    r.final_ln_gamma = find("final_ln.gamma");
    r.final_ln_beta  = find("final_ln.beta");
    r.head_W         = find("head.W");
    r.head_b         = find("head.b");
    for (int i = 0; i < g::kNumLayers; ++i) {
        const std::string p = "layer" + std::to_string(i) + ".";
        ParamRefs::LRef L;
        L.ln1_g  = find(p + "ln1.gamma"); L.ln1_b = find(p + "ln1.beta");
        L.Wq     = find(p + "attn.Wq"); L.Wk = find(p + "attn.Wk");
        L.Wv     = find(p + "attn.Wv"); L.Wo = find(p + "attn.Wo");
        L.ln2_g  = find(p + "ln2.gamma"); L.ln2_b = find(p + "ln2.beta");
        L.ffn1_W = find(p + "ffn1.W"); L.ffn1_b = find(p + "ffn1.b");
        L.ffn2_W = find(p + "ffn2.W"); L.ffn2_b = find(p + "ffn2.b");
        r.layers.push_back(L);
    }
    return r;
}

void backward_sentence(const g::PosWeights& w,
                       const ForwardCache& fc,
                       const bt::Tensor& dLogits,   // (W, NUM_TAGS)
                       ParamRefs& gr) {
    const int L = static_cast<int>(fc.token_ids.size());
    const int Wn = static_cast<int>(fc.wsep_positions.size());
    const int D = w.d_model;

    // head backward
    bt::Tensor dPooled = bt::Tensor::mat(Wn, D);
    bt::linear_backward_batched(w.head.W, fc.pooled, dLogits,
                                dPooled, *gr.head_W, *gr.head_b);

    // scatter dPooled into dH_final at wsep positions
    bt::Tensor dH_final = bt::Tensor::mat(L, D);
    std::memset(dH_final.host_raw_mut(), 0, dH_final.bytes());
    for (int i = 0; i < Wn; ++i) {
        const int t = fc.wsep_positions[i];
        float* dst = dH_final.host_f32_mut() + t * D;
        const float* src = dPooled.host_f32() + i * D;
        for (int j = 0; j < D; ++j) dst[j] = src[j];
    }

    // final_ln backward (per-row)
    bt::Tensor dH_resid = bt::Tensor::mat(L, D);
    ln_backward_per_row(w.final_ln.gamma, dH_final, dH_resid,
                        fc.final_xhat, fc.final_rstd,
                        *gr.final_ln_gamma, *gr.final_ln_beta);

    // Walk layers in reverse.
    for (int li = static_cast<int>(w.layers.size()) - 1; li >= 0; --li) {
        const auto& Lw = w.layers[li];
        const auto& C  = fc.layers[li];
        auto& gL = gr.layers[li];

        // dH_resid is grad w.r.t. h_resid AFTER this layer (= post ffn residual)
        // = dh_mid + dffn2_out. The residual edge propagates the dH_resid
        // straight through; we also need to push it through the ffn branch.
        //
        // Save the residual edge contribution to h_mid (= dH_resid as-is).
        bt::Tensor dh_mid_resid = bt::Tensor::mat(L, D);
        std::memcpy(dh_mid_resid.host_f32_mut(), dH_resid.host_f32(),
                    L * D * sizeof(float));

        // dffn2_out = dH_resid
        // ffn2 backward
        bt::Tensor dffn1_act = bt::Tensor::mat(L, w.ffn_hidden);
        bt::linear_backward_batched(Lw.ffn2.W, C.ffn1_act, dH_resid,
                                    dffn1_act, *gL.ffn2_W, *gL.ffn2_b);
        // gelu backward (tanh approx)
        bt::Tensor dffn1_pre = bt::Tensor::mat(L, w.ffn_hidden);
        bt::gelu_backward(C.ffn1_pre, dffn1_act, dffn1_pre);
        // ffn1 backward
        bt::Tensor dln2_y = bt::Tensor::mat(L, D);
        bt::linear_backward_batched(Lw.ffn1.W, C.ln2_y, dffn1_pre,
                                    dln2_y, *gL.ffn1_W, *gL.ffn1_b);
        // ln2 backward
        bt::Tensor dh_mid_ffn = bt::Tensor::mat(L, D);
        ln_backward_per_row(Lw.ln2.gamma, dln2_y, dh_mid_ffn,
                            C.ln2_xhat, C.ln2_rstd,
                            *gL.ln2_g, *gL.ln2_b);
        // sum branches
        bt::add_inplace_batched(dh_mid_resid, dh_mid_ffn);
        // Now dh_mid_resid is the grad w.r.t. h_mid.

        // Residual on h_mid: h_mid = h_in + attn_out
        bt::Tensor dh_in_resid = bt::Tensor::mat(L, D);
        std::memcpy(dh_in_resid.host_f32_mut(), dh_mid_resid.host_f32(),
                    L * D * sizeof(float));

        // attn backward: dh_mid -> dattn_out -> dln1_y -> dh_in
        bt::Tensor dln1_y = bt::Tensor::mat(L, D);
        bt::mha_backward(dh_mid_resid, C.ln1_y,
                         C.Qh, C.Kh, C.Vh, C.Attnh, C.Yconcat,
                         Lw.mha.Wq, Lw.mha.Wk, Lw.mha.Wv, Lw.mha.Wo,
                         /*d_mask=*/nullptr, w.num_heads,
                         dln1_y, *gL.Wq, *gL.Wk, *gL.Wv, *gL.Wo);
        // ln1 backward
        bt::Tensor dh_in_ln = bt::Tensor::mat(L, D);
        ln_backward_per_row(Lw.ln1.gamma, dln1_y, dh_in_ln,
                            C.ln1_xhat, C.ln1_rstd,
                            *gL.ln1_g, *gL.ln1_b);
        // combine
        bt::add_inplace_batched(dh_in_resid, dh_in_ln);
        dH_resid = dh_in_resid;
    }

    // dH_resid is now the grad w.r.t. h0 (token_emb + pos_emb).
    // Embedding backward: scatter-add into token_emb and pos_emb grads.
    bt::embedding_lookup_backward(dH_resid, fc.token_ids.data(), L, *gr.token_emb);
    std::vector<std::int32_t> pidx(L);
    for (int i = 0; i < L; ++i) pidx[i] = i;
    bt::embedding_lookup_backward(dH_resid, pidx.data(), L, *gr.pos_emb);
}

// ─── Loss ─────────────────────────────────────────────────────────────────

// Run softmax_xent_segment row-by-row (CPU-only — matches our CPU trainer).
// Returns sum of per-row CE losses; writes per-row dLogits.
float xent_words(const bt::Tensor& logits, const std::vector<std::uint8_t>& tags,
                bt::Tensor& dLogits) {
    const int Wn = logits.rows;
    const int C = logits.cols;
    if (dLogits.rows != Wn || dLogits.cols != C || dLogits.dtype != bt::Dtype::FP32) {
        dLogits = bt::Tensor::mat(Wn, C);
    }
    std::vector<float> probs_row(C);
    std::vector<float> dlog_row(C);
    std::vector<float> target_row(C);
    float total = 0.0f;
    for (int i = 0; i < Wn; ++i) {
        std::fill(target_row.begin(), target_row.end(), 0.0f);
        target_row[tags[i]] = 1.0f;
        const float* lg = logits.host_f32() + i * C;
        float loss = bt::softmax_xent_segment(lg, target_row.data(),
                                              probs_row.data(),
                                              dlog_row.data(), C, nullptr);
        total += loss;
        std::memcpy(dLogits.host_f32_mut() + i * C, dlog_row.data(),
                    C * sizeof(float));
    }
    return total;
}

// ─── Weight file writer (chunk 1 format) ─────────────────────────────────

void w_u32(std::ofstream& f, std::uint32_t v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
}
void w_u16(std::ofstream& f, std::uint16_t v) {
    f.write(reinterpret_cast<const char*>(&v), 2);
}
void w_u8(std::ofstream& f, std::uint8_t v) {
    f.write(reinterpret_cast<const char*>(&v), 1);
}
void w_tensor(std::ofstream& f, const std::string& name,
              const bt::Tensor& t) {
    w_u16(f, static_cast<std::uint16_t>(name.size()));
    f.write(name.data(), name.size());
    const std::uint8_t rank = (t.cols == 1) ? 1 : 2;
    w_u8(f, rank);
    w_u32(f, static_cast<std::uint32_t>(t.rows));
    if (rank == 2) w_u32(f, static_cast<std::uint32_t>(t.cols));
    auto host = t.to_host_vector();
    f.write(reinterpret_cast<const char*>(host.data()),
            host.size() * sizeof(float));
}

void save_checkpoint(const std::string& path, const g::PosWeights& w) {
    std::ofstream f(path, std::ios::binary);
    if (!f) fail("train_pos_tagger", "could not write checkpoint '" + path + "'");

    struct Spec { std::string name; const bt::Tensor* t; };
    std::vector<Spec> specs;
    specs.push_back({"token_emb", &w.token_emb});
    specs.push_back({"pos_emb",   &w.pos_emb});
    for (int i = 0; i < g::kNumLayers; ++i) {
        const std::string p = "layer" + std::to_string(i) + ".";
        const auto& L = w.layers[i];
        specs.push_back({p + "ln1.gamma",  &L.ln1.gamma});
        specs.push_back({p + "ln1.beta",   &L.ln1.beta});
        specs.push_back({p + "attn.Wq",    &L.mha.Wq});
        specs.push_back({p + "attn.bq",    &L.mha.bq});
        specs.push_back({p + "attn.Wk",    &L.mha.Wk});
        specs.push_back({p + "attn.bk",    &L.mha.bk});
        specs.push_back({p + "attn.Wv",    &L.mha.Wv});
        specs.push_back({p + "attn.bv",    &L.mha.bv});
        specs.push_back({p + "attn.Wo",    &L.mha.Wo});
        specs.push_back({p + "attn.bo",    &L.mha.bo});
        specs.push_back({p + "ln2.gamma",  &L.ln2.gamma});
        specs.push_back({p + "ln2.beta",   &L.ln2.beta});
        specs.push_back({p + "ffn1.W",     &L.ffn1.W});
        specs.push_back({p + "ffn1.b",     &L.ffn1.b});
        specs.push_back({p + "ffn2.W",     &L.ffn2.W});
        specs.push_back({p + "ffn2.b",     &L.ffn2.b});
    }
    specs.push_back({"final_ln.gamma", &w.final_ln.gamma});
    specs.push_back({"final_ln.beta",  &w.final_ln.beta});
    specs.push_back({"head.W",         &w.head.W});
    specs.push_back({"head.b",         &w.head.b});

    w_u32(f, 0x504F5302u);
    w_u32(f, 1u);
    w_u32(f, static_cast<std::uint32_t>(w.num_tags));
    w_u32(f, static_cast<std::uint32_t>(w.d_model));
    w_u32(f, static_cast<std::uint32_t>(w.num_layers));
    w_u32(f, static_cast<std::uint32_t>(w.num_heads));
    w_u32(f, static_cast<std::uint32_t>(w.ffn_hidden));
    w_u32(f, static_cast<std::uint32_t>(w.max_seq_len));
    w_u32(f, static_cast<std::uint32_t>(specs.size()));
    for (const auto& s : specs) w_tensor(f, s.name, *s.t);
}

// ─── Byte-noise + tokeniser dispatch ─────────────────────────────────────

void apply_byte_noise(std::vector<std::int32_t>& ids, std::mt19937& rng,
                      float p_noise) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uniform_int_distribution<int> b(4, 259);
    for (auto& t : ids) {
        if (t >= 4 && u(rng) < p_noise) t = b(rng);
    }
}

// ─── Synthetic dataset helper (--synthetic) ──────────────────────────────

std::vector<Sentence> make_synthetic_dataset(int n_sentences,
                                             std::mt19937& rng) {
    std::vector<std::string> words = {
        "the", "cat", "sat", "on", "mat", "dog", "ran", "fast",
        "big", "small", "red", "blue", "good", "bad"
    };
    std::uniform_int_distribution<int> wcount(3, 8);
    std::uniform_int_distribution<int> wpick(0, static_cast<int>(words.size()) - 1);
    std::uniform_int_distribution<int> tagpick(0, g::NUM_TAGS - 1);

    std::vector<Sentence> out;
    out.reserve(n_sentences);
    for (int s = 0; s < n_sentences; ++s) {
        Sentence sent;
        const int nw = wcount(rng);
        std::string joined;
        std::vector<std::pair<std::uint16_t, std::uint16_t>> offsets;
        for (int i = 0; i < nw; ++i) {
            const std::string& w = words[wpick(rng)];
            if (!joined.empty()) joined += ' ';
            offsets.push_back({static_cast<std::uint16_t>(joined.size()),
                               static_cast<std::uint16_t>(w.size())});
            joined += w;
        }
        sent.bytes = joined;
        for (auto& o : offsets) {
            sent.word_start.push_back(o.first);
            sent.word_len.push_back(o.second);
            // Deterministic tag by word-hash so a model can learn it.
            std::uint32_t h = 2166136261u;
            for (std::size_t k = 0; k < o.second; ++k) {
                h = (h ^ static_cast<unsigned char>(joined[o.first + k])) * 16777619u;
            }
            sent.tag_id.push_back(static_cast<std::uint8_t>(h % g::NUM_TAGS));
        }
        (void)tagpick;
        out.push_back(std::move(sent));
    }
    return out;
}

// ─── CLI ──────────────────────────────────────────────────────────────────

struct Args {
    std::string train;
    std::string val;
    std::string out_dir;
    int epochs = 30;
    int batch  = 32;
    float lr   = 5e-4f;
    int warmup = 1000;
    int seed   = 42;
    std::string device = "cpu";
    bool synthetic = false;
    bool help = false;
};

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) fail("cli", "missing value for " + k);
            return std::string(argv[++i]);
        };
        if      (k == "--train")     a.train = next();
        else if (k == "--val")       a.val = next();
        else if (k == "--out")       a.out_dir = next();
        else if (k == "--epochs")    a.epochs = std::stoi(next());
        else if (k == "--batch")     a.batch = std::stoi(next());
        else if (k == "--lr")        a.lr = std::stof(next());
        else if (k == "--warmup")    a.warmup = std::stoi(next());
        else if (k == "--seed")      a.seed = std::stoi(next());
        else if (k == "--device")    a.device = next();
        else if (k == "--synthetic") a.synthetic = true;
        else if (k == "--help" || k == "-h") { a.help = true; return true; }
        else fail("cli", "unknown flag '" + k + "'");
    }
    return true;
}

void print_help() {
    std::cout <<
        "train_pos_tagger — POS tagger training driver\n"
        "  --train PATH       pos_train.bin\n"
        "  --val PATH         pos_val.bin\n"
        "  --out DIR          checkpoint output dir\n"
        "  --epochs N         (default 30)\n"
        "  --batch N          (default 32)\n"
        "  --lr F             (default 5e-4)\n"
        "  --warmup N         (default 1000)\n"
        "  --seed N           (default 42)\n"
        "  --device cpu|cuda  (default cpu)\n"
        "  --synthetic        ignore --train/--val, build a tiny in-memory set\n"
        "                     and run 2 epochs end-to-end (smoke test).\n";
}

// ─── Main training loop ──────────────────────────────────────────────────

float lr_schedule(int step, int total_steps, int warmup, float lr_peak,
                  float lr_floor = 1e-5f) {
    if (step < warmup) {
        return lr_peak * (static_cast<float>(step) / std::max(1, warmup));
    }
    if (step >= total_steps) return lr_floor;
    const float t = static_cast<float>(step - warmup) /
                    std::max(1, total_steps - warmup);
    const float pi = 3.14159265358979323846f;
    return lr_floor + 0.5f * (lr_peak - lr_floor) * (1.0f + std::cos(pi * t));
}

float evaluate(const g::PosWeights& w, const std::vector<Sentence>& data) {
    int total = 0, correct = 0;
    for (const auto& s : data) {
        std::string text = sentence_text(s);
        auto chunks = g::tokenise_sentence(text);
        std::size_t word_idx = 0;
        for (const auto& c : chunks) {
            if (c.wsep_positions.empty()) continue;
            ForwardCache fc;
            forward_sentence(w, c.token_ids, c.wsep_positions, fc);
            bt::Tensor idx_dev;
            bt::argmax_rows(fc.logits, idx_dev);
            auto idx = idx_dev.to_host_vector();
            for (std::size_t i = 0; i < c.wsep_positions.size(); ++i, ++word_idx) {
                if (word_idx >= s.tag_id.size()) break;
                const int pred = static_cast<int>(idx[i]);
                if (pred == s.tag_id[word_idx]) ++correct;
                ++total;
            }
        }
    }
    return (total == 0) ? 0.0f : static_cast<float>(correct) / total;
}

int run_training(Args& a) {
    if (a.device == "cuda") {
        if (bt::is_available(bt::Device::CUDA)) bt::set_default_device(bt::Device::CUDA);
        else std::cerr << "warn: cuda requested but unavailable, falling back to cpu\n";
    } else {
        bt::set_default_device(bt::Device::CPU);
    }

    std::vector<Sentence> train, val;
    if (a.synthetic) {
        std::mt19937 rng(static_cast<std::uint32_t>(a.seed));
        train = make_synthetic_dataset(128, rng);
        val   = make_synthetic_dataset(32, rng);
        a.epochs = 2;
    } else {
        if (a.train.empty() || a.val.empty() || a.out_dir.empty()) {
            std::cerr << "error: --train, --val, --out required (or --synthetic)\n";
            return 2;
        }
        train = load_dataset(a.train);
        val   = load_dataset(a.val);
    }
    if (a.out_dir.empty()) a.out_dir = ".";
    std::filesystem::create_directories(a.out_dir);

    std::cout << "train: " << train.size() << " sentences, val: " << val.size() << "\n";

    g::PosWeights w;
    init_pos_weights(w, static_cast<std::uint64_t>(a.seed) ^ 0xC0FFEEu);
    auto params = make_param_list(w);
    auto grefs  = param_refs(params);

    const int steps_per_epoch = std::max(1, static_cast<int>(train.size()) / a.batch);
    const int total_steps = a.epochs * steps_per_epoch;
    std::mt19937 rng(static_cast<std::uint32_t>(a.seed));

    int global_step = 1;
    float best_val = 0.0f;
    int patience = 0;
    std::string best_ckpt;

    for (int epoch = 0; epoch < a.epochs; ++epoch) {
        std::vector<std::size_t> order(train.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        float epoch_loss = 0.0f;
        int   epoch_words = 0;

        for (int step = 0; step < steps_per_epoch; ++step) {
            zero_grads(params);

            // Tokenise all sentences in this micro-batch first.
            struct Item {
                std::vector<std::int32_t> ids;
                std::vector<std::int32_t> wsep;
                std::vector<std::uint8_t> tags;
            };
            std::vector<Item> items;
            items.reserve(a.batch);
            for (int b = 0; b < a.batch; ++b) {
                const auto& s = train[order[(step * a.batch + b) % train.size()]];
                std::string text = sentence_text(s);
                auto chunks = g::tokenise_sentence(text);
                std::size_t word_idx = 0;
                for (const auto& c : chunks) {
                    if (c.wsep_positions.empty()) { word_idx += c.word_spans.size(); continue; }
                    if (static_cast<int>(c.token_ids.size()) > g::kMaxSeqLen) {
                        fail("train", "sentence exceeds max_seq_len after tokenisation");
                    }
                    Item it;
                    it.ids  = c.token_ids;
                    apply_byte_noise(it.ids, rng, 0.02f);
                    it.wsep = c.wsep_positions;
                    for (std::size_t i = 0; i < c.word_spans.size(); ++i, ++word_idx) {
                        if (word_idx >= s.tag_id.size()) break;
                        it.tags.push_back(s.tag_id[word_idx]);
                    }
                    if (it.tags.size() != it.wsep.size()) {
                        // alignment drifted from chunking — skip
                        continue;
                    }
                    items.push_back(std::move(it));
                }
            }

            if (items.empty()) continue;

            // Forward all sentences, concat logits + targets.
            std::vector<ForwardCache> fcs(items.size());
            int total_words = 0;
            for (std::size_t i = 0; i < items.size(); ++i) {
                forward_sentence(w, items[i].ids, items[i].wsep, fcs[i]);
                total_words += static_cast<int>(items[i].wsep.size());
            }
            bt::Tensor logits_cat = bt::Tensor::mat(total_words, g::NUM_TAGS);
            std::vector<std::uint8_t> tags_cat;
            tags_cat.reserve(total_words);
            int off = 0;
            for (std::size_t i = 0; i < items.size(); ++i) {
                const int Wn = static_cast<int>(items[i].wsep.size());
                std::memcpy(logits_cat.host_f32_mut() + off * g::NUM_TAGS,
                            fcs[i].logits.host_f32(),
                            Wn * g::NUM_TAGS * sizeof(float));
                off += Wn;
                for (auto t : items[i].tags) tags_cat.push_back(t);
            }
            // Loss + dLogits.
            bt::Tensor dLogits;
            float loss_sum = xent_words(logits_cat, tags_cat, dLogits);
            // Mean-over-words scaling of gradients.
            const float scale = 1.0f / static_cast<float>(total_words);
            bt::scale_inplace(dLogits, scale);

            epoch_loss  += loss_sum;
            epoch_words += total_words;

            // Slice dLogits back per-sentence and backward.
            off = 0;
            for (std::size_t i = 0; i < items.size(); ++i) {
                const int Wn = static_cast<int>(items[i].wsep.size());
                bt::Tensor dlog_i = bt::Tensor::mat(Wn, g::NUM_TAGS);
                std::memcpy(dlog_i.host_f32_mut(),
                            dLogits.host_f32() + off * g::NUM_TAGS,
                            Wn * g::NUM_TAGS * sizeof(float));
                off += Wn;
                backward_sentence(w, fcs[i], dlog_i, grefs);
            }

            const float lr = lr_schedule(global_step, total_steps, a.warmup, a.lr);
            adamw_step(params, lr, 0.9f, 0.98f, 1e-9f, /*wd=*/0.01f,
                       global_step, /*grad_scale=*/1.0f);

            if (global_step % 20 == 0) {
                std::cout << "epoch " << epoch << " step " << global_step
                          << "/" << total_steps
                          << " loss " << std::fixed << std::setprecision(4)
                          << (loss_sum / total_words)
                          << " lr " << std::scientific << std::setprecision(2) << lr
                          << std::defaultfloat << std::endl;
            }
            ++global_step;
        }

        const float val_acc = evaluate(w, val);
        std::cout << "epoch " << epoch
                  << " mean_loss " << (epoch_loss / std::max(1, epoch_words))
                  << " val_acc " << val_acc << std::endl;

        // Checkpoint.
        char buf[64];
        std::snprintf(buf, sizeof(buf), "checkpoint_%02d.bin", epoch);
        std::string ckpt = (std::filesystem::path(a.out_dir) / buf).string();
        save_checkpoint(ckpt, w);

        if (val_acc > best_val + 1e-6f) {
            best_val = val_acc;
            best_ckpt = ckpt;
            patience = 0;
        } else if (++patience >= 3) {
            std::cout << "early stop at epoch " << epoch << "\n";
            break;
        }
    }

    // Copy best (or last) checkpoint to model.bin.
    std::string final_ckpt = best_ckpt.empty() ? "" : best_ckpt;
    if (final_ckpt.empty()) {
        // fall back to last
        std::string last = (std::filesystem::path(a.out_dir) / "model.bin").string();
        save_checkpoint(last, w);
        final_ckpt = last;
    } else {
        std::string dst = (std::filesystem::path(a.out_dir) / "model.bin").string();
        std::filesystem::copy_file(final_ckpt, dst,
            std::filesystem::copy_options::overwrite_existing);
        final_ckpt = dst;
    }

    // Round-trip: load via PosTagger::load and tag a sentence.
    std::cout << "round-trip: loading " << final_ckpt << "\n";
    try {
        auto tagger = g::PosTagger::load(final_ckpt);
        const std::string s = "The cat sat on the mat.";
        auto tags = tagger.tag(s);
        std::cout << "round-trip tag output: ";
        for (const auto& wt : tags) {
            std::cout << wt.word << "/"
                      << g::kPosTagNames[static_cast<int>(wt.tag)] << " ";
        }
        std::cout << "\n";
    } catch (const std::exception& e) {
        std::cerr << "round-trip FAILED: " << e.what() << "\n";
        return 3;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bt::init();
    Args a;
    try {
        parse_args(argc, argv, a);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        print_help();
        return 2;
    }
    if (a.help) { print_help(); return 0; }
    try {
        return run_training(a);
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
