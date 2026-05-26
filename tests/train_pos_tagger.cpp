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
#include <limits>
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
    std::vector<std::uint8_t>  tag_id;      // XPOS ids
    std::vector<std::uint8_t>  upos_id;     // UPOS ids
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
    const std::uint32_t magic = u32();
    if (magic != 0x504F5302u) fail("train_pos_tagger",
        "bad dataset magic in '" + path + "' (expected v2 0x504F5302; "
        "regenerate with tools/build_pos_dataset.py)");
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
        sent.upos_id.resize(nw);
        f.read(reinterpret_cast<char*>(sent.upos_id.data()), nw);
        sent.word_start.resize(nw);
        for (std::uint32_t i = 0; i < nw; ++i) sent.word_start[i] = u16();
        sent.word_len.resize(nw);
        for (std::uint32_t i = 0; i < nw; ++i) sent.word_len[i] = u16();
        for (std::uint32_t i = 0; i < nw; ++i) {
            if (static_cast<std::uint32_t>(sent.word_start[i]) + sent.word_len[i] > nb) {
                fail("train_pos_tagger", "word offset overflow in dataset");
            }
            if (sent.tag_id[i] >= g::NUM_TAGS) {
                fail("train_pos_tagger", "xpos id out of range in dataset");
            }
            if (sent.upos_id[i] >= g::NUM_UPOS_TAGS) {
                fail("train_pos_tagger", "upos id out of range in dataset");
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

// Device-aware init helpers. All allocate on the current default device.
void init_zero(bt::Tensor& t, int r, int c) {
    t = bt::Tensor::zeros(r, c);
}

void init_ones(bt::Tensor& t, int n) {
    t = bt::Tensor::zeros(n, 1);
    bt::add_scalar_inplace(t, 1.0f);
}

void init_normal(bt::Tensor& t, int r, int c, float stddev,
                 std::uint64_t& rng_state) {
    std::mt19937_64 mt(rng_state); rng_state = mt();
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> buf(static_cast<std::size_t>(r) * c);
    for (auto& v : buf) v = dist(mt);
    t = bt::Tensor::from_host_on(bt::default_device(), buf.data(), r, c);
}

void init_xavier(bt::Tensor& W, int out_dim, int in_dim,
                 std::uint64_t& rng_state) {
    // xavier_init is documented CPU-only — build on host, migrate to device.
    bt::Tensor cpu_W = bt::Tensor::mat(out_dim, in_dim);
    bt::xavier_init(cpu_W, rng_state);
    W = cpu_W.to(bt::default_device());
}

// Training-only auxiliary UPOS head — never serialised into model.bin.
struct AuxHead {
    int               in_features  = 0;
    int               out_features = 0;
    bt::Tensor        W;   // (NUM_UPOS_TAGS, d_model)
    bt::Tensor        b;   // (NUM_UPOS_TAGS, 1)
};

void init_aux_head(AuxHead& h, int d_model, std::uint64_t& rng) {
    h.in_features  = d_model;
    h.out_features = static_cast<int>(g::NUM_UPOS_TAGS);
    init_xavier(h.W, h.out_features, d_model, rng);
    init_zero(h.b, h.out_features, 1);
}

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

    init_normal(w.token_emb, g::kVocab,     g::kDModel, 0.02f, rng);
    init_normal(w.pos_emb,   g::kMaxSeqLen, g::kDModel, 0.02f, rng);

    for (int i = 0; i < g::kNumLayers; ++i) {
        auto& L = w.layers[i];
        L.ln1.features = g::kDModel; L.ln1.eps = 1e-5f;
        init_ones(L.ln1.gamma, g::kDModel);
        init_zero(L.ln1.beta,  g::kDModel, 1);

        L.mha.num_heads = g::kNumHeads; L.mha.embed_dim = g::kDModel;
        init_xavier(L.mha.Wq, g::kDModel, g::kDModel, rng);
        init_xavier(L.mha.Wk, g::kDModel, g::kDModel, rng);
        init_xavier(L.mha.Wv, g::kDModel, g::kDModel, rng);
        init_xavier(L.mha.Wo, g::kDModel, g::kDModel, rng);
        init_zero(L.mha.bq, g::kDModel, 1);
        init_zero(L.mha.bk, g::kDModel, 1);
        init_zero(L.mha.bv, g::kDModel, 1);
        init_zero(L.mha.bo, g::kDModel, 1);

        L.ln2.features = g::kDModel; L.ln2.eps = 1e-5f;
        init_ones(L.ln2.gamma, g::kDModel);
        init_zero(L.ln2.beta,  g::kDModel, 1);

        L.ffn1.in_features = g::kDModel; L.ffn1.out_features = g::kFFN;
        init_xavier(L.ffn1.W, g::kFFN, g::kDModel, rng);
        init_zero(L.ffn1.b, g::kFFN, 1);

        L.ffn2.in_features = g::kFFN; L.ffn2.out_features = g::kDModel;
        init_xavier(L.ffn2.W, g::kDModel, g::kFFN, rng);
        init_zero(L.ffn2.b, g::kDModel, 1);
    }

    w.final_ln.features = g::kDModel; w.final_ln.eps = 1e-5f;
    init_ones(w.final_ln.gamma, g::kDModel);
    init_zero(w.final_ln.beta,  g::kDModel, 1);

    w.head.in_features = g::kDModel; w.head.out_features = g::NUM_TAGS;
    init_xavier(w.head.W, g::NUM_TAGS, g::kDModel, rng);
    init_zero(w.head.b, g::NUM_TAGS, 1);
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
        p.grad  = bt::Tensor::zeros(t->rows, t->cols);
        p.m     = bt::Tensor::zeros(t->rows, t->cols);
        p.v     = bt::Tensor::zeros(t->rows, t->cols);
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

void append_aux_params(std::vector<Param>& ps, AuxHead& h) {
    auto add = [&](bt::Tensor* t, bool decay, const std::string& nm) {
        Param p;
        p.data  = t;
        p.grad  = bt::Tensor::zeros(t->rows, t->cols);
        p.m     = bt::Tensor::zeros(t->rows, t->cols);
        p.v     = bt::Tensor::zeros(t->rows, t->cols);
        p.decay = decay;
        p.name  = nm;
        ps.push_back(std::move(p));
    };
    add(&h.W, true,  "upos_head.W");
    add(&h.b, false, "upos_head.b");
}

void zero_grads(std::vector<Param>& ps) {
    for (auto& p : ps) p.grad.zero();
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
            // decoupled weight decay: p -= lr * wd * p  ==  p *= (1 - lr*wd)
            bt::scale_inplace(*p.data, 1.0f - lr * wd);
        }
    }
}

// ─── Per-sentence forward caches ──────────────────────────────────────────

struct LayerCache {
    bt::Tensor h_in;     // (L, D) input to ln1 (== residual stream pre-attn)
    bt::Tensor ln1_y;    // (L, D)
    bt::Tensor ln1_xhat; // (L, D)
    bt::Tensor ln1_rstd; // (L, 1) FP32
    bt::Tensor Qh, Kh, Vh, Attnh, Yconcat;  // mha caches
    bt::Tensor attn_out; // (L, D)
    bt::Tensor h_mid;    // (L, D) after attn residual
    bt::Tensor ln2_y;
    bt::Tensor ln2_xhat;
    bt::Tensor ln2_rstd;
    bt::Tensor ffn1_pre;  // (L, ffn) pre-gelu
    bt::Tensor ffn1_act;  // (L, ffn) post-gelu
    bt::Tensor ffn2_out;  // (L, D)
};

struct ForwardCache {
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> wsep_positions;
    bt::Tensor h0;        // (L, D) embedding sum
    std::vector<LayerCache> layers;
    std::vector<std::uint8_t> layer_active;  // 1 = ran block, 0 = skipped
    float      layer_eval_scale = 1.0f;      // eval scaling applied to outputs
    bt::Tensor h_pre_final_ln;   // (L, D)
    bt::Tensor h_final;          // (L, D) post final_ln
    bt::Tensor final_xhat;       // (L, D)
    bt::Tensor final_rstd;       // (L, 1)
    bt::Tensor pooled;           // (W, D)
    bt::Tensor logits;           // (W, NUM_TAGS) — XPOS
    bt::Tensor upos_logits;      // (W, NUM_UPOS_TAGS) — aux
};

// Upload an int32 host vector to an INT32 tensor on the current default
// device, return the device-resident int32 pointer. The Tensor is held in
// `out_buf` by the caller to keep storage alive. On CPU default this is just
// a host INT32 tensor (no copy).
const std::int32_t* upload_idx(const std::int32_t* host, int n,
                               bt::Tensor& out_buf) {
    bt::Tensor cpu = bt::Tensor::zeros_on(bt::Device::CPU, n, 1,
                                          bt::Dtype::INT32);
    auto* p = static_cast<std::int32_t*>(cpu.host_raw_mut());
    for (int i = 0; i < n; ++i) p[i] = host[i];
    out_buf = cpu.to(bt::default_device());
    return static_cast<const std::int32_t*>(out_buf.data);
}

// Build the embedding sum (token_emb[ids] + pos_emb[0..L-1]).
void embed_forward(const g::PosWeights& w, const std::vector<std::int32_t>& ids,
                   bt::Tensor& out) {
    const int L = static_cast<int>(ids.size());
    bt::Tensor tok_idx_buf;
    const std::int32_t* d_tok = upload_idx(ids.data(), L, tok_idx_buf);
    bt::embedding_lookup_forward(w.token_emb, d_tok, L, out);
    std::vector<std::int32_t> pidx(L);
    for (int i = 0; i < L; ++i) pidx[i] = i;
    bt::Tensor pos_idx_buf;
    const std::int32_t* d_pos = upload_idx(pidx.data(), L, pos_idx_buf);
    bt::Tensor pe;
    bt::embedding_lookup_forward(w.pos_emb, d_pos, L, pe);
    bt::add_inplace_batched(out, pe);
}

// Forward one sentence; populates fc.logits and (when aux != nullptr)
// fc.upos_logits. `layer_active` is a per-layer 0/1 mask: 1 = run the block,
// 0 = skip (residual = identity). When non-empty its size must equal num
// layers. `layer_eval_scale` is multiplied into each non-skipped block's
// output (use 1.0 in training; (1 - p_drop_layer) at eval).
void forward_sentence(const g::PosWeights& w,
                      const std::vector<std::int32_t>& token_ids,
                      const std::vector<std::int32_t>& wsep_positions,
                      ForwardCache& fc,
                      const AuxHead* aux = nullptr,
                      const std::vector<std::uint8_t>* layer_active = nullptr,
                      float layer_eval_scale = 1.0f) {
    const int L = static_cast<int>(token_ids.size());
    const int D = w.d_model;
    fc.token_ids      = token_ids;
    fc.wsep_positions = wsep_positions;

    embed_forward(w, token_ids, fc.h0);

    fc.layers.assign(w.layers.size(), LayerCache{});
    fc.layer_active.assign(w.layers.size(), 1u);
    if (layer_active && layer_active->size() == w.layers.size()) {
        fc.layer_active = *layer_active;
    }
    fc.layer_eval_scale = layer_eval_scale;
    // residual stream — start as a deep copy of h0 (device-aware via clone()).
    bt::Tensor h_resid = fc.h0.clone();

    for (std::size_t li = 0; li < w.layers.size(); ++li) {
        const auto& Lw = w.layers[li];
        auto& C = fc.layers[li];

        // Stochastic depth: skip the entire block (residual = identity).
        if (!fc.layer_active[li]) {
            continue;
        }

        // Save input to ln1 (== current residual)
        C.h_in = h_resid.clone();

        // pre-norm ln1 (batched, with caches)
        C.ln1_y    = bt::Tensor::zeros(L, D);
        C.ln1_xhat = bt::Tensor::zeros(L, D);
        bt::Tensor ln1_mean;
        bt::layernorm_forward_batched_with_caches(
            h_resid, Lw.ln1.gamma, Lw.ln1.beta,
            C.ln1_y, C.ln1_xhat, ln1_mean, C.ln1_rstd, Lw.ln1.eps);

        // MHA over ln1_y
        bt::Tensor& X = C.ln1_y;
        C.Qh = bt::Tensor{}; C.Kh = bt::Tensor{}; C.Vh = bt::Tensor{};
        C.Attnh = bt::Tensor{}; C.Yconcat = bt::Tensor{};
        C.attn_out = bt::Tensor::zeros(L, D);
        bt::mha_forward(X, Lw.mha.Wq, Lw.mha.Wk, Lw.mha.Wv, Lw.mha.Wo,
                        &Lw.mha.bq, &Lw.mha.bk, &Lw.mha.bv, &Lw.mha.bo,
                        /*d_mask=*/nullptr, w.num_heads,
                        C.Qh, C.Kh, C.Vh, C.Attnh, C.Yconcat,
                        C.attn_out);

        // residual: h_resid += scale * attn_out  (scale = 1 in train)
        if (fc.layer_eval_scale != 1.0f) {
            bt::scale_inplace(C.attn_out, fc.layer_eval_scale);
        }
        bt::add_inplace_batched(h_resid, C.attn_out);
        // snapshot pre-ln2 state
        C.h_mid = h_resid.clone();

        // pre-norm ln2 (batched)
        C.ln2_y    = bt::Tensor::zeros(L, D);
        C.ln2_xhat = bt::Tensor::zeros(L, D);
        bt::Tensor ln2_mean;
        bt::layernorm_forward_batched_with_caches(
            h_resid, Lw.ln2.gamma, Lw.ln2.beta,
            C.ln2_y, C.ln2_xhat, ln2_mean, C.ln2_rstd, Lw.ln2.eps);

        // ffn1: linear -> gelu
        C.ffn1_pre = bt::Tensor::zeros(L, w.ffn_hidden);
        Lw.ffn1.forward_batched(C.ln2_y, C.ffn1_pre);
        C.ffn1_act = bt::Tensor::zeros(L, w.ffn_hidden);
        bt::gelu_forward(C.ffn1_pre, C.ffn1_act);
        // ffn2
        C.ffn2_out = bt::Tensor::zeros(L, D);
        Lw.ffn2.forward_batched(C.ffn1_act, C.ffn2_out);
        // residual (scale eval-side)
        if (fc.layer_eval_scale != 1.0f) {
            bt::scale_inplace(C.ffn2_out, fc.layer_eval_scale);
        }
        bt::add_inplace_batched(h_resid, C.ffn2_out);
    }

    // final_ln (pre-residual snapshot for backward)
    fc.h_pre_final_ln = h_resid.clone();
    fc.h_final    = bt::Tensor::zeros(L, D);
    fc.final_xhat = bt::Tensor::zeros(L, D);
    bt::Tensor final_mean;
    bt::layernorm_forward_batched_with_caches(
        h_resid, w.final_ln.gamma, w.final_ln.beta,
        fc.h_final, fc.final_xhat, final_mean, fc.final_rstd, w.final_ln.eps);

    // Word pool + head (device-aware row gather via copy_d2d).
    const int W = static_cast<int>(wsep_positions.size());
    fc.pooled = bt::Tensor::zeros(W, D);
    for (int i = 0; i < W; ++i) {
        const int t = wsep_positions[i];
        bt::copy_d2d(fc.h_final, t * D, fc.pooled, i * D, D);
    }
    fc.logits = bt::Tensor::zeros(W, w.num_tags);
    w.head.forward_batched(fc.pooled, fc.logits);

    if (aux) {
        fc.upos_logits = bt::Tensor::zeros(W, aux->out_features);
        bt::linear_forward_batched(aux->W, aux->b, fc.pooled, fc.upos_logits);
    }
}

// Backward one sentence given dLogits — accumulates grads into Param entries.
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
        bt::Tensor* bq; bt::Tensor* bk; bt::Tensor* bv; bt::Tensor* bo;
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
        L.bq     = find(p + "attn.bq"); L.bk = find(p + "attn.bk");
        L.bv     = find(p + "attn.bv"); L.bo = find(p + "attn.bo");
        L.ln2_g  = find(p + "ln2.gamma"); L.ln2_b = find(p + "ln2.beta");
        L.ffn1_W = find(p + "ffn1.W"); L.ffn1_b = find(p + "ffn1.b");
        L.ffn2_W = find(p + "ffn2.W"); L.ffn2_b = find(p + "ffn2.b");
        r.layers.push_back(L);
    }
    return r;
}

void backward_sentence(const g::PosWeights& w,
                       const ForwardCache& fc,
                       const bt::Tensor& dLogits,        // (W, NUM_TAGS)
                       ParamRefs& gr,
                       const AuxHead* aux = nullptr,
                       const bt::Tensor* dUposLogits = nullptr,
                       bt::Tensor* gAuxW = nullptr,
                       bt::Tensor* gAuxB = nullptr) {
    const int L = static_cast<int>(fc.token_ids.size());
    const int Wn = static_cast<int>(fc.wsep_positions.size());
    const int D = w.d_model;

    // head backward (XPOS)
    bt::Tensor dPooled = bt::Tensor::zeros(Wn, D);
    bt::linear_backward_batched(w.head.W, fc.pooled, dLogits,
                                dPooled, *gr.head_W, *gr.head_b);

    // aux head backward (UPOS) — gradient to dPooled is summed with XPOS branch.
    if (aux && dUposLogits && gAuxW && gAuxB) {
        bt::Tensor dPooled_upos = bt::Tensor::zeros(Wn, D);
        bt::linear_backward_batched(aux->W, fc.pooled, *dUposLogits,
                                    dPooled_upos, *gAuxW, *gAuxB);
        bt::add_inplace_batched(dPooled, dPooled_upos);
    }

    // scatter dPooled into dH_final at wsep positions (device-aware).
    bt::Tensor dH_final = bt::Tensor::zeros(L, D);
    for (int i = 0; i < Wn; ++i) {
        const int t = fc.wsep_positions[i];
        bt::copy_d2d(dPooled, i * D, dH_final, t * D, D);
    }

    // final_ln backward (batched)
    bt::Tensor dH_resid = bt::Tensor::zeros(L, D);
    bt::layernorm_backward_batched_with_caches(
        dH_final, fc.final_xhat, w.final_ln.gamma, fc.final_rstd,
        dH_resid, *gr.final_ln_gamma, *gr.final_ln_beta);

    // Walk layers in reverse.
    for (int li = static_cast<int>(w.layers.size()) - 1; li >= 0; --li) {
        const auto& Lw = w.layers[li];
        const auto& C  = fc.layers[li];
        auto& gL = gr.layers[li];

        // Stochastic depth: layer was skipped — residual is identity, gradient
        // passes through unchanged. No weight grads for this layer.
        if (!fc.layer_active[static_cast<std::size_t>(li)]) {
            continue;
        }

        // dH_resid is grad w.r.t. h_resid AFTER this layer (= post ffn residual).
        // The residual edge propagates dH_resid through to h_mid as-is.
        bt::Tensor dh_mid_resid = dH_resid.clone();

        // ffn2 backward
        bt::Tensor dffn1_act = bt::Tensor::zeros(L, w.ffn_hidden);
        bt::linear_backward_batched(Lw.ffn2.W, C.ffn1_act, dH_resid,
                                    dffn1_act, *gL.ffn2_W, *gL.ffn2_b);
        // gelu backward (tanh approx)
        bt::Tensor dffn1_pre = bt::Tensor::zeros(L, w.ffn_hidden);
        bt::gelu_backward(C.ffn1_pre, dffn1_act, dffn1_pre);
        // ffn1 backward
        bt::Tensor dln2_y = bt::Tensor::zeros(L, D);
        bt::linear_backward_batched(Lw.ffn1.W, C.ln2_y, dffn1_pre,
                                    dln2_y, *gL.ffn1_W, *gL.ffn1_b);
        // ln2 backward
        bt::Tensor dh_mid_ffn = bt::Tensor::zeros(L, D);
        bt::layernorm_backward_batched_with_caches(
            dln2_y, C.ln2_xhat, Lw.ln2.gamma, C.ln2_rstd,
            dh_mid_ffn, *gL.ln2_g, *gL.ln2_b);
        // sum branches
        bt::add_inplace_batched(dh_mid_resid, dh_mid_ffn);
        // Now dh_mid_resid is the grad w.r.t. h_mid.

        // Residual on h_mid: h_mid = h_in + attn_out
        bt::Tensor dh_in_resid = dh_mid_resid.clone();

        // attn backward: dh_mid -> dattn_out -> dln1_y -> dh_in
        bt::Tensor dln1_y = bt::Tensor::zeros(L, D);
        bt::mha_backward(dh_mid_resid, C.ln1_y,
                         C.Qh, C.Kh, C.Vh, C.Attnh, C.Yconcat,
                         Lw.mha.Wq, Lw.mha.Wk, Lw.mha.Wv, Lw.mha.Wo,
                         /*d_mask=*/nullptr, w.num_heads,
                         dln1_y, *gL.Wq, *gL.Wk, *gL.Wv, *gL.Wo,
                         gL.bq, gL.bk, gL.bv, gL.bo);
        // ln1 backward
        bt::Tensor dh_in_ln = bt::Tensor::zeros(L, D);
        bt::layernorm_backward_batched_with_caches(
            dln1_y, C.ln1_xhat, Lw.ln1.gamma, C.ln1_rstd,
            dh_in_ln, *gL.ln1_g, *gL.ln1_b);
        // combine
        bt::add_inplace_batched(dh_in_resid, dh_in_ln);
        dH_resid = dh_in_resid;
    }

    // dH_resid is now the grad w.r.t. h0 (token_emb + pos_emb).
    // Embedding backward: scatter-add into token_emb and pos_emb grads.
    bt::Tensor tok_idx_buf;
    const std::int32_t* d_tok = upload_idx(fc.token_ids.data(), L, tok_idx_buf);
    bt::embedding_lookup_backward(dH_resid, d_tok, L, *gr.token_emb);
    std::vector<std::int32_t> pidx(L);
    for (int i = 0; i < L; ++i) pidx[i] = i;
    bt::Tensor pos_idx_buf;
    const std::int32_t* d_pos = upload_idx(pidx.data(), L, pos_idx_buf);
    bt::embedding_lookup_backward(dH_resid, d_pos, L, *gr.pos_emb);
}

// ─── Loss ─────────────────────────────────────────────────────────────────

// Compute softmax cross-entropy over the W rows of `logits` and return loss
// sum plus an on-device dLogits tensor. softmax_xent_segment is a host
// scalar op, so we transfer logits to host once, compute on host, then upload
// dLogits back to the default device.
float xent_words(const bt::Tensor& logits, const std::vector<std::uint8_t>& tags,
                 bt::Tensor& dLogits_dev) {
    const int Wn = logits.rows;
    const int C = logits.cols;
    std::vector<float> logits_host = logits.to_host_vector();
    std::vector<float> dlog_host(static_cast<std::size_t>(Wn) * C);
    std::vector<float> probs_row(C);
    std::vector<float> target_row(C);
    float total = 0.0f;
    for (int i = 0; i < Wn; ++i) {
        std::fill(target_row.begin(), target_row.end(), 0.0f);
        target_row[tags[i]] = 1.0f;
        const float* lg = logits_host.data() + static_cast<std::size_t>(i) * C;
        float* dl = dlog_host.data() + static_cast<std::size_t>(i) * C;
        float loss = bt::softmax_xent_segment(lg, target_row.data(),
                                              probs_row.data(),
                                              dl, C, nullptr);
        total += loss;
    }
    dLogits_dev = bt::Tensor::from_host_on(bt::default_device(),
                                           dlog_host.data(), Wn, C);
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
    // to_host_vector handles d2h for non-CPU tensors.
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

// Aux UPOS head — never goes into model.bin. Magic 0x504F5303 ("POS\x03").
void save_aux_head(const std::string& path, const AuxHead& h) {
    std::ofstream f(path, std::ios::binary);
    if (!f) fail("train_pos_tagger", "could not write aux head '" + path + "'");
    w_u32(f, 0x504F5303u);
    w_u32(f, 1u);
    w_u32(f, static_cast<std::uint32_t>(h.out_features));
    w_u32(f, static_cast<std::uint32_t>(h.in_features));
    w_u32(f, 2u);  // num tensors
    w_tensor(f, "upos_head.W", h.W);
    w_tensor(f, "upos_head.b", h.b);
}

// ─── Byte-noise + tokeniser dispatch ─────────────────────────────────────

void apply_byte_noise(std::vector<std::int32_t>& ids, std::mt19937& rng,
                      float p_noise) {
    if (p_noise <= 0.0f) return;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uniform_int_distribution<int> b(4, 259);
    for (auto& t : ids) {
        if (t >= 4 && u(rng) < p_noise) t = b(rng);
    }
}

// Drop entire word byte ranges with probability p_word, keeping each <wsep>
// anchor in place. Renumbers `wsep_positions` to match the trimmed sequence.
// `ids` and `wsep_positions` are rewritten in-place. The tag vector is left
// alone (one tag per surviving wsep, same order).
void apply_word_dropout(std::vector<std::int32_t>& ids,
                        std::vector<std::int32_t>& wsep_positions,
                        std::mt19937& rng, float p_word) {
    if (p_word <= 0.0f || wsep_positions.empty()) return;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    const int W = static_cast<int>(wsep_positions.size());
    const int L = static_cast<int>(ids.size());
    std::vector<std::int32_t> new_ids;
    std::vector<std::int32_t> new_wsep;
    new_ids.reserve(ids.size());
    new_wsep.reserve(W);

    // bytes before the first wsep (the <bos>): copy verbatim.
    for (int i = 0; i < wsep_positions[0]; ++i) new_ids.push_back(ids[i]);

    for (int wi = 0; wi < W; ++wi) {
        const int wsep_pos = wsep_positions[wi];
        const int next_pos = (wi + 1 < W) ? wsep_positions[wi + 1] : L;
        new_wsep.push_back(static_cast<std::int32_t>(new_ids.size()));
        new_ids.push_back(ids[wsep_pos]);                      // <wsep>
        const bool drop = (u(rng) < p_word);
        if (!drop) {
            for (int j = wsep_pos + 1; j < next_pos; ++j) {
                new_ids.push_back(ids[j]);
            }
        }
    }
    // Trailing <eos> (and anything past the last word) — copy from after last
    // word's byte range; but we already consumed everything up to `L`. The
    // tail after the final word is L..L → empty. The <eos> sat at L-1
    // immediately after the final word's bytes, but it's included as part of
    // the final word's [wsep_pos+1, next_pos=L) range above. Wait — the
    // tokeniser places <eos> AFTER the final word's bytes; with wsep_positions
    // listing only wseps, next_pos for the last word == L includes the eos.
    // If the last word was dropped, we dropped the eos too. Append it back.
    if (!ids.empty() && ids.back() == 2 /*kEos*/) {
        if (new_ids.empty() || new_ids.back() != 2) {
            new_ids.push_back(2);
        }
    }
    ids = std::move(new_ids);
    wsep_positions = std::move(new_wsep);
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
            // Deterministic UPOS by a separate hash variant.
            std::uint32_t hu = h ^ 0x9E3779B9u;
            sent.upos_id.push_back(
                static_cast<std::uint8_t>(hu % g::NUM_UPOS_TAGS));
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
    bool  synthetic    = false;
    bool  help         = false;
    float byte_noise   = 0.10f;
    float word_dropout = 0.10f;
    float layer_dropout = 0.10f;
    float upos_weight  = 1.0f;
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
        else if (k == "--byte-noise")    a.byte_noise    = std::stof(next());
        else if (k == "--word-dropout")  a.word_dropout  = std::stof(next());
        else if (k == "--layer-dropout") a.layer_dropout = std::stof(next());
        else if (k == "--upos-weight")   a.upos_weight   = std::stof(next());
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
        "                     and run 2 epochs end-to-end (smoke test).\n"
        "  --byte-noise F     byte-noise probability (default 0.10)\n"
        "  --word-dropout F   whole-word dropout probability (default 0.10)\n"
        "  --layer-dropout F  per-layer stochastic depth probability (default 0.10)\n"
        "  --upos-weight F    aux UPOS loss weight (default 1.0)\n";
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

struct EvalStats {
    float xpos_acc;
    float upos_acc;
    float val_loss;   // mean (xpos_xent + upos_weight * upos_xent) per word
};

// Compute mean cross-entropy on (logits, target_ids), no dropout, no noise.
// Returns sum of per-word losses (caller divides by total).
float xent_only(const bt::Tensor& logits,
                const std::vector<std::uint8_t>& targets) {
    const int Wn = logits.rows;
    const int C  = logits.cols;
    std::vector<float> lh = logits.to_host_vector();
    std::vector<float> probs(C), tgt(C), dlog(C);
    float total = 0.0f;
    for (int i = 0; i < Wn; ++i) {
        std::fill(tgt.begin(), tgt.end(), 0.0f);
        tgt[targets[i]] = 1.0f;
        const float* lg = lh.data() + static_cast<std::size_t>(i) * C;
        total += bt::softmax_xent_segment(lg, tgt.data(), probs.data(),
                                          dlog.data(), C, nullptr);
    }
    return total;
}

EvalStats evaluate(const g::PosWeights& w, const AuxHead& aux,
                   const std::vector<Sentence>& data,
                   float layer_dropout, float upos_weight) {
    const float eval_scale = 1.0f - layer_dropout;
    int total = 0, x_correct = 0, u_correct = 0;
    float loss_sum = 0.0f;
    for (const auto& s : data) {
        std::string text = sentence_text(s);
        auto chunks = g::tokenise_sentence(text);
        std::size_t word_idx = 0;
        for (const auto& c : chunks) {
            if (c.wsep_positions.empty()) continue;
            ForwardCache fc;
            forward_sentence(w, c.token_ids, c.wsep_positions, fc,
                             &aux, /*layer_active=*/nullptr, eval_scale);
            bt::Tensor xi_dev, ui_dev;
            bt::argmax_rows(fc.logits, xi_dev);
            bt::argmax_rows(fc.upos_logits, ui_dev);
            auto xi = xi_dev.to_host_vector();
            auto ui = ui_dev.to_host_vector();

            // Gather targets in chunk order.
            std::vector<std::uint8_t> xtgt, utgt;
            std::size_t saved_word_idx = word_idx;
            for (std::size_t i = 0; i < c.wsep_positions.size(); ++i, ++word_idx) {
                if (word_idx >= s.tag_id.size()) break;
                xtgt.push_back(s.tag_id[word_idx]);
                utgt.push_back(s.upos_id[word_idx]);
            }
            loss_sum += xent_only(fc.logits, xtgt);
            loss_sum += upos_weight * xent_only(fc.upos_logits, utgt);

            word_idx = saved_word_idx;
            for (std::size_t i = 0; i < c.wsep_positions.size(); ++i, ++word_idx) {
                if (word_idx >= s.tag_id.size()) break;
                if (static_cast<int>(xi[i]) == s.tag_id[word_idx])  ++x_correct;
                if (static_cast<int>(ui[i]) == s.upos_id[word_idx]) ++u_correct;
                ++total;
            }
        }
    }
    EvalStats r;
    r.xpos_acc = (total == 0) ? 0.0f : static_cast<float>(x_correct) / total;
    r.upos_acc = (total == 0) ? 0.0f : static_cast<float>(u_correct) / total;
    r.val_loss = (total == 0) ? 0.0f : loss_sum / total;
    return r;
}

int run_training(Args& a) {
    const bool cuda_avail = bt::is_available(bt::Device::CUDA);
    std::cout << "cuda available: " << (cuda_avail ? "yes" : "no") << "\n";
    if (a.device == "cuda") {
        if (cuda_avail) bt::set_default_device(bt::Device::CUDA);
        else std::cerr << "warn: cuda requested but unavailable, falling back to cpu\n";
    } else {
        bt::set_default_device(bt::Device::CPU);
    }
    std::cout << "train_pos_tagger: device="
              << bt::device_name(bt::default_device()) << "\n";

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
    AuxHead aux;
    {
        std::uint64_t aux_rng = static_cast<std::uint64_t>(a.seed) ^ 0xA0FACEu;
        init_aux_head(aux, w.d_model, aux_rng);
    }
    auto params = make_param_list(w);
    append_aux_params(params, aux);
    auto grefs  = param_refs(params);

    // Locate aux head grads so backward can find them by name.
    bt::Tensor* gAuxW = nullptr;
    bt::Tensor* gAuxB = nullptr;
    for (auto& p : params) {
        if (p.name == "upos_head.W") gAuxW = &p.grad;
        if (p.name == "upos_head.b") gAuxB = &p.grad;
    }

    std::cout << "regularisation: byte_noise=" << a.byte_noise
              << " word_dropout=" << a.word_dropout
              << " layer_dropout=" << a.layer_dropout << "\n";
    std::cout << "aux: upos_weight=" << a.upos_weight << "\n";

    const int steps_per_epoch = std::max(1, static_cast<int>(train.size()) / a.batch);
    const int total_steps = a.epochs * steps_per_epoch;
    std::mt19937 rng(static_cast<std::uint32_t>(a.seed));

    int global_step = 1;
    float best_val_loss = std::numeric_limits<float>::infinity();
    int patience = 0;
    std::string best_ckpt;
    std::string best_aux_ckpt;

    for (int epoch = 0; epoch < a.epochs; ++epoch) {
        std::vector<std::size_t> order(train.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        float epoch_loss = 0.0f;
        int   epoch_words = 0;

        for (int step = 0; step < steps_per_epoch; ++step) {
            zero_grads(params);

            struct Item {
                std::vector<std::int32_t> ids;
                std::vector<std::int32_t> wsep;
                std::vector<std::uint8_t> xtags;
                std::vector<std::uint8_t> utags;
                std::vector<std::uint8_t> layer_active;  // per-layer drop mask
            };
            std::vector<Item> items;
            items.reserve(a.batch);
            std::uniform_real_distribution<float> u01(0.0f, 1.0f);
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
                    it.wsep = c.wsep_positions;
                    for (std::size_t i = 0; i < c.word_spans.size(); ++i, ++word_idx) {
                        if (word_idx >= s.tag_id.size()) break;
                        it.xtags.push_back(s.tag_id[word_idx]);
                        it.utags.push_back(s.upos_id[word_idx]);
                    }
                    if (it.xtags.size() != it.wsep.size()) continue;

                    // (b) whole-word dropout BEFORE byte-noise.
                    apply_word_dropout(it.ids, it.wsep, rng, a.word_dropout);
                    // (a) byte-noise.
                    apply_byte_noise(it.ids, rng, a.byte_noise);

                    // (c) per-layer stochastic-depth mask, sampled per-sentence.
                    it.layer_active.assign(g::kNumLayers, 1u);
                    for (int li = 0; li < g::kNumLayers; ++li) {
                        if (u01(rng) < a.layer_dropout) it.layer_active[li] = 0u;
                    }
                    items.push_back(std::move(it));
                }
            }

            if (items.empty()) continue;

            int total_words = 0;
            for (const auto& it : items) total_words += static_cast<int>(it.wsep.size());
            const float scale = 1.0f / static_cast<float>(total_words);

            float xpos_loss_sum = 0.0f;
            float upos_loss_sum = 0.0f;
            for (std::size_t i = 0; i < items.size(); ++i) {
                ForwardCache fc;
                forward_sentence(w, items[i].ids, items[i].wsep, fc,
                                 &aux, &items[i].layer_active, /*eval_scale=*/1.0f);

                bt::Tensor dLogits;
                xpos_loss_sum += xent_words(fc.logits, items[i].xtags, dLogits);
                bt::scale_inplace(dLogits, scale);

                bt::Tensor dUposLogits;
                upos_loss_sum += xent_words(fc.upos_logits, items[i].utags, dUposLogits);
                // Apply 1/total_words scaling and upos loss weight.
                bt::scale_inplace(dUposLogits, scale * a.upos_weight);

                backward_sentence(w, fc, dLogits, grefs,
                                  &aux, &dUposLogits, gAuxW, gAuxB);
            }

            const float loss_sum = xpos_loss_sum + a.upos_weight * upos_loss_sum;
            epoch_loss  += loss_sum;
            epoch_words += total_words;

            const float lr = lr_schedule(global_step, total_steps, a.warmup, a.lr);
            adamw_step(params, lr, 0.9f, 0.98f, 1e-9f, /*wd=*/0.01f,
                       global_step, /*grad_scale=*/1.0f);

            if (global_step % 20 == 0) {
                std::cout << "epoch " << epoch << " step " << global_step
                          << "/" << total_steps
                          << " xpos_loss " << std::fixed << std::setprecision(4)
                          << (xpos_loss_sum / total_words)
                          << " upos_loss " << (upos_loss_sum / total_words)
                          << " total "     << (loss_sum / total_words)
                          << " lr " << std::scientific << std::setprecision(2) << lr
                          << std::defaultfloat << std::endl;
            }
            ++global_step;
        }

        const EvalStats es = evaluate(w, aux, val, a.layer_dropout, a.upos_weight);
        std::cout << "epoch " << epoch
                  << " mean_loss " << std::fixed << std::setprecision(4)
                  << (epoch_loss / std::max(1, epoch_words))
                  << " val_loss " << es.val_loss
                  << " xpos_val_acc " << es.xpos_acc
                  << " upos_val_acc " << es.upos_acc
                  << std::defaultfloat << std::endl;

        // Checkpoint XPOS-only weights for the inference path. Aux head saved
        // to a sibling file (different magic, training-only).
        char buf[64];
        std::snprintf(buf, sizeof(buf), "checkpoint_%02d.bin", epoch);
        std::string ckpt = (std::filesystem::path(a.out_dir) / buf).string();
        save_checkpoint(ckpt, w);

        std::snprintf(buf, sizeof(buf), "aux_head_%02d.bin", epoch);
        std::string aux_ckpt = (std::filesystem::path(a.out_dir) / buf).string();
        save_aux_head(aux_ckpt, aux);

        if (es.val_loss + 1e-4f < best_val_loss) {
            best_val_loss = es.val_loss;
            best_ckpt = ckpt;
            best_aux_ckpt = aux_ckpt;
            patience = 0;
        } else if (++patience >= 3) {
            std::cout << "early stop at epoch " << epoch
                      << " (val_loss plateau)\n";
            break;
        }
    }

    // Copy best (or last) checkpoint to model.bin and aux_head.bin.
    std::string final_ckpt = best_ckpt.empty() ? "" : best_ckpt;
    if (final_ckpt.empty()) {
        // fall back to last
        std::string last = (std::filesystem::path(a.out_dir) / "model.bin").string();
        save_checkpoint(last, w);
        final_ckpt = last;
        std::string aux_last = (std::filesystem::path(a.out_dir) / "aux_head.bin").string();
        save_aux_head(aux_last, aux);
    } else {
        std::string dst = (std::filesystem::path(a.out_dir) / "model.bin").string();
        std::filesystem::copy_file(final_ckpt, dst,
            std::filesystem::copy_options::overwrite_existing);
        final_ckpt = dst;
        if (!best_aux_ckpt.empty()) {
            std::string adst = (std::filesystem::path(a.out_dir) / "aux_head.bin").string();
            std::filesystem::copy_file(best_aux_ckpt, adst,
                std::filesystem::copy_options::overwrite_existing);
        }
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
