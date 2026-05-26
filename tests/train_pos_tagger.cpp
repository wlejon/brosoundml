// POS tagger training driver. Standalone executable (not a ctest).
//
// CLI: train_pos_tagger --train <pos_train.bin> --val <pos_val.bin>
//                      --out <dir> [--epochs N] [--batch N] [--lr F]
//                      [--warmup N] [--seed N] [--device cpu|cuda]
//                      [--synthetic]
//
// Packed-varlen forward+backward: each minibatch's sentences are concatenated
// into a single (total_tokens, D) residual stream; attention runs through
// brotensor::flash_attention_varlen_{forward,backward} with cu_seqlens; word
// pooling is an embedding_lookup_forward over the wsep global indices. One
// forward + one backward per step instead of one per sentence.

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

std::string sentence_text(const Sentence& s) {
    return s.bytes;
}

// ─── Weight init ──────────────────────────────────────────────────────────

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
    bt::Tensor cpu_W = bt::Tensor::mat(out_dim, in_dim);
    bt::xavier_init(cpu_W, rng_state);
    W = cpu_W.to(bt::default_device());
}

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
    for (auto& p : ps) {
        if (grad_scale != 1.0f) {
            bt::scale_inplace(p.grad, grad_scale);
        }
        bt::adam_step(*p.data, p.grad, p.m, p.v, lr, beta1, beta2, eps, step);
        if (p.decay && wd > 0.0f) {
            bt::scale_inplace(*p.data, 1.0f - lr * wd);
        }
    }
}

// ─── Packed-batch index helpers ───────────────────────────────────────────

// Upload an int32 host vector to an INT32 tensor on the current default
// device. Storage lives in `out_buf` so the caller controls lifetime.
// Returns the device-resident int32 pointer.
const std::int32_t* upload_idx(const std::int32_t* host, int n,
                               bt::Tensor& out_buf) {
    bt::Tensor cpu = bt::Tensor::zeros_on(bt::Device::CPU, n, 1,
                                          bt::Dtype::INT32);
    auto* p = static_cast<std::int32_t*>(cpu.host_raw_mut());
    for (int i = 0; i < n; ++i) p[i] = host[i];
    out_buf = cpu.to(bt::default_device());
    return static_cast<const std::int32_t*>(out_buf.data);
}

// Build a host-only INT32 tensor (used for cu_seqlens on CPU, where the
// varlen API wants host pointers).
bt::Tensor int_tensor_cpu(const std::vector<std::int32_t>& v) {
    bt::Tensor t = bt::Tensor::zeros_on(bt::Device::CPU,
                                        static_cast<int>(v.size()), 1,
                                        bt::Dtype::INT32);
    auto* p = static_cast<std::int32_t*>(t.host_raw_mut());
    for (std::size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

// ─── Packed batch ─────────────────────────────────────────────────────────

struct PackedBatch {
    int B            = 0;
    int total_tokens = 0;
    int total_words  = 0;
    int max_seqlen   = 0;

    // Device-resident packed tensors.
    bt::Tensor tokens;           // (total_tokens, 1) INT32
    bt::Tensor pos_ids;          // (total_tokens, 1) INT32
    bt::Tensor wsep_indices;     // (total_words, 1)  INT32
    bt::Tensor xpos_tags;        // (total_words, 1)  INT32  (host-side only used for loss)
    bt::Tensor upos_tags;        // (total_words, 1)  INT32

    // cu_seqlens: host vector + device tensor. The varlen API takes host
    // pointers on CPU and device pointers on CUDA/Metal; we keep both around
    // and pass the right one per device.
    std::vector<std::int32_t> cu_seqlens_host;   // (B+1)
    bt::Tensor                cu_seqlens_dev;    // (B+1, 1) INT32 — on default device

    // Host mirrors used by the loss (which is a host scalar op).
    std::vector<std::uint8_t> xtags_host;
    std::vector<std::uint8_t> utags_host;
};

const std::int32_t* cu_seqlens_ptr(const PackedBatch& pb) {
    if (bt::default_device() == bt::Device::CPU) {
        return pb.cu_seqlens_host.data();
    }
    return static_cast<const std::int32_t*>(pb.cu_seqlens_dev.data);
}

// ─── Byte-noise + word-dropout (host-side, on the packed buffer) ──────────

void apply_byte_noise(std::vector<std::int32_t>& ids, std::mt19937& rng,
                      float p_noise) {
    if (p_noise <= 0.0f) return;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uniform_int_distribution<int> b(4, 259);
    for (auto& t : ids) {
        if (t >= 4 && u(rng) < p_noise) t = b(rng);
    }
}

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

    for (int i = 0; i < wsep_positions[0]; ++i) new_ids.push_back(ids[i]);

    for (int wi = 0; wi < W; ++wi) {
        const int wsep_pos = wsep_positions[wi];
        const int next_pos = (wi + 1 < W) ? wsep_positions[wi + 1] : L;
        new_wsep.push_back(static_cast<std::int32_t>(new_ids.size()));
        new_ids.push_back(ids[wsep_pos]);
        const bool drop = (u(rng) < p_word);
        if (!drop) {
            for (int j = wsep_pos + 1; j < next_pos; ++j) {
                new_ids.push_back(ids[j]);
            }
        }
    }
    if (!ids.empty() && ids.back() == 2 /*kEos*/) {
        if (new_ids.empty() || new_ids.back() != 2) {
            new_ids.push_back(2);
        }
    }
    ids = std::move(new_ids);
    wsep_positions = std::move(new_wsep);
}

// ─── Packed-forward caches ────────────────────────────────────────────────

struct PackedLayerCache {
    bt::Tensor ln1_y;      // (T, D)  pre-attn ln output
    bt::Tensor ln1_xhat;
    bt::Tensor ln1_rstd;
    bt::Tensor Q, K, V;    // (T, D)
    bt::Tensor attn_out;   // (T, D) flash-attn output (pre Wo)
    bt::Tensor attn_proj;  // (T, D) after Wo (and scale)
    bt::Tensor h_mid;      // (T, D) residual stream after attn branch
    bt::Tensor ln2_y;
    bt::Tensor ln2_xhat;
    bt::Tensor ln2_rstd;
    bt::Tensor ffn1_pre;   // (T, ffn)
    bt::Tensor ffn1_act;   // (T, ffn)
    bt::Tensor ffn2_out;   // (T, D) (before scale)
};

struct PackedForwardCache {
    bt::Tensor h0;                 // (T, D) embedding sum
    std::vector<PackedLayerCache> layers;
    std::vector<std::uint8_t>     layer_active;  // 1 = ran, 0 = skipped
    float                         layer_eval_scale = 1.0f;
    bt::Tensor h_final;            // (T, D) after final LN
    bt::Tensor final_xhat;
    bt::Tensor final_rstd;
    bt::Tensor pooled;             // (W, D) gathered at wsep_indices
    bt::Tensor logits;             // (W, NUM_TAGS) XPOS
    bt::Tensor upos_logits;        // (W, NUM_UPOS_TAGS) aux
};

// ─── Packed forward ───────────────────────────────────────────────────────

void packed_forward(const g::PosWeights& w,
                    const PackedBatch& pb,
                    PackedForwardCache& fc,
                    const AuxHead& aux,
                    const std::vector<std::uint8_t>& layer_active,
                    float layer_eval_scale) {
    const int T = pb.total_tokens;
    const int D = w.d_model;
    const int Wn = pb.total_words;

    fc.layer_active     = layer_active;
    fc.layer_eval_scale = layer_eval_scale;

    // Embedding: token + position.
    bt::embedding_lookup_forward(w.token_emb,
        static_cast<const std::int32_t*>(pb.tokens.data), T, fc.h0);
    bt::Tensor pe;
    bt::embedding_lookup_forward(w.pos_emb,
        static_cast<const std::int32_t*>(pb.pos_ids.data), T, pe);
    bt::add_inplace_batched(fc.h0, pe);

    bt::Tensor h_resid = fc.h0.clone();

    fc.layers.assign(w.layers.size(), PackedLayerCache{});
    const std::int32_t* cuq = cu_seqlens_ptr(pb);

    for (std::size_t li = 0; li < w.layers.size(); ++li) {
        const auto& Lw = w.layers[li];
        auto&       C  = fc.layers[li];

        if (!fc.layer_active[li]) continue;

        // ln1 (batched, with caches)
        C.ln1_y    = bt::Tensor::zeros(T, D);
        C.ln1_xhat = bt::Tensor::zeros(T, D);
        bt::Tensor ln1_mean;
        bt::layernorm_forward_batched_with_caches(
            h_resid, Lw.ln1.gamma, Lw.ln1.beta,
            C.ln1_y, C.ln1_xhat, ln1_mean, C.ln1_rstd, Lw.ln1.eps);

        // Q, K, V projections — one linear over (T, D) each.
        C.Q = bt::Tensor::zeros(T, D);
        C.K = bt::Tensor::zeros(T, D);
        C.V = bt::Tensor::zeros(T, D);
        bt::linear_forward_batched(Lw.mha.Wq, Lw.mha.bq, C.ln1_y, C.Q);
        bt::linear_forward_batched(Lw.mha.Wk, Lw.mha.bk, C.ln1_y, C.K);
        bt::linear_forward_batched(Lw.mha.Wv, Lw.mha.bv, C.ln1_y, C.V);

        // Varlen flash attention — FP32 on all backends (dtype-dispatched).
        bt::flash_attention_varlen_forward(
            C.Q, C.K, C.V, cuq, cuq,
            pb.B, pb.max_seqlen, pb.max_seqlen,
            w.num_heads, w.d_model / w.num_heads,
            /*causal=*/false, C.attn_out);

        // Wo projection.
        C.attn_proj = bt::Tensor::zeros(T, D);
        bt::linear_forward_batched(Lw.mha.Wo, Lw.mha.bo, C.attn_out, C.attn_proj);
        if (fc.layer_eval_scale != 1.0f) {
            bt::scale_inplace(C.attn_proj, fc.layer_eval_scale);
        }
        bt::add_inplace_batched(h_resid, C.attn_proj);
        C.h_mid = h_resid.clone();

        // ln2
        C.ln2_y    = bt::Tensor::zeros(T, D);
        C.ln2_xhat = bt::Tensor::zeros(T, D);
        bt::Tensor ln2_mean;
        bt::layernorm_forward_batched_with_caches(
            h_resid, Lw.ln2.gamma, Lw.ln2.beta,
            C.ln2_y, C.ln2_xhat, ln2_mean, C.ln2_rstd, Lw.ln2.eps);

        // FFN1 → GELU → FFN2
        C.ffn1_pre = bt::Tensor::zeros(T, w.ffn_hidden);
        Lw.ffn1.forward_batched(C.ln2_y, C.ffn1_pre);
        C.ffn1_act = bt::Tensor::zeros(T, w.ffn_hidden);
        bt::gelu_forward(C.ffn1_pre, C.ffn1_act);
        C.ffn2_out = bt::Tensor::zeros(T, D);
        Lw.ffn2.forward_batched(C.ffn1_act, C.ffn2_out);
        if (fc.layer_eval_scale != 1.0f) {
            bt::scale_inplace(C.ffn2_out, fc.layer_eval_scale);
        }
        bt::add_inplace_batched(h_resid, C.ffn2_out);
    }

    // final LN
    fc.h_final    = bt::Tensor::zeros(T, D);
    fc.final_xhat = bt::Tensor::zeros(T, D);
    bt::Tensor final_mean;
    bt::layernorm_forward_batched_with_caches(
        h_resid, w.final_ln.gamma, w.final_ln.beta,
        fc.h_final, fc.final_xhat, final_mean, fc.final_rstd, w.final_ln.eps);

    // Word pool — gather at wsep global indices via embedding_lookup.
    bt::embedding_lookup_forward(fc.h_final,
        static_cast<const std::int32_t*>(pb.wsep_indices.data), Wn, fc.pooled);

    // Heads.
    fc.logits      = bt::Tensor::zeros(Wn, w.num_tags);
    w.head.forward_batched(fc.pooled, fc.logits);
    fc.upos_logits = bt::Tensor::zeros(Wn, aux.out_features);
    bt::linear_forward_batched(aux.W, aux.b, fc.pooled, fc.upos_logits);
}

// ─── ParamRefs ────────────────────────────────────────────────────────────

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

// ─── Packed backward ──────────────────────────────────────────────────────

void packed_backward(const g::PosWeights& w,
                     const PackedBatch& pb,
                     const PackedForwardCache& fc,
                     const bt::Tensor& dLogits,        // (W, NUM_TAGS)
                     const bt::Tensor& dUposLogits,    // (W, NUM_UPOS_TAGS)
                     ParamRefs& gr,
                     const AuxHead& aux,
                     bt::Tensor& gAuxW, bt::Tensor& gAuxB) {
    const int T = pb.total_tokens;
    const int D = w.d_model;
    const int Wn = pb.total_words;

    // Heads backward — both contribute to dPooled.
    bt::Tensor dPooled = bt::Tensor::zeros(Wn, D);
    bt::linear_backward_batched(w.head.W, fc.pooled, dLogits,
                                dPooled, *gr.head_W, *gr.head_b);
    bt::Tensor dPooled_upos = bt::Tensor::zeros(Wn, D);
    bt::linear_backward_batched(aux.W, fc.pooled, dUposLogits,
                                dPooled_upos, gAuxW, gAuxB);
    bt::add_inplace_batched(dPooled, dPooled_upos);

    // Pool backward: scatter-add into dH_final at wsep global indices.
    bt::Tensor dH_final = bt::Tensor::zeros(T, D);
    bt::embedding_lookup_backward(dPooled,
        static_cast<const std::int32_t*>(pb.wsep_indices.data), Wn, dH_final);

    // Final LN backward.
    bt::Tensor dH_resid = bt::Tensor::zeros(T, D);
    bt::layernorm_backward_batched_with_caches(
        dH_final, fc.final_xhat, w.final_ln.gamma, fc.final_rstd,
        dH_resid, *gr.final_ln_gamma, *gr.final_ln_beta);

    const std::int32_t* cuq = cu_seqlens_ptr(pb);

    for (int li = static_cast<int>(w.layers.size()) - 1; li >= 0; --li) {
        const auto& Lw = w.layers[li];
        const auto& C  = fc.layers[li];
        auto&       gL = gr.layers[li];

        if (!fc.layer_active[static_cast<std::size_t>(li)]) {
            // Residual = identity for the whole block; dH_resid passes through.
            continue;
        }

        // FFN residual: branch grad equals dH_resid.
        bt::Tensor dh_mid_resid = dH_resid.clone();

        // FFN2 backward.
        bt::Tensor dffn1_act = bt::Tensor::zeros(T, w.ffn_hidden);
        bt::linear_backward_batched(Lw.ffn2.W, C.ffn1_act, dH_resid,
                                    dffn1_act, *gL.ffn2_W, *gL.ffn2_b);
        bt::Tensor dffn1_pre = bt::Tensor::zeros(T, w.ffn_hidden);
        bt::gelu_backward(C.ffn1_pre, dffn1_act, dffn1_pre);
        bt::Tensor dln2_y = bt::Tensor::zeros(T, D);
        bt::linear_backward_batched(Lw.ffn1.W, C.ln2_y, dffn1_pre,
                                    dln2_y, *gL.ffn1_W, *gL.ffn1_b);
        bt::Tensor dh_mid_ffn = bt::Tensor::zeros(T, D);
        bt::layernorm_backward_batched_with_caches(
            dln2_y, C.ln2_xhat, Lw.ln2.gamma, C.ln2_rstd,
            dh_mid_ffn, *gL.ln2_g, *gL.ln2_b);
        bt::add_inplace_batched(dh_mid_resid, dh_mid_ffn);
        // dh_mid_resid is now d/d h_mid.

        // Attn residual: branch grad equals dh_mid_resid.
        bt::Tensor dh_in_resid = dh_mid_resid.clone();

        // Wo backward.
        bt::Tensor dattn_out = bt::Tensor::zeros(T, D);
        bt::linear_backward_batched(Lw.mha.Wo, C.attn_out, dh_mid_resid,
                                    dattn_out, *gL.Wo, *gL.bo);

        // Varlen attention backward — FP32 on all backends (dtype-dispatched).
        bt::Tensor dQ, dK, dV;
        bt::flash_attention_varlen_backward(
            C.Q, C.K, C.V, C.attn_out, dattn_out, cuq, cuq,
            pb.B, pb.max_seqlen, pb.max_seqlen,
            w.num_heads, w.d_model / w.num_heads,
            /*causal=*/false, dQ, dK, dV);

        // Q/K/V projection backwards → dln1_y is the sum of three branches.
        bt::Tensor dln1_y = bt::Tensor::zeros(T, D);
        bt::Tensor dln1_q = bt::Tensor::zeros(T, D);
        bt::Tensor dln1_k = bt::Tensor::zeros(T, D);
        bt::Tensor dln1_v = bt::Tensor::zeros(T, D);
        bt::linear_backward_batched(Lw.mha.Wq, C.ln1_y, dQ,
                                    dln1_q, *gL.Wq, *gL.bq);
        bt::linear_backward_batched(Lw.mha.Wk, C.ln1_y, dK,
                                    dln1_k, *gL.Wk, *gL.bk);
        bt::linear_backward_batched(Lw.mha.Wv, C.ln1_y, dV,
                                    dln1_v, *gL.Wv, *gL.bv);
        bt::add_inplace_batched(dln1_y, dln1_q);
        bt::add_inplace_batched(dln1_y, dln1_k);
        bt::add_inplace_batched(dln1_y, dln1_v);

        // ln1 backward.
        bt::Tensor dh_in_ln = bt::Tensor::zeros(T, D);
        bt::layernorm_backward_batched_with_caches(
            dln1_y, C.ln1_xhat, Lw.ln1.gamma, C.ln1_rstd,
            dh_in_ln, *gL.ln1_g, *gL.ln1_b);
        bt::add_inplace_batched(dh_in_resid, dh_in_ln);
        dH_resid = dh_in_resid;
    }

    // Embedding backward: scatter-add into token_emb and pos_emb grads.
    bt::embedding_lookup_backward(dH_resid,
        static_cast<const std::int32_t*>(pb.tokens.data), T, *gr.token_emb);
    bt::embedding_lookup_backward(dH_resid,
        static_cast<const std::int32_t*>(pb.pos_ids.data), T, *gr.pos_emb);
}

// ─── Loss (device-side fused softmax-xent) ────────────────────────────────

// Build a (Wn, C) FP32 one-hot target tensor on the default device.
bt::Tensor make_onehot_dev(const std::vector<std::uint8_t>& tags, int C) {
    const int Wn = static_cast<int>(tags.size());
    bt::Tensor cpu = bt::Tensor::zeros_on(bt::Device::CPU, Wn, C, bt::Dtype::FP32);
    auto* p = static_cast<float*>(cpu.host_raw_mut());
    for (int i = 0; i < Wn; ++i) {
        p[static_cast<std::size_t>(i) * C + tags[i]] = 1.0f;
    }
    return cpu.to(bt::default_device());
}

// head_offsets for a single-head call: length 2, {0, C}. CPU: host ptr; else
// upload to a device buffer kept alive by `dev_buf`. Returns the pointer to
// pass to softmax_xent_fused_batched.
const int* head_offsets_ptr(int C, std::vector<int>& host_buf,
                            bt::Tensor& dev_buf) {
    host_buf = {0, C};
    if (bt::default_device() == bt::Device::CPU) {
        return host_buf.data();
    }
    bt::Tensor cpu = bt::Tensor::zeros_on(bt::Device::CPU, 2, 1, bt::Dtype::INT32);
    auto* p = static_cast<std::int32_t*>(cpu.host_raw_mut());
    p[0] = 0; p[1] = C;
    dev_buf = cpu.to(bt::default_device());
    return static_cast<const int*>(dev_buf.data);
}

// Device-side fused softmax-xent over (Wn, C) logits with INT8 class targets.
// Writes dLogits on device; loss_per_sample is the per-row loss on device.
void xent_words_packed_dev(const bt::Tensor& logits,
                           const std::vector<std::uint8_t>& tags,
                           bt::Tensor& probs_dev,
                           bt::Tensor& dLogits_dev,
                           bt::Tensor& loss_per_sample_dev) {
    const int Wn = logits.rows;
    const int C  = logits.cols;
    bt::Tensor target = make_onehot_dev(tags, C);
    std::vector<int> hoff;
    bt::Tensor doff_buf;
    const int* hoff_ptr = head_offsets_ptr(C, hoff, doff_buf);
    probs_dev           = bt::Tensor::zeros(Wn, C);
    dLogits_dev         = bt::Tensor::zeros(Wn, C);
    loss_per_sample_dev = bt::Tensor::zeros(Wn, 1);
    bt::softmax_xent_fused_batched(logits, target, /*d_mask=*/nullptr,
                                   hoff_ptr, /*n_heads=*/1,
                                   probs_dev, dLogits_dev,
                                   loss_per_sample_dev);
}

// Sum-reduce a (Wn,1) device tensor into a (1,1) device tensor by
// matmul(ones(1,Wn), x(Wn,1)). No device→host sync.
void accum_scalar(const bt::Tensor& loss_per_sample,
                  bt::Tensor& running_scalar /* (1,1) */) {
    const int Wn = loss_per_sample.rows;
    if (Wn == 0) return;
    std::vector<float> ones_host(Wn, 1.0f);
    bt::Tensor ones = bt::Tensor::from_host_on(bt::default_device(),
                                               ones_host.data(), 1, Wn);
    bt::Tensor s = bt::Tensor::zeros(1, 1);
    bt::matmul(ones, loss_per_sample, s);
    bt::add_inplace(running_scalar, s);
}

// Loss-only variant used in eval: returns the host-side sum (single small sync
// per batch — eval is not throughput-critical, but we still keep it cheap).
float xent_only_dev(const bt::Tensor& logits,
                    const std::vector<std::uint8_t>& targets) {
    bt::Tensor probs, dLog, lps;
    xent_words_packed_dev(logits, targets, probs, dLog, lps);
    auto v = lps.to_host_vector();
    float total = 0.0f;
    for (float x : v) total += x;
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

void save_aux_head(const std::string& path, const AuxHead& h) {
    std::ofstream f(path, std::ios::binary);
    if (!f) fail("train_pos_tagger", "could not write aux head '" + path + "'");
    w_u32(f, 0x504F5303u);
    w_u32(f, 1u);
    w_u32(f, static_cast<std::uint32_t>(h.out_features));
    w_u32(f, static_cast<std::uint32_t>(h.in_features));
    w_u32(f, 2u);
    w_tensor(f, "upos_head.W", h.W);
    w_tensor(f, "upos_head.b", h.b);
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
            std::uint32_t h = 2166136261u;
            for (std::size_t k = 0; k < o.second; ++k) {
                h = (h ^ static_cast<unsigned char>(joined[o.first + k])) * 16777619u;
            }
            sent.tag_id.push_back(static_cast<std::uint8_t>(h % g::NUM_TAGS));
            std::uint32_t hu = h ^ 0x9E3779B9u;
            sent.upos_id.push_back(
                static_cast<std::uint8_t>(hu % g::NUM_UPOS_TAGS));
        }
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
        "train_pos_tagger — POS tagger training driver (packed varlen)\n"
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

// ─── Pack one minibatch ──────────────────────────────────────────────────

// Tokenise one sentence (yields >=1 chunks) and append it onto the packed
// host buffers. `apply_dropout` controls whether byte-noise + word-dropout
// are applied (true during training, false during evaluation).
//
// Returns the number of words successfully packed (== number of new wsep
// entries appended).
struct PackHostBuffers {
    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> pos_ids;
    std::vector<std::int32_t> wsep_global;
    std::vector<std::uint8_t> xtags;
    std::vector<std::uint8_t> utags;
    std::vector<std::int32_t> cu_seqlens{0};
    int max_seqlen = 0;
};

bool pack_sentence(const Sentence& s, PackHostBuffers& pb,
                   std::mt19937& rng,
                   float byte_noise, float word_dropout,
                   bool apply_dropout) {
    std::string text = sentence_text(s);
    auto chunks = g::tokenise_sentence(text);
    std::size_t word_idx = 0;
    bool any = false;
    for (const auto& c : chunks) {
        if (c.wsep_positions.empty()) {
            word_idx += c.word_spans.size();
            continue;
        }
        if (static_cast<int>(c.token_ids.size()) > g::kMaxSeqLen) {
            fail("train", "sentence exceeds max_seq_len after tokenisation");
        }

        std::vector<std::int32_t> ids  = c.token_ids;
        std::vector<std::int32_t> wsep = c.wsep_positions;
        std::vector<std::uint8_t> xt, ut;
        for (std::size_t i = 0; i < c.word_spans.size(); ++i, ++word_idx) {
            if (word_idx >= s.tag_id.size()) break;
            xt.push_back(s.tag_id[word_idx]);
            ut.push_back(s.upos_id[word_idx]);
        }
        if (xt.size() != wsep.size()) continue;

        if (apply_dropout) {
            apply_word_dropout(ids, wsep, rng, word_dropout);
            apply_byte_noise(ids, rng, byte_noise);
        }

        const int base = static_cast<int>(pb.tokens.size());
        for (std::size_t i = 0; i < ids.size(); ++i) {
            pb.tokens.push_back(ids[i]);
            pb.pos_ids.push_back(static_cast<std::int32_t>(i));
        }
        for (auto wp : wsep) {
            pb.wsep_global.push_back(base + wp);
        }
        for (auto t : xt) pb.xtags.push_back(t);
        for (auto t : ut) pb.utags.push_back(t);

        const int seqlen = static_cast<int>(ids.size());
        if (seqlen > pb.max_seqlen) pb.max_seqlen = seqlen;
        pb.cu_seqlens.push_back(pb.cu_seqlens.back() + seqlen);

        any = true;
    }
    return any;
}

// Build a device-resident PackedBatch from the host buffers.
PackedBatch finalize_pack(PackHostBuffers&& h) {
    PackedBatch pb;
    pb.B            = static_cast<int>(h.cu_seqlens.size()) - 1;
    pb.total_tokens = static_cast<int>(h.tokens.size());
    pb.total_words  = static_cast<int>(h.wsep_global.size());
    pb.max_seqlen   = h.max_seqlen;
    pb.xtags_host   = std::move(h.xtags);
    pb.utags_host   = std::move(h.utags);
    pb.cu_seqlens_host = h.cu_seqlens;

    // Upload device tensors.
    bt::Tensor tok_cpu = bt::Tensor::zeros_on(bt::Device::CPU,
                                              pb.total_tokens, 1, bt::Dtype::INT32);
    std::memcpy(tok_cpu.host_raw_mut(), h.tokens.data(),
                static_cast<std::size_t>(pb.total_tokens) * sizeof(std::int32_t));
    pb.tokens = tok_cpu.to(bt::default_device());

    bt::Tensor pos_cpu = bt::Tensor::zeros_on(bt::Device::CPU,
                                              pb.total_tokens, 1, bt::Dtype::INT32);
    std::memcpy(pos_cpu.host_raw_mut(), h.pos_ids.data(),
                static_cast<std::size_t>(pb.total_tokens) * sizeof(std::int32_t));
    pb.pos_ids = pos_cpu.to(bt::default_device());

    bt::Tensor wse_cpu = bt::Tensor::zeros_on(bt::Device::CPU,
                                              pb.total_words, 1, bt::Dtype::INT32);
    std::memcpy(wse_cpu.host_raw_mut(), h.wsep_global.data(),
                static_cast<std::size_t>(pb.total_words) * sizeof(std::int32_t));
    pb.wsep_indices = wse_cpu.to(bt::default_device());

    bt::Tensor cuq_cpu = bt::Tensor::zeros_on(bt::Device::CPU,
                                              static_cast<int>(h.cu_seqlens.size()), 1,
                                              bt::Dtype::INT32);
    std::memcpy(cuq_cpu.host_raw_mut(), h.cu_seqlens.data(),
                h.cu_seqlens.size() * sizeof(std::int32_t));
    pb.cu_seqlens_dev = cuq_cpu.to(bt::default_device());

    return pb;
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
    float val_loss;
};

EvalStats evaluate(const g::PosWeights& w, const AuxHead& aux,
                   const std::vector<Sentence>& data,
                   int batch, float layer_dropout, float upos_weight) {
    const float eval_scale = 1.0f - layer_dropout;
    const std::vector<std::uint8_t> layer_active_all(w.layers.size(), 1u);
    int total = 0, x_correct = 0, u_correct = 0;
    float loss_sum = 0.0f;
    std::mt19937 rng(0);  // unused at eval (no dropout)

    for (std::size_t i = 0; i < data.size(); i += static_cast<std::size_t>(batch)) {
        PackHostBuffers hb;
        std::size_t end = std::min(data.size(), i + static_cast<std::size_t>(batch));
        for (std::size_t j = i; j < end; ++j) {
            pack_sentence(data[j], hb, rng, 0.0f, 0.0f, /*apply_dropout=*/false);
        }
        if (hb.tokens.empty()) continue;

        PackedBatch pb = finalize_pack(std::move(hb));
        PackedForwardCache fc;
        packed_forward(w, pb, fc, aux, layer_active_all, eval_scale);

        bt::Tensor xi_dev, ui_dev;
        bt::argmax_rows(fc.logits, xi_dev);
        bt::argmax_rows(fc.upos_logits, ui_dev);
        auto xi = xi_dev.to_host_vector();
        auto ui = ui_dev.to_host_vector();

        loss_sum += xent_only_dev(fc.logits, pb.xtags_host);
        loss_sum += upos_weight * xent_only_dev(fc.upos_logits, pb.utags_host);

        for (int k = 0; k < pb.total_words; ++k) {
            if (static_cast<int>(xi[k]) == pb.xtags_host[k]) ++x_correct;
            if (static_cast<int>(ui[k]) == pb.utags_host[k]) ++u_correct;
            ++total;
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
    std::cout << "packed batches: B=" << a.batch << " (varlen)\n";

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

    bt::Tensor* gAuxW = nullptr;
    bt::Tensor* gAuxB = nullptr;
    for (auto& p : params) {
        if (p.name == "upos_head.W") gAuxW = &p.grad;
        if (p.name == "upos_head.b") gAuxB = &p.grad;
    }
    if (!gAuxW || !gAuxB) fail("train", "aux head grads not found");

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

        // Device-side accumulators avoid per-step host syncs of the loss.
        bt::Tensor epoch_xpos_dev = bt::Tensor::zeros(1, 1);
        bt::Tensor epoch_upos_dev = bt::Tensor::zeros(1, 1);
        int   epoch_words = 0;

        for (int step = 0; step < steps_per_epoch; ++step) {
            zero_grads(params);

            // Build packed batch.
            PackHostBuffers hb;
            for (int b = 0; b < a.batch; ++b) {
                const auto& s = train[order[(step * a.batch + b) % train.size()]];
                pack_sentence(s, hb, rng, a.byte_noise, a.word_dropout,
                              /*apply_dropout=*/true);
            }
            if (hb.tokens.empty() || hb.wsep_global.empty()) continue;
            PackedBatch pb = finalize_pack(std::move(hb));

            // Per-step stochastic-depth mask: whole batch shares the decision.
            std::vector<std::uint8_t> layer_active(g::kNumLayers, 1u);
            std::uniform_real_distribution<float> u01(0.0f, 1.0f);
            for (int li = 0; li < g::kNumLayers; ++li) {
                if (u01(rng) < a.layer_dropout) layer_active[li] = 0u;
            }

            PackedForwardCache fc;
            packed_forward(w, pb, fc, aux, layer_active, /*layer_eval_scale=*/1.0f);

            const float scale = 1.0f / static_cast<float>(pb.total_words);

            bt::Tensor xpos_probs, dLogits, xpos_lps;
            bt::Tensor upos_probs, dUposLogits, upos_lps;
            xent_words_packed_dev(fc.logits,      pb.xtags_host,
                                  xpos_probs, dLogits,     xpos_lps);
            xent_words_packed_dev(fc.upos_logits, pb.utags_host,
                                  upos_probs, dUposLogits, upos_lps);
            bt::scale_inplace(dLogits, scale);
            bt::scale_inplace(dUposLogits, scale * a.upos_weight);

            packed_backward(w, pb, fc, dLogits, dUposLogits, grefs,
                            aux, *gAuxW, *gAuxB);

            // Accumulate epoch loss on device — no per-step D→H sync.
            accum_scalar(xpos_lps, epoch_xpos_dev);
            accum_scalar(upos_lps, epoch_upos_dev);
            epoch_words += pb.total_words;

            const float lr = lr_schedule(global_step, total_steps, a.warmup, a.lr);
            adamw_step(params, lr, 0.9f, 0.98f, 1e-9f, /*wd=*/0.01f,
                       global_step, /*grad_scale=*/1.0f);

            if (global_step % 20 == 0) {
                // Sync just the small (Wn,1) per-sample losses for this step.
                auto xv = xpos_lps.to_host_vector();
                auto uv = upos_lps.to_host_vector();
                float xpos_loss_sum = 0.0f, upos_loss_sum = 0.0f;
                for (float x : xv) xpos_loss_sum += x;
                for (float x : uv) upos_loss_sum += x;
                const float loss_sum = xpos_loss_sum + a.upos_weight * upos_loss_sum;
                std::cout << "epoch " << epoch << " step " << global_step
                          << "/" << total_steps
                          << " xpos_loss " << std::fixed << std::setprecision(4)
                          << (xpos_loss_sum / pb.total_words)
                          << " upos_loss " << (upos_loss_sum / pb.total_words)
                          << " total "     << (loss_sum / pb.total_words)
                          << " lr " << std::scientific << std::setprecision(2) << lr
                          << std::defaultfloat << std::endl;
            }
            ++global_step;
        }

        const EvalStats es = evaluate(w, aux, val, a.batch, a.layer_dropout, a.upos_weight);
        const float xpos_total = epoch_xpos_dev.to_host_vector().at(0);
        const float upos_total = epoch_upos_dev.to_host_vector().at(0);
        const float epoch_loss = xpos_total + a.upos_weight * upos_total;
        std::cout << "epoch " << epoch
                  << " mean_loss " << std::fixed << std::setprecision(4)
                  << (epoch_loss / std::max(1, epoch_words))
                  << " val_loss " << es.val_loss
                  << " xpos_val_acc " << es.xpos_acc
                  << " upos_val_acc " << es.upos_acc
                  << std::defaultfloat << std::endl;

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

    std::string final_ckpt = best_ckpt.empty() ? "" : best_ckpt;
    if (final_ckpt.empty()) {
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
