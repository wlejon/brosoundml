#include "brosoundml/kokoro_modules.h"

#include <brotensor/ops.h>
#include <brotensor/safetensors.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt  = brotensor;
namespace stf = brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

// Upload a tensor named `key` from `f` into `dst` reshaped to (rows, cols).
// rows*cols must equal the safetensors tensor's element count; the source
// dtype must be F32 (Kokoro's converted checkpoint).
void upload(const stf::File& f, const std::string& key,
            int rows, int cols, bt::Tensor& dst,
            const std::string& where) {
    const stf::TensorView* view = f.find(key);
    if (!view) fail(where, "missing safetensors key '" + key + "'");
    if (view->dtype != stf::Dtype::F32) {
        fail(where, "tensor '" + key + "' is not F32 (got dtype " +
                    std::to_string(static_cast<int>(view->dtype)) + ")");
    }
    const int64_t n = view->numel();
    if (n != static_cast<int64_t>(rows) * cols) {
        fail(where, "tensor '" + key + "' has " + std::to_string(n) +
                    " elements; expected " +
                    std::to_string(static_cast<int64_t>(rows) * cols));
    }
    stf::upload(*view, rows, cols, dst);
}

// Per-row LayerNorm over a (N, D) batch of features. Wraps the existing
// brosoundml::LayerNorm but skips its sanity checks — used inside the inner
// loops where shapes are known to match.
void layernorm_rows(int N, int D,
                    const bt::Tensor& gamma, const bt::Tensor& beta,
                    float eps, const bt::Tensor& X, bt::Tensor& Y) {
    Y.resize(N, D, bt::Dtype::FP32);
    bt::Tensor xhat;
    float mean_out = 0.0f, rstd_out = 0.0f;
    for (int r = 0; r < N; ++r) {
        const auto offset_bytes = static_cast<std::size_t>(r) * D * sizeof(float);
        bt::Tensor x_row = bt::Tensor::view(bt::Device::CPU,
                                            static_cast<std::uint8_t*>(X.data) + offset_bytes,
                                            D, 1, bt::Dtype::FP32);
        bt::Tensor y_row = bt::Tensor::view(bt::Device::CPU,
                                            static_cast<std::uint8_t*>(Y.data) + offset_bytes,
                                            D, 1, bt::Dtype::FP32);
        bt::layernorm_forward(x_row, gamma, beta, y_row, xhat,
                              mean_out, rstd_out, eps);
    }
}

// Add `b` broadcast across all rows: `out[r, c] += b[c]`. b is (D, 1) or (1, D).
void add_bias_rows(bt::Tensor& out, const bt::Tensor& b, int rows, int cols) {
    float* o = out.host_f32_mut();
    const float* bd = b.host_f32();
    for (int r = 0; r < rows; ++r) {
        float* row = o + static_cast<std::size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) row[c] += bd[c];
    }
}

// Standard multi-head scaled-dot-product attention with biases on every
// projection (Q, K, V, dense) — the ALBERT layout. Implemented from primitives
// because brotensor's mha_forward has no Q/K/V bias slot.
//
// hidden: (L, D). num_heads divides D. head_dim = D/num_heads. mask is null
// for our test input (all positions valid); a non-null length-L mask gates
// keys by adding -inf to the masked scores before softmax.
void mha_with_bias(const bt::Tensor& hidden, int L, int D, int num_heads,
                   const bt::Tensor& Wq, const bt::Tensor& bq,
                   const bt::Tensor& Wk, const bt::Tensor& bk,
                   const bt::Tensor& Wv, const bt::Tensor& bv,
                   const bt::Tensor& Wo, const bt::Tensor& bo,
                   const float* d_mask,
                   bt::Tensor& out) {
    const int head_dim = D / num_heads;
    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Q, K, V projections — each (L, D).
    bt::Tensor Q, K, V;
    bt::linear_forward_batched(Wq, bq, hidden, Q);
    bt::linear_forward_batched(Wk, bk, hidden, K);
    bt::linear_forward_batched(Wv, bv, hidden, V);

    const float* Qd = Q.host_f32();
    const float* Kd = K.host_f32();
    const float* Vd = V.host_f32();

    // Per-head attention context: ctx is (L, D), filled by stacking heads
    // along the feature axis at offsets `h*head_dim`.
    bt::Tensor ctx = bt::Tensor::zeros_on(bt::Device::CPU, L, D, bt::Dtype::FP32);
    float* ctx_d = ctx.host_f32_mut();

    // Reused row buffer for softmax-per-query.
    std::vector<float> scores(L, 0.0f);

    for (int h = 0; h < num_heads; ++h) {
        // Pointers into the head's slice of Q, K, V.
        // Q[L, D] -> Q_h slice: q[r, h*head_dim + j] for r in [0,L), j in [0,head_dim).
        for (int q = 0; q < L; ++q) {
            const float* q_vec = Qd + static_cast<std::size_t>(q) * D + h * head_dim;
            // scores[k] = (q . K_h[k]) * scale; mask out invalid keys.
            float max_score = -INFINITY;
            for (int k = 0; k < L; ++k) {
                const float* k_vec = Kd + static_cast<std::size_t>(k) * D + h * head_dim;
                float s = 0.0f;
                for (int j = 0; j < head_dim; ++j) s += q_vec[j] * k_vec[j];
                s *= scale;
                if (d_mask && d_mask[k] < 0.5f) s = -1e30f;
                scores[k] = s;
                if (s > max_score) max_score = s;
            }
            // Softmax (numerically stable).
            float sum = 0.0f;
            for (int k = 0; k < L; ++k) {
                scores[k] = std::exp(scores[k] - max_score);
                sum += scores[k];
            }
            const float inv = sum > 0 ? 1.0f / sum : 0.0f;
            for (int k = 0; k < L; ++k) scores[k] *= inv;

            // ctx_h[q] = sum_k scores[k] * V_h[k]; write into ctx[q, h*head_dim+j].
            float* ctx_row = ctx_d + static_cast<std::size_t>(q) * D + h * head_dim;
            for (int k = 0; k < L; ++k) {
                const float w = scores[k];
                const float* v_vec = Vd + static_cast<std::size_t>(k) * D + h * head_dim;
                for (int j = 0; j < head_dim; ++j) ctx_row[j] += w * v_vec[j];
            }
        }
    }

    // Output projection: out = ctx @ Wo^T + bo.
    bt::linear_forward_batched(Wo, bo, ctx, out);
}

}  // namespace

// ─── PLBert ────────────────────────────────────────────────────────────────

void PLBert::load_from(const stf::File& f, const PLBertConfig& c) {
    cfg = c;
    const std::string p = "bert.module.";
    const std::string where = "PLBert::load_from";
    // Embeddings live at embedding_size=128 (factorised); we infer it from the
    // word-embedding shape rather than threading another config field in.
    const stf::TensorView* word = f.find(p + "embeddings.word_embeddings.weight");
    if (!word) fail(where, "missing " + p + "embeddings.word_embeddings.weight");
    if (word->shape.size() != 2) fail(where, "word_embeddings is not 2D");
    const int vocab_size     = static_cast<int>(word->shape[0]);
    const int embedding_size = static_cast<int>(word->shape[1]);
    if (vocab_size != cfg.vocab_size) {
        fail(where, "word_embeddings rows=" + std::to_string(vocab_size) +
                    " != plbert.vocab_size=" + std::to_string(cfg.vocab_size));
    }

    upload(f, p + "embeddings.word_embeddings.weight",
           vocab_size, embedding_size, word_embeddings, where);
    upload(f, p + "embeddings.position_embeddings.weight",
           cfg.max_position_embeddings, embedding_size, position_embeddings, where);
    upload(f, p + "embeddings.token_type_embeddings.weight",
           2, embedding_size, token_type_embeddings, where);

    emb_layernorm.features = embedding_size;
    emb_layernorm.eps      = 1e-12f;  // HF Albert default
    upload(f, p + "embeddings.LayerNorm.weight",
           embedding_size, 1, emb_layernorm.gamma, where);
    upload(f, p + "embeddings.LayerNorm.bias",
           embedding_size, 1, emb_layernorm.beta, where);

    emb_to_hidden.in_features  = embedding_size;
    emb_to_hidden.out_features = cfg.hidden_size;
    upload(f, p + "encoder.embedding_hidden_mapping_in.weight",
           cfg.hidden_size, embedding_size, emb_to_hidden.W, where);
    upload(f, p + "encoder.embedding_hidden_mapping_in.bias",
           cfg.hidden_size, 1, emb_to_hidden.b, where);

    // The single shared ALBERT layer.
    const std::string lp = p + "encoder.albert_layer_groups.0.albert_layers.0.";
    auto load_linear = [&](Linear& lin, const std::string& name,
                           int out_features, int in_features) {
        lin.in_features  = in_features;
        lin.out_features = out_features;
        upload(f, lp + name + ".weight", out_features, in_features, lin.W, where);
        upload(f, lp + name + ".bias",   out_features, 1,           lin.b, where);
    };
    load_linear(attn_q,     "attention.query", cfg.hidden_size, cfg.hidden_size);
    load_linear(attn_k,     "attention.key",   cfg.hidden_size, cfg.hidden_size);
    load_linear(attn_v,     "attention.value", cfg.hidden_size, cfg.hidden_size);
    load_linear(attn_dense, "attention.dense", cfg.hidden_size, cfg.hidden_size);

    attn_layernorm.features = cfg.hidden_size;
    attn_layernorm.eps      = 1e-12f;
    upload(f, lp + "attention.LayerNorm.weight",
           cfg.hidden_size, 1, attn_layernorm.gamma, where);
    upload(f, lp + "attention.LayerNorm.bias",
           cfg.hidden_size, 1, attn_layernorm.beta, where);

    load_linear(ffn,        "ffn",        cfg.intermediate_size, cfg.hidden_size);
    load_linear(ffn_output, "ffn_output", cfg.hidden_size,       cfg.intermediate_size);

    full_layernorm.features = cfg.hidden_size;
    full_layernorm.eps      = 1e-12f;
    upload(f, lp + "full_layer_layer_norm.weight",
           cfg.hidden_size, 1, full_layernorm.gamma, where);
    upload(f, lp + "full_layer_layer_norm.bias",
           cfg.hidden_size, 1, full_layernorm.beta, where);
}

void PLBert::forward(const std::vector<int32_t>& input_ids,
                     const std::vector<int>& attention_mask,
                     bt::Tensor& bert_dur) const {
    const std::string where = "PLBert::forward";
    const int L  = static_cast<int>(input_ids.size());
    if (L <= 0) fail(where, "empty input_ids");
    if (L > cfg.max_position_embeddings) {
        fail(where, "input length " + std::to_string(L) +
                    " exceeds max_position_embeddings " +
                    std::to_string(cfg.max_position_embeddings));
    }
    const int E = word_embeddings.cols;       // embedding_size (128)
    const int H = cfg.hidden_size;            // 768
    const int F = cfg.intermediate_size;      // 2048

    // Build mask. If empty or all-1, use null (no masking) inside the attention.
    std::vector<float> mask_f;
    const float* d_mask = nullptr;
    if (!attention_mask.empty()) {
        bool any_zero = false;
        for (int v : attention_mask) if (v == 0) { any_zero = true; break; }
        if (any_zero) {
            mask_f.assign(attention_mask.begin(), attention_mask.end());
            d_mask = mask_f.data();
        }
    }

    // ─── Embedding lookup + LayerNorm ──────────────────────────────────────
    bt::Tensor word_emb;
    bt::embedding_lookup_forward(word_embeddings, input_ids.data(), L, word_emb);

    std::vector<int32_t> pos_ids(L);
    std::iota(pos_ids.begin(), pos_ids.end(), 0);
    bt::Tensor pos_emb;
    bt::embedding_lookup_forward(position_embeddings, pos_ids.data(), L, pos_emb);

    std::vector<int32_t> tt_ids(L, 0);
    bt::Tensor tt_emb;
    bt::embedding_lookup_forward(token_type_embeddings, tt_ids.data(), L, tt_emb);

    // emb_sum = word + pos + tt.
    bt::Tensor emb_sum = word_emb;        // owns its own buffer (copy-ctor)
    bt::add_inplace(emb_sum, pos_emb);
    bt::add_inplace(emb_sum, tt_emb);

    bt::Tensor emb;                       // (L, E)
    layernorm_rows(L, E, emb_layernorm.gamma, emb_layernorm.beta,
                   emb_layernorm.eps, emb_sum, emb);

    // Project to hidden_size.
    bt::Tensor hidden;                    // (L, H)
    bt::linear_forward_batched(emb_to_hidden.W, emb_to_hidden.b, emb, hidden);

    // ─── Shared ALBERT layer × num_hidden_layers ───────────────────────────
    bt::Tensor attn_out, ffn_in, ffn_act, ffn_out_t, hidden_next;
    for (int layer = 0; layer < cfg.num_hidden_layers; ++layer) {
        // attn_out = MHA(hidden) -- with biases on Q, K, V, dense.
        mha_with_bias(hidden, L, H, cfg.num_attention_heads,
                      attn_q.W, attn_q.b, attn_k.W, attn_k.b,
                      attn_v.W, attn_v.b, attn_dense.W, attn_dense.b,
                      d_mask, attn_out);
        // Residual + attention LayerNorm.
        bt::add_inplace(attn_out, hidden);
        bt::Tensor attn_normed;
        layernorm_rows(L, H, attn_layernorm.gamma, attn_layernorm.beta,
                       attn_layernorm.eps, attn_out, attn_normed);

        // FFN: ffn -> GELU -> ffn_output.
        bt::linear_forward_batched(ffn.W, ffn.b, attn_normed, ffn_in);
        bt::gelu_forward(ffn_in, ffn_act);
        bt::linear_forward_batched(ffn_output.W, ffn_output.b, ffn_act, ffn_out_t);

        // Residual + full layer LayerNorm.
        bt::add_inplace(ffn_out_t, attn_normed);
        layernorm_rows(L, H, full_layernorm.gamma, full_layernorm.beta,
                       full_layernorm.eps, ffn_out_t, hidden_next);
        // Swap for next layer.
        hidden = std::move(hidden_next);
    }

    bert_dur = std::move(hidden);
}

// ─── BertEncoder ───────────────────────────────────────────────────────────

void BertEncoder::load_from(const stf::File& f, int bert_hidden, int out_hidden) {
    const std::string where = "BertEncoder::load_from";
    const std::string p = "bert_encoder.module.";
    projection.in_features  = bert_hidden;
    projection.out_features = out_hidden;
    upload(f, p + "weight", out_hidden, bert_hidden, projection.W, where);
    upload(f, p + "bias",   out_hidden, 1,           projection.b, where);
}

void BertEncoder::forward(const bt::Tensor& bert_dur, bt::Tensor& d_en) const {
    // bert_dur: (L, bert_hidden) -> (L, out_hidden), then transpose to NCL.
    bt::Tensor projected;
    bt::linear_forward_batched(projection.W, projection.b, bert_dur, projected);
    const int L  = projected.rows;
    const int Co = projected.cols;
    // Transpose (L, C) -> (1, C, L) flattened as (1, C*L) row-major.
    d_en.resize(1, Co * L, bt::Dtype::FP32);
    const float* src = projected.host_f32();
    float* dst = d_en.host_f32_mut();
    for (int c = 0; c < Co; ++c) {
        for (int l = 0; l < L; ++l) {
            dst[c * L + l] = src[l * Co + c];
        }
    }
}

// ─── layernorm_1d_ncl ──────────────────────────────────────────────────────

void layernorm_1d_ncl(const bt::Tensor& X,
                      const bt::Tensor& gamma, const bt::Tensor& beta,
                      int N, int C, int L, float eps,
                      bt::Tensor& Y) {
    const std::string where = "layernorm_1d_ncl";
    if (X.device != bt::Device::CPU || X.dtype != bt::Dtype::FP32) {
        fail(where, "CPU FP32 only");
    }
    if (X.rows != N || X.cols != C * L) fail(where, "X shape mismatch");
    if (gamma.rows * gamma.cols != C || beta.rows * beta.cols != C) {
        fail(where, "gamma/beta must hold C elements");
    }
    Y.resize(N, C * L, bt::Dtype::FP32);
    const float* xd = X.host_f32();
    float* yd = Y.host_f32_mut();
    const float* gd = gamma.host_f32();
    const float* bd = beta.host_f32();
    for (int n = 0; n < N; ++n) {
        for (int l = 0; l < L; ++l) {
            // Length-C vector at (n, :, l): elements at flat index (n*C + c)*L + l.
            float mean = 0.0f, sumsq = 0.0f;
            for (int c = 0; c < C; ++c) {
                const float v = xd[(static_cast<std::size_t>(n) * C + c) * L + l];
                mean += v;
                sumsq += v * v;
            }
            mean /= C;
            const float var  = sumsq / C - mean * mean;
            const float rstd = 1.0f / std::sqrt(var + eps);
            for (int c = 0; c < C; ++c) {
                const std::size_t idx = (static_cast<std::size_t>(n) * C + c) * L + l;
                const float xhat = (xd[idx] - mean) * rstd;
                yd[idx] = xhat * gd[c] + bd[c];
            }
        }
    }
}

// ─── TextEncoder ───────────────────────────────────────────────────────────

namespace {

// Helper for loading PyTorch nn.LSTM weights stored under the prefix
// `<prefix>weight_{ih,hh}_l0{,_reverse}` + `bias_{ih,hh}_l0{,_reverse}`.
void load_lstm_cell(const stf::File& f, const std::string& prefix,
                    int input_size, int hidden, bool reverse,
                    LSTMCellWeights& cell, const std::string& where) {
    const std::string sfx = reverse ? "_reverse" : "";
    const int four_h = 4 * hidden;
    upload(f, prefix + "weight_ih_l0" + sfx, four_h, input_size, cell.W_ih, where);
    upload(f, prefix + "weight_hh_l0" + sfx, four_h, hidden,     cell.W_hh, where);
    upload(f, prefix + "bias_ih_l0"   + sfx, four_h, 1,          cell.b_ih, where);
    upload(f, prefix + "bias_hh_l0"   + sfx, four_h, 1,          cell.b_hh, where);
}

}  // namespace

void TextEncoder::load_from(const stf::File& f, const KokoroConfig& cfg) {
    const std::string where = "TextEncoder::load_from";
    const std::string p = "text_encoder.module.";

    channels    = cfg.hidden_dim;
    n_symbols   = cfg.n_tokens;
    kernel_size = cfg.text_encoder_kernel_size;
    depth       = cfg.n_layer;

    upload(f, p + "embedding.weight", n_symbols, channels, embedding, where);

    cnn.clear();
    ln_gamma.clear();
    ln_beta.clear();
    cnn.reserve(depth);
    ln_gamma.reserve(depth);
    ln_beta.reserve(depth);
    const int pad = (kernel_size - 1) / 2;
    for (int i = 0; i < depth; ++i) {
        Conv1d c{};
        c.in_channels  = channels;
        c.out_channels = channels;
        c.kernel_size  = kernel_size;
        c.padding      = pad;
        c.stride       = 1;
        c.dilation     = 1;
        c.groups       = 1;
        const std::string cp = p + "cnn." + std::to_string(i) + ".0.";
        // Conv1d weight stored as (C_out, C_in, kL) — flatten to (C_out, C_in*kL).
        upload(f, cp + "weight", channels, channels * kernel_size, c.W, where);
        upload(f, cp + "bias",   channels, 1,                      c.b, where);
        cnn.push_back(std::move(c));

        bt::Tensor g, b;
        const std::string lp = p + "cnn." + std::to_string(i) + ".1.";
        upload(f, lp + "gamma", channels, 1, g, where);
        upload(f, lp + "beta",  channels, 1, b, where);
        ln_gamma.push_back(std::move(g));
        ln_beta.push_back(std::move(b));
    }

    lstm.input_size  = channels;
    lstm.hidden_size = channels / 2;
    load_lstm_cell(f, p + "lstm.", channels, channels / 2, false,
                   lstm.forward_cell, where);
    load_lstm_cell(f, p + "lstm.", channels, channels / 2, true,
                   lstm.reverse_cell, where);
}

void TextEncoder::forward(const std::vector<int32_t>& input_ids,
                          const std::vector<int>& text_mask,
                          bt::Tensor& t_en) const {
    const std::string where = "TextEncoder::forward";
    const int L = static_cast<int>(input_ids.size());
    if (L <= 0) fail(where, "empty input_ids");
    const int C = channels;

    // Embedding lookup -> (L, C) NLC.
    bt::Tensor x_nlc;
    bt::embedding_lookup_forward(embedding, input_ids.data(), L, x_nlc);

    // Transpose to NCL: (1, C*L).
    bt::Tensor x = bt::Tensor::zeros_on(bt::Device::CPU, 1, C * L, bt::Dtype::FP32);
    {
        const float* src = x_nlc.host_f32();
        float* dst = x.host_f32_mut();
        for (int l = 0; l < L; ++l) {
            for (int c = 0; c < C; ++c) {
                dst[c * L + l] = src[l * C + c];
            }
        }
    }

    // Apply mask: positions where text_mask == 1 (a *pad* mask) zero out the
    // feature vector. text_mask in PyTorch Kokoro is the PADDING mask (true =
    // pad, false = valid). The vector handed in mirrors that convention.
    auto apply_mask = [&](bt::Tensor& y, int N, int Cy, int Ly) {
        if (text_mask.empty()) return;
        float* yd = y.host_f32_mut();
        for (int n = 0; n < N; ++n) {
            for (int l = 0; l < Ly; ++l) {
                if (l < static_cast<int>(text_mask.size()) && text_mask[l] != 0) {
                    for (int c = 0; c < Cy; ++c) {
                        yd[(static_cast<std::size_t>(n) * Cy + c) * Ly + l] = 0.0f;
                    }
                }
            }
        }
    };
    apply_mask(x, 1, C, L);

    // depth × (Conv1d + LayerNorm1dNCL + LeakyReLU + mask).
    for (int i = 0; i < depth; ++i) {
        bt::Tensor x_conv;
        cnn[i].forward(x, /*N=*/1, /*L=*/L, x_conv);  // (1, C*L)
        bt::Tensor x_ln;
        layernorm_1d_ncl(x_conv, ln_gamma[i], ln_beta[i],
                         /*N=*/1, C, L, 1e-5f, x_ln);
        bt::Tensor x_act;
        bt::leaky_relu_forward(x_ln, 0.2f, x_act);
        apply_mask(x_act, 1, C, L);
        x = std::move(x_act);
    }

    // Transpose to (L, C) for the LSTM.
    bt::Tensor x_lc = bt::Tensor::zeros_on(bt::Device::CPU, L, C, bt::Dtype::FP32);
    {
        const float* src = x.host_f32();
        float* dst = x_lc.host_f32_mut();
        for (int c = 0; c < C; ++c) {
            for (int l = 0; l < L; ++l) {
                dst[l * C + c] = src[c * L + l];
            }
        }
    }

    // BiLSTM: (L, C) -> (L, C). (hidden_size = C/2 per direction; concat = C.)
    bt::Tensor lstm_out;
    lstm.forward(x_lc, lstm_out);

    // Transpose back to (C, L) and return as t_en flattened (1, C*L).
    t_en.resize(1, C * L, bt::Dtype::FP32);
    {
        const float* src = lstm_out.host_f32();
        float* dst = t_en.host_f32_mut();
        for (int l = 0; l < L; ++l) {
            for (int c = 0; c < C; ++c) {
                dst[c * L + l] = src[l * C + c];
            }
        }
    }
}

// ─── Style-conditioned norms ───────────────────────────────────────────────

namespace {

// Compute (gamma, beta) = chunk(fc(style)) for an AdaLayerNorm / AdaIN1d.
// Returns two length-C views over a freshly-allocated (1, 2*C) buffer. The
// caller keeps `scratch` alive until the views go out of scope.
void compute_style_affine(const Linear& fc, int C,
                          const bt::Tensor& style, bt::Tensor& scratch,
                          const float*& gamma_out, const float*& beta_out) {
    bt::linear_forward_batched(fc.W, fc.b, style, scratch);  // (1, 2*C)
    const float* h = scratch.host_f32();
    gamma_out = h;
    beta_out  = h + C;
}

// AdaLayerNorm forward on (L, C):
//   y = (1 + gamma) * LayerNorm_no_affine(x_row, C) + beta
void ada_layernorm(const AdaLayerNormWeights& w, int L,
                   const bt::Tensor& x_lc, const bt::Tensor& style,
                   bt::Tensor& y_lc) {
    bt::Tensor scratch;
    const float* gamma = nullptr;
    const float* beta  = nullptr;
    compute_style_affine(w.fc, w.channels, style, scratch, gamma, beta);
    const int C = w.channels;
    y_lc.resize(L, C, bt::Dtype::FP32);
    const float* xd = x_lc.host_f32();
    float* yd       = y_lc.host_f32_mut();
    for (int l = 0; l < L; ++l) {
        const float* row = xd + static_cast<std::size_t>(l) * C;
        float mean = 0.0f, sumsq = 0.0f;
        for (int c = 0; c < C; ++c) { mean += row[c]; sumsq += row[c] * row[c]; }
        mean /= C;
        const float var  = sumsq / C - mean * mean;
        const float rstd = 1.0f / std::sqrt(var + w.eps);
        float* y_row = yd + static_cast<std::size_t>(l) * C;
        for (int c = 0; c < C; ++c) {
            const float xhat = (row[c] - mean) * rstd;
            y_row[c] = (1.0f + gamma[c]) * xhat + beta[c];
        }
    }
}

// AdaIN1d forward on (1, C*L) NCL:
//   per channel c: y[n,c,l] = (1 + gamma[c]) * (x[n,c,l] - mean_c)/std_c + beta[c]
//   where mean_c / std_c are taken over the L axis (instance norm).
void ada_in_1d_styled(const AdaIN1dWeights& w, int N, int C, int L,
                      const bt::Tensor& x_ncl, const bt::Tensor& style,
                      bt::Tensor& y_ncl) {
    bt::Tensor scratch;
    const float* gamma = nullptr;
    const float* beta  = nullptr;
    compute_style_affine(w.fc, w.channels, style, scratch, gamma, beta);
    y_ncl.resize(N, C * L, bt::Dtype::FP32);
    const float* xd = x_ncl.host_f32();
    float* yd       = y_ncl.host_f32_mut();
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const std::size_t base = (static_cast<std::size_t>(n) * C + c) * L;
            float mean = 0.0f, sumsq = 0.0f;
            for (int l = 0; l < L; ++l) { mean += xd[base + l]; sumsq += xd[base + l] * xd[base + l]; }
            mean /= L;
            const float var  = sumsq / L - mean * mean;
            const float rstd = 1.0f / std::sqrt(var + w.eps);
            for (int l = 0; l < L; ++l) {
                const float xhat = (xd[base + l] - mean) * rstd;
                yd[base + l] = (1.0f + gamma[c]) * xhat + beta[c];
            }
        }
    }
}

// Per-(L) leaky_relu on NCL.
void leaky_relu_ncl(bt::Tensor& y, float slope) {
    bt::Tensor tmp;
    bt::leaky_relu_forward(y, slope, tmp);
    y = std::move(tmp);
}

// Nearest-neighbour 2x upsample along L: (1, C*L_in) NCL -> (1, C*(2*L_in)) NCL.
void upsample_nearest_2x_ncl(const bt::Tensor& x, int N, int C, int L_in,
                             bt::Tensor& y) {
    const int L_out = 2 * L_in;
    y.resize(N, C * L_out, bt::Dtype::FP32);
    const float* xd = x.host_f32();
    float* yd       = y.host_f32_mut();
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const std::size_t bx = (static_cast<std::size_t>(n) * C + c) * L_in;
            const std::size_t by = (static_cast<std::size_t>(n) * C + c) * L_out;
            for (int l = 0; l < L_in; ++l) {
                const float v = xd[bx + l];
                yd[by + 2 * l + 0] = v;
                yd[by + 2 * l + 1] = v;
            }
        }
    }
}

// AdainResBlk1d forward: residual + shortcut, both / sqrt(2). x in NCL with
// N=1; L_out = upsample ? 2*L_in : L_in. dropout is no-op at inference.
void adain_resblk_1d_forward(const AdainResBlk1dWeights& w,
                             const bt::Tensor& x, int L_in,
                             const bt::Tensor& style,
                             int& L_out, bt::Tensor& y) {
    const int C_in  = w.channels_in;
    const int C_out = w.channels_out;
    L_out = w.upsample ? 2 * L_in : L_in;

    // ─── residual path ────────────────────────────────────────────────────
    bt::Tensor r;
    ada_in_1d_styled(w.norm1, 1, C_in, L_in, x, style, r);
    leaky_relu_ncl(r, 0.2f);

    bt::Tensor pooled;
    if (w.upsample) {
        // Depthwise ConvTranspose1d: groups = C_in = C_out (when depthwise on dim_in).
        // Weight layout (C_in, (C_out_per_group=1) * kL=3) = (C_in, 3).
        bt::conv_transpose1d_forward(r, w.pool_W, &w.pool_b,
                                     /*N=*/1, /*C_in=*/C_in, /*L=*/L_in,
                                     /*C_out=*/C_in, /*kL=*/3,
                                     /*stride=*/2, /*padding=*/1,
                                     /*output_padding=*/1, /*dilation=*/1,
                                     /*groups=*/C_in, pooled);
    } else {
        pooled = std::move(r);
    }

    bt::Tensor conv1_out;
    w.conv1.forward(pooled, /*N=*/1, /*L=*/L_out, conv1_out);

    bt::Tensor n2;
    ada_in_1d_styled(w.norm2, 1, C_out, L_out, conv1_out, style, n2);
    leaky_relu_ncl(n2, 0.2f);

    bt::Tensor conv2_out;
    w.conv2.forward(n2, /*N=*/1, /*L=*/L_out, conv2_out);

    // ─── shortcut path ────────────────────────────────────────────────────
    bt::Tensor short_x;
    if (w.upsample) {
        upsample_nearest_2x_ncl(x, 1, C_in, L_in, short_x);
    } else {
        short_x = x;
    }
    if (w.learned_sc) {
        bt::Tensor sc;
        w.conv1x1.forward(short_x, /*N=*/1, /*L=*/L_out, sc);
        short_x = std::move(sc);
    }

    // ─── combine: y = (residual + shortcut) / sqrt(2) ─────────────────────
    bt::add_inplace(conv2_out, short_x);
    bt::scale_inplace(conv2_out, 1.0f / std::sqrt(2.0f));
    y = std::move(conv2_out);
}

// Load an AdaIN1d weight set from `<prefix>.fc.{weight,bias}`.
void load_ada_in_1d(const stf::File& f, const std::string& prefix,
                    int channels, int style_dim, AdaIN1dWeights& w,
                    const std::string& where) {
    w.channels  = channels;
    w.style_dim = style_dim;
    w.fc.in_features  = style_dim;
    w.fc.out_features = 2 * channels;
    upload(f, prefix + ".fc.weight", 2 * channels, style_dim, w.fc.W, where);
    upload(f, prefix + ".fc.bias",   2 * channels, 1,         w.fc.b, where);
}

void load_ada_layernorm(const stf::File& f, const std::string& prefix,
                        int channels, int style_dim, AdaLayerNormWeights& w,
                        const std::string& where) {
    w.channels  = channels;
    w.style_dim = style_dim;
    w.fc.in_features  = style_dim;
    w.fc.out_features = 2 * channels;
    upload(f, prefix + ".fc.weight", 2 * channels, style_dim, w.fc.W, where);
    upload(f, prefix + ".fc.bias",   2 * channels, 1,         w.fc.b, where);
}

void load_conv1d(const stf::File& f, const std::string& prefix,
                 int C_out, int C_in, int kL, bool has_bias,
                 Conv1d& c, const std::string& where) {
    c.in_channels  = C_in;
    c.out_channels = C_out;
    c.kernel_size  = kL;
    c.padding      = (kL == 3) ? 1 : 0;   // matches AdainResBlk1d's hardcoded paddings
    c.stride       = 1;
    c.dilation     = 1;
    c.groups       = 1;
    upload(f, prefix + ".weight", C_out, C_in * kL, c.W, where);
    if (has_bias) {
        upload(f, prefix + ".bias", C_out, 1, c.b, where);
    }
}

void load_adain_resblk(const stf::File& f, const std::string& prefix,
                       int dim_in, int dim_out, int style_dim, bool upsample,
                       AdainResBlk1dWeights& w, const std::string& where) {
    w.channels_in  = dim_in;
    w.channels_out = dim_out;
    w.upsample     = upsample;
    w.learned_sc   = (dim_in != dim_out);

    load_ada_in_1d(f, prefix + ".norm1", dim_in,  style_dim, w.norm1, where);
    load_ada_in_1d(f, prefix + ".norm2", dim_out, style_dim, w.norm2, where);
    load_conv1d   (f, prefix + ".conv1", dim_out, dim_in,  3, true, w.conv1, where);
    load_conv1d   (f, prefix + ".conv2", dim_out, dim_out, 3, true, w.conv2, where);
    if (w.learned_sc) {
        load_conv1d(f, prefix + ".conv1x1", dim_out, dim_in, 1, false, w.conv1x1, where);
    }
    if (upsample) {
        // Depthwise ConvTranspose1d: weight shape (C_in, 1, 3) -> flatten to (C_in, 3).
        upload(f, prefix + ".pool.weight", dim_in, 3,  w.pool_W, where);
        upload(f, prefix + ".pool.bias",   dim_in, 1,  w.pool_b, where);
    }
}

// Transpose (1, C*L) NCL <-> (L, C) NLC.
void ncl_to_lc(const bt::Tensor& x_ncl, int C, int L, bt::Tensor& x_lc) {
    x_lc.resize(L, C, bt::Dtype::FP32);
    const float* src = x_ncl.host_f32();
    float* dst       = x_lc.host_f32_mut();
    for (int c = 0; c < C; ++c) {
        for (int l = 0; l < L; ++l) {
            dst[l * C + c] = src[c * L + l];
        }
    }
}
void lc_to_ncl(const bt::Tensor& x_lc, int L, int C, bt::Tensor& x_ncl) {
    x_ncl.resize(1, C * L, bt::Dtype::FP32);
    const float* src = x_lc.host_f32();
    float* dst       = x_ncl.host_f32_mut();
    for (int l = 0; l < L; ++l) {
        for (int c = 0; c < C; ++c) {
            dst[c * L + l] = src[l * C + c];
        }
    }
}

}  // namespace

// ─── DurationEncoder ───────────────────────────────────────────────────────

void DurationEncoder::forward(const bt::Tensor& d_en, const bt::Tensor& style,
                              int L, bt::Tensor& d) const {
    const int C = channels;
    const int S = style_dim;

    // (1, C*L) NCL -> (L, C) NLC.
    bt::Tensor x;
    ncl_to_lc(d_en, C, L, x);

    // Concat style across L: (L, C) -> (L, C + S).
    auto cat_with_style = [&](const bt::Tensor& xs, bt::Tensor& out) {
        out.resize(L, C + S, bt::Dtype::FP32);
        const float* xd = xs.host_f32();
        const float* sd = style.host_f32();
        float* od = out.host_f32_mut();
        for (int l = 0; l < L; ++l) {
            float* row = od + static_cast<std::size_t>(l) * (C + S);
            std::memcpy(row, xd + static_cast<std::size_t>(l) * C,
                        static_cast<std::size_t>(C) * sizeof(float));
            std::memcpy(row + C, sd, static_cast<std::size_t>(S) * sizeof(float));
        }
    };

    bt::Tensor cat;
    cat_with_style(x, cat);

    for (const auto& blk : blocks) {
        // BiLSTM step: (L, C+S) -> (L, C).
        bt::Tensor lstm_out;
        blk.bilstm.forward(cat, lstm_out);

        // AdaLayerNorm step: (L, C) -> (L, C).
        bt::Tensor aln_out;
        ada_layernorm(blk.aln, L, lstm_out, style, aln_out);

        // Concat style back: (L, C) + style -> (L, C+S).
        cat_with_style(aln_out, cat);
    }

    d = std::move(cat);  // (L, C + S)
}

// ─── Predictor ─────────────────────────────────────────────────────────────

void Predictor::load_from(const stf::File& f, const KokoroConfig& c) {
    cfg = c;
    const std::string where = "Predictor::load_from";
    const std::string p = "predictor.module.";
    const int C = c.hidden_dim;       // 512
    const int S = c.style_dim;        // 128
    const int H = C / 2;              // BiLSTM hidden = 256

    // ─── DurationEncoder ──────────────────────────────────────────────────
    text_encoder.channels  = C;
    text_encoder.style_dim = S;
    text_encoder.nlayers   = c.n_layer;
    text_encoder.blocks.clear();
    text_encoder.blocks.reserve(c.n_layer);
    for (int i = 0; i < c.n_layer; ++i) {
        DurationEncoder::Block blk;
        // lstms.{2*i}: BiLSTM input=C+S, hidden=H.
        blk.bilstm.input_size  = C + S;
        blk.bilstm.hidden_size = H;
        const std::string lp = p + "text_encoder.lstms." + std::to_string(2 * i) + ".";
        load_lstm_cell(f, lp, C + S, H, false, blk.bilstm.forward_cell, where);
        load_lstm_cell(f, lp, C + S, H, true,  blk.bilstm.reverse_cell, where);

        const std::string ap = p + "text_encoder.lstms." + std::to_string(2 * i + 1);
        load_ada_layernorm(f, ap, C, S, blk.aln, where);
        text_encoder.blocks.push_back(std::move(blk));
    }

    // ─── Duration LSTM + proj ─────────────────────────────────────────────
    lstm.input_size  = C + S;
    lstm.hidden_size = H;
    load_lstm_cell(f, p + "lstm.", C + S, H, false, lstm.forward_cell, where);
    load_lstm_cell(f, p + "lstm.", C + S, H, true,  lstm.reverse_cell, where);

    duration_proj.in_features  = C;
    duration_proj.out_features = c.max_dur;
    upload(f, p + "duration_proj.linear_layer.weight",
           c.max_dur, C, duration_proj.W, where);
    upload(f, p + "duration_proj.linear_layer.bias",
           c.max_dur, 1, duration_proj.b, where);

    // ─── Shared BiLSTM ────────────────────────────────────────────────────
    shared.input_size  = C + S;
    shared.hidden_size = H;
    load_lstm_cell(f, p + "shared.", C + S, H, false, shared.forward_cell, where);
    load_lstm_cell(f, p + "shared.", C + S, H, true,  shared.reverse_cell, where);

    // ─── F0 / N blocks (3 each, with the middle one upsampling) ───────────
    auto load_f0n = [&](const std::string& prefix,
                        std::vector<AdainResBlk1dWeights>& blocks) {
        blocks.clear();
        blocks.resize(3);
        load_adain_resblk(f, prefix + ".0", C,     C,     S, /*upsample=*/false, blocks[0], where);
        load_adain_resblk(f, prefix + ".1", C,     C / 2, S, /*upsample=*/true,  blocks[1], where);
        load_adain_resblk(f, prefix + ".2", C / 2, C / 2, S, /*upsample=*/false, blocks[2], where);
    };
    load_f0n(p + "F0", F0_blocks);
    load_f0n(p + "N",  N_blocks);

    // ─── Final 1x1 conv projections (C/2 -> 1) ────────────────────────────
    load_conv1d(f, p + "F0_proj", /*C_out=*/1, /*C_in=*/C / 2, /*kL=*/1, true,
                F0_proj, where);
    load_conv1d(f, p + "N_proj",  /*C_out=*/1, /*C_in=*/C / 2, /*kL=*/1, true,
                N_proj, where);
    F0_proj.padding = 0;
    N_proj.padding  = 0;
}

void Predictor::forward(const bt::Tensor& d_en, const bt::Tensor& ref_s,
                        int L, float speed, Output& out) const {
    const int C = cfg.hidden_dim;
    const int S = cfg.style_dim;

    // Style for predictor = ref_s[:, style_dim:].
    bt::Tensor style = bt::Tensor::zeros_on(bt::Device::CPU, 1, S, bt::Dtype::FP32);
    {
        const float* rs = ref_s.host_f32();
        std::memcpy(style.host_f32_mut(), rs + S,
                    static_cast<std::size_t>(S) * sizeof(float));
    }

    // ─── DurationEncoder ──────────────────────────────────────────────────
    text_encoder.forward(d_en, style, L, out.d);  // (L, C + S)

    // ─── Duration LSTM + projection ──────────────────────────────────────
    lstm.forward(out.d, out.lstm_x);              // (L, C)
    duration_proj.forward_batched(out.lstm_x, out.duration);  // (L, max_dur)

    // sigmoid + sum over the max_dur axis, / speed, round, clamp to >= 1.
    out.pred_dur.assign(L, 0);
    const float* dur = out.duration.host_f32();
    int total = 0;
    for (int l = 0; l < L; ++l) {
        float s = 0.0f;
        for (int k = 0; k < cfg.max_dur; ++k) {
            const float v = dur[l * cfg.max_dur + k];
            s += 1.0f / (1.0f + std::exp(-v));
        }
        s /= speed;
        int rounded = static_cast<int>(std::round(s));
        if (rounded < 1) rounded = 1;
        out.pred_dur[l] = rounded;
        total += rounded;
    }

    // ─── Length regulator ────────────────────────────────────────────────
    // en = d.T @ alignment, where alignment is (L, total) one-hot expansion.
    // Equivalently: en[c, t] = d[phoneme(t), c] (with `phoneme(t)` mapping
    // each frame t back to the source phoneme).
    out.en.resize(1, (C + S) * total, bt::Dtype::FP32);
    {
        const float* dd = out.d.host_f32();
        float* en_d = out.en.host_f32_mut();
        int t = 0;
        for (int l = 0; l < L; ++l) {
            const int reps = out.pred_dur[l];
            for (int r = 0; r < reps; ++r) {
                for (int c = 0; c < C + S; ++c) {
                    en_d[c * total + t] = dd[l * (C + S) + c];
                }
                ++t;
            }
        }
    }

    // ─── F0Ntrain ────────────────────────────────────────────────────────
    // shared LSTM input = en.T  -> (total, C+S) -> (total, C). Then split into
    // F0 path and N path, each a stack of 3 AdaINResBlk1d (one upsamples).
    bt::Tensor en_lc;
    ncl_to_lc(out.en, C + S, total, en_lc);  // (total, C+S)

    bt::Tensor shared_out;
    shared.forward(en_lc, shared_out);        // (total, C)

    bt::Tensor shared_ncl;
    lc_to_ncl(shared_out, total, C, shared_ncl);  // (1, C*total) NCL

    // F0 stack.
    bt::Tensor F0_x = shared_ncl;
    int F0_L = total;
    int next_L;
    for (const auto& blk : F0_blocks) {
        bt::Tensor y;
        adain_resblk_1d_forward(blk, F0_x, F0_L, style, next_L, y);
        F0_x = std::move(y);
        F0_L = next_L;
    }
    // F0_proj: Conv1d(C/2 -> 1, k=1). Output (1, 1*F0_L).
    F0_proj.forward(F0_x, /*N=*/1, /*L=*/F0_L, out.F0_pred);

    // N stack.
    bt::Tensor N_x = shared_ncl;
    int N_L = total;
    for (const auto& blk : N_blocks) {
        bt::Tensor y;
        adain_resblk_1d_forward(blk, N_x, N_L, style, next_L, y);
        N_x = std::move(y);
        N_L = next_L;
    }
    N_proj.forward(N_x, /*N=*/1, /*L=*/N_L, out.N_pred);
}

}  // namespace brosoundml
