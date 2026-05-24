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

}  // namespace brosoundml
