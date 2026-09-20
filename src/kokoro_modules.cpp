#include "kokoro_internal.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt  = brotensor;
namespace stf = brotensor::safetensors;

// Target device for the file-local `upload` helper. The Kokoro `load_from`
// chain doesn't thread a `device` argument through (the public header takes
// only `(File, KokoroConfig)`), so kokoro.cpp calls `set_kokoro_load_device`
// here before invoking the chain. safetensors::upload lands tensors on CPU;
// without this migration step they stay on CPU even when the rest of the
// model runs on CUDA, and brotensor's dispatcher then refuses the mixed call.
// This is the upload-time half of the systemic CUDA-port bug.
bt::Device g_kokoro_load_device = bt::Device::CPU;

void set_kokoro_load_device(bt::Device d) { g_kokoro_load_device = d; }

// ─── Stage profiler ─────────────────────────────────────────────────────────
//
// Env-gated (BROSOUNDML_KOKORO_PROFILE=1) sequential interval stamps: each
// mark prints the wall time since the previous mark on this thread, syncing
// the device first so async backends attribute cost to the stage that ran.
// A nullptr name resets the interval origin without printing (call it at the
// top of synthesize). Shared by kokoro.cpp via forward declaration.
bool kokoro_profile_enabled() {
    static const bool on = []() {
        const char* v = std::getenv("BROSOUNDML_KOKORO_PROFILE");
        return v && v[0] && v[0] != '0';
    }();
    return on;
}

void kokoro_profile_mark(bt::Device dev, const char* name) {
    if (!kokoro_profile_enabled()) return;
    bt::sync(dev);
    static thread_local std::chrono::steady_clock::time_point last =
        std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (name) {
        const double ms =
            std::chrono::duration<double, std::milli>(now - last).count();
        std::fprintf(stderr, "[kokoro-prof] %-24s %9.2f ms\n", name, ms);
    }
    last = now;
}


// Upload a host int32 index buffer to a (n, 1) INT32 tensor on `dev`.
// brotensor lacks `Tensor::from_host_int32_on`, so we reach through the
// public memcpy_h2d hook on the backend's vtable. CUDA's
// `embedding_lookup_forward` reads `d_idx` as a DEVICE pointer, so calls that
// previously passed `host_vector.data()` straight through crashed CUDA with
// an illegal memory access.
bt::Tensor upload_int32_idx(bt::Device dev, const std::int32_t* host_idx, int n) {
    bt::Tensor t = bt::Tensor::empty_on(dev, n, 1, bt::Dtype::INT32);
    if (n == 0) return t;
    if (dev == bt::Device::CPU) {
        std::memcpy(t.data, host_idx, static_cast<std::size_t>(n) * sizeof(std::int32_t));
    } else {
        bt::detail::alloc_for(dev).memcpy_h2d(
            t.data, host_idx,
            static_cast<std::size_t>(n) * sizeof(std::int32_t), dev.index);
    }
    return t;
}



[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

// Upload a tensor named `key` from `f` into `dst` reshaped to (rows, cols).
// rows*cols must equal the safetensors tensor's element count; the source
// dtype must be F32 (Kokoro's converted checkpoint). After the host-side
// safetensors upload, migrate `dst` to `g_kokoro_load_device` so the rest of
// the module sees device-matched weights.
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
    // safetensors::upload uses Tensor::from_host which lands on
    // brotensor::default_device() — that's CUDA once init() registers it, not
    // CPU. Migrate unconditionally to g_kokoro_load_device so the model's
    // weights end up exactly where forward() expects them.
    if (dst.device != g_kokoro_load_device) {
        dst = dst.to(g_kokoro_load_device);
    }
}

// Per-row LayerNorm over a (N, D) batch of features. Wraps the device-aware
// brotensor::layernorm_forward_inference_batched — one launch instead of N
// scalar-row calls. Y is resized to (N, D); X.device drives dispatch.
void layernorm_rows(int N, int D,
                    const bt::Tensor& gamma, const bt::Tensor& beta,
                    float eps, const bt::Tensor& X, bt::Tensor& Y) {
    // Tensor::resize preserves the existing device field; a caller that
    // default-constructed Y would leave it on CPU and crash CUDA dispatch.
    // Migrate Y to X.device before the resize so the realloc lands correctly.
    if (Y.device != X.device) {
        Y = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    }
    Y.resize(N, D, bt::Dtype::FP32);
    bt::layernorm_forward_inference_batched(X, gamma, beta, Y, eps);
}


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
    // Pre-allocate every op-out tensor on the model's device (taken from the
    // embedding tables — they share the device every other weight was loaded
    // onto). Default-constructed Tensors live on CPU and brotensor's CUDA
    // dispatch refuses mixed-device calls; Tensor::resize preserves device,
    // so a (0, 0) seed is enough for ops that re-size their out param.
    const bt::Device dev = word_embeddings.device;

    bt::Tensor word_emb = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    // CUDA `embedding_lookup_forward` reads `d_idx` as a DEVICE pointer, so on
    // CUDA we stage indices through `upload_int32_idx`. On CPU we keep the
    // host pointer to avoid a needless allocation.
    bt::Tensor word_idx;
    const std::int32_t* word_idx_ptr;
    if (dev == bt::Device::CPU) {
        word_idx_ptr = input_ids.data();
    } else {
        word_idx = upload_int32_idx(dev, input_ids.data(), L);
        word_idx_ptr = static_cast<const std::int32_t*>(word_idx.data);
    }
    bt::embedding_lookup_forward(word_embeddings, word_idx_ptr, L, word_emb);

    std::vector<int32_t> pos_ids(L);
    std::iota(pos_ids.begin(), pos_ids.end(), 0);
    bt::Tensor pos_emb = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor pos_idx;
    const std::int32_t* pos_idx_ptr;
    if (dev == bt::Device::CPU) {
        pos_idx_ptr = pos_ids.data();
    } else {
        pos_idx = upload_int32_idx(dev, pos_ids.data(), L);
        pos_idx_ptr = static_cast<const std::int32_t*>(pos_idx.data);
    }
    bt::embedding_lookup_forward(position_embeddings, pos_idx_ptr, L, pos_emb);

    std::vector<int32_t> tt_ids(L, 0);
    bt::Tensor tt_emb = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor tt_idx;
    const std::int32_t* tt_idx_ptr;
    if (dev == bt::Device::CPU) {
        tt_idx_ptr = tt_ids.data();
    } else {
        tt_idx = upload_int32_idx(dev, tt_ids.data(), L);
        tt_idx_ptr = static_cast<const std::int32_t*>(tt_idx.data);
    }
    bt::embedding_lookup_forward(token_type_embeddings, tt_idx_ptr, L, tt_emb);

    // emb_sum = word + pos + tt.
    bt::Tensor emb_sum = word_emb;        // owns its own buffer (copy-ctor)
    bt::add_inplace(emb_sum, pos_emb);
    bt::add_inplace(emb_sum, tt_emb);

    bt::Tensor emb = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);  // (L, E)
    layernorm_rows(L, E, emb_layernorm.gamma, emb_layernorm.beta,
                   emb_layernorm.eps, emb_sum, emb);

    // Project to hidden_size.
    bt::Tensor hidden = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);  // (L, H)
    bt::linear_forward_batched(emb_to_hidden.W, emb_to_hidden.b, emb, hidden);

    // ─── Shared ALBERT layer × num_hidden_layers ───────────────────────────
    bt::Tensor Q_proj    = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor K_proj    = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor V_proj    = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor attn_out  = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor ffn_in    = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor ffn_act   = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor ffn_out_t = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor hidden_next = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    for (int layer = 0; layer < cfg.num_hidden_layers; ++layer) {
        // attn_out = MHA(hidden) -- with biases on Q, K, V, dense.
        bt::linear_forward_batched(attn_q.W, attn_q.b, hidden, Q_proj);
        bt::linear_forward_batched(attn_k.W, attn_k.b, hidden, K_proj);
        bt::linear_forward_batched(attn_v.W, attn_v.b, hidden, V_proj);
        mha_attention_fp32(Q_proj, K_proj, V_proj, cfg.num_attention_heads,
                           attn_dense.W, attn_dense.b, d_mask, attn_out);
        // Residual + attention LayerNorm.
        bt::add_inplace(attn_out, hidden);
        bt::Tensor attn_normed = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
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
        hidden_next = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
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
    // Pre-allocate every op-out on bert_dur.device so brotensor's dispatch
    // sees matched devices (Tensor::resize preserves device).
    bt::Tensor projected = bt::Tensor::empty_on(bert_dur.device, 0, 0, bt::Dtype::FP32);
    bt::linear_forward_batched(projection.W, projection.b, bert_dur, projected);
    const int L  = projected.rows;
    const int Co = projected.cols;
    // Transpose (L, C) -> (1, C*L) NCL via sequence_to_nchw (the inverse of
    // nchw_to_sequence): NLC (L, C) maps to NCHW with N=1, H=1, W=L.
    if (d_en.device != bert_dur.device) {
        d_en = bt::Tensor::empty_on(bert_dur.device, 0, 0, bt::Dtype::FP32);
    }
    bt::sequence_to_nchw(projected, /*N=*/1, /*C=*/Co, /*H=*/1, /*W=*/L, d_en);
}

// ─── layernorm_1d_ncl ──────────────────────────────────────────────────────

void layernorm_1d_ncl(const bt::Tensor& X,
                      const bt::Tensor& gamma, const bt::Tensor& beta,
                      int N, int C, int L, float eps,
                      bt::Tensor& Y) {
    const std::string where = "layernorm_1d_ncl";
    if (X.dtype != bt::Dtype::FP32) fail(where, "FP32 only");
    if (X.rows != N || X.cols != C * L) fail(where, "X shape mismatch");
    if (gamma.rows * gamma.cols != C || beta.rows * beta.cols != C) {
        fail(where, "gamma/beta must hold C elements");
    }
    // Compose: NCL -> (N*L, C) via nchw_to_sequence (H=1, W=L); per-row
    // LayerNorm over the C axis with the existing batched op; then back to
    // NCL via sequence_to_nchw. All three ops dispatch on X.device.
    bt::Tensor X_seq = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    bt::nchw_to_sequence(X, N, C, /*H=*/1, /*W=*/L, X_seq);
    // Pre-allocate Y_seq on X.device. Tensor::resize preserves device; a
    // default-constructed (CPU) Y_seq would crash CUDA dispatch.
    bt::Tensor Y_seq = bt::Tensor::empty_on(X.device, N * L, C, bt::Dtype::FP32);
    bt::layernorm_forward_inference_batched(X_seq, gamma, beta, Y_seq, eps);
    if (Y.device != X.device) {
        Y = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    }
    bt::sequence_to_nchw(Y_seq, N, C, /*H=*/1, /*W=*/L, Y);
}

// ─── TextEncoder ───────────────────────────────────────────────────────────


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
}

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

    // Embedding lookup -> (L, C) NLC. The lookup runs on `embedding`'s device.
    // Pre-allocate on embedding.device so the dispatch sees a matched out
    // tensor (Tensor::resize preserves device).
    const bt::Device dev = embedding.device;
    bt::Tensor x_nlc = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    // CUDA `embedding_lookup_forward` reads `d_idx` as a DEVICE pointer.
    bt::Tensor idx_dev;
    const std::int32_t* idx_ptr;
    if (dev == bt::Device::CPU) {
        idx_ptr = input_ids.data();
    } else {
        idx_dev = upload_int32_idx(dev, input_ids.data(), L);
        idx_ptr = static_cast<const std::int32_t*>(idx_dev.data);
    }
    bt::embedding_lookup_forward(embedding, idx_ptr, L, x_nlc);

    // Transpose to NCL: (1, C*L). NLC (L, C) -> NCHW with N=1, H=1, W=L.
    bt::Tensor x = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::sequence_to_nchw(x_nlc, /*N=*/1, /*C=*/C, /*H=*/1, /*W=*/L, x);

    // Apply mask: positions where text_mask == 1 (a *pad* mask) zero out the
    // feature vector. text_mask in PyTorch Kokoro is the PADDING mask (true =
    // pad, false = valid). Build a per-(n, l) keep-mask on host (1=keep,
    // 0=zero), upload to `dev`, and multiply channel-wise via broadcast_mul
    // through the NCL -> NLC -> NCL composition. No-op when mask is empty.
    auto apply_mask = [&](bt::Tensor& y, int Cy, int Ly) {
        if (text_mask.empty()) return;
        // The mask is the same length-Ly for the single batch row; turn it
        // into a (Ly, 1) keep tensor on `dev`, broadcast-multiplied into a
        // (Ly, Cy) NLC view, then sent back to NCL.
        std::vector<float> keep_host(Ly, 1.0f);
        for (int l = 0; l < Ly && l < static_cast<int>(text_mask.size()); ++l) {
            if (text_mask[l] != 0) keep_host[l] = 0.0f;
        }
        bt::Tensor y_seq = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        bt::nchw_to_sequence(y, /*N=*/1, Cy, /*H=*/1, /*W=*/Ly, y_seq);  // (Ly, Cy)
        bt::Tensor keep_col = bt::Tensor::from_host_on(dev, keep_host.data(), Ly, 1);
        // Multiply each row of y_seq by keep_col[l]: do it manually as
        // per-row scale via copy_d2d/scale — but the cleanest device-aware
        // op is to fold the mask into a length-Cy vector per row. Cheaper:
        // broadcast keep_col across Cy by expanding to a (Ly, Cy) factor
        // and using mul_inplace.
        std::vector<float> factor_host(static_cast<std::size_t>(Ly) * Cy);
        for (int l = 0; l < Ly; ++l) {
            const float k = keep_host[l];
            for (int c = 0; c < Cy; ++c) factor_host[l * Cy + c] = k;
        }
        bt::Tensor factor = bt::Tensor::from_host_on(dev, factor_host.data(),
                                                    Ly, Cy);
        bt::mul_inplace(y_seq, factor);
        bt::sequence_to_nchw(y_seq, /*N=*/1, Cy, /*H=*/1, /*W=*/Ly, y);
    };
    apply_mask(x, C, L);

    // depth × (Conv1d + LayerNorm1dNCL + LeakyReLU + mask).
    for (int i = 0; i < depth; ++i) {
        bt::Tensor x_conv = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        cnn[i].forward(x, /*N=*/1, /*L=*/L, x_conv);  // (1, C*L)
        bt::Tensor x_ln = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        layernorm_1d_ncl(x_conv, ln_gamma[i], ln_beta[i],
                         /*N=*/1, C, L, 1e-5f, x_ln);
        bt::Tensor x_act = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        bt::leaky_relu_forward(x_ln, 0.2f, x_act);
        apply_mask(x_act, C, L);
        x = std::move(x_act);
    }

    // Transpose to (L, C) for the LSTM via nchw_to_sequence (H=1, W=L).
    bt::Tensor x_lc = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::nchw_to_sequence(x, /*N=*/1, C, /*H=*/1, /*W=*/L, x_lc);

    // BiLSTM: (L, C) -> (L, C). (hidden_size = C/2 per direction; concat = C.)
    // LSTM::forward in modules.cpp already pre-allocates Y on X.device, so the
    // default-CPU `lstm_out` is safe here — but seeding it consistently keeps
    // the pattern uniform across the file.
    bt::Tensor lstm_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    lstm.forward(x_lc, lstm_out);

    // Transpose back to (1, C*L) NCL via sequence_to_nchw.
    if (t_en.device != dev) {
        t_en = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    }
    bt::sequence_to_nchw(lstm_out, /*N=*/1, C, /*H=*/1, /*W=*/L, t_en);
}


}  // namespace brosoundml
