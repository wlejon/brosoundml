#include "brosoundml/whisper_modules.h"

#include "qwen_tts_device.h"

#include <brotensor/cuda_graph.h>
#include <brotensor/detail/dispatch.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
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

thread_local bt::Device g_load_device = bt::Device::CPU;

void upload(const stf::File& f, const std::string& key,
            int rows, int cols, bt::Tensor& dst,
            const std::string& where) {
    const stf::TensorView* view = f.find(key);
    if (!view) fail(where, "missing safetensors key '" + key + "'");
    const std::int64_t n = view->numel();
    if (n != static_cast<std::int64_t>(rows) * cols) {
        fail(where, "tensor '" + key + "' has " + std::to_string(n) +
                    " elements; expected " +
                    std::to_string(static_cast<std::int64_t>(rows) * cols));
    }

    if (view->dtype == stf::Dtype::F32) {
        stf::upload(*view, rows, cols, dst);
    } else if (view->dtype == stf::Dtype::F16 || view->dtype == stf::Dtype::BF16) {
        const bool isF16 = (view->dtype == stf::Dtype::F16);
        const auto* src = reinterpret_cast<const std::uint16_t*>(view->data);
        std::vector<float> tmp(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
            tmp[static_cast<std::size_t>(i)] =
                isF16 ? bt::fp16_bits_to_fp32(src[i]) : bt::bf16_bits_to_fp32(src[i]);
        dst = bt::Tensor::empty_on(bt::Device::CPU, rows, cols, bt::Dtype::FP32);
        std::memcpy(dst.data, tmp.data(), tmp.size() * sizeof(float));
    } else {
        fail(where, "tensor '" + key + "' has unsupported dtype " +
                    std::to_string(static_cast<int>(view->dtype)) +
                    " (need F32, F16, or BF16)");
    }
    if (dst.device != g_load_device) {
        dst = dst.to(g_load_device);
    }
}

void load_linear(const stf::File& f, const std::string& prefix,
                 int out_features, int in_features,
                 Linear& lin, const std::string& where) {
    lin.in_features  = in_features;
    lin.out_features = out_features;
    upload(f, prefix + ".weight", out_features, in_features, lin.W, where);
    upload(f, prefix + ".bias",   out_features, 1,           lin.b, where);
}

void load_layernorm(const stf::File& f, const std::string& prefix,
                    int features, float eps,
                    LayerNorm& ln, const std::string& where) {
    ln.features = features;
    ln.eps      = eps;
    upload(f, prefix + ".weight", features, 1, ln.gamma, where);
    upload(f, prefix + ".bias",   features, 1, ln.beta,  where);
}

}  // namespace

// ─── WhisperKVCache ────────────────────────────────────────────────────────

void WhisperKVCache::allocate(int decoder_layers, int d_model,
                              int max_target_positions, int max_source_positions,
                              bt::Device device) {
    const std::string where = "WhisperKVCache::allocate";
    if (decoder_layers <= 0) fail(where, "decoder_layers must be positive");
    if (d_model <= 0)        fail(where, "d_model must be positive");
    if (max_target_positions <= 0) fail(where, "max_target_positions must be positive");
    if (max_source_positions <= 0) fail(where, "max_source_positions must be positive");

    layers.clear();
    layers.resize(static_cast<std::size_t>(decoder_layers));
    for (auto& l : layers) {
        l.self_k = bt::Tensor::zeros_on(device, max_target_positions,
                                        d_model, bt::Dtype::FP32);
        l.self_v = bt::Tensor::zeros_on(device, max_target_positions,
                                        d_model, bt::Dtype::FP32);
        l.cross_k = bt::Tensor::zeros_on(device, max_source_positions,
                                         d_model, bt::Dtype::FP32);
        l.cross_v = bt::Tensor::zeros_on(device, max_source_positions,
                                         d_model, bt::Dtype::FP32);
        l.self_len = 0;
        l.cross_primed = false;
    }
}

void WhisperKVCache::reset() {
    for (auto& l : layers) {
        l.self_len = 0;
        l.cross_primed = false;
    }
}

// ─── mha_causal_cached_fp32 ────────────────────────────────────────────────

void mha_causal_cached_fp32(const bt::Tensor& X,
                            const bt::Tensor& Wq, const bt::Tensor& bq,
                            const bt::Tensor& Wk, const bt::Tensor& bk,
                            const bt::Tensor& Wv, const bt::Tensor& bv,
                            const bt::Tensor& Wo, const bt::Tensor& bo,
                            int num_heads,
                            WhisperLayerCache& cache,
                            bt::Tensor& out) {
    const std::string where = "mha_causal_cached_fp32";
    if (X.dtype != bt::Dtype::FP32) {
        fail(where, "currently FP32 only");
    }
    const int T = X.rows;
    const int D = X.cols;
    if (D <= 0 || T <= 0) fail(where, "X must be non-empty");
    if (D % num_heads != 0) {
        fail(where, "embed_dim " + std::to_string(D) +
             " not divisible by num_heads " + std::to_string(num_heads));
    }
    if (cache.self_k.rows == 0 || cache.self_k.cols != D ||
        cache.self_v.rows == 0 || cache.self_v.cols != D) {
        fail(where, "cache.self_k/self_v not pre-allocated to (>=T, D)");
    }
    if (cache.self_len + T > cache.self_k.rows) {
        fail(where, "cache overflow: self_len=" + std::to_string(cache.self_len) +
             " + T=" + std::to_string(T) + " > " +
             std::to_string(cache.self_k.rows));
    }
    if (cache.self_len > 0 && T > 1) {
        fail(where, "mixed prefill + step not supported "
             "(self_len=" + std::to_string(cache.self_len) +
             ", T=" + std::to_string(T) + ")");
    }

    bt::Tensor Q    = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    bt::Tensor Knew = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    bt::Tensor Vnew = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    bt::linear_forward_batched(Wq, bq, X, Q);     // (T, D)
    bt::linear_forward_batched(Wk, bk, X, Knew);  // (T, D)
    bt::linear_forward_batched(Wv, bv, X, Vnew);  // (T, D)

    bt::copy_d2d(Knew, 0, cache.self_k, cache.self_len * D, T * D);
    bt::copy_d2d(Vnew, 0, cache.self_v, cache.self_len * D, T * D);

    const int Lk_total = cache.self_len + T;

    bt::Tensor K_live = bt::Tensor::view(cache.self_k.device,
                                         cache.self_k.data,
                                         Lk_total, D, bt::Dtype::FP32);
    bt::Tensor V_live = bt::Tensor::view(cache.self_v.device,
                                         cache.self_v.data,
                                         Lk_total, D, bt::Dtype::FP32);

    const bool causal = (cache.self_len == 0);
    bt::Tensor ctx = bt::Tensor::empty_on(X.device, T, D, bt::Dtype::FP32);
    if (X.device == bt::Device::CPU) {
        bt::flash_attention_forward(Q, K_live, V_live,
                                    /*d_mask=*/nullptr, num_heads, causal, ctx);
    } else {
        bt::flash_attention_windowed_forward(Q, K_live, V_live,
                                             /*d_mask=*/nullptr, num_heads,
                                             /*window=*/0, ctx);
    }

    if (out.device != X.device || out.rows != T || out.cols != D ||
        out.dtype != bt::Dtype::FP32) {
        out = bt::Tensor::empty_on(X.device, T, D, bt::Dtype::FP32);
    }
    bt::linear_forward_batched(Wo, bo, ctx, out);

    cache.self_len += T;
}

// ─── cross_attn_cached_fp32 ────────────────────────────────────────────────

void cross_attn_cached_fp32(const bt::Tensor& X,
                            const bt::Tensor& Wq, const bt::Tensor& bq,
                            const bt::Tensor& Wo, const bt::Tensor& bo,
                            int num_heads,
                            const WhisperLayerCache& cache,
                            bt::Tensor& out) {
    const std::string where = "cross_attn_cached_fp32";
    if (X.dtype != bt::Dtype::FP32) {
        fail(where, "currently FP32 only");
    }
    if (!cache.cross_primed) {
        fail(where, "cross-attn cache not primed — call prime_cross first");
    }
    const int T  = X.rows;
    const int D  = X.cols;
    const int Lk = cache.cross_k.rows;
    if (D <= 0 || T <= 0) fail(where, "X must be non-empty");
    if (D % num_heads != 0) {
        fail(where, "embed_dim " + std::to_string(D) +
             " not divisible by num_heads " + std::to_string(num_heads));
    }
    if (cache.cross_k.cols != D || cache.cross_v.cols != D ||
        cache.cross_v.rows != Lk) {
        fail(where, "cross_k/cross_v shape mismatch");
    }

    bt::Tensor Q = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    bt::linear_forward_batched(Wq, bq, X, Q);  // (T, D)

    bt::Tensor ctx = bt::Tensor::empty_on(X.device, T, D, bt::Dtype::FP32);
    if (X.device == bt::Device::CPU) {
        bt::flash_attention_forward(Q, cache.cross_k, cache.cross_v,
                                    /*d_mask=*/nullptr, num_heads,
                                    /*causal=*/false, ctx);
    } else {
        if (T == 1) {
            bt::flash_attention_windowed_forward(Q, cache.cross_k, cache.cross_v,
                                                 /*d_mask=*/nullptr, num_heads,
                                                 /*window=*/0, ctx);
        } else {
            bt::Tensor ctx_row = bt::Tensor::empty_on(X.device, 1, D, bt::Dtype::FP32);
            for (int t = 0; t < T; ++t) {
                bt::Tensor q_row = bt::Tensor::view(
                    X.device,
                    static_cast<float*>(Q.data) + static_cast<std::size_t>(t) * D,
                    1, D, bt::Dtype::FP32);
                bt::flash_attention_windowed_forward(q_row, cache.cross_k,
                                                     cache.cross_v,
                                                     /*d_mask=*/nullptr, num_heads,
                                                     /*window=*/0, ctx_row);
                bt::copy_d2d(ctx_row, 0, ctx, t * D, D);
            }
        }
    }

    if (out.device != X.device || out.rows != T || out.cols != D ||
        out.dtype != bt::Dtype::FP32) {
        out = bt::Tensor::empty_on(X.device, T, D, bt::Dtype::FP32);
    }
    bt::linear_forward_batched(Wo, bo, ctx, out);
}

// ─── prime_cross_cache_fp32 ────────────────────────────────────────────────

void prime_cross_cache_fp32(const bt::Tensor& encoder_hidden,
                            const bt::Tensor& Wk, const bt::Tensor& bk,
                            const bt::Tensor& Wv, const bt::Tensor& bv,
                            WhisperLayerCache& cache) {
    const std::string where = "prime_cross_cache_fp32";
    if (encoder_hidden.dtype != bt::Dtype::FP32) {
        fail(where, "currently FP32 only");
    }
    const int L = encoder_hidden.rows;
    const int D = encoder_hidden.cols;
    if (cache.cross_k.rows != L || cache.cross_k.cols != D ||
        cache.cross_v.rows != L || cache.cross_v.cols != D) {
        fail(where, "cross_k/cross_v cache shape (" +
             std::to_string(cache.cross_k.rows) + "," +
             std::to_string(cache.cross_k.cols) + ") != encoder hidden (" +
             std::to_string(L) + "," +
             std::to_string(D) +
             ") — call WhisperKVCache::allocate with the matching dims first");
    }

    bt::Tensor Kproj = bt::Tensor::empty_on(encoder_hidden.device, 0, 0, bt::Dtype::FP32);
    bt::Tensor Vproj = bt::Tensor::empty_on(encoder_hidden.device, 0, 0, bt::Dtype::FP32);
    bt::linear_forward_batched(Wk, bk, encoder_hidden, Kproj);
    bt::linear_forward_batched(Wv, bv, encoder_hidden, Vproj);

    bt::copy_d2d(Kproj, 0, cache.cross_k, 0, L * D);
    bt::copy_d2d(Vproj, 0, cache.cross_v, 0, L * D);
    cache.cross_primed = true;
}

// ─── WhisperDecoderLayer ───────────────────────────────────────────────────

void WhisperDecoderLayer::load_from(const stf::File& f,
                                    const std::string& prefix,
                                    int dm, int ffn, int nh,
                                    bt::Device device) {
    const std::string where = "WhisperDecoderLayer::load_from";
    const bt::Device prev_dev = g_load_device;
    g_load_device = device;
    struct DevGuard { bt::Device& slot; bt::Device prev;
        ~DevGuard() { slot = prev; } } _guard{g_load_device, prev_dev};
    d_model   = dm;
    ffn_dim   = ffn;
    num_heads = nh;

    load_layernorm(f, prefix + "self_attn_layer_norm", dm, 1e-5f,
                   self_attn_layer_norm, where);

    upload(f, prefix + "self_attn.q_proj.weight",   dm, dm, self_Wq, where);
    upload(f, prefix + "self_attn.q_proj.bias",     dm, 1,  self_bq, where);
    upload(f, prefix + "self_attn.k_proj.weight",   dm, dm, self_Wk, where);
    self_bk = bt::Tensor::zeros_on(device, dm, 1, bt::Dtype::FP32);
    upload(f, prefix + "self_attn.v_proj.weight",   dm, dm, self_Wv, where);
    upload(f, prefix + "self_attn.v_proj.bias",     dm, 1,  self_bv, where);
    upload(f, prefix + "self_attn.out_proj.weight", dm, dm, self_Wo, where);
    upload(f, prefix + "self_attn.out_proj.bias",   dm, 1,  self_bo, where);

    load_layernorm(f, prefix + "encoder_attn_layer_norm", dm, 1e-5f,
                   encoder_attn_layer_norm, where);

    upload(f, prefix + "encoder_attn.q_proj.weight",   dm, dm, cross_Wq, where);
    upload(f, prefix + "encoder_attn.q_proj.bias",     dm, 1,  cross_bq, where);
    upload(f, prefix + "encoder_attn.k_proj.weight",   dm, dm, cross_Wk, where);
    cross_bk = bt::Tensor::zeros_on(device, dm, 1, bt::Dtype::FP32);
    upload(f, prefix + "encoder_attn.v_proj.weight",   dm, dm, cross_Wv, where);
    upload(f, prefix + "encoder_attn.v_proj.bias",     dm, 1,  cross_bv, where);
    upload(f, prefix + "encoder_attn.out_proj.weight", dm, dm, cross_Wo, where);
    upload(f, prefix + "encoder_attn.out_proj.bias",   dm, 1,  cross_bo, where);

    load_layernorm(f, prefix + "final_layer_norm", dm, 1e-5f,
                   final_layer_norm, where);
    load_linear(f, prefix + "fc1", ffn, dm, fc1, where);
    load_linear(f, prefix + "fc2", dm, ffn, fc2, where);
}

void WhisperDecoderLayer::prime_cross(const bt::Tensor& encoder_hidden,
                                      WhisperLayerCache& cache) const {
    prime_cross_cache_fp32(encoder_hidden,
                           cross_Wk, cross_bk, cross_Wv, cross_bv,
                           cache);
}

void WhisperDecoderLayer::forward(const bt::Tensor& X,
                                  WhisperLayerCache& cache,
                                  bt::Tensor& Y) const {
    const std::string where = "WhisperDecoderLayer::forward";
    if (X.dtype != bt::Dtype::FP32) {
        fail(where, "currently FP32 only");
    }
    if (X.cols != d_model) {
        fail(where, "X cols=" + std::to_string(X.cols) +
                    " != d_model=" + std::to_string(d_model));
    }

    bt::Tensor x_ln = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    self_attn_layer_norm.forward(X, x_ln);                  // (T, D)
    bt::Tensor self_out = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    mha_causal_cached_fp32(x_ln,
                           self_Wq, self_bq,
                           self_Wk, self_bk,
                           self_Wv, self_bv,
                           self_Wo, self_bo,
                           num_heads, cache, self_out);     // (T, D)
    bt::add_inplace(self_out, X);                            // residual

    bt::Tensor x_ca_ln = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    encoder_attn_layer_norm.forward(self_out, x_ca_ln);
    bt::Tensor cross_out = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    cross_attn_cached_fp32(x_ca_ln,
                           cross_Wq, cross_bq,
                           cross_Wo, cross_bo,
                           num_heads, cache, cross_out);    // (T, D)
    bt::add_inplace(cross_out, self_out);                    // residual

    bt::Tensor ffn_ln = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    final_layer_norm.forward(cross_out, ffn_ln);
    bt::Tensor h1 = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    fc1.forward_batched(ffn_ln, h1);
    bt::Tensor h1_act = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    bt::gelu_forward(h1, h1_act);
    bt::Tensor h2 = bt::Tensor::empty_on(X.device, 0, 0, bt::Dtype::FP32);
    fc2.forward_batched(h1_act, h2);
    bt::add_inplace(h2, cross_out);                          // residual

    Y = std::move(h2);
}

// ─── WhisperDecoder ────────────────────────────────────────────────────────

void WhisperDecoder::load_from(const stf::File& f,
                               int dm, int n_layer, int ffn, int n_head,
                               int vocab, int max_tgt, int max_src,
                               bt::Device device) {
    const std::string where = "WhisperDecoder::load_from";
    const bt::Device prev_dev = g_load_device;
    g_load_device = device;
    struct DevGuard { bt::Device& slot; bt::Device prev;
        ~DevGuard() { slot = prev; } } _guard{g_load_device, prev_dev};
    d_model                  = dm;
    decoder_layers           = n_layer;
    decoder_ffn_dim          = ffn;
    decoder_attention_heads  = n_head;
    vocab_size               = vocab;
    max_target_positions     = max_tgt;
    max_source_positions     = max_src;

    const std::string p = "model.decoder.";

    upload(f, p + "embed_tokens.weight",    vocab,   dm, embed_tokens,    where);
    upload(f, p + "embed_positions.weight", max_tgt, dm, embed_positions, where);

    layers.clear();
    layers.resize(static_cast<std::size_t>(n_layer));
    for (int i = 0; i < n_layer; ++i) {
        const std::string lp = p + "layers." + std::to_string(i) + ".";
        layers[static_cast<std::size_t>(i)].load_from(f, lp, dm, ffn, n_head, device);
    }

    load_layernorm(f, p + "layer_norm", dm, 1e-5f, layer_norm, where);

    proj_out_explicit = false;
    if (const stf::TensorView* v = f.find("proj_out.weight")) {
        (void)v;
        upload(f, "proj_out.weight", vocab, dm, proj_out_weight, where);
        proj_out_explicit = true;
    } else {
        proj_out_weight = bt::Tensor{};
    }

    const bt::Tensor& Wlm = proj_out_explicit ? proj_out_weight : embed_tokens;
    lm_head_T = bt::Tensor::empty_on(Wlm.device, 0, 0, bt::Dtype::FP32);
    bt::nchw_to_sequence(Wlm, /*N=*/1, /*C=*/vocab, /*H=*/1, /*W=*/dm, lm_head_T);
}

void WhisperDecoder::prime_cross(const bt::Tensor& encoder_hidden,
                                 WhisperKVCache& cache) const {
    const std::string where = "WhisperDecoder::prime_cross";
    if (static_cast<int>(cache.layers.size()) != decoder_layers) {
        fail(where, "cache layer count " +
             std::to_string(cache.layers.size()) + " != decoder_layers=" +
             std::to_string(decoder_layers));
    }
    if (encoder_hidden.rows != max_source_positions ||
        encoder_hidden.cols != d_model) {
        fail(where, "encoder_hidden shape (" +
             std::to_string(encoder_hidden.rows) + "," +
             std::to_string(encoder_hidden.cols) + ") != (" +
             std::to_string(max_source_positions) + "," +
             std::to_string(d_model) + ")");
    }
    for (int i = 0; i < decoder_layers; ++i) {
        layers[static_cast<std::size_t>(i)].prime_cross(
            encoder_hidden, cache.layers[static_cast<std::size_t>(i)]);
    }
}

void WhisperDecoder::forward(const std::int32_t* token_ids, int T,
                             int pos_offset,
                             WhisperKVCache& cache,
                             bt::Tensor& logits) const {
    const std::string where = "WhisperDecoder::forward";
    if (T <= 0)               fail(where, "T must be positive");
    if (pos_offset < 0)       fail(where, "pos_offset must be non-negative");
    if (pos_offset + T > max_target_positions) {
        fail(where, "pos_offset+T=" + std::to_string(pos_offset + T) +
             " exceeds max_target_positions=" + std::to_string(max_target_positions));
    }
    if (static_cast<int>(cache.layers.size()) != decoder_layers) {
        fail(where, "cache layer count " +
             std::to_string(cache.layers.size()) +
             " != decoder_layers=" + std::to_string(decoder_layers));
    }
    for (int i = 0; i < decoder_layers; ++i) {
        const auto& l = cache.layers[static_cast<std::size_t>(i)];
        if (!l.cross_primed) {
            fail(where, "cache layer " + std::to_string(i) +
                 " cross-attn not primed; call prime_cross first");
        }
        if (l.self_len != pos_offset) {
            fail(where, "cache layer " + std::to_string(i) +
                 " self_len=" + std::to_string(l.self_len) +
                 " != pos_offset=" + std::to_string(pos_offset));
        }
    }

    bt::Tensor x = bt::Tensor::empty_on(embed_tokens.device, 0, 0, bt::Dtype::FP32);
    if (embed_tokens.device == bt::Device::CPU) {
        bt::embedding_lookup_forward(embed_tokens, token_ids, T, x);  // (T, d_model)
    } else {
        bt::Tensor idx_dev = upload_int32_idx(embed_tokens.device, token_ids, T);
        bt::embedding_lookup_forward(embed_tokens,
                                     static_cast<const std::int32_t*>(idx_dev.data),
                                     T, x);
    }

    {
        float* pe_slab = static_cast<float*>(embed_positions.data)
                       + static_cast<std::size_t>(pos_offset) * d_model;
        bt::Tensor pe_view = bt::Tensor::view(embed_positions.device, pe_slab,
                                              T, d_model, bt::Dtype::FP32);
        bt::add_inplace(x, pe_view);
    }

    bt::Tensor y = bt::Tensor::empty_on(x.device, 0, 0, bt::Dtype::FP32);
    for (int i = 0; i < decoder_layers; ++i) {
        layers[static_cast<std::size_t>(i)].forward(
            x, cache.layers[static_cast<std::size_t>(i)], y);
        x = std::move(y);
        y = bt::Tensor::empty_on(x.device, 0, 0, bt::Dtype::FP32);
    }

    bt::Tensor hidden = bt::Tensor::empty_on(x.device, 0, 0, bt::Dtype::FP32);
    layer_norm.forward(x, hidden);          // (T, d_model)

    if (lm_head_T.rows != d_model || lm_head_T.cols != vocab_size) {
        fail(where, "lm_head_T not built — load_from() must run first");
    }
    if (logits.device != hidden.device || logits.rows != T ||
        logits.cols != vocab_size || logits.dtype != bt::Dtype::FP32) {
        logits = bt::Tensor::empty_on(hidden.device, T, vocab_size, bt::Dtype::FP32);
    }
    bt::matmul(hidden, lm_head_T, logits);  // (T, vocab_size)
}

// ─── Captured decode session ───────────────────────────────────────────────

struct WhisperStepScratch {
    bt::Tensor x_ln, q, k, v, ctx, attn_out;        // self-attention sublayer
    bt::Tensor ca_ln, cq, cctx, cross_out;          // cross-attention sublayer
    bt::Tensor ffn_ln, h1, h1_act, h2;              // FFN sublayer
};

struct WhisperDecoderStepState {
    int cap = 0;                 // mask/ramp rows == max_target_positions
    bt::Tensor mask;             // (cap, 1) FP32 — 1 valid / 0 invalid key rows
    bt::Tensor ones;             // (1, 1) FP32 constant 1.0 (d2d mask updates)
    bt::Tensor ramp;             // (cap, 1) INT32 0..cap-1 (d2d index updates)
    bt::Tensor idx;              // (1, 1) INT32 — the step's append row
    bt::Tensor in_tok;           // (1, d_model) staged token-embedding row
    bt::Tensor in_pos;           // (1, d_model) staged positional-embedding row
    bt::Tensor hs;               // (1, d_model) residual stream
    bt::Tensor hidden_n;         // (1, d_model) post-final-LN output
    bt::Tensor logits;           // (1, vocab_size) LM-head output
    WhisperStepScratch sc;
    bt::CudaGraph graph;
    bool captured = false;
    std::vector<const void*> keys;
};

WhisperDecoder::WhisperDecoder() = default;
WhisperDecoder::~WhisperDecoder() = default;
WhisperDecoder::WhisperDecoder(WhisperDecoder&&) noexcept = default;
WhisperDecoder& WhisperDecoder::operator=(WhisperDecoder&&) noexcept = default;

namespace {

std::vector<const void*> whisper_step_keys(const WhisperDecoder& d,
                                           const WhisperKVCache& cache) {
    std::vector<const void*> k;
    k.reserve(cache.layers.size() * 4 + 1);
    for (const auto& l : cache.layers) {
        k.push_back(l.self_k.data);
        k.push_back(l.self_v.data);
        k.push_back(l.cross_k.data);
        k.push_back(l.cross_v.data);
    }
    k.push_back(d.embed_tokens.data);
    return k;
}

void whisper_step_body(const WhisperDecoder& d, WhisperKVCache& cache,
                       WhisperDecoderStepState& st) {
    WhisperStepScratch& sc = st.sc;
    const int D = d.d_model;

    bt::copy_d2d(st.in_tok, 0, st.hs, 0, D);
    bt::add_inplace(st.hs, st.in_pos);

    for (int l = 0; l < d.decoder_layers; ++l) {
        const WhisperDecoderLayer& dl = d.layers[static_cast<std::size_t>(l)];
        WhisperLayerCache& lc = cache.layers[static_cast<std::size_t>(l)];

        dl.self_attn_layer_norm.forward(st.hs, sc.x_ln);
        bt::linear_forward_batched(dl.self_Wq, dl.self_bq, sc.x_ln, sc.q);
        bt::linear_forward_batched(dl.self_Wk, dl.self_bk, sc.x_ln, sc.k);
        bt::linear_forward_batched(dl.self_Wv, dl.self_bv, sc.x_ln, sc.v);
        bt::scatter_rows(sc.k, st.idx, lc.self_k);
        bt::scatter_rows(sc.v, st.idx, lc.self_v);
        qtd::flash_attn_masked(sc.q, lc.self_k, lc.self_v,
                               static_cast<const float*>(st.mask.data),
                               dl.num_heads, sc.ctx);
        bt::linear_forward_batched(dl.self_Wo, dl.self_bo, sc.ctx, sc.attn_out);
        bt::add_inplace(st.hs, sc.attn_out);

        dl.encoder_attn_layer_norm.forward(st.hs, sc.ca_ln);
        bt::linear_forward_batched(dl.cross_Wq, dl.cross_bq, sc.ca_ln, sc.cq);
        bt::flash_attention_windowed_forward(sc.cq, lc.cross_k, lc.cross_v,
                                             /*d_mask=*/nullptr, dl.num_heads,
                                             /*window=*/0, sc.cctx);
        bt::linear_forward_batched(dl.cross_Wo, dl.cross_bo, sc.cctx,
                                   sc.cross_out);
        bt::add_inplace(st.hs, sc.cross_out);

        dl.final_layer_norm.forward(st.hs, sc.ffn_ln);
        dl.fc1.forward_batched(sc.ffn_ln, sc.h1);
        bt::gelu_forward(sc.h1, sc.h1_act);
        dl.fc2.forward_batched(sc.h1_act, sc.h2);
        bt::add_inplace(st.hs, sc.h2);
    }

    d.layer_norm.forward(st.hs, st.hidden_n);
    bt::matmul(st.hidden_n, d.lm_head_T, st.logits);   // (1, vocab_size)
}

}  // namespace

bool WhisperDecoder::step_begin(WhisperKVCache& cache) const {
    if (const char* dis = std::getenv("BROSOUNDML_DISABLE_STEP_GRAPH")) {
        if (dis[0] != '\0' && std::string(dis) != "0") return false;
    }
    const bt::Device dev = layer_norm.gamma.device;
    if (!bt::graph_capture_available(dev)) return false;
    if (static_cast<int>(cache.layers.size()) != decoder_layers) return false;
    bt::DeviceScope scope(dev);

    if (!step_state) step_state = std::make_unique<WhisperDecoderStepState>();
    WhisperDecoderStepState& st = *step_state;

    const int cap = max_target_positions;
    if (st.cap != cap) {
        st.mask = bt::Tensor::zeros_on(dev, cap, 1, bt::Dtype::FP32);
        std::vector<std::int32_t> hramp(static_cast<std::size_t>(cap));
        for (int i = 0; i < cap; ++i) hramp[static_cast<std::size_t>(i)] = i;
        st.ramp = upload_int32_idx(dev, hramp.data(), cap);
        const float one = 1.0f;
        st.ones = bt::Tensor::from_host_on(dev, &one, 1, 1);
        st.idx  = bt::Tensor::empty_on(dev, 1, 1, bt::Dtype::INT32);
        st.in_tok   = bt::Tensor::empty_on(dev, 1, d_model, bt::Dtype::FP32);
        st.in_pos   = bt::Tensor::empty_on(dev, 1, d_model, bt::Dtype::FP32);
        st.hs       = bt::Tensor::empty_on(dev, 1, d_model, bt::Dtype::FP32);
        st.hidden_n = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        st.logits   = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        for (bt::Tensor* t : {&st.sc.x_ln, &st.sc.q, &st.sc.k, &st.sc.v,
                              &st.sc.ctx, &st.sc.attn_out, &st.sc.ca_ln,
                              &st.sc.cq, &st.sc.cctx, &st.sc.cross_out,
                              &st.sc.ffn_ln, &st.sc.h1, &st.sc.h1_act,
                              &st.sc.h2}) {
            *t = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        }
        st.cap = cap;
        st.graph.reset();
        st.captured = false;
    } else {
        std::vector<float> zeros(static_cast<std::size_t>(cap), 0.0f);
        bt::detail::alloc_for(dev).memcpy_h2d(
            st.mask.data, zeros.data(),
            static_cast<std::size_t>(cap) * sizeof(float), dev.index);
    }

    std::vector<const void*> keys = whisper_step_keys(*this, cache);
    if (keys != st.keys) {
        st.keys = std::move(keys);
        st.graph.reset();
        st.captured = false;
    }
    return true;
}

void WhisperDecoder::step_mask_prefill(int T) const {
    const std::string where = "WhisperDecoder::step_mask_prefill";
    if (!step_state || step_state->cap == 0) {
        fail(where, "no decode session — call step_begin first");
    }
    WhisperDecoderStepState& st = *step_state;
    if (T < 0 || T > st.cap) {
        fail(where, "prefill length " + std::to_string(T) +
                    " out of range [0, " + std::to_string(st.cap) + "]");
    }
    const bt::Device dev = layer_norm.gamma.device;
    bt::DeviceScope scope(dev);

    std::vector<float> hmask(static_cast<std::size_t>(st.cap), 0.0f);
    std::fill(hmask.begin(), hmask.begin() + T, 1.0f);
    bt::detail::alloc_for(dev).memcpy_h2d(
        st.mask.data, hmask.data(),
        static_cast<std::size_t>(st.cap) * sizeof(float), dev.index);
}

void WhisperDecoder::step_decode(std::int32_t token_id, int pos,
                                 WhisperKVCache& cache,
                                 bt::Tensor& logits_view) const {
    const std::string where = "WhisperDecoder::step_decode";
    if (!step_state || step_state->cap == 0) {
        fail(where, "no decode session — call step_begin first");
    }
    WhisperDecoderStepState& st = *step_state;
    const bt::Device dev = layer_norm.gamma.device;
    bt::DeviceScope scope(dev);

    if (token_id < 0 || token_id >= vocab_size) {
        fail(where, "token_id " + std::to_string(token_id) +
                    " out of vocab range");
    }
    if (pos < 0 || pos >= max_target_positions) {
        fail(where, "pos " + std::to_string(pos) +
                    " exceeds max_target_positions=" +
                    std::to_string(max_target_positions));
    }
    if (static_cast<int>(cache.layers.size()) != decoder_layers) {
        fail(where, "cache layer count " + std::to_string(cache.layers.size()) +
                    " != decoder_layers=" + std::to_string(decoder_layers));
    }
    for (int i = 0; i < decoder_layers; ++i) {
        const auto& l = cache.layers[static_cast<std::size_t>(i)];
        if (!l.cross_primed) {
            fail(where, "cache layer " + std::to_string(i) +
                        " cross-attn not primed; call prime_cross first");
        }
        if (l.self_len != pos) {
            fail(where, "cache layer " + std::to_string(i) +
                        " self_len=" + std::to_string(l.self_len) +
                        " != pos=" + std::to_string(pos));
        }
    }

    std::vector<const void*> keys = whisper_step_keys(*this, cache);
    if (keys != st.keys) {
        st.keys = std::move(keys);
        st.graph.reset();
        st.captured = false;
    }

    bt::copy_d2d(st.ramp, pos, st.idx, 0, 1);              // append row
    bt::copy_d2d(st.ones, 0, st.mask, pos, 1);             // mask[pos] = 1
    bt::copy_d2d(embed_tokens, token_id * d_model, st.in_tok, 0, d_model);
    bt::copy_d2d(embed_positions, pos * d_model, st.in_pos, 0, d_model);

    if (!st.captured) {
        whisper_step_body(*this, cache, st);
        bt::sync_all();
        {
            bt::CudaGraphCapture capture;
            whisper_step_body(*this, cache, st);
            st.graph = capture.finish();
        }
        st.captured = true;
    } else {
        st.graph.launch();
    }

    for (auto& l : cache.layers) l.self_len = pos + 1;
    logits_view = bt::Tensor::view(dev, st.logits.data, 1, vocab_size,
                                   bt::Dtype::FP32);
}

}  // namespace brosoundml
