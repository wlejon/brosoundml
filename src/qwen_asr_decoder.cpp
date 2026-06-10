#include "qwen_asr_decoder.h"

#include "qwen_tts_device.h"   // qtd:: device-neutral helpers (model-agnostic)

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: QwenAsrDecoder: " + msg);
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Upload a weight to FP32 on `dev` (BF16 on disk; widened host-side first —
// same pattern as the Qwen3-TTS Talker).
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols,
              bt::Device dev) {
    bt::Tensor t;
    {
        bt::DeviceScope cpu(bt::Device::CPU);
        sf::upload_compute_checked(need(f, name), rows, cols, t, name);
    }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
bt::Tensor up_vec(const sf::File& f, const std::string& name, int n,
                  bt::Device dev) {
    return up(f, name, n, 1, dev);
}

}  // namespace

void QwenAsrDecoder::load(const sf::File& f, const QwenAsrConfig& cfg,
                          bt::Device dev) {
    num_layers   = cfg.num_hidden_layers;
    hidden       = cfg.hidden_size;
    intermediate = cfg.intermediate_size;
    n_q_heads    = cfg.num_attention_heads;
    n_kv_heads   = cfg.num_key_value_heads;
    head_dim     = cfg.head_dim;
    vocab        = cfg.vocab_size;
    rms_eps      = cfg.rms_norm_eps;
    rope_theta   = cfg.rope_theta;

    if (num_layers <= 0 || hidden <= 0 || head_dim <= 0 || vocab <= 0)
        fail("config not parsed (zero dims)");

    embed_tokens = up(f, "thinker.model.embed_tokens.weight", vocab, hidden, dev);
    lm_head      = up(f, "thinker.lm_head.weight", vocab, hidden, dev);
    final_norm   = up_vec(f, "thinker.model.norm.weight", hidden, dev);

    // RoPE convention bridge: permute q/k projection rows (and q/k_norm) from
    // HF rotate-half pairs into brotensor's adjacent-pair layout once at load
    // (see qwen_tts_talker.cpp for the full rationale).
    const std::vector<std::int32_t> hd_perm = qtd::rotate_half_perm(head_dim);
    const std::vector<std::int32_t> q_perm =
        qtd::per_head_perm_rows(hd_perm, n_q_heads, head_dim);
    const std::vector<std::int32_t> k_perm =
        qtd::per_head_perm_rows(hd_perm, n_kv_heads, head_dim);

    const int qd = n_q_heads * head_dim;
    const int kd = n_kv_heads * head_dim;
    layers.clear();
    layers.resize(static_cast<std::size_t>(num_layers));
    for (int i = 0; i < num_layers; ++i) {
        const std::string L =
            "thinker.model.layers." + std::to_string(i) + ".";
        QwenAsrDecoderLayer& dl = layers[static_cast<std::size_t>(i)];
        dl.in_ln   = up_vec(f, L + "input_layernorm.weight", hidden, dev);
        dl.post_ln = up_vec(f, L + "post_attention_layernorm.weight", hidden, dev);
        dl.qw      = qtd::gather_rows(up(f, L + "self_attn.q_proj.weight", qd, hidden, dev), q_perm);
        dl.kw      = qtd::gather_rows(up(f, L + "self_attn.k_proj.weight", kd, hidden, dev), k_perm);
        dl.vw      = up(f, L + "self_attn.v_proj.weight", kd, hidden, dev);
        dl.ow      = up(f, L + "self_attn.o_proj.weight", hidden, qd, dev);
        dl.q_norm  = qtd::gather_rows(up_vec(f, L + "self_attn.q_norm.weight", head_dim, dev), hd_perm);
        dl.k_norm  = qtd::gather_rows(up_vec(f, L + "self_attn.k_norm.weight", head_dim, dev), hd_perm);
        dl.gate    = up(f, L + "mlp.gate_proj.weight", intermediate, hidden, dev);
        dl.up      = up(f, L + "mlp.up_proj.weight", intermediate, hidden, dev);
        dl.down    = up(f, L + "mlp.down_proj.weight", hidden, intermediate, dev);
    }
}

void QwenAsrDecoder::run_dev(const bt::Tensor& embeds, int n,
                             QwenAsrDecoderCache* cache,
                             bt::Tensor& hidden_out) const {
    const bt::Device dev = final_norm.device;
    bt::DeviceScope scope(dev);
    if (n <= 0) {
        hidden_out = bt::Tensor::zeros_on(dev, 0, hidden, bt::Dtype::FP32);
        return;
    }
    if (!cache) fail("run_dev requires a cache");

    const int kd     = n_kv_heads * head_dim;
    const int half   = head_dim / 2;
    const int offset = cache->len;

    // ── 1D RoPE tables for the new tokens (positions offset..offset+n-1) ──
    std::vector<int>   pos_grid(static_cast<std::size_t>(n) * half);
    std::vector<float> inv_freq(static_cast<std::size_t>(half));
    for (int i = 0; i < half; ++i) {
        inv_freq[static_cast<std::size_t>(i)] =
            std::pow(rope_theta, -(2.0f * i) / head_dim);
        for (int t = 0; t < n; ++t)
            pos_grid[static_cast<std::size_t>(t) * half + i] = offset + t;
    }
    bt::Tensor cosT, sinT;
    qtd::build_rope_tables(dev, n, half, pos_grid, inv_freq, cosT, sinT);

    bt::Tensor hs = embeds;   // mutated in place by the residual adds

    // ── ensure the KV cache has room for the new tokens ──
    {
        const int need_rows = offset + n;
        if (cache->cap < need_rows) {
            const int newcap =
                std::max(need_rows, std::max(cache->cap * 2, 64));
            for (int l = 0; l < num_layers; ++l) {
                bt::Tensor nk = bt::Tensor::zeros_on(dev, newcap, kd, bt::Dtype::FP32);
                bt::Tensor nv = bt::Tensor::zeros_on(dev, newcap, kd, bt::Dtype::FP32);
                if (offset > 0) {
                    bt::copy_d2d(cache->k[static_cast<std::size_t>(l)], 0, nk, 0, offset * kd);
                    bt::copy_d2d(cache->v[static_cast<std::size_t>(l)], 0, nv, 0, offset * kd);
                }
                cache->k[static_cast<std::size_t>(l)] = std::move(nk);
                cache->v[static_cast<std::size_t>(l)] = std::move(nv);
            }
            cache->cap = newcap;
        }
    }

    for (int l = 0; l < num_layers; ++l) {
        const QwenAsrDecoderLayer& dl = layers[static_cast<std::size_t>(l)];

        // ── self-attention ──
        bt::Tensor normed;
        bt::rms_norm_forward(hs, dl.in_ln, rms_eps, normed);
        bt::Tensor q, k, v;
        qtd::linear(dl.qw, nullptr, normed, q);   // (n, qd)
        qtd::linear(dl.kw, nullptr, normed, k);   // (n, kd)
        qtd::linear(dl.vw, nullptr, normed, v);   // (n, kd)

        bt::Tensor qn, kn;
        qtd::head_rms_norm(q, n, n_q_heads,  head_dim, dl.q_norm, rms_eps, qn);
        qtd::head_rms_norm(k, n, n_kv_heads, head_dim, dl.k_norm, rms_eps, kn);
        bt::Tensor qr, kr;
        bt::rope_apply(qn, cosT, sinT, head_dim, n_q_heads,  qr);
        bt::rope_apply(kn, cosT, sinT, head_dim, n_kv_heads, kr);

        // The cache holds the n_kv (grouped) heads; the windowed attention op
        // expands them to the n_q query heads internally (GQA).
        bt::copy_d2d(kr, 0, cache->k[static_cast<std::size_t>(l)], offset * kd, n * kd);
        bt::copy_d2d(v,  0, cache->v[static_cast<std::size_t>(l)], offset * kd, n * kd);
        const int valid = offset + n;
        bt::Tensor Kf = bt::Tensor::view(dev, cache->k[static_cast<std::size_t>(l)].data,
                                         valid, kd, bt::Dtype::FP32);
        bt::Tensor Vf = bt::Tensor::view(dev, cache->v[static_cast<std::size_t>(l)].data,
                                         valid, kd, bt::Dtype::FP32);
        bt::Tensor ctx;
        qtd::flash_attn(qr, Kf, Vf, n_q_heads, head_dim,
                        /*causal=*/offset == 0, ctx);
        bt::Tensor attn;
        qtd::linear(dl.ow, nullptr, ctx, attn);   // (n, hidden)
        bt::add_inplace_batched(hs, attn);

        // ── SwiGLU MLP ──
        bt::Tensor n2;
        bt::rms_norm_forward(hs, dl.post_ln, rms_eps, n2);
        bt::Tensor g, u;
        qtd::linear(dl.gate, nullptr, n2, g);     // (n, intermediate)
        qtd::linear(dl.up,   nullptr, n2, u);
        qtd::swiglu(g, u);
        bt::Tensor dn;
        qtd::linear(dl.down, nullptr, g, dn);      // (n, hidden)
        bt::add_inplace_batched(hs, dn);
    }

    cache->len = offset + n;
    bt::rms_norm_forward(hs, final_norm, rms_eps, hidden_out);   // (n, hidden)
}

void QwenAsrDecoder::logits_last(const bt::Tensor& hidden_rows,
                                 bt::Tensor& logits_out) const {
    const bt::Device dev = lm_head.device;
    bt::DeviceScope scope(dev);
    if (hidden_rows.rows < 1) fail("logits_last on an empty hidden tensor");
    float* last = static_cast<float*>(hidden_rows.data) +
                  static_cast<std::size_t>(hidden_rows.rows - 1) * hidden;
    bt::Tensor row = bt::Tensor::view(dev, last, 1, hidden, bt::Dtype::FP32);
    qtd::linear(lm_head, nullptr, row, logits_out);   // (1, vocab)
}

}  // namespace brosoundml
