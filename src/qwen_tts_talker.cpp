#include "qwen_tts_talker.h"

#include "qwen_tts_device.h"

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
    throw std::runtime_error("brosoundml: QwenTtsTalker: " + msg);
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Upload a weight to FP32 on `dev`. The talker checkpoint is BF16 on disk;
// upload_compute_checked under a CPU scope widens it to host FP32, then it is
// migrated to the target device. FP32 on every backend keeps the CUDA path
// numerically aligned with CPU (brotensor's matmul / rms_norm / rope_apply /
// silu all have FP32 CUDA kernels).
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

void QwenTtsTalker::load(const sf::File& f, const QwenTtsTalkerConfig& cfg,
                         bt::Device dev) {
    num_layers   = cfg.transformer.num_hidden_layers;
    hidden       = cfg.transformer.hidden_size;
    intermediate = cfg.transformer.intermediate_size;
    n_q_heads    = cfg.transformer.num_attention_heads;
    n_kv_heads   = cfg.transformer.num_key_value_heads;
    head_dim     = cfg.transformer.head_dim;
    vocab        = cfg.vocab_size;
    text_vocab   = cfg.text_vocab_size;
    text_hidden  = cfg.text_hidden_size;
    rms_eps      = cfg.transformer.rms_norm_eps;
    rope_theta   = cfg.transformer.rope_theta;
    mrope_section     = cfg.mrope_section;
    mrope_interleaved = cfg.mrope_interleaved;

    const std::string P = "talker.";
    codec_embedding = up(f, P + "model.codec_embedding.weight", vocab, hidden, dev);
    text_embedding  = up(f, P + "model.text_embedding.weight", text_vocab, text_hidden, dev);
    final_norm      = up_vec(f, P + "model.norm.weight", hidden, dev);
    tp_fc1_w = up(f, P + "text_projection.linear_fc1.weight", text_hidden, text_hidden, dev);
    tp_fc1_b = up_vec(f, P + "text_projection.linear_fc1.bias", text_hidden, dev);
    tp_fc2_w = up(f, P + "text_projection.linear_fc2.weight", hidden, text_hidden, dev);
    tp_fc2_b = up_vec(f, P + "text_projection.linear_fc2.bias", hidden, dev);
    codec_head = up(f, P + "codec_head.weight", vocab, hidden, dev);

    // RoPE convention bridge: brotensor's rope_apply rotates adjacent pairs
    // (2i, 2i+1); HF Qwen rotates split-half pairs (i, i+half). Permute each
    // q/k projection's output rows (and q/k_norm) into the adjacent-pair layout
    // once at load, so the runtime rope_apply reproduces HF rotate-half while
    // attention scores (a permutation-invariant dot product) stay unchanged.
    const std::vector<std::int32_t> hd_perm = qtd::rotate_half_perm(head_dim);
    const std::vector<std::int32_t> q_perm =
        qtd::per_head_perm_rows(hd_perm, n_q_heads, head_dim);
    const std::vector<std::int32_t> k_perm =
        qtd::per_head_perm_rows(hd_perm, n_kv_heads, head_dim);

    const int qd = n_q_heads * head_dim;
    const int kd = n_kv_heads * head_dim;
    layers.clear();
    layers.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const std::string L = P + "model.layers." + std::to_string(i) + ".";
        QwenTtsTalkerLayer& tl = layers[i];
        tl.in_ln   = up_vec(f, L + "input_layernorm.weight", hidden, dev);
        tl.post_ln = up_vec(f, L + "post_attention_layernorm.weight", hidden, dev);
        tl.qw      = qtd::gather_rows(up(f, L + "self_attn.q_proj.weight", qd, hidden, dev), q_perm);
        tl.kw      = qtd::gather_rows(up(f, L + "self_attn.k_proj.weight", kd, hidden, dev), k_perm);
        tl.vw      = up(f, L + "self_attn.v_proj.weight", kd, hidden, dev);
        tl.ow      = up(f, L + "self_attn.o_proj.weight", hidden, qd, dev);
        tl.q_norm  = qtd::gather_rows(up_vec(f, L + "self_attn.q_norm.weight", head_dim, dev), hd_perm);
        tl.k_norm  = qtd::gather_rows(up_vec(f, L + "self_attn.k_norm.weight", head_dim, dev), hd_perm);
        tl.gate    = up(f, L + "mlp.gate_proj.weight", intermediate, hidden, dev);
        tl.up      = up(f, L + "mlp.up_proj.weight", intermediate, hidden, dev);
        tl.down    = up(f, L + "mlp.down_proj.weight", hidden, intermediate, dev);
    }
}

void QwenTtsTalker::run(const float* embeds, int n, const int32_t* pos3n,
                        QwenTtsTalkerCache* cache, bt::Tensor& hidden_out) const {
    const bt::Device dev = final_norm.device;
    bt::DeviceScope scope(dev);
    if (n <= 0) {
        hidden_out = bt::Tensor::zeros_on(dev, 0, hidden, bt::Dtype::FP32);
        return;
    }
    bt::Tensor e = bt::Tensor::from_host_on(dev, embeds, n, hidden);
    run_dev(e, n, pos3n, cache, hidden_out);
}

void QwenTtsTalker::run_dev(const bt::Tensor& embeds, int n, const int32_t* pos3n,
                            QwenTtsTalkerCache* cache, bt::Tensor& hidden_out) const {
    const bt::Device dev = final_norm.device;
    bt::DeviceScope scope(dev);
    if (n <= 0) {
        hidden_out = bt::Tensor::zeros_on(dev, 0, hidden, bt::Dtype::FP32);
        return;
    }

    const int qd     = n_q_heads * head_dim;   // == expanded K/V width
    const int half   = head_dim / 2;
    const int offset = cache ? cache->len : 0;

    // ── interleaved M-RoPE tables for the new tokens ──
    //   pair i (after the load-time rotate-half permutation) takes its position
    //   from axis axis_of[i] and frequency inv_freq[i]. axis a in {1..} claims
    //   pairs with i%mn==a and i<section[a]*mn; the rest stay on axis 0.
    const int mn = static_cast<int>(mrope_section.size());
    std::vector<int>   pos_grid(static_cast<std::size_t>(n) * half);
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; ++i) {
        int a = 0;
        for (int m = 1; m < mn; ++m)
            if (i % mn == m && i < mrope_section[m] * mn) { a = m; break; }
        inv_freq[i] = std::pow(rope_theta, -(2.0f * i) / head_dim);
        for (int t = 0; t < n; ++t)
            pos_grid[static_cast<std::size_t>(t) * half + i] = pos3n[a * n + t];
    }
    bt::Tensor cosT, sinT;
    qtd::build_rope_tables(dev, n, half, pos_grid, inv_freq, cosT, sinT);

    // ── input embeddings (owning copy; mutated in place by residual adds) ──
    bt::Tensor hs = embeds;

    // ── ensure the KV cache has room for the new tokens ──
    if (cache) {
        const int need = offset + n;
        if (cache->cap < need) {
            const int newcap = std::max(need, std::max(cache->cap * 2, 64));
            for (int l = 0; l < num_layers; ++l) {
                bt::Tensor nk = bt::Tensor::zeros_on(dev, newcap, qd, bt::Dtype::FP32);
                bt::Tensor nv = bt::Tensor::zeros_on(dev, newcap, qd, bt::Dtype::FP32);
                if (offset > 0) {
                    bt::copy_d2d(cache->k[l], 0, nk, 0, offset * qd);
                    bt::copy_d2d(cache->v[l], 0, nv, 0, offset * qd);
                }
                cache->k[l] = std::move(nk);
                cache->v[l] = std::move(nv);
            }
            cache->cap = newcap;
        }
    }

    for (int l = 0; l < num_layers; ++l) {
        const QwenTtsTalkerLayer& tl = layers[l];

        // ── self-attention ──
        bt::Tensor normed;
        bt::rms_norm_forward(hs, tl.in_ln, rms_eps, normed);
        bt::Tensor q, k, v;
        qtd::linear(tl.qw, nullptr, normed, q);   // (n, qd)
        qtd::linear(tl.kw, nullptr, normed, k);   // (n, kd)
        qtd::linear(tl.vw, nullptr, normed, v);   // (n, kd)

        bt::Tensor qn, kn;
        qtd::head_rms_norm(q, n, n_q_heads,  head_dim, tl.q_norm, rms_eps, qn);
        qtd::head_rms_norm(k, n, n_kv_heads, head_dim, tl.k_norm, rms_eps, kn);
        bt::Tensor qr, kr;
        bt::rope_apply(qn, cosT, sinT, head_dim, n_q_heads,  qr);
        bt::rope_apply(kn, cosT, sinT, head_dim, n_kv_heads, kr);

        bt::Tensor kE = qtd::expand_kv(kr, n, n_kv_heads, n_q_heads, head_dim);  // (n, qd)
        bt::Tensor vE = qtd::expand_kv(v,  n, n_kv_heads, n_q_heads, head_dim);  // (n, qd)

        bt::Tensor ctx;
        if (cache) {
            bt::copy_d2d(kE, 0, cache->k[l], offset * qd, n * qd);
            bt::copy_d2d(vE, 0, cache->v[l], offset * qd, n * qd);
            const int valid = offset + n;
            bt::Tensor Kf = bt::Tensor::view(dev, cache->k[l].data, valid, qd, bt::Dtype::FP32);
            bt::Tensor Vf = bt::Tensor::view(dev, cache->v[l].data, valid, qd, bt::Dtype::FP32);
            // offset==0: a fresh prefill (Lq==Lk) is causal; a later step has a
            // single query at the latest position attending the whole cache.
            qtd::flash_attn(qr, Kf, Vf, n_q_heads, /*causal=*/offset == 0, ctx);
        } else {
            qtd::flash_attn(qr, kE, vE, n_q_heads, /*causal=*/true, ctx);
        }
        bt::Tensor attn;
        qtd::linear(tl.ow, nullptr, ctx, attn);   // (n, hidden)
        bt::add_inplace_batched(hs, attn);

        // ── SwiGLU MLP ──
        bt::Tensor n2;
        bt::rms_norm_forward(hs, tl.post_ln, rms_eps, n2);
        bt::Tensor g, u;
        qtd::linear(tl.gate, nullptr, n2, g);     // (n, intermediate)
        qtd::linear(tl.up,   nullptr, n2, u);
        qtd::swiglu(g, u);
        bt::Tensor dn;
        qtd::linear(tl.down, nullptr, g, dn);      // (n, hidden)
        bt::add_inplace_batched(hs, dn);
    }

    if (cache) cache->len = offset + n;
    bt::rms_norm_forward(hs, final_norm, rms_eps, hidden_out);   // (n, hidden)
}

void QwenTtsTalker::codec_logits(const float* hidden_rows, int n,
                                 bt::Tensor& logits_out) const {
    const bt::Device dev = codec_head.device;
    bt::DeviceScope scope(dev);
    if (n <= 0) {
        logits_out = bt::Tensor::zeros_on(dev, 0, vocab, bt::Dtype::FP32);
        return;
    }
    bt::Tensor h = bt::Tensor::from_host_on(dev, hidden_rows, n, hidden);
    qtd::linear(codec_head, nullptr, h, logits_out);   // (n, vocab)
}

void QwenTtsTalker::forward(const float* inputs_embeds, int T,
                            const int32_t* pos3T, bt::Tensor& hidden_out,
                            bt::Tensor& logits_out) const {
    run(inputs_embeds, T, pos3T, nullptr, hidden_out);   // device tensor
    bt::DeviceScope scope(codec_head.device);
    qtd::linear(codec_head, nullptr, hidden_out, logits_out);
}

void QwenTtsTalker::codec_embed(int id, float* out) const {
    bt::DeviceScope scope(codec_embedding.device);
    bt::Tensor row = qtd::gather_rows(codec_embedding,
                                      {static_cast<std::int32_t>(id)});  // (1, hidden)
    qtd::to_host(row, out);
}

const float* QwenTtsTalker::codec_embedding_row(int id) const {
    // Raw host pointer into the embedding table — valid only when the table is
    // CPU-resident (the host-side AR assembly path). The device AR loop gathers
    // rows on-device instead.
    return codec_embedding.host_f32() + static_cast<std::size_t>(id) * hidden;
}

void QwenTtsTalker::text_embed_proj(int id, float* out) const {
    bt::DeviceScope scope(text_embedding.device);
    bt::Tensor te = qtd::gather_rows(text_embedding,
                                     {static_cast<std::int32_t>(id)});  // (1, text_hidden)
    bt::Tensor h1;
    qtd::linear(tp_fc1_w, &tp_fc1_b, te, h1);   // (1, text_hidden)
    bt::silu_forward(h1, h1);
    bt::Tensor h2;
    qtd::linear(tp_fc2_w, &tp_fc2_b, h1, h2);   // (1, hidden)
    qtd::to_host(h2, out);
}

}  // namespace brosoundml
