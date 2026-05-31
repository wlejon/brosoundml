#include "qwen_tts_talker.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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

// Upload at compute dtype — FP32 on the host (the talker checkpoint is BF16 on
// disk; upload_compute_checked widens it to FP32 under the CPU DeviceScope).
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols) {
    bt::Tensor t;
    sf::upload_compute_checked(need(f, name), rows, cols, t, name);
    return t;
}
bt::Tensor up_vec(const sf::File& f, const std::string& name, int n) {
    return up(f, name, n, 1);
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Y(B,out) = X(B,in) @ W^T (+ bias). A null bias adds zero.
void linear(const bt::Tensor& W, const bt::Tensor* bias, const bt::Tensor& X,
            bt::Tensor& Y) {
    if (bias) {
        bt::linear_forward_batched(W, *bias, X, Y);
    } else {
        bt::Tensor zero = bt::Tensor::mat(W.rows, 1);
        bt::linear_forward_batched(W, zero, X, Y);
    }
}

// Per-head RMSNorm over head_dim, in place: rows are (T*num_heads) vectors of
// length head_dim, each normalized then scaled by gamma (head_dim).
void head_rmsnorm(float* buf, int rows, int head_dim, const float* gamma,
                  float eps) {
    for (int r = 0; r < rows; ++r) {
        float* v = buf + static_cast<std::size_t>(r) * head_dim;
        float ss = 0.0f;
        for (int d = 0; d < head_dim; ++d) ss += v[d] * v[d];
        const float inv = 1.0f / std::sqrt(ss / head_dim + eps);
        for (int d = 0; d < head_dim; ++d) v[d] = v[d] * inv * gamma[d];
    }
}

}  // namespace

void QwenTtsTalker::load(const sf::File& f, const QwenTtsTalkerConfig& cfg) {
    bt::DeviceScope cpu(bt::Device::CPU);  // host FP32 path (BF16 widened on load)

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
    codec_embedding = up(f, P + "model.codec_embedding.weight", vocab, hidden);
    text_embedding  = up(f, P + "model.text_embedding.weight", text_vocab, text_hidden);
    final_norm      = up_vec(f, P + "model.norm.weight", hidden);
    tp_fc1_w = up(f, P + "text_projection.linear_fc1.weight", text_hidden, text_hidden);
    tp_fc1_b = up_vec(f, P + "text_projection.linear_fc1.bias", text_hidden);
    tp_fc2_w = up(f, P + "text_projection.linear_fc2.weight", hidden, text_hidden);
    tp_fc2_b = up_vec(f, P + "text_projection.linear_fc2.bias", hidden);
    codec_head = up(f, P + "codec_head.weight", vocab, hidden);

    const int qd = n_q_heads * head_dim;
    const int kd = n_kv_heads * head_dim;
    layers.clear();
    layers.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const std::string L = P + "model.layers." + std::to_string(i) + ".";
        QwenTtsTalkerLayer& tl = layers[i];
        tl.in_ln   = up_vec(f, L + "input_layernorm.weight", hidden);
        tl.post_ln = up_vec(f, L + "post_attention_layernorm.weight", hidden);
        tl.qw      = up(f, L + "self_attn.q_proj.weight", qd, hidden);
        tl.kw      = up(f, L + "self_attn.k_proj.weight", kd, hidden);
        tl.vw      = up(f, L + "self_attn.v_proj.weight", kd, hidden);
        tl.ow      = up(f, L + "self_attn.o_proj.weight", hidden, qd);
        tl.q_norm  = up_vec(f, L + "self_attn.q_norm.weight", head_dim);
        tl.k_norm  = up_vec(f, L + "self_attn.k_norm.weight", head_dim);
        tl.gate    = up(f, L + "mlp.gate_proj.weight", intermediate, hidden);
        tl.up      = up(f, L + "mlp.up_proj.weight", intermediate, hidden);
        tl.down    = up(f, L + "mlp.down_proj.weight", hidden, intermediate);
    }
}

void QwenTtsTalker::run(const float* embeds, int n, const int32_t* pos3n,
                        QwenTtsTalkerCache* cache, bt::Tensor& hidden_out) const {
    bt::DeviceScope cpu(bt::Device::CPU);
    if (n <= 0) { hidden_out.resize(0, hidden); return; }

    // ── interleaved M-RoPE: per rotary pair i, which position axis feeds it,
    //    and the pair frequency. Section [s0,s1,s2]: axis a in {1,2} claims
    //    pairs i with i%3==a and i < section[a]*3; the rest stay on axis 0.
    const int half = head_dim / 2;
    const int mn = static_cast<int>(mrope_section.size());
    std::vector<int>   axis_of(half, 0);
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; ++i) {
        int a = 0;
        for (int m = 1; m < mn; ++m) {
            if (i % mn == m && i < mrope_section[m] * mn) { a = m; break; }
        }
        axis_of[i] = a;
        inv_freq[i] = std::pow(rope_theta, -(2.0f * i) / head_dim);
    }

    const int qd = n_q_heads * head_dim;
    const int kd = n_kv_heads * head_dim;
    const int group = n_q_heads / n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Global position of new token i, and the keys it may attend (causal):
    // [0, offset+i]. With no cache offset==0 and total==n.
    const int offset = cache ? cache->len : 0;
    const int total  = offset + n;

    bt::Tensor hs = bt::Tensor::mat(n, hidden);
    std::copy(embeds, embeds + static_cast<std::size_t>(n) * hidden,
              hs.host_f32_mut());

    // RoPE the new tokens in place (positions from pos3n; cached keys are
    // already post-RoPE from when they were appended).
    auto apply_rope = [&](float* buf, int n_heads) {
        for (int t = 0; t < n; ++t) {
            for (int h = 0; h < n_heads; ++h) {
                float* base = buf + (static_cast<std::size_t>(t) * n_heads + h) * head_dim;
                for (int i = 0; i < half; ++i) {
                    const int pos = pos3n[axis_of[i] * n + t];
                    const float ang = pos * inv_freq[i];
                    const float c = std::cos(ang), s = std::sin(ang);
                    const float x0 = base[i], x1 = base[i + half];
                    base[i]        = x0 * c - x1 * s;
                    base[i + half] = x1 * c + x0 * s;
                }
            }
        }
    };

    std::vector<float> scores(total);
    std::vector<float> kbuf, vbuf;   // post-RoPE K / V for all `total` tokens
    for (int l = 0; l < num_layers; ++l) {
        const QwenTtsTalkerLayer& tl = layers[l];
        // ── self-attention ──
        bt::Tensor normed;
        bt::rms_norm_forward(hs, tl.in_ln, rms_eps, normed);
        bt::Tensor q, k, v;
        linear(tl.qw, nullptr, normed, q);   // (n, qd)
        linear(tl.kw, nullptr, normed, k);   // (n, kd)
        linear(tl.vw, nullptr, normed, v);   // (n, kd)
        // QK-norm (per head, over head_dim) then M-RoPE.
        head_rmsnorm(q.host_f32_mut(), n * n_q_heads,  head_dim, tl.q_norm.host_f32(), rms_eps);
        head_rmsnorm(k.host_f32_mut(), n * n_kv_heads, head_dim, tl.k_norm.host_f32(), rms_eps);
        apply_rope(q.host_f32_mut(), n_q_heads);
        apply_rope(k.host_f32_mut(), n_kv_heads);

        // Assemble the full K/V for this layer: cached prefix + new tokens.
        const std::size_t row_kv = static_cast<std::size_t>(kd);
        if (cache) {
            std::vector<float>& ck = cache->k[l];
            std::vector<float>& cv = cache->v[l];
            ck.insert(ck.end(), k.host_f32(), k.host_f32() + static_cast<std::size_t>(n) * kd);
            cv.insert(cv.end(), v.host_f32(), v.host_f32() + static_cast<std::size_t>(n) * kd);
            const float* kp = ck.data();
            const float* vp = cv.data();
            // attend below uses kp/vp; capture by recomputing pointers per use.
            const float* qp = q.host_f32();
            bt::Tensor ctx = bt::Tensor::mat(n, qd);
            float* cp = ctx.host_f32_mut();
            for (int h = 0; h < n_q_heads; ++h) {
                const int kvh = h / group;
                for (int t = 0; t < n; ++t) {
                    const int g = offset + t;   // global query position
                    const float* qi = qp + (static_cast<std::size_t>(t) * n_q_heads + h) * head_dim;
                    float maxs = -1e30f;
                    for (int jj = 0; jj <= g; ++jj) {
                        const float* kj = kp + static_cast<std::size_t>(jj) * row_kv + static_cast<std::size_t>(kvh) * head_dim;
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
                        dot *= scale;
                        scores[jj] = dot;
                        if (dot > maxs) maxs = dot;
                    }
                    float sum = 0.0f;
                    for (int jj = 0; jj <= g; ++jj) {
                        const float e = std::exp(scores[jj] - maxs);
                        scores[jj] = e; sum += e;
                    }
                    const float inv = 1.0f / sum;
                    float* ci = cp + (static_cast<std::size_t>(t) * n_q_heads + h) * head_dim;
                    for (int d = 0; d < head_dim; ++d) ci[d] = 0.0f;
                    for (int jj = 0; jj <= g; ++jj) {
                        const float w = scores[jj] * inv;
                        const float* vj = vp + static_cast<std::size_t>(jj) * row_kv + static_cast<std::size_t>(kvh) * head_dim;
                        for (int d = 0; d < head_dim; ++d) ci[d] += w * vj[d];
                    }
                }
            }
            bt::Tensor attn;
            linear(tl.ow, nullptr, ctx, attn);
            float* hp = hs.host_f32_mut();
            const float* ap = attn.host_f32();
            for (int i = 0; i < n * hidden; ++i) hp[i] += ap[i];
        } else {
            const float* qp = q.host_f32();
            const float* kp = k.host_f32();
            const float* vp = v.host_f32();
            bt::Tensor ctx = bt::Tensor::mat(n, qd);
            float* cp = ctx.host_f32_mut();
            for (int h = 0; h < n_q_heads; ++h) {
                const int kvh = h / group;
                for (int t = 0; t < n; ++t) {
                    const float* qi = qp + (static_cast<std::size_t>(t) * n_q_heads + h) * head_dim;
                    float maxs = -1e30f;
                    for (int jj = 0; jj <= t; ++jj) {
                        const float* kj = kp + (static_cast<std::size_t>(jj) * n_kv_heads + kvh) * head_dim;
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
                        dot *= scale;
                        scores[jj] = dot;
                        if (dot > maxs) maxs = dot;
                    }
                    float sum = 0.0f;
                    for (int jj = 0; jj <= t; ++jj) {
                        const float e = std::exp(scores[jj] - maxs);
                        scores[jj] = e; sum += e;
                    }
                    const float inv = 1.0f / sum;
                    float* ci = cp + (static_cast<std::size_t>(t) * n_q_heads + h) * head_dim;
                    for (int d = 0; d < head_dim; ++d) ci[d] = 0.0f;
                    for (int jj = 0; jj <= t; ++jj) {
                        const float w = scores[jj] * inv;
                        const float* vj = vp + (static_cast<std::size_t>(jj) * n_kv_heads + kvh) * head_dim;
                        for (int d = 0; d < head_dim; ++d) ci[d] += w * vj[d];
                    }
                }
            }
            bt::Tensor attn;
            linear(tl.ow, nullptr, ctx, attn);
            float* hp = hs.host_f32_mut();
            const float* ap = attn.host_f32();
            for (int i = 0; i < n * hidden; ++i) hp[i] += ap[i];
        }

        // ── SwiGLU MLP ──
        bt::Tensor n2;
        bt::rms_norm_forward(hs, tl.post_ln, rms_eps, n2);
        bt::Tensor g, u;
        linear(tl.gate, nullptr, n2, g);   // (n, intermediate)
        linear(tl.up, nullptr, n2, u);
        float* gp = g.host_f32_mut();
        const float* upp = u.host_f32();
        for (int i = 0; i < g.size(); ++i) gp[i] = silu(gp[i]) * upp[i];
        bt::Tensor dn;
        linear(tl.down, nullptr, g, dn);   // (n, hidden)
        float* hp = hs.host_f32_mut();
        const float* dp = dn.host_f32();
        for (int i = 0; i < n * hidden; ++i) hp[i] += dp[i];
    }

    if (cache) cache->len = total;
    bt::rms_norm_forward(hs, final_norm, rms_eps, hidden_out);   // (n, hidden)
}

void QwenTtsTalker::codec_logits(const float* hidden_rows, int n,
                                 bt::Tensor& logits_out) const {
    bt::DeviceScope cpu(bt::Device::CPU);
    if (n <= 0) { logits_out.resize(0, vocab); return; }
    bt::Tensor h = bt::Tensor::mat(n, hidden);
    std::copy(hidden_rows, hidden_rows + static_cast<std::size_t>(n) * hidden,
              h.host_f32_mut());
    linear(codec_head, nullptr, h, logits_out);   // (n, vocab)
}

void QwenTtsTalker::forward(const float* inputs_embeds, int T,
                            const int32_t* pos3T, bt::Tensor& hidden_out,
                            bt::Tensor& logits_out) const {
    run(inputs_embeds, T, pos3T, nullptr, hidden_out);
    codec_logits(hidden_out.host_f32(), T, logits_out);
}

void QwenTtsTalker::codec_embed(int id, float* out) const {
    bt::DeviceScope cpu(bt::Device::CPU);
    const float* row = codec_embedding.host_f32() + static_cast<std::size_t>(id) * hidden;
    std::copy(row, row + hidden, out);
}

const float* QwenTtsTalker::codec_embedding_row(int id) const {
    return codec_embedding.host_f32() + static_cast<std::size_t>(id) * hidden;
}

void QwenTtsTalker::text_embed_proj(int id, float* out) const {
    bt::DeviceScope cpu(bt::Device::CPU);
    bt::Tensor te = bt::Tensor::mat(1, text_hidden);
    const float* row = text_embedding.host_f32() + static_cast<std::size_t>(id) * text_hidden;
    std::copy(row, row + text_hidden, te.host_f32_mut());
    bt::Tensor h1;
    linear(tp_fc1_w, &tp_fc1_b, te, h1);   // (1, text_hidden)
    float* h1p = h1.host_f32_mut();
    for (int i = 0; i < h1.size(); ++i) h1p[i] = silu(h1p[i]);
    bt::Tensor h2;
    linear(tp_fc2_w, &tp_fc2_b, h1, h2);   // (1, hidden)
    std::copy(h2.host_f32(), h2.host_f32() + hidden, out);
}

}  // namespace brosoundml
