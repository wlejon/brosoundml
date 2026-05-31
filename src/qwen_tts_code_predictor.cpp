#include "qwen_tts_code_predictor.h"

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
    throw std::runtime_error("brosoundml: QwenTtsCodePredictor: " + msg);
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols) {
    bt::Tensor t;
    sf::upload_compute_checked(need(f, name), rows, cols, t, name);
    return t;
}
bt::Tensor up_vec(const sf::File& f, const std::string& name, int n) {
    return up(f, name, n, 1);
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

void linear(const bt::Tensor& W, const bt::Tensor& X, bt::Tensor& Y) {
    bt::Tensor zero = bt::Tensor::mat(W.rows, 1);
    bt::linear_forward_batched(W, zero, X, Y);
}

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

// A depth-local KV cache: per-layer post-RoPE K/V for the ≤16 depth tokens.
struct DepthCache {
    int len = 0;
    std::vector<std::vector<float>> k, v;
    void reset(int L) { len = 0; k.assign(L, {}); v.assign(L, {}); }
};

}  // namespace

void QwenTtsCodePredictor::load(const sf::File& f,
                                const QwenTtsCodePredictorConfig& cfg) {
    bt::DeviceScope cpu(bt::Device::CPU);

    num_layers      = cfg.transformer.num_hidden_layers;
    hidden          = cfg.transformer.hidden_size;
    intermediate    = cfg.transformer.intermediate_size;
    n_q_heads       = cfg.transformer.num_attention_heads;
    n_kv_heads      = cfg.transformer.num_key_value_heads;
    head_dim        = cfg.transformer.head_dim;
    vocab           = cfg.vocab_size;
    num_code_groups = cfg.num_code_groups;
    rms_eps         = cfg.transformer.rms_norm_eps;
    rope_theta      = cfg.transformer.rope_theta;

    const int n_tables = num_code_groups - 1;   // 15
    const std::string P = "talker.code_predictor.";
    final_norm = up_vec(f, P + "model.norm.weight", hidden);

    codec_embedding.clear();
    lm_head.clear();
    for (int j = 0; j < n_tables; ++j) {
        codec_embedding.push_back(
            up(f, P + "model.codec_embedding." + std::to_string(j) + ".weight", vocab, hidden));
        lm_head.push_back(
            up(f, P + "lm_head." + std::to_string(j) + ".weight", vocab, hidden));
    }

    const int qd = n_q_heads * head_dim;
    const int kd = n_kv_heads * head_dim;
    layers.clear();
    layers.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const std::string L = P + "model.layers." + std::to_string(i) + ".";
        QwenTtsCodePredictorLayer& cl = layers[i];
        cl.in_ln   = up_vec(f, L + "input_layernorm.weight", hidden);
        cl.post_ln = up_vec(f, L + "post_attention_layernorm.weight", hidden);
        cl.qw      = up(f, L + "self_attn.q_proj.weight", qd, hidden);
        cl.kw      = up(f, L + "self_attn.k_proj.weight", kd, hidden);
        cl.vw      = up(f, L + "self_attn.v_proj.weight", kd, hidden);
        cl.ow      = up(f, L + "self_attn.o_proj.weight", hidden, qd);
        cl.q_norm  = up_vec(f, L + "self_attn.q_norm.weight", head_dim);
        cl.k_norm  = up_vec(f, L + "self_attn.k_norm.weight", head_dim);
        cl.gate    = up(f, L + "mlp.gate_proj.weight", intermediate, hidden);
        cl.up      = up(f, L + "mlp.up_proj.weight", intermediate, hidden);
        cl.down    = up(f, L + "mlp.down_proj.weight", hidden, intermediate);
    }
}

namespace {

// One cached decoder pass over `n` new tokens at global positions
// [pos_start, pos_start+n). Plain single-axis rotate-half RoPE, full causal,
// GQA + QK-norm. Appends K/V to `cache` and writes hidden_out (n,hidden).
void cp_run(const QwenTtsCodePredictor& cp, const float* embeds, int n,
            int pos_start, DepthCache& cache, bt::Tensor& hidden_out) {
    const int hidden = cp.hidden, head_dim = cp.head_dim;
    const int n_q = cp.n_q_heads, n_kv = cp.n_kv_heads;
    const int qd = n_q * head_dim, kd = n_kv * head_dim;
    const int group = n_q / n_kv, half = head_dim / 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; ++i)
        inv_freq[i] = std::pow(cp.rope_theta, -(2.0f * i) / head_dim);

    const int offset = cache.len;
    const int total  = offset + n;

    bt::Tensor hs = bt::Tensor::mat(n, hidden);
    std::copy(embeds, embeds + static_cast<std::size_t>(n) * hidden, hs.host_f32_mut());

    auto apply_rope = [&](float* buf, int n_heads) {
        for (int t = 0; t < n; ++t) {
            const int pos = pos_start + t;
            for (int h = 0; h < n_heads; ++h) {
                float* base = buf + (static_cast<std::size_t>(t) * n_heads + h) * head_dim;
                for (int i = 0; i < half; ++i) {
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
    const std::size_t row_kv = static_cast<std::size_t>(kd);
    for (int l = 0; l < cp.num_layers; ++l) {
        const QwenTtsCodePredictorLayer& cl = cp.layers[l];
        bt::Tensor normed;
        bt::rms_norm_forward(hs, cl.in_ln, cp.rms_eps, normed);
        bt::Tensor q, k, v;
        linear(cl.qw, normed, q);
        linear(cl.kw, normed, k);
        linear(cl.vw, normed, v);
        head_rmsnorm(q.host_f32_mut(), n * n_q,  head_dim, cl.q_norm.host_f32(), cp.rms_eps);
        head_rmsnorm(k.host_f32_mut(), n * n_kv, head_dim, cl.k_norm.host_f32(), cp.rms_eps);
        apply_rope(q.host_f32_mut(), n_q);
        apply_rope(k.host_f32_mut(), n_kv);

        std::vector<float>& ck = cache.k[l];
        std::vector<float>& cv = cache.v[l];
        ck.insert(ck.end(), k.host_f32(), k.host_f32() + static_cast<std::size_t>(n) * kd);
        cv.insert(cv.end(), v.host_f32(), v.host_f32() + static_cast<std::size_t>(n) * kd);
        const float* kp = ck.data();
        const float* vp = cv.data();
        const float* qp = q.host_f32();

        bt::Tensor ctx = bt::Tensor::mat(n, qd);
        float* cptr = ctx.host_f32_mut();
        for (int h = 0; h < n_q; ++h) {
            const int kvh = h / group;
            for (int t = 0; t < n; ++t) {
                const int g = offset + t;
                const float* qi = qp + (static_cast<std::size_t>(t) * n_q + h) * head_dim;
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
                float* ci = cptr + (static_cast<std::size_t>(t) * n_q + h) * head_dim;
                for (int d = 0; d < head_dim; ++d) ci[d] = 0.0f;
                for (int jj = 0; jj <= g; ++jj) {
                    const float w = scores[jj] * inv;
                    const float* vj = vp + static_cast<std::size_t>(jj) * row_kv + static_cast<std::size_t>(kvh) * head_dim;
                    for (int d = 0; d < head_dim; ++d) ci[d] += w * vj[d];
                }
            }
        }
        bt::Tensor attn;
        linear(cl.ow, ctx, attn);
        {
            float* hp = hs.host_f32_mut();
            const float* ap = attn.host_f32();
            for (int i = 0; i < n * hidden; ++i) hp[i] += ap[i];
        }

        bt::Tensor n2;
        bt::rms_norm_forward(hs, cl.post_ln, cp.rms_eps, n2);
        bt::Tensor gg, uu;
        linear(cl.gate, n2, gg);
        linear(cl.up, n2, uu);
        float* gp = gg.host_f32_mut();
        const float* upp = uu.host_f32();
        for (int i = 0; i < gg.size(); ++i) gp[i] = silu(gp[i]) * upp[i];
        bt::Tensor dn;
        linear(cl.down, gg, dn);
        {
            float* hp = hs.host_f32_mut();
            const float* dp = dn.host_f32();
            for (int i = 0; i < n * hidden; ++i) hp[i] += dp[i];
        }
    }

    cache.len = total;
    bt::rms_norm_forward(hs, cp.final_norm, cp.rms_eps, hidden_out);
}

// argmax over a single logit row (vocab entries).
int argmax_row(const float* logits, int vocab) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < vocab; ++i) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

}  // namespace

void QwenTtsCodePredictor::predict(const float* past_hidden,
                                   const float* c0_embed,
                                   std::vector<int>& out_codes) const {
    bt::DeviceScope cpu(bt::Device::CPU);
    const int n_out = num_code_groups - 1;   // 15
    out_codes.assign(n_out, 0);

    DepthCache cache;
    cache.reset(num_layers);

    // ── prefill: two tokens [past_hidden, c0_embed] at positions 0, 1 ──
    std::vector<float> prefill(static_cast<std::size_t>(2) * hidden);
    std::copy(past_hidden, past_hidden + hidden, prefill.data());
    std::copy(c0_embed,    c0_embed + hidden,    prefill.data() + hidden);
    bt::Tensor h;
    cp_run(*this, prefill.data(), 2, /*pos_start=*/0, cache, h);

    // codebook 1 from lm_head[0] applied to the last prefill row.
    auto code_from = [&](const float* hidden_row, int head_idx) {
        bt::Tensor hr = bt::Tensor::mat(1, hidden);
        std::copy(hidden_row, hidden_row + hidden, hr.host_f32_mut());
        bt::Tensor lg;
        linear(lm_head[head_idx], hr, lg);   // (1, vocab)
        return argmax_row(lg.host_f32(), vocab);
    };
    out_codes[0] = code_from(h.host_f32() + static_cast<std::size_t>(hidden), 0);

    // ── steps: emit codebooks 2..(num_code_groups-1) ──
    // step j (j=1..n_out-1): embed code_j via codec_embedding[j-1] at the next
    // depth position, then lm_head[j] predicts code_{j+1}.
    for (int j = 1; j < n_out; ++j) {
        const int prev_code = out_codes[j - 1];
        const float* erow = codec_embedding_row(j - 1, prev_code);
        bt::Tensor hstep;
        cp_run(*this, erow, 1, /*pos_start=*/1 + j, cache, hstep);
        out_codes[j] = code_from(hstep.host_f32(), j);
    }
}

const float* QwenTtsCodePredictor::codec_embedding_row(int table, int id) const {
    return codec_embedding[table].host_f32() + static_cast<std::size_t>(id) * hidden;
}

}  // namespace brosoundml
