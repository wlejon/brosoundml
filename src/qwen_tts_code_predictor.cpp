#include "qwen_tts_code_predictor.h"

#include "qwen_tts_device.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace {

// Env-gated internal profiling for predict_dev: splits the 2-token prefill pass
// from the 14 single-token depth steps, and the per-step argmax read-back.
struct CpProf {
    static bool on() {
        static const bool v = std::getenv("BROSOUNDML_QWEN_PROFILE") != nullptr;
        return v;
    }
    static double& prefill() { static double v = 0; return v; }
    static double& steps()   { static double v = 0; return v; }
    static double& argmax()  { static double v = 0; return v; }
    static long&   calls()   { static long v = 0;   return v; }
};

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

// A depth-local KV cache: per-layer post-RoPE K/V for the <=16 depth tokens,
// stored expanded to full per-query heads (plain MHA for flash). FP32 device
// tensors, preallocated to the known depth length.
struct DepthCache {
    int len = 0;
    std::vector<bt::Tensor> k, v;
    void reset(int num_layers, int cap, bt::Device dev, int D) {
        len = 0;
        k.assign(num_layers, {});
        v.assign(num_layers, {});
        for (int l = 0; l < num_layers; ++l) {
            k[l] = bt::Tensor::zeros_on(dev, cap, D, bt::Dtype::FP32);
            v[l] = bt::Tensor::zeros_on(dev, cap, D, bt::Dtype::FP32);
        }
    }
};

// One cached decoder pass over `n` new tokens at depth positions
// [pos_start, pos_start+n). Plain single-axis RoPE, GQA + QK-norm, full causal.
// `embeds` is an (n, hidden) device tensor; writes hidden_out (n, hidden).
void cp_run(const QwenTtsCodePredictor& cp, const bt::Tensor& embeds, int n,
            int pos_start, DepthCache& cache, bt::Tensor& hidden_out) {
    const bt::Device dev = cp.final_norm.device;
    const int head_dim = cp.head_dim;
    const int n_q = cp.n_q_heads, n_kv = cp.n_kv_heads;
    const int qd = n_q * head_dim;          // == expanded K/V width
    const int half = head_dim / 2;
    const int offset = cache.len;

    // Plain RoPE tables for positions [pos_start, pos_start+n): slice the
    // precomputed (num_code_groups, half) tables — contiguous rows in row-major,
    // so this is a view, with no per-step host build or host->device upload.
    bt::Tensor cosT = bt::Tensor::view(
        dev, static_cast<float*>(cp.rope_cos.data) + static_cast<std::size_t>(pos_start) * half,
        n, half, bt::Dtype::FP32);
    bt::Tensor sinT = bt::Tensor::view(
        dev, static_cast<float*>(cp.rope_sin.data) + static_cast<std::size_t>(pos_start) * half,
        n, half, bt::Dtype::FP32);

    bt::Tensor hs = embeds;   // deep copy; mutated in place by residual adds

    for (int l = 0; l < cp.num_layers; ++l) {
        const QwenTtsCodePredictorLayer& cl = cp.layers[l];
        bt::Tensor normed;
        bt::rms_norm_forward(hs, cl.in_ln, cp.rms_eps, normed);
        bt::Tensor q, k, v;
        qtd::linear(cl.qw, nullptr, normed, q);
        qtd::linear(cl.kw, nullptr, normed, k);
        qtd::linear(cl.vw, nullptr, normed, v);
        bt::Tensor qn, kn;
        qtd::head_rms_norm(q, n, n_q,  head_dim, cl.q_norm, cp.rms_eps, qn);
        qtd::head_rms_norm(k, n, n_kv, head_dim, cl.k_norm, cp.rms_eps, kn);
        bt::Tensor qr, kr;
        bt::rope_apply(qn, cosT, sinT, head_dim, n_q,  qr);
        bt::rope_apply(kn, cosT, sinT, head_dim, n_kv, kr);
        bt::Tensor kE = qtd::expand_kv(kr, n, n_kv, n_q, head_dim);
        bt::Tensor vE = qtd::expand_kv(v,  n, n_kv, n_q, head_dim);

        bt::copy_d2d(kE, 0, cache.k[l], offset * qd, n * qd);
        bt::copy_d2d(vE, 0, cache.v[l], offset * qd, n * qd);
        const int valid = offset + n;
        bt::Tensor Kf = bt::Tensor::view(dev, cache.k[l].data, valid, qd, bt::Dtype::FP32);
        bt::Tensor Vf = bt::Tensor::view(dev, cache.v[l].data, valid, qd, bt::Dtype::FP32);
        bt::Tensor ctx;
        qtd::flash_attn(qr, Kf, Vf, n_q, head_dim, /*causal=*/offset == 0, ctx);
        bt::Tensor attn;
        qtd::linear(cl.ow, nullptr, ctx, attn);
        bt::add_inplace_batched(hs, attn);

        bt::Tensor n2;
        bt::rms_norm_forward(hs, cl.post_ln, cp.rms_eps, n2);
        bt::Tensor g, u;
        qtd::linear(cl.gate, nullptr, n2, g);
        qtd::linear(cl.up,   nullptr, n2, u);
        qtd::swiglu(g, u);
        bt::Tensor dn;
        qtd::linear(cl.down, nullptr, g, dn);
        bt::add_inplace_batched(hs, dn);
    }

    cache.len = offset + n;
    bt::rms_norm_forward(hs, cp.final_norm, cp.rms_eps, hidden_out);
}

}  // namespace

void QwenTtsCodePredictor::load(const sf::File& f,
                                const QwenTtsCodePredictorConfig& cfg,
                                bt::Device dev) {
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
    final_norm = up_vec(f, P + "model.norm.weight", hidden, dev);

    codec_embedding.clear();
    lm_head.clear();
    for (int j = 0; j < n_tables; ++j) {
        codec_embedding.push_back(
            up(f, P + "model.codec_embedding." + std::to_string(j) + ".weight", vocab, hidden, dev));
        lm_head.push_back(
            up(f, P + "lm_head." + std::to_string(j) + ".weight", vocab, hidden, dev));
    }

    // Same HF rotate-half -> adjacent-pair RoPE permutation as the Talker.
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
        QwenTtsCodePredictorLayer& cl = layers[i];
        cl.in_ln   = up_vec(f, L + "input_layernorm.weight", hidden, dev);
        cl.post_ln = up_vec(f, L + "post_attention_layernorm.weight", hidden, dev);
        cl.qw      = qtd::gather_rows(up(f, L + "self_attn.q_proj.weight", qd, hidden, dev), q_perm);
        cl.kw      = qtd::gather_rows(up(f, L + "self_attn.k_proj.weight", kd, hidden, dev), k_perm);
        cl.vw      = up(f, L + "self_attn.v_proj.weight", kd, hidden, dev);
        cl.ow      = up(f, L + "self_attn.o_proj.weight", hidden, qd, dev);
        cl.q_norm  = qtd::gather_rows(up_vec(f, L + "self_attn.q_norm.weight", head_dim, dev), hd_perm);
        cl.k_norm  = qtd::gather_rows(up_vec(f, L + "self_attn.k_norm.weight", head_dim, dev), hd_perm);
        cl.gate    = up(f, L + "mlp.gate_proj.weight", intermediate, hidden, dev);
        cl.up      = up(f, L + "mlp.up_proj.weight", intermediate, hidden, dev);
        cl.down    = up(f, L + "mlp.down_proj.weight", hidden, intermediate, dev);
    }

    // Precompute the plain-RoPE cos/sin tables for the fixed depth positions
    // 0..num_code_groups-1 (interleaved-pair layout, matching build_rope_tables).
    const int half = head_dim / 2;
    std::vector<float> cb(static_cast<std::size_t>(num_code_groups) * half);
    std::vector<float> sb(static_cast<std::size_t>(num_code_groups) * half);
    for (int p = 0; p < num_code_groups; ++p) {
        for (int i = 0; i < half; ++i) {
            const float ang = static_cast<float>(p) *
                std::pow(rope_theta, -(2.0f * i) / head_dim);
            cb[static_cast<std::size_t>(p) * half + i] = std::cos(ang);
            sb[static_cast<std::size_t>(p) * half + i] = std::sin(ang);
        }
    }
    rope_cos = bt::Tensor::from_host_on(dev, cb.data(), num_code_groups, half);
    rope_sin = bt::Tensor::from_host_on(dev, sb.data(), num_code_groups, half);
}

void QwenTtsCodePredictor::predict(const float* past_hidden,
                                   const float* c0_embed,
                                   std::vector<int>& out_codes) const {
    const bt::Device dev = final_norm.device;
    bt::DeviceScope scope(dev);
    bt::Tensor ph = bt::Tensor::from_host_on(dev, past_hidden, 1, hidden);
    bt::Tensor ce = bt::Tensor::from_host_on(dev, c0_embed,    1, hidden);
    predict_dev(ph, ce, out_codes);
}

void QwenTtsCodePredictor::predict_dev(const bt::Tensor& past_hidden,
                                       const bt::Tensor& c0_embed,
                                       std::vector<int>& out_codes) const {
    const bt::Device dev = final_norm.device;
    bt::DeviceScope scope(dev);
    const int n_out = num_code_groups - 1;   // 15
    const int qd    = n_q_heads * head_dim;
    out_codes.assign(n_out, 0);

    DepthCache cache;
    cache.reset(num_layers, num_code_groups, dev, qd);

    // lm_head[head_idx] over one (1,hidden) device row -> on-device argmax over
    // vocab. argmax_rows returns the index as a float in a (1,1) tensor; only
    // that scalar comes back to the host (4 bytes), not the whole vocab row.
    // Ties keep the lowest index, matching the greedy host reference.
    const bool prof = CpProf::on();
    using clk = std::chrono::steady_clock;
    auto code_from = [&](const bt::Tensor& row, int head_idx) -> int {
        bt::Tensor lg;
        qtd::linear(lm_head[head_idx], nullptr, row, lg);   // (1, vocab)
        bt::Tensor idx;
        bt::argmax_rows(lg, idx);                           // (1, 1) FP32
        clk::time_point t0;
        if (prof) { bt::sync_all(); t0 = clk::now(); }
        float f = 0.0f;
        qtd::to_host(idx, &f);
        if (prof) CpProf::argmax() +=
            std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        return static_cast<int>(f);
    };

    // ── prefill: two tokens [past_hidden, c0_embed] at depth positions 0, 1,
    //    assembled on-device (no host staging of the two hidden-width rows) ──
    clk::time_point tpf;
    if (prof) { bt::sync_all(); tpf = clk::now(); ++CpProf::calls(); }
    bt::Tensor pre_t = bt::Tensor::empty_on(dev, 2, hidden, bt::Dtype::FP32);
    bt::copy_d2d(past_hidden, 0, pre_t, 0,      hidden);
    bt::copy_d2d(c0_embed,    0, pre_t, hidden, hidden);
    bt::Tensor h;
    cp_run(*this, pre_t, 2, /*pos_start=*/0, cache, h);   // (2, hidden)

    // codebook 1 from lm_head[0] applied to the last prefill row (row 1).
    bt::Tensor row1 = bt::Tensor::view(
        dev, static_cast<float*>(h.data) + hidden, 1, hidden, bt::Dtype::FP32);
    out_codes[0] = code_from(row1, 0);
    if (prof) { bt::sync_all(); CpProf::prefill() +=
        std::chrono::duration<double, std::milli>(clk::now() - tpf).count(); }

    // ── steps: emit codebooks 2..(num_code_groups-1) ──
    clk::time_point tst;
    if (prof) { bt::sync_all(); tst = clk::now(); }
    for (int j = 1; j < n_out; ++j) {
        const int prev_code = out_codes[j - 1];
        bt::Tensor erow = qtd::gather_rows(
            codec_embedding[j - 1], {static_cast<std::int32_t>(prev_code)});  // (1, hidden)
        bt::Tensor hstep;
        cp_run(*this, erow, 1, /*pos_start=*/1 + j, cache, hstep);
        out_codes[j] = code_from(hstep, j);
    }
    if (prof) { bt::sync_all(); CpProf::steps() +=
        std::chrono::duration<double, std::milli>(clk::now() - tst).count(); }
}

// Print + reset the Code Predictor internal profile (called from generate_codes).
void qwen_cp_profile_report() {
    if (!CpProf::on() || CpProf::calls() == 0) return;
    const double f = static_cast<double>(CpProf::calls());
    std::fprintf(stderr,
        "  [code_predictor internals over %ld frames]\n"
        "    prefill (2-tok, 5L)  %9.1f ms  %7.3f ms/frame\n"
        "    14 depth steps (5L)  %9.1f ms  %7.3f ms/frame\n"
        "      of which argmax rb %9.1f ms  %7.3f ms/frame\n",
        CpProf::calls(),
        CpProf::prefill(), CpProf::prefill() / f,
        CpProf::steps(),   CpProf::steps() / f,
        CpProf::argmax(),  CpProf::argmax() / f);
    CpProf::prefill() = CpProf::steps() = CpProf::argmax() = 0;
    CpProf::calls() = 0;
}

const float* QwenTtsCodePredictor::codec_embedding_row(int table, int id) const {
    // Host pointer into the table — valid only when CPU-resident (host AR path).
    return codec_embedding[table].host_f32() + static_cast<std::size_t>(id) * hidden;
}

}  // namespace brosoundml
