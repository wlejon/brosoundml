#include "qwen_tts_talker.h"

#include "qwen_tts_device.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#ifdef BROSOUNDML_HAS_CUDA
#include <brotensor/cuda_graph.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
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

using qtd::narrow_bf16;

// Per-step scratch — every intermediate of one captured (n=1) decoder pass.
// Persisting these lets the step run with no allocation, which is what makes
// it CUDA-graph-capturable (capture forbids mid-capture allocation).
struct StepScratch {
    bt::Tensor normed, q, k, v, qn, kn, qr, kr, ctx, attn, n2, g, u, dn;
};

}  // namespace

// Captured decode-session state. The KV cache lives here at FIXED capacity so
// every device pointer the step graph bakes stays stable across replays; the
// per-frame variables (append row, valid-key mask, RoPE row, input embedding)
// are staged into persistent device buffers by tiny d2d copies before launch.
struct QwenTtsTalkerStepState {
    int cap = 0;            // allocated cache rows
    int len = 0;            // valid rows (prefill + steps so far)
    std::vector<bt::Tensor> k, v;     // [layers] (cap, n_kv*head_dim) FP32
    bt::Tensor mask;        // (cap, 1) FP32 — 1 valid / 0 invalid key rows
    bt::Tensor ones;        // (1, 1) FP32 constant 1.0 (d2d mask updates)
    bt::Tensor ramp;        // (cap, 1) INT32 0..cap-1 (d2d index updates)
    bt::Tensor idx;         // (1, 1) INT32 — the step's append row
    bt::Tensor in;          // (1, hidden) staged input embedding
    bt::Tensor hs;          // (1, hidden) residual stream
    bt::Tensor hidden;      // (1, hidden) post-final-norm output
    bt::Tensor rope_cos, rope_sin;    // (cap, half) generation-phase tables
    bt::Tensor cos_step, sin_step;    // (1, half) the step's RoPE row
    StepScratch sc;
#ifdef BROSOUNDML_HAS_CUDA
    bt::CudaGraph graph;
#endif
    bool captured = false;
};

void QwenTtsTalkerStepDeleter::operator()(QwenTtsTalkerStepState* p) const noexcept {
    delete p;
}

QwenTtsTalker::QwenTtsTalker() = default;
QwenTtsTalker::~QwenTtsTalker() = default;
QwenTtsTalker::QwenTtsTalker(QwenTtsTalker&&) noexcept = default;
QwenTtsTalker& QwenTtsTalker::operator=(QwenTtsTalker&&) noexcept = default;

void QwenTtsTalker::load(const sf::File& f, const QwenTtsTalkerConfig& cfg,
                         bt::Device dev, bool bf16_weights) {
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
    tp_fc1_w = narrow_bf16(up(f, P + "text_projection.linear_fc1.weight", text_hidden, text_hidden, dev), bf16_weights);
    tp_fc1_b = up_vec(f, P + "text_projection.linear_fc1.bias", text_hidden, dev);
    tp_fc2_w = narrow_bf16(up(f, P + "text_projection.linear_fc2.weight", hidden, text_hidden, dev), bf16_weights);
    tp_fc2_b = up_vec(f, P + "text_projection.linear_fc2.bias", hidden, dev);
    codec_head = narrow_bf16(up(f, P + "codec_head.weight", vocab, hidden, dev), bf16_weights);

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
        tl.qw      = narrow_bf16(qtd::gather_rows(up(f, L + "self_attn.q_proj.weight", qd, hidden, dev), q_perm), bf16_weights);
        tl.kw      = narrow_bf16(qtd::gather_rows(up(f, L + "self_attn.k_proj.weight", kd, hidden, dev), k_perm), bf16_weights);
        tl.vw      = narrow_bf16(up(f, L + "self_attn.v_proj.weight", kd, hidden, dev), bf16_weights);
        tl.ow      = narrow_bf16(up(f, L + "self_attn.o_proj.weight", hidden, qd, dev), bf16_weights);
        tl.q_norm  = qtd::gather_rows(up_vec(f, L + "self_attn.q_norm.weight", head_dim, dev), hd_perm);
        tl.k_norm  = qtd::gather_rows(up_vec(f, L + "self_attn.k_norm.weight", head_dim, dev), hd_perm);
        tl.gate    = narrow_bf16(up(f, L + "mlp.gate_proj.weight", intermediate, hidden, dev), bf16_weights);
        tl.up      = narrow_bf16(up(f, L + "mlp.up_proj.weight", intermediate, hidden, dev), bf16_weights);
        tl.down    = narrow_bf16(up(f, L + "mlp.down_proj.weight", hidden, intermediate, dev), bf16_weights);
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

    const int qd     = n_q_heads * head_dim;    // query width
    const int kd     = n_kv_heads * head_dim;   // K/V (GQA) cache width
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
                bt::Tensor nk = bt::Tensor::zeros_on(dev, newcap, kd, bt::Dtype::FP32);
                bt::Tensor nv = bt::Tensor::zeros_on(dev, newcap, kd, bt::Dtype::FP32);
                if (offset > 0) {
                    bt::copy_d2d(cache->k[l], 0, nk, 0, offset * kd);
                    bt::copy_d2d(cache->v[l], 0, nv, 0, offset * kd);
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

        // The K/V cache holds the n_kv (grouped) heads; the windowed attention
        // op expands them to the n_q query heads internally (GQA), so no
        // per-layer expand_kv gather is needed.
        bt::Tensor ctx;
        if (cache) {
            bt::copy_d2d(kr, 0, cache->k[l], offset * kd, n * kd);
            bt::copy_d2d(v,  0, cache->v[l], offset * kd, n * kd);
            const int valid = offset + n;
            bt::Tensor Kf = bt::Tensor::view(dev, cache->k[l].data, valid, kd, bt::Dtype::FP32);
            bt::Tensor Vf = bt::Tensor::view(dev, cache->v[l].data, valid, kd, bt::Dtype::FP32);
            // offset==0: a fresh prefill (Lq==Lk) is causal; a later step has a
            // single query at the latest position attending the whole cache.
            qtd::flash_attn(qr, Kf, Vf, n_q_heads, head_dim, /*causal=*/offset == 0, ctx);
        } else {
            qtd::flash_attn(qr, kr, v, n_q_heads, head_dim, /*causal=*/true, ctx);
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

namespace {

// One fixed-shape decode step over the session state: the exact op sequence of
// run_dev with n == 1, but over persistent buffers (no allocation), a
// device-resident append row (scatter_rows), and full-capacity masked
// attention. Masked-out keys are skipped before the dot product and softmax to
// exact zeros, so the result is bit-identical to run_dev's growing-cache call;
// allocation-free means the whole body is CUDA-graph-capturable.
void talker_step_body(const QwenTtsTalker& t, QwenTtsTalkerStepState& st) {
    StepScratch& sc = st.sc;
    bt::copy_d2d(st.in, 0, st.hs, 0, t.hidden);

    for (int l = 0; l < t.num_layers; ++l) {
        const QwenTtsTalkerLayer& tl = t.layers[l];

        bt::rms_norm_forward(st.hs, tl.in_ln, t.rms_eps, sc.normed);
        qtd::linear(tl.qw, nullptr, sc.normed, sc.q);
        qtd::linear(tl.kw, nullptr, sc.normed, sc.k);
        qtd::linear(tl.vw, nullptr, sc.normed, sc.v);
        qtd::head_rms_norm(sc.q, 1, t.n_q_heads,  t.head_dim, tl.q_norm, t.rms_eps, sc.qn);
        qtd::head_rms_norm(sc.k, 1, t.n_kv_heads, t.head_dim, tl.k_norm, t.rms_eps, sc.kn);
        bt::rope_apply(sc.qn, st.cos_step, st.sin_step, t.head_dim, t.n_q_heads,  sc.qr);
        bt::rope_apply(sc.kn, st.cos_step, st.sin_step, t.head_dim, t.n_kv_heads, sc.kr);

        bt::scatter_rows(sc.kr, st.idx, st.k[l]);
        bt::scatter_rows(sc.v,  st.idx, st.v[l]);
        qtd::flash_attn_masked(sc.qr, st.k[l], st.v[l],
                               static_cast<const float*>(st.mask.data),
                               t.n_q_heads, sc.ctx);
        qtd::linear(tl.ow, nullptr, sc.ctx, sc.attn);
        bt::add_inplace_batched(st.hs, sc.attn);

        bt::rms_norm_forward(st.hs, tl.post_ln, t.rms_eps, sc.n2);
        qtd::linear(tl.gate, nullptr, sc.n2, sc.g);
        qtd::linear(tl.up,   nullptr, sc.n2, sc.u);
        qtd::swiglu(sc.g, sc.u);
        qtd::linear(tl.down, nullptr, sc.g, sc.dn);
        bt::add_inplace_batched(st.hs, sc.dn);
    }

    bt::rms_norm_forward(st.hs, t.final_norm, t.rms_eps, st.hidden);
}

// (Re)allocate the session to `cap` rows, preserving the first `keep` valid
// rows of K/V and the mask (the grow-mid-utterance path; keep == 0 on a fresh
// session). Invalidates any captured graph — the cache pointers change.
void talker_state_alloc(const QwenTtsTalker& t, QwenTtsTalkerStepState& st,
                        int cap, int keep) {
    const bt::Device dev = t.final_norm.device;
    const int kd = t.n_kv_heads * t.head_dim;
    const int half = t.head_dim / 2;

    std::vector<bt::Tensor> nk(t.num_layers), nv(t.num_layers);
    for (int l = 0; l < t.num_layers; ++l) {
        nk[l] = bt::Tensor::zeros_on(dev, cap, kd, bt::Dtype::FP32);
        nv[l] = bt::Tensor::zeros_on(dev, cap, kd, bt::Dtype::FP32);
        if (keep > 0) {
            bt::copy_d2d(st.k[l], 0, nk[l], 0, keep * kd);
            bt::copy_d2d(st.v[l], 0, nv[l], 0, keep * kd);
        }
    }
    st.k = std::move(nk);
    st.v = std::move(nv);

    // Mask: 1 for the kept rows, 0 beyond (host rebuild — one tiny upload).
    std::vector<float> hmask(static_cast<std::size_t>(cap), 0.0f);
    std::fill(hmask.begin(), hmask.begin() + keep, 1.0f);
    st.mask = bt::Tensor::from_host_on(dev, hmask.data(), cap, 1);

    std::vector<std::int32_t> hramp(static_cast<std::size_t>(cap));
    for (int i = 0; i < cap; ++i) hramp[i] = i;
    st.ramp = qtd::upload_idx(dev, hramp.data(), cap);

    const float one = 1.0f;
    st.ones = bt::Tensor::from_host_on(dev, &one, 1, 1);
    st.idx  = bt::Tensor::empty_on(dev, 1, 1, bt::Dtype::INT32);
    st.in     = bt::Tensor::empty_on(dev, 1, t.hidden, bt::Dtype::FP32);
    st.hs     = bt::Tensor::empty_on(dev, 1, t.hidden, bt::Dtype::FP32);
    st.hidden = bt::Tensor::empty_on(dev, 1, t.hidden, bt::Dtype::FP32);
    st.cos_step = bt::Tensor::empty_on(dev, 1, half, bt::Dtype::FP32);
    st.sin_step = bt::Tensor::empty_on(dev, 1, half, bt::Dtype::FP32);

    // Generation-phase RoPE rows 0..cap-1. In the generation phase all three
    // M-RoPE axes carry the same scalar position, so the axis interleave is
    // irrelevant and the table is plain RoPE — built with the same float
    // expressions as run_dev's per-step host build, so the rows are bitwise
    // identical to what the eager path uploads.
    std::vector<float> cb(static_cast<std::size_t>(cap) * half);
    std::vector<float> sb(static_cast<std::size_t>(cap) * half);
    for (int i = 0; i < half; ++i) {
        const float inv_freq = std::pow(t.rope_theta, -(2.0f * i) / t.head_dim);
        for (int p = 0; p < cap; ++p) {
            const float ang = static_cast<float>(p) * inv_freq;
            cb[static_cast<std::size_t>(p) * half + i] = std::cos(ang);
            sb[static_cast<std::size_t>(p) * half + i] = std::sin(ang);
        }
    }
    st.rope_cos = bt::Tensor::from_host_on(dev, cb.data(), cap, half);
    st.rope_sin = bt::Tensor::from_host_on(dev, sb.data(), cap, half);

    st.cap = cap;
#ifdef BROSOUNDML_HAS_CUDA
    st.graph.reset();
#endif
    st.captured = false;
}

}  // namespace

bool QwenTtsTalker::decode_begin(QwenTtsTalkerStepPtr& st_ptr, int min_cap) const {
    if (final_norm.device != bt::Device::CUDA) return false;
    bt::DeviceScope scope(final_norm.device);
    if (!st_ptr) st_ptr.reset(new QwenTtsTalkerStepState());
    QwenTtsTalkerStepState& st = *st_ptr;
    // Bucket the capacity so back-to-back utterances of similar length reuse
    // the captured graph; only grow (a bigger session serves smaller calls).
    const int bucket = ((std::max(min_cap, 256) + 511) / 512) * 512;
    if (st.cap < bucket) talker_state_alloc(*this, st, bucket, /*keep=*/0);
    st.len = 0;
    return true;
}

void QwenTtsTalker::decode_prefill(QwenTtsTalkerStepState& st, const float* embeds,
                                   int T, const int32_t* pos3T,
                                   bt::Tensor& hidden_out) const {
    const bt::Device dev = final_norm.device;
    bt::DeviceScope scope(dev);
    const int kd = n_kv_heads * head_dim;
    if (T > st.cap) {
        throw std::runtime_error(
            "brosoundml: QwenTtsTalker::decode_prefill: prefill exceeds session capacity");
    }

    // Run the ordinary prefill into the session's fixed buffers via a cache of
    // views — same compute as run(), same K/V rows, no growth (cap covers T).
    QwenTtsTalkerCache tmp;
    tmp.reset(num_layers);
    tmp.cap = st.cap;
    for (int l = 0; l < num_layers; ++l) {
        tmp.k[l] = bt::Tensor::view(dev, st.k[l].data, st.cap, kd, bt::Dtype::FP32);
        tmp.v[l] = bt::Tensor::view(dev, st.v[l].data, st.cap, kd, bt::Dtype::FP32);
    }
    run(embeds, T, pos3T, &tmp, hidden_out);
    st.len = T;

    // Valid-key mask: rows [0, T) on, the rest off.
    std::vector<float> hmask(static_cast<std::size_t>(st.cap), 0.0f);
    std::fill(hmask.begin(), hmask.begin() + T, 1.0f);
    bt::detail::alloc_for(dev).memcpy_h2d(
        st.mask.data, hmask.data(), static_cast<std::size_t>(st.cap) * sizeof(float));
}

void QwenTtsTalker::decode_step(QwenTtsTalkerStepState& st, const bt::Tensor& embed,
                                int pos, bt::Tensor& hidden_view) const {
    const bt::Device dev = final_norm.device;
    bt::DeviceScope scope(dev);
    const int half = head_dim / 2;

    // Out of room (cache row or RoPE table row): grow + re-capture. Keeps the
    // valid rows and mask; one-time cost, amortized over the longer utterance.
    if (st.len >= st.cap || pos >= st.cap) {
        const int need = std::max(st.len + 1, pos + 1);
        talker_state_alloc(*this, st, std::max(st.cap * 2, ((need + 511) / 512) * 512),
                           /*keep=*/st.len);
    }

    // Stage the step's variables into the captured buffers — all device-side
    // copies, stream-ordered ahead of the replay.
    bt::copy_d2d(st.ramp, st.len, st.idx, 0, 1);              // append row
    bt::copy_d2d(st.ones, 0, st.mask, st.len, 1);             // mask[len] = 1
    bt::copy_d2d(st.rope_cos, pos * half, st.cos_step, 0, half);
    bt::copy_d2d(st.rope_sin, pos * half, st.sin_step, 0, half);
    bt::copy_d2d(embed, 0, st.in, 0, hidden);

#ifdef BROSOUNDML_HAS_CUDA
    if (!st.captured) {
        // Warm-up sizes every scratch buffer (capture must not allocate). The
        // step body is idempotent over the staged inputs — the capture re-run
        // recomputes the identical row — so no state reset is needed between
        // the warm-up and the captured run.
        talker_step_body(*this, st);
        bt::sync_all();
        {
            bt::CudaGraphCapture cap;
            talker_step_body(*this, st);
            st.graph = cap.finish();
        }
        st.captured = true;
    } else {
        st.graph.launch();
    }
#else
    talker_step_body(*this, st);
#endif

    st.len += 1;
    hidden_view = bt::Tensor::view(dev, st.hidden.data, 1, hidden, bt::Dtype::FP32);
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
