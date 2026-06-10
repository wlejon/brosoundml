#include "qwen_tts_code_predictor.h"

#include "qwen_tts_device.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#ifdef BROSOUNDML_HAS_CUDA
#include <brotensor/cuda_graph.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace {

// Env-gated internal profiling for predict_dev: the per-frame depth compute
// (whole-frame eager body, or one CUDA-graph replay) and the final batched
// code read-back. `graph` records whether the steady-state path is graph replay.
struct CpProf {
    static bool on() {
        static const bool v = std::getenv("BROSOUNDML_QWEN_PROFILE") != nullptr;
        return v;
    }
    static double& frame()    { static double v = 0; return v; }
    static double& readback() { static double v = 0; return v; }
    static long&   calls()    { static long v = 0;   return v; }
    static bool&   graph()    { static bool v = false; return v; }
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
// stored as the n_kv (grouped) heads — the windowed attention op expands to the
// n_q query heads internally (GQA). FP32 device tensors, allocated once to the
// known depth length and reused across frames (len resets to 0 per frame).
struct DepthCache {
    int len = 0;
    std::vector<bt::Tensor> k, v;
    bool allocated() const { return !k.empty(); }
    void alloc(int num_layers, int cap, bt::Device dev, int D) {
        k.assign(num_layers, {});
        v.assign(num_layers, {});
        for (int l = 0; l < num_layers; ++l) {
            k[l] = bt::Tensor::zeros_on(dev, cap, D, bt::Dtype::FP32);
            v[l] = bt::Tensor::zeros_on(dev, cap, D, bt::Dtype::FP32);
        }
    }
};

// Per-decoder-pass scratch — every intermediate of one cp_run_into call. Reused
// across all layers within a pass (each layer fully overwrites it) and across
// frames. Persisting these is what lets a frame allocate nothing on the steady
// path and lets the whole pass be recorded into a CUDA graph (capture forbids
// mid-capture allocation, so the buffers must already exist and be reused).
struct LayerScratch {
    bt::Tensor normed, q, k, v, qn, kn, qr, kr, ctx, attn, n2, g, u, dn;
};

}  // namespace

// Persistent decode state for one predictor instance: the reused scratch for a
// frame's prefill (n=2) and depth-step (n=1) passes, the depth KV cache, the
// conditioning-input staging buffer, and the (15,1) INT32 code buffer. On CUDA
// greedy it also owns the captured whole-frame graph. At namespace scope (not
// anonymous) so it is the same brosoundml::CpFrameState the header forward-
// declares for the predictor's unique_ptr member; it still uses the
// anonymous-namespace DepthCache / LayerScratch (visible in the enclosing
// namespace).
struct CpFrameState {
    bt::Tensor cond_in;        // (2, talker_hidden) conditioning rows, in place
    bt::Tensor proj_pre;       // (2, hidden) prefill projection (1.7B only)
    bt::Tensor proj_step;      // (1, hidden) step projection (1.7B only)
    bt::Tensor hs_pre;         // (2, hidden) prefill residual stream
    bt::Tensor hs_step;        // (1, hidden) step residual stream
    LayerScratch sc_pre;       // prefill (n=2) layer scratch
    LayerScratch sc_step;      // step (n=1) layer scratch
    bt::Tensor erow;           // (1, talker_hidden) gathered code embedding
    bt::Tensor hidden_pre;     // (2, hidden) prefill final-norm output
    bt::Tensor hidden_step;    // (1, hidden) step final-norm output
    bt::Tensor logits;         // (1, vocab) lm_head output
    bt::Tensor code_dev;       // (n_out, 1) INT32 accumulated codes
    DepthCache cache;
#ifdef BROSOUNDML_HAS_CUDA
    bt::CudaGraph graph;       // captured whole-frame greedy step (CUDA only)
#endif
    bool captured = false;
};

namespace {

// One cached decoder pass over `n` new tokens at depth positions
// [pos_start, pos_start+n). Plain single-axis RoPE, GQA + QK-norm, full causal.
// `embeds` (n, hidden) is copied into the persistent residual `hs`; all
// intermediates land in `sc`; the final-norm output is written to `hidden_out`.
// Every tensor it touches is persistent and reused — sized on the first (warm-up)
// call, a no-op resize thereafter — so a captured replay re-runs with no alloc.
void cp_run_into(const QwenTtsCodePredictor& cp, const bt::Tensor& embeds, int n,
                 int pos_start, DepthCache& cache, LayerScratch& sc,
                 bt::Tensor& hs, bt::Tensor& hidden_out) {
    const bt::Device dev = cp.final_norm.device;
    const int head_dim = cp.head_dim;
    const int n_q = cp.n_q_heads, n_kv = cp.n_kv_heads;
    const int kd = n_kv * head_dim;          // K/V (GQA) cache width
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

    // Residual stream starts as a copy of the (projected) input rows. Sized on
    // the warm-up call; a plain device copy into the same buffer thereafter.
    if (hs.rows != n || hs.cols != cp.hidden || hs.dtype != bt::Dtype::FP32)
        hs = bt::Tensor::empty_on(dev, n, cp.hidden, bt::Dtype::FP32);
    bt::copy_d2d(embeds, 0, hs, 0, n * cp.hidden);

    for (int l = 0; l < cp.num_layers; ++l) {
        const QwenTtsCodePredictorLayer& cl = cp.layers[l];
        bt::rms_norm_forward(hs, cl.in_ln, cp.rms_eps, sc.normed);
        qtd::linear(cl.qw, nullptr, sc.normed, sc.q);
        qtd::linear(cl.kw, nullptr, sc.normed, sc.k);
        qtd::linear(cl.vw, nullptr, sc.normed, sc.v);
        qtd::head_rms_norm(sc.q, n, n_q,  head_dim, cl.q_norm, cp.rms_eps, sc.qn);
        qtd::head_rms_norm(sc.k, n, n_kv, head_dim, cl.k_norm, cp.rms_eps, sc.kn);
        bt::rope_apply(sc.qn, cosT, sinT, head_dim, n_q,  sc.qr);
        bt::rope_apply(sc.kn, cosT, sinT, head_dim, n_kv, sc.kr);
        bt::copy_d2d(sc.kr, 0, cache.k[l], offset * kd, n * kd);
        bt::copy_d2d(sc.v,  0, cache.v[l], offset * kd, n * kd);
        const int valid = offset + n;
        bt::Tensor Kf = bt::Tensor::view(dev, cache.k[l].data, valid, kd, bt::Dtype::FP32);
        bt::Tensor Vf = bt::Tensor::view(dev, cache.v[l].data, valid, kd, bt::Dtype::FP32);
        qtd::flash_attn(sc.qr, Kf, Vf, n_q, head_dim, /*causal=*/offset == 0, sc.ctx);
        qtd::linear(cl.ow, nullptr, sc.ctx, sc.attn);
        bt::add_inplace_batched(hs, sc.attn);

        bt::rms_norm_forward(hs, cl.post_ln, cp.rms_eps, sc.n2);
        qtd::linear(cl.gate, nullptr, sc.n2, sc.g);
        qtd::linear(cl.up,   nullptr, sc.n2, sc.u);
        qtd::swiglu(sc.g, sc.u);
        qtd::linear(cl.down, nullptr, sc.g, sc.dn);
        bt::add_inplace_batched(hs, sc.dn);
    }

    cache.len = offset + n;
    bt::rms_norm_forward(hs, cp.final_norm, cp.rms_eps, hidden_out);
}

// Project an (n, talker_hidden) input down to the depth-transformer hidden via
// small_to_mtp_projection, into the persistent `scratch`. Identity (returns the
// input unchanged) when widths match (0.6B). Returns a reference, never a copy.
const bt::Tensor& cp_project_into(const QwenTtsCodePredictor& cp,
                                  const bt::Tensor& in, bt::Tensor& scratch) {
    if (!cp.has_mtp_proj) return in;
    qtd::linear(cp.mtp_proj_w, &cp.mtp_proj_b, in, scratch);
    return scratch;
}

// The full per-frame depth compute over the persistent state: a 2-token prefill
// then the 14 single-token steps, emitting all 15 codes into st.code_dev. Every
// op is device-resident and reuses st's buffers, so this same body runs eagerly
// (CPU / sampling) and is the exact sequence captured into the CUDA graph. The
// conditioning rows must already be staged in st.cond_in and st.cache.len == 0.
void run_frame_body(const QwenTtsCodePredictor& cp, CpFrameState& st,
                    bool sampling, float temperature, int top_k, float top_p,
                    std::uint64_t key, std::uint64_t* counter) {
    const bt::Device dev = cp.final_norm.device;
    const int n_out = cp.num_code_groups - 1;   // 15
    const int hidden = cp.hidden;

    auto code_slot = [&](int j) {                // device INT32 view (1,1)
        return bt::Tensor::view(
            dev, static_cast<std::int32_t*>(st.code_dev.data) + j, 1, 1, bt::Dtype::INT32);
    };
    // lm_head[head_idx] over one (1,hidden) row -> winning code id, written as a
    // device INT32 into code_dev[j]; the returned view is the next step's gather
    // index. Greedy argmax writes in place; sampling copies the seeded draw in.
    auto emit = [&](const bt::Tensor& row, int head_idx, int j) -> bt::Tensor {
        qtd::linear(cp.lm_head[head_idx], nullptr, row, st.logits);   // (1, vocab)
        bt::Tensor slot = code_slot(j);
        if (sampling) {
            bt::Tensor idx;
            bt::sample_logits(st.logits, temperature, top_k, top_p, key, *counter, idx);
            ++(*counter);
            bt::copy_d2d(idx, 0, st.code_dev, j, 1);
        } else {
            bt::argmax_rows(st.logits, slot);                        // INT32 in place
        }
        return slot;
    };

    // prefill: two tokens [past_hidden, c0_embed] at depth positions 0, 1.
    const bt::Tensor& pre_in = cp_project_into(cp, st.cond_in, st.proj_pre);
    cp_run_into(cp, pre_in, 2, /*pos_start=*/0, st.cache, st.sc_pre, st.hs_pre, st.hidden_pre);
    // codebook 1 from lm_head[0] applied to the last prefill row (row 1).
    bt::Tensor row1 = bt::Tensor::view(
        dev, static_cast<float*>(st.hidden_pre.data) + hidden, 1, hidden, bt::Dtype::FP32);
    bt::Tensor idx_prev = emit(row1, 0, 0);

    // steps: emit codebooks 2..(num_code_groups-1).
    for (int j = 1; j < n_out; ++j) {
        // Embed the previous code straight from its device INT32 index, into the
        // persistent buffer (no fresh allocation — capture forbids it).
        bt::gather_rows(cp.codec_embedding[j - 1], idx_prev, st.erow);  // (1, talker_hidden)
        const bt::Tensor& step_in = cp_project_into(cp, st.erow, st.proj_step);
        cp_run_into(cp, step_in, 1, /*pos_start=*/1 + j, st.cache, st.sc_step,
                    st.hs_step, st.hidden_step);
        idx_prev = emit(st.hidden_step, j, j);
    }
}

}  // namespace

// Out-of-line so frame_state (unique_ptr<CpFrameState>) can be an incomplete
// type in the header — CpFrameState is only fully defined here.
QwenTtsCodePredictor::QwenTtsCodePredictor() = default;
QwenTtsCodePredictor::~QwenTtsCodePredictor() = default;
QwenTtsCodePredictor::QwenTtsCodePredictor(QwenTtsCodePredictor&&) noexcept = default;
QwenTtsCodePredictor&
QwenTtsCodePredictor::operator=(QwenTtsCodePredictor&&) noexcept = default;

void QwenTtsCodePredictor::load(const sf::File& f,
                                const QwenTtsCodePredictorConfig& cfg,
                                int talker_hidden_, bt::Device dev,
                                bool bf16_weights) {
    frame_state.reset();   // drop any stale per-instance decode state on reload
    num_layers      = cfg.transformer.num_hidden_layers;
    hidden          = cfg.transformer.hidden_size;
    talker_hidden   = talker_hidden_;
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

    // The code embeddings live in the Talker hidden width; lm_head projects from
    // the depth-transformer hidden width to the per-codebook vocab.
    codec_embedding.clear();
    lm_head.clear();
    for (int j = 0; j < n_tables; ++j) {
        codec_embedding.push_back(
            up(f, P + "model.codec_embedding." + std::to_string(j) + ".weight", vocab, talker_hidden, dev));
        lm_head.push_back(qtd::narrow_bf16(
            up(f, P + "lm_head." + std::to_string(j) + ".weight", vocab, hidden, dev),
            bf16_weights));
    }

    // small_to_mtp_projection (Linear talker_hidden -> hidden + bias). Present
    // only when the widths differ (1.7B); an identity is omitted upstream (0.6B).
    has_mtp_proj = (talker_hidden != hidden);
    if (has_mtp_proj) {
        mtp_proj_w = qtd::narrow_bf16(
            up(f, P + "small_to_mtp_projection.weight", hidden, talker_hidden, dev),
            bf16_weights);
        mtp_proj_b = up_vec(f, P + "small_to_mtp_projection.bias", hidden, dev);
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
        cl.qw      = qtd::narrow_bf16(qtd::gather_rows(up(f, L + "self_attn.q_proj.weight", qd, hidden, dev), q_perm), bf16_weights);
        cl.kw      = qtd::narrow_bf16(qtd::gather_rows(up(f, L + "self_attn.k_proj.weight", kd, hidden, dev), k_perm), bf16_weights);
        cl.vw      = qtd::narrow_bf16(up(f, L + "self_attn.v_proj.weight", kd, hidden, dev), bf16_weights);
        cl.ow      = qtd::narrow_bf16(up(f, L + "self_attn.o_proj.weight", hidden, qd, dev), bf16_weights);
        cl.q_norm  = qtd::gather_rows(up_vec(f, L + "self_attn.q_norm.weight", head_dim, dev), hd_perm);
        cl.k_norm  = qtd::gather_rows(up_vec(f, L + "self_attn.k_norm.weight", head_dim, dev), hd_perm);
        cl.gate    = qtd::narrow_bf16(up(f, L + "mlp.gate_proj.weight", intermediate, hidden, dev), bf16_weights);
        cl.up      = qtd::narrow_bf16(up(f, L + "mlp.up_proj.weight", intermediate, hidden, dev), bf16_weights);
        cl.down    = qtd::narrow_bf16(up(f, L + "mlp.down_proj.weight", hidden, intermediate, dev), bf16_weights);
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
    bt::Tensor ph = bt::Tensor::from_host_on(dev, past_hidden, 1, talker_hidden);
    bt::Tensor ce = bt::Tensor::from_host_on(dev, c0_embed,    1, talker_hidden);
    predict_dev(ph, ce, out_codes);
}

void QwenTtsCodePredictor::predict_dev(const bt::Tensor& past_hidden,
                                       const bt::Tensor& c0_embed,
                                       std::vector<int>& out_codes,
                                       float temperature, int top_k, float top_p,
                                       std::uint64_t key,
                                       std::uint64_t* counter) const {
    const bt::Device dev = final_norm.device;
    bt::DeviceScope scope(dev);
    const int n_out = num_code_groups - 1;   // 15
    out_codes.assign(n_out, 0);

    const bool prof = CpProf::on();
    const bool sampling = temperature > 0.0f && counter != nullptr;
    using clk = std::chrono::steady_clock;
    auto dur = [](clk::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };

    // Lazily build the per-instance persistent state: the depth KV cache, the
    // conditioning-input staging buffer, and the (15,1) INT32 code buffer. The
    // pass scratch sizes itself on the first run_frame_body call.
    if (!frame_state) {
        frame_state = std::make_unique<CpFrameState>();
        const int kd = n_kv_heads * head_dim;
        frame_state->cache.alloc(num_layers, num_code_groups, dev, kd);
        frame_state->cond_in  = bt::Tensor::empty_on(dev, 2, talker_hidden, bt::Dtype::FP32);
        frame_state->code_dev = bt::Tensor::empty_on(dev, n_out, 1, bt::Dtype::INT32);
    }
    CpFrameState& st = *frame_state;

    // Stage the two conditioning rows into the persistent input buffer in place —
    // the only per-frame input a captured graph reads (everything downstream is
    // deterministic device compute over st's buffers).
    bt::copy_d2d(past_hidden, 0, st.cond_in, 0,             talker_hidden);
    bt::copy_d2d(c0_embed,    0, st.cond_in, talker_hidden, talker_hidden);

    if (prof) { bt::sync_all(); ++CpProf::calls(); }
    clk::time_point t0;
    if (prof) t0 = clk::now();

#ifdef BROSOUNDML_HAS_CUDA
    // CUDA greedy: the whole frame is a fixed-shape sequence of device ops with
    // no host control flow, so capture it once and replay it as a single launch
    // (~700 tiny kernel launches/frame -> one cudaGraphLaunch). Sampling is NOT
    // captured: its Philox counter must advance on the host per code per frame.
    // BROSOUNDML_QWEN_NO_GRAPH forces the eager path (A/B + escape hatch).
    static const bool no_graph = std::getenv("BROSOUNDML_QWEN_NO_GRAPH") != nullptr;
    const bool use_graph = (dev == bt::Device::CUDA) && !sampling && !no_graph;
    if (use_graph) {
        CpProf::graph() = true;
        if (!st.captured) {
            // Warm-up: an eager run allocates + sizes every scratch buffer (the
            // capture itself must allocate nothing). It also yields this frame's
            // codes — capture re-runs the identical compute, so they agree.
            st.cache.len = 0;
            run_frame_body(*this, st, /*sampling=*/false, 0.0f, 0, 1.0f, 0, nullptr);
            bt::sync_all();
            st.cache.len = 0;   // re-bake offsets 0,1,2,... into the graph
            {
                bt::CudaGraphCapture cap;
                run_frame_body(*this, st, /*sampling=*/false, 0.0f, 0, 1.0f, 0, nullptr);
                st.graph = cap.finish();
            }
            st.captured = true;
        } else {
            st.graph.launch();   // replay: reads st.cond_in, writes st.code_dev
        }
        bt::sync(bt::Device::CUDA);
    } else
#endif
    {
        // Eager path (CPU, or any sampling frame): the same body over the same
        // persistent buffers, run directly. cache.len resets per frame.
        st.cache.len = 0;
        run_frame_body(*this, st, sampling, temperature, top_k, top_p, key, counter);
        if (prof) bt::sync_all();
    }
    if (prof) CpProf::frame() += dur(clk::now() - t0);

    // Single batched read-back of the whole (15,1) INT32 code vector.
    clk::time_point trb;
    if (prof) trb = clk::now();
    bt::Tensor codes_host = (dev == bt::Device::CPU) ? st.code_dev
                                                     : st.code_dev.to(bt::Device::CPU);
    const std::int32_t* codes = static_cast<const std::int32_t*>(codes_host.host_raw());
    for (int j = 0; j < n_out; ++j) out_codes[j] = codes[j];
    if (prof) CpProf::readback() += dur(clk::now() - trb);
}

// Print + reset the Code Predictor internal profile (called from generate_codes).
void qwen_cp_profile_report() {
    if (!CpProf::on() || CpProf::calls() == 0) return;
    const double f = static_cast<double>(CpProf::calls());
    std::fprintf(stderr,
        "  [code_predictor internals over %ld frames, mode=%s]\n"
        "    frame compute        %9.1f ms  %7.3f ms/frame\n"
        "    code read-back (1xD2H)%8.1f ms  %7.3f ms/frame\n",
        CpProf::calls(), CpProf::graph() ? "graph-replay" : "eager",
        CpProf::frame(),    CpProf::frame() / f,
        CpProf::readback(), CpProf::readback() / f);
    CpProf::frame() = CpProf::readback() = 0;
    CpProf::calls() = 0;
    CpProf::graph() = false;
}

const float* QwenTtsCodePredictor::codec_embedding_row(int table, int id) const {
    // Host pointer into the table — valid only when CPU-resident (host AR path).
    // The code embeddings are talker_hidden-wide (the Talker hidden space).
    return codec_embedding[table].host_f32() + static_cast<std::size_t>(id) * talker_hidden;
}

}  // namespace brosoundml
