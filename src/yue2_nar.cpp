// YuE2 backbone: the NAR (flow-matching) path — see yue2_model.h.
//
// nar.CachedNAR in brotensor ops: the chunk's AR tokens are prefilled once on
// the AR path (or copied from a semantic session's cache) and every layer's
// K/V kept; a velocity call embeds the padded ODE state (zero LATENT_START /
// LATENT_END rows around the frames) through vae2llm + time embedding +
// sinusoidal frame position, runs the NAR path with each layer's queries over
// [AR K/V | NAR K/V] (bidirectional, RoPE positions continuing after the AR
// rows), and reads llm2vae(norm(x)) at the frame rows.

#include "yue2_model_impl.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace brosoundml::yue2 {

namespace {

float bf16r(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

// Upload host FP32 rows into rows [row0, row0 + rows) of a device tensor at
// its dtype (rounding to BF16 for a BF16 tensor).
void upload_rows(const ModelWeights& w, const float* src, int rows, bt::Tensor& dst, int row0) {
    bt::Tensor f = bt::Tensor::from_host_on(w.dev, src, rows, dst.cols);
    bt::Tensor view = rows_view(dst, row0, rows);
    if (dst.dtype == bt::Dtype::FP32) {
        bt::copy_d2d(f, 0, view, 0, rows * dst.cols);
    } else {
        bt::Tensor c;
        bt::cast(f, c, dst.dtype);
        bt::copy_d2d(c, 0, view, 0, rows * dst.cols);
    }
}

// _shift_t_value: sigmoid of the raw t in the model dtype, then the timestep
// shift, each op rounded like the reference's BF16 tensors.
float shifted_t(const ModelWeights& w, double raw) {
    const bool b = w.dt != bt::Dtype::FP32;
    auto R = [&](float v) { return b ? bf16r(v) : v; };
    const float t = R(static_cast<float>(raw));
    const float sig = R(1.0f / (1.0f + std::exp(-t)));
    const float shift = w.cfg.timestep_shift;
    if (shift == 1.0f) return sig;
    return R(R(shift * sig) / R(1.0f + R((shift - 1.0f) * sig)));
}

void layers(const ModelWeights& w, NarChunk& c, const bt::Tensor& xin, bool capture);

void velocity(const ModelWeights& w, NarChunk& c, const bt::Tensor& xin) {
    // Time embedding (TimestepEmbedder): FP32 sinusoid on the host, cast; the
    // MLP runs with the layers.
    {
        const float ts = shifted_t(w, c.cur_raw);
        constexpr int half = 128;
        float emb[2 * half];
        const float k = static_cast<float>(-9.210340371976184);  // -log(10000)
        for (int i = 0; i < half; ++i) {
            const float f = std::exp((k * static_cast<float>(i)) / static_cast<float>(half));
            const float a = ts * f;
            emb[i] = std::cos(a);
            emb[half + i] = std::sin(a);
        }
        if (c.temb_in.rows != 1) c.temb_in = bt::Tensor::zeros_on(w.dev, 1, 2 * half, w.dt);
        upload_rows(w, emb, 1, c.temb_in, 0);
    }
    const bool capture =
        c.hooks && std::find(c.capture_evals.begin(), c.capture_evals.end(), c.eval) != c.capture_evals.end();
    // From here on every op reads and writes the chunk's fixed buffers: one
    // CUDA graph per input buffer (the state and the midpoint) replays the
    // ~600 launches of an evaluation as one. An evaluation that captures
    // activations to the host runs eagerly.
    static const bool no_graph = [] {
        const char* v = std::getenv("BROSOUNDML_YUE2_NO_GRAPH");
        return v && *v && *v != '0';
    }();
    const int slot = xin.data == c.xpad.data ? 0 : 1;
    if (!no_graph && !capture && bt::graph_capture_available(w.dev)) {
        if (c.graphed[slot]) {
            c.graph[slot].launch();
        } else {
            layers(w, c, xin, false);  // warm-up: sizes every buffer and computes this evaluation
            bt::sync_all();
            bt::CudaGraphCapture cap;
            layers(w, c, xin, false);
            c.graph[slot] = cap.finish();
            c.graphed[slot] = true;
        }
    } else {
        layers(w, c, xin, capture);
    }
    ++c.eval;
}

void layers(const ModelWeights& w, NarChunk& c, const bt::Tensor& xin, bool capture) {
    const Config& cf = w.cfg;
    const int hd = cf.head_dim, qd = cf.n_q * hd, kd = cf.n_kv * hd, N = c.N;
    RowScratch& sc = c.sc;
    lin(w.t0_w, &w.t0_b, c.temb_in, c.temb_h, &sc.ws);
    bt::silu_forward(c.temb_h, c.temb_h);
    lin(w.t2_w, &w.t2_b, c.temb_h, c.temb, &sc.ws);
    lin(w.vae2llm_w, &w.vae2llm_b, xin, sc.x, &sc.ws);
    bt::add_row_bias_inplace(sc.x, c.temb);
    bt::add_inplace(sc.x, rows_view(w.pe, 0, N));

    for (int l = 0; l < cf.layers; ++l) {
        const PathW& p = w.layers[l].nar;
        bt::rms_norm_forward(sc.x, p.in_ln, cf.eps, sc.n);
        lin(p.q, nullptr, sc.n, sc.q, &sc.ws);
        lin(p.k, nullptr, sc.n, sc.k, &sc.ws);
        lin(p.v, nullptr, sc.n, sc.v, &sc.ws);
        head_norm(w, sc.q, cf.n_q, p.q_norm, sc.qn);
        head_norm(w, sc.k, cf.n_kv, p.k_norm, sc.kn);
        bt::Tensor qv = reshape_view(sc.qn, N, qd), kv = reshape_view(sc.kn, N, kd);
        bt::rope_apply(qv, c.cos, c.sin, hd, cf.n_q, qv);
        bt::rope_apply(kv, c.cos, c.sin, hd, cf.n_kv, kv);
        bt::copy_d2d(kv, 0, c.K[l], c.ar_len * kd, N * kd);
        bt::copy_d2d(sc.v, 0, c.V[l], c.ar_len * kd, N * kd);
        bt::flash_attention_gqa_forward(qv, c.K[l], c.V[l], nullptr, cf.n_q, cf.n_kv, false, sc.att);
        lin(p.o, nullptr, sc.att, sc.x, &sc.ws, true);
        bt::rms_norm_forward(sc.x, p.post_ln, cf.eps, sc.n);
        bt::linear_forward_batched_ex(p.gate_up, nullptr, sc.n, bt::kLinearActNone, bt::kLinearEpiSwiglu, &sc.ws,
                                      sc.g);
        lin(p.down, nullptr, sc.g, sc.x, &sc.ws, true);
        apply_hooks(w, c.hooks, l, sc.x, c.frame0 - 1, 1, N - 1, c.eval, capture);
    }
    bt::rms_norm_forward(sc.x, w.final_norm, cf.eps, sc.n);
    lin(w.llm2vae_w, &w.llm2vae_b, rows_view(sc.n, 1, c.frames), c.vel, &sc.ws);
}

double raw_t(double t) {
    const double v = std::log(t / (1.0 - t));
    return std::clamp(v, -20.0, 20.0);
}

}  // namespace

std::unique_ptr<NarChunk> Model::nar_chunk(const std::vector<int32_t>& ar, int frames,
                                           const NarOptions& opt) const {
    if (!loaded()) model_fail("not loaded");
    const ModelWeights& w = *w_;
    const Config& cf = w.cfg;
    bt::DeviceScope scope(w.dev);
    const int ar_len = static_cast<int>(ar.size()), N = frames + 2;
    if (ar_len < 1 || frames < 1) model_fail("NAR chunk needs AR tokens and frames");
    if (ar_len + N > cf.max_pos) model_fail("acoustic chunk exceeds the model context");
    if (N > cf.max_latent_frames) model_fail("acoustic chunk exceeds max_latent_frames");
    for (int32_t t : ar)
        if (t < 0 || t >= cf.vocab) model_fail("AR token outside the vocabulary");
    const int kd = cf.n_kv * cf.head_dim;
    auto c = std::make_unique<NarChunk>();
    c->ar_len = ar_len;
    c->frames = frames;
    c->N = N;
    c->hooks = opt.hooks;
    c->frame0 = opt.frame0;
    c->capture_evals = opt.capture_evals;
    if (opt.in_place) {
        ArSession& s = *opt.in_place;
        if (opt.reuse != opt.in_place || opt.reuse_rows != s.len[0] || opt.reuse_rows > ar_len)
            model_fail("an in-place chunk extends its session's branch 0 from its length");
        ar_reserve(s, ar_len + N);
        if (opt.reuse_rows < ar_len)
            ar_rows(w, c->sc, ar.data() + opt.reuse_rows, ar_len - opt.reuse_rows, opt.reuse_rows, s.K[0], s.V[0],
                    /*kv_only=*/true, opt.ar_hooks, /*capture=*/false);
        for (int l = 0; l < cf.layers; ++l) {
            c->K.push_back(rows_view(s.K[0][l], 0, ar_len + N));
            c->V.push_back(rows_view(s.V[0][l], 0, ar_len + N));
        }
        c->cos = rows_view(w.cos_tbl, ar_len, N);
        c->sin = rows_view(w.sin_tbl, ar_len, N);
        c->xpad = bt::Tensor::zeros_on(w.dev, N, cf.latent, w.dt);
        c->mpad = bt::Tensor::zeros_on(w.dev, N, cf.latent, w.dt);
        c->tmp = bt::Tensor::zeros_on(w.dev, frames, cf.latent, w.dt);
        return c;
    }
    for (int l = 0; l < cf.layers; ++l) {
        c->K.push_back(bt::Tensor::empty_on(w.dev, ar_len + N, kd, w.dt));
        c->V.push_back(bt::Tensor::empty_on(w.dev, ar_len + N, kd, w.dt));
    }
    int done = 0;
    if (opt.reuse && opt.reuse_rows > 0) {
        if (opt.reuse_rows > ar_len || opt.reuse_rows > opt.reuse->len[0])
            model_fail("reused rows exceed the session's cache");
        for (int l = 0; l < cf.layers; ++l) {
            bt::copy_d2d(opt.reuse->K[0][l], 0, c->K[l], 0, opt.reuse_rows * kd);
            bt::copy_d2d(opt.reuse->V[0][l], 0, c->V[l], 0, opt.reuse_rows * kd);
        }
        done = opt.reuse_rows;
    }
    if (done < ar_len)
        ar_rows(w, c->sc, ar.data() + done, ar_len - done, done, c->K, c->V, /*kv_only=*/true, opt.ar_hooks,
                /*capture=*/false);
    c->cos = rows_view(w.cos_tbl, ar_len, N);
    c->sin = rows_view(w.sin_tbl, ar_len, N);
    c->xpad = bt::Tensor::zeros_on(w.dev, N, cf.latent, w.dt);
    c->mpad = bt::Tensor::zeros_on(w.dev, N, cf.latent, w.dt);
    c->tmp = bt::Tensor::zeros_on(w.dev, frames, cf.latent, w.dt);
    return c;
}

std::vector<float> Model::nar_velocity(NarChunk& c, const std::vector<float>& x, double raw) const {
    const ModelWeights& w = *w_;
    bt::DeviceScope scope(w.dev);
    if (x.size() != static_cast<std::size_t>(c.frames) * w.cfg.latent) model_fail("NAR state shape");
    upload_rows(w, x.data(), c.frames, c.xpad, 1);
    c.cur_raw = raw;
    velocity(w, c, c.xpad);
    std::vector<float> out;
    to_host_f32(w, c.vel, out);
    return out;
}

std::vector<float> Model::nar_solve(NarChunk& c, const float* noise, int steps, const CancelCheck& cancel,
                                    const std::function<void(int)>& on_step, const float* known,
                                    int known_frames) const {
    const ModelWeights& w = *w_;
    bt::DeviceScope scope(w.dev);
    if (steps < 1) model_fail("ode_steps must be positive");
    const int F = c.frames, D = w.cfg.latent;
    if (known_frames < 0 || known_frames > F || (known_frames > 0 && !known)) model_fail("bad inpainting span");
    upload_rows(w, noise, F, c.xpad, 1);
    bt::Tensor S = rows_view(c.xpad, 1, F), M = rows_view(c.mpad, 1, F);
    // Inpainting (lab.nar's clamp): the known frames are replaced by the
    // exact flow interpolant before every evaluation, in BF16 ops as torch
    // does them on the BF16 state: bf(bf(t * n) + bf((1 - t) * k)).
    const bool bf = w.dt != bt::Dtype::FP32;
    auto R = [&](float v) { return bf ? bf16r(v) : v; };
    std::vector<float> kn, kk, kv;
    if (known_frames > 0) {
        const std::size_t n = static_cast<std::size_t>(known_frames) * D;
        kn.resize(n);
        kk.resize(n);
        kv.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            kn[i] = R(noise[i]);
            kk[i] = R(known[i]);
        }
    }
    auto clamp = [&](bt::Tensor& pad, double t) {
        if (known_frames == 0) return;
        const float tf = static_cast<float>(t), sf = static_cast<float>(1.0 - t);
        for (std::size_t i = 0; i < kv.size(); ++i) kv[i] = R(R(tf * kn[i]) + R(sf * kk[i]));
        upload_rows(w, kv.data(), known_frames, pad, 1);
    };
    const double dt = 1.0 / steps;
    for (int step = 0; step < steps; ++step) {
        if (cancel && cancel()) return {};
        const double t = 1.0 - step * dt;
        clamp(c.xpad, t);
        c.cur_raw = raw_t(t);
        velocity(w, c, c.xpad);
        // mid = state - v * (dt / 2), each op rounded to the state dtype.
        bt::copy_d2d(c.vel, 0, c.tmp, 0, F * D);
        bt::scale_inplace(c.tmp, static_cast<float>(-(dt / 2)));
        bt::copy_d2d(S, 0, M, 0, F * D);
        bt::add_inplace(M, c.tmp);
        clamp(c.mpad, t - dt / 2);
        if (cancel && cancel()) return {};
        c.cur_raw = raw_t(t - dt / 2);
        velocity(w, c, c.mpad);
        bt::scale_inplace(c.vel, static_cast<float>(-dt));
        bt::add_inplace(S, c.vel);
        if (on_step) on_step(step + 1);
    }
    std::vector<float> out;
    to_host_f32(w, S, out);
    for (float v : out)
        if (!std::isfinite(v)) model_fail("acoustic flow matching produced non-finite latents");
    for (std::size_t i = 0; i < kk.size(); ++i) out[i] = kk[i];   // clamp(state, 0)
    return out;
}

}  // namespace brosoundml::yue2
