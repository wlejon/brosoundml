// YuE2 backbone: the AR decode step (R tokens per branch, CUDA graph per R) —
// see yue2_model.h.

#include "yue2_model_impl.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace brosoundml::yue2 {

void Model::ar_step(ArSession& s, int32_t token) const { ar_step_rows(s, &token, 1); }

void Model::ar_truncate(ArSession& s, int b, int len) const {
    if (b < 0 || b >= s.B || len < 1 || len > s.len[b]) model_fail("truncate outside the branch");
    s.len[b] = len;
    if (b == 0) s.watch_rows = std::min(s.watch_rows, len);   // the rows past it will be rewritten
}

void Model::ar_watch(ArSession& s, int layer, int head) const {
    const Config& c = w_->cfg;
    if (layer < 0 || layer >= c.layers || head < 0 || head >= c.n_q) model_fail("watched head outside the model");
    s.watch_layer = layer;
    s.watch_head = head;
    s.watch_k.clear();
    s.watch_rows = 0;
    s.steps.clear();   // the step buffers and graphs gain the query copy
}

void Model::ar_watched(ArSession& s, const std::vector<std::pair<int, int>>& spans, std::vector<float>& out) const {
    const ModelWeights& w = *w_;
    const Config& c = w.cfg;
    bt::DeviceScope scope(w.dev);
    if (s.watch_layer < 0) model_fail("ar_watched without ar_watch");
    const int R = s.last_R;
    auto it = s.steps.find(R);
    if (R < 1 || it == s.steps.end() || it->second.wq.rows != R) model_fail("ar_watched before a watched step");
    const int hd = c.head_dim, kd = c.n_kv * hd, qd = c.n_q * hd;
    const int kvh = s.watch_head / (c.n_q / c.n_kv);   // GQA: query heads share their KV head in groups
    const int len = s.len[0];
    // The key rows written since the last read (a prefill, steps).
    if (s.watch_rows < len) {
        std::vector<float> rows;
        to_host_f32(w, rows_view(s.K[0][s.watch_layer], s.watch_rows, len - s.watch_rows), rows);
        s.watch_k.resize(static_cast<std::size_t>(len) * hd);
        for (int r = 0; r < len - s.watch_rows; ++r)
            std::copy_n(rows.data() + static_cast<std::size_t>(r) * kd + kvh * hd, hd,
                        s.watch_k.data() + static_cast<std::size_t>(s.watch_rows + r) * hd);
        s.watch_rows = len;
    }
    std::vector<float> q;
    to_host_f32(w, it->second.wq, q);
    const std::size_t S = spans.size();
    out.assign(static_cast<std::size_t>(R) * S, 0.0f);
    const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
    std::vector<double> e(static_cast<std::size_t>(len));
    for (int r = 0; r < R; ++r) {
        const int n = len - R + r + 1;   // row r sees the keys up to its own
        const float* qh = q.data() + static_cast<std::size_t>(r) * qd + static_cast<std::size_t>(s.watch_head) * hd;
        double mx = -1e300;
        for (int j = 0; j < n; ++j) {
            const float* kj = s.watch_k.data() + static_cast<std::size_t>(j) * hd;
            float d = 0;
            for (int i = 0; i < hd; ++i) d += qh[i] * kj[i];
            e[j] = d * scale;
            mx = std::max(mx, e[j]);
        }
        double z = 0;
        for (int j = 0; j < n; ++j) z += (e[j] = std::exp(e[j] - mx));
        for (std::size_t k = 0; k < S; ++k) {
            double m = 0;
            for (int j = std::max(0, spans[k].first); j < std::min(n, spans[k].second); ++j) m += e[j];
            out[static_cast<std::size_t>(r) * S + k] = static_cast<float>(m / z);
        }
    }
}

void Model::ar_step_rows(ArSession& s, const int32_t* tokens, int R) const {
    const ModelWeights& w = *w_;
    const Config& c = w.cfg;
    bt::DeviceScope scope(w.dev);
    if (R < 1 || R > 8) model_fail("a decode step takes 1 to 8 tokens");
    for (int r = 0; r < R; ++r)
        if (tokens[r] < 0 || tokens[r] >= c.vocab) model_fail("token id outside the vocabulary");
    if (s.hi <= s.lo) model_fail("step before prefill");
    const int B = s.B, M = B * R, hd = c.head_dim, half = hd / 2, qd = c.n_q * hd;
    for (int b = 0; b < B; ++b) {
        if (s.len[b] < 1) model_fail("step before prefill");
        ar_reserve(s, s.len[b] + R);   // grows on demand (drops the graphs)
    }
    // CUDA reads each branch's key count on the device (the work divides the
    // valid keys, not the session capacity); the CPU reads the same length
    // from host memory; other devices take a key mask (one token a step).
    const bool dev_len = w.dev.type == bt::DeviceType::CUDA || w.dev.is_cpu();
    if (R > 1 && !dev_len) model_fail("multi-token decode needs CUDA or the CPU");
    auto& st = s.steps[R];
    if (st.rowsb.empty()) {
        st.tok = bt::Tensor::empty_on(w.dev, M, 1, bt::Dtype::INT32);
        st.rows = bt::Tensor::empty_on(w.dev, M, 1, bt::Dtype::INT32);
        st.lens = bt::Tensor::empty_on(w.dev, B, 1, bt::Dtype::INT32);
        for (int b = 0; b < B; ++b) {
            st.rowsb.push_back(rows_view(st.rows, b * R, R));
            if (!dev_len) st.mask.push_back(bt::Tensor::zeros_on(w.dev, s.cap, 1, bt::Dtype::FP32));
        }
        st.mask_len.assign(B, 0);
        const float one = 1.0f;
        st.ones = bt::Tensor::from_host_on(w.dev, &one, 1, 1);
        st.cos = bt::Tensor::zeros_on(w.dev, M, half, bt::Dtype::FP32);
        st.sin = bt::Tensor::zeros_on(w.dev, M, half, bt::Dtype::FP32);
        st.att = bt::Tensor::zeros_on(w.dev, M, qd, w.dt);
        if (s.watch_layer >= 0) st.wq = bt::Tensor::zeros_on(w.dev, R, qd, w.dt);
    }
    // Stage this step: tokens, cache rows, each branch's key count after the
    // step (the last row's; row r sees len + r + 1 keys), RoPE rows.
    std::vector<int32_t> toks(M), rows(M), lens(B);
    for (int b = 0; b < B; ++b) {
        for (int r = 0; r < R; ++r) {
            toks[b * R + r] = tokens[r];
            rows[b * R + r] = s.len[b] + r;
        }
        lens[b] = s.len[b] + R;
    }
    h2d(w.dev, st.tok.data, toks.data(), M * sizeof(int32_t));
    h2d(w.dev, st.rows.data, rows.data(), M * sizeof(int32_t));
    h2d(w.dev, st.lens.data, lens.data(), B * sizeof(int32_t));
    for (int b = 0; b < B; ++b) {
        const int pos = s.len[b];
        bt::copy_d2d(w.cos_tbl, pos * half, st.cos, b * R * half, R * half);
        bt::copy_d2d(w.sin_tbl, pos * half, st.sin, b * R * half, R * half);
        if (dev_len) continue;
        if (st.mask_len[b] != pos) {
            std::vector<float> m(s.cap, 0.0f);
            std::fill(m.begin(), m.begin() + pos, 1.0f);
            h2d(w.dev, st.mask[b].data, m.data(), m.size() * sizeof(float));
        }
        bt::copy_d2d(st.ones, 0, st.mask[b], pos, 1);
    }

    const bt::Tensor lm = lm_rows(w, s.lo, s.hi);
    const QuantW qlm = q_lm_rows(w, s.lo, s.hi);
    auto body = [&]() {
        const int kd = c.n_kv * hd;
        bt::embedding_lookup_forward(w.embed, static_cast<const int32_t*>(st.tok.data), M, st.x);
        bt::rms_norm_forward(st.x, w.layers[0].ar.in_ln, c.eps, st.n);
        for (int l = 0; l < c.layers; ++l) {
            const PathW& p = w.layers[l].ar;
            dlin(p.q, p.qq, st.n, st.q, &st.ws);
            dlin(p.k, p.qk, st.n, st.k, &st.ws);
            dlin(p.v, p.qv, st.n, st.v, &st.ws);
            head_norm(w, st.q, c.n_q, p.q_norm, st.qn);
            head_norm(w, st.k, c.n_kv, p.k_norm, st.kn);
            bt::Tensor qv = reshape_view(st.qn, M, qd), kv = reshape_view(st.kn, M, kd);
            bt::rope_apply(qv, st.cos, st.sin, hd, c.n_q, qv);
            bt::rope_apply(kv, st.cos, st.sin, hd, c.n_kv, kv);
            if (l == s.watch_layer) bt::copy_d2d(qv, 0, st.wq, 0, R * qd);   // branch 0's rows lead
            for (int b = 0; b < B; ++b) {
                bt::scatter_rows(rows_view(kv, b * R, R), st.rowsb[b], s.K[b][l]);
                bt::scatter_rows(rows_view(st.v, b * R, R), st.rowsb[b], s.V[b][l]);
                bt::Tensor ab = rows_view(st.att, b * R, R);
                if (dev_len)
                    bt::flash_attention_decode_len(rows_view(qv, b * R, R), s.K[b][l], s.V[b][l],
                                                   static_cast<const int32_t*>(st.lens.data) + b, c.n_q, c.n_kv,
                                                   ab);
                else
                    bt::flash_attention_decode_masked(rows_view(qv, b, 1), s.K[b][l], s.V[b][l],
                                                      static_cast<const float*>(st.mask[b].data), c.n_q, c.n_kv,
                                                      ab);
            }
            dlin(p.o, p.qo, st.att, st.x, &st.ws, bt::kLinearEpiAccumulate);
            bt::rms_norm_forward(st.x, p.post_ln, c.eps, st.n);
            // silu(gate) * up in the GEMV's FP32 epilogue (one BF16 rounding
            // where the reference rounds gate, silu and the product).
            dlin(p.gate_up, p.qgate_up, st.n, st.g, &st.ws, bt::kLinearEpiSwiglu);
            dlin(p.down, p.qdown, st.g, st.x, &st.ws, bt::kLinearEpiAccumulate);
            if (s.hooks)
                for (int b = 0; b < B; ++b) {
                    bt::Tensor xb = rows_view(st.x, b * R, R);
                    apply_hooks(w, s.hooks, l, xb, s.len[b], 0, R, -1, b == 0);
                }
            const bt::Tensor& next = l + 1 < c.layers ? w.layers[l + 1].ar.in_ln : w.final_norm;
            bt::rms_norm_forward(st.x, next, c.eps, st.n);
        }
        dlin(lm, qlm, st.n, st.logits, &st.ws);
    };

    static const bool no_graph = [] {
        const char* v = std::getenv("BROSOUNDML_YUE2_NO_GRAPH");
        return v && *v && *v != '0';
    }();
    const bool hooks_eager = s.hooks && (s.hooks->any_capture() || s.hooks->any_per_position());
    const bool graph = !no_graph && !hooks_eager && bt::graph_capture_available(w.dev);
    if (!graph) {
        body();
    } else if (!st.captured || st.graph_lo != s.lo || st.graph_hi != s.hi) {
        body();  // warm-up: sizes every buffer and computes this step
        bt::sync_all();
        {
            bt::CudaGraphCapture cap;
            body();
            st.graph = cap.finish();
        }
        st.captured = true;
        st.graph_lo = s.lo;
        st.graph_hi = s.hi;
    } else {
        st.graph.launch();
    }
    if (s.logits.rows != M || s.logits.cols != s.hi - s.lo)
        s.logits = bt::Tensor::zeros_on(w.dev, M, s.hi - s.lo, w.dt);
    bt::copy_d2d(st.logits, 0, s.logits, 0, M * (s.hi - s.lo));
    for (int b = 0; b < B; ++b) {
        st.mask_len[b] = s.len[b] + R;
        s.len[b] += R;
    }
    s.last_R = R;
}

}  // namespace brosoundml::yue2
