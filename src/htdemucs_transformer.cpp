// HTDemucs cross transformer (demucs transformer.py, CrossTransformerEncoder)
// and the per-batch constants. See htdemucs_model.h.
//
// Both streams run as token rows (B*L, 512): the spectral stream's tokens are
// ordered (t, f) like the reference's "b c fr t1 -> b (t1 fr) c", the temporal
// stream's are its time steps. Layers alternate self (even) and cross (odd);
// attention is per segment (varlen over B sequences), 8 heads of 64.
#include "htdemucs_model.h"

#include <brotensor/ops.h>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace brosoundml::htd {

namespace {

bt::Tensor as(const bt::Tensor& t, int r, int c) {
    if (static_cast<long long>(r) * c != static_cast<long long>(t.rows) * t.cols)
        fail("internal: reshape element count mismatch");
    return bt::Tensor::view(t.device, t.data, r, c, t.dtype);
}

void layernorm(const bt::Tensor& X, const LayerNorm& n, bt::Tensor& Y) {
    bt::layernorm_forward_inference_batched(X, n.gamma, n.beta, Y, 1e-5f);
}

void linear(const Linear& l, const bt::Tensor& X, bt::Tensor& Y) {
    bt::linear_forward_batched_ex(l.w, &l.b, X, bt::kLinearActNone, bt::kLinearEpiStore, nullptr, Y);
}

// create_2d_sin_embedding(D, Fr, T1) rearranged to tokens (t1, fr): (T1*Fr, D).
std::vector<float> pos_2d(int D, int Fr, int T1, float max_period) {
    const int half = D / 2;
    std::vector<float> div(half / 2);
    const float k = static_cast<float>(-(std::log(static_cast<double>(max_period)) / half));
    for (int i = 0; i < half / 2; ++i) div[i] = std::exp(static_cast<float>(2 * i) * k);
    std::vector<float> pe(static_cast<std::size_t>(T1) * Fr * D);
    for (int t = 0; t < T1; ++t)
        for (int f = 0; f < Fr; ++f) {
            float* row = pe.data() + (static_cast<std::size_t>(t) * Fr + f) * D;
            for (int i = 0; i < half / 2; ++i) {
                const float pw = static_cast<float>(t) * div[i];
                const float ph = static_cast<float>(f) * div[i];
                row[2 * i] = std::sin(pw);
                row[2 * i + 1] = std::cos(pw);
                row[half + 2 * i] = std::sin(ph);
                row[half + 2 * i + 1] = std::cos(ph);
            }
        }
    return pe;
}

// create_sin_embedding(T, D): (T, D), [cos(phase) | sin(phase)].
std::vector<float> pos_1d(int T, int D, float max_period) {
    const int half = D / 2;
    std::vector<float> pe(static_cast<std::size_t>(T) * D);
    std::vector<float> den(half);
    for (int a = 0; a < half; ++a)
        den[a] = std::pow(max_period, static_cast<float>(a) / static_cast<float>(half - 1));
    for (int t = 0; t < T; ++t) {
        float* row = pe.data() + static_cast<std::size_t>(t) * D;
        for (int a = 0; a < half; ++a) {
            const float ph = static_cast<float>(t) / den[a];
            row[a] = std::cos(ph);
            row[half + a] = std::sin(ph);
        }
    }
    return pe;
}

std::vector<float> repeat(const std::vector<float>& v, int times) {
    std::vector<float> out;
    out.reserve(v.size() * times);
    for (int i = 0; i < times; ++i) out.insert(out.end(), v.begin(), v.end());
    return out;
}

}  // namespace

// ── per-batch constants ──────────────────────────────────────────────────────

const BatchConsts& Net::consts(int B) const {
    std::lock_guard<std::mutex> lock(consts_mu_);
    auto it = consts_.find(B);
    if (it != consts_.end()) return *it->second;
    auto bc = std::make_unique<BatchConsts>();
    bc->B = B;
    const int A = cfg_.audio_channels, S = cfg_.sources, F = cfg_.nfft / 2, bins = F + 1;
    const int hl = cfg_.nfft / 4, T1 = (cfg_.segment_samples + hl - 1) / hl;
    int Fr = F, T2 = cfg_.segment_samples;
    for (int i = 0; i < cfg_.depth; ++i) {
        Fr /= cfg_.stride;
        T2 = (T2 + cfg_.stride - 1) / cfg_.stride;
    }
    {
        std::vector<int> idx(static_cast<std::size_t>(B) * A * 2 * F);
        for (int s = 0; s < B * A; ++s)
            for (int ri = 0; ri < 2; ++ri)
                for (int k = 0; k < F; ++k)
                    idx[(static_cast<std::size_t>(s) * 2 + ri) * F + k] = s * 2 * bins + 2 * k + ri;
        bc->spec_idx = int_tensor(dev_, idx);
    }
    {
        const int n = B * S * A;
        std::vector<int> idx(static_cast<std::size_t>(n) * F * 2);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < F; ++k)
                for (int ri = 0; ri < 2; ++ri)
                    idx[(static_cast<std::size_t>(i) * F + k) * 2 + ri] = (i * 2 + ri) * F + k;
        bc->ispec_idx = int_tensor(dev_, idx);
    }
    {
        std::vector<float> fe = freq_emb_.to(bt::Device::CPU).to_host_vector();
        std::vector<float> rep = repeat(fe, B);
        bc->freq_emb = bt::Tensor::from_host_on(dev_, rep.data(), static_cast<int>(rep.size()), 1);
    }
    const int D = cfg_.bottom_channels;
    {
        std::vector<float> p = repeat(pos_2d(D, Fr, T1, cfg_.t_max_period), B);
        bc->pos_x = bt::Tensor::from_host_on(dev_, p.data(), B * T1 * Fr, D);
        std::vector<float> q = repeat(pos_1d(T2, D, cfg_.t_max_period), B);
        bc->pos_t = bt::Tensor::from_host_on(dev_, q.data(), B * T2, D);
    }
    {
        std::vector<int> cx(B + 1), ct(B + 1);
        for (int b = 0; b <= B; ++b) {
            cx[b] = b * T1 * Fr;
            ct[b] = b * T2;
        }
        bc->cu_x = int_tensor(dev_, cx);
        bc->cu_t = int_tensor(dev_, ct);
    }
    return *consts_.emplace(B, std::move(bc)).first->second;
}

// ── layers ───────────────────────────────────────────────────────────────────

void Net::tlayer(const TLayer& L, bt::Tensor& x, const bt::Tensor* kv, int B, int Lq, int Lk,
                 const BatchConsts& bc, bool time_stream) const {
    const int D = L.q.out, H = cfg_.t_heads;
    bt::Tensor n1, q, k, v, a;
    layernorm(x, L.norm1, n1);
    linear(L.q, n1, q);
    if (L.cross) {
        bt::Tensor n2;
        layernorm(*kv, L.norm2, n2);
        linear(L.k, n2, k);
        linear(L.v, n2, v);
    } else {
        linear(L.k, n1, k);
        linear(L.v, n1, v);
    }
    const bt::Tensor& cuq = time_stream ? bc.cu_t : bc.cu_x;
    const bt::Tensor& cuk = L.cross ? (time_stream ? bc.cu_x : bc.cu_t) : cuq;
    bt::flash_attention_varlen_forward(q, k, v, static_cast<const std::int32_t*>(cuq.data),
                                       static_cast<const std::int32_t*>(cuk.data), B, Lq, Lk, H, D / H,
                                       /*causal*/ false, a);
    bt::linear_forward_batched_ex(L.o.w, &L.o.b, a, bt::kLinearActNone, bt::kLinearEpiAccumulate, nullptr, x);

    bt::Tensor n3, hh;
    layernorm(x, L.cross ? L.norm3 : L.norm2, n3);
    bt::linear_forward_batched_ex(L.l1.w, &L.l1.b, n3, bt::kLinearActGeluExact, bt::kLinearEpiStore, nullptr, hh);
    bt::linear_forward_batched_ex(L.l2.w, &L.l2.b, hh, bt::kLinearActNone, bt::kLinearEpiAccumulate, nullptr, x);

    // norm_out: GroupNorm(1, D) over each segment's (L, D), per-feature affine.
    bt::Tensor g;
    bt::group_norm_forward(as(x, B, Lq * D), unit_gamma_, unit_beta_, B, 1, 1, Lq * D, 1, 1e-5f, g);
    reshape(g, B * Lq, D);
    bt::broadcast_mul(g, L.norm_out.gamma, x);
    bt::add_row_bias_inplace(x, L.norm_out.beta);
}

void Net::transformer(bt::Tensor& x, bt::Tensor& xt, int B, int Fr, int T1, int T2) const {
    const BatchConsts& bc = consts(B);
    const int C = up_.in, Lx = T1 * Fr;
    bt::Tensor a, tok, X, XT;
    // (B, C, Fr, T1) -> (B*T1*Fr, C)
    bt::nchw_to_sequence(as(x, B * C, Fr * T1), B * C, Fr, 1, T1, a);   // (B*C*T1, Fr)
    bt::nchw_to_sequence(as(a, B, C * T1 * Fr), B, C, T1, Fr, tok);    // (B*T1*Fr, C)
    linear(up_, tok, a);
    layernorm(a, norm_in_, X);
    bt::add_inplace(X, bc.pos_x);
    // (B, C, T2) -> (B*T2, C)
    bt::nchw_to_sequence(as(xt, B, C * T2), B, C, 1, T2, tok);
    linear(up_t_, tok, a);
    layernorm(a, norm_in_t_, XT);
    bt::add_inplace(XT, bc.pos_t);

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        if (!layers_[i].cross) {
            tlayer(layers_[i], X, nullptr, B, Lx, Lx, bc, false);
            tlayer(layers_t_[i], XT, nullptr, B, T2, T2, bc, true);
        } else {
            bt::Tensor old = X;
            tlayer(layers_[i], X, &XT, B, Lx, T2, bc, false);
            tlayer(layers_t_[i], XT, &old, B, T2, Lx, bc, true);
        }
    }

    // Back to (B, C, Fr, T1) and (B, C, T2).
    linear(down_, X, tok);                                             // (B*T1*Fr, C)
    bt::sequence_to_nchw(tok, B, C, T1, Fr, a);                        // (B, C*T1*Fr)
    bt::nchw_to_sequence(as(a, B * C, T1 * Fr), B * C, T1, 1, Fr, x);  // (B*C*Fr, T1)
    reshape(x, B, C * Fr * T1);
    linear(down_t_, XT, tok);                                          // (B*T2, C)
    bt::sequence_to_nchw(tok, B, C, 1, T2, xt);
}

}  // namespace brosoundml::htd
