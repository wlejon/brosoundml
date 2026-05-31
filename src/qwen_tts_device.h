#pragma once

// Shared device-neutral helpers for the Qwen3-TTS modules (Talker, Code
// Predictor, codec). Each helper composes brotensor ops so a module runs on
// whatever device its weights live on. Weights are uploaded FP32 on every
// backend — brotensor's matmul / rms_norm / rope_apply / silu all have FP32
// CUDA kernels — so the CUDA path stays bit-close to the CPU path. FP16 is used
// only inside flash attention (the one attention op with no FP32 GPU kernel),
// cast in and out the way whisper/kokoro do; the KV cache itself stays FP32.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace brosoundml {
namespace qtd {

namespace bt = brotensor;

// Upload host int32 indices to a device INT32 buffer. embedding_lookup_forward
// reads d_idx as a *device* pointer on CUDA, so a host vector cannot be passed
// straight through; on CPU this just holds the host copy.
inline bt::Tensor upload_idx(bt::Device dev, const std::int32_t* h, int n) {
    bt::Tensor t = bt::Tensor::empty_on(dev, n, 1, bt::Dtype::INT32);
    if (n == 0) return t;
    if (dev == bt::Device::CPU) {
        std::memcpy(t.data, h, static_cast<std::size_t>(n) * sizeof(std::int32_t));
    } else {
        bt::detail::alloc_for(dev).memcpy_h2d(
            t.data, h, static_cast<std::size_t>(n) * sizeof(std::int32_t));
    }
    return t;
}

// Gather rows of `W` (V,D) by `idx` -> (idx.size(), D), device-neutral.
inline bt::Tensor gather_rows(const bt::Tensor& W,
                              const std::vector<std::int32_t>& idx) {
    bt::Tensor out = bt::Tensor::empty_on(W.device, 0, 0, W.dtype);
    const int B = static_cast<int>(idx.size());
    if (W.device == bt::Device::CPU) {
        bt::embedding_lookup_forward(W, idx.data(), B, out);
    } else {
        bt::Tensor di = upload_idx(W.device, idx.data(), B);
        bt::embedding_lookup_forward(
            W, static_cast<const std::int32_t*>(di.data), B, out);
    }
    return out;
}

// Y = X @ W^T (+ bias). A null bias adds a zero column on W's device/dtype.
inline void linear(const bt::Tensor& W, const bt::Tensor* bias,
                   const bt::Tensor& X, bt::Tensor& Y) {
    if (bias) {
        bt::linear_forward_batched(W, *bias, X, Y);
        return;
    }
    bt::Tensor zero = bt::Tensor::zeros_on(W.device, W.rows, 1, W.dtype);
    bt::linear_forward_batched(W, zero, X, Y);
}

// Per-head RMSNorm over head_dim. X (n, heads*hd) viewed as (n*heads, hd),
// normalized with gamma (hd,1), reshaped back into Y (n, heads*hd).
inline void head_rms_norm(const bt::Tensor& X, int n, int heads, int hd,
                          const bt::Tensor& gamma, float eps, bt::Tensor& Y) {
    bt::Tensor xv = bt::Tensor::view(X.device, X.data, n * heads, hd, X.dtype);
    bt::rms_norm_forward(xv, gamma, eps, Y);   // (n*heads, hd)
    Y.rows = n;
    Y.cols = heads * hd;                       // same buffer, reshaped
}

// Expand a GQA K/V (n, n_kv*hd) to (n, n_q*hd) by repeating each KV head across
// its query-head group via a row-gather. Returns a non-owning view when
// n_kv == n_q (plain MHA), so callers must keep `K` alive while using it.
inline bt::Tensor expand_kv(const bt::Tensor& K, int n, int n_kv, int n_q,
                            int hd) {
    if (n_kv == n_q) {
        return bt::Tensor::view(K.device, K.data, K.rows, K.cols, K.dtype);
    }
    const int group = n_q / n_kv;
    std::vector<std::int32_t> idx(static_cast<std::size_t>(n) * n_q);
    for (int t = 0; t < n; ++t)
        for (int h = 0; h < n_q; ++h)
            idx[static_cast<std::size_t>(t) * n_q + h] = t * n_kv + h / group;
    bt::Tensor kv = bt::Tensor::view(K.device, K.data, n * n_kv, hd, K.dtype);
    bt::Tensor out = gather_rows(kv, idx);     // (n*n_q, hd)
    out.rows = n;
    out.cols = n_q * hd;
    return out;
}

// Fused FP32 attention over pre-projected Q (Lq,D), K/V (Lk,D) with `num_heads`
// heads of `head_dim`. Both CPU and CUDA run in FP32, so the GPU path matches
// the CPU path (and upstream) to float round-off:
//   - CPU: flash_attention_forward (FP32 native).
//   - CUDA: flash_attention_varlen_forward, which (unlike flash_attention_
//     forward) has an FP32 GPU kernel; a single batch=1 sequence reproduces
//     plain attention.
// `causal` requires Lq == Lk (prefill); a single query over the full valid
// cache uses causal == false (it legitimately attends every cached key).
inline void flash_attn(const bt::Tensor& Q, const bt::Tensor& K,
                       const bt::Tensor& V, int num_heads, int head_dim,
                       bool causal, bt::Tensor& O) {
    if (Q.device == bt::Device::CPU) {
        bt::flash_attention_forward(Q, K, V, nullptr, num_heads, causal, O);
        return;
    }
    const int Lq = Q.rows, Lk = K.rows;
    const std::int32_t cuq[2] = {0, Lq};
    const std::int32_t cuk[2] = {0, Lk};
    bt::Tensor dq = upload_idx(Q.device, cuq, 2);
    bt::Tensor dk = upload_idx(Q.device, cuk, 2);
    bt::flash_attention_varlen_forward(
        Q, K, V, static_cast<const std::int32_t*>(dq.data),
        static_cast<const std::int32_t*>(dk.data), /*batch_size=*/1,
        /*max_seqlen_q=*/Lq, /*max_seqlen_k=*/Lk, num_heads, head_dim, causal, O);
}

// SwiGLU gate: g <- silu(g) * u, in place on g. Both (n, inter).
inline void swiglu(bt::Tensor& g, const bt::Tensor& u) {
    bt::silu_forward(g, g);
    bt::mul_inplace(g, u);
}

// Copy a tensor (any device/dtype) into a host FP32 buffer of t.size() floats.
inline void to_host(const bt::Tensor& t, float* out) {
    if (t.dtype == bt::Dtype::FP32) {
        if (t.device == bt::Device::CPU) {
            std::memcpy(out, t.host_f32(),
                        static_cast<std::size_t>(t.size()) * sizeof(float));
            return;
        }
        bt::Tensor c = t.to(bt::Device::CPU);
        std::memcpy(out, c.host_f32(),
                    static_cast<std::size_t>(c.size()) * sizeof(float));
        return;
    }
    bt::Tensor g = bt::Tensor::empty_on(t.device, t.rows, t.cols, bt::Dtype::FP32);
    bt::cast(t, g, bt::Dtype::FP32);
    bt::Tensor c = (g.device == bt::Device::CPU) ? std::move(g)
                                                 : g.to(bt::Device::CPU);
    std::memcpy(out, c.host_f32(),
                static_cast<std::size_t>(c.size()) * sizeof(float));
}

// Transpose a (rows, cols) FP32 tensor to (cols, rows). brotensor has no device
// transpose op, so this round-trips through host — used only for the codec's
// NCL<->SEQ layout swaps, which sit on modest pre-upsample tensors.
inline bt::Tensor transpose2d(const bt::Tensor& x) {
    const int R = x.rows, C = x.cols;
    std::vector<float> s(static_cast<std::size_t>(R) * C);
    to_host(x, s.data());
    std::vector<float> d(static_cast<std::size_t>(R) * C);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            d[static_cast<std::size_t>(c) * R + r] = s[static_cast<std::size_t>(r) * C + c];
    return bt::Tensor::from_host_on(x.device, d.data(), C, R);
}

// Build interleaved-pair RoPE cos/sin tables (n, half) on `dev` for one rotary
// axis-position grid. pos[i] for table column i is pos_of(i); inv_freq[i] is
// the pair frequency. Tables are FP32 on every backend (rope_apply requires
// FP32 tables).
inline void build_rope_tables(bt::Device dev, int n, int half,
                              const std::vector<int>& pos_per_col_per_row,
                              const std::vector<float>& inv_freq,
                              bt::Tensor& cos_t, bt::Tensor& sin_t) {
    std::vector<float> cb(static_cast<std::size_t>(n) * half);
    std::vector<float> sb(static_cast<std::size_t>(n) * half);
    for (int t = 0; t < n; ++t) {
        for (int i = 0; i < half; ++i) {
            const float ang =
                static_cast<float>(pos_per_col_per_row[static_cast<std::size_t>(t) * half + i]) *
                inv_freq[i];
            cb[static_cast<std::size_t>(t) * half + i] = std::cos(ang);
            sb[static_cast<std::size_t>(t) * half + i] = std::sin(ang);
        }
    }
    cos_t = bt::Tensor::from_host_on(dev, cb.data(), n, half);
    sin_t = bt::Tensor::from_host_on(dev, sb.data(), n, half);
}

// Row-permutation indices that reorder each head's head_dim from the HF
// rotate-half layout (pair i = dims {i, i+half}) to brotensor rope_apply's
// adjacent-pair layout (pair i = dims {2i, 2i+1}). Applied to q/k projection
// rows and q/k_norm so adjacent-pair rope_apply reproduces HF rotate-half.
inline std::vector<std::int32_t> rotate_half_perm(int head_dim) {
    const int half = head_dim / 2;
    std::vector<std::int32_t> p(head_dim);
    for (int i = 0; i < half; ++i) {
        p[2 * i]     = i;
        p[2 * i + 1] = i + half;
    }
    return p;
}

// Expand a per-head permutation to full projection-row indices for `n_heads`
// heads of `head_dim` (row h*head_dim + j -> h*head_dim + perm[j]).
inline std::vector<std::int32_t> per_head_perm_rows(
    const std::vector<std::int32_t>& perm, int n_heads, int head_dim) {
    std::vector<std::int32_t> idx(static_cast<std::size_t>(n_heads) * head_dim);
    for (int h = 0; h < n_heads; ++h)
        for (int j = 0; j < head_dim; ++j)
            idx[static_cast<std::size_t>(h) * head_dim + j] =
                h * head_dim + perm[j];
    return idx;
}

}  // namespace qtd
}  // namespace brosoundml
