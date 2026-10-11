#pragma once

// YuE2 backbone internals shared by yue2_model.cpp (load, AR path) and
// yue2_nar.cpp (NAR path). See yue2_model.h.

#include "yue2_model.h"

#include <brotensor/cuda_graph.h>
#include <brotensor/tensor.h>

#include <map>
#include <string>
#include <vector>

namespace brosoundml::yue2 {

namespace bt = brotensor;

[[noreturn]] void model_fail(const std::string& msg);

// A weight-only quantised copy of a weight (Model::load's ArWeights): codes
// (out, in) INT8 / FP8_E4M3 and FP32 per-row scales. Empty = use the BF16.
struct QuantW {
    bt::Tensor w, scales;
    bool empty() const { return w.rows == 0; }
};

// One attention + MLP path of one layer (AR or NAR).
struct PathW {
    bt::Tensor in_ln, q, k, v, o, q_norm, k_norm, post_ln, gate_up, gate, up, down;  // gate/up: views
    // AR decode's quantised copies (prefill and the NAR path keep the BF16).
    QuantW qq, qk, qv, qo, qgate_up, qdown;
};

struct LayerW {
    PathW ar, nar;
};

struct ModelWeights {
    Config cfg;
    bt::Device dev = bt::Device::CPU;
    bt::Dtype dt = bt::Dtype::BF16;      // activation + weight dtype
    bool loaded = false;

    bt::Tensor embed, lm_head, final_norm;  // (V,H) (V,H) (H,1)
    ArWeights ar_weights = ArWeights::Bf16;
    QuantW q_lm_head;                       // decode's lm_head (empty = BF16)
    std::vector<LayerW> layers;
    bt::Tensor vae2llm_w, vae2llm_b;        // (H,64) (H,1)
    bt::Tensor llm2vae_w, llm2vae_b;        // (64,H) (64,1)
    bt::Tensor t0_w, t0_b, t2_w, t2_b;      // time_embedder.mlp.0 / .2
    bt::Tensor pe;                          // (max_latent_frames, H)
    // RoPE tables for every position, FP32 values rounded to BF16 (the
    // reference casts cos/sin to the activation dtype): (max_pos, hd/2).
    bt::Tensor cos_tbl, sin_tbl;

    int esize() const { return dt == bt::Dtype::FP32 ? 4 : 2; }
};

// Row-major row view [row0, row0 + rows) of a (R, cols) tensor.
bt::Tensor rows_view(const bt::Tensor& t, int row0, int rows);
// Reinterpreting view of a whole contiguous tensor with another shape.
bt::Tensor reshape_view(const bt::Tensor& t, int rows, int cols);

// Y = X W^T (+ bias); `acc` adds into Y instead (residual epilogue).
void lin(const bt::Tensor& W, const bt::Tensor* bias, const bt::Tensor& X, bt::Tensor& Y,
         bt::Tensor* ws, bool acc = false);
// Per-head RMSNorm over (L, heads*hd) into dst (L, heads*hd).
void head_norm(const ModelWeights& w, const bt::Tensor& src, int heads, const bt::Tensor& gain,
               bt::Tensor& dst);
// Upload host int32 ids to a device (n,1) INT32 tensor.
void upload_ids(const ModelWeights& w, const int32_t* ids, int n, bt::Tensor& dst);
// Host -> device bytes (a memcpy on the CPU).
void h2d(bt::Device dev, void* dst, const void* src, std::size_t bytes);

// lm_head rows [lo, hi) (BF16), and their quantised copy (empty without one).
bt::Tensor lm_rows(const ModelWeights& w, int lo, int hi);
QuantW q_lm_rows(const ModelWeights& w, int lo, int hi);
// A decode linear: the quantised copy when there is one, else the BF16 W.
void dlin(const bt::Tensor& W, const QuantW& q, const bt::Tensor& X, bt::Tensor& Y, bt::Tensor* ws,
          int epi = 0 /* kLinearEpiStore */);

// Apply `hooks` for layer l to the residual rows `x`: row r is position
// pos_row0 + r; per-position steers and the capture cover rows [lo, hi)
// only, whole-sequence steers every row. `eval` is passed to on_capture.
void apply_hooks(const ModelWeights& w, const LayerHooks* hooks, int l, bt::Tensor& x, int pos_row0,
                 int lo, int hi, int eval, bool capture = true);

// Download a BF16 / FP32 device tensor (or view) to host FP32.
void to_host_f32(const ModelWeights& w, const bt::Tensor& t, std::vector<float>& out);

// Scratch for an L-row forward (prefill, NAR).
struct RowScratch {
    bt::Tensor x, n, q, k, v, qn, kn, att, g, u, idx, ws, cos, sin;
};

// Run the AR path over `n` tokens at positions [pos0, pos0 + n), writing
// every layer's K/V into rows [pos0, pos0 + n) of K[l] / V[l] (rows [0, pos0)
// must already hold the earlier tokens'). The residual ends in sc.x. With
// `kv_only` the last layer stops after writing its K/V (a NAR chunk needs no
// AR output).
void ar_rows(const ModelWeights& w, RowScratch& sc, const int32_t* ids, int n, int pos0,
             std::vector<bt::Tensor>& K, std::vector<bt::Tensor>& V, bool kv_only,
             const LayerHooks* hooks = nullptr, bool capture = true);

struct ArSession {
    int B = 1, cap = 0;
    const LayerHooks* hooks = nullptr;
    std::vector<int> len;                          // tokens cached per branch
    std::vector<std::vector<bt::Tensor>> K, V;     // [branch][layer] (cap, n_kv*hd)
    int lo = 0, hi = 0;                            // lm_head row range
    bt::Tensor logits;                             // (B, hi - lo)
    RowScratch pre;

    // Decode step of R tokens per branch (B * R rows, branch-major), one per
    // R used, persistent across replays (each keeps its own graph).
    struct Step {
        bt::Tensor tok, rows, lens;                // (B*R,1) / (B*R,1) / (B,1) INT32 staged per step
        std::vector<bt::Tensor> rowsb;             // per branch (R,1) INT32 views of rows
        std::vector<bt::Tensor> mask;              // per branch (cap,1) FP32 (non-CUDA decode, R = 1)
        bt::Tensor ones;                           // (1,1) FP32
        bt::Tensor cos, sin;                       // (B*R, hd/2) FP32
        bt::Tensor x, n, q, k, v, qn, kn, att, g, u, logits, ws;
        bt::CudaGraph graph;
        bool captured = false;
        int graph_lo = -1, graph_hi = -1;
        std::vector<int> mask_len;
    };
    std::map<int, Step> steps;                     // by R
};

struct NarChunk {
    int ar_len = 0, frames = 0, N = 0;             // N = frames + 2
    const LayerHooks* hooks = nullptr;
    int frame0 = 0;                                // absolute latent frame of content row 0
    std::vector<int> capture_evals;
    int eval = 0;                                  // velocity evaluations run so far
    double cur_raw = 0;                            // raw t of the evaluation in flight
    std::vector<bt::Tensor> K, V;                  // [layer] (ar_len + N, n_kv*hd)
    RowScratch sc;
    bt::Tensor xpad, mpad, vel, tmp;               // (N,64) (N,64) (frames,64) (frames,64)
    bt::Tensor temb_in, temb_h, temb;              // (1,256) (1,H) (1,H)
    bt::Tensor cos, sin;                           // (N, hd/2) views of the tables
};

}  // namespace brosoundml::yue2
