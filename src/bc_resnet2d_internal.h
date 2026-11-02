#pragma once

#include "brosoundml/bc_resnet2d.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;

// Adam/grad state for the GPU training surface. One Slot per trainable tensor;
// the param pointer goes back into the model's Impl so adam_step rewrites it in
// place. Grad/m/v live on the parameter's device. Defined here (not just
// forward-declared) so unique_ptr<...> in Impl has a complete type for its
// destructor.
struct BcResnet2dTrainState {
    int step = 0;
    struct Slot {
        bt::Tensor* param = nullptr;
        bt::Tensor  grad, m, v;
        std::string name;
    };
    std::vector<Slot> slots;

    void add(bt::Tensor* p, const std::string& name) {
        Slot s;
        s.param = p;
        s.grad = bt::Tensor::zeros_on(p->device, p->rows, p->cols, bt::Dtype::FP32);
        s.m    = bt::Tensor::zeros_on(p->device, p->rows, p->cols, bt::Dtype::FP32);
        s.v    = bt::Tensor::zeros_on(p->device, p->rows, p->cols, bt::Dtype::FP32);
        s.name = name;
        slots.push_back(std::move(s));
    }
    bt::Tensor& grad(const std::string& name) {
        for (auto& s : slots) if (s.name == name) return s.grad;
        throw std::runtime_error("brosoundml: bc_resnet2d train: grad not found: " + name);
    }
    Slot* find(const std::string& name) {
        for (auto& s : slots) if (s.name == name) return &s;
        return nullptr;
    }
    void zero_grads() { for (auto& s : slots) s.grad.zero(); }
    void adam(float lr) {
        ++step;
        for (auto& s : slots)
            bt::adam_step(*s.param, s.grad, s.m, s.v, lr, 0.9f, 0.999f, 1e-8f, step);
    }
};

namespace {
namespace detail_bc2d {

[[noreturn]] inline void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}
// ─── 2D conv layer ──────────────────────────────────────────────────────────
//
// A general NCHW 2D conv over (H=frequency, W=time). Frequency is convolved
// non-causally with a symmetric pad (pad_h); time is convolved CAUSALLY — we
// left-pad W by dil_w*(kW-1) (with zeros one-shot, or the streaming cache) and
// call conv2d_forward with pad_w=0, so output column t depends only on input
// columns <= t. groups==Cout==Cin is depthwise; groups==1 with kH=kW=1 is a
// pointwise channel mix.
struct Conv2dLayer {
    int Cin = 0, Cout = 0;
    int kH = 1, kW = 1;
    int sH = 1;                 // frequency stride (time stride always 1)
    int padH = 0;               // symmetric frequency pad
    int dilW = 1;               // temporal dilation
    int groups = 1;
    int padLeftW = 0;           // = dilW*(kW-1); causal left-pad along time
    bt::Tensor W;               // (Cout, (Cin/groups)*kH*kW) OIHW
    bt::Tensor b;               // (Cout, 1)
    // Streaming state no longer lives here — the per-stream causal time-ring
    // (the last padLeftW columns per (Cin,H) row, laid out (1, Cin*Hcache*
    // padLeftW)) is a tensor in BcResnet2dSession, addressed by this index
    // (>=0 for a cache-bearing conv, -1 for a kW==1 conv). Hcache is the
    // layer's input H, used to size that cache.
    int cache_index = -1;
    int Hcache = 0;

    // Cols of this layer's per-stream cache (0 if it needs none).
    int cache_cols() const { return padLeftW > 0 ? Cin * Hcache * padLeftW : 0; }
};

struct BatchNorm2d {
    int        channels = 0;
    bt::Tensor gamma, beta, mean, var;
    float      eps = 1e-5f;
    bool       present = false;
};

// One broadcasted-residual BC block.
//   transition (is_transition=true): cin -> cout, frequency stride fsf.
//     pw_in (1x1, cin->cout) + BN + ReLU, then the BC body on cout channels,
//     residual via a strided 1x1 (cin->cout) to match the downsampled shape.
//   normal: cin==cout, stride 1, residual = identity.
struct BCBlock {
    bool is_transition = false;
    int  cin = 0, cout = 0, fsf = 1, tdil = 1;

    Conv2dLayer pw_in;   BatchNorm2d bn_in;     // transition only
    Conv2dLayer f2;      BatchNorm2d bn_f2;     // freq depthwise (kf x 1)
    Conv2dLayer f1;      BatchNorm2d bn_f1;     // time depthwise (1 x kt), causal
    Conv2dLayer pw_f1;                          // 1x1 on the temporal branch
    Conv2dLayer res;     BatchNorm2d bn_res;    // transition residual (strided 1x1)

    // Fixed (non-trainable) depthwise kernels for freq-average-pool and its
    // transpose (broadcast over frequency), sized to this block's z stage
    // (cout channels, Hz frequency rows). Built in build().
    bt::Tensor avg_kernel;   // (cout, Hz)  all 1/Hz
    bt::Tensor ones_kernel;  // (cout, Hz)  all 1
    int Hz = 0;              // frequency height of z (post-f2)
};

[[maybe_unused]] inline void init_conv(Conv2dLayer& c, int cin, int cout, int kH, int kW,
               int sH, int padH, int dilW, int groups, int Hin,
               bt::Device dev) {
    c.Cin = cin; c.Cout = cout; c.kH = kH; c.kW = kW;
    c.sH = sH; c.padH = padH; c.dilW = dilW; c.groups = groups;
    c.padLeftW = dilW * (kW - 1);
    c.Hcache = Hin;
    const int win = (cin / groups) * kH * kW;
    c.W = bt::Tensor::zeros_on(dev, cout, win, bt::Dtype::FP32);
    c.b = bt::Tensor::zeros_on(dev, cout, 1, bt::Dtype::FP32);
    // No cache allocated here — per-stream caches live in BcResnet2dSession,
    // sized from cache_cols() and indexed by cache_index (assigned in build()).
}

[[maybe_unused]] inline void init_bn(BatchNorm2d& bn, int C, bt::Device dev, float eps) {
    bn.channels = C; bn.eps = eps; bn.present = true;
    std::vector<float> ones(static_cast<std::size_t>(C), 1.0f);
    bn.gamma = bt::Tensor::from_host_on(dev, ones.data(), C, 1);
    bn.beta  = bt::Tensor::zeros_on(dev, C, 1, bt::Dtype::FP32);
    bn.mean  = bt::Tensor::zeros_on(dev, C, 1, bt::Dtype::FP32);
    bn.var   = bt::Tensor::from_host_on(dev, ones.data(), C, 1);
}

[[maybe_unused]] inline void make_const_kernel(bt::Tensor& k, int C, int H, float val, bt::Device dev) {
    std::vector<float> h(static_cast<std::size_t>(C) * H, val);
    k = bt::Tensor::from_host_on(dev, h.data(), C, H);   // (C, H) depthwise OIHW
}

// Causal-time conv forward on an NCHW (1, Cin*H*W) input. Reads weights only.
// Left-pads the time (W) axis by padLeftW columns per (Cin,H) row — with zeros
// (state==null, the one-shot path) or this conv's per-stream cache (state set,
// cache then refreshed) — and runs a pad_w=0 conv2d, so output column t sees
// only input columns <= t. The cache tensor lives in
// state->conv_caches[c.cache_index].
[[maybe_unused]] inline void conv_forward(const Conv2dLayer& c, const bt::Tensor& X, int H, int W,
                  BcResnet2dSession* state, bt::Tensor& Y) {
    const bt::Device dev = X.device;
    const int rows = c.Cin * H;                 // number of (channel,freq) rows
    if (X.cols != rows * W) {
        fail("bc_resnet2d::conv_forward",
             "input cols " + std::to_string(X.cols) + " != Cin*H*W " +
             std::to_string(rows * W));
    }

    if (c.padLeftW == 0) {
        // No temporal context (kW==1): a plain conv2d, freq pad only.
        conv2d_forward(X, c.W, &c.b, /*N=*/1, c.Cin, H, W, c.Cout,
                       c.kH, c.kW, c.sH, /*sw=*/1, c.padH, /*pad_w=*/0,
                       /*dil_h=*/1, c.dilW, c.groups, Y);
        return;
    }

    bt::Tensor* cache =
        (state && c.cache_index >= 0 &&
         state->conv_caches[static_cast<std::size_t>(c.cache_index)].size() > 0)
            ? &state->conv_caches[static_cast<std::size_t>(c.cache_index)]
            : nullptr;

    const int Wp = W + c.padLeftW;
    bt::Tensor padded = bt::Tensor::zeros_on(dev, 1, rows * Wp, bt::Dtype::FP32);
    if (cache) {
        bt::copy_d2d_strided(*cache, 0, c.padLeftW, padded, 0, Wp,
                             c.padLeftW, rows);
    }
    bt::copy_d2d_strided(X, 0, W, padded, c.padLeftW, Wp, W, rows);
    conv2d_forward(padded, c.W, &c.b, /*N=*/1, c.Cin, H, Wp, c.Cout,
                   c.kH, c.kW, c.sH, /*sw=*/1, c.padH, /*pad_w=*/0,
                   /*dil_h=*/1, c.dilW, c.groups, Y);

    if (cache) {
        bt::copy_d2d_strided(padded, Wp - c.padLeftW, Wp,
                             *cache, 0, c.padLeftW, c.padLeftW, rows);
    }
}

[[maybe_unused]] inline void bn_inplace(const BatchNorm2d& bn, bt::Tensor& X, int C, int H, int W) {
    if (!bn.present) return;
    bt::Tensor Y;
    bt::batch_norm_inference(X, bn.gamma, bn.beta, bn.mean, bn.var,
                             /*N=*/1, C, H, W, bn.eps, Y);
    X = std::move(Y);
}

[[maybe_unused]] inline void relu_into(const bt::Tensor& X, bt::Tensor& Y) {
    Y = bt::Tensor::empty_on(X.device, X.rows, X.cols, bt::Dtype::FP32);
    bt::relu_forward(X, Y);
}

// Frequency-average-pool (C,H,W) -> (C,1,W) via a depthwise conv with a (1/H)
// kernel spanning all H rows (kH=H, one output position).
[[maybe_unused]] inline void freq_avg(const bt::Tensor& X, const bt::Tensor& avg_kernel,
              int C, int H, int W, bt::Tensor& Y) {
    conv2d_forward(X, avg_kernel, /*bias=*/nullptr, /*N=*/1, C, H, W, C,
                   /*kH=*/H, /*kW=*/1, /*sh=*/1, /*sw=*/1, /*ph=*/0, /*pw=*/0,
                   /*dh=*/1, /*dw=*/1, /*groups=*/C, Y);
}

// Broadcast (C,1,W) -> (C,H,W): each frequency row copies the single input row.
// Implemented as the input-gradient of a depthwise all-ones (kH=H) conv, which
// scatters the single value to every frequency tap.
[[maybe_unused]] inline void freq_broadcast(const bt::Tensor& X1, const bt::Tensor& ones_kernel,
                    int C, int H, int W, bt::Tensor& Y) {
    bt::conv2d_backward_input(ones_kernel, X1, /*N=*/1, C, H, W, C,
                              /*kH=*/H, /*kW=*/1, /*sh=*/1, /*sw=*/1,
                              /*ph=*/0, /*pw=*/0, /*dh=*/1, /*dw=*/1,
                              /*groups=*/C, Y);
}

}  // namespace detail_bc2d

namespace {
using namespace detail_bc2d;
}  // namespace

// ─── Impl ────────────────────────────────────────────────────────────────────

struct BcResnet2d::Impl {
    BcResnet2dConfig cfg;
    bt::Device       device = bt::Device::CPU;
    bool             fused = false;

    Conv2dLayer  stem;  BatchNorm2d bn_stem;
    int          stem_Hout = 0;
    std::vector<BCBlock> blocks;       // flattened across stages
    bt::Tensor   head_W;               // (1, c_last)
    bt::Tensor   head_b;               // (1, 1)
    int          c_last = 0;
    int          head_H = 0;           // frequency height entering the head
    bt::Tensor   head_avg_kernel;      // (c_last, head_H) for head freq-avg

    // GAP-ring capacity (a net property; the ring itself is per-stream and
    // lives in BcResnet2dSession).
    int gap_cap = 0;

    // Cache-bearing convs in a stable enumeration order; each is assigned its
    // cache_index = position here in build(). make_session() allocates one
    // cache slot per entry, so slot i always belongs to *cache_convs[i].
    std::vector<Conv2dLayer*> cache_convs;
    // Internally-owned session backing the legacy single-stream API.
    BcResnet2dSession owned_state;

    std::unique_ptr<BcResnet2dTrainState> train_state;

    int freq_pad(int k) const { return (k - 1) / 2; }

    // Walk the trunk convs once and bind each to a cache slot (stem, then per
    // block: pw_in, f2, f1, pw_f1, res). The head pools over time but holds its
    // ring separately (gap_window), not a conv cache.
    void collect_cache_convs() {
        cache_convs.clear();
        auto add = [&](Conv2dLayer& c) {
            c.cache_index = static_cast<int>(cache_convs.size());
            cache_convs.push_back(&c);
        };
        add(stem);
        for (auto& blk : blocks) {
            if (blk.is_transition) add(blk.pw_in);
            add(blk.f2); add(blk.f1); add(blk.pw_f1);
            if (blk.is_transition) add(blk.res);
        }
    }

    BcResnet2dSession alloc_state() const {
        BcResnet2dSession st;
        st.conv_caches.reserve(cache_convs.size());
        for (const Conv2dLayer* c : cache_convs) {
            const int cols = c->cache_cols();
            st.conv_caches.push_back(
                cols > 0 ? bt::Tensor::zeros_on(device, 1, cols, bt::Dtype::FP32)
                         : bt::Tensor{});
        }
        st.gap_window.assign(static_cast<std::size_t>(gap_cap) * c_last, 0.0f);
        st.gap_len = 0; st.gap_head = 0;
        return st;
    }

    void check_state(const BcResnet2dSession& st) const {
        if (st.conv_caches.size() != cache_convs.size() ||
            st.gap_window.size() != static_cast<std::size_t>(gap_cap) * c_last)
            fail("BcResnet2d::forward_streaming",
                 "stream state was not made by this model");
    }

    static void zero_state(BcResnet2dSession& st) {
        for (auto& c : st.conv_caches) if (c.size() > 0) c.zero();
        std::fill(st.gap_window.begin(), st.gap_window.end(), 0.0f);
        st.gap_len = 0; st.gap_head = 0;
    }

    void build() {
        const int H0 = cfg.n_mels;
        // Stem: full 2D conv 1 -> c_stem, causal in time, freq stride.
        init_conv(stem, /*cin=*/1, cfg.c_stem, cfg.stem_kf, cfg.stem_kt,
                  /*sH=*/cfg.stem_sf, /*padH=*/freq_pad(cfg.stem_kf),
                  /*dilW=*/1, /*groups=*/1, /*Hin=*/H0, device);
        init_bn(bn_stem, cfg.c_stem, device, cfg.bn_eps);
        stem_Hout = (H0 + 2 * freq_pad(cfg.stem_kf) - cfg.stem_kf) / cfg.stem_sf + 1;

        int cin = cfg.c_stem;
        int Hin = stem_Hout;
        blocks.clear();
        for (int s = 0; s < BcResnet2dConfig::n_stages; ++s) {
            const int cout = cfg.c[s];
            const int fsf  = cfg.fstride[s];
            const int d    = cfg.tdil[s];
            for (int bidx = 0; bidx < cfg.n_blocks[s]; ++bidx) {
                const bool transition = (bidx == 0);
                BCBlock blk;
                blk.is_transition = transition;
                blk.cin = transition ? cin : cout;
                blk.cout = cout;
                blk.fsf = transition ? fsf : 1;
                blk.tdil = d;
                const int block_cin = blk.cin;
                const int block_fsf = blk.fsf;
                const int pf = freq_pad(cfg.dw_kf);

                if (transition) {
                    init_conv(blk.pw_in, cin, cout, 1, 1, 1, 0, 1, 1, Hin, device);
                    init_bn(blk.bn_in, cout, device, cfg.bn_eps);
                }
                // f2: freq depthwise on `cout` channels (after pw_in), stride fsf.
                init_conv(blk.f2, cout, cout, cfg.dw_kf, 1, block_fsf, pf, 1,
                          /*groups=*/cout, Hin, device);
                init_bn(blk.bn_f2, cout, device, cfg.bn_eps);
                const int Hz = (Hin + 2 * pf - cfg.dw_kf) / block_fsf + 1;
                blk.Hz = Hz;
                // f1: time depthwise on (cout,1,W), causal, dilation d.
                init_conv(blk.f1, cout, cout, 1, cfg.dw_kt, 1, 0, d,
                          /*groups=*/cout, /*Hin=*/1, device);
                init_bn(blk.bn_f1, cout, device, cfg.bn_eps);
                init_conv(blk.pw_f1, cout, cout, 1, 1, 1, 0, 1, 1, /*Hin=*/1, device);
                // residual projection (transition): strided 1x1 cin->cout.
                if (transition) {
                    init_conv(blk.res, block_cin, cout, 1, 1, block_fsf, 0, 1, 1,
                              Hin, device);
                    init_bn(blk.bn_res, cout, device, cfg.bn_eps);
                }
                make_const_kernel(blk.avg_kernel, cout, Hz, 1.0f / Hz, device);
                make_const_kernel(blk.ones_kernel, cout, Hz, 1.0f, device);
                blocks.push_back(std::move(blk));
                cin = cout;
                Hin = Hz;
            }
        }
        c_last = cin;
        head_H = Hin;
        head_W = bt::Tensor::zeros_on(device, 1, c_last, bt::Dtype::FP32);
        head_b = bt::Tensor::zeros_on(device, 1, 1, bt::Dtype::FP32);
        make_const_kernel(head_avg_kernel, c_last, head_H, 1.0f / head_H, device);

        gap_cap = rf_frames();

        // Bind cache slots and allocate the legacy single-stream session.
        collect_cache_convs();
        owned_state = alloc_state();
    }

    int rf_frames() const {
        int rf = (cfg.stem_kt - 1);
        for (int s = 0; s < BcResnet2dConfig::n_stages; ++s) {
            for (int b = 0; b < cfg.n_blocks[s]; ++b)
                rf += cfg.tdil[s] * (cfg.dw_kt - 1);
        }
        return rf + 1;
    }

    // Run one BC block. X: (blk.cin, Hin, W) NCL-flat (1, cin*Hin*W). Returns
    // Y: (cout, Hz, W). Reads weights only; `state` (null for one-shot) carries
    // the per-stream time caches.
    void run_block(const BCBlock& blk, const bt::Tensor& X, int Hin, int W,
                   BcResnet2dSession* state, bt::Tensor& Y) const {
        const int cout = blk.cout;
        // Channel/stride entry.
        bt::Tensor u;     // (cout, Hin, W)
        if (blk.is_transition) {
            bt::Tensor t;
            conv_forward(blk.pw_in, X, Hin, W, state, t);
            bn_inplace(blk.bn_in, t, cout, Hin, W);
            relu_into(t, u);
        } else {
            u = X;        // normal block: identity entry on cout==cin
        }

        // f2: freq depthwise + BN.  (cout, Hz, W)
        bt::Tensor z;
        conv_forward(blk.f2, u, Hin, W, state, z);
        bn_inplace(blk.bn_f2, z, cout, blk.Hz, W);

        // f1: freq-avg -> time depthwise (causal) -> BN -> ReLU -> pw1x1.
        bt::Tensor a;     // (cout,1,W)
        freq_avg(z, blk.avg_kernel, cout, blk.Hz, W, a);
        bt::Tensor tt;
        conv_forward(blk.f1, a, /*H=*/1, W, state, tt);
        bn_inplace(blk.bn_f1, tt, cout, 1, W);
        bt::Tensor tt_relu; relu_into(tt, tt_relu);
        bt::Tensor tt_pw;
        conv_forward(blk.pw_f1, tt_relu, /*H=*/1, W, state, tt_pw);

        // y = ReLU( z + broadcast_H(tt_pw) )
        bt::Tensor bc;
        freq_broadcast(tt_pw, blk.ones_kernel, cout, blk.Hz, W, bc);
        bt::add_inplace(z, bc);
        bt::Tensor y; relu_into(z, y);

        // residual
        bt::Tensor res;
        if (blk.is_transition) {
            conv_forward(blk.res, X, Hin, W, state, res);
            bn_inplace(blk.bn_res, res, cout, blk.Hz, W);
        } else {
            res = X;   // identity (cout==cin, Hz==Hin)
        }
        bt::add_inplace(y, res);
        Y = std::move(y);
    }

    // Shared forward producing per-frame logits (T,1). state==null is the
    // one-shot path (fresh GAP window); non-null streams (persistent caches+GAP
    // in `state`).
    void forward_core(const bt::Tensor& feats, BcResnet2dSession* state,
                      bt::Tensor& out_logits) const {
        if (feats.rows != cfg.n_mels)
            fail("BcResnet2d::forward", "input rows != n_mels");
        if (feats.dtype != bt::Dtype::FP32)
            fail("BcResnet2d::forward", "input must be FP32");
        if (feats.device != device)
            fail("BcResnet2d::forward", "input device mismatch");
        const int T = feats.cols;
        if (T <= 0) { out_logits = bt::Tensor::zeros_on(device, 0, 1, bt::Dtype::FP32); return; }

        // feats (n_mels, T) row-major is already NCHW with C=1,H=n_mels,W=T.
        bt::Tensor x = bt::Tensor::empty_on(device, 1, cfg.n_mels * T, bt::Dtype::FP32);
        bt::copy_d2d(feats, 0, x, 0, cfg.n_mels * T);

        bt::Tensor y_stem;
        conv_forward(stem, x, cfg.n_mels, T, state, y_stem);
        bn_inplace(bn_stem, y_stem, cfg.c_stem, stem_Hout, T);
        bt::Tensor h; relu_into(y_stem, h);

        int Hin = stem_Hout;
        for (const auto& blk : blocks) {
            bt::Tensor y;
            run_block(blk, h, Hin, T, state, y);
            h = std::move(y);
            Hin = blk.Hz;
        }

        // Head: freq-avg the final feature -> (c_last,1,T) -> (c_last,T) host.
        bt::Tensor favg;
        freq_avg(h, head_avg_kernel, c_last, head_H, T, favg);   // (c_last,1,T)
        bt::Tensor favg_host = favg.to(bt::Device::CPU);
        bt::Tensor hW_host = head_W.to(bt::Device::CPU);
        bt::Tensor hb_host = head_b.to(bt::Device::CPU);
        const float* fp = favg_host.host_f32();      // (c_last, T) row-major
        const float* hW = hW_host.host_f32();
        const float  bias = hb_host.host_f32()[0];
        const int C = c_last, cap = gap_cap;

        // Streaming: the persistent GAP ring lives in the caller's state. One-
        // shot: a fresh local ring, so the head pools over just this clip.
        std::vector<float>* win; int* len; int* head_pos;
        std::vector<float> local; int ll = 0, lh = 0;
        if (state) { win = &state->gap_window; len = &state->gap_len; head_pos = &state->gap_head; }
        else { local.assign(static_cast<std::size_t>(cap) * C, 0.0f);
               win = &local; len = &ll; head_pos = &lh; }

        std::vector<float> logits(static_cast<std::size_t>(T), 0.0f);
        for (int t = 0; t < T; ++t) {
            // push feature[:,t] (gather across channel rows) into the ring.
            float* dst = win->data() + static_cast<std::size_t>(*head_pos) * C;
            for (int cc = 0; cc < C; ++cc)
                dst[cc] = fp[static_cast<std::size_t>(cc) * T + t];
            *head_pos = (*head_pos + 1) % cap;
            if (*len < cap) ++(*len);

            const double inv = 1.0 / static_cast<double>(*len);
            double s = static_cast<double>(bias);
            for (int j = 0; j < C; ++j) {
                double sum = 0.0;
                for (int k = 0; k < *len; ++k)
                    sum += static_cast<double>((*win)[static_cast<std::size_t>(k) * C + j]);
                s += static_cast<double>(hW[j]) * (sum * inv);
            }
            logits[static_cast<std::size_t>(t)] = static_cast<float>(s);
        }
        out_logits = bt::Tensor::from_host_on(device, logits.data(), T, 1);
    }
};


}  // namespace brosoundml
