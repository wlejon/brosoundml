#pragma once

#include "brosoundml/phoneme_model.h"

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

// Adam/grad state for the training surface — one Slot per trainable tensor.
// Mirrors BcResnet2dTrainState; defined here (not forward-declared) so the
// unique_ptr<...> in Impl has a complete type for its destructor.
struct PhonemeNetTrainState {
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
        throw std::runtime_error("brosoundml: phoneme_model train: grad not found: " + name);
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

namespace detail_pm {

[[noreturn]] inline void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}
// ─── 2D conv layer (mirrors bc_resnet2d) ────────────────────────────────────
//
// NCHW 2D conv over (H=frequency, W=time). Frequency convolved non-causally with
// a symmetric pad; time convolved CAUSALLY by left-padding W by dilW*(kW-1) (with
// zeros, or the streaming cache) and running a pad_w=0 conv2d, so output column t
// depends only on input columns <= t.
struct Conv2dLayer {
    int Cin = 0, Cout = 0;
    int kH = 1, kW = 1;
    int sH = 1;
    int padH = 0;
    int dilW = 1;
    int groups = 1;
    int padLeftW = 0;           // = dilW*(kW-1)
    bt::Tensor W;               // (Cout, (Cin/groups)*kH*kW)
    bt::Tensor b;               // (Cout, 1)
    // Streaming state no longer lives here — the per-stream causal time-ring is
    // a tensor in PhonemeSession, addressed by this index (>=0 for a
    // cache-bearing conv, -1 for a kW==1 conv that needs no history). Hcache is
    // the layer's input frequency height, used to size that cache.
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

// One broadcasted-residual BC block (mirrors bc_resnet2d).
struct BCBlock {
    bool is_transition = false;
    int  cin = 0, cout = 0, fsf = 1, tdil = 1;

    Conv2dLayer pw_in;   BatchNorm2d bn_in;
    Conv2dLayer f2;      BatchNorm2d bn_f2;
    Conv2dLayer f1;      BatchNorm2d bn_f1;
    Conv2dLayer pw_f1;
    Conv2dLayer res;     BatchNorm2d bn_res;

    bt::Tensor avg_kernel;   // (cout, Hz)  all 1/Hz
    bt::Tensor ones_kernel;  // (cout, Hz)  all 1
    int Hz = 0;
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
    // No cache allocated here — per-stream caches live in PhonemeSession,
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
    k = bt::Tensor::from_host_on(dev, h.data(), C, H);
}

// Causal-time conv forward on an NCHW (1, Cin*H*W) input. Reads weights only
// (const layer). `state` carries the per-stream caches: non-null streams (reads
// this conv's cache as left history, then refreshes it); null is the one-shot /
// pointwise path (zero left-pad, no cache). The cache tensor itself lives in
// state->conv_caches[c.cache_index].
[[maybe_unused]] inline void conv_forward(const Conv2dLayer& c, const bt::Tensor& X, int H, int W,
                  PhonemeSession* state, bt::Tensor& Y) {
    const bt::Device dev = X.device;
    const int rows = c.Cin * H;
    if (X.cols != rows * W) {
        fail("phoneme_model::conv_forward",
             "input cols " + std::to_string(X.cols) + " != Cin*H*W " +
             std::to_string(rows * W));
    }

    if (c.padLeftW == 0) {
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
    if (cache)
        bt::copy_d2d_strided(*cache, 0, c.padLeftW, padded, 0, Wp,
                             c.padLeftW, rows);
    bt::copy_d2d_strided(X, 0, W, padded, c.padLeftW, Wp, W, rows);
    conv2d_forward(padded, c.W, &c.b, /*N=*/1, c.Cin, H, Wp, c.Cout,
                   c.kH, c.kW, c.sH, /*sw=*/1, c.padH, /*pad_w=*/0,
                   /*dil_h=*/1, c.dilW, c.groups, Y);

    if (cache)
        bt::copy_d2d_strided(padded, Wp - c.padLeftW, Wp,
                             *cache, 0, c.padLeftW, c.padLeftW, rows);
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

[[maybe_unused]] inline void freq_avg(const bt::Tensor& X, const bt::Tensor& avg_kernel,
              int C, int H, int W, bt::Tensor& Y) {
    conv2d_forward(X, avg_kernel, /*bias=*/nullptr, /*N=*/1, C, H, W, C,
                   /*kH=*/H, /*kW=*/1, /*sh=*/1, /*sw=*/1, /*ph=*/0, /*pw=*/0,
                   /*dh=*/1, /*dw=*/1, /*groups=*/C, Y);
}

[[maybe_unused]] inline void freq_broadcast(const bt::Tensor& X1, const bt::Tensor& ones_kernel,
                    int C, int H, int W, bt::Tensor& Y) {
    bt::conv2d_backward_input(ones_kernel, X1, /*N=*/1, C, H, W, C,
                              /*kH=*/H, /*kW=*/1, /*sh=*/1, /*sw=*/1,
                              /*ph=*/0, /*pw=*/0, /*dh=*/1, /*dw=*/1,
                              /*groups=*/C, Y);
}

}  // namespace detail_pm

namespace {
using namespace detail_pm;
}  // namespace

// ─── Impl ────────────────────────────────────────────────────────────────────

struct PhonemeNet::Impl {
    PhonemeNetConfig cfg;
    PhonemeClassMap  class_map;
    bt::Device       device = bt::Device::CPU;
    bool             fused = false;

    Conv2dLayer  stem;  BatchNorm2d bn_stem;
    int          stem_Hout = 0;
    std::vector<BCBlock> blocks;
    int          c_last = 0;
    int          head_H = 0;             // frequency height entering the head
    bt::Tensor   head_avg_kernel;        // (c_last, head_H)
    Conv2dLayer  head;                   // 1x1 conv c_last -> K (per-frame head)
    bt::Tensor   head_offsets;           // INT32 {0, K} for softmax_xent_fused_batched

    std::unique_ptr<PhonemeNetTrainState> train_state;

    // Cache-bearing convs in a stable enumeration order; each is assigned its
    // cache_index = position here in build(). make_session() allocates one
    // cache slot per entry, so slot i always belongs to *cache_convs[i].
    std::vector<Conv2dLayer*> cache_convs;
    // Internally-owned session backing the legacy single-stream API.
    PhonemeSession owned_state;

    int K() const { return cfg.num_classes; }
    int freq_pad(int k) const { return (k - 1) / 2; }

    // Walk the trunk convs once and bind each to a cache slot (stem, then per
    // block: pw_in, f2, f1, pw_f1, res). The head is pointwise in time — no
    // cache. Order is arbitrary but must be self-consistent; cache_index lives
    // on the layer, so the forward addresses its own slot regardless of call
    // order.
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

    PhonemeSession alloc_state() const {
        PhonemeSession st;
        st.conv_caches.reserve(cache_convs.size());
        for (const Conv2dLayer* c : cache_convs) {
            const int cols = c->cache_cols();
            st.conv_caches.push_back(
                cols > 0 ? bt::Tensor::zeros_on(device, 1, cols, bt::Dtype::FP32)
                         : bt::Tensor{});
        }
        return st;
    }

    void check_state(const PhonemeSession& st) const {
        if (st.conv_caches.size() != cache_convs.size())
            fail("PhonemeNet::forward_streaming",
                 "stream state was not made by this model");
    }

    static void zero_state(PhonemeSession& st) {
        for (auto& c : st.conv_caches) if (c.size() > 0) c.zero();
    }

    void build() {
        const int H0 = cfg.n_mels;
        init_conv(stem, /*cin=*/1, cfg.c_stem, cfg.stem_kf, cfg.stem_kt,
                  /*sH=*/cfg.stem_sf, /*padH=*/freq_pad(cfg.stem_kf),
                  /*dilW=*/1, /*groups=*/1, /*Hin=*/H0, device);
        init_bn(bn_stem, cfg.c_stem, device, cfg.bn_eps);
        stem_Hout = (H0 + 2 * freq_pad(cfg.stem_kf) - cfg.stem_kf) / cfg.stem_sf + 1;

        int cin = cfg.c_stem;
        int Hin = stem_Hout;
        blocks.clear();
        for (int s = 0; s < PhonemeNetConfig::n_stages; ++s) {
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
                init_conv(blk.f2, cout, cout, cfg.dw_kf, 1, block_fsf, pf, 1,
                          /*groups=*/cout, Hin, device);
                init_bn(blk.bn_f2, cout, device, cfg.bn_eps);
                const int Hz = (Hin + 2 * pf - cfg.dw_kf) / block_fsf + 1;
                blk.Hz = Hz;
                init_conv(blk.f1, cout, cout, 1, cfg.dw_kt, 1, 0, d,
                          /*groups=*/cout, /*Hin=*/1, device);
                init_bn(blk.bn_f1, cout, device, cfg.bn_eps);
                init_conv(blk.pw_f1, cout, cout, 1, 1, 1, 0, 1, 1, /*Hin=*/1, device);
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
        make_const_kernel(head_avg_kernel, c_last, head_H, 1.0f / head_H, device);

        // Per-frame head: 1x1 conv c_last -> K over the freq-pooled (H=1) feature.
        init_conv(head, /*cin=*/c_last, /*cout=*/K(), 1, 1, 1, 0, 1, 1,
                  /*Hin=*/1, device);

        // Device {0, K} offsets for the single-head framewise softmax-CE loss.
        std::vector<int> off = {0, K()};
        bt::Tensor offh = bt::Tensor::zeros_on(bt::Device::CPU, 2, 1, bt::Dtype::INT32);
        auto* op = static_cast<int32_t*>(offh.host_raw_mut());
        op[0] = off[0]; op[1] = off[1];
        head_offsets = offh.to(device);

        // Bind cache slots and allocate the legacy single-stream session.
        collect_cache_convs();
        owned_state = alloc_state();
    }

    int rf_frames() const {
        int rf = (cfg.stem_kt - 1);
        for (int s = 0; s < PhonemeNetConfig::n_stages; ++s)
            for (int b = 0; b < cfg.n_blocks[s]; ++b)
                rf += cfg.tdil[s] * (cfg.dw_kt - 1);
        return rf + 1;
    }

    // Run one BC block (mirrors bc_resnet2d's run_block; inference path). Reads
    // weights only; `state` (null for one-shot) carries the per-stream caches.
    void run_block(const BCBlock& blk, const bt::Tensor& X, int Hin, int W,
                   PhonemeSession* state, bt::Tensor& Y) const {
        const int cout = blk.cout;
        bt::Tensor u;
        if (blk.is_transition) {
            bt::Tensor t;
            conv_forward(blk.pw_in, X, Hin, W, state, t);
            bn_inplace(blk.bn_in, t, cout, Hin, W);
            relu_into(t, u);
        } else {
            u = X;
        }

        bt::Tensor z;
        conv_forward(blk.f2, u, Hin, W, state, z);
        bn_inplace(blk.bn_f2, z, cout, blk.Hz, W);

        bt::Tensor a;
        freq_avg(z, blk.avg_kernel, cout, blk.Hz, W, a);
        bt::Tensor tt;
        conv_forward(blk.f1, a, /*H=*/1, W, state, tt);
        bn_inplace(blk.bn_f1, tt, cout, 1, W);
        bt::Tensor tt_relu; relu_into(tt, tt_relu);
        bt::Tensor tt_pw;
        conv_forward(blk.pw_f1, tt_relu, /*H=*/1, W, state, tt_pw);

        bt::Tensor bc;
        freq_broadcast(tt_pw, blk.ones_kernel, cout, blk.Hz, W, bc);
        bt::add_inplace(z, bc);
        bt::Tensor y; relu_into(z, y);

        bt::Tensor res;
        if (blk.is_transition) {
            conv_forward(blk.res, X, Hin, W, state, res);
            bn_inplace(blk.bn_res, res, cout, blk.Hz, W);
        } else {
            res = X;
        }
        bt::add_inplace(y, res);
        Y = std::move(y);
    }

    // Inference forward producing per-frame logits (T, K). state==null is the
    // one-shot path; non-null streams (persistent causal caches in `state`). The
    // head is pointwise in time, so the two paths agree once the trunk caches do.
    void forward_core(const bt::Tensor& feats, PhonemeSession* state,
                      bt::Tensor& out_logits) const {
        if (feats.rows != cfg.n_mels)
            fail("PhonemeNet::forward", "input rows != n_mels");
        if (feats.dtype != bt::Dtype::FP32)
            fail("PhonemeNet::forward", "input must be FP32");
        if (feats.device != device)
            fail("PhonemeNet::forward", "input device mismatch");
        const int T = feats.cols;
        const int Kc = K();
        if (T <= 0) { out_logits = bt::Tensor::zeros_on(device, 0, Kc, bt::Dtype::FP32); return; }

        bt::Tensor x = bt::Tensor::empty_on(device, 1, cfg.n_mels * T, bt::Dtype::FP32);
        bt::copy_d2d(feats, 0, x, 0, cfg.n_mels * T);

        bt::Tensor y_stem;
        conv_forward(stem, x, cfg.n_mels, T, state, y_stem);
        bn_inplace(bn_stem, y_stem, cfg.c_stem, stem_Hout, T);
        bt::Tensor h; relu_into(y_stem, h);

        int Hin = stem_Hout;
        for (auto& blk : blocks) {
            bt::Tensor y;
            run_block(blk, h, Hin, T, state, y);
            h = std::move(y);
            Hin = blk.Hz;
        }

        // Head: freq-avg → (1, c_last*T) → 1x1 conv → (1, K*T) → (T, K). The head
        // is pointwise in time (no history), so it never touches the stream cache.
        bt::Tensor favg;
        freq_avg(h, head_avg_kernel, c_last, head_H, T, favg);   // (1, c_last*T)
        bt::Tensor hc;
        conv_forward(head, favg, /*H=*/1, T, /*state=*/nullptr, hc);  // (1, K*T)
        bt::nchw_to_sequence(hc, /*N=*/1, /*C=*/Kc, /*H=*/1, /*W=*/T, out_logits);
    }
};


}  // namespace brosoundml
