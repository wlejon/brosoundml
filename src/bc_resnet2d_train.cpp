#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "bc_resnet2d_internal.h"

namespace brosoundml {

// ─── xavier init ──────────────────────────────────────────────────────────────
namespace {
std::uint64_t splitmix64(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
float next_uniform(std::uint64_t& s) {
    return static_cast<float>((splitmix64(s) >> 11) * (1.0 / 9007199254740992.0));
}
void xavier_fill(bt::Tensor& W, std::uint64_t& s) {
    bt::Tensor host = W.to(bt::Device::CPU);
    const int fan_in = host.cols, fan_out = host.rows;
    const float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    float* p = host.host_f32_mut();
    for (int i = 0; i < host.size(); ++i) p[i] = (2.0f * next_uniform(s) - 1.0f) * limit;
    W = host.to(W.device);
}
}  // namespace

void BcResnet2d::xavier_init_weights(std::uint64_t seed) {
    if (impl_->fused) fail("BcResnet2d::xavier_init_weights", "model is fused");
    std::uint64_t s = seed ? seed : 0x123456789ull;
    auto cw = [&](Conv2dLayer& c) { xavier_fill(c.W, s); };
    cw(impl_->stem);
    for (auto& blk : impl_->blocks) {
        if (blk.is_transition) { cw(blk.pw_in); cw(blk.res); }
        cw(blk.f2); cw(blk.f1); cw(blk.pw_f1);
    }
    xavier_fill(impl_->head_W, s);
}

// ─── Training surface (GPU, device-resident) ──────────────────────────────────
//
// Hand-rolled backward through the 2D BC-ResNet, mirroring forward_core() but in
// batched (N=B) layout and with BN in train-mode (batch statistics). Every op is
// device-dispatched through brotensor, so the whole pass runs on the model's
// device (CUDA in production). The single-logit-per-clip training objective pools
// the final feature over the full T window (whole-clip GAP), exactly like the 1D
// trainer; per-frame streaming logits are an inference-only concern.
//
// Causal time convs are handled as in the 1D path: the forward left-pads the time
// (W) axis by padLeftW = dilW*(kW-1), runs a pad_w=0 conv, and caches the padded
// input for the weight grad; the backward runs conv2d_backward_input on the padded
// buffer and strips the leading pad columns. Only the stem and the f1 time-conv
// have padLeftW > 0 (every other conv has kW==1).

namespace {

// Caches captured during the training forward, enough to walk the graph back.
struct Conv2dCache { bt::Tensor X_padded; };          // padded forward input (W-grad)
struct Bn2dCache   { bt::Tensor X_in, saved_mean, saved_rstd; };

struct BlockCache {
    int Hin = 0;                 // block input frequency height
    Conv2dCache pw_in; Bn2dCache bn_in; bt::Tensor in_pre_relu;   // transition entry
    Conv2dCache f2;    Bn2dCache bn_f2;
    bt::Tensor  a;               // freq-avg(z), input to f1 conv  (unused in bwd, kept for clarity)
    Conv2dCache f1;    Bn2dCache bn_f1; bt::Tensor tt_pre_relu;
    Conv2dCache pw_f1;
    bt::Tensor  zc_pre_relu;     // z + broadcast(tt_pw), pre final ReLU
    Conv2dCache res;   Bn2dCache bn_res;                          // transition residual
};

struct TrainCache2d {
    Conv2dCache stem;  Bn2dCache bn_stem; bt::Tensor stem_pre_relu;
    std::vector<BlockCache> blocks;
    bt::Tensor pooled;           // (B, c_last)
};

// Left-pad an NCHW (N, C*H*W) tensor along the time axis W by `pad` zeros, per
// (n,c,h) row. Returns (N, C*H*(W+pad)) on X.device. pad==0 → deep copy.
bt::Tensor left_pad_w(const bt::Tensor& X, int N, int C, int H, int W, int pad) {
    if (pad == 0) return X;
    bt::Tensor Y = bt::Tensor::zeros_on(X.device, N, C * H * (W + pad), bt::Dtype::FP32);
    const int rows = N * C * H;
    bt::copy_d2d_strided(X, 0, W, Y, pad, W + pad, W, rows);
    return Y;
}
// Strip the leading `pad` time columns per (n,c,h) row: (N,C*H*(W+pad)) → (N,C*H*W).
bt::Tensor strip_left_pad_w(const bt::Tensor& X, int N, int C, int H, int W, int pad) {
    if (pad == 0) return X;
    bt::Tensor Y = bt::Tensor::empty_on(X.device, N, C * H * W, bt::Dtype::FP32);
    const int rows = N * C * H;
    bt::copy_d2d_strided(X, pad, W + pad, Y, 0, W, W, rows);
    return Y;
}

// Conv forward (train): caches the time-padded input. Y is (N, Cout*Hout*W).
void conv_fwd(const Conv2dLayer& c, const bt::Tensor& X, int N, int Hin, int W,
              Conv2dCache& cache, bt::Tensor& Y) {
    cache.X_padded = left_pad_w(X, N, c.Cin, Hin, W, c.padLeftW);
    const int Wp = W + c.padLeftW;
    conv2d_forward(cache.X_padded, c.W, &c.b, N, c.Cin, Hin, Wp, c.Cout,
                   c.kH, c.kW, c.sH, /*sw=*/1, c.padH, /*pad_w=*/0,
                   /*dh=*/1, c.dilW, c.groups, Y);
}
// Conv backward (train). dX overwritten (N, Cin*Hin*W); dW/dB accumulated (caller zeros).
void conv_bwd(const Conv2dLayer& c, const Conv2dCache& cache, const bt::Tensor& dY,
              int N, int Hin, int W, int Hout,
              bt::Tensor& dX, bt::Tensor& dW, bt::Tensor& dB) {
    const int Wp = W + c.padLeftW;
    bt::Tensor dX_padded;
    bt::conv2d_backward_input(c.W, dY, N, c.Cin, Hin, Wp, c.Cout, c.kH, c.kW,
                              c.sH, /*sw=*/1, c.padH, /*pad_w=*/0, /*dh=*/1, c.dilW,
                              c.groups, dX_padded);
    dX = strip_left_pad_w(dX_padded, N, c.Cin, Hin, W, c.padLeftW);
    bt::conv2d_backward_weight(cache.X_padded, dY, N, c.Cin, Hin, Wp, c.Cout,
                               c.kH, c.kW, c.sH, /*sw=*/1, c.padH, /*pad_w=*/0,
                               /*dh=*/1, c.dilW, c.groups, dW);
    bt::conv2d_backward_bias(dY, N, c.Cout, Hout, W, dB);
}

void bn_fwd(BatchNorm2d& bn, const bt::Tensor& X, int N, int C, int H, int W,
            bt::Tensor& Y, Bn2dCache& cache) {
    Y = bt::Tensor::empty_on(X.device, X.rows, X.cols, bt::Dtype::FP32);
    cache.X_in       = X;
    cache.saved_mean = bt::Tensor::empty_on(X.device, C, 1, bt::Dtype::FP32);
    cache.saved_rstd = bt::Tensor::empty_on(X.device, C, 1, bt::Dtype::FP32);
    bt::batch_norm_forward(X, bn.gamma, bn.beta, bn.mean, bn.var, N, C, H, W,
                           bn.eps, /*momentum=*/0.1f, Y, cache.saved_mean, cache.saved_rstd);
}
void bn_eval(const BatchNorm2d& bn, const bt::Tensor& X, int N, int C, int H, int W,
             bt::Tensor& Y) {
    Y = bt::Tensor::empty_on(X.device, X.rows, X.cols, bt::Dtype::FP32);
    bt::batch_norm_inference(X, bn.gamma, bn.beta, bn.mean, bn.var, N, C, H, W, bn.eps, Y);
}
void bn_bwd(BatchNorm2d& bn, const Bn2dCache& cache, const bt::Tensor& dY,
            int N, int C, int H, int W, bt::Tensor& dX, bt::Tensor& dG, bt::Tensor& dB) {
    dX = bt::Tensor::empty_on(dY.device, dY.rows, dY.cols, bt::Dtype::FP32);
    dG = bt::Tensor::zeros_on(dY.device, C, 1, bt::Dtype::FP32);
    dB = bt::Tensor::zeros_on(dY.device, C, 1, bt::Dtype::FP32);
    bt::batch_norm_backward(cache.X_in, bn.gamma, cache.saved_mean, cache.saved_rstd,
                            dY, N, C, H, W, dX, dG, dB);
}

void relu_fwd(const bt::Tensor& X, bt::Tensor& Y) {
    Y = bt::Tensor::empty_on(X.device, X.rows, X.cols, bt::Dtype::FP32);
    bt::relu_forward_batched(X, Y);
}
void relu_bwd(const bt::Tensor& pre, const bt::Tensor& dY, bt::Tensor& dX) {
    dX = bt::Tensor::empty_on(dY.device, dY.rows, dY.cols, bt::Dtype::FP32);
    bt::relu_backward_batched(pre, dY, dX);
}

// Batched freq-avg / freq-broadcast and their adjoints (fixed kernels, no grad).
void favg_fwd(const bt::Tensor& X, const bt::Tensor& avg_k, int N, int C, int H, int W,
              bt::Tensor& Y) {
    conv2d_forward(X, avg_k, nullptr, N, C, H, W, C, /*kH=*/H, /*kW=*/1, 1, 1, 0, 0,
                   1, 1, /*groups=*/C, Y);
}
void favg_bwd(const bt::Tensor& avg_k, const bt::Tensor& dY, int N, int C, int H, int W,
              bt::Tensor& dX) {
    bt::conv2d_backward_input(avg_k, dY, N, C, H, W, C, /*kH=*/H, /*kW=*/1, 1, 1, 0, 0,
                              1, 1, /*groups=*/C, dX);
}
void fbcast_fwd(const bt::Tensor& X1, const bt::Tensor& ones_k, int N, int C, int H, int W,
                bt::Tensor& Y) {
    bt::conv2d_backward_input(ones_k, X1, N, C, H, W, C, /*kH=*/H, /*kW=*/1, 1, 1, 0, 0,
                              1, 1, /*groups=*/C, Y);
}
void fbcast_bwd(const bt::Tensor& ones_k, const bt::Tensor& dY, int N, int C, int H, int W,
                bt::Tensor& dX1) {
    conv2d_forward(dY, ones_k, nullptr, N, C, H, W, C, /*kH=*/H, /*kW=*/1, 1, 1, 0, 0,
                   1, 1, /*groups=*/C, dX1);
}

// BN helper that branches train (batch stats + cache) vs eval (running stats).
void bn_step(BatchNorm2d& bn, const bt::Tensor& X, int N, int C, int H, int W,
             bool train, bt::Tensor& Y, Bn2dCache& cache) {
    if (train) bn_fwd(bn, X, N, C, H, W, Y, cache);
    else       bn_eval(bn, X, N, C, H, W, Y);
}

// One BC block forward. X: (N, cin*Hin*W). Y: (N, cout*Hz*W). train toggles BN
// mode + cache writes (block backward consumes the cache, so train must be true
// for any pass that will be backpropagated).
void block_fwd(BCBlock& blk, const bt::Tensor& X, int N, int Hin, int W,
               bool train, BlockCache& bc, bt::Tensor& Y) {
    const int cout = blk.cout, Hz = blk.Hz;
    bc.Hin = Hin;

    bt::Tensor u;
    if (blk.is_transition) {
        bt::Tensor t; conv_fwd(blk.pw_in, X, N, Hin, W, bc.pw_in, t);
        bt::Tensor tbn; bn_step(blk.bn_in, t, N, cout, Hin, W, train, tbn, bc.bn_in);
        bc.in_pre_relu = tbn;
        relu_fwd(tbn, u);
    } else {
        u = X;
    }

    bt::Tensor z_conv; conv_fwd(blk.f2, u, N, Hin, W, bc.f2, z_conv);
    bt::Tensor z; bn_step(blk.bn_f2, z_conv, N, cout, Hz, W, train, z, bc.bn_f2);

    bt::Tensor a; favg_fwd(z, blk.avg_kernel, N, cout, Hz, W, a);
    bt::Tensor tt_conv; conv_fwd(blk.f1, a, N, 1, W, bc.f1, tt_conv);
    bt::Tensor tt; bn_step(blk.bn_f1, tt_conv, N, cout, 1, W, train, tt, bc.bn_f1);
    bc.tt_pre_relu = tt;
    bt::Tensor tt_relu; relu_fwd(tt, tt_relu);
    bt::Tensor tt_pw; conv_fwd(blk.pw_f1, tt_relu, N, 1, W, bc.pw_f1, tt_pw);

    bt::Tensor bcast; fbcast_fwd(tt_pw, blk.ones_kernel, N, cout, Hz, W, bcast);
    bt::Tensor zc = z;                 // deep copy
    bt::add_inplace(zc, bcast);
    bc.zc_pre_relu = zc;
    bt::Tensor y; relu_fwd(zc, y);

    if (blk.is_transition) {
        bt::Tensor r; conv_fwd(blk.res, X, N, Hin, W, bc.res, r);
        bt::Tensor rbn; bn_step(blk.bn_res, r, N, cout, Hz, W, train, rbn, bc.bn_res);
        bt::add_inplace(y, rbn);
    } else {
        bt::add_inplace(y, X);         // identity residual (cout==cin, Hz==Hin)
    }
    Y = std::move(y);
}

// One BC block backward. dY: (N, cout*Hz*W). Accumulates grads via ts; dX overwritten.
void block_bwd(BCBlock& blk, BlockCache& bc, const bt::Tensor& dY, int N, int W,
               BcResnet2dTrainState& ts, const std::string& p, bt::Tensor& dX) {
    const int cout = blk.cout, Hz = blk.Hz, Hin = bc.Hin;

    // y = relu(zc) + res  →  relu branch gets dY; residual branch gets dY.
    bt::Tensor d_relu_in; relu_bwd(bc.zc_pre_relu, dY, d_relu_in);   // d(zc)
    // zc = z + broadcast(tt_pw) → both summands receive d_relu_in.
    bt::Tensor d_tt_pw; fbcast_bwd(blk.ones_kernel, d_relu_in, N, cout, Hz, W, d_tt_pw);
    bt::Tensor d_tt_relu; conv_bwd(blk.pw_f1, bc.pw_f1, d_tt_pw, N, 1, W, /*Hout=*/1,
                                   d_tt_relu, ts.grad(p + ".pw_f1.W"), ts.grad(p + ".pw_f1.b"));
    bt::Tensor d_tt; relu_bwd(bc.tt_pre_relu, d_tt_relu, d_tt);
    bt::Tensor d_f1_conv, dg1, db1;
    bn_bwd(blk.bn_f1, bc.bn_f1, d_tt, N, cout, 1, W, d_f1_conv, dg1, db1);
    bt::add_inplace(ts.grad(p + ".bn_f1.gamma"), dg1);
    bt::add_inplace(ts.grad(p + ".bn_f1.beta"),  db1);
    bt::Tensor d_a; conv_bwd(blk.f1, bc.f1, d_f1_conv, N, 1, W, /*Hout=*/1,
                             d_a, ts.grad(p + ".f1.W"), ts.grad(p + ".f1.b"));
    bt::Tensor d_z_avg; favg_bwd(blk.avg_kernel, d_a, N, cout, Hz, W, d_z_avg);

    bt::Tensor dz = d_relu_in;         // deep copy of the add-branch grad
    bt::add_inplace(dz, d_z_avg);

    bt::Tensor d_f2_conv, dg2, db2;
    bn_bwd(blk.bn_f2, bc.bn_f2, dz, N, cout, Hz, W, d_f2_conv, dg2, db2);
    bt::add_inplace(ts.grad(p + ".bn_f2.gamma"), dg2);
    bt::add_inplace(ts.grad(p + ".bn_f2.beta"),  db2);
    bt::Tensor d_u; conv_bwd(blk.f2, bc.f2, d_f2_conv, N, Hin, W, /*Hout=*/Hz,
                             d_u, ts.grad(p + ".f2.W"), ts.grad(p + ".f2.b"));

    bt::Tensor dX_main;
    if (blk.is_transition) {
        bt::Tensor d_in_bn; relu_bwd(bc.in_pre_relu, d_u, d_in_bn);
        bt::Tensor d_pw_conv, dgi, dbi;
        bn_bwd(blk.bn_in, bc.bn_in, d_in_bn, N, cout, Hin, W, d_pw_conv, dgi, dbi);
        bt::add_inplace(ts.grad(p + ".bn_in.gamma"), dgi);
        bt::add_inplace(ts.grad(p + ".bn_in.beta"),  dbi);
        conv_bwd(blk.pw_in, bc.pw_in, d_pw_conv, N, Hin, W, /*Hout=*/Hin,
                 dX_main, ts.grad(p + ".pw_in.W"), ts.grad(p + ".pw_in.b"));
    } else {
        dX_main = d_u;
    }

    bt::Tensor dX_res;
    if (blk.is_transition) {
        bt::Tensor d_res_conv, dgr, dbr;
        bn_bwd(blk.bn_res, bc.bn_res, dY, N, cout, Hz, W, d_res_conv, dgr, dbr);
        bt::add_inplace(ts.grad(p + ".bn_res.gamma"), dgr);
        bt::add_inplace(ts.grad(p + ".bn_res.beta"),  dbr);
        conv_bwd(blk.res, bc.res, d_res_conv, N, Hin, W, /*Hout=*/Hz,
                 dX_res, ts.grad(p + ".res.W"), ts.grad(p + ".res.b"));
    } else {
        dX_res = dY;                   // identity residual
    }

    dX = std::move(dX_main);
    bt::add_inplace(dX, dX_res);
}

void register_params(BcResnet2dTrainState& ts, BcResnet2d::Impl& m) {
    auto add_conv = [&](const std::string& p, Conv2dLayer& c) {
        ts.add(&c.W, p + ".W"); ts.add(&c.b, p + ".b");
    };
    auto add_bn = [&](const std::string& p, BatchNorm2d& bn) {
        ts.add(&bn.gamma, p + ".gamma"); ts.add(&bn.beta, p + ".beta");
    };
    add_conv("stem", m.stem); add_bn("bn_stem", m.bn_stem);
    for (std::size_t i = 0; i < m.blocks.size(); ++i) {
        BCBlock& blk = m.blocks[i];
        const std::string p = "blk" + std::to_string(i);
        if (blk.is_transition) { add_conv(p + ".pw_in", blk.pw_in); add_bn(p + ".bn_in", blk.bn_in); }
        add_conv(p + ".f2", blk.f2); add_bn(p + ".bn_f2", blk.bn_f2);
        add_conv(p + ".f1", blk.f1); add_bn(p + ".bn_f1", blk.bn_f1);
        add_conv(p + ".pw_f1", blk.pw_f1);
        if (blk.is_transition) { add_conv(p + ".res", blk.res); add_bn(p + ".bn_res", blk.bn_res); }
    }
    ts.add(&m.head_W, "head.W"); ts.add(&m.head_b, "head.b");
}

}  // namespace

// Shared: forward to per-clip logits (B,1). train=true uses batch BN + fills the
// cache; train=false uses running-stat BN (no caches). Returns logits on device.
static bt::Tensor forward_to_logits(BcResnet2d::Impl& I, const bt::Tensor& feats,
                                    int B, int T, bool train, TrainCache2d* tc) {
    const int nm = I.cfg.n_mels;
    Conv2dCache stem_scratch; Bn2dCache bn_scratch;
    Conv2dCache& stem_cc = train ? tc->stem : stem_scratch;
    Bn2dCache&   bn_cc   = train ? tc->bn_stem : bn_scratch;
    bt::Tensor h;
    {
        bt::Tensor sc; conv_fwd(I.stem, feats, B, nm, T, stem_cc, sc);
        bt::Tensor sb; bn_step(I.bn_stem, sc, B, I.cfg.c_stem, I.stem_Hout, T, train, sb, bn_cc);
        if (train) tc->stem_pre_relu = sb;
        relu_fwd(sb, h);
    }
    int Hin = I.stem_Hout;
    if (train) tc->blocks.resize(I.blocks.size());
    BlockCache blk_scratch;
    for (std::size_t i = 0; i < I.blocks.size(); ++i) {
        bt::Tensor y;
        BlockCache& bc = train ? tc->blocks[i] : blk_scratch;
        block_fwd(I.blocks[i], h, B, Hin, T, train, bc, y);
        h = std::move(y);
        Hin = I.blocks[i].Hz;
    }
    // Head: freq-avg → (B, c_last*1*T) → GAP over T → (B, c_last).
    bt::Tensor favg; favg_fwd(h, I.head_avg_kernel, B, I.c_last, I.head_H, T, favg);
    std::vector<float> ones_T(static_cast<std::size_t>(T), 1.0f / static_cast<float>(T));
    bt::Tensor meanT = bt::Tensor::from_host_on(I.device, ones_T.data(), T, 1);
    bt::Tensor favg_view = bt::Tensor::view(I.device, favg.data, B * I.c_last, T, bt::Dtype::FP32);
    bt::Tensor pooled; bt::matmul(favg_view, meanT, pooled);   // (B*c_last,1)
    pooled.rows = B; pooled.cols = I.c_last;
    if (train) tc->pooled = pooled;
    bt::Tensor logits = bt::Tensor::zeros_on(I.device, B, 1, bt::Dtype::FP32);
    bt::linear_forward_batched(I.head_W, I.head_b, pooled, logits);
    return logits;
}

float BcResnet2d::train_step(const bt::Tensor& feats_batch, const bt::Tensor& labels,
                             int B, int T, float lr, float pos_weight) {
    if (impl_->fused) fail("BcResnet2d::train_step", "cannot train a fused model");
    if (feats_batch.device != impl_->device) fail("BcResnet2d::train_step", "batch device mismatch");
    if (feats_batch.rows != B || feats_batch.cols != impl_->cfg.n_mels * T)
        fail("BcResnet2d::train_step", "feats_batch shape mismatch");
    if (labels.rows != B || labels.cols != 1)
        fail("BcResnet2d::train_step", "labels shape mismatch");

    if (!impl_->train_state) {
        impl_->train_state = std::make_unique<BcResnet2dTrainState>();
        register_params(*impl_->train_state, *impl_);
    }
    auto& ts = *impl_->train_state;
    auto& I  = *impl_;
    ts.zero_grads();

    TrainCache2d tc;
    bt::Tensor logits = forward_to_logits(I, feats_batch, B, T, /*train=*/true, &tc);

    bt::Tensor probs    = bt::Tensor::zeros_on(I.device, B, 1, bt::Dtype::FP32);
    bt::Tensor dLogits  = bt::Tensor::zeros_on(I.device, B, 1, bt::Dtype::FP32);
    bt::Tensor loss_per = bt::Tensor::zeros_on(I.device, B, 1, bt::Dtype::FP32);
    bt::bce_with_logits_fused_batched(logits, labels, nullptr, pos_weight, probs, dLogits, loss_per);

    bt::Tensor loss_host = loss_per.to(bt::Device::CPU);
    float total = 0.0f;
    { const float* l = loss_host.host_f32(); for (int i = 0; i < B; ++i) total += l[i]; }
    const float mean_loss = total / static_cast<float>(B);
    bt::scale_inplace(dLogits, 1.0f / static_cast<float>(B));

    // Head backward.
    bt::Tensor dPooled = bt::Tensor::zeros_on(I.device, B, I.c_last, bt::Dtype::FP32);
    bt::linear_backward_batched(I.head_W, tc.pooled, dLogits, dPooled,
                                ts.grad("head.W"), ts.grad("head.b"));
    // GAP backward: scatter dPooled[b,c]/T across T frames → dFavg (B, c_last*T).
    std::vector<float> bcastT(static_cast<std::size_t>(T), 1.0f / static_cast<float>(T));
    bt::Tensor bT = bt::Tensor::from_host_on(I.device, bcastT.data(), 1, T);
    bt::Tensor dPooled_view = bt::Tensor::view(I.device, dPooled.data, B * I.c_last, 1, bt::Dtype::FP32);
    bt::Tensor dFavg; bt::matmul(dPooled_view, bT, dFavg);   // (B*c_last, T)
    dFavg.rows = B; dFavg.cols = I.c_last * T;
    // Head freq-avg backward → dh (B, c_last*head_H*T).
    bt::Tensor dh; favg_bwd(I.head_avg_kernel, dFavg, B, I.c_last, I.head_H, T, dh);

    // Blocks backward (reverse order).
    for (int i = static_cast<int>(I.blocks.size()) - 1; i >= 0; --i) {
        bt::Tensor dXb;
        block_bwd(I.blocks[i], tc.blocks[i], dh, B, T, ts, "blk" + std::to_string(i), dXb);
        dh = std::move(dXb);
    }
    // Stem ReLU → BN → conv backward.
    bt::Tensor dStemBn; relu_bwd(tc.stem_pre_relu, dh, dStemBn);
    bt::Tensor dStemConv, dsg, dsb;
    bn_bwd(I.bn_stem, tc.bn_stem, dStemBn, B, I.cfg.c_stem, I.stem_Hout, T, dStemConv, dsg, dsb);
    bt::add_inplace(ts.grad("bn_stem.gamma"), dsg);
    bt::add_inplace(ts.grad("bn_stem.beta"),  dsb);
    bt::Tensor dInput;
    conv_bwd(I.stem, tc.stem, dStemConv, B, I.cfg.n_mels, T, /*Hout=*/I.stem_Hout,
             dInput, ts.grad("stem.W"), ts.grad("stem.b"));

    ts.adam(lr);
    return mean_loss;
}

BcResnet2d::EvalMetrics BcResnet2d::eval_step(const bt::Tensor& feats_batch,
                                              const bt::Tensor& labels,
                                              int B, int T, float pos_weight) {
    if (impl_->fused) fail("BcResnet2d::eval_step", "fused models use forward()");
    if (feats_batch.device != impl_->device) fail("BcResnet2d::eval_step", "batch device mismatch");
    auto& I = *impl_;
    bt::Tensor logits = forward_to_logits(I, feats_batch, B, T, /*train=*/false, nullptr);

    bt::Tensor probs    = bt::Tensor::zeros_on(I.device, B, 1, bt::Dtype::FP32);
    bt::Tensor dLogits  = bt::Tensor::zeros_on(I.device, B, 1, bt::Dtype::FP32);
    bt::Tensor loss_per = bt::Tensor::zeros_on(I.device, B, 1, bt::Dtype::FP32);
    bt::bce_with_logits_fused_batched(logits, labels, nullptr, pos_weight, probs, dLogits, loss_per);

    bt::Tensor loss_host  = loss_per.to(bt::Device::CPU);
    bt::Tensor probs_host = probs.to(bt::Device::CPU);
    bt::Tensor lab_host   = labels.to(bt::Device::CPU);
    EvalMetrics m; m.n = B;
    const float* lp = loss_host.host_f32();
    const float* pp = probs_host.host_f32();
    const float* yl = lab_host.host_f32();
    float total = 0.0f; int correct = 0, pos = 0, neg = 0, fn = 0, fp = 0;
    for (int b = 0; b < B; ++b) {
        total += lp[b];
        const bool pred = pp[b] >= 0.5f;
        const bool truth = yl[b] >= 0.5f;
        if (pred == truth) ++correct;
        if (truth) { ++pos; if (!pred) ++fn; }
        else       { ++neg; if (pred)  ++fp; }
    }
    m.loss = total / static_cast<float>(B);
    m.accuracy = static_cast<float>(correct) / static_cast<float>(B);
    m.frr = pos ? static_cast<float>(fn) / static_cast<float>(pos) : 0.0f;
    m.fpr = neg ? static_cast<float>(fp) / static_cast<float>(neg) : 0.0f;
    return m;
}

// ─── Test/debug seam ──────────────────────────────────────────────────────────
namespace {
void ensure_train_state(BcResnet2d::Impl& I, std::unique_ptr<BcResnet2dTrainState>& ts) {
    if (!ts) { ts = std::make_unique<BcResnet2dTrainState>(); register_params(*ts, I); }
}
}  // namespace

std::vector<std::pair<std::string, int>> BcResnet2d::debug_params() const {
    ensure_train_state(*impl_, impl_->train_state);
    std::vector<std::pair<std::string, int>> out;
    for (auto& s : impl_->train_state->slots)
        out.push_back({s.name, s.param->size()});
    return out;
}
float BcResnet2d::debug_get_param(const std::string& name, int idx) const {
    ensure_train_state(*impl_, impl_->train_state);
    auto* s = impl_->train_state->find(name);
    if (!s) fail("BcResnet2d::debug_get_param", "no param '" + name + "'");
    bt::Tensor h = s->param->to(bt::Device::CPU);
    return h.host_f32()[idx];
}
void BcResnet2d::debug_set_param(const std::string& name, int idx, float value) {
    ensure_train_state(*impl_, impl_->train_state);
    auto* s = impl_->train_state->find(name);
    if (!s) fail("BcResnet2d::debug_set_param", "no param '" + name + "'");
    bt::Tensor h = s->param->to(bt::Device::CPU);
    h.host_f32_mut()[idx] = value;
    *s->param = h.to(s->param->device);
}
float BcResnet2d::debug_grad(const std::string& name, int idx) const {
    if (!impl_->train_state) fail("BcResnet2d::debug_grad", "no train state (run train_step first)");
    auto* s = impl_->train_state->find(name);
    if (!s) fail("BcResnet2d::debug_grad", "no grad '" + name + "'");
    bt::Tensor h = s->grad.to(bt::Device::CPU);
    return h.host_f32()[idx];
}

}  // namespace brosoundml
