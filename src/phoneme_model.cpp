#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

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

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
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
    bt::Tensor cache;           // streaming time-history (1, Cin*Hcache*padLeftW)
    int Hcache = 0;
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

void ensure_cache(bt::Tensor& t, bt::Device dev, int cols) {
    if (cols <= 0) { t = bt::Tensor{}; return; }
    if (t.device != dev || t.rows != 1 || t.cols != cols ||
        t.dtype != bt::Dtype::FP32) {
        t = bt::Tensor::zeros_on(dev, 1, cols, bt::Dtype::FP32);
    } else {
        t.zero();
    }
}

void init_conv(Conv2dLayer& c, int cin, int cout, int kH, int kW,
               int sH, int padH, int dilW, int groups, int Hin,
               bt::Device dev) {
    c.Cin = cin; c.Cout = cout; c.kH = kH; c.kW = kW;
    c.sH = sH; c.padH = padH; c.dilW = dilW; c.groups = groups;
    c.padLeftW = dilW * (kW - 1);
    c.Hcache = Hin;
    const int win = (cin / groups) * kH * kW;
    c.W = bt::Tensor::zeros_on(dev, cout, win, bt::Dtype::FP32);
    c.b = bt::Tensor::zeros_on(dev, cout, 1, bt::Dtype::FP32);
    ensure_cache(c.cache, dev, c.padLeftW > 0 ? cin * Hin * c.padLeftW : 0);
}

void init_bn(BatchNorm2d& bn, int C, bt::Device dev, float eps) {
    bn.channels = C; bn.eps = eps; bn.present = true;
    std::vector<float> ones(static_cast<std::size_t>(C), 1.0f);
    bn.gamma = bt::Tensor::from_host_on(dev, ones.data(), C, 1);
    bn.beta  = bt::Tensor::zeros_on(dev, C, 1, bt::Dtype::FP32);
    bn.mean  = bt::Tensor::zeros_on(dev, C, 1, bt::Dtype::FP32);
    bn.var   = bt::Tensor::from_host_on(dev, ones.data(), C, 1);
}

void make_const_kernel(bt::Tensor& k, int C, int H, float val, bt::Device dev) {
    std::vector<float> h(static_cast<std::size_t>(C) * H, val);
    k = bt::Tensor::from_host_on(dev, h.data(), C, H);
}

// Causal-time conv forward on an NCHW (1, Cin*H*W) input (streaming-capable).
void conv_forward(Conv2dLayer& c, const bt::Tensor& X, int H, int W,
                  bool use_cache, bt::Tensor& Y) {
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

    const int Wp = W + c.padLeftW;
    bt::Tensor padded = bt::Tensor::zeros_on(dev, 1, rows * Wp, bt::Dtype::FP32);
    for (int r = 0; r < rows; ++r) {
        if (use_cache && c.cache.size() > 0)
            bt::copy_d2d(c.cache, r * c.padLeftW, padded, r * Wp, c.padLeftW);
        bt::copy_d2d(X, r * W, padded, r * Wp + c.padLeftW, W);
    }
    conv2d_forward(padded, c.W, &c.b, /*N=*/1, c.Cin, H, Wp, c.Cout,
                   c.kH, c.kW, c.sH, /*sw=*/1, c.padH, /*pad_w=*/0,
                   /*dil_h=*/1, c.dilW, c.groups, Y);

    if (use_cache && c.cache.size() > 0) {
        for (int r = 0; r < rows; ++r)
            bt::copy_d2d(padded, r * Wp + (Wp - c.padLeftW),
                         c.cache, r * c.padLeftW, c.padLeftW);
    }
}

void bn_inplace(BatchNorm2d& bn, bt::Tensor& X, int C, int H, int W) {
    if (!bn.present) return;
    bt::Tensor Y;
    bt::batch_norm_inference(X, bn.gamma, bn.beta, bn.mean, bn.var,
                             /*N=*/1, C, H, W, bn.eps, Y);
    X = std::move(Y);
}

void relu_into(const bt::Tensor& X, bt::Tensor& Y) {
    Y = bt::Tensor::empty_on(X.device, X.rows, X.cols, bt::Dtype::FP32);
    bt::relu_forward(X, Y);
}

void freq_avg(const bt::Tensor& X, const bt::Tensor& avg_kernel,
              int C, int H, int W, bt::Tensor& Y) {
    conv2d_forward(X, avg_kernel, /*bias=*/nullptr, /*N=*/1, C, H, W, C,
                   /*kH=*/H, /*kW=*/1, /*sh=*/1, /*sw=*/1, /*ph=*/0, /*pw=*/0,
                   /*dh=*/1, /*dw=*/1, /*groups=*/C, Y);
}

void freq_broadcast(const bt::Tensor& X1, const bt::Tensor& ones_kernel,
                    int C, int H, int W, bt::Tensor& Y) {
    bt::conv2d_backward_input(ones_kernel, X1, /*N=*/1, C, H, W, C,
                              /*kH=*/H, /*kW=*/1, /*sh=*/1, /*sw=*/1,
                              /*ph=*/0, /*pw=*/0, /*dh=*/1, /*dw=*/1,
                              /*groups=*/C, Y);
}

// ─── Binary format ('BPM1') ─────────────────────────────────────────────────
constexpr std::uint32_t kMagic   = 0x314D5042u;   // 'B''P''M''1'
constexpr std::uint32_t kVersion = 1u;

struct Reader {
    std::FILE* fp = nullptr; std::string path;
    ~Reader() { if (fp) std::fclose(fp); }
    void open(const std::string& p) {
        path = p; fp = std::fopen(p.c_str(), "rb");
        if (!fp) fail("phoneme_model::load", "could not open '" + p + "'");
    }
    void rd(void* d, std::size_t n) {
        if (std::fread(d, 1, n, fp) != n)
            fail("phoneme_model::load", "unexpected EOF in '" + path + "'");
    }
    std::uint32_t u32() { std::uint32_t v; rd(&v, 4); return v; }
    std::uint16_t u16() { std::uint16_t v; rd(&v, 2); return v; }
    std::uint8_t  u8 () { std::uint8_t  v; rd(&v, 1); return v; }
    float         f32() { float v; rd(&v, 4); return v; }
    std::string   str(std::size_t n) { std::string s(n, '\0'); rd(s.data(), n); return s; }
};
struct Writer {
    std::FILE* fp = nullptr; std::string path;
    ~Writer() { if (fp) std::fclose(fp); }
    void open(const std::string& p) {
        path = p; fp = std::fopen(p.c_str(), "wb");
        if (!fp) fail("phoneme_model::save", "could not open '" + p + "'");
    }
    void wr(const void* s, std::size_t n) {
        if (std::fwrite(s, 1, n, fp) != n)
            fail("phoneme_model::save", "short write to '" + path + "'");
    }
    void u32(std::uint32_t v) { wr(&v, 4); }
    void u16(std::uint16_t v) { wr(&v, 2); }
    void u8 (std::uint8_t  v) { wr(&v, 1); }
    void f32(float v)         { wr(&v, 4); }
};

void write_tensor(Writer& w, const std::string& name, const bt::Tensor& t) {
    bt::Tensor host = t.to(bt::Device::CPU);
    if (host.dtype != bt::Dtype::FP32) fail("phoneme_model::save", "non-FP32 tensor");
    w.u16(static_cast<std::uint16_t>(name.size()));
    w.wr(name.data(), name.size());
    w.u32(static_cast<std::uint32_t>(host.rows));
    w.u32(static_cast<std::uint32_t>(host.cols));
    w.wr(host.host_f32(), static_cast<std::size_t>(host.size()) * sizeof(float));
}

bt::Tensor read_tensor(Reader& r, const std::string& want, int rows, int cols,
                       bt::Device dev) {
    const std::uint16_t nl = r.u16();
    const std::string name = r.str(nl);
    const int rr = static_cast<int>(r.u32());
    const int cc = static_cast<int>(r.u32());
    if (name != want)
        fail("phoneme_model::load", "expected '" + want + "' got '" + name + "'");
    if (rr != rows || cc != cols)
        fail("phoneme_model::load", "tensor '" + name + "' shape mismatch");
    std::vector<float> buf(static_cast<std::size_t>(rr) * cc);
    r.rd(buf.data(), buf.size() * sizeof(float));
    return bt::Tensor::from_host_on(bt::Device::CPU, buf.data(), rr, cc).to(dev);
}

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

    int K() const { return cfg.num_classes; }
    int freq_pad(int k) const { return (k - 1) / 2; }

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
    }

    int rf_frames() const {
        int rf = (cfg.stem_kt - 1);
        for (int s = 0; s < PhonemeNetConfig::n_stages; ++s)
            for (int b = 0; b < cfg.n_blocks[s]; ++b)
                rf += cfg.tdil[s] * (cfg.dw_kt - 1);
        return rf + 1;
    }

    // Run one BC block (mirrors bc_resnet2d's run_block; inference path).
    void run_block(BCBlock& blk, const bt::Tensor& X, int Hin, int W,
                   bool use_cache, bt::Tensor& Y) {
        const int cout = blk.cout;
        bt::Tensor u;
        if (blk.is_transition) {
            bt::Tensor t;
            conv_forward(blk.pw_in, X, Hin, W, use_cache, t);
            bn_inplace(blk.bn_in, t, cout, Hin, W);
            relu_into(t, u);
        } else {
            u = X;
        }

        bt::Tensor z;
        conv_forward(blk.f2, u, Hin, W, use_cache, z);
        bn_inplace(blk.bn_f2, z, cout, blk.Hz, W);

        bt::Tensor a;
        freq_avg(z, blk.avg_kernel, cout, blk.Hz, W, a);
        bt::Tensor tt;
        conv_forward(blk.f1, a, /*H=*/1, W, use_cache, tt);
        bn_inplace(blk.bn_f1, tt, cout, 1, W);
        bt::Tensor tt_relu; relu_into(tt, tt_relu);
        bt::Tensor tt_pw;
        conv_forward(blk.pw_f1, tt_relu, /*H=*/1, W, use_cache, tt_pw);

        bt::Tensor bc;
        freq_broadcast(tt_pw, blk.ones_kernel, cout, blk.Hz, W, bc);
        bt::add_inplace(z, bc);
        bt::Tensor y; relu_into(z, y);

        bt::Tensor res;
        if (blk.is_transition) {
            conv_forward(blk.res, X, Hin, W, use_cache, res);
            bn_inplace(blk.bn_res, res, cout, blk.Hz, W);
        } else {
            res = X;
        }
        bt::add_inplace(y, res);
        Y = std::move(y);
    }

    // Inference forward producing per-frame logits (T, K). use_cache=false is the
    // one-shot path; true streams (persistent causal caches). The head is
    // pointwise in time, so the two paths agree once the trunk caches do.
    void forward_core(const bt::Tensor& feats, bool use_cache,
                      bt::Tensor& out_logits) {
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
        conv_forward(stem, x, cfg.n_mels, T, use_cache, y_stem);
        bn_inplace(bn_stem, y_stem, cfg.c_stem, stem_Hout, T);
        bt::Tensor h; relu_into(y_stem, h);

        int Hin = stem_Hout;
        for (auto& blk : blocks) {
            bt::Tensor y;
            run_block(blk, h, Hin, T, use_cache, y);
            h = std::move(y);
            Hin = blk.Hz;
        }

        // Head: freq-avg → (1, c_last*T) → 1x1 conv → (1, K*T) → (T, K).
        bt::Tensor favg;
        freq_avg(h, head_avg_kernel, c_last, head_H, T, favg);   // (1, c_last*T)
        bt::Tensor hc;
        conv_forward(head, favg, /*H=*/1, T, /*use_cache=*/false, hc);  // (1, K*T)
        bt::nchw_to_sequence(hc, /*N=*/1, /*C=*/Kc, /*H=*/1, /*W=*/T, out_logits);
    }
};

// ─── Public surface ───────────────────────────────────────────────────────────

PhonemeNet::PhonemeNet() : impl_(std::make_unique<Impl>()) {}
PhonemeNet::~PhonemeNet() = default;
PhonemeNet::PhonemeNet(PhonemeNet&&) noexcept = default;
PhonemeNet& PhonemeNet::operator=(PhonemeNet&&) noexcept = default;

PhonemeNet PhonemeNet::make(const PhonemeNetConfig& cfg,
                            const PhonemeClassMap& class_map, bt::Device device) {
    if (class_map.num_classes <= 0)
        fail("PhonemeNet::make", "class map has no classes (K <= 0)");
    PhonemeNet m;
    m.impl_->cfg = cfg;
    m.impl_->cfg.num_classes = class_map.num_classes;
    m.impl_->class_map = class_map;
    m.impl_->class_map.rebuild_inverse();
    m.impl_->device = device;
    m.impl_->fused = false;
    m.impl_->build();
    return m;
}

const PhonemeNetConfig& PhonemeNet::config()    const { return impl_->cfg; }
const PhonemeClassMap&  PhonemeNet::class_map() const { return impl_->class_map; }
bt::Device              PhonemeNet::device()    const { return impl_->device; }
bool                    PhonemeNet::fused()     const { return impl_->fused; }
int PhonemeNet::receptive_field_frames() const { return impl_->rf_frames(); }

int PhonemeNet::param_count() const {
    int n = 0;
    auto conv = [](const Conv2dLayer& c) { return c.Cout * (c.Cin / c.groups) * c.kH * c.kW + c.Cout; };
    auto bn = [&](const BatchNorm2d& b) { return impl_->fused ? 0 : 2 * b.channels; };
    n += conv(impl_->stem) + bn(impl_->bn_stem);
    for (auto& blk : impl_->blocks) {
        if (blk.is_transition) { n += conv(blk.pw_in) + bn(blk.bn_in); n += conv(blk.res) + bn(blk.bn_res); }
        n += conv(blk.f2) + bn(blk.bn_f2);
        n += conv(blk.f1) + bn(blk.bn_f1) + conv(blk.pw_f1);
    }
    n += conv(impl_->head);
    return n;
}

void PhonemeNet::reset_streaming_state() {
    auto rc = [](Conv2dLayer& c) { if (c.cache.size() > 0) c.cache.zero(); };
    rc(impl_->stem);
    for (auto& blk : impl_->blocks) {
        if (blk.is_transition) { rc(blk.pw_in); rc(blk.res); }
        rc(blk.f2); rc(blk.f1); rc(blk.pw_f1);
    }
}

void PhonemeNet::forward(const bt::Tensor& feats, bt::Tensor& out_logits) const {
    auto& mut = const_cast<Impl&>(*impl_);
    mut.forward_core(feats, /*use_cache=*/false, out_logits);
}

void PhonemeNet::forward_streaming(const bt::Tensor& new_feats, bt::Tensor& out) {
    impl_->forward_core(new_feats, /*use_cache=*/true, out);
}

// ─── BN fold ──────────────────────────────────────────────────────────────────
namespace {
void fuse_one(Conv2dLayer& c, BatchNorm2d& bn) {
    if (!bn.present) return;
    const bt::Device tgt = c.W.device;
    bt::Tensor Wh = c.W.to(bt::Device::CPU), bh = c.b.to(bt::Device::CPU);
    bt::Tensor gh = bn.gamma.to(bt::Device::CPU), eh = bn.beta.to(bt::Device::CPU);
    bt::Tensor mh = bn.mean.to(bt::Device::CPU), vh = bn.var.to(bt::Device::CPU);
    const int C = c.Cout, row = Wh.cols;
    float* W = Wh.host_f32_mut(); float* B = bh.host_f32_mut();
    const float* g = gh.host_f32(); const float* be = eh.host_f32();
    const float* m = mh.host_f32(); const float* v = vh.host_f32();
    for (int co = 0; co < C; ++co) {
        const float a = g[co] / std::sqrt(v[co] + bn.eps);
        for (int j = 0; j < row; ++j) W[co * row + j] *= a;
        B[co] = (B[co] - m[co]) * a + be[co];
    }
    c.W = Wh.to(tgt); c.b = bh.to(tgt);
    bn.present = false;
    bn.gamma = bt::Tensor{}; bn.beta = bt::Tensor{};
    bn.mean = bt::Tensor{}; bn.var = bt::Tensor{};
}
}  // namespace

void PhonemeNet::fuse_bn() {
    if (impl_->fused) return;
    fuse_one(impl_->stem, impl_->bn_stem);
    for (auto& blk : impl_->blocks) {
        if (blk.is_transition) { fuse_one(blk.pw_in, blk.bn_in); fuse_one(blk.res, blk.bn_res); }
        fuse_one(blk.f2, blk.bn_f2);
        fuse_one(blk.f1, blk.bn_f1);
    }
    impl_->fused = true;
}

// ─── Weights I/O ────────────────────────────────────────────────────────────
namespace {
void for_each_tensor(PhonemeNet::Impl& m, bool include_bn,
                     std::vector<std::pair<std::string, bt::Tensor*>>& out) {
    auto add = [&](const std::string& n, bt::Tensor& t) { out.push_back({n, &t}); };
    auto add_conv = [&](const std::string& p, Conv2dLayer& c) {
        add(p + ".W", c.W); add(p + ".b", c.b);
    };
    auto add_bn = [&](const std::string& p, BatchNorm2d& bn) {
        if (!include_bn || !bn.present) return;
        add(p + ".gamma", bn.gamma); add(p + ".beta", bn.beta);
        add(p + ".mean", bn.mean);   add(p + ".var", bn.var);
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
    add_conv("head", m.head);
}
}  // namespace

void PhonemeNet::save(const std::string& path, bool fused) const {
    if (fused && !impl_->fused) fail("PhonemeNet::save", "fuse_bn() has not run");
    if (!fused && impl_->fused) fail("PhonemeNet::save", "BN already fused");
    Writer w; w.open(path);
    w.u32(kMagic); w.u32(kVersion);
    const auto& c = impl_->cfg;
    // Front-end framing params.
    w.u32(c.sample_rate); w.u32(c.n_fft); w.u32(c.win_length); w.u32(c.hop_length);
    // Trunk hyperparameters.
    w.u32(c.n_mels); w.u32(c.c_stem); w.u32(c.stem_kf); w.u32(c.stem_kt); w.u32(c.stem_sf);
    for (int s = 0; s < PhonemeNetConfig::n_stages; ++s) {
        w.u32(c.c[s]); w.u32(c.fstride[s]); w.u32(c.tdil[s]); w.u32(c.n_blocks[s]);
    }
    w.u32(c.dw_kf); w.u32(c.dw_kt); w.u32(c.ssn_subbands); w.f32(c.bn_eps);
    w.u32(static_cast<std::uint32_t>(c.num_classes));
    w.u8(fused ? 1u : 0u);
    // Embedded class map.
    write_classmap(w.fp, impl_->class_map);
    // Weights.
    std::vector<std::pair<std::string, bt::Tensor*>> ts;
    for_each_tensor(*impl_, /*include_bn=*/!fused, ts);
    w.u32(static_cast<std::uint32_t>(ts.size()));
    for (auto& [n, t] : ts) write_tensor(w, n, *t);
}

PhonemeNet PhonemeNet::load(const std::string& path, bt::Device device) {
    Reader r; r.open(path);
    if (r.u32() != kMagic)   fail("PhonemeNet::load", "bad magic");
    if (r.u32() != kVersion) fail("PhonemeNet::load", "unsupported version");
    PhonemeNetConfig c;
    c.sample_rate = (int)r.u32(); c.n_fft = (int)r.u32();
    c.win_length = (int)r.u32(); c.hop_length = (int)r.u32();
    c.n_mels = (int)r.u32(); c.c_stem = (int)r.u32(); c.stem_kf = (int)r.u32();
    c.stem_kt = (int)r.u32(); c.stem_sf = (int)r.u32();
    for (int s = 0; s < PhonemeNetConfig::n_stages; ++s) {
        c.c[s] = (int)r.u32(); c.fstride[s] = (int)r.u32();
        c.tdil[s] = (int)r.u32(); c.n_blocks[s] = (int)r.u32();
    }
    c.dw_kf = (int)r.u32(); c.dw_kt = (int)r.u32(); c.ssn_subbands = (int)r.u32();
    c.bn_eps = r.f32();
    c.num_classes = (int)r.u32();
    const bool on_disk_fused = (r.u8() != 0u);
    PhonemeClassMap cm = read_classmap(r.fp);
    if (cm.num_classes != c.num_classes)
        fail("PhonemeNet::load", "class-map K mismatch with header");
    const std::uint32_t num = r.u32();

    PhonemeNet m = PhonemeNet::make(c, cm, device);
    m.impl_->fused = on_disk_fused;
    if (on_disk_fused) {
        auto clr = [](BatchNorm2d& bn) {
            bn.present = false; bn.gamma = bt::Tensor{}; bn.beta = bt::Tensor{};
            bn.mean = bt::Tensor{}; bn.var = bt::Tensor{};
        };
        clr(m.impl_->bn_stem);
        for (auto& blk : m.impl_->blocks) {
            if (blk.is_transition) { clr(blk.bn_in); clr(blk.bn_res); }
            clr(blk.bn_f2); clr(blk.bn_f1);
        }
    }
    std::vector<std::pair<std::string, bt::Tensor*>> ts;
    for_each_tensor(*m.impl_, /*include_bn=*/!on_disk_fused, ts);
    if (ts.size() != num)
        fail("PhonemeNet::load", "tensor count mismatch");
    for (auto& [n, t] : ts) *t = read_tensor(r, n, t->rows, t->cols, device);
    return m;
}

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

void PhonemeNet::xavier_init_weights(std::uint64_t seed) {
    if (impl_->fused) fail("PhonemeNet::xavier_init_weights", "model is fused");
    std::uint64_t s = seed ? seed : 0x123456789ull;
    auto cw = [&](Conv2dLayer& c) { xavier_fill(c.W, s); };
    cw(impl_->stem);
    for (auto& blk : impl_->blocks) {
        if (blk.is_transition) { cw(blk.pw_in); cw(blk.res); }
        cw(blk.f2); cw(blk.f1); cw(blk.pw_f1);
    }
    cw(impl_->head);
}

// ─── Training surface ─────────────────────────────────────────────────────────
//
// Hand-rolled backward through the 2D BC-ResNet trunk, mirroring forward_core()
// in batched (N=B) layout with BN in train-mode, every op device-dispatched
// through brotensor. The phoneme head is a per-frame 1x1 conv (c_last -> K)
// followed by a framewise softmax cross-entropy over the (B*T, K) logits, with a
// per-frame class weight folded into the gradient.

namespace {

struct Conv2dCache { bt::Tensor X_padded; };
struct Bn2dCache   { bt::Tensor X_in, saved_mean, saved_rstd; };

struct BlockCache {
    int Hin = 0;
    Conv2dCache pw_in; Bn2dCache bn_in; bt::Tensor in_pre_relu;
    Conv2dCache f2;    Bn2dCache bn_f2;
    bt::Tensor  a;
    Conv2dCache f1;    Bn2dCache bn_f1; bt::Tensor tt_pre_relu;
    Conv2dCache pw_f1;
    bt::Tensor  zc_pre_relu;
    Conv2dCache res;   Bn2dCache bn_res;
};

struct TrainCache {
    Conv2dCache stem;  Bn2dCache bn_stem; bt::Tensor stem_pre_relu;
    std::vector<BlockCache> blocks;
    Conv2dCache head;            // per-frame 1x1 head conv (favg → logits)
};

bt::Tensor left_pad_w(const bt::Tensor& X, int N, int C, int H, int W, int pad) {
    if (pad == 0) return X;
    bt::Tensor Y = bt::Tensor::zeros_on(X.device, N, C * H * (W + pad), bt::Dtype::FP32);
    const int rows = N * C * H;
    for (int r = 0; r < rows; ++r)
        bt::copy_d2d(X, r * W, Y, r * (W + pad) + pad, W);
    return Y;
}
bt::Tensor strip_left_pad_w(const bt::Tensor& X, int N, int C, int H, int W, int pad) {
    if (pad == 0) return X;
    bt::Tensor Y = bt::Tensor::empty_on(X.device, N, C * H * W, bt::Dtype::FP32);
    const int rows = N * C * H;
    for (int r = 0; r < rows; ++r)
        bt::copy_d2d(X, r * (W + pad) + pad, Y, r * W, W);
    return Y;
}

void conv_fwd(const Conv2dLayer& c, const bt::Tensor& X, int N, int Hin, int W,
              Conv2dCache& cache, bt::Tensor& Y) {
    cache.X_padded = left_pad_w(X, N, c.Cin, Hin, W, c.padLeftW);
    const int Wp = W + c.padLeftW;
    conv2d_forward(cache.X_padded, c.W, &c.b, N, c.Cin, Hin, Wp, c.Cout,
                   c.kH, c.kW, c.sH, /*sw=*/1, c.padH, /*pad_w=*/0,
                   /*dh=*/1, c.dilW, c.groups, Y);
}
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

void bn_step(BatchNorm2d& bn, const bt::Tensor& X, int N, int C, int H, int W,
             bool train, bt::Tensor& Y, Bn2dCache& cache) {
    if (train) bn_fwd(bn, X, N, C, H, W, Y, cache);
    else       bn_eval(bn, X, N, C, H, W, Y);
}

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
    bt::Tensor zc = z;
    bt::add_inplace(zc, bcast);
    bc.zc_pre_relu = zc;
    bt::Tensor y; relu_fwd(zc, y);

    if (blk.is_transition) {
        bt::Tensor rr; conv_fwd(blk.res, X, N, Hin, W, bc.res, rr);
        bt::Tensor rbn; bn_step(blk.bn_res, rr, N, cout, Hz, W, train, rbn, bc.bn_res);
        bt::add_inplace(y, rbn);
    } else {
        bt::add_inplace(y, X);
    }
    Y = std::move(y);
}

void block_bwd(BCBlock& blk, BlockCache& bc, const bt::Tensor& dY, int N, int W,
               PhonemeNetTrainState& ts, const std::string& p, bt::Tensor& dX) {
    const int cout = blk.cout, Hz = blk.Hz, Hin = bc.Hin;

    bt::Tensor d_relu_in; relu_bwd(bc.zc_pre_relu, dY, d_relu_in);
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

    bt::Tensor dz = d_relu_in;
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
        dX_res = dY;
    }

    dX = std::move(dX_main);
    bt::add_inplace(dX, dX_res);
}

void register_params(PhonemeNetTrainState& ts, PhonemeNet::Impl& m) {
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
    add_conv("head", m.head);
}

// Forward to per-frame logits (B*T, K). train=true uses batch BN + caches.
bt::Tensor forward_to_logits(PhonemeNet::Impl& I, const bt::Tensor& feats,
                             int B, int T, bool train, TrainCache* tc) {
    const int nm = I.cfg.n_mels, Kc = I.K();
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
    // Head: freq-avg → (B, c_last*T) → 1x1 conv → (B, K*T) → (B*T, K).
    bt::Tensor favg; favg_fwd(h, I.head_avg_kernel, B, I.c_last, I.head_H, T, favg);
    Conv2dCache head_scratch;
    Conv2dCache& head_cc = train ? tc->head : head_scratch;
    bt::Tensor hc; conv_fwd(I.head, favg, B, /*Hin=*/1, T, head_cc, hc);  // (B, K*T)
    bt::Tensor logits; bt::nchw_to_sequence(hc, B, Kc, 1, T, logits);     // (B*T, K)
    return logits;
}

}  // namespace

float PhonemeNet::train_step(const bt::Tensor& feats_batch, const bt::Tensor& labels,
                             int B, int T, float lr,
                             const std::vector<float>& class_weights) {
    auto& I = *impl_;
    const int Kc = I.K();
    if (I.fused) fail("PhonemeNet::train_step", "cannot train a fused model");
    if (feats_batch.device != I.device) fail("PhonemeNet::train_step", "batch device mismatch");
    if (feats_batch.rows != B || feats_batch.cols != I.cfg.n_mels * T)
        fail("PhonemeNet::train_step", "feats_batch shape mismatch");
    if (labels.rows != B || labels.cols != T)
        fail("PhonemeNet::train_step", "labels shape mismatch (want (B,T))");
    if (!class_weights.empty() && static_cast<int>(class_weights.size()) != Kc)
        fail("PhonemeNet::train_step", "class_weights length != K");

    if (!I.train_state) {
        I.train_state = std::make_unique<PhonemeNetTrainState>();
        register_params(*I.train_state, I);
    }
    auto& ts = *I.train_state;
    ts.zero_grads();

    const int F = B * T;   // frames

    // Build one-hot targets (F, K) + per-frame weights from the integer labels.
    bt::Tensor lab_host = labels.to(bt::Device::CPU);
    const float* lp = lab_host.host_f32();
    std::vector<float> tgt(static_cast<std::size_t>(F) * Kc, 0.0f);
    std::vector<float> wf(static_cast<std::size_t>(F), 1.0f);
    double Z = 0.0;
    for (int f = 0; f < F; ++f) {
        const int cls = static_cast<int>(std::lround(lp[f]));
        if (cls < 0 || cls >= Kc) fail("PhonemeNet::train_step", "label out of range [0,K)");
        tgt[static_cast<std::size_t>(f) * Kc + cls] = 1.0f;
        const float w = class_weights.empty() ? 1.0f : class_weights[cls];
        wf[f] = w;
        Z += w;
    }
    if (Z <= 0.0) return 0.0f;   // every frame dropped — nothing to learn from
    bt::Tensor target = bt::Tensor::from_host_on(I.device, tgt.data(), F, Kc);

    TrainCache tc;
    bt::Tensor logits = forward_to_logits(I, feats_batch, B, T, /*train=*/true, &tc);

    bt::Tensor probs   = bt::Tensor::zeros_on(I.device, F, Kc, bt::Dtype::FP32);
    bt::Tensor dLogits = bt::Tensor::zeros_on(I.device, F, Kc, bt::Dtype::FP32);
    bt::Tensor loss_per = bt::Tensor::zeros_on(I.device, F, 1, bt::Dtype::FP32);
    bt::softmax_xent_fused_batched(logits, target, /*d_mask=*/nullptr,
                                   static_cast<const int*>(I.head_offsets.data),
                                   /*n_heads=*/1, probs, dLogits, loss_per);

    // Weighted-mean reduction: loss = (Σ w_f CE_f)/Z; dLogits row f *= w_f/Z.
    bt::Tensor loss_host = loss_per.to(bt::Device::CPU);
    const float* lhp = loss_host.host_f32();
    double total = 0.0;
    std::vector<float> wmat(static_cast<std::size_t>(F) * Kc);
    const float invZ = static_cast<float>(1.0 / Z);
    for (int f = 0; f < F; ++f) {
        total += static_cast<double>(wf[f]) * static_cast<double>(lhp[f]);
        const float scale = wf[f] * invZ;
        for (int k = 0; k < Kc; ++k) wmat[static_cast<std::size_t>(f) * Kc + k] = scale;
    }
    const float mean_loss = static_cast<float>(total / Z);
    bt::Tensor wt = bt::Tensor::from_host_on(I.device, wmat.data(), F, Kc);
    bt::mul_inplace(dLogits, wt);

    // Head backward: (F, K) → (B, K*T) → 1x1 conv → dFavg (B, c_last*T).
    bt::Tensor dHc; bt::sequence_to_nchw(dLogits, B, Kc, 1, T, dHc);   // (B, K*T)
    bt::Tensor dFavg;
    conv_bwd(I.head, tc.head, dHc, B, /*Hin=*/1, T, /*Hout=*/1,
             dFavg, ts.grad("head.W"), ts.grad("head.b"));
    // Head freq-avg backward → dh (B, c_last*head_H*T).
    bt::Tensor dh; favg_bwd(I.head_avg_kernel, dFavg, B, I.c_last, I.head_H, T, dh);

    for (int i = static_cast<int>(I.blocks.size()) - 1; i >= 0; --i) {
        bt::Tensor dXb;
        block_bwd(I.blocks[i], tc.blocks[i], dh, B, T, ts, "blk" + std::to_string(i), dXb);
        dh = std::move(dXb);
    }
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

PhonemeNet::EvalMetrics PhonemeNet::eval_step(const bt::Tensor& feats_batch,
                                              const bt::Tensor& labels,
                                              int B, int T,
                                              const std::vector<float>& class_weights) {
    auto& I = *impl_;
    const int Kc = I.K();
    if (I.fused) fail("PhonemeNet::eval_step", "fused models use forward()");
    if (feats_batch.device != I.device) fail("PhonemeNet::eval_step", "batch device mismatch");
    if (labels.rows != B || labels.cols != T)
        fail("PhonemeNet::eval_step", "labels shape mismatch (want (B,T))");
    if (!class_weights.empty() && static_cast<int>(class_weights.size()) != Kc)
        fail("PhonemeNet::eval_step", "class_weights length != K");

    const int F = B * T;
    bt::Tensor logits = forward_to_logits(I, feats_batch, B, T, /*train=*/false, nullptr);

    bt::Tensor lab_host = labels.to(bt::Device::CPU);
    const float* lp = lab_host.host_f32();
    std::vector<float> tgt(static_cast<std::size_t>(F) * Kc, 0.0f);
    std::vector<float> wf(static_cast<std::size_t>(F), 1.0f);
    double Z = 0.0;
    for (int f = 0; f < F; ++f) {
        const int cls = static_cast<int>(std::lround(lp[f]));
        if (cls < 0 || cls >= Kc) fail("PhonemeNet::eval_step", "label out of range [0,K)");
        tgt[static_cast<std::size_t>(f) * Kc + cls] = 1.0f;
        wf[f] = class_weights.empty() ? 1.0f : class_weights[cls];
        Z += wf[f];
    }
    bt::Tensor target = bt::Tensor::from_host_on(I.device, tgt.data(), F, Kc);
    bt::Tensor probs   = bt::Tensor::zeros_on(I.device, F, Kc, bt::Dtype::FP32);
    bt::Tensor dLogits = bt::Tensor::zeros_on(I.device, F, Kc, bt::Dtype::FP32);
    bt::Tensor loss_per = bt::Tensor::zeros_on(I.device, F, 1, bt::Dtype::FP32);
    bt::softmax_xent_fused_batched(logits, target, nullptr,
                                   static_cast<const int*>(I.head_offsets.data),
                                   1, probs, dLogits, loss_per);

    bt::Tensor loss_host   = loss_per.to(bt::Device::CPU);
    bt::Tensor logits_host = logits.to(bt::Device::CPU);
    const float* lhp = loss_host.host_f32();
    const float* glp = logits_host.host_f32();

    EvalMetrics m; m.n_frames = F;
    double total = 0.0;
    int correct = 0, ns_total = 0, ns_correct = 0;
    for (int f = 0; f < F; ++f) {
        total += static_cast<double>(wf[f]) * static_cast<double>(lhp[f]);
        const int truth = static_cast<int>(std::lround(lp[f]));
        int argmax = 0; float best = glp[static_cast<std::size_t>(f) * Kc];
        for (int k = 1; k < Kc; ++k) {
            const float v = glp[static_cast<std::size_t>(f) * Kc + k];
            if (v > best) { best = v; argmax = k; }
        }
        if (argmax == truth) ++correct;
        if (truth != 0) { ++ns_total; if (argmax == truth) ++ns_correct; }
    }
    m.loss = Z > 0.0 ? static_cast<float>(total / Z) : 0.0f;
    m.frame_accuracy = F ? static_cast<float>(correct) / static_cast<float>(F) : 0.0f;
    m.nonsilence_frame_accuracy =
        ns_total ? static_cast<float>(ns_correct) / static_cast<float>(ns_total) : 0.0f;
    return m;
}

// ─── Test/debug seam ──────────────────────────────────────────────────────────
namespace {
void ensure_train_state(PhonemeNet::Impl& I, std::unique_ptr<PhonemeNetTrainState>& ts) {
    if (!ts) { ts = std::make_unique<PhonemeNetTrainState>(); register_params(*ts, I); }
}
}  // namespace

std::vector<std::pair<std::string, int>> PhonemeNet::debug_params() const {
    ensure_train_state(*impl_, impl_->train_state);
    std::vector<std::pair<std::string, int>> out;
    for (auto& s : impl_->train_state->slots)
        out.push_back({s.name, s.param->size()});
    return out;
}
float PhonemeNet::debug_get_param(const std::string& name, int idx) const {
    ensure_train_state(*impl_, impl_->train_state);
    auto* s = impl_->train_state->find(name);
    if (!s) fail("PhonemeNet::debug_get_param", "no param '" + name + "'");
    bt::Tensor h = s->param->to(bt::Device::CPU);
    return h.host_f32()[idx];
}
void PhonemeNet::debug_set_param(const std::string& name, int idx, float value) {
    ensure_train_state(*impl_, impl_->train_state);
    auto* s = impl_->train_state->find(name);
    if (!s) fail("PhonemeNet::debug_set_param", "no param '" + name + "'");
    bt::Tensor h = s->param->to(bt::Device::CPU);
    h.host_f32_mut()[idx] = value;
    *s->param = h.to(s->param->device);
}
float PhonemeNet::debug_grad(const std::string& name, int idx) const {
    if (!impl_->train_state) fail("PhonemeNet::debug_grad", "no train state (run train_step first)");
    auto* s = impl_->train_state->find(name);
    if (!s) fail("PhonemeNet::debug_grad", "no grad '" + name + "'");
    bt::Tensor h = s->grad.to(bt::Device::CPU);
    return h.host_f32()[idx];
}

}  // namespace brosoundml
