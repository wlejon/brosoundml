#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

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

// Adam/grad state for the GPU training surface. Empty until the training stage
// lands; defined here (not just forward-declared) so unique_ptr<...> in Impl
// has a complete type for its destructor.
struct BcResnet2dTrainState {};

namespace bt = brotensor;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
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
    // Streaming time-history cache: the last padLeftW columns per (Cin,H) row,
    // laid out (1, Cin*Hcache*padLeftW). Hcache is the layer's input H.
    bt::Tensor cache;
    int Hcache = 0;
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
    k = bt::Tensor::from_host_on(dev, h.data(), C, H);   // (C, H) depthwise OIHW
}

// Causal-time conv forward on an NCHW (1, Cin*H*W) input. Left-pads the time
// (W) axis by padLeftW columns per (Cin,H) row — with zeros (use_cache=false)
// or the streaming cache (use_cache=true, cache then refreshed) — and runs a
// pad_w=0 conv2d, so output column t sees only input columns <= t.
void conv_forward(Conv2dLayer& c, const bt::Tensor& X, int H, int W,
                  bool use_cache, bt::Tensor& Y) {
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

    const int Wp = W + c.padLeftW;
    bt::Tensor padded = bt::Tensor::zeros_on(dev, 1, rows * Wp, bt::Dtype::FP32);
    for (int r = 0; r < rows; ++r) {
        if (use_cache && c.cache.size() > 0) {
            bt::copy_d2d(c.cache, r * c.padLeftW, padded, r * Wp, c.padLeftW);
        }
        bt::copy_d2d(X, r * W, padded, r * Wp + c.padLeftW, W);
    }
    conv2d_forward(padded, c.W, &c.b, /*N=*/1, c.Cin, H, Wp, c.Cout,
                   c.kH, c.kW, c.sH, /*sw=*/1, c.padH, /*pad_w=*/0,
                   /*dil_h=*/1, c.dilW, c.groups, Y);

    if (use_cache && c.cache.size() > 0) {
        for (int r = 0; r < rows; ++r) {
            bt::copy_d2d(padded, r * Wp + (Wp - c.padLeftW),
                         c.cache, r * c.padLeftW, c.padLeftW);
        }
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

// Frequency-average-pool (C,H,W) -> (C,1,W) via a depthwise conv with a (1/H)
// kernel spanning all H rows (kH=H, one output position).
void freq_avg(const bt::Tensor& X, const bt::Tensor& avg_kernel,
              int C, int H, int W, bt::Tensor& Y) {
    conv2d_forward(X, avg_kernel, /*bias=*/nullptr, /*N=*/1, C, H, W, C,
                   /*kH=*/H, /*kW=*/1, /*sh=*/1, /*sw=*/1, /*ph=*/0, /*pw=*/0,
                   /*dh=*/1, /*dw=*/1, /*groups=*/C, Y);
}

// Broadcast (C,1,W) -> (C,H,W): each frequency row copies the single input row.
// Implemented as the input-gradient of a depthwise all-ones (kH=H) conv, which
// scatters the single value to every frequency tap.
void freq_broadcast(const bt::Tensor& X1, const bt::Tensor& ones_kernel,
                    int C, int H, int W, bt::Tensor& Y) {
    bt::conv2d_backward_input(ones_kernel, X1, /*N=*/1, C, H, W, C,
                              /*kH=*/H, /*kW=*/1, /*sh=*/1, /*sw=*/1,
                              /*ph=*/0, /*pw=*/0, /*dh=*/1, /*dw=*/1,
                              /*groups=*/C, Y);
}

// ─── Binary format ('BWK2') ─────────────────────────────────────────────────
constexpr std::uint32_t kMagic   = 0x324B5742u;   // 'B''W''K''2'
constexpr std::uint32_t kVersion = 1u;

struct Reader {
    std::FILE* fp = nullptr; std::string path;
    ~Reader() { if (fp) std::fclose(fp); }
    void open(const std::string& p) {
        path = p; fp = std::fopen(p.c_str(), "rb");
        if (!fp) fail("bc_resnet2d::load", "could not open '" + p + "'");
    }
    void rd(void* d, std::size_t n) {
        if (std::fread(d, 1, n, fp) != n)
            fail("bc_resnet2d::load", "unexpected EOF in '" + path + "'");
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
        if (!fp) fail("bc_resnet2d::save", "could not open '" + p + "'");
    }
    void wr(const void* s, std::size_t n) {
        if (std::fwrite(s, 1, n, fp) != n)
            fail("bc_resnet2d::save", "short write to '" + path + "'");
    }
    void u32(std::uint32_t v) { wr(&v, 4); }
    void u16(std::uint16_t v) { wr(&v, 2); }
    void u8 (std::uint8_t  v) { wr(&v, 1); }
    void f32(float v)         { wr(&v, 4); }
};

void write_tensor(Writer& w, const std::string& name, const bt::Tensor& t) {
    bt::Tensor host = t.to(bt::Device::CPU);
    if (host.dtype != bt::Dtype::FP32) fail("bc_resnet2d::save", "non-FP32 tensor");
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
        fail("bc_resnet2d::load", "expected '" + want + "' got '" + name + "'");
    if (rr != rows || cc != cols)
        fail("bc_resnet2d::load", "tensor '" + name + "' shape mismatch");
    std::vector<float> buf(static_cast<std::size_t>(rr) * cc);
    r.rd(buf.data(), buf.size() * sizeof(float));
    return bt::Tensor::from_host_on(bt::Device::CPU, buf.data(), rr, cc).to(dev);
}

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

    // Streaming GAP ring over the head's temporal feature (length c_last).
    std::vector<float> gap_window;
    int gap_len = 0, gap_head = 0, gap_cap = 0;

    std::unique_ptr<BcResnet2dTrainState> train_state;

    int freq_pad(int k) const { return (k - 1) / 2; }

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
        gap_window.assign(static_cast<std::size_t>(gap_cap) * c_last, 0.0f);
        gap_len = 0; gap_head = 0;
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
    // Y: (cout, Hz, W). use_cache toggles the streaming time caches.
    void run_block(BCBlock& blk, const bt::Tensor& X, int Hin, int W,
                   bool use_cache, bt::Tensor& Y) {
        const int cout = blk.cout;
        // Channel/stride entry.
        bt::Tensor u;     // (cout, Hin, W)
        if (blk.is_transition) {
            bt::Tensor t;
            conv_forward(blk.pw_in, X, Hin, W, use_cache, t);
            bn_inplace(blk.bn_in, t, cout, Hin, W);
            relu_into(t, u);
        } else {
            u = X;        // normal block: identity entry on cout==cin
        }

        // f2: freq depthwise + BN.  (cout, Hz, W)
        bt::Tensor z;
        conv_forward(blk.f2, u, Hin, W, use_cache, z);
        bn_inplace(blk.bn_f2, z, cout, blk.Hz, W);

        // f1: freq-avg -> time depthwise (causal) -> BN -> ReLU -> pw1x1.
        bt::Tensor a;     // (cout,1,W)
        freq_avg(z, blk.avg_kernel, cout, blk.Hz, W, a);
        bt::Tensor tt;
        conv_forward(blk.f1, a, /*H=*/1, W, use_cache, tt);
        bn_inplace(blk.bn_f1, tt, cout, 1, W);
        bt::Tensor tt_relu; relu_into(tt, tt_relu);
        bt::Tensor tt_pw;
        conv_forward(blk.pw_f1, tt_relu, /*H=*/1, W, use_cache, tt_pw);

        // y = ReLU( z + broadcast_H(tt_pw) )
        bt::Tensor bc;
        freq_broadcast(tt_pw, blk.ones_kernel, cout, blk.Hz, W, bc);
        bt::add_inplace(z, bc);
        bt::Tensor y; relu_into(z, y);

        // residual
        bt::Tensor res;
        if (blk.is_transition) {
            conv_forward(blk.res, X, Hin, W, use_cache, res);
            bn_inplace(blk.bn_res, res, cout, blk.Hz, W);
        } else {
            res = X;   // identity (cout==cin, Hz==Hin)
        }
        bt::add_inplace(y, res);
        Y = std::move(y);
    }

    // Shared forward producing per-frame logits (T,1). use_cache=false is the
    // one-shot path (fresh GAP window); true streams (persistent caches+GAP).
    void forward_core(const bt::Tensor& feats, bool use_cache,
                      bt::Tensor& out_logits) {
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

        std::vector<float>* win; int* len; int* head_pos;
        std::vector<float> local; int ll = 0, lh = 0;
        if (use_cache) { win = &gap_window; len = &gap_len; head_pos = &gap_head; }
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

// ─── Public surface ───────────────────────────────────────────────────────────

BcResnet2d::BcResnet2d() : impl_(std::make_unique<Impl>()) {}
BcResnet2d::~BcResnet2d() = default;
BcResnet2d::BcResnet2d(BcResnet2d&&) noexcept = default;
BcResnet2d& BcResnet2d::operator=(BcResnet2d&&) noexcept = default;

BcResnet2d BcResnet2d::make(const BcResnet2dConfig& cfg, bt::Device device) {
    BcResnet2d m;
    m.impl_->cfg = cfg;
    m.impl_->device = device;
    m.impl_->fused = false;
    m.impl_->build();
    return m;
}

const BcResnet2dConfig& BcResnet2d::config() const { return impl_->cfg; }
bt::Device              BcResnet2d::device() const { return impl_->device; }
bool                    BcResnet2d::fused()  const { return impl_->fused; }
int BcResnet2d::receptive_field_frames() const {
    return impl_->rf_frames() + impl_->gap_cap - 1;
}

int BcResnet2d::param_count() const {
    int n = 0;
    auto conv = [](const Conv2dLayer& c) { return c.Cout * (c.Cin / c.groups) * c.kH * c.kW + c.Cout; };
    auto bn = [&](const BatchNorm2d& b) { return impl_->fused ? 0 : 2 * b.channels; };
    n += conv(impl_->stem) + bn(impl_->bn_stem);
    for (auto& blk : impl_->blocks) {
        if (blk.is_transition) { n += conv(blk.pw_in) + bn(blk.bn_in); n += conv(blk.res) + bn(blk.bn_res); }
        n += conv(blk.f2) + bn(blk.bn_f2);
        n += conv(blk.f1) + bn(blk.bn_f1) + conv(blk.pw_f1);
    }
    n += impl_->c_last + 1;
    return n;
}

void BcResnet2d::reset_streaming_state() {
    auto rc = [](Conv2dLayer& c) { if (c.cache.size() > 0) c.cache.zero(); };
    rc(impl_->stem);
    for (auto& blk : impl_->blocks) {
        if (blk.is_transition) { rc(blk.pw_in); rc(blk.res); }
        rc(blk.f2); rc(blk.f1); rc(blk.pw_f1);
    }
    std::fill(impl_->gap_window.begin(), impl_->gap_window.end(), 0.0f);
    impl_->gap_len = 0; impl_->gap_head = 0;
}

void BcResnet2d::forward(const bt::Tensor& feats, bt::Tensor& out_logit_per_frame) const {
    auto& mut = const_cast<Impl&>(*impl_);
    mut.forward_core(feats, /*use_cache=*/false, out_logit_per_frame);
}

void BcResnet2d::forward_streaming(const bt::Tensor& new_feats, bt::Tensor& out) {
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

void BcResnet2d::fuse_bn() {
    if (impl_->fused) return;
    fuse_one(impl_->stem, impl_->bn_stem);
    for (auto& blk : impl_->blocks) {
        if (blk.is_transition) { fuse_one(blk.pw_in, blk.bn_in); fuse_one(blk.res, blk.bn_res); }
        fuse_one(blk.f2, blk.bn_f2);
        fuse_one(blk.f1, blk.bn_f1);   // pw_f1 has no BN
    }
    impl_->fused = true;
}

// ─── Weights I/O ────────────────────────────────────────────────────────────
namespace {
void for_each_tensor(BcResnet2d::Impl& m, bool include_bn,
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
    add("head.W", m.head_W); add("head.b", m.head_b);
}
}  // namespace

void BcResnet2d::save(const std::string& path, bool fused) const {
    if (fused && !impl_->fused) fail("BcResnet2d::save", "fuse_bn() has not run");
    if (!fused && impl_->fused) fail("BcResnet2d::save", "BN already fused");
    Writer w; w.open(path);
    w.u32(kMagic); w.u32(kVersion);
    const auto& c = impl_->cfg;
    w.u32(c.n_mels); w.u32(c.c_stem); w.u32(c.stem_kf); w.u32(c.stem_kt); w.u32(c.stem_sf);
    for (int s = 0; s < BcResnet2dConfig::n_stages; ++s) {
        w.u32(c.c[s]); w.u32(c.fstride[s]); w.u32(c.tdil[s]); w.u32(c.n_blocks[s]);
    }
    w.u32(c.dw_kf); w.u32(c.dw_kt); w.u32(c.ssn_subbands); w.f32(c.bn_eps);
    w.u8(fused ? 1u : 0u);
    std::vector<std::pair<std::string, bt::Tensor*>> ts;
    for_each_tensor(*impl_, /*include_bn=*/!fused, ts);
    w.u32(static_cast<std::uint32_t>(ts.size()));
    for (auto& [n, t] : ts) write_tensor(w, n, *t);
}

BcResnet2d BcResnet2d::load(const std::string& path, bt::Device device) {
    Reader r; r.open(path);
    if (r.u32() != kMagic)   fail("BcResnet2d::load", "bad magic");
    if (r.u32() != kVersion) fail("BcResnet2d::load", "unsupported version");
    BcResnet2dConfig c;
    c.n_mels = (int)r.u32(); c.c_stem = (int)r.u32(); c.stem_kf = (int)r.u32();
    c.stem_kt = (int)r.u32(); c.stem_sf = (int)r.u32();
    for (int s = 0; s < BcResnet2dConfig::n_stages; ++s) {
        c.c[s] = (int)r.u32(); c.fstride[s] = (int)r.u32();
        c.tdil[s] = (int)r.u32(); c.n_blocks[s] = (int)r.u32();
    }
    c.dw_kf = (int)r.u32(); c.dw_kt = (int)r.u32(); c.ssn_subbands = (int)r.u32();
    c.bn_eps = r.f32();
    const bool on_disk_fused = (r.u8() != 0u);
    const std::uint32_t num = r.u32();

    BcResnet2d m = BcResnet2d::make(c, device);
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
        fail("BcResnet2d::load", "tensor count mismatch");
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

// ─── Training surface (next stage) ────────────────────────────────────────────
float BcResnet2d::train_step(const bt::Tensor&, const bt::Tensor&, int, int, float, float) {
    fail("BcResnet2d::train_step", "GPU training not yet landed (stage: train)");
}
BcResnet2d::EvalMetrics BcResnet2d::eval_step(const bt::Tensor&, const bt::Tensor&, int, int, float) {
    fail("BcResnet2d::eval_step", "not yet landed (stage: train)");
}

}  // namespace brosoundml
