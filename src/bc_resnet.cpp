#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "brosoundml/bc_resnet.h"

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

// Forward-declared so BcResnet::Impl can hold a unique_ptr to it; full
// definition lives below in the training-surface section.
struct BcResnetTrainState;

namespace bt = brotensor;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

// One streamable causal-conv layer: weight (C_out, (C_in/groups)*kL), bias
// (C_out, 1), kernel kL, dilation d, groups g, and a streaming ring buffer
// holding the last pad_left = d*(kL-1) input frames as a (1, C_in*pad_left)
// NCL tensor. Bias is always present after fuse_bn so the (C_out,1) shape is
// guaranteed non-empty at inference time.
struct ConvLayer {
    int               in_channels  = 0;
    int               out_channels = 0;
    int               kernel_size  = 1;
    int               dilation     = 1;
    int               groups       = 1;
    bt::Tensor        W;       // (C_out, (C_in/groups)*kL) — fused after fuse_bn
    bt::Tensor        b;       // (C_out, 1)                 — fused after fuse_bn
    bt::Tensor        cache;   // (1, C_in*pad_left) NCL, or empty when pad_left==0
    int               pad_left = 0;
};

// BN parameters in their unfused form. After fuse_bn they are gone (the fold
// step folds them into the preceding ConvLayer's W/b and clears these).
struct BatchNorm1d {
    int        channels = 0;
    bt::Tensor gamma;   // (C, 1)
    bt::Tensor beta;    // (C, 1)
    bt::Tensor mean;    // (C, 1)
    bt::Tensor var;     // (C, 1)
    float      eps      = 1e-5f;
    bool       present  = false;   // false once fused into the preceding conv
};

// One BC-ResNet residual block: depthwise conv → BN → ReLU → pointwise conv
// → BN, plus an optional pointwise residual projection (used when c_in !=
// c_out so the residual add can match channel counts). The residual BN
// fuses into the residual conv just like the main path.
struct Block {
    ConvLayer    dw;
    BatchNorm1d  bn_dw;
    ConvLayer    pw;
    BatchNorm1d  bn_pw;

    bool         has_residual_proj = false;
    ConvLayer    res_proj;       // pointwise 1x1; only when has_residual_proj
    BatchNorm1d  bn_res;
};

// Resize an NCL ring cache, preserving its device + dtype.
void ensure_cache(bt::Tensor& t, bt::Device dev, int rows, int cols) {
    if (t.device != dev || t.rows != rows || t.cols != cols ||
        t.dtype  != bt::Dtype::FP32) {
        t = bt::Tensor::zeros_on(dev, rows, cols, bt::Dtype::FP32);
    } else {
        t.zero();
    }
}

void init_conv(ConvLayer& c, int c_in, int c_out, int kL, int dilation,
               int groups, bt::Device dev) {
    c.in_channels  = c_in;
    c.out_channels = c_out;
    c.kernel_size  = kL;
    c.dilation     = dilation;
    c.groups       = groups;
    c.pad_left     = dilation * (kL - 1);
    const int weight_in = (c_in / groups) * kL;
    c.W = bt::Tensor::zeros_on(dev, c_out, weight_in, bt::Dtype::FP32);
    c.b = bt::Tensor::zeros_on(dev, c_out, 1,         bt::Dtype::FP32);
    if (c.pad_left > 0) {
        c.cache = bt::Tensor::zeros_on(dev, 1, c_in * c.pad_left,
                                       bt::Dtype::FP32);
    } else {
        c.cache = bt::Tensor{};
    }
}

void init_bn(BatchNorm1d& bn, int C, bt::Device dev, float eps) {
    bn.channels = C;
    bn.eps      = eps;
    bn.present  = true;
    // BN defaults: gamma=1, mean=0, var=1 (so the unit on first construction
    // is exactly identity — the trainer overwrites these before fusing).
    std::vector<float> ones(static_cast<std::size_t>(C), 1.0f);
    bn.gamma = bt::Tensor::from_host_on(dev, ones.data(), C, 1);
    bn.beta  = bt::Tensor::zeros_on   (dev, C, 1, bt::Dtype::FP32);
    bn.mean  = bt::Tensor::zeros_on   (dev, C, 1, bt::Dtype::FP32);
    bn.var   = bt::Tensor::from_host_on(dev, ones.data(), C, 1);
}

// Run one causal Conv1d on a contiguous NCL input. `X_in` is (1, C_in*L_in)
// already laid out flat in NCL with the per-call pad_left zeros if `use_cache`
// is false (one-shot path), otherwise streaming mode prepends the layer's
// cache to `X_in` and updates the cache from the trailing pad_left columns of
// (cache ++ X_in). Output is (1, C_out*L_in) NCL.
void conv_forward(ConvLayer& c, const bt::Tensor& X_in, int L_in,
                  bool use_cache, bt::Tensor& Y) {
    if (X_in.rows != 1 || X_in.cols != c.in_channels * L_in) {
        fail("bc_resnet::conv_forward",
             "input shape (" + std::to_string(X_in.rows) + "," +
             std::to_string(X_in.cols) + ") != (1," +
             std::to_string(c.in_channels * L_in) + ")");
    }

    if (!use_cache || c.pad_left == 0) {
        // One-shot path: brotensor's header-only wrapper does the left-pad +
        // valid conv for us. The scratch buffer is owned per call — fine for
        // the offline forward, which is not on the per-frame hot path.
        bt::Tensor scratch;
        bt::causal_conv1d(X_in, c.W, &c.b, /*N=*/1, c.in_channels, L_in,
                          c.out_channels, c.kernel_size, /*stride=*/1,
                          c.dilation, c.groups, scratch, Y);
        return;
    }

    // Streaming path: prepend the cache (1, C_in*pad_left) to X_in
    // (1, C_in*L_in) along the L axis. Both are NCL row-major so the join is
    // per-channel: dst[c, :pad_left] = cache[c, :], dst[c, pad_left:] = X[c, :].
    const int L_total = c.pad_left + L_in;
    bt::Tensor joined = bt::Tensor::empty_on(X_in.device, 1,
                                             c.in_channels * L_total,
                                             bt::Dtype::FP32);
    for (int ch = 0; ch < c.in_channels; ++ch) {
        bt::copy_d2d(c.cache, ch * c.pad_left,
                     joined,  ch * L_total,
                     c.pad_left);
        bt::copy_d2d(X_in,    ch * L_in,
                     joined,  ch * L_total + c.pad_left,
                     L_in);
    }

    // Valid conv (padding=0) over the joined buffer reproduces the causal
    // conv's outputs for the new L_in frames.
    bt::conv1d(joined, c.W, &c.b, /*N=*/1, c.in_channels, L_total,
               c.out_channels, c.kernel_size, /*stride=*/1, /*padding=*/0,
               c.dilation, c.groups, Y);

    // Update cache to the trailing pad_left columns of joined (per channel).
    for (int ch = 0; ch < c.in_channels; ++ch) {
        bt::copy_d2d(joined, ch * L_total + (L_total - c.pad_left),
                     c.cache, ch * c.pad_left,
                     c.pad_left);
    }
}

// Apply BN (when still present) channel-wise to an NCL (1, C*L) tensor in
// place. Used only during the fuse_bn round-trip self-check and the chunk-6
// unfused-forward path; after fuse_bn() this is never called.
void bn_inplace_ncl(BatchNorm1d& bn, bt::Tensor& X, int C, int L) {
    if (!bn.present) return;
    if (X.device != bt::Device::CPU) {
        // CPU-only — unfused BN is a training-time crutch; inference always
        // runs through fuse_bn first. Folded weights are device-agnostic.
        fail("bc_resnet::bn_inplace",
             "unfused BN only supported on CPU; call fuse_bn() before "
             "running on a non-CPU device");
    }
    const float* g = bn.gamma.host_f32();
    const float* be= bn.beta .host_f32();
    const float* m = bn.mean .host_f32();
    const float* v = bn.var  .host_f32();
    float* x = X.host_f32_mut();
    for (int ch = 0; ch < C; ++ch) {
        const float inv = 1.0f / std::sqrt(v[ch] + bn.eps);
        const float a   = g[ch] * inv;
        const float b   = be[ch] - m[ch] * g[ch] * inv;
        float* row = x + ch * L;
        for (int t = 0; t < L; ++t) row[t] = row[t] * a + b;
    }
}

// Fold a BN's (gamma, beta, mean, var) into a ConvLayer's (W, b) on the host.
// W' = W * (gamma / sqrt(var+eps))[c_out],  b' = (b - mean) * a + beta
// where a = gamma / sqrt(var + eps). The conv's weight is OIL = (C_out,
// (C_in/groups)*kL) so the per-output-channel scale broadcasts over the row.
void fuse_one(ConvLayer& c, BatchNorm1d& bn) {
    if (!bn.present) return;
    if (bn.channels != c.out_channels) {
        fail("bc_resnet::fuse_one",
             "BN channels=" + std::to_string(bn.channels) +
             " != conv out_channels=" + std::to_string(c.out_channels));
    }
    // Migrate to host for the fold then migrate back. The model is small
    // enough that this round-trip is unmeasurable next to a real inference.
    const bt::Device target = c.W.device;
    bt::Tensor W_host  = c.W.to(bt::Device::CPU);
    bt::Tensor b_host  = c.b.to(bt::Device::CPU);
    bt::Tensor g_host  = bn.gamma.to(bt::Device::CPU);
    bt::Tensor be_host = bn.beta .to(bt::Device::CPU);
    bt::Tensor m_host  = bn.mean .to(bt::Device::CPU);
    bt::Tensor v_host  = bn.var  .to(bt::Device::CPU);

    const int C   = c.out_channels;
    const int row = W_host.cols;
    float* W = W_host.host_f32_mut();
    float* B = b_host.host_f32_mut();
    const float* g  = g_host .host_f32();
    const float* be = be_host.host_f32();
    const float* m  = m_host .host_f32();
    const float* v  = v_host .host_f32();
    for (int co = 0; co < C; ++co) {
        const float a = g[co] / std::sqrt(v[co] + bn.eps);
        for (int j = 0; j < row; ++j) W[co * row + j] *= a;
        B[co] = (B[co] - m[co]) * a + be[co];
    }
    c.W = W_host.to(target);
    c.b = b_host.to(target);

    bn.present = false;
    bn.gamma   = bt::Tensor{};
    bn.beta    = bt::Tensor{};
    bn.mean    = bt::Tensor{};
    bn.var     = bt::Tensor{};
}

// Reset a conv's ring cache to zeros (preserves shape/device).
void reset_one(ConvLayer& c) {
    if (c.pad_left > 0 && c.cache.size() > 0) c.cache.zero();
}

// ─── Binary format ────────────────────────────────────────────────────────
//
// Header magic 'B','W','A','K' + version u32. Then a fixed BcResnetConfig
// payload, a u8 fused-flag, and a u32 tensor count. Each tensor record is
//   u16 name_len, name bytes, u32 rows, u32 cols, rows*cols FP32 little-endian.
// Only FP32 / 2D tensors are written — the model is small and dense, so the
// minimal record format is sufficient.

constexpr std::uint32_t kMagic   = 0x4B415742u;   // 'B''W''A''K'
constexpr std::uint32_t kVersion = 1u;

struct Reader {
    std::FILE*  fp = nullptr;
    std::string path;
    ~Reader() { if (fp) std::fclose(fp); }
    void open(const std::string& p) {
        path = p;
        fp = std::fopen(p.c_str(), "rb");
        if (!fp) fail("bc_resnet::load", "could not open '" + p + "'");
    }
    void read_bytes(void* dst, std::size_t n) {
        if (std::fread(dst, 1, n, fp) != n) {
            fail("bc_resnet::load",
                 "unexpected EOF reading " + std::to_string(n) +
                 " bytes from '" + path + "'");
        }
    }
    std::uint32_t u32() { std::uint32_t v; read_bytes(&v, 4); return v; }
    std::uint16_t u16() { std::uint16_t v; read_bytes(&v, 2); return v; }
    std::uint8_t  u8 () { std::uint8_t  v; read_bytes(&v, 1); return v; }
    float         f32() { float v;         read_bytes(&v, 4); return v; }
    std::string   str(std::size_t n) {
        std::string s(n, '\0'); read_bytes(s.data(), n); return s;
    }
};

struct Writer {
    std::FILE*  fp = nullptr;
    std::string path;
    ~Writer() { if (fp) std::fclose(fp); }
    void open(const std::string& p) {
        path = p;
        fp = std::fopen(p.c_str(), "wb");
        if (!fp) fail("bc_resnet::save", "could not open '" + p + "' for writing");
    }
    void write_bytes(const void* src, std::size_t n) {
        if (std::fwrite(src, 1, n, fp) != n) {
            fail("bc_resnet::save", "short write to '" + path + "'");
        }
    }
    void u32(std::uint32_t v) { write_bytes(&v, 4); }
    void u16(std::uint16_t v) { write_bytes(&v, 2); }
    void u8 (std::uint8_t  v) { write_bytes(&v, 1); }
    void f32(float v)         { write_bytes(&v, 4); }
};

void write_tensor(Writer& w, const std::string& name, const bt::Tensor& t) {
    bt::Tensor host = t.to(bt::Device::CPU);
    if (host.dtype != bt::Dtype::FP32) {
        fail("bc_resnet::save", "tensor '" + name + "' must be FP32");
    }
    if (name.size() > 0xFFFFu) fail("bc_resnet::save", "name too long");
    w.u16(static_cast<std::uint16_t>(name.size()));
    w.write_bytes(name.data(), name.size());
    w.u32(static_cast<std::uint32_t>(host.rows));
    w.u32(static_cast<std::uint32_t>(host.cols));
    w.write_bytes(host.host_f32(),
                  static_cast<std::size_t>(host.size()) * sizeof(float));
}

bt::Tensor read_tensor(Reader& r, const std::string& expected_name,
                       int expected_rows, int expected_cols, bt::Device dev) {
    const std::uint16_t name_len = r.u16();
    const std::string name = r.str(name_len);
    const int rows = static_cast<int>(r.u32());
    const int cols = static_cast<int>(r.u32());
    if (name != expected_name) {
        fail("bc_resnet::load",
             "expected tensor '" + expected_name + "' got '" + name + "'");
    }
    if (rows != expected_rows || cols != expected_cols) {
        fail("bc_resnet::load",
             "tensor '" + name + "' shape (" + std::to_string(rows) + "," +
             std::to_string(cols) + ") != (" + std::to_string(expected_rows) +
             "," + std::to_string(expected_cols) + ")");
    }
    std::vector<float> buf(static_cast<std::size_t>(rows) * cols);
    r.read_bytes(buf.data(), buf.size() * sizeof(float));
    bt::Tensor host = bt::Tensor::from_host_on(bt::Device::CPU, buf.data(),
                                               rows, cols);
    return host.to(dev);
}

}  // namespace

// ─── BcResnet::Impl ───────────────────────────────────────────────────────

struct BcResnet::Impl {
    BcResnetConfig   cfg;
    bt::Device       device = bt::Device::CPU;
    bool             fused  = false;

    ConvLayer        stem;
    BatchNorm1d      bn_stem;
    Block            b1, b2, b3, b4;
    bt::Tensor       head_W;   // (1, c4)
    bt::Tensor       head_b;   // (1, 1)

    // Streaming-state sliding-window GAP buffer. Holds up to
    // `receptive_field_frames` of the most recent post-block-4 feature
    // vectors (length c4 each), host-side. Persisted across
    // forward_streaming() calls; cleared by reset_streaming_state().
    std::vector<float> gap_window;       // flat (gap_len, c4), row-major
    int                gap_len   = 0;    // current number of valid rows
    int                gap_head  = 0;    // ring write position
    int                gap_cap   = 0;    // = receptive_field_frames

    // Adam state + grads — lazy-built on first train_step(). Empty during
    // inference-only sessions.
    std::unique_ptr<BcResnetTrainState> train_state;

    int block_cout(int idx) const {
        switch (idx) {
            case 1: return cfg.c1;
            case 2: return cfg.c2;
            case 3: return cfg.c3;
            case 4: return cfg.c4;
            default: return 0;
        }
    }
    int block_cin(int idx) const {
        switch (idx) {
            case 1: return cfg.c0;
            case 2: return cfg.c1;
            case 3: return cfg.c2;
            case 4: return cfg.c3;
            default: return 0;
        }
    }
    int block_dilation(int idx) const {
        switch (idx) {
            case 1: return cfg.dil1;
            case 2: return cfg.dil2;
            case 3: return cfg.dil3;
            case 4: return cfg.dil4;
            default: return 1;
        }
    }

    void build() {
        // Stem: standard conv with k=3.
        init_conv(stem, cfg.n_mels, cfg.c0, /*kL=*/3, /*dil=*/1,
                  /*groups=*/1, device);
        init_bn(bn_stem, cfg.c0, device, cfg.bn_eps);

        auto build_block = [&](Block& blk, int idx) {
            const int ci  = block_cin(idx);
            const int co  = block_cout(idx);
            const int dil = block_dilation(idx);
            // Depthwise.
            init_conv(blk.dw, ci, ci, cfg.kernel_size, dil, /*groups=*/ci,
                      device);
            init_bn(blk.bn_dw, ci, device, cfg.bn_eps);
            // Pointwise.
            init_conv(blk.pw, ci, co, /*kL=*/1, /*dil=*/1, /*groups=*/1,
                      device);
            init_bn(blk.bn_pw, co, device, cfg.bn_eps);
            // Residual projection: identity if ci==co, else 1x1.
            if (ci != co) {
                blk.has_residual_proj = true;
                init_conv(blk.res_proj, ci, co, /*kL=*/1, /*dil=*/1,
                          /*groups=*/1, device);
                init_bn(blk.bn_res, co, device, cfg.bn_eps);
            } else {
                blk.has_residual_proj = false;
            }
        };
        build_block(b1, 1);
        build_block(b2, 2);
        build_block(b3, 3);
        build_block(b4, 4);

        // Linear head: maps c4 → 1 with bias.
        head_W = bt::Tensor::zeros_on(device, 1, cfg.c4, bt::Dtype::FP32);
        head_b = bt::Tensor::zeros_on(device, 1, 1,      bt::Dtype::FP32);

        // GAP window covers the conv-stack receptive field — see
        // BcResnet::receptive_field_frames() for the formula. Sized once at
        // construction; reset to zero by reset_streaming_state().
        gap_cap = 2 + (cfg.kernel_size - 1) *
                      (cfg.dil1 + cfg.dil2 + cfg.dil3 + cfg.dil4) + 1;
        gap_window.assign(static_cast<std::size_t>(gap_cap) * cfg.c4, 0.0f);
        gap_len  = 0;
        gap_head = 0;
    }

    // Apply one block in either one-shot mode (use_cache=false) or streaming
    // mode (use_cache=true). X comes in as NCL (1, ci*L); Y is overwritten
    // with NCL (1, co*L).
    void run_block(Block& blk, const bt::Tensor& X, int ci, int co, int L,
                   bool use_cache, bt::Tensor& Y) {
        // Depthwise.
        bt::Tensor dw_out;
        conv_forward(blk.dw, X, L, use_cache, dw_out);
        bn_inplace_ncl(blk.bn_dw, dw_out, ci, L);
        bt::Tensor dw_relu = bt::Tensor::empty_on(X.device, 1, ci * L,
                                                  bt::Dtype::FP32);
        bt::relu_forward(dw_out, dw_relu);

        // Pointwise.
        bt::Tensor pw_out;
        conv_forward(blk.pw, dw_relu, L, use_cache, pw_out);
        bn_inplace_ncl(blk.bn_pw, pw_out, co, L);

        // Residual.
        bt::Tensor res;
        if (blk.has_residual_proj) {
            conv_forward(blk.res_proj, X, L, use_cache, res);
            bn_inplace_ncl(blk.bn_res, res, co, L);
        } else {
            res = X;   // Tensor copy = device-aware deep copy per tensor.h.
        }
        bt::add_inplace(pw_out, res);

        Y = bt::Tensor::empty_on(X.device, 1, co * L, bt::Dtype::FP32);
        bt::relu_forward(pw_out, Y);
    }

    // Forward pass shared by forward() and forward_streaming(). When
    // use_cache=true the per-conv ring caches are read + written; when false
    // the caches are bypassed (one-shot causal-pad path).
    void forward_core(const bt::Tensor& log_mel, bool use_cache,
                      bt::Tensor& out_logits) {
        if (log_mel.rows != cfg.n_mels) {
            fail("BcResnet::forward",
                 "input rows=" + std::to_string(log_mel.rows) +
                 " != n_mels=" + std::to_string(cfg.n_mels));
        }
        if (log_mel.dtype != bt::Dtype::FP32) {
            fail("BcResnet::forward", "input must be FP32");
        }
        if (log_mel.device != device) {
            fail("BcResnet::forward",
                 "input device does not match model device");
        }
        const int T = log_mel.cols;
        if (T <= 0) {
            // No frames in — emit zero rows. Caller's responsibility to handle.
            out_logits = bt::Tensor::zeros_on(device, 0, 1, bt::Dtype::FP32);
            return;
        }

        // Reshape (n_mels, T) to NCL (1, n_mels*T). Row-major (n_mels, T)
        // already has the per-channel-contiguous layout NCL expects so this is
        // just a logical view, but we need a (1, n_mels*T) Tensor object —
        // build it by copying values on-device (small enough — n_mels=40).
        bt::Tensor x_ncl = bt::Tensor::empty_on(device, 1, cfg.n_mels * T,
                                                bt::Dtype::FP32);
        bt::copy_d2d(log_mel, 0, x_ncl, 0, cfg.n_mels * T);

        // Stem.
        bt::Tensor y_stem;
        conv_forward(stem, x_ncl, T, use_cache, y_stem);
        bn_inplace_ncl(bn_stem, y_stem, cfg.c0, T);
        bt::Tensor stem_relu = bt::Tensor::empty_on(device, 1, cfg.c0 * T,
                                                    bt::Dtype::FP32);
        bt::relu_forward(y_stem, stem_relu);

        // Blocks.
        bt::Tensor y1, y2, y3, y4;
        run_block(b1, stem_relu, cfg.c0, cfg.c1, T, use_cache, y1);
        run_block(b2, y1,        cfg.c1, cfg.c2, T, use_cache, y2);
        run_block(b3, y2,        cfg.c2, cfg.c3, T, use_cache, y3);
        run_block(b4, y3,        cfg.c3, cfg.c4, T, use_cache, y4);

        // Per-frame logits = head(GAP over the last receptive_field_frames
        // post-block-4 feature vectors). The conv stack's receptive field
        // bounds how far the convolutions look back; pooling over the same
        // window means the model's effective lookback equals the configured
        // receptive field — and changing input outside that window cannot
        // change the output (test_bc_resnet locks this).
        //
        // y4 layout: (1, c4*T) NCL. Reorder to (T, c4) (per-timestep rows)
        // and pull to host: the sliding-window GAP + linear head is a few
        // hundred multiplies per frame; the device round-trip and
        // accumulation-order determinism win us back the cost.
        bt::Tensor y4_seq;
        bt::nchw_to_sequence(y4, /*N=*/1, cfg.c4, /*H=*/1, /*W=*/T, y4_seq);
        bt::Tensor y_host  = y4_seq.to(bt::Device::CPU);
        bt::Tensor hW_host = head_W.to(bt::Device::CPU);
        bt::Tensor hb_host = head_b.to(bt::Device::CPU);
        const float* y_p  = y_host.host_f32();
        const float* hW   = hW_host.host_f32();
        const float  bias = hb_host.host_f32()[0];
        const int C   = cfg.c4;
        const int cap = gap_cap;

        // For the one-shot path we ignore any prior streaming state and
        // restart the window — the test that compares paths feeds the
        // identical T frames through forward() (use_cache=false → here) and
        // through forward_streaming() with a fresh reset, so both start from
        // an empty window.
        std::vector<float>* window;
        int* len;
        int* head_pos;
        std::vector<float> local_window;
        int local_len = 0, local_head = 0;
        if (use_cache) {
            window   = &gap_window;
            len      = &gap_len;
            head_pos = &gap_head;
        } else {
            local_window.assign(static_cast<std::size_t>(cap) * C, 0.0f);
            window   = &local_window;
            len      = &local_len;
            head_pos = &local_head;
        }

        std::vector<float> logits(static_cast<std::size_t>(T), 0.0f);
        for (int t = 0; t < T; ++t) {
            // Push y_p[t,:] into the ring window.
            const float* row = y_p + static_cast<std::size_t>(t) * C;
            float* dst = window->data() +
                         static_cast<std::size_t>(*head_pos) * C;
            std::memcpy(dst, row,
                        static_cast<std::size_t>(C) * sizeof(float));
            *head_pos = (*head_pos + 1) % cap;
            if (*len < cap) ++(*len);

            // Mean over the *len rows currently in the window. The ring's
            // physical layout is arbitrary; averaging is order-independent.
            const double inv = 1.0 / static_cast<double>(*len);
            double s = static_cast<double>(bias);
            for (int j = 0; j < C; ++j) {
                double sum = 0.0;
                for (int k = 0; k < *len; ++k) {
                    sum += static_cast<double>(
                        (*window)[static_cast<std::size_t>(k) * C + j]);
                }
                s += static_cast<double>(hW[j]) * (sum * inv);
            }
            logits[static_cast<std::size_t>(t)] = static_cast<float>(s);
        }

        out_logits = bt::Tensor::from_host_on(device, logits.data(),
                                              T, 1);
    }
};

// ─── BcResnet public ──────────────────────────────────────────────────────

BcResnet::BcResnet() : impl_(std::make_unique<Impl>()) {}
// Out-of-line below — needs the full type of BcResnetTrainState (defined in
// the training-surface section further down) so unique_ptr<Impl>::~unique_ptr
// can resolve Impl's implicit destructor.
BcResnet::BcResnet(BcResnet&&) noexcept = default;
BcResnet& BcResnet::operator=(BcResnet&&) noexcept = default;

BcResnet BcResnet::make(const BcResnetConfig& cfg, bt::Device device) {
    BcResnet m;
    m.impl_->cfg    = cfg;
    m.impl_->device = device;
    m.impl_->fused  = false;
    m.impl_->build();
    return m;
}

const BcResnetConfig& BcResnet::config() const { return impl_->cfg; }
bt::Device            BcResnet::device() const { return impl_->device; }
bool                  BcResnet::fused()  const { return impl_->fused; }

int BcResnet::receptive_field_frames() const {
    const auto& cfg = impl_->cfg;
    // Total INPUT receptive field of the final pooled logit, in frames: each
    // post-block-4 feature depends on conv_rf input frames, and the GAP
    // window pools the last conv_rf such features. The overall input window
    // is therefore conv_rf + gap_cap - 1.
    //   stem contributes (3-1)*1 = 2 frames; each block's DW conv contributes
    //   (k-1)*dil; pointwise convs (k=1) and residual projections add 0.
    int conv_rf = 2;
    conv_rf += (cfg.kernel_size - 1) * cfg.dil1;
    conv_rf += (cfg.kernel_size - 1) * cfg.dil2;
    conv_rf += (cfg.kernel_size - 1) * cfg.dil3;
    conv_rf += (cfg.kernel_size - 1) * cfg.dil4;
    conv_rf += 1;
    return conv_rf + impl_->gap_cap - 1;
}

int BcResnet::param_count() const {
    const auto& cfg = impl_->cfg;
    auto conv_params = [](int c_in, int c_out, int kL, int groups) {
        const int w = c_out * (c_in / groups) * kL;
        const int b = c_out;
        return w + b;
    };
    auto bn_params = [](int C) { return 2 * C; };   // gamma + beta only

    int n = 0;
    n += conv_params(cfg.n_mels, cfg.c0, 3, 1);
    if (!impl_->fused) n += bn_params(cfg.c0);

    auto block_params = [&](int ci, int co) {
        int s = 0;
        s += conv_params(ci, ci, cfg.kernel_size, ci);   // depthwise
        if (!impl_->fused) s += bn_params(ci);
        s += conv_params(ci, co, 1, 1);                  // pointwise
        if (!impl_->fused) s += bn_params(co);
        if (ci != co) {
            s += conv_params(ci, co, 1, 1);              // residual proj
            if (!impl_->fused) s += bn_params(co);
        }
        return s;
    };
    n += block_params(cfg.c0, cfg.c1);
    n += block_params(cfg.c1, cfg.c2);
    n += block_params(cfg.c2, cfg.c3);
    n += block_params(cfg.c3, cfg.c4);

    n += cfg.c4 + 1;   // head W + b
    return n;
}

void BcResnet::fuse_bn() {
    if (impl_->fused) return;
    fuse_one(impl_->stem, impl_->bn_stem);

    auto fuse_block = [](Block& b) {
        fuse_one(b.dw, b.bn_dw);
        fuse_one(b.pw, b.bn_pw);
        if (b.has_residual_proj) fuse_one(b.res_proj, b.bn_res);
    };
    fuse_block(impl_->b1);
    fuse_block(impl_->b2);
    fuse_block(impl_->b3);
    fuse_block(impl_->b4);

    impl_->fused = true;
}

void BcResnet::reset_streaming_state() {
    reset_one(impl_->stem);
    auto reset_block = [](Block& b) {
        reset_one(b.dw);
        reset_one(b.pw);
        if (b.has_residual_proj) reset_one(b.res_proj);
    };
    reset_block(impl_->b1);
    reset_block(impl_->b2);
    reset_block(impl_->b3);
    reset_block(impl_->b4);

    std::fill(impl_->gap_window.begin(), impl_->gap_window.end(), 0.0f);
    impl_->gap_len  = 0;
    impl_->gap_head = 0;
}

void BcResnet::forward(const bt::Tensor& log_mel,
                       bt::Tensor& out_logit) const {
    bt::Tensor per_frame;
    // forward_core mutates per-conv cache iff use_cache=true. For the one-shot
    // path it does not, so the const cast is safe — the impl object's
    // observable state is unchanged.
    auto& mut = const_cast<Impl&>(*impl_);
    mut.forward_core(log_mel, /*use_cache=*/false, per_frame);
    const int T = per_frame.rows;
    if (T <= 0) {
        out_logit = bt::Tensor::zeros_on(impl_->device, 1, 1, bt::Dtype::FP32);
        return;
    }
    // Last frame's logit — that's the one whose GAP covers all T frames.
    bt::Tensor host = per_frame.to(bt::Device::CPU);
    const float v = host.host_f32()[(T - 1)];
    bt::Tensor scalar_host = bt::Tensor::from_host_on(bt::Device::CPU,
                                                     &v, 1, 1);
    out_logit = scalar_host.to(impl_->device);
}

void BcResnet::forward_streaming(const bt::Tensor& new_frames,
                                 bt::Tensor& out_logit_per_frame) {
    impl_->forward_core(new_frames, /*use_cache=*/true, out_logit_per_frame);
}

// ─── Binary format ────────────────────────────────────────────────────────

// Walk every tensor that defines the model in a fixed order. For the unfused
// form (fused=false) the BN params are included; the fused form drops them.
// Defined as a free function rather than an Impl member so it stays adjacent
// to the read/write helpers it pairs with; Impl is public-within-this-TU
// since the impl is wholly inside this file.
namespace {

void for_each_tensor(BcResnet::Impl& m, bool include_bn,
                     std::vector<std::pair<std::string, bt::Tensor*>>& out) {
    auto add = [&](const char* n, bt::Tensor& t) {
        out.push_back({std::string(n), &t});
    };
    auto add_conv = [&](const std::string& prefix, ConvLayer& c) {
        add((prefix + ".W").c_str(), c.W);
        add((prefix + ".b").c_str(), c.b);
    };
    auto add_bn = [&](const std::string& prefix, BatchNorm1d& bn) {
        if (!include_bn) return;
        add((prefix + ".gamma").c_str(), bn.gamma);
        add((prefix + ".beta" ).c_str(), bn.beta);
        add((prefix + ".mean" ).c_str(), bn.mean);
        add((prefix + ".var"  ).c_str(), bn.var);
    };
    auto add_block = [&](const std::string& p, Block& b) {
        add_conv(p + ".dw", b.dw);
        add_bn  (p + ".bn_dw", b.bn_dw);
        add_conv(p + ".pw", b.pw);
        add_bn  (p + ".bn_pw", b.bn_pw);
        if (b.has_residual_proj) {
            add_conv(p + ".res_proj", b.res_proj);
            add_bn  (p + ".bn_res",   b.bn_res);
        }
    };

    add_conv("stem", m.stem);
    add_bn  ("bn_stem", m.bn_stem);
    add_block("b1", m.b1);
    add_block("b2", m.b2);
    add_block("b3", m.b3);
    add_block("b4", m.b4);
    add("head.W", m.head_W);
    add("head.b", m.head_b);
}

void write_config(Writer& w, const BcResnetConfig& c) {
    w.u32(static_cast<std::uint32_t>(c.n_mels));
    w.u32(static_cast<std::uint32_t>(c.c0));
    w.u32(static_cast<std::uint32_t>(c.c1));
    w.u32(static_cast<std::uint32_t>(c.c2));
    w.u32(static_cast<std::uint32_t>(c.c3));
    w.u32(static_cast<std::uint32_t>(c.c4));
    w.u32(static_cast<std::uint32_t>(c.kernel_size));
    w.u32(static_cast<std::uint32_t>(c.dil1));
    w.u32(static_cast<std::uint32_t>(c.dil2));
    w.u32(static_cast<std::uint32_t>(c.dil3));
    w.u32(static_cast<std::uint32_t>(c.dil4));
    w.f32(c.bn_eps);
}

BcResnetConfig read_config(Reader& r) {
    BcResnetConfig c;
    c.n_mels      = static_cast<int>(r.u32());
    c.c0          = static_cast<int>(r.u32());
    c.c1          = static_cast<int>(r.u32());
    c.c2          = static_cast<int>(r.u32());
    c.c3          = static_cast<int>(r.u32());
    c.c4          = static_cast<int>(r.u32());
    c.kernel_size = static_cast<int>(r.u32());
    c.dil1        = static_cast<int>(r.u32());
    c.dil2        = static_cast<int>(r.u32());
    c.dil3        = static_cast<int>(r.u32());
    c.dil4        = static_cast<int>(r.u32());
    c.bn_eps      = r.f32();
    return c;
}

}  // namespace

void BcResnet::save(const std::string& path, bool fused) const {
    if (fused && !impl_->fused) {
        fail("BcResnet::save",
             "asked to write a fused checkpoint, but fuse_bn() has not run");
    }
    if (!fused && impl_->fused) {
        fail("BcResnet::save",
             "asked to write an unfused checkpoint, but BN was already fused");
    }
    Writer w;
    w.open(path);
    w.u32(kMagic);
    w.u32(kVersion);
    write_config(w, impl_->cfg);
    w.u8(fused ? 1u : 0u);

    std::vector<std::pair<std::string, bt::Tensor*>> tensors;
    for_each_tensor(*impl_, /*include_bn=*/!fused, tensors);
    w.u32(static_cast<std::uint32_t>(tensors.size()));
    for (auto& [name, t] : tensors) write_tensor(w, name, *t);
}

BcResnet BcResnet::load(const std::string& path, bt::Device device) {
    Reader r;
    r.open(path);
    const std::uint32_t magic   = r.u32();
    const std::uint32_t version = r.u32();
    if (magic   != kMagic)   fail("BcResnet::load",
                                  "bad magic 0x" + std::to_string(magic));
    if (version != kVersion) fail("BcResnet::load",
                                  "unsupported version " + std::to_string(version));
    BcResnetConfig cfg = read_config(r);
    const bool on_disk_fused = (r.u8() != 0u);
    const std::uint32_t num_tensors = r.u32();

    BcResnet m = BcResnet::make(cfg, device);
    m.impl_->fused = on_disk_fused;
    if (on_disk_fused) {
        // Mark every BN inactive so the inference path skips it — load() of a
        // fused checkpoint must produce a model whose forward bypasses BN
        // (otherwise the default-built identity BN would run with its eps,
        // introducing tiny rounding versus the source model).
        auto clear_bn = [](BatchNorm1d& bn) {
            bn.present = false;
            bn.gamma = bt::Tensor{};
            bn.beta  = bt::Tensor{};
            bn.mean  = bt::Tensor{};
            bn.var   = bt::Tensor{};
        };
        clear_bn(m.impl_->bn_stem);
        clear_bn(m.impl_->b1.bn_dw); clear_bn(m.impl_->b1.bn_pw);
        clear_bn(m.impl_->b2.bn_dw); clear_bn(m.impl_->b2.bn_pw);
        clear_bn(m.impl_->b3.bn_dw); clear_bn(m.impl_->b3.bn_pw);
        clear_bn(m.impl_->b4.bn_dw); clear_bn(m.impl_->b4.bn_pw);
        if (m.impl_->b1.has_residual_proj) clear_bn(m.impl_->b1.bn_res);
        if (m.impl_->b2.has_residual_proj) clear_bn(m.impl_->b2.bn_res);
        if (m.impl_->b3.has_residual_proj) clear_bn(m.impl_->b3.bn_res);
        if (m.impl_->b4.has_residual_proj) clear_bn(m.impl_->b4.bn_res);
    }

    std::vector<std::pair<std::string, bt::Tensor*>> tensors;
    for_each_tensor(*m.impl_, /*include_bn=*/!on_disk_fused, tensors);
    if (tensors.size() != num_tensors) {
        fail("BcResnet::load",
             "tensor count mismatch: header=" + std::to_string(num_tensors) +
             " expected=" + std::to_string(tensors.size()));
    }
    for (auto& [name, t] : tensors) {
        *t = read_tensor(r, name, t->rows, t->cols, device);
    }

    if (!on_disk_fused) m.fuse_bn();
    return m;
}

// ─── Training surface (chunk 6) ───────────────────────────────────────────
//
// Hand-rolled CPU-only backward through the BC-ResNet, mirroring the chunk-5
// inference forward but with batched layout (N=B) and BN running in
// train-mode (batch statistics) so the running mean/var pick up the dataset
// distribution. Eval mode uses the running stats and is the same path the
// final fused checkpoint takes after fuse_bn().
//
// Topology (matches the chunk-5 header):
//   stem_conv → stem_bn → relu
//   block × 4: dw_conv → dw_bn → relu → pw_conv → pw_bn → +residual → relu
//   GAP over T → linear head → 1 logit per sample → fused BCE-with-logits.
//
// All ops are CPU FP32. The BN backward is hand-rolled here (no brotensor
// op).  Backward through causal_conv1d is implemented by left-padding the
// forward input by pad_left=dilation*(kL-1), running conv1d_backward_*
// against the padded buffer, and slicing the trailing L cols of dX_padded
// back into dX (the pad columns are discarded).

namespace {

// Splitmix64 — same routine brotensor::xavier_init advances internally; we
// keep one local copy so xavier_init_weights stays deterministic regardless
// of how many other things the global rng has been used for.
inline std::uint64_t sm64(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Fill a (rows, cols) FP32 tensor on CPU with xavier-uniform; bound = sqrt(6 /
// (fan_in + fan_out)). For conv weight (C_out, (C_in/groups)*kL), fan_in =
// (C_in/groups)*kL, fan_out = C_out * kL / groups — but PyTorch's default
// _calculate_fan_in_and_fan_out treats the weight as (out, in*kL) so fan_in =
// cols, fan_out = rows. We follow that convention.
void xavier_uniform_fill(bt::Tensor& t, std::uint64_t& seed) {
    bt::xavier_init(t, seed);
}

// Run a forward pass through BN with batch statistics (training mode).
// X is NCL (N, C*L). Updates running mean/var via 0.1 momentum (PyTorch
// default). Caches xhat / mean / inv_std for the backward pass.
void bn_train_forward(BatchNorm1d& bn, const bt::Tensor& X, int N, int C, int L,
                      bt::Tensor& Y,
                      std::vector<float>& xhat_cache,
                      std::vector<float>& mean_cache,
                      std::vector<float>& inv_std_cache,
                      float momentum = 0.1f) {
    if (X.device != bt::Device::CPU) {
        fail("bc_resnet::bn_train", "training BN is CPU-only");
    }
    const int M = N * L;
    const float* x = X.host_f32();
    Y = bt::Tensor::empty_on(bt::Device::CPU, X.rows, X.cols, bt::Dtype::FP32);
    float* y = Y.host_f32_mut();

    xhat_cache.assign(static_cast<std::size_t>(N) * C * L, 0.0f);
    mean_cache.assign(static_cast<std::size_t>(C), 0.0f);
    inv_std_cache.assign(static_cast<std::size_t>(C), 0.0f);

    float* g    = bn.gamma.host_f32_mut();
    float* be   = bn.beta .host_f32_mut();
    float* rmu  = bn.mean .host_f32_mut();
    float* rvar = bn.var  .host_f32_mut();

    for (int c = 0; c < C; ++c) {
        // mean & var over the (N, L) plane for channel c.
        double sum = 0.0;
        for (int n = 0; n < N; ++n) {
            const float* row = x + (static_cast<std::size_t>(n) * C + c) * L;
            for (int t = 0; t < L; ++t) sum += row[t];
        }
        const double mean = sum / static_cast<double>(M);
        double sq = 0.0;
        for (int n = 0; n < N; ++n) {
            const float* row = x + (static_cast<std::size_t>(n) * C + c) * L;
            for (int t = 0; t < L; ++t) {
                const double d = row[t] - mean;
                sq += d * d;
            }
        }
        const double var     = sq / static_cast<double>(M);
        const double inv_std = 1.0 / std::sqrt(var + bn.eps);

        mean_cache[static_cast<std::size_t>(c)]    = static_cast<float>(mean);
        inv_std_cache[static_cast<std::size_t>(c)] = static_cast<float>(inv_std);

        const float gc  = g[c];
        const float bec = be[c];
        for (int n = 0; n < N; ++n) {
            const float* xr = x + (static_cast<std::size_t>(n) * C + c) * L;
            float*       yr = y + (static_cast<std::size_t>(n) * C + c) * L;
            float*       xh = xhat_cache.data() +
                              (static_cast<std::size_t>(n) * C + c) * L;
            for (int t = 0; t < L; ++t) {
                const float h = static_cast<float>((xr[t] - mean) * inv_std);
                xh[t] = h;
                yr[t] = gc * h + bec;
            }
        }
        // Running stats update (PyTorch-style): the unbiased variance is used
        // for the running estimate (denominator M-1 instead of M).
        const double var_unbiased = (M > 1) ? sq / static_cast<double>(M - 1)
                                            : var;
        rmu[c]  = static_cast<float>((1.0 - momentum) * rmu[c]  +
                                     momentum * mean);
        rvar[c] = static_cast<float>((1.0 - momentum) * rvar[c] +
                                     momentum * var_unbiased);
    }
}

// Eval-mode BN forward: uses running mean/var. NCL layout. Y overwritten.
void bn_eval_forward(const BatchNorm1d& bn, const bt::Tensor& X, int N, int C,
                     int L, bt::Tensor& Y) {
    const float* x = X.host_f32();
    Y = bt::Tensor::empty_on(bt::Device::CPU, X.rows, X.cols, bt::Dtype::FP32);
    float* y = Y.host_f32_mut();
    const float* g  = bn.gamma.host_f32();
    const float* be = bn.beta .host_f32();
    const float* mu = bn.mean .host_f32();
    const float* va = bn.var  .host_f32();
    for (int c = 0; c < C; ++c) {
        const float inv = 1.0f / std::sqrt(va[c] + bn.eps);
        const float a   = g[c] * inv;
        const float b   = be[c] - mu[c] * g[c] * inv;
        for (int n = 0; n < N; ++n) {
            const float* xr = x + (static_cast<std::size_t>(n) * C + c) * L;
            float*       yr = y + (static_cast<std::size_t>(n) * C + c) * L;
            for (int t = 0; t < L; ++t) yr[t] = xr[t] * a + b;
        }
    }
}

// Standard BN backward over NCL (N, C*L). dX overwritten. dGamma, dBeta
// overwritten (not accumulated — the trainer aggregates by Adam's m/v).
void bn_train_backward(BatchNorm1d& bn,
                       const std::vector<float>& xhat,
                       const std::vector<float>& /*mean*/,
                       const std::vector<float>& inv_std,
                       const bt::Tensor& dY,
                       int N, int C, int L,
                       bt::Tensor& dX,
                       bt::Tensor& dGamma, bt::Tensor& dBeta) {
    const float* dy = dY.host_f32();
    dX = bt::Tensor::empty_on(bt::Device::CPU, dY.rows, dY.cols,
                              bt::Dtype::FP32);
    float* dx = dX.host_f32_mut();
    dGamma = bt::Tensor::zeros_on(bt::Device::CPU, C, 1, bt::Dtype::FP32);
    dBeta  = bt::Tensor::zeros_on(bt::Device::CPU, C, 1, bt::Dtype::FP32);
    float* dg = dGamma.host_f32_mut();
    float* db = dBeta .host_f32_mut();
    const float* g = bn.gamma.host_f32();
    const int M = N * L;

    for (int c = 0; c < C; ++c) {
        // Per-channel reductions: sum_dy, sum_dy_xhat, plus the accumulators
        // for dgamma/dbeta.
        double sum_dy      = 0.0;
        double sum_dy_xhat = 0.0;
        for (int n = 0; n < N; ++n) {
            const float* dyr = dy   + (static_cast<std::size_t>(n) * C + c) * L;
            const float* xh  = xhat.data() +
                               (static_cast<std::size_t>(n) * C + c) * L;
            for (int t = 0; t < L; ++t) {
                sum_dy      += dyr[t];
                sum_dy_xhat += static_cast<double>(dyr[t]) * xh[t];
            }
        }
        db[c] = static_cast<float>(sum_dy);
        dg[c] = static_cast<float>(sum_dy_xhat);

        // dx = (gamma * inv_std / M) * (M*dy - sum_dy - xhat*sum_dy_xhat)
        const double scale = static_cast<double>(g[c]) * inv_std[c] /
                             static_cast<double>(M);
        for (int n = 0; n < N; ++n) {
            const float* dyr = dy   + (static_cast<std::size_t>(n) * C + c) * L;
            const float* xh  = xhat.data() +
                               (static_cast<std::size_t>(n) * C + c) * L;
            float*       dxr = dx   + (static_cast<std::size_t>(n) * C + c) * L;
            for (int t = 0; t < L; ++t) {
                const double term = static_cast<double>(M) * dyr[t] -
                                    sum_dy -
                                    static_cast<double>(xh[t]) * sum_dy_xhat;
                dxr[t] = static_cast<float>(scale * term);
            }
        }
    }
}

// Left-pad an NCL tensor along L by `pad_left` zeros. Out shape (N, C*(L+pad)).
bt::Tensor left_pad_ncl(const bt::Tensor& X, int N, int C, int L, int pad) {
    if (pad == 0) return X.to(bt::Device::CPU);  // no-op clone
    bt::Tensor Y = bt::Tensor::zeros_on(bt::Device::CPU, N, C * (L + pad),
                                        bt::Dtype::FP32);
    const float* x = X.host_f32();
    float*       y = Y.host_f32_mut();
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* src = x + (static_cast<std::size_t>(n) * C + c) * L;
            float*       dst = y + (static_cast<std::size_t>(n) * C + c) *
                                    (L + pad) + pad;
            std::memcpy(dst, src,
                        static_cast<std::size_t>(L) * sizeof(float));
        }
    }
    return Y;
}

// Strip the leading `pad` cols per (n,c) row from an NCL (N, C*(L+pad)) tensor,
// returning a fresh NCL (N, C*L).
bt::Tensor strip_left_pad_ncl(const bt::Tensor& X, int N, int C, int L,
                              int pad) {
    if (pad == 0) return X.to(bt::Device::CPU);
    bt::Tensor Y = bt::Tensor::empty_on(bt::Device::CPU, N, C * L,
                                        bt::Dtype::FP32);
    const float* x = X.host_f32();
    float*       y = Y.host_f32_mut();
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* src = x + (static_cast<std::size_t>(n) * C + c) *
                                    (L + pad) + pad;
            float*       dst = y + (static_cast<std::size_t>(n) * C + c) * L;
            std::memcpy(dst, src,
                        static_cast<std::size_t>(L) * sizeof(float));
        }
    }
    return Y;
}

// Forward of one conv in training mode: store the padded forward input so the
// backward can use it for the weight grad. Y is (N, C_out*L) NCL.
void conv_train_forward(const ConvLayer& c, const bt::Tensor& X, int N, int L,
                        bt::Tensor& X_padded,    // (N, C_in*(L+pad_left))
                        bt::Tensor& Y) {
    X_padded = left_pad_ncl(X, N, c.in_channels, L, c.pad_left);
    bt::conv1d(X_padded, c.W, &c.b, N, c.in_channels, L + c.pad_left,
               c.out_channels, c.kernel_size, /*stride=*/1, /*padding=*/0,
               c.dilation, c.groups, Y);
}

// Backward of one conv. dY (N, C_out*L). dX overwritten (N, C_in*L). dW + dB
// accumulated into the provided grads (caller zeros).
void conv_train_backward(const ConvLayer& c, const bt::Tensor& X_padded,
                         const bt::Tensor& dY, int N, int L,
                         bt::Tensor& dX,
                         bt::Tensor& dW, bt::Tensor& dB) {
    // dX_padded via conv1d_backward_input on the padded buffer (padding=0).
    bt::Tensor dX_padded;
    bt::conv1d_backward_input(c.W, dY, N, c.in_channels, L + c.pad_left,
                              c.out_channels, c.kernel_size, /*stride=*/1,
                              /*padding=*/0, c.dilation, c.groups, dX_padded);
    dX = strip_left_pad_ncl(dX_padded, N, c.in_channels, L, c.pad_left);

    // dW accumulates from X_padded.
    bt::conv1d_backward_weight(X_padded, dY, N, c.in_channels, L + c.pad_left,
                               c.out_channels, c.kernel_size, /*stride=*/1,
                               /*padding=*/0, c.dilation, c.groups, dW);

    // dB accumulates from dY.
    bt::conv1d_backward_bias(dY, N, c.out_channels, L, dB);
}

// ReLU forward over NCL — writes Y, returns nothing. Mask = (x > 0) cached
// implicitly via X (we still have X around for backward).
void relu_train_forward(const bt::Tensor& X, bt::Tensor& Y) {
    Y = bt::Tensor::empty_on(bt::Device::CPU, X.rows, X.cols, bt::Dtype::FP32);
    const float* x = X.host_f32();
    float*       y = Y.host_f32_mut();
    const int n = X.rows * X.cols;
    for (int i = 0; i < n; ++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// Caches collected for a single training forward, enough to walk the whole
// graph back. Allocated once per train_step. (B, C*L) for everything below
// is NCL with N=B.
struct ConvCache {
    bt::Tensor X_padded;     // (B, C_in*(L+pad_left)) — what the W grad needs
};
struct BnCache {
    std::vector<float> xhat;     // (B, C, L) flat
    std::vector<float> mean;     // (C)
    std::vector<float> inv_std;  // (C)
};
struct BlockCache {
    bt::Tensor X_in;          // block input — needed for residual_proj backward
    ConvCache  dw, pw, res;
    BnCache    bn_dw, bn_pw, bn_res;
    bt::Tensor dw_pre;        // pre-relu dw branch (== bn_dw output) — needed
                              // to mask dw-branch backward through ReLU
    bt::Tensor pw_post_bn;    // pw + bn_pw output, pre-residual-add
    bt::Tensor pre_relu;      // pw_post_bn + residual (input to final ReLU)
};

struct TrainCache {
    bt::Tensor stem_X_padded;
    BnCache    bn_stem;
    bt::Tensor stem_pre_relu;
    BlockCache b1, b2, b3, b4;
    bt::Tensor y4;            // (B, c4*T)
    bt::Tensor pooled;        // (B, c4)
};

// Per-tensor Adam state.
struct AdamSlot {
    bt::Tensor m;
    bt::Tensor v;
};

}  // namespace

// The training state lives in a separate object hung off the Impl. Built
// lazily on first train_step() so untrained inference users don't pay for it.
struct BcResnetTrainState {
    int step = 0;

    // Every trainable tensor + its grad + Adam (m, v). The pointer goes back
    // into the model's Impl so adam_step rewrites the parameter in place.
    struct Slot {
        bt::Tensor* param;
        bt::Tensor  grad;
        bt::Tensor  m;
        bt::Tensor  v;
        std::string name;
    };
    std::vector<Slot> slots;

    void add(bt::Tensor* p, const std::string& name) {
        Slot s;
        s.param = p;
        s.grad  = bt::Tensor::zeros_on(bt::Device::CPU, p->rows, p->cols,
                                       bt::Dtype::FP32);
        s.m     = bt::Tensor::zeros_on(bt::Device::CPU, p->rows, p->cols,
                                       bt::Dtype::FP32);
        s.v     = bt::Tensor::zeros_on(bt::Device::CPU, p->rows, p->cols,
                                       bt::Dtype::FP32);
        s.name  = name;
        slots.push_back(std::move(s));
    }

    bt::Tensor& grad(const std::string& name) {
        for (auto& s : slots) if (s.name == name) return s.grad;
        throw std::runtime_error("brosoundml: train: grad not found: " + name);
    }

    void zero_grads() {
        for (auto& s : slots) s.grad.zero();
    }

    void adam(float lr) {
        ++step;
        for (auto& s : slots) {
            bt::adam_step(*s.param, s.grad, s.m, s.v,
                          lr, 0.9f, 0.999f, 1e-8f, step);
        }
    }
};

namespace {

void register_params(BcResnetTrainState& t, BcResnet::Impl& m) {
    auto add_conv = [&](const std::string& p, ConvLayer& c) {
        t.add(&c.W, p + ".W");
        t.add(&c.b, p + ".b");
    };
    auto add_bn = [&](const std::string& p, BatchNorm1d& bn) {
        t.add(&bn.gamma, p + ".gamma");
        t.add(&bn.beta,  p + ".beta");
    };
    auto add_block = [&](const std::string& p, Block& b) {
        add_conv(p + ".dw", b.dw);
        add_bn  (p + ".bn_dw", b.bn_dw);
        add_conv(p + ".pw", b.pw);
        add_bn  (p + ".bn_pw", b.bn_pw);
        if (b.has_residual_proj) {
            add_conv(p + ".res_proj", b.res_proj);
            add_bn  (p + ".bn_res",   b.bn_res);
        }
    };
    add_conv("stem", m.stem);
    add_bn  ("bn_stem", m.bn_stem);
    add_block("b1", m.b1);
    add_block("b2", m.b2);
    add_block("b3", m.b3);
    add_block("b4", m.b4);
    t.add(&m.head_W, "head.W");
    t.add(&m.head_b, "head.b");
}

// Forward through one residual block in training mode. X is (B, ci*T) NCL.
void block_train_forward(Block& blk, const bt::Tensor& X, int B, int ci,
                         int co, int T, BlockCache& bc, bt::Tensor& Y) {
    bc.X_in = X.to(bt::Device::CPU);   // deep copy for backward

    // Depthwise.
    bt::Tensor dw_conv_out;
    conv_train_forward(blk.dw, X, B, T, bc.dw.X_padded, dw_conv_out);
    bt::Tensor dw_bn_out;
    bn_train_forward(blk.bn_dw, dw_conv_out, B, ci, T, dw_bn_out,
                     bc.bn_dw.xhat, bc.bn_dw.mean, bc.bn_dw.inv_std);
    bc.dw_pre = dw_bn_out.to(bt::Device::CPU);   // for ReLU backward
    bt::Tensor dw_relu;
    relu_train_forward(dw_bn_out, dw_relu);

    // Pointwise.
    bt::Tensor pw_conv_out;
    conv_train_forward(blk.pw, dw_relu, B, T, bc.pw.X_padded, pw_conv_out);
    bt::Tensor pw_bn_out;
    bn_train_forward(blk.bn_pw, pw_conv_out, B, co, T, pw_bn_out,
                     bc.bn_pw.xhat, bc.bn_pw.mean, bc.bn_pw.inv_std);
    bc.pw_post_bn = pw_bn_out.to(bt::Device::CPU);

    // Residual.
    bt::Tensor res;
    if (blk.has_residual_proj) {
        bt::Tensor res_conv;
        conv_train_forward(blk.res_proj, X, B, T, bc.res.X_padded, res_conv);
        bt::Tensor res_bn;
        bn_train_forward(blk.bn_res, res_conv, B, co, T, res_bn,
                         bc.bn_res.xhat, bc.bn_res.mean, bc.bn_res.inv_std);
        res = res_bn;
    } else {
        res = X.to(bt::Device::CPU);
    }
    // pre_relu = pw_post_bn + res
    bt::Tensor pre_relu = bt::Tensor::empty_on(bt::Device::CPU, B, co * T,
                                                bt::Dtype::FP32);
    {
        const float* a = pw_bn_out.host_f32();
        const float* b = res.host_f32();
        float* o = pre_relu.host_f32_mut();
        const int n = B * co * T;
        for (int i = 0; i < n; ++i) o[i] = a[i] + b[i];
    }
    bc.pre_relu = pre_relu;

    relu_train_forward(pre_relu, Y);
}

// Eval-mode (running-stats BN, no caches) version.
void block_eval_forward(const Block& blk, const bt::Tensor& X, int B, int ci,
                        int co, int T, bt::Tensor& Y) {
    bt::Tensor dw_padded = left_pad_ncl(X, B, ci, T, blk.dw.pad_left);
    bt::Tensor dw_conv;
    bt::conv1d(dw_padded, blk.dw.W, &blk.dw.b, B, ci, T + blk.dw.pad_left,
               ci, blk.dw.kernel_size, 1, 0, blk.dw.dilation, blk.dw.groups,
               dw_conv);
    bt::Tensor dw_bn;
    bn_eval_forward(blk.bn_dw, dw_conv, B, ci, T, dw_bn);
    bt::Tensor dw_relu;
    relu_train_forward(dw_bn, dw_relu);

    bt::Tensor pw_padded = left_pad_ncl(dw_relu, B, ci, T, blk.pw.pad_left);
    bt::Tensor pw_conv;
    bt::conv1d(pw_padded, blk.pw.W, &blk.pw.b, B, ci, T + blk.pw.pad_left,
               co, blk.pw.kernel_size, 1, 0, blk.pw.dilation, blk.pw.groups,
               pw_conv);
    bt::Tensor pw_bn;
    bn_eval_forward(blk.bn_pw, pw_conv, B, co, T, pw_bn);

    bt::Tensor res;
    if (blk.has_residual_proj) {
        bt::Tensor r_padded = left_pad_ncl(X, B, ci, T, blk.res_proj.pad_left);
        bt::Tensor r_conv;
        bt::conv1d(r_padded, blk.res_proj.W, &blk.res_proj.b, B, ci,
                   T + blk.res_proj.pad_left, co, blk.res_proj.kernel_size, 1,
                   0, blk.res_proj.dilation, blk.res_proj.groups, r_conv);
        bn_eval_forward(blk.bn_res, r_conv, B, co, T, res);
    } else {
        res = X.to(bt::Device::CPU);
    }
    bt::Tensor pre_relu = bt::Tensor::empty_on(bt::Device::CPU, B, co * T,
                                                bt::Dtype::FP32);
    const float* a = pw_bn.host_f32();
    const float* b = res.host_f32();
    float* o = pre_relu.host_f32_mut();
    const int n = B * co * T;
    for (int i = 0; i < n; ++i) o[i] = a[i] + b[i];
    relu_train_forward(pre_relu, Y);
}

// Backward through one block. dY is the upstream grad (B, co*T).
// Updates dW/dB for every conv and dGamma/dBeta for every BN (via train_state
// names). Returns dX (B, ci*T) overwritten.
void block_train_backward(Block& blk, BlockCache& bc, const bt::Tensor& dY,
                          int B, int ci, int co, int T,
                          BcResnetTrainState& ts,
                          const std::string& prefix,
                          bt::Tensor& dX) {
    // Backward through the final ReLU using pre_relu as the mask source.
    bt::Tensor dPreRelu = bt::Tensor::empty_on(bt::Device::CPU, B, co * T,
                                                bt::Dtype::FP32);
    {
        const float* m = bc.pre_relu.host_f32();
        const float* g = dY.host_f32();
        float* o = dPreRelu.host_f32_mut();
        const int n = B * co * T;
        for (int i = 0; i < n; ++i) o[i] = (m[i] > 0.0f) ? g[i] : 0.0f;
    }

    // pre_relu = pw_post_bn + res → both branches receive dPreRelu.
    // ── PW BN backward ──
    bt::Tensor d_pw_conv, d_pw_g, d_pw_b;
    bn_train_backward(blk.bn_pw, bc.bn_pw.xhat, bc.bn_pw.mean,
                      bc.bn_pw.inv_std, dPreRelu, B, co, T,
                      d_pw_conv, d_pw_g, d_pw_b);
    bt::add_inplace(ts.grad(prefix + ".bn_pw.gamma"), d_pw_g);
    bt::add_inplace(ts.grad(prefix + ".bn_pw.beta"),  d_pw_b);

    // ── PW conv backward ──
    bt::Tensor d_dw_relu;
    conv_train_backward(blk.pw, bc.pw.X_padded, d_pw_conv, B, T,
                        d_dw_relu, ts.grad(prefix + ".pw.W"),
                        ts.grad(prefix + ".pw.b"));

    // ── DW ReLU backward ──
    bt::Tensor d_dw_bn = bt::Tensor::empty_on(bt::Device::CPU, B, ci * T,
                                               bt::Dtype::FP32);
    {
        const float* m = bc.dw_pre.host_f32();
        const float* g = d_dw_relu.host_f32();
        float* o = d_dw_bn.host_f32_mut();
        const int n = B * ci * T;
        for (int i = 0; i < n; ++i) o[i] = (m[i] > 0.0f) ? g[i] : 0.0f;
    }

    // ── DW BN backward ──
    bt::Tensor d_dw_conv, d_dw_g, d_dw_b;
    bn_train_backward(blk.bn_dw, bc.bn_dw.xhat, bc.bn_dw.mean,
                      bc.bn_dw.inv_std, d_dw_bn, B, ci, T,
                      d_dw_conv, d_dw_g, d_dw_b);
    bt::add_inplace(ts.grad(prefix + ".bn_dw.gamma"), d_dw_g);
    bt::add_inplace(ts.grad(prefix + ".bn_dw.beta"),  d_dw_b);

    // ── DW conv backward → dX_main (block input via main path) ──
    bt::Tensor dX_main;
    conv_train_backward(blk.dw, bc.dw.X_padded, d_dw_conv, B, T,
                        dX_main, ts.grad(prefix + ".dw.W"),
                        ts.grad(prefix + ".dw.b"));

    // ── Residual branch backward ──
    bt::Tensor dX_res;
    if (blk.has_residual_proj) {
        bt::Tensor d_res_conv, d_res_g, d_res_b;
        bn_train_backward(blk.bn_res, bc.bn_res.xhat, bc.bn_res.mean,
                          bc.bn_res.inv_std, dPreRelu, B, co, T,
                          d_res_conv, d_res_g, d_res_b);
        bt::add_inplace(ts.grad(prefix + ".bn_res.gamma"), d_res_g);
        bt::add_inplace(ts.grad(prefix + ".bn_res.beta"),  d_res_b);
        conv_train_backward(blk.res_proj, bc.res.X_padded, d_res_conv, B, T,
                            dX_res, ts.grad(prefix + ".res_proj.W"),
                            ts.grad(prefix + ".res_proj.b"));
    } else {
        // res = X identity → dX_res = dPreRelu (the residual branch's grad
        // equals the upstream pre-relu grad directly).
        dX_res = dPreRelu.to(bt::Device::CPU);
    }

    // dX = dX_main + dX_res
    dX = bt::Tensor::empty_on(bt::Device::CPU, B, ci * T, bt::Dtype::FP32);
    const float* a = dX_main.host_f32();
    const float* b = dX_res.host_f32();
    float* o = dX.host_f32_mut();
    const int n = B * ci * T;
    for (int i = 0; i < n; ++i) o[i] = a[i] + b[i];
}

}  // namespace

// ─── Public training surface ──────────────────────────────────────────────

void BcResnet::xavier_init_weights(std::uint64_t seed) {
    if (impl_->fused) {
        fail("BcResnet::xavier_init_weights",
             "cannot re-init a fused model — make() first");
    }
    if (impl_->device != bt::Device::CPU) {
        fail("BcResnet::xavier_init_weights",
             "training-mode init is CPU-only (consume on CPU, then move once "
             "fused)");
    }
    std::uint64_t s = seed;

    auto init_conv_w = [&](ConvLayer& c) {
        xavier_uniform_fill(c.W, s);
        c.b.zero();
    };
    auto init_bn_w = [&](BatchNorm1d& bn) {
        if (!bn.present) return;
        std::vector<float> ones(static_cast<std::size_t>(bn.channels), 1.0f);
        bn.gamma = bt::Tensor::from_host_on(bt::Device::CPU, ones.data(),
                                            bn.channels, 1);
        bn.beta.zero();
        bn.mean.zero();
        bn.var  = bt::Tensor::from_host_on(bt::Device::CPU, ones.data(),
                                            bn.channels, 1);
    };
    init_conv_w(impl_->stem);
    init_bn_w(impl_->bn_stem);

    auto init_block_w = [&](Block& b) {
        init_conv_w(b.dw); init_bn_w(b.bn_dw);
        init_conv_w(b.pw); init_bn_w(b.bn_pw);
        if (b.has_residual_proj) {
            init_conv_w(b.res_proj);
            init_bn_w(b.bn_res);
        }
    };
    init_block_w(impl_->b1);
    init_block_w(impl_->b2);
    init_block_w(impl_->b3);
    init_block_w(impl_->b4);

    xavier_uniform_fill(impl_->head_W, s);
    impl_->head_b.zero();

    // Splitmix64 advance to swallow any unused state — sm64 mutates `s`.
    (void)sm64(s);
}

// Forward + backward + adam step on one minibatch. Returns mean BCE loss.
float BcResnet::train_step(const bt::Tensor& mel_batch,
                           const bt::Tensor& labels,
                           int B, int T,
                           float lr, float pos_weight) {
    if (impl_->fused) {
        fail("BcResnet::train_step",
             "cannot train a fused model — make() + xavier_init_weights() "
             "first");
    }
    if (impl_->device != bt::Device::CPU) {
        fail("BcResnet::train_step", "training is CPU-only");
    }
    if (mel_batch.rows != B || mel_batch.cols != impl_->cfg.n_mels * T) {
        fail("BcResnet::train_step",
             "mel_batch shape (" + std::to_string(mel_batch.rows) + "," +
             std::to_string(mel_batch.cols) + ") != expected (" +
             std::to_string(B) + "," +
             std::to_string(impl_->cfg.n_mels * T) + ")");
    }
    if (labels.rows != B || labels.cols != 1) {
        fail("BcResnet::train_step",
             "labels shape (" + std::to_string(labels.rows) + "," +
             std::to_string(labels.cols) + ") != (" + std::to_string(B) +
             ",1)");
    }

    // Lazy build of the training state.
    if (!impl_->train_state) {
        impl_->train_state = std::make_unique<BcResnetTrainState>();
        register_params(*impl_->train_state, *impl_);
    }
    auto& ts = *impl_->train_state;
    ts.zero_grads();

    const auto& cfg = impl_->cfg;
    const int C0 = cfg.c0, C1 = cfg.c1, C2 = cfg.c2, C3 = cfg.c3, C4 = cfg.c4;

    TrainCache tc;

    // Stem.
    bt::Tensor stem_conv;
    conv_train_forward(impl_->stem, mel_batch, B, T, tc.stem_X_padded,
                       stem_conv);
    bt::Tensor stem_bn;
    bn_train_forward(impl_->bn_stem, stem_conv, B, C0, T, stem_bn,
                     tc.bn_stem.xhat, tc.bn_stem.mean, tc.bn_stem.inv_std);
    tc.stem_pre_relu = stem_bn.to(bt::Device::CPU);
    bt::Tensor stem_relu;
    relu_train_forward(stem_bn, stem_relu);

    // Blocks.
    bt::Tensor y1, y2, y3, y4;
    block_train_forward(impl_->b1, stem_relu, B, C0, C1, T, tc.b1, y1);
    block_train_forward(impl_->b2, y1,        B, C1, C2, T, tc.b2, y2);
    block_train_forward(impl_->b3, y2,        B, C2, C3, T, tc.b3, y3);
    block_train_forward(impl_->b4, y3,        B, C3, C4, T, tc.b4, y4);
    tc.y4 = y4.to(bt::Device::CPU);

    // GAP over T → (B, C4).
    tc.pooled = bt::Tensor::zeros_on(bt::Device::CPU, B, C4, bt::Dtype::FP32);
    {
        const float* y = y4.host_f32();
        float*       p = tc.pooled.host_f32_mut();
        const float inv = 1.0f / static_cast<float>(T);
        for (int n = 0; n < B; ++n) {
            for (int c = 0; c < C4; ++c) {
                double s = 0.0;
                const float* row = y + (static_cast<std::size_t>(n) * C4 + c)
                                       * T;
                for (int t = 0; t < T; ++t) s += row[t];
                p[static_cast<std::size_t>(n) * C4 + c] =
                    static_cast<float>(s) * inv;
            }
        }
    }

    // Head: linear (B, C4) → (B, 1) logits.
    bt::Tensor logits = bt::Tensor::zeros_on(bt::Device::CPU, B, 1,
                                              bt::Dtype::FP32);
    bt::linear_forward_batched(impl_->head_W, impl_->head_b, tc.pooled, logits);

    // Fused BCE. Targets (B,1), pos_weight scales the positive class.
    bt::Tensor probs       = bt::Tensor::zeros_on(bt::Device::CPU, B, 1,
                                                  bt::Dtype::FP32);
    bt::Tensor dLogits     = bt::Tensor::zeros_on(bt::Device::CPU, B, 1,
                                                  bt::Dtype::FP32);
    bt::Tensor loss_per    = bt::Tensor::zeros_on(bt::Device::CPU, B, 1,
                                                  bt::Dtype::FP32);
    bt::bce_with_logits_fused_batched(logits, labels, /*mask=*/nullptr,
                                      pos_weight, probs, dLogits, loss_per);

    // Mean loss = sum(loss_per) / B.  Scale dLogits by 1/B so the gradient is
    // the mean-reduction gradient (matches PyTorch's mean reduction).
    float total_loss = 0.0f;
    {
        const float* l = loss_per.host_f32();
        for (int i = 0; i < B; ++i) total_loss += l[i];
    }
    const float mean_loss = total_loss / static_cast<float>(B);
    bt::scale_inplace(dLogits, 1.0f / static_cast<float>(B));

    // Head backward.
    bt::Tensor dPooled = bt::Tensor::zeros_on(bt::Device::CPU, B, C4,
                                               bt::Dtype::FP32);
    bt::linear_backward_batched(impl_->head_W, tc.pooled, dLogits,
                                dPooled, ts.grad("head.W"), ts.grad("head.b"));

    // GAP backward: scatter dPooled[b,c] / T across T frames into dY4.
    bt::Tensor dY4 = bt::Tensor::empty_on(bt::Device::CPU, B, C4 * T,
                                           bt::Dtype::FP32);
    {
        const float* p = dPooled.host_f32();
        float*       y = dY4.host_f32_mut();
        const float inv = 1.0f / static_cast<float>(T);
        for (int n = 0; n < B; ++n) {
            for (int c = 0; c < C4; ++c) {
                const float v = p[static_cast<std::size_t>(n) * C4 + c] * inv;
                float* row = y + (static_cast<std::size_t>(n) * C4 + c) * T;
                for (int t = 0; t < T; ++t) row[t] = v;
            }
        }
    }

    // Blocks backward.
    bt::Tensor dY3, dY2, dY1, dStemRelu;
    block_train_backward(impl_->b4, tc.b4, dY4, B, C3, C4, T, ts, "b4", dY3);
    block_train_backward(impl_->b3, tc.b3, dY3, B, C2, C3, T, ts, "b3", dY2);
    block_train_backward(impl_->b2, tc.b2, dY2, B, C1, C2, T, ts, "b2", dY1);
    block_train_backward(impl_->b1, tc.b1, dY1, B, C0, C1, T, ts, "b1",
                         dStemRelu);

    // Stem ReLU backward.
    bt::Tensor dStemBn = bt::Tensor::empty_on(bt::Device::CPU, B, C0 * T,
                                                bt::Dtype::FP32);
    {
        const float* m = tc.stem_pre_relu.host_f32();
        const float* g = dStemRelu.host_f32();
        float* o = dStemBn.host_f32_mut();
        const int n = B * C0 * T;
        for (int i = 0; i < n; ++i) o[i] = (m[i] > 0.0f) ? g[i] : 0.0f;
    }

    // Stem BN backward.
    bt::Tensor dStemConv, dStemG, dStemBeta;
    bn_train_backward(impl_->bn_stem, tc.bn_stem.xhat, tc.bn_stem.mean,
                      tc.bn_stem.inv_std, dStemBn, B, C0, T,
                      dStemConv, dStemG, dStemBeta);
    bt::add_inplace(ts.grad("bn_stem.gamma"), dStemG);
    bt::add_inplace(ts.grad("bn_stem.beta"),  dStemBeta);

    // Stem conv backward. We discard dX into the input — not used.
    bt::Tensor dInput;
    conv_train_backward(impl_->stem, tc.stem_X_padded, dStemConv, B, T,
                        dInput, ts.grad("stem.W"), ts.grad("stem.b"));

    // Adam step.
    ts.adam(lr);

    return mean_loss;
}

BcResnet::EvalMetrics BcResnet::eval_step(const bt::Tensor& mel_batch,
                                          const bt::Tensor& labels,
                                          int B, int T, float pos_weight) {
    if (impl_->fused) {
        fail("BcResnet::eval_step",
             "fused models are inference-ready; call forward() instead");
    }
    if (impl_->device != bt::Device::CPU) {
        fail("BcResnet::eval_step", "training-mode eval is CPU-only");
    }
    const auto& cfg = impl_->cfg;
    const int C0 = cfg.c0, C1 = cfg.c1, C2 = cfg.c2, C3 = cfg.c3, C4 = cfg.c4;

    // Forward (eval) — running BN, no caches.
    bt::Tensor stem_padded = left_pad_ncl(mel_batch, B, cfg.n_mels, T,
                                          impl_->stem.pad_left);
    bt::Tensor stem_conv;
    bt::conv1d(stem_padded, impl_->stem.W, &impl_->stem.b, B, cfg.n_mels,
               T + impl_->stem.pad_left, C0, impl_->stem.kernel_size, 1, 0,
               impl_->stem.dilation, impl_->stem.groups, stem_conv);
    bt::Tensor stem_bn;
    bn_eval_forward(impl_->bn_stem, stem_conv, B, C0, T, stem_bn);
    bt::Tensor stem_relu;
    relu_train_forward(stem_bn, stem_relu);

    bt::Tensor y1, y2, y3, y4;
    block_eval_forward(impl_->b1, stem_relu, B, C0, C1, T, y1);
    block_eval_forward(impl_->b2, y1,        B, C1, C2, T, y2);
    block_eval_forward(impl_->b3, y2,        B, C2, C3, T, y3);
    block_eval_forward(impl_->b4, y3,        B, C3, C4, T, y4);

    bt::Tensor pooled = bt::Tensor::zeros_on(bt::Device::CPU, B, C4,
                                              bt::Dtype::FP32);
    {
        const float* y = y4.host_f32();
        float*       p = pooled.host_f32_mut();
        const float inv = 1.0f / static_cast<float>(T);
        for (int n = 0; n < B; ++n) {
            for (int c = 0; c < C4; ++c) {
                double s = 0.0;
                const float* row = y + (static_cast<std::size_t>(n) * C4 + c)
                                       * T;
                for (int t = 0; t < T; ++t) s += row[t];
                p[static_cast<std::size_t>(n) * C4 + c] =
                    static_cast<float>(s) * inv;
            }
        }
    }

    bt::Tensor logits = bt::Tensor::zeros_on(bt::Device::CPU, B, 1,
                                              bt::Dtype::FP32);
    bt::linear_forward_batched(impl_->head_W, impl_->head_b, pooled, logits);

    bt::Tensor probs    = bt::Tensor::zeros_on(bt::Device::CPU, B, 1,
                                                bt::Dtype::FP32);
    bt::Tensor dLogits  = bt::Tensor::zeros_on(bt::Device::CPU, B, 1,
                                                bt::Dtype::FP32);
    bt::Tensor loss_per = bt::Tensor::zeros_on(bt::Device::CPU, B, 1,
                                                bt::Dtype::FP32);
    bt::bce_with_logits_fused_batched(logits, labels, nullptr, pos_weight,
                                      probs, dLogits, loss_per);

    EvalMetrics m{};
    m.n = B;
    float total = 0.0f;
    int correct = 0, fp = 0, fn_ = 0, n_pos = 0, n_neg = 0;
    const float* lab = labels.host_f32();
    const float* pr  = probs.host_f32();
    const float* lp  = loss_per.host_f32();
    for (int i = 0; i < B; ++i) {
        total += lp[i];
        const int yhat = (pr[i] >= 0.5f) ? 1 : 0;
        const int yt   = (lab[i] >= 0.5f) ? 1 : 0;
        if (yhat == yt) ++correct;
        if (yt == 1) {
            ++n_pos;
            if (yhat == 0) ++fn_;
        } else {
            ++n_neg;
            if (yhat == 1) ++fp;
        }
    }
    m.loss     = total / static_cast<float>(B);
    m.accuracy = static_cast<float>(correct) / static_cast<float>(B);
    m.frr      = n_pos > 0 ? static_cast<float>(fn_) / n_pos : 0.0f;
    m.fpr      = n_neg > 0 ? static_cast<float>(fp)  / n_neg : 0.0f;
    return m;
}

BcResnet::~BcResnet() = default;

}  // namespace brosoundml
