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
BcResnet::~BcResnet() = default;
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

}  // namespace brosoundml
