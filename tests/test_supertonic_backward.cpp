// Finite-difference coverage for the Supertonic-3 backward FOUNDATION atoms
// (src/supertonic_backward.{h,cpp}): the conv / transpose / convnext-block
// input-backward kernels the later agents (style attention, flow field,
// vocoder, ECAPA) compose on top of. Built up one atom per chunk and each
// gradient-checked in isolation before the next relies on it.
//
// Verification methodology (mirrors test_kokoro_decoder_backward.cpp): the
// linear / smooth atoms (the convs and the transpose) are checked per element
// against a central finite difference; the LayerNorm-heavy ConvNeXt block is
// checked with a directional dot-product test instead (per-element FD of a
// stacked-norm Jacobian is unreliable in fp32). Each test owns a fixed RNG so
// test order never perturbs another's inputs. CPU-resident FP32.

#include "supertonic_backward.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;
namespace st = brosoundml::st_detail;

static int g_failures = 0;

static void expect_near(double a, double b, double abs_eps, double rel_eps,
                        const std::string& ctx) {
    const double d = std::fabs(a - b);
    const double m = std::fmax(std::fabs(a), std::fabs(b));
    if (d <= abs_eps || d <= rel_eps * m) return;
    std::printf("  FAIL %s: %.8g vs %.8g (|d|=%.3g)\n", ctx.c_str(), a, b, d);
    ++g_failures;
}

static Tensor make_random(int r, int c, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> u(-scale, scale);
    std::vector<float> v(static_cast<std::size_t>(r) * c);
    for (auto& x : v) x = u(rng);
    return Tensor::from_host_on(Device::CPU, v.data(), r, c);
}

// A conv weight in brotensor (Cout, (Cin/groups)*k) OIL layout.
static st::ConvW make_convw(int cout, int cin_pg, int k, bool bias,
                            std::mt19937& rng, float scale = 0.3f) {
    st::ConvW c;
    c.cout = cout; c.cin_pg = cin_pg; c.k = k; c.has_b = bias;
    c.w = make_random(cout, cin_pg * k, rng, scale);
    if (bias) c.b = make_random(cout, 1, rng, scale);
    return c;
}

// Per-element central finite-difference check of `analytic` w.r.t. tensor P.
static void fd_per_element(Tensor& P, const Tensor& analytic,
                           const std::function<double()>& loss_now,
                           double abs_tol, double rel_tol,
                           const std::string& name) {
    const float h = 1e-3f;
    for (int i = 0; i < P.size(); ++i) {
        float saved = P[i];
        P[i] = saved + h; double lp = loss_now();
        P[i] = saved - h; double lm = loss_now();
        P[i] = saved;
        double fd = (lp - lm) / (2.0 * h);
        expect_near(analytic[i], fd, abs_tol, rel_tol,
                    name + "[" + std::to_string(i) + "]");
    }
}

// Directional check: <analytic, v> against the central difference of the loss
// along a random direction v over the whole tensor P.
static void fd_directional(Tensor& P, const Tensor& analytic,
                           const std::function<double()>& loss_now,
                           std::mt19937& rng, const std::string& name,
                           double abs_tol = 4e-3, double rel_tol = 8e-3) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(static_cast<std::size_t>(P.size()));
    std::vector<float> saved(static_cast<std::size_t>(P.size()));
    double g_an = 0.0;
    for (int i = 0; i < P.size(); ++i) { v[i] = u(rng); saved[i] = P[i]; g_an += static_cast<double>(analytic[i]) * v[i]; }
    const float h = 1e-3f;
    for (int i = 0; i < P.size(); ++i) P[i] = saved[i] + h * v[i];
    double lp = loss_now();
    for (int i = 0; i < P.size(); ++i) P[i] = saved[i] - h * v[i];
    double lm = loss_now();
    for (int i = 0; i < P.size(); ++i) P[i] = saved[i];
    double g_fd = (lp - lm) / (2.0 * h);
    expect_near(g_an, g_fd, abs_tol, rel_tol, name);
}

static std::function<double()> mse_loss(const Tensor& out, const Tensor& target) {
    return [&out, &target]() {
        double s = 0.0;
        for (int i = 0; i < out.size(); ++i) { double d = out[i] - target[i]; s += 0.5 * d * d; }
        return s;
    };
}

// ─── pconv: 1×1 matmul path ─────────────────────────────────────────────────
static void test_pconv_1x1() {
    std::mt19937 rng(0x5701u);
    const int Cin = 6, Cout = 8, L = 10;
    st::ConvW c = make_convw(Cout, Cin, 1, /*bias=*/true, rng);

    Tensor x      = make_random(1, Cin * L, rng, 0.7f);
    Tensor target = make_random(1, Cout * L, rng, 0.5f);

    st::PConvCache cache; Tensor y;
    auto fwd = [&]() { st::pconv_forward_train(x, c, Cin, L, 1, 1, 0, 0, 0, y, cache); };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX; st::pconv_backward(c, cache, dY, dX);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_per_element(x, dX, loss, 2e-3, 5e-3, "pconv_1x1.dX");
}

// ─── pconv: padded k>1 conv path (groups=1) ─────────────────────────────────
static void test_pconv_k3() {
    std::mt19937 rng(0x5702u);
    const int Cin = 4, Cout = 5, L = 12, k = 3;
    st::ConvW c = make_convw(Cout, Cin, k, /*bias=*/true, rng);

    Tensor x      = make_random(1, Cin * L, rng, 0.7f);
    // k=3, dil=1, pad (1,1) -> L_out = L.
    Tensor target = make_random(1, Cout * L, rng, 0.5f);

    st::PConvCache cache; Tensor y;
    auto fwd = [&]() { st::pconv_forward_train(x, c, Cin, L, 1, 1, 1, 1, 0, y, cache); };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX; st::pconv_backward(c, cache, dY, dX);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_per_element(x, dX, loss, 2e-3, 5e-3, "pconv_k3.dX");
}

// ─── pconv: depthwise dilated edge-pad (groups=C), as used by ConvNeXt ───────
static void test_pconv_depthwise() {
    std::mt19937 rng(0x5703u);
    const int C = 6, L = 14, k = 5, dil = 2;
    const int pad = 2 * dil;
    st::ConvW c = make_convw(/*cout=*/C, /*cin_pg=*/1, k, /*bias=*/true, rng);

    Tensor x      = make_random(1, C * L, rng, 0.7f);
    Tensor target = make_random(1, C * L, rng, 0.5f);  // L_out = L

    st::PConvCache cache; Tensor y;
    auto fwd = [&]() { st::pconv_forward_train(x, c, C, L, dil, C, pad, pad, /*edge=*/2, y, cache); };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX; st::pconv_backward(c, cache, dY, dX);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_per_element(x, dX, loss, 2e-3, 5e-3, "pconv_dw.dX");
}

// ─── rconv: causal left-replicate-pad conv ──────────────────────────────────
static void test_rconv() {
    std::mt19937 rng(0x5704u);
    const int Cin = 3, Cout = 5, L = 12, k = 3, dil = 1;
    st::ConvW c = make_convw(Cout, Cin, k, /*bias=*/true, rng);

    Tensor x      = make_random(1, Cin * L, rng, 0.7f);
    Tensor target = make_random(1, Cout * L, rng, 0.5f);  // L_out = L

    st::PConvCache cache; Tensor y;
    auto fwd = [&]() { st::rconv_forward_train(x, c, Cin, L, dil, 1, y, cache); };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX; st::pconv_backward(c, cache, dY, dX);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_per_element(x, dX, loss, 2e-3, 5e-3, "rconv.dX");
}

// ─── seq transpose (host adjoint) ───────────────────────────────────────────
static void test_transpose() {
    std::mt19937 rng(0x5705u);
    const int R = 5, C = 7;
    Tensor x      = make_random(1, R * C, rng, 0.8f);  // [R,C] channel-major
    Tensor target = make_random(1, C * R, rng, 0.5f);  // [C,R]

    Tensor y;
    auto fwd = [&]() { st::transpose2d_forward(x, R, C, y); };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX; st::transpose2d_backward(dY, R, C, dX);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_per_element(x, dX, loss, 1e-3, 3e-3, "transpose.dX");
}

// ─── ConvNeXt-1D block ──────────────────────────────────────────────────────
static st::ConvNeXtBlock make_convnext(int C, int inter, int dil,
                                       std::mt19937& rng) {
    st::ConvNeXtBlock blk;
    blk.dil = dil;
    blk.dw  = make_convw(/*cout=*/C, /*cin_pg=*/1, /*k=*/5, /*bias=*/true, rng);  // depthwise
    blk.ln_g = make_random(C, 1, rng, 0.4f);
    blk.ln_b = make_random(C, 1, rng, 0.3f);
    blk.pw1 = make_convw(/*cout=*/inter, /*cin_pg=*/C, /*k=*/1, /*bias=*/true, rng);
    blk.pw2 = make_convw(/*cout=*/C, /*cin_pg=*/inter, /*k=*/1, /*bias=*/true, rng);
    return blk;
}

static void test_convnext(int dil, std::uint32_t seed, const char* label) {
    std::mt19937 rng(seed);
    const int C = 8, L = 16, inter = 16;
    st::ConvNeXtBlock blk = make_convnext(C, inter, dil, rng);

    Tensor x      = make_random(1, C * L, rng, 0.6f);
    Tensor target = make_random(1, C * L, rng, 0.5f);

    st::ConvNeXtCache cache; Tensor y;
    auto fwd = [&]() { st::convnext_block_forward_train(x, blk, C, L, dil, y, cache); };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX; st::convnext_block_backward(blk, cache, dY, dX);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_directional(x, dX, loss, rng, std::string(label) + ".dX");
}

// ─── tanh-gated style cross-attention ────────────────────────────────────────
static void test_style_attention() {
    std::mt19937 rng(0x57A1u);
    const int nh = 2, hd = 4, inner = nh * hd;  // kSpteHeads=2
    const int Cq = 8, Ck = 6, Cout = 8, Lq = 7, S = 5;

    st::StyleAttn A;
    A.wq = make_convw(/*cout=*/inner, /*cin_pg=*/Cq, 1, /*bias=*/true, rng);
    A.wk = make_convw(/*cout=*/inner, /*cin_pg=*/Ck, 1, /*bias=*/true, rng);
    A.wv = make_convw(/*cout=*/inner, /*cin_pg=*/Ck, 1, /*bias=*/true, rng);
    A.wo = make_convw(/*cout=*/Cout,  /*cin_pg=*/inner, 1, /*bias=*/true, rng);

    Tensor query  = make_random(1, Cq * Lq, rng, 0.6f);
    Tensor keysrc = make_random(1, Ck * S,  rng, 0.6f);  // FROZEN prototype
    Tensor valsrc = make_random(1, Ck * S,  rng, 0.6f);  // = style_ttl (target)
    Tensor target = make_random(1, Cout * Lq, rng, 0.5f);

    st::StyleAttnCache cache; Tensor y;
    auto fwd = [&]() {
        st::style_attention_forward_train(query, keysrc, valsrc, A, Lq, S, y, cache);
    };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dQuery, dValue;
    st::style_attention_backward(A, cache, dY, dQuery, dValue);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_directional(query,  dQuery, loss, rng, "style_attn.dQuery");
    fd_directional(valsrc, dValue, loss, rng, "style_attn.dValue");
}

// ─── RoPE text cross-attention block (attention + residual + post-attn LN) ────
static void test_rope_attention() {
    std::mt19937 rng(0x57B2u);
    const int C = 512, Ck = 256, L = 4, T = 5, half = 32;  // model-fixed dims

    st::VeRope A;
    A.conv_q = make_convw(/*cout=*/C, /*cin_pg=*/C,  1, /*bias=*/true, rng, 0.08f);
    A.conv_k = make_convw(/*cout=*/C, /*cin_pg=*/Ck, 1, /*bias=*/true, rng, 0.08f);
    A.conv_v = make_convw(/*cout=*/C, /*cin_pg=*/Ck, 1, /*bias=*/true, rng, 0.08f);
    A.conv_o = make_convw(/*cout=*/C, /*cin_pg=*/C,  1, /*bias=*/true, rng, 0.08f);
    A.theta.resize(half);
    for (int f = 0; f < half; ++f)
        A.theta[f] = 1.0f / std::pow(10000.0f, static_cast<float>(f) / half);
    A.norm_g = make_random(C, 1, rng, 0.4f);
    A.norm_b = make_random(C, 1, rng, 0.3f);

    Tensor h        = make_random(1, C  * L, rng, 0.4f);
    Tensor text_src = make_random(1, Ck * T, rng, 0.4f);  // FROZEN text encoding
    Tensor target   = make_random(1, C  * L, rng, 0.5f);

    st::RopeAttnCache cache; Tensor y;
    auto fwd = [&]() {
        st::rope_attention_forward_train(h, text_src, A, L, T, y, cache);
    };
    fwd();
    Tensor dY = Tensor::mat(1, y.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dH;
    st::rope_attention_backward(A, cache, dY, dH);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_directional(h, dH, loss, rng, "rope_attn.dH");
}

// ─── flow-field single-step backward (cond pass), synthetic 5-block field ─────
//
// Covers all FOUR VeBlock types — type 0 (two ConvNeXt sub-blocks, dil 1,2),
// type 1 FiLM (additive-only), type 2 RoPE text-attn, and TWO type-3 style
// blocks at different depths (so the value-grad accumulation AND the upstream
// threading past one style block to the next are both exercised). The field
// hidden width is fixed at 512 (the rope/style atoms hardcode kVeC=512); L/T/S
// are kept tiny. FD-checks the returned d(style_val) — the only escaping
// gradient — against a directional central difference of the field MSE.
static st::VeBlock make_convnext_pair_block(int C, int inter, std::mt19937& rng) {
    st::VeBlock blk; blk.type = 0;
    blk.conv.push_back(make_convnext(C, inter, /*dil=*/1, rng));  // sub-block s=0
    blk.conv.push_back(make_convnext(C, inter, /*dil=*/2, rng));  // sub-block s=1
    return blk;
}

static st::VeBlock make_film_block(int C, std::mt19937& rng) {
    st::VeBlock blk; blk.type = 1;
    blk.film.w = make_random(64, C, rng, 0.1f);   // [64,512] (in,out)
    blk.film.b = make_random(1,  C, rng, 0.1f);   // [1,512]
    return blk;
}

static st::VeBlock make_rope_block(int C, int Ck, int half, std::mt19937& rng) {
    st::VeBlock blk; blk.type = 2;
    blk.rope.conv_q = make_convw(/*cout=*/C, /*cin_pg=*/C,  1, true, rng, 0.08f);
    blk.rope.conv_k = make_convw(/*cout=*/C, /*cin_pg=*/Ck, 1, true, rng, 0.08f);
    blk.rope.conv_v = make_convw(/*cout=*/C, /*cin_pg=*/Ck, 1, true, rng, 0.08f);
    blk.rope.conv_o = make_convw(/*cout=*/C, /*cin_pg=*/C,  1, true, rng, 0.08f);
    blk.rope.theta.resize(half);
    for (int f = 0; f < half; ++f)
        blk.rope.theta[f] = 1.0f / std::pow(10000.0f, static_cast<float>(f) / half);
    blk.rope.norm_g = make_random(C, 1, rng, 0.4f);
    blk.rope.norm_b = make_random(C, 1, rng, 0.3f);
    return blk;
}

static st::VeBlock make_style_block(int C, int Ck, int inner, std::mt19937& rng) {
    st::VeBlock blk; blk.type = 3;
    blk.style.attn.wq = make_convw(/*cout=*/inner, /*cin_pg=*/C,     1, true, rng, 0.08f);
    blk.style.attn.wk = make_convw(/*cout=*/inner, /*cin_pg=*/Ck,    1, true, rng, 0.08f);
    blk.style.attn.wv = make_convw(/*cout=*/inner, /*cin_pg=*/Ck,    1, true, rng, 0.08f);
    blk.style.attn.wo = make_convw(/*cout=*/C,     /*cin_pg=*/inner, 1, true, rng, 0.08f);
    blk.style.norm_g  = make_random(C, 1, rng, 0.4f);
    blk.style.norm_b  = make_random(C, 1, rng, 0.3f);
    return blk;
}

static void test_field_backward() {
    std::mt19937 rng(0x57F3u);
    const int C = 512, Ck = 256, inner = 256, half = 32;  // model-fixed widths
    const int L = 4, T = 5, S = 5;                         // tiny seq dims

    std::vector<st::VeBlock> blocks;
    blocks.push_back(make_convnext_pair_block(C, /*inter=*/16, rng));  // type 0
    blocks.push_back(make_film_block(C, rng));                          // type 1
    blocks.push_back(make_style_block(C, Ck, inner, rng));             // type 3 (A)
    blocks.push_back(make_rope_block(C, Ck, half, rng));               // type 2
    blocks.push_back(make_style_block(C, Ck, inner, rng));             // type 3 (B)

    std::vector<st::ConvNeXtBlock> ve_last;
    ve_last.push_back(make_convnext(C, /*inter=*/16, /*dil=*/1, rng));
    ve_last.push_back(make_convnext(C, /*inter=*/16, /*dil=*/1, rng));

    st::ConvW ve_proj_out = make_convw(/*cout=*/144, /*cin_pg=*/C, 1, /*bias=*/false, rng, 0.08f);

    Tensor h0        = make_random(1, C  * L, rng, 0.3f);
    Tensor time_emb  = make_random(1, 64,     rng, 0.3f);  // FROZEN
    Tensor text_src  = make_random(1, Ck * T, rng, 0.3f);  // FROZEN text encoding
    Tensor style_key = make_random(1, Ck * S, rng, 0.3f);  // FROZEN prototype
    Tensor style_val = make_random(1, Ck * S, rng, 0.3f);  // = style (optimisation target)
    Tensor target    = make_random(1, 144 * L, rng, 0.4f);

    std::vector<float> onesh(static_cast<std::size_t>(L), 1.0f);
    Tensor onesL = Tensor::from_host_on(Device::CPU, onesh.data(), 1, L);

    st::FieldCache cache; Tensor field_out;
    auto fwd = [&]() {
        st::field_forward_train(h0, time_emb, text_src, style_key, style_val,
                                L, T, S, blocks, ve_last, ve_proj_out, onesL,
                                field_out, cache);
    };
    fwd();
    Tensor dY = Tensor::mat(1, field_out.size());
    for (int i = 0; i < dY.size(); ++i) dY[i] = field_out[i] - target[i];
    Tensor dStyleVal;
    st::field_backward(blocks, ve_last, ve_proj_out, cache, dY, dStyleVal);

    auto loss = [&]() { fwd(); return mse_loss(field_out, target)(); };
    fd_directional(style_val, dStyleVal, loss, rng, "field.dStyleVal");
}

// ─── vocoder decode (de-chunk + causal ConvNeXt + BN + leaky head + interleave)
static void test_vocoder() {
    std::mt19937 rng(0x5706u);
    const int D = 3, CC = 2, frames = 4, LF = CC * frames;   // 8
    const int C = 6, inter = 10, h1c = 5, BC = 4;
    const float inv_scale = 0.9f, bn_eps = 1.0e-5f, slope = 0.1f;

    st::ConvW conv_in = make_convw(C, D, /*k=*/3, /*bias=*/true, rng);   // causal k3
    std::vector<st::ConvNeXtBlock> blocks;
    blocks.push_back(make_convnext(C, inter, /*dil=*/1, rng));
    blocks.push_back(make_convnext(C, inter, /*dil=*/2, rng));
    st::ConvW head1 = make_convw(h1c, C, /*k=*/3, /*bias=*/true, rng);
    st::ConvW head2 = make_convw(BC, h1c, /*k=*/1, /*bias=*/true, rng);

    Tensor bn_g    = make_random(C, 1, rng, 0.5f);
    Tensor bn_b    = make_random(C, 1, rng, 0.3f);
    Tensor bn_mean = make_random(C, 1, rng, 0.2f);
    Tensor bn_var  = make_random(C, 1, rng, 0.4f);
    for (int i = 0; i < bn_var.size(); ++i) bn_var[i] = std::fabs(bn_var[i]) + 0.5f;

    std::vector<float> lmean(D), lstd(D);
    { std::uniform_real_distribution<float> u(-0.3f, 0.3f);
      for (int d = 0; d < D; ++d) { lmean[d] = u(rng); lstd[d] = std::fabs(u(rng)) + 0.5f; } }

    Tensor latent = make_random(1, D * CC * frames, rng, 0.6f);
    Tensor target = make_random(1, BC * LF, rng, 0.5f);

    st::VocoderCache cache; Tensor wave;
    auto fwd = [&]() {
        st::vocoder_decode_forward_train(latent, D, CC, frames, inv_scale, lmean, lstd,
                                         conv_in, blocks, bn_g, bn_b, bn_mean, bn_var,
                                         bn_eps, head1, slope, head2, wave, cache);
    };
    fwd();
    Tensor dW = Tensor::mat(1, wave.size());
    for (int i = 0; i < dW.size(); ++i) dW[i] = wave[i] - target[i];
    Tensor dL;
    st::vocoder_decode_backward(conv_in, blocks, head1, slope, head2, inv_scale,
                                lstd, cache, dW, dL);

    auto loss = [&]() { fwd(); return mse_loss(wave, target)(); };
    fd_directional(latent, dL, loss, rng, "vocoder.dLatent");
}

int main() {
    brotensor::init();
    test_pconv_1x1();
    test_pconv_k3();
    test_pconv_depthwise();
    test_rconv();
    test_transpose();
    test_convnext(/*dil=*/1, 0xC0DE01u, "convnext_dil1");
    test_convnext(/*dil=*/2, 0xC0DE02u, "convnext_dil2");
    test_style_attention();
    test_rope_attention();
    test_field_backward();
    test_vocoder();

    if (g_failures) { std::printf("test_supertonic_backward: %d failure(s)\n", g_failures); return 1; }
    std::printf("test_supertonic_backward: OK\n");
    return 0;
}
