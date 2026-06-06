// Finite-difference coverage for the Kokoro decoder backward atoms
// (src/kokoro_decoder_backward.{h,cpp}), built up one atom per chunk.
//
// Verification methodology. The shallow / smooth atoms (single AdaIN affine,
// the spectral head) are checked per element against a central finite
// difference. The instance-norm-heavy residual blocks are NOT: per-element FD
// of a stacked-GroupNorm Jacobian is unreliable in fp32 — a channel with small
// variance gives a near-singular 1/std, and the loss-difference subtraction
// hits the float roundoff floor — so those are checked with directional
// dot-product tests instead (<grad, v> against a central difference of the loss
// along a random direction v). Averaging over a whole tensor makes the signal
// large relative to the per-element noise, which is the standard robust check.
//
// Each test owns a fixed RNG so test order never perturbs another's inputs.
// CPU-resident FP32 (every composed op has a CPU backend).

#include "kokoro_decoder_backward.h"

#include "brosoundml/decoder_lora.h"

#include <brotensor/ops/concat.h>   // copy_d2d
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;

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

// AdaIN style→(γ,β) projection: a Linear (2C, style_dim). The assembly atoms
// ignore fc (they take γ,β directly), but DecoderLora's LoRA rides on it.
static constexpr int kStyleDim = 128;
static void fill_fc(brosoundml::AdaIN1dWeights& w, int C, std::mt19937& rng) {
    w.channels = C; w.style_dim = kStyleDim; w.eps = 1e-5f;
    w.fc.in_features = kStyleDim; w.fc.out_features = 2 * C;
    w.fc.W = [&]{ std::uniform_real_distribution<float> u(-0.1f, 0.1f);
        std::vector<float> v(static_cast<std::size_t>(2 * C) * kStyleDim);
        for (auto& x : v) x = u(rng);
        return Tensor::from_host_on(Device::CPU, v.data(), 2 * C, kStyleDim); }();
    w.fc.b = [&]{ std::uniform_real_distribution<float> u(-0.1f, 0.1f);
        std::vector<float> v(static_cast<std::size_t>(2 * C));
        for (auto& x : v) x = u(rng);
        return Tensor::from_host_on(Device::CPU, v.data(), 2 * C, 1); }();
}

static brosoundml::Conv1d make_conv(int in, int out, int k, int stride, int pad,
                                    std::mt19937& rng) {
    brosoundml::Conv1d c;
    c.in_channels = in; c.out_channels = out; c.kernel_size = k;
    c.stride = stride; c.padding = pad; c.dilation = 1; c.groups = 1;
    c.W = make_random(out, in * k, rng, 0.3f);   // (out, (in/groups)*kL), no bias
    return c;
}

// Per-element central finite-difference check of `analytic` w.r.t. tensor P.
// `loss_now` re-runs the forward against the current P and returns the scalar.
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
// along a random direction v over the whole tensor P. Robust to the per-element
// FD noise that stacked instance-norms produce in fp32.
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

// ─── AdaIN affine (instance-norm + style scale/shift) ───────────────────────
static void test_adain_affine() {
    std::mt19937 rng(0xADA1u);
    const int C = 8, L = 16;
    const float eps = 1e-5f;

    Tensor x      = make_random(1, C * L, rng, 0.8f);
    Tensor gamma  = make_random(C, 1, rng, 0.5f);
    Tensor beta   = make_random(C, 1, rng, 0.4f);
    Tensor target = make_random(1, C * L, rng, 0.5f);

    Tensor y, x_seq;
    auto fwd = [&]() { brosoundml::detail::adain_affine_forward(x, gamma, beta, C, L, eps, y, x_seq); };
    fwd();
    Tensor dY = Tensor::mat(1, C * L);
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX, dGamma, dBeta;
    brosoundml::detail::adain_affine_backward(x, x_seq, gamma, dY, C, L, eps, dX, dGamma, dBeta);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    // Single instance-norm: per-element FD is stable (the norm Jacobian needs
    // the ~0.5% tolerance standard for normalization layers).
    fd_per_element(x,     dX,     loss, 3e-3, 6e-3, "adain.dX");
    fd_per_element(gamma, dGamma, loss, 3e-3, 6e-3, "adain.dGamma");
    fd_per_element(beta,  dBeta,  loss, 3e-3, 6e-3, "adain.dBeta");
}

// ─── AdainResBlk1d residual block ───────────────────────────────────────────
static void test_adain_resblk(bool upsample, bool learned_sc,
                              int C_in, int C_out, int L_in,
                              std::uint32_t seed, const char* label) {
    std::mt19937 rng(seed);
    const float eps = 1e-5f;
    const int L_out = upsample ? 2 * L_in : L_in;

    brosoundml::AdainResBlk1dWeights w;
    w.channels_in = C_in; w.channels_out = C_out;
    w.upsample = upsample; w.learned_sc = learned_sc;
    w.conv1 = make_conv(C_in,  C_out, 3, 1, 1, rng);
    w.conv2 = make_conv(C_out, C_out, 3, 1, 1, rng);
    if (learned_sc) w.conv1x1 = make_conv(C_in, C_out, 1, 1, 0, rng);
    if (upsample) {
        w.pool_W = make_random(C_in, 3, rng, 0.3f);
        w.pool_b = Tensor::zeros_on(Device::CPU, C_in, 1);
    }

    Tensor x      = make_random(1, C_in * L_in, rng, 0.7f);
    Tensor gamma1 = make_random(C_in, 1, rng, 0.4f);
    Tensor beta1  = make_random(C_in, 1, rng, 0.3f);
    Tensor gamma2 = make_random(C_out, 1, rng, 0.4f);
    Tensor beta2  = make_random(C_out, 1, rng, 0.3f);
    Tensor target = make_random(1, C_out * L_out, rng, 0.5f);

    brosoundml::detail::AdainResBlkCache cache;
    int Lo = 0; Tensor y;
    auto fwd = [&]() { brosoundml::detail::adain_resblk_1d_forward_train(w, x, L_in, gamma1, beta1, gamma2, beta2, eps, Lo, y, cache); };
    fwd();
    Tensor dY = Tensor::mat(1, C_out * L_out);
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX, dG1, dB1, dG2, dB2;
    brosoundml::detail::adain_resblk_1d_backward(w, cache, gamma1, gamma2, dY, eps, dX, dG1, dB1, dG2, dB2);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    const std::string p = label;
    fd_directional(x,      dX,  loss, rng, p + ".dX");
    fd_directional(gamma1, dG1, loss, rng, p + ".dGamma1");
    fd_directional(beta1,  dB1, loss, rng, p + ".dBeta1");
    fd_directional(gamma2, dG2, loss, rng, p + ".dGamma2");
    fd_directional(beta2,  dB2, loss, rng, p + ".dBeta2");
}

static brosoundml::AdainResBlk1dWeights make_resblk(int C_in, int C_out,
                                                    bool upsample, bool learned_sc,
                                                    std::mt19937& rng) {
    brosoundml::AdainResBlk1dWeights w;
    w.channels_in = C_in; w.channels_out = C_out;
    w.upsample = upsample; w.learned_sc = learned_sc;
    w.conv1 = make_conv(C_in,  C_out, 3, 1, 1, rng);
    w.conv2 = make_conv(C_out, C_out, 3, 1, 1, rng);
    if (learned_sc) w.conv1x1 = make_conv(C_in, C_out, 1, 1, 0, rng);
    if (upsample) {                       // depthwise ConvTranspose1d (groups=C_in)
        w.pool_W = make_random(C_in, 3, rng, 0.3f);
        w.pool_b = Tensor::zeros_on(Device::CPU, C_in, 1);
    }
    fill_fc(w.norm1, C_in,  rng);
    fill_fc(w.norm2, C_out, rng);
    return w;
}

// ─── DecoderBackbone assembly (encode + decode chain, cat'd inputs) ─────────
static void test_decoder_backbone() {
    std::mt19937 rng(0xB0BAu);
    const float eps = 1e-5f;
    // Tiny faithful miniature: a non-upsampling encode, two cat-fed decode
    // blocks (the second upsamples → ends the cat region), then a plain block.
    const int T = 8;
    const int C_asr = 6, Ca = 4, Cenc = 10, Cdec = 8, Cgen = 5;

    brosoundml::DecoderBackbone bb;
    bb.F0_conv = make_conv(1, 1, 3, 2, 1, rng);          // (1,2T) -> (1,T)
    bb.N_conv  = make_conv(1, 1, 3, 2, 1, rng);
    bb.asr_res = make_conv(C_asr, Ca, 1, 1, 0, rng);     // 1x1 projection
    bb.encode  = make_resblk(C_asr + 2, Cenc, false, true, rng);
    bb.decode.push_back(make_resblk(Cenc + Ca + 2, Cenc, false, true, rng));  // cat
    bb.decode.push_back(make_resblk(Cenc + Ca + 2, Cdec, true,  true, rng));  // cat + 2x up
    bb.decode.push_back(make_resblk(Cdec,          Cgen, false, true, rng));  // no cat

    auto mkaff = [&](int Cin, int Cout) {
        brosoundml::detail::BlockAffines a;
        a.g1 = make_random(Cin,  1, rng, 0.4f); a.b1 = make_random(Cin,  1, rng, 0.3f);
        a.g2 = make_random(Cout, 1, rng, 0.4f); a.b2 = make_random(Cout, 1, rng, 0.3f);
        return a;
    };
    std::vector<brosoundml::detail::BlockAffines> aff;
    aff.push_back(mkaff(C_asr + 2,    Cenc));   // encode
    aff.push_back(mkaff(Cenc + Ca + 2, Cenc));  // decode[0]
    aff.push_back(mkaff(Cenc + Ca + 2, Cdec));  // decode[1]
    aff.push_back(mkaff(Cdec,          Cgen));  // decode[2]

    Tensor asr = make_random(1, C_asr * T, rng, 0.7f);
    Tensor F0  = make_random(1, 2 * T,     rng, 0.5f);
    Tensor N   = make_random(1, 2 * T,     rng, 0.5f);
    const int L_gen = 2 * T;                    // decode[1] upsamples once
    Tensor target = make_random(1, Cgen * L_gen, rng, 0.5f);

    brosoundml::detail::BackboneCache cache;
    Tensor gen_in;
    auto fwd = [&]() {
        brosoundml::detail::decoder_backbone_forward_train(bb, asr, F0, N, T, aff,
                                                           eps, gen_in, cache);
    };
    fwd();
    Tensor dGenIn = Tensor::mat(1, gen_in.size());
    for (int i = 0; i < dGenIn.size(); ++i) dGenIn[i] = gen_in[i] - target[i];
    std::vector<brosoundml::detail::BlockAffines> dAff;
    brosoundml::detail::decoder_backbone_backward(bb, cache, aff, dGenIn, eps, dAff);

    auto loss = [&]() { fwd(); return mse_loss(gen_in, target)(); };
    const char* names[4] = {"encode", "decode0", "decode1", "decode2"};
    for (int k = 0; k < 4; ++k) {
        const std::string p = std::string("backbone.") + names[k];
        fd_directional(aff[k].g1, dAff[k].g1, loss, rng, p + ".dG1");
        fd_directional(aff[k].b1, dAff[k].b1, loss, rng, p + ".dB1");
        fd_directional(aff[k].g2, dAff[k].g2, loss, rng, p + ".dG2");
        fd_directional(aff[k].b2, dAff[k].b2, loss, rng, p + ".dB2");
    }
}

// ─── AdaINResBlock1 (generator residual block, Snake1D) ─────────────────────
static Tensor make_alpha(int C, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0.5f, 1.5f);  // away from 0 (snake 1/alpha)
    std::vector<float> v(static_cast<std::size_t>(C));
    for (auto& x : v) x = u(rng);
    return Tensor::from_host_on(Device::CPU, v.data(), C, 1);
}

static brosoundml::AdaINResBlock1Weights make_resblock1(int C, std::mt19937& rng) {
    const int k = 3; const int dils[3] = {1, 2, 3};
    brosoundml::AdaINResBlock1Weights w;
    w.channels = C; w.kernel_size = k; w.dilations = {1, 2, 3};
    for (int i = 0; i < 3; ++i) {
        w.convs1[i] = make_conv(C, C, k, 1, dils[i] * (k - 1) / 2, rng);
        w.convs1[i].dilation = dils[i];
        w.convs2[i] = make_conv(C, C, k, 1, (k - 1) / 2, rng);
        w.alpha1[i] = make_alpha(C, rng);
        w.alpha2[i] = make_alpha(C, rng);
        fill_fc(w.adain1[i], C, rng);
        fill_fc(w.adain2[i], C, rng);
    }
    return w;
}

static void test_adain_resblock1() {
    std::mt19937 rng(0x5A4Eu);
    const int C = 8, L = 16, k = 3;
    const int dils[3] = {1, 2, 3};

    brosoundml::AdaINResBlock1Weights w;
    w.channels = C; w.kernel_size = k; w.dilations = {1, 2, 3};
    for (int i = 0; i < 3; ++i) {
        w.convs1[i] = make_conv(C, C, k, 1, dils[i] * (k - 1) / 2, rng);
        w.convs1[i].dilation = dils[i];
        w.convs2[i] = make_conv(C, C, k, 1, (k - 1) / 2, rng);
        w.alpha1[i] = make_alpha(C, rng);
        w.alpha2[i] = make_alpha(C, rng);
        w.adain1[i].eps = 1e-5f;
        w.adain2[i].eps = 1e-5f;
    }

    std::array<Tensor, 3> g1, b1, g2, b2;
    for (int i = 0; i < 3; ++i) {
        g1[i] = make_random(C, 1, rng, 0.4f); b1[i] = make_random(C, 1, rng, 0.3f);
        g2[i] = make_random(C, 1, rng, 0.4f); b2[i] = make_random(C, 1, rng, 0.3f);
    }
    Tensor x      = make_random(1, C * L, rng, 0.6f);
    Tensor target = make_random(1, C * L, rng, 0.5f);

    brosoundml::detail::AdainResBlock1Cache cache;
    Tensor y;
    auto fwd = [&]() { brosoundml::detail::adain_resblock1_forward_train(w, x, C, L, g1, b1, g2, b2, y, cache); };
    fwd();
    Tensor dY = Tensor::mat(1, C * L);
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];
    Tensor dX;
    std::array<Tensor, 3> dG1, dB1, dG2, dB2;
    brosoundml::detail::adain_resblock1_backward(w, cache, g1, g2, dY, dX, dG1, dB1, dG2, dB2);

    auto loss = [&]() { fwd(); return mse_loss(y, target)(); };
    fd_directional(x, dX, loss, rng, "resblock1.dX");
    for (int i = 0; i < 3; ++i) {
        const std::string s = std::to_string(i);
        fd_directional(g1[i], dG1[i], loss, rng, "resblock1.dG1." + s);
        fd_directional(b1[i], dB1[i], loss, rng, "resblock1.dB1." + s);
        fd_directional(g2[i], dG2[i], loss, rng, "resblock1.dG2." + s);
        fd_directional(b2[i], dB2[i], loss, rng, "resblock1.dB2." + s);
    }
}

// ─── Spectral head (conv_post -> exp/sin polar -> iSTFT) ────────────────────
static Tensor make_hann(int win) {
    std::vector<float> w(static_cast<std::size_t>(win));
    const double pi = 3.14159265358979323846;
    for (int n = 0; n < win; ++n)
        w[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * pi * n / win));  // periodic
    return Tensor::from_host_on(Device::CPU, w.data(), 1, win);
}

static void test_spectral_head() {
    std::mt19937 rng(0x57EDu);
    const int C = 4, L = 8, n_fft = 16, hop = 4, win = 16;
    const int C_post = n_fft + 2;
    const int signal_len = (L - 1) * hop;

    brosoundml::Conv1d conv_post = make_conv(C, C_post, 7, 1, 3, rng);
    for (int i = 0; i < conv_post.W.size(); ++i) conv_post.W[i] *= 0.3f;  // modest exp() magnitude

    Tensor window = make_hann(win);
    Tensor x      = make_random(1, C * L, rng, 0.4f);
    Tensor target = make_random(1, signal_len, rng, 0.4f);

    brosoundml::detail::SpectralHeadCache cache;
    Tensor audio;
    auto fwd = [&]() { brosoundml::detail::spectral_head_forward_train(conv_post, x, C, L, window, n_fft, hop, win, audio, cache); };
    fwd();
    Tensor dAudio = Tensor::mat(1, signal_len);
    for (int i = 0; i < dAudio.size(); ++i) dAudio[i] = audio[i] - target[i];
    Tensor dX;
    brosoundml::detail::spectral_head_backward(conv_post, cache, window, n_fft, hop, win, dAudio, dX);

    auto loss = [&]() { fwd(); return mse_loss(audio, target)(); };
    // No GroupNorm here — the conv/exp/sin/iSTFT chain is smooth, so per-element FD is tight.
    fd_per_element(x, dX, loss, 2e-3, 6e-3, "spectral.dX");
}

// ─── Generator assembly (upsample loop + noise branch + spectral head) ──────
static void test_generator() {
    std::mt19937 rng(0x6E1Eu);
    // Tiny faithful miniature: 2 upsample stages, 2 kernels each, a noise
    // branch per stage, the final stage's reflect-pad, and the spectral head.
    brosoundml::Generator gen;
    gen.num_upsamples = 2;
    gen.num_kernels   = 2;
    gen.n_fft = 8; gen.hop_size = 4; gen.win_size = 8;

    // ups: 8->4 (k4 s2 p1), 4->2 (k4 s2 p1).  init_C = 8.
    const int Cin[2]  = {8, 4};
    const int Cout[2] = {4, 2};
    for (int i = 0; i < 2; ++i) {
        gen.ups_C_in.push_back(Cin[i]);
        gen.ups_C_out.push_back(Cout[i]);
        gen.ups_k.push_back(4);
        gen.ups_stride.push_back(2);
        gen.ups_pad.push_back(1);
        // ConvTranspose weight layout: (C_in, C_out*kL).
        gen.ups_W.push_back(make_random(Cin[i], Cout[i] * 4, rng, 0.3f));
        gen.ups_b.push_back(Tensor::zeros_on(Device::CPU, Cout[i], 1));
    }
    // noise_convs map har(frames=17, C=1) → (Cout[i], L_src) matching each stage:
    // stage0 needs L_src=8 (17→8 via k4 s2 p1); stage1 needs L_src=17 (k3 s1 p1).
    gen.noise_convs.push_back(make_conv(1, 4, 4, 2, 1, rng));
    gen.noise_convs.push_back(make_conv(1, 2, 3, 1, 1, rng));
    gen.noise_res.push_back(make_resblock1(4, rng));
    gen.noise_res.push_back(make_resblock1(2, rng));
    gen.resblocks.push_back(make_resblock1(4, rng));  // stage0
    gen.resblocks.push_back(make_resblock1(4, rng));
    gen.resblocks.push_back(make_resblock1(2, rng));  // stage1
    gen.resblocks.push_back(make_resblock1(2, rng));
    gen.conv_post = make_conv(2, gen.n_fft + 2, 7, 1, 3, rng);
    for (int i = 0; i < gen.conv_post.W.size(); ++i) gen.conv_post.W[i] *= 0.3f;  // bound exp()

    auto mkraff = [&](int C) {
        brosoundml::detail::ResBlock1Affines a;
        for (int i = 0; i < 3; ++i) {
            a.g1[i] = make_random(C, 1, rng, 0.4f); a.b1[i] = make_random(C, 1, rng, 0.3f);
            a.g2[i] = make_random(C, 1, rng, 0.4f); a.b2[i] = make_random(C, 1, rng, 0.3f);
        }
        return a;
    };
    std::vector<brosoundml::detail::ResBlock1Affines> naff = {mkraff(4), mkraff(2)};
    std::vector<brosoundml::detail::ResBlock1Affines> raff = {
        mkraff(4), mkraff(4), mkraff(2), mkraff(2)};

    const int L_in = 4, frames = 17;
    Tensor gen_in = make_random(1, 8 * L_in, rng, 0.5f);
    Tensor har    = make_random(1, 1 * frames, rng, 0.4f);
    const int signal_len = (17 - 1) * gen.hop_size;   // last stage L = 17
    Tensor target = make_random(1, signal_len, rng, 0.4f);

    brosoundml::detail::GeneratorCache cache;
    Tensor audio;
    auto fwd = [&]() {
        brosoundml::detail::generator_forward_train(gen, gen_in, L_in, har, frames,
                                                    naff, raff, audio, cache);
    };
    fwd();
    Tensor dAudio = Tensor::mat(1, audio.size());
    for (int i = 0; i < dAudio.size(); ++i) dAudio[i] = audio[i] - target[i];
    std::vector<brosoundml::detail::ResBlock1Affines> dnaff, draff;
    Tensor dGenIn;  // grad into the generator input (checked in the DecoderLora e2e test)
    brosoundml::detail::generator_backward(gen, cache, naff, raff, dAudio, dGenIn, dnaff, draff);

    auto loss = [&]() { fwd(); return mse_loss(audio, target)(); };
    auto check_block = [&](brosoundml::detail::ResBlock1Affines& a,
                           brosoundml::detail::ResBlock1Affines& d,
                           const std::string& p) {
        for (int i = 0; i < 3; ++i) {
            const std::string s = std::to_string(i);
            fd_directional(a.g1[i], d.g1[i], loss, rng, p + ".g1." + s);
            fd_directional(a.b1[i], d.b1[i], loss, rng, p + ".b1." + s);
            fd_directional(a.g2[i], d.g2[i], loss, rng, p + ".g2." + s);
            fd_directional(a.b2[i], d.b2[i], loss, rng, p + ".b2." + s);
        }
    };
    for (int i = 0; i < 2; ++i)
        check_block(naff[i], dnaff[i], "gen.noise" + std::to_string(i));
    for (int i = 0; i < 4; ++i)
        check_block(raff[i], draff[i], "gen.res" + std::to_string(i));
}

// ─── DecoderLora end-to-end (neutral identity + grad check + checkpoint) ────
static void build_combined(std::mt19937& rng, brosoundml::DecoderBackbone& bb,
                           brosoundml::Generator& gen) {
    // Backbone (T=4): asr(6) → encode 8→10 → decode0 16→10 → decode1 16→8 (2x up).
    bb.F0_conv = make_conv(1, 1, 3, 2, 1, rng);
    bb.N_conv  = make_conv(1, 1, 3, 2, 1, rng);
    bb.asr_res = make_conv(6, 4, 1, 1, 0, rng);
    bb.encode  = make_resblk(8, 10, false, true, rng);
    bb.decode.push_back(make_resblk(16, 10, false, true, rng));
    bb.decode.push_back(make_resblk(16, 8,  true,  true, rng));

    // Generator (init_C=8, L_in=8): ups 8→4→2, 2 kernels, har frames=33.
    gen.num_upsamples = 2; gen.num_kernels = 2;
    gen.n_fft = 8; gen.hop_size = 4; gen.win_size = 8;
    const int Cin[2] = {8, 4}, Cout[2] = {4, 2};
    for (int i = 0; i < 2; ++i) {
        gen.ups_C_in.push_back(Cin[i]); gen.ups_C_out.push_back(Cout[i]);
        gen.ups_k.push_back(4); gen.ups_stride.push_back(2); gen.ups_pad.push_back(1);
        gen.ups_W.push_back(make_random(Cin[i], Cout[i] * 4, rng, 0.3f));
        gen.ups_b.push_back(Tensor::zeros_on(Device::CPU, Cout[i], 1));
    }
    gen.noise_convs.push_back(make_conv(1, 4, 4, 2, 1, rng));  // 33→16
    gen.noise_convs.push_back(make_conv(1, 2, 3, 1, 1, rng));  // 33→33
    gen.noise_res.push_back(make_resblock1(4, rng));
    gen.noise_res.push_back(make_resblock1(2, rng));
    gen.resblocks.push_back(make_resblock1(4, rng));
    gen.resblocks.push_back(make_resblock1(4, rng));
    gen.resblocks.push_back(make_resblock1(2, rng));
    gen.resblocks.push_back(make_resblock1(2, rng));
    gen.conv_post = make_conv(2, gen.n_fft + 2, 7, 1, 3, rng);
    for (int i = 0; i < gen.conv_post.W.size(); ++i) gen.conv_post.W[i] *= 0.3f;
}

static void test_decoder_lora() {
    std::mt19937 rng(0xE3071Cu);
    brosoundml::DecoderBackbone bb;
    brosoundml::Generator gen;
    build_combined(rng, bb, gen);

    const int T = 4, frames = 33;
    Tensor asr   = make_random(1, 6 * T, rng, 0.6f);
    Tensor F0    = make_random(1, 2 * T, rng, 0.5f);
    Tensor N     = make_random(1, 2 * T, rng, 0.5f);
    Tensor ref_s = make_random(1, 2 * kStyleDim, rng, 0.5f);
    Tensor har   = make_random(1, 1 * frames, rng, 0.4f);
    brosoundml::DecoderLoraContext ctx{&asr, &F0, &N, &ref_s, &har, T, frames};

    brosoundml::DecoderLora lora =
        brosoundml::DecoderLora::make(bb, gen, /*rank=*/4, /*scale=*/1.0f, Device::CPU, 123);

    // ── Neutral identity: cond=0 reproduces the base decoder (gate g(0)=0). ──
    Tensor gen_in_ref;
    bb.forward(asr, F0, N, ref_s, T, gen_in_ref);
    Tensor style_dec = Tensor::zeros_on(Device::CPU, 1, kStyleDim);
    brotensor::copy_d2d(ref_s, 0, style_dec, 0, kStyleDim);
    Tensor audio_ref;
    gen.forward(gen_in_ref, 2 * T, har, frames, style_dec, audio_ref);

    Tensor vad0 = Tensor::zeros_on(Device::CPU, 3, 1);
    Tensor audio0;
    lora.forward(bb, gen, ctx, vad0, audio0);
    double maxd = 0.0;
    for (int i = 0; i < audio0.size(); ++i)
        maxd = std::fmax(maxd, std::fabs(audio0[i] - audio_ref[i]));
    expect_near(maxd, 0.0, 1e-4, 0.0, "decoder_lora.neutral_identity");

    // ── End-to-end gradient check (gate opened so the LoRA path is live). ──
    Tensor& W2 = lora.gate().W2;     // (rank, hidden)
    std::uniform_real_distribution<float> u(-0.2f, 0.2f);
    for (int i = 0; i < W2.size(); ++i) W2[i] = u(rng);

    float vadv[3] = {0.5f, -0.3f, 0.7f};
    Tensor vad = Tensor::from_host_on(Device::CPU, vadv, 3, 1);
    Tensor target = make_random(1, audio_ref.size(), rng, 0.3f);
    Tensor audio;
    lora.zero_grad();
    lora.forward(bb, gen, ctx, vad, audio);
    // The open gate must actually move the audio — otherwise the grad check
    // below would be vacuously flat.
    double moved = 0.0;
    for (int i = 0; i < audio.size(); ++i)
        moved = std::fmax(moved, std::fabs(audio[i] - audio0[i]));
    if (moved < 1e-3) { std::printf("  FAIL decoder_lora: LoRA inert (max move %.3g)\n", moved); ++g_failures; }
    Tensor dAudio = Tensor::mat(1, audio.size());
    for (int i = 0; i < dAudio.size(); ++i) dAudio[i] = audio[i] - target[i];
    lora.backward(bb, gen, vad, dAudio);

    auto loss = [&]() { lora.forward(bb, gen, ctx, vad, audio); return mse_loss(audio, target)(); };
    auto ads = lora.adapters();
    fd_directional(ads.front()->A, ads.front()->dA, loss, rng, "decoder_lora.bb.A");
    fd_directional(ads.front()->B, ads.front()->dB, loss, rng, "decoder_lora.bb.B");
    fd_directional(ads.back()->A,  ads.back()->dA,  loss, rng, "decoder_lora.gen.A");
    fd_directional(ads.back()->B,  ads.back()->dB,  loss, rng, "decoder_lora.gen.B");
    fd_directional(lora.gate().W1, lora.gate().dW1, loss, rng, "decoder_lora.gate.W1");
    fd_directional(lora.gate().W2, lora.gate().dW2, loss, rng, "decoder_lora.gate.W2");

    // ── Checkpoint round-trip: load restores every parameter exactly. ──
    const std::string ckpt = "decoder_lora_test.ckpt";
    lora.save(ckpt);
    brosoundml::DecoderLora lora2 =
        brosoundml::DecoderLora::make(bb, gen, 4, 1.0f, Device::CPU, /*seed=*/999);
    lora2.load(ckpt);
    auto a2 = lora2.adapters();
    double pmax = std::fabs(lora2.scale() - lora.scale());
    for (int i = 0; i < ads.front()->A.size(); ++i)
        pmax = std::fmax(pmax, std::fabs(a2.front()->A[i] - ads.front()->A[i]));
    for (int i = 0; i < ads.back()->B.size(); ++i)
        pmax = std::fmax(pmax, std::fabs(a2.back()->B[i] - ads.back()->B[i]));
    for (int i = 0; i < W2.size(); ++i)
        pmax = std::fmax(pmax, std::fabs(lora2.gate().W2[i] - W2[i]));
    expect_near(pmax, 0.0, 1e-7, 0.0, "decoder_lora.checkpoint_roundtrip");
    std::remove(ckpt.c_str());
}

int main() {
    brotensor::init();
    test_adain_affine();
    test_adain_resblk(/*upsample=*/false, /*learned_sc=*/false, 8, 8, 16, 0xD001u, "resblk_plain");
    test_adain_resblk(/*upsample=*/true,  /*learned_sc=*/true,  8, 6, 12, 0xD002u, "resblk_up");
    test_adain_resblock1();
    test_spectral_head();
    test_decoder_backbone();
    test_generator();
    test_decoder_lora();

    if (g_failures) { std::printf("test_kokoro_decoder_backward: %d failure(s)\n", g_failures); return 1; }
    std::printf("test_kokoro_decoder_backward: OK\n");
    return 0;
}
