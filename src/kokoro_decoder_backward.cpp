// brosoundml/src/kokoro_decoder_backward.cpp — manual backward atoms through
// the Kokoro decoder back half (see kokoro_decoder_backward.h).

#include "kokoro_decoder_backward.h"

#include <brotensor/ops/activation.h>   // leaky_relu_forward / _backward
#include <brotensor/ops/concat.h>       // copy_d2d
#include <brotensor/ops/conv1d.h>       // conv1d / conv_transpose1d backward
#include <brotensor/ops/diffusion.h>    // modulate
#include <brotensor/ops/elementwise.h>  // mul_inplace, add_inplace, scale_inplace
#include <brotensor/ops/norm.h>         // group_norm_forward / _backward
#include <brotensor/ops/reduction.h>    // sum_cols
#include <brotensor/ops/spatial.h>      // nchw_to_sequence / sequence_to_nchw
#include <brotensor/ops/spectral.h>     // complex_from_polar / istft / istft_backward

#include <cmath>
#include <utility>
#include <vector>

namespace bt = brotensor;

namespace brosoundml {
namespace detail {

namespace {

// A (C,1) column of ones — the unit gamma that leaves GroupNorm's own affine
// as the identity (the style affine is applied by the modulate step).
bt::Tensor ones_col(bt::Device dev, int C) {
    std::vector<float> u(static_cast<std::size_t>(C), 1.0f);
    return bt::Tensor::from_host_on(dev, u.data(), C, 1);
}

// Backward of modulate (Y = X*(1+gamma) + beta), X:(M,C), gamma/beta:(C,1)
// broadcast across the M rows. Device-resident: dX reuses the forward op, the
// param grads are column reductions.
//   dX = dY*(1+gamma)                     = modulate(dY, gamma, 0)
//   dGamma[c] = sum_m dY[m,c] * X[m,c]
//   dBeta[c]  = sum_m dY[m,c]
// dGamma / dBeta are overwritten (shape (C,1)).
void modulate_backward(const bt::Tensor& X, const bt::Tensor& gamma,
                       const bt::Tensor& dY,
                       bt::Tensor& dX, bt::Tensor& dGamma, bt::Tensor& dBeta) {
    const bt::Device dev = X.device;
    const int C = gamma.rows;
    bt::Tensor zero = bt::Tensor::zeros_on(dev, C, 1);
    bt::modulate(dY, gamma, zero, dX);

    bt::Tensor t = dY.clone();
    bt::mul_inplace(t, X);
    bt::Tensor gsum;
    bt::sum_cols(t, gsum);   // (1,C)
    bt::Tensor bsum;
    bt::sum_cols(dY, bsum);  // (1,C)

    // (1,C) and (C,1) are the same C contiguous floats — copy straight across.
    dGamma = bt::Tensor::zeros_on(dev, C, 1);
    dBeta  = bt::Tensor::zeros_on(dev, C, 1);
    bt::copy_d2d(gsum, 0, dGamma, 0, C);
    bt::copy_d2d(bsum, 0, dBeta,  0, C);
}

// Nearest-neighbour 2x upsample along L of an NCL (1, C*L_in) tensor — the
// shortcut path's spatial match for the upsampling residual block. (brotensor
// has no 1D upsample; a host gather mirrors the forward's own host round-trip.)
//   y[c, 2l] = y[c, 2l+1] = x[c, l]
bt::Tensor nearest2x_forward(const bt::Tensor& x, int C, int L_in) {
    const int L_out = 2 * L_in;
    std::vector<float> xh = x.to_host_vector();
    std::vector<float> yh(static_cast<std::size_t>(C) * L_out);
    for (int c = 0; c < C; ++c) {
        const std::size_t bx = static_cast<std::size_t>(c) * L_in;
        const std::size_t by = static_cast<std::size_t>(c) * L_out;
        for (int l = 0; l < L_in; ++l) {
            const float v = xh[bx + l];
            yh[by + 2 * l + 0] = v;
            yh[by + 2 * l + 1] = v;
        }
    }
    return bt::Tensor::from_host_on(x.device, yh.data(), 1, C * L_out);
}

// Adjoint of nearest2x_forward: each input sample sums the two outputs that
// read it.  dx[c,l] = dy[c,2l] + dy[c,2l+1].
bt::Tensor nearest2x_backward(const bt::Tensor& dy, int C, int L_in) {
    const int L_out = 2 * L_in;
    std::vector<float> dh = dy.to_host_vector();
    std::vector<float> dx(static_cast<std::size_t>(C) * L_in, 0.0f);
    for (int c = 0; c < C; ++c) {
        const std::size_t bx = static_cast<std::size_t>(c) * L_in;
        const std::size_t by = static_cast<std::size_t>(c) * L_out;
        for (int l = 0; l < L_in; ++l)
            dx[bx + l] = dh[by + 2 * l + 0] + dh[by + 2 * l + 1];
    }
    return bt::Tensor::from_host_on(dy.device, dx.data(), 1, C * L_in);
}

}  // namespace

void adain_affine_forward(const bt::Tensor& x_ncl,
                          const bt::Tensor& gamma,
                          const bt::Tensor& beta,
                          int C, int L, float eps,
                          bt::Tensor& y_ncl,
                          bt::Tensor& x_seq_cache) {
    const bt::Device dev = x_ncl.device;
    bt::Tensor unit = ones_col(dev, C);
    bt::Tensor zero = bt::Tensor::zeros_on(dev, C, 1);

    // Instance norm = GroupNorm(num_groups = C), unit gamma / zero beta.
    bt::Tensor x_norm = bt::Tensor::empty_on(dev, 1, C * L, bt::Dtype::FP32);
    bt::group_norm_forward(x_ncl, unit, zero, /*N=*/1, C, /*H=*/1, /*W=*/L,
                           /*num_groups=*/C, eps, x_norm);

    // Per-channel style affine on the (L, C) sequence layout.
    bt::nchw_to_sequence(x_norm, /*N=*/1, C, /*H=*/1, /*W=*/L, x_seq_cache);
    bt::Tensor y_seq;
    bt::modulate(x_seq_cache, gamma, beta, y_seq);
    bt::sequence_to_nchw(y_seq, /*N=*/1, C, /*H=*/1, /*W=*/L, y_ncl);
}

void adain_affine_backward(const bt::Tensor& x_ncl,
                           const bt::Tensor& x_seq_cache,
                           const bt::Tensor& gamma,
                           const bt::Tensor& dY_ncl,
                           int C, int L, float eps,
                           bt::Tensor& dX_ncl,
                           bt::Tensor& dGamma,
                           bt::Tensor& dBeta) {
    const bt::Device dev = x_ncl.device;

    // adjoint of sequence_to_nchw is nchw_to_sequence
    bt::Tensor dY_seq;
    bt::nchw_to_sequence(dY_ncl, /*N=*/1, C, /*H=*/1, /*W=*/L, dY_seq);

    // modulate backward -> dX_seq, dGamma, dBeta
    bt::Tensor dX_seq;
    modulate_backward(x_seq_cache, gamma, dY_seq, dX_seq, dGamma, dBeta);

    // adjoint of nchw_to_sequence is sequence_to_nchw
    bt::Tensor dX_norm;
    bt::sequence_to_nchw(dX_seq, /*N=*/1, C, /*H=*/1, /*W=*/L, dX_norm);

    // GroupNorm backward (unit gamma); its own dGamma/dBeta are not trained.
    bt::Tensor unit = ones_col(dev, C);
    bt::Tensor gn_dgamma = bt::Tensor::zeros_on(dev, C, 1);
    bt::Tensor gn_dbeta  = bt::Tensor::zeros_on(dev, C, 1);
    bt::group_norm_backward(x_ncl, unit, dX_norm, /*N=*/1, C, /*H=*/1, /*W=*/L,
                            /*num_groups=*/C, eps, dX_ncl, gn_dgamma, gn_dbeta);
}

void adain_resblk_1d_forward_train(const AdainResBlk1dWeights& w,
                                   const bt::Tensor& x, int L_in,
                                   const bt::Tensor& gamma1,
                                   const bt::Tensor& beta1,
                                   const bt::Tensor& gamma2,
                                   const bt::Tensor& beta2,
                                   float eps,
                                   int& L_out, bt::Tensor& y,
                                   AdainResBlkCache& cache) {
    const int C_in  = w.channels_in;
    const int C_out = w.channels_out;
    L_out = w.upsample ? 2 * L_in : L_in;
    cache.x = x;
    cache.L_in = L_in;
    cache.L_out = L_out;

    // ─── residual ─────────────────────────────────────────────────────────
    bt::Tensor r;  // adain1 output, pre-leaky (cached for the backward)
    adain_affine_forward(x, gamma1, beta1, C_in, L_in, eps, cache.r_pre,
                         cache.x_seq1);
    bt::leaky_relu_forward(cache.r_pre, 0.2f, r);

    bt::Tensor pooled;
    if (w.upsample) {
        bt::conv_transpose1d_forward(r, w.pool_W, &w.pool_b,
                                     /*N=*/1, C_in, L_in, /*C_out=*/C_in,
                                     /*kL=*/3, /*stride=*/2, /*padding=*/1,
                                     /*output_padding=*/1, /*dilation=*/1,
                                     /*groups=*/C_in, pooled);
    } else {
        pooled = std::move(r);
    }

    w.conv1.forward(pooled, /*N=*/1, /*L=*/L_out, cache.conv1_out);

    bt::Tensor n2;  // adain2 output, pre-leaky cached
    adain_affine_forward(cache.conv1_out, gamma2, beta2, C_out, L_out, eps,
                         cache.n2_pre, cache.x_seq2);
    bt::leaky_relu_forward(cache.n2_pre, 0.2f, n2);

    bt::Tensor conv2_out;
    w.conv2.forward(n2, /*N=*/1, /*L=*/L_out, conv2_out);

    // ─── shortcut ─────────────────────────────────────────────────────────
    bt::Tensor short_x = w.upsample ? nearest2x_forward(x, C_in, L_in) : x;
    if (w.learned_sc) {
        bt::Tensor sc;
        w.conv1x1.forward(short_x, /*N=*/1, /*L=*/L_out, sc);
        short_x = std::move(sc);
    }

    // y = (residual + shortcut) / sqrt(2)
    bt::add_inplace(conv2_out, short_x);
    bt::scale_inplace(conv2_out, 1.0f / std::sqrt(2.0f));
    y = std::move(conv2_out);
}

void adain_resblk_1d_backward(const AdainResBlk1dWeights& w,
                              const AdainResBlkCache& cache,
                              const bt::Tensor& gamma1,
                              const bt::Tensor& gamma2,
                              const bt::Tensor& dY, float eps,
                              bt::Tensor& dX,
                              bt::Tensor& dGamma1, bt::Tensor& dBeta1,
                              bt::Tensor& dGamma2, bt::Tensor& dBeta2) {
    const int C_in  = w.channels_in;
    const int C_out = w.channels_out;
    const int L_in  = cache.L_in;
    const int L_out = cache.L_out;
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);

    // y = (conv2_out + short_x) / sqrt(2)
    bt::Tensor d_branch = dY.clone();
    bt::scale_inplace(d_branch, inv_sqrt2);   // grad into both conv2_out and short_x

    // ─── residual branch ──────────────────────────────────────────────────
    // conv2_out = conv2(n2)  (frozen weights → input-backward only)
    bt::Tensor d_n2;
    bt::conv1d_backward_input(w.conv2.W, d_branch, /*N=*/1, /*C_in=*/C_out,
                              /*L=*/L_out, /*C_out=*/C_out,
                              w.conv2.kernel_size, w.conv2.stride,
                              w.conv2.padding, w.conv2.dilation,
                              w.conv2.groups, d_n2);
    // n2 = leaky(n2_pre)
    bt::Tensor d_n2_pre;
    bt::leaky_relu_backward(cache.n2_pre, d_n2, 0.2f, d_n2_pre);
    // adain2: n2_pre = adain(conv1_out, gamma2, beta2)
    bt::Tensor d_conv1_out;
    adain_affine_backward(cache.conv1_out, cache.x_seq2, gamma2, d_n2_pre,
                          C_out, L_out, eps, d_conv1_out, dGamma2, dBeta2);
    // conv1_out = conv1(pooled)
    bt::Tensor d_pooled;
    bt::conv1d_backward_input(w.conv1.W, d_conv1_out, /*N=*/1, /*C_in=*/C_in,
                              /*L=*/L_out, /*C_out=*/C_out,
                              w.conv1.kernel_size, w.conv1.stride,
                              w.conv1.padding, w.conv1.dilation,
                              w.conv1.groups, d_pooled);
    // pool: identity, or depthwise ConvTranspose1d (upsample)
    bt::Tensor d_r;
    if (w.upsample) {
        bt::conv_transpose1d_backward_input(w.pool_W, d_pooled, /*N=*/1, C_in,
                                            L_in, /*C_out=*/C_in, /*kL=*/3,
                                            /*stride=*/2, /*padding=*/1,
                                            /*output_padding=*/1, /*dilation=*/1,
                                            /*groups=*/C_in, d_r);
    } else {
        d_r = std::move(d_pooled);
    }
    // r = leaky(r_pre)
    bt::Tensor d_r_pre;
    bt::leaky_relu_backward(cache.r_pre, d_r, 0.2f, d_r_pre);
    // adain1: r_pre = adain(x, gamma1, beta1)  -> dX (residual contribution)
    bt::Tensor dX_res;
    adain_affine_backward(cache.x, cache.x_seq1, gamma1, d_r_pre,
                          C_in, L_in, eps, dX_res, dGamma1, dBeta1);

    // ─── shortcut branch ──────────────────────────────────────────────────
    bt::Tensor d_short_pre;  // grad into (upsample ? nearest2x(x) : x)
    if (w.learned_sc) {
        bt::conv1d_backward_input(w.conv1x1.W, d_branch, /*N=*/1, /*C_in=*/C_in,
                                  /*L=*/L_out, /*C_out=*/C_out,
                                  w.conv1x1.kernel_size, w.conv1x1.stride,
                                  w.conv1x1.padding, w.conv1x1.dilation,
                                  w.conv1x1.groups, d_short_pre);
    } else {
        d_short_pre = d_branch;  // shortcut is the identity (C_in == C_out)
    }
    bt::Tensor dX_short = w.upsample ? nearest2x_backward(d_short_pre, C_in, L_in)
                                     : d_short_pre;

    // dX = residual + shortcut contributions
    bt::add_inplace(dX_res, dX_short);
    dX = std::move(dX_res);
}

void adain_resblock1_forward_train(const AdaINResBlock1Weights& w,
                                   const bt::Tensor& x_in, int C, int L,
                                   const std::array<bt::Tensor, 3>& g1,
                                   const std::array<bt::Tensor, 3>& b1,
                                   const std::array<bt::Tensor, 3>& g2,
                                   const std::array<bt::Tensor, 3>& b2,
                                   bt::Tensor& x_out,
                                   AdainResBlock1Cache& cache) {
    cache.C = C;
    cache.L = L;
    bt::Tensor x = x_in;
    for (int i = 0; i < 3; ++i) {
        cache.x_in[i] = x;

        // snake(adain1(x, s), alpha1) -> convs1 (dilated)
        adain_affine_forward(x, g1[i], b1[i], C, L, w.adain1[i].eps,
                             cache.s1_pre[i], cache.x_seq1[i]);
        bt::Tensor xt1;
        bt::snake_forward(cache.s1_pre[i], w.alpha1[i], /*beta=*/nullptr,
                          1, C, L, xt1);
        w.convs1[i].forward(xt1, /*N=*/1, /*L=*/L, cache.c1_out[i]);

        // snake(adain2(c1, s), alpha2) -> convs2 (dilation 1)
        adain_affine_forward(cache.c1_out[i], g2[i], b2[i], C, L, w.adain2[i].eps,
                             cache.s2_pre[i], cache.x_seq2[i]);
        bt::Tensor xt2;
        bt::snake_forward(cache.s2_pre[i], w.alpha2[i], /*beta=*/nullptr,
                          1, C, L, xt2);
        bt::Tensor c2_out;
        w.convs2[i].forward(xt2, /*N=*/1, /*L=*/L, c2_out);

        // x = xt + x  (residual)
        bt::add_inplace(c2_out, x);
        x = std::move(c2_out);
    }
    x_out = std::move(x);
}

void adain_resblock1_backward(const AdaINResBlock1Weights& w,
                              const AdainResBlock1Cache& cache,
                              const std::array<bt::Tensor, 3>& g1,
                              const std::array<bt::Tensor, 3>& g2,
                              const bt::Tensor& dX_out,
                              bt::Tensor& dX_in,
                              std::array<bt::Tensor, 3>& dG1,
                              std::array<bt::Tensor, 3>& dB1,
                              std::array<bt::Tensor, 3>& dG2,
                              std::array<bt::Tensor, 3>& dB2) {
    const int C = cache.C;
    const int L = cache.L;
    const bt::Device dev = dX_out.device;

    bt::Tensor dx = dX_out.clone();
    for (int i = 2; i >= 0; --i) {
        // x_{i+1} = c2_out + x_in[i] ; the skip carries dx straight back to x_in.
        bt::Tensor d_c2 = dx;          // grad into c2_out
        bt::Tensor d_skip = dx;        // grad into x_in[i] via the residual add

        // convs2[i] (dilation 1, frozen) -> snake2 -> adain2
        bt::Tensor d_xt2;
        bt::conv1d_backward_input(w.convs2[i].W, d_c2, /*N=*/1, /*C_in=*/C, /*L=*/L,
                                  /*C_out=*/C, w.convs2[i].kernel_size,
                                  w.convs2[i].stride, w.convs2[i].padding,
                                  w.convs2[i].dilation, w.convs2[i].groups, d_xt2);
        bt::Tensor d_s2_pre, dAlpha2 = bt::Tensor::zeros_on(dev, C, 1);
        bt::snake_backward(cache.s2_pre[i], w.alpha2[i], /*beta=*/nullptr, d_xt2,
                           1, C, L, d_s2_pre, dAlpha2, /*dBeta=*/nullptr);
        bt::Tensor d_c1_out;
        adain_affine_backward(cache.c1_out[i], cache.x_seq2[i], g2[i], d_s2_pre,
                              C, L, w.adain2[i].eps, d_c1_out, dG2[i], dB2[i]);

        // convs1[i] (dilated, frozen) -> snake1 -> adain1
        bt::Tensor d_xt1;
        bt::conv1d_backward_input(w.convs1[i].W, d_c1_out, /*N=*/1, /*C_in=*/C, /*L=*/L,
                                  /*C_out=*/C, w.convs1[i].kernel_size,
                                  w.convs1[i].stride, w.convs1[i].padding,
                                  w.convs1[i].dilation, w.convs1[i].groups, d_xt1);
        bt::Tensor d_s1_pre, dAlpha1 = bt::Tensor::zeros_on(dev, C, 1);
        bt::snake_backward(cache.s1_pre[i], w.alpha1[i], /*beta=*/nullptr, d_xt1,
                           1, C, L, d_s1_pre, dAlpha1, /*dBeta=*/nullptr);
        bt::Tensor d_x_adain1;
        adain_affine_backward(cache.x_in[i], cache.x_seq1[i], g1[i], d_s1_pre,
                              C, L, w.adain1[i].eps, d_x_adain1, dG1[i], dB1[i]);

        // grad into x_in[i] = residual-skip + adain1 path
        bt::add_inplace(d_x_adain1, d_skip);
        dx = std::move(d_x_adain1);
    }
    dX_in = std::move(dx);
}

void spectral_head_forward_train(const Conv1d& conv_post,
                                 const bt::Tensor& x, int C, int L,
                                 const bt::Tensor& window,
                                 int n_fft, int hop, int win,
                                 bt::Tensor& audio,
                                 SpectralHeadCache& cache) {
    const bt::Device dev = x.device;
    cache.C = C; cache.L = L; cache.x_pre = x;

    bt::Tensor xl;
    bt::leaky_relu_forward(x, 0.01f, xl);
    conv_post.forward(xl, /*N=*/1, /*L=*/L, cache.post);

    // host: post -> (mag = exp(log-mag), ang = sin(phase)), frame-major (L, n_freq)
    const int n_freq = n_fft / 2 + 1;
    const std::vector<float> ph = cache.post.to_host_vector();
    std::vector<float> mag_host(static_cast<std::size_t>(L) * n_freq);
    std::vector<float> pha_host(static_cast<std::size_t>(L) * n_freq);
    for (int c = 0; c < n_freq; ++c) {
        for (int l = 0; l < L; ++l) {
            mag_host[l * n_freq + c] = std::exp(ph[c * L + l]);
            pha_host[l * n_freq + c] = std::sin(ph[(n_freq + c) * L + l]);
        }
    }
    bt::Tensor mag = bt::Tensor::from_host_on(dev, mag_host.data(), L, n_freq);
    bt::Tensor pha = bt::Tensor::from_host_on(dev, pha_host.data(), L, n_freq);

    bt::Tensor spec;
    bt::complex_from_polar(mag, pha, spec);
    const int signal_len = (L - 1) * hop;
    bt::istft(spec, window, /*N=*/1, signal_len, n_fft, hop, win,
              /*center=*/true, /*normalized=*/false, audio);
}

void spectral_head_backward(const Conv1d& conv_post,
                            const SpectralHeadCache& cache,
                            const bt::Tensor& window,
                            int n_fft, int hop, int win,
                            const bt::Tensor& dAudio,
                            bt::Tensor& dX) {
    const bt::Device dev = cache.x_pre.device;
    const int C = cache.C, L = cache.L;
    const int n_freq = n_fft / 2 + 1;
    const int signal_len = (L - 1) * hop;

    // iSTFT backward -> dSpec, interleaved-complex (L, 2*n_freq)
    bt::Tensor dSpec;
    bt::istft_backward(dAudio, window, /*N=*/1, signal_len, n_fft, hop, win,
                       /*center=*/true, /*normalized=*/false, dSpec);

    // host: complex_from_polar + exp/sin adjoint -> dPost (1, (n_fft+2)*L)
    const std::vector<float> ph = cache.post.to_host_vector();  // channel-major
    const std::vector<float> ds = dSpec.to_host_vector();       // (L, 2*n_freq)
    const int C_post = n_fft + 2;                               // = 2*n_freq
    std::vector<float> dpost(static_cast<std::size_t>(C_post) * L, 0.0f);
    for (int c = 0; c < n_freq; ++c) {
        for (int l = 0; l < L; ++l) {
            const float pm = ph[c * L + l];               // log-magnitude
            const float pp = ph[(n_freq + c) * L + l];    // pre-sin phase
            const float mag = std::exp(pm);
            const float ang = std::sin(pp);
            const float ca = std::cos(ang), sa = std::sin(ang);
            const float dRe = ds[l * (2 * n_freq) + 2 * c];
            const float dIm = ds[l * (2 * n_freq) + 2 * c + 1];
            // re = mag*cos(ang), im = mag*sin(ang)
            const float dmag = dRe * ca + dIm * sa;
            const float dang = mag * (-dRe * sa + dIm * ca);
            dpost[c * L + l]              = dmag * mag;            // exp adjoint
            dpost[(n_freq + c) * L + l]   = dang * std::cos(pp);  // sin adjoint
        }
    }
    bt::Tensor dPost = bt::Tensor::from_host_on(dev, dpost.data(), 1, C_post * L);

    // conv_post backward (frozen weights -> input grad only) -> leaky backward
    bt::Tensor dXl;
    bt::conv1d_backward_input(conv_post.W, dPost, /*N=*/1, /*C_in=*/C, /*L=*/L,
                              /*C_out=*/C_post, conv_post.kernel_size,
                              conv_post.stride, conv_post.padding,
                              conv_post.dilation, conv_post.groups, dXl);
    bt::leaky_relu_backward(cache.x_pre, dXl, 0.01f, dX);
}

// ─── DecoderBackbone assembly ───────────────────────────────────────────────

namespace {

// Channel-major concat along the C axis of NCL (1, C*L) tensors at a common L.
bt::Tensor cat_channels(const std::vector<const bt::Tensor*>& parts, int L,
                        bt::Device dev) {
    int C_total = 0;
    for (const auto* p : parts) C_total += p->cols / L;
    bt::Tensor out = bt::Tensor::zeros_on(dev, 1, C_total * L, bt::Dtype::FP32);
    int c_off = 0;
    for (const auto* p : parts) {
        const int C = p->cols / L;
        bt::copy_d2d(*p, 0, out, static_cast<std::size_t>(c_off) * L,
                     static_cast<std::size_t>(C) * L);
        c_off += C;
    }
    return out;
}

}  // namespace

void decoder_backbone_forward_train(const DecoderBackbone& bb,
                                    const bt::Tensor& asr,
                                    const bt::Tensor& F0_pred,
                                    const bt::Tensor& N_pred,
                                    int T,
                                    const std::vector<BlockAffines>& aff,
                                    float eps,
                                    bt::Tensor& gen_in,
                                    BackboneCache& cache) {
    const bt::Device dev = asr.device;
    const int n_dec = static_cast<int>(bb.decode.size());
    cache.T = T;
    cache.decode.resize(n_dec);
    cache.res_applied.assign(n_dec, 0);
    cache.x_channels.assign(n_dec, 0);

    // Frozen front convs: F0/N downsample (stride 2) and the asr 1x1 projection.
    bt::Tensor F0_dn = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor N_dn  = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bt::Tensor asr_res_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    bb.F0_conv.forward(F0_pred, /*N=*/1, /*L=*/2 * T, F0_dn);
    bb.N_conv .forward(N_pred,  /*N=*/1, /*L=*/2 * T, N_dn);
    bb.asr_res.forward(asr,     /*N=*/1, /*L=*/T,     asr_res_out);

    // encode: AdainResBlk1d over cat[asr, F0_dn, N_dn] (a frozen input).
    bt::Tensor dec_pre = cat_channels({&asr, &F0_dn, &N_dn}, T, dev);
    int L_after = 0;
    bt::Tensor enc_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    adain_resblk_1d_forward_train(bb.encode, dec_pre, T, aff[0].g1, aff[0].b1,
                                  aff[0].g2, aff[0].b2, eps, L_after, enc_out,
                                  cache.encode);

    bt::Tensor x = std::move(enc_out);
    int L_now = L_after;
    bool res = true;
    for (int i = 0; i < n_dec; ++i) {
        cache.x_channels[i] = x.cols / L_now;
        bt::Tensor blk_in;
        if (res) {
            cache.res_applied[i] = 1;
            blk_in = cat_channels({&x, &asr_res_out, &F0_dn, &N_dn}, L_now, dev);
        } else {
            blk_in = x;
        }
        int L_next = 0;
        bt::Tensor y = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        adain_resblk_1d_forward_train(bb.decode[i], blk_in, L_now,
                                      aff[1 + i].g1, aff[1 + i].b1,
                                      aff[1 + i].g2, aff[1 + i].b2, eps,
                                      L_next, y, cache.decode[i]);
        x = std::move(y);
        L_now = L_next;
        if (bb.decode[i].upsample) res = false;
    }
    gen_in = std::move(x);
}

void decoder_backbone_backward(const DecoderBackbone& bb,
                               const BackboneCache& cache,
                               const std::vector<BlockAffines>& aff,
                               const bt::Tensor& dGenIn,
                               float eps,
                               std::vector<BlockAffines>& dAff) {
    const int n_dec = static_cast<int>(bb.decode.size());
    dAff.resize(static_cast<std::size_t>(n_dec) + 1);

    bt::Tensor dX = dGenIn.clone();
    for (int i = n_dec - 1; i >= 0; --i) {
        bt::Tensor blk_dX;
        BlockAffines& g = dAff[1 + i];
        adain_resblk_1d_backward(bb.decode[i], cache.decode[i], aff[1 + i].g1,
                                 aff[1 + i].g2, dX, eps, blk_dX,
                                 g.g1, g.b1, g.g2, g.b2);
        if (cache.res_applied[i]) {
            // Slice the leading x channels out of [x, asr_res, F0_dn, N_dn]; the
            // trailing channels are grads w.r.t. frozen inputs and are dropped.
            const int Cx = cache.x_channels[i];
            const int L_in = cache.decode[i].L_in;
            bt::Tensor sliced =
                bt::Tensor::zeros_on(blk_dX.device, 1, Cx * L_in, bt::Dtype::FP32);
            bt::copy_d2d(blk_dX, 0, sliced, 0, static_cast<std::size_t>(Cx) * L_in);
            dX = std::move(sliced);
        } else {
            dX = std::move(blk_dX);
        }
    }

    // encode: its input is a frozen cat, so the returned dX is discarded.
    bt::Tensor enc_dX;
    BlockAffines& ge = dAff[0];
    adain_resblk_1d_backward(bb.encode, cache.encode, aff[0].g1, aff[0].g2, dX,
                             eps, enc_dX, ge.g1, ge.b1, ge.g2, ge.b2);
}

// ─── Generator assembly ─────────────────────────────────────────────────────

namespace {

// Periodic Hann window (matches kokoro_modules.cpp::hann_window_periodic).
bt::Tensor hann_periodic(int N, bt::Device dev) {
    std::vector<float> w(static_cast<std::size_t>(N));
    constexpr float kTwoPi = 6.28318530717958647692f;
    for (int n = 0; n < N; ++n)
        w[n] = 0.5f - 0.5f * std::cos(kTwoPi * static_cast<float>(n) /
                                              static_cast<float>(N));
    return bt::Tensor::from_host_on(dev, w.data(), 1, N);
}

}  // namespace

void generator_forward_train(const Generator& gen,
                             const bt::Tensor& gen_in, int L_in,
                             const bt::Tensor& har, int frames,
                             const std::vector<ResBlock1Affines>& naff,
                             const std::vector<ResBlock1Affines>& raff,
                             bt::Tensor& audio,
                             GeneratorCache& cache) {
    const bt::Device dev = gen_in.device;
    const int K = gen.num_kernels;
    cache.stages.resize(gen.num_upsamples);

    bt::Tensor x = gen_in;
    int L = L_in;
    for (int i = 0; i < gen.num_upsamples; ++i) {
        GenStageCache& sc = cache.stages[i];
        sc.x_in = x;                       // pre-leaky input (for the leaky backward)
        sc.C_in = gen.ups_C_in[i];
        sc.L_in = L;

        // a = leaky(x, 0.1)
        bt::Tensor a;
        bt::leaky_relu_forward(x, 0.1f, a);

        // noise: src = noise_res[i](noise_convs[i](har), s)
        bt::Tensor x_source;
        gen.noise_convs[i].forward(har, /*N=*/1, /*L=*/frames, x_source);
        const int L_src = x_source.cols / gen.ups_C_out[i];
        bt::Tensor src;
        adain_resblock1_forward_train(gen.noise_res[i], x_source, gen.ups_C_out[i],
                                      L_src, naff[i].g1, naff[i].b1, naff[i].g2,
                                      naff[i].b2, src, sc.noise);

        // u = convT(ups[i], a)  [+ left reflect-pad 1 on the final stage]
        const int kL = gen.ups_k[i], s = gen.ups_stride[i], p = gen.ups_pad[i];
        bt::Tensor u;
        bt::conv_transpose1d_forward(a, gen.ups_W[i], &gen.ups_b[i], /*N=*/1,
                                     gen.ups_C_in[i], L, gen.ups_C_out[i], kL, s,
                                     p, /*output_padding=*/0, /*dilation=*/1, u);
        int L_x = (L - 1) * s - 2 * p + (kL - 1) + 1;
        if (i == gen.num_upsamples - 1) {
            bt::Tensor padded;
            bt::pad1d_forward(u, /*N=*/1, gen.ups_C_out[i], L_x,
                              /*pad_left=*/1, /*pad_right=*/0, /*mode=*/1, padded);
            u = std::move(padded);
            L_x += 1;
        }
        sc.C_out = gen.ups_C_out[i];
        sc.L_out = L_x;

        // m = u + src
        bt::add_inplace(u, src);

        // x = mean_j resblocks[i*K + j](m, s)
        sc.res.resize(K);
        bt::Tensor xs;
        for (int j = 0; j < K; ++j) {
            const ResBlock1Affines& ra = raff[static_cast<std::size_t>(i) * K + j];
            bt::Tensor xj;
            adain_resblock1_forward_train(gen.resblocks[i * K + j], u,
                                          gen.ups_C_out[i], L_x, ra.g1, ra.b1,
                                          ra.g2, ra.b2, xj, sc.res[j]);
            if (j == 0) xs = std::move(xj);
            else        bt::add_inplace(xs, xj);
        }
        bt::scale_inplace(xs, 1.0f / static_cast<float>(K));
        x = std::move(xs);
        L = L_x;
    }

    // Trailing spectral head (leaky 0.01 → conv_post → polar exp/sin → iSTFT).
    cache.window = hann_periodic(gen.win_size, dev);
    spectral_head_forward_train(gen.conv_post, x,
                                gen.ups_C_out[gen.num_upsamples - 1], L,
                                cache.window, gen.n_fft, gen.hop_size,
                                gen.win_size, audio, cache.head);
}

void generator_backward(const Generator& gen,
                        const GeneratorCache& cache,
                        const std::vector<ResBlock1Affines>& naff,
                        const std::vector<ResBlock1Affines>& raff,
                        const bt::Tensor& dAudio,
                        bt::Tensor& dGenIn,
                        std::vector<ResBlock1Affines>& dnaff,
                        std::vector<ResBlock1Affines>& draff) {
    const int K = gen.num_kernels;
    const int U = gen.num_upsamples;
    dnaff.resize(U);
    draff.resize(static_cast<std::size_t>(U) * K);

    // Spectral head backward → dX into the final stack output.
    bt::Tensor dX;
    spectral_head_backward(gen.conv_post, cache.head, cache.window, gen.n_fft,
                           gen.hop_size, gen.win_size, dAudio, dX);

    for (int i = U - 1; i >= 0; --i) {
        const GenStageCache& sc = cache.stages[i];
        const int L = sc.L_out;

        // x = mean_j rb_j(m): dL/dm = sum_j rb_j_backward((1/K)·dX).
        bt::Tensor d_scaled = dX.clone();
        bt::scale_inplace(d_scaled, 1.0f / static_cast<float>(K));
        bt::Tensor d_m;
        for (int j = 0; j < K; ++j) {
            ResBlock1Affines& dr = draff[static_cast<std::size_t>(i) * K + j];
            bt::Tensor dxin_j;
            adain_resblock1_backward(gen.resblocks[i * K + j], sc.res[j],
                                     raff[static_cast<std::size_t>(i) * K + j].g1,
                                     raff[static_cast<std::size_t>(i) * K + j].g2,
                                     d_scaled, dxin_j, dr.g1, dr.b1, dr.g2, dr.b2);
            if (j == 0) d_m = std::move(dxin_j);
            else        bt::add_inplace(d_m, dxin_j);
        }

        // m = u + src.  Noise branch: noise_res[i] backward; the dX into the
        // frozen noise_convs (har is constant) is dropped, only its affines kept.
        {
            ResBlock1Affines& dn = dnaff[i];
            bt::Tensor dsrc_in;  // discarded
            adain_resblock1_backward(gen.noise_res[i], sc.noise, naff[i].g1,
                                     naff[i].g2, d_m, dsrc_in, dn.g1, dn.b1,
                                     dn.g2, dn.b2);
        }

        // ups branch: u = (pad?) convT(ups[i], a).  du = d_m.
        bt::Tensor du = d_m;
        if (i == U - 1) {                  // strip the final left reflect-pad
            bt::Tensor du_pre;
            bt::pad1d_backward(du, /*N=*/1, gen.ups_C_out[i], /*L=*/L - 1,
                               /*pad_left=*/1, /*pad_right=*/0, /*mode=*/1, du_pre);
            du = std::move(du_pre);
        }
        bt::Tensor da;
        bt::conv_transpose1d_backward_input(gen.ups_W[i], du, /*N=*/1, sc.C_in,
                                            sc.L_in, gen.ups_C_out[i], gen.ups_k[i],
                                            gen.ups_stride[i], gen.ups_pad[i],
                                            /*output_padding=*/0, /*dilation=*/1, da);
        // a = leaky(x_in, 0.1) → grad into the stage input (= previous stage out).
        bt::Tensor dx_in;
        bt::leaky_relu_backward(sc.x_in, da, 0.1f, dx_in);
        dX = std::move(dx_in);
    }
    // dX is now the grad into gen_in — the backbone output. Hand it back so the
    // loss threads on into the backbone's affines (the assemblies run in series).
    dGenIn = std::move(dX);
}

}  // namespace detail
}  // namespace brosoundml
