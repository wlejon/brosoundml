#include "supertonic_encoder.h"

#include "supertonic_backward.h"  // st_detail forward_train atoms (+ caches)

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cmath>
#include <random>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;
namespace st = st_detail;

namespace {

// Xavier-uniform host init for a conv weight (cout, cin_pg*k) in OIL layout,
// with fan_in = cin_pg*k, fan_out = cout. Bias (if any) zero-initialised.
st::ConvW init_convw(int cout, int cin_pg, int k, bool has_b,
                     bt::Device dev, std::mt19937& rng) {
    st::ConvW c;
    c.cout = cout; c.cin_pg = cin_pg; c.k = k; c.has_b = has_b;
    const int fan_in  = cin_pg * k;
    const int fan_out = cout;
    const float lim = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> u(-lim, lim);
    std::vector<float> w(static_cast<std::size_t>(cout) * fan_in);
    for (auto& x : w) x = u(rng);
    c.w = bt::Tensor::from_host_on(dev, w.data(), cout, fan_in);
    if (has_b) {
        std::vector<float> b(static_cast<std::size_t>(cout), 0.0f);
        c.b = bt::Tensor::from_host_on(dev, b.data(), cout, 1);
    }
    return c;
}

}  // namespace

void SupertonicEncoder::init(bt::Device device, std::uint64_t seed) {
    bt::init();
    dev = device;
    std::mt19937 rng(static_cast<std::uint32_t>(seed));

    conv_in = init_convw(hidden, idim, ksz_init, /*has_b=*/true, dev, rng);

    blocks.clear();
    blocks.reserve(static_cast<std::size_t>(num_layers));
    for (int i = 0; i < num_layers; ++i) {
        st::ConvNeXtBlock blk;
        blk.dil = 1;
        blk.dw  = init_convw(hidden, /*cin_pg=*/1, ksz, /*has_b=*/true, dev, rng);
        blk.pw1 = init_convw(intermediate, hidden, 1, /*has_b=*/true, dev, rng);
        blk.pw2 = init_convw(hidden, intermediate, 1, /*has_b=*/true, dev, rng);
        std::vector<float> g(static_cast<std::size_t>(hidden), 1.0f);
        std::vector<float> b(static_cast<std::size_t>(hidden), 0.0f);
        blk.ln_g = bt::Tensor::from_host_on(dev, g.data(), hidden, 1);
        blk.ln_b = bt::Tensor::from_host_on(dev, b.data(), hidden, 1);
        blocks.push_back(std::move(blk));
    }

    proj_out = init_convw(latent_dim, hidden, 1, /*has_b=*/true, dev, rng);
}

bt::Tensor SupertonicEncoder::forward(const bt::Tensor& spec, int T) const {
    // pconv expects its input tagged (1, Cin*L) channel-major; SupertonicSpec
    // hands back (idim, T) (same bytes). Re-tag via a non-owning flat view.
    bt::Tensor spec_flat =
        bt::Tensor::view(spec.device, spec.data, 1, idim * T, spec.dtype);

    // conv_in: symmetric edge-pad keeps length (pad = dilation*(k-1)/2 each side).
    const int pad_in = (ksz_init - 1) / 2;
    bt::Tensor h;
    st::PConvCache cin_cache;
    st::pconv_forward_train(spec_flat, conv_in, idim, T, /*dilation=*/1, /*groups=*/1,
                            pad_in, pad_in, /*mode=*/2 /*edge*/, h, cin_cache);

    for (const st::ConvNeXtBlock& blk : blocks) {
        bt::Tensor y;
        st::ConvNeXtCache cache;
        st::convnext_block_forward_train(h, blk, hidden, T, blk.dil, y, cache);
        h = std::move(y);
    }

    bt::Tensor latent;
    st::PConvCache pout_cache;
    st::pconv_forward_train(h, proj_out, hidden, T, /*dilation=*/1, /*groups=*/1,
                            /*pad_left=*/0, /*pad_right=*/0, /*mode=*/0,
                            latent, pout_cache);
    // pconv's matmul path re-tags its output (1, cout*L); present it as the
    // (latent_dim, T) channel-major grid callers expect (same bytes).
    return bt::Tensor::view(latent.device, latent.data, latent_dim, T, latent.dtype)
        .clone();
}

namespace {
// Flat (1, rows*cols) non-owning view.
bt::Tensor flat1(const bt::Tensor& t) {
    return bt::Tensor::view(t.device, t.data, 1, t.rows * t.cols, t.dtype);
}
// Re-tag a tensor's (rows, cols) in place (same bytes).
bt::Tensor retag(const bt::Tensor& t, int r, int c) {
    return bt::Tensor::view(t.device, t.data, r, c, t.dtype);
}
}  // namespace

void SupertonicEncoder::forward_train(const bt::Tensor& spec, int T,
                                      bt::Tensor& latent,
                                      SupertonicEncoderCache& cache) const {
    cache.T = T;
    cache.spec_in = retag(spec, idim, T).clone();   // conv_in unpadded input
    bt::Tensor spec_flat = bt::Tensor::view(spec.device, spec.data, 1, idim * T,
                                            spec.dtype);

    const int pad_in = (ksz_init - 1) / 2;
    bt::Tensor h;
    st::pconv_forward_train(spec_flat, conv_in, idim, T, /*dilation=*/1, /*groups=*/1,
                            pad_in, pad_in, /*mode=*/2, h, cache.conv_in);

    cache.blocks.clear();
    cache.blocks.resize(static_cast<std::size_t>(num_layers));
    for (int i = 0; i < num_layers; ++i) {
        SupertonicEncoderCache::Block& bc = cache.blocks[static_cast<std::size_t>(i)];
        bc.h_in = retag(h, hidden, T).clone();       // dw unpadded input
        bt::Tensor y;
        st::convnext_block_forward_train(h, blocks[static_cast<std::size_t>(i)],
                                         hidden, T, blocks[static_cast<std::size_t>(i)].dil,
                                         y, bc.cn);
        h = std::move(y);
    }

    bt::Tensor lat;
    st::pconv_forward_train(h, proj_out, hidden, T, /*dilation=*/1, /*groups=*/1,
                            0, 0, 0, lat, cache.proj_out);
    latent = retag(lat, latent_dim, T).clone();
}

void SupertonicEncoderGrads::zero(const SupertonicEncoder& enc) {
    const bt::Device d = enc.dev;
    auto zero_convw = [&](const st::ConvW& w) {
        st::ConvW g;
        g.cout = w.cout; g.cin_pg = w.cin_pg; g.k = w.k; g.has_b = w.has_b;
        g.w = bt::Tensor::zeros_on(d, w.w.rows, w.w.cols);
        if (w.has_b) g.b = bt::Tensor::zeros_on(d, w.cout, 1);
        return g;
    };
    conv_in  = zero_convw(enc.conv_in);
    proj_out = zero_convw(enc.proj_out);
    blocks.clear();
    blocks.resize(enc.blocks.size());
    for (std::size_t i = 0; i < enc.blocks.size(); ++i) {
        const st::ConvNeXtBlock& b = enc.blocks[i];
        st::ConvNeXtBlock& g = blocks[i];
        g.dil = b.dil;
        g.dw  = zero_convw(b.dw);
        g.pw1 = zero_convw(b.pw1);
        g.pw2 = zero_convw(b.pw2);
        g.ln_g = bt::Tensor::zeros_on(d, enc.hidden, 1);
        g.ln_b = bt::Tensor::zeros_on(d, enc.hidden, 1);
    }
}

void SupertonicEncoder::backward(const SupertonicEncoderCache& cache,
                                 const bt::Tensor& dLatent,
                                 SupertonicEncoderGrads& grads) const {
    const bt::Device d = dev;
    const int T = cache.T;
    const int C = hidden;

    // ── proj_out (1×1 matmul): dW += dC @ x_inᵀ, dInput = wᵀ @ dC ──────────────
    bt::Tensor dC = retag(dLatent, latent_dim, T);
    bt::Tensor dH = bt::Tensor::zeros_on(d, C, T);
    bt::matmul_backward(proj_out.w, cache.proj_out.x_in, dC, grads.proj_out.w, dH);
    if (proj_out.has_b)
        bt::conv1d_backward_bias(retag(dLatent, 1, latent_dim * T), 1, latent_dim, T,
                                 grads.proj_out.b);
    dH = retag(dH, 1, C * T).clone();

    // ── ConvNeXt blocks, reverse ───────────────────────────────────────────────
    for (int i = num_layers - 1; i >= 0; --i) {
        const SupertonicEncoderCache::Block& bc = cache.blocks[static_cast<std::size_t>(i)];
        const st::ConvNeXtBlock& blk = blocks[static_cast<std::size_t>(i)];
        st::ConvNeXtBlock& g = grads.blocks[static_cast<std::size_t>(i)];
        const int inter = blk.pw1.cout;

        // pw2: yc = pw2.w @ ga.  dW += dY @ gaᵀ ; dGa = pw2.wᵀ @ dY.
        bt::Tensor dYc = retag(dH, C, T);
        bt::Tensor dGa = bt::Tensor::zeros_on(d, inter, T);
        bt::matmul_backward(blk.pw2.w, bc.cn.pw2.x_in, dYc, g.pw2.w, dGa);
        if (blk.pw2.has_b)
            bt::conv1d_backward_bias(retag(dH, 1, C * T), 1, C, T, g.pw2.b);

        // gelu: dA = gelu'(a) * dGa.
        bt::Tensor dA;
        bt::gelu_exact_backward(bc.cn.a, retag(dGa, 1, inter * T), dA);

        // pw1: a = pw1.w @ normed.  dW += dA @ normedᵀ ; dNormed = pw1.wᵀ @ dA.
        bt::Tensor dCa = retag(dA, inter, T);
        bt::Tensor dNormed = bt::Tensor::zeros_on(d, C, T);
        bt::matmul_backward(blk.pw1.w, bc.cn.pw1.x_in, dCa, g.pw1.w, dNormed);
        if (blk.pw1.has_b)
            bt::conv1d_backward_bias(retag(dA, 1, inter * T), 1, inter, T, g.pw1.b);

        // LayerNorm over C (transpose [C,T]->[T,C], LN backward, transpose back).
        bt::Tensor dSeqn;
        bt::nchw_to_sequence(retag(dNormed, 1, C * T), 1, C, 1, T, dSeqn);  // [T,C]
        bt::Tensor dSeq;
        bt::layernorm_backward_batched_with_caches(dSeqn, bc.cn.ln_xhat, blk.ln_g,
                                                   bc.cn.ln_rstd, dSeq, g.ln_g, g.ln_b);
        bt::Tensor dDwy;
        bt::sequence_to_nchw(dSeq, 1, C, 1, T, dDwy);                       // [C,T]
        dDwy = retag(dDwy, 1, C * T).clone();

        // depthwise dw (conv path): weight grad needs the re-padded input.
        const int dil = blk.dil, pad = 2 * dil, Lp = T + 2 * pad;
        bt::Tensor hpad;
        bt::pad1d_forward(flat1(bc.h_in), 1, C, T, pad, pad, /*mode=*/2, hpad);
        bt::conv1d_backward_weight(hpad, dDwy, 1, C, Lp, C, blk.dw.k, 1, 0, dil, C, g.dw.w);
        if (blk.dw.has_b)
            bt::conv1d_backward_bias(dDwy, 1, C, T, g.dw.b);
        // input grad through dw -> block input.
        bt::Tensor dHpad;
        bt::conv1d_backward_input(blk.dw.w, dDwy, 1, C, Lp, C, blk.dw.k, 1, 0, dil, C, dHpad);
        bt::Tensor dh_conv;
        bt::pad1d_backward(dHpad, 1, C, T, pad, pad, /*mode=*/2, dh_conv);

        // residual: dInput = conv-chain grad + dY.
        bt::add_inplace(dh_conv, retag(dH, 1, C * T));
        dH = retag(dh_conv, 1, C * T).clone();
    }

    // ── conv_in (conv path): weight grad from re-padded spec input ─────────────
    const int pad_in = (ksz_init - 1) / 2, Lp_in = T + 2 * pad_in;
    bt::Tensor specpad;
    bt::pad1d_forward(flat1(cache.spec_in), 1, idim, T, pad_in, pad_in, /*mode=*/2, specpad);
    bt::conv1d_backward_weight(specpad, retag(dH, 1, C * T), 1, idim, Lp_in, C,
                               conv_in.k, 1, 0, 1, 1, grads.conv_in.w);
    if (conv_in.has_b)
        bt::conv1d_backward_bias(retag(dH, 1, C * T), 1, C, T, grads.conv_in.b);
}

}  // namespace brosoundml
