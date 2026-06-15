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

}  // namespace brosoundml
