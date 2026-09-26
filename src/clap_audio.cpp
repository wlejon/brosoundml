// CLAP audio tower: HTSAT (unfused), transformers' ClapAudioEncoder — see
// include/brosoundml/clap.h and docs/clap.md.
//
// Layout conventions: the token grid of a stage is (H*W, C) in raster order
// (row h, column w), exactly transformers' (B, H*W, C) with B = 1. The Swin
// window partition, the cyclic shift and their inverses are all row
// permutations of that grid, so each block carries two INT32 index tensors
// and runs them through brotensor::gather_rows on the device; attention then
// runs window by window over non-owning views of the window-major buffer, with
// the relative-position bias and the shifted-window mask folded into one
// additive bias per window kind.
#include "clap_internal.h"

#include "qwen_tts_device.h"   // qtd::linear, qtd::upload_idx

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace brosoundml {
namespace clap {

namespace {

Linear load_linear(const sf::File& f, const std::string& p, int out, int in, bt::Device dev,
                   bool bias = true) {
    Linear l;
    l.w = upload(f, p + ".weight", out, in, dev);
    if (bias) l.b = upload_vec(f, p + ".bias", out, dev);
    return l;
}

Norm load_norm(const sf::File& f, const std::string& p, int d, bt::Device dev) {
    return Norm{upload_vec(f, p + ".weight", d, dev), upload_vec(f, p + ".bias", d, dev)};
}

bt::Tensor linear(const Linear& l, const bt::Tensor& x) {
    bt::Tensor y;
    qtd::linear(l.w, l.b.empty() ? nullptr : &l.b, x, y);
    return y;
}

bt::Tensor layer_norm(const Norm& n, const bt::Tensor& x, float eps) {
    bt::Tensor y;
    bt::layernorm_forward_inference_batched(x, n.w, n.b, y, eps);
    return y;
}

bt::Tensor gather(const bt::Tensor& x, const bt::Tensor& idx) {
    bt::Tensor y = bt::Tensor::empty_on(x.device, 0, 0, x.dtype);
    bt::gather_rows(x, idx, y);
    return y;
}

bt::Tensor index_tensor(bt::Device dev, const std::vector<std::int32_t>& v) {
    return qtd::upload_idx(dev, v.data(), static_cast<int>(v.size()));
}

// Build a block's window permutations and additive attention biases.
void build_block_tables(SwinBlock& b, int H, int W, const std::vector<float>& table, bt::Device dev) {
    const int ws = b.window, s = b.shift, heads = b.heads;
    const int L = ws * ws, nWh = H / ws, nWw = W / ws, N = H * W;
    std::vector<std::int32_t> to(N), from(N);
    for (int j = 0; j < N; ++j) {
        const int w = j / L, i = j % L;
        const int y = (w / nWw) * ws + i / ws, x = (w % nWw) * ws + i % ws;
        const int src = ((y + s) % H) * W + (x + s) % W;   // roll by -shift
        to[j] = src;
        from[src] = j;
    }
    b.to_windows = index_tensor(dev, to);
    b.from_windows = index_tensor(dev, from);

    // Relative-position bias: bias[h*L + q][k] = table[rpi(q, k)][h], with
    // rpi = (qh - kh + ws - 1) * (2ws - 1) + (qw - kw + ws - 1).
    std::vector<float> rel(static_cast<std::size_t>(heads) * L * L);
    for (int q = 0; q < L; ++q)
        for (int k = 0; k < L; ++k) {
            const int idx = (q / ws - k / ws + ws - 1) * (2 * ws - 1) + (q % ws - k % ws + ws - 1);
            for (int h = 0; h < heads; ++h)
                rel[(static_cast<std::size_t>(h) * L + q) * L + k] = table[static_cast<std::size_t>(idx) * heads + h];
        }

    // Window kinds: with a shift, the last window row / column straddles the
    // roll seam and masks cross-region pairs with -100 (transformers'
    // get_attn_mask). kind = (last row) + 2 * (last column).
    const int n_kinds = s > 0 ? 4 : 1;
    b.bias.clear();
    for (int kind = 0; kind < n_kinds; ++kind) {
        std::vector<float> bias = rel;
        if (s > 0) {
            auto region = [&](int v, int extent) { return v < extent - ws ? 0 : (v < extent - s ? 1 : 2); };
            // Any window of this kind: row / column index nWh-1 or 0.
            const int wr = (kind & 1) ? nWh - 1 : 0, wc = (kind & 2) ? nWw - 1 : 0;
            std::vector<int> label(L);
            for (int i = 0; i < L; ++i)
                label[i] = region(wr * ws + i / ws, H) * 3 + region(wc * ws + i % ws, W);
            for (int h = 0; h < heads; ++h)
                for (int q = 0; q < L; ++q)
                    for (int k = 0; k < L; ++k)
                        if (label[q] != label[k]) bias[(static_cast<std::size_t>(h) * L + q) * L + k] += -100.0f;
        }
        b.bias.push_back(bt::Tensor::from_host_on(dev, bias.data(), heads * L, L));
    }
    b.bias_kind.assign(static_cast<std::size_t>(nWh) * nWw, 0);
    if (s > 0)
        for (int w = 0; w < nWh * nWw; ++w)
            b.bias_kind[w] = ((w / nWw) == nWh - 1 ? 1 : 0) + ((w % nWw) == nWw - 1 ? 2 : 0);
}

// Window attention: one fused biased-MHA call per window, over non-owning
// views of the window-major buffer (projections + scores + bias + softmax + Wo
// in one op; the window's kind picks its bias).
bt::Tensor window_attention(const SwinBlock& b, const bt::Tensor& xw, float scale) {
    const int C = b.dim, L = b.window * b.window, N = xw.rows;
    bt::Tensor att = bt::Tensor::empty_on(xw.device, N, C, xw.dtype);
    float* xin = static_cast<float*>(xw.data);
    float* xout = static_cast<float*>(att.data);
    for (int w = 0; w < N / L; ++w) {
        const std::size_t off = static_cast<std::size_t>(w) * L * C;
        bt::Tensor xv = bt::Tensor::view(xw.device, xin + off, L, C);
        bt::Tensor ov = bt::Tensor::view(xw.device, xout + off, L, C);
        bt::self_attention_bias_forward(xv, b.q.w, b.k.w, b.v.w, b.o.w, &b.q.b, &b.k.b, &b.v.b, &b.o.b,
                                        /*d_mask=*/nullptr, &b.bias[b.bias_kind[w]], b.heads, scale, ov);
    }
    return att;
}

// Swin layer: x <- x + W-MSA(LN(x)); x <- x + MLP(LN(x)).
void swin_block(const SwinBlock& b, bt::Tensor& x, float eps) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(b.dim / b.heads));
    bt::Tensor xw = gather(layer_norm(b.ln_before, x, eps), b.to_windows);   // window-major
    bt::add_inplace(x, gather(window_attention(b, xw, scale), b.from_windows));

    bt::Tensor h = linear(b.fc1, layer_norm(b.ln_after, x, eps));
    bt::gelu_exact_forward(h, h);
    bt::add_inplace(x, linear(b.fc2, h));
}

}  // namespace

void AudioTower::load(const sf::File& f, const ClapConfig& cfg, bt::Device dev) {
    const std::string A = "audio_model.audio_encoder.";
    const int M = cfg.num_mel_bins, E0 = cfg.patch_embed_dim, P = cfg.patch_size;
    if (cfg.depths.size() != cfg.heads.size() || cfg.depths.empty()) fail("depths / heads mismatch");
    if (cfg.spec_size % M != 0) fail("spec_size must be a multiple of num_mel_bins");

    bn_w = upload_vec(f, A + "batch_norm.weight", M, dev);
    bn_b = upload_vec(f, A + "batch_norm.bias", M, dev);
    bn_mean = upload_vec(f, A + "batch_norm.running_mean", M, dev);
    bn_var = upload_vec(f, A + "batch_norm.running_var", M, dev);
    patch_w = upload(f, A + "patch_embed.proj.weight", E0, P * P, dev);
    patch_b = upload_vec(f, A + "patch_embed.proj.bias", E0, dev);
    patch_norm = load_norm(f, A + "patch_embed.norm", E0, dev);

    const int n_stages = static_cast<int>(cfg.depths.size());
    int H = cfg.spec_size / P, W = cfg.spec_size / P, dim = E0;
    stages.clear();
    stages.resize(n_stages);
    for (int s = 0; s < n_stages; ++s) {
        SwinStage& st = stages[s];
        st.H = H;
        st.W = W;
        st.dim = dim;
        // transformers' set_shift_and_window_size: a grid no larger than the
        // window is one unshifted window.
        int ws = cfg.window_size, shift = cfg.window_size / 2;
        if (std::min(H, W) <= ws) {
            ws = std::min(H, W);
            shift = 0;
        }
        if (H % ws != 0 || W % ws != 0) fail("stage grid is not a multiple of the window (padding unsupported)");
        const int heads = cfg.heads[s];
        st.blocks.resize(cfg.depths[s]);
        for (int i = 0; i < cfg.depths[s]; ++i) {
            SwinBlock& b = st.blocks[i];
            const std::string p = A + "layers." + std::to_string(s) + ".blocks." + std::to_string(i) + ".";
            b.dim = dim;
            b.heads = heads;
            b.window = ws;
            b.shift = (i % 2 == 1) ? shift : 0;
            b.ln_before = load_norm(f, p + "layernorm_before", dim, dev);
            b.ln_after = load_norm(f, p + "layernorm_after", dim, dev);
            b.q = load_linear(f, p + "attention.self.query", dim, dim, dev);
            b.k = load_linear(f, p + "attention.self.key", dim, dim, dev);
            b.v = load_linear(f, p + "attention.self.value", dim, dim, dev);
            b.o = load_linear(f, p + "attention.output.dense", dim, dim, dev);
            b.fc1 = load_linear(f, p + "intermediate.dense", 4 * dim, dim, dev);
            b.fc2 = load_linear(f, p + "output.dense", dim, 4 * dim, dev);
            // The table is (2*window_size - 1)^2 rows for the configured
            // window; a clamped window (the last stage) indexes its centre
            // block the same way only when ws == window_size, which holds for
            // every CLAP checkpoint (8x8 grid, window 8).
            const std::vector<float> table = host_f32(f, p + "attention.self.relative_position_bias_table");
            if (static_cast<int>(table.size()) != (2 * ws - 1) * (2 * ws - 1) * heads)
                fail("relative_position_bias_table size mismatch at " + p);
            build_block_tables(b, H, W, table, dev);
        }
        st.merge = (s + 1 < n_stages);
        if (st.merge) {
            const std::string p = A + "layers." + std::to_string(s) + ".downsample.";
            if (H % 2 || W % 2) fail("patch merging needs an even grid");
            st.merge_norm = load_norm(f, p + "norm", 4 * dim, dev);
            st.merge_red = load_linear(f, p + "reduction", 2 * dim, 4 * dim, dev, /*bias=*/false);
            // Output token (i, j) concatenates rows (2i,2j), (2i+1,2j),
            // (2i,2j+1), (2i+1,2j+1) — torch.cat over `for col .. for row`.
            std::vector<std::int32_t> idx;
            idx.reserve(static_cast<std::size_t>(H) * W);
            for (int i = 0; i < H / 2; ++i)
                for (int j = 0; j < W / 2; ++j) {
                    idx.push_back((2 * i) * W + 2 * j);
                    idx.push_back((2 * i + 1) * W + 2 * j);
                    idx.push_back((2 * i) * W + 2 * j + 1);
                    idx.push_back((2 * i + 1) * W + 2 * j + 1);
                }
            st.merge_idx = index_tensor(dev, idx);
            H /= 2;
            W /= 2;
            dim *= 2;
        }
    }
    if (dim != cfg.audio_hidden) fail("final stage width != audio hidden size");
    final_norm = load_norm(f, A + "norm", dim, dev);
    proj1 = load_linear(f, "audio_projection.linear1", cfg.projection_dim, dim, dev);
    proj2 = load_linear(f, "audio_projection.linear2", cfg.projection_dim, cfg.projection_dim, dev);
}

void AudioTower::forward(const std::vector<float>& mel, const ClapConfig& cfg, bt::Device dev,
                         std::vector<float>& pooled, std::vector<float>& projected) const {
    bt::DeviceScope scope(dev);
    const int T = cfg.frames(), M = cfg.num_mel_bins;
    if (static_cast<int>(mel.size()) != T * M) fail("log-mel must be frames x mel bins");
    const float eps = cfg.audio_ln_eps;
    const int freq_ratio = cfg.spec_size / M;             // 4
    const int spec_w = cfg.spec_size * freq_ratio;         // 1024 time frames
    if (T > spec_w) fail("more frames than the Swin input holds");

    // BatchNorm2d over mel bins: host (T, M) -> NCHW (1, M*1*T) with C = M.
    bt::Tensor x;
    {
        std::vector<float> cm(static_cast<std::size_t>(M) * T);
        for (int t = 0; t < T; ++t)
            for (int m = 0; m < M; ++m) cm[static_cast<std::size_t>(m) * T + t] = mel[static_cast<std::size_t>(t) * M + m];
        bt::Tensor in = bt::Tensor::from_host_on(dev, cm.data(), 1, M * T);
        bt::batch_norm_inference(in, bn_w, bn_b, bn_mean, bn_var, /*N=*/1, M, /*H=*/1, /*W=*/T, 1e-5f, x);
    }
    // Back to (T, M), then the bicubic (a = -0.75, align_corners) time stretch
    // to spec_w frames: interpolate((T, M) -> (spec_w, M)).
    bt::Tensor tm;
    bt::nchw_to_sequence(x, /*N=*/1, M, /*H=*/1, /*W=*/T, tm);   // (T, M)
    bt::Tensor st;
    if (T < spec_w) {
        tm.rows = 1;
        tm.cols = T * M;
        bt::interp2d_align_corners_forward(tm, /*N=*/1, /*C=*/1, T, M, spec_w, M, /*mode=*/3, st);
    } else {
        st = std::move(tm);
    }
    // reshape_mel2img: four 256-frame chunks, each transposed to (M, 256) and
    // stacked -> the (256, 256) image. As NCHW (N=4, C=256, H=1, W=M) that is
    // exactly nchw_to_sequence: Y[chunk*M + f, t] = X[chunk, t, f].
    bt::Tensor img;
    bt::nchw_to_sequence(st, /*N=*/freq_ratio, /*C=*/cfg.spec_size, /*H=*/1, /*W=*/M, img);   // (256, 256)
    img.rows = 1;
    img.cols = cfg.spec_size * cfg.spec_size;

    // Patch embed: 4x4 stride-4 conv -> (E0, 64, 64) -> tokens (4096, E0) -> LN.
    const int P = cfg.patch_size, G = cfg.spec_size / P, E0 = cfg.patch_embed_dim;
    bt::Tensor pe;
    bt::conv2d_forward(img, patch_w, &patch_b, /*N=*/1, /*C_in=*/1, cfg.spec_size, cfg.spec_size, E0, P, P,
                       P, P, /*pad=*/0, 0, /*dil=*/1, 1, pe);
    bt::Tensor tok;
    bt::nchw_to_sequence(pe, /*N=*/1, E0, G, G, tok);   // (G*G, E0)
    bt::Tensor h = layer_norm(patch_norm, tok, eps);

    for (const SwinStage& s : stages) {
        for (const SwinBlock& b : s.blocks) swin_block(b, h, eps);
        if (s.merge) {
            bt::Tensor g = gather(h, s.merge_idx);   // (H*W, C) in 2x2 groups
            g.rows = (s.H / 2) * (s.W / 2);
            g.cols = 4 * s.dim;
            h = linear(s.merge_red, layer_norm(s.merge_norm, g, eps));
        }
    }
    h = layer_norm(final_norm, h, eps);

    // The group-2D-CNN reshape before AdaptiveAvgPool1d only permutes the
    // tokens, so the pooled output is the plain mean over them.
    bt::Tensor sum;
    bt::sum_cols(h, sum);   // (1, D)
    bt::scale_inplace(sum, 1.0f / static_cast<float>(h.rows));

    bt::Tensor p1 = linear(proj1, sum);
    bt::relu_forward(p1, p1);
    bt::Tensor p2 = linear(proj2, p1);
    bt::sync(dev);
    pooled = sum.to(bt::Device::CPU).to_host_vector();
    projected = p2.to(bt::Device::CPU).to_host_vector();
}

}  // namespace clap
}  // namespace brosoundml
