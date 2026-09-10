// HuBERT-base — see src/hubert.h.
#include "hubert.h"

#include "higgs_codec_common.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cmath>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;
using namespace hcodec;   // fail/need/up/up_vec, ncl<->seq, conv helpers, lnorm, gelu

void HubertModel::load(const sf::File& f, const std::string& prefix,
                       const HubertConfig& config, bt::Device dev) {
    cfg = config;
    const std::string P = prefix + ".";
    const int D = cfg.hidden_size;
    const int I = cfg.intermediate_size;
    const int n_conv = static_cast<int>(cfg.conv_dim.size());
    if (n_conv == 0 || static_cast<int>(cfg.conv_kernel.size()) != n_conv ||
        static_cast<int>(cfg.conv_stride.size()) != n_conv)
        fail("hubert: conv_dim / conv_kernel / conv_stride must have equal length");
    if (cfg.do_stable_layer_norm) fail("hubert: do_stable_layer_norm (pre-LN) is not supported");
    if (D % cfg.num_attention_heads != 0) fail("hubert: hidden_size % num_attention_heads != 0");

    // ── feature extractor ──
    conv.clear();
    conv.resize(n_conv);
    int cin = 1;
    for (int i = 0; i < n_conv; ++i) {
        HubertConvLayer& c = conv[i];
        const std::string L = P + "feature_extractor.conv_layers." + std::to_string(i) + ".";
        c.cin = cin;
        c.cout = cfg.conv_dim[i];
        c.k = cfg.conv_kernel[i];
        c.stride = cfg.conv_stride[i];
        c.w = up(f, L + "conv.weight", c.cout, c.cin * c.k, dev);
        c.has_bias = cfg.conv_bias;
        if (c.has_bias) c.b = up_vec(f, L + "conv.bias", c.cout, dev);
        c.has_gn = (i == 0 && cfg.feat_extract_norm_group);
        if (c.has_gn) {
            c.gn_w = up_vec(f, L + "layer_norm.weight", c.cout, dev);
            c.gn_b = up_vec(f, L + "layer_norm.bias", c.cout, dev);
        }
        cin = c.cout;
    }
    const int feat = cin;   // conv_dim.back()

    // ── feature projection ──
    if (cfg.feat_proj_layer_norm) {
        fp_ln_w = up_vec(f, P + "feature_projection.layer_norm.weight", feat, dev);
        fp_ln_b = up_vec(f, P + "feature_projection.layer_norm.bias", feat, dev);
    }
    fp_w = up(f, P + "feature_projection.projection.weight", D, feat, dev);
    fp_b = up_vec(f, P + "feature_projection.projection.bias", D, dev);

    // ── positional conv: fold weight_norm(dim=2) host-side ──
    //   w[o,i,k] = g[k] * v[o,i,k] / sqrt(sum_{o,i} v[o,i,k]^2)
    {
        const int k = cfg.num_conv_pos_embeddings;
        const int ipg = D / cfg.num_conv_pos_embedding_groups;   // in-channels per group
        const std::string C = P + "encoder.pos_conv_embed.conv.";
        bt::Tensor g, v;
        {
            bt::DeviceScope cpu(bt::Device::CPU);
            sf::upload(need(f, C + "parametrizations.weight.original0"), k, 1, g);
            sf::upload(need(f, C + "parametrizations.weight.original1"), D, ipg * k, v);
        }
        const float* gp = g.host_f32();
        float* vp = v.host_f32_mut();
        std::vector<double> norm2(static_cast<std::size_t>(k), 0.0);
        for (int o = 0; o < D; ++o)
            for (int i = 0; i < ipg; ++i) {
                const float* row = vp + (static_cast<std::size_t>(o) * ipg + i) * k;
                for (int kk = 0; kk < k; ++kk) norm2[kk] += static_cast<double>(row[kk]) * row[kk];
            }
        std::vector<float> scale(static_cast<std::size_t>(k));
        for (int kk = 0; kk < k; ++kk)
            scale[kk] = static_cast<float>(gp[kk] / std::sqrt(norm2[kk]));
        for (int o = 0; o < D; ++o)
            for (int i = 0; i < ipg; ++i) {
                float* row = vp + (static_cast<std::size_t>(o) * ipg + i) * k;
                for (int kk = 0; kk < k; ++kk) row[kk] *= scale[kk];
            }
        pos_w = (dev == bt::Device::CPU) ? v : v.to(dev);
        pos_b = up_vec(f, C + "bias", D, dev);
    }
    enc_ln_w = up_vec(f, P + "encoder.layer_norm.weight", D, dev);
    enc_ln_b = up_vec(f, P + "encoder.layer_norm.bias", D, dev);

    // ── transformer layers ──
    layers.clear();
    layers.resize(cfg.num_hidden_layers);
    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        const std::string L = P + "encoder.layers." + std::to_string(l) + ".";
        HubertLayer& t = layers[l];
        t.qw = up(f, L + "attention.q_proj.weight", D, D, dev);
        t.qb = up_vec(f, L + "attention.q_proj.bias", D, dev);
        t.kw = up(f, L + "attention.k_proj.weight", D, D, dev);
        t.kb = up_vec(f, L + "attention.k_proj.bias", D, dev);
        t.vw = up(f, L + "attention.v_proj.weight", D, D, dev);
        t.vb = up_vec(f, L + "attention.v_proj.bias", D, dev);
        t.ow = up(f, L + "attention.out_proj.weight", D, D, dev);
        t.ob = up_vec(f, L + "attention.out_proj.bias", D, dev);
        t.ln_w = up_vec(f, L + "layer_norm.weight", D, dev);
        t.ln_b = up_vec(f, L + "layer_norm.bias", D, dev);
        t.fc1_w = up(f, L + "feed_forward.intermediate_dense.weight", I, D, dev);
        t.fc1_b = up_vec(f, L + "feed_forward.intermediate_dense.bias", I, dev);
        t.fc2_w = up(f, L + "feed_forward.output_dense.weight", D, I, dev);
        t.fc2_b = up_vec(f, L + "feed_forward.output_dense.bias", D, dev);
        t.fln_w = up_vec(f, L + "final_layer_norm.weight", D, dev);
        t.fln_b = up_vec(f, L + "final_layer_norm.bias", D, dev);
    }
}

int HubertModel::output_frames(int n) const {
    for (const HubertConvLayer& c : conv) {
        if (n < c.k) return 0;
        n = (n - c.k) / c.stride + 1;
    }
    return n;
}

bt::Tensor HubertModel::forward_mean_hidden(const bt::Tensor& wav16, int n, int* Th_out) const {
    if (!loaded()) fail("hubert: not loaded");
    const bt::Device dev = fp_w.device;
    bt::DeviceScope scope(dev);
    const int D = cfg.hidden_size;
    const int H = cfg.num_attention_heads;
    const float eps = cfg.layer_norm_eps;

    // ── feature extractor (NCL) ──
    bt::Tensor x = wav16;   // (1, 1*n)
    int L = n;
    for (const HubertConvLayer& c : conv) {
        if (L < c.k) fail("hubert: input too short for the feature extractor");
        x = conv_valid(x, c.cin, L, c.w, c.has_bias ? &c.b : nullptr, c.cout, c.k, c.stride);
        L = (L - c.k) / c.stride + 1;
        if (c.has_gn) {
            bt::Tensor y;
            bt::group_norm_forward(x, c.gn_w, c.gn_b, /*N=*/1, c.cout, /*H=*/1, /*W=*/L,
                                   /*num_groups=*/c.cout, 1e-5f, y);
            x = std::move(y);
        }
        gelu_inplace(x);
    }
    const int Th = L;
    const int feat = conv.back().cout;

    // ── feature projection ──
    bt::Tensor h = ncl_to_seq(x, feat, Th);   // (Th, feat)
    if (cfg.feat_proj_layer_norm) h = lnorm(h, fp_ln_w, fp_ln_b, eps);
    {
        bt::Tensor p;
        qtd::linear(fp_w, &fp_b, h, p);   // (Th, D)
        h = std::move(p);
    }

    // ── positional conv embedding (+ SamePad trim + GELU) ──
    {
        const int k = cfg.num_conv_pos_embeddings;
        const int groups = cfg.num_conv_pos_embedding_groups;
        bt::Tensor ncl = seq_to_ncl(h, Th, D);   // (1, D*Th)
        // torch pads k/2 both sides and SamePad drops the last output sample
        // (k even): identical to pad (k/2, k/2 - 1) then a valid conv.
        bt::Tensor padded;
        bt::pad1d_forward(ncl, /*N=*/1, D, Th, k / 2, k / 2 - (k % 2 == 0 ? 1 : 0),
                          /*mode=*/0, padded);
        const int Lp = Th + k - (k % 2 == 0 ? 1 : 0);
        bt::Tensor pos;
        bt::conv1d(padded, pos_w, &pos_b, /*N=*/1, D, Lp, D, k, /*stride=*/1,
                   /*padding=*/0, /*dilation=*/1, groups, pos);   // (1, D*Th)
        gelu_inplace(pos);
        bt::Tensor pos_seq = ncl_to_seq(pos, D, Th);
        bt::add_inplace(h, pos_seq);
    }
    h = lnorm(h, enc_ln_w, enc_ln_b, eps);

    // ── 13 hidden states, summed as they are produced ──
    bt::Tensor acc = h;   // deep copy: hidden_states[0]
    for (const HubertLayer& t : layers) {
        bt::Tensor q, kk, v;
        qtd::linear(t.qw, &t.qb, h, q);
        qtd::linear(t.kw, &t.kb, h, kk);
        qtd::linear(t.vw, &t.vb, h, v);
        bt::Tensor ctx;
        bt::flash_attention_gqa_forward(q, kk, v, /*d_mask=*/nullptr, H, H,
                                        /*causal=*/false, ctx);
        bt::Tensor attn;
        qtd::linear(t.ow, &t.ob, ctx, attn);
        bt::add_inplace(h, attn);
        h = lnorm(h, t.ln_w, t.ln_b, eps);

        bt::Tensor f1;
        qtd::linear(t.fc1_w, &t.fc1_b, h, f1);
        gelu_inplace(f1);
        bt::Tensor f2;
        qtd::linear(t.fc2_w, &t.fc2_b, f1, f2);
        bt::add_inplace(h, f2);
        h = lnorm(h, t.fln_w, t.fln_b, eps);

        bt::add_inplace(acc, h);
    }
    bt::scale_inplace(acc, 1.0f / static_cast<float>(layers.size() + 1));
    if (Th_out) *Th_out = Th;
    return acc;
}

}  // namespace brosoundml
