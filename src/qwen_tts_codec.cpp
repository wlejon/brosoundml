#include "qwen_tts_codec.h"

#include "qwen_tts_device.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: QwenTtsCodecDecoder: " + msg);
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Upload a 2D-flattened weight to FP32 on `dev` (codec weights are F32 on disk).
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols,
              bt::Device dev) {
    bt::Tensor t;
    { bt::DeviceScope cpu(bt::Device::CPU); sf::upload(need(f, name), rows, cols, t); }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
bt::Tensor up_vec(const sf::File& f, const std::string& name, int c, bt::Device dev) {
    return up(f, name, c, 1, dev);
}

// SnakeBeta parameter [C], exponentiated host-side (the upstream SnakeBeta
// applies exp() to its raw alpha/beta), then placed on `dev`.
bt::Tensor up_snake(const sf::File& f, const std::string& name, int c, bt::Device dev) {
    bt::Tensor t;
    { bt::DeviceScope cpu(bt::Device::CPU); sf::upload(need(f, name), c, 1, t); }
    float* p = t.host_f32_mut();
    for (int i = 0; i < c; ++i) p[i] = std::exp(p[i]);
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}

// EMA codebook -> effective embedding: embed[k] = embedding_sum[k] /
// max(cluster_usage[k], 1e-5). Combined host-side, then placed on `dev`.
bt::Tensor load_codebook(const sf::File& f, const std::string& prefix,
                         int bins, int dim, bt::Device dev) {
    bt::Tensor embed, usage;
    {
        bt::DeviceScope cpu(bt::Device::CPU);
        sf::upload(need(f, prefix + ".embedding_sum"), bins, dim, embed);
        sf::upload(need(f, prefix + ".cluster_usage"), bins, 1, usage);
    }
    float* e = embed.host_f32_mut();
    const float* u = usage.host_f32();
    for (int k = 0; k < bins; ++k) {
        const float d = 1.0f / (u[k] > 1e-5f ? u[k] : 1e-5f);
        float* row = e + static_cast<std::size_t>(k) * dim;
        for (int j = 0; j < dim; ++j) row[j] *= d;
    }
    return (dev == bt::Device::CPU) ? embed : embed.to(dev);
}

// ── layout converters (device, via host-roundtrip transpose) ─────────────────
// NCL (1, C*L) viewed channel-major as (C, L) <-> SEQ (L, C).
bt::Tensor ncl_to_seq(const bt::Tensor& x, int C, int L) {
    bt::Tensor cl = bt::Tensor::view(x.device, x.data, C, L, x.dtype);
    return qtd::transpose2d(cl);   // (L, C)
}
bt::Tensor seq_to_ncl(const bt::Tensor& x, int L, int C) {
    bt::Tensor cl = qtd::transpose2d(x);   // (C, L)
    cl.rows = 1; cl.cols = C * L;
    return cl;
}

// Stride-1 causal conv: left-pad dilation*(k-1), valid conv, Lout == L.
bt::Tensor causal_conv(const bt::Tensor& x_ncl, int Cin, int L, const bt::Tensor& w,
                       const bt::Tensor* b, int Cout, int k, int dilation, int groups) {
    bt::Tensor scratch, y;
    bt::causal_conv1d(x_ncl, w, b, /*N=*/1, Cin, L, Cout, k, /*stride=*/1,
                      dilation, groups, scratch, y);
    return y;
}

// Causal transposed conv: conv_transpose1d then trim (kernel-stride) from the
// right per channel (left_pad is 0 for these causal upsamplers). Lout = L*stride.
bt::Tensor trans_conv(const bt::Tensor& x_ncl, int Cin, int L, const bt::Tensor& w,
                      const bt::Tensor* b, int Cout, int k, int stride) {
    bt::Tensor full;
    bt::conv_transpose1d_forward(x_ncl, w, b, /*N=*/1, Cin, L, Cout, k, stride,
                                 /*padding=*/0, /*output_padding=*/0,
                                 /*dilation=*/1, full);
    const int L_full = (L - 1) * stride + k;
    const int trim   = k - stride;
    const int L_out  = L_full - trim;
    bt::Tensor y = bt::Tensor::zeros_on(x_ncl.device, 1, Cout * L_out, bt::Dtype::FP32);
    for (int c = 0; c < Cout; ++c)
        bt::copy_d2d(full, c * L_full, y, c * L_out, L_out);   // drop the right tail
    return y;
}

// SnakeBeta in place (alpha/beta pre-exp'd).
void snake(bt::Tensor& x_ncl, int C, int L, const bt::Tensor& alpha,
           const bt::Tensor& beta) {
    bt::snake_forward(x_ncl, alpha, &beta, /*N=*/1, C, L, x_ncl);
}

// hs(T,C) += scale[c] * delta(T,C)  (LayerScale residual).
void add_layerscale(bt::Tensor& hs, const bt::Tensor& delta, const bt::Tensor& scale) {
    bt::Tensor scaled;
    bt::broadcast_mul(delta, scale, scaled);
    bt::add_inplace(hs, scaled);
}

// Windowed causal self-attention over pre-RoPE Q/K/V (T, num_heads*head_dim).
// For T <= window it is plain causal (device FP32 flash); beyond the window the
// per-query left window differs per row (flash has no sliding-window mode), so
// fall back to a host computation (the heavy long-audio path).
bt::Tensor codec_attn(const bt::Tensor& q, const bt::Tensor& k, const bt::Tensor& v,
                      int num_heads, int head_dim, int T, int window) {
    bt::Tensor O;
    if (T <= window) {
        qtd::flash_attn(q, k, v, num_heads, head_dim, /*causal=*/true, O);
        return O;
    }
    const int inner = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> qh(static_cast<std::size_t>(T) * inner);
    std::vector<float> kh(static_cast<std::size_t>(T) * inner);
    std::vector<float> vh(static_cast<std::size_t>(T) * inner);
    qtd::to_host(q, qh.data());
    qtd::to_host(k, kh.data());
    qtd::to_host(v, vh.data());
    std::vector<float> ctx(static_cast<std::size_t>(T) * inner, 0.0f);
    std::vector<float> scores(T);
    for (int h = 0; h < num_heads; ++h) {
        const int off = h * head_dim;
        for (int i = 0; i < T; ++i) {
            const int jlo = (i - window + 1 > 0) ? i - window + 1 : 0;
            const float* qi = qh.data() + static_cast<std::size_t>(i) * inner + off;
            float maxs = -1e30f;
            for (int j = jlo; j <= i; ++j) {
                const float* kj = kh.data() + static_cast<std::size_t>(j) * inner + off;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
                dot *= scale;
                scores[j] = dot;
                if (dot > maxs) maxs = dot;
            }
            float sum = 0.0f;
            for (int j = jlo; j <= i; ++j) {
                const float e = std::exp(scores[j] - maxs);
                scores[j] = e; sum += e;
            }
            const float inv = 1.0f / sum;
            float* ci = ctx.data() + static_cast<std::size_t>(i) * inner + off;
            for (int j = jlo; j <= i; ++j) {
                const float wgt = scores[j] * inv;
                const float* vj = vh.data() + static_cast<std::size_t>(j) * inner + off;
                for (int d = 0; d < head_dim; ++d) ci[d] += wgt * vj[d];
            }
        }
    }
    return bt::Tensor::from_host_on(q.device, ctx.data(), T, inner);
}

}  // namespace

void QwenTtsCodecDecoder::load(const sf::File& f, const QwenTtsCodecConfig& cfg,
                               bt::Device dev) {
    num_quantizers = cfg.num_quantizers;
    num_semantic   = cfg.num_semantic_quantizers;
    codebook_dim   = cfg.codebook_dim;
    vq_dim         = cfg.codebook_dim / 2;
    codebook_bins  = cfg.codebook_size;
    latent_dim     = cfg.latent_dim;
    hidden         = cfg.pre_transformer.hidden_size;
    decoder_dim    = cfg.decoder_dim;
    num_heads      = cfg.pre_transformer.num_attention_heads;
    head_dim       = cfg.pre_transformer.head_dim;
    sliding_window = cfg.sliding_window;
    rms_eps        = cfg.pre_transformer.rms_norm_eps;
    rope_theta     = cfg.pre_transformer.rope_theta;
    upsample_rates    = cfg.upsample_rates;
    upsampling_ratios = cfg.upsampling_ratios;

    const std::string P = "decoder.";

    // ── quantizer ──
    codebook.clear();
    codebook.push_back(load_codebook(
        f, P + "quantizer.rvq_first.vq.layers.0._codebook", codebook_bins, vq_dim, dev));
    for (int l = 0; l < num_quantizers - num_semantic; ++l) {
        codebook.push_back(load_codebook(
            f, P + "quantizer.rvq_rest.vq.layers." + std::to_string(l) + "._codebook",
            codebook_bins, vq_dim, dev));
    }
    out_proj_first = up(f, P + "quantizer.rvq_first.output_proj.weight", codebook_dim, vq_dim, dev);
    out_proj_rest  = up(f, P + "quantizer.rvq_rest.output_proj.weight",  codebook_dim, vq_dim, dev);

    // ── pre_conv (codebook_dim -> latent_dim, k3) ──
    pre_conv_w = up(f, P + "pre_conv.conv.weight", latent_dim, codebook_dim * 3, dev);
    pre_conv_b = up_vec(f, P + "pre_conv.conv.bias", latent_dim, dev);

    // ── pre_transformer ──
    in_proj_w  = up(f, P + "pre_transformer.input_proj.weight",  hidden, latent_dim, dev);
    in_proj_b  = up_vec(f, P + "pre_transformer.input_proj.bias", hidden, dev);
    out_proj_w = up(f, P + "pre_transformer.output_proj.weight", latent_dim, hidden, dev);
    out_proj_b = up_vec(f, P + "pre_transformer.output_proj.bias", latent_dim, dev);
    final_norm = up_vec(f, P + "pre_transformer.norm.weight", hidden, dev);

    // HF rotate-half -> brotensor adjacent-pair RoPE: permute q/k rows at load
    // (the codec attention has no QK-norm, so only the projections move).
    const std::vector<std::int32_t> q_perm =
        qtd::per_head_perm_rows(qtd::rotate_half_perm(head_dim), num_heads, head_dim);

    const int inner = num_heads * head_dim;
    const int inter = cfg.pre_transformer.intermediate_size;
    layers.clear();
    layers.resize(cfg.pre_transformer.num_hidden_layers);
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const std::string L = P + "pre_transformer.layers." + std::to_string(i) + ".";
        QwenTtsCodecTfLayer& tl = layers[i];
        tl.in_ln       = up_vec(f, L + "input_layernorm.weight", hidden, dev);
        tl.post_ln     = up_vec(f, L + "post_attention_layernorm.weight", hidden, dev);
        tl.qw          = qtd::gather_rows(up(f, L + "self_attn.q_proj.weight", inner, hidden, dev), q_perm);
        tl.kw          = qtd::gather_rows(up(f, L + "self_attn.k_proj.weight", inner, hidden, dev), q_perm);
        tl.vw          = up(f, L + "self_attn.v_proj.weight", inner, hidden, dev);
        tl.ow          = up(f, L + "self_attn.o_proj.weight", hidden, inner, dev);
        tl.attn_scale  = up_vec(f, L + "self_attn_layer_scale.scale", hidden, dev);
        tl.mlp_scale   = up_vec(f, L + "mlp_layer_scale.scale", hidden, dev);
        tl.gate        = up(f, L + "mlp.gate_proj.weight", inter, hidden, dev);
        tl.up          = up(f, L + "mlp.up_proj.weight", inter, hidden, dev);
        tl.down        = up(f, L + "mlp.down_proj.weight", hidden, inter, dev);
    }

    // ── ConvNeXt upsample stages ──
    upsample.clear();
    upsample.resize(upsampling_ratios.size());
    for (std::size_t s = 0; s < upsample.size(); ++s) {
        const int factor = upsampling_ratios[s];
        const std::string U = P + "upsample." + std::to_string(s) + ".";
        QwenTtsCodecUpStage& us = upsample[s];
        us.factor  = factor;
        us.tconv_w = up(f, U + "0.conv.weight", latent_dim, latent_dim * factor, dev);
        us.tconv_b = up_vec(f, U + "0.conv.bias", latent_dim, dev);
        us.dw_w    = up(f, U + "1.dwconv.conv.weight", latent_dim, 7, dev);  // depthwise
        us.dw_b    = up_vec(f, U + "1.dwconv.conv.bias", latent_dim, dev);
        us.ln_w    = up_vec(f, U + "1.norm.weight", latent_dim, dev);
        us.ln_b    = up_vec(f, U + "1.norm.bias", latent_dim, dev);
        us.pw1_w   = up(f, U + "1.pwconv1.weight", 4 * latent_dim, latent_dim, dev);
        us.pw1_b   = up_vec(f, U + "1.pwconv1.bias", 4 * latent_dim, dev);
        us.pw2_w   = up(f, U + "1.pwconv2.weight", latent_dim, 4 * latent_dim, dev);
        us.pw2_b   = up_vec(f, U + "1.pwconv2.bias", latent_dim, dev);
        us.gamma   = up_vec(f, U + "1.gamma", latent_dim, dev);
    }

    // ── SEANet decoder ──
    dec0_w = up(f, P + "decoder.0.conv.weight", decoder_dim, latent_dim * 7, dev);
    dec0_b = up_vec(f, P + "decoder.0.conv.bias", decoder_dim, dev);

    static const int kResDilations[3] = {1, 3, 9};
    blocks.clear();
    blocks.resize(upsample_rates.size());
    for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
        QwenTtsCodecDecBlock& blk = blocks[bi];
        blk.in_dim  = decoder_dim >> bi;
        blk.out_dim = decoder_dim >> (bi + 1);
        blk.rate    = upsample_rates[bi];
        const std::string B = P + "decoder." + std::to_string(bi + 1) + ".block.";
        blk.s_alpha = up_snake(f, B + "0.alpha", blk.in_dim, dev);
        blk.s_beta  = up_snake(f, B + "0.beta",  blk.in_dim, dev);
        blk.tconv_w = up(f, B + "1.conv.weight", blk.in_dim, blk.out_dim * 2 * blk.rate, dev);
        blk.tconv_b = up_vec(f, B + "1.conv.bias", blk.out_dim, dev);
        blk.units.resize(3);
        for (int ui = 0; ui < 3; ++ui) {
            QwenTtsCodecResUnit& u = blk.units[ui];
            const std::string R = B + std::to_string(2 + ui) + ".";
            u.dim      = blk.out_dim;
            u.dilation = kResDilations[ui];
            u.a1_alpha = up_snake(f, R + "act1.alpha", u.dim, dev);
            u.a1_beta  = up_snake(f, R + "act1.beta",  u.dim, dev);
            u.a2_alpha = up_snake(f, R + "act2.alpha", u.dim, dev);
            u.a2_beta  = up_snake(f, R + "act2.beta",  u.dim, dev);
            u.c1w      = up(f, R + "conv1.conv.weight", u.dim, u.dim * 7, dev);
            u.c1b      = up_vec(f, R + "conv1.conv.bias", u.dim, dev);
            u.c2w      = up(f, R + "conv2.conv.weight", u.dim, u.dim * 1, dev);
            u.c2b      = up_vec(f, R + "conv2.conv.bias", u.dim, dev);
        }
    }

    const int out_dim = decoder_dim >> upsample_rates.size();
    dec_final_alpha = up_snake(f, P + "decoder.5.alpha", out_dim, dev);
    dec_final_beta  = up_snake(f, P + "decoder.5.beta",  out_dim, dev);
    dec_out_w = up(f, P + "decoder.6.conv.weight", 1, out_dim * 7, dev);
    dec_out_b = up_vec(f, P + "decoder.6.conv.bias", 1, dev);
}

void QwenTtsCodecDecoder::decode(const int32_t* codes, int K, int T,
                                 std::vector<float>& wav) const {
    const bt::Device dev = final_norm.device;
    bt::DeviceScope scope(dev);
    if (K != num_quantizers)
        fail("expected " + std::to_string(num_quantizers) + " codebooks, got " +
             std::to_string(K));
    if (T <= 0) return;

    const int bins = codebook_bins;
    // Sum the codebook lookups of `count` consecutive code rows (device gather),
    // then project vq_dim -> codebook_dim.
    auto gather_group = [&](int first, int count, const bt::Tensor& proj) {
        bt::Tensor acc = bt::Tensor::zeros_on(dev, T, vq_dim, bt::Dtype::FP32);
        std::vector<std::int32_t> idx(T);
        for (int l = 0; l < count; ++l) {
            const int32_t* row = codes + static_cast<std::size_t>(first + l) * T;
            for (int t = 0; t < T; ++t) {
                int v = row[t];
                idx[t] = (v < 0) ? 0 : (v >= bins ? bins - 1 : v);
            }
            bt::Tensor looked = qtd::gather_rows(codebook[first + l], idx);  // (T, vq_dim)
            bt::add_inplace(acc, looked);
        }
        bt::Tensor out;
        qtd::linear(proj, nullptr, acc, out);   // (T, codebook_dim)
        return out;
    };

    // ── quantizer.decode: semantic + acoustic groups summed ──
    bt::Tensor y_first = gather_group(0, num_semantic, out_proj_first);
    bt::Tensor y_rest  = gather_group(num_semantic, num_quantizers - num_semantic, out_proj_rest);
    bt::Tensor latent_seq = y_first;          // (T, codebook_dim)
    bt::add_inplace(latent_seq, y_rest);

    // ── pre_conv (codebook_dim -> latent_dim, causal k3) ──
    bt::Tensor ncl = seq_to_ncl(latent_seq, T, codebook_dim);
    ncl = causal_conv(ncl, codebook_dim, T, pre_conv_w, &pre_conv_b, latent_dim, 3, 1, 1);

    // ── pre_transformer ──
    bt::Tensor hs;  // (T, hidden)
    {
        bt::Tensor lat_seq = ncl_to_seq(ncl, latent_dim, T);   // (T, latent_dim)
        qtd::linear(in_proj_w, &in_proj_b, lat_seq, hs);       // (T, hidden)
    }
    // Plain RoPE tables (pos = row index), shared across layers.
    const int half = head_dim / 2;
    {
        std::vector<int>   pos_grid(static_cast<std::size_t>(T) * half);
        std::vector<float> inv_freq(half);
        for (int i = 0; i < half; ++i) inv_freq[i] = std::pow(rope_theta, -(2.0f * i) / head_dim);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < half; ++i)
                pos_grid[static_cast<std::size_t>(t) * half + i] = t;
        bt::Tensor cosT, sinT;
        qtd::build_rope_tables(dev, T, half, pos_grid, inv_freq, cosT, sinT);

        for (const QwenTtsCodecTfLayer& tl : layers) {
            bt::Tensor normed;
            bt::rms_norm_forward(hs, tl.in_ln, rms_eps, normed);
            bt::Tensor q, k, v;
            qtd::linear(tl.qw, nullptr, normed, q);
            qtd::linear(tl.kw, nullptr, normed, k);
            qtd::linear(tl.vw, nullptr, normed, v);
            bt::Tensor qr, kr;
            bt::rope_apply(q, cosT, sinT, head_dim, num_heads, qr);
            bt::rope_apply(k, cosT, sinT, head_dim, num_heads, kr);
            bt::Tensor ctx = codec_attn(qr, kr, v, num_heads, head_dim, T, sliding_window);
            bt::Tensor attn;
            qtd::linear(tl.ow, nullptr, ctx, attn);            // (T, hidden)
            add_layerscale(hs, attn, tl.attn_scale);

            bt::Tensor n2;
            bt::rms_norm_forward(hs, tl.post_ln, rms_eps, n2);
            bt::Tensor g, u;
            qtd::linear(tl.gate, nullptr, n2, g);
            qtd::linear(tl.up,   nullptr, n2, u);
            qtd::swiglu(g, u);
            bt::Tensor dn;
            qtd::linear(tl.down, nullptr, g, dn);              // (T, hidden)
            add_layerscale(hs, dn, tl.mlp_scale);
        }
    }
    {
        bt::Tensor normed;
        bt::rms_norm_forward(hs, final_norm, rms_eps, normed);
        bt::Tensor lat;
        qtd::linear(out_proj_w, &out_proj_b, normed, lat);     // (T, latent_dim)
        ncl = seq_to_ncl(lat, T, latent_dim);                  // (1, latent_dim*T)
    }

    // ── ConvNeXt upsample (x2 x2) ──
    int C = latent_dim;
    int L = T;
    for (const QwenTtsCodecUpStage& us : upsample) {
        ncl = trans_conv(ncl, C, L, us.tconv_w, &us.tconv_b, C, us.factor, us.factor);
        L *= us.factor;
        // ConvNeXt block (depthwise k7, LN, pw 1->4->1 GELU, gamma, residual).
        bt::Tensor dw = causal_conv(ncl, C, L, us.dw_w, &us.dw_b, C, 7, 1, /*groups=*/C);
        bt::Tensor seq = ncl_to_seq(dw, C, L);  // (L, C)
        bt::layernorm_forward_inference_batched(seq, us.ln_w, us.ln_b, seq, 1e-6f);
        bt::Tensor h1;
        qtd::linear(us.pw1_w, &us.pw1_b, seq, h1);   // (L, 4C)
        bt::gelu_exact_forward(h1, h1);
        bt::Tensor h2;
        qtd::linear(us.pw2_w, &us.pw2_b, h1, h2);    // (L, C)
        bt::Tensor h2g;
        bt::broadcast_mul(h2, us.gamma, h2g);
        bt::Tensor back = seq_to_ncl(h2g, L, C);     // (1, C*L)
        bt::add_inplace(ncl, back);
    }

    // ── SEANet decoder ──
    ncl = causal_conv(ncl, latent_dim, L, dec0_w, &dec0_b, decoder_dim, 7, 1, 1);
    C = decoder_dim;
    for (const QwenTtsCodecDecBlock& blk : blocks) {
        snake(ncl, blk.in_dim, L, blk.s_alpha, blk.s_beta);
        ncl = trans_conv(ncl, blk.in_dim, L, blk.tconv_w, &blk.tconv_b, blk.out_dim,
                         2 * blk.rate, blk.rate);
        L *= blk.rate;
        C = blk.out_dim;
        for (const QwenTtsCodecResUnit& u : blk.units) {
            bt::Tensor res = ncl;   // residual input (deep copy)
            snake(ncl, u.dim, L, u.a1_alpha, u.a1_beta);
            bt::Tensor t1 = causal_conv(ncl, u.dim, L, u.c1w, &u.c1b, u.dim, 7, u.dilation, 1);
            snake(t1, u.dim, L, u.a2_alpha, u.a2_beta);
            bt::Tensor t2 = causal_conv(t1, u.dim, L, u.c2w, &u.c2b, u.dim, 1, 1, 1);
            bt::add_inplace(res, t2);
            ncl = std::move(res);
        }
    }
    snake(ncl, C, L, dec_final_alpha, dec_final_beta);
    bt::Tensor wav_ncl = causal_conv(ncl, C, L, dec_out_w, &dec_out_b, 1, 7, 1, 1);

    // ── clamp [-1, 1] and append ──
    const int n = wav_ncl.size();  // 1 * L
    std::vector<float> host(static_cast<std::size_t>(n));
    qtd::to_host(wav_ncl, host.data());
    wav.reserve(wav.size() + n);
    for (int i = 0; i < n; ++i) {
        float s = host[i];
        s = s < -1.0f ? -1.0f : (s > 1.0f ? 1.0f : s);
        wav.push_back(s);
    }
}

}  // namespace brosoundml
