#include "qwen_tts_codec.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brosoundml {

namespace {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: QwenTtsCodecDecoder: " + msg);
}

// ── weight upload helpers ────────────────────────────────────────────────────

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Upload a 2D-flattened tensor to a fresh CPU FP32 tensor (all codec weights
// are stored F32, so upload() performs no dtype conversion).
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols) {
    bt::Tensor t;
    sf::upload(need(f, name), rows, cols, t);
    return t;
}

// Upload a rank-1 parameter [C] as a (C,1) column.
bt::Tensor up_vec(const sf::File& f, const std::string& name, int c) {
    return up(f, name, c, 1);
}

// Upload a SnakeBeta parameter [C] and exponentiate it in place — the upstream
// SnakeBeta applies exp() to its raw alpha/beta before use.
bt::Tensor up_snake(const sf::File& f, const std::string& name, int c) {
    bt::Tensor t = up_vec(f, name, c);
    float* p = t.host_f32_mut();
    for (int i = 0; i < c; ++i) p[i] = std::exp(p[i]);
    return t;
}

// Combine an EMA codebook into its effective embedding:
//   embed[k] = embedding_sum[k] / max(cluster_usage[k], 1e-5).
bt::Tensor load_codebook(const sf::File& f, const std::string& prefix,
                         int bins, int dim) {
    bt::Tensor embed = up(f, prefix + ".embedding_sum", bins, dim);
    bt::Tensor usage = up_vec(f, prefix + ".cluster_usage", bins);
    float* e = embed.host_f32_mut();
    const float* u = usage.host_f32();
    for (int k = 0; k < bins; ++k) {
        const float d = 1.0f / (u[k] > 1e-5f ? u[k] : 1e-5f);
        float* row = e + static_cast<std::size_t>(k) * dim;
        for (int j = 0; j < dim; ++j) row[j] *= d;
    }
    return embed;
}

// ── layout converters: NCL (1, C*L) <-> SEQ (L, C) ───────────────────────────

void ncl_to_seq(const bt::Tensor& x, int C, int L, bt::Tensor& out) {
    out.resize(L, C);
    const float* s = x.host_f32();
    float* d = out.host_f32_mut();
    for (int c = 0; c < C; ++c)
        for (int l = 0; l < L; ++l) d[l * C + c] = s[static_cast<std::size_t>(c) * L + l];
}

void seq_to_ncl(const bt::Tensor& x, int L, int C, bt::Tensor& out) {
    out.resize(1, C * L);
    const float* s = x.host_f32();
    float* d = out.host_f32_mut();
    for (int l = 0; l < L; ++l)
        for (int c = 0; c < C; ++c) d[static_cast<std::size_t>(c) * L + l] = s[l * C + c];
}

// ── op wrappers ──────────────────────────────────────────────────────────────

// Stride-1 causal conv: left-pad dilation*(k-1), valid conv. Lout == L.
void causal_conv(const bt::Tensor& x_ncl, int Cin, int L, const bt::Tensor& w,
                 const bt::Tensor* b, int Cout, int k, int dilation, int groups,
                 bt::Tensor& y) {
    bt::Tensor scratch;
    bt::causal_conv1d(x_ncl, w, b, /*N=*/1, Cin, L, Cout, k, /*stride=*/1,
                      dilation, groups, scratch, y);
}

// Causal transposed conv: conv_transpose1d then trim (kernel-stride) from the
// right (left_pad is 0 for these causal upsamplers). Lout = L*stride.
void trans_conv(const bt::Tensor& x_ncl, int Cin, int L, const bt::Tensor& w,
                const bt::Tensor* b, int Cout, int k, int stride, bt::Tensor& y) {
    bt::Tensor full;
    bt::conv_transpose1d_forward(x_ncl, w, b, /*N=*/1, Cin, L, Cout, k, stride,
                                 /*padding=*/0, /*output_padding=*/0,
                                 /*dilation=*/1, full);
    const int L_full = (L - 1) * stride + k;
    const int trim   = k - stride;
    const int L_out  = L_full - trim;
    y.resize(1, Cout * L_out);
    const float* s = full.host_f32();
    float* d = y.host_f32_mut();
    for (int c = 0; c < Cout; ++c) {
        const float* src = s + static_cast<std::size_t>(c) * L_full;
        float* dst = d + static_cast<std::size_t>(c) * L_out;
        for (int l = 0; l < L_out; ++l) dst[l] = src[l];
    }
}

// SnakeBeta in place: y = x + (1/beta)*sin^2(alpha*x); alpha/beta pre-exp'd.
void snake(bt::Tensor& x_ncl, int C, int L, const bt::Tensor& alpha,
           const bt::Tensor& beta) {
    bt::snake_forward(x_ncl, alpha, &beta, /*N=*/1, C, L, x_ncl);
}

// Y(B,out) = X(B,in) @ W^T (+ bias). A null bias adds zero.
void linear(const bt::Tensor& W, const bt::Tensor* bias, const bt::Tensor& X,
            bt::Tensor& Y) {
    if (bias) {
        bt::linear_forward_batched(W, *bias, X, Y);
    } else {
        bt::Tensor zero = bt::Tensor::mat(W.rows, 1);
        bt::linear_forward_batched(W, zero, X, Y);
    }
}

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Channels-last LayerNorm in place over the C axis of a (T, C) SEQ tensor.
void layernorm_seq(bt::Tensor& x, int T, int C, const bt::Tensor& w,
                   const bt::Tensor& b, float eps) {
    float* p = x.host_f32_mut();
    const float* wp = w.host_f32();
    const float* bp = b.host_f32();
    for (int t = 0; t < T; ++t) {
        float* row = p + static_cast<std::size_t>(t) * C;
        float mean = 0.0f;
        for (int c = 0; c < C; ++c) mean += row[c];
        mean /= C;
        float var = 0.0f;
        for (int c = 0; c < C; ++c) {
            const float dlt = row[c] - mean;
            var += dlt * dlt;
        }
        var /= C;
        const float inv = 1.0f / std::sqrt(var + eps);
        for (int c = 0; c < C; ++c) row[c] = (row[c] - mean) * inv * wp[c] + bp[c];
    }
}

// hs(T,C) += scale[c] * delta(T,C)  (LayerScale residual).
void add_layerscale(bt::Tensor& hs, const bt::Tensor& delta,
                    const bt::Tensor& scale, int T, int C) {
    float* h = hs.host_f32_mut();
    const float* d = delta.host_f32();
    const float* s = scale.host_f32();
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < C; ++c) h[t * C + c] += s[c] * d[t * C + c];
}

// Rotate-half (NeoX/Llama) RoPE, in place on a (T, num_heads*head_dim) buffer.
// Pairs dim i with i+head_dim/2; position is the row index.
void rope_inplace(bt::Tensor& buf, int T, int num_heads, int head_dim,
                  float theta) {
    const int inner = num_heads * head_dim;
    const int half = head_dim / 2;
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; ++i)
        inv_freq[i] = std::pow(theta, -(2.0f * i) / head_dim);
    float* p = buf.host_f32_mut();
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < num_heads; ++h) {
            float* base = p + static_cast<std::size_t>(t) * inner + h * head_dim;
            for (int i = 0; i < half; ++i) {
                const float ang = t * inv_freq[i];
                const float cs = std::cos(ang), sn = std::sin(ang);
                const float x0 = base[i], x1 = base[i + half];
                base[i]        = x0 * cs - x1 * sn;
                base[i + half] = x1 * cs + x0 * sn;
            }
        }
    }
}

}  // namespace

// ── load ─────────────────────────────────────────────────────────────────────

void QwenTtsCodecDecoder::load(const sf::File& f, const QwenTtsCodecConfig& cfg) {
    // The decoder is a CPU FP32 path (the hand-rolled glue reads host buffers),
    // so pin every allocation here to the host even on a CUDA/Metal build.
    bt::DeviceScope cpu(bt::Device::CPU);

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
        f, P + "quantizer.rvq_first.vq.layers.0._codebook", codebook_bins, vq_dim));
    for (int l = 0; l < num_quantizers - num_semantic; ++l) {
        codebook.push_back(load_codebook(
            f, P + "quantizer.rvq_rest.vq.layers." + std::to_string(l) + "._codebook",
            codebook_bins, vq_dim));
    }
    out_proj_first = up(f, P + "quantizer.rvq_first.output_proj.weight", codebook_dim, vq_dim);
    out_proj_rest  = up(f, P + "quantizer.rvq_rest.output_proj.weight",  codebook_dim, vq_dim);

    // ── pre_conv (codebook_dim -> latent_dim, k3) ──
    pre_conv_w = up(f, P + "pre_conv.conv.weight", latent_dim, codebook_dim * 3);
    pre_conv_b = up_vec(f, P + "pre_conv.conv.bias", latent_dim);

    // ── pre_transformer ──
    in_proj_w  = up(f, P + "pre_transformer.input_proj.weight",  hidden, latent_dim);
    in_proj_b  = up_vec(f, P + "pre_transformer.input_proj.bias", hidden);
    out_proj_w = up(f, P + "pre_transformer.output_proj.weight", latent_dim, hidden);
    out_proj_b = up_vec(f, P + "pre_transformer.output_proj.bias", latent_dim);
    final_norm = up_vec(f, P + "pre_transformer.norm.weight", hidden);

    const int inner = num_heads * head_dim;
    const int inter = cfg.pre_transformer.intermediate_size;
    layers.clear();
    layers.resize(cfg.pre_transformer.num_hidden_layers);
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const std::string L = P + "pre_transformer.layers." + std::to_string(i) + ".";
        QwenTtsCodecTfLayer& tl = layers[i];
        tl.in_ln       = up_vec(f, L + "input_layernorm.weight", hidden);
        tl.post_ln     = up_vec(f, L + "post_attention_layernorm.weight", hidden);
        tl.qw          = up(f, L + "self_attn.q_proj.weight", inner, hidden);
        tl.kw          = up(f, L + "self_attn.k_proj.weight", inner, hidden);
        tl.vw          = up(f, L + "self_attn.v_proj.weight", inner, hidden);
        tl.ow          = up(f, L + "self_attn.o_proj.weight", hidden, inner);
        tl.attn_scale  = up_vec(f, L + "self_attn_layer_scale.scale", hidden);
        tl.mlp_scale   = up_vec(f, L + "mlp_layer_scale.scale", hidden);
        tl.gate        = up(f, L + "mlp.gate_proj.weight", inter, hidden);
        tl.up          = up(f, L + "mlp.up_proj.weight", inter, hidden);
        tl.down        = up(f, L + "mlp.down_proj.weight", hidden, inter);
    }

    // ── ConvNeXt upsample stages ──
    upsample.clear();
    upsample.resize(upsampling_ratios.size());
    for (std::size_t s = 0; s < upsample.size(); ++s) {
        const int factor = upsampling_ratios[s];
        const std::string U = P + "upsample." + std::to_string(s) + ".";
        QwenTtsCodecUpStage& us = upsample[s];
        us.factor  = factor;
        us.tconv_w = up(f, U + "0.conv.weight", latent_dim, latent_dim * factor);
        us.tconv_b = up_vec(f, U + "0.conv.bias", latent_dim);
        us.dw_w    = up(f, U + "1.dwconv.conv.weight", latent_dim, 7);  // depthwise
        us.dw_b    = up_vec(f, U + "1.dwconv.conv.bias", latent_dim);
        us.ln_w    = up_vec(f, U + "1.norm.weight", latent_dim);
        us.ln_b    = up_vec(f, U + "1.norm.bias", latent_dim);
        us.pw1_w   = up(f, U + "1.pwconv1.weight", 4 * latent_dim, latent_dim);
        us.pw1_b   = up_vec(f, U + "1.pwconv1.bias", 4 * latent_dim);
        us.pw2_w   = up(f, U + "1.pwconv2.weight", latent_dim, 4 * latent_dim);
        us.pw2_b   = up_vec(f, U + "1.pwconv2.bias", latent_dim);
        us.gamma   = up_vec(f, U + "1.gamma", latent_dim);
    }

    // ── SEANet decoder ──
    dec0_w = up(f, P + "decoder.0.conv.weight", decoder_dim, latent_dim * 7);
    dec0_b = up_vec(f, P + "decoder.0.conv.bias", decoder_dim);

    static const int kResDilations[3] = {1, 3, 9};
    blocks.clear();
    blocks.resize(upsample_rates.size());
    for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
        QwenTtsCodecDecBlock& blk = blocks[bi];
        blk.in_dim  = decoder_dim >> bi;
        blk.out_dim = decoder_dim >> (bi + 1);
        blk.rate    = upsample_rates[bi];
        const std::string B = P + "decoder." + std::to_string(bi + 1) + ".block.";
        blk.s_alpha = up_snake(f, B + "0.alpha", blk.in_dim);
        blk.s_beta  = up_snake(f, B + "0.beta",  blk.in_dim);
        blk.tconv_w = up(f, B + "1.conv.weight", blk.in_dim, blk.out_dim * 2 * blk.rate);
        blk.tconv_b = up_vec(f, B + "1.conv.bias", blk.out_dim);
        blk.units.resize(3);
        for (int ui = 0; ui < 3; ++ui) {
            QwenTtsCodecResUnit& u = blk.units[ui];
            const std::string R = B + std::to_string(2 + ui) + ".";
            u.dim      = blk.out_dim;
            u.dilation = kResDilations[ui];
            u.a1_alpha = up_snake(f, R + "act1.alpha", u.dim);
            u.a1_beta  = up_snake(f, R + "act1.beta",  u.dim);
            u.a2_alpha = up_snake(f, R + "act2.alpha", u.dim);
            u.a2_beta  = up_snake(f, R + "act2.beta",  u.dim);
            u.c1w      = up(f, R + "conv1.conv.weight", u.dim, u.dim * 7);
            u.c1b      = up_vec(f, R + "conv1.conv.bias", u.dim);
            u.c2w      = up(f, R + "conv2.conv.weight", u.dim, u.dim * 1);
            u.c2b      = up_vec(f, R + "conv2.conv.bias", u.dim);
        }
    }

    const int out_dim = decoder_dim >> upsample_rates.size();
    dec_final_alpha = up_snake(f, P + "decoder.5.alpha", out_dim);
    dec_final_beta  = up_snake(f, P + "decoder.5.beta",  out_dim);
    dec_out_w = up(f, P + "decoder.6.conv.weight", 1, out_dim * 7);
    dec_out_b = up_vec(f, P + "decoder.6.conv.bias", 1);
}

// ── decode ───────────────────────────────────────────────────────────────────

void QwenTtsCodecDecoder::decode(const int32_t* codes, int K, int T,
                                 std::vector<float>& wav) const {
    bt::DeviceScope cpu(bt::Device::CPU);  // CPU FP32 path (see load()).
    if (K != num_quantizers)
        fail("expected " + std::to_string(num_quantizers) + " codebooks, got " +
             std::to_string(K));
    if (T <= 0) return;

    const int bins = codebook_bins;
    auto gather_group = [&](int first, int count, const bt::Tensor& proj,
                            bt::Tensor& out_TC) {
        // Sum the codebook lookups of `count` consecutive code rows, then
        // project vq_dim -> codebook_dim.
        bt::Tensor acc = bt::Tensor::mat(T, vq_dim);  // zero-filled
        float* a = acc.host_f32_mut();
        for (int l = 0; l < count; ++l) {
            const bt::Tensor& cb = codebook[first + l];
            const float* table = cb.host_f32();
            const int32_t* row = codes + static_cast<std::size_t>(first + l) * T;
            for (int t = 0; t < T; ++t) {
                int idx = row[t];
                if (idx < 0) idx = 0;
                else if (idx >= bins) idx = bins - 1;
                const float* e = table + static_cast<std::size_t>(idx) * vq_dim;
                float* dst = a + static_cast<std::size_t>(t) * vq_dim;
                for (int d = 0; d < vq_dim; ++d) dst[d] += e[d];
            }
        }
        linear(proj, nullptr, acc, out_TC);  // (T, codebook_dim)
    };

    // ── quantizer.decode: semantic + acoustic groups summed ──
    bt::Tensor y_first, y_rest;
    gather_group(0, num_semantic, out_proj_first, y_first);
    gather_group(num_semantic, num_quantizers - num_semantic, out_proj_rest, y_rest);
    bt::Tensor latent_seq = bt::Tensor::mat(T, codebook_dim);
    {
        float* o = latent_seq.host_f32_mut();
        const float* a = y_first.host_f32();
        const float* b = y_rest.host_f32();
        for (int i = 0; i < T * codebook_dim; ++i) o[i] = a[i] + b[i];
    }

    // ── pre_conv (codebook_dim -> latent_dim, causal k3) ──
    bt::Tensor ncl;
    seq_to_ncl(latent_seq, T, codebook_dim, ncl);
    bt::Tensor conv_out;
    causal_conv(ncl, codebook_dim, T, pre_conv_w, &pre_conv_b, latent_dim, 3, 1, 1,
                conv_out);

    // ── pre_transformer ──
    bt::Tensor hs;  // (T, hidden)
    {
        bt::Tensor lat_seq;
        ncl_to_seq(conv_out, latent_dim, T, lat_seq);  // (T, latent_dim)
        linear(in_proj_w, &in_proj_b, lat_seq, hs);    // (T, hidden)
    }
    const int inner = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(T);
    for (const QwenTtsCodecTfLayer& tl : layers) {
        // self-attention
        bt::Tensor normed;
        bt::rms_norm_forward(hs, tl.in_ln, rms_eps, normed);
        bt::Tensor q, k, v;
        linear(tl.qw, nullptr, normed, q);
        linear(tl.kw, nullptr, normed, k);
        linear(tl.vw, nullptr, normed, v);
        rope_inplace(q, T, num_heads, head_dim, rope_theta);
        rope_inplace(k, T, num_heads, head_dim, rope_theta);
        bt::Tensor ctx = bt::Tensor::mat(T, inner);
        const float* qp = q.host_f32();
        const float* kp = k.host_f32();
        const float* vp = v.host_f32();
        float* cp = ctx.host_f32_mut();
        for (int h = 0; h < num_heads; ++h) {
            const int off = h * head_dim;
            for (int i = 0; i < T; ++i) {
                const int jlo = (i - sliding_window + 1 > 0) ? i - sliding_window + 1 : 0;
                const float* qi = qp + static_cast<std::size_t>(i) * inner + off;
                float maxs = -1e30f;
                for (int j = jlo; j <= i; ++j) {
                    const float* kj = kp + static_cast<std::size_t>(j) * inner + off;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
                    dot *= scale;
                    scores[j] = dot;
                    if (dot > maxs) maxs = dot;
                }
                float sum = 0.0f;
                for (int j = jlo; j <= i; ++j) {
                    const float e = std::exp(scores[j] - maxs);
                    scores[j] = e;
                    sum += e;
                }
                const float invsum = 1.0f / sum;
                float* ci = cp + static_cast<std::size_t>(i) * inner + off;
                for (int d = 0; d < head_dim; ++d) ci[d] = 0.0f;
                for (int j = jlo; j <= i; ++j) {
                    const float w = scores[j] * invsum;
                    const float* vj = vp + static_cast<std::size_t>(j) * inner + off;
                    for (int d = 0; d < head_dim; ++d) ci[d] += w * vj[d];
                }
            }
        }
        bt::Tensor attn;
        linear(tl.ow, nullptr, ctx, attn);  // (T, hidden)
        add_layerscale(hs, attn, tl.attn_scale, T, hidden);

        // SwiGLU MLP
        bt::Tensor n2;
        bt::rms_norm_forward(hs, tl.post_ln, rms_eps, n2);
        bt::Tensor g, u;
        linear(tl.gate, nullptr, n2, g);
        linear(tl.up, nullptr, n2, u);
        float* gp = g.host_f32_mut();
        const float* upp = u.host_f32();
        for (int i = 0; i < g.size(); ++i) gp[i] = silu(gp[i]) * upp[i];
        bt::Tensor dn;
        linear(tl.down, nullptr, g, dn);  // (T, hidden)
        add_layerscale(hs, dn, tl.mlp_scale, T, hidden);
    }
    {
        bt::Tensor normed;
        bt::rms_norm_forward(hs, final_norm, rms_eps, normed);
        bt::Tensor lat;
        linear(out_proj_w, &out_proj_b, normed, lat);  // (T, latent_dim)
        seq_to_ncl(lat, T, latent_dim, ncl);           // -> (1, latent_dim*T)
    }

    // ── ConvNeXt upsample (x2 x2) ──
    int C = latent_dim;
    int L = T;
    for (const QwenTtsCodecUpStage& us : upsample) {
        bt::Tensor up_ncl;
        trans_conv(ncl, C, L, us.tconv_w, &us.tconv_b, C, us.factor, us.factor, up_ncl);
        L *= us.factor;
        ncl = std::move(up_ncl);
        // ConvNeXt block (depthwise k7, LN, pw 1->4->1 GELU, gamma, residual).
        bt::Tensor dw;
        causal_conv(ncl, C, L, us.dw_w, &us.dw_b, C, 7, 1, /*groups=*/C, dw);
        bt::Tensor seq;
        ncl_to_seq(dw, C, L, seq);  // (L, C)
        layernorm_seq(seq, L, C, us.ln_w, us.ln_b, 1e-6f);
        bt::Tensor h1;
        linear(us.pw1_w, &us.pw1_b, seq, h1);  // (L, 4C)
        float* h1p = h1.host_f32_mut();
        for (int i = 0; i < h1.size(); ++i) h1p[i] = gelu(h1p[i]);
        bt::Tensor h2;
        linear(us.pw2_w, &us.pw2_b, h1, h2);  // (L, C)
        float* h2p = h2.host_f32_mut();
        const float* gam = us.gamma.host_f32();
        for (int l = 0; l < L; ++l)
            for (int c = 0; c < C; ++c) h2p[l * C + c] *= gam[c];
        bt::Tensor back;
        seq_to_ncl(h2, L, C, back);  // (1, C*L)
        float* xp = ncl.host_f32_mut();
        const float* bp = back.host_f32();
        for (int i = 0; i < C * L; ++i) xp[i] += bp[i];
    }

    // ── SEANet decoder ──
    {
        bt::Tensor d0;
        causal_conv(ncl, latent_dim, L, dec0_w, &dec0_b, decoder_dim, 7, 1, 1, d0);
        ncl = std::move(d0);
        C = decoder_dim;
    }
    for (const QwenTtsCodecDecBlock& blk : blocks) {
        snake(ncl, blk.in_dim, L, blk.s_alpha, blk.s_beta);
        bt::Tensor up_ncl;
        trans_conv(ncl, blk.in_dim, L, blk.tconv_w, &blk.tconv_b, blk.out_dim,
                   2 * blk.rate, blk.rate, up_ncl);
        L *= blk.rate;
        C = blk.out_dim;
        ncl = std::move(up_ncl);
        for (const QwenTtsCodecResUnit& u : blk.units) {
            bt::Tensor res = ncl;  // deep copy
            snake(ncl, u.dim, L, u.a1_alpha, u.a1_beta);
            bt::Tensor t1;
            causal_conv(ncl, u.dim, L, u.c1w, &u.c1b, u.dim, 7, u.dilation, 1, t1);
            snake(t1, u.dim, L, u.a2_alpha, u.a2_beta);
            bt::Tensor t2;
            causal_conv(t1, u.dim, L, u.c2w, &u.c2b, u.dim, 1, 1, 1, t2);
            float* xp = ncl.host_f32_mut();
            const float* rp = res.host_f32();
            const float* tp = t2.host_f32();
            for (int i = 0; i < u.dim * L; ++i) xp[i] = rp[i] + tp[i];
        }
    }
    snake(ncl, C, L, dec_final_alpha, dec_final_beta);
    bt::Tensor wav_ncl;
    causal_conv(ncl, C, L, dec_out_w, &dec_out_b, 1, 7, 1, 1, wav_ncl);

    // ── clamp [-1, 1] and append ──
    const float* w = wav_ncl.host_f32();
    const int n = wav_ncl.size();  // 1 * L
    wav.reserve(wav.size() + n);
    for (int i = 0; i < n; ++i) {
        float s = w[i];
        s = s < -1.0f ? -1.0f : (s > 1.0f ? 1.0f : s);
        wav.push_back(s);
    }
}

}  // namespace brosoundml
