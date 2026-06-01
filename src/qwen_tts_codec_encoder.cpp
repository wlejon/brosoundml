#include "qwen_tts_codec_encoder.h"

#include "qwen_tts_codec_common.h"
#include "qwen_tts_device.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;
using namespace qcodec;   // shared codec glue: up/up_vec/load_codebook,
                          // ncl<->seq, causal/strided causal conv, add_layerscale,
                          // codec_attn, fail/need

namespace {

// ELU (EnCodec/SEANet activation) over an NCL buffer; shape preserved.
bt::Tensor elu(const bt::Tensor& x) {
    bt::Tensor y;
    bt::elu_forward(x, y);
    return y;
}

// LayerNorm (weight + bias) over a (T, D) sequence.
bt::Tensor lnorm(const bt::Tensor& x, const bt::Tensor& w, const bt::Tensor& b, float eps) {
    bt::Tensor y;
    bt::layernorm_forward_inference_batched(x, w, b, y, eps);
    return y;
}

// Copy a length-`n` INT32 device tensor into rows[row*T .. row*T+n) of `dst`.
void write_idx_row(const bt::Tensor& idx, int n, int row, int T,
                   std::vector<std::int32_t>& dst) {
    if (idx.device == bt::Device::CPU) {
        const auto* p = static_cast<const std::int32_t*>(idx.data);
        for (int t = 0; t < n; ++t) dst[static_cast<std::size_t>(row) * T + t] = p[t];
    } else {
        bt::Tensor host = idx.to(bt::Device::CPU);
        const auto* p = static_cast<const std::int32_t*>(host.data);
        for (int t = 0; t < n; ++t) dst[static_cast<std::size_t>(row) * T + t] = p[t];
    }
}

}  // namespace

void QwenTtsCodecEncoder::load(const sf::File& f, const QwenTtsCodecConfig& cfg,
                               bt::Device dev) {
    const QwenTtsCodecEncoderConfig& ec = cfg.encoder;

    num_filters    = ec.num_filters;
    model_dim      = ec.transformer.hidden_size;
    codebook_dim   = ec.codebook_dim;
    codebook_bins  = cfg.codebook_size;
    kernel         = ec.kernel_size;
    last_kernel    = ec.last_kernel_size;
    num_heads      = ec.transformer.num_attention_heads;
    head_dim       = ec.transformer.head_dim;
    sliding_window = ec.sliding_window;
    ln_eps         = ec.transformer.rms_norm_eps;   // reused as the LayerNorm eps
    rope_theta     = ec.transformer.rope_theta;
    n_semantic     = cfg.num_semantic_quantizers;
    n_acoustic     = ec.valid_num_quantizers - n_semantic;
    downsample_rate = cfg.encode_downsample_rate;

    const int residual_k = ec.residual_kernel_size;
    const int compress   = ec.compress;

    // SEANet strides are the decode-order ratios applied in reverse.
    std::vector<int> strides(ec.ratios.rbegin(), ec.ratios.rend());
    int prod_ratios = 1;
    for (int r : strides) prod_ratios *= r;
    downsample_stride = (prod_ratios > 0) ? downsample_rate / prod_ratios : 0;

    const std::string P = "encoder.";

    // ── SEANet input conv (1 -> num_filters, k) ──
    conv_in_w = up(f, P + "encoder.layers.0.conv.weight", num_filters, 1 * kernel, dev);
    conv_in_b = up_vec(f, P + "encoder.layers.0.conv.bias", num_filters, dev);

    // ── downsample blocks: residual unit at layer (1+3i), strided conv at (3+3i) ──
    blocks.clear();
    blocks.resize(strides.size());
    for (std::size_t i = 0; i < strides.size(); ++i) {
        QwenTtsEncBlock& blk = blocks[i];
        blk.in_dim  = num_filters << i;          // 64, 128, 256, 512
        blk.out_dim = num_filters << (i + 1);    // 128, 256, 512, 1024
        blk.stride  = strides[i];

        const int res_layer  = 1 + 3 * static_cast<int>(i);
        const int down_layer = 3 + 3 * static_cast<int>(i);
        const std::string RU = P + "encoder.layers." + std::to_string(res_layer) + ".block.";
        const std::string DN = P + "encoder.layers." + std::to_string(down_layer) + ".conv.";

        QwenTtsEncResUnit& u = blk.unit;
        u.dim    = blk.in_dim;
        u.hidden = blk.in_dim / compress;
        u.kernel = residual_k;
        u.c1w = up(f, RU + "1.conv.weight", u.hidden, u.dim * residual_k, dev);
        u.c1b = up_vec(f, RU + "1.conv.bias", u.hidden, dev);
        u.c2w = up(f, RU + "3.conv.weight", u.dim, u.hidden * 1, dev);
        u.c2b = up_vec(f, RU + "3.conv.bias", u.dim, dev);

        blk.down_w = up(f, DN + "weight", blk.out_dim, blk.in_dim * (2 * blk.stride), dev);
        blk.down_b = up_vec(f, DN + "bias", blk.out_dim, dev);
    }

    // ── SEANet final conv (last_dim -> model_dim, last_kernel) ──
    last_dim = num_filters << strides.size();    // 1024
    const int last_layer = 2 + 3 * static_cast<int>(strides.size());   // 14
    conv_out_w = up(f, P + "encoder.layers." + std::to_string(last_layer) + ".conv.weight",
                    model_dim, last_dim * last_kernel, dev);
    conv_out_b = up_vec(f, P + "encoder.layers." + std::to_string(last_layer) + ".conv.bias",
                        model_dim, dev);

    // ── encoder_transformer (LayerNorm + GELU MLP + LayerScale, RoPE) ──
    // HF rotate-half -> brotensor adjacent-pair RoPE: permute q/k rows at load.
    const std::vector<std::int32_t> q_perm =
        qtd::per_head_perm_rows(qtd::rotate_half_perm(head_dim), num_heads, head_dim);
    const int inner = num_heads * head_dim;
    const int inter = ec.transformer.intermediate_size;
    layers.clear();
    layers.resize(ec.transformer.num_hidden_layers);
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const std::string L = P + "encoder_transformer.layers." + std::to_string(i) + ".";
        QwenTtsEncTfLayer& tl = layers[i];
        tl.in_ln_w    = up_vec(f, L + "input_layernorm.weight", model_dim, dev);
        tl.in_ln_b    = up_vec(f, L + "input_layernorm.bias",   model_dim, dev);
        tl.post_ln_w  = up_vec(f, L + "post_attention_layernorm.weight", model_dim, dev);
        tl.post_ln_b  = up_vec(f, L + "post_attention_layernorm.bias",   model_dim, dev);
        tl.qw         = qtd::gather_rows(up(f, L + "self_attn.q_proj.weight", inner, model_dim, dev), q_perm);
        tl.kw         = qtd::gather_rows(up(f, L + "self_attn.k_proj.weight", inner, model_dim, dev), q_perm);
        tl.vw         = up(f, L + "self_attn.v_proj.weight", inner, model_dim, dev);
        tl.ow         = up(f, L + "self_attn.o_proj.weight", model_dim, inner, dev);
        tl.attn_scale = up_vec(f, L + "self_attn_layer_scale.scale", model_dim, dev);
        tl.mlp_scale  = up_vec(f, L + "mlp_layer_scale.scale", model_dim, dev);
        tl.fc1_w      = up(f, L + "mlp.fc1.weight", inter, model_dim, dev);
        tl.fc2_w      = up(f, L + "mlp.fc2.weight", model_dim, inter, dev);
    }

    // ── downsample conv (model_dim -> model_dim, k = 2*stride), no bias ──
    down_w = up(f, P + "downsample.conv.weight", model_dim,
                model_dim * (2 * downsample_stride), dev);

    // ── quantizer (split RVQ): input_proj + euclidean codebooks per group ──
    auto load_group = [&](QwenTtsEncQuantGroup& g, const std::string& which, int n_layers) {
        const std::string Q = P + "quantizer." + which + "_residual_vector_quantizer.";
        g.input_proj = up(f, Q + "input_proj.weight", codebook_dim, model_dim, dev);
        g.codebook.clear();
        for (int l = 0; l < n_layers; ++l) {
            g.codebook.push_back(load_codebook(
                f, Q + "layers." + std::to_string(l) + ".codebook",
                codebook_bins, codebook_dim, dev, /*sum_name=*/"embed_sum"));
        }
    };
    load_group(semantic, "semantic", n_semantic);
    load_group(acoustic, "acoustic", n_acoustic);
}

int QwenTtsCodecEncoder::encode(const float* wav, int n_samples,
                                std::vector<std::int32_t>& codes) const {
    const bt::Device dev = conv_in_w.device;
    bt::DeviceScope scope(dev);

    if (downsample_rate <= 0) fail("encoder not loaded");
    if (n_samples <= 0) { codes.clear(); return 0; }

    // Right-pad the waveform to a whole frame so every SEANet stride divides
    // evenly (EnCodec's extra causal padding is then zero) and T is exact.
    const int L = ((n_samples + downsample_rate - 1) / downsample_rate) * downsample_rate;
    const int T = L / downsample_rate;
    std::vector<float> buf(static_cast<std::size_t>(L), 0.0f);
    std::copy(wav, wav + n_samples, buf.begin());
    bt::Tensor ncl = bt::Tensor::from_host_on(dev, buf.data(), 1, L);   // (1, 1*L)

    // ── SEANet: input conv, per-ratio [resunit, ELU, strided conv], final ──
    ncl = causal_conv(ncl, /*Cin=*/1, L, conv_in_w, &conv_in_b, num_filters, kernel, 1, 1);
    int C = num_filters, Lc = L;
    for (const QwenTtsEncBlock& blk : blocks) {
        const QwenTtsEncResUnit& u = blk.unit;
        bt::Tensor res = ncl;                                  // residual (deep copy)
        bt::Tensor h = elu(ncl);
        h = causal_conv(h, u.dim, Lc, u.c1w, &u.c1b, u.hidden, u.kernel, 1, 1);
        h = elu(h);
        h = causal_conv(h, u.hidden, Lc, u.c2w, &u.c2b, u.dim, 1, 1, 1);
        bt::add_inplace(res, h);
        ncl = elu(res);
        ncl = strided_causal_conv(ncl, blk.in_dim, Lc, blk.down_w, &blk.down_b,
                                  blk.out_dim, 2 * blk.stride, blk.stride);
        Lc /= blk.stride;
        C = blk.out_dim;
    }
    ncl = elu(ncl);
    ncl = causal_conv(ncl, C, Lc, conv_out_w, &conv_out_b, model_dim, last_kernel, 1, 1);
    C = model_dim;   // Lc is now the 25 Hz length (L / prod(strides))

    // ── encoder_transformer (causal windowed, RoPE, LayerNorm, GELU MLP) ──
    bt::Tensor seq = ncl_to_seq(ncl, model_dim, Lc);   // (Lc, model_dim)
    {
        const int half = head_dim / 2;
        std::vector<int>   pos_grid(static_cast<std::size_t>(Lc) * half);
        std::vector<float> inv_freq(half);
        for (int i = 0; i < half; ++i) inv_freq[i] = std::pow(rope_theta, -(2.0f * i) / head_dim);
        for (int t = 0; t < Lc; ++t)
            for (int i = 0; i < half; ++i)
                pos_grid[static_cast<std::size_t>(t) * half + i] = t;
        bt::Tensor cosT, sinT;
        qtd::build_rope_tables(dev, Lc, half, pos_grid, inv_freq, cosT, sinT);

        for (const QwenTtsEncTfLayer& tl : layers) {
            bt::Tensor normed = lnorm(seq, tl.in_ln_w, tl.in_ln_b, ln_eps);
            bt::Tensor q, k, v;
            qtd::linear(tl.qw, nullptr, normed, q);
            qtd::linear(tl.kw, nullptr, normed, k);
            qtd::linear(tl.vw, nullptr, normed, v);
            bt::Tensor qr, kr;
            bt::rope_apply(q, cosT, sinT, head_dim, num_heads, qr);
            bt::rope_apply(k, cosT, sinT, head_dim, num_heads, kr);
            // Upstream MimiTransformerModel builds its mask with create_causal_mask
            // (not the sliding variant) and layer_types is unset, so the encoder
            // transformer is plain *unbounded* causal — the config's sliding_window
            // (250) never applies here (unlike the decoder's own transformer).
            bt::Tensor ctx = codec_attn(qr, kr, v, num_heads, /*window=*/0);
            bt::Tensor attn;
            qtd::linear(tl.ow, nullptr, ctx, attn);
            add_layerscale(seq, attn, tl.attn_scale);

            bt::Tensor n2 = lnorm(seq, tl.post_ln_w, tl.post_ln_b, ln_eps);
            bt::Tensor h1;
            qtd::linear(tl.fc1_w, nullptr, n2, h1);
            bt::gelu_exact_forward(h1, h1);
            bt::Tensor h2;
            qtd::linear(tl.fc2_w, nullptr, h1, h2);
            add_layerscale(seq, h2, tl.mlp_scale);
        }
    }

    // ── downsample /stride to 12.5 Hz ──
    ncl = seq_to_ncl(seq, Lc, model_dim);
    ncl = strided_causal_conv(ncl, model_dim, Lc, down_w, /*bias=*/nullptr, model_dim,
                              2 * downsample_stride, downsample_stride);
    Lc /= downsample_stride;   // == T

    // ── split-RVQ encode: project, then nearest-codebook with residual update ──
    bt::Tensor zseq = ncl_to_seq(ncl, model_dim, T);   // (T, model_dim)
    const int K = n_semantic + n_acoustic;
    codes.assign(static_cast<std::size_t>(K) * T, 0);

    auto encode_group = [&](const QwenTtsEncQuantGroup& g, int k_offset) {
        bt::Tensor residual;
        qtd::linear(g.input_proj, nullptr, zseq, residual);   // (T, codebook_dim)
        for (std::size_t l = 0; l < g.codebook.size(); ++l) {
            bt::Tensor idx, quant;
            bt::vq_encode_forward(residual, g.codebook[l], idx, quant);   // (T,1) idx, (T,cb) quant
            write_idx_row(idx, T, k_offset + static_cast<int>(l), T, codes);
            bt::scale_inplace(quant, -1.0f);
            bt::add_inplace(residual, quant);   // residual -= quantized
        }
    };
    encode_group(semantic, 0);
    encode_group(acoustic, n_semantic);

    return T;
}

}  // namespace brosoundml
