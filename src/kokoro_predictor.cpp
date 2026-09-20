#include "kokoro_internal.h"

namespace brosoundml {

// ─── Style-conditioned norms ───────────────────────────────────────────────


// Compute (gamma, beta) = chunk(fc(style)) for an AdaLayerNorm / AdaIN1d.
// Returns two length-C device tensors on the style's device — shape (C, 1) so
// they slot directly into brotensor::modulate / ada_in_1d as the scale/shift
// vectors. (fc.forward_batched yields a (1, 2*C) tensor; we split it via two
// copy_d2d into freshly allocated (C, 1) buffers.)
void compute_style_affine(const Linear& fc, int C,
                          const bt::Tensor& style,
                          bt::Tensor& gamma, bt::Tensor& beta) {
    bt::Tensor scratch = bt::Tensor::empty_on(style.device, 0, 0, bt::Dtype::FP32);
    bt::linear_forward_batched(fc.W, fc.b, style, scratch);  // (1, 2*C) on style.device
    gamma = bt::Tensor::zeros_on(scratch.device, C, 1, bt::Dtype::FP32);
    beta  = bt::Tensor::zeros_on(scratch.device, C, 1, bt::Dtype::FP32);
    bt::copy_d2d(scratch, 0, gamma, 0, C);
    bt::copy_d2d(scratch, C, beta,  0, C);
}

// AdaLayerNorm forward on (L, C):
//   y = (1 + gamma) * LayerNorm_no_affine(x_row, C) + beta
// layernorm_forward_inference_batched's gamma/beta are already the per-feature
// affine, so the style affine rides the norm op directly: pass (1 + gamma) as
// the LayerNorm scale — one op instead of norm + modulate.
void ada_layernorm(const AdaLayerNormWeights& w, int L,
                   const bt::Tensor& x_lc, const bt::Tensor& style,
                   bt::Tensor& y_lc) {
    const int C = w.channels;
    const bt::Device dev = x_lc.device;
    (void)L;

    bt::Tensor gamma, beta;
    compute_style_affine(w.fc, C, style, gamma, beta);
    bt::add_scalar_inplace(gamma, 1.0f);   // LayerNorm scale = 1 + gamma

    if (y_lc.device != dev) {
        y_lc = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    }
    bt::layernorm_forward_inference_batched(x_lc, gamma, beta, y_lc, w.eps);
}

// AdaIN1d forward on (1, C*L) NCL:
//   per channel c: y[n,c,l] = (1 + gamma[c]) * (x[n,c,l] - mean_c)/std_c + beta[c]
//   where mean_c / std_c are taken over the L axis (instance norm).
// Instance norm = GroupNorm with num_groups == C, and group_norm_forward's
// gamma/beta are already the per-channel affine — so the style affine rides
// the norm op directly: pass (1 + gamma) as the GroupNorm scale. One fused
// pass instead of the earlier norm ▸ nchw_to_sequence ▸ modulate ▸
// sequence_to_nchw chain (two full-tensor transposes and a unit-gamma H2D
// upload per call, in the generator's hottest loop).
void ada_in_1d_styled(const AdaIN1dWeights& w, int N, int C, int L,
                      const bt::Tensor& x_ncl, const bt::Tensor& style,
                      bt::Tensor& y_ncl) {
    const bt::Device dev = x_ncl.device;

    bt::Tensor gamma, beta;
    compute_style_affine(w.fc, C, style, gamma, beta);
    bt::add_scalar_inplace(gamma, 1.0f);   // GroupNorm scale = 1 + gamma

    if (y_ncl.device != dev) {
        y_ncl = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    }
    bt::group_norm_forward(x_ncl, gamma, beta,
                           N, C, /*H=*/1, /*W=*/L,
                           /*num_groups=*/C, w.eps, y_ncl);
}

// Per-(L) leaky_relu on NCL.
void leaky_relu_ncl(bt::Tensor& y, float slope) {
    bt::Tensor tmp = bt::Tensor::empty_on(y.device, 0, 0, bt::Dtype::FP32);
    bt::leaky_relu_forward(y, slope, tmp);
    y = std::move(tmp);
}

// Nearest-neighbour 2x upsample along L: (1, C*L_in) NCL -> (1, C*(2*L_in)) NCL.
// Composed device-side as a depthwise ConvTranspose1d with k=2, stride=2 and
// an all-ones weight — each input element scatters identically into output
// positions 2l and 2l+1, which is exactly the nearest-neighbour duplication.
// No host round-trip (the earlier composition downloaded x, gathered on host,
// and re-uploaded — a device sync stall per call inside the F0/N and decoder
// stacks).
void upsample_nearest_2x_ncl(const bt::Tensor& x, int N, int C, int L_in,
                             bt::Tensor& y) {
    const bt::Device dev = x.device;
    bt::Tensor ones = bt::Tensor::zeros_on(dev, C, 2, bt::Dtype::FP32);
    bt::add_scalar_inplace(ones, 1.0f);
    bt::conv_transpose1d_forward(x, ones, /*bias=*/nullptr,
                                 N, /*C_in=*/C, /*L=*/L_in,
                                 /*C_out=*/C, /*kL=*/2,
                                 /*stride=*/2, /*padding=*/0,
                                 /*output_padding=*/0, /*dilation=*/1,
                                 /*groups=*/C, y);
}

// AdainResBlk1d forward: residual + shortcut, both / sqrt(2). x in NCL with
// N=1; L_out = upsample ? 2*L_in : L_in. dropout is no-op at inference.
void adain_resblk_1d_forward(const AdainResBlk1dWeights& w,
                             const bt::Tensor& x, int L_in,
                             const bt::Tensor& style,
                             int& L_out, bt::Tensor& y) {
    const int C_in  = w.channels_in;
    const int C_out = w.channels_out;
    L_out = w.upsample ? 2 * L_in : L_in;

    // ─── residual path ────────────────────────────────────────────────────
    const bt::Device dev = x.device;
    bt::Tensor r = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    ada_in_1d_styled(w.norm1, 1, C_in, L_in, x, style, r);
    leaky_relu_ncl(r, 0.2f);

    bt::Tensor pooled = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    if (w.upsample) {
        // Depthwise ConvTranspose1d: groups = C_in = C_out (when depthwise on dim_in).
        // Weight layout (C_in, (C_out_per_group=1) * kL=3) = (C_in, 3).
        bt::conv_transpose1d_forward(r, w.pool_W, &w.pool_b,
                                     /*N=*/1, /*C_in=*/C_in, /*L=*/L_in,
                                     /*C_out=*/C_in, /*kL=*/3,
                                     /*stride=*/2, /*padding=*/1,
                                     /*output_padding=*/1, /*dilation=*/1,
                                     /*groups=*/C_in, pooled);
    } else {
        pooled = std::move(r);
    }

    bt::Tensor conv1_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    w.conv1.forward(pooled, /*N=*/1, /*L=*/L_out, conv1_out);

    bt::Tensor n2 = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    ada_in_1d_styled(w.norm2, 1, C_out, L_out, conv1_out, style, n2);
    leaky_relu_ncl(n2, 0.2f);

    bt::Tensor conv2_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    w.conv2.forward(n2, /*N=*/1, /*L=*/L_out, conv2_out);

    // ─── shortcut path ────────────────────────────────────────────────────
    bt::Tensor short_x = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    if (w.upsample) {
        upsample_nearest_2x_ncl(x, 1, C_in, L_in, short_x);
    } else {
        short_x = x;
    }
    if (w.learned_sc) {
        bt::Tensor sc = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        w.conv1x1.forward(short_x, /*N=*/1, /*L=*/L_out, sc);
        short_x = std::move(sc);
    }

    // ─── combine: y = (residual + shortcut) / sqrt(2) ─────────────────────
    bt::add_inplace(conv2_out, short_x);
    bt::scale_inplace(conv2_out, 1.0f / std::sqrt(2.0f));
    y = std::move(conv2_out);
}

// Load an AdaIN1d weight set from `<prefix>.fc.{weight,bias}`.
void load_ada_in_1d(const stf::File& f, const std::string& prefix,
                    int channels, int style_dim, AdaIN1dWeights& w,
                    const std::string& where) {
    w.channels  = channels;
    w.style_dim = style_dim;
    w.fc.in_features  = style_dim;
    w.fc.out_features = 2 * channels;
    upload(f, prefix + ".fc.weight", 2 * channels, style_dim, w.fc.W, where);
    upload(f, prefix + ".fc.bias",   2 * channels, 1,         w.fc.b, where);
}

void load_ada_layernorm(const stf::File& f, const std::string& prefix,
                        int channels, int style_dim, AdaLayerNormWeights& w,
                        const std::string& where) {
    w.channels  = channels;
    w.style_dim = style_dim;
    w.fc.in_features  = style_dim;
    w.fc.out_features = 2 * channels;
    upload(f, prefix + ".fc.weight", 2 * channels, style_dim, w.fc.W, where);
    upload(f, prefix + ".fc.bias",   2 * channels, 1,         w.fc.b, where);
}

void load_conv1d(const stf::File& f, const std::string& prefix,
                 int C_out, int C_in, int kL, bool has_bias,
                 Conv1d& c, const std::string& where) {
    c.in_channels  = C_in;
    c.out_channels = C_out;
    c.kernel_size  = kL;
    c.padding      = (kL == 3) ? 1 : 0;   // matches AdainResBlk1d's hardcoded paddings
    c.stride       = 1;
    c.dilation     = 1;
    c.groups       = 1;
    upload(f, prefix + ".weight", C_out, C_in * kL, c.W, where);
    if (has_bias) {
        upload(f, prefix + ".bias", C_out, 1, c.b, where);
    }
}

void load_adain_resblk(const stf::File& f, const std::string& prefix,
                       int dim_in, int dim_out, int style_dim, bool upsample,
                       AdainResBlk1dWeights& w, const std::string& where) {
    w.channels_in  = dim_in;
    w.channels_out = dim_out;
    w.upsample     = upsample;
    w.learned_sc   = (dim_in != dim_out);

    load_ada_in_1d(f, prefix + ".norm1", dim_in,  style_dim, w.norm1, where);
    load_ada_in_1d(f, prefix + ".norm2", dim_out, style_dim, w.norm2, where);
    load_conv1d   (f, prefix + ".conv1", dim_out, dim_in,  3, true, w.conv1, where);
    load_conv1d   (f, prefix + ".conv2", dim_out, dim_out, 3, true, w.conv2, where);
    if (w.learned_sc) {
        load_conv1d(f, prefix + ".conv1x1", dim_out, dim_in, 1, false, w.conv1x1, where);
    }
    if (upsample) {
        // Depthwise ConvTranspose1d: weight shape (C_in, 1, 3) -> flatten to (C_in, 3).
        upload(f, prefix + ".pool.weight", dim_in, 3,  w.pool_W, where);
        upload(f, prefix + ".pool.bias",   dim_in, 1,  w.pool_b, where);
    }
}

// Transpose (1, C*L) NCL <-> (L, C) NLC via brotensor's NCHW<->sequence ops
// (H=1, W=L). Both dispatched on the input's device.
void ncl_to_lc(const bt::Tensor& x_ncl, int C, int L, bt::Tensor& x_lc) {
    bt::nchw_to_sequence(x_ncl, /*N=*/1, C, /*H=*/1, /*W=*/L, x_lc);
}
void lc_to_ncl(const bt::Tensor& x_lc, int L, int C, bt::Tensor& x_ncl) {
    bt::sequence_to_nchw(x_lc, /*N=*/1, C, /*H=*/1, /*W=*/L, x_ncl);
}


// ─── DurationEncoder ───────────────────────────────────────────────────────

void DurationEncoder::forward(const bt::Tensor& d_en, const bt::Tensor& style,
                              int L, bt::Tensor& d) const {
    const int C = channels;
    const int S = style_dim;

    // (1, C*L) NCL -> (L, C) NLC.
    const bt::Device dev = d_en.device;
    bt::Tensor x = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    ncl_to_lc(d_en, C, L, x);

    // Concat style across L: (L, C) -> (L, C + S). The style tile is built
    // once per forward — modulate with zero X and scale = -1 broadcasts the
    // style row across L in one launch (Y = 0*(1+(-1)) + shift = shift) —
    // then each concat is a single batched column-block op instead of the
    // earlier 2*L copy_d2d storm.
    bt::Tensor style_tile = bt::Tensor::zeros_on(dev, L, S, bt::Dtype::FP32);
    {
        bt::Tensor minus_one = bt::Tensor::zeros_on(dev, S, 1, bt::Dtype::FP32);
        bt::add_scalar_inplace(minus_one, -1.0f);
        bt::Tensor tile = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        bt::modulate(style_tile, minus_one, style, tile);
        style_tile = std::move(tile);
    }
    auto cat_with_style = [&](const bt::Tensor& xs, bt::Tensor& out) {
        // out may have been default-constructed on first call; migrate to dev
        // before resize since Tensor::resize preserves device.
        if (out.device != dev) {
            out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        }
        bt::concat_batched_rows({&xs, &style_tile}, out);
    };

    bt::Tensor cat = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    cat_with_style(x, cat);

    for (const auto& blk : blocks) {
        // BiLSTM step: (L, C+S) -> (L, C).
        bt::Tensor lstm_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        blk.bilstm.forward(cat, lstm_out);

        // AdaLayerNorm step: (L, C) -> (L, C).
        bt::Tensor aln_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        ada_layernorm(blk.aln, L, lstm_out, style, aln_out);

        // Concat style back: (L, C) + style -> (L, C+S).
        cat_with_style(aln_out, cat);
    }

    d = std::move(cat);  // (L, C + S)
}

// ─── Predictor ─────────────────────────────────────────────────────────────

void Predictor::load_from(const stf::File& f, const KokoroConfig& c) {
    cfg = c;
    const std::string where = "Predictor::load_from";
    const std::string p = "predictor.module.";
    const int C = c.hidden_dim;       // 512
    const int S = c.style_dim;        // 128
    const int H = C / 2;              // BiLSTM hidden = 256

    // ─── DurationEncoder ──────────────────────────────────────────────────
    text_encoder.channels  = C;
    text_encoder.style_dim = S;
    text_encoder.nlayers   = c.n_layer;
    text_encoder.blocks.clear();
    text_encoder.blocks.reserve(c.n_layer);
    for (int i = 0; i < c.n_layer; ++i) {
        DurationEncoder::Block blk;
        // lstms.{2*i}: BiLSTM input=C+S, hidden=H.
        blk.bilstm.input_size  = C + S;
        blk.bilstm.hidden_size = H;
        const std::string lp = p + "text_encoder.lstms." + std::to_string(2 * i) + ".";
        load_lstm_cell(f, lp, C + S, H, false, blk.bilstm.forward_cell, where);
        load_lstm_cell(f, lp, C + S, H, true,  blk.bilstm.reverse_cell, where);

        const std::string ap = p + "text_encoder.lstms." + std::to_string(2 * i + 1);
        load_ada_layernorm(f, ap, C, S, blk.aln, where);
        text_encoder.blocks.push_back(std::move(blk));
    }

    // ─── Duration LSTM + proj ─────────────────────────────────────────────
    lstm.input_size  = C + S;
    lstm.hidden_size = H;
    load_lstm_cell(f, p + "lstm.", C + S, H, false, lstm.forward_cell, where);
    load_lstm_cell(f, p + "lstm.", C + S, H, true,  lstm.reverse_cell, where);

    duration_proj.in_features  = C;
    duration_proj.out_features = c.max_dur;
    upload(f, p + "duration_proj.linear_layer.weight",
           c.max_dur, C, duration_proj.W, where);
    upload(f, p + "duration_proj.linear_layer.bias",
           c.max_dur, 1, duration_proj.b, where);

    // ─── Shared BiLSTM ────────────────────────────────────────────────────
    shared.input_size  = C + S;
    shared.hidden_size = H;
    load_lstm_cell(f, p + "shared.", C + S, H, false, shared.forward_cell, where);
    load_lstm_cell(f, p + "shared.", C + S, H, true,  shared.reverse_cell, where);

    // ─── F0 / N blocks (3 each, with the middle one upsampling) ───────────
    auto load_f0n = [&](const std::string& prefix,
                        std::vector<AdainResBlk1dWeights>& blocks) {
        blocks.clear();
        blocks.resize(3);
        load_adain_resblk(f, prefix + ".0", C,     C,     S, /*upsample=*/false, blocks[0], where);
        load_adain_resblk(f, prefix + ".1", C,     C / 2, S, /*upsample=*/true,  blocks[1], where);
        load_adain_resblk(f, prefix + ".2", C / 2, C / 2, S, /*upsample=*/false, blocks[2], where);
    };
    load_f0n(p + "F0", F0_blocks);
    load_f0n(p + "N",  N_blocks);

    // ─── Final 1x1 conv projections (C/2 -> 1) ────────────────────────────
    load_conv1d(f, p + "F0_proj", /*C_out=*/1, /*C_in=*/C / 2, /*kL=*/1, true,
                F0_proj, where);
    load_conv1d(f, p + "N_proj",  /*C_out=*/1, /*C_in=*/C / 2, /*kL=*/1, true,
                N_proj, where);
    F0_proj.padding = 0;
    N_proj.padding  = 0;
}

void Predictor::forward(const bt::Tensor& d_en, const bt::Tensor& ref_s,
                        int L, float speed, Output& out) const {
    const int C = cfg.hidden_dim;
    const int S = cfg.style_dim;

    // Style for predictor = ref_s[:, style_dim:]. ref_s lives on ref_s.device
    // (the model's device); slice via copy_d2d so style stays on-device.
    bt::Tensor style = bt::Tensor::zeros_on(ref_s.device, 1, S, bt::Dtype::FP32);
    bt::copy_d2d(ref_s, S, style, 0, S);

    // ─── DurationEncoder ──────────────────────────────────────────────────
    // Pre-allocate every Predictor::Output tensor on ref_s.device so callers
    // that default-construct the Output struct don't drag CPU-resident out
    // params through brotensor's device dispatch.
    const bt::Device dev = ref_s.device;
    if (out.d.device      != dev) out.d        = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    if (out.lstm_x.device != dev) out.lstm_x   = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    if (out.duration.device != dev) out.duration = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    text_encoder.forward(d_en, style, L, out.d);  // (L, C + S)
    kokoro_profile_mark(dev, "pred:dur_encoder");

    // ─── Duration LSTM + projection ──────────────────────────────────────
    lstm.forward(out.d, out.lstm_x);              // (L, C)
    duration_proj.forward_batched(out.lstm_x, out.duration);  // (L, max_dur)
    kokoro_profile_mark(dev, "pred:dur_lstm+proj");

    // sigmoid + sum over the max_dur axis, / speed, round, clamp to >= 1.
    // The integer output is inherently host-side control flow — round-trip
    // out.duration through to_host_vector to read the values.
    out.pred_dur.assign(L, 0);
    const std::vector<float> dur_host = out.duration.to_host_vector();
    int total = 0;
    for (int l = 0; l < L; ++l) {
        float s = 0.0f;
        for (int k = 0; k < cfg.max_dur; ++k) {
            const float v = dur_host[l * cfg.max_dur + k];
            s += 1.0f / (1.0f + std::exp(-v));
        }
        s /= speed;
        int rounded = static_cast<int>(std::round(s));
        if (rounded < 1) rounded = 1;
        out.pred_dur[l] = rounded;
        total += rounded;
    }

    // ─── Length regulator ────────────────────────────────────────────────
    // en = d.T @ alignment, where alignment is (L, total) one-hot expansion.
    // Equivalently: en[c, t] = d[phoneme(t), c] (with `phoneme(t)` mapping
    // each frame t back to the source phoneme). The repeat-counts come from
    // pred_dur (host control flow), so the gather runs on host once and the
    // result is uploaded to ref_s.device (== dev).
    {
        const std::vector<float> d_host = out.d.to_host_vector();
        std::vector<float> en_host(static_cast<std::size_t>(C + S) * total);
        int t = 0;
        for (int l = 0; l < L; ++l) {
            const int reps = out.pred_dur[l];
            for (int r = 0; r < reps; ++r) {
                for (int c = 0; c < C + S; ++c) {
                    en_host[static_cast<std::size_t>(c) * total + t] =
                        d_host[static_cast<std::size_t>(l) * (C + S) + c];
                }
                ++t;
            }
        }
        out.en = bt::Tensor::from_host_on(dev, en_host.data(),
                                          1, (C + S) * total);
    }
    kokoro_profile_mark(dev, "pred:length_reg");

    // ─── F0Ntrain ────────────────────────────────────────────────────────
    // shared LSTM input = en.T  -> (total, C+S) -> (total, C). Then split into
    // F0 path and N path, each a stack of 3 AdaINResBlk1d (one upsamples).
    bt::Tensor en_lc = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    ncl_to_lc(out.en, C + S, total, en_lc);  // (total, C+S)

    bt::Tensor shared_out = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    shared.forward(en_lc, shared_out);        // (total, C)
    kokoro_profile_mark(dev, "pred:shared_lstm");

    bt::Tensor shared_ncl = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    lc_to_ncl(shared_out, total, C, shared_ncl);  // (1, C*total) NCL

    // F0 stack.
    bt::Tensor F0_x = shared_ncl;
    int F0_L = total;
    int next_L;
    for (const auto& blk : F0_blocks) {
        bt::Tensor y = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        adain_resblk_1d_forward(blk, F0_x, F0_L, style, next_L, y);
        F0_x = std::move(y);
        F0_L = next_L;
    }
    // F0_proj: Conv1d(C/2 -> 1, k=1). Output (1, 1*F0_L).
    if (out.F0_pred.device != dev) {
        out.F0_pred = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    }
    F0_proj.forward(F0_x, /*N=*/1, /*L=*/F0_L, out.F0_pred);
    kokoro_profile_mark(dev, "pred:F0_stack");

    // N stack.
    bt::Tensor N_x = shared_ncl;
    int N_L = total;
    for (const auto& blk : N_blocks) {
        bt::Tensor y = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
        adain_resblk_1d_forward(blk, N_x, N_L, style, next_L, y);
        N_x = std::move(y);
        N_L = next_L;
    }
    if (out.N_pred.device != dev) {
        out.N_pred = bt::Tensor::empty_on(dev, 0, 0, bt::Dtype::FP32);
    }
    N_proj.forward(N_x, /*N=*/1, /*L=*/N_L, out.N_pred);
    kokoro_profile_mark(dev, "pred:N_stack");
}


}  // namespace brosoundml
