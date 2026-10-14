// HTDemucs network: weight loading, the two encoder/decoder branches, the
// spectrogram front and back ends. The cross transformer is in
// htdemucs_transformer.cpp. See htdemucs_model.h.
#include "htdemucs_model.h"

#include <brotensor/ops.h>

#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace brosoundml::htd {

namespace sf = brotensor::safetensors;

namespace {

// ── loading ──────────────────────────────────────────────────────────────────

class Loader {
public:
    Loader(const sf::File& f, bt::Device dev) : f_(f), dev_(dev) {}

    const float* raw(const std::string& name, long long numel) const {
        const sf::TensorView* v = f_.find(name);
        if (!v) fail("missing tensor '" + name + "'");
        if (v->dtype != sf::Dtype::F32) fail("tensor '" + name + "' is not F32");
        if (v->numel() != numel)
            fail("tensor '" + name + "' has " + std::to_string(v->numel()) + " elements, expected " +
                 std::to_string(numel));
        return reinterpret_cast<const float*>(v->data);
    }
    bt::Tensor mat(const std::string& name, int r, int c) const {
        return bt::Tensor::from_host_on(dev_, raw(name, static_cast<long long>(r) * c), r, c);
    }
    bt::Tensor vec(const std::string& name, int n) const { return mat(name, n, 1); }
    bt::Tensor upload(const std::vector<float>& v, int r, int c) const {
        return bt::Tensor::from_host_on(dev_, v.data(), r, c);
    }

    Conv conv(const std::string& P, int cin, int cout, int kh, int kw, int sh, int sw, int ph, int pw,
              int dil = 1) const {
        Conv c;
        c.cin = cin; c.cout = cout; c.kh = kh; c.kw = kw; c.sh = sh; c.sw = sw;
        c.ph = ph; c.pw = pw; c.dw = dil;
        c.w = mat(P + "weight", cout, cin * kh * kw);
        c.b = vec(P + "bias", cout);
        return c;
    }
    Conv conv_tr(const std::string& P, int cin, int cout, int kh, int kw, int sh, int sw) const {
        Conv c;
        c.cin = cin; c.cout = cout; c.kh = kh; c.kw = kw; c.sh = sh; c.sw = sw;
        c.transpose = true;
        c.w = mat(P + "weight", cin, cout * kh * kw);
        c.b = vec(P + "bias", cout);
        return c;
    }
    GroupNorm1 gn(const std::string& P, int C, const float* value_scale = nullptr) const {
        GroupNorm1 g;
        g.C = C;
        std::vector<float> w(raw(P + "weight", C), raw(P + "weight", C) + C);
        std::vector<float> b(raw(P + "bias", C), raw(P + "bias", C) + C);
        if (value_scale) {   // LayerScale after the GLU: scale the value half's affine
            for (int c = 0; c < C / 2; ++c) {
                w[c] *= value_scale[c];
                b[c] *= value_scale[c];
            }
        }
        g.gamma = upload(w, C, 1);
        g.beta = upload(b, C, 1);
        return g;
    }
    DConv dconv(const std::string& P, int C, int depth, int compress) const {
        DConv d;
        d.C = C;
        const int hid = C / compress;
        for (int i = 0; i < depth; ++i) {
            const std::string L = P + "layers." + std::to_string(i) + ".";
            DConvLayer l;
            l.dil = 1 << i;
            l.c1 = conv(L + "0.", C, hid, 1, 3, 1, 1, 0, l.dil, l.dil);
            l.n1 = gn(L + "1.", hid);
            l.c2 = conv(L + "3.", hid, 2 * C, 1, 1, 1, 1, 0, 0);
            l.n2 = gn(L + "4.", 2 * C, raw(L + "6.scale", C));
            d.layers.push_back(std::move(l));
        }
        return d;
    }
    Linear linear(const std::string& P, int in, int out) const {
        Linear l;
        l.in = in; l.out = out;
        l.w = mat(P + "weight", out, in);
        l.b = vec(P + "bias", out);
        return l;
    }
    LayerNorm ln(const std::string& P, int D) const {
        LayerNorm n;
        n.gamma = vec(P + "weight", D);
        n.beta = vec(P + "bias", D);
        return n;
    }

    // nn.MultiheadAttention's packed in_proj, split into q / k / v.
    void attn(const std::string& P, int D, TLayer& t) const {
        const float* w = raw(P + "in_proj_weight", 3LL * D * D);
        const float* b = raw(P + "in_proj_bias", 3LL * D);
        Linear* dst[3] = {&t.q, &t.k, &t.v};
        for (int i = 0; i < 3; ++i) {
            dst[i]->in = D; dst[i]->out = D;
            dst[i]->w = bt::Tensor::from_host_on(dev_, w + static_cast<std::size_t>(i) * D * D, D, D);
            dst[i]->b = bt::Tensor::from_host_on(dev_, b + static_cast<std::size_t>(i) * D, D, 1);
        }
    }
    // Linear with a LayerScale on its output folded in: diag(g) (W x + b).
    Linear scaled_linear(const std::string& P, int in, int out, const float* g) const {
        std::vector<float> w(raw(P + "weight", 1LL * out * in), raw(P + "weight", 1LL * out * in) + 1LL * out * in);
        std::vector<float> b(raw(P + "bias", out), raw(P + "bias", out) + out);
        for (int r = 0; r < out; ++r) {
            for (int c = 0; c < in; ++c) w[static_cast<std::size_t>(r) * in + c] *= g[r];
            b[r] *= g[r];
        }
        Linear l;
        l.in = in; l.out = out;
        l.w = upload(w, out, in);
        l.b = upload(b, out, 1);
        return l;
    }

    TLayer tlayer(const std::string& P, int D, int hidden, bool cross) const {
        TLayer t;
        t.cross = cross;
        attn(P + (cross ? "cross_attn." : "self_attn."), D, t);
        const float* g1 = raw(P + "gamma_1.scale", D);
        const float* g2 = raw(P + "gamma_2.scale", D);
        t.o = scaled_linear(P + (cross ? "cross_attn.out_proj." : "self_attn.out_proj."), D, D, g1);
        t.l1 = linear(P + "linear1.", D, hidden);
        t.l2 = scaled_linear(P + "linear2.", hidden, D, g2);
        t.norm1 = ln(P + "norm1.", D);
        t.norm2 = ln(P + "norm2.", D);
        if (cross) t.norm3 = ln(P + "norm3.", D);
        t.norm_out = gn(P + "norm_out.", D);
        return t;
    }

private:
    const sf::File& f_;
    bt::Device dev_;
};

// The same buffer seen as (r, c).
bt::Tensor as(const bt::Tensor& t, int r, int c) {
    if (static_cast<long long>(r) * c != static_cast<long long>(t.rows) * t.cols)
        fail("internal: reshape element count mismatch");
    return bt::Tensor::view(t.device, t.data, r, c, t.dtype);
}

}  // namespace

// ── per-batch constants ──────────────────────────────────────────────────────

const std::pair<bt::Tensor, bt::Tensor>& BatchConsts::permute(bt::Device dev, int C, int F) const {
    std::lock_guard<std::mutex> lock(perm_mu);
    const int key = C * 65536 + F;
    auto it = perm_cache.find(key);
    if (it != perm_cache.end()) return it->second;
    std::vector<int> fwd(static_cast<std::size_t>(B) * C * F), inv(fwd.size());
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int f = 0; f < F; ++f) {
                const int cf = (b * C + c) * F + f;   // (B,C,F) row
                const int fc = (b * F + f) * C + c;   // (B,F,C) row
                fwd[fc] = cf;
                inv[cf] = fc;
            }
    return perm_cache.emplace(key, std::make_pair(int_tensor(dev, fwd), int_tensor(dev, inv))).first->second;
}

Net::Net() = default;
Net::~Net() = default;

// ── load ─────────────────────────────────────────────────────────────────────

void Net::load(const sf::File& f, const NetConfig& cfg, bt::Device dev) {
    cfg_ = cfg;
    dev_ = dev;
    if (cfg.depth != 4 || cfg.nfft != 4096 || cfg.kernel_size != 8 || cfg.stride != 4 ||
        cfg.audio_channels != 2 || cfg.bottom_channels <= 0)
        fail("only the htdemucs_ft configuration (depth 4, nfft 4096, kernel 8, stride 4, "
             "stereo, bottom channels) is supported");
    Loader L(f, dev);
    const int S = cfg.sources, A = cfg.audio_channels, K = cfg.kernel_size, st = cfg.stride;
    const int pad = K / 4;
    enc_.clear(); tenc_.clear(); dec_.clear(); tdec_.clear();
    int chin = A, chin_z = 2 * A, chout = cfg.channels, freqs = cfg.nfft / 2;
    std::vector<DecLayer> dec_build, tdec_build;
    for (int i = 0; i < cfg.depth; ++i) {
        if (freqs <= K) fail("a last frequency layer is not supported");
        const std::string si = std::to_string(i);
        EncLayer e;
        e.freq = true;
        e.conv = L.conv("encoder." + si + ".conv.", chin_z, chout, K, 1, st, 1, pad, 0);
        e.rewrite = L.conv("encoder." + si + ".rewrite.", chout, 2 * chout, 1, 1, 1, 1, 0, 0);
        e.dconv = L.dconv("encoder." + si + ".dconv.", chout, cfg.dconv_depth, cfg.dconv_comp);
        enc_.push_back(std::move(e));
        EncLayer t;
        t.freq = false;
        t.conv = L.conv("tencoder." + si + ".conv.", chin, chout, 1, K, 1, st, 0, pad);
        t.rewrite = L.conv("tencoder." + si + ".rewrite.", chout, 2 * chout, 1, 1, 1, 1, 0, 0);
        t.dconv = L.dconv("tencoder." + si + ".dconv.", chout, cfg.dconv_depth, cfg.dconv_comp);
        tenc_.push_back(std::move(t));

        // Decoder index (depth-1-i) in state-dict order is the mirror of encoder i.
        const int di = cfg.depth - 1 - i;
        const int dout_z = i == 0 ? S * 2 * A : chin_z;
        const int dout_t = i == 0 ? S * A : chin;
        const std::string sd = std::to_string(di);
        DecLayer d;
        d.freq = true;
        d.last = i == 0;
        d.rewrite = L.conv("decoder." + sd + ".rewrite.", chout, 2 * chout, 3, 3, 1, 1, cfg.context, cfg.context);
        d.dconv = L.dconv("decoder." + sd + ".dconv.", chout, cfg.dconv_depth, cfg.dconv_comp);
        d.conv_tr = L.conv_tr("decoder." + sd + ".conv_tr.", chout, dout_z, K, 1, st, 1);
        dec_build.push_back(std::move(d));
        DecLayer td;
        td.freq = false;
        td.last = i == 0;
        td.rewrite = L.conv("tdecoder." + sd + ".rewrite.", chout, 2 * chout, 1, 1 + 2 * cfg.context, 1, 1, 0,
                            cfg.context);
        td.dconv = L.dconv("tdecoder." + sd + ".dconv.", chout, cfg.dconv_depth, cfg.dconv_comp);
        td.conv_tr = L.conv_tr("tdecoder." + sd + ".conv_tr.", chout, dout_t, 1, K, 1, st);
        tdec_build.push_back(std::move(td));

        chin = chout;
        chin_z = chout;
        chout *= cfg.growth;
        freqs /= st;
        if (i == 0) {
            // freq_emb: x += freq_emb * (scale * W)^T, broadcast over time.
            const float* w = L.raw("freq_emb.embedding.weight", 1LL * freqs * chin);
            std::vector<float> fe(static_cast<std::size_t>(chin) * freqs);
            for (int fr = 0; fr < freqs; ++fr)
                for (int c = 0; c < chin; ++c)
                    fe[static_cast<std::size_t>(c) * freqs + fr] =
                        cfg.freq_emb * (w[static_cast<std::size_t>(fr) * chin + c] * cfg.emb_scale);
            freq_emb_ = L.upload(fe, chin * freqs, 1);
        }
    }
    dec_.assign(std::make_move_iterator(dec_build.rbegin()), std::make_move_iterator(dec_build.rend()));
    tdec_.assign(std::make_move_iterator(tdec_build.rbegin()), std::make_move_iterator(tdec_build.rend()));

    const int Ct = chin, D = cfg.bottom_channels;   // 384, 512
    up_ = L.linear("channel_upsampler.", Ct, D);
    down_ = L.linear("channel_downsampler.", D, Ct);
    up_t_ = L.linear("channel_upsampler_t.", Ct, D);
    down_t_ = L.linear("channel_downsampler_t.", D, Ct);
    norm_in_ = L.ln("crosstransformer.norm_in.", D);
    norm_in_t_ = L.ln("crosstransformer.norm_in_t.", D);
    const int hidden = static_cast<int>(D * cfg.t_hidden_scale);
    layers_.clear(); layers_t_.clear();
    for (int i = 0; i < cfg.t_layers; ++i) {
        const bool cross = i % 2 == 1;
        const std::string si = std::to_string(i);
        layers_.push_back(L.tlayer("crosstransformer.layers." + si + ".", D, hidden, cross));
        layers_t_.push_back(L.tlayer("crosstransformer.layers_t." + si + ".", D, hidden, cross));
    }

    // torch.hann_window(nfft) (periodic).
    std::vector<float> win(cfg.nfft);
    for (int n = 0; n < cfg.nfft; ++n) {
        const double s = std::sin(3.14159265358979323846 * n / cfg.nfft);
        win[n] = static_cast<float>(s * s);
    }
    window_ = L.upload(win, 1, cfg.nfft);
    unit_gamma_ = fill(dev, 1, 1, 1.0f);
    unit_beta_ = fill(dev, 1, 1, 0.0f);
    std::lock_guard<std::mutex> lock(consts_mu_);
    consts_.clear();
}

// ── spectrogram front / back ─────────────────────────────────────────────────

// mix (B, 2*L) -> CaC magnitude (B, 2*2*F*T): HTDemucs._spec + _magnitude.
void Net::encode_freq_input(const bt::Tensor& mix, int B, bt::Tensor& spec) const {
    const int L = cfg_.segment_samples, nfft = cfg_.nfft, hl = nfft / 4, bins = nfft / 2 + 1;
    const int le = (L + hl - 1) / hl, pad = hl / 2 * 3, N = B * cfg_.audio_channels;
    const int Lp = L + 2 * pad + le * hl - L;
    bt::Tensor xp, z, zc, zt;
    bt::pad1d_forward(as(mix, N, L), N, 1, L, pad, pad + le * hl - L, /*reflect*/ 1, xp);
    bt::stft(as(xp, N, Lp), window_, N, nfft, hl, nfft, /*center*/ true, /*normalized*/ true, z);
    const int frames = le + 4;
    if (z.rows != N * frames) fail("internal: unexpected STFT frame count");
    zc = bt::Tensor::empty_on(dev_, N * le, 2 * bins);
    bt::copy_d2d_strided(z, 2 * 2 * bins, frames * 2 * bins, zc, 0, le * 2 * bins, le * 2 * bins, N);
    bt::nchw_to_sequence(as(zc, N, le * 2 * bins), N, le, 1, 2 * bins, zt);   // (N*2*bins, le)
    bt::gather_rows(zt, consts(B).spec_idx, spec);                              // (N*2*F, le)
}

// Decoder output (B, S*C*2*F*T), de-normalised -> waveform (B, S*C*L):
// HTDemucs._mask (CaC) + _ispec.
void Net::decode_freq_output(const bt::Tensor& x, int B, bt::Tensor& wave) const {
    const int L = cfg_.segment_samples, nfft = cfg_.nfft, hl = nfft / 4, F = nfft / 2, bins = F + 1;
    const int T = (L + hl - 1) / hl, pad = hl / 2 * 3;
    const int n = B * cfg_.sources * cfg_.audio_channels;
    bt::Tensor g, seq, w;
    bt::gather_rows(as(x, n * 2 * F, T), consts(B).ispec_idx, g);   // (n*F*2, T)
    bt::nchw_to_sequence(as(g, n, 2 * F * T), n, 2 * F, 1, T, seq); // (n*T, 2F)
    bt::Tensor zp = bt::Tensor::zeros_on(dev_, n * (T + 4), 2 * bins);
    // The decoder's DC bin has an imaginary part, which an inverse real FFT
    // has no use for and ignores. torch.istft on CUDA (the reference) does
    // not: cuFFT's strided C2R plan folds Im(X0) = a into the output as
    // -a * (-1)^n, exactly what a Nyquist bin of real part -a adds. The
    // Nyquist bin is otherwise zero here, so writing -Im(X0) there makes
    // brotensor's istft give the reference's waveform.
    bt::Tensor nyq = bt::Tensor::empty_on(dev_, n * T, 1);
    bt::copy_d2d_strided(seq, 1, 2 * F, nyq, 0, 1, 1, n * T);
    bt::scale_inplace(nyq, -1.0f);
    for (int i = 0; i < n; ++i) {
        const int off = (i * (T + 4) + 2) * 2 * bins;
        bt::copy_d2d_strided(seq, i * T * 2 * F, 2 * F, zp, off, 2 * bins, 2 * F, T);
        bt::copy_d2d_strided(nyq, i * T, 1, zp, off + 2 * F, 2 * bins, 1, T);
    }
    const int le = hl * T + 2 * pad;
    bt::istft(zp, window_, n, le, nfft, hl, nfft, /*center*/ true, /*normalized*/ true, w);
    slice_cols(w, pad, L, wave);   // (n, L)
    reshape(wave, B, cfg_.sources * cfg_.audio_channels * L);
}

// ── forward ──────────────────────────────────────────────────────────────────

void Net::forward(const bt::Tensor& mix, int B, bt::Tensor& out) const {
    const int L = cfg_.segment_samples, A = cfg_.audio_channels, S = cfg_.sources;
    if (mix.rows * static_cast<long long>(mix.cols) != 1LL * B * A * L) fail("forward: mix shape mismatch");
    const BatchConsts& bc = consts(B);
    const int hl = cfg_.nfft / 4, T = (L + hl - 1) / hl;

    // Frequency branch input, normalised over (C, F, T).
    bt::Tensor mag, mean, sd, x;
    encode_freq_input(mix, B, mag);
    reshape(mag, B, 2 * A * (cfg_.nfft / 2) * T);
    row_mean_std(mag, mean, sd);
    row_normalize(mag, mean, sd, 1e-5f, x);
    // Time branch input, normalised over (C, L).
    bt::Tensor meant, sdt, xt;
    const bt::Tensor mixr = as(mix, B, A * L);
    row_mean_std(mixr, meant, sdt);
    row_normalize(mixr, meant, sdt, 1e-5f, xt);

    // Encoders. Each layer's output is its skip connection and the next
    // layer's input.
    std::vector<bt::Tensor> saved, saved_t;
    std::vector<int> lengths_t;
    int F = cfg_.nfft / 2, Cf = 2 * A, Lt = L, Ct = A;
    bt::Tensor y, h, p;
    for (std::size_t i = 0; i < enc_.size(); ++i) {
        // time branch
        const EncLayer& te = tenc_[i];
        const bt::Tensor* tin = i == 0 ? &xt : &saved_t.back();
        lengths_t.push_back(Lt);
        if (Lt % cfg_.stride) {
            const int extra = cfg_.stride - Lt % cfg_.stride;
            bt::pad1d_forward(*tin, B, Ct, Lt, 0, extra, /*zero*/ 0, p);
            tin = &p;
            Lt += extra;
        }
        te.conv.forward(*tin, B, 1, Lt, y);
        Lt = te.conv.out_w(Lt);
        Ct = te.conv.cout;
        bt::gelu_exact_forward(y, y);
        te.dconv.forward(y, B, Lt);
        te.rewrite.forward(y, B, 1, Lt, h);
        bt::Tensor ot;
        glu(h, B, Ct * Lt, ot);
        saved_t.push_back(std::move(ot));

        // frequency branch
        const EncLayer& e = enc_[i];
        e.conv.forward(i == 0 ? x : saved.back(), B, F, T, y);
        F = e.conv.out_h(F);
        Cf = e.conv.cout;
        bt::gelu_exact_forward(y, y);
        {
            const auto& pr = bc.permute(dev_, Cf, F);
            bt::gather_rows(as(y, B * Cf * F, T), pr.first, h);    // (B*F, C*T)
            reshape(h, B * F, Cf * T);
            e.dconv.forward(h, B * F, T);
            bt::gather_rows(as(h, B * F * Cf, T), pr.second, y);
        }
        e.rewrite.forward(as(y, B, Cf * F * T), B, F, T, h);
        bt::Tensor o;
        glu(h, B, Cf * F * T, o);
        if (i == 0) bt::add_channel_bias_inplace(o, bc.freq_emb, B * Cf * F, T);
        saved.push_back(std::move(o));
    }

    // Cross transformer over (B, C, F, T) and (B, C, Lt).
    x = saved.back();
    xt = saved_t.back();
    transformer(x, xt, B, F, T, Lt);

    for (std::size_t i = 0; i < dec_.size(); ++i) {
        // frequency branch
        const DecLayer& d = dec_[i];
        bt::add_inplace(x, saved.back());
        saved.pop_back();
        d.rewrite.forward(x, B, F, T, h);
        glu(h, B, Cf * F * T, y);
        {
            const auto& pr = bc.permute(dev_, Cf, F);
            bt::gather_rows(as(y, B * Cf * F, T), pr.first, h);
            reshape(h, B * F, Cf * T);
            d.dconv.forward(h, B * F, T);
            bt::gather_rows(as(h, B * F * Cf, T), pr.second, y);
        }
        d.conv_tr.forward(as(y, B, Cf * F * T), B, F, T, h);
        const int Fo = d.conv_tr.out_h(F), pad = cfg_.kernel_size / 4;
        bt::slice2d_forward(h, B, d.conv_tr.cout, Fo, T, pad, 0, Fo - 2 * pad, T, x);
        F = Fo - 2 * pad;
        Cf = d.conv_tr.cout;
        if (!d.last) bt::gelu_exact_forward(x, x);

        // time branch
        const DecLayer& td = tdec_[i];
        bt::add_inplace(xt, saved_t.back());
        saved_t.pop_back();
        td.rewrite.forward(xt, B, 1, Lt, h);
        glu(h, B, Ct * Lt, y);
        td.dconv.forward(y, B, Lt);
        td.conv_tr.forward(y, B, 1, Lt, h);
        const int Lo = td.conv_tr.out_w(Lt), length = lengths_t.back();
        lengths_t.pop_back();
        Ct = td.conv_tr.cout;
        bt::slice2d_forward(h, B, Ct, 1, Lo, 0, cfg_.kernel_size / 4, 1, length, xt);
        Lt = length;
        if (!td.last) bt::gelu_exact_forward(xt, xt);
    }
    if (Lt != L || Ct != S * A || F != cfg_.nfft / 2 || Cf != S * A * 2) fail("internal: decoder shape mismatch");

    // De-normalise both branches and sum.
    row_affine(as(x, B, Cf * F * T), sd, mean, x);
    decode_freq_output(x, B, out);
    row_affine(as(xt, B, S * A * L), sdt, meant, xt);
    bt::add_inplace(out, xt);
}

}  // namespace brosoundml::htd
