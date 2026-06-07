#include "brosoundml/rave.h"

#include "brosoundml/detail/json.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;
namespace j  = detail::json;
namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: rave: " + msg);
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Upload a 2D-flattened weight to FP32 on `dev` (RAVE weights are F32 on disk).
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols, bt::Device dev) {
    bt::Tensor t;
    { bt::DeviceScope cpu(bt::Device::CPU); sf::upload(need(f, name), rows, cols, t); }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
bt::Tensor up_vec(const sf::File& f, const std::string& name, int c, bt::Device dev) {
    return up(f, name, c, 1, dev);
}

// ── parameter blocks ─────────────────────────────────────────────────────────

struct ConvW {
    bt::Tensor w;          // (Cout, (Cin/groups)*K) for conv; (Cin,(Cout)*K) for convT
    bt::Tensor b;
    bool       has_b = false;
    int        cin = 0, cout = 0, k = 0;
};

struct BN {
    bt::Tensor g, beta, mean, var;
};

struct EncLayer {
    ConvW conv;
    int   stride  = 1;
    int   groups  = 1;
    bool  has_bn  = false;
    BN    bn;
    bool  act     = true;   // trailing LeakyReLU
};

struct ResBlock {
    ConvW c1;               // dilated conv
    int   dil = 1;
    ConvW c2;               // dilation-1 conv
};

struct UpStack {
    ConvW                 convt;
    int                   stride = 1;
    std::vector<ResBlock> res;
};

// Load a conv1d weight (PyTorch layout (Cout, Cin/groups, K)) from its view's
// 3D shape, flattening to brotensor's (Cout, (Cin/groups)*K).
ConvW load_conv(const sf::File& f, const std::string& name, int groups, bt::Device dev) {
    const sf::TensorView& wv = need(f, name + ".weight");
    if (wv.shape.size() != 3) fail("conv '" + name + "' is not rank-3");
    const int d0 = static_cast<int>(wv.shape[0]);
    const int d1 = static_cast<int>(wv.shape[1]);
    const int k  = static_cast<int>(wv.shape[2]);
    ConvW c;
    c.cout = d0;
    c.cin  = d1 * groups;
    c.k    = k;
    c.w    = up(f, name + ".weight", d0, d1 * k, dev);
    if (const sf::TensorView* bv = f.find(name + ".bias")) {
        c.b = up_vec(f, name + ".bias", static_cast<int>(bv->shape[0]), dev);
        c.has_b = true;
    }
    return c;
}

// Load a conv_transpose1d weight (PyTorch layout (Cin, Cout/groups, K)) — direct
// flatten to brotensor's (Cin, (Cout/groups)*K). groups == 1 for RAVE upsamplers.
ConvW load_convt(const sf::File& f, const std::string& name, bt::Device dev) {
    const sf::TensorView& wv = need(f, name + ".weight");
    if (wv.shape.size() != 3) fail("convT '" + name + "' is not rank-3");
    const int d0 = static_cast<int>(wv.shape[0]);   // Cin
    const int d1 = static_cast<int>(wv.shape[1]);   // Cout
    const int k  = static_cast<int>(wv.shape[2]);
    ConvW c;
    c.cin  = d0;
    c.cout = d1;
    c.k    = k;
    c.w    = up(f, name + ".weight", d0, d1 * k, dev);
    if (const sf::TensorView* bv = f.find(name + ".bias")) {
        c.b = up_vec(f, name + ".bias", static_cast<int>(bv->shape[0]), dev);
        c.has_b = true;
    }
    return c;
}

BN load_bn(const sf::File& f, const std::string& name, int c, bt::Device dev) {
    BN bn;
    bn.g    = up_vec(f, name + ".weight",       c, dev);
    bn.beta = up_vec(f, name + ".bias",         c, dev);
    bn.mean = up_vec(f, name + ".running_mean", c, dev);
    bn.var  = up_vec(f, name + ".running_var",  c, dev);
    return bn;
}

// ── offline-causal compute primitives (cached_conv with zeroed caches) ────────

// Causal conv: left-pad dilation*(k-1), then strided/dilated/grouped valid conv.
// Lout = (L-1)/stride + 1. Matches rave_reference.py::causal_conv.
bt::Tensor cconv(const bt::Tensor& x, int Cin, int L, const ConvW& c,
                 int stride, int dilation, int groups, int& Lout) {
    const int pad_left = dilation * (c.k - 1);
    bt::Tensor padded, y;
    bt::pad1d_forward(x, /*N=*/1, Cin, L, pad_left, /*pad_right=*/0, /*mode=*/0, padded);
    bt::conv1d(padded, c.w, c.has_b ? &c.b : nullptr, /*N=*/1, Cin, L + pad_left,
               c.cout, c.k, stride, /*padding=*/0, dilation, groups, y);
    Lout = (L - 1) / stride + 1;
    return y;
}

// Causal transposed conv: full conv_transpose1d, drop the last (k - stride).
// Lout = L*stride. Matches rave_reference.py::causal_convt.
bt::Tensor cconvt(const bt::Tensor& x, int Cin, int L, const ConvW& c, int stride, int& Lout) {
    bt::Tensor full;
    bt::conv_transpose1d_forward(x, c.w, c.has_b ? &c.b : nullptr, /*N=*/1, Cin, L,
                                 c.cout, c.k, stride, /*padding=*/0, /*output_padding=*/0,
                                 /*dilation=*/1, full);
    const int L_full = (L - 1) * stride + c.k;
    const int trim   = c.k - stride;
    Lout = L_full - trim;
    bt::Tensor y = bt::Tensor::zeros_on(x.device, 1, c.cout * Lout, bt::Dtype::FP32);
    for (int ch = 0; ch < c.cout; ++ch)
        bt::copy_d2d(full, ch * L_full, y, ch * Lout, Lout);
    return y;
}

bt::Tensor leaky(const bt::Tensor& x, float slope) {
    bt::Tensor y;
    bt::leaky_relu_forward(x, slope, y);
    return y;
}

// PQMF reverse_half (in place): multiply ODD channels by (-1)^(t+1) along time
// (t even -> -1, t odd -> +1). NOT a flat sign flip. x is (1, C*L) NCL.
void reverse_half_inplace(bt::Tensor& x, int C, int L) {
    std::vector<float> mask(static_cast<std::size_t>(C) * L, 1.0f);
    for (int c = 1; c < C; c += 2)
        for (int t = 0; t < L; ++t)
            mask[static_cast<std::size_t>(c) * L + t] = (t % 2 == 0) ? -1.0f : 1.0f;
    bt::Tensor m = bt::Tensor::from_host_on(x.device, mask.data(), 1, C * L);
    bt::mul_inplace(x, m);
}

// mod_sigmoid(x) = 2*sigmoid(x)^2.3 + 1e-7, via s^2.3 = exp(2.3*log(s)).
bt::Tensor mod_sigmoid(const bt::Tensor& x) {
    bt::Tensor s;
    bt::sigmoid_forward(x, s);
    bt::Tensor lg;
    bt::log_forward(s, lg);
    bt::scale_inplace(lg, 2.3f);
    bt::Tensor p;
    bt::exp_forward(lg, p);             // s^2.3
    bt::scale_inplace(p, 2.0f);
    bt::add_scalar_inplace(p, 1.0e-7f);
    return p;
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

// ── Impl ─────────────────────────────────────────────────────────────────────

struct Rave::Impl {
    RaveConfig cfg;
    bt::Device dev = bt::Device::CPU;
    bool       is_loaded = false;

    ConvW pqmf_fwd;        // 1 -> n_band, stride n_band
    ConvW pqmf_inv;        // n_band -> n_band, stride 1

    std::vector<EncLayer> enc;

    // PCA projection as 1x1 convs. encode: z = pca @ mean + enc_pca_b, with
    // enc_pca_b = -(pca @ latent_mean). decode unproject: x = pca^T @ z + mean.
    ConvW enc_pca;         // weight (full,full), k=1; bias = enc_pca_b
    ConvW dec_pca;         // weight (full,full) = pca^T, k=1; bias = latent_mean

    ConvW                dec_in;      // decoder.net.0
    std::vector<UpStack> dec_up;      // 4 upsample stacks
    ConvW                synth_wave;  // decoder.synth.branches.0 (-> n_band)
    ConvW                synth_loud;  // decoder.synth.branches.1 (-> 1)

    void load(const std::string& model_dir, bt::Device device);
    RaveLatent encode(const float* audio, int n) const;
    AudioBuffer decode(const float* latent, int n_latent, int frames) const;
};

void Rave::Impl::load(const std::string& model_dir, bt::Device device) {
    dev = device;

    const fs::path dir = model_dir;
    const std::string config_path = (dir / "config.json").string();
    const std::string weights_path = (dir / "model.safetensors").string();
    if (!fs::exists(config_path))  fail("no config.json under '" + model_dir + "'");
    if (!fs::exists(weights_path)) fail("no model.safetensors under '" + model_dir + "'");

    // ── config.json ──
    const j::Value root = j::parse(slurp(config_path));
    if (!root.is_object()) fail("config.json is not a JSON object");
    cfg.sampling_rate       = root.get_int("sampling_rate", 48000);
    cfg.full_latent_size    = root.get_int("full_latent_size", 128);
    cfg.cropped_latent_size = root.get_int("cropped_latent_size", 0);
    cfg.n_band              = root.get_int("n_band", 16);
    cfg.leaky_slope         = root.get_float("leaky_slope", 0.2f);
    cfg.bn_eps              = root.get_float("bn_eps", 1.0e-5f);
    if (cfg.cropped_latent_size <= 0) fail("config.json: cropped_latent_size must be > 0");

    const sf::File f = sf::File::open(weights_path);
    const int full = cfg.full_latent_size;

    // ── PQMF analysis / synthesis convs ──
    pqmf_fwd = load_conv(f, "pqmf.forward_conv", /*groups=*/1, dev);
    pqmf_inv = load_conv(f, "pqmf.inverse_conv", /*groups=*/1, dev);

    // ── encoder: [conv, BN, LeakyReLU] x4 downsample, then [conv, LeakyReLU,
    //    grouped-conv(groups=2)] -> (mean | scale). Fixed RAVE v2 topology
    //    (mirrors rave_reference.py ENC / ENC_BN). ──
    struct Spec { int idx, stride, groups, bn; bool act; };
    const Spec specs[] = {
        {0,  1, 1,  1, true},
        {3,  4, 1,  4, true},
        {6,  4, 1,  7, true},
        {9,  4, 1, 10, true},
        {12, 2, 1, -1, true},
        {14, 1, 2, -1, false},
    };
    enc.clear();
    for (const Spec& s : specs) {
        EncLayer el;
        const std::string base = "encoder.net." + std::to_string(s.idx);
        el.conv   = load_conv(f, base, s.groups, dev);
        el.stride = s.stride;
        el.groups = s.groups;
        el.act    = s.act;
        if (s.bn >= 0) {
            el.has_bn = true;
            el.bn = load_bn(f, "encoder.net." + std::to_string(s.bn), el.conv.cout, dev);
        }
        enc.push_back(std::move(el));
    }

    // Encoder total downsample ratio = n_band * product(encoder strides).
    cfg.total_ratio = cfg.n_band;
    for (const EncLayer& el : enc) cfg.total_ratio *= el.stride;

    // ── PCA projection (fold the latent_mean shift into the 1x1 conv bias) ──
    {
        bt::Tensor pca_cpu, mean_cpu;
        {
            bt::DeviceScope cpu(bt::Device::CPU);
            sf::upload(need(f, "latent_pca"),  full, full, pca_cpu);
            sf::upload(need(f, "latent_mean"), full, 1,    mean_cpu);
        }
        const float* P = pca_cpu.host_f32();    // (full, full), row = out dim
        const float* M = mean_cpu.host_f32();   // (full)
        std::vector<float> enc_b(full), dec_w(static_cast<std::size_t>(full) * full), dec_b(full);
        for (int i = 0; i < full; ++i) {
            float s = 0.0f;
            for (int k = 0; k < full; ++k) s += P[static_cast<std::size_t>(i) * full + k] * M[k];
            enc_b[i] = -s;                       // encode bias: -(pca @ mean)
            dec_b[i] = M[i];                     // decode bias: latent_mean
        }
        for (int i = 0; i < full; ++i)           // dec weight = pca^T
            for (int k = 0; k < full; ++k)
                dec_w[static_cast<std::size_t>(i) * full + k] = P[static_cast<std::size_t>(k) * full + i];

        enc_pca.w = (dev == bt::Device::CPU) ? pca_cpu : pca_cpu.to(dev);
        enc_pca.b = bt::Tensor::from_host_on(dev, enc_b.data(), full, 1);
        enc_pca.has_b = true;
        enc_pca.cin = full; enc_pca.cout = full; enc_pca.k = 1;

        dec_pca.w = bt::Tensor::from_host_on(dev, dec_w.data(), full, full);
        dec_pca.b = bt::Tensor::from_host_on(dev, dec_b.data(), full, 1);
        dec_pca.has_b = true;
        dec_pca.cin = full; dec_pca.cout = full; dec_pca.k = 1;
    }

    // ── decoder: input conv, then 4x [convT upsample + 3 residual blocks] ──
    dec_in = load_conv(f, "decoder.net.0", /*groups=*/1, dev);

    const int up_idx[]    = {1, 3, 5, 7};
    const int up_stride[] = {4, 4, 4, 2};
    const int res_dil[]   = {1, 3, 9};
    dec_up.clear();
    for (int u = 0; u < 4; ++u) {
        UpStack st;
        st.stride = up_stride[u];
        st.convt  = load_convt(f, "decoder.net." + std::to_string(up_idx[u]) + ".net.1", dev);
        const std::string rbase = "decoder.net." + std::to_string(up_idx[u] + 1) + ".net.";
        for (int b = 0; b < 3; ++b) {
            ResBlock rb;
            const std::string p = rbase + std::to_string(b) + ".aligned.branches.0.";
            rb.c1  = load_conv(f, p + "1", /*groups=*/1, dev);
            rb.dil = res_dil[b];
            rb.c2  = load_conv(f, p + "3", /*groups=*/1, dev);
            st.res.push_back(std::move(rb));
        }
        dec_up.push_back(std::move(st));
    }

    // ── synthesis branches ──
    synth_wave = load_conv(f, "decoder.synth.branches.0", /*groups=*/1, dev);
    synth_loud = load_conv(f, "decoder.synth.branches.1", /*groups=*/1, dev);

    is_loaded = true;
}

RaveLatent Rave::Impl::encode(const float* audio, int n) const {
    if (!is_loaded) fail("model not loaded");
    bt::DeviceScope scope(dev);

    // Right-pad to a whole frame so every stride divides evenly and T is exact.
    const int ratio = cfg.total_ratio;
    const int L = ((n + ratio - 1) / ratio) * ratio;
    std::vector<float> buf(static_cast<std::size_t>(L), 0.0f);
    for (int i = 0; i < n; ++i) buf[i] = audio[i];
    bt::Tensor x = bt::Tensor::from_host_on(dev, buf.data(), 1, L);

    // PQMF analysis.
    int Lb;
    bt::Tensor h = cconv(x, /*Cin=*/1, L, pqmf_fwd, /*stride=*/cfg.n_band, 1, 1, Lb);
    reverse_half_inplace(h, cfg.n_band, Lb);

    int C = cfg.n_band, Lc = Lb;
    for (const EncLayer& el : enc) {
        int Lo;
        h = cconv(h, C, Lc, el.conv, el.stride, /*dilation=*/1, el.groups, Lo);
        C = el.conv.cout; Lc = Lo;
        if (el.has_bn) {
            bt::Tensor y;
            bt::batch_norm_inference(h, el.bn.g, el.bn.beta, el.bn.mean, el.bn.var,
                                     /*N=*/1, C, /*H=*/1, /*W=*/Lc, cfg.bn_eps, y);
            h = std::move(y);
        }
        if (el.act) h = leaky(h, cfg.leaky_slope);
    }

    // Split (mean | scale): take the first `full` channels (the posterior mean).
    const int full = cfg.full_latent_size;
    const int T    = Lc;
    bt::Tensor mean = bt::Tensor::zeros_on(dev, 1, full * T, bt::Dtype::FP32);
    bt::copy_d2d(h, 0, mean, 0, full * T);

    // z = pca @ (mean - latent_mean)  (folded into the 1x1 conv bias).
    int Lz;
    bt::Tensor z = cconv(mean, full, T, enc_pca, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Lz);

    // Crop to the kept latent dims.
    const int nl = cfg.cropped_latent_size;
    bt::Tensor zc = bt::Tensor::zeros_on(dev, 1, nl * T, bt::Dtype::FP32);
    bt::copy_d2d(z, 0, zc, 0, nl * T);

    RaveLatent out;
    out.n_latent = nl;
    out.frames   = T;
    out.data     = zc.to_host_vector();   // (1, nl*T) -> channel-major data[c*T + t]
    return out;
}

AudioBuffer Rave::Impl::decode(const float* latent, int n_latent, int frames) const {
    if (!is_loaded) fail("model not loaded");
    if (n_latent != cfg.cropped_latent_size)
        fail("decode: latent has " + std::to_string(n_latent) + " dims, expected " +
             std::to_string(cfg.cropped_latent_size));
    bt::DeviceScope scope(dev);

    const int full = cfg.full_latent_size;
    const int T    = frames;

    // Unproject: zero-pad nl -> full latents, then x = pca^T @ z + latent_mean.
    bt::Tensor zf = bt::Tensor::zeros_on(dev, 1, full * T, bt::Dtype::FP32);
    bt::Tensor zsrc = bt::Tensor::from_host_on(dev, latent, 1, n_latent * T);
    bt::copy_d2d(zsrc, 0, zf, 0, n_latent * T);
    int Lz;
    bt::Tensor h = cconv(zf, full, T, dec_pca, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Lz);

    // Decoder input conv.
    int C = full, Lc = T, Lo;
    h = cconv(h, C, Lc, dec_in, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Lo);
    C = dec_in.cout; Lc = Lo;

    // 4x [LeakyReLU, convT upsample, 3 residual blocks].
    for (const UpStack& st : dec_up) {
        bt::Tensor a = leaky(h, cfg.leaky_slope);
        int Lu;
        h = cconvt(a, C, Lc, st.convt, st.stride, Lu);
        C = st.convt.cout; Lc = Lu;
        for (const ResBlock& rb : st.res) {
            bt::Tensor r = leaky(h, cfg.leaky_slope);
            int L1;
            r = cconv(r, C, Lc, rb.c1, /*stride=*/1, rb.dil, /*groups=*/1, L1);
            r = leaky(r, cfg.leaky_slope);
            int L2;
            r = cconv(r, rb.c1.cout, Lc, rb.c2, /*stride=*/1, /*dilation=*/1, /*groups=*/1, L2);
            bt::add_inplace(h, r);   // x + branch; c2.cout == C
        }
    }

    // Synthesis: tanh(wave) * mod_sigmoid(loud), loud broadcast across bands.
    int Lw;
    bt::Tensor wave = cconv(h, C, Lc, synth_wave, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Lw);
    int Ll;
    bt::Tensor loud = cconv(h, C, Lc, synth_loud, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Ll);

    bt::Tensor tw;
    bt::tanh_forward(wave, tw);                 // (1, n_band*Lc)
    bt::Tensor m = mod_sigmoid(loud);           // (1, 1*Lc)
    bt::Tensor mt = bt::Tensor::zeros_on(dev, 1, cfg.n_band * Lc, bt::Dtype::FP32);
    for (int c = 0; c < cfg.n_band; ++c) bt::copy_d2d(m, 0, mt, c * Lc, Lc);
    bt::mul_inplace(tw, mt);                     // multiband waveform

    // PQMF synthesis: reverse_half, conv, *n_band, then polyphase interleave.
    const int nb = cfg.n_band;
    reverse_half_inplace(tw, nb, Lc);
    int Ly;
    bt::Tensor y = cconv(tw, nb, Lc, pqmf_inv, /*stride=*/1, /*dilation=*/1, /*groups=*/1, Ly);
    std::vector<float> Y = y.to_host_vector();   // (nb, Lc) channel-major, Ly == Lc

    std::vector<float> wav(static_cast<std::size_t>(nb) * Lc);
    for (int t = 0; t < Lc; ++t)
        for (int c = 0; c < nb; ++c)
            wav[static_cast<std::size_t>(t) * nb + c] =
                static_cast<float>(nb) * Y[static_cast<std::size_t>(nb - 1 - c) * Lc + t];

    return AudioBuffer(std::move(wav), cfg.sampling_rate);
}

// ── public wrapper ───────────────────────────────────────────────────────────

Rave::Rave() : impl_(std::make_unique<Impl>()) {}
Rave::~Rave() = default;
Rave::Rave(Rave&&) noexcept = default;
Rave& Rave::operator=(Rave&&) noexcept = default;

void Rave::load(const std::string& model_dir, bt::Device device) {
    impl_->load(model_dir, device);
}

RaveLatent Rave::encode(const std::vector<float>& audio) const {
    return impl_->encode(audio.data(), static_cast<int>(audio.size()));
}
RaveLatent Rave::encode(const float* audio, int n) const {
    return impl_->encode(audio, n);
}

AudioBuffer Rave::decode(const RaveLatent& latent) const {
    return impl_->decode(latent.data.data(), latent.n_latent, latent.frames);
}
AudioBuffer Rave::decode(const float* latent, int n_latent, int frames) const {
    return impl_->decode(latent, n_latent, frames);
}

const RaveConfig& Rave::config() const { return impl_->cfg; }
bool Rave::loaded() const { return impl_->is_loaded; }

}  // namespace brosoundml
