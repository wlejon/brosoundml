#include "brosoundml/supertonic.h"

#include "brosoundml/detail/json.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

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
    throw std::runtime_error("brosoundml: supertonic: " + msg);
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Read an F32 view straight to a host float vector (small denorm / slope params).
std::vector<float> read_host(const sf::File& f, const std::string& name) {
    const sf::TensorView& v = need(f, name);
    if (v.dtype != sf::Dtype::F32) fail("tensor '" + name + "' is not F32");
    const float* p = reinterpret_cast<const float*>(v.data);
    return std::vector<float>(p, p + v.numel());
}

// Upload a 2D-flattened weight to FP32 on `dev` (Supertonic weights are F32).
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols,
              bt::Device dev) {
    bt::Tensor t;
    { bt::DeviceScope cpu(bt::Device::CPU); sf::upload(need(f, name), rows, cols, t); }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
bt::Tensor up_vec(const sf::File& f, const std::string& name, int c, bt::Device dev) {
    return up(f, name, c, 1, dev);
}

std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) fail("cannot open " + p.string());
    std::ostringstream ss; ss << in.rdbuf();
    return ss.str();
}

// ── parameter blocks ─────────────────────────────────────────────────────────

// A conv1d weight flattened to brotensor's (Cout, (Cin/groups)*K) OIL layout.
struct ConvW {
    bt::Tensor w, b;
    bool       has_b = false;
    int        cin_pg = 0, cout = 0, k = 0;  // cin_pg = weight.shape[1] = Cin/groups
};

// Load an ONNX conv weight (Cout, Cin/groups, K) by its exact tensor name.
ConvW load_conv(const sf::File& f, const std::string& wname,
                const std::string& bname, bt::Device dev) {
    const sf::TensorView& wv = need(f, wname);
    if (wv.shape.size() != 3) fail("conv '" + wname + "' is not rank-3");
    ConvW c;
    c.cout   = static_cast<int>(wv.shape[0]);
    c.cin_pg = static_cast<int>(wv.shape[1]);
    c.k      = static_cast<int>(wv.shape[2]);
    c.w = up(f, wname, c.cout, c.cin_pg * c.k, dev);
    if (!bname.empty()) {
        if (const sf::TensorView* bv = f.find(bname)) {
            c.b = up_vec(f, bname, static_cast<int>(bv->shape[0]), dev);
            c.has_b = true;
        }
    }
    return c;
}

// pwconv2 with the ConvNeXt LayerScale gamma folded in: gamma * (W·h + b) =
// (gamma*W)·h + gamma*b, so we scale each output row's weights and bias by
// gamma_c host-side. Eliminates a per-channel broadcast multiply in the forward.
ConvW load_pwconv2_scaled(const sf::File& f, const std::string& prefix,
                          bt::Device dev) {
    const sf::TensorView& wv = need(f, prefix + ".pwconv2.weight");  // (512,2048,1)
    const int cout   = static_cast<int>(wv.shape[0]);
    const int cin_pg = static_cast<int>(wv.shape[1]);
    std::vector<float> w = read_host(f, prefix + ".pwconv2.weight");
    std::vector<float> b = read_host(f, prefix + ".pwconv2.bias");
    std::vector<float> g = read_host(f, prefix + ".gamma");  // (1,cout,1) -> cout
    if (static_cast<int>(g.size()) != cout || static_cast<int>(b.size()) != cout)
        fail("pwconv2 gamma/bias length mismatch in " + prefix);
    for (int c = 0; c < cout; ++c) {
        const float gc = g[c];
        for (int i = 0; i < cin_pg; ++i) w[static_cast<std::size_t>(c) * cin_pg + i] *= gc;
        b[c] *= gc;
    }
    ConvW out;
    out.cout = cout; out.cin_pg = cin_pg; out.k = 1; out.has_b = true;
    out.w = bt::Tensor::from_host_on(dev, w.data(), cout, cin_pg);
    out.b = bt::Tensor::from_host_on(dev, b.data(), cout, 1);
    return out;
}

struct ConvNeXtBlock {
    ConvW      dw;          // depthwise conv (groups = channels), kernel 7, dilated
    int        dil = 1;
    bt::Tensor ln_g, ln_b;  // channel-wise LayerNorm affine
    ConvW      pw1;         // 1x1, channels -> intermediate
    ConvW      pw2;         // 1x1, intermediate -> channels (gamma folded)
};

// Causal conv with LEFT-REPLICATE padding (the upstream cached-conv's offline
// behaviour: clamp the first sample leftward by dilation*(k-1), NOT zero-pad).
// Output length == L. groups == Cin for depthwise. scratch is reused across calls.
bt::Tensor rconv(const bt::Tensor& x, const ConvW& c, int Cin, int L,
                 int dilation, int groups, bt::Tensor& scratch) {
    const int pad_left = dilation * (c.k - 1);
    bt::Tensor y;
    const bt::Tensor* in = &x;
    if (pad_left > 0) {
        bt::pad1d_forward(x, /*N=*/1, Cin, L, pad_left, /*pad_right=*/0,
                          /*mode=*/2 /*replicate*/, scratch);
        in = &scratch;
    }
    bt::conv1d(*in, c.w, c.has_b ? &c.b : nullptr, /*N=*/1, Cin, L + pad_left,
               c.cout, c.k, /*stride=*/1, /*padding=*/0, dilation, groups, y);
    return y;
}

}  // namespace

// ─── Impl ────────────────────────────────────────────────────────────────────

struct Supertonic::Impl {
    SupertonicConfig cfg;
    bt::Device       dev = bt::Device::CPU;
    bool             ready = false;

    // Vocoder (autoencoder decoder) weights.
    float                      inv_scale = 1.0f;   // 1 / ttl.normalizer.scale
    std::vector<float>         latent_mean, latent_std;  // (latent_dim)
    ConvW                      conv_in;            // 24 -> hidden, kernel 7
    std::vector<ConvNeXtBlock> blocks;             // 10 dilated ConvNeXt blocks
    bt::Tensor                 bn_g, bn_b, bn_mean, bn_var;  // final BatchNorm
    ConvW                      head1;              // hidden -> 2*hidden, kernel 3
    float                      prelu_slope = 0.0f;
    ConvW                      head2;              // 2*hidden -> base_chunk, kernel 1

    void load(const std::string& dir, bt::Device device);
    AudioBuffer decode(const float* latent, int channels, int frames) const;
};

void Supertonic::Impl::load(const std::string& dir, bt::Device device) {
    dev = device;
    const fs::path root(dir);

    // tts.json carries the autoencoder geometry; fall back to known defaults.
    if (fs::exists(root / "tts.json")) {
        const j::Value tts = j::parse(slurp(root / "tts.json"));
        if (const j::Value* ae = tts.find("ae")) {
            cfg.sample_rate = ae->get_int("sample_rate", cfg.sample_rate);
            cfg.base_chunk  = ae->get_int("base_chunk_size", cfg.base_chunk);
            cfg.latent_dim  = ae->get_int("ldim", cfg.latent_dim);
        }
        if (const j::Value* ttl = tts.find("ttl"))
            cfg.chunk = ttl->get_int("chunk_compress_factor", cfg.chunk);
    }

    const fs::path voc = root / "vocoder.safetensors";
    if (!fs::exists(voc)) fail("missing vocoder.safetensors under " + dir);
    const sf::File f = sf::File::open(voc.string());

    // De-normalisation: latent / scale, then * latent_std + latent_mean.
    const std::vector<float> scale = read_host(f, "tts.ttl.normalizer.scale");
    inv_scale   = (scale.empty() || scale[0] == 0.0f) ? 1.0f : 1.0f / scale[0];
    latent_mean = read_host(f, "tts.ae.latent_mean");   // (1,latent_dim,1)
    latent_std  = read_host(f, "tts.ae.latent_std");
    if (static_cast<int>(latent_mean.size()) != cfg.latent_dim ||
        static_cast<int>(latent_std.size())  != cfg.latent_dim)
        fail("latent_mean/std length != latent_dim");

    // Initial conv: latent_dim -> hidden (ONNX names this initializer onnx::Conv_*).
    conv_in = load_conv(f, "onnx::Conv_1441", "onnx::Conv_1442", dev);

    // 10 dilated ConvNeXt blocks. Dilations repeat 1,2,4 over the first six, then
    // 1 — taken from the depthwise weight's dilation in the upstream graph.
    static const int kDil[10] = {1, 2, 4, 1, 2, 4, 1, 1, 1, 1};
    blocks.clear();
    for (int i = 0; i < 10; ++i) {
        const std::string p = "tts.ae.decoder.convnext." + std::to_string(i);
        ConvNeXtBlock blk;
        blk.dil  = kDil[i];
        blk.dw   = load_conv(f, p + ".dwconv.net.weight", p + ".dwconv.net.bias", dev);
        blk.ln_g = up_vec(f, p + ".norm.norm.weight", blk.dw.cout, dev);
        blk.ln_b = up_vec(f, p + ".norm.norm.bias",   blk.dw.cout, dev);
        blk.pw1  = load_conv(f, p + ".pwconv1.weight", p + ".pwconv1.bias", dev);
        blk.pw2  = load_pwconv2_scaled(f, p, dev);
        blocks.push_back(std::move(blk));
    }

    // Final BatchNorm (inference: uses running stats).
    bn_g    = up_vec(f, "tts.ae.decoder.final_norm.norm.weight",       conv_in.cout, dev);
    bn_b    = up_vec(f, "tts.ae.decoder.final_norm.norm.bias",         conv_in.cout, dev);
    bn_mean = up_vec(f, "tts.ae.decoder.final_norm.norm.running_mean", conv_in.cout, dev);
    bn_var  = up_vec(f, "tts.ae.decoder.final_norm.norm.running_var",  conv_in.cout, dev);

    // Head: conv (kernel 3) -> PReLU (scalar slope) -> conv (kernel 1, no bias).
    head1 = load_conv(f, "tts.ae.decoder.head.layer1.net.weight",
                      "tts.ae.decoder.head.layer1.net.bias", dev);
    const std::vector<float> slope = read_host(f, "onnx::PRelu_1506");
    prelu_slope = slope.empty() ? 0.0f : slope[0];
    head2 = load_conv(f, "tts.ae.decoder.head.layer2.weight", "", dev);

    ready = true;
}

AudioBuffer Supertonic::Impl::decode(const float* latent, int channels,
                                     int frames) const {
    if (!ready) fail("decode() before load()");
    const int D  = cfg.latent_dim;          // 24
    const int CC = cfg.chunk;                // 6
    if (channels != D * CC)
        fail("latent channels (" + std::to_string(channels) + ") != latent_dim*chunk");
    if (frames <= 0) fail("frames must be positive");
    const int LF = CC * frames;             // de-chunked frame count

    // ── de-chunk [D*CC, frames] -> [D, CC*frames] + de-normalise (host) ──
    // ONNX: reshape [144,F]->[D,CC,F], transpose to [D,F,CC], reshape [D,CC*F];
    // channel (d*CC+j) at frame t lands at de-chunked position t*CC+j.
    std::vector<float> dn(static_cast<std::size_t>(D) * LF);
    for (int d = 0; d < D; ++d) {
        const float sd = latent_std[d], mn = latent_mean[d];
        for (int t = 0; t < frames; ++t) {
            for (int jx = 0; jx < CC; ++jx) {
                const float v = latent[static_cast<std::size_t>(d * CC + jx) * frames + t];
                dn[static_cast<std::size_t>(d) * LF + (t * CC + jx)] = v * inv_scale * sd + mn;
            }
        }
    }

    bt::Tensor scratch;
    bt::Tensor h = rconv(bt::Tensor::from_host_on(dev, dn.data(), 1, D * LF),
                         conv_in, D, LF, /*dilation=*/1, /*groups=*/1, scratch);
    const int C = conv_in.cout;             // hidden channels (512)

    for (const ConvNeXtBlock& blk : blocks) {
        bt::Tensor dwy = rconv(h, blk.dw, C, LF, blk.dil, /*groups=*/C, scratch);
        // channel-wise LayerNorm: [C,L] -> (L,C) -> LN over C -> [C,L].
        bt::Tensor seq, seqn, normed;
        bt::nchw_to_sequence(dwy, /*N=*/1, C, /*H=*/1, /*W=*/LF, seq);
        bt::layernorm_forward_inference_batched(seq, blk.ln_g, blk.ln_b, seqn, 1.0e-6f);
        bt::sequence_to_nchw(seqn, /*N=*/1, C, /*H=*/1, /*W=*/LF, normed);

        bt::Tensor a = rconv(normed, blk.pw1, C, LF, 1, 1, scratch);  // 1x1
        bt::Tensor ga, y;
        bt::gelu_exact_forward(a, ga);
        y = rconv(ga, blk.pw2, blk.pw1.cout, LF, 1, 1, scratch);      // 1x1, gamma folded
        bt::add_inplace(y, h);              // residual
        h = std::move(y);
    }

    bt::Tensor hb;
    bt::batch_norm_inference(h, bn_g, bn_b, bn_mean, bn_var, /*N=*/1, C, /*H=*/1,
                             /*W=*/LF, /*eps=*/1.0e-5f, hb);

    bt::Tensor hp, out;
    bt::Tensor hh = rconv(hb, head1, C, LF, 1, 1, scratch);   // kernel 3
    bt::leaky_relu_forward(hh, prelu_slope, hp);
    out = rconv(hp, head2, head1.cout, LF, 1, 1, scratch);    // kernel 1

    // ── output reshape [base_chunk, LF] -> waveform (interleave to time) ──
    // ONNX: transpose [B,base_chunk,LF]->[B,LF,base_chunk], flatten; sample at
    // position s*base_chunk + c.
    const int BC = head2.cout;              // base_chunk (512)
    const std::vector<float> od = out.to_host_vector();   // (BC*LF) channel-major
    AudioBuffer wav;
    wav.sample_rate = cfg.sample_rate;
    wav.samples.resize(static_cast<std::size_t>(BC) * LF);
    for (int c = 0; c < BC; ++c)
        for (int s = 0; s < LF; ++s)
            wav.samples[static_cast<std::size_t>(s) * BC + c] =
                od[static_cast<std::size_t>(c) * LF + s];
    return wav;
}

// ─── public shell ────────────────────────────────────────────────────────────

Supertonic::Supertonic() : impl_(std::make_unique<Impl>()) {}
Supertonic::~Supertonic() = default;
Supertonic::Supertonic(Supertonic&&) noexcept = default;
Supertonic& Supertonic::operator=(Supertonic&&) noexcept = default;

void Supertonic::load(const std::string& model_dir, bt::Device device) {
    impl_->load(model_dir, device);
}

AudioBuffer Supertonic::decode(const float* latent, int channels, int frames) const {
    return impl_->decode(latent, channels, frames);
}

AudioBuffer Supertonic::decode(const std::vector<float>& latent, int channels,
                               int frames) const {
    return impl_->decode(latent.data(), channels, frames);
}

const SupertonicConfig& Supertonic::config() const { return impl_->cfg; }
bool Supertonic::loaded() const { return impl_->ready; }

}  // namespace brosoundml
