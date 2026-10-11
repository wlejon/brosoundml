// YuE2 VAE — see include/brosoundml/yue2_vae.h. The reference is the yue2
// package's modeling_vae.py (YuE2VAE: Oobleck encoder/decoder from
// stable-audio-tools, SnakeBeta, weight norm); the key map below follows its
// nn.Sequential indices:
//
//   decoder.layers.0          conv k7 latent -> 2048
//   decoder.layers.{1..6}     DecoderBlock: .layers.0 SnakeBeta, .layers.1
//                             ConvTranspose1d, .layers.{2,3,4} ResidualUnit
//   decoder.layers.7 / .8     SnakeBeta, conv k7 64 -> 2 (no bias)
//   encoder.layers.0          conv k7 2 -> 64
//   encoder.layers.{1..6}     EncoderBlock: .layers.{0,1,2} ResidualUnit,
//                             .layers.3 SnakeBeta, .layers.4 strided conv
//   encoder.layers.7 / .8     SnakeBeta, conv k3 2048 -> 128
//   ResidualUnit              .layers.0 SnakeBeta, .1 conv k7 dil d, .2
//                             SnakeBeta, .3 conv k1
#include "brosoundml/yue2_vae.h"

#include "brosoundml/detail/json.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;
namespace j  = detail::json;
namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: yue2 vae: " + msg);
}

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot read '" + path + "'");
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    if (v->dtype != sf::Dtype::F32) fail("tensor '" + name + "' is not F32");
    return *v;
}

const float* f32(const sf::TensorView& v) { return reinterpret_cast<const float*>(v.data); }

std::int64_t floor_div(std::int64_t a, std::int64_t b) {
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

// ── layers ───────────────────────────────────────────────────────────────────

// Conv1d or ConvTranspose1d, weight norm folded. Weights in brotensor's
// layouts: conv (C_out, C_in*k), transposed (C_in, C_out*k) — both torch's
// tensors flattened.
struct Conv {
    bt::Tensor w, b;
    int cin = 0, cout = 0, k = 1, stride = 1, pad = 0, dil = 1;
    bool transpose = false, bias = true;

    int out_len(int L) const {
        if (transpose) return (L - 1) * stride - 2 * pad + dil * (k - 1) + 1;
        return (L + 2 * pad - dil * (k - 1) - 1) / stride + 1;
    }
    // Inclusive input support of output interval [lo, hi] (modeling_vae's
    // _dependency_interval).
    void dep(std::int64_t& lo, std::int64_t& hi) const {
        if (transpose) {
            lo = -floor_div(-(lo + pad - dil * (k - 1)), stride);
            hi = floor_div(hi + pad, stride);
        } else {
            lo = lo * stride - pad;
            hi = hi * stride - pad + dil * (k - 1);
        }
    }
    bt::Tensor forward(const bt::Tensor& x, int L) const {
        bt::Tensor y;
        const bt::Tensor* bp = bias ? &b : nullptr;
        if (transpose)
            bt::conv_transpose1d_forward(x, w, bp, 1, cin, L, cout, k, stride, pad,
                                         /*output_padding=*/0, dil, y);
        else
            bt::conv1d(x, w, bp, 1, cin, L, cout, k, stride, pad, dil, /*groups=*/1, y);
        return y;
    }
};

// Log-scale SnakeBeta, alpha / beta pre-exponentiated at load:
// y = x + sin^2(alpha*x) / beta.
struct Snake {
    bt::Tensor alpha, beta;   // (C,1)
    int C = 0;
    void apply(const bt::Tensor& x, int L, bt::Tensor& y) const {
        bt::snake_forward(x, alpha, &beta, 1, C, L, y);
    }
};

struct ResUnit {
    Snake s1, s2;
    Conv c1, c2;
    void forward(bt::Tensor& x, int L) const {
        bt::Tensor t;
        s1.apply(x, L, t);
        t = c1.forward(t, L);
        s2.apply(t, L, t);
        t = c2.forward(t, L);
        bt::add_inplace(x, t);
    }
    void dep(std::int64_t& lo, std::int64_t& hi) const {
        std::int64_t a = lo, b = hi;
        c2.dep(a, b);
        c1.dep(a, b);
        lo = std::min(lo, a);
        hi = std::max(hi, b);
    }
};

struct DecBlock { Snake s; Conv up; ResUnit r[3]; };
struct EncBlock { ResUnit r[3]; Snake s; Conv down; };

// ── loading ──────────────────────────────────────────────────────────────────

class Loader {
public:
    Loader(const sf::File& f, bt::Device dev) : f_(f), dev_(dev) {}

    // weight_norm(dim=0): w = g * v / ||v||, the norm over every dim but 0.
    Conv conv(const std::string& P, int cin, int cout, int k, int stride, int pad, int dil,
              bool transpose, bool bias) {
        Conv c;
        c.cin = cin; c.cout = cout; c.k = k; c.stride = stride; c.pad = pad; c.dil = dil;
        c.transpose = transpose; c.bias = bias;
        const sf::TensorView& v = need(f_, P + "weight_v");
        const sf::TensorView& g = need(f_, P + "weight_g");
        const int rows = transpose ? cin : cout;
        const int cols = (transpose ? cout : cin) * k;
        if (v.numel() != static_cast<std::int64_t>(rows) * cols || g.numel() != rows)
            fail("tensor '" + P + "weight_v' has an unexpected shape");
        std::vector<float> w(static_cast<std::size_t>(rows) * cols);
        const float* vp = f32(v);
        const float* gp = f32(g);
        for (int r = 0; r < rows; ++r) {
            const float* src = vp + static_cast<std::size_t>(r) * cols;
            double ss = 0;
            for (int i = 0; i < cols; ++i) ss += double(src[i]) * src[i];
            const double s = ss > 0 ? gp[r] / std::sqrt(ss) : 0.0;
            float* dst = w.data() + static_cast<std::size_t>(r) * cols;
            for (int i = 0; i < cols; ++i) dst[i] = static_cast<float>(src[i] * s);
        }
        c.w = bt::Tensor::from_host_on(dev_, w.data(), rows, cols);
        if (bias) c.b = vec(P + "bias", cout);
        return c;
    }

    Snake snake(const std::string& P, int C) {
        Snake s;
        s.C = C;
        s.alpha = exp_vec(P + "alpha", C);
        s.beta  = exp_vec(P + "beta", C);
        return s;
    }

    ResUnit res(const std::string& P, int C, int dil) {
        ResUnit u;
        u.s1 = snake(P + "layers.0.", C);
        u.c1 = conv(P + "layers.1.", C, C, 7, 1, 3 * dil, dil, false, true);
        u.s2 = snake(P + "layers.2.", C);
        u.c2 = conv(P + "layers.3.", C, C, 1, 1, 0, 1, false, true);
        return u;
    }

private:
    bt::Tensor vec(const std::string& name, int n) {
        const sf::TensorView& v = need(f_, name);
        if (v.numel() != n) fail("tensor '" + name + "' has an unexpected shape");
        return bt::Tensor::from_host_on(dev_, f32(v), n, 1);
    }
    bt::Tensor exp_vec(const std::string& name, int n) {
        const sf::TensorView& v = need(f_, name);
        if (v.numel() != n) fail("tensor '" + name + "' has an unexpected shape");
        std::vector<float> e(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) e[i] = std::exp(f32(v)[i]);
        return bt::Tensor::from_host_on(dev_, e.data(), n, 1);
    }

    const sf::File& f_;
    bt::Device dev_;
};

void to_host(const bt::Tensor& t, float* out, std::size_t n) {
    if (t.device == bt::Device::CPU) {
        std::memcpy(out, t.host_f32(), n * sizeof(float));
    } else {
        bt::Tensor c = t.to(bt::Device::CPU);
        std::memcpy(out, c.host_f32(), n * sizeof(float));
    }
}

}  // namespace

// ─── model ───────────────────────────────────────────────────────────────────

struct Yue2Vae::Impl {
    Yue2VaeConfig cfg;
    bt::Device dev = bt::Device::CPU;
    bool is_loaded = false;

    Conv dec_in, dec_out;
    DecBlock dec[6];
    Snake dec_snake;
    int nblocks = 0;

    Conv enc_in, enc_out;
    EncBlock enc[6];
    Snake enc_snake;

    // ── shape algebra ──
    int dec_len(int T) const {
        int L = dec_in.out_len(T);
        for (int i = 0; i < nblocks; ++i) L = dec[i].up.out_len(L);
        return dec_out.out_len(L);
    }
    int enc_len(int S) const {
        int L = enc_in.out_len(S);
        for (int i = 0; i < nblocks; ++i) L = enc[i].down.out_len(L);
        return enc_out.out_len(L);
    }
    // Inclusive latent-frame support of decoder output samples [lo, hi].
    void dec_dep(std::int64_t& lo, std::int64_t& hi) const {
        dec_out.dep(lo, hi);
        for (int i = nblocks - 1; i >= 0; --i) {
            for (int u = 2; u >= 0; --u) dec[i].r[u].dep(lo, hi);
            dec[i].up.dep(lo, hi);
        }
        dec_in.dep(lo, hi);
    }
    // Inclusive input-sample support of encoder output frames [lo, hi].
    void enc_dep(std::int64_t& lo, std::int64_t& hi) const {
        enc_out.dep(lo, hi);
        for (int i = nblocks - 1; i >= 0; --i) {
            enc[i].down.dep(lo, hi);
            for (int u = 2; u >= 0; --u) enc[i].r[u].dep(lo, hi);
        }
        enc_in.dep(lo, hi);
    }
    int required_halo(int core) const {
        std::int64_t lo = 0, hi = std::int64_t(core) * cfg.downsampling_ratio - 1;
        dec_dep(lo, hi);
        return static_cast<int>(std::max<std::int64_t>({0, -lo, hi - core + 1}));
    }
    int enc_halo(int core) const {
        const std::int64_t R = cfg.downsampling_ratio;
        std::int64_t lo = 0, hi = core - 1;
        enc_dep(lo, hi);
        const std::int64_t left = std::max<std::int64_t>(0, -lo);
        const std::int64_t right = std::max<std::int64_t>(0, hi - (core * R - 1));
        return static_cast<int>((std::max(left, right) + R - 1) / R);
    }

    // ── forward ──
    // Latent frames [left, right) of a frame-major buffer -> (2, dec_len) host,
    // planar.
    std::vector<float> decode_window(const float* lat, int left, int right, int* L_out) const {
        const int T = right - left, D = cfg.latent_dim;
        std::vector<float> ncl(static_cast<std::size_t>(D) * T);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < D; ++c)
                ncl[static_cast<std::size_t>(c) * T + t] = lat[static_cast<std::size_t>(left + t) * D + c];
        bt::DeviceScope scope(dev);
        bt::Tensor x = bt::Tensor::from_host_on(dev, ncl.data(), 1, D * T);
        int L = T;
        x = dec_in.forward(x, L);
        L = dec_in.out_len(L);
        for (int i = 0; i < nblocks; ++i) {
            const DecBlock& b = dec[i];
            b.s.apply(x, L, x);
            x = b.up.forward(x, L);
            L = b.up.out_len(L);
            for (const ResUnit& u : b.r) u.forward(x, L);
        }
        dec_snake.apply(x, L, x);
        x = dec_out.forward(x, L);
        L = dec_out.out_len(L);
        std::vector<float> out(static_cast<std::size_t>(cfg.channels) * L);
        to_host(x, out.data(), out.size());
        *L_out = L;
        return out;
    }

    // Planar input samples [s0, s1) of an S-sample signal -> (128, frames)
    // host, channel-major.
    std::vector<float> encode_window(const float* planar, int S, int s0, int s1, int* T_out) const {
        const int n = s1 - s0, C = cfg.channels;
        std::vector<float> in(static_cast<std::size_t>(C) * n);
        for (int c = 0; c < C; ++c)
            std::memcpy(in.data() + static_cast<std::size_t>(c) * n,
                        planar + static_cast<std::size_t>(c) * S + s0, n * sizeof(float));
        bt::DeviceScope scope(dev);
        bt::Tensor x = bt::Tensor::from_host_on(dev, in.data(), 1, C * n);
        int L = n;
        x = enc_in.forward(x, L);
        L = enc_in.out_len(L);
        for (int i = 0; i < nblocks; ++i) {
            const EncBlock& b = enc[i];
            for (const ResUnit& u : b.r) u.forward(x, L);
            b.s.apply(x, L, x);
            x = b.down.forward(x, L);
            L = b.down.out_len(L);
        }
        enc_snake.apply(x, L, x);
        x = enc_out.forward(x, L);
        L = enc_out.out_len(L);
        std::vector<float> out(static_cast<std::size_t>(enc_out.cout) * L);
        to_host(x, out.data(), out.size());
        *T_out = L;
        return out;
    }

    void load(const std::string& dir, bt::Device device, bool decoder_only);
};

void Yue2Vae::Impl::load(const std::string& dir, bt::Device device, bool decoder_only) {
    const fs::path d(dir);
    const std::string config_path  = (d / "config.json").string();
    const std::string weights_path = (d / "model.safetensors").string();
    if (!fs::exists(config_path))  fail("no config.json under '" + dir + "'");
    if (!fs::exists(weights_path)) fail("no model.safetensors under '" + dir + "' (sharded exports are not supported)");
    if (!bt::is_available(device)) fail("requested device is not available");

    const j::Value root = j::parse(slurp(config_path));
    if (!root.is_object()) fail("config.json is not a JSON object");
    Yue2VaeConfig c;
    c.sample_rate        = root.get_int("sample_rate", 48000);
    c.latent_dim         = root.get_int("latent_dim", 64);
    c.downsampling_ratio = root.get_int("downsampling_ratio", 1920);
    c.channels           = root.get_int("audio_channels", 2);
    c.decode_core_frames = root.get_int("decode_core_frames", 1024);
    c.decode_halo_frames = root.get_int("decode_halo_frames", 16);
    const j::Value* dc = root.find("decoder_config");
    if (!dc || !dc->is_object()) fail("config.json: missing decoder_config");
    c.strides       = dc->get_int_array("strides", {2, 2, 4, 4, 5, 6});
    c.c_mults       = dc->get_int_array("c_mults", {1, 2, 4, 8, 16, 32});
    c.channels_base = dc->get_int("channels", 64);
    if (!dc->get_bool("use_snake", true) || dc->get_bool("final_tanh", false) ||
        dc->get_string("snake_type", "vanilla") != "vanilla")
        fail("config.json: only the released decoder (SnakeBeta, no tanh) is supported");
    if (dc->get_int("latent_dim", 64) != c.latent_dim || dc->get_int("out_channels", 2) != c.channels)
        fail("config.json: decoder_config disagrees with latent_dim / audio_channels");
    if (c.strides.size() != c.c_mults.size() || c.strides.empty() || c.strides.size() > 6)
        fail("config.json: strides / c_mults must have the same length (1..6)");
    int prod = 1;
    for (int s : c.strides) prod *= s;
    if (prod != c.downsampling_ratio) fail("config.json: strides do not multiply to downsampling_ratio");

    const sf::File f = sf::File::open(weights_path);
    Loader ld(f, device);
    static const int kDil[3] = {1, 3, 9};
    const int depth = static_cast<int>(c.strides.size());
    std::vector<int> mults = {1};
    mults.insert(mults.end(), c.c_mults.begin(), c.c_mults.end());   // [1] + c_mults
    const int Cb = c.channels_base;
    nblocks = depth;

    // decoder
    dec_in = ld.conv("decoder.layers.0.", c.latent_dim, mults[depth] * Cb, 7, 1, 3, 1, false, true);
    for (int i = depth, n = 0; i >= 1; --i, ++n) {
        const std::string P = "decoder.layers." + std::to_string(n + 1) + ".";
        const int cin = mults[i] * Cb, cout = mults[i - 1] * Cb, s = c.strides[i - 1];
        DecBlock& b = dec[n];
        b.s  = ld.snake(P + "layers.0.", cin);
        b.up = ld.conv(P + "layers.1.", cin, cout, 2 * s, s, (s + 1) / 2, 1, true, true);
        for (int u = 0; u < 3; ++u)
            b.r[u] = ld.res(P + "layers." + std::to_string(u + 2) + ".", cout, kDil[u]);
    }
    dec_snake = ld.snake("decoder.layers." + std::to_string(depth + 1) + ".", mults[0] * Cb);
    dec_out = ld.conv("decoder.layers." + std::to_string(depth + 2) + ".", mults[0] * Cb, c.channels,
                      7, 1, 3, 1, false, /*bias=*/false);

    // encoder
    c.has_encoder = !decoder_only;
    if (!decoder_only) {
        const j::Value* ec = root.find("encoder_config");
        if (!ec || !ec->is_object()) fail("config.json: missing encoder_config");
        if (ec->get_int_array("strides", {}) != c.strides || ec->get_int_array("c_mults", {}) != c.c_mults ||
            ec->get_int("channels", 64) != Cb || !ec->get_bool("use_snake", true))
            fail("config.json: encoder_config does not mirror the decoder");
        const int lat2 = ec->get_int("latent_dim", 2 * c.latent_dim);
        if (lat2 != 2 * c.latent_dim) fail("config.json: encoder latent_dim must be 2 * latent_dim");
        enc_in = ld.conv("encoder.layers.0.", c.channels, mults[0] * Cb, 7, 1, 3, 1, false, true);
        for (int i = 0; i < depth; ++i) {
            const std::string P = "encoder.layers." + std::to_string(i + 1) + ".";
            const int cin = mults[i] * Cb, cout = mults[i + 1] * Cb, s = c.strides[i];
            EncBlock& b = enc[i];
            for (int u = 0; u < 3; ++u)
                b.r[u] = ld.res(P + "layers." + std::to_string(u) + ".", cin, kDil[u]);
            b.s    = ld.snake(P + "layers.3.", cin);
            b.down = ld.conv(P + "layers.4.", cin, cout, 2 * s, s, (s + 1) / 2, 1, false, true);
        }
        enc_snake = ld.snake("encoder.layers." + std::to_string(depth + 1) + ".", mults[depth] * Cb);
        enc_out = ld.conv("encoder.layers." + std::to_string(depth + 2) + ".", mults[depth] * Cb, lat2,
                          3, 1, 1, 1, false, true);
    }

    cfg = c;
    dev = device;
    is_loaded = true;
    if (cfg.decode_halo_frames < required_halo(cfg.decode_core_frames))
        cfg.decode_halo_frames = required_halo(cfg.decode_core_frames);
}

// ─── public Yue2Vae ──────────────────────────────────────────────────────────

Yue2Vae::Yue2Vae() : impl_(new Impl) {}
Yue2Vae::~Yue2Vae() = default;
Yue2Vae::Yue2Vae(Yue2Vae&&) noexcept = default;
Yue2Vae& Yue2Vae::operator=(Yue2Vae&&) noexcept = default;

void Yue2Vae::load(const std::string& dir, bt::Device device, bool decoder_only) {
    auto fresh = std::make_unique<Impl>();
    fresh->load(dir, device, decoder_only);   // throws -> the current model is untouched
    impl_ = std::move(fresh);
}

StereoAudio Yue2Vae::decode(const float* latents, int frames) const {
    if (!loaded()) fail("decode: no model loaded");
    if (!latents || frames <= 0) fail("decode: empty latents");
    for (std::size_t i = 0, n = std::size_t(frames) * impl_->cfg.latent_dim; i < n; ++i)
        if (!std::isfinite(latents[i])) fail("decode: latents contain non-finite values");
    StereoAudio a;
    a.sample_rate = impl_->cfg.sample_rate;
    a.channels = impl_->cfg.channels;
    a.samples = impl_->decode_window(latents, 0, frames, &a.frames);
    return a;
}

StereoAudio Yue2Vae::decode_tiled(const float* latents, int frames, int core, int halo,
                                  const std::function<void(int, int)>& on_progress,
                                  const CancelCheck& cancel) const {
    if (!loaded()) fail("decode_tiled: no model loaded");
    if (!latents || frames <= 0) fail("decode_tiled: empty latents");
    const Impl& m = *impl_;
    if (core <= 0) core = m.cfg.decode_core_frames;
    if (halo <= 0) halo = std::max(m.cfg.decode_halo_frames, m.required_halo(core));
    if (halo < m.required_halo(core))
        fail("decode_tiled: halo_frames must be at least " + std::to_string(m.required_halo(core)));
    const int R = m.cfg.downsampling_ratio, C = m.cfg.channels;
    const int total = m.dec_len(frames);
    StereoAudio a;
    a.sample_rate = m.cfg.sample_rate;
    a.channels = C;
    a.frames = total;
    a.samples.assign(static_cast<std::size_t>(C) * total, 0.0f);
    const int tiles = (frames + core - 1) / core;
    int done_samples = 0;
    for (int t = 0, start = 0; start < frames; ++t, start += core) {
        if (cancel && cancel()) break;
        const int end = std::min(frames, start + core);
        const int left = std::max(0, start - halo), right = std::min(frames, end + halo);
        int L = 0;
        const std::vector<float> tile = m.decode_window(latents, left, right, &L);
        const int out0 = start * R, out1 = std::min(end * R, total);
        const int crop = (start - left) * R;
        if (crop + (out1 - out0) > L) fail("decode_tiled: tile did not cover its core");
        for (int c = 0; c < C; ++c)
            std::memcpy(a.channel(c) + out0, tile.data() + std::size_t(c) * L + crop,
                        std::size_t(out1 - out0) * sizeof(float));
        done_samples = out1;
        if (on_progress) on_progress(t + 1, tiles);
    }
    if (done_samples < total) {   // cancelled: keep the finished prefix
        StereoAudio cut;
        cut.sample_rate = a.sample_rate;
        cut.channels = C;
        cut.frames = done_samples;
        cut.samples.resize(std::size_t(C) * done_samples);
        for (int c = 0; c < C; ++c)
            std::memcpy(cut.channel(c), a.channel(c), std::size_t(done_samples) * sizeof(float));
        return cut;
    }
    return a;
}

std::vector<float> Yue2Vae::encode(const StereoAudio& audio, int* frames_out) const {
    if (!loaded()) fail("encode: no model loaded");
    if (!has_encoder()) fail("encode: the encoder was not loaded (decoder_only)");
    const Impl& m = *impl_;
    const int C = m.cfg.channels, R = m.cfg.downsampling_ratio, D = m.cfg.latent_dim;
    if (audio.frames <= 0 || audio.channels < 1 ||
        audio.samples.size() < std::size_t(audio.channels) * audio.frames)
        fail("encode: empty audio");

    // To planar C-channel audio at the model rate.
    std::vector<float> planar;
    int S = audio.frames;
    for (int c = 0; c < C; ++c) {
        const int src = std::min(c, audio.channels - 1);
        AudioBuffer ch(std::vector<float>(audio.channel(src), audio.channel(src) + audio.frames),
                       audio.sample_rate);
        if (audio.sample_rate != m.cfg.sample_rate) ch = resample(ch, m.cfg.sample_rate);
        if (c == 0) { S = static_cast<int>(ch.samples.size()); planar.resize(std::size_t(C) * S); }
        std::copy_n(ch.samples.begin(), std::min<std::size_t>(S, ch.samples.size()),
                    planar.begin() + std::size_t(c) * S);
    }
    if (S < R) fail("encode: need at least " + std::to_string(R) + " samples");
    for (float v : planar)
        if (!std::isfinite(v)) fail("encode: audio contains non-finite values");

    // Exact tiles, as decode_tiled: each tile carries enough samples of context
    // each side that its core frames match the whole-signal encoder's.
    const int T = m.enc_len(S);
    const int core = 512, halo = m.enc_halo(core);
    std::vector<float> lat(std::size_t(T) * D);
    for (int start = 0; start < T; start += core) {
        const int end = std::min(T, start + core);
        const int lf = std::max(0, start - halo);
        const int s0 = lf * R;
        const int s1 = std::min<std::int64_t>(S, std::int64_t(end + halo) * R) == S
                           ? S : (end + halo) * R;
        int Tt = 0;
        const std::vector<float> z = m.encode_window(planar.data(), S, s0, s1, &Tt);
        const int off = start - lf;
        if (off + (end - start) > Tt) fail("encode: tile did not cover its core");
        for (int t = start; t < end; ++t)
            for (int c = 0; c < D; ++c)   // the first D channels are the posterior mean
                lat[std::size_t(t) * D + c] = z[std::size_t(c) * Tt + (t - lf)];
    }
    if (frames_out) *frames_out = T;
    return lat;
}

std::vector<float> Yue2Vae::encode(const AudioBuffer& mono, int* frames_out) const {
    StereoAudio a;
    a.channels = 1;
    a.frames = static_cast<int>(mono.samples.size());
    a.sample_rate = mono.sample_rate;
    a.samples = mono.samples;
    return encode(a, frames_out);
}

int Yue2Vae::required_halo(int core_frames) const {
    if (!loaded()) fail("required_halo: no model loaded");
    return impl_->required_halo(core_frames > 0 ? core_frames : impl_->cfg.decode_core_frames);
}

int Yue2Vae::output_length(int frames) const {
    if (!loaded()) fail("output_length: no model loaded");
    if (frames <= 0) fail("output_length: frames must be positive");
    return impl_->dec_len(frames);
}

const Yue2VaeConfig& Yue2Vae::config() const {
    static const Yue2VaeConfig kDefault{};
    return impl_ ? impl_->cfg : kDefault;
}
bool Yue2Vae::loaded() const { return impl_ && impl_->is_loaded; }
bool Yue2Vae::has_encoder() const { return loaded() && impl_->cfg.has_encoder; }
bt::Device Yue2Vae::device() const { return impl_ ? impl_->dev : bt::Device::CPU; }

// ─── StereoAudio ─────────────────────────────────────────────────────────────

void StereoAudio::write_wav(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) fail("write_wav: cannot open '" + path + "'");
    auto u32 = [&](std::uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](std::uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
    const std::uint32_t data = std::uint32_t(frames) * channels * 2;
    out.write("RIFF", 4); u32(36 + data); out.write("WAVE", 4);
    out.write("fmt ", 4); u32(16); u16(1); u16(std::uint16_t(channels));
    u32(std::uint32_t(sample_rate)); u32(std::uint32_t(sample_rate) * channels * 2);
    u16(std::uint16_t(channels * 2)); u16(16);
    out.write("data", 4); u32(data);
    std::vector<std::int16_t> pcm(std::size_t(frames) * channels);
    for (int i = 0; i < frames; ++i)
        for (int c = 0; c < channels; ++c) {
            const float v = std::clamp(channel(c)[i], -1.0f, 1.0f);
            pcm[std::size_t(i) * channels + c] = static_cast<std::int16_t>(std::lrint(v * 32767.0f));
        }
    out.write(reinterpret_cast<const char*>(pcm.data()), std::streamsize(pcm.size() * 2));
    if (!out) fail("write_wav: write failed for '" + path + "'");
}

AudioBuffer StereoAudio::mono() const {
    std::vector<float> m(static_cast<std::size_t>(frames), 0.0f);
    if (channels > 0) {
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < frames; ++i) m[i] += channel(c)[i];
        for (float& v : m) v /= float(channels);
    }
    return AudioBuffer(std::move(m), sample_rate);
}

// ─── Yue2VaeStream ───────────────────────────────────────────────────────────

Yue2VaeStream::Yue2VaeStream(const Yue2Vae& vae, ChunkFn on_chunk, int core_frames, int halo_frames,
                             int first_core_frames)
    : vae_(&vae), on_chunk_(std::move(on_chunk)), core_(core_frames > 0 ? core_frames : 50),
      first_(std::min(core_, first_core_frames > 0 ? first_core_frames : 8)) {
    if (!vae.loaded()) fail("stream: the VAE is not loaded");
    // The halo is set by the largest tile; smaller tiles need no more.
    const int need_halo = std::max(vae.required_halo(core_), vae.required_halo(first_));
    halo_ = halo_frames > 0 ? halo_frames : need_halo;
    if (halo_ < need_halo) fail("stream: halo_frames must be at least " + std::to_string(need_halo));
}

Yue2VaeStream::~Yue2VaeStream() = default;

void Yue2VaeStream::push(int frame0, int frames, const float* data) {
    if (finished_) fail("stream: push after finish");
    if (frame0 != pushed_)
        fail("stream: expected frame " + std::to_string(pushed_) + ", got " + std::to_string(frame0));
    if (frames <= 0) return;
    if (!data) fail("stream: null latents");
    const int D = vae_->config().latent_dim;
    latents_.insert(latents_.end(), data, data + std::size_t(frames) * D);
    pushed_ += frames;
    decode_ready(false);
}

void Yue2VaeStream::finish() {
    if (finished_) return;
    finished_ = true;
    decode_ready(true);
}

void Yue2VaeStream::reset() {
    latents_.clear();
    pushed_ = decoded_ = last_tile_ = 0;
    emitted_ = 0;
    finished_ = false;
}

void Yue2VaeStream::decode_ready(bool final_pass) {
    const Yue2Vae::Impl& m = *vae_->impl_;
    const int R = m.cfg.downsampling_ratio, C = m.cfg.channels;
    while (decoded_ < pushed_) {
        const int start = decoded_;
        int end = std::min(pushed_, start + core_);
        if (!final_pass) {
            // Frames before pushed_ - halo_ have all their right context.
            end = std::min(end, pushed_ - halo_);
            const int want = last_tile_ ? std::min(core_, 2 * last_tile_) : first_;
            if (end - start < want) break;
        }
        const int left = std::max(0, start - halo_), right = std::min(pushed_, end + halo_);
        int L = 0;
        const std::vector<float> tile = m.decode_window(latents_.data(), left, right, &L);
        // Interior cores end on a frame boundary; the song's last core ends at
        // the natural length (1920*T - 64).
        const std::int64_t out0 = std::int64_t(start) * R;
        const std::int64_t out1 = (final_pass && end == pushed_) ? m.dec_len(pushed_)
                                                                 : std::int64_t(end) * R;
        const int crop = (start - left) * R;
        const int n = static_cast<int>(out1 - out0);
        if (crop + n > L) fail("stream: tile did not cover its core");
        StereoAudio chunk;
        chunk.sample_rate = m.cfg.sample_rate;
        chunk.channels = C;
        chunk.frames = n;
        chunk.samples.resize(std::size_t(C) * n);
        for (int c = 0; c < C; ++c)
            std::memcpy(chunk.channel(c), tile.data() + std::size_t(c) * L + crop, std::size_t(n) * sizeof(float));
        decoded_ = end;
        last_tile_ = end - start;
        const std::int64_t s0 = emitted_;
        emitted_ += n;
        if (on_chunk_) on_chunk_(chunk, s0);
    }
}

int Yue2VaeStream::frames_pushed() const { return pushed_; }
int Yue2VaeStream::frames_decoded() const { return decoded_; }
std::int64_t Yue2VaeStream::samples_emitted() const { return emitted_; }
bool Yue2VaeStream::finished() const { return finished_; }

}  // namespace brosoundml
