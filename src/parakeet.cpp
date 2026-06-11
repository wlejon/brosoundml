#include "brosoundml/parakeet.h"

#include "brosoundml/detail/json.h"
#include "parakeet_modules.h"
#include "qwen_tts_device.h"   // qtd:: device-neutral helpers

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace fs = std::filesystem;
namespace j  = detail::json;
namespace bt = brotensor;
namespace sf = brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

std::string slurp(const std::string& path, const std::string& where) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail(where, "cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int int_at(const j::Value& obj, const std::string& key, const std::string& where) {
    const j::Value* v = obj.find(key);
    if (!v) fail(where, "config.json missing required key '" + key + "'");
    return static_cast<int>(v->as_number());
}

ParakeetConfig parse_config(const std::string& path) {
    const std::string where = "Parakeet::load";
    const std::string text = slurp(path, where);
    const j::Value    root = j::parse(text);
    if (!root.is_object()) fail(where, "config.json is not a JSON object");

    ParakeetConfig c;
    c.vocab_size           = int_at(root, "vocab_size", where);
    c.blank_token_id       = root.get_int("blank_token_id", 8192);
    c.pad_token_id         = root.get_int("pad_token_id", 2);
    c.decoder_hidden_size  = int_at(root, "decoder_hidden_size", where);
    c.num_decoder_layers   = int_at(root, "num_decoder_layers", where);
    c.max_symbols_per_step = root.get_int("max_symbols_per_step", 10);
    c.durations            = root.get_int_array("durations", {0, 1, 2, 3, 4});

    const j::Value* enc = root.find("encoder_config");
    if (!enc || !enc->is_object())
        fail(where, "config.json missing object 'encoder_config'");
    ParakeetEncoderConfig& e = c.encoder;
    e.num_mel_bins                 = int_at(*enc, "num_mel_bins", where);
    e.hidden_size                  = int_at(*enc, "hidden_size", where);
    e.num_hidden_layers            = int_at(*enc, "num_hidden_layers", where);
    e.num_attention_heads          = int_at(*enc, "num_attention_heads", where);
    e.intermediate_size            = int_at(*enc, "intermediate_size", where);
    e.conv_kernel_size             = enc->get_int("conv_kernel_size", 9);
    e.subsampling_factor           = enc->get_int("subsampling_factor", 8);
    e.subsampling_conv_channels    = enc->get_int("subsampling_conv_channels", 256);
    e.subsampling_conv_kernel_size = enc->get_int("subsampling_conv_kernel_size", 3);
    e.subsampling_conv_stride      = enc->get_int("subsampling_conv_stride", 2);
    e.max_position_embeddings      = enc->get_int("max_position_embeddings", 5000);
    e.scale_input                  = enc->get_bool("scale_input", false);
    e.attention_bias               = enc->get_bool("attention_bias", false);
    e.convolution_bias             = enc->get_bool("convolution_bias", false);
    return c;
}

// Weight-upload helpers (FP32 on `dev`, widening F16/BF16).
const sf::TensorView& need(const sf::File& f, const std::string& name,
                           const std::string& where) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail(where, "missing tensor '" + name + "'");
    return *v;
}
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols,
              bt::Device dev, const std::string& where) {
    bt::Tensor t;
    {
        bt::DeviceScope cpu(bt::Device::CPU);
        sf::upload_compute_checked(need(f, name, where), rows, cols, t, name);
    }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
bt::Tensor up_vec(const sf::File& f, const std::string& name, int n,
                  bt::Device dev, const std::string& where) {
    return up(f, name, n, 1, dev, where);
}

}  // namespace

// ─── ParakeetPrediction ────────────────────────────────────────────────────

void ParakeetPrediction::load(const sf::File& f, const ParakeetConfig& cfg,
                              bt::Device dev) {
    const std::string where = "ParakeetPrediction::load";
    hidden   = cfg.decoder_hidden_size;
    n_layers = cfg.num_decoder_layers;
    device   = dev;

    embedding = up(f, "decoder.embedding.weight", cfg.vocab_size, hidden, dev, where);
    w_ih.resize(static_cast<std::size_t>(n_layers));
    w_hh.resize(static_cast<std::size_t>(n_layers));
    b_ih.resize(static_cast<std::size_t>(n_layers));
    b_hh.resize(static_cast<std::size_t>(n_layers));
    for (int l = 0; l < n_layers; ++l) {
        const std::string s = "decoder.lstm.";
        const std::string suf = "_l" + std::to_string(l);
        const int in = hidden;   // input width is `hidden` for every layer
        w_ih[static_cast<std::size_t>(l)] =
            up(f, s + "weight_ih" + suf, 4 * hidden, in, dev, where);
        w_hh[static_cast<std::size_t>(l)] =
            up(f, s + "weight_hh" + suf, 4 * hidden, hidden, dev, where);
        b_ih[static_cast<std::size_t>(l)] =
            up_vec(f, s + "bias_ih" + suf, 4 * hidden, dev, where);
        b_hh[static_cast<std::size_t>(l)] =
            up_vec(f, s + "bias_hh" + suf, 4 * hidden, dev, where);
    }
    proj_w = up(f, "decoder.decoder_projector.weight", hidden, hidden, dev, where);
    proj_b = up_vec(f, "decoder.decoder_projector.bias", hidden, dev, where);
}

ParakeetPrediction::State ParakeetPrediction::init_state() const {
    State st;
    st.h.resize(static_cast<std::size_t>(n_layers));
    st.c.resize(static_cast<std::size_t>(n_layers));
    for (int l = 0; l < n_layers; ++l) {
        st.h[static_cast<std::size_t>(l)] =
            bt::Tensor::zeros_on(device, 1, hidden, bt::Dtype::FP32);
        st.c[static_cast<std::size_t>(l)] =
            bt::Tensor::zeros_on(device, 1, hidden, bt::Dtype::FP32);
    }
    return st;
}

void ParakeetPrediction::step(int32_t token_id, State& st,
                              bt::Tensor& out) const {
    bt::DeviceScope scope(device);
    const int H = hidden;

    // Embed the token -> (1, H).
    bt::Tensor x = qtd::gather_rows(embedding, std::vector<int32_t>{token_id});

    for (int l = 0; l < n_layers; ++l) {
        const std::size_t li = static_cast<std::size_t>(l);
        // gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh  -> (1, 4H)
        bt::Tensor g;
        bt::linear_forward_batched(w_ih[li], b_ih[li], x, g);
        bt::Tensor gh;
        bt::linear_forward_batched(w_hh[li], b_hh[li], st.h[li], gh);
        bt::add_inplace(g, gh);

        // Slice gates (i,f,g,o) — contiguous columns of the (1, 4H) row.
        float* gd = static_cast<float*>(g.data);
        bt::Tensor gi = bt::Tensor::view(device, gd + 0 * H, 1, H, bt::Dtype::FP32);
        bt::Tensor gf = bt::Tensor::view(device, gd + 1 * H, 1, H, bt::Dtype::FP32);
        bt::Tensor gg = bt::Tensor::view(device, gd + 2 * H, 1, H, bt::Dtype::FP32);
        bt::Tensor go = bt::Tensor::view(device, gd + 3 * H, 1, H, bt::Dtype::FP32);
        bt::sigmoid_forward(gi, gi);
        bt::sigmoid_forward(gf, gf);
        bt::tanh_forward(gg, gg);
        bt::sigmoid_forward(go, go);

        // c = f * c_prev + i * g ; h = o * tanh(c).
        bt::Tensor c_new = gf.clone();
        bt::mul_inplace(c_new, st.c[li]);
        bt::Tensor ig = gi.clone();
        bt::mul_inplace(ig, gg);
        bt::add_inplace(c_new, ig);

        bt::Tensor h_new = c_new.clone();
        bt::tanh_forward(h_new, h_new);
        bt::mul_inplace(h_new, go);

        st.c[li] = std::move(c_new);
        st.h[li] = h_new.clone();
        x = std::move(h_new);   // feed this layer's output to the next
    }

    // decoder_projector(last hidden) -> (1, H).
    qtd::linear(proj_w, &proj_b, x, out);
}

// ─── ParakeetJoint ─────────────────────────────────────────────────────────

void ParakeetJoint::load(const sf::File& f, const ParakeetConfig& cfg,
                         bt::Device dev) {
    const std::string where = "ParakeetJoint::load";
    const int H  = cfg.decoder_hidden_size;
    const int nd = static_cast<int>(cfg.durations.size());
    device = dev;
    head_w = up(f, "joint.head.weight", cfg.vocab_size + nd, H, dev, where);
    head_b = up_vec(f, "joint.head.bias", cfg.vocab_size + nd, dev, where);
}

void ParakeetJoint::forward(const bt::Tensor& enc_proj_row,
                            const bt::Tensor& dec_proj,
                            bt::Tensor& out) const {
    bt::DeviceScope scope(device);
    bt::Tensor h = enc_proj_row.clone();
    bt::add_inplace(h, dec_proj);
    bt::relu_forward(h, h);
    qtd::linear(head_w, &head_b, h, out);   // (1, V+nd)
}

// ─── Parakeet::Impl ────────────────────────────────────────────────────────

struct Parakeet::Impl {
    ParakeetConfig    config;
    bt::Device        device = bt::Device::CPU;
    bool              loaded = false;

    ParakeetEncoder    encoder;
    bt::Tensor         enc_proj_w, enc_proj_b;   // encoder_projector (640,1024)
    ParakeetPrediction prediction;
    ParakeetJoint      joint;

    // Greedy joint step: download the (V+nd) logits row and argmax token /
    // duration. Returns the token id and the chosen duration value.
    void joint_argmax(const bt::Tensor& enc_proj_row, const bt::Tensor& dec_proj,
                      int32_t& token, int& duration) const {
        bt::Tensor logits;
        joint.forward(enc_proj_row, dec_proj, logits);
        const int V  = config.vocab_size;
        const int nd = static_cast<int>(config.durations.size());
        std::vector<float> host(static_cast<std::size_t>(V) + nd);
        qtd::to_host(logits, host.data());

        int best_t = 0; float best_tv = host[0];
        for (int v = 1; v < V; ++v)
            if (host[static_cast<std::size_t>(v)] > best_tv) {
                best_tv = host[static_cast<std::size_t>(v)]; best_t = v;
            }
        int best_d = 0; float best_dv = host[static_cast<std::size_t>(V)];
        for (int d = 1; d < nd; ++d)
            if (host[static_cast<std::size_t>(V + d)] > best_dv) {
                best_dv = host[static_cast<std::size_t>(V + d)]; best_d = d;
            }
        token    = static_cast<int32_t>(best_t);
        duration = config.durations[static_cast<std::size_t>(best_d)];
    }
};

Parakeet::Parakeet() : impl_(std::make_unique<Impl>()) {}
Parakeet::~Parakeet() = default;
Parakeet::Parakeet(Parakeet&&) noexcept = default;
Parakeet& Parakeet::operator=(Parakeet&&) noexcept = default;

void Parakeet::load(const std::string& model_dir, bt::Device device) {
    bt::init();

    const fs::path dir         = model_dir;
    const fs::path config_path = dir / "config.json";
    const fs::path weight_path = dir / "model.safetensors";
    if (!fs::exists(config_path))
        fail("Parakeet::load", "no config.json under '" + model_dir + "'");
    if (!fs::exists(weight_path))
        fail("Parakeet::load", "no model.safetensors under '" + model_dir + "'");

    impl_->config = parse_config(config_path.string());
    impl_->device = device;

    sf::File weights = sf::File::open(weight_path.string());
    const ParakeetConfig& c = impl_->config;

    impl_->encoder.load(weights, c.encoder, device);
    impl_->enc_proj_w = up(weights, "encoder_projector.weight",
                           c.decoder_hidden_size, c.encoder.hidden_size,
                           device, "Parakeet::load");
    impl_->enc_proj_b = up_vec(weights, "encoder_projector.bias",
                               c.decoder_hidden_size, device, "Parakeet::load");
    impl_->prediction.load(weights, c, device);
    impl_->joint.load(weights, c, device);

    impl_->loaded = true;
}

Parakeet::Transcription Parakeet::transcribe(const AudioBuffer& audio) const {
    return transcribe(audio, TranscribeOptions{});
}

Parakeet::Transcription Parakeet::transcribe(const AudioBuffer& audio,
                                             const TranscribeOptions& opts) const {
    if (!impl_->loaded)
        fail("Parakeet::transcribe", "no model loaded; call Parakeet::load() first");
    if (audio.samples.empty())
        fail("Parakeet::transcribe", "audio buffer is empty");
    if (audio.sample_rate != impl_->config.sample_rate)
        fail("Parakeet::transcribe",
             "audio.sample_rate must be 16000 Hz; resampling is the caller's "
             "responsibility");

    const ParakeetConfig& cfg = impl_->config;
    const bt::Device dev = impl_->device;
    bt::DeviceScope scope(dev);
    const int H = cfg.decoder_hidden_size;

    // ── Encoder + projector: audio -> (T, 640) ──
    bt::Tensor enc;
    impl_->encoder.forward(audio, enc);                    // (T, 1024)
    const int T = enc.rows;
    bt::Tensor enc_proj;
    bt::linear_forward_batched(impl_->enc_proj_w, impl_->enc_proj_b,
                               enc, enc_proj);              // (T, 640)

    // ── Greedy TDT decode ──
    ParakeetPrediction::State st = impl_->prediction.init_state();
    bt::Tensor dec_proj;
    // Initial decoder output from the SOS = blank token.
    impl_->prediction.step(cfg.blank_token_id, st, dec_proj);

    Transcription out;
    const int max_new = opts.max_new_tokens;
    const int max_sym = cfg.max_symbols_per_step;

    int time = 0;
    bool stop = false;
    while (time < T && !stop) {
        if (opts.cancel && opts.cancel()) break;

        // Current encoder frame as a (1, H) view into enc_proj.
        bt::Tensor enc_row = bt::Tensor::view(
            dev, static_cast<float*>(enc_proj.data) +
                     static_cast<std::size_t>(time) * H,
            1, H, bt::Dtype::FP32);

        int  symbols  = 0;
        bool advanced = false;
        while (symbols < max_sym) {
            int32_t token; int duration;
            impl_->joint_argmax(enc_row, dec_proj, token, duration);

            if (token == cfg.blank_token_id) {
                if (duration == 0) duration = 1;   // never stall on a blank
                time += duration;
                advanced = true;
                break;
            }

            out.token_ids.push_back(token);
            out.token_frames.push_back(time);
            if (opts.on_token) opts.on_token(token);
            impl_->prediction.step(token, st, dec_proj);   // advance predictor
            ++symbols;
            time += duration;

            if (max_new > 0 &&
                static_cast<int>(out.token_ids.size()) >= max_new) {
                stop = true;
                advanced = true;
                break;
            }
            if (duration > 0) { advanced = true; break; }
            // duration == 0: stay on this frame, emit another symbol.
        }
        if (!advanced) ++time;   // forced progress after max_symbols at one frame
    }

    return out;
}

const ParakeetConfig& Parakeet::config() const { return impl_->config; }
bool Parakeet::loaded() const { return impl_->loaded; }

}  // namespace brosoundml
