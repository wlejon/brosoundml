#include "brosoundml/parakeet.h"

#include "brosoundml/detail/json.h"
#include "parakeet_modules.h"
#include "qwen_tts_device.h"   // qtd:: device-neutral helpers

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
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

// Env-gated stage profiling, in `kokoro_profile_mark`'s convention:
// BROSOUNDML_PARAKEET_PROFILE=1 prints one line per transcribe() to stderr.
// Read once — this is asked on a path that runs a few hundred times a second.
bool parakeet_profile_enabled() {
    static const bool on = []() {
        const char* v = std::getenv("BROSOUNDML_PARAKEET_PROFILE");
        return v && v[0] && v[0] != '0';
    }();
    return on;
}

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

    // The shared transcribe body: validate, encode, then run the greedy TDT
    // decode into the caller-supplied prediction state `st` (re-initialised to
    // the zero state on entry, so every call is a self-contained one-shot
    // decode). Both the legacy and the session transcribe() overloads funnel
    // through here. Defined out-of-line below.
    Parakeet::Transcription run_transcribe(ParakeetPrediction::State& st,
                                           const AudioBuffer& audio,
                                           const Parakeet::TranscribeOptions& opts) const;
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

Parakeet::Transcription Parakeet::Impl::run_transcribe(
    ParakeetPrediction::State& st,
    const AudioBuffer& audio,
    const Parakeet::TranscribeOptions& opts) const {
    if (!loaded)
        fail("Parakeet::transcribe", "no model loaded; call Parakeet::load() first");
    if (audio.samples.empty())
        fail("Parakeet::transcribe", "audio buffer is empty");
    if (audio.sample_rate <= 0)
        fail("Parakeet::transcribe", "audio.sample_rate must be positive");
    if (audio.sample_rate != config.sample_rate) {
        const AudioBuffer native = resample(audio, config.sample_rate);
        return run_transcribe(st, native, opts);
    }

    const ParakeetConfig& cfg = config;
    const bt::Device dev = device;
    bt::DeviceScope scope(dev);
    const int H = cfg.decoder_hidden_size;

    // Env-gated (BROSOUNDML_PARAKEET_PROFILE=1) stage split, in kokoro's
    // convention. **Two costs of completely different shapes live in this
    // function** — an encoder whose self-attention is quadratic in the window,
    // and a greedy decode that is a sequence of tiny steps each ending in a
    // device→host download — and which of them dominates decides what is worth
    // optimising. Guessing got it wrong once already.
    const bool prof = parakeet_profile_enabled();
    const auto clock_now = [] { return std::chrono::steady_clock::now(); };
    const auto since_ms = [](std::chrono::steady_clock::time_point a,
                             std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto t_begin = clock_now();

    // ── Encoder + projector: audio -> (T, 640) ──
    bt::Tensor enc;
    encoder.forward(audio, enc);                           // (T, 1024)
    const int T = enc.rows;
    bt::Tensor enc_proj;
    bt::linear_forward_batched(enc_proj_w, enc_proj_b,
                               enc, enc_proj);              // (T, 640)

    if (prof) bt::sync(dev);
    const auto t_encoded = clock_now();

    // ── Greedy TDT decode ──
    // One-shot: re-init the prediction state to the zero state so a reused
    // session decodes this utterance independently of any prior one.
    st = prediction.init_state();
    bt::Tensor dec_proj;
    // Initial decoder output from the SOS = blank token.
    prediction.step(cfg.blank_token_id, st, dec_proj);

    Transcription out;
    const int max_new = opts.max_new_tokens;
    const int max_sym = cfg.max_symbols_per_step;

    int time = 0;
    long long steps = 0;      // joint evaluations, which is what a decode costs
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
            ++steps;
            joint_argmax(enc_row, dec_proj, token, duration);

            if (token == cfg.blank_token_id) {
                if (duration == 0) duration = 1;   // never stall on a blank
                time += duration;
                advanced = true;
                break;
            }

            out.token_ids.push_back(token);
            out.token_frames.push_back(time);
            out.token_durations.push_back(duration);
            if (opts.on_token) opts.on_token(token);
            prediction.step(token, st, dec_proj);          // advance predictor
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

    if (prof) {
        bt::sync(dev);
        const auto t_end = clock_now();
        const double enc_ms = since_ms(t_begin, t_encoded);
        const double dec_ms = since_ms(t_encoded, t_end);
        const double secs = audio.duration_seconds();
        const double total = enc_ms + dec_ms;
        std::fprintf(stderr,
                     "[parakeet-prof] audio %6.2f s · encoder %8.2f ms · "
                     "decode %8.2f ms · frames %5d · steps %6lld · tokens %5zu · "
                     "%6.3f ms a step · %6.2fx realtime\n",
                     secs, enc_ms, dec_ms, T, steps, out.token_ids.size(),
                     steps ? dec_ms / static_cast<double>(steps) : 0.0,
                     total > 0.0 ? secs * 1000.0 / total : 0.0);
    }

    return out;
}

// ── Forced alignment over the TDT lattice ──
//
// The joint network is evaluated for every (prefix length u, encoder frame t)
// pair in row blocks — relu(enc_proj[t] + dec_proj[u]) is assembled on the
// host, one batched head GEMM runs on the model's device, and each logits row
// is reduced on the host to the three numbers the lattice needs: the target
// token's and the blank's log-softmax over the vocabulary, and the duration
// log-softmax. A Viterbi pass then walks frames in order; a zero-duration
// token keeps the frame and moves to the next prefix, which the inner loop
// over u (ascending) reaches in the same sweep.
Parakeet::Alignment Parakeet::align(const AudioBuffer& audio,
                                    const std::vector<int32_t>& token_ids) const {
    const Impl& m = *impl_;
    const char* where = "Parakeet::align";
    if (!m.loaded) fail(where, "no model loaded; call Parakeet::load() first");
    if (audio.samples.empty()) fail(where, "audio buffer is empty");
    if (audio.sample_rate <= 0) fail(where, "audio.sample_rate must be positive");
    if (token_ids.empty()) fail(where, "token list is empty");
    const ParakeetConfig& cfg = m.config;
    for (int32_t id : token_ids)
        if (id < 0 || id >= cfg.vocab_size || id == cfg.blank_token_id)
            fail(where, "token id " + std::to_string(id) + " is outside the vocabulary");
    if (audio.sample_rate != cfg.sample_rate)
        return align(resample(audio, cfg.sample_rate), token_ids);

    const bt::Device dev = m.device;
    bt::DeviceScope scope(dev);
    const int H  = cfg.decoder_hidden_size;
    const int V  = cfg.vocab_size;
    const int nd = static_cast<int>(cfg.durations.size());
    const int U  = static_cast<int>(token_ids.size());

    bt::Tensor enc;
    m.encoder.forward(audio, enc);
    const int T = enc.rows;
    bt::Tensor enc_proj;
    bt::linear_forward_batched(m.enc_proj_w, m.enc_proj_b, enc, enc_proj);
    std::vector<float> enc_host(static_cast<std::size_t>(T) * H);
    qtd::to_host(enc_proj, enc_host.data());

    std::vector<float> dec_host(static_cast<std::size_t>(U + 1) * H);
    {
        ParakeetPrediction::State st = m.prediction.init_state();
        bt::Tensor dec;
        m.prediction.step(cfg.blank_token_id, st, dec);
        qtd::to_host(dec, dec_host.data());
        for (int u = 0; u < U; ++u) {
            m.prediction.step(token_ids[static_cast<std::size_t>(u)], st, dec);
            qtd::to_host(dec, dec_host.data() + static_cast<std::size_t>(u + 1) * H);
        }
    }

    const std::size_t R = static_cast<std::size_t>(U + 1) * T;
    std::vector<float> lp_tok(R), lp_blank(R), lp_dur(R * nd);
    const int W = V + nd;
    const std::size_t block = 2048;
    std::vector<float> hid, logits;
    for (std::size_t r0 = 0; r0 < R; r0 += block) {
        const std::size_t n = std::min(block, R - r0);
        hid.resize(n * H);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t r = r0 + i;
            const float* e = enc_host.data() + (r % T) * H;
            const float* d = dec_host.data() + (r / T) * H;
            float* h = hid.data() + i * H;
            for (int k = 0; k < H; ++k) h[k] = std::max(0.0f, e[k] + d[k]);
        }
        bt::Tensor x = bt::Tensor::from_host_on(dev, hid.data(), static_cast<int>(n), H);
        bt::Tensor y;
        qtd::linear(m.joint.head_w, &m.joint.head_b, x, y);
        logits.resize(n * W);
        qtd::to_host(y, logits.data());
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t r = r0 + i;
            const int u = static_cast<int>(r / T);
            const float* l = logits.data() + i * W;
            float mx = l[0];
            for (int v = 1; v < V; ++v) mx = std::max(mx, l[v]);
            double den = 0.0;
            for (int v = 0; v < V; ++v) den += std::exp(static_cast<double>(l[v] - mx));
            const double lse = static_cast<double>(mx) + std::log(den);
            lp_blank[r] = static_cast<float>(l[cfg.blank_token_id] - lse);
            lp_tok[r] = u < U ? static_cast<float>(l[token_ids[static_cast<std::size_t>(u)]] - lse)
                              : -std::numeric_limits<float>::infinity();
            float dm = l[V];
            for (int k = 1; k < nd; ++k) dm = std::max(dm, l[V + k]);
            double dden = 0.0;
            for (int k = 0; k < nd; ++k) dden += std::exp(static_cast<double>(l[V + k] - dm));
            const double dlse = static_cast<double>(dm) + std::log(dden);
            for (int k = 0; k < nd; ++k)
                lp_dur[r * nd + k] = static_cast<float>(l[V + k] - dlse);
        }
    }

    const double kNeg = -std::numeric_limits<double>::infinity();
    const std::size_t S = static_cast<std::size_t>(T + 1) * (U + 1);
    std::vector<double> score(S, kNeg);
    struct Back { int32_t t = -1; int32_t u = -1; int16_t dur = 0; bool token = false; };
    std::vector<Back> back(S);
    const auto at = [U](int t, int u) { return static_cast<std::size_t>(t) * (U + 1) + u; };
    score[at(0, 0)] = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int u = 0; u <= U; ++u) {
            const double s = score[at(t, u)];
            if (s == kNeg) continue;
            const std::size_t r = static_cast<std::size_t>(u) * T + t;
            for (int k = 0; k < nd; ++k) {
                const int d = cfg.durations[static_cast<std::size_t>(k)];
                const double pd = lp_dur[r * nd + k];
                if (d > 0) {
                    const int nt = std::min(t + d, T);
                    const double v = s + lp_blank[r] + pd;
                    if (v > score[at(nt, u)]) {
                        score[at(nt, u)] = v;
                        back[at(nt, u)] = Back{t, u, static_cast<int16_t>(d), false};
                    }
                }
                if (u < U) {
                    const int nt = std::min(t + d, T);
                    const double v = s + lp_tok[r] + pd;
                    if (v > score[at(nt, u + 1)]) {
                        score[at(nt, u + 1)] = v;
                        back[at(nt, u + 1)] = Back{t, u, static_cast<int16_t>(d), true};
                    }
                }
            }
        }
    }
    if (score[at(T, U)] == kNeg) fail(where, "no alignment path reaches the end of the audio");

    Alignment out;
    out.num_frames = T;
    out.log_prob = score[at(T, U)];
    out.token_frames.assign(static_cast<std::size_t>(U), 0);
    out.token_durations.assign(static_cast<std::size_t>(U), 0);
    int t = T, u = U;
    while (t > 0 || u > 0) {
        const Back b = back[at(t, u)];
        if (b.t < 0) fail(where, "broken alignment back-pointer");
        if (b.token) {
            out.token_frames[static_cast<std::size_t>(b.u)] = b.t;
            out.token_durations[static_cast<std::size_t>(b.u)] = b.dur;
        }
        t = b.t;
        u = b.u;
    }
    return out;
}

// ── Legacy single-call transcribe: decode with a throwaway local state. ──
Parakeet::Transcription Parakeet::transcribe(const AudioBuffer& audio) const {
    return transcribe(audio, TranscribeOptions{});
}

Parakeet::Transcription Parakeet::transcribe(const AudioBuffer& audio,
                                             const TranscribeOptions& opts) const {
    ParakeetPrediction::State st;
    return impl_->run_transcribe(st, audio, opts);
}

// ─── ParakeetSession (per-decode prediction-net state) ─────────────────────
//
// The opaque session state is just the TDT prediction LSTM (h, c); the encoder,
// joint, and weights stay in the shared Parakeet. make_session() seeds the zero
// state; transcribe(session, ...) funnels through the same run_transcribe() the
// legacy path uses, against the session's own state.

struct ParakeetSessionState {
    ParakeetPrediction::State st;
};

ParakeetSession::ParakeetSession()
    : state_(std::make_unique<ParakeetSessionState>()) {}
ParakeetSession::~ParakeetSession() = default;
ParakeetSession::ParakeetSession(ParakeetSession&&) noexcept = default;
ParakeetSession& ParakeetSession::operator=(ParakeetSession&&) noexcept = default;

ParakeetSession Parakeet::make_session() const {
    if (!impl_->loaded)
        fail("Parakeet::make_session", "no model loaded; call Parakeet::load() first");
    ParakeetSession s;
    s.state_->st = impl_->prediction.init_state();
    return s;
}

void Parakeet::reset(ParakeetSession& session) const {
    if (!impl_->loaded)
        fail("Parakeet::reset", "no model loaded");
    session.state_->st = impl_->prediction.init_state();
}

Parakeet::Transcription Parakeet::transcribe(ParakeetSession& session,
                                             const AudioBuffer& audio,
                                             const TranscribeOptions& opts) const {
    return impl_->run_transcribe(session.state_->st, audio, opts);
}

Parakeet::Transcription Parakeet::transcribe(ParakeetSession& session,
                                             const AudioBuffer& audio) const {
    return transcribe(session, audio, TranscribeOptions{});
}

const ParakeetConfig& Parakeet::config() const { return impl_->config; }
bool Parakeet::loaded() const { return impl_->loaded; }

}  // namespace brosoundml
