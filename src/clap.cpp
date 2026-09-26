// CLAP pipeline: config + weight loading, the ClapFeatureExtractor front-end
// (host), and scoring. The towers live in clap_audio.cpp / clap_text.cpp. See
// include/brosoundml/clap.h and docs/clap.md.
#include "clap_internal.h"

#include "mel_slaney.h"

#include <brosoundml/detail/json.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace brosoundml {

namespace clap {

bt::Tensor upload(const sf::File& f, const std::string& name, int rows, int cols, bt::Device dev) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    if (v->dtype != sf::Dtype::F32) fail("tensor '" + name + "' is not F32 (run scripts/convert-clap.py)");
    if (v->numel() != static_cast<std::int64_t>(rows) * cols)
        fail("tensor '" + name + "' has " + std::to_string(v->numel()) + " elements, expected " +
             std::to_string(static_cast<std::int64_t>(rows) * cols));
    bt::Tensor t;
    {
        bt::DeviceScope cpu(bt::Device::CPU);
        sf::upload(*v, rows, cols, t);
    }
    return dev == bt::Device::CPU ? t : t.to(dev);
}

std::vector<float> host_f32(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    if (v->dtype != sf::Dtype::F32) fail("tensor '" + name + "' is not F32");
    std::vector<float> out(static_cast<std::size_t>(v->numel()));
    if (!out.empty()) std::memcpy(out.data(), v->data, out.size() * sizeof(float));
    return out;
}

void l2_normalize(std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * x;
    // torch F.normalize: x / max(||x||, 1e-12).
    const double n = std::max(std::sqrt(s), 1e-12);
    for (float& x : v) x = static_cast<float>(x / n);
}

}  // namespace clap

namespace {

using clap::fail;
namespace fs = std::filesystem;
namespace bt = brotensor;
namespace sf = brotensor::safetensors;

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) fail("cannot open " + p.string());
    std::ostringstream os;
    os << f.rdbuf();
    return os.str();
}

// config.json (+ preprocessor_config.json when present) over the defaults.
ClapConfig parse_config(const fs::path& dir) {
    namespace j = brosoundml::detail::json;
    ClapConfig c;
    const j::Value root = j::parse(slurp(dir / "config.json"));
    c.projection_dim = root.get_int("projection_dim", c.projection_dim);
    if (const j::Value* a = root.find("audio_config"); a && a->is_object()) {
        if (a->get_bool("enable_fusion", false)) fail("fused CLAP checkpoints are not supported (unfused HTSAT only)");
        c.depths = a->get_int_array("depths", c.depths);
        c.heads = a->get_int_array("num_attention_heads", c.heads);
        c.patch_embed_dim = a->get_int("patch_embeds_hidden_size", c.patch_embed_dim);
        c.patch_size = a->get_int("patch_size", c.patch_size);
        c.window_size = a->get_int("window_size", c.window_size);
        c.spec_size = a->get_int("spec_size", c.spec_size);
        c.num_mel_bins = a->get_int("num_mel_bins", c.num_mel_bins);
        c.audio_hidden = a->get_int("hidden_size", c.audio_hidden);
        c.audio_ln_eps = a->get_float("layer_norm_eps", c.audio_ln_eps);
    }
    if (const j::Value* t = root.find("text_config"); t && t->is_object()) {
        c.vocab_size = t->get_int("vocab_size", c.vocab_size);
        c.text_hidden = t->get_int("hidden_size", c.text_hidden);
        c.text_layers = t->get_int("num_hidden_layers", c.text_layers);
        c.text_heads = t->get_int("num_attention_heads", c.text_heads);
        c.text_intermediate = t->get_int("intermediate_size", c.text_intermediate);
        c.max_position = t->get_int("max_position_embeddings", c.max_position);
        c.pad_token_id = t->get_int("pad_token_id", c.pad_token_id);
        c.text_ln_eps = t->get_float("layer_norm_eps", c.text_ln_eps);
        if (t->get_string("hidden_act", "gelu") != "gelu") fail("text hidden_act must be gelu");
    }
    if (root.get_string("projection_hidden_act", "relu") != "relu") fail("projection_hidden_act must be relu");
    if (fs::exists(dir / "preprocessor_config.json")) {
        const j::Value p = j::parse(slurp(dir / "preprocessor_config.json"));
        if (p.get_string("truncation", "rand_trunc") != "rand_trunc" ||
            p.get_string("padding", "repeatpad") != "repeatpad")
            fail("preprocessor_config: only truncation=rand_trunc, padding=repeatpad are supported");
        c.sample_rate = p.get_int("sampling_rate", c.sample_rate);
        c.max_samples = p.get_int("max_length_s", c.max_samples / c.sample_rate) * c.sample_rate;
        c.n_fft = p.get_int("fft_window_size", c.n_fft);
        c.hop_length = p.get_int("hop_length", c.hop_length);
        c.num_mel_bins = p.get_int("feature_size", c.num_mel_bins);
        c.f_min = p.get_float("frequency_min", static_cast<float>(c.f_min));
        c.f_max = p.get_float("frequency_max", static_cast<float>(c.f_max));
    }
    return c;
}

}  // namespace

Clap::Clap() : impl_(std::make_unique<Impl>()) {}
Clap::~Clap() = default;
Clap::Clap(Clap&&) noexcept = default;
Clap& Clap::operator=(Clap&&) noexcept = default;

void Clap::load(const std::string& dir, bt::Device device) {
    auto im = std::make_unique<Impl>();
    const fs::path d(dir);
    if (!fs::exists(d / "config.json")) fail("no config.json under '" + dir + "'");
    if (!fs::exists(d / "model.safetensors"))
        fail("no model.safetensors under '" + dir + "' (run scripts/download-clap.sh + scripts/convert-clap.py)");
    im->cfg = parse_config(d);
    im->device = device;
    bt::DeviceScope scope(device);
    const sf::File f = sf::File::open((d / "model.safetensors").string());
    im->audio.load(f, im->cfg, device);
    im->text.load(f, dir, im->cfg, device);
    im->logit_scale_a = std::exp(clap::host_f32(f, "logit_scale_a").at(0));
    im->logit_scale_t = std::exp(clap::host_f32(f, "logit_scale_t").at(0));
    im->loaded = true;
    impl_ = std::move(im);
}

bool Clap::loaded() const { return impl_ && impl_->loaded; }
const ClapConfig& Clap::config() const { return impl_->cfg; }
bt::Device Clap::device() const { return impl_->device; }
float Clap::logit_scale() const { return impl_->logit_scale_a; }
float Clap::text_logit_scale() const { return impl_->logit_scale_t; }

// ── front-end ───────────────────────────────────────────────────────────────

std::vector<float> Clap::to_model_rate(const AudioBuffer& audio) const {
    if (audio.sample_rate <= 0) fail("sample rate must be positive");
    if (audio.sample_rate == impl_->cfg.sample_rate) return audio.samples;
    return resample(audio, impl_->cfg.sample_rate).samples;
}

std::vector<float> Clap::log_mel(const std::vector<float>& wave48, int crop_offset) const {
    const ClapConfig& c = impl_->cfg;
    const int n = static_cast<int>(wave48.size());
    const int MAX = c.max_samples;
    if (n == 0) fail("empty clip");

    // ClapFeatureExtractor._get_input_mel: crop a longer clip (rand_trunc, at
    // a chosen offset), repeat-pad a shorter one ("repeatpad": whole tiles,
    // then zeros).
    std::vector<float> x(static_cast<std::size_t>(MAX), 0.0f);
    if (n > MAX) {
        const int span = n - MAX;
        const int off = crop_offset < 0 ? span / 2 : std::min(crop_offset, span);
        std::copy(wave48.begin() + off, wave48.begin() + off + MAX, x.begin());
    } else {
        const int reps = MAX / n;
        for (int r = 0; r < reps; ++r)
            std::copy(wave48.begin(), wave48.end(), x.begin() + static_cast<std::ptrdiff_t>(r) * n);
    }

    // STFT (centred, reflect-padded, periodic Hann) on the host.
    const int n_bins = c.n_fft / 2 + 1;
    bt::Tensor spec;
    {
        bt::DeviceScope cpu(bt::Device::CPU);
        const std::vector<float> hw = melslaney::build_hann_window(c.n_fft);
        bt::Tensor window = bt::Tensor::from_host_on(bt::Device::CPU, hw.data(), 1, c.n_fft);
        bt::Tensor signal = bt::Tensor::from_host_on(bt::Device::CPU, x.data(), 1, MAX);
        bt::stft(signal, window, /*N=*/1, c.n_fft, c.hop_length, c.n_fft, /*center=*/true,
                 /*normalized=*/false, spec);
    }
    const int T = spec.rows;
    if (T != c.frames()) fail("stft frame count mismatch");

    // Power -> Slaney mel (sparse triangles) -> 10*log10(max(x, 1e-10)).
    const std::vector<float> fb =
        melslaney::build_filterbank(c.num_mel_bins, c.n_fft, c.sample_rate, c.f_min, c.f_max);
    std::vector<int> lo(c.num_mel_bins), hi(c.num_mel_bins);
    for (int m = 0; m < c.num_mel_bins; ++m) {
        lo[m] = n_bins;
        hi[m] = 0;
        for (int k = 0; k < n_bins; ++k)
            if (fb[static_cast<std::size_t>(m) * n_bins + k] != 0.0f) {
                lo[m] = std::min(lo[m], k);
                hi[m] = k + 1;
            }
    }
    const float* sd = spec.host_f32();
    std::vector<double> power(static_cast<std::size_t>(n_bins));
    std::vector<float> mel(static_cast<std::size_t>(T) * c.num_mel_bins);
    for (int t = 0; t < T; ++t) {
        const float* row = sd + static_cast<std::size_t>(t) * 2 * n_bins;
        for (int k = 0; k < n_bins; ++k) {
            const double re = row[2 * k], im = row[2 * k + 1];
            power[k] = re * re + im * im;
        }
        for (int m = 0; m < c.num_mel_bins; ++m) {
            double acc = 0.0;
            const float* w = fb.data() + static_cast<std::size_t>(m) * n_bins;
            for (int k = lo[m]; k < hi[m]; ++k) acc += w[k] * power[k];
            mel[static_cast<std::size_t>(t) * c.num_mel_bins + m] =
                static_cast<float>(10.0 * std::log10(std::max(acc, 1e-10)));
        }
    }
    return mel;
}

std::vector<int> Clap::window_starts(int n) const {
    const int MAX = impl_->cfg.max_samples;
    if (n <= MAX) return {0};
    const int k = (n + MAX - 1) / MAX;
    std::vector<int> s(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i)
        s[i] = static_cast<int>(std::nearbyint(static_cast<double>(i) * (n - MAX) / (k - 1)));
    return s;
}

// ── towers ──────────────────────────────────────────────────────────────────

std::vector<float> Clap::embed_mel(const std::vector<float>& mel, std::vector<float>* pooled) const {
    if (!loaded()) fail("model not loaded");
    std::vector<float> pool, proj;
    impl_->audio.forward(mel, impl_->cfg, impl_->device, pool, proj);
    clap::l2_normalize(proj);
    if (pooled) *pooled = std::move(pool);
    return proj;
}

std::vector<float> Clap::embed_audio(const AudioBuffer& audio, const ClapAudioOptions& opts) const {
    if (!loaded()) fail("model not loaded");
    if (audio.samples.empty()) fail("empty clip");
    const std::vector<float> wave = to_model_rate(audio);
    const int n = static_cast<int>(wave.size());
    if (n <= impl_->cfg.max_samples || opts.long_mode == ClapLongMode::Crop)
        return embed_mel(log_mel(wave, opts.crop_offset));
    const int MAX = impl_->cfg.max_samples;
    std::vector<double> acc(static_cast<std::size_t>(impl_->cfg.projection_dim), 0.0);
    for (int s : window_starts(n)) {
        const std::vector<float> win(wave.begin() + s, wave.begin() + s + MAX);
        const std::vector<float> e = embed_mel(log_mel(win));
        for (std::size_t i = 0; i < acc.size(); ++i) acc[i] += e[i];
    }
    std::vector<float> out(acc.size());
    for (std::size_t i = 0; i < acc.size(); ++i) out[i] = static_cast<float>(acc[i]);
    clap::l2_normalize(out);
    return out;
}

std::vector<std::int32_t> Clap::tokenize(const std::string& text) const {
    if (!loaded()) fail("model not loaded");
    return impl_->text.tokenize(text, impl_->cfg);
}

std::vector<float> Clap::embed_ids(const std::vector<std::int32_t>& ids) const {
    if (!loaded()) fail("model not loaded");
    std::vector<float> e = impl_->text.forward(ids, impl_->cfg, impl_->device);
    clap::l2_normalize(e);
    return e;
}

std::vector<float> Clap::embed_text(const std::string& text) const {
    return embed_ids(tokenize(text));
}

// ── scoring ─────────────────────────────────────────────────────────────────

ClapScore Clap::score_embeddings(const std::vector<float>& a, const std::vector<float>& text, int n_prompts,
                                 float logit_scale) {
    if (n_prompts < 0 || text.size() != a.size() * static_cast<std::size_t>(n_prompts))
        fail("score: text embeddings must hold n_prompts x embedding floats");
    ClapScore s;
    s.audio_embedding = a;
    s.similarity.resize(n_prompts);
    s.logits.resize(n_prompts);
    s.probability.resize(n_prompts);
    double mx = -1e300;
    for (int p = 0; p < n_prompts; ++p) {
        double dot = 0.0;
        const float* t = text.data() + static_cast<std::size_t>(p) * a.size();
        for (std::size_t i = 0; i < a.size(); ++i) dot += static_cast<double>(a[i]) * t[i];
        s.similarity[p] = static_cast<float>(dot);
        s.logits[p] = static_cast<float>(dot * logit_scale);
        mx = std::max(mx, dot * logit_scale);
    }
    double z = 0.0;
    std::vector<double> e(static_cast<std::size_t>(n_prompts));
    for (int p = 0; p < n_prompts; ++p) z += (e[p] = std::exp(static_cast<double>(s.similarity[p]) * logit_scale - mx));
    for (int p = 0; p < n_prompts; ++p) s.probability[p] = static_cast<float>(e[p] / z);
    return s;
}

ClapScore Clap::score(const AudioBuffer& audio, const std::vector<std::string>& prompts,
                      const ClapAudioOptions& opts) const {
    const std::vector<float> a = embed_audio(audio, opts);
    std::vector<float> text;
    text.reserve(prompts.size() * a.size());
    for (const std::string& p : prompts) {
        const std::vector<float> t = embed_text(p);
        text.insert(text.end(), t.begin(), t.end());
    }
    return score_embeddings(a, text, static_cast<int>(prompts.size()), logit_scale());
}

}  // namespace brosoundml
