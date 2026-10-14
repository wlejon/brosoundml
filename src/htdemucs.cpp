// HTDemucs — see include/brosoundml/htdemucs.h. This file is the bag and
// demucs.apply.apply_model as sep.py drives it (split, overlap 0.25, no
// shifts) plus sep.py's resampling and normalisation; the network is
// htdemucs_net.cpp / htdemucs_transformer.cpp.
#include "brosoundml/htdemucs.h"

#include "brotensor/json.h"
#include "htdemucs_model.h"
#include "u8path.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;
namespace j  = brotensor::json;
namespace fs = std::filesystem;
using htd::fail;

namespace {

std::string slurp(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot read '" + path.string() + "'");
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// ── torchaudio.functional.resample (sinc_interp_hann, width 6, rolloff 0.99) ─

class Resampler {
public:
    Resampler(bt::Device dev, int from, int to) {
        const int g = std::gcd(from, to);
        orig_ = from / g;
        new_ = to / g;
        const float base = static_cast<float>(std::min(orig_, new_)) * 0.99f;
        const int lpw = 6;
        width_ = static_cast<int>(std::ceil(static_cast<double>(lpw) * orig_ / base));
        const int K = 2 * width_ + orig_;
        // _get_sinc_resample_kernel in FP32, op for op.
        std::vector<float> k(static_cast<std::size_t>(new_) * K);
        const float pi = 3.14159265358979323846f, scale = base / static_cast<float>(orig_);
        for (int jn = 0; jn < new_; ++jn)
            for (int i = 0; i < K; ++i) {
                const float idx = static_cast<float>(i - width_) / static_cast<float>(orig_);
                float t = static_cast<float>(-jn) / static_cast<float>(new_) + idx;
                t *= base;
                t = std::clamp(t, -static_cast<float>(lpw), static_cast<float>(lpw));
                const float c = std::cos(t * pi / static_cast<float>(lpw) / 2.0f);
                const float window = c * c;
                t *= pi;
                float v = t == 0.0f ? 1.0f : std::sin(t) / t;
                v *= window * scale;
                k[static_cast<std::size_t>(jn) * K + i] = v;
            }
        kernel_ = bt::Tensor::from_host_on(dev, k.data(), new_, K);
    }

    // X (R, L) -> (R, ceil(new * L / orig)).
    void apply(const bt::Tensor& X, bt::Tensor& Y) const {
        const int R = X.rows, L = X.cols, K = 2 * width_ + orig_;
        bt::Tensor p, c, s;
        bt::pad1d_forward(X, R, 1, L, width_, width_ + orig_, /*zero*/ 0, p);
        const int Lp = L + 2 * width_ + orig_;
        bt::conv1d(p, kernel_, nullptr, R, 1, Lp, new_, K, orig_, 0, 1, c);   // (R, new*Lo)
        const int Lo = (Lp - K) / orig_ + 1;
        bt::nchw_to_sequence(c, R, new_, 1, Lo, s);                            // (R*Lo, new)
        htd::reshape(s, R, Lo * new_);
        const long long target = (static_cast<long long>(new_) * L + orig_ - 1) / orig_;
        htd::slice_cols(s, 0, static_cast<int>(target), Y);
    }

private:
    int orig_ = 1, new_ = 1, width_ = 0;
    bt::Tensor kernel_;
};

// Poll `wait` every 2 ms until the device has drained (yue2_vae's pattern).
void drain(bt::Device dev, const std::function<void()>& wait) {
    if (!wait) return;
    auto next = std::chrono::steady_clock::now();
    while (!bt::idle(dev)) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next) {
            wait();
            next = now + std::chrono::milliseconds(2);
        } else {
            std::this_thread::yield();
        }
    }
}

std::vector<float> download(const bt::Tensor& t) {
    if (t.device == bt::Device::CPU) return t.to_host_vector();
    return t.to(bt::Device::CPU).to_host_vector();
}

}  // namespace

const StereoAudio* Separation::find(const std::string& name) const {
    for (std::size_t i = 0; i < names.size(); ++i)
        if (names[i] == name) return &stems[i];
    return nullptr;
}

// ─── model ───────────────────────────────────────────────────────────────────

struct HtDemucs::Impl {
    HtDemucsConfig cfg;
    bt::Device dev = bt::Device::CPU;
    std::vector<std::unique_ptr<htd::Net>> nets;   // per source; null when not loaded
    bool is_loaded = false;

    int source_index(const std::string& name) const {
        for (std::size_t i = 0; i < cfg.sources.size(); ++i)
            if (cfg.sources[i] == name) return static_cast<int>(i);
        return -1;
    }

    void load(const std::string& dir, bt::Device device, const std::vector<std::string>& want);
    Separation separate(const StereoAudio& audio, const SeparateOptions& opts) const;
};

void HtDemucs::Impl::load(const std::string& dir, bt::Device device, const std::vector<std::string>& want) {
    const fs::path d = u8path(dir);
    if (!fs::exists(d / "config.json")) fail("no config.json under '" + dir + "'");
    if (!bt::is_available(device)) fail("requested device is not available");
    const j::Value root = j::parse(slurp(d / "config.json"));
    if (!root.is_object()) fail("config.json is not a JSON object");
    HtDemucsConfig c;
    const j::Value* src = root.find("sources");
    if (!src || !src->is_array() || src->as_array().empty()) fail("config.json: missing sources");
    for (const auto& s : src->as_array()) c.sources.push_back(s.as_string());
    c.sample_rate = root.get_int("samplerate", 44100);
    c.channels = root.get_int("audio_channels", 2);
    c.overlap = root.get_float("overlap", 0.25f);
    const double segment = root.find("segment") ? root.find("segment")->as_number() : 7.8;
    // int(Fraction(39, 5) * 44100): round the double before truncating.
    c.segment_samples = static_cast<int>(std::floor(segment * c.sample_rate + 1e-6));
    if (c.channels != 2) fail("config.json: only stereo models are supported");

    const j::Value* models = root.find("models");
    if (!models || !models->is_array()) fail("config.json: missing models");
    c.files.assign(c.sources.size(), std::string());
    std::vector<const j::Value*> kw(c.sources.size(), nullptr);
    for (const auto& m : models->as_array()) {
        const int si = [&] {
            const std::string stem = m.get_string("stem", "");
            for (std::size_t i = 0; i < c.sources.size(); ++i)
                if (c.sources[i] == stem) return static_cast<int>(i);
            return -1;
        }();
        if (si < 0) fail("config.json: a model names an unknown stem");
        c.files[si] = m.get_string("file", "");
        kw[si] = m.find("kwargs");
    }
    for (const std::string& w : want)
        if (std::find(c.sources.begin(), c.sources.end(), w) == c.sources.end())
            fail("unknown stem '" + w + "'");

    nets.clear();
    nets.resize(c.sources.size());
    for (std::size_t i = 0; i < c.sources.size(); ++i) {
        if (!want.empty() && std::find(want.begin(), want.end(), c.sources[i]) == want.end()) continue;
        if (c.files[i].empty() || !kw[i] || !kw[i]->is_object())
            fail("config.json: no network for stem '" + c.sources[i] + "'");
        const j::Value& k = *kw[i];
        htd::NetConfig nc;
        nc.sources = static_cast<int>(c.sources.size());
        nc.audio_channels = k.get_int("audio_channels", 2);
        nc.channels = k.get_int("channels", 48);
        nc.growth = k.get_int("growth", 2);
        nc.nfft = k.get_int("nfft", 4096);
        nc.depth = k.get_int("depth", 4);
        nc.kernel_size = k.get_int("kernel_size", 8);
        nc.stride = k.get_int("stride", 4);
        nc.context = k.get_int("context", 1);
        nc.dconv_depth = k.get_int("dconv_depth", 2);
        nc.dconv_comp = k.get_int("dconv_comp", 8);
        nc.bottom_channels = k.get_int("bottom_channels", 512);
        nc.t_layers = k.get_int("t_layers", 5);
        nc.t_heads = k.get_int("t_heads", 8);
        nc.t_hidden_scale = k.get_float("t_hidden_scale", 4.0f);
        nc.freq_emb = k.get_float("freq_emb", 0.2f);
        nc.emb_scale = k.get_float("emb_scale", 10.0f);
        nc.t_max_period = k.get_float("t_max_period", 10000.0f);
        nc.samplerate = c.sample_rate;
        nc.segment_samples = c.segment_samples;
        if (k.get_int("dconv_mode", 3) != 3 || k.get_int("norm_starts", 4) < nc.depth ||
            !k.get_bool("cac", true) || k.get_int("wiener_iters", 0) != 0 ||
            k.get_int("context_enc", 0) != 0 || k.get_bool("t_cross_first", false) ||
            k.get_bool("t_sparse_self_attn", false) || k.get_bool("t_sparse_cross_attn", false) ||
            k.get_string("t_emb", "sin") != "sin" || (k.find("multi_freqs") && k.find("multi_freqs")->is_array() &&
                                                      !k.find("multi_freqs")->as_array().empty()))
            fail("config.json: '" + c.sources[i] + "' is not an htdemucs_ft-shaped network");
        const fs::path wp = d / u8path(c.files[i]);
        if (!fs::exists(wp)) fail("missing weights '" + wp.string() + "'");
        const sf::File f = sf::File::open(wp.string());
        auto net = std::make_unique<htd::Net>();
        net->load(f, nc, device);
        nets[i] = std::move(net);
    }
    cfg = std::move(c);
    dev = device;
    is_loaded = true;
}

Separation HtDemucs::Impl::separate(const StereoAudio& audio, const SeparateOptions& opts) const {
    if (!is_loaded) fail("separate: not loaded");
    if (audio.channels != 1 && audio.channels != 2) fail("separate: audio must be mono or stereo");
    if (audio.frames <= 0 || audio.sample_rate <= 0) fail("separate: empty audio");
    if (audio.samples.size() < static_cast<std::size_t>(audio.channels) * audio.frames)
        fail("separate: audio.samples is shorter than channels * frames");

    std::vector<int> stems;
    if (opts.stems.empty()) {
        for (std::size_t i = 0; i < nets.size(); ++i)
            if (nets[i]) stems.push_back(static_cast<int>(i));
    } else {
        for (const std::string& s : opts.stems) {
            const int i = source_index(s);
            if (i < 0) fail("separate: unknown stem '" + s + "'");
            if (!nets[i]) fail("separate: stem '" + s + "' was not loaded");
            if (std::find(stems.begin(), stems.end(), i) == stems.end()) stems.push_back(i);
        }
    }

    bt::DeviceScope scope(dev);
    const int Lin = audio.frames, sr = audio.sample_rate, msr = cfg.sample_rate;
    std::vector<float> planar(static_cast<std::size_t>(2) * Lin);
    for (int ch = 0; ch < 2; ++ch)
        std::memcpy(planar.data() + static_cast<std::size_t>(ch) * Lin,
                    audio.channel(audio.channels == 1 ? 0 : ch), sizeof(float) * Lin);
    bt::Tensor x = bt::Tensor::from_host_on(dev, planar.data(), 2, Lin), xi;
    if (sr != msr) Resampler(dev, sr, msr).apply(x, xi);
    else xi = std::move(x);
    const int L = xi.cols;

    // sep.py: ref = xi.mean(0); mu, sd = ref.mean(), ref.std() + 1e-8.
    double mu = 0.0, sd = 1e-8;
    {
        bt::Tensor s;
        bt::sum_rows(xi, s);
        const std::vector<float> rs = download(s);
        mu = (static_cast<double>(rs[0]) + rs[1]) / 2.0 / L;
        // Two passes: sum((ref - mu)^2) with ref = (x0 + x1) / 2.
        bt::Tensor r0 = bt::Tensor::empty_on(dev, 1, L), r1 = bt::Tensor::empty_on(dev, 1, L);
        bt::copy_d2d(xi, 0, r0, 0, L);
        bt::copy_d2d(xi, L, r1, 0, L);
        bt::axpby_inplace(r0, r1, 0.5f, 0.5f);
        bt::add_scalar_inplace(r0, static_cast<float>(-mu));
        bt::mul_inplace(r0, r0);
        bt::sum_rows(r0, s);
        const double ss = download(s)[0];
        sd = (L > 1 ? std::sqrt(ss / (L - 1)) : 0.0) + 1e-8;
    }
    bt::Tensor xn = xi.clone();
    bt::add_scalar_inplace(xn, static_cast<float>(-mu));
    bt::scale_inplace(xn, static_cast<float>(1.0 / sd));

    // apply_model(split=True): segments every `stride`, each centred in its
    // padded context; triangular weights, normalised by their sum.
    const int seg = cfg.segment_samples;
    const int stride = static_cast<int>((1.0 - cfg.overlap) * seg);
    struct Chunk { int offset, length, start, trim; };
    std::vector<Chunk> chunks;
    for (int off = 0; off < L; off += stride) {
        Chunk c;
        c.offset = off;
        c.length = std::min(L - off, seg);
        const int delta = seg - c.length;
        c.start = off - delta / 2;
        c.trim = delta / 2;
        chunks.push_back(c);
    }
    std::vector<float> weight(seg);
    const float wmax = static_cast<float>(seg - seg / 2);
    for (int t = 0; t < seg; ++t) weight[t] = static_cast<float>(t < seg / 2 ? t + 1 : seg - t) / wmax;
    std::vector<double> wsum(L, 0.0);
    for (const Chunk& c : chunks)
        for (int t = 0; t < c.length; ++t) wsum[c.offset + t] += weight[t];

    // Padded segment inputs (built once, shared by every stem's network).
    const int nseg = static_cast<int>(chunks.size());
    const int max_batch = opts.max_batch > 0 ? opts.max_batch : 2;
    std::vector<bt::Tensor> batches;
    std::vector<int> batch_n;
    for (int b0 = 0; b0 < nseg; b0 += max_batch) {
        const int nb = std::min(max_batch, nseg - b0);
        bt::Tensor m = bt::Tensor::zeros_on(dev, nb, 2 * seg);
        for (int b = 0; b < nb; ++b) {
            const Chunk& c = chunks[b0 + b];
            const int lo = std::max(0, c.start), hi = std::min(L, c.start + seg);
            if (hi > lo)
                bt::copy_d2d_strided(xn, lo, L, m, b * 2 * seg + (lo - c.start), seg, hi - lo, 2);
        }
        batches.push_back(std::move(m));
        batch_n.push_back(nb);
    }
    // Per-chunk normalised weights, (2, length) each.
    std::vector<bt::Tensor> wn(nseg);
    for (int i = 0; i < nseg; ++i) {
        const Chunk& c = chunks[i];
        std::vector<float> w(static_cast<std::size_t>(2) * c.length);
        for (int t = 0; t < c.length; ++t)
            w[t] = w[c.length + t] = static_cast<float>(weight[t] / wsum[c.offset + t]);
        wn[i] = bt::Tensor::from_host_on(dev, w.data(), 2, c.length);
    }

    Separation out;
    const int total = static_cast<int>(stems.size() * batches.size());
    int done = 0;
    std::unique_ptr<Resampler> back;
    if (sr != msr) back = std::make_unique<Resampler>(dev, msr, sr);
    const int S = static_cast<int>(cfg.sources.size());
    for (int si : stems) {
        const htd::Net& net = *nets[si];
        bt::Tensor acc = bt::Tensor::zeros_on(dev, 2, L);
        bt::Tensor y, tmp = bt::Tensor::empty_on(dev, 2, L), wfull = bt::Tensor::empty_on(dev, 2, L);
        int c0 = 0;
        for (std::size_t bi = 0; bi < batches.size(); ++bi) {
            if (opts.cancel && opts.cancel()) {
                out.cancelled = true;
                out.names.clear();
                out.stems.clear();
                return out;
            }
            const int nb = batch_n[bi];
            net.forward(batches[bi], nb, y);   // (nb, S*2*seg)
            for (int b = 0; b < nb; ++b) {
                const Chunk& c = chunks[c0 + b];
                tmp.zero();
                wfull.zero();
                bt::copy_d2d_strided(y, (b * S + si) * 2 * seg + c.trim, seg, tmp, c.offset, L, c.length, 2);
                bt::copy_d2d_strided(wn[c0 + b], 0, c.length, wfull, c.offset, L, c.length, 2);
                bt::mul_inplace(tmp, wfull);
                bt::add_inplace(acc, tmp);
            }
            c0 += nb;
            drain(dev, opts.wait);
            ++done;
            if (opts.on_progress) opts.on_progress(done, total);
        }
        bt::scale_inplace(acc, static_cast<float>(sd));
        bt::add_scalar_inplace(acc, static_cast<float>(mu));
        bt::Tensor res;
        if (back) back->apply(acc, res);
        else res = std::move(acc);
        bt::Tensor fin;
        htd::slice_cols(res, 0, Lin, fin);
        drain(dev, opts.wait);
        StereoAudio a;
        a.channels = 2;
        a.frames = Lin;
        a.sample_rate = sr;
        a.samples = download(fin);
        out.names.push_back(cfg.sources[si]);
        out.stems.push_back(std::move(a));
    }
    return out;
}

// ─── public ──────────────────────────────────────────────────────────────────

HtDemucs::HtDemucs() : impl_(std::make_unique<Impl>()) {}
HtDemucs::~HtDemucs() = default;
HtDemucs::HtDemucs(HtDemucs&&) noexcept = default;
HtDemucs& HtDemucs::operator=(HtDemucs&&) noexcept = default;

void HtDemucs::load(const std::string& dir, bt::Device device, const std::vector<std::string>& stems) {
    impl_->load(dir, device, stems);
}

Separation HtDemucs::separate(const StereoAudio& audio, const SeparateOptions& opts) const {
    return impl_->separate(audio, opts);
}

const HtDemucsConfig& HtDemucs::config() const { return impl_->cfg; }

std::vector<std::string> HtDemucs::loaded_stems() const {
    std::vector<std::string> v;
    for (std::size_t i = 0; i < impl_->nets.size(); ++i)
        if (impl_->nets[i]) v.push_back(impl_->cfg.sources[i]);
    return v;
}

bool HtDemucs::loaded() const { return impl_->is_loaded; }
bt::Device HtDemucs::device() const { return impl_->dev; }

}  // namespace brosoundml
