#include "laya_audio_augment.h"

#include "laya_audio_data.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <stdexcept>

namespace laya_audio {

namespace {

using cd = std::complex<double>;

void fft(std::vector<cd>& a, bool inverse) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const double pi = 3.14159265358979323846;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2 * pi / static_cast<double>(len) * (inverse ? 1 : -1);
        const cd wl(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            cd w(1);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const cd u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
    if (inverse)
        for (cd& x : a) x /= static_cast<double>(n);
}

std::vector<std::string> read_list(const std::string& path) {
    std::vector<std::string> out;
    if (path.empty()) return out;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("laya_audio: cannot open list " + path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

double rms(const std::vector<float>& x) {
    double e = 0;
    for (float v : x) e += double(v) * v;
    return std::sqrt(e / std::max<std::size_t>(1, x.size()));
}

}  // namespace

std::vector<float> fft_convolve(const std::vector<float>& x, const std::vector<float>& h) {
    if (x.empty() || h.empty()) return {};
    const std::size_t n_out = x.size() + h.size() - 1;
    std::size_t n = 1;
    while (n < n_out) n <<= 1;
    std::vector<cd> a(n), b(n);
    for (std::size_t i = 0; i < x.size(); ++i) a[i] = x[i];
    for (std::size_t i = 0; i < h.size(); ++i) b[i] = h[i];
    fft(a, false);
    fft(b, false);
    for (std::size_t i = 0; i < n; ++i) a[i] *= b[i];
    fft(a, true);
    std::vector<float> y(n_out);
    for (std::size_t i = 0; i < n_out; ++i) y[i] = static_cast<float>(a[i].real());
    return y;
}

void Augmenter::load(const AugmentConfig& cfg) {
    cfg_ = cfg;
    rirs_ = read_list(cfg.rir_list);
    noises_.files = read_list(cfg.noise_list);
    music_.files = read_list(cfg.music_list);
}

const std::vector<float>& Augmenter::track(Source& s, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const bool load_new = s.pool.size() < static_cast<std::size_t>(cfg_.pool) || u(rng) < 0.05f;
    if (load_new) {
        const std::string& f = s.files[std::uniform_int_distribution<std::size_t>(0, s.files.size() - 1)(rng)];
        s.hold = std::make_shared<const std::vector<float>>(load_audio_16k(f));
        if (s.hold->size() >= 1600) {
            if (s.pool.size() < static_cast<std::size_t>(cfg_.pool)) s.pool.push_back(s.hold);
            else s.pool[std::uniform_int_distribution<std::size_t>(0, s.pool.size() - 1)(rng)] = s.hold;
            return *s.hold;
        }
        if (s.pool.empty()) return *s.hold;  // too short and nothing else pooled yet
    }
    return *s.pool[std::uniform_int_distribution<std::size_t>(0, s.pool.size() - 1)(rng)];
}

void Augmenter::mix(std::vector<float>& pcm, Source& s, float snr_db, std::mt19937& rng) {
    const std::vector<float>& t = track(s, rng);
    if (t.empty()) return;
    const std::size_t off = std::uniform_int_distribution<std::size_t>(0, t.size() - 1)(rng);
    std::vector<float> seg(pcm.size());
    for (std::size_t i = 0; i < seg.size(); ++i) seg[i] = t[(off + i) % t.size()];
    const double rs = rms(pcm), rn = rms(seg);
    if (rn < 1e-6 || rs < 1e-6) return;
    const float g = static_cast<float>(rs / (rn * std::pow(10.0, snr_db / 20.0)));
    for (std::size_t i = 0; i < pcm.size(); ++i) pcm[i] += g * seg[i];
}

AugmentApplied Augmenter::apply(std::vector<float>& pcm, std::mt19937& rng) {
    AugmentApplied a;
    if (pcm.empty()) return a;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    if (!rirs_.empty() && u(rng) < cfg_.p_rir) {
        std::vector<float> h = load_audio_16k(rirs_[std::uniform_int_distribution<std::size_t>(0, rirs_.size() - 1)(rng)]);
        if (!h.empty()) {
            std::size_t peak = 0;
            for (std::size_t i = 1; i < h.size(); ++i)
                if (std::abs(h[i]) > std::abs(h[peak])) peak = i;
            const double dry = rms(pcm);
            const std::vector<float> y = fft_convolve(pcm, h);
            for (std::size_t i = 0; i < pcm.size(); ++i) pcm[i] = y[i + peak];
            const double wet = rms(pcm);
            if (wet > 1e-9)
                for (float& v : pcm) v = static_cast<float>(v * dry / wet);
            a.rir = true;
        }
    }
    if (!noises_.files.empty() && u(rng) < cfg_.p_noise) {
        mix(pcm, noises_, cfg_.noise_snr_lo + u(rng) * (cfg_.noise_snr_hi - cfg_.noise_snr_lo), rng);
        a.noise = true;
    }
    if (!music_.files.empty() && u(rng) < cfg_.p_music) {
        mix(pcm, music_, cfg_.music_snr_lo + u(rng) * (cfg_.music_snr_hi - cfg_.music_snr_lo), rng);
        a.music = true;
    }
    float peak = 0;
    for (float v : pcm) peak = std::max(peak, std::abs(v));
    if (peak > 0.99f)
        for (float& v : pcm) v *= 0.99f / peak;
    return a;
}

}  // namespace laya_audio
