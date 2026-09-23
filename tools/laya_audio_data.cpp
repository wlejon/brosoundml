#include "laya_audio_data.h"

#include "brosoundml/audio.h"

#include <brolm/detail/json.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace laya_audio {

namespace json = brolm::detail::json;

std::vector<Utterance> load_manifest(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("laya_audio: cannot open manifest " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    const json::Value root = json::parse(ss.str());
    std::vector<Utterance> out;
    for (const json::Value& v : root.as_array()) {
        Utterance u;
        u.id = v.find("id")->as_string();
        u.wav = v.find("wav")->as_string();
        u.speaker = v.find("speaker")->as_string();
        u.subset = v.find("subset")->as_string();
        u.text = v.find("text")->as_string();
        u.duration_s = v.find("duration_s")->as_number();
        out.push_back(std::move(u));
    }
    return out;
}

std::vector<float> resample_sinc(const std::vector<float>& in, int in_rate, int out_rate) {
    if (in_rate == out_rate) return in;
    const double ratio = static_cast<double>(out_rate) / in_rate;
    const double cutoff = std::min(1.0, ratio) * 0.95;  // of the input Nyquist
    constexpr int kZeros = 16;
    const double half_width = kZeros / cutoff;  // input samples each side
    const std::size_t n_out = static_cast<std::size_t>(std::floor(in.size() * ratio));
    std::vector<float> out(n_out);
    const double pi = 3.14159265358979323846;
    for (std::size_t n = 0; n < n_out; ++n) {
        const double t = n / ratio;
        const long lo = static_cast<long>(std::ceil(t - half_width));
        const long hi = static_cast<long>(std::floor(t + half_width));
        double acc = 0, wsum = 0;
        for (long j = lo; j <= hi; ++j) {
            const double x = (t - j) * cutoff;
            const double sinc = std::abs(x) < 1e-9 ? 1.0 : std::sin(pi * x) / (pi * x);
            const double w = 0.5 + 0.5 * std::cos(pi * (t - j) / half_width);
            const double k = sinc * w * cutoff;
            wsum += k;
            if (j >= 0 && j < static_cast<long>(in.size())) acc += k * in[static_cast<std::size_t>(j)];
        }
        out[n] = static_cast<float>(wsum > 0 ? acc / wsum : 0.0);
    }
    return out;
}

std::vector<float> load_audio_16k(const std::string& wav_path) {
    brosoundml::AudioBuffer a = brosoundml::read_wav(wav_path);
    if (a.sample_rate == 16000) return std::move(a.samples);
    return resample_sinc(a.samples, a.sample_rate, 16000);
}

std::vector<std::string> normalize_words(const std::string& text) {
    std::vector<std::string> words;
    std::string cur;
    auto flush = [&] {
        while (!cur.empty() && cur.back() == '\'') cur.pop_back();
        std::size_t s = 0;
        while (s < cur.size() && cur[s] == '\'') ++s;
        if (s < cur.size()) words.push_back(cur.substr(s));
        cur.clear();
    };
    for (unsigned char c : text) {
        if (std::isalnum(c)) {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else if (c == '\'' && !cur.empty()) {
            cur.push_back('\'');
        } else {
            flush();
        }
    }
    flush();
    return words;
}

std::vector<TimedWord> align_reference(const std::vector<std::string>& ref, const std::vector<TimedWord>& hyp) {
    const std::size_t R = ref.size(), H = hyp.size();
    // cost[i][j]: edit distance between ref[:i] and hyp[:j].
    std::vector<std::vector<int>> cost(R + 1, std::vector<int>(H + 1, 0));
    for (std::size_t i = 0; i <= R; ++i) cost[i][0] = static_cast<int>(i);
    for (std::size_t j = 0; j <= H; ++j) cost[0][j] = static_cast<int>(j);
    for (std::size_t i = 1; i <= R; ++i) {
        for (std::size_t j = 1; j <= H; ++j) {
            const int sub = cost[i - 1][j - 1] + (ref[i - 1] == hyp[j - 1].word ? 0 : 1);
            cost[i][j] = std::min({sub, cost[i - 1][j] + 1, cost[i][j - 1] + 1});
        }
    }
    std::vector<TimedWord> out(R);
    for (std::size_t i = 0; i < R; ++i) out[i].word = ref[i];
    std::size_t i = R, j = H;
    while (i > 0 && j > 0) {
        const int sub = cost[i - 1][j - 1] + (ref[i - 1] == hyp[j - 1].word ? 0 : 1);
        if (cost[i][j] == sub) {
            out[i - 1].t0 = hyp[j - 1].t0;
            out[i - 1].t1 = hyp[j - 1].t1;
            out[i - 1].asr_match = ref[i - 1] == hyp[j - 1].word;
            --i;
            --j;
        } else if (cost[i][j] == cost[i - 1][j] + 1) {
            --i;  // deletion: timed below
        } else {
            --j;  // insertion: hyp word with no reference word
        }
    }
    // Interpolate deleted reference words between timed neighbours.
    for (std::size_t k = 0; k < R; ++k) {
        if (out[k].t0 >= 0) continue;
        std::size_t e = k;
        while (e < R && out[e].t0 < 0) ++e;
        const float a = k > 0 ? out[k - 1].t1 : 0.0f;
        const float b = e < R ? out[e].t0 : -1.0f;
        if (b > a + 0.05f) {
            const float step = (b - a) / static_cast<float>(e - k);
            for (std::size_t m = k; m < e; ++m) {
                out[m].t0 = a + step * static_cast<float>(m - k);
                out[m].t1 = out[m].t0 + step;
            }
        }
        k = e;
    }
    return out;
}

void write_alignments(const std::string& path, const std::vector<AlignedUtterance>& utts, bool append) {
    std::ofstream f(path, append ? std::ios::app : std::ios::trunc);
    if (!f) throw std::runtime_error("laya_audio: cannot write " + path);
    char buf[64];
    for (const AlignedUtterance& u : utts) {
        f << u.id << '\t' << u.wav << '\t' << u.speaker << '\t' << u.subset << '\t';
        std::snprintf(buf, sizeof(buf), "%.3f", u.duration_s);
        f << buf << '\t';
        for (std::size_t k = 0; k < u.words.size(); ++k) {
            const TimedWord& w = u.words[k];
            std::snprintf(buf, sizeof(buf), "|%.3f|%.3f|%d", w.t0, w.t1, w.asr_match ? 1 : 0);
            if (k) f << ' ';
            f << w.word << buf;
        }
        f << '\n';
    }
}

std::vector<float> window_audio(const std::vector<float>& pcm16k, float t_end, float window_s, uint32_t seed) {
    const int n = static_cast<int>(std::lround(window_s * 16000.0f));
    const long end = std::lround(t_end * 16000.0);
    std::vector<float> w(static_cast<std::size_t>(n), 0.0f);
    uint32_t s = seed * 2654435761u + 12345u;
    for (int i = 0; i < n; ++i) {
        const long src = end - n + i;
        s = s * 1664525u + 1013904223u;
        const float dither = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 2e-4f;
        w[static_cast<std::size_t>(i)] =
            (src >= 0 && src < static_cast<long>(pcm16k.size()) ? pcm16k[static_cast<std::size_t>(src)] : 0.0f) + dither;
    }
    return w;
}

namespace {
constexpr char kCacheMagic[4] = {'L', 'A', 'C', '1'};
}

CacheWriter::CacheWriter(const std::string& path, float window_s, int dim) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("laya_audio: cannot write " + path);
    f_ = f;
    const int zero = 0;
    std::fwrite(kCacheMagic, 1, 4, f);
    std::fwrite(&window_s, sizeof(float), 1, f);
    std::fwrite(&dim, sizeof(int), 1, f);
    std::fwrite(&zero, sizeof(int), 1, f);  // window count, patched in the destructor
}

CacheWriter::~CacheWriter() {
    std::FILE* f = static_cast<std::FILE*>(f_);
    if (!f) return;
    std::fseek(f, 12, SEEK_SET);
    std::fwrite(&n_, sizeof(int), 1, f);
    std::fclose(f);
}

void CacheWriter::add(const CachedWindow& w) {
    std::FILE* f = static_cast<std::FILE*>(f_);
    std::fwrite(&w.utt, sizeof(int), 1, f);
    std::fwrite(&w.t_end, sizeof(float), 1, f);
    std::fwrite(&w.frames, sizeof(int), 1, f);
    std::fwrite(w.lat.data(), sizeof(uint16_t), w.lat.size(), f);
    ++n_;
}

WindowCache read_cache(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("laya_audio: cannot open " + path);
    WindowCache c;
    char magic[4];
    int n = 0;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, kCacheMagic, 4) != 0 ||
        std::fread(&c.window_s, sizeof(float), 1, f) != 1 || std::fread(&c.dim, sizeof(int), 1, f) != 1 ||
        std::fread(&n, sizeof(int), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("laya_audio: not a window cache: " + path);
    }
    c.windows.resize(static_cast<std::size_t>(n));
    for (CachedWindow& w : c.windows) {
        if (std::fread(&w.utt, sizeof(int), 1, f) != 1 || std::fread(&w.t_end, sizeof(float), 1, f) != 1 ||
            std::fread(&w.frames, sizeof(int), 1, f) != 1) {
            std::fclose(f);
            throw std::runtime_error("laya_audio: truncated cache " + path);
        }
        w.lat.resize(static_cast<std::size_t>(w.frames) * c.dim);
        if (std::fread(w.lat.data(), sizeof(uint16_t), w.lat.size(), f) != w.lat.size()) {
            std::fclose(f);
            throw std::runtime_error("laya_audio: truncated cache " + path);
        }
    }
    std::fclose(f);
    return c;
}

std::vector<AlignedUtterance> read_alignments(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("laya_audio: cannot open " + path);
    std::vector<AlignedUtterance> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> cols;
        std::size_t s = 0;
        for (int c = 0; c < 5; ++c) {
            const std::size_t t = line.find('\t', s);
            if (t == std::string::npos) throw std::runtime_error("laya_audio: bad alignment line in " + path);
            cols.push_back(line.substr(s, t - s));
            s = t + 1;
        }
        AlignedUtterance u;
        u.id = cols[0];
        u.wav = cols[1];
        u.speaker = cols[2];
        u.subset = cols[3];
        u.duration_s = std::stof(cols[4]);
        std::istringstream ws(line.substr(s));
        std::string tok;
        while (ws >> tok) {
            TimedWord w;
            const std::size_t p1 = tok.find('|');
            const std::size_t p2 = tok.find('|', p1 + 1);
            const std::size_t p3 = tok.find('|', p2 + 1);
            if (p1 == std::string::npos || p2 == std::string::npos || p3 == std::string::npos) continue;
            w.word = tok.substr(0, p1);
            w.t0 = std::stof(tok.substr(p1 + 1, p2 - p1 - 1));
            w.t1 = std::stof(tok.substr(p2 + 1, p3 - p2 - 1));
            w.asr_match = tok.substr(p3 + 1) == "1";
            u.words.push_back(std::move(w));
        }
        out.push_back(std::move(u));
    }
    return out;
}

}  // namespace laya_audio
