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
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".tsv") == 0) {
        // id \t audio \t speaker \t subset \t text
        std::vector<Utterance> out;
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::vector<std::string> c;
            std::size_t s = 0;
            for (int k = 0; k < 4; ++k) {
                const std::size_t t = line.find('\t', s);
                if (t == std::string::npos) throw std::runtime_error("laya_audio: bad manifest line in " + path);
                c.push_back(line.substr(s, t - s));
                s = t + 1;
            }
            Utterance u;
            u.id = c[0];
            u.wav = c[1];
            u.speaker = c[2];
            u.subset = c[3];
            u.text = line.substr(s);
            out.push_back(std::move(u));
        }
        return out;
    }
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
    const std::size_t n = text.size();
    for (std::size_t i = 0; i < n;) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            if (std::isalnum(c)) cur.push_back(static_cast<char>(std::tolower(c)));
            else if (c == '\'' && !cur.empty()) cur.push_back('\'');
            else flush();
            ++i;
            continue;
        }
        // UTF-8 sequence.
        const std::size_t len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        const std::size_t e = std::min(n, i + len);
        const unsigned char c1 = e > i + 1 ? static_cast<unsigned char>(text[i + 1]) : 0;
        if (c == 0xC2) {
            flush();  // U+0080..U+00BF: Latin-1 punctuation and symbols
        } else if (c == 0xC3 && c1 >= 0x80 && c1 <= 0x9E && c1 != 0x97) {
            cur.push_back(static_cast<char>(c));  // U+00C0..U+00DE capitals -> lower case
            cur.push_back(static_cast<char>(c1 + 0x20));
        } else if (c == 0xC3 && c1 == 0x97) {
            flush();  // multiplication sign
        } else if (c == 0xE2 && c1 == 0x80) {
            // General punctuation: the right single quote is an apostrophe.
            const unsigned char c2 = e > i + 2 ? static_cast<unsigned char>(text[i + 2]) : 0;
            if (c2 == 0x99 && !cur.empty()) cur.push_back('\'');
            else flush();
        } else if (c >= 0xC4 && c <= 0xC5 && c1 >= 0x80 && c1 <= 0xBF) {
            // Latin Extended-A (U+0100..U+017F): upper case sits on even code
            // points for most of the block; lower it by setting bit 0.
            const unsigned cp = ((c & 0x1Fu) << 6) | (c1 & 0x3Fu);
            const bool odd_upper = (cp >= 0x139 && cp <= 0x148) || (cp >= 0x179 && cp <= 0x17E);
            unsigned lc = cp;
            if (cp != 0x130 && cp != 0x131 && cp != 0x138 && cp != 0x149 && cp != 0x17F) {
                if (odd_upper) lc = (cp & 1u) ? cp + 1 : cp;
                else lc = (cp & 1u) ? cp : cp + 1;
            }
            cur.push_back(static_cast<char>(0xC0 | (lc >> 6)));
            cur.push_back(static_cast<char>(0x80 | (lc & 0x3F)));
        } else {
            cur.append(text, i, e - i);  // other letters pass through
        }
        i = e;
    }
    flush();
    return words;
}

std::string language_of(const std::string& subset) {
    // "voxpopuli-de-train", "mswc-fr-test": the language is the second field.
    for (const char* corpus : {"voxpopuli-", "mswc-"}) {
        const std::size_t n = std::strlen(corpus);
        if (subset.compare(0, n, corpus) == 0 && subset.size() >= n + 2) return subset.substr(n, 2);
    }
    return "en";
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
