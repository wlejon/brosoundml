#include "brosoundml/word_align.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace brosoundml {

namespace {

std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') {
            if (!cur.empty()) out.push_back(std::move(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(std::move(cur));
    return out;
}

std::vector<bool> voiced_frames(const AudioBuffer& audio, int hop) {
    const std::size_t n = audio.samples.size() / static_cast<std::size_t>(hop);
    std::vector<double> db(n);
    for (std::size_t i = 0; i < n; ++i) {
        double e = 0.0;
        for (int k = 0; k < hop; ++k) {
            const double s = audio.samples[i * hop + k];
            e += s * s;
        }
        db[i] = 10.0 * std::log10(e / hop + 1e-12);
    }
    std::vector<bool> out(n, false);
    if (n == 0) return out;
    std::vector<double> sorted = db;
    const std::size_t q = std::min(n - 1, static_cast<std::size_t>(0.98 * static_cast<double>(n)));
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(q), sorted.end());
    const double floor_db = sorted[q] - 40.0;
    for (std::size_t i = 0; i < n; ++i) out[i] = db[i] > floor_db;
    return out;
}

void refine_with_energy(std::vector<WordTiming>& words, const std::vector<std::size_t>& count,
                        const AudioBuffer& audio) {
    const int hop = std::max(1, audio.sample_rate / 100);
    const std::vector<bool> v = voiced_frames(audio, hop);
    const long long n = static_cast<long long>(v.size());
    if (n == 0) return;
    const double step = static_cast<double>(hop) / audio.sample_rate;
    const auto idx = [&](double t) {
        return std::max(0LL, std::min(n, static_cast<long long>(std::llround(t / step))));
    };
    const auto voiced_in = [&](long long a, long long b) {
        for (long long i = std::max(0LL, a); i < std::min(n, b); ++i) if (v[i]) return true;
        return false;
    };
    const long long reach = static_cast<long long>(std::llround(0.12 / step));
    const long long pause = static_cast<long long>(std::llround(0.12 / step));
    double prev_end = 0.0;
    for (std::size_t w = 0; w < words.size(); ++w) {
        if (count[w] == 0) continue;
        WordTiming& cur = words[w];
        double limit = audio.duration_seconds();
        for (std::size_t m = w + 1; m < words.size(); ++m)
            if (count[m] != 0) { limit = words[m].start; break; }

        const long long s = idx(cur.start);
        if (!voiced_in(s - 2 * reach, s - reach / 2)) {
            long long best = -1;
            for (long long f = std::max(0LL, s - reach); f < std::min(n - 1, s + reach); ++f)
                if (!v[f] && v[f + 1] && (best < 0 || std::llabs(f + 1 - s) < std::llabs(best - s))) best = f + 1;
            if (best >= 0) {
                const double t = best * step;
                if (t >= prev_end && t < cur.end) cur.start = t;
            }
        }

        const long long e = idx(cur.end), lim = idx(limit);
        long long last = e, run = 0;
        for (long long i = e; i < lim; ++i) {
            if (v[i]) { last = i + 1; run = 0; }
            else if (++run >= pause) break;
        }
        long long end = last;
        const long long floor_end = idx(cur.start) + static_cast<long long>(std::llround(0.08 / step));
        while (end > floor_end && end - 1 < n && !v[end - 1]) --end;
        cur.end = std::max(cur.start, std::min(end * step, limit));
        prev_end = cur.end;
    }
}

}  // namespace

WordAlignment align_words(const Parakeet& model,
                          const brolm::t5::Tokenizer& tokenizer,
                          const AudioBuffer& audio,
                          const std::string& text) {
    const std::vector<std::string> words = split_words(text);
    if (words.empty())
        throw std::runtime_error("brosoundml: align_words: text has no words");

    const int vocab = model.config().vocab_size;
    const int blank = model.config().blank_token_id;
    std::vector<int32_t> ids;
    std::vector<std::size_t> first(words.size()), count(words.size());
    for (std::size_t w = 0; w < words.size(); ++w) {
        first[w] = ids.size();
        for (int32_t id : tokenizer.tokenize(words[w]))
            if (id >= 0 && id < vocab && id != blank) ids.push_back(id);
        count[w] = ids.size() - first[w];
    }
    if (ids.empty())
        throw std::runtime_error("brosoundml: align_words: text tokenizes to nothing");

    const Parakeet::Alignment al = model.align(audio, ids);
    const double fs = model.config().frame_seconds();
    const double total = audio.duration_seconds();

    WordAlignment out;
    out.log_prob = al.log_prob;
    out.words.resize(words.size());
    for (std::size_t w = 0; w < words.size(); ++w) {
        out.words[w].text = words[w];
        if (count[w] == 0) continue;
        const std::size_t a = first[w];
        const std::size_t b = a + count[w] - 1;
        const int hand = al.token_frames[b] + std::max(1, al.token_durations[b]);
        out.words[w].start = al.token_frames[a] * fs;
        out.words[w].end = std::min(hand * fs, total);
    }
    double prev_end = 0.0;
    for (std::size_t w = 0; w < words.size(); ++w) {
        WordTiming& cur = out.words[w];
        if (count[w] == 0) {
            cur.start = cur.end = prev_end;
            continue;
        }
        for (std::size_t n = w + 1; n < words.size(); ++n) {
            if (count[n] == 0) continue;
            cur.end = std::min(cur.end, out.words[n].start);
            break;
        }
        cur.start = std::min(cur.start, total);
        cur.end = std::max(cur.end, cur.start);
        prev_end = cur.end;
    }
    refine_with_energy(out.words, count, audio);
    return out;
}

}  // namespace brosoundml
