#include "laya_audio_corpus.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace laya_audio {

namespace {

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream is(s);
    while (std::getline(is, cur, sep)) out.push_back(cur);
    return out;
}

}  // namespace

DomainSpec parse_domain(const std::string& spec, bool with_weight) {
    const std::vector<std::string> f = split(spec, ',');
    const std::size_t need = with_weight ? 4 : 3;
    if (f.size() < need - 1 || f.size() > need)
        throw std::runtime_error("laya_audio: bad domain spec '" + spec + "'");
    DomainSpec d;
    std::size_t i = 0;
    d.name = f[i++];
    if (with_weight) d.weight = std::stof(f[i++]);
    d.align = f[i++];
    if (i < f.size())
        for (const std::string& c : split(f[i], '+'))
            if (!c.empty()) d.caches.push_back(c);
    return d;
}

void Corpus::add(const DomainSpec& d) {
    Domain dom;
    dom.name = d.name;
    dom.weight = d.weight;
    dom.u0 = static_cast<int>(utts.size());
    dom.w0 = static_cast<int>(cache.windows.size());
    for (AlignedUtterance& u : read_alignments(d.align)) utts.push_back(std::move(u));
    dom.u1 = static_cast<int>(utts.size());
    for (const std::string& path : d.caches) {
        WindowCache c = read_cache(path);
        if (cache.windows.empty() && cache.dim == 0) {
            cache.window_s = c.window_s;
            cache.dim = c.dim;
        } else if (c.window_s != cache.window_s || c.dim != cache.dim) {
            throw std::runtime_error("laya_audio: caches differ in window length or dim: " + path);
        }
        for (CachedWindow& w : c.windows) {
            if (w.utt < 0 || w.utt >= dom.u1 - dom.u0)
                throw std::runtime_error("laya_audio: cache " + path + " does not match " + d.align);
            w.utt += dom.u0;
            cache.windows.push_back(std::move(w));
        }
    }
    dom.w1 = static_cast<int>(cache.windows.size());
    domains.push_back(dom);
}

void Corpus::add_alignment(const std::string& name, const std::string& align) {
    add(DomainSpec{name, 0.0f, align, {}});
}

int Corpus::sample_window(std::mt19937& rng) const {
    double total = 0;
    for (const Domain& d : domains)
        if (d.w1 > d.w0) total += d.weight;
    std::uniform_real_distribution<double> u(0.0, total);
    double r = u(rng);
    for (const Domain& d : domains) {
        if (d.w1 <= d.w0) continue;
        if (r < d.weight || &d == &domains.back()) return std::uniform_int_distribution<int>(d.w0, d.w1 - 1)(rng);
        r -= d.weight;
    }
    const Domain& d = domains.back();
    return std::uniform_int_distribution<int>(d.w0, d.w1 - 1)(rng);
}

int Corpus::domain_of_window(int w) const {
    for (std::size_t i = 0; i < domains.size(); ++i)
        if (w >= domains[i].w0 && w < domains[i].w1) return static_cast<int>(i);
    return -1;
}

std::vector<int> Corpus::windows_of(int domain) const {
    std::vector<int> out;
    for (int w = domains[static_cast<std::size_t>(domain)].w0; w < domains[static_cast<std::size_t>(domain)].w1; ++w)
        out.push_back(w);
    return out;
}

void VocabSet::build(const std::vector<AlignedUtterance>& utts, int min_count, bool exclude_held_out) {
    std::map<std::string, std::vector<AlignedUtterance>> by;
    for (const AlignedUtterance& u : utts) {
        if (u.words.empty()) continue;
        AlignedUtterance slim;
        slim.words = u.words;
        by[language_of(u.subset)].push_back(std::move(slim));
    }
    by_lang_.clear();
    langs_.clear();
    for (auto& [lang, us] : by) {
        by_lang_[lang].build(us, min_count, exclude_held_out);
        if (!by_lang_[lang].empty()) langs_.push_back(lang);
    }
}

Vocab& VocabSet::get(const std::string& lang) {
    auto it = by_lang_.find(lang);
    if (it != by_lang_.end() && !it->second.empty()) return it->second;
    it = by_lang_.find("en");
    if (it == by_lang_.end()) throw std::runtime_error("laya_audio: no vocabulary for " + lang);
    return it->second;
}

Vocab& VocabSet::for_utterance(const AlignedUtterance& u, std::mt19937& rng) {
    if (u.words.empty() && !langs_.empty())
        return get(langs_[std::uniform_int_distribution<std::size_t>(0, langs_.size() - 1)(rng)]);
    return get(language_of(u.subset));
}

}  // namespace laya_audio
