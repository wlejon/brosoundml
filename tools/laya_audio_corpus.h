#pragma once

// laya-audio: several corpora ("domains") as one window set, and the
// per-language keyword vocabularies the question sampler draws from.
//
// A domain is an alignment file plus the window caches made from it. Loading
// appends its utterances and windows to one Corpus (utterance indices
// shifted), so every tool that takes a (WindowCache, utterances) pair works
// on a mix unchanged. Training draws a domain by weight, then a window of it.

#include "laya_audio_task.h"

#include <map>
#include <random>
#include <string>
#include <vector>

namespace laya_audio {

// "name,weight,align.tsv,cache1+cache2" (with_weight) or
// "name,align.tsv,cache1+cache2".
struct DomainSpec {
    std::string name;
    float weight = 1.0f;
    std::string align;
    std::vector<std::string> caches;
};
DomainSpec parse_domain(const std::string& spec, bool with_weight);

struct Corpus {
    struct Domain {
        std::string name;
        float weight = 1.0f;
        int w0 = 0, w1 = 0;  // window range
        int u0 = 0, u1 = 0;  // utterance range
    };
    std::vector<AlignedUtterance> utts;
    WindowCache cache;
    std::vector<Domain> domains;

    void add(const DomainSpec& d);
    // Alignment-only domain (no cache), for vocabularies and counts.
    void add_alignment(const std::string& name, const std::string& align);
    int sample_window(std::mt19937& rng) const;
    int domain_of_window(int w) const;
    std::vector<int> windows_of(int domain) const;
};

// Keyword vocabularies by language (language_of(subset)). Windows without
// words (non-speech) draw their negatives from a random language.
class VocabSet {
public:
    void build(const std::vector<AlignedUtterance>& utts, int min_count, bool exclude_held_out);
    Vocab& get(const std::string& lang);
    Vocab& for_utterance(const AlignedUtterance& u, std::mt19937& rng);
    const std::vector<std::string>& languages() const { return langs_; }

private:
    std::map<std::string, Vocab> by_lang_;
    std::vector<std::string> langs_;
};

}  // namespace laya_audio
