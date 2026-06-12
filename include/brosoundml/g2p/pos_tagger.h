#pragma once

// Byte-level Transformer POS tagger. See docs/pos_tagger.md.

#include "brosoundml/g2p/tags.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace brosoundml::g2p {

struct WordTag {
    // NOTE: `word` borrows from the sentence string passed to tag(). Caller
    // must keep the input alive for the lifetime of the returned vector, or
    // copy into std::string before letting the input go.
    std::string_view word;
    PosTag           tag;
};

class PosTagger {
 public:
    static PosTagger load(const std::string& weights_path);

    // Tag one sentence. Caller is responsible for sentence splitting.
    std::vector<WordTag> tag(std::string_view sentence) const;

    PosTagger(PosTagger&&) noexcept;
    PosTagger& operator=(PosTagger&&) noexcept;
    ~PosTagger();

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PosTagger();
};

}  // namespace brosoundml::g2p
