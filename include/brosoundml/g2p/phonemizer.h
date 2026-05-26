#pragma once

// Sentence → Kokoro-token-ids orchestrator. See docs/phonemizer.md § Slice 5
// for the spec.
//
// Composes the entire in-tree English G2P stack: pre-tokenisation,
// POS tagging (PosTagger), special-case allomorphy (SpecialCases), the
// pronunciation dictionary (Lexicon), affix-stripping fallback (Morphology),
// and finally codepoint-level encoding (PhonemeAdapter). Per-word context
// (future-vowel / future-to) is propagated in a single right-to-left walk so
// the SpecialCases rules see the same lookahead they would in misaki.
//
// US English only. Numbers / currency / years / sentence-stress redistribution
// are deliberate v1 gaps — see the spec's "Behaviour gaps vs. misaki".
//
// Move-constructible; non-owning pointers to its five dependencies. Throws
// std::runtime_error (with the "brosoundml: g2p::Phonemizer::..." prefix) only
// on unrecoverable internal inconsistencies (tag/word-count mismatch).

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brosoundml::g2p {

class Lexicon;
class Morphology;
class SpecialCases;
class PosTagger;
class PhonemeAdapter;

class Phonemizer {
 public:
    // Borrows every dependency. All lifetimes must exceed this Phonemizer.
    Phonemizer(const PosTagger&       tagger,
               const Lexicon&         lexicon,
               const Morphology&      morphology,
               const SpecialCases&    special,
               const PhonemeAdapter&  adapter);

    // Full pipeline: text → Kokoro phoneme ids. Empty / whitespace-only
    // input → empty vector. Punctuation that survives pre-tokenisation is
    // passed through to the adapter, which tokenises known punctuation
    // verbatim and silently drops the rest.
    std::vector<std::int32_t> phonemize(std::string_view sentence) const;

    // Same pipeline, returns the assembled IPA string (the input to
    // PhonemeAdapter::encode) instead of token ids. For tests, debugging,
    // and the brosoundml_synth CLI's verbose mode.
    std::string phonemize_to_ipa(std::string_view sentence) const;

    Phonemizer(Phonemizer&&) noexcept            = default;
    Phonemizer& operator=(Phonemizer&&) noexcept = default;
    Phonemizer(const Phonemizer&)                = default;
    Phonemizer& operator=(const Phonemizer&)     = default;

 private:
    const PosTagger*      tagger_;
    const Lexicon*        lexicon_;
    const Morphology*     morphology_;
    const SpecialCases*   special_;
    const PhonemeAdapter* adapter_;
};

}  // namespace brosoundml::g2p
