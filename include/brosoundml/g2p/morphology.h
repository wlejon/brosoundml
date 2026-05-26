#pragma once

// Affix-stripping rules engine for English G2P. See docs/morphology.md for
// the spec.
//
// Backstops `brosoundml::g2p::Lexicon` for words missing from the dictionary
// by handling regular inflections (`-s`, `-ed`, `-ing`) and possessive
// variants (`'s`, `s'`, trailing `'`). Direct port of misaki/en.py's
// `stem_s` / `stem_ed` / `stem_ing` plus the two possessive branches inside
// `get_word`. US English only — British variants are a future slice.
//
// Sole intended consumer is the planned `Phonemizer` façade; Morphology
// itself exposes only the rule chain, never orchestration. `try_phonemize`
// does NOT attempt a direct lexicon hit on `word` — the caller probes the
// lexicon first; morphology is the miss path.

#include <string>
#include <string_view>

namespace brosoundml::g2p {

class Lexicon;  // forward — full def in lexicon.h

class Morphology {
 public:
    // Borrows `lex`. `lex` must outlive this Morphology. Stored as a
    // non-owning pointer (not a reference) so the class stays
    // move-constructible/assignable.
    explicit Morphology(const Lexicon& lex);

    // Try to phonemize `word` by applying the inflectional and possessive
    // rules in misaki's `get_word()` order:
    //
    //   1. Possessive s'  →  rewrite "foos'" as "foo's", lex-lookup.
    //   2. Trailing  '    →  rewrite "foo'" as "foo",   lex-lookup.
    //   3. stem_s   → try -s   inflectional decomposition.
    //   4. stem_ed  → try -ed  inflectional decomposition.
    //   5. stem_ing → try -ing inflectional decomposition.
    //
    // Returns the assembled IPA (stem IPA + glue) on the first rule that
    // produces a non-empty stem IPA, or an empty string if no rule fires.
    // `ptb_pos` is forwarded verbatim to `Lexicon::lookup` when probing
    // candidate stems.
    std::string try_phonemize(std::string_view word,
                              std::string_view ptb_pos = "") const;

    Morphology(Morphology&&) noexcept            = default;
    Morphology& operator=(Morphology&&) noexcept = default;
    Morphology(const Morphology&)                = default;
    Morphology& operator=(const Morphology&)     = default;

 private:
    const Lexicon* lex_;
};

}  // namespace brosoundml::g2p
