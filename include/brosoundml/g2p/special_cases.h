#pragma once

// Special-case rule engine for English G2P. See docs/special_cases.md for
// the spec.
//
// Runs BEFORE the lexicon: handles function-word allomorphy (`a`/`an`/`the`,
// `to`, `in`, `by`, `I`, `am`, `used`), symbol words (`%` → "percent" etc.),
// the `vs.` → `versus` rewrite, dotted acronyms (`U.S.A.`), and the
// letter-by-letter spelling fallback used by the Phonemizer for unknown
// proper nouns / acronyms. Direct port of misaki/en.py's `get_special_case`
// and `get_NNP`. US English only.
//
// Sole intended consumer is the planned `Phonemizer` façade. `try_phonemize`
// returns `""` when no rule applies — the caller falls through to the
// Lexicon + Morphology chain.

#include <string>
#include <string_view>

namespace brosoundml::g2p {

class Lexicon;  // forward — full def in lexicon.h

// Per-word context derived from the word AFTER this one. Phonemizer assembles
// these in a right-to-left sentence walk; SpecialCases only reads them.
struct TokenContext {
    // 0 = no info (no following IPA-producing token).
    // 1 = the following word starts with a vowel phoneme.
    //-1 = the following word starts with a consonant phoneme.
    int future_vowel = 0;

    // True iff the next token is "to" (case-aware per misaki).
    bool future_to = false;
};

class SpecialCases {
 public:
    // Borrows `lex`. `lex` must outlive this SpecialCases. Stored as a
    // non-owning pointer (not a reference) so the class stays
    // move-constructible/assignable.
    explicit SpecialCases(const Lexicon& lex);

    // If a special-case rule matches, returns the assembled IPA. Returns ""
    // if no rule applies — caller falls through to the Lexicon / Morphology
    // chain. Rule order mirrors misaki's `get_special_case`; first match
    // wins. See docs/special_cases.md § "Rule table".
    std::string try_phonemize(std::string_view word,
                              std::string_view ptb_pos,
                              const TokenContext& ctx) const;

    // Letter-by-letter spelling fallback for unknown proper nouns and
    // acronyms. Each ASCII letter is upper-cased and looked up in the
    // Lexicon; non-letter chars are skipped. The last secondary-stress mark
    // in the assembled IPA, if any, is promoted to primary — mirroring
    // misaki's `get_NNP` / `apply_stress(0)` trailer.
    //
    // Returns "" if `word` produces no letters or any required letter is
    // missing from the Lexicon (effectively never for A-Z).
    std::string spell_letter_by_letter(std::string_view word) const;

    SpecialCases(SpecialCases&&) noexcept            = default;
    SpecialCases& operator=(SpecialCases&&) noexcept = default;
    SpecialCases(const SpecialCases&)                = default;
    SpecialCases& operator=(const SpecialCases&)     = default;

 private:
    const Lexicon* lex_;
};

}  // namespace brosoundml::g2p
