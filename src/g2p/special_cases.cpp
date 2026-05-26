#include "brosoundml/g2p/special_cases.h"

#include "brosoundml/g2p/lexicon.h"

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

namespace brosoundml::g2p {

namespace {

// ─── Small helpers ───────────────────────────────────────────────────────

bool ascii_alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
}

// PTB → misaki parent-tag mapping. Mirrors the table in
// docs/special_cases.md § "by / By / BY" and the equivalent helper inside
// lexicon.cpp. Duplicated (not shared) intentionally — six branches, not
// worth a cross-TU dependency just for special-case dispatch.
std::string_view parent_tag(std::string_view ptb) {
    if (starts_with(ptb, "VB")) return "VERB";
    if (starts_with(ptb, "NN")) return "NOUN";
    if (starts_with(ptb, "RB") || starts_with(ptb, "ADV")) return "ADV";
    if (starts_with(ptb, "JJ") || starts_with(ptb, "ADJ")) return "ADJ";
    return ptb;
}

// True iff `word` is exactly one of the listed string_views.
template <std::size_t N>
bool word_in(std::string_view word, const std::array<std::string_view, N>& set) {
    for (auto v : set) if (word == v) return true;
    return false;
}

// ASCII lower-case helper (does not touch non-ASCII bytes).
char ascii_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c;
}

// Misaki `vs.` test: case-insensitive match of "vs" or "vs.", ASCII only.
bool is_vs_form(std::string_view word) {
    if (word.size() != 2 && word.size() != 3) return false;
    if (ascii_lower(word[0]) != 'v') return false;
    if (ascii_lower(word[1]) != 's') return false;
    if (word.size() == 3 && word[2] != '.') return false;
    return true;
}

// Dotted-acronym test — open-coded port of misaki's
//   "." in word.strip(".") and word.replace(".", "").isalpha()
//   and len(max(word.split("."), key=len)) < 3
//
//   1. word stripped of leading/trailing '.' still contains a '.'
//      (equivalently: the word has an interior dot).
//   2. every non-'.' char is ASCII alphabetic
//      (Python's str.isalpha is non-ASCII-aware; misaki's actual inputs are
//      ASCII proper-noun acronyms, so ASCII-only matches behaviour for the
//      cases that fire.)
//   3. the longest run of letters between dots in the ORIGINAL word is < 3.
bool is_dotted_acronym(std::string_view word) {
    if (word.empty()) return false;

    // (1) strip leading/trailing '.' then look for an interior dot.
    std::size_t lo = 0;
    std::size_t hi = word.size();
    while (lo < hi && word[lo] == '.') ++lo;
    while (hi > lo && word[hi - 1] == '.') --hi;
    bool interior_dot = false;
    for (std::size_t i = lo; i < hi; ++i) {
        if (word[i] == '.') { interior_dot = true; break; }
    }
    if (!interior_dot) return false;

    // (2) every non-'.' char must be ASCII alphabetic. Also reject the
    //     degenerate all-dots case (no letters at all).
    bool any_letter = false;
    for (char c : word) {
        if (c == '.') continue;
        if (!ascii_alpha(c)) return false;
        any_letter = true;
    }
    if (!any_letter) return false;

    // (3) longest run between dots in the ORIGINAL word < 3.
    std::size_t run = 0;
    std::size_t max_run = 0;
    for (char c : word) {
        if (c == '.') {
            if (run > max_run) max_run = run;
            run = 0;
        } else {
            ++run;
        }
    }
    if (run > max_run) max_run = run;
    return max_run < 3;
}

// Copy a Lexicon lookup string_view into an owned std::string. Returns ""
// if the lookup misses.
std::string lex_get(const Lexicon& lex,
                    std::string_view word,
                    std::string_view ptb_pos = "") {
    return std::string(lex.lookup(word, ptb_pos));
}

}  // namespace

// ─── Public API ──────────────────────────────────────────────────────────

SpecialCases::SpecialCases(const Lexicon& lex) : lex_(&lex) {}

std::string SpecialCases::try_phonemize(std::string_view word,
                                        std::string_view ptb_pos,
                                        const TokenContext& ctx) const {
    if (word.empty()) return {};
    const Lexicon& lex = *lex_;

    // ── Symbol words (ADD-tagged punctuation + free-standing symbols) ──
    if (ptb_pos == "ADD" && word == ".") return lex_get(lex, "dot");
    if (ptb_pos == "ADD" && word == "/") return lex_get(lex, "slash");
    if (word == "%")                     return lex_get(lex, "percent");
    if (word == "&")                     return lex_get(lex, "and");
    if (word == "+")                     return lex_get(lex, "plus");
    if (word == "@")                     return lex_get(lex, "at");

    // ── Dotted acronyms (U.S.A. etc.) ──
    if (is_dotted_acronym(word)) {
        return spell_letter_by_letter(word);
    }

    // ── a / A ──
    {
        static constexpr std::array<std::string_view, 2> kA = {"a", "A"};
        if (word_in(word, kA)) {
            if (ptb_pos == "DT") return "\xC9\x90";          // "ɐ"
            return "\xCB\x88" "A";                            // "ˈA"
        }
    }

    // ── am / Am / AM ──
    {
        static constexpr std::array<std::string_view, 3> kAm = {"am", "Am", "AM"};
        if (word_in(word, kAm)) {
            if (starts_with(ptb_pos, "NN")) return spell_letter_by_letter(word);
            if (ctx.future_vowel == 0 || word != "am") return lex_get(lex, "am");
            return "\xC9\x90" "m";                            // "ɐm"
        }
    }

    // ── an / An / AN ──
    {
        static constexpr std::array<std::string_view, 3> kAn = {"an", "An", "AN"};
        if (word_in(word, kAn)) {
            if (word == "AN" && starts_with(ptb_pos, "NN"))
                return spell_letter_by_letter(word);
            return "\xC9\x90" "n";                            // "ɐn"
        }
    }

    // ── I (the pronoun, PTB PRP) ──
    if (word == "I" && ptb_pos == "PRP") {
        return "\xCB\x8C" "I";                                // "ˌI"
    }

    // ── by / By / BY (only when used as an adverb) ──
    {
        static constexpr std::array<std::string_view, 3> kBy = {"by", "By", "BY"};
        if (word_in(word, kBy) && parent_tag(ptb_pos) == "ADV") {
            return "b\xCB\x88" "I";                           // "bˈI"
        }
    }

    // ── to / To / TO ──
    {
        const bool to_lc = (word == "to" || word == "To");
        const bool to_uc = (word == "TO" && (ptb_pos == "TO" || ptb_pos == "IN"));
        if (to_lc || to_uc) {
            switch (ctx.future_vowel) {
                case 0:  return lex_get(lex, "to");
                case -1: return "t\xC9\x99";                  // "tə"
                case 1:  return "t\xCA\x8A";                  // "tʊ"
                default: return lex_get(lex, "to");
            }
        }
    }

    // ── in / In / IN ──
    {
        const bool in_lc = (word == "in" || word == "In");
        const bool in_uc = (word == "IN" && ptb_pos != "NNP");
        if (in_lc || in_uc) {
            if (ctx.future_vowel == 0 || ptb_pos != "IN")
                return "\xCB\x88\xC9\xAA" "n";                // "ˈɪn"
            return "\xC9\xAA" "n";                            // "ɪn"
        }
    }

    // ── the / The / THE ──
    {
        const bool the_lc = (word == "the" || word == "The");
        const bool the_uc = (word == "THE" && ptb_pos == "DT");
        if (the_lc || the_uc) {
            if (ctx.future_vowel == 1) return "\xC3\xB0i";    // "ði"
            return "\xC3\xB0\xC9\x99";                        // "ðə"
        }
    }

    // ── vs / vs. / VS / Vs.  (PTB IN, case-insensitive, optional dot) ──
    if (ptb_pos == "IN" && is_vs_form(word)) {
        return lex_get(lex, "versus");
    }

    // ── used / Used / USED ──
    {
        static constexpr std::array<std::string_view, 3> kUsed = {"used", "Used", "USED"};
        if (word_in(word, kUsed)) {
            if ((ptb_pos == "VBD" || ptb_pos == "JJ") && ctx.future_to)
                return lex_get(lex, "used", "VBD");
            return lex_get(lex, "used");
        }
    }

    return {};
}

std::string SpecialCases::spell_letter_by_letter(std::string_view word) const {
    if (word.empty()) return {};
    const Lexicon& lex = *lex_;

    std::string ps;
    ps.reserve(word.size() * 4);  // single-letter IPAs are short; pad enough
    bool any_letter = false;
    for (char c : word) {
        if (!ascii_alpha(c)) continue;
        const char upper = (c >= 'a' && c <= 'z')
                               ? static_cast<char>(c - ('a' - 'A'))
                               : c;
        const std::string_view letter(&upper, 1);
        const auto ipa = lex.lookup(letter);
        if (ipa.empty()) return {};
        ps.append(ipa.data(), ipa.size());
        any_letter = true;
    }
    if (!any_letter) return {};

    // Promote the LAST secondary stress (U+02CC = "\xCB\x8C") to primary
    // (U+02C8 = "\xCB\x88"). Both are 2-byte UTF-8 sequences; the
    // replacement is an in-place fixed-size byte edit.
    static constexpr std::string_view kSecondary = "\xCB\x8C";   // "ˌ"
    static constexpr std::string_view kPrimary   = "\xCB\x88";   // "ˈ"
    const auto pos = ps.rfind(kSecondary);
    if (pos != std::string::npos) {
        ps[pos]     = kPrimary[0];
        ps[pos + 1] = kPrimary[1];
    }
    return ps;
}

}  // namespace brosoundml::g2p
