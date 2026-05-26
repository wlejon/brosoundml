#include "brosoundml/g2p/morphology.h"

#include "brosoundml/g2p/lexicon.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace brosoundml::g2p {

namespace {

// ─── UTF-8 codepoint helpers ─────────────────────────────────────────────
//
// The IPA blob is UTF-8 and the codepoints used by the affix glue (Greek θ,
// IPA Latin extended ʃ ʒ ʧ ʤ æ ɑ ə ɛ ɪ ɹ ʊ ʌ ɾ ŋ) all sit in the BMP, so a
// minimal decoder covering 1-4 byte sequences suffices. Returns `{0, 0}` on
// an empty input or a malformed lead/continuation byte — callers treat that
// as "rule did not fire".

struct CodePoint {
    char32_t cp;
    std::size_t bytes;  // encoded length of `cp` at the inspected position
};

// Forward-decode a single codepoint starting at byte index `i` of `s`.
// Returns `{0, 0}` on malformed input.
CodePoint decode_at(std::string_view s, std::size_t i) {
    if (i >= s.size()) return {0, 0};
    const auto b0 = static_cast<unsigned char>(s[i]);
    if (b0 < 0x80) {
        return {static_cast<char32_t>(b0), 1};
    }
    auto cont = [&](std::size_t k, char32_t& acc) -> bool {
        if (k >= s.size()) return false;
        const auto bk = static_cast<unsigned char>(s[k]);
        if ((bk & 0xC0) != 0x80) return false;
        acc = (acc << 6) | static_cast<char32_t>(bk & 0x3F);
        return true;
    };
    if ((b0 & 0xE0) == 0xC0) {
        char32_t cp = static_cast<char32_t>(b0 & 0x1F);
        if (!cont(i + 1, cp)) return {0, 0};
        return {cp, 2};
    }
    if ((b0 & 0xF0) == 0xE0) {
        char32_t cp = static_cast<char32_t>(b0 & 0x0F);
        if (!cont(i + 1, cp)) return {0, 0};
        if (!cont(i + 2, cp)) return {0, 0};
        return {cp, 3};
    }
    if ((b0 & 0xF8) == 0xF0) {
        char32_t cp = static_cast<char32_t>(b0 & 0x07);
        if (!cont(i + 1, cp)) return {0, 0};
        if (!cont(i + 2, cp)) return {0, 0};
        if (!cont(i + 3, cp)) return {0, 0};
        return {cp, 4};
    }
    return {0, 0};
}

// Returns the final codepoint of `s` and the byte length of its UTF-8
// encoding. `{0, 0}` if `s` is empty or malformed.
CodePoint last_codepoint(std::string_view s) {
    if (s.empty()) return {0, 0};
    // Step back to the start byte of the final codepoint: the last byte
    // whose high two bits are not `10` (continuation).
    std::size_t i = s.size();
    while (i > 0) {
        --i;
        const auto b = static_cast<unsigned char>(s[i]);
        if ((b & 0xC0) != 0x80) break;
        // Defensive bound — UTF-8 codepoints used here are at most 4 bytes.
        if (s.size() - i > 4) return {0, 0};
    }
    const auto b = static_cast<unsigned char>(s[i]);
    if ((b & 0xC0) == 0x80) return {0, 0};  // ran off the front
    const CodePoint cp = decode_at(s, i);
    if (cp.bytes == 0) return {0, 0};
    if (i + cp.bytes != s.size()) return {0, 0};  // trailing junk
    return cp;
}

// Returns just the codepoint immediately preceding the final codepoint of
// `s`, or 0 if no such codepoint exists / `s` is malformed.
char32_t second_to_last_codepoint(std::string_view s) {
    const CodePoint last = last_codepoint(s);
    if (last.bytes == 0) return 0;
    std::string_view prefix = s;
    prefix.remove_suffix(last.bytes);
    return last_codepoint(prefix).cp;
}

// ─── Codepoint sets (single-codepoint each, linear scan is fine at N≤16) ──

// IPA characters used in glue tables:
//   p=0x70, t=0x74, k=0x6B, f=0x66, s=0x73, z=0x7A, d=0x64
//   θ=U+03B8, ʃ=U+0283, ʒ=U+0292, ʧ=U+02A7, ʤ=U+02A4
//   æ=U+00E6, ɑ=U+0251, ə=U+0259, ɛ=U+025B, ɪ=U+026A,
//   ɹ=U+0279, ʊ=U+028A, ʌ=U+028C
//   ɾ=U+027E, ŋ=U+014B
constexpr char32_t kTheta = 0x03B8;
constexpr char32_t kSh    = 0x0283;
constexpr char32_t kZh    = 0x0292;
constexpr char32_t kCh    = 0x02A7;  // ʧ
constexpr char32_t kJh    = 0x02A4;  // ʤ

// -s glue: voiceless non-sibilants → append "s".
constexpr std::array<char32_t, 5> kSVoiceless = {U'p', U't', U'k', U'f', kTheta};
// -s glue: sibilants → append "ᵻz".
constexpr std::array<char32_t, 6> kSSibilant  = {U's', U'z', kSh, kZh, kCh, kJh};

// -ed glue: voiceless → append "t".
constexpr std::array<char32_t, 7> kEdVoiceless = {U'p', U'k', U'f', kTheta, kSh, U's', kCh};

// US_TAUS: codepoints triggering the intervocalic /t/ → flap (ɾ) allophone
// for both -ed and -ing.
constexpr std::array<char32_t, 15> kUsTaus = {
    U'A', U'I', U'O', U'W', U'Y', U'i', U'u',
    0x00E6,  // æ
    0x0251,  // ɑ
    0x0259,  // ə
    0x025B,  // ɛ
    0x026A,  // ɪ
    0x0279,  // ɹ
    0x028A,  // ʊ
    0x028C,  // ʌ
};

template <std::size_t N>
bool in_set(char32_t c, const std::array<char32_t, N>& set) {
    for (auto v : set) if (v == c) return true;
    return false;
}

// ─── Glue tables ─────────────────────────────────────────────────────────

// Append the misaki `_s` glue to `stem_ipa`. Returns the assembled string.
std::string apply_s_glue(std::string_view stem_ipa) {
    const char32_t last = last_codepoint(stem_ipa).cp;
    if (last == 0) return {};
    std::string out;
    if (in_set(last, kSVoiceless)) {
        out.reserve(stem_ipa.size() + 1);
        out.assign(stem_ipa);
        out.push_back('s');
    } else if (in_set(last, kSSibilant)) {
        // U+1D7B  LATIN SMALL CAPITAL I WITH STROKE (ᵻ) = E1 B5 BB
        // followed by ASCII 'z'.
        // See kIzGlue split-literal note in apply_ed_glue: "\xBBz" is fine
        // ('z' is not a hex digit) but we keep the split-literal form for
        // visual consistency with the other glue tables.
        static constexpr char kIzGlue[] = "\xE1\xB5\xBB" "z";  // "ᵻz"
        out.reserve(stem_ipa.size() + 4);
        out.assign(stem_ipa);
        out.append(kIzGlue, 4);
    } else {
        out.reserve(stem_ipa.size() + 1);
        out.assign(stem_ipa);
        out.push_back('z');
    }
    return out;
}

// Append/replace per the misaki `_ed` US-branch glue table.
std::string apply_ed_glue(std::string_view stem_ipa) {
    const CodePoint last = last_codepoint(stem_ipa);
    if (last.bytes == 0) return {};
    // Split string literals: MSVC would otherwise extend "\xBBd" into a
    // single hex escape (d is a hex digit). The concatenation here keeps
    // the bytes "\xE1\xB5\xBB" followed by an ASCII 'd'.
    static constexpr char kIzGlue[] = "\xE1\xB5\xBB" "d";              // "ᵻd"
    static constexpr char kFlap[]   = "\xC9\xBE\xE1\xB5\xBB" "d";      // "ɾᵻd"
    std::string out;
    if (in_set(last.cp, kEdVoiceless)) {
        out.reserve(stem_ipa.size() + 1);
        out.assign(stem_ipa);
        out.push_back('t');
    } else if (last.cp == U'd') {
        out.reserve(stem_ipa.size() + 4);
        out.assign(stem_ipa);
        out.append(kIzGlue, 4);
    } else if (last.cp != U't') {
        out.reserve(stem_ipa.size() + 1);
        out.assign(stem_ipa);
        out.push_back('d');
    } else {
        // Stem-final 't' — distinguish flap vs. /ᵻd/ vs. short stem fallback.
        // "second-to-last codepoint" requires at least two codepoints, which
        // for the IPA characters involved means ≥ 2 bytes — spec's "len < 2"
        // is the misaki guard against single-codepoint stems.
        const char32_t prev = second_to_last_codepoint(stem_ipa);
        if (prev != 0 && in_set(prev, kUsTaus)) {
            // Replace the trailing 't' with "ɾᵻd".
            const std::size_t trunc = stem_ipa.size() - last.bytes;
            out.reserve(trunc + 6);
            out.assign(stem_ipa.substr(0, trunc));
            out.append(kFlap, 6);
        } else {
            out.reserve(stem_ipa.size() + 4);
            out.assign(stem_ipa);
            out.append(kIzGlue, 4);
        }
    }
    return out;
}

// Append/replace per the misaki `_ing` US-branch glue table.
std::string apply_ing_glue(std::string_view stem_ipa) {
    const CodePoint last = last_codepoint(stem_ipa);
    if (last.bytes == 0) return {};
    static constexpr char kIng[]     = "\xC9\xAA\xC5\x8B";        // "ɪŋ"
    static constexpr char kFlapIng[] = "\xC9\xBE\xC9\xAA\xC5\x8B";  // "ɾɪŋ"
    std::string out;
    if (last.cp == U't') {
        const char32_t prev = second_to_last_codepoint(stem_ipa);
        if (prev != 0 && in_set(prev, kUsTaus)) {
            const std::size_t trunc = stem_ipa.size() - last.bytes;
            out.reserve(trunc + 6);
            out.assign(stem_ipa.substr(0, trunc));
            out.append(kFlapIng, 6);
            return out;
        }
    }
    out.reserve(stem_ipa.size() + 4);
    out.assign(stem_ipa);
    out.append(kIng, 4);
    return out;
}

// ─── -ing doubled-consonant tail check ────────────────────────────────────
//
// Mirrors misaki's regex `r"([bcdgklmnprstvxz])\1ing$|cking$"`: the suffix
// is either `cking` (the orthographic `ck`-doubling case) or `Cing` where
// C ∈ {b,c,d,g,k,l,m,n,p,r,s,t,v,x,z} and the two consonants immediately
// before `ing` are the same letter.
bool tail_is_doubled_cons_ing(std::string_view w) {
    if (w.size() < 5) return false;
    if (w.size() >= 5 && w.compare(w.size() - 5, 5, "cking") == 0) return true;
    // Gemination: word[size-5] == word[size-4], both in the allowed class.
    const char c1 = w[w.size() - 5];
    const char c2 = w[w.size() - 4];
    if (c1 != c2) return false;
    constexpr std::string_view kAllowed = "bcdgklmnprstvxz";
    return kAllowed.find(c1) != std::string_view::npos;
}

// ─── Rule helpers ─────────────────────────────────────────────────────────

bool ends_with(std::string_view w, std::string_view suffix) {
    return w.size() >= suffix.size() &&
           w.compare(w.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string try_possessive_s_quote(const Lexicon& lex,
                                   std::string_view word,
                                   std::string_view ptb_pos) {
    // word ends with "s'" → rewrite to word[:-2] + "'s".
    if (!ends_with(word, "s'")) return {};
    std::string key;
    key.reserve(word.size());
    key.append(word.data(), word.size() - 2);
    key.append("'s");
    const auto ipa = lex.lookup(key, ptb_pos);
    if (ipa.empty()) return {};
    return std::string(ipa);
}

std::string try_possessive_trailing_quote(const Lexicon& lex,
                                          std::string_view word,
                                          std::string_view ptb_pos) {
    // word ends with "'" (but not "s'", which was tried above) → rewrite
    // to word[:-1].
    if (word.empty() || word.back() != '\'') return {};
    const std::string_view key = word.substr(0, word.size() - 1);
    const auto ipa = lex.lookup(key, ptb_pos);
    if (ipa.empty()) return {};
    return std::string(ipa);
}

std::string try_stem_s(const Lexicon& lex,
                       std::string_view word,
                       std::string_view ptb_pos) {
    if (word.size() < 3 || word.back() != 's') return {};

    // Candidate 1: drop a single trailing 's' (when not part of "ss").
    if (!ends_with(word, "ss")) {
        const std::string_view stem = word.substr(0, word.size() - 1);
        const auto ipa = lex.lookup(stem, ptb_pos);
        if (!ipa.empty()) return apply_s_glue(ipa);
    }

    // Candidate 2: drop "'s", or drop "es" (when not "ies" and len > 4).
    const bool apos_s = ends_with(word, "'s");
    const bool es_not_ies =
        word.size() > 4 && ends_with(word, "es") && !ends_with(word, "ies");
    if (apos_s || es_not_ies) {
        const std::string_view stem = word.substr(0, word.size() - 2);
        const auto ipa = lex.lookup(stem, ptb_pos);
        if (!ipa.empty()) return apply_s_glue(ipa);
    }

    // Candidate 3: "ies" → "y".
    if (word.size() > 4 && ends_with(word, "ies")) {
        std::string stem;
        stem.reserve(word.size() - 2);
        stem.append(word.data(), word.size() - 3);
        stem.push_back('y');
        const auto ipa = lex.lookup(stem, ptb_pos);
        if (!ipa.empty()) return apply_s_glue(ipa);
    }

    return {};
}

std::string try_stem_ed(const Lexicon& lex,
                        std::string_view word,
                        std::string_view ptb_pos) {
    if (word.size() < 4 || word.back() != 'd') return {};

    // Candidate 1: drop trailing 'd' (when not "dd").
    if (!ends_with(word, "dd")) {
        const std::string_view stem = word.substr(0, word.size() - 1);
        const auto ipa = lex.lookup(stem, ptb_pos);
        if (!ipa.empty()) return apply_ed_glue(ipa);
    }

    // Candidate 2: drop "ed" (when not "eed" and len > 4).
    if (word.size() > 4 && ends_with(word, "ed") && !ends_with(word, "eed")) {
        const std::string_view stem = word.substr(0, word.size() - 2);
        const auto ipa = lex.lookup(stem, ptb_pos);
        if (!ipa.empty()) return apply_ed_glue(ipa);
    }

    return {};
}

std::string try_stem_ing(const Lexicon& lex,
                         std::string_view word,
                         std::string_view ptb_pos) {
    if (word.size() < 5 || !ends_with(word, "ing")) return {};

    // Candidate 1: drop trailing "ing" (len > 5).
    if (word.size() > 5) {
        const std::string_view stem = word.substr(0, word.size() - 3);
        const auto ipa = lex.lookup(stem, ptb_pos);
        if (!ipa.empty()) return apply_ing_glue(ipa);
    }

    // Candidate 2: drop "ing", restore "e".
    {
        std::string stem;
        stem.reserve(word.size() - 2);
        stem.append(word.data(), word.size() - 3);
        stem.push_back('e');
        const auto ipa = lex.lookup(stem, ptb_pos);
        if (!ipa.empty()) return apply_ing_glue(ipa);
    }

    // Candidate 3: doubled-consonant gemination or "cking" — drop the
    // doubled letter as well as "ing".
    if (word.size() > 5 && tail_is_doubled_cons_ing(word)) {
        const std::string_view stem = word.substr(0, word.size() - 4);
        const auto ipa = lex.lookup(stem, ptb_pos);
        if (!ipa.empty()) return apply_ing_glue(ipa);
    }

    return {};
}

}  // namespace

// ─── Public API ──────────────────────────────────────────────────────────

Morphology::Morphology(const Lexicon& lex) : lex_(&lex) {}

std::string Morphology::try_phonemize(std::string_view word,
                                      std::string_view ptb_pos) const {
    if (word.empty()) return {};
    const Lexicon& lex = *lex_;

    if (auto r = try_possessive_s_quote(lex, word, ptb_pos); !r.empty()) return r;
    if (auto r = try_possessive_trailing_quote(lex, word, ptb_pos); !r.empty()) return r;
    if (auto r = try_stem_s(lex, word, ptb_pos);   !r.empty()) return r;
    if (auto r = try_stem_ed(lex, word, ptb_pos);  !r.empty()) return r;
    if (auto r = try_stem_ing(lex, word, ptb_pos); !r.empty()) return r;
    return {};
}

}  // namespace brosoundml::g2p
