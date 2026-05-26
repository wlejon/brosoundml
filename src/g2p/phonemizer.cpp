#include "brosoundml/g2p/phonemizer.h"

#include "brosoundml/g2p/lexicon.h"
#include "brosoundml/g2p/morphology.h"
#include "brosoundml/g2p/special_cases.h"
#include "brosoundml/g2p/pos_tagger.h"
#include "brosoundml/g2p/phoneme_adapter.h"
#include "brosoundml/g2p/tags.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brosoundml::g2p {

namespace {

// ─── UTF-8 helpers ───────────────────────────────────────────────────────
//
// Same one-codepoint walker as phoneme_adapter.cpp — duplicated rather than
// shared to keep slice 5 a single-TU contribution. Returns the encoded byte
// length of the codepoint starting at byte index `i`, or 0 on a malformed
// lead / truncation.

std::size_t utf8_len_at(std::string_view s, std::size_t i) {
    if (i >= s.size()) return 0;
    const auto b = static_cast<unsigned char>(s[i]);
    std::size_t n = 0;
    if      ((b & 0x80u) == 0x00u) n = 1;
    else if ((b & 0xE0u) == 0xC0u) n = 2;
    else if ((b & 0xF0u) == 0xE0u) n = 3;
    else if ((b & 0xF8u) == 0xF0u) n = 4;
    else                           return 0;
    if (i + n > s.size()) return 0;
    for (std::size_t k = 1; k < n; ++k) {
        if ((static_cast<unsigned char>(s[i + k]) & 0xC0u) != 0x80u) return 0;
    }
    return n;
}

// Decode one codepoint at byte index `i` into a char32_t. Returns 0 on a
// malformed sequence. Mirrors the morphology slice's decoder.
char32_t utf8_decode_at(std::string_view s, std::size_t i, std::size_t n) {
    if (n == 0 || i + n > s.size()) return 0;
    const auto b0 = static_cast<unsigned char>(s[i]);
    char32_t cp = 0;
    switch (n) {
        case 1: cp = b0 & 0x7Fu; break;
        case 2: cp =  (b0 & 0x1Fu) << 6;
                cp |= (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
                break;
        case 3: cp =  (b0 & 0x0Fu) << 12;
                cp |= (static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6;
                cp |= (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
                break;
        case 4: cp =  (b0 & 0x07u) << 18;
                cp |= (static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12;
                cp |= (static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6;
                cp |= (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
                break;
        default: return 0;
    }
    return cp;
}

bool is_ascii_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// ─── Pre-tokeniser ───────────────────────────────────────────────────────

struct Triple {
    std::string leading;
    std::string word;
    std::string trailing;
};

// Leading-punctuation codepoints (peeled from front).
//   ( [ { " '  and smart "  '
constexpr std::array<char32_t, 7> kLeadingPunct = {
    U'(', U'[', U'{', U'"', U'\'', U'“', U'‘'
};
// Trailing-punctuation codepoints (peeled from back).
//   . , ; : ! ? ) ] } " '  plus smart "  '  em-dash, ellipsis
constexpr std::array<char32_t, 15> kTrailingPunct = {
    U'.', U',', U';', U':', U'!', U'?',
    U')', U']', U'}', U'"', U'\'',
    U'”', U'’', U'—', U'…'
};

template <std::size_t N>
bool cp_in(char32_t cp, const std::array<char32_t, N>& set) {
    for (auto v : set) if (v == cp) return true;
    return false;
}

std::vector<Triple> pretokenize(std::string_view sentence) {
    std::vector<Triple> out;

    std::size_t i = 0;
    while (i < sentence.size()) {
        // Skip whitespace.
        while (i < sentence.size() && is_ascii_ws(sentence[i])) ++i;
        if (i >= sentence.size()) break;

        // Find the end of this whitespace-delimited chunk.
        const std::size_t chunk_start = i;
        while (i < sentence.size() && !is_ascii_ws(sentence[i])) ++i;
        const std::size_t chunk_end = i;

        std::string_view chunk = sentence.substr(chunk_start, chunk_end - chunk_start);

        // Peel leading-punct codepoints.
        std::size_t lead_end = 0;
        while (lead_end < chunk.size()) {
            const std::size_t n = utf8_len_at(chunk, lead_end);
            if (n == 0) break;
            const char32_t cp = utf8_decode_at(chunk, lead_end, n);
            if (!cp_in(cp, kLeadingPunct)) break;
            lead_end += n;
        }

        // Peel trailing-punct codepoints by walking codepoints from chunk[lead_end..].
        // We need a list of codepoint byte boundaries to walk backward safely.
        std::vector<std::size_t> boundaries;  // byte offsets within chunk
        boundaries.reserve(chunk.size() - lead_end + 1);
        boundaries.push_back(lead_end);
        for (std::size_t p = lead_end; p < chunk.size(); ) {
            const std::size_t n = utf8_len_at(chunk, p);
            if (n == 0) { ++p; boundaries.push_back(p); continue; }
            p += n;
            boundaries.push_back(p);
        }

        // Walk back from the end.
        std::size_t trail_start = chunk.size();
        while (boundaries.size() >= 2) {
            const std::size_t b_end   = boundaries.back();
            const std::size_t b_start = boundaries[boundaries.size() - 2];
            const std::size_t n = b_end - b_start;
            const char32_t cp = utf8_decode_at(chunk, b_start, n);
            if (!cp_in(cp, kTrailingPunct)) break;
            trail_start = b_start;
            boundaries.pop_back();
        }

        if (trail_start < lead_end) trail_start = lead_end;  // safety

        Triple t;
        t.leading.assign(chunk.data(), lead_end);
        t.word.assign(chunk.data() + lead_end, trail_start - lead_end);
        t.trailing.assign(chunk.data() + trail_start, chunk.size() - trail_start);

        // Skip pure-punctuation chunks (empty word).
        if (!t.word.empty()) out.push_back(std::move(t));
    }

    return out;
}

// ─── Context propagation codepoint sets ──────────────────────────────────
//
// Mirror the sets documented in docs/phonemizer.md § "Right-to-left
// phonemization". Indexed by codepoint via a small linear search — N is tiny
// so std::array beats unordered_set for this hot path.

constexpr std::array<char32_t, 20> kVowels = {
    U'A', U'I', U'O', U'Q', U'W', U'Y',
    U'a', U'i', U'u',
    U'æ',  // æ
    U'ɑ',  // ɑ
    U'ɒ',  // ɒ
    U'ɔ',  // ɔ
    U'ə',  // ə
    U'ɛ',  // ɛ
    U'ɜ',  // ɜ
    U'ɪ',  // ɪ
    U'ʊ',  // ʊ
    U'ʌ',  // ʌ
    U'ᵻ',  // ᵻ
};

constexpr std::array<char32_t, 25> kConsonants = {
    U'b', U'd', U'f', U'h', U'j', U'k', U'l', U'm', U'n',
    U'p', U's', U't', U'v', U'w', U'z',
    U'ð',  // ð
    U'ŋ',  // ŋ
    U'ɡ',  // ɡ (Latin script small g — IPA voiced velar plosive)
    U'ɹ',  // ɹ
    U'ɾ',  // ɾ
    U'ʃ',  // ʃ
    U'ʒ',  // ʒ
    U'ʤ',  // ʤ
    U'ʧ',  // ʧ
    U'θ',  // θ
};

constexpr std::array<char32_t, 6> kNonQuotePunct = {
    U'!', U',', U'.', U':', U';', U'?'
};

// Update the rolling future context from the IPA assigned to the current
// token. Returns the context to feed the *previous* token (we walk
// right-to-left, so this context describes what comes *after* that previous
// token — which is the token we just processed).
TokenContext update_ctx(const TokenContext& ctx,
                        std::string_view    ipa,
                        std::string_view    word,
                        std::string_view    ptb) {
    TokenContext next = ctx;

    // Scan ipa for the first vowel / consonant / non-quote-punct codepoint.
    for (std::size_t i = 0; i < ipa.size(); ) {
        const std::size_t n = utf8_len_at(ipa, i);
        if (n == 0) { ++i; continue; }
        const char32_t cp = utf8_decode_at(ipa, i, n);
        if (cp_in(cp, kVowels))         { next.future_vowel =  1; break; }
        if (cp_in(cp, kConsonants))     { next.future_vowel = -1; break; }
        if (cp_in(cp, kNonQuotePunct))  { next.future_vowel =  0; break; }
        i += n;
    }

    // future_to: per misaki's token_context, the next token is "to" / "To",
    // or "TO" when its tag is TO or IN.
    next.future_to = (word == "to" || word == "To"
                      || (word == "TO" && (ptb == "TO" || ptb == "IN")));
    return next;
}

// ─── PTB tag name lookup ─────────────────────────────────────────────────

std::string_view ptb_name(PosTag t) {
    const auto idx = static_cast<std::size_t>(t);
    if (idx >= static_cast<std::size_t>(NUM_TAGS)) return "";
    return kPosTagNames[idx];
}

// ─── ASCII helpers ───────────────────────────────────────────────────────

bool has_ascii_upper(std::string_view s) {
    for (char c : s) if (c >= 'A' && c <= 'Z') return true;
    return false;
}

std::string ascii_tolower(std::string_view s) {
    std::string r;
    r.reserve(s.size());
    for (char c : s) {
        r.push_back((c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c);
    }
    return r;
}

// ─── Per-word phonemisation ──────────────────────────────────────────────

std::string phonemize_word(std::string_view     word,
                           std::string_view     ptb,
                           const TokenContext&  ctx,
                           const Lexicon&       lex,
                           const Morphology&    morph,
                           const SpecialCases&  sc) {
    // 1. Special cases (function-word allomorphy, symbol words, vs., used …).
    std::string out = sc.try_phonemize(word, ptb, ctx);
    if (!out.empty()) return out;

    // 2. Lexicon direct lookup.
    out = std::string(lex.lookup(word, ptb));
    if (!out.empty()) return out;

    // 3. Morphology (-s / -ed / -ing / possessives).
    out = morph.try_phonemize(word, ptb);
    if (!out.empty()) return out;

    // 4. Simplified case-fold pre-rule: if the word had any ASCII upper,
    //    retry the lexicon + morphology chain on the lowercased form.
    if (has_ascii_upper(word)) {
        const std::string wl = ascii_tolower(word);
        out = std::string(lex.lookup(wl, ptb));
        if (!out.empty()) return out;
        out = morph.try_phonemize(wl, ptb);
        if (!out.empty()) return out;
    }

    // 5. NNP fallback: spell letter by letter. Returns "" if the word has no
    //    spellable letters at all.
    return sc.spell_letter_by_letter(word);
}

// ─── Pipeline core (shared by phonemize / phonemize_to_ipa) ──────────────

struct PipelineResult {
    std::vector<Triple> triples;
    std::vector<std::string> ipa_per_token;  // stem IPA only, no punct
};

PipelineResult run_pipeline(std::string_view     sentence,
                            const PosTagger&     tagger,
                            const Lexicon&       lex,
                            const Morphology&    morph,
                            const SpecialCases&  sc) {
    PipelineResult r;
    r.triples = pretokenize(sentence);
    if (r.triples.empty()) return r;

    // Build the joined sentence for the POS tagger. The tagger borrows
    // string_views into `joined`, so the lifetime of `joined` must cover the
    // for-loop below.
    std::string joined;
    {
        std::size_t cap = 0;
        for (const auto& t : r.triples) cap += t.word.size() + 1;
        joined.reserve(cap);
        for (std::size_t i = 0; i < r.triples.size(); ++i) {
            if (i > 0) joined.push_back(' ');
            joined.append(r.triples[i].word);
        }
    }

    const auto tags = tagger.tag(joined);
    if (tags.size() != r.triples.size()) {
        throw std::runtime_error(
            std::string("brosoundml: g2p::Phonemizer::tag_align: tagger returned ")
            + std::to_string(tags.size()) + " tags for "
            + std::to_string(r.triples.size()) + " words");
    }

    r.ipa_per_token.assign(r.triples.size(), std::string{});

    TokenContext ctx;  // last token has no successor → defaults
    for (std::size_t k = r.triples.size(); k-- > 0; ) {
        const auto& t = r.triples[k];
        const std::string_view ptb = ptb_name(tags[k].tag);
        std::string ipa = phonemize_word(t.word, ptb, ctx, lex, morph, sc);
        // Update ctx for the previous (left-neighbour) token.
        ctx = update_ctx(ctx, ipa, t.word, ptb);
        r.ipa_per_token[k] = std::move(ipa);
    }

    return r;
}

std::string assemble_ipa(const PipelineResult& r) {
    std::string buf;
    std::size_t cap = 0;
    for (std::size_t i = 0; i < r.triples.size(); ++i) {
        cap += r.triples[i].leading.size()
             + r.ipa_per_token[i].size()
             + r.triples[i].trailing.size()
             + 1;
    }
    buf.reserve(cap);
    for (std::size_t i = 0; i < r.triples.size(); ++i) {
        if (i > 0) buf.push_back(' ');
        buf.append(r.triples[i].leading);
        buf.append(r.ipa_per_token[i]);
        buf.append(r.triples[i].trailing);
    }
    return buf;
}

}  // namespace

// ─── Phonemizer methods ──────────────────────────────────────────────────

Phonemizer::Phonemizer(const PosTagger&      tagger,
                       const Lexicon&        lexicon,
                       const Morphology&     morphology,
                       const SpecialCases&   special,
                       const PhonemeAdapter& adapter)
    : tagger_(&tagger),
      lexicon_(&lexicon),
      morphology_(&morphology),
      special_(&special),
      adapter_(&adapter) {}

std::vector<std::int32_t> Phonemizer::phonemize(std::string_view sentence) const {
    PipelineResult r = run_pipeline(sentence, *tagger_, *lexicon_,
                                    *morphology_, *special_);
    if (r.triples.empty()) return {};
    const std::string ipa = assemble_ipa(r);
    return adapter_->encode(ipa);
}

std::string Phonemizer::phonemize_to_ipa(std::string_view sentence) const {
    PipelineResult r = run_pipeline(sentence, *tagger_, *lexicon_,
                                    *morphology_, *special_);
    if (r.triples.empty()) return {};
    return assemble_ipa(r);
}

}  // namespace brosoundml::g2p
