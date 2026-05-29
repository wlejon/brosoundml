#include "brosoundml/g2p/normalizer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace brosoundml::g2p {

namespace {

// ─── UTF-8 helpers (one-codepoint walker, duplicated per-TU by convention) ──

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

// ─── Step 1: fold typographic apostrophes/quotes to ASCII ───────────────────

std::string fold_quotes(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ) {
        const std::size_t n = utf8_len_at(s, i);
        if (n == 0) { out.push_back(s[i]); ++i; continue; }
        const char32_t cp = utf8_decode_at(s, i, n);
        switch (cp) {
            case U'‘':  // left single quotation mark
            case U'’':  // right single quotation mark (the apostrophe LLMs emit)
            case U'ʼ':  // modifier letter apostrophe
                out.push_back('\'');
                break;
            case U'“':  // left double quotation mark
            case U'”':  // right double quotation mark
                out.push_back('"');
                break;
            default:
                out.append(s.data() + i, n);
                break;
        }
        i += n;
    }
    return out;
}

// ─── Number-to-words ────────────────────────────────────────────────────────

constexpr std::array<const char*, 20> ONES = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen"
};
constexpr std::array<const char*, 10> TENS = {
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety"
};
constexpr std::array<const char*, 7> SCALES = {
    "", "thousand", "million", "billion", "trillion", "quadrillion",
    "quintillion"
};

bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool is_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool is_alnum(char c) { return is_alpha(c) || is_digit(c); }
char lower(char c)    { return (c >= 'A' && c <= 'Z') ? char(c + 32) : c; }

// ─── Markdown stripping ─────────────────────────────────────────────────────
//
// LLMs frequently emit inline markdown even when asked not to. The markers are
// attached to words (*Hamlet*, **bold**, `code`), so the pre-tokenizer can't
// peel them and the word degrades to letter-by-letter spelling. Strip the
// formatting, keeping the readable text:
//   - links     [text](url) / ![alt](url) -> text / alt
//   - emphasis  *  **  ~~  -> removed; _ removed only at word boundaries
//   - code      `code` -> code
//   - line lead headings (#), blockquotes (>), and bullets (-, *, +) removed
// A '*' kept only between two digits survives as a multiplication operator
// (3*4 -> "three times four", handled downstream).
std::string strip_markdown(std::string_view in) {
    // Pass 1: unwrap links, dropping the URL.
    std::string a;
    a.reserve(in.size());
    for (std::size_t i = 0; i < in.size(); ) {
        if (in[i] == '!' && i + 1 < in.size() && in[i + 1] == '[') { ++i; continue; }
        if (in[i] == '[') {
            const std::size_t close = in.find(']', i + 1);
            if (close != std::string_view::npos && close + 1 < in.size() &&
                in[close + 1] == '(') {
                const std::size_t paren = in.find(')', close + 2);
                if (paren != std::string_view::npos) {
                    a.append(in.data() + i + 1, close - (i + 1));  // the link text
                    i = paren + 1;
                    continue;
                }
            }
        }
        a.push_back(in[i]);
        ++i;
    }

    // Pass 2: line-lead structure + inline markers.
    std::string out;
    out.reserve(a.size());
    bool at_line_start = true;
    for (std::size_t i = 0; i < a.size(); ) {
        if (at_line_start) {
            std::size_t j = i;
            while (j < a.size() && (a[j] == ' ' || a[j] == '\t')) ++j;
            // Heading: #{1,6} followed by space.
            std::size_t k = j; int hashes = 0;
            while (k < a.size() && a[k] == '#') { ++hashes; ++k; }
            if (hashes >= 1 && hashes <= 6 && k < a.size() && (a[k] == ' ' || a[k] == '\t')) {
                while (k < a.size() && (a[k] == ' ' || a[k] == '\t')) ++k;
                i = k; at_line_start = false; continue;
            }
            // Blockquote: one or more '>' with optional spaces.
            if (j < a.size() && a[j] == '>') {
                std::size_t b = j;
                while (b < a.size() && (a[b] == '>' || a[b] == ' ')) ++b;
                i = b; at_line_start = false; continue;
            }
            // Bullet: -, *, or + followed by whitespace.
            if (j + 1 < a.size() && (a[j] == '-' || a[j] == '*' || a[j] == '+') &&
                (a[j + 1] == ' ' || a[j + 1] == '\t')) {
                std::size_t b = j + 1;
                while (b < a.size() && (a[b] == ' ' || a[b] == '\t')) ++b;
                i = b; at_line_start = false; continue;
            }
            at_line_start = false;
        }

        const char c = a[i];
        if (c == '\n') { out.push_back('\n'); at_line_start = true; ++i; continue; }
        if (c == '`' || c == '~') { ++i; continue; }
        if (c == '*') {
            const char prev = out.empty() ? '\0' : out.back();
            const char next = (i + 1 < a.size()) ? a[i + 1] : '\0';
            if (is_digit(prev) && is_digit(next)) { out.push_back('*'); ++i; continue; }
            ++i; continue;  // emphasis/bold marker
        }
        if (c == '_') {
            const char prev = out.empty() ? '\0' : out.back();
            const char next = (i + 1 < a.size()) ? a[i + 1] : '\0';
            if (!(is_alnum(prev) && is_alnum(next))) { ++i; continue; }  // boundary → drop
        }
        out.push_back(c);
        ++i;
    }
    return out;
}

// 0..999 → words ("" for 0).
std::string three(unsigned n) {
    std::string s;
    const unsigned h = n / 100, r = n % 100;
    if (h) { s += ONES[h]; s += " hundred"; }
    if (r) {
        if (!s.empty()) s += ' ';
        if (r < 20) {
            s += ONES[r];
        } else {
            s += TENS[r / 10];
            if (r % 10) { s += ' '; s += ONES[r % 10]; }
        }
    }
    return s;
}

std::string cardinal(std::uint64_t n) {
    if (n == 0) return "zero";
    std::array<unsigned, 7> groups{};
    std::size_t ng = 0;
    while (n) { groups[ng++] = static_cast<unsigned>(n % 1000); n /= 1000; }
    std::string s;
    for (std::size_t gi = ng; gi-- > 0; ) {
        if (groups[gi] == 0) continue;
        if (!s.empty()) s += ' ';
        s += three(groups[gi]);
        if (gi > 0) { s += ' '; s += SCALES[gi]; }
    }
    return s;
}

// Spell digits one at a time — fallback for numbers too long for uint64.
std::string digitwise(const std::string& digits) {
    std::string s;
    for (char d : digits) {
        if (!is_digit(d)) continue;
        if (!s.empty()) s += ' ';
        s += ONES[d - '0'];
    }
    return s;
}

// Parse up to 18 digits into a uint64 (18 digits never overflows). Longer
// inputs return false so the caller falls back to digitwise().
bool to_u64(const std::string& d, std::uint64_t& out) {
    if (d.empty() || d.size() > 18) return false;
    std::uint64_t v = 0;
    for (char c : d) v = v * 10 + static_cast<std::uint64_t>(c - '0');
    out = v;
    return true;
}

// Make the last space-separated word of `words` ordinal.
std::string ordinalize(const std::string& words) {
    const std::size_t sp = words.find_last_of(' ');
    const std::string head = (sp == std::string::npos) ? "" : words.substr(0, sp + 1);
    const std::string last = (sp == std::string::npos) ? words : words.substr(sp + 1);

    struct Irr { const char* from; const char* to; };
    static constexpr std::array<Irr, 7> kIrr = {{
        {"one", "first"}, {"two", "second"}, {"three", "third"},
        {"five", "fifth"}, {"eight", "eighth"}, {"nine", "ninth"},
        {"twelve", "twelfth"},
    }};
    for (const auto& e : kIrr) {
        if (last == e.from) return head + e.to;
    }
    if (!last.empty() && last.back() == 'y') {
        return head + last.substr(0, last.size() - 1) + "ieth";  // twenty -> twentieth
    }
    return head + last + "th";  // four -> fourth, hundred -> hundredth
}

// 4-digit year reading: 1984 -> "nineteen eighty four", 2005 -> "twenty oh
// five", 1900 -> "nineteen hundred", 2000 -> "two thousand".
std::string year_reading(unsigned value) {
    const unsigned hi = value / 100, lo = value % 100;
    if (lo == 0) {
        if (hi % 10 == 0) return cardinal(value);       // 2000, 1000
        return cardinal(hi) + " hundred";               // 1900 -> nineteen hundred
    }
    std::string s = cardinal(hi);
    if (lo < 10) { s += " oh "; s += ONES[lo]; }         // 2005 -> twenty oh five
    else         { s += ' '; s += cardinal(lo); }        // 1984 -> nineteen eighty four
    return s;
}

// ─── Number-token parser ────────────────────────────────────────────────────

struct NumToken {
    std::string intDigits;   // grouping commas stripped
    std::string frac;        // fractional digits (empty if none)
    bool hasFrac   = false;
    bool hadComma  = false;
    bool isOrdinal = false;  // trailing st/nd/rd/th
    bool isPercent = false;  // trailing %
    std::size_t bytes = 0;   // bytes consumed from the start position
};

// Parse a numeric token starting at s[i] (s[i] must be a digit).
NumToken parse_number(const std::string& s, std::size_t i) {
    NumToken t;
    const std::size_t n = s.size();
    std::size_t j = i;

    while (j < n) {
        if (is_digit(s[j])) {
            t.intDigits.push_back(s[j]);
            ++j;
        } else if (s[j] == ',' && j + 3 < n &&
                   is_digit(s[j + 1]) && is_digit(s[j + 2]) && is_digit(s[j + 3]) &&
                   !(j + 4 < n && is_digit(s[j + 4]))) {
            t.hadComma = true;  // thousands separator: skip the comma, keep digits
            ++j;
        } else {
            break;
        }
    }

    if (j + 1 < n && s[j] == '.' && is_digit(s[j + 1])) {
        t.hasFrac = true;
        ++j;
        while (j < n && is_digit(s[j])) { t.frac.push_back(s[j]); ++j; }
    }

    if (!t.hasFrac && j + 1 < n) {
        const char a = lower(s[j]), b = lower(s[j + 1]);
        const bool suff = (a == 's' && b == 't') || (a == 'n' && b == 'd') ||
                          (a == 'r' && b == 'd') || (a == 't' && b == 'h');
        const bool boundary = (j + 2 >= n) || !is_alpha(s[j + 2]);
        if (suff && boundary) { t.isOrdinal = true; j += 2; }
    }
    if (!t.isOrdinal && j < n && s[j] == '%') { t.isPercent = true; ++j; }

    t.bytes = j - i;
    return t;
}

// Render a parsed numeric token to words (no currency context).
std::string render_number(const NumToken& t) {
    std::uint64_t val = 0;
    const bool ok = to_u64(t.intDigits, val);

    std::string num;
    if (t.hasFrac) {
        num = ok ? cardinal(val) : digitwise(t.intDigits);
        num += " point";
        for (char d : t.frac) { num += ' '; num += ONES[d - '0']; }
    } else if (!ok) {
        num = digitwise(t.intDigits);
    } else if (t.isOrdinal) {
        num = ordinalize(cardinal(val));
    } else if (!t.hadComma && t.intDigits.size() == 4 && val >= 1000 && val <= 2099) {
        num = year_reading(static_cast<unsigned>(val));
    } else {
        num = cardinal(val);
    }
    if (t.isPercent) num += " percent";
    return num;
}

// ─── Currency ───────────────────────────────────────────────────────────────

// Returns the byte length of a currency symbol at s[i] and sets `unit`
// (0=dollar, 1=pound, 2=euro), or 0 if none.
std::size_t currency_at(const std::string& s, std::size_t i, int& unit) {
    if (i >= s.size()) return 0;
    if (s[i] == '$') { unit = 0; return 1; }
    // £ = U+00A3 = 0xC2 0xA3
    if (i + 1 < s.size() &&
        static_cast<unsigned char>(s[i]) == 0xC2 &&
        static_cast<unsigned char>(s[i + 1]) == 0xA3) { unit = 1; return 2; }
    // € = U+20AC = 0xE2 0x82 0xAC
    if (i + 2 < s.size() &&
        static_cast<unsigned char>(s[i]) == 0xE2 &&
        static_cast<unsigned char>(s[i + 1]) == 0x82 &&
        static_cast<unsigned char>(s[i + 2]) == 0xAC) { unit = 2; return 3; }
    return 0;
}

std::string render_currency(int unit, const NumToken& t) {
    static constexpr std::array<const char*, 3> kMajorSing = {"dollar", "pound", "euro"};
    static constexpr std::array<const char*, 3> kMajorPlur = {"dollars", "pounds", "euros"};
    static constexpr std::array<const char*, 3> kMinorSing = {"cent", "penny", "cent"};
    static constexpr std::array<const char*, 3> kMinorPlur = {"cents", "pence", "cents"};

    std::uint64_t val = 0;
    const bool ok = to_u64(t.intDigits, val);
    const std::string major = ok ? cardinal(val) : digitwise(t.intDigits);

    int cents = -1;
    if (t.hasFrac) {
        std::string c2 = t.frac.substr(0, 2);
        while (c2.size() < 2) c2.push_back('0');     // ".5" -> 50 cents
        cents = (c2[0] - '0') * 10 + (c2[1] - '0');
    }

    // "$0.50" reads as just the cents.
    if (ok && val == 0 && cents > 0) {
        return cardinal(static_cast<std::uint64_t>(cents)) + " " +
               (cents == 1 ? kMinorSing[unit] : kMinorPlur[unit]);
    }

    std::string w = major + " " + (ok && val == 1 ? kMajorSing[unit] : kMajorPlur[unit]);
    if (cents > 0) {
        w += " and " + cardinal(static_cast<std::uint64_t>(cents)) + " " +
             (cents == 1 ? kMinorSing[unit] : kMinorPlur[unit]);
    }
    return w;
}

}  // namespace

// ─── Public entry point ─────────────────────────────────────────────────────

std::string normalize_text(std::string_view sentence) {
    const std::string s = fold_quotes(strip_markdown(sentence));
    const std::size_t n = s.size();

    std::string out;
    out.reserve(n + 16);

    std::size_t i = 0;
    while (i < n) {
        // Currency prefix immediately (allowing spaces) before a number.
        int unit = 0;
        const std::size_t clen = currency_at(s, i, unit);
        if (clen) {
            std::size_t k = i + clen;
            while (k < n && s[k] == ' ') ++k;
            if (k < n && is_digit(s[k])) {
                NumToken t = parse_number(s, k);
                out += ' ';
                out += render_currency(unit, t);
                out += ' ';
                i = k + t.bytes;
                continue;
            }
            // Bare symbol with no number — pass through verbatim.
            out.append(s, i, clen);
            i += clen;
            continue;
        }

        const char c = s[i];
        if (is_digit(c)) {
            NumToken t = parse_number(s, i);
            out += ' ';
            out += render_number(t);
            out += ' ';
            i += t.bytes;
            continue;
        }

        // Bare operators → spoken words (spaced so they tokenize cleanly).
        switch (c) {
            case '+': out += " plus ";    ++i; continue;
            case '*': out += " times ";   ++i; continue;
            case '=': out += " equals ";  ++i; continue;
            case '&': out += " and ";     ++i; continue;
            case '@': out += " at ";      ++i; continue;
            case '%': out += " percent "; ++i; continue;
            default: break;
        }

        // Negative sign: '-' before a digit, at a word boundary, reads as
        // "minus". Internal hyphens (well-known) are left untouched.
        if (c == '-' && i + 1 < n && is_digit(s[i + 1])) {
            const char prev = out.empty() ? '\0' : out.back();
            if (prev == '\0' || prev == ' ' || prev == '(') {
                out += " minus ";
                ++i;
                continue;
            }
        }

        out.push_back(c);
        ++i;
    }

    // Tidy the expansion padding: collapse whitespace runs to single spaces,
    // drop the space before closing punctuation, and trim. Functionally inert
    // for the pre-tokenizer, but keeps the normalized text clean for display
    // and for the unit tests.
    auto is_close_punct = [](char ch) {
        return ch == ',' || ch == '.' || ch == ';' || ch == ':' ||
               ch == '!' || ch == '?' || ch == ')' || ch == ']' || ch == '}';
    };
    std::string clean;
    clean.reserve(out.size());
    for (std::size_t k = 0; k < out.size(); ) {
        if (out[k] == ' ' || out[k] == '\t' || out[k] == '\n' || out[k] == '\r') {
            std::size_t m = k;
            while (m < out.size() &&
                   (out[m] == ' ' || out[m] == '\t' || out[m] == '\n' || out[m] == '\r')) {
                ++m;
            }
            if (!clean.empty() && (m >= out.size() || !is_close_punct(out[m]))) {
                clean.push_back(' ');
            }
            k = m;
        } else {
            clean.push_back(out[k]);
            ++k;
        }
    }
    while (!clean.empty() && clean.back() == ' ') clean.pop_back();
    return clean;
}

}  // namespace brosoundml::g2p
