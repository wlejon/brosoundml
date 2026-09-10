// OmniVoice prompt layer — see omnivoice_prompt.h. Every function here is a
// port of an upstream Python rule (omnivoice/models/omnivoice.py and
// omnivoice/utils/{duration,voice_design,text,audio,lang_map}.py at the
// fixture's pinned commit), written to reproduce the Python result exactly:
// the same operation order in the same precision, Python's Unicode string
// semantics over decoded code points, and pydub's int16 / millisecond
// arithmetic for the silence tools.

#include "omnivoice_prompt.h"

#include <brolm/detail/unicode.h>
#include <brolm/qwen_tokenizer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brosoundml {
namespace ovp {

namespace uni = brolm::detail::unicode;

// ═══════════════════════════════════════════════════════════════════════════
// Code points and Python string semantics
// ═══════════════════════════════════════════════════════════════════════════

std::vector<uint32_t> to_codepoints(std::string_view s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) out.push_back(uni::decode_utf8(s, i));
    return out;
}

std::string from_codepoints(const std::vector<uint32_t>& cps) {
    std::string out;
    out.reserve(cps.size());
    for (uint32_t cp : cps) uni::encode_utf8(cp, out);
    return out;
}

bool is_cjk(uint32_t cp) { return cp >= 0x4E00 && cp <= 0x9FFF; }

bool text_has_cjk(std::string_view s) {
    std::size_t i = 0;
    while (i < s.size())
        if (is_cjk(uni::decode_utf8(s, i))) return true;
    return false;
}

// Python str.isspace(): the White_Space property plus the four ASCII
// separators U+001C..U+001F (bidi class B / S), which Python counts and the
// property does not.
bool is_space(uint32_t cp) {
    if (cp >= 0x1C && cp <= 0x1F) return true;
    return uni::is_white_space(cp);
}

std::string strip(std::string_view s) {
    std::vector<uint32_t> cps = to_codepoints(s);
    std::size_t a = 0, b = cps.size();
    while (a < b && is_space(cps[a])) ++a;
    while (b > a && is_space(cps[b - 1])) --b;
    return from_codepoints(std::vector<uint32_t>(cps.begin() + static_cast<std::ptrdiff_t>(a),
                                                 cps.begin() + static_cast<std::ptrdiff_t>(b)));
}

namespace {

// Simple case mapping for the scripts the language table and the instruct
// vocabulary use: ASCII, Latin-1, Latin Extended-A, schwa, Greek and
// Cyrillic. Everything else maps to itself.
uint32_t lower_cp(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    if (c < 0x80) return c;
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 32;
    if (c >= 0x100 && c <= 0x137) return (c & 1) ? c : c + 1;
    if (c == 0x130) return 'i';
    if (c >= 0x139 && c <= 0x148) return (c & 1) ? c + 1 : c;
    if (c >= 0x14A && c <= 0x177) return (c & 1) ? c : c + 1;
    if (c == 0x178) return 0xFF;
    if (c >= 0x179 && c <= 0x17E) return (c & 1) ? c + 1 : c;
    if (c == 0x18F) return 0x259;
    if (c >= 0x391 && c <= 0x3A9 && c != 0x3A2) return c + 32;
    if (c >= 0x410 && c <= 0x42F) return c + 32;
    if (c >= 0x400 && c <= 0x40F) return c + 80;
    return c;
}

uint32_t upper_cp(uint32_t c) {
    if (c >= 'a' && c <= 'z') return c - 32;
    if (c < 0x80) return c;
    if (c >= 0xE0 && c <= 0xFE && c != 0xF7) return c - 32;
    if (c == 0xFF) return 0x178;
    if (c >= 0x101 && c <= 0x137) return (c & 1) ? c - 1 : c;
    if (c >= 0x13A && c <= 0x148) return (c & 1) ? c : c - 1;
    if (c >= 0x14B && c <= 0x177) return (c & 1) ? c - 1 : c;
    if (c >= 0x17A && c <= 0x17E) return (c & 1) ? c : c - 1;
    if (c == 0x259) return 0x18F;
    if (c >= 0x3B1 && c <= 0x3C9 && c != 0x3C2) return c - 32;
    if (c >= 0x430 && c <= 0x44F) return c - 32;
    if (c >= 0x450 && c <= 0x45F) return c - 80;
    return c;
}

// "Cased letter" for str.title(): a letter of a bicameral script. The
// language table is Latin-script, so letters below the CJK blocks are the
// only ones that matter.
bool is_cased(uint32_t c) { return uni::is_letter(c) && c < 0x2E80; }

// Python str.title(): uppercase a letter that does not follow a cased letter,
// lowercase one that does. Apostrophes and hyphens break words, exactly as in
// Python ("fe'fe'" -> "Fe'Fe'"), which the display-name exceptions correct.
std::string title(std::string_view s) {
    std::vector<uint32_t> cps = to_codepoints(s);
    bool prev_cased = false;
    for (uint32_t& c : cps) {
        const bool cased = is_cased(c);
        if (cased) c = prev_cased ? lower_cp(c) : upper_cp(c);
        prev_cased = cased;
    }
    return from_codepoints(cps);
}

}  // namespace

std::string lower(std::string_view s) {
    std::vector<uint32_t> cps = to_codepoints(s);
    for (uint32_t& c : cps) c = lower_cp(c);
    return from_codepoints(cps);
}

// ═══════════════════════════════════════════════════════════════════════════
// Duration rule (omnivoice/utils/duration.py RuleDurationEstimator)
// ═══════════════════════════════════════════════════════════════════════════

#include "omnivoice_unicode.inc"

namespace {

enum class Script {
    Cjk, Hangul, Kana, Ethiopic, Yi, Indic, ThaiLao, KhmerMyanmar, Arabic, Hebrew,
    Latin, Cyrillic, Greek, Armenian, Georgian, Punctuation, Space, Digit, Mark, Default
};

double script_weight(Script s) {
    switch (s) {
        case Script::Cjk:          return 3.0;
        case Script::Hangul:       return 2.5;
        case Script::Kana:         return 2.2;
        case Script::Ethiopic:     return 3.0;
        case Script::Yi:           return 3.0;
        case Script::Indic:        return 1.8;
        case Script::ThaiLao:      return 1.5;
        case Script::KhmerMyanmar: return 1.8;
        case Script::Arabic:       return 1.5;
        case Script::Hebrew:       return 1.5;
        case Script::Latin:        return 1.0;
        case Script::Cyrillic:     return 1.0;
        case Script::Greek:        return 1.0;
        case Script::Armenian:     return 1.0;
        case Script::Georgian:     return 1.0;
        case Script::Punctuation:  return 0.5;
        case Script::Space:        return 0.2;
        case Script::Digit:        return 3.5;
        case Script::Mark:         return 0.0;
        case Script::Default:      return 1.0;
    }
    return 1.0;
}

// duration.py's `ranges` — (end code point, script), searched with
// bisect_left on the end points.
struct BlockRange { uint32_t end; Script script; };
const BlockRange kBlocks[] = {
    {0x02AF, Script::Latin},        {0x03FF, Script::Greek},        {0x052F, Script::Cyrillic},
    {0x058F, Script::Armenian},     {0x05FF, Script::Hebrew},       {0x077F, Script::Arabic},
    {0x089F, Script::Arabic},       {0x08FF, Script::Arabic},       {0x097F, Script::Indic},
    {0x09FF, Script::Indic},        {0x0A7F, Script::Indic},        {0x0AFF, Script::Indic},
    {0x0B7F, Script::Indic},        {0x0BFF, Script::Indic},        {0x0C7F, Script::Indic},
    {0x0CFF, Script::Indic},        {0x0D7F, Script::Indic},        {0x0DFF, Script::Indic},
    {0x0EFF, Script::ThaiLao},      {0x0FFF, Script::Indic},        {0x109F, Script::KhmerMyanmar},
    {0x10FF, Script::Georgian},     {0x11FF, Script::Hangul},       {0x137F, Script::Ethiopic},
    {0x139F, Script::Ethiopic},     {0x13FF, Script::Default},      {0x167F, Script::Default},
    {0x169F, Script::Default},      {0x16FF, Script::Default},      {0x171F, Script::Default},
    {0x173F, Script::Default},      {0x175F, Script::Default},      {0x177F, Script::Default},
    {0x17FF, Script::KhmerMyanmar}, {0x18AF, Script::Default},      {0x18FF, Script::Default},
    {0x194F, Script::Indic},        {0x19DF, Script::Indic},        {0x19FF, Script::KhmerMyanmar},
    {0x1A1F, Script::Indic},        {0x1AAF, Script::Indic},        {0x1B7F, Script::Indic},
    {0x1BBF, Script::Indic},        {0x1BFF, Script::Indic},        {0x1C4F, Script::Indic},
    {0x1C7F, Script::Indic},        {0x1C8F, Script::Cyrillic},     {0x1CBF, Script::Georgian},
    {0x1CCF, Script::Indic},        {0x1CFF, Script::Indic},        {0x1D7F, Script::Latin},
    {0x1DBF, Script::Latin},        {0x1DFF, Script::Default},      {0x1EFF, Script::Latin},
    {0x309F, Script::Kana},         {0x30FF, Script::Kana},         {0x312F, Script::Cjk},
    {0x318F, Script::Hangul},       {0x9FFF, Script::Cjk},          {0xA4CF, Script::Yi},
    {0xA4FF, Script::Default},      {0xA63F, Script::Default},      {0xA69F, Script::Cyrillic},
    {0xA6FF, Script::Default},      {0xA7FF, Script::Latin},        {0xA82F, Script::Indic},
    {0xA87F, Script::Default},      {0xA8DF, Script::Indic},        {0xA8FF, Script::Indic},
    {0xA92F, Script::Indic},        {0xA95F, Script::Indic},        {0xA97F, Script::Hangul},
    {0xA9DF, Script::Indic},        {0xA9FF, Script::KhmerMyanmar}, {0xAA5F, Script::Indic},
    {0xAA7F, Script::KhmerMyanmar}, {0xAADF, Script::Indic},        {0xAAFF, Script::Indic},
    {0xAB2F, Script::Ethiopic},     {0xAB6F, Script::Latin},        {0xABBF, Script::Default},
    {0xABFF, Script::Indic},        {0xD7AF, Script::Hangul},       {0xFAFF, Script::Cjk},
    {0xFDFF, Script::Arabic},       {0xFE6F, Script::Default},      {0xFEFF, Script::Arabic},
    {0xFFEF, Script::Latin},
};

// unicodedata.category(c)[0] restricted to the classes the rule tests:
// 1 = M, 2 = P, 3 = S, 4 = Z, 5 = N, 0 = anything else.
int category_class(uint32_t cp) {
    const auto* first = kOmniVoiceUnicodeCategories;
    const auto* last  = first + sizeof(kOmniVoiceUnicodeCategories) / sizeof(kOmniVoiceUnicodeCategories[0]);
    // first range whose hi >= cp
    const auto* it = std::lower_bound(first, last, cp,
                                      [](const decltype(*first)& r, uint32_t v) { return r.hi < v; });
    if (it != last && it->lo <= cp && cp <= it->hi) return it->cls;
    return 0;
}

}  // namespace

double char_weight(uint32_t code) {
    if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122)) return script_weight(Script::Latin);
    if (code == 32) return script_weight(Script::Space);
    if (code == 0x0640) return script_weight(Script::Mark);   // Arabic tatweel
    switch (category_class(code)) {
        case 1: return script_weight(Script::Mark);
        case 2:
        case 3: return script_weight(Script::Punctuation);
        case 4: return script_weight(Script::Space);
        case 5: return script_weight(Script::Digit);
        default: break;
    }
    const std::size_t n = sizeof(kBlocks) / sizeof(kBlocks[0]);
    std::size_t idx = 0;   // bisect_left over the end points
    {
        std::size_t lo = 0, hi = n;
        while (lo < hi) {
            const std::size_t mid = (lo + hi) / 2;
            if (kBlocks[mid].end < code) lo = mid + 1; else hi = mid;
        }
        idx = lo;
    }
    if (idx < n) return script_weight(kBlocks[idx].script);
    if (code > 0x20000) return script_weight(Script::Cjk);
    return script_weight(Script::Default);
}

double total_weight(std::string_view text) {
    // Python's sum() over the per-character weights, left to right.
    double s = 0.0;
    std::size_t i = 0;
    while (i < text.size()) s += char_weight(uni::decode_utf8(text, i));
    return s;
}

double estimate_duration(std::string_view target_text, std::string_view ref_text,
                         double ref_duration, double low_threshold, double boost_strength) {
    if (ref_duration <= 0 || ref_text.empty()) return 0.0;
    const double ref_weight = total_weight(ref_text);
    if (ref_weight == 0) return 0.0;
    const double speed_factor = ref_weight / ref_duration;
    const double target_weight = total_weight(target_text);
    const double est = target_weight / speed_factor;
    if (est < low_threshold) {
        const double alpha = 1.0 / boost_strength;
        return low_threshold * std::pow(est / low_threshold, alpha);
    }
    return est;
}

int estimate_target_tokens(std::string_view text, const std::string* ref_text,
                           int n_ref, double speed) {
    std::string_view ref = "Nice to meet you.";
    int n = 25;
    if (ref_text && !ref_text->empty() && n_ref >= 0) {
        ref = *ref_text;
        n = n_ref;
    }
    double est = estimate_duration(text, ref, static_cast<double>(n));
    if (speed > 0 && speed != 1.0) est = est / speed;
    // max(1, int(est)) — int() truncates toward zero.
    const int t = static_cast<int>(est);
    return std::max(1, t);
}

// ═══════════════════════════════════════════════════════════════════════════
// Language map (omnivoice/utils/lang_map.py)
// ═══════════════════════════════════════════════════════════════════════════

#include "omnivoice_lang_map.inc"

namespace {

const std::size_t kLangCount = sizeof(kOmniVoiceLangMap) / sizeof(kOmniVoiceLangMap[0]);

struct TitleException { const char* name; const char* display; };
const TitleException kTitleExceptions[] = {
    {"fe'fe'", "Fe'fe'"},
    {"dũya", "Dũya"},
    {"santiago del estero quichua", "Santiago del Estero Quichua"},
    {"santa ana de tusi pasco quechua", "Santa Ana de Tusi Pasco Quechua"},
    {"malinaltepec me'phaa", "Malinaltepec Me'phaa"},
    {"tlacoapa me'phaa", "Tlacoapa Me'phaa"},
};

}  // namespace

std::string resolve_language(const std::string& language, bool* recognized) {
    if (recognized) *recognized = true;
    if (language.empty() || lower(language) == "none") return "";
    for (std::size_t i = 0; i < kLangCount; ++i)
        if (language == kOmniVoiceLangMap[i].id) return language;
    const std::string key = lower(language);
    for (std::size_t i = 0; i < kLangCount; ++i)
        if (key == kOmniVoiceLangMap[i].name) return kOmniVoiceLangMap[i].id;
    if (recognized) *recognized = false;
    return "";
}

std::vector<std::string> language_names() {
    std::vector<std::string> out;
    out.reserve(kLangCount);
    for (std::size_t i = 0; i < kLangCount; ++i) {
        const char* name = kOmniVoiceLangMap[i].name;
        bool done = false;
        for (const TitleException& e : kTitleExceptions)
            if (std::strcmp(e.name, name) == 0) { out.emplace_back(e.display); done = true; break; }
        if (!done) out.push_back(title(name));
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Voice-design instruct (omnivoice/utils/voice_design.py + _resolve_instruct)
// ═══════════════════════════════════════════════════════════════════════════

namespace {

struct EnZh { const char* en; const char* zh; };
const EnZh kGender[] = {{"male", "男"}, {"female", "女"}};
const EnZh kAge[] = {{"child", "儿童"}, {"teenager", "少年"}, {"young adult", "青年"},
                     {"middle-aged", "中年"}, {"elderly", "老年"}};
const EnZh kPitch[] = {{"very low pitch", "极低音调"}, {"low pitch", "低音调"},
                       {"moderate pitch", "中音调"}, {"high pitch", "高音调"},
                       {"very high pitch", "极高音调"}};
const EnZh kStyle[] = {{"whisper", "耳语"}};
const char* kAccent[] = {"american accent", "british accent", "australian accent",
                         "chinese accent", "canadian accent", "indian accent",
                         "korean accent", "portuguese accent", "russian accent",
                         "japanese accent"};
const char* kDialect[] = {"河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话",
                          "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话"};

struct PairCat { const EnZh* items; std::size_t n; };
const PairCat kPairCats[] = {
    {kGender, sizeof(kGender) / sizeof(kGender[0])},
    {kAge, sizeof(kAge) / sizeof(kAge[0])},
    {kPitch, sizeof(kPitch) / sizeof(kPitch[0])},
    {kStyle, sizeof(kStyle) / sizeof(kStyle[0])},
};
const std::size_t kAccentN = sizeof(kAccent) / sizeof(kAccent[0]);
const std::size_t kDialectN = sizeof(kDialect) / sizeof(kDialect[0]);

bool in_list(const char* const* list, std::size_t n, const std::string& s) {
    for (std::size_t i = 0; i < n; ++i)
        if (s == list[i]) return true;
    return false;
}

bool is_valid_item(const std::string& s) {
    for (const PairCat& c : kPairCats)
        for (std::size_t i = 0; i < c.n; ++i)
            if (s == c.items[i].en || s == c.items[i].zh) return true;
    return in_list(kAccent, kAccentN, s) || in_list(kDialect, kDialectN, s);
}

std::string en_to_zh(const std::string& s) {
    for (const PairCat& c : kPairCats)
        for (std::size_t i = 0; i < c.n; ++i)
            if (s == c.items[i].en) return c.items[i].zh;
    return s;
}

std::string zh_to_en(const std::string& s) {
    for (const PairCat& c : kPairCats)
        for (std::size_t i = 0; i < c.n; ++i)
            if (s == c.items[i].zh) return c.items[i].en;
    return s;
}

bool ends_with(const std::string& s, const char* suffix) {
    const std::size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

std::string join_valid(bool zh) {
    // The valid-item listing of the upstream error (sorted, as Python's
    // sorted() orders the strings: by code point).
    std::vector<std::string> items;
    for (const PairCat& c : kPairCats)
        for (std::size_t i = 0; i < c.n; ++i) items.emplace_back(zh ? c.items[i].zh : c.items[i].en);
    if (zh) for (std::size_t i = 0; i < kDialectN; ++i) items.emplace_back(kDialect[i]);
    else    for (std::size_t i = 0; i < kAccentN; ++i)  items.emplace_back(kAccent[i]);
    std::sort(items.begin(), items.end());
    std::string out;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) out += zh ? "，" : ", ";
        out += items[i];
    }
    return out;
}

}  // namespace

std::string resolve_instruct(const std::string& instruct, bool use_zh) {
    const std::string s = strip(instruct);
    if (s.empty()) return "";

    // re.split(r"\s*[,，]\s*") + drop empties: split at either comma, strip
    // each piece.
    std::vector<std::string> raw_items;
    {
        std::vector<uint32_t> cps = to_codepoints(s);
        std::vector<uint32_t> cur;
        auto flush = [&]() {
            const std::string piece = strip(from_codepoints(cur));
            if (!piece.empty()) raw_items.push_back(piece);
            cur.clear();
        };
        for (uint32_t c : cps) {
            if (c == ',' || c == 0xFF0C) flush(); else cur.push_back(c);
        }
        flush();
    }

    std::vector<std::string> normalised;
    std::string unknown;
    for (const std::string& raw : raw_items) {
        const std::string n = lower(strip(raw));
        if (is_valid_item(n)) normalised.push_back(n);
        else unknown += "\n  '" + raw + "' -> '" + n + "' (unsupported)";
    }
    if (!unknown.empty()) {
        throw std::runtime_error(
            "brosoundml: OmniVoice: unsupported instruct items found in '" + s + "':" + unknown +
            "\n\nValid English items: " + join_valid(false) +
            "\nValid Chinese items: " + join_valid(true) +
            "\n\nTip: use only English or only Chinese instructs; English items are separated by "
            "comma + space (e.g. 'male, indian accent'), Chinese items by a full-width comma "
            "(e.g. '男，河南话').");
    }

    bool has_dialect = false, has_accent = false;
    for (const std::string& n : normalised) {
        if (ends_with(n, "话")) has_dialect = true;
        if (n.find(" accent") != std::string::npos) has_accent = true;
    }
    if (has_dialect && has_accent)
        throw std::runtime_error(
            "brosoundml: OmniVoice: cannot mix a Chinese dialect and an English accent in one "
            "instruct; dialects are for Chinese speech, accents for English speech");
    if (has_dialect) use_zh = true;
    else if (has_accent) use_zh = false;

    for (std::string& n : normalised) n = use_zh ? en_to_zh(n) : zh_to_en(n);

    // Category conflict: more than one item of a category.
    std::string conflicts;
    auto check = [&](auto member) {
        std::vector<std::string> hits;
        for (const std::string& n : normalised)
            if (member(n)) hits.push_back(n);
        if (hits.size() > 1) {
            if (!conflicts.empty()) conflicts += "; ";
            for (std::size_t i = 0; i < hits.size(); ++i) {
                if (i) conflicts += " vs ";
                conflicts += "'" + hits[i] + "'";
            }
        }
    };
    for (const PairCat& c : kPairCats)
        check([&](const std::string& n) {
            for (std::size_t i = 0; i < c.n; ++i)
                if (n == c.items[i].en || n == c.items[i].zh) return true;
            return false;
        });
    check([&](const std::string& n) { return in_list(kAccent, kAccentN, n); });
    check([&](const std::string& n) { return in_list(kDialect, kDialectN, n); });
    if (!conflicts.empty())
        throw std::runtime_error(
            "brosoundml: OmniVoice: conflicting instruct items within the same category: " +
            conflicts + "; each category (gender, age, pitch, style, accent, dialect) allows "
            "at most one item");

    bool has_zh = false;
    for (const std::string& n : normalised)
        if (text_has_cjk(n)) has_zh = true;
    const char* sep = has_zh ? "，" : ", ";
    std::string out;
    for (std::size_t i = 0; i < normalised.size(); ++i) {
        if (i) out += sep;
        out += normalised[i];
    }
    return out;
}

std::vector<OmniVoice::InstructCategory> instruct_categories() {
    std::vector<OmniVoice::InstructCategory> out;
    const char* names[] = {"gender", "age", "pitch", "style"};
    for (std::size_t c = 0; c < 4; ++c) {
        OmniVoice::InstructCategory cat;
        cat.name = names[c];
        for (std::size_t i = 0; i < kPairCats[c].n; ++i) cat.values.emplace_back(kPairCats[c].items[i].en);
        out.push_back(std::move(cat));
    }
    {
        OmniVoice::InstructCategory cat;
        cat.name = "accent";
        for (std::size_t i = 0; i < kAccentN; ++i) cat.values.emplace_back(kAccent[i]);
        out.push_back(std::move(cat));
    }
    {
        OmniVoice::InstructCategory cat;
        cat.name = "dialect";
        for (std::size_t i = 0; i < kDialectN; ++i) cat.values.emplace_back(kDialect[i]);
        out.push_back(std::move(cat));
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Text (_combine_text, add_punctuation, style string, tags, chunking)
// ═══════════════════════════════════════════════════════════════════════════

std::string combine_text(const std::string& text, const std::string* ref_text) {
    std::string full;
    if (ref_text && !ref_text->empty()) full = strip(*ref_text) + " " + strip(text);
    else full = strip(text);

    std::vector<uint32_t> in = to_codepoints(full);
    std::vector<uint32_t> cps;
    cps.reserve(in.size());
    // re.sub(r"[\r\n]+", "") and the parenthesis replacement.
    for (uint32_t c : in) {
        if (c == '\r' || c == '\n') continue;
        if (c == 0xFF08) c = '(';
        else if (c == 0xFF09) c = ')';
        cps.push_back(c);
    }
    // re.sub(r"[ \t]+", " ")
    {
        std::vector<uint32_t> out;
        out.reserve(cps.size());
        for (std::size_t i = 0; i < cps.size(); ++i) {
            if (cps[i] == ' ' || cps[i] == '\t') {
                out.push_back(' ');   // the whole run becomes one space
                while (i + 1 < cps.size() && (cps[i + 1] == ' ' || cps[i + 1] == '\t')) ++i;
            } else {
                out.push_back(cps[i]);
            }
        }
        cps.swap(out);
    }
    // re.sub(r"(?<=[一-鿿])\s+|\s+(?=[一-鿿])", ""): a whitespace run is
    // dropped when a CJK character precedes or follows it.
    {
        std::vector<uint32_t> out;
        out.reserve(cps.size());
        std::size_t i = 0;
        while (i < cps.size()) {
            if (!is_space(cps[i])) { out.push_back(cps[i]); ++i; continue; }
            std::size_t j = i;
            while (j < cps.size() && is_space(cps[j])) ++j;
            const bool before = i > 0 && is_cjk(cps[i - 1]);
            const bool after  = j < cps.size() && is_cjk(cps[j]);
            if (!(before || after))
                for (std::size_t k = i; k < j; ++k) out.push_back(cps[k]);
            i = j;
        }
        cps.swap(out);
    }
    return from_codepoints(cps);
}

std::string add_punctuation(const std::string& text) {
    std::string t = strip(text);
    if (t.empty()) return t;
    std::vector<uint32_t> cps = to_codepoints(t);
    static const uint32_t kEnd[] = {
        ';', ':', ',', '.', '!', '?', 0x2026, ')', ']', '}', '"', '\'', 0x201C, 0x201D, 0x2018,
        0x2019, 0xFF1B, 0xFF1A, 0xFF0C, 0x3002, 0xFF01, 0xFF1F, 0x3001, 0xFF09, 0x3011,
    };
    const uint32_t last = cps.back();
    for (uint32_t e : kEnd)
        if (last == e) return t;
    bool chinese = false;
    for (uint32_t c : cps)
        if (is_cjk(c)) { chinese = true; break; }
    t += chinese ? "。" : ".";
    return t;
}

std::string style_text(const std::string& lang, const std::string& instruct,
                       bool denoise, bool has_ref) {
    std::string s;
    if (denoise && has_ref) s += "<|denoise|>";
    s += "<|lang_start|>" + (lang.empty() ? std::string("None") : lang) + "<|lang_end|>";
    s += "<|instruct_start|>" + (instruct.empty() ? std::string("None") : instruct) + "<|instruct_end|>";
    return s;
}

std::vector<std::string> nonverbal_tags() {
    return {"[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", "[question-ah]",
            "[question-oh]", "[question-ei]", "[question-yi]", "[surprise-ah]",
            "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", "[dissatisfaction-hnn]"};
}

std::vector<int32_t> tokenize_with_tags(const std::string& text,
                                        const brolm::qwen::Tokenizer& tok) {
    static const std::vector<std::string> tags = nonverbal_tags();
    std::vector<int32_t> out;
    std::size_t last_end = 0, i = 0;
    auto emit = [&](std::string_view seg) {
        std::vector<int32_t> ids = tok.encode(seg);
        out.insert(out.end(), ids.begin(), ids.end());
    };
    while (i < text.size()) {
        if (text[i] == '[') {
            std::size_t hit = 0;
            for (const std::string& t : tags)
                if (text.compare(i, t.size(), t) == 0) { hit = t.size(); break; }
            if (hit) {
                if (i > last_end) emit(std::string_view(text).substr(last_end, i - last_end));
                emit(std::string_view(text).substr(i, hit));
                i += hit;
                last_end = i;
                continue;
            }
        }
        ++i;
    }
    if (last_end < text.size()) emit(std::string_view(text).substr(last_end));
    return out;
}

namespace {

bool in_set(uint32_t c, const char32_t* set) {
    for (; *set; ++set)
        if (*set == static_cast<char32_t>(c)) return true;
    return false;
}

const char32_t kSplitPunct[] = U".,;:!?。，；：！？";
const char32_t kClosing[] = U"\"'“”‘’）]》>」】";
const char* kAbbreviations[] = {
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.", "Pres.", "Gov.",
    "Capt.", "Gen.", "Sen.", "Rep.", "Col.", "Maj.", "Lt.", "Cmdr.", "Sgt.", "Cpl.", "Co.",
    "Corp.", "Inc.", "Ltd.", "Est.", "Dept.", "St.", "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.",
    "No.", "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",
    "i.e.", "e.g.", "vs.", "Vs.", "Etc.", "approx.", "fig.", "def.",
};

// Python's str.split()[-1]: the last maximal run of non-whitespace.
std::string last_word(const std::vector<uint32_t>& cps) {
    std::size_t b = cps.size();
    while (b > 0 && is_space(cps[b - 1])) --b;
    std::size_t a = b;
    while (a > 0 && !is_space(cps[a - 1])) --a;
    return from_codepoints(std::vector<uint32_t>(cps.begin() + static_cast<std::ptrdiff_t>(a),
                                                 cps.begin() + static_cast<std::ptrdiff_t>(b)));
}

}  // namespace

std::vector<std::string> chunk_text(const std::string& text, int chunk_len, int min_chunk_len) {
    using Sentence = std::vector<uint32_t>;
    std::vector<Sentence> sentences;
    Sentence current;
    for (uint32_t c : to_codepoints(text)) {
        if (current.empty() && !sentences.empty() && (in_set(c, kSplitPunct) || in_set(c, kClosing))) {
            sentences.back().push_back(c);
            continue;
        }
        current.push_back(c);
        if (in_set(c, kSplitPunct)) {
            bool abbreviation = false;
            if (c == '.') {
                const std::string temp = strip(from_codepoints(current));
                if (!temp.empty()) {
                    const std::string w = last_word(to_codepoints(temp));
                    for (const char* a : kAbbreviations)
                        if (w == a) { abbreviation = true; break; }
                }
            }
            if (!abbreviation) {
                sentences.push_back(current);
                current.clear();
            }
        }
    }
    if (!current.empty()) sentences.push_back(current);

    std::vector<Sentence> merged;
    Sentence chunk;
    for (const Sentence& s : sentences) {
        if (static_cast<int>(chunk.size() + s.size()) <= chunk_len) {
            chunk.insert(chunk.end(), s.begin(), s.end());
        } else {
            if (!chunk.empty()) merged.push_back(chunk);
            chunk = s;
        }
    }
    if (!chunk.empty()) merged.push_back(chunk);

    std::vector<Sentence> final_chunks;
    if (min_chunk_len > 0) {
        const bool first_short = !merged.empty() && static_cast<int>(merged[0].size()) < min_chunk_len;
        for (std::size_t i = 0; i < merged.size(); ++i) {
            const Sentence& ch = merged[i];
            if (i == 1 && first_short) {
                final_chunks.back().insert(final_chunks.back().end(), ch.begin(), ch.end());
            } else if (static_cast<int>(ch.size()) >= min_chunk_len) {
                final_chunks.push_back(ch);
            } else if (final_chunks.empty()) {
                final_chunks.push_back(ch);
            } else {
                final_chunks.back().insert(final_chunks.back().end(), ch.begin(), ch.end());
            }
        }
    } else {
        final_chunks = merged;
    }

    std::vector<std::string> out;
    for (const Sentence& ch : final_chunks) {
        std::string s = strip(from_codepoints(ch));
        if (!s.empty()) out.push_back(std::move(s));
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Audio (pydub AudioSegment semantics, mono int16)
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// A mono 16-bit pydub AudioSegment: `s` holds the samples, `sr` the frame
// rate. Lengths are in milliseconds exactly as pydub computes them.
struct Seg {
    std::vector<int16_t> s;
    int sr = 24000;

    // len(seg): round(1000 * (frame_count / frame_rate)) — Python's
    // round-half-to-even on the double.
    long long len_ms() const {
        const double v = 1000.0 * (static_cast<double>(s.size()) / static_cast<double>(sr));
        return static_cast<long long>(std::nearbyint(v));
    }
    // frame_count(ms): ms * (frame_rate / 1000.0), truncated by int().
    long long ms_to_samples(long long ms) const {
        return static_cast<long long>(static_cast<double>(ms) * (static_cast<double>(sr) / 1000.0));
    }
    // audioop.rms: int(sqrt(sum(x^2) / n)); 0 when empty.
    long long rms() const {
        if (s.empty()) return 0;
        double sum = 0.0;
        for (int16_t v : s) sum += static_cast<double>(v) * static_cast<double>(v);
        return static_cast<long long>(std::sqrt(sum / static_cast<double>(s.size())));
    }
    // seg[a:b] in ms (both clamped to len), with pydub's zero fill of up to
    // 2 ms of missing frames — which only happens when the slice already
    // holds at least one frame (the fill is a copy of a zeroed first frame).
    Seg slice(long long a, long long b) const {
        const long long L = len_ms();
        a = std::min(a, L);
        b = std::min(b, L);
        if (a < 0) a = L - (-a);
        if (b < 0) b = L - (-b);
        const long long sa = ms_to_samples(a), sb = ms_to_samples(b);
        const long long n = static_cast<long long>(s.size());
        const long long ca = std::max(0LL, std::min(sa, n));
        const long long cb = std::max(ca, std::min(sb, n));
        Seg out;
        out.sr = sr;
        out.s.assign(s.begin() + ca, s.begin() + cb);
        const long long expected = sb - sa;
        const long long missing = expected - static_cast<long long>(out.s.size());
        if (missing > 0 && !out.s.empty()) {
            if (missing > ms_to_samples(2)) throw std::runtime_error("brosoundml: OmniVoice: pydub slice: too many missing frames");
            out.s.insert(out.s.end(), static_cast<std::size_t>(missing), int16_t{0});
        }
        return out;
    }
    Seg reversed() const {
        Seg out;
        out.sr = sr;
        out.s.assign(s.rbegin(), s.rend());
        return out;
    }
};

Seg to_seg(const std::vector<float>& x, int sr) {
    Seg seg;
    seg.sr = sr;
    seg.s.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        float v = x[i] * 32768.0f;
        if (v < -32768.0f) v = -32768.0f;
        if (v > 32767.0f) v = 32767.0f;
        seg.s[i] = static_cast<int16_t>(v);   // astype(int16): truncation toward zero
    }
    return seg;
}

std::vector<float> from_seg(const Seg& seg) {
    std::vector<float> out(seg.s.size());
    for (std::size_t i = 0; i < seg.s.size(); ++i) out[i] = static_cast<float>(seg.s[i]) / 32768.0f;
    return out;
}

// db_to_float(db) * max_possible_amplitude
double silence_rms_threshold(double db) { return std::pow(10.0, db / 20.0) * 32768.0; }

// seg.dBFS < threshold  (dBFS = -inf when rms == 0)
bool quieter_than(const Seg& seg, double db) {
    const long long r = seg.rms();
    if (r == 0) return true;
    const double dbfs = 20.0 * std::log10(static_cast<double>(r) / 32768.0);
    return dbfs < db;
}

std::vector<std::pair<long long, long long>> detect_silence(const Seg& seg, long long min_len,
                                                            double thresh_db, long long step) {
    std::vector<std::pair<long long, long long>> out;
    const long long L = seg.len_ms();
    if (L < min_len) return out;
    const double thr = silence_rms_threshold(thresh_db);
    const long long last = L - min_len;
    std::vector<long long> starts;
    for (long long i = 0; i <= last; i += step)
        if (static_cast<double>(seg.slice(i, i + min_len).rms()) <= thr) starts.push_back(i);
    if (last % step)
        if (static_cast<double>(seg.slice(last, last + min_len).rms()) <= thr) starts.push_back(last);
    if (starts.empty()) return out;
    long long prev = starts[0];
    long long cur_start = prev;
    for (std::size_t k = 1; k < starts.size(); ++k) {
        const long long i = starts[k];
        const bool continuous = (i == prev + step);
        const bool has_gap = i > (prev + min_len);
        if (!continuous && has_gap) {
            out.emplace_back(cur_start, prev + min_len);
            cur_start = i;
        }
        prev = i;
    }
    out.emplace_back(cur_start, prev + min_len);
    return out;
}

std::vector<std::pair<long long, long long>> detect_nonsilent(const Seg& seg, long long min_len,
                                                              double thresh_db, long long step) {
    const auto silent = detect_silence(seg, min_len, thresh_db, step);
    const long long L = seg.len_ms();
    std::vector<std::pair<long long, long long>> out;
    if (silent.empty()) { out.emplace_back(0, L); return out; }
    if (silent[0].first == 0 && silent[0].second == L) return out;
    long long prev_end = 0;
    long long end = 0;
    for (const auto& r : silent) {
        out.emplace_back(prev_end, r.first);
        prev_end = r.second;
        end = r.second;
    }
    if (end != L) out.emplace_back(prev_end, L);
    if (out[0].first == 0 && out[0].second == 0) out.erase(out.begin());
    return out;
}

long long floor_div(long long a, long long b) {
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

std::vector<Seg> split_on_silence(const Seg& seg, long long min_len, double thresh_db,
                                  long long keep, long long step) {
    auto ranges = detect_nonsilent(seg, min_len, thresh_db, step);
    for (auto& r : ranges) { r.first -= keep; r.second += keep; }
    for (std::size_t i = 0; i + 1 < ranges.size(); ++i) {
        const long long last_end = ranges[i].second;
        const long long next_start = ranges[i + 1].first;
        if (next_start < last_end) {
            ranges[i].second = floor_div(last_end + next_start, 2);
            ranges[i + 1].first = ranges[i].second;
        }
    }
    const long long L = seg.len_ms();
    std::vector<Seg> out;
    for (const auto& r : ranges) out.push_back(seg.slice(std::max(r.first, 0LL), std::min(r.second, L)));
    return out;
}

long long detect_leading_silence(const Seg& seg, double thresh_db, long long chunk) {
    long long trim = 0;
    const long long L = seg.len_ms();
    while (quieter_than(seg.slice(trim, trim + chunk), thresh_db) && trim < L) trim += chunk;
    return std::min(trim, L);
}

Seg remove_silence_edges(const Seg& in, long long lead, long long trail, double thresh_db) {
    long long start = detect_leading_silence(in, thresh_db, 10);
    start = std::max(0LL, start - lead);
    Seg a = in.slice(start, in.len_ms());
    Seg r = a.reversed();
    start = detect_leading_silence(r, thresh_db, 10);
    start = std::max(0LL, start - trail);
    Seg b = r.slice(start, r.len_ms());
    return b.reversed();
}

// numpy.linspace(start, stop, n, dtype=float32): float64 i*step + start with
// the endpoint forced, then a cast.
std::vector<float> linspace_f32(double start, double stop, int n) {
    std::vector<float> out(static_cast<std::size_t>(std::max(n, 0)));
    if (n <= 0) return out;
    const int div = n - 1;
    const double delta = stop - start;
    for (int i = 0; i < n; ++i) {
        double v;
        if (div > 0) {
            const double step = delta / div;
            const double prod = static_cast<double>(i) * step;
            v = prod + start;
        } else {
            v = start;   // num == 1: y = [0 * delta] + start
        }
        out[static_cast<std::size_t>(i)] = static_cast<float>(v);
    }
    if (n > 1) out[static_cast<std::size_t>(n - 1)] = static_cast<float>(stop);
    return out;
}

}  // namespace

std::vector<float> remove_silence(const std::vector<float>& x, int sr, int mid_sil,
                                  int lead_sil, int trail_sil) {
    Seg wave = to_seg(x, sr);
    if (mid_sil > 0) {
        std::vector<Seg> parts = split_on_silence(wave, mid_sil, -50.0, mid_sil, 10);
        Seg joined;
        joined.sr = sr;
        for (const Seg& p : parts) joined.s.insert(joined.s.end(), p.s.begin(), p.s.end());
        wave = std::move(joined);
    }
    wave = remove_silence_edges(wave, lead_sil, trail_sil, -50.0);
    return from_seg(wave);
}

std::vector<float> trim_long_audio(const std::vector<float>& x, int sr, double max_duration,
                                   double min_duration, double trim_threshold) {
    const double duration = static_cast<double>(x.size()) / sr;
    if (duration <= trim_threshold) return x;
    Seg seg = to_seg(x, sr);
    const auto nonsilent = detect_nonsilent(seg, 100, -40.0, 10);
    if (nonsilent.empty()) return x;
    const long long max_ms = static_cast<long long>(max_duration * 1000);
    const long long min_ms = static_cast<long long>(min_duration * 1000);
    long long best = 0;
    for (const auto& r : nonsilent) {
        if (r.first > best && r.first <= max_ms) best = r.first;
        if (r.second > max_ms) break;
    }
    if (best < min_ms) best = std::min(max_ms, seg.len_ms());
    return from_seg(seg.slice(0, best));
}

std::vector<float> fade_and_pad(const std::vector<float>& x, double pad_duration,
                                double fade_duration, int sr) {
    if (x.empty()) return x;
    const int fade_samples = static_cast<int>(fade_duration * sr);
    const int pad_samples = static_cast<int>(pad_duration * sr);
    std::vector<float> p = x;
    if (fade_samples > 0) {
        const int k = std::min<int>(fade_samples, static_cast<int>(p.size() / 2));
        if (k > 0) {
            const std::vector<float> fin = linspace_f32(0.0, 1.0, k);
            for (int i = 0; i < k; ++i) p[static_cast<std::size_t>(i)] *= fin[static_cast<std::size_t>(i)];
            const std::vector<float> fout = linspace_f32(1.0, 0.0, k);
            const std::size_t off = p.size() - static_cast<std::size_t>(k);
            for (int i = 0; i < k; ++i) p[off + static_cast<std::size_t>(i)] *= fout[static_cast<std::size_t>(i)];
        }
    }
    if (pad_samples > 0) {
        std::vector<float> out(static_cast<std::size_t>(pad_samples), 0.0f);
        out.insert(out.end(), p.begin(), p.end());
        out.insert(out.end(), static_cast<std::size_t>(pad_samples), 0.0f);
        return out;
    }
    return p;
}

std::vector<float> cross_fade_chunks(const std::vector<std::vector<float>>& chunks, int sr,
                                     double silence_duration) {
    if (chunks.empty()) return {};
    if (chunks.size() == 1) return chunks[0];
    const int total_n = static_cast<int>(silence_duration * sr);
    const int fade_n = total_n / 3;
    const int silence_n = fade_n;
    std::vector<float> merged = chunks[0];
    for (std::size_t c = 1; c < chunks.size(); ++c) {
        const int fout_n = std::min<int>(fade_n, static_cast<int>(merged.size()));
        if (fout_n > 0) {
            const std::vector<float> w = linspace_f32(1.0, 0.0, fout_n);
            const std::size_t off = merged.size() - static_cast<std::size_t>(fout_n);
            for (int i = 0; i < fout_n; ++i) merged[off + static_cast<std::size_t>(i)] *= w[static_cast<std::size_t>(i)];
        }
        merged.insert(merged.end(), static_cast<std::size_t>(std::max(silence_n, 0)), 0.0f);
        std::vector<float> fade_in = chunks[c];
        const int fin_n = std::min<int>(fade_n, static_cast<int>(fade_in.size()));
        if (fin_n > 0) {
            const std::vector<float> w = linspace_f32(0.0, 1.0, fin_n);
            for (int i = 0; i < fin_n; ++i) fade_in[static_cast<std::size_t>(i)] *= w[static_cast<std::size_t>(i)];
        }
        merged.insert(merged.end(), fade_in.begin(), fade_in.end());
    }
    return merged;
}

void gain_rms_match(std::vector<float>& x, float ref_rms) {
    for (float& v : x) {
        const float a = v * ref_rms;
        v = a / 0.1f;
    }
}

void peak_normalize(std::vector<float>& x) {
    float peak = 0.0f;
    for (float v : x) peak = std::max(peak, std::fabs(v));
    if (!(peak > 1e-6f)) return;
    for (float& v : x) {
        const float a = v / peak;
        v = a * 0.5f;
    }
}

float rms(const std::vector<float>& x) {
    if (x.empty()) return 0.0f;
    double s = 0.0;
    for (float v : x) s += static_cast<double>(v) * static_cast<double>(v);
    return static_cast<float>(std::sqrt(s / static_cast<double>(x.size())));
}

// ═══════════════════════════════════════════════════════════════════════════
// Schedule (_get_time_steps + the per-step counts of _generate_iterative)
// ═══════════════════════════════════════════════════════════════════════════

std::vector<int> unmask_schedule(int total_masked, int num_steps, float t_shift) {
    std::vector<int> out;
    if (num_steps <= 0) return out;
    const int steps = num_steps + 1;
    // torch.linspace(0, 1, steps) in FP32: the symmetric two-sided formula of
    // the CPU kernel.
    std::vector<float> t(static_cast<std::size_t>(steps));
    const float start = 0.0f, end = 1.0f;
    const float step = (end - start) / static_cast<float>(steps - 1);
    const int halfway = steps / 2;
    for (int i = 0; i < steps; ++i) {
        float v;
        if (i < halfway) v = start + step * static_cast<float>(i);
        else             v = end - step * static_cast<float>(steps - i - 1);
        t[static_cast<std::size_t>(i)] = v;
    }
    // t' = s * t / (1 + (s - 1) * t), every op FP32 with FP32 scalars.
    const float sm1 = t_shift - 1.0f;
    for (float& v : t) {
        const float num = t_shift * v;
        const float den = 1.0f + sm1 * v;
        v = num / den;
    }
    int rem = total_masked;
    for (int s = 0; s < num_steps; ++s) {
        int num;
        if (s == num_steps - 1) {
            num = rem;
        } else {
            const double diff = static_cast<double>(t[static_cast<std::size_t>(s + 1)]) -
                                static_cast<double>(t[static_cast<std::size_t>(s)]);
            const double c = std::ceil(static_cast<double>(total_masked) * diff);
            num = std::min(static_cast<int>(c), rem);
        }
        out.push_back(num);
        rem -= num;
    }
    return out;
}

}  // namespace ovp
}  // namespace brosoundml
