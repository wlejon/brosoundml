// SpecialCases tests: function-word allomorphy, symbol words, dotted
// acronyms, `vs.` rewrite, `used` heteronym, letter-by-letter spelling
// fallback. Coverage matches docs/special_cases.md § "Tests".
//
// Gated on the presence of the packed lexicon bin. Resolution order:
//   1. BROSOUNDML_LEXICON_PATH env var, if set.
//   2. <repo>/../brosoundml-data/g2p/lexicon_en_us.bin (sibling repo).
// If neither resolves, the gated section is skipped (printed note, not FAIL).
//
// Lexicon-derived expected strings are computed in-test by calling
// `lex.lookup(...)` directly; only the hand-typed IPA literals from the
// rule table appear as string-literal assertions.

#include "brosoundml/g2p/lexicon.h"
#include "brosoundml/g2p/special_cases.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

namespace g = brosoundml::g2p;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

// ─── Resolve lexicon path (mirrors test_morphology.cpp verbatim) ─────────

static std::string env_or_empty(const char* name) {
    if (const char* v = std::getenv(name)) return std::string(v);
    return {};
}

static std::string resolve_lexicon_path() {
    const auto env = env_or_empty("BROSOUNDML_LEXICON_PATH");
    if (!env.empty() && std::filesystem::exists(env)) return env;

#ifdef BROSOUNDML_REPO_DIR
    const std::string sibling =
        std::string(BROSOUNDML_REPO_DIR) + "/../brosoundml-data/g2p/lexicon_en_us.bin";
    if (std::filesystem::exists(sibling)) return sibling;
#endif
    return {};
}

// ─── Hand-typed IPA literals from the rule table (must match impl) ───────

static constexpr std::string_view kIpa_aDT     = "\xC9\x90";              // "ɐ"
static constexpr std::string_view kIpa_aPRP    = "\xCB\x88" "A";          // "ˈA"
static constexpr std::string_view kIpa_an      = "\xC9\x90" "n";          // "ɐn"
static constexpr std::string_view kIpa_to_cons = "t\xC9\x99";             // "tə"
static constexpr std::string_view kIpa_to_vow  = "t\xCA\x8A";             // "tʊ"
static constexpr std::string_view kIpa_the_v   = "\xC3\xB0i";             // "ði"
static constexpr std::string_view kIpa_the_c   = "\xC3\xB0\xC9\x99";      // "ðə"
static constexpr std::string_view kSecondary   = "\xCB\x8C";              // "ˌ"
static constexpr std::string_view kPrimary     = "\xCB\x88";              // "ˈ"

// ─── Tests ────────────────────────────────────────────────────────────────

static void test_function_words(const g::Lexicon& lx, const g::SpecialCases& sc) {
    const g::TokenContext none;            // future_vowel=0, future_to=false
    g::TokenContext vow;  vow.future_vowel = 1;
    g::TokenContext con;  con.future_vowel = -1;

    // a / A
    CHECK(sc.try_phonemize("a", "DT",  none) == kIpa_aDT,  "'a' DT  → ɐ");
    CHECK(sc.try_phonemize("A", "DT",  none) == kIpa_aDT,  "'A' DT  → ɐ");
    CHECK(sc.try_phonemize("a", "PRP", none) == kIpa_aPRP, "'a' PRP → ˈA");
    CHECK(sc.try_phonemize("A", "PRP", none) == kIpa_aPRP, "'A' PRP → ˈA");

    // the (vowel/cons/none branches)
    CHECK(sc.try_phonemize("the", "DT", vow)  == kIpa_the_v, "'the' +V → ði");
    CHECK(sc.try_phonemize("the", "DT", con)  == kIpa_the_c, "'the' +C → ðə");
    CHECK(sc.try_phonemize("the", "DT", none) == kIpa_the_c, "'the' +0 → ðə");
    CHECK(sc.try_phonemize("THE", "DT", vow)  == kIpa_the_v, "'THE' DT +V → ði");

    // to (all three future_vowel states)
    const auto to_default = std::string(lx.lookup("to"));
    CHECK(!to_default.empty(), "lex has 'to'");
    CHECK(sc.try_phonemize("to", "TO", none) == to_default, "'to' +0 → lex('to')");
    CHECK(sc.try_phonemize("to", "TO", vow)  == kIpa_to_vow,  "'to' +V → tʊ");
    CHECK(sc.try_phonemize("to", "TO", con)  == kIpa_to_cons, "'to' +C → tə");
    CHECK(sc.try_phonemize("TO", "TO", vow)  == kIpa_to_vow,  "'TO' TO +V → tʊ");
    CHECK(sc.try_phonemize("TO", "IN", con)  == kIpa_to_cons, "'TO' IN +C → tə");

    // an (always ɐn except AN+NN* → spell)
    CHECK(sc.try_phonemize("an", "DT",  none) == kIpa_an, "'an' DT  → ɐn");
    CHECK(sc.try_phonemize("An", "DT",  none) == kIpa_an, "'An' DT  → ɐn");
    CHECK(sc.try_phonemize("AN", "DT",  none) == kIpa_an, "'AN' DT  → ɐn");
    {
        const auto spelled = sc.spell_letter_by_letter("AN");
        CHECK(!spelled.empty(), "spell('AN') non-empty");
        CHECK(sc.try_phonemize("AN", "NNP", none) == spelled,
              "'AN' NNP → spell_letter_by_letter('AN')");
    }

    // I PRP
    {
        const std::string expect = std::string(kSecondary) + "I";
        CHECK(sc.try_phonemize("I", "PRP", none) == expect, "'I' PRP → ˌI");
    }

    // by RB → "bˈI"; by IN (preposition) → empty (no rule)
    {
        const std::string expect = std::string("b") + std::string(kPrimary) + "I";
        CHECK(sc.try_phonemize("by", "RB", none) == expect, "'by' RB → bˈI");
        CHECK(sc.try_phonemize("By", "RB", none) == expect, "'By' RB → bˈI");
        CHECK(sc.try_phonemize("by", "IN", none).empty(),  "'by' IN → no rule");
    }
}

static void test_symbol_words(const g::Lexicon& lx, const g::SpecialCases& sc) {
    const g::TokenContext none;

    const std::string percent = std::string(lx.lookup("percent"));
    const std::string andw    = std::string(lx.lookup("and"));
    const std::string plus    = std::string(lx.lookup("plus"));
    const std::string at      = std::string(lx.lookup("at"));
    const std::string dot     = std::string(lx.lookup("dot"));
    const std::string slash   = std::string(lx.lookup("slash"));
    CHECK(!percent.empty(), "lex has 'percent'");
    CHECK(!andw.empty(),    "lex has 'and'");
    CHECK(!plus.empty(),    "lex has 'plus'");
    CHECK(!at.empty(),      "lex has 'at'");
    CHECK(!dot.empty(),     "lex has 'dot'");
    CHECK(!slash.empty(),   "lex has 'slash'");

    CHECK(sc.try_phonemize("%", "",    none) == percent, "'%' → percent");
    CHECK(sc.try_phonemize("&", "",    none) == andw,    "'&' → and");
    CHECK(sc.try_phonemize("+", "",    none) == plus,    "'+' → plus");
    CHECK(sc.try_phonemize("@", "",    none) == at,      "'@' → at");
    CHECK(sc.try_phonemize(".", "ADD", none) == dot,     "'.' ADD → dot");
    CHECK(sc.try_phonemize("/", "ADD", none) == slash,   "'/' ADD → slash");

    // Bare '.' with the literal-dot PTB tag (".") does NOT fire — only
    // ADD-tagged dots are symbol words. Spec § "Tests" requirement.
    CHECK(sc.try_phonemize(".", ".", none).empty(), "'.' '.' → no rule");
}

static void test_dotted_acronym(const g::Lexicon& /*lx*/, const g::SpecialCases& sc) {
    const g::TokenContext none;
    const auto usa  = sc.try_phonemize("U.S.A.", "NNP", none);
    const auto spell_dotted = sc.spell_letter_by_letter("U.S.A.");
    const auto spell_plain  = sc.spell_letter_by_letter("USA");
    CHECK(!usa.empty(),         "'U.S.A.' fires dotted-acronym rule");
    CHECK(usa == spell_dotted,  "'U.S.A.' → spell('U.S.A.')");
    CHECK(usa == spell_plain,   "spell('U.S.A.') == spell('USA')");

    // Negative: longest part is 3 letters → no fire. The lexicon has no
    // entry for "abc.def" either, so the special-case engine returns "".
    CHECK(sc.try_phonemize("abc.def", "NN", none).empty(),
          "'abc.def' → no dotted-acronym rule (longest part = 3)");
    // Negative: no interior dot at all.
    CHECK(sc.try_phonemize("hello", "NN", none).empty(),
          "'hello' → no rule");
    // Negative: only-leading/trailing dots (no interior) — also no fire.
    CHECK(sc.try_phonemize("hi", "NN", none).empty(), "'hi' → no rule");
}

static void test_vs(const g::Lexicon& lx, const g::SpecialCases& sc) {
    const g::TokenContext none;
    const std::string versus = std::string(lx.lookup("versus"));
    CHECK(!versus.empty(), "lex has 'versus'");

    CHECK(sc.try_phonemize("vs",  "IN", none) == versus, "'vs'  IN → versus");
    CHECK(sc.try_phonemize("vs.", "IN", none) == versus, "'vs.' IN → versus");
    CHECK(sc.try_phonemize("VS",  "IN", none) == versus, "'VS'  IN → versus");
    CHECK(sc.try_phonemize("Vs.", "IN", none) == versus, "'Vs.' IN → versus");
    CHECK(sc.try_phonemize("vss", "IN", none).empty(),   "'vss' IN → no rule");
    // Wrong tag: no fire.
    CHECK(sc.try_phonemize("vs",  "NN", none).empty(),   "'vs' NN → no rule");
}

static void test_used(const g::Lexicon& lx, const g::SpecialCases& sc) {
    g::TokenContext to_true;  to_true.future_to  = true;
    g::TokenContext to_false; to_false.future_to = false;

    const std::string used_default = std::string(lx.lookup("used"));
    const std::string used_vbd     = std::string(lx.lookup("used", "VBD"));
    CHECK(!used_default.empty(), "lex has 'used' (default)");
    CHECK(!used_vbd.empty(),     "lex has 'used' (VBD variant)");

    CHECK(sc.try_phonemize("used", "VBD", to_true)  == used_vbd,
          "'used' VBD + future_to → VBD variant");
    CHECK(sc.try_phonemize("used", "JJ",  to_true)  == used_vbd,
          "'used' JJ + future_to → VBD variant");
    CHECK(sc.try_phonemize("used", "VBD", to_false) == used_default,
          "'used' VBD - future_to → default");
    CHECK(sc.try_phonemize("Used", "VBN", to_true)  == used_default,
          "'Used' VBN + future_to → default (VBN not in trigger set)");
}

static void test_spell_letter_by_letter(const g::Lexicon& lx, const g::SpecialCases& sc) {
    // Empty / non-letter inputs.
    CHECK(sc.spell_letter_by_letter("").empty(),     "spell('')   → empty");
    CHECK(sc.spell_letter_by_letter("---").empty(),  "spell('---') → empty");

    // 'hello' returns the concatenation of its letter IPAs with last-secondary
    // promotion. Per spec test 7, the no-special-case path for "hello" is
    // try_phonemize → ""; spell_letter_by_letter("hello") on the other hand
    // is a legitimate spelling and may be non-empty. Both behaviours are
    // documented in the spec.
    {
        const g::TokenContext none;
        CHECK(sc.try_phonemize("hello", "NN", none).empty(),
              "'hello' has no special case (caller falls through)");
    }

    // Build the expected "USA" spelling from the lexicon directly.
    const std::string U = std::string(lx.lookup("U"));
    const std::string S = std::string(lx.lookup("S"));
    const std::string A = std::string(lx.lookup("A"));
    CHECK(!U.empty(), "lex has 'U'");
    CHECK(!S.empty(), "lex has 'S'");
    CHECK(!A.empty(), "lex has 'A'");

    std::string concat = U + S + A;
    // Apply the last-secondary-to-primary promotion to `concat` to mirror
    // what spell_letter_by_letter must do.
    {
        const auto pos = concat.rfind(std::string(kSecondary));
        if (pos != std::string::npos) {
            concat[pos]     = kPrimary[0];
            concat[pos + 1] = kPrimary[1];
        }
    }
    const auto usa = sc.spell_letter_by_letter("USA");
    CHECK(!usa.empty(), "spell('USA') non-empty");
    CHECK(usa == concat,
          "spell('USA') == concat(lex(U,S,A)) with last-secondary promoted");

    // Case-insensitivity of the letter folding: spell("usa") == spell("USA").
    CHECK(sc.spell_letter_by_letter("usa") == usa, "spell('usa') == spell('USA')");

    // Stress-promotion no-op detection. Sweep A-Z and report whether any of
    // them carry a secondary stress mark in the bin. If none do, the
    // promotion branch is unreachable via spell_letter_by_letter — we still
    // exercise it by verifying the no-op behaviour (concat == promoted).
    bool any_secondary = false;
    for (char c = 'A'; c <= 'Z'; ++c) {
        const std::string_view letter(&c, 1);
        const auto ipa = lx.lookup(letter);
        if (ipa.find(kSecondary) != std::string_view::npos) {
            any_secondary = true;
            break;
        }
    }
    if (!any_secondary) {
        std::printf("test_special_cases: no A-Z lexicon entry carries U+02CC; "
                    "promotion branch of spell_letter_by_letter is a no-op for "
                    "the real bin (faithful misaki port, unreachable in "
                    "practice).\n");
        // The "promoted" concat above with no secondary in the source must
        // equal the bare concatenation.
        CHECK(usa == (U + S + A),
              "no-secondary case: spell('USA') == bare concat");
    } else {
        std::printf("test_special_cases: at least one A-Z entry carries U+02CC; "
                    "promotion branch covered by the USA assertion above.\n");
    }
}

// ─── Main ────────────────────────────────────────────────────────────────

int main() {
    const std::string path = resolve_lexicon_path();
    if (path.empty()) {
        std::printf("test_special_cases: BROSOUNDML_LEXICON_PATH unset and "
                    "sibling ../brosoundml-data/g2p/lexicon_en_us.bin not found "
                    "— real-bin section SKIPPED.\n"
                    "  Set BROSOUNDML_LEXICON_PATH=/path/to/lexicon_en_us.bin to enable.\n");
        return 0;
    }

    std::printf("test_special_cases: loading bin from %s\n", path.c_str());
    try {
        g::Lexicon lx = g::Lexicon::load(path);
        g::SpecialCases sc(lx);

        test_function_words(lx, sc);
        test_symbol_words(lx, sc);
        test_dotted_acronym(lx, sc);
        test_vs(lx, sc);
        test_used(lx, sc);
        test_spell_letter_by_letter(lx, sc);

        // Empty input — try_phonemize returns empty.
        {
            const g::TokenContext none;
            CHECK(sc.try_phonemize("", "", none).empty(),
                  "empty word → empty");
        }

        // Move-construct sanity.
        {
            g::SpecialCases a(lx);
            g::SpecialCases b = std::move(a);
            const g::TokenContext none;
            CHECK(b.try_phonemize("%", "", none) == std::string(lx.lookup("percent")),
                  "moved SpecialCases still works");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: real-bin section threw: %s\n", e.what());
        ++failures;
    }

    if (failures == 0) {
        std::printf("test_special_cases: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_special_cases: %d check(s) failed\n", failures);
    return 1;
}
