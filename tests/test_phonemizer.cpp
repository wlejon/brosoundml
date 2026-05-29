// Phonemizer tests: end-to-end sentence → Kokoro phoneme ids. This test
// composes every prior G2P slice (PosTagger + Lexicon + Morphology +
// SpecialCases + PhonemeAdapter) so it needs all three artifacts:
//
//   1. Lexicon bin:     BROSOUNDML_LEXICON_PATH   or sibling ../brosoundml-data
//   2. POS weights:     BROSOUNDML_POS_WEIGHTS    or sibling ../brosoundml-data
//   3. Kokoro config:   BROSOUNDML_KOKORO_DIR     or <repo>/weights/kokoro
//
// SKIP (return 0, printed note) if any are missing.

#include "brosoundml/g2p/lexicon.h"
#include "brosoundml/g2p/morphology.h"
#include "brosoundml/g2p/special_cases.h"
#include "brosoundml/g2p/pos_tagger.h"
#include "brosoundml/g2p/phoneme_adapter.h"
#include "brosoundml/g2p/phonemizer.h"
#include "brosoundml/g2p/tags.h"
#include "brosoundml/detail/json.h"

#include <brotensor/runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace g = brosoundml::g2p;
namespace j = brosoundml::detail::json;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

// ─── Path resolution ─────────────────────────────────────────────────────

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

static std::string resolve_pos_weights_path() {
    const auto env = env_or_empty("BROSOUNDML_POS_WEIGHTS");
    if (!env.empty() && std::filesystem::exists(env)) return env;
#ifdef BROSOUNDML_REPO_DIR
    // Try a few conventional sibling locations.
    const std::string candidates[] = {
        std::string(BROSOUNDML_REPO_DIR) + "/weights/pos_tagger/model.bin",
        std::string(BROSOUNDML_REPO_DIR) + "/../brosoundml-data/g2p/pos_tagger.bin",
        std::string(BROSOUNDML_REPO_DIR) + "/../brosoundml-data/g2p/pos_tagger.safetensors",
    };
    for (const auto& c : candidates) {
        if (std::filesystem::exists(c)) return c;
    }
#endif
    return {};
}

static std::string resolve_kokoro_dir() {
    const auto env = env_or_empty("BROSOUNDML_KOKORO_DIR");
    if (!env.empty() && std::filesystem::exists(env + "/config.json"))
        return env;
#ifdef BROSOUNDML_REPO_DIR
    const std::string sibling = std::string(BROSOUNDML_REPO_DIR) + "/weights/kokoro";
    if (std::filesystem::exists(sibling + "/config.json")) return sibling;
#endif
    return {};
}

static std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream os;
    os << f.rdbuf();
    return os.str();
}

static std::unordered_map<std::string, int> load_vocab(const std::string& kokoro_dir) {
    std::unordered_map<std::string, int> vocab;
    const std::string text = slurp(kokoro_dir + "/config.json");
    const j::Value    root = j::parse(text);
    const j::Value* v = root.find("vocab");
    if (!v || !v->is_object()) return vocab;
    for (const auto& m : v->as_object()) {
        vocab.emplace(m.first, static_cast<int>(m.second.as_number()));
    }
    return vocab;
}

// ─── Helpers ─────────────────────────────────────────────────────────────

// Build an inverse vocab: id → IPA string. Drops the multi-codepoint combining
// entries' priority for keys (we just want one decode per id — first wins).
static std::unordered_map<int, std::string> invert_vocab(
        const std::unordered_map<std::string, int>& vocab) {
    std::unordered_map<int, std::string> inv;
    for (const auto& kv : vocab) {
        inv.emplace(kv.second, kv.first);
    }
    return inv;
}

static std::string decode_ids(const std::vector<std::int32_t>& ids,
                              const std::unordered_map<int, std::string>& inv) {
    std::string out;
    for (auto id : ids) {
        auto it = inv.find(id);
        if (it != inv.end()) out += it->second;
    }
    return out;
}

static bool contains(std::string_view haystack, std::string_view needle) {
    return haystack.find(needle) != std::string_view::npos;
}

// ─── Tests ───────────────────────────────────────────────────────────────

int main() {
    brotensor::init();

    const std::string lex_path = resolve_lexicon_path();
    const std::string pos_path = resolve_pos_weights_path();
    const std::string kok_dir  = resolve_kokoro_dir();
    if (lex_path.empty() || pos_path.empty() || kok_dir.empty()) {
        std::printf("test_phonemizer: missing artifact, SKIPPED.\n"
                    "  lexicon: %s\n  pos weights: %s\n  kokoro dir: %s\n"
                    "  Set BROSOUNDML_LEXICON_PATH / BROSOUNDML_POS_WEIGHTS / "
                    "BROSOUNDML_KOKORO_DIR or place artifacts at the standard "
                    "sibling paths.\n",
                    lex_path.empty() ? "MISSING" : lex_path.c_str(),
                    pos_path.empty() ? "MISSING" : pos_path.c_str(),
                    kok_dir.empty()  ? "MISSING" : kok_dir.c_str());
        return 0;
    }
    std::printf("test_phonemizer:\n  lexicon: %s\n  pos: %s\n  kokoro: %s\n",
                lex_path.c_str(), pos_path.c_str(), kok_dir.c_str());

    try {
        g::Lexicon       lex     = g::Lexicon::load(lex_path);
        g::Morphology    morph(lex);
        g::SpecialCases  sc(lex);
        g::PosTagger     tagger  = g::PosTagger::load(pos_path);
        const auto       vocab   = load_vocab(kok_dir);
        const auto       inv     = invert_vocab(vocab);
        g::PhonemeAdapter adapter(vocab);

        g::Phonemizer phon(tagger, lex, morph, sc, adapter);

        // ─── Test 1: Hello, world. ──────────────────────────────────────
        {
            const auto ids = phon.phonemize("Hello, world.");
            CHECK(!ids.empty(), "phonemize('Hello, world.') is non-empty");
            // Last id is '.' (id 4 per Kokoro vocab).
            CHECK(!ids.empty() && ids.back() == 4,
                  "phonemize('Hello, world.') ends with id 4 ('.')");
            const std::string decoded = decode_ids(ids, inv);
            CHECK(!decoded.empty() && decoded[0] == 'h',
                  "decoded IPA starts with 'h'");
            std::printf("  hello-world ids (%zu): ", ids.size());
            for (auto id : ids) std::printf("%d ", id);
            std::printf("\n  decoded: %s\n", decoded.c_str());
        }

        // ─── Test 2: function-word vowel allomorphy ─────────────────────
        {
            const auto a = phon.phonemize_to_ipa("the apple");
            const auto b = phon.phonemize_to_ipa("the cat");
            // "ði" (ð U+00F0, i U+0069) → "\xC3\xB0i"
            // "ðə" (ð U+00F0, ə U+0259) → "\xC3\xB0\xC9\x99"
            CHECK(contains(a, "\xC3\xB0i"),
                  "'the apple' contains 'ði' (vowel form)");
            CHECK(contains(b, "\xC3\xB0\xC9\x99"),
                  "'the cat' contains 'ðə' (consonant form)");
            std::printf("  the-apple: %s\n  the-cat:   %s\n", a.c_str(), b.c_str());
        }

        // ─── Test 3: 'to' allomorphy ────────────────────────────────────
        {
            const auto a = phon.phonemize_to_ipa("go to apple");
            const auto b = phon.phonemize_to_ipa("go to school");
            const auto c = phon.phonemize_to_ipa("go to");
            // "tʊ" → "t\xCA\x8A"   "tə" → "t\xC9\x99"
            CHECK(contains(a, "t\xCA\x8A"),
                  "'go to apple' contains 'tʊ' (vowel form)");
            CHECK(contains(b, "t\xC9\x99"),
                  "'go to school' contains 'tə' (consonant form)");
            const std::string to_default = std::string(lex.lookup("to"));
            CHECK(!to_default.empty(), "lex has 'to' default");
            CHECK(contains(c, to_default),
                  "'go to' (sentence-final) contains lex('to') default");
            std::printf("  go-to-apple:  %s\n  go-to-school: %s\n  go-to:        %s\n",
                        a.c_str(), b.c_str(), c.c_str());
        }

        // ─── Test 4: POS-driven heteronym (record) ──────────────────────
        {
            const auto verb_sentence = "I record songs";
            const auto noun_sentence = "a record player";

            // Inspect the actual tagger output for "record" — diagnostic only.
            const auto vt = tagger.tag(verb_sentence);
            const auto nt = tagger.tag(noun_sentence);
            const char* vt_record = (vt.size() >= 2)
                ? g::kPosTagNames[static_cast<int>(vt[1].tag)] : "?";
            const char* nt_record = (nt.size() >= 2)
                ? g::kPosTagNames[static_cast<int>(nt[1].tag)] : "?";
            std::printf("  POS tagger: 'record' in \"%s\" → %s; in \"%s\" → %s\n",
                        verb_sentence, vt_record, noun_sentence, nt_record);

            const auto verb_ipa = phon.phonemize_to_ipa(verb_sentence);
            const auto noun_ipa = phon.phonemize_to_ipa(noun_sentence);
            std::printf("  record VERB ipa: %s\n  record NOUN ipa: %s\n",
                        verb_ipa.c_str(), noun_ipa.c_str());

            // Expected variants from the lexicon: VERB vs DEFAULT for "record".
            const std::string verb_form    = std::string(lex.lookup("record", "VB"));
            const std::string default_form = std::string(lex.lookup("record"));
            CHECK(!verb_form.empty(),    "lex has 'record' VERB variant");
            CHECK(!default_form.empty(), "lex has 'record' DEFAULT");
            CHECK(verb_form != default_form,
                  "lex 'record' VERB and DEFAULT differ (heteronym)");

            CHECK(contains(verb_ipa, verb_form),
                  "'I record songs' contains VERB-variant IPA");
            CHECK(contains(noun_ipa, default_form),
                  "'a record player' contains DEFAULT IPA");
        }

        // ─── Test 5: Morphology fallback (cats sat → cat + s) ───────────
        {
            const auto ipa = phon.phonemize_to_ipa("cats sat");
            const std::string cat_ipa = std::string(lex.lookup("cat"));
            CHECK(!cat_ipa.empty(), "lex has 'cat'");
            const std::string expect = cat_ipa + "s";
            CHECK(contains(ipa, expect),
                  "'cats sat' contains lex('cat') + 's' (morphology -s rule)");
            std::printf("  cats-sat: %s\n", ipa.c_str());
        }

        // ─── Test 6: Spelling fallback ──────────────────────────────────
        {
            const auto nasa_ipa = phon.phonemize_to_ipa("NASA launched");
            CHECK(!nasa_ipa.empty(), "'NASA launched' produces non-empty IPA");
            // KZQX is not in the lexicon; should spell letter by letter.
            const auto kzqx_ipa = phon.phonemize_to_ipa("KZQX launched");
            const std::string spelled = sc.spell_letter_by_letter("KZQX");
            CHECK(!spelled.empty(), "spell_letter_by_letter('KZQX') non-empty");
            CHECK(contains(kzqx_ipa, spelled),
                  "'KZQX launched' contains the letter-by-letter spelling");
            std::printf("  NASA-launched: %s\n  KZQX-launched: %s\n",
                        nasa_ipa.c_str(), kzqx_ipa.c_str());
        }

        // ─── Test 7: Punctuation passthrough ────────────────────────────
        {
            const auto a = phon.phonemize("hi!");
            CHECK(!a.empty() && a.back() == 5, "phonemize('hi!') ends with id 5 ('!')");
            const auto b = phon.phonemize("(yes)");
            CHECK(!b.empty() && b.front() == 12,
                  "phonemize('(yes)') starts with id 12 ('(')");
            CHECK(!b.empty() && b.back() == 13,
                  "phonemize('(yes)') ends with id 13 (')')");
        }

        // ─── Test 8: Empty input ────────────────────────────────────────
        {
            CHECK(phon.phonemize("").empty(), "phonemize('') → empty");
            CHECK(phon.phonemize("   ").empty(), "phonemize('   ') → empty");
            CHECK(phon.phonemize_to_ipa("").empty(),
                  "phonemize_to_ipa('') → empty");
        }

        // ─── Test 9: Smart-apostrophe normalisation ─────────────────────
        // Contractions written with a typographic apostrophe (U+2019, the one
        // LLMs emit) must phonemise identically to the straight-ASCII form, and
        // must NOT degrade to letter-by-letter spelling.
        {
            const auto straight = phon.phonemize_to_ipa("don't");
            const auto curly    = phon.phonemize_to_ipa("don\xE2\x80\x99t");  // don’t
            const auto spelled  = sc.spell_letter_by_letter("dont");
            CHECK(!straight.empty(), "'don\\'t' (straight) is non-empty");
            CHECK(curly == straight,
                  "'don\\u2019t' (curly) phonemises like the straight form");
            CHECK(curly != spelled,
                  "'don\\u2019t' is not spelled letter-by-letter");
            // Smart double quotes fold too: same ids as straight ASCII quotes.
            const auto dq_curly    = phon.phonemize("\xE2\x80\x9Cyes\xE2\x80\x9D"); // “yes”
            const auto dq_straight = phon.phonemize("\"yes\"");
            CHECK(!dq_straight.empty(), "phonemize('\"yes\"') is non-empty");
            CHECK(dq_curly == dq_straight,
                  "smart double quotes phonemise like straight quotes");
            std::printf("  smart-apos: straight=%s curly=%s\n",
                        straight.c_str(), curly.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: phonemizer section threw: %s\n", e.what());
        ++failures;
    }

    if (failures == 0) {
        std::printf("test_phonemizer: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_phonemizer: %d check(s) failed\n", failures);
    return 1;
}
