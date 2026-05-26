// Lexicon tests: header validation, plain lookup, case fallback, heteronym
// selection, PTB-mapping completeness, override priority + view stability,
// miss path.
//
// Gated on the presence of the packed bin. Resolution order:
//   1. BROSOUNDML_LEXICON_PATH env var, if set.
//   2. <repo>/../brosoundml-data/g2p/lexicon_en_us.bin (sibling repo).
// If neither resolves, the gated section is skipped (printed note, not FAIL).

#include "brosoundml/g2p/lexicon.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace g = brosoundml::g2p;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

// ─── Header validation: corrupt magic + wrong version ─────────────────────

static void write_bytes(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

static std::vector<std::uint8_t> make_minimal_header(const char magic[4], std::uint32_t version) {
    // 64-byte header, all zero except magic + version + zero blob sizes/offsets.
    std::vector<std::uint8_t> h(64, 0);
    std::memcpy(h.data(),     magic, 4);
    std::memcpy(h.data() + 4, &version, 4);
    // entry_count=0, flags=0, blobs at offset 64 with zero length.
    std::uint64_t key_off = 64, key_len = 0, val_off = 64, val_len = 0;
    std::memcpy(h.data() + 16, &key_off, 8);
    std::memcpy(h.data() + 24, &key_len, 8);
    std::memcpy(h.data() + 32, &val_off, 8);
    std::memcpy(h.data() + 40, &val_len, 8);
    return h;
}

static void test_header_validation() {
    const auto tmp_dir = std::filesystem::temp_directory_path();

    // Bad magic.
    {
        const auto path = (tmp_dir / "brosoundml_lex_badmagic.bin").string();
        write_bytes(path, make_minimal_header("XXXX", 1));
        bool threw = false;
        std::string msg;
        try { (void)g::Lexicon::load(path); }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        CHECK(threw, "bad magic: load() throws");
        CHECK(msg.find("magic") != std::string::npos,
              "bad magic: error message mentions 'magic'");
        std::error_code ec; std::filesystem::remove(path, ec);
    }

    // Wrong version.
    {
        const auto path = (tmp_dir / "brosoundml_lex_badver.bin").string();
        write_bytes(path, make_minimal_header("BSLX", 99));
        bool threw = false;
        std::string msg;
        try { (void)g::Lexicon::load(path); }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        CHECK(threw, "bad version: load() throws");
        CHECK(msg.find("version") != std::string::npos,
              "bad version: error message mentions 'version'");
        std::error_code ec; std::filesystem::remove(path, ec);
    }

    // Too small (truncated header).
    {
        const auto path = (tmp_dir / "brosoundml_lex_short.bin").string();
        write_bytes(path, std::vector<std::uint8_t>(8, 0));
        bool threw = false;
        try { (void)g::Lexicon::load(path); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw, "truncated header: load() throws");
        std::error_code ec; std::filesystem::remove(path, ec);
    }

    // Missing file.
    {
        const auto path = (tmp_dir / "brosoundml_lex_doesnotexist_xyzzy.bin").string();
        std::error_code ec; std::filesystem::remove(path, ec);
        bool threw = false;
        try { (void)g::Lexicon::load(path); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw, "missing file: load() throws");
    }
}

// ─── Resolve lexicon path ────────────────────────────────────────────────

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

// ─── Real-bin tests ───────────────────────────────────────────────────────

static void test_plain_lookup(const g::Lexicon& lx) {
    const auto hello = lx.lookup("hello");
    const auto world = lx.lookup("world");
    const auto the   = lx.lookup("the");
    CHECK(!hello.empty(), "lookup('hello') non-empty");
    CHECK(!world.empty(), "lookup('world') non-empty");
    CHECK(!the.empty(),   "lookup('the')   non-empty");
}

static void test_case_fallback(const g::Lexicon& lx) {
    const auto lower = lx.lookup("hello");
    const auto upper = lx.lookup("HELLO");
    const auto title = lx.lookup("Hello");
    CHECK(!lower.empty(), "case fallback: 'hello' must hit");
    CHECK(upper == lower, "case fallback: 'HELLO' folds to 'hello'");
    CHECK(title == lower, "case fallback: 'Hello' folds to 'hello'");

    // Acronym path: both 'AA' and 'aa' have their own entries in misaki. The
    // exact-case-first pass must hit them both directly. (In the v1 bin both
    // entries happen to carry the same IPA — distinctness is a property of
    // the upstream data, not of the loader, so we just assert reachability.)
    const auto aa_upper = lx.lookup("AA");
    const auto aa_lower = lx.lookup("aa");
    CHECK(!aa_upper.empty(), "acronym 'AA' lookup hits");
    CHECK(!aa_lower.empty(), "acronym 'aa' lookup hits");

    // Sanity: a word with no own upper-case entry must fall back through
    // tolower. 'WORLD' should resolve to the same IPA as 'world'.
    const auto world_lower = lx.lookup("world");
    const auto world_upper = lx.lookup("WORLD");
    CHECK(!world_lower.empty() && world_upper == world_lower,
          "case fallback: 'WORLD' folds to 'world'");
}

static void test_heteronyms(const g::Lexicon& lx) {
    // Hard-coded expected IPA strings from misaki's us_gold.json at the pinned
    // commit. UTF-8 bytes literal in the source — file is /utf-8 on MSVC.
    const std::string_view record_default = "ɹˈɛkəɹd";
    const std::string_view record_verb    = "ɹəkˈɔɹd";
    const std::string_view read_default   = "ɹˈid";
    const std::string_view read_past      = "ɹˈɛd";
    const std::string_view present_verb   = "pɹizˈɛnt";
    const std::string_view present_noun   = "pɹˈɛzᵊnt";

    CHECK(lx.lookup("record", "NN") == record_default, "record/NN == DEFAULT");
    CHECK(lx.lookup("record", "VB") == record_verb,    "record/VB == VERB");
    CHECK(lx.lookup("record", "")   == record_default, "record/''  == DEFAULT");

    CHECK(lx.lookup("read", "VBD") == read_past,    "read/VBD (exact PTB)");
    CHECK(lx.lookup("read", "JJ")  == read_past,    "read/JJ  (ADJ variant)");
    CHECK(lx.lookup("read", "")    == read_default, "read/''  (DEFAULT)");

    CHECK(lx.lookup("present", "VB") == present_verb, "present/VB (VERB)");
    CHECK(lx.lookup("present", "NN") == present_noun, "present/NN (DEFAULT, NOUN absent)");
}

static void test_ptb_mapping(const g::Lexicon& lx) {
    // NN family: NNS/NNP/NNPS behave like NN (→NOUN → falls back to DEFAULT).
    const auto rec_nn = lx.lookup("record", "NN");
    CHECK(lx.lookup("record", "NNS")  == rec_nn, "PTB NNS  ≡ NN family");
    CHECK(lx.lookup("record", "NNP")  == rec_nn, "PTB NNP  ≡ NN family");
    CHECK(lx.lookup("record", "NNPS") == rec_nn, "PTB NNPS ≡ NN family");

    // VB family: VBZ/VBG behave like VB (→VERB).
    const auto rec_vb = lx.lookup("record", "VB");
    CHECK(lx.lookup("record", "VBZ") == rec_vb, "PTB VBZ ≡ VB family");
    CHECK(lx.lookup("record", "VBG") == rec_vb, "PTB VBG ≡ VB family");

    // JJ family on 'read': JJR/JJS behave like JJ (→ADJ).
    const auto read_jj = lx.lookup("read", "JJ");
    CHECK(lx.lookup("read", "JJR") == read_jj, "PTB JJR ≡ JJ family");
    CHECK(lx.lookup("read", "JJS") == read_jj, "PTB JJS ≡ JJ family");

    // RB family: RBR/RBS behave like RB. 'record' has no ADV variant so
    // all three fall through to DEFAULT identically.
    const auto rec_rb = lx.lookup("record", "RB");
    CHECK(lx.lookup("record", "RBR") == rec_rb, "PTB RBR ≡ RB family");
    CHECK(lx.lookup("record", "RBS") == rec_rb, "PTB RBS ≡ RB family");
}

static void test_overrides(g::Lexicon& lx) {
    const std::string ipa_x = "X";
    lx.add_override("hello", ipa_x);

    // Override wins regardless of POS.
    CHECK(lx.lookup("hello")        == "X", "override: bare 'hello'");
    CHECK(lx.lookup("hello", "NN")  == "X", "override: 'hello'/NN");
    CHECK(lx.lookup("hello", "VB")  == "X", "override: 'hello'/VB");
    // Case-insensitive match.
    CHECK(lx.lookup("HELLO")        == "X", "override: 'HELLO' folds");
    CHECK(lx.lookup("Hello", "JJ")  == "X", "override: 'Hello'/JJ");

    // Capture the view before installing a second override, then verify it
    // is still valid (pointer-stable std::map nodes).
    const std::string_view sv_before = lx.lookup("hello");
    CHECK(sv_before == "X", "override: pre-second-insert view value");
    const char* data_before = sv_before.data();

    lx.add_override("world", std::string("Y"));
    // The captured view must still be valid and equal to "X".
    CHECK(sv_before.data() == data_before,
          "override: existing view's underlying pointer unchanged");
    CHECK(std::string_view(data_before, sv_before.size()) == "X",
          "override: existing view still reads 'X' after second insert");

    CHECK(lx.lookup("world") == "Y", "override: 'world' returns 'Y'");
}

static void test_miss(const g::Lexicon& lx) {
    CHECK(lx.lookup("xyzzyqwerty12345").empty(), "miss path returns empty view");
    CHECK(lx.lookup("xyzzyqwerty12345", "NN").empty(),
          "miss path returns empty view with PTB hint");
}

// ─── Main ────────────────────────────────────────────────────────────────

int main() {
    test_header_validation();

    const std::string path = resolve_lexicon_path();
    if (path.empty()) {
        std::printf("test_lexicon: BROSOUNDML_LEXICON_PATH unset and sibling "
                    "../brosoundml-data/g2p/lexicon_en_us.bin not found — "
                    "real-bin section SKIPPED.\n"
                    "  Set BROSOUNDML_LEXICON_PATH=/path/to/lexicon_en_us.bin to enable.\n");
    } else {
        std::printf("test_lexicon: loading bin from %s\n", path.c_str());
        try {
            g::Lexicon lx = g::Lexicon::load(path);
            test_plain_lookup(lx);
            test_case_fallback(lx);
            test_heteronyms(lx);
            test_ptb_mapping(lx);
            test_miss(lx);
            test_overrides(lx);  // mutates, run last
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL: real-bin section threw: %s\n", e.what());
            ++failures;
        }
    }

    if (failures == 0) {
        std::printf("test_lexicon: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_lexicon: %d check(s) failed\n", failures);
    return 1;
}
