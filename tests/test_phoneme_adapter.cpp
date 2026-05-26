// PhonemeAdapter tests: codepoint-level IPA → Kokoro id mapping.
//
// Gated on a Kokoro config.json. Resolution order:
//   1. BROSOUNDML_KOKORO_DIR env var, if set.
//   2. <repo>/weights/kokoro/ (sibling fallback).
// If neither resolves, the test SKIPs (printed note, not FAIL).
//
// Vocab is loaded by parsing config.json directly with brosoundml's vendored
// JSON parser — avoids the heavyweight Kokoro::load path (which requires the
// full safetensors model on disk).

#include "brosoundml/g2p/phoneme_adapter.h"
#include "brosoundml/detail/json.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
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

static std::string env_or_empty(const char* name) {
    if (const char* v = std::getenv(name)) return std::string(v);
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

int main() {
    const std::string dir = resolve_kokoro_dir();
    if (dir.empty()) {
        std::printf("test_phoneme_adapter: BROSOUNDML_KOKORO_DIR unset and "
                    "sibling weights/kokoro/config.json not found — SKIPPED.\n"
                    "  Set BROSOUNDML_KOKORO_DIR=/path/to/kokoro to enable.\n");
        return 0;
    }
    std::printf("test_phoneme_adapter: loading vocab from %s/config.json\n",
                dir.c_str());

    auto vocab = load_vocab(dir);
    CHECK(vocab.size() >= 100, "vocab has >= 100 entries (real Kokoro config)");

    g::PhonemeAdapter adapter(vocab);

    // 1. Hello-world canary — verified against weights/kokoro/ids.txt line 1.
    {
        const std::vector<std::int32_t> expected = {
            50, 86, 156, 54, 57, 135, 16, 65, 83, 123, 54, 46
        };
        const auto got = adapter.encode("h\xC9\x9B\xCB\x88lo\xCA\x8A w\xC9\x99\xC9\xB9ld");
        CHECK(got == expected, "encode('hɛˈloʊ wəɹld') matches upstream sample");
        if (got != expected) {
            std::fprintf(stderr, "  got: [");
            for (std::size_t i = 0; i < got.size(); ++i)
                std::fprintf(stderr, "%s%d", i ? "," : "", got[i]);
            std::fprintf(stderr, "]\n");
        }
    }

    // 2. Empty input.
    CHECK(adapter.encode("").empty(), "encode('') → empty");

    // 3. Unknown codepoint dropped: capital X is not in the vocab.
    CHECK(adapter.encode("X").empty(), "encode('X') → empty (X absent)");

    // 4. Mixed known/unknown.
    {
        const std::vector<std::int32_t> expected = {50, 86};
        const auto got = adapter.encode("hX\xC9\x9B");
        CHECK(got == expected, "encode('hXɛ') → [50, 86]");
    }

    // 5. Stress marks: ˈ (U+02C8) = 156, ˌ (U+02CC) = 157.
    {
        const auto primary   = adapter.encode("\xCB\x88");
        const auto secondary = adapter.encode("\xCB\x8C");
        CHECK(primary.size()   == 1 && primary[0]   == 156, "encode('ˈ') → [156]");
        CHECK(secondary.size() == 1 && secondary[0] == 157, "encode('ˌ') → [157]");
    }

    // 6. Space is preserved (id 16, the word-separator).
    {
        const std::vector<std::int32_t> expected = {50, 16, 50};
        CHECK(adapter.encode("h h") == expected, "encode('h h') → [50, 16, 50]");
    }

    // 7. max_id().
    CHECK(adapter.max_id() == 177, "max_id() == 177");

    if (failures == 0) {
        std::printf("test_phoneme_adapter: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_phoneme_adapter: %d check(s) failed\n", failures);
    return 1;
}
