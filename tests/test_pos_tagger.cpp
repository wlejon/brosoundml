// POS tagger tests:
//   - link/smoke
//   - tokeniser byte-exact round-trip
//   - synthetic-weights load + forward (shape check only)
//   - gated on BROSOUNDML_POS_WEIGHTS env var: load real artifact + tag golden
//
// The accuracy assertion lands in chunk 2 once a trained model exists.

#include "brosoundml/g2p/pos_tagger.h"
#include "brosoundml/g2p/tags.h"
#include "g2p/pos_tagger_internal.h"  // tokenise_for_test, kVocab etc.

#include <brotensor/runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
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

// ─── Tokeniser round-trip ──────────────────────────────────────────────────

static void test_tokeniser_roundtrip() {
    const std::string sentence = "The cat sat";
    const auto chunks = g::tokenise_for_test(sentence);
    CHECK(chunks.size() == 1, "one chunk for short sentence");
    const auto& c = chunks[0];

    // Expected token stream: <bos> <wsep> T h e <wsep> c a t <wsep> s a t <eos>
    const std::vector<std::int32_t> expected = {
        g::kBos,
        g::kWsep, 'T'+4, 'h'+4, 'e'+4,
        g::kWsep, 'c'+4, 'a'+4, 't'+4,
        g::kWsep, 's'+4, 'a'+4, 't'+4,
        g::kEos
    };
    CHECK(c.token_ids == expected, "token stream matches expected encoding");
    CHECK(c.wsep_positions.size() == 3, "three wsep positions for three words");
    CHECK(c.word_spans.size() == 3,     "three word spans");

    // Byte-exact round-trip: recover each word from its byte span.
    const std::string_view sv = sentence;
    CHECK(sv.substr(c.word_spans[0].byte_start, c.word_spans[0].byte_len) == "The",
          "word 0 roundtrip");
    CHECK(sv.substr(c.word_spans[1].byte_start, c.word_spans[1].byte_len) == "cat",
          "word 1 roundtrip");
    CHECK(sv.substr(c.word_spans[2].byte_start, c.word_spans[2].byte_len) == "sat",
          "word 2 roundtrip");

    // Multi-space + leading/trailing whitespace.
    const auto c2 = g::tokenise_for_test("  hello   world  ");
    CHECK(c2.size() == 1, "whitespace-padded: one chunk");
    CHECK(c2[0].word_spans.size() == 2, "whitespace-padded: two words");

    // Empty input.
    const auto c3 = g::tokenise_for_test("");
    CHECK(c3.empty(), "empty input -> no chunks");

    // Whitespace-only input.
    const auto c4 = g::tokenise_for_test("   \t  ");
    CHECK(c4.empty(), "whitespace-only -> no chunks");

    // UTF-8 multi-byte word.
    const std::string s5 = "naive caf\xc3\xa9";   // "naive café"
    const auto c5 = g::tokenise_for_test(s5);
    CHECK(c5.size() == 1 && c5[0].word_spans.size() == 2, "utf8 two-word split");
    CHECK(std::string_view(s5).substr(c5[0].word_spans[1].byte_start,
                                      c5[0].word_spans[1].byte_len) == "caf\xc3\xa9",
          "utf8 word byte-exact");

    // Chunking: pack words until the sequence cap forces a split.
    std::string big;
    for (int i = 0; i < 100; ++i) {
        if (!big.empty()) big += ' ';
        big += "abcdefghij";                    // 10 bytes per word
    }
    const auto c6 = g::tokenise_for_test(big);
    CHECK(c6.size() > 1, "long input forces chunking");
    std::size_t total_words = 0;
    for (const auto& cc : c6) {
        CHECK(static_cast<int>(cc.token_ids.size()) <= g::kMaxSeqLen,
              "every chunk fits in max_seq_len");
        CHECK(cc.token_ids.front() == g::kBos, "chunk starts with bos");
        CHECK(cc.token_ids.back()  == g::kEos, "chunk ends with eos");
        total_words += cc.word_spans.size();
    }
    CHECK(total_words == 100, "no word lost across chunking boundary");
}

// ─── Synthetic weight blob ─────────────────────────────────────────────────

namespace {

void w_u32(std::ofstream& f, std::uint32_t v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
}
void w_u16(std::ofstream& f, std::uint16_t v) {
    f.write(reinterpret_cast<const char*>(&v), 2);
}
void w_u8 (std::ofstream& f, std::uint8_t v) {
    f.write(reinterpret_cast<const char*>(&v), 1);
}

void write_tensor(std::ofstream& f, const std::string& name,
                  int rows, int cols, std::mt19937& rng) {
    w_u16(f, static_cast<std::uint16_t>(name.size()));
    f.write(name.data(), name.size());
    const std::uint8_t rank = (cols == 1) ? 1 : 2;
    w_u8(f, rank);
    w_u32(f, static_cast<std::uint32_t>(rows));
    if (rank == 2) w_u32(f, static_cast<std::uint32_t>(cols));
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    for (std::size_t i = 0; i < n; ++i) {
        float v = dist(rng);
        f.write(reinterpret_cast<const char*>(&v), 4);
    }
}

struct TensorSpec { std::string name; int rows; int cols; };

std::vector<TensorSpec> make_specs() {
    std::vector<TensorSpec> v;
    auto push = [&](const std::string& n, int r, int c) {
        v.push_back({n, r, c});
    };
    push("token_emb", g::kVocab,     g::kDModel);
    push("pos_emb",   g::kMaxSeqLen, g::kDModel);
    for (int i = 0; i < g::kNumLayers; ++i) {
        const std::string p = "layer" + std::to_string(i) + ".";
        push(p + "ln1.gamma",  g::kDModel, 1);
        push(p + "ln1.beta",   g::kDModel, 1);
        push(p + "attn.Wq",    g::kDModel, g::kDModel);
        push(p + "attn.bq",    g::kDModel, 1);
        push(p + "attn.Wk",    g::kDModel, g::kDModel);
        push(p + "attn.bk",    g::kDModel, 1);
        push(p + "attn.Wv",    g::kDModel, g::kDModel);
        push(p + "attn.bv",    g::kDModel, 1);
        push(p + "attn.Wo",    g::kDModel, g::kDModel);
        push(p + "attn.bo",    g::kDModel, 1);
        push(p + "ln2.gamma",  g::kDModel, 1);
        push(p + "ln2.beta",   g::kDModel, 1);
        push(p + "ffn1.W",     g::kFFN,    g::kDModel);
        push(p + "ffn1.b",     g::kFFN,    1);
        push(p + "ffn2.W",     g::kDModel, g::kFFN);
        push(p + "ffn2.b",     g::kDModel, 1);
    }
    push("final_ln.gamma", g::kDModel, 1);
    push("final_ln.beta",  g::kDModel, 1);
    push("head.W",         g::NUM_TAGS, g::kDModel);
    push("head.b",         g::NUM_TAGS, 1);
    return v;
}

std::string write_synthetic_weights() {
    const auto path = (std::filesystem::temp_directory_path() /
                       "brosoundml_pos_synth.bin").string();
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("could not open temp weights file");

    const auto specs = make_specs();

    w_u32(f, 0x504F5302u);                              // magic
    w_u32(f, 1u);                                       // version
    w_u32(f, static_cast<std::uint32_t>(g::NUM_TAGS));
    w_u32(f, static_cast<std::uint32_t>(g::kDModel));
    w_u32(f, static_cast<std::uint32_t>(g::kNumLayers));
    w_u32(f, static_cast<std::uint32_t>(g::kNumHeads));
    w_u32(f, static_cast<std::uint32_t>(g::kFFN));
    w_u32(f, static_cast<std::uint32_t>(g::kMaxSeqLen));
    w_u32(f, static_cast<std::uint32_t>(specs.size()));

    std::mt19937 rng(0xC0FFEEu);
    for (const auto& s : specs) {
        write_tensor(f, s.name, s.rows, s.cols, rng);
    }
    f.close();
    return path;
}

}  // namespace

// ─── Synthetic-weights forward (shape only) ───────────────────────────────

static void test_synthetic_forward() {
    std::string path;
    try { path = write_synthetic_weights(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: synthetic weights write: %s\n", e.what());
        ++failures;
        return;
    }

    g::PosTagger tagger = g::PosTagger::load(path);

    const std::string sentence = "The quick brown fox jumps over the lazy dog";
    const auto tags = tagger.tag(sentence);
    CHECK(tags.size() == 9, "tag() returns one WordTag per input word");

    // Word strings must view into the input sentence.
    const std::vector<std::string_view> expected_words = {
        "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"
    };
    for (std::size_t i = 0; i < tags.size(); ++i) {
        CHECK(tags[i].word == expected_words[i], "word i matches input word");
        const int t = static_cast<int>(tags[i].tag);
        CHECK(t >= 0 && t < g::NUM_TAGS, "tag id in [0, NUM_TAGS)");
    }

    // Long sentence forced into multiple chunks must still yield one tag per word.
    std::string big;
    for (int i = 0; i < 120; ++i) {
        if (!big.empty()) big += ' ';
        big += "abcdefghij";
    }
    const auto big_tags = tagger.tag(big);
    CHECK(big_tags.size() == 120, "chunked tagging covers every word");

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ─── Optional: real-weights smoke against pos_golden.txt ──────────────────

static std::string env_or_empty(const char* name) {
    if (const char* v = std::getenv(name)) return std::string(v);
    return {};
}

static void test_golden_smoke() {
    const std::string weights = env_or_empty("BROSOUNDML_POS_WEIGHTS");
    if (weights.empty()) {
        std::printf("test_pos_tagger: BROSOUNDML_POS_WEIGHTS unset — gated branch skipped\n");
        return;
    }

#ifndef BROSOUNDML_REPO_DIR
    std::fprintf(stderr, "FAIL: BROSOUNDML_REPO_DIR not defined\n");
    ++failures;
    return;
#else
    const std::string golden_path =
        std::string(BROSOUNDML_REPO_DIR) + "/tests/pos_golden.txt";
    std::ifstream gf(golden_path);
    if (!gf) {
        std::fprintf(stderr, "FAIL: could not open %s\n", golden_path.c_str());
        ++failures;
        return;
    }

    g::PosTagger tagger = g::PosTagger::load(weights);

    std::string line;
    int total_words = 0;
    while (std::getline(gf, line)) {
        if (line.empty()) continue;
        // Strip word/TAG markers to recover the sentence text.
        std::string sentence;
        std::size_t i = 0;
        while (i < line.size()) {
            while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
            const std::size_t start = i;
            while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i]))) ++i;
            std::string tok = line.substr(start, i - start);
            const auto slash = tok.rfind('/');
            const std::string word = (slash == std::string::npos) ? tok : tok.substr(0, slash);
            if (!sentence.empty()) sentence += ' ';
            sentence += word;
        }
        const auto tags = tagger.tag(sentence);
        for (const auto& t : tags) {
            CHECK(static_cast<int>(t.tag) >= 0 &&
                  static_cast<int>(t.tag) <  g::NUM_TAGS,
                  "real-weights tag id in [0, NUM_TAGS)");
            ++total_words;
        }
    }
    CHECK(total_words > 0, "golden tagging produced at least one tag");
#endif
}

int main() {
    brotensor::init();
    test_tokeniser_roundtrip();
    test_synthetic_forward();
    test_golden_smoke();

    if (failures == 0) {
        std::printf("test_pos_tagger: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_pos_tagger: %d check(s) failed\n", failures);
    return 1;
}
