// Text-normalizer tests: smart-quote folding, cardinals, thousands separators,
// decimals, ordinals, the 4-digit year heuristic, currency, percent, and bare
// operators. Pure function — no lexicon or model needed.

#include "brosoundml/g2p/normalizer.h"

#include <cstdio>
#include <string>
#include <string_view>

namespace g = brosoundml::g2p;

static int failures = 0;

// Collapse runs of whitespace and trim, so tests can assert the readable form
// without caring about the normalizer's tokenization padding.
static std::string squeeze(std::string_view s) {
    std::string out;
    bool inSpace = true;  // leading-trim
    for (char c : s) {
        const bool ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (ws) {
            if (!inSpace) { out.push_back(' '); inSpace = true; }
        } else {
            out.push_back(c);
            inSpace = false;
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

static void check_eq(std::string_view in, std::string_view expect) {
    const std::string got = squeeze(g::normalize_text(in));
    if (got != expect) {
        std::fprintf(stderr, "FAIL: normalize_text(\"%.*s\") = \"%s\", expected \"%.*s\"\n",
                     static_cast<int>(in.size()), in.data(), got.c_str(),
                     static_cast<int>(expect.size()), expect.data());
        ++failures;
    }
}

int main() {
    std::printf("test_normalizer:\n");

    // ── Cardinals ──
    check_eq("0", "zero");
    check_eq("7", "seven");
    check_eq("42", "forty two");
    check_eq("100", "one hundred");
    check_eq("305", "three hundred five");
    check_eq("1000000", "one million");
    check_eq("1234567", "one million two hundred thirty four thousand five hundred sixty seven");

    // ── Thousands separators ──
    check_eq("1,000", "one thousand");
    check_eq("12,345", "twelve thousand three hundred forty five");
    // A grammatical comma after a number is NOT a separator.
    check_eq("I have 100, and more", "I have one hundred, and more");

    // ── Decimals ──
    check_eq("3.14", "three point one four");
    check_eq("0.5", "zero point five");
    check_eq("12.05", "twelve point zero five");

    // ── Ordinals ──
    check_eq("1st", "first");
    check_eq("2nd", "second");
    check_eq("3rd", "third");
    check_eq("21st", "twenty first");
    check_eq("100th", "one hundredth");

    // ── 4-digit year heuristic ──
    check_eq("1984", "nineteen eighty four");
    check_eq("2024", "twenty twenty four");
    check_eq("2005", "twenty oh five");
    check_eq("1900", "nineteen hundred");
    check_eq("2000", "two thousand");
    check_eq("1066", "ten sixty six");
    // Outside the year range, or grouped, it stays cardinal.
    check_eq("3000", "three thousand");
    check_eq("1,984", "one thousand nine hundred eighty four");

    // ── Currency ──
    check_eq("$5", "five dollars");
    check_eq("$1", "one dollar");
    check_eq("$5.50", "five dollars and fifty cents");
    check_eq("$1.01", "one dollar and one cent");
    check_eq("$0.50", "fifty cents");
    check_eq("$1,234.56",
             "one thousand two hundred thirty four dollars and fifty six cents");

    // ── Percent ──
    check_eq("50%", "fifty percent");
    check_eq("100%", "one hundred percent");
    check_eq("3.5%", "three point five percent");

    // ── Bare operators ──
    check_eq("3+4", "three plus four");
    check_eq("2=2", "two equals two");
    check_eq("R&D", "R and D");
    // Negative sign at a word boundary; internal hyphens are left alone.
    check_eq("it's -5 outside", "it's minus five outside");
    check_eq("well-known", "well-known");

    // ── Smart-quote folding ──
    check_eq("don\xE2\x80\x99t", "don't");                 // don’t
    check_eq("\xE2\x80\x9Cyes\xE2\x80\x9D", "\"yes\"");    // “yes”

    // ── Mixed sentence ──
    check_eq("It costs $20 and is 15% off on the 2nd.",
             "It costs twenty dollars and is fifteen percent off on the second.");

    if (failures == 0) {
        std::printf("test_normalizer: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_normalizer: %d failure(s)\n", failures);
    return 1;
}
