#pragma once

// GENERATED PLACEHOLDER — will be overwritten by tools/build_pos_dataset.py in chunk 2.
// 45 canonical PTB tags in fixed order. Order is frozen until the dataset
// builder re-emits this header from real UD data.

#include <cstdint>

namespace brosoundml::g2p {

enum class PosTag : std::uint8_t {
    CC = 0,
    CD,
    DT,
    EX,
    FW,
    IN,
    JJ,
    JJR,
    JJS,
    LS,
    MD,
    NN,
    NNS,
    NNP,
    NNPS,
    PDT,
    POS,
    PRP,
    PRP_DOLLAR,   // PRP$
    RB,
    RBR,
    RBS,
    RP,
    SYM,
    TO,
    UH,
    VB,
    VBD,
    VBG,
    VBN,
    VBP,
    VBZ,
    WDT,
    WP,
    WP_DOLLAR,    // WP$
    WRB,
    PUNCT_DOT,    // .
    PUNCT_COMMA,  // ,
    PUNCT_COLON,  // :
    PUNCT_OQUOTE, // ``
    PUNCT_CQUOTE, // ''
    PUNCT_LRB,    // -LRB-
    PUNCT_RRB,    // -RRB-
    HYPH,
    NFP,
};

inline constexpr int NUM_TAGS = 45;

// Canonical tag string by enum value. Index is static_cast<int>(PosTag).
inline constexpr const char* kPosTagNames[NUM_TAGS] = {
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
    "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB",
    "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN",
    "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", ".", ",", ":", "``",
    "''", "-LRB-", "-RRB-", "HYPH", "NFP",
};

}  // namespace brosoundml::g2p
