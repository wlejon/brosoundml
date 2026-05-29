#pragma once

// English text normalization for G2P. Runs at the very front of the Phonemizer,
// before pre-tokenization, rewriting a raw sentence into one whose tokens the
// lexicon / morphology / special-case chain can actually pronounce. Without it,
// digit and symbol tokens miss every lookup stage and either vanish to silence
// (numbers, %, $) or get spelled letter-by-letter.
//
// Transforms (US English only):
//   - smart apostrophes/quotes -> ASCII   ('’“” -> ' ")
//   - cardinal numbers          (42      -> "forty two")
//   - thousands separators       (1,000   -> "one thousand")
//   - decimals                  (3.14    -> "three point one four")
//   - ordinals                  (21st    -> "twenty first")
//   - years (4-digit heuristic) (1984    -> "nineteen eighty four")
//   - currency ($ £ €)          ($5.50   -> "five dollars and fifty cents")
//   - percent                   (50%     -> "fifty percent")
//   - bare operators            (3+4     -> "three plus four")
//
// The 4-digit year reading is a heuristic: a bare four-digit integer in
// [1000, 2099] with no grouping/decimal/currency is read as two pairs
// ("nineteen eighty four"); everything else reads as a cardinal. Numeric tokens
// inside currency, with thousands commas, or outside that range stay cardinal.

#include <string>
#include <string_view>

namespace brosoundml::g2p {

// Normalize `sentence` into pronounceable words. Pure function — no model or
// lexicon needed. Output is whitespace-padded around expansions; the caller's
// pre-tokenizer collapses the extra spaces.
std::string normalize_text(std::string_view sentence);

}  // namespace brosoundml::g2p
