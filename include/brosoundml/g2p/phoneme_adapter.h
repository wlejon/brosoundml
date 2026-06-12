#pragma once

// IPA-codepoint → Kokoro phoneme-id adapter. See docs/phonemizer.md.
//
// Kokoro's phoneme vocab (read from `weights/kokoro/config.json` into
// `KokoroConfig::vocab`) maps one IPA codepoint (or, in three combining-mark
// cases, two codepoints) to a small integer id. The adapter is a deterministic
// codepoint-level encoder over that map — no BPE, no merges, no normalisation.
// Codepoints absent from the vocab are dropped silently, mirroring misaki's
// behaviour and Kokoro upstream.
//
// Sole consumer is the `Phonemizer` façade; the adapter is stateless besides
// the borrowed vocab pointer and a cached `max_id_`.

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brosoundml::g2p {

class PhonemeAdapter {
 public:
    // Borrows `vocab`. Lifetime of `vocab` must exceed this adapter.
    explicit PhonemeAdapter(const std::unordered_map<std::string, int>& vocab);

    // Encode an IPA string into Kokoro phoneme ids. Walks `ipa` codepoint by
    // codepoint, looks up each codepoint's UTF-8 byte sequence in the vocab,
    // and appends the id. Codepoints absent from the vocab are SKIPPED
    // silently — matching misaki / Kokoro upstream.
    //
    // The vocab carries a few two-codepoint sequences for combining-mark
    // diacritics (`ã`, `ẽ`, `ĩ` etc.). When the current codepoint paired with
    // the next codepoint exists as a vocab key, the two-codepoint key wins
    // and both codepoints are consumed. This is a one-codepoint right-greedy
    // peek, not a BPE merge step.
    std::vector<std::int32_t> encode(std::string_view ipa) const;

    // Largest id present in the vocab. Useful for sizing embedding tables in
    // tests; mirrors `KokoroConfig::n_tokens` without depending on it.
    int max_id() const { return max_id_; }

 private:
    const std::unordered_map<std::string, int>* vocab_;
    int                                         max_id_;
};

}  // namespace brosoundml::g2p
