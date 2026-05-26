#pragma once

// Packed English pronunciation dictionary. See docs/lexicon.md for the spec.
//
// First lookup stage of the in-tree G2P stack: word (+ optional PTB POS tag)
// → IPA UTF-8 string. Returns an empty string_view when the word is absent
// after case fallback; callers fall through to the morphology slice.

#include <memory>
#include <string>
#include <string_view>

namespace brosoundml::g2p {

class Lexicon {
 public:
    // Load the packed binary from disk. Throws std::runtime_error on missing
    // file, bad magic, unsupported version, or any other malformed-header
    // condition.
    static Lexicon load(const std::string& path);

    // Look up `word`, optionally biased by a PTB tag for heteronym
    // selection. Returns an empty string_view if the word is not present
    // (after ASCII case fallback).
    //
    // Resolution order:
    //   1. Overrides (matched by ASCII-folded key).
    //   2. Exact-case bin lookup.
    //   3. ASCII-lowercased bin lookup (only if word had any ASCII upper).
    //   4. Empty string_view.
    // On a hit with a variant entry, the variant is chosen via:
    //   exact PTB tag id → mapped misaki tag → DEFAULT.
    //
    // The returned view is stable for the lifetime of this Lexicon and
    // remains valid across add_override() calls.
    std::string_view lookup(std::string_view word,
                            std::string_view ptb_pos = "") const;

    // Install a runtime override. Subsequent lookup(word, _) returns `ipa`
    // regardless of POS. The override matches by ASCII-folded word, so
    // add_override("Foo", …) is the same as add_override("FOO", …).
    //
    // Overrides are in-memory only; they are not written back to disk.
    // Re-loading the Lexicon drops them. Existing string_views handed out
    // by lookup() remain valid across add_override() calls (overrides are
    // stored in an std::map; nodes are pointer-stable).
    void add_override(std::string_view word, std::string ipa);

    Lexicon(Lexicon&&) noexcept;
    Lexicon& operator=(Lexicon&&) noexcept;
    ~Lexicon();

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Lexicon();
};

}  // namespace brosoundml::g2p
