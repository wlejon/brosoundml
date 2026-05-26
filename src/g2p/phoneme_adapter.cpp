#include "brosoundml/g2p/phoneme_adapter.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace brosoundml::g2p {

namespace {

// Decode the UTF-8 byte length at position `i` of `s`. Returns 0 on a malformed
// lead byte or truncation — caller advances by 1 to resynchronise.
std::size_t utf8_len_at(std::string_view s, std::size_t i) {
    if (i >= s.size()) return 0;
    const auto b = static_cast<unsigned char>(s[i]);
    std::size_t n = 0;
    if      ((b & 0x80u) == 0x00u) n = 1;
    else if ((b & 0xE0u) == 0xC0u) n = 2;
    else if ((b & 0xF0u) == 0xE0u) n = 3;
    else if ((b & 0xF8u) == 0xF0u) n = 4;
    else                           return 0;
    if (i + n > s.size()) return 0;
    // Verify continuation bytes (10xxxxxx).
    for (std::size_t k = 1; k < n; ++k) {
        if ((static_cast<unsigned char>(s[i + k]) & 0xC0u) != 0x80u) return 0;
    }
    return n;
}

}  // namespace

PhonemeAdapter::PhonemeAdapter(const std::unordered_map<std::string, int>& vocab)
    : vocab_(&vocab), max_id_(0) {
    for (const auto& kv : vocab) {
        if (kv.second > max_id_) max_id_ = kv.second;
    }
}

std::vector<std::int32_t> PhonemeAdapter::encode(std::string_view ipa) const {
    std::vector<std::int32_t> out;
    out.reserve(ipa.size());

    std::size_t i = 0;
    while (i < ipa.size()) {
        std::size_t n = utf8_len_at(ipa, i);
        if (n == 0) {
            // Malformed byte — drop silently and advance by one to recover.
            ++i;
            continue;
        }

        // Peek the next codepoint to try a two-codepoint vocab key first
        // (right-greedy combining-mark match — never two-mark deep).
        const std::size_t n2 = utf8_len_at(ipa, i + n);
        if (n2 != 0) {
            std::string key2;
            key2.reserve(n + n2);
            key2.append(ipa.data() + i, n + n2);
            auto it2 = vocab_->find(key2);
            if (it2 != vocab_->end()) {
                out.push_back(static_cast<std::int32_t>(it2->second));
                i += n + n2;
                continue;
            }
        }

        // Single-codepoint lookup.
        std::string key1;
        key1.reserve(n);
        key1.append(ipa.data() + i, n);
        auto it1 = vocab_->find(key1);
        if (it1 != vocab_->end()) {
            out.push_back(static_cast<std::int32_t>(it1->second));
        }
        // Else: unknown codepoint, silently dropped.
        i += n;
    }

    return out;
}

}  // namespace brosoundml::g2p
