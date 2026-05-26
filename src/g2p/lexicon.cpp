#include "brosoundml/g2p/lexicon.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brosoundml::g2p {

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: g2p::Lexicon::" + where + ": " + msg);
}

// ─── Header / index record layout (mirrors docs/lexicon.md § "File format") ──

constexpr std::size_t kHeaderSize     = 64;
constexpr std::size_t kIndexRecordLen = 16;

struct IndexEntry {
    std::uint32_t key_off;
    std::uint16_t key_len;
    std::uint8_t  flags;          // bit 0: variant table
    std::uint8_t  variant_count;
    std::uint32_t val_off;
    std::uint32_t val_len;
};

// ─── Misaki tag ids (closed set, mirrors builder + spec) ──────────────────

enum class MisakiTag : std::uint8_t {
    DEFAULT = 0,
    NOUN    = 1,
    VERB    = 2,
    ADJ     = 3,
    ADV     = 4,
    VBD     = 5,
    VBN     = 6,
    VBP     = 7,
    DT      = 8,
    NONE    = 255,   // sentinel: no exact PTB → misaki id match
};

// PTB → exact misaki id when one exists. Used as the "exact PTB tag first"
// step in variant selection; returns NONE if no direct match.
MisakiTag ptb_exact_id(std::string_view ptb) {
    if (ptb == "VBD") return MisakiTag::VBD;
    if (ptb == "VBN") return MisakiTag::VBN;
    if (ptb == "VBP") return MisakiTag::VBP;
    if (ptb == "DT")  return MisakiTag::DT;
    return MisakiTag::NONE;
}

// PTB → mapped misaki bucket. The spec's mapping table is canonical.
MisakiTag ptb_to_misaki(std::string_view ptb) {
    if (ptb.empty())                                                return MisakiTag::DEFAULT;
    if (ptb == "NN" || ptb == "NNS" || ptb == "NNP" || ptb == "NNPS") return MisakiTag::NOUN;
    if (ptb == "VB" || ptb == "VBZ" || ptb == "VBP" ||
        ptb == "VBG" || ptb == "VBD" || ptb == "VBN")               return MisakiTag::VERB;
    if (ptb == "JJ" || ptb == "JJR" || ptb == "JJS")                return MisakiTag::ADJ;
    if (ptb == "RB" || ptb == "RBR" || ptb == "RBS")                return MisakiTag::ADV;
    if (ptb == "DT")                                                return MisakiTag::DT;
    return MisakiTag::DEFAULT;
}

// ASCII-only case folding. We deliberately do not touch bytes ≥ 0x80 — the
// few non-ASCII keys are proper-noun characters whose lowercase form (if
// any) is not encoded the same way in misaki's source, and we'd just
// produce a key that misses anyway. UTF-8 IPA bytes never appear in keys
// in practice but the rule is the same.
inline unsigned char ascii_tolower(unsigned char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<unsigned char>(c + 32) : c;
}

bool has_ascii_upper(std::string_view s) {
    for (unsigned char c : s) if (c >= 'A' && c <= 'Z') return true;
    return false;
}

void ascii_lower_inplace(std::string& s) {
    for (auto& c : s) {
        const unsigned char u = static_cast<unsigned char>(c);
        c = static_cast<char>(ascii_tolower(u));
    }
}

// Heterogeneous lookup for std::map<std::string, …, std::less<>>: lets us
// probe with a std::string_view key without allocating a std::string.

// ─── Little-endian readers ────────────────────────────────────────────────

inline std::uint16_t rd_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) |
           (static_cast<std::uint16_t>(p[1]) << 8);
}
inline std::uint32_t rd_u32(const std::uint8_t* p) {
    return  static_cast<std::uint32_t>(p[0])         |
           (static_cast<std::uint32_t>(p[1]) << 8)   |
           (static_cast<std::uint32_t>(p[2]) << 16)  |
           (static_cast<std::uint32_t>(p[3]) << 24);
}
inline std::uint64_t rd_u64(const std::uint8_t* p) {
    return  static_cast<std::uint64_t>(p[0])         |
           (static_cast<std::uint64_t>(p[1]) << 8)   |
           (static_cast<std::uint64_t>(p[2]) << 16)  |
           (static_cast<std::uint64_t>(p[3]) << 24)  |
           (static_cast<std::uint64_t>(p[4]) << 32)  |
           (static_cast<std::uint64_t>(p[5]) << 40)  |
           (static_cast<std::uint64_t>(p[6]) << 48)  |
           (static_cast<std::uint64_t>(p[7]) << 56);
}

}  // namespace

// ─── Impl ─────────────────────────────────────────────────────────────────

struct Lexicon::Impl {
    std::vector<std::uint8_t> buf;
    std::vector<IndexEntry>   index;
    const std::uint8_t*       key_blob = nullptr;
    std::size_t               key_blob_len = 0;
    const std::uint8_t*       val_blob = nullptr;
    std::size_t               val_blob_len = 0;

    // Overrides: ASCII-folded key → IPA. std::map (not unordered_map) so the
    // string_views we return remain valid across further insertions.
    std::map<std::string, std::string, std::less<>> overrides;

    std::string_view key_of(const IndexEntry& e) const {
        return std::string_view(
            reinterpret_cast<const char*>(key_blob + e.key_off), e.key_len);
    }

    // Binary search the sorted index for an exact-case key.
    const IndexEntry* find_exact(std::string_view key) const {
        auto it = std::lower_bound(
            index.begin(), index.end(), key,
            [this](const IndexEntry& a, std::string_view b) {
                const std::string_view ka = key_of(a);
                const std::size_t n = std::min(ka.size(), b.size());
                const int c = n ? std::memcmp(ka.data(), b.data(), n) : 0;
                if (c != 0) return c < 0;
                return ka.size() < b.size();
            });
        if (it == index.end()) return nullptr;
        if (key_of(*it) != key) return nullptr;
        return &*it;
    }

    // Extract IPA for a hit, applying variant selection if needed.
    std::string_view ipa_for(const IndexEntry& e, std::string_view ptb) const {
        const bool is_variant = (e.flags & 0x01u) != 0;
        if (!is_variant) {
            return std::string_view(
                reinterpret_cast<const char*>(val_blob + e.val_off), e.val_len);
        }

        // Walk the variant block (sorted ascending by tag_id; DEFAULT first).
        // Collect (tag_id, ipa) pairs, then pick by priority.
        struct V { std::uint8_t id; std::string_view ipa; };
        V vs[16];
        const int n = e.variant_count;
        const std::uint8_t* p = val_blob + e.val_off;
        const std::uint8_t* end = p + e.val_len;
        int read = 0;
        for (int i = 0; i < n; ++i) {
            if (p + 3 > end) return {};
            const std::uint8_t id = p[0];
            const std::uint16_t ipa_len = rd_u16(p + 1);
            p += 3;
            if (p + ipa_len > end) return {};
            if (read < 16) {
                vs[read++] = V{id, std::string_view(
                    reinterpret_cast<const char*>(p), ipa_len)};
            }
            p += ipa_len;
        }

        auto find_id = [&](MisakiTag t) -> std::string_view {
            const std::uint8_t want = static_cast<std::uint8_t>(t);
            for (int i = 0; i < read; ++i) {
                if (vs[i].id == want) return vs[i].ipa;
            }
            return {};
        };

        // 1. Exact PTB → misaki id (only for VBD/VBN/VBP/DT).
        const MisakiTag exact = ptb_exact_id(ptb);
        if (exact != MisakiTag::NONE) {
            const auto r = find_id(exact);
            if (!r.empty()) return r;
        }
        // 2. Mapped misaki bucket.
        const MisakiTag mapped = ptb_to_misaki(ptb);
        if (mapped != MisakiTag::DEFAULT) {
            const auto r = find_id(mapped);
            if (!r.empty()) return r;
        }
        // 3. DEFAULT (always present per builder invariant).
        const auto def = find_id(MisakiTag::DEFAULT);
        if (!def.empty()) return def;
        // Defensive: if DEFAULT is missing (shouldn't happen), fall back
        // to the first record.
        return read ? vs[0].ipa : std::string_view{};
    }
};

// ─── Public API ────────────────────────────────────────────────────────────

Lexicon::Lexicon() : impl_(std::make_unique<Impl>()) {}
Lexicon::Lexicon(Lexicon&&) noexcept = default;
Lexicon& Lexicon::operator=(Lexicon&&) noexcept = default;
Lexicon::~Lexicon() = default;

Lexicon Lexicon::load(const std::string& path) {
    Lexicon lx;
    auto& I = *lx.impl_;

    std::ifstream f(path, std::ios::binary);
    if (!f) fail("load", "cannot open file '" + path + "'");
    f.seekg(0, std::ios::end);
    const std::streamoff sz = f.tellg();
    if (sz < static_cast<std::streamoff>(kHeaderSize)) {
        fail("load", "file too small to contain header");
    }
    f.seekg(0, std::ios::beg);
    I.buf.resize(static_cast<std::size_t>(sz));
    if (!f.read(reinterpret_cast<char*>(I.buf.data()), sz)) {
        fail("load", "short read on '" + path + "'");
    }

    const std::uint8_t* h = I.buf.data();
    if (std::memcmp(h, "BSLX", 4) != 0) {
        fail("load", "bad magic (expected BSLX)");
    }
    const std::uint32_t version     = rd_u32(h + 4);
    if (version != 1u) {
        fail("load", "unsupported version " + std::to_string(version));
    }
    const std::uint32_t entry_count = rd_u32(h + 8);
    // flags at h+12, reserved.
    const std::uint64_t key_off     = rd_u64(h + 16);
    const std::uint64_t key_len     = rd_u64(h + 24);
    const std::uint64_t val_off     = rd_u64(h + 32);
    const std::uint64_t val_len     = rd_u64(h + 40);

    const std::uint64_t index_off   = kHeaderSize;
    const std::uint64_t index_bytes = static_cast<std::uint64_t>(entry_count) * kIndexRecordLen;

    const std::uint64_t file_sz = I.buf.size();
    auto in_bounds = [&](std::uint64_t off, std::uint64_t len) {
        return off <= file_sz && len <= file_sz - off;
    };
    if (!in_bounds(index_off, index_bytes)) fail("load", "index extends past EOF");
    if (!in_bounds(key_off,   key_len))     fail("load", "key_blob extends past EOF");
    if (!in_bounds(val_off,   val_len))     fail("load", "val_blob extends past EOF");
    if (index_off + index_bytes > key_off)  fail("load", "index/key_blob overlap");

    I.key_blob     = I.buf.data() + key_off;
    I.key_blob_len = static_cast<std::size_t>(key_len);
    I.val_blob     = I.buf.data() + val_off;
    I.val_blob_len = static_cast<std::size_t>(val_len);

    I.index.reserve(entry_count);
    const std::uint8_t* ip = I.buf.data() + index_off;
    for (std::uint32_t i = 0; i < entry_count; ++i, ip += kIndexRecordLen) {
        IndexEntry e;
        e.key_off       = rd_u32(ip + 0);
        e.key_len       = rd_u16(ip + 4);
        e.flags         = ip[6];
        e.variant_count = ip[7];
        e.val_off       = rd_u32(ip + 8);
        e.val_len       = rd_u32(ip + 12);
        if (e.key_off + e.key_len > I.key_blob_len) {
            fail("load", "index entry " + std::to_string(i) +
                 " key range outside key_blob");
        }
        if (static_cast<std::uint64_t>(e.val_off) + e.val_len > I.val_blob_len) {
            fail("load", "index entry " + std::to_string(i) +
                 " val range outside val_blob");
        }
        I.index.push_back(e);
    }
    return lx;
}

std::string_view Lexicon::lookup(std::string_view word,
                                 std::string_view ptb_pos) const {
    const auto& I = *impl_;

    // 1. Override (ASCII-folded match).
    std::string folded;
    folded.reserve(word.size());
    for (unsigned char c : word) folded.push_back(static_cast<char>(ascii_tolower(c)));
    if (auto it = I.overrides.find(std::string_view(folded));
        it != I.overrides.end()) {
        return std::string_view(it->second);
    }

    // 2. Exact-case bin lookup.
    if (const IndexEntry* e = I.find_exact(word)) {
        return I.ipa_for(*e, ptb_pos);
    }

    // 3. Case fallback only if there was any ASCII upper in the original.
    if (has_ascii_upper(word)) {
        if (const IndexEntry* e = I.find_exact(folded)) {
            return I.ipa_for(*e, ptb_pos);
        }
    }

    // 4. Miss.
    return {};
}

void Lexicon::add_override(std::string_view word, std::string ipa) {
    auto& I = *impl_;
    std::string folded(word);
    ascii_lower_inplace(folded);
    // operator[]/insert_or_assign on std::map preserves the storage of
    // existing nodes; only the matching node (if any) is overwritten.
    I.overrides.insert_or_assign(std::move(folded), std::move(ipa));
}

}  // namespace brosoundml::g2p
