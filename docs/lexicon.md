# Lexicon

A packed English pronunciation dictionary for `brosoundml::g2p::`. It is the
first lookup stage of the in-tree G2P stack: input is a word (and an optional
Penn Treebank POS tag for heteronym disambiguation), output is its IPA
pronunciation as a UTF-8 string. A miss returns an empty `string_view`, and the
[Phonemizer](phonemizer.md) falls through to the [morphology](morphology.md)
slice. The table is phoneme-set-agnostic — IPA strings flow through opaquely; the
IPA → Kokoro-token mapping is the [adapter](phonemizer.md)'s job, not the
lexicon's.

Public surface: `include/brosoundml/g2p/lexicon.h` (`Lexicon`).

## Weights and data

The C++ loader, the Python builder, and the unit tests live in this repo. The
packed binary `lexicon_en_us.bin` lives in the sibling
[`brosoundml-data`](https://huggingface.co/datasets/wlejon/brosoundml-data) repo
under `g2p/`. brosoundml ships no fetch logic, no manifest, no network code;
`Lexicon::load(path)` takes a file path and the caller materialises the file.
The builder script is one-time offline tooling — re-run when misaki upstream
changes or to bump the format version — and the produced `.bin` is checked into
`brosoundml-data` so consumers need no Python in their build.

## Upstream source

The lexicon is derived from
[hexgrad/misaki](https://github.com/hexgrad/misaki)'s US-English dictionaries:

- `misaki/data/us_gold.json` — hand-curated, ~90k words, including the ~790
  POS-conditioned heteronym entries.
- `misaki/data/us_silver.json` — automatically generated, ~93k words, all plain
  (no POS variants). Almost entirely additive to gold (~1 word of overlap).

License: **Apache 2.0** (misaki's); the packed `.bin` inherits it, and
`brosoundml-data/g2p/` carries the NOTICE. The builder pins upstream commit
`fba1236595f2d2bf21d414ba6e57d25256afada3` in its fetch URL; bumping the pin is
a deliberate act (update the script, re-run, commit the new `.bin`, bump the
format version if anything changed structurally).

### Source schema

`us_gold.json` is a flat JSON object whose values are either a **plain string**
(89,411 entries — the IPA, UTF-8) or a **POS-keyed object** (790 entries —
`{"DEFAULT": "ipa", "VERB": "ipa", …}`). Observed POS keys, descending by
frequency: `DEFAULT` (790), `NOUN` (425), `VERB` (272), `ADJ` (63), `None` (32 —
a stringified Python `None`, treated as no variant), `VBD` (4), `ADV` (3), `VBN`
(3), `VBP` (3), `DT` (1). A `null` value under a POS key (e.g. `"NOUN": null` on
acronyms) means "no special pronunciation for this POS" — the variant is dropped
and the entry's `DEFAULT` covers it. `us_silver.json` is a flat object of plain
string values.

Keys are mixed case (lowercase-first dominate, ~81k; uppercase-first cover proper
nouns and acronyms, ~9k). Both casings are preserved verbatim — the lookup relies
on exact-case entries for acronyms like `"AA"` or `"NASA"`. IPA values are UTF-8
IPA plus a few ASCII capitals (`A I O W Y`) encoding Kokoro-style
stress/diphthong markers; the byte sequence is treated as opaque, never
interpreted or validated.

## Lookup algorithm

```
lookup(word, ptb_pos):
  1. If overrides contains tolower_ascii(word): return overrides[…].
  2. Try exact-case bin lookup of `word`.
  3. If not found and word has any ASCII uppercase letter: lowercase it
     (ASCII tolower only, do not touch UTF-8 bytes) and try again.
  4. If still not found: return "".
  5. On a hit with a plain string entry: return it.
  6. On a hit with a variant table, map `ptb_pos` to a misaki tag:
       NN, NNS, NNP, NNPS                  → NOUN
       VB, VBZ, VBP, VBG, VBD, VBN         → exact PTB match first, then VERB
       JJ, JJR, JJS                        → ADJ
       RB, RBR, RBS                        → ADV
       DT                                  → DT
       (anything else, or "")              → DEFAULT
     Try in order: exact PTB tag → mapped misaki tag → "DEFAULT".
  7. Return the IPA for the first variant key that hits.
```

`DEFAULT` is always present on variant entries, so step 6 never falls off the
end — the builder refuses to write a variant entry missing `DEFAULT`. There is no
separate ALLCAPS / Title-case pass: the misaki source already encodes those as
their own entries (`"AA"` distinct from `"aa"`), which the exact-case-first pass
picks up, matching misaki's own dictionary lookup semantics.

A runtime `add_override(word, ipa)` installs an in-memory override matched by
ASCII-folded key; overrides are not written to disk and are dropped on reload.
`string_view`s handed out by `lookup` stay valid across `add_override` calls
(overrides are stored in pointer-stable map nodes).

## File format

A sorted-key flat array — simple, mmap-friendly, uncompressed. All multi-byte
integers are little-endian.

```
+------------------+
|     header       |   64 bytes
+------------------+
|     index        |   entry_count × 16 bytes, sorted bytewise by key
+------------------+
|    key_blob      |   concatenation of key bytes, no separators
+------------------+
|    val_blob      |   concatenation of value records, no separators
+------------------+
```

### Header (64 bytes)

| Field | Type | Notes |
|---|---|---|
| `magic` | char[4] | `"BSLX"` |
| `version` | u32 | format version, starts at `1` |
| `entry_count` | u32 | number of index entries |
| `flags` | u32 | reserved, `0` in v1 |
| `key_blob_off` | u64 | byte offset of key_blob |
| `key_blob_len` | u64 | byte length of key_blob |
| `val_blob_off` | u64 | byte offset of val_blob |
| `val_blob_len` | u64 | byte length of val_blob |
| `_reserved` | u8[16] | zero |

The loader rejects wrong magic, `version != 1`, blob offsets that overflow the
file, and an index size mismatched against `entry_count × 16`.

### Index entry (16 bytes)

| Field | Type | Notes |
|---|---|---|
| `key_off` | u32 | byte offset into key_blob |
| `key_len` | u16 | byte length of key (observed max ~30) |
| `flags` | u8 | bit 0: 1 = variant table, 0 = plain IPA |
| `variant_count` | u8 | number of variants if flags.bit0 set; else 0 |
| `val_off` | u32 | byte offset into val_blob |
| `val_len` | u32 | byte length of the value record |

The index is sorted bytewise by its key, so lookup is a binary search.

### Value records

For a plain entry (`flags.bit0 == 0`), the value blob slice is the raw IPA UTF-8
bytes. For a variant entry (`flags.bit0 == 1`), the block holds `variant_count`
records back-to-back, each:

| Field | Type | Notes |
|---|---|---|
| `tag_id` | u8 | misaki tag id (table below) |
| `ipa_len` | u16 | byte length of the IPA string |
| `ipa` | u8[ipa_len] | UTF-8 IPA |

`tag_id` table (closed set; builder rejects unknown tags):

| id | tag | | id | tag |
|---|---|---|---|---|
| 0 | `DEFAULT` | | 5 | `VBD` |
| 1 | `NOUN` | | 6 | `VBN` |
| 2 | `VERB` | | 7 | `VBP` |
| 3 | `ADJ` | | 8 | `DT` |
| 4 | `ADV` | | | |

Variant records inside a block are stored in `tag_id` ascending order, so the
`DEFAULT` record (id 0) is always first.

### Size

Packing gold ∪ silver (~183k entries) uncompressed: index ~2.9 MB, key_blob
~1.6 MB, val_blob ~2.4 MB, total ~6.9 MB. The bin is mmap-able and the data
sibling absorbs the size cleanly. The magic + version header leaves room for a
front-coded / varint-offset v2 as a clean break if size ever becomes a
deployment problem.

## Builder

`tools/build_lexicon.py` is a single-file Python 3.10+ script, standard library
only. It fetches gold + silver at the pinned commit, merges them (silver first,
gold wins on conflict), sanitises variant entries (drops `null` / `None` /
unknown POS keys, requires `DEFAULT`), sorts bytewise by UTF-8 key, and writes
the bin with a two-pass key_blob/val_blob/index layout. CLI:

```sh
python tools/build_lexicon.py --out ../brosoundml-data/g2p/lexicon_en_us.bin
```

The default `--out` matches the resolution convention. The script prints a
summary on exit: entry count, variant count, file size, and the misaki commit.

## API

```cpp
namespace brosoundml::g2p {

class Lexicon {
 public:
    static Lexicon load(const std::string& path);

    // word (+ optional PTB tag) → IPA, or empty string_view on a miss after
    // ASCII case fallback. Returned view is stable for the Lexicon's lifetime.
    std::string_view lookup(std::string_view word,
                            std::string_view ptb_pos = "") const;

    // In-memory override, matched by ASCII-folded key; not persisted.
    void add_override(std::string_view word, std::string ipa);
};

}  // namespace brosoundml::g2p
```

Errors throw `std::runtime_error` with a
`"brosoundml: g2p::Lexicon::<where>: <reason>"` message.

`tests/test_lexicon.cpp` is gated on `BROSOUNDML_LEXICON_PATH` (falling back to
`../brosoundml-data/g2p/lexicon_en_us.bin`) and SKIPs if neither resolves. It
covers header validation, plain lookup, case fallback (`HELLO`→`hello`, `AA`
distinct), heteronym selection (`record` NN vs VB, `read` VBD/JJ exact-match),
PTB→misaki mapping, override priority and view stability, and the miss path. The
bin is a deterministic pack of public data, so no separate golden file is
needed.
