# Lexicon Spec

A packed English pronunciation dictionary for `brosoundml::g2p::`. First lookup
stage of the in-tree G2P stack: input is a word (and an optional Penn Treebank
POS tag for heteronym disambiguation), output is its IPA pronunciation as a
UTF-8 string. If the word is not in the lexicon the caller falls through to
the morphology slice (planned, not yet built).

Sole consumer is the Kokoro phoneme-id adapter (planned). The same lookup
table will also back any other Latin-script TTS we add later, so the API is
phoneme-set-agnostic — IPA strings flow through opaquely; the IPA → token-id
mapping is the adapter's job, not the lexicon's.

## Repo posture: code only, no data

The C++ loader, the Python builder, and the unit tests live here in
`brosoundml`. The packed binary itself (`lexicon_en_us.bin`) lives in the
sibling [`brosoundml-data`](https://huggingface.co/datasets/wlejon/brosoundml-data) repo under `g2p/`. brosoundml
ships no fetch logic, no manifest, no network code. `Lexicon::load(path)`
takes a file path; the caller materialises the file.

The builder script is one-time tooling — not part of the C++ build, not
called at runtime. You re-run it when misaki upstream changes or when you
want to bump the format version. The produced `.bin` is checked into
`brosoundml-data` so consumers don't need Python in their build environment.

## Upstream source

The lexicon is derived from [hexgrad/misaki](https://github.com/hexgrad/misaki)'s
US-English dictionaries:

- `misaki/data/us_gold.json` — hand-curated entries, ~90k words, includes the
  ~790 POS-conditioned heteronym entries.
- `misaki/data/us_silver.json` — automatically generated entries, ~93k words,
  all plain (no POS variants). Almost entirely additive to gold (~1 word of
  overlap).

License: **Apache 2.0** (misaki's). The packed `.bin` inherits Apache 2.0.
`brosoundml-data/g2p/` carries the NOTICE.

Pinned upstream commit: **`fba1236595f2d2bf21d414ba6e57d25256afada3`**.
The builder script hard-codes this hash in its fetch URL. Bumping the pin is
deliberate — update the script, re-run, commit the new `.bin`, bump the file
format version if anything changed structurally.

### Source schema (recap, for the builder)

`us_gold.json` is a flat JSON object. Values are either:

- a **plain string** (89,411 entries): the IPA pronunciation, UTF-8.
- a **POS-keyed object** (790 entries): `{"DEFAULT": "ipa", "VERB": "ipa", …}`.
  Observed keys, in descending frequency: `DEFAULT` (790), `NOUN` (425),
  `VERB` (272), `ADJ` (63), `None` (32 — a stringified Python `None`, treat
  as no variant), `VBD` (4), `ADV` (3), `VBN` (3), `VBP` (3), `DT` (1).
  A `null` value under a POS key (e.g. `"NOUN": null` on acronyms) means
  "no special pronunciation for this POS" — drop the variant; the entry's
  `DEFAULT` covers it.

`us_silver.json` is a flat JSON object, all plain string values.

Keys: mixed case. Lowercase-first keys dominate (~81k), uppercase-first cover
proper nouns and acronyms (~9k). Both casings must be preserved in the bin
verbatim — the lookup algorithm relies on exact-case entries being present
for acronyms like `"AA"` or `"NASA"`.

IPA values use UTF-8 IPA plus a few ASCII capital letters (`A I O W Y`) which
encode Kokoro-style stress/diphthong markers. Treat the byte sequence as
opaque — the lexicon doesn't interpret or validate it.

## API

```cpp
// include/brosoundml/g2p/lexicon.h

#pragma once

#include <memory>
#include <string>
#include <string_view>

namespace brosoundml::g2p {

class Lexicon {
 public:
    // Load the packed binary from disk. Throws std::runtime_error on
    // missing file, bad magic, or unsupported version.
    static Lexicon load(const std::string& path);

    // Look up `word`, optionally biased by a PTB tag for heteronym
    // selection. Returns an empty string_view if the word is not present
    // (after case fallback). See "Lookup algorithm" below for the full
    // resolution order.
    //
    // The returned view is stable for the lifetime of this Lexicon and
    // remains valid across add_override() calls.
    std::string_view lookup(std::string_view word,
                            std::string_view ptb_pos = "") const;

    // Install a runtime override. Subsequent lookup(word, _) returns
    // `ipa` regardless of POS. The override matches by exact case-folded
    // word (ASCII tolower), so add_override("Foo", …) is the same as
    // add_override("FOO", …).
    //
    // Overrides are in-memory only; they are not written back to disk.
    // Re-loading the Lexicon drops them.
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
```

Errors throw `std::runtime_error` with `"brosoundml: g2p::Lexicon::<where>: <reason>"`,
matching the project convention.

## Lookup algorithm

```
lookup(word, ptb_pos):
  1. If overrides contains tolower_ascii(word): return overrides[…].
  2. Try exact-case bin lookup of `word`.
  3. If not found and word has any ASCII uppercase letter: lowercase it
     (ASCII tolower only, do not touch UTF-8 bytes) and try again.
  4. If still not found: return "".
  5. We have a hit. If the entry is a plain string: return it.
  6. The entry is a variant table. Map `ptb_pos` to a misaki tag:
       NN, NNS, NNP, NNPS                  → NOUN
       VB, VBZ, VBP, VBG, VBD, VBN         → exact PTB match first, then VERB
       JJ, JJR, JJS                        → ADJ
       RB, RBR, RBS                        → ADV
       DT                                  → DT
       (anything else, or "")              → DEFAULT
     Try in order: exact PTB tag → mapped misaki tag → "DEFAULT". The
     mapping table is fixed and hard-coded in the loader.
  7. Return the IPA for the first variant key that hits.
```

`DEFAULT` is always present on variant entries, so step 6 never falls off the
end. The builder validates this and refuses to write a bin that contains a
variant entry missing `DEFAULT`.

### Why not a separate ALLCAPS / Title-case lookup pass?

The misaki source data already encodes those cases as their own entries
(e.g. `"AA"` has its own entry distinct from `"aa"`). The exact-case-first
pass picks them up without special logic. The simple two-pass exact→lower
fallback matches misaki's own dictionary lookup semantics.

## File format

Sorted-key flat array. Simple, mmap-friendly, no compression on the first
version — optimize later only if the size budget is missed badly enough to
matter. All multi-byte integers are little-endian.

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

Loader rejects: wrong magic, `version != 1`, blob offsets that overflow the
file, index size mismatched against `entry_count × 16`.

### Index entry (16 bytes)

| Field | Type | Notes |
|---|---|---|
| `key_off` | u32 | byte offset into key_blob |
| `key_len` | u16 | byte length of key (keys are short — observed max ~30) |
| `flags` | u8 | bit 0: 1 = variant table, 0 = plain IPA |
| `variant_count` | u8 | number of variants if flags.bit0 set; else 0 |
| `val_off` | u32 | byte offset into val_blob |
| `val_len` | u32 | byte length of the value record |

Index is sorted bytewise by `(key_blob[key_off : key_off+key_len])` ascending,
so lookup is a binary search over the index.

### Value records

For a plain entry (`flags.bit0 == 0`), `val_blob[val_off : val_off+val_len]`
is the raw IPA UTF-8 bytes.

For a variant entry (`flags.bit0 == 1`), the block at `val_off` contains
`variant_count` records back-to-back. Each record is:

| Field | Type | Notes |
|---|---|---|
| `tag_id` | u8 | misaki tag id, see table below |
| `ipa_len` | u16 | byte length of the IPA string |
| `ipa` | u8[ipa_len] | UTF-8 IPA |

`tag_id` table (closed set; builder rejects unknown tags):

| id | tag |
|---|---|
| 0 | `DEFAULT` |
| 1 | `NOUN` |
| 2 | `VERB` |
| 3 | `ADJ` |
| 4 | `ADV` |
| 5 | `VBD` |
| 6 | `VBN` |
| 7 | `VBP` |
| 8 | `DT` |

Variant records inside a block are stored in `tag_id` ascending order. The
`DEFAULT` record (id 0) is guaranteed to be the first record in every
variant block.

### Size budget

Observed when packing gold ∪ silver (~183k entries) without compression:

- index:   183k × 16 ≈ 2.9 MB
- key_blob: ~1.6 MB (average key ~8 bytes)
- val_blob: ~2.4 MB
- total:   ~6.9 MB

That overshoots the 2-3 MB target stated in the original brief. v1 ships at
this size — the loader is mmap-able and the data sibling repo absorbs the few
extra MB cleanly. v2 can introduce front-coded keys + varint offsets if size
becomes a deployment problem; the magic + version header makes that a clean
break.

If even v1 needs to come in under 4 MB to please the brief, the cheapest cut
is to **drop silver entries that already lose to a morphology fallback**.
That decision lives with the morphology slice, not here — skip it for now.

## Builder pipeline

`tools/build_lexicon.py` is a single-file Python 3.10+ script. No third-party
dependencies; standard library only.

Pseudocode:

```python
GOLD_URL   = f"https://raw.githubusercontent.com/hexgrad/misaki/{COMMIT}/misaki/data/us_gold.json"
SILVER_URL = f"https://raw.githubusercontent.com/hexgrad/misaki/{COMMIT}/misaki/data/us_silver.json"
COMMIT     = "fba1236595f2d2bf21d414ba6e57d25256afada3"

TAG_IDS = {"DEFAULT":0,"NOUN":1,"VERB":2,"ADJ":3,"ADV":4,"VBD":5,"VBN":6,"VBP":7,"DT":8}

def main():
    gold   = fetch_json(GOLD_URL)
    silver = fetch_json(SILVER_URL)

    entries: dict[str, str | dict[str, str]] = {}
    # Silver first, gold wins on conflict
    for k, v in silver.items(): entries[k] = v
    for k, v in gold.items():   entries[k] = v

    # Sanitize variant entries: drop None values, drop unknown POS keys (warn),
    # require DEFAULT to be present. Convert "None" key (stringified Python None)
    # to nothing (treat as variant absence).
    cleaned = sanitize(entries)

    # Sort bytewise by UTF-8 key
    items = sorted(cleaned.items(), key=lambda kv: kv[0].encode("utf-8"))

    # Two-pass write: build key_blob + val_blob + index records, then write
    # the file with header pointing at them.
    write_bin("lexicon_en_us.bin", items)

if __name__ == "__main__":
    main()
```

CLI:

```sh
python tools/build_lexicon.py \
       --out ../brosoundml-data/g2p/lexicon_en_us.bin
```

Default `--out` is `../brosoundml-data/g2p/lexicon_en_us.bin` relative to the
brosoundml repo root, matching the resolution convention.

The script prints a summary on exit: entry count, variant count, file size,
and the misaki commit it pulled.

## Tests

`tests/test_lexicon.cpp` is gated on the same env var pattern as the POS
tagger — `BROSOUNDML_LEXICON_PATH`, with a fallback to
`../brosoundml-data/g2p/lexicon_en_us.bin`. If neither resolves to an existing
file the test is skipped (returns SKIPPED rather than failing), so CI without
the data sibling does not break.

Required coverage:

1. **Header validation**: corrupt magic / wrong version → throws.
2. **Plain lookup**: a handful of canonical words (`"hello"`, `"world"`,
   `"the"`) → expected IPA exact-match.
3. **Case fallback**: `lookup("HELLO")` matches `"hello"` if no `"HELLO"`
   entry exists; `lookup("AA")` returns the all-caps acronym entry (not the
   lowercase one).
4. **Heteronym selection**:
   - `lookup("record", "NN")` → `"ɹˈɛkəɹd"` (default / NOUN).
   - `lookup("record", "VB")` → `"ɹəkˈɔɹd"` (VERB variant).
   - `lookup("record", "")` → DEFAULT.
   - `lookup("read", "VBD")` → `"ɹˈɛd"` (exact PTB match wins over generic
     VERB mapping).
   - `lookup("read", "JJ")` → `"ɹˈɛd"` (ADJ variant).
5. **PTB mapping completeness**: spot-check that each PTB family
   (NN/NNS/NNP/NNPS, VB/VBZ/VBG, JJ/JJR/JJS, RB/RBR/RBS) maps through to the
   right misaki tag.
6. **Override priority**: `add_override("hello", "x"); lookup("hello", _)`
   returns `"x"`, regardless of POS. Override survives further `add_override`
   calls on other words (returned `string_view` stability).
7. **Miss path**: `lookup("xyzzy", _)` returns an empty `string_view`.

A small hand-authored golden file is **not** needed — the misaki entries
themselves are stable enough to assert against directly, and the bin is a
deterministic pack of public data, so re-running the build reproduces it
bit-for-bit.

## Build-out plan

This is **slice 1** of the in-tree English G2P stack. Subsequent slices:

1. **Lexicon** (this doc) — packed misaki dictionary, byte-level lookup.
2. **Morphology fallback** — split unknown words at known affix boundaries,
   look up the stem, glue the affix's IPA. Backstops words missing from the
   lexicon.
3. **Special-case rules** — `"a"`/`"the"` vowel-conditioned allomorphy,
   `"used to"` collocation, contractions, numbers, abbreviations. Runs
   before the lexicon and can short-circuit it.
4. **Kokoro phoneme-id adapter** — IPA string → Kokoro vocab token ids,
   using `weights/kokoro/ids.txt`.
5. **`Phonemizer` façade** — orchestrates PosTagger + Lexicon + morphology +
   special cases, exposes the `sentence → vector<int32_t>` entry point that
   `Kokoro::synthesize()` wants on the other end.

Land in order. Each slice is independently testable.
