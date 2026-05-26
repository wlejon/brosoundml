# Morphology Spec

A small affix-stripping rules engine for `brosoundml::g2p::`. Backstops the
[lexicon](lexicon.md) by handling regular inflections (`-s`, `-ed`, `-ing`)
and possessive variants (`'s`, `s'`, trailing `'`) for words the lexicon
doesn't carry verbatim. The rules and the IPA "glue" they produce are a
direct port of `misaki/en.py`'s `stem_s` / `stem_ed` / `stem_ing` plus the
two possessive rewrites.

Sole consumer is the planned `Phonemizer` façade; the morphology class only
exposes the rules, not the orchestration. Phonemizer decides when to try the
lexicon vs morphology vs (eventually) a letter-rules fallback.

US English only. The British branches in misaki are dropped — brosoundml's
lexicon is `us_gold + us_silver`; British support is a future slice with its
own data sibling and its own `gb_*` variants of these rules.

## Repo posture: code only, no data

The class itself, its tests, and this spec live in `brosoundml`. No
artifacts; the only persistent input is the Lexicon, already covered by its
own slice. `Morphology` borrows a `const Lexicon&` and lives entirely in
memory.

## Upstream source

- File: `misaki/en.py` (commit `fba1236595f2d2bf21d414ba6e57d25256afada3`,
  same pin as the lexicon).
- Class: `G2P` (the Python class spanning the misaki English pipeline).
- Methods ported: `_s`, `stem_s`, `_ed`, `stem_ed`, `_ing`, `stem_ing`,
  and the two possessive branches inside `get_word`.
- Constants ported: `US_TAUS` (the IPA-character set that triggers the
  intervocalic /t/ → flap (`ɾ`) allophone in stem-final `-t` + `-ed`/`-ing`).

License: Apache 2.0 (misaki's). The C++ port inherits — no separate NOTICE
needed in the code repo since it's a re-expression of misaki's algorithm
rather than a redistribution of misaki's data. The `brosoundml-data/g2p/`
NOTICE already attributes misaki for the lexicon bytes, which covers the
upstream credit.

## API

```cpp
// include/brosoundml/g2p/morphology.h

#pragma once

#include <string>
#include <string_view>

namespace brosoundml::g2p {

class Lexicon;  // forward — full def in lexicon.h

class Morphology {
 public:
    // Borrows `lex`. `lex` must outlive this Morphology.
    explicit Morphology(const Lexicon& lex);

    // Try to phonemize `word` by applying the inflectional and possessive
    // rules below, in misaki's get_word() order. Returns the assembled IPA
    // (stem IPA + glue) on success, or an empty string if no rule fires.
    //
    // The returned string is owned — it is the concatenation of a stem's
    // IPA bytes (looked up via the borrowed Lexicon) with a small glue
    // suffix computed from the stem's final phoneme.
    //
    // `ptb_pos` is forwarded verbatim to Lexicon::lookup when probing
    // candidate stems.
    std::string try_phonemize(std::string_view word,
                              std::string_view ptb_pos = "") const;
};

}  // namespace brosoundml::g2p
```

Move-only, no pImpl needed — the class is a thin wrapper around a borrowed
`Lexicon` reference. Constructible only from a `Lexicon&`; no default
constructor. Errors throw `std::runtime_error` with the project's
`"brosoundml: g2p::Morphology::<where>: <reason>"` prefix.

## Rule order

```
try_phonemize(word, ptb_pos):
  1. Possessive s'  →  rewrite "foos'" as "foo's", lex-lookup; if hit, return.
  2. Trailing  '    →  rewrite "foo'" as "foo", lex-lookup; if hit, return.
  3. stem_s   → try -s   inflectional decomposition.
  4. stem_ed  → try -ed  inflectional decomposition.
  5. stem_ing → try -ing inflectional decomposition.
  6. Return "".
```

This matches misaki's `get_word` fallback chain (after special-case rules,
direct lexicon hit, and the case-fold pre-rule — none of which live here;
they belong to `Phonemizer`). The chain is short-circuit: first rule that
produces a non-empty stem IPA wins.

`try_phonemize` does **not** attempt a direct lexicon lookup of `word`
itself. Caller (`Phonemizer`) probes the lexicon first; morphology is the
miss path.

## Possessive rules

```
word.ends_with("s'") and lex.lookup(word[:-2] + "'s", ptb_pos) is non-empty:
    return lex.lookup(word[:-2] + "'s", ptb_pos)   // copy into std::string

word.ends_with("'") and lex.lookup(word[:-1], ptb_pos) is non-empty:
    return lex.lookup(word[:-1], ptb_pos)
```

These are pure rewrites — no IPA glue. The result is a copy of the stem's
lexicon IPA.

## `-s` rule (`stem_s`)

Eligibility:

```
len(word) >= 3 and word.ends_with("s")
```

Stem candidates, tried in order:

1. `word` does not end in `"ss"`, and `word[:-1]` is in the lexicon →
   stem = `word[:-1]`        (e.g. `"cats"` → `"cat"`)
2. (`word` ends in `"'s"`) **or** (`len(word) > 4` and ends in `"es"` and
   not ends in `"ies"`), and `word[:-2]` is in the lexicon →
   stem = `word[:-2]`        (e.g. `"girl's"` → `"girl"`, `"boxes"` → `"box"`)
3. `len(word) > 4`, ends in `"ies"`, and `word[:-3] + "y"` is in the
   lexicon → stem = `word[:-3] + "y"`     (e.g. `"babies"` → `"baby"`)

If no candidate succeeds, return `""`.

IPA glue (`_s` in misaki):

Inspect the **last UTF-8 codepoint** of the stem's IPA. Then:

| stem-final codepoint | append | reason (informal) |
|---|---|---|
| `p`, `t`, `k`, `f`, `θ` | `s` | voiceless non-sibilant → /s/ |
| `s`, `z`, `ʃ`, `ʒ`, `ʧ`, `ʤ` | `ᵻz` | sibilant → /ᵻz/ |
| anything else | `z` | voiced → /z/ |

(`θ` is U+03B8 Greek theta. The set members are single-codepoint each;
none of them are multi-codepoint.)

## `-ed` rule (`stem_ed`)

Eligibility:

```
len(word) >= 4 and word.ends_with("d")
```

Stem candidates:

1. `word` does not end in `"dd"`, and `word[:-1]` is in the lexicon →
   stem = `word[:-1]`        (e.g. `"loved"` → `"love"`)
2. `len(word) > 4`, ends in `"ed"`, does **not** end in `"eed"`, and
   `word[:-2]` is in the lexicon →
   stem = `word[:-2]`        (e.g. `"walked"` → `"walk"`)

IPA glue (`_ed`, US branch only):

Inspect the last UTF-8 codepoint(s) of the stem's IPA:

| stem-final codepoint | append / replace | result |
|---|---|---|
| in `{p, k, f, θ, ʃ, s, ʧ}` | append `t` | voiceless → /t/ |
| `d` | append `ᵻd` | dental → /ᵻd/ |
| anything else **except** `t` | append `d` | voiced → /d/ |
| `t`, len(stem_ipa) < 2 | append `ᵻd` | short stem ending /t/ → /ᵻd/ |
| `t`, second-last codepoint in `US_TAUS` | **replace** trailing `t` with `ɾᵻd` | intervocalic flap |
| `t`, second-last codepoint not in `US_TAUS` | append `ᵻd` | /ᵻd/ |

Where `US_TAUS = { A, I, O, W, Y, i, u, æ, ɑ, ə, ɛ, ɪ, ɹ, ʊ, ʌ }`.

The "replace" case is the only branch that mutates the stem IPA; all
others append. Care: "second-last codepoint" means the codepoint *before*
the final `t`, not a byte offset.

## `-ing` rule (`stem_ing`)

Eligibility:

```
len(word) >= 5 and word.ends_with("ing")
```

Stem candidates (regex syntax `\1` is a back-reference to the captured
character class):

1. `len(word) > 5` and `word[:-3]` is in the lexicon →
   stem = `word[:-3]`        (e.g. `"walking"` → `"walk"`)
2. `word[:-3] + "e"` is in the lexicon →
   stem = `word[:-3] + "e"`  (e.g. `"loving"` → `"love"`)
3. `len(word) > 5` and the tail matches the regex
   `r"([bcdgklmnprstvxz])\1ing$|cking$"`, and `word[:-4]` is in the
   lexicon → stem = `word[:-4]`     (e.g. `"running"` → `"run"`,
   `"trafficking"` → `"traffic"`)

The regex captures **doubled-consonant gemination** plus the special
`-cking` orthographic case (which is `ck` doubling on a final `c`). In
the C++ port the regex can be open-coded as: tail is `Cing` where the
fourth-from-last character equals the third-from-last (with the alphabet
restricted to the listed letters), **or** the tail is exactly `cking`.
No regex library is required.

IPA glue (`_ing`, US branch only):

| stem-final codepoint sequence | append / replace |
|---|---|
| `t` preceded by a codepoint in `US_TAUS`, len(stem_ipa) > 1 | replace trailing `t` with `ɾɪŋ` |
| anything else | append `ɪŋ` |

Same intervocalic-flap idea as `-ed`: stem-final `t` after a US_TAUS vowel
or `ɹ` collapses into the flap.

## UTF-8 codepoint helpers

The rules above repeatedly need "last codepoint of an IPA string" and
"second-to-last codepoint". The IPA blob is UTF-8 and the codepoints we
care about are all in the BMP, so the helper is small:

```cpp
// Returns the codepoint that ends `s` and the byte length of its encoding,
// or {0, 0} if s is empty.
std::pair<char32_t, size_t> last_codepoint(std::string_view s);
```

Decode by stepping back from the end until the byte is not a UTF-8
continuation byte (`(b & 0xC0) != 0x80`), then forward-decode that prefix.
Implementation lives in the .cpp; not exposed in the header. Membership
tests over `{p, t, k, …, ʃ, s, ʧ}` are codepoint comparisons against a
small `static constexpr` table.

## Behaviour on malformed input

- Empty `word` → return `""`.
- Lexicon miss on stem → return `""`. Do not surface this as an error.
- Stem IPA that decodes to an empty codepoint sequence (zero-length UTF-8
  string) — defensively, do not call into the codepoint helpers; treat as
  if the rule did not fire and return `""`. Should not occur with the real
  bin; defensive only.

## Tests

`tests/test_morphology.cpp`, gated on `BROSOUNDML_LEXICON_PATH` (or the
sibling default) using the same discovery shape as `test_lexicon.cpp`. SKIP
if the lexicon bin isn't available.

Required coverage:

1. **`-s` inflection.**
   - `"cats"` → stem `"cat"`, glue `s` (voiceless /t/ → /s/).
   - `"dogs"` → stem `"dog"`, glue `z`.
   - `"boxes"` → stem `"box"`, glue `ᵻz` (sibilant).
   - `"babies"` → stem `"baby"`, glue `z` (after rewrite, baby ends in a
     vowel-coded character).
   - `"girl's"` → stem `"girl"`, glue `z`.
   - `"glass"` (ends in `ss`) → rule rejects, returns `""`.
2. **`-ed` inflection.**
   - `"walked"` → stem `"walk"`, glue `t` (voiceless /k/).
   - `"loved"` → stem `"love"`, glue `d`.
   - `"needed"` → stem `"need"`, glue `ᵻd` (after /d/).
   - `"waited"` → stem `"wait"`, replace trailing `t` with `ɾᵻd`
     (intervocalic flap).
   - `"seed"` (ends in `eed`) → rule rejects.
3. **`-ing` inflection.**
   - `"walking"` → stem `"walk"`, glue `ɪŋ`.
   - `"loving"` → stem `"love"`, glue `ɪŋ`.
   - `"running"` → stem `"run"`, glue `ɪŋ` (gemination).
   - `"trafficking"` → stem `"traffic"`, glue `ɪŋ` (the `-cking` special).
   - `"waiting"` → stem `"wait"`, replace trailing `t` with `ɾɪŋ`.
4. **Possessive.**
   - `"dog's"` → lex-direct via the rewritten key (note: misaki's
     stem_s would also fire here if the lexicon doesn't carry the
     possessive form; the order is rule-1 trying `"dog's"` as a key
     match, then rule-3 trying `stem_s("dog's")`. Test the path the
     real lexicon actually exercises — pick a word the lexicon
     carries the bare form of but not the possessive).
   - `"dogs'"` → rewritten to `"dog's"` lookup if available.
5. **All rules miss.** A made-up word like `"xyzzyqing"` with a
   non-lexical stem returns `""`.
6. **Empty input.** `try_phonemize("", _)` returns `""`.

Expected IPA strings can be derived programmatically inside the test by
calling `Lexicon::lookup` on the predicted stem and concatenating the
glue, rather than hard-coding bytes. The test thereby verifies *the
algorithm*, not specific lexicon bytes — which keeps it robust when
misaki upstream is re-pinned.

The intervocalic-flap branches (`-ted` and `-ting` after a US_TAUS
codepoint) **do** need hard-coded expected substrings (the trailing
`ɾᵻd` / `ɾɪŋ`) because the test is verifying the replacement, not the
glue table; otherwise the assertion is circular.

## Build-out plan revisited

This is slice 2 of 5 in the G2P stack (numbering from
`docs/lexicon.md` § "Build-out plan"). After this lands:

3. **Special-case rules** — `"a"`/`"the"` vowel-conditioned allomorphy,
   `"used to"`, contractions, numbers, abbreviations. Runs at the
   sentence level; needs the POS tagger output for some branches.
4. **Kokoro phoneme-id adapter** — IPA → Kokoro vocab ids using
   `weights/kokoro/ids.txt`.
5. **`Phonemizer` façade** — orchestrates POS-tagger + special-cases +
   lexicon + morphology + adapter into the `sentence → vector<int32_t>`
   API that `Kokoro::synthesize` consumes.

Morphology is the last slice that is purely local (no sentence context,
no POS-driven orchestration). Slices 3-5 introduce sentence-level state.
