# Morphology

An affix-stripping rules engine for `brosoundml::g2p::`. It backstops the
[lexicon](lexicon.md) by handling regular inflections (`-s`, `-ed`, `-ing`) and
possessive variants (`'s`, `s'`, trailing `'`) for words the lexicon doesn't
carry verbatim. The rules and the IPA "glue" they produce are a direct port of
`misaki/en.py`'s `stem_s` / `stem_ed` / `stem_ing` plus the two possessive
rewrites. US English only — the British branches in misaki are dropped, since
brosoundml's lexicon is `us_gold + us_silver`.

The [Phonemizer](phonemizer.md) is the only consumer. Morphology exposes only the
rule chain, not orchestration: it borrows a `const Lexicon&`, lives entirely in
memory, and `try_phonemize` does **not** attempt a direct lexicon hit on `word`
itself — the Phonemizer probes the lexicon first and morphology is the miss path.

Public surface: `include/brosoundml/g2p/morphology.h` (`Morphology`).

## Upstream source

- File: `misaki/en.py`, commit `fba1236595f2d2bf21d414ba6e57d25256afada3` (same
  pin as the lexicon).
- Methods ported: `_s`, `stem_s`, `_ed`, `stem_ed`, `_ing`, `stem_ing`, and the
  two possessive branches inside `get_word`.
- Constants ported: `US_TAUS`, the IPA-character set that triggers the
  intervocalic /t/ → flap (`ɾ`) allophone in stem-final `-t` + `-ed`/`-ing`.

License: Apache 2.0 (misaki's), inherited by the C++ port. No separate NOTICE is
needed in the code repo — the port is a re-expression of misaki's algorithm
rather than a redistribution of its data, and `brosoundml-data/g2p/`'s NOTICE
already attributes misaki for the lexicon bytes.

## Rule order

```
try_phonemize(word, ptb_pos):
  1. Possessive s'  →  rewrite "foos'" as "foo's", lex-lookup; if hit, return.
  2. Trailing  '    →  rewrite "foo'" as "foo",    lex-lookup; if hit, return.
  3. stem_s   → try -s   inflectional decomposition.
  4. stem_ed  → try -ed  inflectional decomposition.
  5. stem_ing → try -ing inflectional decomposition.
  6. Return "".
```

This matches misaki's `get_word` fallback chain after special-case rules, a
direct lexicon hit, and the case-fold pre-rule — none of which live here; they
belong to the [Phonemizer](phonemizer.md). The chain short-circuits: the first
rule producing a non-empty stem IPA wins. `ptb_pos` is forwarded verbatim to
`Lexicon::lookup` when probing candidate stems.

## Possessive rules

```
word.ends_with("s'") and lex.lookup(word[:-2] + "'s", ptb_pos) non-empty:
    return that IPA
word.ends_with("'") and lex.lookup(word[:-1], ptb_pos) non-empty:
    return that IPA
```

These are pure rewrites — no IPA glue; the result is a copy of the stem's
lexicon IPA.

## `-s` rule (`stem_s`)

Eligibility: `len(word) >= 3 and word.ends_with("s")`. Stem candidates, in order:

1. `word` does not end in `"ss"` and `word[:-1]` is in the lexicon →
   stem = `word[:-1]`        (`"cats"` → `"cat"`)
2. (`word` ends in `"'s"`) or (`len(word) > 4`, ends in `"es"`, not `"ies"`) and
   `word[:-2]` is in the lexicon → stem = `word[:-2]`  (`"girl's"` → `"girl"`,
   `"boxes"` → `"box"`)
3. `len(word) > 4`, ends in `"ies"`, and `word[:-3] + "y"` is in the lexicon →
   stem = `word[:-3] + "y"`   (`"babies"` → `"baby"`)

IPA glue (`_s`) inspects the stem IPA's **last UTF-8 codepoint**:

| stem-final codepoint | append | reason |
|---|---|---|
| `p`, `t`, `k`, `f`, `θ` | `s` | voiceless non-sibilant → /s/ |
| `s`, `z`, `ʃ`, `ʒ`, `ʧ`, `ʤ` | `ᵻz` | sibilant → /ᵻz/ |
| anything else | `z` | voiced → /z/ |

(`θ` is U+03B8; every set member is a single codepoint.)

## `-ed` rule (`stem_ed`)

Eligibility: `len(word) >= 4 and word.ends_with("d")`. Stem candidates:

1. `word` does not end in `"dd"` and `word[:-1]` is in the lexicon →
   stem = `word[:-1]`        (`"loved"` → `"love"`)
2. `len(word) > 4`, ends in `"ed"`, not `"eed"`, and `word[:-2]` is in the
   lexicon → stem = `word[:-2]`  (`"walked"` → `"walk"`)

IPA glue (`_ed`, US branch) inspects the stem IPA's final codepoint(s):

| stem-final codepoint | append / replace | result |
|---|---|---|
| in `{p, k, f, θ, ʃ, s, ʧ}` | append `t` | voiceless → /t/ |
| `d` | append `ᵻd` | dental → /ᵻd/ |
| anything else except `t` | append `d` | voiced → /d/ |
| `t`, len(stem_ipa) < 2 | append `ᵻd` | short stem ending /t/ → /ᵻd/ |
| `t`, second-last codepoint in `US_TAUS` | **replace** trailing `t` with `ɾᵻd` | intervocalic flap |
| `t`, second-last codepoint not in `US_TAUS` | append `ᵻd` | /ᵻd/ |

`US_TAUS = { A, I, O, W, Y, i, u, æ, ɑ, ə, ɛ, ɪ, ɹ, ʊ, ʌ }`. The replace case is
the only branch that mutates the stem IPA; "second-last codepoint" means the
codepoint before the final `t`, not a byte offset.

## `-ing` rule (`stem_ing`)

Eligibility: `len(word) >= 5 and word.ends_with("ing")`. Stem candidates:

1. `len(word) > 5` and `word[:-3]` is in the lexicon →
   stem = `word[:-3]`        (`"walking"` → `"walk"`)
2. `word[:-3] + "e"` is in the lexicon →
   stem = `word[:-3] + "e"`  (`"loving"` → `"love"`)
3. `len(word) > 5` and the tail matches `r"([bcdgklmnprstvxz])\1ing$|cking$"` and
   `word[:-4]` is in the lexicon → stem = `word[:-4]`  (`"running"` → `"run"`,
   `"trafficking"` → `"traffic"`)

The regex captures doubled-consonant gemination plus the `-cking` orthographic
case; in C++ it is open-coded (no regex library): the tail is `Cing` where the
fourth-from-last char equals the third-from-last over the listed alphabet, **or**
the tail is exactly `cking`.

IPA glue (`_ing`, US branch):

| stem-final codepoint sequence | append / replace |
|---|---|
| `t` preceded by a `US_TAUS` codepoint, len(stem_ipa) > 1 | replace trailing `t` with `ɾɪŋ` |
| anything else | append `ɪŋ` |

Same intervocalic-flap idea as `-ed`: stem-final `t` after a `US_TAUS` vowel or
`ɹ` collapses into the flap.

## UTF-8 codepoint helper

The rules repeatedly need "last codepoint" and "second-to-last codepoint" of an
IPA string. The IPA blob is UTF-8 and the relevant codepoints are all in the BMP,
so a small helper decodes by stepping back from the end past UTF-8 continuation
bytes (`(b & 0xC0) != 0x80`) then forward-decoding. It lives in the .cpp, not the
header. Membership tests against the small phoneme sets are codepoint comparisons
over `static constexpr` tables.

On malformed input — empty `word`, a lexicon miss on the stem, or a stem IPA that
decodes to an empty codepoint sequence — `try_phonemize` returns `""` rather than
throwing.

## API

```cpp
namespace brosoundml::g2p {

class Lexicon;  // forward

class Morphology {
 public:
    explicit Morphology(const Lexicon& lex);  // borrows lex; must outlive this

    // Apply the inflectional + possessive rules in get_word() order.
    // Returns assembled IPA (stem IPA + glue) on the first rule that fires,
    // or "" if none does. Does NOT try a direct lexicon hit on `word`.
    std::string try_phonemize(std::string_view word,
                              std::string_view ptb_pos = "") const;
};

}  // namespace brosoundml::g2p
```

`Morphology` is a move-constructible thin wrapper around a non-owning
`const Lexicon*`. Errors throw `std::runtime_error` with the
`"brosoundml: g2p::Morphology::<where>: <reason>"` prefix.

`tests/test_morphology.cpp` is gated on `BROSOUNDML_LEXICON_PATH` (or the sibling
default) and SKIPs without the bin. It covers each inflection (`-s`, `-ed`,
`-ing`), the possessive rewrites, the intervocalic-flap branches, and the
all-rules-miss / empty-input paths. Expected IPA is derived inside the test by
calling `Lexicon::lookup` on the predicted stem and concatenating the glue (so
the test verifies the algorithm, not specific lexicon bytes), except the flap
branches, which assert the hard-coded `ɾᵻd` / `ɾɪŋ` substrings.
