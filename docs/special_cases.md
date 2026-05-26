# Special-Cases Spec

A small rule engine for `brosoundml::g2p::` that handles English words
the lexicon + morphology pair can't disambiguate on its own: function-word
allomorphy (`a`/`an`/`the` before vowels, `to` before vowels, etc.),
symbol words (`%` → "percent", `&` → "and", …), dotted acronyms
(`U.S.A.` → spelled out), and the letter-by-letter spelling fallback for
unknown proper nouns and all-caps acronyms. Direct port of misaki/en.py's
`get_special_case` and `get_NNP`.

Sole consumer is the planned `Phonemizer` façade. SpecialCases is run
**before** the lexicon — if it returns non-empty, that result is used and
the lexicon / morphology chain is skipped for that word.

Sentence context (the "what does the next word start with" needed for
`a`/`an`/`the`/`to`) is carried in a small `TokenContext` struct that the
Phonemizer assembles during a right-to-left walk. SpecialCases consumes
the struct but does not build it; that's slice 5.

US English only. Scope-cut from the full misaki special-case logic:

- **In:** function-word allomorphy, symbol words, dotted acronyms,
  letter-by-letter NNP fallback, the minimal subset of `apply_stress`
  needed for NNP spelling, the `vs.` → `versus` rewrite.
- **Out (deferred to a later slice):** number / currency / year
  parsing, preprocessing-stage tokenization (hyphens, slashes,
  contractions, "used to" collocation detection — those live in
  Phonemizer's preprocessor pass), and the full `apply_stress` engine
  (only the rsplit+primary-replace branch used by NNP is needed here).

## Repo posture: code only, no data

Code, tests, this spec — all in `brosoundml`. Depends on `Lexicon` for the
NNP letter-by-letter fallback and the symbol-word lookups (`"percent"`,
`"and"`, etc. are real lexicon entries). No new artifacts.

## Upstream source

- File: `misaki/en.py` (commit `fba1236595f2d2bf21d414ba6e57d25256afada3`,
  same pin as the lexicon).
- Methods ported: `get_special_case`, `get_NNP`, the `apply_stress`
  branches reached from `get_NNP` (level `0` and the `rsplit/join`
  trailer).
- Constants ported: `SYMBOLS = {'%':'percent','&':'and','+':'plus','@':'at'}`,
  `ADD_SYMBOLS = {'.':'dot','/':'slash'}`.
- Misaki license: Apache 2.0; inherited by the C++ port.

## API

```cpp
// include/brosoundml/g2p/special_cases.h

#pragma once

#include <string>
#include <string_view>

namespace brosoundml::g2p {

class Lexicon;  // forward — full def in lexicon.h

// Per-word context derived from the word AFTER this one.
// Phonemizer assembles these in a right-to-left sentence walk.
// SpecialCases reads them to choose function-word allomorphs.
struct TokenContext {
    // 0 = no info (no following IPA-producing token).
    // 1 = the following word starts with a vowel phoneme.
    //-1 = the following word starts with a consonant phoneme.
    int future_vowel = 0;

    // True iff the next token is "to" (case-aware per misaki).
    bool future_to = false;
};

class SpecialCases {
 public:
    explicit SpecialCases(const Lexicon& lex);

    // If a special-case rule matches, returns the assembled IPA.
    // Returns "" if no rule applies — caller falls through to the
    // lexicon / morphology chain.
    //
    // Rule decisions depend on the PTB tag (e.g. 'a' DT vs PRP,
    // 'the' DT, 'I' PRP, 'used' VBD, '.' ADD) and on ctx.
    std::string try_phonemize(std::string_view word,
                              std::string_view ptb_pos,
                              const TokenContext& ctx) const;

    // Letter-by-letter spelling fallback for unknown proper nouns and
    // acronyms. Each ASCII letter is looked up as its uppercase form
    // ('A', 'B', …) in the Lexicon and the IPAs are concatenated.
    // Non-letter characters are skipped.
    //
    // Returns "" if any required letter is missing from the Lexicon
    // (effectively never for A-Z).
    //
    // The trailing secondary-stress mark in the assembled string, if
    // any, is promoted to primary — matching misaki's get_NNP. If no
    // secondary mark is present, the result is the bare concatenation.
    std::string spell_letter_by_letter(std::string_view word) const;
};

}  // namespace brosoundml::g2p
```

Class shape mirrors `Morphology`: non-owning `const Lexicon*`, no pImpl,
move-constructible. Throws `std::runtime_error` with
`"brosoundml: g2p::SpecialCases::<where>: <reason>"` on
non-recoverable errors (none expected; the rules are pure, total
functions over their inputs).

## TokenContext semantics

`future_vowel` is **tri-state**, not a `bool`:

| Value | Meaning |
|---|---|
| `0` | No following IPA-producing token. (e.g. sentence-final word, or only punctuation follows.) |
| `1` | The following word's first phoneme is a vowel (per misaki's `VOWELS` set). |
| `-1` | The following word's first phoneme is a consonant (or non-vowel non-punct). |

Several rules branch on all three states; collapsing to `bool` loses
information. `int` is the right type.

`future_to` is straight `bool`.

Phonemizer is responsible for the right-to-left walk that populates
these fields. SpecialCases never mutates a `TokenContext`.

## Rule table

In order — first match wins. `word` is the surface form (preserves case);
`ptb_pos` is the Penn Treebank tag (or empty for "no tag"); `ctx` is the
TokenContext for this word.

### Symbol words

```
ptb_pos == "ADD" and word == "."  →  lex.lookup("dot")
ptb_pos == "ADD" and word == "/"  →  lex.lookup("slash")
word == "%"                       →  lex.lookup("percent")
word == "&"                       →  lex.lookup("and")
word == "+"                       →  lex.lookup("plus")
word == "@"                       →  lex.lookup("at")
```

`lex.lookup` is called with empty `ptb_pos` — these spell-outs are
context-free.

### Dotted acronyms

If `word` matches the misaki test
`("." in word.strip(".") and word.replace(".", "").isalpha()
   and max(part length) < 3)`, fire `spell_letter_by_letter(word)`.

In plain English: the word contains at least one `.` that is not a
leading/trailing decorator, all non-`.` chars are ASCII letters, and the
longest run between dots is 1 or 2 letters. Examples that match:
`"U.S.A."`, `"a.k.a."`, `"M.I.T."`. Examples that do NOT match:
`"e.g."` (longest part is `eg` after strip — yes, 2 < 3 so this matches),
`"U.S.S.R."` (longest part is 1), `"abc.def"` (longest part is 3 — no).

### `a` / `A`

```
word in ("a", "A"):
    if ptb_pos == "DT":  return "ɐ"
    else:                return "ˈA"     // the letter A spelled out
```

### `am` / `Am` / `AM`

```
word in ("am", "Am", "AM"):
    if ptb_pos starts with "NN":         return spell_letter_by_letter(word)
    elif ctx.future_vowel == 0
         or word != "am"                 return lex.lookup("am")
    else:                                return "ɐm"
```

(The `stress and stress > 0` branch in misaki is dropped — slice 3
doesn't carry the stress parameter through this layer.)

### `an` / `An` / `AN`

```
word in ("an", "An", "AN"):
    if word == "AN" and ptb_pos starts with "NN":
        return spell_letter_by_letter(word)
    return "ɐn"
```

### `I`

```
word == "I" and ptb_pos == "PRP":
    return "ˌI"           // secondary stress + capital I
```

The capital `I` here is the misaki diphthong marker (`AY` in ARPAbet),
not the orthographic letter. Single-codepoint U+0049.

### `by` / `By` / `BY`

```
word in ("by", "By", "BY") and parent_tag(ptb_pos) == "ADV":
    return "bˈI"
```

`parent_tag` collapses PTB to misaki families:

| PTB starts with | parent_tag |
|---|---|
| `VB`  | `VERB` |
| `NN`  | `NOUN` |
| `RB` or `ADV` | `ADV` |
| `JJ` or `ADJ` | `ADJ` |
| (otherwise)   | (the input tag verbatim) |

Already implemented for the lexicon's variant selection; reuse the same
helper.

### `to` / `To` / `TO`

```
word in ("to", "To") or (word == "TO" and ptb_pos in ("TO", "IN")):
    switch ctx.future_vowel:
        case  0: return lex.lookup("to")     // no info → default
        case -1: return "tə"                 // before consonant
        case  1: return "tʊ"                 // before vowel
```

### `in` / `In` / `IN`

```
word in ("in", "In") or (word == "IN" and ptb_pos != "NNP"):
    if ctx.future_vowel == 0 or ptb_pos != "IN":
        return "ˈɪn"        // primary-stressed
    else:
        return "ɪn"
```

### `the` / `The` / `THE`

```
word in ("the", "The") or (word == "THE" and ptb_pos == "DT"):
    if ctx.future_vowel == 1: return "ði"
    else:                     return "ðə"
```

### `vs.` / `vs` / `VS` / `Vs`

```
ptb_pos == "IN" and word matches /^vs\.?$/i:
    return lex.lookup("versus")
```

Case-insensitive match, optional trailing `.`. Both `"vs"` and `"vs."`
fire; `"vss"` does not. Open-code without a regex library.

### `used` / `Used` / `USED`

```
word in ("used", "Used", "USED"):
    if ptb_pos in ("VBD", "JJ") and ctx.future_to:
        return lex.lookup("used", "VBD")   // VBD variant
    return lex.lookup("used")              // DEFAULT
```

`lex.lookup("used", "VBD")` is the path that triggers the VBD heteronym
variant inside the bin.

## Letter-by-letter spelling (`spell_letter_by_letter`)

Algorithm:

```
ps = empty string
for ch in word:
    if ch is ASCII alphabetic:
        letter = uppercase(ch)         // single ASCII byte
        ipa = lex.lookup(letter)       // ptb_pos = ""
        if ipa.empty(): return ""      // missing letter
        ps += ipa

// Promote the last secondary stress to primary, per misaki get_NNP.
// rsplit-on-secondary then join with primary; if no secondary mark
// exists this is a no-op.
i = last_occurrence_of("ˌ", ps)
if i >= 0: ps[i:i+strlen("ˌ")] = "ˈ"

return ps
```

`ˌ` is U+02CC (secondary stress), `ˈ` is U+02C8 (primary stress). Both
are 2 bytes in UTF-8. The "last occurrence" search and replacement
operate on byte offsets — there is no codepoint accounting needed
because both stress marks are the same length.

## Behaviour on edge inputs

- Empty `word` → `""`.
- Letter for which the lexicon returns an empty IPA (e.g. an exotic
  character that case-folds to nothing usable) inside
  `spell_letter_by_letter` → `""`.
- Symbol word lookup that misses in the lexicon → `""` (defensive; the
  lexicon does carry `"percent"`, `"and"`, `"plus"`, `"at"`, `"dot"`,
  `"slash"`, `"versus"` — verify in the test).

## Tests

`tests/test_special_cases.cpp`, gated on the same env var pattern as
`test_lexicon.cpp` and `test_morphology.cpp`. SKIP if the lexicon bin
isn't available.

Required coverage:

1. **Function words, all three context states for the vowel-conditioned
   ones:**
   - `a`/`A` with `ptb_pos = "DT"` → `"ɐ"`; with `"PRP"` → `"ˈA"`.
   - `the` with `future_vowel = 1` → `"ði"`; with `0` or `-1` → `"ðə"`.
   - `to` with `future_vowel = 0` → `lex.lookup("to")`;
     with `1` → `"tʊ"`; with `-1` → `"tə"`.
   - `an` with any ptb_pos → `"ɐn"`. With `"AN"` + `NNP` → spelling.
2. **Symbol words:**
   - `"%"`, `"&"`, `"+"`, `"@"` each return the corresponding
     lexicon lookup.
   - `"."` with `ptb_pos = "ADD"` → `lex.lookup("dot")`;
     with `ptb_pos = "."` (the literal-dot PTB tag) → `""` (no fire).
   - `"/"` with `"ADD"` → `lex.lookup("slash")`.
3. **Dotted acronym:**
   - `"U.S.A."` → equals `spell_letter_by_letter("U.S.A.")` =
     `spell_letter_by_letter("USA")` =
     concat of `lex.lookup("U")`, `lex.lookup("S")`, `lex.lookup("A")`
     with the last secondary-stress promoted.
   - `"abc.def"` → no match (longest part is 3 letters).
4. **Letter-by-letter spelling:**
   - `spell_letter_by_letter("USA")` is non-empty, equals the
     concatenation of A/B/.../Z lookups for the letters `U`, `S`, `A`.
   - Verify the last-secondary-to-primary promotion: pick a multi-letter
     word where at least one of the letters' lexicon IPAs carries
     secondary stress; assert the rebuilt string has primary stress where
     that letter's secondary used to be (and any earlier secondaries are
     untouched). If none of the A-Z letters carry secondary stress in
     the actual bin, document that and skip this assertion.
5. **`vs.` rewrite:**
   - `lookup("vs.", "IN", _)` and `lookup("vs", "IN", _)` both equal
     `lex.lookup("versus")`.
   - `lookup("VS", "IN", _)` also equals it (case insensitivity).
   - `lookup("vss", "IN", _)` returns `""` (not a match).
6. **`used` + `future_to`:**
   - `lookup("used", "VBD", {future_to=true})` equals
     `lex.lookup("used", "VBD")` (the VBD variant from the bin).
   - `lookup("used", "JJ", {future_to=true})` equals same.
   - `lookup("used", "VBD", {future_to=false})` equals
     `lex.lookup("used")` (DEFAULT).
7. **Empty/no-match:**
   - `lookup("hello", "NN", _)` returns `""` — this is not a special
     case; caller falls through to the lexicon.
   - `lookup("", "", _)` returns `""`.

Expected IPA strings are computed by calling `lex.lookup` inside the
test for the lexicon-derived paths (NNP letters, `versus`, `used`/VBD,
`to`/default, etc.). Only the hand-typed IPA literals from the rule
table (`"ɐ"`, `"ˈA"`, `"ɐm"`, `"ɐn"`, `"ˌI"`, `"bˈI"`, `"tə"`, `"tʊ"`,
`"ˈɪn"`, `"ɪn"`, `"ði"`, `"ðə"`) appear hard-coded in assertions.

## Build-out plan revisited

Slice 3 of 5. After this lands:

4. **Kokoro phoneme-id adapter** — IPA → Kokoro vocab ids using
   `weights/kokoro/ids.txt`.
5. **`Phonemizer` façade** — orchestrates POS-tagger + SpecialCases +
   Lexicon + Morphology + adapter into the `sentence → vector<int32_t>`
   API that `Kokoro::synthesize` consumes. Owns sentence tokenization,
   the right-to-left context walk that builds `TokenContext`, the
   case-fold pre-rule, and (eventually) the deferred number / currency
   parsing.

The deferred bits — full `apply_stress`, numbers, "used to" collocation
detection, sentence-stress redistribution (`resolve_tokens`) — land
either inside Phonemizer or as a follow-up slice once an end-to-end
text → audio test is in place.
