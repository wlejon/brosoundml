# Special cases

A rule engine for `brosoundml::g2p::` that handles English words the
[lexicon](lexicon.md) + [morphology](morphology.md) pair can't disambiguate on
their own: function-word allomorphy (`a`/`an`/`the` before vowels, `to` before
vowels, etc.), symbol words (`%` → "percent", `&` → "and", …), dotted acronyms
(`U.S.A.` → spelled out), and the letter-by-letter spelling fallback for unknown
proper nouns and all-caps acronyms. It is a direct port of `misaki/en.py`'s
`get_special_case` and `get_NNP`. US English only.

`SpecialCases` runs **before** the lexicon: if `try_phonemize` returns non-empty,
that result is used and the lexicon / morphology chain is skipped for the word.
Sentence context (what the next word starts with, needed for `a`/`an`/`the`/`to`)
is carried in a small `TokenContext` struct that the [Phonemizer](phonemizer.md)
assembles during a right-to-left walk; `SpecialCases` reads the struct but never
builds or mutates it.

Public surface: `include/brosoundml/g2p/special_cases.h` (`SpecialCases`,
`TokenContext`). The letter-by-letter fallback and the symbol-word spell-outs
(`"percent"`, `"and"`, … are real lexicon entries) depend on `Lexicon`.

## Upstream source

- File: `misaki/en.py`, commit `fba1236595f2d2bf21d414ba6e57d25256afada3` (same
  pin as the lexicon).
- Methods ported: `get_special_case`, `get_NNP`, and the `apply_stress` branches
  reached from `get_NNP` (level `0` and the `rsplit/join` trailer).
- Constants ported: `SYMBOLS = {'%':'percent','&':'and','+':'plus','@':'at'}`,
  `ADD_SYMBOLS = {'.':'dot','/':'slash'}`.
- License: Apache 2.0 (misaki's), inherited by the C++ port.

The full `apply_stress` engine, number / currency / year parsing, and
preprocessing-stage tokenization (hyphens, slashes, contractions) are not part of
this layer — number/currency/year handling lives in the
[normalizer](phonemizer.md), and the rest in the Phonemizer's preprocessing.

## TokenContext semantics

`future_vowel` is **tri-state**:

| Value | Meaning |
|---|---|
| `0` | No following IPA-producing token (sentence-final word, or only punctuation follows). |
| `1` | The following word's first phoneme is a vowel (per misaki's `VOWELS` set). |
| `-1` | The following word's first phoneme is a consonant (or non-vowel non-punct). |

Several rules branch on all three states, so collapsing to `bool` loses
information. `future_to` is a plain `bool`. The Phonemizer populates both fields
in its right-to-left walk.

## Rule table

In order — first match wins. `word` is the surface form (case preserved);
`ptb_pos` is the PTB tag (or empty); `ctx` is the `TokenContext` for this word.

### Symbol words

```
ptb_pos == "ADD" and word == "."  →  lex.lookup("dot")
ptb_pos == "ADD" and word == "/"  →  lex.lookup("slash")
word == "%"                       →  lex.lookup("percent")
word == "&"                       →  lex.lookup("and")
word == "+"                       →  lex.lookup("plus")
word == "@"                       →  lex.lookup("at")
```

`lex.lookup` is called with empty `ptb_pos` — these spell-outs are context-free.

### Dotted acronyms

If `word` matches the misaki test `("." in word.strip(".") and
word.replace(".", "").isalpha() and max(part length) < 3)`, fire
`spell_letter_by_letter(word)`. In plain terms: the word contains at least one
non-decorator `.`, every non-`.` char is an ASCII letter, and the longest run
between dots is 1 or 2 letters. Matches: `"U.S.A."`, `"a.k.a."`, `"M.I.T."`,
`"e.g."` (longest part `eg` = 2 < 3), `"U.S.S.R."`. Does not match: `"abc.def"`
(longest part 3).

### Function words

```
a / A:
    ptb_pos == "DT" → "ɐ"        else → "ˈA"  (the letter A spelled out)

am / Am / AM:
    ptb_pos starts "NN" → spell_letter_by_letter(word)
    ctx.future_vowel == 0 or word != "am" → lex.lookup("am")
    else → "ɐm"

an / An / AN:
    word == "AN" and ptb_pos starts "NN" → spell_letter_by_letter(word)
    else → "ɐn"

I:
    ptb_pos == "PRP" → "ˌI"      (secondary stress + the diphthong marker I)

by / By / BY:
    parent_tag(ptb_pos) == "ADV" → "bˈI"

to / To / TO:
    (word in {"to","To"}) or (word == "TO" and ptb_pos in {"TO","IN"}):
        future_vowel  0 → lex.lookup("to")   (default)
        future_vowel -1 → "tə"               (before consonant)
        future_vowel  1 → "tʊ"               (before vowel)

in / In / IN:
    (word in {"in","In"}) or (word == "IN" and ptb_pos != "NNP"):
        future_vowel == 0 or ptb_pos != "IN" → "ˈɪn"   else → "ɪn"

the / The / THE:
    (word in {"the","The"}) or (word == "THE" and ptb_pos == "DT"):
        future_vowel == 1 → "ði"   else → "ðə"

vs / vs. / VS / Vs:
    ptb_pos == "IN" and /^vs\.?$/i → lex.lookup("versus")

used / Used / USED:
    ptb_pos in {"VBD","JJ"} and ctx.future_to → lex.lookup("used","VBD")
    else → lex.lookup("used")     (DEFAULT)
```

`parent_tag` collapses PTB to misaki families (`VB*`→VERB, `NN*`→NOUN, `RB*`/ADV
→ADV, `JJ*`/ADJ→ADJ, otherwise the tag verbatim) — the same helper the lexicon
uses for variant selection. The capital `I` in the `I` and `by` rules is misaki's
diphthong marker (ARPAbet `AY`), the single codepoint U+0049, not the
orthographic letter. The `vs` match is case-insensitive with an optional trailing
`.`, open-coded without a regex.

## Letter-by-letter spelling

```
ps = ""
for ch in word:
    if ch is ASCII alphabetic:
        ipa = lex.lookup(uppercase(ch))     # ptb_pos = ""
        if ipa.empty(): return ""           # missing letter
        ps += ipa
# Promote the last secondary stress to primary, per misaki get_NNP:
i = last_occurrence_of("ˌ", ps)
if i >= 0: replace that "ˌ" with "ˈ"
return ps
```

`ˌ` is U+02CC, `ˈ` is U+02C8 — both 2 bytes in UTF-8, so the last-occurrence
search and replacement operate on byte offsets with no codepoint accounting.
Non-letter characters are skipped; an empty result or a missing letter (never for
A-Z) returns `""`.

## API

```cpp
namespace brosoundml::g2p {

class Lexicon;  // forward

struct TokenContext {
    int  future_vowel = 0;   // 0 none / 1 vowel / -1 consonant
    bool future_to    = false;
};

class SpecialCases {
 public:
    explicit SpecialCases(const Lexicon& lex);  // borrows lex

    // Returns assembled IPA on a rule match, "" otherwise (caller falls
    // through to the Lexicon / Morphology chain). First match wins.
    std::string try_phonemize(std::string_view word,
                              std::string_view ptb_pos,
                              const TokenContext& ctx) const;

    // Spell `word` letter-by-letter via the Lexicon; the last secondary
    // stress is promoted to primary. Used by the Phonemizer's NNP fallback.
    std::string spell_letter_by_letter(std::string_view word) const;
};

}  // namespace brosoundml::g2p
```

`SpecialCases` mirrors `Morphology`: move-constructible, a non-owning
`const Lexicon*`, no pImpl. The rules are pure total functions over their inputs;
`std::runtime_error` (with the `"brosoundml: g2p::SpecialCases::<where>"` prefix)
is reserved for unrecoverable errors that are not expected to occur.

`tests/test_special_cases.cpp` is gated on the same env var as `test_lexicon` and
SKIPs without the bin. It covers the function words (all three vowel-context
states for the conditioned ones), the symbol words, dotted-acronym matching,
letter-by-letter spelling and its stress promotion, the `vs.` rewrite, and
`used` + `future_to`. Lexicon-derived expectations are computed by calling
`lex.lookup` inside the test; only the hand-typed IPA literals from the rule
table (`"ɐ"`, `"ˈA"`, `"ɐm"`, `"ɐn"`, `"ˌI"`, `"bˈI"`, `"tə"`, `"tʊ"`, `"ˈɪn"`,
`"ɪn"`, `"ði"`, `"ðə"`) are asserted directly.
