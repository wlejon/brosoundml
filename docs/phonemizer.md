# Phonemizer Spec (slices 4 + 5)

The last two slices of the in-tree English G2P stack: the **PhonemeAdapter**
that turns an IPA codepoint sequence into Kokoro phoneme token ids, and the
**Phonemizer** façade that orchestrates `PosTagger` + `SpecialCases` +
`Lexicon` + `Morphology` + `PhonemeAdapter` into the
`sentence → vector<int32_t>` entry point that `Kokoro::synthesize` consumes.

After this lands, the G2P pipeline runs end-to-end from raw English text
to Kokoro input ids. Kokoro's model-side stages (plBERT, text encoder,
predictor, decoder) are independent and gated by their own build-out.

US English only.

## Repo posture

Code only, in `brosoundml`. No new artifacts. The Kokoro vocab is already
loaded by `KokoroConfig::vocab` from `weights/kokoro/config.json`; the
adapter consumes that map directly. The lexicon bin lives in
`brosoundml-data` (unchanged from prior slices).

---

# Slice 4 — `PhonemeAdapter`

## Background

Kokoro's vocab is a single-codepoint phoneme set. From
`weights/kokoro/config.json`:

- **114 entries** total. Keys are one Unicode codepoint each
  (or in three cases a two-codepoint sequence for combining marks —
  see "Edge cases" below).
- IDs are gapped (`max id = 177`, only 114 used). IDs start at 1, not 0;
  id 0 is reserved for padding by the upstream model.
- Vocab contents are a mix of:
  - ASCII punctuation: `; : , . ! ?`
  - ASCII space (id 16 — the **word separator**)
  - Stress marks: `ˈ` (primary, id 156), `ˌ` (secondary, id 157)
  - Vowels and consonants from the IPA Extended block
  - A few exotic / non-English entries (`ʈ`, `ɖ`, `ɽ`, … — never produced
    by the US-English lexicon, but live in the vocab for cross-lingual
    Kokoro variants)

The mapping is **codepoint-level**: there is no multi-character
sub-tokenisation, no BPE, no merges. The IPA string `"hɛˈloʊ"` is
exactly six codepoints → six ids.

Sample decode (verified against `weights/kokoro/ids.txt`'s first line):

```
[50, 86, 156, 54, 57, 135, 16, 65, 83, 123, 54, 46]
 h   ɛ   ˈ    l   o   ʊ   ' ' w   ə   ɹ   l   d
```

Two ids in the upstream test lines decode to the **same** IPA when looked
up — that is fine; the adapter never inverts the map.

## API

```cpp
// include/brosoundml/g2p/phoneme_adapter.h

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <string>

namespace brosoundml::g2p {

class PhonemeAdapter {
 public:
    // Borrows the vocab. Vocab keys are one codepoint each (or a small
    // combining sequence — see encode() behaviour). Lifetime of `vocab`
    // must exceed this adapter.
    explicit PhonemeAdapter(const std::unordered_map<std::string, int>& vocab);

    // Encode an IPA string into Kokoro phoneme ids. Walks `ipa` codepoint
    // by codepoint, looks up each codepoint in the vocab, and appends
    // the resulting id. Codepoints absent from the vocab are SKIPPED
    // silently — matching misaki's behaviour and Kokoro upstream.
    //
    // The vocab carries a few two-codepoint sequences where a base
    // codepoint is followed by a combining mark (U+0303 etc.). When the
    // current codepoint is a base that, combined with the next
    // codepoint, exists as a vocab key, encode the two-codepoint key
    // and advance past both. This is a deterministic right-greedy
    // peek-one lookup, not a BPE merge step.
    std::vector<std::int32_t> encode(std::string_view ipa) const;

    // Largest id present in the vocab. Useful for sizing embedding
    // tables in tests; mirrors KokoroConfig::n_tokens but doesn't
    // depend on it.
    int max_id() const;
};

}  // namespace brosoundml::g2p
```

## Implementation notes

- Walk UTF-8 left to right: read one codepoint, look up its UTF-8 string
  representation in `vocab`. If it hits, append the id.
- For the combining-mark cases (the vocab contains keys like `"ã"` =
  `"ã"`), peek the **next** codepoint: if `(current + next)` as a
  UTF-8 string exists in vocab, prefer that and advance past both. This
  is a one-codepoint lookahead, never two.
- On a codepoint miss, do not throw, do not insert id 0. Just drop it.
  This mirrors what upstream Kokoro's preprocessor does for unsupported
  characters and avoids spurious tokens that would derail the model.
- Allocate result vector with `reserve(ipa.size())` — a rough upper
  bound (≤ 1 id per byte).
- `max_id()` is a single pass over the vocab values at construction;
  cached in a member.

## Edge cases

- Empty IPA → empty vector.
- IPA consisting entirely of unknown codepoints → empty vector.
- IPA containing the literal ASCII space → space gets id 16 (the
  word-separator id). The caller (Phonemizer) is responsible for
  inserting spaces between word IPAs; the adapter doesn't add them.

## Tests

`tests/test_phoneme_adapter.cpp`. Requires Kokoro `config.json`; gate on
the test's standard sibling-path discovery for the Kokoro model dir:
`BROSOUNDML_KOKORO_DIR` env var → `weights/kokoro/` fallback. SKIP if
neither resolves to an existing `config.json`.

Required coverage:

1. **Hello-world decode** matches the upstream sample:
   `encode("hɛˈloʊ wəɹld")` → `[50, 86, 156, 54, 57, 135, 16, 65, 83, 123, 54, 46]`.
2. **Empty input** → empty vector.
3. **Unknown codepoint dropped**: `encode("X")` (capital X is not in
   the vocab) → empty vector.
4. **Mixed known/unknown**: `encode("hXɛ")` → `[50, 86]`.
5. **Stress marks tokenize**: `encode("ˈ")` → `[156]`; `encode("ˌ")` →
   `[157]`.
6. **Space is preserved**: `encode("h h")` → `[50, 16, 50]`.
7. **`max_id()`** returns `177`.

---

# Slice 5 — `Phonemizer`

## Job

`Phonemizer::phonemize(string sentence) → vector<int32_t>` runs the full
text-to-token-ids pipeline. Steps, in order:

1. **Pre-tokenize** the sentence into a flat sequence of `(leading_punct,
   word, trailing_punct)` triples, splitting on ASCII whitespace.
2. **POS-tag** the words by space-joining them and calling
   `PosTagger::tag(sentence)`. Match the returned tags to the
   pre-tokenized words by index.
3. **Right-to-left walk** over the tokens to phonemize each word and
   propagate `TokenContext` from the *next* word backwards.
4. **Assemble** an IPA string: for each token in original order, emit
   `leading_punct + word_ipa + trailing_punct`, separated by a single
   ASCII space (which the adapter tokenises to the Kokoro word separator
   id 16).
5. **Encode** the IPA via `PhonemeAdapter::encode` → token ids.

## API

```cpp
// include/brosoundml/g2p/phonemizer.h

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brosoundml {
class KokoroConfig;  // forward
namespace g2p {

class Lexicon;
class Morphology;
class SpecialCases;
class PosTagger;
class PhonemeAdapter;

class Phonemizer {
 public:
    // Borrows every dependency. All lifetimes must exceed this
    // Phonemizer. None is owned.
    Phonemizer(const PosTagger& tagger,
               const Lexicon& lexicon,
               const Morphology& morphology,
               const SpecialCases& special,
               const PhonemeAdapter& adapter);

    // Full pipeline: text → ids. Empty input → empty vector.
    std::vector<std::int32_t> phonemize(std::string_view sentence) const;

    // Intermediate IPA (no id encoding). Mostly for tests, debugging,
    // and the `brosoundml_synth` CLI's verbose mode.
    std::string phonemize_to_ipa(std::string_view sentence) const;
};

}  // namespace g2p
}  // namespace brosoundml
```

Pimpl is unnecessary — the class holds five non-owning pointers and a
small handful of plain methods. Move-constructible. No throws on normal
input; throws `std::runtime_error` (with the project prefix) only on
unrecoverable internal errors (POS tagger producing a different word
count from our pre-tokenisation, etc.).

## Pre-tokenisation

```
input: "Hello, world."
output:
  [
    {leading: "",  word: "Hello", trailing: ","},
    {leading: "",  word: "world", trailing: "."}
  ]
```

Algorithm:

1. Split on runs of ASCII whitespace (`' '`, `'\t'`, `'\n'`, `'\r'`).
2. For each non-empty chunk:
   - Peel a contiguous run of **leading-punctuation** characters from
     the front: `( [ { " '`. The opening smart-quotes `“` and `‘` count
     too — emit them as their UTF-8 byte sequences in the leading
     field.
   - Peel a contiguous run of **trailing-punctuation** from the back:
     `. , ; : ! ? ) ] } " '` plus closing smart-quotes `”`, `’`, the
     em-dash `—`, the ellipsis `…`. Multi-character trailing runs are
     preserved as-is (e.g. `"!?"` stays as `"!?"`).
   - **Apostrophes in the middle of a word stay attached**: `"don't"`,
     `"girl's"` — the trailing-punct peeler stops as soon as it sees a
     non-punct character.
   - The remaining middle is the `word`.
   - **Skip empty `word` triples** entirely (chunks that are pure
     punctuation — uncommon but possible).

The pre-tokeniser is a single function `pretokenize(string_view) →
vector<Triple>`. It performs no normalisation, no number-handling, no
contraction splitting. Punctuation that is preserved becomes part of
the assembled IPA and gets tokenised to the adapter's punctuation ids
verbatim.

## POS tagging

Build a single normalised sentence by joining the `word` fields with
spaces, pass to `PosTagger::tag(joined)`. The tagger returns a
`vector<WordTag>` with one entry per whitespace-separated word.

The tagger's word splitting **must match** our pre-tokenisation result:
both split on whitespace. If the returned tag count differs from the
triple count, throw
`"brosoundml: g2p::Phonemizer::tag_align: tagger returned N tags for M words"`.

Per-word PTB string is `pos_tag_name(WordTag::tag)` — the slice-1
generated `tags.h` provides the int→string mapping.

## Right-to-left phonemization

For each token from last to first:

```
phonemize_word(word, ptb, ctx):
    # Try chain in misaki's get_word() order
    out = special_cases.try_phonemize(word, ptb, ctx)
    if out empty: out = string(lexicon.lookup(word, ptb))
    if out empty: out = morphology.try_phonemize(word, ptb)
    # Case-fold pre-rule (simplified):
    if out empty and any uppercase ASCII in word:
        wl = ascii_tolower(word)
        out = string(lexicon.lookup(wl, ptb))
        if out empty: out = morphology.try_phonemize(wl, ptb)
    # NNP fallback: spell it out
    if out empty:
        out = special_cases.spell_letter_by_letter(word)
    # If still empty, the word will produce no audio for this token.
    return out
```

The **simplified case-fold pre-rule** is a deliberate deviation from
misaki, which has a fairly intricate gating predicate before deciding
to lowercase. Slice 5 ships the simple "try original, then try
lowercased" approach; a more faithful port can land later if needed.

Context propagation, per misaki `token_context`:

```
update_ctx(ctx, ipa, word, ptb) -> TokenContext:
    next_ctx = ctx
    # Scan ipa for the first phoneme classified as vowel / consonant /
    # punctuation. If we find one, that determines the new future_vowel.
    for codepoint in ipa:
        if codepoint in VOWELS:          next_ctx.future_vowel =  1;  break
        if codepoint in CONSONANTS:      next_ctx.future_vowel = -1;  break
        if codepoint in NON_QUOTE_PUNCT: next_ctx.future_vowel =  0;  break
    # (else: leave future_vowel as the inherited value)

    next_ctx.future_to = (word == "to" || word == "To"
                         || (word == "TO" && (ptb == "TO" || ptb == "IN")))
    return next_ctx
```

The three codepoint sets are the same as in `morphology.md`'s
`US_TAUS` discussion plus a small consonant set:

- `VOWELS = {A, I, O, Q, W, Y, a, i, u, æ, ɑ, ɒ, ɔ, ə, ɛ, ɜ, ɪ, ʊ, ʌ, ᵻ}`
- `CONSONANTS = {b, d, f, h, j, k, l, m, n, p, s, t, v, w, z, ð, ŋ, ɡ, ɹ, ɾ, ʃ, ʒ, ʤ, ʧ, θ}`
- `NON_QUOTE_PUNCT = {! , . : ; ?}`

Walk **right to left** so each `ctx` is filled in before it's read:

```
ipa_per_token = vector<string>(N)
ctx = TokenContext{0, false}   // last token has no successor
for i in (N-1, N-2, ..., 0):
    ipa = phonemize_word(triples[i].word, ptb_tags[i], ctx)
    ipa_per_token[i] = ipa
    ctx = update_ctx(ctx, ipa, triples[i].word, ptb_tags[i])
```

`ipa_per_token` is filled with the **stem IPA only** — no leading or
trailing punctuation. Punctuation is interleaved during assembly.

## Assembly

```
buf = ""
for i in 0..N-1:
    if i > 0: buf += " "                  // word separator → adapter id 16
    buf += triples[i].leading              // pass through verbatim
    buf += ipa_per_token[i]
    buf += triples[i].trailing
return buf
```

A leading or trailing field that contains a codepoint absent from the
Kokoro vocab is silently dropped by the adapter — no Phonemizer-side
filtering needed.

## Encoding

`adapter.encode(buf)` produces the final `vector<int32_t>`. Done.

## Tests

`tests/test_phonemizer.cpp`. Bin discovery: lexicon (`BROSOUNDML_LEXICON_PATH`
+ sibling fallback) AND POS weights (`BROSOUNDML_POS_WEIGHTS_PATH`
+ sibling fallback) AND Kokoro config (`BROSOUNDML_KOKORO_DIR`
+ sibling fallback). SKIP if any are missing.

Required coverage:

1. **Hello, world.** `phonemize("Hello, world.")` returns a non-empty
   id vector whose decoded IPA (via the inverse vocab) starts with the
   `h` phoneme and ends with a `.`. Specific bytes don't need to match
   any reference — just verify the pipeline produces *something* coherent
   and the last id is `4` (the `.` id).
2. **Function-word vowel allomorphy.**
   `phonemize_to_ipa("the apple")` contains `"ði"` (vowel form of "the").
   `phonemize_to_ipa("the cat")` contains `"ðə"` (consonant form).
3. **`to` allomorphy.**
   `phonemize_to_ipa("go to apple")` contains `"tʊ"`.
   `phonemize_to_ipa("go to school")` contains `"tə"`.
   `phonemize_to_ipa("go to")` (sentence-final) contains the
   `lex.lookup("to")` default form.
4. **POS-driven heteronym.**
   `phonemize_to_ipa("I record songs")` and
   `phonemize_to_ipa("a record player")` produce different IPA for
   "record" — the first is the VERB variant, the second is DEFAULT (the
   `record` lexicon entry has a VERB heteronym).
5. **Morphology fallback.**
   `phonemize_to_ipa("cats sat")` — "cats" should resolve through the
   `-s` morphology rule, producing `lex.lookup("cat") + "s"`.
6. **Spelling fallback.**
   `phonemize_to_ipa("NASA launched")` — "NASA" is in the lexicon as
   an ALLCAPS entry, but a made-up `phonemize_to_ipa("KZQX launched")`
   spells "KZQX" letter by letter.
7. **Punctuation passthrough.**
   `phonemize("hi!")` ends in id `5` (the `!` id).
   `phonemize("(yes)")` starts with `12` and ends with `13`.
8. **Empty input.**
   `phonemize("")` and `phonemize("   ")` both return empty vectors.

The test does not assert exact id sequences for whole sentences — those
are too brittle against future tuning of the chain (e.g. tweaking the
case-fold pre-rule). It asserts structural properties: presence of
specific phoneme substrings, terminal punctuation ids, lexicon-derived
expected IPA for predictable tokens.

## Behaviour gaps vs. misaki

Documented so future slices know what to tighten:

- **No number / currency / year handling.** A sentence containing
  digits will emit punctuation/spelling-fallback noise. Defer to a
  follow-up slice once Kokoro forward is up.
- **No `"used to"` collocation special case.** Misaki's preprocessor
  collapses `used to` into a single token with `_.alias = "to"` before
  POS tagging. Our `future_to` mechanism partially covers the
  pronunciation effect (the `used` rule fires on JJ/VBD when
  `future_to` is set), so the audible result is close.
- **No `vs.` rewrite at preprocess time** — the SpecialCases rule
  handles it at lookup time, which works for sentence-internal `vs.`
  but doesn't strip the trailing `.` from the token. Acceptable for v1.
- **No sentence-stress redistribution** (misaki's `resolve_tokens`).
  Some sentences will produce phonologically over-stressed output. Not
  audible enough to block Kokoro forward; defer.
- **No British branches**, no non-English fallbacks. US-only.
- **Tagger word splitting is trusted to match our pre-tokeniser** —
  both split on ASCII whitespace, so this holds in practice. Any
  drift throws an exception rather than silently misaligning context.

## Build-out plan revisited (final)

```
✓ 1. Lexicon          (artifact + loader + tests)
✓ 2. Morphology       (-s / -ed / -ing + possessives)
✓ 3. SpecialCases     (function-word allomorphy + NNP spelling)
  4. PhonemeAdapter   (this slice — IPA → Kokoro ids)
  5. Phonemizer       (this slice — sentence orchestrator)
```

After this commit, the G2P side of Kokoro is feature-complete to the
point where `Phonemizer::phonemize("Hello, world.")` produces a token-id
sequence that Kokoro's model can consume — provided Kokoro's own model
forward stages are stood up. Model-side build-out is independent and
tracked by `kokoro.h`'s own build-out plan.
