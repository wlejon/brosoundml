# Phonemizer

The front end of the in-tree English G2P stack and its two final pieces: the
**PhonemeAdapter** that turns an IPA codepoint sequence into Kokoro phoneme token
ids, and the **Phonemizer** façade that orchestrates text
[normalization](#normalization) + [PosTagger](pos_tagger.md) +
[SpecialCases](special_cases.md) + [Lexicon](lexicon.md) +
[Morphology](morphology.md) + PhonemeAdapter into the `sentence →
vector<int32_t>` entry point that `Kokoro::synthesize` consumes. US English only.

Public surface: `include/brosoundml/g2p/phonemizer.h` (`Phonemizer`),
`include/brosoundml/g2p/phoneme_adapter.h` (`PhonemeAdapter`),
`include/brosoundml/g2p/normalizer.h` (`normalize_text`). The Kokoro vocab the
adapter maps into is loaded by `KokoroConfig::vocab` from
`weights/kokoro/config.json`; no new artifacts are introduced.

## PhonemeAdapter — IPA → Kokoro ids

Kokoro's phoneme vocab is a single-codepoint phoneme set read from
`config.json`: 114 entries, keys one Unicode codepoint each (three are
two-codepoint combining-mark sequences). IDs are gapped (`max id = 177`, only
114 used) and start at 1 — id 0 is reserved for padding. The vocab holds ASCII
punctuation (`; : , . ! ?`), the ASCII space (id 16, the word separator), the
stress marks `ˈ` (156) / `ˌ` (157), IPA vowels and consonants, and a few exotic
cross-lingual entries never produced by the US-English lexicon.

The mapping is **codepoint-level** — no sub-tokenisation, no BPE, no merges. The
IPA string `"hɛˈloʊ"` is exactly six codepoints → six ids. A verified decode
against `weights/kokoro/ids.txt`:

```
[50, 86, 156, 54, 57, 135, 16, 65, 83, 123, 54, 46]
 h   ɛ   ˈ    l   o   ʊ   ' ' w   ə   ɹ   l   d
```

`encode(ipa)` walks the string left to right, looking up each codepoint's UTF-8
byte sequence in the vocab and appending the id. Behaviour:

- **Misses are dropped silently** — no throw, no id 0 — mirroring misaki / Kokoro
  upstream and avoiding spurious tokens that would derail the model.
- **Combining marks** (`ã`, `ẽ`, `ĩ`, …) are handled by a one-codepoint
  right-greedy peek: if the current codepoint paired with the next exists as a
  vocab key, that two-codepoint key wins and both are consumed. This is a
  peek-one lookahead, never two — not a BPE merge step.
- The literal ASCII space maps to id 16; the Phonemizer (not the adapter) is
  responsible for inserting spaces between word IPAs.

`max_id()` returns the largest id in the vocab (177), computed once at
construction — useful for sizing embedding tables in tests without depending on
`KokoroConfig::n_tokens`.

```cpp
class PhonemeAdapter {
 public:
    explicit PhonemeAdapter(const std::unordered_map<std::string, int>& vocab);
    std::vector<std::int32_t> encode(std::string_view ipa) const;
    int max_id() const;
};
```

## Normalization

`normalize_text(sentence)` runs at the very front of the Phonemizer, before
pre-tokenisation, rewriting a raw sentence into one whose tokens the lexicon /
morphology / special-case chain can actually pronounce. Without it, digit and
symbol tokens miss every lookup stage and either vanish to silence (numbers, `%`,
`$`) or get spelled letter-by-letter. It is a pure function — no model or lexicon
needed — whose output is whitespace-padded around expansions, which the
pre-tokeniser's whitespace split collapses. Transforms (US English):

| Input | Output |
|---|---|
| `*Hamlet*`, `[text](url)` | `Hamlet`, `text` (markdown stripped) |
| `' ’ “ ”` | ASCII `'` `"` (smart quotes folded) |
| `42` | `forty two` (cardinal) |
| `1,000` | `one thousand` (thousands separators) |
| `3.14` | `three point one four` (decimal) |
| `21st` | `twenty first` (ordinal) |
| `1984` | `nineteen eighty four` (4-digit year heuristic) |
| `$5.50` | `five dollars and fifty cents` (currency $ £ €) |
| `50%` | `fifty percent` |
| `3+4` | `three plus four` (bare operators) |

The year reading is a heuristic: a bare four-digit integer in [1000, 2099] with
no grouping / decimal / currency is read as two pairs; everything else reads as a
cardinal. Numeric tokens inside currency, with thousands commas, or outside that
range stay cardinal.

## Phonemizer — the orchestrator

`phonemize(sentence) → vector<int32_t>` runs the full pipeline:

1. **Normalize** the sentence (`normalize_text`).
2. **Pre-tokenize** into a flat sequence of `(leading_punct, word,
   trailing_punct)` triples, splitting on ASCII whitespace.
3. **POS-tag** by space-joining the `word` fields and calling
   `PosTagger::tag`; tags align to triples by index.
4. **Right-to-left walk** to phonemize each word, propagating `TokenContext`
   from the next word backwards.
5. **Assemble** an IPA string — for each token in original order,
   `leading_punct + word_ipa + trailing_punct`, separated by a single ASCII space
   (which the adapter tokenises to the word-separator id 16).
6. **Encode** the IPA via `PhonemeAdapter::encode`.

`phonemize_to_ipa` runs the same pipeline but returns the assembled IPA string
instead of ids — for tests, debugging, and the `brosoundml_synth` CLI's verbose
mode.

### Pre-tokenisation

```
input: "Hello, world."
output:
  [ {leading: "", word: "Hello", trailing: ","},
    {leading: "", word: "world", trailing: "."} ]
```

`pretokenize(string_view) → vector<Triple>` splits on runs of ASCII whitespace,
then for each non-empty chunk peels a leading-punctuation run (`( [ { " '` plus
opening smart-quotes `“` `‘`) and a trailing-punctuation run (`. , ; : ! ? ) ] }
" '` plus closing smart-quotes `”` `’`, em-dash `—`, ellipsis `…`). Multi-char
trailing runs are preserved (`"!?"` stays `"!?"`). Mid-word apostrophes stay
attached (`"don't"`, `"girl's"`) because the peeler stops at the first non-punct
char. Pure-punctuation chunks (empty `word`) are skipped. The pre-tokeniser does
no normalisation itself — that is the normalizer's job — and surviving
punctuation becomes part of the assembled IPA, tokenised to the adapter's
punctuation ids verbatim.

### POS tagging

The `word` fields are space-joined and passed to `PosTagger::tag`, which splits
on whitespace exactly as the pre-tokeniser does, so the returned `WordTag` count
matches the triple count. A mismatch throws
`"brosoundml: g2p::Phonemizer::tag_align: tagger returned N tags for M words"`.
The per-word PTB string is `pos_tag_name(WordTag::tag)` from the generated
`tags.h`.

### Right-to-left phonemization

Each word is phonemized in misaki's `get_word()` order:

```
phonemize_word(word, ptb, ctx):
    out = special_cases.try_phonemize(word, ptb, ctx)
    if out empty: out = lexicon.lookup(word, ptb)
    if out empty: out = morphology.try_phonemize(word, ptb)
    if out empty and word has ASCII upper:               # case-fold pre-rule
        wl = ascii_tolower(word)
        out = lexicon.lookup(wl, ptb)
        if out empty: out = morphology.try_phonemize(wl, ptb)
    if out empty: out = special_cases.spell_letter_by_letter(word)  # NNP fallback
    return out
```

The case-fold pre-rule is a deliberate simplification of misaki's more intricate
lowercasing predicate — "try original, then try lowercased". Context propagates
per misaki's `token_context`:

```
update_ctx(ctx, ipa, word, ptb):
    for codepoint in ipa:            # first vowel / consonant / punct decides
        if codepoint in VOWELS:          future_vowel =  1; break
        if codepoint in CONSONANTS:      future_vowel = -1; break
        if codepoint in NON_QUOTE_PUNCT: future_vowel =  0; break
    future_to = (word in {"to","To"}) or (word == "TO" and ptb in {"TO","IN"})
```

with

- `VOWELS = {A, I, O, Q, W, Y, a, i, u, æ, ɑ, ɒ, ɔ, ə, ɛ, ɜ, ɪ, ʊ, ʌ, ᵻ}`
- `CONSONANTS = {b, d, f, h, j, k, l, m, n, p, s, t, v, w, z, ð, ŋ, ɡ, ɹ, ɾ, ʃ, ʒ, ʤ, ʧ, θ}`
- `NON_QUOTE_PUNCT = {! , . : ; ?}`

The walk runs last-to-first so each `ctx` is filled before it is read; the last
token starts with `TokenContext{0, false}`. The per-token IPA buffer holds stem
IPA only — leading/trailing punctuation is interleaved during assembly.

### Assembly and encoding

```
buf = ""
for i in 0..N-1:
    if i > 0: buf += " "                  # word separator → adapter id 16
    buf += triples[i].leading + ipa_per_token[i] + triples[i].trailing
return adapter.encode(buf)
```

A leading/trailing field with a codepoint absent from the Kokoro vocab is
silently dropped by the adapter — no Phonemizer-side filtering needed.

## API

```cpp
namespace brosoundml::g2p {

class Phonemizer {
 public:
    // Borrows every dependency; all must outlive this Phonemizer.
    Phonemizer(const PosTagger&      tagger,
               const Lexicon&        lexicon,
               const Morphology&     morphology,
               const SpecialCases&   special,
               const PhonemeAdapter& adapter);

    // text → Kokoro phoneme ids. Empty / whitespace-only input → empty vector.
    std::vector<std::int32_t> phonemize(std::string_view sentence) const;

    // Same pipeline, returns the assembled IPA string instead of ids.
    std::string phonemize_to_ipa(std::string_view sentence) const;
};

}  // namespace brosoundml::g2p
```

`Phonemizer` is move-constructible and holds five non-owning pointers — no pImpl.
It throws `std::runtime_error` (with the `"brosoundml: g2p::Phonemizer::..."`
prefix) only on unrecoverable internal inconsistencies such as a tag/word-count
mismatch.

`tests/test_phonemizer.cpp` is gated on the lexicon, POS weights, and Kokoro
config all resolving (env var + sibling fallback each) and SKIPs otherwise. It
asserts structural properties rather than exact id sequences (which are brittle
against tuning): function-word vowel allomorphy (`the apple` → `ði`, `the cat` →
`ðə`), `to` allomorphy, POS-driven heteronyms (`I record songs` vs `a record
player`), the morphology fallback, the spelling fallback, punctuation
passthrough, and empty input.

## Known gaps vs. misaki

US English only — no British branches, no non-English fallbacks. Numbers,
currency, years, and percent are handled by the [normalizer](#normalization)
ahead of the chain. Remaining differences from misaki, documented for future
tightening:

- **No `"used to"` collocation pass.** Misaki collapses `used to` into one token
  before POS tagging; the `future_to` mechanism partially covers the pronunciation
  effect (the `used` rule fires on JJ/VBD when `future_to` is set), so the audible
  result is close.
- **`vs.` is rewritten at lookup time, not preprocess time** — the SpecialCases
  rule handles sentence-internal `vs.` but does not strip the trailing `.` from
  the token.
- **No sentence-stress redistribution** (misaki's `resolve_tokens`); some
  sentences are mildly over-stressed.
- **Tagger word splitting is trusted to match the pre-tokeniser** — both split on
  ASCII whitespace, so this holds in practice; any drift throws rather than
  silently misaligning context.
