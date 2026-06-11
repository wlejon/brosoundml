# G2P — in-tree English grapheme-to-phoneme

Kokoro upstream phonemizes text with [misaki](https://github.com/hexgrad/misaki)
(Apache 2.0), a Python frontend. brosoundml ships an in-tree English G2P at
`brosoundml::g2p::` so Kokoro can phonemize text with **no misaki/Python
dependency** — the path that matters for embedded / no-Python deployments. It
turns raw English text into the Kokoro phoneme token ids that
`Kokoro::synthesize()` consumes (US English only). `synthesize()` still accepts
phoneme ids directly for callers who supply their own.

Public surface: `include/brosoundml/g2p/` — `pos_tagger.h`, `tags.h`,
`lexicon.h`, `morphology.h`, `special_cases.h`, `normalizer.h`,
`phoneme_adapter.h`, `phonemizer.h`.

## Pipeline

`Phonemizer` (`phonemizer.h`) is the façade — it orchestrates the components
into the `sentence → vector<int32_t>` entry point:

```
   text ─▶ Normalizer       expand numbers, dates, symbols, abbreviations into
                            spoken-form words.
        ─▶ PosTagger        a byte-level Transformer POS tagger; the part of
                            speech disambiguates heteronyms (e.g. "read",
                            "lead") before lookup.
        ─▶ SpecialCases     hand-authored overrides for words the lexicon gets
                            wrong or can't reach.
        ─▶ Lexicon          the packed English pronunciation dictionary —
                            grapheme (+ POS) ─▶ IPA.
        ─▶ Morphology       fallback for out-of-lexicon words: strip affixes,
                            look up the stem, reattach.
        ─▶ PhonemeAdapter   IPA codepoint sequence ─▶ Kokoro phoneme token ids
                            (codepoint-level; no BPE/merges).
```

## Components

Each component has a detailed spec under `docs/`:

| Component | Header | Spec | Role |
|---|---|---|---|
| POS tagger | `pos_tagger.h` | [pos_tagger.md](pos_tagger.md) | byte-level Transformer POS tagger (heteronym disambiguation) |
| Lexicon | `lexicon.h` | [lexicon.md](lexicon.md) | packed grapheme(+POS) → IPA dictionary |
| Morphology | `morphology.h` | [morphology.md](morphology.md) | affix-strip fallback for out-of-lexicon words |
| Special cases | `special_cases.h` | [special_cases.md](special_cases.md) | hand-authored overrides |
| Phoneme adapter + Phonemizer | `phoneme_adapter.h`, `phonemizer.h` | [phonemizer.md](phonemizer.md) | IPA → Kokoro ids, and the orchestrating façade |
| Normalizer | `normalizer.h` | — | text normalization (numbers, dates, symbols) |

## Data and weights

The POS tagger weights (`pos_tagger/model.bin`) and the packed lexicon live in
the sibling [`brosoundml-data`](https://huggingface.co/datasets/wlejon/brosoundml-data)
repo, resolved by the standard path convention (caller-supplied path >
`BROSOUNDML_DATA_DIR` > `../brosoundml-data`). The Kokoro vocab the adapter maps
into is loaded by `KokoroConfig::vocab` from the Kokoro `config.json`.

## Data-prep tools (offline)

- `tools/build_pos_dataset.py` — build the POS-tagger training dataset.
- `tools/build_lexicon.py` — pack the English lexicon binary.

The POS tagger itself is trained in-tree by `tests/train_pos_tagger.cpp`
(`brotensor` training ops), with `tests/eval_pos_confusion.cpp` for evaluation
and `tests/pos_golden.txt` as the regression fixture.
