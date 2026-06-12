# POS tagger

A small byte-level Transformer encoder that produces Penn Treebank POS tags for
English words. Its only consumer is `brosoundml::g2p::Phonemizer`, which uses the
tag to disambiguate heteronyms (`record` N vs V), function-word allomorphy
(`a`/`the` before vowels, `used to` vs `used`), and stress placement.

It runs on CPU in under ~2 ms per 20-word sentence, is ~3 M parameters, and is
trained in-tree on brotensor from license-clean Universal Dependencies data.

Public surface: `include/brosoundml/g2p/pos_tagger.h` (`PosTagger`, `WordTag`).
The tag inventory is the generated `include/brosoundml/g2p/tags.h`.

## Weights and data

The repo carries the C++ implementation, the in-tree training driver, the
dataset-builder script, and a hand-authored golden file. It does **not** carry:

- training data (`pos_train.bin` / `pos_val.bin`) — produced by the build script
  from a UD checkout outside the repo.
- trained weights (`pos_tagger/model.bin`) — produced by the trainer and hosted
  in the sibling `brosoundml-data` repo, resolved by the standard path
  convention (caller-supplied path > `BROSOUNDML_DATA_DIR` > `../brosoundml-data`).

`PosTagger::load(path)` takes a file path; the caller materialises the file.
brosoundml ships no fetch logic, no manifest, no network code. Tests that need
real weights are gated on an env var pointing at a local weights file.

## Architecture

Byte-level Transformer encoder: byte input, PTB tag output, word-level pooling
via dedicated separator tokens.

| Component | Value |
|---|---|
| Input vocab | 256 bytes + 4 specials (`<pad>=0`, `<bos>=1`, `<eos>=2`, `<wsep>=3`) → 260 |
| Max sequence length | 384 tokens |
| `d_model` | 192 |
| Layers | 4 |
| Heads | 4 (head_dim = 48) |
| FFN hidden | 768 (4× `d_model`) |
| Activation | GELU |
| Position encoding | Learned absolute (384 × 192) |
| Norm | Pre-norm LayerNorm |
| Attention | `brotensor::flash_attention` (FP32) — the 2 ms/sentence CPU target depends on it |
| Output head | Linear `d_model → NUM_TAGS` |
| Params | ~3.1 M |

There is no dropout (brotensor has no dropout backward); regularisation during
training comes from input byte-noise, AdamW decoupled weight decay, and early
stopping (see "Training"). The encoder block is composed inline in
`pos_tagger.cpp` from the shared `modules.h` layers (`Linear`, `LayerNorm`) — it
is small enough not to warrant its own module.

### Input encoding

A sentence is rendered as UTF-8 bytes. Words are separated by whitespace at the
caller level (any Unicode whitespace collapses to a word boundary). Tokenisation:

- Emit `<bos>`.
- For each word in input order: emit `<wsep>`, then the word's bytes.
- Emit `<eos>`.

For "The cat sat":

```
<bos> <wsep> T h e <wsep> c a t <wsep> s a t <eos>
```

Each `<wsep>` is the pooling anchor for the word immediately following it. After
the encoder, hidden states at the `<wsep>` positions are gathered and passed
through the output head, giving one logit vector per word. Sentences longer than
384 tokens are split at the last `<wsep>` that fits and tagged in chunks;
truncation never splits a word.

### Output

One PTB tag per word, in input order. The tag set is whatever the dataset script
emits, frozen into `include/brosoundml/g2p/tags.h` as a generated header (tag
table only, no model code). `tags.h` is not hand-edited — it is regenerated if
the inventory changes, which also bumps a weights version since the head shape
changes.

## Weight file format

```
[uint32 magic     = 0x504F5302 ("POS\x02")]
[uint32 version   = 1]
[uint32 num_tags]                              # must match tags.h NUM_TAGS at load
[uint32 d_model]
[uint32 num_layers]
[uint32 num_heads]
[uint32 ffn_hidden]
[uint32 max_seq_len]
[uint32 num_tensors]
per tensor:
  [uint16 name_len][name bytes]
  [uint8 rank][uint32 dim × rank]
  [float32 × prod(dims)]
```

The loader validates magic, version, and that the arch fields match the
compiled-in defaults; tensor names match a known set; shapes match. Mismatches
throw `std::runtime_error` with a `"brosoundml: pos_tagger: <reason>"` message.
The shipped artifact is ~12 MB FP32.

## Training

The model is trained in-tree by `tests/train_pos_tagger.cpp` (compiled as a
standalone executable, not part of the test runner) using brotensor's training
ops.

| Hyperparameter | Value |
|---|---|
| Batch size | 32 sentences |
| Sequence padding | per-batch, to longest in batch |
| Optimiser | AdamW (β₁=0.9, β₂=0.98, ε=1e-9, λ=0.01) |
| LR | 5e-4, linear warmup 1000 steps, cosine to 1e-5 |
| Loss | Fused softmax cross-entropy over word positions; padded positions ignored |
| Input noise | byte-noise p=0.02 on non-special positions (a stochastic input transform, no backward) |
| Epochs | up to 30, early-stop patience 3 on validation accuracy |
| Precision | FP32 |
| Device | CPU primary; CUDA if available |
| Checkpoint cadence | every epoch to `<out_dir>/checkpoint_NN.bin` |

AdamW decoupled weight decay (λ = 0.01) applies to weight matrices and
embeddings; biases, LayerNorm gains, and position embeddings are excluded.
The model reaches ≥97% per-word accuracy on held-out UD English XPOS at this
scale and corpus size.

CLI:

```
train_pos_tagger \
  --train data/pos_train.bin \
  --val   data/pos_val.bin \
  --tags  include/brosoundml/g2p/tags.h \
  --out   weights/pos_tagger/
```

`tests/eval_pos_confusion.cpp` evaluates a trained checkpoint and
`tests/pos_golden.txt` (100 hand-tagged sentences) is the regression fixture.

## Training data

License-clean Universal Dependencies English treebanks, combining the CC-BY and
CC-BY-SA sources (NC variants skipped); the resulting weights inherit SA.
Sources include UD English-EWT (CC-BY-SA 4.0, ~254k tokens), English-GUM (CC-BY
4.0, ~150k tokens), English-Atis, English-Pronouns, and English-PUD. The XPOS
column of each CoNLL-U file is the PTB tag; sentence boundaries come from blank
lines / `# sent_id` markers. 5% per source is held out for validation.

### Dataset script

`tools/build_pos_dataset.py` — Python, one-time, offline, not part of the C++
build:

- Reads a UD checkout (path passed via CLI flag).
- Walks the configured treebanks, parses CoNLL-U, collects `(sentence,
  [(word, xpos)])` pairs.
- Builds the canonical tag inventory by sorting observed XPOS strings and emits
  the generated `include/brosoundml/g2p/tags.h`.
- Writes binary `pos_train.bin` + `pos_val.bin` to a caller-specified output dir.

Binary format:

```
[uint32 magic = 0x504F5301 ("POS\x01")]
[uint32 num_tags]
[uint32 num_sentences]
per sentence:
  [uint32 num_bytes][bytes...]                   # raw UTF-8, no specials
  [uint32 num_words]
  [uint8 tag_id × num_words]
  [uint16 word_byte_start × num_words]           # offset of each word's first byte
  [uint16 word_byte_len × num_words]
```

Word offsets (not `<wsep>` positions) are stored so the file is independent of
the tokenisation rule — the C++ trainer inserts the specials itself. The script
and the generated `tags.h` are the only Python-touched artefacts in the repo;
the `.bin` outputs are not checked in.

## API

```cpp
namespace brosoundml::g2p {

struct WordTag {
  // `word` borrows from the sentence passed to tag(); keep the input alive
  // for the lifetime of the returned vector, or copy into std::string.
  std::string_view word;
  PosTag           tag;
};

class PosTagger {
 public:
  static PosTagger load(const std::string& weights_path);

  // Tag one sentence. Caller is responsible for sentence splitting.
  std::vector<WordTag> tag(std::string_view sentence) const;
};

}  // namespace brosoundml::g2p
```

`PosTag` and `NUM_TAGS` live in the generated `tags.h`. POS is internal to G2P:
there is no batched tag API and no JS binding.

## File layout

```
include/brosoundml/g2p/
  pos_tagger.h
  tags.h                  # GENERATED by build_pos_dataset.py
src/g2p/
  pos_tagger.cpp          # tokenise + encoder forward + word-pool + argmax
  pos_tagger_load.cpp     # weight-file parsing
  pos_tagger_internal.h
tests/
  test_pos_tagger.cpp     # smoke + golden + (gated) accuracy
  train_pos_tagger.cpp    # training driver (standalone exe)
  eval_pos_confusion.cpp  # confusion-matrix evaluation
  pos_golden.txt          # 100 hand-tagged sentences
tools/
  build_pos_dataset.py    # UD → binary + tags.h (offline)
```

`test_pos_tagger` runs with no env var (links, loads `tags.h`, byte-exact
tokenisation round-trip, a forward pass on a synthetic weight blob, the public
API) and, with `BROSOUNDML_POS_WEIGHTS=<path>` set, loads the real artifact and
tags the golden sentences at ≥97% per-word accuracy.
