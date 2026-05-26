# POS Tagger Spec

A small Transformer encoder that produces Penn Treebank POS tags for English words. Sole consumer is `brosoundml::g2p::Phonemizer`, which uses POS to disambiguate heteronyms (`record` N vs V), function-word allomorphy (`the`/`a` before vowels, `used to` vs `used`), and stress placement.

The tagger must be fast on CPU (≤2 ms/sentence at 20 words), small (~3 M params), trainable on brotensor, and built from license-clean UD data.

## Repo posture: code only, no data, no weights

This repo carries the C++ implementation, the training driver, the dataset-builder script, and a small hand-authored golden file. It does **not** carry:

- training data (`pos_train.bin` / `pos_val.bin`) — produced by the build script from a UD checkout outside this repo
- trained weights (`model.bin`) — produced by the trainer; hosted externally by the upstream app (`../bro`) under whatever license the training corpus requires

`PosTagger::load(path)` takes a file path; the caller (bro) is responsible for materialising the file. brosoundml ships no fetch logic, no manifest, no network code. Tests that need real weights are gated on an env var pointing at a local weights file.

## Architecture

Byte-level Transformer encoder. Byte input, PTB tag output, word-level pooling via dedicated separator tokens.

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
| Attention | `brotensor::flash_attention` (FP32). Locked — the 2 ms/sentence CPU target depends on it. |
| Output head | Linear `d_model → NUM_TAGS` |
| Params | ~3.1 M |

No dropout (brotensor has no dropout backward). Regularisation comes from:

- **Input byte-noise**: at training time, with probability `p_noise = 0.02`, a non-special input token is replaced with a random byte before embedding lookup. Zero backward through the noise op — just a stochastic input transform.
- **AdamW** (decoupled weight decay, λ = 0.01) on all weight matrices and embeddings. Biases, LayerNorm gains, and position embeddings are excluded from decay.
- **Aggressive early stop** on validation accuracy plateau (patience = 3 epochs).

Use existing `brosoundml::` modules from `modules.h` where they fit (`Linear`, `LayerNorm`). The encoder block is composed inline in `pos_tagger.cpp` — it's small enough not to warrant a new module.

### Input encoding

Sentence is rendered as UTF-8 bytes. Words are separated by whitespace at the caller level (any unicode whitespace collapses to a word boundary). Tokenisation:

- Emit `<bos>`.
- For each word in input order: emit `<wsep>`, then the word's bytes.
- Emit `<eos>`.

Example for "The cat sat":

```
<bos> <wsep> T h e <wsep> c a t <wsep> s a t <eos>
```

Each `<wsep>` is the **pooling anchor for the word immediately following it** — one rule, one token type, no asymmetry. After the encoder, gather hidden states at the `<wsep>` positions, pass through the output head, get one logit vector per word.

Sentences longer than 384 tokens are split at the last `<wsep>` that fits and tagged in chunks. Truncation never splits a word.

### Output

One PTB tag per word, in input order. The tag set is **whatever the dataset script emits** — see "Training data" below. Order is frozen by `tools/build_pos_dataset.py` and written to `include/brosoundml/g2p/tags.h` as a generated file. Don't hand-edit `tags.h`; re-run the script if the inventory changes (and bump a weights version, since the head is now incompatible).

## Training data

License-clean UD English treebanks. Combine the CC-BY and CC-BY-SA sources (skip NC variants); the resulting weights inherit SA. Candidate set:

- **UD English-EWT** (CC-BY-SA 4.0) — ~254k tokens
- **UD English-GUM** (CC-BY 4.0) — ~150k tokens
- **UD English-Atis** (CC-BY 4.0)
- **UD English-Pronouns** (CC-BY 4.0)
- **UD English-PUD** (CC-BY-SA 4.0)
- **UD English-LinES** (LGPL-LR — review before including)

Final inventory is whatever the build script ends up reading from the UD checkout it's pointed at. Target: ~600-700k tokens, ~30-40k sentences. Hold out 5% per source for validation.

XPOS column of each CoNLL-U file is the PTB tag. Sentence boundaries come from blank lines / `# sent_id` markers per the CoNLL-U spec.

### Dataset script

`tools/build_pos_dataset.py` — Python, one-time, **not** part of the C++ build:

- Reads a UD checkout (path passed via CLI flag).
- Walks the configured treebanks, parses CoNLL-U, collects (sentence, [(word, xpos)]) pairs.
- Builds the canonical tag inventory by sorting observed XPOS strings — emits `include/brosoundml/g2p/tags.h` (generated; checked in as a tag-table-only header, no model code).
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
  [uint16 word_byte_start × num_words]           # offset of each word's first byte in the byte buffer
  [uint16 word_byte_len × num_words]
```

Storing word offsets (not `<wsep>` positions) keeps the file independent of the tokenisation rule — the C++ trainer inserts specials itself.

The script and the generated `tags.h` are the only Python-touched artefacts in the repo. The `.bin` outputs are not.

## Training

C++ training driver: `tests/train_pos_tagger.cpp` (compiled as a standalone executable, not part of the test runner).

| Hyperparameter | Value |
|---|---|
| Batch size | 32 sentences |
| Sequence padding | per-batch, to longest in batch |
| Optimiser | AdamW (β₁=0.9, β₂=0.98, ε=1e-9, λ=0.01) |
| LR | 5e-4, linear warmup 1000 steps, cosine to 1e-5 |
| Loss | Fused softmax cross-entropy over word positions; ignore padded positions |
| Input noise | byte-noise p=0.02 on non-special positions |
| Epochs | up to 30, early-stop patience 3 on val accuracy |
| Mixed precision | FP32 |
| Device | CPU primary; CUDA if available |
| Checkpoint cadence | Every epoch to `<out_dir>/checkpoint_NN.bin` |

CLI:

```
train_pos_tagger \
  --train data/pos_train.bin \
  --val   data/pos_val.bin \
  --tags  include/brosoundml/g2p/tags.h \
  --out   weights/pos_tagger/
```

Acceptance: **≥97% per-word accuracy** on the held-out validation set. (Byte-level Transformers on UD English XPOS reach 97%+ at this scale and corpus size.)

Final artifact: `model.bin` (~12 MB FP32). Format: small header + concatenation of named tensors — match brolm's weight-file convention.

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

Loader validates magic, version, and that arch fields match the compiled-in defaults; tensor names match a known set; shapes match. Mismatches throw `std::runtime_error` with `"brosoundml: pos_tagger: <reason>"`.

## C++ API

`include/brosoundml/g2p/pos_tagger.h`:

```cpp
namespace brosoundml::g2p {

struct WordTag {
  // NOTE: `word` borrows from the sentence string passed to tag(). Caller
  // must keep the input alive for the lifetime of the returned vector, or
  // copy into std::string before letting the input go.
  std::string_view word;
  PosTag tag;
};

class PosTagger {
 public:
  static PosTagger load(const std::string& weights_path);

  // Tag one sentence. Caller is responsible for sentence splitting.
  std::vector<WordTag> tag(std::string_view sentence) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  PosTagger();
};

}  // namespace brosoundml::g2p
```

`PosTag` and `NUM_TAGS` live in the generated `tags.h`. No batched tag API in v1 — add it when there's a real caller. No JS bindings — POS is internal to G2P.

## File layout

```
brosoundml/
  include/brosoundml/g2p/
    pos_tagger.h
    tags.h                  # GENERATED by build_pos_dataset.py — do not edit
  src/g2p/
    pos_tagger.cpp          # tokenise + encoder forward + word-pool + argmax
    pos_tagger_load.cpp     # weight file parsing
  tests/
    test_pos_tagger.cpp     # smoke + golden + (gated) accuracy
    train_pos_tagger.cpp    # training driver (standalone exe)
    pos_golden.txt          # 100 hand-tagged sentences, author-owned
  tools/
    build_pos_dataset.py    # UD → binary + tags.h (reference; not built)
```

No `data/` and no `weights/` in this repo.

## Acceptance criteria

1. `test_pos_tagger` passes with **no env var set**: links, loads tags, tokenises correctly (byte-exact round-trip on the golden sentences), runs a forward pass on a model loaded from a synthetic weight blob (deterministic shape check), exercises the public API.
2. `test_pos_tagger` passes with `BROSOUNDML_POS_WEIGHTS=<path>` set: loads the real artifact, tags the 100 golden sentences with ≥97% per-word accuracy.
3. `train_pos_tagger` runs end-to-end on CPU in <60 min on the UD-EWT+GUM subset and produces a model hitting ≥97% on val.
4. CPU inference: <2 ms per 20-word sentence on the dev machine (Ryzen-class), flash attention enabled.
5. No Python or external POS taggers invoked at runtime.
6. No LDC-licensed data anywhere in the repo.

## Out of scope

- GPU inference (allowed, not required).
- Multilingual POS — English only.
- Subword/BPE input — byte-level keeps the artifact dep-free.
- Sub-word tag prediction — one tag per word.
- Artifact distribution mechanism — lives in `../bro`, not here.
