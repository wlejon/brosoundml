# Laya audio: letting the decision model hear a stream

Laya answers choice, score and noul questions about a *state* in a single
non-autoregressive forward pass. It uses a ModernBERT-large encoder and a
2-layer head. This document covers giving it an audio state: a few seconds of
a live stream, re-asked every 30 ms hop. It includes a design, a working
prototype, and measurements made on this machine (RTX 4090).

The question it has to answer: is a Laya adapter the right structure for
open-vocabulary keywords, "someone is speaking" and "a word was just
spoken"? For each of the three, the answer is below, measured against the
alternatives.

## Verdict

**The structure is LLaVA-style, not diffusion-style.** A diffusion model uses
a language model as a *conditioner*. The text encoder's states enter the
denoiser through cross-attention layers that were trained jointly with it.
Laya has no cross-attention, and adding some would mean retraining Laya. The
structure that fits is the one LLaVA, Qwen-Audio and Qwen3-ASR itself use:

- A pretrained audio encoder produces latents.
- A small trained projector maps each latent into the language model's
  *input embedding space*.
- Those rows sit in the sequence as soft tokens, here in Laya's state span.

Laya's encoder is bidirectional, so the question tokens and the audio tokens
attend to each other in every layer: early fusion, for free. Laya stays
frozen and only the projector trains. The gradient reaches the projector
through all 28 encoder layers and the head.

**Evidence that it works.** On speakers never seen in training, keyword AUC
is 0.993. For keywords never asked in training it is 0.969. Against spelling
neighbours as negatives ("their" vs "there" class) it is 0.990. A hop costs
10 ms with one keyword and 16 ms with ten, inside the 30 ms budget.

**Where it is and is not the right tool:**

| signal | right structure | evidence (dev, unseen speakers) |
|---|---|---|
| keyword X spoken | Laya adapter is competitive, with a different profile from ASR | AUC 0.993 vs 0.975 for streaming ASR plus string match. At ≤ 1 % FPR, ASR still recalls more: 0.95 vs 0.86, and 0.90 vs 0.48 on unseen words. |
| someone speaking | **not Laya.** A tiny head or a VAD. | Energy gate: AUC 0.986. 2-layer probe on the latents: 0.995. Laya: 0.990. |
| a word just ended | **not Laya.** A small head on the latents. | Probe 0.904, Laya 0.909. The label timing (80 ms ASR frames) is the ceiling, not the model. |
| any other question about the audio | **Laya's actual value** | The only structure here that answers arbitrary noul questions about audio with no new training per question. |

In short, the adapter is sound and fast, and it reads audio well enough to
beat a text Laya fed the *true* transcript (AUC 0.927). But keyword spotting
on its own is ASR's home turf. At strict false-alarm budgets, a streaming
recognizer plus string match remains more precise, especially for words the
adapter never saw. Speaking and word-end detection belong in a VAD and a
small head that share the encoder, not in a 400M-parameter decision model.

## What was built

```
16 kHz stream ──► last 3 s window, every 30 ms hop
                     │
                     ▼
     Qwen3-ASR AuT encoder (0.6B checkpoint's audio tower, FP16)   7.4 ms
       18 layers, d=896, block-windowed attention, 12.5 latents/s
       + its own projector ──► (39, 1024) latents per 3 s window
                     │
                     ▼
     projector MLP 1024 → 1024 → GELU → 1024 (trained, 2.1M params)  0.3 ms
                     │   soft rows, cast to FP16
                     ▼
     Laya (frozen): [CLS] noul question: <instructions> [SEP] [MASK] no [MASK] yes [SEP]
                    <39 soft rows in the state span> [SEP]
       one packed forward_items() over every registered question,
       all reading the same soft rows                       2.2 ms + ~0.7 ms/question
                     │
                     ▼
     per question: p = sigmoid((logit_yes − logit_no) / 1.9834)
```

The questions are ordinary Laya instructions:

- `The word "X" is spoken in this audio.`
- `Someone is speaking at the end of this audio.`
- `A word has just been spoken, ending at the end of this audio.`

A new keyword is a new string. There is no enrollment, no g2p and no per-word
model.

Code:

- **brotensor** `flash_attention_packed_qkv_backward` (`6d04375`): the
  adjoint of Laya's packed windowed encoder attention, with FD and
  FP32/FP16/BF16 parity tests.
- **brolm** (`512dbfa`):
  - `LayaItem::soft_pos/soft_count/soft_row` plus the `forward_items(items,
    soft)` soft-row path, scattered before the embedding LayerNorm. It is
    CUDA-graph-bucketed like the text path. Soft rows equal to the token
    embeddings reproduce the text logits exactly.
  - `LayaGrad`: the training twin. It runs the same packed forward with
    per-layer checkpoints, then backpropagates dLoss/dlogits through the
    scorer, head and every encoder layer to the soft rows. Checks:
    - `tests/test_laya_grad.cpp` against central differences on the FP32 CPU
      backend (relative error 0.4–3 %);
    - the LayerNorm invariances of the gradient.
- **brosoundml**:
  - `f04767e`: Parakeet token durations; `QwenAsr::load_encoder`.
  - `f311127`: an FP16 tensor-core AuT encoder path, 36 → 8 ms per 3 s
    window, 0.25 % mean relative distance from the FP32 latents. The FP32
    `transcribe` path is unchanged and its tests pass.
  - `abcd14b`, `815864c`: tools under `tools/laya_audio_*`:
    - `align`: word timings;
    - `cache`: window latents;
    - `train`: the projector through `LayaGrad`, or `--probe`;
    - `eval`: every method on the same probes;
    - `listen`: a `LayaListener` streaming hop by hop.

The product path and the training loop are C++ on brotensor; there is no
Python anywhere.

## Latency budget per 30 ms hop (RTX 4090, measured)

| stage | 1 keyword | 10 keywords | notes |
|---|---|---|---|
| AuT encoder, 3 s window, FP16 | 7.4 ms | 7.4 ms | FP32 was 36 ms (79 % in a scalar GEMM) |
| projector + casts | ~0.3 ms | ~0.3 ms | |
| Laya forward_items | 2.2 ms | 8.3 ms | ≈ 60 rows per question; one soft block shared by all questions |
| **hop total, p50 / p95** | **10.0 / 10.6 ms** | **16.3 / 16.9 ms** | `brosoundml_laya_audio_listen`, idle GPU |

Headroom: about 20 questions fit in a 30 ms hop on one GPU. On a GPU shared
with another job, the same hop measured 24–41 ms. A shipped listener needs
the GPU to itself, or a longer hop.

The encoder re-encodes the whole 3 s window every hop. That is 90 % wasted
work: the AuT encoder attends in 1 s blocks with a per-chunk conv stem, so a
block-streaming encoder (`QwenAsrStream` already does this for ASR) would
re-encode only the partial last block, about 3 ms. That is the obvious next
latency win.

For comparison, the ASR baseline costs 81 ms (whole stream so far) to 100 ms
(3 s window) per call. Parakeet-TDT here is an offline model re-run per
window, so this is *not* what a real streaming transducer would cost; treat
it as an accuracy reference, not a latency one.

## Data and training recipe

- **Audio:** LibriTTS-R (read audiobook speech, 24 kHz, resampled to 16 kHz
  with a windowed sinc).
  - Train: `train-clean-100`, 31,222 utterances, 247 speakers, 53 h.
  - Dev: `dev-clean`, 5,435 utterances, 40 **different** speakers.
- **Word timings:** Parakeet-TDT 0.6B runs over each whole utterance. Its
  token frames and TDT durations give hypothesis word spans, whose times are
  transferred onto the *reference* transcript by Levenshtein alignment
  (`align`). ASR word match is 97.8 % and 99.7 % of words are timed. Times
  are on an 80 ms grid.
- **Windows:** 3 random window ends per utterance, uniform in [0.4 s,
  duration + 0.8 s], so onsets, mid-speech and trailing silence are all
  covered. Early windows are zero-padded plus ±1e-4 dither. Latents are
  cached in FP16.
  - Train: 93,666 windows. Dev: 16,305.
- **Labels, from the alignment:**
  - keyword X: positive if an occurrence lies fully inside the window;
    ignored if one is cut by a window edge.
  - speaking: a word overlaps the last 0.3 s.
  - word_end: a word ended in the last 0.3 s.
- **Questions per window:**
  - 1 positive keyword, when present;
  - 1 hard negative: a vocabulary word within edit distance 2 of a word in
    the window;
  - 1 frequency-weighted random negative;
  - speaking and word_end.

  That is about 80 Laya items per 16-window batch.
- **Held-out vocabulary:** words hashing to 0 mod 10 are never asked in
  training. Together with dev words absent from train, they form the
  "unseen" set.
- **Optimisation:**
  - Projector with FP32 master weights, Adam, lr 3e-4, 200-step warmup,
    cosine schedule, BCE on logit_yes − logit_no.
  - Laya frozen in FP16 with dynamic loss scaling; no step was skipped.
  - 12,000 steps × 16 windows at 228 ms/step: 46 min on one 4090. The Laya
    forward is 49 ms and the backward 169 ms per step.

Dev keyword AUC during training: 0.87 at step 3k, 0.984 at 6k, 0.991 at 9k,
0.993 at 12k. It is still creeping up.

## Evaluation (dev-clean, unseen speakers)

The evaluation uses 1,500 random dev windows and 11,774 questions, the same
probes for every method (`brosoundml_laya_audio_eval`).

Methods:

- **laya_audio**: this adapter.
- **asr_stream**: Parakeet over the whole stream up to the window end, with
  the keyword among the words whose midpoint is in the window (string
  match).
- **asr_win**: Parakeet on the 3 s window alone.
- **asr_laya**: text Laya, same question, with the asr_stream words as the
  state. This is the "ASR text into text Laya" design.
- **oracle_laya**: text Laya on the *true* words of the window.
- **phoneme**: the open-vocabulary phoneme spotter (PhonemeNet posteriors
  plus a g2p-enrolled Viterbi template; score = best completion
  confidence).
- **probe**: a 2-layer MLP on the last two latent frames.
- **energy**: RMS of the last 0.3 s.

Keyword questions have 2,837 positives and 5,937 negatives; 514 positives
and 636 negatives are unseen.

| category | method | AUC | TPR @ 1 % FPR | TPR @ 5 % FPR |
|---|---|---|---|---|
| keyword, all | **laya_audio** | **0.9926** | 0.855 | **0.964** |
| | asr_stream | 0.9749 | **0.950** | 0.950 |
| | asr_win | 0.8801 | 0.760 | 0.760 |
| | asr_laya | 0.9144 | 0.323 | 0.631 |
| | oracle_laya | 0.9274 | 0.308 | 0.651 |
| | phoneme | 0.7268 | 0.122 | 0.208 |
| keyword, unseen words | **laya_audio** | **0.9694** | 0.484 | 0.693 |
| | asr_stream | 0.9494 | **0.899** | **0.899** |
| | phoneme | 0.7254 | 0.169 | 0.313 |
| keyword vs hard negatives | **laya_audio** | **0.9903** | 0.803 | 0.933 |
| | asr_stream | 0.9748 | **0.950** | **0.950** |
| | phoneme | 0.6262 | 0.068 | 0.187 |
| speaking | laya_audio | 0.9900 | 0.588 | 0.994 |
| | probe | **0.9950** | **0.885** | 0.984 |
| | energy | 0.9856 | 0.666 | 0.946 |
| word_end | laya_audio | 0.9090 | 0.318 | 0.586 |
| | probe | 0.9042 | 0.295 | 0.562 |

How to read this:

- **ASR string match is nearly binary.** Its AUC is capped by its misses:
  rare names such as "quox" (heard as "quax") and "krajiek's", and words cut
  at the start of a stream. When it does say yes, it is almost never wrong.
  The adapter ranks better overall but has a softer top end. It is weakest
  on unseen words: 0.48 recall at 1 % FPR against ASR's 0.90. The projector
  has learned acoustic-to-word matching mostly for the vocabulary it trained
  on.
- **Text Laya is not a keyword matcher.** Even given the true words,
  `The word "X" is spoken` gets 0.93 AUC, because Laya was never trained on
  that question. Putting ASR text into text Laya (0.91) is *worse* than
  plain string matching of the same text (0.975). "ASR → text Laya" is the
  wrong design for keywords. It is the right design for *semantic* questions
  about what was said.
- **The phoneme spotter** was built for keywords after a pause. In
  continuous read speech it is near chance against hard negatives (0.63).
- **Window-only ASR fails in a Parakeet-specific way.** 256 of 1,500
  windows holding at least 3 words came back empty. The short clips starting
  mid-speech, or ending in (even dithered) silence, break Parakeet's
  whole-clip feature normalization. Full-left-context ASR does not have this
  problem.
- **Answers are graded, not binary.** 33 % of calibrated answers fall
  between 0.1 and 0.9. On a stream, binariness has to come from a hold time.

### Streaming: word boundary timing vs "trigger immediately"

100 dev utterances plus 1 s of silence were pushed in 30 ms chunks. Each
utterance asked about 2 present keywords, 1 spelling neighbour and 4 random
absent words (`brosoundml_laya_audio_listen`). Delay is measured from the
aligned *end* of the word.

| operating point | recall | delay after word end (p10 / p50 / p90) | false alarms per keyword-hour |
|---|---|---|---|
| p ≥ 0.5, 1 hop | 0.990 | −0.43 / −0.18 / −0.01 s | 389 |
| p ≥ 0.7, 1 hop | 0.975 | −0.35 / −0.13 / +0.04 s | 247 |
| p ≥ 0.7, 5 hops (150 ms) | 0.940 | −0.20 / −0.00 / +0.16 s | 33 |
| p ≥ 0.9, 5 hops | 0.485 | −0.06 / +0.15 / +1.14 s | 1.9 |
| p ≥ 0.9, 10 hops | 0.450 | +0.10 / +0.32 / +1.29 s | 1.0 |

Speaking (p ≥ 0.5) turns on 90 ms after the first word starts (p50) and
releases 320 ms after the last word ends.

**Word boundary vs immediate.** At permissive thresholds the adapter fires
*before* the word has finished, with a median 130–180 ms early. It
recognises words from their first syllables and context. That is "trigger
immediately", but the price is false alarms on prefixes: "fire" when the
word is "fireplace". The training label says "fully inside the window", so
the model is being asked to anticipate. Requiring the answer to hold for
150 ms moves the median fire point to the word end, and cuts false alarms by
7×.

The rates show why a single keyword needs a stricter point. Reaching the
≤ 1 FA/hour class a wake word needs costs half the recall. Also, fewer than
3 false-alarm events underlie the 1.0–1.9/h figures, so they are indicative
only. Using the product safely means choosing per use:

- **Immediate:** p ≥ 0.7 held for 5 hops. About word end + 0 s, 94 %
  recall, tens of false alarms per keyword-hour. Fine for UI hints, ducking,
  or pre-loading.
- **Confirmed:** high threshold plus a long hold, or AND with asr_stream.
  Wake-word-grade false alarms, about +0.3 s and roughly half the recall
  from the adapter alone. Fusing it with ASR is the next thing to measure.

## Alternatives, compared honestly

1. **ASR → string match** (streaming recognizer). It is the most precise
   keyword detector at strict false-alarm budgets, and it handles unseen
   words best (it spells them). Costs:
   - it needs a true streaming transducer, which brosoundml does not have
     yet; Parakeet re-run per window is 80–100 ms;
   - misspelled rare names are misses by construction;
   - it answers only "was this string recognized".
2. **ASR → text Laya.** Strictly worse than string matching for keywords (see
   above). Keep it for semantic questions about *content*, where it will
   beat the audio adapter on anything that needs long context.
3. **Phoneme lattice / template spotter.** It is cheap (4 ms net per window)
   and needs no LLM, but it is weak in continuous speech (AUC 0.73). It is
   the right tool for command words after a pause, which is what it was
   tuned for.
4. **Dedicated heads.** Speaking and word-end need nothing near Laya's size:
   - a 2-layer MLP on two latent frames matches or beats Laya (0.995 / 0.904);
   - an energy gate is already at 0.986 on clean speech;
   - with noise, it is a VAD net's job.

   These heads share the encoder's latents for free, so they cost about zero
   per hop.
5. **The Laya adapter** is best at ranking keywords (AUC), best against hard
   negatives by AUC, and costs no per-word model. It is also the *only*
   option that answers arbitrary noul questions about the audio in the same
   pass. Its weak spots:
   - precision at strict operating points on unseen words;
   - a big frozen model per hop.

**Recommended product shape:** one shared AuT encoder per stream, feeding:

- small heads for speaking and word-end on every hop;
- the Laya adapter for registered keywords and questions, gated by the
  speaking head so Laya only runs while someone is talking;
- optionally, a streaming ASR as the confirming second stage for
  wake-word-grade keywords.

## C++ streaming API (shape)

The prototype `laya_audio::LayaListener` (tools/laya_audio_listener.h) is
what this would ship as, promoted to `include/brosoundml/laya_listener.h`:

```cpp
namespace brosoundml {
struct LayaListenerConfig {
    std::string encoder_dir;        // Qwen3-ASR (audio tower only is loaded)
    std::string laya_dir;           // any Laya checkpoint with the same hidden size
    std::string projector;          // trained projector for that checkpoint
    float window_s = 3.0f;
    int   hop_ms = 30;
};
struct LayaQuestionPolicy { float threshold = 0.7f; int hold_hops = 5; int refractory_ms = 1000; };
struct LayaHop {
    int64_t frame;                  // stream sample index at the window end
    float   speaking, word_end;     // dedicated heads, every hop
    std::vector<float> p;           // one calibrated probability per question
    std::vector<int>   fired;       // question ids whose policy fired this hop
};
class LayaListener {
public:
    void load(const LayaListenerConfig&, brotensor::Device);
    int  add_keyword(const std::string& word, LayaQuestionPolicy = {});
    int  add_question(const std::string& noul_instructions, LayaQuestionPolicy = {});
    void remove(int id);
    std::vector<LayaHop> feed(const float* pcm16k, int n);   // RT-safe caller: copy + enqueue
    void reset();
};
}
```

Threading follows bro's rules:

- `feed` from the audio thread only copies into a lock-free ring.
- A worker thread runs the hops.
- Results cross back through a snapshot, so no lock is ever taken on the RT
  thread.

Multiple streams share one encoder, one Laya and one projector (weights
once). Their hops batch into one `forward_items` call: the soft blocks
concatenate and the items name their block by `soft_row`.

## `bro.listen` attach (sketch; not shipped)

This mirrors the existing `stream.kws` / `stream.wake` views:

```js
const s = bro.listen.open({ mic: true });
const L = s.laya;                         // LayaStreamView, attached on first use
await L.load({ projector: 'models/laya-audio/proj.mlp' });
const ko = L.keyword('kokoro', { threshold: 0.7, holdHops: 5 });
const q  = L.ask('The speaker sounds annoyed.');   // any noul question
L.listen();
L.onfire = (e) => { /* e.id, e.p, e.frame, e.name */ };
function tick() { const snap = L.poll(); /* snap.p[ko], snap.speaking, snap.wordEnd */ }
```

`poll()` is lock-free, like `kws.progress_snapshot`. `onfire` is posted to
the JS thread. `feed(samples)` on the stream works headless for tests, as it
does for kws.

## Risks

- **Domain.** Everything here is clean read English audiobook speech
  (LibriTTS-R): no noise, far-field audio, music, overlapping talkers,
  spontaneous speech or accents beyond the corpus. The Qwen3-ASR encoder is
  robust to these, but the projector has never seen them. Expect the
  numbers to drop.
- **In-domain encoders.** Parakeet and Qwen3-ASR were very likely trained on
  LibriSpeech, the source of LibriTTS, so the ASR baselines, the encoder and
  the alignments are all in-domain. The *comparison* stays fair: every
  method shares that advantage.
- **Label timing.** Labels come from ASR word times on an 80 ms grid. The
  word-end task is bounded by that (both models about 0.90 AUC).
- **Unseen-word precision** is the adapter's weak point. More vocabulary in
  training (more hours; all of LibriTTS is 585 h) and hard negatives sampled
  across the whole vocabulary should help. LoRA on Laya's lower layers is
  the next lever if the projector alone saturates.
- **Graded answers.** A third of the answers sit between 0.1 and 0.9.
  Streams need hold-time policies, and per-keyword calibration is not done.
- **Cost of a frozen 400M model per hop.** It is fine on a desktop GPU (16 ms
  for 10 questions), not on a laptop iGPU, and CPU is out. Gate Laya on the
  speaking head.
- **Checkpoint coupling.** A projector is tied to one Laya checkpoint (the
  English one here). The multilingual checkpoint needs its own projector and
  its own data.

## What production needs

1. **Data:**
   - noise, room and music augmentation;
   - non-speech negatives;
   - conversational and far-field speech;
   - more vocabulary (all of LibriTTS-R, 585 h);
   - an LS-style test-clean/test-other split for reporting.
2. **Block-streaming AuT encoder:** re-encode only the partial 1 s block
   (about 3 ms instead of 7.4 ms), with CUDA-graph capture of the whole hop.
3. **Train on stream hops**, not random windows, with a label that encodes
   the chosen firing point. The hold-time policy then becomes the model's
   job.
4. **Calibration:** per-keyword thresholds, a speaking gate, and the ASR
   fusion experiment for wake-word-grade use.
5. **Promote `LayaListener`** to brosoundml's public API, with the RT-safe
   ring and multi-stream batching. Then add the `bro.listen` `laya` view in
   bro, whose binding is `brosoundml_api` per the sibling-API rules.
6. **Fix Parakeet's short-clip failure** (empty transcripts under
   whole-clip normalization) before it is used as a streaming baseline or
   product.

## Reproduce

Run from `D:/projects/brosoundml`, with `D=D:/projects/brosoundml-data/laya-audio`:

```bash
build-cuda/Release/brosoundml_laya_audio_align --manifest D:/datasets/libritts_r/manifest.json --subset train-clean-100 --out $D/align_train.tsv
build-cuda/Release/brosoundml_laya_audio_align --manifest D:/datasets/libritts_r/manifest.json --subset dev-clean --out $D/align_dev.tsv
build-cuda/Release/brosoundml_laya_audio_cache --align $D/align_train.tsv --out $D/train_w3_a.lac --per-utt 3 --limit 20000 --seed 21
build-cuda/Release/brosoundml_laya_audio_cache --align $D/align_train.tsv --out $D/train_w3_b.lac --per-utt 3 --start 20000 --seed 22
build-cuda/Release/brosoundml_laya_audio_cache --align $D/align_dev.tsv --out $D/dev_w3.lac --per-utt 3 --seed 11
build-cuda/Release/brosoundml_laya_audio_train --align-train $D/align_train.tsv --cache-train $D/train_w3_a.lac,$D/train_w3_b.lac \
    --align-dev $D/align_dev.tsv --cache-dev $D/dev_w3.lac --steps 12000 --eval-every 3000 --out $D/proj_ab.mlp
build-cuda/Release/brosoundml_laya_audio_train --probe ... --steps 3000 --lr 1e-3 --out $D/probe_head.mlp
build-cuda/Release/brosoundml_laya_audio_eval --align-train $D/align_train.tsv --align-dev $D/align_dev.tsv \
    --cache-dev $D/dev_w3.lac --proj $D/proj_ab.mlp --probe-head $D/probe_head.mlp
build-cuda/Release/brosoundml_laya_audio_listen --align-train $D/align_train.tsv --align-dev $D/align_dev.tsv --proj $D/proj_ab.mlp --utts 100
```

Laya weights come from `D:/projects/laya` (the English ModernBERT-large
checkpoint). The logs of the runs above are next to the data (`train_ab.log`,
`eval_ab.log`, `listen_ab.log`).
