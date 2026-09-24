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
| any other question about the audio | **Only with in-domain audio; otherwise ASR → text Laya** | Measured in [laya-audio-questions.md](laya-audio-questions.md). With in-domain training audio, intents, numbers and times reach real-label AUC 0.92–0.97, level with or above ASR → text Laya, and fire ~2 s before the utterance ends. On new question families it gets 0.58–0.85 against 0.83–0.99 for ASR → text. Tone of voice is at chance. |

In short, the adapter is sound and fast, and it reads audio well enough to
beat a text Laya fed the *true* transcript (AUC 0.927). But keyword spotting
on its own is ASR's home turf. At strict false-alarm budgets, a streaming
recognizer plus string match remains more precise, especially for words the
adapter never saw. Speaking and word-end detection belong in a VAD and a
small head that share the encoder, not in a 400M-parameter decision model.

**Update: broadened training data** (see [the section below](#broadened-real-data-training)).
The first adapter was trained on clean read English only. A second round
trained it on a real-data mix: meetings, parliament speech in four languages,
isolated keywords, reverberation, noise and music, all licensed for an
Apache-2.0 release. That generalises across domains and languages:

- far-field meetings: keyword AUC 0.955 → 0.977;
- German, Spanish, French: 0.73–0.81 → 0.95–0.97;
- noise and music: "someone is speaking" said yes on 67–76 % of windows,
  now under 1 %.

Clean English holds (0.993 → 0.992–0.993) and so does latency. What did not
improve is precision on words never asked in training. Recall at 1 % FPR on
a fixed held-out word set stays at 0.35–0.51 for English. More vocabulary
alone does not fix that.

**Update: free-form questions about speech** (full write-up with numbers:
[laya-audio-questions.md](laya-audio-questions.md)). The hypothesis was that
the adapter's distinctive strength is answering arbitrary questions about
live speech: intent, addressee, topic, sentiment, speech acts, entity
mentions and tone. It was tested with a 131-question bank that holds out
whole families and phrasings, three real-label sets and transcript
distillation from text Laya:

- **Zero-shot**, the keyword-trained projector barely understands such
  questions. Its AUC against the gold-transcript ceiling is 0.53–0.76, where
  ASR → text Laya gets 0.85–0.99.
- **Distillation** raises the adapter to 0.82–0.93 on trained questions,
  0.73–0.81 on new phrasings and 0.69–0.82 on new families, with no keyword,
  speaking or word-end regression. ASR → text is still ahead everywhere
  except far-field AMI, where they tie.
- **Real labels:**
  - Where training included in-domain audio (Timers and Such: timer, alarm,
    arithmetic, numbers, times), the adapter scores AUC 0.89–0.97, matching
    or beating ASR → text. It crosses 0.5 at a median 1.6–2.4 s *before* the
    speaker finishes.
  - Elsewhere it trails by 0.1–0.3.
- **Tone** questions are at chance for every method.
- **Latency:** 18 questions cost 15.5 ms p50 per 30 ms hop, against 73–150 ms
  per Parakeet call.

Positioning: an always-on pre-filter and early trigger for a known set of
intents trained with in-domain audio. It is not a general replacement for
ASR → text Laya.

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

## Broadened real-data training

The second round asks how the adapter generalises once it has seen more than
clean read English. It uses real recordings only (no synthetic or TTS
speech), and every source had to allow training and releasing the adapter
under Apache-2.0.

### Data

Licences were checked at each source. The release NOTICE, a per-file
attribution table for the MUSAN music and noise used, and the reasoning per
source are in `DATA_LICENSES.md`, `NOTICE` and `ATTRIBUTIONS.tsv`, next to
the adapter artifacts (`brosoundml-data/laya-audio2/`, with a copy in
`docs/laya-audio-release/`). The models the adapter depends on are all
Apache-2.0:

- Qwen3-ASR-0.6B;
- Laya and Laya multilingual (built on ModernBERT-large, Apache-2.0, and
  mmBERT-base, MIT).

The Parakeet-TDT-0.6B-v3 labels are CC BY 4.0.

| domain | source (licence) | taken | train | held out |
|---|---|---|---|---|
| read English | LibriTTS-R (CC BY 4.0) | train-clean-100 plus one augmented copy of each utterance | 53 h | dev-clean, 40 unseen speakers |
| meetings, far-field | AMI SDM Array1-01 (CC BY 4.0) | all 169 meetings; word annotations; 5–20 s chunks cut at pauses | 73.1 h | Kaldi dev (8.6 h) and eval (8.2 h) meetings |
| parliament speech | VoxPopuli ASR subset (CC0 transcripts; audio © EU, reuse authorised with acknowledgement) | en 2018; de/es/fr 2017–18; random train segments | en 37.8, de 29.0, es 23.7, fr 24.8 h | the corpus dev/test splits, 0.7–1.9 h each |
| isolated keywords | MSWC (CC BY 4.0, from Common Voice CC0) | en: 1,500 random words, 150 per bucket of 1, 2, 4 … 256 training clips, plus 150 with none. de/es/fr: 350 words each, buckets 4/16/64 plus 50 with none | 76,650 en + 3 × 8,400 clips | 8 clips per word from **speakers never in training** (1/8 of speakers) |
| room acoustics | OpenSLR 28 simulated RIRs (Apache-2.0) | 10 RIRs from each of 600 rooms | augmentation | – |
| noise | OpenSLR 28 point-source noises (PD) + MUSAN sound-bible (CC BY 3.0 / PD) | 929 files, cut into 12 s pieces | 5.0 h | 1.2 h (15 % of files, by name hash) |
| music | MUSAN music, per-file filtered to PD / CC BY, instrumental only | 261 tracks, 12 s pieces | 13.6 h | 3.0 h |

New audio totals **5.3 GB**, stored as Opus or FLAC. The archives were
streamed and cut on the fly, never stored whole. Dropped:

- **People's Speech.** Only parquet is still distributed, which needs Python
  to read. Its licence is per configuration, not per row, and the
  validation/test splits are shared between the CC BY and CC BY-SA
  configurations.
- **OpenSLR 28 real RIRs.** They bundle RWCP ("research and development use
  only") and AIR (no licence stated).
- **MUSAN music** under share-alike, NC or ND licences, or none.
- **One sound-bible file** with no licence entry.

MSWC's own splits share speakers between train and test: 45k of the 47k
English test speakers also appear in train. The split above is
speaker-disjoint instead.

Pipeline (`tools/laya_audio_fetch/`, bash + C++ only):

- `laya_audio_prep` parses the AMI word XML and cuts the chunks.
- `laya_audio_prep pcmselect` streams each VoxPopuli session and keeps only
  the annotated segments.
- Parakeet `align` supplies VoxPopuli word timings. ASR word match is 0.92
  for en, 0.90 for de, 0.95 for es and 0.93 for fr; 98–99 % of words are
  timed.
- MSWC word spans come from clip energy.
- Opus is decoded straight at 16 kHz (libopus).

**Augmentation** is applied to the whole utterance before windowing, so the
word timings still hold:

- an RIR convolution, aligned on the RIR's direct path and rescaled to the
  dry RMS;
- noise at 0–20 dB SNR;
- music at 5–20 dB SNR.

Each is drawn per utterance with p = 0.3–0.5 / 0.2–0.5 / 0.1–0.2 by domain;
AMI gets no extra reverberation. Parakeet re-aligns augmented audio to the
same word times.

**Non-speech negatives** are windows from the noise and music pieces. On
them "someone is speaking" and every keyword are *no*.

### Recipe

`train_mix.sh`: the same projector, loss and schedule as before, 24,000 steps
× 16 windows. Each window's domain is drawn by weight:

- LibriTTS 0.22, AMI 0.18;
- VoxPopuli en 0.14, de/es/fr 0.06 each;
- MSWC en 0.10, de/es/fr 0.03 each;
- noise 0.045, music 0.045.

Keyword negatives are drawn from the window's own language's vocabulary.
The same words hashing to 0 mod 10 are held out in every language.

**Which Laya checkpoint.** Both were trained on the identical mix:

- the English ModernBERT-large checkpoint (**en**), T = 1.9834;
- the multilingual mmBERT-base checkpoint (**mm**), T = 1.0.

mm trains at 133 ms/step against about 215 ms/step for en; the en run shared
its GPU with another job. Per hop, mm's Laya part costs 1.4 ms with one
keyword and 4.5 ms with ten, against 2.1 and 8.1 ms for en.

### Results per domain

`eval_mix.sh` runs every adapter on the same held-out windows with the same
probes: a fixed negative vocabulary (all training alignments of the mix) and
a fixed seed. **old** is the first adapter (English Laya, LibriTTS only).
Cells are AUC / TPR at 1 % FPR, keyword questions over all words. Each set
has 1,500 windows, except VoxPopuli test, which has 478–896.

| set | old | en | mm |
|---|---|---|---|
| LibriTTS dev-clean | **0.993** / **0.877** | **0.993** / 0.840 | 0.992 / 0.856 |
| AMI test (far-field meetings) | 0.955 / 0.640 | **0.977** / **0.687** | 0.971 / 0.643 |
| VoxPopuli en test | 0.976 / 0.698 | **0.982** / **0.784** | 0.978 / 0.739 |
| VoxPopuli de test | 0.729 / 0.106 | 0.958 / 0.509 | **0.959** / **0.515** |
| VoxPopuli es test | 0.808 / 0.168 | 0.963 / 0.605 | **0.967** / **0.653** |
| VoxPopuli fr test | 0.809 / 0.131 | 0.949 / 0.454 | **0.955** / **0.461** |
| MSWC en, unseen speakers | 0.927 / 0.334 | **0.962** / **0.473** | 0.960 / 0.465 |
| MSWC de / es / fr | 0.84 / 0.84 / 0.81 | 0.92 / 0.93 / 0.89 | 0.92 / 0.93 / 0.90 |

Speaking and word-end, AUC:

| set | speaking old → en / mm | word_end old → en / mm |
|---|---|---|
| LibriTTS dev-clean | 0.990 → 0.994 / 0.995 | 0.909 → 0.907 / 0.904 |
| AMI test | 0.948 → 0.975 / 0.974 | 0.840 → 0.914 / 0.911 |
| VoxPopuli en test | 0.925 → 0.955 / 0.943 | 0.844 → 0.863 / 0.848 |
| VoxPopuli de / es / fr | 0.89 / 0.87 / 0.92 → 0.91 / 0.90 / 0.93 (en), 0.91 / 0.87 / 0.94 (mm) | 0.82 / 0.81 / 0.84 → 0.85–0.87 |
| MSWC en | 0.951 → 0.990 / 0.990 | 0.934 → 0.981 / 0.981 |

**Non-speech negatives** (822 noise and 909 music windows):

| question | old | en | mm |
|---|---|---|---|
| speaking, p ≥ 0.5 on noise / music | **67 % / 76 %** | 0.9 % / 0.1 % | 0.9 % / 0.0 % |
| any keyword, p ≥ 0.5 on noise / music | 0.18 % / 0.17 % | 0 / 0 | 0.06 % / 0 |

Reading it:

- **The domain gap was real and training on it closes most of it.** AMI and
  VoxPopuli English gain 1–2 AUC points and 4–9 points of recall at 1 % FPR.
  The other languages go from barely working to about 0.96 AUC.
- **Clean LibriTTS is not hurt in AUC.** It gives up 2–4 points of strict
  recall. It now has 22 % of the training windows instead of all of them.
- **The first adapter thought all sound was speech.** On noise and music it
  answered "someone is speaking" in two thirds of windows, because it had
  never heard anything else. Non-speech negatives fix that completely.
- **en vs mm.** en is ahead on the English far-field sets: AMI 0.977 vs
  0.971, and +4.5 points of strict recall. mm is ahead on the other
  languages, and clearly so on held-out words (below). The gaps are small
  next to the gains over old.

### Trained vs never-seen words

"Seen" depends on each adapter's own training vocabulary, so the fair
comparison is a fixed word set. Words hashing to 0 mod 10 are never asked in
*any* training run. The table gives their recall at the threshold that
passes 1 % of all keyword negatives, next to words the adapter was trained
to answer.

| set | trained words: old / en / mm | held-out words: old / en / mm |
|---|---|---|
| LibriTTS dev-clean | 0.96 / 0.92 / 0.93 | **0.44** / 0.42 / 0.41 |
| AMI test | 0.71 / **0.73** / 0.69 | **0.43** / 0.42 / 0.35 |
| VoxPopuli en | 0.78 / **0.85** / 0.81 | 0.40 / **0.51** / 0.45 |
| VoxPopuli de | 0.07 / 0.53 / 0.53 | 0.09 / 0.36 / **0.38** |
| VoxPopuli es | 0.16 / 0.61 / **0.65** | 0.25 / 0.52 / **0.72** |
| VoxPopuli fr | 0.22 / 0.48 / 0.47 | 0.12 / 0.21 / **0.42** |
| MSWC en | 0.34 / **0.48** / 0.47 | 0.25 / 0.38 / **0.44** |

**The open-vocabulary gap did not close.** The broader vocabulary leaves
held-out English recall where it was: 0.35–0.51. English words asked in
training rose from 13.9k to 19.6k. Trained words are recalled about twice as
often. The projector still learns words, not a general acoustic-to-spelling
map. Two signs point at the language model's side as the lever:

- mm, whose text side is smaller but multilingual, holds up better on unseen
  words in de/es/fr;
- LoRA on Laya's lower layers was the next lever named in the first round.

### Examples-per-word curve

Recall at the all-negatives 1 % / 0.1 % FPR threshold, against how many
training utterances contain the word.

**LibriTTS dev-clean** (natural frequency, counts over each adapter's own
training data):

| training examples | old | en | mm |
|---|---|---|---|
| 0 (never asked) | 0.50 / 0.21 | 0.47 / 0.22 | 0.49 / 0.17 |
| 1 | 0.83 / 0.26 | 0.74 / 0.21 | 0.79 / 0.18 |
| 2–3 | 0.73 / 0.18 | 0.75 / 0.09 | 0.70 / 0.16 |
| 4–7 | 0.80 / 0.22 | 0.85 / 0.28 | 0.79 / 0.23 |
| 8–15 | 0.83 / 0.26 | 0.76 / 0.22 | 0.76 / 0.20 |
| 16–31 | 0.92 / 0.33 | 0.80 / 0.27 | 0.72 / 0.24 |
| 32–63 | 0.89 / 0.49 | 0.89 / 0.28 | 0.80 / 0.18 |
| 64–127 | 0.95 / 0.57 | 0.86 / 0.38 | 0.85 / 0.34 |
| 128–255 | 0.96 / 0.64 | 0.91 / 0.43 | 0.87 / 0.41 |
| 256+ | 0.98 / 0.86 | 0.97 / 0.76 | 0.95 / 0.68 |

Word frequency confounds this curve: frequent words are also short and
easy. **MSWC controls for it.** Its words were drawn at random from one
eligible pool and assigned a training budget at random. Each is tested on
speakers never in training, with counts over the MSWC training clips only.
Recall at 1 % FPR:

| MSWC training clips of the word | 0 | 1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64–127 | 128–255 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|
| old (no MSWC in training) | 0.29 | 0.29 | 0.31 | 0.29 | 0.34 | 0.32 | 0.32 | 0.26 | 0.32 | 0.32 |
| en | 0.39 | 0.47 | 0.44 | 0.42 | 0.45 | 0.46 | 0.37 | 0.47 | 0.54 | 0.50 |
| mm | 0.34 | 0.35 | 0.39 | 0.33 | 0.41 | 0.37 | 0.36 | 0.52 | 0.48 | 0.51 |

Each cell holds 124–250 positives. With frequency controlled, the
per-word effect is modest: +0.1 to +0.15 from none to 64 or more, and the
first 30 clips buy little. Most of the gain over old, +0.1 to +0.2 in every
column including zero, comes from having heard *the clip domain at all*: a
word just finished, then silence. It does not come from having heard *that
word*. The de/es/fr buckets (4/16/64 clips) show the same shape for mm:
0.21 → 0.28 → 0.30 in de, 0.15 → 0.20 → 0.20 in fr.

### Early fire: boop / unboop

A "boop" fires on the first hop whose probability crosses a single-hop
threshold. It is retracted ("unboop") if the run above threshold ends before
the 150 ms hold (5 hops) that would confirm it. The sweep (`listen_mix.sh`)
streams 100 utterances per set, with +1 s of silence, in 30 ms hops. Each
utterance asks 2 present keywords, a spelling neighbour and 4 random absent
words.

- **fire** is the median fire time relative to the aligned word end
  (negative = before the word has finished).
- **false/kh** counts false boops per keyword-hour of speech.
- **false retract** is the fraction of false boops the hold would retract;
  **true retract** is the same for true boops.

The probability scales differ: en and old use T = 1.9834, mm uses T = 1.0,
which makes it sharper. Compare rows at similar false-boop rates, not at
the same threshold.

**mm, the recommended adapter.** Each cell is recall / fire / false per
keyword-hour / false retract / true retract:

| thr | LibriTTS dev-clean | AMI test | VoxPopuli en test |
|---|---|---|---|
| 0.50 | 0.975 / −0.15 s / 351 / 0.80 / 0.16 | 0.940 / −0.09 s / 975 / 0.84 / 0.40 | 0.940 / −0.21 s / 499 / 0.83 / 0.26 |
| 0.70 | 0.965 / −0.12 s / 282 / 0.83 / 0.18 | 0.900 / −0.05 s / 644 / 0.84 / 0.37 | 0.925 / −0.16 s / 318 / 0.85 / 0.22 |
| 0.90 | 0.925 / −0.08 s / 127 / 0.83 / 0.29 | 0.810 / +0.01 s / 252 / 0.88 / 0.53 | 0.875 / −0.11 s / 116 / 0.80 / 0.41 |
| 0.95 | 0.880 / −0.04 s / 76 / 0.78 / 0.43 | 0.710 / +0.04 s / 136 / 0.85 / 0.61 | 0.805 / −0.01 s / 75 / 0.88 / 0.53 |
| 0.98 | 0.700 / +0.05 s / 37 / 0.95 / 0.61 | 0.500 / +0.22 s / 47 / 0.92 / 0.71 | 0.555 / +0.07 s / 7.4 / 0.91 / 0.67 |
| 0.99 | 0.505 / +0.18 s / 1.0 / 1.00 / 0.74 | 0.265 / +0.26 s / 11 / 1.00 / 0.70 | 0.270 / +0.13 s / 0 / – / 0.70 |

**All three adapters at matched false-boop budgets.** Each cell is the best
single-hop recall with at most 40 (or 10) false boops per keyword-hour, and
the threshold that gives it:

| set | budget | old | en | mm |
|---|---|---|---|---|
| LibriTTS dev-clean | ≤ 40 / kh | 0.675 (0.90) | 0.675 (0.90) | **0.700** (0.98) |
| | ≤ 10 / kh | 0.425 (0.95) | 0.415 (0.95) | **0.505** (0.99) |
| AMI test | ≤ 40 / kh | **0.415** (0.90) | 0.370 (0.90) | 0.265 (0.99) |
| | ≤ 10 / kh | 0.225 (0.95) | **0.370** (0.90)¹ | 0.055 (0.995) |
| VoxPopuli en test | ≤ 40 / kh | 0.345 (0.95) | **0.555** (0.90) | **0.555** (0.98) |
| | ≤ 10 / kh | 0.155 (0.99) | 0.155 (0.98) | **0.555** (0.98) |

¹ en's AMI row is not monotone: 5.1/kh at 0.90 but 6.5 at 0.95. These
budgets rest on 1–2.2 absent-keyword hours per set, so a cell at ≤ 10/kh
stands on about 2–20 false boops. Treat differences under about 0.1 as
noise.

What the sweep says:

- **Boops come early, then the hold decides.** At permissive thresholds
  (0.5–0.7) a boop fires 0.1–0.2 s *before* the word ends; the adapter
  recognises words from their first syllables. The 150 ms hold retracts
  78–88 % of false boops there, and 90–100 % above 0.98. It also retracts
  15–45 % of true ones at 0.5–0.7, rising to 45–95 % at 0.95–0.99. A
  retracted true boop is a run above threshold shorter than 150 ms; most
  are prefixes that flickered and fired again later, not misses.
- **"Boop now, unboop if the hold fails" is a usable UI contract.** mm at
  0.9 fires at −0.08 s on LibriTTS with 93 % recall. About 83 % of its 127
  false boops per keyword-hour are withdrawn within 150 ms, and a confirmed
  fire lands at +0.06 s. On far-field AMI the same point gives 81 % recall
  and 252 false boops, of which 88 % are withdrawn.
- **Strict single-hop budgets are still the weak end.** Below 10 false
  boops per keyword-hour, mm keeps 0.51–0.56 recall on clean and
  parliament speech; old and en keep 0.16–0.43. On AMI every adapter is
  at 0.05–0.37, on too few events to rank them.
- **Non-speech is quiet.** On the held-out noise and music, every adapter
  stays at or under 1 false boop per keyword-hour from threshold 0.7 up,
  and at 0 from 0.8 up. At 0.5 the counts are old 1.8 / 24, en 8.9 / 1.5
  and mm 1.8 / 0.8 (noise / music). The real change is the speaking
  question on full-window non-speech hops:
  - old said yes on 82 % (noise) and 85 % (music);
  - en says yes on 0.3 % and 0.7 %;
  - mm says yes on 0.35 % and 0.9 %.

  Speaking onset (p50 0.01–0.09 s) and release (0.3 s after the last word)
  are unchanged.

### Latency

Per 30 ms hop on an idle RTX 4090 (the other GPU was busy with another
session's job), 195 hops of a LibriTTS stream, p50 / p95:

| adapter | 1 keyword | 10 keywords | encode + project | Laya, 1 / 10 keywords |
|---|---|---|---|---|
| old (English Laya) | 10.0 / 11.5 ms | 16.0 / 17.5 ms | 7.8 ms | 2.1 / 8.1 ms |
| en (English Laya) | 9.9 / 11.5 ms | 15.8 / 17.4 ms | 7.8 ms | 2.1 / 8.1 ms |
| mm (multilingual) | **9.2 / 10.8 ms** | **12.5 / 14.1 ms** | 7.7 ms | **1.4 / 4.5 ms** |

Retraining does not change the cost; the projector is the same size. mm
is 1.3× faster at ten keywords.

### Which checkpoint to ship

**mm, with en as an English-only option.** mm:

- is within 0.006 AUC of en on every English set;
- is better on the three other languages, and clearly better on their
  unseen words;
- costs 21 % less per hop at ten keywords and trains 1.6× faster.

In streaming, mm also holds the most recall at strict false-boop budgets
on clean and parliament speech. en keeps a real edge on English far-field
speech: about +4 points of strict recall on AMI and VoxPopuli en in the
window eval, and the better strict streaming points on AMI. That is a reason to ship it too for
English-only products. Both projectors are 2.1M parameters.

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

- **Domain.** The first adapter saw only clean read English, and dropped
  exactly as expected elsewhere. The broadened mix (above) covers meetings,
  parliament speech in four languages, reverberation, noise and music.
  Still uncovered:
  - children's and accented conversational speech;
  - phone-band audio;
  - languages beyond en/de/es/fr;
  - speech over loud music. Music negatives are instrumental only, so
    sung words are untested.
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
- **Checkpoint coupling.** A projector is tied to one Laya checkpoint. The
  broadened round trained one per checkpoint (`proj_en.mlp`, `proj_mm.mlp`)
  on the same mix.

## What production needs

1. **Data.** Done in the broadened round:
   - noise, room and music augmentation;
   - non-speech negatives;
   - conversational and far-field speech;
   - three more languages.

   Still open:
   - unseen-word precision (LoRA on Laya's lower layers, or
     spelling-aware negatives);
   - keeping VoxPopuli segments that abut each other (the cutter drops
     them, which cost 24–37 % of the selected hours);
   - the rest of LibriTTS-R (585 h).
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
build-cuda/Release/brosoundml_laya_audio_eval --vocab-align $D/align_train.tsv --train-align $D/align_train.tsv \
    --set libri-dev,$D/align_dev.tsv,$D/dev_w3.lac --proj $D/proj_ab.mlp --temperature 1.9834 --asr --probe-head $D/probe_head.mlp
build-cuda/Release/brosoundml_laya_audio_listen --vocab-align $D/align_train.tsv --stream $D/align_dev.tsv \
    --proj $D/proj_ab.mlp --temperature 1.9834 --utts 100
```

Laya weights come from `D:/projects/laya` (the English ModernBERT-large
checkpoint). The logs of the runs above are next to the data (`train_ab.log`,
`eval_ab.log`, `listen_ab.log`).

The broadened round (outputs in `D:/projects/brosoundml-data/laya-audio2`):

```bash
F=tools/laya_audio_fetch
bash $F/fetch_rirs_noises.sh; bash $F/fetch_musan.sh; bash $F/musan_filter_licenses.sh
bash $F/fetch_ami.sh D:/datasets/ami 12
bash $F/fetch_voxpopuli.sh en "2018" 50        # de / es / fr: "2017 2018" 40
bash $F/fetch_mswc.sh en 150 "1 2 4 8 16 32 64 128 256" 150 8
bash $F/fetch_mswc.sh de 100 "4 16 64" 50 8    # and es, fr
bash $F/build_mix.sh nonspeech; bash $F/build_mix.sh libri; bash $F/build_mix.sh ami
for l in en de es fr; do bash $F/build_mix.sh vp $l; bash $F/build_mix.sh mswc $l; done
bash $F/attributions.sh $A/aug_music.list $A/aug_noise.list > $A/ATTRIBUTIONS.tsv
bash $F/train_mix.sh mm D:/projects/laya/multilingual; bash $F/train_mix.sh en D:/projects/laya
bash $F/eval_mix.sh mm $A/proj_mm.mlp D:/projects/laya/multilingual 1.0 <train aligns>   # likewise en, old
bash $F/listen_mix.sh mm $A/proj_mm.mlp D:/projects/laya/multilingual 1.0
```

The logs are `train_{mm,en}.log`, `eval_{old,en,mm}{,_libri,_mswc}.log`,
`listen_{old,en,mm}.log` and `timing.log`.
