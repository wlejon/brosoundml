# Phoneme spotter (open-vocabulary KWS)

An **open-vocabulary** streaming keyword spotter: enroll a keyword by typing it
(or from a reference clip) — no per-keyword training, no retrained logit. It is
the open-vocabulary analogue of the [wake-word](wake-word.md) model. Where
WakeWord fires one trained-in keyword logit, the phoneme spotter runs a small
per-frame phoneme-class model (**PhonemeNet**) and a streaming template matcher
(**PhonemeSpotter**) that scores enrolled phoneme-sequence templates against the
posterior stream. This is the "type a word, spot it" path. PhonemeNet is
device-neutral (CPU + CUDA).

Public surface: `include/brosoundml/phoneme_model.h` (PhonemeNet — the 2D
BC-ResNet posterior model), `include/brosoundml/phoneme_spotter.h`
(PhonemeSpotter — the matcher / runtime), `include/brosoundml/phoneme_data.h`
(the class map, frame-label builder, and the BPDS dataset format).

## PhonemeNet — per-frame phoneme posteriors

PhonemeNet shares the 2D BC-ResNet trunk of `bc_resnet2d` (the single-logit wake
model) verbatim — a stem conv2d → per-stage transition + Broadcasted-Residual
blocks (each BC block = `f2` freq-depthwise + BN + ReLU, `f1` freq-avg-pool →
causal temporal depthwise + BN → broadcast-over-freq, pointwise mix, residual
add, ReLU) — but swaps the head and the loss:

- **Head.** Instead of `bc_resnet2d`'s global-average-pool to one logit per
  clip, PhonemeNet pools over frequency only, then applies a per-frame
  `Linear(C_last → K)` at every 10 ms frame, emitting a `(T, K)` tensor — one
  K-way phoneme-class posterior per frame. No temporal pool, no GAP ring, no
  streaming warm-up: every frame is an independent classification over the causal
  receptive field ending at that frame, so streaming is exactly the trunk's
  per-conv causal time-ring caches with a pointwise-in-time head.
- **Loss.** Framewise softmax cross-entropy (the mean over frames of
  softmax-CE(logits_t, label_t)), composed from brotensor's
  `softmax_xent_fused_batched`. A per-class weight vector and silence handling
  (a class weight of 0 on the silence class drops those frames) keep silence
  from dominating.

The trunk is a 4-stage recipe (one more stage than the wake model) at ~2× the
channel width, lifting the causal receptive field to ~1 s. The checkpoint
(`'BPM1'`) carries the front-end framing params, the trunk hyperparameters, the
embedded `PhonemeClassMap`, and the weights — K (the output dimension) comes from
the class map.

## PhonemeSpotter — the streaming matcher

PhonemeSpotter turns PhonemeNet's per-frame K-way posteriors into detections
against enrolled phoneme-sequence templates. It scores a streaming Viterbi
alignment of each enrolled template (a sequence of distinct adjacent phoneme
classes) against the posterior stream and fires when an alignment completes with
a high-enough geometric-mean posterior, gated by an entry-silence word boundary,
an M-of-N smoother, and a per-template refractory debounce. An `emission_floor`
bounds the per-frame veto so a single unreliable phoneme (a stop burst, a glide)
can't multiplicatively crush a citation template's score to zero.

Two seams, mirroring WakeWord but split for testability:

- **REAL path** — `load()` the PhonemeNet checkpoint (+ its embedded class map),
  enroll templates from phonemizer ids / class ids / reference audio, then push
  mic PCM through `feed()`. Internally `feed()` runs the PCEN mel front-end →
  `forward_streaming` → per-frame softmax → `feed_posteriors()`.
- **OFFLINE / TEST path** — `set_class_map()` installs an inventory with no model
  on disk, enabling `enroll()` + `feed_posteriors()` so the whole matcher can be
  unit-tested on synthetic posterior streams with zero weights / DSP / GPU.

Thread-safety: single-producer. `feed()` / `feed_posteriors()` / `enroll()` /
`remove()` / `clear()` / `reset()` all run on one thread (the audio thread); the
scalar cross-thread readers (`prefix_progress()`, `last_posterior()`) are
lock-free (relaxed atomics / a seqlock) for a UI thread to poll. No locks
anywhere.

## Data layer

`phoneme_data.h` is the phoneme analogue of `wake_data.h`, all pure host DSP/IO:

- **`PhonemeClassMap`** — the model's output inventory, a partition of the Kokoro
  phoneme-id space into K classes (class 0 == silence). Punctuation / whitespace
  / pause ids route into silence; suprasegmental modifiers (stress, length, …)
  are flagged *transparent* so frame labelling inherits the adjacent segmental
  class instead of punching spurious silence gaps.
- **`build_frame_labels`** — turns Kokoro's per-phoneme duration vector into a
  per-10 ms-frame class-label track aligned to the log-mel framing.
- **BPDS** — a packed binary frame-labelled dataset (PCM + per-frame labels + the
  embedded class map), written incrementally and read back with full validation,
  plus a validator mirroring wake_data's. The augmentation / resample helpers are
  reused from `wake_data.h`, not duplicated.

## Training toolchain

| Tool | Role |
|---|---|
| `brosoundml_phoneme_synth` | render a sentence corpus through Kokoro across many voices into one BPDS file (frame labels free from Kokoro's predicted durations) |
| `brosoundml_phoneme_align` | turn real (audio, transcript) speech into a BPDS shard via model-based constrained forced alignment |
| `brosoundml_phoneme_aug` | waveform-domain augmentation (additive noise at sampled SNR, room IRs, gain) — BPDS in → BPDS out, labels unchanged |
| `brosoundml_phoneme_melcache` | precompute the PCEN-mel front-end for a shard into a BPMC cache (skip the per-run recompute) |
| `brosoundml_phoneme_inspect` | validate a BPDS dataset (a drop-in pre-training CI gate) |
| `brosoundml_phoneme_train` | PhonemeNet trainer (Adam + framewise softmax-CE), held-out eval, fused-BN checkpoint out |
| `brosoundml_phoneme_test` | frame-level eval (frame accuracy, non-silence accuracy, per-class recall/precision, K×K confusion) |
| `brosoundml_phoneme_calibrate` | open-vocab KWS false-accept / false-reject sweep on real human speech |

`tools/prep_kws_corpus.py` decodes a transcribed speech corpus (LibriSpeech,
VCTK) into the aligner's 16 kHz PCM16 + TSV-manifest form for
`phoneme_align --manifest`.

## brotensor op coverage

Every layer uses brotensor's `conv2d` / `batch_norm` / `relu` / `linear`
forward+backward; the framewise softmax-CE head is composed from
`softmax_xent_fused_batched`. The front-end is the PCEN mel (`stft` +
`complex_abs` + mel matmul + PCEN). No op brotensor lacks; no hand-rolled
backward.
