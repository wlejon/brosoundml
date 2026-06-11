# Kokoro-82M

Kokoro-82M is an 82M-parameter text-to-speech model derived from **StyleTTS 2**
(24 kHz output). Unlike StyleTTS 2 it does not sample a style with a diffusion
model — the "voice" is a precomputed embedding (a *voice pack*), so synthesis is
a single deterministic non-autoregressive forward pass. Device-neutral: `load()`
takes a `brotensor::Device`, and the whole forward pass dispatches through
brotensor device ops, so a CUDA build runs on the GPU.

Public surface: `include/brosoundml/kokoro.h` (the `Kokoro` pipeline,
`KokoroConfig`, `Voice`), `include/brosoundml/kokoro_modules.h` (the
Kokoro-specific module graph).

## Pipeline

```
text ──▶ [G2P]──▶ phonemes ──▶ [token ids]
                                    │
                    ┌───────────────┘
                    ▼
   1. plBERT          phoneme-level BERT (ALBERT-style, weight-shared layers)
                      encodes the phoneme sequence into context features.
   2. Text encoder    embedding ▶ 3× (conv1d + LayerNorm + LeakyReLU)
                      ▶ bidirectional LSTM ▶ per-phoneme features.
   3. Predictor       conditioned on the voice embedding: a duration
                      predictor (LSTM + projection) gives per-phoneme frame
                      counts; a length regulator expands features to frame
                      rate; F0 (pitch) and energy are predicted at frame rate.
   4. Decoder         iSTFTNet generator: AdaIN residual blocks + transposed-
                      conv upsampling + a harmonic/noise source excitation,
                      a final layer emitting an STFT magnitude/phase pair,
                      and an iSTFT head ─▶ 24 kHz waveform.
```

## G2P

Kokoro upstream uses the [misaki](https://github.com/hexgrad/misaki) (Apache
2.0) grapheme-to-phoneme frontend. brosoundml ships an in-tree English G2P at
`brosoundml::g2p::` that removes that runtime dependency for embedded /
no-Python deployments — a byte-level Transformer POS tagger, a packed lexicon,
a morphology fallback, special-case overrides, a text normalizer, and the
Kokoro phoneme-id adapter, tied together by `Phonemizer` into a
`sentence → vector<int32_t>` entry point. `Kokoro::synthesize()` still accepts
phoneme ids directly for callers that supply their own. See [g2p.md](g2p.md).

## Streaming

Kokoro is a single non-autoregressive forward pass, so there is no growing token
prefix to stream the way Qwen3-TTS does. `synthesize_stream()` instead chunks
the *input*: the caller passes phoneme chunks split at sentence/clause
boundaries, each is synthesized independently and its 24 kHz audio is handed to
an `on_chunk` sink the moment it finishes (first-chunk latency for a long
script), with the full concatenation also returned. Each chunk reports its own
per-phoneme frame durations so a caller can align words to the streamed audio.
The `synth --stream` CLI drives this, one `--ids-file` line per chunk.

## Authoring, introspection, and a trainable decoder

A handful of seams open Kokoro up beyond one-shot synthesis:

- **`make_voice()`** builds a `Voice` from raw style data in memory (rather than
  a voice-pack file) — for authoring, blending, or otherwise constructing a
  voice the application holds itself.
- **`decode_from()`** re-runs only the decoder from the intermediate stages of a
  prior synthesis (the `asr` / `F0_pred` / `N_pred` tensors), so an edited
  pitch/energy curve can be re-rendered without re-running the front half.
- **`KokoroTrace`** — passing a `KokoroTrace*` to `synthesize()` fills a
  per-stage host copy of the intermediates for introspection / visualization. A
  normal `synthesize()` with no trace requested pays nothing.
- **Trainable decoder LoRA** (`decoder_lora.h`). The iSTFTNet decoder's AdaIN
  style→(γ,β) projections are made trainable: a backward pass over the decoder
  back half (`kokoro_decoder_backward.{h,cpp}`) plus a conditioned LoRA
  (`DecoderLora`) with Adam, checkpoint I/O, and a conditioning gate that is
  exactly identity at condition 0 (so cond = 0 reproduces the base voice). The
  condition vector is generic — any small control signal (e.g. a style/affect
  coordinate) can drive it.

## brotensor op coverage

Most of Kokoro maps straight onto the existing op surface:

| Kokoro component | brotensor ops |
|---|---|
| Phoneme embedding, plBERT | `embedding_lookup`, `flash_attention_forward`, `layernorm`, `gelu` |
| Text encoder CNN | `conv1d`, `pad1d`, `layernorm`, `leaky_relu` |
| iSTFTNet decoder | `conv1d`, `conv_transpose1d`, `leaky_relu`, `snake`, `group_norm` (instance norm via `num_groups == C`) |
| iSTFT head | `istft` (and `stft` / `complex_*` for the magnitude/phase pair) |

F0 / harmonic-source upsampling is host-side nearest-neighbour plus a depthwise
`conv_transpose1d` (an all-ones-weight 2× upsample), not a resampling op.

Two compositions carry the recurrent and style-conditioned parts:

**LSTM is composed, not a brotensor op — but the composition is fast.** The text
encoder and both predictors are bidirectional-LSTM-based and brotensor has no
recurrent inference primitive, so `brosoundml::LSTM` / `BiLSTM`
(`modules.{h,cpp}`) compose the cell from brotensor ops. The input-side
projection for the whole sequence is hoisted into one batched GEMM; on CUDA the
remaining per-step recurrent body is captured as a CUDA graph once per cell
(lazily, cached on the module via `LstmGraphPlan`) and replayed per timestep —
the same launch-overhead fix as the Qwen Talker decode step. A step is then a
staging copy in, one graph launch, and a row copy out.

**The AdaIN/AdaLN affine rides the norm op directly.** `group_norm_forward`'s
gamma/beta are per-channel and `layernorm_forward_inference_batched`'s are
per-feature, so the style-conditioned affine is passed straight into the norm
call: `ada_in_1d_styled` / `ada_layernorm` (`kokoro_modules.cpp`) hand the norm
`(1 + gamma)` / `beta`. One fused pass — no `nchw_to_sequence` / `modulate`
transpose chain, no unit-gamma upload.

## Caveat

The harmonic-source branch (SineGen / SourceModuleHnNSF in upstream Kokoro)
uses a deterministic approximation that drops torch's random initial phases and
additive noise. Output is intelligible but not bit-equal to upstream.

## Tools

- `brosoundml_synth` — text → WAV (also `--stream`, `--ids-file`).
- `brosoundml_kokoro_bench` — throughput benchmark (in-tree G2P, no Python):
  per-iteration wall time and the realtime factor. Set
  `BROSOUNDML_KOKORO_PROFILE=1` for the library's per-stage breakdown.

Weights are converted from the upstream checkpoint by `scripts/convert-kokoro.py`
and fetched by `scripts/download-kokoro.sh`.
