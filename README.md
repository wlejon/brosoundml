# brosoundml

Audio-ML model inference. brosoundml is the **expression layer** for neural
audio models — it composes the FP32 audio op family in `brotensor` (FFT/STFT,
1D convolution, vocoder/codec activations, codec quantization, resampling,
autoregressive sampling) into runnable text-to-speech, speech-to-text,
neural-codec, neural-autoencoder, and wake-word models.

It is to audio what [`brodiffusion`](https://github.com/wlejon/brodiffusion) is to images and
[`brolm`](https://github.com/wlejon/brolm) is to text: a sibling library that turns a tensor op
surface into a model. One flat namespace, `brosoundml::`.

Models implemented (all complete, CPU FP32; CUDA where noted):

- **Kokoro-82M** — text-to-speech (StyleTTS 2 derivative, 24 kHz output).
- **Qwen3-TTS** — text-to-speech (12 Hz multi-codebook, end-to-end discrete
  token, 24 kHz output; CustomVoice presets, VoiceDesign instruct prompts, and
  Base-variant zero-shot voice cloning). Device-neutral CPU + CUDA.
- **Whisper** — speech-to-text (HF transformers checkpoints, tiny → large-v3).
- **RAVE** — neural audio autoencoder (ACIDS/IRCAM v2): a waveform ⇄ low-rate
  PCA-sorted latent that the lab edits per-dimension. Device-neutral (CPU /
  CUDA / Metal); deterministic encode/decode plus the optional stochastic
  noise-synth and per-channel stereo decode.
- **Wake-word** — a small BC-ResNet streaming keyword spotter for an always-on
  mic loop, plus its Kokoro-driven training toolchain.

brosoundml also ships an in-tree English **G2P** (`brosoundml::g2p::`) so Kokoro
can phonemize text with no misaki/Python dependency.

CLI tools drive each model end-to-end — `brosoundml_synth` (Kokoro),
`brosoundml_transcribe` (Whisper), the `brosoundml_qwen_tts_*` tools
(bench / roundtrip / clone), `brosoundml_build_speaker_encoder` (pack the
standalone Qwen voice-clone enroller), and the `brosoundml_wake_*` toolchain
(synth / inspect / train / test / probe / melcmp). RAVE is a library-only model
(no CLI driver), exercised by `test_rave`.

## Dependencies

brosoundml is a standalone sibling repo. It links three siblings and ships no
GPU kernels of its own — GPU work happens inside `brotensor`.

| Library | Role |
|---|---|
| [`bromath`](https://github.com/wlejon/bromath) | header-only math (Vec/Quat/Mat, easing) |
| [`brotensor`](https://github.com/wlejon/brotensor) | the unified `Tensor` + device-neutral op surface (including the audio op family) |
| [`brolm`](https://github.com/wlejon/brolm) | tokenizers used by the speech models (`brolm::whisper::Tokenizer`, the Qwen BPE tokenizer) |

## Data and weights

brosoundml ships **code only** — no trained weights, no packed data, no
voice packs are checked into this repo. Anything that gets built (POS tagger
weights, the packed English lexicon, Kokoro voice packs, wake-word
checkpoints, …) lives in the sibling
[`brosoundml-data`](https://huggingface.co/datasets/wlejon/brosoundml-data) repo. Loaders take file paths;
the application (or the CLI tools in this repo) is responsible for resolving
them — conventionally caller-supplied path > `BROSOUNDML_DATA_DIR` env var
> `../brosoundml-data`. See `brosoundml-data/README.md` for the artifact
inventory and per-file licenses.

## Build

```bash
# CPU-only
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release

# CPU + CUDA (forwards the choice to brotensor's CUDA backend)
cmake -B build -DBROTENSOR_WITH_CUDA=ON
cmake --build build --config Release
```

On Windows use the Visual Studio multi-config generator (`--config` picks the
config); on Linux/macOS use a separate build dir per config. brosoundml builds
no GPU language of its own — `BROTENSOR_WITH_CUDA` / `_WITH_METAL` only forward
the backend choice so a standalone GPU build resolves brotensor's backend.

## Kokoro

Kokoro-82M is an 82M-parameter TTS model derived from **StyleTTS 2**. Unlike
StyleTTS 2 it does not sample a style with a diffusion model — the "voice" is a
precomputed embedding (a *voice pack*), so synthesis is a single deterministic
forward pass.

### Pipeline

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

**G2P.** Kokoro upstream uses the [misaki](https://github.com/hexgrad/misaki)
(Apache 2.0) grapheme-to-phoneme frontend. brosoundml ships an in-tree English
G2P at `brosoundml::g2p::` that removes that runtime dependency for embedded /
no-Python deployments. The byte-level Transformer POS tagger (`PosTagger`,
weights in [`brosoundml-data`](https://huggingface.co/datasets/wlejon/brosoundml-data) under
`pos_tagger/`), the lexicon loader (`Lexicon`), morphology fallback
(`Morphology`), special-case overrides (`SpecialCases`), text normalizer
(`Normalizer`), and the Kokoro phoneme-id adapter (`PhonemeAdapter`) are tied
together by `Phonemizer` — so callers can phonemize English text in-tree with no
misaki/Python dependency. `Kokoro::synthesize()` still accepts phoneme ids
directly for callers that supply their own.

Kokoro is a single non-autoregressive forward pass, so there is no growing token
prefix to stream the way Qwen3-TTS does. `synthesize_stream()` instead chunks the
*input*: the caller passes phoneme chunks split at sentence/clause boundaries,
each is synthesized independently and its 24 kHz audio is handed to an `on_chunk`
sink the moment it finishes (first-chunk latency for a long script), with the
full concatenation also returned (each chunk also reports its own per-phoneme
frame durations so a caller can align words to the streamed audio). The
`synth --stream` CLI drives this, one `--ids-file` line per chunk.

### Authoring, introspection, and a trainable decoder

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

### brotensor op coverage

Most of Kokoro maps straight onto the existing op surface:

| Kokoro component | brotensor ops |
|---|---|
| Phoneme embedding, plBERT | `embedding_lookup`, `self_attention`, `layernorm`, `gelu` |
| Text encoder CNN | `conv1d`, `pad1d`, `layernorm`, `leaky_relu` |
| iSTFTNet decoder | `conv1d`, `conv_transpose1d`, `leaky_relu`, `snake`, `group_norm` (instance norm via `num_groups == C`) + `modulate` (the AdaIN affine) |
| iSTFT head | `istft` (and `stft` / `complex_*` for the magnitude/phase pair) |
| Resampling | `resample1d` |

The text encoder and both predictors use bidirectional LSTMs and `brotensor`
has no recurrent primitive, so brosoundml composes the LSTM cell from `matmul`
+ `sigmoid` + `tanh` per timestep. A fused `brotensor` LSTM op is a later
performance optimisation.

### Caveat

The harmonic-source branch (SineGen / SourceModuleHnNSF in upstream Kokoro)
uses a deterministic approximation that drops torch's random initial phases
and additive noise. Output is intelligible but not bit-equal to upstream.

## Qwen3-TTS

Qwen3-TTS is Alibaba's open-weight TTS series (Jan 2026). brosoundml targets the
12 Hz multi-codebook track — an end-to-end discrete-token model, no diffusion,
no external vocoder. The whole pipeline runs device-neutrally on **CPU and
CUDA**: `load(device)` places the weights on the chosen backend, and because
compute is FP32 on both, CUDA reproduces the CPU/upstream discrete-code stream
bit-for-bit (the codec tail then matches to ~1e-5).

### Pipeline

```
   1. Text         text -> Qwen BPE token ids (brolm Qwen tokenizer; the
                   vocab.json + merges.txt ship in the model dir).
   2. Talker       a 28-layer Qwen3 decoder backbone over a dual stream: text-
                   token embeddings (projected to the Talker hidden) interleaved
                   with codec-token embeddings, GQA + QK-norm + interleaved
                   M-RoPE. Per frame it emits a hidden state and, via codec_head,
                   acoustic codebook 0.
   3. Code         a 5-layer depth transformer that, conditioned on the Talker
      Predictor    hidden, autoregressively emits acoustic codebooks 1..15.
   4. Codec        the bundled 12 Hz codec (speech_tokenizer/): 16 RVQ codes per
      decoder      frame ─▶ dequantize ─▶ windowed pre-transformer ─▶ ConvNeXt
                   upsample ─▶ a SEANet causal-conv decoder with Snake activations
                   ─▶ 24 kHz. Total upsample 1920; 12.5 Hz · 1920 = 24 kHz.
```

### Variants and voice control

`QwenTts::synthesize()` picks the voice by checkpoint variant:

- **CustomVoice** — pass a preset `speaker` name (see `speakers()`). The 1.7B
  checkpoint also honours an `instruct` style prompt.
- **VoiceDesign** — pass a natural-language `instruct` describing the voice
  (e.g. *"a warm, low-pitched elderly storyteller"*); there are no presets.
- **Base** — zero-shot voice cloning. `synthesize_clone()` encodes a reference
  clip into an ECAPA-TDNN speaker x-vector and splices it into the Talker prefill
  where a CustomVoice preset token would sit ("x-vector-only" enrollment, no
  reference transcript). `embed_speaker()` exposes that enrollment step on its
  own — the honest audio→identity front-end for training an adapter into another
  model's style space — and `synthesize_with_xvector()` renders straight from a
  supplied x-vector, so a caller can enroll real voices and then interpolate /
  morph / steer in that 1024-d identity space before speaking.

**Standalone speaker encoder.** The Base ECAPA-TDNN extractor is also liftable
out of the ~2.5 GB checkpoint: `brosoundml_build_speaker_encoder` packs it into a
~18 MB two-file artifact, and the `SpeakerEncoder` class (`speaker_encoder.h`)
loads just that to enroll a clip — bit-identical to `QwenTts::embed_speaker` —
without touching the Talker, codec, or tokenizer it never needs.

**Streaming, steering, and tracing.** `synthesize_stream()` decodes the growing
code stream as the AR loop runs and hands each new 24 kHz chunk to an `on_chunk`
sink (the codec is causal, so delivered samples never change). `QwenTtsSampling`
exposes optional seeded sampling (temperature/top-p, reproducible per `seed`) and
codebook-0 logit steering (repetition penalty, a logit bias, and an adaptive
temperature that nudges only the frames where the model hedged); the default
(temperature 0) stays the greedy, bit-exact upstream policy. Passing a
`QwenTtsTrace*` to `synthesize()` captures the per-frame codes and codebook-0
confidence for visualization at near-zero cost.

The bundled codec is reachable directly: `encode_audio()` (waveform → RVQ codes)
and `decode_codes()` (RVQ codes → 24 kHz waveform) are inverses that round-trip,
useful on their own once a caller holds a code stream. The autoregressive loop
polls a `CancelCheck` once per generated frame so a long synthesis can be
aborted.

### brotensor op coverage

`rms_norm`, `rope`, `silu`/SwiGLU, GQA self-attention (CUDA via
`flash_attention_varlen_forward`'s FP32 kernel), `conv1d` / `conv_transpose1d`,
`snake`, `embedding_lookup`, `softmax`, `sample_logits` / `argmax` — all already
on the op surface. Qwen3-TTS adds no op brotensor lacks. See
`docs/qwen-tts-weights.md` for the full tensor map.

## Whisper

OpenAI's encoder-decoder speech-to-text model. brosoundml targets the HF
transformers checkpoints (`whisper-tiny` / `-base` / `-small` / `-medium` /
`-large-v3`) — `config.json` + `model.safetensors` in a model directory.
Tokenization is delegated to `brolm::whisper::Tokenizer`; brosoundml itself
takes already-tokenized prompts and emits token ids.

### Pipeline

```
   1. Log-mel front-end  16 kHz mono PCM ─▶ log-mel spectrogram
                         (num_mel_bins × 3000 frames, 30 s padded/truncated).
                         stft + mel-filterbank matmul + log.
   2. Encoder            two strided conv1d stems + sinusoidal positional
                         embeddings + a pre-LN Transformer stack.
   3. Decoder            cross-attention Transformer with a KV cache,
                         autoregressive greedy decode.
   4. Tokenizer          brolm::whisper::Tokenizer (external) maps id ─▶ text.
```

`transcribe()` has a `TranscribeOptions` overload for realtime use:
`on_token` fires per decoded id so a caller can emit partial text mid-utterance
instead of waiting for the whole clip, and `timestamp_begin_id` opts into
sequential long-form decode — audio past the fixed 30 s log-mel window is split
into 30 s segments and advanced by each segment's last emitted timestamp
(falling back to full-30 s hops) instead of being truncated. Only that single
int crosses the tokenizer boundary; the caller still owns `build_prompt`/`decode`
(`brolm::whisper::Tokenizer::first_timestamp_id()` supplies it). The CLI's
`--stream` flag and its automatic long-form for >30 s clips drive both.

### brotensor op coverage

`stft` + `complex_abs` + `matmul` (mel front-end), `conv1d`, `gelu`,
`layer_norm`, MHA (composed via brosoundml's FP32 MHA / CrossAttention
modules — see `modules.h`), `embedding_lookup`, `sample_logits` / `argmax`.

## RAVE

[RAVE](https://github.com/acids-ircam/RAVE) (Realtime Audio Variational
autoEncoder, ACIDS/IRCAM) is a small (<20M-param), faster-than-realtime neural
audio autoencoder. A multiband (PQMF) variational convolutional encoder
compresses a waveform to a low-rate latent `z` of shape `(n_latent × frames)`; a
residual upsampling decoder synthesises the waveform back. The latent axes are
PCA-sorted by variance — dim 0 tracks loudness, dim 1 pitch/centroid, the rest
timbre — so editing the per-dimension time series (the lab's use case) morphs the
audio in musically meaningful ways. brosoundml runs **inference** of an exported
RAVE v2 model; the whole forward pass composes brotensor's existing op surface,
so a CUDA / Metal build runs on the GPU with no model-specific kernels.

### Pipeline

```
   1. Convert   the exported streaming `.ts` (the cached_conv build used by the
                nn~/VST) -> safetensors + config.json, offline, by
                scripts/convert-rave.py. The loader reads that layout.
   2. Encode    waveform -> PQMF analysis -> variational conv encoder -> PCA
                crop -> latent (cropped_latent_size × frames). Deterministic:
                the posterior mean, no reparameterisation sampling.
   3. Decode    latent -> residual upsampling decoder -> deterministic waveform +
                loudness branches -> PQMF synthesis -> waveform (sampling_rate Hz).
                A single fresh-cache (offline causal) pass.
```

### Editing, noise, and stereo

`encode()` / `decode()` round-trip a clip reproducibly — the right default for
editing the latent curves. Two options extend the decoder:

- **Stochastic noise synth** — `RaveDecodeOptions{add_noise = true}` adds RAVE's
  third synthesis branch, an FFT filtered-noise synthesizer for breathy /
  unvoiced / textural energy. Its white noise is redrawn each call, so pin it
  with a `seed` (or inject a `noise` buffer) when you need reproducibility.
- **Stereo decode** — RAVE has no stereo decoder; the VST runs the mono decoder
  once per channel and the channels decorrelate only because the discarded latent
  dims are padded with *independent* N(0,1) noise per channel. `decode_multi()`
  reproduces that exactly: `latent_pad_std` is the per-channel pad std (RAVE-native
  1.0; it doubles as the "stereo width" knob, 0 = identical to mono), each channel
  drawn reproducibly from `seed`. Output is interleaved, dropping straight into an
  interleaved sink.

### brotensor op coverage

PQMF analysis/synthesis and every encoder/decoder block are `conv1d` /
`conv_transpose1d` + `batch_norm` + `leaky_relu` + `snake`; the waveform and
loudness heads use `tanh` / `sigmoid` / `exp`; the optional noise branch is an
`rfft` / `irfft` filtered-noise synth. RAVE adds no op brotensor lacks.

## Wake-word

A small streaming convolutional keyword spotter that fires once when a target
keyword (`"computer"` is the first target) is spoken. The model is BC-ResNet-
style: log-mel front-end ▶ depthwise-separable Conv1d residual blocks ▶ global
average pool ▶ linear head ▶ a single logit. It is designed for an always-on mic
loop — feature extract and forward pass are each a low-millisecond per-frame
cost — and `WakeWord::feed()` buffers arbitrary chunk sizes, advancing the
streaming front-end and model one 10 ms frame at a time, returning true exactly
once per detected event (2-of-3 smoothing, then a `refractory_ms` debounce).
Detector policy (threshold / smoothing / refractory) is caller-tunable at
runtime; the front-end and model hyperparameters come from the weights file.

The full training toolchain lives in `tools/`:

| Tool | Role |
|---|---|
| `brosoundml_wake_synth` | Kokoro-driven dataset builder (positive/negative clips) |
| `brosoundml_wake_inspect` | dataset validator |
| `brosoundml_wake_train` | backward + Adam + BCE-with-logits |
| `brosoundml_wake_test` | held-out evaluation |
| `brosoundml_wake_probe` / `_melcmp` | front-end / mel diagnostics |

### brotensor op coverage

`stft` + `complex_abs` + mel matmul + log (the streaming front-end), `conv1d`
(depthwise + pointwise) + `relu`, and global average pool. Batch norm is folded
into the preceding conv at inference, so it costs no runtime op; the training
toolchain composes BN inline and uses brotensor's `*_backward` ops, a fused
BCE-with-logits, and Adam — all on the existing op surface.

## License

[MIT](LICENSE)
