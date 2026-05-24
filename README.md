# brosoundml

Audio-ML model inference. brosoundml is the **expression layer** for neural
audio models — it composes the FP32 audio op family in `brotensor` (FFT/STFT,
1D convolution, vocoder/codec activations, codec quantization, resampling,
autoregressive sampling) into runnable text-to-speech, speech-to-text, and
neural-codec models.

It is to audio what [`brodiffusion`](../brodiffusion) is to images and
[`brolm`](../brolm) is to text: a sibling library that turns a tensor op
surface into a model.

The first operational target is **Kokoro-82M** text-to-speech.

## Status

Early — the repo is **stood up**, not yet operational.

| Area | State |
|---|---|
| Build system, sibling wiring (bromath + brotensor) | ✅ done |
| `AudioBuffer` + 16-bit PCM WAV read/write | ✅ done, tested |
| Kokoro public API (`Kokoro`, `KokoroConfig`, `Voice`) | ✅ committed |
| Stage 1 — `config.json` parser, `model.safetensors` open, voice-pack loader | ✅ done, tested |
| Stage 2 — module layer (`Linear`, `LayerNorm`, `Conv1d`, `LSTM`, `BiLSTM`, `ada_in_1d`) | ✅ done, tested |
| Stage 3 — plBERT + bert_encoder + TextEncoder | ✅ validated to upstream (max-abs 1e-5) |
| Stage 4 — ProsodyPredictor (duration + F0 + N) | ✅ validated to upstream (max-abs 5e-4) |
| Stage 5a — Decoder backbone (encode + decode loop) | ✅ validated to upstream (max-abs 1e-4) |
| Stage 5b — Generator (ups + resblocks + iSTFT) | ✅ validated to upstream (max-abs 7e-5) |
| `Kokoro::synthesize` end-to-end | ✅ runs; SineGen / harmonic source stubbed |
| SineGen / SourceModuleHnNSF | ❌ not implemented — har_source replaced by deterministic placeholder; audio lacks natural breath excitation |

While the forward pass is in build-out, `Kokoro::load` / `synthesize` throw a
`std::runtime_error` naming the stage; the API shape itself is committed and
covered by `test_kokoro`.

## Dependencies

brosoundml is a standalone sibling repo. It links two siblings and ships no
GPU kernels of its own — GPU work happens inside `brotensor`.

| Library | Role |
|---|---|
| [`bromath`](../bromath) | header-only math (Vec/Quat/Mat, easing) |
| [`brotensor`](../brotensor) | the unified `Tensor` + the device-neutral op surface, including the audio op family |

CMake auto-detects standalone repos at `../bromath` and `../brotensor`, falling
back to `third_party/` submodules — the pattern in
[`bro/docs/multi-repo-workflow.md`](../bro/docs/multi-repo-workflow.md).

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

## The Kokoro target

Kokoro-82M is an 82M-parameter TTS model derived from **StyleTTS 2**. Unlike
StyleTTS 2 it does not sample a style with a diffusion model — the "voice" is a
precomputed embedding (a *voice pack*), so synthesis is a single deterministic
forward pass.

### Pipeline

```
text ──▶ [G2P: misaki]──▶ phonemes ──▶ [token ids]
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

**G2P is out of scope.** Kokoro uses the [misaki](https://github.com/hexgrad/misaki)
grapheme-to-phoneme frontend; brosoundml takes phoneme token ids as input and
does not bundle a G2P engine.

### brotensor op coverage

Most of Kokoro maps straight onto the existing op surface:

| Kokoro component | brotensor ops |
|---|---|
| Phoneme embedding, plBERT | `embedding_lookup`, `self_attention`, `layernorm`, `gelu` |
| Text encoder CNN | `conv1d`, `pad1d`, `layernorm`, `leaky_relu` |
| iSTFTNet decoder | `conv1d`, `conv_transpose1d`, `leaky_relu`, `snake`, `group_norm` (instance norm via `num_groups == C`) + `modulate` (the AdaIN affine) |
| iSTFT head | `istft` (and `stft` / `complex_*` for the magnitude/phase pair) |
| Resampling | `resample1d` |

**The one gap is a recurrent (LSTM) op.** The text encoder and both predictors
use bidirectional LSTMs, and `brotensor` has no recurrent primitive. brosoundml
will compose the LSTM cell from `matmul` + `sigmoid` + `tanh` per timestep —
correct first, and enough to get Kokoro operational. A fused `brotensor` LSTM
op is a later performance optimisation, to be specified once the real cell
shapes are known.

## Build-out plan

Ordered so each stage is independently testable:

1. **Weight loading** ✅ — parse `config.json` into `KokoroConfig`; open
   `model.safetensors` (via `brotensor::safetensors`) and the voice packs into
   tensors. `load` / `load_voice` / `Voice::pick_for` are real. Voice packs are
   loaded as raw little-endian FP32 of shape `(rows, 2*style_dim)`; convert
   upstream Kokoro `.pt` voices to this format once on the host.
2. **Module layer** ✅ — a small nn-module set over brotensor ops: `Linear`
   (single-vec + batched), `LayerNorm`, `Conv1d`, `LSTM` + `BiLSTM` (composed
   from `matmul` + `sigmoid` + `tanh` per timestep), and the `ada_in_1d` affine
   primitive. Inference-only. Unit-tested in `test_modules` with hand-rolled
   synthetic weights against a from-scratch LSTM-cell reference.
3. **plBERT + text encoder** ✅ — `brosoundml::PLBert` (HuggingFace
   `AlbertModel` topology with 12 shared layers + hand-rolled MHA with biases),
   `brosoundml::BertEncoder` (768 → 512 Linear), and `brosoundml::TextEncoder`
   (embedding + 3 × (Conv1d + per-channel NCL LayerNorm + LeakyReLU) +
   bidirectional LSTM). Validated row-by-row against a reference activation
   dump from the upstream `kokoro` Python package (max-abs 1e-5).
4. **Predictor** ✅ — `brosoundml::Predictor` covers `DurationEncoder` (3
   alternating BiLSTM + AdaLayerNorm blocks), the duration LSTM, the
   `LinearNorm` projection, the length regulator, the shared LSTM, and the
   AdaINResBlk1d-based F0 / N stacks. Reference-validated (max-abs 5e-4).
5. **Decoder + iSTFT head** ✅ (with one caveat) — `brosoundml::DecoderBackbone`
   covers the encode + decode loop (`F0_conv`, `N_conv`, `asr_res`, encode,
   4 decode blocks including the upsample). `brosoundml::Generator` covers
   the deterministic iSTFTNet backbone (2 ConvTranspose ups, 2 noise convs,
   2 noise_res AdaINResBlock1 blocks, 6 resblocks, conv_post, iSTFT) and
   matches the upstream audio bit-for-bit given the same harmonic source
   (max-abs 7e-5). **SineGen / SourceModuleHnNSF is not implemented** — the
   end-to-end `Kokoro::synthesize` feeds a deterministic noise placeholder
   into the noise branch, so the synthesised audio lacks the natural breath
   excitation. Implementing a torch-compatible RNG (or a deterministic sine
   generator) is the remaining work to reach upstream audio quality.
6. **bro integration** — deferred until after SineGen.

## License

[MIT](LICENSE)
