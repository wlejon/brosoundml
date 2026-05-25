# brosoundml

Audio-ML model inference. brosoundml is the **expression layer** for neural
audio models — it composes the FP32 audio op family in `brotensor` (FFT/STFT,
1D convolution, vocoder/codec activations, codec quantization, resampling,
autoregressive sampling) into runnable text-to-speech, speech-to-text, and
neural-codec models.

It is to audio what [`brodiffusion`](../brodiffusion) is to images and
[`brolm`](../brolm) is to text: a sibling library that turns a tensor op
surface into a model.

Models implemented:

- **Kokoro-82M** — text-to-speech (StyleTTS 2 derivative, 24 kHz output).
- **Whisper** — speech-to-text (HF transformers checkpoints, tiny → large-v3).

CLI tools `brosoundml_synth` and `brosoundml_transcribe` drive each end-to-end.

## Dependencies

brosoundml is a standalone sibling repo. It links three siblings and ships no
GPU kernels of its own — GPU work happens inside `brotensor`.

| Library | Role |
|---|---|
| [`bromath`](../bromath) | header-only math (Vec/Quat/Mat, easing) |
| [`brotensor`](../brotensor) | the unified `Tensor` + device-neutral op surface (including the audio op family) |
| [`brolm`](../brolm) | tokenizers used by the speech models (e.g. `brolm::whisper::Tokenizer`) |

CMake auto-detects standalone repos at `../<name>`, falling back to
`third_party/` submodules — the pattern in
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

## Kokoro

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

The text encoder and both predictors use bidirectional LSTMs and `brotensor`
has no recurrent primitive, so brosoundml composes the LSTM cell from `matmul`
+ `sigmoid` + `tanh` per timestep. A fused `brotensor` LSTM op is a later
performance optimisation.

### Caveat

The harmonic-source branch (SineGen / SourceModuleHnNSF in upstream Kokoro)
uses a deterministic approximation that drops torch's random initial phases
and additive noise. Output is intelligible but not bit-equal to upstream.

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

### brotensor op coverage

`stft` + `complex_abs` + `matmul` (mel front-end), `conv1d`, `gelu`,
`layer_norm`, MHA (composed via brosoundml's FP32 MHA / CrossAttention
modules — see `modules.h`), `embedding_lookup`, `sample_logits` / `argmax`.

## License

[MIT](LICENSE)
