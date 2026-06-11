# brosoundml

brosoundml is a C++ library that runs neural audio models: text-to-speech,
speech-to-text, a neural audio autoencoder, and keyword spotting. You hand it a
converted model directory and either text or an `AudioBuffer` of PCM, and it
gives you back synthesized audio or token ids.

Each model is built by composing the audio operations in
[`brotensor`](https://github.com/wlejon/brotensor) FFT/STFT, 1D/2D convolution,
vocoder/codec activations, codec quantization, resampling, autoregressive
sampling. A model here is a graph of brotensor op calls plus weight loading and pre/post-processing.
Everything lives in one flat namespace, `brosoundml::`.

## Models

All models are complete and run FP32 on CPU; the device-neutral ones place
weights on the chosen backend and dispatch the whole forward pass through
`brotensor` device ops, so a CUDA build reproduces the CPU result (bit-identical
discrete-token stream for token models, ~1e-5 for the continuous codec/vocoder
tail).

| Model | Task | Device | Notes |
|---|---|---|---|
| [Kokoro-82M](docs/kokoro.md) | text → speech | CPU + CUDA | StyleTTS 2 derivative, 24 kHz; in-tree English G2P |
| [Qwen3-TTS](docs/qwen-tts.md) | text → speech | CPU + CUDA | 12 Hz multi-codebook discrete-token, 24 kHz; presets, VoiceDesign, zero-shot clone |
| [Whisper](docs/whisper.md) | speech → text | CPU + CUDA | encoder-decoder; HF checkpoints tiny → large-v3 |
| [Parakeet-TDT](docs/parakeet.md) | speech → text | CPU + CUDA | FastConformer + TDT transducer; multilingual 0.6B-v3 + timestamps |
| [Qwen3-ASR](docs/qwen-asr.md) | speech → text | CPU + CUDA | AuT encoder + Qwen3 decoder; 52-language + language ID, context biasing |
| [RAVE](docs/rave.md) | waveform ⇄ latent | CPU + CUDA + Metal | ACIDS/IRCAM v2 neural audio autoencoder; editable PCA latent |
| [Wake-word](docs/wake-word.md) | keyword spotting | CPU + CUDA | 2D BC-ResNet (PCEN) single-keyword streaming spotter + training toolchain |
| [Phoneme spotter](docs/phoneme-spotter.md) | open-vocab spotting | CPU + CUDA | PhonemeNet posteriors + streaming template matcher; "type a word, spot it" |

The in-tree English **[G2P](docs/g2p.md)** (`brosoundml::g2p::`) lets Kokoro
phonemize text with no misaki/Python dependency.

## Dependencies

brosoundml ships no GPU kernels of its own — all compute (and all GPU work)
happens inside `brotensor`. It depends on three libraries:

| Library | Role |
|---|---|
| [`brotensor`](https://github.com/wlejon/brotensor) | the unified `Tensor` + device-neutral op surface (including the audio op family) — where every model's compute runs |
| [`brolm`](https://github.com/wlejon/brolm) | tokenizers used by the speech models (`brolm::whisper::Tokenizer`, the Qwen BPE tokenizer, `brolm::t5::Tokenizer`) |
| [`bromath`](https://github.com/wlejon/bromath) | header-only math (Vec/Quat/Mat, easing) |

Each resolves either to a standalone repo at `../<name>` or to a `third_party/`
submodule fallback (the standard multi-repo pattern — see
`bro/docs/multi-repo-workflow.md`).

## Data and weights

brosoundml ships **code only** — no trained weights, no packed data, no
voice packs are checked into this repo. Anything that gets built (POS tagger
weights, the packed English lexicon, Kokoro voice packs, wake-word
checkpoints, …) lives in a separate data repo,
[`brosoundml-data`](https://huggingface.co/datasets/wlejon/brosoundml-data).
Loaders take file paths; the application (or the CLI tools in this repo) is
responsible for resolving them — conventionally caller-supplied path >
`BROSOUNDML_DATA_DIR` env var > `../brosoundml-data`. The library itself never
touches the filesystem beyond the paths handed to it. The `scripts/` directory
holds the upstream-checkpoint converters and downloaders
(`convert-kokoro.py`, `convert-rave.py`, `download-qwen-tts.sh`, …).

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
the backend choice so a standalone GPU build resolves brotensor's backend. The
CLI tools and tests build only when brosoundml is the top-level project
(`BROSOUNDML_TOOLS` / `BROSOUNDML_TESTS`, both ON by default standalone).

## Conventions

- **`AudioBuffer` is the waveform currency** — mono FP32 PCM nominally in
  [-1, 1], carrying its `sample_rate`. Synthesis returns one; file I/O consumes
  one. Long-running loops poll a `CancelCheck` (see `include/brosoundml/audio.h`).
- **Heavy model state lives behind a pImpl** so public headers stay free of
  brotensor module internals.
- **Errors throw `std::runtime_error`** with a `"brosoundml: <where>: <reason>"`
  message — matching the brotensor convention.

## Documentation

Per-architecture detail (pipeline, voice/decode control, brotensor op map, CLI
tools, caveats) lives in [`docs/`](docs):

- [Kokoro-82M](docs/kokoro.md) · [Qwen3-TTS](docs/qwen-tts.md) — text-to-speech
- [Whisper](docs/whisper.md) · [Parakeet-TDT](docs/parakeet.md) · [Qwen3-ASR](docs/qwen-asr.md) — speech-to-text
- [RAVE](docs/rave.md) — neural audio autoencoder
- [Wake-word](docs/wake-word.md) · [Phoneme spotter](docs/phoneme-spotter.md) — keyword spotting
- [G2P](docs/g2p.md) — in-tree English grapheme-to-phoneme

Reference dumps: [Qwen3-TTS weight map](docs/qwen-tts-weights.md). G2P component
specs: [pos_tagger](docs/pos_tagger.md), [lexicon](docs/lexicon.md),
[morphology](docs/morphology.md), [special_cases](docs/special_cases.md),
[phonemizer](docs/phonemizer.md).

## License

[MIT](LICENSE)
