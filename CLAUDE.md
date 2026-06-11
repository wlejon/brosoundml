# CLAUDE.md — brosoundml

Audio-ML model inference library. brosoundml is the **expression layer** for
neural audio models — it composes `brotensor`'s audio op family into runnable
TTS / STT / neural-codec / wake-word models, the same way `brodiffusion`
composes the diffusion ops and `brolm` composes the text-model ops. One flat
namespace, `brosoundml::`.

**Status: operational.** Five model families are complete (CPU FP32; CUDA where
noted) and back the `bro.tts` / `bro.stt` / `bro.wake` JS bindings in bro:

- **Kokoro-82M** — text-to-speech (StyleTTS 2 derivative, 24 kHz).
- **Qwen3-TTS** — text-to-speech (12 Hz multi-codebook, end-to-end discrete
  token, 24 kHz). Device-neutral CPU + CUDA; CustomVoice presets, VoiceDesign
  instruct prompts, Base-variant zero-shot voice cloning.
- **Whisper** — speech-to-text (HF checkpoints, tiny → large-v3).
- **Parakeet-TDT** — speech-to-text (NVIDIA FastConformer encoder + TDT
  transducer decoder, multilingual 0.6B-v3). Device-neutral CPU + CUDA;
  validated bit-faithful against the reference checkpoint.
- **Wake-word** — BC-ResNet streaming keyword spotter + its training toolchain.

Plus an in-tree English **G2P** (`brosoundml::g2p::`) so Kokoro phonemizes with
no misaki/Python dependency. Full per-model detail is in `README.md`.

## Layout

```
include/brosoundml/
  version.h          library version constants + version_string()
  audio.h            AudioBuffer (mono FP32 PCM) + WAV read/write; CancelCheck
  modules.h          inference-only nn-modules over brotensor ops:
                     Linear, LayerNorm, Conv1d, LSTM/BiLSTM, ada_in_1d, MHA
  kokoro.h           Kokoro-82M: KokoroConfig, Voice, the Kokoro pipeline
  kokoro_modules.h   Kokoro-specific module graph (plBERT, iSTFTNet, AdaIN)
  qwen_tts.h         Qwen3-TTS: QwenTts pipeline, Talker/CodePredictor/codec
                     configs, synthesize / synthesize_clone / encode/decode
  whisper.h          Whisper: WhisperConfig + the encoder/decoder pipeline
  whisper_modules.h  Whisper-specific modules (conv stems, cross-attn decoder)
  parakeet.h         Parakeet-TDT: ParakeetConfig + the FastConformer/TDT
                     pipeline (src/parakeet_modules.h holds the module graph)
  wake.h             WakeWord streaming detector (front-end + model + policy)
  bc_resnet.h        BC-ResNet wake model: forward, streaming, train_step
  bc_resnet2d.h      2D BC-ResNet variant (freq×time) + training surface
  mel.h              shared log-mel front-end helpers
  wake_data.h        wake training-dataset binary format
  g2p/               in-tree English G2P: pos_tagger, lexicon, morphology,
                     special_cases, normalizer, phoneme_adapter, phonemizer
  detail/json.h      vendored JSON parser (kept in sync with brolm's)
src/                 one .cpp per header above; qwen_tts split across
                     qwen_tts{,_talker,_code_predictor,_generate,_codec,
                     _codec_encoder,_speaker_encoder}.cpp
tools/               CLI drivers — see "Tools" below
tests/               one test_*.cpp per model + module layer (ctest)
```

## Build

```sh
# CPU-only
cmake -S . -B build && cmake --build build --config Release
ctest --test-dir build -C Release

# CPU + CUDA — forwarded to brotensor's CUDA backend (brosoundml ships no kernels)
cmake -S . -B build -DBROTENSOR_WITH_CUDA=ON && cmake --build build --config Release
```

On Windows use the Visual Studio multi-config generator (`--config` picks the
config); on Linux/macOS use a separate build dir per config. brosoundml builds
no GPU language of its own — `BROTENSOR_WITH_CUDA` / `_WITH_METAL` only forward
the backend choice so a standalone GPU build resolves brotensor's backend.

## Dependencies

Code-side siblings, resolved by the standard multi-repo pattern (standalone
repo at `../<name>`, else a `third_party/` submodule fallback — see
`bro/docs/multi-repo-workflow.md`):

- **bromath** — header-only math.
- **brotensor** — the unified `Tensor` + the device-neutral op surface. All of
  brosoundml's compute goes through `<brotensor/ops.h>`; brosoundml writes no
  kernels. The audio op family it leans on (FFT/STFT, conv1d, vocoder/codec
  activations, codec quantization, resampling, `sample_logits`) is FP32 on all
  three backends — CPU, CUDA, Metal.
- **brolm** — tokenizers for the speech models: `brolm::whisper::Tokenizer` for
  Whisper, the Qwen BPE tokenizer for Qwen3-TTS.

Data sibling, separate from the code dependency chain — loaders take paths,
the application resolves them:

- **brosoundml-data** (`../brosoundml-data`) — trained weights and packed data
  artifacts (POS tagger `model.bin`, the packed English lexicon, Kokoro voice
  packs not shipped with upstream checkpoints, wake-word checkpoints). Local-
  only for now; eventual home is a Hugging Face dataset repo. Path-resolution
  convention used by callers/tools: caller-supplied path > `BROSOUNDML_DATA_DIR`
  env var > `../brosoundml-data`. The library itself never touches the
  filesystem beyond the paths handed to it.

## Conventions

- **One flat namespace, `brosoundml::`.** No sub-namespaces (except the in-tree
  `brosoundml::g2p::` for the English phonemizer).
- **Compute is brotensor; brosoundml is composition.** A new model is a graph
  of `brotensor` op calls plus weight loading and pre/post-processing — never a
  new kernel. If an op is genuinely missing, add it to `brotensor` (and mirror
  it across CPU/CUDA/Metal there), not here.
- **`AudioBuffer` is the waveform currency** — mono FP32 PCM nominally in
  [-1, 1], carrying its `sample_rate`. Synthesis returns one; file I/O consumes
  one. Long-running synthesis loops poll a `CancelCheck` (see `audio.h`).
- **Heavy model state lives behind a pImpl** (`Kokoro::Impl`, `QwenTts::Impl`,
  `Whisper::Impl`, `WakeWord::Impl`) so public headers stay free of brotensor
  module internals.
- **Errors throw `std::runtime_error`** with a `"brosoundml: <where>: <reason>"`
  message — matching the brotensor convention.
- **Device-neutrality is the bar for a model "done on GPU."** A `load(device)`
  must place weights on the chosen backend and the whole forward pass must
  dispatch through brotensor device ops. Qwen3-TTS is the reference: its FP32
  CPU/CUDA paths produce a bit-identical discrete-code stream.

## Model notes

Per-model architecture and the brotensor op map live in `README.md`. Two
brotensor-level notes that recur in the code:

**LSTM is composed, not a brotensor op — but the composition is fast.**
Kokoro's text encoder and predictors are bidirectional-LSTM-based, and
brotensor has no recurrent inference primitive, so `brosoundml::LSTM` /
`BiLSTM` (in `modules.h`/`modules.cpp`) compose the cell from brotensor ops.
The input-side projection for the whole sequence is hoisted into one batched
GEMM; on CUDA the remaining per-step recurrent body is captured as a CUDA
graph once per cell (lazily, cached on the module via `LstmGraphPlan`) and
replayed per timestep — the same launch-overhead fix as the Qwen Talker
decode step. A step is then a staging copy in, one graph launch, and a row
copy out. (brotensor does have `lstm_forward_train` — a CPU training forward
with BPTT caches — but it is not the inference path here.)

**The AdaIN/AdaLN affine rides the norm op directly.** `group_norm_forward`'s
gamma/beta are per-channel and `layernorm_forward_inference_batched`'s are
per-feature, so the style-conditioned affine is passed straight into the norm
call: `ada_in_1d_styled` / `ada_layernorm` (kokoro_modules.cpp) hand the norm
`(1 + gamma)` / `beta`, and the public `ada_in_1d` (modules.cpp, plain
`Y = X*scale + shift` contract) hands it `scale` / `shift` unchanged. One
fused pass — no `nchw_to_sequence`/`modulate` transpose chain, no unit-gamma
upload. The earlier modulate-based composition (and its ±1 scale fix-ups) is
gone; don't reintroduce it.

## Adding a model

The shape is the same for every model already here:

1. **Config + weight loading** — a `<Model>Config` struct parsed from the
   upstream `config.json`; `model.safetensors` → tensors via
   `brotensor::safetensors`, placed on the load device.
2. **Module graph** — compose the forward pass from `brotensor` ops (and the
   shared `modules.h` layers). No new kernels — if an op is missing, it goes in
   brotensor, mirrored across CPU/CUDA/Metal.
3. **Public pipeline class** — pImpl, `load(dir, device)` + a `synthesize` /
   `transcribe` / `feed` entry point, throwing `std::runtime_error` on misuse.
4. **A `test_<model>.cpp`** locking the loader contract and an end-to-end
   forward pass, plus a CLI driver under `tools/`.
5. **bro integration** — wire the `bro.<x>` JS binding and confirm the device
   path: a CUDA build must reproduce the CPU result (bit-identical discrete
   stream for token models; ~1e-5 for the continuous codec/vocoder tail).

## Tools

CLI drivers, built when brosoundml is the top-level project
(`BROSOUNDML_TOOLS`, ON by default standalone):

- `brosoundml_synth` — Kokoro text → WAV.
- `brosoundml_kokoro_bench` — Kokoro throughput benchmark (in-tree G2P, no
  Python): reports per-iteration wall time and the realtime factor. Set
  `BROSOUNDML_KOKORO_PROFILE=1` for the library's per-stage breakdown.
- `brosoundml_transcribe` — Whisper WAV → text.
- `brosoundml_parakeet_transcribe` — Parakeet-TDT WAV → text (+ `--timestamps`).
- `brosoundml_qwen_tts_bench` / `_roundtrip` / `_clone` — Qwen3-TTS synthesis,
  codec encode↔decode round-trip, and zero-shot voice clone.
- `brosoundml_wake_synth` / `_inspect` / `_train` / `_test` / `_probe` /
  `_melcmp` — the wake-word training toolchain (Kokoro-driven dataset builder,
  validator, trainer, evaluator, front-end diagnostics).
- `build_pos_dataset.py` / `build_lexicon.py` — G2P data prep (offline).

## Tests

`ctest --test-dir build -C Release`. Tests are built only when brosoundml is the
top-level project (`BROSOUNDML_TESTS`, ON by default standalone) — when consumed
as a subdirectory by bro they are skipped.
