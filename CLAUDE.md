# CLAUDE.md — brosoundml

Audio-ML model inference library. brosoundml is the **expression layer** for
neural audio models — it composes `brotensor`'s audio op family into runnable
TTS / STT / neural-codec models, the same way `brodiffusion` composes the
diffusion ops and `brolm` composes the text-model ops. One flat namespace,
`brosoundml::`.

**First operational target: Kokoro-82M text-to-speech.** The repo is currently
stood up but not yet operational — see the build-out plan below.

## Layout

```
include/brosoundml/
  version.h        library version constants + version_string()
  audio.h          AudioBuffer (mono FP32 PCM) + WAV read/write
  modules.h        inference-only nn-modules over brotensor ops:
                   Linear, LayerNorm, Conv1d, LSTM, BiLSTM, ada_in_1d
  kokoro.h         Kokoro-82M: KokoroConfig, Voice, the Kokoro pipeline class
  detail/json.h    vendored JSON parser (kept in sync with brolm's)
src/
  version.cpp
  audio.cpp        AudioBuffer math + 16-bit PCM WAV I/O  (real, tested)
  modules.cpp      module-layer implementations (CPU FP32 path proven; GPU
                   paths follow once brotensor ops land for them)
  kokoro.cpp       Kokoro pipeline — pImpl; stage-1 loader real, forward pass
                   in build-out
tests/
  test_smoke.cpp    links + versions + reaches brotensor
  test_audio.cpp    AudioBuffer math + WAV round trip
  test_modules.cpp  module layer vs. a from-scratch LSTM-cell reference
  test_kokoro.cpp   Kokoro stage-1 loader contract + staged forward-pass stub
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
- **brolm** — tokenizers for the speech models (e.g. `brolm::whisper::Tokenizer`).

Data sibling, separate from the code dependency chain — loaders take paths,
the application resolves them:

- **brosoundml-data** (`../brosoundml-data`) — trained weights and packed data
  artifacts (POS tagger `model.bin`, the planned lexicon binary, voice packs
  not shipped with upstream model checkpoints). Local-only for now; eventual
  home is a Hugging Face dataset repo. Path resolution convention used by
  callers/tools: caller-supplied path > `BROSOUNDML_DATA_DIR` env var >
  `../brosoundml-data`. The library itself never touches the filesystem
  beyond the paths handed to it.

## Conventions

- **One flat namespace, `brosoundml::`.** No sub-namespaces.
- **Compute is brotensor; brosoundml is composition.** A new model is a graph
  of `brotensor` op calls plus weight loading and pre/post-processing — never a
  new kernel. If an op is genuinely missing, add it to `brotensor` (and mirror
  it across CPU/CUDA/Metal there), not here.
- **`AudioBuffer` is the waveform currency** — mono FP32 PCM nominally in
  [-1, 1], carrying its `sample_rate`. Synthesis returns one; file I/O consumes
  one.
- **Heavy model state lives behind a pImpl** (`Kokoro::Impl`) so public headers
  stay free of brotensor module internals.
- **Errors throw `std::runtime_error`** with a `"brosoundml: <where>: <reason>"`
  message — matching the brotensor convention.
- **Staged stubs throw, loudly.** A not-yet-built entry point throws a
  `runtime_error` naming its build-out stage rather than returning a wrong
  result. `test_kokoro` locks this contract; the checks flip to real behaviour
  as each stage lands.

## Kokoro architecture

Kokoro-82M is derived from **StyleTTS 2**, minus the diffusion style sampler —
the voice is a precomputed embedding, so synthesis is one deterministic forward
pass. Stages: G2P (external — misaki) → plBERT → text encoder (CNN + BiLSTM) →
predictor (duration → length regulation → F0/energy) → iSTFTNet decoder + iSTFT
head → 24 kHz waveform. Full detail and the brotensor op mapping are in
`README.md`.

**Known op gap: no recurrent (LSTM) primitive in brotensor.** The text encoder
and predictors are bidirectional-LSTM-based. brosoundml composes the LSTM cell
from `matmul` + `sigmoid` + `tanh` per timestep (see `brosoundml::LSTM` /
`BiLSTM` in `modules.h`) — correct and sufficient to get Kokoro running. A
fused `brotensor` `lstm` op is a deferred optimisation; do not add it
speculatively — specify it once the real cell shapes are pinned down by a
working model (north-star vs. task).

**Known op gap: no per-channel affine on NCL in brotensor.** The AdaIN affine
step (used by every iSTFTNet AdaIN1d resblock) is a per-channel scale + shift
broadcast across `(n, l)`. brosoundml's `ada_in_1d` does `group_norm_forward`
(with `num_groups == C`) followed by a hand-rolled CPU loop for the affine —
fine to get Kokoro running on CPU, but the affine should move to a brotensor
op (`per_channel_affine_1d`, NCL, FP32, all three backends) before stage 5
matters on GPU.

## Build-out plan

Each stage is independently testable; land them in order:

1. **Weight loading** — `config.json` → `KokoroConfig`; `model.safetensors` +
   voice packs → tensors via `brotensor::safetensors`.
2. **Module layer** — `Linear` / `Conv1d` / `LSTM` / `AdaINResBlock` /
   `LayerNorm` over brotensor ops; unit-tested with synthetic weights.
3. **plBERT + text encoder** — phonemes → per-phoneme features.
4. **Predictor** — duration → length regulation → F0 / energy.
5. **Decoder + iSTFT head** — frame features → waveform; `synthesize` real.
6. **bro integration** — `bro.sound` JS bindings + wire into
   `bro/third_party/CMakeLists.txt`, after brotensor in the load order.

## Tests

`ctest --test-dir build -C Release`. Tests are built only when brosoundml is the
top-level project (`BROSOUNDML_TESTS`, ON by default standalone) — when consumed
as a subdirectory by bro they are skipped.
