# Wake-word

A small streaming convolutional keyword spotter that fires once when a single
target keyword (`"computer"` is the first target) is spoken. It is designed for
an always-on mic loop — feature extract and forward pass are each a
low-millisecond per-frame cost — and `WakeWord::feed()` buffers arbitrary chunk
sizes, advancing the streaming front-end and model one 10 ms frame at a time,
returning true exactly once per detected event. Detector policy (threshold /
smoothing / refractory) is caller-tunable at runtime; the front-end and model
hyperparameters come from the weights file.

For an *open-vocabulary* spotter — enroll any word by typing it, no per-keyword
training — see [phoneme-spotter.md](phoneme-spotter.md).

Public surface: `include/brosoundml/wake.h` (the `WakeWord` detector — front-end
+ model + policy), `include/brosoundml/bc_resnet2d.h` (the shipped 2D BC-ResNet
model: forward, streaming, `train_step`), `include/brosoundml/mel.h` (the shared
log-mel/PCEN front-end), `include/brosoundml/wake_data.h` (the training-dataset
binary format). `include/brosoundml/bc_resnet.h` is the earlier 1D variant (mel
bins as conv channels) — retained for its tests, but **not** the runtime model.

## Model — 2D BC-ResNet on PCEN features

The shipped detector is a genuine 2D Broadcasted-Residual network (Kim et al.,
Interspeech 2021), the microphone-robust replacement for the 1D model. The 1D
model used the 40 mel bins as conv *channels* and convolved over time only, so
it baked the TTS training spectral envelope into its first layer and collapsed
on real microphones. `WakeWord::load()` therefore **rejects** the legacy 1D
checkpoint (`'BWAK'`) and requires the 2D model (`'BWK2'`).

- **Front-end.** A **PCEN** mel front-end (per-channel energy normalization that
  cancels mic/channel spectral tilt), not plain log-mel — the runtime front-end
  is forced to PCEN to match how the 2D recipe is trained (`wake.cpp`).
- **Trunk.** The input is a single-channel (freq × time) image; convolutions
  slide over both axes with weights shared across frequency. Each Broadcasted-
  Residual block splits into `f2` (frequency-depthwise conv2d + SubSpectralNorm +
  ReLU) and `f1` (frequency-average-pool → causal temporal-depthwise conv + BN →
  broadcast back over frequency), summed with a residual and mixed by a 1×1
  pointwise conv. Time convolutions are strictly causal (left-pad only), so a
  streaming forward emits a logit the moment a frame arrives (≤30 ms
  end-of-word→fire budget).
- **Head.** Frequency-average-pool → temporal global-average-pool over the
  receptive-field window → `Linear(C_last → 1)` → a single logit.

`BcResnet2d` is device-neutral: `WakeWord::load()` takes a `brotensor::Device`
(default CPU) and the trunk runs on CPU or CUDA.

## Detection policy

`feed()` buffers arbitrary chunk sizes and advances the front-end and model one
10 ms frame at a time. A fire requires: the streaming GAP ring to be warmed up
(suppressed until it fills — `gap_window_frames()` frames after a reset), then a
2-of-3 smoother over the per-frame logit, then a `refractory_ms` debounce — and
returns true at most once per `feed()` call. Threshold, smoothing window, and
refractory are caller-tunable at runtime (`set_threshold` / `set_smoothing` /
`set_refractory_ms`, mirrored into relaxed atomics for the audio thread).

## Training toolchain

The full Kokoro-driven training toolchain lives in `tools/`:

| Tool | Role |
|---|---|
| `brosoundml_wake_synth` | Kokoro-driven dataset builder (positive/negative clips) |
| `brosoundml_wake_inspect` | dataset validator |
| `brosoundml_wake_train` | backward + Adam + BCE-with-logits |
| `brosoundml_wake_test` | held-out evaluation |
| `brosoundml_wake_probe` / `_melcmp` | front-end / mel diagnostics |

## brotensor op coverage

The PCEN mel front-end (`stft` + `complex_abs` + mel matmul + PCEN), `conv2d`
(depthwise `f2`/`f1` + pointwise mix) + `batch_norm` + `relu`, frequency-average
and temporal global-average pools, and a `linear` head. SubSpectralNorm is the
one module with a hand-rolled backward; everything else reuses brotensor's
conv2d / batch_norm / relu / linear forward+backward. Batch norm is fused into
the preceding conv at inference (`fuse_bn()`), so it costs no runtime op; the
training toolchain composes BN inline and uses a fused BCE-with-logits and Adam.
