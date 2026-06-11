# Wake-word

A small streaming convolutional keyword spotter that fires once when a single
target keyword (`"computer"` is the first target) is spoken. The model is
BC-ResNet-style: log-mel front-end ▶ depthwise-separable Conv1d residual blocks
▶ global average pool ▶ linear head ▶ a single logit. It is designed for an
always-on mic loop — feature extract and forward pass are each a low-millisecond
per-frame cost — and `WakeWord::feed()` buffers arbitrary chunk sizes, advancing
the streaming front-end and model one 10 ms frame at a time, returning true
exactly once per detected event (2-of-3 smoothing, then a `refractory_ms`
debounce). Detector policy (threshold / smoothing / refractory) is caller-tunable
at runtime; the front-end and model hyperparameters come from the weights file.

For an *open-vocabulary* spotter — enroll any word by typing it, no per-keyword
training — see [phoneme-spotter.md](phoneme-spotter.md).

Public surface: `include/brosoundml/wake.h` (the `WakeWord` detector — front-end
+ model + policy), `include/brosoundml/bc_resnet.h` (the BC-ResNet model:
forward, streaming, `train_step`), `include/brosoundml/bc_resnet2d.h` (the 2D
freq×time variant + training surface), `include/brosoundml/mel.h` (the shared
log-mel front-end), `include/brosoundml/wake_data.h` (the training-dataset
binary format).

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

`stft` + `complex_abs` + mel matmul + log (the streaming front-end), `conv1d`
(depthwise + pointwise) + `relu`, and global average pool. Batch norm is folded
into the preceding conv at inference, so it costs no runtime op; the training
toolchain composes BN inline and uses brotensor's `*_backward` ops, a fused
BCE-with-logits, and Adam — all on the existing op surface.
