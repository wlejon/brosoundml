# RAVE

[RAVE](https://github.com/acids-ircam/RAVE) (Realtime Audio Variational
autoEncoder, ACIDS/IRCAM) is a small (<20M-param), faster-than-realtime neural
audio autoencoder. A multiband (PQMF) variational convolutional encoder
compresses a waveform to a low-rate latent `z` of shape `(n_latent × frames)`; a
residual upsampling decoder synthesises the waveform back. The latent axes are
PCA-sorted by variance — dim 0 tracks loudness, dim 1 pitch/centroid, the rest
timbre — so editing the per-dimension time series (the lab's use case) morphs the
audio in musically meaningful ways. brosoundml runs **inference** of an exported
RAVE v2 model; the whole forward pass composes brotensor's existing op surface,
so a CUDA / Metal build runs on the GPU with no model-specific kernels. RAVE is
a library-only model (no CLI driver), exercised by `test_rave`.

Public surface: `include/brosoundml/rave.h`.

## Pipeline

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

## Editing, noise, and stereo

`encode()` / `decode()` round-trip a clip reproducibly — the right default for
editing the latent curves. Two options extend the decoder:

- **Stochastic noise synth** — `RaveDecodeOptions{add_noise = true}` adds RAVE's
  third synthesis branch, an FFT filtered-noise synthesizer for breathy /
  unvoiced / textural energy. Its white noise is redrawn each call, so pin it
  with a `seed` (or inject a `noise` buffer) when you need reproducibility.
- **Stereo decode** — RAVE has no stereo decoder; the VST runs the mono decoder
  once per channel and the channels decorrelate only because the discarded latent
  dims are padded with *independent* N(0,1) noise per channel. `decode_multi()`
  reproduces that exactly: `latent_pad_std` is the per-channel pad std
  (RAVE-native 1.0; it doubles as the "stereo width" knob, 0 = identical to
  mono), each channel drawn reproducibly from `seed`. Output is interleaved,
  dropping straight into an interleaved sink.

## brotensor op coverage

PQMF analysis/synthesis and every encoder/decoder block are `conv1d` /
`conv_transpose1d` + `batch_norm` + `leaky_relu` + `snake`; the waveform and
loudness heads use `tanh` / `sigmoid` / `exp`; the optional noise branch is an
`rfft` / `irfft` filtered-noise synth. RAVE adds no op brotensor lacks.

## Conversion

`scripts/convert-rave.py` turns the exported streaming `.ts` model into the
safetensors + config.json layout the loader reads; `scripts/rave_reference.py`
is the reference harness the test validates against.
