# Sortformer — streaming speaker diarization

`brosoundml::Sortformer` runs NVIDIA's **Streaming Sortformer** diarizer
(`nvidia/diar_streaming_sortformer_4spk-v2.1`): given 16 kHz mono audio it emits,
per 80 ms frame, an independent sigmoid activity probability for up to **four**
speakers, with speaker labels assigned in **arrival-time order**. It is the
diarization member of brosoundml's NeMo family and shares its acoustic backbone
(the FastConformer encoder) with Parakeet-TDT.

## Architecture

```
16 kHz PCM
  └─ log-mel front-end          128 mel, n_fft 512, win 25 ms, hop 10 ms,
                                pre-emphasis 0.97, symmetric Hann, Slaney mel,
                                log(x + 2⁻²⁴). NO per-feature normalization, and
                                NeMo's zero-padded (pad_mode="constant") center STFT.
  └─ NEST FastConformer enc     8× conv subsampling (pre-encode) → 17 Conformer
                                blocks (d_model 512, 8 heads, kernel 9, xscaling,
                                projection/conv biases, Transformer-XL rel-pos
                                attn). Output: (T/8, 512) at an 80 ms frame step.
  └─ Sortformer head            encoder_proj 512→192, 18-layer post-LN Transformer
                                encoder (inner 768, 8 heads, ReLU FFN), then a
                                two-layer sigmoid head (192→192→4).
  └─ (T, 4) speaker activity probabilities in [0, 1]
```

The FastConformer encoder is the shared `FastConformerEncoder`
(`src/fastconformer_modules.h`) — the same code Parakeet uses, parameterized by
`scale_input` (xscaling), the optional projection/FFN/conv biases, the mel
`normalize_features` flag, and `mask_padding` (NeMo's length handling: the one
trailing center-STFT frame is zeroed and masked out of the valid frames).

## Offline vs streaming

- **`diarize(audio)`** — one pass over the whole clip (NeMo's non-streaming
  `forward_infer`). Bit-faithful to the reference: the valid diarization frames
  match NeMo to ~1e-6 (FP32 noise floor); the single trailing center-STFT frame
  is a masked boundary artifact.
- **`make_session()` + `feed(session, audio, is_last)`** — streaming inference
  with the **Arrival-Order Speaker Cache (AOSC)**. The cache holds pre-encode
  (512-d) embeddings of previously seen speakers; each chunk re-runs the
  Conformer layers + head over `[spkcache | fifo | chunk]` and folds the chunk
  into the cache, compressing it to `spkcache_len` frames by the arrival-time
  importance scoring (log-odds score, silence gating, recency + per-speaker
  top-K boosting, mean-silence padding). Matches NeMo `forward_streaming` to
  ~1e-6 mean / 5e-4 max. `feed()` buffers audio and finalizes on `is_last`.

## Checkpoint

The upstream model ships as a native NeMo `.nemo` archive. Convert it once:

```sh
scripts/download-sortformer.sh                       # fetch the .nemo
scripts/convert-sortformer.py --src weights/sortformer/4spk-v2.1
```

`convert-sortformer.py` untars the `.nemo`, remaps the NeMo state-dict onto
brosoundml's scheme (the encoder onto the shared HF-Parakeet names, the head onto
`transformer.*` / `sortformer.*`), and writes `config.json` + `model.safetensors`.

## CLI

```sh
brosoundml_sortformer_diarize <wav> weights/sortformer/4spk-v2.1 \
    [--device cuda] [--streaming] [--threshold 0.5] [--probs-out FILE]
```

Prints RTTM speaker segments (contiguous frames above `--threshold`, merged).
On CUDA the offline forward runs at RTF ≈ 0.04 (≈20× realtime).

## Parity

`scripts/sortformer_parity.py` (offline) and `scripts/sortformer_parity_streaming.py`
(streaming AOSC) compare the C++ output against the reference NeMo model and are
the source of the numbers above. The synthetic `tests/test_sortformer.cpp` pins
the loader + forward contract in CI (CPU + CUDA), with a CUDA-only real-weights
smoke when the converted checkpoint is present.
