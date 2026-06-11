# Parakeet-TDT

[NVIDIA Parakeet-TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) is a
FastConformer encoder feeding a Token-and-Duration Transducer (TDT) decoder —
the v3 0.6B model is multilingual (25 European languages). brosoundml targets
the HF `transformers` `ParakeetForTDT` checkpoint (`config.json` +
`model.safetensors`); the unified SentencePiece tokenizer (`tokenizer.json`,
loaded by `brolm::t5::Tokenizer`) is the caller's, as with Whisper — brosoundml
emits token ids plus their encoder-frame positions (×0.08 s) for word
timestamps. Device-neutral: FP32 on CPU and CUDA, both reproducing the
reference transcript (validated bit-faithful against the reference checkpoint).

Public surface: `include/brosoundml/parakeet.h` (`ParakeetConfig` + the
pipeline); the module graph is in `src/parakeet_modules.h`.

## Pipeline

```
   1. Log-mel front-end  16 kHz mono PCM ─▶ (128-mel × T) log-mel. NeMo recipe:
                         pre-emphasis 0.97, STFT (n_fft 512, win 400, hop 160,
                         symmetric Hann), power, Slaney mel, log(x + 2^-24),
                         per-feature mean/var normalization.
   2. FastConformer enc  8x conv2d subsampling (a full conv2d then depthwise-
                         separable convs), then 24
                         Conformer blocks (½-FFN macaron, Transformer-XL
                         relative-position attention, conv module
                         [pointwise ▶ GLU ▶ depthwise k=9 ▶ BatchNorm ▶ SiLU ▶
                         pointwise], ½-FFN, LayerNorm); a projector to width 640.
   3. TDT decoder        a prediction network (token embedding + 2-layer LSTM +
                         projection) and a joint network (relu(enc + dec) ▶
                         token+duration logits). Greedy TDT decode emits a token
                         and a frame-duration per step, skipping `duration`
                         encoder frames at once — TDT's speed-up.
```

The relative-position attention rides `brotensor::self_attention_bias_forward`:
the Transformer-XL content bias `pos_bias_u` folds into the Q-projection bias
and the position term `rel_shift(Q_v · p)` is supplied as the additive
attention bias.

## brotensor op coverage

`stft` + a host power loop + `matmul` (mel front-end), `conv2d` (subsampling +
conv module, depthwise via groups), `silu`, `sigmoid`, `layer_norm`,
`batch_norm_inference`, `self_attention_bias_forward` (rel-pos attention),
`matmul`. The prediction network embeds tokens by a row gather and composes its
LSTM cell from `matmul` / `sigmoid` / `tanh`; greedy TDT selection is a host
argmax. No op brotensor lacks.

## Tools

- `brosoundml_parakeet_transcribe` — WAV → text (also `--timestamps`).

Weights are fetched by `scripts/download-parakeet.sh`.
