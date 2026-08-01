# Whisper

OpenAI's encoder-decoder speech-to-text model. brosoundml targets the HF
transformers checkpoints (`whisper-tiny` / `-base` / `-small` / `-medium` /
`-large-v3`) — `config.json` + `model.safetensors` in a model directory.
Tokenization is delegated to `brolm::whisper::Tokenizer`; brosoundml itself
takes already-tokenized prompts and emits token ids.

Public surface: `include/brosoundml/whisper.h` (`WhisperConfig`, the
encoder/decoder pipeline), `include/brosoundml/whisper_modules.h` (conv stems,
the cross-attention decoder).

## Pipeline

```
   1. Log-mel front-end  16 kHz mono PCM ─▶ log-mel spectrogram
                         (num_mel_bins × 3000 frames, 30 s padded/truncated).
                         stft + mel-filterbank matmul + log.
   2. Encoder            two conv1d stems (the second strided ×2) + sinusoidal
                         positional embeddings + a pre-LN Transformer stack.
   3. Decoder            cross-attention Transformer with a KV cache,
                         autoregressive greedy decode.
   4. Tokenizer          brolm::whisper::Tokenizer (external) maps id ─▶ text.
```

## Streaming and long-form

`transcribe()` has a `TranscribeOptions` overload for realtime use:

- **`on_token`** fires per decoded id so a caller can emit partial text
  mid-utterance instead of waiting for the whole clip.
- **`timestamp_begin_id`** opts into sequential long-form decode — audio past the
  fixed 30 s log-mel window is split into 30 s segments and advanced by each
  segment's last emitted timestamp (falling back to full-30 s hops) instead of
  being truncated. Only that single int crosses the tokenizer boundary; the
  caller still owns `build_prompt` / `decode`
  (`brolm::whisper::Tokenizer::first_timestamp_id()` supplies it).
- **`no_timestamps_id`** forbids the greedy loop from ever selecting
  `<|notimestamps|>`. Pass it (`Tokenizer::no_timestamps_id()`) whenever you
  want timings at all. Building the prompt with `with_timestamps=true` only
  *omits* the token from the prefix — it does not stop the decoder generating
  it, and whisper-tiny does exactly that on a clean 11 s clip, after which the
  segment carries no timestamp and seek degrades to blind 30 s hops. HF's
  Whisper names the same token in `generation_config.json`
  (`no_timestamps_token_id`) and forces its logit to -inf when timestamps are
  requested; brosoundml reads no generation_config, so the id comes from the
  caller. The suppression is free — the greedy argmax already materialises the
  logits row on the host, so it is one skipped index rather than a masking
  kernel per token.

Measured on an RTX 4090, large-v3, an 11 s clip: without `no_timestamps_id` the
run returns 0 timestamps; with it, `<|0.00|> … <|11.00|>`.

The CLI's `--stream` flag and its automatic long-form for >30 s clips drive
both.

Long-form callers should know that the timestamps a segment emits are
**window-relative** — every 30 s window restarts at `<|0.00|>` — and the
returned token stream does not mark where one window ended and the next began.
A caller that needs absolute times should drive the 30 s windows itself and add
each window's own offset, rather than reading them out of one long-form run.

## brotensor op coverage

`stft` + a host power loop + `matmul` (mel front-end), `conv1d`, `gelu`,
`layer_norm`, `embedding_lookup`. Encoder self-attention uses brosoundml's FP32
`MHA` module (`modules.h`); the decoder's causal self-attention and cross-
attention are free functions over a KV cache — `flash_attention_forward` on
CPU, FP32 `flash_attention_windowed_forward` on GPU so the cached K/V never
round-trips through FP16. Device-neutral CPU + CUDA: weights load to the
requested device and the encoder/decoder forwards stay on-device (one logits
download per decode step). Greedy token selection is a host argmax, not a
`sample_logits` op.

## Tools

- `brosoundml_transcribe` — WAV → text (also `--stream`).

Weights are converted by `scripts/convert-whisper.py` and fetched by
`scripts/download-whisper.sh`.
