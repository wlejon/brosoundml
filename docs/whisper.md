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

The CLI's `--stream` flag and its automatic long-form for >30 s clips drive
both.

## brotensor op coverage

`stft` + a host power loop + `matmul` (mel front-end), `conv1d`, `gelu`,
`layer_norm`, `embedding_lookup`. Encoder self-attention uses brosoundml's FP32
`MHA` module (`modules.h`); the decoder's causal self-attention and cross-
attention are free functions over `brotensor::flash_attention_forward` with a KV
cache (FP16-cast Q/K/V on CUDA). Greedy token selection is a host argmax, not a
`sample_logits` op.

## Tools

- `brosoundml_transcribe` — WAV → text (also `--stream`).

Weights are converted by `scripts/convert-whisper.py` and fetched by
`scripts/download-whisper.sh`.
