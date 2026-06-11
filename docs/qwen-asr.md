# Qwen3-ASR

Qwen3-ASR (Alibaba, Jan 2026) is the open speech-to-text series built from
Qwen3-Omni: 52-language ASR plus language ID. brosoundml targets the HF
checkpoints (`Qwen/Qwen3-ASR-0.6B` / `-1.7B`) — `config.json` +
`model.safetensors` in a model directory, plus the Qwen BPE tokenizer files
consumed by `brolm::qwen::Tokenizer` (held by the caller; brosoundml takes
already-tokenized context text and emits token ids). Device-neutral: `load()`
places the weights on the chosen `brotensor::Device` and the whole forward pass
dispatches through brotensor device ops, so a CUDA build reproduces the CPU
result.

Public surface: `include/brosoundml/qwen_asr.h` (`QwenAsrConfig`, the `QwenAsr`
pipeline); the encoder/decoder module graphs are in
`src/qwen_asr_encoder.h` / `src/qwen_asr_decoder.h`.

## Pipeline

```
   1. Log-mel front-end  16 kHz mono PCM -> (frames, 128) log-mel features.
                         Whisper's exact recipe (25 ms / 10 ms / Slaney /
                         log10 / max-8 clamp / (x+4)/4) but NOT padded to 30 s
                         — the encoder consumes the true frame count.
   2. AuT audio encoder  per-chunk Conv2d stem (3x 3x3 stride-2, applied to
                         independent 100-frame chunks; 8x time downsample,
                         12.5 Hz token rate) + per-chunk sinusoidal positions
                         + 18 pre-LN Transformer layers with block-windowed
                         bidirectional attention (~104-token windows) + a
                         2-layer projector into the decoder width.
                         Outputs (n_audio_tokens, 1024).
   3. Qwen3 text decoder the stock Qwen3 LLM (28 layers, GQA 16/8, per-head QK
                         RMSNorm, RoPE θ=1e6, SwiGLU). The chat-template prompt
                         embeds the audio tokens between <|audio_start|> /
                         <|audio_end|>; the transcript is decoded greedily with
                         a KV cache.
   4. Tokenizer          brolm::qwen::Tokenizer (separate dep) maps id -> text.
                         brosoundml returns the raw id stream.
```

The generated stream is the model's native output format —
`"language <Language><asr_text>transcript…"`, where `<asr_text>` is the special
id 151704; callers split on it (or decode everything and split on the literal
once detokenized).

## Options, context biasing, and the latent tap

`transcribe()` takes a `TranscribeOptions`:

- **`context_ids`** — optional context biasing: already-tokenized text (Qwen BPE
  ids) placed in the system block of the chat template, biasing recognition
  toward names / domain terms appearing in the context.
- **`on_token`** fires once per generated id for partial-text streaming;
  **`cancel`** is polled once per decoded token (the dominant cost);
  **`max_new_tokens`** caps the loop (default 1024; the model emits
  ~3 tokens/second of speech).

**Latent tap.** `encode()` runs the AuT encoder + projector only (no decoder)
and returns the post-projector latents as a `(T, latent_dim)` FP32 tensor on the
model device, at `config().latent_hz` (12.5 Hz at 16 kHz) — exactly the rows
`transcribe()` splices into the decoder over the `<|audio_pad|>` block (there is
one encoder path, not two). A host-copy overload writes the same latents into a
caller buffer so a bridge harness can drive a decoder on another device/library.

## brotensor op coverage

Same Qwen3-transformer op set as Qwen3-TTS — `rms_norm`, `rope`, `silu`/SwiGLU,
GQA self-attention (`flash_attention_varlen_forward`'s FP32 kernel),
`embedding_lookup`, `softmax`, and `argmax_rows` for the greedy step — plus
`stft` + a host power loop + `matmul` (mel front-end) and `conv2d` (the AuT
stem). No op brotensor lacks.

## Tools

- `brosoundml_qwen_asr_transcribe` — WAV → detected language + transcript
  (`--device cpu|cuda`, `--context TEXT`, `--max-new-tokens N`, `--stream`,
  `--ids`).

Weights are fetched by `scripts/download-qwen-asr.sh`.
