# Qwen3-TTS

Qwen3-TTS is Alibaba's open-weight TTS series (Jan 2026). brosoundml targets the
12 Hz multi-codebook track — an end-to-end discrete-token model, no diffusion,
no external vocoder. The whole pipeline runs device-neutrally on **CPU and
CUDA**: `load(device)` places the weights on the chosen backend, and because
compute is FP32 on both, CUDA reproduces the CPU/upstream discrete-code stream
bit-for-bit (the codec tail then matches to ~1e-5). This is the reference
device-neutral model in the library.

Public surface: `include/brosoundml/qwen_tts.h` (the `QwenTts` pipeline,
Talker / CodePredictor / codec configs). The implementation is split across
`src/qwen_tts{,_talker,_code_predictor,_generate,_codec,_codec_encoder,_speaker_encoder}.cpp`.

## Pipeline

```
   1. Text         text -> Qwen BPE token ids (brolm Qwen tokenizer; the
                   vocab.json + merges.txt ship in the model dir).
   2. Talker       a 28-layer Qwen3 decoder backbone over a dual stream: text-
                   token embeddings (projected to the Talker hidden) interleaved
                   with codec-token embeddings, GQA + QK-norm + interleaved
                   M-RoPE. Per frame it emits a hidden state and, via codec_head,
                   acoustic codebook 0.
   3. Code         a 5-layer depth transformer that, conditioned on the Talker
      Predictor    hidden, autoregressively emits acoustic codebooks 1..15.
   4. Codec        the bundled 12 Hz codec (speech_tokenizer/): 16 RVQ codes per
      decoder      frame ─▶ dequantize ─▶ windowed pre-transformer ─▶ ConvNeXt
                   upsample ─▶ a SEANet causal-conv decoder with Snake activations
                   ─▶ 24 kHz. Total upsample 1920; 12.5 Hz · 1920 = 24 kHz.
```

The autoregressive decode is the floor on synthesis speed. Two performance
levers shave it:

- **Captured Talker step.** On CUDA the per-frame Talker decode (fixed-cap KV +
  masked attention) is captured as a CUDA graph once and replayed per frame —
  collapsing per-step launch overhead, the same fix applied to the Code
  Predictor's GEMVs and Kokoro's LSTM.
- **BF16 weight mode.** `load(dir, device, QwenTtsWeightPrecision::BF16)` keeps
  the Talker / Code Predictor projection, MLP and head weights at the
  checkpoint's native BF16 instead of widening them (activations, accumulation,
  norms, embeddings and the KV cache stay FP32) — halving the weight-read
  bandwidth that floors the autoregressive decode. The kernels widen each BF16
  weight to FP32 in register and accumulate in the same order as the FP32 path,
  so the output has matched FP32 mode byte-for-byte in practice (the checkpoint
  is natively BF16); FP32 stays the fixture-gated reference. The
  `BROSOUNDML_QWEN_BF16` env var switches the qwen-tts tools to this mode.

## Variants and voice control

`QwenTts::synthesize()` picks the voice by checkpoint variant:

- **CustomVoice** — pass a preset `speaker` name (see `speakers()`). The 1.7B
  checkpoint also honours an `instruct` style prompt.
- **VoiceDesign** — pass a natural-language `instruct` describing the voice
  (e.g. *"a warm, low-pitched elderly storyteller"*); there are no presets.
- **Base** — zero-shot voice cloning. `synthesize_clone()` encodes a reference
  clip into an ECAPA-TDNN speaker x-vector and splices it into the Talker prefill
  where a CustomVoice preset token would sit ("x-vector-only" enrollment, no
  reference transcript). `embed_speaker()` exposes that enrollment step on its
  own — the honest audio→identity front-end for training an adapter into another
  model's style space — and `synthesize_with_xvector()` renders straight from a
  supplied x-vector, so a caller can enroll real voices and then interpolate /
  morph / steer in that 1024-d identity space before speaking.

**Standalone speaker encoder.** The Base ECAPA-TDNN extractor is also liftable
out of the ~2.5 GB checkpoint: `brosoundml_build_speaker_encoder` packs it into
a ~18 MB two-file artifact, and the `SpeakerEncoder` class
(`speaker_encoder.h`) loads just that to enroll a clip — bit-identical to
`QwenTts::embed_speaker` — without touching the Talker, codec, or tokenizer it
never needs.

## Streaming, steering, and tracing

- **Streaming.** `synthesize_stream()` decodes the growing code stream as the AR
  loop runs and hands each new 24 kHz chunk to an `on_chunk` sink (the codec is
  causal, so delivered samples never change).
- **Steering.** `QwenTtsSampling` exposes optional seeded sampling
  (temperature/top-p, reproducible per `seed`) and codebook-0 logit steering
  (repetition penalty, a logit bias, and an adaptive temperature that nudges only
  the frames where the model hedged); the default (temperature 0) stays the
  greedy, bit-exact upstream policy.
- **Tracing.** Passing a `QwenTtsTrace*` to `synthesize()` captures the per-frame
  codes and codebook-0 confidence for visualization at near-zero cost.
- **Cancel.** The AR loop polls a `CancelCheck` once per generated frame so a
  long synthesis can be aborted.

The bundled codec is reachable directly: `encode_audio()` (waveform → RVQ codes)
and `decode_codes()` (RVQ codes → 24 kHz waveform) are inverses that round-trip,
useful on their own once a caller holds a code stream.

## brotensor op coverage

`rms_norm`, `rope`, `silu`/SwiGLU, GQA self-attention (CUDA via
`flash_attention_varlen_forward`'s FP32 kernel), `conv1d` / `conv_transpose1d`,
`snake`, `embedding_lookup`, `softmax`, `sample_logits` / `argmax` — all already
on the op surface. Qwen3-TTS adds no op brotensor lacks. The full tensor map is
in [qwen-tts-weights.md](qwen-tts-weights.md).

## Tools

- `brosoundml_qwen_tts_say` — minimal one-shot driver (load → synthesize one
  line → WAV); small enough to sit under nsys/ncu for kernel-level profiling.
- `brosoundml_qwen_tts_bench` — synthesis throughput benchmark.
- `brosoundml_qwen_tts_roundtrip` — codec encode↔decode round-trip.
- `brosoundml_qwen_tts_clone` — zero-shot voice clone from a reference clip.
- `brosoundml_build_speaker_encoder` — pack the standalone ECAPA enroller.

Weights are fetched by `scripts/download-qwen-tts.sh`.
