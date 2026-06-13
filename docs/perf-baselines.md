# Performance baselines

Reference numbers captured before the QwenTts session work, so sessioning a model
can be proven not to regress its forward path. The session API moves per-stream
scratch into a caller-owned `Session` but does not touch the compute graph, so a
sessioned model must reproduce these to within run-to-run noise (and bit-exact
where a test already pins it).

**Hardware:** NVIDIA GeForce RTX 4090 (24 GB), Windows, Release build, brotensor
CUDA backend. Each figure is the best (min) wall time over the bench's warmed
iterations, captured with the tools under `tools/`.

## Kokoro-82M — `brosoundml_kokoro_bench --device cuda`

Voice `af_heart`, 152-char line → 155 phonemes → 9.47 s of 24 kHz audio.

| Metric | Value |
|---|---|
| mean synth time | 263.7 ms |
| realtime factor  | **35.9× RT** |

## Qwen3-TTS 0.6B (CustomVoice) — `brosoundml_qwen_tts_bench`

Speaker `serena`, English. Codec is FP32 on both weight modes; the AR Talker /
CodePredictor loop is what BF16 accelerates.

### codec `decode_codes` (CUDA) — independent of weight precision

| frames T | ms (FP32 / BF16) | audio | RT | ms/frame | windowed |
|---|---|---|---|---|---|
| 72   | 250.2 / 247.8 | 5.76 s  | ~23×  | ~3.47 | no  |
| 144  | 508.3 / 516.3 | 11.52 s | ~22×  | ~3.55 | yes |
| 300  | 1186 / 1179   | 24.0 s  | ~20×  | ~3.94 | yes |
| 600  | 2831 / 2855   | 48.0 s  | ~17×  | ~4.74 | yes |
| 1200 | 7351 / 7276   | 96.0 s  | ~13×  | ~6.1  | yes |

### full `synthesize()` (CUDA) — end to end, text → 24 kHz wav

| text | FP32 | BF16 |
|---|---|---|
| "Hello there." (1.28 s audio)       | 328.0 ms · **3.90× RT** | 255.0 ms · **5.02× RT** |
| long sentence (10.96 s audio)       | 2593 ms · **4.23× RT**  | 1997 ms · **5.49× RT** |

Notes:
- BF16 is byte-identical to FP32 on the discrete code stream (the checkpoint is
  natively BF16); it only changes the AR-loop weight bandwidth, hence the
  synthesize() speedup with no codec change.
- The per-frame Talker/CodePredictor step time (the figure the AR-loop
  engineering targeted) is captured separately by the profiling paths; the
  end-to-end `synthesize()` RT above is the consumer-visible number to hold.
