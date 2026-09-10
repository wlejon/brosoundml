# OmniVoice

k2-fsa's OmniVoice (2026): zero-shot text-to-speech over 600+ languages whose
language model is a stock Qwen3-0.6B trunk with the text head replaced by eight
audio heads — one 1025-way classifier per HiggsAudio v2 codebook (1024 codes +
a MASK id) — decoded by **masked diffusion** instead of autoregression: every
frame of the output starts masked, and `num_steps` full bidirectional forwards
each fix the k most confident (codebook, frame) cells until none is masked.
Speech comes out of the HiggsAudio v2 codec ([higgs-codec.md](higgs-codec.md))
at 24 kHz, 25 frames per second.

Public surface: `include/brosoundml/omnivoice.h` (`OmniVoice`,
`OmniVoiceParams`, `OmniVoicePrompt`, `OmniVoiceInit`, `OmniVoiceStep`,
`OmniVoiceTrace`). Internals: `src/omnivoice.cpp` (pipeline: length rule,
chunking, codec, post-processing, prompt files), `src/omnivoice_lm.{h,cpp}`
(the trunk + heads + the diffusion step loop, CUDA-graph captured),
`src/omnivoice_prompt.{h,cpp}` (every text / duration / instruct / language /
pydub-audio rule, ported one-to-one from upstream's Python),
`src/omnivoice_unicode.inc` (Unicode general categories for the duration
rule), `src/omnivoice_lang_map.inc` (the 646-entry language table). Weights:
`weights/omnivoice/` — `config.json`, `tokenizer.json`, `model.safetensors`
(F32, `llm.*` + `audio_embeddings` + `audio_heads`), `audio_tokenizer/`.

Every numeric step is a brotensor op; FP32 on every backend, so CUDA reproduces
CPU to float round-off and the fixture tests gate both. No new kernels were
needed: the two masked-diffusion ops (`masked_diffusion_scores`,
`masked_diffusion_commit`) and `top_k_rows` already existed.

## Pipeline

```
text ──> resolve language (lang_map) + instruct (voice_design)
     ──> T = duration rule (per-script character weights; speed / duration)
     ──> [T > audio_chunk_threshold * 25 ? chunk at punctuation]
     ──> per chunk: prompt ids = style ids + <|text_start|>{ref_text + " " + text}<|text_end|>
                    LM: num_steps forwards over the packed cond + uncond rows
     ──> codec decode per chunk ──> cross-fade ──> post-process ──> 24 kHz PCM
```

**Prompt.** `style = [<|denoise|>]<|lang_start|>{lang|None}<|lang_end|>
<|instruct_start|>{instruct|None}<|instruct_end|>` (`<|denoise|>` only with a
reference and `denoise = true`), then the wrapped text. `_combine_text`:
`ref.strip() + " " + text.strip()`, `\r`/`\n` removed, full-width parentheses
ASCII-ised, `[ \t]+` collapsed, whitespace adjacent to a CJK character dropped.
Non-verbal tags (`[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-*]`,
`[surprise-*]`, `[dissatisfaction-hnn]`) are tokenized standalone so their ids
never depend on the surrounding language; everything between them goes
through the Qwen BPE (brolm's `qwen::Tokenizer` over `tokenizer.json`, whose
`added_tokens` carry the seven framing specials). The reference codes (8 x
T_ref) follow the text, embedded as the sum of the eight codebook embeddings
(`audio_embeddings[c * 1025 + code]`), and the T target frames come last,
embedded the same way with MASK (1024) while masked.

**Duration rule** (`RuleDurationEstimator`, exact port). Every character
carries a weight: ASCII letters 1.0, space 0.2, tatweel 0, then by Unicode
category — marks 0, punctuation / symbols 0.5, separators 0.2, numbers 3.5 —
then by block (CJK 3.0, Hangul 2.5, kana 2.2, Ethiopic / Yi 3.0, Indic 1.8,
Thai / Lao 1.5, Khmer / Myanmar 1.8, Arabic / Hebrew 1.5, Latin / Cyrillic /
Greek / Armenian / Georgian 1.0, unknown 1.0; planes above U+20000 count as
CJK). `est = weight(text) / (weight(ref_text) / T_ref)`, boosted below 50
frames as `50 * (est / 50)^(1/3)`; without a reference the fixed pair
("Nice to meet you.", 25 frames) stands in. `speed` divides the estimate;
`duration > 0` sets `T = max(1, int(duration * 25))` outright and records the
ratio `est / T` for chunk estimates. `T = max(1, int(est))`.

**Language / instruct.** `language` may be an ISO id from the table (passes
through, case-sensitive) or a name (case-insensitive, `languages()` lists the
display forms); anything else, "" or "none" is language-agnostic (`None`).
`instruct` is split at `,` / `，`, lower-cased and validated against the six
categories (gender, age, pitch, whisper, accent, dialect — `instruct_attributes()`);
a dialect forces the Chinese spelling of every item, an accent the English
one, otherwise Chinese is used when the target text contains CJK; two items of
one category, an unknown item, or a dialect mixed with an accent throw
`std::runtime_error` from `synthesize` / `generate_codes` / `estimate_frames`.
The normalised string is joined with `，` when any item is Chinese, `, ` otherwise.

## The LM

Qwen3-0.6B: 28 layers, hidden 1024, intermediate 3072, 16 query / 8 key-value
heads of 128, q/k RMSNorm, RoPE θ = 1e6, RMS eps 1e-6, tied text embeddings
(151 676 x 1024, kept host-resident: 620 MB of which a prompt reads a few
dozen rows), `audio_embeddings` (8200 x 1024) and `audio_heads` (8200 x 1024,
no bias). Attention is fully bidirectional — no KV cache is possible, every
step is a full forward.

One packed sequence per step (`src/omnivoice_lm.cpp`):

```
rows [0, Lc)        conditional document: text ids | ref codes | T target frames
rows [Lc, Lc + T)   unconditional document: the T target frames alone
```

Each document attends only within itself (`flash_attention_gqa_forward`,
`causal = false`, over row views of the projected Q/K/V) and restarts its
positions at 0 — the packed form of upstream's two-row batch, where the
unconditional row is target-only. The 2T target rows are gathered, final-normed
and projected by `audio_heads` into logits `(2T, 8 * 1025)`, the layout
`masked_diffusion_scores` takes (rows `[0, T)` conditional, `[T, 2T)`
unconditional). `guidance_scale = 0` packs the conditional document alone.

**Step loop** (`_generate_iterative`, exact). `timesteps = linspace(0, 1,
num_steps + 1)` in FP32, warped `t' = s·t / (1 + (s − 1)·t)` with
`s = t_shift`; step i unmasks `k_i = ceil(total_masked · (t'[i+1] − t'[i]))`
(FP64, capped by what is left; the last step takes the remainder). Per step:
rebuild the target-frame embeddings from the token grid (eight `gather_rows`
into the persistent embedding buffer, `scatter_rows` at the cond and uncond
target rows), forward, `masked_diffusion_scores` (CFG log-softmax, MASK
excluded, argmax or top-10 % Gumbel class sample, confidence − codebook ·
`layer_penalty`, `/ position_temperature + Gumbel`, −inf where fixed),
`top_k_rows` over the scores viewed as one `(1, 8T)` row, and
`masked_diffusion_commit`. The Gumbel uniforms are brotensor's counter-based
hash of `(seed_step, cell)` with `seed_step = hash(params.seed, chunk, step)`,
so CPU and CUDA draw identical noise and a seed reproduces an utterance;
`gumbel_noise = false` passes temperature 0 (argmax order, no noise).
`OmniVoiceInit` fixes cells before the schedule: kept cells start with their
token, are excluded from `total_masked`, and keep `unmask_step = −1`.
`cancel` is polled once per step; `on_step` receives the host grid, the scores
and the raw confidence after each commit, plus `chunk` / `num_chunks` (the
long-form chunk this step belongs to — `step` restarts at 0 for every chunk).

**Scores vs confidence.** `masked_diffusion_scores` writes two grids and they
answer different questions. `scores` is the *selection* score: the raw
confidence minus `codebook · layer_penalty`, then `/ position_temperature +
Gumbel` when that temperature is on, and −inf at every already-fixed cell — it
ranks the competition, and with `layer_penalty = 5` its range is dominated by
which codebook a cell sits in. `confidence` is the model's raw
`max(log_probs)` at the cell, before the penalty and before any noise, and it
is written for *every* cell including the already-unmasked ones (the model
predicts every target position on every forward, so those logits exist and
their max is a real number). That is the honest "where did the model hedge"
signal; a heat map wants `confidence`, the unmask order wants `scores`. With
`position_temperature = 0` the two are related exactly, in FP32:
`scores[c][t] == confidence[c][t] − c · layer_penalty` at every masked cell.
`OmniVoiceTrace::confidence` carries, per position, the confidence at the step
that position was committed (so it pairs with `unmask_step`); a cell an
`OmniVoiceInit` kept is never committed and carries the last step's value.

**CUDA graphs.** The per-shape session (one per distinct `(n_text, n_ref, T,
cfg)`, four cached) pre-allocates every buffer so the step body never
allocates; on CUDA the first step runs the body eagerly (warm-up), captures an
identical re-run into a `CudaGraph`, and every later step replays it with one
launch — the same pattern as the Qwen3-TTS Talker. The scores / top-k / commit
ops stay eager: `masked_diffusion_scores` takes its seed as a kernel argument,
so a replayed graph could not vary the noise per step (three small launches
per step). A chunked utterance captures one graph per chunk shape.

**Precision.** `OmniVoicePrecision::FP32` (default) widens everything; the
fixture tests run this mode. `BF16` narrows the q/k/v/o, gate/up/down and
`audio_heads` weights and runs those GEMMs on BF16 operands: each linear's
FP32 input is cast to a BF16 scratch, `linear_forward_batched_fp16` (the WMMA
tensor-core kernel, FP32 accumulation) writes a BF16 product, and a second
cast widens it back into the FP32 stream. Attention runs on the fused
FlashAttention-2 WMMA kernel (`flash_attention_forward`, BF16 operands, FP32
accumulation): the RoPE'd q/k and v are cast to BF16, k/v are expanded from
the 8 kv heads to the 16 q heads with one `gather_rows` over a fixed index
(the fused kernel has no GQA form), and the output lands directly in the
o_proj's BF16 input. The residual stream, norms, RoPE, the SwiGLU product,
embeddings and the score/commit ops stay FP32.

Two things were measured on the way here (RTX 4090, 250 frames, 32 steps).
Weight-only narrowing gave no speedup (90 ms/step both ways): brotensor's
FP32-activation `linear_forward_batched` runs the same 64x64 tiled SGEMM for
FP32 and 16-bit weights, and at L of several hundred rows the step is
compute-bound. With BF16 GEMMs and FP32 attention the step fell to 41 ms, of
which Nsight showed 70% in the FP32 `flash_attention_windowed_kernel` (one
block per query row per head, a decode-shaped kernel: ~0.54 ms per document
per layer). Moving attention onto the fused kernel is what the rest of the
gain comes from.

On CPU, BF16 mode emulates the same rounding (GEMM inputs rounded to BF16 in
place, FP32 dot over the BF16 weights, outputs rounded to BF16; q/k/v rounded
to BF16 before the FP32 GQA attention — the attention probabilities are not
rounded, so the emulation is close, not bit-level) through the scalar
16-bit-weight path, so it is the reference for the CUDA numbers but far
slower than FP32 on CPU. The discrete stream may differ from FP32 at
near-ties.

## Codec and post-processing

Each chunk's `(8, T)` codes go through `HiggsCodec::decode` (T · 960 samples);
several chunks are joined by `cross_fade_chunks` (0.3 s total: 2400-sample
linear fade-out, 2400 samples of silence, 2400-sample fade-in). Then
`_post_process_audio`: `remove_silence(mid 500 ms, lead 100 ms, trail 100 ms)`
when `postprocess_output` (pydub semantics: the waveform is quantised to
int16 by truncation, silence is `rms <= 103` on 10 ms steps at −50 dBFS,
`split_on_silence` keeps 500 ms around each segment with the midpoint overlap
fix, then the edges are trimmed keeping 100 ms), the loudness rule — with a
reference whose RMS is below 0.1, `audio *= ref_rms / 0.1`; without a
reference, peak-normalise to 0.5; otherwise nothing — and `fade_and_pad`
(linear `fade_duration` ramps in numpy's float64-then-float32 `linspace`,
`pad_duration` of silence each side), which upstream applies regardless of
`postprocess_output`; brosoundml keeps that behaviour.

**Voice prompts** (`create_prompt`): the clip is resampled to 24 kHz with the
codec's torchaudio-exact sinc resampler, its RMS recorded (that is the
`OmniVoicePrompt::rms` the output loudness rule reads), boosted to 0.1 RMS
when quieter, then with `preprocess`: `trim_long_audio` (clips over 20 s are
cut at the widest silence before 15 s, never below 3 s) — only when
`ref_text` is empty, as upstream only trims an untranscribed clip —
`remove_silence(200, 100, 200)`, an error if nothing remains; the length is
cut to a whole number of frames and the codec encoder (DAC + HuBERT) produces
the codes; `add_punctuation` appends `.` (or `。` for CJK) to the transcript.
brosoundml has no ASR, so an empty `ref_text` is allowed and simply yields a
text-free reference (upstream would auto-transcribe).

## Long text

When the estimate exceeds `audio_chunk_threshold` (30 s = 750 frames) the text
is split by `chunk_text_punctuation`: sentences end at `.,;:!?。，；：！？`
(a `.` after an abbreviation such as `Mr.` / `e.g.` does not split; a leading
punctuation or closing quote sticks to the previous sentence), merged up to
`chunk_len = int(audio_chunk_duration · 25 / (T / len(text)))` characters,
with chunks under 3 characters merged into a neighbour. With a reference every
chunk is estimated against it (times the speed ratio) and generated with it;
without one the first chunk is generated alone and its codes + text become
the reference for the rest. The trace concatenates the chunk grids along the
frame axis and lists `chunk_frames`. Each chunk restarts the diffusion
schedule, so `OmniVoiceStep::step` runs `0 … num_steps-1` once per chunk;
`OmniVoiceStep::chunk` (0-based) and `num_chunks` say which one, and a
consumer that wants a single timeline should key on `(chunk, step)`. An
unchunked `synthesize` and every `generate_codes` report `chunk = 0`,
`num_chunks = 1`.

## Prompt files (`OmniVoicePrompt::save` / `load`)

A small self-describing little-endian binary, magic `OVCP`:

```
char[4]  "OVCP"
u32      version (1)
i32      num_codebooks
i32      num_frames
f32      rms
i32      text_bytes;  u8 text[text_bytes]      (UTF-8 transcript)
i32      codes[num_codebooks * num_frames]      ([q * num_frames + t])
```

## brotensor op map

| stage | op |
|---|---|
| text embedding | `embedding_lookup_forward` (host table), `copy_d2d` |
| audio-frame embedding | `gather_rows` (per codebook, row views of `audio_embeddings`), `add_inplace`, `scatter_rows` |
| RMSNorm / q,k norm | `rms_norm_forward` (per-head via a `(L·heads, 128)` view) |
| projections, MLP, heads | `linear_forward_batched` (FP32 or BF16 weights) |
| RoPE | `rope_apply` (adjacent-pair; q/k rows and q/k norm permuted at load) |
| attention | `flash_attention_gqa_forward`, non-causal, per document |
| SwiGLU | `silu_forward` + `mul_inplace` |
| target rows | `gather_rows`, `rms_norm_forward`, `linear_forward_batched` |
| step | `masked_diffusion_scores`, `top_k_rows`, `masked_diffusion_commit` |
| CUDA | `CudaGraphCapture` / `CudaGraph::launch` |
| resample | `pad1d_forward` + `conv1d` (the codec's sinc resampler) |

## Validation (`tests/test_omnivoice.cpp`)

Fixtures come from `tests/ref/gen_omnivoice_fixture.py`, which runs the
genuine upstream implementation (pinned commit, FP32, CUDA, TF32 off) and
records every intermediate through hooks; regenerate with
`OMNIVOICE_SRC=<upstream checkout> python tests/ref/gen_omnivoice_fixture.py`.
The CPU pass runs only the model-free stages (A, B, F, the Part E
preprocessing and the OVCP round trip) without loading any weights; every
stage that runs the LM trunk or the codec (C, D, E, the model contract checks,
end-to-end, timings) runs on CUDA only and is skipped with a message when CUDA
is unavailable — the upstream fixtures, generated on CUDA, are the oracle:

* **A** tokenizer — plain and tag-aware ids of 65 strings, exact.
* **B** prompt assembly — 18 cases (language / instruct resolution, style and
  combined text, ids, audio-mask layout, T, speed ratio) exact; the duration
  rule — 53 cases (weights, raw estimate, frames) exact.
* **C** the first forward — step-0 logits at the target rows, the MASK-frame
  embedding, text embeddings, the packed input embeddings and the post-norm
  hidden rows; then the same forward in BF16 mode against the FP32 fixture
  (max |Δ| < 3; the step-0 per-cell argmax agreement is reported with a
  > 50 % floor — at step 0 every cell is masked and most rows are near-flat)
  and the 16-step BF16 generation (codes / grid agreement reported; every
  cell must commit to an in-range code). The BF16 correctness gate is the
  Whisper transcript of the BF16 end-to-end synthesis.
* **D** 16-step deterministic generation (Gumbel uniform fixed) — per-step
  unmask counts exact, per-step predictions and scores, final codes and the
  unmask-order grid; the raw confidence is checked against the score it is
  derived from (`score == confidence − codebook · layer_penalty` bit-exactly,
  the run having `position_temperature = 0`) at every masked cell of every
  step, is finite everywhere including the already-fixed cells, and the trace
  confidence is finite at every position; decode + post-processing of the
  reference codes.
* **E** voice clone — preprocessing reproduces the encoder input exactly,
  prompt codes 100 %, the OVCP round trip, the cloned generation.
* **F** `remove_silence` / gain / `fade_and_pad` on two signals x 3 variants,
  exact.
* The contract checks, an end-to-end synthesis of "Hello there, this is a
  test of the OmniVoice pipeline." (FP32 and BF16) and a clone, all
  transcribed by Whisper (`weights/whisper`), a chunked long paragraph, and
  FP32 / BF16 timings of a 10 s utterance at 32 steps.

Build and run:

```
cmake --build build-omni-codec --config Release --target brosoundml_test_omnivoice brosoundml_omnivoice_say
build-omni-codec/tests/Release/brosoundml_test_omnivoice.exe
```

## CLI

```
brosoundml_omnivoice_say <model_dir> "<text>" <out.wav>
    [--device cpu|cuda] [--bf16] [--lang X] [--instruct "..."]
    [--ref ref.wav --ref-text "..."] [--prompt file.ovcp | --save-prompt file.ovcp]
    [--steps N] [--guidance G] [--speed S] [--duration D] [--seed N]
    [--no-noise] [--no-post] [--trace]
```

Prints load / LM / codec timings; `--trace` adds the prompt ids, the chunk
frame counts and the unmask-order grid (one row per codebook, one character
per frame, ` .:-=+*#%@` from first to last step, `k` for cells kept by an
init).

## Deviations from upstream

* The step's scores / top-k / commit run eagerly outside the CUDA graph (the
  scores op's seed is a kernel argument); the graph holds the embedding
  rebuild, the trunk and the heads.
* Gumbel noise comes from brotensor's counter-based hash RNG seeded by
  `(seed, chunk, step)`, not torch's generator — deterministic and identical on
  CPU and CUDA, but a given seed does not reproduce a specific PyTorch draw.
* `create_prompt` trims a >20 s clip only when `ref_text` is empty (upstream:
  only when the transcript is `None`, since it would otherwise auto-transcribe);
  there is no ASR fallback, an empty transcript is accepted as-is.
* Instruct errors list the unsupported items and the valid vocabulary but
  offer no "did you mean" suggestion.
* The Unicode category table is Unicode 16.0 (perl 5.42); a Python built
  against an older database would weigh characters new in 16.0 differently.
* Unrecognised languages fall back to language-agnostic silently (upstream
  logs a warning).
