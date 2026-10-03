# CLAP

[LAION CLAP](https://huggingface.co/laion/larger_clap_general) (Contrastive
Language-Audio Pretraining) maps a sound clip and a sentence into one 512-d
space, where the cosine between the two says how well the words describe the
sound. brosoundml runs `laion/larger_clap_general`: the HF `transformers`
`ClapModel` with an unfused HTSAT audio tower, a RoBERTa text tower and one
projection per tower. It is the ML half of `bro.ear`. A script scores a clip
against prompts ("a whistle", "rain on a tin roof") to judge how something
sounds without listening to it.

Every stage is hand-written on brotensor, device-neutral FP32 on CPU and CUDA.
That covers the `ClapFeatureExtractor` front-end, the HTSAT Swin encoder, the
RoBERTa encoder with its byte-level BPE, and both projections. It is pinned to
the `transformers` reference at each of those stages.

Public surface: `include/brosoundml/clap.h` (`Clap`, `ClapConfig`,
`ClapAudioOptions`, `ClapScore`). The internals are `src/clap.cpp` (load,
front-end, scoring), `src/clap_audio.cpp` (HTSAT) and `src/clap_text.cpp`
(RoBERTa).

## Pipeline

```
 audio  any rate, mono ─▶ 48 kHz (brosoundml::resample; matches torchaudio to 8e-9)
   1. Front-end    ClapFeatureExtractor, truncation "rand_trunc", padding "repeatpad":
                   a clip shorter than 10 s (480000 samples) is tiled int(480000/n)
                   times, then zero-padded to 10 s; exactly 10 s is used as is; a
                   longer clip is cropped to one 10 s window (see "Long clips").
                   ClapAudioOptions::pad picks the short-clip fill (see "Short
                   clips"); only ClapPad::Repeat is the reference's.
                   STFT n_fft 1024, hop 480, centred (reflect pad), periodic Hann,
                   power; Slaney mel 64 bins over 50-14000 Hz (Slaney norm);
                   10*log10(max(x, 1e-10)), no top_db. -> (1001 frames x 64 mels).
   2. HTSAT        BatchNorm over the 64 mel bins ▶ bicubic time stretch 1001->1024
                   (align_corners) ▶ fold (1024 x 64) into a 256 x 256 image
                   (reshape_mel2img) ▶ 4x4/4 patch conv, 1 -> 128 ch, LayerNorm ▶
                   4 Swin stages, depths 2/2/12/2, heads 4/8/16/32, window 8,
                   shift 4 on odd blocks (0 once a stage's grid fits one window),
                   relative-position bias, -100 shifted-window mask, 2x2 patch
                   merging between stages ▶ LayerNorm ▶ mean over the 64 tokens
                   -> 1024-d.
   3. Audio proj   Linear 1024->512 ▶ ReLU ▶ Linear 512->512 ▶ L2 normalise.
 text  ─▶ RoBERTa byte-level BPE (tokenizer.json via brolm's LayaTokenizer),
          <s> ... </s>, truncated at the 512-position table
   4. RoBERTa      12 post-LN layers (eps 1e-12), width 768, 12 heads; positions
                   pad_id+1.. (pad_id 1), token type 0; pooler tanh(dense(<s>)).
   5. Text proj    Linear 768->512 ▶ ReLU ▶ Linear 512->512 ▶ L2 normalise.
 score  cos = a·t;  logits = cos * exp(logit_scale_a) (≈ 38.66);
        probability = softmax(logits) over the prompts.
```

The Swin window attention is `brotensor::self_attention_bias_forward` with the
relative-position bias (plus the shift mask on seam windows) as the additive
bias. There is one bias per window kind: interior, last row, last column and
corner. The cyclic roll, the window partition and the patch merge are all row
permutations, precomputed at load as INT32 index tensors and applied with
`gather_rows`. There is no host round trip inside the tower.

## Short clips

The reference ("repeatpad") tiles a clip shorter than 10 s as many whole
times as fit, then zero-pads the rest. That suits a texture, but it turns a
one-shot into a pattern: a 0.3 s gunshot becomes 33 shots in a row, which
CLAP scores as a drum loop. `ClapAudioOptions::pad` makes the choice
explicit:

- `ClapPad::Repeat` is the reference, tiles then zeros.
- `ClapPad::Silence` places the clip once at the start, then zeros
  (transformers' `padding="pad"`).
- `ClapPad::Auto` (the default) is `Silence` below `kClapShortClipSeconds`
  (2 s, counted in 48 kHz samples after resampling) and `Repeat` from there
  to 10 s.

`Clap::fill_window` returns the 10 s window a clip becomes, and
`Clap::log_mel` takes the same `pad` argument, which defaults to `Repeat` so
that it keeps matching the reference. The parity test pins the end-to-end
check to `Repeat`, and it checks the fills themselves without weights.

## Long clips

The reference crops a clip longer than 10 s at a random offset, so a single
forward of a long clip is not reproducible. `ClapAudioOptions` makes the
choice explicit:

- `ClapLongMode::Mean` (the default) embeds k = ceil(n / 480000) evenly spaced
  10 s windows, from `start_i = round(i * (n - 480000) / (k - 1))`, covering
  the head and the tail. It averages the unit embeddings and renormalises.
  Every second of the clip counts, and the result is deterministic.
- `ClapLongMode::Crop` embeds one window at `crop_offset` samples. The default
  of -1 is the centre, and offsets past the end clamp to the last window. With
  the reference's offset it reproduces `ClapModel` exactly, which is how the
  parity test pins it.

A clip of 10 s or less has no choice to make; both modes give the same result.

## The score

`score()` returns three views of the same comparison:

- `similarity`: the raw cosine between the clip and each prompt, in [-1, 1].
  It is absolute, so a threshold on it means the same thing whatever else is
  in the prompt list. Calibrate any threshold on your own clips.
- `logits`: the cosine times the learnt logit scale `exp(logit_scale_a)`,
  about 38.66. This is `ClapModel.logits_per_audio`.
- `probability`: the softmax of the logits over the prompts given. It sums to 1,
  so it is relative: it answers "which of these describes it best". A
  confident 0.95 over three bad prompts still means the best of three bad
  prompts. To ask whether a sound is X at all, compare X against a few
  contrasting prompts, or threshold the cosine.

The text logit scale (`logit_scale_t`) is loaded but only `text_logit_scale()`
reports it. The audio-to-text direction uses `logit_scale_a`, as
`logits_per_audio` does.

## JS surface: `bro.ear`

The binding lives in `src/api/native_soundml_ear.cpp` and is installed by
`installSoundML` / `installSoundMLCompute`. `bro.ear` is shared: broaudio's DSP
half mounts on it too. The installer adds `loadClap` and `ClapModel` to an
existing `bro.ear` and creates the object only when it is absent. It never
replaces it.

```js
const clap = bro.ear.loadClap();                   // default dir, GPU by default
// bro.ear.loadClap(dir?, { device: 'cpu'|'cuda'|'vulkan'|'metal', onReady, onError })
clap.loaded; clap.device; clap.sampleRate;         // true, 'CUDA' (or 'Vulkan' / 'Metal'), 48000
clap.embeddingSize; clap.windowSeconds;            // 512, 10
clap.logitScale;                                   // ≈ 38.66

const r = clap.score(clip, ['a whistle', 'a dog barking', 'rain']);
r.scores        // Float32Array: softmax(similarities * logitScale), sums to 1
r.similarities  // Float32Array: raw cosines
r.logits        // Float32Array: similarities * logitScale
r.embedding     // Float32Array(512): the clip's unit embedding
r.bestIndex, r.best   // argmax, and its prompt text (null for a cached prompt)

clap.embedAudio(clip, opts?)          // Float32Array(512)
clap.embedText('a whistle')           // Float32Array(512)
clap.embedText(['a', 'b'])            // [Float32Array(512), ...]
clap.scoreEmbedding(embedding, prompts)   // score() without the audio tower
clap.dispose()
```

- **Clip.** A clip is any of these. Multi-channel audio is averaged to mono,
  and any rate is resampled to 48 kHz.
  - A WAV path (16-bit PCM, resolved like the host's `fs`).
  - A `Float32Array` at `opts.sampleRate`, 48000 by default.
  - `{ samples, sampleRate }`.
  - A Web Audio `AudioBuffer`, or anything with `getChannelData`,
    `numberOfChannels` and `sampleRate`.
- **Prompts.** A string, or an array whose entries are strings or cached
  `Float32Array(512)` text embeddings from `embedText`. With cached entries, a
  fixed vocabulary is encoded once.
- **Options.**
  - `opts.long` is `'mean'` (the default) or `'crop'`.
  - `opts.cropAt` is the crop's start in seconds (centred by default).
  - `opts.onDone(result, { cancelled, error? })` runs `score`, `embedAudio`,
    `embedText` or `scoreEmbedding` on a work thread and returns an
    `AsyncHandle`.
  - Without `onDone` the call blocks.
  - One operation at a time per model: a second call while one is in flight
    throws.
- **Default directory.** It is `<root>/weights/clap`. The first match wins
  from: the root `bro.tts.setAssetRoot` set, then `../brosoundml`,
  `./brosoundml` and `.`. When none exists, the error names every path it
  tried.
- **Errors.** Argument shape errors are `TypeError`s. A failed load, or a
  model call that threw, is an `Error` prefixed with the entry point.

`tests/test_soundml_api.cpp` covers the weights-free contract: mounting on a
host's `bro.ear`, the TypeErrors, the brand checks and the non-constructible
class. `tests/test_soundml_api_weights.cpp` loads the model and exercises
everything above, including under `BRONZE_GC_STRESS`.

## Weights

```
scripts/download-clap.sh              # -> weights/clap/{config,preprocessor_config,tokenizer*}.json,
                                      #    vocab.json, merges.txt, pytorch_model.bin (776 MB)
python scripts/convert-clap.py        # -> weights/clap/model.safetensors (FP32, 534 tensors)
```

The converter drops the checkpoint's non-parameter buffers (`position_ids`,
`token_type_ids`, `relative_position_index`, `num_batches_tracked`) and checks
every tensor's name and shape against the config. It refuses a fused
(`enable_fusion`) checkpoint, which this port does not implement. Loading also
rejects a preprocessor config other than `rand_trunc` / `repeatpad`.

## Parity

`tests/ref/gen_clap_fixture.py` runs `transformers.ClapModel` +
`ClapFeatureExtractor` in float64/float32 PyTorch and writes
`tests/fixtures/clap.bin`. `brosoundml_test_clap` checks against it on CPU and
on CUDA when available.

The clips cover each case:

- `short`: 2.7 s synthetic, repeat-padded.
- `exact10`: exactly 480000 samples.
- `hum`: a 0.58 s 44.1 kHz stereo WAV.
- `long`: an 11 s WAV, cropped at the reference offset, plus the window mean.
- `speech16k`: 16 kHz speech.

It also uses seven prompts, one of them with punctuation and a non-ASCII
character. The table gives the worst max-abs error over all clips and prompts.

| stage | CPU | CUDA | tolerance |
|-------|-----|------|-----------|
| resample to 48 kHz (vs torchaudio) | 7.5e-9 | 7.5e-9 | 2e-5 |
| log-mel (dB) | 1.1e-4 (2.2e-3 on `speech16k`) | same | 5e-3 |
| HTSAT pooled (1024-d) | 1.2e-5 | 1.2e-5 | 2e-4 |
| audio embedding, from the reference mel | 7.4e-7 | 6.7e-7 | 2e-5 |
| audio embedding, end to end from the clip | 1.4e-6 | 1.4e-6 | 2e-4 |
| window-mean embedding (long clip) | 7.3e-7 | 6.9e-7 | 2e-4 |
| token ids | exact | exact | exact |
| text embedding | 5.4e-7 | 1.7e-7 | 2e-5 |
| cosine similarity | 1.3e-6 | 1.1e-6 | 2e-5 |
| logits | 5.0e-5 | 4.3e-5 | 1e-3 |
| softmax probability | 4.2e-6 | 4.1e-6 | 1e-4 |
| CPU vs CUDA embeddings | audio 2.8e-7, text 4.6e-7 | | 2e-5 |

The `speech16k` log-mel outlier comes from the FP32 STFT against the
reference's float64. It sits in the empty 8-24 kHz band of a 16 kHz source,
80+ dB below the clip's peak, and its embedding still agrees to 1.4e-6.

**Determinism.** On one device, `embed_mel` and `embed_text` are
bit-identical across repeated calls, and the test asserts it. The binding
test asserts that an async `score` / `embedAudio` returns the same bits as
the sync call.

## Speed

| | audio tower, one 10 s window |
|---|---|
| CUDA (RTX-class) | ~48 ms (first call 80-340 ms) |
| CPU | ~8.8 s |

brotensor's CPU backend is scalar and single-threaded, `linear` included, so
the CPU path is for parity and GPU-less machines. The binding defaults to the
GPU. A long clip in `mean` mode costs one audio-tower pass per 10 s window.

## brotensor op coverage

These ops cover the whole model; it needs no op that brotensor lacks.

- **Front-end:** `stft` (CPU); the power, mel and dB steps are host loops.
- **HTSAT:** `batch_norm_inference`, `interp2d_align_corners_forward` (bicubic),
  `nchw_to_sequence`, `conv2d_forward`, `layernorm_forward_inference_batched`,
  `gather_rows`, `self_attention_bias_forward`, `gelu_exact_forward` and
  `sum_cols`.
- **Both towers:** `linear` for the projections, plus `relu_forward` and
  `tanh_forward`.

The model is FP32 throughout, about 776 MB of weights on the device.
