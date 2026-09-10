# HiggsAudio v2 codec

The neural audio codec OmniVoice speaks: Boson AI's HiggsAudio v2 tokenizer
(transformers `HiggsAudioV2TokenizerModel`, `bosonai/higgs-audio-v2-tokenizer`).
24 kHz mono in, 25 codes per second out — eight residual-VQ codebooks of 1024
entries (a frame is `hop_length = 960` samples) — and back. brosoundml runs both
halves: the **decoder** serves synthesis (OmniVoice -> codes -> waveform), the
**encoder** (with its HuBERT-base semantic branch) serves voice cloning
(reference clip -> codes) and analysis. Both load from the one
`audio_tokenizer/` directory (`config.json` + `model.safetensors`, 527 F32
tensors, 806 MB); `decoder_only = true` skips the encoder + HuBERT.

Public surface: `include/brosoundml/higgs_codec.h` (`HiggsCodec`,
`HiggsCodecConfig`). Internals: `src/higgs_codec.cpp` (module graph +
loader), `src/higgs_codec_model.h` (the white-box `HiggsCodecModel`),
`src/higgs_codec_common.h` (upload / layout / conv helpers + the sinc
resampler), `src/hubert.{h,cpp}` (HuBERT-base). Every step is a brotensor op
— conv1d / conv_transpose1d / pad1d, snake, elu, exact gelu, layernorm /
group_norm, flash attention, embedding lookup, vq_encode, linear — FP32 on
every backend, so CUDA reproduces CPU to float round-off. No new kernels.

## Pipeline

```
decode   codes (8, T)
         -> per level q: embed_q[code] (1024 x 64 table) -> project_out (64 -> 1024)
         -> sum over the 8 levels                                     (T, 1024)
         -> fc2 (1024 -> 256)                                          (T, 256)
         -> DAC decoder: conv 256 -> 1024 k7 "same"
            5 blocks (stride 8, 5, 4, 2, 3; channels 1024 -> 512 -> 256 -> 128 -> 64 -> 32):
              Snake -> ConvTranspose1d(k = 2s, padding = ceil(s/2),
                       output_padding = s % 2)  [Lout == L*s exactly]
              -> 3 residual units (Snake -> conv k7 dil {1,3,9} "same" -> Snake -> conv k1, + x)
            Snake -> conv 32 -> 1 k7. No tanh.                          T * 960 samples

encode   24 kHz mono, right-padded to a whole frame (n = T*960)
         acoustic  DAC encoder: conv 1 -> 64 k7 "same"; 5 blocks (stride 8, 5, 4, 2, 3;
                   64 -> 128 -> 256 -> 512 -> 1024 -> 2048): 3 residual units -> Snake ->
                   conv k = 2s, padding ceil(s/2); Snake; conv 2048 -> 256 k3      (256, T)
         semantic  resample 24 -> 16 kHz (windowed sinc, see below) -> pad 160 each side
                   -> HuBERT-base, mean of its 13 hidden states (2T frames)
                   -> every 2nd frame (semantic_downsample_factor 2)               (T, 768)
                   -> SemanticEncoder: conv 768 k3 (no bias); 2 stride-1 blocks of
                      [2 residual units (ELU -> conv k3 -> ELU -> conv k1, no bias, + x)
                       -> conv k3 (+bias)]                                          (768, T)
         concat [256 acoustic | 768 semantic] -> fc (1024 -> 1024)                 (T, 1024)
         residual VQ, 8 levels: project_in (1024 -> 64) -> nearest codeword by
         Euclidean distance -> residual -= project_out(codeword)                    codes (8, T)
```

With `n` a multiple of 960 the two encoder branches agree on `T` with no
extra padding (the reference pads the acoustic input by `hop/2` only when
they disagree): the acoustic strides divide exactly, and 16 kHz gives
`640·T + 320` padded samples -> HuBERT's stride-320 / receptive-field-400
stack yields exactly `2T` frames. `HiggsCodec::encode()` resamples any input
rate to 24 kHz and right-pads to the frame, so `encode ▸ decode` round-trips
with `T*960` samples out.

### HuBERT-base (`src/hubert.h`)

transformers `HubertModel` with the tokenizer's config (`do_stable_layer_norm
= false`, `feat_extract_norm = group`, `feat_proj_layer_norm = true`,
`conv_bias = false`):

- feature extractor: 7 valid strided convs 1 -> 512 (kernels 10,3,3,3,3,2,2;
  strides 5,2,2,2,2,2,2), no bias; layer 0 is followed by a per-channel
  GroupNorm (`num_groups == channels`, eps 1e-5); every layer ends in exact GELU;
- feature projection: LayerNorm(512) -> Linear(512 -> 768);
- positional conv: grouped Conv1d(768, k 128, groups 16, padding 64), the
  SamePad trim of the last sample (done as pad (64, 63) + a valid conv), GELU,
  added to the features, then `encoder.layer_norm` — that sum is hidden state 0;
- 12 post-LN layers: `h = LN(h + attn(h)); h = LN2(h + fc2(gelu(fc1(h))))`,
  12 heads of 64, biases everywhere, bidirectional (no mask);
- output: the mean of the 13 hidden states (state 0 + each layer's output).

The positional conv's weight norm is stored as parametrizations
(`original0` = g `[1,1,128]`, `original1` = v `[768,48,128]`, `dim = 2`) and
folded at load: `w[o,i,k] = g[k] · v[o,i,k] / ‖v[:,:,k]‖`.

### 24 -> 16 kHz resample

The reference resamples with `torchaudio.functional.resample` (windowed sinc,
`sinc_interp_hann`, `lowpass_filter_width 6`, `rolloff 0.99`). Measured with
brotensor's linear `resample1d_forward` in its place, HuBERT's codes move:
codebook-0 agreement 98.7% and total 93.3% on the 3 s test clip (100% / 99%
on a 1 s chirp), with the 16 kHz signal off by up to 5.5e-2 (peak 0.32) — more
than the ~1% budget. torchaudio's resample *is* a strided conv1d (one output
channel per output phase, stride = the gcd-reduced input rate, kernel
`2·width + orig` taps), so `hcodec::SincResampler` composes it from `pad1d`
+ `conv1d` + the NCL -> SEQ transpose (which interleaves the phases), with
the kernel built host-side once at load. It matches torchaudio to 2.4e-7 and
every codebook then agrees 100%. The same resampler converts `encode()`'s
input to 24 kHz.

## Tensor map (`audio_tokenizer/model.safetensors`)

All F32. Weight norm on the DAC convs is already folded (plain `weight`).
Conv weights `[Cout, Cin, k]` upload as `(Cout, Cin*k)`; transposed convs
`[Cin, Cout, k]` as `(Cin, Cout*k)` (brotensor's `conv_transpose1d` layout);
Snake alphas `[1, C, 1]` as `(C, 1)`, raw (the DAC `Snake1d` does not
exponentiate).

| Key | Shape | Used by |
|---|---|---|
| `quantizer.quantizers.{q}.codebook.embed` | [1024, 64] | codebook `q` (both directions) |
| `quantizer.quantizers.{q}.project_out.{weight,bias}` | [1024, 64], [1024] | RVQ decode / residual update |
| `quantizer.quantizers.{q}.project_in.{weight,bias}` | [64, 1024], [64] | RVQ encode |
| `quantizer.quantizers.{q}.codebook.{embed_avg,cluster_size,inited}` | — | EMA training state, ignored |
| `fc2.{weight,bias}` | [256, 1024], [256] | decoder input |
| `acoustic_decoder.conv1.{weight,bias}` | [1024, 256, 7] | decoder |
| `acoustic_decoder.block.{i}.snake1.alpha` | [1, 1024>>i, 1] | block pre-Snake |
| `acoustic_decoder.block.{i}.conv_t1.{weight,bias}` | [in, out, 2s] | upsampler |
| `acoustic_decoder.block.{i}.res_unit{1,2,3}.{snake1.alpha, conv1.weight [C,C,7], conv1.bias, snake2.alpha, conv2.weight [C,C,1], conv2.bias}` | | residual units (dil 1, 3, 9) |
| `acoustic_decoder.snake1.alpha`, `acoustic_decoder.conv2.{weight,bias}` | [1,32,1], [1, 32, 7] | output |
| `acoustic_encoder.conv1.{weight,bias}` | [64, 1, 7] | encoder |
| `acoustic_encoder.block.{i}.res_unit{1,2,3}.*` | at 64<<i | residual units |
| `acoustic_encoder.block.{i}.snake1.alpha`, `.conv1.{weight,bias}` | [out, in, 2s] | strided downsampler |
| `acoustic_encoder.snake1.alpha`, `acoustic_encoder.conv2.{weight,bias}` | [1,2048,1], [256, 2048, 3] | latent |
| `encoder_semantic.conv.weight` | [768, 768, 3] | SemanticEncoder |
| `encoder_semantic.conv_blocks.{i}.res_units.{j}.conv{1,2}.weight` | [768,768,3], [768,768,1] | (no bias) |
| `encoder_semantic.conv_blocks.{i}.conv.{weight,bias}` | [768, 768, 3] | |
| `fc.{weight,bias}` | [1024, 1024], [1024] | concat -> RVQ input |
| `semantic_model.feature_extractor.conv_layers.{0..6}.conv.weight` | [512, Cin, k] | HuBERT features |
| `semantic_model.feature_extractor.conv_layers.0.layer_norm.{weight,bias}` | [512] | GroupNorm affine |
| `semantic_model.feature_projection.{layer_norm,projection}.{weight,bias}` | [512], [768, 512] | |
| `semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original{0,1}`, `.bias` | [1,1,128], [768,48,128] | folded pos conv |
| `semantic_model.encoder.layer_norm.{weight,bias}` | [768] | pre-layer LN |
| `semantic_model.encoder.layers.{l}.attention.{q,k,v,out}_proj.{weight,bias}` | [768, 768] | |
| `semantic_model.encoder.layers.{l}.{layer_norm,final_layer_norm}.{weight,bias}` | [768] | |
| `semantic_model.encoder.layers.{l}.feed_forward.{intermediate_dense,output_dense}.{weight,bias}` | [3072,768], [768,3072] | |
| `decoder_semantic.*`, `fc1.*` | — | training-only (semantic reconstruction), never read |

`config.json`: `sample_rate 24000`, `semantic_sample_rate 16000`,
`codebook_size 1024`, `codebook_dim 64`, `target_bandwidths [.., 2]` ->
`num_quantizers = 1000·2 / (25·10) = 8`, `acoustic_model_config`
(`hidden_size 256`, `encoder_hidden_size 64`, `decoder_hidden_size 1024`,
`downsampling_ratios == upsampling_ratios == [8,5,4,2,3]` -> `hop 960`,
`frame_rate 25`), `semantic_model_config` (the HuBERT fields above),
`downsample_factor 320` -> `semantic_downsample_factor = 960 / 1.5 / 320 = 2`,
`kernel_size 3`, `unit_kernel_size 3`, `strides / channel_ratios /
block_dilations [1,1]`.

## Op map

| Stage | brotensor ops |
|---|---|
| RVQ decode | `embedding_lookup_forward` (codes -> rows), `linear_forward_batched` (project_out), `add_inplace` |
| RVQ encode | `linear_forward_batched` (project_in / project_out), `vq_encode_forward`, `scale_inplace` + `add_inplace` (residual) |
| "same" / strided / valid convs | `conv1d` (conv2d with H = 1) with padding, stride, dilation, groups |
| DAC upsampler | `conv_transpose1d_forward` (padding ceil(s/2), output_padding s % 2) |
| Snake | `snake_forward` (beta = null: plain Snake) |
| SemanticEncoder activations | `elu_forward` |
| HuBERT GroupNorm / LayerNorm / GELU | `group_norm_forward` (num_groups = C, H = 1, W = L), `layernorm_forward_inference_batched`, `gelu_exact_forward` |
| HuBERT attention | `flash_attention_gqa_forward` (q heads == kv heads, causal = false, FP32) |
| Positional conv | `pad1d_forward` (64, 63) + `conv1d` (groups 16) |
| Resample | `pad1d_forward` + strided `conv1d` (one channel per phase) + `nchw_to_sequence` (phase interleave) + `copy_d2d` (trim) |
| Layout | `nchw_to_sequence` / `sequence_to_nchw` (NCL <-> SEQ), `concat_nchw_channels` ([acoustic | semantic]) |

## Validation (`tests/test_higgs_codec.cpp`)

Fixture: `tests/ref/gen_higgs_codec_fixture.py` runs the reference in FP32 on
CUDA (TF32 off) over the 24 kHz-resampled test clip (77 frames) and a 1 s
chirp (25 frames) and dumps every stage (input, 16 kHz semantic input,
HuBERT mean hidden, semantic-encoder output, acoustic latent, fc output,
codes, decoded waveform) to the gitignored `tests/fixtures/higgs_codec.bin`.

```sh
python3 tests/ref/gen_higgs_codec_fixture.py          # rebuild the fixture
cmake --build build --config Release --target brosoundml_test_higgs_codec
./build/tests/Release/brosoundml_test_higgs_codec       # or: ctest -R higgs -C Release
```

Measured (CPU / CUDA, both cases):

| Check | Result |
|---|---|
| decode vs reference waveform | max abs 1.1e-6 / 1.2e-6 (clip), 5.7e-6 / 5.6e-6 (chirp); mean 4e-8 – 5e-7 |
| HuBERT mean hidden (max abs, ref max ~5) | 6.3e-5 / 1.6e-5 (clip), 1.1e-5 / 6.7e-6 (chirp) |
| semantic-encoder / acoustic latent / fc output (ref max 14 – 32) | 2e-5 – 1.7e-4 |
| resample vs torchaudio | max abs 2.4e-7 |
| codes vs reference, all 8 codebooks | 100% (with either resampler input) |
| CPU vs CUDA decode | max abs 1.2e-6 (clip), 4.5e-6 (chirp) |
| CPU vs CUDA codes | identical |
| `decoder_only` decode vs full load | bit-identical |
| round trip on the clip -> Whisper | " Hello there, this is a test of the pipeline." |

The test skips (printing why) when `weights/omnivoice/audio_tokenizer`, the
fixture, the test clip or `weights/whisper` is absent. The CPU pass is slow
(brotensor's CPU conv over 74k-sample, 64–2048-channel activations: ~65 s to
encode and ~26 s to decode the 3 s clip; CUDA does the same in 0.08 s /
0.02 s), so the whole test takes ~4 min with both devices.

## CLI

```
brosoundml_higgs_codec_roundtrip <audio_tokenizer_dir> <in.wav> <out.wav> [--device cpu|cuda] [--levels N]
```

Encodes the WAV (any rate; resampled to 24 kHz mono), prints `T` and a
codebook-0 snippet, decodes with the first `N` of 8 levels (default all) and
writes the 24 kHz reconstruction.
