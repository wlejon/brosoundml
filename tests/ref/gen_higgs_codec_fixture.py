#!/usr/bin/env python3
"""Regenerate the HiggsAudio v2 codec test fixture (tests/fixtures/higgs_codec.bin).

The C++ codec (src/higgs_codec.cpp + src/hubert.cpp) is validated against the
genuine reference — transformers' `HiggsAudioV2TokenizerModel` run in float32
on CUDA over the OmniVoice audio_tokenizer weights. Two cases are dumped:

  * "clip"  — weights/qwen-tts-hello-there-this-is-a-test-of-th.wav (16 kHz on
              disk, resampled to 24 kHz here with torchaudio's windowed sinc),
  * "chirp" — a synthetic 1 s 24 kHz chirp (100 Hz -> 4 kHz, plus a little
              deterministic noise), the small fast case.

Every case's input is right-padded to a whole number of 960-sample frames, so
the reference's acoustic and semantic branches agree on T with no extra
padding — the same convention encode() uses in C++ — and every intermediate
stage is dumped so a mismatch localises:

  input (24 kHz) -> sem16 (torchaudio resample 24 -> 16 kHz, the reference's
  semantic input) -> hubert mean of the 13 hidden states (pre-downsample,
  Th = 2T frames) -> semantic-encoder output (768 x T) -> acoustic latent
  (256 x T) -> fc output (1024 x T, the RVQ input) -> codes (8 x T) ->
  decoded waveform (T*960 samples, decoded from those codes).

The fixture is gitignored (*.bin); tests/test_higgs_codec.cpp skips its
numeric checks when it is absent.

Requirements: torch (CUDA), torchaudio, transformers >= 5.x (with
HiggsAudioV2TokenizerModel), safetensors, numpy; the weights under
weights/omnivoice/audio_tokenizer/.

    python3 tests/ref/gen_higgs_codec_fixture.py

Fixture binary layout (little-endian):
    int32   magic = 0x31434748  ('HGC1')
    int32   n_cases
    per case:
      int32   n24        # input samples at 24 kHz (a multiple of 960)
      int32   n16        # reference 16 kHz semantic-input samples
      int32   Th         # HuBERT frames before the /2 downsample (== 2*T)
      int32   T          # codec frames (== n24 / 960)
      int32   K          # RVQ levels (8)
      int32   n_out      # decoded samples (== T * 960)
      float32 [n24]      # input samples
      float32 [n16]      # sem16: the 16 kHz semantic input (after resample)
      float32 [Th*768]   # hubert mean hidden, frame-major: h[t*768 + c]
      float32 [768*T]    # semantic-encoder output, channel-major: s[c*T + t]
      float32 [256*T]    # acoustic latent, channel-major: a[c*T + t]
      float32 [1024*T]   # fc output (RVQ input), channel-major: e[c*T + t]
      int32   [K*T]      # codes, codebook-major: code[k*T + t]
      float32 [n_out]    # decoded waveform (no clamp)
"""
import os
import struct
import sys
import wave

import numpy as np

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEIGHTS = os.path.join(REPO, "weights", "omnivoice", "audio_tokenizer")
CLIP = os.path.join(REPO, "weights", "qwen-tts-hello-there-this-is-a-test-of-th.wav")
OUT = os.path.join(REPO, "tests", "fixtures", "higgs_codec.bin")

MAGIC = 0x31434748


def read_wav_mono(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, "16-bit PCM expected"
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm, sr


def main():
    if not os.path.exists(os.path.join(WEIGHTS, "model.safetensors")):
        sys.exit(f"missing codec weights under {WEIGHTS}")

    import torch
    import torchaudio
    from transformers import HiggsAudioV2TokenizerModel

    if not torch.cuda.is_available():
        sys.exit("CUDA is required (the reference runs on the GPU)")
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True

    model = HiggsAudioV2TokenizerModel.from_pretrained(WEIGHTS, dtype=torch.float32).to(dev).eval()
    cfg = model.config
    hop = cfg.hop_length
    K = cfg.num_quantizers
    assert hop == 960 and K == 8, (hop, K)

    # ── inputs ──
    clip, sr = read_wav_mono(CLIP)
    clip_t = torch.from_numpy(clip)[None, :]
    if sr != cfg.sample_rate:
        clip_t = torchaudio.functional.resample(clip_t, sr, cfg.sample_rate)
    clip24 = clip_t[0].numpy()

    n_chirp = cfg.sample_rate  # 1 s
    t = np.arange(n_chirp, dtype=np.float64) / cfg.sample_rate
    f0, f1 = 100.0, 4000.0
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / 2.0)
    rng = np.random.default_rng(1234)
    chirp = 0.5 * np.sin(phase) + 0.02 * rng.standard_normal(n_chirp)
    chirp = chirp.astype(np.float32)

    cases = [("clip", clip24), ("chirp", chirp)]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<ii", MAGIC, len(cases)))
        for name, wav in cases:
            n = len(wav)
            n24 = ((n + hop - 1) // hop) * hop
            x = np.zeros(n24, dtype=np.float32)
            x[:n] = wav
            T = n24 // hop
            xt = torch.from_numpy(x).to(dev)[None, None, :]  # (1,1,n24)

            with torch.no_grad():
                # semantic branch, staged exactly as _extract_semantic_features
                sem16 = torchaudio.functional.resample(xt, cfg.sample_rate, cfg.semantic_sample_rate)
                n16 = sem16.shape[-1]
                padded = torch.nn.functional.pad(sem16[:, 0, :], (160, 160))
                hs = model.semantic_model(padded, output_hidden_states=True).hidden_states
                assert len(hs) == 13, len(hs)
                hub = torch.stack(list(hs), dim=1).mean(dim=1)  # (1, Th, 768)
                Th = hub.shape[1]
                sem_feat = hub[:, :: cfg.semantic_downsample_factor, :]
                ref_feat = model._extract_semantic_features(xt)
                assert torch.equal(sem_feat, ref_feat)
                e_sem = model.encoder_semantic(sem_feat.transpose(1, 2))  # (1,768,T)
                # acoustic branch — no extra padding needed for a whole frame count
                e_ac = model.acoustic_encoder(xt)  # (1,256,T)
                assert e_ac.shape[2] == T and e_sem.shape[2] == T, (e_ac.shape, e_sem.shape, T)
                assert Th == 2 * T, (Th, T)
                emb = torch.cat([e_ac, e_sem], dim=1)
                emb = model.fc(emb.transpose(1, 2)).transpose(1, 2)  # (1,1024,T)
                codes = model.quantizer.encode(emb).transpose(0, 1)  # (1,K,T)
                ref_codes = model.encode(xt, return_dict=False)
                assert torch.equal(codes, ref_codes)
                dec = model.decode(codes, return_dict=False)  # (1,1,n_out)
                n_out = dec.shape[-1]
                assert n_out == T * hop, (n_out, T)

            def w_f32(a):
                f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())

            f.write(struct.pack("<iiiiii", n24, n16, Th, T, K, n_out))
            w_f32(x)
            w_f32(sem16[0, 0].cpu().numpy())
            w_f32(hub[0].cpu().numpy())                 # (Th, 768) frame-major
            w_f32(e_sem[0].cpu().numpy())               # (768, T) channel-major
            w_f32(e_ac[0].cpu().numpy())                # (256, T)
            w_f32(emb[0].cpu().numpy())                 # (1024, T)
            f.write(np.ascontiguousarray(codes[0].cpu().numpy(), dtype="<i4").tobytes())
            w_f32(dec[0, 0].cpu().numpy())
            c0 = codes[0, 0, :8].tolist()
            print(f"  {name}: n24={n24} n16={n16} Th={Th} T={T} n_out={n_out} codes[0,:8]={c0}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
