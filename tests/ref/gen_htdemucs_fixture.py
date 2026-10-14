#!/usr/bin/env python3
"""Regenerate the HTDemucs test fixture (tests/fixtures/htdemucs.bin).

The C++ separator (src/htdemucs*.cpp) is validated against the genuine
reference: demucs 4.0.1's htdemucs_ft through demucs.apply.apply_model, driven
exactly as yue2-research's sep.py drives it (torchaudio resample to 44.1 kHz,
mono-mix mean / std normalisation, split=True, overlap 0.25, shifts 0, then
de-normalise and resample back), in strict FP32 on CUDA (cuDNN TF32 off).

Each case is a synthetic stereo mix with something for every stem: a sung-like
harmonic line with vibrato, a bass line, a kick + hat pattern and a pad chord,
panned apart, plus a little deterministic noise.

  "mix48k_9s"  48 kHz, 9.0 s: resampling, two overlapping segments; all stems
  "mix44k_2s"  44.1 kHz, 2.0 s: no resampling, one padded segment; vocals only

The fixture is gitignored (*.bin); tests/test_htdemucs.cpp skips when it or
the weights (weights/htdemucs_ft, from tools/htdemucs_convert.py) are absent.

Requirements: torch with CUDA, torchaudio, demucs 4.0.1, numpy (the
yue2-research venv has them):

    D:/projects/yue2-research/.venv/Scripts/python tests/ref/gen_htdemucs_fixture.py

Fixture binary layout (little-endian):
    int32   magic = 0x31445448  ('HTD1')
    int32   n_cases
    per case:
      int32   sample_rate
      int32   frames        # per channel
      int32   n_stems
      n_stems x (int32 len, bytes name)
      float32 [2*frames]    # input, planar: x[c*frames + i]
      n_stems x float32 [2*frames]   # the reference stem, planar
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
from demucs.apply import apply_model
from demucs.pretrained import get_model

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "htdemucs.bin"


def synth(sr: int, seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    out = np.zeros((2, n))

    def pan(x, p):  # p in [-1, 1]
        return np.stack([x * np.sqrt((1 - p) / 2), x * np.sqrt((1 + p) / 2)])

    # sung-like line: notes of 0.45 s, vibrato, harmonics with a formant tilt
    notes = 220 * 2 ** (np.array([0, 3, 5, 7, 5, 3, 0, -2, 0, 3, 7, 12, 7, 5, 3, 0, 0, 3, 5, 7]) / 12)
    idx = np.minimum((t / 0.45).astype(int), len(notes) - 1)
    f0 = notes[idx] * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t))
    ph = 2 * np.pi * np.cumsum(f0) / sr
    env = np.clip(np.sin(np.pi * ((t / 0.45) % 1)) * 1.6, 0, 1)
    voice = sum(np.sin(k * ph) * (1 / k) * np.exp(-((k * f0 - 700) / 900) ** 2 * 0.5) for k in range(1, 12))
    out += pan(0.35 * env * voice, 0.1)
    # bass: root notes, 0.9 s each
    broot = 55 * 2 ** (np.array([0, 0, 5, 7, 3, 0, 5, 7, 0, 3, 5]) / 12)
    bf = broot[np.minimum((t / 0.9).astype(int), len(broot) - 1)]
    bph = 2 * np.pi * np.cumsum(bf) / sr
    out += pan(0.3 * (np.sin(bph) + 0.3 * np.sin(2 * bph)), -0.2)
    # drums: kick every 0.5 s, hats every 0.25 s
    tk = t % 0.5
    kick = np.sin(2 * np.pi * (50 * tk + 70 * (1 - np.exp(-tk * 30)) / 30)) * np.exp(-tk * 12)
    th = t % 0.25
    hat = np.diff(rng.standard_normal(n + 1)) * np.exp(-th * 60) * 0.15
    out += pan(0.6 * kick, 0.0) + pan(hat, 0.4)
    # pad: a minor chord, slow tremolo
    pad = sum(np.sign(np.sin(2 * np.pi * f * t)) * 0.04 for f in (261.6, 311.1, 392.0))
    out += pan(pad * (0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t)), -0.5)
    out += 1e-3 * rng.standard_normal(out.shape)
    return (out / np.abs(out).max() * 0.6).astype(np.float32)


def oracle(model, a: np.ndarray, sr: int):
    x = torch.from_numpy(a).cuda()
    msr = model.samplerate
    xi = AF.resample(x, sr, msr) if sr != msr else x
    ref = xi.mean(0)
    mu, sd = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(model, ((xi - mu) / sd)[None], device="cuda", shifts=0, split=True,
                          overlap=0.25, progress=False)[0] * sd + mu
    stems = {}
    for i, name in enumerate(model.sources):
        s = AF.resample(out[i], msr, sr) if sr != msr else out[i]
        stems[name] = s[:, :a.shape[1]].cpu().numpy().astype(np.float32)
    return stems


def main():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = get_model("htdemucs_ft").cuda().eval()
    cases = [(48000, 9.0, 1, ["drums", "bass", "other", "vocals"]),
             (44100, 2.0, 2, ["vocals"])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<ii", 0x31445448, len(cases)))
        for sr, secs, seed, names in cases:
            a = synth(sr, secs, seed)
            stems = oracle(model, a, sr)
            f.write(struct.pack("<iii", sr, a.shape[1], len(names)))
            for nm in names:
                b = nm.encode()
                f.write(struct.pack("<i", len(b)) + b)
            f.write(np.ascontiguousarray(a).tobytes())
            for nm in names:
                f.write(np.ascontiguousarray(stems[nm]).tobytes())
                print(f"{sr} Hz {secs} s {nm}: rms {np.sqrt((stems[nm] ** 2).mean()):.3e}")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
