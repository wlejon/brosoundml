#!/usr/bin/env python3
"""Streaming parity: brosoundml::Sortformer feed() (AOSC) vs NeMo forward_streaming.

Generates a long (multi-chunk) deterministic WAV so the Arrival-Order Speaker
Cache actually compresses, runs both the C++ streaming path and the reference
NeMo streaming forward, and reports the max/mean abs diff of the (T x 4) matrices.

Run in the NeMo venv:
    .venv-nemo/Scripts/python.exe scripts/sortformer_parity_streaming.py --device cuda
"""

from __future__ import annotations

import argparse
import glob
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def gen_wav(path: Path, seconds: float, sr: int = 16000) -> None:
    rng = np.random.default_rng(7)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float64)
    # Several speakers turning on/off across the clip so the cache fills.
    spans = [(150.0, 0.00, 0.30), (240.0, 0.20, 0.55),
             (330.0, 0.45, 0.75), (200.0, 0.65, 0.95)]
    for k, (f0, a, b) in enumerate(spans):
        env = ((t >= a * seconds) & (t < b * seconds)).astype(np.float64)
        sig += 0.3 * env * np.sin(2 * np.pi * f0 * t + 0.4 * k)
        sig += 0.08 * env * np.sin(2 * np.pi * (2 * f0) * t)
    sig += 0.01 * rng.standard_normal(n)
    sig /= max(1e-9, np.abs(sig).max())
    sig *= 0.9
    pcm = (sig * 32767.0).astype("<i2")
    with open(path, "wb") as f:
        data = pcm.tobytes()
        f.write(b"RIFF"); f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVEfmt "); f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data"); f.write(struct.pack("<I", len(data))); f.write(data)


def reference(nemo_path: str, wav: Path) -> np.ndarray:
    from nemo.collections.asr.models import SortformerEncLabelModel
    import wave
    model = SortformerEncLabelModel.restore_from(nemo_path, map_location="cpu")
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    model.streaming_mode = True
    model.sortformer_modules.log = False
    with wave.open(str(wav), "rb") as w:
        raw = w.readframes(w.getnframes())
    audio = torch.from_numpy(
        np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).unsqueeze(0)
    length = torch.tensor([audio.shape[1]], dtype=torch.long)
    with torch.no_grad():
        preds = model.forward(audio_signal=audio, audio_signal_length=length)
    return preds.squeeze(0).cpu().numpy()


def cpp(exe: str, wav: Path, model_dir: str, device: str) -> np.ndarray:
    out = wav.with_suffix(".cpp_stream.bin")
    subprocess.run([exe, str(wav), model_dir, "--device", device,
                    "--streaming", "--probs-out", str(out)], check=True)
    with open(out, "rb") as f:
        T, S = struct.unpack("<ii", f.read(8))
        return np.frombuffer(f.read(), dtype="<f4").reshape(T, S)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(repo / "weights" / "sortformer" / "4spk-v2.1"))
    ap.add_argument("--exe", default=str(repo / "build" / "Release" /
                                        "brosoundml_sortformer_diarize.exe"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--tol", type=float, default=2e-2)
    args = ap.parse_args()

    wav = Path(args.model) / "parity_stream.wav"
    gen_wav(wav, args.seconds)
    print(f"wrote {wav} ({args.seconds}s, ~{args.seconds/15.04:.1f} chunks)")

    nemo = sorted(glob.glob(str(Path(args.model) / "*.nemo")))[0]
    print("running NeMo streaming reference ...")
    ref = reference(nemo, wav)
    print(f"  ref {ref.shape}")
    print(f"running C++ streaming ({args.device}) ...")
    got = cpp(args.exe, wav, args.model, args.device)
    print(f"  cpp {got.shape}")

    T = min(ref.shape[0], got.shape[0])
    if ref.shape[0] != got.shape[0]:
        print(f"WARNING: length mismatch ref {ref.shape[0]} cpp {got.shape[0]}")
    ref, got = ref[:T], got[:T]
    diff = np.abs(ref - got)
    print(f"\nmax  abs diff: {diff.max():.6e}")
    print(f"mean abs diff: {diff.mean():.6e}")
    for t in np.argsort(diff.max(axis=1))[-4:][::-1]:
        print(f"  frame {t:4d}  ref={np.round(ref[t],4)}  cpp={np.round(got[t],4)}")
    ok = diff.max() < args.tol
    print("\nSTREAMING PARITY OK" if ok else "\nSTREAMING PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
