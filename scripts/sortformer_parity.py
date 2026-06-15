#!/usr/bin/env python3
"""Parity check: brosoundml::Sortformer (C++) vs the reference NeMo model.

Generates a deterministic 16 kHz mono WAV, runs the offline diarization forward
both ways, and reports the max / mean absolute difference of the (T x 4) speaker
activity matrices.

Reference path (NeMo, the model's non-streaming forward_infer):
    mel = preprocessor(audio)            # dither + pad disabled for determinism
    emb = frontend_encoder(mel)          # FastConformer + encoder_proj
    preds = forward_infer(emb)           # transformer head + sigmoids

C++ path: brosoundml_sortformer_diarize --probs-out (Sortformer::diarize).

Run in the NeMo venv:
    .venv-nemo/Scripts/python.exe scripts/sortformer_parity.py \
        --model weights/sortformer/4spk-v2.1 \
        --exe build/Release/brosoundml_sortformer_diarize.exe [--device cuda]
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


def gen_wav(path: Path, seconds: float = 8.0, sr: int = 16000) -> None:
    """A deterministic multi-tone + noise signal — content is irrelevant, only
    that both paths see identical samples."""
    rng = np.random.default_rng(1234)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float64)
    # Three "speakers": tone bursts in disjoint-ish time windows + light noise.
    for k, (f0, a, b) in enumerate([(180.0, 0.05, 0.40), (260.0, 0.30, 0.70),
                                    (320.0, 0.55, 0.85)]):
        env = ((t >= a * seconds) & (t < b * seconds)).astype(np.float64)
        sig += 0.3 * env * np.sin(2 * np.pi * f0 * t + 0.5 * k)
        sig += 0.1 * env * np.sin(2 * np.pi * (2 * f0) * t)
    sig += 0.01 * rng.standard_normal(n)
    sig /= max(1e-9, np.abs(sig).max())
    sig *= 0.9
    pcm = (sig * 32767.0).astype("<i2")
    with open(path, "wb") as f:
        data = pcm.tobytes()
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def reference_preds(nemo_path: str, wav: Path) -> np.ndarray:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(nemo_path, map_location="cpu")
    model.eval()
    # Determinism: drop dither noise and frame padding so the reference matches a
    # plain forward.
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    # Load the same int16 PCM the C++ reads.
    import wave
    with wave.open(str(wav), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    audio = torch.from_numpy(
        np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    ).unsqueeze(0)
    length = torch.tensor([audio.shape[1]], dtype=torch.long)

    with torch.no_grad():
        processed, plen = model.preprocessor(input_signal=audio, length=length)
        emb, emb_len = model.frontend_encoder(
            processed_signal=processed, processed_signal_length=plen
        )
        preds = model.forward_infer(emb, emb_len)  # (1, T, n_spk)
    return preds.squeeze(0).cpu().numpy()


def cpp_preds(exe: str, wav: Path, model_dir: str, device: str) -> np.ndarray:
    out = wav.with_suffix(".cpp_probs.bin")
    cmd = [exe, str(wav), model_dir, "--device", device, "--probs-out", str(out)]
    subprocess.run(cmd, check=True)
    with open(out, "rb") as f:
        T, S = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(), dtype="<f4")
    return data.reshape(T, S)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(repo / "weights" / "sortformer" / "4spk-v2.1"))
    ap.add_argument("--exe", default=str(repo / "build" / "Release" /
                                        "brosoundml_sortformer_diarize.exe"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()

    wav = Path(args.model) / "parity_input.wav"
    gen_wav(wav, args.seconds)
    print(f"wrote {wav} ({args.seconds}s)")

    nemo_files = sorted(glob.glob(str(Path(args.model) / "*.nemo")))
    if not nemo_files:
        print("error: no .nemo under --model", file=sys.stderr)
        return 2

    print("running NeMo reference ...")
    ref = reference_preds(nemo_files[0], wav)
    print(f"  reference preds: {ref.shape}")

    print(f"running C++ ({args.device}) ...")
    got = cpp_preds(args.exe, wav, args.model, args.device)
    print(f"  C++ preds:       {got.shape}")

    T = min(ref.shape[0], got.shape[0])
    if ref.shape[0] != got.shape[0]:
        print(f"WARNING: frame-count mismatch ref {ref.shape[0]} vs cpp {got.shape[0]}; "
              f"comparing first {T} frames")
    ref, got = ref[:T], got[:T]

    # NeMo's center-STFT yields one trailing pad diarization frame
    # (valid = floor(samples/hop) subsampled 8x). NeMo masks it; its value is a
    # meaningless boundary artifact, so parity is judged on the valid frames.
    def conv_len(n):
        return (n - 1) // 2 + 1
    valid = conv_len(conv_len(conv_len(int(args.seconds * 16000) // 160)))
    valid = min(valid, T)

    diff = np.abs(ref - got)
    vdiff = diff[:valid]
    print(f"\nvalid frames: {valid} of {T}")
    print(f"max  abs diff (valid): {vdiff.max():.6e}")
    print(f"mean abs diff (valid): {vdiff.mean():.6e}")
    if valid < T:
        pdiff = diff[valid:]
        print(f"trailing pad frames ({T - valid}): max {pdiff.max():.6e} "
              f"(masked boundary artifact, excluded from parity)")
    worst = np.argsort(vdiff.max(axis=1))[-3:][::-1]
    for t in worst:
        print(f"  frame {t:4d}  ref={np.round(ref[t],4)}  cpp={np.round(got[t],4)}")

    ok = vdiff.max() < args.tol
    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
