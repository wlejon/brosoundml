#!/usr/bin/env python3
"""Run the reference NeMo Sortformer (offline forward_infer) on an arbitrary WAV
and print per-speaker activity, plus optionally compare to a C++ --probs-out dump.

Run in the NeMo venv:
  .venv-nemo/Scripts/python.exe scripts/sortformer_nemo_ref.py \
      --wav weights/sortformer/4spk-v2.1/twospk.wav [--cpp twospk.cpp_probs.bin]
"""
from __future__ import annotations

import argparse
import glob
import struct
import wave
from pathlib import Path

import numpy as np
import torch


def load_audio(wav: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with wave.open(str(wav), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, "need 16k mono"
        raw = w.readframes(w.getnframes())
    audio = torch.from_numpy(
        np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).unsqueeze(0)
    length = torch.tensor([audio.shape[1]], dtype=torch.long)
    return audio, length


def nemo_preds(nemo_path: str, wav: Path) -> np.ndarray:
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(nemo_path, map_location="cpu")
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    audio, length = load_audio(wav)
    with torch.no_grad():
        processed, plen = model.preprocessor(input_signal=audio, length=length)
        emb, emb_len = model.frontend_encoder(
            processed_signal=processed, processed_signal_length=plen)
        preds = model.forward_infer(emb, emb_len)
    return preds.squeeze(0).cpu().numpy()


def summarize(name: str, preds: np.ndarray, thr: float = 0.5) -> None:
    T, S = preds.shape
    fs = 0.08
    print(f"\n== {name}: {T} frames x {S} spk ==")
    print(f"  per-speaker: max | mean | #frames>thr")
    for s in range(S):
        col = preds[:, s]
        print(f"   spk{s}:  {col.max():.3f} | {col.mean():.3f} | "
              f"{int((col > thr).sum())}")
    # active spans per speaker
    for s in range(S):
        col = preds[:, s] > thr
        spans = []
        run = -1
        for t in range(T + 1):
            on = t < T and col[t]
            if on and run < 0:
                run = t
            elif not on and run >= 0:
                spans.append((run * fs, t * fs))
                run = -1
        if spans:
            sp = ", ".join(f"[{a:.2f},{b:.2f})" for a, b in spans)
            print(f"   spk{s} spans: {sp}")


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(repo / "weights" / "sortformer" / "4spk-v2.1"))
    ap.add_argument("--wav", required=True)
    ap.add_argument("--cpp", default="")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    wav = Path(args.wav)
    nemo = sorted(glob.glob(str(Path(args.model) / "*.nemo")))[0]
    print(f"NeMo: {nemo}\nwav:  {wav}")
    ref = nemo_preds(nemo, wav)
    summarize("NeMo forward_infer", ref, args.thr)

    if args.cpp:
        with open(args.cpp, "rb") as f:
            T, S = struct.unpack("<ii", f.read(8))
            got = np.frombuffer(f.read(), dtype="<f4").reshape(T, S)
        summarize("C++ diarize", got, args.thr)
        n = min(ref.shape[0], got.shape[0])
        diff = np.abs(ref[:n] - got[:n])
        print(f"\nmax abs diff (NeMo vs C++): {diff.max():.6e}  "
              f"mean {diff.mean():.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
