#!/usr/bin/env python3
"""Build a real two-speaker clip from two distinct Kokoro voices and report the
ground-truth turn spans, so we can check whether Sortformer separates two
clearly-different (here: female) voices at all.

Concatenates alternating turns A,B,A,B,... from two voices' synth_*.wav, each
resampled to 16 kHz mono, with a short silence between turns. Writes a 16 kHz
mono WAV and prints the [start,end) seconds of each turn + which speaker.

Then run the offline diarizer on it:
  build/Release/brosoundml_sortformer_diarize.exe <out.wav> weights/sortformer/4spk-v2.1 \
      --device cuda --threshold 0.5

Plain stdlib (wave + struct) so it runs under any python.
"""
from __future__ import annotations

import argparse
import struct
import wave
from pathlib import Path


def read_wav_mono16k(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise SystemExit(f"{path}: expected 16-bit PCM, got sampwidth {sw}")
    import array
    a = array.array("h")
    a.frombytes(raw)
    # downmix
    if ch > 1:
        mono = [sum(a[i * ch + c] for c in range(ch)) / ch for i in range(n)]
    else:
        mono = [float(x) for x in a]
    # to [-1,1]
    mono = [x / 32768.0 for x in mono]
    # linear resample to 16k
    if sr != 16000:
        ratio = 16000 / sr
        m = int(len(mono) * ratio)
        out = [0.0] * m
        for i in range(m):
            t = i / ratio
            j = int(t)
            f = t - j
            a0 = mono[j] if j < len(mono) else 0.0
            a1 = mono[j + 1] if j + 1 < len(mono) else a0
            out[i] = a0 * (1 - f) + a1 * f
        mono = out
    return mono


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-a", default=str(repo / "weights/kokoro/out/af_bella"))
    ap.add_argument("--voice-b", default=str(repo / "weights/kokoro/out/bf_emma"))
    ap.add_argument("--voice-c", default="", help="optional third voice dir")
    ap.add_argument("--turns", type=int, default=3, help="turns per speaker")
    ap.add_argument("--gap", type=float, default=0.4, help="silence between turns (s)")
    ap.add_argument("--out", default=str(repo / "weights/sortformer/4spk-v2.1/twospk.wav"))
    args = ap.parse_args()

    va = Path(args.voice_a)
    vb = Path(args.voice_b)
    a_clips = sorted(va.glob("synth_*.wav"))
    b_clips = sorted(vb.glob("synth_*.wav"))
    if not a_clips or not b_clips:
        raise SystemExit(f"no synth_*.wav under {va} or {vb}")

    gap = [0.0] * int(args.gap * 16000)
    sig: list[float] = []
    voices = [("A:" + va.name, a_clips), ("B:" + vb.name, b_clips)]
    if args.voice_c:
        vc = Path(args.voice_c)
        c_clips = sorted(vc.glob("synth_*.wav"))
        if c_clips:
            voices.append(("C:" + vc.name, c_clips))

    spans = []
    for k in range(args.turns):
        for label, clips in voices:
            clip = clips[k % len(clips)]
            pcm = read_wav_mono16k(clip)
            start = len(sig) / 16000
            sig.extend(pcm)
            end = len(sig) / 16000
            spans.append((label, start, end, clip.name))
            sig.extend(gap)

    # write 16k mono int16
    out = Path(args.out)
    pcm16 = b"".join(struct.pack("<h", max(-32768, min(32767, int(x * 32767))))
                     for x in sig)
    with open(out, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(pcm16)))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, 16000, 16000 * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(pcm16)))
        f.write(pcm16)

    print(f"wrote {out}  ({len(sig)/16000:.2f}s, 16 kHz mono)")
    print("ground-truth turns (speaker A = voice-a, B = voice-b):")
    for label, s, e, name in spans:
        print(f"  [{s:6.2f}, {e:6.2f})  {label:18s}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
