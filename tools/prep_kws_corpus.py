#!/usr/bin/env python3
"""prep_kws_corpus.py — decode a transcribed speech corpus into the aligner's
manifest form: 16 kHz mono PCM16 WAVs + a TSV manifest (wav<TAB>text<TAB>tag).

phoneme_align --manifest consumes the output directly:
  brosoundml_phoneme_align --manifest <out>/manifest.tsv --out shard.bpds

Subcommands:
  librispeech --src <LibriSpeech root containing train-clean-100/...> \
              --subsets train-clean-100,train-clean-360 --out <dir>
  vctk        --src <VCTK-Corpus-0.92 root> --out <dir>

Selection is capped per speaker (--cap-per-speaker) with a seeded shuffle so
reruns are deterministic; durations outside [--min-dur, --max-dur] are skipped.
The tag column is the speaker id, so the aligner's per-tag cap also buckets by
speaker. Requires: soundfile, scipy (pip install soundfile scipy).
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def write_wav16k(dst: Path, audio: np.ndarray, sr: int) -> float:
    """Downmix/resample to 16 kHz mono PCM16, write WAV; returns duration s."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        g = np.gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g)
    audio = np.clip(audio, -1.0, 1.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), (audio * 32767.0).astype(np.int16), TARGET_SR,
             subtype="PCM_16")
    return len(audio) / TARGET_SR


def emit(rows, out_dir: Path, manifest_name="manifest.tsv"):
    man = out_dir / manifest_name
    with open(man, "w", encoding="utf-8", newline="\n") as f:
        for rel, text, tag in rows:
            f.write(f"{rel}\t{text}\t{tag}\n")
    print(f"wrote {len(rows)} entries -> {man}")


def iter_capped(per_speaker: dict, cap: int, seed: int):
    rng = random.Random(seed)
    for spk in sorted(per_speaker):
        utts = per_speaker[spk]
        rng.shuffle(utts)
        yield from utts[: cap if cap > 0 else len(utts)]


def cmd_librispeech(args):
    src = Path(args.src)
    out = Path(args.out)
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    per_speaker = {}
    for sub in subsets:
        root = src / sub
        if not root.is_dir():
            sys.exit(f"missing subset dir: {root}")
        for trans in root.rglob("*.trans.txt"):
            chap_dir = trans.parent
            spk = chap_dir.parent.name
            for line in trans.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                utt_id, _, text = line.partition(" ")
                flac = chap_dir / f"{utt_id}.flac"
                if not text or not flac.exists():
                    continue
                per_speaker.setdefault(spk, []).append((flac, text.lower(), spk))
    convert(args, out, per_speaker, tag_prefix="ls")


def cmd_vctk(args):
    src = Path(args.src)
    out = Path(args.out)
    wav_root = src / "wav48_silence_trimmed"
    txt_root = src / "txt"
    if not wav_root.is_dir() or not txt_root.is_dir():
        sys.exit(f"missing wav48_silence_trimmed/ or txt/ under {src}")
    per_speaker = {}
    for txt in txt_root.rglob("*.txt"):
        utt = txt.stem                      # e.g. p225_001
        spk = txt.parent.name
        flac = wav_root / spk / f"{utt}_mic1.flac"
        if not flac.exists():
            flac = wav_root / spk / f"{utt}_mic2.flac"
        if not flac.exists():
            continue
        text = txt.read_text(encoding="utf-8").strip()
        if not text:
            continue
        per_speaker.setdefault(spk, []).append((flac, text, spk))
    convert(args, out, per_speaker, tag_prefix="vctk")


def convert(args, out: Path, per_speaker: dict, tag_prefix: str):
    rows, skipped, total_dur = [], 0, 0.0
    n_max = args.max_utts if args.max_utts > 0 else None
    for flac, text, spk in iter_capped(per_speaker, args.cap_per_speaker,
                                       args.seed):
        if n_max is not None and len(rows) >= n_max:
            break
        try:
            audio, sr = sf.read(str(flac), dtype="float64")
        except Exception:
            skipped += 1
            continue
        dur = (audio.shape[0] / sr) if sr else 0.0
        if dur < args.min_dur or dur > args.max_dur:
            skipped += 1
            continue
        rel = Path("wav") / spk / (flac.stem + ".wav")
        total_dur += write_wav16k(out / rel, audio, sr)
        rows.append((rel.as_posix(), text, f"{tag_prefix}-{spk}"))
        if len(rows) % 2000 == 0:
            print(f"  {len(rows)} converted ({total_dur/3600.0:.1f} h) ...")
    emit(rows, out)
    print(f"done: {len(rows)} utts, {total_dur/3600.0:.2f} h, skipped {skipped}, "
          f"{len(per_speaker)} speakers")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("librispeech", cmd_librispeech), ("vctk", cmd_vctk)):
        p = sub.add_parser(name)
        p.add_argument("--src", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--cap-per-speaker", type=int, default=100)
        p.add_argument("--max-utts", type=int, default=0, help="0 = no cap")
        p.add_argument("--min-dur", type=float, default=1.0)
        p.add_argument("--max-dur", type=float, default=20.0)
        p.add_argument("--seed", type=int, default=1234)
        if name == "librispeech":
            p.add_argument("--subsets",
                           default="train-clean-100,train-clean-360")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
