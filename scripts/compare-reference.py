#!/usr/bin/env python3
"""Compare brosoundml's C++ Kokoro against the upstream PyTorch reference.

For each test sentence (and optionally each comma-split chunk of it):
  1. Phonemize the text into Kokoro's IPA vocab.
  2. Synthesize with the Python reference (`kokoro.model.KModel`) -> py_*.wav.
  3. Synthesize with `brosoundml_synth.exe` over the same phoneme ids
     -> cpp_*.wav.
  4. Compute alignment-tolerant similarity metrics (length, RMS, peak
     cross-correlation, log-mel L1) and print a report.

Both backends consume the *same* phoneme id sequence -- the only divergence
left is the model arithmetic itself. If a sentence sounds different the
metrics tell you whether it's a length / duration miss (pred_dur drift), a
gain mismatch (RMS), or a timbre / spectral divergence (mel L1).

Usage:
  scripts/compare-reference.py                      # built-in sentences
  scripts/compare-reference.py --text "Hello there. The quick brown fox."
  scripts/compare-reference.py --voice af_bella --chunks word

Outputs:
  weights/kokoro/compare/<voice>/<idx>_<tag>_{py,cpp}.wav
  + a printed table.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

# Suppress kokoro/__init__.py's KPipeline import (pulls misaki/spacy).
sys.modules.setdefault('kokoro.pipeline', types.ModuleType('kokoro.pipeline'))
sys.modules['kokoro.pipeline'].KPipeline = None  # type: ignore[attr-defined]

import torch  # noqa: E402
import eng_to_ipa as ipa  # noqa: E402
from kokoro.model import KModel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS   = REPO_ROOT / "weights" / "kokoro"
SR        = 24000

# Same map as scripts/phonemize.py -- keep in sync.
IPA_SUBSTITUTIONS = {"g": "ɡ", "r": "ɹ", "*": ""}

DEFAULT_SENTENCES = [
    "Hello there.",
    "The quick brown fox jumps over the lazy dog.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Antidisestablishmentarianism is a remarkably long word.",
]


def normalise(text: str) -> str:
    return "".join(IPA_SUBSTITUTIONS.get(c, c) for c in text)


def phonemize(text: str, vocab: dict[str, int]) -> tuple[str, list[int]]:
    """Return (phoneme_string, id_list) matching what KModel expects."""
    raw = normalise(ipa.convert(text))
    phonemes = "".join(c for c in raw if c in vocab)
    ids = [vocab[c] for c in phonemes]
    return phonemes, ids


def split_chunks(text: str, mode: str) -> list[str]:
    if mode == "sentence":
        return [text]
    if mode == "comma":
        parts = [p.strip() for p in re.split(r"[,;]", text) if p.strip()]
        return parts or [text]
    if mode == "word":
        return [w for w in re.findall(r"[A-Za-z']+", text) if w]
    raise ValueError(f"unknown --chunks mode {mode!r}")


# ─── Python reference synthesis ───────────────────────────────────────────────

def synth_py(model: KModel, voice_pack: torch.Tensor, phonemes: str) -> np.ndarray:
    """Mirror KModel.forward with no diffusion -- deterministic single pass."""
    device = model.device
    vocab = model.vocab
    input_ids = [0] + [vocab[c] for c in phonemes if c in vocab] + [0]
    L = len(input_ids)
    input_ids_t = torch.LongTensor([input_ids]).to(device)
    input_lengths = torch.LongTensor([L]).to(device)
    text_mask = torch.arange(L, device=device).unsqueeze(0).expand(1, -1)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
    ref_s = voice_pack[L - 1].to(device)
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)

    with torch.no_grad():
        bert_dur = model.bert(input_ids_t, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        lstm_x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(lstm_x)
        pred_dur = torch.round(torch.sigmoid(duration).sum(axis=-1)
                              ).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(
            torch.arange(L, device=device), pred_dur)
        aln = torch.zeros((L, indices.shape[0]), device=device)
        aln[indices, torch.arange(indices.shape[0], device=device)] = 1
        aln = aln.unsqueeze(0)
        en = d.transpose(-1, -2) @ aln
        F0, N = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(input_ids_t, input_lengths, text_mask)
        asr = t_en @ aln
        torch.manual_seed(0)
        audio = model.decoder(asr, F0, N, ref_s[:, :128]).squeeze().cpu().numpy()
    return audio.astype(np.float32)


# ─── C++ synthesis via brosoundml_synth.exe ──────────────────────────────────

def find_synth_exe() -> Path:
    candidates = [
        REPO_ROOT / "build" / "Release" / "brosoundml_synth.exe",
        REPO_ROOT / "build" / "brosoundml_synth.exe",
        REPO_ROOT / "build" / "Release" / "brosoundml_synth",
        REPO_ROOT / "build" / "brosoundml_synth",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "brosoundml_synth not found -- build with "
        "`cmake --build build --config Release` first")


def synth_cpp(exe: Path, voice_bin: Path, ids: list[int], out_wav: Path) -> np.ndarray:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe),
           "--model", str(WEIGHTS),
           "--voice", str(voice_bin),
           "--out",   str(out_wav),
           ",".join(str(i) for i in ids)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"brosoundml_synth failed:\n{r.stderr}")
    return read_wav(out_wav)


# ─── WAV I/O ──────────────────────────────────────────────────────────────────

def read_wav(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise RuntimeError(f"unsupported wav sample width {sw}")
    return a


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    a16 = np.clip(audio, -1.0, 1.0)
    a16 = (a16 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(a16.tobytes())


# ─── Comparison metrics ───────────────────────────────────────────────────────

def mel_filterbank(n_fft: int, n_mels: int, sr: int) -> np.ndarray:
    f_min, f_max = 0.0, sr / 2
    def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l or r == c: continue
        fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def log_mel(a: np.ndarray, n_fft=1024, hop=256, n_mels=64) -> np.ndarray:
    if len(a) < n_fft:
        a = np.pad(a, (0, n_fft - len(a)))
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(a, n_fft)[::hop]
    spec = np.fft.rfft(frames * win, axis=-1)
    mag = np.abs(spec).astype(np.float32)
    fb = mel_filterbank(n_fft, n_mels, SR)
    mel = mag @ fb.T
    return np.log(mel + 1e-6)


def best_align_xcorr(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Peak normalized cross-correlation (alignment-tolerant) and lag."""
    n = min(len(a), len(b))
    if n < 64: return 0.0, 0
    a = a[:n].copy(); b = b[:n].copy()
    a -= a.mean(); b -= b.mean()
    da = np.sqrt((a * a).sum()); db = np.sqrt((b * b).sum())
    if da == 0 or db == 0: return 0.0, 0
    # FFT-based xcorr
    N = 1 << int(np.ceil(np.log2(2 * n)))
    fa = np.fft.rfft(a, N)
    fb_ = np.fft.rfft(b, N)
    xc = np.fft.irfft(fa * np.conj(fb_), N)
    xc = np.concatenate([xc[-n + 1:], xc[:n]])
    peak = int(np.argmax(xc))
    return float(xc[peak] / (da * db)), peak - (n - 1)


def compare(py: np.ndarray, cpp: np.ndarray) -> dict:
    rms = lambda x: float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    xcorr, lag = best_align_xcorr(py, cpp)
    mp = log_mel(py); mc = log_mel(cpp)
    T = min(mp.shape[0], mc.shape[0])
    mel_l1 = float(np.mean(np.abs(mp[:T] - mc[:T]))) if T > 0 else float("nan")
    return {
        "py_n":     len(py),
        "cpp_n":    len(cpp),
        "len_ratio": len(cpp) / max(len(py), 1),
        "py_rms":   rms(py),
        "cpp_rms":  rms(cpp),
        "xcorr":    xcorr,
        "lag":      lag,
        "mel_l1":   mel_l1,
    }


def fmt_row(tag: str, m: dict) -> str:
    return (f"{tag:<40s}  "
            f"py={m['py_n']:>6d}  cpp={m['cpp_n']:>6d}  "
            f"len%={m['len_ratio']:5.2f}  "
            f"rms py={m['py_rms']:.3f}/cpp={m['cpp_rms']:.3f}  "
            f"xcorr={m['xcorr']:+.3f}  "
            f"lag={m['lag']:>+5d}  "
            f"mel_L1={m['mel_l1']:.3f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--voice", default="af_heart",
                    help="Voice name under weights/kokoro/voices/")
    ap.add_argument("--text",  action="append", default=None,
                    help="Sentence to compare (repeatable). Default: built-in set.")
    ap.add_argument("--chunks", choices=["sentence", "comma", "word"],
                    default="sentence",
                    help="How to split each sentence into chunks for comparison")
    ap.add_argument("--out-dir", type=Path,
                    default=WEIGHTS / "compare",
                    help="Where to drop {py,cpp}.wav pairs")
    ap.add_argument("--device", default="auto",
                    help="torch device for the Python reference: auto|cpu|cuda|cuda:N")
    args = ap.parse_args()

    # Unbuffered stdout so progress shows up in piped / redirected runs.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    sentences = args.text if args.text else DEFAULT_SENTENCES

    cfg = json.loads((WEIGHTS / "config.json").read_text(encoding="utf-8"))
    vocab = cfg["vocab"]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Loading KModel on {device} ...")
    model = KModel(config=str(WEIGHTS / "config.json"),
                   model=str(WEIGHTS / "kokoro-v1_0.pth"))
    model.eval().to(device)
    voice_pack = torch.load(WEIGHTS / "voices" / f"{args.voice}.pt",
                            map_location=device, weights_only=True)
    voice_bin = WEIGHTS / "voices" / f"{args.voice}.bin"
    if not voice_bin.exists():
        print(f"error: missing C++ voice pack {voice_bin}", file=sys.stderr)
        return 2

    synth_exe = find_synth_exe()
    out_root = args.out_dir / args.voice
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"synth.exe = {synth_exe}")
    print(f"voice     = {args.voice}")
    print(f"chunks    = {args.chunks}")
    print(f"out_dir   = {out_root}\n")

    print(f"{'tag':<40s}  {'sizes (samples)':<22s}  metrics")
    print("-" * 132)

    idx = 0
    for s_i, sentence in enumerate(sentences):
        chunks = split_chunks(sentence, args.chunks)
        for c_i, chunk in enumerate(chunks):
            phonemes, ids = phonemize(chunk, vocab)
            if not ids:
                print(f"[{idx:02d}] skipping empty chunk {chunk!r}")
                continue
            tag = f"s{s_i}c{c_i}: {chunk[:30]!r}"
            py_wav  = out_root / f"{idx:02d}_py.wav"
            cpp_wav = out_root / f"{idx:02d}_cpp.wav"
            try:
                py  = synth_py(model, voice_pack, phonemes)
                write_wav(py_wav, py)
                cpp = synth_cpp(synth_exe, voice_bin, ids, cpp_wav)
            except Exception as e:
                print(f"[{idx:02d}] {tag} -- ERROR: {e}")
                idx += 1
                continue
            m = compare(py, cpp)
            print(fmt_row(tag, m))
            idx += 1

    print(f"\nWrote {idx} comparison pairs under {out_root}")
    print("Listen with any wav player; metric definitions in script header.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
