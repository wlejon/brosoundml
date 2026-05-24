#!/usr/bin/env python3
"""Convert English text to a Kokoro phoneme-id sequence (offline).

Pipeline:
  text -> eng_to_ipa -> normalize to Kokoro vocab chars -> id list

eng_to_ipa is pure Python and depends on no native libs. The resulting IPA
isn't bit-identical to misaki (the upstream Kokoro frontend), but it covers
the same vocabulary and pronounces clearly. Substitutions match the chars
present in Kokoro-82M's config.json vocab — e.g. ASCII 'g' is rewritten to
'ɡ' (U+0261) and 'r' to 'ɹ' so the model sees the IPA codepoints it was
trained on.

Usage:
  scripts/phonemize.py "your sentence here"
  scripts/phonemize.py --file sentences.txt
  scripts/phonemize.py --out ids.txt "Hello world"

If `--out` is set, writes one phoneme-id sequence per line (the model's
BOS/EOS 0s are NOT added — Kokoro::synthesize adds them). Otherwise prints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import eng_to_ipa as ipa

# Map characters eng_to_ipa emits but Kokoro's vocab doesn't carry, to the
# vocab character with the same phonetic meaning. The right-hand side must
# appear in config.json's vocab map (verified at runtime).
IPA_SUBSTITUTIONS = {
    "g": "ɡ",  # U+0067 -> U+0261 (latin script g) — Kokoro uses the IPA glyph
    "r": "ɹ",  # alveolar approximant
    "*": "",   # eng_to_ipa marks "uncertain" with '*' — drop
}


def normalise(text: str) -> str:
    out = []
    for ch in text:
        sub = IPA_SUBSTITUTIONS.get(ch, ch)
        out.append(sub)
    return "".join(out)


def to_ids(text: str, vocab: dict[str, int]) -> list[int]:
    raw = ipa.convert(text)
    normed = normalise(raw)
    ids = []
    skipped: list[str] = []
    for ch in normed:
        if ch in vocab:
            ids.append(vocab[ch])
        else:
            skipped.append(ch)
    if skipped:
        # Surface dropped chars to stderr so the caller can refine the mapping;
        # eng_to_ipa never emits a huge variety so this should stay quiet.
        unique = sorted(set(skipped))
        print(f"warning: dropped {len(skipped)} chars not in Kokoro vocab: "
              f"{unique!r}", file=sys.stderr)
    return ids


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", nargs="*", help="Sentence to phonemize (or --file)")
    ap.add_argument("--file", type=Path, help="Read sentences (one per line) from FILE")
    ap.add_argument("--out",  type=Path, help="Write `ids,ids,...` lines here")
    ap.add_argument("--config", type=Path,
                    default=repo_root / "weights" / "kokoro" / "config.json")
    args = ap.parse_args()

    if args.file:
        sentences = [l.strip() for l in args.file.read_text(encoding="utf-8").splitlines()
                     if l.strip()]
    elif args.text:
        sentences = [" ".join(args.text)]
    else:
        ap.print_help()
        return 2

    vocab: dict[str, int] = json.loads(args.config.read_text(encoding="utf-8"))["vocab"]
    out_lines = []
    for s in sentences:
        ids = to_ids(s, vocab)
        out_lines.append(",".join(str(i) for i in ids))
        print(f"{s!r}")
        print(f"  ipa: {normalise(ipa.convert(s))!r}")
        print(f"  ids: {ids}")
    if args.out:
        args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
