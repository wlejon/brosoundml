#!/usr/bin/env python3
"""Builder for brosoundml's packed English lexicon.

Fetches misaki's US-English pronunciation dictionaries (us_gold.json and
us_silver.json) at a pinned upstream commit, merges them (silver first, gold
wins on conflict), sanitises POS-variant entries, and packs the result into
the v1 ``lexicon_en_us.bin`` format documented in ``docs/lexicon.md``.

One-time tooling. Not part of the C++ build. Re-run when bumping the misaki
pin or the file format version.

Usage::

    python tools/build_lexicon.py
    python tools/build_lexicon.py --out path/to/lexicon_en_us.bin
    python tools/build_lexicon.py --commit <sha>

Python 3.10+, standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path
from typing import Any

# Pinned misaki commit. See docs/lexicon.md "Upstream source".
DEFAULT_COMMIT = "fba1236595f2d2bf21d414ba6e57d25256afada3"

GOLD_PATH = "misaki/data/us_gold.json"
SILVER_PATH = "misaki/data/us_silver.json"

# Closed set of misaki POS tags, with their on-disk ids. Builder rejects any
# variant entry that names a tag outside this set.
TAG_IDS: dict[str, int] = {
    "DEFAULT": 0,
    "NOUN": 1,
    "VERB": 2,
    "ADJ": 3,
    "ADV": 4,
    "VBD": 5,
    "VBN": 6,
    "VBP": 7,
    "DT": 8,
}

MAGIC = b"BSLX"
FORMAT_VERSION = 1
HEADER_SIZE = 64
INDEX_ENTRY_SIZE = 16

# Header layout: magic(4) version(I) entry_count(I) flags(I)
# key_blob_off(Q) key_blob_len(Q) val_blob_off(Q) val_blob_len(Q)
# reserved(16s)  -> total 4+4+4+4+8+8+8+8+16 = 64
HEADER_STRUCT = struct.Struct("<4sIII QQQQ 16s")
assert HEADER_STRUCT.size == HEADER_SIZE

# Index entry: key_off(I) key_len(H) flags(B) variant_count(B) val_off(I) val_len(I)
INDEX_STRUCT = struct.Struct("<IHBBII")
assert INDEX_STRUCT.size == INDEX_ENTRY_SIZE


def fetch_json(url: str) -> dict[str, Any]:
    """Download ``url`` and parse it as JSON, returning the object."""
    print(f"  fetching {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "brosoundml-build-lexicon"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def sanitise(
    entries: dict[str, Any],
) -> tuple[dict[str, str | dict[str, str]], set[str]]:
    """Clean up the merged misaki dict.

    For each entry:
    - plain string values pass through unchanged.
    - dict (variant) values are filtered: drop the stringified ``"None"`` key,
      drop any POS key whose value is ``null``, drop any POS key not in the
      closed ``TAG_IDS`` set (and record it for a warning).
    - any variant entry that lacks ``DEFAULT`` after filtering is fatal.

    Returns the cleaned mapping and the set of unknown POS keys encountered.
    """
    cleaned: dict[str, str | dict[str, str]] = {}
    unknown_pos: set[str] = set()
    missing_default: list[str] = []

    for key, value in entries.items():
        if isinstance(value, str):
            cleaned[key] = value
            continue
        if not isinstance(value, dict):
            raise RuntimeError(
                f"unexpected value type for key {key!r}: {type(value).__name__}"
            )

        filtered: dict[str, str] = {}
        for pos, ipa in value.items():
            if pos == "None":
                # Stringified Python None — treat as variant absence.
                continue
            if ipa is None:
                # Explicit null variant — "no special pronunciation".
                continue
            if pos not in TAG_IDS:
                unknown_pos.add(pos)
                continue
            if not isinstance(ipa, str):
                raise RuntimeError(
                    f"variant {pos!r} on {key!r} is not a string: {type(ipa).__name__}"
                )
            filtered[pos] = ipa

        if "DEFAULT" not in filtered:
            missing_default.append(key)
            continue

        if len(filtered) == 1:
            # Only DEFAULT survived — degrade to a plain entry. Equivalent
            # semantically and saves a record header per word.
            cleaned[key] = filtered["DEFAULT"]
        else:
            cleaned[key] = filtered

    if missing_default:
        raise RuntimeError(
            f"refusing to write bin: {len(missing_default)} variant entries "
            f"lack DEFAULT after sanitising "
            f"(first few: {missing_default[:5]!r})"
        )

    return cleaned, unknown_pos


def encode_variant_block(variants: dict[str, str]) -> bytes:
    """Pack a variant table into the on-disk record format.

    Each record is ``tag_id (u8) | ipa_len (u16) | ipa (utf-8 bytes)``.
    Records are sorted by ``tag_id`` ascending, so ``DEFAULT`` (id 0) is
    first.
    """
    items = sorted(variants.items(), key=lambda kv: TAG_IDS[kv[0]])
    parts: list[bytes] = []
    for tag, ipa in items:
        ipa_bytes = ipa.encode("utf-8")
        if len(ipa_bytes) > 0xFFFF:
            raise RuntimeError(
                f"variant ipa too long ({len(ipa_bytes)} bytes) for tag {tag!r}"
            )
        parts.append(struct.pack("<BH", TAG_IDS[tag], len(ipa_bytes)))
        parts.append(ipa_bytes)
    return b"".join(parts)


def pack(items: list[tuple[str, str | dict[str, str]]]) -> bytes:
    """Build the full on-disk byte buffer from a sorted list of entries."""
    key_blob = bytearray()
    val_blob = bytearray()
    index_records: list[bytes] = []
    variant_count_total = 0

    for key, value in items:
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise RuntimeError(f"key too long ({len(key_bytes)} bytes): {key!r}")
        key_off = len(key_blob)
        key_blob.extend(key_bytes)

        val_off = len(val_blob)
        if isinstance(value, str):
            value_bytes = value.encode("utf-8")
            val_blob.extend(value_bytes)
            flags = 0
            variant_count = 0
            val_len = len(value_bytes)
        else:
            block = encode_variant_block(value)
            val_blob.extend(block)
            flags = 1
            variant_count = len(value)
            if variant_count > 0xFF:
                raise RuntimeError(
                    f"too many variants ({variant_count}) for key {key!r}"
                )
            val_len = len(block)
            variant_count_total += 1

        if key_off > 0xFFFFFFFF or val_off > 0xFFFFFFFF or val_len > 0xFFFFFFFF:
            raise RuntimeError("offset/length overflow — exceeds u32 range")

        index_records.append(
            INDEX_STRUCT.pack(key_off, len(key_bytes), flags, variant_count, val_off, val_len)
        )

    entry_count = len(items)
    index_bytes = b"".join(index_records)
    assert len(index_bytes) == entry_count * INDEX_ENTRY_SIZE

    key_blob_off = HEADER_SIZE + len(index_bytes)
    key_blob_len = len(key_blob)
    val_blob_off = key_blob_off + key_blob_len
    val_blob_len = len(val_blob)

    header = HEADER_STRUCT.pack(
        MAGIC,
        FORMAT_VERSION,
        entry_count,
        0,  # flags
        key_blob_off,
        key_blob_len,
        val_blob_off,
        val_blob_len,
        b"\x00" * 16,
    )

    return header + index_bytes + bytes(key_blob) + bytes(val_blob), variant_count_total


def repo_root() -> Path:
    """Return the brosoundml repo root (parent of ``tools/``)."""
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_out = repo_root().parent / "brosoundml-data" / "g2p" / "lexicon_en_us.bin"
    p.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"output path (default: {default_out})",
    )
    p.add_argument(
        "--commit",
        default=DEFAULT_COMMIT,
        help=f"misaki commit to pull (default: {DEFAULT_COMMIT})",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    commit = args.commit
    out_path: Path = args.out

    base = f"https://raw.githubusercontent.com/hexgrad/misaki/{commit}"
    gold_url = f"{base}/{GOLD_PATH}"
    silver_url = f"{base}/{SILVER_PATH}"

    print(f"misaki commit: {commit}", file=sys.stderr)
    gold = fetch_json(gold_url)
    silver = fetch_json(silver_url)

    print(
        f"  fetched {len(silver)} silver entries, {len(gold)} gold entries",
        file=sys.stderr,
    )

    merged: dict[str, Any] = {}
    for k, v in silver.items():
        merged[k] = v
    for k, v in gold.items():
        merged[k] = v

    cleaned, unknown_pos = sanitise(merged)
    if unknown_pos:
        print(
            f"warning: dropped unknown POS keys: {sorted(unknown_pos)!r}",
            file=sys.stderr,
        )

    items = sorted(cleaned.items(), key=lambda kv: kv[0].encode("utf-8"))
    buf, variant_count = pack(items)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(buf)
    os.replace(tmp_path, out_path)

    size_bytes = len(buf)
    print(
        f"wrote {out_path} ({size_bytes:,} bytes, {size_bytes / 1024 / 1024:.2f} MB)",
        file=sys.stderr,
    )
    print(
        f"  entries: {len(items):,}  variant-typed: {variant_count:,}",
        file=sys.stderr,
    )
    print(f"  misaki commit: {commit}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
