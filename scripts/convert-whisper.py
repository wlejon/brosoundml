#!/usr/bin/env python3
"""Validate an upstream Whisper checkpoint for brosoundml's expected layout.

Hugging Face's `openai/whisper-*` repos ship:
  * config.json                    (already brosoundml-compatible)
  * model.safetensors              (flat name -> tensor)
  * vocab.json + merges.txt        (consumed by brolm::whisper::Tokenizer)

brosoundml's loader expects the same key naming, so unlike Kokoro there is no
.pth-to-safetensors conversion step — this script's job is to:

  1. Verify config.json carries every field the C++ loader requires.
  2. Walk model.safetensors and verify every expected encoder / decoder key
     is present at the expected shape.
  3. If any tensor is not FP32, re-cast and rewrite the file in place
     (brosoundml's CPU forward path is FP32 only).
  4. Print a one-page summary so you can eyeball what's there.

Usage:
  scripts/convert-whisper.py [--src DIR] [--dump-keys] [--force-fp32]

  --src DIR       checkpoint directory (default: weights/whisper)
  --dump-keys     print every (name, shape, dtype) row and exit
  --force-fp32    rewrite model.safetensors in FP32 even if it's already FP32
                  (useful if you suspect bit-rot)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# --- contract: what brosoundml's C++ loader requires --------------------------
#
# Mirrors include/brosoundml/whisper.h (WhisperConfig) and
# src/whisper_modules.cpp (encoder load_from + decoder load_from).
REQUIRED_CONFIG_KEYS = [
    "vocab_size",
    "num_mel_bins",
    "d_model",
    "max_source_positions",
    "max_target_positions",
    "encoder_layers",
    "encoder_attention_heads",
    "encoder_ffn_dim",
    "decoder_layers",
    "decoder_attention_heads",
    "decoder_ffn_dim",
]


def expected_tensor_keys(cfg: dict) -> list[tuple[str, tuple[int, ...]]]:
    """Return the (name, shape) contract derived from the config.

    Shapes follow the HF convention as we observed it loading whisper-tiny:
      - conv weights are (C_out, C_in, kL)
      - linear weights are (out, in)
      - layer-norm weights/biases are (D,)
      - embeddings are (V, D) or (max_pos, D)
      - Whisper's K projection has NO bias — k_proj.bias is intentionally absent.
    """
    d   = cfg["d_model"]
    ne  = cfg["encoder_layers"]
    nd  = cfg["decoder_layers"]
    eff = cfg["encoder_ffn_dim"]
    dff = cfg["decoder_ffn_dim"]
    mb  = cfg["num_mel_bins"]
    msp = cfg["max_source_positions"]
    mtp = cfg["max_target_positions"]
    vs  = cfg["vocab_size"]

    keys: list[tuple[str, tuple[int, ...]]] = []

    # Encoder stem
    keys += [
        ("model.encoder.conv1.weight",          (d, mb, 3)),
        ("model.encoder.conv1.bias",            (d,)),
        ("model.encoder.conv2.weight",          (d, d, 3)),
        ("model.encoder.conv2.bias",            (d,)),
        ("model.encoder.embed_positions.weight",(msp, d)),
    ]
    for i in range(ne):
        p = f"model.encoder.layers.{i}"
        keys += [
            (f"{p}.self_attn.q_proj.weight",   (d, d)),
            (f"{p}.self_attn.q_proj.bias",     (d,)),
            (f"{p}.self_attn.k_proj.weight",   (d, d)),
            # k_proj.bias intentionally absent
            (f"{p}.self_attn.v_proj.weight",   (d, d)),
            (f"{p}.self_attn.v_proj.bias",     (d,)),
            (f"{p}.self_attn.out_proj.weight", (d, d)),
            (f"{p}.self_attn.out_proj.bias",   (d,)),
            (f"{p}.self_attn_layer_norm.weight", (d,)),
            (f"{p}.self_attn_layer_norm.bias",   (d,)),
            (f"{p}.fc1.weight",                  (eff, d)),
            (f"{p}.fc1.bias",                    (eff,)),
            (f"{p}.fc2.weight",                  (d, eff)),
            (f"{p}.fc2.bias",                    (d,)),
            (f"{p}.final_layer_norm.weight",     (d,)),
            (f"{p}.final_layer_norm.bias",       (d,)),
        ]
    keys += [
        ("model.encoder.layer_norm.weight", (d,)),
        ("model.encoder.layer_norm.bias",   (d,)),
    ]

    # Decoder
    keys += [
        ("model.decoder.embed_tokens.weight",    (vs, d)),
        ("model.decoder.embed_positions.weight", (mtp, d)),
    ]
    for i in range(nd):
        p = f"model.decoder.layers.{i}"
        keys += [
            (f"{p}.self_attn.q_proj.weight",   (d, d)),
            (f"{p}.self_attn.q_proj.bias",     (d,)),
            (f"{p}.self_attn.k_proj.weight",   (d, d)),
            # k_proj.bias absent
            (f"{p}.self_attn.v_proj.weight",   (d, d)),
            (f"{p}.self_attn.v_proj.bias",     (d,)),
            (f"{p}.self_attn.out_proj.weight", (d, d)),
            (f"{p}.self_attn.out_proj.bias",   (d,)),
            (f"{p}.self_attn_layer_norm.weight", (d,)),
            (f"{p}.self_attn_layer_norm.bias",   (d,)),
            (f"{p}.encoder_attn.q_proj.weight",   (d, d)),
            (f"{p}.encoder_attn.q_proj.bias",     (d,)),
            (f"{p}.encoder_attn.k_proj.weight",   (d, d)),
            # encoder_attn.k_proj.bias absent
            (f"{p}.encoder_attn.v_proj.weight",   (d, d)),
            (f"{p}.encoder_attn.v_proj.bias",     (d,)),
            (f"{p}.encoder_attn.out_proj.weight", (d, d)),
            (f"{p}.encoder_attn.out_proj.bias",   (d,)),
            (f"{p}.encoder_attn_layer_norm.weight", (d,)),
            (f"{p}.encoder_attn_layer_norm.bias",   (d,)),
            (f"{p}.fc1.weight",                  (dff, d)),
            (f"{p}.fc1.bias",                    (dff,)),
            (f"{p}.fc2.weight",                  (d, dff)),
            (f"{p}.fc2.bias",                    (d,)),
            (f"{p}.final_layer_norm.weight",     (d,)),
            (f"{p}.final_layer_norm.bias",       (d,)),
        ]
    keys += [
        ("model.decoder.layer_norm.weight", (d,)),
        ("model.decoder.layer_norm.bias",   (d,)),
    ]
    # proj_out.weight is OPTIONAL — usually tied to decoder.embed_tokens.weight.
    # The C++ loader prefers it if present, otherwise ties.

    return keys


def load_safetensors_meta(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Return {name: (shape, dtype_str)} without materialising tensors."""
    out: dict[str, tuple[tuple[int, ...], str]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            out[name] = (tuple(t.shape), str(t.dtype))
    return out


def all_fp32(path: Path) -> bool:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if t.dtype != torch.float32:
                return False
    return True


def rewrite_fp32(path: Path) -> int:
    """Re-cast every tensor to FP32 contiguous on CPU and overwrite the file.
    Returns the number of tensors rewritten."""
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            tensors[name] = t.detach().to(dtype=torch.float32,
                                           device="cpu").contiguous()
    save_file(tensors, str(path))
    return len(tensors)


def dump_keys(meta: dict[str, tuple[tuple[int, ...], str]]) -> None:
    rows = sorted(meta.items())
    width = max((len(n) for n, _ in rows), default=0)
    for name, (shape, dtype) in rows:
        print(f"  {name.ljust(width)}  {shape}  {dtype}")
    print(f"\n{len(rows)} tensors")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_src = repo_root / "weights" / "whisper"

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=default_src,
                    help=f"checkpoint dir (default: {default_src.relative_to(repo_root)})")
    ap.add_argument("--dump-keys", action="store_true",
                    help="print every (name, shape, dtype) row and exit")
    ap.add_argument("--force-fp32", action="store_true",
                    help="rewrite model.safetensors in FP32 even if already FP32")
    args = ap.parse_args()

    src: Path = args.src
    if not src.exists():
        print(f"error: source dir {src} does not exist — run "
              "scripts/download-whisper.sh first", file=sys.stderr)
        return 2

    cfg_path = src / "config.json"
    st_path  = src / "model.safetensors"
    vocab_path = src / "vocab.json"
    added_path = src / "added_tokens.json"
    if not cfg_path.exists():
        print(f"error: no config.json under {src}", file=sys.stderr); return 2
    if not st_path.exists():
        print(f"error: no model.safetensors under {src}", file=sys.stderr); return 2

    # --- tokenizer: merge added_tokens.json into vocab.json ------------------
    #
    # HF splits Whisper's vocabulary: vocab.json has the 50257 BPE entries,
    # added_tokens.json has the ~1600 special tokens (sot, lang ids, task ids,
    # timestamp ids). Brolm's whisper::Tokenizer auto-registers anything that
    # looks like "<|...|>" but only sees what's in vocab.json — so we merge.
    if vocab_path.exists() and added_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        added = json.loads(added_path.read_text(encoding="utf-8"))
        new_keys = [k for k in added if k not in vocab]
        if new_keys:
            # Detect id collisions before overwriting — would silently corrupt
            # the tokenizer otherwise.
            existing_ids = set(vocab.values())
            for k in new_keys:
                if added[k] in existing_ids:
                    print(f"error: added token '{k}' id {added[k]} collides "
                          f"with an existing vocab entry", file=sys.stderr)
                    return 2
            for k in new_keys:
                vocab[k] = added[k]
            vocab_path.write_text(
                json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
            print(f"merged {len(new_keys)} special tokens "
                  f"from added_tokens.json into vocab.json "
                  f"(vocab is now {len(vocab)} entries)")
        else:
            print(f"vocab.json already contains all added_tokens entries "
                  f"({len(vocab)} entries) — no merge needed")
    elif vocab_path.exists() and not added_path.exists():
        print("warning: vocab.json present but no added_tokens.json — special "
              "tokens like <|startoftranscript|> may be missing", file=sys.stderr)

    # --- config.json checks -------------------------------------------------
    # Whisper's config.json carries non-ASCII strings (language names etc.);
    # force UTF-8 so Windows' cp1252 default doesn't trip us.
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        print(f"error: config.json missing required keys: {missing}", file=sys.stderr)
        return 2
    print(f"config.json OK")
    print(f"  vocab_size={cfg['vocab_size']}  num_mel_bins={cfg['num_mel_bins']}  "
          f"d_model={cfg['d_model']}")
    print(f"  encoder: layers={cfg['encoder_layers']} heads={cfg['encoder_attention_heads']} "
          f"ffn={cfg['encoder_ffn_dim']}")
    print(f"  decoder: layers={cfg['decoder_layers']} heads={cfg['decoder_attention_heads']} "
          f"ffn={cfg['decoder_ffn_dim']}")
    print(f"  positions: source={cfg['max_source_positions']} "
          f"target={cfg['max_target_positions']}")

    # --- model.safetensors walk ---------------------------------------------
    meta = load_safetensors_meta(st_path)
    if args.dump_keys:
        dump_keys(meta)
        return 0

    expected = expected_tensor_keys(cfg)
    present_names = set(meta.keys())
    expected_names = {n for n, _ in expected}

    missing_keys = [n for n, _ in expected if n not in present_names]
    bad_shapes = []
    for name, want in expected:
        got = meta.get(name)
        if got is None:
            continue
        if got[0] != want:
            bad_shapes.append((name, want, got[0]))

    extras = sorted(present_names - expected_names)
    # proj_out.weight is the expected "extra" — every other unexpected key
    # is worth surfacing.
    benign_extras = {"proj_out.weight"}
    noisy_extras = [n for n in extras if n not in benign_extras]

    print(f"\nmodel.safetensors: {len(meta)} tensors")
    if "proj_out.weight" in present_names:
        print(f"  proj_out.weight: explicit (C++ loader will use it)")
    else:
        print(f"  proj_out.weight: absent — C++ loader will tie to decoder.embed_tokens")

    if missing_keys:
        print(f"\nMISSING ({len(missing_keys)}) expected tensors:", file=sys.stderr)
        for k in missing_keys[:20]:
            print(f"  - {k}", file=sys.stderr)
        if len(missing_keys) > 20:
            print(f"  ...and {len(missing_keys) - 20} more", file=sys.stderr)

    if bad_shapes:
        print(f"\nWRONG SHAPE ({len(bad_shapes)}) tensors:", file=sys.stderr)
        for name, want, got in bad_shapes[:20]:
            print(f"  - {name}: want {want} got {got}", file=sys.stderr)

    if noisy_extras:
        print(f"\nUnexpected extras ({len(noisy_extras)}):")
        for k in noisy_extras[:10]:
            print(f"  + {k}  shape={meta[k][0]}  dtype={meta[k][1]}")
        if len(noisy_extras) > 10:
            print(f"  ...and {len(noisy_extras) - 10} more")

    if missing_keys or bad_shapes:
        return 1

    # --- dtype check / optional rewrite -------------------------------------
    if args.force_fp32 or not all_fp32(st_path):
        action = "force-rewriting" if args.force_fp32 else "re-casting non-FP32 tensors"
        print(f"\n{action} {st_path.name} in FP32 ...")
        n = rewrite_fp32(st_path)
        size = st_path.stat().st_size
        print(f"  wrote {n} tensors ({size:,} bytes)")
    else:
        print(f"\nall tensors already FP32 — no rewrite needed")

    print("\nDone. The checkpoint is ready for brosoundml::Whisper::load().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
