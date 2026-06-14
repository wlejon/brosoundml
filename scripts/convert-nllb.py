#!/usr/bin/env python3
"""Convert an upstream NLLB-200 (M2M-100) checkpoint to brolm's layout.

Hugging Face's `facebook/nllb-200-distilled-600M` ships ONLY a pickled
`pytorch_model.bin` (no safetensors) plus:
  * config.json                    (M2M-100 config — brolm reads it verbatim)
  * sentencepiece.bpe.model        (Unigram SPM)
  * tokenizer.json                 (fast tokenizer w/ the 200+ language codes)
  * special_tokens_map.json, tokenizer_config.json

brolm loads HF safetensors directly and runs a pure C++ FP32/FP16 forward, so
this script's job (mirrors convert-kokoro.py's .pth->safetensors step + the
convert-whisper.py validation) is to:

  1. Load pytorch_model.bin and flatten to a single {name: tensor} namespace.
  2. Drop the sinusoidal `embed_positions` buffers — M2M-100 positions are
     COMPUTED, not learned, so brolm regenerates them (nothing to load).
  3. Verify config.json carries every field the C++ loader requires.
  4. Walk the tensors and verify every expected encoder/decoder key is present
     at the expected shape (cross-attention included).
  5. Re-cast every tensor to contiguous FP32 and write model.safetensors.
  6. Print a one-page summary so you can eyeball what's there.

This is an encoder-decoder model whose BART/fairseq layer naming is identical
to Whisper's (self_attn / encoder_attn / fc1 / fc2 / final_layer_norm), so the
contract below deliberately parallels convert-whisper.py.

The converted directory (model.safetensors + config.json + the tokenizer files)
is what gets published to the brosoundml-data dataset repo — for now under a
text-model name even though that repo is otherwise audio artifacts.

Usage:
  scripts/convert-nllb.py [--src DIR] [--dst DIR] [--dump-keys] [--force]

  --src DIR       upstream checkpoint dir (default: weights/nllb)
  --dst DIR       output dir (default: same as --src — in-place)
  --dump-keys     print every (name, shape, dtype) row and exit
  --force         overwrite an existing model.safetensors
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


# --- contract: what brolm's C++ loader requires ------------------------------
#
# Mirrors include/brolm/nllb_config.h (NllbConfig) and the encoder/decoder
# load paths. M2M-100 puts bias on all four attention projections (unlike
# Whisper, whose k_proj has no bias), and uses pre-norm with a final
# layer_norm on each stack (normalize_before=True).
REQUIRED_CONFIG_KEYS = [
    "d_model",
    "encoder_layers",
    "decoder_layers",
    "encoder_attention_heads",
    "decoder_attention_heads",
    "encoder_ffn_dim",
    "decoder_ffn_dim",
    "vocab_size",
    "max_position_embeddings",
]


def attn_keys(prefix: str, name: str, d: int) -> list[tuple[str, tuple[int, ...]]]:
    """q/k/v/out projections — M2M-100 carries bias on all four."""
    keys: list[tuple[str, tuple[int, ...]]] = []
    for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
        keys.append((f"{prefix}.{name}.{proj}.weight", (d, d)))
        keys.append((f"{prefix}.{name}.{proj}.bias", (d,)))
    return keys


def layernorm_keys(name: str, d: int) -> list[tuple[str, tuple[int, ...]]]:
    return [(f"{name}.weight", (d,)), (f"{name}.bias", (d,))]


def expected_tensor_keys(cfg: dict) -> list[tuple[str, tuple[int, ...]]]:
    d = cfg["d_model"]
    ne = cfg["encoder_layers"]
    nd = cfg["decoder_layers"]
    eff = cfg["encoder_ffn_dim"]
    dff = cfg["decoder_ffn_dim"]
    vs = cfg["vocab_size"]

    keys: list[tuple[str, tuple[int, ...]]] = []

    # Shared token embedding (tied: encoder embed, decoder embed, lm_head).
    keys.append(("model.shared.weight", (vs, d)))

    # Encoder
    for i in range(ne):
        p = f"model.encoder.layers.{i}"
        keys += attn_keys(p, "self_attn", d)
        keys += layernorm_keys(f"{p}.self_attn_layer_norm", d)
        keys += [
            (f"{p}.fc1.weight", (eff, d)),
            (f"{p}.fc1.bias", (eff,)),
            (f"{p}.fc2.weight", (d, eff)),
            (f"{p}.fc2.bias", (d,)),
        ]
        keys += layernorm_keys(f"{p}.final_layer_norm", d)
    keys += layernorm_keys("model.encoder.layer_norm", d)

    # Decoder
    for i in range(nd):
        p = f"model.decoder.layers.{i}"
        keys += attn_keys(p, "self_attn", d)
        keys += layernorm_keys(f"{p}.self_attn_layer_norm", d)
        keys += attn_keys(p, "encoder_attn", d)
        keys += layernorm_keys(f"{p}.encoder_attn_layer_norm", d)
        keys += [
            (f"{p}.fc1.weight", (dff, d)),
            (f"{p}.fc1.bias", (dff,)),
            (f"{p}.fc2.weight", (d, dff)),
            (f"{p}.fc2.bias", (d,)),
        ]
        keys += layernorm_keys(f"{p}.final_layer_norm", d)
    keys += layernorm_keys("model.decoder.layer_norm", d)

    return keys


def load_state_dict(pth: Path) -> dict:
    """Load the upstream pytorch_model.bin — usually a flat {name: tensor}
    state dict, occasionally nested under 'model'/'state_dict'."""
    blob = torch.load(pth, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        if "state_dict" in blob and isinstance(blob["state_dict"], dict):
            return blob["state_dict"]
        if "model" in blob and isinstance(blob["model"], dict):
            return blob["model"]
    return blob


def is_droppable(name: str) -> bool:
    """Sinusoidal position buffers are recomputed in C++ — never exported.
    Also drop any leftover non-persistent buffers HF sometimes serialises."""
    return ".embed_positions." in name or name.endswith(".embed_positions.weights")


def dump_keys(sd: dict) -> None:
    rows = sorted((n, tuple(t.shape), str(t.dtype)) for n, t in sd.items())
    width = max((len(n) for n, _, _ in rows), default=0)
    for name, shape, dtype in rows:
        print(f"  {name.ljust(width)}  {shape}  {dtype}")
    print(f"\n{len(rows)} tensors")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_src = repo_root / "weights" / "nllb"

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=default_src,
                    help=f"checkpoint dir (default: {default_src})")
    ap.add_argument("--dst", type=Path, default=None,
                    help="output dir (default: same as --src)")
    ap.add_argument("--dump-keys", action="store_true",
                    help="print every (name, shape, dtype) row and exit")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing model.safetensors")
    args = ap.parse_args()

    src: Path = args.src
    dst: Path = args.dst or src
    if not src.exists():
        print(f"error: source dir {src} does not exist — run "
              "scripts/download-nllb.sh first", file=sys.stderr)
        return 2

    cfg_path = src / "config.json"
    bin_path = src / "pytorch_model.bin"
    if not cfg_path.exists():
        print(f"error: no config.json under {src}", file=sys.stderr); return 2
    if not bin_path.exists():
        print(f"error: no pytorch_model.bin under {src}", file=sys.stderr); return 2

    # --- config.json checks --------------------------------------------------
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        print(f"error: config.json missing required keys: {missing}", file=sys.stderr)
        return 2
    print("config.json OK")
    print(f"  d_model={cfg['d_model']}  vocab_size={cfg['vocab_size']}  "
          f"max_pos={cfg['max_position_embeddings']}  "
          f"scale_embedding={cfg.get('scale_embedding')}  "
          f"activation={cfg.get('activation_function')}")
    print(f"  encoder: layers={cfg['encoder_layers']} "
          f"heads={cfg['encoder_attention_heads']} ffn={cfg['encoder_ffn_dim']}")
    print(f"  decoder: layers={cfg['decoder_layers']} "
          f"heads={cfg['decoder_attention_heads']} ffn={cfg['decoder_ffn_dim']}")

    # --- load + flatten + drop position buffers ------------------------------
    print(f"\nloading {bin_path.name} ({bin_path.stat().st_size:,} bytes) ...")
    sd = load_state_dict(bin_path)
    dropped = [n for n in sd if is_droppable(n)]
    for n in dropped:
        del sd[n]
    if dropped:
        print(f"  dropped {len(dropped)} sinusoidal position buffer(s) "
              "(recomputed in C++)")

    if args.dump_keys:
        dump_keys(sd)
        return 0

    # --- validate against the contract ---------------------------------------
    expected = expected_tensor_keys(cfg)
    present = {n: tuple(t.shape) for n, t in sd.items()}
    present_names = set(present)
    expected_names = {n for n, _ in expected}

    # The shared embedding may surface under any of these tied names; accept
    # whichever the checkpoint provides and synthesise model.shared.weight.
    tie_aliases = ("model.shared.weight",
                   "model.encoder.embed_tokens.weight",
                   "model.decoder.embed_tokens.weight",
                   "lm_head.weight")
    have_embed = next((a for a in tie_aliases if a in present_names), None)
    if have_embed and "model.shared.weight" not in present_names:
        sd["model.shared.weight"] = sd[have_embed]
        present["model.shared.weight"] = tuple(sd["model.shared.weight"].shape)
        present_names.add("model.shared.weight")
        print(f"  tied embedding sourced from '{have_embed}'")

    missing_keys = [n for n, _ in expected if n not in present_names]
    bad_shapes = []
    for name, want in expected:
        got = present.get(name)
        if got is not None and got != want:
            bad_shapes.append((name, want, got))

    extras = sorted(present_names - expected_names - set(tie_aliases))

    print(f"\ntensors: {len(sd)}")
    if missing_keys:
        print(f"\nMISSING ({len(missing_keys)}) expected tensors:", file=sys.stderr)
        for k in missing_keys[:20]:
            print(f"  - {k}", file=sys.stderr)
        if len(missing_keys) > 20:
            print(f"  ...and {len(missing_keys) - 20} more", file=sys.stderr)
    if bad_shapes:
        print(f"\nWRONG SHAPE ({len(bad_shapes)}):", file=sys.stderr)
        for name, want, got in bad_shapes[:20]:
            print(f"  - {name}: want {want} got {got}", file=sys.stderr)
    if extras:
        print(f"\nUnexpected extras ({len(extras)}) — kept but worth a look:")
        for k in extras[:10]:
            print(f"  + {k}  shape={present[k]}")
        if len(extras) > 10:
            print(f"  ...and {len(extras) - 10} more")

    if missing_keys or bad_shapes:
        return 1

    # --- recast FP32 + write model.safetensors -------------------------------
    dst.mkdir(parents=True, exist_ok=True)
    out_path = dst / "model.safetensors"
    if out_path.exists() and not args.force:
        print(f"\nerror: {out_path} exists — pass --force to overwrite",
              file=sys.stderr)
        return 2

    tensors = {n: t.detach().to(dtype=torch.float32, device="cpu").contiguous()
               for n, t in sd.items()}
    save_file(tensors, str(out_path))
    print(f"\nwrote {len(tensors)} tensors -> {out_path} "
          f"({out_path.stat().st_size:,} bytes)")

    # --- carry the config + tokenizer files alongside (if dst != src) --------
    if dst.resolve() != src.resolve():
        for fname in ("config.json", "generation_config.json",
                      "sentencepiece.bpe.model", "tokenizer.json",
                      "tokenizer_config.json", "special_tokens_map.json"):
            srcf = src / fname
            if srcf.exists():
                shutil.copy2(srcf, dst / fname)
                print(f"  copied {fname}")

    print("\nDone. Publish the dst directory to brosoundml-data and point "
          "brolm's loader at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
