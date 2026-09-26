#!/usr/bin/env python3
"""Convert the upstream LAION CLAP checkpoint to brosoundml's layout.

laion/larger_clap_general ships ONLY a pickled `pytorch_model.bin`. brosoundml
loads safetensors directly (src/clap*.cpp), so this script:

  1. Loads pytorch_model.bin (a flat ClapModel state dict).
  2. Drops the integer buffers the C++ port recomputes or never reads:
     text position_ids / token_type_ids (positions are rebuilt from the
     RoBERTa padding_idx rule), every Swin block's relative_position_index
     (rebuilt from the window size), and the BatchNorm num_batches_tracked.
  3. Checks config.json against the loader contract (unfused HTSAT, depths
     2/2/12/2, window 8, spec 256, 64 mel bins; RoBERTa 12x768) and every
     expected tensor's presence and shape.
  4. Re-casts every tensor to contiguous FP32 (the two logit scales stay
     0-d, as transformers expects) and writes model.safetensors beside the
     .bin.

Usage:
  scripts/convert-clap.py [--src DIR] [--force] [--dump-keys]

  --src DIR     checkpoint dir (default: weights/clap)
  --force       overwrite an existing model.safetensors
  --dump-keys   print every (name, shape, dtype) row and exit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

DROP_SUFFIXES = (
    "embeddings.position_ids",
    "embeddings.token_type_ids",
    "relative_position_index",
    "num_batches_tracked",
)


def expected_keys(cfg: dict) -> list[tuple[str, tuple[int, ...]]]:
    a = cfg.get("audio_config", {})
    depths = a.get("depths", [2, 2, 12, 2])
    heads = a.get("num_attention_heads", [4, 8, 16, 32])
    e0 = a.get("patch_embeds_hidden_size", 128)
    win = a.get("window_size", 8)
    mels = a.get("num_mel_bins", 64)
    P = cfg.get("projection_dim", 512)
    keys: list[tuple[str, tuple[int, ...]]] = [("logit_scale_a", ()), ("logit_scale_t", ())]

    A = "audio_model.audio_encoder."
    keys += [(A + "patch_embed.proj.weight", (e0, 1, 4, 4)), (A + "patch_embed.proj.bias", (e0,)),
             (A + "patch_embed.norm.weight", (e0,)), (A + "patch_embed.norm.bias", (e0,))]
    for n in ("weight", "bias", "running_mean", "running_var"):
        keys.append((A + "batch_norm." + n, (mels,)))
    for s, depth in enumerate(depths):
        d = e0 * 2 ** s
        for b in range(depth):
            p = f"{A}layers.{s}.blocks.{b}."
            keys += [(p + "layernorm_before.weight", (d,)), (p + "layernorm_before.bias", (d,)),
                     (p + "attention.self.relative_position_bias_table", ((2 * win - 1) ** 2, heads[s]))]
            for m in ("query", "key", "value"):
                keys += [(p + f"attention.self.{m}.weight", (d, d)), (p + f"attention.self.{m}.bias", (d,))]
            keys += [(p + "attention.output.dense.weight", (d, d)), (p + "attention.output.dense.bias", (d,)),
                     (p + "layernorm_after.weight", (d,)), (p + "layernorm_after.bias", (d,)),
                     (p + "intermediate.dense.weight", (4 * d, d)), (p + "intermediate.dense.bias", (4 * d,)),
                     (p + "output.dense.weight", (d, 4 * d)), (p + "output.dense.bias", (d,))]
        if s + 1 < len(depths):
            p = f"{A}layers.{s}.downsample."
            keys += [(p + "reduction.weight", (2 * d, 4 * d)),
                     (p + "norm.weight", (4 * d,)), (p + "norm.bias", (4 * d,))]
    top = e0 * 2 ** (len(depths) - 1)
    keys += [(A + "norm.weight", (top,)), (A + "norm.bias", (top,)),
             ("audio_projection.linear1.weight", (P, top)), ("audio_projection.linear1.bias", (P,)),
             ("audio_projection.linear2.weight", (P, P)), ("audio_projection.linear2.bias", (P,))]

    t = cfg.get("text_config", {})
    D = t.get("hidden_size", 768)
    I = t.get("intermediate_size", 3072)
    V = t.get("vocab_size", 50265)
    L = t.get("num_hidden_layers", 12)
    T = "text_model."
    keys += [(T + "embeddings.word_embeddings.weight", (V, D)),
             (T + "embeddings.position_embeddings.weight", (t.get("max_position_embeddings", 514), D)),
             (T + "embeddings.token_type_embeddings.weight", (t.get("type_vocab_size", 1), D)),
             (T + "embeddings.LayerNorm.weight", (D,)), (T + "embeddings.LayerNorm.bias", (D,))]
    for i in range(L):
        p = f"{T}encoder.layer.{i}."
        for m in ("query", "key", "value"):
            keys += [(p + f"attention.self.{m}.weight", (D, D)), (p + f"attention.self.{m}.bias", (D,))]
        keys += [(p + "attention.output.dense.weight", (D, D)), (p + "attention.output.dense.bias", (D,)),
                 (p + "attention.output.LayerNorm.weight", (D,)), (p + "attention.output.LayerNorm.bias", (D,)),
                 (p + "intermediate.dense.weight", (I, D)), (p + "intermediate.dense.bias", (I,)),
                 (p + "output.dense.weight", (D, I)), (p + "output.dense.bias", (D,)),
                 (p + "output.LayerNorm.weight", (D,)), (p + "output.LayerNorm.bias", (D,))]
    keys += [(T + "pooler.dense.weight", (D, D)), (T + "pooler.dense.bias", (D,)),
             ("text_projection.linear1.weight", (P, D)), ("text_projection.linear1.bias", (P,)),
             ("text_projection.linear2.weight", (P, P)), ("text_projection.linear2.bias", (P,))]
    return keys


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=repo / "weights" / "clap")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dump-keys", action="store_true")
    args = ap.parse_args()

    src: Path = args.src
    cfg_path, bin_path = src / "config.json", src / "pytorch_model.bin"
    for p in (cfg_path, bin_path):
        if not p.exists():
            print(f"error: {p} missing — run scripts/download-clap.sh first", file=sys.stderr)
            return 2
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    a = cfg.get("audio_config", {})
    if a.get("enable_fusion", False):
        print("error: fused CLAP checkpoints are not supported (unfused HTSAT only)", file=sys.stderr)
        return 2

    sd = torch.load(bin_path, map_location="cpu", weights_only=False)
    if args.dump_keys:
        for n, t in sorted(sd.items()):
            print(f"  {n}  {tuple(t.shape)}  {t.dtype}")
        return 0

    out: dict[str, torch.Tensor] = {}
    dropped = 0
    for n, t in sd.items():
        if n.endswith(DROP_SUFFIXES):
            dropped += 1
            continue
        out[n] = t.detach().to(dtype=torch.float32, device="cpu").contiguous()
    print(f"loaded {len(sd)} tensors, dropped {dropped} integer buffers")

    bad = False
    for name, shape in expected_keys(cfg):
        if name not in out:
            print(f"MISSING {name}", file=sys.stderr)
            bad = True
        elif tuple(out[name].shape) != shape:
            print(f"WRONG SHAPE {name}: want {shape} got {tuple(out[name].shape)}", file=sys.stderr)
            bad = True
    extras = sorted(set(out) - {n for n, _ in expected_keys(cfg)})
    for n in extras:
        print(f"  extra (kept): {n} {tuple(out[n].shape)}")
    if bad:
        return 1

    dst = src / "model.safetensors"
    if dst.exists() and not args.force:
        print(f"error: {dst} exists — pass --force to overwrite", file=sys.stderr)
        return 2
    save_file(out, str(dst))
    print(f"wrote {len(out)} tensors -> {dst} ({dst.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
