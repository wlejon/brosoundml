#!/usr/bin/env python3
"""Convert an upstream Kokoro-82M checkpoint to brosoundml's expected layout.

Upstream `hexgrad/Kokoro-82M` ships:
  * config.json                            (already brosoundml-compatible)
  * kokoro-v1_0.pth                        (PyTorch pickled state dict)
  * voices/<name>.pt                       (PyTorch pickled tensor)

brosoundml's stage-1 loader expects:
  * config.json                            (passed through verbatim)
  * model.safetensors                      (flat name -> tensor)
  * voices/<name>.bin                      (raw little-endian FP32, shape
                                            (rows, voice_dim), row-major)

This script reads the upstream files, prints the model's top-level keys
(useful while wiring stage 3+), and emits the converted artifacts in-place
under the same directory tree.

Usage:
  scripts/convert-kokoro.py [--src DIR] [--dst DIR] [--dump-keys] [--force]

  --src DIR       upstream directory (default: weights/kokoro)
  --dst DIR       output directory   (default: same as --src — in-place)
  --dump-keys     print every (name, shape, dtype) row from the checkpoint and exit
  --force         overwrite existing model.safetensors / voices/*.bin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def load_state_dict(pth: Path) -> dict:
    """Load the upstream .pth — sometimes a bare state dict, sometimes nested
    under a "model"/"state_dict" key."""
    blob = torch.load(pth, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        # Some Kokoro checkpoints ship the whole module dict (text_encoder,
        # predictor, decoder, plbert, ...) at the top level — each entry is
        # itself a state_dict. Flatten to "<group>.<name>" keys so the C++
        # loader sees one flat namespace.
        if all(isinstance(v, dict) for v in blob.values()):
            flat = {}
            for group, sd in blob.items():
                for name, tensor in sd.items():
                    flat[f"{group}.{name}"] = tensor
            return flat
        if "state_dict" in blob and isinstance(blob["state_dict"], dict):
            return blob["state_dict"]
        if "model" in blob and isinstance(blob["model"], dict):
            return blob["model"]
    return blob  # already a {name: tensor} dict


def to_fp32_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Force FP32, contiguous, on CPU. Stage 1 reads tensors as F32 from
    safetensors; the upstream Kokoro checkpoint is already FP32 but we keep
    the conversion defensive."""
    return t.detach().to(dtype=torch.float32, device="cpu").contiguous()


def fuse_weight_norm(state: dict) -> dict:
    """Replace every (weight_g, weight_v) pair with a single fused `weight`.

    Kokoro's checkpoint stores all of its 1D convolutions under PyTorch's
    weight-norm parameterization: weight = weight_g * (weight_v / ||weight_v||).
    The brosoundml C++ side has no weight-norm runtime, so we fuse host-side
    and the loader sees a plain `weight` key alongside `bias`.

    Norm is computed over every axis EXCEPT dim 0 (the output-channel axis),
    matching torch.nn.utils.weight_norm(module, name='weight', dim=0).
    """
    fused: dict = {}
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            fused[name] = value
            continue
        if name.endswith(".weight_g"):
            stem = name[: -len(".weight_g")]
            pairs.setdefault(stem, {})["g"] = value
        elif name.endswith(".weight_v"):
            stem = name[: -len(".weight_v")]
            pairs.setdefault(stem, {})["v"] = value
        else:
            fused[name] = value

    for stem, parts in pairs.items():
        if "g" not in parts or "v" not in parts:
            # Unpaired half — pass through under the original name so a
            # subsequent debugger sees the orphan.
            for half, t in parts.items():
                fused[f"{stem}.weight_{half}"] = t
            continue
        g, v = parts["g"], parts["v"]
        # weight = g * v / ||v||, norm taken over every axis but the first.
        reduce_dims = tuple(range(1, v.dim()))
        denom = v.norm(p=2, dim=reduce_dims, keepdim=True).clamp_min(1e-12)
        weight = g * (v / denom)
        fused[f"{stem}.weight"] = weight
    return fused


def write_safetensors(state: dict, out_path: Path) -> None:
    """Write the (flat name -> tensor) state dict as a single safetensors
    file. Filters out any non-tensor entries (some checkpoints stash scalars
    like epoch counters alongside the weights)."""
    tensors = {}
    skipped = []
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            tensors[name] = to_fp32_contiguous(value)
        else:
            skipped.append((name, type(value).__name__))
    if skipped:
        print(f"  (skipping {len(skipped)} non-tensor entries: "
              f"{', '.join(n for n, _ in skipped[:5])}"
              f"{'...' if len(skipped) > 5 else ''})")
    save_file(tensors, str(out_path))


def write_voice_bin(pt_path: Path, bin_path: Path) -> tuple[int, int]:
    """Convert one upstream voice .pt to a raw FP32 .bin matching stage 1's
    `(rows, voice_dim)` layout. Returns (rows, voice_dim) of the flattened
    pack so the caller can sanity-check shapes."""
    voice = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(voice, torch.Tensor):
        raise RuntimeError(f"voice {pt_path.name} is not a tensor: {type(voice)}")
    voice = to_fp32_contiguous(voice)
    # Upstream packs are commonly (rows, 1, voice_dim) — flatten the singleton.
    if voice.ndim == 3 and voice.shape[1] == 1:
        voice = voice.reshape(voice.shape[0], voice.shape[2])
    if voice.ndim != 2:
        raise RuntimeError(
            f"voice {pt_path.name} has unexpected shape {tuple(voice.shape)} "
            "(want 2D after squeezing the singleton middle dim)"
        )
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    voice.numpy().tofile(str(bin_path))
    return int(voice.shape[0]), int(voice.shape[1])


def dump_keys(state: dict) -> None:
    """Tabular dump of every parameter in the checkpoint — the primary
    artifact you read while writing stage 3+'s name-to-module wiring."""
    rows = []
    for name, tensor in state.items():
        if isinstance(tensor, torch.Tensor):
            rows.append((name, tuple(tensor.shape), str(tensor.dtype)))
    rows.sort(key=lambda r: r[0])
    width = max((len(n) for n, _, _ in rows), default=0)
    for name, shape, dtype in rows:
        print(f"  {name.ljust(width)}  {shape}  {dtype}")
    print(f"\n{len(rows)} tensors")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_src = repo_root / "weights" / "kokoro"

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=default_src)
    ap.add_argument("--dst", type=Path, default=None,
                    help="output dir (default: same as --src)")
    ap.add_argument("--dump-keys", action="store_true",
                    help="print every (name, shape, dtype) row and exit")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing converted outputs")
    args = ap.parse_args()

    src: Path = args.src
    dst: Path = args.dst or args.src
    if not src.exists():
        print(f"error: source dir {src} does not exist — run "
              "scripts/download-kokoro.sh first", file=sys.stderr)
        return 2

    pth = src / "kokoro-v1_0.pth"
    if not pth.exists():
        print(f"error: no kokoro-v1_0.pth under {src}", file=sys.stderr)
        return 2

    print(f"Loading {pth} ...")
    state = load_state_dict(pth)
    n_tensors = sum(1 for v in state.values() if isinstance(v, torch.Tensor))
    print(f"  {n_tensors} tensors (raw)")

    state = fuse_weight_norm(state)
    n_after = sum(1 for v in state.values() if isinstance(v, torch.Tensor))
    print(f"  {n_after} tensors (after weight-norm fusion)")

    if args.dump_keys:
        dump_keys(state)
        return 0

    # --- config.json: passthrough ------------------------------------------
    src_cfg = src / "config.json"
    dst_cfg = dst / "config.json"
    if src_cfg != dst_cfg:
        dst_cfg.parent.mkdir(parents=True, exist_ok=True)
        dst_cfg.write_bytes(src_cfg.read_bytes())

    # Print a few fields for sanity.
    # Force UTF-8: Kokoro's config.json carries non-ASCII bytes in its vocab
    # (e.g. apostrophe glyphs) and Windows defaults `read_text` to cp1252.
    cfg = json.loads(src_cfg.read_text(encoding="utf-8"))
    print(f"  config: n_token={cfg.get('n_token')} hidden_dim={cfg.get('hidden_dim')} "
          f"style_dim={cfg.get('style_dim')}")

    # --- model.safetensors --------------------------------------------------
    out_st = dst / "model.safetensors"
    if out_st.exists() and not args.force:
        print(f"  {out_st.name} exists — skipping (use --force to overwrite)")
    else:
        write_safetensors(state, out_st)
        print(f"  wrote {out_st} ({out_st.stat().st_size:,} bytes)")

    # --- voices/*.bin -------------------------------------------------------
    src_voices = src / "voices"
    dst_voices = dst / "voices"
    if not src_voices.is_dir():
        print(f"warning: no voices/ dir under {src} — skipping voice conversion")
        return 0

    pt_files = sorted(src_voices.glob("*.pt"))
    if not pt_files:
        print("warning: no .pt voice packs found — skipping voice conversion")
        return 0

    shapes_seen: set[tuple[int, int]] = set()
    for pt in pt_files:
        out_bin = dst_voices / (pt.stem + ".bin")
        if out_bin.exists() and not args.force:
            continue
        rows, voice_dim = write_voice_bin(pt, out_bin)
        shapes_seen.add((rows, voice_dim))
    print(f"  voices: {len(pt_files)} converted; shapes seen: {sorted(shapes_seen)}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
