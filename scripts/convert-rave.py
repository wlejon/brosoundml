#!/usr/bin/env python3
"""Convert an exported RAVE v2 TorchScript model (.ts) into the safetensors +
config.json layout brosoundml's RAVE loader reads.

RAVE models from the ACIDS / Intelligent-Instruments-Lab collections ship as
*streaming* TorchScript exports (the `cached_conv` causal variant used by the
nn~/VST). They are not plain PyTorch state dicts, so this script loads the
scripted module, extracts the inner `_rave.*` weights, drops the streaming-only
buffers (conv caches, downsampling delays, batch-norm step counters) and the
`_prior.*` generative model we don't use, and writes:

    <out>/model.safetensors   all encoder/decoder/pqmf weights + latent PCA, FP32
    <out>/config.json         scalars the C++ port needs (sample rate, latent
                              dims, band count, leaky slope, bn eps, topology)

The streaming export uses causal convolutions (left-pad only), which line up
exactly with brosoundml's existing qcodec causal-conv helpers, so the C++ port
reproduces a single fresh-cache forward pass.

Usage:
    python convert-rave.py path/to/model_z8.ts  out/rave_z8
    python convert-rave.py path/to/model_z8.ts  out/rave_z8 --parity   # also dump reference I/O

Requires: torch, safetensors, numpy.
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

# Streaming-only / non-weight buffers that the offline C++ forward never reads.
_DROP_SUFFIXES = (
    ".cache.pad",
    ".downsampling_delay.pad",
    ".num_batches_tracked",
)


def _scalar(mod, name, default=None):
    """Read a scalar attribute/buffer off the scripted RAVE module."""
    try:
        v = getattr(mod, name)
    except Exception:
        return default
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.tolist()
    return v


def extract(ts_path):
    m = torch.jit.load(ts_path, map_location="cpu")
    m.eval()
    rave = m._rave

    sd = m.state_dict()
    tensors = {}
    for k, v in sd.items():
        if not k.startswith("_rave."):
            continue  # skip _prior.* and the top-level *_params buffers
        name = k[len("_rave."):]
        if any(name.endswith(s) for s in _DROP_SUFFIXES):
            continue
        tensors[name] = v.contiguous().float()

    # Per-encoder-conv groups: the variational head is a grouped conv
    # (groups=2 -> mean|scale). Record groups so the C++ side doesn't guess.
    import re
    enc_groups = {}
    for name, sub in m.named_modules():
        if not name.startswith("_rave.encoder.net."):
            continue
        code = str(getattr(sub, "code", ""))
        # torch.conv1d(x, weight, bias, [stride], [padding], [dilation], groups)
        mobj = re.search(r"torch\.conv1d\([^)]*?,\s*(\d+)\s*\)", code)
        if mobj:
            enc_groups[name[len("_rave.encoder.net."):]] = int(mobj.group(1))

    cfg = {
        "model": "rave_v2",
        "sampling_rate": int(_scalar(rave, "sampling_rate", 48000)),
        "full_latent_size": int(_scalar(rave, "latent_size", 128)),
        "cropped_latent_size": int(_scalar(rave, "cropped_latent_size", 0)),
        "n_band": int(tensors["pqmf.hk"].shape[0]),
        "deterministic": bool(_scalar(rave, "deterministic", True)),
        "trained_cropped": bool(_scalar(rave, "trained_cropped", False)),
        "stereo": bool(_scalar(rave, "stereo", False)),
        "use_noise": bool(_scalar(rave.decoder, "use_noise", True)),
        "loud_stride": int(_scalar(rave.decoder, "loud_stride", 1)),
        "leaky_slope": 0.2,          # nn.LeakyReLU(0.2) throughout (verified in graph)
        "bn_eps": 1.0e-5,            # nn.BatchNorm1d default
        "softplus_floor": 1.0e-4,    # post_process_distribution: std = softplus + 1e-4
        "mod_sigmoid": "2*sigmoid(x)^2.3 + 1e-7",
        "encoder_conv_groups": enc_groups,
    }
    return tensors, cfg, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ts_path", help="input exported RAVE .ts file")
    ap.add_argument("out_dir", help="output directory")
    ap.add_argument("--parity", action="store_true",
                    help="also dump a reference encode/decode fixture for the C++ test")
    args = ap.parse_args()

    if not os.path.isfile(args.ts_path):
        sys.exit(f"no such file: {args.ts_path}")
    os.makedirs(args.out_dir, exist_ok=True)

    tensors, cfg, m = extract(args.ts_path)

    st_path = os.path.join(args.out_dir, "model.safetensors")
    save_file(tensors, st_path)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"wrote {st_path}  ({len(tensors)} tensors)")
    print(f"wrote {os.path.join(args.out_dir, 'config.json')}")
    print(json.dumps(cfg, indent=2))

    # Self-check: reload and confirm tensors survived the round trip bit-for-bit.
    from safetensors.torch import load_file
    back = load_file(st_path)
    assert len(back) == len(tensors), "tensor count mismatch after reload"
    for k in tensors:
        if not torch.equal(back[k], tensors[k]):
            sys.exit(f"FAIL: {k} differs after reload")
    print("self-check: safetensors round-trips bit-for-bit OK")

    if args.parity:
        dump_parity(m, cfg, args.out_dir)


def dump_parity(m, cfg, out_dir):
    """Dump a deterministic encode/decode fixture: the C++ port must reproduce
    these within tolerance from the converted safetensors."""
    import numpy as np
    torch.set_grad_enabled(False)   # streaming caches are in-place; autograd would reject them
    torch.manual_seed(0)
    ratio = 2048
    frames = 8
    x = (torch.randn(1, 1, ratio * frames) * 0.1).float()
    z = m.encode(x)
    # decode on a fresh model so its caches start zeroed (offline-equivalent).
    y = m.decode(z)

    pdir = os.path.join(out_dir, "parity")
    os.makedirs(pdir, exist_ok=True)
    np.save(os.path.join(pdir, "input.npy"), x.numpy())
    np.save(os.path.join(pdir, "latent.npy"), z.numpy())
    np.save(os.path.join(pdir, "output.npy"), y.numpy())
    meta = {
        "input_shape": list(x.shape),
        "latent_shape": list(z.shape),
        "output_shape": list(y.shape),
        "input_rms": float(x.pow(2).mean().sqrt()),
        "latent_mean": float(z.mean()), "latent_std": float(z.std()),
        "output_rms": float(y.pow(2).mean().sqrt()),
        "note": "deterministic encode (uses posterior mean); decode add_noise per config['use_noise']",
    }
    with open(os.path.join(pdir, "parity.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote parity fixture to {pdir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
