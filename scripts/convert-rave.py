#!/usr/bin/env python3
"""Convert an exported RAVE TorchScript model (.ts) into the safetensors +
config.json layout brosoundml's RAVE loader reads.

RAVE checkpoints ship as *streaming* TorchScript exports (the `cached_conv`
causal variant used by nn~/the VST). They are not plain state dicts, so this
script loads the scripted module, extracts the encoder/decoder/pqmf/latent
weights, drops streaming-only buffers and the generative prior, and writes
`<out>/model.safetensors` + `<out>/config.json`.

Two export generations are handled:

  * LEGACY ("rave"): the old `_rave.*`-wrapped export (e.g. magnets). Plain
    conv + BatchNorm + LeakyReLU encoder. config.json carries only scalars; the
    C++ port uses its fixed RAVE-v2 topology. (Unchanged from the original.)

  * FLAT: the newer top-level export (encoder/decoder/pqmf/latent_pca at the
    root). These span several architectures (residual encoders, Snake
    activations, gimbal latent affine, variable downsample stages), so config
    additionally carries an explicit TOPOLOGY op-list — a flattened execution
    plan (conv/convT/leaky/snake/push/add) walked straight off the scripted
    module graph — that the reference and C++ interpreters replay. Strides /
    dilations come from each submodule's TorchScript `.code`; the causal
    left-pad is precomputed (newer cached convs use dil*(k-1)-(stride-1)).

Usage:
    python convert-rave.py model.ts  out/dir
    python convert-rave.py model.ts  out/dir --parity   # also dump reference I/O

Requires: torch, safetensors, numpy.
"""

import argparse
import json
import os
import re
import sys

import torch
from safetensors.torch import save_file

# Streaming-only / non-weight buffers the offline forward never reads.
_DROP_SUFFIXES = (
    ".cache.pad",
    ".downsampling_delay.pad",
    ".num_batches_tracked",
)


def _scalar(mod, name, default=None):
    try:
        v = getattr(mod, name)
    except Exception:
        return default
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.tolist()
    return v


# ─────────────────────────── topology walker (FLAT) ──────────────────────────

def _conv_stride_dil(mod):
    """Parse (stride, dilation) from a CachedConv1d/ConvTranspose1d's .code."""
    c = mod.code
    m = re.search(r'torch\.conv(?:_transpose)?1d\(([^)]*)\)', c)
    lists = re.findall(r'\[([\d, ]*)\]', m.group(1)) if m else []
    stride = int(lists[0].split(',')[0]) if lists else 1
    if 'conv_transpose1d' in c:
        return stride, 1
    dil = int(lists[2].split(',')[0]) if len(lists) > 2 else 1
    return stride, dil


def _cached_left(mod):
    """Exact causal left-pad of a CachedConv1d, read from its cache modules
    (authoritative — no formula reproduces every conv: e.g. PQMF k257/s16 stores
    256 while an encoder k5/s2 downsampler stores 3)."""
    left = 0
    cache = getattr(mod, 'cache', None)
    if cache is not None:
        left += int(getattr(cache, 'padding', 0))
    dsd = getattr(mod, 'downsampling_delay', None)
    if dsd is not None:
        left += int(getattr(dsd, 'padding', 0))
    return left


def _convt_pad(mod):
    """conv_transpose1d padding (2nd arg) from a ConvTranspose1d's .code; the
    offline-causal convT is conv_transpose1d(stride, padding) with output_pad 0."""
    m = re.search(r'torch\.conv_transpose1d\(([^)]*)\)', mod.code)
    lists = re.findall(r'\[([\d, ]*)\]', m.group(1)) if m else []
    return int(lists[1].split(',')[0]) if len(lists) > 1 else 0


class TopoWalker:
    """Flatten a scripted submodule tree into an op-list. Residual blocks become
    push/…body…/add. Weight keys are emitted with the `_rave.` prefix stripped
    (matching the saved safetensors)."""

    def __init__(self, names, prefix_strip):
        self.names = names            # id(module) -> dotted name
        self.strip = prefix_strip     # "_rave." or ""

    def key(self, mod, suffix):
        n = self.names[id(mod)]
        if self.strip and n.startswith(self.strip):
            n = n[len(self.strip):]
        return n + suffix

    def _emit_convt(self, mod, ops, cpad):
        stride, _ = _conv_stride_dil(mod)
        k = int(mod.weight.shape[-1])
        ops.append({"kind": "convT", "w": self.key(mod, ".weight"),
                    "b": self.key(mod, ".bias") if hasattr(mod, "bias") and mod.bias is not None else None,
                    "k": k, "stride": int(stride), "pad": int(_convt_pad(mod)), "cpad": int(cpad)})

    def emit(self, mod, ops, fmt):
        on = mod.original_name
        if on in ('CachedConv1d', 'Conv1d'):
            stride, dil = _conv_stride_dil(mod)
            k = mod.weight.shape[-1]
            left = _cached_left(mod) if hasattr(mod, 'cache') else dil * (k - 1)
            ops.append({"kind": "conv", "w": self.key(mod, ".weight"),
                        "b": self.key(mod, ".bias") if hasattr(mod, "bias") and mod.bias is not None else None,
                        "k": int(k), "stride": int(stride), "dil": int(dil), "left_pad": int(left)})
        elif on in ('ConvTranspose1d', 'CachedConvTranspose1d'):
            # A bare convT (no preceding CachedPadding, e.g. amp_mod decoders).
            self._emit_convt(mod, ops, cpad=0)
        elif on == 'UpsampleLayer':
            # [CachedPadding(cpad), ConvTranspose1d]. Fold the cpad into the convT
            # op (output trimmed to L*stride either way).
            kids = list((mod.net if hasattr(mod, 'net') else mod).named_children())
            cp = kids[0][1] if kids and kids[0][1].original_name == 'CachedPadding1d' else None
            ct = next(c for _, c in kids if c.original_name in ('ConvTranspose1d', 'CachedConvTranspose1d'))
            self._emit_convt(ct, ops, cpad=int(getattr(cp, 'padding', 0)) if cp is not None else 0)
        elif on == 'LeakyReLU':
            ops.append({"kind": "leaky"})
        elif on == 'Snake':
            ops.append({"kind": "snake", "alpha": self.key(mod, ".alpha")})
        elif on == 'CachedPadding1d':
            # A standalone left-pad. padding 0 (residual-branch align delays) is a
            # true offline no-op; >0 (before an UpsampleLayer convT) is structural.
            p = int(getattr(mod, 'padding', 0))
            if p > 0:
                ops.append({"kind": "lpad", "n": p})
        elif on in ('Identity', 'AdaptiveInstanceNormalization'):
            pass  # offline no-ops (fake adain has empty stats)
        elif on == 'Residual':
            ops.append({"kind": "push"})
            branch0 = dict(mod.aligned.branches.named_children())['0']
            self.emit(branch0, ops, fmt)
            ops.append({"kind": "add"})
        else:
            sub = getattr(mod, 'net', None)
            if sub is None:
                sub = getattr(mod, 'encoder', None)
            if sub is None:
                sub = mod
            for _, child in sub.named_children():
                self.emit(child, ops, fmt)
        return ops


def _alpha_keys_present(tensors):
    return any(k.endswith(".alpha") for k in tensors)


def extract(ts_path, sr_override=None):
    m = torch.jit.load(ts_path, map_location="cpu")
    m.eval()
    legacy = hasattr(m, "_rave")
    rave = m._rave if legacy else m
    strip = "_rave." if legacy else ""

    # Weights: keep encoder/decoder/pqmf/latent_*/gimbal; drop prior, streaming
    # buffers, params blobs. (For FLAT we additionally need gimbal + snake alpha.)
    sd = m.state_dict()
    keep_prefixes = ("encoder.", "decoder.", "pqmf.", "gimbal.", "prior_net.")
    keep_exact = ("latent_pca", "latent_mean")
    tensors = {}
    for k, v in sd.items():
        name = k[len(strip):] if (strip and k.startswith(strip)) else k
        if strip and not k.startswith(strip):
            continue  # legacy: only _rave.*
        if any(name.endswith(s) for s in _DROP_SUFFIXES):
            continue
        if name in keep_exact or any(name.startswith(p) for p in keep_prefixes):
            tensors[name] = v.contiguous().float()

    full = int(tensors["latent_pca"].shape[0])
    enc_params = _scalar(rave, "encode_params") or _scalar(m, "encode_params")
    crop = int(enc_params[2]) if enc_params else int(_scalar(rave, "cropped_latent_size", 0))
    ratio = int(enc_params[3]) if enc_params else 0
    n_band = int(tensors["pqmf.hk"].shape[0]) if "pqmf.hk" in tensors else \
        int(_scalar(rave, "n_band", 16))

    # Output sampling rate. Models with a `resample` module carry a sampling_rate
    # attribute (the resampler's target). Models without one run at their native
    # rate, which they DON'T store — recover it from the `_r<NNNNN>` filename token
    # (or an explicit --sampling-rate). Mis-tagging a 44.1 kHz model as 48 kHz
    # plays it ~8.8 % sharp, so this matters.
    sr = sr_override if sr_override else _scalar(rave, "sampling_rate", None)
    if sr is None:
        sr = _scalar(m, "sampling_rate", None)
    if sr is None:
        mt = re.search(r"_r(\d{4,6})", os.path.basename(ts_path))
        sr = int(mt.group(1)) if mt else 48000
        print(f"  (no sampling_rate attr; using {sr} "
              f"{'from filename' if mt else 'as default'})")

    cfg = {
        "model": "rave_v2",
        "format": "rave" if legacy else "flat",
        "sampling_rate": int(sr),
        "full_latent_size": full,
        "cropped_latent_size": crop,
        "n_band": n_band,
        "leaky_slope": 0.2,
        "bn_eps": 1.0e-5,
    }
    if ratio:
        cfg["total_ratio"] = ratio

    if legacy:
        # Unchanged legacy metadata; C++ uses its fixed topology.
        import re as _re
        enc_groups = {}
        for name, sub in m.named_modules():
            if not name.startswith("_rave.encoder.net."):
                continue
            mobj = _re.search(r"torch\.conv1d\([^)]*?,\s*(\d+)\s*\)", str(getattr(sub, "code", "")))
            if mobj:
                enc_groups[name[len("_rave.encoder.net."):]] = int(mobj.group(1))
        cfg.update({
            "deterministic": bool(_scalar(rave, "deterministic", True)),
            "trained_cropped": bool(_scalar(rave, "trained_cropped", False)),
            "stereo": bool(_scalar(rave, "stereo", False)),
            "use_noise": bool(_scalar(rave.decoder, "use_noise", True)),
            "loud_stride": int(_scalar(rave.decoder, "loud_stride", 1)),
            "softplus_floor": 1.0e-4,
            "mod_sigmoid": "2*sigmoid(x)^2.3 + 1e-7",
            "encoder_conv_groups": enc_groups,
        })
        return tensors, cfg, m

    # ── FLAT: emit topology ──
    names = {id(mod): name for name, mod in m.named_modules()}
    mods = dict(m.named_modules())
    walker = TopoWalker(names, strip)

    enc_mod = mods["encoder"]
    cfg["encoder_ops"] = walker.emit(enc_mod, [], "flat")
    dec_net = mods["decoder.net"]
    cfg["decoder_ops"] = walker.emit(dec_net, [], "flat")

    # PQMF (forward_conv stride n_band then reverse_half; inverse_conv stride 1) +
    # PCA + gimbal. forward/inverse are CachedConv1d modules, so reuse conv_meta.
    def conv_meta0(name):
        sub = mods[name]; stride, dil = _conv_stride_dil(sub); k = int(sub.weight.shape[-1])
        return {"w": name + ".weight",
                "b": (name + ".bias") if (name + ".bias") in tensors else None,
                "k": k, "stride": int(stride), "dil": int(dil),
                "left_pad": int(_cached_left(sub)) if hasattr(sub, 'cache') else int(dil * (k - 1))}
    cfg["pqmf"] = {"forward": conv_meta0("pqmf.forward_conv"),
                   "inverse": conv_meta0("pqmf.inverse_conv")}
    cfg["has_gimbal"] = "gimbal.log_a" in tensors
    if cfg["has_gimbal"]:
        cfg["gimbal"] = {"log_a": "gimbal.log_a", "b": "gimbal.b"}

    # ── latent-completion scheme (how the discarded dims are filled at decode) ──
    # Family A (kld_idxs buffer present): decode runs an autoregressive `prior_net`
    # frame-by-frame to generate ALL latent dims, keeps the encoded crop, takes the
    # prior's output for the rest, then either scatters back via argsort(kld_idxs)
    # (use_pca=False) or inverse-PCAs (use_pca=True), and finally inverts the gimbal
    # before the decoder. Encode crops by kld-gather (use_pca=False) or PCA
    # (use_pca=True). Family B (no kld_idxs): decode is `pre_process_latent` — fill
    # the discarded dims with N(0,1), inverse-PCA, decode. No prior, no gimbal.
    def _get_attr(name):
        for src in (rave, m):
            try:
                return getattr(src, name)
            except Exception:
                pass
        return None
    kld = _get_attr("kld_idxs")
    use_pca = bool(_scalar(rave, "use_pca", _scalar(m, "use_pca", False)))
    if kld is not None:
        cfg["kld_idxs"] = [int(v) for v in kld.tolist()]
        cfg["use_pca"] = use_pca
        cfg["latent_mode"] = "prior_pca" if use_pca else "prior_kld"
        # Autoregressive prior network: input = previous full latent frame, output
        # = 2*full (mean | log-scale). Walked into the same op-list the encoder/
        # decoder use; the interpreter chunks the output and runs it windowed over
        # the prior's receptive field, frame by frame.
        cfg["prior_ops"] = walker.emit(mods["prior_net"], [], "flat")
    else:
        cfg["latent_mode"] = "noise_pca"
        cfg["use_pca"] = True

    # Two synthesis types:
    #  * "rave"   (A/B): decoder.net -> h; wave=tanh(conv0(h)), loud=mod_sigmoid(
    #               conv1(h)), noise=noise_branch(conv2(h)); out = wave*loud[+noise].
    #  * "amp_mod" (C):  decoder.net ends at 2*n_band; out = tanh(wave*sigmoid(amp))
    #               where (wave, amp) = split(h, n_band). No loudness/noise branch.
    def conv_meta(name):
        sub = mods[name]
        stride, dil = _conv_stride_dil(sub)
        k = int(sub.weight.shape[-1])
        return {"w": name + ".weight",
                "b": (name + ".bias") if (name + ".bias") in tensors else None,
                "k": k, "stride": int(stride), "dil": int(dil),
                "left_pad": int(_cached_left(sub)) if hasattr(sub, 'cache') else int(dil * (k - 1))}

    if "decoder.synth.branches.0.weight" in tensors:
        cfg["synth_type"] = "rave"
        cfg["synth"] = {"wave": conv_meta("decoder.synth.branches.0"),
                        "loud": conv_meta("decoder.synth.branches.1"),
                        "loud_stride": int(_scalar(mods["decoder"], "loud_stride", 1))}
        has_noise = "decoder.synth.branches.2.net.0.weight" in tensors
        cfg["has_noise"] = has_noise
        if has_noise:
            cfg["synth"]["noise"] = [conv_meta("decoder.synth.branches.2.net.0"),
                                     conv_meta("decoder.synth.branches.2.net.2"),
                                     conv_meta("decoder.synth.branches.2.net.4")]
            cfg["noise_bands"] = int(tensors["decoder.synth.branches.2.net.4.weight"].shape[0] // n_band)
    else:
        cfg["synth_type"] = "amp_mod"
        cfg["has_noise"] = False
        cfg["amplitude_modulation"] = bool(_scalar(mods["decoder"], "amplitude_modulation", True))

    cfg["has_snake"] = _alpha_keys_present(tensors)
    return tensors, cfg, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ts_path")
    ap.add_argument("out_dir")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--sampling-rate", type=int, default=None,
                    help="override output sampling rate (for models that store none "
                         "and have no _r<rate> filename token, e.g. crozzoli)")
    args = ap.parse_args()

    if not os.path.isfile(args.ts_path):
        sys.exit(f"no such file: {args.ts_path}")
    os.makedirs(args.out_dir, exist_ok=True)

    tensors, cfg, m = extract(args.ts_path, sr_override=args.sampling_rate)

    st_path = os.path.join(args.out_dir, "model.safetensors")
    save_file(tensors, st_path)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"wrote {st_path}  ({len(tensors)} tensors)")
    print(f"wrote {os.path.join(args.out_dir, 'config.json')}  format={cfg['format']} "
          f"z={cfg['cropped_latent_size']} sr={cfg['sampling_rate']} "
          f"snake={cfg.get('has_snake')} gimbal={cfg.get('has_gimbal')} noise={cfg.get('has_noise')} "
          f"latent_mode={cfg.get('latent_mode')}")

    from safetensors.torch import load_file
    back = load_file(st_path)
    assert len(back) == len(tensors), "tensor count mismatch after reload"
    for k in tensors:
        if not torch.equal(back[k], tensors[k]):
            sys.exit(f"FAIL: {k} differs after reload")
    print("self-check: safetensors round-trips bit-for-bit OK")


if __name__ == "__main__":
    main()
