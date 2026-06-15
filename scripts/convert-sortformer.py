#!/usr/bin/env python3
"""Convert an NVIDIA Streaming Sortformer `.nemo` checkpoint for brosoundml.

The diar_streaming_sortformer repos ship a single native NeMo archive:

  diar_streaming_sortformer_<size>.nemo   (a tar of)
    model_config.yaml      hyperparameters
    model_weights.ckpt     a torch state_dict

brosoundml::Sortformer loads the same `config.json` + `model.safetensors` pair
every other model uses, so this script:

  1. Untars the `.nemo`.
  2. Parses model_config.yaml into brosoundml's config.json schema.
  3. Remaps the NeMo state-dict key names onto brosoundml's scheme:
       - the FastConformer encoder onto the shared HF-Parakeet encoder names
         (so the C++ FastConformerEncoder loader serves both models), and
       - the Sortformer transformer + output heads onto a flat `transformer.*`
         / `sortformer.*` scheme.
  4. Casts every tensor to FP32 and writes model.safetensors.

The preprocessor mel filterbank + window are rebuilt closed-form in C++ (the
Parakeet-validated Slaney recipe), so the checkpoint's `preprocessor.*` tensors
are dropped — but their shapes are sanity-checked against the config.

Usage:
  scripts/convert-sortformer.py [--src DIR] [--out DIR] [--dump-keys]

  --src DIR     directory holding the .nemo (default: weights/sortformer/4spk-v2.1)
  --out DIR     where to write config.json + model.safetensors (default: --src)
  --dump-keys   print every source (name, shape, dtype) and exit
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file


# ── key remapping ────────────────────────────────────────────────────────────

# NeMo self_attn linear names -> brosoundml FastConformer (HF Parakeet) names.
_ATTN = {
    "linear_q": "q_proj",
    "linear_k": "k_proj",
    "linear_v": "v_proj",
    "linear_out": "o_proj",
    "linear_pos": "relative_k_proj",
}


def remap_encoder_key(k: str):
    """Map an `encoder.*` NeMo key to the FastConformer loader's name (or None
    to drop it)."""
    # subsampling (pre-encode) conv stack + output projection
    if k.startswith("encoder.pre_encode.conv."):
        # encoder.pre_encode.conv.{i}.{weight,bias}
        return "encoder.subsampling.layers." + k[len("encoder.pre_encode.conv."):]
    if k.startswith("encoder.pre_encode.out."):
        return "encoder.subsampling.linear." + k[len("encoder.pre_encode.out."):]

    if k.startswith("encoder.layers."):
        rest = k[len("encoder.layers."):]
        idx, _, tail = rest.partition(".")
        p = f"encoder.layers.{idx}."
        # self-attention
        if tail.startswith("self_attn."):
            sub = tail[len("self_attn."):]
            if sub == "pos_bias_u":
                return p + "self_attn.bias_u"
            if sub == "pos_bias_v":
                return p + "self_attn.bias_v"
            for nemo_name, hf_name in _ATTN.items():
                if sub.startswith(nemo_name + "."):
                    return p + "self_attn." + hf_name + "." + sub[len(nemo_name) + 1:]
            return p + "self_attn." + sub
        # conv module: batch_norm -> norm; drop num_batches_tracked
        if tail.startswith("conv."):
            sub = tail[len("conv."):]
            if sub.startswith("batch_norm."):
                bn = sub[len("batch_norm."):]
                if bn == "num_batches_tracked":
                    return None
                return p + "conv.norm." + bn
            return p + "conv." + sub
        # norms + feed-forwards pass through with identical names
        return p + tail

    return None  # any other encoder.* key is unexpected


def remap_head_key(k: str):
    """Map a `transformer_encoder.*` or `sortformer_modules.*` NeMo key to the
    brosoundml head scheme (or None to drop it)."""
    if k.startswith("transformer_encoder.layers."):
        rest = k[len("transformer_encoder.layers."):]
        idx, _, tail = rest.partition(".")
        p = f"transformer.layers.{idx}."
        if tail.startswith("layer_norm_1."):
            return p + "norm1." + tail[len("layer_norm_1."):]
        if tail.startswith("layer_norm_2."):
            return p + "norm2." + tail[len("layer_norm_2."):]
        if tail.startswith("first_sub_layer."):
            sub = tail[len("first_sub_layer."):]
            m = {
                "query_net": "attn.q",
                "key_net": "attn.k",
                "value_net": "attn.v",
                "out_projection": "attn.o",
            }
            for nemo_name, hf_name in m.items():
                if sub.startswith(nemo_name + "."):
                    return p + hf_name + "." + sub[len(nemo_name) + 1:]
        if tail.startswith("second_sub_layer."):
            sub = tail[len("second_sub_layer."):]
            if sub.startswith("dense_in."):
                return p + "ff.in." + sub[len("dense_in."):]
            if sub.startswith("dense_out."):
                return p + "ff.out." + sub[len("dense_out."):]
        return None

    if k.startswith("sortformer_modules."):
        sub = k[len("sortformer_modules."):]
        # The inference path uses encoder_proj + the two-layer sigmoid head.
        # hidden_to_spks (2*hidden -> n_spk) is unused at inference; drop it.
        keep = {"encoder_proj", "first_hidden_to_hidden", "single_hidden_to_spks"}
        head = sub.split(".")[0]
        if head in keep:
            return "sortformer." + sub
        return None

    return None


# ── config.json from model_config.yaml ───────────────────────────────────────

def build_config(cfg: dict) -> dict:
    enc = cfg["encoder"]
    tf = cfg["transformer_encoder"]
    sm = cfg["sortformer_modules"]
    pp = cfg["preprocessor"]

    ff_exp = enc.get("ff_expansion_factor", 4)
    d_model = enc["d_model"]

    out = {
        "model_type": "sortformer_diar",
        "sample_rate": cfg.get("sample_rate", 16000),
        "num_spks": cfg.get("max_num_of_spks", sm.get("num_spks", 4)),
        "fc_d_model": cfg["model_defaults"]["fc_d_model"],
        "tf_d_model": cfg["model_defaults"]["tf_d_model"],
        # FastConformer encoder (shared loader schema)
        "encoder_config": {
            "num_mel_bins": pp["features"],
            "hidden_size": d_model,
            "num_hidden_layers": enc["n_layers"],
            "num_attention_heads": enc["n_heads"],
            "intermediate_size": d_model * ff_exp,
            "conv_kernel_size": enc["conv_kernel_size"],
            "subsampling_factor": enc["subsampling_factor"],
            "subsampling_conv_channels": enc["subsampling_conv_channels"],
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "max_position_embeddings": enc.get("pos_emb_max_len", 5000),
            "scale_input": bool(enc.get("xscaling", False)),
            "attention_bias": True,
            "convolution_bias": True,
            # NeMo preprocessor normalize: "NA"/None => no per-feature norm.
            "normalize_features": str(pp.get("normalize", "NA")).lower()
            not in ("na", "none", ""),
        },
        # Sortformer transformer head
        "transformer_config": {
            "num_layers": tf["num_layers"],
            "hidden_size": tf["hidden_size"],
            "inner_size": tf["inner_size"],
            "num_attention_heads": tf["num_attention_heads"],
            "hidden_act": tf.get("hidden_act", "relu"),
            "pre_ln": bool(tf.get("pre_ln", False)),
        },
        # Streaming (AOSC) parameters
        "streaming_config": {
            "spkcache_len": sm["spkcache_len"],
            "fifo_len": sm["fifo_len"],
            "chunk_len": sm["chunk_len"],
            "spkcache_update_period": sm["spkcache_update_period"],
            "chunk_left_context": sm["chunk_left_context"],
            "chunk_right_context": sm["chunk_right_context"],
            "spkcache_sil_frames_per_spk": sm["spkcache_sil_frames_per_spk"],
            "pred_score_threshold": sm["pred_score_threshold"],
            "scores_boost_latest": sm["scores_boost_latest"],
            "sil_threshold": sm["sil_threshold"],
            "strong_boost_rate": sm["strong_boost_rate"],
            "weak_boost_rate": sm["weak_boost_rate"],
            "min_pos_scores_rate": sm["min_pos_scores_rate"],
        },
    }
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_src = repo_root / "weights" / "sortformer" / "4spk-v2.1"

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=default_src)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump-keys", action="store_true")
    args = ap.parse_args()

    src: Path = args.src
    out: Path = args.out or src
    if not src.exists():
        print(f"error: src {src} does not exist — run download-sortformer.sh first",
              file=sys.stderr)
        return 2

    nemo_files = sorted(glob.glob(str(src / "*.nemo")))
    if not nemo_files:
        print(f"error: no .nemo archive under {src}", file=sys.stderr)
        return 2
    nemo_path = nemo_files[0]
    print(f"source: {nemo_path}")

    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(nemo_path) as tar:
            tar.extractall(td)
        # .nemo may nest files under a subdir; find them.
        yaml_path = next(Path(td).rglob("*model_config.yaml"), None) \
            or next(Path(td).rglob("*.yaml"))
        ckpt_path = next(Path(td).rglob("*model_weights.ckpt"), None) \
            or next(Path(td).rglob("*.ckpt"))
        cfg = yaml.safe_load(Path(yaml_path).read_text())
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if not isinstance(sd, dict):
            sd = sd.state_dict()

    if args.dump_keys:
        for k in sorted(sd):
            print(f"  {k:60s} {tuple(sd[k].shape)} {str(sd[k].dtype).replace('torch.','')}")
        print(f"\n{len(sd)} tensors")
        return 0

    config = build_config(cfg)

    # ── remap weights ────────────────────────────────────────────────────────
    tensors: dict[str, torch.Tensor] = {}
    dropped, unmapped = [], []
    for k, v in sd.items():
        if k.startswith("encoder."):
            nk = remap_encoder_key(k)
        elif k.startswith("transformer_encoder.") or k.startswith("sortformer_modules."):
            nk = remap_head_key(k)
        elif k.startswith("preprocessor."):
            nk = None  # rebuilt closed-form in C++
        else:
            nk = None
            unmapped.append(k)
        if nk is None:
            dropped.append(k)
            continue
        t = v.detach().to(dtype=torch.float32, device="cpu")
        # pos_bias_u/v are (n_heads, head_dim) -> flatten to (d_model,).
        if nk.endswith("self_attn.bias_u") or nk.endswith("self_attn.bias_v"):
            t = t.reshape(-1)
        tensors[nk] = t.contiguous()

    if unmapped:
        print(f"warning: {len(unmapped)} unexpected top-level keys (dropped):",
              file=sys.stderr)
        for k in unmapped[:8]:
            print(f"  ? {k}", file=sys.stderr)

    # ── sanity checks against config ──────────────────────────────────────────
    ec = config["encoder_config"]
    nl, tnl = ec["num_hidden_layers"], config["transformer_config"]["num_layers"]
    enc_layer_keys = [k for k in tensors if k.startswith("encoder.layers.")]
    tf_layer_keys = [k for k in tensors if k.startswith("transformer.layers.")]
    max_enc = max((int(k.split(".")[2]) for k in enc_layer_keys), default=-1) + 1
    max_tf = max((int(k.split(".")[2]) for k in tf_layer_keys), default=-1) + 1
    assert max_enc == nl, f"encoder layers {max_enc} != config {nl}"
    assert max_tf == tnl, f"transformer layers {max_tf} != config {tnl}"
    for req in ("sortformer.encoder_proj.weight",
                "sortformer.first_hidden_to_hidden.weight",
                "sortformer.single_hidden_to_spks.weight",
                "encoder.subsampling.linear.weight"):
        assert req in tensors, f"missing expected tensor {req}"

    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.json"
    st_path = out / "model.safetensors"
    cfg_path.write_text(json.dumps(config, indent=2))
    save_file(tensors, str(st_path))

    print(f"\nwrote {cfg_path}")
    print(f"wrote {st_path}  ({len(tensors)} tensors, "
          f"{st_path.stat().st_size:,} bytes; dropped {len(dropped)})")
    print(f"  encoder: {nl} layers d_model={ec['hidden_size']} "
          f"heads={ec['num_attention_heads']} xscale={ec['scale_input']} "
          f"normalize={ec['normalize_features']}")
    print(f"  head: {tnl} transformer layers hidden={config['transformer_config']['hidden_size']} "
          f"spks={config['num_spks']}")
    print("\nDone. Ready for brosoundml::Sortformer::load().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
