"""Convert Demucs v4 htdemucs_ft (MIT code and weights) to brosoundml's layout.

htdemucs_ft is a bag of four HTDemucs networks, each fine-tuned for one stem
(drums, bass, other, vocals) with an identity weight matrix, so stem k is model
k's output k. This writes, under <out>:

    config.json                 sources, sample rate, segment, the per-model
                                file + stem, and the HTDemucs kwargs
    <stem>.safetensors          that stem's network, FP32, torch state-dict names

Needs the demucs package (pip install demucs==4.0.1); the checkpoints come from
torch hub's cache (~/.cache/torch/hub/checkpoints) like `demucs` itself.

    python tools/htdemucs_convert.py weights/htdemucs_ft
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from demucs.pretrained import get_model
    from safetensors.torch import save_file

    bag = get_model("htdemucs_ft")
    models = []
    for i, m in enumerate(bag.models):
        w = bag.weights[i]
        stems = [s for s, v in zip(bag.sources, w) if v != 0]
        if len(stems) != 1 or w[bag.sources.index(stems[0])] != 1:
            raise SystemExit(f"model {i}: expected an identity bag weight, got {w}")
        stem = stems[0]
        kwargs = dict(m._init_args_kwargs[1])
        for k, v in list(kwargs.items()):
            if isinstance(v, Fraction):
                kwargs[k] = float(v)
        state = {k: v.detach().to(torch.float32).contiguous() for k, v in m.state_dict().items()}
        fname = f"{stem}.safetensors"
        save_file(state, str(out / fname))
        models.append({"stem": stem, "file": fname, "kwargs": kwargs})
        print(f"{fname}: {sum(v.numel() for v in state.values())} params")

    cfg = {
        "name": "htdemucs_ft",
        "sources": list(bag.sources),
        "samplerate": int(bag.samplerate),
        "audio_channels": int(bag.audio_channels),
        "segment": float(bag.models[0].segment),
        "overlap": 0.25,
        "models": models,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"wrote {out / 'config.json'}")


if __name__ == "__main__":
    main()
