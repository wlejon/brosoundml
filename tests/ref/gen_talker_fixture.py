#!/usr/bin/env python3
"""Regenerate the Qwen3-TTS Talker forward test fixture.

Validates the C++ Talker (src/qwen_tts_talker.cpp) against the *genuine
upstream* 28-layer Qwen3 decoder backbone in isolation: it instantiates the
upstream Qwen3TTSTalkerModel + text_projection + codec_head, loads the real
talker.* weights, and runs one prefill forward on controlled inputs_embeds +
3-axis M-RoPE positions, plus a few embedding lookups.

The fixture is gitignored (*.bin) — regenerate locally to enable the Talker
numeric check in tests/test_qwen_tts.cpp (it skips when absent).

Requirements (same as gen_qwen_tts_codec_fixture.py): the weights, and a
transformers==4.57.3 venv (with system torch). Network on first run (downloads
+ patches two upstream .py files into tests/ref/_cache/).

Binary layout (little-endian):
    int32  T, H(=1024), V(=3072), n_text, n_codec
    int32  pos[3*T]                  # M-RoPE position ids, axis-major pos[a*T+t]
    f32    inputs_embeds[T*H]
    f32    hidden[T*H]               # expected last_hidden_state (post final norm)
    f32    logits[T*V]               # expected codec_head(hidden)
    int32  text_ids[n_text]
    f32    text_proj[n_text*H]       # text_projection(text_embedding(id))
    int32  codec_ids[n_codec]
    f32    codec_emb[n_codec*H]      # codec_embedding(id)
"""
import json
import os
import struct
import sys
import urllib.request

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(os.path.dirname(__file__), "_cache")
WEIGHTS = os.path.join(REPO, "weights", "qwen-tts", "0.6B-customvoice")
OUT = os.path.join(REPO, "tests", "fixtures")

GH = "https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/qwen_tts/core/models"
# Upstream file -> (cache name, list of literal text substitutions). The Talker
# modeling file pulls in librosa + an inference-side tokenizer + a relative
# config import, none of which the decoder forward needs; neutralize them so the
# cache dir is importable as flat modules under stock transformers.
FILES = {
    "configuration_qwen3_tts.py": ("talker_configuration_qwen3_tts.py", []),
    "modeling_qwen3_tts.py": ("talker_modeling_qwen3_tts.py", [
        ("from .configuration_qwen3_tts import",
         "from talker_configuration_qwen3_tts import"),
        ("from ...inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer",
         "Qwen3TTSTokenizer = object"),
        ("from librosa.filters import mel as librosa_mel_fn",
         "librosa_mel_fn = None"),
    ]),
}


def ensure_upstream():
    os.makedirs(CACHE, exist_ok=True)
    for src, (name, subs) in FILES.items():
        dest = os.path.join(CACHE, name)
        if not os.path.exists(dest):
            url = f"{GH}/{src}"
            print(f"downloading {url}")
            data = urllib.request.urlopen(url).read().decode("utf-8")
            for a, b in subs:
                data = data.replace(a, b)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(data)
    sys.path.insert(0, CACHE)


def main():
    if not os.path.exists(os.path.join(WEIGHTS, "model.safetensors")):
        sys.exit(f"missing weights under {WEIGHTS} (run scripts/download-qwen-tts.sh)")
    ensure_upstream()

    import torch
    from torch import nn
    from safetensors.torch import load_file
    import talker_configuration_qwen3_tts as C
    import talker_modeling_qwen3_tts as M

    cfg = C.Qwen3TTSTalkerConfig(**json.load(open(os.path.join(WEIGHTS, "config.json")))["talker_config"])
    torch.manual_seed(0)
    model = M.Qwen3TTSTalkerModel(cfg).eval()
    text_projection = M.Qwen3TTSTalkerResizeMLP(
        cfg.text_hidden_size, cfg.text_hidden_size, cfg.hidden_size, cfg.hidden_act, bias=True).eval()
    codec_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).eval()

    full = load_file(os.path.join(WEIGHTS, "model.safetensors"))

    def load_into(mod, prefix):
        sd = {k[len(prefix):]: v.float() for k, v in full.items() if k.startswith(prefix)}
        miss, unexp = mod.load_state_dict(sd, strict=False)
        miss = [m for m in miss if not m.endswith("inv_freq")]
        assert not miss and not unexp, f"{prefix}: missing {miss[:5]} unexpected {unexp[:5]}"
        return len(sd)

    load_into(model, "talker.model.")
    load_into(text_projection, "talker.text_projection.")
    codec_head.weight.data.copy_(full["talker.codec_head.weight"].float())
    print("loaded talker.model + text_projection + codec_head")

    H, V, T = cfg.hidden_size, cfg.vocab_size, 24
    g = torch.Generator().manual_seed(7)
    inputs_embeds = torch.randn(1, T, H, generator=g) * 0.5
    pos = torch.zeros(3, 1, T, dtype=torch.long)
    pos[0, 0] = torch.arange(T)
    pos[1, 0] = 2 * torch.arange(T)
    pos[2, 0] = 3 * torch.arange(T)

    with torch.no_grad():
        out = model(inputs_embeds=inputs_embeds, position_ids=pos, use_cache=False)
        hidden = out.last_hidden_state
        logits = codec_head(hidden)
        n_text, n_codec = 6, 6
        gt = torch.Generator().manual_seed(11)
        text_ids = torch.randint(0, cfg.text_vocab_size, (n_text,), generator=gt)
        codec_ids = torch.randint(0, cfg.vocab_size, (n_codec,), generator=gt)
        text_proj = text_projection(model.get_text_embeddings()(text_ids))
        codec_emb = model.get_input_embeddings()(codec_ids)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "qwen_tts_talker.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<iiiii", T, H, V, n_text, n_codec))
        f.write(pos.squeeze(1).to(torch.int32).numpy().tobytes(order="C"))
        f.write(inputs_embeds.squeeze(0).contiguous().numpy().astype("<f4").tobytes())
        f.write(hidden.squeeze(0).contiguous().numpy().astype("<f4").tobytes())
        f.write(logits.squeeze(0).contiguous().numpy().astype("<f4").tobytes())
        f.write(text_ids.to(torch.int32).numpy().tobytes())
        f.write(text_proj.contiguous().numpy().astype("<f4").tobytes())
        f.write(codec_ids.to(torch.int32).numpy().tobytes())
        f.write(codec_emb.contiguous().numpy().astype("<f4").tobytes())
    print(f"wrote {path}: T={T} H={H} V={V}")


if __name__ == "__main__":
    main()
