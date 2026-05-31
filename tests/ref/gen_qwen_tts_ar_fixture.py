#!/usr/bin/env python3
"""Regenerate the Qwen3-TTS stage-4 fixture: Code Predictor + AR generation loop.

Validates the C++ Code Predictor (src/qwen_tts_code_predictor.cpp) and the
dual-track AR loop (src/qwen_tts_generate.cpp) against the *genuine upstream*
Qwen3TTSTalkerForConditionalGeneration — which wires the 28-layer Talker, the
codec_head, the text_projection, and the 5-layer Code Predictor together with
the per-frame generate loop. It loads the real talker.* weights and runs:

  (A) a standalone Code Predictor expansion: code_predictor.generate over a
      controlled (past_hidden, talker-embedded codebook-0) prefill, do_sample
      False, dumping the 15 produced codebooks; and
  (B) a few frames of the full AR loop: talker.generate over a controlled
      prefill embedding stream + all-ones attention mask + trailing-text /
      tts_pad embeddings, do_sample False, dumping the emitted [c0..c15] frames.

The all-ones mask makes the M-RoPE positions a plain 0..T-1 ramp (identical on
all three axes), so this fixture exercises the loop wiring — KV cache, Code
Predictor, next-frame embedding sum, position advance, stop — while the 3-axis
get_rope_index assembly is stage 5 (and the 3-axis M-RoPE itself is already
covered by gen_talker_fixture.py).

Requirements (same as gen_talker_fixture.py): the weights, and a
transformers==4.57.3 venv (with system torch). Network on first run (downloads
+ patches the upstream .py files into tests/ref/_cache/).

Binary layout (little-endian):
    # Part A — standalone Code Predictor
    int32  H, n_codes(=15), c0
    f32    past_hidden[H]
    int32  cp_codes[n_codes]                  # expected codebooks 1..15
    # Part B — AR loop
    int32  T, L, F, num_code_groups(=16), eos_id, rope_delta
    f32    prefill_embeds[T*H]
    int32  pos3[3*T]                          # axis-major prefill positions
    f32    trailing_text_hidden[L*H]
    f32    tts_pad_embed[H]
    int32  frames[F*num_code_groups]          # [c0..c15] per frame
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
# Same patches as gen_talker_fixture.py — neutralize librosa / inference-side
# tokenizer / relative config imports the decoder forward does not need.
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
    from safetensors.torch import load_file
    import talker_configuration_qwen3_tts as C
    import talker_modeling_qwen3_tts as M

    cfg = C.Qwen3TTSTalkerConfig(**json.load(open(os.path.join(WEIGHTS, "config.json")))["talker_config"])
    cfg._attn_implementation = "eager"
    cfg.code_predictor_config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = M.Qwen3TTSTalkerForConditionalGeneration(cfg).eval()

    full = load_file(os.path.join(WEIGHTS, "model.safetensors"))
    sd = {k[len("talker."):]: v.float() for k, v in full.items() if k.startswith("talker.")}
    miss, unexp = model.load_state_dict(sd, strict=False)
    miss = [m for m in miss if not m.endswith("inv_freq")]
    assert not miss and not unexp, f"missing {miss[:5]} unexpected {unexp[:5]}"
    print("loaded full talker.* (model + text_projection + codec_head + code_predictor)")

    H = cfg.hidden_size
    G = cfg.num_code_groups
    n_codes = G - 1
    eos_id = cfg.codec_eos_token_id

    # ── Part A: standalone Code Predictor ───────────────────────────────────
    gA = torch.Generator().manual_seed(20)
    past_hidden = torch.randn(1, 1, H, generator=gA)
    c0 = 137
    c0_embed = model.get_input_embeddings()(torch.tensor([[c0]]))  # talker codec_embedding[c0]
    with torch.no_grad():
        cp_out = model.code_predictor.generate(
            inputs_embeds=torch.cat((past_hidden, c0_embed), dim=1),
            max_new_tokens=n_codes,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    cp_codes = cp_out.sequences[0].to(torch.int32).tolist()
    assert len(cp_codes) == n_codes, f"expected {n_codes} codes, got {len(cp_codes)}"
    print(f"Part A: code predictor -> {cp_codes}")

    # ── Part B: AR loop ─────────────────────────────────────────────────────
    T, L, max_new = 10, 4, 8
    gB = torch.Generator().manual_seed(21)
    prefill = torch.randn(1, T, H, generator=gB) * 0.5
    trailing = torch.randn(1, L, H, generator=gB) * 0.5
    tts_pad = torch.randn(1, 1, H, generator=gB) * 0.5
    attn = torch.ones(1, T, dtype=torch.long)

    pos, deltas = model.get_rope_index(attn)          # (3,1,T), (1,1)
    delta0 = (1 - attn).sum(dim=-1).unsqueeze(1)
    rope_delta = int((deltas - delta0).reshape(-1)[0].item())
    pos = pos.squeeze(1)                              # (3, T)
    assert rope_delta == 0, f"expected rope_delta 0 for all-ones mask, got {rope_delta}"

    with torch.no_grad():
        result = model.generate(
            inputs_embeds=prefill,
            attention_mask=attn,
            trailing_text_hidden=trailing,
            tts_pad_embed=tts_pad,
            max_new_tokens=max_new,
            do_sample=False,
            subtalker_dosample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    talker_codes = torch.stack([h[-1] for h in result.hidden_states if h[-1] is not None], dim=1)
    # (1, F, G); truncate at the first eos in codebook 0, like the upstream driver.
    first_cb = talker_codes[0, :, 0]
    is_stop = (first_cb == eos_id)
    F = int(torch.argmax(is_stop.int()).item()) if bool(is_stop.any()) else talker_codes.shape[1]
    frames = talker_codes[0, :F].to(torch.int32).contiguous()
    print(f"Part B: AR loop emitted F={F} frames (T={T} L={L} max_new={max_new} eos={eos_id})")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "qwen_tts_ar.bin")
    with open(path, "wb") as f:
        # Part A
        f.write(struct.pack("<iii", H, n_codes, c0))
        f.write(past_hidden.reshape(-1).contiguous().numpy().astype("<f4").tobytes())
        f.write(struct.pack(f"<{n_codes}i", *cp_codes))
        # Part B
        f.write(struct.pack("<iiiiii", T, L, F, G, eos_id, rope_delta))
        f.write(prefill.reshape(-1).contiguous().numpy().astype("<f4").tobytes())
        f.write(pos.to(torch.int32).contiguous().numpy().tobytes(order="C"))
        f.write(trailing.reshape(-1).contiguous().numpy().astype("<f4").tobytes())
        f.write(tts_pad.reshape(-1).contiguous().numpy().astype("<f4").tobytes())
        f.write(frames.numpy().tobytes(order="C"))
    print(f"wrote {path}: H={H} T={T} L={L} F={F} G={G}")


if __name__ == "__main__":
    main()
