#!/usr/bin/env python3
"""Regenerate the Qwen3-TTS stage-5 fixture: end-to-end CustomVoice synthesis.

Validates the C++ synthesize() path (src/qwen_tts.cpp + assemble_custom_voice_
prefill / generate_codes in src/qwen_tts_generate.cpp) against the *genuine
upstream* assembly + talker generation, deterministically (greedy):

  - the tokenizer: the C++ brolm Qwen BPE must reproduce the upstream Qwen2
    tokenizer's ids for the chat prompt;
  - the prefill assembly: the streaming CustomVoice prefill embedding stream +
    trailing-text embeddings + tts_pad embedding (this script mirrors the
    upstream generate() assembly exactly — the same logic the C++ implements);
  - the AR loop with the upstream codebook-0 logits policy: suppress the top
    1024 codec tokens (except EOS), min_new_tokens=2, repetition_penalty=1.05,
    do_sample=False / subtalker_dosample=False — the emitted [c0..c15] frames
    must match exactly.

The codec decode (codes -> 24 kHz) is stage 2 (already verified bit-exact), so
this fixture compares the *code stream*, not the waveform — keeping it fast (no
x1920 decode). Requirements: the weights + a transformers==4.57.3 venv. Network
on first run (downloads + patches the upstream .py into tests/ref/_cache/).

Binary layout (little-endian):
    int32  text_len;  bytes text[text_len]      # the user text
    int32  H, n_ids
    int32  input_ids[n_ids]                      # tokenized chat prompt (ref)
    int32  spk_id, language_id
    int32  T;  f32 prefill[T*H]
    int32  L;  f32 trailing[L*H]
    f32    tts_pad[H]
    int32  F, G;  int32 frames[F*G]
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

TEXT = "Hello there, my friend."
SPEAKER = "serena"
LANGUAGE = "english"
MAX_NEW = 24

GH = "https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/qwen_tts/core/models"
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
    from transformers import AutoTokenizer
    import talker_configuration_qwen3_tts as C
    import talker_modeling_qwen3_tts as M

    top = json.load(open(os.path.join(WEIGHTS, "config.json")))
    cfg = C.Qwen3TTSTalkerConfig(**top["talker_config"])
    cfg._attn_implementation = "eager"
    cfg.code_predictor_config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = M.Qwen3TTSTalkerForConditionalGeneration(cfg).eval()

    full = load_file(os.path.join(WEIGHTS, "model.safetensors"))
    sd = {k[len("talker."):]: v.float() for k, v in full.items() if k.startswith("talker.")}
    miss, unexp = model.load_state_dict(sd, strict=False)
    miss = [m for m in miss if not m.endswith("inv_freq")]
    assert not miss and not unexp, f"missing {miss[:5]} unexpected {unexp[:5]}"

    tok = AutoTokenizer.from_pretrained(WEIGHTS)
    prompt = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    input_id = torch.tensor(tok(prompt)["input_ids"], dtype=torch.long).unsqueeze(0)  # [1,L]
    print(f"input_ids ({input_id.shape[1]}): {input_id[0].tolist()}")

    H = cfg.hidden_size
    G = cfg.num_code_groups
    eos_id = cfg.codec_eos_token_id
    spk_id = cfg.spk_id[SPEAKER.lower()]
    language_id = cfg.codec_language_id[LANGUAGE.lower()]

    def TP(ids):  # text_projection(text_embedding(ids)); ids LongTensor [1,n]
        return model.text_projection(model.get_text_embeddings()(ids))

    def CE(ids):  # talker codec_embedding(ids); ids LongTensor [1,n]
        return model.get_input_embeddings()(ids)

    with torch.no_grad():
        tts_e = TP(torch.tensor([[top["tts_bos_token_id"],
                                  top["tts_eos_token_id"],
                                  top["tts_pad_token_id"]]]))
        tts_bos_e, tts_eos_e, tts_pad_e = tts_e.chunk(3, dim=1)  # 3 * [1,1,H]

        codec_prefill = [cfg.codec_think_id, cfg.codec_think_bos_id, language_id, cfg.codec_think_eos_id]
        ci0 = CE(torch.tensor([codec_prefill]))
        ci1 = CE(torch.tensor([[cfg.codec_pad_id, cfg.codec_bos_id]]))
        spk_e = CE(torch.tensor([spk_id]))                       # [1,H]
        codec_input = torch.cat([ci0, spk_e.view(1, 1, -1), ci1], dim=1)  # [1,7,H]

        role = TP(input_id[:, :3])                               # [1,3,H]
        text_side = torch.cat([tts_pad_e.expand(-1, codec_input.shape[1] - 2, -1),
                               tts_bos_e], dim=1)
        _tie = text_side + codec_input[:, :-1]                   # [1,6,H]
        prefill = torch.cat([role, _tie], dim=1)                 # [1,9,H]
        prefill = torch.cat([prefill,
                             TP(input_id[:, 3:4]) + codec_input[:, -1:]], dim=1)  # [1,10,H]
        trailing = torch.cat([TP(input_id[:, 4:-5]), tts_eos_e], dim=1)
        tts_pad = tts_pad_e

        T = prefill.shape[1]
        L = trailing.shape[1]
        suppress = [i for i in range(cfg.vocab_size - 1024, cfg.vocab_size) if i != eos_id]
        result = model.generate(
            inputs_embeds=prefill,
            attention_mask=torch.ones(1, T, dtype=torch.long),
            trailing_text_hidden=trailing,
            tts_pad_embed=tts_pad,
            max_new_tokens=MAX_NEW,
            min_new_tokens=2,
            do_sample=False,
            subtalker_dosample=False,
            repetition_penalty=1.05,
            eos_token_id=eos_id,
            suppress_tokens=suppress,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    talker_codes = torch.stack([h[-1] for h in result.hidden_states if h[-1] is not None], dim=1)
    first_cb = talker_codes[0, :, 0]
    is_stop = (first_cb == eos_id)
    F = int(torch.argmax(is_stop.int()).item()) if bool(is_stop.any()) else talker_codes.shape[1]
    frames = talker_codes[0, :F].to(torch.int32).contiguous()
    print(f"T={T} L={L} F={F} G={G} spk={spk_id} lang={language_id}")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "qwen_tts_synth.bin")
    tb = TEXT.encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(tb)))
        f.write(tb)
        f.write(struct.pack("<ii", H, input_id.shape[1]))
        f.write(input_id[0].to(torch.int32).numpy().tobytes())
        f.write(struct.pack("<ii", spk_id, language_id))
        f.write(struct.pack("<i", T))
        f.write(prefill[0].contiguous().numpy().astype("<f4").tobytes())
        f.write(struct.pack("<i", L))
        f.write(trailing[0].contiguous().numpy().astype("<f4").tobytes())
        f.write(tts_pad.reshape(-1).contiguous().numpy().astype("<f4").tobytes())
        f.write(struct.pack("<ii", F, G))
        f.write(frames.numpy().tobytes(order="C"))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
