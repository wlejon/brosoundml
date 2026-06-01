#!/usr/bin/env python3
"""Regenerate the Qwen3-TTS Base zero-shot voice-clone fixture (x-vector-only).

End-to-end check of the C++ synthesize_clone() path against the genuine upstream
assembly + talker generation, deterministically (greedy). Mirrors
gen_qwen_tts_synth_fixture.py (CustomVoice) but for the Base variant, where the
speaker is a raw ECAPA-TDNN x-vector spliced into the codec prefill instead of a
preset speaker token:

  - the speaker encoder: a deterministic reference waveform -> the upstream
    extract_speaker_embedding x-vector (the C++ ECAPA encoder must reproduce it);
  - the prefill assembly: the streaming Base clone prefill (x-vector in the
    speaker slot) + trailing-text + tts_pad embeddings;
  - the greedy AR loop (same logits policy as synth): the emitted [c0..c15]
    frames must match exactly.

Binary layout (little-endian):
    int32  text_len;  bytes text[text_len]
    int32  H, n_ids
    int32  input_ids[n_ids]
    int32  language_id
    int32  enc_dim;   f32 xvec[enc_dim]
    int32  n_ref;     f32 ref_wav[n_ref]              # 24 kHz mono
    int32  T;  f32 prefill[T*H]
    int32  L;  f32 trailing[L*H]
    f32    tts_pad[H]
    int32  F, G;  int32 frames[F*G]
"""
import json, math, os, struct, sys

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(os.path.dirname(__file__), "_cache")
WEIGHTS = os.path.join(REPO, "weights", "qwen-tts", "0.6B-Base")
OUT = os.path.join(REPO, "tests", "fixtures")

TEXT = "Hello there, my friend."
LANGUAGE = "english"
MAX_NEW = 40
REF_SECS = 3.0
SR = 24000


def ref_waveform():
    import torch
    g = torch.Generator().manual_seed(20260601)
    n = int(REF_SECS * SR)
    t = torch.arange(n, dtype=torch.float32) / SR
    f0 = 140.0 + 10.0 * torch.sin(2 * math.pi * 4.0 * t)
    wav = (0.5 * torch.sin(2 * math.pi * f0 * t)
           + 0.25 * torch.sin(2 * math.pi * 2 * f0 * t + 0.4)
           + 0.12 * torch.sin(2 * math.pi * 3 * f0 * t)
           + 0.04 * torch.randn(n, generator=g))
    return wav.clamp(-1.0, 1.0).contiguous()


def main():
    if not os.path.exists(os.path.join(WEIGHTS, "model.safetensors")):
        sys.exit(f"missing weights under {WEIGHTS}")
    sys.path.insert(0, CACHE)

    import torch, librosa
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    import talker_configuration_qwen3_tts as C
    import talker_modeling_qwen3_tts as M
    M.librosa_mel_fn = librosa.filters.mel

    top = json.load(open(os.path.join(WEIGHTS, "config.json")))
    full = load_file(os.path.join(WEIGHTS, "model.safetensors"))

    # ── speaker x-vector from the reference clip (upstream ECAPA-TDNN) ──
    se_cfg = C.Qwen3TTSSpeakerEncoderConfig(**top["speaker_encoder_config"])
    spk_enc = M.Qwen3TTSSpeakerEncoder(se_cfg).eval()
    se_sd = {k[len("speaker_encoder."):]: v.float()
             for k, v in full.items() if k.startswith("speaker_encoder.")}
    miss, unexp = spk_enc.load_state_dict(se_sd, strict=False)
    assert not miss and not unexp, f"spk miss {miss[:4]} unexp {unexp[:4]}"
    ref = ref_waveform()
    with torch.no_grad():
        mels = M.mel_spectrogram(ref.unsqueeze(0), n_fft=1024, num_mels=128,
                                 sampling_rate=SR, hop_size=256, win_size=1024,
                                 fmin=0, fmax=12000).transpose(1, 2)
        xvec = spk_enc(mels)[0]                     # [enc_dim]

    # ── talker (greedy clone generation) ──
    cfg = C.Qwen3TTSTalkerConfig(**top["talker_config"])
    cfg._attn_implementation = "eager"
    cfg.code_predictor_config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = M.Qwen3TTSTalkerForConditionalGeneration(cfg).eval()
    sd = {k[len("talker."):]: v.float() for k, v in full.items() if k.startswith("talker.")}
    miss, unexp = model.load_state_dict(sd, strict=False)
    miss = [m for m in miss if not m.endswith("inv_freq")]
    assert not miss and not unexp, f"talker miss {miss[:5]} unexp {unexp[:5]}"

    tok = AutoTokenizer.from_pretrained(WEIGHTS)
    prompt = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    input_id = torch.tensor(tok(prompt)["input_ids"], dtype=torch.long).unsqueeze(0)

    H = cfg.hidden_size
    G = cfg.num_code_groups
    eos_id = cfg.codec_eos_token_id
    language_id = cfg.codec_language_id[LANGUAGE.lower()]

    def TP(ids): return model.text_projection(model.get_text_embeddings()(ids))
    def CE(ids): return model.get_input_embeddings()(ids)

    with torch.no_grad():
        tts_e = TP(torch.tensor([[top["tts_bos_token_id"],
                                  top["tts_eos_token_id"],
                                  top["tts_pad_token_id"]]]))
        tts_bos_e, tts_eos_e, tts_pad_e = tts_e.chunk(3, dim=1)

        codec_prefill = [cfg.codec_think_id, cfg.codec_think_bos_id,
                         language_id, cfg.codec_think_eos_id]
        ci0 = CE(torch.tensor([codec_prefill]))
        ci1 = CE(torch.tensor([[cfg.codec_pad_id, cfg.codec_bos_id]]))
        # speaker slot = the raw x-vector (NOT a codec-token lookup)
        codec_input = torch.cat([ci0, xvec.view(1, 1, -1).float(), ci1], dim=1)

        role = TP(input_id[:, :3])
        text_side = torch.cat([tts_pad_e.expand(-1, codec_input.shape[1] - 2, -1),
                               tts_bos_e], dim=1)
        _tie = text_side + codec_input[:, :-1]
        prefill = torch.cat([role, _tie], dim=1)
        prefill = torch.cat([prefill,
                             TP(input_id[:, 3:4]) + codec_input[:, -1:]], dim=1)
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
    print(f"T={T} L={L} F={F} G={G} lang={language_id} enc_dim={xvec.shape[0]} "
          f"n_ref={ref.shape[0]} xvec_norm={xvec.norm().item():.3f}")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "qwen_tts_clone.bin")
    tb = TEXT.encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(tb))); f.write(tb)
        f.write(struct.pack("<ii", H, input_id.shape[1]))
        f.write(input_id[0].to(torch.int32).numpy().tobytes())
        f.write(struct.pack("<i", language_id))
        f.write(struct.pack("<i", int(xvec.shape[0])))
        f.write(xvec.contiguous().numpy().astype("<f4").tobytes())
        f.write(struct.pack("<i", int(ref.shape[0])))
        f.write(ref.numpy().astype("<f4").tobytes())
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
