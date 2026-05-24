#!/usr/bin/env python3
"""Dump per-stage Kokoro intermediates to safetensors for C++ comparison.

Loads the upstream `kokoro` package's `KModel` (which gives us a known-good
PyTorch forward pass), runs it once on a fixed phoneme sequence + voice, and
saves every interesting intermediate tensor under
`weights/kokoro/reference/<stage>.safetensors`. brosoundml's C++ tests load
those files and check their own intermediates row-by-row.

Why hand-rolled forward pass instead of `model.forward(...)`?
Because KModel.forward() collapses every stage into one black-box call;
hand-rolling lets us snapshot bert_dur, d_en, d, lstm_x, duration, pred_dur,
F0_pred, N_pred, t_en, asr, and audio individually. The order and op list
mirror kokoro/model.py:KModel.forward exactly — keep them in sync if a
future upstream Kokoro release reshapes the pipeline.

Usage:
  scripts/dump-reference.py [--src DIR] [--voice NAME] [--phonemes STR]
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

# Suppress kokoro/__init__.py's KPipeline import (which pulls misaki/spacy);
# we only need KModel here.
sys.modules.setdefault('kokoro.pipeline', types.ModuleType('kokoro.pipeline'))
sys.modules['kokoro.pipeline'].KPipeline = None  # type: ignore[attr-defined]

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from kokoro.model import KModel  # noqa: E402


def to_save(t: torch.Tensor) -> torch.Tensor:
    """Detach + force FP32 + contiguous + on CPU before save_file."""
    return t.detach().to(dtype=torch.float32, device='cpu').contiguous()


def dump(out_dir: Path, stage: str, tensors: dict[str, torch.Tensor]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: to_save(v) for k, v in tensors.items()},
              str(out_dir / f"{stage}.safetensors"))
    print(f"  -> {stage}.safetensors  (" +
          ", ".join(f"{k}{tuple(v.shape)}" for k, v in tensors.items()) + ")")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_src = repo_root / "weights" / "kokoro"

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=default_src,
                    help="Directory holding config.json + kokoro-v1_0.pth")
    ap.add_argument("--voice", type=str, default="af_heart",
                    help="Voice pack stem under <src>/voices/ (loaded from .pt)")
    # A short fixed phoneme sequence — readable letters in the Kokoro vocab.
    # Stage 3+ tests pin to this exact input so a regression flips a diff.
    ap.add_argument("--phonemes", type=str, default="hello",
                    help="Phoneme string fed to the model; each character must "
                         "be in the Kokoro vocab")
    args = ap.parse_args()

    src: Path = args.src
    voice_path = src / "voices" / f"{args.voice}.pt"
    pth_path   = src / "kokoro-v1_0.pth"
    cfg_path   = src / "config.json"
    for p in (cfg_path, pth_path, voice_path):
        if not p.exists():
            print(f"error: missing {p}", file=sys.stderr)
            return 2

    # KModel loads config + weights from the paths we hand it; no HF download.
    print(f"Loading KModel from {pth_path} ...")
    model = KModel(config=str(cfg_path), model=str(pth_path))
    model.eval()

    # ─── input prep (mirror KModel.forward) ────────────────────────────────
    vocab = model.vocab
    input_ids = [vocab[c] for c in args.phonemes if c in vocab]
    input_ids = [0, *input_ids, 0]
    L = len(input_ids)
    input_ids_t  = torch.LongTensor([input_ids])
    input_lengths = torch.LongTensor([L])
    text_mask = torch.arange(L).unsqueeze(0).expand(1, -1)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

    voice_pack = torch.load(voice_path, map_location='cpu', weights_only=True)
    # KModel selects the row matching (len(input_ids) - 1) phonemes. Our voice
    # pack rows are indexed by phoneme count - 1; KModel.forward does
    # `ref_s = voice[len(phonemes) - 1]`. We mirror that.
    ref_s = voice_pack[L - 1]  # (1, voice_dim) or (voice_dim,) per pack layout

    out_dir = src / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Phonemes={args.phonemes!r}  ids={input_ids}  L={L}")
    print(f"Voice ref_s shape: {tuple(ref_s.shape)}")
    print(f"Writing to {out_dir}\n")

    # ─── inputs ────────────────────────────────────────────────────────────
    dump(out_dir, "00_inputs", {
        "input_ids": input_ids_t.float(),       # (1, L) — int as float for safetensors F32
        "text_mask": text_mask.float(),         # (1, L) — bool as float
        "ref_s":     ref_s.unsqueeze(0).float() if ref_s.dim() == 1 else ref_s.float(),
    })

    with torch.no_grad():
        # 1. plBERT.
        bert_dur = model.bert(input_ids_t, attention_mask=(~text_mask).int())
        dump(out_dir, "01_bert", {"bert_dur": bert_dur})

        # 2. bert_encoder + transpose.
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        dump(out_dir, "02_bert_encoder", {"d_en": d_en})

        # 3. predictor.text_encoder + duration LSTM + duration_proj.
        ref_s_b = ref_s.unsqueeze(0) if ref_s.dim() == 1 else ref_s
        s = ref_s_b[:, 128:]          # predictor style (1, 128)
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        lstm_x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(lstm_x)
        duration_sig_sum = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration_sig_sum).clamp(min=1).long().squeeze()
        dump(out_dir, "03_predictor_pre", {
            "d":           d,
            "lstm_x":      lstm_x,
            "duration":    duration,
            "duration_ss": duration_sig_sum,
            "pred_dur":    pred_dur.float(),
        })

        # 4. Length regulator (one-hot expand).
        indices = torch.repeat_interleave(torch.arange(L), pred_dur)
        pred_aln_trg = torch.zeros((L, indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg
        dump(out_dir, "04_length_reg", {
            "pred_aln_trg": pred_aln_trg,
            "en":           en,
        })

        # 5. F0 + N predictor.
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        dump(out_dir, "05_f0_energy", {"F0_pred": F0_pred, "N_pred": N_pred})

        # 6. The other (StyleTTS2-style) text encoder.
        t_en = model.text_encoder(input_ids_t, input_lengths, text_mask)
        dump(out_dir, "06_text_encoder", {"t_en": t_en})
        asr = t_en @ pred_aln_trg
        dump(out_dir, "07_asr", {"asr": asr})

        # 7. Decoder + iSTFT head — capture every intermediate so each stage
        #    of the C++ port can be validated independently.
        dec = model.decoder
        dec_style = ref_s_b[:, :128]

        F0_dn = dec.F0_conv(F0_pred.unsqueeze(1))     # (1, 1, T)
        N_dn  = dec.N_conv (N_pred.unsqueeze(1))      # (1, 1, T)
        dec_pre_x = torch.cat([asr, F0_dn, N_dn], axis=1)  # (1, 514, T)
        dec_enc_x = dec.encode(dec_pre_x, dec_style)        # (1, 1024, T)
        asr_res   = dec.asr_res(asr)                        # (1, 64, T)
        dump(out_dir, "08_decoder_pre", {
            "F0_dn":      F0_dn,
            "N_dn":       N_dn,
            "dec_pre_x":  dec_pre_x,
            "dec_enc_x":  dec_enc_x,
            "asr_res":    asr_res,
        })

        # decode loop — record after each block.
        x = dec_enc_x
        res = True
        cat_inputs = []
        block_outs = []
        for i, block in enumerate(dec.decode):
            if res:
                x = torch.cat([x, asr_res, F0_dn, N_dn], axis=1)
                cat_inputs.append(x.detach())
            x = block(x, dec_style)
            block_outs.append(x.detach())
            if block.upsample_type != "none":
                res = False
        dec_gen_in = x  # (1, 512, 2*T) after the upsample block
        dec_tensors = {}
        for i, t in enumerate(cat_inputs):
            dec_tensors[f"cat_in_{i}"] = t
        for i, t in enumerate(block_outs):
            dec_tensors[f"block_out_{i}"] = t
        dec_tensors["gen_in"] = dec_gen_in.clone()
        dump(out_dir, "09_decoder_decode", dec_tensors)

        # Generator: capture the source (random-driven) and the final audio.
        torch.manual_seed(0)
        gen = dec.generator
        f0_for_src = F0_pred[:, None]
        f0_up = gen.f0_upsamp(f0_for_src).transpose(1, 2)
        har_source, noi_source, uv = gen.m_source(f0_up)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
        dump(out_dir, "10_generator_src", {
            "har_source": har_source,
            "har_spec":   har_spec,
            "har_phase":  har_phase,
            "har":        har,
        })

        # Re-run the full audio with the same seed so the audio matches the
        # captured har source (otherwise random reuse would diverge).
        torch.manual_seed(0)
        audio = dec(asr, F0_pred, N_pred, dec_style).squeeze()
        dump(out_dir, "11_audio", {"audio": audio})

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
