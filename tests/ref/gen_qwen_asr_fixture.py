#!/usr/bin/env python3
"""Regenerate the Qwen3-ASR end-to-end test fixture.

Validates the C++ pipeline (src/qwen_asr*.cpp) against the *genuine upstream*
model: it loads the official QwenLM/Qwen3-ASR transformers-backend modeling
code, the real Qwen/Qwen3-ASR-0.6B weights, and runs the full
mel ▶ AuT encoder ▶ Qwen3 decoder pipeline on a real speech clip, dumping
every stage boundary:

    * the WhisperFeatureExtractor log-mel features,
    * the audio encoder output (the embeddings spliced into the LLM),
    * the prefill logits at the last prompt position,
    * the greedy-decoded token ids.

The fixture is gitignored (*.bin) — regenerate locally to enable the numeric
check in tests/test_qwen_asr.cpp (it skips when absent).

Requirements
------------
  * The weights: scripts/download-qwen-asr.sh  (-> weights/qwen-asr/0.6B)
  * transformers 4.57.x (the version the upstream modeling code targets) +
    torch. A clean way that reuses your system torch:
        python -m venv --system-site-packages .venv57
        .venv57/Scripts/python -m pip install "transformers==4.57.6"
        .venv57/Scripts/python tests/ref/gen_qwen_asr_fixture.py
  * Network access on first run (downloads the upstream modeling .py files
    into tests/ref/_cache/; cached thereafter).

Everything runs in FP32 on the CPU — the same precision contract the C++ port
keeps on every backend.

Fixture binary layout (little-endian):
    int32   n_samples                # 16 kHz mono audio length
    int32   frames, n_mels           # log-mel feature shape
    int32   n_audio, enc_dim         # encoder output shape (enc_dim = 1024)
    int32   prompt_len               # chat-template prompt incl. audio pads
    int32   vocab                    # 151936
    int32   n_gen                    # greedy tokens (EOS stripped)
    f32     audio[n_samples]
    f32     mel[frames * n_mels]     # frame-major: mel[t * n_mels + m]
    f32     enc[n_audio * enc_dim]
    int32   prompt_ids[prompt_len]
    f32     logits[vocab]            # prefill logits at the last position
    int32   gen_ids[n_gen]
"""
import os
import struct
import sys
import urllib.request
import wave

import numpy as np
import torch

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(os.path.dirname(__file__), "_cache")
WEIGHTS = os.path.join(REPO, "weights", "qwen-asr", "0.6B")
WAV = os.path.join(REPO, "weights", "qwen-tts-hello-there-this-is-a-test-of-th.wav")
OUT = os.path.join(REPO, "tests", "fixtures", "qwen_asr.bin")

GH = "https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/qwen_asr/core/transformers_backend"
# Upstream file -> (cache name, list of literal text substitutions). The
# modeling file's relative config import is rewritten so the cache dir is
# importable as flat modules under stock transformers.
FILES = {
    "configuration_qwen3_asr.py": ("asr_configuration_qwen3_asr.py", []),
    "modeling_qwen3_asr.py": ("asr_modeling_qwen3_asr.py", [
        ("from .configuration_qwen3_asr import",
         "from asr_configuration_qwen3_asr import"),
    ]),
}

# Chat-template token ids (tokenizer facts, fixed for the Qwen BPE vocab):
# <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
PROMPT_PREFIX = [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
# <|audio_end|><|im_end|>\n<|im_start|>assistant\n
PROMPT_SUFFIX = [151670, 151645, 198, 151644, 77091, 198]
AUDIO_PAD = 151676
EOS_IDS = {151643, 151645}
MAX_NEW_TOKENS = 64


def ensure_upstream():
    os.makedirs(CACHE, exist_ok=True)
    for src, (name, subs) in FILES.items():
        dest = os.path.join(CACHE, name)
        if not os.path.exists(dest):
            print(f"fetching {src} ...")
            with urllib.request.urlopen(f"{GH}/{src}") as r:
                text = r.read().decode("utf-8")
            for a, b in subs:
                assert a in text, f"substitution anchor missing in {src}: {a!r}"
                text = text.replace(a, b)
            with open(dest, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
    sys.path.insert(0, CACHE)


def read_wav_mono(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        sr = w.getframerate()
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
        return data.astype(np.float32) / 32768.0, sr


def resample_linear(x, sr_in, sr_out):
    n_out = int(round(len(x) * sr_out / sr_in))
    pos = np.linspace(0.0, len(x) - 1, n_out)
    return np.interp(pos, np.arange(len(x)), x).astype(np.float32)


def main():
    ensure_upstream()
    from asr_configuration_qwen3_asr import Qwen3ASRConfig
    from asr_modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
    from transformers import WhisperFeatureExtractor

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    audio, sr = read_wav_mono(WAV)
    if sr != 16000:
        audio = resample_linear(audio, sr, 16000)
    print(f"audio: {len(audio)} samples ({len(audio)/16000.0:.2f}s)")

    cfg = Qwen3ASRConfig.from_pretrained(WEIGHTS)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        WEIGHTS, config=cfg, dtype=torch.float32)
    model.eval()
    thinker = model.thinker

    # ── 1. log-mel features (the official front-end) ──
    fe = WhisperFeatureExtractor.from_pretrained(WEIGHTS)
    feats = fe([audio], sampling_rate=16000, padding=True, truncation=False,
               return_attention_mask=True, return_tensors="pt")
    input_features = feats["input_features"].float()       # (1, 128, frames)
    feat_mask = feats["attention_mask"]                     # (1, frames)
    frames = int(feat_mask.sum().item())
    n_mels = input_features.shape[1]
    mel = input_features[0, :, :frames].T.contiguous()      # (frames, n_mels)
    print(f"mel: ({frames}, {n_mels})")

    # ── 2. audio encoder ──
    enc = thinker.get_audio_features(
        input_features, feature_attention_mask=feat_mask)   # (n_audio, 1024)
    n_audio, enc_dim = enc.shape
    print(f"encoder out: ({n_audio}, {enc_dim})")

    # ── 3. prompt + prefill logits at the last position ──
    ids = PROMPT_PREFIX + [AUDIO_PAD] * n_audio + PROMPT_SUFFIX
    input_ids = torch.tensor([ids], dtype=torch.long)
    out = thinker(input_ids=input_ids,
                  input_features=input_features,
                  feature_attention_mask=feat_mask,
                  use_cache=True)
    logits_last = out.logits[0, -1].float()                 # (vocab,)
    vocab = logits_last.shape[0]
    print(f"prompt: {len(ids)} tokens; prefill logits argmax: "
          f"{int(logits_last.argmax())}")

    # ── 4. greedy decode with the upstream cache ──
    past = out.past_key_values
    token = int(logits_last.argmax())
    gen = []
    while len(gen) < MAX_NEW_TOKENS and token not in EOS_IDS:
        gen.append(token)
        step = thinker(input_ids=torch.tensor([[token]], dtype=torch.long),
                       past_key_values=past, use_cache=True)
        past = step.past_key_values
        token = int(step.logits[0, -1].argmax())
    print(f"generated {len(gen)} tokens: {gen}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<8i", len(audio), frames, n_mels, n_audio,
                            enc_dim, len(ids), vocab, len(gen)))
        f.write(audio.astype("<f4").tobytes())
        f.write(mel.numpy().astype("<f4").tobytes())
        f.write(enc.numpy().astype("<f4").tobytes())
        f.write(np.asarray(ids, dtype="<i4").tobytes())
        f.write(logits_last.numpy().astype("<f4").tobytes())
        f.write(np.asarray(gen, dtype="<i4").tobytes())
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
