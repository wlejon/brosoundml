#!/usr/bin/env python3
"""Regenerate the CLAP (laion/larger_clap_general) parity fixture.

Validates the C++ port (src/clap*.cpp) against the genuine upstream model:
transformers' ClapModel + ClapFeatureExtractor + the RoBERTa tokenizer on the
real checkpoint, dumping every stage boundary for a handful of clips and
prompts:

    * the 48 kHz waveform the feature extractor sees (after resampling),
    * the log-mel features (1001 x 64, slaney, repeatpad / rand_trunc),
    * the HTSAT pooled output (1024) and the projected, normalised audio
      embedding (512),
    * the RoBERTa token ids and the normalised text embedding (512) per prompt,
    * the cosine similarities, the logit scale and the per-clip softmax.

Clips cover every branch of the extractor: a synthetic 48 kHz clip shorter
than 10 s (repeat-pad), one exactly 10 s, two real 44.1 kHz stereo clips (a
0.58 s hum and an 11 s long one, the latter through the rand_trunc crop at a
seeded offset, recorded in the fixture), and a 16 kHz speech clip. Resampling
to 48 kHz uses torchaudio.functional.resample, the recipe brosoundml::resample
implements. The long clip also pins the C++ "mean over 10 s windows" mode,
computed here with the same window placement.

The fixture is gitignored (*.bin) — regenerate locally to enable the numeric
checks in tests/test_clap.cpp (they skip when it is absent).

Requirements: the weights (scripts/download-clap.sh + scripts/convert-clap.py
-> weights/clap), torch, torchaudio, transformers >= 4.35. FP32 on the CPU.

Fixture layout (little-endian): the magic "CLAPFIX1", int32 record count, then
per record: int32 name length, the name bytes, int32 dtype (0 f32, 1 i32,
2 utf-8 bytes), int32 ndim, int32 dims[ndim], the data.
"""
import os
import struct
import wave

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEIGHTS = os.path.join(REPO, "weights", "clap")
OUT = os.path.join(REPO, "tests", "fixtures", "clap.bin")
MAX = 480000
CROP_SEED = 1234

PROMPTS = [
    "a dog barking",
    "a person humming",
    "a whistle",
    "xylophone",
    "synthetic electronic beeps",
    "a man speaking",
    "Rain falling on a tin roof, café ambience!",
]


def read_wav_mono(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        sr = w.getframerate()
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1, dtype=np.float32)
        return data.astype(np.float32), sr


def synth_short():
    rng = np.random.default_rng(7)
    sr = 48000
    t = np.arange(int(2.7 * sr)) / sr
    chirp = 0.3 * np.sin(2 * np.pi * (300 * t + 900 * t * t))
    noise = 0.05 * rng.standard_normal(t.shape)
    env = np.minimum(1.0, t * 20.0)
    return (env * (chirp + noise)).astype(np.float32), sr


def synth_exact10():
    rng = np.random.default_rng(11)
    sr = 48000
    t = np.arange(MAX) / sr
    beeps = 0.25 * np.sin(2 * np.pi * 1000 * t) * ((t * 4) % 1.0 < 0.3)
    return (beeps + 0.02 * rng.standard_normal(t.shape)).astype(np.float32), sr


def to48k(x, sr):
    if sr == 48000:
        return x.copy()
    y = torchaudio.functional.resample(torch.from_numpy(x).double(), sr, 48000)
    return y.float().numpy()


def window_starts(n):
    k = -(-n // MAX)
    if k <= 1:
        return [0]
    return [int(round(i * (n - MAX) / (k - 1))) for i in range(k)]


class Writer:
    def __init__(self):
        self.recs = []

    def add(self, name, arr, dtype=None):
        if isinstance(arr, str):
            data = arr.encode("utf-8")
            self.recs.append((name, 2, [len(data)], data))
            return
        a = np.asarray(arr)
        if dtype == "i32" or a.dtype.kind in "iu":
            a = a.astype("<i4")
            code = 1
        else:
            a = a.astype("<f4")
            code = 0
        self.recs.append((name, code, list(a.shape) or [1], a.tobytes()))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"CLAPFIX1")
            f.write(struct.pack("<i", len(self.recs)))
            for name, code, dims, data in self.recs:
                nb = name.encode("utf-8")
                f.write(struct.pack("<i", len(nb)))
                f.write(nb)
                f.write(struct.pack("<ii", code, len(dims)))
                f.write(struct.pack("<%di" % len(dims), *dims))
                f.write(data)


def main():
    from transformers import ClapFeatureExtractor, ClapModel, AutoTokenizer

    torch.set_grad_enabled(False)
    model = ClapModel.from_pretrained(WEIGHTS, dtype=torch.float32).eval()
    fe = ClapFeatureExtractor.from_pretrained(WEIGHTS)
    tok = AutoTokenizer.from_pretrained(WEIGHTS)
    assert fe.truncation == "rand_trunc" and fe.padding == "repeatpad"

    clips = []
    clips.append(("short", *synth_short()))
    clips.append(("exact10", *synth_exact10()))
    clips.append(("hum", *read_wav_mono(os.path.join(REPO, "weights", "mhmm.wav"))))
    clips.append(("long", *read_wav_mono(os.path.join(REPO, "weights", "theraputic.wav"))))
    clips.append(("speech16k", *read_wav_mono(
        os.path.join(REPO, "weights", "qwen-tts-hello-there-this-is-a-test-of-th.wav"))))

    w = Writer()
    w.add("clip_names", "\n".join(c[0] for c in clips))
    w.add("prompt_texts", "\n".join(PROMPTS))
    scale_a = model.logit_scale_a.exp().item()
    w.add("logit_scale_a", [scale_a])
    w.add("logit_scale_t", [model.logit_scale_t.exp().item()])

    def embed_mel(mel):
        out = model.audio_model(input_features=mel)
        pooled = out.pooler_output
        return pooled, F.normalize(model.audio_projection(pooled), dim=-1)

    audio_embeds = []
    for name, src, sr in clips:
        wav48 = to48k(src, sr)
        crop = -1
        if len(wav48) > MAX:
            np.random.seed(CROP_SEED)
            crop = int(np.random.randint(0, len(wav48) - MAX + 1))
            np.random.seed(CROP_SEED)
        feats = fe(wav48, sampling_rate=48000, return_tensors="pt")
        mel = feats["input_features"].float()               # (1, 1, 1001, 64)
        if crop >= 0:
            direct = fe(wav48[crop:crop + MAX], sampling_rate=48000, return_tensors="pt")
            assert torch.equal(direct["input_features"], feats["input_features"]), "crop replay mismatch"
        pooled, emb = embed_mel(mel)
        print(f"{name}: {len(src)} @ {sr} -> {len(wav48)} @ 48k, crop {crop}, mel {tuple(mel.shape)}")
        w.add(f"clip/{name}/src", src)
        w.add(f"clip/{name}/rate", [sr], "i32")
        w.add(f"clip/{name}/wave48", wav48)
        w.add(f"clip/{name}/crop", [crop], "i32")
        w.add(f"clip/{name}/mel", mel[0, 0].numpy())
        w.add(f"clip/{name}/pooled", pooled[0].numpy())
        w.add(f"clip/{name}/embed", emb[0].numpy())
        audio_embeds.append(emb[0])
        if len(wav48) > MAX:
            starts = window_starts(len(wav48))
            embs = []
            for s in starts:
                m = fe(wav48[s:s + MAX], sampling_rate=48000, return_tensors="pt")["input_features"].float()
                embs.append(embed_mel(m)[1][0])
            mean = F.normalize(torch.stack(embs).mean(0), dim=-1)
            w.add(f"clip/{name}/win_starts", starts, "i32")
            w.add(f"clip/{name}/win_embed", mean.numpy())

    enc = tok(PROMPTS, padding=True, return_tensors="pt")
    tout = model.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    text_embeds = F.normalize(model.text_projection(tout.pooler_output), dim=-1)
    for i, p in enumerate(PROMPTS):
        n = int(enc["attention_mask"][i].sum())
        w.add(f"prompt/{i}/ids", enc["input_ids"][i, :n].numpy(), "i32")
        w.add(f"prompt/{i}/embed", text_embeds[i].numpy())

    A = torch.stack(audio_embeds)
    sim = A @ text_embeds.T
    logits = sim * scale_a
    probs = logits.softmax(dim=-1)
    # Cross-check against ClapModel.forward for the first clip.
    ref = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                input_features=fe(to48k(clips[0][1], clips[0][2]), sampling_rate=48000,
                                  return_tensors="pt")["input_features"])
    assert torch.allclose(ref.logits_per_audio[0], logits[0], atol=1e-4)
    w.add("sim", sim.numpy())
    w.add("logits", logits.numpy())
    w.add("probs", probs.numpy())
    np.set_printoptions(precision=3, suppress=True)
    print("prompts:", PROMPTS)
    print("probs:\n", probs.numpy())
    w.save(OUT)
    print(f"wrote {OUT} ({os.path.getsize(OUT):,} bytes)")


if __name__ == "__main__":
    main()
