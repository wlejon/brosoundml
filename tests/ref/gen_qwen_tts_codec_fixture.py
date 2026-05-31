#!/usr/bin/env python3
"""Regenerate the Qwen3-TTS codec-decode test fixtures.

The C++ codec decoder (src/qwen_tts_codec.cpp) is validated bit-for-bit against
the *genuine upstream* 12 Hz tokenizer decoder. This script builds that decoder
from the official Qwen3-TTS modeling code, loads the real `decoder.*` weights
from speech_tokenizer/model.safetensors, runs forward() on seeded random codes,
and writes the (codes, waveform) pairs the test consumes.

The fixtures are gitignored (*.bin) — regenerate them locally to enable the
opt-in numeric check in tests/test_qwen_tts.cpp (it skips when they are absent).

Requirements
------------
  * The weights: scripts/download-qwen-tts.sh  (-> weights/qwen-tts/0.6B-customvoice)
  * transformers 4.57.x (the version the upstream modeling code targets) + torch.
    A clean way that reuses your system torch:
        python -m venv --system-site-packages .venv57
        .venv57/Scripts/python -m pip install "transformers==4.57.3"
        .venv57/Scripts/python tests/ref/gen_qwen_tts_codec_fixture.py
  * Network access on first run (downloads two upstream .py files into
    tests/ref/_cache/, pinned by commit; cached thereafter).

Fixture binary layout (little-endian):
    int32   K              # num_quantizers (16)
    int32   T              # frames
    int32   n_samples      # = T * 1920
    int32   [K*T]          # codes, codebook-major: code[k*T + t]
    float32 [n_samples]    # decoded waveform (post clamp[-1,1])
"""
import json
import os
import struct
import sys
import urllib.request

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(os.path.dirname(__file__), "_cache")
WEIGHTS = os.path.join(REPO, "weights", "qwen-tts", "0.6B-customvoice", "speech_tokenizer")
OUT = os.path.join(REPO, "tests", "fixtures")

# Pin the upstream modeling + config to a commit so regeneration is reproducible.
GH = "https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/qwen_tts/core/tokenizer_12hz"
FILES = {
    "configuration_qwen3_tts_tokenizer_v2.py": GH + "/configuration_qwen3_tts_tokenizer_v2.py",
    "qwen_codec.py":                            GH + "/modeling_qwen3_tts_tokenizer_v2.py",
}


def ensure_upstream():
    os.makedirs(CACHE, exist_ok=True)
    for fname, url in FILES.items():
        dest = os.path.join(CACHE, fname)
        if not os.path.exists(dest):
            print(f"downloading {url}")
            data = urllib.request.urlopen(url).read().decode("utf-8")
            # The modeling file uses a relative import of its config module;
            # neutralize it so the cache dir is importable as plain modules.
            data = data.replace(
                "from .configuration_qwen3_tts_tokenizer_v2 import",
                "from configuration_qwen3_tts_tokenizer_v2 import",
            )
            with open(dest, "w", encoding="utf-8") as f:
                f.write(data)
    sys.path.insert(0, CACHE)


def main():
    if not os.path.exists(os.path.join(WEIGHTS, "model.safetensors")):
        sys.exit(f"missing codec weights under {WEIGHTS} "
                 f"(run scripts/download-qwen-tts.sh first)")
    ensure_upstream()

    import torch
    from safetensors.torch import load_file
    from configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderConfig
    import qwen_codec as M

    cfg = Qwen3TTSTokenizerV2DecoderConfig(
        **json.load(open(os.path.join(WEIGHTS, "config.json")))["decoder_config"])
    dec = M.Qwen3TTSTokenizerV2Decoder._from_config(cfg).eval()

    full = load_file(os.path.join(WEIGHTS, "model.safetensors"))
    sd = {k[len("decoder."):]: v for k, v in full.items() if k.startswith("decoder.")}
    missing, unexpected = dec.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("inv_freq")]
    assert not missing, f"missing weights: {missing[:8]}"
    assert not unexpected, f"unexpected weights: {unexpected[:8]}"
    print(f"loaded {len(sd)} decoder tensors")

    K = cfg.num_quantizers
    os.makedirs(OUT, exist_ok=True)

    def gen(T, seed, fname):
        g = torch.Generator().manual_seed(seed)
        codes = torch.randint(0, cfg.codebook_size, (1, K, T), generator=g)
        with torch.no_grad():
            wav = dec(codes).squeeze(0).squeeze(0).contiguous().float().numpy()
        assert wav.shape[0] == T * 1920
        c = codes.squeeze(0).contiguous().to(torch.int32).numpy()  # (K, T)
        path = os.path.join(OUT, fname)
        with open(path, "wb") as f:
            f.write(struct.pack("<iii", K, T, wav.shape[0]))
            f.write(c.tobytes(order="C"))
            f.write(wav.astype("<f4").tobytes())
        print(f"  wrote {path}: K={K} T={T} n={wav.shape[0]}")

    gen(T=37,  seed=1234, fname="qwen_tts_codec_small.bin")   # < sliding_window
    gen(T=120, seed=5678, fname="qwen_tts_codec.bin")         # > sliding_window (72)
    print("done")


if __name__ == "__main__":
    main()
