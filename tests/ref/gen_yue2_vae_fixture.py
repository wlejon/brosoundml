#!/usr/bin/env python3
"""Regenerate the YuE2 VAE test fixture (tests/fixtures/yue2_vae.bin).

The C++ VAE (src/yue2_vae.cpp) is validated against the genuine reference:
the yue2 package's `YuE2VAE` (modeling_vae.py) in float32 on CUDA, over the
m-a-p/YuE2-Vae weights. Each case is a synthetic 48 kHz stereo signal (a
chord with a chirp in the left channel and a pulse train in the right, plus a
little deterministic noise) that is encoded to its posterior-mean latents and
decoded back:

  input (2 x S) -> latents (64 x T, encode mean) -> decoded (2 x N, full
  decode, N = 1920*T - 64) -> decoded_tiled (2 x N, decode_tiled with a small
  core so the song spans several tiles)

plus required_halo() for a few core sizes, which the C++ dependency walk must
reproduce exactly.

Cases: "short" (2.0 s, small enough for the CPU test) and "long" (8.0 s, GPU).

The fixture is gitignored (*.bin); tests/test_yue2_vae.cpp skips its numeric
checks when it is absent.

Requirements: the yue2 reference package (../YuE2-3B/.venv has it), torch with
CUDA, safetensors, numpy; the VAE under weights/yue2-vae or the Hugging Face
cache (models--m-a-p--YuE2-Vae).

    ../YuE2-3B/.venv/Scripts/python.exe tests/ref/gen_yue2_vae_fixture.py

Fixture binary layout (little-endian):
    int32   magic = 0x31455659  ('YVE1')
    int32   n_halo
    n_halo x (int32 core_frames, int32 required_halo)
    int32   n_cases
    per case:
      int32   S          # input samples per channel (48 kHz)
      int32   T          # latent frames
      int32   N          # decoded samples per channel (1920*T - 64)
      int32   core       # decode_tiled core_frames
      int32   halo       # decode_tiled halo_frames
      float32 [2*S]      # input, channel-major (planar): x[c*S + i]
      float32 [64*T]     # latents, channel-major: z[c*T + t]
      float32 [2*N]      # full decode, planar
      float32 [2*N]      # tiled decode, planar
"""
import glob
import os
import struct
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "tests", "fixtures", "yue2_vae.bin")
MAGIC = 0x31455659


def find_vae():
    local = os.path.join(REPO, "weights", "yue2-vae")
    if os.path.exists(os.path.join(local, "model.safetensors")):
        return local
    hub = os.path.expanduser("~/.cache/huggingface/hub/models--m-a-p--YuE2-Vae/snapshots/*")
    for d in sorted(glob.glob(hub)):
        if os.path.exists(os.path.join(d, "model.safetensors")):
            return d
    sys.exit("no YuE2 VAE under weights/yue2-vae or the Hugging Face cache")


def signal(seconds, sr=48000, seed=7):
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float64) / sr
    chord = sum(0.12 * np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63))
    f0, f1 = 80.0, 6000.0
    chirp = 0.2 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * seconds)))
    pulses = 0.3 * (np.sin(2 * np.pi * 2.0 * t) > 0.95) * np.sin(2 * np.pi * 1000.0 * t)
    rng = np.random.default_rng(seed)
    left = chord + chirp + 0.01 * rng.standard_normal(n)
    right = chord + pulses + 0.01 * rng.standard_normal(n)
    return np.stack([left, right]).astype(np.float32)


def main():
    import torch
    sys.path.insert(0, os.path.normpath(os.path.join(REPO, "..", "YuE2-3B", ".venv", "Lib", "site-packages")))
    from yue2.modeling_vae import YuE2VAE

    if not torch.cuda.is_available():
        sys.exit("CUDA is required (the reference runs on the GPU)")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True

    vae = YuE2VAE.from_pretrained(find_vae(), device="cuda")
    halos = [(c, vae.required_halo(c)) for c in (1, 8, 16, 64, 1024)]
    print("required_halo:", halos)

    cases = [("short", 2.0, 16), ("long", 8.0, 64)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<ii", MAGIC, len(halos)))
        for c, h in halos:
            f.write(struct.pack("<ii", c, h))
        f.write(struct.pack("<i", len(cases)))
        for name, seconds, core in cases:
            x = signal(seconds)
            xt = torch.from_numpy(x)[None].cuda()
            z = vae.encode(xt)                                   # (1, 64, T)
            full = vae.decode(z)                                 # (1, 2, N)
            halo = vae.required_halo(core)
            tiled = vae.decode_tiled(z, core_frames=core, halo_frames=halo)
            S, T, N = x.shape[1], z.shape[-1], full.shape[-1]
            assert N == 1920 * T - 64, (N, T)
            d = (full.cpu() - tiled).abs().max().item()
            print(f"{name}: S={S} T={T} N={N} core={core} halo={halo} "
                  f"|full|max={full.abs().max().item():.4f} full-vs-tiled={d:.3e}")
            f.write(struct.pack("<iiiii", S, T, N, core, halo))
            f.write(x.astype("<f4").tobytes())
            f.write(z[0].cpu().numpy().astype("<f4").tobytes())
            f.write(full[0].cpu().numpy().astype("<f4").tobytes())
            f.write(tiled[0].numpy().astype("<f4").tobytes())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
