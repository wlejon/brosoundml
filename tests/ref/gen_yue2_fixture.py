#!/usr/bin/env python3
"""Regenerate the YuE2-3B backbone fixtures from the genuine reference package
(the `yue2` wheel installed in ../YuE2-3B/.venv: protocol.py, sampling.py,
modeling_yue2.py, nar.py).

Two files (tests/fixtures/, gitignored *.bin; the tests skip what is absent):

  yue2_protocol.bin   CPU only, no weights: tokenizer ids, prompt prefixes,
                      chunk_ranges, torch's CPU randn for a seed, and
                      sampling.distribution on fixed logits (plus the BF16 CFG
                      combine and the legacy cot=off arithmetic).
  yue2_model.bin      CUDA, BF16 weights: AR logits for real prompt prefixes,
                      greedy (temperature 0) plan / semantic / CFG token
                      streams through sampling.generate_tokens (its CUDA-graph
                      path), and nar.CachedNAR on a short chunk: one velocity,
                      then the full midpoint solve from the chunk's noise.

    ../YuE2-3B/.venv/Scripts/python.exe tests/ref/gen_yue2_fixture.py [--protocol-only]

Model folder: $YUE2_MODEL_DIR, else ../YuE2-3B.

Binary conventions (little-endian): i32 / i64 / f32 / f64; str = i32 nbytes +
UTF-8; i32list = i32 n + n x i32; f32 arrays are written with their length
implied by preceding fields.

yue2_protocol.bin
    i32 magic 0x31505559 ('YUP1')
    i32 n_strings; per: str s; i32list ids                    # tokenizer.encode
    i32 n_prompts; per:
        str style; str lyrics; i32 cot (0 off, 1 melody, 2 full); str abc
        i32list plan_prefix      # token_prefixes(request)  (no ABC yet)
        i32list abc_ids          # tokenizer.encode(abc) ([] for off)
        i32list sem_prefix       # token_prefixes(request, abc_ids)
        i32list negative         # negative_prefix(request, abc_ids)
        str text                 # request.text()
    i32 n_ranges; per: i32 frames; i32 prefix_tokens; i32list flat (a0,b0,a1,b1,..)
    u64 seed; i32 n; f32 randn[n]                               # torch CPU randn((n/64,64))
    i32 n_cases; per:
        i32 phase (0 abc, 1 semantic)
        f32 temperature; f32 top_p; i32 top_k; f32 penalty; i32 window; i32 min; i32 max
        i32 step; i32 legacy_off; f32 cfg (1 = none)
        i32list history
        i32 n                    # logits row over the phase's id range
        f32 cond[n]; [f32 uncond[n] if cfg != 1]; f32 combined[n]; f32 scores[n]

yue2_model.bin
    i32 magic 0x314d5559 ('YUM1')
    i32 n_logits; per: i32list ids; f32 logits[184704]          # last row, bf16 -> f32
    greedy stream x3 (plan, semantic, cfg):
        i32 phase; i32list prefix; i32list negative; f32 cfg; i32 legacy
        f32 temperature; f32 top_p; i32 top_k; f32 penalty; i32 window; i32 min; i32 max
        i32list tokens; i32 truncated
    NAR: i32list ar_tokens; i32 frames; f32 noise[frames*64]; f64 raw_t
         f32 velocity[frames*64]; i32 steps; f32 latents[frames*64]
"""
import argparse
import os
import struct
import sys

import torch

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIX = os.path.join(REPO, "tests", "fixtures")
MODEL = os.environ.get("YUE2_MODEL_DIR", os.path.normpath(os.path.join(REPO, "..", "YuE2-3B")))

from yue2 import protocol as P  # noqa: E402
from yue2.sampling import distribution, generate_tokens  # noqa: E402
from yue2.tokenization_yue2 import YuE2TextTokenizer  # noqa: E402


class W:
    def __init__(self, path):
        self.f = open(path, "wb")

    def i32(self, v): self.f.write(struct.pack("<i", int(v)))
    def u64(self, v): self.f.write(struct.pack("<Q", int(v)))
    def f32(self, v): self.f.write(struct.pack("<f", float(v)))
    def f64(self, v): self.f.write(struct.pack("<d", float(v)))

    def str(self, s):
        b = s.encode("utf-8")
        self.i32(len(b))
        self.f.write(b)

    def i32list(self, xs):
        xs = [int(x) for x in xs]
        self.i32(len(xs))
        self.f.write(struct.pack("<%di" % len(xs), *xs))

    def f32arr(self, t):
        a = t.detach().float().cpu().contiguous().numpy().astype("<f4")
        self.f.write(a.tobytes())

    def close(self): self.f.close()


ABC = "X:1\nT:Neon\nM:4/4\nL:1/8\nK:C\n\"C\" E2 G2 c2 G2 | \"Am\" A2 c2 e2 c2 |\n\"F\" F2 A2 c2 A2 | \"G\" G2 B2 d2 B2 |]\n"


def load_request():
    import json
    with open(os.path.join(MODEL, "examples", "tonight-awake.json"), encoding="utf-8") as f:
        return json.load(f)


def sampling_args(w, s):
    w.f32(s.temperature); w.f32(s.top_p); w.i32(s.top_k); w.f32(s.repetition_penalty)
    w.i32(s.penalty_window); w.i32(s.min_tokens); w.i32(s.max_tokens)


def phase_range(phase):
    return (0, P.ABC_END + 1) if phase == "abc" else (P.MUSIC_END, P.CODEC_OFFSET + P.CODEC_SIZE)


def protocol_fixture(tok):
    w = W(os.path.join(FIX, "yue2_protocol.bin"))
    w.i32(0x31505559)
    req = load_request()
    strings = ["hello world", "  leading spaces and\ttabs\n\nnewlines  ", "数字 12345 and 3.14159",
               req["style"], req["lyrics"], ABC, "é NFC test café", "<|endoftext|> not special",
               "emoji \U0001F3B5\U0001F3B6 mixed 中文English"]
    w.i32(len(strings))
    for s in strings:
        w.str(s)
        w.i32list(tok.encode(s))
    prompts = [(req["style"], req["lyrics"], "full", ABC), ("acoustic folk", "[verse]\nla la la\n", "melody", ABC),
               ("lofi", "[chorus]\nhey\n", "off", "")]
    w.i32(len(prompts))
    for style, lyrics, cot, abc in prompts:
        r = P.SongRequest(style=style, lyrics=lyrics, cot=cot)
        w.str(style); w.str(lyrics); w.i32({"off": 0, "melody": 1, "full": 2}[cot]); w.str(abc)
        w.i32list(P.token_prefixes(r, tok))
        ids = tok.encode(abc) if cot != "off" else []
        w.i32list(ids)
        w.i32list(P.token_prefixes(r, tok, ids if cot != "off" else None))
        w.i32list(P.negative_prefix(r, tok, ids if cot != "off" else None))
        w.str(r.text())
    ranges = [(100, 900), (5400, 1200), (20000, 1500), (1, 24000)]
    w.i32(len(ranges))
    for frames, pt in ranges:
        w.i32(frames); w.i32(pt)
        w.i32list([v for ab in P.chunk_ranges(frames, pt) for v in ab])
    seed, n = 12300, 64 * 300
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((n // 64, 64), dtype=torch.float32, device="cpu", generator=g)
    w.u64(seed); w.i32(n); w.f32arr(noise)

    gen = torch.Generator().manual_seed(7)
    cases = []
    abc_s = P.GenerationConfig().abc
    sem_s = P.GenerationConfig().semantic
    rnd_hist = lambda lo, hi, k: torch.randint(lo, hi, (k,), generator=gen).tolist()
    cases.append(("abc", abc_s, 10, False, 1.0, rnd_hist(0, 2000, 150)))
    cases.append(("abc", abc_s, 40, False, 1.0, rnd_hist(0, 400, 120)))
    cases.append(("semantic", sem_s, 300, False, 1.0,
                  [P.CODEC_OFFSET + v for v in rnd_hist(0, 64, 80)]))
    cases.append(("semantic", sem_s, 5, False, 1.0, [P.CODEC_OFFSET + v for v in rnd_hist(0, 32768, 5)]))
    cases.append(("semantic", P.Sampling(0.0, 0.95, 100, 1.2, 50, 2, 64), 30, False, 1.0,
                  [P.CODEC_OFFSET + v for v in rnd_hist(0, 128, 30)]))
    cases.append(("semantic", sem_s, 250, True, 1.01, [P.CODEC_OFFSET + v for v in rnd_hist(0, 256, 60)]))
    cases.append(("semantic", P.Sampling(0.8, 0.5, 100, 1.1, 20, 0, 64), 3, False, 1.0,
                  [P.CODEC_OFFSET + v for v in rnd_hist(0, 64, 40)]))
    w.i32(len(cases))
    for phase, s, step, legacy, cfg, hist in cases:
        lo, hi = phase_range(phase)
        full = (torch.randn(1, P.VOCAB_SIZE, generator=gen) * 4.0)
        if cfg != 1.0:
            cond = full.to(torch.bfloat16)
            uncond = (full + torch.randn(1, P.VOCAB_SIZE, generator=gen) * 0.5).to(torch.bfloat16)
            logits = uncond + cfg * (cond - uncond)
        else:
            cond = full.to(torch.bfloat16).float() if legacy else full
            uncond = None
            logits = cond.to(torch.bfloat16) if legacy else cond
        scores = distribution(logits, s, hist, step, phase, legacy_off=legacy)
        outside = torch.cat([scores[0, :lo], scores[0, hi:]])
        assert bool(torch.isinf(outside).all()), "reference kept an id outside the phase range"
        w.i32(0 if phase == "abc" else 1)
        sampling_args(w, s)
        w.i32(step); w.i32(int(legacy)); w.f32(cfg)
        w.i32list(hist)
        w.i32(hi - lo)
        w.f32arr(cond[0, lo:hi])
        if cfg != 1.0:
            w.f32arr(uncond[0, lo:hi])
        w.f32arr(logits[0, lo:hi])
        w.f32arr(scores[0, lo:hi])
    w.close()


def model_fixture(tok):
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.nar import CachedNAR, song_chunks
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = YuE2ForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
                                            low_cpu_mem_usage=True).eval().to("cuda")
    req = load_request()
    full = P.SongRequest(style=req["style"], lyrics=req["lyrics"], cot="full", seed=req["seed"])
    plan_prefix = P.token_prefixes(full, tok)
    abc_ids = tok.encode(ABC)
    sem_prefix = P.token_prefixes(full, tok, abc_ids)
    w = W(os.path.join(FIX, "yue2_model.bin"))
    w.i32(0x314d5559)
    cases = [plan_prefix, plan_prefix[:37], sem_prefix]
    w.i32(len(cases))
    with torch.inference_mode():
        for ids in cases:
            out = model(torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
            w.i32list(ids)
            w.f32arr(out.logits[0, -1])

    off = P.SongRequest(style=req["style"], lyrics=req["lyrics"], cot="off", seed=req["seed"])
    off_prefix = P.token_prefixes(off, tok)
    off_neg = P.negative_prefix(off, tok)
    streams = [
        ("abc", plan_prefix, [], 1.0, False, P.Sampling(0.0, 0.9, 30, 1.005, 100, 32, 48)),
        ("semantic", sem_prefix, [], 1.0, False, P.Sampling(0.0, 0.95, 100, 1.2, 50, 32, 64)),
        ("semantic", off_prefix, off_neg, 1.01, True, P.Sampling(0.0, 0.95, 100, 1.2, 50, 32, 64)),
    ]
    sem_tokens = None
    for phase, prefix, neg, cfg, legacy, s in streams:
        hist, _, trunc = generate_tokens(model, prefix, s, req["seed"], phase,
                                         negative=neg if cfg != 1.0 else None, cfg_scale=cfg, legacy_off=legacy)
        w.i32(0 if phase == "abc" else 1)
        w.i32list(prefix); w.i32list(neg); w.f32(cfg); w.i32(int(legacy))
        sampling_args(w, s)
        w.i32list(hist); w.i32(int(trunc))
        if phase == "semantic" and cfg == 1.0:
            sem_tokens = hist
        print(phase, cfg, len(hist), "tokens", hist[:12], file=sys.stderr)

    codec = [t - P.CODEC_OFFSET for t in sem_tokens]
    chunk = song_chunks(sem_prefix, codec, req["seed"])[0]
    engine = CachedNAR(model, chunk)
    raw = torch.logit(torch.tensor(0.75, dtype=torch.float64)).clamp(-20, 20).item()
    v = engine.velocity(chunk.noise.to(device="cuda", dtype=torch.bfloat16), raw)
    steps = 32
    lat = engine.solve(steps)
    w.i32list(chunk.ar_tokens); w.i32(len(codec)); w.f32arr(chunk.noise); w.f64(raw)
    w.f32arr(v); w.i32(steps); w.f32arr(lat)
    engine.close()
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(FIX, exist_ok=True)
    tok = YuE2TextTokenizer(os.path.join(MODEL, "qwen.tiktoken"))
    protocol_fixture(tok)
    print("wrote yue2_protocol.bin", file=sys.stderr)
    if not a.protocol_only:
        model_fixture(tok)
        print("wrote yue2_model.bin", file=sys.stderr)


if __name__ == "__main__":
    main()
