#!/usr/bin/env python3
"""Numerically-exact offline reference for RAVE v2 inference, built from the
converted safetensors (NOT the TorchScript). This is the executable spec the
C++ `brosoundml::Rave` port is checked against, stage by stage.

It also self-tests against the original `.ts`: with the streaming caches zeroed,
every DETERMINISTIC stage reproduces the scripted model bit-for-bit (PQMF
analysis/synthesis, encoder, decoder net, waveform/loudness synth), and the
stochastic noise synthesizer matches in distribution. Run:

    python rave_reference.py converted_dir [original.ts]

Findings baked in (verified on magnets_b2048_r48000_z8):
  * The `.ts` is the streaming (cached_conv) causal export. Its conv caches hold
    non-zero warmup state; ZERO every buffer ending in `.cache.pad`, `.cache`, or
    `.downsampling_delay.pad` to get offline causal behavior. cache.padding ==
    dilation*(kernel-1); downsampling_delay/paddings are no-ops here (padding 0).
  * PQMF: causal conv (stride n_band) then `reverse_half` = multiply ODD bands by
    (-1)^(t+1) along time (per-sample, NOT a flat sign flip). Inverse mirrors it.
  * Encoder: [conv, BatchNorm1d(1e-5), LeakyReLU(0.2)] x4 downsample, then
    [conv, LeakyReLU, grouped-conv(groups=2)] -> split (mean, scale).
  * Causal transposed conv = conv_transpose1d then DROP the last (kernel-stride).
  * Decoder: input conv -> 4x [LeakyReLU, convT upsample, 3 residual blocks];
    residual block = x + [leaky, conv(dil), leaky, conv]; dilations 1,3,9.
  * Synth: tanh(wave_branch) * mod_sigmoid(loud_branch) [+ noise_branch].
    mod_sigmoid(x) = 2*sigmoid(x)^2.3 + 1e-7. Noise = fft_convolve(U(-1,1),
    amp_to_impulse_response(mod_sigmoid(net(x)-5))) — stochastic.
  * encode/decode are stochastic in the model (frozen `deterministic` flag):
    z = mean + softplus(scale)*N(0,1); decode pads missing latents with N(0,1)
    and adds the noise synth. The offline default here is deterministic (mean,
    zero-pad); pass add_noise to include the noise synth.

Requires: torch, numpy, safetensors.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


# ── offline-causal primitives (cached_conv with zeroed caches) ───────────────
def causal_conv(x, w, b, stride=1, dilation=1, groups=1):
    left = dilation * (w.shape[-1] - 1)
    return F.conv1d(F.pad(x, (left, 0)), w, b, stride=stride,
                    dilation=dilation, groups=groups)


def causal_convt(x, w, b, stride):
    """Causal transposed conv: full conv_transpose1d, drop the last kernel-stride."""
    k = w.shape[-1]
    y = F.conv_transpose1d(x, w, b, stride=stride)
    return y[..., :y.shape[-1] - (k - stride)]


def leaky(x):
    return F.leaky_relu(x, 0.2)


def mod_sigmoid(x):
    return 2 * torch.sigmoid(x) ** 2.3 + 1e-7


def reverse_half(x):
    t = torch.arange(x.shape[-1])
    sgn = torch.where(t % 2 == 0, -1.0, 1.0)
    y = x.clone()
    y[:, 1::2, :] = y[:, 1::2, :] * sgn
    return y


def amp_to_impulse_response(amp, target):   # verbatim rave.core
    a = torch.view_as_complex(torch.stack([amp, torch.zeros_like(amp)], -1))
    a = torch.fft.irfft(a)
    fs = a.shape[-1]
    a = torch.roll(a, fs // 2, -1) * torch.hann_window(fs, dtype=a.dtype)
    a = F.pad(a, (0, target - fs))
    return torch.roll(a, -fs // 2, -1)


def fft_convolve(signal, kernel):           # verbatim rave.core
    signal = F.pad(signal, (0, signal.shape[-1]))
    kernel = F.pad(kernel, (kernel.shape[-1], 0))
    out = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    return out[..., out.shape[-1] // 2:]


class RaveReference:
    # Fixed RAVE v2 topology for the b2048 ratio (16-band PQMF + 128x compression).
    ENC = [(0, 1, 1), (3, 4, 1), (6, 4, 1), (9, 4, 1), (12, 2, 1), (14, 1, 2)]  # (idx, stride, groups)
    ENC_BN = {0: 1, 3: 4, 6: 7, 9: 10}                                          # conv idx -> BN idx
    DEC_UP = [(1, 4), (3, 4), (5, 4), (7, 2)]                                   # (block idx, stride)
    DEC_RES_DIL = [1, 3, 9]

    def __init__(self, converted_dir):
        self.W = load_file(os.path.join(converted_dir, "model.safetensors"))
        with open(os.path.join(converted_dir, "config.json")) as f:
            self.cfg = json.load(f)

    def w(self, name):
        return self.W[name]

    def b(self, name):
        return self.W.get(name)

    # ── PQMF ──
    def pqmf_forward(self, x):
        return reverse_half(causal_conv(x, self.w("pqmf.forward_conv.weight"),
                                        None, stride=self.cfg["n_band"]))

    def pqmf_inverse(self, x):
        nb = self.cfg["n_band"]
        y = causal_conv(reverse_half(x), self.w("pqmf.inverse_conv.weight"), None) * nb
        y = torch.flip(y, [1]).permute(0, 2, 1)
        y = y.reshape(y.shape[0], y.shape[1], -1, nb).permute(0, 2, 1, 3)
        return y.reshape(y.shape[0], y.shape[1], -1)

    # ── encode (deterministic: posterior mean) ──
    def encode(self, x):
        h = self.pqmf_forward(x)
        for idx, stride, groups in self.ENC:
            h = causal_conv(h, self.w(f"encoder.net.{idx}.weight"),
                            self.b(f"encoder.net.{idx}.bias"), stride, groups=groups)
            if idx in self.ENC_BN:
                p = f"encoder.net.{self.ENC_BN[idx]}"
                h = F.batch_norm(h, self.w(p + ".running_mean"), self.w(p + ".running_var"),
                                 self.w(p + ".weight"), self.w(p + ".bias"), False,
                                 0.0, self.cfg["bn_eps"])
            if idx != 14:
                h = leaky(h)
        mean, scale = torch.split(h, h.shape[1] // 2, 1)
        z = mean - self.w("latent_mean").view(1, -1, 1)
        z = F.conv1d(z, self.w("latent_pca").unsqueeze(-1))
        return z[:, :self.cfg["cropped_latent_size"]], mean, scale

    # ── decode ──
    def _unproject(self, z, pad=None):
        # Pad the cropped latent back to full_latent_size before the inverse PCA.
        # Offline-deterministic decode zero-pads; the real model (and stereo)
        # pads the discarded dims with N(0,1). `pad` (1, full-cropped, T) injects
        # that verbatim so the C++ port matches bit-for-bit.
        full = self.cfg["full_latent_size"]
        if pad is None:
            pad = torch.zeros(z.shape[0], full - z.shape[1], z.shape[2])
        z = torch.cat([z, pad], 1)
        return F.conv1d(z, self.w("latent_pca").t().unsqueeze(-1)) + \
            self.w("latent_mean").view(1, -1, 1)

    def _resblock(self, x, stack, blk, dil):
        p = f"decoder.net.{stack}.net.{blk}.aligned.branches.0."
        h = leaky(x)
        h = causal_conv(h, self.w(p + "1.weight"), self.b(p + "1.bias"), dilation=dil)
        h = leaky(h)
        h = causal_conv(h, self.w(p + "3.weight"), self.b(p + "3.bias"))
        return x + h

    def _decoder_net(self, z5):
        h = causal_conv(z5, self.w("decoder.net.0.weight"), self.b("decoder.net.0.bias"))
        for idx, stride in self.DEC_UP:
            h = causal_convt(leaky(h), self.w(f"decoder.net.{idx}.net.1.weight"),
                             self.b(f"decoder.net.{idx}.net.1.bias"), stride)
            for blk, dil in enumerate(self.DEC_RES_DIL):
                h = self._resblock(h, idx + 1, blk, dil)
        return h

    def _noise_branch(self, h, noise=None):
        for i in (0, 2, 4):
            h = causal_conv(h, self.w(f"decoder.synth.branches.2.net.{i}.weight"),
                            self.b(f"decoder.synth.branches.2.net.{i}.bias"), stride=4)
            if i < 4:
                h = leaky(h)
        amp = mod_sigmoid(h - 5).permute(0, 2, 1)
        amp = amp.reshape(1, amp.shape[1], self.cfg["n_band"], -1)
        ir = amp_to_impulse_response(amp, 64)
        # `noise` (if given) is injected verbatim so the C++ port can match the
        # stochastic branch bit-for-bit; otherwise sample fresh U(-1,1).
        if noise is None:
            noise = torch.rand_like(ir) * 2 - 1
        out = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        return out.reshape(1, self.cfg["n_band"], -1)

    def decode(self, z, add_noise=False, noise=None, latent_pad=None):
        h = self._decoder_net(self._unproject(z, latent_pad))
        wave = causal_conv(h, self.w("decoder.synth.branches.0.weight"),
                           self.b("decoder.synth.branches.0.bias"))
        loud = causal_conv(h, self.w("decoder.synth.branches.1.weight"),
                           self.b("decoder.synth.branches.1.bias"))
        wf = torch.tanh(wave) * mod_sigmoid(loud.reshape(1, 1, -1))
        if add_noise:
            wf = wf + self._noise_branch(h, noise)
        return self.pqmf_inverse(wf)

    def decode_stereo(self, z, pads):
        # RAVE has no stereo decoder: the VST runs the mono decoder once per
        # channel and concatenates (export.py: torch.cat([y[:n], y[n:]], 1)). The
        # channels decorrelate only via each channel's independent N(0,1) latent
        # pad. `pads` is a list of (1, full-cropped, T) tensors, one per channel.
        chans = [self.decode(z, add_noise=False, latent_pad=p) for p in pads]
        return torch.cat(chans, 1)   # (1, channels, L)


def _fresh(ts_path):
    """Load the .ts with all streaming caches zeroed -> offline causal behavior.
    Zero every buffer ending in `.pad` (covers `.cache.pad`, the standalone
    CachedPadding `.pad` used before UpsampleLayer convTs, and
    `.downsampling_delay.pad`) or `.cache`. Missing the bare `.pad` buffers leaves
    nonzero warmup in the cached convTs and corrupts the offline comparison."""
    torch.set_grad_enabled(False)
    m = torch.jit.load(ts_path, map_location="cpu")
    m.eval()
    for name, buf in m.named_buffers():
        if name.endswith((".pad", ".cache")):
            buf.zero_()
    return m


def selftest(ref, ts_path):
    torch.set_grad_enabled(False)
    err = lambda a, b: float((a - b).abs().max())
    torch.manual_seed(0)
    x = (torch.randn(1, 1, 2048 * 8) * 0.1).float()

    e = {}
    pq = ref.pqmf_forward(x)
    e["pqmf.forward"] = err(pq, _fresh(ts_path)._rave.pqmf.forward(x))
    z, mean, scale = ref.encode(x)
    o_mean, o_scale = _fresh(ts_path)._rave.encoder.forward(pq)
    e["encoder.mean"] = err(mean, o_mean)
    e["encoder.scale"] = err(scale, o_scale)

    z5 = ref._unproject(z)
    e["decoder.net"] = err(ref._decoder_net(z5), _fresh(ts_path)._rave.decoder.net.forward(z5))
    wf_det = ref.decode(z, add_noise=False)
    o_det = _fresh(ts_path)._rave.decoder.forward(z5, False)
    e["pqmf.inverse"] = err(wf_det, _fresh(ts_path)._rave.pqmf.inverse(o_det))
    e["decode.deterministic"] = err(wf_det, _fresh(ts_path)._rave.pqmf.inverse(
        _fresh(ts_path)._rave.decoder.forward(z5, False)))

    print("  deterministic stage max-errors:")
    for k, v in e.items():
        print(f"    {k:22s} {v:.2e}")
    det_ok = all(v < 1e-4 for v in e.values())

    # stochastic noise synth: compare RMS distributions
    mine = [float(ref.decode(z, add_noise=True).pow(2).mean().sqrt()) for _ in range(5)]
    orac = [float(_fresh(ts_path)._rave.decode(z).pow(2).mean().sqrt()) for _ in range(5)]
    print(f"  noise (RMS, 5 draws):  mine={[round(v,4) for v in mine]}")
    print(f"                       oracle={[round(v,4) for v in orac]}")
    noise_ok = abs(sum(mine)/5 - sum(orac)/5) < 0.1 * (sum(orac)/5 + 1e-9)

    print("  RESULT:", "PASS — offline encode+decode reproduce the model "
          "(deterministic exact, noise statistically matched)"
          if det_ok and noise_ok else "FAIL")
    return det_ok and noise_ok


def dump_fixtures(ref, out_dir):
    """Write raw little-endian float32 fixtures + meta.json for the C++ test:
    a fixed deterministic input, its deterministic encode latent, and the
    deterministic (no-noise) decode waveform. The C++ Rave port must reproduce
    the latent and waveform from these within tolerance."""
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    ratio = 2048
    x = (torch.randn(1, 1, ratio * 8) * 0.1).float()
    z, _, _ = ref.encode(x)
    y = ref.decode(z, add_noise=False)

    # Noise-branch fixture: a fixed injected white-noise buffer + the noise-on
    # decode it produces. The stochastic branch can't be matched against fresh
    # RNG, so the C++ test injects this same buffer and compares bit-for-bit.
    h = ref._decoder_net(ref._unproject(z))
    Lc = h.shape[-1]                                  # per-band decoder length
    T_n = Lc // 64                                    # noise frames (3x stride-4 convs)
    nb = ref.cfg["n_band"]
    noise_bands = ref.w("decoder.synth.branches.2.net.4.weight").shape[0] // nb
    torch.manual_seed(1234)
    noise = (torch.rand(1, T_n, nb, 64) * 2 - 1).float()
    y_noise = ref.decode(z, add_noise=True, noise=noise)

    # Stereo fixture: two channels decoded from the SAME latent but with
    # independent injected N(0,1) latent pads (the only decorrelation source).
    # Saved as (channels, full-cropped, T) so the C++ test reads channel c's pad
    # at offset c*pad_dims*T; decode_stereo.bin is (1, channels, L).
    pad_dims = ref.cfg["full_latent_size"] - z.shape[1]
    T_lat = z.shape[2]
    n_stereo = 2
    torch.manual_seed(4321)
    pads = torch.randn(n_stereo, pad_dims, T_lat).float()
    y_stereo = ref.decode_stereo(z, [pads[c:c + 1] for c in range(n_stereo)])

    os.makedirs(out_dir, exist_ok=True)
    def save_bin(name, t):
        t.detach().contiguous().numpy().astype("<f4").tofile(os.path.join(out_dir, name))
    save_bin("input.bin", x)
    save_bin("latent.bin", z)
    save_bin("decode_det.bin", y)
    save_bin("noise.bin", noise)             # (1, T_n, nb, 64) -> rows (t*nb+band)
    save_bin("decode_noise.bin", y_noise)
    save_bin("latent_pad.bin", pads)         # (channels, full-cropped, T)
    save_bin("decode_stereo.bin", y_stereo)  # (1, channels, L)
    meta = {
        "input_len":    int(x.shape[-1]),
        "n_latent":     int(z.shape[1]),
        "frames":       int(z.shape[2]),
        "output_len":   int(y.shape[-1]),
        "latent_mean":  float(z.mean()), "latent_std": float(z.std()),
        "output_rms":   float(y.pow(2).mean().sqrt()),
        "noise_frames": int(T_n),
        "noise_bands":  int(noise_bands),
        "ir_target":    64,
        "decode_noise_rms": float(y_noise.pow(2).mean().sqrt()),
        "stereo_channels":  int(n_stereo),
        "pad_dims":         int(pad_dims),
        "decode_stereo_rms": float(y_stereo.pow(2).mean().sqrt()),
        "stereo_lr_l1":      float((y_stereo[:, 0] - y_stereo[:, 1]).abs().mean()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote C++ fixtures to {out_dir}")
    print(json.dumps(meta, indent=2))


# ─────────────────────── FLAT topology executor ─────────────────────────────
#
# The newer (flat-layout) RAVE exports span several architectures (residual
# encoders, Snake, gimbal, amplitude-modulation synth). Rather than hardcode a
# topology, convert-rave.py emits an op-list walked straight off the scripted
# module graph; this class replays it. Op kinds: conv / convT / leaky / snake /
# push / add (push+…+add = a residual block). The deterministic encode (posterior
# mean) and the decoder NETWORK reproduce the `.ts` bit-for-bit; latent-pad
# completion is zero-pad (we don't run RAVE's optional autoregressive prior, the
# same simplification used for the legacy magnets path).

class TopoRave:
    def __init__(self, converted_dir):
        self.W = load_file(os.path.join(converted_dir, "model.safetensors"))
        with open(os.path.join(converted_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.nb = self.cfg["n_band"]
        self.full = self.cfg["full_latent_size"]
        self.crop = self.cfg["cropped_latent_size"]

    def w(self, k):
        return self.W[k]

    def _conv(self, x, op):
        w = self.W[op["w"]]
        b = self.W.get(op["b"]) if op["b"] else None
        g = x.shape[1] // w.shape[1]
        return F.conv1d(F.pad(x, (op["left_pad"], 0)), w, b,
                        stride=op["stride"], dilation=op["dil"], groups=g)

    def _convt(self, x, op):
        # Offline-causal upsample: prepend the UpsampleLayer's CachedPadding
        # (cpad), conv_transpose1d with the graph's padding, then trim the output
        # to exactly L*stride (drops the causal convT's trailing context). Handles
        # both the CachedPadding+convT(pad>0) and bare convT(pad=0) conventions.
        w = self.W[op["w"]]
        b = self.W.get(op["b"]) if op["b"] else None
        L0 = x.shape[-1]
        xp = F.pad(x, (op.get("cpad", 0), 0))
        y = F.conv_transpose1d(xp, w, b, stride=op["stride"], padding=op.get("pad", 0))
        return y[..., :L0 * op["stride"]]

    def run(self, ops, x):
        stack = []
        for op in ops:
            k = op["kind"]
            if k == "conv":    x = self._conv(x, op)
            elif k == "convT": x = self._convt(x, op)
            elif k == "lpad":  x = F.pad(x, (op["n"], 0))
            elif k == "leaky": x = leaky(x)
            elif k == "snake":
                a = self.W[op["alpha"]]; x = x + torch.sin(a * x) ** 2 / (a + 1e-9)
            elif k == "push":  stack.append(x)
            elif k == "add":   x = x + stack.pop()
            else: raise RuntimeError("unknown op " + k)
        return x

    def pqmf_forward(self, x):
        return reverse_half(self._conv(x, self.cfg["pqmf"]["forward"]))

    def pqmf_inverse(self, x):
        nb = self.nb
        y = self._conv(reverse_half(x), self.cfg["pqmf"]["inverse"]) * nb
        y = torch.flip(y, [1]).permute(0, 2, 1)
        y = y.reshape(y.shape[0], y.shape[1], -1, nb).permute(0, 2, 1, 3)
        return y.reshape(y.shape[0], y.shape[1], -1)

    # latent-completion scheme (see convert-rave.py): "prior_pca"/"prior_kld" are
    # Family A (gimbal + autoregressive prior fill); "noise_pca" is Family B
    # (N(0,1) fill, no gimbal). Default to noise_pca for older configs.
    @property
    def mode(self):
        return self.cfg.get("latent_mode", "noise_pca")

    def _kld(self):
        return torch.tensor(self.cfg["kld_idxs"], dtype=torch.long)

    def _gimbal_fwd(self, mean):       # mean*exp(log_a) + b
        return mean * torch.exp(self.w("gimbal.log_a").view(1, -1, 1)) \
            + self.w("gimbal.b").view(1, -1, 1)

    def _gimbal_inv(self, z):          # (z - b) / exp(log_a)
        return (z - self.w("gimbal.b").view(1, -1, 1)) \
            / torch.exp(self.w("gimbal.log_a").view(1, -1, 1))

    def _pca_proj(self, mean):         # encode: pca @ (mean - latent_mean)
        return F.conv1d(mean - self.w("latent_mean").view(1, -1, 1),
                        self.w("latent_pca").unsqueeze(-1))

    def _pca_unproj(self, zfull):      # decode: pca^T @ z + latent_mean
        return F.conv1d(zfull, self.w("latent_pca").t().unsqueeze(-1)) \
            + self.w("latent_mean").view(1, -1, 1)

    def encode(self, x):
        h = self.run(self.cfg["encoder_ops"], self.pqmf_forward(x))   # (1, 2*full, T)
        mean = h[:, :self.full]
        if self.mode.startswith("prior"):       # Family A: gimbal, then crop
            mean = self._gimbal_fwd(mean)
            if self.mode == "prior_pca":
                return self._pca_proj(mean)[:, :self.crop]
            return mean[:, self._kld()[:self.crop]]              # prior_kld: gather
        return self._pca_proj(mean)[:, :self.crop]              # noise_pca: PCA crop

    def _ar_prior_fill(self, z, deterministic=True, gen=None):
        """Family A latent completion: run the autoregressive `prior_net` frame by
        frame to generate the discarded dims. Each step feeds the previous full
        latent frame (z5) into the prior; the kept dims come from the encoded crop
        `z`, the rest from the prior (mean if deterministic, else reparametrised).
        The prior is causal stride-1, so a trailing window of its receptive field
        reproduces each output frame exactly — O(T) instead of O(T^2)."""
        full, crop, T = self.full, self.crop, z.shape[-1]
        kld = self._kld()
        win = 1 + sum(op.get("left_pad", 0)
                      for op in self.cfg["prior_ops"] if op["kind"] == "conv")
        z6 = torch.zeros(1, full, T)
        hist = [torch.zeros(full)]              # hist[t] = last_z fed at frame t (0 init)
        for t in range(T):
            inp = torch.stack(hist[max(0, len(hist) - win):], -1).unsqueeze(0)
            pr = self.run(self.cfg["prior_ops"], inp)           # (1, 2*full, Lw)
            pm = pr[:, :full, -1:]
            if deterministic:
                pad = pm
            else:
                ps = torch.exp(pr[:, full:, -1:])
                pad = pm + torch.randn(pm.shape, generator=gen) * ps
            frame = torch.cat([z[:, :, t:t + 1], pad[:, kld[crop:]]], 1)   # (1, full, 1)
            z6[:, :, t:t + 1] = frame
            hist.append(frame[0, :, 0])
        return z6

    def _synth(self, h):
        if self.cfg["synth_type"] == "amp_mod":
            wave, amp = torch.split(h, h.shape[1] // 2, 1)
            return torch.tanh(wave * torch.sigmoid(amp))
        # "rave": tanh(wave) * mod_sigmoid(repeat(loud, loud_stride))  (noise off)
        s = self.cfg["synth"]
        wave = self._conv(h, s["wave"])
        loud = self._conv(h, s["loud"])
        ls = s.get("loud_stride", 1)
        if ls > 1:
            loud = torch.repeat_interleave(loud, ls, dim=-1)
        return torch.tanh(wave) * mod_sigmoid(loud.reshape(1, 1, -1))

    def decode_from_full(self, z_full):
        """Decode a FULL (full_latent_size) latent through the decoder network +
        synth + PQMF — the parity target (matches scripted decoder.forward)."""
        return self.pqmf_inverse(self._synth(self.run(self.cfg["decoder_ops"], z_full)))

    def decode(self, z, deterministic=True, gen=None, latent_noise=None):
        """Decode a cropped latent back to audio. Family A (prior_*): autoregressive
        prior fills the discarded dims, then scatter (kld) or inverse-PCA (pca) back
        to encoder space and invert the gimbal. Family B (noise_pca): fill with
        N(0,1) (or an injected `latent_noise` for reproducible parity) and
        inverse-PCA. The result feeds the decoder network unchanged."""
        if self.mode.startswith("prior"):
            z6 = self._ar_prior_fill(z, deterministic, gen)
            if self.mode == "prior_pca":
                z7 = self._pca_unproj(z6)
            else:                                       # prior_kld: scatter back
                z7 = z6[:, torch.argsort(self._kld())]
            z10 = self._gimbal_inv(z7)
        else:                                           # noise_pca (Family B)
            if latent_noise is None:
                latent_noise = torch.randn(z.shape[0], self.full - z.shape[1],
                                           z.shape[-1], generator=gen)
            z10 = self._pca_unproj(torch.cat([z, latent_noise], 1))
        return self.decode_from_full(z10)


# ── scripted-deterministic reference (ground truth) ──────────────────────────
# Reproduce the .ts encode/decode with the streaming caches zeroed and ALL
# randomness removed: encode takes the posterior mean; decode takes the AR
# prior's mean (Family A) or an injected noise buffer (Family B); the decoder's
# own noise branch is off. This is the exact, deterministic target for both the
# Python TopoRave and the C++ port.

def _scripted_encode(M, m, topo, x):
    full = topo.full
    pq = M["pqmf"].forward(x)
    try: eo = M["encoder"].forward(pq, False)
    except Exception: eo = M["encoder"].forward(pq)
    mean = eo[:, :full]
    if topo.mode.startswith("prior"):
        mean = M["gimbal"].forward(mean, eo[:, full:])[0]
        if topo.mode == "prior_pca":
            return topo._pca_proj(mean)[:, :topo.crop]
        return mean[:, topo._kld()[:topo.crop]]
    return topo._pca_proj(mean)[:, :topo.crop]


def _scripted_decode(M, m, topo, z, latent_noise=None):
    full, crop, T = topo.full, topo.crop, z.shape[-1]
    if topo.mode.startswith("prior"):
        kld = topo._kld()
        for n, b in m.named_buffers():                  # streaming prior, caches zeroed
            if n.startswith("prior_net") and n.endswith((".pad", ".cache")):
                b.zero_()
        last_z = torch.zeros(1, full, 1)
        frames = []
        for t in range(T):
            pm = torch.chunk(M["prior_net"].forward(last_z), 2, 1)[0]   # (1,full,1)
            z5 = torch.cat([z[:, :, t:t + 1], pm[:, kld[crop:]]], 1)
            frames.append(z5); last_z = z5
        z6 = torch.cat(frames, -1)
        z7 = topo._pca_unproj(z6) if topo.mode == "prior_pca" else z6[:, torch.argsort(kld)]
        z10 = M["gimbal"].inv(z7)
    else:
        z10 = topo._pca_unproj(torch.cat([z, latent_noise], 1))
    try: d = M["decoder"].forward(z10, False)
    except Exception: d = M["decoder"].forward(z10)
    return M["pqmf"].inverse(d)


def selftest_flat(topo, ts_path):
    """End-to-end gate: the TopoRave deterministic encode→decode must reproduce the
    scripted model's own modules run deterministically (prior mean / injected noise,
    decoder noise off). This is the correct target — NOT a self-authored latent
    reformulation, the mistake that let near-silent output pass earlier."""
    torch.set_grad_enabled(False)
    err = lambda a, b: float((a - b).abs().max())
    torch.manual_seed(0)
    x = (torch.randn(1, 1, 2048 * 8) * 0.1).float()
    full = topo.full

    e = {"pqmf.forward": err(topo.pqmf_forward(x), dict(_fresh(ts_path).named_modules())["pqmf"].forward(x))}

    M = dict(_fresh(ts_path).named_modules())
    z_ref = _scripted_encode(M, _fresh(ts_path), topo, x)
    z_mine = topo.encode(x)
    e["encode"] = err(z_mine, z_ref)

    # Deterministic decode end-to-end. Family B needs an injected noise buffer
    # (its fill is intrinsically N(0,1)); Family A is fully deterministic.
    torch.manual_seed(1234)
    noise = None if topo.mode.startswith("prior") else \
        torch.randn(z_mine.shape[0], full - z_mine.shape[1], z_mine.shape[-1])
    m2 = _fresh(ts_path); M2 = dict(m2.named_modules())
    y_ref = _scripted_decode(M2, m2, topo, z_ref, latent_noise=noise)
    y_mine = topo.decode(z_mine, deterministic=True, latent_noise=noise)
    e["decode.e2e"] = err(y_mine, y_ref)

    print("  flat end-to-end max-errors (vs scripted deterministic):")
    for k, v in e.items():
        print(f"    {k:16s} {v:.2e}")
    print(f"  deterministic decode: peak={float(y_mine.abs().max()):.4f} "
          f"rms={float(y_mine.pow(2).mean().sqrt()):.4f}")

    # Stochastic sanity: our randomized decode RMS should track m.decode's.
    g = torch.Generator().manual_seed(7)
    mine_rms = [float(topo.decode(z_mine, deterministic=False, gen=g).pow(2).mean().sqrt())
                for _ in range(4)]
    scr_rms = [float(_fresh(ts_path).decode(z_ref).pow(2).mean().sqrt()) for _ in range(4)]
    print(f"  stochastic RMS: ours={[round(r,3) for r in mine_rms]} "
          f"scripted={[round(r,3) for r in scr_rms]}")

    ok = all(v < 2e-4 for v in e.values())
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def dump_fixtures_flat(topo, out_dir):
    """C++ fixtures for a flat model: input, deterministic encode latent, and the
    deterministic decode the C++ port must reproduce. Family B also saves the
    injected N(0,1) latent-noise buffer (its fill can't be deterministic), which
    the C++ test feeds back in for a bit-exact comparison."""
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    x = (torch.randn(1, 1, 2048 * 8) * 0.1).float()
    z = topo.encode(x)
    is_prior = topo.mode.startswith("prior")
    noise = None
    if not is_prior:
        torch.manual_seed(1234)
        noise = torch.randn(z.shape[0], topo.full - z.shape[1], z.shape[-1]).float()
    y = topo.decode(z, deterministic=True, latent_noise=noise)
    os.makedirs(out_dir, exist_ok=True)
    def save_bin(name, t):
        t.detach().contiguous().numpy().astype("<f4").tofile(os.path.join(out_dir, name))
    save_bin("input.bin", x)
    save_bin("latent.bin", z)
    save_bin("decode_det.bin", y)
    meta = {"format": "flat", "latent_mode": topo.mode,
            "input_len": int(x.shape[-1]),
            "n_latent": int(z.shape[1]), "frames": int(z.shape[2]),
            "output_len": int(y.shape[-1]),
            "output_rms": float(y.pow(2).mean().sqrt())}
    if noise is not None:
        save_bin("latent_noise.bin", noise)   # (1, full-n_latent, frames)
        meta["latent_noise_dims"] = int(noise.shape[1])
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote flat C++ fixtures to {out_dir}")
    print(json.dumps(meta, indent=2))


def _is_flat(converted_dir):
    with open(os.path.join(converted_dir, "config.json")) as f:
        return json.load(f).get("format") == "flat"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--dump":
        # rave_reference.py --dump <converted_dir> <fixture_out_dir>
        cdir = sys.argv[2]
        if _is_flat(cdir):
            dump_fixtures_flat(TopoRave(cdir), sys.argv[3])
        else:
            dump_fixtures(RaveReference(cdir), sys.argv[3])
        return
    if _is_flat(sys.argv[1]):
        topo = TopoRave(sys.argv[1])
        print("config: flat", topo.cfg.get("synth_type"), "z", topo.crop,
              "snake", topo.cfg.get("has_snake"), "gimbal", topo.cfg.get("has_gimbal"))
        if len(sys.argv) >= 3:
            selftest_flat(topo, sys.argv[2])
        else:
            print("(pass the original .ts as arg 2 to run the parity self-test)")
        return
    ref = RaveReference(sys.argv[1])
    print("config:", json.dumps(ref.cfg, indent=2))
    if len(sys.argv) >= 3:
        selftest(ref, sys.argv[2])
    else:
        print("(pass the original .ts as arg 2 to run the parity self-test)")


if __name__ == "__main__":
    main()
