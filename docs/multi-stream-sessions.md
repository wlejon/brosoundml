# Multi-stream sessions — load weights once, drive N streams

A single loaded model should be able to serve many independent audio streams (or
synthesis voices) at once without copying its weights. A wake word listens to the
mic *and* the system-audio loopback; one phoneme model spots a vocabulary on
several callers; one Kokoro checkpoint speaks for a dozen NPCs. In every case the
weights are read-only and identical across streams — only the per-stream scratch
differs.

This note is the **convention** every model in brosoundml follows so callers get
one mental model. It is a convention, not a base class: the models share too
little data-flow shape (audio→logit, audio→posteriors, audio→tokens, text→audio)
for a single virtual interface to fit without forcing a lowest-common-denominator
API. What they share is an *ownership contract* and a *vocabulary*.

## The contract

Split a model into two parts:

1. **Weights — load once, shared, read-only.** Held behind a
   `std::shared_ptr<const Model>`. Every stream reads the same weight set; nobody
   mutates it. The model's forward methods are `const`.

2. **Session — per-stream mutable scratch, caller-owned.** A plain struct (or a
   thin handle) holding everything that differs between streams: streaming
   ring-caches, a head's pooling ring, a bound voice, a decode KV-cache. One
   `Session` per stream; N streams = one shared model + N sessions.

The forward pass **reads weights and writes only into the session it is handed**,
so two streams over one model never cross-talk.

## Vocabulary

Each model that supports multiple streams exposes:

```cpp
struct ModelSession { /* per-stream scratch on the model's device */ };

class Model {
public:
    // Allocate a fresh per-stream session on the model's device. const — reads
    // only layer shapes, so it is callable through shared_ptr<const Model>.
    ModelSession make_session() const;

    // Advance one stream. Reads weights; writes ONLY into `session`. const.
    // The verb is model-specific (forward_streaming / synthesize / decode), but
    // the session is always the first argument and the method is always const.
    void forward_streaming(ModelSession& session, const Tensor& in, Tensor& out) const;

    // Zero a session for a clean restart (e.g. on a silence boundary). const.
    void reset(ModelSession& session) const;
};
```

- The session type is named **`<Model>Session`** (`PhonemeSession`,
  `BcResnet2dSession`, `KokoroSession`).
- The factory is **`make_session()`**, `const`, returning a session by value
  (move-only — the scratch is device tensors).
- The work method takes **`session` as its first parameter** and is **`const`**.
- **`reset(session)`** zeroes a session in place, `const`.

A sequence of calls on one session matches a one-shot `forward()` over the
concatenated input (within FP rounding). Keep a legacy single-stream overload
(model owns one internal session) only where back-compat needs it; new
multi-stream code uses the session API.

## The consumer pattern

A higher-level detector/spotter/speaker is itself a per-stream consumer: it is
**constructed over a `shared_ptr<const Model>`** and owns exactly one session
plus its own bookkeeping (front-end framing, templates, detector policy, a bound
voice). Build N consumers over one shared model to run N streams.

```cpp
auto net = std::make_shared<const BcResnet2d>(BcResnet2d::load(path, device));
WakeWord mic   (net);   // owns its own BcResnet2dSession
WakeWord loop  (net);   // a second session over the same weights
// net.use_count() == 3; weights held once.
```

`WakeWord`, `PhonemeSpotter`, and `KokoroSession` are the reference consumers.

## Concurrency tiers — each model declares one

Sharing weights is universal; whether sessions may run *at the same time* is not.
A model documents its tier:

- **Concurrent.** The forward writes only into the session and touches no mutable
  model state, so sessions run truly in parallel with zero cross-talk. The
  streaming conv models are here: `PhonemeNet` and `BcResnet2d`. Their tests pin
  the guarantee — two sessions interleaved over one net produce bit-identical
  output to each run standalone (cross-talk diff = 0).

- **Serialized.** Sessions share warmed/captured state and/or the single GPU
  stream, so calls over one model must not overlap — drive them from one worker
  or queue. `Kokoro` is here: on CUDA its LSTM cells replay a single lazily
  captured CUDA graph whose step buffers are shared, and the GPU runs one stream
  regardless of how many sessions exist. The captured graph is voice- and
  length-independent, so *serial* reuse across voices is exact; only *concurrent*
  calls would race. This is the right tier for NPC turn-taking — many bound
  voices over one 82M weight set, speaking when scheduled — and it costs nothing
  the hardware wouldn't already serialize.

Lifting a model from serialized to concurrent means moving its remaining shared
mutable scratch (e.g. the per-cell CUDA-graph step buffers) into the session.
That is a worthwhile upgrade only when the hardware can actually run the streams
in parallel; on a single CUDA stream it buys nothing.

## Where each model stands

| Model            | Session type          | Tier        | Per-stream scratch                         |
|------------------|-----------------------|-------------|--------------------------------------------|
| `BcResnet2d`     | `BcResnet2dSession`   | Concurrent  | conv causal ring-caches + head GAP ring    |
| `PhonemeNet`     | `PhonemeSession`      | Concurrent  | conv causal ring-caches                     |
| `Kokoro`         | `KokoroSession`       | Serialized  | bound voice (one-shot forward; no caches)  |
| `Whisper`        | `WhisperSession`      | Serialized  | decode KV-cache (per 30 s window)          |
| `Parakeet`       | `ParakeetSession`     | Concurrent  | TDT prediction-net LSTM (h, c)             |
| `QwenAsr`        | `QwenAsrSession`      | Concurrent  | Qwen3 decoder KV-cache (per clip)          |
| `Rave`           | — (none needed)       | Concurrent  | none — stateless, see below                |
| `QwenTts`        | —                     | (single)    | Talker + CodePredictor captured-graph step state still on `Impl` — not yet sessioned |

`Whisper` sits at the serialized tier for the same reason as `Kokoro`: its
decoder replays a single lazily-captured CUDA step graph whose buffers are shared
across sessions (re-keyed to each session's cache on switch), so calls must not
overlap on one model. `Parakeet` reaches the concurrent tier — it has no captured
step graph and no `Impl`-resident cache, so once the prediction state lives in the
session nothing mutable is shared and sessions are fully independent.

`QwenAsr` is concurrent too: its `transcribe()` already allocates the Qwen3
decoder KV-cache per call (it never lived on `Impl`), the AuT encoder is a
stateless one-shot forward, and the decoder holds no captured graph. The session
just moves that per-call cache into caller-owned scratch so the convention and the
concurrency guarantee are explicit — `make_session()` / `transcribe(session, …)`
mirror the other models, and the test pins two interleaved sessions to the
single-call token stream.

`Rave` needs **no session type at all**: `encode()`/`decode()` are `const` and
fully stateless — there is no KV-cache, no captured graph, and no RNG on `Impl`
(the stochastic noise/latent-pad branches take their randomness from the caller's
`RaveDecodeOptions`, so they are deterministic given the options). N threads can
already call `decode()` on one shared `Rave` with zero cross-talk. Manufacturing
an empty `RaveSession` would only add ceremony, so the convention is satisfied by
declaring the tier and skipping the struct.

The one remaining autoregressive holdout, `QwenTts`, keeps its Talker and
CodePredictor per-step scratch — each a lazily-captured CUDA graph with baked
device pointers, reached through the `const` `synthesize()` — on the pImpl today,
so it is single-session and lands at the **serialized** tier once sessioned.
Sessioning it is the Whisper move at larger scale: hoist both captured-graph step
states (and the decode KV-cache) off `Impl` into a `QwenTtsSession` threaded
through `synth_core` and its five entry points. That is the natural next step when
multi-voice Qwen synthesis is needed.
