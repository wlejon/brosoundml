# Laya audio: free-form questions about live speech

The companion to [laya-audio.md](laya-audio.md). The adapter (the Qwen3-ASR AuT
encoder at FP16, then an MLP projector, then soft tokens in frozen Laya's
state span) was built and measured for keywords, "speaking" and "word ended".
This document tests the hypothesis that its distinctive strength is
**answering arbitrary questions about live speech** at wake-word latency, for
example:

- intent and routing ("is the user asking to pause the game?");
- help requests, and addressee or command;
- topic, sentiment, and question vs statement;
- agreement or refusal;
- entity mentions ("did they mention a price / a place / a person?").

Every question is a noul question (p = sigmoid((yes − no) / T)). The questions
are asked in batches on every 30 ms hop, with no ASR and no text anywhere in
the loop.

All numbers are from an RTX 4090 pair; another session shared the GPUs at
times, and each latency figure below says under what load it was measured.

## Verdict first

**Where it's strong / where it's not** (numbers in the sections below; "audio"
= the free-form-trained projector `proj_free2_mm.mlp` on multilingual Laya;
"ASR → text" = Parakeet-TDT transcript → English text Laya):

| kind of question | audio adapter | ASR → text Laya | who wins |
|---|---|---|---|
| **Intents it has in-domain audio for** (Timers and Such: timer, alarm, arithmetic; numbers, times) | real-label AUC 0.94–0.97 | 0.84–0.98 | **adapter ties or wins**, at ~16 ms/hop vs ~100 ms per ASR call, and it fires 1–2.5 s *before the speaker finishes* |
| Topic of long-form speech (politics on VoxPopuli vs the rest) | 0.875 | 0.80 | adapter (domain acoustics help it; see caveats) |
| "Addressed to a machine / giving a command" (domain proxy) | 0.67–0.84 | 0.54–0.67 | adapter, but mostly because it hears *close-talk command audio*, which the text does not see |
| Trained intent families without in-domain audio (music, calendar, email, convert, volume, transport on SLURP) | 0.58–0.86 | 0.82–0.98 | **ASR → text**, clearly |
| **Question families never trained** (weather, lights, cooking, news, place, colour) | 0.58–0.85 | 0.83–0.99 | **ASR → text**, by 0.1–0.3 AUC |
| New *phrasings* of trained families, vs the text ceiling | agreement AUC 0.73–0.81 | 0.85–0.99 | ASR → text |
| Dialogue acts in meetings (question, suggestion, opinion) | 0.53–0.65 | 0.58–0.66 | nobody: the ceiling itself is weak on 3 s windows |
| **Tone of voice** (angry, happy, sad, afraid, disgust, calm) | chance (0.46–0.60) | chance (0.39–0.56) | **nobody.** Transcript distillation cannot teach it, and there is no licence-clean emotion set to train it on |

In short:

- **It is not a general "ask anything about speech" engine that beats
  ASR → text Laya.** For an arbitrary semantic question, transcribing and
  asking text Laya is far more accurate: it agrees with the gold-transcript
  ceiling at AUC 0.85–0.99, against 0.69–0.93 for the adapter. It also
  generalises to new questions for free, which the adapter does only
  partially.
- **What the adapter does have:**
  1. A fixed, low cost: 18 questions per 30 ms hop at 15.5 ms p50 / 17.7 ms
     p95. Parakeet costs 73–152 ms per call on the utterance so far, and it
     has to be re-run to track a stream.
  2. Calibrated streaming probabilities with no decode step.
  3. **Early answers** on families it has in-domain audio for. On Timers and
     Such it crosses 0.5 at a median 2.35 s *before the end of speech*, on
     100 % of timer and alarm commands.
  4. Some command/assistant and topic cues the transcript lacks: the
     recording setting and prosody.
- **Product positioning:**
  - An always-on, many-questions-per-hop **pre-filter and early trigger** for
    a *known* set of intents and entities, trained with in-domain audio. It
    wakes the ASR → text path, or pre-empts it.
  - Not a replacement for that path on open-ended questions.
  - Not a tone-of-voice sensor.

## Method

**Question bank** (`tools/laya_audio_questions.tsv`): 131 noul questions in 44
families. Each family is marked:

- **train**: asked in distillation;
- **heldout**: never asked in training, to measure *new questions*;
- **para**: tone of voice, never trained.

Each phrasing is also marked train or heldout: several families have
phrasings never asked in training, to measure *new wordings of a trained
question*.

| group | families |
|---|---|
| intent (train) | alarm, timer, math, convert, music, calendar, email, volume, pause, transport |
| intent (held out) | weather, lights, cooking, news |
| speech act (train) | command, assistant (addressee), question, suggest, opinion, agree, greeting, backchannel |
| speech act (held out) | help, refuse, thanks |
| sentiment (train) | positive, negative, unsure |
| topic (train / held out) | money, design, politics, food, family, travel, nature, war, religion / sport, health, tech |
| entity (train / held out) | number, price, time, person / place, color |
| tone (para) | angry, happy, sad, afraid, disgust, calm, excited, hesitant |

**Three ways to answer a (window, question) pair:**

- **audio**: the projector on the window's encoder latents; no text at all.
- **gold (the ceiling)**: text Laya on `Audio transcript: "<words>"`, using
  the gold words whose midpoint lies in the heard window.
- **ASR → text (the baseline)**: Parakeet-TDT-0.6b-v3 on the same 3 s window,
  then text Laya.

The ceiling and the baseline use English text Laya (ModernBERT-large,
T = 1.9834), the stronger text model. The multilingual one is weaker on the
domain-proxy labels.

**Empty transcripts answer "no".** Asked about `Audio transcript: (no speech)`,
English text Laya says **yes** to 9 of the 131 questions on every empty window:

- "These words are directed at a machine."
- "There is agreement with the previous point."
- "The speaker wants to stop or pause what is playing."
- "…louder / quieter"
- "…scared or afraid"
- "a short acknowledgement"
- "talking to a voice assistant"
- "play music"
- "unit conversion"

A first distillation round learned this: its "voice assistant" question fired
on silent stretches of meetings, and its music intent on the silence before
SLURP commands. Since commit cd5310c, the teacher, the ceiling and the
baseline all answer every content question "no" (logit −12) when the window
has no words. All numbers here are from after that fix.

**Real labels**, where a label set exists. A labelled span counts as heard if
≥ 50 % of it, or ≥ 2 s of it, lies in the window. A window with no overlap at
all is negative, and a partly heard span is not scored.

| set | labels | licence | use |
|---|---|---|---|
| Timers and Such v1.0, test-real (240 commands) | intent (timer / alarm / math / convert), numbers, times | CC0 | train-real is **trained on**; test is eval |
| SLURP test, slurp_real (2,780 recordings) | 18 scenarios / 46 intents, entities (place, colour, person, time, date) | text CC BY 4.0, **audio CC BY-NC 4.0** | **evaluation only** |
| AMI test meetings | dialogue acts (question, suggest, opinion, …), named entities, topic segments | CC BY 4.0 | evaluation labels |
| CREMA-D (7,442 acted clips) | 6 emotions and intensity | ODbL / DbCL | **evaluation only** |
| corpus domain | politics = VoxPopuli; addressee/command = Timers + SLURP vs everything else | n/a | proxy labels, flagged as such |

Licences and the search for a clean emotion set:
`docs/laya-audio-release/DATA_LICENSES.md`.

## 1. Zero-shot: the keyword-trained projector as it is

`proj_mm.mlp` (keywords, speaking, word-end only) on the 8 eval sets, 600
windows × 131 questions each (`ask_zs2_mm_ent.log`). Agreement with the
ceiling is the AUC of the audio logit against the ceiling's yes (p ≥ 0.5):

| set | seen family+phrasing | seen family, new phrasing | new family | ASR → text (all groups) |
|---|---|---|---|---|
| LibriTTS-R dev | 0.65 | 0.71 | 0.57 | 0.96–0.97 |
| AMI test (far-field) | 0.69 | 0.72 | 0.70 | 0.85–0.92 |
| VoxPopuli en | 0.69 | 0.68 | 0.64 | 0.94–0.96 |
| VoxPopuli de | 0.63 | 0.65 | 0.53 | 0.89–0.90 |
| Timers test | 0.71 | 0.65 | 0.63 | 0.99 |
| SLURP test | 0.65 | 0.66 | 0.70 | 0.96 |
| CREMA-D | 0.76 | 0.66 | 0.67 | 1.00 |

**Zero-shot, the adapter barely understands the questions.** Training on
keywords taught it which words were said, not what they mean together. Its
AUC of 0.53–0.76 against the ceiling means it separates yes from no only
weakly. The exceptions are questions that reduce to a keyword. For
intent.alarm, the real-label AUC is 0.945–0.977 against the gold text's
0.925–0.949, because "alarm" is the word. Intent.timer's new phrasing reaches
0.905.

## 2. Distillation training

**Recipe** (`tools/laya_audio_fetch/train_free.sh`):

- Fine-tune `proj_mm.mlp` for 12k steps at lr 2e-4 on the broadened mix
  (`train_mix.sh`'s domains, weights scaled by 0.96), plus Timers and Such
  train-real at 4 %.
- Every window gets its usual keyword / speaking / word-end probes, plus
  **free-form items**: 8 trained families are sampled per window. The English
  text-Laya teacher scores one trained phrasing of each on the window's gold
  transcript. 4 are kept: the teacher's two most-yes and two at random, so
  rare yes answers are not drowned out.
- The student is trained with soft-target BCE of its calibrated probability
  against the teacher's.
- Held-out families, held-out phrasings and tone are never asked.
- Cost: 226 ms/step (teacher 37, Laya forward 37, backward 143 ms, ~138 items
  per step), about 46 min. The free-form loss plateaus at ~0.32 from step 3k.

**No regression on the original tasks** (`laya-audio2/eval_free2_mm.log` vs
`eval_mm.log`: same probes, same seed):

| set | keyword AUC | speaking AUC | word-end AUC |
|---|---|---|---|
| LibriTTS-R dev | 0.9915 → 0.9902 | 0.9950 → 0.9941 | 0.9035 → 0.9071 |
| AMI test | 0.9710 → 0.9701 | 0.9740 → 0.9781 | 0.9108 → 0.9109 |
| VoxPopuli en / de / es / fr | 0.978 / 0.959 / 0.967 / 0.955 → 0.978 / 0.959 / 0.973 / 0.959 | ±0.01 | ±0.01 |
| MSWC en / de / es / fr | 0.960 / 0.916 / 0.933 / 0.901 → 0.961 / 0.919 / 0.940 / 0.905 | flat | flat |

On non-speech, keyword and speaking false-yes rates are unchanged: at p ≥ 0.5
they are 0.0006 / 0.0085 on noise and 0.0006 / 0.0000 on music.

**Trained, agreement with the ceiling** (`ask_free2_mm_ent.log`). The first
number is the audio AUC, the second the ASR → text AUC:

| set | seen family+phrasing | seen family, new phrasing | new family | tone (para) |
|---|---|---|---|---|
| LibriTTS-R dev | 0.84 / 0.97 | 0.78 / 0.97 | 0.75 / 0.96 | 0.69 / 0.97 |
| AMI test | **0.90 / 0.90** | 0.80 / 0.85 | 0.78 / 0.92 | 0.72 / 0.88 |
| VoxPopuli en | 0.89 / 0.95 | 0.80 / 0.94 | 0.75 / 0.96 | 0.65 / 0.99 |
| VoxPopuli de | 0.82 / 0.90 | 0.73 / 0.89 | 0.69 / 0.89 | 0.69 / 0.93 |
| Timers test | 0.93 / 0.99 | 0.81 / 0.99 | 0.80 / 1.00 | n/a |
| SLURP test | 0.82 / 0.97 | 0.76 / 0.96 | 0.82 / 0.96 | 0.67 / 0.98 |
| CREMA-D | 0.87 / 1.00 | 0.75 / 1.00 | 0.76 / 1.00 | 0.48 / 0.99 |

Zero-shot → trained, for seen families: 0.63–0.76 → 0.82–0.93. New
phrasings: 0.65–0.72 → 0.73–0.81. New families: 0.53–0.70 → 0.69–0.82.

The step-12k dev check on the training domains' dev sets agrees:

- seen: 0.83–0.89;
- new phrasing: 0.74–0.84;
- new family: 0.67–0.82.

**So distillation works, and part of it transfers to unseen questions, but it
does not close the gap to ASR → text.** Only on far-field AMI does the adapter
match the ASR route, because there the ASR's own errors pull the text route
down to 0.85–0.92. On noise-only windows the ceiling is always no. The
adapter's mean |Δp| there is 0.006–0.007, so it now stays quiet on
non-speech.

## 3. Real labels

AUC of the logit, pooled over the sets where each rule applies
(`ask_zs2_mm_ent.log`, `ask_free2_mm_ent.log`). Balanced accuracy at p ≥ 0.5
is low for *every* method: all three are under-confident on these rare
positives, so the table ranks by AUC.

| family [split] | zero-shot audio | trained audio | gold text (ceiling) | ASR → text |
|---|---|---|---|---|
| intent.timer (Timers + SLURP) | 0.78 | **0.94** | 0.98 | 0.98 |
| intent.timer, new phrasing | 0.91 | **0.95** | 0.93 | 0.94 |
| intent.alarm | 0.95 | **0.95** | 0.95 | 0.94 |
| intent.alarm, new phrasing | 0.98 | **0.97** | 0.93 | 0.92 |
| intent.math | 0.50 | **0.95** | 0.86 | 0.86 |
| intent.math, new phrasing | 0.51 | **0.96** | 0.94 | 0.95 |
| ent.number (Timers) | 0.83 | **0.92** | 0.88 | 0.85 |
| ent.time (SLURP) | 0.70 | **0.89** | 0.85 | 0.85 |
| topic.politics (VoxPopuli domain) | 0.66 | **0.88** | 0.82 | 0.80 |
| act.command (domain proxy) | 0.56 | 0.72 | 0.65 | 0.64 |
| act.assistant, new phrasing (proxy) | 0.71 | 0.84 | 0.66 | 0.67 |
| intent.convert | 0.61 | 0.78 | 0.92 | 0.91 |
| intent.music | 0.78 | 0.82 | 0.97 | 0.97 |
| intent.calendar | 0.62 | 0.74 | 0.83 | 0.82 |
| intent.email | 0.65 | 0.67 | 0.86 | 0.84 |
| intent.volume | 0.62 | 0.72 | 0.97 | 0.97 |
| intent.transport | 0.67 | 0.64 | 0.89 | 0.87 |
| ent.price | 0.59 | 0.76 | 0.89 | 0.85 |
| ent.person | 0.67 | 0.73 | 0.86 | 0.81 |
| **new family** intent.lights | 0.72 | 0.82 | 0.995 | 0.99 |
| **new family** intent.cooking | 0.67 | 0.77 | 0.985 | 0.98 |
| **new family** intent.weather | 0.62 | 0.73 | 0.91 | 0.89 |
| **new family** intent.news | 0.58 | 0.58 | 0.87 | 0.88 |
| **new family** ent.place | 0.58 | 0.75 | 0.87 | 0.83 |
| **new family** ent.color | 0.78 | 0.85 | 0.99 | 0.99 |
| act.question (AMI DAs) | 0.48 | 0.54 | 0.66 | 0.66 |
| act.suggest (AMI DAs) | 0.56 | 0.65 | 0.65 | 0.64 |
| act.opinion (AMI DAs) | 0.57 | 0.64 | 0.62 | 0.62 |

**Reading it:**

- **Where it has in-domain audio (Timers train-real), the adapter reaches or
  passes the text route.** The families are timer, alarm, arithmetic, numbers
  and times.
  - That includes phrasings it never trained on. "Someone wants to be woken
    up at a certain time" gives 0.97 against the text's 0.93.
  - It passes the text route where the ASR route and even the gold transcript
    stumble on read-out numbers and "what is X plus Y". Math: 0.95 vs 0.86.
- **SLURP intents that were trained only through transcript distillation on
  other domains trail the text route by 0.1–0.25.** Music, calendar, email,
  convert, volume and transport have no SLURP-like audio in training.
- **New families gain from training (+0.05–0.17) but trail by 0.1–0.3.**
- **Where the adapter "wins" on proxies, it is partly a domain cue.**
  Command/assistant and politics are corpus-level labels. The adapter can hear
  a close-talk command recording or a parliament hall, which the transcript
  does not show. That is a real signal in deployment (it hears who is
  addressing the device), but it is not understanding.
- **AMI dialogue acts on 3 s windows are hard for everyone.** The ceiling is
  0.62–0.66, and the trained adapter matches it on suggest and opinion, and
  not on question.

## 4. Tone of voice (paralinguistics)

The CREMA-D actors read 12 fixed, emotionally neutral sentences in 6
emotions, so the words carry *no* emotion and any signal has to come from the
voice.

| question family | zero-shot audio | trained audio | gold text | ASR → text |
|---|---|---|---|---|
| angry | 0.52 | 0.56 | 0.50 | 0.50 |
| happy | 0.52 | 0.60 | 0.50 | 0.50 |
| sad | 0.48 | 0.50 | 0.56 | 0.56 |
| afraid | 0.51 | 0.46 | 0.49 | 0.49 |
| disgust | 0.64 | 0.57 | 0.40 | 0.39 |
| calm (neutral) | 0.44 | 0.57 | 0.51 | 0.53 |

**Verdict: tone questions do not work.** All methods are at chance:

- The text routes cannot hear tone by construction.
- The adapter, though it reads audio, answers them from the words it infers.
- Distillation from transcripts teaches only what is in the words, so it
  cannot fix this.

On meetings the "sounds angry / hesitant" questions correlate with the text
ceiling (up to r 0.64 along the stream), which means they fire on the *words*
"i hate", "i don't think", not on the voice. A trained comparison needs an
emotion set licensed for commercial training. None was found: RAVDESS is NC-SA,
TESS NC-ND, CREMA-D ODbL, and ESD, IEMOCAP and MSP-Podcast are research-only.
The Qwen3-ASR encoder may well carry prosody, so this is a data gap, not
necessarily a model gap.

## 5. Streaming

### Short commands: how early is the answer confident?

`brosoundml_laya_audio_stream utts` streams each utterance hop by hop, with
0.6 s of trailing silence. Its outputs:

- the adapter's first crossing of 0.5 relative to the end of speech;
- the adapter's value at speech end + 0.3 s;
- its maximum over the stream;
- for comparison, Parakeet on the utterance *prefix* ending at −0.9 … +0.3 s,
  then text Laya.

Results: `stream_utts_{timers,slurp}_free2_mm.log`.

| set, family | audio: positives crossing 0.5 | first crossing, rel. to speech end (p10/p50/p90) | negatives crossing | ASR→text yes on prefix ending −0.9 / −0.3 / +0.3 s |
|---|---|---|---|---|
| Timers timer (80) | 100 % | −3.6 / **−2.35** / −1.6 s | 42 % * | 0.80 / 0.78 / 0.78 |
| Timers alarm (40) | 100 % | −3.0 / **−1.6** / −1.0 s | 40 % * | 0.65 / 0.78 / 0.80 |
| Timers math (40) | 90 % | −3.6 / −2.45 / +0.1 s | 18 % | 0.93 / 0.98 / 0.98 |
| Timers convert (80) | 50 % | −4.2 / −3.0 / −2.0 s | 0 % | 0.69 / 0.75 / 0.76 |
| SLURP alarm (10) | 100 % | −2.3 / −0.67 / −0.3 s | 3.8 % | 0.30 / 0.50 / 0.60 |
| SLURP email (28) | 61 % | −4.1 / −1.6 / −0.5 s | 1.1 % | 0.18 / 0.39 / 0.50 |
| SLURP music (17) | 65 % | −2.5 / −1.4 / −0.6 s | 10.6 % | 0.47 / 0.47 / 0.59 |
| SLURP lights (13, new family) | 85 % | −2.3 / −0.9 / 0.0 s | 22 % | 0.46 / 0.69 / 0.69 |
| SLURP time mention (51) | 53 % | −1.3 / −0.4 / +0.2 s | 7.2 % | 0.28 / 0.37 / 0.33 |
| all Timers | 82 % | −3.8 / −2.35 / −1.2 s | 22 % | 0.76 / 0.80 / 0.81 |
| all SLURP (18 questions) | 36 % | −2.8 / −0.8 / 0.0 s | 3.9 % | 0.24 / 0.34 / 0.38 |

\* Timers negatives for "timer" are mostly alarm commands and vice versa. The
two intents share the words "set … for … minutes / o'clock", so the adapter
fires on both at 0.5. AUC separates them (stream-max 0.99 and 0.95); a
threshold at p = 0.5 does not.

Where it is trained with in-domain audio, **the adapter's answer is there
1.6–2.4 s before the speaker stops**, which is about when the intent words
("set a timer", "wake me") have been said and the duration or time has not. The ASR route needs the words first and costs 104 ms per call on
Timers (73 ms on SLURP, p50), and the whole prefix must be re-transcribed to
follow the stream. On SLURP families without in-domain audio, the adapter
crosses 0.5 on only a third of positives. Its AUC is fine (0.82 overall) but
it is under-confident, so a per-question threshold is needed.

The ASR route's absolute recall at p ≥ 0.5 is also low on SLURP (0.38 at the
end). Text Laya is conservative on terse commands, so both routes need a
per-question threshold in a product.

### Meetings: 18 questions per hop

`brosoundml_laya_audio_stream meeting` streams AMI SDM audio from 300 to
900 s, with the 18-question panel in `tools/laya_audio_panel.txt` on every
30 ms hop. It writes a per-hop trace TSV (`stream_<meeting>_free2_mm.tsv`,
audio and gold-ceiling columns) and lists the top yes-runs with their words.

**Per-hop latency, 18 questions:**

| meeting | total per hop, p50 / p95 / max | encode + project | Laya |
|---|---|---|---|
| ES2004c | 15.5 / 17.7 / 43.9 ms | 7.6 ms | 7.8 ms |
| TS3003c | 15.4 / 17.4 / 47.5 ms | 7.5 ms | 7.9 ms |

Both ran on GPU 0 while GPU 1 ran another eval. The first run, under
contention on the same GPU, measured 15.9 / 18.0 ms. That is the same budget
as ten keyword questions ([laya-audio.md](laya-audio.md#latency)), inside the
30 ms hop, so latency has not regressed. Parakeet on a 3 s window costs
82–145 ms (`ask_*` logs).

**The trace against what is said** (ES2004c and TS3003c; p = peak):

- Sensible hits:
  - **number**: "wind it for two minutes" 0.93; "the numbers zero to nine" 0.93;
  - **colour**: "the grey black colour" 0.97, "a wood like colour" 0.97;
    along TS3003c, r = 0.82 with the ceiling and AUC 0.94;
  - **negative**: "i hate those little things" 0.68;
  - **disagree**: "i don't think it's really an option" 0.52, "i don't
    think" 0.77;
  - **agree**: "…on the side okay" 0.81, "the form of it so okay" 0.73;
  - **command**: "let's start from the inside and work our way out" 0.69,
    "we would advise to bring two" 0.71;
  - **question**: "anyone has any questions about that" 0.63;
  - **tech topic**: "energy like in a watch which you just shake" 0.79.
- Misfires:
  - **colour** on "you can't get the leverage on them" 0.92 and "most current
    remotes" 0.94 (ES2004c: r = 0.01 with the ceiling there);
  - **pause the game** on "okay well i'll move on" 0.61;
  - **voice assistant** at 0.55–0.65 on "…take into consideration size";
  - **tech topic** yes on 7.5–15 % of hops against the ceiling's 18–21 %.
- Along-stream agreement with the ceiling for the semantic questions is
  r 0.14–0.47 and AUC 0.57–0.88, with colour on TS3003c the outlier at
  r 0.82 and AUC 0.94. The speaking question is not comparable: text Laya
  cannot answer it.
- The **"voice assistant" misfires on silence are gone** with the empty-
  transcript fix: yes on 3.4 % of hops → 0.4 %. The yes runs before the fix
  had no words in them at all.

## Reproduce

```bash
cd D:/projects/brosoundml
# data (Timers is trained on; SLURP and CREMA-D are evaluation-only)
bash tools/laya_audio_fetch/fetch_timers.sh
bash tools/laya_audio_fetch/fetch_slurp.sh
bash tools/laya_audio_fetch/fetch_cremad.sh
for s in timers cremad slurp ami; do bash tools/laya_audio_fetch/build_questions.sh $s; done
# zero-shot and trained free-form eval (audio / gold / ASR, 600 windows per set)
TEXT_LAYA=D:/projects/laya TEXT_T=1.9834 bash tools/laya_audio_fetch/ask_mix.sh zs2_mm_ent \
    D:/projects/brosoundml-data/laya-audio2/proj_mm.mlp D:/projects/laya/multilingual 1.0
bash tools/laya_audio_fetch/train_free.sh free2_mm D:/projects/laya/multilingual \
    D:/projects/brosoundml-data/laya-audio2/proj_mm.mlp D:/projects/laya 1.9834 1.0
TEXT_LAYA=D:/projects/laya TEXT_T=1.9834 bash tools/laya_audio_fetch/ask_mix.sh free2_mm_ent \
    D:/projects/brosoundml-data/laya-audio3/proj_free2_mm.mlp D:/projects/laya/multilingual 1.0
# regression on keywords / speaking / word-end (TR = the mix's training alignments, see eval_mix.sh)
bash tools/laya_audio_fetch/eval_mix.sh free2_mm $A3/proj_free2_mm.mlp D:/projects/laya/multilingual 1.0 "$TR"
# streaming
build-cuda/Release/brosoundml_laya_audio_stream utts --proj $A3/proj_free2_mm.mlp \
    --bank tools/laya_audio_questions.tsv --align $A3/align_timers_test.tsv --labels $A3/labels_timers_test.tsv --asr
build-cuda/Release/brosoundml_laya_audio_stream meeting --proj $A3/proj_free2_mm.mlp \
    --panel tools/laya_audio_panel.txt --meeting ES2004c --ami-align $A2/align_ami_test.tsv \
    --from 300 --to 900 --trace $A3/stream_ES2004c_free2_mm.tsv
```

Artifacts are in `D:/projects/brosoundml-data/laya-audio3/`:

- `proj_free2_mm.mlp`;
- `train_free2_mm.log`;
- the `ask_*.log` files, and `ask_*_{answers,windows}.tsv` per-answer dumps;
- the `stream_*` logs and traces;
- `labels_*`, `align_*`, `*.lac`.

`proj_free_mm.mlp` / `ask_free_mm_ent.log` / `ask_zs_*.log` are from before
the empty-transcript fix, and are superseded.

## What would move it

- **In-domain audio per intent family.** This is the single biggest lever.
  Timers took alarm, timer and math from 0.5–0.8 to ≥ 0.94. A product's own
  command set, recorded or synthesised with licence-clean TTS, is the path.
- **Per-question thresholds or a calibration pass.** AUCs are fine but the
  probabilities are under-confident, so p ≥ 0.5 recalls too little.
- **A licence-clean emotion set**, for tone questions. Without one there is
  nothing to train.
- **A no-speech gate** from the speaking question, for new questions whose
  teacher behaviour on silence is unknown.
