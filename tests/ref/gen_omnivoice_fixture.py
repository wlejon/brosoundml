#!/usr/bin/env python3
"""Regenerate the OmniVoice reference fixtures from the *genuine upstream*
PyTorch implementation (github.com/k2-fsa/OmniVoice, omnivoice/models/omnivoice.py)
so the C++ port (src/omnivoice.cpp) can be validated stage by stage.

Nothing in the pipeline is re-implemented here: every number comes out of the
upstream code, captured by hooking the functions it calls (see HOOKS below).
The only behavioural change is that Gumbel noise is made deterministic by
replacing ``torch.rand_like`` with a constant 0.5 during generation, which
turns the noise into a constant offset and therefore leaves the top-k ordering
exactly equal to the noise-free ordering.

Requirements: the weights under weights/omnivoice (config.json, tokenizer.json,
model.safetensors, audio_tokenizer/), an NVIDIA GPU, torch + transformers>=5.3
(tested: torch 2.9.1+cu130, transformers 5.9.0, tokenizers 0.22.1, pydub 0.25.1),
and the upstream source tree.  The source is taken from $OMNIVOICE_SRC when
set, else tests/ref/_cache/omnivoice-src (git-cloned at the pinned commit on
first run).  Runs in ~1 min; everything is float32 on CUDA with TF32 off and
cuDNN deterministic.

Outputs (tests/fixtures/, one file per part so a consumer can skip parts):
    omnivoice_tokens.bin     Part A  tokenizer
    omnivoice_prompt.bin     Part B  prompt assembly + duration estimator
    omnivoice_forward.bin    Part C  LM forward, step 0 of generation
    omnivoice_generate.bin   Part D  deterministic 16-step generation + decode
    omnivoice_clone.bin      Part E  voice clone prompt + cloned generation
    omnivoice_post.bin       Part F  post-processing helpers in isolation
    omnivoice_fixture_manifest.txt   human-readable case list + versions

Binary conventions (all little-endian):
    i32        int32
    f32 / f64  float32 / float64
    u8         uint8
    str        i32 nbytes, then nbytes of UTF-8 (nbytes == -1 encodes None,
               with no payload)
    i32list    i32 n, then n x i32
    f32list    i32 n, then n x f32
    C = 8 codebooks, V = 1025 (1024 codes + MASK id 1024), H = 1024 hidden.
    Every [C, T] grid is codebook-major (c*T + t); [N, C, T] is n-major.

---------------------------------------------------------------------------
Part A — omnivoice_tokens.bin
    i32  n_special;  per special token: str name, i32 id
    i32  n_strings;  per string:
         str  s
         i32list plain      # tokenizer.encode(s, add_special_tokens=False)
         i32list tagaware   # _tokenize_with_nonverbal_tags(s, tokenizer)[0]

Part B — omnivoice_prompt.bin
    i32  C, mask_id, frame_rate
    i32  n_prompt_cases;  per case:
         str text; str lang_in; str instruct_in; str ref_text_in
         i32 ref_kind          # 0 none, 1 encoded (Part E prompt), 2 synthetic
         i32 denoise; f32 speed_in (0 = None); f32 duration_in (0 = None)
         str lang_resolved; str instruct_resolved     # what _preprocess_all made
         str style_text; str full_text; str wrapped_text
         i32 T_target          # task.target_lens[0]
         f32 speed_ratio       # task.speed[0] (1.0 when task.speed is None)
         i32 n_ref; i32 ref_codes[C*n_ref]            # the ref tokens used
         i32 n_style; i32 n_text                      # style ids, wrapped-text ids
         i32 c_len; i32 cond_ids[C*c_len]             # _prepare_inference_inputs
         u8  layout[c_len]     # 0 text (style+text), 1 ref audio, 2 target
         i32list cond_text_ids # == cond_ids[0, :n_style+n_text]
         i32list uncond_text_ids   # always empty: the uncond row is target-only
         i32 u_len             # == T_target
    i32  n_dur_cases;  per case:
         str text; str ref_text_in; i32 n_ref_in (0 = None); f32 speed
         str ref_text_used; i32 n_ref_used     # after the upstream fallback
         f64 target_weight; f64 ref_weight; f64 raw_estimate   # estimate_duration()
         i32 est_tokens        # _estimate_target_tokens()

Part C — omnivoice_forward.bin
    i32  C, V, H, T, c_len, u_len(=T)
    str  text; str lang_resolved; str instruct_resolved
    str  style_text; str wrapped_text
    i32  cond_ids[C*c_len]; u8 cond_audio_mask[c_len]
    i32  uncond_ids[C*u_len]            # all MASK
    f32  logits[2*C*T*V]                # step-0 logits at the target positions,
                                        # row 0 cond, row 1 uncond
    f32  mask_frame_embed[H]            # sum_c audio_embeddings[c*V + MASK]
    i32  n_text_embed(=4); i32 ids[4]; f32 text_embed[4*H]   # first 4 cond ids
    f32  cond_inputs_embeds[c_len*H]    # _prepare_embed_inputs output, cond row
    f32  hidden[2*T*H]                  # llm last_hidden_state at the target
                                        # positions (post final norm), cond/uncond

Part D — omnivoice_generate.bin
    i32  C, V, T, sr
    f32  guidance_scale, t_shift, layer_penalty_factor, position_temperature,
         class_temperature, gumbel_u, gumbel_noise_const
    i32  num_step; i32 k[num_step]      # positions unmasked per step
    i32  codes[C*T]                     # final codes
    i32  unmask_step[C*T]               # step index at which (c,t) was unmasked
    i32  n_score_steps(=num_step)
    f32  scores[n_score_steps*C*T]      # per step: confidence after the layer
                                        # penalty, before noise / masking
    i32  pred[n_score_steps*C*T]        # per step: argmax token per position
    f32list raw                         # decoded waveform (before post-processing)
    f32list post                        # generate() output (post-processed)
    i32  n_steps; per: str              # post-processing steps applied, in order

Part E — omnivoice_clone.bin
    i32  C, sr, hop_length
    f32list ref_in                      # what went into create_voice_clone_prompt
                                        # (= Part D raw)
    f32  ref_rms
    f32list ref_pre                     # waveform fed to audio_tokenizer.encode
    i32  T_ref; i32 ref_codes[C*T_ref]
    str  ref_text_in; str ref_text_out  # after add_punctuation
    str  text; str lang_resolved; i32 denoise
    i32  est_tokens                     # _estimate_target_tokens with the ref
    i32  c_len, T
    i32  cond_ids[C*c_len]; u8 cond_audio_mask[c_len]
    i32list uncond_text_ids             # empty
    i32  num_step; i32 k[num_step]
    i32  codes[C*T]; i32 unmask_step[C*T]
    f32list raw; f32list post
    i32  n_steps; per: str

Part F — omnivoice_post.bin
    i32  sr
    i32  n_signals;  per signal:
         str name; f32list x
         i32 n_rs; per: i32 mid_sil, lead_sil, trail_sil; f32list out
         i32 n_gain; per: i32 kind (0 peak->0.5, 1 rms match); f32 ref_rms; f32list out
         i32 n_fp; per: f32 pad_duration, fade_duration; f32list out

HOOKS (file:line in omnivoice/models/omnivoice.py at the pinned commit):
    OmniVoice.forward             :492  (called per step at :1385)
    OmniVoice._prepare_embed_inputs :470
    OmniVoice.llm (forward hook)  :522  last_hidden_state
    _prepare_inference_inputs     :1198
    _generate_iterative           :1275 return value
    _predict_tokens_with_scoring  :1429
    _gumbel_sample                :1632 (called at :1410)
    torch.topk                    :1417 (k)
    torch.rand_like               :1634 (constant 0.5)
    _combine_text                 :1698, _tokenize_with_nonverbal_tags :1658
    text_tokenizer.__call__       :1231 (style text)
    create_voice_clone_prompt     :729 ; audio_tokenizer.encode :821
    audio_tokenizer.decode        :861 ; _post_process_audio :874
    remove_silence / fade_and_pad_audio  omnivoice/utils/audio.py :150 / :208
"""
import contextlib
import os
import struct
import subprocess
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(os.path.dirname(__file__), "_cache")
WEIGHTS = os.path.join(REPO, "weights", "omnivoice")
OUT = os.path.join(REPO, "tests", "fixtures")

UPSTREAM_URL = "https://github.com/k2-fsa/OmniVoice.git"
UPSTREAM_SHA = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"

# The voice-design case shared by Parts C and D.
TEXT_C = "Nice to meet you, this is a test."
LANG_C = "English"
INSTRUCT_C = "female, young adult, high pitch"
# The clone case (Part E).
TEXT_E = "Hello there, this is a cloned voice speaking."
LANG_E = "English"

GEN_KW = dict(
    num_step=16,
    guidance_scale=2.0,
    t_shift=0.1,
    layer_penalty_factor=5.0,
    position_temperature=5.0,
    class_temperature=0.0,
)
GUMBEL_U = 0.5

SPECIAL_TOKENS = [
    "<|denoise|>", "<|lang_start|>", "<|lang_end|>", "<|instruct_start|>",
    "<|instruct_end|>", "<|text_start|>", "<|text_end|>", "<|endoftext|>",
    "<|im_start|>", "<|im_end|>",
]

# ── Part A strings ────────────────────────────────────────────────────────
TOKEN_STRINGS = [
    # English: contractions, numbers, punctuation
    "Hello, world!",
    TEXT_C,
    TEXT_E,
    "I can't believe it's already 2026.",
    "She said, \"Don't go!\" and he didn't.",
    "The price is $1,234.56 — that's 12% off.",
    "Call me at 555-0199 or e-mail john.doe@example.com.",
    "Mr. Smith vs. Dr. Jones: round 3?",
    "It's 9:45 a.m.; we're late...",
    "WOW!!! This is AMAZING!!!",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines",
    "",
    "a",
    "Mixed 中英文 text with 123 numbers and ,.!? punctuation",
    "ﬁ ligature and ﬀ",
    " non-breaking space",
    # Chinese
    "你好，世界！",
    "今天天气真不错，我们去公园散步吧。",
    "我有３个苹果和10个橘子。",
    "这是（括号）测试",
    # Japanese
    "こんにちは、世界！",
    "私は東京に住んでいます。",
    "カタカナとひらがなと漢字。",
    # Korean
    "안녕하세요, 만나서 반갑습니다.",
    "한국어 텍스트입니다.",
    # Russian
    "Привет, мир!",
    "Я не знаю, что сказать.",
    # Arabic (plain, and with harakat)
    "مرحبا بالعالم",
    "مَرْحَبًا بِالْعَالَم",
    # Hindi
    "नमस्ते दुनिया",
    "मेरा नाम राहुल है।",
    # Thai
    "สวัสดีครับ ยินดีที่ได้รู้จัก",
    # Vietnamese
    "Chào thế giới, tôi tên là Minh.",
    # German / Spanish / French
    "Grüß Gott! Straße, Äpfel, Öl und Übung.",
    "¿Qué tal? ¡Buenos días, señor!",
    "C'est la vie, n'est-ce pas ?",
    # Emoji
    "Hello \U0001f30d! This is fun \U0001f389",
    "\U0001f600\U0001f603\U0001f604 \U0001f44d\U0001f3fd ❤️",
    # Non-verbal tags
    "[laughter]",
    "Haha [laughter] that was funny.",
    "That's so funny[laughter]!",
    "[sigh] I guess so.",
    "Really? [question-ah] I had no idea.",
    "哈哈[laughter]太好笑了。",
    "[laughter][sigh][question-ah]",
    "[confirmation-en] yes. [surprise-wa] wow! [dissatisfaction-hnn]",
    "[question-en][question-oh][question-ei][question-yi][surprise-ah][surprise-oh][surprise-yo]",
    "[unknown-tag] stays plain",
    "[Laughter] is case sensitive",
    # CMU phoneme brackets
    "[B EY1 S]",
    "The [B EY1 S] guitar sounds great.",
    "Read as [R EH1 D] or [R IY1 D]?",
    # Pinyin
    "NI3 HAO3",
    "我们(WO3 MEN5)去了商店。",
    "他的名字叫XING2。",
    # Special tokens
    "<|text_start|>Hello<|text_end|>",
    "<|denoise|>",
    "<|lang_start|>en<|lang_end|><|instruct_start|>None<|instruct_end|>",
    "<|denoise|><|lang_start|>None<|lang_end|><|instruct_start|>" + INSTRUCT_C + "<|instruct_end|>",
    "<|lang_start|>zh<|lang_end|><|instruct_start|>女，青年，高音调<|instruct_end|>",
    "<|text_start|>" + TEXT_C + " " + TEXT_E + "<|text_end|>",
    "<|im_start|>user<|im_end|>",
    "<|endoftext|>",
    "<|text_start|>Haha [laughter] 你好 world.<|text_end|>",
]

# ── Part B prompt-assembly cases ──────────────────────────────────────────
# (text, lang, instruct, ref_text, ref_kind, denoise, speed, duration)
#   ref_kind: None | "encoded" (Part E prompt; ref_text ignored -> prompt's)
#             | ("synthetic", n_frames)
PROMPT_CASES = [
    (TEXT_C, LANG_C, INSTRUCT_C, None, None, True, None, None),
    (TEXT_C, None, None, None, None, True, None, None),
    ("你好，世界！", "Chinese", "female, young adult", None, None, True, None, None),
    ("你好，世界！", "zh", "female, american accent", None, None, True, None, None),
    (TEXT_E, LANG_E, None, None, "encoded", True, None, None),
    (TEXT_E, LANG_E, None, None, "encoded", False, None, None),
    (TEXT_E, None, None, None, "encoded", True, None, None),
    ("こんにちは、世界！", "ja", None,
     "テストです。", ("synthetic", 40), True, None, None),
    ("Haha [laughter] that was funny.", "en", "male, british accent", None, None, True, None, None),
    ("Hello", "English", None, None, None, True, 1.5, None),
    ("Hello world.", "en", None, None, None, True, None, 2.0),
    ("Hello world.", "en", None, None, ("synthetic", 30), True, 0.8, 3.0),
    ("Hello world.", "Klingon", "whisper", None, None, True, None, None),
    ("  multiple   spaces  and\nnewline ", "en", None, None, None, True, None, None),
    ("我 爱 北京 天安门", "zh", None,
     "你好 世界", ("synthetic", 30), True, None, None),
    ("The [B EY1 S] guitar", "en", "elderly, very low pitch", None, None, True, None, None),
    ("Bonjour tout le monde.", "French", "女，少年", None, None, True, None, None),
    ("Short.", "en", None, "A much longer reference transcript than the target.",
     ("synthetic", 120), True, None, None),
]

# ── Part B duration-estimator cases: (text, ref_text, n_ref, speed) ───────
LONG_TEXT = ("This is a deliberately long paragraph of English text that goes on and on, "
             "sentence after sentence, so that the estimated duration crosses the thirty "
             "second chunking threshold used by the upstream generator; it keeps talking "
             "about nothing in particular, adding clauses, commas, and the occasional "
             "semicolon; numbers like 1,024 and 2,048 appear too, as do question marks? "
             "and exclamation marks! It really does not stop until it has said enough "
             "to be considered long by any reasonable definition of the word.")
DURATION_CASES = [
    (TEXT_C, None, None, 1.0),
    (TEXT_E, None, None, 1.0),
    ("Hi", None, None, 1.0),
    ("a", None, None, 1.0),
    ("Hello.", None, None, 1.0),
    ("Hello world.", None, None, 1.0),
    ("Hello world.", None, None, 1.5),
    ("Hello world.", None, None, 0.8),
    ("Hello world.", None, None, 0.0),
    ("Hello world.", None, None, -1.0),
    ("", None, None, 1.0),
    ("   ", None, None, 1.0),
    ("...!!!???", None, None, 1.0),
    ("1234567890", None, None, 1.0),
    ("The year 2026, at 9:45 a.m. sharp.", None, None, 1.0),
    ("你好", None, None, 1.0),
    ("你好，世界！", None, None, 1.0),
    ("今天天气真不错，我们去公园散步吧。", None, None, 1.0),
    ("我有３个苹果和10个橘子。", None, None, 1.0),
    ("こんにちは、世界！", None, None, 1.0),
    ("カタカナとひらがなと漢字。", None, None, 1.0),
    ("안녕하세요, 만나서 반갑습니다.", None, None, 1.0),
    ("Привет, мир!", None, None, 1.0),
    ("مرحبا بالعالم", None, None, 1.0),
    ("مَرْحَبًا بِالْعَالَم", None, None, 1.0),
    ("ــمرحبا", None, None, 1.0),
    ("नमस्ते दुनिया", None, None, 1.0),
    ("สวัสดีครับ ยินดีที่ได้รู้จัก", None, None, 1.0),
    ("Chào thế giới, tôi tên là Minh.", None, None, 1.0),
    ("Grüß Gott! Straße, Äpfel, Öl und Übung.", None, None, 1.0),
    ("γεια σου κόσμε", None, None, 1.0),
    ("שלום עולם", None, None, 1.0),
    ("ሰላም አለም", None, None, 1.0),
    ("ភាសាខ្មែរ", None, None, 1.0),
    ("გამარჯობა", None, None, 1.0),
    ("Hello \U0001f30d! This is fun \U0001f389", None, None, 1.0),
    ("Haha [laughter] that was funny.", None, None, 1.0),
    ("[B EY1 S] NI3 HAO3", None, None, 1.0),
    ("Ｈｅｌｌｏ　Ｗｏｒｌｄ", None, None, 1.0),
    ("\U00020000\U00020001 upper plane", None, None, 1.0),
    ("  　", None, None, 1.0),
    (LONG_TEXT, None, None, 1.0),
    # with references
    (TEXT_E, TEXT_C, 40, 1.0),
    (TEXT_E, TEXT_C, 40, 1.3),
    (TEXT_E, "This is a considerably longer reference sentence.", 60, 1.0),
    ("你好，世界！", "你好世界。", 30, 1.0),
    ("Hi", TEXT_C, 40, 1.0),
    ("Hello world.", "", 40, 1.0),          # empty ref_text -> fallback
    ("Hello world.", TEXT_C, None, 1.0),    # no ref frames -> fallback
    ("Hello world.", TEXT_C, 0, 1.0),       # 0 ref frames -> est 0 -> max(1, .)
    ("Hello world.", "...", 25, 1.0),
    ("Hello world.", "!!!", 1, 1.0),
    (LONG_TEXT, TEXT_C, 40, 1.0),
]


# ── binary writer ─────────────────────────────────────────────────────────
class W:
    def __init__(self, f):
        self.f = f

    def i32(self, *v):
        self.f.write(struct.pack(f"<{len(v)}i", *[int(x) for x in v]))

    def f32(self, *v):
        self.f.write(struct.pack(f"<{len(v)}f", *[float(x) for x in v]))

    def f64(self, *v):
        self.f.write(struct.pack(f"<{len(v)}d", *[float(x) for x in v]))

    def str(self, s):
        if s is None:
            self.i32(-1)
            return
        b = s.encode("utf-8")
        self.i32(len(b))
        self.f.write(b)

    def i32arr(self, a):
        import numpy as np
        a = np.ascontiguousarray(np.asarray(a).reshape(-1), dtype="<i4")
        self.f.write(a.tobytes())

    def f32arr(self, a):
        import numpy as np
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float32).reshape(-1), dtype="<f4")
        self.f.write(a.tobytes())

    def u8arr(self, a):
        import numpy as np
        self.f.write(np.ascontiguousarray(np.asarray(a).reshape(-1), dtype=np.uint8).tobytes())

    def i32list(self, a):
        a = list(a)
        self.i32(len(a))
        self.i32arr(a)

    def f32list(self, a):
        import numpy as np
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        self.i32(a.size)
        self.f32arr(a)


def ensure_upstream():
    src = os.environ.get("OMNIVOICE_SRC")
    if not src:
        src = os.path.join(CACHE, "omnivoice-src")
        if not os.path.isdir(os.path.join(src, "omnivoice")):
            os.makedirs(CACHE, exist_ok=True)
            print(f"cloning {UPSTREAM_URL} -> {src}")
            subprocess.check_call(["git", "clone", "--quiet", UPSTREAM_URL, src])
            subprocess.check_call(["git", "-C", src, "checkout", "--quiet", UPSTREAM_SHA])
    sha = subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"]).decode().strip()
    if sha != UPSTREAM_SHA:
        print(f"WARNING: upstream at {sha}, fixture pinned to {UPSTREAM_SHA}")
    sys.path.insert(0, src)
    return src, sha


# ── recording proxies / hooks ─────────────────────────────────────────────
class TokProxy:
    """Transparent proxy around the HF tokenizer that logs every call string."""

    def __init__(self, tok):
        self._tok = tok
        self.calls = []

    def __call__(self, text, *a, **kw):
        self.calls.append(text)
        return self._tok(text, *a, **kw)

    def __getattr__(self, n):
        return getattr(self._tok, n)


class GenRecorder:
    """Hooks installed around one model.generate() call (B == 1)."""

    def __init__(self, model, ovm):
        import torch
        self.model, self.ovm, self.torch = model, ovm, torch
        self.step = 0
        self.fwd_cond_ids = []      # per step: cond row ids [C, max_c_len]
        self.step0 = {}
        self.pred, self.scores, self.ks = [], [], []
        self.prep = None            # _prepare_inference_inputs (args, out)
        self.combine, self.wrapped = None, None
        self.gen_tokens = None
        self.raw = None
        self.post = None
        self.post_steps = []
        self.llm_hidden = None

    def __enter__(self):
        m, ovm, torch, self_ = self.model, self.ovm, self.torch, self
        self.orig = dict(
            forward=m.forward, embed=m._prepare_embed_inputs, prep=m._prepare_inference_inputs,
            predict=m._predict_tokens_with_scoring, gen_it=m._generate_iterative,
            decode=m.audio_tokenizer.decode, post=m._post_process_audio,
            gumbel=ovm._gumbel_sample, combine=ovm._combine_text, tag=ovm._tokenize_with_nonverbal_tags,
            rs=ovm.remove_silence, fp=ovm.fade_and_pad_audio,
            topk=torch.topk, rand_like=torch.rand_like,
        )
        o = self.orig

        def forward(input_ids, audio_mask, **kw):
            self_.fwd_cond_ids.append(input_ids[0].detach().cpu().clone())
            out = o["forward"](input_ids, audio_mask, **kw)
            if self_.step == 0:
                self_.step0 = dict(
                    input_ids=input_ids.detach().cpu().clone(),
                    audio_mask=audio_mask.detach().cpu().clone(),
                    attention_mask=kw.get("attention_mask").detach().cpu().clone(),
                    logits=out.logits.detach().float().cpu().clone(),
                )
            self_.step += 1
            return out

        def embed(input_ids, audio_mask):
            e = o["embed"](input_ids, audio_mask)
            if self_.step == 0:
                self_.step0_embeds = e.detach().cpu().clone()
            return e

        def prep(*a, **kw):
            out = o["prep"](*a, **kw)
            self_.prep = (a, kw, {k: v.detach().cpu().clone() for k, v in out.items()})
            return out

        def predict(c_logits, u_logits, cfg):
            p, s = o["predict"](c_logits, u_logits, cfg)
            self_.pred.append(p.detach().cpu().clone())
            return p, s

        def gumbel(scores, temperature):
            self_.scores.append(scores.detach().cpu().clone())
            return o["gumbel"](scores, temperature)

        def topk(x, k, *a, **kw):
            if x.dim() == 1:
                self_.ks.append(int(k))
            return o["topk"](x, k, *a, **kw)

        def rand_like(t, **kw):
            return torch.full_like(t, GUMBEL_U)

        def gen_it(task, cfg):
            r = o["gen_it"](task, cfg)
            self_.gen_tokens = [t.detach().cpu().clone() for t in r]
            return r

        def decode(*a, **kw):
            r = o["decode"](*a, **kw)
            self_.raw = r.audio_values[0].detach().cpu().numpy().copy()
            return r

        def post(audio, ref_rms, gen_config):
            self_.post_steps = []
            r = o["post"](audio, ref_rms, gen_config)
            if ref_rms is not None and ref_rms < 0.1:
                gain = f"gain: rms_match  audio *= ref_rms/0.1  (ref_rms={ref_rms!r})"
            elif ref_rms is None:
                gain = "gain: peak_normalize  audio = audio/peak*0.5 (if peak > 1e-6)"
            else:
                gain = f"gain: none  (ref_rms={ref_rms!r} >= 0.1)"
            self_.post_steps.insert(1, gain)
            self_.post = r.copy()
            return r

        def rs(audio, sr, mid_sil=300, lead_sil=100, trail_sil=300):
            self_.post_steps.append(
                f"remove_silence(mid_sil={mid_sil}, lead_sil={lead_sil}, trail_sil={trail_sil}, thresh=-50 dBFS, seek_step=10)")
            return o["rs"](audio, sr, mid_sil=mid_sil, lead_sil=lead_sil, trail_sil=trail_sil)

        def fp(audio, pad_duration=0.1, fade_duration=0.1, sample_rate=24000):
            self_.post_steps.append(
                f"fade_and_pad_audio(pad_duration={pad_duration}, fade_duration={fade_duration}, sample_rate={sample_rate})")
            return o["fp"](audio, pad_duration=pad_duration, fade_duration=fade_duration, sample_rate=sample_rate)

        def combine(text, ref_text=None):
            r = o["combine"](text, ref_text=ref_text)
            self_.combine = r
            return r

        def tag(text, tokenizer):
            r = o["tag"](text, tokenizer)
            self_.wrapped = (text, r.detach().cpu().clone())
            return r

        def llm_hook(mod, args, out):
            if self_.step == 0:
                self_.llm_hidden = out[0].detach().cpu().clone()

        m.forward, m._prepare_embed_inputs, m._prepare_inference_inputs = forward, embed, prep
        m._predict_tokens_with_scoring, m._generate_iterative = predict, gen_it
        m.audio_tokenizer.decode, m._post_process_audio = decode, post
        ovm._gumbel_sample, ovm._combine_text, ovm._tokenize_with_nonverbal_tags = gumbel, combine, tag
        ovm.remove_silence, ovm.fade_and_pad_audio = rs, fp
        torch.topk, torch.rand_like = topk, rand_like
        self.hook_handle = m.llm.register_forward_hook(llm_hook)
        return self

    def __exit__(self, *exc):
        m, ovm, torch, o = self.model, self.ovm, self.torch, self.orig
        for k in ("forward", "_prepare_embed_inputs", "_prepare_inference_inputs",
                  "_predict_tokens_with_scoring", "_generate_iterative", "_post_process_audio"):
            if k in m.__dict__:
                del m.__dict__[k]
        if "decode" in m.audio_tokenizer.__dict__:
            del m.audio_tokenizer.__dict__["decode"]
        ovm._gumbel_sample, ovm._combine_text, ovm._tokenize_with_nonverbal_tags = o["gumbel"], o["combine"], o["tag"]
        ovm.remove_silence, ovm.fade_and_pad_audio = o["rs"], o["fp"]
        torch.topk, torch.rand_like = o["topk"], o["rand_like"]
        self.hook_handle.remove()
        return False

    # derived quantities ---------------------------------------------------
    def geometry(self):
        (a, kw, out) = self.prep
        c_len = out["input_ids"].shape[2]
        T = a[1]
        return c_len, T

    def unmask_grid(self, mask_id):
        import numpy as np
        c_len, T = self.geometry()
        snaps = [ids[:, c_len - T:c_len].numpy() for ids in self.fwd_cond_ids]
        final = self.gen_tokens[0].numpy()
        assert final.shape == (snaps[0].shape[0], T)
        snaps.append(final)
        n = len(snaps) - 1
        grid = np.full(final.shape, -1, dtype=np.int32)
        for s in range(n):
            newly = (snaps[s] == mask_id) & (snaps[s + 1] != mask_id)
            grid[newly] = s
        assert (snaps[0] == mask_id).all()
        assert (final != mask_id).all() and (grid >= 0).all()
        k_from_diff = [int(((snaps[s] == mask_id) & (snaps[s + 1] != mask_id)).sum()) for s in range(n)]
        assert k_from_diff == self.ks, (k_from_diff, self.ks)
        return grid

    def layout(self):
        import numpy as np
        (a, kw, out) = self.prep
        c_len, T = self.geometry()
        am = out["audio_mask"][0].numpy()
        lay = np.zeros(c_len, dtype=np.uint8)
        lay[am] = 1
        lay[c_len - T:] = 2
        return lay


def write_gen_parts(w, rec, cfg, mask_id, sr, with_scores):
    """Common Part D / Part E tail: k, codes, unmask grid, [scores/pred], audio."""
    import numpy as np
    codes = rec.gen_tokens[0].numpy().astype(np.int32)
    grid = rec.unmask_grid(mask_id)
    w.i32(GEN_KW["num_step"])
    w.i32arr(rec.ks)
    w.i32arr(codes)
    w.i32arr(grid)
    if with_scores:
        w.i32(len(rec.scores))
        w.f32arr(np.stack([s[0].numpy() for s in rec.scores]))
        w.i32arr(np.stack([p[0].numpy() for p in rec.pred]).astype(np.int32))
    w.f32list(rec.raw.reshape(-1))
    w.f32list(rec.post.reshape(-1))
    w.i32(len(rec.post_steps))
    for s in rec.post_steps:
        w.str(s)
    return codes, grid


def main():
    if not os.path.exists(os.path.join(WEIGHTS, "model.safetensors")):
        sys.exit(f"missing weights under {WEIGHTS}")
    src, sha = ensure_upstream()

    import numpy as np
    import torch
    import transformers
    import tokenizers
    from importlib.metadata import version as pkg_version
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "this fixture is generated on CUDA"

    import omnivoice.models.omnivoice as ovm
    from omnivoice.models.omnivoice import OmniVoice, VoiceClonePrompt
    from omnivoice.utils import audio as ovaudio

    model = OmniVoice.from_pretrained(WEIGHTS, dtype=torch.float32, device_map="cuda",
                                      attn_implementation="sdpa")
    model.eval()
    tok = model.text_tokenizer
    C = model.config.num_audio_codebook
    V = model.config.audio_vocab_size
    MASK = model.config.audio_mask_id
    H = model.config.llm_config.hidden_size
    sr = model.sampling_rate
    frame_rate = model.audio_tokenizer.config.frame_rate
    hop = model.audio_tokenizer.config.hop_length
    assert (C, V, MASK, H, sr, frame_rate, hop) == (8, 1025, 1024, 1024, 24000, 25, 960)
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    sizes = {}

    def done(path):
        sizes[os.path.basename(path)] = os.path.getsize(path)
        print(f"wrote {path} ({sizes[os.path.basename(path)]} bytes)")

    # ── Part A: tokenizer ─────────────────────────────────────────────────
    path = os.path.join(OUT, "omnivoice_tokens.bin")
    with open(path, "wb") as f:
        w = W(f)
        w.i32(len(SPECIAL_TOKENS))
        for s in SPECIAL_TOKENS:
            sid = tok.convert_tokens_to_ids(s)
            assert isinstance(sid, int) and sid >= 0, s
            w.str(s)
            w.i32(sid)
            manifest.append(f"A special {s!r} = {sid}")
        w.i32(len(TOKEN_STRINGS))
        for s in TOKEN_STRINGS:
            plain = tok.encode(s, add_special_tokens=False)
            tagaware = ovm._tokenize_with_nonverbal_tags(s, tok)[0].tolist()
            w.str(s)
            w.i32list(plain)
            w.i32list(tagaware)
            manifest.append(f"A {s!r}: plain={plain} tagaware={tagaware}")
    done(path)

    # ── Parts C + D: voice-design generation, hooked ──────────────────────
    model.text_tokenizer = TokProxy(tok)
    with GenRecorder(model, ovm) as rec:
        audios = model.generate(text=TEXT_C, language=LANG_C, instruct=INSTRUCT_C,
                                denoise=True, **GEN_KW)
    style_calls = [s for s in model.text_tokenizer.calls if s.startswith(("<|denoise|>", "<|lang_start|>"))]
    model.text_tokenizer = tok
    assert rec.step == GEN_KW["num_step"] and len(rec.ks) == GEN_KW["num_step"]
    assert len(style_calls) == 1
    (pa, pkw, pout) = rec.prep
    c_len, T_C = rec.geometry()
    lang_res, instr_res = pa[4], pa[5]
    style_text = style_calls[0]
    wrapped_text = rec.wrapped[0]
    n_style = len(tok.encode(style_text, add_special_tokens=False))
    cond_ids = pout["input_ids"][0].numpy().astype(np.int32)          # [C, c_len]
    cond_mask = pout["audio_mask"][0].numpy().astype(np.uint8)
    step0 = rec.step0
    assert step0["input_ids"].shape[0] == 2 and step0["input_ids"].shape[2] == c_len
    assert torch.equal(step0["input_ids"][0], pout["input_ids"][0])
    uncond_ids = step0["input_ids"][1, :, :T_C].numpy().astype(np.int32)
    assert (uncond_ids == MASK).all()
    logits0 = torch.stack([step0["logits"][0, :, c_len - T_C:c_len, :],
                           step0["logits"][1, :, :T_C, :]]).numpy()      # [2, C, T, V]
    assert logits0.shape == (2, C, T_C, V)
    hidden0 = torch.stack([rec.llm_hidden[0, c_len - T_C:c_len, :],
                           rec.llm_hidden[1, :T_C, :]]).numpy()
    embeds0 = rec.step0_embeds[0].numpy()                                 # [c_len, H]
    with torch.no_grad():
        emb = model.audio_embeddings.weight
        mask_frame = sum(emb[c * V + MASK] for c in range(C)).detach().cpu().numpy()
        first4 = cond_ids[0, :4].tolist()
        text_emb = model.get_input_embeddings()(torch.tensor([first4], device=model.device))[0].cpu().numpy()
    gumbel_const = float(-np.log(-np.log(GUMBEL_U + 1e-10) + 1e-10))

    path = os.path.join(OUT, "omnivoice_forward.bin")
    with open(path, "wb") as f:
        w = W(f)
        w.i32(C, V, H, T_C, c_len, T_C)
        w.str(TEXT_C); w.str(lang_res); w.str(instr_res)
        w.str(style_text); w.str(wrapped_text)
        w.i32arr(cond_ids)
        w.u8arr(cond_mask)
        w.i32arr(uncond_ids)
        w.f32arr(logits0)
        w.f32arr(mask_frame)
        w.i32(4); w.i32arr(first4); w.f32arr(text_emb)
        w.f32arr(embeds0)
        w.f32arr(hidden0)
    done(path)
    manifest.append(f"C text={TEXT_C!r} lang={LANG_C!r}->{lang_res!r} instruct={INSTRUCT_C!r}->{instr_res!r}")
    manifest.append(f"C style_text={style_text!r} n_style={n_style}")
    manifest.append(f"C wrapped_text={wrapped_text!r}")
    manifest.append(f"C c_len={c_len} T={T_C} (text positions {c_len - T_C}, target {T_C}); cond_ids[0]={cond_ids[0].tolist()}")

    path = os.path.join(OUT, "omnivoice_generate.bin")
    with open(path, "wb") as f:
        w = W(f)
        w.i32(C, V, T_C, sr)
        w.f32(GEN_KW["guidance_scale"], GEN_KW["t_shift"], GEN_KW["layer_penalty_factor"],
              GEN_KW["position_temperature"], GEN_KW["class_temperature"], GUMBEL_U, gumbel_const)
        codes_D, grid_D = write_gen_parts(w, rec, GEN_KW, MASK, sr, with_scores=True)
    done(path)
    raw_D = rec.raw.copy()                       # (1, N)
    post_D = rec.post.copy()
    assert np.array_equal(post_D.reshape(-1), audios[0])
    manifest.append(f"D num_step={GEN_KW['num_step']} k={rec.ks} sum={sum(rec.ks)} (= C*T = {C * T_C})")
    manifest.append(f"D raw samples={raw_D.shape[-1]} ({raw_D.shape[-1] / sr:.3f}s) post samples={post_D.shape[-1]}")
    manifest.append(f"D post steps: {rec.post_steps}")
    manifest.append(f"D codes[:, :6]={codes_D[:, :6].tolist()}")
    manifest.append(f"D unmask_step[:, :6]={grid_D[:, :6].tolist()}")

    # ── Part E: voice clone from Part D's raw waveform ────────────────────
    enc_in = {}
    orig_encode = model.audio_tokenizer.encode

    def encode(x, *a, **kw):
        enc_in["x"] = x.detach().cpu().numpy().copy()
        return orig_encode(x, *a, **kw)

    model.audio_tokenizer.encode = encode
    prompt = model.create_voice_clone_prompt((torch.from_numpy(raw_D), sr), ref_text=TEXT_C,
                                             preprocess_prompt=True)
    del model.audio_tokenizer.__dict__["encode"]
    ref_pre = enc_in["x"].reshape(-1)
    ref_codes = prompt.ref_audio_tokens.detach().cpu().numpy().astype(np.int32)
    T_ref = ref_codes.shape[1]
    assert ref_codes.shape == (C, T_ref)
    if ref_pre.size != T_ref * hop:
        print(f"NOTE: encoder input {ref_pre.size} samples -> {T_ref} frames (not exactly {hop} per frame)")
    est_E = model._estimate_target_tokens(TEXT_E, prompt.ref_text, T_ref, speed=1.0)

    model.text_tokenizer = TokProxy(tok)
    with GenRecorder(model, ovm) as recE:
        audiosE = model.generate(text=TEXT_E, language=LANG_E, voice_clone_prompt=prompt,
                                 denoise=True, **GEN_KW)
    style_calls = [s for s in model.text_tokenizer.calls if s.startswith(("<|denoise|>", "<|lang_start|>"))]
    model.text_tokenizer = tok
    (pa, pkw, pout) = recE.prep
    c_lenE, T_E = recE.geometry()
    assert T_E == est_E and torch.equal(pa[3].cpu(), prompt.ref_audio_tokens.cpu())
    cond_idsE = pout["input_ids"][0].numpy().astype(np.int32)
    cond_maskE = pout["audio_mask"][0].numpy().astype(np.uint8)
    assert np.array_equal(cond_idsE[:, c_lenE - T_E - T_ref:c_lenE - T_E], ref_codes)

    path = os.path.join(OUT, "omnivoice_clone.bin")
    with open(path, "wb") as f:
        w = W(f)
        w.i32(C, sr, hop)
        w.f32list(raw_D.reshape(-1))
        w.f32(prompt.ref_rms)
        w.f32list(ref_pre)
        w.i32(T_ref); w.i32arr(ref_codes)
        w.str(TEXT_C); w.str(prompt.ref_text)
        w.str(TEXT_E); w.str(pa[4]); w.i32(1)
        w.i32(est_E)
        w.i32(c_lenE, T_E)
        w.i32arr(cond_idsE); w.u8arr(cond_maskE)
        w.i32list([])
        codes_E, grid_E = write_gen_parts(w, recE, GEN_KW, MASK, sr, with_scores=False)
    done(path)
    assert np.array_equal(recE.post.reshape(-1), audiosE[0])
    manifest.append(f"E ref_in samples={raw_D.shape[-1]} ref_rms={prompt.ref_rms!r} ref_pre samples={ref_pre.size} T_ref={T_ref}")
    manifest.append(f"E ref_text_out={prompt.ref_text!r}")
    manifest.append(f"E text={TEXT_E!r} lang={pa[4]!r} denoise=True est_tokens={est_E} c_len={c_lenE} T={T_E}")
    manifest.append(f"E style_text={style_calls[0]!r} wrapped_text={recE.wrapped[0]!r}")
    manifest.append(f"E k={recE.ks}; raw samples={recE.raw.shape[-1]} post samples={recE.post.shape[-1]}")
    manifest.append(f"E post steps: {recE.post_steps}")

    # ── Part B: prompt assembly + duration estimator ──────────────────────
    rng = np.random.RandomState(1234)
    path = os.path.join(OUT, "omnivoice_prompt.bin")
    with open(path, "wb") as f:
        w = W(f)
        w.i32(C, MASK, frame_rate)
        w.i32(len(PROMPT_CASES))
        for ci, (text, lang, instruct, ref_text, ref_kind, denoise, speed, duration) in enumerate(PROMPT_CASES):
            vcp, kind = None, 0
            if ref_kind == "encoded":
                vcp, kind = prompt, 1
            elif ref_kind is not None:
                n = ref_kind[1]
                codes = torch.from_numpy(rng.randint(0, V - 1, size=(C, n)).astype(np.int64)).to(model.device)
                vcp, kind = VoiceClonePrompt(ref_audio_tokens=codes, ref_text=ref_text, ref_rms=0.05), 2
            task = model._preprocess_all(text=text, language=lang, voice_clone_prompt=vcp,
                                         instruct=instruct, preprocess_prompt=True,
                                         speed=speed, duration=duration)
            assert task.batch_size == 1
            model.text_tokenizer = TokProxy(tok)
            with GenRecorder(model, ovm) as r:
                inp = model._prepare_inference_inputs(
                    task.texts[0], task.target_lens[0], task.ref_texts[0], task.ref_audio_tokens[0],
                    task.langs[0], task.instructs[0], denoise)
            style_calls = [s for s in model.text_tokenizer.calls if s.startswith(("<|denoise|>", "<|lang_start|>"))]
            model.text_tokenizer = tok
            assert len(style_calls) == 1
            st = style_calls[0]
            n_style = len(tok.encode(st, add_special_tokens=False))
            n_text = r.wrapped[1].shape[1]
            ids = inp["input_ids"][0].cpu().numpy().astype(np.int32)
            am = inp["audio_mask"][0].cpu().numpy()
            cl = ids.shape[1]
            T = task.target_lens[0]
            n_ref = task.ref_audio_tokens[0].shape[1] if task.ref_audio_tokens[0] is not None else 0
            assert cl == n_style + n_text + n_ref + T
            lay = np.zeros(cl, dtype=np.uint8)
            lay[am] = 1
            lay[cl - T:] = 2
            assert (lay[:n_style + n_text] == 0).all()
            w.str(text); w.str(lang); w.str(instruct); w.str(ref_text)
            w.i32(kind, int(denoise)); w.f32(speed or 0.0, duration or 0.0)
            w.str(task.langs[0]); w.str(task.instructs[0])
            w.str(st); w.str(r.combine); w.str(r.wrapped[0])
            w.i32(T)
            w.f32(task.speed[0] if task.speed else 1.0)
            w.i32(n_ref)
            if n_ref:
                w.i32arr(task.ref_audio_tokens[0].cpu().numpy().astype(np.int32))
            w.i32(n_style, n_text)
            w.i32(cl); w.i32arr(ids); w.u8arr(lay)
            w.i32list(ids[0, :n_style + n_text].tolist())
            w.i32list([])
            w.i32(T)
            manifest.append(
                f"B prompt[{ci}] text={text!r} lang={lang!r}->{task.langs[0]!r} instruct={instruct!r}->{task.instructs[0]!r} "
                f"ref={ref_kind!r} ref_text={task.ref_texts[0]!r} denoise={denoise} speed={speed} duration={duration} "
                f"=> style={st!r} full={r.combine!r} n_style={n_style} n_text={n_text} n_ref={n_ref} T={T} "
                f"speed_ratio={task.speed[0] if task.speed else 1.0} c_len={cl}")
        w.i32(len(DURATION_CASES))
        de = model.duration_estimator
        for di, (text, ref_text, n_ref, speed) in enumerate(DURATION_CASES):
            est = model._estimate_target_tokens(text, ref_text, n_ref, speed=speed)
            if n_ref is None or ref_text is None or len(ref_text) == 0:
                ref_used, n_used = "Nice to meet you.", 25
            else:
                ref_used, n_used = ref_text, n_ref
            tw = de.calculate_total_weight(text)
            rw = de.calculate_total_weight(ref_used)
            raw = de.estimate_duration(text, ref_used, n_used)
            w.str(text); w.str(ref_text); w.i32(n_ref or 0); w.f32(speed)
            w.str(ref_used); w.i32(n_used)
            w.f64(tw, rw, raw); w.i32(est)
            manifest.append(f"B dur[{di}] text={text[:40]!r} ref={ref_text!r} n_ref={n_ref} speed={speed} "
                            f"=> tw={tw} rw={rw} raw={raw} est={est}")
    done(path)

    # ── Part F: post-processing helpers in isolation ──────────────────────
    t = np.arange(int(3.0 * sr)) / sr
    synth = np.random.RandomState(0).randn(t.size).astype(np.float32) * 1e-4   # -80 dBFS floor
    seg1 = (t >= 0.4) & (t < 1.2)
    seg2 = (t >= 2.1) & (t < 2.7)
    synth[seg1] += (0.3 * np.sin(2 * np.pi * 440 * t[seg1]) * np.exp(-2.0 * (t[seg1] - 0.4))).astype(np.float32)
    synth[seg2] += (0.25 * np.sin(2 * np.pi * 660 * t[seg2])).astype(np.float32)
    synth = synth[np.newaxis, :]
    signals = [("part_d_raw", raw_D.astype(np.float32)), ("synthetic_3s", synth)]
    rs_variants = [(500, 100, 100), (200, 100, 200), (300, 100, 300)]
    gain_variants = [(0, None), (1, float(prompt.ref_rms)), (1, 0.05)]
    fp_variants = [(0.1, 0.1), (0.0, 0.05), (0.05, 0.0)]
    path = os.path.join(OUT, "omnivoice_post.bin")
    with open(path, "wb") as f:
        w = W(f)
        w.i32(sr)
        w.i32(len(signals))
        for name, x in signals:
            w.str(name); w.f32list(x.reshape(-1))
            w.i32(len(rs_variants))
            for mid, lead, trail in rs_variants:
                y = ovaudio.remove_silence(x, sr, mid_sil=mid, lead_sil=lead, trail_sil=trail)
                w.i32(mid, lead, trail); w.f32list(y.reshape(-1))
                manifest.append(f"F {name} remove_silence({mid},{lead},{trail}): {x.shape[-1]} -> {y.shape[-1]} samples")
            w.i32(len(gain_variants))
            for kind, r in gain_variants:
                if kind == 0:
                    peak = np.abs(x).max()
                    y = x / peak * 0.5 if peak > 1e-6 else x
                else:
                    y = x * r / 0.1
                w.i32(kind); w.f32(r or 0.0); w.f32list(y.reshape(-1))
            w.i32(len(fp_variants))
            for pad, fade in fp_variants:
                y = ovaudio.fade_and_pad_audio(x, pad_duration=pad, fade_duration=fade, sample_rate=sr)
                w.f32(pad, fade); w.f32list(y.reshape(-1))
                manifest.append(f"F {name} fade_and_pad({pad},{fade}): {x.shape[-1]} -> {y.shape[-1]} samples")
    done(path)

    # ── manifest ──────────────────────────────────────────────────────────
    head = [
        "OmniVoice reference fixtures (generated by tests/ref/gen_omnivoice_fixture.py)",
        f"upstream: {UPSTREAM_URL} @ {sha} (pinned {UPSTREAM_SHA}); source dir {src}",
        f"weights: {WEIGHTS} (dtype float32, device cuda, attn_implementation sdpa, TF32 off, cudnn deterministic)",
        f"gpu: {torch.cuda.get_device_name(0)}",
        f"torch {torch.__version__}  transformers {transformers.__version__}  tokenizers {tokenizers.__version__}  "
        f"numpy {np.__version__}  pydub {pkg_version('pydub')}",
        f"model: C={C} V={V} MASK={MASK} H={H} sr={sr} frame_rate={frame_rate} hop={hop} "
        f"llm={model.config.llm_config.model_type} layers={model.config.llm_config.num_hidden_layers}",
        f"generation: {GEN_KW} denoise=True postprocess_output=True pad=0.1 fade=0.1; "
        f"Gumbel u := {GUMBEL_U} (noise const {gumbel_const!r}; scores/temperature + const)",
        "notes: the uncond row carries only the T MASK target frames (no style/text/ref); "
        "remove_silence round-trips through int16 (truncation toward zero, clip [-32768,32767]) so its output is "
        "int16-quantized; -50 dBFS silence threshold, 10 ms seek step; Qwen2 tokenizer adds no BOS/EOS so "
        "tokenizer(s).input_ids == encode(s, add_special_tokens=False)",
        "sizes: " + ", ".join(f"{k}={v}" for k, v in sizes.items()),
        "",
    ]
    mpath = os.path.join(OUT, "omnivoice_fixture_manifest.txt")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(head + manifest) + "\n")
    print(f"wrote {mpath}")
    print(f"T_C={T_C} c_len={c_len} | T_ref={T_ref} T_E={T_E} c_lenE={c_lenE} | ref_rms={prompt.ref_rms}")


if __name__ == "__main__":
    main()
