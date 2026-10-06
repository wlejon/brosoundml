---
name: Bug report
about: A model loads wrongly or its audio or transcript differs from the reference, a spotter misfires, a crash, or a test fails
labels: bug
---

**The model:** the checkpoint (Hugging Face repo id, or the converted
directory and the script that made it) and the call: the text or input clip,
voice, language and settings.

**What the reference produces** (the upstream implementation with the same
weights and input; for speech-to-text the transcript, for speech output a
short clip or a description of what is wrong):

**What brosoundml produced instead** (the output, the error, a crash, or the
failing `ctest --output-on-failure` output — paste it):

```
```

**Environment:**
- OS:
- brotensor backend (CPU / CUDA / Metal / Vulkan), GPU and driver version:
- Compiler / toolchain (MSVC / GCC / Clang):
- Called from C++ or from JavaScript (`bro.tts`, `bro.stt`, ...):
- brosoundml commit, and sibling commits if built from siblings:
