#pragma once

// Qwen3-TTS dual-track autoregressive generation loop — ties the Talker
// (frame-level Qwen3 decoder) and the Code Predictor (codebook-axis depth
// transformer) into the per-frame loop that produces the RVQ code stream.
//
// Per frame: the Talker hidden state yields codebook 0 (codec_head argmax); the
// Code Predictor expands it into codebooks 1..15; the 16 code embeddings are
// summed with the next trailing-text (or tts_pad) embedding to form the next
// Talker input, which is stepped through the Talker's KV cache. The loop stops
// at the codec EOS token or after max_frames.
//
// This is the mechanics of synthesis; assembling the *prefill* embedding stream
// and M-RoPE positions from text / speaker / language tokens (the chat template
// + get_rope_index) is stage 5. Greedy (argmax) decoding for now — matching the
// reference run with do_sample=False; sampling is a later option.

#include "qwen_tts_code_predictor.h"
#include "qwen_tts_talker.h"

#include "brosoundml/audio.h"   // CancelCheck

#include <cstdint>
#include <vector>

namespace brosoundml {

struct QwenTtsGenParams {
    int eos_id     = 0;   // codec EOS in the Talker vocab; stops the loop
    int max_frames = 0;   // hard cap on emitted frames
    int rope_delta = 0;   // M-RoPE position offset for the generation phase

    // Codebook-0 logits processing, matching the upstream talker generate. All
    // default to no-ops so the bare AR loop (stage-4 fixture) is unaffected.
    int   suppress_lo = 0, suppress_hi = 0;  // force logits[lo,hi) (except eos) to -inf
    int   min_frames  = 0;   // suppress eos until this many frames are emitted
    float repetition_penalty = 1.0f;  // penalize already-emitted codebook-0 ids
};

// Run the AR loop. `prefill_embeds` is T*hidden row-major; `pos3T` the 3-axis
// M-RoPE prefill positions (axis-major pos3T[a*T + t]). `trailing_text_hidden`
// is L*hidden (the look-ahead text embeddings consumed one per frame), and
// `tts_pad_embed` (hidden floats) is added once the trailing text is exhausted.
// Emitted frames are appended to `out_frames` as num_code_groups ints each,
// frame-major then codebook-within-frame: [c0..c15, c0..c15, ...]. Returns the
// number of frames.
//
// `cancel` is polled once per frame; when it returns true the loop stops early
// and returns the frames emitted so far (the caller discards them). Empty (the
// default) never cancels, so existing callers are unaffected.
int generate_codes(const QwenTtsTalker& talker, const QwenTtsCodePredictor& cp,
                   const float* prefill_embeds, int T, const int32_t* pos3T,
                   const float* trailing_text_hidden, int L,
                   const float* tts_pad_embed, const QwenTtsGenParams& params,
                   std::vector<int32_t>& out_frames,
                   const CancelCheck& cancel = {});

// Assemble the Talker prefill embedding stream + trailing-text embeddings,
// mirroring the upstream Qwen3TTS generate() (streaming, no voice clone). Covers
// both the CustomVoice and VoiceDesign variants:
//
//   `input_ids`    — the tokenized body chat prompt
//                    (<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n).
//   `instruct_ids` — the tokenized natural-language voice instruction
//                    (<|im_start|>user\n{instruct}<|im_end|>\n), or empty for no
//                    instruction. When present its text_projection embeddings are
//                    prepended to the prefill, exactly as upstream prepends the
//                    instruct turn (VoiceDesign; also the 1.7B CustomVoice
//                    instruct path). Pure text rows — no codec component.
//   `spk_id`       — the codec speaker token (>=0), or -1 for none (VoiceDesign,
//                    or a model with no speaker presets).
//   `language_id`  — the codec language token (>=0), or -1 for "auto".
//
// Writes the prefill (T*hidden row-major), the per-frame trailing-text hidden
// (L*hidden), and the tts_pad embedding (hidden). The prefill positions are a
// plain 0..T-1 ramp (the Talker's get_rope_index is cumsum of an all-ones mask,
// and the prepended instruct rows are ordinary text tokens), so the caller pairs
// this with rope_delta = 0.
void assemble_talker_prefill(const QwenTtsTalker& talker,
                             const QwenTtsConfig& cfg,
                             const std::vector<int32_t>& input_ids,
                             const std::vector<int32_t>& instruct_ids,
                             int spk_id, int language_id,
                             std::vector<float>& prefill, int& T,
                             std::vector<float>& trailing, int& L,
                             std::vector<float>& tts_pad);

}  // namespace brosoundml
