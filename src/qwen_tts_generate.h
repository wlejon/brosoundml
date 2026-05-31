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

#include <cstdint>
#include <vector>

namespace brosoundml {

struct QwenTtsGenParams {
    int eos_id     = 0;   // codec EOS in the Talker vocab; stops the loop
    int max_frames = 0;   // hard cap on emitted frames
    int rope_delta = 0;   // M-RoPE position offset for the generation phase
};

// Run the AR loop. `prefill_embeds` is T*hidden row-major; `pos3T` the 3-axis
// M-RoPE prefill positions (axis-major pos3T[a*T + t]). `trailing_text_hidden`
// is L*hidden (the look-ahead text embeddings consumed one per frame), and
// `tts_pad_embed` (hidden floats) is added once the trailing text is exhausted.
// Emitted frames are appended to `out_frames` as num_code_groups ints each,
// frame-major then codebook-within-frame: [c0..c15, c0..c15, ...]. Returns the
// number of frames.
int generate_codes(const QwenTtsTalker& talker, const QwenTtsCodePredictor& cp,
                   const float* prefill_embeds, int T, const int32_t* pos3T,
                   const float* trailing_text_hidden, int L,
                   const float* tts_pad_embed, const QwenTtsGenParams& params,
                   std::vector<int32_t>& out_frames);

}  // namespace brosoundml
