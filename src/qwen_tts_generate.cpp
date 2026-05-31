#include "qwen_tts_generate.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <vector>

namespace brosoundml {

namespace {
namespace bt = brotensor;

int argmax_row(const float* v, int n) {
    int best = 0;
    float bv = v[0];
    for (int i = 1; i < n; ++i) {
        if (v[i] > bv) { bv = v[i]; best = i; }
    }
    return best;
}
}  // namespace

int generate_codes(const QwenTtsTalker& talker, const QwenTtsCodePredictor& cp,
                   const float* prefill_embeds, int T, const int32_t* pos3T,
                   const float* trailing_text_hidden, int L,
                   const float* tts_pad_embed, const QwenTtsGenParams& params,
                   std::vector<int32_t>& out_frames) {
    bt::DeviceScope cpu(bt::Device::CPU);
    const int H = talker.hidden;
    const int G = cp.num_code_groups;     // 16
    out_frames.clear();

    // ── Talker prefill: build the KV cache, take the last token's hidden ──
    QwenTtsTalkerCache cache;
    cache.reset(talker.num_layers);
    bt::Tensor hidden;
    talker.run(prefill_embeds, T, pos3T, &cache, hidden);
    const float* hl = hidden.host_f32() + static_cast<std::size_t>(T - 1) * H;
    std::vector<float> hcur(hl, hl + H);   // post-final-norm hidden of the last token

    std::vector<int> rest;                 // codebooks 1..15 from the Code Predictor
    std::vector<float> e(H);

    for (int step = 0; step < params.max_frames; ++step) {
        // codebook 0 from the Talker.
        bt::Tensor lg;
        talker.codec_logits(hcur.data(), 1, lg);
        const int c0 = argmax_row(lg.host_f32(), talker.vocab);
        if (c0 == params.eos_id) break;

        // codebooks 1..15 from the Code Predictor (greedy).
        cp.predict(hcur.data(), talker.codec_embedding_row(c0), rest);

        // emit the frame [c0, c1..c15].
        out_frames.push_back(static_cast<int32_t>(c0));
        for (int x : rest) out_frames.push_back(static_cast<int32_t>(x));

        // next Talker input = sum of the 16 code embeddings + trailing/pad text.
        const float* c0e = talker.codec_embedding_row(c0);
        for (int d = 0; d < H; ++d) e[d] = c0e[d];
        for (int i = 0; i < G - 1; ++i) {
            const float* er = cp.codec_embedding_row(i, rest[i]);
            for (int d = 0; d < H; ++d) e[d] += er[d];
        }
        const float* add = (step < L) ? (trailing_text_hidden + static_cast<std::size_t>(step) * H)
                                       : tts_pad_embed;
        for (int d = 0; d < H; ++d) e[d] += add[d];

        // step the Talker (single token, KV-cached). Generation-phase M-RoPE
        // uses one scalar position on all three axes (T + step + rope_delta).
        const int pos = T + step + params.rope_delta;
        const int32_t pos3[3] = {pos, pos, pos};
        bt::Tensor hstep;
        talker.run(e.data(), 1, pos3, &cache, hstep);
        const float* hs = hstep.host_f32();
        hcur.assign(hs, hs + H);
    }

    return static_cast<int>(out_frames.size()) / G;
}

}  // namespace brosoundml
