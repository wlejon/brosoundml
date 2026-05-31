#include "qwen_tts_generate.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <limits>
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
    std::vector<int32_t> gen_c0;           // codebook-0 ids emitted so far (rep. penalty)
    const float kNegInf = -std::numeric_limits<float>::infinity();

    for (int step = 0; step < params.max_frames; ++step) {
        // codebook 0 from the Talker, with the upstream logits processing.
        bt::Tensor lg;
        talker.codec_logits(hcur.data(), 1, lg);
        float* lp = lg.host_f32_mut();
        if (params.repetition_penalty != 1.0f) {
            const float p = params.repetition_penalty;
            for (int id : gen_c0)
                lp[id] = (lp[id] < 0.0f) ? lp[id] * p : lp[id] / p;
        }
        for (int i = params.suppress_lo; i < params.suppress_hi; ++i)
            if (i != params.eos_id) lp[i] = kNegInf;
        if (step < params.min_frames) lp[params.eos_id] = kNegInf;
        const int c0 = argmax_row(lp, talker.vocab);
        if (c0 == params.eos_id) break;
        gen_c0.push_back(static_cast<int32_t>(c0));

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

void assemble_custom_voice_prefill(const QwenTtsTalker& talker,
                                   const QwenTtsConfig& cfg,
                                   const std::vector<int32_t>& input_ids,
                                   int spk_id, int language_id,
                                   std::vector<float>& prefill, int& T,
                                   std::vector<float>& trailing, int& L,
                                   std::vector<float>& tts_pad) {
    bt::DeviceScope cpu(bt::Device::CPU);
    const int H = talker.hidden;
    const QwenTtsTalkerConfig& tk = cfg.talker;

    // tts framing embeds: text_projection(text_embedding(id)).
    std::vector<float> tts_bos(H), tts_eos(H), tts_pad_e(H);
    talker.text_embed_proj(cfg.tts_bos_id, tts_bos.data());
    talker.text_embed_proj(cfg.tts_eos_id, tts_eos.data());
    talker.text_embed_proj(cfg.tts_pad_id, tts_pad_e.data());
    tts_pad.assign(tts_pad_e.begin(), tts_pad_e.end());

    // codec prefix tokens: think tag (+ language) [+ speaker] + pad + bos.
    std::vector<int> codec_input;
    if (language_id < 0) {
        codec_input = {tk.codec_nothink_id, tk.codec_think_bos_id, tk.codec_think_eos_id};
    } else {
        codec_input = {tk.codec_think_id, tk.codec_think_bos_id, language_id, tk.codec_think_eos_id};
    }
    if (spk_id >= 0) codec_input.push_back(spk_id);
    codec_input.push_back(tk.codec_pad_id);
    codec_input.push_back(tk.codec_bos_id);
    const int C = static_cast<int>(codec_input.size());

    // prefill = role(3) + _tie(C-1) + first-text-token(1)  →  C+3 rows.
    T = C + 3;
    prefill.assign(static_cast<std::size_t>(T) * H, 0.0f);

    // role: text_projection over the first three prompt tokens
    // (<|im_start|>, assistant, \n).
    for (int r = 0; r < 3; ++r)
        talker.text_embed_proj(input_ids[r], prefill.data() + static_cast<std::size_t>(r) * H);

    // _tie row i (i in 0..C-2): text_side[i] + codec_embedding(codec_input[i]),
    // where text_side = [tts_pad ×(C-2), tts_bos].
    for (int i = 0; i < C - 1; ++i) {
        float* dst = prefill.data() + static_cast<std::size_t>(3 + i) * H;
        const float* ce = talker.codec_embedding_row(codec_input[i]);
        const float* txt = (i < C - 2) ? tts_pad_e.data() : tts_bos.data();
        for (int d = 0; d < H; ++d) dst[d] = txt[d] + ce[d];
    }

    // last prefill row: text_projection(first body token) + codec_embedding(codec_bos).
    {
        float* dst = prefill.data() + static_cast<std::size_t>(T - 1) * H;
        std::vector<float> tpb(H);
        talker.text_embed_proj(input_ids[3], tpb.data());
        const float* ce = talker.codec_embedding_row(codec_input[C - 1]);
        for (int d = 0; d < H; ++d) dst[d] = tpb[d] + ce[d];
    }

    // trailing = text_projection(body[1:]) + tts_eos. body = input_ids[3:-5];
    // body[1:] = input_ids[4 : len-5].
    const int body_hi = static_cast<int>(input_ids.size()) - 5;   // exclusive
    const int n_body  = body_hi - 4;                              // body[1:] count (>=0)
    L = n_body + 1;
    trailing.assign(static_cast<std::size_t>(L) * H, 0.0f);
    for (int i = 0; i < n_body; ++i)
        talker.text_embed_proj(input_ids[4 + i],
                               trailing.data() + static_cast<std::size_t>(i) * H);
    std::copy(tts_eos.begin(), tts_eos.end(),
              trailing.data() + static_cast<std::size_t>(n_body) * H);
}

}  // namespace brosoundml
