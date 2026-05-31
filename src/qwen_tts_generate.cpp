#include "qwen_tts_generate.h"

#include "qwen_tts_device.h"

#include <brotensor/ops.h>
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
    const bt::Device dev = talker.final_norm.device;
    bt::DeviceScope scope(dev);
    const int H = talker.hidden;
    const int G = cp.num_code_groups;     // 16
    out_frames.clear();

    // ── Talker prefill: seed the KV cache, keep the last token's hidden ──
    QwenTtsTalkerCache cache;
    cache.reset(talker.num_layers);
    bt::Tensor hidden;
    talker.run(prefill_embeds, T, pos3T, &cache, hidden);   // (T, H) device
    bt::Tensor hcur = bt::Tensor::empty_on(dev, 1, H, bt::Dtype::FP32);
    bt::copy_d2d(hidden, (T - 1) * H, hcur, 0, H);           // last row

    // Trailing / pad text embeddings on-device (added one per frame).
    bt::Tensor trailing_dev =
        (L > 0) ? bt::Tensor::from_host_on(dev, trailing_text_hidden, L, H)
                : bt::Tensor::empty_on(dev, 0, H, bt::Dtype::FP32);
    bt::Tensor pad_dev = bt::Tensor::from_host_on(dev, tts_pad_embed, 1, H);

    std::vector<int>     rest;             // codebooks 1..15 from the Code Predictor
    std::vector<int32_t> gen_c0;           // codebook-0 ids emitted so far (rep. penalty)
    std::vector<float>   logits_host(talker.vocab);
    const float kNegInf = -std::numeric_limits<float>::infinity();

    for (int step = 0; step < params.max_frames; ++step) {
        // codebook 0: codec_head over the current hidden, with the upstream
        // logits processing applied on the host (a single 3072-wide row).
        bt::Tensor lg;
        qtd::linear(talker.codec_head, nullptr, hcur, lg);   // (1, vocab) device
        qtd::to_host(lg, logits_host.data());
        float* lp = logits_host.data();
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

        // codebooks 1..15 from the Code Predictor (greedy). Fed device-resident:
        // the Talker hidden (hcur) and the Talker embedding of codebook 0 stay on
        // device, so the depth expansion runs without a host round-trip.
        bt::Tensor c0row = qtd::gather_rows(
            talker.codec_embedding, {static_cast<std::int32_t>(c0)});   // (1, H) device
        cp.predict_dev(hcur, c0row, rest);

        // emit the frame [c0, c1..c15].
        out_frames.push_back(static_cast<int32_t>(c0));
        for (int x : rest) out_frames.push_back(static_cast<int32_t>(x));

        // next Talker input = sum of the 16 code embeddings + trailing/pad text,
        // assembled on-device (reusing c0row as the accumulator).
        bt::Tensor& e = c0row;
        for (int i = 0; i < G - 1; ++i) {
            bt::Tensor er = qtd::gather_rows(
                cp.codec_embedding[i], {static_cast<std::int32_t>(rest[i])});
            bt::add_inplace_batched(e, er);
        }
        if (step < L) {
            bt::Tensor trow = bt::Tensor::view(
                dev, static_cast<float*>(trailing_dev.data) + static_cast<std::size_t>(step) * H,
                1, H, bt::Dtype::FP32);
            bt::add_inplace_batched(e, trow);
        } else {
            bt::add_inplace_batched(e, pad_dev);
        }

        // step the Talker (single token, KV-cached). Generation-phase M-RoPE
        // uses one scalar position on all three axes (T + step + rope_delta).
        const int pos = T + step + params.rope_delta;
        const int32_t pos3[3] = {pos, pos, pos};
        bt::Tensor hstep;
        talker.run_dev(e, 1, pos3, &cache, hstep);
        hcur = std::move(hstep);
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
    const int H = talker.hidden;
    const QwenTtsTalkerConfig& tk = cfg.talker;

    // codec-embedding row -> host buffer (device-safe; the table may be on GPU).
    auto codec_row = [&](int id, std::vector<float>& buf) {
        buf.resize(H);
        talker.codec_embed(id, buf.data());
    };

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

    // role: text_projection over the first three prompt tokens.
    for (int r = 0; r < 3; ++r)
        talker.text_embed_proj(input_ids[r], prefill.data() + static_cast<std::size_t>(r) * H);

    // _tie row i (i in 0..C-2): text_side[i] + codec_embedding(codec_input[i]),
    // where text_side = [tts_pad ×(C-2), tts_bos].
    std::vector<float> ce;
    for (int i = 0; i < C - 1; ++i) {
        float* dst = prefill.data() + static_cast<std::size_t>(3 + i) * H;
        codec_row(codec_input[i], ce);
        const float* txt = (i < C - 2) ? tts_pad_e.data() : tts_bos.data();
        for (int d = 0; d < H; ++d) dst[d] = txt[d] + ce[d];
    }

    // last prefill row: text_projection(first body token) + codec_embedding(codec_bos).
    {
        float* dst = prefill.data() + static_cast<std::size_t>(T - 1) * H;
        std::vector<float> tpb(H);
        talker.text_embed_proj(input_ids[3], tpb.data());
        codec_row(codec_input[C - 1], ce);
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
