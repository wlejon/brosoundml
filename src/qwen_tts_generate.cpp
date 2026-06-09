#include "qwen_tts_generate.h"

#include "qwen_tts_device.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace brosoundml {

void qwen_cp_profile_report();   // defined in qwen_tts_code_predictor.cpp

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

// Env-gated wall-clock profiler for the AR loop. Each region is bracketed by a
// device sync so the elapsed time reflects that region's GPU work, not async
// launch latency. Reports ms total + ms/frame when BROSOUNDML_QWEN_PROFILE is set.
struct ArProfiler {
    bool on;
    using clk = std::chrono::steady_clock;
    double prefill = 0, head = 0, predict = 0, assemble = 0, talker = 0;
    ArProfiler() : on(std::getenv("BROSOUNDML_QWEN_PROFILE") != nullptr) {}
    clk::time_point tick() {
        if (on) bt::sync_all();
        return clk::now();
    }
    void add(double& acc, clk::time_point t0) {
        if (!on) return;
        bt::sync_all();
        acc += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    }
    void report(int frames) {
        if (!on || frames <= 0) return;
        const double f = static_cast<double>(frames);
        std::fprintf(stderr,
            "\n[qwen AR profile] %d frames, audio %.2fs @ 12.5Hz\n"
            "  prefill        %9.1f ms  (one-time)\n"
            "  codec_head+pol %9.1f ms  %7.3f ms/frame\n"
            "  code_predictor %9.1f ms  %7.3f ms/frame\n"
            "  embed assemble %9.1f ms  %7.3f ms/frame\n"
            "  talker step    %9.1f ms  %7.3f ms/frame\n"
            "  per-frame total           %7.3f ms/frame\n",
            frames, f / 12.5, prefill,
            head, head / f, predict, predict / f,
            assemble, assemble / f, talker, talker / f,
            (head + predict + assemble + talker) / f);
        qwen_cp_profile_report();
    }
};
}  // namespace

int generate_codes(const QwenTtsTalker& talker, const QwenTtsCodePredictor& cp,
                   const float* prefill_embeds, int T, const int32_t* pos3T,
                   const float* trailing_text_hidden, int L,
                   const float* tts_pad_embed, const QwenTtsGenParams& params,
                   std::vector<int32_t>& out_frames,
                   const CancelCheck& cancel, QwenTtsTrace* trace) {
    const bt::Device dev = talker.final_norm.device;
    bt::DeviceScope scope(dev);
    const int H = talker.hidden;
    const int G = cp.num_code_groups;     // 16
    out_frames.clear();
    ArProfiler prof;

    // ── Talker prefill: seed the KV cache, keep the last token's hidden ──
    QwenTtsTalkerCache cache;
    cache.reset(talker.num_layers);
    bt::Tensor hidden;
    auto tp = prof.tick();
    talker.run(prefill_embeds, T, pos3T, &cache, hidden);   // (T, H) device
    prof.add(prof.prefill, tp);
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
    std::vector<float>   conf;             // per-frame codebook-0 top-1 prob (trace)
    const float kNegInf = -std::numeric_limits<float>::infinity();

    // Philox counter for sampling: advances one step per code drawn (codebook 0
    // here, codebooks 1..15 inside predict_dev), so a fixed seed reproduces the
    // whole utterance. Unused on the greedy path (temperature == 0).
    const bool     sampling = params.temperature > 0.0f;
    std::uint64_t  rng_counter = 0;

    for (int step = 0; step < params.max_frames; ++step) {
        // Cooperative cancellation: poll once per frame (the dominant cost) and
        // bail with whatever's emitted so far — the caller discards a cancelled
        // result.
        if (cancel && cancel()) break;

        // codebook 0: codec_head over the current hidden, with the upstream
        // logits processing applied on the host (a single 3072-wide row).
        auto th = prof.tick();
        bt::Tensor lg;
        qtd::linear(talker.codec_head, nullptr, hcur, lg);   // (1, vocab) device
        qtd::to_host(lg, logits_host.data());
        float* lp = logits_host.data();
        if (params.repetition_penalty != 1.0f) {
            const float p = params.repetition_penalty;
            for (int id : gen_c0)
                lp[id] = (lp[id] < 0.0f) ? lp[id] * p : lp[id] / p;
        }
        // Additive logit bias on specific codebook-0 ids (after rep. penalty, before
        // suppression) — favor / forbid chosen codes. Ignore out-of-range ids.
        for (const auto& [id, delta] : params.logit_bias)
            if (id >= 0 && id < talker.vocab) lp[id] += delta;
        for (int i = params.suppress_lo; i < params.suppress_hi; ++i)
            if (i != params.eos_id) lp[i] = kNegInf;
        if (step < params.min_frames) lp[params.eos_id] = kNegInf;

        // top-1 softmax probability of the (edited) codebook-0 distribution:
        // exp(0)/Σexp(v-max) = 1/Σexp(v-max). Suppressed -inf entries -> 0. Needed
        // for the trace and to drive confidence-adaptive temperature.
        const bool need_conf = trace || (sampling && params.adaptive > 0.0f);
        float conf_f = 0.0f;
        if (need_conf) {
            float mx = lp[0];
            for (int i = 1; i < talker.vocab; ++i) if (lp[i] > mx) mx = lp[i];
            double den = 0.0;
            for (int i = 0; i < talker.vocab; ++i) den += std::exp(lp[i] - mx);
            conf_f = den > 0.0 ? static_cast<float>(1.0 / den) : 0.0f;
        }

        int c0;
        if (!sampling) {
            c0 = argmax_row(lp, talker.vocab);
        } else {
            // Confidence-adaptive temperature scales only codebook 0 (the Talker's
            // semantic driver, where "hedging" is meaningful and what the
            // c0_confidence trace shows): hotter where conf is low, near base where
            // high. The Code Predictor's acoustic codebooks keep the base temp.
            float eff_temp = params.temperature;
            if (params.adaptive > 0.0f)
                eff_temp *= 1.0f + params.adaptive * (1.0f - conf_f);
            // Draw codebook 0 from the edited logits via brotensor's seeded
            // sampler (the suppress/min_frames -inf entries softmax to 0).
            bt::Tensor lgs = bt::Tensor::from_host_on(dev, lp, 1, talker.vocab);
            bt::Tensor idx;
            bt::sample_logits(lgs, eff_temp, params.top_k, params.top_p,
                              params.seed, rng_counter, idx);
            ++rng_counter;
            bt::Tensor cidx = idx.to(bt::Device::CPU);
            c0 = *static_cast<const std::int32_t*>(cidx.host_raw());
        }
        prof.add(prof.head, th);
        if (c0 == params.eos_id) break;
        gen_c0.push_back(static_cast<int32_t>(c0));

        if (trace) conf.push_back(conf_f);

        // codebooks 1..15 from the Code Predictor (greedy). Fed device-resident:
        // the Talker hidden (hcur) and the Talker embedding of codebook 0 stay on
        // device, so the depth expansion runs without a host round-trip.
        auto tpr = prof.tick();
        bt::Tensor c0row = qtd::gather_rows(
            talker.codec_embedding, {static_cast<std::int32_t>(c0)});   // (1, H) device
        cp.predict_dev(hcur, c0row, rest, params.temperature, params.top_k,
                       params.top_p, params.seed, sampling ? &rng_counter : nullptr);
        prof.add(prof.predict, tpr);

        // emit the frame [c0, c1..c15].
        out_frames.push_back(static_cast<int32_t>(c0));
        for (int x : rest) out_frames.push_back(static_cast<int32_t>(x));

        // Streaming hook: let the caller decode + emit the new tail mid-loop.
        if (params.on_frame) params.on_frame(out_frames);

        // next Talker input = sum of the 16 code embeddings + trailing/pad text,
        // assembled on-device (reusing c0row as the accumulator).
        auto ta = prof.tick();
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
        prof.add(prof.assemble, ta);

        // step the Talker (single token, KV-cached). Generation-phase M-RoPE
        // uses one scalar position on all three axes (T + step + rope_delta).
        const int pos = T + step + params.rope_delta;
        const int32_t pos3[3] = {pos, pos, pos};
        auto tt = prof.tick();
        bt::Tensor hstep;
        talker.run_dev(e, 1, pos3, &cache, hstep);
        hcur = std::move(hstep);
        prof.add(prof.talker, tt);
    }

    const int nf = static_cast<int>(out_frames.size()) / G;

    if (trace) {
        // code raster: G rows x nf cols, codebook-major (row k = codebook k).
        std::vector<float> raster(static_cast<std::size_t>(G) * nf);
        for (int t = 0; t < nf; ++t)
            for (int k = 0; k < G; ++k)
                raster[static_cast<std::size_t>(k) * nf + t] =
                    static_cast<float>(out_frames[static_cast<std::size_t>(t) * G + k]);
        trace->add("codes", G, nf, std::move(raster));
        const int cw = static_cast<int>(conf.size());   // read before the move (arg-eval order)
        trace->add("c0_confidence", 1, cw, std::move(conf));
    }

    prof.report(nf);
    return nf;
}

void assemble_talker_prefill(const QwenTtsTalker& talker,
                             const QwenTtsConfig& cfg,
                             const std::vector<int32_t>& input_ids,
                             const std::vector<int32_t>& instruct_ids,
                             int spk_id, const float* spk_embed,
                             const float* spk_steer, int language_id,
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
    // The speaker sits between the think tag and the pad/bos. CustomVoice uses a
    // preset speaker token (looked up in codec_embedding); the Base zero-shot
    // clone instead splices a raw x-vector at that position (spk_pos), so push a
    // placeholder token there and override the row below. `slot_pos` records
    // where the speaker row lands for EITHER source, so an optional steer offset
    // can nudge it regardless of how it was filled.
    int spk_pos = -1;     // x-vector splice position (codec source = spk_embed)
    int slot_pos = -1;    // speaker row, whether token- or x-vector-filled
    if (spk_embed) {
        spk_pos = static_cast<int>(codec_input.size());
        slot_pos = spk_pos;
        codec_input.push_back(tk.codec_pad_id);
    } else if (spk_id >= 0) {
        slot_pos = static_cast<int>(codec_input.size());
        codec_input.push_back(spk_id);
    }
    codec_input.push_back(tk.codec_pad_id);
    codec_input.push_back(tk.codec_bos_id);
    const int C = static_cast<int>(codec_input.size());

    // VoiceDesign (and 1.7B CustomVoice) prepend a natural-language instruction
    // turn before the role/body: each instruct token contributes one pure
    // text_projection row (no codec component), exactly as upstream prepends
    // text_projection(text_embedding(instruct_ids)). Empty = no instruction.
    const int I = static_cast<int>(instruct_ids.size());

    // prefill = instruct(I) + role(3) + _tie(C-1) + first-text-token(1).
    T = I + C + 3;
    prefill.assign(static_cast<std::size_t>(T) * H, 0.0f);

    // instruction rows: text_projection over each instruction token.
    for (int i = 0; i < I; ++i)
        talker.text_embed_proj(instruct_ids[i],
                               prefill.data() + static_cast<std::size_t>(i) * H);

    // role: text_projection over the first three body prompt tokens.
    for (int r = 0; r < 3; ++r)
        talker.text_embed_proj(input_ids[r],
                               prefill.data() + static_cast<std::size_t>(I + r) * H);

    // _tie row i (i in 0..C-2): text_side[i] + codec_embedding(codec_input[i]),
    // where text_side = [tts_pad ×(C-2), tts_bos].
    std::vector<float> ce;
    for (int i = 0; i < C - 1; ++i) {
        float* dst = prefill.data() + static_cast<std::size_t>(I + 3 + i) * H;
        const float* txt = (i < C - 2) ? tts_pad_e.data() : tts_bos.data();
        const float* codec = spk_embed;             // raw x-vector at spk_pos
        if (i != spk_pos) { codec_row(codec_input[i], ce); codec = ce.data(); }
        for (int d = 0; d < H; ++d) dst[d] = txt[d] + codec[d];
        // Voice-slot steering: an additive direction on the speaker row (emotion /
        // masc-fem), applied once whether the row came from a token or an x-vector.
        if (spk_steer && i == slot_pos)
            for (int d = 0; d < H; ++d) dst[d] += spk_steer[d];
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
