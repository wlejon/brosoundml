// OmniVoice LM — see omnivoice_lm.h.

#include "omnivoice_lm.h"

#include "omnivoice_prompt.h"
#include "qwen_tts_device.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/detail/hash_rng.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#ifdef BROSOUNDML_HAS_CUDA
#include <brotensor/cuda_graph.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: OmniVoiceLm: " + msg);
}

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Upload a (rows, cols) weight as FP32 (widening a BF16/F16 checkpoint on the
// way) onto `dev`.
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols, bt::Device dev) {
    const sf::TensorView& v = need(f, name);
    long long n = 1;
    for (int64_t d : v.shape) n *= d;
    if (n != static_cast<long long>(rows) * cols)
        fail("tensor '" + name + "' has " + std::to_string(n) + " elements, expected " +
             std::to_string(rows) + "x" + std::to_string(cols));
    bt::Tensor t;
    {
        bt::DeviceScope cpu(bt::Device::CPU);
        sf::upload_as(v, rows, cols, bt::Dtype::FP32, t);
    }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}

bt::Tensor up_vec(const sf::File& f, const std::string& name, int n, bt::Device dev) {
    return up(f, name, n, 1, dev);
}

void h2d(bt::Device dev, void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    if (dev == bt::Device::CPU) std::memcpy(dst, src, bytes);
    else bt::detail::alloc_for(dev).memcpy_h2d(dst, src, bytes, dev.index);
}

void d2h(bt::Device dev, void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    if (dev == bt::Device::CPU) std::memcpy(dst, src, bytes);
    else bt::detail::alloc_for(dev).memcpy_d2h(dst, src, bytes, dev.index);
}

bt::Tensor row_view(const bt::Tensor& t, int row0, int rows, int cols, bt::Dtype dt) {
    const std::size_t esz = (dt == bt::Dtype::INT32 || dt == bt::Dtype::FP32) ? 4 : 2;
    char* p = static_cast<char*>(t.data) + static_cast<std::size_t>(row0) * cols * esz;
    return bt::Tensor::view(t.device, p, rows, cols, dt);
}

struct Layer {
    bt::Tensor in_ln, post_ln, qw, kw, vw, ow, q_norm, k_norm, gate, up, down;
};

struct Scratch {
    bt::Tensor normed, q, k, v, qn, kn, qr, kr, ctx, attn, n2, g, u, dn;
    // BF16 mode: the 16-bit GEMM operands (the FP32 activation cast in, the
    // product cast back out into the FP32 stream), one per operand shape.
    bt::Tensor xb_h, xb_q, xb_i, xb_r;          // (L,H) (L,qd) (L,I) (R,H)
    bt::Tensor yb_q, yb_k, yb_h, yb_i, yb_l;    // (L,qd) (L,kd) (L,H) (L,I) (R,C*V)
    // BF16 attention operands: Q/K/V cast to BF16, and (CUDA) K/V expanded
    // from the kv heads to the q heads for the fused tensor-core kernel,
    // whose output lands straight in xb_q (the o_proj input).
    bt::Tensor qb, kb, vb;                      // (L,qd) (L,kd) (L,kd)
    bt::Tensor kx, vx;                          // (L,qd) (L,qd)
};

// Every buffer of one (n_text, n_ref, T, cfg) shape, allocated once so the
// step body never allocates — the CUDA-graph capture requirement.
struct Session {
    int n_text = 0, n_ref = 0, T = 0, Tu = 0, L = 0, Lc = 0, R = 0;
    bt::Tensor embeds, hs;             // (L, H)
    bt::Tensor cos_t, sin_t;           // (L, half)
    Scratch sc;
    bt::Tensor target_idx;             // (R, 1) INT32: rows Lc-T..Lc-1, then Lc..L-1
    bt::Tensor cond_idx, uncond_idx;   // (T, 1) INT32
    bt::Tensor kv_idx;                 // (L*n_q, 1) INT32: kv-head row of each (row, q head)
    bt::Tensor tokens, unmask, pred;   // (C, T) INT32
    bt::Tensor scores, conf;           // (C, T) FP32 (conf = raw, unpenalised)
    bt::Tensor frame, tmp;             // (T, H)
    bt::Tensor htar, hnorm;            // (R, H)
    bt::Tensor logits;                 // (R, C*V)
    bt::Tensor vals, idx;              // top-k outputs
#ifdef BROSOUNDML_HAS_CUDA
    bt::CudaGraph graph;
#endif
    bool captured = false;
    unsigned long long last_use = 0;
};

constexpr std::uint64_t kStepSeedDomain = 0x2545F4914F6CDD1DULL;

}  // namespace

struct OmniVoiceLm::Impl {
    OmniVoiceLmConfig cfg;
    bt::Device dev = bt::Device::CPU;
    bool bf16 = false;
    bool loaded = false;

    bt::Tensor embed_tokens_cpu;   // (vocab, H) host FP32
    bt::Tensor audio_emb;          // (C*V, H)
    bt::Tensor audio_heads;        // (C*V, H), BF16 when narrowed
    bt::Tensor final_norm;         // (H, 1)
    std::vector<Layer> layers;

    mutable std::vector<std::unique_ptr<Session>> sessions;
    mutable unsigned long long use_counter = 0;

    int H() const { return cfg.hidden_size; }
    int C() const { return cfg.num_codebooks; }
    int V() const { return cfg.audio_vocab_size; }

    // sum_c audio_emb[c*V + tokens[c, t]] over a (C, n) INT32 grid -> (n, H)
    // in `out`, with `tmp` as the per-codebook gather target. Sequential
    // accumulation in codebook order (the same float order on every backend).
    void frames_from_tokens(const bt::Tensor& tokens, int n, bt::Tensor& out, bt::Tensor& tmp) const {
        for (int c = 0; c < C(); ++c) {
            bt::Tensor W = row_view(audio_emb, c * V(), V(), H(), bt::Dtype::FP32);
            bt::Tensor I = row_view(tokens, c * n, n, 1, bt::Dtype::INT32);
            if (c == 0) {
                bt::gather_rows(W, I, out);
            } else {
                bt::gather_rows(W, I, tmp);
                bt::add_inplace(out, tmp);
            }
        }
    }

    // One projection Y = X W^T.
    //   FP32 mode: the FP32 GEMM.
    //   BF16 mode, CUDA: the tensor-core GEMM on BF16 operands — X cast to
    //     BF16 (`xb`), FP32 accumulation, the BF16 product (`yb`) widened back
    //     into the FP32 stream. Residual stream, norms, RoPE, attention and
    //     the SwiGLU product stay FP32.
    //   BF16 mode, CPU: the same arithmetic without a 16-bit GEMM — X is
    //     rounded to BF16 in place, the FP32 dot runs over the BF16 weights,
    //     and Y is rounded to BF16 — so the CPU result is the reference for
    //     the CUDA one (accumulation order aside).
    //   `cast_x` false reuses the BF16 `xb` (and rounded X) a previous call
    //   on the same X produced — q/k/v share one normed input, gate/up too.
    void proj(const bt::Tensor& W, bt::Tensor& X, bt::Tensor& Y, bt::Tensor& xb, bt::Tensor& yb,
              bool cast_x = true) const {
        if (!bf16) {
            qtd::linear(W, nullptr, X, Y);
            return;
        }
        if (cast_x) bt::cast(X, xb, bt::Dtype::BF16);
        if (dev.type == bt::DeviceType::CPU) {
            if (cast_x) bt::cast(xb, X, bt::Dtype::FP32);
            qtd::linear(W, nullptr, X, Y);
            bt::cast(Y, yb, bt::Dtype::BF16);
        } else {
            bt::linear_forward_batched_fp16(W, nullptr, xb, yb);
        }
        bt::cast(yb, Y, bt::Dtype::FP32);
    }

    // BF16 mode on CUDA runs attention on the fused tensor-core kernel
    // (flash_attention_forward, BF16 operands, FP32 accumulation); CPU and
    // FP32 mode use the FP32 GQA kernel.
    bool bf16_attention() const { return bf16 && dev.type != bt::DeviceType::CPU; }

    // Self-attention of both documents over the RoPE'd q/k (sc.qr, sc.kr)
    // and v (sc.v); each document attends only within itself, non-causal.
    //   FP32 mode: flash_attention_gqa_forward -> sc.ctx (FP32).
    //   BF16 mode, CUDA: q/k/v cast to BF16, k/v expanded from the kv heads
    //     to the q heads with one gather (the fused kernel has no GQA), the
    //     output written to sc.xb_q, which is the o_proj's BF16 input.
    //   BF16 mode, CPU: q/k/v rounded to BF16 in place, then the FP32 GQA
    //     kernel -> sc.ctx (the probabilities are not rounded, so this is a
    //     near-emulation of the CUDA path, not a bit-level one).
    void attention(Session& s) const {
        const int nq = cfg.num_attention_heads, nkv = cfg.num_key_value_heads, hd = cfg.head_dim;
        const int qd = nq * hd, kd = nkv * hd;
        Scratch& sc = s.sc;
        const bt::Dtype F = bt::Dtype::FP32, Bf = bt::Dtype::BF16;

        if (bf16_attention()) {
            bt::cast(sc.qr, sc.qb, Bf);
            bt::cast(sc.kr, sc.kb, Bf);
            bt::cast(sc.v, sc.vb, Bf);
            {
                bt::Tensor ksrc = bt::Tensor::view(dev, sc.kb.data, s.L * nkv, hd, Bf);
                bt::Tensor vsrc = bt::Tensor::view(dev, sc.vb.data, s.L * nkv, hd, Bf);
                bt::Tensor kdst = bt::Tensor::view(dev, sc.kx.data, s.L * nq, hd, Bf);
                bt::Tensor vdst = bt::Tensor::view(dev, sc.vx.data, s.L * nq, hd, Bf);
                bt::gather_rows(ksrc, s.kv_idx, kdst);
                bt::gather_rows(vsrc, s.kv_idx, vdst);
            }
            for (int doc = 0; doc < 2; ++doc) {
                const int r0 = doc == 0 ? 0 : s.Lc;
                const int n = doc == 0 ? s.Lc : s.Tu;
                if (n == 0) continue;
                bt::Tensor Q = row_view(sc.qb, r0, n, qd, Bf);
                bt::Tensor K = row_view(sc.kx, r0, n, qd, Bf);
                bt::Tensor Vv = row_view(sc.vx, r0, n, qd, Bf);
                bt::Tensor O = row_view(sc.xb_q, r0, n, qd, Bf);
                bt::flash_attention_forward(Q, K, Vv, nullptr, nq, false, O);
            }
            return;
        }
        if (bf16) {   // CPU emulation: round the operands
            bt::cast(sc.qr, sc.qb, Bf); bt::cast(sc.qb, sc.qr, F);
            bt::cast(sc.kr, sc.kb, Bf); bt::cast(sc.kb, sc.kr, F);
            bt::cast(sc.v, sc.vb, Bf);  bt::cast(sc.vb, sc.v, F);
        }
        for (int doc = 0; doc < 2; ++doc) {
            const int r0 = doc == 0 ? 0 : s.Lc;
            const int n = doc == 0 ? s.Lc : s.Tu;
            if (n == 0) continue;
            bt::Tensor Q = row_view(sc.qr, r0, n, qd, F);
            bt::Tensor K = row_view(sc.kr, r0, n, kd, F);
            bt::Tensor Vv = row_view(sc.v, r0, n, kd, F);
            bt::Tensor O = row_view(sc.ctx, r0, n, qd, F);
            bt::flash_attention_gqa_forward(Q, K, Vv, nullptr, nq, nkv, false, O);
        }
    }

    // The captured / eager step body: target embeddings from the token grid,
    // the trunk over the packed sequence, the audio heads on the target rows.
    void body(Session& s) const {
        const int H_ = H();
        Scratch& sc = s.sc;

        frames_from_tokens(s.tokens, s.T, s.frame, s.tmp);
        bt::scatter_rows(s.frame, s.cond_idx, s.embeds);
        if (s.Tu > 0) bt::scatter_rows(s.frame, s.uncond_idx, s.embeds);
        bt::copy_d2d(s.embeds, 0, s.hs, 0, s.L * H_);

        for (int l = 0; l < cfg.num_hidden_layers; ++l) {
            const Layer& tl = layers[static_cast<std::size_t>(l)];
            bt::rms_norm_forward(s.hs, tl.in_ln, cfg.rms_norm_eps, sc.normed);
            proj(tl.qw, sc.normed, sc.q, sc.xb_h, sc.yb_q);
            proj(tl.kw, sc.normed, sc.k, sc.xb_h, sc.yb_k, false);
            proj(tl.vw, sc.normed, sc.v, sc.xb_h, sc.yb_k, false);
            qtd::head_rms_norm(sc.q, s.L, cfg.num_attention_heads, cfg.head_dim, tl.q_norm,
                               cfg.rms_norm_eps, sc.qn);
            qtd::head_rms_norm(sc.k, s.L, cfg.num_key_value_heads, cfg.head_dim, tl.k_norm,
                               cfg.rms_norm_eps, sc.kn);
            bt::rope_apply(sc.qn, s.cos_t, s.sin_t, cfg.head_dim, cfg.num_attention_heads, sc.qr);
            bt::rope_apply(sc.kn, s.cos_t, s.sin_t, cfg.head_dim, cfg.num_key_value_heads, sc.kr);
            // Two documents, each attending only within itself, non-causal.
            attention(s);
            proj(tl.ow, sc.ctx, sc.attn, sc.xb_q, sc.yb_h, /*cast_x=*/!bf16_attention());
            bt::add_inplace_batched(s.hs, sc.attn);

            bt::rms_norm_forward(s.hs, tl.post_ln, cfg.rms_norm_eps, sc.n2);
            proj(tl.gate, sc.n2, sc.g, sc.xb_h, sc.yb_i);
            proj(tl.up, sc.n2, sc.u, sc.xb_h, sc.yb_i, false);
            qtd::swiglu(sc.g, sc.u);
            proj(tl.down, sc.g, sc.dn, sc.xb_i, sc.yb_h);
            bt::add_inplace_batched(s.hs, sc.dn);
        }

        bt::gather_rows(s.hs, s.target_idx, s.htar);
        bt::rms_norm_forward(s.htar, final_norm, cfg.rms_norm_eps, s.hnorm);
        proj(audio_heads, s.hnorm, s.logits, s.sc.xb_r, s.sc.yb_l);
    }

    Session& session(int n_text, int n_ref, int T, bool use_cfg) const {
        const int Tu = use_cfg ? T : 0;
        for (auto& sp : sessions)
            if (sp->n_text == n_text && sp->n_ref == n_ref && sp->T == T && sp->Tu == Tu) {
                sp->last_use = ++use_counter;
                return *sp;
            }
        if (sessions.size() >= 4) {
            auto oldest = std::min_element(sessions.begin(), sessions.end(),
                                           [](const auto& a, const auto& b) { return a->last_use < b->last_use; });
            sessions.erase(oldest);
        }
        auto sp = std::make_unique<Session>();
        Session& s = *sp;
        s.n_text = n_text; s.n_ref = n_ref; s.T = T; s.Tu = Tu;
        s.Lc = n_text + n_ref + T;
        s.L = s.Lc + Tu;
        s.R = T + Tu;
        const int H_ = H(), C_ = C(), V_ = V();
        const int qd = cfg.num_attention_heads * cfg.head_dim;
        const int kd = cfg.num_key_value_heads * cfg.head_dim;
        const int I = cfg.intermediate_size;
        const int half = cfg.head_dim / 2;
        const bt::Dtype F = bt::Dtype::FP32, N = bt::Dtype::INT32;

        s.embeds = bt::Tensor::zeros_on(dev, s.L, H_, F);
        s.hs     = bt::Tensor::zeros_on(dev, s.L, H_, F);
        s.sc.normed = bt::Tensor::zeros_on(dev, s.L, H_, F);
        s.sc.q  = bt::Tensor::zeros_on(dev, s.L, qd, F);
        s.sc.k  = bt::Tensor::zeros_on(dev, s.L, kd, F);
        s.sc.v  = bt::Tensor::zeros_on(dev, s.L, kd, F);
        s.sc.qn = bt::Tensor::zeros_on(dev, s.L, qd, F);
        s.sc.kn = bt::Tensor::zeros_on(dev, s.L, kd, F);
        s.sc.qr = bt::Tensor::zeros_on(dev, s.L, qd, F);
        s.sc.kr = bt::Tensor::zeros_on(dev, s.L, kd, F);
        s.sc.ctx  = bt::Tensor::zeros_on(dev, s.L, qd, F);
        s.sc.attn = bt::Tensor::zeros_on(dev, s.L, H_, F);
        s.sc.n2 = bt::Tensor::zeros_on(dev, s.L, H_, F);
        s.sc.g  = bt::Tensor::zeros_on(dev, s.L, I, F);
        s.sc.u  = bt::Tensor::zeros_on(dev, s.L, I, F);
        s.sc.dn = bt::Tensor::zeros_on(dev, s.L, H_, F);
        s.frame = bt::Tensor::zeros_on(dev, T, H_, F);
        s.tmp   = bt::Tensor::zeros_on(dev, T, H_, F);
        s.htar  = bt::Tensor::zeros_on(dev, s.R, H_, F);
        s.hnorm = bt::Tensor::zeros_on(dev, s.R, H_, F);
        s.logits = bt::Tensor::zeros_on(dev, s.R, C_ * V_, F);
        s.tokens = bt::Tensor::zeros_on(dev, C_, T, N);
        s.unmask = bt::Tensor::zeros_on(dev, C_, T, N);
        s.pred   = bt::Tensor::zeros_on(dev, C_, T, N);
        s.scores = bt::Tensor::zeros_on(dev, C_, T, F);
        s.conf   = bt::Tensor::zeros_on(dev, C_, T, F);
        if (bf16) {
            const bt::Dtype Bf = bt::Dtype::BF16;
            s.sc.xb_h = bt::Tensor::zeros_on(dev, s.L, H_, Bf);
            s.sc.xb_q = bt::Tensor::zeros_on(dev, s.L, qd, Bf);
            s.sc.xb_i = bt::Tensor::zeros_on(dev, s.L, I, Bf);
            s.sc.xb_r = bt::Tensor::zeros_on(dev, s.R, H_, Bf);
            s.sc.yb_q = bt::Tensor::zeros_on(dev, s.L, qd, Bf);
            s.sc.yb_k = bt::Tensor::zeros_on(dev, s.L, kd, Bf);
            s.sc.yb_h = bt::Tensor::zeros_on(dev, s.L, H_, Bf);
            s.sc.yb_i = bt::Tensor::zeros_on(dev, s.L, I, Bf);
            s.sc.yb_l = bt::Tensor::zeros_on(dev, s.R, C_ * V_, Bf);
            s.sc.qb = bt::Tensor::zeros_on(dev, s.L, qd, Bf);
            s.sc.kb = bt::Tensor::zeros_on(dev, s.L, kd, Bf);
            s.sc.vb = bt::Tensor::zeros_on(dev, s.L, kd, Bf);
            s.sc.kx = bt::Tensor::zeros_on(dev, s.L, qd, Bf);
            s.sc.vx = bt::Tensor::zeros_on(dev, s.L, qd, Bf);
            const int nq = cfg.num_attention_heads, nkv = cfg.num_key_value_heads;
            const int group = nq / nkv;
            std::vector<int32_t> kvi(static_cast<std::size_t>(s.L) * nq);
            for (int r = 0; r < s.L; ++r)
                for (int h = 0; h < nq; ++h)
                    kvi[static_cast<std::size_t>(r) * nq + h] = r * nkv + h / group;
            s.kv_idx = qtd::upload_idx(dev, kvi.data(), s.L * nq);
        }

        std::vector<int32_t> tidx(static_cast<std::size_t>(s.R));
        std::vector<int32_t> cidx(static_cast<std::size_t>(T)), uidx(static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t) {
            cidx[static_cast<std::size_t>(t)] = s.Lc - T + t;
            uidx[static_cast<std::size_t>(t)] = s.Lc + t;
            tidx[static_cast<std::size_t>(t)] = s.Lc - T + t;
            if (Tu > 0) tidx[static_cast<std::size_t>(T + t)] = s.Lc + t;
        }
        s.target_idx = qtd::upload_idx(dev, tidx.data(), s.R);
        s.cond_idx   = qtd::upload_idx(dev, cidx.data(), T);
        s.uncond_idx = qtd::upload_idx(dev, uidx.data(), T);

        // RoPE: positions restart at 0 for the unconditional document.
        std::vector<float> cb(static_cast<std::size_t>(s.L) * half), sb(cb.size());
        for (int i = 0; i < half; ++i) {
            const float inv_freq = std::pow(cfg.rope_theta, -(2.0f * static_cast<float>(i)) / static_cast<float>(cfg.head_dim));
            for (int r = 0; r < s.L; ++r) {
                const int pos = r < s.Lc ? r : r - s.Lc;
                const float ang = static_cast<float>(pos) * inv_freq;
                cb[static_cast<std::size_t>(r) * half + i] = std::cos(ang);
                sb[static_cast<std::size_t>(r) * half + i] = std::sin(ang);
            }
        }
        s.cos_t = bt::Tensor::from_host_on(dev, cb.data(), s.L, half);
        s.sin_t = bt::Tensor::from_host_on(dev, sb.data(), s.L, half);

        s.last_use = ++use_counter;
        sessions.push_back(std::move(sp));
        return *sessions.back();
    }

    // Text + reference rows of the packed sequence (outside the captured body).
    void write_prefix(Session& s, const std::vector<int32_t>& text_ids,
                      const std::vector<int32_t>& ref_codes) const {
        const int H_ = H();
        if (s.n_text > 0) {
            bt::Tensor te = qtd::gather_rows(embed_tokens_cpu, text_ids);   // host (n_text, H)
            bt::Tensor ted = (dev == bt::Device::CPU) ? te : te.to(dev);
            bt::copy_d2d(ted, 0, s.embeds, 0, s.n_text * H_);
        }
        if (s.n_ref > 0) {
            bt::Tensor rt = qtd::upload_idx(dev, ref_codes.data(), C() * s.n_ref);   // (C*n_ref, 1)
            bt::Tensor rgrid = bt::Tensor::view(dev, rt.data, C(), s.n_ref, bt::Dtype::INT32);
            bt::Tensor out = bt::Tensor::zeros_on(dev, s.n_ref, H_, bt::Dtype::FP32);
            bt::Tensor tmp = bt::Tensor::zeros_on(dev, s.n_ref, H_, bt::Dtype::FP32);
            frames_from_tokens(rgrid, s.n_ref, out, tmp);
            bt::copy_d2d(out, 0, s.embeds, s.n_text * H_, s.n_ref * H_);
        }
    }

    void run_body(Session& s) const {
#ifdef BROSOUNDML_HAS_CUDA
        if (dev.type == bt::DeviceType::CUDA) {
            if (!s.captured) {
                body(s);            // warm-up (every output already sized; idempotent)
                bt::sync_all();
                {
                    bt::CudaGraphCapture cap;
                    body(s);
                    s.graph = cap.finish();
                }
                s.captured = true;
            } else {
                s.graph.launch();
            }
            return;
        }
#endif
        body(s);
    }
};

OmniVoiceLm::OmniVoiceLm() : impl_(new Impl()) {}
OmniVoiceLm::~OmniVoiceLm() = default;
OmniVoiceLm::OmniVoiceLm(OmniVoiceLm&&) noexcept = default;
OmniVoiceLm& OmniVoiceLm::operator=(OmniVoiceLm&&) noexcept = default;

void OmniVoiceLm::load(const sf::File& f, const OmniVoiceLmConfig& cfg, bt::Device dev, bool bf16) {
    Impl& m = *impl_;
    m.sessions.clear();
    m.cfg = cfg;
    m.dev = dev;
    m.bf16 = bf16;
    m.loaded = false;
    if (cfg.head_dim % 2) fail("head_dim must be even");
    if (cfg.num_attention_heads % cfg.num_key_value_heads) fail("num_key_value_heads must divide num_attention_heads");
    bt::DeviceScope scope(dev);

    const int H = cfg.hidden_size;
    const int CV = cfg.num_codebooks * cfg.audio_vocab_size;
    const int qd = cfg.num_attention_heads * cfg.head_dim;
    const int kd = cfg.num_key_value_heads * cfg.head_dim;

    m.embed_tokens_cpu = up(f, "llm.embed_tokens.weight", cfg.vocab_size, H, bt::Device::CPU);
    m.audio_emb   = up(f, "audio_embeddings.weight", CV, H, dev);
    m.audio_heads = qtd::narrow_bf16(up(f, "audio_heads.weight", CV, H, dev), bf16);
    m.final_norm  = up_vec(f, "llm.norm.weight", H, dev);

    const std::vector<int32_t> hd_perm = qtd::rotate_half_perm(cfg.head_dim);
    const std::vector<int32_t> q_perm = qtd::per_head_perm_rows(hd_perm, cfg.num_attention_heads, cfg.head_dim);
    const std::vector<int32_t> k_perm = qtd::per_head_perm_rows(hd_perm, cfg.num_key_value_heads, cfg.head_dim);

    m.layers.clear();
    m.layers.resize(static_cast<std::size_t>(cfg.num_hidden_layers));
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string L = "llm.layers." + std::to_string(i) + ".";
        Layer& tl = m.layers[static_cast<std::size_t>(i)];
        tl.in_ln   = up_vec(f, L + "input_layernorm.weight", H, dev);
        tl.post_ln = up_vec(f, L + "post_attention_layernorm.weight", H, dev);
        tl.qw = qtd::narrow_bf16(qtd::gather_rows(up(f, L + "self_attn.q_proj.weight", qd, H, dev), q_perm), bf16);
        tl.kw = qtd::narrow_bf16(qtd::gather_rows(up(f, L + "self_attn.k_proj.weight", kd, H, dev), k_perm), bf16);
        tl.vw = qtd::narrow_bf16(up(f, L + "self_attn.v_proj.weight", kd, H, dev), bf16);
        tl.ow = qtd::narrow_bf16(up(f, L + "self_attn.o_proj.weight", H, qd, dev), bf16);
        tl.q_norm = qtd::gather_rows(up_vec(f, L + "self_attn.q_norm.weight", cfg.head_dim, dev), hd_perm);
        tl.k_norm = qtd::gather_rows(up_vec(f, L + "self_attn.k_norm.weight", cfg.head_dim, dev), hd_perm);
        tl.gate = qtd::narrow_bf16(up(f, L + "mlp.gate_proj.weight", cfg.intermediate_size, H, dev), bf16);
        tl.up   = qtd::narrow_bf16(up(f, L + "mlp.up_proj.weight", cfg.intermediate_size, H, dev), bf16);
        tl.down = qtd::narrow_bf16(up(f, L + "mlp.down_proj.weight", H, cfg.intermediate_size, dev), bf16);
    }
    m.loaded = true;
}

OmniVoiceLmResult OmniVoiceLm::generate(const std::vector<int32_t>& text_ids,
                                        const std::vector<int32_t>& ref_codes, int n_ref, int T,
                                        const OmniVoiceInit* init, const OmniVoiceLmRun& run,
                                        const CancelCheck& cancel, const OmniVoiceStepFn& on_step,
                                        OmniVoiceLmDebug* dbg) const {
    const Impl& m = *impl_;
    if (!m.loaded) fail("no model loaded");
    if (T < 1) fail("num_frames must be >= 1");
    if (text_ids.empty()) fail("empty text prompt");
    if (n_ref < 0 || ref_codes.size() != static_cast<std::size_t>(m.C()) * static_cast<std::size_t>(n_ref))
        fail("reference codes must hold num_codebooks * n_ref entries");
    if (run.num_steps < 1) fail("num_steps must be >= 1");
    for (int32_t id : text_ids)
        if (id < 0 || id >= m.cfg.vocab_size) fail("text id out of range");
    for (int32_t c : ref_codes)
        if (c < 0 || c >= m.V()) fail("reference code out of range");
    const int C = m.C(), V = m.V(), MASK = m.cfg.audio_mask_id;
    const std::size_t cells = static_cast<std::size_t>(C) * static_cast<std::size_t>(T);
    if (init) {
        if (init->tokens.size() != cells || init->keep.size() != cells)
            fail("init grid must hold num_codebooks * num_frames tokens and keep flags");
        for (std::size_t i = 0; i < cells; ++i)
            if (init->keep[i] && (init->tokens[i] < 0 || init->tokens[i] >= MASK))
                fail("init token out of range");
    }

    bt::DeviceScope scope(m.dev);
    const bool use_cfg = run.guidance_scale != 0.0f;
    Session& s = m.session(static_cast<int>(text_ids.size()), n_ref, T, use_cfg);
    m.write_prefix(s, text_ids, ref_codes);

    // Token grid: MASK everywhere except the init's kept cells.
    std::vector<int32_t> tokens(cells, MASK), unmask(cells, -1);
    int total_masked = static_cast<int>(cells);
    if (init) {
        for (std::size_t i = 0; i < cells; ++i)
            if (init->keep[i]) { tokens[i] = init->tokens[i]; --total_masked; }
    }
    h2d(m.dev, s.tokens.data, tokens.data(), cells * sizeof(int32_t));
    h2d(m.dev, s.unmask.data, unmask.data(), cells * sizeof(int32_t));

    const std::vector<int> schedule = ovp::unmask_schedule(total_masked, run.num_steps, run.t_shift);

    OmniVoiceLmResult res;
    std::vector<int32_t> pred_h;
    std::vector<float> scores_h, conf_h;
    const bool want_hooks = static_cast<bool>(on_step) || (dbg && dbg->on_scores);
    // The raw confidence only leaves the device when somebody asked for it: a
    // per-step callback, the white-box hook, or the trace.
    const bool want_conf = want_hooks || run.want_confidence;
    std::vector<float> conf_final;
    std::vector<int32_t> idx_h;
    if (run.want_confidence) conf_final.assign(cells, 0.0f);
    const auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step < run.num_steps; ++step) {
        if (cancel && cancel()) { res.cancelled = true; break; }

        m.run_body(s);

        if (dbg && dbg->capture_forward0 && step == 0) {
            dbg->L = s.L; dbg->Lc = s.Lc; dbg->R = s.R;
            dbg->embeds0.resize(static_cast<std::size_t>(s.L) * m.H());
            dbg->hidden0.resize(static_cast<std::size_t>(s.R) * m.H());
            dbg->logits0.resize(static_cast<std::size_t>(s.R) * C * V);
            qtd::to_host(s.embeds, dbg->embeds0.data());
            qtd::to_host(s.hnorm, dbg->hidden0.data());
            qtd::to_host(s.logits, dbg->logits0.data());
        }

        const std::uint64_t seed_step = bt::detail::hash_u64(
            run.seed, kStepSeedDomain, (run.chunk << 32) | static_cast<std::uint64_t>(step));
        const float pos_temp = run.gumbel_noise ? run.position_temperature : 0.0f;
        const float cls_temp = run.gumbel_noise ? run.class_temperature : 0.0f;
        bt::masked_diffusion_scores(s.logits, s.tokens, T, C, V, MASK, run.guidance_scale,
                                    run.layer_penalty, pos_temp, cls_temp, 0.1f, seed_step,
                                    s.pred, s.scores, s.conf);

        const int k = schedule[static_cast<std::size_t>(step)];
        bool scores_copied = false;
        if (dbg && dbg->on_scores) {
            pred_h.resize(cells);
            scores_h.resize(cells);
            d2h(m.dev, pred_h.data(), s.pred.data, cells * sizeof(int32_t));
            d2h(m.dev, scores_h.data(), s.scores.data, cells * sizeof(float));
            scores_copied = true;
        }
        if (want_conf) {
            conf_h.resize(cells);
            d2h(m.dev, conf_h.data(), s.conf.data, cells * sizeof(float));
        }
        if (k > 0) {
            bt::Tensor row = bt::Tensor::view(m.dev, s.scores.data, 1, static_cast<int>(cells), bt::Dtype::FP32);
            bt::top_k_rows(row, k, s.vals, s.idx);
            bt::masked_diffusion_commit(s.pred, s.idx, k, step, s.tokens, s.unmask);
            // The cells this step fixed take this step's confidence; the rest
            // keep whatever step fixes them later (or the last step's value).
            if (run.want_confidence) {
                idx_h.resize(static_cast<std::size_t>(k));
                d2h(m.dev, idx_h.data(), s.idx.data, static_cast<std::size_t>(k) * sizeof(int32_t));
                for (int i = 0; i < k; ++i) {
                    const int32_t p = idx_h[static_cast<std::size_t>(i)];
                    if (p >= 0 && static_cast<std::size_t>(p) < cells) conf_final[static_cast<std::size_t>(p)] = conf_h[static_cast<std::size_t>(p)];
                }
            }
        }
        res.steps_run = step + 1;

        if (want_hooks) {
            d2h(m.dev, tokens.data(), s.tokens.data, cells * sizeof(int32_t));
            if (on_step) {
                // Once per step, not once per run: the buffer keeps its size
                // across steps, so a size check would hand every later step the
                // FIRST step's scores.
                if (!scores_copied) {
                    scores_h.resize(cells);
                    d2h(m.dev, scores_h.data(), s.scores.data, cells * sizeof(float));
                }
                OmniVoiceStep st;
                st.step = step; st.num_steps = run.num_steps;
                st.chunk = static_cast<int>(run.chunk); st.num_chunks = run.num_chunks;
                st.num_frames = T; st.num_codebooks = C;
                st.unmasked = k;
                st.tokens = tokens.data();
                st.scores = scores_h.data();
                st.confidence = conf_h.data();
                on_step(st);
            }
            if (dbg && dbg->on_scores) dbg->on_scores(step, k, pred_h, scores_h, conf_h, tokens);
        }
    }

    d2h(m.dev, tokens.data(), s.tokens.data, cells * sizeof(int32_t));
    d2h(m.dev, unmask.data(), s.unmask.data, cells * sizeof(int32_t));
    if (run.want_confidence && conf_h.size() == cells) {
        // A cell an init grid kept is never committed, so it has no "step it
        // was decided at" — give it the last step's confidence.
        for (std::size_t i = 0; i < cells; ++i)
            if (unmask[i] < 0) conf_final[i] = conf_h[i];
        res.confidence = std::move(conf_final);
    }
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    res.codes = std::move(tokens);
    res.unmask_step = std::move(unmask);
    return res;
}

std::vector<float> OmniVoiceLm::mask_frame_embedding() const {
    const Impl& m = *impl_;
    if (!m.loaded) fail("no model loaded");
    bt::DeviceScope scope(m.dev);
    std::vector<int32_t> rows(static_cast<std::size_t>(m.C()));
    for (int c = 0; c < m.C(); ++c) rows[static_cast<std::size_t>(c)] = c * m.V() + m.cfg.audio_mask_id;
    bt::Tensor g = qtd::gather_rows(m.audio_emb, rows);
    std::vector<float> h(static_cast<std::size_t>(m.C()) * m.H());
    qtd::to_host(g, h.data());
    std::vector<float> out(static_cast<std::size_t>(m.H()), 0.0f);
    for (int c = 0; c < m.C(); ++c)
        for (int i = 0; i < m.H(); ++i)
            out[static_cast<std::size_t>(i)] += h[static_cast<std::size_t>(c) * m.H() + i];
    return out;
}

std::vector<float> OmniVoiceLm::text_embeddings(const std::vector<int32_t>& ids) const {
    const Impl& m = *impl_;
    if (!m.loaded) fail("no model loaded");
    for (int32_t id : ids)
        if (id < 0 || id >= m.cfg.vocab_size) fail("text id out of range");
    bt::DeviceScope scope(bt::Device::CPU);
    bt::Tensor g = qtd::gather_rows(m.embed_tokens_cpu, ids);
    std::vector<float> out(ids.size() * static_cast<std::size_t>(m.H()));
    if (!ids.empty()) qtd::to_host(g, out.data());
    return out;
}

const OmniVoiceLmConfig& OmniVoiceLm::config() const { return impl_->cfg; }
bool OmniVoiceLm::loaded() const { return impl_->loaded; }
bool OmniVoiceLm::bf16() const { return impl_->bf16; }
bt::Device OmniVoiceLm::device() const { return impl_->dev; }

}  // namespace brosoundml
