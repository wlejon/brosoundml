// Stage-4 Whisper decoder + KV cache unit tests.
//
// Self-contained: builds a tiny synthetic decoder checkpoint on disk
// (2 layers, d_model=8, ffn=16, heads=2, vocab=32, max_target_positions=16),
// drives it through WhisperDecoder::load_from + forward, and checks shapes,
// cache equivalence, and the tied vs. explicit LM-head paths. No real
// Whisper weights required.
#include "brosoundml/whisper_modules.h"

#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs  = std::filesystem;
namespace bt  = brotensor;
namespace stf = brotensor::safetensors;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

template <typename Fn>
static bool throws_runtime_error(Fn&& fn) {
    try { fn(); }
    catch (const std::runtime_error&) { return true; }
    catch (...) { return false; }
    return false;
}

// ─── Synthetic decoder weight generator ────────────────────────────────────

struct StubBuffers {
    std::vector<std::vector<float>> bufs;
    std::vector<stf::WriteEntry>    entries;

    void add(const std::string& name, const std::vector<std::int64_t>& shape,
             std::vector<float>&& payload) {
        bufs.push_back(std::move(payload));
        stf::WriteEntry e;
        e.name      = name;
        e.dtype     = stf::Dtype::F32;
        e.shape     = shape;
        e.host_data = bufs.back().data();
        e.bytes     = bufs.back().size() * sizeof(float);
        entries.push_back(std::move(e));
    }
};

static std::vector<float> rand_vec(std::mt19937& rng, std::size_t n,
                                   float scale = 0.05f) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

static std::vector<float> zeros_vec(std::size_t n) {
    return std::vector<float>(n, 0.0f);
}

static std::vector<float> ones_vec(std::size_t n) {
    return std::vector<float>(n, 1.0f);
}

// Build a stub safetensors holding every key WhisperDecoder::load_from needs.
// `include_proj_out` controls whether `proj_out.weight` is present (tied vs.
// explicit LM head test paths). Uses a fixed PRNG seed for reproducibility.
static void write_stub_decoder(const fs::path& path,
                               int d_model, int max_tgt,
                               int n_layers, int ffn, int n_heads,
                               int vocab,
                               bool include_proj_out) {
    std::mt19937 rng(54321);
    StubBuffers sb;
    const std::string p = "model.decoder.";

    sb.add(p + "embed_tokens.weight", {vocab, d_model},
           rand_vec(rng, static_cast<std::size_t>(vocab) * d_model));
    sb.add(p + "embed_positions.weight", {max_tgt, d_model},
           rand_vec(rng, static_cast<std::size_t>(max_tgt) * d_model, 0.02f));

    for (int i = 0; i < n_layers; ++i) {
        const std::string lp = p + "layers." + std::to_string(i) + ".";

        sb.add(lp + "self_attn_layer_norm.weight", {d_model}, ones_vec(d_model));
        sb.add(lp + "self_attn_layer_norm.bias",   {d_model}, zeros_vec(d_model));

        sb.add(lp + "self_attn.q_proj.weight", {d_model, d_model},
               rand_vec(rng, static_cast<std::size_t>(d_model) * d_model));
        sb.add(lp + "self_attn.q_proj.bias",   {d_model}, zeros_vec(d_model));
        sb.add(lp + "self_attn.k_proj.weight", {d_model, d_model},
               rand_vec(rng, static_cast<std::size_t>(d_model) * d_model));
        // NB: NO k_proj.bias on disk — load_from must zero-fill it.
        sb.add(lp + "self_attn.v_proj.weight", {d_model, d_model},
               rand_vec(rng, static_cast<std::size_t>(d_model) * d_model));
        sb.add(lp + "self_attn.v_proj.bias",   {d_model}, zeros_vec(d_model));
        sb.add(lp + "self_attn.out_proj.weight", {d_model, d_model},
               rand_vec(rng, static_cast<std::size_t>(d_model) * d_model));
        sb.add(lp + "self_attn.out_proj.bias",   {d_model}, zeros_vec(d_model));

        sb.add(lp + "encoder_attn_layer_norm.weight", {d_model}, ones_vec(d_model));
        sb.add(lp + "encoder_attn_layer_norm.bias",   {d_model}, zeros_vec(d_model));

        sb.add(lp + "encoder_attn.q_proj.weight", {d_model, d_model},
               rand_vec(rng, static_cast<std::size_t>(d_model) * d_model));
        sb.add(lp + "encoder_attn.q_proj.bias",   {d_model}, zeros_vec(d_model));
        sb.add(lp + "encoder_attn.k_proj.weight", {d_model, d_model},
               rand_vec(rng, static_cast<std::size_t>(d_model) * d_model));
        // NB: NO encoder_attn.k_proj.bias on disk.
        sb.add(lp + "encoder_attn.v_proj.weight", {d_model, d_model},
               rand_vec(rng, static_cast<std::size_t>(d_model) * d_model));
        sb.add(lp + "encoder_attn.v_proj.bias",   {d_model}, zeros_vec(d_model));
        sb.add(lp + "encoder_attn.out_proj.weight", {d_model, d_model},
               rand_vec(rng, static_cast<std::size_t>(d_model) * d_model));
        sb.add(lp + "encoder_attn.out_proj.bias",   {d_model}, zeros_vec(d_model));

        sb.add(lp + "final_layer_norm.weight", {d_model}, ones_vec(d_model));
        sb.add(lp + "final_layer_norm.bias",   {d_model}, zeros_vec(d_model));

        sb.add(lp + "fc1.weight", {ffn, d_model},
               rand_vec(rng, static_cast<std::size_t>(ffn) * d_model));
        sb.add(lp + "fc1.bias",   {ffn}, zeros_vec(ffn));
        sb.add(lp + "fc2.weight", {d_model, ffn},
               rand_vec(rng, static_cast<std::size_t>(d_model) * ffn));
        sb.add(lp + "fc2.bias",   {d_model}, zeros_vec(d_model));
    }

    sb.add(p + "layer_norm.weight", {d_model}, ones_vec(d_model));
    sb.add(p + "layer_norm.bias",   {d_model}, zeros_vec(d_model));

    if (include_proj_out) {
        // A distinct random tensor — if the loader honours it, end-to-end
        // logits should differ from the tied path.
        sb.add("proj_out.weight", {vocab, d_model},
               rand_vec(rng, static_cast<std::size_t>(vocab) * d_model));
    }

    (void)n_heads;
    stf::write_file(path.string(), sb.entries);
}

// ─── Scalar reference attention kernels ────────────────────────────────────
//
// Independent reimplementations of the cached attention paths in
// whisper_modules.cpp — used to lock the numerics layer down without trusting
// the same code under test.

static void scalar_causal_attn(const std::vector<float>& X,    // (T, D)
                               int T, int D, int H,
                               const std::vector<float>& Wq,
                               const std::vector<float>& bq,
                               const std::vector<float>& Wk,
                               const std::vector<float>& Wv,
                               const std::vector<float>& bv,
                               const std::vector<float>& Wo,
                               const std::vector<float>& bo,
                               std::vector<float>& Y) {           // (T, D)
    // Linear projection helper: out[t, o] = sum_i W[o, i] * in[t, i] + b[o]
    auto linear = [&](const std::vector<float>& W,
                      const std::vector<float>& b,
                      const std::vector<float>& in, int rows,
                      std::vector<float>& out) {
        out.assign(static_cast<std::size_t>(rows) * D, 0.0f);
        for (int t = 0; t < rows; ++t) {
            for (int o = 0; o < D; ++o) {
                float s = b.empty() ? 0.0f : b[o];
                for (int i = 0; i < D; ++i) {
                    s += W[o * D + i] * in[t * D + i];
                }
                out[t * D + o] = s;
            }
        }
    };

    std::vector<float> Q, K, V;
    linear(Wq, bq, X, T, Q);
    linear(Wk, {}, X, T, K);          // K has no bias
    linear(Wv, bv, X, T, V);

    const int hd  = D / H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    std::vector<float> ctx(static_cast<std::size_t>(T) * D, 0.0f);
    std::vector<float> scores(T);

    for (int h = 0; h < H; ++h) {
        for (int q = 0; q < T; ++q) {
            float max_s = -INFINITY;
            for (int k = 0; k <= q; ++k) {     // causal
                float s = 0.0f;
                for (int j = 0; j < hd; ++j) {
                    s += Q[q * D + h * hd + j] * K[k * D + h * hd + j];
                }
                s *= scale;
                scores[k] = s;
                if (s > max_s) max_s = s;
            }
            float sum = 0.0f;
            for (int k = 0; k <= q; ++k) {
                scores[k] = std::exp(scores[k] - max_s);
                sum += scores[k];
            }
            for (int k = 0; k <= q; ++k) scores[k] /= sum;
            for (int k = 0; k <= q; ++k) {
                for (int j = 0; j < hd; ++j) {
                    ctx[q * D + h * hd + j] += scores[k] * V[k * D + h * hd + j];
                }
            }
        }
    }

    // Out projection.
    Y.assign(static_cast<std::size_t>(T) * D, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < D; ++o) {
            float s = bo[o];
            for (int i = 0; i < D; ++i) s += Wo[o * D + i] * ctx[t * D + i];
            Y[t * D + o] = s;
        }
    }
}

static void scalar_cross_attn(const std::vector<float>& X,    // (Tq, D)
                              int Tq, int D, int H,
                              const std::vector<float>& K_cached,   // (Lk, D)
                              const std::vector<float>& V_cached,   // (Lk, D)
                              int Lk,
                              const std::vector<float>& Wq,
                              const std::vector<float>& bq,
                              const std::vector<float>& Wo,
                              const std::vector<float>& bo,
                              std::vector<float>& Y) {           // (Tq, D)
    std::vector<float> Q(static_cast<std::size_t>(Tq) * D, 0.0f);
    for (int t = 0; t < Tq; ++t) {
        for (int o = 0; o < D; ++o) {
            float s = bq[o];
            for (int i = 0; i < D; ++i) s += Wq[o * D + i] * X[t * D + i];
            Q[t * D + o] = s;
        }
    }

    const int hd = D / H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> ctx(static_cast<std::size_t>(Tq) * D, 0.0f);
    std::vector<float> scores(Lk);
    for (int h = 0; h < H; ++h) {
        for (int q = 0; q < Tq; ++q) {
            float max_s = -INFINITY;
            for (int k = 0; k < Lk; ++k) {
                float s = 0.0f;
                for (int j = 0; j < hd; ++j) {
                    s += Q[q * D + h * hd + j] * K_cached[k * D + h * hd + j];
                }
                s *= scale;
                scores[k] = s;
                if (s > max_s) max_s = s;
            }
            float sum = 0.0f;
            for (int k = 0; k < Lk; ++k) {
                scores[k] = std::exp(scores[k] - max_s);
                sum += scores[k];
            }
            for (int k = 0; k < Lk; ++k) scores[k] /= sum;
            for (int k = 0; k < Lk; ++k) {
                for (int j = 0; j < hd; ++j) {
                    ctx[q * D + h * hd + j] += scores[k] * V_cached[k * D + h * hd + j];
                }
            }
        }
    }

    Y.assign(static_cast<std::size_t>(Tq) * D, 0.0f);
    for (int t = 0; t < Tq; ++t) {
        for (int o = 0; o < D; ++o) {
            float s = bo[o];
            for (int i = 0; i < D; ++i) s += Wo[o * D + i] * ctx[t * D + i];
            Y[t * D + o] = s;
        }
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────────

static bt::Tensor make_f32_on(bt::Device dev, const std::vector<float>& v,
                              int rows, int cols) {
    return bt::Tensor::from_host_on(dev, v.data(), rows, cols);
}

static std::string tagmsg(const char* msg, const char* dev_name) {
    std::string s = "["; s += dev_name; s += "] "; s += msg; return s;
}

static std::vector<float> deterministic(std::size_t n, std::uint32_t seed,
                                        float scale = 0.1f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ─── KV-cache lifecycle test ───────────────────────────────────────────────

static void test_cache_allocate_and_reset(bt::Device dev, const char* dev_name) {
    brosoundml::WhisperKVCache c;
    c.allocate(/*layers=*/3, /*d=*/8, /*max_tgt=*/16, /*max_src=*/20, dev);
    CHECK(static_cast<int>(c.layers.size()) == 3,
          tagmsg("cache.allocate creates one slab per decoder layer", dev_name).c_str());
    CHECK(c.size() == 0, tagmsg("fresh cache has size 0", dev_name).c_str());
    for (auto& l : c.layers) {
        CHECK(l.self_k.rows == 16 && l.self_k.cols == 8,
              tagmsg("self_k preallocated to (max_tgt, d)", dev_name).c_str());
        CHECK(l.self_v.rows == 16 && l.self_v.cols == 8,
              tagmsg("self_v preallocated to (max_tgt, d)", dev_name).c_str());
        CHECK(l.cross_k.rows == 20 && l.cross_k.cols == 8,
              tagmsg("cross_k preallocated to (max_src, d)", dev_name).c_str());
        CHECK(l.cross_v.rows == 20 && l.cross_v.cols == 8,
              tagmsg("cross_v preallocated to (max_src, d)", dev_name).c_str());
        CHECK(l.self_len == 0,
              tagmsg("self_len initialised to 0", dev_name).c_str());
        CHECK(l.cross_primed == false,
              tagmsg("cross_primed initialised to false", dev_name).c_str());
    }

    c.layers[0].self_len = 5;
    c.layers[1].cross_primed = true;
    c.reset();
    CHECK(c.size() == 0, tagmsg("reset zeros size()", dev_name).c_str());
    CHECK(c.layers[0].self_len == 0 && c.layers[1].cross_primed == false,
          tagmsg("reset clears self_len and cross_primed on every layer", dev_name).c_str());
}

// ─── mha_causal_cached_fp32 numeric test ──────────────────────────────────

static void test_mha_causal_cached_matches_scalar_T1(bt::Device dev,
                                                      const char* dev_name) {
    const int D = 4, H = 2, max_tgt = 8;
    auto Wq = deterministic(D * D, 11), bq = deterministic(D, 12);
    auto Wk = deterministic(D * D, 13);
    auto Wv = deterministic(D * D, 14), bv = deterministic(D, 15);
    auto Wo = deterministic(D * D, 16), bo = deterministic(D, 17);

    auto X1 = deterministic(D, 21, 0.2f);

    std::vector<float> ref;
    scalar_causal_attn(X1, /*T=*/1, D, H, Wq, bq, Wk, Wv, bv, Wo, bo, ref);

    brosoundml::WhisperKVCache c;
    c.allocate(/*layers=*/1, D, max_tgt, /*max_src=*/4, dev);
    bt::Tensor X    = make_f32_on(dev, X1, 1, D);
    bt::Tensor Wq_t = make_f32_on(dev, Wq, D, D), bq_t = make_f32_on(dev, bq, D, 1);
    bt::Tensor Wk_t = make_f32_on(dev, Wk, D, D);
    bt::Tensor bk_t = bt::Tensor::zeros_on(dev, D, 1, bt::Dtype::FP32);
    bt::Tensor Wv_t = make_f32_on(dev, Wv, D, D), bv_t = make_f32_on(dev, bv, D, 1);
    bt::Tensor Wo_t = make_f32_on(dev, Wo, D, D), bo_t = make_f32_on(dev, bo, D, 1);

    bt::Tensor out;
    brosoundml::mha_causal_cached_fp32(X, Wq_t, bq_t, Wk_t, bk_t,
                                       Wv_t, bv_t, Wo_t, bo_t, H,
                                       c.layers[0], out);
    CHECK(out.rows == 1 && out.cols == D,
          tagmsg("mha_causal_cached_fp32 T=1 output shape", dev_name).c_str());
    CHECK(c.layers[0].self_len == 1,
          tagmsg("self_len grew to 1 after a single-step call", dev_name).c_str());

    std::vector<float> od = out.to_host_vector();
    const float tol = (dev == bt::Device::CPU) ? 1e-5f : 1e-3f;
    bool close = true;
    for (int i = 0; i < D; ++i) {
        if (std::abs(od[i] - ref[i]) > tol) close = false;
    }
    CHECK(close, tagmsg("mha_causal_cached_fp32 T=1 matches scalar reference",
                        dev_name).c_str());
}

static void test_mha_causal_cached_prefill_then_step_equivalence(
        bt::Device dev, const char* dev_name) {
    const int D = 4, H = 2, max_tgt = 16;
    auto Wq = deterministic(D * D, 31), bq = deterministic(D, 32);
    auto Wk = deterministic(D * D, 33);
    auto Wv = deterministic(D * D, 34), bv = deterministic(D, 35);
    auto Wo = deterministic(D * D, 36), bo = deterministic(D, 37);

    bt::Tensor Wq_t = make_f32_on(dev, Wq, D, D), bq_t = make_f32_on(dev, bq, D, 1);
    bt::Tensor Wk_t = make_f32_on(dev, Wk, D, D);
    bt::Tensor bk_t = bt::Tensor::zeros_on(dev, D, 1, bt::Dtype::FP32);
    bt::Tensor Wv_t = make_f32_on(dev, Wv, D, D), bv_t = make_f32_on(dev, bv, D, 1);
    bt::Tensor Wo_t = make_f32_on(dev, Wo, D, D), bo_t = make_f32_on(dev, bo, D, 1);

    auto X_all = deterministic(7 * D, 41, 0.2f);

    brosoundml::WhisperKVCache cA;
    cA.allocate(1, D, max_tgt, 4, dev);
    bt::Tensor XA = make_f32_on(dev, X_all, 7, D);
    bt::Tensor outA;
    brosoundml::mha_causal_cached_fp32(XA, Wq_t, bq_t, Wk_t, bk_t,
                                       Wv_t, bv_t, Wo_t, bo_t, H,
                                       cA.layers[0], outA);
    CHECK(cA.layers[0].self_len == 7,
          tagmsg("single prefill grows self_len to 7", dev_name).c_str());

    brosoundml::WhisperKVCache cB;
    cB.allocate(1, D, max_tgt, 4, dev);
    bt::Tensor XB0 = make_f32_on(dev,
                                 std::vector<float>(X_all.begin(),
                                                    X_all.begin() + 4 * D),
                                 4, D);
    bt::Tensor outB0;
    brosoundml::mha_causal_cached_fp32(XB0, Wq_t, bq_t, Wk_t, bk_t,
                                       Wv_t, bv_t, Wo_t, bo_t, H,
                                       cB.layers[0], outB0);
    std::vector<std::vector<float>> step_outs_h;
    for (int s = 0; s < 3; ++s) {
        std::vector<float> row(X_all.begin() + (4 + s) * D,
                               X_all.begin() + (4 + s + 1) * D);
        bt::Tensor xs = make_f32_on(dev, row, 1, D);
        bt::Tensor ys;
        brosoundml::mha_causal_cached_fp32(xs, Wq_t, bq_t, Wk_t, bk_t,
                                           Wv_t, bv_t, Wo_t, bo_t, H,
                                           cB.layers[0], ys);
        step_outs_h.push_back(ys.to_host_vector());
    }
    CHECK(cB.layers[0].self_len == 7,
          tagmsg("prefill-then-steps grows self_len to 7", dev_name).c_str());

    std::vector<float> aA = outA.to_host_vector();
    std::vector<float> outB0_h = outB0.to_host_vector();
    const float tol = (dev == bt::Device::CPU) ? 1e-4f : 3e-3f;
    bool ok = true;
    for (int t = 0; t < 4; ++t) {
        for (int j = 0; j < D; ++j) {
            const float ref = outB0_h[t * D + j];
            if (std::abs(aA[t * D + j] - ref) > tol) ok = false;
        }
    }
    for (int s = 0; s < 3; ++s) {
        const auto& sd = step_outs_h[s];
        for (int j = 0; j < D; ++j) {
            if (std::abs(aA[(4 + s) * D + j] - sd[j]) > tol) ok = false;
        }
    }
    CHECK(ok, tagmsg("prefill T=7 == prefill T=4 + 3 single steps (cache equivalence)",
                     dev_name).c_str());
}

// ─── cross_attn_cached_fp32 numeric test ──────────────────────────────────

static void test_cross_attn_cached_matches_scalar(bt::Device dev,
                                                   const char* dev_name) {
    const int D = 4, H = 2, Lk = 5;
    auto Wq = deterministic(D * D, 51), bq = deterministic(D, 52);
    auto Wo = deterministic(D * D, 53), bo = deterministic(D, 54);

    auto Kdata = deterministic(Lk * D, 55, 0.3f);
    auto Vdata = deterministic(Lk * D, 56, 0.3f);

    auto X = deterministic(2 * D, 57, 0.2f);
    std::vector<float> ref;
    scalar_cross_attn(X, /*Tq=*/2, D, H, Kdata, Vdata, Lk,
                      Wq, bq, Wo, bo, ref);

    // Seed the cross-K/V from host data by going through from_host_on and then
    // copying into the cache slabs. For non-CPU devices we use a host
    // round-trip: build a CPU tensor with the data, then `.to(dev)` and
    // overwrite the cache slab via Tensor copy semantics — but the cache
    // slabs are zeros_on(dev) allocations, so an upload-into-slab needs a
    // copy_d2d. Simplest: just construct a fresh device tensor and reseat the
    // slab via move-assignment (the WhisperLayerCache cross_k/cross_v are
    // plain Tensor members).
    brosoundml::WhisperKVCache c;
    c.allocate(1, D, /*max_tgt=*/4, /*max_src=*/Lk, dev);
    c.layers[0].cross_k = bt::Tensor::from_host_on(dev, Kdata.data(), Lk, D);
    c.layers[0].cross_v = bt::Tensor::from_host_on(dev, Vdata.data(), Lk, D);
    c.layers[0].cross_primed = true;

    bt::Tensor X_t = make_f32_on(dev, X, 2, D);
    bt::Tensor Wq_t = make_f32_on(dev, Wq, D, D), bq_t = make_f32_on(dev, bq, D, 1);
    bt::Tensor Wo_t = make_f32_on(dev, Wo, D, D), bo_t = make_f32_on(dev, bo, D, 1);
    bt::Tensor out;
    brosoundml::cross_attn_cached_fp32(X_t, Wq_t, bq_t, Wo_t, bo_t, H,
                                       c.layers[0], out);

    CHECK(out.rows == 2 && out.cols == D,
          tagmsg("cross_attn_cached_fp32 output shape (2, D)", dev_name).c_str());
    const float tol = (dev == bt::Device::CPU) ? 1e-5f : 1e-3f;
    bool ok = true;
    std::vector<float> od = out.to_host_vector();
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(od[i] - ref[i]) > tol) ok = false;
    }
    CHECK(ok, tagmsg("cross_attn_cached_fp32 matches scalar reference",
                     dev_name).c_str());

    brosoundml::WhisperKVCache c2;
    c2.allocate(1, D, 4, Lk, dev);
    bt::Tensor tmp;
    CHECK(throws_runtime_error([&] {
        brosoundml::cross_attn_cached_fp32(X_t, Wq_t, bq_t, Wo_t, bo_t, H,
                                           c2.layers[0], tmp);
    }), tagmsg("cross_attn_cached_fp32 rejects un-primed cache", dev_name).c_str());
}

// ─── DecoderLayer shape preservation ──────────────────────────────────────

static void test_decoder_layer_forward_shape(bt::Device dev, const char* dev_name) {
    const int D = 8, ffn = 16, H = 2, max_tgt = 16, max_src = 6;
    const int vocab = 32, n_layers = 1;
    const fs::path tmp = fs::temp_directory_path() /
                         "brosoundml_whisper_decoder_layer.safetensors";
    fs::remove(tmp);
    write_stub_decoder(tmp, D, max_tgt, n_layers, ffn, H, vocab,
                       /*include_proj_out=*/false);
    brosoundml::WhisperDecoderLayer layer;
    {
        stf::File f = stf::File::open(tmp.string());
        layer.load_from(f, "model.decoder.layers.0.", D, ffn, H, dev);
    }
    CHECK(layer.self_bk.rows == D && layer.self_bk.cols == 1,
          tagmsg("self-attn k_proj bias zero-filled to (D, 1)", dev_name).c_str());
    CHECK(layer.cross_bk.rows == D && layer.cross_bk.cols == 1,
          tagmsg("cross-attn k_proj bias zero-filled to (D, 1)", dev_name).c_str());
    std::vector<float> sbk = layer.self_bk.to_host_vector();
    std::vector<float> cbk = layer.cross_bk.to_host_vector();
    bool zero = true;
    for (int i = 0; i < D; ++i) {
        if (sbk[i] != 0.0f || cbk[i] != 0.0f) zero = false;
    }
    CHECK(zero, tagmsg("both k_proj biases are zero", dev_name).c_str());

    brosoundml::WhisperKVCache cache;
    cache.allocate(n_layers, D, max_tgt, max_src, dev);
    auto edata = deterministic(max_src * D, 71, 0.2f);
    bt::Tensor enc = bt::Tensor::from_host_on(dev, edata.data(), max_src, D);
    layer.prime_cross(enc, cache.layers[0]);
    CHECK(cache.layers[0].cross_primed,
          tagmsg("prime_cross flips cross_primed", dev_name).c_str());

    auto xdata = deterministic(D, 72, 0.2f);
    bt::Tensor X = make_f32_on(dev, xdata, 1, D);
    bt::Tensor Y;
    layer.forward(X, cache.layers[0], Y);
    CHECK(Y.rows == 1 && Y.cols == D,
          tagmsg("decoder layer preserves (1, D)", dev_name).c_str());
    CHECK(cache.layers[0].self_len == 1,
          tagmsg("layer.forward grew self_len by 1", dev_name).c_str());

    brosoundml::WhisperKVCache cache2;
    cache2.allocate(n_layers, D, max_tgt, max_src, dev);
    layer.prime_cross(enc, cache2.layers[0]);
    auto xdata3 = deterministic(3 * D, 73, 0.2f);
    bt::Tensor X3 = make_f32_on(dev, xdata3, 3, D);
    bt::Tensor Y3;
    layer.forward(X3, cache2.layers[0], Y3);
    CHECK(Y3.rows == 3 && Y3.cols == D,
          tagmsg("decoder layer preserves (T, D)", dev_name).c_str());
    CHECK(cache2.layers[0].self_len == 3,
          tagmsg("self_len grew by 3", dev_name).c_str());

    std::vector<float> yd = Y3.to_host_vector();
    bool finite = true;
    for (int i = 0; i < 3 * D; ++i) if (!std::isfinite(yd[i])) finite = false;
    CHECK(finite, tagmsg("decoder layer output is all-finite", dev_name).c_str());

    fs::remove(tmp);
}

// ─── End-to-end WhisperDecoder forward test ───────────────────────────────

static void test_decoder_forward_e2e_and_cache_equivalence(bt::Device dev,
                                                            const char* dev_name) {
    const int D = 8, ffn = 16, H = 2;
    const int n_layers = 2, vocab = 32, max_tgt = 16, max_src = 6;

    const fs::path tmp = fs::temp_directory_path() /
                         "brosoundml_whisper_decoder_e2e.safetensors";
    fs::remove(tmp);
    write_stub_decoder(tmp, D, max_tgt, n_layers, ffn, H, vocab,
                       /*include_proj_out=*/false);
    brosoundml::WhisperDecoder dec;
    {
        stf::File f = stf::File::open(tmp.string());
        dec.load_from(f, D, n_layers, ffn, H, vocab, max_tgt, max_src, dev);
    }
    CHECK(static_cast<int>(dec.layers.size()) == n_layers,
          tagmsg("WhisperDecoder loads requested layer count", dev_name).c_str());
    CHECK(dec.proj_out_explicit == false,
          tagmsg("no proj_out.weight on disk -> tied LM head", dev_name).c_str());
    CHECK(dec.embed_tokens.rows == vocab && dec.embed_tokens.cols == D,
          tagmsg("embed_tokens shape (vocab, d)", dev_name).c_str());
    CHECK(dec.embed_positions.rows == max_tgt && dec.embed_positions.cols == D,
          tagmsg("embed_positions shape (max_tgt, d)", dev_name).c_str());

    auto edata = deterministic(max_src * D, 81, 0.2f);
    bt::Tensor enc = bt::Tensor::from_host_on(dev, edata.data(), max_src, D);

    const std::int32_t prompt[5] = {2, 7, 3, 11, 5};
    brosoundml::WhisperKVCache cacheA;
    cacheA.allocate(n_layers, D, max_tgt, max_src, dev);
    dec.prime_cross(enc, cacheA);
    bt::Tensor logitsA;
    dec.forward(prompt, 5, /*pos_offset=*/0, cacheA, logitsA);
    CHECK(logitsA.rows == 5 && logitsA.cols == vocab,
          tagmsg("decoder prefill produces (T, vocab) logits", dev_name).c_str());
    CHECK(cacheA.size() == 5,
          tagmsg("decoder prefill grew cache by 5", dev_name).c_str());

    std::vector<float> la = logitsA.to_host_vector();
    bool finite_A = true;
    for (int i = 0; i < 5 * vocab; ++i) if (!std::isfinite(la[i])) finite_A = false;
    CHECK(finite_A, tagmsg("decoder prefill logits all-finite", dev_name).c_str());

    brosoundml::WhisperKVCache cacheB;
    cacheB.allocate(n_layers, D, max_tgt, max_src, dev);
    dec.prime_cross(enc, cacheB);
    bt::Tensor logitsB_pre;
    dec.forward(prompt, 4, 0, cacheB, logitsB_pre);
    CHECK(logitsB_pre.rows == 4 && logitsB_pre.cols == vocab,
          tagmsg("decoder prefill of 4 produces (4, vocab)", dev_name).c_str());

    bt::Tensor logitsB_step;
    dec.forward(&prompt[4], 1, /*pos_offset=*/4, cacheB, logitsB_step);
    CHECK(logitsB_step.rows == 1 && logitsB_step.cols == vocab,
          tagmsg("decoder single-step produces (1, vocab)", dev_name).c_str());
    CHECK(cacheB.size() == 5,
          tagmsg("decoder cache grew to 5 after the single step", dev_name).c_str());

    std::vector<float> lb = logitsB_step.to_host_vector();
    const float tol = (dev == bt::Device::CPU) ? 5e-4f : 5e-3f;
    bool ok = true;
    float max_err = 0.0f;
    for (int v = 0; v < vocab; ++v) {
        const float e = std::abs(lb[v] - la[4 * vocab + v]);
        if (e > max_err) max_err = e;
        if (e > tol) ok = false;
    }
    if (!ok) {
        std::fprintf(stderr, "  [%s] cache-equiv max error: %g\n", dev_name, max_err);
    }
    CHECK(ok, tagmsg("single-step logits at pos 4 == prefill-of-5 last row",
                     dev_name).c_str());

    brosoundml::WhisperKVCache cacheC;
    cacheC.allocate(n_layers, D, max_tgt, max_src, dev);
    bt::Tensor tmp_logits;
    CHECK(throws_runtime_error([&] {
        dec.forward(prompt, 1, 0, cacheC, tmp_logits);
    }), tagmsg("decoder forward refuses un-primed cross-attn cache",
               dev_name).c_str());

    CHECK(throws_runtime_error([&] {
        dec.forward(prompt, 1, /*pos_offset=*/3, cacheA, tmp_logits);
    }), tagmsg("decoder forward refuses pos_offset != cache.self_len",
               dev_name).c_str());

    fs::remove(tmp);
}

// ─── Tied vs. explicit LM head ────────────────────────────────────────────

static void test_lm_head_tied_and_explicit(bt::Device dev, const char* dev_name) {
    const int D = 8, ffn = 16, H = 2;
    const int n_layers = 1, vocab = 32, max_tgt = 8, max_src = 4;

    const fs::path tied = fs::temp_directory_path() /
                          "brosoundml_whisper_decoder_tied.safetensors";
    const fs::path expl = fs::temp_directory_path() /
                          "brosoundml_whisper_decoder_explicit.safetensors";
    fs::remove(tied); fs::remove(expl);
    write_stub_decoder(tied, D, max_tgt, n_layers, ffn, H, vocab, false);
    write_stub_decoder(expl, D, max_tgt, n_layers, ffn, H, vocab, true);

    brosoundml::WhisperDecoder dT, dE;
    {
        stf::File f = stf::File::open(tied.string());
        dT.load_from(f, D, n_layers, ffn, H, vocab, max_tgt, max_src, dev);
    }
    {
        stf::File f = stf::File::open(expl.string());
        dE.load_from(f, D, n_layers, ffn, H, vocab, max_tgt, max_src, dev);
    }
    CHECK(!dT.proj_out_explicit,
          tagmsg("tied checkpoint reports proj_out_explicit=false", dev_name).c_str());
    CHECK(dE.proj_out_explicit,
          tagmsg("explicit checkpoint reports proj_out_explicit=true", dev_name).c_str());
    CHECK(dT.proj_out_weight.rows == 0 || dT.proj_out_weight.cols == 0,
          tagmsg("tied path leaves proj_out_weight empty", dev_name).c_str());
    CHECK(dE.proj_out_weight.rows == vocab && dE.proj_out_weight.cols == D,
          tagmsg("explicit path loads proj_out_weight (vocab, d)", dev_name).c_str());

    auto edata = deterministic(max_src * D, 91, 0.2f);
    bt::Tensor enc = bt::Tensor::from_host_on(dev, edata.data(), max_src, D);

    auto run = [&](brosoundml::WhisperDecoder& d, bt::Tensor& logits) {
        brosoundml::WhisperKVCache c;
        c.allocate(n_layers, D, max_tgt, max_src, dev);
        d.prime_cross(enc, c);
        const std::int32_t p[2] = {1, 4};
        d.forward(p, 2, 0, c, logits);
    };
    bt::Tensor lT, lE;
    run(dT, lT); run(dE, lE);
    CHECK(lT.rows == 2 && lT.cols == vocab,
          tagmsg("tied path logits shape", dev_name).c_str());
    CHECK(lE.rows == 2 && lE.cols == vocab,
          tagmsg("explicit path logits shape", dev_name).c_str());

    std::vector<float> lT_h = lT.to_host_vector();
    std::vector<float> lE_h = lE.to_host_vector();
    bool finite = true;
    for (int i = 0; i < 2 * vocab; ++i) {
        if (!std::isfinite(lT_h[i])) finite = false;
        if (!std::isfinite(lE_h[i])) finite = false;
    }
    CHECK(finite, tagmsg("both LM-head paths produce finite logits",
                         dev_name).c_str());

    fs::remove(tied); fs::remove(expl);
}

static void run_all(bt::Device dev, const char* dev_name) {
    test_cache_allocate_and_reset(dev, dev_name);
    test_mha_causal_cached_matches_scalar_T1(dev, dev_name);
    test_mha_causal_cached_prefill_then_step_equivalence(dev, dev_name);
    test_cross_attn_cached_matches_scalar(dev, dev_name);
    test_decoder_layer_forward_shape(dev, dev_name);
    test_decoder_forward_e2e_and_cache_equivalence(dev, dev_name);
    test_lm_head_tied_and_explicit(dev, dev_name);
}

int main() {
    bt::init();
    try {
        run_all(bt::Device::CPU, "CPU");
        if (bt::is_available(bt::Device::CUDA)) {
            run_all(bt::Device::CUDA, "CUDA");
        } else {
            std::printf("test_whisper_decoder: CUDA not available — CUDA path skipped\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_whisper_decoder: uncaught exception: %s\n",
                     e.what());
        return 2;
    } catch (...) {
        std::fprintf(stderr, "test_whisper_decoder: uncaught non-std exception\n");
        return 2;
    }
    if (failures) {
        std::fprintf(stderr, "test_whisper_decoder: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_whisper_decoder: all checks passed\n");
    return 0;
}
