// brosoundml/decoder_lora.cpp — a trainable LoRA over the Kokoro decoder's
// gate, the per-fc adapters, and the DecoderLora model (cached forward/backward
// over the decoder back half, Adam, checkpoint I/O).

#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "brosoundml/decoder_lora.h"

#include "brosoundml/kokoro_modules.h"   // DecoderBackbone, Generator, Linear
#include "kokoro_decoder_backward.h"     // detail:: assemblies + cache/affine types

#include <brotensor/ops/activation.h>
#include <brotensor/ops/concat.h>        // copy_d2d
#include <brotensor/ops/linear.h>
#include <brotensor/ops/lora.h>
#include <brotensor/ops/optim.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace bt = brotensor;

namespace brosoundml {

ConditioningGate ConditioningGate::make(int gate_dim, int hidden, bt::Device dev,
                      std::uint64_t seed) {
    ConditioningGate g;
    g.gate_dim = gate_dim;
    g.hidden = hidden;
    g.in_dim = 3;

    // xavier_init is a host-side op — initialise on CPU, then place on `dev`.
    bt::Tensor W1 = bt::Tensor::zeros_on(bt::Device::CPU, hidden, g.in_dim);
    std::uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ull;
    bt::xavier_init(W1, rng);
    g.W1 = W1.to(dev);

    // W2 zero-initialised: g(v) starts at 0 for all v, so the adapter delta
    // starts at 0 (LoRA B=0 convention) and training departs smoothly.
    g.W2 = bt::Tensor::zeros_on(dev, gate_dim, hidden);
    return g;
}

void ConditioningGate::zero_grad() {
    const bt::Device dev = W1.device;
    dW1 = bt::Tensor::zeros_on(dev, hidden, in_dim);
    dW2 = bt::Tensor::zeros_on(dev, gate_dim, hidden);
}

void ConditioningGate::forward(const bt::Tensor& v, bt::Tensor& g) {
    const bt::Device dev = W1.device;
    // pre1 = W1 v   (no bias)
    bt::Tensor zb1 = bt::Tensor::zeros_on(dev, hidden, 1);
    bt::linear_forward(W1, zb1, v, pre1_);
    // act = tanh(pre1)
    bt::tanh_forward(pre1_, act_);
    // g = W2 act   (no bias)
    bt::Tensor zb2 = bt::Tensor::zeros_on(dev, gate_dim, 1);
    bt::linear_forward(W2, zb2, act_, g);
}

void ConditioningGate::backward(const bt::Tensor& v, const bt::Tensor& dG) {
    const bt::Device dev = W1.device;
    // through W2:  dW2 += dG act^T ;  dAct = W2^T dG
    bt::Tensor dAct, dB2 = bt::Tensor::zeros_on(dev, gate_dim, 1);
    bt::linear_backward(W2, act_, dG, dAct, dW2, dB2);
    // through tanh:  dPre1 = dAct * (1 - act^2)
    bt::Tensor dPre1;
    bt::tanh_backward(act_, dAct, dPre1);
    // through W1:  dW1 += dPre1 v^T ;  dV discarded (control input)
    bt::Tensor dV, dB1 = bt::Tensor::zeros_on(dev, hidden, 1);
    bt::linear_backward(W1, v, dPre1, dV, dW1, dB1);
}

// ─── DecoderLora ────────────────────────────────────────────────────────────

namespace {

DecoderLoraAdapter make_adapter(const Linear* fc, int C, int rank, int style_dim,
                            bt::Device dev, std::uint64_t& rng) {
    DecoderLoraAdapter ad;
    ad.fc = fc;
    ad.C = C;
    // A small (xavier), B small (xavier). The gate's W2=0 makes the delta zero
    // at init regardless, so neither needs the classic B=0 — and keeping both
    // non-zero lets gradient reach A and B as soon as the gate opens.
    bt::Tensor A = bt::Tensor::zeros_on(bt::Device::CPU, rank, style_dim);
    bt::xavier_init(A, rng);
    ad.A = A.to(dev);
    bt::Tensor B = bt::Tensor::zeros_on(bt::Device::CPU, 2 * C, rank);
    bt::xavier_init(B, rng);
    ad.B = B.to(dev);
    ad.mA = bt::Tensor::zeros_on(dev, rank, style_dim);
    ad.vA = bt::Tensor::zeros_on(dev, rank, style_dim);
    ad.mB = bt::Tensor::zeros_on(dev, 2 * C, rank);
    ad.vB = bt::Tensor::zeros_on(dev, 2 * C, rank);
    ad.dA = bt::Tensor::zeros_on(dev, rank, style_dim);
    ad.dB = bt::Tensor::zeros_on(dev, 2 * C, rank);
    return ad;
}

}  // namespace

DecoderLora DecoderLora::make(const DecoderBackbone& bb, const Generator& gen,
                              int rank, float scale, bt::Device dev,
                              std::uint64_t seed) {
    DecoderLora m;
    m.rank_ = rank;
    m.scale_ = scale;
    m.dev_ = dev;
    m.style_dim_ = bb.encode.norm1.fc.in_features;   // 128 for Kokoro
    std::uint64_t rng = seed ? seed : 0xD1B54A32D192ED03ull;

    auto push_block = [&](const AdainResBlk1dWeights& blk) {
        std::array<DecoderLoraAdapter, 2> pr;
        pr[0] = make_adapter(&blk.norm1.fc, blk.channels_in,  rank, m.style_dim_, dev, rng);
        pr[1] = make_adapter(&blk.norm2.fc, blk.channels_out, rank, m.style_dim_, dev, rng);
        m.bb_.push_back(std::move(pr));
    };
    push_block(bb.encode);
    for (const auto& d : bb.decode) push_block(d);

    auto resblock1_adapters = [&](const AdaINResBlock1Weights& w) {
        std::array<DecoderLoraAdapter, 6> a;
        for (int j = 0; j < 3; ++j)
            a[j] = make_adapter(&w.adain1[j].fc, w.channels, rank, m.style_dim_, dev, rng);
        for (int j = 0; j < 3; ++j)
            a[3 + j] = make_adapter(&w.adain2[j].fc, w.channels, rank, m.style_dim_, dev, rng);
        return a;
    };
    for (const auto& nr : gen.noise_res) m.noise_.push_back(resblock1_adapters(nr));
    for (const auto& rb : gen.resblocks) m.res_.push_back(resblock1_adapters(rb));

    // Shared gate: width = rank, a small hidden layer.
    m.gate_ = ConditioningGate::make(rank, /*hidden=*/16, dev, seed ^ 0xABCDEFu);
    m.gate_.zero_grad();
    m.gW1_m_ = bt::Tensor::zeros_on(dev, m.gate_.hidden, m.gate_.in_dim);
    m.gW1_v_ = bt::Tensor::zeros_on(dev, m.gate_.hidden, m.gate_.in_dim);
    m.gW2_m_ = bt::Tensor::zeros_on(dev, m.gate_.gate_dim, m.gate_.hidden);
    m.gW2_v_ = bt::Tensor::zeros_on(dev, m.gate_.gate_dim, m.gate_.hidden);
    return m;
}

DecoderLora::~DecoderLora() {
    delete static_cast<detail::BackboneCache*>(bb_cache_);
    delete static_cast<detail::GeneratorCache*>(gen_cache_);
}

DecoderLora::DecoderLora(DecoderLora&& o) noexcept { *this = std::move(o); }

DecoderLora& DecoderLora::operator=(DecoderLora&& o) noexcept {
    if (this != &o) {
        delete static_cast<detail::BackboneCache*>(bb_cache_);
        delete static_cast<detail::GeneratorCache*>(gen_cache_);
        rank_ = o.rank_; scale_ = o.scale_; style_dim_ = o.style_dim_;
        dev_ = o.dev_; step_ = o.step_;
        gate_ = std::move(o.gate_);
        bb_ = std::move(o.bb_); noise_ = std::move(o.noise_); res_ = std::move(o.res_);
        gW1_m_ = std::move(o.gW1_m_); gW1_v_ = std::move(o.gW1_v_);
        gW2_m_ = std::move(o.gW2_m_); gW2_v_ = std::move(o.gW2_v_);
        style_col_ = std::move(o.style_col_); g_ = std::move(o.g_);
        bb_cache_ = o.bb_cache_; gen_cache_ = o.gen_cache_;
        o.bb_cache_ = nullptr; o.gen_cache_ = nullptr;
    }
    return *this;
}

std::vector<DecoderLoraAdapter*> DecoderLora::adapters() {
    std::vector<DecoderLoraAdapter*> v;
    for (auto& pr : bb_) { v.push_back(&pr[0]); v.push_back(&pr[1]); }
    for (auto& a : noise_) for (auto& ad : a) v.push_back(&ad);
    for (auto& a : res_)   for (auto& ad : a) v.push_back(&ad);
    return v;
}

namespace {

// Run one adapter's LoRA forward and split the (2C,1) output into (gamma,beta).
void adapter_fwd(DecoderLoraAdapter& ad, const bt::Tensor& style_col,
                 const bt::Tensor& g, float scale, bt::Device dev,
                 bt::Tensor& gamma, bt::Tensor& beta) {
    bt::Tensor y;
    bt::lora_forward(ad.fc->W, ad.fc->b, style_col, ad.A, ad.B, scale, &g,
                     y, ad.h, ad.hg);
    gamma = bt::Tensor::zeros_on(dev, ad.C, 1);
    beta  = bt::Tensor::zeros_on(dev, ad.C, 1);
    bt::copy_d2d(y, 0, gamma, 0, ad.C);
    bt::copy_d2d(y, ad.C, beta, 0, ad.C);
}

// Adjoint: concat (dGamma,dBeta) → dY, run lora_backward, accumulate into the
// shared gate-grad. Base/style frozen (dX null).
void adapter_bwd(DecoderLoraAdapter& ad, const bt::Tensor& style_col,
                 const bt::Tensor& g, float scale, bt::Device dev,
                 const bt::Tensor& dGamma, const bt::Tensor& dBeta,
                 bt::Tensor& dG_accum) {
    bt::Tensor dY = bt::Tensor::zeros_on(dev, 2 * ad.C, 1);
    bt::copy_d2d(dGamma, 0, dY, 0, ad.C);
    bt::copy_d2d(dBeta,  0, dY, ad.C, ad.C);
    bt::lora_backward(ad.fc->W, style_col, ad.A, ad.B, scale, &g, ad.h, ad.hg,
                      dY, ad.dA, ad.dB, &dG_accum, /*dX=*/nullptr);
}

}  // namespace

void DecoderLora::forward(const DecoderBackbone& bb, const Generator& gen,
                          const DecoderLoraContext& ctx, const bt::Tensor& cond,
                          bt::Tensor& audio) {
    // style = ref_s[:, :style_dim] as a column; gate from the condition.
    style_col_ = bt::Tensor::zeros_on(dev_, style_dim_, 1);
    bt::copy_d2d(*ctx.ref_s, 0, style_col_, 0, style_dim_);
    gate_.forward(cond, g_);

    // backbone affines → cached backbone forward.
    std::vector<detail::BlockAffines> bbaff(bb_.size());
    for (std::size_t b = 0; b < bb_.size(); ++b) {
        adapter_fwd(bb_[b][0], style_col_, g_, scale_, dev_, bbaff[b].g1, bbaff[b].b1);
        adapter_fwd(bb_[b][1], style_col_, g_, scale_, dev_, bbaff[b].g2, bbaff[b].b2);
    }
    if (!bb_cache_) bb_cache_ = new detail::BackboneCache();
    auto& bbc = *static_cast<detail::BackboneCache*>(bb_cache_);
    bt::Tensor gen_in;
    detail::decoder_backbone_forward_train(bb, *ctx.asr, *ctx.F0_pred, *ctx.N_pred,
                                           ctx.total, bbaff, /*eps=*/1e-5f,
                                           gen_in, bbc);

    // generator affines → cached generator forward.
    auto fill_rb1 = [&](std::array<DecoderLoraAdapter, 6>& a, detail::ResBlock1Affines& out) {
        for (int j = 0; j < 3; ++j)
            adapter_fwd(a[j],     style_col_, g_, scale_, dev_, out.g1[j], out.b1[j]);
        for (int j = 0; j < 3; ++j)
            adapter_fwd(a[3 + j], style_col_, g_, scale_, dev_, out.g2[j], out.b2[j]);
    };
    std::vector<detail::ResBlock1Affines> naff(noise_.size()), raff(res_.size());
    for (std::size_t i = 0; i < noise_.size(); ++i) fill_rb1(noise_[i], naff[i]);
    for (std::size_t i = 0; i < res_.size();   ++i) fill_rb1(res_[i],   raff[i]);
    if (!gen_cache_) gen_cache_ = new detail::GeneratorCache();
    auto& genc = *static_cast<detail::GeneratorCache*>(gen_cache_);
    detail::generator_forward_train(gen, gen_in, /*L_in=*/2 * ctx.total, *ctx.har,
                                    ctx.frames, naff, raff, audio, genc);
}

void DecoderLora::backward(const DecoderBackbone& bb, const Generator& gen,
                           const bt::Tensor& cond, const bt::Tensor& dAudio) {
    auto& bbc  = *static_cast<detail::BackboneCache*>(bb_cache_);
    auto& genc = *static_cast<detail::GeneratorCache*>(gen_cache_);

    // Re-pack the forward affines (cheap; refreshes ad.h/ad.hg identically) so
    // the assembly backwards have the gamma values they read.
    std::vector<detail::BlockAffines> bbaff(bb_.size());
    for (std::size_t b = 0; b < bb_.size(); ++b) {
        adapter_fwd(bb_[b][0], style_col_, g_, scale_, dev_, bbaff[b].g1, bbaff[b].b1);
        adapter_fwd(bb_[b][1], style_col_, g_, scale_, dev_, bbaff[b].g2, bbaff[b].b2);
    }
    auto fill_rb1 = [&](std::array<DecoderLoraAdapter, 6>& a, detail::ResBlock1Affines& out) {
        for (int j = 0; j < 3; ++j)
            adapter_fwd(a[j],     style_col_, g_, scale_, dev_, out.g1[j], out.b1[j]);
        for (int j = 0; j < 3; ++j)
            adapter_fwd(a[3 + j], style_col_, g_, scale_, dev_, out.g2[j], out.b2[j]);
    };
    std::vector<detail::ResBlock1Affines> naff(noise_.size()), raff(res_.size());
    for (std::size_t i = 0; i < noise_.size(); ++i) fill_rb1(noise_[i], naff[i]);
    for (std::size_t i = 0; i < res_.size();   ++i) fill_rb1(res_[i],   raff[i]);

    // The gate is shared, so every adapter's dG accumulates into one vector.
    bt::Tensor dG = bt::Tensor::zeros_on(dev_, rank_, 1);

    // generator backward (downstream first) → affine grads + d(gen_in).
    bt::Tensor dGenIn;
    std::vector<detail::ResBlock1Affines> dnaff, draff;
    detail::generator_backward(gen, genc, naff, raff, dAudio, dGenIn, dnaff, draff);
    auto drain_rb1 = [&](std::array<DecoderLoraAdapter, 6>& a, detail::ResBlock1Affines& d) {
        for (int j = 0; j < 3; ++j)
            adapter_bwd(a[j],     style_col_, g_, scale_, dev_, d.g1[j], d.b1[j], dG);
        for (int j = 0; j < 3; ++j)
            adapter_bwd(a[3 + j], style_col_, g_, scale_, dev_, d.g2[j], d.b2[j], dG);
    };
    for (std::size_t i = 0; i < noise_.size(); ++i) drain_rb1(noise_[i], dnaff[i]);
    for (std::size_t i = 0; i < res_.size();   ++i) drain_rb1(res_[i],   draff[i]);

    // backbone backward (fed by d(gen_in)) → affine grads.
    std::vector<detail::BlockAffines> dbbaff;
    detail::decoder_backbone_backward(bb, bbc, bbaff, dGenIn, /*eps=*/1e-5f, dbbaff);
    for (std::size_t b = 0; b < bb_.size(); ++b) {
        adapter_bwd(bb_[b][0], style_col_, g_, scale_, dev_, dbbaff[b].g1, dbbaff[b].b1, dG);
        adapter_bwd(bb_[b][1], style_col_, g_, scale_, dev_, dbbaff[b].g2, dbbaff[b].b2, dG);
    }

    // gate backward with the summed bottleneck grad.
    gate_.backward(cond, dG);
}

void DecoderLora::zero_grad() {
    for (auto* ad : adapters()) {
        ad->dA = bt::Tensor::zeros_on(dev_, ad->A.rows, ad->A.cols);
        ad->dB = bt::Tensor::zeros_on(dev_, ad->B.rows, ad->B.cols);
    }
    gate_.zero_grad();
}

void DecoderLora::adam_step(float lr, float beta1, float beta2, float eps) {
    ++step_;
    for (auto* ad : adapters()) {
        bt::adam_step(ad->A, ad->dA, ad->mA, ad->vA, lr, beta1, beta2, eps, step_);
        bt::adam_step(ad->B, ad->dB, ad->mB, ad->vB, lr, beta1, beta2, eps, step_);
    }
    bt::adam_step(gate_.W1, gate_.dW1, gW1_m_, gW1_v_, lr, beta1, beta2, eps, step_);
    bt::adam_step(gate_.W2, gate_.dW2, gW2_m_, gW2_v_, lr, beta1, beta2, eps, step_);
}

// ─── Checkpoint I/O ─────────────────────────────────────────────────────────
//
// Compact little-endian binary: magic, rank/style_dim/scale/gate-hidden, the
// gate weights, then each adapter's A and B (canonical adapters() order). The
// architecture (adapter count + shapes) is reconstructed by make() before
// load(); load only restores parameter values.

namespace {
void write_tensor(std::FILE* f, const bt::Tensor& t) {
    const std::int32_t r = t.rows, c = t.cols;
    std::fwrite(&r, sizeof(r), 1, f);
    std::fwrite(&c, sizeof(c), 1, f);
    std::vector<float> h = t.to_host_vector();
    std::fwrite(h.data(), sizeof(float), h.size(), f);
}
bt::Tensor read_tensor(std::FILE* f, bt::Device dev) {
    std::int32_t r = 0, c = 0;
    if (std::fread(&r, sizeof(r), 1, f) != 1 || std::fread(&c, sizeof(c), 1, f) != 1)
        throw std::runtime_error("DecoderLora::load: truncated tensor header");
    std::vector<float> h(static_cast<std::size_t>(r) * c);
    if (std::fread(h.data(), sizeof(float), h.size(), f) != h.size())
        throw std::runtime_error("DecoderLora::load: truncated tensor data");
    return bt::Tensor::from_host_on(dev, h.data(), r, c);
}
constexpr char kMagic[8] = {'E', 'L', 'R', 'A', '1', 0, 0, 0};
}  // namespace

void DecoderLora::save(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("DecoderLora::save: cannot open " + path);
    std::fwrite(kMagic, 1, 8, f);
    const std::int32_t rank = rank_, sd = style_dim_, gh = gate_.hidden;
    std::fwrite(&rank, sizeof(rank), 1, f);
    std::fwrite(&sd, sizeof(sd), 1, f);
    std::fwrite(&scale_, sizeof(scale_), 1, f);
    std::fwrite(&gh, sizeof(gh), 1, f);
    write_tensor(f, gate_.W1);
    write_tensor(f, gate_.W2);
    // const adapters() — walk the structure directly (adapters() is non-const).
    auto write_ad = [&](const DecoderLoraAdapter& ad) {
        const std::int32_t C = ad.C;
        std::fwrite(&C, sizeof(C), 1, f);
        write_tensor(f, ad.A);
        write_tensor(f, ad.B);
    };
    for (const auto& pr : bb_) { write_ad(pr[0]); write_ad(pr[1]); }
    for (const auto& a : noise_) for (const auto& ad : a) write_ad(ad);
    for (const auto& a : res_)   for (const auto& ad : a) write_ad(ad);
    std::fclose(f);
}

void DecoderLora::load(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("DecoderLora::load: cannot open " + path);
    char magic[8];
    if (std::fread(magic, 1, 8, f) != 8 ||
        std::memcmp(magic, kMagic, 8) != 0) {
        std::fclose(f);
        throw std::runtime_error("DecoderLora::load: bad magic in " + path);
    }
    std::int32_t rank = 0, sd = 0, gh = 0; float scale = 0.0f;
    std::fread(&rank, sizeof(rank), 1, f);
    std::fread(&sd, sizeof(sd), 1, f);
    std::fread(&scale, sizeof(scale), 1, f);
    std::fread(&gh, sizeof(gh), 1, f);
    if (rank != rank_ || sd != style_dim_)
        throw std::runtime_error("DecoderLora::load: architecture mismatch");
    scale_ = scale;
    gate_.W1 = read_tensor(f, dev_);
    gate_.W2 = read_tensor(f, dev_);
    auto read_ad = [&](DecoderLoraAdapter& ad) {
        std::int32_t C = 0;
        std::fread(&C, sizeof(C), 1, f);
        ad.A = read_tensor(f, dev_);
        ad.B = read_tensor(f, dev_);
    };
    for (auto& pr : bb_) { read_ad(pr[0]); read_ad(pr[1]); }
    for (auto& a : noise_) for (auto& ad : a) read_ad(ad);
    for (auto& a : res_)   for (auto& ad : a) read_ad(ad);
    std::fclose(f);
}

}  // namespace brosoundml
