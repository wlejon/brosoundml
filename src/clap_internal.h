#pragma once

// CLAP internals shared by src/clap.cpp (load, front-end, scoring),
// src/clap_audio.cpp (the HTSAT Swin tower) and src/clap_text.cpp (the
// RoBERTa tower + tokenizer). Not part of the public include/ surface.

#include <brosoundml/clap.h>

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::laya { class LayaTokenizer; }

namespace brosoundml {
namespace clap {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;

[[noreturn]] inline void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: clap: " + msg);
}

// Upload an F32 checkpoint tensor as (rows, cols) on `dev`; the element count
// must match (sf::upload checks the bytes).
bt::Tensor upload(const sf::File& f, const std::string& name, int rows, int cols, bt::Device dev);
inline bt::Tensor upload_vec(const sf::File& f, const std::string& name, int n, bt::Device dev) {
    return upload(f, name, n, 1, dev);
}
// A host copy of an F32 checkpoint tensor (any shape).
std::vector<float> host_f32(const sf::File& f, const std::string& name);

struct Linear {
    bt::Tensor w, b;   // (out, in), (out, 1) — b empty when bias-less
};
struct Norm {
    bt::Tensor w, b;   // (D, 1)
};

// ── audio tower ─────────────────────────────────────────────────────────────

struct SwinBlock {
    Norm ln_before, ln_after;
    Linear q, k, v, o, fc1, fc2;
    int dim = 0, heads = 0, window = 0, shift = 0;
    // Pre-softmax additive bias (heads*L, L), L = window^2, per window kind:
    // relative-position bias, plus the shifted-window mask for the windows
    // that straddle the roll seam. `bias_kind[w]` indexes `bias` per window.
    std::vector<bt::Tensor> bias;
    std::vector<int> bias_kind;
    // Token-row permutations (grid raster <-> window-major), on the device:
    // gather(x, to_windows) rolls by -shift and partitions; gather(y,
    // from_windows) reverses both.
    bt::Tensor to_windows, from_windows;
};

struct SwinStage {
    int H = 0, W = 0, dim = 0;
    std::vector<SwinBlock> blocks;
    bool merge = false;
    Norm merge_norm;       // (4*dim)
    Linear merge_red;      // (2*dim, 4*dim), no bias
    bt::Tensor merge_idx;  // (H*W, 1) INT32: 2x2 gather into (H/2*W/2, 4*dim)
};

struct AudioTower {
    bt::Tensor bn_w, bn_b, bn_mean, bn_var;   // eval BatchNorm over mel bins (C, 1)
    bt::Tensor patch_w, patch_b;    // (E0, 1*4*4), (E0, 1)
    Norm patch_norm;
    std::vector<SwinStage> stages;
    Norm final_norm;
    Linear proj1, proj2;
    bt::Tensor pool_row;            // (1, tokens) of 1/tokens: the mean pool

    void load(const sf::File& f, const ClapConfig& cfg, bt::Device dev);
    // mel: host (frames x mels). Returns the 1024-d pooled output and the
    // 512-d (un-normalised) projection, both on the host.
    void forward(const std::vector<float>& mel, const ClapConfig& cfg, bt::Device dev,
                 std::vector<float>& pooled, std::vector<float>& projected) const;
};

// ── text tower ──────────────────────────────────────────────────────────────

struct TextLayer {
    Linear q, k, v, o, fc1, fc2;
    Norm ln_attn, ln_out;
};

struct TextTower {
    bt::Tensor word_emb, pos_emb;   // (V, D), (P, D)
    bt::Tensor type_row;            // (1, D): token_type_embeddings[0]
    Norm emb_norm;
    std::vector<TextLayer> layers;
    Linear pooler, proj1, proj2;
    std::shared_ptr<brolm::laya::LayaTokenizer> tokenizer;
    std::int32_t bos_id = 0, eos_id = 2;

    void load(const sf::File& f, const std::string& dir, const ClapConfig& cfg, bt::Device dev);
    std::vector<std::int32_t> tokenize(const std::string& text, const ClapConfig& cfg) const;
    // ids -> the 512-d (un-normalised) projection on the host.
    std::vector<float> forward(const std::vector<std::int32_t>& ids, const ClapConfig& cfg,
                               bt::Device dev) const;
};

// L2-normalise in place (double accumulation); a zero vector stays zero.
void l2_normalize(std::vector<float>& v);

}  // namespace clap

struct Clap::Impl {
    ClapConfig cfg;
    brotensor::Device device = brotensor::Device::CPU;
    clap::AudioTower audio;
    clap::TextTower text;
    float logit_scale_a = 1.0f, logit_scale_t = 1.0f;   // exp'd
    bool loaded = false;
};

}  // namespace brosoundml
