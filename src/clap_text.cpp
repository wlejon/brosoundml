// CLAP text tower: RoBERTa-base (transformers' ClapTextModel) + the text
// projection — see include/brosoundml/clap.h and docs/clap.md.
//
// Post-LN BERT layers over one unpadded sequence (every prompt is encoded on
// its own, so no attention mask is needed): embeddings = word + position +
// token_type[0], LayerNorm; per layer h = LN(h + attn(h)), h = LN(h + fc2(gelu(
// fc1(h)))); the pooler is tanh(dense(h[<s>])). Position ids follow RoBERTa's
// padding_idx rule, so an unpadded sequence of L tokens takes positions
// pad_id + 1 .. pad_id + L. The byte-level BPE is brolm's tokenizer.json
// driven loader (brolm::laya::LayaTokenizer — generic ByteLevel BPE, the
// same tokenizers semantics), with <s> ... </s> added here as the
// RobertaProcessing post-processor does.
#include "clap_internal.h"

#include "qwen_tts_device.h"   // qtd::linear, qtd::gather_rows

#include <brolm/laya_tokenizer.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace brosoundml {
namespace clap {

namespace {

Linear load_linear(const sf::File& f, const std::string& p, int out, int in, bt::Device dev) {
    return Linear{upload(f, p + ".weight", out, in, dev), upload_vec(f, p + ".bias", out, dev)};
}

Norm load_norm(const sf::File& f, const std::string& p, int d, bt::Device dev) {
    return Norm{upload_vec(f, p + ".weight", d, dev), upload_vec(f, p + ".bias", d, dev)};
}

bt::Tensor linear(const Linear& l, const bt::Tensor& x) {
    bt::Tensor y;
    qtd::linear(l.w, &l.b, x, y);
    return y;
}

bt::Tensor layer_norm(const Norm& n, const bt::Tensor& x, float eps) {
    bt::Tensor y;
    bt::layernorm_forward_inference_batched(x, n.w, n.b, y, eps);
    return y;
}

}  // namespace

void TextTower::load(const sf::File& f, const std::string& dir, const ClapConfig& cfg, bt::Device dev) {
    const std::string T = "text_model.";
    const int D = cfg.text_hidden, I = cfg.text_intermediate;
    word_emb = upload(f, T + "embeddings.word_embeddings.weight", cfg.vocab_size, D, dev);
    pos_emb = upload(f, T + "embeddings.position_embeddings.weight", cfg.max_position, D, dev);
    {
        const std::vector<float> tt = host_f32(f, T + "embeddings.token_type_embeddings.weight");
        if (static_cast<int>(tt.size()) < D) fail("token_type_embeddings too small");
        type_row = bt::Tensor::from_host_on(dev, tt.data(), 1, D);
    }
    emb_norm = load_norm(f, T + "embeddings.LayerNorm", D, dev);
    layers.clear();
    layers.resize(cfg.text_layers);
    for (int i = 0; i < cfg.text_layers; ++i) {
        const std::string p = T + "encoder.layer." + std::to_string(i) + ".";
        TextLayer& l = layers[i];
        l.q = load_linear(f, p + "attention.self.query", D, D, dev);
        l.k = load_linear(f, p + "attention.self.key", D, D, dev);
        l.v = load_linear(f, p + "attention.self.value", D, D, dev);
        l.o = load_linear(f, p + "attention.output.dense", D, D, dev);
        l.ln_attn = load_norm(f, p + "attention.output.LayerNorm", D, dev);
        l.fc1 = load_linear(f, p + "intermediate.dense", I, D, dev);
        l.fc2 = load_linear(f, p + "output.dense", D, I, dev);
        l.ln_out = load_norm(f, p + "output.LayerNorm", D, dev);
    }
    pooler = load_linear(f, T + "pooler.dense", D, D, dev);
    proj1 = load_linear(f, "text_projection.linear1", cfg.projection_dim, D, dev);
    proj2 = load_linear(f, "text_projection.linear2", cfg.projection_dim, cfg.projection_dim, dev);

    const std::string tok_path = (std::filesystem::path(dir) / "tokenizer.json").string();
    if (!std::filesystem::exists(tok_path)) fail("missing " + tok_path);
    tokenizer = std::make_shared<brolm::laya::LayaTokenizer>(brolm::laya::LayaTokenizer::load(tok_path));
    if (tokenizer->kind() != brolm::laya::LayaTokenizer::Kind::ByteLevel)
        fail("tokenizer.json is not a byte-level BPE");
    bos_id = tokenizer->cls_token_id();
    eos_id = tokenizer->sep_token_id();
}

std::vector<std::int32_t> TextTower::tokenize(const std::string& text, const ClapConfig& cfg) const {
    if (!tokenizer) fail("tokenizer not loaded");
    std::vector<std::int32_t> body = tokenizer->encode(text);
    // Positions run pad_id+1 .. pad_id+L, so L <= max_position - pad_id - 1.
    const std::size_t cap = static_cast<std::size_t>(cfg.max_position - cfg.pad_token_id - 1) - 2;
    if (body.size() > cap) body.resize(cap);
    std::vector<std::int32_t> ids;
    ids.reserve(body.size() + 2);
    ids.push_back(bos_id);
    ids.insert(ids.end(), body.begin(), body.end());
    ids.push_back(eos_id);
    return ids;
}

std::vector<float> TextTower::forward(const std::vector<std::int32_t>& ids, const ClapConfig& cfg,
                                      bt::Device dev) const {
    bt::DeviceScope scope(dev);
    const int L = static_cast<int>(ids.size());
    const int D = cfg.text_hidden;
    if (L < 1) fail("empty token sequence");
    if (L > cfg.max_position - cfg.pad_token_id - 1) fail("token sequence longer than the position table");
    for (std::int32_t id : ids)
        if (id < 0 || id >= cfg.vocab_size) fail("token id out of range");
    const float eps = cfg.text_ln_eps;

    std::vector<std::int32_t> pos(L);
    for (int i = 0; i < L; ++i) pos[i] = cfg.pad_token_id + 1 + i;
    bt::Tensor h = qtd::gather_rows(word_emb, ids);   // (L, D)
    bt::add_inplace(h, qtd::gather_rows(pos_emb, pos));
    bt::add_row_bias_inplace(h, type_row);
    h = layer_norm(emb_norm, h, eps);

    const float scale = 1.0f / std::sqrt(static_cast<float>(D / cfg.text_heads));
    for (const TextLayer& l : layers) {
        bt::Tensor a;
        bt::self_attention_bias_forward(h, l.q.w, l.k.w, l.v.w, l.o.w, &l.q.b, &l.k.b, &l.v.b, &l.o.b,
                                        /*d_mask=*/nullptr, /*attn_bias=*/nullptr, cfg.text_heads, scale, a);
        bt::add_inplace(a, h);
        h = layer_norm(l.ln_attn, a, eps);
        bt::Tensor f = linear(l.fc1, h);
        bt::gelu_exact_forward(f, f);
        bt::Tensor o = linear(l.fc2, f);
        bt::add_inplace(o, h);
        h = layer_norm(l.ln_out, o, eps);
    }

    bt::Tensor cls = bt::Tensor::empty_on(dev, 1, D, h.dtype);
    bt::copy_d2d(h, 0, cls, 0, D);
    bt::Tensor pooled = linear(pooler, cls);
    bt::tanh_forward(pooled, pooled);
    bt::Tensor p1 = linear(proj1, pooled);
    bt::relu_forward(p1, p1);
    bt::Tensor p2 = linear(proj2, p1);
    bt::sync(dev);
    return p2.to(bt::Device::CPU).to_host_vector();
}

}  // namespace clap
}  // namespace brosoundml
