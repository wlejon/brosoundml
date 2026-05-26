#include "brosoundml/g2p/pos_tagger.h"

#include "pos_tagger_internal.h"
#include "brosoundml/modules.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brosoundml::g2p {

namespace bt = brotensor;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

// Word splitting: runs of ASCII/unicode whitespace collapse to a word boundary.
// For UTF-8 non-ASCII whitespace this is conservative — matches the typical
// English inputs the spec scopes to.
std::vector<WordSpan> split_words(std::string_view sentence) {
    std::vector<WordSpan> spans;
    std::size_t i = 0;
    const std::size_t n = sentence.size();
    while (i < n) {
        while (i < n && std::isspace(static_cast<unsigned char>(sentence[i]))) ++i;
        if (i >= n) break;
        const std::size_t start = i;
        while (i < n && !std::isspace(static_cast<unsigned char>(sentence[i]))) ++i;
        spans.push_back({start, i - start});
    }
    return spans;
}

// Tokenise the sentence into one or more chunks, each ≤ kMaxSeqLen tokens.
// A chunk ends at a word boundary — no word's bytes are split across chunks.
std::vector<TokenChunk> tokenise(std::string_view sentence) {
    const auto words = split_words(sentence);
    std::vector<TokenChunk> chunks;

    std::size_t w = 0;
    while (w < words.size()) {
        TokenChunk c;
        c.token_ids.push_back(kBos);
        for (; w < words.size(); ++w) {
            const auto& span = words[w];
            // Each word adds 1 (wsep) + span.byte_len tokens; the eos at the
            // end needs 1 more slot.
            const std::size_t added = 1 + span.byte_len;
            if (static_cast<int>(c.token_ids.size() + added + 1) > kMaxSeqLen) {
                if (c.wsep_positions.empty()) {
                    fail("PosTagger::tag",
                         "single word exceeds max sequence length " +
                         std::to_string(kMaxSeqLen));
                }
                break;
            }
            c.wsep_positions.push_back(static_cast<std::int32_t>(c.token_ids.size()));
            c.token_ids.push_back(kWsep);
            for (std::size_t b = 0; b < span.byte_len; ++b) {
                const unsigned char byte = static_cast<unsigned char>(
                    sentence[span.byte_start + b]);
                c.token_ids.push_back(static_cast<std::int32_t>(byte) + 4);
            }
            c.word_spans.push_back(span);
        }
        c.token_ids.push_back(kEos);
        chunks.push_back(std::move(c));
    }
    return chunks;
}

}  // namespace

std::vector<TokenChunk> tokenise_for_test(std::string_view s) {
    return tokenise(s);
}

std::vector<TokenChunk> tokenise_sentence(std::string_view s) {
    return tokenise(s);
}

// ─── Encoder forward ───────────────────────────────────────────────────────

struct PosTagger::Impl {
    PosWeights w;

    void forward_chunk(const TokenChunk& chunk, bt::Tensor& word_logits) const;
};

void PosTagger::Impl::forward_chunk(const TokenChunk& chunk,
                                    bt::Tensor& word_logits) const {
    const int L = static_cast<int>(chunk.token_ids.size());
    if (L > kMaxSeqLen) {
        fail("PosTagger::forward", "chunk len " + std::to_string(L) +
             " > max_seq_len " + std::to_string(kMaxSeqLen));
    }
    const int D = w.d_model;

    bt::Tensor h;
    bt::embedding_lookup_forward(w.token_emb, chunk.token_ids.data(), L, h);

    std::vector<std::int32_t> pos_idx(static_cast<std::size_t>(L));
    for (int i = 0; i < L; ++i) pos_idx[i] = i;
    bt::Tensor pe;
    bt::embedding_lookup_forward(w.pos_emb, pos_idx.data(), L, pe);
    bt::add_inplace_batched(h, pe);

    bt::Tensor x_norm   = bt::Tensor::empty_on(h.device, L, D, h.dtype);
    bt::Tensor attn_out = bt::Tensor::empty_on(h.device, L, D, h.dtype);
    bt::Tensor ffn_h    = bt::Tensor::empty_on(h.device, L, w.ffn_hidden, h.dtype);
    bt::Tensor ffn_h_act= bt::Tensor::empty_on(h.device, L, w.ffn_hidden, h.dtype);
    bt::Tensor ffn_out  = bt::Tensor::empty_on(h.device, L, D, h.dtype);

    for (const auto& layer : w.layers) {
        layer.ln1.forward(h, x_norm);
        layer.mha.forward(x_norm, /*d_mask=*/nullptr, attn_out);
        bt::add_inplace_batched(h, attn_out);

        layer.ln2.forward(h, x_norm);
        layer.ffn1.forward_batched(x_norm, ffn_h);
        bt::gelu_forward(ffn_h, ffn_h_act);
        layer.ffn2.forward_batched(ffn_h_act, ffn_out);
        bt::add_inplace_batched(h, ffn_out);
    }

    bt::Tensor h_final = bt::Tensor::empty_on(h.device, L, D, h.dtype);
    w.final_ln.forward(h, h_final);

    const int W = static_cast<int>(chunk.wsep_positions.size());
    if (W == 0) {
        word_logits = bt::Tensor::empty_on(h.device, 0, w.num_tags, h.dtype);
        return;
    }
    bt::Tensor pooled = bt::Tensor::empty_on(h.device, W, D, h.dtype);
    for (int i = 0; i < W; ++i) {
        const int t = chunk.wsep_positions[i];
        bt::copy_d2d(h_final, t * D, pooled, i * D, D);
    }
    word_logits = bt::Tensor::empty_on(h.device, W, w.num_tags, h.dtype);
    w.head.forward_batched(pooled, word_logits);
}

// ─── Public API ────────────────────────────────────────────────────────────

PosTagger::PosTagger() : impl_(std::make_unique<Impl>()) {}
PosTagger::PosTagger(PosTagger&&) noexcept = default;
PosTagger& PosTagger::operator=(PosTagger&&) noexcept = default;
PosTagger::~PosTagger() = default;

PosTagger PosTagger::load(const std::string& weights_path) {
    PosTagger pt;
    load_pos_weights(weights_path, pt.impl_->w);
    return pt;
}

std::vector<WordTag> PosTagger::tag(std::string_view sentence) const {
    const auto chunks = tokenise(sentence);
    std::vector<WordTag> out;

    for (const auto& c : chunks) {
        if (c.wsep_positions.empty()) continue;

        bt::Tensor logits;
        impl_->forward_chunk(c, logits);

        bt::Tensor idx_dev;
        bt::argmax_rows(logits, idx_dev);
        const auto idx_host = idx_dev.to_host_vector();

        for (std::size_t i = 0; i < c.word_spans.size(); ++i) {
            const int t = static_cast<int>(idx_host[i]);
            if (t < 0 || t >= impl_->w.num_tags) {
                fail("PosTagger::tag",
                     "argmax produced out-of-range tag id " + std::to_string(t));
            }
            WordTag wt;
            wt.word = sentence.substr(c.word_spans[i].byte_start,
                                      c.word_spans[i].byte_len);
            wt.tag  = static_cast<PosTag>(t);
            out.push_back(wt);
        }
    }
    return out;
}

}  // namespace brosoundml::g2p
