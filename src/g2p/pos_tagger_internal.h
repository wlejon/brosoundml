#pragma once

// Internal POS tagger types — shared between pos_tagger.cpp (forward pass) and
// pos_tagger_load.cpp (weight loader). Not exposed via the public API.

#include "brosoundml/g2p/tags.h"
#include "brosoundml/modules.h"

#include <brotensor/tensor.h>

#include <string>
#include <vector>

namespace brosoundml::g2p {

// Locked arch constants — match docs/pos_tagger.md.
inline constexpr int kVocab     = 260;
inline constexpr int kMaxSeqLen = 384;
inline constexpr int kDModel    = 192;
inline constexpr int kNumLayers = 4;
inline constexpr int kNumHeads  = 4;
inline constexpr int kFFN       = 768;

inline constexpr std::int32_t kPad  = 0;
inline constexpr std::int32_t kBos  = 1;
inline constexpr std::int32_t kEos  = 2;
inline constexpr std::int32_t kWsep = 3;

struct LayerWeights {
    LayerNorm ln1;
    MHA       mha;
    LayerNorm ln2;
    Linear    ffn1;
    Linear    ffn2;
};

struct PosWeights {
    int num_tags    = NUM_TAGS;
    int d_model     = kDModel;
    int num_layers  = kNumLayers;
    int num_heads   = kNumHeads;
    int ffn_hidden  = kFFN;
    int max_seq_len = kMaxSeqLen;

    brotensor::Tensor         token_emb;
    brotensor::Tensor         pos_emb;
    std::vector<LayerWeights> layers;
    LayerNorm                 final_ln;
    Linear                    head;
};

void load_pos_weights(const std::string& path, PosWeights& out);

// Tokeniser types — exposed for unit tests via tokenise_for_test().
struct WordSpan {
    std::size_t byte_start;
    std::size_t byte_len;
};

struct TokenChunk {
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> wsep_positions;
    std::vector<WordSpan>     word_spans;
};

std::vector<TokenChunk> tokenise_for_test(std::string_view s);

// exposed for trainer
std::vector<TokenChunk> tokenise_sentence(std::string_view s);

}  // namespace brosoundml::g2p
