#pragma once

// Parakeet-specific submodules (internal): the TDT prediction network and joint
// network. The FastConformer encoder is shared (see fastconformer_modules.h) —
// Parakeet's encoder is just a FastConformerEncoder. These structs know the HF
// Parakeet state-dict key naming for the decoder side. FP32 on whichever device
// the weights live on; every forward dispatches through brotensor device ops.
//
// HF key scheme (transformers ParakeetForTDT), decoder side:
//   encoder_projector.{weight,bias}
//   decoder.embedding.weight
//   decoder.lstm.{weight_ih,weight_hh,bias_ih,bias_hh}_l{0,1}
//   decoder.decoder_projector.{weight,bias}
//   joint.head.{weight,bias}

#include "brosoundml/audio.h"
#include "brosoundml/parakeet.h"
#include "fastconformer_modules.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <string>
#include <vector>

namespace brosoundml {

// Parakeet's acoustic encoder is the shared FastConformer encoder.
using ParakeetEncoder = FastConformerEncoder;

// ─── TDT prediction network (RNN-T decoder) ────────────────────────────────
//
// Token embedding -> num_decoder_layers LSTM -> decoder_projector. Stateful:
// the greedy decoder steps it one emitted token at a time, carrying (h,c) per
// layer. PyTorch nn.LSTM gate order i,f,g,o.
struct ParakeetPrediction {
    int hidden    = 0;
    int n_layers  = 0;

    brotensor::Tensor embedding;                 // (vocab, hidden)
    std::vector<brotensor::Tensor> w_ih, w_hh;   // (4*hidden, in/hidden) per layer
    std::vector<brotensor::Tensor> b_ih, b_hh;   // (4*hidden, 1) per layer
    brotensor::Tensor proj_w, proj_b;            // decoder_projector (hidden,hidden)

    brotensor::Device device = brotensor::Device::CPU;

    void load(const brotensor::safetensors::File& f,
              const ParakeetConfig& cfg, brotensor::Device device);

    // Per-layer hidden / cell state, each (1, hidden) on `device`.
    struct State {
        std::vector<brotensor::Tensor> h, c;
    };
    State init_state() const;

    // Advance one step on `token_id`: embed, run the LSTM stack, project.
    // Writes dec_proj (1, hidden) to `out`; mutates `st` in place.
    void step(int32_t token_id, State& st, brotensor::Tensor& out) const;
};

// ─── TDT joint network ─────────────────────────────────────────────────────
//
// logits = head( relu(enc_proj[t] + dec_proj) ). Split: first vocab_size are
// token logits, the remaining len(durations) are duration logits.
struct ParakeetJoint {
    brotensor::Tensor head_w, head_b;   // (vocab + n_durations, hidden)
    brotensor::Device device = brotensor::Device::CPU;

    void load(const brotensor::safetensors::File& f,
              const ParakeetConfig& cfg, brotensor::Device device);

    // enc_proj_row, dec_proj: each (1, hidden). Writes logits (1, V+nd) to out.
    void forward(const brotensor::Tensor& enc_proj_row,
                 const brotensor::Tensor& dec_proj,
                 brotensor::Tensor& out) const;
};

}  // namespace brosoundml
