#pragma once

// HiggsAudio v2 tokenizer — the module graph behind the public HiggsCodec
// (include/brosoundml/higgs_codec.h). Internal to brosoundml: HiggsCodec::Impl
// owns one HiggsCodecModel; tests/test_higgs_codec.cpp reaches in for the
// per-stage encoder trace so a fixture mismatch localises.
//
// Layouts: NCL activations are (1, C*L) channel-major; SEQ activations are
// (L, C). Weights are FP32 on the load device; every forward step dispatches
// through brotensor device ops.

#include "brosoundml/higgs_codec.h"
#include "higgs_codec_common.h"
#include "hubert.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <string>
#include <vector>

namespace brosoundml {

// One DAC residual unit: Snake -> conv k7 (dilation d, "same") -> Snake ->
// conv k1, residual add (the centred crop is a no-op under "same" padding).
struct HiggsResUnit {
    brotensor::Tensor a1, a2;      // (C,1) plain Snake alphas
    brotensor::Tensor c1w, c1b;    // (C, C*7), (C,1)
    brotensor::Tensor c2w, c2b;    // (C, C*1), (C,1)
    int dim = 0, dilation = 1;
};

// DAC decoder block: Snake -> ConvTranspose1d (in -> out, x stride) -> three
// residual units (dilation 1, 3, 9).
struct HiggsDecBlock {
    brotensor::Tensor snake;       // (in,1)
    brotensor::Tensor tw, tb;      // (in, out*2*stride), (out,1)
    int in = 0, out = 0, stride = 0;
    HiggsResUnit units[3];
};

// DAC encoder block: three residual units (dilation 1, 3, 9) -> Snake ->
// strided conv (in -> out, kernel 2*stride, padding ceil(stride/2)).
struct HiggsEncBlock {
    HiggsResUnit units[3];
    brotensor::Tensor snake;       // (in,1)
    brotensor::Tensor cw, cb;      // (out, in*2*stride), (out,1)
    int in = 0, out = 0, stride = 0;
};

// SemanticEncoder residual unit: ELU -> conv k3 (dilation d, no bias) -> ELU
// -> conv k1 (no bias), residual add.
struct HiggsSemUnit {
    brotensor::Tensor c1w, c2w;    // (768, 768*3), (768, 768*1)
    int dilation = 1;
};

// SemanticEncoder block (stride 1): two residual units -> conv k3 (+bias).
struct HiggsSemBlock {
    std::vector<HiggsSemUnit> units;
    brotensor::Tensor cw, cb;      // (768, 768*3), (768,1)
};

// Per-stage encoder intermediates, for white-box validation. Every buffer is
// host FP32 in the layout the fixture uses (see tests/ref/gen_higgs_codec_fixture.py).
struct HiggsEncodeTrace {
    int n24 = 0, n16 = 0, Th = 0, T = 0;
    std::vector<float> sem16;      // [n16]      the 16 kHz semantic input
    std::vector<float> hubert;     // [Th*768]   mean of the 13 HuBERT hidden states, frame-major
    std::vector<float> sem_enc;    // [768*T]    SemanticEncoder output, channel-major
    std::vector<float> acoustic;   // [256*T]    DAC encoder latent, channel-major
    std::vector<float> fc_out;     // [1024*T]   fc output (the RVQ input), channel-major
};

struct HiggsCodecModel {
    HiggsCodecConfig cfg;
    brotensor::Device dev = brotensor::Device::CPU;
    bool is_loaded = false;
    int sem_kernel = 3, sem_unit_kernel = 3;
    std::vector<int> sem_dilations = {1, 1};
    int num_sem_blocks = 2;

    // ── residual VQ (both directions) ──
    std::vector<brotensor::Tensor> embed;    // per level (codebook_size, codebook_dim)
    std::vector<brotensor::Tensor> pin_w, pin_b;    // project_in  (64,1024) + (64,1)
    std::vector<brotensor::Tensor> pout_w, pout_b;  // project_out (1024,64) + (1024,1)

    // ── decoder ──
    brotensor::Tensor fc2_w, fc2_b;          // (256,1024), (256,1)
    brotensor::Tensor dconv1_w, dconv1_b;    // (1024, 256*7), (1024,1)
    std::vector<HiggsDecBlock> dec_blocks;
    brotensor::Tensor dsnake;                // (32,1)
    brotensor::Tensor dconv2_w, dconv2_b;    // (1, 32*7), (1,1)

    // ── encoder ──
    brotensor::Tensor econv1_w, econv1_b;    // (64, 1*7), (64,1)
    std::vector<HiggsEncBlock> enc_blocks;
    brotensor::Tensor esnake;                // (2048,1)
    brotensor::Tensor econv2_w, econv2_b;    // (256, 2048*3), (256,1)
    brotensor::Tensor sconv_w;               // SemanticEncoder.conv (768, 768*3), no bias
    std::vector<HiggsSemBlock> sem_blocks;
    brotensor::Tensor fc_w, fc_b;            // (1024,1024), (1024,1)
    HubertModel hubert;
    hcodec::SincResampler sem_resampler;     // sample_rate -> semantic_sample_rate

    // config.json + model.safetensors from `dir`, weights on `device`.
    void load(const std::string& dir, brotensor::Device device, bool decoder_only);

    // codes[q*T + t] (K <= cfg.num_quantizers levels) -> T*hop_length samples.
    std::vector<float> decode(const std::int32_t* codes, int K, int T) const;

    // wav24: n samples of 24 kHz mono, n a multiple of hop_length (the caller
    // pads). Returns num_quantizers*T codes laid out codes[q*T + t]; *T_out
    // receives T. `trace` (optional) receives the stage intermediates. When
    // `sem16_override` is non-null it replaces the internal 24 -> 16 kHz
    // windowed-sinc resample as the semantic input (n16_override samples) —
    // the test uses it to isolate HuBERT + the RVQ from the resampler.
    std::vector<std::int32_t> encode(const float* wav24, int n, int* T_out,
                                     HiggsEncodeTrace* trace = nullptr,
                                     const float* sem16_override = nullptr,
                                     int n16_override = 0) const;
};

}  // namespace brosoundml
