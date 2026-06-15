#pragma once

// brosoundml/src/supertonic_internal.h — the anonymous-namespace weight structs
// that supertonic.cpp builds its forward graph from, surfaced into a named
// namespace so a separate backward translation unit
// (supertonic_backward.{h,cpp}) can compose against the exact same types.
//
// These are pure type definitions (no free functions with internal linkage), so
// living in a shared header changes nothing about linkage: supertonic.cpp pulls
// them in with `using namespace st_detail;` and behaves identically; the
// backward TU reimplements the forward math it needs (composing brotensor ops
// directly, mirroring kokoro_decoder_backward.h) against these same structs.
//
// Private to the brosoundml build (lives under src/, not installed).

#include <brotensor/tensor.h>

#include <vector>

namespace brosoundml {
namespace st_detail {

// A conv1d weight flattened to brotensor's (Cout, (Cin/groups)*K) OIL layout.
struct ConvW {
    brotensor::Tensor w, b;
    bool              has_b = false;
    int               cin_pg = 0, cout = 0, k = 0;  // cin_pg = weight.shape[1] = Cin/groups
};

struct ConvNeXtBlock {
    ConvW             dw;          // depthwise conv (groups = channels), kernel 7, dilated
    int               dil = 1;
    brotensor::Tensor ln_g, ln_b;  // channel-wise LayerNorm affine
    ConvW             pw1;         // 1x1, channels -> intermediate
    ConvW             pw2;         // 1x1, intermediate -> channels (gamma folded)
};

// One Glow-TTS relative-position attention block (text encoder attn_encoder):
// 1x1 q/k/v/o projections, windowed relative key/value embeddings, two
// channel-wise LayerNorms, and a 1x1-conv FFN (conv_1 -> relu -> conv_2).
struct AttnLayer {
    ConvW             conv_q, conv_k, conv_v, conv_o;  // 1x1
    brotensor::Tensor emb_rel_k, emb_rel_v;            // [2*window+1, kc]
    brotensor::Tensor n1_g, n1_b, n2_g, n2_b;          // post-attn / post-ffn LN
    ConvW             ffn1, ffn2;                       // 1x1 (C->filter, filter->C)
};

// One speech-prompted style cross-attention (tanh-gated, query=text,
// key=learned style prototype, value=style_ttl). Folded linears carried as
// 1x1 convs over channel-major activations.
struct StyleAttn {
    ConvW wq, wk, wv, wo;  // 1x1, from the transposed onnx::MatMul_* weights
};

// A plain Linear (ONNX Gemm, transB=1): y = x @ W^T + b. The weight is stored
// transposed to [in, out] so a row-vector matmul needs no further transpose.
struct Gemm {
    brotensor::Tensor wt;  // [in, out]
    brotensor::Tensor b;   // [1, out]
    int in = 0, out = 0;
};

// ─── flow-field (vector estimator) block types ───────────────────────────────

struct VeFilm {                // time FiLM (main_blocks %6==1): h += time_emb @ W + b
    brotensor::Tensor w;       // [64, 512] (in,out) for the row matmul
    brotensor::Tensor b;       // [1, 512]
};

struct VeRope {                // text cross-attention with RoPE (main_blocks %6==3)
    ConvW conv_q, conv_k, conv_v, conv_o;  // q:512->512, k/v:256->512, o:512->512
    std::vector<float> theta;              // [32] RoPE base frequencies (host)
    brotensor::Tensor norm_g, norm_b;      // post-attention LayerNorm
};

struct VeStyle {               // tanh-gated style cross-attention (main_blocks %6==5)
    StyleAttn         attn;    // q:512->256, k/v:256->256, o:256->512
    brotensor::Tensor norm_g, norm_b;      // post-attention LayerNorm
};

struct VeBlock {
    int type = 0;                          // 0 convnext(x2), 1 film, 2 rope, 3 style
    std::vector<ConvNeXtBlock> conv;       // type 0: two ConvNeXt sub-blocks
    VeFilm  film;
    VeRope  rope;
    VeStyle style;
};

}  // namespace st_detail
}  // namespace brosoundml
