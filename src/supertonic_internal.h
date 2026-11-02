#pragma once

// brosoundml/src/supertonic_internal.h — the weight structs, architecture
// constants, and Supertonic::Impl definition shared across supertonic.cpp and
// supertonic_weights.cpp.
//
// Private to the brosoundml build (lives under src/, not installed).

#include "brosoundml/supertonic.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace brosoundml {
namespace st_detail {

// Text-encoder architecture constants (fixed for supertonic-3, from tts.json).
inline constexpr int   kTeC       = 256;   // hidden channels
inline constexpr int   kTeHeads   = 4;     // attn heads
inline constexpr int   kTeKc      = 64;    // per-head channels (kTeC / kTeHeads)
inline constexpr int   kTeWindow  = 4;     // relative-attention window (emb_rel rows = 9)
inline constexpr int   kNStyle    = 50;    // style tokens
inline constexpr int   kSpteHeads = 2;     // style cross-attn heads
inline constexpr int   kSpteHd    = 128;   // per-head channels (kTeC / kSpteHeads)
inline constexpr float kSpteScale = 16.0f; // QK divisor (upstream Div constant)

// Duration-predictor sentence-encoder constants (from duration_predictor.onnx).
inline constexpr int kDpC      = 64;   // hidden channels
inline constexpr int kDpHeads  = 2;    // attn heads
inline constexpr int kDpKc     = 32;   // per-head channels (kDpC / kDpHeads)
inline constexpr int kDpWindow = 4;    // relative-attention window (emb_rel rows = 9)
inline constexpr int kDpStyle  = 128;  // flattened style_dp (8*16)
inline constexpr int kDpHidden = 128;  // predictor hidden width

// Flow-field RoPE text cross-attention constants.
inline constexpr int   kVeC         = 512;   // field hidden channels
inline constexpr int   kVeHeads     = 8;     // text-attn heads
inline constexpr int   kVeHd        = 64;    // per-head channels
inline constexpr int   kVeHalf      = 32;    // RoPE rotates the 32+32 split halves
inline constexpr float kVeAttnScale = 16.0f; // QK divisor (both field attn types)

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

// Prebuilt RoPE cos/sin tables for one (L, T) pair, shared by every text-attn
// block (they all use the same theta) and by every flow step (L, T fixed for an
// utterance). Hoisted out of rope_attention so the per-step field body carries
// no host->device upload — a hard requirement for CUDA-graph capture, and a win
// regardless (the tables were rebuilt 16x per utterance before).
struct RopeTables {
    brotensor::Tensor cosL, sinL, cosT, sinT;
};

[[noreturn]] void fail(const std::string& msg);
brotensor::Tensor transpose2d(const brotensor::Tensor& x, int R, int C);
std::u32string preprocess_text(const std::string& utf8, const std::string& lang);
std::vector<std::string> split_sentences_impl(const std::string& text, int max_chars);

}  // namespace st_detail

// ─── Impl ────────────────────────────────────────────────────────────────────

struct Supertonic::Impl {
    SupertonicConfig   cfg;
    brotensor::Device  dev = brotensor::Device::CPU;
    bool               ready = false;

    // Vocoder (autoencoder decoder) weights.
    float                               inv_scale = 1.0f;   // 1 / ttl.normalizer.scale
    std::vector<float>                  latent_mean, latent_std;  // (latent_dim)
    st_detail::ConvW                    conv_in;            // 24 -> hidden, kernel 7
    std::vector<st_detail::ConvNeXtBlock> blocks;           // 10 dilated ConvNeXt blocks
    brotensor::Tensor                   bn_g, bn_b, bn_mean, bn_var;  // final BatchNorm
    st_detail::ConvW                    bn_scale_conv;      // depthwise 1x1 = gamma/sqrt(var+eps);
                                                            // the frozen BN-inference input-grad adjoint
    st_detail::ConvW                    head1;              // hidden -> 2*hidden, kernel 3
    float                               prelu_slope = 0.0f;
    st_detail::ConvW                    head2;              // 2*hidden -> base_chunk, kernel 1

    // Text encoder (text_ids + TTL style -> text_emb). Loaded only if
    // text_encoder.safetensors is present alongside the vocoder.
    bool                                te_ready = false;
    std::vector<float>                  te_char_emb;        // [8322 x 256] row-major (host gather)
    int                                 te_vocab = 0;
    std::vector<st_detail::ConvNeXtBlock> te_blocks;        // 6 ConvNeXt (ksz 5, dilated)
    std::vector<st_detail::AttnLayer>   te_attn;            // 4 relative-position attn
    st_detail::StyleAttn                spte1, spte2;       // style cross-attn x2
    brotensor::Tensor                   spte_norm_g, spte_norm_b;  // final LayerNorm
    brotensor::Tensor                   te_style_key;       // [256 x 50] channel-major

    // Duration predictor (text_ids + style_dp -> scalar total duration). Loaded
    // only if duration_predictor.safetensors is present alongside the vocoder.
    bool                                dp_ready = false;
    std::vector<float>                  dp_char_emb;        // [vocab x 64] row-major (host)
    int                                 dp_vocab = 0;
    std::vector<float>                  dp_sentence_token;  // [64] prepended sequence token
    std::vector<st_detail::ConvNeXtBlock> dp_blocks;        // 6 ConvNeXt (ksz 5, dil 1)
    std::vector<st_detail::AttnLayer>   dp_attn;            // 2 relative-position attn
    st_detail::ConvW                    dp_proj;            // proj_out 1x1 (no bias)
    st_detail::Gemm                     dp_pred0, dp_pred1; // predictor MLP (Gemm/PReLU/Gemm)
    float                               dp_prelu = 0.0f;    // predictor PReLU slope

    // Flow-matching vector estimator (the DiT field). Optional until present.
    bool                                ve_ready = false;
    st_detail::ConvW                    ve_proj_in, ve_proj_out;  // 144<->512 convs (no bias)
    std::vector<float>                  ve_t0w, ve_t0b, ve_t1w, ve_t1b;  // time mlp (host)
    std::vector<float>                  ve_sin_freqs;       // [32] sinusoidal frequencies
    std::vector<st_detail::VeBlock>     ve_blocks;          // 24 main blocks
    std::vector<st_detail::ConvNeXtBlock> ve_last;          // 4 last_convnext blocks
    brotensor::Tensor                   ve_style_key_proto; // [256,50] learned style key
    std::vector<float>                  ve_text_uncond;     // [256] text_special_token
    brotensor::Tensor                   ve_style_key_unc;   // [256,50] uncond style key
    brotensor::Tensor                   ve_style_val_unc;   // [256,50] uncond style value
    float                               ve_guid_cond = 4.0f, ve_guid_uncond = 3.0f;  // CFG

    // UnicodeProcessor frontend: codepoint -> token id. Loaded from
    // unicode_indexer.json (a flat 65536-entry uint16->id table, -1 = OOV).
    bool                                frontend_ready = false;
    std::vector<int>                    unicode_index;      // 65536 entries (-1 = OOV)

    void load(const std::string& dir, brotensor::Device device);
    void load_text_encoder(const std::filesystem::path& path);
    void load_duration_predictor(const std::filesystem::path& path);
    void load_vector_estimator(const std::filesystem::path& path);
    AudioBuffer decode(const float* latent, int channels, int frames) const;
    // The vocoder conv stack over a de-chunked, de-normalised real latent
    // [latent_dim, LF] (conv_in -> 10 ConvNeXt -> BN -> head -> waveform).
    // decode() builds `dn` (de-chunk + de-normalise) then calls this.
    AudioBuffer decode_dn(const std::vector<float>& dn, int LF) const;
    std::vector<float> encode_text(const std::vector<int>& ids,
                                   const std::vector<float>& style) const;
    float predict_duration(const std::vector<int>& ids,
                           const std::vector<float>& style_dp) const;
    brotensor::Tensor run_field(const brotensor::Tensor& h0,
                                const brotensor::Tensor& time_emb,
                                const brotensor::Tensor& text_src,
                                const brotensor::Tensor& style_key,
                                const brotensor::Tensor& style_val,
                                int L, int T,
                                const st_detail::RopeTables& rt,
                                const brotensor::Tensor& onesL) const;
    // One classifier-free-guided Euler step done entirely on-device, writing the
    // updated latent back into `noisyT` in place. No host<->device traffic and no
    // host-built tensors inside, so synthesize() can CUDA-graph-capture it and
    // replay it per step. All conditioning (text/style/rope/ones/dt) is prebuilt
    // once per utterance and handed in.
    void field_step(brotensor::Tensor& noisyT,
                    const brotensor::Tensor& time_emb,
                    const brotensor::Tensor& text_cond,
                    const brotensor::Tensor& text_unc,
                    const brotensor::Tensor& style_val_cond,
                    const brotensor::Tensor& onesL,
                    const st_detail::RopeTables& rt,
                    int L, int T, float dt,
                    float guid_cond, float guid_uncond) const;
    std::vector<float> denoise(const std::vector<float>& noisy, int channels,
                               int frames, const std::vector<float>& text_emb,
                               int text_len, const std::vector<float>& style_ttl,
                               int current_step, int total_step) const;
    std::vector<int> text_to_ids(const std::string& text,
                                 const std::string& lang) const;
    AudioBuffer synthesize(const std::string& text, const std::string& lang,
                           const VoiceStyle& voice, int total_step, float speed,
                           std::uint64_t seed, float guidance) const;
    AudioBuffer synthesize_long(const std::string& text, const std::string& lang,
                                const VoiceStyle& voice, int total_step, float speed,
                                std::uint64_t seed, float gap_seconds,
                                int max_chars, float guidance) const;
};

}  // namespace brosoundml
