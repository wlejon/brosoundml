#pragma once

// Qwen3-TTS Base-variant speaker encoder (`speaker_encoder.*`) — an ECAPA-TDNN
// x-vector extractor. A 24 kHz reference clip -> log-mel (128 bands) -> a single
// `enc_dim` (1024) speaker embedding. The zero-shot voice clone splices that
// embedding into the Talker codec prefill in the slot a CustomVoice preset
// speaker token would occupy. Internal to the qwen_tts target; QwenTts::Impl
// owns one (Base only) and the clone synthesis path runs it.
//
// Architecture (ECAPA-TDNN, SpeechBrain-style, but with NO BatchNorm — just
// Conv1d + ReLU): an initial time-delay conv, three SE-Res2Net blocks (TDNN ->
// Res2Net dilated convs -> TDNN -> squeeze-excitation, residual), multi-layer
// feature aggregation, attentive statistics pooling (mean+std over time), and a
// final 1x1 projection to enc_dim. The mel frontend is a reflect-padded STFT
// (periodic Hann) -> magnitude -> librosa slaney mel basis -> log.
//
// Enrollment is one-shot and offline (not in the realtime synthesis loop). The
// convolution stack (the GFLOP-heavy part) runs on brotensor's default device —
// GPU when one is available, CPU otherwise — in FP32, so the embedding is
// numerically identical to the CPU reference (the voice-clone bridge/PCA basis
// were fit against FP32 x-vectors). The mel frontend and the cheap channel
// chunking / pooling reductions stay host-side. The resulting embedding is
// returned as host floats and uploaded to the Talker device by the caller.

#include "brosoundml/qwen_tts.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <vector>

namespace brosoundml {

// One Conv1d (weights flattened OIL: w is (C_out, C_in*k), b is (C_out,1)).
struct QwenTtsSpkConv {
    brotensor::Tensor w, b;
    int cin = 0, cout = 0, k = 1, dilation = 1;
};

// One SE-Res2Net block: tdnn1 (1x1), a Res2Net group of (scale-1) dilated
// convs, tdnn2 (1x1), and a squeeze-excitation gate (two 1x1 convs).
struct QwenTtsSpkSERes2 {
    QwenTtsSpkConv tdnn1, tdnn2;
    std::vector<QwenTtsSpkConv> res2net;   // scale-1 convs (C/scale -> C/scale, k)
    QwenTtsSpkConv se1, se2;               // C -> se_channels -> C
};

struct QwenTtsSpeakerEncoder {
    QwenTtsSpeakerEncoderConfig cfg;
    brotensor::Device device = brotensor::Device::CPU;  // conv weights live here

    QwenTtsSpkConv             block0;     // initial TDNN: mel_dim -> ch[0], k5
    std::vector<QwenTtsSpkSERes2> blocks;  // the three SE-Res2Net blocks
    QwenTtsSpkConv             mfa;        // 3*ch -> ch[-1], 1x1
    QwenTtsSpkConv             asp_tdnn;   // 3*ch[-1] -> attn_ch, 1x1
    QwenTtsSpkConv             asp_conv;   // attn_ch -> ch[-1], 1x1
    QwenTtsSpkConv             fc;         // 2*ch[-1] -> enc_dim, 1x1

    brotensor::Tensor mel_basis;           // (mel_dim, n_fft/2+1) librosa slaney
    brotensor::Tensor hann;                // (1, win_size) periodic Hann

    // Build from the `speaker_encoder.*` tensors of model.safetensors. Conv
    // weights are placed on brotensor's default device (GPU when available) in
    // FP32; the mel frontend tensors stay on CPU. `cfg` is the parsed config.
    void load(const brotensor::safetensors::File& f,
              const QwenTtsSpeakerEncoderConfig& cfg);

    // Log-mel of `n` samples of 24 kHz mono PCM -> (mel_dim, n_frames)
    // channel-major (mel[m*n_frames + t]). Exposed for fixture testing.
    void mel(const float* wav, int n, std::vector<float>& out, int& n_frames) const;

    // Reference clip (n samples, 24 kHz mono) -> enc_dim speaker embedding,
    // returned as host floats.
    std::vector<float> embed(const float* wav, int n) const;
};

}  // namespace brosoundml
