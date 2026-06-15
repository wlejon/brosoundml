#pragma once

#include <brotensor/tensor.h>

namespace brosoundml {

// ─── SupertonicSpec ──────────────────────────────────────────────────────────
//
// The Supertonic AE encoder's input "spec_processor": a 44.1 kHz waveform →
// [idim, T] log-spectral feature, where idim = (n_fft/2 + 1) linear-magnitude
// bins ⊕ n_mels HTK-mel bins (1025 ⊕ 228 = 1253, from tts.json `ae.encoder`).
//
// This is a FIXED input transform — there is no backward. During encoder
// training it is computed once per clip and cached, the same way wake/phoneme
// cache their mel front-ends. The compute path is all brotensor ops:
//   stft (center) → complex_abs → [linear mag | mel = mag @ filterbank]
//   → log(clamp(·, eps)).
//
// Framing matches torch.stft(center=True): the signal is reflect-padded by
// n_fft/2 each side, so T = 1 + L / hop_length. With hop_length == the AE
// base_chunk_size (512), one spec frame corresponds to one decoder output
// chunk — the caller crops/pads to the decoder's latent frame count.
class SupertonicSpec {
public:
    // n_fft 2048, win 2048, hop 512, n_mels 228, sr 44100 — the upstream config.
    explicit SupertonicSpec(brotensor::Device device,
                            int sample_rate = 44100, int n_fft = 2048,
                            int win_length = 2048, int hop_length = 512,
                            int n_mels = 228, float fmin = 0.0f,
                            float fmax = 22050.0f, float eps = 1.0e-5f);

    // signal: (1, L) host FP32 @ sample_rate (L >= 1). Returns [idim, T]
    // channel-major on the device, with T = 1 + L / hop_length.
    brotensor::Tensor compute(const float* signal, int L) const;

    int idim()       const { return n_bins_ + n_mels_; }
    int n_bins()     const { return n_bins_; }
    int n_mels()     const { return n_mels_; }
    int hop_length() const { return hop_; }

private:
    brotensor::Device dev_;
    int   sr_, n_fft_, win_, hop_, n_mels_, n_bins_;
    float eps_;
    brotensor::Tensor window_;        // (1, win) periodic Hann
    brotensor::Tensor mel_filter_T_;  // (n_bins, n_mels) transposed HTK filterbank
};

}  // namespace brosoundml
