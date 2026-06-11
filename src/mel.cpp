#include "brosoundml/mel.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

constexpr float kEps = 1e-10f;

// HTK mel: m = 2595 * log10(1 + f/700).
double hz_to_mel_htk(double hz) {
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}
double mel_to_hz_htk(double m) {
    return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0);
}

// Triangular HTK mel filterbank, (n_mels, n_bins) row-major.
std::vector<float> build_mel_filterbank_htk(int n_mels, int n_fft,
                                            int sample_rate,
                                            double fmin, double fmax) {
    const int n_bins = n_fft / 2 + 1;
    // Bin centre frequencies (Hz) — linearly spaced from 0 to sr/2 across the
    // n_bins rfft outputs.
    std::vector<double> bin_hz(static_cast<std::size_t>(n_bins));
    for (int k = 0; k < n_bins; ++k) {
        bin_hz[static_cast<std::size_t>(k)] =
            static_cast<double>(k) * sample_rate / static_cast<double>(n_fft);
    }
    // n_mels + 2 mel-spaced edges, converted back to Hz.
    const double mel_lo = hz_to_mel_htk(fmin);
    const double mel_hi = hz_to_mel_htk(fmax);
    std::vector<double> edges_hz(static_cast<std::size_t>(n_mels) + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        const double m = mel_lo + (mel_hi - mel_lo) * i / (n_mels + 1);
        edges_hz[static_cast<std::size_t>(i)] = mel_to_hz_htk(m);
    }
    std::vector<float> fb(static_cast<std::size_t>(n_mels) * n_bins, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        const double lo = edges_hz[static_cast<std::size_t>(m)];
        const double ce = edges_hz[static_cast<std::size_t>(m) + 1];
        const double hi = edges_hz[static_cast<std::size_t>(m) + 2];
        for (int k = 0; k < n_bins; ++k) {
            const double f = bin_hz[static_cast<std::size_t>(k)];
            double w = 0.0;
            if (f >= lo && f <= ce && ce > lo) {
                w = (f - lo) / (ce - lo);
            } else if (f > ce && f <= hi && hi > ce) {
                w = (hi - f) / (hi - ce);
            }
            if (w < 0.0) w = 0.0;
            fb[static_cast<std::size_t>(m) * n_bins + k] = static_cast<float>(w);
        }
    }
    return fb;
}

// Periodic Hann: w[n] = 0.5 * (1 - cos(2*pi*n / N)). Matches torch.hann_window
// (periodic=True, the default).
std::vector<float> build_hann_window(int win_length) {
    std::vector<float> w(static_cast<std::size_t>(win_length));
    for (int n = 0; n < win_length; ++n) {
        const double phase = 2.0 * 3.14159265358979323846 *
                             static_cast<double>(n) /
                             static_cast<double>(win_length);
        w[static_cast<std::size_t>(n)] =
            static_cast<float>(0.5 * (1.0 - std::cos(phase)));
    }
    return w;
}

// Compute LINEAR mel frames from a (1, signal_len) CPU FP32 signal — assumes
// signal_len >= win_length. Pipeline runs on `device`: stft → complex_abs →
// matmul(mel_filter), all out tensors device-resident. The linear mel energy
// is downloaded to `out_linear_TM` as (T, n_mels) row-major (T frames, each a
// contiguous n_mels row — the layout PCEN's per-frame recursion wants).
// Compression (log / PCEN) and the transpose to (n_mels, T) happen in the
// MelFrontend member that owns the streaming state. Returns T.
int compute_frames(const float* signal_host, int signal_len,
                   const MelConfig& cfg,
                   const bt::Tensor& mel_filter_T,
                   const bt::Tensor& window,
                   bt::Device device,
                   std::vector<float>& out_linear_TM) {
    const std::string where = "MelFrontend::compute_frames";
    if (signal_len < cfg.win_length) {
        fail(where, "signal_len < win_length — caller must filter this out");
    }
    const int T = 1 + (signal_len - cfg.win_length) / cfg.hop_length;

    // STFT (center=false): frames = 1 + (L - n_fft)/hop. We're using
    // win_length <= n_fft framing with center=false — the op centres the
    // win_length samples inside the n_fft buffer per frame; the first frame
    // starts at sample 0. The op's center=false formula uses n_fft (not
    // win_length) for the frame-count denominator, so feed it a buffer sized
    // so the formula yields the same T as our offline rule. Specifically:
    // 1 + (L_padded - n_fft)/hop == T  ⇒  L_padded == (T-1)*hop + n_fft.
    // We right-pad with zeros from signal_len up to L_padded. The mel result
    // is identical because the analysis window has length win_length and is
    // centred in n_fft, so the n_fft - win_length tail samples are multiplied
    // by zero — they never reach the spectrum.
    // brotensor's stft (center=false) frames the signal as:
    //   frame f reads padded[f*hop + pad_lo : f*hop + pad_lo + win_length)
    //   with pad_lo = (n_fft - win_length) / 2.
    // For T frames we need padded length L_padded = (T-1)*hop + n_fft (from
    // brotensor's frames = 1 + (L - n_fft)/hop), and to make frame f land on
    // our logical sample f*hop we left-pad the signal by pad_lo zeros. That
    // produces matching tail-pad on the right too: L_padded - (pad_lo +
    // signal_len) = (n_fft - win_length)/2 = pad_lo.
    const int pad_lo   = (cfg.n_fft - cfg.win_length) / 2;
    const int L_padded = (T - 1) * cfg.hop_length + cfg.n_fft;
    bt::Tensor signal;  // (1, L_padded) on `device` — stft expects (N, len)
    {
        std::vector<float> padded(static_cast<std::size_t>(L_padded), 0.0f);
        // Clip the copy to the destination capacity. signal_len may exceed
        // (T-1)*hop + win_length when the caller passes a non-aligned N; the
        // extra tail samples sit past the last frame's read window anyway.
        const int max_copy = L_padded - pad_lo;
        const int copy_n   = std::min(signal_len, max_copy);
        std::memcpy(padded.data() + pad_lo, signal_host,
                    static_cast<std::size_t>(copy_n) * sizeof(float));
        signal = bt::Tensor::from_host_on(device, padded.data(), 1, L_padded);
    }

    bt::Tensor spec;
    bt::stft(signal, window, /*N=*/1, cfg.n_fft, cfg.hop_length, cfg.win_length,
             /*center=*/false, /*normalized=*/false, spec);
    if (spec.rows < T) {
        fail(where, "stft returned " + std::to_string(spec.rows) +
                    " frames; expected " + std::to_string(T));
    }

    // Magnitude: (T, n_bins) on `device`.
    bt::Tensor mag;
    bt::complex_abs(spec, mag);
    // Drop any extra trailing frames stft may have emitted (defensive — the
    // L_padded math above is exact, so this is a no-op in practice).
    if (mag.rows != T) {
        // View the first T rows — but Tensor::view returns a non-owning view
        // and downstream matmul would refuse to resize a view output. Instead
        // copy out the prefix via a host round-trip. This branch should never
        // fire under the L_padded math; assert it doesn't to avoid the cost.
        fail(where, "magnitude frame count " + std::to_string(mag.rows) +
                    " != expected " + std::to_string(T));
    }

    // Mel projection: (T, n_bins) @ (n_bins, n_mels) = (T, n_mels). The
    // transposed filterbank is built once at construction and lives on
    // `device` — corpus tools call this per clip, so a per-call host
    // round-trip of the filterbank is pure overhead.
    bt::Tensor mel_T;  // (T, n_mels) linear
    bt::matmul(mag, mel_filter_T, mel_T);

    // Download the linear mel energy as (T, n_mels). Compression (log / PCEN)
    // is applied by the caller, which owns the streaming state PCEN needs.
    out_linear_TM = mel_T.to_host_vector();
    return T;
}

}  // namespace

// ─── MelFrontend ───────────────────────────────────────────────────────────

MelFrontend::MelFrontend(const MelConfig& cfg, bt::Device device)
    : cfg_(cfg), device_(device) {
    const std::string where = "MelFrontend::MelFrontend";
    if (cfg_.sample_rate <= 0) fail(where, "sample_rate must be positive");
    if (cfg_.n_fft <= 0)       fail(where, "n_fft must be positive");
    if (cfg_.win_length <= 0)  fail(where, "win_length must be positive");
    if (cfg_.hop_length <= 0)  fail(where, "hop_length must be positive");
    if (cfg_.n_mels <= 0)      fail(where, "n_mels must be positive");
    if (cfg_.win_length > cfg_.n_fft) {
        fail(where, "win_length (" + std::to_string(cfg_.win_length) +
                    ") > n_fft (" + std::to_string(cfg_.n_fft) + ")");
    }
    if (!(cfg_.fmin >= 0.0f && cfg_.fmax > cfg_.fmin)) {
        fail(where, "require 0 <= fmin < fmax");
    }
    if (cfg_.fmax > static_cast<float>(cfg_.sample_rate) / 2.0f + 1e-3f) {
        fail(where, "fmax exceeds Nyquist (sample_rate/2)");
    }
    if (cfg_.formula != MelFormula::HTK) {
        fail(where, "only MelFormula::HTK is implemented");
    }
    if (cfg_.window != MelWindow::Hann) {
        fail(where, "only MelWindow::Hann is implemented");
    }
    bt::init();

    // Host-build the mel filterbank + window, then upload to `device_`.
    std::vector<float> fb = build_mel_filterbank_htk(
        cfg_.n_mels, cfg_.n_fft, cfg_.sample_rate, cfg_.fmin, cfg_.fmax);
    const int n_bins = cfg_.n_fft / 2 + 1;
    mel_filter_ = bt::Tensor::from_host_on(device_, fb.data(),
                                           cfg_.n_mels, n_bins);
    std::vector<float> fb_T(static_cast<std::size_t>(n_bins) * cfg_.n_mels);
    for (int m = 0; m < cfg_.n_mels; ++m) {
        for (int k = 0; k < n_bins; ++k) {
            fb_T[static_cast<std::size_t>(k) * cfg_.n_mels + m] =
                fb[static_cast<std::size_t>(m) * n_bins + k];
        }
    }
    mel_filter_T_ = bt::Tensor::from_host_on(device_, fb_T.data(),
                                             n_bins, cfg_.n_mels);

    std::vector<float> w = build_hann_window(cfg_.win_length);
    window_ = bt::Tensor::from_host_on(device_, w.data(), 1, cfg_.win_length);

    ring_.reserve(static_cast<std::size_t>(cfg_.win_length + cfg_.hop_length));
}

void MelFrontend::reset() {
    ring_.clear();
    samples_dropped_ = 0;
    pcen_init_ = false;            // next frame re-seeds the PCEN smoother
}

// Apply cfg_.compression to linear mel laid out (T, n_mels) row-major, writing
// the result transposed to (n_mels, T) into `out` on device_. For PCEN this
// advances pcen_m_ one frame at a time, carrying state across calls so the
// streaming path matches compute_offline frame-for-frame.
void MelFrontend::compress_and_emit(const std::vector<float>& linear_mel_TM,
                                    int T, bt::Tensor& out) {
    const int M = cfg_.n_mels;
    std::vector<float> out_host(static_cast<std::size_t>(M) * T);

    if (cfg_.compression == MelCompression::Log) {
        // log(max(mel, eps)) — bit-identical to the previous front-end.
        for (int t = 0; t < T; ++t) {
            for (int m = 0; m < M; ++m) {
                float v = linear_mel_TM[static_cast<std::size_t>(t) * M + m];
                if (v < kEps) v = kEps;
                out_host[static_cast<std::size_t>(m) * T + t] = std::log(v);
            }
        }
    } else {  // PCEN
        const float s     = cfg_.pcen_s;
        const float alpha = cfg_.pcen_alpha;
        const float delta = cfg_.pcen_delta;
        const float r     = cfg_.pcen_r;
        const float eps   = cfg_.pcen_eps;
        const float delta_r = std::pow(delta, r);
        if (static_cast<int>(pcen_m_.size()) != M) pcen_m_.assign(M, 0.0f);
        for (int t = 0; t < T; ++t) {
            for (int m = 0; m < M; ++m) {
                const float E = linear_mel_TM[static_cast<std::size_t>(t) * M + m];
                float& Mst = pcen_m_[static_cast<std::size_t>(m)];
                // Seed the smoother with the first frame's energy to avoid a
                // startup transient (matches librosa's filter init).
                Mst = pcen_init_ ? (1.0f - s) * Mst + s * E : E;
                const float smooth = std::pow(eps + Mst, alpha);
                const float v = std::pow(E / smooth + delta, r) - delta_r;
                out_host[static_cast<std::size_t>(m) * T + t] = v;
            }
            pcen_init_ = true;
        }
    }
    out = bt::Tensor::from_host_on(device_, out_host.data(), M, T);
}

int MelFrontend::frames_buffered() const {
    return static_cast<int>(ring_.size());
}

int MelFrontend::consume(const float* samples, int n,
                         bt::Tensor& out_frames_appended) {
    const std::string where = "MelFrontend::consume";
    if (n < 0) fail(where, "negative sample count");
    if (n > 0 && samples == nullptr) fail(where, "null samples with n > 0");

    if (n > 0) {
        ring_.insert(ring_.end(), samples, samples + n);
    }
    const int buffered = static_cast<int>(ring_.size());
    if (buffered < cfg_.win_length) {
        return 0;
    }
    // How many frames can we emit from the current ring?
    const int T_new = 1 + (buffered - cfg_.win_length) / cfg_.hop_length;
    if (T_new <= 0) return 0;

    // The compute pass needs a contiguous host buffer covering the samples
    // for those T_new frames. Frame f spans [f*hop, f*hop + win_length).
    // Frame T_new-1 ends at (T_new-1)*hop + win_length, which is the chunk
    // length we hand to compute_frames.
    const int chunk_len = (T_new - 1) * cfg_.hop_length + cfg_.win_length;

    std::vector<float> linear_TM;
    compute_frames(ring_.data(), chunk_len, cfg_, mel_filter_T_, window_,
                   device_, linear_TM);
    bt::Tensor frames;  // (n_mels, T_new) after compression
    compress_and_emit(linear_TM, T_new, frames);

    // Drop the consumed prefix from the ring. Carry-over is whatever's past
    // sample (T_new * hop_length) — the start of the next un-emitted frame.
    const int consumed = T_new * cfg_.hop_length;
    if (consumed >= buffered) {
        ring_.clear();
    } else {
        ring_.erase(ring_.begin(), ring_.begin() + consumed);
    }
    samples_dropped_ += consumed;

    // Append `frames` (n_mels, T_new) along the time axis of out_frames_appended
    // (n_mels, T_existing). Result: (n_mels, T_existing + T_new). The whole
    // append happens on the host (download → splice → upload) because there's
    // no concat op exposed and the per-call sizes are small.
    if (out_frames_appended.rows == 0 && out_frames_appended.cols == 0) {
        out_frames_appended = frames;
        return T_new;
    }
    if (out_frames_appended.rows != cfg_.n_mels) {
        fail(where, "out_frames_appended.rows (" +
                    std::to_string(out_frames_appended.rows) +
                    ") != n_mels (" + std::to_string(cfg_.n_mels) + ")");
    }
    const int T_old = out_frames_appended.cols;
    std::vector<float> old_host = out_frames_appended.to_host_vector();
    std::vector<float> new_host = frames.to_host_vector();
    std::vector<float> merged(
        static_cast<std::size_t>(cfg_.n_mels) * (T_old + T_new));
    for (int m = 0; m < cfg_.n_mels; ++m) {
        // Old prefix: T_old samples per row.
        std::memcpy(
            merged.data() +
                static_cast<std::size_t>(m) * (T_old + T_new),
            old_host.data() + static_cast<std::size_t>(m) * T_old,
            static_cast<std::size_t>(T_old) * sizeof(float));
        // New suffix: T_new samples per row.
        std::memcpy(
            merged.data() +
                static_cast<std::size_t>(m) * (T_old + T_new) + T_old,
            new_host.data() + static_cast<std::size_t>(m) * T_new,
            static_cast<std::size_t>(T_new) * sizeof(float));
    }
    out_frames_appended = bt::Tensor::from_host_on(
        device_, merged.data(), cfg_.n_mels, T_old + T_new);
    return T_new;
}

void MelFrontend::compute_offline(const float* samples, int n,
                                  bt::Tensor& out) {
    const std::string where = "MelFrontend::compute_offline";
    if (n < 0) fail(where, "negative sample count");
    if (n > 0 && samples == nullptr) fail(where, "null samples with n > 0");

    if (n < cfg_.win_length) {
        // Zero frames. Allocate an empty (n_mels, 0) tensor for shape sanity.
        out = bt::Tensor::empty_on(device_, cfg_.n_mels, 0, bt::Dtype::FP32);
        return;
    }
    // One-shot: re-seed PCEN from this call's first frame (independent of any
    // prior streaming state). No-op for Log.
    pcen_init_ = false;
    std::vector<float> linear_TM;
    const int T = compute_frames(samples, n, cfg_, mel_filter_T_, window_,
                                 device_, linear_TM);
    compress_and_emit(linear_TM, T, out);
}

}  // namespace brosoundml
