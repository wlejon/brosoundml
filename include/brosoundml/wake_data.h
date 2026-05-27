#pragma once

// brosoundml::wake_data — pure-DSP / IO helpers behind brosoundml_wake_synth.
//
// Kept independent of Kokoro / G2P so the unit tests link without any model
// artefacts on disk. The CLI tool in tools/wake_synth.cpp drives Kokoro to
// fill in the speech buffers, then leans on these helpers for noise, mixing,
// reverberation, resampling, and manifest assembly.

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace brosoundml {

// ─── Noise generators ──────────────────────────────────────────────────────
//
// All three produce zero-mean FP32 samples. `amplitude` scales the output
// linearly; pink/brown filters are normalised so a unit-amplitude request
// produces roughly unit-RMS output regardless of the filter chosen, but the
// per-octave roll-off is the load-bearing property — exact RMS is left to the
// SNR mixer.
std::vector<float> gen_white_noise(int n_samples, float amplitude,
                                   std::mt19937& rng);
std::vector<float> gen_pink_noise (int n_samples, float amplitude,
                                   std::mt19937& rng);
std::vector<float> gen_brown_noise(int n_samples, float amplitude,
                                   std::mt19937& rng);

enum class NoiseKind : std::uint8_t { White = 0, Pink = 1, Brown = 2 };

// Dispatch helper — returns the kind's name as the manifest spells it
// ("white" | "pink" | "brown"). `gen_noise` writes into a pre-sized output.
const char*        noise_kind_name(NoiseKind k);
std::vector<float> gen_noise(NoiseKind k, int n_samples, float amplitude,
                             std::mt19937& rng);

// ─── Level helpers ─────────────────────────────────────────────────────────

float rms(const std::vector<float>& x);
float peak(const std::vector<float>& x);

// In-place dB gain. +6 dB ≈ ×2.
void apply_gain_db(std::vector<float>& x, float gain_db);

// Scale so peak(x) == target. No-op on a silent buffer.
void peak_normalize(std::vector<float>& x, float target = 0.99f);

// ─── SNR mixer ─────────────────────────────────────────────────────────────
//
// Returns signal + scaled noise, where `noise` is rescaled so that
//   10 * log10(rms(signal)^2 / rms(scaled_noise)^2) == snr_db.
// `signal` and `noise` must be the same length. If `signal` is silent
// (rms == 0) the mix is skipped and `signal` is returned unchanged — there
// is no meaningful SNR to target. If `noise` is silent the function also
// returns `signal` unchanged.
std::vector<float> mix_at_snr(const std::vector<float>& signal,
                              const std::vector<float>& noise,
                              float snr_db);

// ─── Synthetic room impulse responses ──────────────────────────────────────
//
// Each RIR is an exponential-decay envelope sampled at `sample_rate`, with a
// handful of randomly-placed early-reflection taps overlaid in the first
// ~50 ms. Length is `rir_len`, normalised so the first non-zero tap equals 1
// — i.e. the direct path is unit gain. Direct-form host convolution is fine
// because every wake clip is ~1 s and every RIR is ~200 ms at 16 kHz.
std::vector<float> gen_rir(int sample_rate, int rir_len, float t60_seconds,
                           int n_early_taps, std::mt19937& rng);

// Causal direct-form 1D convolution: y[n] = sum_k h[k] * x[n - k]. Output is
// `signal.size()` long (truncated to the input length) — the canonical
// "in-room recording" length. Pure FP32 host loop; both operands must be
// non-empty.
std::vector<float> convolve_rir(const std::vector<float>& signal,
                                const std::vector<float>& rir);

// ─── Resampler ─────────────────────────────────────────────────────────────
//
// Thin wrapper over brotensor::resample1d_forward (linear mode). Runs on the
// CPU device — every caller in the wake-synth tool round-trips host buffers,
// so there is no win from staging on a GPU. Throws std::runtime_error on a
// non-positive rate or an empty input.
std::vector<float> resample_to(const std::vector<float>& in,
                               int in_rate, int out_rate);

// ─── Clip shaping ──────────────────────────────────────────────────────────
//
// Centre-crop or zero-pad `samples` to exactly `target_len` samples. Used to
// land every Kokoro output on the fixed 16 kHz × 1 s grid of the dataset.
std::vector<float> crop_or_pad_centered(const std::vector<float>& samples,
                                        int target_len);

// ─── Manifest writer ───────────────────────────────────────────────────────
//
// Incremental CSV writer. The header (`path,label,class,voice,speed,
// snr_db,noise_kind,seed`) is emitted by the first append; subsequent appends
// only push rows. Fields containing a comma, quote, CR or LF are double-quote
// quoted and any embedded `"` is escaped as `""` (RFC-4180 quoting).
// Line endings are `\n` so byte-comparing two seeded runs is portable.

struct ManifestRow {
    std::string path;        // path relative to the dataset root
    int         label = 0;   // 1 = positive ("computer"), 0 = negative
    std::string clazz;       // "positive" | "confusable" | "sentence" | "noise"
    std::string voice;       // voice pack file stem, or "" for pure-noise rows
    float       speed = 1.0f;
    float       snr_db = 0.0f; // 0 for clean rows (no noise added)
    std::string noise_kind;  // "white"|"pink"|"brown"|"" (clean)
    std::uint64_t seed = 0;  // per-row sub-seed used to derive RNG state
};

class Manifest {
 public:
    explicit Manifest(const std::string& path);
    ~Manifest();

    // Append a single row. Writes the CSV header on the first call.
    void append(const ManifestRow& row);

    // Total rows appended (excluding the header).
    int rows() const { return rows_; }

    // Path of the manifest file on disk.
    const std::string& path() const { return path_; }

 private:
    std::string path_;
    void*       fp_ = nullptr;   // FILE*, opaque to avoid <cstdio> in the header
    bool        wrote_header_ = false;
    int         rows_ = 0;
};

// Escape a single field per RFC 4180. Public for the tests.
std::string csv_escape_field(const std::string& s);

}  // namespace brosoundml
