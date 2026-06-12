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

// Scale so the peak sits at a dBFS value drawn uniformly from [lo_db, hi_db].
// The AGC-free presentation-level draw: a raw (no-AGC) mic tap delivers
// whatever level the room does, so every training clip gets a random level
// instead of the fixed peak an AGC would enforce. PCEN cancels static gain
// while frame energy clears pcen_eps; the wide draw also covers the quiet
// regime where that normalization fades. No-op on a silent buffer.
void random_level(std::vector<float>& x, std::mt19937& rng,
                  float lo_db = -45.0f, float hi_db = -3.0f);

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

// ─── Biquad IIR filtering ────────────────────────────────────────────────
//
// Second-order section. Coefficients are normalised so a0 == 1 (folded out).
// All designers below follow the Robert Bristow-Johnson "Audio EQ Cookbook"
// formulas; `sr` is the sample rate, `f0` the corner/centre frequency in Hz,
// `q` the quality factor, and `gain_db` the band gain (used by the peaking and
// shelf types; ignored by low/high-pass). These are the building blocks of the
// acquisition-channel colouration below — modelling a recording chain as a few
// cascaded biquads is cheaper and easier to randomise than convolving a literal
// measured microphone impulse response, and PCEN already absorbs the static
// part of any channel anyway (so the residual the model must see is exactly
// this kind of mild, varied spectral shaping plus the dynamics below).
struct Biquad {
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;   // feed-forward
    float a1 = 0.0f, a2 = 0.0f;              // feed-back (a0 == 1)
};

Biquad design_lowpass  (float sr, float f0, float q);
Biquad design_highpass (float sr, float f0, float q);
Biquad design_peaking  (float sr, float f0, float q, float gain_db);
Biquad design_low_shelf (float sr, float f0, float q, float gain_db);
Biquad design_high_shelf(float sr, float f0, float q, float gain_db);

// Apply one biquad in place (Direct-Form-II transposed, zero initial state).
// Pure FP32 host loop — clips are ~1 s so the O(N) cost is negligible.
void apply_biquad(std::vector<float>& x, const Biquad& bq);

// ─── Acquisition-channel augmentation ──────────────────────────────────────
//
// Waveform-domain DSP that emulates the recording path a real "computer" travels
// through but a Kokoro TTS clip never does: speaker → room → microphone → ADC.
// This is the explicit synthetic→real lever. PCEN (the mel front-end) cancels
// the *static* per-channel gain/tilt, so the job of these blocks is the residual
// PCEN cannot normalise — nonlinear/dynamic channel variation and (via pitch
// jitter) speaker diversity beyond Kokoro's handful of voices. All randomness is
// drawn from the caller's `std::mt19937` so the synth dataset stays byte-
// deterministic per seed; nothing here records into the (fixed) manifest schema.

// Random parametric EQ: a gentle low/high-shelf spectral tilt plus a few random
// peaking bands — microphone colouration.
void apply_random_eq(std::vector<float>& x, int sr, std::mt19937& rng);

// Low-shelf LF boost modelling the close-mic proximity effect.
void apply_proximity_boost(std::vector<float>& x, int sr, std::mt19937& rng);

// Random band-limiting: a high-pass (~60–200 Hz) and a low-pass (~3.4–7.5 kHz)
// modelling the limited bandwidth of a cheap or distant microphone.
void apply_bandlimit(std::vector<float>& x, int sr, std::mt19937& rng);

// Feed-forward dynamic-range compressor. Envelope follower (attack/release) over
// the instantaneous level, static dB threshold/ratio curve, then makeup gain —
// emulating mic-preamp / AGC dynamics. Operates in place.
struct CompressorConfig {
    float threshold_db = -24.0f;  // gain reduction begins above this level
    float ratio        = 3.0f;    // > 1
    float attack_ms    = 5.0f;
    float release_ms   = 80.0f;
    float makeup_db     = 0.0f;
};
void apply_drc(std::vector<float>& x, int sr, const CompressorConfig& cfg);

// Resample-based pitch jitter: shift pitch by `semitones` (positive = up) by
// resampling and reinterpreting at the original rate. This couples pitch and
// duration (a true formant-independent shift needs a phase vocoder and is
// deliberately deferred); the caller re-crops to the fixed grid, so the coupled
// duration change is absorbed. Returns a new buffer; `x` is unchanged.
std::vector<float> pitch_jitter(const std::vector<float>& x, int sr,
                                float semitones);

// Toggles for the individual stages of apply_acquisition_channel. Each enabled
// stage additionally fires with an internal probability drawn from `rng`, so a
// fraction of clips pass through with only some stages active.
struct AcquisitionChannel {
    bool do_pitch     = true;
    bool do_rir       = true;
    bool do_eq        = true;
    bool do_proximity = true;
    bool do_bandlimit = true;
    bool do_drc       = true;
};

// Apply a randomized recording channel in physical order:
//   pitch jitter → crop/pad to target_len → room RIR → EQ + proximity →
//   band-limit → DRC.
// `target_len` re-lands the (possibly length-changed) pitch-jittered buffer on
// the dataset grid before the rest of the chain. The output is NOT peak-
// normalised — the caller mixes ambient noise and normalises afterwards.
std::vector<float> apply_acquisition_channel(const std::vector<float>& x,
                                             int sr, int target_len,
                                             std::mt19937& rng,
                                             const AcquisitionChannel& cfg = {});

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

// ─── Manifest reader ───────────────────────────────────────────────────────
//
// Parses the CSV emitted by `Manifest::append`. Throws on a missing/wrong
// header, the wrong column count, or an unknown class label. Honours the
// RFC-4180 quoting used by `csv_escape_field` (double-quote wrapped, embedded
// `""` → `"`). Newlines/CRs inside quoted fields are accepted.
std::vector<ManifestRow> read_manifest(const std::string& path);

// ─── Audio quality stats ───────────────────────────────────────────────────

struct AudioStats {
    float peak           = 0.0f;
    float rms            = 0.0f;
    int   silent_samples = 0;   // |x| < 1e-4
    int   clipped_samples= 0;   // |x| >= 0.999
    bool  has_nan        = false;
};

AudioStats compute_audio_stats(const std::vector<float>& x);

// ─── Dataset validation ────────────────────────────────────────────────────

struct ValidationConfig {
    int   expected_sample_rate = 16000;
    float min_duration_s       = 0.95f;
    float max_duration_s       = 1.05f;
    // Per-clip silence / clip / NaN gates expressed as fractions of the buffer.
    float max_silent_fraction  = 0.99f;  // a clip is "silent" if >= this
    float max_clipped_fraction = 0.01f;  // a clip is "clipped" if >= this
    // Dataset-wide thresholds (percent of total clips).
    float max_silent_pct       = 1.0f;
    float max_clipped_pct      = 0.5f;
    // Class-balance requirements.
    bool  require_positive_class = true;
    bool  require_negative_class = true;
};

// One bucket = 5 dB; index 0 is the "clean" bucket (rows with no noise_kind).
struct DatasetReport {
    int total_rows            = 0;
    int missing_files         = 0;
    int malformed_wavs        = 0;
    int sample_rate_mismatch  = 0;
    int duration_out_of_range = 0;
    int label_path_mismatch   = 0;
    int unknown_class         = 0;
    int silent_clips          = 0;
    int clipped_clips         = 0;
    int nan_clips             = 0;
    int rejected_clips        = 0;   // unique clips that hit >= 1 anomaly

    // Distribution summaries over the clips that were successfully decoded.
    int   decoded_clips = 0;
    float peak_min  = 0.0f, peak_mean = 0.0f, peak_max = 0.0f, peak_std = 0.0f;
    float rms_min   = 0.0f, rms_mean  = 0.0f, rms_max  = 0.0f, rms_std  = 0.0f;

    // path,label,class,voice,speed,snr_db,noise_kind keyed histograms.
    // class -> count
    std::vector<std::pair<std::string, int>> class_counts;
    // label -> count (label is 0/1 → key "0" / "1")
    std::vector<std::pair<std::string, int>> label_counts;
    // voice -> count (skips empty voice)
    std::vector<std::pair<std::string, int>> voice_counts;
    // noise_kind -> count ("" rendered as "clean")
    std::vector<std::pair<std::string, int>> noise_counts;
    // speed bucket key (e.g. "1.00") -> count, bucketed by 0.05
    std::vector<std::pair<std::string, int>> speed_counts;
    // snr bucket key (e.g. "clean", "-5", "0", "5", "10", ...) -> count
    std::vector<std::pair<std::string, int>> snr_counts;
};

DatasetReport validate_dataset(const std::vector<ManifestRow>& rows,
                               const std::string& dataset_root,
                               const ValidationConfig& cfg);

// True if the report clears every dataset-wide threshold and no fatal
// per-row anomalies (missing files, malformed WAVs, label/path mismatch,
// unknown class) were recorded. On failure, fills `*reason` with a one-line
// human-readable description of the first failing check. `reason` may be
// nullptr.
bool report_passes(const DatasetReport& r, const ValidationConfig& cfg,
                   std::string* reason);

// Plain-text human summary. The output line order is stable so two runs over
// the same dataset diff cleanly.
void print_report(const DatasetReport& r, std::FILE* out);

}  // namespace brosoundml
