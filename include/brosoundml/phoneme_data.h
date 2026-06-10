#pragma once

// brosoundml::phoneme_data — data layer for open-vocabulary phoneme-posterior
// keyword spotting. The phoneme analogue of wake_data.h.
//
// This layer owns four things, all pure host DSP / IO (no model artefacts, so
// the unit tests link without any weights on disk):
//   A. PhonemeClassMap — the model's output inventory. A partition of the
//      Kokoro phoneme-id space into K integer classes (class 0 == silence).
//   B. build_frame_labels — turn Kokoro's per-phoneme duration vector into a
//      per-10 ms-frame class-label track on a 16 kHz timeline, frame-aligned to
//      the log-mel front-end (mel.h) framing formula.
//   C. A packed binary frame-labelled dataset format (BPDS): PCM + per-frame
//      labels + the embedded class map, written incrementally and read back with
//      full validation.
//   D. A dataset validator mirroring wake_data's validate/report/passes/print.
//
// The augmentation / resample helpers a synth tool needs (gen_noise,
// mix_at_snr, apply_acquisition_channel, resample_to, crop_or_pad_centered)
// already live in wake_data.h and are REUSED — they are not duplicated here.

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace brosoundml {

// ─── A. Phoneme class map / inventory ───────────────────────────────────────
//
// A model class is one integer in [0, K). Class 0 is ALWAYS the dedicated
// silence / non-speech class. Classes 1..K-1 each own a set of Kokoro phoneme
// ids. The map is a partition of the phoneme-id space used by the dataset:
// every id that can appear in a frame label maps to exactly one class.
//
// class_to_ids[0] (silence) is NOT necessarily empty: the default-English
// builder routes punctuation / whitespace / pause vocab ids into the silence
// class, because those tokens correspond to pauses and CAN appear in a
// phonemizer's id stream. This intentionally deviates from the abstract "silence
// owns no ids" view so that class_for_id is total over every id the dataset can
// produce (an unmapped id is surfaced as a bug, never silently silenced).
struct PhonemeClassMap {
    int                           num_classes = 0;   // K (includes silence = 0)
    std::vector<std::string>      class_names;       // size K; [0] == "sil"
    std::vector<std::vector<int>> class_to_ids;      // size K; partition of ids

    // Suprasegmental / diacritic modifier ids (stress, length, aspiration,
    // palatalization, nasalization, intonation arrows). These are ALSO members
    // of the silence class in class_to_ids (so class_for_id maps them to silence
    // — the spotter strips them from enrolled templates), but they are flagged
    // here as "transparent": when build_frame_labels paints a dataset's frame
    // labels, a transparent token's frames are NOT silenced — they inherit the
    // class of the nearest adjacent segmental phoneme. A stress mark precedes a
    // stressed vowel, so its frames are voiced speech, and silencing them would
    // both mislabel vowel onsets and punch spurious mid-word "silence" gaps that
    // the spotter's entry gate would misread. Empty for hand-built / toy maps.
    std::vector<int>              transparent_ids;

    // Class owning `phoneme_id`. Throws std::runtime_error if `phoneme_id` is
    // not present in the map — a dataset built against a given vocab must be
    // fully covered, so an unmapped id is a bug, not silence. The inverse index
    // is built lazily from class_to_ids on first use (and after a manual
    // rebuild_inverse()), so a freshly-deserialized map works directly.
    int class_for_id(int phoneme_id) const;

    // True if `phoneme_id` is a transparent modifier (see transparent_ids). The
    // lookup set is built lazily alongside the inverse index.
    bool is_transparent(int phoneme_id) const;

    int silence_class() const { return 0; }

    // Force a rebuild of the id -> class inverse index (and the transparent set).
    // Call after mutating class_to_ids / transparent_ids by hand; the builder
    // and reader already call it.
    void rebuild_inverse() const;

    // Structural equality (ignores the cached lookup indices).
    bool operator==(const PhonemeClassMap& o) const {
        return num_classes == o.num_classes &&
               class_names == o.class_names &&
               class_to_ids == o.class_to_ids &&
               transparent_ids == o.transparent_ids;
    }
    bool operator!=(const PhonemeClassMap& o) const { return !(*this == o); }

 private:
    // id -> class inverse, built from class_to_ids. Mutable so class_for_id can
    // populate it lazily on a const map.
    mutable std::unordered_map<int, int> id_to_class_;
    mutable std::unordered_map<int, bool> transparent_set_;  // id -> transparent
    mutable bool                         inverse_built_ = false;
};

// Build the default US-English inventory over a Kokoro phoneme vocab (IPA-string
// -> id, the shape of g2p::PhonemeAdapter's vocab). Policy:
//   * ~44 named phoneme classes (monophthongs, diphthongs, stops, fricatives,
//     affricates, nasals, liquids, glides) plus silence(0) and an "other"
//     catch-all, so K is typically ~46 for a real vocab (it varies with the
//     vocab's exact symbol set).
//   * Each named class lists several candidate IPA spellings (pure IPA and the
//     misaki single-char diphthong letters); spellings absent from `vocab` are
//     dropped. An empty class is kept (it simply never appears).
//   * Every vocab id NOT claimed by a named class is still covered: a SPOKEN
//     leftover id goes to "other"; a NON-spoken id (punctuation / whitespace /
//     digit / control — a key whose bytes are all ASCII non-letters, or empty)
//     goes to the silence class, since those are pauses.
//   * Deterministic: class order is fixed; ids within each class are sorted, so
//     two builds over the same vocab are byte-identical.
PhonemeClassMap build_default_english_classmap(
    const std::unordered_map<std::string, int>& vocab);

// ─── Class-map serialization (little-endian) ────────────────────────────────
//
// Layout: u32 num_classes, then per class { u32 name_len, name bytes,
// u32 id_count, id_count * i32 ids }, then u32 transparent_count followed by
// transparent_count * i32 transparent ids. Embedded verbatim in the dataset
// header (and, later, in a model checkpoint). The reader rebuilds the indices.
void            write_classmap(std::FILE* f, const PhonemeClassMap& cm);
PhonemeClassMap read_classmap (std::FILE* f);

// ─── B. Frame-label alignment ───────────────────────────────────────────────
//
// Convert Kokoro's per-phoneme frame-duration vector into a per-10 ms-frame
// class-label track on a 16 kHz timeline of exactly `n_samples_16k` samples.
//
//   pred_dur_wrapped : Kokoro pred_dur_out — length == phoneme_ids.size()+2,
//                      entries [0]=BOS dur, [1..n]=per-phoneme durs, [n+1]=EOS.
//   phoneme_ids      : the UNWRAPPED ids passed to synthesize() (size n). Each
//                      maps to a class via cm.class_for_id; the BOS/EOS wrap
//                      tokens are treated as silence.
//   n_samples_16k    : length of the (already 16 kHz) clip these labels
//                      describe — the SAME buffer stored as PCM.
//   win_length, hop_length : the mel framing params (400, 160). Output frame
//                      count == 1 + (n_samples_16k - win_length)/hop_length
//                      (floor; 0 if n_samples_16k < win_length).
//
// Boundary / rounding policy (monotone, total, edge-silent):
//   * samples_per_kdur = n_samples_16k / sum(pred_dur_wrapped) (double).
//   * Wrapped token i occupies the half-open 16 kHz sample interval
//     [boundary_i, boundary_{i+1}), where boundary_0 = 0 and
//     boundary_{i+1} = lround(cumdur_{<=i} * samples_per_kdur). lround is
//     round-half-away-from-zero; cumdur is exact (int64). The final boundary
//     equals n_samples_16k by construction. Zero-duration tokens get a
//     zero-width interval and are skipped.
//   * Output frame t has center sample frame_center = t*hop + win_length/2.
//     Its label is the class of the token whose interval contains frame_center
//     (BOS/EOS -> silence). Because frame_center is always in (0, n_samples_16k)
//     for a valid framing, every frame lands in some token; the edges fall in
//     the BOS/EOS spans and so are silent.
//   * Transparent-modifier merge: a token whose id is cm.is_transparent (a
//     stress/length/etc. diacritic) does NOT silence its frames — they inherit
//     the class of the nearest adjacent interior NON-transparent token (scanning
//     forward first, then backward; silence only if no segmental neighbour
//     exists). This keeps stressed-vowel onsets voiced and avoids spurious
//     mid-word silence gaps. Genuine pauses (non-transparent silence-class
//     tokens, e.g. punctuation) stay silent.
//   * If sum(pred_dur_wrapped) <= 0 the whole track is silence.
std::vector<int16_t> build_frame_labels(
    const std::vector<int32_t>& pred_dur_wrapped,
    const std::vector<int32_t>& phoneme_ids,
    const PhonemeClassMap& cm,
    int n_samples_16k, int win_length, int hop_length);

// Nearest-neighbour rescale of a frame-label track from labels.size() frames to
// `new_len` frames (used by the synth tool's speed-perturb augmentation).
// Center-aligned: out[j] = labels[ clamp( floor((j+0.5)*old/new), 0, old-1 ) ].
std::vector<int16_t> resample_labels_nn(const std::vector<int16_t>& labels,
                                        int new_len);

// Centered crop / pad of a frame-label track to `target_len` frames, MIRRORING
// crop_or_pad_centered (wake_data.h) sample-for-frame so labels and PCM stay
// aligned when the synth tool centers a clip. Pads with `silence_class`.
std::vector<int16_t> crop_or_pad_labels_centered(
    const std::vector<int16_t>& labels, int target_len, int silence_class);

// ─── C. Binary frame-labelled dataset (BPDS) ────────────────────────────────
//
// Magic 'B''P''D''S' as a u32, first char in the low byte (== 0x53445042u).
// Version 1. Header (little-endian fixed fields, then the class-map blob, then
// the clip count):
//   u32 magic, u32 version, u32 sample_rate, u32 n_fft, u32 win_length,
//   u32 hop_length, u32 n_mels, <classmap blob>, u32 clip_count
// Then `clip_count` clips appended in order, each:
//   i32 n_samples, n_samples * i16 PCM (16 kHz mono, float->int16 like the WAV
//   writer: lround(clamp(x,-1,1) * 32767)), i32 n_frames, n_frames * i16 labels.
// Invariant per clip: n_frames == 1 + (n_samples - win_length)/hop_length.
constexpr std::uint32_t kMagicBPDS   = 0x53445042u;  // 'B''P''D''S'
constexpr std::uint32_t kBPDSVersion = 1u;

// Front-end framing params describing how the stored PCM is meant to be framed.
struct PhonemeDatasetHeader {
    int sample_rate = 16000;
    int n_fft       = 512;
    int win_length  = 400;
    int hop_length  = 160;
    int n_mels      = 40;
};

// Incremental writer. Opens the path, writes the header (params + class map +
// a clip-count placeholder), then streams clips via append(). The clip count is
// patched back into the header on finalize()/destruction. Mirrors Manifest.
class PhonemeDatasetWriter {
 public:
    PhonemeDatasetWriter(const std::string& path,
                         const PhonemeDatasetHeader& header,
                         const PhonemeClassMap& class_map);
    ~PhonemeDatasetWriter();

    // Append one clip: float 16 kHz PCM + its per-frame int16 labels. Converts
    // PCM to int16 and validates n_frames == framing(n_samples) AND that
    // `labels.size()` matches that frame count; throws on mismatch.
    void append(const std::vector<float>& pcm16k,
                const std::vector<int16_t>& labels);

    // Flush + patch the header clip count. Idempotent; also run by the
    // destructor. After finalize() the writer is closed.
    void finalize();

    int                clips() const { return clips_; }
    const std::string& path()  const { return path_; }

 private:
    std::string          path_;
    void*                fp_ = nullptr;   // FILE*, opaque to avoid <cstdio> uses
    PhonemeDatasetHeader header_;
    int                  clips_ = 0;
    long                 clip_count_pos_ = 0;  // file offset of the u32 count
    bool                 finalized_ = false;
};

// One decoded clip. PCM is kept as int16 (exactly as stored); pcm_float()
// rescales to [-1, 1] on demand.
struct PhonemeClip {
    std::vector<int16_t> pcm;       // 16 kHz mono
    std::vector<int16_t> labels;    // one class id per mel frame

    std::vector<float> pcm_float() const {
        std::vector<float> out(pcm.size());
        for (std::size_t i = 0; i < pcm.size(); ++i)
            out[i] = static_cast<float>(pcm[i]) / 32767.0f;
        return out;
    }
};

struct PhonemeDataset {
    PhonemeDatasetHeader     header;
    PhonemeClassMap          class_map;
    std::vector<PhonemeClip> clips;
};

// Parse + validate a BPDS file. Throws on a bad magic/version, a truncated
// stream, or any per-clip length-invariant violation.
PhonemeDataset read_phoneme_dataset(const std::string& path);

// ─── D. Dataset validation ──────────────────────────────────────────────────

struct PhonemeDatasetReport {
    int       total_clips          = 0;
    int       length_mismatch_clips = 0;   // n_frames != framing formula
    int       empty_clips          = 0;    // 0 samples or 0 frames
    int       label_out_of_range   = 0;    // a label not in [0, K)
    int       sample_rate_mismatch = 0;    // header sr != expected (0 or 1)
    long long total_frames         = 0;
    long long silence_frames       = 0;
    std::vector<long long> per_class_frames;   // size K

    // Per-clip peak/rms distribution over the decoded PCM (cheap host scan).
    int   decoded_clips = 0;
    float peak_min = 0.0f, peak_mean = 0.0f, peak_max = 0.0f, peak_std = 0.0f;
    float rms_min  = 0.0f, rms_mean  = 0.0f, rms_max  = 0.0f, rms_std  = 0.0f;
};

struct PhonemeValidationConfig {
    int       expected_sample_rate = 16000;
    long long min_frames_per_class = 1;     // per-class minimum coverage
    float     max_silence_fraction = 0.95f; // dataset-wide
};

PhonemeDatasetReport validate_phoneme_dataset(
    const PhonemeDataset& ds, const PhonemeValidationConfig& cfg);

// True if the report clears every gate. On failure fills *reason with the first
// failing check (reason may be null). The class map supplies K and is used to
// scope the per-class coverage gate to classes that can actually appear
// (silence + any class that owns ids); structurally-empty classes are skipped.
bool report_passes(const PhonemeDatasetReport& r, const PhonemeClassMap& cm,
                   const PhonemeValidationConfig& cfg, std::string* reason);

// Plain-text human summary with a stable line order (two runs diff cleanly).
void print_report(const PhonemeDatasetReport& r, const PhonemeClassMap& cm,
                  std::FILE* out);

// ─── E. Binary mel-feature cache (BPMC) ─────────────────────────────────────
//
// Precomputed front-end features: exactly what phoneme_train builds in host
// memory at startup (per-clip mel (n_mels, n_frames) freq-major + the frame
// label track), serialized so large corpora load with one sequential read
// instead of a long per-clip mel pass. Magic 'B''P''M''C' as a u32, first
// char in the low byte. Version 1:
//   u32 magic, u32 version, u32 sample_rate, u32 n_fft, u32 win_length,
//   u32 hop_length, u32 n_mels, u32 compression, <classmap blob>,
//   u32 clip_count
// Then `clip_count` clips, each:
//   i32 n_frames, n_mels*n_frames * f32 mel (freq-major: row m spans
//   [m*n_frames, (m+1)*n_frames)), n_frames * i16 labels.
// `compression` mirrors MelCompression (0 = Log, 1 = PCEN). Unlike BPDS there
// is no PCM and no framing invariant to re-derive — n_frames is authoritative.
constexpr std::uint32_t kMagicBPMC   = 0x434D5042u;  // 'B''P''M''C'
constexpr std::uint32_t kBPMCVersion = 1u;

// Incremental writer, mirroring PhonemeDatasetWriter: header up front with a
// clip-count placeholder patched on finalize(); clips streamed via append()
// so a corpus-sized cache never has to fit in memory while being built.
class PhonemeMelCacheWriter {
 public:
    PhonemeMelCacheWriter(const std::string& path,
                          const PhonemeDatasetHeader& header,
                          std::uint32_t compression,
                          const PhonemeClassMap& class_map);
    ~PhonemeMelCacheWriter();

    // Append one clip: freq-major mel of size n_mels*labels.size() + labels.
    // Throws if the mel size is not an exact n_mels multiple matching labels.
    void append(const std::vector<float>& mel,
                const std::vector<int16_t>& labels);

    void finalize();

    int                clips() const { return clips_; }
    const std::string& path()  const { return path_; }

 private:
    std::string          path_;
    void*                fp_ = nullptr;   // FILE*, opaque
    PhonemeDatasetHeader header_;
    int                  clips_ = 0;
    long                 clip_count_pos_ = 0;
    bool                 finalized_ = false;
};

struct PhonemeMelClip {
    std::vector<float>   mel;       // n_mels * n_frames, freq-major
    std::vector<int16_t> labels;    // n_frames
};

struct PhonemeMelCache {
    PhonemeDatasetHeader        header;
    std::uint32_t               compression = 1;   // MelCompression as u32
    PhonemeClassMap             class_map;
    std::vector<PhonemeMelClip> clips;
};

// Parse + validate a BPMC file. Throws on bad magic/version, truncation, or a
// mel/label size mismatch.
PhonemeMelCache read_phoneme_melcache(const std::string& path);

// Leading u32 of a file (0 if it cannot be read) — lets tools sniff BPDS vs
// BPMC entries in a mixed --dataset list.
std::uint32_t peek_dataset_magic(const std::string& path);

}  // namespace brosoundml
