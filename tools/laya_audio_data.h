#pragma once

// Shared dataset plumbing for the laya-audio tools (align / cache / train /
// eval): the LibriTTS-R manifest, a windowed-sinc resampler to the 16 kHz the
// speech encoders take, transcript word normalisation, and the word-timing
// file the aligner writes and every later tool reads.
//
// Host-only, no model code. See docs/laya-audio.md.

#include <cstdint>
#include <string>
#include <vector>

namespace laya_audio {

struct Utterance {
    std::string id;
    std::string wav;
    std::string speaker;
    std::string subset;  // "dev-clean", "train-clean-100"
    std::string text;
    double duration_s = 0;
};

// Read a manifest: the LibriTTS-R manifest.json (an array of {id, wav,
// speaker, subset, text, sample_rate, duration_s}), or a .tsv with
// id \t audio \t speaker \t subset \t text per line (duration unknown).
std::vector<Utterance> load_manifest(const std::string& path);

// Windowed-sinc (Hann, 16 zero crossings) band-limited resample, polyphase
// with precomputed kernels. Host FP32. (laya_audio_decode.cpp)
std::vector<float> resample_sinc(const std::vector<float>& in, int in_rate, int out_rate);

// Audio at 16 kHz mono: .wav, .flac, .opus/.ogg (Ogg Opus), or
// "path@t0:t1" for a time range of one of those. (laya_audio_decode.cpp)
std::vector<float> load_audio_16k(const std::string& spec);

// Lower-case words of a transcript: letters (UTF-8 included; Latin-1 and
// Latin Extended-A capitals are lowered), digits and inner apostrophes
// (' or U+2019); ASCII and Latin-1 / general punctuation separate words.
std::vector<std::string> normalize_words(const std::string& text);

// Language of a subset name: "voxpopuli-de-train" / "mswc-fr-test" -> "de" /
// "fr"; everything else (LibriTTS, AMI, non-speech) -> "en".
std::string language_of(const std::string& subset);

struct TimedWord {
    std::string word;
    float t0 = -1;  // seconds from utterance start; < 0 = unknown
    float t1 = -1;
    bool asr_match = false;  // the ASR hypothesis had this exact word here
};

struct AlignedUtterance {
    std::string id;
    std::string wav;
    std::string speaker;
    std::string subset;
    float duration_s = 0;
    std::vector<TimedWord> words;  // the reference transcript's words, timed
};

// Transfer hypothesis word times onto the reference words by a Levenshtein
// alignment of the two word sequences: matched and substituted reference
// words take their hypothesis word's times, deleted ones are interpolated
// between their timed neighbours (unknown when there is no gap to put them
// in).
std::vector<TimedWord> align_reference(const std::vector<std::string>& ref,
                                       const std::vector<TimedWord>& hyp);

// Tab-separated, one utterance per line:
//   id \t wav \t speaker \t subset \t duration \t word|t0|t1|m word|t0|t1|m ...
void write_alignments(const std::string& path, const std::vector<AlignedUtterance>& utts, bool append);
std::vector<AlignedUtterance> read_alignments(const std::string& path);

// The audio a streaming listener holds at time t_end: the last window_s
// seconds of a 16 kHz stream that is silent before 0 and after the
// utterance, with a -80 dBFS dither so no frame is exactly zero.
std::vector<float> window_audio(const std::vector<float>& pcm16k, float t_end, float window_s, uint32_t seed);

// Window feature cache ('LAC1'): speech-encoder latents of audio windows,
// FP16 row-major (frames, dim), keyed by (utterance index in the alignment
// file, window end time).
struct CachedWindow {
    int utt = 0;
    float t_end = 0;
    int frames = 0;
    std::vector<uint16_t> lat;
};
struct WindowCache {
    float window_s = 0;
    int dim = 0;
    std::vector<CachedWindow> windows;
};
class CacheWriter {
public:
    CacheWriter(const std::string& path, float window_s, int dim);
    ~CacheWriter();
    void add(const CachedWindow& w);
    int count() const { return n_; }

private:
    void* f_ = nullptr;
    int n_ = 0;
};
WindowCache read_cache(const std::string& path);

}  // namespace laya_audio
