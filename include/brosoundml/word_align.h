#pragma once

#include "brosoundml/audio.h"
#include "brosoundml/parakeet.h"

#include <brolm/tokenizer_t5.h>

#include <string>
#include <vector>

namespace brosoundml {

// ─── Word timings by forced alignment ───────────────────────────────────────
//
// Where each word of a known text is spoken in a clip: the text is split on
// whitespace, each word is tokenized on its own with Parakeet's SentencePiece
// tokenizer (so every word owns a contiguous run of pieces, its leading "▁"
// included), and the concatenated pieces are force-aligned by
// Parakeet::align. A word starts at its first piece's frame and ends when its
// last piece's predicted duration (at least one frame) runs out, clipped so
// no word runs into the next or past the end of the audio. Those frame times
// (80 ms steps for Parakeet-TDT-0.6B) are then refined on the waveform at
// 10 ms: a word that follows a pause starts at the energy onset within
// 120 ms of its frame, and a word's end follows the voiced audio (a frame
// more than 40 dB under the clip's loud level is unvoiced) until 120 ms of
// silence or the next word, trimmed back to its last voiced frame. Times are
// seconds from the start of `audio`. Words keep their text exactly as written,
// punctuation included. A word the tokenizer yields no pieces for gets a
// zero-length span where the previous word ended.
//
// This is what bro.tts attaches to a synthesis result as `words` and what
// bro.stt exposes as ParakeetModel.align(); the TTS itself has no alignment
// of its own (Qwen3-TTS streams text into the Talker one token per codec
// frame with no durations), so the speech is heard back by the recognizer.
struct WordTiming {
    std::string text;
    double      start = 0.0;
    double      end   = 0.0;
};

struct WordAlignment {
    std::vector<WordTiming> words;
    double                  log_prob = 0.0;
};

WordAlignment align_words(const Parakeet& model,
                          const brolm::t5::Tokenizer& tokenizer,
                          const AudioBuffer& audio,
                          const std::string& text);

}  // namespace brosoundml
