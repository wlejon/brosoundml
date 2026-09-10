#pragma once

// OmniVoice prompt layer — everything between the user's strings / waveforms
// and the LM's token rows. Internal (src/ only; the test includes it to
// white-box each rule against the upstream fixtures). Ports of, in
// omnivoice/models/omnivoice.py and omnivoice/utils/{duration,voice_design,
// text,audio,lang_map}.py:
//
//   text      _combine_text, _tokenize_with_nonverbal_tags, the style-token
//             string of _prepare_inference_inputs, add_punctuation,
//             chunk_text_punctuation
//   duration  RuleDurationEstimator + _estimate_target_tokens
//   voice     _resolve_language (lang_map), _resolve_instruct (voice_design)
//   audio     remove_silence / trim_long_audio / fade_and_pad_audio /
//             cross_fade_chunks with pydub's int16 + dBFS semantics, and the
//             gain rules of _post_process_audio
//   schedule  _get_time_steps + the per-step unmask counts of _generate_iterative
//
// Strings are UTF-8; text rules that upstream applies per Python character
// run over decoded code points (brolm's unicode helpers).

#include "brosoundml/omnivoice.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::qwen { class Tokenizer; }

namespace brosoundml {
namespace ovp {

// ─── code points ─────────────────────────────────────────────────────────────
std::vector<uint32_t> to_codepoints(std::string_view s);
std::string from_codepoints(const std::vector<uint32_t>& cps);
bool is_cjk(uint32_t cp);                 // [一-鿿]
bool text_has_cjk(std::string_view s);
bool is_space(uint32_t cp);               // Python str.isspace() / re \s
std::string strip(std::string_view s);    // Python str.strip()
std::string lower(std::string_view s);    // ASCII + Latin-1/Ext-A lowercase

// ─── duration rule (RuleDurationEstimator) ───────────────────────────────────
double char_weight(uint32_t cp);
double total_weight(std::string_view text);
double estimate_duration(std::string_view target_text, std::string_view ref_text,
                         double ref_duration, double low_threshold = 50.0,
                         double boost_strength = 3.0);
// _estimate_target_tokens: `ref_text` null or empty, or `n_ref` < 0, selects
// the fixed "Nice to meet you." / 25-frame fallback. speed <= 0 or == 1 is a
// no-op. Returns max(1, int(est)).
int estimate_target_tokens(std::string_view text, const std::string* ref_text,
                           int n_ref, double speed);

// ─── language / instruct ─────────────────────────────────────────────────────
// _resolve_language: "" / "none" -> "" (language-agnostic); a known ISO id
// passes through; a known name (case-insensitive) maps to its id; anything
// else -> "" with *recognized = false.
std::string resolve_language(const std::string& language, bool* recognized = nullptr);
std::vector<std::string> language_names();   // display names, alphabetical

// _resolve_instruct: validates + normalises; "" -> "". Throws
// std::runtime_error on an unknown item, a dialect + accent mix, or two items
// of one category.
std::string resolve_instruct(const std::string& instruct, bool use_zh);
std::vector<OmniVoice::InstructCategory> instruct_categories();

// ─── text ────────────────────────────────────────────────────────────────────
std::string combine_text(const std::string& text, const std::string* ref_text);
std::string add_punctuation(const std::string& text);
// _prepare_inference_inputs' style string. `has_ref` gates <|denoise|>.
std::string style_text(const std::string& lang, const std::string& instruct,
                       bool denoise, bool has_ref);
std::vector<std::string> nonverbal_tags();
std::vector<int32_t> tokenize_with_tags(const std::string& text,
                                        const brolm::qwen::Tokenizer& tok);
// chunk_text_punctuation(text, chunk_len, min_chunk_len); min_chunk_len <= 0
// disables the short-chunk merge.
std::vector<std::string> chunk_text(const std::string& text, int chunk_len,
                                    int min_chunk_len);

// ─── audio (pydub semantics) ─────────────────────────────────────────────────
// All operate on mono FP32 at `sr`. remove_silence / trim_long_audio quantise
// through int16 exactly as pydub's AudioSegment does (truncation toward zero,
// clip to [-32768, 32767], back to float / 32768).
std::vector<float> remove_silence(const std::vector<float>& x, int sr, int mid_sil,
                                  int lead_sil, int trail_sil);
std::vector<float> trim_long_audio(const std::vector<float>& x, int sr,
                                   double max_duration = 15.0, double min_duration = 3.0,
                                   double trim_threshold = 20.0);
std::vector<float> fade_and_pad(const std::vector<float>& x, double pad_duration,
                                double fade_duration, int sr);
std::vector<float> cross_fade_chunks(const std::vector<std::vector<float>>& chunks, int sr,
                                     double silence_duration = 0.3);
// audio * ref_rms / 0.1 (two FP32 ops, numpy order).
void gain_rms_match(std::vector<float>& x, float ref_rms);
// audio / peak * 0.5 when peak > 1e-6.
void peak_normalize(std::vector<float>& x);
float rms(const std::vector<float>& x);

// ─── schedule ────────────────────────────────────────────────────────────────
// Per-step unmask counts: torch.linspace(0, 1, n+1) in FP32, warped by
// t' = s*t / (1 + (s-1)*t) in FP32, then ceil(total * (t[i+1] - t[i])) in
// FP64 with the remainder on the last step — bit-for-bit upstream's rounding.
std::vector<int> unmask_schedule(int total_masked, int num_steps, float t_shift);

}  // namespace ovp
}  // namespace brosoundml
