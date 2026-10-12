#pragma once

// YuE2 protocol "yue2-native-v1" — token ids, prompt assembly, chunking, and
// the AR sampling arithmetic of the reference yue2 package (protocol.py,
// sampling.py, nar.py's noise draw). Host-only; internal to src/ (the tests
// include it to white-box the sampler against the reference on fixed logits).

#include "brosoundml/yue2.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace brosoundml::yue2 {

constexpr int32_t EOD = 151643;
constexpr int32_t ABC_START = 151847, ABC_END = 151848;
constexpr int32_t MUSIC_START = 151851, MUSIC_END = 151852;
constexpr int32_t CODEC_OFFSET = 151853, CODEC_SIZE = 32768;
constexpr int32_t LATENT_START = 184621, LATENT_END = 184622, LATENT_PAD = 184623;
constexpr int32_t VOCAB_SIZE = 184704;
constexpr int CONTEXT = 24576;

enum class Phase { Abc, Semantic };

const char* instruction(Yue2Cot cot);
// f"{INSTRUCTIONS[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"
std::string request_text(Yue2Cot cot, const std::string& style, const std::string& lyrics);

// The protocol's CFG default: 1.01 for cot = Off, 1.0 otherwise.
float guidance(const Yue2Params& p);

// Logit rows the phase can ever sample, as one contiguous lm_head row range
// [lo, hi): Abc [0, ABC_END], Semantic [MUSIC_END, CODEC_OFFSET + CODEC_SIZE).
// Rows inside the range that are not allowed (the specials between EOD and
// ABC_END) are masked by the sampler.
std::pair<int, int> logit_range(Phase ph);
int32_t end_token(Phase ph);

// protocol.chunk_ranges.
std::vector<std::pair<int, int>> chunk_ranges(int frames, int prefix_tokens, int context = CONTEXT);

// torch.randn((n / 64, 64), dtype=float32, generator=torch.Generator("cpu")
// .manual_seed(seed)): mt19937 + ATen's 16-wide Box-Muller normal_fill. The
// transcendental functions are the C library's, so a value can differ from
// ATen's vectorised (Sleef) kernel by an ulp or so; the draw order is exact.
std::vector<float> torch_cpu_randn(std::uint64_t seed, std::size_t n);

// ─── Sampling (sampling.py) ─────────────────────────────────────────────────

// Round to bfloat16 and back (round-to-nearest-even), the BF16 arithmetic
// the legacy (cot = Off) path and the CFG combine run in.
float bf16_round(float x);

// unconditional + cfg * (conditional - unconditional), each op rounded to
// BF16 the way the reference's BF16 logits tensors compute it.
void cfg_combine_bf16(const float* cond, const float* uncond, float cfg, float* out, int n);

// sampling.distribution over one row of logits restricted to `logit_range`:
// `logits` holds n = hi - lo values for ids [lo, hi). Returns the scores row
// (same indexing), -inf where removed; at temperature 0 the pre-temperature
// scores (the argmax input). `legacy_off` keeps the BF16 arithmetic and the
// 3-token top-p floor of the historical cot = Off path.
std::vector<float> distribution(const float* logits, int n, Phase ph, const Yue2Sampling& s,
                                const std::vector<int32_t>& history, int step, bool legacy_off);

// Uniform [0, 1) source the token draw consumes once per sampled token.
using UniformFn = std::function<double()>;
// Default source: a 64-bit generator seeded with `seed` (not torch's — CUDA
// multinomial is not reproducible; parity tests inject their uniforms).
UniformFn make_uniform(std::uint64_t seed);

// Draw one token id (absolute vocabulary id) from a scores row: argmax at
// temperature 0, else inverse-CDF of softmax(scores) at u = uniform().
int32_t draw(const std::vector<float>& scores, int lo, float temperature, const UniformFn& uniform);

// ─── Prompts (protocol.token_prefixes / negative_prefix) ────────────────────

using Encode = std::function<std::vector<int32_t>(const std::string&)>;

// [EOD] + text, then ABC_START (planning), or ABC_START abc ABC_END
// MUSIC_START (semantic), or ABC_START ABC_END MUSIC_START for cot = Off.
std::vector<int32_t> token_prefix(const Yue2Params& p, const Encode& enc,
                                  const std::vector<int32_t>* abc_ids);
std::vector<int32_t> negative_prefix(const Yue2Params& p, const Encode& enc,
                                     const std::vector<int32_t>& abc_ids);

// Checks protocol.Sampling's validation; throws std::runtime_error.
void validate(const Yue2Sampling& s, const char* which);

// ─── Guards (not protocol) ──────────────────────────────────────────────────

// Seconds the score plays: the "Vocal" voice's bars (every voice spans the
// song), each M: num/den whole notes at Q: n/d=bpm. 0 when the ABC has no
// M:, Q: or vocal bars.
double abc_seconds(const std::string& abc);
// semantic_sampling with Yue2Params::semantic_length_guard applied for this
// plan: max_tokens cut to the hard cap, end bias past the soft point.
Yue2Sampling guarded_semantic(const Yue2Params& p, const std::string& abc);
// Add the end bias (Yue2Sampling::end_bias_*) for `step` to row[end_index].
void apply_end_bias(float* row, int end_index, const Yue2Sampling& s, int step);
// Throws "non-finite logits" when a row holds NaN or +-inf: a steer that
// blew the residual up (B1) ends the stage with an error instead of reaching
// the sampler, where an out-of-range id would read past the logits.
void check_finite(const float* row, int n, const char* stage);

}  // namespace brosoundml::yue2
