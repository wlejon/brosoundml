// YuE2 protocol + sampling arithmetic — see yue2_protocol.h.

#include "yue2_protocol.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>

namespace brosoundml::yue2 {

namespace {

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("brosoundml: Yue2: " + m); }

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

}  // namespace

const char* instruction(Yue2Cot cot) {
    switch (cot) {
        case Yue2Cot::Off:
            return "Generate music with codec tokens from the given conditions.";
        case Yue2Cot::Melody:
            return "Generate a melody-only ABC transcription without chord symbols, then generate "
                   "music with codec tokens from the given conditions.";
        case Yue2Cot::Full:
        default:
            return "Generate a chord-annotated ABC transcription, then generate music with codec "
                   "tokens from the given conditions.";
    }
}

std::string request_text(Yue2Cot cot, const std::string& style, const std::string& lyrics) {
    std::string t = instruction(cot);
    t += "\n[Tags]\n";
    t += style;
    t += "\n[Lyrics]\n";
    t += lyrics;
    t += "\n";
    return t;
}

float guidance(const Yue2Params& p) {
    if (p.cfg_scale >= 0.0f) return p.cfg_scale;
    return p.cot == Yue2Cot::Off ? 1.01f : 1.0f;
}

std::pair<int, int> logit_range(Phase ph) {
    if (ph == Phase::Abc) return {0, ABC_END + 1};
    return {MUSIC_END, CODEC_OFFSET + CODEC_SIZE};
}

int32_t end_token(Phase ph) { return ph == Phase::Abc ? ABC_END : MUSIC_END; }

std::vector<std::pair<int, int>> chunk_ranges(int frames, int prefix_tokens, int context) {
    const int size = std::min((context - prefix_tokens - 3) / 2, CONTEXT);
    if (frames < 1 || size < 1) fail("empty codec or prefix leaves no acoustic context");
    std::vector<std::pair<int, int>> r;
    for (int a = 0; a < frames; a += size) r.emplace_back(a, std::min(a + size, frames));
    return r;
}

std::vector<float> torch_cpu_randn(std::uint64_t seed, std::size_t n) {
    // at::mt19937 seeds through init_with_uint32 — the standard mt19937 init.
    std::mt19937 gen(static_cast<std::uint32_t>(seed & 0xffffffffu));
    auto uniform = [&]() -> float {
        const std::uint32_t v = gen();
        return static_cast<float>(static_cast<double>(v & 0xFFFFFFu) * (1.0 / 16777216.0));
    };
    std::vector<float> d(n);
    auto fill16 = [](float* data) {
        for (int j = 0; j < 8; ++j) {
            const float u1 = 1.0f - data[j];
            const float u2 = data[j + 8];
            const float radius = std::sqrt(-2.0f * std::log(u1));
            const float theta = static_cast<float>(2.0f * 3.14159265358979323846 * u2);
            data[j] = radius * std::cos(theta);
            data[j + 8] = radius * std::sin(theta);
        }
    };
    if (n < 16) fail("torch_cpu_randn: fewer than 16 values take another ATen path");
    for (std::size_t i = 0; i < n; ++i) d[i] = uniform();
    for (std::size_t i = 0; i + 15 < n; i += 16) fill16(d.data() + i);
    if (n % 16 != 0) {
        float* tail = d.data() + n - 16;
        for (int i = 0; i < 16; ++i) tail[i] = uniform();
        fill16(tail);
    }
    return d;
}

float bf16_round(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) {           // inf / nan pass through
        u &= 0xffff0000u;
        if ((u & 0x007f0000u) == 0 && std::isnan(x)) u |= 0x00400000u;
    } else {
        u += 0x7fffu + ((u >> 16) & 1u);
        u &= 0xffff0000u;
    }
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

void cfg_combine_bf16(const float* cond, const float* uncond, float cfg, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        const float d = bf16_round(cond[i] - uncond[i]);
        const float m = bf16_round(cfg * d);
        out[i] = bf16_round(uncond[i] + m);
    }
}

std::vector<float> distribution(const float* logits, int n, Phase ph, const Yue2Sampling& s,
                                const std::vector<int32_t>& history, int step, bool legacy_off) {
    const auto [lo, hi] = logit_range(ph);
    if (n != hi - lo) fail("distribution: logits row does not match the phase's range");
    auto R = [&](float v) { return legacy_off ? bf16_round(v) : v; };
    const int32_t end = end_token(ph);
    std::vector<float> sc(logits, logits + n);
    if (legacy_off)
        for (float& v : sc) v = bf16_round(v);
    // Allowed set: Abc = [0, EOD) + ABC_END; Semantic = codec block + MUSIC_END
    // (the whole range).
    if (ph == Phase::Abc)
        for (int id = EOD; id < hi; ++id)
            if (id != end) sc[id - lo] = kNegInf;
    if (step < s.min_tokens) sc[end - lo] = kNegInf;

    // window_penalty: alpha = penalty ** count; logit < 0 ? logit*alpha : logit/alpha.
    if (s.repetition_penalty != 1.0f && !history.empty()) {
        const std::size_t w = static_cast<std::size_t>(std::max(0, s.penalty_window));
        const std::size_t from = history.size() > w ? history.size() - w : 0;
        std::vector<std::pair<int32_t, int>> counts;
        for (std::size_t i = from; i < history.size(); ++i) {
            const int32_t t = history[i];
            auto it = std::find_if(counts.begin(), counts.end(), [&](auto& c) { return c.first == t; });
            if (it == counts.end()) counts.emplace_back(t, 1);
            else ++it->second;
        }
        for (const auto& [t, c] : counts) {
            if (t < lo || t >= hi) continue;
            const float alpha = R(static_cast<float>(std::pow(static_cast<double>(s.repetition_penalty), c)));
            float& v = sc[t - lo];
            v = v < 0 ? R(v * alpha) : R(v / alpha);
        }
    }
    if (s.temperature == 0.0f) return sc;
    if (s.temperature != 1.0f)
        for (float& v : sc) v = R(v / s.temperature);

    // top-k threshold: values strictly below the k-th largest are removed.
    // The k-th largest value through a size-k min-heap: one pass, and after
    // the first k values almost every candidate is rejected by one compare
    // (an nth_element over the 151849 ABC logits cost ~2 ms per token).
    const int k = std::min(s.top_k, n);
    std::vector<float> heap;
    heap.reserve(static_cast<std::size_t>(k));
    for (int i = 0; i < n; ++i) {
        const float v = sc[i];
        if (static_cast<int>(heap.size()) < k) {
            heap.push_back(v);
            std::push_heap(heap.begin(), heap.end(), std::greater<float>());
        } else if (v > heap.front()) {
            std::pop_heap(heap.begin(), heap.end(), std::greater<float>());
            heap.back() = v;
            std::push_heap(heap.begin(), heap.end(), std::greater<float>());
        }
    }
    const float threshold = heap.front();
    std::vector<int> kept;
    for (int i = 0; i < n; ++i) {
        if (sc[i] < threshold) sc[i] = kNegInf;
        else if (sc[i] != kNegInf) kept.push_back(i);
    }
    if (s.top_p < 1.0f && !kept.empty()) {
        // Sort the survivors descending (stable on index), softmax, drop those
        // whose preceding cumulative probability exceeds top_p — in the scores
        // dtype (FP32, or BF16 on the legacy path), as the reference computes it.
        std::stable_sort(kept.begin(), kept.end(), [&](int a, int b) { return sc[a] > sc[b]; });
        const float mx = sc[kept[0]];
        std::vector<float> p(kept.size());
        float z = 0;
        for (std::size_t j = 0; j < kept.size(); ++j) {
            p[j] = std::exp(sc[kept[j]] - mx);
            z += p[j];
        }
        const std::size_t floor_keep = legacy_off ? 3 : 1;
        // torch's cumsum accumulates in FP32 and rounds each output to the
        // scores dtype; the comparison runs in that dtype too (top_p rounded).
        const float top_p = R(s.top_p);
        float acc = 0;
        for (std::size_t j = 0; j < kept.size(); ++j) {
            const float pj = R(p[j] / z);
            acc += pj;
            const bool removed = R(R(acc) - pj) > top_p;
            if (removed && j >= floor_keep) sc[kept[j]] = kNegInf;
        }
    }
    return sc;
}

UniformFn make_uniform(std::uint64_t seed) {
    auto gen = std::make_shared<std::mt19937_64>(seed ^ 0x9E3779B97F4A7C15ULL);
    return [gen]() { return static_cast<double>((*gen)() >> 11) * (1.0 / 9007199254740992.0); };
}

int32_t draw(const std::vector<float>& scores, int lo, float temperature, const UniformFn& uniform) {
    const int n = static_cast<int>(scores.size());
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (scores[i] > scores[best]) best = i;
    if (temperature == 0.0f) return lo + best;
    const double mx = scores[best];
    double z = 0;
    for (int i = 0; i < n; ++i)
        if (scores[i] != kNegInf) z += std::exp(static_cast<double>(scores[i]) - mx);
    const double u = uniform() * z;
    double c = 0;
    int last = best;
    for (int i = 0; i < n; ++i) {
        if (scores[i] == kNegInf) continue;
        c += std::exp(static_cast<double>(scores[i]) - mx);
        last = i;
        if (u < c) return lo + i;
    }
    return lo + last;
}

std::vector<int32_t> token_prefix(const Yue2Params& p, const Encode& enc,
                                  const std::vector<int32_t>* abc_ids) {
    std::vector<int32_t> base{EOD};
    const auto t = enc(request_text(p.cot, p.style, p.lyrics));
    base.insert(base.end(), t.begin(), t.end());
    if (p.cot == Yue2Cot::Off) {
        base.insert(base.end(), {ABC_START, ABC_END, MUSIC_START});
        return base;
    }
    base.push_back(ABC_START);
    if (!abc_ids) return base;
    for (int32_t id : *abc_ids)
        if (id < 0 || id >= EOD) fail("ABC ids must stay inside the ordinary text vocabulary");
    base.insert(base.end(), abc_ids->begin(), abc_ids->end());
    base.insert(base.end(), {ABC_END, MUSIC_START});
    return base;
}

std::vector<int32_t> negative_prefix(const Yue2Params& p, const Encode& enc,
                                     const std::vector<int32_t>& abc_ids) {
    std::vector<int32_t> base{EOD};
    const auto t = enc(instruction(p.cot));
    base.insert(base.end(), t.begin(), t.end());
    if (p.cot == Yue2Cot::Off) {
        base.push_back(MUSIC_START);
        return base;
    }
    base.push_back(ABC_START);
    base.insert(base.end(), abc_ids.begin(), abc_ids.end());
    base.insert(base.end(), {ABC_END, MUSIC_START});
    return base;
}

void validate(const Yue2Sampling& s, const char* which) {
    const std::string w = which;
    if (!std::isfinite(s.temperature) || !std::isfinite(s.top_p) || !std::isfinite(s.repetition_penalty))
        fail(w + " sampling numbers must be finite");
    if (s.temperature < 0 || s.temperature > 5 || !(s.top_p > 0 && s.top_p <= 1) || s.top_k < 1)
        fail(w + " sampling: invalid temperature/top_p/top_k");
    if (s.repetition_penalty <= 0 || s.penalty_window < 1 || s.penalty_window > 100)
        fail(w + " sampling: invalid repetition penalty/window");
    if (s.min_tokens < 0 || s.min_tokens > s.max_tokens || s.max_tokens < 1)
        fail(w + " sampling: require 0 <= min_tokens <= max_tokens");
}

}  // namespace brosoundml::yue2
