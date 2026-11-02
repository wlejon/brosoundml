#include "omnivoice_prompt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace brosoundml {
namespace ovp {

// ═══════════════════════════════════════════════════════════════════════════
// Audio (pydub AudioSegment semantics, mono int16)
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// A mono 16-bit pydub AudioSegment: `s` holds the samples, `sr` the frame
// rate. Lengths are in milliseconds exactly as pydub computes them.
struct Seg {
    std::vector<int16_t> s;
    int sr = 24000;

    // len(seg): round(1000 * (frame_count / frame_rate)) — Python's
    // round-half-to-even on the double.
    long long len_ms() const {
        const double v = 1000.0 * (static_cast<double>(s.size()) / static_cast<double>(sr));
        return static_cast<long long>(std::nearbyint(v));
    }
    // frame_count(ms): ms * (frame_rate / 1000.0), truncated by int().
    long long ms_to_samples(long long ms) const {
        return static_cast<long long>(static_cast<double>(ms) * (static_cast<double>(sr) / 1000.0));
    }
    // audioop.rms: int(sqrt(sum(x^2) / n)); 0 when empty.
    long long rms() const {
        if (s.empty()) return 0;
        double sum = 0.0;
        for (int16_t v : s) sum += static_cast<double>(v) * static_cast<double>(v);
        return static_cast<long long>(std::sqrt(sum / static_cast<double>(s.size())));
    }
    // seg[a:b] in ms (both clamped to len), with pydub's zero fill of up to
    // 2 ms of missing frames — which only happens when the slice already
    // holds at least one frame (the fill is a copy of a zeroed first frame).
    Seg slice(long long a, long long b) const {
        const long long L = len_ms();
        a = std::min(a, L);
        b = std::min(b, L);
        if (a < 0) a = L - (-a);
        if (b < 0) b = L - (-b);
        const long long sa = ms_to_samples(a), sb = ms_to_samples(b);
        const long long n = static_cast<long long>(s.size());
        const long long ca = std::max(0LL, std::min(sa, n));
        const long long cb = std::max(ca, std::min(sb, n));
        Seg out;
        out.sr = sr;
        out.s.assign(s.begin() + ca, s.begin() + cb);
        const long long expected = sb - sa;
        const long long missing = expected - static_cast<long long>(out.s.size());
        if (missing > 0 && !out.s.empty()) {
            if (missing > ms_to_samples(2)) throw std::runtime_error("brosoundml: OmniVoice: pydub slice: too many missing frames");
            out.s.insert(out.s.end(), static_cast<std::size_t>(missing), int16_t{0});
        }
        return out;
    }
    Seg reversed() const {
        Seg out;
        out.sr = sr;
        out.s.assign(s.rbegin(), s.rend());
        return out;
    }
};

Seg to_seg(const std::vector<float>& x, int sr) {
    Seg seg;
    seg.sr = sr;
    seg.s.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        float v = x[i] * 32768.0f;
        if (v < -32768.0f) v = -32768.0f;
        if (v > 32767.0f) v = 32767.0f;
        seg.s[i] = static_cast<int16_t>(v);   // astype(int16): truncation toward zero
    }
    return seg;
}

std::vector<float> from_seg(const Seg& seg) {
    std::vector<float> out(seg.s.size());
    for (std::size_t i = 0; i < seg.s.size(); ++i) out[i] = static_cast<float>(seg.s[i]) / 32768.0f;
    return out;
}

// db_to_float(db) * max_possible_amplitude
double silence_rms_threshold(double db) { return std::pow(10.0, db / 20.0) * 32768.0; }

// seg.dBFS < threshold  (dBFS = -inf when rms == 0)
bool quieter_than(const Seg& seg, double db) {
    const long long r = seg.rms();
    if (r == 0) return true;
    const double dbfs = 20.0 * std::log10(static_cast<double>(r) / 32768.0);
    return dbfs < db;
}

std::vector<std::pair<long long, long long>> detect_silence(const Seg& seg, long long min_len,
                                                            double thresh_db, long long step) {
    std::vector<std::pair<long long, long long>> out;
    const long long L = seg.len_ms();
    if (L < min_len) return out;
    const double thr = silence_rms_threshold(thresh_db);
    const long long last = L - min_len;
    std::vector<long long> starts;
    for (long long i = 0; i <= last; i += step)
        if (static_cast<double>(seg.slice(i, i + min_len).rms()) <= thr) starts.push_back(i);
    if (last % step)
        if (static_cast<double>(seg.slice(last, last + min_len).rms()) <= thr) starts.push_back(last);
    if (starts.empty()) return out;
    long long prev = starts[0];
    long long cur_start = prev;
    for (std::size_t k = 1; k < starts.size(); ++k) {
        const long long i = starts[k];
        const bool continuous = (i == prev + step);
        const bool has_gap = i > (prev + min_len);
        if (!continuous && has_gap) {
            out.emplace_back(cur_start, prev + min_len);
            cur_start = i;
        }
        prev = i;
    }
    out.emplace_back(cur_start, prev + min_len);
    return out;
}

std::vector<std::pair<long long, long long>> detect_nonsilent(const Seg& seg, long long min_len,
                                                              double thresh_db, long long step) {
    const auto silent = detect_silence(seg, min_len, thresh_db, step);
    const long long L = seg.len_ms();
    std::vector<std::pair<long long, long long>> out;
    if (silent.empty()) { out.emplace_back(0, L); return out; }
    if (silent[0].first == 0 && silent[0].second == L) return out;
    long long prev_end = 0;
    long long end = 0;
    for (const auto& r : silent) {
        out.emplace_back(prev_end, r.first);
        prev_end = r.second;
        end = r.second;
    }
    if (end != L) out.emplace_back(prev_end, L);
    if (out[0].first == 0 && out[0].second == 0) out.erase(out.begin());
    return out;
}

long long floor_div(long long a, long long b) {
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

std::vector<Seg> split_on_silence(const Seg& seg, long long min_len, double thresh_db,
                                  long long keep, long long step) {
    auto ranges = detect_nonsilent(seg, min_len, thresh_db, step);
    for (auto& r : ranges) { r.first -= keep; r.second += keep; }
    for (std::size_t i = 0; i + 1 < ranges.size(); ++i) {
        const long long last_end = ranges[i].second;
        const long long next_start = ranges[i + 1].first;
        if (next_start < last_end) {
            ranges[i].second = floor_div(last_end + next_start, 2);
            ranges[i + 1].first = ranges[i].second;
        }
    }
    const long long L = seg.len_ms();
    std::vector<Seg> out;
    for (const auto& r : ranges) out.push_back(seg.slice(std::max(r.first, 0LL), std::min(r.second, L)));
    return out;
}

long long detect_leading_silence(const Seg& seg, double thresh_db, long long chunk) {
    long long trim = 0;
    const long long L = seg.len_ms();
    while (quieter_than(seg.slice(trim, trim + chunk), thresh_db) && trim < L) trim += chunk;
    return std::min(trim, L);
}

Seg remove_silence_edges(const Seg& in, long long lead, long long trail, double thresh_db) {
    long long start = detect_leading_silence(in, thresh_db, 10);
    start = std::max(0LL, start - lead);
    Seg a = in.slice(start, in.len_ms());
    Seg r = a.reversed();
    start = detect_leading_silence(r, thresh_db, 10);
    start = std::max(0LL, start - trail);
    Seg b = r.slice(start, r.len_ms());
    return b.reversed();
}

// numpy.linspace(start, stop, n, dtype=float32): float64 i*step + start with
// the endpoint forced, then a cast.
std::vector<float> linspace_f32(double start, double stop, int n) {
    std::vector<float> out(static_cast<std::size_t>(std::max(n, 0)));
    if (n <= 0) return out;
    const int div = n - 1;
    const double delta = stop - start;
    for (int i = 0; i < n; ++i) {
        double v;
        if (div > 0) {
            const double step = delta / div;
            const double prod = static_cast<double>(i) * step;
            v = prod + start;
        } else {
            v = start;   // num == 1: y = [0 * delta] + start
        }
        out[static_cast<std::size_t>(i)] = static_cast<float>(v);
    }
    if (n > 1) out[static_cast<std::size_t>(n - 1)] = static_cast<float>(stop);
    return out;
}

}  // namespace

std::vector<float> remove_silence(const std::vector<float>& x, int sr, int mid_sil,
                                  int lead_sil, int trail_sil) {
    Seg wave = to_seg(x, sr);
    if (mid_sil > 0) {
        std::vector<Seg> parts = split_on_silence(wave, mid_sil, -50.0, mid_sil, 10);
        Seg joined;
        joined.sr = sr;
        for (const Seg& p : parts) joined.s.insert(joined.s.end(), p.s.begin(), p.s.end());
        wave = std::move(joined);
    }
    wave = remove_silence_edges(wave, lead_sil, trail_sil, -50.0);
    return from_seg(wave);
}

std::vector<float> trim_long_audio(const std::vector<float>& x, int sr, double max_duration,
                                   double min_duration, double trim_threshold) {
    const double duration = static_cast<double>(x.size()) / sr;
    if (duration <= trim_threshold) return x;
    Seg seg = to_seg(x, sr);
    const auto nonsilent = detect_nonsilent(seg, 100, -40.0, 10);
    if (nonsilent.empty()) return x;
    const long long max_ms = static_cast<long long>(max_duration * 1000);
    const long long min_ms = static_cast<long long>(min_duration * 1000);
    long long best = 0;
    for (const auto& r : nonsilent) {
        if (r.first > best && r.first <= max_ms) best = r.first;
        if (r.second > max_ms) break;
    }
    if (best < min_ms) best = std::min(max_ms, seg.len_ms());
    return from_seg(seg.slice(0, best));
}

std::vector<float> fade_and_pad(const std::vector<float>& x, double pad_duration,
                                double fade_duration, int sr) {
    if (x.empty()) return x;
    const int fade_samples = static_cast<int>(fade_duration * sr);
    const int pad_samples = static_cast<int>(pad_duration * sr);
    std::vector<float> p = x;
    if (fade_samples > 0) {
        const int k = std::min<int>(fade_samples, static_cast<int>(p.size() / 2));
        if (k > 0) {
            const std::vector<float> fin = linspace_f32(0.0, 1.0, k);
            for (int i = 0; i < k; ++i) p[static_cast<std::size_t>(i)] *= fin[static_cast<std::size_t>(i)];
            const std::vector<float> fout = linspace_f32(1.0, 0.0, k);
            const std::size_t off = p.size() - static_cast<std::size_t>(k);
            for (int i = 0; i < k; ++i) p[off + static_cast<std::size_t>(i)] *= fout[static_cast<std::size_t>(i)];
        }
    }
    if (pad_samples > 0) {
        std::vector<float> out(static_cast<std::size_t>(pad_samples), 0.0f);
        out.insert(out.end(), p.begin(), p.end());
        out.insert(out.end(), static_cast<std::size_t>(pad_samples), 0.0f);
        return out;
    }
    return p;
}

std::vector<float> cross_fade_chunks(const std::vector<std::vector<float>>& chunks, int sr,
                                     double silence_duration) {
    if (chunks.empty()) return {};
    if (chunks.size() == 1) return chunks[0];
    const int total_n = static_cast<int>(silence_duration * sr);
    const int fade_n = total_n / 3;
    const int silence_n = fade_n;
    std::vector<float> merged = chunks[0];
    for (std::size_t c = 1; c < chunks.size(); ++c) {
        const int fout_n = std::min<int>(fade_n, static_cast<int>(merged.size()));
        if (fout_n > 0) {
            const std::vector<float> w = linspace_f32(1.0, 0.0, fout_n);
            const std::size_t off = merged.size() - static_cast<std::size_t>(fout_n);
            for (int i = 0; i < fout_n; ++i) merged[off + static_cast<std::size_t>(i)] *= w[static_cast<std::size_t>(i)];
        }
        merged.insert(merged.end(), static_cast<std::size_t>(std::max(silence_n, 0)), 0.0f);
        std::vector<float> fade_in = chunks[c];
        const int fin_n = std::min<int>(fade_n, static_cast<int>(fade_in.size()));
        if (fin_n > 0) {
            const std::vector<float> w = linspace_f32(0.0, 1.0, fin_n);
            for (int i = 0; i < fin_n; ++i) fade_in[static_cast<std::size_t>(i)] *= w[static_cast<std::size_t>(i)];
        }
        merged.insert(merged.end(), fade_in.begin(), fade_in.end());
    }
    return merged;
}

void gain_rms_match(std::vector<float>& x, float ref_rms) {
    for (float& v : x) {
        const float a = v * ref_rms;
        v = a / 0.1f;
    }
}

void peak_normalize(std::vector<float>& x) {
    float peak = 0.0f;
    for (float v : x) peak = std::max(peak, std::fabs(v));
    if (!(peak > 1e-6f)) return;
    for (float& v : x) {
        const float a = v / peak;
        v = a * 0.5f;
    }
}

float rms(const std::vector<float>& x) {
    if (x.empty()) return 0.0f;
    double s = 0.0;
    for (float v : x) s += static_cast<double>(v) * static_cast<double>(v);
    return static_cast<float>(std::sqrt(s / static_cast<double>(x.size())));
}

// ═══════════════════════════════════════════════════════════════════════════
// Schedule (_get_time_steps + the per-step counts of _generate_iterative)
// ═══════════════════════════════════════════════════════════════════════════

std::vector<int> unmask_schedule(int total_masked, int num_steps, float t_shift) {
    std::vector<int> out;
    if (num_steps <= 0) return out;
    const int steps = num_steps + 1;
    // torch.linspace(0, 1, steps) in FP32: the symmetric two-sided formula of
    // the CPU kernel.
    std::vector<float> t(static_cast<std::size_t>(steps));
    const float start = 0.0f, end = 1.0f;
    const float step = (end - start) / static_cast<float>(steps - 1);
    const int halfway = steps / 2;
    for (int i = 0; i < steps; ++i) {
        float v;
        if (i < halfway) v = start + step * static_cast<float>(i);
        else             v = end - step * static_cast<float>(steps - i - 1);
        t[static_cast<std::size_t>(i)] = v;
    }
    // t' = s * t / (1 + (s - 1) * t), every op FP32 with FP32 scalars.
    const float sm1 = t_shift - 1.0f;
    for (float& v : t) {
        const float num = t_shift * v;
        const float den = 1.0f + sm1 * v;
        v = num / den;
    }
    int rem = total_masked;
    for (int s = 0; s < num_steps; ++s) {
        int num;
        if (s == num_steps - 1) {
            num = rem;
        } else {
            const double diff = static_cast<double>(t[static_cast<std::size_t>(s + 1)]) -
                                static_cast<double>(t[static_cast<std::size_t>(s)]);
            const double c = std::ceil(static_cast<double>(total_masked) * diff);
            num = std::min(static_cast<int>(c), rem);
        }
        out.push_back(num);
        rem -= num;
    }
    return out;
}

}  // namespace ovp
}  // namespace brosoundml
