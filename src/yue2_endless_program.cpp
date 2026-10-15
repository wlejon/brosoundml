// Yue2Endless: the program timeline. The main line's latents decoded just in
// time (energy added), handed out paced by the playhead, and the service the
// worker runs between and inside long operations.

#include "yue2_endless_impl.h"

#include <algorithm>
#include <cmath>

namespace brosoundml {

using namespace endless;

namespace {
constexpr int kRamp = 12;            // energy ramps over 0.5 s
constexpr std::size_t kXf = 960;     // a re-decode meets the old samples over 20 ms
}  // namespace

// ─── energy ─────────────────────────────────────────────────────────────────

float Yue2Endless::Impl::energy_at(int frame) const {
    auto it = energy.upper_bound(frame);
    if (it == energy.begin()) return 0.0f;
    const auto cur = std::prev(it);
    const float before = cur == energy.begin() ? 0.0f : std::prev(cur)->second;
    const int into = frame - cur->first;
    if (into >= kRamp - 1) return cur->second;
    const float w = static_cast<float>(into + 1) / static_cast<float>(kRamp);
    return before + (cur->second - before) * w;
}

void Yue2Endless::Impl::set_energy(int frame, float value) {
    frame = std::max(0, frame);
    energy.erase(energy.lower_bound(frame), energy.end());
    const float now = energy_at(frame);
    if (now == value && (energy.empty() || energy.rbegin()->second == value)) return;
    energy[frame] = value;
    if (frame < dec_frame) redecode_from(frame);
}

// ─── decode ─────────────────────────────────────────────────────────────────

bool Yue2Endless::Impl::stream_complete() const {
    return opt.max_chunks > 0 && main->sampled() >= opt.max_chunks && main->rendered >= opt.max_chunks;
}

int Yue2Endless::Impl::decodable() const {
    const int top = main->rendered * C;
    return stream_complete() ? top : std::max(0, top - kHalo);
}

std::int64_t Yue2Endless::Impl::natural_end() const {
    if (!stream_complete()) return -1;
    return vae.output_length(main->rendered * C);
}

int Yue2Endless::Impl::decode_target() const {
    if (!opt.paced) return INT32_MAX;
    const std::int64_t at = std::max(playhead.load(), emitted) + static_cast<std::int64_t>(opt.decode_lead_seconds * kSr);
    const int f = static_cast<int>(at / kR) + 1;
    return (f + kBlock - 1) / kBlock * kBlock;
}

void Yue2Endless::Impl::redecode_from(int frame) {
    const int f = std::max(frame, static_cast<int>((emitted + kR - 1) / kR));
    if (f >= dec_frame) return;
    dec_frame = f;
    xf_at = static_cast<std::int64_t>(f) * kR;
}

bool Yue2Endless::Impl::decode_tile(int max_frames) {
    if (in_decode) return false;
    const Line& m = *main;
    const int top = m.rendered * C;
    const int f = dec_frame;
    const int e = std::min({f + max_frames, decodable(), decode_target()});
    if (e <= f) return false;
    const bool end = stream_complete() && e == top;
    const int l0 = std::max(m.lat_base, f - kHalo), h0 = std::min(top, e + kHalo);
    std::vector<float> z(static_cast<std::size_t>(h0 - l0) * kD);
    for (int fr = l0; fr < h0; ++fr) {
        const float g = energy_at(fr);
        const float* src = m.lat_at(fr);
        float* dst = z.data() + static_cast<std::size_t>(fr - l0) * kD;
        for (int d = 0; d < kD; ++d) dst[d] = src[d] + g * opt.directions.latent[d];
    }
    in_decode = true;
    const auto t0 = Clock::now();
    const StereoAudio a = vae.decode_span(z.data(), h0 - l0, f - l0, e - l0, end && h0 == top, 0, waiting());
    in_decode = false;
    const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    est_decode = 0.9 * est_decode + 0.1 * sec * (static_cast<double>(C) / std::max(1, e - f));
    const std::int64_t at = static_cast<std::int64_t>(f) * kR;
    const std::size_t off = static_cast<std::size_t>(at - prog_base), n = static_cast<std::size_t>(a.frames);
    const std::size_t had = L.size();
    if (L.size() < off + n) {
        L.resize(off + n, 0.0f);
        R.resize(off + n, 0.0f);
    }
    const std::size_t xf = xf_at >= prog_base ? static_cast<std::size_t>(xf_at - prog_base) : SIZE_MAX;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = off + i;
        float l = a.channel(0)[i], r = a.channel(1)[i];
        if (xf != SIZE_MAX && j >= xf && j < xf + kXf && j < had) {
            const float w = (static_cast<float>(j - xf) + 0.5f) / static_cast<float>(kXf);
            l = L[j] * (1 - w) + l * w;
            r = R[j] * (1 - w) + r * w;
        }
        L[j] = l;
        R[j] = r;
    }
    if (xf != SIZE_MAX && off + n >= xf + kXf) xf_at = -1;
    // Samples past the decode frontier are old (a re-decode): kept only for its crossfade.
    dec_frame = e;
    return true;
}

// ─── emit ───────────────────────────────────────────────────────────────────

void Yue2Endless::Impl::emit() {
    const std::int64_t target = opt.paced ? playhead.load() + static_cast<std::int64_t>(opt.emit_lead_seconds * kSr)
                                          : INT64_MAX;
    std::int64_t avail = static_cast<std::int64_t>(dec_frame) * kR;
    const std::int64_t nat = natural_end();
    if (nat >= 0 && dec_frame >= main->rendered * C) avail = nat;
    while (emitted < target && emitted < avail) {
        const std::int64_t end = std::min({target, avail, emitted + kSr});
        const int n = static_cast<int>(end - emitted);
        StereoAudio chunk;
        chunk.sample_rate = kSr;
        chunk.channels = 2;
        chunk.frames = n;
        chunk.samples.resize(static_cast<std::size_t>(2) * n);
        const std::size_t off = static_cast<std::size_t>(emitted - prog_base);
        std::copy_n(L.begin() + static_cast<std::ptrdiff_t>(off), n, chunk.samples.begin());
        std::copy_n(R.begin() + static_cast<std::ptrdiff_t>(off), n, chunk.samples.begin() + n);
        const std::int64_t at = emitted;
        emitted = end;
        // Chunk starts handed out.
        const std::int64_t per = static_cast<std::int64_t>(C) * kR;
        while (static_cast<std::int64_t>(last_started + 1) * per < emitted) {
            ++last_started;
            Yue2EndlessEvent e{Yue2EndlessEvent::Kind::ChunkStart};
            e.chunk = last_started;
            e.sample = static_cast<std::int64_t>(last_started) * per;
            if (main->has_chunk(last_started)) e.tag = main->chunk(last_started).row.tag;
            event(e);
        }
        if (on_audio) on_audio(chunk, at);
        if (!opt.paced) playhead.store(emitted);
    }
    if (!opt.paced && nat >= 0 && emitted >= nat && !done) {
        done = true;
        event(Yue2EndlessEvent{Yue2EndlessEvent::Kind::Done});
    }
}

void Yue2Endless::Impl::trim() {
    // Program samples handed out are not needed again.
    const std::int64_t keep = emitted - kSr;
    if (keep - prog_base > 20 * kSr) {
        const std::size_t n = static_cast<std::size_t>(keep - prog_base);
        L.erase(L.begin(), L.begin() + static_cast<std::ptrdiff_t>(n));
        R.erase(R.begin(), R.begin() + static_cast<std::ptrdiff_t>(n));
        prog_base = keep;
    }
    // Latents: a render needs nar_left frames before the first unrendered
    // chunk, the decoder its halo before the decode frontier.
    auto trim_line = [&](Line& l, int need) {
        const int keep_f = std::max(0, need - 50);
        if (keep_f - l.lat_base > 2000) {
            l.lat.erase(l.lat.begin(), l.lat.begin() + static_cast<std::ptrdiff_t>(keep_f - l.lat_base) * kD);
            l.lat_base = keep_f;
        }
        const int keep_c = std::max(0, std::min(l.rendered, static_cast<int>(emitted / (static_cast<std::int64_t>(C) * kR))) - 4);
        while (l.chunk_base < keep_c && l.chunks.size() > 1) {
            l.chunks.pop_front();
            ++l.chunk_base;
        }
    };
    int need = std::min(dec_frame - kHalo, main->rendered * C - opt.nar_left);
    for (const auto& b : branches) need = std::min(need, b->fork * C - opt.nar_left - C);
    trim_line(*main, need);
    for (auto& b : branches) trim_line(*b, std::min(b->rendered * C - opt.nar_left, b->fork * C - opt.nar_left - C));
}

// ─── service ────────────────────────────────────────────────────────────────

void Yue2Endless::Impl::service(bool light) {
    if (in_service) return;
    in_service = true;
    const auto now = Clock::now();
    if (service_end.time_since_epoch().count() != 0) {
        const double gap = std::chrono::duration<double>(now - service_end).count();
        if (gap >= 0.2) {
            Yue2StreamStats::Gap g;
            g.seconds = gap;
            g.what = doing;
            g.song = doing_chunk;
            g.emitted = emitted;
            g.lead = service_end_lead;
            gaps.push_back(std::move(g));
            while (gaps.size() > 16) gaps.pop_front();
            worst_gap = std::max(worst_gap, gap);
        }
    }
    if (opt.cancel && opt.cancel()) stopping.store(true);
    if (!light) drain_commands();
    // A splice making a chunk whose target just changed stops, to be made again.
    if (!light && sampling && !preempt) {
        const Yue2EndlessTarget& t = target_for(*sampling, sampling_k);
        if (!t.same_music(sampling_target) || t.knobs.semantic != sampling_target.knobs.semantic || sampling->failed)
            preempt = true;
    }
    emit();
    // The program about to run dry: decode its next second now.
    if (opt.paced && static_cast<std::int64_t>(dec_frame) * kR <
                         playhead.load() + static_cast<std::int64_t>((opt.emit_lead_seconds + 0.5) * kSr))
        if (decode_tile(kBlock)) emit();
    publish();
    service_end = Clock::now();
    service_end_lead = opt.paced ? static_cast<double>(emitted - playhead.load()) / kSr : 0.0;
    in_service = false;
}

}  // namespace brosoundml
