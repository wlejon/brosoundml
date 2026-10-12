// Yue2Stream: controls, the run loop, scheduling and the program mix — see
// yue2_stream.h. A song's stages are in yue2_stream_song.cpp.

#include "yue2_stream_impl.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace brosoundml {

using namespace yue2;

namespace {

constexpr double kPi = 3.14159265358979323846;

double since(StreamClock::time_point t) { return std::chrono::duration<double>(StreamClock::now() - t).count(); }

}  // namespace

Yue2Knobs Yue2Knobs::clamp(const Yue2Knobs& k) {
    auto c = [](float v, float lo, float hi) { return std::isfinite(v) ? std::clamp(v, lo, hi) : 0.0f; };
    Yue2Knobs o;
    o.energy = c(k.energy, -2.0f, 1.0f);
    o.texture = c(k.texture, -2.0f, 2.0f);
    o.percussion = c(k.percussion, -6.0f, 6.0f);
    o.arrangement = c(k.arrangement, -2.0f, 1.0f);
    return o;
}

Yue2Stream::Impl::Impl(const Core& c, const Yue2Vae& v, Yue2StreamOptions o, AudioFn a, EventFn e)
    : core(c), vae(v), opt(std::move(o)), on_audio(std::move(a)), on_event(std::move(e)) {
    if (opt.windows.empty()) throw std::runtime_error("brosoundml: Yue2Stream: windows must not be empty");
    for (int w : opt.windows)
        if (w < 25 || w > 2000) throw std::runtime_error("brosoundml: Yue2Stream: window sizes must be 25..2000 frames");
    if (opt.left < 0 || opt.left > 250 || opt.right < 0 || opt.right > 250)
        throw std::runtime_error("brosoundml: Yue2Stream: left / right context must be 0..250 frames");
    const auto& d = opt.directions;
    if (d.latent.size() != 64 || d.nar_l21.size() != 2048 || d.nar_l9.size() != 2048 || d.ar_l21.size() != 2048)
        throw std::runtime_error("brosoundml: Yue2Stream: directions must be 64 / 2048 / 2048 / 2048 floats");
    opt.crossfade_seconds = std::clamp(opt.crossfade_seconds, 2.0, 4.0);
    opt.skip_fade_seconds = std::clamp(opt.skip_fade_seconds, 0.05, 2.0);
    R = v.config().downsampling_ratio;
    knobs = knobs_applied = Yue2Knobs::clamp(opt.knobs);
    snapshot.knobs = knobs;
}

Yue2Stream::Yue2Stream(const Yue2& model, const Yue2Vae& vae, Yue2StreamOptions options, AudioFn on_audio,
                       EventFn on_event) {
    if (!model.loaded()) throw std::runtime_error("brosoundml: Yue2Stream: the model is not loaded");
    if (!vae.loaded()) throw std::runtime_error("brosoundml: Yue2Stream: the VAE is not loaded");
    impl_ = std::make_unique<Impl>(*model.impl_, vae, std::move(options), std::move(on_audio), std::move(on_event));
}

Yue2Stream::~Yue2Stream() = default;

const Yue2StreamOptions& Yue2Stream::options() const { return impl_->opt; }

// ─── controls (any thread) ──────────────────────────────────────────────────

namespace {
void push(Yue2Stream::Impl& I, Command c) {
    {
        std::lock_guard<std::mutex> lk(I.mu);
        I.commands.push_back(std::move(c));
    }
    I.cv.notify_all();
}
}  // namespace

void Yue2Stream::enqueue(Yue2Song song) {
    Command c{Command::Kind::Enqueue};
    c.song = std::move(song);
    push(*impl_, std::move(c));
}
void Yue2Stream::set_playhead(std::int64_t s) {
    impl_->playhead_at.store(StreamClock::now().time_since_epoch().count());
    impl_->playhead.store(s);
    impl_->cv.notify_all();
}
void Yue2Stream::set_knobs(const Yue2Knobs& k) {
    Command c{Command::Kind::Knobs};
    c.knobs = Yue2Knobs::clamp(k);
    push(*impl_, std::move(c));
}
void Yue2Stream::interrupt(float amount) {
    Command c{Command::Kind::Interrupt};
    c.amount = std::isfinite(amount) ? std::clamp(amount, -1.0f, 1.0f) : 0.0f;
    push(*impl_, std::move(c));
}
void Yue2Stream::skip() { push(*impl_, Command{Command::Kind::Skip}); }
void Yue2Stream::set_next_plan(const std::string& abc) {
    Command c{Command::Kind::NextPlan};
    c.abc = abc;
    push(*impl_, std::move(c));
}
void Yue2Stream::stop() {
    impl_->stopping.store(true);
    impl_->cv.notify_all();
}

Yue2StreamStats Yue2Stream::stats() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    Yue2StreamStats s = impl_->snapshot;
    s.playhead = impl_->playhead.load();
    return s;
}

// ─── worker ─────────────────────────────────────────────────────────────────

void Yue2Stream::Impl::event(Yue2StreamEvent e) {
    if (on_event) on_event(e);
}

Song* Yue2Stream::Impl::playing() { return songs.empty() ? nullptr : songs.front().get(); }

Song* Yue2Stream::Impl::next_song(const Song* after) {
    for (std::size_t i = 0; i + 1 < songs.size(); ++i)
        if (songs[i].get() == after) return songs[i + 1].get();
    return nullptr;
}

void Yue2Stream::Impl::apply(Command& c) {
    switch (c.kind) {
        case Command::Kind::Enqueue: {
            auto s = std::make_unique<Song>();
            start_song(*s, std::move(c.song));
            s->queued_at = c.at;
            last_req = s->req;
            have_last = true;
            idle_announced = false;
            songs.push_back(std::move(s));
            break;
        }
        case Command::Kind::Knobs: {
            // Energy acts on the latents at decode: applied at once (it is
            // cheap and safe inside a long op). The rest re-solve or continue
            // a song, which step() does.
            const float e0 = knobs.energy;
            knobs = c.knobs;
            if (knobs.energy != e0) {
                for (auto& sp : songs) {
                    Song& s = *sp;
                    const bool on_air = s.start >= 0 && s.start <= emitted;
                    set_offset(s, on_air ? emitted_frame(s) + 1 : 0, knobs.energy, on_air ? kHalo : 0);
                }
                knobs_applied.energy = knobs.energy;
                Yue2StreamEvent e{Yue2StreamEvent::Kind::Knobs};
                e.text = "energy";
                e.sample = emitted;
                event(e);
            }
            knobs_pending = !(knobs == knobs_applied);
            break;
        }
        case Command::Kind::Interrupt: {
            // Stage 1, now: the latent offset (at most +1 over the knob, 0.5 s
            // ramp) on what is not handed out yet.
            Song* s = playing();
            if (!s || s->start < 0 || c.amount == 0) break;
            int_at = c.at;
            int_amount = c.amount;
            const float target = std::clamp(knobs.energy + c.amount, -2.0f, 1.0f);
            const int f0 = emitted_frame(*s) + 1;
            set_offset(*s, f0, target, kHalo);
            int_audible1 = s->start + static_cast<std::int64_t>(f0) * R;
            int_audible2 = -1;
            int_ev_song = s->index;
            int_frame = f0;
            int_stage2_pending = true;
            break;
        }
        case Command::Kind::Skip:
            skip_pending = true;
            skip_at = c.at;
            break;
        case Command::Kind::NextPlan:
            next_plan = c.abc;
            break;
        case Command::Kind::Stop:
            stopping.store(true);
            break;
    }
}

void Yue2Stream::Impl::drain_commands() {
    std::deque<Command> cmds;
    {
        std::lock_guard<std::mutex> lk(mu);
        cmds.swap(commands);
    }
    for (Command& c : cmds) apply(c);
}

std::int64_t Yue2Stream::Impl::emit_target() const {
    if (!opt.paced) return INT64_MAX;
    return playhead.load() + static_cast<std::int64_t>(opt.emit_lead_seconds * 48000.0);
}

int Yue2Stream::Impl::emitted_frame(const Song& s) const {
    if (s.start < 0 || emitted <= s.start) return 0;
    return static_cast<int>((emitted - s.start) / R);
}

std::int64_t Yue2Stream::Impl::decode_target(const Song& s) const {
    if (!opt.paced) return INT32_MAX;
    const std::int64_t lead = static_cast<std::int64_t>(opt.decode_lead_seconds * 48000.0);
    if (s.start < 0) return lead / R + 1;   // prefetch the opening (a skip, the crossfade)
    const std::int64_t at = std::max(playhead.load(), emitted) + lead - s.start;
    return at <= 0 ? lead / R + 1 : at / R + 1;
}

// Place each song after its predecessor once that one's length is known:
// the successor starts `crossfade` before the end, fading in as the other
// fades out.
void Yue2Stream::Impl::place_next() {
    for (std::size_t i = 0; i + 1 < songs.size(); ++i) {
        Song& a = *songs[i];
        Song& b = *songs[i + 1];
        if (b.start >= 0 || a.start < 0 || a.length() < 0 || b.failed) continue;
        const std::int64_t len = a.length();
        const std::int64_t x =
            std::min<std::int64_t>(static_cast<std::int64_t>(opt.crossfade_seconds * 48000.0), len / 3);
        b.start = std::max(emitted, a.start + len - x);
        b.fade_in = a.start + len - b.start;
    }
}

// ── the program mix ─────────────────────────────────────────────────────────

void Yue2Stream::Impl::emit() {
    const std::int64_t target = emit_target();
    while (emitted < target && !songs.empty()) {
        // Songs covering the next samples, and how far each lets the mix go.
        std::int64_t limit = std::min<std::int64_t>(target, emitted + 48000);
        bool any = false;
        for (std::size_t i = 0; i < songs.size(); ++i) {
            Song& s = *songs[i];
            if (s.failed || s.start < 0) continue;
            if (s.start >= limit) {
                limit = std::max(emitted, std::min(limit, s.start));
                continue;
            }
            const std::int64_t end = s.end();
            if (end >= 0 && end <= emitted) continue;
            any = true;
            std::int64_t avail = static_cast<std::int64_t>(s.dec_frame) * R;
            if (s.sem_done && s.dec_frame >= s.frames() && s.natural >= 0) avail = s.natural;
            if (s.length() >= 0) avail = std::min(avail, s.length());
            limit = std::min(limit, s.start + avail);
            // A known end with nothing placed after it waits for the
            // successor's crossfade, unless no successor will come.
            if (end >= 0 && i + 1 == songs.size() && opt.radio) {
                const std::int64_t x = std::min<std::int64_t>(
                    static_cast<std::int64_t>(opt.crossfade_seconds * 48000.0), s.length() / 3);
                limit = std::min(limit, end - x);
            }
        }
        if (!any || limit <= emitted) break;
        const int n = static_cast<int>(limit - emitted);
        StereoAudio chunk;
        chunk.sample_rate = 48000;
        chunk.channels = 2;
        chunk.frames = n;
        chunk.samples.assign(static_cast<std::size_t>(2) * n, 0.0f);
        for (std::size_t i = 0; i < songs.size(); ++i) {
            Song& s = *songs[i];
            if (s.failed || s.start < 0 || s.start >= limit) continue;
            const std::int64_t end = s.end();
            if (end >= 0 && end <= emitted) continue;
            const Song* nx = i + 1 < songs.size() ? songs[i + 1].get() : nullptr;
            const std::int64_t out0 = nx && nx->start >= 0 && end >= 0 ? nx->start : -1;
            for (int k = 0; k < n; ++k) {
                const std::int64_t t = emitted + k;
                if (t < s.start || (end >= 0 && t >= end)) continue;
                const std::int64_t j = t - s.start;
                if (j >= static_cast<std::int64_t>(s.L.size())) continue;
                double g = 1.0;
                if (s.fade_in > 0 && t < s.start + s.fade_in)
                    g *= std::sin(0.5 * kPi * double(t - s.start) / double(s.fade_in));
                if (out0 >= 0 && t >= out0 && end > out0) g *= std::cos(0.5 * kPi * double(t - out0) / double(end - out0));
                chunk.samples[k] += static_cast<float>(g * s.L[j]);
                chunk.samples[n + k] += static_cast<float>(g * s.R[j]);
            }
            if (!s.first_audio) {
                s.first_audio = true;
                const double sec = since(s.queued_at);
                if (s.index == songs.front()->index) first_audio_s = sec;
                Yue2StreamEvent e{Yue2StreamEvent::Kind::FirstAudio};
                e.song = s.index;
                e.tag = s.req.tag;
                e.seconds = sec;
                e.sample = std::max(emitted, s.start);
                event(e);
            }
        }
        const std::int64_t at = emitted;
        emitted = limit;
        if (on_audio) on_audio(chunk, at);
        // Unpaced, the program is played as it is made.
        if (!opt.paced) playhead.store(emitted);
    }
    // Songs played out.
    while (!songs.empty()) {
        Song& s = *songs.front();
        const std::int64_t end = s.end();
        if (&s == busy) break;   // a long op is running on it: freed after
        if (!s.failed && (end < 0 || end > emitted)) break;
        if (!s.failed) {
            Yue2StreamEvent e{Yue2StreamEvent::Kind::SongEnd};
            e.song = s.index;
            e.tag = s.req.tag;
            e.sample = end;
            event(e);
        }
        songs.pop_front();
    }
}

void Yue2Stream::Impl::service() {
    if (in_service) return;
    in_service = true;
    if (opt.cancel && opt.cancel()) stopping.store(true);
    drain_commands();
    emit();
    // Interrupt stages become audible as the player passes their samples:
    // request -> the moment the reported playhead reached the sample
    // (extrapolated back from the latest report, not when this noticed).
    const std::int64_t ph = opt.paced ? playhead.load() : emitted;
    auto heard = [&](std::int64_t sample) {
        if (!opt.paced) return since(int_at);
        const StreamClock::time_point at{StreamClock::duration{playhead_at.load()}};
        const double t = std::chrono::duration<double>(at - int_at).count() - double(ph - sample) / 48000.0;
        return std::max(0.0, t);
    };
    if (int_audible1 >= 0 && ph >= int_audible1) {
        int1_s = heard(int_audible1);
        Yue2StreamEvent e{Yue2StreamEvent::Kind::Interrupt};
        e.value = 1;
        e.song = int_ev_song;
        e.sample = int_audible1;
        e.seconds = int1_s;
        event(e);
        int_audible1 = -1;
    }
    if (int_audible2 >= 0 && ph >= int_audible2) {
        int2_s = heard(int_audible2);
        Yue2StreamEvent e{Yue2StreamEvent::Kind::Interrupt};
        e.value = 2;
        e.song = int_ev_song;
        e.sample = int_audible2;
        e.seconds = int2_s;
        event(e);
        int_audible2 = -1;
    }
    // An urgent decode: the playing song's audio about to run out.
    if (Song* s = playing(); s && s->start >= 0 && opt.paced) {
        const std::int64_t need = emit_target() + 24000 - s->start;
        if (static_cast<std::int64_t>(s->dec_frame) * R < need) decode_tile(*s, 25);
    }
    publish();
    in_service = false;
}

CancelCheck Yue2Stream::Impl::servicing() {
    return [this]() {
        service();
        return stopping.load();
    };
}

CancelCheck Yue2Stream::Impl::preemptible() {
    return [this]() {
        service();
        return stopping.load() || int_stage2_pending || knobs_pending;
    };
}

void Yue2Stream::Impl::radio_refill() {
    if (!opt.radio || !have_last || songs.size() >= 2 || songs.empty()) return;
    Yue2Song r = last_req;
    r.semantic.clear();
    r.params.abc.clear();
    r.params.seed = last_req.params.seed + 1;
    Command c{Command::Kind::Enqueue};
    c.song = std::move(r);
    apply(c);
}

void Yue2Stream::Impl::do_skip() {
    Song* s = playing();
    if (!s) {
        skip_pending = false;
        return;
    }
    if (s->start < 0) {   // nothing on air yet: drop it
        songs.pop_front();
        skip_pending = false;
        return;
    }
    Song* nx = next_song(s);
    if (!nx) {
        radio_refill();
        nx = next_song(s);
        if (!nx) {   // nothing to skip to: end this one now
            s->cut = std::max<std::int64_t>(0, emitted - s->start + static_cast<std::int64_t>(opt.skip_fade_seconds * 48000));
            if (s->natural >= 0) s->cut = std::min(s->cut, s->natural);
            skip_pending = false;
        }
        return;
    }
    // Wait until the next song's opening is decoded, then cut over.
    const std::int64_t x = static_cast<std::int64_t>(opt.skip_fade_seconds * 48000);
    if (static_cast<std::int64_t>(nx->dec_frame) * R < x + 48000) {
        decode_tile(*nx, 50);   // its opening as soon as its latents exist
        return;
    }
    std::int64_t cut = emitted - s->start + x;
    if (s->natural >= 0) cut = std::min(cut, s->natural);
    s->cut = cut;
    nx->start = emitted;
    nx->fade_in = s->start + cut - emitted;
    nx->queued_at = skip_at;
    nx->first_audio = false;
    nx->start_announced = true;
    Yue2StreamEvent e{Yue2StreamEvent::Kind::SongStart};
    e.song = nx->index;
    e.tag = nx->req.tag;
    e.sample = nx->start;
    event(e);
    skip_pending = false;
}

// One unit of work, the playing song first. False when there is none.
bool Yue2Stream::Impl::step() {
    if (skip_pending) do_skip();
    if (knobs_pending) {
        // Texture / percussion: re-solve the playing song from a little past
        // the emitted audio; arrangement: continue it from there. Songs not
        // on air yet are rebuilt under the new knobs.
        const Yue2Knobs k = knobs;
        for (auto& sp : songs) {
            Song& s = *sp;
            const bool on_air = s.start >= 0 && s.start <= emitted;
            if (k.texture != knobs_applied.texture || k.percussion != knobs_applied.percussion) {
                set_nar_steer(s, k.texture, k.percussion);
                resolve_from(s, on_air ? emitted_frame(s) + static_cast<int>(opt.knob_latency_seconds * 25) : 0);
            }
            if (k.arrangement != knobs_applied.arrangement && s.planned)
                continue_from(s,
                              on_air ? emitted_frame(s) + static_cast<int>(std::ceil(
                                                              (int_latency_est > 0 ? int_latency_est
                                                                                   : opt.interrupt_latency_seconds) *
                                                              25))
                                     : 0,
                              k.arrangement);
            else if (k.arrangement != knobs_applied.arrangement)
                s.ar_scale = k.arrangement;   // applied when it is planned
        }
        Yue2StreamEvent e{Yue2StreamEvent::Kind::Knobs};
        e.text = "texture/percussion/arrangement";
        e.sample = emitted;
        event(e);
        knobs_applied = k;
        knobs_pending = false;
        place_next();
        return true;
    }
    if (int_stage2_pending) {
        // Stage 2: an arrangement-steered continuation from where the audio
        // handed out ended at the request + the stage's latency (request ->
        // landed), so the cut is handed out about when its window lands; at
        // least a second past the frontier now (a long op may have run since).
        int_stage2_pending = false;
        Song* s = playing();
        if (s && s->planned && s->start >= 0 && s->index == int_ev_song) {
            const double lat = int_latency_est > 0 ? int_latency_est : opt.interrupt_latency_seconds;
            const int c = std::max(int_frame + static_cast<int>(std::ceil(lat * 25)), emitted_frame(*s) + 25);
            const Song* nx = next_song(s);
            const bool room = s->frames() < 0 || c + opt.windows.front() < s->frames();
            if (room && !(nx && nx->start >= 0 && nx->start <= emitted)) {
                const float a = std::clamp(knobs.arrangement + int_amount, -2.0f, 1.0f);
                knobs.arrangement = knobs_applied.arrangement = a;
                continue_from(*s, c, a);
                int_song = s->index;
                int_cut = c;
            }
        }
        return true;
    }
    if (!next_plan.empty()) {
        // The next song's edited score: rebuild it from the plan on.
        Song* s = playing();
        Song* nx = s ? next_song(s) : nullptr;
        if (nx && nx->start < 0) {
            Yue2Song r = nx->req;
            r.params.abc = next_plan;
            r.semantic.clear();
            const int idx = nx->index;
            const auto at = nx->queued_at;
            start_song(*nx, std::move(r));
            nx->index = idx;
            nx->queued_at = at;
        }
        next_plan.clear();
        return true;
    }

    // Deadline-ordered decode: whichever song's next undecoded audio plays first.
    Song* urgent = nullptr;
    std::int64_t best = INT64_MAX;
    for (auto& sp : songs) {
        Song& s = *sp;
        if (s.failed || s.dec_frame >= decode_target(s)) continue;
        const std::int64_t deadline = s.start >= 0 ? s.start + static_cast<std::int64_t>(s.dec_frame) * R
                                                   : INT64_MAX - 1;
        if (deadline < best) {
            // only if it can make progress
            const int T = s.frames();
            const int limit = (T >= 0 && s.lat_extent >= T && s.lat_valid >= T) ? T : s.lat_extent - kHalo;
            if (limit > s.dec_frame) {
                best = deadline;
                urgent = &s;
            }
        }
    }
    Song* p = playing();
    // The playing song's audio first when it is due within the decode lead.
    if (urgent && (urgent == p || best <= emitted + static_cast<std::int64_t>(opt.decode_lead_seconds * 48000))) {
        if (decode_tile(*urgent, 50)) return true;
    }
    // Then build songs in order: the playing one, then the next (the next
    // first while a skip waits for its opening).
    for (std::size_t k = 0; k < songs.size() && k < 2; ++k) {
        const std::size_t i = skip_pending && songs.size() >= 2 && songs[0]->start >= 0 ? 1 - k : k;
        Song& s = *songs[i];
        if (s.failed) continue;
        if (i == 0 && s.start < 0) {
            s.start = emitted;   // the first song (or one after a gap) goes on air as soon as it has audio
            s.fade_in = 0;
        }
        if (!s.start_announced && s.start >= 0) {
            s.start_announced = true;
            Yue2StreamEvent e{Yue2StreamEvent::Kind::SongStart};
            e.song = s.index;
            e.tag = s.req.tag;
            e.sample = s.start;
            event(e);
        }
        try {
            if (!s.planned) {
                plan(s);
                return true;
            }
            int a = 0, b = 0;
            if (window_ready(s, &a, &b)) {
                solve_window(s, a, b);
                return true;
            }
            if (!s.sem_done) {
                semantic_step(s, i == 0 ? 25 : 50);
                if (s.sem_done) place_next();
                return true;
            }
        } catch (const std::exception& e) {
            s.failed = true;
            Yue2StreamEvent ev{Yue2StreamEvent::Kind::Error};
            ev.song = s.index;
            ev.tag = s.req.tag;
            ev.text = e.what();
            event(ev);
            // A failed song on air ends where its audio does.
            if (s.start >= 0) {
                s.failed = false;
                s.cut = std::min<std::int64_t>(static_cast<std::int64_t>(s.dec_frame) * R,
                                               s.natural >= 0 ? s.natural : INT64_MAX);
                s.sem_done = true;
                if (s.natural < 0) s.natural = s.cut;
                s.lat_valid = s.lat_extent = s.dec_frame;
                s.codec.resize(std::min<std::size_t>(s.codec.size(), static_cast<std::size_t>(s.dec_frame)));
                place_next();
            }
            return true;
        }
    }
    if (urgent && decode_tile(*urgent, 50)) return true;
    if (songs.size() < 2) {
        const std::size_t before = songs.size();
        radio_refill();
        if (songs.size() != before) return true;
    }
    return false;
}

void Yue2Stream::Impl::publish() {
    Yue2StreamStats st;
    st.emitted = emitted;
    st.knobs = knobs;
    st.first_audio_seconds = first_audio_s;
    st.interrupt1_seconds = int1_s;
    st.interrupt2_seconds = int2_s;
    for (auto& sp : songs) {
        const Song& s = *sp;
        Yue2StreamStats::Song o;
        o.index = s.index;
        o.tag = s.req.tag;
        o.planned = s.planned;
        o.semantic_done = s.sem_done;
        o.semantic_frames = static_cast<int>(s.codec.size());
        o.latent_frames = s.lat_valid;
        o.decoded_frames = s.dec_frame;
        o.start = s.start;
        o.end = s.end();
        o.latent_norm = s.last_norm;
        st.songs.push_back(o);
    }
    std::lock_guard<std::mutex> lk(mu);
    snapshot = std::move(st);
}

void Yue2Stream::run() {
    Impl& I = *impl_;
    int stalls = 0;
    while (!I.stopping.load()) {
        I.service();
        I.place_next();   // songs go on the program as their predecessors' lengths become known
        if (I.step()) {
            stalls = 0;
            continue;
        }
        I.service();
        bool pending;
        {
            std::lock_guard<std::mutex> lk(I.mu);
            pending = !I.commands.empty();
        }
        if (I.songs.empty() && !pending) {
            if (!I.idle_announced) {
                I.idle_announced = true;
                I.event(Yue2StreamEvent{Yue2StreamEvent::Kind::Idle});
            }
            if (!I.opt.paced) break;   // an offline render is done
        }
        if (!I.opt.paced) {
            // Unpaced, nothing waits on a playhead: no work and no commands
            // twice running is a stall (a song that can never finish).
            if (!pending && ++stalls > 2) {
                Yue2StreamEvent e{Yue2StreamEvent::Kind::Error};
                e.text = "stream stalled";
                I.event(e);
                break;
            }
            continue;
        }
        std::unique_lock<std::mutex> lk(I.mu);
        I.cv.wait_for(lk, std::chrono::milliseconds(10), [&] { return !I.commands.empty() || I.stopping.load(); });
    }
    I.publish();
}

}  // namespace brosoundml
