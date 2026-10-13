// Yue2Stream: controls, the run loop, scheduling and the program mix — see
// yue2_stream.h. A song's stages are in yue2_stream_song.cpp.

#include "yue2_stream_impl.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace brosoundml {

using namespace yue2;

namespace {

constexpr double kPi = 3.14159265358979323846;

double since(StreamClock::time_point t) { return std::chrono::duration<double>(StreamClock::now() - t).count(); }

float clampf(float v, float lo, float hi) { return std::isfinite(v) ? std::clamp(v, lo, hi) : 0.0f; }

}  // namespace

float Yue2Knobs::get(const std::string& axis) const {
    const auto it = semantic.find(axis);
    return it == semantic.end() ? 0.0f : it->second;
}

void Yue2Knobs::set(const std::string& axis, float v) {
    if (v == 0 || !std::isfinite(v)) semantic.erase(axis);
    else semantic[axis] = v;
}

Yue2Knobs Yue2Knobs::clamp(const Yue2Knobs& k) { return clamp(k, Yue2Directions::builtin_ref()); }

Yue2Knobs Yue2Knobs::clamp(const Yue2Knobs& k, const Yue2Directions& dirs) {
    Yue2Knobs o;
    o.energy = clampf(k.energy, -2.0f, 1.0f);
    o.texture = clampf(k.texture, -2.0f, 2.0f);
    o.percussion = clampf(k.percussion, -6.0f, 6.0f);
    for (const auto& [name, v] : k.semantic)
        if (const Yue2SemanticAxis* a = dirs.axis(name)) o.set(name, clampf(v, a->min, a->max));
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
    if (d.latent.size() != 64 || d.nar_l21.size() != 2048 || d.nar_l9.size() != 2048)
        throw std::runtime_error("brosoundml: Yue2Stream: directions must be 64 / 2048 / 2048 floats");
    std::set<std::string> names;
    for (const auto& ax : d.semantic) {
        if (ax.direction.size() != 2048 || ax.name.empty() || !names.insert(ax.name).second || !(ax.min <= ax.max))
            throw std::runtime_error("brosoundml: Yue2Stream: a semantic axis needs a unique name, 2048 floats and min <= max");
    }
    if (opt.ahead_songs < 1 || opt.ahead_songs > 8)
        throw std::runtime_error("brosoundml: Yue2Stream: ahead_songs must be 1..8");
    opt.crossfade_seconds = std::clamp(opt.crossfade_seconds, 2.0, 4.0);
    opt.skip_fade_seconds = std::clamp(opt.skip_fade_seconds, 0.05, 2.0);
    opt.want_grace_seconds = std::max(0.0, opt.want_grace_seconds);
    if (!(opt.preview_seconds >= 0) || (opt.preview_seconds > 0 && (opt.preview_seconds < 10 || opt.preview_seconds > 600)))
        throw std::runtime_error("brosoundml: Yue2Stream: preview_seconds must be 0 or 10..600");
    R = v.config().downsampling_ratio;
    knobs = Yue2Knobs::clamp(opt.knobs, opt.directions);
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
    c.knobs = Yue2Knobs::clamp(k, impl_->opt.directions);
    push(*impl_, std::move(c));
}
void Yue2Stream::interrupt(float amount) {
    Command c{Command::Kind::Interrupt};
    c.amount = std::isfinite(amount) ? std::clamp(amount, -1.0f, 1.0f) : 0.0f;
    push(*impl_, std::move(c));
}
void Yue2Stream::skip() { push(*impl_, Command{Command::Kind::Skip}); }
void Yue2Stream::set_next_plan(const std::string& abc) { set_plan(-1, abc); }
void Yue2Stream::set_plan(int song, const std::string& abc) {
    Command c{Command::Kind::NextPlan};
    c.index = song;
    c.abc = abc;
    push(*impl_, std::move(c));
}
void Yue2Stream::drop(int song) {
    Command c{Command::Kind::Drop};
    c.index = song;
    push(*impl_, std::move(c));
}
void Yue2Stream::make_full(int song) {
    Command c{Command::Kind::Full};
    c.index = song;
    push(*impl_, std::move(c));
}
void Yue2Stream::save_song(int song, const std::string& path) {
    Command c{Command::Kind::Save};
    c.index = song;
    c.abc = path;
    push(*impl_, std::move(c));
}
void Yue2Stream::set_auto_full(bool on) {
    Command c{Command::Kind::AutoFull};
    c.amount = on ? 1.0f : 0.0f;
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

Song* Yue2Stream::Impl::playing() {
    for (auto& sp : songs)
        if (!sp->dropped) return sp.get();
    return nullptr;
}

Song* Yue2Stream::Impl::next_song(const Song* after) {
    bool seen = false;
    for (auto& sp : songs) {
        if (seen && !sp->dropped && !sp->failed) return sp.get();
        if (sp.get() == after) seen = true;
    }
    return nullptr;
}

int Yue2Stream::Impl::pool() const {
    int live = 0;
    for (const auto& sp : songs) live += !sp->dropped && !sp->failed;
    return std::max(0, live - 1);
}

void Yue2Stream::Impl::unplace_after(const Song* s) {
    bool seen = false;
    for (auto& sp : songs) {
        Song& x = *sp;
        if (seen && x.start > emitted) {
            x.start = -1;
            x.fade_in = 0;
            x.start_announced = false;
        }
        if (&x == s) seen = true;
    }
}

bool Yue2Stream::Impl::steer_current(const Song& s) const {
    return s.nar21 == knobs.texture && s.nar9 == knobs.percussion && s.ar == knobs.semantic;
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
            // cheap and safe inside a long op) to the songs on air (from the
            // emit frontier) and the songs not built yet (from their start).
            // A song built ahead keeps its audio until it is about to play
            // (energy_before_air). The rest re-solve or continue a song,
            // which step() does.
            const Yue2Knobs k0 = knobs;
            knobs = c.knobs;
            if (knobs.energy != k0.energy) {
                for (auto& sp : songs) {
                    Song& s = *sp;
                    if (s.dropped) continue;
                    if (on_air(s)) set_offset(s, emitted_frame(s) + 1, knobs.energy, kHalo);
                    else if (!s.complete()) set_offset(s, 0, knobs.energy, 0);
                }
                Yue2StreamEvent e{Yue2StreamEvent::Kind::Knobs};
                e.text = "energy";
                e.sample = emitted;
                if (const Song* p = playing()) {
                    e.song = p->index;
                    e.tag = p->req.tag;
                    e.frame0 = emitted_frame(*p) + 1;
                }
                event(e);
            }
            if (knobs.texture != k0.texture || knobs.percussion != k0.percussion || knobs.semantic != k0.semantic)
                knobs_pending = true;
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
            plan_edits.emplace_back(c.index, c.abc);
            break;
        case Command::Kind::Drop:
            for (auto& sp : songs)
                if (sp->index == c.index && !on_air(*sp)) sp->dropped = true;
            break;
        case Command::Kind::Full:
            full_pending.push_back(c.index);
            break;
        case Command::Kind::AutoFull:
            opt.auto_full = c.amount != 0;
            break;
        case Command::Kind::Save:
            save_pending.emplace_back(c.index, c.abc);
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
    // Not on air: built ahead, the whole song waits as audio; until its
    // latents are complete, the opening (a skip, the crossfade).
    if (!on_air(s)) {
        if (s.complete()) return s.frames();
        if (s.start < 0) return lead / R + 1;
    }
    const std::int64_t at = std::max(playhead.load(), emitted) + lead - s.start;
    return at <= 0 ? lead / R + 1 : at / R + 1;
}

// Place each song after its predecessor once that one's length is known:
// the successor starts `crossfade` before the end, fading in as the other
// fades out.
void Yue2Stream::Impl::place_next() {
    Song* a = nullptr;
    for (auto& sp : songs) {
        Song& b = *sp;
        if (b.dropped || b.failed) continue;
        if (a && b.start < 0 && a->start >= 0 && a->length() >= 0) {
            const std::int64_t len = a->length();
            const std::int64_t x =
                std::min<std::int64_t>(static_cast<std::int64_t>(opt.crossfade_seconds * 48000.0), len / 3);
            b.start = std::max(emitted, a->start + len - x);
            b.fade_in = a->start + len - b.start;
        }
        a = &b;
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
            if (s.failed || s.dropped || s.start < 0) continue;
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
            if (end >= 0 && !next_song(&s) && opt.radio) {
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
            if (s.failed || s.dropped || s.start < 0 || s.start >= limit) continue;
            const std::int64_t end = s.end();
            if (end >= 0 && end <= emitted) continue;
            const Song* nx = next_song(&s);
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
        if (s.dropped) break;    // drop_marked() erases it (with its event)
        if (!s.failed && (end < 0 || end > emitted)) break;
        if (!s.failed) {
            Yue2StreamEvent e{Yue2StreamEvent::Kind::SongEnd};
            e.song = s.index;
            e.tag = s.req.tag;
            e.sample = end;
            event(e);
            // Its audio stays a while, for save_song().
            if (opt.keep_played > 0) {
                const std::size_t n = std::min<std::size_t>(s.L.size(), static_cast<std::size_t>(std::max<std::int64_t>(0, end - s.start)));
                Played p{s.index, s.req.tag, std::move(s.L), std::move(s.R),
                         make_take(s, static_cast<int>((n + R - 1) / R))};
                p.L.resize(n);
                p.R.resize(n);
                played.push_back(std::move(p));
                while (static_cast<int>(played.size()) > opt.keep_played) played.pop_front();
            }
        }
        songs.pop_front();
    }
}

void Yue2Stream::Impl::work(const char* what, const yue2::Song* s) {
    doing = what;
    doing_song = s ? s->index : -1;
}

void Yue2Stream::Impl::service() {
    if (in_service) return;
    in_service = true;
    {
        const auto now = StreamClock::now();
        if (service_end.time_since_epoch().count() != 0) {
            const double gap = std::chrono::duration<double>(now - service_end).count();
            if (gap >= 0.2) {
                Yue2StreamStats::Gap g;
                g.seconds = gap;
                g.what = doing;
                g.song = doing_song;
                g.emitted = emitted;
                g.lead = service_end_lead;
                gaps.push_back(std::move(g));
                while (gaps.size() > 16) gaps.pop_front();
                worst_gap = std::max(worst_gap, gap);            }
        }
    }
    if (opt.cancel && opt.cancel()) stopping.store(true);
    drain_commands();
    // A skip to a built song is a placement only: it happens here, inside
    // whatever long op is running (building another song, or re-solving the
    // playing one for a knob, whose work past the cut is then never heard),
    // rather than after it.
    if (skip_pending) {
        Song* s = playing();
        Song* nx = s ? next_song(s) : nullptr;
        if (s && nx && nx != busy && s->start >= 0 && nx->built()) do_skip();
    }
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
    service_end = StreamClock::now();
    service_end_lead = opt.paced ? static_cast<double>(emitted - playhead.load()) / 48000.0 : 0.0;
    in_service = false;
}

CancelCheck Yue2Stream::Impl::servicing() {
    return [this]() {
        service();
        return stopping.load();
    };
}

CancelCheck Yue2Stream::Impl::preemptible(bool whole) {
    return [this, whole]() {
        service();
        return stopping.load() || int_stage2_pending || knobs_pending || (whole && skip_pending) ||
               (busy && busy->dropped) || !full_pending.empty();
    };
}

// The radio: n songs from the last request, each with the next seed.
void Yue2Stream::Impl::radio_refill(int n) {
    if (!opt.radio || !have_last) return;
    for (int i = 0; i < n; ++i) {
        Yue2Song r = last_req;
        r.semantic.clear();
        r.params.abc.clear();
        r.params.seed = last_req.params.seed + 1;
        Command c{Command::Kind::Enqueue};
        c.song = std::move(r);
        apply(c);
    }
}

// Want events: the app is asked for songs whenever fewer than ahead_songs
// wait after the playing one (again each time the shortfall grows). With
// radio on, a shortfall the app leaves for want_grace_seconds is filled from
// the last request.
void Yue2Stream::Impl::check_want() {
    int live = 0;
    for (const auto& sp : songs) live += !sp->dropped && !sp->failed;
    const int want = std::max(0, opt.ahead_songs + 1 - live);
    if (want == 0) {
        want_open = false;
        want_announced = 0;
        return;
    }
    if (!want_open) {
        want_open = true;
        want_since = StreamClock::now();
    }
    if (want > want_announced) {
        Yue2StreamEvent e{Yue2StreamEvent::Kind::Want};
        e.value = want;
        e.frames = live;
        event(e);
    }
    want_announced = want;
    if (opt.radio && have_last && since(want_since) >= opt.want_grace_seconds) {
        radio_refill(want);
        want_since = StreamClock::now();
    }
}

void Yue2Stream::Impl::announce() {
    for (auto& sp : songs) {
        Song& s = *sp;
        if (s.dropped || s.failed) continue;
        if (!s.start_announced && s.start >= 0) {
            s.start_announced = true;
            Yue2StreamEvent e{Yue2StreamEvent::Kind::SongStart};
            e.song = s.index;
            e.tag = s.req.tag;
            e.sample = s.start;
            event(e);
        }
        // Built ahead: its AR session's device memory goes (rebuilt from the
        // tokens if it is continued on air).
        if (s.complete() && !on_air(s)) release_session(s);
        if (s.built() && !s.built_announced) {
            s.built_announced = true;
            Yue2StreamEvent e{Yue2StreamEvent::Kind::Built};
            e.song = s.index;
            e.tag = s.req.tag;
            e.frames = s.frames();
            e.seconds = since(s.queued_at);
            event(e);
        }
    }
}

bool Yue2Stream::Impl::drop_marked() {
    for (auto it = songs.begin(); it != songs.end(); ++it) {
        Song& s = **it;
        if (!s.dropped || &s == busy) continue;
        unplace_after(&s);
        Yue2StreamEvent e{Yue2StreamEvent::Kind::Dropped};
        e.song = s.index;
        e.tag = s.req.tag;
        event(e);
        songs.erase(it);
        place_next();
        return true;
    }
    return false;
}

Song* Yue2Stream::Impl::to_build() {
    const Song* p = playing();
    int pos = 0;
    for (auto& sp : songs) {
        Song& s = *sp;
        if (s.dropped || s.failed) continue;
        if (pos > opt.ahead_songs) break;
        ++pos;
        if (&s != p && !s.built()) return &s;
    }
    return nullptr;
}

void Yue2Stream::Impl::retune(Song& s, bool air, bool* changed) {
    if (s.dropped || s.failed || steer_current(s)) return;
    if (!air && s.complete()) return;   // built ahead: when it plays
    const bool nar = s.nar21 != knobs.texture || s.nar9 != knobs.percussion;
    const bool ar = s.ar != knobs.semantic;
    int from = air ? INT32_MAX : 0;
    if (nar) {
        set_nar_steer(s, knobs.texture, knobs.percussion);
        const int f = air ? emitted_frame(s) + static_cast<int>(opt.knob_latency_seconds * 25) : 0;
        resolve_from(s, f);
        from = std::min(from, std::min(f, s.lat_valid));
    }
    if (ar) {
        const double lat = int_latency_est > 0 ? int_latency_est : opt.interrupt_latency_seconds;
        const int c = air ? emitted_frame(s) + static_cast<int>(std::ceil(lat * 25)) : 0;
        const int cut = continue_from(s, c, knobs.semantic);
        if (cut >= 0) from = std::min(from, cut);
    }
    if (changed) *changed = true;
    Yue2StreamEvent e{Yue2StreamEvent::Kind::Knobs};
    e.song = s.index;
    e.tag = s.req.tag;
    e.text = std::string(nar ? "texture/percussion" : "") + (nar && ar ? "/" : "") + (ar ? "semantic" : "");
    e.sample = emitted;
    e.frame0 = from == INT32_MAX ? -1 : from;
    event(e);
}

void Yue2Stream::Impl::energy_before_air(Song& s) {
    if (s.dropped || s.failed || on_air(s) || !s.complete() || s.off_tail == knobs.energy) return;
    set_offset(s, 0, knobs.energy, 0);
}

std::shared_ptr<const Yue2StreamTake> Yue2Stream::Impl::make_take(const Song& s, int frames) const {
    auto t = std::make_shared<Yue2StreamTake>();
    t->plan = s.plan;
    t->params = s.req.params;
    const int D = Yue2::kLatentDim;
    const int lat = std::min(s.lat_valid, static_cast<int>(s.lat.size() / D));
    const int n = std::max(0, std::min({frames, static_cast<int>(s.codec.size()), lat}));
    t->semantic.assign(s.codec.begin(), s.codec.begin() + n);
    t->latents.assign(s.lat.begin(), s.lat.begin() + static_cast<std::size_t>(n) * D);
    t->energy.resize(n);
    for (int f = 0; f < n; ++f) t->energy[f] = f < static_cast<int>(s.off.size()) ? s.off[f] : s.off_tail;
    return t;
}

void Yue2Stream::Impl::save_song(int index, const std::string& path) {
    Yue2StreamEvent e{Yue2StreamEvent::Kind::Saved};
    e.song = index;
    e.value = -1;
    const std::vector<float>* L = nullptr;
    const std::vector<float>* Rc = nullptr;
    std::size_t n = 0;
    bool whole = false;
    for (auto& sp : songs) {
        const Song& s = *sp;
        if (s.index != index || s.dropped || s.failed) continue;
        e.tag = s.req.tag;
        // What is decoded under the current latents, to its length once known.
        n = std::min<std::size_t>(s.L.size(), static_cast<std::size_t>(s.dec_frame) * R);
        if (s.length() >= 0) n = std::min<std::size_t>(n, static_cast<std::size_t>(s.length()));
        whole = s.limit < 0 && s.built();
        L = &s.L;
        Rc = &s.R;
        e.take = make_take(s, static_cast<int>((n + R - 1) / R));
    }
    for (const auto& p : played)
        if (p.index == index) {
            e.tag = p.tag;
            n = p.L.size();
            whole = true;   // as it played (a preview ends at its preview)
            L = &p.L;
            Rc = &p.R;
            e.take = p.take;
        }
    if (!L) {
        e.text = "song " + std::to_string(index) + " is not held (played too long ago, dropped, or unknown)";
    } else if (n == 0) {
        e.text = "song " + std::to_string(index) + " has no audio yet";
    } else {
        StereoAudio a;
        a.frames = static_cast<int>(n);
        a.samples.resize(2 * n);
        std::copy_n(L->begin(), n, a.samples.begin());
        std::copy_n(Rc->begin(), n, a.samples.begin() + static_cast<std::ptrdiff_t>(n));
        try {
            a.write_wav(path);
            e.text = path;
            e.frames = a.frames;
            e.seconds = double(n) / 48000.0;
            e.value = whole ? 1 : 0;
        } catch (const std::exception& ex) {
            e.text = ex.what();
        }
    }
    event(e);
}

bool Yue2Stream::Impl::step_saves() {
    for (auto it = save_pending.begin(); it != save_pending.end(); ++it) {
        const int index = it->first;
        Song* s = nullptr;
        for (auto& sp : songs)
            if (sp->index == index) s = sp.get();
        if (s && !s->dropped && !s->failed) {
            if (!s->complete()) continue;   // still building: its build completes it
            if (s->dec_frame < s->frames()) {
                if (decode_tile(*s, 100, true)) return true;
                continue;                   // cut by a skip: written once it has played
            }
        }
        // Complete and decoded, played (its audio held), or gone: write what there is.
        const std::string path = it->second;
        save_pending.erase(it);
        save_song(index, path);
        return true;
    }
    return false;
}

void Yue2Stream::Impl::do_skip() {
    Song* s = playing();
    if (!s) {
        skip_pending = false;
        return;
    }
    if (s->start < 0) {   // nothing on air yet: drop it
        s->dropped = true;
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
            unplace_after(s);
        }
        return;
    }
    // A built song takes the energy knob from its first sample; then wait
    // until its opening is decoded (a built song's already is), and cut over.
    energy_before_air(*nx);
    const std::int64_t x = static_cast<std::int64_t>(opt.skip_fade_seconds * 48000);
    if (static_cast<std::int64_t>(nx->dec_frame) * R < x + 48000 && !nx->built()) {
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
    unplace_after(nx);
    Yue2StreamEvent e{Yue2StreamEvent::Kind::SongStart};
    e.song = nx->index;
    e.tag = nx->req.tag;
    e.sample = nx->start;
    e.seconds = since(skip_at);
    event(e);
    skip_pending = false;
}

// One unit of work, the playing song first while it is short of the air
// margin. False when there is none.
bool Yue2Stream::Impl::step() {
    work("drop / want / skip", nullptr);
    if (drop_marked()) return true;
    check_want();
    if (skip_pending) do_skip();
    // Previews: auto_full makes the song on air whole while its preview has
    // room left for the join; make_full() does it for one song.
    if (Song* p = playing(); opt.auto_full && p && p->limit >= 0 && on_air(*p) && p->cut < 0 &&
                             emitted_frame(*p) + 25 < p->limit - kJoinFrames &&
                             std::find(full_pending.begin(), full_pending.end(), p->index) == full_pending.end())
        full_pending.push_back(p->index);
    if (!full_pending.empty()) {
        const int idx = full_pending.front();
        full_pending.erase(full_pending.begin());
        // -1 = the song heard now: the newest on air, so during a crossfade the
        // one coming in, not the one going out (playing() until it ends).
        Song* t = idx < 0 ? playing() : nullptr;
        for (auto& sp : songs) {
            if (sp->dropped || sp->failed) continue;
            if (idx >= 0 ? sp->index == idx : on_air(*sp)) t = sp.get();
        }
        if (t && t->limit >= 0) {
            Yue2StreamEvent e{Yue2StreamEvent::Kind::Full};
            e.song = t->index;
            e.tag = t->req.tag;
            Song* nx = next_song(t);
            // Too late once a skip cut it or its successor is on air.
            if (t->cut >= 0 || (nx && on_air(*nx))) {
                e.value = 0;
            } else {
                work("make_full", t);
                const int join = make_full(*t);
                e.value = join >= 0 ? 1 : 0;
                e.frame0 = std::max(0, join);
                place_next();
            }
            event(e);
        }
        return true;
    }
    if (knobs_pending) {
        // Texture / percussion: re-solve a song on air from a little past the
        // emitted audio; the semantic steers: continue it from there. Songs
        // not built yet are rebuilt under the new knobs; songs built ahead
        // keep theirs until they play (below).
        knobs_pending = false;
        work("retune (knobs)", nullptr);
        for (auto& sp : songs) retune(*sp, on_air(*sp));
        place_next();
        return true;
    }
    // A song built ahead under other knobs, now on air, catches up.
    for (auto& sp : songs) {
        Song& s = *sp;
        if (!s.dropped && !s.failed && on_air(s) && !steer_current(s)) {
            work("retune (on air)", &s);
            retune(s, true);
            place_next();
            return true;
        }
    }
    // A song built ahead about to play takes the energy knob from its start.
    const std::int64_t lead = static_cast<std::int64_t>(opt.decode_lead_seconds * 48000.0);
    work("energy before air", nullptr);
    for (auto& sp : songs)
        if (sp->start >= 0 && sp->start - emitted <= lead) energy_before_air(*sp);
    if (int_stage2_pending) {
        // Stage 2: an arrangement-steered continuation from where the audio
        // handed out ended at the request + the stage's latency (request ->
        // landed), so the cut is handed out about when its window lands; at
        // least a second past the frontier now (a long op may have run since).
        int_stage2_pending = false;
        work("interrupt stage 2", nullptr);
        Song* s = playing();
        if (s && s->planned && s->start >= 0 && s->index == int_ev_song) {
            const double lat = int_latency_est > 0 ? int_latency_est : opt.interrupt_latency_seconds;
            const int c = std::max(int_frame + static_cast<int>(std::ceil(lat * 25)), emitted_frame(*s) + 25);
            const Song* nx = next_song(s);
            const bool room = s->frames() < 0 || c + opt.windows.front() < s->frames();
            const Yue2SemanticAxis* ax = opt.directions.axis("arrangement");
            if (ax && room && !(nx && on_air(*nx))) {
                // The arrangement carries (a knob): songs not built yet take
                // it from their start, songs built ahead when they play.
                knobs.set("arrangement", std::clamp(knobs.get("arrangement") + int_amount, ax->min, ax->max));
                continue_from(*s, c, knobs.semantic);
                int_song = s->index;
                int_cut = c;
                knobs_pending = true;
            }
        }
        return true;
    }
    if (!plan_edits.empty()) {
        // An edited score: rebuild that song from the plan on (not on air).
        const auto [idx, abc] = plan_edits.front();
        plan_edits.erase(plan_edits.begin());
        work("plan edit", nullptr);
        Song* p = playing();
        Song* t = nullptr;
        if (idx < 0) t = p ? next_song(p) : nullptr;
        else
            for (auto& sp : songs)
                if (sp->index == idx && !sp->dropped) t = sp.get();
        if (t && !on_air(*t) && t != busy) {
            Yue2Song r = t->req;
            r.params.abc = abc;
            r.semantic.clear();
            const int i = t->index;
            const auto at = t->queued_at;
            start_song(*t, std::move(r));
            --next_index;   // start_song took an index; the song keeps its own
            t->index = i;
            t->queued_at = at;
            unplace_after(t);
        }
        return true;
    }

    // Deadline-ordered decode: whichever song's next undecoded audio plays first.
    Song* urgent = nullptr;
    std::int64_t best = INT64_MAX;
    for (auto& sp : songs) {
        Song& s = *sp;
        if (s.failed || s.dropped || s.dec_frame >= decode_target(s)) continue;
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
    if (urgent && (urgent == p || best <= emitted + lead)) {
        work("decode (due)", urgent);
        if (decode_tile(*urgent, 50)) return true;
    }
    // A save whose song is whole: decode the rest of it, then write it.
    work("save", nullptr);
    if (step_saves()) return true;
    // Then build: the playing song, then (the next first while a skip waits
    // for it) the first song after it, within ahead_songs, not built yet.
    // One song at a time, so device memory holds one build beside the
    // playing song.
    auto build = [&](Song& s) -> bool {
        if (s.failed || s.dropped) return false;
        try {
            if (!s.planned) {
                work("plan", &s);
                plan(s);
                return true;
            }
            int a = 0, b = 0;
            if (window_ready(s, &a, &b)) {
                work(&s == p ? "window (on air)" : "window", &s);
                solve_window(s, a, b);
                return true;
            }
            if (!s.sem_done) {
                work(&s == p ? "semantic (on air)" : "semantic", &s);
                semantic_step(s, &s == p ? 25 : 50);
                if (s.sem_done) place_next();
                return true;
            }
            // Built ahead: the whole song decoded, so it waits as audio.
            work("decode ahead", &s);
            if (&s != p && s.complete() && !s.built() && !on_air(s)) return decode_tile(s, 100);
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
            } else {
                unplace_after(&s);
            }
            return true;
        }
        return false;
    };
    if (p) {
        if (p->start < 0) {
            p->start = emitted;   // the first song (or one after a gap) goes on air as soon as it has audio
            p->fade_in = 0;
        }
        announce();
        Song* nx = next_song(p);
        if (skip_pending && p->start >= 0 && nx && build(*nx)) return true;
        // Far enough ahead of the playhead, the songs ahead come before the
        // rest of the playing one, so a whole song does not starve them.
        const int margin = static_cast<int>(opt.air_margin_seconds * 25);
        const bool covered = margin > 0 && p->start >= 0 && p->planned && !p->failed &&
                             p->lat_valid >= emitted_frame(*p) + margin;
        if (covered)
            if (Song* t = to_build(); t && build(*t)) return true;
        if (build(*p)) return true;
        if (Song* t = to_build(); t && build(*t)) return true;
    }
    announce();
    work("decode", urgent);
    if (urgent && decode_tile(*urgent, 50)) return true;
    work("idle", nullptr);
    return false;
}

void Yue2Stream::Impl::publish() {
    Yue2StreamStats st;
    st.emitted = emitted;
    st.knobs = knobs;
    st.first_audio_seconds = first_audio_s;
    st.interrupt1_seconds = int1_s;
    st.interrupt2_seconds = int2_s;
    st.gaps.assign(gaps.begin(), gaps.end());
    st.worst_gap_seconds = worst_gap;
    const Song* p = playing();
    const std::int64_t opening = static_cast<std::int64_t>(opt.skip_fade_seconds * 48000) + 48000;
    for (auto& sp : songs) {
        const Song& s = *sp;
        if (s.dropped) continue;
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
        o.built = s.built();
        o.ready = s.complete() && (s.built() || static_cast<std::int64_t>(s.dec_frame) * R >= opening);
        o.on_air = s.start >= 0 && s.start <= emitted;
        o.host_bytes = (s.L.capacity() + s.R.capacity() + s.lat.capacity() + s.noise.capacity() + s.off.capacity()) *
                           sizeof(float) +
                       (s.codec.capacity() + s.tokens.capacity()) * sizeof(int32_t);
        o.knobs.energy = s.off_tail;
        o.knobs.texture = s.nar21;
        o.knobs.percussion = s.nar9;
        o.knobs.semantic = s.ar;
        o.preview = s.limit >= 0;
        o.preview_frames = std::max(0, s.limit);
        o.estimate_frames = s.est_frames;
        o.ended = s.ended;
        if (&s != p && !s.failed) {
            ++st.ahead;
            st.ready_ahead += o.ready;
        }
        st.host_bytes += o.host_bytes;
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
        I.work("place", nullptr);
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
