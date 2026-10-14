// Yue2Endless: controls, the run loop, scheduling and stats — see
// yue2_endless.h. Chunks are made in yue2_endless_chunk.cpp, the program in
// yue2_endless_program.cpp.

#include "yue2_endless_impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace brosoundml {

using namespace endless;

namespace {

void need(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("brosoundml: Yue2Endless: ") + what);
}

}  // namespace

Yue2Endless::Impl::Impl(const Yue2& m, const Yue2Vae& v, Yue2EndlessOptions o, AudioFn a, EventFn e)
    : model(m), vae(v), opt(std::move(o)), on_audio(std::move(a)), on_event(std::move(e)) {
    need(opt.chunk_frames >= 25 && opt.chunk_frames <= 1000, "chunk_frames must be 25..1000");
    need(opt.memory >= 0 && opt.memory <= 4000, "memory must be 0..4000 tokens");
    need(opt.change_memory >= 0 && opt.change_memory <= opt.memory, "change_memory must be 0..memory");
    need(opt.lookahead >= 0 && opt.lookahead < opt.chunk_frames, "lookahead must be 0..chunk_frames - 1");
    need(opt.nar_left >= 0 && opt.nar_left + opt.chunk_frames + opt.lookahead <= 1000,
         "nar_left + chunk_frames + lookahead must fit a 1000-frame window");
    need(opt.ode_steps >= 1, "ode_steps must be >= 1");
    need(opt.bridge_candidates >= 1 && opt.bridge_candidates <= 32, "bridge_candidates must be 1..32");
    need(opt.max_chunks >= 0, "max_chunks must be >= 0");
    need(opt.paced || opt.max_chunks > 0, "an unpaced stream needs max_chunks");
    need(opt.max_branches >= 0 && opt.max_branches <= 3, "max_branches must be 0..3");
    const auto& d = opt.directions;
    need(d.latent.size() == 64 && d.nar_l21.size() == 2048 && d.nar_l9.size() == 2048,
         "directions must be 64 / 2048 / 2048 floats");
    opt.branch_fade_seconds = std::clamp(opt.branch_fade_seconds, 0.1, 4.0);
    opt.emit_lead_seconds = std::clamp(opt.emit_lead_seconds, 0.05, 10.0);
    opt.decode_lead_seconds = std::max(opt.decode_lead_seconds, opt.emit_lead_seconds + 0.5);
    C = opt.chunk_frames;
    main = std::make_unique<Line>();
    main->seed = opt.seed;
    main->base = opt.target;
    main->base.at = -1;
    main->base.knobs = Yue2Knobs::clamp(main->base.knobs, opt.directions);
    if (!opt.path.empty()) {   // placed by the worker (its Planned events on run()'s thread)
        Command c{Command::Kind::Path};
        c.path = opt.path;
        commands.push_back(std::move(c));
    }
    if (main->base.knobs.energy != 0) energy[0] = main->base.knobs.energy;
}

Yue2Endless::Yue2Endless(const Yue2& model, const Yue2Vae& vae, Yue2EndlessOptions options, AudioFn on_audio,
                         EventFn on_event) {
    if (!model.loaded()) throw std::runtime_error("brosoundml: Yue2Endless: the model is not loaded");
    if (!vae.loaded()) throw std::runtime_error("brosoundml: Yue2Endless: the VAE is not loaded");
    impl_ = std::make_unique<Impl>(model, vae, std::move(options), std::move(on_audio), std::move(on_event));
}

Yue2Endless::~Yue2Endless() = default;

const Yue2EndlessOptions& Yue2Endless::options() const { return impl_->opt; }

// ─── controls (any thread) ──────────────────────────────────────────────────

namespace {
void push(Yue2Endless::Impl& I, Command c) {
    {
        std::lock_guard<std::mutex> lk(I.mu);
        I.commands.push_back(std::move(c));
    }
    I.cv.notify_all();
}
}  // namespace

void Yue2Endless::set_playhead(std::int64_t s) {
    impl_->playhead.store(s);
    impl_->cv.notify_all();
}
void Yue2Endless::set_target(const Yue2EndlessTarget& t) {
    Command c{Command::Kind::Target};
    c.target = t;
    push(*impl_, std::move(c));
}
void Yue2Endless::set_path(std::vector<Yue2EndlessTarget> path) {
    Command c{Command::Kind::Path};
    c.path = std::move(path);
    push(*impl_, std::move(c));
}
void Yue2Endless::set_knobs(const Yue2Knobs& k) {
    Command c{Command::Kind::Knobs};
    c.knobs = k;
    push(*impl_, std::move(c));
}
int Yue2Endless::branch(const Yue2EndlessTarget& t, std::vector<Yue2EndlessTarget> path) {
    Command c{Command::Kind::Branch};
    c.target = t;
    c.path = std::move(path);
    c.id = impl_->next_branch_id.fetch_add(1);
    const int id = c.id;
    push(*impl_, std::move(c));
    return id;
}
void Yue2Endless::commit_branch(int id) {
    Command c{Command::Kind::Commit};
    c.id = id;
    push(*impl_, std::move(c));
}
void Yue2Endless::discard_branch(int id) {
    Command c{Command::Kind::Discard};
    c.id = id;
    push(*impl_, std::move(c));
}
void Yue2Endless::stop() {
    impl_->stopping.store(true);
    impl_->cv.notify_all();
}

Yue2EndlessStats Yue2Endless::stats() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    Yue2EndlessStats s = impl_->snapshot;
    s.playhead = impl_->playhead.load();
    return s;
}

// ─── worker ─────────────────────────────────────────────────────────────────

void Yue2Endless::Impl::event(Yue2EndlessEvent e) {
    if (on_event) on_event(e);
}

void Yue2Endless::Impl::work(const char* what, int chunk) {
    const auto now = Clock::now();
    if (doing_since.time_since_epoch().count() != 0) {
        const double d = std::chrono::duration<double>(now - doing_since).count();
        auto it = std::find_if(work_s.begin(), work_s.end(), [&](const auto& w) { return std::strcmp(w.first, doing) == 0; });
        if (it == work_s.end()) work_s.emplace_back(doing, d);
        else it->second += d;
    }
    doing_since = now;
    doing = what;
    doing_chunk = chunk;
}

double Yue2Endless::Impl::seconds_until(std::int64_t sample) const {
    if (!opt.paced) return 1e30;
    return static_cast<double>(sample - std::max(playhead.load(), emitted - static_cast<std::int64_t>(opt.emit_lead_seconds * kSr))) /
           kSr;
}

void Yue2Endless::Impl::apply(Command& c) {
    switch (c.kind) {
        case Command::Kind::Target:
            place_target(*main, c.target, c.target.at < 0);
            break;
        case Command::Kind::Path: {
            const int open = open_chunk(*main);
            main->path.erase(main->path.lower_bound(open), main->path.end());
            for (Yue2EndlessTarget t : c.path) {
                if (t.at < 0) t.at = 0;
                place_target(*main, std::move(t), false);
            }
            break;
        }
        case Command::Kind::Knobs: {
            // The newest target's knobs, from the first chunk that can change.
            const Yue2Knobs k = Yue2Knobs::clamp(c.knobs, opt.directions);
            const int open = open_chunk(*main);
            Yue2EndlessTarget t = target_for(*main, open);
            const Yue2Knobs k0 = t.knobs;
            std::string moved;
            auto note = [&](bool m, const char* name) {
                if (m) moved += (moved.empty() ? "" : "/") + std::string(name);
            };
            note(k.energy != k0.energy, "energy");
            note(k.texture != k0.texture, "texture");
            note(k.percussion != k0.percussion, "percussion");
            note(k.semantic != k0.semantic, "semantic");
            t.knobs = k;
            // Later path entries keep their own knobs; the current one moves.
            main->path[open] = t;
            const int ef = static_cast<int>((emitted + kR - 1) / kR);
            set_energy(ef, k.energy);
            Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Knobs};
            e.text = moved;
            e.chunk = open;
            e.sample = static_cast<std::int64_t>(ef) * kR;
            event(e);
            break;
        }
        case Command::Kind::Branch:
            open_branch(c.id, c.target, std::move(c.path));
            break;
        case Command::Kind::Commit:
            request_commit(c.id);
            break;
        case Command::Kind::Discard:
            discard(c.id, "discarded");
            break;
    }
}

void Yue2Endless::Impl::drain_commands() {
    std::deque<Command> cmds;
    {
        std::lock_guard<std::mutex> lk(mu);
        cmds.swap(commands);
    }
    for (Command& c : cmds) apply(c);
}

// One unit of work, the most urgent first. False when there is none.
bool Yue2Endless::Impl::step() {
    Line& m = *main;
    try {
        work("commit");
        if (try_commit()) return true;
        // The program first when it is about to run dry.
        work("decode (due)");
        if (opt.paced && static_cast<std::int64_t>(dec_frame) * kR <
                             playhead.load() + static_cast<std::int64_t>((opt.emit_lead_seconds + 1.5) * kSr) &&
            decode_tile(2 * kBlock))
            return true;
        if (can_render(m)) {
            render(m);
            return true;
        }
        if (retarget_due(m)) return true;
        if (sample_due(m)) {
            sample(m);
            return true;
        }
        work("decode");
        if (decode_tile(2 * kBlock)) return true;
        if (step_branches()) return true;
    } catch (const std::exception& ex) {
        Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Error};
        e.text = ex.what();
        event(e);
        stopping.store(true);
        return false;
    }
    work("idle");
    return false;
}

void Yue2Endless::Impl::publish() {
    Yue2EndlessStats st;
    const Line& m = *main;
    st.emitted = emitted;
    st.decoded = static_cast<std::int64_t>(dec_frame) * kR;
    st.playing = static_cast<int>(std::max(playhead.load(), std::int64_t{0}) / (static_cast<std::int64_t>(C) * kR));
    st.sampled = m.sampled();
    st.rendered = m.rendered;
    st.open_chunk = open_chunk(m);
    st.committed = st.open_chunk;
    st.open_sample = static_cast<std::int64_t>(st.open_chunk) * C * kR;
    st.ahead_seconds = static_cast<double>(st.decoded - playhead.load()) / kSr;
    st.target = m.sampled() > 0 && m.has_chunk(m.sampled() - 1) ? m.chunk(m.sampled() - 1).target : m.base;
    st.path = static_cast<int>(m.path.size());
    for (int k = std::max(m.chunk_base, st.playing); k < m.sampled(); ++k) st.chunks.push_back(m.chunk(k).row);
    for (const auto& b : branches) {
        Yue2EndlessStats::Branch o;
        o.id = b->id;
        o.fork = b->fork;
        o.sampled = b->sampled();
        o.rendered = b->rendered;
        o.tag = b->base.tag;
        o.committing = b->committing;
        st.branches.push_back(o);
    }
    st.gaps.assign(gaps.begin(), gaps.end());
    st.worst_gap_seconds = worst_gap;
    for (const auto& w : work_s) st.work.push_back({w.first, w.second});
    st.splice_seconds = est_splice;
    st.render_seconds = est_render;
    st.decode_seconds = est_decode;
    st.listen_seconds = est_listen;
    std::lock_guard<std::mutex> lk(mu);
    snapshot = std::move(st);
}

void Yue2Endless::run() {
    Impl& I = *impl_;
    while (!I.stopping.load() && !I.done) {
        I.service();
        if (I.step()) {
            I.trim();
            continue;
        }
        I.service();
        if (!I.opt.paced) {
            // Unpaced, nothing waits on a playhead: the end (service emitted
            // it), or a stall.
            if (I.done || I.stopping.load()) break;
            bool pending;
            {
                std::lock_guard<std::mutex> lk(I.mu);
                pending = !I.commands.empty();
            }
            if (pending) continue;
            Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Error};
            e.text = "stream stalled";
            I.event(e);
            break;
        }
        std::unique_lock<std::mutex> lk(I.mu);
        I.cv.wait_for(lk, std::chrono::milliseconds(10), [&] { return !I.commands.empty() || I.stopping.load(); });
    }
    I.publish();
}

}  // namespace brosoundml
