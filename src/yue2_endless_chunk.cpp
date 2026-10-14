// Yue2Endless: targets, plans, and making chunks (endless.js's splice per
// chunk and its NAR render), see yue2_endless.h.

#include "yue2_endless_impl.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace brosoundml {

using namespace endless;

namespace {

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

double norm_of(const Line& l, int f0, int f1) {
    double s = 0;
    for (int f = f0; f < f1; ++f) {
        const float* z = l.lat_at(f);
        double n = 0;
        for (int d = 0; d < kD; ++d) n += static_cast<double>(z[d]) * z[d];
        s += std::sqrt(n);
    }
    return f1 > f0 ? s / (f1 - f0) : 0;
}

// What a chunk's tokens depend on (the rest, NAR knobs and energy, act later).
bool same_tokens(const Yue2EndlessTarget& a, const Yue2EndlessTarget& b) {
    return a.style == b.style && a.score == b.score && a.lyrics == b.lyrics && a.roll == b.roll &&
           a.lyric_window == b.lyric_window && a.knobs.semantic == b.knobs.semantic;
}

void mean_in(double& est, double v) { est = 0.7 * est + 0.3 * v; }

}  // namespace

void Line::lat_ensure(int frames) {
    const int have = lat_base + static_cast<int>(lat.size() / kD);
    if (frames > have) lat.resize(static_cast<std::size_t>(frames - lat_base) * kD, 0.0f);
}

// ─── targets ────────────────────────────────────────────────────────────────

const Yue2EndlessTarget& Yue2Endless::Impl::target_for(const Line& l, int k) const {
    auto it = l.path.upper_bound(k);
    if (it == l.path.begin()) return l.base;
    return std::prev(it)->second;
}

void Yue2Endless::Impl::fold(Line& l, int k) {
    l.base = target_for(l, k);
    l.path.erase(l.path.begin(), l.path.upper_bound(k));
}

bool Yue2Endless::Impl::retargetable(const Line& l, int k) const {
    if (!opt.retarget || k < 0 || k != l.sampled() - 1 || k < l.rendered || !l.has_chunk(k)) return false;
    if (l.chunk(k).fresh || (l.id != 0 && k < l.fork)) return false;
    if (!opt.paced) return true;
    // Sampled again, then the next chunk, then rendered, before it plays.
    const double left = seconds_until(static_cast<std::int64_t>(k) * C * kR) - opt.emit_lead_seconds;
    return left > 2 * est_splice + est_render + 1.0;
}

int Yue2Endless::Impl::open_chunk(const Line& l) const {
    const int k = l.sampled() - 1;
    return retargetable(l, k) ? k : l.sampled();
}

void Yue2Endless::Impl::place_target(Line& l, Yue2EndlessTarget t, bool now) {
    t.knobs = Yue2Knobs::clamp(t.knobs, opt.directions);
    t.lyric_window = std::clamp(t.lyric_window, 1, 32);
    const int open = open_chunk(l);
    int k = open;
    if (!now) {
        const double frames = std::max(0.0, t.at) * Yue2::kFrameRate;
        k = std::max(open, static_cast<int>(std::ceil(frames / C - 1e-9)));
    }
    if (now) l.path.clear();
    else l.path.erase(l.path.lower_bound(k), l.path.end());
    Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Planned};
    e.chunk = k;
    e.branch = l.id;
    e.tag = t.tag;
    e.sample = static_cast<std::int64_t>(k) * C * kR;
    e.seconds = static_cast<double>(e.sample) / kSr;
    l.path[k] = std::move(t);
    // Energy is heard from the audio not yet handed out.
    if (now && l.id == 0) set_energy(static_cast<int>((emitted + kR - 1) / kR), l.path[k].knobs.energy);
    event(e);
}

// ─── plans and params ───────────────────────────────────────────────────────

const Yue2Plan& Yue2Endless::Impl::plan_for(const Yue2EndlessTarget& t, const std::string& lyrics) {
    const std::string key = t.style + '\x1f' + t.score + '\x1f' + lyrics;
    auto it = plans.find(key);
    if (it != plans.end()) return it->second;
    Yue2Params p;
    p.style = t.style;
    p.lyrics = lyrics;
    p.abc = t.score;
    p.cot = t.score.empty() ? Yue2Cot::Off : Yue2Cot::Full;
    p.seed = 1;
    work("plan");
    Yue2Plan plan = model.plan(p, {}, servicing());
    if (plans.size() > 64) plans.clear();   // rolled lyrics make a plan per window
    return plans.emplace(key, std::move(plan)).first->second;
}

Yue2Params Yue2Endless::Impl::params_for(const Yue2EndlessTarget& t, const std::string& lyrics, std::uint64_t seed,
                                         bool ar) const {
    Yue2Params p;
    p.style = t.style;
    p.lyrics = lyrics;
    p.abc = t.score;
    p.cot = t.score.empty() ? Yue2Cot::Off : Yue2Cot::Full;
    p.seed = seed;
    p.cfg_scale = opt.cfg_scale;
    p.ode_steps = opt.ode_steps;
    if (ar)
        for (const auto& [name, scale] : t.knobs.semantic) {
            const Yue2SemanticAxis* a = opt.directions.axis(name);
            if (!a || scale == 0) continue;
            Yue2Steer s;
            s.nar = false;
            s.layer = a->layer;
            s.delta = a->direction;
            s.scale = scale;
            s.plan = false;
            s.semantic = true;
            p.steer.push_back(std::move(s));
        }
    return p;
}

// Texture / percussion: NAR steers over the window's frame rows.
void Yue2Endless::Impl::nar_steer(Yue2Params& p, const Yue2Knobs& k, int frames) const {
    auto add = [&](int layer, const std::vector<float>& dir, float scale) {
        if (scale == 0) return;
        Yue2Steer s;
        s.nar = true;
        s.layer = layer;
        s.per_position = true;
        s.pos0 = 0;
        s.scale = scale;
        s.delta.resize(static_cast<std::size_t>(frames) * dir.size());
        for (int r = 0; r < frames; ++r) std::copy(dir.begin(), dir.end(), s.delta.begin() + static_cast<std::ptrdiff_t>(r * dir.size()));
        p.steer.push_back(std::move(s));
    };
    add(21, opt.directions.nar_l21, k.texture);
    add(9, opt.directions.nar_l9, k.percussion);
}

// ─── scheduling ─────────────────────────────────────────────────────────────

bool Yue2Endless::Impl::needs_bridge(const Line& l, int k, const Yue2EndlessTarget& t) const {
    if (k <= 0 || !l.has_chunk(k - 1) || t.same_music(l.chunk(k - 1).target)) return false;
    if (opt.change_mode != Yue2ChangeMode::Auto && opt.change_mode != Yue2ChangeMode::Bridge) return false;
    if (t.score.empty() || score_beat_frames(t.score) <= 0) return false;
    if (opt.max_chunks > 0 && k + 1 >= opt.max_chunks) return false;
    return target_for(l, k + 1).same_music(t);
}

bool Yue2Endless::Impl::sample_due(const Line& l) const {
    if (l.failed || finished_sampling(l)) return false;
    if (!opt.paced) return true;
    const int k = l.sampled();
    const double lead = opt.ahead_seconds + (needs_bridge(l, k, target_for(l, k)) ? opt.bridge_lead_seconds : 0.0);
    return seconds_until(static_cast<std::int64_t>(k) * C * kR) < lead;
}

bool Yue2Endless::Impl::can_render(const Line& l) const {
    const int r = l.rendered;
    if (l.failed || r >= l.sampled() || (l.id != 0 && r < l.fork)) return false;
    return static_cast<int>(l.tokens.size()) >= (r + 1) * C + opt.lookahead || finished_sampling(l);
}

CancelCheck Yue2Endless::Impl::servicing() {
    return [this] {
        service();
        return stopping.load();
    };
}

CancelCheck Yue2Endless::Impl::preemptible(Line& l, int k, const Yue2EndlessTarget& t) {
    sampling = &l;
    sampling_k = k;
    sampling_target = t;
    preempt = false;
    return [this] {
        service();
        return stopping.load() || preempt;
    };
}

void Yue2Endless::Impl::end_preemptible() {
    sampling = nullptr;
    sampling_k = -1;
    preempt = false;
}

void Yue2Endless::Impl::append(Line& l, const int32_t* t, int n) {
    l.tokens.insert(l.tokens.end(), t, t + n);
    l.lat_ensure(static_cast<int>(l.tokens.size()));
}

void Yue2Endless::Impl::add_chunk(Line& l, Chunk c) {
    const int k = c.row.index;
    if (l.id == 0 && energy_at(k * C) != c.target.knobs.energy) set_energy(k * C, c.target.knobs.energy);
    l.chunks.push_back(std::move(c));
    Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Sampled};
    const Chunk& h = l.chunk(k);
    e.chunk = k;
    e.branch = l.id;
    e.tag = h.row.tag;
    e.sample = static_cast<std::int64_t>(k) * C * kR;
    e.seconds = h.row.splice_seconds;
    e.row = std::make_shared<Yue2EndlessChunk>(h.row);
    if (opt.chunk_data)
        e.tokens = std::make_shared<std::vector<int32_t>>(l.tokens.begin() + static_cast<std::ptrdiff_t>(k) * C,
                                                          l.tokens.begin() + static_cast<std::ptrdiff_t>(k + 1) * C);
    event(e);
}

void Yue2Endless::Impl::retarget(Line& l, int k) {
    Chunk& c = l.chunk(k);
    const int heard = l.roll.last_heard;
    const bool same_sheet = c.roll_sheet == l.roll.sheet;
    if (!same_sheet) {   // the chunk had started a new sheet: back to the old one
        l.roll = Roll{};
        l.roll.sheet = c.roll_sheet;
    }
    l.roll.at = c.roll_at;
    l.roll.moved = c.roll_moved;
    if (same_sheet) l.roll.last_heard = heard;
    if (!same_sheet && !c.roll_sheet.empty()) {
        // Rebuilt by the next lyrics_of() from the sheet text.
        l.roll.lines.clear();
    }
    l.tokens.resize(static_cast<std::size_t>(k) * C);
    l.chunks.pop_back();
}

bool Yue2Endless::Impl::retarget_due(Line& l) {
    const int k = l.sampled() - 1;
    if (!retargetable(l, k)) return false;
    const Yue2EndlessTarget& t = target_for(l, k);
    Chunk& c = l.chunk(k);
    if (same_tokens(t, c.target)) {
        // Only what acts after sampling moved (NAR knobs, energy, tag).
        if (!(t.knobs == c.target.knobs) || t.tag != c.target.tag) {
            c.target.knobs = t.knobs;
            c.target.tag = t.tag;
            c.row.knobs = t.knobs;
            c.row.tag = t.tag;
        }
        return false;
    }
    work("retarget", k);
    retarget(l, k);
    return true;
}

// ─── sampling a chunk ───────────────────────────────────────────────────────

bool Yue2Endless::Impl::sample(Line& l) {
    const int k = l.sampled();
    Yue2EndlessTarget t = target_for(l, k);
    bool use = true;   // fold the path at k (false: a change deferred to the next chunk)
    if (needs_bridge(l, k, t)) {
        // The bridge's probe renders from the first unrendered chunk: there
        // must be time for it before that chunk plays, else the change waits
        // for the next chunk.
        const double left = seconds_until(static_cast<std::int64_t>(l.rendered) * C * kR - kHalo * kR) -
                            opt.emit_lead_seconds;
        if (!opt.paced || left > est_bridge * 1.2) return bridge(l, k, t);
        t = l.chunk(k - 1).target;
        use = false;
    }
    const Chunk* prev = k > 0 && l.has_chunk(k - 1) ? &l.chunk(k - 1) : nullptr;
    const bool change = prev && !t.same_music(prev->target);
    Yue2ChangeMode mode = opt.change_mode;
    if (mode == Yue2ChangeMode::Auto) mode = t.score.empty() ? Yue2ChangeMode::Drop : Yue2ChangeMode::Plain;
    if (mode == Yue2ChangeMode::Bridge) mode = t.score.empty() ? Yue2ChangeMode::Drop : Yue2ChangeMode::Plain;

    Roll roll = l.roll;
    Chunk c;
    c.roll_at = l.roll.at;
    c.roll_moved = l.roll.moved;
    c.roll_sheet = l.roll.sheet;
    c.row.index = k;
    c.row.branch = l.id;
    const std::string lyrics = lyrics_of(l, k, t, roll, &c.row.lyric_from, &c.row.lyric_to);
    const Yue2Plan& plan = plan_for(t, lyrics);
    const int mem = change && mode == Yue2ChangeMode::Drop ? opt.change_memory : opt.memory;
    Yue2Splice sp;
    const int T = static_cast<int>(l.tokens.size());
    sp.before.assign(l.tokens.begin() + std::max(0, T - mem), l.tokens.end());
    sp.middle_frames = C;
    sp.candidates = 1;
    sp.continue_after = false;
    const Yue2Params p = params_for(t, lyrics, l.seed * 1000 + static_cast<std::uint64_t>(k), true);
    work(l.id ? "branch: splice" : "splice", k);
    const auto t0 = Clock::now();
    Yue2SpliceResult r = model.splice(p, plan, sp, {}, preemptible(l, k, t));
    end_preemptible();
    if (r.cancelled) return false;
    const int middle = static_cast<int>(r.semantic.size()) - r.middle0;
    if (middle < C) throw std::runtime_error("brosoundml: Yue2Endless: chunk " + std::to_string(k) + ": the splice made " +
                                             std::to_string(middle) + " tokens, wanted " + std::to_string(C));
    append(l, r.semantic.data() + r.middle0, C);
    c.row.splice_seconds = since(t0);
    mean_in(est_splice, c.row.splice_seconds);
    c.row.memory = static_cast<int>(sp.before.size());
    c.row.tag = t.tag;
    c.row.style = t.style;
    c.row.scored = !t.score.empty();
    c.row.change = change;
    c.row.how = change ? (mode == Yue2ChangeMode::Drop ? "drop" : "plain") : "";
    c.row.lyrics = lyrics;
    c.row.knobs = t.knobs;
    c.target = t;
    l.roll = std::move(roll);
    if (use) fold(l, k);
    add_chunk(l, std::move(c));
    if (change) {
        Yue2EndlessEvent e{Yue2EndlessEvent::Kind::StyleChange};
        e.chunk = k;
        e.branch = l.id;
        e.tag = t.tag;
        e.text = l.chunk(k).row.how;
        e.sample = static_cast<std::int64_t>(k) * C * kR;
        event(e);
    }
    return true;
}

// ─── rendering a chunk ──────────────────────────────────────────────────────

void Yue2Endless::Impl::built(Line& l, int k, double seconds) {
    Chunk& c = l.chunk(k);
    c.row.rendered = true;
    c.row.latent_norm = norm_of(l, k * C, (k + 1) * C);
    c.row.inpaint_seconds = seconds;
    hear(l, k);
    Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Built};
    e.chunk = k;
    e.branch = l.id;
    e.tag = c.row.tag;
    e.sample = static_cast<std::int64_t>(k) * C * kR;
    e.seconds = seconds;
    e.value = c.row.latent_norm;
    e.row = std::make_shared<Yue2EndlessChunk>(c.row);
    e.frame0 = k * C;
    if (opt.chunk_data) e.latents = std::make_shared<std::vector<float>>(l.lat_at(k * C), l.lat_at((k + 1) * C));
    event(e);
    if (c.row.latent_norm < 6.5 || c.row.latent_norm > 9.5) {
        Yue2EndlessEvent w{Yue2EndlessEvent::Kind::BreakWarning};
        w.chunk = k;
        w.branch = l.id;
        w.tag = c.row.tag;
        w.value = c.row.latent_norm;
        w.sample = e.sample;
        event(w);
    }
}

void Yue2Endless::Impl::render(Line& l) {
    const int k = l.rendered, T = static_cast<int>(l.tokens.size());
    const int c0 = k * C, c1 = c0 + C, e1 = std::min(T, c1 + opt.lookahead);
    const int s0 = std::max(0, c0 - opt.nar_left);
    if (s0 < l.lat_base) throw std::runtime_error("brosoundml: Yue2Endless: latents trimmed under a render");
    Chunk& c = l.chunk(k);
    const Yue2EndlessTarget t = c.target;
    const std::string lyrics = c.row.lyrics;
    const Yue2Plan& plan = plan_for(t, lyrics);
    std::vector<int32_t> win(l.tokens.begin() + s0, l.tokens.begin() + e1);
    std::vector<float> wl(win.size() * kD, 0.0f);
    std::copy(l.lat_at(s0), l.lat_at(c0), wl.begin());
    Yue2Params p = params_for(t, lyrics, l.seed * 1000 + 500 + static_cast<std::uint64_t>(k), false);
    nar_steer(p, t.knobs, static_cast<int>(win.size()));
    work(l.id ? "branch: render" : "render", k);
    const auto t0 = Clock::now();
    std::vector<float> out = model.inpaint(p, plan, win, std::move(wl), c0 - s0, e1 - s0, {}, servicing());
    if (out.empty()) return;   // stopping
    l.lat_ensure(c1);
    std::copy(out.begin() + static_cast<std::ptrdiff_t>(c0 - s0) * kD, out.begin() + static_cast<std::ptrdiff_t>(c1 - s0) * kD,
              l.lat_at(c0));
    const double sec = since(t0);
    mean_in(est_render, sec);
    l.rendered = k + 1;
    l.chunk(k).row.nar_window = static_cast<int>(win.size());
    built(l, k, sec);
}

}  // namespace brosoundml
