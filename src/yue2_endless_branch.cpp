// Yue2Endless: branches. A branch is another line from the main line's
// history: the main line's tokens up to its fork and the main line's renders
// of the chunks before it, then chunks of its own under its own targets. A
// commit makes it the main line, seamlessly at its fork when that has not
// been handed out, else crossfading into it at the first chunk boundary it
// can make.

#include "yue2_endless_impl.h"

#include <algorithm>
#include <cmath>

namespace brosoundml {

using namespace endless;

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

Yue2Endless::Impl::Line* Yue2Endless::Impl::branch_of(int id) {
    for (auto& b : branches)
        if (b->id == id && !b->failed) return b.get();
    return nullptr;
}

void Yue2Endless::Impl::open_branch(int id, const Yue2EndlessTarget& t, std::vector<Yue2EndlessTarget> path) {
    int live = 0;
    for (const auto& b : branches) live += !b->failed;
    if (live >= opt.max_branches) {
        Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Discarded};
        e.branch = id;
        e.text = "max_branches (" + std::to_string(opt.max_branches) + ") are open";
        event(e);
        return;
    }
    const Line& m = *main;
    const int f = open_chunk(m);
    auto b = std::make_unique<Line>();
    b->id = id;
    b->seed = opt.seed + static_cast<std::uint64_t>(id);
    b->fork = f;
    b->tokens.assign(m.tokens.begin(), m.tokens.begin() + static_cast<std::ptrdiff_t>(f) * C);
    b->rendered = std::min(m.rendered, f);
    b->lat_base = m.lat_base;
    b->lat.assign(m.lat.begin(), m.lat.begin() + static_cast<std::ptrdiff_t>(std::max(0, b->rendered * C - m.lat_base)) * kD);
    b->lat_ensure(static_cast<int>(b->tokens.size()));
    b->chunk_base = m.chunk_base;
    for (int k = m.chunk_base; k < f; ++k) b->chunks.push_back(m.chunk(k));
    b->base = t;
    b->base.at = -1;
    b->base.knobs = Yue2Knobs::clamp(b->base.knobs, opt.directions);
    Line& ref = *b;
    branches.push_back(std::move(b));
    for (Yue2EndlessTarget p : path) {
        if (p.at < 0) p.at = 0;
        place_target(ref, std::move(p), false);
    }
    Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Branch};
    e.branch = id;
    e.chunk = f;
    e.tag = t.tag;
    e.sample = static_cast<std::int64_t>(f) * C * kR;
    event(e);
}

// The chunks before the fork are the main line's: their renders copied as
// the main line makes them.
void Yue2Endless::Impl::sync_branch(Line& b) {
    const Line& m = *main;
    while (b.rendered < b.fork && m.rendered > b.rendered) {
        const int k = b.rendered;
        b.lat_ensure((k + 1) * C);
        std::copy(m.lat_at(k * C), m.lat_at((k + 1) * C), b.lat_at(k * C));
        if (b.has_chunk(k) && m.has_chunk(k)) b.chunk(k).row = m.chunk(k).row;
        b.rendered = k + 1;
    }
}

bool Yue2Endless::Impl::step_branches() {
    // Discarded branches go once nothing works on them.
    branches.erase(std::remove_if(branches.begin(), branches.end(), [](const auto& b) { return b->failed; }),
                   branches.end());
    for (auto& bp : branches) {
        Line& b = *bp;
        sync_branch(b);
        if (can_render(b)) {
            render(b);
            return true;
        }
        if (sample_due(b) && b.sampled() <= main->sampled()) {
            sample(b);
            return true;
        }
    }
    return false;
}

void Yue2Endless::Impl::request_commit(int id) {
    Line* b = branch_of(id);
    if (!b) {
        Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Error};
        e.branch = id;
        e.text = "commit_branch: no open branch " + std::to_string(id);
        event(e);
        return;
    }
    for (auto& o : branches) o->committing = false;
    b->committing = true;
}

void Yue2Endless::Impl::discard(int id, const std::string& why) {
    Line* b = branch_of(id);
    if (!b) return;
    b->failed = true;
    Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Discarded};
    e.branch = id;
    e.text = why;
    event(e);
}

bool Yue2Endless::Impl::try_commit() {
    Line* bp = nullptr;
    for (auto& b : branches)
        if (b->committing && !b->failed) bp = b.get();
    if (!bp) return false;
    Line& b = *bp;
    sync_branch(b);
    const int f = b.fork;
    const std::int64_t lead = static_cast<std::int64_t>((opt.emit_lead_seconds + 0.25) * kSr);
    int g = f;
    double fade = 0;
    std::int64_t at = static_cast<std::int64_t>(f) * C * kR;   // the program sample the branch is heard alone from
    if (emitted + lead <= static_cast<std::int64_t>(f * C - kHalo) * kR &&
        (b.rendered > f || seconds_until(static_cast<std::int64_t>(f) * C * kR) > 1.5 * est_render + est_splice)) {
        // Seamless: the branch shares the history up to its fork, and the
        // program re-decodes from there (the frames before the fork are the
        // same latents; their right halo is the branch's now).
        redecode_from(f * C - kHalo);
    } else {
        // The fade ends a decoder halo before boundary g, so both sides
        // decode exactly from their chunks before g: the main line's audio
        // up to there needs only its chunk g-1 rendered, the branch's too.
        const std::int64_t x = static_cast<std::int64_t>(opt.branch_fade_seconds * kSr);
        const std::int64_t per = static_cast<std::int64_t>(C) * kR;
        const std::int64_t halo = static_cast<std::int64_t>(kHalo) * kR;
        g = static_cast<int>(std::max<std::int64_t>(f + 1, (emitted + lead + x + halo + per - 1) / per));
        if (b.rendered < g) return false;                  // it keeps building; the boundary moves on
        const int fb = g * C - kHalo;
        if (dec_frame < fb) return decode_tile(std::min(2 * kBlock, fb - dec_frame));   // the main line up to the fade's end
        // The branch's audio under the fade, from its own latents.
        const std::int64_t gS = static_cast<std::int64_t>(fb) * kR, xs = gS - x;
        const int fa = static_cast<int>(xs / kR);
        const int l0 = std::max(b.lat_base, fa - kHalo), h0 = std::min(b.rendered * C, fb + kHalo);
        std::vector<float> z(static_cast<std::size_t>(h0 - l0) * kD);
        for (int fr = l0; fr < h0; ++fr) {
            const float e = energy_at(fr);
            for (int d = 0; d < kD; ++d)
                z[static_cast<std::size_t>(fr - l0) * kD + d] = b.lat_at(fr)[d] + e * opt.directions.latent[d];
        }
        work("commit: fade", g);
        in_decode = true;
        const StereoAudio a = vae.decode_span(z.data(), h0 - l0, fa - l0, fb - l0, false, 0, waiting());
        in_decode = false;
        for (std::int64_t j = xs; j < gS; ++j) {
            const double w = (static_cast<double>(j - xs) + 0.5) / static_cast<double>(x);
            const double go = std::cos(0.5 * kPi * w), gi = std::sin(0.5 * kPi * w);
            const std::size_t pj = static_cast<std::size_t>(j - prog_base);
            const std::size_t aj = static_cast<std::size_t>(j - static_cast<std::int64_t>(fa) * kR);
            L[pj] = static_cast<float>(go * L[pj] + gi * a.channel(0)[aj]);
            R[pj] = static_cast<float>(go * R[pj] + gi * a.channel(1)[aj]);
        }
        L.resize(static_cast<std::size_t>(gS - prog_base));
        R.resize(static_cast<std::size_t>(gS - prog_base));
        dec_frame = fb;
        xf_at = -1;
        fade = opt.branch_fade_seconds;
        at = gS;
    }
    // The branch is the main line.
    const int id = b.id;
    std::unique_ptr<Line> nb;
    for (auto& o : branches)
        if (o.get() == bp) nb = std::move(o);
    for (auto& o : branches)
        if (o && !o->failed) {
            Yue2EndlessEvent d{Yue2EndlessEvent::Kind::Discarded};
            d.branch = o->id;
            d.text = "another branch was committed";
            event(d);
        }
    branches.clear();
    main = std::move(nb);
    main->id = 0;
    main->committing = false;
    for (auto& c : main->chunks) c.row.branch = 0;
    for (int k = std::max(g, main->chunk_base); k < main->sampled(); ++k)
        if (energy_at(k * C) != main->chunk(k).target.knobs.energy) set_energy(k * C, main->chunk(k).target.knobs.energy);
    Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Committed};
    e.branch = id;
    e.chunk = g;
    e.value = fade;
    e.sample = at;
    event(e);
    return true;
}

}  // namespace brosoundml
