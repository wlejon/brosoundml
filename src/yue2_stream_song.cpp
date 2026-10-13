// Yue2Stream: one song's stages — plan, the resumable semantic sampler, the
// inpainted NAR windows and the just-in-time decode — and the knob
// reactions on them. See yue2_stream.h.

#include "yue2_stream_impl.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace brosoundml {

using namespace yue2;

namespace {

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("brosoundml: Yue2Stream: " + m); }

double since(StreamClock::time_point t) { return std::chrono::duration<double>(StreamClock::now() - t).count(); }

// Marks a song as in a long op, so the program mix (serviced from inside
// it) does not free it under the op.
struct Busy {
    const Song*& slot;
    const Song* prev;
    Busy(const Song*& s, const Song* v) : slot(s), prev(s) { slot = v; }
    ~Busy() { slot = prev; }
};

}  // namespace

void Yue2Stream::Impl::start_song(Song& s, Yue2Song req) {
    const int idx = next_index++;
    s = Song{};
    s.index = idx;
    s.req = std::move(req);
    s.p = s.req.params;
    s.p.steer.clear();      // the knobs steer
    s.p.capture = Yue2Capture{};
    s.queued_at = StreamClock::now();
    s.off_tail = knobs.energy;
    s.nar21 = knobs.texture;
    s.nar9 = knobs.percussion;
    s.ar = knobs.semantic;
    s.limit = opt.preview_seconds > 0 ? std::max(50, static_cast<int>(std::lround(opt.preview_seconds * 25.0))) : -1;
}

int Yue2Stream::Impl::make_full(Song& s) {
    if (s.limit < 0) return -1;
    const int L = s.limit;
    s.limit = -1;
    if (!s.planned || !s.sem_done) return 0;   // still being built: it simply goes on
    if (s.ended && static_cast<int>(s.codec.size()) <= L) {
        s.natural = vae.output_length(s.frames());   // its own end came within the preview
        return -1;
    }
    if (s.ended || !s.req.semantic.empty()) {
        // The tokens are all there (its end came in the lookahead, or given
        // tokens): the rest is latents.
        s.natural = vae.output_length(s.frames());
    } else {
        Busy busy_guard(busy, &s);
        // The session holds the prompt and every token so far (a preview
        // built ahead released it: rebuilt over them), so sampling resumes
        // after the lookahead tokens.
        if (!s.sess) open_session(s, static_cast<int>(s.codec.size()), true);
        s.sem_done = false;
        s.natural = -1;
    }
    s.cut = -1;
    s.built_announced = false;
    // The preview's last frames were solved against MUSIC_END after its
    // lookahead: re-solve them (what is not handed out yet) with the
    // continuation as their lookahead, inpainted against what precedes.
    const int join = std::min(s.lat_valid, std::max(L - kJoinFrames, emitted_frame(s) + 25));
    resolve_from(s, join);
    // The long window from the join: the short first window is for audio
    // someone is waiting for, and a 250-frame window solves loud (mean latent
    // norm ~9.4 against the whole render's ~8.0, past the corpus range).
    s.win_k = static_cast<int>(opt.windows.size()) - 1;
    unplace_after(&s);
    return join;
}

// The semantic session over the prompt (and the CFG negative), then the
// song's first `codec_rows` codec tokens: a fresh song (0), or one whose
// session was released when it was built ahead and is continued on air.
// `stepped`: the tokens go in one decode step at a time, as they were
// sampled, so the cache is the one sampling built and the song continues
// with the tokens one uninterrupted run gives (a batched prefill rounds
// differently and the next token can differ). A preview's tokens take
// about 2 s; a steer's continuation (latency first) prefills.
void Yue2Stream::Impl::open_session(Song& s, int codec_rows, bool stepped) {
    const Model& m = core.model;
    const auto [lo, hi] = logit_range(Phase::Semantic);
    codec_rows = std::clamp(codec_rows, 0, static_cast<int>(s.codec.size()));
    const int longest = std::max<int>(s.P, static_cast<int>(s.neg.size())) + codec_rows;
    s.sess.reset();
    s.sess = m.ar_session(s.neg.empty() ? 1 : 2, std::min(CONTEXT, longest + 1024), s.ar_hooks.get(), CONTEXT);
    const int prefilled = stepped ? 0 : codec_rows;
    auto with_codec = [&](std::vector<int32_t> ids) {
        for (int f = 0; f < prefilled; ++f) ids.push_back(s.codec[f] + CODEC_OFFSET);
        return ids;
    };
    m.ar_prefill(*s.sess, 0, with_codec(s.plan.prefix), lo, hi);
    if (!s.neg.empty()) m.ar_prefill(*s.sess, 1, with_codec(s.neg), lo, hi);
    for (int f = prefilled; f < codec_rows; ++f) {
        m.ar_step(*s.sess, s.codec[f] + CODEC_OFFSET);
        if (f % 32 == 31) service();
    }
}

void Yue2Stream::Impl::release_session(Song& s) {
    if (!s.sess || &s == busy) return;
    s.sess.reset();
}

void Yue2Stream::Impl::plan(Song& s) {
    Busy busy_guard(busy, &s);
    const Core& I = core;
    Yue2Params& p = s.p;
    const auto cursor = std::make_shared<std::size_t>(0);
    // The plan and semantic uniforms as Yue2::generate draws them.
    PlanOut po = plan_impl(I, p, Yue2Callbacks{}, servicing(), stage_uniform(p, cursor));
    if (stopping.load()) return;
    if (po.run.cancelled) fail("plan cancelled");
    s.uniform = stage_uniform(p, cursor);
    s.plan = std::move(po.plan);
    s.P = static_cast<int>(s.plan.prefix.size());
    const Encode enc = [&](const std::string& t) { return I.encode(t); };
    s.cfg = guidance(p);
    if (s.cfg != 1.0f) s.neg = negative_prefix(p, enc, s.plan.abc_ids);
    s.sampling = guarded_semantic(p, s.plan.abc);
    validate(s.sampling, "semantic");
    int nf = s.sampling.max_tokens;
    if (!s.req.semantic.empty()) nf = static_cast<int>(s.req.semantic.size());
    if (!p.test_noise.empty()) nf = std::min<int>(nf, static_cast<int>(p.test_noise.size() / Yue2::kLatentDim));
    s.noise = song_noise(p, nf);
    set_ar_steer(s, s.ar);   // before the prefill: a whole-song semantic steer covers the prompt too
    open_session(s, 0);
    if (!s.req.semantic.empty()) {
        for (int32_t c : s.req.semantic)
            if (c < 0 || c >= CODEC_SIZE) fail("semantic ids must be 0..32767");
        s.codec = s.req.semantic;
        s.tokens.clear();
        for (int32_t c : s.codec) s.tokens.push_back(c + CODEC_OFFSET);
        s.sem_done = true;
        s.ended = s.limit < 0 || static_cast<int>(s.codec.size()) <= s.limit;
        s.natural = vae.output_length(s.frames());
    }
    s.est_frames = s.plan.abc.empty() ? 0 : static_cast<int>(abc_seconds(s.plan.abc) * 25.0);
    s.planned = true;
    Yue2StreamEvent e{Yue2StreamEvent::Kind::Plan};
    e.song = s.index;
    e.tag = s.req.tag;
    e.text = s.plan.abc;
    event(e);
    if (s.sem_done) {
        Yue2StreamEvent d{Yue2StreamEvent::Kind::SemanticDone};
        d.song = s.index;
        d.tag = s.req.tag;
        d.frames = s.frames();
        event(d);
    }
}

// Yue2::generate's semantic loop (run_ar), a few tokens at a time.
void Yue2Stream::Impl::semantic_step(Song& s, int n) {
    Busy busy_guard(busy, &s);
    const Model& m = core.model;
    const auto [lo, hi] = logit_range(Phase::Semantic);
    const int nn = hi - lo;
    std::vector<float> logits, combined;
    auto finish = [&](bool ended) {
        s.sem_done = true;
        s.ended = ended;
        const int T = s.frames();
        if (T == 0) fail("the semantic stage ended at once (an empty song)");
        s.natural = vae.output_length(T);
        if (opt.on_semantic) opt.on_semantic(s.index, s.codec);
        Yue2StreamEvent d{Yue2StreamEvent::Kind::SemanticDone};
        d.song = s.index;
        d.tag = s.req.tag;
        d.frames = T;
        event(d);
    };
    for (int k = 0; k < n && !s.sem_done; ++k) {
        const int step = static_cast<int>(s.codec.size());
        if (step >= s.sampling.max_tokens || step >= static_cast<int>(s.noise.size() / Yue2::kLatentDim)) {
            finish(true);
            break;
        }
        if (s.limit >= 0 && step >= s.limit + opt.right) {   // a preview: its tokens and the lookahead
            finish(false);
            break;
        }
        m.ar_logits(*s.sess, logits);
        float* row = logits.data();
        if (!s.neg.empty()) {
            combined.resize(nn);
            cfg_combine_bf16(logits.data(), logits.data() + nn, s.cfg, combined.data(), nn);
            row = combined.data();
        }
        check_finite(row, nn, "semantic");
        apply_end_bias(row, end_token(Phase::Semantic) - lo, s.sampling, step);
        const auto sc = distribution(row, nn, Phase::Semantic, s.sampling, s.tokens, step, s.p.cot == Yue2Cot::Off);
        const int32_t tok = draw(sc, lo, s.sampling.temperature, s.uniform);
        if (tok == end_token(Phase::Semantic)) {
            finish(true);
            break;
        }
        s.tokens.push_back(tok);
        s.codec.push_back(tok - CODEC_OFFSET);
        if (step + 1 < s.sampling.max_tokens) m.ar_step(*s.sess, tok);
        if (k % 8 == 7) service();
    }
}

bool Yue2Stream::Impl::window_ready(const Song& s, int* a_out, int* b_out) const {
    if (!s.planned || s.failed) return false;
    int a = s.lat_valid;
    const int T = s.frames();
    if (T >= 0 && a >= T) return false;
    // A song built ahead (not on air, no skip waiting for it) is solved in
    // one chunk once its tokens are all there, the whole-song render itself,
    // when that fits before it is needed. Windows are for audio someone is
    // waiting for.
    const double budget = ahead_budget(s);
    if (budget > 0) {
        // The semantic stage first, when the rest of it and the one chunk
        // fit in the time left: always that far ahead, else by the tokens
        // left (a preview's ~2 s; a whole song's by the plan's estimate, at
        // a conservative 200 tokens/s).
        if (T < 0) {
            const int end = s.limit >= 0 ? s.limit + opt.right : std::max(s.est_frames, static_cast<int>(s.codec.size()));
            const int left = std::max(0, end - static_cast<int>(s.codec.size()));
            if (budget > kWholeWaitSeconds || (end > 0 && budget > left / 200.0 + whole_seconds(s, end - a) + 2.0))
                return false;
        }
        if (T >= 0 && whole_seconds(s, T - a) < budget) {
            *a_out = a;
            *b_out = T;
            return true;
        }
    }
    const int size = opt.windows[std::min<std::size_t>(s.win_k, opt.windows.size() - 1)];
    int b = a + size;
    if (T >= 0) {
        // A short window solves loud (a 250-frame one: mean latent norm ~9.4
        // against the whole render's ~8.0; a 50-frame end ~11.5), so the
        // end never gets a short window of its own: the window before it
        // runs to the end, and a short last window (the end came after the
        // window before it was cut) starts earlier, re-solving frames not
        // handed out yet.
        const int full = *std::max_element(opt.windows.begin(), opt.windows.end());
        b = std::min(b, T);
        if (T - b < full) b = T;
        if (T - a < full) {
            const int earliest = s.start >= 0 ? std::max(0, emitted_frame(s) + 25) : 0;
            a = std::min(a, std::max(T - full, earliest));
        }
    } else if (static_cast<int>(s.codec.size()) < b + opt.right) {
        return false;
    }
    *a_out = a;
    *b_out = b;
    return true;
}

double Yue2Stream::Impl::ahead_budget(const Song& s) const {
    const Song* prev = nullptr;
    for (std::size_t i = 1; i < songs.size(); ++i)
        if (songs[i].get() == &s) prev = songs[i - 1].get();
    if (!prev) return 0;                                 // the first song: someone waits for it
    if (skip_pending && songs.size() >= 2 && songs[1].get() == &s) return 0;
    if (!opt.paced) return 1e30;                         // offline: nobody waits, after the first song
    if (s.start >= 0 && s.start <= emitted) return 0;   // on air
    std::int64_t need = s.start;
    if (need < 0) need = prev->end();                    // not placed yet: by its predecessor's end
    if (need < 0) return 1e30;                           // its predecessor's length is not known yet
    return static_cast<double>(need - std::max(emitted, playhead.load())) / 48000.0;
}

double Yue2Stream::Impl::whole_seconds(const Song& s, int frames) const {
    return 1.5 * nar_spfs * (frames + opt.left) * std::max(1, s.p.ode_steps) + 2.0;   // + the opening's decode
}

void Yue2Stream::Impl::solve_window(Song& s, int a, int b) {
    Busy busy_guard(busy, &s);
    const Model& m = core.model;
    const auto t0 = StreamClock::now();
    const int gen0 = s.gen, valid0 = s.lat_valid;   // a <= valid0: a last window may re-solve unheard frames
    const int lo = std::max(0, a - opt.left);
    const int hi = std::min(static_cast<int>(s.codec.size()), b + opt.right);
    const int F = hi - lo, D = Yue2::kLatentDim, H = m.config().hidden;
    std::vector<int32_t> ar(s.plan.prefix);
    for (int f = lo; f < hi; ++f) ar.push_back(s.codec[f] + CODEC_OFFSET);
    ar.push_back(MUSIC_END);
    Model::NarOptions o;
    if (s.sess && m.ar_len(*s.sess, 0) >= s.P) {
        o.reuse = s.sess.get();
        o.reuse_rows = s.P;
    }
    o.frame0 = lo;
    // Texture / percussion: per-position NAR steers over the window's frames
    // (lab.nar steers the frame rows only, not LATENT_START / END).
    LayerHooks h;
    h.steer.resize(m.config().layers);
    auto add = [&](int layer, const std::vector<float>& dir, float scale) {
        if (scale == 0) return;
        std::vector<float> rows(static_cast<std::size_t>(F) * H);
        for (int r = 0; r < F; ++r) std::copy(dir.begin(), dir.end(), rows.begin() + static_cast<std::size_t>(r) * H);
        h.steer[layer].push_back(m.make_steer(rows, scale, lo, true));
    };
    add(21, opt.directions.nar_l21, s.nar21);
    add(9, opt.directions.nar_l9, s.nar9);
    if (s.nar21 != 0 || s.nar9 != 0) o.hooks = &h;
    std::vector<float> out;
    {
        work("window: prefill", &s);
        auto chunk = m.nar_chunk(ar, F, o);
        work("window: step", &s);
        const bool whole = b - a > *std::max_element(opt.windows.begin(), opt.windows.end()) + 25;
        const int steps = s.p.ode_steps;
        out = m.nar_solve(*chunk, s.noise.data() + static_cast<std::size_t>(lo) * D, steps, preemptible(whole),
                          [&](int done) { if (done == steps) work("window: readback", &s); },
                          a > lo ? s.lat.data() + static_cast<std::size_t>(lo) * D : nullptr, a - lo);
        work("window: after", &s);
    }
    if (out.empty()) return;   // stopping, or preempted (re-solved later)
    // The conditions may have moved while it solved (a knob re-solve from
    // earlier than a): then this window is stale.
    if (s.gen != gen0 || s.lat_valid != valid0) return;
    if (static_cast<int>(s.lat.size() / D) < b) {
        s.lat.resize(static_cast<std::size_t>(b) * D, 0.0f);
        s.off.resize(b, s.off_tail);
    }
    std::copy(out.begin() + static_cast<std::size_t>(a - lo) * D, out.begin() + static_cast<std::size_t>(b - lo) * D,
              s.lat.begin() + static_cast<std::size_t>(a) * D);
    s.lat_valid = b;
    s.lat_extent = std::max(s.lat_extent, b);
    ++s.win_k;
    // Mean latent norm as decoded (offset included): the break meter.
    double norm = 0;
    for (int f = a; f < b; ++f) {
        double q = 0;
        for (int d = 0; d < D; ++d) {
            const double v = s.lat[static_cast<std::size_t>(f) * D + d] + s.off[f] * opt.directions.latent[d];
            q += v * v;
        }
        norm += std::sqrt(q);
    }
    norm /= std::max(1, b - a);
    s.last_norm = norm;
    invalidate_audio(s, a - kHalo);
    // Stage 2 of an interrupt landed: the offset ramps back to the knob at the cut.
    if (int_song == s.index && int_cut >= a && int_cut < b) {
        set_offset(s, int_cut, knobs.energy, kHalo);
        if (s.start >= 0) int_audible2 = s.start + static_cast<std::int64_t>(int_cut) * R;
        const double took = since(int_at);
        int_latency_est = std::clamp(1.15 * took, 1.0, 6.0);
        int_song = -1;
        int_cut = -1;
    }
    if (opt.on_latents) opt.on_latents(s.index, a, b - a, s.lat.data() + static_cast<std::size_t>(a) * D);
    Yue2StreamEvent e{Yue2StreamEvent::Kind::Window};
    e.song = s.index;
    e.tag = s.req.tag;
    e.frame0 = a;
    e.frames = b - a;
    e.seconds = since(t0);
    e.value = norm;
    // Service work done inside the solve is counted too (a conservative rate).
    nar_spfs = e.seconds / (static_cast<double>(F) * std::max(1, s.p.ode_steps));
    event(e);
    if (norm < 6.5 || norm > 9.5) {
        Yue2StreamEvent w{Yue2StreamEvent::Kind::BreakWarning};
        w.song = s.index;
        w.tag = s.req.tag;
        w.frame0 = a;
        w.frames = b - a;
        w.value = norm;
        event(w);
    }
}

bool Yue2Stream::Impl::decode_tile(Song& s, int max_frames, bool whole) {
    if (s.failed || !s.planned) return false;
    const int D = Yue2::kLatentDim;
    const int T = s.frames();
    const bool complete = T >= 0 && s.lat_valid >= T;
    const int top = complete ? T : s.lat_extent;   // frames held under or ahead of the current conditions
    int limit = complete ? T : s.lat_extent - kHalo;
    if (s.cut >= 0) limit = std::min<int>(limit, static_cast<int>((s.cut + R - 1) / R));
    const int f = s.dec_frame;
    const int e = static_cast<int>(std::min<std::int64_t>({static_cast<std::int64_t>(f) + max_frames, limit,
                                                           whole ? INT32_MAX : decode_target(s)}));
    if (e <= f) return false;
    const int l0 = std::max(0, f - kHalo), h0 = std::min(top, e + kHalo);
    // Effective latents: z + offset * the energy direction.
    std::vector<float> z(static_cast<std::size_t>(h0 - l0) * D);
    if (static_cast<int>(s.off.size()) < h0) s.off.resize(h0, s.off_tail);
    for (int fr = l0; fr < h0; ++fr)
        for (int d = 0; d < D; ++d)
            z[static_cast<std::size_t>(fr - l0) * D + d] =
                s.lat[static_cast<std::size_t>(fr) * D + d] + s.off[fr] * opt.directions.latent[d];
    const StereoAudio a = vae.decode_span(z.data(), h0 - l0, f - l0, e - l0, complete && e == T && h0 == T);
    const std::size_t at = static_cast<std::size_t>(f) * R, n = static_cast<std::size_t>(a.frames);
    const std::size_t had = s.L.size();
    if (s.L.size() < at + n) {
        s.L.resize(at + n, 0.0f);
        s.R.resize(at + n, 0.0f);
    }
    // Re-decoded audio meets what was already handed out with a 20 ms
    // crossfade at the emit frontier (that part cannot change any more).
    std::size_t es = SIZE_MAX;
    if (s.start >= 0 && emitted > s.start) es = static_cast<std::size_t>(emitted - s.start);
    const std::size_t XF = 960;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = at + i;
        float l = a.channel(0)[i], r = a.channel(1)[i];
        if (es != SIZE_MAX && j >= es && j < es + XF && j < had) {
            const float w = (static_cast<float>(j - es) + 0.5f) / static_cast<float>(XF);
            l = s.L[j] * (1 - w) + l * w;
            r = s.R[j] * (1 - w) + r * w;
        }
        s.L[j] = l;
        s.R[j] = r;
    }
    s.dec_frame = e;
    return true;
}

void Yue2Stream::Impl::set_ar_steer(Song& s, const std::map<std::string, float>& scales) {
    const Model& m = core.model;
    s.ar = scales;
    StageHooks& h = s.ar_hooks;
    h.hooks.steer.assign(m.config().layers, {});
    h.hooks.capture.assign(m.config().layers, 0);
    h.active = false;
    for (const auto& [name, scale] : scales) {
        const Yue2SemanticAxis* a = opt.directions.axis(name);
        if (!a || scale == 0 || a->layer < 0 || a->layer >= m.config().layers) continue;
        h.hooks.steer[a->layer].push_back(m.make_steer(a->direction, scale, 0, false));
        h.active = true;
    }
    if (s.sess) m.ar_set_hooks(*s.sess, h.get());
}

void Yue2Stream::Impl::set_nar_steer(Song& s, float t21, float p9) {
    s.nar21 = t21;
    s.nar9 = p9;
}

void Yue2Stream::Impl::invalidate_audio(Song& s, int frame) {
    const int f = std::max(0, frame);
    if (f < s.dec_frame) {
        s.dec_frame = f;
        s.built_announced = false;
    }
}

void Yue2Stream::Impl::set_offset(Song& s, int f0, float value, int ramp) {
    f0 = std::max(0, f0);
    if (static_cast<int>(s.off.size()) < s.lat_extent) s.off.resize(s.lat_extent, s.off_tail);
    const int n = static_cast<int>(s.off.size());
    const float from = f0 == 0 ? (ramp > 0 ? s.off_tail : value) : (f0 - 1 < n ? s.off[f0 - 1] : s.off_tail);
    for (int f = f0; f < n; ++f) {
        const float w = ramp > 0 ? std::min(1.0f, static_cast<float>(f - f0 + 1) / static_cast<float>(ramp)) : 1.0f;
        s.off[f] = from + (value - from) * w;
    }
    // Frames past the solved ones take the target (a ramp still in progress
    // at the frontier finishes there).
    s.off_tail = value;
    invalidate_audio(s, f0 - kHalo);
}

void Yue2Stream::Impl::resolve_from(Song& s, int f) {
    f = std::clamp(f, 0, s.lat_valid);
    ++s.gen;
    s.built_announced = false;
    s.lat_valid = f;
    s.win_from = f;
    s.win_k = 0;   // the first (short) window again: the change is heard soonest
}

int Yue2Stream::Impl::continue_from(Song& s, int c, const std::map<std::string, float>& scales) {
    if (!s.planned) {
        s.ar = scales;   // applied when it is planned
        return -1;
    }
    if (!s.req.semantic.empty()) {   // given tokens: nothing to continue
        s.ar = scales;
        return -1;
    }
    Busy busy_guard(busy, &s);
    const Model& m = core.model;
    c = std::max(0, c);
    if (c > 0 && c >= static_cast<int>(s.codec.size()) && !s.sem_done) {
        set_ar_steer(s, scales);   // not sampled that far yet: steer from here on
        return c;
    }
    if (c > 0 && c >= static_cast<int>(s.codec.size())) {   // past the song's end
        s.ar = scales;
        return -1;
    }
    if (c == 0) {
        s.sess.reset();
        set_ar_steer(s, scales);
        s.codec.clear();
        s.tokens.clear();
        open_session(s, 0);
    } else {
        // A song built ahead released its session: rebuild it over the
        // tokens before the cut (under the steer they were sampled with).
        if (!s.sess) open_session(s, c - 1);
        // Re-feed the token before the cut under the new steer, which
        // recomputes the logits for the first continued position.
        m.ar_truncate(*s.sess, 0, s.P + c - 1);
        if (!s.neg.empty()) m.ar_truncate(*s.sess, 1, static_cast<int>(s.neg.size()) + c - 1);
        set_ar_steer(s, scales);
        m.ar_step(*s.sess, s.codec[c - 1] + CODEC_OFFSET);
        s.codec.resize(c);
        s.tokens.resize(c);
    }
    ++s.interrupts;
    s.uniform = make_uniform(s.p.seed + 1000003ull * static_cast<std::uint64_t>(s.interrupts));
    s.sem_done = false;
    s.natural = -1;
    s.cut = -1;
    s.built_announced = false;
    resolve_from(s, c);
    // Successors placed by the old length move with the new one, unless
    // they are already playing.
    unplace_after(&s);
    return c;
}

}  // namespace brosoundml
