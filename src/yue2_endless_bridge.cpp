// Yue2Endless: a style change under a score, the measured bridge
// (endless.js's bridge with phaseFrom 'bridge', beatSensor 'spectral',
// keepProbe): a fresh clip of the new style, a bridge chunk spliced toward
// it, the bridge probe-rendered (the probe kept as its render) and its beat
// read, the fresh clip probe-rendered and cut by up to a beat so its beat
// lands on the bridge's.

#include "yue2_endless_impl.h"

#include "brosoundml/beat.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace brosoundml {

using namespace endless;

namespace {

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

}  // namespace

double Yue2Endless::score_beat_frames(const std::string& abc) {
    // The Q: header: "Q:1/4=90" (a beat of the stated note), "Q:90" (quarters).
    std::size_t at = 0;
    while (at < abc.size()) {
        const std::size_t eol = abc.find('\n', at);
        std::string line = abc.substr(at, eol == std::string::npos ? std::string::npos : eol - at);
        at = eol == std::string::npos ? abc.size() : eol + 1;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.size() < 2 || line[0] != 'Q' || line[1] != ':') continue;
        const std::string v = line.substr(2);
        const std::size_t eq = v.find('=');
        const double bpm = std::atof((eq == std::string::npos ? v : v.substr(eq + 1)).c_str());
        if (bpm > 0) return 25.0 * 60.0 / bpm;
    }
    return 0;
}

bool Yue2Endless::Impl::bridge(Line& l, int k, const Yue2EndlessTarget& t) {
    const auto t0 = Clock::now();
    const double P0 = score_beat_frames(t.score);
    const int span = static_cast<int>(std::ceil(P0 - 1e-9));   // one beat: 17 frames at 90 bpm
    const int NL = opt.nar_left, LA = opt.lookahead;
    Roll roll = l.roll;
    Chunk cb, cf;
    cb.roll_at = cf.roll_at = l.roll.at;
    cb.roll_reached = cf.roll_reached = l.roll.reached;
    cb.roll_sheet = cf.roll_sheet = l.roll.sheet;
    const std::string lyrics = lyrics_of(t, roll, &cb.row.lyric_from, &cb.row.lyric_to);
    const Yue2Plan& plan = plan_for(t, lyrics);
    const LyricWords words = lyric_words(plan, lyrics, cb.row.lyric_from);
    const int T = static_cast<int>(l.tokens.size());
    const std::vector<int32_t> before(l.tokens.begin() + std::max(0, T - opt.memory), l.tokens.end());
    const std::uint64_t S = l.seed * 1000;

    // The fresh clip: no history, a beat longer than a chunk.
    Yue2Splice fs;
    fs.middle_frames = C + span;
    fs.candidates = 1;
    watch_lyrics(fs, words);
    work(l.id ? "branch: bridge fresh" : "bridge: fresh", k + 1);
    Yue2SpliceResult fresh = model.splice(params_for(t, lyrics, S + k + 1, true), plan, fs, {}, preemptible(l, k, t));
    end_preemptible();
    if (fresh.cancelled) return false;
    if (static_cast<int>(fresh.semantic.size()) - fresh.middle0 < C + span)
        throw std::runtime_error("brosoundml: Yue2Endless: the fresh clip came short");
    const std::vector<int32_t> F(fresh.semantic.begin() + fresh.middle0, fresh.semantic.begin() + fresh.middle0 + C + span);
    const auto t1 = Clock::now();

    // The bridge, toward the uncut clip.
    Yue2Splice bs;
    bs.before = before;
    bs.middle_frames = C;
    bs.after.assign(F.begin(), F.begin() + C);
    bs.candidates = opt.bridge_candidates;
    bs.score_frames = opt.bridge_score_frames;
    watch_lyrics(bs, words);
    work(l.id ? "branch: bridge splice" : "bridge: splice", k);
    Yue2SpliceResult r = model.splice(params_for(t, lyrics, S + k, true), plan, bs, {}, preemptible(l, k, t));
    end_preemptible();
    if (r.cancelled) return false;
    const std::vector<int32_t> bridge_tok(r.semantic.begin() + r.middle0, r.semantic.begin() + r.middle0 + C);
    const auto tb = Clock::now();

    // The unrendered chunks up to the bridge's end probe-rendered, the fresh
    // clip's opening as lookahead.
    const int r1 = l.rendered * C, s0 = std::max(0, r1 - NL);
    if (s0 < l.lat_base) throw std::runtime_error("brosoundml: Yue2Endless: latents trimmed under a bridge");
    std::vector<int32_t> win(l.tokens.begin() + s0, l.tokens.end());
    win.insert(win.end(), bridge_tok.begin(), bridge_tok.end());
    win.insert(win.end(), F.begin(), F.begin() + LA);
    std::vector<float> wl(win.size() * kD, 0.0f);
    std::copy(l.lat_at(s0), l.lat_at(r1), wl.begin());
    Yue2Params pp = params_for(t, lyrics, S + 800 + k, false);
    nar_steer(pp, t.knobs, static_cast<int>(win.size()));
    work(l.id ? "branch: bridge probe" : "bridge: probe", k);
    std::vector<float> pr = model.inpaint(pp, plan, win, std::move(wl), r1 - s0, static_cast<int>(win.size()), {}, servicing());
    if (pr.empty()) return false;   // stopping
    const int b0 = k * C - s0;
    work("bridge: beat", k);
    in_decode = true;
    const StereoAudio bd = vae.decode(pr.data() + static_cast<std::size_t>(b0) * kD, C, waiting());
    in_decode = false;
    service();
    const auto bm = beat_mono(bd.samples.data(), static_cast<std::size_t>(bd.frames), 0, static_cast<std::size_t>(C) * kR);
    const BeatFit sb = fit_beat(beat_onsets(bm.data(), bm.size(), BeatSensor::Spectral), static_cast<double>(k) * C * 4,
                                P0 * 4, false);
    service();
    // The fresh clip's own beat, from a probe render of it.
    Yue2Params fp = params_for(t, lyrics, S + 900 + k, false);
    nar_steer(fp, t.knobs, static_cast<int>(F.size()));
    work(l.id ? "branch: fresh probe" : "bridge: fresh probe", k + 1);
    std::vector<float> probe = model.inpaint(fp, plan, F, std::vector<float>(F.size() * kD, 0.0f), 0,
                                             static_cast<int>(F.size()), {}, servicing());
    if (probe.empty()) return false;
    work("bridge: beat", k + 1);
    in_decode = true;
    const StereoAudio pd = vae.decode(probe.data(), static_cast<int>(F.size()), waiting());
    in_decode = false;
    service();
    const auto pm = beat_mono(pd.samples.data(), static_cast<std::size_t>(pd.frames), 0, F.size() * kR);
    const BeatFit fb = fit_beat(beat_onsets(pm.data(), pm.size(), BeatSensor::Spectral), 0, P0 * 4, false);
    double dd = std::fmod((k + 1) * C + fb.phase * P0 - sb.phase * P0, P0);
    if (dd < 0) dd += P0;
    const int cut = std::min(span, static_cast<int>(std::floor(dd + 0.5)));
    const double fit_seconds = since(tb);

    // Commit: the bridge, the cut clip, and the probe as the render.
    append(l, bridge_tok.data(), C);
    append(l, F.data() + cut, C);
    const int from = l.rendered;
    std::copy(pr.begin() + static_cast<std::ptrdiff_t>(r1 - s0) * kD,
              pr.begin() + static_cast<std::ptrdiff_t>((k + 1) * C - s0) * kD, l.lat_at(r1));
    const double probe_seconds = fit_seconds;   // endless.js: the probe's share counts from the bridge's end
    const bool change = k > 0 && l.has_chunk(k - 1) && !t.same_music(l.chunk(k - 1).target);

    cb.row.index = k;
    cb.row.branch = l.id;
    cb.row.tag = t.tag;
    cb.row.style = t.style;
    cb.row.scored = true;
    cb.row.change = change;
    cb.row.how = "bridge";
    cb.row.memory = static_cast<int>(before.size());
    cb.row.splice_seconds = std::chrono::duration<double>(tb - t1).count();
    cb.row.bridge_scores = r.scores;
    cb.row.lyrics = lyrics;
    cb.row.knobs = t.knobs;
    cb.target = t;
    cf.row = cb.row;
    cf.row.index = k + 1;
    cf.row.change = false;
    cf.row.how = "fresh";
    cf.row.memory = 0;
    cf.row.splice_seconds = std::chrono::duration<double>(t1 - t0).count();
    cf.row.bridge_scores.clear();
    cf.row.phase_fit = true;
    cf.row.period = P0;
    cf.row.stream_phase = sb.phase;
    cf.row.fresh_phase = fb.phase;
    cf.row.cut = cut;
    cf.row.fit_seconds = fit_seconds;
    cf.target = t;
    cf.fresh = true;
    // The head over the bridge, then over the clip as cut (one window for both).
    read_lyrics(roll, cb, words, r.watched, 0);
    const int reached_b = roll.reached;
    read_lyrics(roll, cf, words, fresh.watched, cut);
    l.roll = std::move(roll);
    fold(l, k + 1);
    est_bridge = 0.7 * est_bridge + 0.3 * since(t0);
    add_chunk(l, std::move(cb));
    lyrics_event(l, k, reached_b);
    add_chunk(l, std::move(cf));
    lyrics_event(l, k + 1, l.roll.reached);
    {
        Yue2EndlessEvent e{Yue2EndlessEvent::Kind::StyleChange};
        e.chunk = k;
        e.branch = l.id;
        e.tag = t.tag;
        e.text = "bridge";
        e.sample = static_cast<std::int64_t>(k) * C * kR;
        event(e);
        Yue2EndlessEvent f{Yue2EndlessEvent::Kind::PhaseFit};
        f.chunk = k + 1;
        f.branch = l.id;
        f.tag = t.tag;
        f.value = cut;
        f.seconds = fit_seconds;
        f.sample = static_cast<std::int64_t>(k + 1) * C * kR;
        f.row = std::make_shared<Yue2EndlessChunk>(l.chunk(k + 1).row);
        event(f);
    }
    // The probe is the render of the chunks it covered.
    l.rendered = k + 1;
    for (int j = from; j <= k; ++j) {
        l.chunk(j).row.probe_render = true;
        l.chunk(j).row.nar_window = static_cast<int>(win.size());
        built(l, j, probe_seconds / (k + 1 - from));
    }
    return true;
}

}  // namespace brosoundml
