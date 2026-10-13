#define _CRT_SECURE_NO_WARNINGS
// Yue2Stream previews on the GPU (minutes; runs only with YUE2_E2E=1):
// preview_seconds 30, three previews kept built ahead, fed through Want (the
// request is ../YuE2-3B/examples/tonight-awake.json or $YUE2_REQUEST, a new
// seed per song), paced by a simulated real-time player:
//   1. the previews ahead are built while the first one plays, before any
//      skip: the next one before the first preview ends, all three within
//      about a minute (each about 11 s of GPU);
//   2. a preview nobody chose ends at its length and crossfades into the
//      next, with no underrun;
//   3. make_full() 15 s into the third: a Full event, the playing song's
//      continuation built before any other song's work, its first new
//      latents past the preview's end long before the playhead gets there,
//      the audio contiguous through the join with no step at it (spectral
//      flux at the join against the flux around it, the research's switch
//      measure, and the click ratio), and the song plays on past 30 s;
//   4. skip() during the full song cuts over to the next built preview in
//      well under a quarter second;
//   5. save_song(): song 0 saved after it played (its 30 s, as it played),
//      the full song saved at the press and written once the skip has
//      played it out (past its preview), an unknown song refused.
// Numbers are reported and asserted; nothing judges the audio by ear.

#include "test_yue2_stream_common.h"

#include <complex>

using namespace yue2_stream_test;

namespace {

// Log-magnitude spectral flux of the mono mix over [a, b) program samples,
// one value per 512-sample hop (2048-sample Hann frames).
std::vector<double> spectral_flux(const std::vector<float>& L, const std::vector<float>& R, std::int64_t a,
                                  std::int64_t b) {
    constexpr int N = 2048, H = 512;
    std::vector<double> win(N);
    for (int i = 0; i < N; ++i) win[i] = 0.5 - 0.5 * std::cos(2 * 3.14159265358979 * i / N);
    auto fft = [](std::vector<std::complex<double>>& x) {
        const std::size_t n = x.size();
        for (std::size_t i = 1, j = 0; i < n; ++i) {
            std::size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(x[i], x[j]);
        }
        for (std::size_t len = 2; len <= n; len <<= 1) {
            const double ang = -2 * 3.14159265358979 / static_cast<double>(len);
            const std::complex<double> wl(std::cos(ang), std::sin(ang));
            for (std::size_t i = 0; i < n; i += len) {
                std::complex<double> w(1);
                for (std::size_t k = 0; k < len / 2; ++k) {
                    const auto u = x[i + k], v = x[i + k + len / 2] * w;
                    x[i + k] = u + v;
                    x[i + k + len / 2] = u - v;
                    w *= wl;
                }
            }
        }
    };
    std::vector<double> flux, prev;
    a = std::max<std::int64_t>(0, a);
    b = std::min<std::int64_t>(static_cast<std::int64_t>(L.size()), b);
    for (std::int64_t s = a; s + N <= b; s += H) {
        std::vector<std::complex<double>> x(N);
        for (int i = 0; i < N; ++i) x[i] = 0.5 * (double(L[s + i]) + double(R[s + i])) * win[i];
        fft(x);
        std::vector<double> mag(N / 2);
        for (int k = 0; k < N / 2; ++k) mag[k] = std::log1p(10.0 * std::abs(x[k]));
        if (!prev.empty()) {
            double f = 0;
            for (int k = 0; k < N / 2; ++k) f += std::max(0.0, mag[k] - prev[k]);
            flux.push_back(f);
        } else {
            flux.push_back(0);
        }
        prev = std::move(mag);
    }
    return flux;
}

void section_previews(Env& E) {
    constexpr int kAhead = 3;
    constexpr double kPreview = 30.0;
    std::printf("Previews: %.0f s each, %d kept built ahead\n", kPreview, kAhead);
    std::atomic<int> fed{0};
    auto make = [&](int n) {
        brosoundml::Yue2Song s;
        s.params = E.req;
        s.params.seed = E.req.seed + 101 * static_cast<std::uint64_t>(n);
        s.tag = "P" + std::to_string(n);
        return s;
    };
    VramMeter vm;
    vm.start();
    Capture cap;
    brosoundml::Yue2StreamOptions o;
    o.paced = true;
    o.radio = false;
    o.ahead_songs = kAhead;
    o.preview_seconds = kPreview;
    cap.hook(o);
    std::unique_ptr<Yue2Stream> stp;
    auto log = cap.event_fn(true);
    auto on_event = [&](const Yue2StreamEvent& e) {
        log(e);
        if (e.kind == Kind::Want)
            for (int i = 0; i < static_cast<int>(e.value); ++i) stp->enqueue(make(++fed));
    };
    stp = std::make_unique<Yue2Stream>(E.y, E.vae, o, cap.audio_fn(), on_event);
    Yue2Stream& st = *stp;
    st.enqueue(make(0));
    std::thread worker([&] { st.run(); });

    std::int64_t play = 0;
    double last = -1, underrun = 0;
    const int L = static_cast<int>(kPreview * 25);
    double pool_at = -1, press_wall = -1, past_preview_wall = -1, skip_wall = -1;
    std::int64_t press_play = 0, past_preview_play = 0;
    int full_song = -1, skip_to = -1, stage = 0, best_ahead = 0;
    std::int64_t full_start = -1;
    const fs::path tmp = fs::temp_directory_path();
    const fs::path wav0 = tmp / "yue2_stream_save0.wav", wavf = tmp / "yue2_stream_savefull.wav";
    bool saved0 = false;
    const double t0 = now_s(), t_end = t0 + 400;
    while (now_s() < t_end && stage < 5) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const double tn = now_s();
        if (cap.first_audio_at >= 0) {
            if (last < 0) last = tn;
            const std::int64_t want = play + static_cast<std::int64_t>((tn - last) * kSr);
            last = tn;
            const std::int64_t have = cap.emitted;
            if (want > have) underrun += double(want - have) / kSr;
            play = std::min(want, have);
            st.set_playhead(play);
        }
        if (cap.find(Kind::Error)) break;
        const brosoundml::Yue2StreamStats s = st.stats();
        const brosoundml::Yue2StreamStats::Song* air = nullptr;
        for (const auto& g : s.songs)
            if (g.on_air) air = &g;   // the latest on air (a crossfade's incoming song)
        bool all_built = s.ahead >= kAhead;
        int built_ahead = 0;
        for (const auto& g : s.songs)
            if (!g.on_air) {
                all_built = all_built && g.built && g.preview;
                built_ahead += g.built && g.preview;
            }
        if (stage <= 1) best_ahead = std::max(best_ahead, built_ahead);
        if (pool_at < 0 && all_built) {
            pool_at = tn - t0;
            std::printf("  %d previews built ahead %.1f s after the start, playhead %.1f s (no skip yet); host %.1f MB\n",
                        s.ahead, pool_at, play / double(kSr), s.host_bytes / 1e6);
        }
        if (stage == 0) stage = 1;
        // 5. Song 0 has played: its audio is held for a save.
        if (!saved0 && cap.find(Kind::SongEnd, 0)) {
            saved0 = true;
            st.save_song(0, wav0.string());
            st.save_song(999, (tmp / "yue2_stream_save999.wav").string());
        }
        // 3. 15 s into the third song (two played out as previews).
        if (stage == 1 && air && air->index == 2 && air->start >= 0 && play >= air->start + 15 * kSr) {
            full_song = 2;
            full_start = air->start;
            press_wall = tn;
            press_play = play;
            st.make_full(-1);
            st.save_song(full_song, wavf.string());
            stage = 2;
        }
        if (stage == 2) {
            for (const auto& g : s.songs)
                if (g.index == full_song && g.latent_frames > L && past_preview_wall < 0) {
                    past_preview_wall = tn;
                    past_preview_play = play;
                }
            if (past_preview_wall >= 0 && play >= full_start + static_cast<std::int64_t>((kPreview + 8) * kSr)) stage = 3;
        }
        // 4. skip during the full song.
        if (stage == 3) {
            for (const auto& g : s.songs)
                if (!g.on_air && g.index > full_song) {
                    skip_to = g.index;
                    break;
                }
            skip_wall = tn;
            st.skip();
            stage = 4;
        }
        if (stage == 4) {
            std::lock_guard<std::mutex> lk(cap.mu);
            const auto* ss = cap.find_last(Kind::SongStart, skip_to);
            if (ss && ss->at >= skip_wall && play >= ss->e.sample + 2 * kSr && cap.find(Kind::Saved, full_song))
                stage = 5;
        }
    }
    st.stop();
    worker.join();
    const double peak = vm.stop_gb();
    CHECK(!cap.find(Kind::Error), "previews: the stream reported an error");
    CHECK(stage == 5, "previews: stopped at stage %d", stage);
    CHECK(cap.contiguous, "previews: program audio not contiguous");

    // 1. Built ahead before any skip, while the first preview played.
    // A preview plays 27 s (its crossfade overlaps the next) and builds in
    // about 11 s, so the pool grows by about 1.3 per preview played: all
    // three built ahead about a minute in, before the third song's 15 s.
    std::printf("  before make full: %d previews built ahead at most; all %d built %s\n", best_ahead, kAhead,
                pool_at > 0 ? (std::to_string(pool_at) + " s after the start").c_str() : "not reached");
    CHECK(best_ahead >= kAhead && pool_at > 0 && pool_at < 75, "previews: %d built ahead before make full (all at %.1f s)",
          best_ahead, pool_at);
    const auto* b1 = cap.find(Kind::Built, 1);
    const auto* end0 = cap.find(Kind::SongEnd, 0);
    CHECK(b1 && end0 && b1->at < end0->at, "previews: song 1 was not built before song 0's preview ended");
    if (b1 && cap.first_audio_at >= 0)
        std::printf("  song 1's preview built %.1f s after the first audio\n", b1->at - cap.first_audio_at);

    // 2. The unchosen first preview ends at its length, crossfading.
    const auto* st0 = cap.find(Kind::SongStart, 0);
    const auto* st1 = cap.find(Kind::SongStart, 1);
    if (st0 && end0 && st1) {
        const double len0 = double(end0->e.sample - st0->e.sample) / kSr;
        const double overlap = double(end0->e.sample - st1->e.sample) / kSr;
        std::printf("  song 0 (preview, not chosen): %.2f s, crossfading %.2f s into song 1; click %.2f\n", len0,
                    overlap, click_ratio(cap.L, st1->e.sample));
        CHECK(len0 > kPreview - 1 && len0 < kPreview + 1, "previews: an unchosen preview played %.2f s", len0);
        CHECK(overlap >= 1.9 && overlap <= 4.1, "previews: crossfade %.2f s", overlap);
    } else {
        CHECK(false, "previews: song 0's start / end or song 1's start missing");
    }

    // 3. make full.
    const Capture::Ev* full = nullptr;
    for (const auto& e : cap.events)
        if (e.e.kind == Kind::Full && e.e.song == full_song) full = &e;
    CHECK(full && full->e.value == 1, "previews: no accepted Full event for song %d", full_song);
    if (full) {
        const int join = full->e.frame0;
        // Nothing else built between the request and the continuation's first window.
        const Capture::Ev* first_cont = nullptr;
        bool others = false;
        for (const auto& e : cap.events) {
            if (e.at < full->at) continue;
            if (e.e.kind == Kind::Window && e.e.song == full_song && e.e.frame0 >= join) {
                first_cont = &e;
                break;
            }
            if ((e.e.kind == Kind::Window || e.e.kind == Kind::Plan) && e.e.song != full_song) others = true;
        }
        CHECK(first_cont, "previews: the continuation's first window never landed");
        CHECK(!others, "previews: another song's work ran before the playing song's continuation");
        const double margin = double(full_start + static_cast<std::int64_t>(L) * kHop - past_preview_play) / kSr;
        std::printf("  make full at %.1f s into song %d: join frame %d (%.2f s); first continued window %.2f s after "
                    "the press; latents past the preview's end %.2f s after it, %.1f s before the playhead got there\n",
                    double(press_play - full_start) / kSr, full_song, join, join / 25.0,
                    first_cont ? first_cont->at - press_wall : -1.0, past_preview_wall - press_wall, margin);
        CHECK(past_preview_wall > 0 && margin > 3.0, "previews: the continuation reached the preview's end only %.1f s "
                                                     "before the playhead", margin);
        // The join, as heard: flux at it against the 10 s around it.
        const std::int64_t at = full_start + static_cast<std::int64_t>(join) * kHop;
        const auto flux = spectral_flux(cap.L, cap.R, at - 5 * kSr, at + 5 * kSr);
        if (flux.size() > 40) {
            const std::size_t c = static_cast<std::size_t>(5 * kSr / 512);
            double mean = 0, sd = 0, mx = 0;
            std::size_t n = 0;
            for (std::size_t i = 1; i < flux.size(); ++i)
                if (i + 3 < c || i > c + 3) mean += flux[i], ++n;
            mean /= std::max<std::size_t>(1, n);
            for (std::size_t i = 1; i < flux.size(); ++i)
                if (i + 3 < c || i > c + 3) sd += (flux[i] - mean) * (flux[i] - mean);
            sd = std::sqrt(sd / std::max<std::size_t>(1, n));
            for (std::size_t i = c - 2; i <= c + 2 && i < flux.size(); ++i) mx = std::max(mx, flux[i]);
            std::size_t below = 0;
            for (std::size_t i = 1; i < flux.size(); ++i) below += flux[i] < mx;
            const double z = (mx - mean) / (sd + 1e-9);
            std::printf("  join: flux z %.2f (percentile %.0f of the 10 s around it), click %.2f\n", z,
                        100.0 * below / std::max<std::size_t>(1, flux.size() - 1), click_ratio(cap.L, at));
            CHECK(z < 3.0, "previews: a step at the join (flux z %.2f)", z);
            CHECK(click_ratio(cap.L, at) < 2.0, "previews: a click at the join");
        }
        const auto* endf = cap.find(Kind::SongEnd, full_song);
        CHECK(!endf || double(endf->e.sample - full_start) / kSr > kPreview + 5,
              "previews: the full song ended at its preview's length");
    }

    // 4. Skip during the full song.
    if (skip_to >= 0) {
        const auto* ss = cap.find_last(Kind::SongStart, skip_to);
        const double cut = ss ? ss->e.seconds : -1;
        std::printf("  skip during the full song -> preview %d: cut over %.3f s\n", skip_to, cut);
        CHECK(cut >= 0 && cut < 0.25, "previews: skip to a built preview took %.3f s", cut);
    }
    // 5. Saves: the file holds exactly the frames the event reports.
    auto saved = [&](int song, const fs::path& p, const char* what) {
        const auto* e = cap.find(Kind::Saved, song);
        CHECK(e && e->e.value >= 0 && e->e.text == p.string(), "previews: %s not saved (%s)", what,
              e ? e->e.text.c_str() : "no Saved event");
        if (!e || e->e.value < 0) return 0.0;
        const auto bytes = fs::exists(p) ? fs::file_size(p) : 0;
        CHECK(bytes == 44 + static_cast<std::uintmax_t>(e->e.frames) * 4, "previews: %s: %llu bytes for %d frames", what,
              static_cast<unsigned long long>(bytes), e->e.frames);
        // Its take covers the audio written: a token, 64 latents and an energy per frame.
        const auto& t = e->e.take;
        const int tf = t ? static_cast<int>(t->semantic.size()) : 0;
        CHECK(t && !t->plan.prefix.empty() && t->latents.size() == static_cast<std::size_t>(tf) * 64 &&
                  t->energy.size() == static_cast<std::size_t>(tf) && std::abs(tf * kHop - e->e.frames) <= 2 * kHop,
              "previews: %s: take of %d frames for %d samples", what, tf, e->e.frames);
        fs::remove(p);
        return e->e.frames / double(kSr);
    };
    const double len0 = saved(0, wav0, "song 0 (played)");
    const double lenf = saved(full_song, wavf, "the full song (saved at the press, cut by the skip)");
    const auto* none = cap.find(Kind::Saved, 999);
    std::printf("  saved: song 0 %.2f s, the full song %.2f s; an unknown song: %s\n", len0, lenf,
                none ? none->e.text.c_str() : "no event");
    CHECK(len0 > kPreview - 1 && len0 < kPreview + 1, "previews: song 0 saved %.2f s", len0);
    CHECK(lenf > kPreview + 5, "previews: the full song saved %.2f s, not past its preview", lenf);
    CHECK(none && none->e.value < 0, "previews: a save of an unknown song was not refused");
    std::printf("  underrun %.2f s; peak device memory %.2f GB\n", underrun, peak);
    CHECK(underrun < 0.5, "previews: %.2f s of underrun", underrun);
}

}  // namespace

int main() {
    auto Env_ = std::make_unique<Env>();
    Env& E = *Env_;
    if (!load_env(E)) return 0;
    const double t = now_s();
    section_previews(E);
    std::printf("  (%.1f s)\n", now_s() - t);
    std::printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
