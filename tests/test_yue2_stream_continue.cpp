#define _CRT_SECURE_NO_WARNINGS
// A preview continued into the whole song, against the whole song (minutes;
// runs only with YUE2_E2E=1). The yue2-lab radio's case: song 1 is built
// ahead as a 30 s preview while song 0 plays, and is made full as it goes on
// air; it then plays to its end, paced by a simulated real-time player.
//   - its semantic tokens equal Yue2::generate's for the same request (the
//     session resumed after the preview samples what one run would have);
//   - its latents, per window, against the whole-song render of those tokens
//     (Yue2::synthesize, one chunk): the cosine per frame and the mean latent
//     norm, so a window that breaks shows as its own row; no window's norm
//     may sit 0.8 over the whole render's (a short window solved ~9.4
//     against ~8.0: the continuation's first window and a song's end).
// The request is $YUE2_REQUEST or tonight-awake.json, its semantic stage
// capped at $YUE2_CONT_TOKENS (default 2000, 80 s) to keep the run short.

#include "test_yue2_stream_common.h"

using namespace yue2_stream_test;

namespace {

double cos_frames(const std::vector<float>& a, const std::vector<float>& b, int f0, int f1) {
    double s = 0;
    int n = 0;
    for (int f = f0; f < f1; ++f) {
        double ab = 0, aa = 0, bb = 0;
        for (int d = 0; d < 64; ++d) {
            const double x = a[static_cast<std::size_t>(f) * 64 + d], y = b[static_cast<std::size_t>(f) * 64 + d];
            ab += x * y, aa += x * x, bb += y * y;
        }
        s += ab / (std::sqrt(aa * bb) + 1e-12);
        ++n;
    }
    return n ? s / n : 0;
}

double norm_frames(const std::vector<float>& a, int f0, int f1) {
    double s = 0;
    for (int f = f0; f < f1; ++f) {
        double q = 0;
        for (int d = 0; d < 64; ++d) q += double(a[static_cast<std::size_t>(f) * 64 + d]) * a[static_cast<std::size_t>(f) * 64 + d];
        s += std::sqrt(q);
    }
    return f1 > f0 ? s / (f1 - f0) : 0;
}

}  // namespace

int main() {
    auto Env_ = std::make_unique<Env>();
    Env& E = *Env_;
    if (!load_env(E)) return 0;
    const int cap_tokens = std::stoi(env("YUE2_CONT_TOKENS", "2000"));
    auto song = [&](int n) {
        brosoundml::Yue2Song s;
        s.params = E.req;
        s.params.seed = E.req.seed + 101 * static_cast<std::uint64_t>(n);
        s.params.semantic_sampling.max_tokens = cap_tokens;
        s.tag = "S" + std::to_string(n);
        return s;
    };
    const brosoundml::Yue2Song s1 = song(1);
    Capture cap;
    brosoundml::Yue2StreamOptions o;
    o.paced = true;
    o.radio = false;
    o.ahead_songs = 1;
    o.preview_seconds = 30;
    cap.hook(o);
    Yue2Stream st(E.y, E.vae, o, cap.audio_fn(), cap.event_fn(true));
    st.enqueue(song(0));
    st.enqueue(s1);
    std::thread worker([&] { st.run(); });
    std::int64_t play = 0;
    double last = -1, underrun = 0;
    bool kept = false;
    const double t0 = now_s();
    while (now_s() < t0 + 400) {
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
        std::lock_guard<std::mutex> lk(cap.mu);
        if (cap.find(Kind::Error)) break;
        const auto* on1 = cap.find(Kind::SongStart, 1);   // placed: its program start
        if (!kept && on1 && cap.find(Kind::Built, 1) && play >= on1->e.sample) {
            st.make_full(1);
            kept = true;
            std::printf("  make full song 1 as it goes on air (playhead %.1f s)\n", play / double(kSr));
        }
        if (cap.find(Kind::SongEnd, 1)) break;
    }
    st.stop();
    worker.join();
    CHECK(!cap.find(Kind::Error), "the stream reported an error");
    CHECK(cap.find(Kind::SongEnd, 1), "song 1 never played to its end");
    const auto* full = cap.find(Kind::Full, 1);
    CHECK(full && full->e.value == 1, "no accepted Full event for song 1");
    const int join = full ? full->e.frame0 : 0;
    std::printf("  join frame %d (%.2f s); underrun %.2f s\n", join, join / 25.0, underrun);

    // Tokens: the continued song against one run of the same request.
    brosoundml::Yue2Params p = s1.params;
    p.run_synthesize = false;
    const brosoundml::Yue2Result r = E.y.generate(p);
    const std::vector<int32_t> sem = cap.sem.size() > 1 ? cap.sem[1] : std::vector<int32_t>{};
    std::size_t first_diff = std::min(sem.size(), r.semantic.size());
    for (std::size_t i = 0; i < first_diff; ++i)
        if (sem[i] != r.semantic[i]) {
            first_diff = i;
            break;
        }
    std::printf("  tokens: stream %zu, generate %zu, equal up to %zu\n", sem.size(), r.semantic.size(), first_diff);
    CHECK(sem == r.semantic, "the continued tokens differ from generate's from token %zu (%.2f s)", first_diff,
          first_diff / 25.0);

    // Latents, per window, against the whole-song render of the stream's tokens.
    const int T = static_cast<int>(sem.size());
    const std::vector<float> whole = E.y.synthesize(s1.params, E.y.plan(s1.params), sem);
    const std::vector<float>& lat = cap.lat.size() > 1 ? cap.lat[1] : std::vector<float>{};
    CHECK(static_cast<int>(lat.size()) >= T * 64 && static_cast<int>(whole.size()) >= T * 64,
          "latents: stream %zu, whole %zu floats for %d frames", lat.size(), whole.size(), T);
    if (static_cast<int>(lat.size()) >= T * 64 && static_cast<int>(whole.size()) >= T * 64) {
        std::vector<std::pair<int, int>> wins;   // the latest solve of each frame: windows landing in order
        for (const auto& e : cap.events)
            if (e.e.kind == Kind::Window && e.e.song == 1) {
                const int a = e.e.frame0, b = e.e.frame0 + e.e.frames;
                while (!wins.empty() && wins.back().second > a) {
                    if (wins.back().first >= a) wins.pop_back();
                    else { wins.back().second = a; break; }
                }
                wins.emplace_back(a, b);
            }
        double worst = 1;
        std::printf("  %-13s %8s %10s %10s\n", "window (s)", "cos", "norm", "whole");
        for (const auto& [a, b] : wins) {
            const double c = cos_frames(lat, whole, a, b), nn = norm_frames(lat, a, b), nw = norm_frames(whole, a, b);
            std::printf("  %5.1f-%5.1f  %8.3f %10.2f %10.2f\n", a / 25.0, b / 25.0, c, nn, nw);
            if (a >= join) worst = std::min(worst, c);
            // A short window solved loud (~9.4 against ~8.0, past the corpus
            // range 6.5..9.5): no window may stray that far from the whole render.
            CHECK(nn < nw + 0.8, "window %.1f-%.1f s solved loud: norm %.2f, the whole render %.2f", a / 25.0, b / 25.0,
                  nn, nw);
        }
        std::printf("  whole song: cos %.3f; past the join %.3f (worst window %.3f)\n", cos_frames(lat, whole, 0, T),
                    cos_frames(lat, whole, join, T), worst);
        const fs::path out = env("YUE2_CONT_OUT");
        if (!out.empty()) {
            fs::create_directories(out);
            write_raw(out / "stream_latents.f32", lat);
            write_raw(out / "whole_latents.f32", whole);
            write_raw(out / "semantic.i32", sem);
            write_raw(out / "program_L.f32", cap.L);
        }
    }
    std::printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
