#define _CRT_SECURE_NO_WARNINGS
// Yue2Stream on the GPU (minutes; runs only with YUE2_E2E=1): the streaming
// player, the radio's crossfade, knobs that change only songs queued after
// them, and the bug guards.
// The request is ../YuE2-3B/examples/tonight-awake.json (or $YUE2_REQUEST);
// the given-semantic sections read yue2-research's rock_gen_W0 run (or
// $YUE2_STREAM_FROM: plan.json + semantic.i32). $YUE2_STREAM_SECTIONS picks
// sections (default "ABCDEF").
//
//   A. Quality: the stream (given plan + semantic, unpaced) against the
//      whole-song render of the same tokens (Yue2::synthesize): latent cosine
//      per frame (the research's design window: 0.77-0.84), the latent jump at
//      window seams, program length = the decoder's natural length. With
//      $YUE2_STREAM_OUT both are written as seams.py run dirs (stream_W250 and
//      stream_syn_W0: latents.f32, audio.f32 planar, run.json with the seams).
//   B. The stream's own semantic tokens equal Yue2::generate's (same seed),
//      and the time to first audio from the request (cot full).
//   C. Paced, a simulated player: knobs moved at 8 s change no song that
//      exists (one Knobs event, every song keeps its own) and a song queued
//      after takes them; then skip() to a queued second song, which on air is
//      decoded whole as soon as its latents are, while a third is built ahead
//      in one NAR chunk and crossfades in; no underrun.
//   D. The radio's crossfade of two given-semantic songs: placement, program
//      length, no power dip across the fade; the second, built ahead, is the
//      whole-song render (one chunk) and sets the peak device memory.
//   E. B1: non-finite logits raise "non-finite logits" instead of sampling an
//      out-of-range token, and the model stays usable.
//   F. B2: an arrangement steer of 4 (the research's never-ending song) ends
//      by the length guard's hard cap.
//   G. Three songs built ahead, fed through Want events (paced, a simulated
//      player): the pool fills with wholly decoded songs; a texture knob
//      moved then leaves them ready; skip() to one cuts over and hands out
//      its audio within a quarter second and it plays as built; the songs
//      fed after the move take the knob; drop() removes the last and Want
//      asks again. Device memory with
//      the pool full, and the host memory the waiting songs take.
// Peak device memory (device_mem_info, sampled) is reported per section.
// Numbers are reported and asserted; nothing judges the audio by ear.

#include "test_yue2_stream_common.h"

using namespace yue2_stream_test;

namespace {

double mean_cos(const std::vector<float>& a, const std::vector<float>& b, int frames) {
    double s = 0;
    for (int t = 0; t < frames; ++t) {
        double ab = 0, aa = 0, bb = 0;
        for (int d = 0; d < 64; ++d) {
            const double x = a[t * 64 + d], y = b[t * 64 + d];
            ab += x * y, aa += x * x, bb += y * y;
        }
        s += ab / (std::sqrt(aa * bb) + 1e-12);
    }
    return s / std::max(1, frames);
}

// Frame-to-frame latent jump at the given seams over the median jump.
double seam_jump(const std::vector<float>& z, int frames, const std::vector<int>& seams) {
    auto jump = [&](int t) {
        double s = 0;
        for (int d = 0; d < 64; ++d) {
            const double e = z[t * 64 + d] - z[(t - 1) * 64 + d];
            s += e * e;
        }
        return std::sqrt(s);
    };
    std::vector<double> all;
    for (int t = 1; t < frames; ++t) all.push_back(jump(t));
    if (all.empty() || seams.empty()) return 0;
    std::nth_element(all.begin(), all.begin() + all.size() / 2, all.end());
    const double med = all[all.size() / 2];
    double s = 0;
    int n = 0;
    for (int t : seams)
        if (t > 0 && t < frames) s += jump(t), ++n;
    return n ? s / n / (med + 1e-12) : 0;
}

bool finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

void write_run(const fs::path& dir, const std::string& name, const std::vector<float>& lat, int frames,
               const std::vector<float>& L, const std::vector<float>& R, const std::vector<int>& seams) {
    fs::create_directories(dir);
    write_raw(dir / "latents.f32", std::vector<float>(lat.begin(), lat.begin() + static_cast<std::size_t>(frames) * 64));
    std::vector<float> planar(L);
    planar.insert(planar.end(), R.begin(), R.end());
    write_raw(dir / "audio.f32", planar);
    std::ofstream f((dir / "run.json").string());
    f << "{\n \"name\": \"" << name << "\",\n \"frames\": " << frames << ",\n \"seams\": [";
    for (std::size_t i = 0; i < seams.size(); ++i) f << (i ? ", " : "") << seams[i];
    f << "]\n}\n";
}

brosoundml::Yue2Song given_song(const Env& E, int f0, int frames, const std::string& tag) {
    brosoundml::Yue2Song s;
    s.params = E.given;
    s.tag = tag;
    const int n = std::min<int>(frames, static_cast<int>(E.given_sem.size()) - f0);
    s.semantic.assign(E.given_sem.begin() + f0, E.given_sem.begin() + f0 + n);
    return s;
}

// ── A ──
void section_a(Env& E) {
    const int frames = std::atoi(env("YUE2_STREAM_FRAMES", "1500").c_str());
    std::printf("A. stream vs whole-song render (%d frames of given semantic)\n", frames);
    brosoundml::Yue2Song song = given_song(E, 0, frames, "A");
    const int T = static_cast<int>(song.semantic.size());

    VramMeter vm;
    vm.start();
    Capture cap;
    brosoundml::Yue2StreamOptions o;
    o.paced = false;
    o.radio = false;
    cap.hook(o);
    double t = now_s();
    {
        Yue2Stream st(E.y, E.vae, o, cap.audio_fn(), cap.event_fn(false));
        st.enqueue(song);
        st.run();
    }
    const double stream_s = now_s() - t;
    const double peak = vm.stop_gb();
    CHECK(!cap.find(Kind::Error), "A: the stream reported an error");
    CHECK(cap.contiguous, "A: program audio not contiguous");
    CHECK(cap.lat.size() == 1 && static_cast<int>(cap.lat[0].size()) == T * 64, "A: stream latents cover %d frames",
          T);
    const std::int64_t natural = E.vae.output_length(T);
    CHECK(static_cast<std::int64_t>(cap.L.size()) == natural, "A: program %zu samples, natural %lld", cap.L.size(),
          static_cast<long long>(natural));
    CHECK(finite(cap.L) && finite(cap.R), "A: non-finite audio");
    if (cap.lat.empty() || static_cast<int>(cap.lat[0].size()) != T * 64) return;

    // The whole-song render of the same tokens.
    brosoundml::Yue2Params p = song.params;
    const brosoundml::Yue2Plan plan = E.y.plan(p);
    t = now_s();
    const std::vector<float> whole = E.y.synthesize(p, plan, song.semantic);
    const double whole_s = now_s() - t;
    CHECK(static_cast<int>(whole.size()) == T * 64, "A: whole render %zu floats", whole.size());
    if (static_cast<int>(whole.size()) != T * 64) return;
    const StereoAudio wa = E.vae.decode_tiled(whole.data(), T);

    std::vector<int> seams;
    for (const auto& [f0, n] : cap.windows)
        if (f0 > 0) seams.push_back(f0);
    std::sort(seams.begin(), seams.end());
    seams.erase(std::unique(seams.begin(), seams.end()), seams.end());
    const double cs = mean_cos(cap.lat[0], whole, T);
    const double js = seam_jump(cap.lat[0], T, seams), jr = seam_jump(whole, T, seams);
    std::printf("  stream %.1f s (%.3f s per audio s), whole %.1f s; %zu windows\n", stream_s, stream_s / (T / 25.0),
                whole_s, cap.windows.size());
    std::printf("  latent cosine %.3f (research design window 0.77-0.84); seam jump %.2f (whole %.2f)\n", cs, js, jr);
    std::printf("  peak device memory %.2f GB\n", peak);
    CHECK(cs >= 0.65, "A: latent cosine %.3f < 0.65", cs);
    CHECK(js <= 2.0 * jr + 0.5, "A: seam jump %.2f vs %.2f", js, jr);

    if (const std::string out = env("YUE2_STREAM_OUT"); !out.empty()) {
        const std::string v = E.song + "_stream_W250", r = E.song + "_syn_W0";
        write_run(fs::path(out) / v, v, cap.lat[0], T, cap.L, cap.R, seams);
        write_run(fs::path(out) / r, r, whole, T,
                  std::vector<float>(wa.channel(0), wa.channel(0) + wa.frames),
                  std::vector<float>(wa.channel(1), wa.channel(1) + wa.frames), {});
        std::printf("  run dirs in %s (seams.py %s)\n", out.c_str(), out.c_str());
    }
}

// ── B ──
void section_b(Env& E) {
    const int n = std::atoi(env("YUE2_STREAM_TOKENS", "600").c_str());
    std::printf("B. the stream's semantic tokens vs generate's (%d tokens, cot full)\n", n);
    brosoundml::Yue2Song song;
    song.params = E.req;
    song.params.semantic_sampling.min_tokens = n;
    song.params.semantic_sampling.max_tokens = n;
    song.tag = "B";
    VramMeter vm;
    vm.start();
    Capture cap;
    brosoundml::Yue2StreamOptions o;
    o.paced = false;
    o.radio = false;
    cap.hook(o);
    const double t = now_s();
    {
        Yue2Stream st(E.y, E.vae, o, cap.audio_fn(), cap.event_fn(true));
        st.enqueue(song);
        st.run();
    }
    const double peak = vm.stop_gb();
    CHECK(!cap.find(Kind::Error), "B: the stream reported an error");
    const double ttfa = cap.first_audio_at >= 0 ? cap.first_audio_at - t : -1;
    const auto* fa = cap.find(Kind::FirstAudio);
    const auto* pe = cap.find(Kind::Plan);
    const double plan_s = pe ? pe->at - t : -1;
    std::printf("  time to first audio %.2f s (event %.2f s; research about 4.3 s = plan 2.4 + 1.9): plan %.2f s, "
                "after the plan %.2f s; peak %.2f GB\n",
                ttfa, fa ? fa->e.seconds : -1.0, plan_s, ttfa - plan_s, peak);
    CHECK(ttfa > 0 && ttfa < 8.0, "B: time to first audio %.2f s", ttfa);
    CHECK(plan_s > 0 && ttfa - plan_s < 3.0, "B: %.2f s from the plan to first audio", ttfa - plan_s);

    brosoundml::Yue2Params p = song.params;
    p.run_synthesize = false;
    const brosoundml::Yue2Result r = E.y.generate(p);
    CHECK(cap.sem.size() == 1 && cap.sem[0] == r.semantic, "B: stream tokens (%zu) differ from generate's (%zu)",
          cap.sem.empty() ? 0 : cap.sem[0].size(), r.semantic.size());
    if (!cap.sem.empty() && cap.sem[0].size() == r.semantic.size()) {
        int same = 0;
        for (std::size_t i = 0; i < r.semantic.size(); ++i) same += cap.sem[0][i] == r.semantic[i];
        std::printf("  tokens equal: %d of %zu\n", same, r.semantic.size());
    }
    E.req.abc = r.plan.abc;   // later sections reuse the plan
}

// ── C ──
void section_c(Env& E) {
    std::printf("C. paced: a knob moved at 8 s changes no song that exists, then skip to a queued song\n");
    brosoundml::Yue2Song s0;
    s0.params = E.req;
    s0.params.semantic_sampling.max_tokens = 1000;
    s0.tag = "C0";
    brosoundml::Yue2Song s1 = given_song(E, 1000, 400, "C1");
    brosoundml::Yue2Song s2 = given_song(E, 0, 300, "C2");   // built ahead while song 1 plays

    VramMeter vm;
    vm.start();
    Capture cap;
    brosoundml::Yue2StreamOptions o;
    o.paced = true;
    o.radio = false;
    cap.hook(o);
    Yue2Stream st(E.y, E.vae, o, cap.audio_fn(), cap.event_fn(true));
    st.enqueue(s0);
    st.enqueue(s1);
    st.enqueue(s2);
    std::thread worker([&] { st.run(); });

    // The player: plays from the first audio at real time; an underrun waits.
    std::int64_t play = 0, knob_at = -1, skip_at = -1;
    double underrun = 0, last = -1;
    const double t_end = now_s() + 150;
    bool stopped = false, queued_after = false;
    // Song 1 (given tokens) on air: decoded whole as soon as its latents are,
    // not a lead ahead of the playhead; the program sample it was seen built.
    std::int64_t built1_at = -1, start1 = -1;
    brosoundml::Yue2StreamStats after_knob;
    while (!stopped && now_s() < t_end) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const double tn = now_s();
        if (cap.first_audio_at < 0) continue;
        if (last < 0) last = tn;
        const std::int64_t want = play + static_cast<std::int64_t>((tn - last) * kSr);
        last = tn;
        const std::int64_t have = cap.emitted;
        if (want > have && knob_at >= 0) underrun += double(want - have) / kSr;
        play = std::min(want, have);
        st.set_playhead(play);
        if (knob_at < 0 && play >= 8 * kSr) {
            knob_at = play;
            brosoundml::Yue2Knobs k;
            k.texture = 1.0f;
            k.energy = 0.5f;
            st.set_knobs(k);
        }
        if (knob_at >= 0 && !queued_after && play >= knob_at + kSr) {
            after_knob = st.stats();   // the knob landed a second ago
            st.enqueue(given_song(E, 1400, 200, "C3"));
            queued_after = true;
        }
        if (start1 >= 0 && built1_at < 0)
            for (const auto& g : st.stats().songs)
                if (g.index == 1 && g.on_air && g.built) built1_at = play;
        std::lock_guard<std::mutex> lk(cap.mu);
        if (skip_at < 0 && queued_after && play >= knob_at + 4 * kSr) {
            skip_at = play;
            st.skip();
        }
        if (const auto* ss1 = cap.find_last(Kind::SongStart, 1); ss1 && skip_at >= 0) start1 = ss1->e.sample;
        if (const auto* e2 = cap.find(Kind::SongEnd, 2); e2 && play >= e2->e.sample) stopped = true;
        if (cap.find(Kind::Error)) stopped = true;
    }
    const brosoundml::Yue2StreamStats end_stats = st.stats();
    st.stop();
    worker.join();
    const double peak = vm.stop_gb();
    CHECK(stopped, "C: did not play out within 150 s");
    CHECK(!cap.find(Kind::Error), "C: the stream reported an error");
    CHECK(cap.contiguous, "C: program audio not contiguous");
    std::printf("  underrun after the knob %.2f s, peak %.2f GB\n", underrun, peak);
    // The knob: one Knobs event naming no song, every song then existing keeps
    // what it was started with, the song queued after takes the new ones.
    const auto* kn = cap.find(Kind::Knobs);
    CHECK(kn && kn->e.song == -1 && kn->e.text == "energy/texture", "C: Knobs event %s (song %d)",
          kn ? kn->e.text.c_str() : "missing", kn ? kn->e.song : -2);
    CHECK(cap.count(Kind::Knobs) == 1, "C: %d Knobs events, want 1", cap.count(Kind::Knobs));
    for (const auto& g : after_knob.songs)
        CHECK(g.knobs.texture == 0.0f && g.knobs.energy == 0.0f, "C: song %d took the knob (texture %.2f, energy %.2f)",
              g.index, g.knobs.texture, g.knobs.energy);
    bool c3 = false;
    for (const auto& g : end_stats.songs)
        if (g.tag == "C3") {
            c3 = true;
            CHECK(g.knobs.texture == 1.0f && g.knobs.energy == 0.5f, "C: the song queued after the knob has texture %.2f, "
                  "energy %.2f", g.knobs.texture, g.knobs.energy);
        }
    CHECK(c3, "C: the song queued after the knob is not in the stats");
    int windows2 = 0;
    for (const auto& e : cap.events) windows2 += e.e.kind == Kind::Window && e.e.song == 2;
    const auto* st2 = cap.find_last(Kind::SongStart, 2);
    std::printf("  song 2 built ahead in %d window(s), on the program at %.2f s, click %.2f\n", windows2,
                st2 ? st2->e.sample / double(kSr) : -1.0, st2 ? click_ratio(cap.L, st2->e.sample) : -1.0);
    CHECK(windows2 == 1, "C: song 2 (built ahead) took %d windows, want one chunk", windows2);
    CHECK(underrun < 1.0, "C: %.2f s of underrun", underrun);
    // Song 1 is 16 s; decoded only a lead (6 s) ahead it would be whole near its end.
    std::printf("  song 1 decoded whole %.2f s into it\n", built1_at >= 0 && start1 >= 0 ? (built1_at - start1) / double(kSr) : -1.0);
    CHECK(built1_at >= 0 && start1 >= 0 && built1_at - start1 < 8 * kSr,
          "C: song 1 on air was not decoded whole within 8 s of its start");
    const auto* st1 = cap.find_last(Kind::SongStart, 1);   // a skip re-places it
    CHECK(skip_at >= 0 && st1, "C: skip did not start song 1");
    if (skip_at >= 0 && st1) {
        const double lag = double(st1->e.sample - skip_at) / kSr;
        std::printf("  skip -> song 1 at +%.2f s of program, click %.2f\n", lag, click_ratio(cap.L, st1->e.sample));
        // Song 1 had not been built (song 0's continuation came first): a
        // skip then costs a fresh stream's first audio.
        CHECK(lag <= 4.5, "C: song 1 started %.2f s after the skip", lag);
    }
    if (const std::string out = env("YUE2_STREAM_OUT"); !out.empty()) {
        std::vector<float> planar(cap.L);
        planar.insert(planar.end(), cap.R.begin(), cap.R.end());
        fs::create_directories(fs::path(out) / "interrupt");
        write_raw(fs::path(out) / "interrupt" / "audio.f32", planar);
    }
}

// ── D ──
void section_d(Env& E) {
    // Song 1 is the rest of the song (built ahead: one chunk, the whole render).
    std::printf("D. radio crossfade of two given-semantic songs, the second built ahead\n");
    brosoundml::Yue2Song a = given_song(E, 0, 500, "D0"), b = given_song(E, 500, 1 << 20, "D1");
    const int T1 = static_cast<int>(b.semantic.size());
    VramMeter vm;
    vm.start();
    Capture cap;
    brosoundml::Yue2StreamOptions o;
    o.paced = false;
    o.radio = false;
    o.crossfade_seconds = 3.0;
    cap.hook(o);
    {
        Yue2Stream st(E.y, E.vae, o, cap.audio_fn(), cap.event_fn(true));
        st.enqueue(a);
        st.enqueue(b);
        st.run();
    }
    const double peak = vm.stop_gb();
    CHECK(!cap.find(Kind::Error), "D: the stream reported an error");
    const std::int64_t n0 = E.vae.output_length(500), n1 = E.vae.output_length(T1), X = 3 * kSr;
    const auto* s1 = cap.find(Kind::SongStart, 1);
    CHECK(s1 && s1->e.sample == n0 - X, "D: song 1 placed at %lld, want %lld", s1 ? (long long)s1->e.sample : -1LL,
          static_cast<long long>(n0 - X));
    CHECK(static_cast<std::int64_t>(cap.L.size()) == n0 + n1 - X, "D: program %zu samples, want %lld", cap.L.size(),
          static_cast<long long>(n0 + n1 - X));
    if (!s1) return;
    const std::int64_t f0 = s1->e.sample;
    const double before = rms(cap.L, cap.R, f0 - 2 * kSr, f0), after = rms(cap.L, cap.R, f0 + X, f0 + X + 2 * kSr);
    double low = 1e9;
    for (std::int64_t q = f0; q + kSr / 4 <= f0 + X; q += kSr / 4) low = std::min(low, rms(cap.L, cap.R, q, q + kSr / 4));
    std::printf("  RMS before %.4f, lowest 0.25 s in the fade %.4f, after %.4f; click at the fade start %.2f\n", before,
                low, after, click_ratio(cap.L, f0));
    CHECK(low >= 0.5 * std::min(before, after), "D: power dip in the crossfade (%.4f)", low);

    // Song 1, built ahead, is the whole-song render of its tokens: one chunk.
    int windows1 = 0;
    for (const auto& e : cap.events) windows1 += e.e.kind == Kind::Window && e.e.song == 1;
    brosoundml::Yue2Params p = b.params;
    const std::vector<float> whole = E.y.synthesize(p, E.y.plan(p), b.semantic);
    const double c1 = cap.lat.size() == 2 && cap.lat[1].size() == whole.size()
                          ? mean_cos(cap.lat[1], whole, T1) : 0.0;
    std::printf("  song 1: %d frames in %d window(s), cosine to the whole render %.4f; peak device memory %.2f GB\n",
                T1, windows1, c1, peak);
    CHECK(windows1 == 1, "D: song 1 built ahead in %d windows, want one chunk", windows1);
    CHECK(c1 > 0.99, "D: song 1 is not the whole render (cosine %.4f)", c1);
}

// ── E ──
void section_e(Env& E) {
    std::printf("E. B1: non-finite logits are an error, not a crash\n");
    brosoundml::Yue2Params p = E.given;
    p.run_synthesize = false;
    p.semantic_sampling.min_tokens = 1;
    p.semantic_sampling.max_tokens = 20;
    brosoundml::Yue2Steer s;
    s.layer = 9;
    s.plan = false;
    s.delta.assign(2048, 1e30f);
    s.scale = 1e30f;
    p.steer.push_back(s);
    std::string err;
    try {
        (void)E.y.generate(p);
    } catch (const std::exception& e) {
        err = e.what();
    }
    std::printf("  error: %s\n", err.c_str());
    CHECK(err.find("non-finite") != std::string::npos, "E: no non-finite error (%s)", err.c_str());
    p.steer.clear();
    const brosoundml::Yue2Result r = E.y.generate(p);
    CHECK(r.semantic.size() == 20, "E: the model is not usable after the error (%zu tokens)", r.semantic.size());
    for (int32_t c : r.semantic) CHECK(c >= 0 && c < 32768, "E: token %d out of range", c);
}

// ── F ──
void section_f(Env& E) {
    std::printf("F. B2: an arrangement steer of 4 ends by the length guard\n");
    brosoundml::Yue2Params p = E.req;
    p.run_synthesize = false;
    brosoundml::Yue2Steer s;
    s.layer = 21;
    s.plan = false;
    s.delta = brosoundml::Yue2Directions::builtin_ref().axis("arrangement")->direction;
    s.scale = 4.0f;
    p.steer.push_back(s);
    const brosoundml::Yue2Plan plan = E.y.plan(p);
    const double est = brosoundml::yue2::abc_seconds(plan.abc);
    const int cap_frames = static_cast<int>(std::max(1.5 * est, est + 20.0) * 25.0) + 1;
    const double t = now_s();
    const std::vector<int32_t> sem = E.y.semantic(p, plan);
    std::printf("  plan estimate %.1f s, hard cap %d frames; steered song %zu frames (%.1f s) in %.1f s\n", est,
                cap_frames, sem.size(), sem.size() / 25.0, now_s() - t);
    CHECK(est > 0, "F: no length estimate from the plan");
    CHECK(static_cast<int>(sem.size()) <= cap_frames, "F: %zu frames past the cap %d", sem.size(), cap_frames);
}

// ── G ──
// Three songs kept built ahead, fed through Want events: the pool fills, each
// built song waits as audio in host memory with its AR session released, a
// skip to one is the skip fade alone, a knob moved meanwhile leaves the built
// songs ready (the one skipped to catches up on air), and drop() takes one out
// (Want asks again).
void section_g(Env& E) {
    constexpr int kAhead = 3;
    std::printf("G. %d songs built ahead (Want-fed), skip to a built one, drop\n", kAhead);
    const int max_tokens = std::atoi(env("YUE2_STREAM_G_TOKENS", "1200").c_str());   // 48 s songs
    std::atomic<int> fed{0};
    auto make = [&](int n) {
        brosoundml::Yue2Song s;
        s.params = E.req;
        s.params.semantic_sampling.max_tokens = max_tokens;
        s.params.seed = E.req.seed + 101 * static_cast<std::uint64_t>(n);
        s.tag = "G" + std::to_string(n);
        return s;
    };
    VramMeter vm;
    vm.start();
    Capture cap;
    brosoundml::Yue2StreamOptions o;
    o.paced = true;
    o.radio = false;
    o.ahead_songs = kAhead;
    cap.hook(o);
    std::unique_ptr<Yue2Stream> stp;
    auto log = cap.event_fn(true);
    std::vector<int> wants;
    std::mutex wmu;
    // The app's side of Want: a fresh request per song wanted.
    auto on_event = [&](const Yue2StreamEvent& e) {
        log(e);
        if (e.kind != Kind::Want) return;
        {
            std::lock_guard<std::mutex> lk(wmu);
            wants.push_back(static_cast<int>(e.value));
        }
        for (int i = 0; i < static_cast<int>(e.value); ++i) stp->enqueue(make(++fed));
    };
    stp = std::make_unique<Yue2Stream>(E.y, E.vae, o, cap.audio_fn(), on_event);
    Yue2Stream& st = *stp;
    st.enqueue(make(0));
    std::thread worker([&] { st.run(); });

    std::int64_t play = 0;
    double last = -1, underrun = 0;
    double pool_full_at = -1, pool_vram = 0, skip_wall = -1;
    std::size_t pool_host = 0;
    int skip_to = -1, knob_song = -1, dropped = -1, stage = 0, last_before_knob = -1;
    brosoundml::Yue2StreamStats at_full, at_drop;
    const double t0 = now_s(), t_end = t0 + 420;
    while (now_s() < t_end && stage < 4) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const double tn = now_s();
        if (cap.first_audio_at >= 0) {
            if (last < 0) last = tn;
            const std::int64_t want = play + static_cast<std::int64_t>((tn - last) * kSr);
            last = tn;
            const std::int64_t have = cap.emitted;
            if (want > have && stage > 0) underrun += double(want - have) / kSr;
            play = std::min(want, have);
            st.set_playhead(play);
        }
        if (cap.find(Kind::Error)) break;
        const brosoundml::Yue2StreamStats s = st.stats();
        bool all_built = true;
        for (const auto& g : s.songs)
            if (!g.on_air) all_built = all_built && g.built;
        if (stage == 0 && s.ready_ahead >= kAhead && s.ahead >= kAhead && all_built) {
            // Every song after the playing one is built: memory now, then a
            // knob move (the built songs must stay ready) and a skip.
            pool_full_at = tn - t0;
            pool_vram = vm.stop_gb();
            vm.start();
            pool_host = s.host_bytes;
            at_full = s;
            std::printf("  pool full at %.1f s: %d ahead, %d ready, all built %d; host %.1f MB; peak device %.2f GB\n",
                        pool_full_at, s.ahead, s.ready_ahead, all_built ? 1 : 0, s.host_bytes / 1e6, pool_vram);
            for (const auto& g : s.songs) last_before_knob = std::max(last_before_knob, g.index);
            brosoundml::Yue2Knobs k;
            k.texture = 1.0f;
            st.set_knobs(k);
            stage = 1;
        } else if (stage == 1 && tn - t0 > pool_full_at + 1.0) {
            CHECK(s.ready_ahead == kAhead, "G: a knob move unready the built songs (%d ready)", s.ready_ahead);
            for (const auto& g : s.songs)
                if (!g.on_air && g.index >= 0) {
                    skip_to = g.index;
                    break;
                }
            {
                std::lock_guard<std::mutex> lk(cap.mu);
                knob_song = skip_to;
            }
            skip_wall = tn;
            st.skip();
            stage = 2;
        } else if (stage == 2) {
            std::lock_guard<std::mutex> lk(cap.mu);
            const auto* fa = cap.find_last(Kind::FirstAudio, skip_to);
            const auto* ss = cap.find_last(Kind::SongStart, skip_to);
            if (fa && ss && fa->at >= skip_wall && play >= ss->e.sample + 3 * kSr) {
                // Then drop the last song queued.
                int last_idx = -1;
                for (const auto& g : s.songs) last_idx = std::max(last_idx, g.index);
                dropped = last_idx;
                st.drop(last_idx);
                stage = 3;
            }
        } else if (stage == 3) {
            std::lock_guard<std::mutex> lk(cap.mu);
            const auto* d = cap.find(Kind::Dropped, dropped);
            if (d && st.stats().ahead >= kAhead) {
                at_drop = st.stats();
                stage = 4;
            }
        }
    }
    st.stop();
    worker.join();
    const double peak_after = vm.stop_gb();
    CHECK(!cap.find(Kind::Error), "G: the stream reported an error");
    CHECK(stage == 4, "G: stopped at stage %d (pool full at %.1f s)", stage, pool_full_at);
    CHECK(cap.contiguous, "G: program audio not contiguous");
    {
        std::lock_guard<std::mutex> lk(wmu);
        std::string ws;
        for (int w : wants) ws += std::to_string(w) + " ";
        std::printf("  wants: %s(fed %d songs)\n", ws.c_str(), fed.load());
        CHECK(!wants.empty() && wants.front() == kAhead, "G: the first Want asked for %d, want %d",
              wants.empty() ? -1 : wants.front(), kAhead);
        CHECK(wants.size() >= 3, "G: %zu Want events; one at start, after the skip and after the drop", wants.size());
    }
    if (skip_to >= 0) {
        const auto* ss = cap.find_last(Kind::SongStart, skip_to);
        const auto* fa = cap.find_last(Kind::FirstAudio, skip_to);
        const double cut = ss ? ss->e.seconds : -1, first = fa ? fa->e.seconds : -1;
        std::printf("  skip -> song %d: cut over %.3f s, first audio handed out %.3f s after the request "
                    "(heard after the emit lead %.2f s + the %.2f s fade); click %.2f\n",
                    skip_to, cut, first, o.emit_lead_seconds, o.skip_fade_seconds,
                    ss ? click_ratio(cap.L, ss->e.sample) : -1.0);
        CHECK(cut >= 0 && cut < 0.25, "G: skip to a built song took %.3f s to cut over", cut);
        CHECK(first >= 0 && first < 0.35, "G: skip to a built song: first audio %.3f s after the request", first);
        // Built before the knob moved, it plays as it was built; the songs
        // fed after the move take the knob.
        int fed_after = 0;
        for (const auto& g : at_drop.songs) {
            if (g.index == knob_song)
                CHECK(g.knobs.texture == 0.0f, "G: song %d (built before the knob) has texture %.2f on air", knob_song,
                      g.knobs.texture);
            if (g.index > last_before_knob) {
                ++fed_after;
                CHECK(g.knobs.texture == 1.0f, "G: song %d (fed after the knob) has texture %.2f", g.index,
                      g.knobs.texture);
            }
        }
        CHECK(fed_after > 0, "G: no song was fed after the knob moved");
        std::printf("  song %d played as built; %d song(s) fed after the knob took it\n", knob_song, fed_after);
    }
    std::printf("  underrun %.2f s; peak device memory with the pool full %.2f GB, after %.2f GB; host %.1f MB "
                "for %d songs\n",
                underrun, pool_vram, peak_after, pool_host / 1e6, static_cast<int>(at_full.songs.size()));
    CHECK(underrun < 1.0, "G: %.2f s of underrun", underrun);
}

}  // namespace

int main() {
    const std::string sections = env("YUE2_STREAM_SECTIONS", "ABCDEFG");
    auto Env_ = std::make_unique<Env>();
    Env& E = *Env_;
    if (!load_env(E)) return 0;

    for (char c : sections) {
        const double t = now_s();
        switch (c) {
            case 'A': section_a(E); break;
            case 'B': section_b(E); break;
            case 'C': section_c(E); break;
            case 'D': section_d(E); break;
            case 'E': section_e(E); break;
            case 'F': section_f(E); break;
            case 'G': section_g(E); break;
            default: continue;
        }
        std::printf("  (%c: %.1f s, %d failures so far)\n", c, now_s() - t, failures);
        std::fflush(stdout);
    }
    std::printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
