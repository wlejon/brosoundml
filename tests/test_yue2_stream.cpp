#define _CRT_SECURE_NO_WARNINGS
// Yue2Stream on the GPU (minutes; runs only with YUE2_E2E=1): the streaming
// player, the radio's crossfade, the two-stage interrupt and the bug guards.
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
//   C. Paced, a simulated player: interrupt(+1) at 8 s; stage 1 and stage 2
//      latencies, the RMS after each, no discontinuity where they land; then
//      skip() to a queued second song, while a third is built ahead in one
//      NAR chunk and crossfades in; no underrun.
//   D. The radio's crossfade of two given-semantic songs: placement, program
//      length, no power dip across the fade; the second, built ahead, is the
//      whole-song render (one chunk) and sets the peak device memory.
//   E. B1: non-finite logits raise "non-finite logits" instead of sampling an
//      out-of-range token, and the model stays usable.
//   F. B2: an arrangement steer of 4 (the research's never-ending song) ends
//      by the length guard's hard cap.
// Peak device memory (device_mem_info, sampled) is reported per section.
// Numbers are reported and asserted; nothing judges the audio by ear.

#include "test_yue2_common.h"

#include "brotensor/json.h"
#include "brosoundml/yue2.h"
#include "brosoundml/yue2_stream.h"
#include "brosoundml/yue2_vae.h"
#include "yue2_protocol.h"

#include <brotensor/runtime.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

using namespace yue2_test;
using brosoundml::StereoAudio;
using brosoundml::Yue2;
using brosoundml::Yue2Stream;
using brosoundml::Yue2StreamEvent;
using Kind = brosoundml::Yue2StreamEvent::Kind;

namespace {

constexpr int kSr = 48000, kHop = 1920;

std::string env(const char* k, const std::string& d = {}) {
    const char* v = std::getenv(k);
    return v && *v ? std::string(v) : d;
}

double now_s() {
    using C = std::chrono::steady_clock;
    static const auto t0 = C::now();
    return std::chrono::duration<double>(C::now() - t0).count();
}

std::string slurp(const fs::path& p) {
    std::ifstream f(p.string(), std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

template <typename T>
std::vector<T> read_raw(const fs::path& p) {
    const std::string s = slurp(p);
    std::vector<T> v(s.size() / sizeof(T));
    std::memcpy(v.data(), s.data(), v.size() * sizeof(T));
    return v;
}

template <typename T>
void write_raw(const fs::path& p, const std::vector<T>& v) {
    std::ofstream f(p.string(), std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}

// Peak device memory in use while a section runs (sampled every 20 ms).
struct VramMeter {
    static inline std::size_t base = 0;   // in use before the model loads
    std::atomic<bool> on{false};
    std::atomic<std::size_t> peak{0};
    std::thread th;
    void start() {
        peak = used();
        on = true;
        th = std::thread([this] {
            while (on) {
                const std::size_t u = used();
                if (u > peak) peak = u;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }
    double stop_gb() {
        on = false;
        if (th.joinable()) th.join();
        return (static_cast<double>(peak) - static_cast<double>(base)) / 1e9;   // ours: net of other processes
    }
    static std::size_t used() {
        std::size_t fr = 0, tot = 0;
        if (!brotensor::device_mem_info(brotensor::Device::CUDA, fr, tot)) return 0;
        return tot - fr;
    }
};

// The program as the stream hands it out (contiguous from 0), and its events.
struct Capture {
    std::mutex mu;
    std::vector<float> L, R;
    std::atomic<std::int64_t> emitted{0};
    std::atomic<double> first_audio_at{-1};
    bool contiguous = true;
    struct Ev {
        Yue2StreamEvent e;
        double at;
    };
    std::vector<Ev> events;
    std::vector<std::vector<float>> lat;   // per song, frame-major (latest solve of each frame)
    std::vector<std::vector<int32_t>> sem;
    std::vector<std::pair<int, int>> windows;   // song 0's landed windows (frame0, frames)

    Yue2Stream::AudioFn audio_fn() {
        return [this](const StereoAudio& c, std::int64_t s0) {
            std::lock_guard<std::mutex> lk(mu);
            if (s0 != static_cast<std::int64_t>(L.size())) contiguous = false;
            L.insert(L.end(), c.channel(0), c.channel(0) + c.frames);
            R.insert(R.end(), c.channel(1), c.channel(1) + c.frames);
            if (first_audio_at < 0) first_audio_at = now_s();
            emitted = static_cast<std::int64_t>(L.size());
        };
    }
    Yue2Stream::EventFn event_fn(bool verbose) {
        return [this, verbose](const Yue2StreamEvent& e) {
            {
                std::lock_guard<std::mutex> lk(mu);
                events.push_back({e, now_s()});
            }
            if (e.kind == Kind::Error) std::printf("  [error] song %d: %s\n", e.song, e.text.c_str());
            if (!verbose) return;
            static const char* names[] = {"plan",  "start",    "semantic", "window", "interrupt", "knobs",
                                          "break", "song-end", "first",    "error",  "idle"};
            if (e.kind == Kind::Window) return;
            std::printf("  [%7.2f] %-9s song %d sample %lld frames %d seconds %.2f value %.3f\n", now_s(),
                        names[static_cast<int>(e.kind)], e.song, static_cast<long long>(e.sample), e.frames,
                        e.seconds, e.value);
        };
    }
    void hook(brosoundml::Yue2StreamOptions& o) {
        o.on_latents = [this](int song, int f0, int n, const float* d) {
            std::lock_guard<std::mutex> lk(mu);
            if (static_cast<int>(lat.size()) <= song) lat.resize(song + 1);
            auto& v = lat[song];
            if (static_cast<int>(v.size()) < (f0 + n) * 64) v.resize(static_cast<std::size_t>(f0 + n) * 64, 0.0f);
            std::copy(d, d + static_cast<std::size_t>(n) * 64, v.begin() + static_cast<std::size_t>(f0) * 64);
            if (song == 0) windows.emplace_back(f0, n);
        };
        o.on_semantic = [this](int song, const std::vector<int32_t>& c) {
            std::lock_guard<std::mutex> lk(mu);
            if (static_cast<int>(sem.size()) <= song) sem.resize(song + 1);
            sem[song] = c;
        };
    }
    const Ev* find(Kind k, int song = -1, double value = -1) {
        for (const auto& e : events)
            if (e.e.kind == k && (song < 0 || e.e.song == song) && (value < 0 || e.e.value == value)) return &e;
        return nullptr;
    }
    const Ev* find_last(Kind k, int song) {
        for (auto it = events.rbegin(); it != events.rend(); ++it)
            if (it->e.kind == k && it->e.song == song) return &*it;
        return nullptr;
    }
    int count(Kind k) {
        int n = 0;
        for (const auto& e : events) n += e.e.kind == k;
        return n;
    }
};

double rms(const std::vector<float>& L, const std::vector<float>& R, std::int64_t a, std::int64_t b) {
    a = std::max<std::int64_t>(0, a);
    b = std::min<std::int64_t>(static_cast<std::int64_t>(L.size()), b);
    if (b <= a) return 0;
    double s = 0;
    for (std::int64_t i = a; i < b; ++i) s += 0.5 * (double(L[i]) * L[i] + double(R[i]) * R[i]);
    return std::sqrt(s / static_cast<double>(b - a));
}

// The largest sample step within 10 ms of `at`, over the 99th percentile of
// 20 ms blocks' largest steps across the program: a click reads far above 1.
double click_ratio(const std::vector<float>& x, std::int64_t at) {
    const std::int64_t n = static_cast<std::int64_t>(x.size());
    std::vector<float> blocks;
    for (std::int64_t b = 1; b + 960 <= n; b += 960) {
        float m = 0;
        for (std::int64_t i = b; i < b + 960; ++i) m = std::max(m, std::fabs(x[i] - x[i - 1]));
        blocks.push_back(m);
    }
    if (blocks.empty()) return 0;
    std::nth_element(blocks.begin(), blocks.begin() + blocks.size() * 99 / 100, blocks.end());
    const float p99 = blocks[blocks.size() * 99 / 100];
    float m = 0;
    for (std::int64_t i = std::max<std::int64_t>(1, at - 480); i < std::min(n, at + 480); ++i)
        m = std::max(m, std::fabs(x[i] - x[i - 1]));
    return m / (p99 + 1e-9);
}

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

struct Env {
    Yue2 y;
    brosoundml::Yue2Vae vae;
    brosoundml::Yue2Params req;     // the request
    brosoundml::Yue2Params given;   // rock_gen_W0's plan (abc, style, lyrics, seed)
    std::vector<int32_t> given_sem;
    std::string song;   // the given run's song name (rock_gen_W0 -> rock)
};

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
    std::printf("C. paced: interrupt(+1) at 8 s, then skip to a queued song\n");
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
    std::int64_t play = 0, int_at = -1, skip_at = -1;
    double underrun = 0, last = -1, int_wall = 0;
    std::vector<std::pair<double, std::int64_t>> hist;   // (wall, playhead) as reported
    const double t_end = now_s() + 150;
    bool stopped = false;
    while (!stopped && now_s() < t_end) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const double tn = now_s();
        if (cap.first_audio_at < 0) continue;
        if (last < 0) last = tn;
        const std::int64_t want = play + static_cast<std::int64_t>((tn - last) * kSr);
        last = tn;
        const std::int64_t have = cap.emitted;
        if (want > have && int_at >= 0) underrun += double(want - have) / kSr;
        play = std::min(want, have);
        st.set_playhead(play);
        hist.emplace_back(tn, play);
        std::lock_guard<std::mutex> lk(cap.mu);
        if (int_at < 0 && play >= 8 * kSr) {
            int_at = play;
            int_wall = tn;
            st.interrupt(1.0f);
        }
        const auto* i2 = cap.find(Kind::Interrupt, 0, 2);
        if (skip_at < 0 && i2 && play >= i2->e.sample + 4 * kSr) {
            skip_at = play;
            st.skip();
        }
        if (const auto* e2 = cap.find(Kind::SongEnd, 2); e2 && play >= e2->e.sample) stopped = true;
        if (cap.find(Kind::Error)) stopped = true;
    }
    st.stop();
    worker.join();
    const double peak = vm.stop_gb();
    CHECK(stopped, "C: did not play out within 150 s");
    CHECK(!cap.find(Kind::Error), "C: the stream reported an error");
    CHECK(cap.contiguous, "C: program audio not contiguous");
    const auto* i1 = cap.find(Kind::Interrupt, 0, 1);
    const auto* i2 = cap.find(Kind::Interrupt, 0, 2);
    CHECK(i1 && i2, "C: interrupt stages missing (%d events)", cap.count(Kind::Interrupt));
    std::printf("  underrun after the interrupt %.2f s, peak %.2f GB\n", underrun, peak);
    int windows2 = 0;
    for (const auto& e : cap.events) windows2 += e.e.kind == Kind::Window && e.e.song == 2;
    const auto* st2 = cap.find_last(Kind::SongStart, 2);
    std::printf("  song 2 built ahead in %d window(s), on the program at %.2f s, click %.2f\n", windows2,
                st2 ? st2->e.sample / double(kSr) : -1.0, st2 ? click_ratio(cap.L, st2->e.sample) : -1.0);
    CHECK(windows2 == 1, "C: song 2 (built ahead) took %d windows, want one chunk", windows2);
    CHECK(underrun < 1.0, "C: %.2f s of underrun", underrun);
    if (i1 && i2) {
        const double r_before = rms(cap.L, cap.R, int_at - 2 * kSr, int_at);
        const double r1 = rms(cap.L, cap.R, i1->e.sample + kSr / 2, i1->e.sample + 2 * kSr);
        const double r2 = rms(cap.L, cap.R, i2->e.sample, i2->e.sample + 3 * kSr);
        // The player's own clock: when its playhead crossed each stage's sample.
        auto crossed = [&](std::int64_t sample) {
            for (const auto& [w, ph] : hist)
                if (ph >= sample) return w - double(ph - sample) / kSr - int_wall;
            return -1.0;
        };
        const double p1 = crossed(i1->e.sample), p2 = crossed(i2->e.sample);
        std::printf("  stage 1 %.2f s, stage 2 %.2f s (request -> audible; the player's clock %.2f / %.2f s); "
                    "RMS before %.4f, after 1 %.4f, after 2 %.4f\n",
                    i1->e.seconds, i2->e.seconds, p1, p2, r_before, r1, r2);
        CHECK(p1 >= 0 && p1 <= 1.0, "C: stage 1 took %.2f s", p1);
        CHECK(p2 >= 0 && p2 <= 3.0, "C: stage 2 took %.2f s", p2);
        CHECK(std::fabs(p1 - i1->e.seconds) < 0.1 && std::fabs(p2 - i2->e.seconds) < 0.1,
              "C: the stream's interrupt timings (%.2f, %.2f) are not the player's", i1->e.seconds, i2->e.seconds);
        CHECK(r1 > r_before, "C: RMS did not rise after stage 1 (%.4f -> %.4f)", r_before, r1);
        const double c1 = click_ratio(cap.L, i1->e.sample), c2 = click_ratio(cap.L, i2->e.sample);
        std::printf("  click ratio at stage 1 %.2f, stage 2 %.2f\n", c1, c2);
        CHECK(c1 < 3.0 && c2 < 3.0, "C: discontinuity (%.2f, %.2f)", c1, c2);
    }
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
    s.delta = brosoundml::Yue2Directions::builtin().ar_l21;
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

}  // namespace

int main() {
    if (env("YUE2_E2E") != "1") {
        std::printf("SKIP: set YUE2_E2E=1 (minutes of GPU)\n");
        return 0;
    }
    brotensor::init();
    const fs::path mdir = model_dir();
    const fs::path vdir = env("YUE2_VAE_DIR");
    const fs::path from = env("YUE2_STREAM_FROM", (repo().parent_path() / "yue2-research" / "runs" / "exp1" /
                                                   "rock_gen_W0").string());
    if (mdir.empty() || vdir.empty() || !fs::exists(from / "semantic.i32")) {
        std::printf("SKIP: needs the YuE2-3B folder, $YUE2_VAE_DIR and %s\n", from.string().c_str());
        return 0;
    }
    const std::string sections = env("YUE2_STREAM_SECTIONS", "ABCDEF");
    auto Env_ = std::make_unique<Env>();
    Env& E = *Env_;

    const auto req = brotensor::json::parse(slurp(env("YUE2_REQUEST", (mdir / "examples" / "tonight-awake.json").string())));
    E.req.style = req.get_string("style", "");
    E.req.lyrics = req.get_string("lyrics", "");
    E.req.cot = brosoundml::Yue2Cot::Full;
    E.req.seed = static_cast<std::uint64_t>(req.get_int("seed", 831001));
    const auto pj = brotensor::json::parse(slurp(from / "plan.json"));
    E.given.style = pj.get_string("style", "");
    E.given.lyrics = pj.get_string("lyrics", "");
    E.given.abc = pj.get_string("abc", "");
    E.given.cot = brosoundml::Yue2Cot::Full;
    E.given.seed = static_cast<std::uint64_t>(pj.get_int("seed", 831001));
    E.given_sem = read_raw<int32_t>(from / "semantic.i32");
    E.song = from.filename().string();
    E.song = E.song.substr(0, E.song.find('_'));

    VramMeter::base = VramMeter::used();
    const double base = VramMeter::base / 1e9;
    double t = now_s();
    E.y.load(mdir.string(), brotensor::Device::CUDA);
    E.vae.load(vdir.string(), brotensor::Device::CUDA, /*decoder_only=*/true);
    std::printf("load %.1f s; device memory %.2f GB before, %.2f GB loaded; %zu given frames\n", now_s() - t, base,
                VramMeter::used() / 1e9, E.given_sem.size());

    for (char c : sections) {
        t = now_s();
        switch (c) {
            case 'A': section_a(E); break;
            case 'B': section_b(E); break;
            case 'C': section_c(E); break;
            case 'D': section_d(E); break;
            case 'E': section_e(E); break;
            case 'F': section_f(E); break;
            default: continue;
        }
        std::printf("  (%c: %.1f s, %d failures so far)\n", c, now_s() - t, failures);
        std::fflush(stdout);
    }
    std::printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
