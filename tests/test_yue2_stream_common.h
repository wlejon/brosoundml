#pragma once
// The Yue2Stream GPU tests' shared harness (test_yue2_stream.cpp,
// test_yue2_stream_preview.cpp): environment, clocks, the program capture
// with its events, device memory sampling, the audio meters, and the loaded
// model + request (load_env).

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

namespace yue2_stream_test {

using namespace yue2_test;
using brosoundml::StereoAudio;
using brosoundml::Yue2;
using brosoundml::Yue2Stream;
using brosoundml::Yue2StreamEvent;
using Kind = brosoundml::Yue2StreamEvent::Kind;

constexpr int kSr = 48000, kHop = 1920;

inline std::string env(const char* k, const std::string& d = {}) {
    const char* v = std::getenv(k);
    return v && *v ? std::string(v) : d;
}

inline double now_s() {
    using C = std::chrono::steady_clock;
    static const auto t0 = C::now();
    return std::chrono::duration<double>(C::now() - t0).count();
}

inline std::string slurp(const fs::path& p) {
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
            static const char* names[] = {"plan",  "start",    "semantic", "window",  "knobs", "break",
                                          "song-end", "first", "error",    "idle",    "want",  "built",
                                          "dropped",  "full",  "saved"};
            if (e.kind == Kind::Window) return;
            std::printf("  [%7.2f] %-9s song %d sample %lld frames %d frame0 %d seconds %.2f value %.3f\n", now_s(),
                        names[static_cast<int>(e.kind)], e.song, static_cast<long long>(e.sample), e.frames, e.frame0,
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

inline double rms(const std::vector<float>& L, const std::vector<float>& R, std::int64_t a, std::int64_t b) {
    a = std::max<std::int64_t>(0, a);
    b = std::min<std::int64_t>(static_cast<std::int64_t>(L.size()), b);
    if (b <= a) return 0;
    double s = 0;
    for (std::int64_t i = a; i < b; ++i) s += 0.5 * (double(L[i]) * L[i] + double(R[i]) * R[i]);
    return std::sqrt(s / static_cast<double>(b - a));
}

// The largest sample step within 10 ms of `at`, over the 99th percentile of
// 20 ms blocks' largest steps across the program: a click reads far above 1.
inline double click_ratio(const std::vector<float>& x, std::int64_t at) {
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

struct Env {
    Yue2 y;
    brosoundml::Yue2Vae vae;
    brosoundml::Yue2Params req;     // the request
    brosoundml::Yue2Params given;   // rock_gen_W0's plan (abc, style, lyrics, seed)
    std::vector<int32_t> given_sem;
    std::string song;   // the given run's song name (rock_gen_W0 -> rock)
};

// The model, the VAE, the request and the given run; false (after printing
// SKIP) without YUE2_E2E=1 or the weights.
inline bool load_env(Env& E) {
    if (env("YUE2_E2E") != "1") {
        std::printf("SKIP: set YUE2_E2E=1 (minutes of GPU)\n");
        return false;
    }
    brotensor::init();
    const fs::path mdir = model_dir();
    const fs::path vdir = env("YUE2_VAE_DIR");
    const fs::path from = env("YUE2_STREAM_FROM", (repo().parent_path() / "yue2-research" / "runs" / "exp1" /
                                                   "rock_gen_W0").string());
    if (mdir.empty() || vdir.empty() || !fs::exists(from / "semantic.i32")) {
        std::printf("SKIP: needs the YuE2-3B folder, $YUE2_VAE_DIR and %s\n", from.string().c_str());
        return false;
    }
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
    const double t = now_s();
    E.y.load(mdir.string(), brotensor::Device::CUDA);
    E.vae.load(vdir.string(), brotensor::Device::CUDA, /*decoder_only=*/true);
    std::printf("load %.1f s; device memory %.2f GB before, %.2f GB loaded; %zu given frames\n", now_s() - t, base,
                VramMeter::used() / 1e9, E.given_sem.size());
    return true;
}

}  // namespace yue2_stream_test
