#define _CRT_SECURE_NO_WARNINGS
// Yue2Endless on the GPU (minutes; YUE2_E2E=1, the YuE2-3B folder and the
// VAE: $YUE2_VAE_DIR, else weights/yue2-vae). Its bit-identity with
// endless.js is yue2-research's check (native/endless_stream.js); this is the
// stream's own behaviour:
//   1. unpaced, 3 chunks of one style: Sampled / Built per chunk in order,
//      250 tokens each, finite latents at the corpus loudness; the program
//      audio contiguous from 0 and exactly vae.decode() of the latents;
//   2. unpaced, the same target with a change at 20 s (memory drop): chunks
//      0-1 token for token the first run's, chunk 2 opened by 12 tokens
//      (how "drop"), a Planned event for chunk 2 and a StyleChange there;
//   3. paced by a wall-clock player: two branches opened, one discarded, the
//      other committed (at the first boundary it can make, within 15 s of
//      the request), the main line's chunks from there carry its tag; then a
//      target set now is planned where stats said and the chunk is made
//      under it; no service gap of 0.2 s, contiguous audio, stop() ends it.

#include "test_yue2_common.h"

#include "brosoundml/yue2.h"
#include "brosoundml/yue2_endless.h"
#include "brosoundml/yue2_vae.h"

#include <brotensor/runtime.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

using namespace yue2_test;
using brosoundml::Yue2Endless;
using brosoundml::Yue2EndlessEvent;
using brosoundml::Yue2EndlessOptions;
using brosoundml::Yue2EndlessTarget;
using Kind = brosoundml::Yue2EndlessEvent::Kind;

namespace {

constexpr int kSr = 48000, kC = 250, kD = 64, kHop = 1920;
const char* kFolk = "folk, acoustic guitar, warm, intimate, male vocal, 90 bpm";
const char* kSynth = "synthwave, analog synthesizer, drum machine, retro, 100 bpm";
const char* kMetal = "heavy metal, distorted electric guitar, aggressive drums, 120 bpm";

double now_s() {
    using C = std::chrono::steady_clock;
    static const auto t0 = C::now();
    return std::chrono::duration<double>(C::now() - t0).count();
}

Yue2EndlessTarget target(const char* style, const char* tag) {
    Yue2EndlessTarget t;
    t.style = style;
    t.tag = tag;
    return t;
}

// What a stream handed out.
struct Capture {
    std::mutex mu;
    std::vector<Yue2EndlessEvent> events;
    std::vector<float> L, R;
    std::atomic<std::int64_t> emitted{0};
    bool contiguous = true, finite = true;

    Yue2Endless::AudioFn audio() {
        return [this](const brosoundml::StereoAudio& a, std::int64_t s0) {
            std::lock_guard<std::mutex> lk(mu);
            if (s0 != static_cast<std::int64_t>(L.size())) contiguous = false;
            for (int i = 0; i < a.frames; ++i) {
                L.push_back(a.channel(0)[i]);
                R.push_back(a.channel(1)[i]);
                if (!std::isfinite(L.back()) || !std::isfinite(R.back())) finite = false;
            }
            emitted = s0 + a.frames;
        };
    }
    Yue2Endless::EventFn event() {
        return [this](const Yue2EndlessEvent& e) {
            std::lock_guard<std::mutex> lk(mu);
            events.push_back(e);
            if (e.kind == Kind::Error) std::fprintf(stderr, "stream error: %s\n", e.text.c_str());
        };
    }
    std::vector<Yue2EndlessEvent> of(Kind k) {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<Yue2EndlessEvent> v;
        for (const auto& e : events)
            if (e.kind == k) v.push_back(e);
        return v;
    }
};

double norm_of(const std::vector<float>& lat) {
    double s = 0;
    const std::size_t n = lat.size() / kD;
    for (std::size_t f = 0; f < n; ++f) {
        double q = 0;
        for (int d = 0; d < kD; ++d) q += double(lat[f * kD + d]) * lat[f * kD + d];
        s += std::sqrt(q);
    }
    return n ? s / static_cast<double>(n) : 0;
}

struct Unpaced {
    std::vector<int32_t> tokens;
    std::vector<float> latents;
    Capture cap;
};

void run_unpaced(const brosoundml::Yue2& y, const brosoundml::Yue2Vae& vae, Yue2EndlessOptions o, Unpaced& u) {
    o.paced = false;
    o.chunk_data = true;
    Yue2Endless s(y, vae, o, u.cap.audio(), u.cap.event());
    s.run();
    u.tokens.assign(static_cast<std::size_t>(o.max_chunks) * kC, -1);
    u.latents.assign(static_cast<std::size_t>(o.max_chunks) * kC * kD, 0.0f);
    for (const auto& e : u.cap.events) {
        if (e.kind == Kind::Sampled && e.tokens)
            std::copy(e.tokens->begin(), e.tokens->end(), u.tokens.begin() + static_cast<std::ptrdiff_t>(e.chunk) * kC);
        if (e.kind == Kind::Built && e.latents)
            std::copy(e.latents->begin(), e.latents->end(), u.latents.begin() + static_cast<std::ptrdiff_t>(e.frame0) * kD);
    }
}

// 1 + 2: unpaced, one style and then a drop.
void unpaced(const brosoundml::Yue2& y, const brosoundml::Yue2Vae& vae) {
    Yue2EndlessOptions o;
    o.max_chunks = 3;
    o.target = target(kFolk, "folk");
    o.seed = 3;
    Unpaced a;
    double t = now_s();
    run_unpaced(y, vae, o, a);
    std::printf("unpaced: 3 chunks in %.1f s\n", now_s() - t);
    auto sampled = a.cap.of(Kind::Sampled), built = a.cap.of(Kind::Built);
    CHECK(sampled.size() == 3 && built.size() == 3, "a Sampled and a Built per chunk (%zu, %zu)", sampled.size(), built.size());
    for (int k = 0; k < static_cast<int>(std::min<std::size_t>(3, built.size())); ++k) {
        CHECK(sampled[k].chunk == k && built[k].chunk == k, "chunk order: sampled %d, built %d at %d", sampled[k].chunk,
              built[k].chunk, k);
        CHECK(sampled[k].tokens && sampled[k].tokens->size() == kC, "chunk %d: %d tokens", k,
              sampled[k].tokens ? static_cast<int>(sampled[k].tokens->size()) : -1);
        const double n = built[k].value;
        CHECK(n > 6.0 && n < 10.5, "chunk %d: latent norm %.2f", k, n);
        CHECK(built[k].row && built[k].row->tag == "folk" && built[k].row->memory == (k == 0 ? 0 : std::min(250, k * kC)),
              "chunk %d row: tag %s, memory %d", k, built[k].row ? built[k].row->tag.c_str() : "?",
              built[k].row ? built[k].row->memory : -1);
    }
    CHECK(std::all_of(a.tokens.begin(), a.tokens.end(), [](int32_t v) { return v >= 0; }), "every token sampled");
    CHECK(std::all_of(a.latents.begin(), a.latents.end(), [](float v) { return std::isfinite(v); }), "finite latents");
    const auto whole = vae.decode(a.latents.data(), 3 * kC);
    CHECK(a.cap.contiguous && a.cap.finite, "audio contiguous from 0 and finite");
    CHECK(static_cast<int>(a.cap.L.size()) == whole.frames, "program length %zu, the decoder's %d", a.cap.L.size(), whole.frames);
    double md = 0;
    for (std::size_t i = 0; i < std::min<std::size_t>(a.cap.L.size(), static_cast<std::size_t>(whole.frames)); ++i)
        md = std::max({md, std::fabs(double(a.cap.L[i]) - whole.channel(0)[i]), std::fabs(double(a.cap.R[i]) - whole.channel(1)[i])});
    std::printf("unpaced: program vs vae.decode max |d| %.3g; norms %.2f\n", md, norm_of(a.latents));
    CHECK(md == 0, "the program is vae.decode() of the latents exactly (max |d| %.3g)", md);
    CHECK(!a.cap.of(Kind::Done).empty(), "a Done event");

    // The same, a drop to synth at 20 s.
    o.max_chunks = 4;
    Yue2EndlessTarget b = target(kSynth, "synth");
    b.at = 20;
    o.path = {b};
    Unpaced c;
    run_unpaced(y, vae, o, c);
    CHECK(std::equal(c.tokens.begin(), c.tokens.begin() + 2 * kC, a.tokens.begin()),
          "chunks 0-1 are the first run's, token for token");
    const auto pl = c.cap.of(Kind::Planned);
    CHECK(pl.size() == 1 && pl[0].chunk == 2 && pl[0].sample == 2LL * kC * kHop && pl[0].tag == "synth",
          "Planned: chunk 2, heard at 20 s (%zu events, chunk %d)", pl.size(), pl.empty() ? -1 : pl[0].chunk);
    const auto sc = c.cap.of(Kind::StyleChange);
    CHECK(sc.size() == 1 && sc[0].chunk == 2 && sc[0].text == "drop", "StyleChange: drop at chunk 2 (%zu)", sc.size());
    for (const auto& e : c.cap.of(Kind::Built)) {
        if (e.chunk == 2)
            CHECK(e.row && e.row->memory == 12 && e.row->how == "drop" && e.row->change && e.row->tag == "synth",
                  "chunk 2: memory %d, how %s", e.row ? e.row->memory : -1, e.row ? e.row->how.c_str() : "?");
        if (e.chunk == 3) CHECK(e.row && e.row->memory == 250 && !e.row->change, "chunk 3: memory 250 again");
    }
}

// 3: paced, a player on the wall clock.
void paced(const brosoundml::Yue2& y, const brosoundml::Yue2Vae& vae) {
    Yue2EndlessOptions o;
    o.target = target(kFolk, "folk");
    o.seed = 7;
    Capture cap;
    Yue2Endless s(y, vae, o, cap.audio(), cap.event());
    std::thread worker([&] { s.run(); });
    const double t0 = now_s();
    std::atomic<double> first{-1};
    auto play = [&] {
        const std::int64_t e = cap.emitted.load();
        if (e > 0 && first.load() < 0) first = now_s();
        if (first.load() < 0) return std::int64_t(0);
        const std::int64_t p = std::min<std::int64_t>(e, static_cast<std::int64_t>((now_s() - first.load()) * kSr));
        s.set_playhead(p);
        return p;
    };
    auto until = [&](const char* what, double seconds, const std::function<bool()>& f) {
        const double end = now_s() + seconds;
        while (now_s() < end) {
            play();
            if (f()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        CHECK(false, "timed out waiting for %s", what);
        return false;
    };
    bool ok = until("2 s of playback", 60, [&] { return play() >= 2 * kSr; });

    // Two branches; discard the second, commit the first.
    const int b1 = s.branch(target(kMetal, "metal"));
    const int b2 = s.branch(target(kSynth, "synth"));
    CHECK(b1 == 1 && b2 == 2, "branch ids count from 1 (%d, %d)", b1, b2);
    ok = ok && until("both Branch events", 10, [&] { return cap.of(Kind::Branch).size() == 2; });
    s.discard_branch(b2);
    ok = ok && until("the discard", 10, [&] {
        for (const auto& e : cap.of(Kind::Discarded))
            if (e.branch == b2) return true;
        return false;
    });
    int fork = -1;
    for (const auto& e : cap.of(Kind::Branch))
        if (e.branch == b1) fork = e.chunk;
    ok = ok && until("the branch's first chunk rendered", 60, [&] {
        for (const auto& b : s.stats().branches)
            if (b.id == b1 && b.rendered > fork) return true;
        return false;
    });
    const std::int64_t asked = play();
    s.commit_branch(b1);
    ok = ok && until("the commit", 60, [&] { return !cap.of(Kind::Committed).empty(); });
    int g = -1;
    if (!cap.of(Kind::Committed).empty()) {
        const auto c = cap.of(Kind::Committed)[0];
        g = c.chunk;
        std::printf("paced: branch %d forked at chunk %d, committed at chunk %d (heard alone from %.2f s, %.2f s after the "
                    "request), fade %.1f s\n",
                    b1, fork, g, c.sample / double(kSr), (c.sample - asked) / double(kSr), c.value);
        CHECK(c.branch == b1, "the committed branch is %d", b1);
        CHECK(c.sample - asked < 15 * kSr, "committed within 15 s of the request (%.2f s)", (c.sample - asked) / double(kSr));
    }
    ok = ok && until("main chunk g + 1 sampled", 60, [&] { return s.stats().sampled > g + 1; });
    for (const auto& r : s.stats().chunks)
        if (r.index >= g) CHECK(r.tag == "metal" && r.branch == 0, "main chunk %d after the commit: tag %s", r.index, r.tag.c_str());
    CHECK(s.stats().branches.empty(), "no branch left open");

    // A branch committed after its fork has played: a crossfade at a later boundary.
    const int b3 = s.branch(target(kSynth, "synth"));
    ok = ok && until("the third Branch event", 10, [&] { return cap.of(Kind::Branch).size() == 3; });
    int fork3 = -1;
    for (const auto& e : cap.of(Kind::Branch))
        if (e.branch == b3) fork3 = e.chunk;
    ok = ok && until("the playhead past the fork", 60, [&] { return play() > (static_cast<std::int64_t>(fork3) * kC + 25) * kHop; });
    const std::int64_t asked3 = play();
    s.commit_branch(b3);
    ok = ok && until("the second commit", 60, [&] { return cap.of(Kind::Committed).size() == 2; });
    if (cap.of(Kind::Committed).size() == 2) {
        const auto c = cap.of(Kind::Committed)[1];
        std::printf("paced: branch %d forked at chunk %d, committed at chunk %d (heard alone from %.2f s, %.2f s after the "
                    "request), fade %.1f s\n",
                    b3, fork3, c.chunk, c.sample / double(kSr), (c.sample - asked3) / double(kSr), c.value);
        CHECK(c.branch == b3 && c.chunk > fork3 && c.value > 0, "a crossfade at a boundary past the fork");
        CHECK(c.sample - asked3 < 15 * kSr, "committed within 15 s of the request (%.2f s)", (c.sample - asked3) / double(kSr));
        g = c.chunk;
    }
    ok = ok && until("main chunk g + 1 sampled", 60, [&] { return s.stats().sampled > g + 1; });
    for (const auto& r : s.stats().chunks)
        if (r.index >= g) CHECK(r.tag == "synth", "main chunk %d after the second commit: tag %s", r.index, r.tag.c_str());

    // A target now: planned where stats said it would go.
    const auto st = s.stats();
    s.set_target(target(kFolk, "folk again"));
    ok = ok && until("the Planned event", 10, [&] { return !cap.of(Kind::Planned).empty(); });
    int planned = -1;
    if (!cap.of(Kind::Planned).empty()) planned = cap.of(Kind::Planned).back().chunk;
    CHECK(planned == st.open_chunk || planned == st.open_chunk + 1,
          "planned at chunk %d, stats said %d (a chunk later if the boundary passed meanwhile)", planned, st.open_chunk);
    ok = ok && until("the planned chunk built", 60, [&] {
        for (const auto& e : cap.of(Kind::Built))
            if (e.chunk == planned && e.branch == 0) return true;
        return false;
    });
    for (const auto& e : cap.of(Kind::Built))
        if (e.chunk == planned && e.branch == 0)
            CHECK(e.row && e.row->tag == "folk again" && e.row->how == "drop", "chunk %d: tag %s, how %s", planned,
                  e.row ? e.row->tag.c_str() : "?", e.row ? e.row->how.c_str() : "?");
    for (const double end = now_s() + 5; now_s() < end;) {   // play on 5 s
        play();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    (void)ok;

    const auto fs_ = s.stats();
    s.stop();
    worker.join();
    std::printf("paced: %.1f s of program in %.1f s; worst gap %.3f s; %zu events\n", cap.emitted.load() / double(kSr),
                now_s() - t0, fs_.worst_gap_seconds, cap.events.size());
    CHECK(fs_.worst_gap_seconds < 0.2, "no service gap of 0.2 s (worst %.3f s)", fs_.worst_gap_seconds);
    CHECK(cap.contiguous && cap.finite, "paced audio contiguous and finite");
    CHECK(cap.of(Kind::Error).empty(), "no Error events");
    CHECK(fs_.emitted - fs_.playhead <= static_cast<std::int64_t>(1.0 * kSr), "at most the emit lead ahead (%.2f s)",
          (fs_.emitted - fs_.playhead) / double(kSr));
}

}  // namespace

int main() {
    if (const char* e = std::getenv("YUE2_E2E"); !e || std::string(e) != "1") {
        std::printf("SKIP: set YUE2_E2E=1 (minutes of GPU)\n");
        return 0;
    }
    brotensor::init();
    const fs::path mdir = model_dir();
    fs::path vdir = repo() / "weights" / "yue2-vae";
    if (const char* v = std::getenv("YUE2_VAE_DIR")) vdir = v;
    if (mdir.empty() || !fs::exists(vdir / "model.safetensors")) {
        std::printf("SKIP: needs the YuE2-3B folder and the VAE (%s)\n", vdir.string().c_str());
        return 0;
    }
    brosoundml::Yue2 y;
    brosoundml::Yue2Vae vae;
    const double t = now_s();
    y.load(mdir.string(), brotensor::Device::CUDA);
    vae.load(vdir.string(), brotensor::Device::CUDA, /*decoder_only=*/true);
    std::printf("load %.1f s\n", now_s() - t);
    unpaced(y, vae);
    paced(y, vae);
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
