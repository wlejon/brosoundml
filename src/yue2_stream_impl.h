#pragma once

// Yue2Stream internals (yue2_stream.h): a song's build state and the
// scheduler's. yue2_stream.cpp holds the controls, the run loop and the
// program mix; yue2_stream_song.cpp a song's stages (plan, semantic sampler,
// NAR windows, just-in-time decode) and the knob reactions.

#include "brosoundml/yue2_stream.h"

#include "yue2_internal.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>

namespace brosoundml::yue2 {

using StreamClock = std::chrono::steady_clock;

struct Song {
    int index = -1;
    Yue2Song req;
    Yue2Params p;                      // resolved request
    StreamClock::time_point queued_at;
    bool failed = false;

    // ── plan ──
    bool planned = false;
    Yue2Plan plan;
    int P = 0;                         // prompt tokens
    std::vector<int32_t> neg;          // CFG negative prefix (empty = no CFG)
    float cfg = 1.0f;
    bool plan_announced = false;

    // ── semantic ──
    std::unique_ptr<ArSession> sess;   // branch 0 = prompt + codec, branch 1 = negative (CFG)
    Yue2Sampling sampling;             // guarded
    UniformFn uniform;
    std::vector<int32_t> tokens;       // sampled absolute ids (the sampler's history)
    std::vector<int32_t> codec;        // tokens - CODEC_OFFSET
    bool sem_done = false;
    StageHooks ar_hooks;               // arrangement steer (semantic AR L21)
    float ar_scale = 0;
    int interrupts = 0;                // continuations so far (reseeds each)

    // ── NAR ──
    std::vector<float> noise;          // song_noise for the budget
    std::vector<float> lat;            // frame-major, lat_extent frames (placeholders past lat_valid)
    int lat_valid = 0;                 // frames solved under the current conditions
    int lat_extent = 0;                // frames held (>= lat_valid; the rest old, to be re-solved)
    int win_from = 0, win_k = 0;       // window schedule restarts at win_from
    int gen = 0;                       // bumped by every re-solve / continuation (stale windows are dropped)
    StageHooks nar_hooks;              // texture / percussion
    float nar21 = 0, nar9 = 0;
    double last_norm = 0;

    // ── audio ──
    std::vector<float> off;            // per-frame latent offset scale (energy), sized with lat
    float off_tail = 0;                // the offset of frames not solved yet
    std::vector<float> L, R;           // song audio (song samples)
    int dec_frame = 0;                 // frames [0, dec_frame) decoded under the current latents
    std::int64_t natural = -1;         // the decoder's length for the final frames (-1 unknown)
    std::int64_t start = -1;           // program sample of song sample 0 (-1 = not placed)
    std::int64_t fade_in = 0;          // program samples it fades in over (its predecessor's overlap)
    std::int64_t cut = -1;             // song samples it plays (-1 = natural)
    bool start_announced = false, first_audio = false;

    int frames() const { return sem_done ? static_cast<int>(codec.size()) : -1; }   // final length, -1 unknown
    std::int64_t length() const { return cut >= 0 ? cut : natural; }               // song samples, -1 unknown
    std::int64_t end() const { return start >= 0 && length() >= 0 ? start + length() : -1; }
};

// Commands from the control side (any thread), applied by the worker.
struct Command {
    enum class Kind { Enqueue, Knobs, Interrupt, Skip, NextPlan, Stop } kind;
    Yue2Song song;
    Yue2Knobs knobs;
    float amount = 0;
    std::string abc;
    StreamClock::time_point at = StreamClock::now();
};

}  // namespace brosoundml::yue2

namespace brosoundml {

struct Yue2Stream::Impl {
    const yue2::Core& core;
    const Yue2Vae& vae;
    Yue2StreamOptions opt;
    AudioFn on_audio;
    EventFn on_event;

    // control side
    mutable std::mutex mu;
    std::condition_variable cv;
    std::deque<yue2::Command> commands;
    std::atomic<std::int64_t> playhead{0};
    std::atomic<yue2::StreamClock::rep> playhead_at{0};   // when it was reported
    std::atomic<bool> stopping{false};
    Yue2StreamStats snapshot;          // under mu

    // worker side
    std::deque<std::unique_ptr<yue2::Song>> songs;   // playing first; finished ones popped
    int next_index = 0;
    Yue2Knobs knobs;
    std::int64_t emitted = 0;          // program samples handed out
    bool in_service = false;
    const yue2::Song* busy = nullptr;  // the song a long op is running on (never freed under it)
    Yue2Song last_req;                 // the radio's template
    bool have_last = false;
    bool idle_announced = false;
    // deferred reactions (applied by step(), never inside a long op)
    bool skip_pending = false;
    yue2::StreamClock::time_point skip_at{};
    bool knobs_pending = false;
    Yue2Knobs knobs_applied;           // what the songs were last built / steered with
    std::string next_plan;             // set_next_plan's ABC, pending
    // interrupt bookkeeping
    yue2::StreamClock::time_point int_at{};
    float int_amount = 0;
    bool int_stage2_pending = false;   // a continuation to start
    int int_song = -1, int_cut = -1;   // the continuation's song and cut frame (stage 2 in flight)
    int int_ev_song = -1;              // the song the interrupt acts on (events)
    int int_frame = -1;                // its emit frontier at the request (stage 2's cut counts from it)
    std::int64_t int_audible1 = -1, int_audible2 = -1;   // program samples, reported when played
    double int_latency_est = 0;        // measured stage 2 latency (adapts opt.interrupt_latency_seconds)
    double first_audio_s = 0, int1_s = 0, int2_s = 0;

    Impl(const yue2::Core& c, const Yue2Vae& v, Yue2StreamOptions o, AudioFn a, EventFn e);

    int R = 1920;                      // samples per latent frame
    static constexpr int kHalo = 12;   // the decoder's (Yue2Vae::required_halo)

    // ── yue2_stream.cpp ──
    void apply(yue2::Command& c);
    void drain_commands();
    void service();                    // commands + emit + an urgent decode; safe inside long ops
    bool step();                       // one unit of work; false = nothing to do
    void emit();
    std::int64_t emit_target() const;
    std::int64_t decode_target(const yue2::Song& s) const;   // song frame to decode up to
    void place_next();
    void event(Yue2StreamEvent e);
    void publish();
    yue2::Song* playing();
    yue2::Song* next_song(const yue2::Song* after);
    void radio_refill();
    void do_skip();

    // ── yue2_stream_song.cpp ──
    void start_song(yue2::Song& s, Yue2Song req);
    void plan(yue2::Song& s);
    void semantic_step(yue2::Song& s, int tokens);
    bool window_ready(const yue2::Song& s, int* a, int* b) const;
    // Seconds until a song built ahead is needed (its program start, or its
    // predecessor's end), +inf unpaced; 0 when it is on air or a skip waits
    // for it (windows then).
    double ahead_budget(const yue2::Song& s) const;
    double whole_seconds(const yue2::Song& s, int frames) const;   // a one-chunk solve's estimate, with margin
    static constexpr double kWholeWaitSeconds = 45.0;   // wait for all the tokens when this far ahead
    double nar_spfs = 1.2e-4;          // NAR seconds per frame per ODE step (measured, the latest window)
    void solve_window(yue2::Song& s, int a, int b);
    bool decode_tile(yue2::Song& s, int max_frames);
    void set_ar_steer(yue2::Song& s, float scale);
    void set_nar_steer(yue2::Song& s, float t21, float p9);
    void set_offset(yue2::Song& s, int f0, float value, int ramp);
    void resolve_from(yue2::Song& s, int f);
    void continue_from(yue2::Song& s, int c, float ar_scale);
    void invalidate_audio(yue2::Song& s, int frame);
    int emitted_frame(const yue2::Song& s) const;   // song frame of the emit frontier
    CancelCheck servicing();                        // a cancel check that services instead
    // servicing(), and also cancels (the window re-runs later) when a
    // reaction step() must make first is pending: an interrupt's stage 2, a
    // knob change. A 500-frame window is ~2 s of GPU; stage 2 waiting it out
    // would be heard that much later. A whole-song chunk (`whole`) also
    // yields to a skip, which wants the song's opening in windows instead.
    CancelCheck preemptible(bool whole);
};

}  // namespace brosoundml
