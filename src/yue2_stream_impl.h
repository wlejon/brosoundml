#pragma once

// Yue2Stream internals (yue2_stream.h): a song's build state and the
// scheduler's. yue2_stream.cpp holds the controls, the run loop and the
// program mix; yue2_stream_song.cpp a song's stages (plan, semantic sampler,
// NAR windows, just-in-time decode) and the knob reactions.

#include "brosoundml/yue2_stream.h"

#include "yue2_internal.h"

#include <algorithm>
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
    StageHooks ar_hooks;               // the semantic steers
    std::map<std::string, float> ar;   // their scales (Yue2Knobs::semantic, clamped, no zeros)
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
    bool built_announced = false;
    bool dropped = false;              // drop(): removed by step() once no long op runs on it

    // ── preview ──
    int limit = -1;                    // a preview's length in frames (-1 = the whole song); the semantic
                                       // stage stops at limit + the lookahead, kept as the last window's
    bool ended = false;                // the semantic stage reached the song's own end
    int est_frames = 0;                // the plan's length estimate (0 = no ABC)

    // Final length, -1 unknown: the tokens, or a preview's limit.
    int frames() const {
        if (!sem_done) return -1;
        const int n = static_cast<int>(codec.size());
        return limit >= 0 ? std::min(limit, n) : n;
    }
    // Tokens and latents complete: what a knob change would have to rebuild.
    bool complete() const { return planned && !failed && sem_done && lat_valid >= frames(); }
    bool built() const { return complete() && dec_frame >= frames(); }
    std::int64_t length() const { return cut >= 0 ? cut : natural; }               // song samples, -1 unknown
    std::int64_t end() const { return start >= 0 && length() >= 0 ? start + length() : -1; }
};

// Commands from the control side (any thread), applied by the worker.
struct Command {
    enum class Kind { Enqueue, Knobs, Interrupt, Skip, NextPlan, Drop, Full, AutoFull, Save, Stop } kind;
    Yue2Song song;
    int index = -1;                    // NextPlan (-1 = the next song) / Drop / Full (-1 = the playing one): the song
    Yue2Knobs knobs;
    float amount = 0;
    std::string abc;                   // NextPlan: the score; Save: the path
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
    struct Played {                    // a finished song's audio, for save_song()
        int index = -1;
        std::string tag;
        std::vector<float> L, R;
    };
    std::deque<Played> played;         // the last keep_played, oldest first
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
    bool knobs_pending = false;        // texture / percussion / semantic moved: step() retunes the songs
    std::vector<std::pair<int, std::string>> plan_edits;   // set_plan / set_next_plan (-1 = the next song), pending
    std::vector<int> full_pending;     // make_full (-1 = the playing song), pending
    std::vector<std::pair<int, std::string>> save_pending;   // save_song: (song, path) until it is whole
    // wants (ahead_songs)
    int want_announced = 0;            // the last Want's value (0 = none outstanding)
    yue2::StreamClock::time_point want_since{};            // when the pool first fell short
    bool want_open = false;
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
    void radio_refill(int n = 1);
    void do_skip();
    bool on_air(const yue2::Song& s) const { return s.start >= 0 && s.start <= emitted; }
    int pool() const;                  // live songs queued after the playing one
    void check_want();                 // Want events, and the radio's fill after the grace
    void announce();                   // SongStart / Built events
    void unplace_after(const yue2::Song* s);   // songs after s are placed again
    bool drop_marked();                // erase dropped songs no long op runs on
    yue2::Song* to_build();            // the first song after the playing one, within ahead_songs, not built
    // Knobs: apply the current ones to a song (see yue2_stream.h). A song not
    // on air whose latents are complete is left for when it plays.
    void retune(yue2::Song& s, bool on_air_now, bool* changed = nullptr);
    void energy_before_air(yue2::Song& s);     // a built song about to play: its offset from frame 0
    // Write a song's audio as held now (a Saved event); step_saves() calls it
    // once a pending save's song is complete and decoded, or has played.
    void save_song(int index, const std::string& path);
    bool step_saves();                 // one unit of a pending save's work; false = none
    bool steer_current(const yue2::Song& s) const;

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
    // `whole`: past the paced decode lead (a save decodes the whole song).
    bool decode_tile(yue2::Song& s, int max_frames, bool whole = false);
    void set_ar_steer(yue2::Song& s, const std::map<std::string, float>& scales);
    void set_nar_steer(yue2::Song& s, float t21, float p9);
    void set_offset(yue2::Song& s, int f0, float value, int ramp);
    void resolve_from(yue2::Song& s, int f);
    // Continue the song from frame c under `scales` (returns the cut used, -1
    // when nothing was continued: given tokens, past its end); always records
    // the scales.
    int continue_from(yue2::Song& s, int c, const std::map<std::string, float>& scales);
    // A preview continues into the whole song; returns the join frame (the
    // re-solve's start), -1 = nothing to continue (its song had ended).
    int make_full(yue2::Song& s);
    static constexpr int kJoinFrames = 50;   // a preview's tail re-solved with the continuation as lookahead
    void open_session(yue2::Song& s, int codec_rows, bool stepped = false);   // prompt (+ negative) + codec[0, codec_rows)
    void release_session(yue2::Song& s);                // a song built ahead: its AR session's device memory
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
