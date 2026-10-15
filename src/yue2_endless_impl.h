#pragma once

// Yue2Endless internals (yue2_endless.h): one Line per history (the main
// line and its branches), the program buffer, the control queue and the
// worker's scheduling state. Split by responsibility:
//   yue2_endless.cpp         controls, the run loop, scheduling, stats
//   yue2_endless_program.cpp the program: decode just in time, energy, emit, service
//   yue2_endless_chunk.cpp   targets, plans, sampling a chunk, rendering it
//   yue2_endless_bridge.cpp  the measured bridge under a score
//   yue2_endless_lyrics.cpp  rolled lyrics, the lyric head's reading
//   yue2_endless_branch.cpp  branches: fork, keep pace, commit, discard

#include "brosoundml/yue2_endless.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <optional>

namespace brosoundml {

namespace endless {

using Clock = std::chrono::steady_clock;
constexpr int kD = 64;        // latent dim
constexpr int kR = 1920;      // samples per latent frame
constexpr int kHalo = 12;     // the VAE's required halo
constexpr int kSr = 48000;
constexpr int kBlock = 25;    // the program is decoded a second at a time

// The lyric pointer (yue2-research FINDINGS 11): attention head 14.10 points
// at the lyric word being sung. A line is reached in a chunk when the head
// pointed into kReachWords of its words, each for kReachFrames frames or
// more, counting frames whose mass on the window's words is kReachGate or more.
constexpr int kLyricLayer = 14, kLyricHead = 10;
constexpr double kReachGate = 0.2, kReachWords = 0.6;
constexpr int kReachFrames = 2;

// The roll's place in a sheet: the window starts at `at`, after the
// furthest line reached.
struct Roll {
    std::string sheet;                     // the sheet it rolls (a change resets it)
    std::vector<std::string> lines;        // its non-empty lines
    std::vector<int> sung;                 // indices of lines that are not [section] tags
    int at = 0, reached = -1;
};

// A window's words as columns of a chunk's prompt (lyric_track.py
// lyric_word_spans): the tokens [a, b) covering each word.
struct LyricWords {
    std::vector<std::string> words;
    std::vector<int> lines;                // sheet line (sung-line index) of each word
    std::vector<std::pair<int, int>> cols;
};

// A chunk as the worker holds it: its public row, the target it was made
// under, the lyric text its prefix carried and the roll before it (a
// retarget samples it again from there).
struct Chunk {
    Yue2EndlessChunk row;
    Yue2EndlessTarget target;
    int roll_at = 0, roll_reached = -1;    // the roll before this chunk was sampled
    std::string roll_sheet;
    bool fresh = false;                    // a bridge's fresh clip (never re-sampled)
};

struct Line {
    int id = 0;                            // 0 = main
    std::uint64_t seed = 1;
    std::vector<int32_t> tokens;           // the history, one codec id per frame
    int lat_base = 0;                      // latents held from this frame
    std::vector<float> lat;                // frames [lat_base, lat_base + size / 64), frame-major
    int rendered = 0;                      // chunks with latents
    int fork = 0;                          // branch: its first own chunk
    std::deque<Chunk> chunks;              // [chunk_base, sampled)
    int chunk_base = 0;
    Yue2EndlessTarget base;                // governs chunks before the path's first entry
    std::map<int, Yue2EndlessTarget> path; // chunk -> target from it
    Roll roll;
    bool committing = false;
    bool failed = false;
    int deferred = -1;   // a change sample() put off and announced for this chunk (-1: none)

    int sampled() const { return chunk_base + static_cast<int>(chunks.size()); }
    Chunk& chunk(int k) { return chunks[static_cast<std::size_t>(k - chunk_base)]; }
    const Chunk& chunk(int k) const { return chunks[static_cast<std::size_t>(k - chunk_base)]; }
    bool has_chunk(int k) const { return k >= chunk_base && k < sampled(); }
    float* lat_at(int frame) { return lat.data() + static_cast<std::size_t>(frame - lat_base) * kD; }
    const float* lat_at(int frame) const { return lat.data() + static_cast<std::size_t>(frame - lat_base) * kD; }
    void lat_ensure(int frames);   // hold frames up to `frames` (zeros)
};

struct Command {
    enum class Kind { Target, Path, Knobs, Branch, Commit, Discard };
    Kind kind = Kind::Target;
    Yue2EndlessTarget target;
    std::vector<Yue2EndlessTarget> path;
    Yue2Knobs knobs;
    int id = 0;
};

}  // namespace endless

struct Yue2Endless::Impl {
    using Line = endless::Line;
    using Chunk = endless::Chunk;
    using Clock = endless::Clock;

    const Yue2& model;
    const Yue2Vae& vae;
    Yue2EndlessOptions opt;
    AudioFn on_audio;
    EventFn on_event;
    int C = 250;

    // ── controls ──
    mutable std::mutex mu;
    std::condition_variable cv;
    std::deque<endless::Command> commands;
    std::atomic<std::int64_t> playhead{0};
    std::atomic<bool> stopping{false};
    std::atomic<int> next_branch_id{1};
    Yue2EndlessStats snapshot;

    // ── lines ──
    std::unique_ptr<Line> main;
    std::vector<std::unique_ptr<Line>> branches;
    std::map<std::string, Yue2Plan> plans;   // by style | score | lyric text

    // ── the program ──
    std::int64_t emitted = 0;
    std::int64_t prog_base = 0;              // first sample held in L / R
    std::vector<float> L, R;                 // decoded program samples [prog_base, prog_base + size)
    int dec_frame = 0;                       // program frames decoded under the current latents
    std::int64_t xf_at = -1;                 // a re-decode meets the old samples here (20 ms crossfade)
    std::map<int, float> energy;             // frame -> the energy offset from it (ramped over 0.5 s)
    int last_started = -1;                   // the last chunk whose ChunkStart was sent
    bool done = false;

    // ── servicing ──
    bool in_service = false, in_decode = false;
    Clock::time_point service_end{};
    double service_end_lead = 0;
    std::deque<Yue2StreamStats::Gap> gaps;
    double worst_gap = 0;
    const char* doing = "idle";
    int doing_chunk = -1;
    Clock::time_point doing_since{};
    std::vector<std::pair<const char*, double>> work_s;
    // A splice in progress, preempted when the target of the chunk it makes changes.
    Line* sampling = nullptr;
    int sampling_k = -1;
    Yue2EndlessTarget sampling_target;
    bool preempt = false;

    // ── measured costs (seconds, running means) ──
    double est_splice = 1.0, est_render = 2.0, est_decode = 0.15, est_bridge = 7.0;

    Impl(const Yue2& m, const Yue2Vae& v, Yue2EndlessOptions o, AudioFn a, EventFn e);

    // yue2_endless.cpp
    void apply(endless::Command& c);
    void drain_commands();
    bool step();
    void publish();
    void event(Yue2EndlessEvent e);
    void work(const char* what, int chunk = -1);
    double seconds_until(std::int64_t sample) const;   // paced: until the playhead reaches it (huge unpaced)
    int final_chunks() const { return opt.max_chunks; }
    bool finished_sampling(const Line& l) const { return opt.max_chunks > 0 && l.sampled() >= opt.max_chunks; }

    // yue2_endless_chunk.cpp: targets and making chunks
    const Yue2EndlessTarget& target_for(const Line& l, int k) const;
    void fold(Line& l, int k);
    int open_chunk(const Line& l) const;
    bool retargetable(const Line& l, int k) const;
    void place_target(Line& l, Yue2EndlessTarget t, bool now);
    const Yue2Plan& plan_for(const Yue2EndlessTarget& t, const std::string& lyrics);
    Yue2Params params_for(const Yue2EndlessTarget& t, const std::string& lyrics, std::uint64_t seed, bool ar) const;
    void nar_steer(Yue2Params& p, const Yue2Knobs& k, int frames) const;
    bool needs_bridge(const Line& l, int k, const Yue2EndlessTarget& t) const;
    bool bridge_fits(const Line& l) const;   // paced: a bridge sampled now is rendered before it plays
    double sample_lead() const;              // seconds ahead of a chunk's start its sampling begins
    void planned(const Line& l, int k, const std::string& tag);
    bool sample_due(const Line& l) const;
    bool can_render(const Line& l) const;
    bool sample(Line& l);                     // false: preempted / stopping (nothing changed)
    void render(Line& l);
    void retarget(Line& l, int k);            // drop chunk k's tokens to sample it again
    bool retarget_due(Line& l);
    void append(Line& l, const int32_t* t, int n);
    void add_chunk(Line& l, Chunk c);
    void built(Line& l, int k, double seconds_share);
    CancelCheck servicing();
    CancelCheck preemptible(Line& l, int k, const Yue2EndlessTarget& t);
    void end_preemptible();

    // yue2_endless_bridge.cpp
    bool bridge(Line& l, int k, const Yue2EndlessTarget& t);

    // yue2_endless_lyrics.cpp
    std::string lyrics_of(const Yue2EndlessTarget& t, endless::Roll& roll, int* from, int* to) const;
    // The window's words as prompt columns of `plan` (empty unless rolled).
    endless::LyricWords lyric_words(const Yue2Plan& plan, const std::string& lyrics, int from) const;
    // Watch the lyric head over a splice's M (no-op without words).
    static void watch_lyrics(Yue2Splice& sp, const endless::LyricWords& w);
    // Read the head over M rows [row0, row0 + C) of `watched` into the row,
    // and move the roll past the lines reached.
    void read_lyrics(endless::Roll& roll, Chunk& c, const endless::LyricWords& w, const std::vector<float>& watched,
                     int row0) const;
    void lyrics_event(const Line& l, int k, int reached);

    // yue2_endless_branch.cpp
    void open_branch(int id, const Yue2EndlessTarget& t, std::vector<Yue2EndlessTarget> path);
    void sync_branch(Line& b);
    bool step_branches();
    void request_commit(int id);
    bool try_commit();
    void discard(int id, const std::string& why);
    Line* branch_of(int id);

    // yue2_endless_program.cpp
    // The player's turn: gaps, commands (unless `light`), preemption, audio
    // out, an urgent decode, stats. Light inside a decode: a command could
    // re-decode under it.
    void service(bool light = false);
    std::function<void()> waiting() { return [this] { service(true); }; }
    void emit();
    bool decode_tile(int max_frames);
    int decodable() const;                    // program frames decodable exactly now
    int decode_target() const;
    bool stream_complete() const;
    std::int64_t natural_end() const;         // unpaced: the program's last sample + 1 (-1 = open)
    float energy_at(int frame) const;
    void set_energy(int frame, float value);
    void redecode_from(int frame);
    void trim();
};

}  // namespace brosoundml
