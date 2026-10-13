#pragma once

// YuE2 streaming playback: a continuous, steerable "radio" over Yue2 + Yue2Vae.
//
// Design and measurements: yue2-research FINDINGS.md. One process, one device;
// the stages interleave on the thread that calls run():
//   plan      Yue2's planner (or a given ABC), then a prompt session;
//   semantic  a live AR session, sampled a few tokens at a time (the same
//             sampler and uniforms as Yue2::generate, so a seed gives the same
//             tokens), its KV cache growing on demand;
//   NAR       windows of 250 frames, then 500 (Yue2StreamOptions::windows),
//             each solved over 25 frames of inpainted left context and 25 of
//             lookahead, its AR prefix the prompt (rows reused from the
//             session) + the solved span's codec + MUSIC_END, latent positions
//             restarting at 0 (stream quality matches the whole-song render);
//   VAE       just in time, a few seconds ahead of the playhead, so a knob
//             that acts on the latents is heard as soon as the audio already
//             handed out has played.
// The playing song comes first. Then, one song at a time, the stream builds
// the songs queued after it, up to `ahead_songs` of them (about 22% of a 4090
// per song), each whole: plan, semantic tokens, latents, and its audio
// decoded, so a built song waits as finished audio in host memory (its AR
// session is released: device memory stays one song's build beside the
// playing one). A song built ahead waits for all its tokens and is solved in
// one NAR chunk, so it is the whole-song render itself, whenever that fits
// before it is needed (measured rate, 1.5x margin); else, and for the first
// song or one a skip waits for, it takes the windows. Unpaced, every song
// after the first is one chunk. Songs crossfade at the outro (equal power,
// 2-4 s); skip() to a built song is the skip fade alone.
//
// When fewer than `ahead_songs` songs wait after the playing one, a Want
// event asks the app for more (value = how many), so it can supply a fresh
// request; with radio on, a want unanswered for `want_grace_seconds` is
// filled from the last request with the next seed.
//
// Previews (`preview_seconds` > 0): each song's semantic stage stops at the
// preview plus the lookahead (its plan is still the whole song's), so a song
// ahead costs ~11 s of GPU, not ~45, and a pool of `ahead_songs` fills within
// a minute. An unchosen preview crossfades into the next one. make_full()
// continues a preview: its AR session resumes (reopened over its codec rows
// if released), and from the join (the last kJoinFrames not yet heard) the
// NAR re-solves with the continuation as lookahead; the make-full preempts the
// songs ahead, and its continuation is built first until it runs
// `air_margin_seconds` past the audio handed out, then the songs ahead are,
// then the rest of it. `auto_full` does that for each song as it goes on air.
//
// Steering (Yue2Knobs, each clamped below where the research found the music
// breaks): energy is a latent offset added at decode; texture and percussion
// are NAR steers; the semantic steers (Yue2Directions::semantic: arrangement,
// and the style axes) steer the AR stage. A song takes the knobs when it is
// queued and keeps them: a change applies to the songs queued after it, and
// every song that exists (on air, built ahead, being built, a preview made
// full) stays as it was started. To hear a change sooner, skip to the next
// song. So nothing on air changes once it is solved, and a song on air whose
// latents are complete is decoded to its end at once rather than a lead
// ahead of the playhead.
//
// Audio comes out as one program timeline (crossfades mixed in), contiguous
// from sample 0, 48 kHz planar stereo. Paced (the default), the stream hands
// out audio only `emit_lead_seconds` ahead of the playhead the player reports
// (set_playhead), so what is not yet handed out can still change; unpaced, it
// renders everything as fast as it can (offline, tests, the quality oracle).

#include "brosoundml/yue2.h"
#include "brosoundml/yue2_vae.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// A semantic (AR) steer: a residual-stream direction after `layer`, its
// scale clamped to [min, max] (where yue2-research found the music breaking
// or the song not ending).
struct Yue2SemanticAxis {
    std::string name;
    int layer = 21;
    std::vector<float> direction;   // 2048
    float min = 0, max = 0;
};

// Steering directions, raw differences of means of paired contrasts
// (yue2-research): a latent-space offset, NAR residual directions, and the
// semantic axes. builtin() is the set yue2-research measured
// (src/yue2_directions.cpp).
struct Yue2Directions {
    std::vector<float> latent;    // 64
    std::vector<float> nar_l21;   // 2048, NAR after layer 21
    std::vector<float> nar_l9;    // 2048, NAR after layer 9
    std::vector<Yue2SemanticAxis> semantic;   // arrangement first, then the style axes
    const Yue2SemanticAxis* axis(const std::string& name) const;
    static Yue2Directions builtin();
    static const Yue2Directions& builtin_ref();   // one shared copy
};

// Knob scales along the directions. Ranges are the clamps (the research's
// break points: offset 4 and NAR L21 >= 4 broke the audio, AR |s| >= 3 never
// ended, NAR L9 broke at 8); clamp() applies them.
struct Yue2Knobs {
    float energy = 0;        // latent offset          [-2, 1]  louder, brighter
    float texture = 0;       // NAR layer 21           [-2, 2]
    float percussion = 0;    // NAR layer 9            [-6, 6]  onsets
    // Semantic steers by axis name (Yue2Directions::semantic): "arrangement"
    // (energy, AR layer 21, [-2, 1], a real arrangement change) and the style
    // axes "density" (full band + / sparse -) and "timbre" (bright + / warm
    // -), AR layer 24, [-1.5, 1]. Absent = 0; clamp() drops unknown names
    // and zeros.
    std::map<std::string, float> semantic;
    float get(const std::string& axis) const;
    void set(const std::string& axis, float v);
    // Clamped to `dirs`' ranges (builtin_ref() by default).
    static Yue2Knobs clamp(const Yue2Knobs& k);
    static Yue2Knobs clamp(const Yue2Knobs& k, const Yue2Directions& dirs);
    bool operator==(const Yue2Knobs&) const = default;
};

// One song of the queue.
struct Yue2Song {
    Yue2Params params;              // style, lyrics, cot, seed, sampling, ode_steps; params.steer is ignored (knobs)
    std::vector<int32_t> semantic;  // non-empty: these codec ids (0..32767) instead of the AR stage (offline renders)
    std::string tag;                // the caller's label, echoed in events
};

struct Yue2StreamOptions {
    std::vector<int> windows{250, 500};   // NAR window sizes in frames (the last repeats)
    int left = 25, right = 25;            // inpainted left context, lookahead
    bool paced = true;                    // false: render as fast as possible (no playhead)
    double emit_lead_seconds = 0.75;      // paced: audio handed out ahead of the playhead
    double decode_lead_seconds = 6.0;     // paced: audio decoded ahead of the playhead
    double crossfade_seconds = 3.0;       // song to song, equal power (clamped to 2..4)
    double skip_fade_seconds = 0.5;       // skip()'s crossfade
    bool radio = true;                    // a want unanswered: continue with the last song's request, next seed
    int ahead_songs = 1;                  // songs kept built after the playing one (1..8)
    double want_grace_seconds = 10.0;     // radio: how long a Want waits for the app before the radio fills it
    // Previews (> 0): each song is built as its first `preview_seconds` only
    // (the plan is the whole song's, the semantic stage stops there plus the
    // lookahead), and plays that long, crossfading into the next as a song
    // end does. make_full() continues a preview into the whole song from
    // where its audio ends, seamlessly; auto_full does it for every song as
    // it goes on air. 0 = whole songs.
    double preview_seconds = 0;
    bool auto_full = false;
    // The playing song is built first only until its solved audio runs this
    // far past the audio handed out; beyond that the songs ahead are built
    // (so a skip has somewhere to go), then the rest of it. 0 = the playing
    // song always first.
    double air_margin_seconds = 20.0;
    // Played songs whose audio is kept for save_song() (0..16).
    int keep_played = 3;
    Yue2Knobs knobs;                      // initial
    Yue2Directions directions = Yue2Directions::builtin();
    // Oracle / test seams (run()'s thread): each landed window's core
    // latents (song, frame0, frames, frames * 64 floats, before the energy
    // offset; a re-solve repeats frames), and a song's codec ids when its
    // semantic stage ends.
    std::function<void(int, int, int, const float*)> on_latents;
    std::function<void(int, const std::vector<int32_t>&)> on_semantic;
    // Polled on run()'s thread (at least every 10 ms, and inside long ops):
    // true = stop(), as from another thread.
    CancelCheck cancel;
};

// A song as the stream built it, enough to render it again or edit it: its
// plan, its codec tokens, its latents as the NAR solved them (the energy knob
// is an offset added at decode: `energy` per frame, along the stream's
// latent direction) and its request. Frames cover what the saved audio holds
// (a preview's own frames, a played song's as it played).
struct Yue2StreamTake {
    Yue2Plan plan;
    std::vector<int32_t> semantic;   // codec ids, one per frame
    std::vector<float> latents;      // frames x 64, frame-major
    std::vector<float> energy;       // per frame: the offset decoded with it (0 = none)
    Yue2Params params;               // the request (seed, style, lyrics, sampling)
};

struct Yue2StreamEvent {
    enum class Kind {
        Plan,          // a song's score is ready: text = ABC (the next song's: editable, set_next_plan)
        SongStart,     // a song is placed on the program: sample = its first program sample
        SemanticDone,  // frames = the song's length in frames
        Window,        // a NAR window landed: frame0, frames, seconds = its solve time, value = mean latent norm
        Knobs,         // set_knobs() took effect (text = which moved): songs queued from now take them
        BreakWarning,  // a window's latent norm left the corpus range (6.5 .. 9.5): value = the norm
        SongEnd,       // sample = the song's last program sample + 1
        FirstAudio,    // seconds = request -> first audio handed out (time to first audio)
        Error,         // text = the message; the song is dropped
        Idle,          // nothing queued, everything played out (radio off)
        Want,          // fewer than ahead_songs wait after the playing one: value = how many more it wants
        Built,         // a song is built: its whole audio waits (frames = its length); a skip to it is instant
        Dropped,       // a queued song was removed (drop()); songs after it are placed again
        Full,          // make_full(): value 1 = the preview continues into the whole song, re-solved from
                       // frame0 (the join); 0 = too late (its successor is on air) or its song had ended
        Saved,         // save_song(): text = the path written, frames = the samples (48 kHz), value 1 = the
                       // whole song (a played one: as it played), 0 = a preview; -1 = nothing written,
                       // text = why. `take` = the song's take (its plan, tokens and latents)
    };
    Kind kind = Kind::Idle;
    int song = -1;                 // song index (queue order from 0)
    std::string tag;
    std::int64_t sample = 0;
    int frame0 = 0, frames = 0;
    double seconds = 0, value = 0;
    std::string text;
    std::shared_ptr<const Yue2StreamTake> take;   // Saved
};

// A snapshot for meters.
struct Yue2StreamStats {
    std::int64_t playhead = 0, emitted = 0;   // program samples
    struct Song {
        int index = -1;
        std::string tag;
        bool planned = false, semantic_done = false;
        int semantic_frames = 0, latent_frames = 0, decoded_frames = 0;
        std::int64_t start = -1, end = -1;     // program samples, -1 = not placed / not known
        double latent_norm = 0;                // the last window's mean latent norm
        bool built = false;                    // latents complete and the whole song decoded
        bool ready = false;                    // latents complete and the opening decoded: a skip to it is instant
        bool on_air = false;
        std::size_t host_bytes = 0;            // audio + latents + noise held for it in host memory
        Yue2Knobs knobs;                       // what it was started with, and keeps
        bool preview = false;                  // still a preview (preview_frames long)
        int preview_frames = 0;
        int estimate_frames = 0;               // the plan's length estimate (its ABC), 0 = none
        bool ended = false;                    // the semantic stage reached the song's own end
    };
    std::vector<Song> songs;                   // queued and playing, in order
    int ahead = 0;                             // songs after the playing one
    int ready_ahead = 0;                       // of those, ready (skip is instant)
    std::size_t host_bytes = 0;                // all songs'
    Yue2Knobs knobs;
    double first_audio_seconds = 0;            // the latest song's request -> first audio
    // Service gaps: the worker hands out audio (and decodes the playing
    // song's when it is about to run out) only between pieces of work, so a
    // piece longer than the emit lead starves the player. Each gap of 0.2 s or
    // more, the last 16, with what ran in it.
    struct Gap {
        double seconds = 0;
        // The piece of work last begun when it ended (plan / semantic /
        // window: prefill|step|readback / make_full / decode / save
        // ...); device work queued by an earlier piece can land in a later one.
        std::string what;
        int song = -1;           // the song it worked on (-1 = none)
        std::int64_t emitted = 0;   // the emit frontier when it ended
        double lead = 0;         // seconds handed out ahead of the playhead when it began
    };
    std::vector<Gap> gaps;
    double worst_gap_seconds = 0;
    // The worker's seconds since the stream opened, per piece of work (the
    // labels of Gap::what, each timed from its start to the next one's);
    // "idle" is the time it waited with nothing to do.
    struct Work {
        std::string what;
        double seconds = 0;
    };
    std::vector<Work> work;
};

class Yue2Stream {
public:
    using AudioFn = std::function<void(const StereoAudio& chunk, std::int64_t sample0)>;
    using EventFn = std::function<void(const Yue2StreamEvent&)>;

    // `model` and `vae` must outlive the stream. Callbacks run on run()'s thread.
    Yue2Stream(const Yue2& model, const Yue2Vae& vae, Yue2StreamOptions options, AudioFn on_audio,
               EventFn on_event = {});
    ~Yue2Stream();
    Yue2Stream(const Yue2Stream&) = delete;
    Yue2Stream& operator=(const Yue2Stream&) = delete;

    // Controls: any thread, never block on the model.
    void enqueue(Yue2Song song);
    void set_playhead(std::int64_t program_sample);   // samples the player has played
    void set_knobs(const Yue2Knobs& knobs);           // for the songs queued after it
    void skip();                                     // to the next song (built, or as soon as it is)
    void set_next_plan(const std::string& abc);      // the next song's score (an edited Plan event)
    // A queued song's score (by index, an edited Plan event): it is rebuilt
    // from the plan on, unless it is on air (then ignored).
    void set_plan(int song, const std::string& abc);
    // Remove a queued song that is not on air (built or not); the songs after
    // it move up. A Dropped event follows; an on-air or unknown index is ignored.
    void drop(int song);
    // A preview continues into the whole song (song -1 = the one heard now:
    // the newest on air, so during a crossfade the one coming in):
    // its semantic stage resumes after the preview's tokens, and its last
    // two seconds not yet handed out are re-solved with the continuation as
    // their lookahead, joined inpainted like a window edge. A Full event
    // follows. It builds before anything else when that song is on air.
    void make_full(int song = -1);
    void set_auto_full(bool on);
    // Write a song's own audio (no crossfade into its neighbours) to `path` as
    // a 16-bit stereo 48 kHz WAV, once it is whole: a song still building is
    // written when its latents are complete (the rest decoded for it; a
    // preview is its preview), one that plays out first as it played. The
    // last `keep_played` played songs can be saved too. A Saved event follows.
    void save_song(int song, const std::string& path);
    void stop();                                    // run() returns soon after

    // The worker: blocks until stop(), or (unpaced, radio off) until the
    // queue is played out. Throws only on misuse; model errors become Error
    // events and drop the song.
    void run();

    Yue2StreamStats stats() const;
    const Yue2StreamOptions& options() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
