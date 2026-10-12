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
// Steering (Yue2Knobs, each clamped below where the research found the music
// breaks): energy is a latent offset (re-decoded, heard within the emit lead);
// texture and percussion are NAR steers (windows re-solved from ~1.5 s ahead);
// the semantic steers (Yue2Directions::semantic: arrangement, and the style
// axes) steer the AR stage (the song continues from ~2.5 s ahead under them).
// interrupt() is the two-stage "make it energetic": an instant latent offset
// with a 0.5 s ramp, then an arrangement-steered continuation from the
// playing frame + the stage's latency, joined at a window edge (inpainted),
// after which the offset ramps back to the energy knob. The knobs carry into
// every song built after the change. A song already built when a knob moves
// keeps what it was built with until it plays (so a skip to it stays
// instant), then catches up as the playing song does: energy from its first
// sample (re-decoded just before it starts), the others from ~1.5-2.5 s in.
// A song being built, or not started, takes the change from its start.
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
    // Interrupt stage 2 cuts at the audio handed out at the request + this
    // (its own latency, request -> landed: continuation tokens + one
    // 250-frame window), then measured and adapted. Heard about
    // emit_lead_seconds + this after the request.
    double interrupt_latency_seconds = 1.75;
    double knob_latency_seconds = 1.5;    // texture / percussion re-solve from this far ahead
    bool radio = true;                    // a want unanswered: continue with the last song's request, next seed
    int ahead_songs = 1;                  // songs kept built after the playing one (1..8)
    double want_grace_seconds = 10.0;     // radio: how long a Want waits for the app before the radio fills it
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

struct Yue2StreamEvent {
    enum class Kind {
        Plan,          // a song's score is ready: text = ABC (the next song's: editable, set_next_plan)
        SongStart,     // a song is placed on the program: sample = its first program sample
        SemanticDone,  // frames = the song's length in frames
        Window,        // a NAR window landed: frame0, frames, seconds = its solve time, value = mean latent norm
        Interrupt,     // stage 1 (value = 1) or 2 (value = 2) audible from `sample`; seconds = request -> audible
        Knobs,         // knobs applied to `song` (text = which); sample = the emit frontier then, frame0 = the
                       // song frame it changes from (a continuation's or re-solve's cut; 0 = the whole song)
        BreakWarning,  // a window's latent norm left the corpus range (6.5 .. 9.5): value = the norm
        SongEnd,       // sample = the song's last program sample + 1
        FirstAudio,    // seconds = request -> first audio handed out (time to first audio)
        Error,         // text = the message; the song is dropped
        Idle,          // nothing queued, everything played out (radio off)
        Want,          // fewer than ahead_songs wait after the playing one: value = how many more it wants
        Built,         // a song is built: its whole audio waits (frames = its length); a skip to it is instant
        Dropped,       // a queued song was removed (drop()); songs after it are placed again
    };
    Kind kind = Kind::Idle;
    int song = -1;                 // song index (queue order from 0)
    std::string tag;
    std::int64_t sample = 0;
    int frame0 = 0, frames = 0;
    double seconds = 0, value = 0;
    std::string text;
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
        Yue2Knobs knobs;                       // what it is built / steered with (energy: its offset now)
    };
    std::vector<Song> songs;                   // queued and playing, in order
    int ahead = 0;                             // songs after the playing one
    int ready_ahead = 0;                       // of those, ready (skip is instant)
    std::size_t host_bytes = 0;                // all songs'
    Yue2Knobs knobs;
    double first_audio_seconds = 0;            // the latest song's request -> first audio
    double interrupt1_seconds = 0, interrupt2_seconds = 0;   // request -> audible, the last interrupt
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
    void set_knobs(const Yue2Knobs& knobs);
    // Two-stage interrupt: +1 = energetic, -1 = calm (scaled, |amount| <= 1).
    void interrupt(float amount = 1.0f);
    void skip();                                     // to the next song (built, or as soon as it is)
    void set_next_plan(const std::string& abc);      // the next song's score (an edited Plan event)
    // A queued song's score (by index, an edited Plan event): it is rebuilt
    // from the plan on, unless it is on air (then ignored).
    void set_plan(int song, const std::string& abc);
    // Remove a queued song that is not on air (built or not); the songs after
    // it move up. A Dropped event follows; an on-air or unknown index is ignored.
    void drop(int song);
    void stop();                                     // run() returns soon after

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
