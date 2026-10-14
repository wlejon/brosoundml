#pragma once

// YuE2 endless stream: no songs, one token history in 10 s chunks, steered by
// targets. Beside Yue2Stream (yue2_stream.h, the song radio), on the same
// model and VAE. Design and measurements: yue2-research FINDINGS 7-10; the
// reference implementation is its native/endless.js, which this reproduces
// token for token from the same seeds (yue2-research/native/endless_stream.js
// is the check).
//
// A chunk (`chunk_frames`, 250 = 10 s) is sampled by Yue2::splice opened by
// the last `memory` tokens of the history (positions re-based: the model sees
// a clip, not a song), cot off with no score, or the score with cot full.
// It is rendered by Yue2::inpaint over a window of up to `nar_left` solved
// frames kept on the left, the chunk, and `lookahead` frames of the next
// chunk solved and dropped (the next chunk re-solves them with this one
// kept), so a chunk is rendered once the next one's tokens exist. Audio is
// decoded just in time, `decode_lead_seconds` ahead of the playhead, with the
// energy knob added to the latents, so energy is heard as soon as the audio
// not yet handed out plays. Seeds: chunk k's splice seed*1000 + k, its render
// seed*1000 + 500 + k (endless.js's scheme).
//
// Targets. A target is what chunks are made under: style text, an optional
// score (ABC: under it the tempo is exact, Q: sets it), a lyric sheet
// (whole, or rolled: a window of lines moved past the last line the listener
// heard), and knobs (AR steers act on the chunk's splice, NAR steers on its
// render, energy at decode). set_target() applies from the first chunk whose
// tokens can still change: the first not sampled, or the newest sampled one
// while it is only the previous chunk's lookahead and there is time to sample
// it again (`retarget`). A path of targets stamped in program seconds applies
// each from the first chunk boundary at or after its time, so an app can
// draw a path ahead and the stream builds along it on time. Stats say which
// chunks are committed and where a target set now would first be heard.
//
// Style changes, as the research found best: with no score, the change chunk
// is opened by only `change_memory` tokens (12 = 0.5 s) of the old music.
// With a score, the measured bridge: a fresh clip of the new style (no
// history, a beat longer than a chunk), a bridge chunk spliced from the old
// music toward it (`bridge_candidates`, scored by how well the clip follows),
// the bridge probe-rendered with the chunks before it, its beat read with
// the spectral-flux sensor at the score's period (Q:), the fresh clip
// probe-rendered and cut by up to one beat so its beat lands on the bridge's,
// and the bridge's probe render kept as its render. A change takes two
// chunks then (the bridge, then the fresh clip), and starts early enough for
// its ~5 s of work.
//
// Branches: alternative continuations built from the same history beside
// the main line (another target or path), at most `max_branches`, each
// keeping pace with the main line. commit_branch() makes one the main line:
// seamlessly at its fork when that has not been handed out yet, else with a
// crossfade into it at the first chunk boundary it can make; the other
// branches are dropped. discard_branch() drops one. A stream costs ~30% of a
// 4090; each branch about as much again.
//
// Audio is one program timeline, contiguous from sample 0, 48 kHz planar
// stereo; program frame f (25 Hz) is sample f * 1920, chunk k starts at frame
// k * chunk_frames. Paced (the default), audio is handed out only
// `emit_lead_seconds` ahead of the playhead the player reports; the worker
// services the player inside every long operation (splice per token, render
// and decode while the device runs), and stats record any gap of 0.2 s or
// more between services. Unpaced (offline, tests, the harness check), it
// renders `max_chunks` chunks as fast as it can and returns.

#include "brosoundml/audio.h"
#include "brosoundml/yue2.h"
#include "brosoundml/yue2_stream.h"
#include "brosoundml/yue2_vae.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

class Whisper;

// YuE2's instrumental sheet (a stream with no lyrics sings none).
inline constexpr const char* kYue2Instrumental = "[Intro]\n\n[Instrumental]\n\n[Interlude]\n\n[Instrumental]\n";

struct Yue2EndlessTarget {
    std::string style;          // "[Tags]" text
    std::string score;          // ABC; empty = none (cot off). Under a score the chunks carry it (cot full)
    std::string lyrics;         // the sheet (YuE2's format, [section] lines); empty = instrumental
    // Rolled: each chunk's prefix carries `lyric_window` lines of the sheet
    // (section tags kept as lines and not counted), starting after the last
    // line the listener heard in the rendered chunks (a line heard when 60%
    // of its words are, near the window), one line on when it has not moved
    // for 3 chunks. Off: the whole sheet in every chunk (it loops on the first
    // verse and chorus). A change of sheet starts the roll again at line 0.
    bool roll = false;
    int lyric_window = 4;
    Yue2Knobs knobs;            // clamped
    std::string tag;            // the caller's label, echoed
    double at = -1;             // program seconds (path entries); < 0 = now
    bool same_music(const Yue2EndlessTarget& o) const { return style == o.style && score == o.score; }
};

enum class Yue2ChangeMode {
    Auto,     // no score: the memory drop; a score: the measured bridge
    Drop,     // always the memory drop (change_memory)
    Bridge,   // always the measured bridge (needs a score: a target without one drops)
    Plain,    // the style text changes, nothing else (barely moves the music: FINDINGS 7)
};

// The listener behind rolled lyrics: 16 kHz mono audio -> the text heard.
// Called on run()'s thread; `cancel` services the stream and must be polled.
using Yue2Listen = std::function<std::string(const AudioBuffer& audio16k, const CancelCheck& cancel)>;

struct Yue2EndlessOptions {
    int chunk_frames = 250;             // 25..1000
    int memory = 250;                   // tokens opening a chunk
    int change_memory = 12;             // tokens opening the first chunk of a new style (drop)
    int lookahead = 25;                 // frames of the next chunk solved and dropped
    int nar_left = 475;                 // solved frames kept on a render's left
    int ode_steps = 32;
    float cfg_scale = -1;               // < 0: the protocol default
    std::uint64_t seed = 1;
    Yue2ChangeMode change_mode = Yue2ChangeMode::Auto;
    int bridge_candidates = 4;
    int bridge_score_frames = 125;
    bool paced = true;
    double emit_lead_seconds = 0.75;    // audio handed out ahead of the playhead
    double decode_lead_seconds = 4.0;   // audio decoded ahead of the playhead (energy acts behind it)
    // A chunk is sampled once its start is this far ahead of the playhead, or
    // further when the measured costs ask for it (a chunk's render waits for
    // the next one's tokens, so the lead covers a chunk, the emit lead, a
    // splice and a render per line being built, the listener and a margin:
    // ~15.7 s on a 4090); a bridge starts `bridge_lead_seconds` earlier still.
    // The less it is, the sooner a target set now is heard.
    double ahead_seconds = 14.0;
    double bridge_lead_seconds = 7.0;
    bool retarget = true;               // a target set now may re-sample the newest chunk (see above)
    int max_chunks = 0;                 // 0 = endless (paced only); unpaced renders this many
    int max_branches = 2;               // 0..3
    double branch_fade_seconds = 1.0;   // a commit past its fork crossfades over this long (0.1..4)
    // Each chunk's tokens and latents on its Sampled / Built events (the
    // harness check and offline renders; the paced player needs neither).
    bool chunk_data = false;
    Yue2EndlessTarget target;           // the first chunk's (at is ignored)
    std::vector<Yue2EndlessTarget> path;   // initial timed targets
    Yue2Directions directions = Yue2Directions::builtin();
    Yue2Listen listen;                  // empty = rolled lyrics move one line per 3 chunks
    // Polled on run()'s thread (at least every 10 ms and inside long ops):
    // true = stop().
    CancelCheck cancel;
};

// One chunk as made (endless.js's chunks.json row).
struct Yue2EndlessChunk {
    int index = -1;
    int branch = 0;                     // 0 = the main line
    std::string tag, style;
    bool scored = false;
    bool change = false;                // the first chunk of a new style (a bridge's bridge chunk)
    std::string how;                    // "", "drop", "bridge", "fresh", "plain"
    int memory = 0;                     // tokens it was opened by
    bool retargeted = false;            // sampled again for a target that came after it was
    double splice_seconds = 0, inpaint_seconds = 0, decode_seconds = 0, listen_seconds = 0;
    int nar_window = 0;
    bool probe_render = false;          // rendered by its bridge's probe (kept)
    double latent_norm = 0;             // mean frame norm of its latents (corpus 6.5..9.5)
    std::vector<float> bridge_scores;   // bridge: each candidate's NLL of the fresh clip's opening
    // fresh: the beat fit (periods in latent frames, phases in beats) and the
    // frames cut from the clip's start; fit_seconds = the measurement's cost.
    bool phase_fit = false;
    double period = 0, stream_phase = 0, fresh_phase = 0, fit_seconds = 0;
    int cut = 0;
    std::string lyrics;                 // the sheet text its prefix carried
    int lyric_from = -1, lyric_to = -1; // rolled: the window [from, to) of sung lines
    std::string heard;                  // rolled + a listener: what was heard in it
    std::vector<int> heard_lines;       // sung lines heard (60% of their words)
    Yue2Knobs knobs;
    bool rendered = false;
};

struct Yue2EndlessEvent {
    enum class Kind {
        Planned,      // a target placed: chunk = where it applies, sample = where it is first heard, text = tag
        Sampled,      // a chunk's tokens exist (row; tokens with chunk_data): committed unless it is the newest
        Built,        // a chunk is rendered (row; latents with chunk_data), value = its latent norm
        StyleChange,  // a change made at `chunk`: text = how ("drop" / "bridge"), sample = where it is heard
        PhaseFit,     // a bridge's beat fit (row of the fresh clip): value = the cut in frames
        Heard,        // rolled lyrics: what the listener heard in a rendered chunk (row: heard, heard_lines)
        BreakWarning, // a chunk's latent norm left the corpus range (6.5 .. 9.5): value = the norm
        ChunkStart,   // a chunk's first sample handed out: sample
        Branch,       // a branch opened: branch, chunk = its fork
        Committed,    // a branch made the main line: branch, chunk = the boundary, value = fade seconds (0 = seamless)
        Discarded,    // a branch dropped: branch, text = why
        Knobs,        // set_knobs(): text = which moved; energy from `sample`, the rest from `chunk`
        Error,        // text = the message
        Done,         // unpaced: max_chunks played out
    };
    Kind kind = Kind::Done;
    int chunk = -1;
    int branch = 0;
    std::string tag;
    std::int64_t sample = 0;
    double seconds = 0, value = 0;
    std::string text;
    std::shared_ptr<const Yue2EndlessChunk> row;
    std::shared_ptr<const std::vector<int32_t>> tokens;   // Sampled with chunk_data: the chunk's codec ids
    std::shared_ptr<const std::vector<float>> latents;    // Built with chunk_data: frames x 64 (frame0 = chunk start)
    int frame0 = 0;                                       // Built: the first frame of `latents`
};

struct Yue2EndlessStats {
    std::int64_t playhead = 0, emitted = 0, decoded = 0;   // program samples
    int playing = 0;                 // the chunk at the playhead
    int sampled = 0;                 // main line: chunks with tokens
    int rendered = 0;                // main line: chunks with latents
    int committed = 0;               // chunks whose tokens can no longer change (the rest: `sampled` - this)
    int open_chunk = 0;              // where a target set now would apply
    std::int64_t open_sample = 0;    // ... and where it would first be heard
    double ahead_seconds = 0;        // audio decoded ahead of the playhead
    Yue2EndlessTarget target;        // the newest chunk's
    int path = 0;                    // timed targets not yet applied
    std::vector<Yue2EndlessChunk> chunks;   // main line, from the playing chunk on
    struct Branch {
        int id = 0, fork = 0, sampled = 0, rendered = 0;
        std::string tag;
        bool committing = false;
    };
    std::vector<Branch> branches;
    std::vector<Yue2StreamStats::Gap> gaps;   // the last 16 gaps of 0.2 s or more without a service
    double worst_gap_seconds = 0;
    std::vector<Yue2StreamStats::Work> work;  // the worker's seconds per kind of work
    // Measured cost per chunk (seconds, running means): splice, render, decode, listen.
    double splice_seconds = 0, render_seconds = 0, decode_seconds = 0, listen_seconds = 0;
};

class Yue2Endless {
public:
    using AudioFn = std::function<void(const StereoAudio& chunk, std::int64_t sample0)>;
    using EventFn = std::function<void(const Yue2EndlessEvent&)>;

    // `model` and `vae` must outlive the stream. Callbacks run on run()'s thread.
    Yue2Endless(const Yue2& model, const Yue2Vae& vae, Yue2EndlessOptions options, AudioFn on_audio,
                EventFn on_event = {});
    ~Yue2Endless();
    Yue2Endless(const Yue2Endless&) = delete;
    Yue2Endless& operator=(const Yue2Endless&) = delete;

    // Controls: any thread, never block on the model.
    void set_playhead(std::int64_t program_sample);
    // A target now (target.at < 0: replaces the path) or on the path (at >= 0:
    // replaces the path's entries at or after its time).
    void set_target(const Yue2EndlessTarget& target);
    void set_path(std::vector<Yue2EndlessTarget> path);   // the timed targets ahead (replaces them)
    // The newest target's knobs: energy from the audio not yet handed out,
    // the rest from the first chunk that can still change.
    void set_knobs(const Yue2Knobs& knobs);
    // A branch from the first chunk whose tokens can still change, under
    // `target` (at < 0) and then `path`. Its id comes back in the Branch event
    // (and stats); ids count from 1 in call order.
    int branch(const Yue2EndlessTarget& target, std::vector<Yue2EndlessTarget> path = {});
    void commit_branch(int id);
    void discard_branch(int id);
    void stop();

    // The worker: until stop(), or (unpaced) until max_chunks are handed out.
    void run();

    Yue2EndlessStats stats() const;
    const Yue2EndlessOptions& options() const;

    // A listener over Whisper (16 kHz mono in, the decoded text out, special
    // tokens skipped, trimmed): transcribe with `prompt_ids` (build_prompt
    // ("en", "transcribe", false)), at most `max_new_tokens`, decoded by
    // `decode`. Both must outlive the stream.
    static Yue2Listen whisper_listener(const Whisper& whisper, std::vector<int32_t> prompt_ids,
                                       std::function<std::string(const std::vector<int32_t>&)> decode,
                                       int max_new_tokens = 120);

    // The score's beat in latent frames (25 Hz) from its Q: header ("Q:1/4=90"
    // -> 16.667; "Q:90" as quarters); 0 when it has none.
    static double score_beat_frames(const std::string& abc);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace brosoundml
