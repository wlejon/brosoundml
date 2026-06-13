#include "brosoundml/gesture_spotter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

// Average a beat's acoustic signature over the window [start, end) of a
// snapshot stream: voicedness (periodicity) and brightness (centroid) over all
// frames, pitch (dominant_hz) over the voiced frames only. The same windowing
// the live matcher reproduces frame-by-frame, so enroll and match agree.
OnsetSig window_sig(const SensorSnapshot* snaps, int start, int end,
                    float voiced_thresh) {
    double per = 0.0, br = 0.0, pitch = 0.0;
    int len = 0, pn = 0;
    for (int i = start; i < end; ++i) {
        per += static_cast<double>(snaps[i].periodicity);
        br  += static_cast<double>(snaps[i].centroid);
        ++len;
        if (snaps[i].periodicity > voiced_thresh) {
            pitch += static_cast<double>(snaps[i].dominant_hz);
            ++pn;
        }
    }
    OnsetSig sig;
    sig.voiced = len > 0 ? static_cast<float>(per / len) : 0.0f;
    sig.bright = len > 0 ? static_cast<float>(br / len) : 0.0f;
    sig.pitch  = pn  > 0 ? static_cast<float>(pitch / pn) : 0.0f;
    return sig;
}

// Does an observed beat sound like the enrolled one? Voicedness and brightness
// must be within shape_tol (both normalized [0,1]); pitch is constrained only
// when the enrolled beat was actually pitched (so a click template doesn't
// demand a pitch, but a whistled/hummed beat does).
bool onset_sig_matches(const OnsetSig& tmpl, const OnsetSig& obs,
                       const GestureConfig& pol) {
    constexpr float kVoiced = 0.5f;
    if (std::abs(obs.voiced - tmpl.voiced) > pol.shape_tol) return false;
    if (std::abs(obs.bright - tmpl.bright) > pol.shape_tol) return false;
    if (tmpl.voiced >= kVoiced && tmpl.pitch > 0.0f) {
        if (obs.pitch <= 0.0f) return false;
        if (std::abs(obs.pitch - tmpl.pitch) / tmpl.pitch > pol.pitch_tol)
            return false;
    }
    return true;
}

}  // namespace

GestureSpotter::GestureSpotter(const GestureConfig& cfg) : cfg_(cfg) {}

void GestureSpotter::set_config(const GestureConfig& cfg) {
    cfg_ = cfg;
    for (auto& t : tmpls_)
        if (!t.has_override) t.policy = cfg;
}

// ─── Template construction from a sensor-frame stream ────────────────────────

GestureSpotter::Template GestureSpotter::build_template(
    const std::string& name, const SensorSnapshot* snaps, int n,
    const GestureConfig& pol, bool has_override) {
    // Extract the event timeline: onset frames, and tonal runs (start, mean
    // pitch, length). Onsets are broadband transients (clicks/taps); tonal
    // runs are sustained periodic sound (whistles/hums). A clip is a RHYTHM if
    // it has >= min_onsets onsets, else a TONE if it holds a long-enough tonal
    // run — and otherwise too sparse to be a gesture.
    std::vector<std::int64_t> onset_frames;
    std::vector<int>          onset_pos;     // array indices of onset frames
    int   best_run_len = 0;
    float best_run_hz  = 0.0f;
    float best_run_spread = 0.0f;
    bool  in_run = false;
    double run_hz_sum = 0.0;
    double run_hz_sumsq = 0.0;
    int    run_len = 0;
    auto close_run = [&] {
        if (in_run && run_len > best_run_len) {
            const double mean = run_hz_sum / static_cast<double>(run_len);
            const double var  = std::max(
                0.0, run_hz_sumsq / static_cast<double>(run_len) - mean * mean);
            best_run_len    = run_len;
            best_run_hz     = static_cast<float>(mean);
            best_run_spread = mean > 0.0
                ? static_cast<float>(std::sqrt(var) / mean) : 0.0f;
        }
        in_run = false; run_hz_sum = 0.0; run_hz_sumsq = 0.0; run_len = 0;
    };
    for (int i = 0; i < n; ++i) {
        const SensorSnapshot& s = snaps[i];
        if (s.onset) { onset_frames.push_back(s.frames); onset_pos.push_back(i); }
        if (s.tonal) {
            if (!in_run) { in_run = true; run_hz_sum = 0.0; run_hz_sumsq = 0.0; run_len = 0; }
            run_hz_sum   += static_cast<double>(s.dominant_hz);
            run_hz_sumsq += static_cast<double>(s.dominant_hz) *
                            static_cast<double>(s.dominant_hz);
            ++run_len;
        } else {
            close_run();
        }
    }
    close_run();

    Template t;
    t.name = name;
    t.policy = pol;
    t.has_override = has_override;

    // Classify by what dominates. A sustained tonal run (a whistle/hum) is
    // checked FIRST: a tone's abrupt attack also trips one or two onsets, so
    // an onsets-first rule would misread a whistle as a rhythm. A real click
    // rhythm is broadband and trips NO sustained tonal run, so it falls
    // through to the onset rule cleanly.
    if (best_run_len >= pol.min_tone_frames) {
        t.kind = GestureKind::Tone;
        t.tone_hz     = best_run_hz;
        t.tone_frames = best_run_len;
        t.tone_spread = best_run_spread;
    } else if (static_cast<int>(onset_frames.size()) >= pol.min_onsets) {
        t.kind = GestureKind::Rhythm;
        for (std::size_t i = 1; i < onset_frames.size(); ++i)
            t.intervals.push_back(static_cast<int>(
                onset_frames[i] - onset_frames[i - 1]));
        // Capture each beat's acoustic signature over a short post-onset
        // window, truncated at the next onset so it never bleeds into the
        // following beat (the live matcher truncates the same way).
        const int   sig_win = std::max(1, pol.onset_sig_frames);
        const float vthr    = pol.sensor.tonal_min_periodicity;
        for (std::size_t j = 0; j < onset_pos.size(); ++j) {
            const int start = onset_pos[j];
            int end = std::min(n, start + sig_win);
            if (j + 1 < onset_pos.size()) end = std::min(end, onset_pos[j + 1]);
            t.onset_sigs.push_back(window_sig(snaps, start, end, vthr));
        }
    } else {
        fail("GestureSpotter::enroll",
             "'" + name + "' is too sparse to be a gesture (need >= " +
             std::to_string(pol.min_onsets) + " onsets for a rhythm, or a "
             "tonal run >= " + std::to_string(pol.min_tone_frames) +
             " frames for a tone)");
    }
    return t;
}

int GestureSpotter::enroll_from_snapshots(const std::string& name,
                                          const SensorSnapshot* snaps, int n,
                                          const GestureConfig* policy_override) {
    if (n < 0 || (n > 0 && snaps == nullptr))
        fail("GestureSpotter::enroll_from_snapshots", "invalid snapshots");
    const GestureConfig& pol = policy_override ? *policy_override : cfg_;
    Template t = build_template(name, snaps, n, pol, policy_override != nullptr);
    remove(name);
    const int beats = t.kind == GestureKind::Rhythm
                          ? static_cast<int>(t.intervals.size()) + 1 : 1;
    tmpls_.push_back(std::move(t));
    return beats;
}

int GestureSpotter::enroll_from_audio(const std::string& name,
                                      const float* samples, int n,
                                      const GestureConfig* policy_override) {
    if (n < 0 || (n > 0 && samples == nullptr))
        fail("GestureSpotter::enroll_from_audio", "invalid samples");
    const GestureConfig& pol = policy_override ? *policy_override : cfg_;

    // Run the clip through a private SensorHub one frame at a time, collecting
    // a snapshot per frame: prime with one window, then advance one hop per
    // frame (SensorHub publishes exactly one frame per such feed).
    SensorHub hub(pol.sensor);
    const int win = pol.sensor.mel.win_length;
    const int hop = pol.sensor.mel.hop_length;
    std::vector<SensorSnapshot> snaps;
    int pos = 0;
    if (n >= win) {
        hub.feed(samples, win);
        snaps.push_back(hub.snapshot());
        pos = win;
        while (pos + hop <= n) {
            hub.feed(samples + pos, hop);
            snaps.push_back(hub.snapshot());
            pos += hop;
        }
    }
    return enroll_from_snapshots(name, snaps.data(),
                                 static_cast<int>(snaps.size()),
                                 policy_override);
}

bool GestureSpotter::remove(const std::string& name) {
    for (std::size_t i = 0; i < tmpls_.size(); ++i) {
        if (tmpls_[i].name == name) {
            tmpls_.erase(tmpls_.begin() + static_cast<std::ptrdiff_t>(i));
            return true;
        }
    }
    return false;
}

void GestureSpotter::clear() { tmpls_.clear(); }

std::vector<std::string> GestureSpotter::templates() const {
    std::vector<std::string> out;
    out.reserve(tmpls_.size());
    for (const auto& t : tmpls_) out.push_back(t.name);
    return out;
}

bool GestureSpotter::inspect(const std::string& name, GestureView& out) const {
    const int hop = cfg_.sensor.mel.hop_length > 0 ? cfg_.sensor.mel.hop_length : 160;
    const int sr  = cfg_.sensor.mel.sample_rate > 0 ? cfg_.sensor.mel.sample_rate : 16000;
    for (const auto& t : tmpls_) {
        if (t.name != name) continue;
        out.name     = t.name;
        out.kind     = t.kind;
        out.frame_ms = 1000.0f * static_cast<float>(hop) / static_cast<float>(sr);
        out.intervals = t.intervals;
        out.onsets    = t.onset_sigs;
        out.tone_hz   = t.tone_hz;
        out.tone_frames = t.tone_frames;
        out.tone_spread = t.tone_spread;
        return true;
    }
    return false;
}

// ─── Streaming match ─────────────────────────────────────────────────────────

std::vector<GestureEvent> GestureSpotter::feed(const SensorSnapshot& s) {
    std::vector<GestureEvent> events;

    // Finalize an in-flight beat capture: average its signature, append it to
    // the rolling history, and — once `need` beats are in hand — match BOTH the
    // inter-onset timing and each beat's sound shape. A re-performance that
    // lands on time but with the wrong timbre (e.g. a laugh at a tongue-click's
    // tempo) fails the per-beat signature check and does not fire.
    auto finalize_capture = [&events](GestureSpotter::Template& t) {
        OnsetSig sig;
        sig.voiced = t.cap_len > 0
            ? static_cast<float>(t.cap_per_sum / t.cap_len) : 0.0f;
        sig.bright = t.cap_len > 0
            ? static_cast<float>(t.cap_bright_sum / t.cap_len) : 0.0f;
        sig.pitch  = t.cap_pitch_n > 0
            ? static_cast<float>(t.cap_pitch_sum / t.cap_pitch_n) : 0.0f;
        t.cap_active = false;

        const int need = static_cast<int>(t.intervals.size()) + 1;
        t.onset_hist.push_back(t.cap_frame);
        t.sig_hist.push_back(sig);
        if (static_cast<int>(t.onset_hist.size()) > need) {
            t.onset_hist.erase(t.onset_hist.begin());
            t.sig_hist.erase(t.sig_hist.begin());
        }
        if (static_cast<int>(t.onset_hist.size()) < need || t.refractory > 0)
            return;

        const float tol = t.policy.tempo_tol;
        double err_sum = 0.0;
        bool   ok = true;
        for (std::size_t k = 0; k < t.intervals.size(); ++k) {
            const double tmpl = static_cast<double>(t.intervals[k]);
            const double obs  = static_cast<double>(
                t.onset_hist[k + 1] - t.onset_hist[k]);
            const double rel  = tmpl > 0 ? std::abs(obs - tmpl) / tmpl
                                         : (obs == 0 ? 0.0 : 1.0);
            if (rel > tol) { ok = false; break; }
            err_sum += rel;
        }
        // Per-beat sound-shape gate: every beat must match its enrolled
        // signature (voicedness, pitch-when-voiced, brightness).
        for (std::size_t k = 0; ok && k < t.sig_hist.size() &&
                                k < t.onset_sigs.size(); ++k)
            if (!onset_sig_matches(t.onset_sigs[k], t.sig_hist[k], t.policy))
                ok = false;
        if (!ok) return;

        GestureEvent ev;
        ev.name = t.name;
        ev.kind = GestureKind::Rhythm;
        ev.confidence = static_cast<float>(std::max(
            0.0, 1.0 - err_sum / static_cast<double>(t.intervals.size())));
        ev.start_frame = t.onset_hist.front();
        ev.end_frame   = t.onset_hist.back();
        events.push_back(ev);
        t.refractory = t.policy.refractory_frames;
        t.onset_hist.clear();   // re-arm from the next fresh sequence
        t.sig_hist.clear();
    };

    for (auto& t : tmpls_) {
        if (t.refractory > 0) --t.refractory;

        if (t.kind == GestureKind::Rhythm) {
            const int   sig_win = std::max(1, t.policy.onset_sig_frames);
            const float vthr    = t.policy.sensor.tonal_min_periodicity;
            // Advance an in-flight capture with this frame's sensors.
            if (t.cap_active) {
                t.cap_per_sum    += static_cast<double>(s.periodicity);
                t.cap_bright_sum += static_cast<double>(s.centroid);
                if (s.periodicity > vthr) {
                    t.cap_pitch_sum += static_cast<double>(s.dominant_hz);
                    ++t.cap_pitch_n;
                }
                ++t.cap_len;
            }
            if (s.onset) {
                // A new beat: close out the previous capture early (so back-to-
                // back beats don't smear), then arm a fresh window on this onset.
                if (t.cap_active) finalize_capture(t);
                t.cap_active   = true;
                t.cap_frame    = s.frames;
                t.cap_len      = 1;
                t.cap_per_sum  = static_cast<double>(s.periodicity);
                t.cap_bright_sum = static_cast<double>(s.centroid);
                t.cap_pitch_sum = s.periodicity > vthr
                    ? static_cast<double>(s.dominant_hz) : 0.0;
                t.cap_pitch_n  = s.periodicity > vthr ? 1 : 0;
                continue;
            }
            // Window full → finalize (this is where the match fires).
            if (t.cap_active && t.cap_len >= sig_win) finalize_capture(t);
        } else {  // Tone
            if (s.tonal) {
                if (!t.in_run) {
                    t.in_run = true; t.run_start = s.frames;
                    t.run_hz_sum = 0.0; t.run_hz_sumsq = 0.0;
                    t.run_len = 0; t.run_fired = false;
                }
                t.run_hz_sum   += static_cast<double>(s.dominant_hz);
                t.run_hz_sumsq += static_cast<double>(s.dominant_hz) *
                                  static_cast<double>(s.dominant_hz);
                ++t.run_len;
                if (t.run_fired || t.refractory > 0) continue;
                const float tol  = t.policy.tempo_tol;
                const int   dlo  = std::max(
                    t.policy.min_tone_frames,
                    static_cast<int>(std::floor(
                        static_cast<double>(t.tone_frames) * (1.0 - tol))));
                if (t.run_len < dlo) continue;
                const double mean_hz =
                    t.run_hz_sum / static_cast<double>(t.run_len);
                const double rel = t.tone_hz > 0.0f
                    ? std::abs(mean_hz - t.tone_hz) / t.tone_hz : 1.0;
                if (rel > t.policy.pitch_tol) continue;
                // Right note, but is it HELD? A cough or throat-clear can pass
                // a sustained tonal run whose mean lands in the band while its
                // pitch sweeps the whole way through. A real whistle holds the
                // pitch nearly constant. Reject runs whose per-frame pitch
                // spread is too wide to be one steady note.
                const double var = std::max(
                    0.0, t.run_hz_sumsq / static_cast<double>(t.run_len) -
                             mean_hz * mean_hz);
                const double spread = mean_hz > 0.0
                    ? std::sqrt(var) / mean_hz : 1.0;
                if (spread > t.policy.pitch_stability_tol) continue;
                GestureEvent ev;
                ev.name = t.name;
                ev.kind = GestureKind::Tone;
                ev.confidence = static_cast<float>(std::max(
                    0.0, 1.0 - rel / static_cast<double>(t.policy.pitch_tol)));
                ev.start_frame = t.run_start;
                ev.end_frame   = s.frames;
                events.push_back(ev);
                t.run_fired = true;
                t.refractory = t.policy.refractory_frames;
            } else {
                t.in_run = false; t.run_hz_sum = 0.0; t.run_hz_sumsq = 0.0;
                t.run_len = 0; t.run_fired = false;
            }
        }
    }
    return events;
}

void GestureSpotter::reset() {
    for (auto& t : tmpls_) {
        t.onset_hist.clear();
        t.sig_hist.clear();
        t.cap_active = false; t.cap_len = 0;
        t.cap_per_sum = 0.0; t.cap_bright_sum = 0.0;
        t.cap_pitch_sum = 0.0; t.cap_pitch_n = 0; t.cap_frame = 0;
        t.refractory = 0;
        t.in_run = false; t.run_start = 0;
        t.run_hz_sum = 0.0; t.run_hz_sumsq = 0.0;
        t.run_len = 0; t.run_fired = false;
    }
}

}  // namespace brosoundml
