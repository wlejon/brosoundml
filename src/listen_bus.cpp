#include "brosoundml/listen_bus.h"

#include <brotensor/tensor.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brosoundml {

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& what) {
    throw std::runtime_error(where + ": " + what);
}

// Field-by-field framing comparison with a message that names the first
// mismatch. `who` identifies the consumer in the error text.
void check_mel_match(const char* who, const MelConfig& bus, const MelConfig& c) {
    auto bad_i = [&](const char* field, int a, int b) {
        fail("ListenBus::check_compatible",
             std::string(who) + " front-end mismatch on " + field + ": bus=" +
             std::to_string(a) + " consumer=" + std::to_string(b));
    };
    auto bad_f = [&](const char* field, float a, float b) {
        fail("ListenBus::check_compatible",
             std::string(who) + " front-end mismatch on " + field + ": bus=" +
             std::to_string(a) + " consumer=" + std::to_string(b));
    };
    if (bus.sample_rate != c.sample_rate) bad_i("sample_rate", bus.sample_rate, c.sample_rate);
    if (bus.n_fft       != c.n_fft)       bad_i("n_fft",       bus.n_fft,       c.n_fft);
    if (bus.win_length  != c.win_length)  bad_i("win_length",  bus.win_length,  c.win_length);
    if (bus.hop_length  != c.hop_length)  bad_i("hop_length",  bus.hop_length,  c.hop_length);
    if (bus.n_mels      != c.n_mels)      bad_i("n_mels",      bus.n_mels,      c.n_mels);
    if (bus.fmin        != c.fmin)        bad_f("fmin",        bus.fmin,        c.fmin);
    if (bus.fmax        != c.fmax)        bad_f("fmax",        bus.fmax,        c.fmax);
    if (bus.window != c.window)
        fail("ListenBus::check_compatible",
             std::string(who) + " front-end mismatch on window function");
    if (bus.formula != c.formula)
        fail("ListenBus::check_compatible",
             std::string(who) + " front-end mismatch on mel formula");
    if (bus.compression != c.compression)
        fail("ListenBus::check_compatible",
             std::string(who) + " front-end mismatch on compression "
             "(bus is PCEN)");
    if (bus.compression == MelCompression::PCEN) {
        if (bus.pcen_s     != c.pcen_s)     bad_f("pcen_s",     bus.pcen_s,     c.pcen_s);
        if (bus.pcen_alpha != c.pcen_alpha) bad_f("pcen_alpha", bus.pcen_alpha, c.pcen_alpha);
        if (bus.pcen_delta != c.pcen_delta) bad_f("pcen_delta", bus.pcen_delta, c.pcen_delta);
        if (bus.pcen_r     != c.pcen_r)     bad_f("pcen_r",     bus.pcen_r,     c.pcen_r);
        if (bus.pcen_eps   != c.pcen_eps)   bad_f("pcen_eps",   bus.pcen_eps,   c.pcen_eps);
    }
}

MelConfig force_pcen(MelConfig c) {
    c.compression = MelCompression::PCEN;
    return c;
}

}  // namespace

ListenBus::ListenBus(const MelConfig& mel)
    : cfg_(force_pcen(mel)), mel_(cfg_, brotensor::Device::CPU) {
    col_.assign(static_cast<std::size_t>(cfg_.n_mels), 0.0f);
}

void ListenBus::check_compatible(const SensorHub& hub) const {
    check_mel_match("SensorHub", cfg_, hub.config().mel);
}

void ListenBus::check_compatible(const PhonemeSpotter& spotter) const {
    if (!spotter.loaded()) {
        fail("ListenBus::check_compatible",
             "PhonemeSpotter has no model loaded (its front-end framing comes "
             "from the checkpoint)");
    }
    check_mel_match("PhonemeSpotter", cfg_, spotter.mel_config());
}

void ListenBus::check_compatible(const WakeWord& wake) const {
    if (!wake.loaded()) {
        fail("ListenBus::check_compatible",
             "WakeWord has no model loaded (its front-end framing comes "
             "from the checkpoint)");
    }
    check_mel_match("WakeWord", cfg_, wake.mel_config());
}

ListenFeedResult ListenBus::feed(const float* samples, int n,
                                 SensorHub* hub, PhonemeSpotter* spotter,
                                 WakeWord* wake, GestureSpotter* gesture) {
    ListenFeedResult out;
    if (!samples || n <= 0) return out;
    const int win = cfg_.win_length;
    const int hop = cfg_.hop_length;
    const int M   = cfg_.n_mels;

    // Mirror the front-end's stream: ring_ always starts exactly at the next
    // unemitted frame's window start, so frame f of this call covers
    // ring_[f*hop .. f*hop + win).
    ring_.insert(ring_.end(), samples, samples + n);

    brotensor::Tensor frames;   // (n_mels, F) on CPU
    const int F = mel_.consume(samples, n, frames);
    if (F <= 0) return out;
    const std::vector<float> host = frames.to_host_vector();

    // Tier-0, per frame: raw window + matching mel column.
    std::size_t off = 0;
    int done = 0;
    for (int f = 0; f < F; ++f) {
        if (off + static_cast<std::size_t>(win) > ring_.size()) break;
        if (hub) {
            for (int m = 0; m < M; ++m) {
                col_[static_cast<std::size_t>(m)] =
                    host[static_cast<std::size_t>(m) * static_cast<std::size_t>(F) +
                         static_cast<std::size_t>(f)];
            }
            hub->feed_frame(ring_.data() + off, col_.data());
            // Gesture matching is a pure consumer of THIS frame's tier-0
            // snapshot — drive it right after the sensors update, so a
            // completed rhythm/tone surfaces on the same frame.
            if (gesture) {
                auto g = gesture->feed(hub->snapshot());
                for (auto& e : g) out.gestures.push_back(std::move(e));
            }
        }
        off += static_cast<std::size_t>(hop);
        ++done;
    }
    ring_.erase(ring_.begin(), ring_.begin() + static_cast<std::ptrdiff_t>(off));
    out.frames = done;

    // Tier-1/2: ONE forward per model over the whole new block. feed_mel
    // wants a contiguous (n_mels, done) block; when the defensive window
    // check above trimmed the frame count (done < F — cannot happen when
    // ring_ mirrors the front-end, but cheap to honour), repack the kept
    // columns once and hand the same block to every model consumer.
    if ((spotter || wake) && done > 0) {
        const float* block = host.data();
        if (done != F) {
            block_.resize(static_cast<std::size_t>(M) * static_cast<std::size_t>(done));
            for (int m = 0; m < M; ++m) {
                for (int f = 0; f < done; ++f) {
                    block_[static_cast<std::size_t>(m) * static_cast<std::size_t>(done) +
                           static_cast<std::size_t>(f)] =
                        host[static_cast<std::size_t>(m) * static_cast<std::size_t>(F) +
                             static_cast<std::size_t>(f)];
                }
            }
            block = block_.data();
        }
        if (spotter) out.spots = spotter->feed_mel(block, done);
        if (wake)    out.wake_fired = wake->feed_mel(block, done);
    }
    return out;
}

void ListenBus::reset() {
    mel_.reset();
    ring_.clear();
}

}  // namespace brosoundml
