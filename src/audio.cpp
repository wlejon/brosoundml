#include "brosoundml/audio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace brosoundml {

namespace {

// ─── Little-endian scalar I/O ──────────────────────────────────────────────
// WAV is a little-endian RIFF format; write/read fixed-width fields explicitly
// so the code is correct on a big-endian host too.

void put_u32(std::ofstream& f, uint32_t v) {
    uint8_t b[4] = {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)};
    f.write(reinterpret_cast<const char*>(b), 4);
}

void put_u16(std::ofstream& f, uint16_t v) {
    uint8_t b[2] = {uint8_t(v), uint8_t(v >> 8)};
    f.write(reinterpret_cast<const char*>(b), 2);
}

uint32_t get_u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

uint16_t get_u16(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

int16_t to_pcm16(float s) {
    // Clamp to [-1, 1], then scale to the int16 range. 32767 (not 32768) keeps
    // +1.0 and -1.0 symmetric and inside range.
    s = std::min(1.0f, std::max(-1.0f, s));
    return static_cast<int16_t>(std::lround(s * 32767.0f));
}

} // namespace

double AudioBuffer::duration_seconds() const {
    if (samples.empty() || sample_rate <= 0) return 0.0;
    return static_cast<double>(samples.size()) / sample_rate;
}

float AudioBuffer::peak() const {
    float p = 0.0f;
    for (float s : samples) p = std::max(p, std::fabs(s));
    return p;
}

float AudioBuffer::rms() const {
    if (samples.empty()) return 0.0f;
    double acc = 0.0;
    for (float s : samples) acc += static_cast<double>(s) * s;
    return static_cast<float>(std::sqrt(acc / samples.size()));
}

void AudioBuffer::normalize(float target) {
    const float p = peak();
    if (p <= 0.0f || target <= 0.0f) return;
    const float g = target / p;
    for (float& s : samples) s *= g;
}

void AudioBuffer::write_wav(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("brosoundml: write_wav: cannot open '" + path + "'");
    }

    const uint16_t channels   = 1;
    const uint16_t bits       = 16;
    const uint32_t rate       = static_cast<uint32_t>(sample_rate > 0 ? sample_rate : 24000);
    const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t byte_rate  = rate * channels * (bits / 8);
    const uint16_t block_algn = channels * (bits / 8);

    f.write("RIFF", 4);
    put_u32(f, 36 + data_bytes);          // RIFF chunk size
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    put_u32(f, 16);                       // fmt chunk size (PCM)
    put_u16(f, 1);                        // audio format: 1 = PCM
    put_u16(f, channels);
    put_u32(f, rate);
    put_u32(f, byte_rate);
    put_u16(f, block_algn);
    put_u16(f, bits);

    f.write("data", 4);
    put_u32(f, data_bytes);
    for (float s : samples) {
        put_u16(f, static_cast<uint16_t>(to_pcm16(s)));
    }

    if (!f) {
        throw std::runtime_error("brosoundml: write_wav: write failed for '" + path + "'");
    }
}

AudioBuffer read_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("brosoundml: read_wav: cannot open '" + path + "'");
    }
    const std::streamsize size = f.tellg();
    f.seekg(0);
    if (size < 44) {
        throw std::runtime_error("brosoundml: read_wav: '" + path + "' is too small to be a WAV");
    }
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    f.read(reinterpret_cast<char*>(buf.data()), size);

    if (std::memcmp(buf.data(), "RIFF", 4) != 0 ||
        std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("brosoundml: read_wav: '" + path + "' is not a RIFF/WAVE file");
    }

    // Walk the chunk list — chunks after "WAVE" are [4-byte id][4-byte size][body].
    uint16_t channels = 0, bits = 0, fmt = 0;
    uint32_t rate = 0;
    const uint8_t* data = nullptr;
    uint32_t data_bytes = 0;

    size_t off = 12;
    while (off + 8 <= buf.size()) {
        const uint8_t* id = buf.data() + off;
        const uint32_t csize = get_u32(buf.data() + off + 4);
        const size_t body = off + 8;
        if (body + csize > buf.size()) break;

        if (std::memcmp(id, "fmt ", 4) == 0 && csize >= 16) {
            fmt      = get_u16(buf.data() + body + 0);
            channels = get_u16(buf.data() + body + 2);
            rate     = get_u32(buf.data() + body + 4);
            bits     = get_u16(buf.data() + body + 14);
        } else if (std::memcmp(id, "data", 4) == 0) {
            data = buf.data() + body;
            data_bytes = csize;
        }
        off = body + csize + (csize & 1u);   // chunks are word-aligned
    }

    if (fmt != 1 || bits != 16 || channels == 0 || data == nullptr) {
        throw std::runtime_error(
            "brosoundml: read_wav: only 16-bit PCM WAV is supported ('" + path + "')");
    }

    const size_t total = data_bytes / sizeof(int16_t);
    const size_t frames = total / channels;
    AudioBuffer out;
    out.sample_rate = static_cast<int>(rate);
    out.samples.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
        // Downmix to mono by averaging channels.
        int acc = 0;
        for (uint16_t c = 0; c < channels; ++c) {
            const int16_t s =
                static_cast<int16_t>(get_u16(data + (i * channels + c) * 2));
            acc += s;
        }
        out.samples[i] = static_cast<float>(acc) / (channels * 32767.0f);
    }
    return out;
}

AudioBuffer resample(const AudioBuffer& in, int rate) {
    if (rate <= 0 || in.sample_rate <= 0)
        throw std::runtime_error("brosoundml: resample: rates must be positive");
    if (rate == in.sample_rate || in.samples.empty()) {
        AudioBuffer same = in;
        same.sample_rate = rate;
        return same;
    }
    long long a = in.sample_rate, b = rate;
    while (b) { const long long t = a % b; a = b; b = t; }
    const long long orig = in.sample_rate / a;
    const long long next = rate / a;

    constexpr double kPi = 3.14159265358979323846;
    constexpr double kZeroCrossings = 6.0;
    constexpr double kRolloff = 0.99;
    const double ratio  = static_cast<double>(next) / static_cast<double>(orig);
    const double cutoff = 0.5 * std::min(1.0, ratio) * kRolloff;
    const double span   = kZeroCrossings / (2.0 * cutoff);

    const long long n = static_cast<long long>(in.samples.size());
    const long long n_out = (n * next + orig - 1) / orig;
    const float* x = in.samples.data();

    AudioBuffer out;
    out.sample_rate = rate;
    out.samples.resize(static_cast<std::size_t>(n_out));
    for (long long j = 0; j < n_out; ++j) {
        const long long num = j * orig;
        const long long base = num / next;
        const double t = static_cast<double>(base) +
                         static_cast<double>(num - base * next) / static_cast<double>(next);
        const long long lo = std::max<long long>(0, static_cast<long long>(std::ceil(t - span)));
        const long long hi = std::min<long long>(n - 1, static_cast<long long>(std::floor(t + span)));
        double acc = 0.0;
        for (long long k = lo; k <= hi; ++k) {
            const double u = static_cast<double>(k) - t;
            const double arg = 2.0 * cutoff * u;
            const double sinc = (arg == 0.0) ? 1.0 : std::sin(kPi * arg) / (kPi * arg);
            const double w = std::cos(kPi * u / (2.0 * span));
            acc += static_cast<double>(x[k]) * 2.0 * cutoff * sinc * w * w;
        }
        out.samples[static_cast<std::size_t>(j)] = static_cast<float>(acc);
    }
    return out;
}

}
