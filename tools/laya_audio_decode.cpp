// laya-audio: compressed-audio loading for the dataset tools.
//
//   .wav              brosoundml::read_wav
//   .flac             dr_flac (header-only, vendored by vcpkg / system)
//   .opus / .ogg      Ogg Opus through libogg + libopus, decoded straight at
//                     16 kHz (the decoder resamples internally); pre-skip and
//                     the end trim from the last granule position are honoured
//   "path@t0:t1"      the [t0, t1) seconds of that file; the last decoded file
//                     is kept so consecutive ranges of one long recording
//                     (AMI meetings, noise tracks) decode it once
//
// Everything comes back as 16 kHz mono FP32.

#include "laya_audio_data.h"

#include "brosoundml/audio.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#if LAYA_AUDIO_HAVE_FLAC
#define DR_FLAC_IMPLEMENTATION
#include <dr_flac.h>
#endif
#if LAYA_AUDIO_HAVE_OPUS
#include <ogg/ogg.h>
#include <opus/opus.h>
#endif

namespace laya_audio {

namespace {

bool ends_with(const std::string& s, const char* suf) {
    const std::size_t n = std::strlen(suf);
    if (s.size() < n) return false;
    for (std::size_t i = 0; i < n; ++i)
        if (std::tolower(static_cast<unsigned char>(s[s.size() - n + i])) != suf[i]) return false;
    return true;
}

// Paths are UTF-8 (MSWC word folders such as "aktivitäten"); on Windows the
// narrow fopen would read them in the ANSI code page, so open wide there.
std::FILE* fopen_utf8(const std::string& path) {
#ifdef _WIN32
    const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (n <= 0) return nullptr;
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
    return _wfopen(w.c_str(), L"rb");
#else
    return std::fopen(path.c_str(), "rb");
#endif
}

std::vector<unsigned char> read_file(const std::string& path) {
    std::FILE* f = fopen_utf8(path);
    if (!f) throw std::runtime_error("laya_audio: cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf(static_cast<std::size_t>(std::max(0L, n)));
    const std::size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    buf.resize(got);
    return buf;
}

#if LAYA_AUDIO_HAVE_OPUS
std::vector<float> decode_ogg_opus_16k(const std::string& path) {
    const std::vector<unsigned char> data = read_file(path);
    ogg_sync_state oy;
    ogg_sync_init(&oy);
    char* dst = ogg_sync_buffer(&oy, static_cast<long>(data.size()));
    std::memcpy(dst, data.data(), data.size());
    ogg_sync_wrote(&oy, static_cast<long>(data.size()));

    ogg_stream_state os;
    bool stream_init = false;
    OpusDecoder* dec = nullptr;
    int channels = 1, pre_skip = 0, packet_no = 0;
    ogg_int64_t last_granule = -1;
    std::vector<float> out, frame(static_cast<std::size_t>(5760) * 2);
    auto cleanup = [&] {
        if (dec) opus_decoder_destroy(dec);
        if (stream_init) ogg_stream_clear(&os);
        ogg_sync_clear(&oy);
    };
    ogg_page og;
    while (ogg_sync_pageout(&oy, &og) == 1) {
        if (!stream_init) {
            ogg_stream_init(&os, ogg_page_serialno(&og));
            stream_init = true;
        }
        if (ogg_page_serialno(&og) != os.serialno) continue;  // first logical stream only
        ogg_stream_pagein(&os, &og);
        if (ogg_page_granulepos(&og) >= 0) last_granule = ogg_page_granulepos(&og);
        ogg_packet op;
        while (ogg_stream_packetout(&os, &op) == 1) {
            if (packet_no == 0) {
                if (op.bytes < 19 || std::memcmp(op.packet, "OpusHead", 8) != 0) {
                    cleanup();
                    throw std::runtime_error("laya_audio: not an Ogg Opus file: " + path);
                }
                channels = op.packet[9];
                pre_skip = op.packet[10] | (op.packet[11] << 8);
                int err = 0;
                dec = opus_decoder_create(16000, channels, &err);
                if (err != OPUS_OK || channels < 1 || channels > 2) {
                    cleanup();
                    throw std::runtime_error("laya_audio: opus decoder init failed for " + path);
                }
            } else if (packet_no >= 2) {  // packet 1 is OpusTags
                const int n = opus_decode_float(dec, op.packet, static_cast<opus_int32>(op.bytes), frame.data(),
                                                5760 / 3, 0);
                if (n > 0) {
                    for (int i = 0; i < n; ++i) {
                        float s = 0;
                        for (int c = 0; c < channels; ++c) s += frame[static_cast<std::size_t>(i) * channels + c];
                        out.push_back(s / static_cast<float>(channels));
                    }
                }
            }
            ++packet_no;
        }
    }
    cleanup();
    // Granule positions count 48 kHz samples including the pre-skip.
    const std::size_t skip = static_cast<std::size_t>(pre_skip / 3);
    std::size_t end = out.size();
    if (last_granule > 0) end = std::min(end, static_cast<std::size_t>(std::max<ogg_int64_t>(0, last_granule / 3)));
    if (skip >= end) return {};
    return std::vector<float>(out.begin() + static_cast<std::ptrdiff_t>(skip),
                              out.begin() + static_cast<std::ptrdiff_t>(end));
}
#endif

std::vector<float> decode_file_16k(const std::string& path) {
    if (ends_with(path, ".opus") || ends_with(path, ".ogg")) {
#if LAYA_AUDIO_HAVE_OPUS
        return decode_ogg_opus_16k(path);
#else
        throw std::runtime_error("laya_audio: built without libopus, cannot read " + path);
#endif
    }
    if (ends_with(path, ".flac")) {
#if LAYA_AUDIO_HAVE_FLAC
        unsigned int ch = 0, rate = 0;
        drflac_uint64 frames = 0;
        const std::vector<unsigned char> data = read_file(path);
        float* p = drflac_open_memory_and_read_pcm_frames_f32(data.data(), data.size(), &ch, &rate, &frames, nullptr);
        if (!p) throw std::runtime_error("laya_audio: cannot decode " + path);
        std::vector<float> mono(static_cast<std::size_t>(frames));
        for (std::size_t i = 0; i < mono.size(); ++i) {
            float s = 0;
            for (unsigned c = 0; c < ch; ++c) s += p[i * ch + c];
            mono[i] = s / static_cast<float>(ch);
        }
        drflac_free(p, nullptr);
        return resample_sinc(mono, static_cast<int>(rate), 16000);
#else
        throw std::runtime_error("laya_audio: built without dr_flac, cannot read " + path);
#endif
    }
    brosoundml::AudioBuffer a = brosoundml::read_wav(path);
    if (a.sample_rate == 16000) return std::move(a.samples);
    return resample_sinc(a.samples, a.sample_rate, 16000);
}

}  // namespace

std::vector<float> load_audio_16k(const std::string& spec) {
    const std::size_t at = spec.rfind('@');
    if (at == std::string::npos || spec.find(':', at) == std::string::npos) return decode_file_16k(spec);
    const std::string path = spec.substr(0, at);
    const std::size_t colon = spec.find(':', at);
    const double t0 = std::stod(spec.substr(at + 1, colon - at - 1));
    const double t1 = std::stod(spec.substr(colon + 1));
    // One-entry memo of the last whole file.
    static std::mutex mu;
    static std::string memo_path;
    static std::shared_ptr<const std::vector<float>> memo;
    std::shared_ptr<const std::vector<float>> whole;
    {
        std::lock_guard<std::mutex> lock(mu);
        if (memo_path == path) whole = memo;
    }
    if (!whole) {
        whole = std::make_shared<const std::vector<float>>(decode_file_16k(path));
        std::lock_guard<std::mutex> lock(mu);
        memo_path = path;
        memo = whole;
    }
    const std::size_t a = std::min(whole->size(), static_cast<std::size_t>(std::max(0.0, t0) * 16000.0 + 0.5));
    const std::size_t b = std::min(whole->size(), static_cast<std::size_t>(std::max(0.0, t1) * 16000.0 + 0.5));
    if (b <= a) return {};
    return std::vector<float>(whole->begin() + static_cast<std::ptrdiff_t>(a),
                              whole->begin() + static_cast<std::ptrdiff_t>(b));
}

std::vector<float> resample_sinc(const std::vector<float>& in, int in_rate, int out_rate) {
    if (in_rate == out_rate) return in;
    // Polyphase: out sample n sits at input time n*M/L (L/M = out/in reduced);
    // its phase (n*M mod L) picks one of L precomputed Hann-windowed sinc
    // kernels, normalised to unit DC gain.
    const int g = std::gcd(in_rate, out_rate);
    const long L = out_rate / g, M = in_rate / g;
    const double ratio = static_cast<double>(out_rate) / in_rate;
    const double cutoff = std::min(1.0, ratio) * 0.95;  // of the input Nyquist
    constexpr int kZeros = 16;
    const double half_width = kZeros / cutoff;  // input samples each side
    const int K = static_cast<int>(std::ceil(half_width)) + 1;
    const double pi = 3.14159265358979323846;
    if (L > 4096) throw std::runtime_error("laya_audio: unsupported resampling ratio");
    std::vector<float> taps(static_cast<std::size_t>(L) * (2 * K + 1), 0.0f);
    for (long ph = 0; ph < L; ++ph) {
        const double frac = static_cast<double>(ph) / L;  // t - floor(t)
        double wsum = 0;
        float* k = &taps[static_cast<std::size_t>(ph) * (2 * K + 1)];
        for (int o = -K; o <= K; ++o) {
            const double d = frac - o;  // t - j with j = floor(t) + o
            if (std::abs(d) > half_width) continue;
            const double x = d * cutoff;
            const double sinc = std::abs(x) < 1e-9 ? 1.0 : std::sin(pi * x) / (pi * x);
            const double w = 0.5 + 0.5 * std::cos(pi * d / half_width);
            k[o + K] = static_cast<float>(sinc * w);
            wsum += sinc * w;
        }
        for (int o = 0; o <= 2 * K; ++o) k[o] = static_cast<float>(k[o] / wsum);
    }
    const std::size_t n_out = static_cast<std::size_t>(std::floor(in.size() * ratio));
    std::vector<float> out(n_out);
    const long n_in = static_cast<long>(in.size());
    for (std::size_t n = 0; n < n_out; ++n) {
        const long long num = static_cast<long long>(n) * M;
        const long base = static_cast<long>(num / L);
        const float* k = &taps[static_cast<std::size_t>(num % L) * (2 * K + 1)];
        float acc = 0;
        const long j0 = base - K;
        if (j0 >= 0 && base + K < n_in) {
            const float* x = &in[static_cast<std::size_t>(j0)];
            for (int o = 0; o <= 2 * K; ++o) acc += k[o] * x[o];
        } else {
            for (int o = 0; o <= 2 * K; ++o) {
                const long j = j0 + o;
                if (j >= 0 && j < n_in) acc += k[o] * in[static_cast<std::size_t>(j)];
            }
        }
        out[n] = acc;
    }
    return out;
}

}  // namespace laya_audio
