#pragma once

#include "brosoundml/audio.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml {

// ─── YuE2 VAE (48 kHz stereo <-> 64-dim latents at 25 Hz) ───────────────────
//
// The acoustic autoencoder behind YuE2 (m-a-p/YuE2-Vae): a Stable Audio
// "Oobleck" conv stack with SnakeBeta activations. A latent frame is
// downsampling_ratio = 1920 samples (2*2*4*4*5*6), so 25 frames per second.
//
//   decode   latents (64, T) -> conv k7 (64 -> 2048) -> six decoder blocks
//            (SnakeBeta, ConvTranspose1d x{6,5,4,4,2,2} with kernel 2*stride,
//            padding ceil(stride/2), no output_padding, then three residual
//            units SnakeBeta -> conv k7 dilation {1,3,9} -> SnakeBeta -> conv
//            k1), 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 64, SnakeBeta,
//            conv k7 (64 -> 2, no bias). No tanh, no clamp. The natural
//            output is 1920*T - 64 samples (the stride-5 upsampler loses one).
//   encode   the mirror: conv k7 (2 -> 64), six encoder blocks (three residual
//            units, SnakeBeta, strided conv kernel 2*stride), SnakeBeta, conv
//            k3 (2048 -> 128) = [mean | scale]; the latent is the posterior
//            mean (sample = false in the reference).
//
// SnakeBeta here is log-scale: y = x + sin^2(x*exp(alpha)) / exp(beta).
// Weight norm is not folded on disk (weight_g / weight_v); load() folds it.
// Weights are FP32 and stay FP32 on every backend.
//
// Tiling. The decoder's receptive field is finite, so a span of latents can
// be decoded from a window that carries `required_halo()` extra frames each
// side and cropped to its core, exactly (no crossfade): this is the
// reference's decode_tiled, and it is what lets Yue2VaeStream play a song
// while later latents are still being synthesized.

struct Yue2VaeConfig {
    int sample_rate        = 48000;
    int channels           = 2;
    int latent_dim         = 64;
    int downsampling_ratio = 1920;   // samples per latent frame
    int decode_core_frames = 1024;   // default tile core for decode_tiled
    int decode_halo_frames = 16;     // default tile halo (>= required_halo())
    std::vector<int> strides;        // encoder order ({2,2,4,4,5,6})
    std::vector<int> c_mults;        // ({1,2,4,8,16,32})
    int channels_base      = 64;
    bool has_encoder       = true;
};

// Planar FP32 stereo PCM: samples[c * frames + i]. Nominally [-1, 1] but the
// decoder does not clamp.
struct StereoAudio {
    std::vector<float> samples;
    int frames      = 0;
    int channels    = 2;
    int sample_rate = 48000;

    const float* channel(int c) const { return samples.data() + static_cast<std::size_t>(c) * frames; }
    float*       channel(int c)       { return samples.data() + static_cast<std::size_t>(c) * frames; }
    double duration_seconds() const { return sample_rate > 0 ? double(frames) / sample_rate : 0.0; }

    // 16-bit PCM WAV, interleaved, clamped to [-1, 1]. Throws on I/O failure.
    void write_wav(const std::string& path) const;
    // Mono mixdown (channel average).
    AudioBuffer mono() const;
};

class Yue2Vae {
public:
    Yue2Vae();
    ~Yue2Vae();
    Yue2Vae(Yue2Vae&&) noexcept;
    Yue2Vae& operator=(Yue2Vae&&) noexcept;
    Yue2Vae(const Yue2Vae&) = delete;
    Yue2Vae& operator=(const Yue2Vae&) = delete;

    // `dir` holds config.json + model.safetensors (the YuE2-Vae repo).
    // `decoder_only` skips the encoder's weights (encode() then throws).
    // Throws std::runtime_error on a missing / malformed model.
    void load(const std::string& dir,
              brotensor::Device device = brotensor::Device::CUDA,
              bool decoder_only = false);

    // Latents, frame-major [frame][64] (Yue2's on_latents layout), -> the
    // full-decoder waveform, 1920*frames - 64 samples per channel. Decodes in
    // one pass; long songs want decode_tiled. `wait`, when set, is called
    // every 2 ms while the device runs the decode (a host loop feeding audio
    // keeps going); it changes nothing in the result.
    StereoAudio decode(const float* latents, int frames, const std::function<void()>& wait = {}) const;

    // The same waveform from tiles of `core_frames` with `halo_frames` of
    // context each side (0 = the config defaults; halo must be >=
    // required_halo()). Peak device memory is bounded by the tile, not the
    // song. `on_progress(done, total)` after each tile; `cancel` between
    // tiles (a cancelled decode returns the samples finished so far).
    StereoAudio decode_tiled(const float* latents, int frames, int core_frames = 0,
                             int halo_frames = 0,
                             const std::function<void(int, int)>& on_progress = {},
                             const CancelCheck& cancel = {}) const;

    // Samples [start * 1920, end * 1920) of decode(latents, frames) — or up
    // to the natural end when `end == frames` and `song_end` — decoded from
    // one window with `halo_frames` (0 = required_halo()) of the available
    // context each side, exactly. A just-in-time decoder: frames before `end`
    // + halo must already be final; anything can be re-decoded after an edit.
    StereoAudio decode_span(const float* latents, int frames, int start, int end, bool song_end = false,
                            int halo_frames = 0, const std::function<void()>& wait = {}) const;
    // 48 kHz stereo -> posterior-mean latents, frame-major [frame][64];
    // *frames_out receives the frame count. Other rates are resampled; a mono
    // input is duplicated to both channels. Needs >= 1920 samples. Encodes in
    // tiles (exact, like decode_tiled) so a whole song fits on the device.
    std::vector<float> encode(const StereoAudio& audio, int* frames_out) const;
    std::vector<float> encode(const AudioBuffer& mono, int* frames_out) const;

    // Smallest halo (in frames) that makes a tile of `core_frames` exact,
    // from the decoder's dependency intervals (12 for the released decoder).
    int required_halo(int core_frames = 0) const;
    // Samples per channel the full decoder produces from `frames` latents.
    int output_length(int frames) const;

    const Yue2VaeConfig& config() const;
    bool loaded() const;
    bool has_encoder() const;
    brotensor::Device device() const;

private:
    friend class Yue2VaeStream;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Streaming decode: latents in as they become final, audio out as soon as a
// tile's right context exists. Push contiguous frames from 0 (Yue2's
// on_latents contract), then finish() once (Yue2's on_done) to flush the
// held-back tail. Output chunks are contiguous from sample 0, planar stereo,
// and concatenate to exactly decode()'s waveform.
//
// Latency: a frame is decodable once `halo` frames past it exist. Tiles grow:
// the first is decoded as soon as `first_core_frames` frames are decodable,
// and each later one waits until it can take at least twice the previous
// tile (capped at `core_frames`), then takes every decodable frame up to the
// cap. Pushes of a whole NAR window (50 frames) so give one 38-frame tile at
// once and full cores after; frame-by-frame pushes give 8, 16, 32, 50, 50...
// so playback is never waiting on a big tile. Every tile carries the halo
// both sides, so small tiles cost relatively more (core 50 + 2x12: ~1.5x).
// first_core_frames >= core_frames gives fixed tiles of core_frames.
class Yue2VaeStream {
public:
    using ChunkFn = std::function<void(const StereoAudio& chunk, std::int64_t sample0)>;

    // `vae` must outlive the stream. core_frames 0 = 50 (2 s); first_core_frames 0 = 8.
    Yue2VaeStream(const Yue2Vae& vae, ChunkFn on_chunk, int core_frames = 0, int halo_frames = 0,
                  int first_core_frames = 0);
    ~Yue2VaeStream();
    Yue2VaeStream(const Yue2VaeStream&) = delete;
    Yue2VaeStream& operator=(const Yue2VaeStream&) = delete;

    // `frame0` must equal frames_pushed(); `data` is frame-major [frames][64].
    // Decodes (and emits) every tile that became ready. Throws on a gap.
    void push(int frame0, int frames, const float* data);
    // Decode the remaining tiles with the true end of the song as their right
    // edge. Idempotent; push() after finish() throws.
    void finish();
    // Drop everything (a cancelled song); nothing more is emitted.
    void reset();

    int frames_pushed() const;
    int frames_decoded() const;          // latent frames whose audio was emitted
    std::int64_t samples_emitted() const;
    bool finished() const;

private:
    void decode_ready(bool final_pass);

    const Yue2Vae* vae_;
    ChunkFn on_chunk_;
    int core_, halo_, first_;
    std::vector<float> latents_;         // every frame pushed, frame-major
    int pushed_ = 0, decoded_ = 0, last_tile_ = 0;
    std::int64_t emitted_ = 0;
    bool finished_ = false;
};

}  // namespace brosoundml
