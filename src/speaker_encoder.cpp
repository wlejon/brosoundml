#include "brosoundml/speaker_encoder.h"

#include "qwen_tts_speaker_encoder.h"
#include "brosoundml/detail/json.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace brosoundml {
namespace {

namespace fs = std::filesystem;
namespace j  = detail::json;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: SpeakerEncoder::load: " + msg);
}

std::string slurp(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open '" + path.string() + "'");
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Parse the `speaker_encoder_config` block. config.json carries only enc_dim +
// sample_rate; the channel / kernel / dilation lists and mel params are the
// upstream defaults — kept identical to QwenTts's parse so the two loaders
// resolve the same config.
QwenTtsSpeakerEncoderConfig parse_config(const fs::path& cfg_path) {
    const j::Value root = j::parse(slurp(cfg_path));
    if (!root.is_object()) fail("config.json is not a JSON object");
    const j::Value* se = root.find("speaker_encoder_config");
    if (!se || !se->is_object())
        fail("config.json has no 'speaker_encoder_config' block");

    QwenTtsSpeakerEncoderConfig s;
    s.present     = true;
    s.mel_dim     = se->get_int("mel_dim", 128);
    s.enc_dim     = se->get_int("enc_dim", 1024);
    s.sample_rate = se->get_int("sample_rate", 24000);
    s.enc_channels     = se->get_int_array("enc_channels", {512, 512, 512, 512, 1536});
    s.enc_kernel_sizes = se->get_int_array("enc_kernel_sizes", {5, 3, 3, 3, 1});
    s.enc_dilations    = se->get_int_array("enc_dilations", {1, 2, 3, 4, 1});
    s.res2net_scale       = se->get_int("enc_res2net_scale", 8);
    s.se_channels         = se->get_int("enc_se_channels", 128);
    s.attention_channels  = se->get_int("enc_attention_channels", 128);
    return s;
}

}  // namespace

struct SpeakerEncoder::Impl {
    QwenTtsSpeakerEncoder enc;
    bool                  loaded = false;
};

SpeakerEncoder::SpeakerEncoder() : impl_(std::make_unique<Impl>()) {}
SpeakerEncoder::~SpeakerEncoder() = default;
SpeakerEncoder::SpeakerEncoder(SpeakerEncoder&&) noexcept = default;
SpeakerEncoder& SpeakerEncoder::operator=(SpeakerEncoder&&) noexcept = default;

void SpeakerEncoder::load(const std::string& dir) {
    brotensor::init();

    const fs::path d           = dir;
    const fs::path cfg_path     = d / "config.json";
    const fs::path weight_path  = d / "model.safetensors";
    if (!fs::exists(cfg_path))
        fail("no config.json under '" + dir + "'");
    if (!fs::exists(weight_path))
        fail("no model.safetensors under '" + dir + "'");

    const QwenTtsSpeakerEncoderConfig cfg = parse_config(cfg_path);

    auto w = brotensor::safetensors::File::open(weight_path.string());
    impl_->enc.load(w, cfg);
    impl_->loaded = true;
}

std::vector<float> SpeakerEncoder::embed(const AudioBuffer& ref) const {
    if (!impl_->loaded)
        throw std::runtime_error(
            "brosoundml: SpeakerEncoder::embed: no artifact loaded; call load() first");
    if (ref.samples.empty())
        throw std::runtime_error(
            "brosoundml: SpeakerEncoder::embed: reference audio is empty");

    // Reference clip -> the encoder's rate (24 kHz) mono -> ECAPA-TDNN x-vector.
    // Mirrors QwenTts::embed_speaker so the embedding is bit-identical.
    const int sr  = impl_->enc.cfg.sample_rate;
    const float* wav = ref.samples.data();
    int n = static_cast<int>(ref.samples.size());
    std::vector<float> resampled;
    if (ref.sample_rate != sr && ref.sample_rate > 0) {
        const int n_out = static_cast<int>(
            std::llround(static_cast<double>(n) * sr / ref.sample_rate));
        brotensor::DeviceScope cpu(brotensor::Device::CPU);
        brotensor::Tensor x =
            brotensor::Tensor::from_host_on(brotensor::Device::CPU, wav, 1, n);
        brotensor::Tensor y;
        brotensor::resample1d_forward(x, /*N=*/1, /*C=*/1, n, n_out, /*mode=*/1, y);
        resampled.assign(y.host_f32(), y.host_f32() + n_out);
        wav = resampled.data();
        n = n_out;
    }
    return impl_->enc.embed(wav, n);
}

bool SpeakerEncoder::loaded() const { return impl_->loaded; }
int  SpeakerEncoder::enc_dim() const { return impl_->enc.cfg.enc_dim; }
int  SpeakerEncoder::sample_rate() const { return impl_->enc.cfg.sample_rate; }

}  // namespace brosoundml
