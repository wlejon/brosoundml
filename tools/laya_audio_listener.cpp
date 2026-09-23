#include "laya_audio_listener.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace laya_audio {

namespace bt = brotensor;

namespace {
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

void LayaListener::load(const ListenerConfig& cfg) {
    cfg_ = cfg;
    const bt::Device dev = bt::default_device();
    asr_.load_encoder(cfg.encoder_dir, dev, /*half=*/dev != bt::Device::CPU);
    model_.load_model(cfg.laya_dir);
    proj_.load(cfg.projector);
    builder_ = std::make_unique<ItemBuilder>(model_);
    reset();
}

int LayaListener::add_question(const std::string& instructions) {
    questions_.push_back(instructions);
    return static_cast<int>(questions_.size()) - 1;
}

void LayaListener::reset() {
    ring_.assign(static_cast<std::size_t>(std::lround(cfg_.window_s * 16000.0f)), 0.0f);
    samples_ = 0;
    next_hop_ = cfg_.hop_ms * 16;
}

std::vector<HopResult> LayaListener::feed(const float* pcm, int n) {
    std::vector<HopResult> out;
    const std::size_t N = ring_.size();
    int i = 0;
    while (i < n) {
        const int take = static_cast<int>(std::min<long long>(n - i, next_hop_ - samples_));
        // Slide the window left by `take` and append.
        if (static_cast<std::size_t>(take) >= N) {
            std::copy(pcm + i + take - N, pcm + i + take, ring_.begin());
        } else {
            std::move(ring_.begin() + take, ring_.end(), ring_.begin());
            std::copy(pcm + i, pcm + i + take, ring_.end() - take);
        }
        i += take;
        samples_ += take;
        if (samples_ == next_hop_) {
            out.push_back(run_hop_());
            next_hop_ += cfg_.hop_ms * 16;
        }
    }
    return out;
}

HopResult LayaListener::run_hop_() {
    HopResult r;
    r.t_end = static_cast<double>(samples_) / 16000.0;
    const double t0 = now_ms();
    const bt::Tensor lat = asr_.encode(brosoundml::AudioBuffer(ring_, 16000));
    bt::Tensor lat32;
    if (lat.dtype == bt::Dtype::FP32) {
        lat32 = lat;
    } else {
        bt::cast(lat, lat32, bt::Dtype::FP32);
    }
    bt::Tensor y, soft;
    proj_.forward(lat32, y);
    bt::cast(y, soft, bt::compute_dtype());
    bt::sync(bt::default_device());
    const double t1 = now_ms();
    r.encode_ms = t1 - t0;
    if (!questions_.empty()) {
        std::vector<brolm::laya::LayaItem> items;
        for (const std::string& q : questions_) items.push_back(builder_->item(q, lat.rows, 0));
        const auto res = model_.forward_items(items, &soft);
        for (const auto& li : res) {
            const float z = (li.logits[1] - li.logits[0]) / cfg_.temperature;
            r.p.push_back(1.0f / (1.0f + std::exp(-z)));
        }
    }
    const double t2 = now_ms();
    r.laya_ms = t2 - t1;
    r.total_ms = t2 - t0;
    return r;
}

}  // namespace laya_audio
