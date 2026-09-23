#include "laya_audio_baselines.h"

#include "laya_audio_data.h"

#include "brosoundml/audio.h"
#include "brosoundml/g2p/lexicon.h"
#include "brosoundml/g2p/morphology.h"
#include "brosoundml/g2p/phoneme_adapter.h"
#include "brosoundml/g2p/phonemizer.h"
#include "brosoundml/g2p/pos_tagger.h"
#include "brosoundml/g2p/special_cases.h"
#include "brosoundml/kokoro.h"
#include "brosoundml/mel.h"
#include "brosoundml/parakeet.h"
#include "brosoundml/phoneme_model.h"
#include "brosoundml/phoneme_spotter.h"

#include <brolm/tokenizer_t5.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <unordered_map>

namespace laya_audio {

namespace bt = brotensor;
namespace bsm = brosoundml;
namespace g = brosoundml::g2p;

namespace {
std::vector<float> dithered_silence(std::size_t n, uint32_t seed) {
    std::vector<float> v(n);
    uint32_t s = seed * 2654435761u + 12345u;
    for (float& x : v) {
        s = s * 1664525u + 1013904223u;
        x = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 2e-4f;
    }
    return v;
}
}  // namespace

struct AsrBaseline::Impl {
    bsm::Parakeet model;
    std::optional<brolm::t5::Tokenizer> tok;
};

AsrBaseline::AsrBaseline() : impl_(std::make_unique<Impl>()) {}
AsrBaseline::~AsrBaseline() = default;

void AsrBaseline::load(const std::string& dir) {
    impl_->model.load(dir, bt::default_device());
    impl_->tok.emplace(brolm::t5::Tokenizer::load((std::filesystem::path(dir) / "tokenizer.json").string()));
}

std::string AsrBaseline::transcribe(const std::vector<float>& pcm) {
    std::string out;
    for (const TimedWord& w : transcribe_timed(pcm, false)) {
        if (!out.empty()) out += ' ';
        out += w.word;
    }
    return out;
}

std::vector<TimedWord> AsrBaseline::transcribe_timed(const std::vector<float>& pcm, bool pad) {
    // Parakeet (this port: no NeMo preprocessor dither) returns an empty
    // transcript for many short clips that end in exact digital silence, so
    // the padding carries the same +-1e-4 dither window_audio adds.
    const int lead = pad ? 8000 : 0;
    std::vector<float> in = dithered_silence(pcm.size() + 2 * static_cast<std::size_t>(lead), 99u);
    std::copy(pcm.begin(), pcm.end(), in.begin() + lead);
    const auto tr = impl_->model.transcribe(bsm::AudioBuffer(std::move(in), 16000));
    const double frame_s = impl_->model.config().frame_seconds();
    const float shift = static_cast<float>(lead) / 16000.0f;
    const brolm::t5::Tokenizer& tok = *impl_->tok;

    std::vector<TimedWord> words;
    std::vector<int32_t> prefix;
    std::string prev_text, cur;
    float t0 = 0, t1 = 0;
    auto flush = [&] {
        for (const std::string& w : normalize_words(cur)) {
            TimedWord tw;
            tw.word = w;
            tw.t0 = t0 - shift;
            tw.t1 = t1 - shift;
            words.push_back(std::move(tw));
        }
        cur.clear();
    };
    for (std::size_t i = 0; i < tr.token_ids.size(); ++i) {
        prefix.push_back(tr.token_ids[i]);
        const std::string text = tok.decode(prefix);
        const std::string piece = text.size() > prev_text.size() ? text.substr(prev_text.size()) : std::string();
        prev_text = text;
        const float s = static_cast<float>(tr.token_frames[i] * frame_s);
        const int dur = i < tr.token_durations.size() ? tr.token_durations[i] : 1;
        const float e = static_cast<float>((tr.token_frames[i] + std::max(1, dur)) * frame_s);
        if (i == 0 || (!piece.empty() && piece[0] == ' ')) {
            flush();
            t0 = s;
        }
        cur += piece;
        t1 = e;
    }
    flush();
    return words;
}

std::string AsrBaseline::transcribe_stream_window(const std::vector<float>& utt, float t_end, float window_s) {
    const std::size_t end = std::min(utt.size(), static_cast<std::size_t>(std::max(0.0f, t_end) * 16000.0f));
    // No trailing silence: appended silence (even dithered) makes Parakeet
    // return nothing for many short prefixes (its per-feature normalization
    // runs over the whole clip), so the prefix goes in as it is.
    const std::vector<float> prefix(utt.begin(), utt.begin() + static_cast<std::ptrdiff_t>(end));
    if (prefix.size() < 1600) return {};
    std::string out;
    for (const TimedWord& w : transcribe_timed(prefix, false)) {
        const float mid = 0.5f * (w.t0 + w.t1);
        if (mid < t_end - window_s || mid > t_end) continue;
        if (!out.empty()) out += ' ';
        out += w.word;
    }
    return out;
}

struct PhonemeBaseline::Impl {
    std::optional<bsm::PhonemeNet> net;
    std::optional<bsm::MelFrontend> mel;
    bsm::Kokoro kokoro;
    std::optional<g::Lexicon> lex;
    std::optional<g::Morphology> morph;
    std::optional<g::SpecialCases> sc;
    std::optional<g::PosTagger> tagger;
    std::optional<g::PhonemeAdapter> adapter;
    std::optional<g::Phonemizer> phon;
    bsm::PhonemeSpotter spotter;
    bsm::SpotterConfig pol;
    std::unordered_map<std::string, std::vector<int>> ids;
    std::vector<float> post;
    int T = 0;
};

PhonemeBaseline::PhonemeBaseline() : impl_(std::make_unique<Impl>()) {}
PhonemeBaseline::~PhonemeBaseline() = default;

void PhonemeBaseline::load(const std::string& weights, const std::string& data_dir, const std::string& kokoro_dir) {
    Impl& m = *impl_;
    const bt::Device dev = bt::default_device();
    m.net.emplace(bsm::PhonemeNet::load(weights, dev));
    const bsm::PhonemeClassMap cm = m.net->class_map();
    cm.rebuild_inverse();
    const bsm::PhonemeNetConfig& mc = m.net->config();
    bsm::MelConfig mcfg;
    mcfg.sample_rate = mc.sample_rate;
    mcfg.n_fft = mc.n_fft;
    mcfg.win_length = mc.win_length;
    mcfg.hop_length = mc.hop_length;
    mcfg.n_mels = mc.n_mels;
    mcfg.compression = bsm::MelCompression::PCEN;
    m.mel.emplace(mcfg, dev);
    m.kokoro.load(kokoro_dir, dev);
    m.lex.emplace(g::Lexicon::load(data_dir + "/g2p/lexicon_en_us.bin"));
    m.morph.emplace(*m.lex);
    m.sc.emplace(*m.lex);
    m.tagger.emplace(g::PosTagger::load(data_dir + "/pos_tagger/model.bin"));
    m.adapter.emplace(m.kokoro.config().vocab);
    m.phon.emplace(*m.tagger, *m.lex, *m.morph, *m.sc, *m.adapter);
    m.spotter.set_class_map(cm);
    // Scoring policy: every completion fires (threshold ~0) so the best
    // completion confidence is a continuous score; matches may begin
    // mid-speech (continuous read speech has no pause before most words).
    m.pol = m.spotter.config();
    m.pol.threshold = 1e-6f;
    m.pol.smoothing_hits = 1;
    m.pol.smoothing_window = 1;
    m.pol.refractory_ms = 0;
    m.pol.min_phonemes = 1;
    m.pol.entry_silence_frames = 0;
}

void PhonemeBaseline::set_window(const std::vector<float>& pcm) {
    Impl& m = *impl_;
    const int n_mels = m.net->config().n_mels;
    bt::Tensor melt;
    m.mel->reset();
    m.mel->compute_offline(pcm.data(), static_cast<int>(pcm.size()), melt);
    const std::vector<float> mh = melt.to_host_vector();
    const int T = static_cast<int>(mh.size() / static_cast<std::size_t>(n_mels));
    m.T = 0;
    if (T <= 0) return;
    bt::Tensor feats = bt::Tensor::from_host_on(bt::default_device(), mh.data(), n_mels, T);
    bt::Tensor logits;
    m.net->forward(feats, logits);
    const int K = logits.cols;
    std::vector<float> lg = logits.to_host_vector();
    m.post.assign(lg.size(), 0.0f);
    for (int t = 0; t < logits.rows; ++t) {
        float* row = &lg[static_cast<std::size_t>(t) * K];
        const float mx = *std::max_element(row, row + K);
        double sum = 0;
        for (int j = 0; j < K; ++j) {
            row[j] = std::exp(row[j] - mx);
            sum += row[j];
        }
        for (int j = 0; j < K; ++j) m.post[static_cast<std::size_t>(t) * K + j] = static_cast<float>(row[j] / sum);
    }
    m.T = logits.rows;
}

float PhonemeBaseline::score(const std::string& keyword) {
    Impl& m = *impl_;
    if (m.T <= 0) return 0.0f;
    auto it = m.ids.find(keyword);
    if (it == m.ids.end()) it = m.ids.emplace(keyword, m.phon->phonemize(keyword)).first;
    m.spotter.clear();
    if (m.spotter.enroll(keyword, it->second, &m.pol) <= 0) return 0.0f;
    m.spotter.reset();
    float best = 0.0f;
    for (const bsm::SpotEvent& e : m.spotter.feed_posteriors(m.post.data(), m.T)) best = std::max(best, e.confidence);
    return best;
}

}  // namespace laya_audio
