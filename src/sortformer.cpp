#include "brosoundml/sortformer.h"

#include "brosoundml/detail/json.h"
#include "fastconformer_modules.h"
#include "sortformer_modules.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace fs = std::filesystem;
namespace j  = detail::json;
namespace bt = brotensor;
namespace sf = brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

std::string slurp(const std::string& path, const std::string& where) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail(where, "cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

const j::Value& obj_at(const j::Value& root, const std::string& key,
                       const std::string& where) {
    const j::Value* v = root.find(key);
    if (!v || !v->is_object())
        fail(where, "config.json missing object '" + key + "'");
    return *v;
}

int int_at(const j::Value& obj, const std::string& key, const std::string& where) {
    const j::Value* v = obj.find(key);
    if (!v) fail(where, "config.json missing required key '" + key + "'");
    return static_cast<int>(v->as_number());
}

SortformerConfig parse_config(const std::string& path) {
    const std::string where = "Sortformer::load";
    const j::Value    root = j::parse(slurp(path, where));
    if (!root.is_object()) fail(where, "config.json is not a JSON object");

    SortformerConfig c;
    c.sample_rate = root.get_int("sample_rate", 16000);
    c.num_spks    = root.get_int("num_spks", 4);
    c.fc_d_model  = root.get_int("fc_d_model", 512);
    c.tf_d_model  = root.get_int("tf_d_model", 192);

    const j::Value& e = obj_at(root, "encoder_config", where);
    FastConformerConfig& ec = c.encoder;
    ec.num_mel_bins                 = int_at(e, "num_mel_bins", where);
    ec.hidden_size                  = int_at(e, "hidden_size", where);
    ec.num_hidden_layers            = int_at(e, "num_hidden_layers", where);
    ec.num_attention_heads          = int_at(e, "num_attention_heads", where);
    ec.intermediate_size            = int_at(e, "intermediate_size", where);
    ec.conv_kernel_size             = e.get_int("conv_kernel_size", 9);
    ec.subsampling_factor           = e.get_int("subsampling_factor", 8);
    ec.subsampling_conv_channels    = e.get_int("subsampling_conv_channels", 256);
    ec.subsampling_conv_kernel_size = e.get_int("subsampling_conv_kernel_size", 3);
    ec.subsampling_conv_stride      = e.get_int("subsampling_conv_stride", 2);
    ec.max_position_embeddings      = e.get_int("max_position_embeddings", 5000);
    ec.scale_input                  = e.get_bool("scale_input", true);
    ec.attention_bias               = e.get_bool("attention_bias", true);
    ec.convolution_bias             = e.get_bool("convolution_bias", true);
    ec.normalize_features           = e.get_bool("normalize_features", false);

    const j::Value& t = obj_at(root, "transformer_config", where);
    SortformerTransformerConfig& tc = c.transformer;
    tc.num_layers          = int_at(t, "num_layers", where);
    tc.hidden_size         = int_at(t, "hidden_size", where);
    tc.inner_size          = int_at(t, "inner_size", where);
    tc.num_attention_heads = int_at(t, "num_attention_heads", where);
    tc.pre_ln              = t.get_bool("pre_ln", false);

    const j::Value* s = root.find("streaming_config");
    if (s && s->is_object()) {
        SortformerStreamingConfig& sc = c.streaming;
        sc.spkcache_len                = s->get_int("spkcache_len", 188);
        sc.fifo_len                    = s->get_int("fifo_len", 0);
        sc.chunk_len                   = s->get_int("chunk_len", 188);
        sc.spkcache_update_period      = s->get_int("spkcache_update_period", 188);
        sc.chunk_left_context          = s->get_int("chunk_left_context", 1);
        sc.chunk_right_context         = s->get_int("chunk_right_context", 1);
        sc.spkcache_sil_frames_per_spk = s->get_int("spkcache_sil_frames_per_spk", 3);
        sc.pred_score_threshold = s->get_float("pred_score_threshold", 0.25f);
        sc.scores_boost_latest  = s->get_float("scores_boost_latest", 0.05f);
        sc.sil_threshold        = s->get_float("sil_threshold", 0.2f);
        sc.strong_boost_rate    = s->get_float("strong_boost_rate", 0.75f);
        sc.weak_boost_rate      = s->get_float("weak_boost_rate", 1.5f);
        sc.min_pos_scores_rate  = s->get_float("min_pos_scores_rate", 0.5f);
    }
    return c;
}

// ── weight-upload helpers (FP32 on `dev`) ──
const sf::TensorView& need(const sf::File& f, const std::string& name,
                           const std::string& where) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail(where, "missing tensor '" + name + "'");
    return *v;
}
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols,
              bt::Device dev, const std::string& where) {
    bt::Tensor t;
    {
        bt::DeviceScope cpu(bt::Device::CPU);
        sf::upload_compute_checked(need(f, name, where), rows, cols, t, name);
    }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
bt::Tensor up_vec(const sf::File& f, const std::string& name, int n,
                  bt::Device dev, const std::string& where) {
    return up(f, name, n, 1, dev, where);
}

Linear load_linear(const sf::File& f, const std::string& prefix, int out, int in,
                   bt::Device dev, const std::string& where) {
    Linear l;
    l.in_features  = in;
    l.out_features = out;
    l.W = up(f, prefix + ".weight", out, in, dev, where);
    l.b = up_vec(f, prefix + ".bias", out, dev, where);
    return l;
}

LayerNorm load_layernorm(const sf::File& f, const std::string& prefix, int d,
                         bt::Device dev, const std::string& where) {
    LayerNorm n;
    n.features = d;
    n.eps      = 1e-5f;
    n.gamma = up_vec(f, prefix + ".weight", d, dev, where);
    n.beta  = up_vec(f, prefix + ".bias",   d, dev, where);
    return n;
}

}  // namespace

// ─── SortformerTransformerBlock::forward ───────────────────────────────────

void SortformerTransformerBlock::forward(const bt::Tensor& x, const float* d_mask,
                                         bt::Tensor& out) const {
    const bt::Device dev = x.device;
    bt::DeviceScope scope(dev);

    // a = MHA(x); x1 = norm1(x + a)
    bt::Tensor a; attn.forward(x, d_mask, a);
    bt::Tensor s = x.clone();
    bt::add_inplace_batched(s, a);
    bt::Tensor x1; norm1.forward(s, x1);

    // f = ff_out(relu(ff_in(x1))); out = norm2(x1 + f)
    bt::Tensor h; ff_in.forward_batched(x1, h);
    bt::relu_forward(h, h);
    bt::Tensor fwd; ff_out.forward_batched(h, fwd);
    bt::add_inplace_batched(x1, fwd);
    norm2.forward(x1, out);
}

// ─── SortformerHead ─────────────────────────────────────────────────────────

void SortformerHead::load(const sf::File& f, const SortformerConfig& c,
                          bt::Device dev) {
    const std::string where = "SortformerHead::load";
    cfg    = c;
    device = dev;

    const int fc  = c.fc_d_model;
    const int tf  = c.tf_d_model;
    const int inr = c.transformer.inner_size;
    const int nh  = c.transformer.num_attention_heads;

    encoder_proj = load_linear(f, "sortformer.encoder_proj", tf, fc, dev, where);

    layers.clear();
    layers.resize(static_cast<std::size_t>(c.transformer.num_layers));
    for (int i = 0; i < c.transformer.num_layers; ++i) {
        const std::string L = "transformer.layers." + std::to_string(i) + ".";
        SortformerTransformerBlock& bl = layers[static_cast<std::size_t>(i)];
        bl.norm1 = load_layernorm(f, L + "norm1", tf, dev, where);
        bl.norm2 = load_layernorm(f, L + "norm2", tf, dev, where);
        bl.attn.num_heads = nh;
        bl.attn.embed_dim = tf;
        bl.attn.Wq = up(f, L + "attn.q.weight", tf, tf, dev, where);
        bl.attn.bq = up_vec(f, L + "attn.q.bias", tf, dev, where);
        bl.attn.Wk = up(f, L + "attn.k.weight", tf, tf, dev, where);
        bl.attn.bk = up_vec(f, L + "attn.k.bias", tf, dev, where);
        bl.attn.Wv = up(f, L + "attn.v.weight", tf, tf, dev, where);
        bl.attn.bv = up_vec(f, L + "attn.v.bias", tf, dev, where);
        bl.attn.Wo = up(f, L + "attn.o.weight", tf, tf, dev, where);
        bl.attn.bo = up_vec(f, L + "attn.o.bias", tf, dev, where);
        bl.ff_in  = load_linear(f, L + "ff.in",  inr, tf,  dev, where);
        bl.ff_out = load_linear(f, L + "ff.out", tf,  inr, dev, where);
    }

    first_hidden_to_hidden =
        load_linear(f, "sortformer.first_hidden_to_hidden", tf, tf, dev, where);
    single_hidden_to_spks =
        load_linear(f, "sortformer.single_hidden_to_spks", c.num_spks, tf, dev, where);
}

void SortformerHead::forward(const bt::Tensor& enc_out, const float* d_mask,
                             bt::Tensor& preds) const {
    bt::DeviceScope scope(device);

    // encoder_proj: (T, fc) -> (T, tf)
    bt::Tensor h; encoder_proj.forward_batched(enc_out, h);

    // post-LN Transformer stack
    bt::Tensor cur = std::move(h);
    for (const SortformerTransformerBlock& bl : layers) {
        bt::Tensor nxt;
        bl.forward(cur, d_mask, nxt);
        cur = std::move(nxt);
    }

    // forward_speaker_sigmoids: relu -> first_hidden_to_hidden -> relu ->
    // single_hidden_to_spks -> sigmoid.
    bt::relu_forward(cur, cur);
    bt::Tensor hh; first_hidden_to_hidden.forward_batched(cur, hh);
    bt::relu_forward(hh, hh);
    bt::Tensor sp; single_hidden_to_spks.forward_batched(hh, sp);   // (T, num_spks)
    bt::sigmoid_forward(sp, preds);
}

// ─── Sortformer::Impl ──────────────────────────────────────────────────────

struct Sortformer::Impl {
    SortformerConfig     config;
    bt::Device           device = bt::Device::CPU;
    bool                 loaded = false;

    FastConformerEncoder encoder;
    SortformerHead       head;

    Sortformer::Diarization diarize(const AudioBuffer& audio) const;
    Sortformer::Diarization run_streaming(SortformerSessionState& st,
                                          const std::vector<float>& mel,
                                          int frames) const;
};

Sortformer::Diarization Sortformer::Impl::diarize(const AudioBuffer& audio) const {
    if (!loaded)
        fail("Sortformer::diarize", "no model loaded; call Sortformer::load() first");
    if (audio.samples.empty())
        fail("Sortformer::diarize", "audio buffer is empty");
    if (audio.sample_rate != config.sample_rate)
        fail("Sortformer::diarize",
             "audio.sample_rate must be 16000 Hz; resampling is the caller's "
             "responsibility");

    bt::DeviceScope scope(device);

    bt::Tensor enc;
    encoder.forward(audio, enc);            // (T, fc_d_model)
    const int T = enc.rows;

    // Mask the trailing pad frame out of the transformer head's attention, the
    // same valid length the encoder used (NeMo forward_infer encoder_mask).
    const int valid = encoder.valid_output_frames(
        static_cast<int>(audio.samples.size()));
    bt::Tensor dmask_t;   // (1, T) on device — the head's MHA reads it there
    const float* dmask_ptr = nullptr;
    if (valid >= 0 && valid < T) {
        std::vector<float> dm(static_cast<std::size_t>(T), 1.0f);
        for (int t = valid; t < T; ++t) dm[static_cast<std::size_t>(t)] = 0.0f;
        dmask_t = bt::Tensor::from_host_on(device, dm.data(), 1, T);
        dmask_ptr = static_cast<const float*>(dmask_t.data);
    }
    bt::Tensor preds_t;
    head.forward(enc, dmask_ptr, preds_t);            // (T, num_spks)

    Sortformer::Diarization out;
    out.num_frames   = T;
    out.num_speakers = config.num_spks;
    out.frame_seconds = config.frame_seconds();
    out.probs.resize(static_cast<std::size_t>(T) * config.num_spks);
    {
        bt::DeviceScope cpu(bt::Device::CPU);
        bt::Tensor host = (preds_t.device == bt::Device::CPU)
                              ? preds_t
                              : preds_t.to(bt::Device::CPU);
        std::memcpy(out.probs.data(), host.host_f32(),
                    out.probs.size() * sizeof(float));
    }
    return out;
}

Sortformer::Sortformer() : impl_(std::make_unique<Impl>()) {}
Sortformer::~Sortformer() = default;
Sortformer::Sortformer(Sortformer&&) noexcept = default;
Sortformer& Sortformer::operator=(Sortformer&&) noexcept = default;

void Sortformer::load(const std::string& model_dir, bt::Device device) {
    bt::init();

    const fs::path dir         = model_dir;
    const fs::path config_path = dir / "config.json";
    const fs::path weight_path = dir / "model.safetensors";
    if (!fs::exists(config_path))
        fail("Sortformer::load", "no config.json under '" + model_dir + "'");
    if (!fs::exists(weight_path))
        fail("Sortformer::load", "no model.safetensors under '" + model_dir + "'");

    impl_->config = parse_config(config_path.string());
    impl_->device = device;

    // NeMo runs the encoder length-aware: the trailing center-STFT pad frame is
    // zeroed and masked out of the valid frames. Replicate it for parity.
    impl_->config.encoder.mask_padding = true;

    sf::File weights = sf::File::open(weight_path.string());
    impl_->encoder.load(weights, impl_->config.encoder, device);
    impl_->head.load(weights, impl_->config, device);
    impl_->loaded = true;
}

Sortformer::Diarization Sortformer::diarize(const AudioBuffer& audio) const {
    return impl_->diarize(audio);
}

const SortformerConfig& Sortformer::config() const { return impl_->config; }
bool Sortformer::loaded() const { return impl_->loaded; }

// ════════════════════════════════════════════════════════════════════════════
//  Streaming inference — Arrival-Order Speaker Cache (AOSC)
//
//  Faithful port of NeMo's SortformerModules streaming_update / _compress_spkcache
//  for a single stream (batch size 1). The speaker cache and FIFO queue hold
//  pre-encode (post-subsampling, 512-d) embeddings on the host; each chunk
//  re-runs the Conformer layers + transformer head over [spkcache | fifo | chunk]
//  and then folds the chunk into the cache, compressing to spkcache_len frames
//  by the arrival-time-ordered importance scoring.
// ════════════════════════════════════════════════════════════════════════════

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr int   kMaxIndex = 99999;   // NeMo placeholder for disabled frames

int conv_len3(int n) {
    auto cl = [](int x) { return (x - 1) / 2 + 1; };
    return cl(cl(cl(n)));
}

// Per-frame, per-speaker importance score (NeMo _get_log_pred_scores):
//   s[f,i] = log P_i - log(1-P_i) + sum_j log(1-P_j) - log(0.5)   (clamped).
void get_log_pred_scores(const float* preds, int n, int spk, float thr,
                         std::vector<float>& scores) {
    scores.assign(static_cast<std::size_t>(n) * spk, 0.0f);
    const float log_half = std::log(0.5f);
    for (int f = 0; f < n; ++f) {
        const float* p = preds + static_cast<std::size_t>(f) * spk;
        float log1_sum = 0.0f;
        for (int i = 0; i < spk; ++i)
            log1_sum += std::log(std::max(1.0f - p[i], thr));
        float* s = scores.data() + static_cast<std::size_t>(f) * spk;
        for (int i = 0; i < spk; ++i) {
            const float lp  = std::log(std::max(p[i], thr));
            const float l1p = std::log(std::max(1.0f - p[i], thr));
            s[i] = lp - l1p + log1_sum - log_half;
        }
    }
}

// NeMo _disable_low_scores: non-speech -> -inf; and per speaker with >= min_pos
// positive frames, its non-positive speech frames -> -inf.
void disable_low_scores(const float* preds, std::vector<float>& scores,
                        int n, int spk, int min_pos) {
    for (int i = 0; i < spk; ++i) {
        int pos_count = 0;
        for (int f = 0; f < n; ++f) {
            const float p = preds[static_cast<std::size_t>(f) * spk + i];
            float& s = scores[static_cast<std::size_t>(f) * spk + i];
            if (!(p > 0.5f)) s = kNegInf;             // non-speech
            else if (s > 0.0f) ++pos_count;
        }
        if (pos_count >= min_pos) {
            for (int f = 0; f < n; ++f) {
                const float p = preds[static_cast<std::size_t>(f) * spk + i];
                float& s = scores[static_cast<std::size_t>(f) * spk + i];
                if (p > 0.5f && !(s > 0.0f)) s = kNegInf;   // overlapped speech
            }
        }
    }
}

// NeMo _boost_topk_scores: add scale*log(2) to the top-`n_boost` frames per spk.
void boost_topk(std::vector<float>& scores, int n, int spk, int n_boost,
                float scale) {
    if (n_boost <= 0) return;
    const float boost = -scale * std::log(0.5f);   // = scale*log(2)
    std::vector<int> idx(static_cast<std::size_t>(n));
    for (int i = 0; i < spk; ++i) {
        for (int f = 0; f < n; ++f) idx[static_cast<std::size_t>(f)] = f;
        const int k = std::min(n_boost, n);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [&](int a, int b) {
                return scores[static_cast<std::size_t>(a) * spk + i] >
                       scores[static_cast<std::size_t>(b) * spk + i];
            });
        for (int j = 0; j < k; ++j)
            scores[static_cast<std::size_t>(idx[static_cast<std::size_t>(j)]) * spk + i] += boost;
    }
}

// NeMo _get_silence_profile: running mean of embeddings whose total speaker prob
// is below sil_threshold.
void update_silence_profile(std::vector<float>& mean_sil, int& n_sil,
                            const float* embs, const float* preds,
                            int n, int fc, int spk, float sil_thr) {
    if (mean_sil.empty()) mean_sil.assign(static_cast<std::size_t>(fc), 0.0f);
    std::vector<double> sum(static_cast<std::size_t>(fc), 0.0);
    int count = 0;
    for (int f = 0; f < n; ++f) {
        float tot = 0.0f;
        for (int i = 0; i < spk; ++i) tot += preds[static_cast<std::size_t>(f) * spk + i];
        if (tot < sil_thr) {
            ++count;
            const float* e = embs + static_cast<std::size_t>(f) * fc;
            for (int d = 0; d < fc; ++d) sum[static_cast<std::size_t>(d)] += e[d];
        }
    }
    if (count == 0) return;
    const int new_n = n_sil + count;
    for (int d = 0; d < fc; ++d) {
        const double old = static_cast<double>(mean_sil[static_cast<std::size_t>(d)]) * n_sil;
        mean_sil[static_cast<std::size_t>(d)] =
            static_cast<float>((old + sum[static_cast<std::size_t>(d)]) /
                               std::max(1, new_n));
    }
    n_sil = new_n;
}

// NeMo _compress_spkcache + _get_topk_indices + _gather_spkcache_and_preds:
// keep the SC most important frames of `embs` (arrival-ordered), filling
// disabled slots with the mean silence embedding.
void compress_spkcache(const std::vector<float>& embs,
                       const std::vector<float>& preds, int n, int fc, int spk,
                       const std::vector<float>& mean_sil,
                       const SortformerStreamingConfig& cfg,
                       std::vector<float>& out_emb, std::vector<float>& out_preds) {
    const int SC  = cfg.spkcache_len;
    const int sil = cfg.spkcache_sil_frames_per_spk;
    const int per_spk = SC / spk - sil;
    const int strong  = static_cast<int>(std::floor(per_spk * cfg.strong_boost_rate));
    const int weak    = static_cast<int>(std::floor(per_spk * cfg.weak_boost_rate));
    const int min_pos = static_cast<int>(std::floor(per_spk * cfg.min_pos_scores_rate));

    std::vector<float> scores;
    get_log_pred_scores(preds.data(), n, spk, cfg.pred_score_threshold, scores);
    disable_low_scores(preds.data(), scores, n, spk, min_pos);
    // Boost newly added frames (those past the previous spkcache length).
    if (cfg.scores_boost_latest > 0.0f)
        for (int f = SC; f < n; ++f)
            for (int i = 0; i < spk; ++i)
                scores[static_cast<std::size_t>(f) * spk + i] += cfg.scores_boost_latest;
    boost_topk(scores, n, spk, strong, 2.0f);
    boost_topk(scores, n, spk, weak,   1.0f);

    // Append `sil` rows of +inf (the silence slots), then take top-SC over all
    // (speaker, frame) entries.
    const int n_ext = n + sil;
    struct Cand { float v; int flat; };
    std::vector<Cand> cand;
    cand.reserve(static_cast<std::size_t>(spk) * n_ext);
    for (int i = 0; i < spk; ++i) {
        for (int f = 0; f < n; ++f)
            cand.push_back({scores[static_cast<std::size_t>(f) * spk + i], i * n_ext + f});
        for (int f = n; f < n_ext; ++f)
            cand.push_back({std::numeric_limits<float>::infinity(), i * n_ext + f});
    }
    const int K = std::min(SC, static_cast<int>(cand.size()));
    std::partial_sort(cand.begin(), cand.begin() + K, cand.end(),
        [](const Cand& a, const Cand& b) { return a.v > b.v; });

    // Map to frame indices; -inf -> placeholder; sort to restore arrival order.
    std::vector<int> topk(static_cast<std::size_t>(K));
    for (int k = 0; k < K; ++k)
        topk[static_cast<std::size_t>(k)] =
            (cand[static_cast<std::size_t>(k)].v == kNegInf)
                ? kMaxIndex : cand[static_cast<std::size_t>(k)].flat;
    std::sort(topk.begin(), topk.end());

    out_emb.assign(static_cast<std::size_t>(SC) * fc, 0.0f);
    out_preds.assign(static_cast<std::size_t>(SC) * spk, 0.0f);
    for (int k = 0; k < SC; ++k) {
        bool disabled = (k >= K);
        int frame = 0;
        if (!disabled) {
            const int t = topk[static_cast<std::size_t>(k)];
            disabled = (t == kMaxIndex);
            if (!disabled) {
                frame = t % n_ext;
                if (frame >= n) disabled = true;   // an appended silence slot
            }
        }
        float* oe = out_emb.data() + static_cast<std::size_t>(k) * fc;
        float* op = out_preds.data() + static_cast<std::size_t>(k) * spk;
        if (disabled) {
            if (!mean_sil.empty())
                std::copy_n(mean_sil.data(), fc, oe);
            // preds stay zero
        } else {
            std::copy_n(embs.data() + static_cast<std::size_t>(frame) * fc, fc, oe);
            std::copy_n(preds.data() + static_cast<std::size_t>(frame) * spk, spk, op);
        }
    }
}

}  // namespace

struct SortformerSessionState {
    // AOSC state (host). Embeddings are fc_d_model-wide pre-encode vectors.
    std::vector<float> spkcache;        // (sc_len * fc)
    int                sc_len = 0;
    std::vector<float> spkcache_preds;  // (sc_len * n_spk)
    bool               sc_preds_set = false;
    std::vector<float> fifo;            // (fifo_cur * fc)
    int                fifo_cur = 0;
    std::vector<float> fifo_preds;      // (fifo_cur * n_spk)
    std::vector<float> mean_sil;        // (fc)
    int                n_sil = 0;
    std::vector<float> pcm;             // buffered input until is_last

    void reset() { *this = SortformerSessionState{}; }
};

namespace {

// Download a device tensor (r, c) to a host row-major vector.
std::vector<float> to_host_vec(const bt::Tensor& t) {
    bt::DeviceScope cpu(bt::Device::CPU);
    bt::Tensor h = (t.device == bt::Device::CPU) ? t : t.to(bt::Device::CPU);
    std::vector<float> v(static_cast<std::size_t>(h.rows) * h.cols);
    std::copy_n(h.host_f32(), v.size(), v.data());
    return v;
}

}  // namespace

// Run the AOSC streaming loop over the full mel of a clip and return all frames.
Sortformer::Diarization Sortformer::Impl::run_streaming(
    SortformerSessionState& st, const std::vector<float>& mel, int frames) const {
    const SortformerConfig& cfg = config;
    const int fc  = cfg.fc_d_model;
    const int spk = cfg.num_spks;
    const int SUB = cfg.encoder.subsampling_factor;
    const int n_mels = cfg.encoder.num_mel_bins;
    const SortformerStreamingConfig& sc = cfg.streaming;
    const bt::Device dev = device;
    bt::DeviceScope scope(dev);

    const int chunk_mel = sc.chunk_len * SUB;
    const int valid_mel_total = frames;   // mel already pad-zeroed by log_mel

    std::vector<float> total_preds;   // (n_total * spk)
    int n_total = 0;

    int stt = 0, end = 0;
    while (end < frames) {
        const int left_off  = std::min(sc.chunk_left_context * SUB, stt);
        end = std::min(stt + chunk_mel, frames);
        const int right_off = std::min(sc.chunk_right_context * SUB, frames - end);
        const int cm_start = stt - left_off;
        const int cm_end   = end + right_off;
        const int cm_width = cm_end - cm_start;
        // valid mel frames within this chunk window.
        const int feat_len = std::max(0, std::min(valid_mel_total, frames) - cm_start);
        const int feat_valid = std::min(feat_len, cm_width);

        // Slice chunk mel (host) and pre-encode it.
        std::vector<float> cmel(static_cast<std::size_t>(cm_width) * n_mels);
        std::copy_n(mel.data() + static_cast<std::size_t>(cm_start) * n_mels,
                    cmel.size(), cmel.data());
        bt::Tensor chunk_pre_dev;
        encoder.pre_encode(cmel, cm_width, chunk_pre_dev);   // (chunk_total, fc)
        const int chunk_total = chunk_pre_dev.rows;
        const int chunk_valid = conv_len3(feat_valid);
        std::vector<float> chunk_pre = to_host_vec(chunk_pre_dev);

        const int lc = static_cast<int>(std::lround(
            static_cast<double>(left_off) / SUB));
        const int rc = static_cast<int>(std::ceil(
            static_cast<double>(right_off) / SUB));

        // Concatenate [spkcache | fifo | chunk] (host) and upload.
        const int concat_total = st.sc_len + st.fifo_cur + chunk_total;
        const int concat_valid = st.sc_len + st.fifo_cur + chunk_valid;
        std::vector<float> concat(static_cast<std::size_t>(concat_total) * fc);
        std::copy_n(st.spkcache.data(), static_cast<std::size_t>(st.sc_len) * fc,
                    concat.data());
        std::copy_n(st.fifo.data(), static_cast<std::size_t>(st.fifo_cur) * fc,
                    concat.data() + static_cast<std::size_t>(st.sc_len) * fc);
        std::copy_n(chunk_pre.data(), static_cast<std::size_t>(chunk_total) * fc,
                    concat.data() +
                        static_cast<std::size_t>(st.sc_len + st.fifo_cur) * fc);
        bt::Tensor concat_dev = bt::Tensor::from_host_on(dev, concat.data(),
                                                         concat_total, fc);

        // Encoder layers (bypass pre-encode) + head over the concatenation.
        const int valid_arg =
            (concat_valid < concat_total) ? concat_valid : -1;
        bt::Tensor enc_dev;
        encoder.encode_layers(concat_dev, enc_dev, valid_arg);
        bt::Tensor dmask_t;
        const float* dmask = nullptr;
        if (valid_arg >= 0) {
            std::vector<float> dm(static_cast<std::size_t>(concat_total), 1.0f);
            for (int t = concat_valid; t < concat_total; ++t)
                dm[static_cast<std::size_t>(t)] = 0.0f;
            dmask_t = bt::Tensor::from_host_on(dev, dm.data(), 1, concat_total);
            dmask = static_cast<const float*>(dmask_t.data);
        }
        bt::Tensor preds_dev;
        head.forward(enc_dev, dmask, preds_dev);   // (concat_total, spk)
        std::vector<float> preds = to_host_vec(preds_dev);
        // apply_mask_to_preds: zero padded frames.
        for (int f = concat_valid; f < concat_total; ++f)
            for (int i = 0; i < spk; ++i)
                preds[static_cast<std::size_t>(f) * spk + i] = 0.0f;

        // ── streaming_update (synchronous, batch 1) ──
        const int chunk_len = chunk_total - lc - rc;
        const int base = st.sc_len + st.fifo_cur;   // offset of chunk in preds
        // chunk preds (the emitted frames) = preds[base+lc : base+lc+chunk_len].
        const float* chunk_preds_src =
            preds.data() + static_cast<std::size_t>(base + lc) * spk;

        // Append chunk (stripped of context) + its preds to the FIFO.
        for (int f = 0; f < chunk_len; ++f) {
            const float* e = chunk_pre.data() +
                             static_cast<std::size_t>(lc + f) * fc;
            st.fifo.insert(st.fifo.end(), e, e + fc);
            const float* p = chunk_preds_src + static_cast<std::size_t>(f) * spk;
            st.fifo_preds.insert(st.fifo_preds.end(), p, p + spk);
        }
        st.fifo_cur += chunk_len;

        if (st.fifo_cur > sc.fifo_len) {
            int pop = sc.spkcache_update_period;
            pop = std::max(pop, chunk_len - sc.fifo_len + (st.fifo_cur - chunk_len));
            pop = std::min(pop, st.fifo_cur);

            update_silence_profile(st.mean_sil, st.n_sil, st.fifo.data(),
                                   st.fifo_preds.data(), pop, fc, spk,
                                   sc.sil_threshold);
            // Move the first `pop` FIFO frames into the speaker cache.
            const float* pe = st.fifo.data();
            const float* pp = st.fifo_preds.data();
            st.spkcache.insert(st.spkcache.end(), pe,
                               pe + static_cast<std::size_t>(pop) * fc);
            if (st.sc_preds_set)
                st.spkcache_preds.insert(st.spkcache_preds.end(), pp,
                                         pp + static_cast<std::size_t>(pop) * spk);
            st.sc_len += pop;
            st.fifo.erase(st.fifo.begin(),
                          st.fifo.begin() + static_cast<std::size_t>(pop) * fc);
            st.fifo_preds.erase(st.fifo_preds.begin(),
                                st.fifo_preds.begin() + static_cast<std::size_t>(pop) * spk);
            st.fifo_cur -= pop;

            if (st.sc_len > sc.spkcache_len) {
                if (!st.sc_preds_set) {
                    // First compression: seed cache preds from the current
                    // forward's spkcache-region preds, then the popped preds.
                    st.spkcache_preds.assign(
                        preds.data(),
                        preds.data() + static_cast<std::size_t>(st.sc_len - pop) * spk);
                    st.spkcache_preds.insert(st.spkcache_preds.end(), pp,
                                             pp + static_cast<std::size_t>(pop) * spk);
                    st.sc_preds_set = true;
                }
                std::vector<float> ne, np;
                compress_spkcache(st.spkcache, st.spkcache_preds, st.sc_len, fc,
                                  spk, st.mean_sil, sc, ne, np);
                st.spkcache = std::move(ne);
                st.spkcache_preds = std::move(np);
                st.sc_len = sc.spkcache_len;
            }
        }

        // Emit the chunk's predictions.
        total_preds.insert(total_preds.end(), chunk_preds_src,
                           chunk_preds_src + static_cast<std::size_t>(chunk_len) * spk);
        n_total += chunk_len;

        stt = end;
    }

    // Trim to ceil(mel_frames / subsampling) — NeMo's final length.
    const int n_keep = std::min(n_total, (frames + SUB - 1) / SUB);
    Sortformer::Diarization out;
    out.num_frames = n_keep;
    out.num_speakers = spk;
    out.frame_seconds = cfg.frame_seconds();
    out.probs.assign(total_preds.begin(),
                     total_preds.begin() + static_cast<std::size_t>(n_keep) * spk);
    return out;
}

SortformerSession::SortformerSession()
    : state_(std::make_unique<SortformerSessionState>()) {}
SortformerSession::~SortformerSession() = default;
SortformerSession::SortformerSession(SortformerSession&&) noexcept = default;
SortformerSession& SortformerSession::operator=(SortformerSession&&) noexcept = default;

SortformerSession Sortformer::make_session() const {
    if (!impl_->loaded)
        fail("Sortformer::make_session", "no model loaded; call load() first");
    return SortformerSession{};
}

void Sortformer::reset(SortformerSession& session) const {
    session.state_->reset();
}

Sortformer::Diarization Sortformer::feed(SortformerSession& session,
                                         const AudioBuffer& audio,
                                         bool is_last) const {
    if (!impl_->loaded)
        fail("Sortformer::feed", "no model loaded; call load() first");
    if (audio.sample_rate != impl_->config.sample_rate)
        fail("Sortformer::feed", "audio.sample_rate must be 16000 Hz");

    SortformerSessionState& st = *session.state_;
    st.pcm.insert(st.pcm.end(), audio.samples.begin(), audio.samples.end());
    if (!is_last) return Sortformer::Diarization{};

    // Finalize: compute the full mel over the buffered PCM and run the AOSC
    // streaming loop. (The cache compression is exercised once the clip exceeds
    // chunk_len; shorter clips reduce to the single-chunk offline forward.)
    AudioBuffer buf;
    buf.sample_rate = impl_->config.sample_rate;
    buf.samples = std::move(st.pcm);
    st.pcm.clear();
    int frames = 0;
    const std::vector<float> mel = impl_->encoder.log_mel(buf, frames);
    return impl_->run_streaming(st, mel, frames);
}

}  // namespace brosoundml
