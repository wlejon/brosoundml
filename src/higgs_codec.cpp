// HiggsAudio v2 tokenizer — see include/brosoundml/higgs_codec.h (public
// contract) and src/higgs_codec_model.h (module graph). The reference is
// transformers' HiggsAudioV2TokenizerModel; the tensor map below names the
// audio_tokenizer/model.safetensors keys each stage reads.
#include "brosoundml/higgs_codec.h"

#include "higgs_codec_common.h"
#include "higgs_codec_model.h"
#include "brosoundml/detail/json.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;
namespace j  = detail::json;
namespace fs = std::filesystem;
using namespace hcodec;   // fail/need/up/up_vec, ncl<->seq, conv_same/strided/
                          // trans_conv_up, snake, elu, lnorm, download_idx

namespace {

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot read '" + path + "'");
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int product(const std::vector<int>& v) {
    int p = 1;
    for (int x : v) p *= x;
    return p;
}

// One DAC residual unit over an NCL activation (in place on `x`).
void dac_res_unit(bt::Tensor& x, int C, int L, const HiggsResUnit& u) {
    bt::Tensor res = x;   // residual input (deep copy)
    snake(x, C, L, u.a1);
    bt::Tensor t1 = conv_same(x, C, L, u.c1w, &u.c1b, C, 7, u.dilation);
    snake(t1, C, L, u.a2);
    bt::Tensor t2 = conv_same(t1, C, L, u.c2w, &u.c2b, C, 1);
    bt::add_inplace(res, t2);
    x = std::move(res);
}

void load_res_unit(const sf::File& f, const std::string& R, int C, int dilation,
                   bt::Device dev, HiggsResUnit& u) {
    u.dim = C;
    u.dilation = dilation;
    u.a1  = up(f, R + "snake1.alpha", C, 1, dev);
    u.c1w = up(f, R + "conv1.weight", C, C * 7, dev);
    u.c1b = up_vec(f, R + "conv1.bias", C, dev);
    u.a2  = up(f, R + "snake2.alpha", C, 1, dev);
    u.c2w = up(f, R + "conv2.weight", C, C * 1, dev);
    u.c2b = up_vec(f, R + "conv2.bias", C, dev);
}

}  // namespace

// ─── HiggsCodecModel::load ───────────────────────────────────────────────────

void HiggsCodecModel::load(const std::string& dir, bt::Device device, bool decoder_only) {
    const fs::path d(dir);
    const std::string config_path  = (d / "config.json").string();
    const std::string weights_path = (d / "model.safetensors").string();
    if (!fs::exists(config_path))  fail("no config.json under '" + dir + "'");
    if (!fs::exists(weights_path)) fail("no model.safetensors under '" + dir + "'");
    if (!bt::is_available(device)) fail("requested device is not available");

    // ── config.json ──
    const j::Value root = j::parse(slurp(config_path));
    if (!root.is_object()) fail("config.json is not a JSON object");
    HiggsCodecConfig c;
    c.sample_rate          = root.get_int("sample_rate", 24000);
    c.semantic_sample_rate = root.get_int("semantic_sample_rate", 16000);
    c.codebook_size        = root.get_int("codebook_size", 1024);
    c.codebook_dim         = root.get_int("codebook_dim", 64);
    const int downsample_factor = root.get_int("downsample_factor", 320);
    sem_kernel      = root.get_int("kernel_size", 3);
    sem_unit_kernel = root.get_int("unit_kernel_size", 3);
    sem_dilations   = root.get_int_array("block_dilations", {1, 1});
    {
        const std::vector<int> strides = root.get_int_array("strides", {1, 1});
        const std::vector<int> ratios  = root.get_int_array("channel_ratios", {1, 1});
        if (strides.size() != ratios.size()) fail("config.json: strides / channel_ratios length mismatch");
        for (std::size_t i = 0; i < strides.size(); ++i)
            if (strides[i] != 1 || ratios[i] != 1)
                fail("config.json: only stride-1, ratio-1 semantic encoder blocks are supported");
        num_sem_blocks = static_cast<int>(strides.size());
    }
    double bw_last = 2.0;
    if (const j::Value* bw = root.find("target_bandwidths"); bw && bw->is_array() && !bw->as_array().empty())
        bw_last = bw->as_array().back().as_number();

    const j::Value* ac = root.find("acoustic_model_config");
    if (!ac || !ac->is_object()) fail("config.json: missing acoustic_model_config");
    c.acoustic_dim  = ac->get_int("hidden_size", 256);
    c.encoder_dim   = ac->get_int("encoder_hidden_size", 64);
    c.decoder_dim   = ac->get_int("decoder_hidden_size", 1024);
    c.encoder_rates = ac->get_int_array("downsampling_ratios", {8, 5, 4, 2, 3});
    c.decoder_rates = ac->get_int_array("upsampling_ratios", {8, 5, 4, 2, 3});
    if (c.encoder_rates.empty() || c.decoder_rates.empty()) fail("config.json: empty DAC rates");
    c.hop_length = product(c.encoder_rates);
    if (product(c.decoder_rates) != c.hop_length) fail("config.json: encoder / decoder rates disagree on hop_length");
    c.frame_rate = (c.sample_rate + c.hop_length - 1) / c.hop_length;

    const j::Value* sm = root.find("semantic_model_config");
    if (!sm || !sm->is_object()) fail("config.json: missing semantic_model_config");
    HubertConfig hc;
    hc.hidden_size         = sm->get_int("hidden_size", 768);
    hc.num_hidden_layers   = sm->get_int("num_hidden_layers", 12);
    hc.num_attention_heads = sm->get_int("num_attention_heads", 12);
    hc.intermediate_size   = sm->get_int("intermediate_size", 3072);
    hc.conv_dim    = sm->get_int_array("conv_dim", hc.conv_dim);
    hc.conv_kernel = sm->get_int_array("conv_kernel", hc.conv_kernel);
    hc.conv_stride = sm->get_int_array("conv_stride", hc.conv_stride);
    hc.conv_bias   = sm->get_bool("conv_bias", false);
    hc.num_conv_pos_embeddings       = sm->get_int("num_conv_pos_embeddings", 128);
    hc.num_conv_pos_embedding_groups = sm->get_int("num_conv_pos_embedding_groups", 16);
    hc.layer_norm_eps          = sm->get_float("layer_norm_eps", 1e-5f);
    hc.feat_extract_norm_group = sm->get_string("feat_extract_norm", "group") == "group";
    hc.feat_proj_layer_norm    = sm->get_bool("feat_proj_layer_norm", true);
    hc.do_stable_layer_norm    = sm->get_bool("do_stable_layer_norm", false);
    if (sm->get_string("hidden_act", "gelu") != "gelu" ||
        sm->get_string("feat_extract_activation", "gelu") != "gelu")
        fail("config.json: HuBERT activations other than gelu are not supported");
    c.semantic_dim = hc.hidden_size;
    c.hidden_size  = c.acoustic_dim + c.semantic_dim;

    const int nbits = static_cast<int>(std::ceil(std::log2(static_cast<double>(c.codebook_size))));
    c.num_quantizers = static_cast<int>(std::floor(1000.0 * bw_last / (c.frame_rate * nbits)));
    if (c.num_quantizers <= 0) fail("config.json: target_bandwidths yields no quantizers");
    if ((static_cast<long long>(c.hop_length) * c.semantic_sample_rate) % c.sample_rate != 0)
        fail("config.json: hop_length does not map to a whole number of semantic samples");
    c.semantic_downsample_factor = static_cast<int>(
        (static_cast<double>(c.hop_length) / (static_cast<double>(c.sample_rate) / c.semantic_sample_rate)) /
        downsample_factor);
    if (c.semantic_downsample_factor <= 0) fail("config.json: bad semantic_downsample_factor");
    c.has_encoder = !decoder_only;

    // ── weights ──
    const sf::File f = sf::File::open(weights_path);
    const int Q  = c.num_quantizers;
    const int CB = c.codebook_size, CD = c.codebook_dim, HD = c.hidden_size;

    // residual VQ: quantizer.quantizers.{q}.{codebook.embed, project_in, project_out}
    embed.clear(); pin_w.clear(); pin_b.clear(); pout_w.clear(); pout_b.clear();
    for (int q = 0; q < Q; ++q) {
        const std::string P = "quantizer.quantizers." + std::to_string(q) + ".";
        embed.push_back(up(f, P + "codebook.embed", CB, CD, device));
        pout_w.push_back(up(f, P + "project_out.weight", HD, CD, device));
        pout_b.push_back(up_vec(f, P + "project_out.bias", HD, device));
        if (!decoder_only) {
            pin_w.push_back(up(f, P + "project_in.weight", CD, HD, device));
            pin_b.push_back(up_vec(f, P + "project_in.bias", CD, device));
        }
    }

    // decoder: fc2, acoustic_decoder.{conv1, block.{i}, snake1, conv2}
    fc2_w = up(f, "fc2.weight", c.acoustic_dim, HD, device);
    fc2_b = up_vec(f, "fc2.bias", c.acoustic_dim, device);
    dconv1_w = up(f, "acoustic_decoder.conv1.weight", c.decoder_dim, c.acoustic_dim * 7, device);
    dconv1_b = up_vec(f, "acoustic_decoder.conv1.bias", c.decoder_dim, device);
    static const int kDil[3] = {1, 3, 9};
    dec_blocks.clear();
    dec_blocks.resize(c.decoder_rates.size());
    for (std::size_t i = 0; i < dec_blocks.size(); ++i) {
        HiggsDecBlock& b = dec_blocks[i];
        b.in     = c.decoder_dim >> i;
        b.out    = c.decoder_dim >> (i + 1);
        b.stride = c.decoder_rates[i];
        const std::string B = "acoustic_decoder.block." + std::to_string(i) + ".";
        b.snake = up(f, B + "snake1.alpha", b.in, 1, device);
        b.tw = up(f, B + "conv_t1.weight", b.in, b.out * 2 * b.stride, device);
        b.tb = up_vec(f, B + "conv_t1.bias", b.out, device);
        for (int u = 0; u < 3; ++u)
            load_res_unit(f, B + "res_unit" + std::to_string(u + 1) + ".", b.out, kDil[u], device, b.units[u]);
    }
    const int dout = c.decoder_dim >> c.decoder_rates.size();
    dsnake   = up(f, "acoustic_decoder.snake1.alpha", dout, 1, device);
    dconv2_w = up(f, "acoustic_decoder.conv2.weight", 1, dout * 7, device);
    dconv2_b = up_vec(f, "acoustic_decoder.conv2.bias", 1, device);

    enc_blocks.clear();
    sem_blocks.clear();
    hubert = HubertModel{};
    if (!decoder_only) {
        // acoustic encoder: acoustic_encoder.{conv1, block.{i}, snake1, conv2}
        econv1_w = up(f, "acoustic_encoder.conv1.weight", c.encoder_dim, 1 * 7, device);
        econv1_b = up_vec(f, "acoustic_encoder.conv1.bias", c.encoder_dim, device);
        enc_blocks.resize(c.encoder_rates.size());
        for (std::size_t i = 0; i < enc_blocks.size(); ++i) {
            HiggsEncBlock& b = enc_blocks[i];
            b.in     = c.encoder_dim << i;
            b.out    = c.encoder_dim << (i + 1);
            b.stride = c.encoder_rates[i];
            const std::string B = "acoustic_encoder.block." + std::to_string(i) + ".";
            for (int u = 0; u < 3; ++u)
                load_res_unit(f, B + "res_unit" + std::to_string(u + 1) + ".", b.in, kDil[u], device, b.units[u]);
            b.snake = up(f, B + "snake1.alpha", b.in, 1, device);
            b.cw = up(f, B + "conv1.weight", b.out, b.in * 2 * b.stride, device);
            b.cb = up_vec(f, B + "conv1.bias", b.out, device);
        }
        const int eout = c.encoder_dim << c.encoder_rates.size();
        esnake   = up(f, "acoustic_encoder.snake1.alpha", eout, 1, device);
        econv2_w = up(f, "acoustic_encoder.conv2.weight", c.acoustic_dim, eout * 3, device);
        econv2_b = up_vec(f, "acoustic_encoder.conv2.bias", c.acoustic_dim, device);

        // semantic encoder: encoder_semantic.{conv, conv_blocks.{i}.{res_units.{j}, conv}}
        const int S = c.semantic_dim;
        sconv_w = up(f, "encoder_semantic.conv.weight", S, S * sem_kernel, device);
        sem_blocks.resize(num_sem_blocks);
        for (int i = 0; i < num_sem_blocks; ++i) {
            HiggsSemBlock& b = sem_blocks[i];
            const std::string B = "encoder_semantic.conv_blocks." + std::to_string(i) + ".";
            b.units.resize(sem_dilations.size());
            for (std::size_t u = 0; u < sem_dilations.size(); ++u) {
                const std::string U = B + "res_units." + std::to_string(u) + ".";
                b.units[u].dilation = sem_dilations[u];
                b.units[u].c1w = up(f, U + "conv1.weight", S, S * sem_unit_kernel, device);
                b.units[u].c2w = up(f, U + "conv2.weight", S, S * 1, device);
            }
            b.cw = up(f, B + "conv.weight", S, S * 3, device);   // stride 1 -> kernel 3
            b.cb = up_vec(f, B + "conv.bias", S, device);
        }

        fc_w = up(f, "fc.weight", HD, HD, device);
        fc_b = up_vec(f, "fc.bias", HD, device);

        hubert.load(f, "semantic_model", hc, device);
        sem_resampler = make_sinc_resampler(c.sample_rate, c.semantic_sample_rate, device);
    }
    // decoder_semantic.* and fc1 are training-only (semantic reconstruction
    // loss) and are never read.

    cfg = c;
    dev = device;
    is_loaded = true;
}

// ─── HiggsCodecModel::decode ─────────────────────────────────────────────────

std::vector<float> HiggsCodecModel::decode(const std::int32_t* codes, int K, int T) const {
    if (!is_loaded) fail("decode: no model loaded");
    if (K < 1 || K > cfg.num_quantizers)
        fail("decode: num_quantizers must be in [1, " + std::to_string(cfg.num_quantizers) + "], got " + std::to_string(K));
    if (T <= 0) fail("decode: num_frames must be > 0");
    if (!codes) fail("decode: null codes");
    bt::DeviceScope scope(dev);

    // ── RVQ decode: sum_q project_out(embed_q[code_q]) -> (T, hidden) ──
    bt::Tensor acc = bt::Tensor::zeros_on(dev, T, cfg.hidden_size, bt::Dtype::FP32);
    std::vector<std::int32_t> idx(static_cast<std::size_t>(T));
    for (int q = 0; q < K; ++q) {
        const std::int32_t* row = codes + static_cast<std::size_t>(q) * T;
        for (int t = 0; t < T; ++t) {
            if (row[t] < 0 || row[t] >= cfg.codebook_size)
                fail("decode: code out of range at level " + std::to_string(q) + ", frame " + std::to_string(t));
            idx[t] = row[t];
        }
        bt::Tensor looked = qtd::gather_rows(embed[q], idx);   // (T, codebook_dim)
        bt::Tensor proj;
        qtd::linear(pout_w[q], &pout_b[q], looked, proj);       // (T, hidden)
        bt::add_inplace(acc, proj);
    }

    // ── fc2 -> DAC decoder ──
    bt::Tensor lat;
    qtd::linear(fc2_w, &fc2_b, acc, lat);                       // (T, acoustic_dim)
    bt::Tensor x = seq_to_ncl(lat, T, cfg.acoustic_dim);        // (1, acoustic_dim*T)
    int C = cfg.decoder_dim, L = T;
    x = conv_same(x, cfg.acoustic_dim, L, dconv1_w, &dconv1_b, C, 7);
    for (const HiggsDecBlock& b : dec_blocks) {
        snake(x, b.in, L, b.snake);
        x = trans_conv_up(x, b.in, L, b.tw, &b.tb, b.out, b.stride);
        L *= b.stride;
        C = b.out;
        for (const HiggsResUnit& u : b.units) dac_res_unit(x, C, L, u);
    }
    snake(x, C, L, dsnake);
    bt::Tensor wav = conv_same(x, C, L, dconv2_w, &dconv2_b, 1, 7);   // (1, L)

    std::vector<float> out(static_cast<std::size_t>(L));
    qtd::to_host(wav, out.data());
    return out;
}

// ─── HiggsCodecModel::encode ─────────────────────────────────────────────────

std::vector<std::int32_t> HiggsCodecModel::encode(const float* wav24, int n, int* T_out,
                                                  HiggsEncodeTrace* trace,
                                                  const float* sem16_override,
                                                  int n16_override) const {
    if (!is_loaded) fail("encode: no model loaded");
    if (!cfg.has_encoder) fail("encode: the encoder was not loaded (decoder_only)");
    if (n <= 0 || !wav24) fail("encode: empty input");
    if (n % cfg.hop_length != 0) fail("encode: input length must be a multiple of hop_length");
    bt::DeviceScope scope(dev);
    const int T = n / cfg.hop_length;
    const int S = cfg.semantic_dim;

    bt::Tensor x = bt::Tensor::from_host_on(dev, wav24, 1, n);   // (1, 1*n)

    // ── acoustic branch: DAC encoder ──
    bt::Tensor a = conv_same(x, 1, n, econv1_w, &econv1_b, cfg.encoder_dim, 7);
    int C = cfg.encoder_dim, L = n;
    for (const HiggsEncBlock& b : enc_blocks) {
        for (const HiggsResUnit& u : b.units) dac_res_unit(a, C, L, u);
        snake(a, b.in, L, b.snake);
        const int pad = (b.stride + 1) / 2;
        a = conv_strided(a, b.in, L, b.cw, &b.cb, b.out, 2 * b.stride, b.stride, pad);
        L = (L + 2 * pad - 2 * b.stride) / b.stride + 1;
        C = b.out;
    }
    if (L != T) fail("encode: acoustic branch produced " + std::to_string(L) + " frames, expected " + std::to_string(T));
    snake(a, C, L, esnake);
    a = conv_same(a, C, L, econv2_w, &econv2_b, cfg.acoustic_dim, 3);   // (1, acoustic_dim*T)

    // ── semantic branch: 24 -> 16 kHz (windowed sinc), pad 160, HuBERT mean-hidden, /2 ──
    bt::Tensor s16;
    int n16 = 0;
    if (sem16_override) {
        n16 = n16_override;
        s16 = bt::Tensor::from_host_on(dev, sem16_override, 1, n16);
    } else {
        s16 = sinc_resample(sem_resampler, x, n, &n16);
    }
    const int sem_pad = 160;
    bt::Tensor sp;
    bt::pad1d_forward(s16, /*N=*/1, /*C=*/1, n16, sem_pad, sem_pad, /*mode=*/0, sp);
    int Th = 0;
    bt::Tensor hub = hubert.forward_mean_hidden(sp, n16 + 2 * sem_pad, &Th);   // (Th, S)
    const int ds = cfg.semantic_downsample_factor;
    const int Ts = (Th + ds - 1) / ds;
    if (Ts != T) fail("encode: semantic branch produced " + std::to_string(Ts) + " frames, expected " + std::to_string(T));
    bt::Tensor sem;
    if (ds == 1) {
        sem = hub;
    } else {
        std::vector<std::int32_t> keep(static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t) keep[t] = t * ds;
        sem = qtd::gather_rows(hub, keep);                       // (T, S)
    }
    bt::Tensor s = seq_to_ncl(sem, T, S);                        // (1, S*T)
    s = conv_same(s, S, T, sconv_w, nullptr, S, sem_kernel);
    for (const HiggsSemBlock& b : sem_blocks) {
        for (const HiggsSemUnit& u : b.units) {
            bt::Tensor res = s;
            bt::Tensor h = elu(s);
            h = conv_same(h, S, T, u.c1w, nullptr, S, sem_unit_kernel, u.dilation);
            h = elu(h);
            h = conv_same(h, S, T, u.c2w, nullptr, S, 1);
            bt::add_inplace(res, h);
            s = std::move(res);
        }
        s = conv_same(s, S, T, b.cw, &b.cb, S, 3);
    }

    // ── concat [acoustic | semantic] -> fc ──
    bt::Tensor cat;
    bt::concat_nchw_channels({&a, &s}, /*N=*/1, /*H=*/1, /*W=*/T, {cfg.acoustic_dim, S}, cat);
    bt::Tensor z = ncl_to_seq(cat, cfg.hidden_size, T);          // (T, hidden)
    bt::Tensor e;
    qtd::linear(fc_w, &fc_b, z, e);                              // (T, hidden)

    if (trace) {
        trace->n24 = n; trace->n16 = n16; trace->Th = Th; trace->T = T;
        trace->sem16.resize(static_cast<std::size_t>(n16));
        qtd::to_host(s16, trace->sem16.data());
        trace->hubert.resize(static_cast<std::size_t>(Th) * S);
        qtd::to_host(hub, trace->hubert.data());
        trace->sem_enc.resize(static_cast<std::size_t>(S) * T);
        qtd::to_host(s, trace->sem_enc.data());
        trace->acoustic.resize(static_cast<std::size_t>(cfg.acoustic_dim) * T);
        qtd::to_host(a, trace->acoustic.data());
        bt::Tensor e_ncl = seq_to_ncl(e, T, cfg.hidden_size);
        trace->fc_out.resize(static_cast<std::size_t>(cfg.hidden_size) * T);
        qtd::to_host(e_ncl, trace->fc_out.data());
    }

    // ── residual VQ ──
    const int Q = cfg.num_quantizers;
    std::vector<std::int32_t> codes(static_cast<std::size_t>(Q) * T);
    bt::Tensor residual = e;
    for (int q = 0; q < Q; ++q) {
        bt::Tensor p;
        qtd::linear(pin_w[q], &pin_b[q], residual, p);           // (T, codebook_dim)
        bt::Tensor idx, quant;
        bt::vq_encode_forward(p, embed[q], idx, quant);          // (T,1) INT32, (T, codebook_dim)
        download_idx(idx, T, codes.data() + static_cast<std::size_t>(q) * T);
        if (q + 1 < Q) {
            bt::Tensor deq;
            qtd::linear(pout_w[q], &pout_b[q], quant, deq);      // (T, hidden)
            bt::scale_inplace(deq, -1.0f);
            bt::add_inplace(residual, deq);                      // residual -= decoded
        }
    }
    if (T_out) *T_out = T;
    return codes;
}

// ─── public HiggsCodec ───────────────────────────────────────────────────────

struct HiggsCodec::Impl {
    HiggsCodecModel m;
};

HiggsCodec::HiggsCodec() : impl_(new Impl) {}
HiggsCodec::~HiggsCodec() = default;
HiggsCodec::HiggsCodec(HiggsCodec&&) noexcept = default;
HiggsCodec& HiggsCodec::operator=(HiggsCodec&&) noexcept = default;

void HiggsCodec::load(const std::string& dir, bt::Device device, bool decoder_only) {
    auto fresh = std::make_unique<Impl>();
    fresh->m.load(dir, device, decoder_only);   // throws -> the current model is untouched
    impl_ = std::move(fresh);
}

AudioBuffer HiggsCodec::decode(const std::int32_t* codes, int num_quantizers, int num_frames) const {
    if (!impl_ || !impl_->m.is_loaded) fail("decode: no model loaded");
    std::vector<float> wav = impl_->m.decode(codes, num_quantizers, num_frames);
    return AudioBuffer(std::move(wav), impl_->m.cfg.sample_rate);
}

AudioBuffer HiggsCodec::decode(const std::vector<std::int32_t>& codes, int num_quantizers,
                               int num_frames) const {
    if (!impl_ || !impl_->m.is_loaded) fail("decode: no model loaded");
    if (num_quantizers <= 0 || num_frames <= 0 ||
        codes.size() != static_cast<std::size_t>(num_quantizers) * static_cast<std::size_t>(num_frames))
        fail("decode: codes.size() must equal num_quantizers * num_frames");
    return decode(codes.data(), num_quantizers, num_frames);
}

std::vector<std::int32_t> HiggsCodec::encode(const AudioBuffer& audio, int* num_frames_out) const {
    if (!impl_ || !impl_->m.is_loaded) fail("encode: no model loaded");
    const HiggsCodecModel& m = impl_->m;
    if (!m.cfg.has_encoder) fail("encode: the encoder was not loaded (decoder_only)");
    if (audio.empty()) fail("encode: empty audio");

    // To 24 kHz (windowed sinc, on the host), then right-pad to a whole frame.
    const int sr = m.cfg.sample_rate;
    const float* wav = audio.samples.data();
    int n = static_cast<int>(audio.samples.size());
    std::vector<float> resampled;
    if (audio.sample_rate != sr && audio.sample_rate > 0) {
        bt::DeviceScope cpu(bt::Device::CPU);
        const SincResampler rs = make_sinc_resampler(audio.sample_rate, sr, bt::Device::CPU);
        bt::Tensor x = bt::Tensor::from_host_on(bt::Device::CPU, wav, 1, n);
        int n_out = 0;
        bt::Tensor y = sinc_resample(rs, x, n, &n_out);
        resampled.assign(y.host_f32(), y.host_f32() + n_out);
        wav = resampled.data();
        n = n_out;
    }
    const int hop = m.cfg.hop_length;
    const int L = ((n + hop - 1) / hop) * hop;
    std::vector<float> buf(static_cast<std::size_t>(L), 0.0f);
    std::copy(wav, wav + n, buf.begin());
    return m.encode(buf.data(), L, num_frames_out);
}

const HiggsCodecConfig& HiggsCodec::config() const {
    static const HiggsCodecConfig kDefault{};
    return impl_ ? impl_->m.cfg : kDefault;   // moved-from: the defaults
}
bool HiggsCodec::loaded() const { return impl_ && impl_->m.is_loaded; }
bool HiggsCodec::has_encoder() const { return loaded() && impl_->m.cfg.has_encoder; }
bt::Device HiggsCodec::device() const { return impl_ ? impl_->m.dev : bt::Device::CPU; }

}  // namespace brosoundml
