// YuE2 backbone: load + AR path — see yue2_model.h.

#include "yue2_model_impl.h"

#include "brosoundml/detail/json.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace brosoundml::yue2 {

namespace sf = brotensor::safetensors;

void model_fail(const std::string& msg) { throw std::runtime_error("brosoundml: Yue2: " + msg); }

namespace {

float bf16r(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

float bf16_to_f32(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

void h2d(bt::Device dev, void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    if (dev.is_cpu()) std::memcpy(dst, src, bytes);
    else bt::detail::alloc_for(dev).memcpy_h2d(dst, src, bytes, dev.index);
}

// The rotate_half -> interleaved-pair permutation of one head's rows:
// new 2i <- old i, new 2i+1 <- old i + hd/2.
std::vector<int> rope_perm(int heads, int hd) {
    std::vector<int> p(static_cast<std::size_t>(heads) * hd);
    const int half = hd / 2;
    for (int h = 0; h < heads; ++h)
        for (int i = 0; i < half; ++i) {
            p[h * hd + 2 * i] = h * hd + i;
            p[h * hd + 2 * i + 1] = h * hd + i + half;
        }
    return p;
}

bt::Tensor load_w(const sf::File& f, const ModelWeights& w, const std::string& name, int rows, int cols,
                  const std::vector<int>* perm = nullptr) {
    const sf::TensorView* v = f.find(name);
    if (!v) model_fail("missing tensor '" + name + "'");
    if (v->dtype != sf::Dtype::BF16) model_fail("tensor '" + name + "' is not BF16");
    if (v->numel() != static_cast<int64_t>(rows) * cols)
        model_fail("tensor '" + name + "' has " + std::to_string(v->numel()) + " elements, expected " +
                   std::to_string(rows) + "x" + std::to_string(cols));
    const auto* src = reinterpret_cast<const std::uint16_t*>(v->data);
    std::vector<std::uint16_t> permuted;
    if (perm) {
        if (static_cast<int>(perm->size()) != rows) model_fail("bad permutation for '" + name + "'");
        permuted.resize(static_cast<std::size_t>(rows) * cols);
        for (int r = 0; r < rows; ++r)
            std::memcpy(permuted.data() + static_cast<std::size_t>(r) * cols,
                        src + static_cast<std::size_t>((*perm)[r]) * cols, cols * sizeof(std::uint16_t));
        src = permuted.data();
    }
    if (w.dt == bt::Dtype::BF16) return bt::Tensor::from_host_bf16_on(w.dev, src, rows, cols);
    std::vector<float> f32(static_cast<std::size_t>(rows) * cols);
    for (std::size_t i = 0; i < f32.size(); ++i) f32[i] = bf16_to_f32(src[i]);
    return bt::Tensor::from_host_on(w.dev, f32.data(), rows, cols);
}

}  // namespace

Config Config::from_json(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) model_fail("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    const auto j = brosoundml::detail::json::parse(ss.str());
    Config c;
    c.hidden = j.get_int("hidden_size", c.hidden);
    c.layers = j.get_int("num_hidden_layers", c.layers);
    c.n_q = j.get_int("num_attention_heads", c.n_q);
    c.n_kv = j.get_int("num_key_value_heads", c.n_kv);
    c.head_dim = j.get_int("head_dim", c.head_dim);
    c.inter = j.get_int("intermediate_size", c.inter);
    c.vocab = j.get_int("vocab_size", c.vocab);
    c.latent = j.get_int("latent_dim", c.latent);
    c.max_latent_frames = j.get_int("max_latent_frames", c.max_latent_frames);
    c.max_pos = j.get_int("max_position_embeddings", c.max_pos);
    c.eps = j.get_float("rms_norm_eps", c.eps);
    c.rope_theta = j.get_float("rope_theta", c.rope_theta);
    c.timestep_shift = j.get_float("timestep_shift", c.timestep_shift);
    if (c.n_q % c.n_kv != 0 || c.head_dim % 2 != 0) model_fail("unsupported head layout");
    return c;
}

// ─── shared helpers ─────────────────────────────────────────────────────────

bt::Tensor rows_view(const bt::Tensor& t, int row0, int rows) {
    const std::size_t es = static_cast<std::size_t>(bt::dtype_size_bytes(t.dtype));
    char* p = static_cast<char*>(t.data) + static_cast<std::size_t>(row0) * t.cols * es;
    return bt::Tensor::view(t.device, p, rows, t.cols, t.dtype);
}

bt::Tensor reshape_view(const bt::Tensor& t, int rows, int cols) {
    if (static_cast<long long>(rows) * cols > static_cast<long long>(t.rows) * t.cols)
        model_fail("reshape_view out of bounds");
    return bt::Tensor::view(t.device, t.data, rows, cols, t.dtype);
}

void lin(const bt::Tensor& W, const bt::Tensor* bias, const bt::Tensor& X, bt::Tensor& Y, bt::Tensor* ws,
         bool acc) {
    bt::linear_forward_batched_ex(W, bias, X, bt::kLinearActNone,
                                  acc ? bt::kLinearEpiAccumulate : bt::kLinearEpiStore, ws, Y);
}

void head_norm(const ModelWeights& w, const bt::Tensor& src, int heads, const bt::Tensor& gain,
               bt::Tensor& dst) {
    const int hd = w.cfg.head_dim;
    bt::Tensor v = bt::Tensor::view(src.device, src.data, src.rows * heads, hd, src.dtype);
    bt::rms_norm_forward(v, gain, w.cfg.eps, dst);
}

void upload_ids(const ModelWeights& w, const int32_t* ids, int n, bt::Tensor& dst) {
    if (dst.rows != n || dst.cols != 1 || dst.dtype != bt::Dtype::INT32 || dst.device != w.dev)
        dst = bt::Tensor::empty_on(w.dev, n, 1, bt::Dtype::INT32);
    h2d(w.dev, dst.data, ids, static_cast<std::size_t>(n) * sizeof(int32_t));
}

bool LayerHooks::any_capture() const {
    if (!on_capture) return false;
    for (char c : capture)
        if (c) return true;
    return false;
}

bool LayerHooks::any_per_position() const {
    for (const auto& v : steer)
        for (const auto& s : v)
            if (s.per_position) return true;
    return false;
}

void to_host_f32(const ModelWeights& w, const bt::Tensor& t, std::vector<float>& out) {
    const std::size_t n = static_cast<std::size_t>(t.rows) * t.cols;
    out.resize(n);
    if (t.dtype == bt::Dtype::FP32) {
        t.copy_to_host(out.data());
        return;
    }
    (void)w;
    std::vector<std::uint16_t> raw(n);
    t.copy_to_host_bf16(raw.data());
    for (std::size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(raw[i]);
}

void apply_hooks(const ModelWeights& w, const LayerHooks* hooks, int l, bt::Tensor& x, int pos_row0,
                 int lo, int hi, int eval, bool capture) {
    if (!hooks) return;
    if (hooks->steers(l)) {
        for (const SteerDev& s : hooks->steer[l]) {
            if (!s.per_position) {
                bt::add_row_bias_inplace(x, s.delta);
                continue;
            }
            const int a = std::max(pos_row0 + lo, s.pos0);
            const int b = std::min(pos_row0 + hi, s.pos0 + s.delta.rows);
            if (a >= b) continue;
            bt::Tensor xs = rows_view(x, a - pos_row0, b - a);
            bt::add_inplace(xs, rows_view(s.delta, a - s.pos0, b - a));
        }
    }
    if (capture && hooks->captures(l) && hi > lo) {
        std::vector<float> host;
        to_host_f32(w, rows_view(x, lo, hi - lo), host);
        hooks->on_capture(l, pos_row0 + lo, hi - lo, eval, host.data());
    }
}

void ar_rows(const ModelWeights& w, RowScratch& sc, const int32_t* ids, int n, int pos0,
             std::vector<bt::Tensor>& K, std::vector<bt::Tensor>& V, bool kv_only,
             const LayerHooks* hooks, bool capture) {
    const Config& c = w.cfg;
    const int hd = c.head_dim, qd = c.n_q * hd, kd = c.n_kv * hd;
    if (pos0 + n > w.cos_tbl.rows) model_fail("position beyond the RoPE table");
    upload_ids(w, ids, n, sc.idx);
    bt::embedding_lookup_forward(w.embed, static_cast<const int32_t*>(sc.idx.data), n, sc.x);
    const bt::Tensor cos = rows_view(w.cos_tbl, pos0, n), sin = rows_view(w.sin_tbl, pos0, n);
    for (int l = 0; l < c.layers; ++l) {
        const PathW& p = w.layers[l].ar;
        bt::rms_norm_forward(sc.x, p.in_ln, c.eps, sc.n);
        lin(p.q, nullptr, sc.n, sc.q, &sc.ws);
        lin(p.k, nullptr, sc.n, sc.k, &sc.ws);
        lin(p.v, nullptr, sc.n, sc.v, &sc.ws);
        head_norm(w, sc.q, c.n_q, p.q_norm, sc.qn);
        head_norm(w, sc.k, c.n_kv, p.k_norm, sc.kn);
        bt::Tensor qv = reshape_view(sc.qn, n, qd), kv = reshape_view(sc.kn, n, kd);
        bt::rope_apply(qv, cos, sin, hd, c.n_q, qv);
        bt::rope_apply(kv, cos, sin, hd, c.n_kv, kv);
        bt::copy_d2d(kv, 0, K[l], pos0 * kd, n * kd);
        bt::copy_d2d(sc.v, 0, V[l], pos0 * kd, n * kd);
        if (kv_only && l == c.layers - 1) break;
        if (pos0 == 0) {
            const bt::Tensor Kv = rows_view(K[l], 0, n), Vv = rows_view(V[l], 0, n);
            bt::flash_attention_gqa_forward(qv, Kv, Vv, nullptr, c.n_q, c.n_kv, true, sc.att);
        } else {
            bt::flash_attention_decode(qv, K[l], V[l], pos0 + n, c.n_q, c.n_kv, sc.att);
        }
        lin(p.o, nullptr, sc.att, sc.x, &sc.ws, true);
        bt::rms_norm_forward(sc.x, p.post_ln, c.eps, sc.n);
        lin(p.gate, nullptr, sc.n, sc.g, &sc.ws);
        lin(p.up, nullptr, sc.n, sc.u, &sc.ws);
        bt::silu_forward(sc.g, sc.g);
        bt::mul_inplace(sc.g, sc.u);
        lin(p.down, nullptr, sc.g, sc.x, &sc.ws, true);
        apply_hooks(w, hooks, l, sc.x, pos0, 0, n, -1, capture);
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

Model::Model() : w_(std::make_unique<ModelWeights>()) {}
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

bool Model::loaded() const { return w_ && w_->loaded; }
const Config& Model::config() const { return w_->cfg; }
bt::Device Model::device() const { return w_->dev; }

void Model::load(const std::string& dir, bt::Device dev) {
    auto w = std::make_unique<ModelWeights>();
    w->cfg = Config::from_json(dir + "/config.json");
    w->dev = dev;
    w->dt = dev.is_cpu() ? bt::Dtype::FP32 : bt::Dtype::BF16;
    const Config& c = w->cfg;
    const int H = c.hidden, hd = c.head_dim, qd = c.n_q * hd, kd = c.n_kv * hd;
    const sf::File f = sf::File::open(dir + "/model.safetensors");
    bt::DeviceScope scope(dev);

    w->embed = load_w(f, *w, "model.embed_tokens.weight", c.vocab, H);
    w->lm_head = load_w(f, *w, "lm_head.weight", c.vocab, H);
    w->final_norm = load_w(f, *w, "model.norm.weight", H, 1);
    const auto pq = rope_perm(c.n_q, hd), pk = rope_perm(c.n_kv, hd), pn = rope_perm(1, hd);
    w->layers.resize(c.layers);
    for (int l = 0; l < c.layers; ++l) {
        const std::string b = "model.layers." + std::to_string(l) + ".";
        auto path = [&](PathW& p, const std::string& attn, const std::string& in_ln,
                        const std::string& post_ln, const std::string& mlp) {
            p.in_ln = load_w(f, *w, b + in_ln + ".weight", H, 1);
            p.q = load_w(f, *w, b + attn + ".q_proj.weight", qd, H, &pq);
            p.k = load_w(f, *w, b + attn + ".k_proj.weight", kd, H, &pk);
            p.v = load_w(f, *w, b + attn + ".v_proj.weight", kd, H);
            p.o = load_w(f, *w, b + attn + ".o_proj.weight", H, qd);
            p.q_norm = load_w(f, *w, b + attn + ".q_norm.weight", hd, 1, &pn);
            p.k_norm = load_w(f, *w, b + attn + ".k_norm.weight", hd, 1, &pn);
            p.post_ln = load_w(f, *w, b + post_ln + ".weight", H, 1);
            // gate and up stacked in one matrix: decode runs them as one
            // SwiGLU GEMV; prefill takes the two row views.
            {
                const bt::Tensor g = load_w(f, *w, b + mlp + ".gate_proj.weight", c.inter, H);
                const bt::Tensor u = load_w(f, *w, b + mlp + ".up_proj.weight", c.inter, H);
                p.gate_up = bt::Tensor::empty_on(w->dev, 2 * c.inter, H, w->dt);
                bt::copy_d2d(g, 0, p.gate_up, 0, c.inter * H);
                bt::copy_d2d(u, 0, p.gate_up, c.inter * H, c.inter * H);
            }
            p.gate = rows_view(p.gate_up, 0, c.inter);
            p.up = rows_view(p.gate_up, c.inter, c.inter);
            p.down = load_w(f, *w, b + mlp + ".down_proj.weight", H, c.inter);
        };
        path(w->layers[l].ar, "self_attn", "input_layernorm", "post_attention_layernorm", "mlp");
        path(w->layers[l].nar, "nar_self_attn", "nar_input_layernorm", "nar_pre_mlp_layernorm", "nar_mlp");
    }
    w->vae2llm_w = load_w(f, *w, "vae2llm.weight", H, c.latent);
    w->vae2llm_b = load_w(f, *w, "vae2llm.bias", H, 1);
    w->llm2vae_w = load_w(f, *w, "llm2vae.weight", c.latent, H);
    w->llm2vae_b = load_w(f, *w, "llm2vae.bias", c.latent, 1);
    w->t0_w = load_w(f, *w, "time_embedder.mlp.0.weight", H, 256);
    w->t0_b = load_w(f, *w, "time_embedder.mlp.0.bias", H, 1);
    w->t2_w = load_w(f, *w, "time_embedder.mlp.2.weight", H, H);
    w->t2_b = load_w(f, *w, "time_embedder.mlp.2.bias", H, 1);
    w->pe = load_w(f, *w, "latent_pos_embed.pe", c.max_latent_frames, H);

    // RoPE: inv_freq = 1 / base ** (arange(0, hd, 2) / hd) and angles =
    // pos * inv_freq in FP32 (RotaryEmbedding), cos/sin rounded to BF16.
    const int half = hd / 2;
    std::vector<float> inv(half);
    for (int i = 0; i < half; ++i)
        inv[i] = 1.0f / std::pow(c.rope_theta, static_cast<float>(2 * i) / static_cast<float>(hd));
    std::vector<float> cs(static_cast<std::size_t>(c.max_pos) * half), sn(cs.size());
    for (int p = 0; p < c.max_pos; ++p)
        for (int i = 0; i < half; ++i) {
            const float a = static_cast<float>(p) * inv[i];
            cs[static_cast<std::size_t>(p) * half + i] = bf16r(std::cos(a));
            sn[static_cast<std::size_t>(p) * half + i] = bf16r(std::sin(a));
        }
    w->cos_tbl = bt::Tensor::from_host_on(dev, cs.data(), c.max_pos, half);
    w->sin_tbl = bt::Tensor::from_host_on(dev, sn.data(), c.max_pos, half);
    if (!dev.is_cpu()) bt::sync(dev);
    w->loaded = true;
    w_ = std::move(w);
}

// ─── AR sessions ────────────────────────────────────────────────────────────

SteerDev Model::make_steer(const std::vector<float>& delta, float scale, int pos0, bool per_position) const {
    const ModelWeights& w = *w_;
    const int H = w.cfg.hidden;
    if (delta.empty() || delta.size() % H != 0) model_fail("steer delta must be a multiple of hidden_size floats");
    const int rows = static_cast<int>(delta.size() / H);
    if (!per_position && rows != 1) model_fail("a whole-sequence steer is one hidden_size vector");
    std::vector<float> scaled(delta.size());
    for (std::size_t i = 0; i < delta.size(); ++i) scaled[i] = delta[i] * scale;
    SteerDev s;
    s.pos0 = pos0;
    s.per_position = per_position;
    bt::Tensor f = bt::Tensor::from_host_on(w.dev, scaled.data(), rows, H);
    if (w.dt == bt::Dtype::FP32) s.delta = std::move(f);
    else bt::cast(f, s.delta, w.dt);
    return s;
}

std::unique_ptr<ArSession> Model::ar_session(int branches, int capacity, const LayerHooks* hooks) const {
    if (!loaded()) model_fail("not loaded");
    const ModelWeights& w = *w_;
    if (branches < 1 || branches > 2) model_fail("a session holds 1 or 2 branches");
    if (capacity < 1 || capacity > w.cfg.max_pos) model_fail("session capacity exceeds the context");
    auto s = std::make_unique<ArSession>();
    s->hooks = hooks;
    s->B = branches;
    s->cap = capacity;
    s->len.assign(branches, 0);
    const int kd = w.cfg.n_kv * w.cfg.head_dim;
    s->K.resize(branches);
    s->V.resize(branches);
    for (int b = 0; b < branches; ++b)
        for (int l = 0; l < w.cfg.layers; ++l) {
            s->K[b].push_back(bt::Tensor::zeros_on(w.dev, capacity, kd, w.dt));
            s->V[b].push_back(bt::Tensor::zeros_on(w.dev, capacity, kd, w.dt));
        }
    return s;
}

int Model::ar_len(const ArSession& s, int b) const { return s.len.at(b); }

namespace {

bt::Tensor lm_rows(const ModelWeights& w, int lo, int hi) {
    if (lo < 0 || hi > w.cfg.vocab || lo >= hi) model_fail("bad logit range");
    return rows_view(w.lm_head, lo, hi - lo);
}

}  // namespace

void Model::ar_prefill(ArSession& s, int b, const std::vector<int32_t>& ids, int lo, int hi) const {
    const ModelWeights& w = *w_;
    bt::DeviceScope scope(w.dev);
    const int n = static_cast<int>(ids.size());
    if (b < 0 || b >= s.B || s.len[b] != 0) model_fail("prefill needs an empty branch");
    if (n < 1 || n > s.cap) model_fail("prefix exceeds the session capacity");
    for (int32_t t : ids)
        if (t < 0 || t >= w.cfg.vocab) model_fail("token id outside the vocabulary");
    // Capture reports branch 0 only; steering acts on every branch.
    ar_rows(w, s.pre, ids.data(), n, 0, s.K[b], s.V[b], false, s.hooks, b == 0);
    bt::Tensor last = rows_view(s.pre.x, n - 1, 1);
    bt::Tensor normed, row;
    bt::rms_norm_forward(last, w.final_norm, w.cfg.eps, normed);
    lin(lm_rows(w, lo, hi), nullptr, normed, row, &s.pre.ws);
    if (s.lo != lo || s.hi != hi || s.logits.rows != s.B) {
        s.lo = lo;
        s.hi = hi;
        s.logits = bt::Tensor::zeros_on(w.dev, s.B, hi - lo, w.dt);
    }
    bt::copy_d2d(row, 0, s.logits, b * (hi - lo), hi - lo);
    s.len[b] = n;
}

void Model::ar_logits(ArSession& s, std::vector<float>& out) const {
    const ModelWeights& w = *w_;
    const std::size_t n = static_cast<std::size_t>(s.B) * (s.hi - s.lo);
    out.resize(n);
    if (w.dt == bt::Dtype::FP32) {
        s.logits.copy_to_host(out.data());
        return;
    }
    std::vector<std::uint16_t> raw(n);
    s.logits.copy_to_host_bf16(raw.data());
    for (std::size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(raw[i]);
}

void Model::ar_step(ArSession& s, int32_t token) const {
    const ModelWeights& w = *w_;
    const Config& c = w.cfg;
    bt::DeviceScope scope(w.dev);
    if (token < 0 || token >= c.vocab) model_fail("token id outside the vocabulary");
    if (s.hi <= s.lo) model_fail("step before prefill");
    const int B = s.B, hd = c.head_dim, half = hd / 2, qd = c.n_q * hd;
    for (int b = 0; b < B; ++b)
        if (s.len[b] < 1 || s.len[b] >= s.cap) model_fail("decode beyond the session capacity");
    auto& st = s.st;
    if (st.mask.empty()) {
        st.tok = bt::Tensor::empty_on(w.dev, B, 1, bt::Dtype::INT32);
        st.rows = bt::Tensor::empty_on(w.dev, B, 1, bt::Dtype::INT32);
        for (int b = 0; b < B; ++b) {
            st.row1.push_back(rows_view(st.rows, b, 1));
            st.mask.push_back(bt::Tensor::zeros_on(w.dev, s.cap, 1, bt::Dtype::FP32));
        }
        st.mask_len.assign(B, 0);
        const float one = 1.0f;
        st.ones = bt::Tensor::from_host_on(w.dev, &one, 1, 1);
        st.cos = bt::Tensor::zeros_on(w.dev, B, half, bt::Dtype::FP32);
        st.sin = bt::Tensor::zeros_on(w.dev, B, half, bt::Dtype::FP32);
        st.att = bt::Tensor::zeros_on(w.dev, B, qd, w.dt);
    }
    // Stage this step: token, cache rows, the valid-key masks, RoPE rows.
    std::vector<int32_t> toks(B, token), rows(B);
    for (int b = 0; b < B; ++b) rows[b] = s.len[b];
    h2d(w.dev, st.tok.data, toks.data(), B * sizeof(int32_t));
    h2d(w.dev, st.rows.data, rows.data(), B * sizeof(int32_t));
    for (int b = 0; b < B; ++b) {
        const int pos = s.len[b];
        if (st.mask_len[b] != pos) {
            std::vector<float> m(s.cap, 0.0f);
            std::fill(m.begin(), m.begin() + pos, 1.0f);
            h2d(w.dev, st.mask[b].data, m.data(), m.size() * sizeof(float));
        }
        bt::copy_d2d(st.ones, 0, st.mask[b], pos, 1);
        bt::copy_d2d(w.cos_tbl, pos * half, st.cos, b * half, half);
        bt::copy_d2d(w.sin_tbl, pos * half, st.sin, b * half, half);
    }

    const bt::Tensor lm = lm_rows(w, s.lo, s.hi);
    auto body = [&]() {
        const int kd = c.n_kv * hd;
        bt::embedding_lookup_forward(w.embed, static_cast<const int32_t*>(st.tok.data), B, st.x);
        bt::rms_norm_forward(st.x, w.layers[0].ar.in_ln, c.eps, st.n);
        for (int l = 0; l < c.layers; ++l) {
            const PathW& p = w.layers[l].ar;
            lin(p.q, nullptr, st.n, st.q, &st.ws);
            lin(p.k, nullptr, st.n, st.k, &st.ws);
            lin(p.v, nullptr, st.n, st.v, &st.ws);
            head_norm(w, st.q, c.n_q, p.q_norm, st.qn);
            head_norm(w, st.k, c.n_kv, p.k_norm, st.kn);
            bt::Tensor qv = reshape_view(st.qn, B, qd), kv = reshape_view(st.kn, B, kd);
            bt::rope_apply(qv, st.cos, st.sin, hd, c.n_q, qv);
            bt::rope_apply(kv, st.cos, st.sin, hd, c.n_kv, kv);
            for (int b = 0; b < B; ++b) {
                bt::scatter_rows(rows_view(kv, b, 1), st.row1[b], s.K[b][l]);
                bt::scatter_rows(rows_view(st.v, b, 1), st.row1[b], s.V[b][l]);
                bt::Tensor ab = rows_view(st.att, b, 1);
                bt::flash_attention_decode_masked(rows_view(qv, b, 1), s.K[b][l], s.V[b][l],
                                                  static_cast<const float*>(st.mask[b].data), c.n_q, c.n_kv,
                                                  ab);
            }
            lin(p.o, nullptr, st.att, st.x, &st.ws, true);
            bt::rms_norm_forward(st.x, p.post_ln, c.eps, st.n);
            // silu(gate) * up in the GEMV's FP32 epilogue (one BF16 rounding
            // where the reference rounds gate, silu and the product).
            bt::linear_forward_batched_ex(p.gate_up, nullptr, st.n, bt::kLinearActNone, bt::kLinearEpiSwiglu,
                                          &st.ws, st.g);
            lin(p.down, nullptr, st.g, st.x, &st.ws, true);
            if (s.hooks)
                for (int b = 0; b < B; ++b) {
                    bt::Tensor xb = rows_view(st.x, b, 1);
                    apply_hooks(w, s.hooks, l, xb, s.len[b], 0, 1, -1, b == 0);
                }
            const bt::Tensor& next = l + 1 < c.layers ? w.layers[l + 1].ar.in_ln : w.final_norm;
            bt::rms_norm_forward(st.x, next, c.eps, st.n);
        }
        lin(lm, nullptr, st.n, st.logits, &st.ws);
    };

    static const bool no_graph = [] {
        const char* v = std::getenv("BROSOUNDML_YUE2_NO_GRAPH");
        return v && *v && *v != '0';
    }();
    const bool hooks_eager = s.hooks && (s.hooks->any_capture() || s.hooks->any_per_position());
    const bool graph = !no_graph && !hooks_eager && bt::graph_capture_available(w.dev);
    if (!graph) {
        body();
    } else if (!st.captured || st.graph_lo != s.lo || st.graph_hi != s.hi) {
        body();  // warm-up: sizes every buffer and computes this step
        bt::sync_all();
        {
            bt::CudaGraphCapture cap;
            body();
            st.graph = cap.finish();
        }
        st.captured = true;
        st.graph_lo = s.lo;
        st.graph_hi = s.hi;
    } else {
        st.graph.launch();
    }
    if (s.logits.rows != B || s.logits.cols != s.hi - s.lo)
        s.logits = bt::Tensor::zeros_on(w.dev, B, s.hi - s.lo, w.dt);
    bt::copy_d2d(st.logits, 0, s.logits, 0, B * (s.hi - s.lo));
    for (int b = 0; b < B; ++b) {
        st.mask_len[b] = s.len[b] + 1;
        ++s.len[b];
    }
}

}  // namespace brosoundml::yue2
