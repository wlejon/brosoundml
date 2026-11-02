#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "bc_resnet2d_internal.h"

namespace brosoundml {
namespace {
// ─── Binary format ('BWK2') ─────────────────────────────────────────────────
constexpr std::uint32_t kMagic   = 0x324B5742u;   // 'B''W''K''2'
constexpr std::uint32_t kVersion = 1u;

struct Reader {
    std::FILE* fp = nullptr; std::string path;
    ~Reader() { if (fp) std::fclose(fp); }
    void open(const std::string& p) {
        path = p; fp = std::fopen(p.c_str(), "rb");
        if (!fp) fail("bc_resnet2d::load", "could not open '" + p + "'");
    }
    void rd(void* d, std::size_t n) {
        if (std::fread(d, 1, n, fp) != n)
            fail("bc_resnet2d::load", "unexpected EOF in '" + path + "'");
    }
    std::uint32_t u32() { std::uint32_t v; rd(&v, 4); return v; }
    std::uint16_t u16() { std::uint16_t v; rd(&v, 2); return v; }
    std::uint8_t  u8 () { std::uint8_t  v; rd(&v, 1); return v; }
    float         f32() { float v; rd(&v, 4); return v; }
    std::string   str(std::size_t n) { std::string s(n, '\0'); rd(s.data(), n); return s; }
};
struct Writer {
    std::FILE* fp = nullptr; std::string path;
    ~Writer() { if (fp) std::fclose(fp); }
    void open(const std::string& p) {
        path = p; fp = std::fopen(p.c_str(), "wb");
        if (!fp) fail("bc_resnet2d::save", "could not open '" + p + "'");
    }
    void wr(const void* s, std::size_t n) {
        if (std::fwrite(s, 1, n, fp) != n)
            fail("bc_resnet2d::save", "short write to '" + path + "'");
    }
    void u32(std::uint32_t v) { wr(&v, 4); }
    void u16(std::uint16_t v) { wr(&v, 2); }
    void u8 (std::uint8_t  v) { wr(&v, 1); }
    void f32(float v)         { wr(&v, 4); }
};

void write_tensor(Writer& w, const std::string& name, const bt::Tensor& t) {
    bt::Tensor host = t.to(bt::Device::CPU);
    if (host.dtype != bt::Dtype::FP32) fail("bc_resnet2d::save", "non-FP32 tensor");
    w.u16(static_cast<std::uint16_t>(name.size()));
    w.wr(name.data(), name.size());
    w.u32(static_cast<std::uint32_t>(host.rows));
    w.u32(static_cast<std::uint32_t>(host.cols));
    w.wr(host.host_f32(), static_cast<std::size_t>(host.size()) * sizeof(float));
}

bt::Tensor read_tensor(Reader& r, const std::string& want, int rows, int cols,
                       bt::Device dev) {
    const std::uint16_t nl = r.u16();
    const std::string name = r.str(nl);
    const int rr = static_cast<int>(r.u32());
    const int cc = static_cast<int>(r.u32());
    if (name != want)
        fail("bc_resnet2d::load", "expected '" + want + "' got '" + name + "'");
    if (rr != rows || cc != cols)
        fail("bc_resnet2d::load", "tensor '" + name + "' shape mismatch");
    std::vector<float> buf(static_cast<std::size_t>(rr) * cc);
    r.rd(buf.data(), buf.size() * sizeof(float));
    return bt::Tensor::from_host_on(bt::Device::CPU, buf.data(), rr, cc).to(dev);
}

}  // namespace

// ─── Public surface ───────────────────────────────────────────────────────────

BcResnet2d::BcResnet2d() : impl_(std::make_unique<Impl>()) {}
BcResnet2d::~BcResnet2d() = default;
BcResnet2d::BcResnet2d(BcResnet2d&&) noexcept = default;
BcResnet2d& BcResnet2d::operator=(BcResnet2d&&) noexcept = default;

BcResnet2d BcResnet2d::make(const BcResnet2dConfig& cfg, bt::Device device) {
    BcResnet2d m;
    m.impl_->cfg = cfg;
    m.impl_->device = device;
    m.impl_->fused = false;
    m.impl_->build();
    return m;
}

const BcResnet2dConfig& BcResnet2d::config() const { return impl_->cfg; }
bt::Device              BcResnet2d::device() const { return impl_->device; }
bool                    BcResnet2d::fused()  const { return impl_->fused; }
int BcResnet2d::receptive_field_frames() const {
    return impl_->rf_frames() + impl_->gap_cap - 1;
}

int BcResnet2d::gap_window_frames() const { return impl_->gap_cap; }

int BcResnet2d::param_count() const {
    int n = 0;
    auto conv = [](const Conv2dLayer& c) { return c.Cout * (c.Cin / c.groups) * c.kH * c.kW + c.Cout; };
    auto bn = [&](const BatchNorm2d& b) { return impl_->fused ? 0 : 2 * b.channels; };
    n += conv(impl_->stem) + bn(impl_->bn_stem);
    for (auto& blk : impl_->blocks) {
        if (blk.is_transition) { n += conv(blk.pw_in) + bn(blk.bn_in); n += conv(blk.res) + bn(blk.bn_res); }
        n += conv(blk.f2) + bn(blk.bn_f2);
        n += conv(blk.f1) + bn(blk.bn_f1) + conv(blk.pw_f1);
    }
    n += impl_->c_last + 1;
    return n;
}

void BcResnet2d::forward(const bt::Tensor& feats, bt::Tensor& out_logit_per_frame) const {
    impl_->forward_core(feats, /*state=*/nullptr, out_logit_per_frame);
}

// ── Multi-stream session API ──
BcResnet2dSession BcResnet2d::make_session() const {
    return impl_->alloc_state();
}

void BcResnet2d::forward_streaming(BcResnet2dSession& state,
                                   const bt::Tensor& new_feats,
                                   bt::Tensor& out_logit_per_frame) const {
    impl_->check_state(state);
    impl_->forward_core(new_feats, &state, out_logit_per_frame);
}

void BcResnet2d::reset(BcResnet2dSession& state) const {
    Impl::zero_state(state);
}

// ── Legacy single-stream API over the internally-owned session ──
void BcResnet2d::reset_streaming_state() { Impl::zero_state(impl_->owned_state); }

void BcResnet2d::forward_streaming(const bt::Tensor& new_feats, bt::Tensor& out) {
    impl_->forward_core(new_feats, &impl_->owned_state, out);
}

// ─── BN fold ──────────────────────────────────────────────────────────────────
namespace {
void fuse_one(Conv2dLayer& c, BatchNorm2d& bn) {
    if (!bn.present) return;
    const bt::Device tgt = c.W.device;
    bt::Tensor Wh = c.W.to(bt::Device::CPU), bh = c.b.to(bt::Device::CPU);
    bt::Tensor gh = bn.gamma.to(bt::Device::CPU), eh = bn.beta.to(bt::Device::CPU);
    bt::Tensor mh = bn.mean.to(bt::Device::CPU), vh = bn.var.to(bt::Device::CPU);
    const int C = c.Cout, row = Wh.cols;
    float* W = Wh.host_f32_mut(); float* B = bh.host_f32_mut();
    const float* g = gh.host_f32(); const float* be = eh.host_f32();
    const float* m = mh.host_f32(); const float* v = vh.host_f32();
    for (int co = 0; co < C; ++co) {
        const float a = g[co] / std::sqrt(v[co] + bn.eps);
        for (int j = 0; j < row; ++j) W[co * row + j] *= a;
        B[co] = (B[co] - m[co]) * a + be[co];
    }
    c.W = Wh.to(tgt); c.b = bh.to(tgt);
    bn.present = false;
    bn.gamma = bt::Tensor{}; bn.beta = bt::Tensor{};
    bn.mean = bt::Tensor{}; bn.var = bt::Tensor{};
}
}  // namespace

void BcResnet2d::fuse_bn() {
    if (impl_->fused) return;
    fuse_one(impl_->stem, impl_->bn_stem);
    for (auto& blk : impl_->blocks) {
        if (blk.is_transition) { fuse_one(blk.pw_in, blk.bn_in); fuse_one(blk.res, blk.bn_res); }
        fuse_one(blk.f2, blk.bn_f2);
        fuse_one(blk.f1, blk.bn_f1);   // pw_f1 has no BN
    }
    impl_->fused = true;
}

// ─── Weights I/O ────────────────────────────────────────────────────────────
namespace {
void for_each_tensor(BcResnet2d::Impl& m, bool include_bn,
                     std::vector<std::pair<std::string, bt::Tensor*>>& out) {
    auto add = [&](const std::string& n, bt::Tensor& t) { out.push_back({n, &t}); };
    auto add_conv = [&](const std::string& p, Conv2dLayer& c) {
        add(p + ".W", c.W); add(p + ".b", c.b);
    };
    auto add_bn = [&](const std::string& p, BatchNorm2d& bn) {
        if (!include_bn || !bn.present) return;
        add(p + ".gamma", bn.gamma); add(p + ".beta", bn.beta);
        add(p + ".mean", bn.mean);   add(p + ".var", bn.var);
    };
    add_conv("stem", m.stem); add_bn("bn_stem", m.bn_stem);
    for (std::size_t i = 0; i < m.blocks.size(); ++i) {
        BCBlock& blk = m.blocks[i];
        const std::string p = "blk" + std::to_string(i);
        if (blk.is_transition) { add_conv(p + ".pw_in", blk.pw_in); add_bn(p + ".bn_in", blk.bn_in); }
        add_conv(p + ".f2", blk.f2); add_bn(p + ".bn_f2", blk.bn_f2);
        add_conv(p + ".f1", blk.f1); add_bn(p + ".bn_f1", blk.bn_f1);
        add_conv(p + ".pw_f1", blk.pw_f1);
        if (blk.is_transition) { add_conv(p + ".res", blk.res); add_bn(p + ".bn_res", blk.bn_res); }
    }
    add("head.W", m.head_W); add("head.b", m.head_b);
}
}  // namespace

void BcResnet2d::save(const std::string& path, bool fused) const {
    if (fused && !impl_->fused) fail("BcResnet2d::save", "fuse_bn() has not run");
    if (!fused && impl_->fused) fail("BcResnet2d::save", "BN already fused");
    Writer w; w.open(path);
    w.u32(kMagic); w.u32(kVersion);
    const auto& c = impl_->cfg;
    w.u32(c.n_mels); w.u32(c.c_stem); w.u32(c.stem_kf); w.u32(c.stem_kt); w.u32(c.stem_sf);
    for (int s = 0; s < BcResnet2dConfig::n_stages; ++s) {
        w.u32(c.c[s]); w.u32(c.fstride[s]); w.u32(c.tdil[s]); w.u32(c.n_blocks[s]);
    }
    w.u32(c.dw_kf); w.u32(c.dw_kt); w.u32(c.ssn_subbands); w.f32(c.bn_eps);
    w.u8(fused ? 1u : 0u);
    std::vector<std::pair<std::string, bt::Tensor*>> ts;
    for_each_tensor(*impl_, /*include_bn=*/!fused, ts);
    w.u32(static_cast<std::uint32_t>(ts.size()));
    for (auto& [n, t] : ts) write_tensor(w, n, *t);
}

BcResnet2d BcResnet2d::load(const std::string& path, bt::Device device) {
    Reader r; r.open(path);
    if (r.u32() != kMagic)   fail("BcResnet2d::load", "bad magic");
    if (r.u32() != kVersion) fail("BcResnet2d::load", "unsupported version");
    BcResnet2dConfig c;
    c.n_mels = (int)r.u32(); c.c_stem = (int)r.u32(); c.stem_kf = (int)r.u32();
    c.stem_kt = (int)r.u32(); c.stem_sf = (int)r.u32();
    for (int s = 0; s < BcResnet2dConfig::n_stages; ++s) {
        c.c[s] = (int)r.u32(); c.fstride[s] = (int)r.u32();
        c.tdil[s] = (int)r.u32(); c.n_blocks[s] = (int)r.u32();
    }
    c.dw_kf = (int)r.u32(); c.dw_kt = (int)r.u32(); c.ssn_subbands = (int)r.u32();
    c.bn_eps = r.f32();
    const bool on_disk_fused = (r.u8() != 0u);
    const std::uint32_t num = r.u32();

    BcResnet2d m = BcResnet2d::make(c, device);
    m.impl_->fused = on_disk_fused;
    if (on_disk_fused) {
        auto clr = [](BatchNorm2d& bn) {
            bn.present = false; bn.gamma = bt::Tensor{}; bn.beta = bt::Tensor{};
            bn.mean = bt::Tensor{}; bn.var = bt::Tensor{};
        };
        clr(m.impl_->bn_stem);
        for (auto& blk : m.impl_->blocks) {
            if (blk.is_transition) { clr(blk.bn_in); clr(blk.bn_res); }
            clr(blk.bn_f2); clr(blk.bn_f1);
        }
    }
    std::vector<std::pair<std::string, bt::Tensor*>> ts;
    for_each_tensor(*m.impl_, /*include_bn=*/!on_disk_fused, ts);
    if (ts.size() != num)
        fail("BcResnet2d::load", "tensor count mismatch");
    for (auto& [n, t] : ts) *t = read_tensor(r, n, t->rows, t->cols, device);
    return m;
}


}  // namespace brosoundml
