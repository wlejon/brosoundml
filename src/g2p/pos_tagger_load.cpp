// POS tagger weight file loader. See docs/pos_tagger.md for the binary format.

#include "pos_tagger_internal.h"
#include "brosoundml/modules.h"

#include <brotensor/tensor.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace brosoundml::g2p {

namespace bt = brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: pos_tagger: " + msg);
}

struct Reader {
    std::FILE* fp = nullptr;
    std::string path;

    ~Reader() { if (fp) std::fclose(fp); }

    void open(const std::string& p) {
        path = p;
        fp = std::fopen(p.c_str(), "rb");
        if (!fp) fail("could not open weights file '" + p + "'");
    }

    void read_bytes(void* dst, std::size_t n) {
        if (std::fread(dst, 1, n, fp) != n) {
            fail("unexpected EOF reading " + std::to_string(n) +
                 " bytes from '" + path + "'");
        }
    }
    std::uint32_t u32() { std::uint32_t v; read_bytes(&v, 4); return v; }
    std::uint16_t u16() { std::uint16_t v; read_bytes(&v, 2); return v; }
    std::uint8_t  u8 () { std::uint8_t  v; read_bytes(&v, 1); return v; }

    std::string str(std::size_t n) {
        std::string s(n, '\0');
        read_bytes(s.data(), n);
        return s;
    }
};

bt::Tensor read_tensor_payload(Reader& r, int rows, int cols) {
    std::vector<float> buf(static_cast<std::size_t>(rows) * cols);
    r.read_bytes(buf.data(), buf.size() * sizeof(float));
    return bt::Tensor::from_host_on(bt::Device::CPU, buf.data(), rows, cols);
}

// Construct the canonical tensor-name set the trainer will write. Chunk 2
// must emit exactly these names.
struct ExpectedTensor {
    std::string name;
    int rows;
    int cols;
};

std::vector<ExpectedTensor> expected_tensors(const PosWeights& w) {
    std::vector<ExpectedTensor> v;
    v.push_back({"token_emb", kVocab, w.d_model});
    v.push_back({"pos_emb",   w.max_seq_len, w.d_model});
    for (int i = 0; i < w.num_layers; ++i) {
        const std::string p = "layer" + std::to_string(i) + ".";
        v.push_back({p + "ln1.gamma",  w.d_model, 1});
        v.push_back({p + "ln1.beta",   w.d_model, 1});
        v.push_back({p + "attn.Wq",    w.d_model, w.d_model});
        v.push_back({p + "attn.bq",    w.d_model, 1});
        v.push_back({p + "attn.Wk",    w.d_model, w.d_model});
        v.push_back({p + "attn.bk",    w.d_model, 1});
        v.push_back({p + "attn.Wv",    w.d_model, w.d_model});
        v.push_back({p + "attn.bv",    w.d_model, 1});
        v.push_back({p + "attn.Wo",    w.d_model, w.d_model});
        v.push_back({p + "attn.bo",    w.d_model, 1});
        v.push_back({p + "ln2.gamma",  w.d_model, 1});
        v.push_back({p + "ln2.beta",   w.d_model, 1});
        v.push_back({p + "ffn1.W",     w.ffn_hidden, w.d_model});
        v.push_back({p + "ffn1.b",     w.ffn_hidden, 1});
        v.push_back({p + "ffn2.W",     w.d_model, w.ffn_hidden});
        v.push_back({p + "ffn2.b",     w.d_model, 1});
    }
    v.push_back({"final_ln.gamma", w.d_model, 1});
    v.push_back({"final_ln.beta",  w.d_model, 1});
    v.push_back({"head.W",         w.num_tags, w.d_model});
    v.push_back({"head.b",         w.num_tags, 1});
    return v;
}

// Bind a loaded tensor to the right slot in PosWeights by name.
void bind(PosWeights& w, const std::string& name, bt::Tensor&& t) {
    if (name == "token_emb") { w.token_emb = std::move(t); return; }
    if (name == "pos_emb")   { w.pos_emb   = std::move(t); return; }
    if (name == "final_ln.gamma") { w.final_ln.gamma = std::move(t); w.final_ln.features = w.d_model; return; }
    if (name == "final_ln.beta")  { w.final_ln.beta  = std::move(t); return; }
    if (name == "head.W") { w.head.W = std::move(t); w.head.in_features = w.d_model; w.head.out_features = w.num_tags; return; }
    if (name == "head.b") { w.head.b = std::move(t); return; }

    // layerN.<part>
    if (name.rfind("layer", 0) == 0) {
        const auto dot = name.find('.');
        if (dot == std::string::npos) fail("malformed tensor name '" + name + "'");
        const int idx = std::stoi(name.substr(5, dot - 5));
        if (idx < 0 || idx >= static_cast<int>(w.layers.size())) {
            fail("tensor '" + name + "' references out-of-range layer " + std::to_string(idx));
        }
        auto& L = w.layers[idx];
        const std::string sub = name.substr(dot + 1);

        if (sub == "ln1.gamma") { L.ln1.gamma = std::move(t); L.ln1.features = w.d_model; return; }
        if (sub == "ln1.beta")  { L.ln1.beta  = std::move(t); return; }
        if (sub == "ln2.gamma") { L.ln2.gamma = std::move(t); L.ln2.features = w.d_model; return; }
        if (sub == "ln2.beta")  { L.ln2.beta  = std::move(t); return; }
        if (sub == "attn.Wq")   { L.mha.Wq = std::move(t); L.mha.embed_dim = w.d_model; L.mha.num_heads = w.num_heads; return; }
        if (sub == "attn.bq")   { L.mha.bq = std::move(t); return; }
        if (sub == "attn.Wk")   { L.mha.Wk = std::move(t); return; }
        if (sub == "attn.bk")   { L.mha.bk = std::move(t); return; }
        if (sub == "attn.Wv")   { L.mha.Wv = std::move(t); return; }
        if (sub == "attn.bv")   { L.mha.bv = std::move(t); return; }
        if (sub == "attn.Wo")   { L.mha.Wo = std::move(t); return; }
        if (sub == "attn.bo")   { L.mha.bo = std::move(t); return; }
        if (sub == "ffn1.W")    { L.ffn1.W = std::move(t); L.ffn1.in_features = w.d_model; L.ffn1.out_features = w.ffn_hidden; return; }
        if (sub == "ffn1.b")    { L.ffn1.b = std::move(t); return; }
        if (sub == "ffn2.W")    { L.ffn2.W = std::move(t); L.ffn2.in_features = w.ffn_hidden; L.ffn2.out_features = w.d_model; return; }
        if (sub == "ffn2.b")    { L.ffn2.b = std::move(t); return; }
    }
    fail("unknown tensor name '" + name + "'");
}

}  // namespace

void load_pos_weights(const std::string& path, PosWeights& w) {
    Reader r;
    r.open(path);

    const std::uint32_t magic   = r.u32();
    const std::uint32_t version = r.u32();
    if (magic != 0x504F5302u) fail("bad magic 0x" + std::to_string(magic));
    if (version != 1u)        fail("unsupported version " + std::to_string(version));

    const std::uint32_t num_tags    = r.u32();
    const std::uint32_t d_model     = r.u32();
    const std::uint32_t num_layers  = r.u32();
    const std::uint32_t num_heads   = r.u32();
    const std::uint32_t ffn_hidden  = r.u32();
    const std::uint32_t max_seq_len = r.u32();
    const std::uint32_t num_tensors = r.u32();

    if (static_cast<int>(num_tags)    != NUM_TAGS)     fail("num_tags mismatch");
    if (static_cast<int>(d_model)     != kDModel)      fail("d_model mismatch");
    if (static_cast<int>(num_layers)  != kNumLayers)   fail("num_layers mismatch");
    if (static_cast<int>(num_heads)   != kNumHeads)    fail("num_heads mismatch");
    if (static_cast<int>(ffn_hidden)  != kFFN)         fail("ffn_hidden mismatch");
    if (static_cast<int>(max_seq_len) != kMaxSeqLen)   fail("max_seq_len mismatch");

    w.num_tags    = NUM_TAGS;
    w.d_model     = kDModel;
    w.num_layers  = kNumLayers;
    w.num_heads   = kNumHeads;
    w.ffn_hidden  = kFFN;
    w.max_seq_len = kMaxSeqLen;
    w.layers.clear();
    w.layers.resize(kNumLayers);

    // Build expected tensor manifest and confirm every named tensor appears
    // exactly once with the matching shape.
    const auto expected = expected_tensors(w);
    std::unordered_map<std::string, ExpectedTensor> by_name;
    for (const auto& e : expected) by_name.emplace(e.name, e);

    std::unordered_map<std::string, bool> seen;
    for (std::uint32_t i = 0; i < num_tensors; ++i) {
        const std::uint16_t name_len = r.u16();
        const std::string name = r.str(name_len);
        const std::uint8_t rank = r.u8();
        if (rank != 1 && rank != 2) fail("tensor '" + name + "' bad rank " + std::to_string(rank));
        int rows = static_cast<int>(r.u32());
        int cols = (rank == 2) ? static_cast<int>(r.u32()) : 1;

        const auto it = by_name.find(name);
        if (it == by_name.end()) fail("unexpected tensor '" + name + "'");
        if (it->second.rows != rows || it->second.cols != cols) {
            fail("tensor '" + name + "' shape (" + std::to_string(rows) + "," +
                 std::to_string(cols) + ") != expected (" +
                 std::to_string(it->second.rows) + "," +
                 std::to_string(it->second.cols) + ")");
        }
        if (seen[name]) fail("duplicate tensor '" + name + "'");
        seen[name] = true;

        auto t = read_tensor_payload(r, rows, cols);
        bind(w, name, std::move(t));
    }

    for (const auto& e : expected) {
        if (!seen[e.name]) fail("missing tensor '" + e.name + "'");
    }
}

}  // namespace brosoundml::g2p
