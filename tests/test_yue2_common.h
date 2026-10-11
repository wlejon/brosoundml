#pragma once
// Shared helpers of the YuE2 backbone tests: CHECK, the fixture reader
// (tests/ref/gen_yue2_fixture.py's binary conventions), the model folder.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace yue2_test {

namespace fs = std::filesystem;

inline int failures = 0;

#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::fprintf(stderr, "FAIL: " __VA_ARGS__);   \
            std::fprintf(stderr, "\n");                   \
            ++::yue2_test::failures;                      \
        }                                                 \
    } while (0)

inline fs::path repo() { return fs::path(BROSOUNDML_REPO_DIR); }
inline fs::path fixture(const char* name) { return repo() / "tests" / "fixtures" / name; }

// $YUE2_MODEL_DIR, else ../YuE2-3B beside the repo; empty when absent.
inline fs::path model_dir() {
    if (const char* e = std::getenv("YUE2_MODEL_DIR")) return fs::path(e);
    const fs::path p = repo().parent_path() / "YuE2-3B";
    return fs::exists(p / "model.safetensors") ? p : fs::path();
}

struct Reader {
    std::ifstream f;
    explicit Reader(const fs::path& p) : f(p.string(), std::ios::binary) {}
    explicit operator bool() const { return static_cast<bool>(f); }
    int32_t i32() { int32_t v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; }
    uint64_t u64() { uint64_t v = 0; f.read(reinterpret_cast<char*>(&v), 8); return v; }
    float f32() { float v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; }
    double f64() { double v = 0; f.read(reinterpret_cast<char*>(&v), 8); return v; }
    std::string str() {
        const int n = i32();
        std::string s(static_cast<std::size_t>(n), '\0');
        f.read(s.data(), n);
        return s;
    }
    std::vector<int32_t> i32list() {
        std::vector<int32_t> v(static_cast<std::size_t>(i32()));
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
        return v;
    }
    std::vector<float> f32arr(std::size_t n) {
        std::vector<float> v(n);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
        return v;
    }
};

// A little-endian C-order .npy of '<f4' or '<i4' (what the reference CLI
// writes); `shape` receives the dimensions. Empty on failure.
template <typename T>
inline std::vector<T> read_npy(const fs::path& p, std::vector<long long>* shape = nullptr) {
    std::ifstream f(p.string(), std::ios::binary);
    if (!f) return {};
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic, 6) != "\x93NUMPY") return {};
    std::size_t hl = 0;
    if (magic[6] == 1) {
        uint16_t h = 0;
        f.read(reinterpret_cast<char*>(&h), 2);
        hl = h;
    } else {
        uint32_t h = 0;
        f.read(reinterpret_cast<char*>(&h), 4);
        hl = h;
    }
    std::string hdr(hl, '\0');
    f.read(hdr.data(), static_cast<std::streamsize>(hl));
    const char* want = sizeof(T) == 4 && std::is_floating_point_v<T> ? "<f4" : "<i4";
    if (hdr.find(want) == std::string::npos || hdr.find("'fortran_order': True") != std::string::npos) return {};
    std::vector<long long> dims;
    const auto a = hdr.find('(', hdr.find("shape")), b = hdr.find(')', a);
    std::size_t n = 1;
    for (std::size_t i = a + 1; i < b;) {
        while (i < b && (hdr[i] == ' ' || hdr[i] == ',')) ++i;
        if (i >= b) break;
        const long long d = std::strtoll(hdr.c_str() + i, nullptr, 10);
        dims.push_back(d);
        n *= static_cast<std::size_t>(d);
        while (i < b && hdr[i] != ',') ++i;
    }
    if (shape) *shape = dims;
    std::vector<T> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
    if (!f) return {};
    return v;
}

// The parts of tests/fixtures/yue2_model.bin the quant / spec tests use: the
// logit prefixes with the reference rows, and the prefixes of the two
// single-branch greedy streams ([0] ABC, [1] semantic).
struct Fixture {
    std::vector<std::vector<int32_t>> prefixes;
    std::vector<std::vector<float>> ref_logits;
    std::vector<int32_t> stream_prefix[2];
};

inline bool read_fixture(Fixture& fx, int V) {
    Reader r(fixture("yue2_model.bin"));
    if (!r || r.i32() != 0x314d5559) return false;
    const int nl = r.i32();
    for (int i = 0; i < nl; ++i) {
        fx.prefixes.push_back(r.i32list());
        fx.ref_logits.push_back(r.f32arr(static_cast<std::size_t>(V)));
    }
    for (int k = 0; k < 3; ++k) {
        const bool abc = r.i32() == 0;
        auto prefix = r.i32list();
        r.i32list();                              // negative prefix
        const float cfg = r.f32();
        r.i32();                                  // legacy
        for (int i = 0; i < 7; ++i) r.i32();      // sampling (7 words)
        r.i32list();                              // reference tokens
        r.i32();                                  // truncated
        if (cfg == 1.0f) fx.stream_prefix[abc ? 0 : 1] = std::move(prefix);
    }
    return !fx.stream_prefix[0].empty() && !fx.stream_prefix[1].empty();
}

// Shape statistics of a sampled ABC score: header fields (X T M L K Q, each
// counted once), bar lines, voice lines.
struct PlanStats {
    int tokens = 0, truncated = 0, headers = 0, bars = 0, voices = 0;
    double tps = 0;
};
inline PlanStats plan_stats(const std::string& abc, int tokens, bool truncated, double tps) {
    PlanStats s;
    s.tokens = tokens;
    s.truncated = truncated;
    s.tps = tps;
    std::istringstream in(abc);
    std::string line;
    std::vector<char> seen;
    while (std::getline(in, line)) {
        if (line.size() >= 2 && line[1] == ':' && std::string("XTMLKQ").find(line[0]) != std::string::npos &&
            std::find(seen.begin(), seen.end(), line[0]) == seen.end()) {
            seen.push_back(line[0]);
            ++s.headers;
        }
        if (line.rfind("V:", 0) == 0) ++s.voices;
        s.bars += static_cast<int>(std::count(line.begin(), line.end(), '|'));
    }
    return s;
}

struct Diff {
    double max_abs = 0, max_rel = 0, rms = 0, ref_rms = 0;
    double cosine = 0;
};
inline Diff diff(const std::vector<float>& a, const std::vector<float>& b) {
    Diff d;
    double dot = 0, na = 0, nb = 0, se = 0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double e = std::fabs(static_cast<double>(a[i]) - b[i]);
        d.max_abs = std::max(d.max_abs, e);
        se += e * e;
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    d.rms = n ? std::sqrt(se / n) : 0;
    d.ref_rms = n ? std::sqrt(nb / n) : 0;
    d.max_rel = d.ref_rms > 0 ? d.max_abs / d.ref_rms : 0;
    d.cosine = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 1;
    return d;
}

}  // namespace yue2_test
