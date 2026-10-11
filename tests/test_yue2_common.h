#pragma once
// Shared helpers of the YuE2 backbone tests: CHECK, the fixture reader
// (tests/ref/gen_yue2_fixture.py's binary conventions), the model folder.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
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
