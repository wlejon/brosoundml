#include "brosoundml/supertonic.h"

#include "supertonic_internal.h"

#include "brotensor/json.h"

#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt = brotensor;
namespace sf = brotensor::safetensors;
namespace j  = brotensor::json;
namespace fs = std::filesystem;

using namespace st_detail;

namespace st_detail {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brosoundml: supertonic: " + msg);
}

}  // namespace st_detail

namespace {

const sf::TensorView& need(const sf::File& f, const std::string& name) {
    const sf::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    return *v;
}

// Read an F32 view straight to a host float vector (small denorm / slope params).
std::vector<float> read_host(const sf::File& f, const std::string& name) {
    const sf::TensorView& v = need(f, name);
    if (v.dtype != sf::Dtype::F32) fail("tensor '" + name + "' is not F32");
    const float* p = reinterpret_cast<const float*>(v.data);
    return std::vector<float>(p, p + v.numel());
}

// Upload a 2D-flattened weight to FP32 on `dev` (Supertonic weights are F32).
bt::Tensor up(const sf::File& f, const std::string& name, int rows, int cols,
              bt::Device dev) {
    bt::Tensor t;
    { bt::DeviceScope cpu(bt::Device::CPU); sf::upload(need(f, name), rows, cols, t); }
    return (dev == bt::Device::CPU) ? t : t.to(dev);
}
bt::Tensor up_vec(const sf::File& f, const std::string& name, int c, bt::Device dev) {
    return up(f, name, c, 1, dev);
}

std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) fail("cannot open " + p.string());
    std::ostringstream ss; ss << in.rdbuf();
    return ss.str();
}

// ── parameter blocks ─────────────────────────────────────────────────────────

// Load an ONNX conv weight (Cout, Cin/groups, K) by its exact tensor name.
ConvW load_conv(const sf::File& f, const std::string& wname,
                const std::string& bname, bt::Device dev) {
    const sf::TensorView& wv = need(f, wname);
    if (wv.shape.size() != 3) fail("conv '" + wname + "' is not rank-3");
    ConvW c;
    c.cout   = static_cast<int>(wv.shape[0]);
    c.cin_pg = static_cast<int>(wv.shape[1]);
    c.k      = static_cast<int>(wv.shape[2]);
    c.w = up(f, wname, c.cout, c.cin_pg * c.k, dev);
    if (!bname.empty()) {
        if (const sf::TensorView* bv = f.find(bname)) {
            c.b = up_vec(f, bname, static_cast<int>(bv->shape[0]), dev);
            c.has_b = true;
        }
    }
    return c;
}

// pwconv2 with the ConvNeXt LayerScale gamma folded in: gamma * (W·h + b) =
// (gamma*W)·h + gamma*b, so we scale each output row's weights and bias by
// gamma_c host-side. Eliminates a per-channel broadcast multiply in the forward.
ConvW load_pwconv2_scaled(const sf::File& f, const std::string& prefix,
                          bt::Device dev) {
    const sf::TensorView& wv = need(f, prefix + ".pwconv2.weight");  // (512,2048,1)
    const int cout   = static_cast<int>(wv.shape[0]);
    const int cin_pg = static_cast<int>(wv.shape[1]);
    std::vector<float> w = read_host(f, prefix + ".pwconv2.weight");
    std::vector<float> b = read_host(f, prefix + ".pwconv2.bias");
    std::vector<float> g = read_host(f, prefix + ".gamma");  // (1,cout,1) -> cout
    if (static_cast<int>(g.size()) != cout || static_cast<int>(b.size()) != cout)
        fail("pwconv2 gamma/bias length mismatch in " + prefix);
    for (int c = 0; c < cout; ++c) {
        const float gc = g[c];
        for (int i = 0; i < cin_pg; ++i) w[static_cast<std::size_t>(c) * cin_pg + i] *= gc;
        b[c] *= gc;
    }
    ConvW out;
    out.cout = cout; out.cin_pg = cin_pg; out.k = 1; out.has_b = true;
    out.w = bt::Tensor::from_host_on(dev, w.data(), cout, cin_pg);
    out.b = bt::Tensor::from_host_on(dev, b.data(), cout, 1);
    return out;
}

// Load a folded ONNX MatMul weight [in, out] (a Linear, x @ W) + bias as a 1x1
// conv: conv weight is [out, in], so transpose host-side. Bias is [out].
ConvW load_linear_as_conv(const sf::File& f, const std::string& wname,
                          const std::string& bname, bt::Device dev) {
    const sf::TensorView& wv = need(f, wname);
    if (wv.shape.size() != 2) fail("linear '" + wname + "' is not rank-2");
    const int in  = static_cast<int>(wv.shape[0]);
    const int out = static_cast<int>(wv.shape[1]);
    const std::vector<float> w = read_host(f, wname);   // [in][out] row-major
    std::vector<float> wt(static_cast<std::size_t>(out) * in);
    for (int i = 0; i < in; ++i)
        for (int o = 0; o < out; ++o)
            wt[static_cast<std::size_t>(o) * in + i] = w[static_cast<std::size_t>(i) * out + o];
    ConvW c;
    c.cout = out; c.cin_pg = in; c.k = 1; c.has_b = true;
    c.w = bt::Tensor::from_host_on(dev, wt.data(), out, in);
    const std::vector<float> b = read_host(f, bname);
    c.b = bt::Tensor::from_host_on(dev, b.data(), out, 1);
    return c;
}

// Load an ONNX Gemm (transB=1) weight [out, in] + bias [out] as a Linear:
// y = x @ W^T + b. The weight is transposed host-side to [in, out] so a
// row-vector matmul (x [1,in] @ wt [in,out]) needs no further transpose.
Gemm load_gemm(const sf::File& f, const std::string& wname,
               const std::string& bname, bt::Device dev) {
    const sf::TensorView& wv = need(f, wname);
    if (wv.shape.size() != 2) fail("gemm '" + wname + "' is not rank-2");
    const int out = static_cast<int>(wv.shape[0]);
    const int in  = static_cast<int>(wv.shape[1]);
    const std::vector<float> w = read_host(f, wname);   // [out][in] row-major
    std::vector<float> wt(static_cast<std::size_t>(in) * out);
    for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i)
            wt[static_cast<std::size_t>(i) * out + o] = w[static_cast<std::size_t>(o) * in + i];
    Gemm g;
    g.in = in; g.out = out;
    g.wt = bt::Tensor::from_host_on(dev, wt.data(), in, out);
    const std::vector<float> b = read_host(f, bname);
    g.b = bt::Tensor::from_host_on(dev, b.data(), 1, out);
    return g;
}

// ─── UnicodeProcessor frontend (mirrors py/helper.py UnicodeProcessor) ─────────

// Decode UTF-8 bytes to a codepoint sequence; malformed leads -> U+FFFD.
std::u32string utf8_to_u32(const std::string& s) {
    std::u32string out;
    const std::size_t n = s.size();
    for (std::size_t i = 0; i < n;) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        char32_t cp; int extra;
        if (c < 0x80)            { cp = c;         extra = 0; }
        else if ((c >> 5) == 0x6){ cp = c & 0x1F; extra = 1; }
        else if ((c >> 4) == 0xE){ cp = c & 0x0F; extra = 2; }
        else if ((c >> 3) == 0x1E){ cp = c & 0x07; extra = 3; }
        else                     { cp = 0xFFFD;   extra = 0; }
        ++i;
        for (int k = 0; k < extra && i < n; ++k, ++i)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i]) & 0x3F);
        out.push_back(cp);
    }
    return out;
}

bool is_emoji_cp(char32_t c) {
    return (c >= 0x1F600 && c <= 0x1F64F) || (c >= 0x1F300 && c <= 0x1F5FF) ||
           (c >= 0x1F680 && c <= 0x1F6FF) || (c >= 0x1F700 && c <= 0x1F77F) ||
           (c >= 0x1F780 && c <= 0x1F7FF) || (c >= 0x1F800 && c <= 0x1F8FF) ||
           (c >= 0x1F900 && c <= 0x1F9FF) || (c >= 0x1FA00 && c <= 0x1FA6F) ||
           (c >= 0x1FA70 && c <= 0x1FAFF) || (c >= 0x2600  && c <= 0x26FF)  ||
           (c >= 0x2700  && c <= 0x27BF)  || (c >= 0x1F1E6 && c <= 0x1F1FF);
}

// The final-character set that suppresses the auto-appended period (the upstream
// regex [.!?;:,'"')\]}…。」』】〉》›»]).
bool ends_with_punct(char32_t c) {
    switch (c) {
        case U'.': case U'!': case U'?': case U';': case U':': case U',':
        case U'\'': case U'"': case U')': case U']': case U'}':
        case 0x2026: case 0x3002: case 0x300D: case 0x300F: case 0x3011:
        case 0x3009: case 0x300B: case 0x203A: case 0x00BB:
            return true;
        default: return false;
    }
}

void replace_all(std::u32string& s, const std::u32string& from,
                 const std::u32string& to) {
    if (from.empty()) return;
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::u32string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// NFKD compatibility decomposition over the BMP. The decomposition table is
// generated offline from the Unicode database (supertonic-convert/gen_nfkd.py);
// Hangul syllables decompose algorithmically; a canonical-combining-class pass
// reorders marks. ASCII passes through unchanged.
#include "supertonic_nfkd.inc"  // kNfkdN/kCccN + kNfkd*/kCcc* tables (BMP)

// Canonical combining class of `c` (0 for starters / unlisted).
int nfkd_ccc(char32_t c) {
    if (c > 0xFFFF) return 0;
    int lo = 0, hi = kCccN;
    while (lo < hi) { const int m = (lo + hi) / 2;
                      if (kCccKey[m] < c) lo = m + 1; else hi = m; }
    return (lo < kCccN && kCccKey[lo] == c) ? kCccVal[lo] : 0;
}

// Append the (already-recursed) NFKD decomposition of one codepoint.
void nfkd_decompose_one(char32_t c, std::u32string& out) {
    constexpr char32_t kSBase = 0xAC00;
    constexpr int kSCount = 11172, kLBase = 0x1100, kVBase = 0x1161,
                  kTBase = 0x11A7, kTCount = 28, kNCount = 21 * 28;
    if (c >= kSBase && c < kSBase + static_cast<char32_t>(kSCount)) {  // Hangul
        const int si = static_cast<int>(c - kSBase);
        out.push_back(static_cast<char32_t>(kLBase + si / kNCount));
        out.push_back(static_cast<char32_t>(kVBase + (si % kNCount) / kTCount));
        if (const int ti = si % kTCount) out.push_back(static_cast<char32_t>(kTBase + ti));
        return;
    }
    if (c <= 0xFFFF) {
        int lo = 0, hi = kNfkdN;
        while (lo < hi) { const int m = (lo + hi) / 2;
                          if (kNfkdKey[m] < c) lo = m + 1; else hi = m; }
        if (lo < kNfkdN && kNfkdKey[lo] == c) {
            for (int i = kNfkdOff[lo]; i < kNfkdOff[lo + 1]; ++i)
                out.push_back(static_cast<char32_t>(kNfkdPool[i]));
            return;
        }
    }
    out.push_back(c);
}

std::u32string nfkd(const std::u32string& in) {
    std::u32string d; d.reserve(in.size());
    for (char32_t c : in) nfkd_decompose_one(c, d);

    // Canonical ordering: stable-sort each maximal run of combining marks by ccc.
    const std::size_t n = d.size();
    for (std::size_t i = 0; i < n;) {
        if (nfkd_ccc(d[i]) == 0) { ++i; continue; }
        std::size_t j = i;
        while (j < n && nfkd_ccc(d[j]) != 0) ++j;
        for (std::size_t a = i + 1; a < j; ++a) {        // stable insertion sort
            const char32_t v = d[a]; const int cv = nfkd_ccc(v);
            std::size_t b = a;
            while (b > i && nfkd_ccc(d[b - 1]) > cv) { d[b] = d[b - 1]; --b; }
            d[b] = v;
        }
        i = j;
    }
    return d;
}

// ─── sentence splitting for long-form synthesis ──────────────────────────────

// Encode a codepoint sequence back to UTF-8 (inverse of utf8_to_u32) — each
// split sentence is handed back to synthesize() as a UTF-8 string.
std::string u32_to_utf8(const std::u32string& s) {
    std::string out;
    for (char32_t c : s) {
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
        } else if (c < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (c >> 6)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else if (c < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (c >> 12)));
            out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (c >> 18)));
            out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
    }
    return out;
}

bool is_space_cp(char32_t c) {
    return c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == U'\f' ||
           c == 0x0B || c == 0xA0 /*nbsp*/ || c == 0x3000 /*ideographic space*/;
}
bool is_digit_cp(char32_t c) { return c >= U'0' && c <= U'9'; }
bool is_alpha_cp(char32_t c) {
    return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z');
}

// The lowercased ASCII-alpha run ending just before `i` (the punctuation), used
// to suppress a sentence split after a known abbreviation or a single-letter
// initial. Empty if the char before `i` is not alpha.
std::u32string preceding_word(const std::u32string& s, std::size_t i) {
    std::size_t b = i;
    while (b > 0 && is_alpha_cp(s[b - 1])) --b;
    std::u32string w = s.substr(b, i - b);
    for (char32_t& ch : w) if (ch >= U'A' && ch <= U'Z') ch = ch - U'A' + U'a';
    return w;
}

// A '.' right after one of these (case-insensitive) is an abbreviation, not a
// sentence end — splitting there would strand a tiny "Dr." / "etc." chunk. The
// list errs toward under-splitting (a slightly longer chunk is harmless; the
// flow field handles it and max_chars still bounds runaways).
bool is_abbrev(const std::u32string& w) {
    if (w.size() == 1) return true;  // single-letter initial: "J. R. R. Tolkien"
    static const std::u32string kAbbrev[] = {
        U"mr", U"mrs", U"ms", U"dr", U"prof", U"st", U"sr", U"jr", U"vs", U"etc",
        U"inc", U"ltd", U"co", U"corp", U"no", U"fig", U"al", U"dept", U"gov",
        U"sen", U"rep", U"gen", U"col", U"capt", U"lt", U"sgt", U"rev", U"hon",
        U"approx", U"e.g", U"i.e", U"a.m", U"p.m", U"u.s"};
    for (const std::u32string& a : kAbbrev) if (w == a) return true;
    return false;
}
bool is_sentence_end_cp(char32_t c) {
    return c == U'.' || c == U'!' || c == U'?' || c == 0x2026 /*…*/ ||
           c == 0x3002 /*。*/ || c == 0xFF01 /*！*/ || c == 0xFF1F /*？*/ ||
           c == 0xFF0E /*．*/;
}
// Closers that ride along after the terminal punctuation (quotes / brackets).
bool is_closer_cp(char32_t c) {
    return is_sentence_end_cp(c) || c == U'"' || c == U'\'' || c == U')' ||
           c == U']' || c == U'}' || c == 0x201D /*”*/ || c == 0x2019 /*’*/ ||
           c == 0x300D /*」*/ || c == 0x300F /*』*/ || c == 0x3011 /*】*/;
}
std::u32string trim_u32(const std::u32string& s, std::size_t a, std::size_t b) {
    while (a < b && is_space_cp(s[a])) ++a;
    while (b > a && is_space_cp(s[b - 1])) --b;
    return s.substr(a, b - a);
}

// Flatten a (possibly nested) JSON array of numbers into `out`, in order.
void flatten_numbers(const j::Value& v, std::vector<float>& out) {
    if (v.is_array())
        for (const j::Value& e : v.as_array()) flatten_numbers(e, out);
    else if (v.is_number())
        out.push_back(static_cast<float>(v.as_number()));
}

}  // namespace

namespace st_detail {

// The full _preprocess_text pipeline, codepoint-exact with the upstream.
std::u32string preprocess_text(const std::string& utf8, const std::string& lang) {
    static const char* kLangs[] = {
        "en","ko","ja","ar","bg","cs","da","de","el","es","et","fi","fr","hi",
        "hr","hu","id","it","lt","lv","nl","pl","pt","ro","ru","sk","sl","sv",
        "tr","uk","vi","na"};
    bool ok = false;
    for (const char* l : kLangs) if (lang == l) { ok = true; break; }
    if (!ok) fail("invalid language: " + lang);

    std::u32string t = nfkd(utf8_to_u32(utf8));

    // remove emoji
    { std::u32string o; o.reserve(t.size());
      for (char32_t c : t) if (!is_emoji_cp(c)) o.push_back(c);
      t.swap(o); }

    // single-codepoint dash/quote/symbol replacements, then special-symbol strip
    { std::u32string o; o.reserve(t.size());
      for (char32_t c : t) {
        switch (c) {
            case 0x2013: case 0x2011: case 0x2014: o.push_back(U'-'); break;
            case U'_':                              o.push_back(U' '); break;
            case 0x201C: case 0x201D:               o.push_back(U'"'); break;
            case 0x2018: case 0x2019: case 0x00B4: case U'`':
                                                    o.push_back(U'\''); break;
            case U'[': case U']': case U'|': case U'/': case U'#':
            case 0x2192: case 0x2190:               o.push_back(U' '); break;
            case 0x2665: case 0x2606: case 0x2661: case 0x00A9: case U'\\':
                break;  // remove ♥ ☆ ♡ © backslash
            default:                                o.push_back(c);
        }
      }
      t.swap(o); }

    // known-expression replacements
    replace_all(t, U"@",      U" at ");
    replace_all(t, U"e.g.,",  U"for example, ");
    replace_all(t, U"i.e.,",  U"that is, ");

    // tighten spacing before punctuation
    replace_all(t, U" ,", U",");
    replace_all(t, U" .", U".");
    replace_all(t, U" !", U"!");
    replace_all(t, U" ?", U"?");
    replace_all(t, U" ;", U";");
    replace_all(t, U" :", U":");
    replace_all(t, U" '", U"'");

    // collapse duplicate quotes
    while (t.find(U"\"\"") != std::u32string::npos) replace_all(t, U"\"\"", U"\"");
    while (t.find(U"''")   != std::u32string::npos) replace_all(t, U"''",   U"'");

    // collapse whitespace runs to a single space, then strip ends
    { std::u32string o; o.reserve(t.size()); bool sp = false;
      for (char32_t c : t) {
        const bool ws = (c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' ||
                         c == U'\f' || c == 0x0B);
        if (ws) { sp = true; continue; }
        if (sp && !o.empty()) o.push_back(U' ');
        sp = false; o.push_back(c);
      }
      t.swap(o); }

    // ensure a sentence-final punctuation
    if (t.empty() || !ends_with_punct(t.back())) t.push_back(U'.');

    // language tag wrap
    std::u32string lt(lang.begin(), lang.end());
    return U"<" + lt + U">" + t + U"</" + lt + U">";
}

// Split text into sentence chunks (UTF-8). A boundary is sentence-final
// punctuation — skipping a '.' between digits (decimal) — followed by whitespace
// or end of text; trailing quote/bracket closers stay with the sentence. Any
// chunk still longer than max_chars codepoints is broken again at the last
// whitespace before the limit (hard cut if a single token is longer), so no
// chunk drives the flow field to an unwieldy latent length.
std::vector<std::string> split_sentences_impl(const std::string& text, int max_chars) {
    const std::u32string s = utf8_to_u32(text);
    const std::size_t n = s.size();
    std::vector<std::u32string> raw;
    std::size_t start = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!is_sentence_end_cp(s[i])) continue;
        if (s[i] == U'.' && i > 0 && i + 1 < n &&
            is_digit_cp(s[i - 1]) && is_digit_cp(s[i + 1]))
            continue;  // decimal point — not a sentence end
        if (s[i] == U'.' && i > 0 && is_alpha_cp(s[i - 1]) &&
            is_abbrev(preceding_word(s, i)))
            continue;  // abbreviation / initial — not a sentence end
        std::size_t j = i + 1;
        while (j < n && is_closer_cp(s[j])) ++j;  // absorb e.g. ?!").  runs
        if (j >= n || is_space_cp(s[j])) {
            raw.push_back(s.substr(start, j - start));
            while (j < n && is_space_cp(s[j])) ++j;  // skip the gap
            start = j;
            i = (j > 0) ? j - 1 : 0;
        }
    }
    if (start < n) raw.push_back(s.substr(start, n - start));

    std::vector<std::string> out;
    for (const std::u32string& c0 : raw) {
        std::u32string c = trim_u32(c0, 0, c0.size());
        while (static_cast<int>(c.size()) > max_chars) {
            std::size_t cut = static_cast<std::size_t>(max_chars);
            while (cut > 0 && !is_space_cp(c[cut])) --cut;  // back up to a space
            if (cut == 0) cut = static_cast<std::size_t>(max_chars);  // no space: hard cut
            std::u32string head = trim_u32(c, 0, cut);
            if (!head.empty()) out.push_back(u32_to_utf8(head));
            std::size_t rest = cut;
            while (rest < c.size() && is_space_cp(c[rest])) ++rest;
            c = trim_u32(c, rest, c.size());
        }
        if (!c.empty()) out.push_back(u32_to_utf8(c));
    }
    return out;
}

}  // namespace st_detail

// ─── Weight Loading ──────────────────────────────────────────────────────────

void Supertonic::Impl::load(const std::string& dir, bt::Device device) {
    dev = device;
    const fs::path root(dir);

    // tts.json carries the autoencoder geometry; fall back to known defaults.
    if (fs::exists(root / "tts.json")) {
        const j::Value tts = j::parse(slurp(root / "tts.json"));
        if (const j::Value* ae = tts.find("ae")) {
            cfg.sample_rate = ae->get_int("sample_rate", cfg.sample_rate);
            cfg.base_chunk  = ae->get_int("base_chunk_size", cfg.base_chunk);
            cfg.latent_dim  = ae->get_int("ldim", cfg.latent_dim);
        }
        if (const j::Value* ttl = tts.find("ttl"))
            cfg.chunk = ttl->get_int("chunk_compress_factor", cfg.chunk);
    }

    const fs::path voc = root / "vocoder.safetensors";
    if (!fs::exists(voc)) fail("missing vocoder.safetensors under " + dir);
    const sf::File f = sf::File::open(voc.string());

    // De-normalisation: latent / scale, then * latent_std + latent_mean.
    const std::vector<float> scale = read_host(f, "tts.ttl.normalizer.scale");
    inv_scale   = (scale.empty() || scale[0] == 0.0f) ? 1.0f : 1.0f / scale[0];
    latent_mean = read_host(f, "tts.ae.latent_mean");   // (1,latent_dim,1)
    latent_std  = read_host(f, "tts.ae.latent_std");
    if (static_cast<int>(latent_mean.size()) != cfg.latent_dim ||
        static_cast<int>(latent_std.size())  != cfg.latent_dim)
        fail("latent_mean/std length != latent_dim");

    // Initial conv: latent_dim -> hidden (ONNX names this initializer onnx::Conv_*).
    conv_in = load_conv(f, "onnx::Conv_1441", "onnx::Conv_1442", dev);

    // 10 dilated ConvNeXt blocks. Dilations repeat 1,2,4 over the first six, then
    // 1 — taken from the depthwise weight's dilation in the upstream graph.
    static const int kDil[10] = {1, 2, 4, 1, 2, 4, 1, 1, 1, 1};
    blocks.clear();
    for (int i = 0; i < 10; ++i) {
        const std::string p = "tts.ae.decoder.convnext." + std::to_string(i);
        ConvNeXtBlock blk;
        blk.dil  = kDil[i];
        blk.dw   = load_conv(f, p + ".dwconv.net.weight", p + ".dwconv.net.bias", dev);
        blk.ln_g = up_vec(f, p + ".norm.norm.weight", blk.dw.cout, dev);
        blk.ln_b = up_vec(f, p + ".norm.norm.bias",   blk.dw.cout, dev);
        blk.pw1  = load_conv(f, p + ".pwconv1.weight", p + ".pwconv1.bias", dev);
        blk.pw2  = load_pwconv2_scaled(f, p, dev);
        blocks.push_back(std::move(blk));
    }

    // Final BatchNorm (inference: uses running stats).
    bn_g    = up_vec(f, "tts.ae.decoder.final_norm.norm.weight",       conv_in.cout, dev);
    bn_b    = up_vec(f, "tts.ae.decoder.final_norm.norm.bias",         conv_in.cout, dev);
    bn_mean = up_vec(f, "tts.ae.decoder.final_norm.norm.running_mean", conv_in.cout, dev);
    bn_var  = up_vec(f, "tts.ae.decoder.final_norm.norm.running_var",  conv_in.cout, dev);

    // Head: conv (kernel 3) -> PReLU (scalar slope) -> conv (kernel 1, no bias).
    head1 = load_conv(f, "tts.ae.decoder.head.layer1.net.weight",
                      "tts.ae.decoder.head.layer1.net.bias", dev);
    const std::vector<float> slope = read_host(f, "onnx::PRelu_1506");
    prelu_slope = slope.empty() ? 0.0f : slope[0];
    head2 = load_conv(f, "tts.ae.decoder.head.layer2.weight", "", dev);

    // Precompute the frozen-BN-inference input-grad adjoint as a depthwise 1x1
    // conv (per-channel scale gamma/sqrt(var+eps)) — lets the on-device encoder
    // reconstruction backward apply it without a per-step host round-trip.
    {
        const int C = conv_in.cout;
        const std::vector<float> gh = bn_g.to_host_vector(), vh = bn_var.to_host_vector();
        std::vector<float> sc(static_cast<std::size_t>(C));
        for (int c = 0; c < C; ++c) sc[c] = gh[c] / std::sqrt(vh[c] + 1.0e-5f);
        bn_scale_conv.cout = C; bn_scale_conv.cin_pg = 1; bn_scale_conv.k = 1;
        bn_scale_conv.has_b = false;
        bn_scale_conv.w = bt::Tensor::from_host_on(dev, sc.data(), C, 1);
    }

    ready = true;

    // Text encoder is optional today — load it when its weights are present.
    const fs::path te = root / "text_encoder.safetensors";
    if (fs::exists(te)) load_text_encoder(te);

    // Duration predictor is likewise optional until synthesize() lands.
    const fs::path dp = root / "duration_predictor.safetensors";
    if (fs::exists(dp)) load_duration_predictor(dp);

    const fs::path ve = root / "vector_estimator.safetensors";
    if (fs::exists(ve)) load_vector_estimator(ve);

    // UnicodeProcessor table: a flat list of 65536 ids (codepoint -> id, -1 OOV).
    const fs::path uix = root / "unicode_indexer.json";
    if (fs::exists(uix)) {
        const j::Value arr = j::parse(slurp(uix));
        const j::Value::Array& a = arr.as_array();
        unicode_index.assign(65536, -1);
        const std::size_t n = std::min<std::size_t>(a.size(), 65536);
        for (std::size_t i = 0; i < n; ++i)
            unicode_index[i] = static_cast<int>(a[i].as_number());
        frontend_ready = true;
    }
}

void Supertonic::Impl::load_text_encoder(const fs::path& path) {
    const sf::File f = sf::File::open(path.string());
    const std::string TE = "tts.ttl.text_encoder.";

    // Char embedding table [vocab x 256], kept host-resident for the gather.
    const sf::TensorView& emb = need(f, TE + "text_embedder.char_embedder.weight");
    if (emb.shape.size() != 2 || emb.shape[1] != kTeC)
        fail("char_embedder weight is not [vocab x 256]");
    te_vocab    = static_cast<int>(emb.shape[0]);
    te_char_emb = read_host(f, TE + "text_embedder.char_embedder.weight");

    // 6 ConvNeXt blocks (ksz 5). gamma folded into pwconv2 (load_pwconv2_scaled).
    te_blocks.clear();
    for (int i = 0; i < 6; ++i) {
        const std::string p = TE + "convnext.convnext." + std::to_string(i);
        ConvNeXtBlock blk;
        blk.dw   = load_conv(f, p + ".dwconv.weight", p + ".dwconv.bias", dev);
        blk.ln_g = up_vec(f, p + ".norm.norm.weight", blk.dw.cout, dev);
        blk.ln_b = up_vec(f, p + ".norm.norm.bias",   blk.dw.cout, dev);
        blk.pw1  = load_conv(f, p + ".pwconv1.weight", p + ".pwconv1.bias", dev);
        blk.pw2  = load_pwconv2_scaled(f, p, dev);
        te_blocks.push_back(std::move(blk));
    }

    // 4 relative-position attention layers.
    te_attn.clear();
    for (int i = 0; i < 4; ++i) {
        const std::string a = TE + "attn_encoder.attn_layers." + std::to_string(i) + ".";
        const std::string n1 = TE + "attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.";
        const std::string n2 = TE + "attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.";
        const std::string ff = TE + "attn_encoder.ffn_layers." + std::to_string(i) + ".";
        AttnLayer L;
        L.conv_q = load_conv(f, a + "conv_q.weight", a + "conv_q.bias", dev);
        L.conv_k = load_conv(f, a + "conv_k.weight", a + "conv_k.bias", dev);
        L.conv_v = load_conv(f, a + "conv_v.weight", a + "conv_v.bias", dev);
        L.conv_o = load_conv(f, a + "conv_o.weight", a + "conv_o.bias", dev);
        L.emb_rel_k = up(f, a + "emb_rel_k", 2 * kTeWindow + 1, kTeKc, dev);
        L.emb_rel_v = up(f, a + "emb_rel_v", 2 * kTeWindow + 1, kTeKc, dev);
        L.n1_g = up_vec(f, n1 + "weight", kTeC, dev);
        L.n1_b = up_vec(f, n1 + "bias",   kTeC, dev);
        L.n2_g = up_vec(f, n2 + "weight", kTeC, dev);
        L.n2_b = up_vec(f, n2 + "bias",   kTeC, dev);
        L.ffn1 = load_conv(f, ff + "conv_1.weight", ff + "conv_1.bias", dev);
        L.ffn2 = load_conv(f, ff + "conv_2.weight", ff + "conv_2.bias", dev);
        te_attn.push_back(std::move(L));
    }

    // Speech-prompted style cross-attention (two layers) + final LayerNorm.
    // Linears load by canonical name (the converter renames the ONNX-export
    // `onnx::MatMul_*` weights to `<scope>.linear.weight`).
    const std::string SP = "tts.ttl.speech_prompted_text_encoder.";
    auto load_spte = [&](const std::string& at) {
        const std::string b = SP + at + ".";
        StyleAttn A;
        A.wq = load_linear_as_conv(f, b + "W_query.linear.weight", b + "W_query.linear.bias", dev);
        A.wk = load_linear_as_conv(f, b + "W_key.linear.weight",   b + "W_key.linear.bias",   dev);
        A.wv = load_linear_as_conv(f, b + "W_value.linear.weight", b + "W_value.linear.bias", dev);
        A.wo = load_linear_as_conv(f, b + "out_fc.linear.weight",  b + "out_fc.linear.bias",  dev);
        return A;
    };
    spte1 = load_spte("attention1");
    spte2 = load_spte("attention2");
    spte_norm_g = up_vec(f, SP + "norm.norm.weight", kTeC, dev);
    spte_norm_b = up_vec(f, SP + "norm.norm.bias",   kTeC, dev);

    // Learned style-key prototype [1,50,256] -> channel-major [256,50].
    const bt::Tensor sk = up(f, "tts.ttl.style_encoder.style_token_layer.style_key",
                             kNStyle, kTeC, dev);             // [50,256]
    te_style_key = transpose2d(sk, kNStyle, kTeC);            // [256,50]

    te_ready = true;
}

void Supertonic::Impl::load_duration_predictor(const fs::path& path) {
    const sf::File f = sf::File::open(path.string());
    const std::string SE = "tts.dp.sentence_encoder.";

    // Char embedding [vocab x 64], kept host-resident for the gather.
    const sf::TensorView& emb = need(f, SE + "text_embedder.char_embedder.weight");
    if (emb.shape.size() != 2 || emb.shape[1] != kDpC)
        fail("dp char_embedder weight is not [vocab x 64]");
    dp_vocab    = static_cast<int>(emb.shape[0]);
    dp_char_emb = read_host(f, SE + "text_embedder.char_embedder.weight");

    // Learned sentence token [1,64,1] -> [64], prepended at sequence position 0.
    dp_sentence_token = read_host(f, SE + "sentence_token");
    if (static_cast<int>(dp_sentence_token.size()) != kDpC)
        fail("dp sentence_token is not length 64");

    // 6 ConvNeXt blocks (ksz 5, dilation 1). gamma folded into pwconv2.
    dp_blocks.clear();
    for (int i = 0; i < 6; ++i) {
        const std::string p = SE + "convnext.convnext." + std::to_string(i);
        ConvNeXtBlock blk;
        blk.dw   = load_conv(f, p + ".dwconv.weight", p + ".dwconv.bias", dev);
        blk.ln_g = up_vec(f, p + ".norm.norm.weight", blk.dw.cout, dev);
        blk.ln_b = up_vec(f, p + ".norm.norm.bias",   blk.dw.cout, dev);
        blk.pw1  = load_conv(f, p + ".pwconv1.weight", p + ".pwconv1.bias", dev);
        blk.pw2  = load_pwconv2_scaled(f, p, dev);
        dp_blocks.push_back(std::move(blk));
    }

    // 2 relative-position attention layers (2 heads, kc 32, window 4).
    dp_attn.clear();
    for (int i = 0; i < 2; ++i) {
        const std::string a  = SE + "attn_encoder.attn_layers." + std::to_string(i) + ".";
        const std::string n1 = SE + "attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.";
        const std::string n2 = SE + "attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.";
        const std::string ff = SE + "attn_encoder.ffn_layers." + std::to_string(i) + ".";
        AttnLayer L;
        L.conv_q = load_conv(f, a + "conv_q.weight", a + "conv_q.bias", dev);
        L.conv_k = load_conv(f, a + "conv_k.weight", a + "conv_k.bias", dev);
        L.conv_v = load_conv(f, a + "conv_v.weight", a + "conv_v.bias", dev);
        L.conv_o = load_conv(f, a + "conv_o.weight", a + "conv_o.bias", dev);
        L.emb_rel_k = up(f, a + "emb_rel_k", 2 * kDpWindow + 1, kDpKc, dev);
        L.emb_rel_v = up(f, a + "emb_rel_v", 2 * kDpWindow + 1, kDpKc, dev);
        L.n1_g = up_vec(f, n1 + "weight", kDpC, dev);
        L.n1_b = up_vec(f, n1 + "bias",   kDpC, dev);
        L.n2_g = up_vec(f, n2 + "weight", kDpC, dev);
        L.n2_b = up_vec(f, n2 + "bias",   kDpC, dev);
        L.ffn1 = load_conv(f, ff + "conv_1.weight", ff + "conv_1.bias", dev);
        L.ffn2 = load_conv(f, ff + "conv_2.weight", ff + "conv_2.bias", dev);
        dp_attn.push_back(std::move(L));
    }

    // proj_out 1x1 conv [64,64,1] applied to the pooled sentence token (no bias).
    dp_proj = load_conv(f, SE + "proj_out.net.weight", "", dev);

    // Predictor MLP: Gemm(192->128) -> PReLU -> Gemm(128->1) -> Exp.
    dp_pred0 = load_gemm(f, "tts.dp.predictor.layers.0.weight",
                         "tts.dp.predictor.layers.0.bias", dev);
    dp_pred1 = load_gemm(f, "tts.dp.predictor.layers.1.weight",
                         "tts.dp.predictor.layers.1.bias", dev);
    const std::vector<float> slope = read_host(f, "tts.dp.predictor.activation.weight");
    dp_prelu = slope.empty() ? 0.0f : slope[0];

    dp_ready = true;
}

void Supertonic::Impl::load_vector_estimator(const fs::path& path) {
    const sf::File f = sf::File::open(path.string());
    const std::string V = "vector_estimator.tts.ttl.vector_field.";

    ve_proj_in  = load_conv(f, V + "proj_in.net.weight",  "", dev);   // 144->512
    ve_proj_out = load_conv(f, V + "proj_out.net.weight", "", dev);   // 512->144

    // Time encoder MLP (kept host-resident; a tiny per-step scalar forward):
    // sinusoidal(t) [64] -> Gemm 64->256 -> Mish -> Gemm 256->64. Gemm weights
    // are [out,in].
    ve_t0w = read_host(f, V + "time_encoder.mlp.0.linear.weight");  // [256,64]
    ve_t0b = read_host(f, V + "time_encoder.mlp.0.linear.bias");    // [256]
    ve_t1w = read_host(f, V + "time_encoder.mlp.2.linear.weight");  // [64,256]
    ve_t1b = read_host(f, V + "time_encoder.mlp.2.linear.bias");    // [64]
    ve_sin_freqs = read_host(
        f, "/vector_estimator/vector_field/time_encoder/sinusoidal/Constant_3_output_0");

    auto load_convnext = [&](const std::string& p) {
        ConvNeXtBlock c;
        c.dw   = load_conv(f, p + ".dwconv.weight", p + ".dwconv.bias", dev);
        c.ln_g = up_vec(f, p + ".norm.norm.weight", c.dw.cout, dev);
        c.ln_b = up_vec(f, p + ".norm.norm.bias",   c.dw.cout, dev);
        c.pw1  = load_conv(f, p + ".pwconv1.weight", p + ".pwconv1.bias", dev);
        c.pw2  = load_pwconv2_scaled(f, p, dev);
        return c;
    };

    // RoPE theta/increments are a single shared parameter (exported once, on the
    // first text-attn block); every text-attn block reuses it. increments are the
    // implicit 0,1,2,... positions, so only theta is needed.
    const std::vector<float> rope_theta = read_host(f, V + "main_blocks.3.attn.theta");

    // 24 main blocks: a period-6 pattern (convnext, film, convnext, text-attn,
    // convnext, style-attn) repeated four times.
    ve_blocks.assign(24, VeBlock{});
    for (int i = 0; i < 24; ++i) {
        const std::string mb = V + "main_blocks." + std::to_string(i) + ".";
        VeBlock blk;
        const int r = i % 6;
        if (r == 0 || r == 2 || r == 4) {
            blk.type = 0;
            // ConvNeXt main-blocks hold a varying number of sub-blocks (4/1/1
            // per period) — load until the next sub-block's weight is absent.
            for (int s = 0;; ++s) {
                const std::string p = mb + "convnext." + std::to_string(s);
                if (!f.find(p + ".dwconv.weight")) break;
                blk.conv.push_back(load_convnext(p));
            }
        } else if (r == 1) {
            blk.type = 1;
            blk.film.w = up(f, mb + "linear.linear.weight", kVeHd, kVeC, dev);  // [64,512]
            blk.film.b = up(f, mb + "linear.linear.bias", 1, kVeC, dev);        // [1,512]
        } else if (r == 3) {
            blk.type = 2;
            const std::string a = mb + "attn.";
            blk.rope.conv_q = load_linear_as_conv(f, a + "W_query.linear.weight", a + "W_query.linear.bias", dev);
            blk.rope.conv_k = load_linear_as_conv(f, a + "W_key.linear.weight",   a + "W_key.linear.bias",   dev);
            blk.rope.conv_v = load_linear_as_conv(f, a + "W_value.linear.weight", a + "W_value.linear.bias", dev);
            blk.rope.conv_o = load_linear_as_conv(f, a + "out_fc.linear.weight",  a + "out_fc.linear.bias",  dev);
            blk.rope.theta  = rope_theta;                                       // shared [32]
            blk.rope.norm_g = up_vec(f, mb + "norm.norm.weight", kVeC, dev);
            blk.rope.norm_b = up_vec(f, mb + "norm.norm.bias",   kVeC, dev);
        } else {  // r == 5
            blk.type = 3;
            const std::string a = mb + "attention.";
            blk.style.attn.wq = load_linear_as_conv(f, a + "W_query.linear.weight", a + "W_query.linear.bias", dev);
            blk.style.attn.wk = load_linear_as_conv(f, a + "W_key.linear.weight",   a + "W_key.linear.bias",   dev);
            blk.style.attn.wv = load_linear_as_conv(f, a + "W_value.linear.weight", a + "W_value.linear.bias", dev);
            blk.style.attn.wo = load_linear_as_conv(f, a + "out_fc.linear.weight",  a + "out_fc.linear.bias",  dev);
            blk.style.norm_g  = up_vec(f, mb + "norm.norm.weight", kVeC, dev);
            blk.style.norm_b  = up_vec(f, mb + "norm.norm.bias",   kVeC, dev);
        }
        ve_blocks[i] = std::move(blk);
    }

    ve_last.clear();
    for (int i = 0; i < 4; ++i)
        ve_last.push_back(load_convnext(V + "last_convnext.convnext." + std::to_string(i)));

    // Learned style-key prototype + the unconditional special tokens, all stored
    // token-major [50,256] -> transposed to channel-major [256,50].
    const bt::Tensor proto = up(f, "/vector_estimator/Expand_output_0", kNStyle, kTeC, dev);
    ve_style_key_proto = transpose2d(proto, kNStyle, kTeC);
    const std::string UM = "vector_estimator.tts.ttl.uncond_masker.";
    ve_text_uncond = read_host(f, UM + "text_special_token");  // [256]
    const bt::Tensor sk = up(f, UM + "style_key_special_token",   kNStyle, kTeC, dev);
    ve_style_key_unc = transpose2d(sk, kNStyle, kTeC);
    const bt::Tensor sv = up(f, UM + "style_value_special_token", kNStyle, kTeC, dev);
    ve_style_val_unc = transpose2d(sv, kNStyle, kTeC);

    ve_ready = true;
}

VoiceStyle Supertonic::load_voice_style(const std::string& path) const {
    const j::Value root = j::parse(slurp(fs::path(path)));
    const j::Value* ttl = root.find("style_ttl");
    const j::Value* dp  = root.find("style_dp");
    if (!ttl || !dp) fail("voice style '" + path + "' missing style_ttl/style_dp");
    const j::Value* ttl_d = ttl->find("data");
    const j::Value* dp_d  = dp->find("data");
    if (!ttl_d || !dp_d) fail("voice style '" + path + "' missing data arrays");

    VoiceStyle vs;
    flatten_numbers(*ttl_d, vs.ttl);
    flatten_numbers(*dp_d,  vs.dp);
    if (static_cast<int>(vs.ttl.size()) != kNStyle * kTeC)
        fail("voice style ttl is " + std::to_string(vs.ttl.size()) + ", expected 50*256");
    if (vs.dp.empty()) fail("voice style dp is empty");
    return vs;
}

std::vector<std::string> Supertonic::split_sentences(const std::string& text,
                                                     int max_chars) {
    if (max_chars < 16) max_chars = 16;
    return split_sentences_impl(text, max_chars);
}

}  // namespace brosoundml
