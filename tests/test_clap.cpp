// CLAP parity against transformers' ClapModel (tests/ref/gen_clap_fixture.py ->
// tests/fixtures/clap.bin) on every available device, plus determinism.
//
// Per clip: the 48 kHz resample, the log-mel (short / exactly 10 s / long
// crop), the HTSAT pooled output, the audio embedding (from the reference mel
// and end to end from the source clip), the long clip's window-mean mode; per
// prompt: the RoBERTa ids and the text embedding; then the cosine matrix,
// logits and softmax. Skips (passes) when the weights or the fixture are
// absent.
#include <brosoundml/clap.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace bt = brotensor;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++g_failures;
    std::cerr << "FAIL: " << what << std::endl;
}

struct Rec {
    int code = 0;
    std::vector<int> dims;
    std::vector<float> f;
    std::vector<int> i;
    std::string s;
};

std::map<std::string, Rec> read_fixture(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::map<std::string, Rec> out;
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "CLAPFIX1", 8) != 0) throw std::runtime_error("bad fixture magic");
    auto rd = [&](void* d, std::size_t n) {
        in.read(static_cast<char*>(d), static_cast<std::streamsize>(n));
        if (!in) throw std::runtime_error("truncated fixture");
    };
    int count = 0;
    rd(&count, 4);
    for (int r = 0; r < count; ++r) {
        int nl = 0;
        rd(&nl, 4);
        std::string name(static_cast<std::size_t>(nl), '\0');
        rd(name.data(), name.size());
        Rec rec;
        int nd = 0;
        rd(&rec.code, 4);
        rd(&nd, 4);
        rec.dims.resize(nd);
        rd(rec.dims.data(), 4u * nd);
        std::size_t n = 1;
        for (int d : rec.dims) n *= static_cast<std::size_t>(d);
        if (rec.code == 0) { rec.f.resize(n); rd(rec.f.data(), 4 * n); }
        else if (rec.code == 1) { rec.i.resize(n); rd(rec.i.data(), 4 * n); }
        else { rec.s.resize(n); rd(rec.s.data(), n); }
        out.emplace(name, std::move(rec));
    }
    return out;
}

std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string line;
    while (std::getline(ss, line)) out.push_back(line);
    return out;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    return m;
}

void report(const char* what, double v, double tol) {
    std::printf("    %-40s %.3e  (tol %.0e)\n", what, v, tol);
    check(v <= tol, std::string(what) + " = " + std::to_string(v) + " > " + std::to_string(tol));
}

struct DeviceResult {
    std::vector<std::vector<float>> audio, text;
};

DeviceResult run_device(const fs::path& dir, bt::Device dev, std::map<std::string, Rec>& fx) {
    DeviceResult res;
    std::printf("  == %s ==\n", bt::to_string(dev).c_str());
    brosoundml::Clap clap;
    clap.load(dir.string(), dev);
    check(clap.loaded(), "loaded");
    const double scale = fx["logit_scale_a"].f[0];
    report("logit_scale_a", std::fabs(clap.logit_scale() - scale), 1e-4);

    const auto clips = split_lines(fx["clip_names"].s);
    const auto prompts = split_lines(fx["prompt_texts"].s);

    for (const std::string& name : clips) {
        const std::string P = "clip/" + name + "/";
        std::printf("  clip %s\n", name.c_str());
        const Rec& src = fx[P + "src"];
        const int rate = fx[P + "rate"].i[0];
        const std::vector<float>& wave48 = fx[P + "wave48"].f;
        const int crop = fx[P + "crop"].i[0];
        brosoundml::AudioBuffer buf(src.f, rate);

        if (rate != 48000) {
            const std::vector<float> ours = clap.to_model_rate(buf);
            check(ours.size() == wave48.size(), name + ": resampled length");
            report("resample vs torchaudio (max abs)", max_abs(ours, wave48), 2e-5);
        }
        const std::vector<float> mel = clap.log_mel(wave48, crop);
        // FP32 STFT vs the reference's float64: the error sits in bins far
        // below the clip's peak (the 16 kHz clip's empty 8-24 kHz band).
        report("log-mel (max abs dB)", max_abs(mel, fx[P + "mel"].f), 5e-3);

        std::vector<float> pooled;
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<float> emb = clap.embed_mel(fx[P + "mel"].f, &pooled);
        std::printf("    audio tower: %.1f ms\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        report("HTSAT pooled (max abs)", max_abs(pooled, fx[P + "pooled"].f), 2e-4);
        report("audio embedding (max abs)", max_abs(emb, fx[P + "embed"].f), 2e-5);

        brosoundml::ClapAudioOptions o;
        o.long_mode = brosoundml::ClapLongMode::Crop;
        o.crop_offset = crop;
        o.pad = brosoundml::ClapPad::Repeat;   // the reference's front-end
        const std::vector<float> e2e = clap.embed_audio(buf, o);
        report("audio embedding end to end (max abs)", max_abs(e2e, fx[P + "embed"].f), 2e-4);
        res.audio.push_back(emb);

        // Determinism: the same mel embeds bit-identically.
        const std::vector<float> again = clap.embed_mel(fx[P + "mel"].f);
        check(again == emb, name + ": embed_mel is deterministic");

        if (fx.count(P + "win_embed")) {
            const std::vector<int> starts = clap.window_starts(static_cast<int>(wave48.size()));
            check(starts == fx[P + "win_starts"].i, name + ": window starts");
            brosoundml::ClapAudioOptions m;   // Mean (default)
            const std::vector<float> win = clap.embed_audio(buf, m);
            report("window-mean embedding (max abs)", max_abs(win, fx[P + "win_embed"].f), 2e-4);
        }
    }

    for (std::size_t p = 0; p < prompts.size(); ++p) {
        const std::string P = "prompt/" + std::to_string(p) + "/";
        const std::vector<std::int32_t> ids = clap.tokenize(prompts[p]);
        const std::vector<int>& ref = fx[P + "ids"].i;
        check(std::vector<int>(ids.begin(), ids.end()) == ref, "tokenize('" + prompts[p] + "') matches");
        const std::vector<float> t = clap.embed_ids(ids);
        const std::vector<float> t2 = clap.embed_text(prompts[p]);
        check(t == t2, "embed_text is deterministic");
        std::printf("  prompt %-45s", ("'" + prompts[p] + "'").c_str());
        report("", max_abs(t, fx[P + "embed"].f), 2e-5);
        res.text.push_back(t);
    }

    // Similarities, logits, softmax from our embeddings.
    std::vector<float> text_flat;
    for (const auto& t : res.text) text_flat.insert(text_flat.end(), t.begin(), t.end());
    const int np = static_cast<int>(res.text.size());
    double dsim = 0, dlog = 0, dprob = 0;
    for (std::size_t c = 0; c < res.audio.size(); ++c) {
        const brosoundml::ClapScore s =
            brosoundml::Clap::score_embeddings(res.audio[c], text_flat, np, clap.logit_scale());
        for (int p = 0; p < np; ++p) {
            const std::size_t k = c * np + p;
            dsim = std::max(dsim, std::fabs(double(s.similarity[p]) - fx["sim"].f[k]));
            dlog = std::max(dlog, std::fabs(double(s.logits[p]) - fx["logits"].f[k]));
            dprob = std::max(dprob, std::fabs(double(s.probability[p]) - fx["probs"].f[k]));
        }
    }
    report("similarity (max abs)", dsim, 2e-5);
    report("logits (max abs)", dlog, 1e-3);
    report("softmax (max abs)", dprob, 1e-4);

    // score() end to end on the first clip agrees with the pieces.
    {
        const std::string P = "clip/" + clips[0] + "/";
        brosoundml::AudioBuffer buf(fx[P + "src"].f, fx[P + "rate"].i[0]);
        const brosoundml::ClapScore s = clap.score(buf, prompts);
        double d = 0;
        for (int p = 0; p < np; ++p) d = std::max(d, std::fabs(double(s.probability[p]) - fx["probs"].f[p]));
        report("score() softmax, clip 0 (max abs)", d, 1e-4);
    }
    return res;
}

// The short-clip padding, which needs no weights: an unloaded Clap carries
// the default config (48 kHz, 480000-sample window).
void check_padding() {
    using brosoundml::ClapPad;
    const brosoundml::Clap clap;
    const int MAX = clap.config().max_samples, RATE = clap.config().sample_rate;
    const int two = static_cast<int>(brosoundml::kClapShortClipSeconds * RATE);
    check(clap.resolve_pad(ClapPad::Auto, 14400) == ClapPad::Silence, "pad: Auto is Silence for 0.3 s");
    check(clap.resolve_pad(ClapPad::Auto, two - 1) == ClapPad::Silence, "pad: Auto is Silence just under 2 s");
    check(clap.resolve_pad(ClapPad::Auto, two) == ClapPad::Repeat, "pad: Auto is Repeat from 2 s");
    check(clap.resolve_pad(ClapPad::Repeat, 10) == ClapPad::Repeat, "pad: Repeat passes through");
    check(clap.resolve_pad(ClapPad::Silence, MAX) == ClapPad::Silence, "pad: Silence passes through");

    // A 0.3 s clip of 1..n: once then zeros, or 33 whole tiles then zeros.
    const int n = 14400;
    std::vector<float> clip(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) clip[i] = static_cast<float>(i + 1);
    const std::vector<float> s = clap.fill_window(clip, -1, ClapPad::Silence);
    const std::vector<float> r = clap.fill_window(clip, -1, ClapPad::Repeat);
    const std::vector<float> a = clap.fill_window(clip, -1, ClapPad::Auto);
    check(static_cast<int>(s.size()) == MAX && static_cast<int>(r.size()) == MAX, "pad: window length");
    check(std::equal(clip.begin(), clip.end(), s.begin()), "pad: Silence starts with the clip");
    check(std::all_of(s.begin() + n, s.end(), [](float v) { return v == 0.0f; }), "pad: Silence is zeros after it");
    const int reps = MAX / n;
    bool tiled = true;
    for (int t = 0; t < reps; ++t) tiled = tiled && std::equal(clip.begin(), clip.end(), r.begin() + t * n);
    check(tiled, "pad: Repeat holds whole tiles");
    check(std::all_of(r.begin() + reps * n, r.end(), [](float v) { return v == 0.0f; }), "pad: Repeat tail is zeros");
    check(a == s, "pad: Auto fills a 0.3 s clip as Silence");
    // A 3 s clip keeps the reference's tiling under Auto.
    const std::vector<float> long3(static_cast<std::size_t>(3 * RATE), 0.5f);
    check(clap.fill_window(long3, -1, ClapPad::Auto) == clap.fill_window(long3, -1, ClapPad::Repeat),
          "pad: Auto fills a 3 s clip as Repeat");
    // The log-mel of the silence-padded clip is the log-mel of that window.
    check(clap.log_mel(clip, -1, ClapPad::Silence) == clap.log_mel(s), "pad: log_mel follows fill_window");
    std::cout << "  short-clip padding checks done" << std::endl;
}

}  // namespace

int main() {
    bt::init();
    check_padding();
    if (g_failures) {
        std::cerr << g_failures << " padding check(s) failed" << std::endl;
        return 1;
    }
    const fs::path repo = BROSOUNDML_REPO_DIR;
    const fs::path dir = repo / "weights" / "clap";
    const fs::path fixture = repo / "tests" / "fixtures" / "clap.bin";
    if (!fs::exists(dir / "model.safetensors") || !fs::exists(fixture)) {
        std::cout << "SKIP: need " << (dir / "model.safetensors").string() << " and " << fixture.string()
                  << " (scripts/download-clap.sh, convert-clap.py, tests/ref/gen_clap_fixture.py)" << std::endl;
        return 0;
    }
    auto fx = read_fixture(fixture);
    std::cout << "=== CLAP parity vs transformers ===" << std::endl;
    try {
        const DeviceResult cpu = run_device(dir, bt::Device::CPU, fx);
        if (bt::is_available(bt::Device::CUDA)) {
            const DeviceResult cuda = run_device(dir, bt::Device::CUDA, fx);
            double da = 0, dt = 0;
            for (std::size_t i = 0; i < cpu.audio.size(); ++i) da = std::max(da, max_abs(cpu.audio[i], cuda.audio[i]));
            for (std::size_t i = 0; i < cpu.text.size(); ++i) dt = std::max(dt, max_abs(cpu.text[i], cuda.text[i]));
            std::printf("  == CPU vs CUDA ==\n");
            report("audio embeddings (max abs)", da, 2e-5);
            report("text embeddings (max abs)", dt, 2e-5);
        } else {
            std::cout << "  (CUDA not available: CPU only)" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: exception: " << e.what() << std::endl;
        return 1;
    }
    if (g_failures) {
        std::cerr << g_failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "All CLAP parity checks passed." << std::endl;
    return 0;
}
