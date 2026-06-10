#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

// brosoundml_phoneme_calibrate — open-vocabulary KWS false-accept / false-reject
// sweep on REAL human speech.
//
// The frame-level gate (phoneme_test) and the single-clip e2e gate
// (test_phoneme_spotter_e2e) prove the path WORKS; this tool measures how WELL
// it works as a detector — the operating curve the runtime threshold actually
// rides. It enrolls a set of keywords via the g2p phonemizer (the headline
// "type a word, spot it" path), runs a corpus of real recordings through the
// spotter, and reports, per detection threshold, the false-reject rate (a
// keyword's own clips that fail to fire) and the false-accept rate (clips of
// OTHER words that wrongly fire the keyword). All keywords are enrolled at once,
// so every negative clip is a chance for any non-present keyword to false-fire —
// the real competitive setting.
//
// Faithful + fast: PhonemeNet::forward() over a whole clip is, per frame, an
// independent classification over that frame's causal receptive field — exactly
// what forward_streaming produces in the live spotter (phoneme_model.h). So we
// run the model ONCE per clip to cache its (T,K) softmax posteriors, then sweep
// the cheap streaming matcher (PhonemeSpotter::feed_posteriors, the offline
// class-map seam) over thresholds with zero extra GPU work.
//
// Input is the CAMEO English KWS set built by cameo/_probe/build_kws.py:
//   <kws-dir>/manifest.jsonl   {"wav","text","emo","actor"} per line
//   <kws-dir>/wav/*.wav        16 kHz mono PCM16
// A keyword's positives are the clips whose transcript contains it as a word;
// negatives are all other clips. (build_kws.py picks keywords unique to one of
// the 15 fixed sentences, so the split is unambiguous.)
//
// Needs real artefacts: the phoneme checkpoint, the Kokoro dir (for the g2p
// adapter's phoneme vocab), and the g2p lexicon / POS tagger. Missing any ->
// prints SKIP and returns 0.

#include "brosoundml/phoneme_spotter.h"
#include "brosoundml/phoneme_model.h"
#include "brosoundml/mel.h"
#include "brosoundml/audio.h"
#include "brosoundml/kokoro.h"

#include "brosoundml/g2p/lexicon.h"
#include "brosoundml/g2p/morphology.h"
#include "brosoundml/g2p/phoneme_adapter.h"
#include "brosoundml/g2p/phonemizer.h"
#include "brosoundml/g2p/pos_tagger.h"
#include "brosoundml/g2p/special_cases.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs  = std::filesystem;
namespace bt  = brotensor;
namespace bsm = brosoundml;
namespace g   = brosoundml::g2p;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

std::string env_or_empty(const char* k) {
    const char* v = std::getenv(k);
    return v ? std::string(v) : std::string();
}

std::string first_existing(const std::vector<std::string>& cands) {
    for (const auto& c : cands)
        if (!c.empty() && fs::exists(c)) return c;
    return {};
}

// The default keyword set: content words each unique to one of the 15 jl_corpus
// sentences. A mix of sentence-INITIAL words (a real silence boundary precedes
// them — the natural wake-word setting) and EMBEDDED words (no boundary; they
// stress the entry-silence gate). Override with --keywords.
const std::vector<std::string> kDefaultKeywords = {
    // sentence-initial
    "carl", "taylor", "linda", "water", "sound", "john", "work", "jack",
    // embedded
    "farmer", "father", "darts", "tooth", "buried",
};

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur.push_back(ch);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Lower-case word tokens of a transcript.
std::set<std::string> word_set(const std::string& text) {
    std::set<std::string> w;
    std::string cur;
    for (char ch : text) {
        if (std::isalpha((unsigned char)ch) || ch == '\'') {
            cur.push_back((char)std::tolower((unsigned char)ch));
        } else if (!cur.empty()) { w.insert(cur); cur.clear(); }
    }
    if (!cur.empty()) w.insert(cur);
    return w;
}

// Minimal JSONL field extractor for the controlled manifest (values are plain
// strings with no escaped quotes). Returns "" if the key is absent.
std::string json_str(const std::string& line, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    auto p = line.find(pat);
    if (p == std::string::npos) return {};
    p = line.find(':', p + pat.size());
    if (p == std::string::npos) return {};
    p = line.find('"', p);
    if (p == std::string::npos) return {};
    auto q = line.find('"', p + 1);
    if (q == std::string::npos) return {};
    return line.substr(p + 1, q - p - 1);
}

struct Clip {
    std::string wav;
    std::string text;
    std::string emo;
    std::set<std::string> words;
    std::vector<float> post;   // (T, K) row-major softmax posteriors
    int T = 0;
};

struct Args {
    std::string weights;
    std::string kws_dir;
    std::string manifest;
    std::string kokoro_dir;
    std::string data_dir;
    std::string device   = "auto";
    std::string keywords;            // CSV; empty -> kDefaultKeywords
    std::string thresholds;          // CSV; empty -> default grid
    float emission_floor = 0.15f;
    int   min_phonemes   = 3;
    int   smoothing_hits = 2;
    int   smoothing_window = 3;
    int   entry_silence  = 2;        // SpotterConfig default
    float tail_seconds   = 0.25f;    // trailing silence so the M-of-N smoother settles
    bool  help = false;
};

void print_help() {
    std::printf(
        "brosoundml_phoneme_calibrate — KWS false-accept/false-reject sweep on real speech\n\n"
        "  --weights PATH       phoneme checkpoint (.bpm)  [or $BROSOUNDML_PHONEME_WEIGHTS]\n"
        "  --kws-dir DIR        CAMEO KWS set dir (manifest.jsonl + wav/)  [default D:/projects/cameo/_kws]\n"
        "  --manifest PATH      override manifest path (default <kws-dir>/manifest.jsonl)\n"
        "  --kokoro-dir DIR     Kokoro model dir (for the g2p phoneme vocab)\n"
        "  --data-dir DIR       g2p data dir (lexicon + pos_tagger)  [or $BROSOUNDML_DATA_DIR]\n"
        "  --keywords a,b,c     keywords to enroll (default: a curated jl_corpus set)\n"
        "  --thresholds t,...   detection thresholds to sweep (default 0.18..0.60)\n"
        "  --floor F            emission_floor (default 0.15; 0 = strict citation match)\n"
        "  --min-phonemes N     template length floor (default 3)\n"
        "  --smoothing-hits M   M-of-N smoother hits (default 2)\n"
        "  --smoothing-window N M-of-N smoother window (default 3)\n"
        "  --entry-silence N    entry-gate silence frames (default 2; 0 = no boundary needed)\n"
        "  --device auto|cuda|cpu\n"
        "  -h --help\n");
}

Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) fail("parse_args", "missing value");
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "-h" || k == "--help") a.help = true;
        else if (k == "--weights") a.weights = need(i);
        else if (k == "--kws-dir") a.kws_dir = need(i);
        else if (k == "--manifest") a.manifest = need(i);
        else if (k == "--kokoro-dir") a.kokoro_dir = need(i);
        else if (k == "--data-dir") a.data_dir = need(i);
        else if (k == "--keywords") a.keywords = need(i);
        else if (k == "--thresholds") a.thresholds = need(i);
        else if (k == "--floor") a.emission_floor = std::stof(need(i));
        else if (k == "--min-phonemes") a.min_phonemes = std::stoi(need(i));
        else if (k == "--smoothing-hits") a.smoothing_hits = std::stoi(need(i));
        else if (k == "--smoothing-window") a.smoothing_window = std::stoi(need(i));
        else if (k == "--entry-silence") a.entry_silence = std::stoi(need(i));
        else if (k == "--device") a.device = need(i);
        else if (!k.empty() && k[0] != '-' && a.weights.empty()) a.weights = k;
        else fail("parse_args", "unknown arg: " + k);
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) try {
    Args a = parse_args(argc, argv);
    if (a.help) { print_help(); return 0; }

    if (a.kws_dir.empty()) a.kws_dir = "D:/projects/cameo/_kws";
    if (a.manifest.empty()) a.manifest = a.kws_dir + "/manifest.jsonl";

    const std::string data = !a.data_dir.empty() ? a.data_dir
        : [&]{ std::string e = env_or_empty("BROSOUNDML_DATA_DIR");
               return !e.empty() ? e : std::string("D:/projects/brosoundml-data"); }();

    const std::string kokoro_dir = first_existing({
        a.kokoro_dir, env_or_empty("BROSOUNDML_KOKORO_DIR"),
        "D:/projects/brosoundml/weights/kokoro", data + "/kokoro",
    });
    const std::string lexicon = first_existing({
        data + "/g2p/lexicon_en_us.bin",
        "D:/projects/brosoundml-data/g2p/lexicon_en_us.bin",
    });
    const std::string pos = first_existing({
        data + "/pos_tagger/model.bin",
        "D:/projects/brosoundml-data/pos_tagger/model.bin",
    });
    const std::string checkpoint = first_existing({
        a.weights, env_or_empty("BROSOUNDML_PHONEME_WEIGHTS"),
        "D:/projects/brosoundml/build-cuda/english.bpm",
    });

    std::string voice_path;  // any voice — only the config vocab is used
    if (!kokoro_dir.empty() && fs::exists(kokoro_dir + "/voices")) {
        std::vector<fs::path> vs;
        for (const auto& e : fs::directory_iterator(kokoro_dir + "/voices"))
            if (e.is_regular_file() && e.path().extension() == ".bin")
                vs.push_back(e.path());
        std::sort(vs.begin(), vs.end());
        if (!vs.empty()) voice_path = vs.front().string();
    }

    if (checkpoint.empty() || kokoro_dir.empty() ||
        !fs::exists(kokoro_dir + "/config.json") || lexicon.empty() ||
        pos.empty() || !fs::exists(a.manifest)) {
        std::fprintf(stderr,
            "SKIP phoneme_calibrate: missing inputs.\n"
            "  checkpoint = %s\n  kokoro_dir = %s\n  lexicon    = %s\n"
            "  pos_tagger = %s\n  manifest   = %s\n",
            checkpoint.c_str(), kokoro_dir.c_str(), lexicon.c_str(),
            pos.c_str(), a.manifest.c_str());
        return 0;
    }

    bt::init();
    bt::Device device = bt::Device::CPU;
    if (a.device == "cuda" || (a.device == "auto" && bt::is_available(bt::Device::CUDA)))
        device = bt::Device::CUDA;
    if (a.device == "cpu") device = bt::Device::CPU;
    const char* dev_name = (device == bt::Device::CUDA) ? "cuda" : "cpu";

    // ── Model (for forward) + its class map / framing ──
    bsm::PhonemeNet model = bsm::PhonemeNet::load(checkpoint, device);
    const bsm::PhonemeClassMap cm = model.class_map();
    cm.rebuild_inverse();
    const bsm::PhonemeNetConfig& mc = model.config();
    const int K = cm.num_classes;
    const int n_mels = mc.n_mels;

    // PCEN mel front-end, identical to PhonemeSpotter::load().
    bsm::MelConfig mcfg;
    mcfg.sample_rate = mc.sample_rate;
    mcfg.n_fft       = mc.n_fft;
    mcfg.win_length  = mc.win_length;
    mcfg.hop_length  = mc.hop_length;
    mcfg.n_mels      = mc.n_mels;
    mcfg.compression = bsm::MelCompression::PCEN;
    bsm::MelFrontend mel(mcfg, device);

    // ── g2p phonemizer (citation enroll path) ──
    bsm::Kokoro k;
    k.load(kokoro_dir, device);
    g::Lexicon        lex   = g::Lexicon::load(lexicon);
    g::Morphology     morph(lex);
    g::SpecialCases   sc(lex);
    g::PosTagger      tagger = g::PosTagger::load(pos);
    g::PhonemeAdapter adapter(k.config().vocab);
    g::Phonemizer     phon(tagger, lex, morph, sc, adapter);

    // ── Matcher (offline seam — class map, no model) ──
    bsm::PhonemeSpotter spotter;
    spotter.set_class_map(cm);

    std::fprintf(stderr, "phoneme_calibrate on %s  K=%d  checkpoint=%s\n",
                 dev_name, K, checkpoint.c_str());

    // ── Keywords ──
    std::vector<std::string> keywords =
        a.keywords.empty() ? kDefaultKeywords : split_csv(a.keywords);
    std::map<std::string, std::vector<int>> kw_ids;
    for (const auto& kw : keywords) kw_ids[kw] = phon.phonemize(kw);

    // ── Load manifest, decode + cache posteriors once per clip ──
    std::vector<Clip> clips;
    {
        std::ifstream f(a.manifest);
        std::string line;
        while (std::getline(f, line)) {
            if (line.find('{') == std::string::npos) continue;
            Clip c;
            c.wav  = json_str(line, "wav");
            c.text = json_str(line, "text");
            c.emo  = json_str(line, "emo");
            if (c.wav.empty()) continue;
            c.words = word_set(c.text);
            clips.push_back(std::move(c));
        }
    }
    if (clips.empty()) fail("phoneme_calibrate", "no clips in manifest");

    const int tail = (int)std::lround(a.tail_seconds * mc.sample_rate);
    std::vector<float> sil_tail((std::size_t)std::max(0, tail), 0.0f);

    std::fprintf(stderr, "decoding + forward on %zu clips ...\n", clips.size());
    int done = 0;
    for (auto& c : clips) {
        bsm::AudioBuffer ab = bsm::read_wav(a.kws_dir + "/" + c.wav);
        std::vector<float> pcm = ab.samples;             // 16 kHz mono
        pcm.insert(pcm.end(), sil_tail.begin(), sil_tail.end());

        bt::Tensor melt;   // (n_mels, T) on CPU
        mel.reset();
        mel.compute_offline(pcm.data(), (int)pcm.size(), melt);
        const std::vector<float> mh = melt.to_host_vector();
        const int T = (int)(mh.size() / (std::size_t)n_mels);
        if (T <= 0) { c.T = 0; ++done; continue; }

        bt::Tensor feats = bt::Tensor::from_host_on(device, mh.data(), n_mels, T);
        bt::Tensor logits; // (T, K)
        model.forward(feats, logits);
        const int Tout = logits.rows;
        std::vector<float> lg = logits.to_host_vector();

        // Row-wise softmax -> posteriors (what feed_posteriors expects).
        c.post.assign((std::size_t)Tout * K, 0.0f);
        for (int t = 0; t < Tout; ++t) {
            float* row = &lg[(std::size_t)t * K];
            float mx = row[0];
            for (int j = 1; j < K; ++j) mx = std::max(mx, row[j]);
            double sum = 0.0;
            for (int j = 0; j < K; ++j) { row[j] = std::exp(row[j] - mx); sum += row[j]; }
            const float inv = (float)(1.0 / (sum > 0 ? sum : 1.0));
            float* out = &c.post[(std::size_t)t * K];
            for (int j = 0; j < K; ++j) out[j] = row[j] * inv;
        }
        c.T = Tout;
        if (++done % 100 == 0)
            std::fprintf(stderr, "  %d/%zu\n", done, clips.size());
    }

    // ── Threshold grid ──
    std::vector<float> grid;
    if (!a.thresholds.empty()) for (auto& s : split_csv(a.thresholds)) grid.push_back(std::stof(s));
    else grid = {0.18f,0.22f,0.26f,0.30f,0.34f,0.38f,0.42f,0.46f,0.50f,0.55f,0.60f};

    // Positive counts per keyword (clips whose transcript contains it).
    std::map<std::string,int> n_pos;
    for (const auto& kw : keywords) {
        int p = 0;
        for (const auto& c : clips) if (c.words.count(kw)) ++p;
        n_pos[kw] = p;
    }
    const int n_clips = (int)clips.size();

    std::fprintf(stderr, "\nemission_floor=%.2f  min_phonemes=%d  smoothing=%d/%d"
                 "  entry_silence=%d\n", a.emission_floor, a.min_phonemes,
                 a.smoothing_hits, a.smoothing_window, a.entry_silence);
    std::printf("\n# keyword positives (each negative pool = %d clips):\n", n_clips);
    for (const auto& kw : keywords)
        std::printf("#   %-10s pos=%d neg=%d\n", kw.c_str(), n_pos[kw], n_clips - n_pos[kw]);

    // ── Sweep ──
    // Per (threshold): aggregate micro FAR/FRR + per-keyword + best-emotion recall.
    std::printf("\n# aggregate operating curve (micro-averaged over %zu keywords):\n",
                keywords.size());
    std::printf("#  thr     FRR      FAR    (TP/P)        (FP/N)\n");

    struct Row { float thr, frr, far; int TP,P,FP,N; };
    std::vector<Row> curve;
    // Per-emotion recall at each threshold (TP,P).
    std::map<std::string,std::map<float,std::pair<int,int>>> emo_tp_p;
    // Per-keyword FRR/FAR at each threshold for a detail table.
    std::map<std::string,std::vector<std::pair<float,std::pair<float,float>>>> kw_detail;
    // Per-keyword (tp,fp) at each threshold — feeds the per-template oracle.
    std::map<std::string,std::map<float,std::pair<int,int>>> kw_tf;

    for (float thr : grid) {
        spotter.clear();
        bsm::SpotterConfig pol = spotter.config();
        pol.threshold        = thr;
        pol.emission_floor   = a.emission_floor;
        pol.min_phonemes     = a.min_phonemes;
        pol.smoothing_hits   = a.smoothing_hits;
        pol.smoothing_window = a.smoothing_window;
        pol.entry_silence_frames = a.entry_silence;
        for (const auto& kw : keywords)
            spotter.enroll(kw, kw_ids[kw], &pol);

        std::map<std::string,int> tp, fp;   // per keyword
        for (const auto& kw : keywords) { tp[kw]=0; fp[kw]=0; }

        for (const auto& c : clips) {
            if (c.T <= 0) continue;
            spotter.reset();
            auto evs = spotter.feed_posteriors(c.post.data(), c.T);
            std::set<std::string> fired;
            for (const auto& e : evs) fired.insert(e.name);
            for (const auto& kw : keywords) {
                const bool is_pos = c.words.count(kw) != 0;
                const bool hit    = fired.count(kw) != 0;
                if (is_pos) {
                    if (hit) ++tp[kw];
                    auto& ep = emo_tp_p[c.emo][thr];
                    ep.second += 1;            // P for this emotion
                    if (hit) ep.first += 1;    // TP
                } else if (hit) ++fp[kw];
            }
        }

        int TP=0,P=0,FP=0,N=0;
        for (const auto& kw : keywords) {
            const int p = n_pos[kw], n = n_clips - p;
            TP += tp[kw]; P += p; FP += fp[kw]; N += n;
            const float frr = p ? (float)(p - tp[kw]) / p : 0.0f;
            const float far = n ? (float)fp[kw] / n : 0.0f;
            kw_detail[kw].push_back({thr,{frr,far}});
            kw_tf[kw][thr] = {tp[kw], fp[kw]};
        }
        const float frr = P ? (float)(P - TP) / P : 0.0f;
        const float far = N ? (float)FP / N : 0.0f;
        curve.push_back({thr,frr,far,TP,P,FP,N});
        std::printf("  %.3f  %6.3f   %6.4f   (%4d/%4d)   (%4d/%5d)\n",
                    thr, frr, far, TP, P, FP, N);
    }

    // ── Equal-error point (min |FAR-FRR|) ──
    const Row* eer = &curve.front();
    for (const auto& r : curve)
        if (std::fabs(r.far - r.frr) < std::fabs(eer->far - eer->frr)) eer = &r;
    std::printf("\n# equal-error-ish operating point: thr=%.3f  FRR=%.3f  FAR=%.4f\n",
                eer->thr, eer->frr, eer->far);
    // Lowest-threshold point with FAR <= 1%.
    const Row* op = nullptr;
    for (const auto& r : curve) if (r.far <= 0.01f) { op = &r; break; }
    if (op)
        std::printf("# first FAR<=1%% point:            thr=%.3f  FRR=%.3f  FAR=%.4f\n",
                    op->thr, op->frr, op->far);
    else
        std::printf("# no threshold reached FAR<=1%% on this grid.\n");

    // ── Per-template oracle ──
    // The ceiling a PER-KEYWORD threshold could reach vs the single global one:
    // for each FAR budget, give every keyword its own best grid threshold (max
    // recall s.t. that keyword's FAR <= budget) and re-aggregate. The gap to the
    // global curve is the headroom from calibrating threshold per template.
    std::printf("\n# per-template oracle (each keyword at its own best threshold):\n");
    std::printf("#  FARbudget   FRR      FAR     (TP/P)        (FP/N)\n");
    for (float budget : {0.005f, 0.01f, 0.02f, 0.05f}) {
        int TP=0,P=0,FP=0,N=0;
        for (const auto& kw : keywords) {
            const int p = n_pos[kw], n = n_clips - p;
            int best_tp = 0, best_fp = 0;   // default: fire nothing (thr above all)
            for (const auto& [thr, tf] : kw_tf[kw]) {
                const float far = n ? (float)tf.second / n : 0.0f;
                if (far <= budget && tf.first > best_tp) { best_tp = tf.first; best_fp = tf.second; }
            }
            TP += best_tp; P += p; FP += best_fp; N += n;
        }
        std::printf("   %.3f      %6.3f   %6.4f   (%4d/%4d)   (%4d/%5d)\n",
                    budget, P?(float)(P-TP)/P:0.0f, N?(float)FP/N:0.0f, TP,P,FP,N);
    }

    // ── Per-keyword detail at the EER threshold ──
    std::printf("\n# per-keyword at thr=%.3f (EER):  FRR     FAR\n", eer->thr);
    for (const auto& kw : keywords) {
        for (const auto& d : kw_detail[kw])
            if (d.first == eer->thr)
                std::printf("#   %-10s  %6.3f  %6.4f\n", kw.c_str(), d.second.first, d.second.second);
    }

    // ── Recall by emotion at the EER threshold ──
    std::printf("\n# recall by emotion at thr=%.3f:\n", eer->thr);
    for (const auto& [emo, m] : emo_tp_p) {
        auto it = m.find(eer->thr);
        if (it == m.end() || it->second.second == 0) continue;
        const int t = it->second.first, p = it->second.second;
        std::printf("#   %-10s  %.3f  (%d/%d)\n", emo.c_str(), (float)t/p, t, p);
    }

    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "phoneme_calibrate: %s\n", e.what());
    return 1;
}
