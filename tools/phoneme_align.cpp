#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

// brosoundml_phoneme_align — turn REAL (audio, transcript) speech into a BPDS
// training shard via model-based constrained forced alignment.
//
// The synthetic Kokoro dataset (phoneme_synth) gets frame labels for free from
// Kokoro's predicted per-phoneme durations. Real recordings have no such oracle,
// so we ALIGN: g2p the transcript to a phoneme-class sequence, run the current
// PhonemeNet to per-frame posteriors, and Viterbi-align the KNOWN sequence to
// those posteriors (each phoneme >=1 frame, optional silence between phonemes
// and at the ends). Alignment is far easier than recognition — the sequence is
// given, so even a model that is weak on out-of-distribution audio only has to
// place boundaries, not decide identities. The result is a per-10ms class-label
// track in exactly the BPDS format phoneme_train consumes, so a real shard mixes
// straight into training alongside the synthetic one (phoneme_train --dataset
// synth.bpds,real.bpds). This is the lever that closes the synth->real gap the
// phoneme_calibrate sweep exposed.
//
// Self-training caveat: labels are only as good as the current model's posterior
// peaks, but the text constraint makes boundary placement robust; one align ->
// retrain pass already injects real acoustics, and the loop can be iterated.
//
// Input is the Emotional Speech Dataset (ESD), English speakers 0011-0020:
//   <esd>/<spk>/<spk>.txt           "<id>\t<text>\t<emotion>" per line
//   <esd>/<spk>/<Emotion>/<id>.wav  16 kHz mono PCM16
//
// Needs: the phoneme checkpoint (posteriors + class map), the Kokoro dir (g2p
// phoneme vocab) and the g2p lexicon / POS tagger. Missing any -> SKIP, return 0.

#include "brosoundml/phoneme_data.h"
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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
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

constexpr double kNegInf = -1e30;

struct Args {
    std::string weights;
    std::string esd_dir;
    std::string manifest;              // generic TSV mode: wav<TAB>text[<TAB>tag]
    std::string speakers = "0011,0012,0013,0014,0015,0016,0017,0018,0019,0020";
    std::string out;
    std::string kokoro_dir;
    std::string data_dir;
    std::string device = "auto";
    int   cap_per_emo_speaker = 200;   // clips per (speaker x emotion) / per tag; 0 = all
    float min_score = -3.5f;           // drop clips whose mean path log-posterior < this
    int   seed = 1234;
    bool  help = false;
};

void print_help() {
    std::printf(
        "brosoundml_phoneme_align — force-align real speech into a BPDS shard\n\n"
        "  --weights PATH       phoneme checkpoint (.bpm)  [or $BROSOUNDML_PHONEME_WEIGHTS]\n"
        "  --esd-dir DIR        ESD 'Emotion Speech Dataset' root\n"
        "  --manifest PATH      generic corpus mode: TSV 'wav<TAB>text[<TAB>tag]' per line;\n"
        "                       relative wav paths resolve against the manifest's dir;\n"
        "                       wavs must be 16 kHz mono. Overrides --esd-dir.\n"
        "  --speakers a,b,..    ESD speaker ids (default English 0011..0020)\n"
        "  --out PATH           output BPDS (default build-cuda/esd_real.bpds)\n"
        "  --kokoro-dir DIR     Kokoro model dir (g2p phoneme vocab)\n"
        "  --data-dir DIR       g2p data dir (lexicon + pos_tagger)  [or $BROSOUNDML_DATA_DIR]\n"
        "  --cap N              max clips per (speaker x emotion) / per manifest tag,\n"
        "                       0=all (default 200)\n"
        "  --min-score F        drop clips with mean path log-posterior < F (default -3.5)\n"
        "  --device auto|cuda|cpu\n  -h --help\n");
}

Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) fail("parse_args", "missing value"); return argv[++i]; };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "-h" || k == "--help") a.help = true;
        else if (k == "--weights") a.weights = need(i);
        else if (k == "--esd-dir") a.esd_dir = need(i);
        else if (k == "--manifest") a.manifest = need(i);
        else if (k == "--speakers") a.speakers = need(i);
        else if (k == "--out") a.out = need(i);
        else if (k == "--kokoro-dir") a.kokoro_dir = need(i);
        else if (k == "--data-dir") a.data_dir = need(i);
        else if (k == "--cap") a.cap_per_emo_speaker = std::stoi(need(i));
        else if (k == "--min-score") a.min_score = std::stof(need(i));
        else if (k == "--device") a.device = need(i);
        else if (!k.empty() && k[0] != '-' && a.weights.empty()) a.weights = k;
        else fail("parse_args", "unknown arg: " + k);
    }
    return a;
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out; std::string cur;
    for (char ch : s) { if (ch==',') { if(!cur.empty()) out.push_back(cur); cur.clear(); }
                        else cur.push_back(ch); }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// One transcript row.
struct Utt { std::string id, text, emo; };

// Read "<id>\t<text>\t<emotion>" lines (strip CR/BOM). Tolerant of stray blanks.
std::vector<Utt> read_transcript(const std::string& path) {
    std::vector<Utt> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() >= 3 && (unsigned char)line[0]==0xEF) line.erase(0,3); // BOM
        if (line.empty()) continue;
        auto t1 = line.find('\t');
        if (t1 == std::string::npos) continue;
        auto t2 = line.find('\t', t1 + 1);
        Utt u;
        u.id   = line.substr(0, t1);
        u.text = (t2==std::string::npos) ? line.substr(t1+1) : line.substr(t1+1, t2-t1-1);
        u.emo  = (t2==std::string::npos) ? std::string() : line.substr(t2+1);
        // trim trailing spaces on emo
        while (!u.emo.empty() && (u.emo.back()==' '||u.emo.back()=='\t')) u.emo.pop_back();
        if (!u.id.empty()) out.push_back(std::move(u));
    }
    return out;
}

// Constrained forced alignment. `S` is the target phoneme-class sequence (no
// silence, no adjacent duplicates). `logP` is (T,K) row-major log-posteriors.
// State graph (left-to-right, optional silence between phonemes and at ends):
//   even q = optional SILENCE slot (class 0); odd q = phoneme S[(q-1)/2].
// Each phoneme state must emit >=1 frame; silence slots may be skipped. Returns
// the per-frame class labels (size T) and sets *mean_logp to the path's average
// per-frame log-posterior; returns false if T < M (sequence cannot fit).
bool viterbi_align(const std::vector<float>& logP, int T, int K,
                   const std::vector<int>& S,
                   std::vector<std::int16_t>& labels, double* mean_logp) {
    const int M = (int)S.size();
    if (M == 0 || T <= 0 || T < M) return false;
    const int Q = 2 * M + 1;

    auto cls = [&](int q) -> int { return (q & 1) ? S[(q - 1) / 2] : 0; };
    auto emit = [&](int t, int q) -> double {
        return (double)logP[(std::size_t)t * K + cls(q)];
    };

    std::vector<double> prev(Q, kNegInf), cur(Q, kNegInf);
    std::vector<std::uint8_t> bp((std::size_t)T * Q, 0);  // 0:self 1:adv(q-1) 2:skip(q-2)

    // Frame 0: may start in leading silence (q=0) or first phoneme (q=1).
    prev[0] = emit(0, 0);
    if (Q > 1) prev[1] = emit(0, 1);

    for (int t = 1; t < T; ++t) {
        std::fill(cur.begin(), cur.end(), kNegInf);
        for (int q = 0; q < Q; ++q) {
            double best = prev[q]; std::uint8_t how = 0;          // self-loop
            if (q >= 1 && prev[q-1] > best) { best = prev[q-1]; how = 1; }   // advance
            if (q >= 3 && (q & 1) && prev[q-2] > best) {                     // skip silence
                best = prev[q-2]; how = 2; }
            if (best <= kNegInf) continue;
            cur[q] = best + emit(t, q);
            bp[(std::size_t)t * Q + q] = how;
        }
        std::swap(prev, cur);
    }

    // End in trailing silence (q=Q-1) or last phoneme (q=Q-2).
    int qend = Q - 1; double bestEnd = prev[Q-1];
    if (Q >= 2 && prev[Q-2] > bestEnd) { bestEnd = prev[Q-2]; qend = Q-2; }
    if (bestEnd <= kNegInf) return false;

    labels.assign((std::size_t)T, 0);
    int q = qend;
    for (int t = T - 1; t >= 0; --t) {
        labels[(std::size_t)t] = (std::int16_t)cls(q);
        if (t == 0) break;
        switch (bp[(std::size_t)t * Q + q]) {
            case 1: q -= 1; break;
            case 2: q -= 2; break;
            default: break;   // self
        }
    }
    *mean_logp = bestEnd / (double)T;
    return true;
}

}  // namespace

int main(int argc, char** argv) try {
    Args a = parse_args(argc, argv);
    if (a.help) { print_help(); return 0; }

    const std::string data = !a.data_dir.empty() ? a.data_dir
        : [&]{ std::string e = env_or_empty("BROSOUNDML_DATA_DIR");
               return !e.empty() ? e : std::string("D:/projects/brosoundml-data"); }();

    const std::string esd = !a.manifest.empty() ? std::string() : first_existing({
        a.esd_dir,
        "C:/Users/jonny/Downloads/Emotional Speech Dataset (ESD)/Emotion Speech Dataset",
    });
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
    const std::string out = !a.out.empty() ? a.out
        : "D:/projects/brosoundml/build-cuda/esd_real.bpds";

    std::string voice_path;
    if (!kokoro_dir.empty() && fs::exists(kokoro_dir + "/voices")) {
        std::vector<fs::path> vs;
        for (const auto& e : fs::directory_iterator(kokoro_dir + "/voices"))
            if (e.is_regular_file() && e.path().extension() == ".bin") vs.push_back(e.path());
        std::sort(vs.begin(), vs.end());
        if (!vs.empty()) voice_path = vs.front().string();
    }

    const bool manifest_mode = !a.manifest.empty();
    const bool source_ok = manifest_mode ? fs::exists(a.manifest) : !esd.empty();
    if (checkpoint.empty() || !source_ok || kokoro_dir.empty() ||
        !fs::exists(kokoro_dir + "/config.json") || lexicon.empty() || pos.empty()) {
        std::fprintf(stderr,
            "SKIP phoneme_align: missing inputs.\n  checkpoint=%s\n  esd=%s\n"
            "  manifest=%s\n  kokoro=%s\n  lexicon=%s\n  pos=%s\n",
            checkpoint.c_str(), esd.c_str(), a.manifest.c_str(),
            kokoro_dir.c_str(), lexicon.c_str(), pos.c_str());
        return 0;
    }

    bt::init();
    bt::Device device = bt::Device::CPU;
    if (a.device == "cuda" || (a.device == "auto" && bt::is_available(bt::Device::CUDA)))
        device = bt::Device::CUDA;
    if (a.device == "cpu") device = bt::Device::CPU;
    const char* dev_name = (device == bt::Device::CUDA) ? "cuda" : "cpu";

    // ── Model + class map + framing ──
    bsm::PhonemeNet model = bsm::PhonemeNet::load(checkpoint, device);
    bsm::PhonemeClassMap cm = model.class_map();
    cm.rebuild_inverse();
    const bsm::PhonemeNetConfig& mcfgm = model.config();
    const int K = cm.num_classes;
    const int n_mels = mcfgm.n_mels;

    bsm::MelConfig mcfg;
    mcfg.sample_rate = mcfgm.sample_rate;
    mcfg.n_fft       = mcfgm.n_fft;
    mcfg.win_length  = mcfgm.win_length;
    mcfg.hop_length  = mcfgm.hop_length;
    mcfg.n_mels      = mcfgm.n_mels;
    mcfg.compression = bsm::MelCompression::PCEN;
    bsm::MelFrontend mel(mcfg, device);

    // ── g2p ──
    bsm::Kokoro k;
    k.load(kokoro_dir, device);
    g::Lexicon        lex   = g::Lexicon::load(lexicon);
    g::Morphology     morph(lex);
    g::SpecialCases   sc(lex);
    g::PosTagger      tagger = g::PosTagger::load(pos);
    g::PhonemeAdapter adapter(k.config().vocab);
    g::Phonemizer     phon(tagger, lex, morph, sc, adapter);

    // ── BPDS writer (model's class map + framing so it mixes with the synth shard) ──
    bsm::PhonemeDatasetHeader hdr;
    hdr.sample_rate = mcfgm.sample_rate; hdr.n_fft = mcfgm.n_fft;
    hdr.win_length  = mcfgm.win_length;  hdr.hop_length = mcfgm.hop_length;
    hdr.n_mels      = mcfgm.n_mels;
    bsm::PhonemeDatasetWriter writer(out, hdr, cm);

    std::fprintf(stderr, "phoneme_align on %s  K=%d  out=%s\n", dev_name, K, out.c_str());

    // Map a g2p phoneme-id sequence -> class sequence (drop transparent +
    // silence, collapse adjacent duplicates). Mirrors PhonemeSpotter::enroll.
    auto to_class_seq = [&](const std::vector<int>& ids) {
        std::vector<int> S;
        for (int id : ids) {
            if (cm.is_transparent(id)) continue;
            int c;
            try { c = cm.class_for_id(id); } catch (...) { continue; }
            if (c == cm.silence_class()) continue;
            if (S.empty() || S.back() != c) S.push_back(c);
        }
        return S;
    };

    const std::vector<std::string> speakers = split_csv(a.speakers);
    std::mt19937 rng((std::uint32_t)a.seed);

    int kept = 0, dropped_fit = 0, dropped_score = 0, dropped_io = 0, dropped_seq = 0;
    std::vector<float> scores;
    std::map<std::string,int> kept_by_tag;

    // Align ONE (wav, text) pair and append it under `tag` (ESD emotion /
    // manifest tag — only used for capping buckets and the kept-by report).
    auto process_one = [&](const std::string& wav, const std::string& text,
                           const std::string& tag) {
        if (!fs::exists(wav)) { ++dropped_io; return; }

        bsm::AudioBuffer ab;
        try { ab = bsm::read_wav(wav); }
        catch (...) { ++dropped_io; return; }
        if (ab.sample_rate != mcfgm.sample_rate) { ++dropped_io; return; }
        const std::vector<float>& pcm = ab.samples;
        if (pcm.empty()) { ++dropped_io; return; }

        std::vector<int> S;
        try { S = to_class_seq(phon.phonemize(text)); }
        catch (...) { ++dropped_seq; return; }
        if ((int)S.size() < 1) { ++dropped_seq; return; }

        // PCEN mel -> forward -> log-softmax (T,K).
        bt::Tensor melt;
        mel.reset();
        mel.compute_offline(pcm.data(), (int)pcm.size(), melt);
        const std::vector<float> mh = melt.to_host_vector();
        const int Tm = (int)(mh.size() / (std::size_t)n_mels);
        if (Tm <= 0) { ++dropped_io; return; }
        bt::Tensor feats = bt::Tensor::from_host_on(device, mh.data(), n_mels, Tm);
        bt::Tensor logits; model.forward(feats, logits);
        const int T = logits.rows;
        std::vector<float> lg = logits.to_host_vector();
        for (int t = 0; t < T; ++t) {
            float* row = &lg[(std::size_t)t * K];
            float mx = row[0];
            for (int c = 1; c < K; ++c) mx = std::max(mx, row[c]);
            double s = 0.0;
            for (int c = 0; c < K; ++c) s += std::exp(row[c] - mx);
            const float lse = mx + (float)std::log(s > 0 ? s : 1.0);
            for (int c = 0; c < K; ++c) row[c] -= lse;   // log-softmax in place
        }

        std::vector<std::int16_t> labels; double mlp = 0.0;
        if (!viterbi_align(lg, T, K, S, labels, &mlp)) { ++dropped_fit; return; }
        if (mlp < a.min_score) { ++dropped_score; return; }

        // BPDS wants n_frames == framing(n_samples); the model framed the
        // SAME pcm, so labels.size()==T already matches read-back framing.
        try { writer.append(pcm, labels); }
        catch (...) { ++dropped_io; return; }
        ++kept; ++kept_by_tag[tag];
        scores.push_back((float)mlp);
        if (kept % 500 == 0) std::fprintf(stderr, "  kept %d ...\n", kept);
    };

    if (manifest_mode) {
        // Generic TSV: wav<TAB>text[<TAB>tag]; relative wavs resolve against
        // the manifest's directory. Cap applies per tag.
        const fs::path mroot = fs::path(a.manifest).parent_path();
        struct MUtt { std::string wav, text, tag; };
        std::vector<MUtt> utts;
        {
            std::ifstream f(a.manifest);
            std::string line;
            while (std::getline(f, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.size() >= 3 && (unsigned char)line[0]==0xEF) line.erase(0,3);
                if (line.empty()) continue;
                auto t1 = line.find('\t');
                if (t1 == std::string::npos) continue;
                auto t2 = line.find('\t', t1 + 1);
                MUtt u;
                u.wav  = line.substr(0, t1);
                u.text = (t2==std::string::npos) ? line.substr(t1+1)
                                                 : line.substr(t1+1, t2-t1-1);
                u.tag  = (t2==std::string::npos) ? std::string("default")
                                                 : line.substr(t2+1);
                while (!u.tag.empty() && (u.tag.back()==' '||u.tag.back()=='\t'))
                    u.tag.pop_back();
                if (u.wav.empty() || u.text.empty()) continue;
                fs::path w(u.wav);
                if (w.is_relative()) u.wav = (mroot / w).string();
                utts.push_back(std::move(u));
            }
        }
        std::fprintf(stderr, "  manifest: %zu utterances\n", utts.size());

        std::map<std::string,std::vector<int>> by_tag;
        for (int i = 0; i < (int)utts.size(); ++i) by_tag[utts[i].tag].push_back(i);
        for (auto& [tag, idxs] : by_tag) {
            std::shuffle(idxs.begin(), idxs.end(), rng);
            const int cap = a.cap_per_emo_speaker > 0
                ? std::min((int)idxs.size(), a.cap_per_emo_speaker) : (int)idxs.size();
            for (int j = 0; j < cap; ++j) {
                const MUtt& u = utts[idxs[j]];
                process_one(u.wav, u.text, tag);
            }
        }
    } else
    for (const auto& spk : speakers) {
        const std::string sdir = esd + "/" + spk;
        const std::string tpath = sdir + "/" + spk + ".txt";
        if (!fs::exists(tpath)) { std::fprintf(stderr, "  (no transcript %s)\n", tpath.c_str()); continue; }
        std::vector<Utt> utts = read_transcript(tpath);

        // Group by emotion for per-(speaker,emotion) capping.
        std::map<std::string,std::vector<int>> by_emo;
        for (int i = 0; i < (int)utts.size(); ++i) by_emo[utts[i].emo].push_back(i);

        for (auto& [emo, idxs] : by_emo) {
            std::shuffle(idxs.begin(), idxs.end(), rng);
            int cap = a.cap_per_emo_speaker > 0
                ? std::min((int)idxs.size(), a.cap_per_emo_speaker) : (int)idxs.size();
            for (int j = 0; j < cap; ++j) {
                const Utt& u = utts[idxs[j]];
                process_one(sdir + "/" + u.emo + "/" + u.id + ".wav", u.text, u.emo);
            }
        }
    }

    writer.finalize();

    std::sort(scores.begin(), scores.end());
    auto pct = [&](double p) {
        if (scores.empty()) return 0.0f;
        int i = std::clamp((int)std::lround(p * (scores.size() - 1)), 0, (int)scores.size()-1);
        return scores[i];
    };
    std::fprintf(stderr,
        "\nphoneme_align done: kept=%d  -> %s\n"
        "  dropped: seq=%d  doesnt-fit(T<M)=%d  low-score=%d  io=%d\n"
        "  path mean-logp  p05=%.3f  p50=%.3f  p95=%.3f  (min-score=%.2f)\n",
        kept, out.c_str(), dropped_seq, dropped_fit, dropped_score, dropped_io,
        pct(0.05), pct(0.50), pct(0.95), a.min_score);
    std::fprintf(stderr, "  kept by tag:");
    for (auto& [e, n] : kept_by_tag) std::fprintf(stderr, "  %s=%d", e.c_str(), n);
    std::fprintf(stderr, "\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "phoneme_align: %s\n", e.what());
    return 1;
}
