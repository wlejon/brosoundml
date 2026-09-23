// brosoundml_laya_audio_prep — word-timing files for corpora that do not go
// through the Parakeet aligner (brosoundml_laya_audio_align).
//
//   ami        AMI meetings: the manual annotation's word files already carry
//              word start/end times. All speakers of a meeting are merged
//              onto one timeline (the single distant mic hears them all) and
//              cut into 5-20 s chunks at pauses; a chunk is the audio range
//              "sdm/<meeting>.opus@t0:t1" and lists every word overlapping
//              it (a word cut by a forced chunk edge is partial, and the
//              labels ignore it). Split: the standard AMI full-corpus-ASR
//              partition (dev / eval meetings held out).
//   clips      single-word clips (MSWC): the word spans the clip's active
//              region (10 ms frames within 25 dB of the loudest).
//   nonspeech  noise / music recordings: no words, cut into 12 s pieces;
//              a file-hash split holds out --test-frac of the files.
//
// Output: the alignment TSV of laya_audio_data.h.
//
// Usage:
//   brosoundml_laya_audio_prep ami --dir D:/datasets/ami --out-prefix P
//       -> P_train.tsv P_dev.tsv P_test.tsv
//   brosoundml_laya_audio_prep clips --manifest M.tsv --out A.tsv
//   brosoundml_laya_audio_prep nonspeech --list files.txt --subset musan-music --out-prefix P
//       [--chunk 12] [--test-frac 0.15]  -> P_train.tsv P_test.tsv

#include "laya_audio_data.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace laya_audio;
namespace fs = std::filesystem;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "laya_audio_prep: %s\n", msg.c_str());
    std::exit(2);
}

uint32_t fnv1a(const std::string& s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) h = (h ^ c) * 16777619u;
    return h;
}

// Reference words of a span of text, each given a share of [t0, t1]
// proportional to its length (a hyphenated "t-shirt" becomes two words).
void add_words(const std::string& text, float t0, float t1, std::vector<TimedWord>& out) {
    const std::vector<std::string> ws = normalize_words(text);
    if (ws.empty()) return;
    std::size_t total = 0;
    for (const std::string& w : ws) total += w.size();
    float t = t0;
    for (const std::string& w : ws) {
        const float d = (t1 - t0) * static_cast<float>(w.size()) / static_cast<float>(total);
        out.push_back({w, t, t + d, true});
        t += d;
    }
}

std::string attr(const std::string& line, const char* name) {
    const std::string key = std::string(name) + "=\"";
    const std::size_t p = line.find(key);
    if (p == std::string::npos) return {};
    const std::size_t e = line.find('"', p + key.size());
    return line.substr(p + key.size(), e - p - key.size());
}

std::string unescape_xml(std::string s) {
    const std::pair<const char*, const char*> ents[] = {
        {"&#39;", "'"}, {"&apos;", "'"}, {"&quot;", "\""}, {"&lt;", "<"}, {"&gt;", ">"}, {"&amp;", "&"}};
    for (const auto& [from, to] : ents) {
        std::size_t p = 0;
        while ((p = s.find(from, p)) != std::string::npos) {
            s.replace(p, std::strlen(from), to);
            p += std::strlen(to);
        }
    }
    return s;
}

// ---------------------------------------------------------------- AMI

// The full-corpus-ASR partition (as in the Kaldi AMI recipe).
const std::set<std::string> kAmiDev = {"ES2011a", "ES2011b", "ES2011c", "ES2011d", "IB4001",  "IB4002",
                                       "IB4003",  "IB4004",  "IB4010",  "IB4011",  "IS1008a", "IS1008b",
                                       "IS1008c", "IS1008d", "TS3004a", "TS3004b", "TS3004c", "TS3004d"};
const std::set<std::string> kAmiEval = {"EN2002a", "EN2002b", "EN2002c", "EN2002d", "ES2004a", "ES2004b",
                                        "ES2004c", "ES2004d", "IS1009a", "IS1009b", "IS1009c", "IS1009d",
                                        "TS3003a", "TS3003b", "TS3003c", "TS3003d"};

int prep_ami(const std::string& dir, const std::string& prefix) {
    std::map<std::string, std::vector<std::pair<std::string, std::vector<TimedWord>>>> by_meeting;  // speaker -> words
    for (const auto& e : fs::directory_iterator(dir + "/words")) {
        const std::string name = e.path().filename().string();  // ES2002a.A.words.xml
        const std::size_t d1 = name.find('.');
        const std::size_t d2 = name.find('.', d1 + 1);
        if (d1 == std::string::npos || d2 == std::string::npos) continue;
        const std::string meeting = name.substr(0, d1), spk = name.substr(d1 + 1, d2 - d1 - 1);
        std::ifstream f(e.path());
        std::string line;
        std::vector<TimedWord> words;
        while (std::getline(f, line)) {
            const std::size_t w = line.find("<w ");
            if (w == std::string::npos || attr(line, "punc") == "true") continue;
            const std::string st = attr(line, "starttime"), et = attr(line, "endtime");
            if (st.empty() || et.empty()) continue;
            const std::size_t gt = line.find('>', w);
            const std::size_t close = line.find("</w>", gt);
            if (gt == std::string::npos || close == std::string::npos) continue;
            const float t0 = std::stof(st), t1 = std::stof(et);
            if (t1 <= t0) continue;
            add_words(unescape_xml(line.substr(gt + 1, close - gt - 1)), t0, t1, words);
        }
        by_meeting[meeting].push_back({meeting + "." + spk, std::move(words)});
    }
    std::vector<AlignedUtterance> sets[3];  // train, dev, test
    std::size_t n_meet = 0;
    double hours[3] = {0, 0, 0};
    for (auto& [meeting, spks] : by_meeting) {
        const std::string audio = dir + "/sdm/" + meeting + ".opus";
        if (!fs::exists(audio)) continue;
        ++n_meet;
        const int split = kAmiEval.count(meeting) ? 2 : kAmiDev.count(meeting) ? 1 : 0;
        std::vector<TimedWord> all;
        for (auto& [s, ws] : spks) all.insert(all.end(), ws.begin(), ws.end());
        std::sort(all.begin(), all.end(), [](const TimedWord& a, const TimedWord& b) { return a.t0 < b.t0; });
        // Chunk boundaries: close at a pause >= 0.6 s once >= 5 s long;
        // force a cut before the word that would push the chunk past 20 s.
        std::vector<std::pair<float, float>> chunks;
        float cs = -1, me = 0;
        for (const TimedWord& w : all) {
            if (cs < 0) {
                cs = std::max(0.0f, w.t0 - 0.3f);
                me = w.t1;
                continue;
            }
            const float gap = w.t0 - me;
            if (gap >= 0.6f && me - cs >= 5.0f) {
                const float pad = std::min(0.3f, gap / 2);
                chunks.push_back({cs, me + pad});
                cs = w.t0 - pad;
            } else if (w.t1 - cs > 20.0f && w.t0 - cs >= 5.0f) {
                const float cut = gap > 0 ? w.t0 - std::min(0.3f, gap / 2) : w.t0;
                chunks.push_back({cs, cut});
                cs = cut;
            }
            me = std::max(me, w.t1);
        }
        if (cs >= 0 && me > cs) chunks.push_back({cs, me + 0.3f});
        for (std::size_t k = 0; k < chunks.size(); ++k) {
            const auto [c0, c1] = chunks[k];
            AlignedUtterance u;
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%s_%05d", meeting.c_str(), static_cast<int>(k));
            u.id = buf;
            std::snprintf(buf, sizeof(buf), "@%.2f:%.2f", c0, c1);
            u.wav = audio + buf;
            u.speaker = meeting;
            u.subset = split == 2 ? "ami-test" : split == 1 ? "ami-dev" : "ami-train";
            u.duration_s = c1 - c0;
            for (const TimedWord& w : all) {
                if (w.t1 <= c0 || w.t0 >= c1) continue;
                u.words.push_back({w.word, w.t0 - c0, w.t1 - c0, true});
            }
            if (u.words.empty()) continue;
            hours[split] += u.duration_s / 3600.0;
            sets[split].push_back(std::move(u));
        }
    }
    const char* names[3] = {"train", "dev", "test"};
    for (int s = 0; s < 3; ++s) {
        write_alignments(prefix + "_" + names[s] + ".tsv", sets[s], false);
        std::fprintf(stderr, "ami %s: %zu chunks, %.1f h\n", names[s], sets[s].size(), hours[s]);
    }
    std::fprintf(stderr, "%zu meetings with SDM audio\n", n_meet);
    return 0;
}

// ---------------------------------------------------------------- clips

int prep_clips(const std::string& manifest, const std::string& out) {
    std::vector<AlignedUtterance> utts;
    std::size_t n_fail = 0;
    for (const Utterance& m : load_manifest(manifest)) {
        std::vector<float> pcm;
        try {
            pcm = load_audio_16k(m.wav);
        } catch (const std::exception&) {
            ++n_fail;
            continue;
        }
        const int hop = 160;
        const int n_frames = static_cast<int>(pcm.size()) / hop;
        if (n_frames < 5) {
            ++n_fail;
            continue;
        }
        std::vector<float> db(static_cast<std::size_t>(n_frames));
        float peak = -200;
        for (int f = 0; f < n_frames; ++f) {
            double e = 0;
            for (int i = 0; i < hop; ++i) e += double(pcm[f * hop + i]) * pcm[f * hop + i];
            db[f] = static_cast<float>(10.0 * std::log10(e / hop + 1e-12));
            peak = std::max(peak, db[f]);
        }
        const float thr = std::max(peak - 25.0f, -65.0f);
        int a = 0, b = n_frames - 1;
        while (a < n_frames && db[a] < thr) ++a;
        while (b > a && db[b] < thr) --b;
        AlignedUtterance u;
        u.id = m.id;
        u.wav = m.wav;
        u.speaker = m.speaker;
        u.subset = m.subset;
        u.duration_s = static_cast<float>(pcm.size()) / 16000.0f;
        add_words(m.text, a * 0.01f, (b + 1) * 0.01f, u.words);
        if (u.words.empty()) continue;
        utts.push_back(std::move(u));
        if (utts.size() % 5000 == 0) std::fprintf(stderr, "%zu clips\n", utts.size());
    }
    write_alignments(out, utts, false);
    std::fprintf(stderr, "clips: %zu written, %zu unreadable or too short\n", utts.size(), n_fail);
    return 0;
}

// ---------------------------------------------------------------- nonspeech

int prep_nonspeech(const std::string& list, const std::string& subset, const std::string& prefix, float chunk,
                   float test_frac) {
    std::ifstream f(list);
    if (!f) die("cannot open " + list);
    std::vector<AlignedUtterance> sets[2];
    double hours[2] = {0, 0};
    std::string path;
    while (std::getline(f, path)) {
        if (!path.empty() && path.back() == '\r') path.pop_back();
        if (path.empty()) continue;
        const std::vector<float> pcm = load_audio_16k(path);
        const float dur = static_cast<float>(pcm.size()) / 16000.0f;
        const int split = (fnv1a(fs::path(path).filename().string()) % 1000u) < test_frac * 1000 ? 1 : 0;
        for (int k = 0; k * chunk < dur; ++k) {
            const float c0 = k * chunk, c1 = std::min(dur, c0 + chunk);
            if (c1 - c0 < 1.0f) break;
            AlignedUtterance u;
            char buf[64];
            u.id = fs::path(path).stem().string() + "_" + std::to_string(k);
            std::snprintf(buf, sizeof(buf), "@%.2f:%.2f", c0, c1);
            u.wav = path + buf;
            u.speaker = subset;
            u.subset = subset + (split ? "-test" : "-train");
            u.duration_s = c1 - c0;
            hours[split] += u.duration_s / 3600.0;
            sets[split].push_back(std::move(u));
        }
    }
    write_alignments(prefix + "_train.tsv", sets[0], false);
    write_alignments(prefix + "_test.tsv", sets[1], false);
    std::fprintf(stderr, "%s: train %zu pieces %.2f h, test %zu pieces %.2f h\n", subset.c_str(), sets[0].size(),
                 hours[0], sets[1].size(), hours[1]);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) die("usage: brosoundml_laya_audio_prep ami|clips|nonspeech ...");
    const std::string mode = argv[1];
    std::string dir, out, out_prefix, manifest, list, subset;
    float chunk = 12.0f, test_frac = 0.15f;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--dir") dir = next();
        else if (a == "--out") out = next();
        else if (a == "--out-prefix") out_prefix = next();
        else if (a == "--manifest") manifest = next();
        else if (a == "--list") list = next();
        else if (a == "--subset") subset = next();
        else if (a == "--chunk") chunk = std::stof(next());
        else if (a == "--test-frac") test_frac = std::stof(next());
        else die("unknown argument " + a);
    }
    try {
        if (mode == "ami") {
            if (dir.empty() || out_prefix.empty()) die("ami needs --dir and --out-prefix");
            return prep_ami(dir, out_prefix);
        }
        if (mode == "clips") {
            if (manifest.empty() || out.empty()) die("clips needs --manifest and --out");
            return prep_clips(manifest, out);
        }
        if (mode == "nonspeech") {
            if (list.empty() || subset.empty() || out_prefix.empty()) die("nonspeech needs --list --subset --out-prefix");
            return prep_nonspeech(list, subset, out_prefix, chunk, test_frac);
        }
        die("unknown mode " + mode);
    } catch (const std::exception& e) {
        die(e.what());
    }
}
