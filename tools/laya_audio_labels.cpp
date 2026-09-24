// brosoundml_laya_audio_labels — real labels for the free-form question
// evaluation (docs/laya-audio.md, "Free-form questions"), plus manifests for
// corpora that come with a transcript but no word timings (the Parakeet
// aligner, brosoundml_laya_audio_align, times them).
//
// Label file (laya_audio_questions.h): one tagged span per line,
//   utt_id \t t0 \t t1 \t tag
// in the utterance's own time; t0 < 0 means the whole utterance (its speech
// extent once aligned).
//
// Modes:
//   slurp   SLURP jsonl -> manifest (id, first recording, text) + labels
//           intent=<intent> scenario=<scenario> entity=<type> (utterance-level).
//           EVAL ONLY: SLURP audio is CC BY-NC 4.0.
//   ami     AMI manual annotations -> labels on the chunks of an AMI alignment
//           file: da=<type> daclass=<class> (dialogue acts), ne=<type> and
//           ne=<parent class> (named entities), topic=<scenario topic>, each
//           a span over the annotated words' times.
//   cremad  CREMA-D -> manifest (clip, fixed sentence text) + labels
//           emotion=<ANG|DIS|FEA|HAP|NEU|SAD> intensity=<LO|MD|HI|XX>
//           voice=<crowd audio-only majority emotion>. EVAL ONLY (ODbL).
//   timers  Timers and Such CSVs (path,semantics,speakerId,...,formatted_transcription)
//           -> manifest + labels intent=<SetTimer|SetAlarm|SimpleMath|UnitConversion>.
//   zipdir  list a zip's central directory (read from a file holding it and
//           the end records): local header offset, compressed size, name.
//
// Usage:
//   brosoundml_laya_audio_labels slurp --jsonl test.jsonl --audio-dir DIR --subset slurp-test --manifest M.tsv --out L.tsv
//   brosoundml_laya_audio_labels ami --dir D:/datasets/ami --align align_ami_test.tsv --out L.tsv
//   brosoundml_laya_audio_labels cremad --dir D:/datasets/cremad --manifest M.tsv --out L.tsv
//   brosoundml_laya_audio_labels timers --csv test-real.csv --audio-dir DIR --subset timers-test --manifest M.tsv --out L.tsv
//   brosoundml_laya_audio_labels zipdir --cd tail.bin

#include "laya_audio_data.h"

#include <brolm/detail/json.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace laya_audio;
namespace fs = std::filesystem;
namespace json = brolm::detail::json;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "laya_audio_labels: %s\n", msg.c_str());
    std::exit(2);
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream is(s);
    while (std::getline(is, cur, sep)) out.push_back(cur);
    return out;
}

// A CSV line with double-quoted fields (commas inside quotes).
std::vector<std::string> csv_fields(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool q = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (q) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                cur += '"';
                ++i;
            } else if (c == '"') {
                q = false;
            } else {
                cur += c;
            }
        } else if (c == '"') {
            q = true;
        } else if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

std::string attr(const std::string& line, const char* name) {
    const std::string key = std::string(name) + "=\"";
    const std::size_t p = line.find(key);
    if (p == std::string::npos) return {};
    const std::size_t e = line.find('"', p + key.size());
    return line.substr(p + key.size(), e - p - key.size());
}

// "#id(X)" -> X
std::string href_id(const std::string& s, std::size_t from = 0) {
    const std::size_t a = s.find("id(", from);
    if (a == std::string::npos) return {};
    const std::size_t b = s.find(')', a);
    return s.substr(a + 3, b - a - 3);
}

struct LabelRow {
    std::string id;
    float t0 = -1, t1 = -1;
    std::string tag;
};

void write_labels(const std::string& path, const std::vector<LabelRow>& rows) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) die("cannot write " + path);
    for (const LabelRow& r : rows) std::fprintf(f, "%s\t%.3f\t%.3f\t%s\n", r.id.c_str(), r.t0, r.t1, r.tag.c_str());
    std::fclose(f);
}

struct ManifestRow {
    std::string id, audio, speaker, subset, text;
};

void write_manifest(const std::string& path, const std::vector<ManifestRow>& rows) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) die("cannot write " + path);
    for (const ManifestRow& r : rows)
        std::fprintf(f, "%s\t%s\t%s\t%s\t%s\n", r.id.c_str(), r.audio.c_str(), r.speaker.c_str(), r.subset.c_str(),
                     r.text.c_str());
    std::fclose(f);
}

// ---------------------------------------------------------------- slurp

int slurp(const std::string& jsonl, const std::string& audio_dir, const std::string& subset,
          const std::string& manifest, const std::string& out) {
    std::ifstream f(jsonl);
    if (!f) die("cannot open " + jsonl);
    std::vector<ManifestRow> man;
    std::vector<LabelRow> lab;
    std::string line;
    int missing = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const json::Value v = json::parse(line);
        const auto& recs = v.at("recordings").as_array();
        if (recs.empty()) continue;
        const std::string file = recs.front().at("file").as_string();
        const std::string path = audio_dir + "/" + file;
        if (!fs::exists(path)) {
            ++missing;
            continue;
        }
        const std::string id = "slurp" + std::to_string(static_cast<long long>(v.at("slurp_id").as_number()));
        man.push_back({id, path, id, subset, v.at("sentence").as_string()});
        lab.push_back({id, -1, -1, "intent=" + v.at("intent").as_string()});
        lab.push_back({id, -1, -1, "scenario=" + v.at("scenario").as_string()});
        std::set<std::string> ents;
        if (const json::Value* es = v.find("entities"))
            for (const json::Value& e : es->as_array()) ents.insert(e.at("type").as_string());
        for (const std::string& e : ents) lab.push_back({id, -1, -1, "entity=" + e});
    }
    write_manifest(manifest, man);
    write_labels(out, lab);
    std::fprintf(stderr, "slurp: %zu utterances, %zu labels, %d recordings missing\n", man.size(), lab.size(), missing);
    return 0;
}

// ---------------------------------------------------------------- ami

struct AmiWord {
    float t0 = -1, t1 = -1;
};

// Per speaker words file: element id -> (document index, times). Every
// element carrying a nite:id counts (words, punctuation, vocal sounds), so a
// range id(a)..id(b) is the document-order run between them.
struct AmiWords {
    std::unordered_map<std::string, int> index;
    std::vector<AmiWord> items;
    // Times of the run a..b (the first and last timed element inside it).
    bool span(const std::string& a, const std::string& b, float& t0, float& t1) const {
        const auto ia = index.find(a), ib = index.find(b.empty() ? a : b);
        if (ia == index.end() || ib == index.end()) return false;
        t0 = -1;
        t1 = -1;
        for (int i = ia->second; i <= ib->second; ++i) {
            if (items[static_cast<std::size_t>(i)].t0 < 0) continue;
            if (t0 < 0) t0 = items[static_cast<std::size_t>(i)].t0;
            t1 = std::max(t1, items[static_cast<std::size_t>(i)].t1);
        }
        return t0 >= 0 && t1 > t0;
    }
};

AmiWords load_ami_words(const std::string& path) {
    AmiWords w;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        const std::string id = attr(line, "nite:id");
        if (id.empty()) continue;
        AmiWord aw;
        const std::string st = attr(line, "starttime"), et = attr(line, "endtime");
        if (!st.empty() && !et.empty()) {
            aw.t0 = std::stof(st);
            aw.t1 = std::stof(et);
        }
        w.index[id] = static_cast<int>(w.items.size());
        w.items.push_back(aw);
    }
    return w;
}

// Ontology file: nite:id -> name, and nite:id -> parent's name.
void load_ontology(const std::string& path, std::map<std::string, std::string>& name,
                   std::map<std::string, std::string>& parent) {
    std::ifstream f(path);
    std::string line;
    std::vector<std::string> stack;
    while (std::getline(f, line)) {
        const std::string id = attr(line, "nite:id");
        const bool closes_only = line.find("</") != std::string::npos && id.empty();
        if (closes_only) {
            if (!stack.empty()) stack.pop_back();
            continue;
        }
        if (id.empty()) continue;
        const std::string nm = attr(line, "name");
        name[id] = nm;
        if (!stack.empty()) parent[id] = name[stack.back()];
        const bool self_closing = line.find("/>") != std::string::npos;
        if (!self_closing) stack.push_back(id);
    }
}

// Spans of one annotation file: each element with a type pointer and word
// children ("href=...words.xml#id(a)..id(b)"), meeting time.
struct Span {
    float t0, t1;
    std::string type_id;
};

std::vector<Span> load_spans(const std::string& path, const char* element, const std::string& words_dir,
                             std::map<std::string, AmiWords>& words_cache) {
    std::vector<Span> out;
    std::ifstream f(path);
    std::string line, type_id;
    float t0 = -1, t1 = -1;
    bool in = false;
    const std::string open = std::string("<") + element;
    const std::string close = std::string("</") + element;
    while (std::getline(f, line)) {
        if (line.find(open) != std::string::npos) {
            in = true;
            type_id.clear();
            t0 = t1 = -1;
            continue;
        }
        if (!in) continue;
        if (line.find("<nite:pointer") != std::string::npos) {
            type_id = href_id(line);
        } else if (line.find("<nite:child") != std::string::npos) {
            const std::string href = attr(line, "href");
            const std::size_t hash = href.find('#');
            if (hash == std::string::npos) continue;
            const std::string file = href.substr(0, hash);
            const std::string a = href_id(href, hash);
            const std::size_t dots = href.find("..", hash);
            const std::string b = dots == std::string::npos ? a : href_id(href, dots);
            auto it = words_cache.find(file);
            if (it == words_cache.end()) it = words_cache.emplace(file, load_ami_words(words_dir + "/" + file)).first;
            float s0, s1;
            if (it->second.span(a, b, s0, s1)) {
                t0 = t0 < 0 ? s0 : std::min(t0, s0);
                t1 = std::max(t1, s1);
            }
        } else if (line.find(close) != std::string::npos) {
            in = false;
            if (!type_id.empty() && t0 >= 0) out.push_back({t0, t1, type_id});
        }
    }
    return out;
}

int ami(const std::string& dir, const std::string& align, const std::string& out) {
    std::map<std::string, std::string> da_name, da_parent, ne_name, ne_parent, top_name, top_parent;
    load_ontology(dir + "/ontologies/da-types.xml", da_name, da_parent);
    load_ontology(dir + "/ontologies/ne-types.xml", ne_name, ne_parent);
    load_ontology(dir + "/ontologies/default-topics.xml", top_name, top_parent);
    const std::vector<AlignedUtterance> chunks = read_alignments(align);
    std::set<std::string> meetings;
    for (const AlignedUtterance& u : chunks) meetings.insert(u.speaker);
    // meeting -> (t0, t1, tag), meeting time
    std::map<std::string, std::vector<LabelRow>> spans;
    std::map<std::string, AmiWords> words;
    int n_da = 0, n_ne = 0, n_top = 0;
    for (const auto& e : fs::directory_iterator(dir + "/dialogueActs")) {
        const std::string fn = e.path().filename().string();
        const std::string m = fn.substr(0, fn.find('.'));
        if (!meetings.count(m) || fn.find(".dialog-act.xml") == std::string::npos) continue;
        for (const Span& s : load_spans(e.path().string(), "dact", dir + "/words", words)) {
            spans[m].push_back({m, s.t0, s.t1, "da=" + da_name[s.type_id]});
            spans[m].push_back({m, s.t0, s.t1, "daclass=" + da_parent[s.type_id]});
            ++n_da;
        }
    }
    for (const auto& e : fs::directory_iterator(dir + "/namedEntities")) {
        const std::string fn = e.path().filename().string();
        const std::string m = fn.substr(0, fn.find('.'));
        if (!meetings.count(m)) continue;
        for (const Span& s : load_spans(e.path().string(), "named-entity", dir + "/words", words)) {
            // Tag the type and every ancestor class (MONEY -> NUMEX).
            std::string id = s.type_id;
            spans[m].push_back({m, s.t0, s.t1, "ne=" + ne_name[id]});
            const auto p = ne_parent.find(id);
            if (p != ne_parent.end() && p->second != "ne-root") spans[m].push_back({m, s.t0, s.t1, "ne=" + p->second});
            ++n_ne;
        }
    }
    for (const auto& e : fs::directory_iterator(dir + "/topics")) {
        const std::string fn = e.path().filename().string();
        const std::string m = fn.substr(0, fn.find('.'));
        if (!meetings.count(m)) continue;
        // Nested topics: only the leaf's word children are listed per topic
        // element, so each element is its own span.
        for (const Span& s : load_spans(e.path().string(), "topic", dir + "/words", words)) {
            spans[m].push_back({m, s.t0, s.t1, "topic=" + top_name[s.type_id]});
            ++n_top;
        }
    }
    std::vector<LabelRow> rows;
    for (const AlignedUtterance& u : chunks) {
        // Chunk audio "path@c0:c1": meeting time c0 is chunk time 0.
        const std::size_t at = u.wav.rfind('@');
        if (at == std::string::npos) continue;
        const float c0 = std::stof(u.wav.substr(at + 1));
        const float c1 = c0 + u.duration_s;
        const auto it = spans.find(u.speaker);
        if (it == spans.end()) continue;
        for (const LabelRow& s : it->second) {
            if (s.t1 <= c0 || s.t0 >= c1) continue;
            rows.push_back({u.id, s.t0 - c0, s.t1 - c0, s.tag});
        }
    }
    write_labels(out, rows);
    std::fprintf(stderr, "ami: %zu chunks, %d dialogue acts, %d named entities, %d topic segments, %zu label rows "
                 "(%zu meetings annotated)\n", chunks.size(), n_da, n_ne, n_top, rows.size(), spans.size());
    return 0;
}

// ---------------------------------------------------------------- cremad

int cremad(const std::string& dir, const std::string& manifest, const std::string& out) {
    const std::map<std::string, std::string> sentence = {
        {"IEO", "It's eleven o'clock."},
        {"TIE", "That is exactly what happened."},
        {"IOM", "I'm on my way to the meeting."},
        {"IWW", "I wonder what this is about."},
        {"TAI", "The airplane is almost full."},
        {"MTI", "Maybe tomorrow it will be cold."},
        {"IWL", "I would like a new alarm clock."},
        {"ITH", "I think I have a doctor's appointment."},
        {"DFA", "Don't forget a jacket."},
        {"ITS", "I think I've seen this before."},
        {"TSI", "The surface is slick."},
        {"WSI", "We'll stop in a couple of minutes."}};
    // Crowd audio-only majority vote per clip (summaryTable.csv: ...,FileName,VoiceVote,...).
    std::map<std::string, std::string> voice;
    {
        std::ifstream f(dir + "/summaryTable.csv");
        std::string line;
        int col_file = -1, col_voice = -1;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const std::vector<std::string> c = csv_fields(line);
            if (col_file < 0) {
                for (std::size_t i = 0; i < c.size(); ++i) {
                    if (c[i] == "FileName") col_file = static_cast<int>(i);
                    if (c[i] == "VoiceVote") col_voice = static_cast<int>(i);
                }
                continue;
            }
            if (col_voice < 0 || static_cast<int>(c.size()) <= std::max(col_file, col_voice)) continue;
            voice[c[static_cast<std::size_t>(col_file)]] = c[static_cast<std::size_t>(col_voice)];
        }
    }
    const std::map<std::string, std::string> vote_code = {{"A", "ANG"}, {"D", "DIS"}, {"F", "FEA"},
                                                          {"H", "HAP"}, {"N", "NEU"}, {"S", "SAD"}};
    std::vector<ManifestRow> man;
    std::vector<LabelRow> lab;
    for (const auto& e : fs::directory_iterator(dir + "/wav")) {
        const std::string stem = e.path().stem().string();  // 1001_DFA_ANG_XX
        const std::vector<std::string> p = split(stem, '_');
        if (p.size() != 4 || !sentence.count(p[1]) || fs::file_size(e.path()) < 4000) continue;
        man.push_back({stem, e.path().generic_string(), p[0], "cremad", sentence.at(p[1])});
        lab.push_back({stem, -1, -1, "emotion=" + p[2]});
        lab.push_back({stem, -1, -1, "intensity=" + p[3]});
        const auto v = voice.find(stem);
        if (v != voice.end()) {
            const auto code = vote_code.find(v->second);
            lab.push_back({stem, -1, -1, "voice=" + (code != vote_code.end() ? code->second : v->second)});
        }
    }
    write_manifest(manifest, man);
    write_labels(out, lab);
    std::fprintf(stderr, "cremad: %zu clips, %zu with a crowd vote\n", man.size(), voice.size());
    return 0;
}

// ---------------------------------------------------------------- timers

int timers(const std::string& csv, const std::string& audio_dir, const std::string& subset,
           const std::string& manifest, const std::string& out) {
    std::ifstream f(csv);
    if (!f) die("cannot open " + csv);
    std::string line;
    std::vector<std::string> head;
    int c_id = -1, c_wav = -1, c_text = -1, c_sem = -1, c_spk = -1;
    std::vector<ManifestRow> man;
    std::vector<LabelRow> lab;
    int missing = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::vector<std::string> c = csv_fields(line);
        if (head.empty()) {
            head = c;
            // path,semantics,speakerId,transcription,formatted_transcription,...
            // The formatted transcription spells numbers out, as they are spoken.
            for (std::size_t i = 0; i < c.size(); ++i) {
                if (c[i] == "path") c_wav = static_cast<int>(i);
                if (c[i] == "formatted_transcription") c_text = static_cast<int>(i);
                if (c[i] == "semantics") c_sem = static_cast<int>(i);
                if (c[i] == "speakerId") c_spk = static_cast<int>(i);
            }
            c_id = c_wav;
            if (c_wav < 0 || c_text < 0 || c_sem < 0) die("unexpected CSV header in " + csv);
            continue;
        }
        if (static_cast<int>(c.size()) < static_cast<int>(head.size())) continue;
        const std::string wav = c[static_cast<std::size_t>(c_wav)];  // relative to the dataset root
        const std::string path = audio_dir + "/" + wav;
        if (!fs::exists(path)) {
            ++missing;
            continue;
        }
        const std::string sem = c[static_cast<std::size_t>(c_sem)];
        const std::size_t ip = sem.find("intent");
        std::string intent;
        if (ip != std::string::npos) {
            const std::size_t a = sem.find_first_of("'\"", sem.find(':', ip) + 1);
            const std::size_t b = sem.find_first_of("'\"", a + 1);
            if (a != std::string::npos && b != std::string::npos) intent = sem.substr(a + 1, b - a - 1);
        }
        const std::string id = "timers_" + fs::path(wav).stem().string();
        man.push_back({id, path, c_spk >= 0 ? c[static_cast<std::size_t>(c_spk)] : id, subset,
                       c[static_cast<std::size_t>(c_text)]});
        if (!intent.empty()) lab.push_back({id, -1, -1, "intent=" + intent});
    }
    write_manifest(manifest, man);
    write_labels(out, lab);
    std::fprintf(stderr, "timers: %zu utterances, %d audio files missing\n", man.size(), missing);
    return 0;
}

// ---------------------------------------------------------------- zipdir

uint64_t rd(const unsigned char* p, int n) {
    uint64_t v = 0;
    for (int i = n - 1; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

int zipdir(const std::string& path) {
    const std::string s = read_file(path);
    const auto* b = reinterpret_cast<const unsigned char*>(s.data());
    std::size_t n = 0;
    for (std::size_t i = 0; i + 46 <= s.size();) {
        if (rd(b + i, 4) != 0x02014b50u) {
            ++i;
            if (n) break;  // entries are contiguous once found
            continue;
        }
        uint64_t csize = rd(b + i + 20, 4), usize = rd(b + i + 24, 4), off = rd(b + i + 42, 4);
        const std::size_t nlen = rd(b + i + 28, 2), xlen = rd(b + i + 30, 2), clen = rd(b + i + 32, 2);
        const std::string name = s.substr(i + 46, nlen);
        for (std::size_t x = i + 46 + nlen; x + 4 <= i + 46 + nlen + xlen;) {
            const uint64_t id = rd(b + x, 2), sz = rd(b + x + 2, 2);
            if (id == 1) {
                std::size_t q = x + 4;
                if (usize == 0xffffffffu) { usize = rd(b + q, 8); q += 8; }
                if (csize == 0xffffffffu) { csize = rd(b + q, 8); q += 8; }
                if (off == 0xffffffffu) off = rd(b + q, 8);
            }
            x += 4 + sz;
        }
        std::printf("%llu\t%llu\t%s\n", static_cast<unsigned long long>(off), static_cast<unsigned long long>(csize),
                    name.c_str());
        ++n;
        i += 46 + nlen + xlen + clen;
    }
    std::fprintf(stderr, "zipdir: %zu entries\n", n);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) die("usage: brosoundml_laya_audio_labels slurp|ami|cremad|timers|zipdir ...");
    const std::string mode = argv[1];
    std::string jsonl, audio_dir, subset, manifest, out, dir, align, csv, cd;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--jsonl") jsonl = next();
        else if (a == "--audio-dir") audio_dir = next();
        else if (a == "--subset") subset = next();
        else if (a == "--manifest") manifest = next();
        else if (a == "--out") out = next();
        else if (a == "--dir") dir = next();
        else if (a == "--align") align = next();
        else if (a == "--csv") csv = next();
        else if (a == "--cd") cd = next();
        else die("unknown argument " + a);
    }
    try {
        if (mode == "slurp") return slurp(jsonl, audio_dir, subset.empty() ? "slurp-test" : subset, manifest, out);
        if (mode == "ami") return ami(dir, align, out);
        if (mode == "cremad") return cremad(dir, manifest, out);
        if (mode == "timers") return timers(csv, audio_dir, subset, manifest, out);
        if (mode == "zipdir") return zipdir(cd);
        die("unknown mode " + mode);
    } catch (const std::exception& e) {
        die(e.what());
    }
}
