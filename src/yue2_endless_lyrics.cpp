// Yue2Endless: rolled lyrics (endless.js lyricsRoll, adaptive): each chunk's
// prefix carries a window of the sheet's lines after the last line the
// listener heard in the rendered chunks, and the listener (Whisper) run on
// each rendered chunk.

#include "yue2_endless_impl.h"

#include "brosoundml/whisper.h"

#include <algorithm>
#include <set>

namespace brosoundml {

using namespace endless;

namespace {

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

bool is_tag(const std::string& l) { return !l.empty() && l[0] == '['; }

// endless.js words(): lower case, ’ as ', the runs of [a-z'].
std::vector<std::string> words_of(const std::string& text) {
    std::string t;
    t.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == 0xE2 && i + 2 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0x80 &&
            static_cast<unsigned char>(text[i + 2]) == 0x99) {
            t.push_back('\'');
            i += 2;
        } else {
            t.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : static_cast<char>(c));
        }
    }
    std::vector<std::string> out;
    std::string w;
    for (char c : t) {
        if ((c >= 'a' && c <= 'z') || c == '\'') w.push_back(c);
        else if (!w.empty()) out.push_back(std::move(w)), w.clear();
    }
    if (!w.empty()) out.push_back(std::move(w));
    return out;
}

void load_sheet(Roll& r, const std::string& sheet) {
    r = Roll{};
    r.sheet = sheet;
    std::size_t at = 0;
    while (at <= sheet.size()) {
        const std::size_t eol = sheet.find('\n', at);
        std::string line = sheet.substr(at, eol == std::string::npos ? std::string::npos : eol - at);
        at = eol == std::string::npos ? sheet.size() + 1 : eol + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        bool blank = true;
        for (char c : line) blank = blank && (c == ' ' || c == '\t');
        if (blank) continue;
        r.lines.push_back(line);
    }
    for (int i = 0; i < static_cast<int>(r.lines.size()); ++i)
        if (!is_tag(r.lines[i])) {
            r.sung.push_back(i);
            r.words.push_back(words_of(r.lines[i]));
        }
}

std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && static_cast<unsigned char>(s[a]) <= ' ') ++a;
    while (b > a && static_cast<unsigned char>(s[b - 1]) <= ' ') --b;
    return s.substr(a, b - a);
}

}  // namespace

std::string Yue2Endless::Impl::lyrics_of(Line& l, int k, const Yue2EndlessTarget& t, Roll& roll, int* from,
                                         int* to) const {
    (void)l;
    *from = *to = -1;
    if (t.lyrics.empty()) return kYue2Instrumental;
    if (!t.roll) return t.lyrics;
    if (roll.sheet != t.lyrics || roll.lines.empty()) {
        const bool fresh = roll.sheet != t.lyrics;
        Roll keep = roll;
        load_sheet(roll, t.lyrics);
        if (fresh) roll.moved = k;
        else roll.at = keep.at, roll.moved = keep.moved, roll.last_heard = keep.last_heard;
    }
    if (roll.sung.empty()) return t.lyrics;
    const int next = std::max(roll.at, roll.last_heard + 1);
    if (next > roll.at) {
        roll.at = next;
        roll.moved = k;
    } else if (k - roll.moved >= 3) {
        ++roll.at;
        roll.moved = k;
    }
    const int n = static_cast<int>(roll.sung.size());
    const int a = std::min(roll.at, n - 1);
    const int b = std::min(n, a + t.lyric_window);
    const int i0 = roll.sung[a], i1 = roll.sung[b - 1] + 1;
    const std::string* tag = nullptr;
    for (int i = i0; i >= 0 && !tag; --i)
        if (is_tag(roll.lines[i])) tag = &roll.lines[i];
    std::vector<std::string> body(roll.lines.begin() + i0, roll.lines.begin() + i1);
    if (tag && body[0] != *tag) body.insert(body.begin(), *tag);
    // body.join('\n').replace(/\n(\[)/g, '\n\n$1') + '\n'
    std::string out;
    for (std::size_t i = 0; i < body.size(); ++i) {
        if (i) out += is_tag(body[i]) ? "\n\n" : "\n";
        out += body[i];
    }
    *from = a;
    *to = b;
    return out + "\n";
}

void Yue2Endless::Impl::hear(Line& l, int k) {
    Chunk& c = l.chunk(k);
    if (!c.target.roll || c.target.lyrics.empty() || !opt.listen || c.row.lyric_from < 0) return;
    if (c.target.lyrics != l.roll.sheet) return;   // the roll has moved on to another sheet
    const auto t0 = Clock::now();
    // endless.js: the chunk decoded with 12 frames of halo on its left, its
    // samples averaged 3:1 from 48 kHz stereo to 16 kHz mono.
    const int c0 = k * C, c1 = c0 + C, h0 = std::max(0, c0 - kHalo);
    work(l.id ? "branch: listen" : "listen", k);
    in_decode = true;
    const StereoAudio d = vae.decode(l.lat_at(h0), c1 - h0);
    in_decode = false;
    service();
    const int n = d.frames, o = (c0 - h0) * kR, len = std::min(C * kR, n - o);
    AudioBuffer x;
    x.sample_rate = 16000;
    x.samples.resize(static_cast<std::size_t>(std::max(0, len / 3)));
    const float* Lc = d.channel(0);
    const float* Rc = d.channel(1);
    for (std::size_t i = 0; i < x.samples.size(); ++i) {
        double v = 0;
        for (int j = 0; j < 3; ++j) v += static_cast<double>(Lc[o + 3 * i + j]) + Rc[o + 3 * i + j];
        x.samples[i] = static_cast<float>(v / 6);
    }
    const std::string text = opt.listen(x, servicing());
    const auto bag_v = words_of(text);
    const std::set<std::string> bag(bag_v.begin(), bag_v.end());
    std::vector<int> heard;
    for (int i = 0; i < static_cast<int>(l.roll.words.size()); ++i) {
        const auto& w = l.roll.words[i];
        if (w.empty()) continue;
        int hit = 0;
        for (const auto& word : w) hit += bag.count(word) ? 1 : 0;
        if (static_cast<double>(hit) / static_cast<double>(w.size()) >= 0.6) heard.push_back(i);
    }
    // The furthest line heard in or just past this chunk's window (a line far
    // behind is a repeat, not progress).
    const int a = c.row.lyric_from, b = c.row.lyric_to;
    for (int i : heard)
        if (i >= a - 1 && i < b + 2) l.roll.last_heard = std::max(l.roll.last_heard, i);
    c.row.heard = text;
    c.row.heard_lines = heard;
    c.row.listen_seconds = since(t0);
    est_listen = 0.7 * est_listen + 0.3 * c.row.listen_seconds;
    Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Heard};
    e.chunk = k;
    e.branch = l.id;
    e.tag = c.row.tag;
    e.text = text;
    e.value = l.roll.last_heard;
    e.sample = static_cast<std::int64_t>(k) * C * kR;
    e.row = std::make_shared<Yue2EndlessChunk>(c.row);
    event(e);
}

Yue2Listen Yue2Endless::whisper_listener(const Whisper& whisper, std::vector<int32_t> prompt_ids,
                                         std::function<std::string(const std::vector<int32_t>&)> decode,
                                         int max_new_tokens) {
    return [&whisper, prompt = std::move(prompt_ids), decode = std::move(decode), max_new_tokens](
               const AudioBuffer& audio, const CancelCheck& cancel) -> std::string {
        Whisper::TranscribeOptions o;
        o.max_new_tokens = max_new_tokens;
        o.cancel = cancel;
        const auto t = whisper.transcribe(audio, prompt, o);
        return trim(decode(t.token_ids));
    };
}

}  // namespace brosoundml
