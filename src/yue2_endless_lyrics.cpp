// Yue2Endless: rolled lyrics. Each chunk's prefix carries a window of the
// sheet's lines after the furthest line the model reached, and where the
// model is in the lyrics is read off its own attention as the chunk is
// sampled: head 14.10 points at the lyric word being sung (yue2-research
// FINDINGS 11; lyric_track.py holds the reference definitions).

#include "yue2_endless_impl.h"

#include <algorithm>
#include <stdexcept>

namespace brosoundml {

using namespace endless;

namespace {

bool is_tag(const std::string& l) { return !l.empty() && l[0] == '['; }

bool blank(const std::string& l) {
    for (char c : l)
        if (c != ' ' && c != '\t' && c != '\r' && c != '\v' && c != '\f') return false;
    return true;
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
        if (blank(line)) continue;
        r.lines.push_back(line);
    }
    for (int i = 0; i < static_cast<int>(r.lines.size()); ++i)
        if (!is_tag(r.lines[i])) r.sung.push_back(i);
}

// The UTF-8 right single quote, read as an apostrophe.
bool rsquo_at(const std::string& s, std::size_t i) {
    return i + 2 < s.size() &&static_cast<unsigned char>(s[i]) == 0xE2 &&
           static_cast<unsigned char>(s[i + 1]) == 0x80 && static_cast<unsigned char>(s[i + 2]) == 0x99;
}

}  // namespace

std::string Yue2Endless::Impl::lyrics_of(const Yue2EndlessTarget& t, Roll& roll, int* from, int* to) const {
    *from = *to = -1;
    if (t.lyrics.empty()) return kYue2Instrumental;
    if (!t.roll) return t.lyrics;
    if (roll.sheet != t.lyrics || roll.lines.empty()) {
        const bool fresh = roll.sheet != t.lyrics;
        const Roll keep = roll;
        load_sheet(roll, t.lyrics);
        if (!fresh) roll.at = keep.at, roll.reached = keep.reached;
    }
    if (roll.sung.empty()) return t.lyrics;
    roll.at = std::max(roll.at, roll.reached + 1);
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

// lyric_track.py lyric_word_spans, on bytes: the prefix's text is its
// tokens' pieces end to end, the lyrics follow "\n[Lyrics]\n" in it, a
// sung line is one with text that is not a [section] tag, a word is a run of
// [a-z'] after lower-casing (’ read as '), and a word's columns are the
// tokens from the one holding its first byte through the one holding its last.
LyricWords Yue2Endless::Impl::lyric_words(const Yue2Plan& plan, const std::string& lyrics, int from) const {
    LyricWords out;
    if (from < 0) return out;
    std::string text;
    std::vector<std::size_t> ends;
    ends.reserve(plan.prefix.size());
    for (int32_t id : plan.prefix) {
        text += model.token_piece(id);
        ends.push_back(text.size());
    }
    const std::string head = "\n[Lyrics]\n";
    const std::size_t at = text.find(head + lyrics);
    if (at == std::string::npos) throw std::runtime_error("brosoundml: Yue2Endless: the lyrics are not in the prompt");
    std::size_t pos = at + head.size();
    int line = -1;
    std::size_t s = 0;
    while (s <= lyrics.size()) {
        std::size_t e = lyrics.find('\n', s);
        if (e == std::string::npos) e = lyrics.size();
        const std::string l = lyrics.substr(s, e - s);
        if (!blank(l) && !is_tag(l)) {
            ++line;
            for (std::size_t i = 0; i < l.size();) {
                std::string w;
                const std::size_t w0 = i;
                while (i < l.size()) {
                    const unsigned char c = static_cast<unsigned char>(l[i]);
                    if (c >= 'a' && c <= 'z') w.push_back(static_cast<char>(c)), ++i;
                    else if (c >= 'A' && c <= 'Z') w.push_back(static_cast<char>(c - 'A' + 'a')), ++i;
                    else if (c == '\'') w.push_back('\''), ++i;
                    else if (rsquo_at(l, i)) w.push_back('\''), i += 3;
                    else break;
                }
                if (w.empty()) {
                    ++i;
                    continue;
                }
                const std::size_t a = pos + w0, b = pos + i;
                const int t0 = static_cast<int>(std::upper_bound(ends.begin(), ends.end(), a) - ends.begin());
                const int t1 = static_cast<int>(std::lower_bound(ends.begin(), ends.end(), b) - ends.begin()) + 1;
                out.words.push_back(std::move(w));
                out.lines.push_back(from + line);
                out.cols.emplace_back(t0, t1);
            }
        }
        pos += e - s + 1;
        s = e + 1;
    }
    return out;
}

void Yue2Endless::Impl::watch_lyrics(Yue2Splice& sp, const LyricWords& w) {
    if (w.cols.empty()) return;
    sp.watch_layer = kLyricLayer;
    sp.watch_head = kLyricHead;
    sp.watch_spans = w.cols;
}

void Yue2Endless::Impl::read_lyrics(Roll& roll, Chunk& c, const LyricWords& w, const std::vector<float>& watched,
                                    int row0) const {
    const int W = static_cast<int>(w.cols.size());
    if (W == 0 || watched.size() < static_cast<std::size_t>(row0 + C) * W) return;
    Yue2EndlessChunk& r = c.row;
    r.lyric_words = w.words;
    r.lyric_word_lines = w.lines;
    r.pointer.assign(C, 0);
    r.pointer_mass.assign(C, 0.0f);
    r.lines_reached.clear();
    std::vector<int> frames(W, 0);   // gated frames each word was pointed at
    for (int f = 0; f < C; ++f) {
        const float* m = watched.data() + static_cast<std::size_t>(row0 + f) * W;
        int best = 0;
        float sum = 0;
        for (int i = 0; i < W; ++i) {
            sum += m[i];
            if (m[i] > m[best]) best = i;
        }
        r.pointer[f] = best;
        r.pointer_mass[f] = sum;
        if (sum >= kReachGate) ++frames[best];
    }
    for (int i = 0; i < W;) {
        int j = i, held = 0;
        for (; j < W && w.lines[j] == w.lines[i]; ++j) held += frames[j] >= kReachFrames ? 1 : 0;
        if (held >= kReachWords * (j - i)) r.lines_reached.push_back(w.lines[i]);
        i = j;
    }
    if (!r.lines_reached.empty()) roll.reached = std::max(roll.reached, r.lines_reached.back());
}

void Yue2Endless::Impl::lyrics_event(const Line& l, int k, int reached) {
    const Chunk& c = l.chunk(k);
    if (c.row.lyric_words.empty()) return;
    Yue2EndlessEvent e{Yue2EndlessEvent::Kind::Lyrics};
    e.chunk = k;
    e.branch = l.id;
    e.tag = c.row.tag;
    e.value = reached;
    e.sample = static_cast<std::int64_t>(k) * C * kR;
    e.row = std::make_shared<Yue2EndlessChunk>(c.row);
    event(e);
}

}  // namespace brosoundml
