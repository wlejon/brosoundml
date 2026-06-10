#include "brosoundml/phoneme_data.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error("brosoundml: " + where + ": " + msg);
}

// Frame count for an N-sample clip under the mel.h framing formula:
//   n_frames(N) = 1 + (N - win) / hop   (floor; 0 if N < win)
int framing_n_frames(int n_samples, int win_length, int hop_length) {
    if (win_length <= 0 || hop_length <= 0)
        fail("framing", "win_length and hop_length must be positive");
    if (n_samples < win_length) return 0;
    return 1 + (n_samples - win_length) / hop_length;
}

// ─── Little-endian fixed-width IO helpers ───────────────────────────────────

void w_u32(std::FILE* f, std::uint32_t v) {
    if (std::fwrite(&v, sizeof(v), 1, f) != 1)
        fail("write", "short write (u32)");
}
void w_i32(std::FILE* f, std::int32_t v) {
    if (std::fwrite(&v, sizeof(v), 1, f) != 1)
        fail("write", "short write (i32)");
}

std::uint32_t r_u32(std::FILE* f, const char* where) {
    std::uint32_t v = 0;
    if (std::fread(&v, sizeof(v), 1, f) != 1)
        fail(where, "unexpected EOF (u32)");
    return v;
}
std::int32_t r_i32(std::FILE* f, const char* where) {
    std::int32_t v = 0;
    if (std::fread(&v, sizeof(v), 1, f) != 1)
        fail(where, "unexpected EOF (i32)");
    return v;
}

// float -> int16 PCM, matching brosoundml::AudioBuffer::write_wav: clamp to
// [-1, 1] then scale by 32767 (not 32768) so +1.0 and -1.0 stay symmetric.
std::int16_t to_pcm16(float s) {
    if (s >  1.0f) s =  1.0f;
    if (s < -1.0f) s = -1.0f;
    return static_cast<std::int16_t>(std::lround(s * 32767.0f));
}

}  // namespace

// ─── A. PhonemeClassMap ─────────────────────────────────────────────────────

void PhonemeClassMap::rebuild_inverse() const {
    id_to_class_.clear();
    for (std::size_t c = 0; c < class_to_ids.size(); ++c) {
        for (int id : class_to_ids[c])
            id_to_class_[id] = static_cast<int>(c);
    }
    transparent_set_.clear();
    for (int id : transparent_ids) transparent_set_[id] = true;
    inverse_built_ = true;
}

int PhonemeClassMap::class_for_id(int phoneme_id) const {
    if (!inverse_built_) rebuild_inverse();
    auto it = id_to_class_.find(phoneme_id);
    if (it == id_to_class_.end())
        fail("PhonemeClassMap::class_for_id",
             "phoneme id " + std::to_string(phoneme_id) +
             " is not covered by the class map");
    return it->second;
}

bool PhonemeClassMap::is_transparent(int phoneme_id) const {
    if (!inverse_built_) rebuild_inverse();
    return transparent_set_.find(phoneme_id) != transparent_set_.end();
}

namespace {

// One named-class entry: the class name plus its candidate IPA spellings. The
// builder resolves each spelling against the vocab and keeps the ones present.
// Both pure IPA and the misaki single-char diphthong letters are listed so the
// class resolves whichever form a given vocab carries; absent forms are dropped.
struct NamedClass {
    const char*              name;
    std::vector<const char*> ipa;
};

// Default US-English phoneme inventory. Ordering is fixed (deterministic K and
// class indices). Stress diacritics / length marks / syllabic variants are
// absorbed into their base class via the many-to-one spelling lists.
const std::vector<NamedClass>& english_named_classes() {
    static const std::vector<NamedClass> kClasses = {
        // ── Monophthong vowels ──
        {"AA", {"ɑ", "ɒ", "a", "ɐ"}},        // ɑ ɒ a ɐ
        {"AE", {"æ"}},                                   // æ
        {"AH", {"ʌ"}},                                   // ʌ
        {"AX", {"ə", "ᵊ"}},                         // ə + ᵊ (superscript schwa)
        {"AO", {"ɔ"}},                                   // ɔ
        {"EH", {"ɛ"}},                                   // ɛ
        {"ER", {"ɝ", "ɚ", "ɜ", "˞"}},   // ɝ ɚ ɜ ˞
        {"IH", {"ɪ", "ᵻ", "ɨ"}},             // ɪ ᵻ ɨ
        {"IY", {"i", "iː"}},                             // i iː
        {"UH", {"ʊ"}},                                   // ʊ
        {"UW", {"u", "uː"}},                             // u uː
        {"OH", {"o"}},                                         // o
        {"EE", {"e"}},                                         // e
        // ── Diphthongs (IPA pairs + misaki single letters) ──
        {"EY", {"eɪ", "A"}},                            // eɪ
        {"AY", {"aɪ", "I"}},                            // aɪ
        {"OW", {"oʊ", "O"}},                            // oʊ
        {"AW", {"aʊ", "W"}},                            // aʊ
        {"OY", {"ɔɪ", "Y"}},                       // ɔɪ
        // ── Stops ──
        {"P", {"p"}},
        {"B", {"b"}},
        {"T", {"t"}},
        {"D", {"d"}},
        {"K", {"k"}},
        {"G", {"ɡ", "g"}},                              // ɡ g
        {"DX", {"ɾ"}},                                  // ɾ (flap)
        {"Q", {"ʔ"}},                                   // ʔ (glottal stop)
        // ── Fricatives ──
        {"F", {"f"}},
        {"V", {"v"}},
        {"TH", {"θ"}},                                  // θ
        {"DH", {"ð"}},                                  // ð
        {"S", {"s"}},
        {"Z", {"z"}},
        {"SH", {"ʃ"}},                                  // ʃ
        {"ZH", {"ʒ"}},                                  // ʒ
        {"HH", {"h", "ɦ"}},                             // h ɦ
        // ── Affricates ──
        {"CH", {"tʃ", "ʧ"}},                       // tʃ ʧ
        {"JH", {"dʒ", "ʤ"}},                       // dʒ ʤ
        // ── Nasals ──
        {"M", {"m"}},
        {"N", {"n"}},
        {"NG", {"ŋ"}},                                  // ŋ
        // ── Liquids ──
        {"L", {"l", "ɫ"}},                              // l ɫ
        {"R", {"ɹ", "r", "ɻ"}},                   // ɹ r ɻ
        // ── Glides ──
        {"W", {"w", "ʍ"}},                              // w ʍ
        {"Y", {"j"}},                                          // j
    };
    return kClasses;
}

// True if `key` is a suprasegmental / diacritic modifier token: stress (ˈ ˌ),
// length (ː), aspiration (ʰ), palatalization (ʲ), nasalization (̃, combining),
// or an intonation arrow (↑ ↓ → ↗ ↘). misaki/Kokoro emit these as their own
// duration-bearing tokens, but they attach to an adjacent segmental phoneme
// rather than denoting a distinct sound. We route them to the silence class so
// they neither inflate a spurious "other" bucket nor become a phoneme the model
// must learn to distinguish acoustically. (The spotter strips silence from
// enrolled templates, so a stress mark never appears in a keyword's class
// sequence.) Trade-off: the few frames a length/aspiration token spans are
// labelled silence rather than merged into the neighbouring phoneme — acceptable
// at a 10 ms hop given how short these tokens are; revisit with neighbour-merge
// if the frame-accuracy gate demands it.
bool is_modifier_key(const std::string& key) {
    static const char* kMods[] = {
        "ˈ",  // ˈ primary stress
        "ˌ",  // ˌ secondary stress
        "ː",  // ː length
        "ʰ",  // ʰ aspiration
        "ʲ",  // ʲ palatalization
        "̃",  // ̃  nasalization (combining)
        "↑",  // ↑
        "↓",  // ↓
        "→",  // →
        "↗",  // ↗
        "↘",  // ↘
    };
    for (const char* m : kMods)
        if (key == m) return true;
    return false;
}

// True if `key` is a non-spoken token (punctuation / whitespace / digit /
// control, or empty) — every byte is ASCII (<0x80) and none is an ASCII letter.
// Such ids are pauses and route to the silence class. A key with any ASCII
// letter or any non-ASCII (IPA) byte is "spoken".
bool is_non_spoken_key(const std::string& key) {
    if (key.empty()) return true;
    for (unsigned char c : key) {
        if (c >= 0x80) return false;                       // IPA byte -> spoken
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
            return false;                                  // ASCII letter -> spoken
    }
    return true;
}

}  // namespace

PhonemeClassMap build_default_english_classmap(
    const std::unordered_map<std::string, int>& vocab) {
    const auto& named = english_named_classes();

    PhonemeClassMap cm;
    // Class 0 = silence, classes 1..named.size() = the named phoneme classes,
    // final class = "other" catch-all for spoken-but-unnamed ids.
    cm.num_classes = static_cast<int>(named.size()) + 2;
    cm.class_names.reserve(static_cast<std::size_t>(cm.num_classes));
    cm.class_to_ids.assign(static_cast<std::size_t>(cm.num_classes), {});

    cm.class_names.push_back("sil");
    for (const auto& nc : named) cm.class_names.push_back(nc.name);
    cm.class_names.push_back("other");
    const int other_class = cm.num_classes - 1;

    // Resolve the named spellings, tracking which ids we have claimed.
    std::unordered_map<int, int> claimed;   // id -> class
    for (std::size_t ci = 0; ci < named.size(); ++ci) {
        const int cls = static_cast<int>(ci) + 1;
        for (const char* spelling : named[ci].ipa) {
            auto it = vocab.find(spelling);
            if (it == vocab.end()) continue;
            const int id = it->second;
            if (claimed.count(id)) continue;   // first class to claim wins
            claimed[id] = cls;
            cm.class_to_ids[static_cast<std::size_t>(cls)].push_back(id);
        }
    }

    // Cover every remaining vocab id: modifier diacritics and non-spoken
    // punctuation -> silence; any other spoken-but-unnamed id -> "other".
    // Modifier diacritics are additionally flagged transparent so their frames
    // merge into the neighbouring phoneme at label-build time (see build_frame_
    // labels) instead of silencing voiced onsets.
    for (const auto& kv : vocab) {
        const int id = kv.second;
        if (claimed.count(id)) continue;
        const bool modifier = is_modifier_key(kv.first);
        const int  cls = (modifier || is_non_spoken_key(kv.first)) ? 0 : other_class;
        claimed[id] = cls;
        cm.class_to_ids[static_cast<std::size_t>(cls)].push_back(id);
        if (modifier) cm.transparent_ids.push_back(id);
    }

    // Determinism: sort ids within each class and the transparent set.
    for (auto& ids : cm.class_to_ids)
        std::sort(ids.begin(), ids.end());
    std::sort(cm.transparent_ids.begin(), cm.transparent_ids.end());

    cm.rebuild_inverse();
    return cm;
}

// ─── Class-map serialization ────────────────────────────────────────────────

void write_classmap(std::FILE* f, const PhonemeClassMap& cm) {
    if (!f) fail("write_classmap", "null file");
    if (cm.num_classes < 0 ||
        static_cast<std::size_t>(cm.num_classes) != cm.class_names.size() ||
        static_cast<std::size_t>(cm.num_classes) != cm.class_to_ids.size())
        fail("write_classmap", "inconsistent class map (size mismatch)");
    w_u32(f, static_cast<std::uint32_t>(cm.num_classes));
    for (int c = 0; c < cm.num_classes; ++c) {
        const std::string& name = cm.class_names[static_cast<std::size_t>(c)];
        const auto&        ids  = cm.class_to_ids[static_cast<std::size_t>(c)];
        w_u32(f, static_cast<std::uint32_t>(name.size()));
        if (!name.empty() &&
            std::fwrite(name.data(), 1, name.size(), f) != name.size())
            fail("write_classmap", "short write (name)");
        w_u32(f, static_cast<std::uint32_t>(ids.size()));
        for (int id : ids) w_i32(f, id);
    }
    w_u32(f, static_cast<std::uint32_t>(cm.transparent_ids.size()));
    for (int id : cm.transparent_ids) w_i32(f, id);
}

PhonemeClassMap read_classmap(std::FILE* f) {
    if (!f) fail("read_classmap", "null file");
    PhonemeClassMap cm;
    const std::uint32_t k = r_u32(f, "read_classmap");
    cm.num_classes = static_cast<int>(k);
    cm.class_names.resize(k);
    cm.class_to_ids.resize(k);
    for (std::uint32_t c = 0; c < k; ++c) {
        const std::uint32_t name_len = r_u32(f, "read_classmap");
        std::string name(name_len, '\0');
        if (name_len &&
            std::fread(name.data(), 1, name_len, f) != name_len)
            fail("read_classmap", "unexpected EOF (name)");
        cm.class_names[c] = std::move(name);
        const std::uint32_t id_count = r_u32(f, "read_classmap");
        std::vector<int> ids(id_count);
        for (std::uint32_t i = 0; i < id_count; ++i)
            ids[i] = r_i32(f, "read_classmap");
        cm.class_to_ids[c] = std::move(ids);
    }
    const std::uint32_t t_count = r_u32(f, "read_classmap");
    cm.transparent_ids.resize(t_count);
    for (std::uint32_t i = 0; i < t_count; ++i)
        cm.transparent_ids[i] = r_i32(f, "read_classmap");
    cm.rebuild_inverse();
    return cm;
}

// ─── B. Frame-label alignment ───────────────────────────────────────────────

std::vector<int16_t> build_frame_labels(
    const std::vector<int32_t>& pred_dur_wrapped,
    const std::vector<int32_t>& phoneme_ids,
    const PhonemeClassMap& cm,
    int n_samples_16k, int win_length, int hop_length) {
    if (pred_dur_wrapped.size() != phoneme_ids.size() + 2)
        fail("build_frame_labels",
             "pred_dur_wrapped length must equal phoneme_ids length + 2");
    if (n_samples_16k < 0) fail("build_frame_labels", "n_samples_16k < 0");

    const int n_frames = framing_n_frames(n_samples_16k, win_length, hop_length);
    const int silence  = cm.silence_class();
    std::vector<int16_t> labels(static_cast<std::size_t>(n_frames),
                                static_cast<int16_t>(silence));
    if (n_frames == 0) return labels;

    // Total duration. A degenerate (<=0) total can't be scaled -> all silence.
    std::int64_t sum_dur = 0;
    for (std::int32_t d : pred_dur_wrapped) sum_dur += d;
    if (sum_dur <= 0) return labels;

    const double spk =
        static_cast<double>(n_samples_16k) / static_cast<double>(sum_dur);

    // boundary[k] = lround(cumdur_{<k tokens summed} * spk); boundary[0] = 0,
    // boundary[T] == n_samples_16k. Token i spans [boundary[i], boundary[i+1]).
    const std::size_t T = pred_dur_wrapped.size();
    std::vector<std::int64_t> boundary(T + 1);
    boundary[0] = 0;
    std::int64_t cum = 0;
    for (std::size_t i = 0; i < T; ++i) {
        cum += pred_dur_wrapped[i];
        boundary[i + 1] =
            std::llround(static_cast<double>(cum) * spk);
    }
    // Pin the final boundary exactly (guards against fp rounding at the tail).
    boundary[T] = n_samples_16k;

    // Per-token raw class + transparency: BOS (i==0) and EOS (i==T-1) -> silence
    // and are never transparent; interior token i carries phoneme_ids[i-1].
    std::vector<int>  token_class(T, silence);
    std::vector<char> token_transp(T, 0);
    for (std::size_t i = 1; i + 1 < T; ++i) {
        const int id = phoneme_ids[i - 1];
        token_class[i]  = cm.class_for_id(id);
        token_transp[i] = cm.is_transparent(id) ? 1 : 0;
    }

    // Effective class: a transparent modifier token inherits the class of the
    // nearest interior NON-transparent token (forward first, then backward), so
    // its frames stay voiced rather than silencing a stressed-vowel onset. No
    // segmental neighbour -> silence.
    std::vector<int> eff(T, silence);
    for (std::size_t i = 0; i < T; ++i) {
        if (!token_transp[i]) { eff[i] = token_class[i]; continue; }
        bool got = false;
        for (std::size_t j = i + 1; j + 1 < T; ++j)
            if (!token_transp[j]) { eff[i] = token_class[j]; got = true; break; }
        if (!got)
            for (std::size_t j = i; j > 1;) {
                --j;
                if (!token_transp[j]) { eff[i] = token_class[j]; break; }
            }
    }

    // Walk frames and tokens in lockstep (both monotone): advance the token
    // cursor until frame_center < boundary[tok+1].
    std::size_t tok = 0;
    for (int t = 0; t < n_frames; ++t) {
        const std::int64_t center =
            static_cast<std::int64_t>(t) * hop_length + win_length / 2;
        while (tok + 1 < T && center >= boundary[tok + 1]) ++tok;
        // center is always in (0, n_samples_16k); clamp defensively.
        if (center < boundary[0] || center >= boundary[T])
            labels[static_cast<std::size_t>(t)] =
                static_cast<int16_t>(silence);
        else
            labels[static_cast<std::size_t>(t)] =
                static_cast<int16_t>(eff[tok]);
    }
    return labels;
}

std::vector<int16_t> resample_labels_nn(const std::vector<int16_t>& labels,
                                        int new_len) {
    if (new_len < 0) fail("resample_labels_nn", "new_len < 0");
    std::vector<int16_t> out(static_cast<std::size_t>(new_len), 0);
    const int old_len = static_cast<int>(labels.size());
    if (new_len == 0 || old_len == 0) return out;
    for (int j = 0; j < new_len; ++j) {
        int src = static_cast<int>(std::floor(
            (static_cast<double>(j) + 0.5) *
            static_cast<double>(old_len) / static_cast<double>(new_len)));
        if (src < 0) src = 0;
        if (src >= old_len) src = old_len - 1;
        out[static_cast<std::size_t>(j)] = labels[static_cast<std::size_t>(src)];
    }
    return out;
}

std::vector<int16_t> crop_or_pad_labels_centered(
    const std::vector<int16_t>& labels, int target_len, int silence_class) {
    if (target_len <= 0) fail("crop_or_pad_labels_centered", "target_len <= 0");
    const int N = static_cast<int>(labels.size());
    std::vector<int16_t> out(static_cast<std::size_t>(target_len),
                             static_cast<int16_t>(silence_class));
    if (N == 0) return out;
    if (N >= target_len) {
        const int start = (N - target_len) / 2;   // mirrors crop_or_pad_centered
        std::memcpy(out.data(), labels.data() + start,
                    static_cast<std::size_t>(target_len) * sizeof(int16_t));
    } else {
        const int start = (target_len - N) / 2;
        std::memcpy(out.data() + start, labels.data(),
                    static_cast<std::size_t>(N) * sizeof(int16_t));
    }
    return out;
}

// ─── C. BPDS writer ─────────────────────────────────────────────────────────

PhonemeDatasetWriter::PhonemeDatasetWriter(const std::string& path,
                                           const PhonemeDatasetHeader& header,
                                           const PhonemeClassMap& class_map)
    : path_(path), header_(header) {
    if (header_.win_length <= 0 || header_.hop_length <= 0)
        fail("PhonemeDatasetWriter::PhonemeDatasetWriter",
             "win_length and hop_length must be positive");
    std::FILE* f = nullptr;
#if defined(_MSC_VER)
    if (fopen_s(&f, path.c_str(), "wb") != 0) f = nullptr;
#else
    f = std::fopen(path.c_str(), "wb");
#endif
    if (!f) fail("PhonemeDatasetWriter::PhonemeDatasetWriter",
                 "cannot open '" + path + "' for write");
    fp_ = f;

    w_u32(f, kMagicBPDS);
    w_u32(f, kBPDSVersion);
    w_u32(f, static_cast<std::uint32_t>(header_.sample_rate));
    w_u32(f, static_cast<std::uint32_t>(header_.n_fft));
    w_u32(f, static_cast<std::uint32_t>(header_.win_length));
    w_u32(f, static_cast<std::uint32_t>(header_.hop_length));
    w_u32(f, static_cast<std::uint32_t>(header_.n_mels));
    write_classmap(f, class_map);
    clip_count_pos_ = std::ftell(f);
    w_u32(f, 0u);   // clip_count placeholder, patched by finalize()
}

PhonemeDatasetWriter::~PhonemeDatasetWriter() {
    try { finalize(); } catch (...) { /* never throw from a destructor */ }
    if (fp_) {
        std::fclose(static_cast<std::FILE*>(fp_));
        fp_ = nullptr;
    }
}

void PhonemeDatasetWriter::append(const std::vector<float>& pcm16k,
                                  const std::vector<int16_t>& labels) {
    if (!fp_ || finalized_)
        fail("PhonemeDatasetWriter::append", "writer is closed");
    std::FILE* f = static_cast<std::FILE*>(fp_);

    const int n_samples = static_cast<int>(pcm16k.size());
    const int expect_frames =
        framing_n_frames(n_samples, header_.win_length, header_.hop_length);
    if (static_cast<int>(labels.size()) != expect_frames)
        fail("PhonemeDatasetWriter::append",
             "label count " + std::to_string(labels.size()) +
             " != framing frame count " + std::to_string(expect_frames) +
             " for " + std::to_string(n_samples) + " samples");

    w_i32(f, n_samples);
    for (float s : pcm16k) {
        const std::int16_t v = to_pcm16(s);
        if (std::fwrite(&v, sizeof(v), 1, f) != 1)
            fail("PhonemeDatasetWriter::append", "short write (pcm)");
    }
    w_i32(f, expect_frames);
    if (expect_frames &&
        std::fwrite(labels.data(), sizeof(int16_t),
                    static_cast<std::size_t>(expect_frames), f) !=
        static_cast<std::size_t>(expect_frames))
        fail("PhonemeDatasetWriter::append", "short write (labels)");
    ++clips_;
}

void PhonemeDatasetWriter::finalize() {
    if (finalized_ || !fp_) return;
    std::FILE* f = static_cast<std::FILE*>(fp_);
    if (std::fflush(f) != 0)
        fail("PhonemeDatasetWriter::finalize", "flush failed");
    if (std::fseek(f, clip_count_pos_, SEEK_SET) != 0)
        fail("PhonemeDatasetWriter::finalize", "seek to clip count failed");
    w_u32(f, static_cast<std::uint32_t>(clips_));
    if (std::fflush(f) != 0)
        fail("PhonemeDatasetWriter::finalize", "flush after patch failed");
    finalized_ = true;
}

// ─── C. BPDS reader ─────────────────────────────────────────────────────────

PhonemeDataset read_phoneme_dataset(const std::string& path) {
    std::FILE* f = nullptr;
#if defined(_MSC_VER)
    if (fopen_s(&f, path.c_str(), "rb") != 0) f = nullptr;
#else
    f = std::fopen(path.c_str(), "rb");
#endif
    if (!f) fail("read_phoneme_dataset", "cannot open '" + path + "'");

    PhonemeDataset ds;
    try {
        const std::uint32_t magic = r_u32(f, "read_phoneme_dataset");
        if (magic != kMagicBPDS)
            fail("read_phoneme_dataset", "bad magic (not a BPDS file)");
        const std::uint32_t version = r_u32(f, "read_phoneme_dataset");
        if (version != kBPDSVersion)
            fail("read_phoneme_dataset",
                 "unsupported version " + std::to_string(version));
        ds.header.sample_rate = static_cast<int>(r_u32(f, "read_phoneme_dataset"));
        ds.header.n_fft       = static_cast<int>(r_u32(f, "read_phoneme_dataset"));
        ds.header.win_length  = static_cast<int>(r_u32(f, "read_phoneme_dataset"));
        ds.header.hop_length  = static_cast<int>(r_u32(f, "read_phoneme_dataset"));
        ds.header.n_mels      = static_cast<int>(r_u32(f, "read_phoneme_dataset"));
        ds.class_map = read_classmap(f);
        const std::uint32_t clip_count = r_u32(f, "read_phoneme_dataset");

        ds.clips.reserve(clip_count);
        for (std::uint32_t c = 0; c < clip_count; ++c) {
            PhonemeClip clip;
            const std::int32_t n_samples = r_i32(f, "read_phoneme_dataset");
            if (n_samples < 0)
                fail("read_phoneme_dataset", "negative n_samples");
            clip.pcm.resize(static_cast<std::size_t>(n_samples));
            if (n_samples &&
                std::fread(clip.pcm.data(), sizeof(int16_t),
                           static_cast<std::size_t>(n_samples), f) !=
                static_cast<std::size_t>(n_samples))
                fail("read_phoneme_dataset", "truncated PCM payload");
            const std::int32_t n_frames = r_i32(f, "read_phoneme_dataset");
            if (n_frames < 0)
                fail("read_phoneme_dataset", "negative n_frames");
            const int expect = framing_n_frames(n_samples, ds.header.win_length,
                                                ds.header.hop_length);
            if (n_frames != expect)
                fail("read_phoneme_dataset",
                     "clip " + std::to_string(c) + " frame count " +
                     std::to_string(n_frames) + " != framing " +
                     std::to_string(expect));
            clip.labels.resize(static_cast<std::size_t>(n_frames));
            if (n_frames &&
                std::fread(clip.labels.data(), sizeof(int16_t),
                           static_cast<std::size_t>(n_frames), f) !=
                static_cast<std::size_t>(n_frames))
                fail("read_phoneme_dataset", "truncated label payload");
            ds.clips.push_back(std::move(clip));
        }
    } catch (...) {
        std::fclose(f);
        throw;
    }
    std::fclose(f);
    return ds;
}

// ─── D. Validation ──────────────────────────────────────────────────────────

namespace {

void pcm_stats(const std::vector<int16_t>& pcm, float& out_peak, float& out_rms) {
    out_peak = 0.0f;
    out_rms  = 0.0f;
    if (pcm.empty()) return;
    double sq = 0.0;
    for (std::int16_t s : pcm) {
        const float v = static_cast<float>(s) / 32767.0f;
        const float a = std::fabs(v);
        if (a > out_peak) out_peak = a;
        sq += static_cast<double>(v) * v;
    }
    out_rms = static_cast<float>(std::sqrt(sq / static_cast<double>(pcm.size())));
}

}  // namespace

PhonemeDatasetReport validate_phoneme_dataset(
    const PhonemeDataset& ds, const PhonemeValidationConfig& cfg) {
    PhonemeDatasetReport r;
    const int K = ds.class_map.num_classes;
    r.per_class_frames.assign(static_cast<std::size_t>(std::max(0, K)), 0);
    r.total_clips = static_cast<int>(ds.clips.size());
    if (ds.header.sample_rate != cfg.expected_sample_rate)
        r.sample_rate_mismatch = 1;

    std::vector<float> peaks, rmses;
    peaks.reserve(ds.clips.size());
    rmses.reserve(ds.clips.size());

    for (const auto& clip : ds.clips) {
        const int n_samples = static_cast<int>(clip.pcm.size());
        const int n_frames  = static_cast<int>(clip.labels.size());
        const int expect = framing_n_frames(n_samples, ds.header.win_length,
                                            ds.header.hop_length);
        if (n_frames != expect) ++r.length_mismatch_clips;
        if (n_samples == 0 || n_frames == 0) ++r.empty_clips;

        for (std::int16_t lbl : clip.labels) {
            r.total_frames += 1;
            if (lbl < 0 || lbl >= K) {
                ++r.label_out_of_range;
                continue;
            }
            if (lbl == ds.class_map.silence_class()) ++r.silence_frames;
            r.per_class_frames[static_cast<std::size_t>(lbl)] += 1;
        }

        float pk = 0.0f, rm = 0.0f;
        pcm_stats(clip.pcm, pk, rm);
        peaks.push_back(pk);
        rmses.push_back(rm);
        ++r.decoded_clips;
    }

    if (!peaks.empty()) {
        float pmin = peaks[0], pmax = peaks[0]; double psum = 0.0;
        float rmin = rmses[0], rmax = rmses[0]; double rsum = 0.0;
        for (std::size_t i = 0; i < peaks.size(); ++i) {
            const float p = peaks[i], q = rmses[i];
            if (p < pmin) pmin = p; if (p > pmax) pmax = p;
            if (q < rmin) rmin = q; if (q > rmax) rmax = q;
            psum += p; rsum += q;
        }
        const double n = static_cast<double>(peaks.size());
        const double pmean = psum / n, rmean = rsum / n;
        double pvar = 0.0, rvar = 0.0;
        for (std::size_t i = 0; i < peaks.size(); ++i) {
            const double dp = peaks[i] - pmean, dq = rmses[i] - rmean;
            pvar += dp * dp; rvar += dq * dq;
        }
        pvar /= n; rvar /= n;
        r.peak_min = pmin; r.peak_max = pmax;
        r.peak_mean = static_cast<float>(pmean);
        r.peak_std  = static_cast<float>(std::sqrt(pvar));
        r.rms_min = rmin; r.rms_max = rmax;
        r.rms_mean = static_cast<float>(rmean);
        r.rms_std  = static_cast<float>(std::sqrt(rvar));
    }

    return r;
}

bool report_passes(const PhonemeDatasetReport& r, const PhonemeClassMap& cm,
                   const PhonemeValidationConfig& cfg, std::string* reason) {
    auto set = [&](const std::string& s) { if (reason) *reason = s; };

    if (r.total_clips == 0) { set("dataset is empty"); return false; }
    if (r.sample_rate_mismatch > 0) {
        set("sample-rate mismatch (expected " +
            std::to_string(cfg.expected_sample_rate) + ")");
        return false;
    }
    if (r.length_mismatch_clips > 0) {
        set("frame-count invariant violated in " +
            std::to_string(r.length_mismatch_clips) + " clip(s)");
        return false;
    }
    if (r.empty_clips > 0) {
        set("empty clips: " + std::to_string(r.empty_clips));
        return false;
    }
    if (r.label_out_of_range > 0) {
        set("labels out of range: " + std::to_string(r.label_out_of_range));
        return false;
    }

    // Per-class minimum coverage — scoped to classes that can actually appear:
    // the silence class and any class that owns ids. Structurally-empty named
    // classes are skipped (they legitimately never appear).
    const int K = cm.num_classes;
    for (int c = 0; c < K; ++c) {
        const bool can_appear =
            (c == cm.silence_class()) ||
            !cm.class_to_ids[static_cast<std::size_t>(c)].empty();
        if (!can_appear) continue;
        const long long got =
            (c < static_cast<int>(r.per_class_frames.size()))
                ? r.per_class_frames[static_cast<std::size_t>(c)] : 0;
        if (got < cfg.min_frames_per_class) {
            set("class '" + cm.class_names[static_cast<std::size_t>(c)] +
                "' has " + std::to_string(got) + " frames (< " +
                std::to_string(cfg.min_frames_per_class) + ")");
            return false;
        }
    }

    if (r.total_frames > 0) {
        const double sil_frac = static_cast<double>(r.silence_frames) /
                                static_cast<double>(r.total_frames);
        if (sil_frac > cfg.max_silence_fraction) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                "silence fraction %.4f exceeds max %.4f",
                sil_frac, static_cast<double>(cfg.max_silence_fraction));
            set(buf);
            return false;
        }
    }

    if (reason) reason->clear();
    return true;
}

void print_report(const PhonemeDatasetReport& r, const PhonemeClassMap& cm,
                  std::FILE* out) {
    std::fprintf(out, "phoneme dataset report\n");
    std::fprintf(out, "  total clips:          %d\n", r.total_clips);
    std::fprintf(out, "  decoded clips:        %d\n", r.decoded_clips);
    std::fprintf(out, "  length mismatch:      %d\n", r.length_mismatch_clips);
    std::fprintf(out, "  empty clips:          %d\n", r.empty_clips);
    std::fprintf(out, "  label out of range:   %d\n", r.label_out_of_range);
    std::fprintf(out, "  sample-rate mismatch: %d\n", r.sample_rate_mismatch);
    std::fprintf(out, "  total frames:         %lld\n", r.total_frames);
    const double sil_frac = r.total_frames > 0
        ? static_cast<double>(r.silence_frames) /
          static_cast<double>(r.total_frames) : 0.0;
    std::fprintf(out, "  silence frames:       %lld (%.4f)\n",
                 r.silence_frames, sil_frac);
    std::fprintf(out, "  peak: min=%.4f mean=%.4f max=%.4f std=%.4f\n",
                 static_cast<double>(r.peak_min),
                 static_cast<double>(r.peak_mean),
                 static_cast<double>(r.peak_max),
                 static_cast<double>(r.peak_std));
    std::fprintf(out, "  rms : min=%.4f mean=%.4f max=%.4f std=%.4f\n",
                 static_cast<double>(r.rms_min),
                 static_cast<double>(r.rms_mean),
                 static_cast<double>(r.rms_max),
                 static_cast<double>(r.rms_std));
    std::fprintf(out, "  per-class frames:\n");
    const int K = cm.num_classes;
    for (int c = 0; c < K; ++c) {
        const long long got =
            (c < static_cast<int>(r.per_class_frames.size()))
                ? r.per_class_frames[static_cast<std::size_t>(c)] : 0;
        const char* name = (c < static_cast<int>(cm.class_names.size()))
            ? cm.class_names[static_cast<std::size_t>(c)].c_str() : "?";
        std::fprintf(out, "    %3d %-8s %lld\n", c, name, got);
    }
}

}  // namespace brosoundml
