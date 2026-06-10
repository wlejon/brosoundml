// brosoundml_phoneme_synth — render a sentence corpus through Kokoro across
// many voices into ONE BPDS file with per-frame phoneme-class labels.
//
// The phoneme analogue of wake_synth: instead of a binary positive/negative
// wake dataset it produces a frame-labelled dataset for open-vocabulary
// phoneme-posterior keyword spotting. Each (sentence, voice, variant) is
// synthesized by the real Kokoro TTS, resampled to 16 kHz, and aligned to a
// per-10 ms-frame phoneme-class track via build_frame_labels (which consumes
// Kokoro's pred_dur_out directly). Augmentation is length-preserving (additive
// noise + gain), and speed perturbation rides Kokoro's own `speed` arg so the
// predicted durations — and therefore the labels — stay exactly aligned.
//
// The dataset is byte-deterministic for a fixed --seed.

#include "brosoundml/audio.h"
#include "brosoundml/kokoro.h"
#include "brosoundml/phoneme_data.h"
#include "brosoundml/wake_data.h"

#include "brosoundml/g2p/lexicon.h"
#include "brosoundml/g2p/morphology.h"
#include "brosoundml/g2p/phoneme_adapter.h"
#include "brosoundml/g2p/phonemizer.h"
#include "brosoundml/g2p/pos_tagger.h"
#include "brosoundml/g2p/special_cases.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace g  = brosoundml::g2p;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "phoneme_synth: %s\n", msg.c_str());
    std::exit(2);
}

std::string env_or_empty(const char* name) {
    if (const char* v = std::getenv(name)) return std::string(v);
    return {};
}

// Standard data-path resolution per CLAUDE.md:
//   caller-supplied > BROSOUNDML_DATA_DIR env > ../brosoundml-data
std::string default_data_dir() {
    const auto env = env_or_empty("BROSOUNDML_DATA_DIR");
    if (!env.empty() && fs::exists(env)) return env;
    return "../brosoundml-data";
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t a = tok.find_first_not_of(" \t\r\n");
        size_t b = tok.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        out.push_back(tok.substr(a, b - a + 1));
    }
    return out;
}

// Built-in fallback corpus — ~12 diverse English sentences so the tool runs
// out of the box without a --corpus file.
const std::vector<std::string>& default_corpus() {
    static const std::vector<std::string> c = {
        "the quick brown fox jumps over the lazy dog",
        "she sells seashells by the seashore",
        "a journey of a thousand miles begins with a single step",
        "bright vivid colors painted the evening sky",
        "please remember to bring your umbrella tomorrow",
        "the children laughed and played in the park",
        "an ancient oak tree stood beside the river",
        "music drifted softly through the open window",
        "we should gather fresh vegetables for dinner",
        "the curious kitten chased a ball of yarn",
        "thunder rumbled in the distance before the storm",
        "he carefully measured each ingredient for the cake",
    };
    return c;
}

// Read a corpus file: one sentence per line, blanks skipped.
std::vector<std::string> read_corpus(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open corpus: " + path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t a = line.find_first_not_of(" \t");
        if (a == std::string::npos) continue;
        out.push_back(line);
    }
    return out;
}

void usage(std::FILE* out) {
    std::fprintf(out,
        "Usage: brosoundml_phoneme_synth [flags]\n"
        "  --corpus PATH       Text file, one sentence per line.\n"
        "                      If omitted, a built-in ~12-sentence default corpus is used.\n"
        "  --model-dir DIR     Kokoro model directory (default weights/kokoro)\n"
        "  --voices-dir DIR    Voice pack directory (default <model-dir>/voices)\n"
        "  --out PATH          Output .bpds file (default <data>/phoneme/english.bpds)\n"
        "  --lexicon PATH      G2P lexicon (.bin) (default <data>/g2p/lexicon_en_us.bin)\n"
        "  --pos-tagger PATH   POS tagger (.bin)  (default <data>/pos_tagger/model.bin)\n"
        "  --device cpu|cuda   Kokoro inference device (default cpu)\n"
        "  --seed N            Deterministic RNG seed (default 42)\n"
        "  --max-voices N      Cap voices (0 = all, default 0)\n"
        "  --noise-dir DIR     Directory of noise WAVs for additive noise.\n"
        "                      If empty, synthetic white/pink/brown noise is used.\n"
        "  --snr-list CSV      SNRs in dB; 'clean' = no noise (default clean,20,10,5)\n"
        "  --speed-list CSV    Kokoro speed factors (default 1.0,0.9,1.1)\n"
        "  --aug-variants N    Augmented variants per (sentence,voice) (default 2)\n"
        "  --small             Tiny smoke run (<=3 voices, built-in corpus, 0 aug)\n"
        "  -h, --help          Show this help\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpus_path;
    std::string model_dir   = "weights/kokoro";
    std::string voices_dir;
    std::string out_path;
    std::string lexicon_path;
    std::string pos_path;
    std::string device_str = "cpu";
    std::string noise_dir;
    std::string snr_csv   = "clean,20,10,5";
    std::string speed_csv = "1.0,0.9,1.1";
    std::uint64_t seed = 42;
    int  max_voices  = 0;
    int  aug_variants = 2;
    bool small_mode  = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string(what) + " needs a value");
            return argv[++i];
        };
        if      (a == "--corpus")      corpus_path  = next("--corpus");
        else if (a == "--model-dir")   model_dir    = next("--model-dir");
        else if (a == "--voices-dir")  voices_dir   = next("--voices-dir");
        else if (a == "--out")         out_path     = next("--out");
        else if (a == "--lexicon")     lexicon_path = next("--lexicon");
        else if (a == "--pos-tagger")  pos_path     = next("--pos-tagger");
        else if (a == "--device")      device_str   = next("--device");
        else if (a == "--seed")        seed = std::strtoull(next("--seed").c_str(), nullptr, 10);
        else if (a == "--max-voices")  max_voices = std::atoi(next("--max-voices").c_str());
        else if (a == "--noise-dir")   noise_dir    = next("--noise-dir");
        else if (a == "--snr-list")    snr_csv      = next("--snr-list");
        else if (a == "--speed-list")  speed_csv    = next("--speed-list");
        else if (a == "--aug-variants") aug_variants = std::atoi(next("--aug-variants").c_str());
        else if (a == "--small")       small_mode = true;
        else if (a == "-h" || a == "--help") { usage(stdout); return 0; }
        else die("unknown flag '" + a + "'");
    }

    if (voices_dir.empty()) voices_dir = model_dir + "/voices";
    const std::string data_dir = default_data_dir();
    if (lexicon_path.empty()) lexicon_path = data_dir + "/g2p/lexicon_en_us.bin";
    if (pos_path.empty())     pos_path     = data_dir + "/pos_tagger/model.bin";
    if (out_path.empty())     out_path     = data_dir + "/phoneme/english.bpds";

    // Header / framing params — must match the dataset format invariant.
    constexpr int kWin = 400;
    constexpr int kHop = 160;
    const brosoundml::PhonemeDatasetHeader header{16000, 512, kWin, kHop, 40};

    try {
        brotensor::init();
        brotensor::Device dev = brotensor::Device::CPU;
        if      (device_str == "cpu")  dev = brotensor::Device::CPU;
        else if (device_str == "cuda") dev = brotensor::Device::CUDA;
        else die("--device must be one of cpu|cuda");
        if (dev != brotensor::Device::CPU && !brotensor::is_available(dev))
            die("--device " + device_str + " not available in this build");

        // ─── Load Kokoro ───────────────────────────────────────────────
        if (!fs::exists(model_dir + "/config.json"))
            die("model-dir missing config.json: " + model_dir);
        brosoundml::Kokoro k;
        k.load(model_dir, dev);
        std::fprintf(stderr, "loaded kokoro from %s on %s\n",
                     model_dir.c_str(), device_str.c_str());

        const int kokoro_sr = k.config().sample_rate;   // 24000

        // ─── Build the phoneme class map ONCE from Kokoro's vocab ───────
        const auto classmap =
            brosoundml::build_default_english_classmap(k.config().vocab);
        std::fprintf(stderr, "class map: K=%d classes (over %zu vocab ids)\n",
                     classmap.num_classes, k.config().vocab.size());

        // ─── Load voices (stable sort by filename) ─────────────────────
        if (!fs::exists(voices_dir))
            die("voices-dir does not exist: " + voices_dir);
        std::vector<fs::path> voice_paths;
        for (const auto& e : fs::directory_iterator(voices_dir)) {
            if (e.is_regular_file() && e.path().extension() == ".bin")
                voice_paths.push_back(e.path());
        }
        std::sort(voice_paths.begin(), voice_paths.end());
        if (voice_paths.empty()) die("no .bin voice packs in " + voices_dir);

        int voice_cap = max_voices;
        if (small_mode) voice_cap = (voice_cap > 0) ? std::min(voice_cap, 3) : 3;
        if (voice_cap > 0 && static_cast<int>(voice_paths.size()) > voice_cap)
            voice_paths.resize(static_cast<std::size_t>(voice_cap));

        std::vector<brosoundml::Voice> voices;
        std::vector<std::string>       voice_names;
        for (const auto& p : voice_paths) {
            voices.push_back(k.load_voice(p.string()));
            voice_names.push_back(p.stem().string());
            std::fprintf(stderr, "  voice %s\n", voice_names.back().c_str());
        }

        // ─── G2P stack ─────────────────────────────────────────────────
        if (!fs::exists(lexicon_path)) die("missing lexicon: " + lexicon_path);
        if (!fs::exists(pos_path))     die("missing pos tagger: " + pos_path);
        g::Lexicon        lex   = g::Lexicon::load(lexicon_path);
        g::Morphology     morph(lex);
        g::SpecialCases   sc(lex);
        g::PosTagger      tagger = g::PosTagger::load(pos_path);
        g::PhonemeAdapter adapter(k.config().vocab);
        g::Phonemizer     phon(tagger, lex, morph, sc, adapter);

        // ─── Corpus ────────────────────────────────────────────────────
        std::vector<std::string> corpus;
        if (small_mode) {
            corpus = default_corpus();
            if (corpus.size() > 4) corpus.resize(4);
        } else if (!corpus_path.empty()) {
            corpus = read_corpus(corpus_path);
        } else {
            corpus = default_corpus();
            std::fprintf(stderr,
                "no --corpus given; using built-in default corpus (%zu sentences)\n",
                corpus.size());
        }
        if (corpus.empty()) die("empty corpus");

        // ─── Augmentation config ───────────────────────────────────────
        const auto snr_tokens   = split_csv(snr_csv);
        const auto speed_tokens = split_csv(speed_csv);
        if (snr_tokens.empty())   die("--snr-list is empty");
        if (speed_tokens.empty()) die("--speed-list is empty");
        std::vector<float> speeds;
        for (const auto& s : speed_tokens) {
            float v = std::strtof(s.c_str(), nullptr);
            if (v <= 0.0f) die("bad speed in --speed-list: " + s);
            speeds.push_back(v);
        }
        const int n_aug = small_mode ? 0 : std::max(0, aug_variants);

        // ─── Noise corpus (optional) ───────────────────────────────────
        std::vector<std::vector<float>> noise_wavs;   // each already @ 16 kHz
        if (!noise_dir.empty()) {
            if (!fs::exists(noise_dir)) die("noise-dir does not exist: " + noise_dir);
            std::vector<fs::path> np;
            for (const auto& e : fs::directory_iterator(noise_dir)) {
                if (e.is_regular_file() && e.path().extension() == ".wav")
                    np.push_back(e.path());
            }
            std::sort(np.begin(), np.end());
            for (const auto& p : np) {
                auto buf = brosoundml::read_wav(p.string());
                auto rs  = (buf.sample_rate == 16000)
                    ? buf.samples
                    : brosoundml::resample_to(buf.samples, buf.sample_rate, 16000);
                if (!rs.empty()) noise_wavs.push_back(std::move(rs));
            }
            std::fprintf(stderr, "loaded %zu noise wav(s) from %s\n",
                         noise_wavs.size(), noise_dir.c_str());
        }

        const std::vector<brosoundml::NoiseKind> noise_kinds = {
            brosoundml::NoiseKind::White,
            brosoundml::NoiseKind::Pink,
            brosoundml::NoiseKind::Brown,
        };

        // ─── Deterministic per-draw RNG (seed, counter) ────────────────
        std::uint64_t variant_counter = 0;
        auto sub_rng = [&]() -> std::mt19937 {
            const std::uint64_t mix = seed +
                static_cast<std::uint64_t>(variant_counter++) *
                0x9E3779B97F4A7C15ull;
            return std::mt19937(static_cast<std::uint32_t>(mix ^ (mix >> 32)));
        };

        // Build a length-n noise buffer: random WAV cropped/looped, else
        // synthetic. `rng` drives every random draw so runs are reproducible.
        auto make_noise = [&](int n, std::mt19937& rng) -> std::vector<float> {
            if (!noise_wavs.empty()) {
                std::uniform_int_distribution<std::size_t> pick(0, noise_wavs.size() - 1);
                const auto& src = noise_wavs[pick(rng)];
                std::vector<float> out(static_cast<std::size_t>(n));
                std::uniform_int_distribution<std::size_t> off(0, src.size() - 1);
                std::size_t start = off(rng);
                for (int i = 0; i < n; ++i)
                    out[static_cast<std::size_t>(i)] = src[(start + i) % src.size()];
                return out;
            }
            std::uniform_int_distribution<std::size_t> pick(0, noise_kinds.size() - 1);
            return brosoundml::gen_noise(noise_kinds[pick(rng)], n, 1.0f, rng);
        };

        // ─── Output file + writer ──────────────────────────────────────
        const fs::path outp(out_path);
        if (outp.has_parent_path()) fs::create_directories(outp.parent_path());
        brosoundml::PhonemeDatasetWriter writer(out_path, header, classmap);

        // Render text at a given Kokoro speed → 16 kHz PCM + aligned labels.
        // Speed perturbation rides Kokoro's own duration prediction, so the
        // labels built from pred_dur are exact for that speed (no resample).
        auto render = [&](const std::string& text, const brosoundml::Voice& voice,
                          float speed, std::vector<int16_t>& labels_out)
            -> std::vector<float> {
            auto ids = phon.phonemize(text);
            if (ids.empty()) { labels_out.clear(); return {}; }
            std::vector<int32_t> pred_dur;
            auto audio = k.synthesize(ids, voice, speed, &pred_dur);
            auto pcm16 = brosoundml::resample_to(audio.samples, kokoro_sr, 16000);
            if (static_cast<int>(pcm16.size()) < kWin) { labels_out.clear(); return {}; }
            labels_out = brosoundml::build_frame_labels(
                pred_dur, ids, classmap,
                static_cast<int>(pcm16.size()), kWin, kHop);
            return pcm16;
        };

        int   clips_written = 0, skipped = 0;
        long long total_frames = 0;

        for (std::size_t si = 0; si < corpus.size(); ++si) {
            for (std::size_t vi = 0; vi < voices.size(); ++vi) {
                // 1) Clean variant — canonical speed 1.0, no noise.
                {
                    std::vector<int16_t> labels;
                    auto pcm = render(corpus[si], voices[vi], 1.0f, labels);
                    if (pcm.empty() || labels.empty()) { ++skipped; }
                    else {
                        writer.append(pcm, labels);
                        ++clips_written;
                        total_frames += static_cast<long long>(labels.size());
                    }
                }

                // 2) Augmented variants — draw (speed, snr), add noise/gain.
                for (int av = 0; av < n_aug; ++av) {
                    auto rng = sub_rng();
                    std::uniform_int_distribution<std::size_t> sp_pick(0, speeds.size() - 1);
                    std::uniform_int_distribution<std::size_t> snr_pick(0, snr_tokens.size() - 1);
                    const float speed = speeds[sp_pick(rng)];
                    const std::string snr_tok = snr_tokens[snr_pick(rng)];

                    std::vector<int16_t> labels;
                    auto pcm = render(corpus[si], voices[vi], speed, labels);
                    if (pcm.empty() || labels.empty()) { ++skipped; continue; }

                    // Additive noise (length-preserving → labels unchanged).
                    if (snr_tok != "clean") {
                        const float snr_db = std::strtof(snr_tok.c_str(), nullptr);
                        auto noise = make_noise(static_cast<int>(pcm.size()), rng);
                        pcm = brosoundml::mix_at_snr(pcm, noise, snr_db);
                    }
                    // Mild random gain (length-preserving).
                    std::uniform_real_distribution<float> gain_d(-6.0f, 0.0f);
                    brosoundml::apply_gain_db(pcm, gain_d(rng));
                    brosoundml::peak_normalize(pcm, 0.97f);

                    writer.append(pcm, labels);
                    ++clips_written;
                    total_frames += static_cast<long long>(labels.size());
                }
            }
            std::fprintf(stderr, "  sentence %zu/%zu — %d clips written\n",
                         si + 1, corpus.size(), clips_written);
        }

        writer.finalize();

        std::fprintf(stderr,
            "\nDone.\n"
            "  out:            %s\n"
            "  clips written:  %d  (skipped %d empty/short)\n"
            "  total frames:   %lld\n"
            "  classes (K):    %d\n"
            "  voices:         %zu\n"
            "  corpus:         %zu sentences\n"
            "  aug variants:   %d per (sentence,voice)\n",
            out_path.c_str(), clips_written, skipped, total_frames,
            classmap.num_classes, voices.size(), corpus.size(), n_aug);
        return 0;
    } catch (const std::exception& e) {
        die(std::string("error: ") + e.what());
    }
}
