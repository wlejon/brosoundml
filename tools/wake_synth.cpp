// brosoundml_wake_synth — synthesize a deterministic wake-word dataset.
//
// Drives Kokoro to produce positives ("computer") plus confusable and
// sentence negatives, layers in synthetic noise + RIR + SNR mixing, resamples
// every clip to 16 kHz mono 16-bit PCM, and writes a CSV manifest alongside
// the WAV files. The dataset is byte-deterministic for a fixed --seed.

#include "brosoundml/audio.h"
#include "brosoundml/kokoro.h"
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
#include <cstring>
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
    std::fprintf(stderr, "wake_synth: %s\n", msg.c_str());
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
    if (fs::exists("../brosoundml-data")) return "../brosoundml-data";
    return "../brosoundml-data";   // returned anyway; caller will fail later
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t a = tok.find_first_not_of(" \t\r\n");
        size_t b = tok.find_last_not_of (" \t\r\n");
        if (a == std::string::npos) continue;
        out.push_back(tok.substr(a, b - a + 1));
    }
    return out;
}

// 16-bit PCM WAV writer at a specific rate. brosoundml::AudioBuffer::write_wav
// uses its own `sample_rate`, so we set that before calling it.
void write_wav_16k(const std::vector<float>& samples, const std::string& path) {
    brosoundml::AudioBuffer buf(samples, 16000);
    buf.write_wav(path);
}

// Vocab parser — Kokoro's KokoroConfig::vocab is already loaded, so we just
// reach into it via the phonemizer wrapper.

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir   = "weights/kokoro";
    std::string voices_dir  = "";
    std::string out_dir     = "../brosoundml-data/wake/computer";
    std::string target_word = "computer";
    std::string confusables_csv =
        "compute,computes,computing,computed,commuter,commuters,"
        "completer,putter,computa,compu";
    std::string lexicon_path;
    std::string pos_path;
    std::string device_str = "cpu";
    std::uint64_t seed = 42;
    bool small_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string(what) + " needs a value");
            return argv[++i];
        };
        if      (a == "--model-dir")   model_dir   = next("--model-dir");
        else if (a == "--voices-dir")  voices_dir  = next("--voices-dir");
        else if (a == "--out-dir")     out_dir     = next("--out-dir");
        else if (a == "--target")      target_word = next("--target");
        else if (a == "--confusables") confusables_csv = next("--confusables");
        else if (a == "--lexicon")     lexicon_path = next("--lexicon");
        else if (a == "--pos-tagger")  pos_path    = next("--pos-tagger");
        else if (a == "--device")      device_str  = next("--device");
        else if (a == "--seed")        seed        = std::strtoull(
                                                   next("--seed").c_str(), nullptr, 10);
        else if (a == "--small")       small_mode  = true;
        else if (a == "-h" || a == "--help") {
            std::printf(
                "Usage: brosoundml_wake_synth [flags]\n"
                "  --model-dir DIR     Kokoro model directory (default weights/kokoro)\n"
                "  --voices-dir DIR    Voice pack directory (default <model-dir>/voices)\n"
                "  --out-dir DIR       Output root (default ../brosoundml-data/wake/computer)\n"
                "  --target WORD       Positive target word (default computer)\n"
                "  --confusables CSV   Comma-separated confusable negatives\n"
                "  --lexicon PATH      G2P lexicon (.bin)\n"
                "  --pos-tagger PATH   POS tagger (.bin)\n"
                "  --device cpu|cuda   Kokoro inference device (default cpu)\n"
                "  --seed N            Deterministic RNG seed (default 42)\n"
                "  --small             Tiny dataset variant for tests / iteration\n");
            return 0;
        }
        else die("unknown flag '" + a + "'");
    }

    if (voices_dir.empty()) voices_dir = model_dir + "/voices";
    const std::string data_dir = default_data_dir();
    if (lexicon_path.empty()) lexicon_path = data_dir + "/g2p/lexicon_en_us.bin";
    if (pos_path.empty())     pos_path     = data_dir + "/pos_tagger/model.bin";

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
        g::Lexicon       lex     = g::Lexicon::load(lexicon_path);
        g::Morphology    morph(lex);
        g::SpecialCases  sc(lex);
        g::PosTagger     tagger  = g::PosTagger::load(pos_path);
        g::PhonemeAdapter adapter(k.config().vocab);
        g::Phonemizer    phon(tagger, lex, morph, sc, adapter);

        // ─── Grid configuration ────────────────────────────────────────
        const std::vector<float> all_speeds   = {0.85f, 0.95f, 1.0f, 1.05f, 1.15f};
        const std::vector<float> small_speeds = {1.0f};
        const std::vector<float>& speeds = small_mode ? small_speeds : all_speeds;

        const std::vector<brosoundml::NoiseKind> all_noises = {
            brosoundml::NoiseKind::White,
            brosoundml::NoiseKind::Pink,
            brosoundml::NoiseKind::Brown,
        };
        const std::vector<float> all_snrs   = {20.0f, 10.0f, 5.0f, 0.0f};
        const std::vector<float> small_snrs = {10.0f};
        const std::vector<float>& snrs = small_mode ? small_snrs : all_snrs;
        const int noise_variants_per_clean = small_mode ? 1 : 4;

        const auto confusables_full = split_csv(confusables_csv);
        const std::vector<std::string> confusables_small =
            confusables_full.empty()
                ? std::vector<std::string>{}
                : std::vector<std::string>(confusables_full.begin(),
                    confusables_full.begin() +
                    std::min<std::size_t>(3, confusables_full.size()));
        const std::vector<std::string>& confusables =
            small_mode ? confusables_small : confusables_full;

        // 30 sentences sampled to cover a broad phonetic range.
        const std::vector<std::string> sentences_full = {
            "the weather is nice today",
            "please pass the salt",
            "i just got back from the store",
            "she said hello to the cat",
            "we will leave in five minutes",
            "the dog ran across the yard",
            "open the window for some air",
            "i love the smell of fresh coffee",
            "could you turn down the music",
            "his car is parked in the driveway",
            "the children played in the park",
            "do you want to go for a walk",
            "i have a meeting at three o'clock",
            "the train arrives at noon",
            "she bought a new pair of shoes",
            "i forgot to lock the front door",
            "the soup needs a little more pepper",
            "the moon was bright last night",
            "he reads a book before bed",
            "they are coming to dinner tomorrow",
            "the printer is out of paper again",
            "i need to charge my phone",
            "the river was calm and clear",
            "we should plant flowers in the spring",
            "the museum closes at six",
            "remember to bring an umbrella",
            "the bread is still warm",
            "she drives to work every morning",
            "let's order pizza tonight",
            "the cat sleeps on the couch",
        };
        const std::vector<std::string> sentences_small(
            sentences_full.begin(),
            sentences_full.begin() + std::min<std::size_t>(5, sentences_full.size()));
        const std::vector<std::string>& sentences =
            small_mode ? sentences_small : sentences_full;

        const int   target_samples = 16000;   // 1 s @ 16 kHz
        const int   kokoro_sr      = k.config().sample_rate;

        // ─── Output tree ───────────────────────────────────────────────
        fs::create_directories(out_dir);
        fs::create_directories(out_dir + "/positives");
        fs::create_directories(out_dir + "/negatives");
        brosoundml::Manifest manifest(out_dir + "/manifest.csv");

        std::mt19937_64 master_rng(seed);
        std::uint64_t variant_counter = 0;
        auto sub_rng = [&]() -> std::mt19937 {
            // Derive a sub-seed deterministically from (seed, counter). Mixing
            // by a large odd prime keeps adjacent counters from producing
            // visibly-correlated noise.
            const std::uint64_t mix = seed +
                static_cast<std::uint64_t>(variant_counter++) *
                0x9E3779B97F4A7C15ull;
            return std::mt19937(static_cast<std::uint32_t>(mix ^ (mix >> 32)));
        };

        int positives = 0, conf_neg = 0, sent_neg = 0, noise_neg = 0;
        std::int64_t total_bytes = 0;
        int total_clips = 0;

        auto write_clip = [&](const std::vector<float>& clip16,
                              const std::string& subdir,
                              const std::string& tag,
                              int label,
                              const std::string& clazz,
                              const std::string& voice,
                              float speed,
                              float snr_db,
                              const std::string& noise_kind,
                              std::uint64_t row_seed) {
            const std::string rel = subdir + "/" + tag + ".wav";
            const std::string abs = out_dir + "/" + rel;
            write_wav_16k(clip16, abs);
            // header (44) + 2 bytes/sample
            total_bytes += 44 + static_cast<std::int64_t>(clip16.size()) * 2;
            brosoundml::ManifestRow row;
            row.path = rel;
            row.label = label;
            row.clazz = clazz;
            row.voice = voice;
            row.speed = speed;
            row.snr_db = snr_db;
            row.noise_kind = noise_kind;
            row.seed = row_seed;
            manifest.append(row);
            ++total_clips;
            if (total_clips % 100 == 0)
                std::fprintf(stderr, "  ... %d clips written\n", total_clips);
        };

        // Per-word synth + crop + resample helper. Returns the 1-s 16 kHz clip.
        auto synth_word = [&](const std::string& text,
                              const brosoundml::Voice& voice,
                              float speed) -> std::vector<float> {
            const auto ids = phon.phonemize(text);
            if (ids.empty()) return std::vector<float>(target_samples, 0.0f);
            auto audio = k.synthesize(ids, voice, speed);
            // audio.samples is FP32 @ 24 kHz; resample to 16 kHz then crop/pad.
            auto rs16 = brosoundml::resample_to(audio.samples,
                                                kokoro_sr, 16000);
            return brosoundml::crop_or_pad_centered(rs16, target_samples);
        };

        // ─── Positives ─────────────────────────────────────────────────
        std::fprintf(stderr, "synthesizing positives for '%s'...\n",
                     target_word.c_str());
        // Optionally cap positives in --small to ~10 total: stop after enough.
        const int small_positive_cap = small_mode ? 10 : -1;
        for (std::size_t vi = 0; vi < voices.size(); ++vi) {
            if (small_positive_cap >= 0 && positives >= small_positive_cap) break;
            for (float speed : speeds) {
                if (small_positive_cap >= 0 && positives >= small_positive_cap) break;
                const std::uint64_t clean_seed = (master_rng)();
                std::vector<float> clean = synth_word(target_word,
                                                     voices[vi], speed);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "pos_%s_sp%03d_clean",
                              voice_names[vi].c_str(),
                              static_cast<int>(speed * 100 + 0.5f));
                write_clip(clean, "positives", buf, 1, "positive",
                           voice_names[vi], speed, 0.0f, "", clean_seed);
                ++positives;

                // Noisy variants.
                for (int nv = 0; nv < noise_variants_per_clean; ++nv) {
                    auto rng = sub_rng();
                    const brosoundml::NoiseKind nk =
                        all_noises[static_cast<std::size_t>(
                            nv % static_cast<int>(all_noises.size()))];
                    const float snr = snrs[static_cast<std::size_t>(
                        nv % static_cast<int>(snrs.size()))];
                    auto noise = brosoundml::gen_noise(
                        nk, target_samples, 1.0f, rng);
                    auto mixed = brosoundml::mix_at_snr(clean, noise, snr);
                    brosoundml::peak_normalize(mixed, 0.99f);
                    std::snprintf(buf, sizeof(buf),
                                  "pos_%s_sp%03d_%s_snr%d",
                                  voice_names[vi].c_str(),
                                  static_cast<int>(speed * 100 + 0.5f),
                                  brosoundml::noise_kind_name(nk),
                                  static_cast<int>(snr));
                    write_clip(mixed, "positives", buf, 1, "positive",
                               voice_names[vi], speed, snr,
                               brosoundml::noise_kind_name(nk), clean_seed + nv + 1);
                    ++positives;
                    if (small_positive_cap >= 0 && positives >= small_positive_cap) break;
                }
            }
        }

        // ─── Confusable negatives ──────────────────────────────────────
        std::fprintf(stderr, "synthesizing confusable negatives...\n");
        const int small_conf_cap = small_mode ? 20 : -1;
        for (const auto& word : confusables) {
            if (small_conf_cap >= 0 && conf_neg >= small_conf_cap) break;
            for (std::size_t vi = 0; vi < voices.size(); ++vi) {
                if (small_conf_cap >= 0 && conf_neg >= small_conf_cap) break;
                for (float speed : speeds) {
                    if (small_conf_cap >= 0 && conf_neg >= small_conf_cap) break;
                    const std::uint64_t clean_seed = (master_rng)();
                    auto clean = synth_word(word, voices[vi], speed);
                    char buf[96];
                    std::snprintf(buf, sizeof(buf), "neg_conf_%s_%s_sp%03d_clean",
                                  word.c_str(),
                                  voice_names[vi].c_str(),
                                  static_cast<int>(speed * 100 + 0.5f));
                    write_clip(clean, "negatives", buf, 0, "confusable",
                               voice_names[vi], speed, 0.0f, "", clean_seed);
                    ++conf_neg;
                    for (int nv = 0; nv < noise_variants_per_clean; ++nv) {
                        if (small_conf_cap >= 0 && conf_neg >= small_conf_cap) break;
                        auto rng = sub_rng();
                        const brosoundml::NoiseKind nk =
                            all_noises[static_cast<std::size_t>(
                                nv % static_cast<int>(all_noises.size()))];
                        const float snr = snrs[static_cast<std::size_t>(
                            nv % static_cast<int>(snrs.size()))];
                        auto noise = brosoundml::gen_noise(
                            nk, target_samples, 1.0f, rng);
                        auto mixed = brosoundml::mix_at_snr(clean, noise, snr);
                        brosoundml::peak_normalize(mixed, 0.99f);
                        std::snprintf(buf, sizeof(buf),
                                      "neg_conf_%s_%s_sp%03d_%s_snr%d",
                                      word.c_str(),
                                      voice_names[vi].c_str(),
                                      static_cast<int>(speed * 100 + 0.5f),
                                      brosoundml::noise_kind_name(nk),
                                      static_cast<int>(snr));
                        write_clip(mixed, "negatives", buf, 0, "confusable",
                                   voice_names[vi], speed, snr,
                                   brosoundml::noise_kind_name(nk),
                                   clean_seed + nv + 1);
                        ++conf_neg;
                    }
                }
            }
        }

        // ─── Sentence negatives ────────────────────────────────────────
        std::fprintf(stderr, "synthesizing sentence negatives...\n");
        const int small_sent_cap = small_mode ? 20 : -1;
        for (std::size_t si = 0; si < sentences.size(); ++si) {
            if (small_sent_cap >= 0 && sent_neg >= small_sent_cap) break;
            for (std::size_t vi = 0; vi < voices.size(); ++vi) {
                if (small_sent_cap >= 0 && sent_neg >= small_sent_cap) break;
                for (float speed : speeds) {
                    if (small_sent_cap >= 0 && sent_neg >= small_sent_cap) break;
                    const std::uint64_t clean_seed = (master_rng)();
                    auto clean = synth_word(sentences[si], voices[vi], speed);
                    char buf[64];
                    std::snprintf(buf, sizeof(buf),
                                  "neg_sent_%03zu_%s_sp%03d",
                                  si,
                                  voice_names[vi].c_str(),
                                  static_cast<int>(speed * 100 + 0.5f));
                    write_clip(clean, "negatives", buf, 0, "sentence",
                               voice_names[vi], speed, 0.0f, "", clean_seed);
                    ++sent_neg;
                }
            }
        }

        // ─── Pure noise negatives ──────────────────────────────────────
        std::fprintf(stderr, "synthesizing pure-noise negatives...\n");
        const int noise_count = small_mode ? 10 : 60;
        const std::vector<float> gains_db = {-30.0f, -24.0f, -18.0f, -12.0f, -6.0f};
        for (int i = 0; i < noise_count; ++i) {
            auto rng = sub_rng();
            const brosoundml::NoiseKind nk =
                all_noises[static_cast<std::size_t>(
                    i % static_cast<int>(all_noises.size()))];
            const float gain = gains_db[static_cast<std::size_t>(
                i % static_cast<int>(gains_db.size()))];
            auto buf = brosoundml::gen_noise(nk, target_samples, 1.0f, rng);
            brosoundml::apply_gain_db(buf, gain);
            brosoundml::peak_normalize(buf, 0.99f);
            char nm[64];
            std::snprintf(nm, sizeof(nm), "neg_noise_%s_%03d_g%d",
                          brosoundml::noise_kind_name(nk), i,
                          static_cast<int>(gain));
            const std::uint64_t row_seed = (master_rng)();
            write_clip(buf, "negatives", nm, 0, "noise", "", 1.0f, 0.0f,
                       brosoundml::noise_kind_name(nk), row_seed);
            ++noise_neg;
        }

        std::fprintf(stderr,
                     "\nDone.\n"
                     "  positives:        %d\n"
                     "  confusables:      %d\n"
                     "  sentences:        %d\n"
                     "  pure-noise:       %d\n"
                     "  total wav bytes:  %lld\n"
                     "  manifest:         %s\n",
                     positives, conf_neg, sent_neg, noise_neg,
                     static_cast<long long>(total_bytes),
                     (out_dir + "/manifest.csv").c_str());
        return 0;
    } catch (const std::exception& e) {
        die(std::string("error: ") + e.what());
    }
}
