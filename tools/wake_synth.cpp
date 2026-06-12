// brosoundml_wake_synth — synthesize a deterministic wake-word dataset.
//
// Drives Kokoro to produce positives ("computer") plus confusable and
// sentence negatives, layers in synthetic noise + RIR + SNR mixing, resamples
// every clip to 16 kHz mono 16-bit PCM, and writes a CSV manifest alongside
// the WAV files. The dataset is byte-deterministic for a fixed --seed.
//
// AGC-free recipe: every clip is written at a random presentation level
// (peak drawn uniformly in dB, see brosoundml::random_level) instead of a
// fixed peak. The runtime tap is raw — the model must be level-invariant,
// which lets WakeWord share the listen bus's no-AGC stream.

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
    // Confusables fall into two groups:
    //  1) Real English words that share spelling / morphology with "computer"
    //     ("compute", "computes", "commuter", ...).
    //  2) Nonsense words that share the *acoustic* trajectory /kəm-pjuː-tɚ/ —
    //     one phoneme off the target. The chunk-7 probe showed the previous
    //     all-real-word list left the model free to ignore phonetic structure
    //     because all confusables shared the same vowel sequence; adding
    //     phoneme-neighbour nonsense forces the model to require the exact
    //     /k-ə-m-p-j-uː-t-ɚ/ trajectory rather than a band-energy heuristic.
    std::string confusables_csv =
        // group 1 — spelling neighbours
        "compute,computes,computing,computed,commuter,commuters,"
        "completer,putter,computa,compu,"
        // group 2 — acoustic neighbours (Kokoro-pronounceable nonsense)
        "kombuter,pomputer,tomputer,gomputer,bomputer,"
        "computor,computa,compooter,computeer,"
        "kemputer,kamputer,kumputer,kimputer,"
        "compater,compiter,compoter,compyter,"
        "kompukar,konputer,komputter,"
        // group 3 — same-vowel-sequence different-onsets (forces phoneme attention)
        "container,consumer,carpenter,confuser,conductor";
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

        // Acquisition-channel variant of a clean speech clip. This is the
        // synthetic→real lever: every non-anchor variant passes through a
        // randomized recording channel (mic colouration / band-limit / room /
        // proximity / DRC / pitch) before any ambient-noise mix, so training is
        // no longer dominated by Kokoro's single pristine channel. Channel
        // params are not recorded in the (fixed-schema) manifest — they ride the
        // seed, so the dataset stays byte-deterministic per --seed.
        //
        //   nv == 0 in full mode → channel-only (no ambient noise): disentangles
        //   "channel" from "noise" so the model learns channel-invariance even
        //   on otherwise-clean clips. nv >= 1 → channel + noise at SNR. In
        //   --small (one variant) we keep the single variant noisy so the smoke
        //   dataset still exercises the noise path.
        //
        // `rng` is the per-variant sub_rng(); the channel draws from it before
        // the noise generator. On return out_nk/out_snr hold the manifest values
        // ("" / 0 for channel-only) and out_tag the filename discriminator.
        //
        // The presentation level is drawn LAST (after the channel's
        // level-sensitive DRC and the SNR mix): the runtime tap is raw — no
        // AGC — so the level a clip arrives at is a free variable the model
        // must not depend on, not a contract the dataset gets to assume.
        auto make_acq_variant = [&](const std::vector<float>& clean, int nv,
                                    std::mt19937& rng, std::string& out_nk,
                                    float& out_snr, std::string& out_tag)
            -> std::vector<float> {
            auto y = brosoundml::apply_acquisition_channel(
                clean, 16000, target_samples, rng);
            const bool chan_only = (nv == 0 && noise_variants_per_clean >= 2);
            if (chan_only) {
                brosoundml::random_level(y, rng);
                out_nk = "";
                out_snr = 0.0f;
                out_tag = "chan";
                return y;
            }
            const brosoundml::NoiseKind nk =
                all_noises[static_cast<std::size_t>(
                    nv % static_cast<int>(all_noises.size()))];
            const float snr = snrs[static_cast<std::size_t>(
                nv % static_cast<int>(snrs.size()))];
            auto noise = brosoundml::gen_noise(nk, target_samples, 1.0f, rng);
            auto mixed = brosoundml::mix_at_snr(y, noise, snr);
            brosoundml::random_level(mixed, rng);
            out_nk = brosoundml::noise_kind_name(nk);
            out_snr = snr;
            char tg[32];
            std::snprintf(tg, sizeof(tg), "chan_%s_snr%d", out_nk.c_str(),
                          static_cast<int>(snr));
            out_tag = tg;
            return mixed;
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
                // Write the anchor at a random presentation level; keep
                // `clean` at its native level for the channel variants (the
                // channel's DRC stage is level-sensitive).
                auto leveled = clean;
                {
                    auto lrng = sub_rng();
                    brosoundml::random_level(leveled, lrng);
                }
                write_clip(leveled, "positives", buf, 1, "positive",
                           voice_names[vi], speed, 0.0f, "", clean_seed);
                ++positives;

                // Acquisition-channel variants (channel-only + channel+noise).
                for (int nv = 0; nv < noise_variants_per_clean; ++nv) {
                    auto rng = sub_rng();
                    std::string nk_name, tag;
                    float snr_used = 0.0f;
                    auto out = make_acq_variant(clean, nv, rng, nk_name,
                                                snr_used, tag);
                    std::snprintf(buf, sizeof(buf), "pos_%s_sp%03d_%s",
                                  voice_names[vi].c_str(),
                                  static_cast<int>(speed * 100 + 0.5f),
                                  tag.c_str());
                    write_clip(out, "positives", buf, 1, "positive",
                               voice_names[vi], speed, snr_used,
                               nk_name, clean_seed + nv + 1);
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
                    auto leveled = clean;
                    {
                        auto lrng = sub_rng();
                        brosoundml::random_level(leveled, lrng);
                    }
                    write_clip(leveled, "negatives", buf, 0, "confusable",
                               voice_names[vi], speed, 0.0f, "", clean_seed);
                    ++conf_neg;
                    for (int nv = 0; nv < noise_variants_per_clean; ++nv) {
                        if (small_conf_cap >= 0 && conf_neg >= small_conf_cap) break;
                        auto rng = sub_rng();
                        std::string nk_name, tag;
                        float snr_used = 0.0f;
                        auto out = make_acq_variant(clean, nv, rng, nk_name,
                                                    snr_used, tag);
                        std::snprintf(buf, sizeof(buf),
                                      "neg_conf_%s_%s_sp%03d_%s",
                                      word.c_str(),
                                      voice_names[vi].c_str(),
                                      static_cast<int>(speed * 100 + 0.5f),
                                      tag.c_str());
                        write_clip(out, "negatives", buf, 0, "confusable",
                                   voice_names[vi], speed, snr_used,
                                   nk_name, clean_seed + nv + 1);
                        ++conf_neg;
                    }
                }
            }
        }

        // ─── Sentence negatives ────────────────────────────────────────
        // Sentence negatives carry both clean and noisy variants so the noise
        // distribution is symmetric with positives + confusables. Without
        // this, every noisy-speech clip in training is either a positive or a
        // confusable, and BC-ResNet learns "speech + noise ⇒ positive" as a
        // free shortcut. Same noise/SNR rotation as the positive path.
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
                    char buf[96];
                    std::snprintf(buf, sizeof(buf),
                                  "neg_sent_%03zu_%s_sp%03d_clean",
                                  si,
                                  voice_names[vi].c_str(),
                                  static_cast<int>(speed * 100 + 0.5f));
                    auto leveled = clean;
                    {
                        auto lrng = sub_rng();
                        brosoundml::random_level(leveled, lrng);
                    }
                    write_clip(leveled, "negatives", buf, 0, "sentence",
                               voice_names[vi], speed, 0.0f, "", clean_seed);
                    ++sent_neg;

                    // Acquisition-channel variants — mirror positive/confusable.
                    for (int nv = 0; nv < noise_variants_per_clean; ++nv) {
                        if (small_sent_cap >= 0 && sent_neg >= small_sent_cap) break;
                        auto rng = sub_rng();
                        std::string nk_name, tag;
                        float snr_used = 0.0f;
                        auto out = make_acq_variant(clean, nv, rng, nk_name,
                                                    snr_used, tag);
                        std::snprintf(buf, sizeof(buf),
                                      "neg_sent_%03zu_%s_sp%03d_%s",
                                      si,
                                      voice_names[vi].c_str(),
                                      static_cast<int>(speed * 100 + 0.5f),
                                      tag.c_str());
                        write_clip(out, "negatives", buf, 0, "sentence",
                                   voice_names[vi], speed, snr_used,
                                   nk_name, clean_seed + nv + 1);
                        ++sent_neg;
                    }
                }
            }
        }

        // ─── Pure noise negatives ──────────────────────────────────────
        std::fprintf(stderr, "synthesizing pure-noise negatives...\n");
        // Level diversity comes from the same random presentation draw as
        // every other clip. (The old gain sweep here was a no-op: the final
        // peak_normalize(0.99) undid apply_gain_db on every clip.)
        const int noise_count = small_mode ? 10 : 60;
        for (int i = 0; i < noise_count; ++i) {
            auto rng = sub_rng();
            const brosoundml::NoiseKind nk =
                all_noises[static_cast<std::size_t>(
                    i % static_cast<int>(all_noises.size()))];
            auto buf = brosoundml::gen_noise(nk, target_samples, 1.0f, rng);
            brosoundml::random_level(buf, rng);
            char nm[64];
            std::snprintf(nm, sizeof(nm), "neg_noise_%s_%03d",
                          brosoundml::noise_kind_name(nk), i);
            const std::uint64_t row_seed = (master_rng)();
            write_clip(buf, "negatives", nm, 0, "noise", "", 1.0f, 0.0f,
                       brosoundml::noise_kind_name(nk), row_seed);
            ++noise_neg;
        }

        // ─── Probe-style synthetic stimulus negatives ──────────────────
        //
        // The chunk-7 wake_probe surfaced a band-energy shortcut: the small
        // model fired on any sustained tone in the 500-2000 Hz / 5-7 kHz
        // bands, on two-tone formant pairs, on AM tones, and on sweeps
        // through those bands. It also produced a 0.51 score on pure silence
        // because no truly-silent samples appear anywhere in the negative
        // class. We close every one of those by emitting the same stimulus
        // families as labelled negatives — directly anchoring the failure
        // modes to label=0 during training.
        std::fprintf(stderr, "synthesizing probe-style negatives...\n");
        int probe_neg = 0;

        const auto pi_d = 3.14159265358979323846;
        auto write_synth_neg = [&](const std::vector<float>& clip,
                                   const std::string& tag,
                                   const std::string& clazz,
                                   const std::string& noise_kind_tag,
                                   float snr_for_row) {
            const std::uint64_t row_seed = (master_rng)();
            write_clip(clip, "negatives", tag, 0, clazz, "", 1.0f,
                       snr_for_row, noise_kind_tag, row_seed);
            ++probe_neg;
        };

        // (a) Pure silence + near-silence — closes the 0.51 silence baseline.
        // Several samples so the model sees silence with a stable label.
        {
            const int n_silence = small_mode ? 5 : 40;
            for (int i = 0; i < n_silence; ++i) {
                std::vector<float> z(target_samples, 0.0f);
                char nm[64];
                std::snprintf(nm, sizeof(nm), "neg_silence_zero_%03d", i);
                write_synth_neg(z, nm, "silence", "", 0.0f);
            }
            // Near-silence: very-low-amplitude white noise (mic floor).
            const std::vector<float> floor_amps = {1e-5f, 1e-4f, 1e-3f};
            const int n_floor = small_mode ? 6 : 30;
            for (int i = 0; i < n_floor; ++i) {
                auto rng = sub_rng();
                const float amp = floor_amps[
                    static_cast<std::size_t>(i % floor_amps.size())];
                auto buf = brosoundml::gen_white_noise(target_samples, amp, rng);
                char nm[64];
                std::snprintf(nm, sizeof(nm),
                              "neg_silence_floor_%03d_amp%.0e", i, amp);
                write_synth_neg(buf, nm, "silence", "white", 0.0f);
            }
        }

        // (b) Sustained pure tones at the probe's loved frequencies × amps.
        //     Each tone gets one clean and one noisy variant. Forces the
        //     model off "narrow band of energy in 500-2000/5-7k Hz ⇒
        //     positive" by labelling that exact stimulus as negative.
        {
            const std::vector<float> tone_hz = small_mode
                ? std::vector<float>{500.f, 1000.f, 2000.f}
                : std::vector<float>{300.f, 500.f, 800.f, 1000.f, 1200.f,
                                     1500.f, 1800.f, 2000.f, 2500.f, 3000.f,
                                     4000.f, 5000.f, 6000.f, 7000.f};
            const std::vector<float> tone_amps = small_mode
                ? std::vector<float>{0.1f, 0.5f}
                : std::vector<float>{0.02f, 0.05f, 0.1f, 0.3f, 0.5f, 0.7f};
            int ti = 0;
            for (float hz : tone_hz) {
                for (float amp : tone_amps) {
                    std::vector<float> tone(target_samples);
                    const double w = 2.0 * pi_d * hz / 16000.0;
                    for (int n = 0; n < target_samples; ++n) {
                        tone[static_cast<std::size_t>(n)] =
                            amp * static_cast<float>(std::sin(w * n));
                    }
                    char nm[80];
                    std::snprintf(nm, sizeof(nm),
                                  "neg_tone_%05dhz_amp%03d",
                                  static_cast<int>(hz),
                                  static_cast<int>(amp * 100 + 0.5f));
                    write_synth_neg(tone, nm, "tone", "", 0.0f);
                    if (!small_mode && (ti % 2 == 0)) {
                        auto rng = sub_rng();
                        const brosoundml::NoiseKind nk =
                            all_noises[static_cast<std::size_t>(
                                ti % all_noises.size())];
                        auto nz = brosoundml::gen_noise(nk, target_samples,
                                                       1.0f, rng);
                        auto mixed = brosoundml::mix_at_snr(tone, nz, 10.0f);
                        brosoundml::random_level(mixed, rng);
                        char nm2[96];
                        std::snprintf(nm2, sizeof(nm2),
                                      "neg_tone_%05dhz_amp%03d_%s_snr10",
                                      static_cast<int>(hz),
                                      static_cast<int>(amp * 100 + 0.5f),
                                      brosoundml::noise_kind_name(nk));
                        write_synth_neg(mixed, nm2, "tone",
                                        brosoundml::noise_kind_name(nk), 10.0f);
                    }
                    ++ti;
                }
            }
        }

        // (c) Two-tone formant pairs — every vowel in /kəm-pjuː-tɚ/ in
        //     isolation scored 1.0 in the probe. Label them as negative when
        //     sustained for the full second so the model has to require
        //     temporal vowel transitions, not steady-state vowel energy.
        {
            struct FP { const char* name; float f1; float f2; };
            const std::vector<FP> fps = {
                {"ee", 300.f, 2800.f}, {"ih", 400.f, 2000.f},
                {"eh", 600.f, 1900.f}, {"ae", 700.f, 1700.f},
                {"ah", 800.f, 1200.f}, {"aw", 600.f,  900.f},
                {"uh", 600.f, 1200.f}, {"oo", 300.f,  900.f},
                {"ow", 500.f,  800.f}, {"er", 500.f, 1400.f}
            };
            const std::vector<float> fp_amps = small_mode
                ? std::vector<float>{0.3f}
                : std::vector<float>{0.1f, 0.3f, 0.5f};
            for (const auto& fp : fps) {
                for (float amp : fp_amps) {
                    std::vector<float> tone(target_samples);
                    const double w1 = 2.0 * pi_d * fp.f1 / 16000.0;
                    const double w2 = 2.0 * pi_d * fp.f2 / 16000.0;
                    for (int n = 0; n < target_samples; ++n) {
                        tone[static_cast<std::size_t>(n)] = amp * 0.5f *
                            (static_cast<float>(std::sin(w1 * n)) +
                             static_cast<float>(std::sin(w2 * n)));
                    }
                    char nm[96];
                    std::snprintf(nm, sizeof(nm),
                                  "neg_formant_%s_%04d_%04d_amp%03d",
                                  fp.name,
                                  static_cast<int>(fp.f1),
                                  static_cast<int>(fp.f2),
                                  static_cast<int>(amp * 100 + 0.5f));
                    write_synth_neg(tone, nm, "formant", "", 0.0f);
                }
            }
        }

        // (d) AM-modulated tones — speech-rate envelope on speech-band
        //     carriers; the probe lit these up at 1.0 in the loved bands.
        {
            const std::vector<float> carriers = small_mode
                ? std::vector<float>{1000.f}
                : std::vector<float>{500.f, 1000.f, 1500.f, 2000.f, 5000.f};
            const std::vector<float> mods = small_mode
                ? std::vector<float>{10.f}
                : std::vector<float>{4.f, 10.f, 25.f};
            const std::vector<float> am_amps = small_mode
                ? std::vector<float>{0.3f}
                : std::vector<float>{0.1f, 0.3f, 0.5f};
            for (float c : carriers) {
                for (float m : mods) {
                    for (float amp : am_amps) {
                        std::vector<float> buf(target_samples);
                        const double wc = 2.0 * pi_d * c / 16000.0;
                        const double wm = 2.0 * pi_d * m / 16000.0;
                        for (int n = 0; n < target_samples; ++n) {
                            const double env = 0.5 + 0.5 * std::sin(wm * n);
                            buf[static_cast<std::size_t>(n)] = amp *
                                static_cast<float>(env * std::sin(wc * n));
                        }
                        char nm[96];
                        std::snprintf(nm, sizeof(nm),
                                      "neg_am_c%04d_m%02d_amp%03d",
                                      static_cast<int>(c),
                                      static_cast<int>(m),
                                      static_cast<int>(amp * 100 + 0.5f));
                        write_synth_neg(buf, nm, "am", "", 0.0f);
                    }
                }
            }
        }

        // (e) Linear chirps — sweeps that pass through the loved bands all
        //     scored 0.9+. Label them as negative.
        if (!small_mode) {
            struct Sw { float f0; float f1; };
            const std::vector<Sw> sweeps = {
                {200.f, 4000.f}, {4000.f, 200.f},
                {100.f, 8000.f}, {8000.f, 100.f},
                {500.f, 2000.f}, {2000.f, 500.f}
            };
            const std::vector<float> sw_amps = {0.1f, 0.3f, 0.5f};
            for (const auto& s : sweeps) {
                for (float amp : sw_amps) {
                    std::vector<float> buf(target_samples);
                    const double T = static_cast<double>(target_samples) / 16000.0;
                    const double k = (s.f1 - s.f0) / T;
                    for (int n = 0; n < target_samples; ++n) {
                        const double t = static_cast<double>(n) / 16000.0;
                        const double phase = 2.0 * pi_d *
                            (s.f0 * t + 0.5 * k * t * t);
                        buf[static_cast<std::size_t>(n)] =
                            amp * static_cast<float>(std::sin(phase));
                    }
                    char nm[96];
                    std::snprintf(nm, sizeof(nm),
                                  "neg_sweep_%04d_to_%04d_amp%03d",
                                  static_cast<int>(s.f0),
                                  static_cast<int>(s.f1),
                                  static_cast<int>(amp * 100 + 0.5f));
                    write_synth_neg(buf, nm, "sweep", "", 0.0f);
                }
            }
        }

        std::fprintf(stderr,
                     "\nDone.\n"
                     "  positives:        %d\n"
                     "  confusables:      %d\n"
                     "  sentences:        %d\n"
                     "  pure-noise:       %d\n"
                     "  probe-style:      %d\n"
                     "  total wav bytes:  %lld\n"
                     "  manifest:         %s\n",
                     positives, conf_neg, sent_neg, noise_neg, probe_neg,
                     static_cast<long long>(total_bytes),
                     (out_dir + "/manifest.csv").c_str());
        return 0;
    } catch (const std::exception& e) {
        die(std::string("error: ") + e.what());
    }
}
