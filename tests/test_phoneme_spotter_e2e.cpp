// PhonemeSpotter end-to-end test — the open-vocabulary KWS gate "with Kokoro".
//
// Renders keywords with the REAL Kokoro TTS, drives a REAL trained PhonemeNet
// through PhonemeSpotter, and asserts the spotter fires on an enrolled keyword
// but not on a distractor — the full phonemize → synthesize → PCEN mel →
// per-frame posteriors → streaming Viterbi matcher path, exactly as a live mic
// loop would run it (only the audio source is Kokoro instead of a microphone).
//
// Unlike the synthetic-posterior unit test (test_phoneme_spotter.cpp), this one
// needs real artefacts on disk: a Kokoro model dir + a voice, the g2p lexicon /
// POS tagger, and a trained phoneme checkpoint. When any of those is absent the
// test prints SKIP and returns 0 so CI without weights stays green; when present
// (and after the chunk-7 training run produces a checkpoint) it asserts.
//
// Resolution order for the phoneme checkpoint:
//   1. argv[1]
//   2. $BROSOUNDML_PHONEME_WEIGHTS
//   3. <repo>/build-cuda/english.bpm  (the chunk-7 real model)
//   4. <repo>/build-cuda/smoke.bpm    (the chunk-4 smoke model)
// Kokoro dir:  $BROSOUNDML_KOKORO_DIR, then <repo>/weights/kokoro, then
//              <data>/kokoro, then weights/kokoro (cwd-relative).
// g2p data:    $BROSOUNDML_DATA_DIR, then <repo>/../brosoundml-data.

#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "brosoundml/phoneme_spotter.h"
#include "brosoundml/phoneme_model.h"
#include "brosoundml/kokoro.h"
#include "brosoundml/audio.h"
#include "brosoundml/wake_data.h"   // resample_to

#include "brosoundml/g2p/lexicon.h"
#include "brosoundml/g2p/morphology.h"
#include "brosoundml/g2p/phoneme_adapter.h"
#include "brosoundml/g2p/phonemizer.h"
#include "brosoundml/g2p/pos_tagger.h"
#include "brosoundml/g2p/special_cases.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs  = std::filesystem;
namespace bt  = brotensor;
namespace bsm = brosoundml;
namespace g   = brosoundml::g2p;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fail; } \
    else { std::fprintf(stderr, "  ok: %s\n", (msg)); } } while (0)

#ifndef BROSOUNDML_REPO_DIR
#  define BROSOUNDML_REPO_DIR "."
#endif

static std::string env_or_empty(const char* k) {
    const char* v = std::getenv(k);
    return v ? std::string(v) : std::string();
}

// First existing path from a candidate list ("" if none exists).
static std::string first_existing(const std::vector<std::string>& cands) {
    for (const auto& c : cands)
        if (!c.empty() && fs::exists(c)) return c;
    return {};
}

int main(int argc, char** argv) {
    const std::string repo = BROSOUNDML_REPO_DIR;
    const std::string data = [&] {
        const std::string e = env_or_empty("BROSOUNDML_DATA_DIR");
        if (!e.empty() && fs::exists(e)) return e;
        return repo + "/../brosoundml-data";
    }();

    const std::string kokoro_dir = first_existing({
        env_or_empty("BROSOUNDML_KOKORO_DIR"),
        repo + "/weights/kokoro",
        data + "/kokoro",
        "weights/kokoro",
    });
    const std::string lexicon = first_existing({
        data + "/g2p/lexicon_en_us.bin",
        repo + "/../brosoundml-data/g2p/lexicon_en_us.bin",
    });
    const std::string pos = first_existing({
        data + "/pos_tagger/model.bin",
        repo + "/../brosoundml-data/pos_tagger/model.bin",
    });
    const std::string checkpoint = first_existing({
        argc > 1 ? std::string(argv[1]) : std::string(),
        env_or_empty("BROSOUNDML_PHONEME_WEIGHTS"),
        repo + "/build-cuda/english.bpm",
        repo + "/build-cuda/smoke.bpm",
    });

    // Locate a voice pack inside the Kokoro dir.
    std::string voice_path;
    if (!kokoro_dir.empty() && fs::exists(kokoro_dir + "/voices")) {
        std::vector<fs::path> vs;
        for (const auto& e : fs::directory_iterator(kokoro_dir + "/voices"))
            if (e.is_regular_file() && e.path().extension() == ".bin")
                vs.push_back(e.path());
        std::sort(vs.begin(), vs.end());
        if (!vs.empty()) voice_path = vs.front().string();
    }

    if (kokoro_dir.empty() || !fs::exists(kokoro_dir + "/config.json") ||
        voice_path.empty() || lexicon.empty() || pos.empty() ||
        checkpoint.empty()) {
        std::fprintf(stderr,
            "SKIP test_phoneme_spotter_e2e: missing real weights.\n"
            "  kokoro_dir = %s\n  voice      = %s\n  lexicon    = %s\n"
            "  pos_tagger = %s\n  checkpoint = %s\n"
            "  (set BROSOUNDML_KOKORO_DIR / BROSOUNDML_DATA_DIR / "
            "BROSOUNDML_PHONEME_WEIGHTS or pass the checkpoint as argv[1])\n",
            kokoro_dir.c_str(), voice_path.c_str(), lexicon.c_str(),
            pos.c_str(), checkpoint.c_str());
        return 0;   // not a failure — CI without weights stays green
    }

    bt::init();
    bt::Device device = bt::Device::CPU;
    if (bt::is_available(bt::Device::CUDA)) device = bt::Device::CUDA;
    const char* dev_name = (device == bt::Device::CUDA) ? "cuda" : "cpu";
    std::fprintf(stderr, "test_phoneme_spotter_e2e on %s\n", dev_name);
    std::fprintf(stderr, "  kokoro    = %s\n  voice     = %s\n  checkpoint= %s\n",
                 kokoro_dir.c_str(), voice_path.c_str(), checkpoint.c_str());

    try {
        // ─ Kokoro + g2p ─
        bsm::Kokoro k;
        k.load(kokoro_dir, device);
        bsm::Voice voice = k.load_voice(voice_path);
        const int kokoro_sr = k.config().sample_rate;   // 24000

        g::Lexicon        lex   = g::Lexicon::load(lexicon);
        g::Morphology     morph(lex);
        g::SpecialCases   sc(lex);
        g::PosTagger      tagger = g::PosTagger::load(pos);
        g::PhonemeAdapter adapter(k.config().vocab);
        g::Phonemizer     phon(tagger, lex, morph, sc, adapter);

        // ─ Spotter on the trained model ─
        bsm::PhonemeSpotter spotter;
        spotter.load(checkpoint, device);
        std::fprintf(stderr, "  model class map K=%d\n",
                     spotter.class_map().num_classes);

        // Render `text` at `speed` → 16 kHz mono PCM (the rate the model wants).
        auto synth16k = [&](const std::string& text, float speed) {
            auto ids = phon.phonemize(text);
            std::vector<std::int32_t> pred_dur;
            auto audio = k.synthesize(ids, voice, speed, &pred_dur);
            return bsm::resample_to(audio.samples, kokoro_sr, 16000);
        };

        // reset() leaves the entry gate wide open (start-of-stream counts as
        // silence), so a keyword fed right after reset matches without a
        // synthetic silence pad. A short trailing-silence tail lets the M-of-N
        // smoother settle on the final phoneme.
        const std::vector<float> tail(static_cast<std::size_t>(16000 / 4), 0.0f);
        auto feed_word = [&](const std::vector<float>& pcm) {
            spotter.reset();
            std::vector<bsm::SpotEvent> evs = spotter.feed(pcm.data(),
                                                           static_cast<int>(pcm.size()));
            auto more = spotter.feed(tail.data(), static_cast<int>(tail.size()));
            evs.insert(evs.end(), more.begin(), more.end());
            return evs;
        };
        auto fired_named = [](const std::vector<bsm::SpotEvent>& evs,
                              const std::string& name) {
            for (const auto& e : evs) if (e.name == name) return true;
            return false;
        };

        // A permissive policy: the model is small and the gate here is "the
        // end-to-end path detects a Kokoro-spoken keyword", not exact calibration.
        bsm::SpotterConfig pol;
        pol.threshold    = 0.30f;
        pol.min_phonemes = 3;
        pol.smoothing_hits = 2;
        pol.smoothing_window = 5;

        const std::string KW = "computer";

        // ─ A. enroll from phonemizer ids, fire on the keyword ─
        const int L = spotter.enroll(KW, phon.phonemize(KW), &pol);
        std::fprintf(stderr, "  enrolled '%s' -> template length %d\n",
                     KW.c_str(), L);
        CHECK(L >= 3, "enroll: keyword reduced to a usable template (>=3)");

        auto kw_pcm = synth16k(KW, 1.0f);
        CHECK(!kw_pcm.empty(), "synth: keyword rendered to PCM");
        auto evs_kw = feed_word(kw_pcm);
        float kw_conf = 0.0f;
        for (const auto& e : evs_kw) if (e.name == KW) kw_conf = e.confidence;
        std::fprintf(stderr, "  keyword fires: %d event(s), conf=%.3f\n",
                     static_cast<int>(evs_kw.size()), kw_conf);
        CHECK(fired_named(evs_kw, KW), "POSITIVE: spotter fires on the keyword");

        // ─ B. a distractor word must NOT fire the keyword template ─
        auto dis_pcm = synth16k("banana", 1.0f);
        auto evs_dis = feed_word(dis_pcm);
        CHECK(!fired_named(evs_dis, KW),
              "NEGATIVE: distractor 'banana' does not fire 'computer'");

        // ─ C. enroll_from_audio (reference render) then fire at a new speed ─
        spotter.clear();
        auto ref_pcm = synth16k(KW, 1.0f);
        const int L2 = spotter.enroll_from_audio(KW, ref_pcm.data(),
                                                 static_cast<int>(ref_pcm.size()),
                                                 &pol);
        std::fprintf(stderr, "  enroll_from_audio '%s' -> template length %d\n",
                     KW.c_str(), L2);
        CHECK(L2 >= 3, "enroll_from_audio: argmax template is usable (>=3)");
        auto kw_pcm2 = synth16k(KW, 0.9f);   // different speed than the reference
        auto evs_kw2 = feed_word(kw_pcm2);
        CHECK(fired_named(evs_kw2, KW),
              "POSITIVE: audio-enrolled template fires on a re-rendered keyword");

        // ─ D. enroll_from_audio is trim-invariant ─
        // A reference HARD-TRIMMED to the sound must enroll the same template
        // as one arriving out of ambient quiet. PCEN's smoother seeds on the
        // first frame, so without the internal lead pad a trimmed reference
        // produces a different feature stream — measured on real recordings:
        // churn templates that missed their own sound in context and
        // false-fired on speech.
        {
            std::size_t a = 0, b = ref_pcm.size();
            while (a < b && std::abs(ref_pcm[a]) < 1e-3f) ++a;
            while (b > a && std::abs(ref_pcm[b - 1]) < 1e-3f) --b;
            const std::vector<float> trimmed(
                ref_pcm.begin() + static_cast<std::ptrdiff_t>(a),
                ref_pcm.begin() + static_cast<std::ptrdiff_t>(b));
            std::vector<float> ambient(trimmed.size() + 12800, 0.0f);
            std::copy(trimmed.begin(), trimmed.end(), ambient.begin() + 6400);
            const int Lt = spotter.enroll_from_audio(
                "trimmed", trimmed.data(), static_cast<int>(trimmed.size()), &pol);
            const int La = spotter.enroll_from_audio(
                "ambient", ambient.data(), static_cast<int>(ambient.size()), &pol);
            std::fprintf(stderr, "  trim-invariance: trimmed len %d, ambient len %d\n",
                         Lt, La);
            CHECK(Lt == La, "enroll_from_audio: trimmed and ambient references "
                            "enroll the same template length");
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        ++g_fail;
    }

    if (g_fail == 0) std::fprintf(stderr, "test_phoneme_spotter_e2e: all checks passed\n");
    else             std::fprintf(stderr, "test_phoneme_spotter_e2e: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
