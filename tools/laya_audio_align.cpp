// brosoundml_laya_audio_align — word timings for a transcribed speech corpus.
//
// Runs Parakeet-TDT over every utterance of a LibriTTS-R manifest subset,
// turns its token stream (emission frame + TDT duration per token) into timed
// hypothesis words, and transfers those times onto the REFERENCE transcript's
// words by a Levenshtein word alignment (laya_audio::align_reference). The
// word identities therefore come from the human transcript; only the times
// come from the ASR model (80 ms encoder frames).
//
// Usage:
//   brosoundml_laya_audio_align --manifest M.json --subset dev-clean --out A.tsv
//                               [--model-dir weights/parakeet/0.6b-v3]
//                               [--limit N] [--every K] [--resume]
//
// --every K keeps every K-th utterance of the subset (a deterministic subsample).
// --resume skips utterance ids already present in --out and appends.

#include "laya_audio_data.h"

#include "brosoundml/audio.h"
#include "brosoundml/parakeet.h"

#include <brolm/tokenizer_t5.h>
#include <brotensor/runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "laya_audio_align: %s\n", msg.c_str());
    std::exit(2);
}

// Timed hypothesis words from Parakeet's token stream: a token whose
// incremental detokenisation starts with a space begins a new word.
std::vector<laya_audio::TimedWord> hyp_words(const brolm::t5::Tokenizer& tok,
                                             const brosoundml::Parakeet::Transcription& tr,
                                             double frame_s) {
    std::vector<laya_audio::TimedWord> words;
    std::vector<int32_t> prefix;
    std::string prev_text;
    std::string cur;
    float t0 = 0, t1 = 0;
    auto flush = [&] {
        for (const std::string& w : laya_audio::normalize_words(cur)) {
            laya_audio::TimedWord tw;
            tw.word = w;
            tw.t0 = t0;
            tw.t1 = t1;
            words.push_back(std::move(tw));
        }
        cur.clear();
    };
    for (std::size_t i = 0; i < tr.token_ids.size(); ++i) {
        prefix.push_back(tr.token_ids[i]);
        const std::string text = tok.decode(prefix);
        const std::string piece = text.size() > prev_text.size() ? text.substr(prev_text.size()) : std::string();
        prev_text = text;
        const float s = static_cast<float>(tr.token_frames[i] * frame_s);
        const int dur = i < tr.token_durations.size() ? tr.token_durations[i] : 1;
        const float e = static_cast<float>((tr.token_frames[i] + std::max(1, dur)) * frame_s);
        const bool starts_word = i == 0 || (!piece.empty() && piece[0] == ' ');
        if (starts_word) {
            flush();
            t0 = s;
        }
        cur += piece;
        t1 = e;
    }
    flush();
    return words;
}

}  // namespace

int main(int argc, char** argv) {
    std::string manifest, subset, out_path, model_dir = "weights/parakeet/0.6b-v3";
    int limit = 0, every = 1;
    bool resume = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--manifest") manifest = next();
        else if (a == "--subset") subset = next();
        else if (a == "--out") out_path = next();
        else if (a == "--model-dir") model_dir = next();
        else if (a == "--limit") limit = std::atoi(next().c_str());
        else if (a == "--every") every = std::max(1, std::atoi(next().c_str()));
        else if (a == "--resume") resume = true;
        else die("unknown argument " + a);
    }
    if (manifest.empty() || subset.empty() || out_path.empty()) die("need --manifest, --subset and --out");

    try {
        brotensor::init();
        const brotensor::Device dev =
            brotensor::is_available(brotensor::Device::CUDA) ? brotensor::Device::CUDA : brotensor::Device::CPU;

        std::vector<laya_audio::Utterance> utts;
        {
            int k = 0;
            for (laya_audio::Utterance& u : laya_audio::load_manifest(manifest)) {
                if (u.subset != subset) continue;
                if (k++ % every) continue;
                utts.push_back(std::move(u));
            }
        }
        if (limit > 0 && static_cast<int>(utts.size()) > limit) utts.resize(static_cast<std::size_t>(limit));

        std::unordered_set<std::string> done;
        if (resume && std::filesystem::exists(out_path)) {
            for (const auto& a : laya_audio::read_alignments(out_path)) done.insert(a.id);
        } else {
            laya_audio::write_alignments(out_path, {}, false);
        }

        brosoundml::Parakeet model;
        model.load(model_dir, dev);
        const auto tok = brolm::t5::Tokenizer::load((std::filesystem::path(model_dir) / "tokenizer.json").string());
        const double frame_s = model.config().frame_seconds();

        std::vector<laya_audio::AlignedUtterance> batch;
        std::size_t n_words = 0, n_match = 0, n_timed = 0, n_done = 0;
        const auto t_start = std::chrono::steady_clock::now();
        for (const laya_audio::Utterance& u : utts) {
            if (done.count(u.id)) continue;
            brosoundml::AudioBuffer audio(laya_audio::load_audio_16k(u.wav), 16000);
            if (audio.samples.size() < 1600) continue;
            const auto tr = model.transcribe(audio);
            laya_audio::AlignedUtterance a;
            a.id = u.id;
            a.wav = u.wav;
            a.speaker = u.speaker;
            a.subset = u.subset;
            a.duration_s = static_cast<float>(audio.duration_seconds());
            a.words = laya_audio::align_reference(laya_audio::normalize_words(u.text), hyp_words(tok, tr, frame_s));
            for (const auto& w : a.words) {
                ++n_words;
                n_match += w.asr_match ? 1 : 0;
                n_timed += w.t0 >= 0 ? 1 : 0;
            }
            batch.push_back(std::move(a));
            ++n_done;
            if (batch.size() >= 100) {
                laya_audio::write_alignments(out_path, batch, true);
                batch.clear();
                const double el =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                std::fprintf(stderr, "%zu/%zu utterances, %.1f s, word match %.3f, timed %.3f\n", n_done,
                             utts.size(), el, double(n_match) / std::max<std::size_t>(1, n_words),
                             double(n_timed) / std::max<std::size_t>(1, n_words));
            }
        }
        laya_audio::write_alignments(out_path, batch, true);
        std::fprintf(stderr, "done: %zu utterances, %zu words, ASR word match %.4f, timed %.4f\n", n_done,
                     n_words, double(n_match) / std::max<std::size_t>(1, n_words),
                     double(n_timed) / std::max<std::size_t>(1, n_words));
        return 0;
    } catch (const std::exception& e) {
        die(e.what());
    }
}
