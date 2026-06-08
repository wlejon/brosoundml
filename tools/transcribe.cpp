// brosoundml::transcribe — command-line Whisper driver.
//
// Reads a 16 kHz mono WAV, runs Whisper::transcribe end-to-end against a
// model directory + the matching brolm::whisper::Tokenizer, and prints the
// decoded transcript (single line) to stdout. Errors go to stderr.
//
// Usage:
//   brosoundml_transcribe <wav> <model_dir>
//                         [--lang en] [--task transcribe]
//                         [--no-timestamps] [--max-new-tokens N] [--stream]
//
// Notes:
//   * The WAV must be 16 kHz mono PCM — Whisper's input rate is fixed and
//     resampling lives outside this CLI (use ffmpeg or sox to convert).
//   * `--lang` is an ISO-639-1 code (en, zh, fr, ...); `--task` is
//     `transcribe` or `translate`.
//   * `--max-new-tokens 0` (the default) lets Whisper run until either EOS
//     or max_target_positions - prompt_len (per 30 s window in long-form).
//   * `--stream` prints the transcript incrementally as each token decodes
//     instead of waiting for the whole clip.
//   * Clips longer than 30 s are decoded with Whisper's sequential long-form
//     windowing (30 s segments + timestamp seek) instead of being truncated;
//     this needs timestamps, so it is disabled under `--no-timestamps`.

#include "brosoundml/audio.h"
#include "brosoundml/whisper.h"

#include <brolm/whisper_tokenizer.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "brosoundml_transcribe: %s\n", msg.c_str());
    std::exit(2);
}

void print_usage() {
    std::printf(
        "Usage:\n"
        "  brosoundml_transcribe <wav> <model_dir>\n"
        "                        [--lang en] [--task transcribe]\n"
        "                        [--no-timestamps] [--max-new-tokens N]\n"
        "\n"
        "  <wav>        Path to a 16 kHz mono PCM WAV file (Whisper's fixed\n"
        "               input rate). Resample externally (ffmpeg / sox) if\n"
        "               your source clip is at a different rate.\n"
        "  <model_dir>  Directory containing config.json, model.safetensors,\n"
        "               vocab.json and merges.txt.\n"
        "\n"
        "Options:\n"
        "  --lang LC           ISO-639-1 language code (default: en).\n"
        "  --task T            'transcribe' (default) or 'translate'.\n"
        "  --no-timestamps     Suppress timestamp tokens (disables long-form).\n"
        "  --max-new-tokens N  Cap generated tokens (0 = model default).\n"
        "  --stream            Print the transcript incrementally as it decodes.\n"
        "  -h, --help          Show this help and exit.\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string wav_path, model_dir;
    std::string lang = "en";
    std::string task = "transcribe";
    bool        with_timestamps = true;
    int         max_new_tokens  = 0;
    bool        stream          = false;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string(what) + " needs a value");
            return argv[++i];
        };
        if      (a == "-h" || a == "--help") { print_usage(); return 0; }
        else if (a == "--lang")             lang = next("--lang");
        else if (a == "--task")             task = next("--task");
        else if (a == "--no-timestamps")    with_timestamps = false;
        else if (a == "--stream")           stream = true;
        else if (a == "--max-new-tokens")   max_new_tokens = std::atoi(next("--max-new-tokens").c_str());
        else if (!a.empty() && a[0] == '-') die("unknown flag '" + a + "'");
        else                                positional.push_back(std::move(a));
    }

    if (positional.size() != 2) {
        print_usage();
        die("expected exactly two positional args: <wav> <model_dir>");
    }
    wav_path  = positional[0];
    model_dir = positional[1];

    try {
        // 1. Load model.
        brosoundml::Whisper model;
        model.load(model_dir);

        // 2. Load tokenizer (vocab.json + merges.txt under the model dir).
        namespace fs = std::filesystem;
        const fs::path md = model_dir;
        const std::string vocab_path  = (md / "vocab.json").string();
        const std::string merges_path = (md / "merges.txt").string();
        auto tok = brolm::whisper::Tokenizer::load(vocab_path, merges_path);

        // 3. Read WAV.
        brosoundml::AudioBuffer audio = brosoundml::read_wav(wav_path);
        if (audio.sample_rate != 16000) {
            die("WAV is " + std::to_string(audio.sample_rate) +
                " Hz; Whisper requires 16 kHz mono PCM. "
                "Resample externally (ffmpeg -ar 16000 -ac 1 ...) and retry.");
        }

        // 4. Build prompt + options.
        std::vector<int32_t> prompt =
            tok.build_prompt(lang, task, with_timestamps);
        std::fprintf(stderr,
                     "brosoundml_transcribe: %.2fs audio, prompt=%zu tokens\n",
                     audio.duration_seconds(), prompt.size());

        brosoundml::Whisper::TranscribeOptions opts;
        opts.max_new_tokens = max_new_tokens;
        // Long-form: window clips past 30 s instead of truncating them. Seek
        // reads timestamp tokens, so only enable it when the prompt carries
        // timestamps. Harmless for <=30 s clips (the windowing gate also checks
        // duration).
        if (with_timestamps) opts.timestamp_begin_id = tok.first_timestamp_id();

        // --stream: re-decode the running id list each token and print only the
        // newly revealed suffix, so the transcript grows live on stdout. (BPE
        // decode of a growing prefix is monotonic for the committed text.)
        std::vector<int32_t> generated;
        std::size_t printed = 0;
        if (stream) {
            opts.on_token = [&](int32_t id) {
                generated.push_back(id);
                std::string text = tok.decode(generated, /*skip_special=*/true);
                if (text.size() > printed) {
                    std::fwrite(text.data() + printed, 1, text.size() - printed,
                                stdout);
                    std::fflush(stdout);
                    printed = text.size();
                }
            };
        }

        // 5. Run. In --stream mode the transcript has already been printed by
        // the callback; just terminate the line. Otherwise decode + print the
        // full transcript at the end (prompt prefix dropped, specials stripped).
        auto result = model.transcribe(audio, prompt, opts);

        if (stream) {
            std::printf("\n");
        } else {
            std::vector<int32_t> out(result.token_ids.begin() + prompt.size(),
                                     result.token_ids.end());
            std::string text = tok.decode(out, /*skip_special=*/true);
            std::printf("%s\n", text.c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        die(std::string("error: ") + e.what());
    }
}
