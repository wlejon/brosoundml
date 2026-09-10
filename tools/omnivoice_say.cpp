// brosoundml_omnivoice_say — OmniVoice text-to-speech from the command line.
//
//   brosoundml_omnivoice_say <model_dir> "<text>" <out.wav>
//       [--device cpu|cuda] [--bf16]
//       [--lang X] [--instruct "..."]
//       [--ref ref.wav --ref-text "..."] [--prompt file.ovcp | --save-prompt file.ovcp]
//       [--steps N] [--guidance G] [--speed S] [--duration D] [--seed N] [--no-noise]
//       [--no-post] [--trace]
//
// Prints the load / LM / codec timings; --trace also prints the prompt ids and
// the unmask-order grid (one row per codebook, one character per frame, darker
// = unmasked later).

#include "brosoundml/audio.h"
#include "brosoundml/omnivoice.h"

#include <brotensor/runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: brosoundml_omnivoice_say <model_dir> \"<text>\" <out.wav>\n"
                 "         [--device cpu|cuda] [--bf16] [--lang X] [--instruct \"...\"]\n"
                 "         [--ref ref.wav --ref-text \"...\"] [--prompt file.ovcp | --save-prompt file.ovcp]\n"
                 "         [--steps N] [--guidance G] [--speed S] [--duration D] [--seed N]\n"
                 "         [--no-noise] [--no-post] [--trace]\n");
}

void print_heatmap(const brosoundml::OmniVoiceTrace& tr, int num_steps, int C) {
    static const char ramp[] = " .:-=+*#%@";
    const int T = tr.num_frames;
    if (T <= 0 || tr.unmask_step.size() != static_cast<std::size_t>(C) * T) return;
    std::printf("unmask order (row = codebook, column = frame; ' ' first ... '@' last, 'k' kept):\n");
    const int block = 100;
    for (int f0 = 0; f0 < T; f0 += block) {
        const int f1 = std::min(T, f0 + block);
        std::printf("  frames %d-%d\n", f0, f1 - 1);
        for (int q = 0; q < C; ++q) {
            std::printf("  q%d |", q);
            for (int t = f0; t < f1; ++t) {
                const int s = tr.unmask_step[static_cast<std::size_t>(q) * T + t];
                char ch;
                if (s < 0) ch = 'k';
                else {
                    int lvl = static_cast<int>(static_cast<long long>(s) * 10 / std::max(1, num_steps));
                    lvl = std::max(0, std::min(9, lvl));
                    ch = ramp[lvl];
                }
                std::putchar(ch);
            }
            std::printf("|\n");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) { usage(); return 2; }
    const std::string model_dir = argv[1];
    const std::string text = argv[2];
    const std::string out_path = argv[3];

    brotensor::Device dev = brotensor::Device::CUDA;
    bool device_given = false, bf16 = false, trace = false;
    std::string ref_wav, ref_text, prompt_path, save_prompt;
    brosoundml::OmniVoiceParams p;

    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--device") {
            const std::string d = need("--device");
            if (d == "cpu") dev = brotensor::Device::CPU;
            else if (d == "cuda") dev = brotensor::Device::CUDA;
            else { std::fprintf(stderr, "unknown device '%s'\n", d.c_str()); return 2; }
            device_given = true;
        } else if (a == "--bf16") bf16 = true;
        else if (a == "--lang") p.language = need("--lang");
        else if (a == "--instruct") p.instruct = need("--instruct");
        else if (a == "--ref") ref_wav = need("--ref");
        else if (a == "--ref-text") ref_text = need("--ref-text");
        else if (a == "--prompt") prompt_path = need("--prompt");
        else if (a == "--save-prompt") save_prompt = need("--save-prompt");
        else if (a == "--steps") p.num_steps = std::atoi(need("--steps"));
        else if (a == "--guidance") p.guidance_scale = static_cast<float>(std::atof(need("--guidance")));
        else if (a == "--speed") p.speed = static_cast<float>(std::atof(need("--speed")));
        else if (a == "--duration") p.duration = static_cast<float>(std::atof(need("--duration")));
        else if (a == "--seed") p.seed = static_cast<std::uint64_t>(std::strtoull(need("--seed"), nullptr, 10));
        else if (a == "--no-noise") p.gumbel_noise = false;
        else if (a == "--no-post") p.postprocess_output = false;
        else if (a == "--trace") trace = true;
        else { std::fprintf(stderr, "unknown option '%s'\n", a.c_str()); usage(); return 2; }
    }

    try {
        brotensor::init();
        // CUDA is the default; the CPU fallback is taken only when CUDA is
        // absent and no --device was given, and it is never silent.
        if (!device_given && !brotensor::is_available(brotensor::Device::CUDA)) {
            std::fprintf(stderr, "warning: CUDA is not available — running on CPU (slow); pass --device cpu to run there deliberately\n");
            dev = brotensor::Device::CPU;
        }
        if (!brotensor::is_available(dev)) { std::fprintf(stderr, "device not available\n"); return 1; }
        if (bf16 && dev == brotensor::Device::CPU) { std::fprintf(stderr, "--bf16 needs --device cuda\n"); return 2; }

        brosoundml::OmniVoice ov;
        const bool need_encoder = !ref_wav.empty();
        auto t0 = std::chrono::steady_clock::now();
        ov.load(model_dir, dev, bf16 ? brosoundml::OmniVoicePrecision::BF16 : brosoundml::OmniVoicePrecision::FP32,
                /*codec_decoder_only=*/!need_encoder);
        std::printf("loaded %s (%s, %s) in %.2fs\n", model_dir.c_str(), dev == brotensor::Device::CUDA ? "cuda" : "cpu",
                    bf16 ? "bf16" : "fp32", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

        brosoundml::OmniVoicePrompt prompt;
        bool have_prompt = false;
        if (!prompt_path.empty()) {
            prompt = brosoundml::OmniVoicePrompt::load(prompt_path);
            have_prompt = true;
            std::printf("prompt %s: %d frames, rms %.4f, text \"%s\"\n", prompt_path.c_str(), prompt.num_frames, prompt.rms, prompt.text.c_str());
        } else if (!ref_wav.empty()) {
            brosoundml::AudioBuffer ref = brosoundml::read_wav(ref_wav);
            t0 = std::chrono::steady_clock::now();
            prompt = ov.create_prompt(ref, ref_text, p.preprocess_prompt);
            have_prompt = true;
            std::printf("reference %s: %.2fs -> %d frames, rms %.4f, in %.2fs\n", ref_wav.c_str(),
                        ref.samples.size() / static_cast<double>(std::max(1, ref.sample_rate)), prompt.num_frames, prompt.rms,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            if (!save_prompt.empty()) {
                prompt.save(save_prompt);
                std::printf("saved prompt to %s\n", save_prompt.c_str());
            }
        } else if (!save_prompt.empty()) {
            std::fprintf(stderr, "--save-prompt needs --ref\n");
            return 2;
        }

        brosoundml::OmniVoiceTrace tr;
        t0 = std::chrono::steady_clock::now();
        brosoundml::AudioBuffer out = ov.synthesize(text, p, have_prompt ? &prompt : nullptr, {}, &tr);
        const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (out.empty()) { std::fprintf(stderr, "no audio produced\n"); return 1; }
        out.write_wav(out_path);
        const double audio_s = out.samples.size() / static_cast<double>(out.sample_rate);
        std::printf("frames %d (%zu chunk%s), %d steps: LM %.3fs (%.1f ms/step), codec %.3fs, total %.3fs -> %.2fs audio (%.2fx real time)\n",
                    tr.num_frames, tr.chunk_frames.size(), tr.chunk_frames.size() == 1 ? "" : "s", p.num_steps, tr.lm_seconds,
                    1000.0 * tr.lm_seconds / std::max(1, p.num_steps) / std::max<std::size_t>(1, tr.chunk_frames.size()),
                    tr.codec_seconds, total, audio_s, audio_s / std::max(1e-9, total));
        std::printf("wrote %s\n", out_path.c_str());

        if (trace) {
            std::printf("prompt text ids (%zu):", tr.text_ids.size());
            for (int32_t id : tr.text_ids) std::printf(" %d", id);
            std::printf("\nchunk frames:");
            for (int f : tr.chunk_frames) std::printf(" %d", f);
            std::printf("\n");
            print_heatmap(tr, p.num_steps, ov.config().lm.num_codebooks);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
