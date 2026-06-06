// Extract the Qwen3-TTS Base ECAPA-TDNN speaker encoder into a standalone
// ~18 MB artifact, so voice cloning can enrol a reference clip without loading
// the full ~2.5 GB Base checkpoint (Talker + codec + tokenizer) just for its
// 1 % speaker-encoder slice.
//
//   build_speaker_encoder [base_dir] [out_dir]
//
// base_dir  defaults to weights/qwen-tts/0.6B-Base (config.json + model.safetensors)
// out_dir   defaults to ../brosoundml-data/qwen-tts/speaker-encoder
//
// Reads the Base checkpoint with brotensor::safetensors, copies every
// `speaker_encoder.*` tensor verbatim (dtype + bytes untouched — BF16 stays
// BF16) into a new container via the library writer, and writes a small
// config.json carrying the `speaker_encoder_config` block the loader reads.
// The library loads the result with brosoundml::SpeakerEncoder::load(dir).

#define _CRT_SECURE_NO_WARNINGS

#include "brosoundml/detail/json.h"

#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace sf = brotensor::safetensors;
namespace j  = brosoundml::detail::json;

static const std::string kPrefix = "speaker_encoder.";

int main(int argc, char** argv) {
    const std::string base_dir = (argc > 1)
        ? argv[1] : std::string("weights/qwen-tts/0.6B-Base");
    const std::string out_dir = (argc > 2)
        ? argv[2] : std::string("../brosoundml-data/qwen-tts/speaker-encoder");

    const fs::path base_weights = fs::path(base_dir) / "model.safetensors";
    const fs::path base_config  = fs::path(base_dir) / "config.json";
    if (!fs::exists(base_weights)) {
        std::fprintf(stderr,
            "error: no model.safetensors under '%s' — point base_dir at the "
            "0.6B-Base checkpoint\n", base_dir.c_str());
        return 2;
    }
    if (!fs::exists(base_config)) {
        std::fprintf(stderr, "error: no config.json under '%s'\n", base_dir.c_str());
        return 2;
    }

    brotensor::init();

    // ── Copy the speaker_encoder.* tensors verbatim ──
    // The File mmap stays alive for the whole scope, so the WriteEntry host_data
    // pointers (into the mapping) remain valid through write_file.
    sf::File in = sf::File::open(base_weights.string());
    std::vector<sf::WriteEntry> entries;
    for (const sf::TensorView& t : in.tensors()) {
        if (t.name.rfind(kPrefix, 0) != 0) continue;  // not a speaker_encoder.* tensor
        sf::WriteEntry e;
        e.name      = t.name;
        e.dtype     = t.dtype;
        e.shape     = t.shape;
        e.host_data = t.data;
        e.bytes     = t.nbytes;
        entries.push_back(std::move(e));
    }
    if (entries.empty()) {
        std::fprintf(stderr,
            "error: no '%s*' tensors in %s — is this the Base variant?\n",
            kPrefix.c_str(), base_weights.string().c_str());
        return 1;
    }

    fs::create_directories(out_dir);
    const fs::path out_weights = fs::path(out_dir) / "model.safetensors";
    sf::write_file(out_weights.string(), entries);

    // ── Carry the speaker_encoder_config block ──
    // config.json holds only enc_dim + sample_rate; the channel/kernel/dilation
    // lists and mel params are upstream defaults the loader fills. Mirror the
    // Base block exactly so the standalone resolves the same config.
    std::ifstream cf(base_config, std::ios::binary);
    std::ostringstream cs; cs << cf.rdbuf();
    const j::Value root = j::parse(cs.str());
    const j::Value* se  = root.is_object() ? root.find("speaker_encoder_config") : nullptr;
    if (!se || !se->is_object()) {
        std::fprintf(stderr,
            "error: Base config.json has no 'speaker_encoder_config' block\n");
        return 1;
    }
    const int enc_dim     = se->get_int("enc_dim", 1024);
    const int sample_rate = se->get_int("sample_rate", 24000);
    const int mel_dim     = se->get_int("mel_dim", 128);

    const fs::path out_config = fs::path(out_dir) / "config.json";
    std::ofstream oc(out_config, std::ios::binary);
    oc << "{\n"
       << "  \"speaker_encoder_config\": {\n"
       << "    \"mel_dim\": " << mel_dim << ",\n"
       << "    \"enc_dim\": " << enc_dim << ",\n"
       << "    \"sample_rate\": " << sample_rate << "\n"
       << "  }\n"
       << "}\n";
    oc.close();

    std::uintmax_t out_bytes = fs::file_size(out_weights);
    std::printf("wrote %zu tensors (%.2f MB) -> %s\n",
                entries.size(), out_bytes / 1e6, out_weights.string().c_str());
    std::printf("wrote config -> %s\n", out_config.string().c_str());
    return 0;
}
