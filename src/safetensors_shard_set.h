#pragma once

#include <brotensor/safetensors.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brosoundml {

struct SafeTensorsShardSet {
    std::vector<brotensor::safetensors::File> files;

    const brotensor::safetensors::TensorView* find(std::string_view name) const {
        for (const auto& f : files) {
            if (const auto* v = f.find(name)) return v;
        }
        return nullptr;
    }

    static SafeTensorsShardSet open(const std::string& model_dir) {
        namespace fs = std::filesystem;
        const fs::path dir(model_dir);
        const fs::path single = dir / "model.safetensors";
        SafeTensorsShardSet set;
        if (fs::exists(single)) {
            set.files.push_back(brotensor::safetensors::File::open(single.string()));
            return set;
        }
        std::vector<fs::path> paths;
        if (fs::is_directory(dir)) {
            for (const auto& entry : fs::directory_iterator(dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                    paths.push_back(entry.path());
                }
            }
        }
        if (paths.empty()) {
            throw std::runtime_error("SafeTensorsShardSet: no *.safetensors files in '" + model_dir + "'");
        }
        std::sort(paths.begin(), paths.end());
        for (const auto& p : paths) {
            set.files.push_back(brotensor::safetensors::File::open(p.string()));
        }
        return set;
    }
};

}  // namespace brosoundml
