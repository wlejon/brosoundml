#pragma once
//
// Device choice shared by the tools/ CLI drivers, so every tool spells
// `--device` the same way and lands on whichever GPU backend the binary was
// built with (HIP on a ROCm build, CUDA on a CUDA build, Metal on a Metal
// build) instead of hard-coding Device::CUDA. Call brotensor::init() first —
// availability is only known after the driver probes.
//
//   auto  default_device(): the best GPU, else CPU (BROTENSOR_DEFAULT_DEVICE
//         overrides, as everywhere in brotensor)
//   gpu   the best GPU (HIP > CUDA > Metal); an error when there is none
//   cuda  CUDA when present, else the best GPU — `--device cuda` has always
//         meant "use the GPU", and keeps meaning it on a HIP or Metal build
//   hip   HIP (alias: rocm); an error when it is not available
//   metal Metal; an error when it is not available
//   cpu   CPU
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <string>

namespace brosoundml_tool {

// The spellings resolve_device() accepts, for usage strings.
inline constexpr const char* kDeviceChoices = "auto|cpu|gpu|cuda|hip|rocm|metal";

// The best registered GPU backend (HIP > CUDA > Metal), or Device::CPU when
// the build or the machine has none.
inline brotensor::Device best_gpu() {
    if (brotensor::is_available(brotensor::Device::HIP))   return brotensor::Device::HIP;
    if (brotensor::is_available(brotensor::Device::CUDA))  return brotensor::Device::CUDA;
    if (brotensor::is_available(brotensor::Device::Metal)) return brotensor::Device::Metal;
    return brotensor::Device::CPU;
}

inline const char* device_name(brotensor::Device d) {
    switch (d.type) {
        case brotensor::DeviceType::HIP:   return "HIP";
        case brotensor::DeviceType::CUDA:  return "CUDA";
        case brotensor::DeviceType::Metal: return "Metal";
        default:                           return "CPU";
    }
}

// Resolve a --device value (see the table above). Returns false and fills
// `err` on an unknown spelling or a backend this binary / machine lacks.
inline bool resolve_device(const std::string& arg, brotensor::Device& out, std::string& err) {
    using brotensor::Device;
    auto require = [&](Device d, const char* what) {
        if (brotensor::is_available(d)) { out = d; return true; }
        err = std::string("--device ") + arg + ": no " + what + " backend is available";
        return false;
    };
    if (arg.empty() || arg == "auto") { out = brotensor::default_device(); return true; }
    if (arg == "cpu")                 { out = Device::CPU; return true; }
    if (arg == "hip" || arg == "rocm") return require(Device::HIP, "HIP");
    if (arg == "metal")                return require(Device::Metal, "Metal");
    if (arg == "gpu" || arg == "cuda") {
        if (arg == "cuda" && brotensor::is_available(Device::CUDA)) { out = Device::CUDA; return true; }
        out = best_gpu();
        if (out.is_gpu()) return true;
        err = std::string("--device ") + arg + ": no GPU backend is available";
        return false;
    }
    err = "unknown --device '" + arg + "' (want " + kDeviceChoices + ")";
    return false;
}

} // namespace brosoundml_tool
