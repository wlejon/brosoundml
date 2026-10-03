#pragma once
//
// Test-only helper for GPU blocks. The model tests run a module on the CPU,
// then re-run it on a GPU backend and assert the two agree (or smoke the GPU
// path on its own). Historically each test hard-coded Device::CUDA, so a HIP
// or Metal build silently ran CPU-only; this helper makes the same block
// exercise whichever GPU backend the binary was built with — Vulkan or HIP on
// an AMD build (the default device: Vulkan unless BROTENSOR_PREFER_HIP=1 or
// BROTENSOR_DEFAULT_DEVICE=hip), CUDA on a CUDA build, Metal on a Metal build.
//
// Call brotensor::init() first (it performs the driver probes), then
// preferred_gpu(): it returns the backend to test against, or Device::CPU when
// no GPU backend is registered (meaning "skip the GPU block").
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

namespace brosoundml_test {

inline brotensor::Device preferred_gpu() {
    if (brotensor::default_device().is_gpu()) return brotensor::default_device();
    for (brotensor::Device d : {brotensor::Device::CUDA, brotensor::Device::HIP,
                                brotensor::Device::Metal, brotensor::Device::VULKAN}) {
        if (brotensor::is_available(d)) return d;
    }
    return brotensor::Device::CPU;
}

inline const char* device_name(brotensor::Device d) {
    switch (d.type) {
        case brotensor::DeviceType::HIP:   return "HIP";
        case brotensor::DeviceType::CUDA:  return "CUDA";
        case brotensor::DeviceType::Metal: return "Metal";
        case brotensor::DeviceType::VULKAN: return "Vulkan";
        default:                           return "CPU";
    }
}

} // namespace brosoundml_test
