#pragma once

// A UTF-8 std::string path as a filesystem path. A narrow std::string opened
// directly is read in the ANSI code page on Windows, so a song title like
// "云朵时光" became mojibake on disk; paths from JS are UTF-8.

#include <filesystem>
#include <string>

namespace brosoundml {

inline std::filesystem::path u8path(const std::string& s) {
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

}  // namespace brosoundml
