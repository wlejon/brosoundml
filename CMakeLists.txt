cmake_minimum_required(VERSION 3.24)
project(brosoundml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# --- GPU backend selection ---
#
# brosoundml composes brotensor ops — it ships no GPU kernels of its own, so it
# enables no CUDA/Metal language. It only forwards the backend choice so a
# standalone GPU build resolves brotensor's CUDA/Metal backend. Built inside
# bro, brotensor is already a target (added transitively) and the
# if(NOT TARGET ...) guards below reuse it, backend and all.
option(BROTENSOR_WITH_CUDA  "Build the CUDA GPU backend"  OFF)
option(BROTENSOR_WITH_METAL "Build the Metal GPU backend" OFF)

# --- Siblings: bromath (header-only) + brotensor (tensor + ops) ---
#
# Prefer standalone repos at ../<name>; fall back to third_party/ submodules.
# Matches the pattern documented in bro/docs/multi-repo-workflow.md.

set(BROMATH_DIR "${CMAKE_SOURCE_DIR}/../bromath" CACHE PATH
    "Path to standalone bromath repo (set to a missing path to force submodule fallback)")
if(NOT TARGET bromath)
    if(EXISTS "${BROMATH_DIR}/CMakeLists.txt")
        add_subdirectory("${BROMATH_DIR}" "${CMAKE_BINARY_DIR}/bromath" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/bromath/CMakeLists.txt")
        add_subdirectory(third_party/bromath EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "bromath not found at ${BROMATH_DIR} or third_party/bromath. "
            "Clone the bromath repo next to this repo, or override with -DBROMATH_DIR=...")
    endif()
endif()

set(BROTENSOR_DIR "${CMAKE_SOURCE_DIR}/../brotensor" CACHE PATH
    "Path to standalone brotensor repo (set to a missing path to force submodule fallback)")
if(NOT TARGET brotensor)
    if(EXISTS "${BROTENSOR_DIR}/CMakeLists.txt")
        add_subdirectory("${BROTENSOR_DIR}" "${CMAKE_BINARY_DIR}/brotensor" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/brotensor/CMakeLists.txt")
        add_subdirectory(third_party/brotensor EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "brotensor not found at ${BROTENSOR_DIR} or third_party/brotensor. "
            "Clone the brotensor repo next to this repo, or override with -DBROTENSOR_DIR=...")
    endif()
endif()

# brolm — sibling text/model library. brosoundml consumes brolm for tokenizers
# (the upcoming Whisper target uses brolm::whisper::Tokenizer); brolm's own
# tests + tools are skipped because we add it with EXCLUDE_FROM_ALL.
set(BROLM_DIR "${CMAKE_SOURCE_DIR}/../brolm" CACHE PATH
    "Path to standalone brolm repo (set to a missing path to force submodule fallback)")
if(NOT TARGET brolm)
    if(EXISTS "${BROLM_DIR}/CMakeLists.txt")
        add_subdirectory("${BROLM_DIR}" "${CMAKE_BINARY_DIR}/brolm" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/brolm/CMakeLists.txt")
        add_subdirectory(third_party/brolm EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "brolm not found at ${BROLM_DIR} or third_party/brolm. "
            "Clone the brolm repo next to this repo, or override with -DBROLM_DIR=...")
    endif()
endif()

# --- brosoundml library ---

add_library(brosoundml STATIC)
add_library(brosoundml::brosoundml ALIAS brosoundml)

target_include_directories(brosoundml PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(brosoundml PUBLIC cxx_std_20)
target_link_libraries(brosoundml PUBLIC
    bromath::bromath
    brotensor::brotensor
    brolm::brolm
)

if(MSVC)
    target_compile_options(brosoundml PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(brosoundml PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

target_sources(brosoundml PRIVATE
    src/version.cpp
    src/audio.cpp
    src/modules.cpp
    src/kokoro_modules.cpp
    src/kokoro.cpp
    src/wake.cpp
    src/wake_data.cpp
    src/bc_resnet.cpp
    src/mel.cpp
    src/whisper.cpp
    src/whisper_modules.cpp
    src/g2p/pos_tagger.cpp
    src/g2p/pos_tagger_load.cpp
    src/g2p/lexicon.cpp
    src/g2p/morphology.cpp
    src/g2p/special_cases.cpp
    src/g2p/phoneme_adapter.cpp
    src/g2p/phonemizer.cpp
)

# --- Install ---
#
# Off by default: brosoundml is consumed as a sibling repo via the same
# add_subdirectory pattern bro uses for its other libraries, not via
# find_package. When enabled, this installs only the static library and the
# public headers — no CMake package config, since the bromath/brotensor
# siblings it links are EXCLUDE_FROM_ALL add_subdirectory targets that are not
# themselves exported. Consume brosoundml with add_subdirectory().
option(BROSOUNDML_INSTALL "Install the brosoundml static library and headers" OFF)
if(BROSOUNDML_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS brosoundml
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# --- Tools ---
option(BROSOUNDML_TOOLS "Build brosoundml CLI tools" ON)
if(BROSOUNDML_TOOLS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(brosoundml_synth tools/synth.cpp)
    target_link_libraries(brosoundml_synth PRIVATE brosoundml)

    add_executable(brosoundml_transcribe tools/transcribe.cpp)
    target_link_libraries(brosoundml_transcribe PRIVATE brosoundml)

    add_executable(brosoundml_wake_synth tools/wake_synth.cpp)
    target_link_libraries(brosoundml_wake_synth PRIVATE brosoundml)

    add_executable(brosoundml_wake_inspect tools/wake_inspect.cpp)
    target_link_libraries(brosoundml_wake_inspect PRIVATE brosoundml)
endif()

# --- Tests ---
option(BROSOUNDML_TESTS "Build brosoundml tests" ON)
if(BROSOUNDML_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(tests)
endif()
