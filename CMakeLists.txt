cmake_minimum_required(VERSION 3.24)

# Pre-1.0: consumers vendor this repo via add_subdirectory and build it from
# source, so the version is a pin point for a tag, not an ABI promise.
project(brosoundml VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Coverage. Scoped to a top-level build: a consumer that add_subdirectory()s
# brosoundml has no business having its own build instrumented from in here.
option(BROSOUNDML_COVERAGE "Instrument the build for gcov coverage (GCC/Clang)" OFF)
if(BROSOUNDML_COVERAGE AND PROJECT_IS_TOP_LEVEL)
    if(MSVC)
        message(FATAL_ERROR
            "BROSOUNDML_COVERAGE requires GCC or Clang (gcov instrumentation); MSVC is unsupported.")
    endif()
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
    message(STATUS "brosoundml: coverage instrumentation ON (-O0)")
endif()

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
# Vulkan compute backend (brotensor's docs/vulkan.md), the AMD GPU path.
# brosoundml needs no GLSL of its own, so this only forwards it.
option(BROTENSOR_WITH_VULKAN "Build the Vulkan GPU backend" OFF)

# --- Dependencies: bromath, bronze (+ brass), brotensor, broimage, brolm, broaudio ---
#
# Each resolves through cmake/bro_deps.cmake, like every repo in the bro
# ecosystem: an existing target (a superbuild such as bro already added it, and
# its pins win) first; otherwise the ../<name> working tree beside the top-level
# project; otherwise the commit pinned here, fetched at configure. Leaves first,
# so each finds the ones it needs already there (broimage is brolm's).
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/bro_deps.cmake)

bro_dependency(bromath GITHUB wlejon/bromath REF 8b511fbd69c014bf0584d2b42e6c96a8cf720f6a TARGET bromath::bromath)

# bronze, the JavaScript runtime the api binds (it brings brass, its code
# generator). The api is bound across bronze's C++ embed boundary, so it has to
# build against the bronze the surrounding tree builds against.
bro_dependency(bronze GITHUB wlejon/bronze REF 15ce4cf5fc25d8afe03882985ae95adcba46a6a9 TARGET bronze_runtime_shared
               OPTIONS BRONZE_BUILD_SHARED_RUNTIME=ON BRONZE_BUILD_TESTS=OFF)

bro_dependency(brotensor GITHUB wlejon/brotensor REF a332d9283c1e23b9f39e8ecaa4393f937dd26c0a)
bro_dependency(broimage GITHUB wlejon/broimage REF 9346cd5e97cb2fba0bae15e9f15fc0d9e37349ee)
# brolm — sibling text/model library. brosoundml consumes brolm for tokenizers
# (brolm::whisper::Tokenizer and friends); brolm's own tests + tools are
# skipped because it is added with EXCLUDE_FROM_ALL.
bro_dependency(brolm GITHUB wlejon/brolm REF 3e2547cba802e183196b95523bc53ee99304664b)

# broaudio probes for an SDL3 and would take an installed one built against the
# DLL CRT (/MD). This standalone tree is static-CRT (/MT, above), and a /MD
# static library does not link into a /MT executable (unresolved __imp_* CRT
# imports the moment the api's listen host pulls in broaudio::Engine). So
# standalone on MSVC, build SDL3 from the pinned source under this tree's own
# runtime setting; inside bro SDL3::SDL3-static already exists and this is
# skipped.
if(MSVC AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND NOT TARGET SDL3::SDL3 AND NOT TARGET SDL3::SDL3-static)
    bro_dependency(SDL GITHUB libsdl-org/SDL REF 1df279a04f604bc50b6f36c9e903b5e64c4fe2a3 THIRD_PARTY
                   TARGET SDL3-static OPTIONS SDL_SHARED=OFF SDL_STATIC=ON SDL_TEST_LIBRARY=OFF)
endif()

# broaudio — sibling audio engine library
bro_dependency(broaudio GITHUB wlejon/broaudio REF 964a428ce8d8eae432cba8ac1a437ca5e2bf1b3f)

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

# The graph-captured decode / step paths (LSTM, Whisper, Qwen3-TTS, OmniVoice,
# Supertonic) need no compile-time flag: brotensor's CudaGraphCapture is
# device-neutral, and brotensor::graph_capture_available(device) gates them at
# run time (CUDA, Vulkan).

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
    src/kokoro_predictor.cpp
    src/kokoro_generator.cpp
    src/kokoro.cpp
    src/decoder_lora.cpp
    src/kokoro_decoder_backward.cpp
    src/qwen_tts.cpp
    src/qwen_tts_codec.cpp
    src/qwen_tts_codec_encoder.cpp
    src/qwen_tts_speaker_encoder.cpp
    src/qwen_tts_talker.cpp
    src/qwen_tts_code_predictor.cpp
    src/qwen_tts_generate.cpp
    src/higgs_codec.cpp
    src/hubert.cpp
    src/omnivoice.cpp
    src/omnivoice_lm.cpp
    src/omnivoice_prompt.cpp
    src/qwen_asr.cpp
    src/qwen_asr_encoder.cpp
    src/qwen_asr_decoder.cpp
    src/rave.cpp
    src/supertonic.cpp
    src/speaker_encoder.cpp
    src/wake.cpp
    src/wake_data.cpp
    src/phoneme_data.cpp
    src/bc_resnet2d.cpp
    src/voice_agent.cpp
    src/phoneme_model.cpp
    src/phoneme_spotter.cpp
    src/gesture_spotter.cpp
    src/listen_bus.cpp
    src/sensor_hub.cpp
    src/mel.cpp
    src/whisper.cpp
    src/whisper_modules.cpp
    src/fastconformer.cpp
    src/parakeet.cpp
    src/word_align.cpp
    src/sortformer.cpp
    src/cluster_diarizer.cpp
    src/clap.cpp
    src/clap_audio.cpp
    src/clap_text.cpp
    src/g2p/pos_tagger.cpp
    src/g2p/pos_tagger_load.cpp
    src/g2p/lexicon.cpp
    src/g2p/morphology.cpp
    src/g2p/normalizer.cpp
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

    add_executable(brosoundml_qwen_asr_transcribe tools/qwen_asr_transcribe.cpp)
    target_link_libraries(brosoundml_qwen_asr_transcribe PRIVATE brosoundml)

    add_executable(brosoundml_parakeet_transcribe tools/parakeet_transcribe.cpp)
    target_link_libraries(brosoundml_parakeet_transcribe PRIVATE brosoundml)

    add_executable(brosoundml_wake_synth tools/wake_synth.cpp)
    target_link_libraries(brosoundml_wake_synth PRIVATE brosoundml)

    add_executable(brosoundml_wake_inspect tools/wake_inspect.cpp)
    target_link_libraries(brosoundml_wake_inspect PRIVATE brosoundml)

    add_executable(brosoundml_wake_train tools/wake_train.cpp)
    target_link_libraries(brosoundml_wake_train PRIVATE brosoundml)

    add_executable(brosoundml_wake_test tools/wake_test.cpp)
    target_link_libraries(brosoundml_wake_test PRIVATE brosoundml)

    add_executable(brosoundml_wake_probe tools/wake_probe.cpp)
    target_link_libraries(brosoundml_wake_probe PRIVATE brosoundml)

    add_executable(brosoundml_wake_melcmp tools/wake_melcmp.cpp)
    target_link_libraries(brosoundml_wake_melcmp PRIVATE brosoundml)

    add_executable(brosoundml_phoneme_synth tools/phoneme_synth.cpp)
    target_link_libraries(brosoundml_phoneme_synth PRIVATE brosoundml)

    add_executable(brosoundml_phoneme_inspect tools/phoneme_inspect.cpp)
    target_link_libraries(brosoundml_phoneme_inspect PRIVATE brosoundml)

    add_executable(brosoundml_phoneme_train tools/phoneme_train.cpp)
    target_link_libraries(brosoundml_phoneme_train PRIVATE brosoundml)

    add_executable(brosoundml_phoneme_test tools/phoneme_test.cpp)
    target_link_libraries(brosoundml_phoneme_test PRIVATE brosoundml)

    add_executable(brosoundml_phoneme_calibrate tools/phoneme_calibrate.cpp)
    target_link_libraries(brosoundml_phoneme_calibrate PRIVATE brosoundml)

    add_executable(brosoundml_phoneme_align tools/phoneme_align.cpp)
    target_link_libraries(brosoundml_phoneme_align PRIVATE brosoundml)

    add_executable(brosoundml_phoneme_melcache tools/phoneme_melcache.cpp)
    target_link_libraries(brosoundml_phoneme_melcache PRIVATE brosoundml)

    add_executable(brosoundml_phoneme_aug tools/phoneme_aug.cpp)
    target_link_libraries(brosoundml_phoneme_aug PRIVATE brosoundml)

    add_executable(brosoundml_sound_units tools/sound_units.cpp)
    target_link_libraries(brosoundml_sound_units PRIVATE brosoundml)

    add_executable(brosoundml_qwen_tts_bench tools/qwen_tts_bench.cpp)
    target_link_libraries(brosoundml_qwen_tts_bench PRIVATE brosoundml)

    add_executable(brosoundml_kokoro_bench tools/kokoro_bench.cpp)
    target_link_libraries(brosoundml_kokoro_bench PRIVATE brosoundml)

    add_executable(brosoundml_qwen_tts_roundtrip tools/qwen_tts_roundtrip.cpp)
    target_link_libraries(brosoundml_qwen_tts_roundtrip PRIVATE brosoundml)

    add_executable(brosoundml_qwen_tts_clone tools/qwen_tts_clone.cpp)
    target_link_libraries(brosoundml_qwen_tts_clone PRIVATE brosoundml)

    add_executable(brosoundml_qwen_tts_say tools/qwen_tts_say.cpp)
    target_link_libraries(brosoundml_qwen_tts_say PRIVATE brosoundml)

    add_executable(brosoundml_higgs_codec_roundtrip tools/higgs_codec_roundtrip.cpp)
    target_link_libraries(brosoundml_higgs_codec_roundtrip PRIVATE brosoundml)

    add_executable(brosoundml_omnivoice_say tools/omnivoice_say.cpp)
    target_link_libraries(brosoundml_omnivoice_say PRIVATE brosoundml)

    add_executable(brosoundml_build_speaker_encoder tools/build_speaker_encoder.cpp)
    target_link_libraries(brosoundml_build_speaker_encoder PRIVATE brosoundml)

    add_executable(brosoundml_supertonic_synth tools/supertonic_synth.cpp)
    target_link_libraries(brosoundml_supertonic_synth PRIVATE brosoundml)

    # masc<->fem voice axis from the model's own labeled F*/M* preset means (the
    # realizable within-group contrast); writes masc_fem_basis.json beside the model.
    add_executable(brosoundml_supertonic_preset_gender tools/supertonic_preset_gender.cpp)
    target_link_libraries(brosoundml_supertonic_preset_gender PRIVATE brosoundml)

    add_executable(brosoundml_sortformer_diarize tools/sortformer_diarize.cpp)
    target_link_libraries(brosoundml_sortformer_diarize PRIVATE brosoundml)

    add_executable(brosoundml_speaker_embed_probe tools/speaker_embed_probe.cpp)
    target_link_libraries(brosoundml_speaker_embed_probe PRIVATE brosoundml)

    add_executable(brosoundml_cluster_diarize tools/cluster_diarize.cpp)
    target_link_libraries(brosoundml_cluster_diarize PRIVATE brosoundml)

    # laya-audio: the audio front end for Laya (docs/laya-audio.md).
    add_library(brosoundml_laya_audio_data STATIC tools/laya_audio_data.cpp tools/laya_audio_decode.cpp
                tools/laya_audio_augment.cpp tools/laya_audio_task.cpp tools/laya_audio_corpus.cpp
                tools/laya_audio_metrics.cpp tools/laya_audio_baselines.cpp tools/laya_audio_listener.cpp
                tools/laya_audio_questions.cpp tools/laya_audio_distill.cpp)
    target_link_libraries(brosoundml_laya_audio_data PUBLIC brosoundml)
    # Compressed dataset audio (tools only): Ogg Opus via libopus + libogg,
    # FLAC via the header-only dr_flac. Found in a vcpkg tree or the system.
    set(_la_vcpkg "$ENV{VCPKG_ROOT}")
    if(NOT _la_vcpkg AND EXISTS "D:/vcpkg")
        set(_la_vcpkg "D:/vcpkg")
    endif()
    set(_la_hint "")
    if(_la_vcpkg)
        set(_la_hint "${_la_vcpkg}/installed/x64-windows")
    endif()
    find_package(Opus CONFIG QUIET PATHS "${_la_hint}")
    find_package(Ogg CONFIG QUIET PATHS "${_la_hint}")
    if(TARGET Opus::opus AND TARGET Ogg::ogg)
        target_link_libraries(brosoundml_laya_audio_data PRIVATE Opus::opus Ogg::ogg)
        target_compile_definitions(brosoundml_laya_audio_data PRIVATE LAYA_AUDIO_HAVE_OPUS=1)
    else()
        message(STATUS "laya-audio tools: libopus/libogg not found, .opus input disabled")
    endif()
    find_path(LAYA_AUDIO_DR_FLAC_DIR dr_flac.h HINTS "${_la_hint}/include")
    if(LAYA_AUDIO_DR_FLAC_DIR)
        target_include_directories(brosoundml_laya_audio_data PRIVATE "${LAYA_AUDIO_DR_FLAC_DIR}")
        target_compile_definitions(brosoundml_laya_audio_data PRIVATE LAYA_AUDIO_HAVE_FLAC=1)
    else()
        message(STATUS "laya-audio tools: dr_flac.h not found, .flac input disabled")
    endif()
    foreach(_la align prep cache train eval listen labels ask stream)
        add_executable(brosoundml_laya_audio_${_la} tools/laya_audio_${_la}.cpp)
        target_link_libraries(brosoundml_laya_audio_${_la} PRIVATE brosoundml_laya_audio_data)
    endforeach()
endif()

# --- Standalone Bronze JavaScript API ---
option(BROSOUNDML_ENABLE_API "Build Bronze JavaScript API" ON)
if(BROSOUNDML_ENABLE_API)
    add_subdirectory(src/api)
endif()

# --- Tests ---
option(BROSOUNDML_TESTS "Build brosoundml tests" ON)
if(BROSOUNDML_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(tests)
endif()
