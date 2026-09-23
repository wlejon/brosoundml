cmake_minimum_required(VERSION 3.24)

set(BROSOUNDML_API_SOURCES
    host_class.cpp
    api.cpp
    soundml_async.cpp
    native_soundml_stt.cpp
    native_soundml_stt_whisper.cpp
    native_soundml_stt_parakeet.cpp
    native_soundml_stt_qwen.cpp
    native_soundml_tts.cpp
    native_soundml_tts_kokoro.cpp
    native_soundml_tts_qwen.cpp
    native_soundml_tts_omnivoice.cpp
    native_soundml_tts_supertonic.cpp
    native_soundml_tts_higgs.cpp
    native_soundml_diar.cpp
    native_soundml_rave.cpp
    soundml_listen_host.cpp
    native_soundml_wake.cpp
    native_soundml_kws.cpp
    native_soundml_sense.cpp
    native_soundml_gesture.cpp
    native_soundml_listen.cpp
    native_soundml_voice_agent.cpp
)

add_library(brosoundml_api STATIC
    ${BROSOUNDML_API_SOURCES}
)

target_include_directories(brosoundml_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(brosoundml_api PUBLIC
    brosoundml
    brotensor
    broaudio
    bronze_runtime_shared
)

target_compile_features(brosoundml_api PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(brosoundml_api PRIVATE /MP /permissive- /utf-8 /W4)
else()
    target_compile_options(brosoundml_api PRIVATE -Wall -Wextra)
endif()
