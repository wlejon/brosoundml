function(brosoundml_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE brosoundml)
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

brosoundml_add_test(brosoundml_test_smoke    test_smoke.cpp)
brosoundml_add_test(brosoundml_test_audio    test_audio.cpp)
brosoundml_add_test(brosoundml_test_modules  test_modules.cpp)
brosoundml_add_test(brosoundml_test_kokoro   test_kokoro.cpp)
brosoundml_add_test(brosoundml_test_qwen_tts test_qwen_tts.cpp)
brosoundml_add_test(brosoundml_test_qwen_asr test_qwen_asr.cpp)
brosoundml_add_test(brosoundml_test_rave     test_rave.cpp)
brosoundml_add_test(brosoundml_test_speaker_encoder test_speaker_encoder.cpp)
target_compile_definitions(brosoundml_test_speaker_encoder PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
brosoundml_add_test(brosoundml_test_kokoro_modules test_kokoro_modules.cpp)
brosoundml_add_test(brosoundml_test_decoder_lora test_decoder_lora.cpp)
brosoundml_add_test(brosoundml_test_kokoro_decoder_backward test_kokoro_decoder_backward.cpp)
target_include_directories(brosoundml_test_kokoro_decoder_backward PRIVATE
    "${CMAKE_SOURCE_DIR}/src")
brosoundml_add_test(brosoundml_test_wake     test_wake.cpp)
brosoundml_add_test(brosoundml_test_wake_data test_wake_data.cpp)
brosoundml_add_test(brosoundml_test_phoneme_data test_phoneme_data.cpp)
brosoundml_add_test(brosoundml_test_wake_inspect test_wake_inspect.cpp)
brosoundml_add_test(brosoundml_test_bc_resnet test_bc_resnet.cpp)
brosoundml_add_test(brosoundml_test_bc_resnet_train test_bc_resnet_train.cpp)
brosoundml_add_test(brosoundml_test_bc_resnet2d test_bc_resnet2d.cpp)
brosoundml_add_test(brosoundml_test_bc_resnet2d_train test_bc_resnet2d_train.cpp)
brosoundml_add_test(brosoundml_test_phoneme_model test_phoneme_model.cpp)
brosoundml_add_test(brosoundml_test_mel      test_mel.cpp)
brosoundml_add_test(brosoundml_test_whisper  test_whisper.cpp)
brosoundml_add_test(brosoundml_test_whisper_modules test_whisper_modules.cpp)
brosoundml_add_test(brosoundml_test_whisper_decoder test_whisper_decoder.cpp)
brosoundml_add_test(brosoundml_test_whisper_e2e test_whisper_e2e.cpp)
brosoundml_add_test(brosoundml_test_pos_tagger test_pos_tagger.cpp)
target_compile_definitions(brosoundml_test_pos_tagger PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(brosoundml_test_pos_tagger PRIVATE
    "${CMAKE_SOURCE_DIR}/src")
brosoundml_add_test(brosoundml_test_lexicon test_lexicon.cpp)
target_compile_definitions(brosoundml_test_lexicon PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
brosoundml_add_test(brosoundml_test_morphology test_morphology.cpp)
target_compile_definitions(brosoundml_test_morphology PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
brosoundml_add_test(brosoundml_test_special_cases test_special_cases.cpp)
target_compile_definitions(brosoundml_test_special_cases PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
brosoundml_add_test(brosoundml_test_normalizer test_normalizer.cpp)
brosoundml_add_test(brosoundml_test_phoneme_adapter test_phoneme_adapter.cpp)
target_compile_definitions(brosoundml_test_phoneme_adapter PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
brosoundml_add_test(brosoundml_test_phonemizer test_phonemizer.cpp)
target_compile_definitions(brosoundml_test_phonemizer PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
# Surface the repo root to the tests' optional real-weights smoke paths.
# Training driver — built but NOT registered as a ctest (manual tool).
add_executable(train_pos_tagger train_pos_tagger.cpp)
target_link_libraries(train_pos_tagger PRIVATE brosoundml)
target_include_directories(train_pos_tagger PRIVATE
    "${CMAKE_SOURCE_DIR}/src")
if(MSVC)
    target_compile_options(train_pos_tagger PRIVATE /W4)
else()
    target_compile_options(train_pos_tagger PRIVATE -Wall -Wextra)
endif()

# Confusion-matrix analyser — built but NOT a ctest (manual tool).
add_executable(eval_pos_confusion eval_pos_confusion.cpp)
target_link_libraries(eval_pos_confusion PRIVATE brosoundml)
if(MSVC)
    target_compile_options(eval_pos_confusion PRIVATE /W4)
else()
    target_compile_options(eval_pos_confusion PRIVATE -Wall -Wextra)
endif()

target_compile_definitions(brosoundml_test_kokoro PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(brosoundml_test_qwen_tts PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(brosoundml_test_qwen_tts PRIVATE
    "${CMAKE_SOURCE_DIR}/src")
target_compile_definitions(brosoundml_test_qwen_asr PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(brosoundml_test_qwen_asr PRIVATE
    "${CMAKE_SOURCE_DIR}/src")
target_compile_definitions(brosoundml_test_kokoro_modules PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(brosoundml_test_whisper PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
