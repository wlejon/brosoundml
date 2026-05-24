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
brosoundml_add_test(brosoundml_test_kokoro_modules test_kokoro_modules.cpp)
brosoundml_add_test(brosoundml_test_whisper  test_whisper.cpp)
brosoundml_add_test(brosoundml_test_whisper_modules test_whisper_modules.cpp)
brosoundml_add_test(brosoundml_test_whisper_decoder test_whisper_decoder.cpp)
# Surface the repo root to the tests' optional real-weights smoke paths.
target_compile_definitions(brosoundml_test_kokoro PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(brosoundml_test_kokoro_modules PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
