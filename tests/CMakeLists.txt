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
# Surface the repo root to test_kokoro's optional real-weights smoke path.
target_compile_definitions(brosoundml_test_kokoro PRIVATE
    BROSOUNDML_REPO_DIR="${CMAKE_SOURCE_DIR}")
