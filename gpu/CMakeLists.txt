# basal-logits and llama-bench for a phone, on llama.cpp from a separate checkout:
#
#   cmake -G Ninja -S gpu -B <build> -DLLAMA_DIR=<llama.cpp> -DCMAKE_TOOLCHAIN_FILE=<ndk>/build/cmake/android.toolchain.cmake
#         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON ...
#   ninja -C <build> basal-logits llama-bench
cmake_minimum_required(VERSION 3.14)
project(basal-gpu C CXX)

set(LLAMA_DIR "" CACHE PATH "llama.cpp checkout")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(basal-logits basal_logits.cpp)
target_link_libraries(basal-logits PRIVATE llama vendor::nlohmann)
target_compile_features(basal-logits PRIVATE cxx_std_17)
