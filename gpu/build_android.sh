#!/bin/bash
# build_android.sh cl|vk <build dir>: basal-logits and llama-bench for arm64 Android with llama.cpp's OpenCL (Adreno
# kernels) or Vulkan backend, both with the CPU backend too (-ngl 0). The OpenCL and Vulkan headers, the OpenCL ICD
# loader and glslc come from android-deps, as for stable-diffusion.cpp's Android builds.
set -euo pipefail
backend=$1 build=$2
LLAMA=${LLAMA:-C:/Users/krzgo/CLionProjects/llama.cpp}
DEPS=${DEPS:-C:/Users/krzgo/CLionProjects/android-deps}
NDK=${NDK:-C:/Users/krzgo/AppData/Local/Android/Sdk/ndk/30.0.16248370}
CLION="C:/Program Files/JetBrains/CLion 2026.2.3/bin"
CMAKE="$CLION/cmake/win/x64/bin/cmake.exe"
NINJA="$CLION/ninja/win/x64/ninja.exe"
here=$(cd "$(dirname "$0")" && pwd -W)

case $backend in
    cl) opts=(-DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON -DGGML_OPENCL_EMBED_KERNELS=ON
              -DOpenCL_INCLUDE_DIR=$DEPS/OpenCL-Headers -DOpenCL_LIBRARY=$DEPS/OpenCL-ICD-Loader/build_ndk/libOpenCL.so) ;;
    vk) opts=(-DGGML_VULKAN=ON -DVulkan_INCLUDE_DIR=$DEPS/Vulkan-Headers/include
              -DVulkan_GLSLC_EXECUTABLE=$DEPS/shaderc/build/glslc/glslc.exe
              -DVulkan_LIBRARY=$NDK/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/aarch64-linux-android/28/libvulkan.so
              -DSPIRV-Headers_DIR=$DEPS/spirv-headers-install/share/cmake/SPIRV-Headers
              "-DCMAKE_CXX_FLAGS=-I$DEPS/spirv-headers-install/include") ;;
    *) echo "backend: cl or vk"; exit 2 ;;
esac

"$CMAKE" -G Ninja -S "$here" -B "$build" -DCMAKE_MAKE_PROGRAM="$NINJA" -DLLAMA_DIR=$LLAMA \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm+fp16 \
    "${opts[@]}"
"$NINJA" -C "$build" -j 16 basal-logits llama-bench
