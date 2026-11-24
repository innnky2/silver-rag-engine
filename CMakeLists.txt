cmake_minimum_required(VERSION 3.16)
project(ours_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# PROD build:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
# Binary: build/ours (Windows: build/Release/ours.exe)

add_library(ours_lib STATIC
  src/lut_gemm.cpp
  src/hlq_tmac.cpp
  src/kv_q8.cpp
  src/blocked_attn.cpp
  src/ngram_spec.cpp
  src/gguf.cpp
  src/quant.cpp
  src/norm_rope.cpp
  src/tokenizer.cpp
  src/sampler.cpp
  src/model.cpp
  src/engine.cpp
)
target_include_directories(ours_lib PUBLIC include)

# x86 SIMD only on x86; ARM (RK3588/M2, Fig.7 targets) uses scalar/NEON paths
# guarded by __AVX2__/__ARM_NEON__ in the sources (no unconditional -mavx2).
if(MSVC)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86|AMD64")
    target_compile_options(ours_lib PRIVATE /arch:AVX2)
  endif()
else()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86|X86|AMD64|amd64")
    target_compile_options(ours_lib PRIVATE -mavx2 -mfma)
  endif()
endif()

add_executable(ours src/main.cpp)
target_link_libraries(ours PRIVATE ours_lib)

option(OURS_BUILD_TESTS "Build PROD tests (run on defense machine)" OFF)
if(OURS_BUILD_TESTS)
  foreach(t quant norm_rope kv_attn gguf token_sampler hlq model)
    add_executable(test_${t} tests/test_${t}.cpp)
    target_link_libraries(test_${t} PRIVATE ours_lib)
    if(NOT MSVC)
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86|X86|AMD64|amd64")
        target_compile_options(test_${t} PRIVATE -mavx2 -mfma)
      endif()
    endif()
  endforeach()
endif()
