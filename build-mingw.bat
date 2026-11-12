@echo off
REM Ours Engine PROD build. No run here.
REM Requires: g++ C++17 + AVX2. Output: build-gpp\ours.exe
setlocal
set CXXFLAGS=-std=c++17 -O2 -mavx2 -mfma -Iinclude
if not exist build-gpp mkdir build-gpp
g++ %CXXFLAGS% -c src/lut_gemm.cpp -o build-gpp/lut_gemm.o || exit /b 1
g++ %CXXFLAGS% -c src/hlq_tmac.cpp -o build-gpp/hlq_tmac.o || exit /b 1
g++ %CXXFLAGS% -c src/kv_q8.cpp -o build-gpp/kv_q8.o || exit /b 1
g++ %CXXFLAGS% -c src/blocked_attn.cpp -o build-gpp/blocked_attn.o || exit /b 1
g++ %CXXFLAGS% -c src/ngram_spec.cpp -o build-gpp/ngram_spec.o || exit /b 1
g++ %CXXFLAGS% -c src/gguf.cpp -o build-gpp/gguf.o || exit /b 1
g++ %CXXFLAGS% -c src/quant.cpp -o build-gpp/quant.o || exit /b 1
g++ %CXXFLAGS% -c src/norm_rope.cpp -o build-gpp/norm_rope.o || exit /b 1
g++ %CXXFLAGS% -c src/tokenizer.cpp -o build-gpp/tokenizer.o || exit /b 1
g++ %CXXFLAGS% -c src/sampler.cpp -o build-gpp/sampler.o || exit /b 1
g++ %CXXFLAGS% -c src/model.cpp -o build-gpp/model.o || exit /b 1
g++ %CXXFLAGS% -c src/engine.cpp -o build-gpp/engine.o || exit /b 1
g++ %CXXFLAGS% -c src/main.cpp -o build-gpp/main.o || exit /b 1
g++ %CXXFLAGS% build-gpp/lut_gemm.o build-gpp/hlq_tmac.o build-gpp/kv_q8.o build-gpp/blocked_attn.o build-gpp/ngram_spec.o build-gpp/gguf.o build-gpp/quant.o build-gpp/norm_rope.o build-gpp/tokenizer.o build-gpp/sampler.o build-gpp/model.o build-gpp/engine.o build-gpp/main.o -o build-gpp/ours.exe || exit /b 1
echo BUILD-OK: build-gpp\ours.exe (PROD, untested here)
