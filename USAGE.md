# Ours Engine — PROD usage (defense machine)

Status: PROD code on real files. Nothing here was run; all numbers must be
measured on your box before any claim.

## 1. Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# Windows + MinGW without cmake: build-mingw.bat  ->  build-gpp/ours.exe
```

Requires C++17 (AVX2 on x86 only, scalar/NEON fallback on ARM). No GPU. No RNG in binary
except sampler stream for temp>0.

## 2. Run (real files only)

```sh
./build/ours info --model model.gguf
./build/ours gemm --model model.gguf --threads 4 --repeats 5
./build/ours kv   --model model.gguf --depths 2K,8K,16K,32K
./build/ours h4   --model model.gguf --threads 4
./build/ours spec --model model.gguf --prompts kerja_nyata.txt --K 8
./build/ours run  --model model.gguf --prompt "Tulis ringkasan:" --max-new 64 --temp 0 --threads 4
./build/ours roofline --bw <GB/s> --bytes <GGUF_bytes> --tg <measured_tg>
```

Rules:
- `kerja_nyata.txt`: real prompts, one per line. Rejects dup_ratio > 20%.
- H1: checksum must be OK on real Q4_0 rows with real token_embd act; K-quants via `run`.
  Integer-dot is AVX2 `maddubs` (x86) / exact scalar (ARM).
- H2: wire bytes are len-aware K+V (H2 claim); resident bytes include exact
  history. Reports wire_ratio + resident_ratio + maxabs/MSE on real attn_k/v
  distributions; overhead % needs same-box decode timing.
- H3: `spec` requires `--model` (real BPE ids); reports coverage only;
  accept/speedup need a bound target model.
- H5: healthy 40–60%; <30% = misconfig (documented heuristics, decimal GB/s);
  Ours valid at >=90% llama same bytes.
- `run`: full decode greedy (temp 0) or sampled; KV-Q8 per layer; GQA+RoPE.

## 3. Baseline (mandatory)

Same machine, same threads, same GGUF file, llama.cpp `-ngl 0`:

```sh
llama-bench -m <model.gguf> -p 512 -n 128 -t <N_fisik> -ngl 0
```

Without this baseline no "beats llama.cpp" claim is valid.

## 4. HLQ-CPU path (ELUTQ-style, real packing)

```sh
python tools/export_hlq_cpu.py --resume_quant ./output/hlq --wbits 2 --group_size 128 --save_path ./output/hlq-cpu
```

Reads `<resume_quant>/scales.json` + `<tensor>.bin` float planes, writes
`planes.bin` + `scales.f32` + `zhat.f32` + manifest. Refuses manifest-only.
Quant + finetune still need GPU; CPU inference uses `hlq_tmac.*`.
