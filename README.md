# OURS ENGINE — PROD (real GGUF weights, full decode)

Status: PROD code path on real `.gguf` files. No synthetic RNG harness.
`ours kv` uses real attn_k/v weight distributions (no zero-fill);
`ours spec` uses the real GGUF BPE tokenizer (no hash surrogate, requires `--model`).
No numbers here are measured — run on the defense machine before any claim.

## 1. Architecture thesis

llama.cpp Q4 loses not because its C++ is bad, but because every decode token
pays: (a) **byte overhead** — ~4.89 bits/weight from DRAM, (b) **dequantization
overhead** — nibble→FP + FMA. Ours attacks both **without changing the model**:
integer-dot plus a single final scale, Q8 KV-cache with separate K/V blocks,
runtime `1/sqrt(D)` attention, tiled-32 + pinned persistent `ThreadPool`,
n-gram draft with target-verify, and explicit roofline attribution.

## 2. PROD scope (complete, real files)

| File | Role |
|---|---|
| `include/gguf.h`, `src/gguf.cpp` | GGUFv3 full-spec: all KV types, tokenizer arrays, all common quants F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q2_K–Q8_K + explicit IQ/BF16 sizes |
| `include/quant.h`, `src/quant.cpp` | GGML-exact dequant F32/F16/Q4_0–Q8_1 + Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K (split-halves/strided, aux-shuffle Q3_K), f32→Q8 act quant, per-block-scales integer dot |
| `include/threadpool.h` | Pinned persistent pool, chunked parallel_for (1 job/worker) |
| `include/norm_rope.h`, `src/norm_rope.cpp` | RMSNorm, NeoX RoPE, llama-faithful YaRN NTK-by-parts (linear ramp + mscale), SiLU, softmax, f32 matvec |
| `include/tokenizer.h`, `src/tokenizer.cpp` | Longest-match BPE from `tokenizer.ggml.*`, byte-fallback, BOS/EOS |
| `include/sampler.h`, `src/sampler.cpp` | Greedy/temp/top-k/top-p/min-p |
| `include/model.h`, `src/model.cpp` | Full Llama decode: embed → attn GQA+YaRN+Q8-KV → SwiGLU → norm → lm_head → generate |
| `include/engine.h`, `src/engine.cpp` | Engine wrapper: info/GEMV-H1/H4 + real-text generate |
| `include/lut_gemm.h`, `src/lut_gemm.cpp` | H1 integer-dot + tiled GEMV (persistent pool, aligned arena) |
| `include/kv_q8.h`, `src/kv_q8.cpp` | H2 separate K/V, len-aware wire vs resident |
| `include/blocked_attn.h`, `src/blocked_attn.cpp` | Exact two-pass decode attention, runtime `1/sqrt(D)` |
| `include/ngram_spec.h`, `src/ngram_spec.cpp` | H3 draft + target-verify |
| `include/hlq_tmac.h`, `src/hlq_tmac.cpp` | HLQ CPU path (complement mirror; D.1 interleave; full per-group + AVX2/NEON; packed-file path; global-scale = upper-bound wrapper) |
| `tools/export_hlq_cpu.py` | Real bitplane packer (scales.json required; ngroups/plane_bytes manifest; read-back verify) |
| `src/main.cpp` | CLI `ours {info|gemm|kv|h4|spec|run|roofline}` |

## 3. Hypotheses H1–H5 (measured on the defense machine)

| H | Mechanism | Pass when | Fail when |
|---|---|---|---|
| H1 | `lut_gemm.*` INT4×INT8, `q4_q8_dot` | checksum OK vs ref on real GGUF rows | mismatch / gain <5% |
| H2 | `kv_q8.*` separate 256-block K/V + exact running min/max, len-aware `bytes_wire_total()` | ≤8K overhead <5% + small error | >5% at ≤8K → per-channel |
| H3 | `ngram_spec.*` order-4, K≤8, 20% dup-guard | speedup≥1.2 ∧ accept≥0.6 on real prompts | below → drop draft |
| H4 | `pack_q4_tiled_mat` + aligned arena + persistent pool (`q4_blockscales_gemv_tiled_pool`) | ≥+5% vs serial naive on real rows | <5% → bandwidth saturated |
| H5 | `run_roofline`: ceiling = BW/bytes | 40–60% healthy; Ours ≥90% of same-bytes llama | <30% = misconfigured |

## 4. PROD notes

1. `q4_0_to_prod` keeps per-block 32-element scales (never averaged);
   the fast Q4_0 path uses per-block scales, the generic path uses full dequant.
   `q4_blockscales_dot` and `matvec_weight_q8fast` use integer-dot
   (AVX2 `maddubs` when available, exact scalar fallback).
2. Q4_K_M / Q5_K_M / Q6_K / Q8_K are supported via full dequant + `ours run`
   (not rejected). IQ/BF16 need model conversion first (explicit error).
3. The `gemm/h4` act = a real `token_embd` row slice (model vector, Q8),
   not a derived-byte probe.
4. `kv` uses real `blk.0.attn_k/v` weight rows (not zeros) + reports
   `wire` (H2 claim) vs `resident` (includes exact history) + maxabs/MSE.
   `spec` uses real BPE ids (requires `--model`); without a model = REFUSED.
5. A "faster than llama.cpp" claim is only valid against a `-ngl 0` baseline on
   the same machine/threads/file (see `USAGE.md`).

## 5. Usage

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/ours info --model model.gguf
./build/ours gemm --model model.gguf --threads 4
./build/ours kv   --model model.gguf --depths 2K,8K,16K,32K
./build/ours h4   --model model.gguf --threads 4
./build/ours spec --model model.gguf --prompts real_work.txt --K 8
./build/ours run  --model model.gguf --prompt "Hello world" --max-new 64 --temp 0 --threads 4
./build/ours roofline --bw 38 --bytes 4900000000 --tg 3.6
```
