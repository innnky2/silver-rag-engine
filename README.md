# OURS ENGINE — PROD (real GGUF weights, full decode)

Status: PROD code path on real `.gguf` files. No synthetic RNG harness.
`ours kv` uses real attn_k/v weight distributions (no zero-fill);
`ours spec` uses the real GGUF BPE tokenizer (no hash surrogate, requires `--model`).
No numbers here are measured — run on the defense machine before any claim.

## 1. Tesis arsitektur

llama.cpp Q4 kalah bukan karena C++-nya jelek, melainkan tiap token decode
membayar: (a) **overhead byte** — ~4,89 bit/bobot dari DRAM, (b) **overhead
dekuantisasi** — nibble→FP + FMA. Ours menyerang keduanya **tanpa ganti model**:
integer-dot + satu skala akhir, KV-cache Q8 blok K/V terpisah, attention
`1/sqrt(D)` runtime, tiled-32 + `ThreadPool` persisten ber-pin, n-gram draft
dengan target-verify, dan atribusi roofline eksplisit.

## 2. Cakupan PROD (lengkap, file nyata)

| File | Peran |
|---|---|
| `include/gguf.h`, `src/gguf.cpp` | GGUFv3 full-spec: semua tipe KV, array tokenizer, semua quant umum F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q2_K–Q8_K + ukuran IQ/BF16 eksplisit |
| `include/quant.h`, `src/quant.cpp` | Dequant GGML-exact F32/F16/Q4_0–Q8_1 + Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K (split-halves/strided, aux-shuffle Q3_K), quant act f32→Q8, dot per-block-scales integer |
| `include/threadpool.h` | Pool persisten ber-pin, chunked parallel_for (1 job/worker) |
| `include/norm_rope.h`, `src/norm_rope.cpp` | RMSNorm, RoPE NeoX, YaRN NTK-by-parts llama-faithful (ramp linear + mscale), SiLU, softmax, matvec f32 |
| `include/tokenizer.h`, `src/tokenizer.cpp` | BPE longest-match dari `tokenizer.ggml.*`, byte-fallback, BOS/EOS |
| `include/sampler.h`, `src/sampler.cpp` | Greedy/temp/top-k/top-p/min-p |
| `include/model.h`, `src/model.cpp` | Llama decode penuh: embed → attn GQA+YaRN+KV-Q8 → SwiGLU → norm → lm_head → generate |
| `include/engine.h`, `src/engine.cpp` | Wrapper Engine: info/GEMV-H1/H4 + generate teks nyata |
| `include/lut_gemm.h`, `src/lut_gemm.cpp` | H1 integer-dot + tiled GEMV (pool persisten, arena aligned) |
| `include/kv_q8.h`, `src/kv_q8.cpp` | H2 K/V terpisah, wire vs residen len-aware |
| `include/blocked_attn.h`, `src/blocked_attn.cpp` | Decode attention two-pass eksak `1/sqrt(D)` runtime |
| `include/ngram_spec.h`, `src/ngram_spec.cpp` | H3 draft + target-verify |
| `include/hlq_tmac.h`, `src/hlq_tmac.cpp` | Jalur HLQ CPU (mirror komplemen; interleave D.1; per-group penuh + AVX2/NEON; jalur file-packed; global-scale = wrapper upper-bound) |
| `tools/export_hlq_cpu.py` | Packer bitplane nyata (wajib scales.json; manifest ngroups/plane_bytes; read-back verify) |
| `src/main.cpp` | CLI `ours {info|gemm|kv|h4|spec|run|roofline}` |

## 3. Hipotesis H1–H5 (diukur di mesin sidang)

| H | Mekanisme | Lulus bila | Gugur bila |
|---|---|---|---|
| H1 | `lut_gemm.*` INT4×INT8, `q4_q8_dot` | checksum OK vs ref di baris GGUF nyata | mismatch / gain <5% |
| H2 | `kv_q8.*` blok-256 K/V terpisah + running min/max eksak, `bytes_wire_total()` len-aware | ≤8K overhead <5% + error kecil | >5% di ≤8K → per-channel |
| H3 | `ngram_spec.*` order-4, K≤8, dup-guard 20% | speedup≥1,2 ∧ accept≥0,6 di prompt nyata | di bawah → lepas draf |
| H4 | `pack_q4_tiled_mat` + arena aligned + pool persisten (`q4_blockscales_gemv_tiled_pool`) | ≥+5% vs serial naive di baris nyata | <5% → bandwidth jenuh |
| H5 | `run_roofline`: plafon = BW/byte | 40–60% sehat; Ours ≥90% llama byte-sama | <30% = miskonfigurasi |

## 4. Catatan PROD

1. `q4_0_to_prod` menyimpan skala per-block 32-elemen (tidak di-average);
   jalur cepat Q4_0 memakai skala per-block, jalur generik memakai dequant penuh.
   `q4_blockscales_dot` dan `matvec_weight_q8fast` memakai integer-dot
   (`maddubs` AVX2 bila tersedia, fallback skalar eksak).
2. Q4_K_M / Q5_K_M / Q6_K / Q8_K didukung via dequant penuh + `ours run`
   (bukan ditolak). IQ/BF16 butuh konversi model dulu (error eksplisit).
3. Act `gemm/h4` = slice baris `token_embd` nyata (vektor model, Q8),
   bukan probe derivasi-byte.
4. `kv` memakai baris bobot `blk.0.attn_k/v` nyata (bukan nol) + lapor
   `wire` (klaim H2) vs `resident` (termasuk histori eksak) + maxabs/MSE.
   `spec` memakai IDs BPE nyata (wajib `--model`); tanpa model = REFUSED.
5. Klaim "lebih cepat dari llama.cpp" hanya sah dengan baseline `-ngl 0`
   mesin/thread/file sama (lihat `USAGE.md`).

## 5. Cara pakai

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/ours info --model model.gguf
./build/ours gemm --model model.gguf --threads 4
./build/ours kv   --model model.gguf --depths 2K,8K,16K,32K
./build/ours h4   --model model.gguf --threads 4
./build/ours spec --model model.gguf --prompts kerja_nyata.txt --K 8
./build/ours run  --model model.gguf --prompt "Halo dunia" --max-new 64 --temp 0 --threads 4
./build/ours roofline --bw 38 --bytes 4900000000 --tg 3.6
```
