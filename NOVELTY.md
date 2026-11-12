# NOVELTY — Ours vs ELUTQ / T-MAC / AnyBCQ (PROD, unmeasured)

Status: PROD code on real files. Synthetic harness deleted.
No measured numbers are claimed here.

## 1. Yang dipinjam dari penulis (bukan novelty, wajib sitasi)

- Bit-serial LUT-GEMM: dekomposisi q bidang 1-bit + tabel `2^g` (Appendix B ELUTQ,
  garis T-MAC/FIGLUT/LUT-GEMM/AnyBCQ). Implementasi: `hlq_tmac.*` (g=4, 16 entri,
  mirror 16->8, Fig.13 NEON 128-bit, AVX2).
- HLQ bipolar `shat/zhat` (Appendix D.2), mirror lossless, table-quant opsional (D.3).
- Kernel-eval T-MAC + prefill/decode split + perbandingan vs llama.cpp/AWQ (H.1,
  Tabel 11-14, Fig.7). Protokol bench kami meniru split ini.

## 2. Novelty yang diajukan (paket drop-in GGUF yang sudah ada)

1. **Integer-dot + satu skala akhir (hapus dekuantisasi)** — `lut_gemm.*` H1:
   `dot = d_w*d_x*(SUM(w*x) - 8*SUM(x))`, inner loop hanya
   `_mm256_maddubs_epi16` + `_mm256_madd_epi16`, tanpa FMA FP. ELUTQ masih
   memakai tabel FP16 + rekonstruksi; Ours menghapus dekuant nibble→FP.
2. **KV-cache Q8 blok-256 + decode attention di atasnya** — `kv_q8.*` +
   `blocked_attn.*` H2: append O(1) dalam rentang berjalan (O(blok) saat
   pelebaran rentang, lalu amortisasi O(1)), load O(1), byte ~1/2 FP16.
   ELUTQ/NanoQuant adalah weight-only; KV bukan cakupan mereka. Two-pass
   streaming eksak (bukan fused/online-softmax ala FlashAttention).
3. **Spekulasi n-gram gratis** — `ngram_spec.*` H3: tabel order-4 FNV-1a,
   draft K≤8, ambang `speedup≥1,2 ∧ accept≥0,6`. Bukan draft-model.
4. **Tiled-32 + pool persisten + arena aligned** — `pack_q4_tiled_mat` +
   bench H4 (`q4_blockscales_gemv_tiled_pool` vs serial naive): repack
   offline tanpa overhead runtime, pool persisten ber-pin (tanpa spawn
   per-call), arena 64B-aligned + advis huge-page best-effort (bukan klaim
   NUMA/huge-page termaterialisasi penuh).
5. **Akuntansi roofline eksplisit** — H5: `plafon = bandwidth_efektif / byte_per_token`,
   rasio realisasi dilaporkan; <30% = salah konfigurasi, bukan alasan ganti engine.
6. **Jalur 2-bit untuk checkpoint ternary** — `build_lut2`/`lut2_dot`
   terimplementasi (gather skalar eksak + add-tree AVX2 32-wide; shuffle
   `shuffle_epi8` penuh = future work), TETAPI tidak dipakai jalur model PKL
   (Q4/Q8); H1 memakai INT4×INT8 yang drop-in ke model PKL yang ada.
   ELUTQ/NanoQuant butuh kuantisasi ulang + kalibrasi (ELUTQ 4K C4 + finetune;
   NanoQuant 128 sampel + ADMM). Ours: tanpa requant.

## 3. Kriteria gugur (falsifikasi = hasil)

- H1 gugur bila gain <5% pada 3 run `ours gemm --model` di baris GGUF nyata.
- H2 gugur bila Q8 <95% FP16 pada ≤8K.
- H4 gugur bila <5% vs naive.
- H5 gagal bila realisasi <90% llama.cpp pada byte sama → arsitektur GAGAL.
- Semua Plot III.16–III.19 = prediksi/model sampai bench di mesin sidang.
  Jika gagal, laporan wajib tulis "gagal".

## 4. Peta file

| File | Peran |
|---|---|
| `include/hlq_tmac.h`, `src/hlq_tmac.cpp` | Pinjaman gaya penulis (HLQ bipolar, LUT g=4, mirror komplemen, table-quant) + transpose antar-bidang D.1 `pack_planes_interleaved`, jalur per-group penuh `bitserial_gemv_tmac_pergroup` (AVX2 8-wide / NEON 4-wide), jalur file-packed `bitserial_gemv_tmac_packed` |
| `include/lut_gemm.h`, `src/lut_gemm.cpp` | Novelty H1 (integer-dot AVX2 `maddubs`, pool persisten `*_pool`, pack tiled matriks + arena aligned, lut2 add-tree AVX2) |
| `include/kv_q8.h`, `src/kv_q8.cpp` | Novelty H2 (K/V terpisah, running min/max + histori eksak; `bytes_wire_*` klaim H2, `bytes_used` residen jujur; O(1) dalam rentang) |
| `include/ngram_spec.h`, `src/ngram_spec.cpp` | Novelty H3 (tabel order-4 FNV-1a + mode) |
| `tools/export_hlq_cpu.py` | Packer bitplane nyata (wajib scales.json; menolak manifest-only dan .npy; manifest catat ngroups/plane_bytes; read-back popcount verify; format dibaca `bitserial_gemv_tmac_packed`) |
| `src/main.cpp`, `include/bench.h` | CLI PROD `ours {info\|gemm\|kv\|h4\|spec\|run\|roofline}` (file nyata; `kv` distribusi bobot nyata + error; `spec` tokenizer BPE nyata wajib `--model`) |
| `include/gguf.h`, `src/gguf.cpp` | GGUFv3 reader (ukuran Q4_1=20B/Q5_1=24B GGML-exact) + Q4_0→prod per-block + dequant GGML-exact Q4_0–Q8_1/Q2_K–Q8_K split-halves & strided (IQ/BF16 ditolak eksplisit) |
| `include/threadpool.h`, `include/engine.h`, `src/engine.cpp` | Pool persisten + Engine baris-nyata |
