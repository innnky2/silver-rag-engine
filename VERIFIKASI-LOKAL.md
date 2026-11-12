# VERIFIKASI-LOKAL — RETIRED (synthetic harness deleted)

File ini dulu berisi log bench sintetis (`bench.exe --mode gemm/kv-oh/h4/spec`
dengan `mt19937` random tensors). Harness sintetis tersebut telah DIHAPUS
sesuai permintaan prod (2026-11-12).

Yang berlaku sekarang (PROD, belum dijalankan di sini):
- Build: `build-mingw.bat` -> `build-gpp/ours.exe`, atau cmake -> `build/ours`
- Run: `ours info|gemm|kv|h4|spec|roofline` pada file GGUF + prompts nyata
- Tidak ada angka terukur di repo ini; `fig_engine/bench_actual.json` tetap
  milik run mesin sidang + `llama-bench` baseline.

Riwayat bug sintetis yang sudah diperbaiki sebelum penghapusan (arsip):
- `ngram_spec.cpp` missing `NgramDraft::` scope; `main.cpp` missing `hlq::`
- `lut_gemm.cpp` 256-bit over-read -> jalur SSE 128-bit
- `hlq_tmac.cpp` mirror negasi -> komplemen `S - lut8[15-idx]`
- `kv_q8.cpp` V-buffer kini terpisah K/V + `bytes_total()` len-aware
- `blocked_attn.cpp` `1/sqrt(D)` kini runtime (bukan hardcode 128)
