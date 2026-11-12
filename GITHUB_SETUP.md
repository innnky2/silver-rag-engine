# Publish to GitHub + insert link into docx builder

Repo name used here: `ours-engine`.

## 1. Create empty GitHub repo (web, no CLI auth needed)

1. github.com -> New repository -> name `ours-engine`, Public, no README/license/gitignore (folder already has them).
2. Copy the HTTPS URL, e.g. `https://github.com/<user>/ours-engine`.

## 2. Push this folder (defense/office machine with git)

```sh
cd ours-engine
git init
git add .
git commit -m "Ours Engine complete PoC (unmeasured H1-H5)"
git branch -M main
git remote add origin https://github.com/<user>/ours-engine
git push -u origin main
```

## 3. Insert link into docx (edit BUILDER, never .docx directly)

In `build_v23.py` / `refactor_bab3_part1.py` (Ours Engine subsection + Daftar Pustaka):

```python
# Bab 3 — Ours Engine (kode lengkap, belum diukur):
# Repo: https://github.com/<user>/ours-engine
# Isi: include/hlq_tmac.h, src/hlq_tmac.cpp (ELUTQ-style CPU Appendix B/D/H.1),
#      lut_gemm/kv_q8/ngram_spec + gguf/threadpool/engine (H1-H5),
#      tools/export_hlq_cpu.py (bitplane packing),
#      README.md (ours info/gemm/kv/h4/spec/roofline).
# Status: PROD baris-nyata, BELUM DIUKUR; Plot III.16-III.19 = prediksi.
# Ukur di mesin sidang vs llama.cpp Q4 sama sebelum klaim apa pun.
```

Daftar Pustaka entry:

```python
"Ours Engine — https://github.com/<user>/ours-engine — PoC INT4xINT8 + KV-Q8 + n-gram spec + HLQ-CPU path (unmeasured H1-H5)."
```

Replace `<user>` with the real username after push, rebuild v31, verify link resolves.
