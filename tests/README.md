# Tests — PROD (ditulis, BELUM dijalankan di sini)

Atas permintaan: kode test dibuat tanpa dieksekusi. Jalankan di mesin sidang.

## File

| File | Tanpa model | Dengan model |
|---|---|---|
| `test_quant.cpp` | Ya (vektor known-answer Q4_0/Q8) | — |
| `test_norm_rope.cpp` | Ya (rms/softmax/RoPE/matvec) | — |
| `test_kv_attn.cpp` | Ya (K/V independen, batas blok 256/257) | — |
| `test_token_sampler.cpp` (greedy/top-k) | Ya | `--model` untuk roundtrip tokenizer |
| `test_gguf.cpp` | Tabel `tensor_nbytes` + `q4_0_to_prod` | `--model` untuk open/read |
| `test_model.cpp` | — | `--model` + `--max-new` (SKIP=2 bila tanpa model) |

## Build manual (mesin sidang)

```sh
g++ -std=c++17 -O2 -mavx2 -mfma -Iinclude tests/test_quant.cpp src/quant.cpp -o /tmp/t_quant && /tmp/t_quant
g++ -std=c++17 -O2 -mavx2 -mfma -Iinclude tests/test_norm_rope.cpp src/norm_rope.cpp -o /tmp/t_nr && /tmp/t_nr
g++ -std=c++17 -O2 -mavx2 -mfma -Iinclude tests/test_kv_attn.cpp src/kv_q8.cpp src/blocked_attn.cpp -o /tmp/t_kv && /tmp/t_kv
g++ -std=c++17 -O2 -mavx2 -mfma -Iinclude tests/test_gguf.cpp src/gguf.cpp -o /tmp/t_gguf && /tmp/t_gguf --model model.gguf
g++ -std=c++17 -O2 -mavx2 -mfma -Iinclude tests/test_token_sampler.cpp src/tokenizer.cpp src/sampler.cpp src/gguf.cpp -o /tmp/t_ts && /tmp/t_ts --model model.gguf
cmake -B build -DOURS_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/test_model --model model.gguf --max-new 8
```

Exit code: 0=PASS, 1=FAIL, 2=SKIP (khusus test_model tanpa model).
