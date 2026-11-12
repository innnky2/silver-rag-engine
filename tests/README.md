# Tests

| File | No model | With model |
|---|---|---|
| `test_quant.cpp` | Yes (known-answer Q4_0/Q8 + K-quants) | — |
| `test_norm_rope.cpp` | Yes (rms/softmax/RoPE/YaRN/matvec) | — |
| `test_kv_attn.cpp` | Yes (K/V independence, 256/257 block edge) | — |
| `test_hlq.cpp` | Yes (interleave/packed roundtrips) | — |
| `test_token_sampler.cpp` (greedy/top-k) | Yes | `--model` for tokenizer roundtrip |
| `test_gguf.cpp` | `tensor_nbytes` tables + `q4_0_to_prod` chain | `--model` for open/read |
| `test_model.cpp` | — | `--model` + `--max-new` (SKIP=2 without model) |

## Manual build

```sh
g++ -std=c++17 -O2 -mavx2 -mfma -Iinclude tests/test_quant.cpp src/quant.cpp -o /tmp/t_quant && /tmp/t_quant
g++ -std=c++17 -O2 -mavx2 -mfma -Iinclude tests/test_norm_rope.cpp src/norm_rope.cpp -o /tmp/t_nr && /tmp/t_nr
g++ -std=c++17 -O2 -mavx2 -mfma -Iinclude tests/test_kv_attn.cpp src/kv_q8.cpp src/blocked_attn.cpp -o /tmp/t_kv && /tmp/t_kv
g++ -std=c++17 -O2 -mavx2 -mfma -Iinclude tests/test_hlq.cpp src/hlq_tmac.cpp -o /tmp/t_hlq && /tmp/t_hlq
g++ -std=c++17 -O2 -mavx2 -mfma -Iinclude tests/test_gguf.cpp src/gguf.cpp src/quant.cpp -o /tmp/t_gguf && /tmp/t_gguf --model model.gguf
g++ -std=c++17 -O2 -mavx2 -mfma -Iinclude tests/test_token_sampler.cpp src/tokenizer.cpp src/sampler.cpp src/gguf.cpp -o /tmp/t_ts && /tmp/t_ts --model model.gguf
cmake -B build -DOURS_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/test_model --model model.gguf --max-new 8
```

Exit code: 0=PASS, 1=FAIL, 2=SKIP (test_model without model only).
