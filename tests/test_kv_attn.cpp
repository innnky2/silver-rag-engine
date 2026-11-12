// test_kv_attn.cpp — PROD tests for Q8 KV + attention (no model file needed).
// Build: g++ -std=c++17 -Iinclude tests/test_kv_attn.cpp src/kv_q8.cpp src/blocked_attn.cpp -o /tmp/t_kv
#include <cmath>
#include <cstdio>
#include <vector>

#include "blocked_attn.h"
#include "kv_q8.h"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg); ++fails; } } while (0)

int main() {
  const int H = 2, D = 8;
  ours::KvCacheQ8 kv(H, D);
  std::vector<float> k((size_t)H * D), v((size_t)H * D);
  for (int t = 0; t < 10; ++t) {
    for (size_t i = 0; i < k.size(); ++i) { k[i] = (float)t * 0.1f + (float)i * 0.01f; v[i] = (float)t * 0.05f; }
    kv.append(k.data(), v.data());
  }
  CHECK(kv.seq_len() == 10, "seq len 10");
  CHECK(kv.bytes_total() >= kv.bytes_wire_total(), "resident includes history");
  CHECK(kv.bytes_fp16_same_len() == (size_t)10 * H * D * 2 * 2, "fp16 bytes");
  // Wire ratio on tiny dims is ~1.1 (scales dominate) — honest. The H2 ~2x
  // claim needs production dims; check it with H=8,D=128,L=256 below.
  // K and V buffers must differ (regression: old code reused K for V).
  {
    std::vector<float> k0((size_t)D), v0((size_t)D);
    kv.blocks[0].load_k_row(0, 9, k0.data());
    kv.blocks[0].load_v_row(0, 9, v0.data());
    bool differ = false;
    for (int d = 0; d < D; ++d) if (std::fabs(k0[d] - v0[d]) > 1e-4) differ = true;
    CHECK(differ, "K/V buffers independent");
  }
  // Attention output is a convex combination of V rows: each out[d] within [min V, max V].
  {
    std::vector<float> q((size_t)D, 1.0f), tmp(10 + 2 * D), out(D), Kf((size_t)10 * H * D), Vf((size_t)10 * H * D);
    for (int p = 0; p < 10; ++p)
      for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d) {
          Kf[((size_t)p * H + h) * D + d] = (float)p * 0.1f + (float)(h * D + d) * 0.01f;
          Vf[((size_t)p * H + h) * D + d] = (float)p * 0.05f;
        }
    ours::attn_decode_step(kv, 0, q.data(), tmp.data(), out.data());
    std::vector<float> tmp2(10 + 2 * D), out2(D);
    ours::attn_decode_step_fp16(Kf.data(), Vf.data(), 10, H, D, 0, q.data(), tmp2.data(), out2.data());
    for (int d = 0; d < D; ++d)
      CHECK(std::fabs(out[d] - out2[d]) < 0.02f, "q8 attn near fp16");
  }
  // Block boundary: 257 tokens -> 2 blocks, len-aware bytes.
  {
    ours::KvCacheQ8 kv2(1, 4);
    std::vector<float> z(4, 0.0f);
    for (int t = 0; t < 257; ++t) kv2.append(z.data(), z.data());
    CHECK(kv2.blocks.size() == 2 && kv2.seq_len() == 257, "block split");
    CHECK(kv2.blocks[0].len == 256 && kv2.blocks[1].len == 1, "block lens");
  }
  // H2 wire ratio at production dims (H=8,D=128,L=256): must approach ~2x.
  {
    ours::KvCacheQ8 kv3(8, 128);
    std::vector<float> k3(8 * 128), v3(8 * 128);
    for (int t = 0; t < 256; ++t) {
      for (size_t i = 0; i < k3.size(); ++i) {
        k3[i] = (float)t * 0.01f + (float)i * 0.001f;
        v3[i] = (float)t * 0.005f - (float)i * 0.0005f;
      }
      kv3.append(k3.data(), v3.data());
    }
    double r = (double)kv3.bytes_fp16_same_len() / (double)kv3.bytes_wire_total();
    CHECK(r > 1.7 && r < 2.2, "wire ratio near 2x at production dims");
  }
  if (fails == 0) std::printf("PASS test_kv_attn (4 groups)\n");
  return fails ? 1 : 0;
}
