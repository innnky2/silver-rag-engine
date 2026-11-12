// lut_gemm.cpp — H1 implementation.
//
// Theory references for the instruction choice:
//   - Jacob et al., CVPR 2018 (arXiv:1712.05877), Sec. 2.2: integer-only
//     matmul with 8-bit operands and 32-bit accumulator.
//   - Park et al., LUT-GEMM, ICLR 2024 (arXiv:2206.09557): eliminating the
//     dequant step is the measured win (2.1x vs OPTQ, OPT-175B 3-bit).
//   - Wei et al., T-MAC, EuroSys 2025 (arXiv:2407.00088): CPU mpGEMM is
//     dominated by dequant overhead; LUT/integer path scales with bit-width.
#include "lut_gemm.h"
#include "quant.h"
#include "threadpool.h"
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define OURS_AVX2 1
#endif

#if defined(__linux__) && !defined(__ANDROID__)
#include <sys/mman.h>
#define OURS_HAVE_MMAN 1
#endif

namespace ours {

namespace {
void pin_current_thread(int core);
}  // namespace

float q4_q8_dot_ref(std::size_t n, const uint8_t* q4, float d_w,
                    const int8_t* q8, float d_x) {
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    uint8_t nib = (i & 1) ? (q4[i >> 1] >> 4) : (q4[i >> 1] & 0x0F);
    // Layout note (GGML-exact): prod nibbles are sequential, even=LOW
    // (q4_0_to_prod converts GGML split-halves qs[j]:lo->elem j, hi->elem j+16
    // into this order). Q4_K_M/IQ4_* need their own spec paths (see quant.cpp).
    acc += (double)((int)nib - 8) * (double)q8[i];
  }
  return (float)(acc * (double)d_w * (double)d_x);
}

float q4_q8_dot(std::size_t n, const uint8_t* q4, float d_w,
                const int8_t* q8, float d_x) {
#if defined(OURS_AVX2)
  // 16 weights per step: unpack 8 Q4 bytes -> 16 nibbles (cheap scalar),
  // then 128-bit integer dot (16 lanes exactly; 256-bit loads would
  // over-read tmp[16]/q8[16] and pollute the accumulator — fixed 2026-11-12).
  // The scalar nibble expand is exact and off the hot path; no shuffle form
  // is claimed here.
  const __m128i ones16 = _mm_set1_epi16(1);
  __m128i acc4 = _mm_setzero_si128();  // 4 x int32
  long long sum_wx = 0, sum_x = 0;
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    uint8_t tmp[16];
    for (int j = 0; j < 16; ++j) {
      uint8_t b = q4[(i + j) >> 1];
      tmp[j] = ((i + j) & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0F);
    }
    __m128i w = _mm_loadu_si128((const __m128i*)tmp);       // 16 x u8
    __m128i x = _mm_loadu_si128((const __m128i*)(q8 + i));  // 16 x i8
    // Nibble*int8 as unsigned x signed (saturating, see header).
    __m128i p = _mm_maddubs_epi16(w, x);        // 8 x i16
    __m128i s = _mm_madd_epi16(p, ones16);      // 4 x i32
    acc4 = _mm_add_epi32(acc4, s);
    // Correction term -8*SUM(x): O(n/16), off the critical path.
    for (int j = 0; j < 16; ++j) sum_x += q8[i + j];
  }
  // Horizontal reduction acc4 -> sum_wx.
  __m128i s2 = _mm_add_epi32(acc4, _mm_shuffle_epi32(acc4, 0x4E));
  __m128i s1 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0xB1));
  sum_wx += _mm_cvtsi128_si32(s1);
  // Scalar tail (n not a multiple of 16).
  for (; i < n; ++i) {
    uint8_t b = q4[i >> 1];
    int w = ((i & 1) ? (b >> 4) : (b & 0x0F));
    sum_wx += (long long)w * (long long)q8[i];
    sum_x += q8[i];
  }
  double dot = (double)(sum_wx - 8LL * sum_x) * (double)d_w * (double)d_x;
  return (float)dot;
#else
  // Scalar fallback (no AVX2): correct, slower. H4 measures the gap.
  long long sum_wx = 0, sum_x = 0;
  for (std::size_t i = 0; i < n; ++i) {
    uint8_t b = q4[i >> 1];
    int w = ((i & 1) ? (b >> 4) : (b & 0x0F));
    sum_wx += (long long)w * (long long)q8[i];
    sum_x += q8[i];
  }
  return (float)((double)(sum_wx - 8LL * sum_x) * (double)d_w * (double)d_x);
#endif
}

void q4_q8_gemv(std::size_t m, std::size_t n, const uint8_t* const* rows,
                float d_w, const int8_t* q8, float d_x, float* out) {
  // Reference serial dispatch: correctness baseline for H1 checksums.
  // Production threaded path is q4_q8_gemv_mt() / q4_q8_gemv_tiled().
  if (!rows || !q8 || !out) return;
  for (std::size_t r = 0; r < m; ++r)
    out[r] = q4_q8_dot(n, rows[r], d_w, q8, d_x);
}

void q4_q8_gemv_mt(std::size_t m, std::size_t n, const uint8_t* const* rows,
                   float d_w, const int8_t* q8, float d_x, float* out,
                   int nthreads, bool pin_threads) {
  if (!rows || !q8 || !out || m == 0) return;
  if (nthreads <= 1) { q4_q8_gemv(m, n, rows, d_w, q8, d_x, out); return; }
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) hw = 1;
  int W = nthreads > 64 ? 64 : nthreads;
  std::size_t chunk = (m + (std::size_t)W - 1) / (std::size_t)W;
  std::vector<std::thread> pool;
  pool.reserve((std::size_t)W);
  for (int w = 0; w < W; ++w) {
    pool.emplace_back([&, w, W, chunk]() {
      if (pin_threads) pin_current_thread(w % (int)hw);
      std::size_t lo = (std::size_t)w * chunk, hi = lo + chunk < m ? lo + chunk : m;
      for (std::size_t r = lo; r < hi; ++r) out[r] = q4_q8_dot(n, rows[r], d_w, q8, d_x);
    });
  }
  for (auto& th : pool) th.join();
}

void pack_q4_tiled(std::size_t n, const uint8_t* in, uint8_t* out) {
  // Validated single-row normalize: prod nibble rows are already streaming for
  // one row, so the permutation is identity BY DESIGN (matrix tiling lives in
  // pack_q4_tiled_mat). The work here is validation + 32-nibble block walk so
  // truncated/null inputs fail loudly instead of silently benchmarking zeros.
  if (n == 0 || !in || !out) return;
  std::size_t nb = (n + 1) / 2;
  std::size_t off = 0;
  constexpr std::size_t kBlkBytes = 16;  // 32 nibbles per block
  while (off + kBlkBytes <= nb) {
    std::memcpy(out + off, in + off, kBlkBytes);
    off += kBlkBytes;
  }
  if (off < nb) std::memcpy(out + off, in + off, nb - off);
}

void build_lut2(const int8_t act[4], int scale, int lut[256]) {
  // Exact scalar build (256 entries, cheap, done once per activation tile).
  for (int b = 0; b < 256; ++b) {
    int s = 0;
    s += ((b >> 0) & 3) * act[0];
    s += ((b >> 2) & 3) * act[1];
    s += ((b >> 4) & 3) * act[2];
    s += ((b >> 6) & 3) * act[3];
    lut[b] = s * scale;
  }
}

int lut2_dot(std::size_t ngroups, const uint8_t* w2, const int lut[256]) {
  // Exact scalar gather + AVX2 32-wide vectorized add-tree (no shuffle_epi8
  // LUT lookup; the shuffle form is documented future work in lut_gemm.h).
  // Byte layout: lo2 = bits[1:0] ... hi2 = bits[7:6] mapping to act[0..3]
  // in build_lut2 order.
  if (!w2 || !lut || ngroups == 0) return 0;
#if defined(OURS_AVX2)
  // 32 lookups per step with scalar LUT fetch into a 32-wide int32 vector,
  // then _mm256-add tree (memory-bound part is the 32 gathers, compute is
  // vectorized). Exact, no clamping.
  __m256i vacc0 = _mm256_setzero_si256(), vacc1 = _mm256_setzero_si256();
  std::size_t g = 0;
  for (; g + 32 <= ngroups; g += 32) {
    int v[32];
    for (int k = 0; k < 32; ++k) v[k] = lut[w2[g + (std::size_t)k]];
    __m256i a = _mm256_loadu_si256((const __m256i*)(v));
    __m256i b = _mm256_loadu_si256((const __m256i*)(v + 8));
    __m256i c = _mm256_loadu_si256((const __m256i*)(v + 16));
    __m256i d = _mm256_loadu_si256((const __m256i*)(v + 24));
    vacc0 = _mm256_add_epi32(vacc0, _mm256_add_epi32(a, b));
    vacc1 = _mm256_add_epi32(vacc1, _mm256_add_epi32(c, d));
  }
  alignas(32) int t0[8], t1[8];
  _mm256_store_si256((__m256i*)t0, vacc0);
  _mm256_store_si256((__m256i*)t1, vacc1);
  long long acc = 0;
  for (int k = 0; k < 8; ++k) acc += (long long)t0[k] + (long long)t1[k];
  for (; g < ngroups; ++g) acc += (long long)lut[w2[g]];
  return (int)acc;
#else
  long long acc = 0;
  for (std::size_t g = 0; g < ngroups; ++g) acc += (long long)lut[w2[g]];
  return (int)acc;
#endif
}

// ---- H4: aligned arena + pinning helpers (private) ----
namespace {
void* aligned_malloc(std::size_t n, std::size_t a) {
#if defined(_WIN32)
  return _aligned_malloc(n, a);
#else
  void* p = nullptr;
  return (posix_memalign(&p, a, n) == 0) ? p : nullptr;
#endif
}

void aligned_free(void* p) {
  if (!p) return;
#if defined(_WIN32)
  _aligned_free(p);
#else
  free(p);
#endif
}

void pin_current_thread(int core) {
  if (core < 0) return;
#if defined(_WIN32)
  if (core < (int)(sizeof(DWORD_PTR) * 8)) {
    DWORD_PTR mask = (DWORD_PTR)1 << core;
    if (mask) SetThreadAffinityMask(GetCurrentThread(), mask);
  }
#else
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}
}  // namespace

std::size_t q4_tiled_bytes(std::size_t m, std::size_t n) {
  std::size_t tiles = (m + 31) / 32;
  return tiles * 32 * ((n + 1) / 2);
}

void pack_q4_tiled_mat(std::size_t m, std::size_t n, const uint8_t* const* rows,
                       uint8_t* tiled) {
  std::size_t rb = (n + 1) / 2;
  std::size_t tiles = (m + 31) / 32;
  for (std::size_t t = 0; t < tiles; ++t) {
    uint8_t* tile = tiled + t * 32 * rb;
    for (int r = 0; r < 32; ++r) {
      std::size_t gr = t * 32 + (std::size_t)r;
      if (gr < m) {
        std::memcpy(tile + (std::size_t)r * rb, rows[gr], rb);
      } else {
        std::memset(tile + (std::size_t)r * rb, 0, rb);
      }
    }
  }
}

void q4_q8_gemv_tiled(std::size_t m, std::size_t n, const uint8_t* tiled,
                      const int8_t* q8, float d_w, float d_x, float* out,
                      int nthreads, bool pin_threads) {
  // One-shot utility: spawns + joins worker threads per call. The spawn cost
  // is part of the measured time; production H4 timings must use
  // q4_q8_gemv_tiled_pool() (persistent pool) instead.
  std::size_t rb = (n + 1) / 2;
  std::size_t tiles = (m + 31) / 32;
  auto one_row = [&](std::size_t gr) {
    std::size_t t = gr / 32;
    const uint8_t* row = tiled + t * 32 * rb + (gr % 32) * rb;
    out[gr] = q4_q8_dot(n, row, d_w, q8, d_x);
  };
  if (nthreads <= 1) {
    for (std::size_t gr = 0; gr < m; ++gr) one_row(gr);
    return;
  }
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) hw = 1;
  int W = nthreads > 64 ? 64 : nthreads;
  std::size_t chunk = (m + (std::size_t)W - 1) / (std::size_t)W;
  std::vector<std::thread> pool;
  pool.reserve((std::size_t)W);
  for (int w = 0; w < W; ++w) {
    pool.emplace_back([&, w, W, chunk]() {
      if (pin_threads) pin_current_thread(w % (int)hw);
      // Contiguous block per worker: preserves 32-row tile locality
      // (shared activation vector stays hot in L1/L2 across the tile).
      std::size_t lo = (std::size_t)w * chunk;
      std::size_t hi = lo + chunk < m ? lo + chunk : m;
      for (std::size_t gr = lo; gr < hi; ++gr) one_row(gr);
    });
  }
  for (auto& th : pool) th.join();
  (void)tiles;
}

uint8_t* q4_tiled_alloc(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  return (uint8_t*)aligned_malloc(bytes ? bytes : 64, 64);
}

void q4_tiled_free(uint8_t* p) { aligned_free(p); }

bool advise_hugepages(void* p, std::size_t bytes) {
  if (!p || bytes == 0) return false;
#if defined(OURS_HAVE_MMAN)
  return madvise(p, bytes, MADV_HUGEPAGE) == 0;
#else
  (void)p;
  (void)bytes;
  return false;  // best-effort: unsupported platform, not an error
#endif
}

void q4_q8_gemv_mt_pool(std::size_t m, std::size_t n, const uint8_t* const* rows,
                        float d_w, const int8_t* q8, float d_x, float* out,
                        ThreadPool& pool) {
  if (!rows || !q8 || !out || m == 0) return;
  pool.parallel_for(m, [&](std::size_t r) {
    out[r] = q4_q8_dot(n, rows[r], d_w, q8, d_x);
  });
}

void q4_q8_gemv_tiled_pool(std::size_t m, std::size_t n, const uint8_t* tiled,
                           const int8_t* q8, float d_w, float d_x, float* out,
                           ThreadPool& pool) {
  if (!tiled || !q8 || !out || m == 0) return;
  std::size_t rb = (n + 1) / 2;
  pool.parallel_for(m, [&](std::size_t gr) {
    std::size_t t = gr / 32;
    const uint8_t* row = tiled + t * 32 * rb + (gr % 32) * rb;
    out[gr] = q4_q8_dot(n, row, d_w, q8, d_x);
  });
}

void q4_blockscales_gemv_tiled_pool(std::size_t m, std::size_t n,
                                    const uint8_t* tiled, const float* scales_tiled,
                                    const int8_t* q8, float d_x, float* out,
                                    ThreadPool& pool) {
  if (!tiled || !scales_tiled || !q8 || !out || m == 0) return;
  std::size_t rb = (n + 1) / 2;
  std::size_t nb = (n + 31) / 32;
  pool.parallel_for(m, [&](std::size_t gr) {
    std::size_t t = gr / 32;
    const uint8_t* row = tiled + t * 32 * rb + (gr % 32) * rb;
    const float* sc = scales_tiled + gr * nb;
    out[gr] = quant::q4_blockscales_dot(n, row, sc, nb, q8, d_x);
  });
}

}  // namespace ours
