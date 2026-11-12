// lut_gemm.h — H1: INT4(Q4) x INT8(Q8) dot/GEMV without FP dequantization.
//
// Affine quantization (Jacob et al., CVPR 2018, arXiv:1712.05877, Eq.1):
//   dot = d_w * d_x * ( SUM(w_nib * x) - 8 * SUM(x) ).
// The zero-point correction is hoisted outside the hot loop; the inner loop
// is pure integer math with a single FP scale at the end (cf. LUT-GEMM,
// Park et al., ICLR 2024, arXiv:2206.09557; T-MAC, Wei et al., EuroSys 2025,
// arXiv:2407.00088). AVX2 uses _mm256_maddubs_epi16 + _mm256_madd_epi16
// (cf. gemmlowp, QNNPACK, llama.cpp GGML Q4 kernels).
//
// SATURATION LIMIT: maddubs saturates per pair to [-32768,32767]. Safe for
// |x| <= 100 and n <= 4096 per 32-bit partial accumulator. The bench asserts
// checksum against the FP reference; on mismatch the kernel is wrong.
#pragma once
#include <cstddef>
#include <cstdint>

namespace ours {

// FP reference (ground truth). Used by bench to assert equality.
float q4_q8_dot_ref(std::size_t n, const uint8_t* q4, float d_w,
                    const int8_t* q8, float d_x);

// Fast path: full integer, AVX2 when available, scalar fallback.
// q4: nibble-packed, 2 weights per byte, n even.
float q4_q8_dot(std::size_t n, const uint8_t* q4, float d_w,
                const int8_t* q8, float d_x);

// GEMV: m rows x n cols. out[m] = d_w*d_x*(...). rows: per-row pointers.
// Serial dispatch (correctness baseline for checksums).
void q4_q8_gemv(std::size_t m, std::size_t n, const uint8_t* const* rows,
                float d_w, const int8_t* q8, float d_x, float* out);
// Threaded dispatch over rows (one-shot: spawns + joins threads per call;
// the spawn cost is included in measured time). Timed runs must use
// q4_q8_gemv_mt_pool() (persistent pool) instead.
// nthreads<=1 falls back to the serial version above.
void q4_q8_gemv_mt(std::size_t m, std::size_t n, const uint8_t* const* rows,
                   float d_w, const int8_t* q8, float d_x, float* out,
                   int nthreads, bool pin_threads);

// H4 tiled layout: matrix m rows x n cols stored as tiles of 32 rows.
// Each tile holds 32 rows contiguously, each row (n+1)/2 bytes.
// Tiled bytes (padded last tile): tiled_bytes(m, n). Buffers should be
// 64B-aligned; correctness never depends on alignment.
// Rationale (Goto & van de Geijn, 2008, cache blocking): grouping rows in
// tiles keeps the shared activation vector hot in L1/L2 across the tile.
std::size_t q4_tiled_bytes(std::size_t m, std::size_t n);
void pack_q4_tiled_mat(std::size_t m, std::size_t n, const uint8_t* const* rows,
                       uint8_t* tiled);
void q4_q8_gemv_tiled(std::size_t m, std::size_t n, const uint8_t* tiled,
                      const int8_t* q8, float d_w, float d_x, float* out,
                      int nthreads, bool pin_threads);

// Single-row layout normalize: prod nibble rows are already streaming
// for one row; matrix-level tiling lives in pack_q4_tiled_mat() below.
// Validated copy (null/size checks, 32-nibble block walk); never changes
// the permutation, hence never breaks q4_q8_dot.
void pack_q4_tiled(std::size_t n, const uint8_t* in, uint8_t* out);

// 2-bit path a la T-MAC (native-ternary / 1.58-bit models):
// each byte holds 4 2-bit weights (values 0..3, meaning via per-group scale/zp).
// A 256-entry LUT is built from 4 int8 activations, then dot = sum of lookups.
// lut2_dot: exact scalar gather + AVX2 32-wide vectorized add-tree (no
// shuffle_epi8 LUT lookup is claimed; the shuffle form is future work).
// NOT used by the Q4 PKL model path (kept for ternary checkpoints).
void build_lut2(const int8_t act[4], int scale, int lut[256]);
int lut2_dot(std::size_t ngroups, const uint8_t* w2, const int lut[256]);

// Persistent-pool dispatches: same math as the ephemeral _mt/_tiled variants
// but reusing a pinned ThreadPool instead of spawning threads per call.
class ThreadPool;
void q4_q8_gemv_mt_pool(std::size_t m, std::size_t n, const uint8_t* const* rows,
                        float d_w, const int8_t* q8, float d_x, float* out,
                        ThreadPool& pool);
void q4_q8_gemv_tiled_pool(std::size_t m, std::size_t n, const uint8_t* tiled,
                           const int8_t* q8, float d_w, float d_x, float* out,
                           ThreadPool& pool);
// Tiled per-block-scales GEMV: tiled nibbles as in pack_q4_tiled_mat +
// scales packed row-major (scales_tiled[r*nb+b], nb=(n+31)/32).
void q4_blockscales_gemv_tiled_pool(std::size_t m, std::size_t n,
                                    const uint8_t* tiled, const float* scales_tiled,
                                    const int8_t* q8, float d_x, float* out,
                                    ThreadPool& pool);
// 64B-aligned tiled arena. Correctness never depends on alignment;
// alignment only helps vector loads.
uint8_t* q4_tiled_alloc(std::size_t bytes);
void q4_tiled_free(uint8_t* p);
// Best-effort huge-page advice for a tiled arena (madvise/MADV_HUGEPAGE on
// Linux, no-op elsewhere or on failure).
bool advise_hugepages(void* p, std::size_t bytes);

}  // namespace ours
