// hlq_tmac.h — ELUTQ-style CPU path (Appendix B/D + H.1, arXiv:2510.19482).
// Drop-in GGUF Q4, no requant. Speedup claims need same-box measurement
// vs llama.cpp Q4 (same threads/file).
#pragma once
#include <cstddef>
#include <cstdint>

namespace ours {
namespace hlq {

// ---- D.2: HLQ -> bipolar (exactly Appendix D.2) ----
// HLQ: What = sum_j s_j * b_j + z, b_j in {0,1}.
// Bipolar: What = sum_j shat_j * bhat_j + zhat, bhat_j in {-1,1},
//   shat_j = 0.5 * s_j, zhat = z + 0.5 * sum_j s_j, bhat = 2*b - 1.
void hlq_to_bipolar(const float* s, int q, float z, float* shat, float* zhat);

// ---- D.1: plane-interleaved rearrange (Fig.13 out-channel contiguous) ----
// Transpose across the q bit-planes, done offline:
//   out[g*q+j] = planes[j][g]   (g = group, j = plane)
void pack_planes_interleaved(const uint8_t* const* planes, int q, uint8_t* out,
                             std::size_t ngroups);
void unpack_planes_interleaved(const uint8_t* in, uint8_t** planes, int q,
                               std::size_t ngroups);
// GEMV directly on the interleaved layout (no per-plane streams).
float bitserial_gemv_interleaved(const uint8_t* packed, int q, std::size_t ngroups,
                                 const float* scales_pg, float z,
                                 const float* act_blocks);
// Nibble expand of one Fig.13 plane: packed bytes hold 2 groups each
// (even index = high nibble, odd = low). AVX2/NEON where available.
void unpack_nibbles(const uint8_t* packed, uint8_t* nibbles, std::size_t n);
// Same as unpack_nibbles (legacy name).
void unpack_nibbles_avx2(const uint8_t* packed, uint8_t* nibbles, std::size_t n);
// ---- Bit-packed disk format (written by export_hlq_cpu.py planes.bin) ----
// planes.bin holds q bit-planes LSB-first, 1 bit per group: plane j occupies
// bytes [j*plane_bytes,(j+1)*plane_bytes) with plane_bytes=(ngroups+7)/8.
// unpack_bitplane_lsb8 expands one plane to ngroups group bytes (0/1);
// bitserial_gemv_tmac_packed runs the per-group path straight from the
// packed file image (unpack-on-the-fly).
std::size_t bitplane_packed_bytes(std::size_t ngroups);
void unpack_bitplane_lsb8(const uint8_t* packed, uint8_t* groups, std::size_t ngroups);
float bitserial_gemv_tmac_packed(const uint8_t* packed_planes, int q, std::size_t ngroups,
                                 const float* scales_pg, float z,
                                 const float* act_blocks);

// ---- Appendix B + D.3: LUT g=4 (16 entries) + mirror compress -> 8 ----
// act[4]: 4 activations. lut16[16]: dot(act, 4-bit combos), bits in {0,1}.
// Mirror is the complement lut16[15-i] = S - lut16[i], S = lut16[15];
// the negation form -lut[15-i] holds only for bipolar {-1,+1} LUTs
// (after hlq_to_bipolar, Appendix D.2), not for the {0,1} LUT here.
void build_lut_g4(const float act[4], float scale, float lut16[16]);
void mirror_compress16to8(const float lut16[16], float lut8[8]);
// Complement-mirror lookup: idx<8 -> lut8[idx]; idx>=8 -> S - lut8[15-idx].
// S must be the group's full sum lut16[15] (== sum(act)*scale).
float lut_lookup_mirror8(const float lut8[8], float S, int idx4);

// ---- Bit-serial GEMV over q planes (CPU, modified T-MAC style, H.1) ----
// wplanes: q 1-bit plane pointers (ngroups bytes each).
// scales_global[q]: ONE scale per plane. ELUTQ Tables 7-8/12-14 use per-group
// scales g=128 instead; for those use bitserial_gemv_tmac_pergroup() below.
float bitserial_gemv_ref(const uint8_t* const* wplanes, int q,
                         std::size_t ngroups, const float* scales_global, float z,
                         const float* act_table, const float* lut_prototype);
// Per-group path: scales_pg[j*ngroups+g] is the scale of plane j, group g
// (g=128 grouping from the quantizer).
float bitserial_gemv_tmac_pergroup(const uint8_t* const* wplanes, int q,
                                   std::size_t ngroups, const float* scales_pg, float z,
                                   const float* act_blocks);
float bitserial_gemv_tmac(const uint8_t* const* wplanes, int q,
                          std::size_t ngroups, const float* scales_global, float z,
                          const float* act_blocks);

// ---- D.3 Table Quant: logical FP16 -> int8 (bandwidth-saving option) ----
void quant_table_fp16_to_int8(const float* in, int8_t* out, std::size_t n, float* scale_out);
void dequant_table_int8_to_fp16(const int8_t* in, float* out, std::size_t n, float scale);

}  // namespace hlq
}  // namespace ours
