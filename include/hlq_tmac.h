// hlq_tmac.h — ELUTQ-style CPU path (Appendix D + H.1 + Fig.13).
//
// Design follows ELUTQ paper v4 (arXiv:2510.19482):
//   Appendix D.1 Weight-Rearrange (1-bit 16x8 blocks -> contiguous out-channel,
//             Fig.13 NEON 128-bit, decode AND 0x0F / SHR>>4)
//   Appendix D.2 Mirror Storage (bipolar {-1,1} via shat/zhat, store 1/2 table)
//   Appendix D.3 Table Quantization (FP16 split into 2x int8 + vld2q_u8, or int8)
//   Appendix B bit-serial LUT-GEMM (q 1-bit planes, 2^g table, g=4 -> 16 entries)
//   H.1 modified T-MAC operators (AVX2/NEON), Table 11 GEMV, Tables 12-14 vs llama.cpp
//   ELUTQ repo flow: HLQ-GPTQ / block_ap / e2e_qp / export_bcq -> here export_hlq_cpu.py
//
// Speedup claims are valid only after ours gemm/kv on the defense machine
// vs llama.cpp Q4 on the same threads/GGUF file (H1/H4/H5).
// Novelty vs ELUTQ is documented in engine/NOVELTY.md (drop-in without requant +
// integer-dot + KV-Q8 + n-gram spec + tiled/pinning/aligned-arena).
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
// Genuine transpose across the q bit-planes, done offline:
//   out[g*q+j] = planes[j][g]   (g = group, j = plane)
// One sequential stream replaces q strided streams at decode (locality win,
// the same reason Fig.13 makes 128-bit loads contiguous). Inverse provided
// for roundtrip tests; the production decoder reads the interleaved stream
// directly (bitserial_gemv_interleaved). Replaces the former identity copy.
void pack_planes_interleaved(const uint8_t* const* planes, int q, uint8_t* out,
                             std::size_t ngroups);
void unpack_planes_interleaved(const uint8_t* in, uint8_t** planes, int q,
                               std::size_t ngroups);
// GEMV directly on the interleaved layout (no per-plane streams).
float bitserial_gemv_interleaved(const uint8_t* packed, int q, std::size_t ngroups,
                                 const float* scales_pg, float z,
                                 const float* act_blocks);
// Nibble expand of one Fig.13 plane: packed bytes hold 2 groups each
// (even index = high nibble (byte>>4), odd = low (byte&0x0F)). AVX2 32-at-a-time
// (and NEON 16-at-a-time where available), scalar tail.
void unpack_nibbles(const uint8_t* packed, uint8_t* nibbles, std::size_t n);
// Legacy name kept for compat; identical to unpack_nibbles.
void unpack_nibbles_avx2(const uint8_t* packed, uint8_t* nibbles, std::size_t n);
// ---- Bit-packed disk format (written by export_hlq_cpu.py planes.bin) ----
// planes.bin holds q bit-planes LSB-first, 1 bit per group: plane j occupies
// bytes [j*plane_bytes,(j+1)*plane_bytes) with plane_bytes=(ngroups+7)/8.
// unpack_bitplane_lsb8 expands one plane to ngroups group bytes (0/1 in the
// low nibble); bitserial_gemv_tmac_packed runs the full per-group path
// straight from the packed file image (unpack-on-the-fly, no temp files).
std::size_t bitplane_packed_bytes(std::size_t ngroups);
void unpack_bitplane_lsb8(const uint8_t* packed, uint8_t* groups, std::size_t ngroups);
float bitserial_gemv_tmac_packed(const uint8_t* packed_planes, int q, std::size_t ngroups,
                                 const float* scales_pg, float z,
                                 const float* act_blocks);

// ---- Appendix B + D.3: LUT g=4 (16 entries) + mirror compress -> 8 ----
// act[4]: 4 activations (logically FP16; float prototype here).
// lut16[16]: dot(act, 4-bit combos), bits in {0,1}. lut8[8]: half table,
// lossless ONLY under the complement rule lut16[15-i] = S - lut16[i] with
// S = lut16[15] = sum(act)*scale. (Negation mirror -lut[15-i] is exact only
// for bipolar {-1,+1} LUTs after hlq_to_bipolar, Appendix D.2 — NOT for the
// {0,1} LUT built here. Fixed 2026-11-12: harness HLQ-FAIL 3x off proved the
// negation form wrong on {0,1} bits.)
// scale: integer prototype multiplier (production: per-group FP16 scales).
void build_lut_g4(const float act[4], float scale, float lut16[16]);
void mirror_compress16to8(const float lut16[16], float lut8[8]);
// Complement-mirror lookup: idx<8 -> lut8[idx]; idx>=8 -> S - lut8[15-idx].
// S must be the group's full sum lut16[15] (== sum(act)*scale).
float lut_lookup_mirror8(const float lut8[8], float S, int idx4);

// ---- Bit-serial GEMV over q planes (CPU, modified T-MAC style, H.1) ----
// wplanes: q 1-bit plane pointers (ngroups bytes each).
// scales_global[q]: ONE scale per plane (prototype global-scale path).
//   LIMITATION: ELUTQ Tables 7-8/12-14 use per-group scales g=128; the global
//   path is an upper-bound reference, NOT ELUTQ parity. Full per-group path is
//   bitserial_gemv_tmac_pergroup() (scales_pg[j*ngroups+g]). z: global zero-point.
float bitserial_gemv_ref(const uint8_t* const* wplanes, int q,
                         std::size_t ngroups, const float* scales_global, float z,
                         const float* act_table, const float* lut_prototype);
// Full per-group path: scales_pg[j*ngroups+g] is the scale of plane j,
// group g (g=128 grouping from the quantizer). No global-scale approximation.
float bitserial_gemv_tmac_pergroup(const uint8_t* const* wplanes, int q,
                                   std::size_t ngroups, const float* scales_pg, float z,
                                   const float* act_blocks);
float bitserial_gemv_tmac(const uint8_t* const* wplanes, int q,
                          std::size_t ngroups, const float* scales_global, float z,
                          const float* act_blocks);

// ---- D.3 Table Quant: logical FP16 -> int8 (bandwidth-saving option) ----
// Accuracy may drop; bench H1 must assert checksums and report the delta.
// NEON path (where available) uses plain vld1/vst1 + vector arithmetic;
// no vld2q_u8 transpose is claimed.
void quant_table_fp16_to_int8(const float* in, int8_t* out, std::size_t n, float* scale_out);
void dequant_table_int8_to_fp16(const int8_t* in, float* out, std::size_t n, float scale);

}  // namespace hlq
}  // namespace ours
