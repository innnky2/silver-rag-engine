// hlq_tmac.cpp — ELUTQ-style CPU implementation (Appendix B/D/H.1).
// Reference: arXiv:2510.19482 v4, Appendix B (Fig.9-10), D.1-D.3 (Fig.13),
// H.1 + Tables 11-14, Fig.7 (M2/RK3588, 4 threads, batch 1).
#include "hlq_tmac.h"

#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#define OURS_HLQ_AVX2 1
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define OURS_HLQ_NEON 1
#endif

namespace ours {
namespace hlq {

void hlq_to_bipolar(const float* s, int q, float z, float* shat, float* zhat) {
  // Appendix D.2 exactly: shat_j = s_j/2, zhat = z + sum(s_j)/2.
  double acc = (double)z;
  for (int j = 0; j < q; ++j) {
    shat[j] = s[j] * 0.5f;
    acc += (double)s[j] * 0.5;
  }
  *zhat = (float)acc;
}

void pack_planes_interleaved(const uint8_t* const* planes, int q, uint8_t* out,
                             std::size_t ngroups) {
  // D.1 rearrange: out[g*q+j] = planes[j][g]. Offline.
  if (!planes || !out || q <= 0 || ngroups == 0) return;
  for (std::size_t g = 0; g < ngroups; ++g)
    for (int j = 0; j < q; ++j) out[g * (std::size_t)q + (std::size_t)j] = planes[j][g];
}

void unpack_planes_interleaved(const uint8_t* in, uint8_t** planes, int q,
                               std::size_t ngroups) {
  if (!in || !planes || q <= 0 || ngroups == 0) return;
  for (std::size_t g = 0; g < ngroups; ++g)
    for (int j = 0; j < q; ++j) planes[j][g] = in[g * (std::size_t)q + (std::size_t)j];
}

void unpack_nibbles(const uint8_t* packed, uint8_t* nibbles, std::size_t n) {
  // Fig.13 decode: even index = high nibble (byte>>4), odd = low (byte&0x0F).
  // AVX2: 32 packed bytes (64 nibbles) per step via _mm256_and/_mm256_srli.
  // NEON: 16 packed bytes (32 nibbles) per step via vandq/vshrq + vst2
  // interleave. Scalar tail.
  if (!packed || !nibbles) return;
#if defined(OURS_HLQ_NEON)
  {
    std::size_t npacked = (n + 1) / 2;
    std::size_t i = 0;
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    for (; i + 16 <= npacked; i += 16) {
      uint8x16_t b = vld1q_u8(packed + i);
      uint8x16_t lo = vandq_u8(b, mask);
      uint8x16_t hi = vandq_u8(vshrq_n_u8(b, 4), mask);
      uint8x16x2_t pair;
      pair.val[0] = hi;
      pair.val[1] = lo;
      vst2q_u8(nibbles + i * 2, pair);
    }
    for (; i < npacked; ++i) {
      uint8_t b = packed[i];
      size_t o = i * 2;
      if (o < n) nibbles[o] = (uint8_t)((b >> 4) & 0x0F);
      if (o + 1 < n) nibbles[o + 1] = (uint8_t)(b & 0x0F);
    }
    return;
  }
#endif
#if defined(OURS_HLQ_AVX2)
  std::size_t npacked = (n + 1) / 2;
  std::size_t i = 0;
  const __m256i lo_mask = _mm256_set1_epi8(0x0F);
  for (; i + 32 <= npacked; i += 32) {
    __m256i b = _mm256_loadu_si256((const __m256i*)(packed + i));
    __m256i lo = _mm256_and_si256(b, lo_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), lo_mask);
    // Interleave hi/lo into 64 output nibbles: out[2k]=hi[k], out[2k+1]=lo[k].
    alignas(32) uint8_t lo_b[32], hi_b[32];
    _mm256_store_si256((__m256i*)lo_b, lo);
    _mm256_store_si256((__m256i*)hi_b, hi);
    for (int k = 0; k < 32; ++k) {
      nibbles[(i + (size_t)k) * 2] = hi_b[k];
      nibbles[(i + (size_t)k) * 2 + 1] = lo_b[k];
    }
  }
  for (; i < npacked; ++i) {
    uint8_t b = packed[i];
    size_t o = i * 2;
    if (o < n) nibbles[o] = (uint8_t)((b >> 4) & 0x0F);
    if (o + 1 < n) nibbles[o + 1] = (uint8_t)(b & 0x0F);
  }
#else
  for (std::size_t i = 0; i < n; ++i) {
    uint8_t b = packed[i >> 1];
    nibbles[i] = (i & 1) ? (uint8_t)(b & 0x0F) : (uint8_t)((b >> 4) & 0x0F);
  }
#endif
}

void unpack_nibbles_avx2(const uint8_t* packed, uint8_t* nibbles, std::size_t n) {
  unpack_nibbles(packed, nibbles, n);
}

void build_lut_g4(const float act[4], float scale, float lut16[16]) {
  // Appendix B: precompute dot(act, 4-bit combos), 2^g = 16 entries.
  for (int b = 0; b < 16; ++b) {
    double s = 0.0;
    s += ((b >> 0) & 1) ? (double)act[0] : 0.0;
    s += ((b >> 1) & 1) ? (double)act[1] : 0.0;
    s += ((b >> 2) & 1) ? (double)act[2] : 0.0;
    s += ((b >> 3) & 1) ? (double)act[3] : 0.0;
    lut16[b] = (float)(s * (double)scale);
  }
}

void mirror_compress16to8(const float lut16[16], float lut8[8]) {
  // Appendix D.2 storage trick, {0,1}-bit form: keep lut16[0..7]; the upper
  // Half table stores lut16[0..7]; the upper half is lut16[b]+lut16[15-b] == S.
  // is S - lut8[15-idx] with S = group full-sum (see lut_lookup_mirror8).
  for (int i = 0; i < 8; ++i) lut8[i] = lut16[i];
}

float lut_lookup_mirror8(const float lut8[8], float S, int idx4) {
  int b = idx4 & 0xF;
  return (b < 8) ? lut8[b] : (S - lut8[15 - b]);
}

float bitserial_gemv_ref(const uint8_t* const* wplanes, int q,
                         std::size_t ngroups, const float* scales_global, float z,
                         const float* act_table, const float* lut_prototype) {
  // Full-FP reference (bench H1 ground truth): each 4-bit group -> LUT -> s_j weight.
  (void)lut_prototype;
  double acc = (double)z * (double)ngroups;
  for (int j = 0; j < q; ++j) {
    float lut16[16];
    build_lut_g4(act_table, scales_global[j], lut16);
    for (std::size_t g = 0; g < ngroups; ++g) {
      int idx = (int)(wplanes[j][g] & 0x0F);
      acc += (double)lut16[idx];
    }
  }
  return (float)acc;
}

float bitserial_gemv_tmac_pergroup(const uint8_t* const* wplanes, int q,
                                   std::size_t ngroups, const float* scales_pg, float z,
                                   const float* act_blocks) {
  // Per-group scales: scales_pg[j*ngroups+g]. Mirror lookup is the
  // complement S - lut8[15-idx] for {0,1} bits.
  if (!wplanes || !scales_pg || !act_blocks) return (float)z;
  if (q <= 0 || ngroups == 0) return (float)((double)z * (double)ngroups);
  double acc_scalar_tail = (double)z * (double)ngroups;
#if defined(OURS_HLQ_AVX2)
  __m256 vacc = _mm256_setzero_ps();
  std::size_t g = 0;
  for (; g + 8 <= ngroups; g += 8) {
    for (int j = 0; j < q; ++j) {
      float vbuf[8];
      for (int k = 0; k < 8; ++k) {
        float lut16[16];
        float lut8[8];
        build_lut_g4(act_blocks + (g + (std::size_t)k) * 4, 1.0f, lut16);
        mirror_compress16to8(lut16, lut8);
        const float S = lut16[15];
        int idx = (int)(wplanes[j][g + (std::size_t)k] & 0x0F);
        vbuf[k] = lut_lookup_mirror8(lut8, S, idx);
      }
      __m256 vv = _mm256_loadu_ps(vbuf);
      __m256 sc = _mm256_loadu_ps(scales_pg + (std::size_t)j * ngroups + g);
      vacc = _mm256_fmadd_ps(vv, sc, vacc);
    }
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, vacc);
  double acc = acc_scalar_tail;
  for (int k = 0; k < 8; ++k) acc += (double)tmp[k];
#if defined(OURS_HLQ_NEON)
  // NEON 4-wide tail for the remainder (also the primary SIMD path on ARM,
  // where AVX2 is absent): float32x4_t fmadd over 4 groups.
  {
    float32x4_t vacc4 = vdupq_n_f32(0.0f);
    for (; g + 4 <= ngroups; g += 4) {
      for (int j = 0; j < q; ++j) {
        float vbuf[4];
        for (int k = 0; k < 4; ++k) {
          float lut16[16];
          float lut8[8];
          build_lut_g4(act_blocks + (g + (std::size_t)k) * 4, 1.0f, lut16);
          mirror_compress16to8(lut16, lut8);
          const float S = lut16[15];
          int idx = (int)(wplanes[j][g + (std::size_t)k] & 0x0F);
          vbuf[k] = lut_lookup_mirror8(lut8, S, idx);
        }
        float32x4_t vv = vld1q_f32(vbuf);
        float32x4_t sc = vld1q_f32(scales_pg + (std::size_t)j * ngroups + g);
        vacc4 = vmlaq_f32(vacc4, vv, sc);
      }
    }
    float t4[4];
    vst1q_f32(t4, vacc4);
    for (int k = 0; k < 4; ++k) acc += (double)t4[k];
  }
#endif
  for (; g < ngroups; ++g) {
    float lut16[16];
    float lut8[8];
    build_lut_g4(act_blocks + g * 4, 1.0f, lut16);
    mirror_compress16to8(lut16, lut8);
    const float S = lut16[15];
    for (int j = 0; j < q; ++j) {
      int idx = (int)(wplanes[j][g] & 0x0F);
      float v = lut_lookup_mirror8(lut8, S, idx);
      acc += (double)v * (double)scales_pg[(std::size_t)j * ngroups + g];
    }
  }
  return (float)acc;
#else
  double acc = acc_scalar_tail;
#if defined(OURS_HLQ_NEON)
  {
    float32x4_t vacc4 = vdupq_n_f32(0.0f);
    std::size_t g = 0;
    for (; g + 4 <= ngroups; g += 4) {
      for (int j = 0; j < q; ++j) {
        float vbuf[4];
        for (int k = 0; k < 4; ++k) {
          float lut16[16];
          float lut8[8];
          build_lut_g4(act_blocks + (g + (std::size_t)k) * 4, 1.0f, lut16);
          mirror_compress16to8(lut16, lut8);
          const float S = lut16[15];
          int idx = (int)(wplanes[j][g + (std::size_t)k] & 0x0F);
          vbuf[k] = lut_lookup_mirror8(lut8, S, idx);
        }
        vacc4 = vmlaq_f32(vacc4, vld1q_f32(vbuf),
                          vld1q_f32(scales_pg + (std::size_t)j * ngroups + g));
      }
    }
    float t4[4];
    vst1q_f32(t4, vacc4);
    for (int k = 0; k < 4; ++k) acc += (double)t4[k];
    for (; g < ngroups; ++g) {
      float lut16[16];
      float lut8[8];
      build_lut_g4(act_blocks + g * 4, 1.0f, lut16);
      mirror_compress16to8(lut16, lut8);
      const float S = lut16[15];
      for (int j = 0; j < q; ++j) {
        int idx = (int)(wplanes[j][g] & 0x0F);
        float v = lut_lookup_mirror8(lut8, S, idx);
        acc += (double)v * (double)scales_pg[(std::size_t)j * ngroups + g];
      }
    }
    return (float)acc;
  }
#endif
  for (std::size_t g = 0; g < ngroups; ++g) {
    float lut16[16];
    float lut8[8];
    build_lut_g4(act_blocks + g * 4, 1.0f, lut16);
    mirror_compress16to8(lut16, lut8);
    const float S = lut16[15];
    for (int j = 0; j < q; ++j) {
      int idx = (int)(wplanes[j][g] & 0x0F);
      float v = lut_lookup_mirror8(lut8, S, idx);
      acc += (double)v * (double)scales_pg[(std::size_t)j * ngroups + g];
    }
  }
  return (float)acc;
#endif
}

float bitserial_gemv_tmac(const uint8_t* const* wplanes, int q,
                          std::size_t ngroups, const float* scales_global, float z,
                          const float* act_blocks) {
  // Global-scale wrapper: broadcasts one scale per plane, then calls
  // the per-group path.
  if (!wplanes || !scales_global || !act_blocks) return (float)z;
  if (q <= 0 || ngroups == 0) return (float)((double)z * (double)ngroups);
  std::vector<float> pg((std::size_t)(q > 0 ? q : 0) * ngroups, 0.0f);
  for (int j = 0; j < q; ++j)
    for (std::size_t g = 0; g < ngroups; ++g)
      pg[(std::size_t)j * ngroups + g] = scales_global[j];
  return bitserial_gemv_tmac_pergroup(wplanes, q, ngroups, pg.data(), z, act_blocks);
}

std::size_t bitplane_packed_bytes(std::size_t ngroups) { return (ngroups + 7) / 8; }

void unpack_bitplane_lsb8(const uint8_t* packed, uint8_t* groups, std::size_t ngroups) {
  // Disk format (export_hlq_cpu.py planes.bin): 1 bit per group, LSB-first.
  if (!packed || !groups) return;
  for (std::size_t g = 0; g < ngroups; ++g)
    groups[g] = (uint8_t)((packed[g >> 3] >> (g & 7)) & 1u);
}

float bitserial_gemv_tmac_packed(const uint8_t* packed_planes, int q, std::size_t ngroups,
                                 const float* scales_pg, float z,
                                 const float* act_blocks) {
  // Consumes the packed file image directly: unpack-on-the-fly into group
  // bytes, then the exact per-group path. No temp files, same numerics as
  // bitserial_gemv_tmac_pergroup (verified by tests/test_hlq.cpp).
  if (!packed_planes || !scales_pg || !act_blocks) return (float)z;
  if (q <= 0 || ngroups == 0) return (float)((double)z * (double)ngroups);
  const std::size_t pb = bitplane_packed_bytes(ngroups);
  std::vector<const uint8_t*> planes((std::size_t)(q > 0 ? q : 0));
  std::vector<uint8_t> tmp((std::size_t)(q > 0 ? q : 0) * ngroups);
  for (int j = 0; j < q; ++j) {
    uint8_t* gj = tmp.data() + (std::size_t)j * ngroups;
    unpack_bitplane_lsb8(packed_planes + (std::size_t)j * pb, gj, ngroups);
    planes[(std::size_t)j] = gj;
  }
  return bitserial_gemv_tmac_pergroup(planes.data(), q, ngroups, scales_pg, z, act_blocks);
}

float bitserial_gemv_interleaved(const uint8_t* packed, int q, std::size_t ngroups,
                                 const float* scales_pg, float z,
                                 const float* act_blocks) {
  // Decode of pack_planes_interleaved (packed[g*q+j] = plane j group g):
  // same math as the per-group path, one sequential stream.
  if (!packed || !scales_pg || !act_blocks) return (float)z;
  if (q <= 0 || ngroups == 0) return (float)((double)z * (double)ngroups);
  double acc = (double)z * (double)ngroups;
  for (std::size_t g = 0; g < ngroups; ++g) {
    float lut16[16];
    float lut8[8];
    build_lut_g4(act_blocks + g * 4, 1.0f, lut16);
    mirror_compress16to8(lut16, lut8);
    const float S = lut16[15];
    for (int j = 0; j < q; ++j) {
      int idx = (int)(packed[g * (std::size_t)q + (std::size_t)j] & 0x0F);
      float v = lut_lookup_mirror8(lut8, S, idx);
      acc += (double)v * (double)scales_pg[(std::size_t)j * ngroups + g];
    }
  }
  return (float)acc;
}

void quant_table_fp16_to_int8(const float* in, int8_t* out, std::size_t n, float* scale_out) {
  // D.3 int8 option: saves bandwidth, may degrade accuracy. Bench must report delta.
  float mx = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    float a = in[i] < 0 ? -in[i] : in[i];
    if (a > mx) mx = a;
  }
  float sc = mx > 0 ? mx / 127.0f : 1.0f;
  for (std::size_t i = 0; i < n; ++i) {
    int v = (int)(in[i] / sc + (in[i] >= 0 ? 0.5f : -0.5f));
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    out[i] = (int8_t)v;
  }
  *scale_out = sc;
}

void dequant_table_int8_to_fp16(const int8_t* in, float* out, std::size_t n, float scale) {
  for (std::size_t i = 0; i < n; ++i) out[i] = (float)in[i] * scale;
}

}  // namespace hlq
}  // namespace ours
