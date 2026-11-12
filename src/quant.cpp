// quant.cpp — PROD dequant for F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 + K-quants.
#include "quant.h"

#include <cmath>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ours {
namespace quant {

float fp16_to_fp32(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
  uint32_t f;
  if (e == 0) {
    if (m == 0) f = s << 31;
    else {
      e = 1;
      while (!(m & 0x400)) { m <<= 1; --e; }
      m &= 0x3FF;
      f = (s << 31) | ((e + 112) << 23) | (m << 13);
    }
  } else if (e == 31) {
    f = (s << 31) | (0xFF << 23) | (m << 13);
  } else {
    f = (s << 31) | ((e + 112) << 23) | (m << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

namespace {
// ---- plain blocks ----
inline void deq_q4_0_blk(const uint8_t* b, float* y) {
  // GGML-exact (ggml-quants.c dequantize_row_q4_0): split-halves nibbles.
  // Byte j holds element j (low nibble) and element j+16 (high nibble).
  uint16_t h; std::memcpy(&h, b, 2);
  float d = fp16_to_fp32(h);
  const uint8_t* qs = b + 2;
  for (int j = 0; j < 16; ++j) {
    y[j]      = d * ((int)(qs[j] & 0x0F) - 8);
    y[j + 16] = d * ((int)(qs[j] >> 4) - 8);
  }
}
inline void deq_q4_1_blk(const uint8_t* b, float* y) {
  // GGML-exact (dequantize_row_q4_1): split-halves, block = d(2)+m(2)+16 = 20B.
  uint16_t hd, hm; std::memcpy(&hd, b, 2); std::memcpy(&hm, b + 2, 2);
  float d = fp16_to_fp32(hd), m = fp16_to_fp32(hm);
  const uint8_t* qs = b + 4;
  for (int j = 0; j < 16; ++j) {
    y[j]      = d * (int)(qs[j] & 0x0F) + m;
    y[j + 16] = d * (int)(qs[j] >> 4) + m;
  }
}
inline void deq_q5_0_blk(const uint8_t* b, float* y) {
  // GGML-exact (dequantize_row_q5_0): split-halves; qh bit j -> element j
  // (high bit), qh bit j+16 -> element j+16. Block = d(2)+qh(4)+16 = 22B.
  uint16_t h; std::memcpy(&h, b, 2);
  float d = fp16_to_fp32(h);
  uint32_t qh; std::memcpy(&qh, b + 2, 4);
  const uint8_t* qs = b + 6;
  for (int j = 0; j < 16; ++j) {
    int h0 = (qh >> j) & 1, h1 = (qh >> (j + 16)) & 1;
    y[j]      = d * ((int)((qs[j] & 0x0F) | (h0 << 4)) - 16);
    y[j + 16] = d * ((int)((qs[j] >> 4) | (h1 << 4)) - 16);
  }
}
inline void deq_q5_1_blk(const uint8_t* b, float* y) {
  // GGML-exact (dequantize_row_q5_1): split-halves + split qh bits.
  // Block = d(2)+m(2)+qh(4)+16 = 24B.
  uint16_t hd, hm; std::memcpy(&hd, b, 2); std::memcpy(&hm, b + 2, 2);
  float d = fp16_to_fp32(hd), m = fp16_to_fp32(hm);
  uint32_t qh; std::memcpy(&qh, b + 4, 4);
  const uint8_t* qs = b + 8;
  for (int j = 0; j < 16; ++j) {
    int h0 = (qh >> j) & 1, h1 = (qh >> (j + 16)) & 1;
    y[j]      = d * (int)((qs[j] & 0x0F) | (h0 << 4)) + m;
    y[j + 16] = d * (int)((qs[j] >> 4) | (h1 << 4)) + m;
  }
}
inline void deq_q8_0_blk(const uint8_t* b, float* y) {
  uint16_t h; std::memcpy(&h, b, 2);
  float d = fp16_to_fp32(h);
  const int8_t* qs = (const int8_t*)(b + 2);
  for (int i = 0; i < 32; ++i) y[i] = d * (float)qs[i];
}
inline void deq_q8_1_blk(const uint8_t* b, float* y) {
  // GGML Q8_1: fp16 d, fp16 s(sum), 32 x int8.
  uint16_t hd, hs; std::memcpy(&hd, b, 2); std::memcpy(&hs, b + 2, 2);
  float d = fp16_to_fp32(hd), s = fp16_to_fp32(hs);
  const int8_t* qs = (const int8_t*)(b + 4);
  for (int i = 0; i < 32; ++i) y[i] = d * (float)qs[i] + s;
}
// ---- K-quants: super-block 256 ----
inline void get_scale_min_k4(int j, const uint8_t* sc, uint8_t& d, uint8_t& m) {
  if (j < 4) { d = sc[j] & 63; m = sc[j + 8] & 63; }
  else { d = (uint8_t)((sc[j + 8] & 0xF) | ((sc[j - 4] >> 6) << 4)); m = (uint8_t)((sc[j + 8] >> 4) | ((sc[j] >> 6) << 4)); }
}
inline void deq_q4_K_blk(const uint8_t* b, float* y) {
  // GGML-exact (dequantize_row_q4_K): per 64-elem chunk c, two (sc,mn) pairs
  // via get_scale_min_k4(2c),(2c+1); first 32 outs = LOW nibbles of qs[0..31]
  // with (d1,m1), next 32 = HIGH nibbles with (d2,m2).
  uint16_t hd, hmin; std::memcpy(&hd, b, 2); std::memcpy(&hmin, b + 2, 2);
  float dall = fp16_to_fp32(hd), dmin = fp16_to_fp32(hmin);
  const uint8_t* scales = b + 4;
  const uint8_t* qs = b + 16;
  for (int c = 0; c < 4; ++c) {
    uint8_t sc, mn; get_scale_min_k4(2 * c, scales, sc, mn);
    float d1 = dall * sc, m1 = dmin * mn;
    get_scale_min_k4(2 * c + 1, scales, sc, mn);
    float d2 = dall * sc, m2 = dmin * mn;
    for (int l = 0; l < 32; ++l) {
      y[c * 64 + l]      = d1 * (float)(qs[c * 32 + l] & 0x0F) - m1;
      y[c * 64 + 32 + l] = d2 * (float)(qs[c * 32 + l] >> 4) - m2;
    }
  }
}
inline void deq_q5_K_blk(const uint8_t* b, float* y) {
  // GGML-exact (dequantize_row_q5_K): block = d(2)+dmin(2)+scales(12)+qs(128)
  // +qh(32) = 176B. Per 64-chunk c: (d1,m1),(d2,m2); low 32 = ql low nibbles
  // + qh bit (2c) of qh[0..31]; high 32 = ql high nibbles + qh bit (2c+1).
  uint16_t hd, hmin; std::memcpy(&hd, b, 2); std::memcpy(&hmin, b + 2, 2);
  float dall = fp16_to_fp32(hd), dmin = fp16_to_fp32(hmin);
  const uint8_t* scales = b + 4;
  const uint8_t* ql = b + 16;
  const uint8_t* qh = b + 144;
  for (int c = 0; c < 4; ++c) {
    uint8_t sc, mn; get_scale_min_k4(2 * c, scales, sc, mn);
    float d1 = dall * sc, m1 = dmin * mn;
    get_scale_min_k4(2 * c + 1, scales, sc, mn);
    float d2 = dall * sc, m2 = dmin * mn;
    for (int l = 0; l < 32; ++l) {
      int hb0 = (qh[l] >> (2 * c)) & 1, hb1 = (qh[l] >> (2 * c + 1)) & 1;
      y[c * 64 + l]      = d1 * (float)(((ql[c * 32 + l]) & 0x0F) + (hb0 ? 16 : 0)) - m1;
      y[c * 64 + 32 + l] = d2 * (float)(((ql[c * 32 + l]) >> 4) + (hb1 ? 16 : 0)) - m2;
    }
  }
}
inline void deq_q6_K_blk(const uint8_t* b, float* y) {
  // GGML-exact (dequantize_row_q6_K): block = ql[128]+qh[64]+scales[16](int8)
  // +d(2) = 210B. Per 128-chunk n: for l<32, is=l/16, four 6-bit quants from
  // ql[l],ql[l+32] low/high nibbles + qh[l] bit-pairs, scales sc0[is..is+6].
  const uint8_t* ql = b;
  const uint8_t* qh = b + 128;
  const int8_t* sc = (const int8_t*)(b + 192);
  uint16_t hd; std::memcpy(&hd, b + 208, 2);
  float d = fp16_to_fp32(hd);
  for (int n = 0; n < 2; ++n) {
    const uint8_t* ql0 = ql + n * 64;
    const uint8_t* qh0 = qh + n * 32;
    const int8_t* sc0 = sc + n * 8;
    for (int l = 0; l < 32; ++l) {
      int is = l / 16;
      int q1 = ((ql0[l] & 0x0F) | (((qh0[l] >> 0) & 3) << 4)) - 32;
      int q2 = ((ql0[l + 32] & 0x0F) | (((qh0[l] >> 2) & 3) << 4)) - 32;
      int q3 = ((ql0[l] >> 4) | (((qh0[l] >> 4) & 3) << 4)) - 32;
      int q4 = ((ql0[l + 32] >> 4) | (((qh0[l] >> 6) & 3) << 4)) - 32;
      y[n * 128 + l]      = d * (float)sc0[is] * (float)q1;
      y[n * 128 + 32 + l] = d * (float)sc0[is + 2] * (float)q2;
      y[n * 128 + 64 + l] = d * (float)sc0[is + 4] * (float)q3;
      y[n * 128 + 96 + l] = d * (float)sc0[is + 6] * (float)q4;
    }
  }
}
inline void deq_q8_K_blk(const uint8_t* b, float* y) {
  // GGML-exact (dequantize_row_q8_K): block = d(f32,4)+qs[256]+bsums[32] = 292B.
  // bsums (int16 partial sums for matvec) are intentionally not read here.
  float d; std::memcpy(&d, b, 4);
  const int8_t* qs = (const int8_t*)(b + 4);
  for (int i = 0; i < 256; ++i) y[i] = d * (float)qs[i];
}
inline void deq_q2_K_blk(const uint8_t* b, float* y) {
  // GGML-exact (dequantize_row_q2_K): block = d(2)+dmin(2)+scales[16]
  // (lo=sc,hi=mn per 16-group) + qs[64] = 84B. qs packing is STRIDED per
  // 128-chunk: byte l of each 32B quarter holds elements {l, l+32, l+64,
  // l+96} of the chunk (2 bits each). 8 scales per 128-chunk.
  uint16_t hd, hmin; std::memcpy(&hd, b, 2); std::memcpy(&hmin, b + 2, 2);
  float dall = fp16_to_fp32(hd), dmin = fp16_to_fp32(hmin);
  const uint8_t* sc = b + 4;
  const uint8_t* qs = b + 20;
  int is = 0;
  for (int n = 0; n < 2; ++n) {
    const uint8_t* qb = qs + n * 32;
    for (int j = 0; j < 4; ++j) {
      uint8_t s0 = sc[is++], s1 = sc[is++];
      float dl0 = dall * (s0 & 0xF), ml0 = dmin * (s0 >> 4);
      float dl1 = dall * (s1 & 0xF), ml1 = dmin * (s1 >> 4);
      for (int l = 0; l < 16; ++l) {
        y[n * 128 + j * 32 + l]      = dl0 * (float)((qb[l] >> (2 * j)) & 3) - ml0;
        y[n * 128 + j * 32 + 16 + l] = dl1 * (float)((qb[16 + l] >> (2 * j)) & 3) - ml1;
      }
    }
  }
}
inline void deq_q3_K_blk(const uint8_t* b, float* y) {
  // GGML-exact (dequantize_row_q3_K): block = d(fp16,2) + qs[64] (2-bit low)
  // + hmask[32] (high bit; bit-plane k of byte e covers element 32*k+e) +
  // scales[12] (custom-packed 16 int8, unpacked via the aux-shuffle below;
  // effective scale = dall*(sc-32)). qs packing is strided per 128-chunk
  // like Q2_K. hmask[32] is shared across both chunks (mask bit selects plane).
  uint16_t hd; std::memcpy(&hd, b, 2);
  float dall = fp16_to_fp32(hd);
  const uint8_t* ql = b + 2;       // 64
  const uint8_t* hm = b + 66;      // 32
  const uint8_t* sc12 = b + 98;    // 12
  uint32_t aux[4];
  std::memcpy(aux, sc12, 12);
  uint32_t tmp = aux[2];
  const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
  aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  const int8_t* sc = (const int8_t*)aux;
  int is = 0;
  uint8_t m = 1;
  for (int n = 0; n < 2; ++n) {
    const uint8_t* qb = ql + n * 32;
    for (int j = 0; j < 4; ++j) {
      float dl0 = dall * (float)(sc[is++] - 32);
      float dl1 = dall * (float)(sc[is++] - 32);
      for (int l = 0; l < 16; ++l) {
        int q0 = (qb[l] >> (2 * j)) & 3, q1 = (qb[16 + l] >> (2 * j)) & 3;
        y[n * 128 + j * 32 + l]      = dl0 * ((float)q0 - ((hm[l] & m) ? 0.0f : 4.0f));
        y[n * 128 + j * 32 + 16 + l] = dl1 * ((float)q1 - ((hm[16 + l] & m) ? 0.0f : 4.0f));
      }
      m <<= 1;
    }
  }
}

}  // namespace

bool dequant_to_f32(uint32_t qtype, const uint8_t* src, size_t src_bytes,
                    size_t n_elems, std::vector<float>& out, std::string& err) {
  out.assign(n_elems, 0.0f);
  if (!src) { err = "null payload"; return false; }
  const uint8_t* p = src;
  float* y = out.data();
  switch (qtype) {
    case 0: {  // F32
      if (src_bytes < n_elems * 4) { err = "F32 truncated"; return false; }
      std::memcpy(y, src, n_elems * 4);
      return true;
    }
    case 1: {  // F16
      if (src_bytes < n_elems * 2) { err = "F16 truncated"; return false; }
      for (size_t i = 0; i < n_elems; ++i) { uint16_t h; std::memcpy(&h, src + i * 2, 2); y[i] = fp16_to_fp32(h); }
      return true;
    }
    case 2: { size_t nb = n_elems / 32; if (src_bytes < nb * 18) { err = "Q4_0 truncated"; return false; }
      for (size_t b = 0; b < nb; ++b) deq_q4_0_blk(p + b * 18, y + b * 32); return true; }
    case 3: { size_t nb = n_elems / 32; if (src_bytes < nb * 20) { err = "Q4_1 truncated"; return false; }
      for (size_t b = 0; b < nb; ++b) deq_q4_1_blk(p + b * 20, y + b * 32); return true; }
    case 6: { size_t nb = n_elems / 32; if (src_bytes < nb * 22) { err = "Q5_0 truncated"; return false; }
      for (size_t b = 0; b < nb; ++b) deq_q5_0_blk(p + b * 22, y + b * 32); return true; }
    case 7: { size_t nb = n_elems / 32; if (src_bytes < nb * 24) { err = "Q5_1 truncated"; return false; }
      for (size_t b = 0; b < nb; ++b) deq_q5_1_blk(p + b * 24, y + b * 32); return true; }
    case 8: { size_t nb = n_elems / 32; if (src_bytes < nb * 34) { err = "Q8_0 truncated"; return false; }
      for (size_t b = 0; b < nb; ++b) deq_q8_0_blk(p + b * 34, y + b * 32); return true; }
    case 9: { size_t nb = n_elems / 32; if (src_bytes < nb * 36) { err = "Q8_1 truncated"; return false; }
      for (size_t b = 0; b < nb; ++b) deq_q8_1_blk(p + b * 36, y + b * 32); return true; }
    case 10: { size_t nb = n_elems / 256; if (src_bytes < nb * 84) { err = "Q2_K truncated"; return false; }
      for (size_t b = 0; b < nb; ++b) deq_q2_K_blk(p + b * 84, y + b * 256); return true; }
    case 11: { size_t nb = n_elems / 256; if (src_bytes < nb * 110) { err = "Q3_K truncated"; return false; }
      for (size_t b = 0; b < nb; ++b) deq_q3_K_blk(p + b * 110, y + b * 256); return true; }
    case 12: { size_t nb = n_elems / 256; if (src_bytes < nb * 144) { err = "Q4_K truncated"; return false; }
      for (size_t b = 0; b < nb; ++b) deq_q4_K_blk(p + b * 144, y + b * 256); return true; }
    case 13: { size_t nb = n_elems / 256; if (src_bytes < nb * 176) { err = "Q5_K truncated"; return false; }
      for (size_t b = 0; b < nb; ++b) deq_q5_K_blk(p + b * 176, y + b * 256); return true; }
    case 14: { size_t nb = n_elems / 256; if (src_bytes < nb * 210) { err = "Q6_K truncated"; return false; }
      for (size_t b = 0; b < nb; ++b) deq_q6_K_blk(p + b * 210, y + b * 256); return true; }
    case 15: { size_t nb = n_elems / 256; if (src_bytes < nb * 292) { err = "Q8_K truncated"; return false; }
      for (size_t b = 0; b < nb; ++b) deq_q8_K_blk(p + b * 292, y + b * 256); return true; }
    default: err = "unsupported quant type (IQ/BF16/unknown): needs conversion"; return false;
  }
}

bool dequant_row_to_f32(uint32_t qtype, const uint8_t* payload, size_t payload_bytes,
                        size_t n_cols, size_t n_rows, size_t row,
                        std::vector<float>& out, std::string& err) {
  out.assign(n_cols, 0.0f);
  if (!payload) { err = "null payload"; return false; }
  if (row >= n_rows) { err = "row out of range"; return false; }
  size_t row_bytes = 0;
  switch (qtype) {
    case 0: row_bytes = n_cols * 4; break;
    case 1: row_bytes = n_cols * 2; break;
    case 2: row_bytes = (n_cols / 32) * 18; break;
    case 3: row_bytes = (n_cols / 32) * 20; break;
    case 6: row_bytes = (n_cols / 32) * 22; break;
    case 7: row_bytes = (n_cols / 32) * 24; break;
    case 8: row_bytes = (n_cols / 32) * 34; break;
    case 9: row_bytes = (n_cols / 32) * 36; break;
    case 10: row_bytes = (n_cols / 256) * 84; break;
    case 11: row_bytes = (n_cols / 256) * 110; break;
    case 12: row_bytes = (n_cols / 256) * 144; break;
    case 13: row_bytes = (n_cols / 256) * 176; break;
    case 14: row_bytes = (n_cols / 256) * 210; break;
    case 15: row_bytes = (n_cols / 256) * 292; break;
    default: err = "unsupported quant for row read"; return false;
  }
  if ((row + 1) * row_bytes > payload_bytes) { err = "row out of payload bounds"; return false; }
  // Dequant only the row slice: no full-tensor temp (bounded to n_cols elems).
  return dequant_to_f32(qtype, payload + row * row_bytes, row_bytes, n_cols, out, err);
}

void quant_f32_to_q8(const float* x, size_t n, int8_t* q8, float& d_x) {
  float amax = 0.0f;
  for (size_t i = 0; i < n; ++i) { float a = std::fabs(x[i]); if (a > amax) amax = a; }
  d_x = amax > 0 ? amax / 127.0f : 1.0f;
  float inv = 1.0f / d_x;
  for (size_t i = 0; i < n; ++i) {
    int q = (int)std::nearbyint(x[i] * inv);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    q8[i] = (int8_t)q;
  }
}

// H1 integer-dot with per-32-block scales: inner loop is pure integer
// (u8 x i8 -> i32) via AVX2 maddubs/madd when available, one FP scale per
// block applied outside. Prod nibbles are sequential even=LOW (element 2k in
// the low nibble of byte k), exactly as q4_0_to_prod emits; same convention
// as q4_q8_dot (Jacob affine, arXiv:1712.05877 Eq.1). Saturation limit identical.
float q4_blockscales_dot(size_t n, const uint8_t* q4, const float* blk_scales,
                         size_t blk_n, const int8_t* q8, float d_x) {
  double acc = 0.0;
  size_t nblocks = (n + 31) / 32;
  if (blk_n < nblocks) nblocks = blk_n;
  for (size_t b = 0; b < nblocks; ++b) {
    size_t base = b * 32;
    size_t blen = (base + 32 <= n) ? 32 : (n - base);
#if defined(__AVX2__)
    // 2 x 16-lane integer dots per 32-block (128-bit lanes; no 256-bit
    // over-read on tails). sum_wx counts raw nibbles, correction -8*sum_x
    // hoisted exactly as in q4_q8_dot (Jacob affine, arXiv:1712.05877 Eq.1).
    const __m128i ones16 = _mm_set1_epi16(1);
    __m128i acc4 = _mm_setzero_si128();
    long long sum_x_blk = 0;
    size_t done = 0;
    for (; done + 16 <= blen; done += 16) {
      uint8_t tmp[16];
      for (int j = 0; j < 16; ++j) {
        size_t gi = base + done + (size_t)j;
        uint8_t byte = q4[gi >> 1];
        tmp[j] = (gi & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
      }
      __m128i w = _mm_loadu_si128((const __m128i*)tmp);
      __m128i x = _mm_loadu_si128((const __m128i*)(q8 + base + done));
      __m128i p = _mm_maddubs_epi16(w, x);
      acc4 = _mm_add_epi32(acc4, _mm_madd_epi16(p, ones16));
      for (int j = 0; j < 16; ++j) sum_x_blk += q8[base + done + (size_t)j];
    }
    __m128i s2 = _mm_add_epi32(acc4, _mm_shuffle_epi32(acc4, 0x4E));
    __m128i s1 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0xB1));
    long long sum_wx_blk = _mm_cvtsi128_si32(s1);
    for (; done < blen; ++done) {
      size_t gi = base + done;
      uint8_t byte = q4[gi >> 1];
      int w = (gi & 1) ? (byte >> 4) : (byte & 0x0F);
      sum_wx_blk += (long long)w * (long long)q8[gi];
      sum_x_blk += q8[gi];
    }
    acc += (double)(sum_wx_blk - 8LL * sum_x_blk) * (double)blk_scales[b];
#else
    double bs = 0.0;
    for (size_t i = 0; i < blen; ++i) {
      size_t gi = base + i;
      uint8_t byte = q4[gi >> 1];
      int w = (gi & 1) ? (byte >> 4) : (byte & 0x0F);
      bs += (double)(w - 8) * (double)q8[gi];
    }
    acc += bs * (double)blk_scales[b];
#endif
  }
  return (float)(acc * (double)d_x);
}

}  // namespace quant
}  // namespace ours
