// kv_q8.cpp — PROD H2: per-block Q8 KV with SEPARATE K/V buffers.
//
// Running min/max per head-dim with exact float history for lossless
// requantize on range widening (bounded 256 rows). Zero is always inside the
// initial range (affine zero-point), then the range only widens.
// KIVI-grade per-channel calibration is future work (documented, not faked).
#include "kv_q8.h"

namespace ours {

KvBlockQ8::KvBlockQ8(int n_head_, int head_dim_)
    : n_head(n_head_), head_dim(head_dim_) {
  kdata.assign((std::size_t)kBlock * n_head * head_dim, 0);
  vdata.assign((std::size_t)kBlock * n_head * head_dim, 0);
  k_hist.assign((std::size_t)kBlock * n_head * head_dim, 0.0f);
  v_hist.assign((std::size_t)kBlock * n_head * head_dim, 0.0f);
  k_mn.assign((std::size_t)n_head * head_dim, 0.0f);
  k_sc.assign((std::size_t)n_head * head_dim, 1.0f);
  v_mn.assign((std::size_t)n_head * head_dim, 0.0f);
  v_sc.assign((std::size_t)n_head * head_dim, 1.0f);
}

static inline int quant_u8(float v, float mn, float sc) {
  int q = (int)((v - mn) / sc + 0.5f);
  if (q < 0) q = 0;
  if (q > 255) q = 255;
  return q;
}

bool KvBlockQ8::append_key(const float* k_row, const float* v_row) {
  if (len >= kBlock) return false;
  if (!k_row || !v_row) return false;
  if (len == 0) {
    for (int h = 0; h < n_head; ++h)
      for (int d = 0; d < head_dim; ++d) {
        std::size_t j = (std::size_t)h * head_dim + (std::size_t)d;
        float kv = k_row[j];
        float vv = v_row[j];
        k_mn[j] = kv < 0 ? kv : 0.0f;
        float kmx = kv > 0 ? kv : 0.0f;
        float kspan = kmx - k_mn[j];
        k_sc[j] = kspan > 0 ? kspan / 255.0f : ((kv < 0 ? -kv : (kv > 0 ? kv : 1.0f)) / 127.0f);
        v_mn[j] = vv < 0 ? vv : 0.0f;
        float vmx = vv > 0 ? vv : 0.0f;
        float vspan = vmx - v_mn[j];
        v_sc[j] = vspan > 0 ? vspan / 255.0f : ((vv < 0 ? -vv : (vv > 0 ? vv : 1.0f)) / 127.0f);
      }
  } else {
    // Running min/max: expand scales to cover the new token, then requantize
    // the block's existing rows when the range grew (bounded: <=256 rows).
    bool k_grew = false, v_grew = false;
    for (int h = 0; h < n_head; ++h)
      for (int d = 0; d < head_dim; ++d) {
        std::size_t j = (std::size_t)h * head_dim + (std::size_t)d;
        float kv = k_row[j], vv = v_row[j];
        float k_lo = k_mn[j], k_hi = k_mn[j] + k_sc[j] * 255.0f;
        float v_lo = v_mn[j], v_hi = v_mn[j] + v_sc[j] * 255.0f;
        if (kv < k_lo || kv > k_hi) {
          float nlo = kv < k_lo ? kv : k_lo, nhi = kv > k_hi ? kv : k_hi;
          k_mn[j] = nlo;
          k_sc[j] = (nhi - nlo) > 0 ? (nhi - nlo) / 255.0f : 1.0f;
          k_grew = true;
        }
        if (vv < v_lo || vv > v_hi) {
          float nlo = vv < v_lo ? vv : v_lo, nhi = vv > v_hi ? vv : v_hi;
          v_mn[j] = nlo;
          v_sc[j] = (nhi - nlo) > 0 ? (nhi - nlo) / 255.0f : 1.0f;
          v_grew = true;
        }
      }
    // Retain exact history first so requantize covers the new row too.
    {
      const std::size_t base = (std::size_t)len * n_head * head_dim;
      for (int h = 0; h < n_head; ++h)
        for (int d = 0; d < head_dim; ++d) {
          std::size_t j = (std::size_t)h * head_dim + (std::size_t)d;
          k_hist[base + j] = k_row[j];
          v_hist[base + j] = v_row[j];
        }
    }
    if (k_grew || v_grew) {
      // Requantize ALL stored rows from exact float history with widened
      // scales (bounded 256 rows; exact, no drift).
      for (int p = 0; p <= len; ++p) {
        const std::size_t base = (std::size_t)p * n_head * head_dim;
        for (int h = 0; h < n_head; ++h)
          for (int d = 0; d < head_dim; ++d) {
            std::size_t j = (std::size_t)h * head_dim + (std::size_t)d;
            kdata[base + j] = (int8_t)(quant_u8(k_hist[base + j], k_mn[j], k_sc[j]) - 128);
            vdata[base + j] = (int8_t)(quant_u8(v_hist[base + j], v_mn[j], v_sc[j]) - 128);
          }
      }
    } else {
      const std::size_t base = (std::size_t)len * n_head * head_dim;
      for (int h = 0; h < n_head; ++h)
        for (int d = 0; d < head_dim; ++d) {
          std::size_t j = (std::size_t)h * head_dim + (std::size_t)d;
          kdata[base + j] = (int8_t)(quant_u8(k_row[j], k_mn[j], k_sc[j]) - 128);
          vdata[base + j] = (int8_t)(quant_u8(v_row[j], v_mn[j], v_sc[j]) - 128);
        }
    }
    ++len;
    return true;
  }
  const std::size_t base = (std::size_t)len * n_head * head_dim;
  for (int h = 0; h < n_head; ++h)
    for (int d = 0; d < head_dim; ++d) {
      std::size_t j = (std::size_t)h * head_dim + (std::size_t)d;
      kdata[base + j] = (int8_t)(quant_u8(k_row[j], k_mn[j], k_sc[j]) - 128);
      vdata[base + j] = (int8_t)(quant_u8(v_row[j], v_mn[j], v_sc[j]) - 128);
      k_hist[base + j] = k_row[j];
      v_hist[base + j] = v_row[j];
    }
  ++len;
  return true;
}

void KvBlockQ8::load_k_row(int h, int p, float* out) const {
  const std::size_t base = (std::size_t)p * n_head * head_dim +
                           (std::size_t)h * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    std::size_t j = (std::size_t)h * head_dim + (std::size_t)d;
    out[d] = ((float)((int)kdata[base + d] + 128)) * k_sc[j] + k_mn[j];
  }
}

void KvBlockQ8::load_v_row(int h, int p, float* out) const {
  const std::size_t base = (std::size_t)p * n_head * head_dim +
                           (std::size_t)h * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    std::size_t j = (std::size_t)h * head_dim + (std::size_t)d;
    out[d] = ((float)((int)vdata[base + d] + 128)) * v_sc[j] + v_mn[j];
  }
}

std::size_t KvBlockQ8::bytes_full() const {
  std::size_t payload =
      (std::size_t)kBlock * n_head * head_dim * 2;  // K+V int8
  std::size_t scales =
      (std::size_t)n_head * head_dim * sizeof(float) * 4;  // k_mn,k_sc,v_mn,v_sc
  return payload + scales;
}

std::size_t KvBlockQ8::bytes_used() const {
  // Resident bytes INCLUDING the exact float history used for lossless
  // requantize (k_hist+v_hist). Production wire/storage bytes exclude history;
  // see bytes_wire_used() for the H2 ~1/2 FP16 claim denominator.
  std::size_t payload = (std::size_t)len * n_head * head_dim * 2;
  std::size_t scales =
      (std::size_t)n_head * head_dim * sizeof(float) * 4;
  std::size_t hist = (std::size_t)len * n_head * head_dim * sizeof(float) * 2;
  return payload + scales + hist;
}

std::size_t KvBlockQ8::bytes_wire_full() const {
  std::size_t payload =
      (std::size_t)kBlock * n_head * head_dim * 2;
  std::size_t scales =
      (std::size_t)n_head * head_dim * sizeof(float) * 4;
  return payload + scales;
}

std::size_t KvBlockQ8::bytes_wire_used() const {
  std::size_t payload = (std::size_t)len * n_head * head_dim * 2;
  std::size_t scales =
      (std::size_t)n_head * head_dim * sizeof(float) * 4;
  return payload + scales;
}

KvCacheQ8::KvCacheQ8(int n_head_, int head_dim_)
    : n_head(n_head_), head_dim(head_dim_) {}

void KvCacheQ8::append(const float* k_row, const float* v_row) {
  if (blocks.empty() || blocks.back().len >= KvBlockQ8::kBlock)
    blocks.emplace_back(n_head, head_dim);
  blocks.back().append_key(k_row, v_row);
}

int KvCacheQ8::seq_len() const {
  int s = 0;
  for (auto& b : blocks) s += b.len;
  return s;
}

std::size_t KvCacheQ8::bytes_total() const {
  std::size_t s = 0;
  for (auto& b : blocks) s += b.bytes_used();
  return s;
}

std::size_t KvCacheQ8::bytes_wire_total() const {
  std::size_t s = 0;
  for (auto& b : blocks) s += b.bytes_wire_used();
  return s;
}

std::size_t KvCacheQ8::bytes_fp16_same_len() const {
  return (std::size_t)seq_len() * n_head * head_dim * 2 /*K+V*/ * 2 /*fp16*/;
}

}  // namespace ours
